"""
Decisive real-data test: does ELHAM's entropy signal beat gradient-free CAM on
ImageNet, where the last conv layer (7x7) is genuinely coarse and a multi-layer
method should have room to help?

Compares, on a pretrained ResNet-50 (clean CAM structure):
  ELHAM-IG        original ELHAM (hierarchical info-gain), class-agnostic
  ELHAM-Hlast     1 - last-layer channel entropy, class-agnostic
  CAM             gradient-free, class-discriminative, single 7x7 layer
  ELHAM-C-refine  CAM * (1 + norm(ELHAM-IG))          single forward pass
  ELHAM-C-Hrefine CAM * (1 + norm(ELHAM-Hlast))       single forward pass
  Score-CAM       gradient-free white-box rival (C forward passes) [--expensive]
  Ablation-CAM    gradient-free white-box rival (C forward passes) [--expensive]
  Grad-CAM        gradient reference

Metrics (per image, explaining the model's top-1 prediction):
  Insertion AUC ↑, Deletion AUC ↓   — model faithfulness, NO ground truth needed
  EPG ↑, PointingGame ↑             — only if ImageNet bbox GT is provided
  cost: ms/image, #forward passes
Reports bootstrap 95% CIs.

Data modes (auto):
  A) --imagenet-root /path  (torchvision ImageNet val layout)  + optional --bbox /path
  B) --folder /path/to/images   (any images; insertion/deletion only)

Run on proxima3 (H200). Examples:
  python eval_imagenet_gt.py --imagenet-root /data/imagenet --bbox /data/imagenet/bbox_val --n 200
  python eval_imagenet_gt.py --folder ./samples --n 100 --no-expensive
"""

import os
import glob
import json
import time
import argparse
import xml.etree.ElementTree as ET
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

HAS_CAPTUM = False
try:
    from captum.attr import LayerGradCam
    HAS_CAPTUM = True
except ImportError:
    pass


# ───────────────────────────── data ─────────────────────────────

_TF = transforms.Compose([
    transforms.Resize((IMG, IMG)),                 # direct resize -> trivial bbox mapping
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


def load_bbox_mask(xml_path, orig_w, orig_h):
    """Union of object bboxes -> binary [IMG,IMG] mask (coords scaled to 224)."""
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None
    m = np.zeros((IMG, IMG), dtype=np.float32)
    sx, sy = IMG / orig_w, IMG / orig_h
    found = False
    for obj in root.findall('object'):
        bb = obj.find('bndbox')
        if bb is None:
            continue
        x0 = int(float(bb.find('xmin').text) * sx); y0 = int(float(bb.find('ymin').text) * sy)
        x1 = int(float(bb.find('xmax').text) * sx); y1 = int(float(bb.find('ymax').text) * sy)
        x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(IMG, x1), min(IMG, y1)
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = 1.0; found = True
    return m if found else None


def _resize_mask(mask_np):
    t = torch.from_numpy(mask_np.astype(np.float32))[None, None]
    return F.interpolate(t, size=(IMG, IMG), mode='nearest')[0, 0].numpy()


def build_samples(args):
    """Returns list of (PIL RGB image, gt_mask[IMG,IMG] or None), length <= args.n."""
    samples = []

    if args.imagenet_root:
        paths = sorted(glob.glob(os.path.join(args.imagenet_root, '**', '*.JPEG'), recursive=True)) \
            or sorted(glob.glob(os.path.join(args.imagenet_root, '**', '*.jpeg'), recursive=True))
        for p in paths[:args.n]:
            pil = Image.open(p).convert('RGB'); ow, oh = pil.size
            gt = None
            if args.bbox:
                cand = os.path.join(args.bbox, os.path.splitext(os.path.basename(p))[0] + '.xml')
                gt = load_bbox_mask(cand, ow, oh) if os.path.exists(cand) else None
            samples.append((pil, gt))

    elif args.folder:
        paths = []
        for e in ('*.JPEG', '*.jpeg', '*.jpg', '*.png'):
            paths += glob.glob(os.path.join(args.folder, '**', e), recursive=True)
        for p in sorted(paths)[:args.n]:
            samples.append((Image.open(p).convert('RGB'), None))

    elif args.dataset == 'pets':
        # Oxford-IIIT Pet: segmentation GT. Pets (dogs/cats) -> ImageNet classes.
        root = args.cache
        try:
            ds = tv.datasets.OxfordIIITPet(root, split='test', target_types='segmentation',
                                           download=True)
        except RuntimeError:
            ds = tv.datasets.OxfordIIITPet(root, split='test', target_types='segmentation',
                                           download=False)
        for i in range(min(args.n, len(ds))):
            pil, seg = ds[i]
            trimap = np.array(seg)             # 1=fg pet, 2=bg, 3=boundary
            gt = _resize_mask((trimap != 2).astype(np.float32))
            samples.append((pil.convert('RGB'), gt))

    else:  # default: imagenette (real full-size ImageNet-class images, no GT)
        root = args.cache
        try:
            ds = tv.datasets.Imagenette(root, split='val', size='full', download=True)
        except RuntimeError:
            ds = tv.datasets.Imagenette(root, split='val', size='full', download=False)
        except AttributeError:
            raise SystemExit('torchvision too old for Imagenette (need >=0.18). '
                             'Use --dataset pets, or --folder PATH, or --imagenet-root PATH.')
        for i in range(min(args.n, len(ds))):
            pil, _ = ds[i]
            samples.append((pil.convert('RGB'), None))

    return samples


# ───────────────────────── forward-only signals ─────────────────────────

class LayerExtractor:
    def __init__(self, model, names):
        self.f = OrderedDict(); self._h = []
        for n, m in model.named_modules():
            if n in names:
                self._h.append(m.register_forward_hook(self._hook(n)))

    def _hook(self, name):
        def fn(m, i, o):
            self.f[name] = o.detach()
        return fn

    def clear(self):
        self.f.clear()

    def remove(self):
        for h in self._h:
            h.remove()


def _n01(t):
    t = t - t.min(); mx = t.max()
    return t / mx if mx > 0 else t


def _chan_ent(feats):
    p = F.softmax(feats, dim=1); C = feats.shape[1]
    return (-(p * torch.log(p + 1e-8)).sum(1) / max(np.log(C), 0.01)).squeeze(0)


def _up(t, size):
    return F.interpolate(t[None, None], size=size, mode='bilinear', align_corners=False)[0, 0]


class ForwardSignals:
    """One forward pass -> ELHAM-IG, ELHAM-Hlast, CAM(class). Gradient-free."""

    def __init__(self, model, layers, final_layer, fc_weight):
        self.model = model; self.layers = layers
        self.final = final_layer; self.W = fc_weight  # [n_cls, C]
        self.ext = LayerExtractor(model, layers)

    def compute(self, x):
        self.ext.clear()
        with torch.no_grad():
            logits = self.model(x)
        feats = self.ext.f
        ents = OrderedDict((n, _chan_ent(feats[n])) for n in self.layers)
        ig = torch.zeros(IMG, IMG, device=x.device); prev = None
        for n in self.layers:
            if prev is not None:
                Hp, Hc = ents[prev], ents[n]
                if Hp.shape != Hc.shape:
                    Hp = _up(Hp, Hc.shape)
                ig += _up(torch.clamp(Hp - Hc, min=0), (IMG, IMG))
            prev = n
        hlast = _up(1.0 - ents[self.final], (IMG, IMG))
        return logits, feats, ig, hlast

    def cam(self, feats, target, size=(IMG, IMG)):
        A = feats[self.final][0]
        w = self.W[target].to(A.device)
        return _up(F.relu((w.view(-1, 1, 1) * A).sum(0)), size)

    def maps(self, x, target):
        _, feats, ig, hlast = self.compute(x)
        cam = self.cam(feats, target)
        e_ig, e_hl, c = _n01(ig), _n01(hlast), _n01(cam)
        out = {
            'ELHAM-IG': e_ig, 'ELHAM-Hlast': e_hl, 'CAM': c,
            'ELHAM-C-refine': _n01(c * (1 + e_ig)),
            'ELHAM-C-Hrefine': _n01(c * (1 + e_hl)),
        }
        return {k: v.detach().cpu().numpy() for k, v in out.items()}

    def remove(self):
        self.ext.remove()


# ─────────────────── expensive gradient-free rivals ───────────────────

class ScoreCAM:
    def __init__(self, model, final_layer, batch=64, max_channels=None):
        self.model = model; self.final = final_layer
        self.batch = batch; self.max_channels = max_channels
        self.ext = LayerExtractor(model, [final_layer]); self.fwd = 0

    def explain(self, x, target):
        self.ext.clear()
        with torch.no_grad():
            _ = self.model(x)
        A = self.ext.f[self.final][0]                # [C,h,w]
        C = A.shape[0] if not self.max_channels else min(A.shape[0], self.max_channels)
        masks = F.interpolate(A[:C].unsqueeze(1), size=(IMG, IMG),
                              mode='bilinear', align_corners=False)  # [C,1,224,224]
        mn = masks.amin(dim=(2, 3), keepdim=True); mx = masks.amax(dim=(2, 3), keepdim=True)
        masks = (masks - mn) / (mx - mn + 1e-8)
        scores = torch.zeros(C, device=x.device)
        with torch.no_grad():
            for s in range(0, C, self.batch):
                mb = masks[s:s + self.batch]                  # [b,1,224,224]
                xb = x * mb                                   # broadcast [b,3,224,224]
                out = F.softmax(self.model(xb), dim=1)[:, target]
                scores[s:s + mb.shape[0]] = out; self.fwd += mb.shape[0]
        cam = F.relu((scores.view(-1, 1, 1) * A[:C]).sum(0))
        return _n01(_up(cam, (IMG, IMG))).detach().cpu().numpy()

    def remove(self):
        self.ext.remove()


class AblationCAM:
    """Ablation-CAM: weight = drop in class score when channel k is zeroed."""
    def __init__(self, model, final_layer, batch=64, max_channels=None):
        self.model = model; self.final = final_layer
        self.batch = batch; self.max_channels = max_channels
        self.act = {}; self.fwd = 0
        self.mod = dict(model.named_modules())[final_layer]

    def explain(self, x, target):
        captured = {}
        h = self.mod.register_forward_hook(lambda m, i, o: captured.__setitem__('a', o.detach()))
        with torch.no_grad():
            base = F.softmax(self.model(x), dim=1)[0, target].item(); self.fwd += 1
        A = captured['a'][0]                          # [C,h,w]
        C = A.shape[0] if not self.max_channels else min(A.shape[0], self.max_channels)
        h.remove()
        weights = torch.zeros(C, device=x.device)
        # ablate by forcing channel k to zero at the final layer output
        def make_hook(kset):
            def fn(m, i, o):
                o = o.clone(); o[:, kset] = 0; return o
            return fn
        with torch.no_grad():
            for s in range(0, C, self.batch):
                ks = list(range(s, min(C, s + self.batch)))
                # batch one image per ablated channel
                xb = x.repeat(len(ks), 1, 1, 1)
                hh = self.mod.register_forward_hook(self._batch_ablate(ks))
                out = F.softmax(self.model(xb), dim=1)[:, target]
                hh.remove(); self.fwd += len(ks)
                weights[s:s + len(ks)] = (base - out) / (base + 1e-8)
        cam = F.relu((weights.view(-1, 1, 1) * A[:C]).sum(0))
        return _n01(_up(cam, (IMG, IMG))).detach().cpu().numpy()

    def _batch_ablate(self, ks):
        def fn(m, i, o):
            o = o.clone()
            for bi, k in enumerate(ks):
                o[bi, k] = 0
            return o
        return fn

    def remove(self):
        pass


# ───────────────────────────── metrics ─────────────────────────────

def _blur(x, k=51, s=21):
    C = x.shape[1]
    ax = torch.arange(k, device=x.device) - k // 2
    g = torch.exp(-ax ** 2 / (2 * s ** 2)); g = g / g.sum()
    kh = g.view(1, 1, 1, -1).repeat(C, 1, 1, 1); kv = g.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    pad = k // 2
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode='reflect'), kh, groups=C)
    x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode='reflect'), kv, groups=C)
    return x


def insertion_auc(model, x, target, attr, steps=50):
    blur = _blur(x); order = np.argsort(attr.flatten())[::-1].copy()
    n = IMG * IMG; ps = max(1, n // steps); sc = []
    with torch.no_grad():
        for s in range(steps + 1):
            mask = torch.zeros(n, device=x.device); ni = s * ps
            if ni > 0:
                mask[torch.from_numpy(order[:ni]).to(x.device)] = 1
            mk = mask.view(1, 1, IMG, IMG)
            out = model(mk * x + (1 - mk) * blur)
            sc.append(F.softmax(out, 1)[0, target].item())
    return float(np.trapz(sc) / steps)


def deletion_auc(model, x, target, attr, steps=50):
    order = np.argsort(attr.flatten())[::-1].copy()
    n = IMG * IMG; ps = max(1, n // steps); sc = []
    with torch.no_grad():
        for s in range(steps + 1):
            mask = torch.ones(n, device=x.device); nr = s * ps
            if nr > 0:
                mask[torch.from_numpy(order[:nr]).to(x.device)] = 0
            out = model(x * mask.view(1, 1, IMG, IMG))
            sc.append(F.softmax(out, 1)[0, target].item())
    return float(np.trapz(sc) / steps)


def epg(attr, gt):
    a = np.clip(attr, 0, None)
    if a.sum() == 0:
        return 0.0
    return float((a * gt).sum() / a.sum())


def pointing(attr, gt):
    my, mx = np.unravel_index(attr.argmax(), attr.shape)
    return float(gt[my, mx] > 0.5)


def boot_ci(v, n=2000, seed=0):
    v = np.asarray(v, float)
    if len(v) == 0:
        return (float('nan'),) * 3
    rng = np.random.default_rng(seed)
    bs = rng.choice(v, (n, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ───────────────────────────── main ─────────────────────────────

def build_model(name):
    if name == 'resnet50':
        m = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2)
        layers = ['layer1', 'layer2', 'layer3', 'layer4']
        final = 'layer4'; fc = m.fc.weight.detach()
        gc_layer = m.layer4
    else:
        raise ValueError(f'unsupported model {name} (clean CAM only for resnet50 here)')
    return m.to(DEVICE).eval(), layers, final, fc.to(DEVICE), gc_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='resnet50')
    ap.add_argument('--dataset', default='imagenette', choices=['imagenette', 'pets'],
                    help='auto-download dataset when no --imagenet-root/--folder given')
    ap.add_argument('--cache', default='./data_cache', help='download dir for auto datasets')
    ap.add_argument('--imagenet-root', default=None)
    ap.add_argument('--bbox', default=None)
    ap.add_argument('--folder', default=None)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--no-expensive', action='store_true', help='skip Score-CAM/Ablation-CAM')
    ap.add_argument('--max-channels', type=int, default=None, help='cap channels for expensive methods')
    ap.add_argument('--out', default='elham_imagenet_gt')
    args = ap.parse_args()

    print(f'Device: {DEVICE} | model: {args.model} | captum: {HAS_CAPTUM}')
    model, layers, final, fc, gc_layer = build_model(args.model)
    sig = ForwardSignals(model, layers, final, fc)
    gradcam = LayerGradCam(model, gc_layer) if HAS_CAPTUM else None
    scam = None if args.no_expensive else ScoreCAM(model, final, max_channels=args.max_channels)
    acam = None if args.no_expensive else AblationCAM(model, final, max_channels=args.max_channels)

    samples = build_samples(args)
    if not samples:
        print('ERROR: no images found. Pass --imagenet-root, --folder, or use --dataset.')
        return
    src = args.imagenet_root or args.folder or f'auto:{args.dataset}'
    print(f'Found {len(samples)} images from [{src}]. '
          f'GT masks: {"yes" if any(s[1] is not None for s in samples) else "no"}\n')

    fwd_methods = ['ELHAM-IG', 'ELHAM-Hlast', 'CAM', 'ELHAM-C-refine', 'ELHAM-C-Hrefine']
    methods = list(fwd_methods)
    if not args.no_expensive:
        methods += ['Score-CAM', 'Ablation-CAM']
    if HAS_CAPTUM:
        methods += ['Grad-CAM']

    R = {m: {'ins': [], 'del': [], 'epg': [], 'point': []} for m in methods}
    timing = {m: 0.0 for m in methods}
    n_used = 0

    for idx, (pil, gt) in enumerate(samples):
        try:
            x = _TF(pil).unsqueeze(0).to(DEVICE)
        except Exception as e:
            print(f'  skip image {idx}: {e}'); continue
        with torch.no_grad():
            target = int(model(x).argmax(1).item())

        t0 = time.perf_counter()
        fmaps = sig.maps(x, target)
        dt = (time.perf_counter() - t0) / len(fwd_methods)
        for m in fwd_methods:
            timing[m] += dt
        amaps = dict(fmaps)

        if not args.no_expensive:
            t0 = time.perf_counter(); amaps['Score-CAM'] = scam.explain(x, target)
            timing['Score-CAM'] += time.perf_counter() - t0
            t0 = time.perf_counter(); amaps['Ablation-CAM'] = acam.explain(x, target)
            timing['Ablation-CAM'] += time.perf_counter() - t0
        if HAS_CAPTUM:
            t0 = time.perf_counter()
            g = gradcam.attribute(x, target=target, relu_attributions=True)
            timing['Grad-CAM'] += time.perf_counter() - t0
            amaps['Grad-CAM'] = F.interpolate(g, size=(IMG, IMG), mode='bilinear',
                                              align_corners=False).squeeze().detach().cpu().numpy()

        for m in methods:
            a = amaps[m]
            R[m]['ins'].append(insertion_auc(model, x, target, a, args.steps))
            R[m]['del'].append(deletion_auc(model, x, target, a, args.steps))
            if gt is not None:
                R[m]['epg'].append(epg(a, gt))
                R[m]['point'].append(pointing(a, gt))
        n_used += 1
        if (idx + 1) % 25 == 0:
            print(f'  {idx + 1}/{len(samples)} done')

    # report
    print(f'\n{"=" * 78}\n  RESULTS  (n={n_used}, model={args.model})\n{"=" * 78}')
    has_gt = len(R[methods[0]]['epg']) > 0
    hdr = f'  {"Method":<17s}{"Ins↑ (95% CI)":>22s}{"Del↓ (95% CI)":>22s}'
    if has_gt:
        hdr += f'{"EPG↑":>9s}{"Point↑":>9s}'
    print(hdr); print('  ' + '-' * (len(hdr)))
    summary = {}
    for m in methods:
        im, il, ih = boot_ci(R[m]['ins']); dm, dl, dh = boot_ci(R[m]['del'])
        row = f'  {m:<17s}{f"{im:.3f} [{il:.3f},{ih:.3f}]":>22s}{f"{dm:.3f} [{dl:.3f},{dh:.3f}]":>22s}'
        if has_gt:
            row += f'{np.mean(R[m]["epg"]):>9.3f}{np.mean(R[m]["point"]):>9.3f}'
        print(row)
        summary[m] = {'ins': boot_ci(R[m]['ins']), 'del': boot_ci(R[m]['del']),
                      'epg': float(np.mean(R[m]['epg'])) if has_gt else None,
                      'point': float(np.mean(R[m]['point'])) if has_gt else None}

    print(f'\n  {"Method":<17s}{"ms/img":>10s}{"fwd passes":>12s}')
    print('  ' + '-' * 39)
    fwd = {'Score-CAM': getattr(scam, 'fwd', 0), 'Ablation-CAM': getattr(acam, 'fwd', 0)}
    for m in methods:
        fp = fwd.get(m, n_used)  # forward-only methods = 1 per image
        per = fp / max(n_used, 1)
        print(f'  {m:<17s}{1000 * timing[m] / max(n_used, 1):>10.2f}{per:>12.1f}')

    # verdict
    print(f'\n{"=" * 78}\n  VERDICT: does any entropy variant beat gradient-free CAM?\n{"=" * 78}')
    cam_ins = np.mean(R['CAM']['ins'])
    for m in ['ELHAM-IG', 'ELHAM-Hlast', 'ELHAM-C-refine', 'ELHAM-C-Hrefine']:
        d = np.mean(R[m]['ins']) - cam_ins
        print(f'    {m:<17s} Ins Δ vs CAM = {d:+.3f}  ({"better" if d > 0.005 else "no gain"})')
    if not args.no_expensive:
        print(f'    Score-CAM Ins Δ vs CAM = {np.mean(R["Score-CAM"]["ins"]) - cam_ins:+.3f} '
              f'(cost {fwd["Score-CAM"] / max(n_used,1):.0f} fwd/img)')

    with open(args.out + '.json', 'w') as f:
        json.dump({'n': n_used, 'model': args.model, 'has_gt': has_gt,
                   'summary': summary, 'raw': R}, f, indent=2)
    _plot(R, methods, has_gt, args.out)
    print(f'\n  Saved: {args.out}.json, {args.out}.png')


def _plot(R, methods, has_gt, out):
    cmap = plt.get_cmap('tab10')
    panels = [('ins', 'Insertion AUC ↑'), ('del', 'Deletion AUC ↓')]
    if has_gt:
        panels += [('epg', 'Energy Pointing Game ↑'), ('point', 'Pointing Game ↑')]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5))
    if len(panels) == 1:
        axes = [axes]
    for ax, (metric, title) in zip(axes, panels):
        means = [np.mean(R[m][metric]) for m in methods]
        errs = [np.std(R[m][metric]) / np.sqrt(max(1, len(R[m][metric]))) for m in methods]
        ax.bar(range(len(methods)), means, yerr=errs, capsize=3,
               color=[cmap(i % 10) for i in range(len(methods))], edgecolor='white')
        ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, rotation=40, ha='right', fontsize=8)
        ax.set_title(title, fontweight='bold')
    plt.suptitle('ELHAM vs gradient-free CAM family on ImageNet (ResNet-50)',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(out + '.png', dpi=170, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
