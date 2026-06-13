"""
Class-Conditional ELHAM (ELHAM-C) — diagnostic proof-of-concept
===============================================================
Two questions decide whether ELHAM can become a real attribution method:

  Q1. Can we make ELHAM class-discriminative *without gradients*?  (the fatal flaw)
  Q2. Does ELHAM's entropy / information-gain signal add anything that
      gradient-free CAM / Score-CAM don't already provide?

This script answers both on a *non-trivial* synthetic task with KNOWN causal
ground-truth regions (64x64 so the last conv layer is genuinely coarse — this is
where a multi-layer method should help over single-layer CAM).

Methods compared
----------------
  ELHAM-IG       original ELHAM: sum of channel-entropy reductions (class-agnostic)
  ELHAM-Hlast    decisiveness: 1 - last-layer channel entropy (class-agnostic)
  CAM            gradient-free, class-discriminative, single (last) layer
  Score-CAM      gradient-free white-box rival, C forward passes
  Grad-CAM       gradient reference (Captum)
  ELHAM-C·       class-conditional fusions of an ELHAM signal with the CAM prior:
                   -mult    : norm(E) * norm(CAM)^gamma   (lets E veto CAM — risky)
                   -refine  : norm(CAM) * (1 + alpha*norm(E))  (E can only sharpen)

Self-contained: own data generator (the imported ground-truth generators had an
out-of-bounds bug and an n_classes/nc mismatch). CPU, deterministic.

Usage: python eval_classcond.py
"""

import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cpu')
SEED = 0
IMG = 64

HAS_CAPTUM = False
try:
    from captum.attr import LayerGradCam
    HAS_CAPTUM = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic data: class determined by a SMALL oriented-grating patch.
# Distractor patches (random gratings) look similar but don't determine class,
# forcing the model to localise the *causal* patch. Object is small relative to
# image -> last-conv CAM (8x8) is coarse -> room for a finer method to win.
# ═══════════════════════════════════════════════════════════════════════════

def _grating(h, w, theta, freq, tint):
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    g = np.sin(freq * (xx * np.cos(theta) + yy * np.sin(theta)))   # [-1,1]
    patch = np.stack([g * tint[c] for c in range(3)], 0).astype(np.float32)
    return torch.from_numpy(patch)


class HardCausalDataset:
    def __init__(self, n_samples, n_classes=4, img=IMG):
        self.n = n_samples; self.k = n_classes; self.img = img

    def _sample(self, competing=False, interaction=False):
        img = torch.randn(3, self.img, self.img) * 0.06
        # low-freq colored background structure
        for c in range(3):
            fx, fy = np.random.uniform(0.1, 0.4, 2)
            yy, xx = np.meshgrid(np.arange(self.img), np.arange(self.img), indexing='ij')
            img[c] += torch.from_numpy(
                (0.12 * np.sin(fx * xx) * np.cos(fy * yy)).astype(np.float32))

        def place(theta, freq, tint, sz=None):
            h = w = sz or np.random.randint(11, 16)
            y = np.random.randint(2, self.img - h - 2)
            x = np.random.randint(2, self.img - w - 2)
            img[:, y:y + h, x:x + w] = _grating(h, w, theta, freq, tint) + \
                torch.randn(3, h, w) * 0.08
            m = torch.zeros(1, self.img, self.img); m[0, y:y + h, x:x + w] = 1.0
            return m

        cls = np.random.randint(0, self.k)
        # causal patch: orientation encodes class; subtle class color tint
        theta_c = cls * np.pi / self.k
        tint_c = [0.8, 0.8, 0.8]
        tint_c[cls % 3] = 1.0
        mask = place(theta_c, freq=1.1, tint=tint_c)

        if competing:  # one distractor grating, random class-like but non-causal
            place(np.random.uniform(0, np.pi), freq=1.1,
                  tint=list(np.random.uniform(0.6, 1.0, 3)))
        if interaction:  # a second causal patch needed (mask covers both)
            theta2 = ((cls + 1) % self.k) * np.pi / self.k
            mask = mask + place(theta2, freq=0.9, tint=[1.0, 0.7, 0.7])
            mask = mask.clamp(0, 1)
        # a couple of generic distractors
        for _ in range(2):
            place(np.random.uniform(0, np.pi), freq=np.random.uniform(0.7, 1.4),
                  tint=list(np.random.uniform(0.5, 1.0, 3)))
        return img, cls, mask

    def make(self, competing=False, interaction=False):
        X, y, M = [], [], []
        for _ in range(self.n):
            im, c, m = self._sample(competing, interaction)
            X.append(im); y.append(c); M.append(m)
        return torch.stack(X), torch.tensor(y), torch.stack(M)


class SyntheticCNN(nn.Module):
    def __init__(self, nc=4):
        super().__init__()
        def blk(i, o, pool=True):
            layers = [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                      nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)
        self.layer1 = blk(3, 32)      # 64 -> 32
        self.layer2 = blk(32, 64)     # 32 -> 16
        self.layer3 = blk(64, 128)    # 16 -> 8
        self.layer4 = blk(128, 256, pool=False)  # 8
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, nc)

    def forward(self, x):
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.fc(self.pool(x).view(x.size(0), -1))


# ═══════════════════════════════════════════════════════════════════════════
# Forward-only activation extractor + helpers
# ═══════════════════════════════════════════════════════════════════════════

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


def _norm01(t):
    t = t - t.min(); m = t.max()
    return t / m if m > 0 else t


def _channel_entropy(feats):
    p = F.softmax(feats, dim=1); C = feats.shape[1]
    return (-(p * torch.log(p + 1e-8)).sum(1) / max(np.log(C), 0.01)).squeeze(0)


def _up(t, size):
    return F.interpolate(t[None, None], size=size, mode='bilinear', align_corners=False)[0, 0]


# ═══════════════════════════════════════════════════════════════════════════
# Unified forward-only explainer producing several signals from ONE pass
# ═══════════════════════════════════════════════════════════════════════════

class ForwardSignals:
    """One forward pass -> {infogain, hlast, cam(target)}. All gradient-free."""

    def __init__(self, model, layers, final_layer, fc):
        self.model = model; self.layers = layers
        self.final_layer = final_layer; self.fc = fc
        self.ext = LayerExtractor(model, layers)

    def compute(self, image):
        self.ext.clear()
        with torch.no_grad():
            _ = self.model(image)
        feats = self.ext.f
        Hin, Win = image.shape[2], image.shape[3]
        ents = OrderedDict((n, _channel_entropy(feats[n])) for n in self.layers)
        # info-gain (entropy reduction across consecutive layers)
        ig = torch.zeros(Hin, Win)
        prev = None
        for n in self.layers:
            if prev is not None:
                Hp, Hc = ents[prev], ents[n]
                if Hp.shape != Hc.shape:
                    Hp = _up(Hp, Hc.shape)
                ig += _up(torch.clamp(Hp - Hc, min=0), (Hin, Win))
            prev = n
        # decisiveness = 1 - last-layer entropy
        hlast = _up(1.0 - ents[self.final_layer], (Hin, Win))
        return {'feats': feats, 'ig': ig, 'hlast': hlast, 'size': (Hin, Win)}

    def cam(self, feats, target_class, size):
        A = feats[self.final_layer][0]
        w = self.fc.weight[target_class].to(A.device)
        return _up(F.relu((w.view(-1, 1, 1) * A).sum(0)), size)

    def remove(self):
        self.ext.remove()


def build_maps(sig, image, target_class):
    """Returns dict name->attribution(np 2D) for all forward-only methods."""
    s = sig.compute(image)
    cam = sig.cam(s['feats'], target_class, s['size'])
    e_ig, e_hl, c = _norm01(s['ig']), _norm01(s['hlast']), _norm01(cam)
    out = {
        'ELHAM-IG':        e_ig,
        'ELHAM-Hlast':     e_hl,
        'CAM':             c,
        'ELHAM-C-mult':    e_ig * (c ** 1.0),
        'ELHAM-C-refine':  c * (1.0 + 1.0 * e_ig),
        'ELHAM-C-Hrefine': c * (1.0 + 1.0 * e_hl),
    }
    return {k: v.detach().cpu().numpy() for k, v in out.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Score-CAM (expensive gradient-free rival)
# ═══════════════════════════════════════════════════════════════════════════

class ScoreCAM:
    def __init__(self, model, final_layer, max_channels=128):
        self.model = model; self.final_layer = final_layer
        self.max_channels = max_channels
        self.ext = LayerExtractor(model, [final_layer]); self.fwd = 0

    def explain(self, image, target_class):
        self.ext.clear()
        with torch.no_grad():
            _ = self.model(image)
        A = self.ext.f[self.final_layer][0]
        size = (image.shape[2], image.shape[3])
        C = min(A.shape[0], self.max_channels)
        w = []
        with torch.no_grad():
            for k in range(C):
                m = _up(_norm01(A[k]), size)[None, None]
                score = F.softmax(self.model(image * m), dim=1)[0, target_class].item()
                self.fwd += 1; w.append(score)
        w = torch.tensor(w)
        cam = F.relu((w.view(-1, 1, 1) * A[:C]).sum(0))
        return _norm01(_up(cam, size)).detach().cpu().numpy()

    def remove(self):
        self.ext.remove()


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def ap(a, gt):
    fg = gt.flatten()
    if fg.sum() == 0 or a.std() == 0:
        return 0.5
    return float(average_precision_score(fg.astype(int), a.flatten()))


def iou(a, gt, frac=0.25):
    k = max(1, int(a.size * frac))
    thr = np.sort(a.flatten())[-k]
    pred = (a >= thr).astype(float); g = gt.astype(float)
    u = (pred + g).clip(0, 1).sum()
    return float((pred * g).sum() / u) if u > 0 else 0.0


def point(a, gt):
    my, mx = np.unravel_index(a.argmax(), a.shape)
    return float(gt[my, mx] > 0.5)


def corr(a, b):
    if a.std() == 0 or b.std() == 0:
        return 1.0 if a.std() == b.std() else 0.0
    r, _ = spearmanr(a.flatten(), b.flatten())
    return 0.0 if np.isnan(r) else float(r)


# ═══════════════════════════════════════════════════════════════════════════
# Train
# ═══════════════════════════════════════════════════════════════════════════

def train(n_classes=4, n_train=2400, n_val=500, epochs=30):
    torch.manual_seed(SEED); np.random.seed(SEED)
    ds = HardCausalDataset(n_train + n_val, n_classes)
    X, y, _ = ds.make()
    Xtr, ytr = X[:n_train].to(DEVICE), y[:n_train].to(DEVICE)
    Xva, yva = X[n_train:].to(DEVICE), y[n_train:].to(DEVICE)
    model = SyntheticCNN(n_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(epochs):
        model.train(); idx = torch.randperm(n_train)
        for b in range(0, n_train, 64):
            bi = idx[b:b + 64]; opt.zero_grad()
            crit(model(Xtr[bi]), ytr[bi]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xva).argmax(1) == yva).float().mean().item()
        best = max(best, acc)
        if ep % 6 == 0 or ep == epochs - 1:
            print(f'  ep {ep + 1:2d}/{epochs}  val_acc={acc:.3f}')
    print(f'  best val_acc={best:.3f}')
    model.eval()
    return model, n_classes


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run(n_test=50):
    print(f'Device: {DEVICE} | Captum: {"yes" if HAS_CAPTUM else "no"} | img={IMG}\n')
    print('Training (non-trivial synthetic causal task)...')
    model, K = train()

    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    fc = dict(model.named_modules())['fc']
    sig = ForwardSignals(model, layers, 'layer4', fc)
    scorecam = ScoreCAM(model, 'layer4', max_channels=128)
    gradcam = LayerGradCam(model, model.layer4) if HAS_CAPTUM else None

    fwd_method_names = ['ELHAM-IG', 'ELHAM-Hlast', 'CAM', 'ELHAM-C-mult',
                        'ELHAM-C-refine', 'ELHAM-C-Hrefine']
    method_names = fwd_method_names + ['Score-CAM'] + (['Grad-CAM'] if HAS_CAPTUM else [])

    tests = [('T1 single', dict()),
             ('T2 competing', dict(competing=True)),
             ('T3 interaction', dict(interaction=True))]

    all_res = OrderedDict()
    timing = {m: 0.0 for m in method_names}

    for tname, kw in tests:
        print(f'\n{"=" * 74}\n  {tname}\n{"=" * 74}')
        torch.manual_seed(100); np.random.seed(100)  # fixed test set per config
        ds = HardCausalDataset(n_test, K)
        tx, ty, tm = ds.make(**kw)
        tx, ty, tm = tx.to(DEVICE), ty.to(DEVICE), tm.numpy()

        res = {m: {'ap': [], 'iou': [], 'point': [], 'disc': [], 'adv': []}
               for m in method_names}

        for i in range(n_test):
            img, tc, gt = tx[i:i + 1], int(ty[i].item()), tm[i, 0]
            wrong = (tc + 1) % K

            t0 = time.perf_counter()
            maps_t = build_maps(sig, img, tc)
            maps_w = build_maps(sig, img, wrong)
            dt = (time.perf_counter() - t0) / len(fwd_method_names)
            for m in fwd_method_names:
                timing[m] += dt

            t0 = time.perf_counter()
            sc_t = scorecam.explain(img, tc); sc_w = scorecam.explain(img, wrong)
            timing['Score-CAM'] += time.perf_counter() - t0
            maps_t['Score-CAM'] = sc_t; maps_w['Score-CAM'] = sc_w

            if HAS_CAPTUM:
                t0 = time.perf_counter()
                g = gradcam.attribute(img, target=tc, relu_attributions=True)
                gw = gradcam.attribute(img, target=wrong, relu_attributions=True)
                timing['Grad-CAM'] += time.perf_counter() - t0
                maps_t['Grad-CAM'] = _up(g.squeeze(), (IMG, IMG)).detach().cpu().numpy() \
                    if g.dim() == 4 else g.squeeze().detach().cpu().numpy()
                maps_t['Grad-CAM'] = F.interpolate(g, size=(IMG, IMG), mode='bilinear',
                                                   align_corners=False).squeeze().detach().cpu().numpy()
                maps_w['Grad-CAM'] = F.interpolate(gw, size=(IMG, IMG), mode='bilinear',
                                                   align_corners=False).squeeze().detach().cpu().numpy()

            for m in method_names:
                a, aw = maps_t[m], maps_w[m]
                res[m]['ap'].append(ap(a, gt))
                res[m]['iou'].append(iou(a, gt))
                res[m]['point'].append(point(a, gt))
                res[m]['disc'].append(corr(a, aw))
                res[m]['adv'].append(ap(a, gt) - ap(aw, gt))

        print(f'\n  {"Method":<17s}{"AP↑":>7s}{"IoU↑":>7s}{"Pt↑":>7s}'
              f'{"disc↓":>8s}{"adv↑":>8s}')
        print('  ' + '-' * 54)
        for m in method_names:
            print(f'  {m:<17s}{np.mean(res[m]["ap"]):>7.3f}{np.mean(res[m]["iou"]):>7.3f}'
                  f'{np.mean(res[m]["point"]):>7.3f}{np.mean(res[m]["disc"]):>8.3f}'
                  f'{np.mean(res[m]["adv"]):>8.3f}')
        all_res[tname] = {m: {k: list(map(float, v)) for k, v in res[m].items()}
                          for m in method_names}

    # ── verdict ──
    print(f'\n\n{"=" * 74}\n  VERDICT\n{"=" * 74}')
    print('\n  Q1 (class-discriminative without gradients?):')
    for t in all_res:
        d_ig = np.mean(all_res[t]['ELHAM-IG']['disc'])
        d_c = np.mean(all_res[t]['ELHAM-C-refine']['disc'])
        print(f'    {t:<16s} ELHAM-IG corr={d_ig:.3f}  ->  ELHAM-C-refine corr={d_c:.3f}')
    print('\n  Q2 (does the entropy signal add value over CAM?):')
    for t in all_res:
        cam = np.mean(all_res[t]['CAM']['ap'])
        best_c = max(['ELHAM-C-refine', 'ELHAM-C-Hrefine', 'ELHAM-C-mult'],
                     key=lambda m: np.mean(all_res[t][m]['ap']))
        bc = np.mean(all_res[t][best_c]['ap'])
        sc = np.mean(all_res[t]['Score-CAM']['ap'])
        verdict = 'ADDS VALUE' if bc > cam + 0.005 else 'no gain over CAM'
        print(f'    {t:<16s} CAM AP={cam:.3f}  Score-CAM AP={sc:.3f}  '
              f'best ELHAM-C ({best_c})={bc:.3f}  -> {verdict}')

    total = n_test * len(tests) * 2
    print(f'\n  {"Method":<17s}{"ms/explain":>12s}')
    print('  ' + '-' * 29)
    for m in method_names:
        print(f'  {m:<17s}{1000 * timing[m] / total:>12.2f}')
    print(f'  (Score-CAM forward passes total: {scorecam.fwd})')

    _plot(all_res, method_names)
    with open('elham_classcond.json', 'w') as f:
        json.dump(all_res, f, indent=2)
    print('\n  Saved: elham_classcond.png, elham_classcond.json')
    return all_res


def _plot(all_res, methods):
    tests = list(all_res.keys())
    cmap = plt.get_cmap('tab10')
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.4))
    panels = [('ap', 'Ground-truth localization AP ↑'),
              ('disc', 'True-vs-wrong class map corr ↓'),
              ('iou', 'IoU @ top-25% ↑')]
    for ax, (metric, title) in zip(axes, panels):
        x = np.arange(len(tests)); w = 0.8 / len(methods)
        for j, m in enumerate(methods):
            vals = [np.mean(all_res[t][m][metric]) for t in tests]
            errs = [np.std(all_res[t][m][metric]) / np.sqrt(len(all_res[t][m][metric]))
                    for t in tests]
            ax.bar(x + j * w - 0.4 + w / 2, vals, w, label=m, color=cmap(j % 10),
                   yerr=errs, capsize=2, edgecolor='white', linewidth=0.4)
        ax.set_xticks(x); ax.set_xticklabels(tests, fontsize=8)
        ax.set_title(title, fontweight='bold', fontsize=11)
        if metric == 'ap':
            ax.legend(fontsize=7, ncol=2)
        if metric == 'disc':
            ax.axhline(1.0, color='red', ls='--', alpha=0.4)
    plt.suptitle('Class-Conditional ELHAM diagnostic (gradient-free, single forward pass)',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig('elham_classcond.png', dpi=170, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    run()
