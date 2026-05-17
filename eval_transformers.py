"""
ELHAM vs Captum Baselines — Clean Comparison
==============================================
Uses Captum (PyTorch's official XAI library) for all baselines.
ELHAM is the only custom implementation.

Install: pip install captum

Usage: python eval_transformers.py
"""

import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
from collections import OrderedDict
import time, os, sys, json, hashlib, urllib.request, io
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ═══════════════════════════════════════════════════════════════════════════
# Image Loading (cached)
# ═══════════════════════════════════════════════════════════════════════════

IMG_CACHE = '/tmp/elham_test_images'
os.makedirs(IMG_CACHE, exist_ok=True)

IMG_URLS = [
    'https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/b/bf/Bulldog_inglese.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/0/0f/Grosser_Panda.JPG',
    'https://upload.wikimedia.org/wikipedia/commons/1/15/Red_Apple.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/d/d9/Collage_of_Nine_Dogs.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/6/6d/Good_Food_Display_-_NCI_Visuals_Online.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/5/5f/Red_Panda_%2824986761703%29.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/9/98/Canis_lupus_familiaris_Puppy.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/2/2e/Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/7/7b/Tabby_cat_with_blue_eyes-2008-04-27.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/8/8f/Keeshond_Magic_Black_White.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/a/af/Golden_Retriever_Carlos_%2810581910556%29.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/9/9e/Giant_Panda_in_Beijing_Zoo_1.JPG',
    'https://upload.wikimedia.org/wikipedia/commons/d/d9/Red-apple.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/3/38/Siberian_Husky_pho.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/5/5f/2010-08-09_Sechuanpfeffer_Panda_Baby_1.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/8/82/Red_Panda_in_Saint_Louis_Zoo.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/4/43/Granny_Smith_Apples.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/1/18/Dog_Breeds.jpg',
]

PREPROCESS = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_images(n=20):
    images = []
    for url in IMG_URLS:
        if len(images) >= n: break
        cache_path = os.path.join(IMG_CACHE, hashlib.md5(url.encode()).hexdigest() + '.jpg')
        if not os.path.exists(cache_path):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=30) as r:
                    with open(cache_path, 'wb') as f: f.write(r.read())
            except Exception: continue
        try:
            img = Image.open(cache_path).convert('RGB')
            images.append(PREPROCESS(img).unsqueeze(0))
        except Exception: continue

    # Fallback: synthetic images
    while len(images) < min(n, 10):
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        s = len(images)
        for y in range(224):
            for x in range(224):
                arr[y, x, 0] = int(255 * (0.5 + 0.5 * np.sin((x + s*50) / 30)))
                arr[y, x, 1] = int(255 * (0.5 + 0.5 * np.cos((y + s*30) / 25)))
                arr[y, x, 2] = int(128 + 127 * np.sin((x*y) / 2000 + s))
        images.append(PREPROCESS(Image.fromarray(arr)).unsqueeze(0))

    return torch.cat(images, dim=0) if images else None


# ═══════════════════════════════════════════════════════════════════════════
# Layer Extraction (for ELHAM)
# ═══════════════════════════════════════════════════════════════════════════

class LayerExtractor:
    def __init__(self, model, names):
        self.f = OrderedDict(); self._h = []
        for n, m in model.named_modules():
            if n in names:
                self._h.append(m.register_forward_hook(self._hook(n)))
    def _hook(self, name):
        def fn(m, i, o): self.f[name] = o.detach()
        return fn
    def clear(self): self.f.clear()
    def remove(self):
        for h in self._h: h.remove()


# ═══════════════════════════════════════════════════════════════════════════
# ELHAM
# ═══════════════════════════════════════════════════════════════════════════

class ELHAMExplainer:
    def __init__(self, model, layer_names):
        self.model = model; self.layer_names = layer_names
        self.extractor = LayerExtractor(model, layer_names)

    def _to_4d(self, feats):
        if feats.dim() == 4: return feats
        B, N, D = feats.shape
        for offset in [0, 1]:
            sq = int(np.sqrt(N - offset))
            if sq * sq == N - offset:
                patches = feats[:, offset:, :]
                return patches.reshape(B, sq, sq, D).permute(0, 3, 1, 2)
        side = int(np.sqrt(N))
        x = feats.permute(0, 2, 1).reshape(B, D, 1, N)
        return F.interpolate(x, size=(side, side), mode='bilinear', align_corners=False)

    def _channel_entropy(self, feats):
        p = F.softmax(feats, dim=1); C = feats.shape[1]
        H_raw = -(p * torch.log(p + 1e-8)).sum(dim=1)
        return (H_raw / max(np.log(C), 0.01)).squeeze(0)

    def explain(self, image, target_class):
        self.extractor.clear()
        with torch.no_grad(): _ = self.model(image)

        entropies = OrderedDict()
        for n in self.layer_names:
            spatial = self._to_4d(self.extractor.f[n])
            entropies[n] = self._channel_entropy(spatial)

        info_gains = OrderedDict(); prev_n = None
        for n in self.layer_names:
            if prev_n is not None:
                Hp, Hc = entropies[prev_n], entropies[n]
                if Hp.shape != Hc.shape:
                    Hp = F.interpolate(Hp.unsqueeze(0).unsqueeze(0), size=Hc.shape,
                                       mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                info_gains[n] = torch.clamp(Hp - Hc, min=0).cpu().numpy()
            prev_n = n

        H_in, W_in = image.shape[2], image.shape[3]
        combined = torch.zeros(H_in, W_in)
        for n, att in info_gains.items():
            t = torch.tensor(att)
            combined += F.interpolate(t.unsqueeze(0).unsqueeze(0), size=(H_in, W_in),
                                      mode='bilinear', align_corners=False).squeeze()
        return combined.numpy(), info_gains, entropies

    def remove(self): self.extractor.remove()


# ═══════════════════════════════════════════════════════════════════════════
# Captum Baselines (reference implementations)
# ═══════════════════════════════════════════════════════════════════════════

try:
    from captum.attr import LayerGradCam, Saliency, IntegratedGradients, NoiseTunnel
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False
    print('WARNING: captum not installed. Run: pip install captum')


class CaptumGradCAM:
    """Grad-CAM via Captum's reference implementation."""
    def __init__(self, model, target_layer_name):
        self.model = model
        # Find the actual module
        for n, m in model.named_modules():
            if n == target_layer_name:
                self.target_layer = m
                break
        else:
            raise ValueError(f'Layer {target_layer_name} not found')
        self.attributor = LayerGradCam(model, self.target_layer)

    def explain(self, image, target_class):
        attr = self.attributor.attribute(image, target=target_class, relu_attributions=True)
        # attr is [1, 1, H, W] or [1, H, W]
        if attr.dim() == 3:
            attr = attr.unsqueeze(1)
        cam = F.interpolate(attr, size=image.shape[2:],
                            mode='bilinear', align_corners=False).squeeze()
        return cam.detach().cpu().numpy()

    def remove(self): pass


class CaptumSaliency:
    """Saliency maps via Captum."""
    def __init__(self, model):
        self.attributor = Saliency(model)

    def explain(self, image, target_class):
        attr = self.attributor.attribute(image, target=target_class, abs=False)
        return attr.abs().max(dim=1)[0].squeeze(0).detach().cpu().numpy()

    def remove(self): pass


class CaptumIG:
    """Integrated Gradients via Captum."""
    def __init__(self, model):
        self.attributor = IntegratedGradients(model)

    def explain(self, image, target_class):
        attr = self.attributor.attribute(image, target=target_class, n_steps=20,
                                         internal_batch_size=1)
        return attr.abs().max(dim=1)[0].squeeze(0).detach().cpu().numpy()

    def remove(self): pass


class CaptumSmoothGrad:
    """SmoothGrad via Captum NoiseTunnel."""
    def __init__(self, model):
        self.saliency = Saliency(model)
        self.attributor = NoiseTunnel(self.saliency)

    def explain(self, image, target_class):
        attr = self.attributor.attribute(image, target=target_class, nt_type='smoothgrad',
                                         nt_samples=15, stdevs=0.15)
        return attr.abs().max(dim=1)[0].squeeze(0).detach().cpu().numpy()

    def remove(self): pass


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def _gaussian_blur(image, ks=15, sigma=5):
    if hasattr(F, 'gaussian_blur'): return F.gaussian_blur(image, ks, sigma)
    C = image.shape[1]
    x = torch.arange(ks, dtype=torch.float32, device=image.device) - ks//2
    g = torch.exp(-x**2/(2*sigma**2)); g /= g.sum()
    kh = g.view(1,1,1,-1).repeat(C,1,1,1); kv = g.view(1,1,-1,1).repeat(C,1,1,1)
    pad = ks//2
    out = F.conv2d(F.pad(image,(pad,pad,0,0),mode='reflect'),kh,groups=C)
    return F.conv2d(F.pad(out,(0,0,pad,pad),mode='reflect'),kv,groups=C)

def insertion_auc(model, image, tc, attr_map, steps=15):
    H,W = image.shape[2],image.shape[3]; blurred = _gaussian_blur(image)
    order = np.argsort(attr_map.flatten())[::-1].copy()
    n_px = H*W; ps = max(1,n_px//steps); scores = []
    with torch.no_grad():
        for s in range(steps+1):
            mask = torch.zeros(n_px, device=DEVICE); n_ins = s*ps
            if n_ins>0: mask[torch.from_numpy(order[:n_ins].copy())]=1
            mask = mask.view(H,W).unsqueeze(0).unsqueeze(0)
            out = model(mask*image+(1-mask)*blurred)
            scores.append(F.softmax(out,dim=1)[0,tc].item())
    return np.trapz(scores)/steps

def deletion_auc(model, image, tc, attr_map, steps=15):
    H,W = image.shape[2],image.shape[3]
    order = np.argsort(attr_map.flatten())[::-1].copy()
    n_px = H*W; ps = max(1,n_px//steps); scores = []
    with torch.no_grad():
        for s in range(steps+1):
            mask = torch.ones(n_px, device=DEVICE); n_rem = s*ps
            if n_rem>0: mask[torch.from_numpy(order[:n_rem].copy())]=0
            mask = mask.view(H,W).unsqueeze(0).unsqueeze(0)
            out = model(image*mask)
            scores.append(F.softmax(out,dim=1)[0,tc].item())
    return np.trapz(scores)/steps

def energy_pointing_game(attr_map):
    H,W = attr_map.shape; my,mx = np.unravel_index(attr_map.argmax(), attr_map.shape)
    r = min(H,W)//4; Y,X = np.ogrid[:H,:W]
    return float(attr_map[(Y-my)**2+(X-mx)**2 < r**2].sum()/(attr_map.sum()+1e-8))

def sparseness(attr_map):
    x = np.sort(attr_map.flatten()); n = len(x)
    if x.sum()==0: return 0.0
    return float((2*np.arange(1,n+1)-n-1).dot(x)/(n*x.sum()))


# ═══════════════════════════════════════════════════════════════════════════
# Architecture Registry
# ═══════════════════════════════════════════════════════════════════════════

def _vit_blocks(model):
    blocks = []
    for n, m in model.named_modules():
        if 'encoder.layers.encoder_layer_' in n:
            suffix = n.split('encoder_layer_')[-1]
            if suffix.isdigit() and '.' not in suffix:
                blocks.append(n)
    blocks.sort(key=lambda x: int(x.split('encoder_layer_')[-1]))
    idxs = [0, len(blocks)//3, 2*len(blocks)//3, len(blocks)-1]
    return [blocks[i] for i in idxs]

def _swin_layers(model):
    """Find Swin stage outputs — use the norm layers after each stage."""
    candidates = []
    for n, m in model.named_modules():
        if n.startswith('features.') and n.count('.') == 1 and n.split('.')[1].isdigit():
            candidates.append(n)
    step = max(1, len(candidates)//4)
    return candidates[::step][:4]

def _cnn_blocks(model):
    """Find named sequential blocks in CNN models."""
    candidates = []
    for n, m in model.named_modules():
        # Look for Sequential containers with common CNN names
        if n in ['layer1','layer2','layer3','layer4'] or \
           (n.startswith('features.') and n.count('.') == 1 and n.split('.')[1].isdigit()):
            candidates.append(n)
    return candidates[:4] if len(candidates) >= 4 else candidates


ARCHITECTURES = [
    {
        'name': 'ViT-B/16',
        'make': lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1).eval(),
        'layers': _vit_blocks,
        'gradcam_layer': lambda m: _vit_blocks(m)[-1],
        'type': 'vit',
    },
    {
        'name': 'ViT-B/32',
        'make': lambda: models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1).eval(),
        'layers': _vit_blocks,
        'gradcam_layer': lambda m: _vit_blocks(m)[-1],
        'type': 'vit',
    },
    {
        'name': 'ViT-L/16',
        'make': lambda: models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1).eval(),
        'layers': _vit_blocks,
        'gradcam_layer': lambda m: _vit_blocks(m)[-1],
        'type': 'vit',
    },
    {
        'name': 'Swin-T',
        'make': lambda: models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1).eval(),
        'layers': _swin_layers,
        'gradcam_layer': lambda m: 'features.7',
        'type': 'hierarchical',
    },
    {
        'name': 'ResNet50',
        'make': lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).eval(),
        'layers': lambda m: ['layer1','layer2','layer3','layer4'],
        'gradcam_layer': lambda m: 'layer4',
        'type': 'cnn',
    },
    {
        'name': 'ConvNeXt-T',
        'make': lambda: models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1).eval(),
        'layers': lambda m: _cnn_blocks(m),
        'gradcam_layer': lambda m: _cnn_blocks(m)[-1],
        'type': 'cnn',
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_architecture(arch, images, n_eval=20):
    name = arch['name']
    print(f'\n{"="*70}\n  {name}\n{"="*70}')

    print('  Loading model...')
    model = arch['make']().to(DEVICE)
    elham_layers = arch['layers'](model)
    gc_layer = arch['gradcam_layer'](model)
    print(f'  ELHAM layers: {elham_layers}')
    print(f'  Grad-CAM layer: {gc_layer}')

    # Build explainers
    elham = ELHAMExplainer(model, elham_layers)

    if HAS_CAPTUM:
        gradcam = CaptumGradCAM(model, gc_layer)
        saliency = CaptumSaliency(model)
        ig = CaptumIG(model)
        sg = CaptumSmoothGrad(model)
        explainers = [
            ('ELHAM', elham, False),
            ('GradCAM', gradcam, True),
            ('Saliency', saliency, True),
            ('IG', ig, True),
            ('SmoothGrad', sg, True),
        ]
        print('  Baselines: Captum (reference implementations)')
    else:
        explainers = [('ELHAM', elham, False)]
        print('  WARNING: Captum not installed — only ELHAM available')

    results = {}
    for method, _, _ in explainers:
        results[method] = {'ins':[], 'del':[], 'epg':[], 'sparse':[], 'time':[]}

    viz_data = []
    n = min(n_eval, len(images))

    for i in range(n):
        img = images[i:i+1].to(DEVICE)
        with torch.no_grad():
            pred = model(img).argmax(1).item()
        tc = pred

        for method, ex, is_captum in explainers:
            t0 = time.time()
            if method == 'ELHAM':
                attr, ig_maps, ent = ex.explain(img, tc)
            else:
                attr = ex.explain(img, tc)
            dt = time.time() - t0
            results[method]['time'].append(dt)
            results[method]['ins'].append(insertion_auc(model, img, tc, attr))
            results[method]['del'].append(deletion_auc(model, img, tc, attr))
            results[method]['epg'].append(energy_pointing_game(attr))
            results[method]['sparse'].append(sparseness(attr))

        if i < 4:
            _, ig_maps, ent = elham.explain(img, tc)
            viz_data.append((img.cpu(), ig_maps, ent, tc))

    # Print
    for method, _, _ in explainers:
        r = results[method]
        print(f'  {method:12s}  Ins={np.mean(r["ins"]):.3f}±{np.std(r["ins"]):.3f}  '
              f'Del={np.mean(r["del"]):.3f}±{np.std(r["del"]):.3f}  '
              f'EPG={np.mean(r["epg"]):.3f}  Spar={np.mean(r["sparse"]):.3f}  '
              f'{np.mean(r["time"])*1000:.1f}ms')

    # ELHAM vs Grad-CAM deltas
    e, g = results['ELHAM'], results['GradCAM']
    deltas = {
        'Ins': float(np.mean(e['ins']) - np.mean(g['ins'])),
        'Del': float(np.mean(g['del']) - np.mean(e['del'])),
        'EPG': float(np.mean(e['epg']) - np.mean(g['epg'])),
        'Spar': float(np.mean(e['sparse']) - np.mean(g['sparse'])),
        'Speed': float(np.mean(g['time']) - np.mean(e['time'])) * 1000,
    }
    winner = sum(1 for v in [deltas['Ins'], deltas['Del'], deltas['EPG'], deltas['Spar']] if v > 0)
    print(f'  Δ ELHAM−GradCAM: Ins={deltas["Ins"]:+.3f} Del={deltas["Del"]:+.3f} '
          f'EPG={deltas["EPG"]:+.3f} Spar={deltas["Spar"]:+.3f} '
          f'→ ELHAM wins {winner}/4')

    elham.remove()

    # Plot
    if viz_data:
        _plot(arch['name'], viz_data[:4], explainers, model, images[:4].to(DEVICE))

    return {'name': name, 'type': arch['type'], 'results': results, 'deltas': deltas, 'wins': winner}


def _plot(name, viz_data, explainers, model, images):
    n = min(len(viz_data), 4); n_cols = 7
    fig, axes = plt.subplots(n, n_cols, figsize=(22, 3.5*n))
    if n == 1: axes = axes.reshape(1, -1)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for row in range(n):
        img_t, info_gains, entropies, tc = viz_data[row]
        img = (img_t.squeeze(0)*std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        H, W = img.shape[0], img.shape[1]

        axes[row, 0].imshow(img); axes[row, 0].set_title('Input', fontsize=8); axes[row, 0].axis('off')

        # Initial entropy
        first_key = list(entropies.keys())[0]
        H0 = entropies[first_key]
        if isinstance(H0, torch.Tensor): H0 = H0.cpu()
        H0_up = F.interpolate(torch.from_numpy(np.asarray(H0)).float().unsqueeze(0).unsqueeze(0),
                              size=(H, W), mode='bilinear', align_corners=False).squeeze().numpy()
        axes[row, 1].imshow(H0_up, cmap='viridis_r')
        axes[row, 1].set_title(f'H({first_key})', fontsize=7); axes[row, 1].axis('off')

        # Info gains
        layer_keys = list(info_gains.keys())
        combined = np.zeros((H, W))
        for j, lk in enumerate(layer_keys[:3]):
            ig = info_gains[lk]
            if isinstance(ig, torch.Tensor): ig = ig.cpu().numpy()
            up = F.interpolate(torch.tensor(ig).unsqueeze(0).unsqueeze(0), size=(H, W),
                               mode='bilinear', align_corners=False).squeeze().numpy()
            combined += up
            axes[row, 2+j].imshow(up, cmap='inferno')
            axes[row, 2+j].set_title(f'ΔI {lk}', fontsize=7); axes[row, 2+j].axis('off')

        # ELHAM overlay
        axes[row, 5].imshow(img)
        if combined.max() > 0: axes[row, 5].imshow(combined, cmap='inferno', alpha=0.55)
        axes[row, 5].set_title('ELHAM', fontsize=8); axes[row, 5].axis('off')

        # Captum Grad-CAM overlay
        axes[row, 6].imshow(img)
        # Re-compute Grad-CAM for this image
        img_cuda = images[row:row+1].to(DEVICE)
        with torch.no_grad(): tc = model(img_cuda).argmax(1).item()
        for method, ex, _ in explainers:
            if method == 'GradCAM':
                gc_attr = ex.explain(img_cuda, tc)
                if gc_attr.max() > 0:
                    axes[row, 6].imshow(gc_attr, cmap='inferno', alpha=0.55)
        axes[row, 6].set_title('Grad-CAM (Captum)', fontsize=8); axes[row, 6].axis('off')

    plt.suptitle(f'ELHAM vs Captum Grad-CAM — {name}', fontweight='bold', fontsize=13)
    plt.tight_layout()
    fname = f'elham_{name.lower().replace(" ", "_").replace("/", "_")}.png'
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {fname}')


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

def generate_summary(all_results):
    print(f'\n\n{"="*70}')
    print('SUMMARY: ELHAM vs Captum Baselines')
    print('='*70)

    print(f'\n{"Architecture":<16s} {"Type":<12s} {"ELHAM Ins":>10s} {"GCAM Ins":>10s} '
          f'{"ELHAM Del":>10s} {"GCAM Del":>10s} {"ELHAM Wins":>10s}')
    print('-'*80)
    for r in all_results:
        e = r['results']['ELHAM']; g = r['results']['GradCAM']
        print(f'{r["name"]:<16s} {r["type"]:<12s} '
              f'{np.mean(e["ins"]):>10.3f} {np.mean(g["ins"]):>10.3f} '
              f'{np.mean(e["del"]):>10.3f} {np.mean(g["del"]):>10.3f} '
              f'{r["wins"]}/4')

    vits = [r for r in all_results if r['type'] == 'vit']
    others = [r for r in all_results if r['type'] != 'vit']
    print(f'\n  ViT avg: ELHAM wins {np.mean([r["wins"] for r in vits]):.1f}/4 metrics')
    print(f'  Others avg: ELHAM wins {np.mean([r["wins"] for r in others]):.1f}/4 metrics')

    # Report
    report = ['# ELHAM vs Captum Baselines\n\n',
              'All baselines use Captum (PyTorch official XAI library) reference implementations.\n',
              'ELHAM is the only custom implementation.\n\n',
              '| Architecture | Type | ELHAM Ins | GradCAM Ins | ELHAM Del | GradCAM Del | Winner |\n',
              '|-------------|------|-----------|-------------|-----------|-------------|--------|\n']
    for r in all_results:
        e = r['results']['ELHAM']; g = r['results']['GradCAM']
        report.append(f'| {r["name"]:<15s} | {r["type"]:<5s} | '
                     f'{np.mean(e["ins"]):.3f} | {np.mean(g["ins"]):.3f} | '
                     f'{np.mean(e["del"]):.3f} | {np.mean(g["del"]):.3f} | '
                     f'{r["wins"]}/4 |\n')
    report.append('\n*Grad-CAM implementation: captum.attr.LayerGradCam (Meta/PyTorch official)*\n')
    with open('ELHAM_vs_CAPTUM.md', 'w') as f: f.write(''.join(report))
    print(f'\n  Saved: ELHAM_vs_CAPTUM.md')


def main():
    print(f'Device: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)} '
              f'({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)')
    print(f'Captum available: {HAS_CAPTUM}')
    if not HAS_CAPTUM:
        print('\n❌ Please install Captum first: pip install captum')
        return

    print('Loading test images (cached)...')
    images = load_images(n=20)
    if images is None:
        print('No images available')
        return
    print(f'  {images.shape[0]} images ready')

    all_results = []
    for arch in ARCHITECTURES:
        try:
            result = evaluate_architecture(arch, images)
            all_results.append(result)
        except Exception as e:
            print(f'  FAILED: {e}')
            import traceback; traceback.print_exc()

    generate_summary(all_results)
    print('\nDONE')


if __name__ == '__main__':
    main()
