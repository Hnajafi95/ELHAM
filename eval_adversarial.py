"""
ELHAM: Adversarial Robustness of Attributions
===============================================
Tests whether ELHAM's attribution maps are more stable than gradient-based
methods under adversarial attack.

Key insight: FGSM computes sign(∇_x L). Gradient-based explainers (Saliency,
Grad-CAM, IG) are causally linked to this gradient direction. ELHAM measures
internal channel entropy — it never touches ∇_x.

Test: Generate FGSM + PGD attacks. Measure attribution stability (Spearman r
between clean and attacked maps) for ELHAM vs Captum Grad-CAM vs Captum Saliency.

Usage: python eval_adversarial.py
"""

import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
from collections import OrderedDict
import time, os, copy, hashlib, urllib.request, io
from PIL import Image
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    from captum.attr import LayerGradCam, Saliency
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False


# ═══════════════════════════════════════════════════════════════════════════
# ELHAM (imported from existing implementation)
# ═══════════════════════════════════════════════════════════════════════════

class LayerExtractor:
    def __init__(self, model, names):
        self.f = OrderedDict(); self._h = []
        for n, m in model.named_modules():
            if n in names: self._h.append(m.register_forward_hook(self._hook(n)))
    def _hook(self, name):
        def fn(m,i,o): self.f[name] = o.detach()
        return fn
    def clear(self): self.f.clear()
    def remove(self):
        for h in self._h: h.remove()


class ELHAMExplainer:
    def __init__(self, model, layer_names):
        self.model = model; self.layer_names = layer_names
        self.extractor = LayerExtractor(model, layer_names)

    def _to_4d(self, feats):
        if feats.dim() == 4: return feats
        B,N,D = feats.shape
        for off in [0,1]:
            sq = int(np.sqrt(N-off))
            if sq*sq == N-off:
                return feats[:,off:,:].reshape(B,sq,sq,D).permute(0,3,1,2)
        side = int(np.sqrt(N))
        return F.interpolate(feats.permute(0,2,1).reshape(B,D,1,N),
                             size=(side,side),mode='bilinear',align_corners=False)

    def _channel_entropy(self, feats):
        p = F.softmax(feats, dim=1); C = feats.shape[1]
        return (-(p*torch.log(p+1e-8)).sum(dim=1)/max(np.log(C),0.01)).squeeze(0)

    def explain(self, image, target_class):
        self.extractor.clear()
        with torch.no_grad(): _ = self.model(image)
        entropies = OrderedDict()
        for n in self.layer_names:
            entropies[n] = self._channel_entropy(self._to_4d(self.extractor.f[n]))
        info_gains = OrderedDict(); prev_n = None
        for n in self.layer_names:
            if prev_n is not None:
                Hp,Hc = entropies[prev_n], entropies[n]
                if Hp.shape != Hc.shape:
                    Hp = F.interpolate(Hp.unsqueeze(0).unsqueeze(0),size=Hc.shape,
                                       mode='bilinear',align_corners=False).squeeze(0).squeeze(0)
                info_gains[n] = torch.clamp(Hp-Hc,min=0).cpu().numpy()
            prev_n = n
        H_in,W_in = image.shape[2],image.shape[3]; combined = torch.zeros(H_in,W_in)
        for n,att in info_gains.items():
            combined += F.interpolate(torch.tensor(att).unsqueeze(0).unsqueeze(0),
                                      size=(H_in,W_in),mode='bilinear',
                                      align_corners=False).squeeze()
        return combined.numpy(), info_gains, entropies
    def remove(self): self.extractor.remove()


# ═══════════════════════════════════════════════════════════════════════════
# Adversarial Attack Generators
# ═══════════════════════════════════════════════════════════════════════════

def fgsm_attack(model, image, target, epsilon):
    """Fast Gradient Sign Method — perturb toward increasing loss."""
    img = image.clone().detach().requires_grad_(True)
    model.zero_grad()
    loss = F.cross_entropy(model(img), torch.tensor([target], device=DEVICE))
    loss.backward()
    perturbed = image + epsilon * img.grad.sign()
    return torch.clamp(perturbed, -2.5, 2.5)  # ImageNet normalization range


def pgd_attack(model, image, target, epsilon, steps=10, alpha=None):
    """Projected Gradient Descent — iterative FGSM with projection."""
    if alpha is None: alpha = epsilon / (steps // 2)
    img_orig = image.clone().detach()
    img_adv = image.clone().detach() + torch.zeros_like(image).uniform_(-epsilon, epsilon)
    img_adv = torch.clamp(img_adv, -2.5, 2.5)

    for _ in range(steps):
        img_adv = img_adv.clone().detach().requires_grad_(True)
        model.zero_grad()
        loss = F.cross_entropy(model(img_adv), torch.tensor([target], device=DEVICE))
        loss.backward()
        with torch.no_grad():
            img_adv = img_adv + alpha * img_adv.grad.sign()
            # Project to epsilon-ball
            delta = torch.clamp(img_adv - img_orig, -epsilon, epsilon)
            img_adv = torch.clamp(img_orig + delta, -2.5, 2.5)
    return img_adv


# ═══════════════════════════════════════════════════════════════════════════
# Image Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_images(n=15):
    urls = [
        'https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/b/bf/Bulldog_inglese.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/0/0f/Grosser_Panda.JPG',
        'https://upload.wikimedia.org/wikipedia/commons/1/15/Red_Apple.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/d/d9/Collage_of_Nine_Dogs.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/5/5f/Red_Panda_%2824986761703%29.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/9/98/Canis_lupus_familiaris_Puppy.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/9/9e/Giant_Panda_in_Beijing_Zoo_1.JPG',
        'https://upload.wikimedia.org/wikipedia/commons/3/38/Siberian_Husky_pho.jpg',
    ]
    cache = '/tmp/elham_test_images'; os.makedirs(cache, exist_ok=True)
    preprocess = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    images = []
    for url in urls:
        if len(images) >= n: break
        fname = os.path.join(cache, hashlib.md5(url.encode()).hexdigest()+'.jpg')
        if not os.path.exists(fname):
            try:
                req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=30) as r:
                    with open(fname,'wb') as f: f.write(r.read())
            except: continue
        try: images.append(preprocess(Image.open(fname).convert('RGB')).unsqueeze(0))
        except: continue
    return torch.cat(images, dim=0) if images else None


# ═══════════════════════════════════════════════════════════════════════════
# Main Test
# ═══════════════════════════════════════════════════════════════════════════

def run_adversarial_test():
    print(f'Device: {DEVICE}')
    print(f'Captum: {"available" if HAS_CAPTUM else "NOT INSTALLED"}')

    images = load_images(n=15)
    if images is None: print('No images'); return
    print(f'  {images.shape[0]} images ready\n')

    # Load model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).eval().to(DEVICE)
    layers = ['layer1','layer2','layer3','layer4']

    # Build explainers
    elham = ELHAMExplainer(model, layers)
    gcam = LayerGradCam(model, model.layer4) if HAS_CAPTUM else None
    saliency = Saliency(model) if HAS_CAPTUM else None

    # Attack configurations
    attacks = [
        ('FGSM ε=2/255', lambda img, tc: fgsm_attack(model, img, tc, 2/255)),
        ('FGSM ε=4/255', lambda img, tc: fgsm_attack(model, img, tc, 4/255)),
        ('FGSM ε=8/255', lambda img, tc: fgsm_attack(model, img, tc, 8/255)),
        ('PGD ε=8/255,10 steps', lambda img, tc: pgd_attack(model, img, tc, 8/255, 10)),
    ]

    # Results storage: {method: {attack_name: [correlations]}}
    results = {'ELHAM': {a[0]: [] for a in attacks},
               'GradCAM': {a[0]: [] for a in attacks},
               'Saliency': {a[0]: [] for a in attacks}}
    attack_success = {a[0]: 0 for a in attacks}
    cross_image_baseline = []  # control: corr between two different clean images

    n = min(15, len(images))

    for i in range(n):
        print(f'  Image {i+1}/{n}...')
        img = images[i:i+1].to(DEVICE)

        # Clean prediction
        model.eval()
        with torch.no_grad():
            tc_clean = model(img).argmax(1).item()

        # Clean attribution maps
        m_elham_clean, _, _ = elham.explain(img, tc_clean)
        if HAS_CAPTUM:
            m_gcam_clean = gcam.attribute(img, target=tc_clean, relu_attributions=True)
            m_gcam_clean = F.interpolate(m_gcam_clean, size=(224,224),
                                         mode='bilinear',align_corners=False).squeeze().detach().cpu().numpy()
            m_sal_clean = saliency.attribute(img, target=tc_clean, abs=False)
            m_sal_clean = m_sal_clean.abs().max(dim=1)[0].squeeze(0).detach().cpu().numpy()

        # Control: correlation with a different clean image
        if i > 0:
            m_other, _, _ = elham.explain(images[i-1:i].to(DEVICE), tc_clean)
            if m_elham_clean.std()>0 and m_other.std()>0:
                r_ctrl,_ = spearmanr(m_elham_clean.flatten(), m_other.flatten())
                cross_image_baseline.append(r_ctrl)

        # Test each attack
        for attack_name, attack_fn in attacks:
            img_adv = attack_fn(img, tc_clean)

            # Check if attack succeeded
            with torch.no_grad():
                tc_adv = model(img_adv).argmax(1).item()
            if tc_adv != tc_clean:
                attack_success[attack_name] += 1

            # ELHAM on attacked image
            m_elham_adv, _, _ = elham.explain(img_adv, tc_clean)  # explain w.r.t ORIGINAL class
            if m_elham_clean.std()>0 and m_elham_adv.std()>0:
                r_e,_ = spearmanr(m_elham_clean.flatten(), m_elham_adv.flatten())
                results['ELHAM'][attack_name].append(r_e)

            # Captum on attacked image
            if HAS_CAPTUM:
                try:
                    m_gcam_adv = gcam.attribute(img_adv, target=tc_clean, relu_attributions=True)
                    m_gcam_adv = F.interpolate(m_gcam_adv, size=(224,224),
                                               mode='bilinear',align_corners=False).squeeze().detach().cpu().numpy()
                    if m_gcam_clean.std()>0 and m_gcam_adv.std()>0:
                        r_g,_ = spearmanr(m_gcam_clean.flatten(), m_gcam_adv.flatten())
                        results['GradCAM'][attack_name].append(r_g)
                except Exception: pass

                try:
                    m_sal_adv = saliency.attribute(img_adv, target=tc_clean, abs=False)
                    m_sal_adv = m_sal_adv.abs().max(dim=1)[0].squeeze(0).detach().cpu().numpy()
                    if m_sal_clean.std()>0 and m_sal_adv.std()>0:
                        r_s,_ = spearmanr(m_sal_clean.flatten(), m_sal_adv.flatten())
                        results['Saliency'][attack_name].append(r_s)
                except Exception: pass

    # ── Print Results ──
    print(f'\n{"="*70}')
    print('RESULTS: Attribution Stability Under Adversarial Attack')
    print('='*70)

    ctrl_r = np.mean(cross_image_baseline) if cross_image_baseline else 0
    print(f'\n  Control (different images): r = {ctrl_r:.3f}')
    print(f'  (This is the baseline for "maps should differ")\n')

    print(f'  {"Attack":<22s} {"ELHAM r":>8s} {"GradCAM r":>8s} {"Saliency r":>8s} {"ELHAM Wins?":>12s} {"Attack Succ":>10s}')
    print('  ' + '-'*72)
    elham_wins = 0; total = 0
    for attack_name, _ in attacks:
        e = np.mean(results['ELHAM'][attack_name]) if results['ELHAM'][attack_name] else 0
        g = np.mean(results['GradCAM'][attack_name]) if results['GradCAM'][attack_name] else 0
        s = np.mean(results['Saliency'][attack_name]) if results['Saliency'][attack_name] else 0
        succ = attack_success[attack_name]
        # Higher correlation = more stable under attack = better
        # ELHAM wins if it has higher correlation than BOTH gradient methods
        winner = '✓ ELHAM' if (e > g and e > s) else ('~ Tied' if (e > max(g,s)-0.05) else 'Gradient')
        if 'ELHAM' in winner: elham_wins += 1
        total += 1
        print(f'  {attack_name:<22s} {e:>8.3f} {g:>8.3f} {s:>8.3f} {winner:>12s} {succ:>3d}/{n}')

    print(f'\n  ELHAM most stable on {elham_wins}/{total} attacks')
    print(f'  Attack success rate: {sum(attack_success.values())}/{n*len(attacks)} ({100*sum(attack_success.values())/(n*len(attacks)):.0f}%)')

    # Interpretation
    print(f'\n  Interpretation:')
    print(f'  - Cross-image baseline r = {ctrl_r:.3f} (maps for different images)')
    print(f'  - Higher r under attack = attribution is more stable')
    print(f'  - r near baseline = maps change as much as viewing a different image (BAD)')
    print(f'  - r near 1.0 = maps are unchanged by attack (attack failed OR method is insensitive)')
    print(f'  - Attack success column: attacks that actually fooled the model')

    # ── Plot ──
    _plot_results(attacks, results, attack_success, n, ctrl_r)

    elham.remove()
    return results


def _plot_results(attacks, results, attack_success, n, ctrl_r):
    attack_names = [a[0] for a in attacks]
    x = np.arange(len(attack_names))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Correlation per method per attack
    ax = axes[0]
    for j, (method, color, offset) in enumerate([
        ('ELHAM', '#E91E63', -width), ('GradCAM', '#2196F3', 0), ('Saliency', '#4CAF50', width)
    ]):
        vals = [np.mean(results[method].get(name, [0])) for name in attack_names]
        errs = [np.std(results[method].get(name, [0])) for name in attack_names]
        ax.bar(x + offset, vals, width, color=color, label=method, yerr=errs,
               capsize=3, edgecolor='white', linewidth=0.5)
    ax.axhline(y=ctrl_r, color='gray', linestyle='--', alpha=0.5, label=f'Cross-img baseline (r={ctrl_r:.2f})')
    ax.set_xticks(x); ax.set_xticklabels(attack_names, fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('Spearman r (clean vs attacked)'); ax.set_title('Attribution Stability Under Attack')
    ax.legend(fontsize=7); ax.set_ylim(0, 1.05)

    # Plot 2: ELHAM advantage over Grad-CAM
    ax = axes[1]
    elham_adv = []
    for name in attack_names:
        e = np.mean(results['ELHAM'].get(name, [0]))
        g = np.mean(results['GradCAM'].get(name, [0]))
        elham_adv.append(e - g)
    colors = ['#4CAF50' if v > 0.05 else '#FF9800' if v > -0.05 else '#F44336' for v in elham_adv]
    ax.bar(x, elham_adv, color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(attack_names, fontsize=8, rotation=15, ha='right')
    ax.set_ylabel('Δ Spearman r (ELHAM − GradCAM)')
    ax.set_title('ELHAM Stability Advantage')
    for i, (v, succ) in enumerate(zip(elham_adv, [attack_success[a[0]] for a in attacks])):
        ax.text(i, v + 0.02, f'{succ}/{n}', ha='center', fontsize=8, fontweight='bold')

    plt.suptitle('ELHAM vs Gradient Methods: Adversarial Attribution Robustness',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig('elham_adversarial.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  Saved: elham_adversarial.png')


if __name__ == '__main__':
    run_adversarial_test()
