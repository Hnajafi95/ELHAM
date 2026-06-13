"""
ELHAM: Cross-Width Attribution Consistency
============================================
Tests ELHAM's log(C) normalization: do maps remain comparable when
channel counts change? Trains CIFARCNN variants at different channel
widths and measures cross-width attribution similarity.

Also tests: Captum Grad-CAM on narrower models (hooks must find
correct layer names despite different internal dimensions).

Usage: python eval_pruning.py
"""

import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    from captum.attr import LayerGradCam
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False


# ═══════════════════════════════════════════════════════════════════════════

class CIFARCNN(nn.Module):
    def __init__(self, widths=[64,128,256,512], nc=10):
        super().__init__()
        c1,c2,c3,c4 = widths
        self.layer1 = nn.Sequential(nn.Conv2d(3,c1,3,padding=1),nn.BatchNorm2d(c1),nn.ReLU(),nn.Conv2d(c1,c1,3,padding=1),nn.BatchNorm2d(c1),nn.ReLU(),nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(c1,c2,3,padding=1),nn.BatchNorm2d(c2),nn.ReLU(),nn.Conv2d(c2,c2,3,padding=1),nn.BatchNorm2d(c2),nn.ReLU(),nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(c2,c3,3,padding=1),nn.BatchNorm2d(c3),nn.ReLU(),nn.Conv2d(c3,c3,3,padding=1),nn.BatchNorm2d(c3),nn.ReLU(),nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(nn.Conv2d(c3,c4,3,padding=1),nn.BatchNorm2d(c4),nn.ReLU(),nn.Conv2d(c4,c4,3,padding=1),nn.BatchNorm2d(c4),nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1); self.fc = nn.Linear(c4,nc)
    def forward(self,x):
        x=self.layer1(x);x=self.layer2(x);x=self.layer3(x);x=self.layer4(x)
        return self.fc(self.pool(x).view(x.size(0),-1))


def train_model(model, epochs=6):
    tr = transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    te = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    train_ds = datasets.CIFAR10('/tmp/c10',train=True,download=True,transform=tr)
    test_ds  = datasets.CIFAR10('/tmp/c10',train=False,download=True,transform=te)
    train_ld = DataLoader(train_ds,128,shuffle=True,num_workers=2); test_ld = DataLoader(test_ds,128,shuffle=False,num_workers=2)
    opt = torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    crit = nn.CrossEntropyLoss(); best = 0
    for ep in range(epochs):
        model.train(); ls,cor=0,0
        for imgs,lbls in train_ld:
            imgs,lbls=imgs.to(DEVICE),lbls.to(DEVICE); opt.zero_grad()
            loss=crit(model(imgs),lbls); loss.backward(); opt.step()
            ls+=loss.item(); cor+=(model(imgs).argmax(1)==lbls).sum().item()
        sch.step(); model.eval(); tc=0
        with torch.no_grad():
            for imgs,lbls in test_ld: imgs,lbls=imgs.to(DEVICE),lbls.to(DEVICE); tc+=(model(imgs).argmax(1)==lbls).sum().item()
        acc=tc/len(test_ds)
        if acc>best: best=acc
        print(f'    Ep {ep+1}: test={acc:.3f}')
    return best


# ═══════════════════════════════════════════════════════════════════════════
# ELHAM
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
    def _channel_entropy(self, feats):
        p = F.softmax(feats, dim=1); C = feats.shape[1]
        return (-(p*torch.log(p+1e-8)).sum(dim=1)/max(np.log(C),0.01)).squeeze(0)
    def explain(self, image, target_class):
        self.extractor.clear()
        with torch.no_grad(): _ = self.model(image)
        entropies = OrderedDict()
        for n in self.layer_names:
            entropies[n] = self._channel_entropy(self.extractor.f[n])
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
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run_test():
    print(f'Device: {DEVICE}')
    print(f'Captum: {"available" if HAS_CAPTUM else "NOT INSTALLED"}\n')

    # Channel width variants (fraction of default 64-128-256-512)
    width_factors = [1.0, 0.75, 0.5, 0.25]
    models = {}
    accuracies = {}

    for factor in width_factors:
        widths = [max(4, int(64*factor)), max(8, int(128*factor)), max(8, int(256*factor)), max(16, int(512*factor))]
        print(f'Training {factor:.0%} width model: {widths}...')
        m = CIFARCNN(widths=widths).to(DEVICE)
        acc = train_model(m, epochs=6)
        models[factor] = m
        accuracies[factor] = acc
        print(f'  Accuracy: {acc:.3f}\n')

    # Test samples
    te = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    test_ds = datasets.CIFAR10('/tmp/c10',train=False,download=True,transform=te)
    test_ld = DataLoader(test_ds, 8, shuffle=True)
    imgs, lbls = next(iter(test_ld)); n = min(8, len(imgs))

    layers = ['layer1','layer2','layer3','layer4']

    # Get reference maps from 100% model
    print('Computing reference maps (100% width)...')
    ref_model = models[1.0]
    ref_elham = ELHAMExplainer(ref_model, layers)
    ref_maps = []
    for i in range(n):
        img = imgs[i:i+1].to(DEVICE); tc = lbls[i].item()
        m, _, _ = ref_elham.explain(img, tc); ref_maps.append(m)
    ref_elham.remove()

    # Compare each narrower model to reference
    results = {'factor': [], 'accuracy': [], 'channels': [], 'elham_r': [], 'gcam_status': []}

    for factor in [0.75, 0.5, 0.25]:
        m = models[factor]
        widths = [max(4,int(64*factor)), max(8,int(128*factor)), max(8,int(256*factor)), max(16,int(512*factor))]
        print(f'\nEvaluating {factor:.0%} width model (channels: {widths})...')

        elham_n = ELHAMExplainer(m, layers)
        elham_corrs = []
        for i in range(n):
            img = imgs[i:i+1].to(DEVICE); tc = lbls[i].item()
            m_n, _, _ = elham_n.explain(img, tc)
            if ref_maps[i].std()>0 and m_n.std()>0:
                r,_ = spearmanr(ref_maps[i].flatten(), m_n.flatten())
                elham_corrs.append(r)

        mean_r = np.mean(elham_corrs) if elham_corrs else 0
        print(f'  ELHAM correlation (100% vs {factor:.0%}): r = {mean_r:.3f}')

        # Captum Grad-CAM
        gcam_status = 'ok'
        if HAS_CAPTUM:
            try:
                gc = LayerGradCam(m, m.layer4)
                img0 = imgs[0:1].to(DEVICE); tc0 = lbls[0].item()
                _ = gc.attribute(img0, target=tc0, relu_attributions=True)
                gcam_status = '✓ ok'
            except Exception as e:
                gcam_status = f'✗ {str(e)[:50]}'
            print(f'  Captum Grad-CAM: {gcam_status}')

        elham_n.remove()
        results['factor'].append(factor)
        results['accuracy'].append(accuracies[factor])
        results['channels'].append(str(widths))
        results['elham_r'].append(mean_r)
        results['gcam_status'].append(gcam_status)

    # Cross-model baseline: two 100% models trained from different seeds
    print('\nTraining second 100% model for cross-model baseline...')
    m2 = CIFARCNN(widths=[64,128,256,512]).to(DEVICE)
    train_model(m2, epochs=6)
    e2 = ELHAMExplainer(m2, layers)
    cross_corrs = []
    for i in range(n):
        img = imgs[i:i+1].to(DEVICE); tc = lbls[i].item()
        m2_map, _, _ = e2.explain(img, tc)
        if ref_maps[i].std()>0 and m2_map.std()>0:
            r,_ = spearmanr(ref_maps[i].flatten(), m2_map.flatten())
            cross_corrs.append(r)
    cross_r = np.mean(cross_corrs) if cross_corrs else 0
    e2.remove()
    print(f'  Cross-model baseline (same width, different seed): r = {cross_r:.3f}')
    print(f'  (This is the UPPER BOUND — maps can\'t be more similar than this)')

    # Summary
    print(f'\n{"="*70}')
    print('RESULTS: Cross-Width Attribution Consistency')
    print('='*70)
    print(f'\n  Cross-model baseline r = {cross_r:.3f} (upper bound)')
    print(f'\n  {"Width":<8s} {"Accuracy":>8s} {"Channels":>20s} {"ELHAM r":>8s} {"vs Baseline":>12s} {"Grad-CAM":>20s}')
    print('  ' + '-'*85)
    for i in range(len(results['factor'])):
        f = results['factor'][i]
        elham_r = results['elham_r'][i]
        vs_base = elham_r / cross_r if cross_r > 0 else 0
        print(f'  {f:>4.0%}    {results["accuracy"][i]:>8.3f}  {results["channels"][i]:>20s}  '
              f'{elham_r:>8.3f}  {vs_base:>11.1%}  {results["gcam_status"][i]:>20s}')

    print(f'\n  Interpretation:')
    print(f'  - Cross-model baseline = {cross_r:.3f}: same-architecture different-seed maps differ this much')
    print(f'  - If narrower model r is close to baseline → channel count change is no worse than retraining')
    print(f'  - If narrower model r ≪ baseline → channel count significantly changes attributions')
    print(f'  - ELHAM\'s log(C) normalization: tested across {len(results["factor"])} width variants')

    # Plot
    _plot(results, cross_r, n)
    return results


def _plot(results, cross_r, n):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(results['factor']))

    ax = axes[0]
    ax.plot(x, results['accuracy'], 'o-', color='#2196F3', linewidth=2, markersize=8, label='Accuracy')
    ax.set_ylabel('Accuracy', color='#2196F3'); ax.set_xlabel('Width Fraction')
    ax2 = ax.twinx()
    ax2.plot(x, results['elham_r'], 's-', color='#E91E63', linewidth=2, markersize=8, label='ELHAM r')
    ax2.axhline(y=cross_r, color='gray', linestyle='--', alpha=0.5, label=f'Cross-model baseline (r={cross_r:.3f})')
    ax2.set_ylabel('Spearman r vs 100% model', color='#E91E63')
    ax.set_xticks(x); ax.set_xticklabels([f'{f:.0%}' for f in results['factor']])
    lines1, labels1 = ax.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, loc='best', fontsize=7)
    ax.set_title('Accuracy vs Attribution Consistency')

    ax = axes[1]
    for i in range(len(results['factor'])):
        vs_base = results['elham_r'][i]/cross_r if cross_r>0 else 0
        color = '#4CAF50' if vs_base > 0.7 else '#FF9800' if vs_base > 0.4 else '#F44336'
        ax.bar(i, results['elham_r'][i], color=color, edgecolor='white', width=0.5)
    ax.axhline(y=cross_r, color='gray', linestyle='--', alpha=0.5, label='Cross-model baseline')
    ax.set_xticks(x); ax.set_xticklabels([f'{f:.0%}' for f in results['factor']])
    ax.set_ylabel('Spearman r'); ax.set_xlabel('Width Fraction')
    ax.set_title('ELHAM Cross-Width Map Consistency'); ax.legend(fontsize=7)

    plt.suptitle('ELHAM: Cross-Width Attribution Robustness', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig('elham_pruning.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  Saved: elham_pruning.png')


if __name__ == '__main__':
    run_test()
