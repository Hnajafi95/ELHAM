"""
ELHAM as an OOD / confidence signal  — the pivot pilot
======================================================
Attribution is falsified (see ROADMAP.md). But ELHAM's channel entropy is a
*class-agnostic, forward-only, training-data-free* measure of how "decided" the
network's features are — which is exactly the shape of an out-of-distribution
score. Hypothesis: in-distribution inputs drive channels to low (peaked) entropy;
OOD inputs leave them flat/high. If aggregated channel entropy separates ID from
OOD as well as the standard cheap forward baselines (MSP, Energy), the method has
a real home where its properties are assets, not flaws.

Setup
-----
ID train/test : CIFAR-10
OOD test      : SVHN (far), CIFAR-100 (near), Gaussian noise, Uniform noise
Model         : CIFAR ResNet-18 (3x3 stem, no maxpool), trained here.

Scores (anomaly = higher means MORE out-of-distribution)
--------------------------------------------------------
Baselines (use logits — the class structure):
  MSP        : -max_c softmax(logit)_c            (Hendrycks & Gimpel 2017)
  Energy     : -logsumexp(logits)                 (Liu et al. 2020)
  MaxLogit   : -max_c logit_c
ELHAM (forward-only channel entropy, class-agnostic, NO logits):
  ELHAM-Hlast: mean spatial entropy of the last conv block
  ELHAM-Hall : mean spatial entropy averaged over all blocks
  ELHAM-IG   : -mean info-gain (entropy reduction across blocks)
  ELHAM-Hmin : (1 - min spatial entropy) i.e. peak decisiveness anywhere

Metrics: AUROC (ID vs OOD), FPR@95%TPR.  Higher AUROC / lower FPR95 = better.

Usage: python eval_ood.py [--epochs 25] [--n-ood 2000]
Run heavy version on proxima3.
"""

import json
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision import transforms
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available()
                      else ('mps' if torch.backends.mps.is_available() else 'cpu'))
SEED = 0
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# ───────────────────────── CIFAR ResNet-18 ─────────────────────────

class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                       nn.BatchNorm2d(cout))

    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        return F.relu(o + self.short(x))


class CIFARResNet18(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(64), nn.ReLU())
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
        self.fc = nn.Linear(512, nc)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


# ───────────────────────── data ─────────────────────────

def loaders(root, n_ood, batch=256):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    tf_test = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

    cifar_tr = tv.datasets.CIFAR10(root, True, download=True, transform=tf_train)
    cifar_te = tv.datasets.CIFAR10(root, False, download=True, transform=tf_test)
    tr = torch.utils.data.DataLoader(cifar_tr, batch, shuffle=True, num_workers=4)
    id_te = torch.utils.data.DataLoader(cifar_te, batch, shuffle=False, num_workers=4)

    ood = OrderedDict()
    svhn = tv.datasets.SVHN(root, 'test', download=True, transform=tf_test)
    ood['SVHN'] = torch.utils.data.Subset(svhn, list(range(min(n_ood, len(svhn)))))
    c100 = tv.datasets.CIFAR100(root, False, download=True, transform=tf_test)
    ood['CIFAR-100'] = torch.utils.data.Subset(c100, list(range(min(n_ood, len(c100)))))

    # synthetic noise OOD (already normalized space)
    g = (torch.randn(n_ood, 3, 32, 32))
    ood['Gaussian'] = torch.utils.data.TensorDataset(g, torch.zeros(n_ood, dtype=torch.long))
    u = (torch.rand(n_ood, 3, 32, 32) - 0.5) / 0.25
    ood['Uniform'] = torch.utils.data.TensorDataset(u, torch.zeros(n_ood, dtype=torch.long))

    ood_loaders = {k: torch.utils.data.DataLoader(v, batch, shuffle=False, num_workers=2)
                   for k, v in ood.items()}
    return tr, id_te, ood_loaders


# ───────────────────────── train ─────────────────────────

def train(model, tr, epochs):
    opt = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        sched.step()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f'  ep {ep + 1}/{epochs}')
    model.eval()


# ───────────────────────── scoring ─────────────────────────

class Scorer:
    """One forward pass per batch -> logit-based + entropy-based anomaly scores."""

    def __init__(self, model, layers):
        self.model = model; self.layers = layers
        self.f = OrderedDict(); self._h = []
        mods = dict(model.named_modules())
        for n in layers:
            self._h.append(mods[n].register_forward_hook(self._hook(n)))

    def _hook(self, n):
        def fn(m, i, o):
            self.f[n] = o.detach()
        return fn

    @staticmethod
    def _ent(feats):  # [B,C,H,W] -> [B] mean spatial normalized channel entropy
        p = F.softmax(feats, dim=1); C = feats.shape[1]
        H = -(p * torch.log(p + 1e-8)).sum(1) / np.log(C)   # [B,H,W]
        return H

    @torch.no_grad()
    def score_batch(self, x):
        self.f.clear()
        logits = self.model(x)                                  # triggers hooks
        sm = F.softmax(logits, dim=1)
        out = {}
        out['MSP'] = (-sm.max(1).values)
        out['Energy'] = (-torch.logsumexp(logits, dim=1))
        out['MaxLogit'] = (-logits.max(1).values)

        ents = OrderedDict((n, self._ent(self.f[n])) for n in self.layers)  # each [B,Hn,Wn]
        last = self.layers[-1]
        out['ELHAM-Hlast'] = ents[last].mean(dim=(1, 2))
        out['ELHAM-Hall'] = torch.stack([e.mean(dim=(1, 2)) for e in ents.values()], 0).mean(0)
        out['ELHAM-Hmin'] = 1.0 - torch.stack([e.amin(dim=(1, 2)) for e in ents.values()], 0).min(0).values
        # info-gain: mean entropy reduction across consecutive blocks (scalar per image)
        ig = torch.zeros(x.shape[0], device=x.device)
        prev = None
        for n in self.layers:
            if prev is not None:
                Hp, Hc = ents[prev], ents[n]
                if Hp.shape[1:] != Hc.shape[1:]:
                    Hp = F.interpolate(Hp[:, None], size=Hc.shape[1:], mode='bilinear',
                                       align_corners=False)[:, 0]
                ig = ig + torch.clamp(Hp - Hc, min=0).mean(dim=(1, 2))
            prev = n
        out['ELHAM-IG'] = (-ig)   # less info-gain -> more OOD (hypothesis)
        return {k: v.cpu().numpy() for k, v in out.items()}

    def score_loader(self, loader):
        accs = None
        for x, _ in loader:
            x = x.to(DEVICE)
            s = self.score_batch(x)
            if accs is None:
                accs = {k: [] for k in s}
            for k, v in s.items():
                accs[k].append(v)
        return {k: np.concatenate(v) for k, v in accs.items()}

    def remove(self):
        for h in self._h:
            h.remove()


def fpr_at_95(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr, 0.95)
    idx = min(idx, len(fpr) - 1)
    return float(fpr[idx])


# ───────────────────────── main ─────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='./data_cache')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--n-ood', type=int, default=2000)
    ap.add_argument('--out', default='elham_ood')
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f'Device: {DEVICE}')
    tr, id_te, ood_loaders = loaders(args.root, args.n_ood)

    model = CIFARResNet18().to(DEVICE)
    print(f'Training CIFAR-10 ResNet-18 ({args.epochs} ep)...')
    train(model, tr, args.epochs)

    # ID test accuracy (sanity)
    correct = total = 0
    with torch.no_grad():
        for x, y in id_te:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item(); total += y.numel()
    print(f'  CIFAR-10 test acc = {correct / total:.3f}\n')

    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    scorer = Scorer(model, layers)

    print('Scoring ID (CIFAR-10 test)...')
    id_scores = scorer.score_loader(id_te)
    n_id = min(len(next(iter(id_scores.values()))), args.n_ood)
    methods = list(id_scores.keys())

    results = {m: {} for m in methods}
    for oname, ol in ood_loaders.items():
        print(f'Scoring OOD: {oname}...')
        os_ = scorer.score_loader(ol)
        for m in methods:
            s_id = id_scores[m][:n_id]; s_ood = os_[m][:n_id]
            labels = np.concatenate([np.zeros(len(s_id)), np.ones(len(s_ood))])  # 1=OOD
            scores = np.concatenate([s_id, s_ood])
            auroc = roc_auc_score(labels, scores)
            results[m][oname] = {'auroc': float(auroc), 'fpr95': fpr_at_95(labels, scores)}
    scorer.remove()

    # ── report ──
    ood_names = list(ood_loaders.keys())
    print(f'\n{"=" * 84}\n  OOD DETECTION  (ID=CIFAR-10, AUROC ↑ / FPR@95 ↓)\n{"=" * 84}')
    hdr = f'  {"Method":<14s}' + ''.join(f'{o[:11]:>13s}' for o in ood_names) + f'{"mean AUROC":>13s}'
    print(hdr); print('  ' + '-' * (len(hdr)))
    is_elham = lambda m: m.startswith('ELHAM')
    summary = {}
    for m in methods:
        aurocs = [results[m][o]['auroc'] for o in ood_names]
        cells = ''.join(f'{a:>13.3f}' for a in aurocs)
        mark = '  *' if is_elham(m) else ''
        print(f'  {m:<14s}{cells}{np.mean(aurocs):>13.3f}{mark}')
        summary[m] = {'mean_auroc': float(np.mean(aurocs)),
                      'per_ood': results[m]}

    # verdict
    print(f'\n{"=" * 84}\n  VERDICT: does channel entropy match the cheap forward baselines?\n{"=" * 84}')
    msp = np.mean([results['MSP'][o]['auroc'] for o in ood_names])
    energy = np.mean([results['Energy'][o]['auroc'] for o in ood_names])
    print(f'    baselines: MSP={msp:.3f}  Energy={energy:.3f}')
    best_elham = max([m for m in methods if is_elham(m)],
                     key=lambda m: np.mean([results[m][o]['auroc'] for o in ood_names]))
    be = np.mean([results[best_elham][o]['auroc'] for o in ood_names])
    print(f'    best ELHAM: {best_elham}={be:.3f}')
    if be >= max(msp, energy) - 0.005:
        print(f'    -> PROMISING: entropy matches/beats cheap baselines. Pivot has legs.')
    elif be >= msp - 0.02:
        print(f'    -> MARGINAL: near MSP. Worth one more setup before deciding.')
    else:
        print(f'    -> ENTROPY UNDERPERFORMS the cheap baselines. Pivot weak; lean negative paper.')

    with open(args.out + '.json', 'w') as f:
        json.dump({'id_acc': correct / total, 'summary': summary}, f, indent=2)
    _plot(results, methods, ood_names, args.out)
    print(f'\n  Saved: {args.out}.json, {args.out}.png')


def _plot(results, methods, ood_names, out):
    cmap = plt.get_cmap('tab10')
    x = np.arange(len(ood_names)); w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for j, m in enumerate(methods):
        vals = [results[m][o]['auroc'] for o in ood_names]
        ax.bar(x + j * w - 0.4 + w / 2, vals, w, label=m, color=cmap(j % 10),
               edgecolor='white', linewidth=0.4)
    ax.axhline(0.5, color='k', ls='--', alpha=0.3)
    ax.set_xticks(x); ax.set_xticklabels(ood_names)
    ax.set_ylabel('AUROC ↑'); ax.set_ylim(0.4, 1.0)
    ax.set_title('ELHAM channel entropy vs cheap forward baselines for OOD (ID=CIFAR-10)',
                 fontweight='bold')
    ax.legend(fontsize=8, ncol=4, loc='lower center')
    plt.tight_layout(); plt.savefig(out + '.png', dpi=170, bbox_inches='tight'); plt.close()


if __name__ == '__main__':
    main()
