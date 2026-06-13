"""
ELHAM-for-OOD, decisive follow-up
=================================
The pilot (eval_ood.py) showed channel entropy BEATS MSP/Energy on CIFAR-10->SVHN
(0.966) but (a) fails on noise with naive aggregation and (b) is weak on near-OOD.
This script tests whether the signal is real and robust:

  1. ELHAM-Fused: standardize each entropy statistic {Hall,Hlast,Hmin,IG} on ID-train
     (no OOD data) and take MAX |z| -> flags OOD if ANY entropy stat is atypical.
     Goal: robust across far-OOD AND noise.
  2. Near vs far taxonomy + 2nd ID set (CIFAR-10 and CIFAR-100 as ID).
  3. Strong baselines: ReAct + KNN (SOTA-competitive), plus MSP/Energy/MaxLogit.

OOD per ID set:  SVHN (far), DTD textures (far), the other CIFAR (near),
                 Gaussian, Uniform (noise).

Metrics: AUROC, FPR@95. Grouped report: FAR / NEAR / NOISE means.
Usage: python eval_ood2.py [--epochs 30] [--n-ood 2000] [--id cifar10,cifar100]
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

from eval_ood import CIFARResNet18  # reuse model

DEVICE = torch.device('cuda' if torch.cuda.is_available()
                      else ('mps' if torch.backends.mps.is_available() else 'cpu'))
SEED = 0
MEAN = (0.4914, 0.4822, 0.4465); STD = (0.2470, 0.2435, 0.2616)

OOD_KIND = {'SVHN': 'far', 'DTD': 'far', 'CIFAR-100': 'near', 'CIFAR-10': 'near',
            'Gaussian': 'noise', 'Uniform': 'noise'}


# ───────────────────────── data ─────────────────────────

def _tf_train():
    return transforms.Compose([transforms.RandomCrop(32, padding=4),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def _tf_test():
    return transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


def id_data(name, root):
    cls = tv.datasets.CIFAR10 if name == 'cifar10' else tv.datasets.CIFAR100
    tr = cls(root, True, download=True, transform=_tf_train())
    te = cls(root, False, download=True, transform=_tf_test())
    nc = 10 if name == 'cifar10' else 100
    return tr, te, nc


def ood_sets(id_name, root, n):
    sets = OrderedDict()
    svhn = tv.datasets.SVHN(root, 'test', download=True, transform=_tf_test())
    sets['SVHN'] = torch.utils.data.Subset(svhn, list(range(min(n, len(svhn)))))
    try:
        dtd = tv.datasets.DTD(root, split='test', download=True, transform=_tf_test())
        sets['DTD'] = torch.utils.data.Subset(dtd, list(range(min(n, len(dtd)))))
    except Exception as e:
        print(f'  (DTD unavailable: {e})')
    # near-OOD = the other CIFAR
    if id_name == 'cifar10':
        other = tv.datasets.CIFAR100(root, False, download=True, transform=_tf_test()); oname = 'CIFAR-100'
    else:
        other = tv.datasets.CIFAR10(root, False, download=True, transform=_tf_test()); oname = 'CIFAR-10'
    sets[oname] = torch.utils.data.Subset(other, list(range(min(n, len(other)))))
    g = torch.randn(n, 3, 32, 32)
    sets['Gaussian'] = torch.utils.data.TensorDataset(g, torch.zeros(n, dtype=torch.long))
    u = (torch.rand(n, 3, 32, 32) - 0.5) / 0.25
    sets['Uniform'] = torch.utils.data.TensorDataset(u, torch.zeros(n, dtype=torch.long))
    return sets


# ───────────────────────── train ─────────────────────────

def train(model, tr, epochs):
    opt = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.CrossEntropyLoss()
    ld = torch.utils.data.DataLoader(tr, 256, shuffle=True, num_workers=4)
    for ep in range(epochs):
        model.train()
        for x, y in ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        sched.step()
        if ep % 6 == 0 or ep == epochs - 1:
            print(f'    ep {ep + 1}/{epochs}')
    model.eval()


# ───────────────────────── scoring ─────────────────────────

class OODScorer:
    LAYERS = ['layer1', 'layer2', 'layer3', 'layer4']

    def __init__(self, model):
        self.model = model
        self.f = OrderedDict(); self._h = []
        mods = dict(model.named_modules())
        for n in self.LAYERS:
            self._h.append(mods[n].register_forward_hook(self._hook(n)))
        # fitted on ID train:
        self.ent_mu = self.ent_sd = None         # entropy-agg standardization
        self.bank = None                         # KNN feature bank (normalized)
        self.react_c = None                      # ReAct clip threshold

    def _hook(self, n):
        def fn(m, i, o):
            self.f[n] = o.detach()
        return fn

    @staticmethod
    def _ent(feats):
        p = F.softmax(feats, dim=1); C = feats.shape[1]
        return -(p * torch.log(p + 1e-8)).sum(1) / np.log(C)   # [B,H,W]

    @torch.no_grad()
    def raw(self, x):
        """One forward pass -> logits, pooled feat [B,512], entropy aggs dict[B]."""
        self.f.clear()
        logits = self.model(x)
        feat = F.adaptive_avg_pool2d(self.f['layer4'], 1).flatten(1)   # [B,512]
        ents = OrderedDict((n, self._ent(self.f[n])) for n in self.LAYERS)
        last = self.LAYERS[-1]
        aggs = OrderedDict()
        aggs['Hlast'] = ents[last].mean(dim=(1, 2))
        aggs['Hall'] = torch.stack([e.mean(dim=(1, 2)) for e in ents.values()], 0).mean(0)
        aggs['Hmin'] = 1.0 - torch.stack([e.amin(dim=(1, 2)) for e in ents.values()], 0).min(0).values
        ig = torch.zeros(x.shape[0], device=x.device); prev = None
        for n in self.LAYERS:
            if prev is not None:
                Hp, Hc = ents[prev], ents[n]
                if Hp.shape[1:] != Hc.shape[1:]:
                    Hp = F.interpolate(Hp[:, None], size=Hc.shape[1:], mode='bilinear',
                                       align_corners=False)[:, 0]
                ig = ig + torch.clamp(Hp - Hc, min=0).mean(dim=(1, 2))
            prev = n
        aggs['IG'] = -ig
        return logits, feat, aggs

    @torch.no_grad()
    def fit(self, tr, n_bank=10000):
        """Fit standardization (entropy aggs), KNN bank, ReAct threshold on ID train."""
        ld = torch.utils.data.DataLoader(tr, 256, shuffle=False, num_workers=4)
        agg_list = {k: [] for k in ['Hlast', 'Hall', 'Hmin', 'IG']}
        feats = []; seen = 0
        for x, _ in ld:
            x = x.to(DEVICE)
            _, feat, aggs = self.raw(x)
            for k in agg_list:
                agg_list[k].append(aggs[k].cpu())
            feats.append(feat.cpu()); seen += x.shape[0]
            if seen >= n_bank:
                break
        self.ent_mu = {k: torch.cat(v).mean().item() for k, v in agg_list.items()}
        self.ent_sd = {k: (torch.cat(v).std().item() + 1e-6) for k, v in agg_list.items()}
        bank = torch.cat(feats)[:n_bank].to(DEVICE)
        self.bank = F.normalize(bank, dim=1)
        self.react_c = torch.quantile(bank.flatten(), 0.90).item()

    @torch.no_grad()
    def score_batch(self, x):
        logits, feat, aggs = self.raw(x)
        out = {}
        # cheap logit baselines
        out['MSP'] = (-F.softmax(logits, 1).max(1).values)
        out['Energy'] = (-torch.logsumexp(logits, 1))
        out['MaxLogit'] = (-logits.max(1).values)
        # ReAct: clip penultimate feat at c, recompute energy
        fr = feat.clamp(max=self.react_c)
        logits_r = F.linear(fr, self.model.fc.weight, self.model.fc.bias)
        out['ReAct'] = (-torch.logsumexp(logits_r, 1))
        # KNN: distance to k-th nearest ID-train feature (cosine)
        fn = F.normalize(feat, dim=1)
        d = 1 - fn @ self.bank.T                       # [B, n_bank] cosine distance
        kth = torch.kthvalue(d, k=min(50, d.shape[1]), dim=1).values
        out['KNN'] = kth
        # ELHAM entropy aggregations
        for k in ['Hlast', 'Hall', 'Hmin', 'IG']:
            out['ELHAM-' + k] = aggs[k]
        # ELHAM-Fused: max |z| over standardized entropy aggs
        zs = torch.stack([(aggs[k] - self.ent_mu[k]) / self.ent_sd[k]
                          for k in ['Hlast', 'Hall', 'Hmin', 'IG']], 0).abs()
        out['ELHAM-Fused'] = zs.max(0).values
        return {k: v.cpu().numpy() for k, v in out.items()}

    @torch.no_grad()
    def score_loader(self, loader):
        acc = None
        for x, _ in loader:
            s = self.score_batch(x.to(DEVICE))
            if acc is None:
                acc = {k: [] for k in s}
            for k, v in s.items():
                acc[k].append(v)
        return {k: np.concatenate(v) for k, v in acc.items()}

    def remove(self):
        for h in self._h:
            h.remove()


def fpr95(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    i = min(np.searchsorted(tpr, 0.95), len(fpr) - 1)
    return float(fpr[i])


# ───────────────────────── main ─────────────────────────

def run_one_id(id_name, root, epochs, n_ood):
    print(f'\n########## ID = {id_name} ##########')
    tr, te, nc = id_data(id_name, root)
    model = CIFARResNet18(nc).to(DEVICE)
    print(f'  training ({epochs} ep)...'); train(model, tr, epochs)
    ld_te = torch.utils.data.DataLoader(te, 256, shuffle=False, num_workers=4)
    correct = total = 0
    with torch.no_grad():
        for x, y in ld_te:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item(); total += y.numel()
    print(f'  test acc = {correct / total:.3f}')

    scorer = OODScorer(model)
    print('  fitting ID-train stats (KNN bank, ReAct, entropy standardization)...')
    scorer.fit(tr)
    id_scores = scorer.score_loader(ld_te)
    methods = list(id_scores.keys())
    n_id = min(len(next(iter(id_scores.values()))), n_ood)

    res = {m: {} for m in methods}
    for oname, ods in ood_sets(id_name, root, n_ood).items():
        ol = torch.utils.data.DataLoader(ods, 256, shuffle=False, num_workers=2)
        osc = scorer.score_loader(ol)
        for m in methods:
            s_id = id_scores[m][:n_id]; s_o = osc[m][:n_id]
            lab = np.r_[np.zeros(len(s_id)), np.ones(len(s_o))]
            sc = np.r_[s_id, s_o]
            res[m][oname] = {'auroc': float(roc_auc_score(lab, sc)),
                             'fpr95': fpr95(lab, sc), 'kind': OOD_KIND.get(oname, '?')}
    scorer.remove()
    return {'acc': correct / total, 'methods': methods, 'res': res}


def report(id_name, R):
    methods, res = R['methods'], R['res']
    onames = list(res[methods[0]].keys())
    print(f'\n{"=" * 92}\n  ID={id_name}  acc={R["acc"]:.3f}   AUROC ↑\n{"=" * 92}')
    hdr = f'  {"Method":<14s}' + ''.join(f'{o[:10]:>11s}' for o in onames) \
        + f'{"FAR":>8s}{"NEAR":>8s}{"NOISE":>8s}'
    print(hdr); print('  ' + '-' * len(hdr))
    grouped = {}
    for m in methods:
        cells = ''.join(f'{res[m][o]["auroc"]:>11.3f}' for o in onames)
        gv = {}
        for kind in ['far', 'near', 'noise']:
            vals = [res[m][o]['auroc'] for o in onames if res[m][o]['kind'] == kind]
            gv[kind] = float(np.mean(vals)) if vals else float('nan')
        grouped[m] = gv
        mark = '  *' if m.startswith('ELHAM') else ''
        print(f'  {m:<14s}{cells}{gv["far"]:>8.3f}{gv["near"]:>8.3f}{gv["noise"]:>8.3f}{mark}')
    return grouped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='./data_cache')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--n-ood', type=int, default=2000)
    ap.add_argument('--id', default='cifar10,cifar100')
    ap.add_argument('--out', default='elham_ood2')
    args = ap.parse_args()
    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f'Device: {DEVICE}')

    allR = {}
    for id_name in args.id.split(','):
        R = run_one_id(id_name.strip(), args.root, args.epochs, args.n_ood)
        allR[id_name.strip()] = {'acc': R['acc'], 'grouped': report(id_name.strip(), R),
                                 'res': R['res']}

    # verdict: ELHAM-Fused vs strong baselines on FAR / NEAR, averaged across ID sets
    print(f'\n{"=" * 92}\n  VERDICT (mean over ID sets)\n{"=" * 92}')
    def mean_kind(method, kind):
        vs = [allR[i]['grouped'][method][kind] for i in allR if method in allR[i]['grouped']]
        return float(np.nanmean(vs)) if vs else float('nan')
    for kind in ['far', 'near', 'noise']:
        line = f'  {kind.upper():<6s}'
        for m in ['MSP', 'Energy', 'ReAct', 'KNN', 'ELHAM-Hall', 'ELHAM-Fused']:
            line += f'  {m}={mean_kind(m, kind):.3f}'
        print(line)
    print('\n  Read: does ELHAM-Fused stay robust across FAR/NEAR/NOISE, and does any')
    print('  ELHAM variant beat the STRONG baselines (ReAct/KNN) on FAR-OOD?')

    with open(args.out + '.json', 'w') as f:
        json.dump({i: {'acc': allR[i]['acc'], 'grouped': allR[i]['grouped'],
                       'res': allR[i]['res']} for i in allR}, f, indent=2)
    _plot(allR, args.out)
    print(f'\n  Saved: {args.out}.json, {args.out}.png')


def _plot(allR, out):
    ids = list(allR.keys())
    fig, axes = plt.subplots(1, len(ids), figsize=(11 * len(ids), 5.5), squeeze=False)
    cmap = plt.get_cmap('tab10')
    for ax, i in zip(axes[0], ids):
        res = allR[i]['res']; methods = list(res.keys()); onames = list(res[methods[0]].keys())
        x = np.arange(len(onames)); w = 0.8 / len(methods)
        for j, m in enumerate(methods):
            vals = [res[m][o]['auroc'] for o in onames]
            ax.bar(x + j * w - 0.4 + w / 2, vals, w, label=m, color=cmap(j % 10),
                   edgecolor='white', linewidth=0.3)
        ax.axhline(0.5, color='k', ls='--', alpha=0.3)
        ax.set_xticks(x); ax.set_xticklabels(onames, fontsize=8, rotation=20)
        ax.set_title(f'ID={i}', fontweight='bold'); ax.set_ylim(0, 1.0); ax.set_ylabel('AUROC')
        if i == ids[0]:
            ax.legend(fontsize=7, ncol=3, loc='lower left')
    plt.suptitle('ELHAM channel-entropy OOD vs strong baselines (ReAct/KNN)', fontweight='bold')
    plt.tight_layout(); plt.savefig(out + '.png', dpi=160, bbox_inches='tight'); plt.close()


if __name__ == '__main__':
    main()
