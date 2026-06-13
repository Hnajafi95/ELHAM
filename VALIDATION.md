# ELHAM — Validated Scope (what it does and doesn't do)

This document is an honest record of what ELHAM was tested for and how it
performed against strong, standard baselines. The scripts that produced these
numbers live in [`validation/`](validation/). We publish the negative results
alongside the positive ones on purpose: a tool you can trust is one whose limits
are stated plainly.

**TL;DR.** ELHAM is a useful *latent-representation diagnostic* (forward-only,
gradient-free, architecture-agnostic, per-layer). It is **not** a faithful
class-causal attribution method and **not** a competitive OOD detector — use the
right tool for those (Grad-CAM/CAM/IG; KNN/ReAct/Energy). ELHAM's maps are
**class-agnostic**: they show where the network's *representation becomes
certain*, not what *causes* a particular class.

---

## 1. Is ELHAM a faithful attribution method?  → No.

**Structural fact.** ELHAM never uses the target class — `explain(x)` is identical
whatever class you ask about. A class-sensitivity check is therefore trivially
failed: true-vs-wrong-class map correlation = **1.00**. A faithful attribution
must be class-discriminative.

**Synthetic ground truth** (`validation/eval_classcond.py`, known causal regions):
gradient-free CAM localizes the causal region (AP ≈ 0.58–0.69); ELHAM's
information-gain map does not (AP ≈ 0.19–0.27). Making ELHAM class-conditional
(fusing a CAM prior) only *reproduces* CAM — it adds nothing.

**Real images** (`validation/eval_imagenet_gt.py`, ResNet-50, n=200, 95% CIs):

| Method | Insertion AUC ↑ | Deletion AUC ↓ | fwd passes |
|---|---|---|---|
| CAM (grad-free) ≡ Grad-CAM | **0.389** | 0.178 | 1 |
| ELHAM-C (class-conditional) | 0.388 | 0.178 | 1 |
| Score-CAM / Ablation-CAM | 0.383 / 0.385 | 0.187 | 2048 |
| ELHAM (info-gain) | 0.305 | 0.161* | 1 |

\* ELHAM's lone "win" (deletion) co-occurs with its *worst* insertion and a
class-agnostic map — the signature of an edge/high-frequency artifact, not
class-causal importance.

**Takeaway:** for attribution, gradient-free **CAM** already does everything ELHAM
aimed for (class-discriminative, 1 forward pass, gradient-free) and localizes
better. Even the expensive Score-CAM/Ablation-CAM don't beat plain CAM here.

## 2. Is ELHAM a competitive OOD detector?  → No.

`validation/eval_ood.py`, `validation/eval_ood2.py` (CIFAR-10 & CIFAR-100 as ID;
SVHN/DTD far, the other CIFAR near, Gaussian/Uniform noise; strong baselines
ReAct, KNN). Mean AUROC over ID sets:

| | FAR | NEAR | NOISE |
|---|---|---|---|
| KNN | **0.902** | 0.777 | **0.998** |
| ReAct | 0.869 | 0.797 | 0.897 |
| Energy / MSP | 0.833 / 0.812 | **0.811** / 0.800 | 0.689 / 0.735 |
| ELHAM (best variant) | 0.768 | 0.735 | (Fused) 0.996 |

A standout — ELHAM channel entropy beats MSP/Energy on **CIFAR-10 → SVHN (0.97)** —
did **not generalize** (DTD 0.76; vanishes with CIFAR-100 as ID). Every single
entropy aggregation has a catastrophic failure mode (e.g. naive aggregation
scores noise at AUROC 0.00). KNN/ReAct dominate.

**Takeaway:** channel entropy is not a competitive OOD score.

---

## 3. What ELHAM *is* good for

- **Per-layer representation inspection.** Unlike saliency methods (one map),
  ELHAM shows how channel decisiveness evolves with depth — useful for
  understanding/debugging how features form across a network.
- **Runs where gradients can't.** Forward-pass only: int8-quantized models,
  ONNX/TensorRT, inference-only graphs.
- **Architecture-agnostic.** Identical code for CNNs (BCHW) and ViTs (tokens → grid).
- **Cheap.** One forward pass; ~1 ms/image overhead.

Treat it as a complementary *representation-inspection* signal, not a replacement
for attribution or OOD methods.

---

## Reproduce

```bash
cd validation
python eval_classcond.py          # synthetic ground-truth (attribution)
python eval_imagenet_gt.py --dataset imagenette --n 200   # real-image insertion/deletion
python eval_ood2.py --epochs 30   # OOD vs ReAct/KNN
```
Full chronological log of the investigation is in [`ROADMAP.md`](ROADMAP.md).
