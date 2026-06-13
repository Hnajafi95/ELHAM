# ELHAM: Entropy-driven Latent Hierarchical Attribution Maps

[![Python](https://img.shields.io/badge/python-3.9+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**A fast, gradient-free, architecture-agnostic tool for inspecting the latent
representations of a neural network — in a single forward pass.**

```
E → Entropy-driven   softmax entropy over channels at each spatial location
L → Latent           operates on intermediate feature maps, across depth
H → Hierarchical     measures entropy reduction between consecutive layers
A → Attribution      spatial maps of where the representation becomes certain
M → Maps             per-layer + a combined multi-resolution map
```

ELHAM measures, at every spatial location and layer, how **decided** the channel
activation distribution is (its normalized Shannon entropy), and how that
decisiveness **sharpens with depth** (information gain). It attributes — spatially
— *where the network's internal representation becomes certain* for an input.

> **Honest scope (please read).** ELHAM's maps are **class-agnostic**: they show
> where the representation is *certain*, not what *causes* a given class. ELHAM is
> **not** a faithful class-causal attribution method and **not** a competitive OOD
> detector — we tested both rigorously against strong baselines and it loses to the
> right tool for each (see [VALIDATION.md](VALIDATION.md)). It *is* a useful
> representation-inspection diagnostic, especially for its **per-layer** view and
> for models where gradients are unavailable. We ship the evidence so you can trust
> the scope.

---

## Install

```bash
pip install torch torchvision numpy matplotlib   # matplotlib only for visualize()
# then drop elham.py into your project (single-file, no packaging needed)
```

## Quick start

```python
import torch, torchvision as tv
from elham import ELHAM, suggest_layers

model = tv.models.resnet50(weights="IMAGENET1K_V2").eval()
x = torch.randn(1, 3, 224, 224)                  # your preprocessed image

with ELHAM(model, suggest_layers(model)) as elham:
    out = elham.maps(x)
    combined  = out["combined"]                  # [H, W] multi-layer map
    per_layer = out["entropy"]                   # {layer_name: [H, W]} entropy maps
    info_gain = out["infogain"]                  # {layer_name: [H, W]} layer-to-layer gain
    elham.visualize(x, save="elham.png")         # input + per-layer + combined panel
```

Works the same on a Vision Transformer (token activations are reshaped to a grid):

```python
vit = tv.models.vit_b_16(weights="IMAGENET1K_V1").eval()
with ELHAM(vit, suggest_layers(vit)) as elham:
    maps = elham.maps(x)                         # identical API, no code changes
```

See [`examples/demo.py`](examples/demo.py) for a runnable end-to-end example.

## What the maps mean

| Output | Meaning |
|--------|---------|
| `entropy[layer]` | Normalized channel entropy ∈ [0,1] at each location. **Low = the layer has committed** to a peaked channel pattern there; high = undecided. |
| `infogain[layer]` | `max(0, H_prev − H_cur)`: where decisiveness **increased** between consecutive layers. |
| `combined` | Sum of per-layer info-gain, upsampled to input size — where the representation sharpened across multiple layers. |

**Interpretation:** these are maps of *representational certainty*, not causal
importance. A region can be highly "decided" without controlling the output.

## When ELHAM is the right tool

- **Per-layer representation inspection / model debugging.** Saliency methods give
  one map; ELHAM shows how feature decisiveness forms across depth — compare layers,
  epochs, or architectures.
- **Gradient-free settings.** Forward-pass only, so it runs on int8-quantized models,
  ONNX/TensorRT, and inference-only graphs where backprop methods can't.
- **Architecture-agnostic.** Identical code for CNNs and ViTs.
- **Cheap.** One forward pass, ~1 ms/image overhead.

## When to use something else

- Want to know **why the model predicted class X** (causal attribution)?
  → Grad-CAM, CAM, Integrated Gradients.
- Want **out-of-distribution detection**? → KNN, ReAct, Energy.

Full benchmarks and the reasoning behind these recommendations: [VALIDATION.md](VALIDATION.md).

## Repository layout

```
elham.py          the library (the tool)
examples/         runnable usage demo
validation/       benchmark scripts behind the scope claims (attribution, OOD)
VALIDATION.md     honest evaluation summary with numbers
ROADMAP.md        chronological log of the investigation
```

## Citation

```bibtex
@software{elham2026,
  title  = {ELHAM: Entropy-driven Latent Hierarchical Attribution Maps},
  author = {Najafi, Hamed},
  year   = {2026},
  url    = {https://github.com/Hnajafi95/ELHAM},
  note   = {Forward-only latent-representation diagnostic (gradient-free,
            architecture-agnostic, per-layer). See VALIDATION.md for scope.}
}
```

## License

MIT — see [LICENSE](LICENSE).
