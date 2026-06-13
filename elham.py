"""
ELHAM — Entropy-driven Latent Hierarchical Attribution Maps
===========================================================
A fast, gradient-free, architecture-agnostic tool for *inspecting the latent
representations* of a neural network. In a single forward pass it measures, at
every spatial location and layer, how "decided" the channel activation
distribution is (its normalized Shannon entropy), and how that decisiveness
sharpens from layer to layer (information gain). ELHAM *attributes* — spatially
— where the network's internal representation becomes certain.

What ELHAM is
-------------
A representation / feature-formation **diagnostic**. It answers:
  "Where, and at which depth, has the network committed to decisive channel
   features for this input?"

What ELHAM is NOT  (validated — see validation/ and VALIDATION.md)
------------------------------------------------------------------
* NOT *class-causal* attribution. ELHAM maps are CLASS-AGNOSTIC (they do not use
  the target class), so they do not explain "why class X" and should not be read
  as causal importance. For class-causal attribution use Grad-CAM / CAM / IG.
* NOT a competitive OOD detector. For OOD use KNN / ReAct / Energy.
We tested both uses rigorously (validation/); ELHAM loses to the right tool for
each. It is a complementary *representation-inspection* signal, honestly scoped.

Why it can still be useful
--------------------------
* Per-LAYER maps: see how feature decisiveness evolves with depth (gradient
  saliency methods give a single map). Useful for debugging / understanding.
* Gradient-free, single forward pass: runs where backprop can't — int8-quantized
  models, ONNX/TensorRT, inference-only graphs.
* Architecture-agnostic: identical code for CNNs (BCHW) and Vision Transformers
  (token BND, reshaped to a grid).
* Light: just torch + numpy (matplotlib only for the optional visualizer).

Quick start
-----------
    import torchvision as tv, torch
    from elham import ELHAM, suggest_layers

    model = tv.models.resnet50(weights="IMAGENET1K_V2").eval()
    with ELHAM(model, suggest_layers(model)) as elham:
        out = elham.maps(image_tensor)          # image_tensor: [1,3,H,W]
        combined = out["combined"]              # [H,W] multi-layer map
        per_layer = out["entropy"]              # {layer_name: [H,W]}
        elham.visualize(image_tensor, save="elham.png")
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["ELHAM", "suggest_layers"]


def suggest_layers(model):
    """Best-effort default tap points for common torchvision backbones.

    Returns a list of module names suitable for `ELHAM(model, layers)`. Falls
    back to the four deepest leaf modules with >=2D spatial/token output if the
    architecture is unrecognised. You can always pass your own list.
    """
    names = dict(model.named_modules())
    for cand in (["layer1", "layer2", "layer3", "layer4"],               # ResNet/ConvNeXt-ish
                 ["features.2", "features.4", "features.6", "features.8"]):
        if all(n in names for n in cand):
            return cand
    # ViT (torchvision): encoder blocks
    enc = [n for n in names if n.startswith("encoder.layers.encoder_layer_")]
    if enc:
        enc = sorted(enc, key=lambda s: int(s.split("_")[-1].split(".")[0])
                     if s.split("_")[-1].split(".")[0].isdigit() else 0)
        # take 4 evenly spaced blocks
        idx = np.linspace(0, len(enc) - 1, 4).round().astype(int)
        uniq = []
        for i in idx:
            if enc[i] not in uniq:
                uniq.append(enc[i])
        return uniq
    raise ValueError("Could not auto-detect layers for this model; pass `layers=` "
                     "explicitly (a list of module names from model.named_modules()).")


class ELHAM:
    """Entropy-driven Latent Hierarchical Activation Maps.

    Parameters
    ----------
    model : torch.nn.Module (already in eval mode for stable BatchNorm)
    layers : list[str]
        Module names (from `model.named_modules()`) to tap, in forward order.
    normalize : bool
        Divide each entropy map by log(C) so layers with different channel counts
        are comparable in [0, 1]. Recommended True.
    eps : float
        Numerical floor inside the log.
    """

    def __init__(self, model, layers, normalize=True, eps=1e-8):
        if not layers:
            raise ValueError("`layers` must be a non-empty list of module names.")
        self.model = model
        self.layers = list(layers)
        self.normalize = normalize
        self.eps = eps
        self._acts = OrderedDict()
        self._handles = []
        mods = dict(model.named_modules())
        for n in self.layers:
            if n not in mods:
                raise KeyError(f"layer '{n}' not found in model.named_modules()")
            self._handles.append(mods[n].register_forward_hook(self._hook(n)))

    # ----- hooks -----
    def _hook(self, name):
        def fn(module, inp, out):
            self._acts[name] = out.detach()
        return fn

    # ----- core math -----
    def _to_bchw(self, act):
        """Normalise an activation to [B, C, H, W].

        * 4D BCHW            -> unchanged (CNN feature map).
        * 3D BND tokens (ViT) -> drop a leading CLS token if present, reshape the
          N patch tokens to a square grid, channels = embedding dim D.
        """
        if act.dim() == 4:
            return act
        if act.dim() == 3:
            B, N, D = act.shape
            g = int(round(N ** 0.5))
            if g * g != N and (g := int(round((N - 1) ** 0.5))) * g == N - 1:
                act = act[:, 1:, :]          # drop CLS
                N = N - 1
            g = int(round(N ** 0.5))
            if g * g != N:
                raise ValueError(f"token count {N} is not a square grid; cannot "
                                 f"reshape this layer to spatial. Tap a different layer.")
            return act.transpose(1, 2).reshape(B, D, g, g)
        raise ValueError(f"unsupported activation rank {act.dim()} (expected 3 or 4).")

    def _channel_entropy(self, act):
        """Per-location normalized Shannon entropy of the channel softmax. -> [B,H,W]"""
        bchw = self._to_bchw(act)
        C = bchw.shape[1]
        p = F.softmax(bchw, dim=1)
        H = -(p * torch.log(p + self.eps)).sum(dim=1)            # [B,H,W]
        if self.normalize and C > 1:
            H = H / float(np.log(C))
        return H

    @torch.no_grad()
    def _forward(self, x):
        self._acts.clear()
        _ = self.model(x)
        if len(self._acts) != len(self.layers):
            missing = [n for n in self.layers if n not in self._acts]
            raise RuntimeError(f"layers did not fire on this input: {missing}")
        return OrderedDict((n, self._acts[n]) for n in self.layers)

    # ----- public API -----
    @torch.no_grad()
    def maps(self, x):
        """Compute ELHAM maps for input batch `x` ([B,3,H,W] or [3,H,W]).

        Returns a dict (numpy arrays, batch squeezed if B==1):
          'entropy'  : {layer: entropy map}            normalized channel entropy
          'infogain' : {layer: info-gain map}          max(0, H_prev - H_cur)
          'combined' : single map = sum of upsampled info-gain across layers
        Higher 'combined' = the network sharpened its channel commitment there
        across multiple layers. (Representational certainty, NOT causal importance.)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        H0, W0 = x.shape[2], x.shape[3]
        feats = self._forward(x)

        ent = OrderedDict((n, self._channel_entropy(a)) for n, a in feats.items())
        infogain = OrderedDict()
        combined = torch.zeros(x.shape[0], H0, W0, device=x.device)
        prev = None
        for n in self.layers:
            if prev is not None:
                Hp, Hc = ent[prev], ent[n]
                if Hp.shape[1:] != Hc.shape[1:]:
                    Hp = F.interpolate(Hp.unsqueeze(1), size=Hc.shape[1:],
                                       mode="bilinear", align_corners=False).squeeze(1)
                gain = torch.clamp(Hp - Hc, min=0.0)
                infogain[n] = gain
                combined += F.interpolate(gain.unsqueeze(1), size=(H0, W0),
                                          mode="bilinear", align_corners=False).squeeze(1)
            prev = n

        def to_np(t):
            t = t.cpu().numpy()
            return t[0] if t.shape[0] == 1 else t

        return {
            "entropy": {n: to_np(v) for n, v in ent.items()},
            "infogain": {n: to_np(v) for n, v in infogain.items()},
            "combined": to_np(combined),
        }

    @torch.no_grad()
    def explain(self, x):
        """Convenience: return only the combined multi-layer map (numpy)."""
        return self.maps(x)["combined"]

    def visualize(self, x, image=None, save=None, cmap="inferno", show_input=True):
        """Render per-layer entropy maps + the combined map as a matplotlib figure.

        `image` (optional [3,H,W] in [0,1]) is shown as the input panel; otherwise
        `x` is de-normalised crudely for display. Returns the matplotlib Figure.
        """
        import matplotlib.pyplot as plt
        out = self.maps(x)
        layers = list(out["entropy"].keys())
        n_panels = (1 if show_input else 0) + len(layers) + 1
        cols = min(n_panels, 4)
        rows = int(np.ceil(n_panels / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.atleast_1d(axes).ravel()
        i = 0
        if show_input:
            img = image
            if img is None:
                img = x[0] if x.dim() == 4 else x
                img = img.detach().cpu()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            axes[i].imshow(img.permute(1, 2, 0).numpy() if hasattr(img, "permute")
                           else np.transpose(img, (1, 2, 0)))
            axes[i].set_title("input"); axes[i].axis("off"); i += 1
        for n in layers:
            axes[i].imshow(out["entropy"][n], cmap=cmap)
            axes[i].set_title(f"entropy: {n}"); axes[i].axis("off"); i += 1
        axes[i].imshow(out["combined"], cmap=cmap)
        axes[i].set_title("combined info-gain"); axes[i].axis("off"); i += 1
        for j in range(i, len(axes)):
            axes[j].axis("off")
        fig.suptitle("ELHAM — latent representation maps (forward-only, gradient-free)",
                     fontweight="bold")
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=160, bbox_inches="tight")
        return fig

    # ----- lifecycle -----
    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()

    def __del__(self):
        try:
            self.remove()
        except Exception:
            pass
