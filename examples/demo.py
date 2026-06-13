"""
ELHAM demo — visualize latent-representation maps on a real image.

Runs a pretrained ResNet-50 (and optionally a ViT) and saves a panel showing the
input, per-layer channel-entropy maps, and the combined multi-layer info-gain map.

Usage:
    python examples/demo.py --image path/to/img.jpg
    python examples/demo.py                      # uses a torchvision sample if no image
    python examples/demo.py --model vit_b_16 --image img.jpg
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torchvision as tv
from torchvision import transforms
from PIL import Image

from elham import ELHAM, suggest_layers

WEIGHTS = {
    "resnet50": tv.models.ResNet50_Weights.IMAGENET1K_V2,
    "vit_b_16": tv.models.ViT_B_16_Weights.IMAGENET1K_V1,
}


def load_image(path):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if path and os.path.exists(path):
        pil = Image.open(path).convert("RGB")
    else:
        print("No --image given or not found; using a synthetic test pattern.")
        import numpy as np
        arr = (np.random.rand(224, 224, 3) * 255).astype("uint8")
        pil = Image.fromarray(arr)
    x = tf(pil).unsqueeze(0)
    disp = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])(pil)
    return x, disp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50", choices=list(WEIGHTS))
    ap.add_argument("--image", default=None)
    ap.add_argument("--out", default="elham_demo.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(tv.models, args.model)(weights=WEIGHTS[args.model]).eval().to(device)
    x, disp = load_image(args.image)
    x = x.to(device)

    with ELHAM(model, suggest_layers(model)) as elham:
        out = elham.maps(x)
        with torch.no_grad():
            pred = int(model(x).argmax(1).item())
        print(f"model={args.model}  top-1 class index={pred}")
        for name, m in out["entropy"].items():
            print(f"  entropy[{name}]: mean={m.mean():.3f}  range=[{m.min():.3f},{m.max():.3f}]")
        elham.visualize(x, image=disp, save=args.out)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
