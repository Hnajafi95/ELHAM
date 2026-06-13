"""
ELHAM on CLIP: Multi-Modal Attribution
========================================
Tests ELHAM on CLIP's vision encoder — a domain where gradient methods
require architecture-specific wrappers but ELHAM works unchanged.

Key question: Does ELHAM's visual feature certainty correlate with
semantic relevance to text prompts?

Honest framing: ELHAM measures what the vision encoder finds visually
certain, NOT what matches a specific text. This is an intrinsic property
of the vision encoder — useful for debugging, limited for text-conditioned XAI.

Usage: pip install transformers pillow
       python eval_clip.py
"""

import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms
import numpy as np
from collections import OrderedDict
import time, os, hashlib, urllib.request, io
from PIL import Image
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    def explain(self, image):
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
# CLIP Wrapper
# ═══════════════════════════════════════════════════════════════════════════

class CLIPVisionWrapper:
    """Wraps CLIP vision encoder so ELHAM can hook transformer blocks."""
    def __init__(self, vision_model):
        self.vision = vision_model
    def forward(self, x):
        return self.vision(x)[0]  # return pooler_output
    def __call__(self, x): return self.forward(x)
    def named_modules(self):
        return self.vision.named_modules()
    def eval(self): self.vision.eval(); return self
    def to(self, d): self.vision.to(d); return self


def find_vit_layers(model):
    """Find encoder blocks in CLIP's ViT model."""
    blocks = []
    for n, m in model.named_modules():
        # CLIP ViT uses 'encoder.layers.X' pattern (same as torchvision ViT)
        if 'encoder.layers.' in n:
            parts = n.split('.')
            # Match 'encoder.layers.N' where N is a digit
            if len(parts) >= 3 and parts[-1].isdigit():
                blocks.append(n)
    if not blocks:
        # Try alternative: 'vision_model.encoder.layers.X'
        for n, m in model.named_modules():
            if 'encoder.layers.' in n:
                parts = n.split('.')
                if len(parts) >= 2 and parts[-1].isdigit():
                    blocks.append(n)
    blocks = sorted(set(blocks), key=lambda x: int(x.split('.')[-1]))
    if len(blocks) >= 4:
        idxs = [0, len(blocks)//3, 2*len(blocks)//3, len(blocks)-1]
        return [blocks[i] for i in idxs]
    return blocks[:4] if len(blocks) >= 4 else blocks


# ═══════════════════════════════════════════════════════════════════════════
# Image Loading
# ═══════════════════════════════════════════════════════════════════════════

CLIP_PREPROCESS = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466,0.4578275,0.40821073],
                         std=[0.26862954,0.26130258,0.27577711]),
])


def load_images(n=10):
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
        try: images.append(CLIP_PREPROCESS(Image.open(fname).convert('RGB')).unsqueeze(0))
        except: continue
    return torch.cat(images, dim=0) if images else None


# ═══════════════════════════════════════════════════════════════════════════
# Metrics (adapted for cosine similarity instead of class probability)
# ═══════════════════════════════════════════════════════════════════════════

def _gaussian_blur(image, ks=15, sigma=5):
    C = image.shape[1]
    x = torch.arange(ks, dtype=torch.float32, device=image.device)-ks//2
    g = torch.exp(-x**2/(2*sigma**2)); g /= g.sum()
    kh = g.view(1,1,1,-1).repeat(C,1,1,1); kv = g.view(1,1,-1,1).repeat(C,1,1,1)
    pad = ks//2
    out = F.conv2d(F.pad(image,(pad,pad,0,0),mode='reflect'),kh,groups=C)
    return F.conv2d(F.pad(out,(0,0,pad,pad),mode='reflect'),kv,groups=C)


def cosine_insertion_auc(vision_model, text_embedding, image, attr_map, steps=15):
    """Insert pixels by importance, measure cosine similarity recovery."""
    H,W = image.shape[2],image.shape[3]
    blurred = _gaussian_blur(image)
    order = np.argsort(attr_map.flatten())[::-1].copy()
    n_px = H*W; ps = max(1,n_px//steps); scores = []
    with torch.no_grad():
        for s in range(steps+1):
            mask = torch.zeros(n_px, device=DEVICE); n_ins = s*ps
            if n_ins>0: mask[torch.from_numpy(order[:n_ins].copy())]=1
            mask = mask.view(H,W).unsqueeze(0).unsqueeze(0)
            img_emb = vision_model(mask*image+(1-mask)*blurred)[0]
            sim = F.cosine_similarity(img_emb, text_embedding, dim=-1).item()
            scores.append(sim)
    return np.trapz(scores)/steps


def cosine_deletion_auc(vision_model, text_embedding, image, attr_map, steps=15):
    """Remove pixels by importance, measure cosine similarity drop."""
    H,W = image.shape[2],image.shape[3]
    order = np.argsort(attr_map.flatten())[::-1].copy()
    n_px = H*W; ps = max(1,n_px//steps); scores = []
    with torch.no_grad():
        for s in range(steps+1):
            mask = torch.ones(n_px, device=DEVICE); n_rem = s*ps
            if n_rem>0: mask[torch.from_numpy(order[:n_rem].copy())]=0
            mask = mask.view(H,W).unsqueeze(0).unsqueeze(0)
            img_emb = vision_model(mask*image)[0]
            sim = F.cosine_similarity(img_emb, text_embedding, dim=-1).item()
            scores.append(sim)
    return np.trapz(scores)/steps


# ═══════════════════════════════════════════════════════════════════════════
# Gradient Baseline: Input-gradient of cosine similarity
# ═══════════════════════════════════════════════════════════════════════════

def clip_gradient_map(vision_model, text_embedding, image):
    """Input gradient of cosine similarity w.r.t. image — naive baseline."""
    img = image.clone().detach().requires_grad_(True)
    img_emb = vision_model(img)[0]
    sim = F.cosine_similarity(img_emb, text_embedding, dim=-1)
    sim.backward()
    return img.grad.abs().max(dim=1)[0].squeeze(0).detach().cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════
# Main Test
# ═══════════════════════════════════════════════════════════════════════════

def run_clip_test():
    print(f'Device: {DEVICE}')
    print('Loading CLIP model...')

    try:
        from transformers import CLIPProcessor, CLIPModel
        clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch16').to(DEVICE).eval()
        processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')
        vision_model = clip_model.vision_model
        text_model = clip_model.text_model
        # Need the projection layers from the full model
        visual_projection = clip_model.visual_projection
        text_projection = clip_model.text_projection
        HAS_CLIP = True
        print('  Using HuggingFace CLIP ViT-B/16')
    except ImportError:
        print('  HuggingFace transformers not available')
        print('  Install: pip install transformers')
        return

    # Image-text pairs for evaluation
    test_pairs = [
        ('a photo of a cat', 'a photo of a dog'),
        ('a photo of a dog', 'a photo of a cat'),
        ('a photo of a panda', 'a photo of a cat'),
        ('a photo of an apple', 'a photo of a dog'),
    ]

    images = load_images(n=8)
    if images is None: print('No images'); return
    print(f'  {images.shape[0]} images ready\n')

    # Find ViT layers for ELHAM
    layers = find_vit_layers(vision_model)
    print(f'  ELHAM layers: {layers}')
    if len(layers) < 2:
        print('  Could not find enough ViT blocks — trying direct layer search')
        # Fallback: find any encoder layer
        all_blocks = []
        for n, m in vision_model.named_modules():
            if hasattr(m, 'num_heads') or ('attention' in n.lower() and 'self' in n.lower()):
                all_blocks.append(n)
        layers = all_blocks[:4] if len(all_blocks) >= 4 else all_blocks
        print(f'  Fallback layers: {layers}')
        if len(layers) < 2:
            print('  FAILED — cannot hook CLIP ViT blocks')
            return

    # Wrap vision model to return pooler_output
    class VisionWrapper(nn.Module):
        def __init__(self, vision, vproj):
            super().__init__()
            self.vision = vision
            self.vproj = vproj
        def forward(self, x):
            out = self.vision(x)
            pooled = out.pooler_output  # CLIP's pooled output
            return self.vproj(pooled)  # project to shared embedding space

    vision_wrapper = VisionWrapper(vision_model, visual_projection).to(DEVICE).eval()
    elham = ELHAMExplainer(vision_model, layers)

    # Encode text prompts
    text_embeddings = {}
    all_prompts = set()
    for pos, neg in test_pairs:
        all_prompts.add(pos); all_prompts.add(neg)

    for prompt in all_prompts:
        inputs = processor(text=[prompt], return_tensors='pt', padding=True).to(DEVICE)
        with torch.no_grad():
            text_out = text_model(**inputs)
            text_emb = text_out.pooler_output
            text_emb = text_projection(text_emb)
            text_emb = F.normalize(text_emb, dim=-1)
            text_embeddings[prompt] = text_emb

    # Evaluate
    print('  Running ELHAM + gradient baseline...\n')
    n = min(8, len(images))

    results = {'ELHAM': {'ins': [], 'del': [], 'prompt_sensitivity': []},
               'Gradient': {'ins': [], 'del': [], 'prompt_sensitivity': []}}

    for i in range(n):
        img = images[i:i+1].to(DEVICE)

        # ELHAM map (text-independent!)
        m_elham, _, _ = elham.explain(img)

        for pos_prompt, neg_prompt in test_pairs:
            # ELHAM: same map for both prompts
            ins_e_pos = cosine_insertion_auc(vision_wrapper, text_embeddings[pos_prompt],
                                             img, m_elham)
            ins_e_neg = cosine_insertion_auc(vision_wrapper, text_embeddings[neg_prompt],
                                             img, m_elham)
            del_e_pos = cosine_deletion_auc(vision_wrapper, text_embeddings[pos_prompt],
                                            img, m_elham)
            results['ELHAM']['ins'].append(ins_e_pos)
            results['ELHAM']['del'].append(del_e_pos)
            # Prompt sensitivity: does the same map score differently for different prompts?
            results['ELHAM']['prompt_sensitivity'].append(abs(ins_e_pos - ins_e_neg))

            # Gradient baseline: text-dependent
            m_grad = clip_gradient_map(vision_wrapper, text_embeddings[pos_prompt], img)
            ins_g = cosine_insertion_auc(vision_wrapper, text_embeddings[pos_prompt],
                                          img, m_grad)
            del_g = cosine_deletion_auc(vision_wrapper, text_embeddings[pos_prompt],
                                         img, m_grad)
            results['Gradient']['ins'].append(ins_g)
            results['Gradient']['del'].append(del_g)
            # Gradient is prompt-specific: sensitivity would be large (different maps)
            m_grad_neg = clip_gradient_map(vision_wrapper, text_embeddings[neg_prompt], img)
            ins_g_neg = cosine_insertion_auc(vision_wrapper, text_embeddings[neg_prompt],
                                              img, m_grad_neg)
            results['Gradient']['prompt_sensitivity'].append(abs(ins_g - ins_g_neg))

    # Print
    print(f'  {"Method":<12s} {"Ins AUC":>8s} {"Del AUC":>8s} {"Prompt Sensitivity":>18s}')
    print('  ' + '-'*55)
    for method in ['ELHAM', 'Gradient']:
        r = results[method]
        print(f'  {method:<12s} {np.mean(r["ins"]):>8.3f} {np.mean(r["del"]):>8.3f} '
              f'{np.mean(r["prompt_sensitivity"]):>18.4f}')

    print(f'\n  Interpretation:')
    print(f'  - ELHAM produces the SAME map regardless of text prompt')
    print(f'  - Low prompt sensitivity = maps are text-invariant (expected for ELHAM)')
    print(f'  - High prompt sensitivity = maps change with text (gradient method)')
    print(f'  - ELHAM\'s Ins/Del AUC measures "does visual certainty correlate with semantic relevance?"')

    # Compare: does the text-INDEPENDENT ELHAM map match text-dependent gradient map?
    elham_ins = np.mean(results['ELHAM']['ins'])
    grad_ins = np.mean(results['Gradient']['ins'])
    print(f'\n  ELHAM Ins AUC / Gradient Ins AUC = {elham_ins/grad_ins:.2f}')
    if elham_ins/grad_ins > 0.7:
        print(f'  → Visual feature certainty strongly correlates with semantic relevance')
        print(f'  → ELHAM maps are useful for CLIP without needing text at all')
    else:
        print(f'  → Visual feature certainty is weakly correlated with semantic relevance')
        print(f'  → ELHAM is useful for vision-encoder debugging, not text-conditioned XAI')

    # Plot (before removing explainer)
    _plot_clip_results(images, vision_wrapper, text_embeddings, elham, layers, results, n)

    elham.remove()

    return results


def _plot_clip_results(images, vision_wrapper, text_embeddings, elham, layers, results, n):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for row in range(2):
        img_idx = row * 2
        if img_idx >= n: break
        img = images[img_idx:img_idx+1].to(DEVICE)
        m_elham, info_gains, entropies = elham.explain(img)

        # Denormalize
        mean = torch.tensor([0.481,0.458,0.408]).view(3,1,1)
        std = torch.tensor([0.269,0.261,0.276]).view(3,1,1)
        img_disp = (img.squeeze(0).cpu()*std+mean).clamp(0,1).permute(1,2,0).numpy()

        # ELHAM overlay
        axes[row,0].imshow(img_disp)
        if m_elham.max()>0: axes[row,0].imshow(m_elham, cmap='inferno', alpha=0.5)
        axes[row,0].set_title('ELHAM (text-independent)',fontsize=9); axes[row,0].axis('off')

        # ELHAM layer maps
        layer_keys = list(info_gains.keys())
        for j in range(min(3, len(layer_keys))):
            ig = info_gains[layer_keys[j]]
            up = F.interpolate(torch.tensor(ig).unsqueeze(0).unsqueeze(0),
                               size=(224,224),mode='bilinear',align_corners=False).squeeze().numpy()
            axes[row,1+j].imshow(img_disp)
            axes[row,1+j].imshow(up, cmap='inferno', alpha=0.5)
            axes[row,1+j].set_title(f'ELHAM {layer_keys[j]}',fontsize=8); axes[row,1+j].axis('off')

    plt.suptitle('ELHAM on CLIP: Text-Independent Visual Attribution',fontweight='bold',fontsize=13)
    plt.tight_layout()
    plt.savefig('elham_clip.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('  Saved: elham_clip.png')


if __name__ == '__main__':
    run_clip_test()
