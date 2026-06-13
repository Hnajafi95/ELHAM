# ELHAM → real XAI contribution: ROADMAP

**Goal (set 2026-06-13):** Turn ELHAM into a genuine, publishable XAI contribution.
Chosen framing: *efficient gradient-free attribution* — a single-forward-pass,
gradient-free, **class-discriminative** method positioned against the expensive
gradient-free white-box family (Score-CAM, Ablation-CAM, which need C forward passes).

Working style: one item at a time, test, record result here, move on.

---

## Diagnosis of the original method (done)
- **Fatal flaw (not in REVIEW.md):** ELHAM is *completely class-agnostic* —
  `target_class` is accepted but never used in any of the 9 eval files.
  explain(img,"dog") == explain(img,"cat"). It is not an attribution method.
- Pre-existing bugs in `eval_ground_truth.py`: n_classes=6 vs model nc=2
  (CrossEntropy crash); T2 generator writes out-of-bounds patches. It never ran.

## Experiment 1 — synthetic ground-truth diagnostic (`eval_classcond.py`) — DONE
64×64 synthetic causal task, known GT masks, 3 sub-tasks (single/competing/interaction),
n=50 each. Compared: ELHAM-IG, ELHAM-Hlast, CAM, ELHAM-C fusions, Score-CAM, Grad-CAM.

**Findings (decisive):**
- **Q1 — class-conditioning works, gradient-free.** ELHAM-IG true-vs-wrong map
  corr 1.000 → ELHAM-C ≈ −0.35. The fatal flaw IS fixable in one forward pass.
- **Q2 — the entropy signal adds NO localization value over gradient-free CAM.**
  Best class-conditional ELHAM (Hrefine) ≈ CAM exactly (0.692 vs 0.689 etc.).
- **ELHAM's signature mechanism (hierarchical info-gain) is its *weakest* variant.**
  ELHAM-IG (0.19–0.27 AP) << ELHAM-Hlast (0.40–0.69) << CAM ≈ ELHAM-C < Score-CAM.
  → The "information gain between layers" story (heart of the paper) is unsupported.
- **Score-CAM wins everywhere** (0.71/0.62/0.42) but costs 590× (128 fwd passes).
  → There IS headroom above CAM; a cheap single-pass method that captures it would
  be a real contribution. Entropy (as defined) does not capture it.

**Honest conclusion:** the entropy-attribution premise is weak on controlled data.
The valuable, still-open problem: **a single-forward-pass gradient-free attribution
that matches Score-CAM/Ablation-CAM quality.** ELHAM-entropy is one (so far failing)
attempt; the testbed + class-conditioning machinery are reusable.

---

## Next decision point
Before declaring the entropy direction dead, run the decisive **real-data** test —
synthetic 8×8 CAM may not expose the resolution gap where a multi-layer method helps.

### Item 2 (proposed) — ImageNet validation on proxima3 (7× H200)
- Pretrained ResNet-50 + ViT-B/16, 200+ ImageNet-val images w/ bbox/seg GT.
- Metrics: GT-overlap (pointing game, EPG on bbox), Insertion/Deletion, *and* the
  single-pass-vs-Score-CAM cost/quality frontier.
- Decision rule: if no entropy variant beats CAM on real data either → **pivot**
  the contribution to "cheap single-pass approximation of Score-CAM" (try better
  mechanisms: activation-magnitude weighting, gradient-free channel-relevance
  propagation, learned 1-pass surrogate) OR write the rigorous negative/analysis paper.

## Experiment 2 — ImageNet real-data test (`eval_imagenet_gt.py`) — DONE (proxima3 H200)
ResNet-50, n=200 real Imagenette images (ImageNet classes → top-1 meaningful),
bootstrap 95% CIs. Insertion/Deletion AUC + cost. Auto-downloads data.

**Results (Insertion AUC ↑ / Deletion AUC ↓):**
- CAM 0.389 / 0.178   ==  Grad-CAM 0.389 / 0.178  (identical — correctness check ✓)
- ELHAM-C-Hrefine 0.388 / 0.178  (class-cond fix ties CAM exactly, adds nothing)
- ELHAM-C-refine 0.384 / 0.169
- Score-CAM 0.383 / 0.187  (2048 fwd/img, 309 ms)   ← does NOT beat CAM
- Ablation-CAM 0.385 / 0.187 (2049 fwd/img)          ← does NOT beat CAM
- ELHAM-Hlast 0.342 / 0.233   (worse than CAM, CIs separated)
- ELHAM-IG 0.305 / **0.161**  (worst insertion; BEST deletion — see below)
- forward-only methods: 1.07 ms/img; Grad-CAM 5.37 ms.

**Verdict (well-powered, matches synthetic):**
1. No entropy variant beats CAM on **insertion** (cleaner metric); IG/Hlast clearly worse.
2. ELHAM-IG's best-**deletion** + worst-insertion + class-agnostic = classic edge/
   high-frequency ARTIFACT, not class-causal attribution (insertion/deletion asymmetry).
3. Class-conditioning works but only reproduces CAM.
4. **Score-CAM/Ablation-CAM do NOT beat plain CAM here** despite 2048× cost — so the
   "cheap single-pass approximation of expensive grad-free CAMs" pivot has no quality
   gap to target on this benchmark. That pivot is also weakened.

→ **Entropy-as-attribution is falsified.** CAM (1 pass, grad-free, class-disc) already
  dominates. The original framing is dead.

## DECISION (2026-06-13): pivot the method to its natural problem
ELHAM's real properties — forward-only, **class-agnostic**, no training data, fast,
quantization-robust — are a *bad* fit for attribution (needs class-discriminativeness)
but a *good* fit for **OOD / confidence estimation** (naturally class-agnostic; benefits
from no-training-data + quantization-robustness + speed). The "class-agnostic flaw"
becomes a non-issue. This repositions ELHAM where its strengths are assets.

### Item 3 (NEXT) — OOD pilot
Hypothesis: aggregated channel entropy (low = decisive = in-distribution) separates
ID from OOD. ID=CIFAR-10/Imagenette; OOD=SVHN/CIFAR-100/noise/textures.
Score variants: mean last-layer norm-entropy, mean info-gain, multi-layer combo.
Baselines: MSP (max-softmax), Energy (logsumexp logits) — the standard cheap forward
baselines. Metric: AUROC, FPR@95. Decision: if ELHAM-entropy ≥ MSP/Energy → real paper
("forward-only, training-data-free, quantization-robust OOD"); else → write honest
negative/methodology paper (debunk + reusable validation protocol; ELHAM as case study).

## Experiment 3 — OOD pilot (`eval_ood.py`) — DONE (proxima3). NUANCED, not dead.
CIFAR-10 ID (ResNet-18, 91.6% acc), OOD = SVHN/CIFAR-100/Gaussian/Uniform. AUROC.
Anomaly = high channel entropy. Baselines MSP/Energy/MaxLogit.

**Per-OOD AUROC (the mean is misleading — report per-type):**
                SVHN    C100    Gauss   Unif
  MSP           0.900   0.861   0.686   0.640
  Energy        0.812   0.878   0.484   0.475
  ELHAM-Hall    0.966   0.712   0.000   0.000
  ELHAM-Hlast   0.814   0.850   0.018   0.022
  ELHAM-Hmin    0.058   0.502   0.999   1.000

**Findings:**
1. **ELHAM-Hall BEATS MSP & Energy on CIFAR-10→SVHN (0.966 vs 0.900/0.812)** — the
   canonical OOD benchmark — with no logits/grads/training-data. Real signal.
2. Entropy aggregations are COMPLEMENTARY: Hall→far/realistic OOD, Hmin→noise. No cheap
   baseline is robust across types either (Energy fails noise, 0.48). A *combined*
   entropy score could be uniquely robust.
3. Hypothesis: entropy excels at FAR-OOD/covariate shift (SVHN 0.966), weak on NEAR-OOD/
   semantic shift (CIFAR-100 0.712). If it holds → "channel entropy as far-OOD detector".
4. Caveat: OOD is a mature, crowded field. Beating MSP/Energy is table stakes; ELHAM's
   forward-only/quantization advantages are thinner here (MSP/Energy are also forward-only
   & run on quantized models). Need to beat STRONG baselines (ReAct/Mahalanobis/KNN) and
   show a genuine niche to publish as a method.

### Item 4 (NEXT) — decisive OOD follow-up
(a) diagnose+fix noise catastrophe (entropy conflates "few channels active" w/ decisive;
    combine w/ magnitude/energy term, or fuse Hall+Hmin); (b) near-vs-far OOD taxonomy
    (add LSUN/Places/Textures/iSUN; 2nd ID set e.g. CIFAR-100-ID) to test if far-OOD win
    generalizes; (c) add ≥1 strong baseline (ReAct or Mahalanobis) for credibility.
    Decision: clear niche/robust-combined win vs strong baselines → OOD method paper;
    else → balanced methodology paper (attribution-negative + OOD-promise, "what channel
    entropy measures").

## Experiment 4 — decisive OOD follow-up (`eval_ood2.py`) — DONE (proxima3)
2 ID sets (CIFAR-10 92.6%, CIFAR-100 73.8%), 5 OOD each, strong baselines ReAct+KNN.
**VERDICT (mean over ID sets), AUROC:**
              FAR    NEAR   NOISE
  KNN        0.902  0.777  0.998
  ReAct      0.869  0.797  0.897
  Energy     0.833  0.811  0.689
  MSP        0.812  0.800  0.735
  ELHAM-Hall 0.768  0.735  0.004
  ELHAM-Fused 0.678 0.546  0.996

**Findings — OOD method path FALSIFIED too:**
1. The CIFAR-10→SVHN win (0.97) was IDIOSYNCRATIC — did not generalize (DTD 0.76 vs
   KNN 0.91; vanished with CIFAR-100 as ID). "Entropy = far-OOD detector" falsified.
2. ELHAM-Fused fixed noise (0.99) but by trading: collapses on far/near (0.68/0.55).
3. KNN/ReAct dominate; no ELHAM variant is both robust AND competitive with strong
   baselines. Every entropy aggregation has a catastrophic failure mode.

→ **Channel entropy is not a competitive method for attribution OR OOD.** Both pivots
  rigorously tested with proper protocols + strong baselines. Stop the method-rescue hunt.

## OUTCOME / next (2026-06-13): consolidate into honest methodology paper
Recommended: "What does channel entropy measure in deep networks?" — a rigorous
negative-results + methodology paper. Positive contributions:
  - Structural critique: forward-only signals are often class-AGNOSTIC and thus cannot be
    faithful attributions; a 1-line class-sensitivity diagnostic (ELHAM fails: target_class
    unused → identical maps for every class).
  - Ground-truth synthetic protocol (eval_classcond.py) + ImageNet insertion/deletion
    (eval_imagenet_gt.py): CAM dominates; ELHAM's deletion "win" is an edge artifact.
  - OOD study (eval_ood/eval_ood2): entropy not competitive vs KNN/ReAct; apparent wins
    idiosyncratic. Cautionary lesson for the crowded XAI field. Template: Adebayo et al.
    "Sanity Checks for Saliency Maps".
Reusable assets (the real lasting value): eval harnesses + the class-sensitivity +
ground-truth + insertion/deletion-asymmetry protocol — usable for the user's other XAI
work (PRISM/FFCA).

### Backlog
- Ablation-CAM baseline (the other expensive grad-free rival).
- Proper statistics everywhere (bootstrap CIs, n≥100) — REVIEW.md item #4.
- Sanity checks (model + data randomization) — Adebayo et al.
- Drop/【downgrade】 the oversold claims (IB connection, "only XAI method": LIME/RISE exist).
