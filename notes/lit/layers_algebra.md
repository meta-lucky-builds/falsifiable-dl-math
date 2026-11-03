# Literature scout: layer and architecture mathematics (thread 1 of 3)

Scout date: 2026-11-04. Source: alphaXiv only (8 discovery queries, ~19 papers opened). No OpenReview or general web search was done, so coverage is thinner than "exhaustive".

## Read this first: how strong is the evidence behind this file

- Most "read" papers were opened through alphaXiv's get_paper_content, which returns an AI-generated report, not the PDF. Numbers below come from those reports. Exceptions where I saw raw paper text: 2604.01563 (full text), 2610.05488, 2607.19514, 2607.15525, 2609.32814 (abstract + start only).
- Treat every number as "reported by the report of the paper", and re-check against the PDF before the coordinator relies on it.
- Retrieval anomaly: fetching 2602.08216 (listed as "Kaniadakis and Reciprocal-Symmetric Abe Operators", Sejong Univ.) returned a different paper (a "Thermodynamic Isomorphism of Transformers" free-energy derivation of softmax). I did NOT verify any claim about Kaniadakis/Abe attention. It appears below only as a listing.
- Seeds, tuning and replication are marked per paper. "Not stated" means the report did not say, not that the paper did not do it.

---

## 1. Papers (19 read, grouped by thread)

Format: ID | core formula | claimed gain | evidence strength.

### Attention normalizers and priors

**1. 2501.19399, Scalable-Softmax (SSMax), Nakanishi**
- Formula: `SSMax(z)_i = n^(s*z_i) / sum_j n^(s*z_j)`, i.e. softmax with logits scaled by `s*log n` (n = context length, s learnable per head). Implemented by scaling q by `s*log n`.
- Motivation (derived): softmax max-weight tends to 0 as n grows ("attention fading"); SSMax keeps max weight near 1 once `z_max - z_2nd > 1/s`. The `log n` form was found empirically from a learned `p_n`, then justified.
- Claim: lower pretraining loss, per-position loss stays low to about 10x training length (20k vs 1024) with RoPE theta changed 50x and no extra training, better needle-in-haystack.
- Evidence: one 162M Llama-2-like model, SlimPajama. Seeds not stated. Baseline tuning not stated. Ablations are informative: fixing s=1 hurts retrieval; adding a bias b hurts long-context; swapping softmax->SSMax after pretraining only partly works. Independent partial check: 2506.16640 uses SSMax as a baseline and beats it on synthetic tasks (not a replication of SSMax's own claim).

**2. 2506.16640, ASEntmax: alpha-entmax with length-adaptive temperature, Vasylenko et al. (ICLR 2026)**
- Formula: attention = alpha-entmax(`(delta + beta*(log n)^gamma) * z`), beta/gamma/delta learned per head; alpha>1 gives exact zeros. Positional scheme NAPE = half heads ALiBi, half NoPE.
- Derived claims: entmax keeps bounded entropy O(log s) as n grows (softmax: O(log n)); gradient paths O(s^L) vs O(n^L).
- Claim: 95.3% accuracy at 65k tokens on associative recall trained on 64 (softmax 3.0%); LM at 420M: decreasing perplexity out to 8x (16k vs 2k) and ~99.8% S-NIAH-1 at 16k; short-context benchmarks equal.
- Evidence: synthetic tasks plus 420M LLaMA-3-style on 7B tokens of DCLM-Edu. Seeds not stated in report. Strong comparison set (softmax, SSMax, adaptive temperature). Confound: NAPE positional choice is changed at the same time as the normalizer, and compared against base PE variants only partly. Needs AdaSplash kernel for speed.

**3. 2601.15380, GOAT: attention as entropic optimal transport with a trainable prior, Litman and Guo (Stanford, ICML)**
- Formula: `p* = argmin_p { -<p,s> + tau*KL(p||pi) } = softmax(s/tau + log pi)`. Softmax = uniform prior. Prior log-term = truncated Fourier series in (i-j) (relative) + key-only term u(j) (sink), folded into q/k by trig identities so plain SDPA works.
- Derived claim: if content-score range is small, attention -> prior; a sink margin delta bounds non-sink mass by `(L-1)/(e^delta + L - 1)`. Fourier form is the only admissible relative prior under SDPA-compatibility + translation equivariance + boundedness (under the authors' constraints).
- Claim: 1.23B FineWeb: perplexity 2k/4k/8k/16k = 20.75/21.04/22.69/24.77 vs RoPE 20.82/28.53/89.45/417.43; 125M C4: +1.55 ppl vs ALiBi in-distribution; genome NLL 1.1294 vs 1.2054 (RoPE) with lower memory.
- Evidence: several scales and modalities; seeds not stated; baselines are RoPE/ALiBi/learned, NOT SSMax or entmax. No independent replication found.

**4. 2604.07925, Sinkhorn doubly-stochastic attention rank decay, Lapenna, Fioresi, Gharesifard**
- Formula: replace row-softmax by Sinkhorn(QK^T/sqrt d) (50 log-domain iterations). Bound via first-order expansion of Sinkhorn with a tangent-space projector Q: `||res(SAN(X))||_2 <= C^(3^(L-1)) ||res(X)||^(3^L)`.
- Result: theory says doubly-exponential rank collapse also holds for Sinkhorn attention (no free lunch asymptotically). Empirically Sinkhorn delays collapse of the attention-product matrices in trained models, esp. with skip connections.
- Evidence: small trained models (AG News, MNIST, Cats-and-Dogs ViT). Important negatives inside the paper: no difference for random stochastic matrices (so benefit depends on learned cross-layer correlations); no consistent output-rank advantage on the image tasks. Bound is first-order approximate.

**5. 2602.11534, Krause Synchronization Transformers, Liu, Yue, Welling, Song**
- Formula: `s_ij = exp(-||q_i - k_j||^2 / (2 sigma^2))`, normalised only over a local window N_i and then top-k within it (bounded-confidence consensus). Linear cost O(N W d).
- Claim: CIFAR-10 KViT-B 95.35% vs ViT-B 92.45%; ImageNet-1K 76.75 vs 75.85 with fewer FLOPs; Llama3-8B/Qwen1.5-7B with LoRA and Krause as an auxiliary path improve zero-shot suites; from-scratch 100M/200M LMs win "3 of 6" benchmarks.
- Evidence: weak-to-moderate. ViT-from-scratch on CIFAR is a soft baseline, and the gains are confounded with a local-window prior. The cleanest control, replacing dot-product by the RBF kernel inside Swin (already windowed), gives only +0.36 pt on CIFAR-10 (91.09 vs 90.73). LLM results are LoRA fine-tunes with an added pathway (extra parameters). Seeds not stated.

**6. 2505.17190, Tropical Attention, Hashemi, Pasque, Teska, Yoshida**
- Formula: tropical (max-plus) projections `(A (x) B)_ij = max_t {A_it + B_tj}`; score `S_ij = -d_H(q_i, k_j)` with the tropical Hilbert projective metric; aggregate `C_i = max_j {S_ij + V_j}`; map back with exp. Claimed 1-Lipschitz and projectively invariant.
- Claim: far better length/value/noise OOD on 11 combinatorial tasks (e.g. Quickselect micro-F1 77.06 vs 4.66 vanilla); 3-9x faster than Universal Transformer; second on average on Long Range Arena. Proofs: universal approximation of tropical circuits, tropical transitive closure.
- Evidence: self-built synthetic datasets, in-house baselines (vanilla, adaptive softmax, Universal Transformer), tuning of baselines unclear, seeds not stated. No language-model result. No independent replication found.

**7. 2610.05488, "Why Softpick Fails at Initialization", Sood, Singh, Bansal (IIT Roorkee)**
- Formula (softpick, from original Zuhri et al. 2025, only seen here): `ReLU(e^(x_i - m) - e^(-m)) / (sum_j |e^(x_j - m) - e^(-m)| + eps)`. Authors split denominator into D+ and D-.
- Result: softpick's advantage fades at scale; at init some rows have D+ = 0 exactly, near-dead rows give gradient norms above 1e12; only softpick and a stop-gradient variant (forward D+ + D-, backward through D+) train from scratch; at 230M the variant matches softpick on quantization, has fewer dead heads, retrieves passkeys better.
- Evidence: 0-vote preprint, one scale, seeds not stated. I did not open the original softpick paper (2504.20966 appeared in search only).

### Normalization replacements and Lipschitz layers

**8. 2512.10938, Derf (Dynamic erf), Chen, Lu, Zhu, Sun, Liu**
- Formula: `Derf(x) = gamma * erf(alpha*x + s) + beta`, alpha init 0.5, s init 0, per-channel gamma/beta. Principles from sweeps: zero-centred, bounded, centre-sensitive, monotone.
- Claim: ViT-B 82.8 vs LN 82.3 vs DyT 82.5; DiT-L/4 FID 43.94 vs LN 45.91; GPT-2 124M val loss 2.94 = LN (DyT 2.97); also wav2vec, DNA models. Higher train loss than LN, so gain attributed to generalisation not fitting.
- Evidence: broad modality coverage but gains are small (0.3-0.6 pt) and seeds not stated. LM result is a tie, not a win. Replicated only adversarially (see 11 and 12 below).

**9. 2604.23434, "When does removing LayerNorm help?", Verma (independent)**
- Claim (a critique of DyT `gamma*tanh(alpha x)+beta`): DyT is a regime-dependent implicit regulariser. 64M model: -27.3% val loss at 1M tokens, +18.8% (worse) at 118M tokens; at 3.78B/118M tokens +27.9% worse. HardTanh reproduces the pattern; LN + dropout 0.5 matches DyT in the data-rich regime. 33% per-seed collapse rate in Llama-style stacks, localized to SwiGLU gating (saturation >= 0.5 predicts collapse, r=0.94). Differential Attention shows the opposite regime preference and collapsed at 3.78B in this study, and hurt LAMBADA.
- Evidence: moderate. 3 seeds mostly, Bonferroni t-tests, 5k steps, compute-limited regime only (T/P < 1.84), single author. Screening heuristic fails leave-one-scale-out (50%).

**10. 2604.01563, Normalization-optimizer coupling, Abouzeid (Georgia Tech)**
- Result: 1B Llama, 1000 steps, 3 seeds per cell. Derf gap to RMSNorm is +0.31 nats under AdamW but +0.97 under Muon; DyT shows no such penalty (-0.10 interaction). Mechanism: Muon grows weights ~2.2x faster, 83% of erf outputs saturate. Fixes: alpha 0.5->0.3 recovers ~80%; EMA-of-scale blend `erf(alpha*[(1-lambda)x + lambda*x/sigma_hat] + s)` recovers ~84%. Failed fixes: grad clipping, weight decay, higher alpha LR, residual scaling; asinh pre-compression only 49%.
- Evidence: strongest of the normalization papers methodologically (factorial, full text read, failed-intervention table), but 1B/1000 steps is far from Chinchilla; authors say fixed alpha may not hold at longer horizons; not independently replicated.

**11. 2507.13338, Training Transformers with Enforced Lipschitz Constants, Newhouse, Hess, Cesista, Zahorodnii, Bernstein, Isola**
- Formulas: Muon + spectral soft cap (odd polynomial approximating `sigma -> min(sigma_max, sigma)`), spectral normalization, spectral hammer; residual `((N-1)/N)x + (1/N)block(x)`; attention logit scale 1/d instead of 1/sqrt d; GeLU/1.1289 made 1-Lipschitz.
- Claim: 145M NanoGPT on FineWeb trains stably without LayerNorm, QK-norm or logit softcap; a 10-Lipschitz model reaches 21.2% val accuracy; matching the 39.4% baseline needs a certified bound of 1e264 (useless bound), yet max activations stay small (160 vs 148,480 in baseline).
- Evidence: honest negative on tight bounds; sweeps with hand-tuned sigma_max; single runs reported. Muon is the enabling ingredient (AdamW worse trade-off).

### KAN, rational and structured layers

**12. 2407.16674, KAN or MLP: A Fairer Comparison, Yu, Yu, Wang (NUS)**
- Setup: equalise parameters or FLOPs. Result: MLP wins in most ML/vision/audio/NLP tasks (vision: all 8 datasets); KAN wins only in symbolic formula fitting (7/8 at equal params, roughly tie at equal FLOPs). Ablation: MLP with learnable B-spline activations matches or beats KAN on symbolic tasks, so the B-spline activation, not the KAN topology, is the active ingredient. Class-incremental continual learning on MNIST: KAN forgets more (first two tasks drop to 0).
- Evidence: good (equal-budget, ablation, code released); one GPU, Adam, 20 epochs for most tasks; widely cited, direction consistent with 13.

**13. 2607.15525, KANs for Small Language Models, Alves and Vicente (USP)**
- Result: FFN replaced by B-spline KAN, grouped Chebyshev, or rational GR-KAN. BabyLM validation loss: gated and KAN-family FFNs all beat vanilla GELU MLP slightly, but on BLiMP (10 seeds, 59,875 pairs) accuracies span 62.4-63.1% with overlapping CIs, EWoK at chance, the single +0.7 GR-KAN edge reverses on the BLiMP supplement. Param-matched MLPEdge transformer is below an MLP on Wikitext-103; corrected 286M GR-KAN stays below SwiGLU on ClimbMix. Pruning 20-25% of edges is nearly free, but structured MLP-neuron pruning does as well.
- Evidence: best KAN-for-LM audit I found (10 seeds, multiple corpora); validation loss does not predict benchmark ranking, a useful warning for our own protocol.

**14. 2602.12390, Rational Neural Networks have Expressivity Advantages, Tang and Townsend (Cornell)**
- Formula: trainable low-degree rational (Pade) activation with strictly positive denominator `P(x)/Q(x)`. Theory: any smooth-activation network is matched by a rational one with O(log^3 log(1/eps)) overhead; some rational functions need Omega(log(1/eps)) parameters for bounded smooth-activation nets.
- Claim: matches or beats GELU/ReLU/SiLU on CIFAR-10 (VGG), Tiny-ImageNet (ViT-S, CaiT-S24, Swin-T), offline RL (IQL, TD3+BC; wins 5 of 6). Key finding for us: normalization layers hurt adaptive rationals (affine absorption into coefficients, non-identifiability); in a no-LayerNorm ViT the rationals win by a large margin and fixed activations often fail to train.
- Evidence: theory solid, experiments small-scale, seeds not stated, only the MLP activation swapped. Not tested in language models; 13 suggests rational (GR-KAN) at LM scale does not beat SwiGLU.

**15. 2609.32814, Associative-algebra layers for Transformers, Koziev and Oseledets**
- Formula: replace the matrix product in FFN projections by a sparser block-interaction product of an associative algebra (provably optimal for its bilinear rank by the Alder-Strassen bound); quadratic arithmetic at fixed block size.
- Result: two ~110M models, same 12.3B-token recipe, +6.2-7.8% generation throughput but lower scores on all three downstream metrics. Authors call it a feasibility check.
- Evidence: single run each, labelled "paper under review"; honest small negative-or-neutral result.

### Geometric and topological layers

**16. 2607.06634, When do geometric algebra layers beat scalarization?, Polly**
- Setup: Cl(3,0) bilinear blocks `gp(Ax, Bx) + Cx` with grade-wise weight tying (exact SO(3) equivariance), against a minimal exactly equivariant "scalarization" baseline (pairwise dot products into an MLP that outputs coefficients for a fixed equivariant basis).
- Result: single-stage vector laws, scalarization is as good or better at 13-19x lower cost (cross product NMSE 5.4e-4 vs 1.4e-2 with 1000 samples). Gain only for chained group operations (composed rotations, NMSE 0.031 at 100 samples vs 0.54 scalarization) and only in low data; scalarization catches up near 1000 samples on torque. Nested cross product without rotations: scalarization 24x better. Nobody extrapolates invariant magnitudes (NMSE > 1 on radial OOD).
- Evidence: well controlled synthetic study with tuned baselines, but tiny models and synthetic physics laws only; single author.

**17. 2601.06793, CliffordNet, Ji**
- Formula: geometric product `uv = u.v + u^v` as a feature interaction; implemented with channel shifts: dot `SiLU(H (.) T_s(C))`, wedge `H (.) T_s(C) - T_s(H) (.) C`, concatenated and linearly projected; C = depthwise-conv "Laplacian" context; gated residual. No Clifford-valued weights.
- Claim: CIFAR-100 76.41% with 1.4M params (ResNet-18 level), 78.05% with 3.0M; FFN-free variant 77.63%; wedge-only 76.35 vs dot-only 76.91, both 1.7M.
- Evidence: weak. Single author, CIFAR-100 only, small models, ViT-Tiny reported at 65.87 under the same recipe (an under-tuned ViT baseline on CIFAR), seeds not stated. Missing control: the dot term is a gated bilinear product (GLU-like) and the wedge is its antisymmetrization; no ablation against a plain GLU/SwiGLU with the same shifts.

**18. 2607.19514, Do Sheaf Neural Networks use holonomy?, Grover and Bourgerie (KTH)**
- Setup: 3-layer Neural Sheaf Diffusion with SO(2) edge transports, triangle-counting task; interventions on frozen checkpoints (flat, shuffled, identity transports).
- Result: predictions depend on the learned assignment of transports to edges, but holonomy is not isolated as the mechanism: identity-trained NSD can outperform learned-connection NSD; twist does not predict counting accuracy; random 6-regular graphs acquire twist without counting skill.
- Evidence: carefully hedged, multi-seed, small graphs. Read abstract and first page of raw text only.

**19. 2607.05268, Is the geometry doing the work? (hyperbolic VLM audit), Kim, Kim, Jang (MADI)**
- Setup: MERU, HyCoCLIP, PHyCLIP (7 released checkpoints plus matched from-scratch runs on 13.1M GRIT pairs). Diagnostics: dimensionless radius `u = sqrt(c*rho)` and distortion `H(u) = u/asinh(u)`; shuffle-controlled radial-order test.
- Result: all checkpoints are near-Euclidean (largest u = 0.37, far below 0.84 where distortion hits 10%); no pair-specific radial ordering above shuffle control; text-side entailment cones saturated at pi/2; entailment loss itself contains a "low-curvature shortcut" (smaller c widens cones, lowers violation without learning order); curvature collapses even without entailment. Hierarchy gains trace to box/compositional supervision, not geometry.
- Evidence: strong (preregistered diagnostics, three seeds in from-scratch runs, matched controls), though limited to hyperbolic VLMs.

### Seen in search listings only (abstract snippet, NOT opened; do not treat as verified)
- 2606.15669 Z-Plane Neural Networks: "bounded geometric activation replaces ReLU and LayerNorm".
- 2602.23057 Affine-Scaled Attention; 2609.22005 Abstention and noise filtering for softmax attention; 2610.22714 First Passage Attention; 2611.01472 Causal Optimal-Transport Attention (posted 2026-11-01).
- 2605.12879 ASAP and 2502.07962 ESPFormer and 2509.23436 LOTFormer (doubly-stochastic attention variants).
- 2409.10594 Kolmogorov-Arnold Transformer (the GR-KAN origin).
- 2606.02927 SaluNet, 2601.09719 Bounded Tanh, 2602.10408 gated normalisation removal, 2503.21708 LN vs dynamic activation functions relationship, 2503.10622 DyT (original).
- 2610.30121 Geometric Algebra for Neural Networks survey; 2610.01322 Clifford Sheaf NNs.
- 2603.14831 Neural networks as local-to-global computations (ReLU net as a cellular sheaf).

---

## 2. Tried and disappointing

| Idea | What happened | Source |
|---|---|---|
| KAN as general MLP replacement | MLP wins at equal params/FLOPs in ML, vision, audio, NLP; KAN edge only on symbolic regression; KAN forgets more in class-incremental learning | 2407.16674 |
| B-spline/Chebyshev/rational KAN in FFN of small LMs | BabyLM loss slightly better than GELU MLP but BLiMP/EWoK benchmark-equivalent (10 seeds); 286M GR-KAN below SwiGLU; "validation loss does not predict benchmark ranking" | 2607.15525 |
| Sinkhorn/doubly stochastic attention as a cure for rank collapse | Proven doubly-exponential rank collapse remains; benefit absent for random matrices; no output-rank gain on image tasks | 2604.07925 |
| Softpick (rectified softmax) | Advantage shrinks with scale; fails at initialization (dead rows, gradients above 1e12) unless a stop-gradient variant is used | 2610.05488 |
| DyT as drop-in LN replacement | Worse in data-rich regime (+18.8% to +27.9% val loss), 33% collapse rate with SwiGLU, hurts LAMBADA/BLiMP | 2604.23434 |
| Derf with Muon | Gap to RMSNorm triples (+0.31 to +0.97 nats), no NaN so easy to miss; alpha=0.3 or scale-EMA needed | 2604.01563 |
| Derf in GPT-2 124M | Ties LayerNorm (2.94); wins are in ViT/DiT with sub-1-pt margins | 2512.10938 |
| Tight Lipschitz certificates for competitive LMs | Matching baseline accuracy requires a bound of 1e264; tight bounds cost accuracy (21.2% vs 39.4%) | 2507.13338 |
| Differential attention at scale | Neutral/harmful in low-data regime, collapsed at 3.78B in that study, hurt LAMBADA (single independent study, treat as unreplicated) | 2604.23434 |
| Associative-algebra product in FFN | +6-8% throughput, lower on all three downstream metrics at 110M | 2609.32814 |
| Clifford/GA layers on simple equivariant tasks | Scalarization baseline equal or better at 13-19x lower cost; GA only wins for chained group operations at low data | 2607.06634 |
| Hyperbolic VLMs actually using hyperbolic geometry | Operating regime near Euclidean (u <= 0.37), no radial hierarchy above shuffle control, curvature collapse | 2607.05268 |
| Sheaf holonomy as the working mechanism | Not isolated; identity transports can match or beat learned ones | 2607.19514 |
| Rational/Pade activation combined with normalization | Normalization hurts adaptive rationals (non-identifiability); gain appears only without LN | 2602.12390 |
| Krause attention gains | Largest effects are on small-data CIFAR with soft ViT baseline; windowed-Swin control only +0.36 pt | 2602.11534 |

Cross-cutting pattern worth recording: results that look best are where baselines are weakest (CIFAR from scratch, synthetic algorithmic tasks, 1000-step runs). Gains that survived hostile follow-up are mostly "attention scaling with log n / sparsity for length extrapolation" (items 1, 2, 3), and even those are single-lab at modest scale.

---

## 3. Open gaps (all TENTATIVE)

Method note: my "searches" are 8 alphaXiv discovery queries (listed at end). Absence in those results means "not found by me", not "nobody did it". OpenReview, Google Scholar and GitHub were not searched.

**Gap A. Length-aware attention normalizers have never been compared head to head.**
SSMax (log n logit scale, 2501.19399), ASEntmax (log n scale plus sparsity, 2506.16640) and GOAT (log-prior Fourier bias, 2601.15380) all target attention dispersion/sinks and long-context extrapolation. GOAT's baselines were RoPE/ALiBi/learned PE only; ASEntmax compared with SSMax but changed the positional scheme to NAPE at the same time; SSMax compared with plain softmax only. In the EOT view `softmax(s/tau + log pi)` of GOAT, SSMax is a specific temperature `tau = 1/(s log n)` and ASEntmax is the same temperature with a sparse Bregman geometry; a joint form `alpha-entmax((s log n) z + log pi)` is trivially writable. Searched: "scalable softmax", "entmax", "attention priors" with softmax-alternative phrasing; found no paper with all three in one table. Tentative.

**Gap B. Doubly-stochastic or OT attention vs log-n/sparse attention on length extrapolation.**
2604.07925 shows Sinkhorn changes rank decay only through learned correlations; the OT-attention papers (ESPFormer, ASAP, LOTFormer seen as snippets only) emphasise sinks and rank collapse, the SSMax/entmax line emphasises dispersion in n. These are two explanations of the same symptom (mass spread over many tokens). A snippet shows 2611.01472 (Causal OT attention, posted 2026-11-01) studies causal column-balancing and length extrapolation, so this gap may close within days; I could not read it. Column normalisation under causal masks is also not well-defined without extra care, which may be why nobody compared. Tentative and time-sensitive.

**Gap C. Rational/Pade activations in a normalization-free LM trained with Muon.**
2602.12390 reports rationals gain only when LayerNorm is removed (ViT, Tiny-ImageNet). 2512.10938 and 2604.23434 show norm-free works in some regimes but SwiGLU gating saturates tanh; 2604.01563 shows Muon changes which pointwise normalizer survives. 2607.15525 shows GR-KAN (rational) at 286M stays below SwiGLU but with normalization present. Nobody I found tests rational gates (GLU with a Pade nonlinearity) with DyT/Derf-style norm-free blocks under Muon. Searched: "rational activation transformer", "Pade", "normalization-free", "Muon normalization". Tentative.

**Gap D. Spectral-norm-capped weights as a fix for Derf/DyT saturation under Muon.**
2604.01563 identifies the mechanism: Muon grows weights ~2.2x faster, so erf inputs blow up (std 11.8, 83% saturated), and fixes by tuning alpha or by an EMA scale. 2507.13338 shows Muon plus spectral soft cap bounds weight norm by construction and trains without LayerNorm. Their combination (cap the spectral norm so the erf argument stays in range, rather than rescaling activations) was not found in either paper's discussion or my searches. Cheap to test in a small GPT. Tentative.

**Gap E. Operating-point diagnostics from the hyperbolic audit applied to hyperbolic layers outside vision-language models.**
2607.05268 defines `u = sqrt(c*rho)` and `H(u) = u/asinh(u)` and finds three hyperbolic VLM families sit at u <= 0.37 (effectively Euclidean) with a low-curvature shortcut in the entailment loss. Hyperbolic attention, hyperbolic LM embeddings and mixed-curvature GNN claims were not audited this way in anything I found. A layer with hierarchical synthetic data (trees) and a trained curvature, reporting u and a shuffle-controlled radial test, would tell whether the gain ever arises from geometry. Searched: "hyperbolic", "mixed-curvature", "Riemannian layers". Tentative.

**Gap F. CliffordNet-style wedge interaction vs a plain gated-bilinear (GLU) control, and at LM or ImageNet scale.**
2601.06793 credits the geometric product (dot plus wedge) for 76-78% CIFAR-100 at 1.4-3M params, with no ablation that replaces the dot/wedge pair by a GLU/SwiGLU on the same shifted channels. 2607.06634 shows GA only helps for chained group operations, which CIFAR classification does not obviously contain. So the effect may be "multiplicative interaction plus depthwise conv", not "geometric algebra". Searched: "Clifford", "geometric algebra transformer", "CliffordNet" ; nothing found replicating CliffordNet. Tentative.

**Gap G. Tropical (max-plus) attention versus sparse/hard-max attention (ASEntmax) on the same synthetic length-extrapolation tasks, and at LM scale.**
Tropical Attention (2505.17190) and ASEntmax (2506.16640) both claim sharp attention fixes OOD length failure on retrieval/sort/copy-like tasks, using different task suites and different baselines. No LM-scale result for tropical attention and no direct comparison found. Tropical aggregation is a hard max over `S_ij + V_j`, the alpha->infinity limit of an entmax-like selection in a different semiring. Searched: "tropical", "tropical attention", "entmax". Tentative.

**Gap H (weaker). Koopman and Hodge/sheaf Laplacian layers in standard deep learning stacks.**
My Koopman and Hodge searches returned control/dynamical-systems and graph papers, not drop-in replacements for FFN/attention in LM or vision backbones. Either the idea is not there yet, or my queries missed it. Low confidence either way; I would not spend Colab time here without a deeper search.

### Queries used (all alphaXiv discover_papers)
1. KAN / Kolmogorov-Arnold / rational / Chebyshev (after 2024-04)
2. doubly stochastic / Sinkhorn / entmax / OT attention (after 2024-01)
3. Koopman / tropical / Clifford / hyperbolic (after 2024-01)
4. Dynamic Tanh / normalization-free / Lipschitz / orthogonal layer (after 2024-06)
5. stick-breaking / SSMax / differential transformer / sigmoid attention (after 2024-01)
6. Monarch / tensor network / low-rank / KAT / rational (after 2024-01)
7. DyT critiques / replication (after 2025-03)
8. Lipschitz / orthogonal / Newton-Schulz layers; tropical + sheaf/Hodge; Pade + Clifford + Koopman (3 further queries)

Not covered in depth: stick-breaking attention, sigmoid attention, Differential Transformer originals, nGPT (2608.01284 Training nGPT appeared but was not read), Monarch/tensor-train layer comparisons (only snippets), Muon-as-layer theory.

---

## 4. Most promising lead

A matched-compute shootout of length-aware attention normalizers in a small nanoGPT (10-60M params, Colab-feasible): softmax vs SSMax vs alpha-entmax vs ASEntmax-style `(s log n)` scaling vs GOAT-style explicit log-prior, with a single positional scheme and tuned learning rate per method, 3+ seeds, evaluated at 1x/2x/4x/8x training length. Why: one-line implementations, three independent theoretical motivations that predict the same symptom, claims from three groups never in one table (Gap A), and the failure modes (softpick init, 2610.05488; DyT/SwiGLU, 2604.23434) tell us what controls to include. Runner-up: Derf/DyT with Muon plus a spectral cap (Gap D), a cheap and clean interaction test.
