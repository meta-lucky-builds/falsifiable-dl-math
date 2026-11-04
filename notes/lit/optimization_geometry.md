# Literature scout: optimization and training geometry

Scout thread: non-Euclidean / curvature / edge-of-stability / scale-invariance / non-backprop. Date of scan: 2026-11-04.
Source: alphaXiv (discover_papers + get_paper_content). Every ID below was returned by the tool; "read" means I pulled the content report or full text, "abstract" means I only read the abstract/intro (long PDFs were saved and I read the head). Nothing here is from memory.
Coverage caveat: about 10 discovery queries and about 20 papers opened. This is a scout pass, not an exhaustive review. Most 2026 papers are single-group preprints that nobody has had time to replicate.

## 0. Cross-cutting reading of the evidence (the "so what")

1. Five independent groups converge on one object: the **effective learning rate (ELR)** = LR / parameter norm (or angular step). Hyperball (2606.16899), HyperTransfer (2609.07017), ELR-collapse (2608.24814), AIR (2610.40362) and the exact schedule law (2609.09116) all say that a large part of what weight decay / weight-norm / hyperball "does" is shape the ELR trajectory. This is the most replicated finding in the thread and the cheapest to test.
2. The **geometry story for Muon is shaky**. Random or inverted singular spectra match Muon (2605.11181), spectral flattening alone does not drive the loss (2610.10965), and what remains is step-size stability. Do not pick a method because its "norm" is elegant.
3. **Tuned-baseline discipline kills most speedups**: <=1.4x at 100M, ~1.1x at 1.2B (2509.02046); the winner flips with batch size (2610.08975) and with horizon (2609.04577); wall-clock at small scale can reverse (2610.21555). Any test I run must tune AdamW properly and report at several batch sizes.

## 1. Papers (20)

Format: ID | core formula | claimed gain | evidence strength.

1. **2509.02046** Fantastic Pretraining Optimizers and Where to Find Them (Wen, Hall, Ma, Liang). READ.
   - Formula: none; a three-phase coordinate-descent tuning protocol applied to 11 optimizers vs AdamW (130M-1.2B, 1x-16x Chinchilla).
   - Gain: true speedup over tuned AdamW <=1.4x (matrix optimizers, <520M), shrinking to ~1.1x at 1.2B; tuning only AdamW's LR in a GPT-3 recipe gave 2x.
   - Evidence: strongest in the set (tuned baselines, scale sweep, open WandB). Same group as 2606.16899 and 2610.08975, so those three are not independent of each other. Seed counts not stated in the report I retrieved.

2. **2606.16899** Fantastic Optimizers II: Hyperball Optimization (Wen, Dang, Lyu, Ma, Liang). READ.
   - Formula: u_hat = u/||u||_F; W~ = W - eta*R*u_hat; W_{t+1} = R * W~/||W~||_F (fixed Frobenius radius R = ||W_0||, any base optimizer; embeddings/norm gains excluded).
   - Gain: MuonH 20-30% token-equivalent speedup over AdamW/MuonW up to 1.2B (MuonW alone ~10%); 8B/159B tokens: 0.04 lower loss; LR drift across depth/width 1.4x vs 3-4x for AdamW/MuonW.
   - Evidence: moderate-strong on scale; seeds not stated; baselines "decoupled weight decay" were tuned by the same group. Partly corroborated by 2609.07017 and 2608.24814 (below) but those two also say much of the gain is a schedule effect.

3. **2609.07017** HyperTransfer (Yuan et al., StepFun). READ.
   - Formula: for scale-invariant w, match angular step: eta_eff = eta*||u|| / ((1 - eta*lambda)*||w||); hyperball LR eta^H_t = eta_t*||u|| / ((1-eta_t*lambda)*s_t) with proxy norm s_t tracking ||w_t||.
   - Gain: shows Base optimizer and Hyperball are dynamically equivalent once effective LR schedules match; nominal "decay to 0.1x" gives an effective final/peak ratio of only 4.7e-2 for Muon. Even after matching, MuonH still reaches 3.221 vs 3.238 for tuned Muon (124M).
   - Evidence: GPT-2 124M, one seed (1234) stated; one 500M/70B-token confirmation. The 0.017 residual gap is the honest size of the "hyperball is special" effect.

4. **2608.24814** Effective Learning Rate Governs Loss Dynamics in LM Pretraining (Liu et al., PKU/Ant). READ.
   - Formula: ELR_k = eta_k / ||W_k||_F; train with eta_k = eta_eff_k * ||W_k||_F to follow any prescribed ELR schedule.
   - Gain: loss curves collapse to mean abs gap 1.8-4.1e-3 (seed noise 1.1-1.6e-2) across AdamW/Muon/Signum, Llama/Qwen3-MoE/KDA, 100M-1B; an ELR-parameterised functional scaling law transfers from weight decay to Hyperball (RMSE 0.021 vs 0.251 for LR-parameterised).
   - Evidence: good (multiple archs/optimizers, calibrated against seed noise). Collapse precision degrades without QK-norm and with fixed RMSNorm gains (counter-intuitive, unexplained).

5. **2610.40362** AIR: Effective LR Scheduling in Weight-Normalised Training (Yu et al., ByteDance Seed/PKU). ABSTRACT + intro read.
   - Formula: ELR = eta/r; schedule = theoretically derived 1/sqrt(t) decay plus affine adjustment to hit prescribed peak and terminal LR.
   - Gain: ~0.01 lower terminal train loss than cosine at matched cumulative LR (128B-token MoE); same ELR trajectory reproduces dynamics under weight decay.
   - Evidence: one large run family; gain is small. Independent group from #2-#4, same conclusion.

6. **2609.09116** When Does Scale-Invariant Optimization Become Unstable? Exact Schedule Law (Amin, Chang, Khanna, Purdue). READ.
   - Formula (exact, per step): Phi_t = eta_t/(a_t r_t^2), Phi_{t+1} = B_t*Phi_t / (1 + Phi_t^2 ||g_bar_t||^2), B_t = (eta_{t+1}/eta_t)/(a_t*a_{t+1}); Adam (eps->0) self-quenches linearly (nu=0), SGD/SGDM quadratically.
   - Gain: diagnostic, not a speedup. Enforcing B_t = B on CIFAR-10 gives a sharp peak at B ~ 1 (2% perturbation cost >20 accuracy points in one experiment).
   - Evidence: exact algebra (residuals ~1e-15 in float64, ~1e-6 in float32); GPT-2 LayerNorm blocks show only intermediate fit for Adam. Theorem excludes finite eps and decoupled weight decay.

7. **2606.25971** Decoupling Magnitude and Direction of Weight Vectors (Hagele, Hernandez-Cano, Kosson, Jaggi, EPFL). READ.
   - Formula: W = diag(g_row) * W_hat * diag(g_col), ||W_hat||_F fixed, W_hat updated by any base optimizer then reprojected, gains by Adam at separate LR.
   - Gain: MuonMD ~2x compute efficiency vs AdamW (CI [1.94, 2.11]) on DeepSeekMoE-style 1.2B-6.7B; no weight decay, no warmup needed; LR transfer across width and depth.
   - Evidence: "well-tuned" baselines claimed; seeds not stated; 2x is a scaling-fit estimate. Also reports that part of nGPT's advantage was a scaling trick for 1D vectors, not the sphere.

8. **2605.11181** Muon is Not That Special: Random or Inverted Spectra Work Just as Well (Shumaylov et al.). READ.
   - Formula: Freon update (G G^T)^(-c) G, c=0 is SGD, c=1/2 is Muon, c>1/2 is outside any proper Schatten norm; Kaon replaces singular values by chaotic noise.
   - Gain: best c ~ 2/3 on GPT-2 124M; Kaon (random spectrum) matches Muon on NanoGPT; success attributed to a nearly constant optimal step size rather than ideal geometry.
   - Evidence: small scale (124M, WikiText-2), seed count not stated in the report. Supported qualitatively by 2610.10965. Threatens the "Muon = steepest descent in spectral norm" explanation.

9. **2609.07597** Quadratic Spectral Descent (Zhang, Sun, Liu). READ.
   - Formula: D* = argmin_{||D||_2<=rho} <G,D> + (eta/2) vec(D)^T C vec(D), C ~ K-FAC (B D A), solved by 3 Frank-Wolfe steps whose atoms are matrix-sign solutions; first atom is exactly Muon.
   - Gain: val loss 3.3158 vs Muon 3.3316 (GPT-124M, FineWeb); 8.5% / 7.4% wall-clock to matched loss at 124M / 350M.
   - Evidence: moderate; ablated on 124M, reused on 350M; curvature inflation tau=1.5 is empirical; theory covers only the surrogate. Not replicated.

10. **2509.03378** KL-Shampoo / KL-SOAP (Lin, Lowe, Dangel, Eschenhagen, Xu, Grosse). READ.
   - Formula: fit Kronecker preconditioner S_a (x) S_b by KL divergence; EMA updates S_a <- (1-b)S_a + (b/d_b) G S_b^-1 G^T, S_b <- (1-b)S_b + (b/d_a) G^T S_a^-1 G; eigenvalue EMA in a stale QR eigenbasis.
   - Gain: beats Shampoo and SOAP on NanoGPT-123M, RWKV7-162M, Llama-134M, MoE-227M, and Muon/SOAP on Llama3-450M; no Adam grafting needed; less memory than SOAP.
   - Evidence: moderate (4 archs, sweeps), same-time-budget curves. Independent use: KL-SOAP-H gave the fastest loss-at-step (3.2780 in 3125 steps) in the Hyperball paper's modded-nanogpt comparison (#2), though that is a different group reusing it, not a replication of the original claim.

11. **2609.04577** Optimizer Memory Schedules for Outscaling the Overtraining Axis (Everett, Qiu). READ.
   - Formula: ADANA/DANA momentum whose memory grows with time (kappa=0.85 -> predicted equivalent-OT exponent 2 - kappa ~ 1.15); log-time weight decay; optimal WD coefficient ~ sqrt(OT factor).
   - Gain: Muon / SOAP hold a constant 1.3-1.7x token multiplier over AdamW; ADANA's advantage grows with horizon and overtakes Muon by 256x OT (51M model).
   - Evidence: LR swept per (size, OT); 51M-253M models, OT up to 256x (small) / 8x (253M); momentum constants mostly fixed; seeds not stated. The sqrt(f) WD rule matches earlier theory. Small models only.

12. **2512.04299** When Do Spectral Gradient Updates Help in Deep Learning? (Davis, Drusvyatskiy). READ.
   - Formula: spectral step beats Euclidean on a block when nr(G) = ||G||_*^2/||G||_F^2 >= st(A) = ||A||_F^2/||A||_op^2 (stable rank of incoming activations).
   - Gain: a layerwise, checkable predictor; ReLU mean-spike keeps st(A) <= pi at init; SwiGLU-centered case violates it.
   - Evidence: theory plus synthetic and NanoGPT diagnostics; not a benchmark of speedups.

13. **2610.10965** Spectrally Targeted Muon (Goyal, Ramkumar). READ.
   - Formula: orthogonalise only singular values above (or below) tau using P_top/bot = (I +/- sign(M^T M - tau^2 I))/2.
   - Gain: none as a method; diagnostic. Orthogonalising everything except the few largest singular values is within 0.03 of Muon while touching ~13% of momentum magnitude; top-only falls behind even normalised SGD (3.425-3.875 vs Muon 3.280, AdamW+warmup 3.396).
   - Evidence: NanoGPT speedrun, 2-3 seeds, 20 seeds on CIFAR-10; student-scale workshop paper. Useful for what NOT to build.

14. **2610.08975** The Best Optimizer Depends on Batch Size (Dang, Wen, Malladi). ABSTRACT + intro read.
   - Formula: noisy quadratic model; per-direction batch-size LR exponent moves from sqrt to linear as curvature-to-noise ratio rises; Newton-vs-SGD bias-variance crossover (Thm 5.1).
   - Gain: diagnostic. No Muon scaling rule works across settings; SOAP best 128K-1M tokens, Shampoo best at 2M; keeping small-batch updates in the sharpest 0.001% of directions removes up to 59.5% of the loss gap after a 16x batch increase.
   - Evidence: tuned (coordinate descent) at every batch size, Modded-NanoGPT benchmark; same group as #1/#2.

15. **2610.36778** Does LLM Pre-Training Typically Occur at the Edge of Stability? (Cai, Wen, Huang, Liu, Ma, Lyu). ABSTRACT read.
   - Formula: "Edge of Convexity": persistent negative curvature makes the quadratic model diverge at any LR while real training is stable; generalised EoS defined by whether raising LR diverges the quadratic dynamics but not the real ones.
   - Gain: diagnostic. EoC always seen up to 787M; EoS prevalent but vanishes when LR is small (after decay) or batch is below a threshold correlated with the critical batch size.
   - Evidence: several runs up to 787M, single paper.

16. **2609.34915** Muon Sublates the Edge of Stability (Chen, Zhao, Xu, Liu). ABSTRACT read.
   - Formula: for stochastic no-momentum Muon, coherence-corrected loss-neutral boundary 2*rho_b/eta, separate from temporal-direction alignment.
   - Gain: theory/diagnostic; 130M and 1B Llama-like runs show boundary tracking but weak or partial update reversal.
   - Evidence: small; one group.

17. **2602.05813** Where Does Warm-Up Come From? Adaptive Scheduling for Norm-Constrained Optimizers (Riabinin et al.). READ.
   - Formula: curvature K(x) = K0 + K1*D + K2*D^2 in suboptimality gap D = f - f*; LR schedule eta(D) = D/(K0 + K1 D + K2 D^2), which rises then falls; constants fixed from peak LR, floor LR and a target shape.
   - Gain: matches or beats best hand-tuned warmup for Muon/Lion/normSGD on LLaMA 124M/210M and Swin-B; most helpful at batch 32, prevents Lion divergence at short warmup.
   - Evidence: needs a target-loss guess f* (robust over [3,4] for 124M); small scale; no seeds in report.

18. **2610.21555** When Does Muon Actually Win? Paired Data/Time/Tuning-Budget Benchmark (Akhtar et al.). READ in full.
   - Formula: none (benchmark, Newton-Schulz coefficients 3.4445, -4.7750, 2.0315).
   - Gain/finding: 5 matched seeds, 23-candidate tuning; Muon wins equal-data on all 4 workloads (3-22% val-AUC) and all 20 paired test comparisons; on language, AdamW wins equal-time by 3.4-7.0% (17.7M and 33.6M models, one A100, reference kernels); optimizer step is 56-70% of Muon training time; one-candidate expected regret 13.7% vs 7.5%.
   - Evidence: the most Colab-relevant paper here: small models, single GPU, frozen test, multiple seeds; but unoptimised kernels and 1-seed tuning selection.

19. **2510.13680** Adam or Gauss-Newton? Basis Alignment and SGD Noise (Liu, Bansal, Morwani, Alvarez-Melis, Vyas, Kakade). READ.
   - Formula: diagonal preconditioners D_ii = (E[(u_i^T g)^2])^(-1/2) (Adam) vs (u_i^T H_GN u_i)^p, p in {-1, -1/2}, in identity vs GN-eigenbasis.
   - Gain/finding: with small batches Adam behaves like GN^(-1/2) irrespective of basis; in a reparameterised attention-like logistic problem Adam beats GN^-1 even in its ideal eigenbasis because GN^-1 steps are unbounded.
   - Evidence: proofs on linear/logistic regression plus small MLP and a synthetic selection task; no LLM pretraining.

20. **2610.26283** Trust Your Targets: Is Target Propagation Block-Diagonal Gauss-Newton? (Castiglione, Buckley, Simpson). ABSTRACT read.
   - Formula: TP rederived as a trust-region step under an implicit norm; CGTP solves layer-local least-squares subproblems with conjugate gradient; K-FAC is the exact special case with batch-mean Jacobians.
   - Gain: matches Adam and K-FAC accuracy on MNIST/Fashion-MNIST MLPs with ~10x fewer backward-pass equivalents.
   - Evidence: weak (tiny MLPs, 1 group). Interesting only as a math link between TP and K-FAC.

## 2. Tried and disappointing (negative or unreplicated)

- **Claimed 1.4-3x optimizer speedups mostly came from weak AdamW baselines.** 2509.02046: tuning AdamW LR alone gave 2x; best true speedup <=1.4x, ~1.1x at 1.2B, no clear downstream gain at 1.2B.
- **Muon geometry (steepest descent in spectral norm) does not explain Muon.** 2605.11181 (random spectrum Kaon matches Muon; best Freon c ~2/3 outside any proper norm) and 2610.10965 (flattening the large singular values is cheap and not what drives loss; top-only orthogonalisation worse than normalised SGD). Caveat: both small scale.
- **No batch-size scaling rule for Muon is reliable.** 2610.08975: SDE-derived and bound-derived rules both lose to "leave hyperparameters unchanged" at the largest batch; optimizer ranking flips with batch size (SOAP -> Shampoo).
- **Muon at Colab scale, wall-clock.** 2610.21555: on 17.7M and 33.6M language models AdamW wins equal-time by 3.4-7.0% because of orthogonalisation cost (reference kernels). Muon needs more tuning (higher expected regret). Do not claim a Muon win without equal-time curves.
- **Muon's pretraining edge can reverse after fine-tuning.** 2611.03633 (abstract + intro read only; submitted 2026-11-02, nothing yet replicates it): at large pretraining token/parameter ratios AdamW-pretrained models give a better learning-forgetting frontier (OLMo 30M-600M, ~600 pretraining runs, ~29,000 fine-tuning runs per abstract). Explained by a margin-to-spread ratio; proven only in a linear associative memory.
- **Quadratic-model / EoS-derived rules do not carry over to LLM pretraining.** 2610.36778: Edge of Convexity makes the quadratic model diverge at any LR while real training is stable; EoS vanishes after LR decay or at small batch. 2609.34915: Muon breaks the single-edge picture. So "set LR = 2/sharpness" style controllers have weak support. (Central flow, Cohen et al. 2025, is only cited inside retrieved papers; I did not retrieve it.)
- **Second-order is not automatically better than Adam.** 2510.13680: GN^-1 can lose to Adam even in its ideal eigenbasis in an attention-like objective, and Adam tracks GN^-1/2 under small-batch noise; so full Newton-like preconditioning needs damping/trust regions (QSD's tau=1.5 inflation is an empirical version of this).
- **Local-learning alternatives still lag backprop.** 2609.33240 proves a Forward-Forward error floor from simultaneous layer updates (moving input distribution) plus exponential similarity-kernel collapse with depth; gap to BP grows with depth. 2610.26283's CGTP evidence is MNIST MLPs only.
- **Hyperball "special geometry" is smaller than it first looks.** 2609.07017 and 2608.24814: once effective LR schedules match, loss curves nearly coincide (collapse error 1.2e-3 for MuonH vs MuonW); the residual MuonH gain in HyperTransfer is 0.017 loss (single seed). The 20-30% claim (2606.16899) is therefore at least partly an ELR-schedule effect, and the gap between the two readings is untested at Colab scale.
- **Weight-normalised training can lose to weight decay under the wrong schedule.** 2610.40362: WN trails WD throughout under WSD, while it overtakes under cosine; LR schedules designed for unconstrained training are not optimal on the sphere (2606.25971: linear decay beat WSD there).
- **Unreplicated and unread (snippet only, treat as leads not evidence):** 2607.16169 (When does Muon help agentic RL), 2608.03941 (Muon meets Mamba), 2607.20548 (NVIDIA SOAP/Muon at scale), 2610.37225 (Sharpness-Aware Muon), 2607.03998 (Armijo-backtracking sharpness probe / LR safeguard for Adam), 2605.07815 (OrScale: abstract read only, claims 3.8% on a 16B-A3B MoE and "matches or beats" Muon+Moonlight at 125M-1.1B, all hyperparameters inherited, no independent check).

## 3. Open gaps (ALL TENTATIVE)

Method for every gap: I ran about 10 alphaXiv discover_papers queries (non-Euclidean/Muon variants, edge of stability, mirror descent/Bregman/natural gradient, free probability/RMT/initialisation/LR transfer, derived Adam components and schedules, backprop alternatives, benchmarks, sharpness/Hessian, Adam-fixes), opened about 20 papers, and checked whether any retrieved result combined the two ingredients. "No hit" means no hit in those result lists, not a proof that nothing exists; I did not search arXiv listing pages, OpenReview, or Google Scholar.

G1. **ELR as a controller, with the B_t law as stability certificate, on plain AdamW.** Formulas exist: ELR = eta/||W|| (2608.24814), derived 1/sqrt(t) ELR decay (2610.40362), exact B_t law (2609.09116). 2608.24814 does match ELR to a reference run, and 2610.40362 derives one shape, but I found no paper that (a) sets a per-layer ELR schedule from the derived AIR shape in AdamW/Muon without weight decay or hyperball, (b) compares against a properly tuned WD+cosine/WSD baseline at small scale, and (c) logs B_t to see whether runs that stay near B_t ~ 1 are better. Searched: "effective learning rate", "weight-normalized", "scale-invariant schedule", "Hyperball" - all results are the five papers already listed. Also open: the schedule-law theorem excludes finite Adam epsilon and decoupled weight decay (stated limitation in 2609.09116); nobody in the results measured how eps changes the Phi_t recursion in a small transformer.

G2. **QSD's curvature inflation tau vs batch size.** QSD (2609.07597) sets tau=1.5 empirically, tuned at one batch size on 124M. 2610.08975 proves curvature estimation/preconditioning has a noise-dependent bias-variance tradeoff and that rankings flip with batch size. Nobody in the results re-tuned QSD or Muon+K-FAC across batch sizes, or derived tau from the K-FAC estimation variance. Searched: "quadratic spectral descent", "curvature Muon", "Newton-Muon / FISMO / Mousse" appear only as citations inside 2609.07597 (not retrieved by me), no batch-size sweep seen.

G3. **Per-layer adaptive Schatten exponent chosen from measurable diagnostics.** Freon shows c ~ 2/3 best on GPT-2 124M but the optimum fluctuates by batch and layer (2605.11181); Davis and Drusvyatskiy give a per-layer test nr(G) >= st(A) (2512.04299) and themselves suggest monitoring it; 2610.10965 shows small singular values are trainable only once amplified. I saw no paper that feeds nr(G)/st(A) into a per-layer choice of c (or Muon vs AdamW routing). Caveat: 2610.10965's reference list names "Muon_p: Muon with fractional spectral powers" and "From SGD to Muon: adaptive optimization via Schatten-p norms"; I did not retrieve them, so part of this may already be done. Mark as partially covered until those are read.

G4. **Scheduled (growing) momentum memory for Muon, and combined with norm control.** ADANA's growing memory yields an outscaling exponent near 2-kappa and overtakes Muon by 256x overtraining (2609.04577), but that paper only varies Muon's fixed momentum, and does not combine scheduled memory with Hyperball / ELR control (which change the relevant time scale: ELR-time vs step-time). Searched: "optimizer memory schedule", "momentum schedule", "Muon momentum"; only 2609.04577 and 2609.02728 (Polyak/Nesterov in large-batch kernel regression, snippet only) appeared.

G5. **Does norm control change the Muon-vs-AdamW fine-tuning reversal?** 2611.03633 finds AdamW overtakes Muon after fine-tuning at large pretraining budgets and explains it by a margin-to-spread ratio; 2606.16899/2608.24814 say weight decay vs hyperball differ mainly through ELR. Nobody (in these results) tested whether matching ELR between Muon and AdamW pretraining removes the reversal, or whether the margin-to-spread ratio is controlled by ELR. The paper is 2 days old, so absence is expected; treat as a fresh opening, very tentative.

G6. **EoS theory turned into an LR controller for Muon.** 2609.34915 derives a stochastic loss-neutral boundary 2*rho_b/eta for Muon and 2610.36778 characterises when EoS holds; neither proposes an online rule. I saw only a snippet for an Armijo-backtracking sharpness probe used as an LR safeguard for Adam (2607.03998, not read), and no Muon analog. Searched: "edge of stability" x "learning rate rule/controller/schedule" - results are descriptive (central flow, bifurcation, rod flow, etc.). Negative prior: #15 suggests quadratic-model-based rules are fragile.

G7. **Link between target propagation and Muon/K-FAC-style trust regions at transformer-block scale; Forward-Forward error floor vs ELR.** CGTP shows TP = trust-region step with K-FAC as special case (2610.26283, MNIST); FF's floor is caused by inter-layer drift of the input distribution (2609.33240). Layerwise ELR (or hyperball angular steps) bounds per-layer drift per step, so a drift-limited local-learning schedule is a natural combination. No paper in the results combines local-learning methods with norm-controlled or ELR-limited updates. Speculative; I have no evidence it would help.

G8. **Free probability / random-matrix results turned into init or LR rules.** I found analyses of trained weights (Marchenko-Pastur outliers in attention, 2608.07921; spectral phase transitions, 2606.28486; Muon spectral dynamics, 2606.08388 - the last two snippets only) and attention Gaussian equivalence (2610.10033), but no retrieved paper that derives a concrete init or per-layer LR rule for transformers from free-probability calculus, except the stable-rank propagation "calculus" of 2512.04299 (which is a layerwise predictor, not an init rule). One focused query only; low confidence that this is a real gap vs a search miss.

## 4. Suggested small-scale tests (not run)

1. ELR-first schedule: AdamW (and Muon) on a ~10-30M GPT, fixed seed set (>=3), tuned WD+cosine and WSD baselines vs eta_t = ELR(t)*||W_t|| with the AIR 1/sqrt(t)+affine shape; report equal-data and equal-time; also log B_t. Cost: a few lines of code, no new kernels. (G1)
2. Freon c-sweep and nr(G)/st(A) logging on the same model, at 2 batch sizes. (G3, G2)
3. Equal-time reporting always; tune AdamW at least as hard as the challenger (2509.02046, 2610.21555).

## 5. Files and retrieval notes

- Long papers came back as large tool results (2610.36778, 2609.34915, 2610.08975, 2610.40362, 2609.33240, 2611.03633, 2605.07815, 2610.26283). Only their abstracts/intros were read.
- Results are returned by alphaXiv with publication dates up to 2026-11-02. I verified IDs only through tool output.
