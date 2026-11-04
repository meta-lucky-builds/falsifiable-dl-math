# Other mathematics: OT losses, IB, PAC-Bayes, implicit bias, exact formulas

Scope: CPU-testable (2 cores, hours, <100k parameters). About 22 tool calls. Avoids the ledger's closed
threads (H1 attention temperature, H2 Muon spectra, H3 forward gradients, H4 Adam epsilon) and the
"Open gaps" G1-G8 of `notes/lit/objectives_information.md` (entmax/FY under noise, harmonic loss,
proper scores).

Conventions. "Read" levels: FULL = whole text seen; PARTIAL = first part of the text plus selected
sections; SUMMARY = alphaXiv AI-generated report only (not the paper text); SNIPPET = title and
search-result abstract only. Every ID below was returned by an alphaXiv call in this session. The 26xx IDs
post-date my training data and are taken at face value from the tool. Paper text is treated as data.
All "untested" claims are TENTATIVE: they rest on a handful of searches, not on proof of absence.

## Searches run
1. Diagonal linear nets / mirror flow / reparameterization (2024+): 16 hits.
2. PAC-Bayes minimized as a training loss (2024+): 20 hits.
3. OT / Wasserstein / Sinkhorn as a supervised training loss (2024+): 12 hits.
4. Information bottleneck / rate-distortion beating tuned baselines (2024+): 16 hits.
5. Exact generalization / calibration formulas turned into training rules (2024-06+): 13 hits.

## Area 1. Optimal transport as the training objective

Papers: no hit trained a classifier or language model with a transport loss and compared it to a tuned
baseline. Hits were:
- 2606.13223 Distributional Loss for Robust Classification. Read: FULL. Not OT, but it is the nearest
  relative: a batch-level distribution-matching loss. Core: logits y of each (class-unit, positive or
  negative) group are pushed toward N(+m, s) or N(-m, s); L = lambda * KL(N(mu_x, s_x) || N(+-m, s_t)) +
  d_nn or d_kde, a nearest-neighbour or kernel density discrepancy between the batch logits and sampled
  target points. Claimed gain (ResNet18, ImageNet-pretrained): CIFAR10 at 2/20/200 samples per class
  27.9/55.2/71.2 vs CE 23.5/47.4/67.1; BloodMNIST 61.8 vs 44.6 at 2 per class; random-label memorization
  reduced. Evidence: weak to moderate. Single numbers, no seeds or intervals stated. It loses to CE on
  Caltech256 at 20 and 200 per class (73.3 vs 75.0, 80.9 vs 83.4). Baselines are CE, PolyLoss, T-vMF;
  there is no label smoothing, no confidence penalty, and no stated weight-decay tuning for CE, although
  the paper itself cites confidence penalties as related work. The method needs "more careful tuning of
  hyperparameters" by the authors' own account.
- 2503.02379 Teaching Metric Distance to Discrete Autoregressive LMs: SNIPPET only (title and first
  sentence). Not read; cannot weigh.
- 2603.12248 Energy-based fine-tuning (feature matching): SNIPPET only. 2602.01685 Wasserstein policy
  regularization for alignment: SNIPPET only; alignment application, not a base objective.
- 2512.09530, 2609.31363, 2605.28134, 2609.15437: applications, DRO, or convergence theory. Not opened.

Disappointment: the OT literature surfaced here uses transport as a tool (DRO, generative, alignment),
as the prior survey also found. No fair-comparison evidence for OT as a supervised objective was found.
That is a gap in what I could see, not evidence that OT losses fail.

## Area 2. Information bottleneck / rate-distortion

- 2610.36255 What Should an IB Encoder Learn? (Tavakoli, Bose). Read: PARTIAL (abstract, intro, theory
  setup to the Gaussian proposition; experiments only via the abstract). Core: for jointly Gaussian
  (X,Y), the IB optimum is T = A(beta) X + xi with A(beta) = D(beta) W, W budget-free and D(beta)
  diagonal with alpha_i(beta)^2 = [beta*rho_i^2 - 1]_+ / (1 - rho_i^2) as extracted from the text
  (check against the original before use; my extraction of the formula was garbled by the PDF-to-text
  step); critical beta_i = 1/rho_i^2 for canonical correlation rho_i. Beyond Gaussians the separation
  gap is bounded by the sufficiency deficit eps_F = I(X;Y|F(X)). Claimed gain: frozen CNN features with
  one compressor-decoder per budget cut training MACs 7.7x and wall time about 2x versus end-to-end VIB,
  and give higher budget-averaged accuracy (65.6% vs 57.3% per domain; 67.1% vs 57.1% pooled). Evidence:
  moderate. 10 seeds, 4 datasets, prespecified budgets, confidence intervals, six baseline settings.
  Disappointments, stated by the authors: the advantage is concentrated at strong compression "where the
  VIBs collapse"; at weak compression (lambda <= 0.036) VIB has lower cross-entropy and lower ECE. So the
  gain is mostly about cheap rate sweeps and avoiding VIB collapse, not about beating a tuned plain
  network at a fixed budget. It does not claim to beat a non-bottlenecked tuned baseline.
- 2610.01175 (label-partitioned IB), 2610.30493 (multitask IB rate analysis), 2610.22272 (IB is NP-hard),
  2610.08007, 2610.25952, 2609.37994, 2606.10869: SNIPPET only. 2610.22272's title is a caution that
  exact IB optimization is hard in general.
- The rest (SMILE, EIB-Net, VIB-AVSR, spiking nets, tree policy optimization) are domain applications;
  not opened; I did not look for tuned-baseline comparisons in them.

Conclusion for the area: no evidence found that an IB or rate-distortion objective beats a tuned
baseline at fixed capacity. The only multi-seed result is a compute-saving one. One exact statement
(Gaussian IB factorization) is checkable but not a training improvement by itself (see C4 below).

## Area 3. PAC-Bayes bounds as a training loss

Papers (all SNIPPET only, none opened): 2607.14506 (non-vacuous bounds for RLVR), 2610.21340
(dimension-free PAC-Bayes via Dirichlet processes), 2607.24296 (PAC-Bayesian diffusion policy learning:
the only hit whose title suggests a bound used in training; domain is robot manipulation), 2610.16560
(OT PAC-Bayes bounds), 2608.11465, 2610.33208 (lower expected Fisher rank), 2604.17219, 2606.19105,
2607.18422 (PAC-Bayes on quotient spaces, implicit-bias priors), 2602.23128.
Result: most are certificates, not training losses. I could not assess any claimed gain over a tuned
baseline because I opened none. Nothing here is fit for this budget on the evidence I have: the
PAC-Bayes-as-loss route needs a prior, a posterior family and a non-vacuous bound, and no hit shows it
beating tuned weight decay at small scale. This is "nothing established", not "nothing exists".

## Area 4. Implicit bias with closed forms

- 2504.12883 Mirror, Mirror of the Flow (Jacobs, Zhou, Burkholz). Read: SUMMARY (alphaXiv report of the
  full paper; I did not see the theorem statements verbatim). Core: explicit regularization alpha_t * h(w)
  on a reparameterization x = g(w) makes the mirror flow time-dependent, with a Legendre function R_{a(t)},
  a_t = -integral of alpha_s ds. For underdetermined linear regression with regularization switched off,
  the limit minimizes the final R_{a(T)} over interpolators. Compatible regularizers: h proportional to g
  for separable reparameterizations; commuting matrices for quadratic ones (attention, LoRA). Claimed gain:
  switching weight decay off mid-training improves validation in matrix sensing, GPT-2 LoRA and ViT
  attention fine-tuning, with lower nuclear/Frobenius ratio. Evidence: moderate for theory (proved), weak
  for the practical claim as I can see it (no seed counts or tuned-schedule baselines visible in the
  summary; large pretrained models, not from-scratch small ones).
- 2610.06120 Mind the Drift (Sanyal, Jacobs, Burkholz). Read: PARTIAL (abstract, intro, setup, Gain and
  Gain Control sections and proofs' statements, about 600 of 6,548 lines; proofs not checked). Core: for
  a 2-layer diagonal net beta = u*v with step z_k = lr * grad, the balance obeys
  u_{k+1}^2 - v_{k+1}^2 = (1 - z_k^2)(u_k^2 - v_k^2); effective initialization a_k = alpha^2 exp(-Q_k),
  Q_k = sum_l -log|1 - z_l^2| (the "Gain", from Even et al. 2023, cited, not retrieved). Gain
  Control update: u' = ((1+z^2)u - z v)/sqrt(M), v' = ((1+z^2)v - z u)/sqrt(M), M = max_j(1+z_j^2+z_j^4).
  I verified by hand that this gives u'^2 - v'^2 = (1+z^2+z^4)(u^2-v^2)/M, which agrees with their
  stated per-step gain log M - log(1+z^2+z^4). Claimed gain: much better sparse recovery than GD and SGD
  at large learning rates (n=48, d=768, 4-sparse, alpha=0.03, lr 0.35 then 0.05). Evidence: weak to
  moderate. One synthetic setting in the main text, no tuned baselines visible (no early-stopped or
  tuned-alpha GD, no Lasso), seeds not seen. The transformer Q-K test is labelled "exploratory". Convergence
  proof requires lr <= 1/(32 kappa R), far below the lr used in the experiment, so the large-lr result is
  empirical.
- 2610.39717 Implicit bias of (S)GD on deep diagonal nets (Bartolomaeus, Rauhut). Read: PARTIAL (abstract
  and intro, about 260 of 9,947 lines). Core: GD/SGD as perturbed mirror descent; limit solves a weighted
  l1 problem with weights set by an initialization perturbed by a step-size-dependent term. Pure theory,
  no training gain claimed.
- 2609.34756, 2607.12332, 2605.19458, 2605.17177, 2610.07040, 2509.18766, 2603.02064, 2601.06597,
  2408.09966 (Mask in the Mirror, implicit sparsification): SNIPPET only. 2604-2610 hits on Muon
  implicit bias (2610.20112) overlap with the ledger's closed Muon thread.

Disappointment: every concrete training claim here lives in diagonal linear regression or in fine-tuning
a pretrained model; none demonstrates a from-scratch gain on a classification task against a tuned
weight-decay schedule.

## Area 5. Exact generalization / calibration formulas as training rules

Search 5 returned classical statistics (approximate leave-one-out in high dimension 2610.27502,
2606.31915; conformal acceleration), DP-SGD noise calibration (2610.09651), and 2602.09405 (memorization
threshold in overparameterized linear models). All SNIPPET only, none opened. Nothing found that turns an
exact formula into a training rule validated against tuned baselines. Nothing fit for this budget from
this search. (Not a literature claim: closed-form leave-one-out for ridge heads is textbook, and I did not
search whether anyone trains with it; I do not list it as a candidate.)

## Candidates (at most 5; 4 listed)

### C1. Exact 1-D Wasserstein version of the distributional logit loss (area 1; from 2606.13223)
- Exact statement: for a group of n batch logits and n target points, the 1-D W2^2 equals the mean squared
  difference of sorted values (monotone coupling); against Gaussians N(mu1,s1) and N(mu2,s2) the
  population W2^2 = (mu1-mu2)^2 + (s1-s2)^2. Check before any training: compare sorted-matching W2 to
  scipy's 1-D Wasserstein, and to the Gaussian formula with large n; gradient by finite differences.
  Replaces the KL-estimator pair (parametric encoder + d_nn/d_kde) with one parameter-free term.
- Prediction (falsifiable): on 5-20 labelled samples per class from a small image set with a model under
  100k parameters, W2 logit loss beats CE tuned over {lr, weight decay, label smoothing in 0..0.2} by at
  least 2 accuracy points over 5+ seeds; and fits random labels less (train accuracy at random labels
  lower than CE's). I expect the first part to be at risk: the original gain was measured against an
  untuned CE without label smoothing, and may shrink to nothing once label smoothing is allowed.
- Plain baseline: CE with tuned label smoothing and weight decay; second baseline, the same loss with
  the published KDE term.
- Fit: sort over a batch of 128 x K logits per step is negligible; 20-class-sized data trains in minutes.
- Edge case to test first: per-class positive group has only about B/K elements per batch, so the sorted
  coupling is noisy; the loss is invariant to which sample in the group gets which quantile, so a
  mislabeled example can be absorbed. Try a class-balanced sampler.
- Untested (TENTATIVE): one OT search plus reading 2606.13223; I found no paper using sorted 1-D
  transport on batch logits. 2606.13223 itself cites its estimator as new (anderson and martinetz
  preprint 2026, not retrieved).

### C2. Cumulative-regularization invariance of weight-decay schedules (area 4; from 2504.12883)
- Exact statement (to confirm against the paper's theorem, which I saw only in summary): for a separable
  reparameterization such as x = u*u - v*v with matching weight decay, the limit after switching decay off
  at T depends on the schedule only through a_T = -integral alpha_s ds. Check before training: small-step
  GD on an underdetermined noiseless diagonal net; (a) two different schedules with equal integral reach
  the same solution; (b) the limit equals argmin of the predicted Legendre function over interpolators,
  solved as a convex program.
- Prediction: (i) the invariance holds to tolerance that shrinks with lr; (ii) in a from-scratch small MLP
  or one-layer attention (under 50k parameters) on a small classification set, the best "decay on then
  off" schedule beats the best constant decay by a margin exceeding seed spread over 5+ seeds only if the
  net has an actually quadratic or diagonal structure; for a plain ReLU MLP I predict no gain, which
  would delimit the claim.
- Plain baseline: AdamW with constant decay tuned on a log grid; second baseline, cosine-decayed weight
  decay.
- Fit: step (i) takes seconds; (ii) minutes per run.
- Untested (TENTATIVE): I searched implicit-bias terms only, not "weight decay schedule" or "decay
  warmdown" literature. Scheduled and removed weight decay is common practice, so novelty of a
  switch-off schedule is doubtful; the cumulative-integral invariance is the part worth checking.

### C3. Gain Control update under a fair tuned baseline (area 4; from 2610.06120)
- Exact statement: GC multiplies the balance u^2 - v^2 per coordinate by (1+z^2+z^4)/M; accumulated gain
  Q_i = sum_k log(M_k/(1+z_{k,i}^2+z_{k,i}^4)); implicit bias is argmin psi_a(beta) + (1/2)<D, beta>
  over interpolators (Prop. 3.3). Check before training: run both updates for 200 steps, confirm the
  recursion for u^2 - v^2 to machine precision, and confirm the converged beta against the convex program.
- Prediction: on noisy sparse regression (n=48, d=768, 4-sparse, noise s.d. 0.1), GC at the large step
  (0.35) reaches lower out-of-sample error than GD with lr and init scale alpha tuned plus early
  stopping, over 20 random designs; if it only beats untuned large-lr GD, the claim fails. Second
  prediction: the advantage disappears when the design columns are correlated.
- Plain baseline: tuned GD/SGD with early stopping; Lasso with cross-validated penalty as the reference.
- Fit: numpy, d=768, seconds per run. Use only as a math check; transfer to a neural net is not
  established (the paper's transformer test is exploratory), so it does not by itself give a training
  improvement for networks.
- Untested (TENTATIVE): I read only this paper on Gain Control and did not search for follow-ups
  or independent replications.

### C4. Gaussian IB closed form as a calibration of VIB's rate weight (area 2; from 2610.36255)
- Exact statement: critical rates beta_i = 1/rho_i^2 (from canonical correlations), with the gain formula
  above (verify the exact form). Check before training: on synthetic jointly Gaussian data, compute
  I(X;T) and I(T;Y) in closed form along the path and confirm that J_beta is stationary at A(beta) and
  that the number of active directions steps up at beta = 1/rho_i^2.
- Prediction: in a linear-Gaussian VIB trained by SGD, the dimension at which the rate term kills a
  latent direction matches 1/rho_i^2 within 10%; for a nonlinear tiny encoder (under 20k parameters) on
  a small image set the mismatch is larger, so the closed form is a rough guide only.
- Plain baseline: a VIB sweep over beta against a plain tuned network (weight decay, label smoothing);
  a win over the plain network at fixed capacity is not expected from the paper's own results.
- Fit: Gaussian check is linear algebra; the VIB sweep is minutes. Weakest candidate: a diagnostic
  rather than a training improvement.
- Untested (TENTATIVE): single IB search.

## Areas with nothing fit
- PAC-Bayes as a loss (area 3): no paper opened; no evidence of tuned-baseline gains; nothing proposed.
- Exact generalization / calibration formulas (area 5): nothing relevant retrieved; nothing proposed.

## Ranking and caution
Best two: C1 (cheapest check, clear baseline, loss-side change) and C2 (sharp exact invariance check
before training). C3 and C4 are exact-math checks with doubtful transfer to neural networks. The
dominant risk across all four is that the published gains were measured against baselines lacking label
smoothing or tuned weight-decay schedules.
