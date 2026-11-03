# Prior-art check for H1 (REM attention temperature) and H2 (noise-edge spectral gate)

Date: 2026-11-04. Adversarial brief: try to show both ideas are not new.
Sources used: alphaXiv (discover_papers, get_paper_content), WebSearch, WebFetch of arXiv abstract pages.
NOT used: OpenReview and Google Scholar (not tried). So absence claims below cover alphaXiv + WebSearch only.

Reading depth key (honest, because it limits what I can say):
- FULL = I read the paper text (or a large part of it, with section-level greps).
- REPORT = alphaXiv AI-generated summary of the paper, not the raw text.
- ABS = arXiv abstract page via WebFetch (a small model paraphrased it for me).
- SNIP = search-result snippet only. Do not rely on these.

## Verdicts

H1: PARTLY COVERED. The physics (REM freezing of softmax attention at beta_c = sqrt(2 log n)/sigma),
the sqrt(log n) vs log n debate, and the Gaussian-REM needle-retrieval threshold are all published.
Per-row adaptive temperature exists in heuristic or oracle form. NOT FOUND: the specific closed-form
smaller-root beta*(mu, sigma, n, kappa) with plug-in row estimates, used as a parameter-free attention
and tested on length extrapolation against SSMax/entmax. Closest: Park, 2610.34682.

H2: PARTLY COVERED, and the measurement half is ALREADY DONE with the opposite sign of action.
BulkBoost-Adaptive (2610.07497, 5 Oct 2026) estimates the noise edge for Muon from the difference of two
independent half-batch gradients, claims the same covariance as full-batch sampling noise, thresholds
at the Marchenko-Pastur edge with multiplier 1, and keeps the Frobenius norm. It then BOOSTS the
below-edge bulk instead of removing it. H2's gate-renorm looks like the rho = 0 corner of that family.
NOT FOUND: a paper that actually deletes below-edge modes in Muon and reports results. Several papers
argue it would hurt. Closest: 2610.07497.

## Queries run (every one)

alphaXiv discover_papers:
1. Random Energy Model / softmax attention / freezing transition / temperature / number of tokens.
2. Muon / singular value thresholding / Marchenko-Pastur / gradient noise / orthogonalization.
3. Giorlandino, Goldt, replica, attention, phase transition, context length (sqrt(log n)).
4. Scalable-Softmax / SSMax / length generalization / attention temperature / log n scaling.
5. "From SGD to Muon" / Schatten-p norms / adaptive optimization.
6. Optimal singular value shrinkage / BBP / spiked model applied to gradient or momentum in optimizers.
7. Optimal contrastive/InfoNCE temperature vs number of negatives.
8. Softmax attention needle detection threshold sqrt(2 log n), signal-to-noise, extreme value theory of logits.
9. Gavish-Donoho / OptShrink / singular value shrinkage of gradient matrices in optimizers.
10. Half-batch / two minibatch halves / per-direction SNR gating in Muon.
11. Lucibello-Mezard exponential capacity dense associative memory, REM, softmax retrieval.

WebSearch:
1. Random Energy Model softmax attention temperature sqrt(2 log n) freezing transition.
2. Muon optimizer singular value thresholding noise Marchenko-Pastur edge momentum matrix.
3. Per-query adaptive attention temperature closed form from score mean/std, n, needle in haystack.
4. InfoNCE optimal temperature vs number of negatives theory log N.
5. Dohmatob "Understanding softmax attention layers: exact mean-field analysis on a toy problem".
6. Softmax attention extreme value theory Gumbel max of n Gaussian logits temperature sqrt(2 log n).
7. Muon spectral gating noise-aware singular value truncation "noise edge" split batch.
8. Entropy invariance attention scaling log n length extrapolation.

WebFetch (arXiv abs pages): 2602.08216, 2506.16640, 2501.19399, 2610.31728, 2610.08975, 2605.11181,
2608.25990, 2602.03001, 2609.11169, 2605.19781, 2606.15268, 2501.08570.

Direct retrieval by ID (get_paper_content): 2605.09331, 2609.32861, 2610.07497, 2606.03899, 2610.10965,
2606.08388, 2606.13867, 2605.12697, 2606.07757, 2610.17347, 2610.34682, 2505.17083, 2609.22005,
2602.23057, 2505.24333, 2510.05554, 2410.01104, 2610.22191, 2606.00262, 2605.08505, 2608.08922.

Note: WebSearch alone did NOT surface 2610.07497 (BulkBoost). alphaXiv did. A search-engine-only
check would have wrongly said "not found" for H2.

---

## H1: closest papers

### Park, "Beyond Multiplicity Correction: Sharp Retrieval Thresholds for Attention with Correlated Distractors", 2610.34682 (FULL, head + Prop 15 + refs)
Does: planted-needle softmax retrieval against exponentially many Gaussian distractors; a (generalized)
REM free-energy calculation gives exact signal thresholds for attention mass to approach one
(sqrt(2 alpha) style, in d-scaled units). Proposition 15 gives an exact oracle temperature: the unique
beta-hat where attention entropy equals log m, with min_beta H(beta) as an EVaR identity; no Gaussian or
independence assumption needed. In the Gaussian model beta-hat converges to sqrt(2 alpha_1), the REM critical
value. Shows that no data-dependent temperature fixes count-correction when distractors cluster.
Does not: give a plug-in estimator that works when the target location is unknown (the paper says the oracle
temperature is "not directly usable when the target location is unknown"); no kappa log-odds target, no
smaller-root formula, no length-extrapolation training experiment (its experiments are a frozen-encoder
retrieval test and a DistilBERT logit intervention). Authors state the REM/GREM calculations themselves are
classical, not new.
Relevance: this is the nearest to H1's E1.0 and its "detection limit" property. H1's statement
"no beta works unless mu > sigma sqrt(2 log n)" is the Gaussian REM threshold already proved here.

### Giorlandino and Goldt, "Two failure modes of deep transformers...", 2505.24333, ICLR 2026 (FULL, abstract + Sec 3.1)
Does: maps a row of softmax attention at initialisation onto Derrida's REM; derives a critical inverse
temperature; sets score variance sigma_a^2 = beta^2 log T so scores scale as sqrt(log T); a trainability
diagram for rank collapse vs entropy collapse. Chen et al. (2510.05554) cite it as the source of the
sqrt(log n) scaling.
Does not: handle trained models, per-query estimation, or a needle-log-odds target. It is an initialisation theory.
Relevance: H1's "REM mapping" is not novel. They also state it was identified earlier by Lucibello and
Mezard for associative memories (their claim; I did not retrieve that paper).

### Hayase and Karakida, "A Unified Framework for Critical Scaling of Inverse Temperature in Self-Attention", 2605.12697 (REPORT, plus citations seen in Park)
Does: a deterministic, per-row gap-counting theory. N_n(t) counts keys within gap t of the row max;
Lambda_n = sup_t log N_n(t)/t is the critical (top-two/entropy) scale; any non-degenerate schedule
beta_n ~ (log n)^xi must have xi equal to the growth exponent of Lambda_n. Reports optimal exponents that are
NOT universal: Qwen-7B-Chat perplexity optimum xi in about [2, 2.5]; GPT-124M learned per-position beta fit
xi about 0.40 to 0.50; their row-measured exponent estimates agree with these in all 8 cells.
Does not: propose using the per-row quantity as the attention temperature (it is a diagnostic; adapting
dynamically is listed as an implication, not done); does not use a REM/needle-log-odds formula.
Relevance: (a) it is a per-row, score-measured critical scale, the same spirit as H1's plug-in. (b) It
is evidence that the Gaussian-REM law (xi = 0.5) fits small from-scratch GPT (H1's regime) but not
pretrained large models. Two-sided.

### Veličković et al., "Softmax is not Enough (for Sharp Size Generalisation)", 2410.01104 (REPORT)
Does: proves softmax must disperse as n grows with bounded logits; proposes a per-query adaptive
temperature that maps measured attention entropy to a temperature through a degree-4 polynomial fit
(fit on a max-retrieval task), used inference-time and in fine-tuning.
Does not: derive the temperature from a model of score statistics; gains were small (e.g. 12.4% to 14.0%
at 16,384 items; 81.3% to 82.1% at 256).
Relevance: prior art for "per-query adaptive attention temperature" as a category, and a warning that
effect sizes may be modest.

### Others on H1 (read; what they cover)
- Chen, Lin, Polyanskiy, Rigollet 2510.05554 (REPORT): beta_n ~ log n is critical in a deterministic
  simplex / almost-simplex model; explicitly contrasts with sqrt(log n) from the Gaussian REM line. So the
  field has an unresolved sqrt(log n) vs log n split; H1's "Limits 1 and 2" is a reconciliation attempt
  but not obviously new (Hayase-Karakida frames the same reconciliation via score statistics).
- Bruno et al. 2605.08505 (FULL, head): fixed query vs i.i.d. keys on the sphere; critical beta_n ~ n^(2/(d-1));
  full limit laws of ordered attention weights; says REM log n-scale free-energy transition was "recently
  connected to long-context attention by [GG26]".
- Gao et al. 2608.08922 (grep only): contains a "static REM benchmark" with a REM transition at
  sqrt(2 log M)/sigma in its supplement and IPR order parameter.
- Anson, Wang, Aitchison, Scale-invariant Attention 2505.17083 (REPORT): assumes marginally Gaussian
  logits and derives a position-dependent mean and std shift; reports QQ plots (App. J) showing Llama and
  Gemma attention logits are close to Gaussian. Supports H1's Gaussian assumption empirically.
- Alemanno 2610.22191 (FULL, abstract + intro): exponential Hopfield (softmax attention step) with
  REM-style freezing at low temperature; says on structured data (MNIST, Fashion-MNIST) the best accuracy
  lies beyond the freezing. No length-adaptive temperature.
- Hu et al. 2610.17347 (FULL, head + greps): free-energy / Hopfield view of attention, first-order
  glassy vs robust-inference phases, temperature and memory-load scans. No closed-form per-query temperature,
  no sqrt(log n) law. This is the "thermodynamic derivation" family; it does not cover H1's formula.
- Concetti and Franchini 2606.07757 (FULL): REM universality for a linear random Hamiltonian; remarks the
  Gibbs average can be read as a single-query cross-attention. Pure probability; no attention method.
- Erol et al. 2606.00262, Extreme-value corrections for InfoNCE (REPORT, ICML 2026): per-anchor adaptive
  InfoNCE using Gumbel/Weibull EVT; adds an endpoint-shortfall logit. Does not give a temperature vs
  number-of-negatives law. Related to H1's secondary InfoNCE claim only in being per-anchor EVT.
- 2602.08216 Kaniadakis/Abe attention (ABS): derives attention operators from generalized entropies;
  no REM, no sequence-length temperature. Not prior art for H1.
- 2609.22005 (FULL, head) abstention and noise filtering: value gating plus sink logits. Not about temperature.
- 2602.23057 Affine-Scaled Attention (REPORT): learned input-dependent scale and bias on softmax weights.
  Not about temperature or REM.
- 2506.16640 ASEntmax (ABS only): alpha-entmax with learnable temperature; I could not confirm a log n
  dependence from the page I read. It is a baseline H1 already lists.
- 2501.19399 SSMax (ABS only): the abstract does not state the s*log n formula or a derivation; I am
  relying on H1's own text and other papers' descriptions for that.
- 2501.08570 InfoScale (ABS + SNIP): entropy-invariance length scaling. I only saw the abstract; the formula,
  and whether it is per-query, are unverified. Flag as a candidate prior art for "length-adaptive temperature".
- 2512.10656 (SNIP only): says Gaussian maxima sit near sqrt(2 ln n). Not read.

### H1 coverage table

| H1 component | Status | Where |
|---|---|---|
| Softmax row as REM; freezing at sqrt(2 log n)/sigma | ALREADY DONE | 2505.24333, 2608.08922 (grep), 2610.34682, 2610.22191 |
| sqrt(log n) vs log n scaling, with reconciliation via score stats | ALREADY DONE (mostly) | 2605.12697, 2510.05554, 2605.08505 |
| Needle detectable only if mu > sigma sqrt(2 log n) | ALREADY DONE (Gaussian REM) | 2610.34682 |
| Oracle per-row temperature from score entropy | ALREADY DONE | 2610.34682 Prop 15; 2410.01104 (heuristic) |
| Closed-form smaller-root beta*(mu, sigma, n, kappa); SSMax linear limit and sqrt(log n) as limits of one formula | NOT FOUND | none |
| Plug-in estimates (z_max minus mean, sigma-hat from non-max scores), detached, parameter-free | NOT FOUND | none |
| Length-extrapolation recall test vs SSMax / ASEntmax / entmax-1.5 | NOT FOUND for this method | n/a |
| InfoNCE best temperature vs number of negatives from REM | NOT FOUND | only generic statements; 2606.00262 is per-anchor EVT, different |

## H1: results that argue the idea will fail or shrink

1. Non-universal exponent. Hayase-Karakida measure optimal exponents of 0.4-0.5 (small GPT) and 2.0-2.5
   (Qwen). Gaussian REM predicts 0.5 only. If H1's plug-in sigma-hat absorbs this it is fine; if
   the score tails are not Gaussian, E1.0's own kill rule predicts failure. Chen et al. get log n from a
   deterministic model.
2. Effective n is not the count. Park shows clustered/redundant distractors (GREM) change the threshold and
   that count correction leaves within-cluster extremes; plugging in n = unmasked count will be
   wrong in text with repeated passages.
3. Small headroom for adaptive temperature. Veličković et al.: a per-query entropy-driven temperature gave
   gains of about 1-2 points on OOD sizes.
4. Best accuracy can lie beyond freezing (Alemanno, on structured data), i.e. the annealed-branch target
   may be the wrong operating point.
5. Own reasoning (not from a paper; check it): (a) The rule is equivariant, beta*(c z) = beta*(z)/c, so the
   model can no longer set sharpness by scaling q.k; only kappa per head remains. Heads that need diffuse
   averaging or sharp argmax lose a degree of freedom. (b) On a no-needle row, z_max - mean is about
   sigma sqrt(2 log n), so mu-hat sits at the threshold and the clamp returns about the critical beta for
   every null row (H1 already notes this risk). (c) For kappa > 0 at the threshold, beta* = sqrt(2L)/sigma
   exceeds beta_c = sqrt(2 log n)/sigma, so the annealed formula is used slightly outside its validity
   (conservative direction, since true log Z is below the annealed value there). (d) Gaussian-logit
   support: 2505.17083 App. J shows near-Gaussian logits in Llama/Gemma, which helps H1.
6. The kill criterion in E1.1 compares against SSMax and ASEntmax; both are published and one is trained
   with a learned parameter, so the baselines are strong and a tie is the likely outcome.

---

## H2: closest papers

### Chaitanya, Zhou, Gionis, "Does Muon Need Fine-Grained Spectral Shaping?" (BulkBoost), 2610.07497, 5 Oct 2026 (FULL: abstract, Sec 1-3, Sec 5, App E.2 table)
Does:
- Diagnostics: 93.7-97.1% of pooled singular modes of Muon's Nesterov input lie below a Marchenko-Pastur
  noise edge, yet collectively align positively with an independent reference gradient in every analyzed matrix.
- BB-Adaptive: two independent equal-size minibatch halves G1, G2; Z_split = (G1 - G2)/2 has zero mean and "the same
  covariance as the sampling noise in the raw full-minibatch mean"; s^2 from ||Z||_F^2/(mn), smoothed
  by EMA; propagated through the Nesterov input with a factor nu_t(beta); threshold
  tau = kappa_edge * nu * s * (sqrt(m) + sqrt(n)) with kappa_edge = 1 in all experiments.
- Two-band reweighting with bulk-to-spike gain rho >= 0, rescaled so the Frobenius norm of the flat polar update is kept;
  an SVD-free polar-plus-low-rank realisation; 30 continued-pretraining settings (Pythia-14M to 410M, six corpora).
- Results: rho > 1 (boost the bulk) beats flat Muon in 24/30 settings, mean -0.077% of initial loss, about 8% fewer
  tokens where it wins; step time 1.36-1.50x Muon. Single exact-SVD update tests at rho in {0.5, 0.75, 1, 1.25, 1.5}
  favour bulk emphasis over suppression in 5 of 6 states (central-difference slope positive in 5/6; one
  state is about zero or slightly negative).
- States that the edge "is neither an exact signal-noise separator nor a prescription to discard below-edge modes"
  (my paraphrase of their App. A.4) and that the isotropic-Gaussian edge is an approximation for real gradient noise.
Does not: test deleting the bulk (rho = 0) in the main experiments; use two momentum buffers (it splits gradients
and propagates variance analytically); use ||N||_2 of the difference matrix as the edge (it uses the Frobenius-derived
variance and the MP formula); test batch-size dependence of the kept fraction; use a random-subset control.
Relevance: H2's measurement (half-batch difference as noise sample, MP-style edge, gamma = 1, Frobenius-matched renorm)
is the same construction. Their parametrisation (spike gain 1, bulk gain rho, norm matching) contains H2's
gate-renorm at rho = 0 (my inference; they note gains (1, rho) and (1/rho, 1) give the same update and that rho = 0
needs a nonzero spike component). H2's remaining novelty is the two-buffer edge estimator, the batch-size
prediction, the random-subset control, and actually running the deletion arm.

### Goyal and Ramkumar, "Spectrally Targeted Muon", 2610.10965, 7 Oct 2026 (FULL)
Does: orthogonalizes only singular values above or below a threshold tau (shifted-Gram Newton-Schulz); CIFAR-10 and
NanoGPT speedrun (3 seeds, 1530 steps). Finding: small momentum singular values are not noise. Orthogonalizing all but the few largest gets within
0.03 loss of Muon while touching about 13% of the momentum norm; orthogonalizing only the top is well short even when it
holds up to 96% of the momentum, and the smallest top doses do worse than lifting nothing. Raising the learning rate of top-only variants does not help.
Does not: use a measured noise edge; tau is absolute.
Relevance: direct empirical evidence against H2's mechanism. A hard gate with c = sqrt(rank/|S_keep|) keeps only the
top few modes and amplifies them; this is the "top-only" arm that lost.

### Others on H2 (read; what they cover)
- Li et al. 2606.03899 "Denoise First, Orthogonalize Later" (REPORT): momentum as spectral filter; signal-plus-perturbation
  model, spiked rank-1 Gaussian; Pre-polar (Muon) beats Post-polar and Polar-only, advantage larger at smaller minibatch.
  Polar-only alignment vanishes as m grows at low SNR. Supports H2's premise (noise amplified at small batch) but gives
  momentum, not gating, as the remedy.
- Long et al. 2605.09331 (FULL, intro + Sec 10): RMT spiked-rectangular analysis of Muon saddle escape. Sec 10 states the
  Gavish-Donoho-style optimal operator-norm shrinker is a hard threshold at the MP upper edge (1+sqrt(gamma))^2 and says
  Newton-Schulz approximates it. This is a theoretical remark with a regret definition, not an algorithm or experiment.
  Prior art for the idea "Muon's polynomial ~ optimal shrinkage", not for the gate.
- Xie 2609.32861 (FULL, intro): near an optimum Muon adds covariance via a nonlinear residual; recommends switching to
  response-matched momentum SGD late; uses an "unbiased split-pool estimate" of ||Y||_F^2 to build a noise-dominance
  indicator and says the switching rule is untested. Prior art for split-estimate noise indicators in Muon and for
  "orthogonalization hurts when noise-dominated". Supports H2's premise at late training.
- Beneventano et al. 2606.08388 (REPORT; the one H2 asked about): flat-spectrum bias in underdetermined regression; Theorem 4 gives a
  critical batch size for Muon tied to polar-map spectral sensitivity S(mu) = sum (sigma_i + sigma_j)^(-2), inflated by small
  singular values; NanoGPT beats AdamW but a small ViT reverses. No gate, no edge.
- Dong and Sawin, Muon_p 2606.13867 (REPORT): fractional spectral power U S^p V^T, Schatten q = 1 + 1/p steepest descent; better
  at fine-tuning, worse at pretraining than Muon. No noise estimate, no threshold.
- Massena et al., "From SGD to Muon: Adaptive Optimization via Schatten-p Norms" 2605.19781 (ABS): data-driven per-layer choice of
  LMO geometry between SGD and Muon. Abstract does not mention noise or thresholds. 2606.15268 (ABS): best p is
  regime-dependent (low-dimensional regime favours smaller p); gives a batch-size scaling rule for arbitrary p.
- Park-style evidence on spectrum shape: Shumaylov et al. 2605.11181 (ABS): random or inverted spectra work about as well; explanation is
  step-size optimality, not geometry (this is the paper H2 cites for the step-size control).
- Spectral Allocation / SAMuon 2608.25990 (ABS): per-direction loss-optimal step sizes show a volatile head and a tolerant bulk;
  SAMuon keeps the head at Muon scale and amplifies the bulk with a static prior; 13-24% fewer tokens than Muon on nanoGPT 124M-1B.
- Best Optimizer Depends on Batch Size 2610.08975 (ABS): rankings flip with batch size; theory via noisy quadratic model; no principled Muon scaling
  rule. Notable: after a 16x batch increase, keeping small-batch updates only in the sharpest 0.001% of directions recovers up to
  59.5% of the loss gap while a random subspace helps little (mild support for selection mattering).
- Spike Estimation via Random Splitting 2609.11169 (ABS), OptShrink 1306.6042, partial-noise-statistics shrinkage 2211.00986
  (both SNIP only): classical RMT uses noise-only side information or data splitting to estimate spikes without a noise model.
  Prior art for the estimation idea in general; none applied to optimizer momentum.
- Seen as snippets only (not read): MuCon 2605.26459 and spectral clipping 2605.11838 (clip large singular values),
  Musec 2609.11655, SpecMuon 2602.16167 (keeps top-k modes, scientific ML), 2606.00371 (how much orthogonalization).
  None showed a noise-edge threshold in the snippets. Check before claiming novelty beyond this list.
- Not found at all: Gavish-Donoho or OptShrink applied to gradients/momentum inside an optimizer (queries 6, 9); a split-batch
  noise estimate with two momentum buffers (queries 10, WebSearch 7).

### H2 coverage table

| H2 component | Status | Where |
|---|---|---|
| Optimal shrinker = hard threshold at MP edge, linked to Muon | Remark only | 2605.09331 Sec 10 |
| Half-batch difference as a noise sample (same covariance as full-batch noise) | ALREADY DONE in Muon | 2610.07497 |
| MP edge threshold on Muon's momentum input, multiplier 1 | ALREADY DONE | 2610.07497 |
| Frobenius-norm-matched reweighting separating "which directions" from "step size" | ALREADY DONE | 2610.07497 |
| Action = boost bulk (rho > 1) | ALREADY DONE; wins | 2610.07497 |
| Action = remove below-edge modes (hard gate, rho = 0) with renorm | NOT FOUND as a reported experiment; contradicted by nearby evidence | 2610.07497 (rho 0.5-0.75 single update), 2610.10965 |
| Two momentum buffers; edge = spectral norm of difference | NOT FOUND (variant of the above) | n/a |
| Kept-fraction rises with batch size; Muon-limit at large batch | Not found as a Muon measurement | 2606.08388 Thm 4 is the nearest theory |
| Random-subset control at matched |S_keep| | NOT FOUND for this | 2605.11181 uses random spectra, a different control |

## H2: results that argue the idea will fail

1. 93.7-97.1% of modes lie below the edge but collectively align with the reference gradient (2610.07497). Gating removes
   nearly all modes, and renormalising to Muon's norm then multiplies the few survivors by roughly sqrt(rank/|S_keep|),
   which can be about 4-5x for a 5% keep rate. That is the top-only regime that lost in 2610.10965.
2. 2610.07497 single-update tests prefer bulk emphasis over suppression in 5 of 6 states; 2608.25990 also amplifies the bulk;
   2610.10965 says small singular values are trainable signal once amplified.
3. 2605.11181 attributes Muon's success to step-size optimality; the gate changes the step geometry drastically, so a
   confound with step size is likely even with the renorm.
4. Gradient noise is not isotropic Gaussian (acknowledged in 2610.07497; 2610.31728 is about structured noise). A single
   realisation of ||N||_2 per step is a noisy edge estimate. Heavy tails make edge exceedances common.
5. Cost: the closest published adaptive variant runs at 1.36-1.50x Muon step time; two buffers add optimizer state and an extra SVD.
6. What is in H2's favour: 2609.32861 (noise-dominated late phase), 2606.03899 (polar on single gradients loses signal at low SNR),
   2610.08975 (sharp-direction selection recovering the batch-increase gap). These support the premise that noise amplification
   matters at small batch or late training, not that deleting below-edge modes is the right fix.
   My own estimate: the 20% prior in the pre-registration looks high; I would put H2 beating tuned Muon at under 10%.
   That is an opinion, not a finding.

## Caveats on this search

- Several key papers were read as alphaXiv AI reports rather than raw text (listed as REPORT). Numbers quoted from them
  should be rechecked in the PDFs before they go in a write-up.
- Papers in this index are dated up to 1 Nov 2026; very recent work (last days) may be missing.
- I did not search OpenReview or Google Scholar, and WebSearch is US-only. A negative result there is weaker than a positive one.
- The two "NOT FOUND" cores (H1 closed-form plug-in estimator; H2 literal deletion arm) are narrow. Both sit one parameter
  setting away from published work (Park Prop 15 and Hayase-Karakida for H1; BulkBoost rho = 0 for H2), so a reviewer
  could reasonably call them incremental. Describe them as such.
