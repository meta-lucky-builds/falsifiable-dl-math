# Hypothesis ledger

Every hypothesis stays here, including dead ones.

| ID | Claim | Status | Evidence |
|---|---|---|---|
| H1 | REM closed form sets attention temperature and unifies SSMax scaling, REM criticality and the Bayes posterior | **Killed at all three noise levels** | E1.1: REM lost to the best baseline by 0.034 to 0.065 at `eta` 0.6 and 0.9, `N` 128 and 256, with paired intervals excluding zero, and by smaller margins at `eta` 0. It ties SSMax at `N = 256` with no learned parameters. The formula itself checked out in E1.0, but holding needle mass did not give a win. See `experiments/E1_1_recall/RESULTS.md`. Prior art: 2505.24333, 2610.34682, 2605.12697 |
| H2 | Gating Muon's spectrum at a measured noise edge beats Muon at small batch | **Retired without an experiment** | 2610.07497 uses a similar construction and reports boosting the below-edge bulk beats flat Muon in 24 of 30 settings, though by only 0.077% of initial loss on average. Its modes are 93.7-97.1% below the edge, so a gate keeps very few. 2610.10965 reports top-only orthogonalization falls short. Not tested by me |
| H3 | Sign updates improve forward-gradient estimates, as heavy-tail theory might suggest | **No estimator advantage for fixed `g`**; training not tested, thread deprioritized | E3.0: closed form and Monte Carlo agree. SNR is 1.00 raw, 0.80 tangent-normalized, 0.64 sign |
| H4 | Finite Adam epsilon gives a closed recursion and a crossover scale `r* = G/(eps sqrt d)`. My stronger conjecture, that epsilon acts as an effective-LR schedule, is expected to fail: equilibrium rotation should not depend on epsilon | **Stopped: pre-registered kill fired on `omega(x)`, E4.1 not run** | E4.0: exact recursion confirmed to about 1e-14, including against real AdamW. On a noise walk, equilibrium rotation does not depend on epsilon (within 0.13%) and `r_eq` and the transit law hold, but `omega` missed by 18.8% against a 15% limit. Toy model only; no route to a speedup |
| H5 | Entmax15's gain on recall comes mainly from training or mainly from the sparse readout | **Launched, result not recorded** | Suggested by E1.1, so it uses fresh seeds. See `experiments/E5_1_swap/RESULTS.md` |
| H6 | In diagonal nets, decay-then-off reaches the same limit as a small init, and small init gets there in no more steps | **Launched, only Part A read** (exact identities pass, about 1e-13; log not archived) | Different area: implicit bias. Diagonal linear models only. See `experiments/E6_0_decay_budget/RESULTS.md` |
| H7 | Attention normalizers on long contexts in a real language model | **Run, but did not test the question.** Pre-registered outcomes: P1 not supported, P2 inconclusive, P3 not supported | All arms break at 2x training length (1.39 to about 5.2 bits per byte), diagnosed in H8. A naive entmax readout costs 0.26 bits per byte at `N = 256`. See `experiments/E7_gpu/RESULTS.md` |
| H8 | Why byte-level models break past training length: position scheme, or a bug | **Toy run done: no sign of a bug, ALiBi survives, RoPE fails.** Full run pending | 2 seeds, 6000 steps, direction only. Break follows the training length. Unseen distances get 22x the per-token weight. Raising `theta` 50x is worse, so the wraparound story is not supported. ALiBi holds to 8x with no short-context cost. See `experiments/E8_long_context/RESULTS.md` |

## Queue, not yet pre-registered

- REM law for InfoNCE: how the best contrastive temperature moves with the number of
  negatives. Prior-art check found no such law.
- Harmonic loss is softmax on `-n log d`; test against plain distance logits
  (objectives survey G2).
- Sparse Fenchel-Young losses under label noise. My expectation from the algebra is
  that they will not be robust: clean examples reach exactly zero gradient, which
  raises the relative weight of noisy ones. This runs against the survey's lead.
- Closed-loop effective-learning-rate control with the `B_t` law on AdamW
  (optimization survey G1).
- Per-band spectral step sizes for Muon, using cross-fit half-batch alignment and a
  curvature estimate. This follows from the BulkBoost result, but the published effect
  sizes are small and may not be measurable at CPU scale.
