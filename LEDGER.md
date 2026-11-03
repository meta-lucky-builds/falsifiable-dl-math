# Hypothesis ledger

Every hypothesis stays here, including dead ones.

| ID | Claim | Status | Evidence |
|---|---|---|---|
| H1 | REM closed form sets attention temperature and unifies SSMax scaling, REM criticality and the Bayes posterior | Open: maths check passed, no training evidence | E1.0: Gaussian needle mass within 0.021 of target at all tested `n` and ratios; heavy tails break it near threshold |
| H2 | Gating Muon's spectrum at a measured noise edge beats Muon at small batch | Pre-registered | None yet |

## Queue, not yet pre-registered

- Harmonic loss is softmax on `-n log d`; test against plain distance logits
  (objectives survey G2).
- Forward gradient with sign or Muon updates, given its heavy tails (objectives
  survey G4).
- Sparse Fenchel-Young losses under label noise. My expectation from the algebra is
  that they will not be robust: clean examples reach exactly zero gradient, which
  raises the relative weight of noisy ones. This runs against the survey's lead.
- Closed-loop effective-learning-rate control with the `B_t` law on AdamW
  (optimization survey G1).
