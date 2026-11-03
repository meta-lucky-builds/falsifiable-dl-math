# H1: attention temperature from the Random Energy Model

Status: pre-registered 2026-11-04, before any experiment. Prior that it beats
tuned baselines on length extrapolation: about 35%. Prior that the closed form
describes the Gaussian case: about 85%.

## Derivation

Softmax attention over `n` tokens is a Gibbs measure with inverse temperature
`beta`. Model one relevant token with score margin `mu` and `n-1` distractors
`z_i ~ N(0, sigma^2)`. The distractor partition function is Derrida's Random
Energy Model, which freezes at

    beta_c = sqrt(2 log n) / sigma

- `beta < beta_c`: `log Z = log n + beta^2 sigma^2 / 2` (annealed branch).
- `beta > beta_c`: `log Z = beta sigma sqrt(2 log n)` (frozen branch).

Needle log-odds on the annealed branch:

    Lambda(beta) = beta mu - log n - beta^2 sigma^2 / 2

Ask for the smallest `beta` reaching target log-odds `kappa`, and write
`L = log n + kappa`. The smaller root, with the discriminant clamped, is

    beta*(mu, sigma, n) = (mu - sqrt((mu^2 - 2 sigma^2 L)_+)) / sigma^2

Why the smallest root: a larger `beta` saturates softmax and shrinks gradients,
so the smallest sufficient `beta` is the differentiable choice.

### Limits

1. **High signal, `mu >> sigma sqrt(2L)`:** `beta* -> L / mu`, which is linear in
   `log n`. This is SSMax's empirical scaling (2501.19399), with `s = 1/mu`.
2. **Threshold, `mu = sigma sqrt(2L)`:** `beta* = sqrt(2L) / sigma`, the REM
   critical scaling, which grows as `sqrt(log n)` and not `log n`.
3. **Below threshold, `mu < sigma sqrt(2L)`:** the clamp gives `beta* = mu /
   sigma^2`. This maximizes `Lambda`, and it is the Bayes posterior over needle
   position under a Gaussian spike model. The same formula therefore covers all
   three regimes and is continuous.

### Properties

- Equivariant: `beta*(c z) = beta*(z) / c`, so `beta* z` does not depend on the
  score scale or shift. There is no temperature hyperparameter.
- Detection limit: when `kappa = 0`, no `beta` makes the needle dominate unless
  `mu > sigma sqrt(2 log n)`. Length failure below that limit is about signal,
  not temperature, so a temperature fix cannot help there.

### Consequences to test

- **Plug-in attention (REM-Attn).** Per query, using scores excluding the row
  maximum, estimate `sigma_hat` and `mean_hat`. Set `mu_hat = z_max - mean_hat`,
  and detach `beta*` from the graph. `n` is that row's unmasked count. `kappa` is
  fixed or learned per head.
- **Scaling depends on tails.** Gaussian-tailed scores imply `sqrt(log n)` at
  threshold. Exponential tails have an `n`-independent critical `beta`. Empirical
  `log n` scaling may therefore reflect where trained models operate.
- **InfoNCE.** The same law predicts how the best contrastive temperature moves
  with the number of negatives. Secondary test; not part of the kill criteria.

## Assumptions that could break it

- Scores in trained models may not be Gaussian or independent.
- Multiple relevant tokens change `L` by about `log k`.
- `max` is a noisy estimate of `mu` in short contexts.
- The estimator may leave attention at criticality on rows with no needle, which
  could add variance when it should stay diffuse.

## Experiments

**E1.0, numerical check.** Monte Carlo needle mass at `beta*`, against target
`p = exp(kappa)/(1+exp(kappa))`, across `n` and signal ratios, for Gaussian and
heavy-tailed distractors.
Kill: for Gaussian scores above threshold, mean needle mass misses the target by
more than 0.10 at `n >= 2^10`. Heavy tails are expected to fail and only delimit
the law's scope.

**E1.1, recall length extrapolation.** Small transformer on associative recall,
trained at one length and tested at 2x to 32x. Arms: softmax, learned-scale
softmax, SSMax, entmax-1.5, length-scaled entmax-1.5, REM-Attn. At least 5 seeds.
Learning rates are tuned with the same grid per arm, selected only on validation
data at the training length.
Predictions: REM-Attn matches or beats SSMax at long lengths and has fewer
learned parameters. Its gain over SSMax is largest when distractor noise is high.
Kill: REM-Attn is not better than the best baseline at the longest two lengths,
with paired 95% intervals excluding zero in its favor. A tie is reported as a tie,
meaning a parameter-free rule that matches, not as a win.

**E1.2, score audit.** Measure score tails, `mu_hat / sigma_hat`, and how well the
formula predicts learned SSMax temperatures in the trained E1.1 models.
Kill for the explanatory claim: the formula predicts learned `beta` no better
than constant `c log n`.
