# E1.0: numerical check of the REM temperature law

Run on Colab (CPU), 2026-11-04, NumPy seed 0. Code: `check.py`. Target needle mass
0.9 (`kappa = log 9`). `sigma = 1`. `ratio = mu / (sigma sqrt(2L))`. Each cell uses
300 Monte Carlo repeats for `log2 n <= 14` and 40 for `log2 n = 18`. Values are mean
needle mass at `beta*`.

| Scores | log2 n | ratio 1.05 | 1.25 | 1.5 | 2.0 | 4.0 |
|---|---|---|---|---|---|---|
| Gaussian | 6 | 0.921 | 0.900 | 0.903 | 0.902 | 0.902 |
| Gaussian | 10 | 0.913 | 0.900 | 0.900 | 0.900 | 0.900 |
| Gaussian | 14 | 0.908 | 0.900 | 0.900 | 0.900 | 0.900 |
| Gaussian | 18 | 0.905 | 0.902 | 0.900 | 0.900 | 0.900 |
| Laplace | 6 | 0.759 | 0.834 | 0.858 | 0.895 | 0.901 |
| Laplace | 10 | 0.347 | 0.569 | 0.747 | 0.873 | 0.899 |
| Laplace | 14 | 0.015 | 0.167 | 0.487 | 0.827 | 0.898 |
| Laplace | 18 | 0.000 | 0.009 | 0.127 | 0.727 | 0.897 |
| Student-t, 3 df | 6 | 0.770 | 0.792 | 0.808 | 0.870 | 0.895 |
| Student-t, 3 df | 10 | 0.101 | 0.203 | 0.346 | 0.591 | 0.861 |
| Student-t, 3 df | 14 | 0.000 | 0.000 | 0.001 | 0.026 | 0.529 |
| Student-t, 3 df | 18 | 0.000 | 0.000 | 0.000 | 0.000 | 0.018 |

Identity checks:

- High-signal limit at `log2 n = 14`: `beta*` against `L/mu` is 0.2386 / 0.2380 at
  `mu = 50`, 0.0595 / 0.0595 at 200, and 0.0119 / 0.0119 at 1000.
- Continuity at threshold: 4.8739 just below and 4.6654 just above, with threshold
  `mu` = 4.8788. This is continuous but has infinite slope from the square root, so
  a 0.1% step in `mu` moves `beta*` by about 4%. **A plug-in estimator will be noisy
  near threshold.**
- Scale equivariance: `beta*(30, 1) * 1 = beta*(60, 2) * 2 = 0.39937`.

## Verdict

**The pre-registered kill did not fire.** For Gaussian scores above threshold at
`n >= 2^10`, the largest miss is 0.013, against a limit of 0.10.

This is a check of the mathematics only. It says nothing about training.

## What this changes

Heavy tails break the law. At `log2 n = 18`, Laplace falls to 0.000 at ratio 1.05,
and Student-t stays near zero even at ratio 2.0. The Gaussian `K(beta) = beta^2
sigma^2 / 2` is standing in for the true cumulant generating function, and it
under-counts distractors when tails are heavy. The general condition is

    beta mu - log n - K(beta) = kappa

so the tail shape of scores in trained models decides whether the closed form is
usable. Two changes follow, both made before E1.1 runs:

1. E1.2 (score audit) now matters more, and its tail measurements decide how far the
   closed form can be trusted.
2. E1.1 gains an arm, **Calib-Attn**: per query, bisect for the smallest `beta` that
   puts mass `p` on the top token. It makes no distribution assumption, in the style
   of t-SNE perplexity calibration. If it beats REM-Attn, the Gaussian form is the
   problem. If REM-Attn beats it, then lowering `beta` on rows with no needle, which
   Calib-Attn cannot do, is what helps.
