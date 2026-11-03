# E3.0: estimator quality of sign updates on forward gradients

Run on Colab (CPU), 2026-11-04, NumPy seed 0, 200,000 tangent samples per row.

## Question

The objectives survey (G4) suggested sign or Muon-style updates might rescue forward
gradients, because the estimator has heavy tails and sign methods are robust to heavy
tails (2602.07425). No training was run: this asks whether the estimator itself
supports that idea.

## Derivation

Fix `g` with `||g|| = 1`, and let `v ~ N(0, I_P)` and `a = v.g`. Three updates:

| Update | Formula | Mean of coordinate `i` | Noise std | SNR per unit `g_i` |
|---|---|---|---|---|
| Raw | `a v` | `g_i` | about 1 | 1 |
| Tangent-normalized | `sign(a) v` | `sqrt(2/pi) g_i` | 1 | `sqrt(2/pi)` = 0.798 |
| Sign | `sign(a) sign(v)` | `(2/pi) arcsin(g_i)` | 1 | `2/pi` = 0.637, for small `g_i` |

The sign row is Sheppard's orthant formula applied to the pair `(a, v_i)`, which has
correlation `g_i`. Because `arcsin(x) >= x`, large coordinates do somewhat better.

## Results

| P | `g` | Raw SNR | Tangent-norm SNR | Sign SNR |
|---|---|---|---|---|
| 100 | dense | 0.997 | 0.801 | 0.641 |
| 100 | sparse (5% nonzero) | 0.994 | 0.801 | 0.720 |
| 100 | heavy-tailed | 0.997 | 0.803 | 0.665 |
| 2000 | dense | 1.002 | 0.801 | 0.637 |
| 2000 | sparse (5% nonzero) | 1.003 | 0.800 | 0.643 |
| 2000 | heavy-tailed | 0.999 | 0.797 | 0.649 |

The measured values match the formulas. The higher sign SNR at `P = 100` with sparse
`g` is the `arcsin` effect: 5 nonzeros means individual `g_i` are large.

## Verdict

For a fixed gradient, sign updates have **no estimator advantage**: they lose about
36% in SNR against raw, which is about `pi^2/4 = 2.47` times as many samples for the
same accuracy. Sign cannot be justified as improving the estimate.

## What this does not show

The model has a stationary `g` and a single tangent. It does not cover the following,
which could still favor sign updates in real training:

- step-size stability, since the raw update's size varies as `|a|`;
- gradients that change over time;
- interaction with Adam, which 2607.16612 blames for the failure of naive forward
  gradients.

A plain training comparison is the only way to settle those. Given that forward
gradients remain far behind backprop in the literature, this thread is deprioritized
and no training run is planned for now.
