# H6: "decay on, then off" against a small initialization, in diagonal nets

Status: pre-registered 2026-11-04, before any experiment. Different area from H1 to H5:
implicit bias. Scope is diagonal linear networks only. Nothing here claims transfer to
general networks.

Priors: exact identities 99%. Schedule invariance to `O(eta)` 90%. Small init no slower 70%.
Noisy test error a tie or better for small init 65%.

## Derivation (by hand, not yet run)

Parameterize `x = u^2 - v^2` per coordinate, with `g = dL/dx`, `L = ||Ax - y||^2 / (2n)`, and
weight decay `(alpha/2)(u^2 + v^2)`. A gradient step gives

    u' = u (1 - eta alpha - 2 eta g),   v' = v (1 - eta alpha + 2 eta g)

Define `c = uv` and `phi = log(u/v)`. Then `x = 2c sinh(phi)`, and exactly, at finite `eta`:

    c'   = c ((1 - eta alpha)^2 - 4 eta^2 g^2)
    phi' = phi + log((1 - eta alpha - 2 eta g) / (1 - eta alpha + 2 eta g))

In the flow limit, `dc/dt = -2 alpha c` and `dphi/dt = -4 g`. Decay touches only `c`, and the
gradient touches only `phi`. The `-4 eta^2 g^2` term is the "Gain" of 2610.06120, so decay and
large-step effects act on the same quantity through one product. I have not seen this
pair written as one exact recursion, but that was not specifically searched.

**Schedule invariance.** Start with `u0 = v0`, so `phi0 = 0`. Then `phi(t)` stays in the row
space of `A`, and with decay off after `T`, the limit is

    x = 2 c_T sinh(A^T lambda),   A x = y,   c_T = c_0 exp(-2 integral alpha)

So the limit depends on the schedule only through its integral. This is the content of
Theorem 3.6 of 2504.12883, which I read only through a summary.

**The new consequence.** Starting at `c_T` with no decay reaches the *same* limit. So in
this model, decay-then-off is a substitute for small initialization, not a different bias.
And since `dphi/dt = -4 g` regardless of `c`, reaching a given `phi` needs the same integrated
gradient on either path. Larger `c` fits the data earlier and so shrinks `g`, slowing `phi`.
**So small init should reach the limit in no more steps.** The practical reading in
2504.12883, that switching decay off is a useful strategy, may therefore apply where small
init is unavailable, as in fine-tuning a pretrained model, and not in general.

## Experiment E6.0 (numpy, Colab CPU, low priority)

Gaussian `A`, `n = 30`, `d = 100`, 3-sparse truth with entries of +1 or -1. `c0 = 0.25`.

- **Part A, identities.** 2000 steps with decay on. The two recursions match direct iteration.
  Limit: relative error below 1e-9.
- **Part B, invariance.** 3 designs, budget `integral alpha = 1.5`, 4 schedules (constant,
  early burst, late burst, linear ramp-down), `eta` in {0.01, 0.005, 0.0025}. Compare the limit
  to the closed form solved by Newton on the dual.
  Kill: relative error above 2% at the smallest `eta`, or an error slope against `eta` below 0.5.
- **Part C, speed, noiseless.** 20 designs, `eta = 0.01`, budgets 1.5 and 3. Steps to relative
  residual 1e-6. Arms: small init at `c_T` with no decay, against decay paths from `c0` (constant
  and early burst, duration in {1, 5, 25} time units), taking the *best* decay arm per design and
  budget. Ratio = steps(best decay) / steps(small init). Both arms must reach closed-form limits
  within 2%, else the speed comparison is void.
  Kill for the speed prediction: median ratio below 0.8 at either budget.
- **Part D, noisy test error.** 20 designs, noise s.d. 0.1, 30 validation points, 1000 test
  points. For each design choose `c` from {1, 0.25, 0.06, 0.015, 0.004, 0.001} and a checkpoint
  using validation error. Arms: small init, and decay from `c0 = 0.25` with the budget that gives
  the same `c` (constant over 5 time units; budgets for `c` above 0.25 are not available to this
  arm). Report paired test-error difference with a 95% interval.
  Kill for the "no advantage" claim: decay arm better, with the interval excluding zero.

## What it cannot show

Everything here is a diagonal linear model at fixed `eta`. Small init may allow a larger stable
`eta`, which only strengthens the prediction, so it is not tested. No claim about ReLU networks
or attention. Fine-tuning, the setting of the paper's empirical wins, is untouched.
