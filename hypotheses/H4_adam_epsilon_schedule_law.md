# H4: Adam's epsilon in the scale-invariant schedule law

Status: pre-registered 2026-11-04, before any experiment. All algebra below is by
hand and nothing has been executed. Priors:

| Claim | Prior |
|---|---|
| Exact identities (K, D, C, R, S below) | 99% |
| Closed forms (`r_eq`, rotation, relaxation rate) on an isotropic noise walk | 85% |
| The same forms hold within tolerance on a BN MLP | 50% |
| Strong conjecture: epsilon acts as an effective-LR schedule at weight-decay equilibrium | 10% |
| Any of this improves training over a tuned epsilon | under 5% |

**Short verdict.** The crossover is real, and `r*` has a closed form. But it changes
less than conjectured. At weight-decay equilibrium the rotation rate does not depend
on epsilon. Without decay, it moves by at most a factor of sqrt(2). Epsilon changes
the norm, the relaxation rate (by up to 2x) and, mainly, how rotation is shared among
coordinates. That last effect is the known Adam-to-momentum interpolation. Epsilon is
a preconditioner-mixing parameter, not an effective-LR schedule.

## What the paper actually says (2609.09116, checked against the full text)

Confirmed:
- Theorem 2.1: `Phi_t = eta_t/(a_t r_t^2)`, `Phi_{t+1} = B_t Phi_t/(1 + Phi_t^2 ||g_bar_t||^2)`,
  `B_t = (eta_{t+1}/eta_t)/(a_t a_{t+1})`.
- Theorem 4.3 and the text after it, verbatim in substance: at `eps > 0` scale
  invariance is perturbed "by an additive eps r_t term in the denominator".
- Listed as open: "formal eps-perturbation theory". The paper only calls `eps > 0` a
  "smooth perturbation", validated empirically (Fig. 3b and Fig. 7).

Corrections to the brief:
1. **The Adam law is not the `Phi` law.** It uses `Psi_t = eta_t/(a_t r_t)`, a
   different forcing `B~_t = (eta_{t+1}/eta_t)/a_{t+1}`, and
   `Psi_{t+1} = B~_t Psi_t / ||u_t - Psi_t p_bar_t||` (eq. 24).
2. **The Adam theorem does not exclude decoupled weight decay.** The discussion says
   Theorem 4.3 applies "external multiplicative shrinkage a_t w_t, which is the
   AdamW-style decoupled form". What falls outside it is L2 inside the moments. The
   paper is inconsistent here: Section 4.2 says "coupled weight decay", and "Open
   directions" says decoupled decay breaks the structure. Only finite epsilon is
   truly excluded.
3. **"Linear versus quadratic" depends on the variable** (see K below). It is not by
   itself a physical statement.

Not verifiable from the extracted text: the values plotted in Fig. 3b and Fig. 7. The
tick labels show an epsilon sweep of 1e-12 to 1e-6 and an exponent axis of 0.80 to
1.20. Curve values did not survive extraction.

## Derivation

### Setup

Block `w_t` in R^d, `r_t = ||w_t||`, `u_t = w_t/r_t`. Assumption A1 (exact scale
invariance): `L(cw) = L(w)`, so `g_t = g_bar_t/r_t` with `g_bar_t` a function of
`(u_t, xi_t)` only, and `<u_t, g_bar_t> = 0`.

AdamW as implemented in PyTorch, with `s = t+1`, `a_t = 1 - eta_t lam_t`:

    m_{t+1} = b1 m_t + (1-b1) g_t
    v_{t+1} = b2 v_t + (1-b2) g_t^2
    p_t = m^_{t+1} / (sqrt(v^_{t+1}) + eps),  m^ = m/(1-b1^s),  v^ = v/(1-b2^s)
    w_{t+1} = a_t w_t - eta_t p_t

Scale-free moments, as in the paper: `m~_{t+1} = r_t m_{t+1}`, `v~_{t+1} = r_t^2 v_{t+1}`.
With `rho_t = r_t/r_{t-1}`:

    m~_{t+1} = b1 rho_t m~_t + (1-b1) g_bar_t
    v~_{t+1} = b2 rho_t^2 v~_t + (1-b2) g_bar_t^2

### Exact identities (no approximation beyond A1 where it is used)

**K, kinematics.** For any update vector `p_t`, let
`N_t = ||u_t - Psi_t p_t|| = r_{t+1}/(a_t r_t)`. Then

    Phi_{t+1} = B_t  Phi_t / N_t^2
    Psi_{t+1} = B~_t Psi_t / N_t
    N_t^2 = 1 - 2 Psi_t c_t + Psi_t^2 s_t,   c_t = <u_t,p_t>,  s_t = ||p_t||^2

This uses no scale invariance and no Adam, so it holds at every epsilon. **A float64
check of K alone therefore proves nothing about optimization.** It also shows that
the exponent is a property of the variable: Adam obeys both a power-2 law in `Phi` and
a power-1 law in `Psi`. The paper's point is which variable closes in scale-free
coordinates.

**D, epsilon decomposition.** Per coordinate,

    kappa_{t,i} = eps / sqrt(v^_{t+1,i}) = eps r_t / sqrt(v~^_{t+1,i})
    alpha_{t,i} = 1/(1 + kappa_{t,i})
    p_{t,i} = alpha_{t,i} p_bar_{t,i},  p_bar_{t,i} = m~^_i / sqrt(v~^_i)   (eps = 0 direction)
    p_{t,i} = (1 - alpha_{t,i}) m~^_i / (eps r_t)                          (dual form)

With `Phi^e_t = (eta_t/eps)/(a_t r_t^2)`, the step becomes

    Psi_t p_{t,i} = Psi_t alpha_i p_bar_i = Phi^e_t (1 - alpha_i) m~^_i

Both error terms are exact: `alpha` measures the distance from the SGD side and
`1 - alpha` the distance from the Adam side.

**C, closure.** Since `eps r_t = E_t/Psi_t` with `E_t = eps eta_t/a_t`, the radius can
be eliminated:

    D_{t,i} = sqrt(v~^_{t+1,i}) + E_t/Psi_t
    p_{t,i} = m~^_{t+1,i} / D_{t,i}
    N_t = ||u_t - Psi_t p_t||,  u_{t+1} = (u_t - Psi_t p_t)/N_t
    Psi_{t+1} = B~_t Psi_t / N_t,  rho_{t+1} = a_t N_t

The state `(u, m~, v~, Psi, rho)` is closed. Finite epsilon does not break closure. It
adds one schedule input, `E_t`. This is the recursion the paper leaves open.

**R, rotation budget.** With `theta_t = angle(u_t, u_{t+1})`, `cos theta_t = (1 - Psi_t
c_t)/N_t`. Assuming `1 - Psi_t c_t > 0`, summing gives the pathwise identity

    sum_t -log cos theta_t = log(r_T/r_0) - sum_t log a_t - sum_t log(1 - Psi_t c_t)

Epsilon enters only through the norm boundary term and the radial term `c_t`.

**S, symmetry.** For a fully scale-invariant model,
`(w_0, eta, lam, eps) -> (c w_0, c eta, lam/c, eps/c)` leaves directions unchanged and
multiplies `r_t` by `c`. Also `(L, eps) -> (sL, s eps)` leaves the whole trajectory
unchanged. So the only dimensionless controls, besides the betas, are `eta_t lam_t`,
`Psi_0`, and `E_t/G`, where `G` is the scale of `||g_bar||`.

### Limits of the exact recursion

- **`eps -> 0`:** `E_t = 0`, `alpha = 1`, `p = p_bar`, which is the paper's eq. 24.
  The relative error in each `p_i` is at most `kappa_i`.
- **`eps -> inf` with `eta' = eta/eps` and `lam' = lam eps` fixed:** `a` is unchanged
  and `Psi p -> Phi' m~^`, where `Phi' = eta'/(a r^2)`. Then
  `Phi'_{t+1} = B_t Phi'_t/||u_t - Phi'_t m~^_t||^2`, which is the paper's SGDM law
  (Theorem 4.2) with `mu = b1`, heavy-ball LR `eta'(1-b1)`, and `z = m~^/(1-b1)`,
  apart from bias correction. At `b1 = 0` it is exactly Theorem 2.1. The relative
  error in each `p_i` is at most `1/kappa_i`.
- **Decay maps as `lam' = lam eps`.** AdamW at large epsilon is SGD with LR `eta/eps`
  and weight decay `lam eps`, so `eta' lam' = eta lam`. This matters below. (Adam+L2
  goes to decay `lam` instead; it is not treated here.)
- **Radial term at `b1 = 0`:** because `<u, g_bar> = 0`, the first-order term cancels:
  `c_t = -(eps r)^{-2} sum_i u_i g_bar_i sqrt(v~^_i) + O(eps^-3)`. The preconditioning
  channel of the paper's Remark 4.4 dies as `eps^-2`.

### Controlled scalar reduction

Add A2: coordinates are statistically alike, `sqrt(v~^_i) = gbar` where `gbar^2 =
E||g_bar||^2/d`. Add A3: `v` is slow, so it tracks the rescaled gradient second moment.
Define

    r* = gbar/eps = G/(eps sqrt(d)),   x_t = r_t/r* = kappa_t,   alpha_t = 1/(1+x_t)

Then, at `b1 = 0` with `c = 0` and `theta_t` measured through `tan theta_t = Psi_t
sqrt(d) alpha_t`, and writing `theta` for that tangent,

    N_t = sqrt(1 + theta_t^2)
    theta_{t+1} = B~_t theta_t / N_t * (1 + x_t)/(1 + a_t N_t x_t)
    x_{t+1} = a_t N_t x_t

**Limits on the `(theta, x)` map.** `x -> 0` gives `B~ theta/sqrt(1+theta^2)`, the
Adam law. `x -> inf` gives `(B~/a) theta/(1+theta^2) = B theta/(1+theta^2)`, the SGD
law. The crossover is therefore one map in one pair of variables, and `B~` becomes `B`
automatically.

**With momentum (A4: gradients uncorrelated across steps).** Per-step `||p||^2 =
rho_b d alpha^2`, with `rho_b = (1-b1)/(1+b1)`. Also
`c = -Psi rho_b d alpha^2 b1/(1-b1)` from correlation with past updates, assuming
`a` is near 1. Adding the terms gives

    E[N^2] = 1 - 2 Psi c + Psi^2 s = 1 + Psi^2 d alpha^2

So `b1` cancels in the norm growth, as in Kosson et al.'s total-update argument, while
per-step rotation carries the factor `rho_b`.

**Local homogeneity.** At fixed scale-free state,
`-d ln||p||/d ln r = nu_eff = sum_i alpha_i^2 (1-alpha_i) / sum_i alpha_i^2`,
which reduces to `x/(1+x)` under A2. So finite-epsilon Adam is a locally
`nu_eff`-homogeneous optimizer in Theorem 4.1's sense, passing `nu_eff = 1/2` at `r*`.

Error terms in this reduction, none of which are bounded rigorously:
- `O(|d ln r/dt| / (1-b2))`, from `v` lag.
- A factor `chi = 1 + 2 sum_k C_k` where gradients are autocorrelated.
- `O(Var(alpha_i))`, from coordinate heterogeneity. Keep the sums rather than A2 when
  this is large.
- `O(eta lam)` and `O(theta^4)`.

### Crossover and weight decay

Continuum form:

    d(r^2)/dt = -2 eta lam r^2 + eta^2 d / (1 + r/r*)^2

**Equilibrium.** With `R0 = sqrt(eta d/(2 lam))` (Kosson's AdamW norm), `r(1 + r/r*) = R0`.
Define

    Gamma = R0/r* = eps d sqrt(eta/(2 lam)) / G

Then

    x_eq (1 + x_eq) = Gamma,    x_eq = (sqrt(1 + 4 Gamma) - 1)/2
    r_eq = x_eq r*

Crossover at equilibrium (`x_eq = 1`) occurs at `Gamma = 2`, that is

    eps_c = 2 G sqrt(2 lam/eta) / d = 2 gbar / R0

Checks:
- `Gamma << 1`: `r_eq -> R0 - R0^2/r*`.
- `Gamma >> 1`: `r_eq^4 -> eta G^2/(2 lam eps^2)`, equal to the SGD result
  `r^4 = eta' G^2/(2 lam')` under `eta' = eta/eps`, `lam' = lam eps`.
- `Gamma` is invariant under S.

Hence the parameter RMS at equilibrium in the Adam regime is `sqrt(eta/(2 lam))`,
independent of `d`, and `x = eps/g_rms`.

**Rotation at equilibrium does not depend on epsilon.** From `E[N^2] = 1/a^2` and A4:

    theta_eq^2 = (1/a^2 - 1) rho_b  ≈ 2 eta lam (1-b1)/(1+b1)    for every x

At `b1 = 0` and `c = 0`, R gives the sharper exact statement `E[-log cos theta] = -log
a`. This agrees with Kosson's table at both ends: AdamW gives `sqrt(2 eta lam rho_b)`,
and SGDM, with total decay rate `eta lam/(1-alpha)`, gives the same. It also matches
Wan et al.'s `sqrt(2 lam eta/(1+alpha))`. The extension to finite epsilon is one line.
**This is the main break of the conjecture.** Epsilon moves `r` so that it exactly
compensates the shrinking `alpha`.

**Relaxation rate does depend on epsilon.** Linearizing `d ln r/dt` at `r_eq`:

    omega(x) = 2 eta lam (1 + 2x)/(1 + x) = 2 eta lam (1 + nu_eff)

This runs from `2 eta lam` to `4 eta lam`. The SGD end matches Wan et al. (2006.08419)
eq. 36, where the contraction coefficient at equilibrium is `1 - 4 lam eta`.

**No decay: transit law.** Integrating with `lam = 0`:

    t(x) = (r*^2/(eta^2 d)) [T(x) - T(x_0)],   T(x) = x^2 + (4/3) x^3 + x^4/2

At `x_0 = 0`, `t* = (17/6) r*^2/(eta^2 d) = (17/6) (G/(eps d eta))^2`. Large `x` gives
`r^4 = 2 t (eta/eps)^2 G^2`, the SGD result.

Rotation along the way:

    theta^2 t / rho_b = (1 + (4/3) x + x^2/2)/(1 + x)^2

This falls monotonically from 1 to 1/2 (0.71 at `x = 1`). So **across the whole
crossover without decay, the rotation rate changes by at most sqrt(2)**, and smoothly.

With decay, `t* 2 eta lam ≈ (17/6)/Gamma^2`, so reaching the crossover before
equilibrium needs `Gamma` above about 2 anyway.

**The other route to the crossover is `G` falling.** `r*` moves with the gradient, so
interpolation (train loss to 0) can cross it with `r` unchanged or decaying. `Gamma`
then grows as `1/G`.

### Realistic numbers

`gbar` below is an assumption, not a measurement. `E4.1` logs it. Reference: `eta =
1e-3`, `lam = 0.01`, `d = 1000` (one conv filter), so `R0 = 7.07`, `eta^2 d = 1e-3`,
initial `r` about 0.6 to 1.4.

| gbar | eps | r* | Gamma | x_eq | t* without decay (steps) |
|---|---|---|---|---|---|
| 1e-2 | 1e-8 | 1e6 | 7.1e-6 | 7e-6 | 2.8e15 |
| 1e-3 | 1e-8 | 1e5 | 7.1e-5 | 7e-5 | 2.8e13 |
| 1e-4 | 1e-8 | 1e4 | 7.1e-4 | 7e-4 | 2.8e11 |
| 1e-6 | 1e-8 | 100 | 0.071 | 0.066 | 2.8e7 |
| 1e-2 | 1e-3 | 10 | 0.71 | 0.48 | 2.8e5 |
| 1e-3 | 1e-3 | 1 | 7.1 | 2.2 | about 2.8e3 |
| 1e-4 | 1e-3 | 0.1 | 71 | 7.9 | past it at init |
| 1e-6 | 1e-3 | 1e-3 | 7071 | 84 | past it at init |

Epsilon values that matter at these settings:

| gbar | eps with x_eq = 0.1 | eps_c (x_eq = 1) |
|---|---|---|
| 1e-2 | 1.6e-4 | 2.8e-3 |
| 1e-3 | 1.6e-5 | 2.8e-4 |
| 1e-4 | 1.6e-6 | 2.8e-5 |
| 1e-6 | 1.6e-8 | 2.8e-7 |

**Plainly:**
- **Default `eps = 1e-8` on small and medium models:** `r*` is unreachable through norm
  growth, by many orders of magnitude. Only a vanishing gradient reaches it, which is
  the deep-interpolation regime.
- **`eps = 1e-3`:** at or past the crossover almost from the start.
- **The band that matters:** roughly `eps` between 1e-6 and 1e-2 at reference settings.
- **Very large models are the exception.** `Gamma` grows with `d sqrt(eta/lam)`.
  Wortsman et al. measured grad RMS reaching 1e-8 at 4.8B parameters and LR 0.3, with
  the update RMS collapsing. With `d` about 1e8, `lam = 1e-4`, `eta = 0.3`: `Gamma`
  about `39/G`, which is above 2 unless `G` is above about 20. That is
  order-of-magnitude agreement only. `G` was not measured, and those MLP blocks are not
  exactly scale invariant.

Diagnostic for any run: the median of `eps/sqrt(v^_i)` per block, read straight from
optimizer state. It is `x`.

## Prior art (searched 2026-11-04)

Searches (alphaXiv `discover_papers`): "Adam epsilon, effective learning rate,
scale-invariant"; "Adam, epsilon, weight decay, batch normalization, equilibrium";
"Adam, epsilon, SGD with momentum, hyperparameter tuning"; "epsilon, Adam, adaptivity,
denominator"; "epsilon, Adam, weight norm, scale invariance, angular update". Then read
the papers below.

**Already known:**
- Large-epsilon limit, as an inclusion with LR `eps eta(1-gamma)`, and epsilon as an
  interpolation between momentum and diagonal natural gradient: Choi et al. 1910.05446
  (appendix A). They report best epsilon spanning 10 orders of magnitude.
- Optimal LR scales linearly with epsilon; best epsilon of 0.1 to 10 on CIFAR: Savarese
  et al. 1912.01823. On PTB, 1e-8 was best.
- **Closest to the conjecture:** Wortsman et al. 2309.14322 (Section 3.4, Appendix C).
  Grad RMS falls with scale and LR, reaches epsilon, update RMS collapses, 1e-15 helps,
  and they link it to parameter norm growth through the inverse-scale gradient of the
  layernorm. So "epsilon meets the gradient as the norm grows" is empirically known.
  Everett et al. 2407.05872 and Yang and Littwin scale epsilon with width; Adam-atan2
  removes it.
- Norm feedback undermines per-coordinate arithmetic: 2610.38169 finds that raising
  epsilon suppresses a rare row less than fixed-trajectory arithmetic predicts. Sign
  analogy only: at large `Gamma`, fixed-trajectory `alpha = 1/(1+Gamma)` while
  self-consistent `alpha = 1/(1+x_eq)` is larger. That row is not scale invariant, and
  their miss is not evidence for H4.

**Explicitly ignored elsewhere:** Kosson et al. 2305.17212 set "eps ... effectively
zero" in the equilibrium analysis. 2609.09116 lists formal epsilon perturbation theory
as open.

**Not found, which is weak evidence of absence from five searches:** `r*`, `Gamma`, the
closed-form `r_eq`, `omega(x)`, `nu_eff`, the transit law, and epsilon-independent
equilibrium rotation at finite epsilon. Some of it is a one-line extension of Kosson's
table, so novelty is modest. Not checked: Yang and Littwin directly, and De et al. 2018,
Zaheer et al. 2018 beyond what Choi says of them.

## Assumptions that could break it

- **A4 fails when signal dominates.** With persistent gradients the `b1` cancellation
  changes, and rotation picks up `rho_chi/chi`, which may depend on epsilon through the
  preconditioner.
- **Stable fixed point.** The reduction gives a stable equilibrium, but the paper
  proves the interior balance is an unstable spiral in its normalized-linear model
  under full batch. There the "equilibrium" here may not exist. `nu_eff` may change
  stability; this is not derived.
- **Large epsilon can destabilize.** Wortsman et al.'s 1e-6 run diverged at 4.8B
  parameters. Nothing here predicts that. It is a real gap: the law says larger epsilon
  means stronger quenching.
- **Coordinate heterogeneity.** With heavy-tailed `v_i`, the crossover is spread over
  many decades in `r`, and "a crossover at `r*`" is a coarse summary. Low-gradient
  coordinates freeze first.
- **Non-stationary `G`.** `G` changes through training and depends on the rotation rate,
  which couples back into `r`.
- **Scale invariance is imperfect.** BN epsilon, biases, shared block boundaries. Most
  transformer blocks are not covered, as the paper itself says.
- **Fast `v`.** At small `b2` or large `eta lam`, `v` lag matters (`omega` near `1-b2`).
- **Large-step regime.** `theta` is not small at small `r`.

## Experiments

**E4.0, float64 checks** (`experiments/E4_0_epsilon_law/check.py`, plain script, numpy
required, torch used if present, under a minute expected).

Part A, identities, on a synthetic scale-invariant logistic loss:
1. K and D, across 5 epsilons and 3 schedules.
2. R.
3. C versus direct AdamW.
4. S, both symmetries.
5. Limits: error slopes in epsilon, plus Theorem 2.1 in the large-epsilon reference.

Part B: the same one-step identities against real `torch.optim.AdamW` on a BN MLP.

Part C, mean-field closed forms on an isotropic noise walk with `b2 = 0.99`:
`r_eq`, `theta_eq`, `omega(x)` at `x` = 0, 1, 10, and the transit law.

Kill: any A or B failure, after an audit of the code, kills the derivation. C failing
beyond its tolerances (5% for `r_eq`, `theta_eq` and transit `x`; 15% for `omega`) kills
the closed forms, and E4.1 is not run. Passing A and B shows only that the algebra is
right.

**E4.1, BN MLP on synthetic data** (Colab CPU, about 90 s per run expected, about 30
min total).

Model and data: MLP 100-256-256-10 with `bias=False` hidden Linears, each followed by
non-affine BN without running statistics, ReLU. Blocks are hidden rows. Fresh Gaussian
inputs each step, labels from a fixed random teacher with 20% uniform noise, batch 128.
Fresh samples avoid interpolation, keeping `G` stationary. AdamW `eta = 1e-3`, `lam =
0.1`, betas (0.9, 0.99). Rows start rescaled to `R0`.

Procedure: 20000 steps, then multiply the first hidden layer's rows by 1.5 and run
12000 more. A pilot at `eps = 1e-12` gives `g* = ` median `sqrt(v^_i)` over the last
2000 pre-perturbation steps. Grid: `1e-12` and `g*` x {0.03, 0.1, 0.3, 1, 3, 10}. 3
seeds. This grid rule is fixed now and is not to be tuned. Measure over the last 3000
pre-perturbation steps, per block then median: `r`, `theta^2`, `alpha_i`, `nu_eff`.
Fit `omega` from `ln(r/r_pre)` between 0.25 and 0.08.

Predictions, relative to the `1e-12` arm of the same seed:
- **P1:** `r^2` ratio equals the measured `mean(alpha_i^2)`.
  Kill: off by more than 1.5x at any arm with `mean(alpha^2) >= 0.05`, in at least 2 of
  3 seeds.
- **P2, the discriminating one:** the `theta^2` ratio stays within [0.6, 1.6] at every
  arm. The effective-LR reading predicts a falling ratio, down to about `alpha^2`.
  Kill: outside the band at 2 or more arms in all 3 seeds. That kill favors the
  coordinator's conjecture over this document.
- **P3:** `omega` ratio equals `1 + nu_eff`.
  Kill: the ratio at the largest-epsilon arm is below 1.25 or above 2.5 in all seeds,
  or Spearman of `omega` against `nu_eff` is below 0.5.

**E4.2, default-epsilon diagnostic** (one run, minutes). Small BN CNN on FashionMNIST,
AdamW, `eps = 1e-8`, `eta = 1e-3`, `lam = 0.01`, 10 epochs. Log median `kappa` per
filter and train loss.
Prediction: median `kappa` stays below 1e-2 while train loss is above 0.01.
Kill for the "unreachable by norm growth" claim: median `kappa >= 0.1` in any filter
quartile while train loss is above 0.1.

No performance experiment is registered. Epsilon tuning is already known to matter
(Choi, Savarese), and this hypothesis predicts nothing new about the best value beyond
a starting point near `eps_c`. Any such test would need its own registration and tuned
baselines.

## Honest assessment

- E4.0 is cheap and should run, mainly because this algebra has not been run by anyone.
- E4.1 is worth running only if compute is spare. It would turn one clean negative into
  a measured result: epsilon is not an effective-LR schedule at equilibrium.
- This is not a route to a speedup. The useful outputs are a one-number diagnostic,
  `eps/sqrt(v^)`, and the formula `eps_c = 2 G sqrt(2 lam/eta)/d`, which turns the
  empirical trend in Wortsman et al. into a scaling law. Even that scaling is untested
  on transformers.
