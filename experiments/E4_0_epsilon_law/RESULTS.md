# E4.0: checks for H4 (Adam epsilon in the scale-invariant schedule law)

Run on Colab (CPU) at low priority, 2026-11-04. `check.py` at sha256
`9dd94a04...c8ba1b1`. The first run stopped in Part B because torch rejects BatchNorm
`eps=0` (see `run_output.txt`). The fix uses `eps=1e-300`, and check B5 confirms scale
invariance is still exact. No tolerance was changed.

## Verdict

**The pre-registered kill fired on one closed form, so E4.1 will not run.** The
exact algebra is confirmed. Most closed forms held closely.

## Part A: exact identities, all pass

| Check | Value | Limit |
|---|---|---|
| A1 kinematics K | 1.1e-15 | 1e-11 |
| A2 decomposition D | 3.2e-16 | 1e-11 |
| A3 rotation budget R | 5.4e-15 | 1e-10 |
| A4 closure C, `Psi` / `u` | 2.6e-14 / 1.9e-14 | 1e-8 |
| A5 scale symmetry, `u` / `r` | 9.7e-16 / 4.4e-16 | 1e-8 |
| A6 loss-scale symmetry | 2.3e-16 | 1e-8 |
| A7 slope as `eps -> 0` | +0.9998 | 0.8 to 1.2 |
| A8 slope as `eps -> inf` | -0.9998 | -1.2 to -0.8 |
| A9 paper's Theorem 2.1 at large `eps` | 4.4e-16 | 1e-11 |

## Part B: real `torch.optim.AdamW` on a BN MLP, all pass

Step formula 2.1e-16, `N` 4.4e-16, `Phi` law 1.1e-15, `Psi` law 6.7e-16, BN scale
invariance 7.9e-16, tangent gradient 2.4e-16.

## Part C: mean-field closed forms on an isotropic noise walk

| Quantity | `x = 0` | `x = 1` | `x = 10` | Tolerance |
|---|---|---|---|---|
| `r_eq`, relative error | 0.51% | 0.22% | 0.07% | 5% |
| `theta_eq`, relative error | 0.02% | 0.12% | 0.01% | 5% |
| `omega`, measured / predicted | 1.76e-4 / 2.00e-4 | 2.55e-4 / 3.00e-4 | 3.10e-4 / 3.82e-4 | 15% |
| `omega`, relative error | 12.1% | 14.9% | **18.8%, FAIL** | 15% |

Transit law without decay: 1.1%, 0.3% and 0.2% error at `x` = 1, 2 and 3, against a 5%
limit.

## What holds and what does not

- **Confirmed on this noise walk:** rotation at equilibrium does not depend on
  epsilon. `theta` stays at 3.24e-3 while `r` moves from 2.22 to 0.20. This is the
  result that broke my stronger conjecture, so **epsilon is not an effective-LR
  schedule at equilibrium, in this model.**
- **Killed:** the relaxation formula `omega(x)`.
- **Not tested:** whether any of this holds on a real network. E4.1 was the test for
  that, and the pre-registration says it does not run after this kill.

An observation made after seeing the data, and not a rescue: all three `omega` values
are low by a similar margin, and the ratio to `x = 0` is closer (1.45 against a
predicted 1.50, and 1.76 against 1.91). That could mean a shared bias, for example from
the large 1.3x perturbation or the fit window. Nothing tested that, and I did not
change the fit to find out.

## Consequence

The H4 thread stops here. It leaves a verified exact recursion and a verified
closed form for equilibrium norm and rotation in a toy model, with no route to better
training. Compute goes to other threads.

Output of the final run is in `run_output_2.txt`.
