# H2: gating Muon's spectrum at a measured noise edge

Status: pre-registered 2026-11-04, before any experiment. Prior that it beats
tuned Muon: about 20%. The evidence against is 2610.10965, where top-only
orthogonalization lost badly, and 2605.11181, where random spectra matched Muon.
It is still worth running because it tests a different cut: a measured noise edge
rather than a fixed count of top directions.

## Idea

Muon replaces momentum `M = U S V^T` by `U V^T`, which amplifies every direction
equally, including ones that are only minibatch noise. Spiked random matrix theory
(BBP transition) says a low-rank signal is recoverable only above the noise bulk
edge. Normally that edge needs a noise model. It can instead be measured.

## Formula

Keep two momentum buffers on independent half-batches, `M1` and `M2`:

    M = (M1 + M2) / 2        N = (M1 - M2) / 2

The signal cancels in `N`. For independent half-batches, `N` has the same mean
(zero) and covariance as the noise in `M`. So `e = ||N||_2` estimates the noise
edge with no Marchenko-Pastur assumption.

    S_keep = { i : sigma_i(M) > gamma e },  gamma = 1 by default
    Update  = U_keep V_keep^T * c

`c = 1` is the raw gate. `c = sqrt(rank / |S_keep|)` keeps Muon's Frobenius norm,
which separates which directions are kept from the size of the step. That control
is required because 2605.11181 credits Muon's success to step-size stability.

## Prediction

- The kept fraction rises with batch size and with momentum averaging. Gate-renorm
  converges to Muon at large batch and differs most at small batch. This gives one
  possible mechanism for optimizer rankings flipping with batch size (2610.08975).
- Gate-renorm beats tuned Muon at small batch.

## Experiment E2.1

Tiny character-level GPT and an MLP on Fashion-MNIST, batch sizes 32 and 256, at
least 5 seeds. Arms: AdamW, SGD with momentum, Muon, Freon `c = 2/3`, gate-raw,
gate-renorm, and random-subset control (same `|S_keep|`, random directions,
renormalized). Learning rates are tuned on the same grid per arm using validation
data only. Report equal-data and equal-time curves, because the gate costs extra
SVDs.

## Kill criteria

- Random-subset control matches gate-renorm within seed noise: selection does not
  matter, so the mechanism is wrong.
- Gate-renorm is not better than Muon at either batch size, with paired 95%
  intervals excluding zero in its favor.

The kept-fraction curve and the measured edge are worth reporting either way, as a
direct measurement of how much of the momentum spectrum clears noise.
