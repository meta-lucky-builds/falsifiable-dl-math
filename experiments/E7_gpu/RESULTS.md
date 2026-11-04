# E7 / H7 results: the experiment did not test its question

> **Correction, 2026-11-04.** The toy run of [E8](../E8_long_context/RESULTS.md) found no sign of a bug, and found that raising `theta` 50x is not a
> fix. So `theta` is not the setting to blame, as the explanation below implies, and wraparound itself remains untested. The rest of this file stands as written at the time.

Run on rented 4x RTX 5090, 2026-11-04: 12 tuning and 16 final runs, 2.7M parameters, 20000
steps, 4 seeds, no non-finite steps. Per-run data, pasted back from the machine:
`results_compact_run2.csv`. A first complete run was lost when its instance was destroyed before
the results were copied. It is not used here, and I never saw its numbers.

## Verdict

**The pre-registered outcomes are recorded below, but they do not answer the question.** Every model
broke as soon as context passed the training length, so the arms were compared in a failure regime.

## What the pre-registered rules say

| Test | Mean gain, bits per byte | Holm p | Outcome |
|---|---|---|---|
| P1 entmax15 lower than softmax | -0.371 | 1.000 | Not supported |
| P2 entmax15 lower than ssmax | +0.628 | 0.136 | Inconclusive |
| P3 softmax model, entmax15 readout lower than softmax readout | -1.765 | 1.000 | Not supported |

- Kill for transfer of E1.1: **fired**.
- Readout swap: **does not succeed**. P3b was not tested.
- G(N) rule: **fired**. The rule states that the loss comparison is uninformative and only the copy
  probe can be read.
- Copy probe: about 0.011 per byte at every distance and arm, and 0.000 for all 4 bytes. That is
  below the 1/62 = 0.016 expected from guessing uniformly in the probe's alphabet. So it cannot be read either.

## Why the outcomes should not be taken at face value

| Arm | L(256) | L(512) | L(1024) | L(2048) |
|---|---|---|---|---|
| softmax | 1.394 | 5.230 | 5.179 | 5.366 |
| ssmax | 1.397 | 5.545 | 6.075 | 6.469 |
| entmax15 | 1.389 | 5.244 | 5.412 | 5.875 |
| entmax15_scaled | 1.390 | 5.261 | 5.425 | 5.949 |

Loss rises by about 3.8 bits per byte at just 2x length, for every arm. These bytes are near the training
context, and the same model predicts them at 1.39 bits per byte given only 256 bytes of context
(`T`). So giving the model more context destroys its predictions.

The rule's wording, "long context carries no usable information", is therefore wrong for this result.
Enwik8 certainly has long-range information. What happened is that the model fails on long inputs.

**Likely cause, not yet confirmed: RoPE with `theta = 10000` at a training length of 256.** Distances past
255 were never seen in training, and the slower RoPE frequencies wrap around at those distances, so
attention may land on arbitrary far tokens. My own survey had the warning: SSMax (2501.19399) only
extrapolated with `theta` changed 50x, and 2601.15380 reports plain RoPE going from perplexity 20.82 to
417 at 8x length. **I missed this when I fixed "RoPE only" in the design.** That is a design error
of mine and not a finding about these methods. A code bug is less likely but not ruled out: `T` and
`L(256)` agree closely, and an independent review found the attention path correct, but no test measured
loss per row across a long window. Checkpoints were deleted after each job, so these models cannot be
examined further.

The copy probe had a design flaw too: every probe input was 2048 bytes long, so even `D = 128` ran
in the failing regime. It could not have measured retrieval.

## What these runs do show

Short context is the only regime where the models worked. These are descriptive, and none was a
primary test:

- **At `N = 256`, the arms are close.** entmax15 minus softmax is -0.0048 bits per byte, with a 95%
  interval of about [-0.013, +0.003] over 4 seeds, by my hand calculation from the CSV. So there is no
  demonstrated gain, and no large cost either.
- **Naive sparse readout hurts.** Reading a softmax-trained model with entmax15 raises `L(256)` from 1.394 to
  1.653 bits per byte, at a length where the model works. This is the most solid result here: consistent
  in all 4 seeds, and independent of the RoPE problem. Any sparse readout would need recalibrating.
- **Cost.** entmax15 runs about 29% slower per step than softmax here (21.7 ms against 16.9 ms).
- **Learning rate.** All arms selected 0.0012, the top of the grid, as in E1.1, so all may be under-tuned.

## What would actually test the question

Fix the failure first, with a small and cheap diagnostic, before any more multi-seed compute: measure
loss per row across a long window, for RoPE at more than one `theta` and a scheme with no wraparound such
as ALiBi. Only if a setup survives past training length is a comparison of normalizers meaningful. That
needs its own pre-registration, with fresh seeds.
