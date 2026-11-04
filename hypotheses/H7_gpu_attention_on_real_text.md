# H7: attention normalizers on long contexts in a real language model (GPU)

Status: pre-registered 2026-11-04, before any H7 run and before H5 finished. To be run by the
owner on 4 GPUs. Everything below is fixed unless an amendment is recorded before the full run.

Priors: entmax15 beats softmax at long context on this task, 35%. Sparse readout helps a
softmax-trained model, 25%. Long context helps softmax at all compared with truncation, 60%.

## Why this experiment

E1.1 suggested plain entmax-1.5 beats softmax on long contexts, but on a 71k-parameter synthetic
task with independent random keys. SSMax (2501.19399) and scaled entmax (2506.16640) make
related claims, with different baselines and, for 2506.16640, a positional change at the same
time. This tests them together on real text, at one positional scheme, with equal tuning.

The arm that could be new is the **readout swap**: train with softmax, then read out sparsely at
inference. If it works, it applies to an already trained model with no retraining.

## Setup

- **Data:** `enwik8`, bytes as tokens, split 90/5/5 in file order into train, validation, test.
- **Model:** about 10M parameters. 6 layers, 6 heads, width 384, feed-forward 4x, pre-norm,
  no dropout, RoPE only (no learned position table, so it can run past training length).
- **Training:** context 256, batch 64, 30000 steps, AdamW with weight decay 0.1 on matrices, betas
  (0.9, 0.99), cosine decay with 1000 warmup steps, gradient clip 1.0, bf16.
- **Trained arms:** `softmax`, `ssmax` (learned `s` per head), `entmax15`, `entmax15_scaled` (learned
  `delta`, `b`, `gamma` per head). Rows use causal masks, so `n` is the row's valid count.
- **Not included:** the optimal-transport prior of 2601.15380, because it replaces the positional
  scheme, which this test holds fixed.
- **Tuning:** one tuning seed per arm, full length, `lr` in {3e-4, 6e-4, 1.2e-3}, chosen by
  validation loss at context 256 only. If an arm selects an edge of the grid, the report says so.
- **Final:** 5 seeds per arm, at each arm's selected rate, sharing initial weights across arms by
  seed and the training batch stream by seed.

## Evaluation

All on test data, with the same fixed windows for every arm and seed. At `N` in {256, 512, 1024,
2048}, on at least 2048 windows:

- **`L(N)`:** next-byte loss in bits per byte, on the last 64 positions of a length-`N` window.
- **`T(N)`:** same positions, same targets, with the model given only the preceding 256 bytes for
  each position (truncated sliding context). This measures what long context adds.
- **Gain `G(N) = T(N) - L(N)`.** Positive means the longer context helped.
- **Copy probe:** a random 8-byte string from a fixed alphabet is inserted at distance `D` in {128,
  512, 1024, 1900} from the end of a real-text window, then its first 4 bytes are repeated at the
  end. Accuracy is on predicting the next 4 bytes. It measures long-range retrieval directly.
- **Readout swap, on `softmax`-trained models only, with no retraining:** repeat all of the above with
  `entmax15`, and with fixed SSMax (`s = 1 / log 256`, so it equals softmax at `N = 256`).
  Same scores' scale as trained. Entmax here changes the short-context readout too, and `L(256)`
  will show the cost.

## Primary tests

Paired over 5 seeds, one-sided direction stated, Holm correction across P1, P2 and P3, at family
level 0.05, using t-tests on the mean over `N` in {1024, 2048}:

- **P1:** `entmax15` has lower `L` than `softmax`.
- **P2:** `entmax15` has lower `L` than `ssmax`.
- **P3:** `softmax`-trained with entmax15 readout has lower `L` than with softmax readout.

## How results will be read, fixed now

- **Kill for transfer of E1.1:** P1 not significant. Then the synthetic entmax gain does not
  transfer to this real-text setup at this scale, and that gets recorded plainly.
- **Readout swap succeeds** only if P3 is significant, and `L(256)` rises by under 0.005 bits per
  byte on average, or a length-gated version (softmax up to `N = 256`, sparse only beyond it) is
  also better. That gated version is computed from the same runs.
- **Practical threshold.** A significant gain under 0.005 bits per byte is reported as real but
  negligible.
- **If `G(N)` is not positive for any arm**, long context carries no usable information at this
  scale, the loss comparison is uninformative, and only the copy probe can be read. Report that.
- **Cost:** report `L(256)`, wall time per step and tokens per second per arm.
- Copy-probe differences are secondary and reported with intervals, without correction.

## Amendment, 2026-11-04, before any run

Found while implementing, with no data seen. This replaces the gated clause above.

- **Row-level gate.** The window-level gate I specified makes that clause identical to P3, so it
  could never fail. The gated readout is now per row: softmax for rows with `n <= 256`, and
  entmax15 for rows with `n > 256`. It equals softmax exactly at `N = 256`.
- **The readout swap succeeds** only if P3 is significant, and either `L(256)` rises by under 0.005
  bits per byte, or **P3b** holds: gated has lower `L` than softmax readout on the mean over `N` in
  {1024, 2048}, one-sided paired t-test at 0.05. P3b is tested only if P3 is significant, so it
  adds no new chance of a false positive to the family.
- Fixed SSMax readout equals softmax on rows with `n = 256`, not across a whole window of length 256.
- `T(256)` sees bytes before the window, so `G` is read only for `N > 256`.
- Validation and test run in float32. Training is bf16, with the attention core, whose cumulative
  sums are too coarse in bf16, kept in float32 for every arm. This may help sparse arms, and it is
  not the same as a bf16-only setup.
- Probe inputs are all 2048 bytes, so only the distance changes.

## Amendment 2, 2026-11-04, before any run: smaller and cheaper

The owner asked for less compute. This replaces the corresponding values above.

- **Model:** width 192, still 6 layers and 6 heads, about 2.7M parameters. **Steps:** 20000, about 3.6
  passes over the training text. Warmup, batch, context and the learning-rate grid are unchanged.
- **Final seeds:** 4, so 28 runs in total. Not fewer: with Holm correction the strictest test needs
  a one-sided t of about 3.7 with 4 seeds, against about 5.3 with 3.
- **P1, P2, P3 now have three outcomes**, since a miss at 4 seeds may only mean too little data:
  - **Supported:** significant under Holm.
  - **Inconclusive:** not significant, but mean gain at least 0.005 bits per byte. It needs new seeds
    to settle. Those would be a separate confirmatory run, not pooled with these.
  - **Not supported:** not significant, and mean gain under 0.005 bits per byte. For P1, this is the
    kill for transfer of E1.1.
- **Scale caveat.** Prior claims for these methods come from larger models, so a null here is weaker
  evidence against them than it would be at 10M.

## What it cannot show

One dataset, one size and byte-level tokens. A win here is not a result about large language
models. A loss does not refute the claims of 2501.19399 or 2506.16640 at their own scales and
setups. Learned-scale arms get the same 20000 steps as the others. P3b is read only inside the swap
rule, never as a claim of its own.
