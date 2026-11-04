# E8 / H8 toy run: the long-context failure is not a bug, and raising `theta` makes it worse

Run on 1x RTX 5090, 2026-11-04, from commit `bb4635f`, in about 18 minutes: 4 arms, 2 seeds, 6000 steps, 512 test
windows, 2.7M parameters, no non-finite steps, the same evaluation windows for every run. Unedited output:
[`report_toy.txt`](report_toy.txt). Per-run data: [`results_compact_toy.csv`](results_compact_toy.csv).

**Status of this evidence.** H8's second amendment, written before this run, says a toy run sets a direction only, so
any claim about a cause or a fix needs the full run. The effects below are large and appear in both seeds, but
two seeds and a third of E7's training cannot support more than that.

## Verdict on the pre-registered rules

| Rule | Measured | Limit | Outcome |
|---|---|---|---|
| C0, measurement, base arm | `Lw(2048)` 1.488, `T` 1.488 | `T + 0.03` | Pass, also in the other 3 arms |
| C1, positive control | control `L(1024)` 1.419 | base `Lw(1024)` 1.467 + 0.10 | Pass |
| C2, reproduction | base `L(512)` 3.918, `E(512)` +2.425 | at least 1.0 | Pass |
| C3, mechanism | far over near, per token: 22.05 | at least 2 | Supported |
| Survival | `alibi` yes; `rope_1e4`, `rope_5e5` and the control no | see H8 | ALiBi only |

The rule on E7's claim prints "supported", because RoPE fails and linear decay survives. **The claim as I wrote it in
E7 was only partly right.** See "What I got wrong" below.

## Results

| Arm | L(256) | L(512) | L(1024) | L(2048) | E(2048) | Far over near |
|---|---|---|---|---|---|---|
| `rope_1e4` | 1.480 | 3.918 | 4.637 | 5.128 | +3.640 | 22.05 |
| `rope_5e5` | 1.486 | 4.764 | 5.199 | 5.277 | +3.785 | 27.88 |
| `alibi` | 1.475 | 1.464 | 1.421 | 1.444 | -0.039 | 0.25 |
| `control_ctx4x` | 1.486 | 1.466 | 1.419 | 4.513 | +3.068 | 0.38 |

Bits per byte, mean over 2 seeds. `E` is loss against the same model with attention limited to its training length, on
identical targets.

## What this shows

- **No sign of a bug in the long-row path.** With far columns removed, long rows give the same loss as truncated input
  (1.488 against 1.488). The control trains well at 1024, so long rows work when trained.
- **The break follows the training length, not any absolute position.** The base breaks just past 256 (1.48, then 1.61,
  then 2.79). The control is fine at those same rows and breaks just past 1024 (1.40, then 1.52, 1.71, 1.92, ending at
  4.51). A fault tied to a position in the code could not move like that.
- **Attention moves to distances the model never saw.** At `N = 512`, the base gives each token at distances 256 to 511
  about 22 times the weight of a token at distances 64 to 255. By `N = 2048`, 93% of its attention mass is past
  training length, averaged over layers.
- **It is not a wraparound with a visible return.** Mean logit relative to the row's top is -13.7 at distances 64 to 256,
  then flat at -9.4, -10.2 and -9.8 further out. So unseen distances sit about 4 nats above seen distractors at every range,
  with no recurring peak at this resolution. The buckets are coarse, so narrow peaks cannot be ruled out.
- **Raising `theta` 50x makes it worse.** In the first bin past training length, loss is 2.77 against 1.61, and `L(512)` is
  4.76 against 3.92, in both seeds. This is softmax only. SSMax's reported extrapolation with 50x `theta` (2501.19399)
  was a different setup and is not contradicted.
- **ALiBi holds to 8x training length, with no cost at short context.** `L(256)` is 1.475 against 1.480. Far mass stays
  near 7%. At `N = 1024` it ties a RoPE model trained at 1024 on the same targets (1.421 against 1.419).
- **ALiBi retrieves past its training length.** Copy accuracy at `D = 400` is 0.161 and 0.110, against 0.016 and 0.015
  for RoPE, where guessing gives 1/62 = 0.016. At `D = 800` it is 0.044 and 0.040. By `D = 1600` it is at chance. This is
  per-byte accuracy, and getting all 4 bytes right stays rare (0.004 at `D = 400`).

## Reported, not read

- **A small gain from long context with ALiBi.** `E(1024)` is -0.047 and -0.040, and `T - L(2048)` is +0.039. The rule
  printed "long context useful", but H8's second amendment says this is reported only at 2 seeds.
- **The failure may grow with training.** `L(512)` is 3.92 here at 6000 steps and 5.23 in E7 at 20000, but on
  other seeds and windows. This is suggestive only.
- **Step times are not clean,** because 4 workers shared the GPU. Ignore the cost table in the report.

## What I got wrong

- **E7's wraparound story.** I blamed the slow frequencies of RoPE at `theta = 10000` wrapping around. A higher `theta`
  moves those frequencies away and the failure got worse, not better, and the logits show no return.
- **My priors.** I gave raising `theta` a 30% chance of surviving. It failed clearly. ALiBi at 75% and bug at 10% held up.
- **Size of the effect.** The gap to GOAT's reported +0.45 bits at 2x (2601.15380) is still unexplained. Mine is +2.4
  at 2x here and +3.8 in E7. Byte-level tokens, model size and softmax are candidates, and none is tested.

## What it cannot show

- **Whether RoPE causes the failure, or whether decay cures it.** `alibi` both removes RoPE and adds decay. The arm
  that separates them, `rope_1e4_alibi`, was left out of the toy run. The full run has it.
- **Why seen distances sit lower.** I measured that unseen distances score higher. That training pushes seen distractors
  down is my reading, and nothing here tests it.
- **Anything beyond this setup.** One dataset, one model size, softmax only, no tuning.
