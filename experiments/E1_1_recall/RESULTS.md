# E1.1 and E1.2: attention normalizers on recall length extrapolation

Run on Colab (CPU), 2026-11-04. 63 tuning runs and 105 final runs: 7 arms, 3 noise
levels, 5 paired seeds, trained at `N = 8` and tested to `N = 256`. No non-finite
training steps in any run. Per-run accuracy is in `results_compact.csv`. The full JSONL
files and the full analysis stayed on Colab because of their size.

## Verdict

**H1's pre-registered kill fired at every noise level, and these are losses, not
ties.** The clustered-distractor run is skipped, as pre-registered.

| Noise | N | REM minus best baseline, paired 95% interval | Best baseline |
|---|---|---|---|
| 0.0 | 128 | -0.006 [-0.012, -0.000] | entmax15 |
| 0.0 | 256 | -0.015 [-0.026, -0.004] | entmax15_scaled |
| 0.6 | 128 | -0.045 [-0.065, -0.024] | entmax15_scaled |
| 0.6 | 256 | -0.065 [-0.074, -0.056] | entmax15 |
| 0.9 | 128 | -0.036 [-0.064, -0.007] | entmax15 |
| 0.9 | 256 | -0.034 [-0.054, -0.015] | entmax15 |

The prediction that REM's gain over SSMax grows with noise is not supported either. The
analysis script prints "consistent" because the largest value is at the highest noise,
but that value is -0.0002 to +0.006 with intervals spanning zero: a tie, not a gain.

REM is a tie with SSMax at `N = 256`, using no learned parameters, at every noise level.
That is the most that can be said for it.

## Accuracy, mean over 5 seeds (needle mass, best head)

| Noise | N | Matched filter | softmax | learned | ssmax | entmax15 | ent_scaled | rem | calib |
|---|---|---|---|---|---|---|---|---|---|
| 0.6 | 8 | 0.999 | 0.982 (.91) | 0.982 (.87) | 0.983 (.85) | 0.987 (.95) | 0.986 (.89) | 0.974 (.90) | 0.177 |
| 0.6 | 64 | 0.994 | 0.891 (.60) | 0.886 (.48) | 0.898 (.79) | 0.918 (.79) | 0.911 (.77) | 0.885 (.77) | 0.073 |
| 0.6 | 256 | 0.982 | 0.737 (.31) | 0.713 (.21) | 0.776 (.67) | 0.814 (.60) | 0.803 (.62) | 0.749 (.59) | 0.064 |
| 0.9 | 8 | 0.969 | 0.887 (.70) | 0.884 (.68) | 0.880 (.67) | 0.903 (.77) | 0.895 (.71) | 0.891 (.81) | 0.180 |
| 0.9 | 64 | 0.844 | 0.611 (.26) | 0.604 (.24) | 0.638 (.50) | 0.671 (.46) | 0.659 (.51) | 0.645 (.52) | 0.071 |
| 0.9 | 256 | 0.684 | 0.384 (.09) | 0.370 (.08) | 0.428 (.33) | 0.462 (.27) | 0.448 (.34) | 0.428 (.32) | 0.067 |

At zero noise every arm except calib is 0.979 or better at `N = 256`, so that level
barely separates them.

Paired differences at `N = 256`, with no multiplicity correction:

| Comparison | Noise 0.6 | Noise 0.9 |
|---|---|---|
| entmax15 - softmax | +0.077 [+0.049, +0.105] | +0.078 [+0.053, +0.102] |
| entmax15 - ssmax | +0.038 [+0.005, +0.071] | +0.034 [+0.002, +0.066] |
| ssmax - softmax | +0.039 [+0.015, +0.064] | +0.043 [+0.028, +0.059] |
| rem - softmax | +0.012 [-0.014, +0.039] | +0.043 [+0.023, +0.063] |
| entmax15 - entmax15_scaled | +0.011 [-0.009, +0.031] | +0.013 [-0.010, +0.036] |

The entmax15 over ssmax intervals only just exclude zero and would likely not survive a
correction for the number of comparisons.

## What the audit shows (E1.2)

- **REM's mass control works as designed, and that did not produce a win.** Top mass
  is 0.92 at `N = 8` against a 0.9 target, at noise 0 and 0.6, and falls as the clamp
  engages at long lengths.
- **Trained scores are not heavy-tailed here.** Median excess kurtosis of cross-attention
  rows is -0.2 to -1.1, lighter than Gaussian, so the E1.0 heavy-tail failure is not
  what went wrong. Light, bounded tails are closer to the deterministic models that give
  `log n` scaling (2510.05554). This is an observation, not a test.
- **The formula did not predict learned temperatures well.** Learned SSMax `beta` against
  the REM formula across lengths: 0.157 / 0.083 to 0.409 / 0.144 at noise 0, 0.113 / 0.133
  to 0.296 / 0.216 at 0.6, and 0.138 / 0.192 to 0.395 / 0.308 at 0.9. The formula's growth
  flattens with length while learned `beta` keeps rising.
- **The E1.2 kill was badly designed.** It asks whether the formula beats `c log n` at
  predicting SSMax's learned `beta`, but SSMax's `beta` is exactly `c log n` within a
  model, so the formula cannot win. It fires by construction and carries little
  information. The rows above are the useful content.

## Correlates of accuracy

Needle mass does matter: across arm-seed cells at `N = 256`, Spearman with accuracy is
0.75 at noise 0.6 and 0.71 at noise 0.9. But equal mass does not give equal accuracy. At
noise 0.6 and `N = 256`, entmax15 has mass 0.60 and accuracy 0.814, while REM has mass
0.59 and accuracy 0.749. REM's extra mass over softmax (0.59 against 0.31) brings no
measurable gain at noise 0.6, although it does at 0.9. So holding needle mass up is not
sufficient, and whatever entmax does beyond that is unexplained. H5 tests this.

Plain entmax15 is the strongest arm overall, with no length term. Adding learned length
scaling did not measurably improve on it in this task.

## Calib-Attn failed to learn

It never got past about 0.18 at the training length, with a median gradient norm of 19
against about 1.2 elsewhere. Forcing 0.9 mass on whichever token is top, starting from an
untrained model, appears to prevent bootstrapping. REM avoids this because its clamp
lowers `beta` when there is no clear needle, and that is the cleanest thing this arm
shows. It does **not** test whether the Gaussian closed form matters, which was its
purpose, because the arm failed for a different reason.

## Limitations

- **Learning rate at the top of the grid.** Every arm except calib selected `lr = 0.003`,
  the largest tried, so a higher rate might change the comparison. H5 checks this.
- **One tuning seed** per noise level, for every arm.
- **Small synthetic task:** 71k parameters, trained for 2500 steps, with fresh random
  keys. REM was favored by this design and still lost, so the negative is meaningful for
  H1. The relative ranking of the other arms may not transfer to language models.
- **Timings are not clean.** The Colab runtime was shared with other work.
