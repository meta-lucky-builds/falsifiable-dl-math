# H8: why do byte-level models break past their training length?

Status: pre-registered 2026-11-04, before any E8 run. This is a diagnostic. It compares positional schemes
to find a setup that survives past training length. It does not compare attention normalizers, and a win
here is not a claim that any scheme is better in general.

Priors, fixed now:

| Claim | Prior |
|---|---|
| E7's failure reproduces on fresh seeds | 90% |
| Attention mass moves to unseen distances, which is the main mechanism | 65% |
| A code bug in the long-row path causes most of the failure | 10% |
| Raising RoPE `theta` 50x alone survives | 30% |
| ALiBi survives | 75% |
| RoPE plus ALiBi survives | 65% |
| Some setup gets a measurable gain from long context at this model size | 35% |

## Why this experiment

In E7 every arm went from 1.39 bits per byte at `N = 256` to about 5.2 at `N = 512`. Order-0 entropy of this
text is near 5, so those rows predict about as well as byte frequencies alone. E7 blamed RoPE at
`theta = 10000`, without evidence.

**The size of the failure is a warning.** GOAT (2601.15380) reports RoPE going from perplexity 20.82 to 28.53
at 2x its training length, about +0.45 bits. Ours is +3.8 bits. That gap could come from our scale or task, but it
could also be a bug. So this design has to be able to catch a bug, and not only to confirm the RoPE story.

Doubling the row length cannot explain this through the token count alone, since doubling `n` costs at most
one bit if far tokens score like near ones. If attention is the mechanism, far tokens must score well *above*
in-range distractors. A sum of cosines fitted on `d < 256` is free to do that at unseen `d`, which is what
the far-mass and distance measurements below look for.

## Setup

Same as E7: byte-level `enwik8` split 90/5/5, 6 layers, 6 heads, width 192, 2,715,264 parameters, 20000 steps, batch
64, context 256, bf16 training with a float32 attention core, softmax only. Learning rate fixed at 1.2e-3, E7's
selection, for every arm with no tuning, so these arms are not tuned comparisons. 3 fresh seeds: 3000, 3001, 3002.
Seeds share initial weights and batch streams across arms where shapes allow.

| Arm | Position scheme | Training context | Why |
|---|---|---|---|
| `rope_1e4` | RoPE, `theta = 10000` | 256 | Reproduces E7 |
| `rope_5e5` | RoPE, `theta = 500000` | 256 | 50x `theta`, as in SSMax (2501.19399) |
| `alibi` | ALiBi only | 256 | Far scores fall linearly by construction |
| `rope_1e4_alibi` | RoPE and ALiBi | 256 | Keeps RoPE, bounds far mass |
| `control_ctx4x` | RoPE, `theta = 10000` | 1024, batch 16 | Positive control: far distances seen in training, same tokens per step |

ALiBi adds `-m_k d` to the row's logits at distance `d`, after the `1/sqrt(dh)` scale, with fixed slopes
`m_k = 2^(-8k/h)` for heads `k = 1..h`, as in the original paper.

## Measurements

All on test data, with the same fixed windows for every arm and seed.

- **Per-row loss.** One pass over 2048 windows of length 2048, with loss kept per row and averaged into bins of 64
  rows. A row depends only on earlier columns, so row `r` here is the same computation as row `r` of a shorter
  window, and this one pass gives every length. `L(N)` is the bin of rows `[N - 64, N)`.
- **Windowed loss `Lw(N)`.** The same pass and targets, with every layer's attention limited to distances
  below `W`, the arm's training context. `Lw` needs no training and is a floor on what a model should get.
- **Excess `E(N) = L(N) - Lw(N)`,** paired on identical targets. This replaces E7's `G(N)`.
- **Truncated control `T`,** as in E7, on the last bin only, to check against E7's number.
- **Far mass.** Per layer and row bin, on 256 windows: attention mass on distances of at least `W`.
- **Distance profile.** Per layer, on the last 64 rows: attention mass and mean logit relative to the row's top, in
  distance buckets with edges at 0, 16, 64, 256, 512, 1024, 2048.
- **Copy probe, corrected.** Input length is `D + 3`, so each probe runs at the length its distance needs, unlike E7's
  fixed 2048 bytes. `D` in {32, 64, 128, 200, 400, 800, 1600}, 2048 windows. Guessing gives about 1/62 per byte.
- **Cost:** step time and tokens per second.

## How results will be read, fixed now

**C1, positive control.** Passes if `control_ctx4x` has `L(1024) <= rope_1e4 L(256) + 0.03`. Long rows then work
when trained, so a bug in that path is not the cause. **If it fails, a bug is the leading explanation:** stop, and do
not interpret the other results.

**C2, reproduction.** Passes if `rope_1e4` has mean `E(512) >= 1.0`. If it fails, E7's failure did not reproduce, and
that discrepancy comes first.

**C3, mechanism.** Supported if `rope_1e4` has far mass of at least 0.5 at `N = 512`, averaged over layers. If not,
the failure is not mainly attention moving far, and the RoPE explanation in E7 is not supported, even if some arm survives.

**Survival, per arm.** An arm survives if at every `N` in {512, 1024, 2048}, mean `E(N) <= 0.02` and no seed has
`E(N) > 0.05`. For the control, this applies only to `N = 2048`.

**Short-context cost, per arm.** Reported as a cost if mean `L(256)` exceeds `rope_1e4` by more than 0.03.

**Long context useful, per arm.** Only if the arm survives: mean `E(N)` below -0.005 at some `N > W`, with a
paired 95% interval below zero. Three seeds and several `N` make this exploratory.

**E7's claim about RoPE.** Supported only if C1, C2 and C3 pass, `rope_1e4` fails, and at least one arm with
linear decay survives. `rope_5e5` surviving shows that `theta` is the setting responsible. `rope_5e5` failing while ALiBi
survives shows that raising `theta` alone is not enough.

**Next step, also fixed now.**

- An arm survives, with no short-context cost: pre-register a normalizer comparison on that arm, with fresh seeds.
- Long context is not useful in any arm, and the copy probe is near chance past `W`: this model size gains nothing
  from long context on this text, so any comparison must use a retrieval task or a larger model.
- No arm survives: do not compare normalizers. Train at length, using the control's setup.

## Amendment, 2026-11-04, before any real run

An independent review found these problems in the reading rules. The only data seen so far is a CPU validation on
synthetic text, whose numbers mean nothing. This replaces the corresponding rules above.

- **New C0, measurement check, per arm.** Passes if mean `Lw(2048) <= T + 0.03`. `T` has the same targets and
  never uses a long row, while `Lw` uses long rows with far columns removed. A long-row bug would raise both `L` and `Lw`,
  leaving `E` near zero, so `E` alone would read it as survival. An arm failing C0 is not read. **If `rope_1e4` fails C0, a
  bug is the leading explanation: stop.**
- **C1 now has identical targets and a narrower meaning.** Passes if `control_ctx4x` has `L(1024) <= rope_1e4 Lw(1024) +
  0.10`. The earlier form compared different bytes. The wider margin allows for the control's batch of 16 at an untuned
  rate. A pass means long rows can learn when trained. It does not rule out a bug, because training could absorb one; that
  is C0's job. A fail stops the reading, as before.
- **C3 now tests the stated mechanism directly.** The old threshold of 0.5 is about what uniform attention gives at `N =
  512`. New measure: on rows `[448, 512)`, the mean weight per token at distances `[256, 512)` divided by the mean weight
  per token at distances `[64, 256)`, averaged over layers. Supported if this ratio is at least 2, so far tokens outscore
  in-range distractors. This needs a second recorded pass, on windows of length 512. Far mass stays reported.
- **Survival also covers every bin.** In addition to the three lengths, the mean over seeds of `E` must be at most 0.05 in
  every bin past `W`, because aliasing need not rise steadily with length.
- **"Useful" is read at every `N` past `W`,** as the original text says. `T - L(2048)` is also reported. Stacked layers
  give `Lw` a reach of several `W`, so `E` understates the value of long context, and the conclusion "no gain" means no gain
  over windowed attention only.
- **An arm that survives only with a short-context cost** has no automatic next step. The report says so and the owner decides.
- **Limitation kept.** Running out of memory during a recorded pass can weight some windows twice in the attention
  statistics. Each row stays normalized, so this affects only the statistics, and only if it happens.

## Amendment 2, 2026-11-04, before any real run: a toy run comes first

The owner asked for the smallest test that is still somewhat reliable, as quick as possible, on one RTX 5090. The toy run
comes first, and the full design above stays available.

- **Toy settings.** Same model and data. 6000 steps with 500 warmup, seeds 3000 and 3001, 512 test windows, 128 recorded
  windows, no smoke stage, 4 workers on the one GPU. Arms: `rope_1e4`, `rope_5e5`, `alibi`, `control_ctx4x`.
  `rope_1e4_alibi` is left out.
- **What it can read.** C0, C1, C2 and C3, and whether an arm clearly breaks or clearly holds, because the effect being
  chased is about 3.8 bits per byte, and the same windows and targets pair every comparison.
- **What it cannot read.** Two seeds give a very wide interval, so "long context useful" is reported only, and small
  differences between arms mean nothing. Survival at the 0.02 limit is a screen, not a finding. Step times are not clean,
  because workers share the GPU.
- **Less training changes one reading.** These models stop at a third of E7's steps, so if C2 fails, the reading is that
  the failure may need more training. It is not read as E7's failure failing to reproduce.
- **Next step.** The toy run sets the direction only. A claim about a cause or a fix needs the full run. A normalizer
  comparison needs an arm that also survives the full run.

## What it cannot show

One dataset, one model size and 3 seeds. No arm is tuned. Survival at 8x training length says nothing about
longer ratios. ALiBi passing here says nothing about its quality relative to RoPE at training length, beyond the
short-context cost read above.
