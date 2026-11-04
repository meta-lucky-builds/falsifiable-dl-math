# H5: does sparse attention help during training or at inference?

Status: pre-registered 2026-11-04, after E1.1 and before any H5 run. This is a question
about mechanism, opened by E1.1, not a new formula. It is exploratory in that sense: E1.1
data suggested it, so it needs fresh seeds.

Priors: inference effect accounts for more than half of entmax15's gain, about 55%.
Learning-rate check changes the entmax15 over softmax conclusion, about 15%.

## Question

In E1.1, plain entmax15 beat softmax by about 0.08 at `N = 256`, at noise 0.6 and 0.9.
It also beat REM at equal needle mass. Two separate explanations predict this:

- **Training:** sparse attention gives better gradients to the scorer, so the learned
  scores are better. Any readout would then benefit.
- **Inference:** the scores are similar, and a sparse readout discards the distractors'
  values, which softmax averages in as noise.

These have different consequences. If inference dominates, a model could be trained
normally and given a sparse readout only for long contexts. If training dominates, the
readout could stay smooth and only the training loop would need to change.

## Experiment E5.1: train-by-readout swap

Train with `T` in {softmax, entmax15, rem}. These arms have no normalizer parameters, so
any trained model can be evaluated with any of them. Noise 0.6 and 0.9. Five fresh seeds,
disjoint from E1.1. Same model, steps and task as E1.1.

Evaluate each trained model at `N` = 8, 64 and 256 with readout `I` in {softmax, entmax15,
rem}, applied to both attention layers, plus softmax-trained with entmax15 on the cross
layer only. Log accuracy, needle mass, and the **retrieval rate**: how often the needle is
the top-scoring token in the cross row, per head and best head. Retrieval depends only on
scores, not the readout, so it directly measures score quality.

**Head refit control.** A swapped readout feeds later layers an input they did not train
on, which biases against the swap. So for each swap, also freeze everything through the
attention scores and refit the rest (attention value and output projections, the
feed-forward after cross-attention, its norms, and the head) for 300 steps at `N = 8`,
using the new readout, then evaluate. Report with and without the refit.

**Learning-rate check.** E1.1 selected the top of the grid. Train softmax and entmax15 at
`lr` = 0.006 and 0.01, noise 0.6 and 0.9, on one tuning seed, and read validation accuracy
at `N = 8` only. If either beats 0.003, the final runs for **every** arm use the best
rate per arm from {0.003, 0.006, 0.01}, selected on validation data at `N = 8`. All arms
get the same grid, so softmax and entmax15 are not the only ones checked: REM runs the same
check.

## Primary measure

At `N = 256`, with refit:

    inference share = (A[soft, ent] - A[soft, soft]) / (A[ent, ent] - A[soft, soft])

where `A[T, I]` is accuracy trained with `T` and read out with `I`. It is computed
per seed and reported with a paired interval. Also report the same share at `N = 64`, and
the difference in retrieval rate, `T = ent` minus `T = soft`.

## Predictions and how they will be read

Fixed now:

- **Inference dominates** if the share's interval lies above 0.5 and retrieval rates differ
  by under 0.01.
- **Training dominates** if the share's interval lies below 0.5 and `T = ent` has higher
  retrieval, with a paired interval excluding zero.
- **Mixed or unresolved** otherwise, and it will be reported that way.

Kill for the premise: if `A[ent, ent] - A[soft, soft]` at `N = 256` has an interval that
includes zero on these fresh seeds at both noise levels, E1.1's entmax gain did not
replicate, and the share is not interpreted.

## Amendment, 2026-11-04, before any E5.1 run

Found while implementing, with no data seen:

- **My design had an error.** Retrieval does not depend only on scores under a both-layer
  swap, because swapping the self-attention normalizer changes the memory, and so the cross
  scores. A both-layer swap therefore mixes readout and scores. **The primary measure now
  uses the cross-only swap for `A[soft, ent]`**, because that alone holds cross scores
  fixed. The both-layer share is reported and not read. Retrieval in the reading is
  measured on the scores as trained, and retrieval as read is logged too.
- The learning-rate rule is now unconditional: every arm takes its best rate per noise from
  the grid. The pre-registered gate is reported but not applied. This differs only if REM
  alone prefers a higher rate.
- Details the text left open: refit learning rate fixed at 1e-3 for every case. The premise
  uses refit accuracy. "Under 0.01" uses the mean paired retrieval difference. If any seed's
  denominator is below 0.02, the share is a ratio of means with a bootstrap over seeds, and
  the output says which was used.

## Limitations stated in advance

This is a small synthetic task, so an explanation here would not automatically apply to
language models. Any follow-up that aims at language models needs its own pre-registration.
