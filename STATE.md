# Mission state

Read this first when resuming. Update it and push at every checkpoint, so a
fresh run can continue from the repo alone.

## Mission

Find mathematical changes to standard deep learning, by reading, hypothesizing,
implementing and testing. Long-running and autonomous: the owner is away and
does not want to be asked about small decisions.

## Standing rules

- Use the how-to-think skill for hypotheses, planning, analysis and design.
- Models, all at medium effort: Opus 5.5 for thinking and implementation,
  Sonnet 5.5 for execution of finished code and web search, Haiku 5.5 for simple
  retrieval.
- Never run tests or experiments locally, even CPU ones. Use the Colab MCP.
- Never put a GitHub token in Colab. Send code by writing cells.
- Work on branch `research/math-foundations`.
- Spend usage carefully: narrow agent briefs, cheapest suitable model, no
  repeated searches. Do not lower the rigor rules in `README.md` to save usage.
- Protect context: keep long outputs in files, and have agents return short
  summaries. Keep this file current, so nothing depends on conversation memory.
- Avoid local optima. When a kill criterion fires, record it and move on, without
  patching the hypothesis to keep it alive. If two experiments in a row on one line
  of work yield nothing, switch to a different line or a different kind of
  mathematics. Refine or add hypotheses as evidence arrives, and pre-register each
  one before it runs. Keep a mix of thread types in `LEDGER.md`, so the work never
  depends on one idea.

## Open question for the owner

The brief named "meta-lucky-builds/ami" and also "the intr repo". `ami` does not
exist, so this work is in `mai`. Move it if that was wrong.

## Compute

Colab runtime seen 2026-11-04: CPU only, 2 cores, 12 GB RAM. Size experiments to
fit.

## Progress

- [x] Literature surveys: `notes/lit/optimization_geometry.md`,
      `notes/lit/layers_algebra.md`, `notes/lit/objectives_information.md`
- [x] H1 and H2 pre-registered in `hypotheses/`, tracked in `LEDGER.md`
- [x] E1.0 numeric check of the H1 formula: passed for Gaussian scores, broken by
      heavy tails. See `experiments/E1_0_rem_numeric/RESULTS.md`
- [ ] Prior-art check for H1 and H2: `notes/lit/prior_art_H1_H2.md`
- [ ] Implementations of E1.1 and E2.1
- [ ] Colab experiments
- [ ] Analysis

## Next action

Read the prior-art verdicts. If `notes/lit/prior_art_H1_H2.md` is missing, redo that
check. Then implement E1.1, including the Calib-Attn arm, and E2.1 for whichever
hypotheses are still worth testing.
