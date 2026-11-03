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
- [x] Prior-art check: `notes/lit/prior_art_H1_H2.md`. H1 goes ahead as incremental
      work. H2 retired without an experiment, so E2.1 will not run.
- [x] H3, sign updates on forward gradients: no estimator advantage. See
      `experiments/E3_0_forward_gradient_snr/RESULTS.md`
- [x] E1.1 implementation: `src/mai/attention.py`, `tests/test_attention.py`,
      `experiments/E1_1_recall/`. 19/19 tests passed on Colab, smoke run clean.
- [ ] E1.1 full run on Colab: launched 2026-11-04 as a background process, with
      output `/content/mai/e11_results.jsonl`, log `/content/mai/e11_run.log`, flag
      `--tune-seeds 1`, about 168 runs, projected about 5 hours. Rerunning the same
      command resumes it, unless the runtime was reclaimed.
- [ ] Analysis, then E1.2 score audit
- [ ] H4, finite-epsilon Adam schedule law: derivation in progress, see
      `hypotheses/H4_adam_epsilon_schedule_law.md` when it exists

## Next action

While E1.1 runs, check the log at most hourly and print a progress count. At each
check, once enough cells are done, save a compact CSV (arm, seed, noise, n, acc,
needle_mass) into `experiments/E1_1_recall/` and push, because `/content` is lost if
the runtime is reclaimed. Do not read full JSONL files into context: they are large.
When it finishes, run `analyze.py`, commit its output, and apply the kill rule as
pre-registered. Do not run other Colab jobs that would disturb its timing.
