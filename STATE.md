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

## Open question for the owner

The brief named "meta-lucky-builds/ami" and also "the intr repo". `ami` does not
exist, so this work is in `mai`. Move it if that was wrong.

## Compute

Colab runtime seen 2026-11-04: CPU only, 2 cores, 12 GB RAM. Size experiments to
fit.

## Progress

- [ ] Literature surveys: `notes/lit/optimization_geometry.md`,
      `notes/lit/layers_algebra.md`, `notes/lit/objectives_information.md`
- [ ] Hypotheses in `LEDGER.md`
- [ ] Implementations
- [ ] Colab experiments
- [ ] Analysis

## Next action

Finish the surveys. If a survey file is missing, redo only that thread.
