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
- Never run tests or experiments locally, even CPU ones. Use the Colab MCP, or start GitHub Actions
  (`.github/workflows/tests.yml`) by hand. On 2026-11-04 GitHub refused to start Actions jobs because of a
  billing problem on the account, so check that first.
- Never put a GitHub token in Colab. Send code by writing cells.
- Work on branch `research/math-foundations`. `main` holds the same history and is only ever
  fast-forwarded from it.
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

The owner renamed this repo from `mai` to `meta-lucky-builds/falsifiable-dl-math` on 2026-11-04, which settles
where the work lives. The Python package stays `src/mai`, and `/content/mai` paths are from before the rename.

## Compute

Colab: CPU only, 2 cores, 12 GB RAM. **The runtime is shared with the owner's other work
under `/content`.** Do not touch those files or processes. The notebook can be reset or its kernel restarted by them, so background
processes and files in `/content/mai` are what survive: find state with `ps` and `ls`,
never kernel variables. Wall-clock timings are not clean. If the Colab tools disappear,
reload them with ToolSearch and call `open_colab_browser_connection`.

## Progress

- [x] Literature surveys in `notes/lit/`, prior-art check in `prior_art_H1_H2.md`
- [x] H1, REM attention temperature: **killed**. `experiments/E1_1_recall/RESULTS.md`
- [x] H2, noise-edge gate for Muon: retired on prior art, no experiment
- [x] H3, sign updates on forward gradients: no estimator advantage.
      `experiments/E3_0_forward_gradient_snr/RESULTS.md`
- [x] H4, finite-epsilon Adam law: stopped, kill fired on `omega`. Do not run E4.1.
      `experiments/E4_0_epsilon_law/RESULTS.md`
- [ ] H5, entmax on recall, training or inference: pre-registered in `hypotheses/`.
      Implementation: `experiments/E5_1_swap/`

- [x] H7, attention normalizers on real text: run on GPUs, **did not test its question**. Every arm
      breaks at 2x training length. `experiments/E7_gpu/RESULTS.md`

## Current priority, set by the owner: waiting on the owner

The H8 toy run is done and recorded in `experiments/E8_long_context/RESULTS.md`: no sign of a bug, ALiBi holds to
8x training length, RoPE fails, and raising `theta` makes it worse. It sets a direction only. The next step, for the
owner to decide, is the full E8 run, which needs GPU time. Do not start other threads until the owner says so. For
anything on a rented machine, give one command for a Jupyter terminal, print results on screen, and never say the
instance can be destroyed before the results are copied.

## Next action, once the owner lifts the H7 limit

E5.1 code is in `experiments/E5_1_swap/`. It was launched on Colab, with output
`/content/mai/e51_results.jsonl` and log `/content/mai/e51_run.log`, but its verdict was never read.
E6.0 was launched too, with log `/content/mai/e60.log`: Part A passed (about 1e-13), the rest is
unread. Colab files may be gone. If neither E5.1 file exists, run tests and a smoke run, then launch
it. Rerunning the same command resumes it. When it finishes, run `analyze.py`, and record the
reading it prints, as pre-registered.

No hypothesis so far has produced a win. H5 is one mechanism question, in the same
attention line as H1. After it, per the rule above, the next thread must be a different
kind of mathematics: take one from the queue in `LEDGER.md` that is outside attention
and optimizer spectra, or run a new literature pass on under-searched areas (optimal
transport losses, information bottleneck, PAC-Bayes as a loss).
