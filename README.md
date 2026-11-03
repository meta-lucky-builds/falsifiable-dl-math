# mai

Applied-mathematics research for AI/ML/DL: looking for mathematical changes to
what deep learning does by default (new formulas, different concepts, new
techniques), tested honestly at small scale.

## Rules of the work

1. **Hypotheses are falsifiable.** Each one states its mechanism, a prediction,
   a baseline and a kill criterion *before* any experiment runs.
2. **Derive first.** A claim about a formula gets a derivation or a numerical
   check of the identity before it gets a training curve.
3. **Baselines are tuned equally.** A comparison where only the new method got a
   hyperparameter search is not evidence.
4. **Multiple seeds, reported spread.** No single-seed claims.
5. **Negative results are kept.** A killed hypothesis stays in the ledger with
   the reason it died.
6. **Prior art is checked.** "New" means searched for, with the search recorded.
7. **All execution happens in the cloud (Colab).** Nothing is run locally.

## Layout

- `LEDGER.md`: every hypothesis with its status and evidence.
- `notes/lit/`: literature surveys, with paper IDs.
- `hypotheses/`: pre-registered hypothesis write-ups.
- `src/mai/`: implementations.
- `experiments/`: experiment scripts and recorded results.
