# mai

Testing mathematical changes to standard deep learning. Each idea is derived by hand, checked
numerically, pre-registered with a kill criterion, and then compared against tuned baselines over
paired seeds. Failed ideas stay on record, with the reason they failed.

**Where it stands (2026-11-04).** Eight hypotheses opened, one of them waiting to run. **No new method from this project has beaten
a tuned baseline so far.** The useful output is verified mathematics, some clear negative results, and
one strong lead on an existing method that has not yet been shown on real text.

## Top findings

Ordered roughly by how much each one changes what to try next. Each states how strong its evidence is.

### 1. A closed-form attention temperature is correct, and still loses

Treating a softmax row as a Random Energy Model with one relevant token gives the smallest inverse
temperature that puts a target mass on that token:

```
beta* = (mu - sqrt((mu^2 - 2 sigma^2 L)_+)) / sigma^2,    L = log n + kappa
```

- **The formula holds for Gaussian scores.** Above threshold, at `n >= 2^10`, needle mass missed its
  0.9 target by at most 0.013.
- **Heavy tails break it.** At `n = 2^18`, needle mass falls to 0.000 for Laplace scores at signal ratio
  1.05, and for Student-t (3 df) at ratio 2.0.
- **As attention, it lost to the best baseline at every noise level.** On recall trained at `N = 8` and
  tested to `N = 256`, the gap was 0.034 to 0.065 in accuracy at noise 0.6 and 0.9, and 0.006 to 0.015
  at noise 0, with all paired intervals excluding zero. It does tie learned SSMax at `N = 256` with no
  learned parameters.
- **Holding needle mass up is not enough.** At noise 0.6 and `N = 256`, entmax-1.5 and REM have nearly
  equal mass (0.60 and 0.59) but accuracies of 0.814 and 0.749.

Evidence: [E1.0](experiments/E1_0_rem_numeric/RESULTS.md), [E1.1](experiments/E1_1_recall/RESULTS.md),
[H1](hypotheses/H1_rem_attention_temperature.md). Strength: high for the kill. Synthetic task, 71k
parameters, 5 seeds.

### 2. Plain entmax-1.5 was the strongest arm on synthetic recall, which is the main open lead

At `N = 256` it beat softmax by **+0.077** [+0.049, +0.105] at noise 0.6 and **+0.078** [+0.053, +0.102]
at noise 0.9, with no length term. Adding learned length scaling did not measurably improve it. The
margin over SSMax is much thinner, about +0.035, with intervals that only just exclude zero and no
multiplicity correction.

This came out of a comparison aimed at a different question, so it needs fresh seeds. That is what
[H5](hypotheses/H5_sparsity_train_or_inference.md) is for, and its result is not yet recorded. On real
text the question is still open (finding 4). At the length where those models work, entmax-1.5 and
softmax are close: -0.0048 bits per byte, about [-0.013, +0.003].

### 3. Swapping a sparse readout into a softmax-trained model is costly

On byte-level enwik8 with a 2.7M-parameter model, reading a softmax-trained model with entmax-1.5 raises
loss at `N = 256` from **1.394 to 1.653 bits per byte**. The cost is 0.243 to 0.267 and appears in all 4
seeds, at a length where the model works, so finding 4 does not explain it. The swap had no recalibration,
so this rules out the naive version only. Separately, training with entmax-1.5 ran about 29% slower per
step than softmax (21.7 ms against 16.9 ms).

Evidence: [E7 results](experiments/E7_gpu/RESULTS.md), [per-run data](experiments/E7_gpu/results_compact_run2.csv).
Strength: high for this model and task, descriptive rather than a primary test.

### 4. Every normalizer broke at twice the training length, so that experiment missed its question

| Arm | L(256) | L(512) | L(1024) | L(2048) |
|---|---|---|---|---|
| softmax | 1.394 | 5.230 | 5.179 | 5.366 |
| ssmax | 1.397 | 5.545 | 6.075 | 6.469 |
| entmax15 | 1.389 | 5.244 | 5.412 | 5.875 |
| entmax15_scaled | 1.390 | 5.261 | 5.425 | 5.949 |

Bits per byte, mean of 4 seeds. The pre-registered outcomes were P1 not supported, P2 inconclusive and P3
not supported, but they compare arms inside a failure regime, so **they say nothing about which normalizer
is better at long context.** The same models score about 1.39 when given only the last 256 bytes, so the
truncated-context control is what exposed this: extra context made predictions worse.

The likely cause is a design choice made here, RoPE with `theta = 10000` at training length 256. **This is
not confirmed**, and a code bug is not ruled out. [H8](hypotheses/H8_long_context_failure.md) is built to tell
these apart, and has not run yet.

### 5. Adam's epsilon has an exact recursion, and it is not an effective-LR schedule at equilibrium

Finite epsilon does not break closure of the scale-free AdamW dynamics. It adds one schedule input,
`E_t = eps eta_t / a_t`, and gives a crossover scale:

```
D_i        = sqrt(v_i) + E_t / Psi_t
p_i        = m_i / D_i
N_t        = ||u_t - Psi_t p_t||
Psi_{t+1}  = B~_t Psi_t / N_t
r*         = G / (eps sqrt(d))
```

(`m` and `v` here are the scale-free, bias-corrected moments; [H4](hypotheses/H4_adam_epsilon_schedule_law.md)
has the full notation.) The recursion extends the epsilon-free schedule law of 2609.09116, which lists
finite epsilon as open.

- The exact identities hold to about 1e-14, including against real `torch.optim.AdamW` on a BatchNorm MLP.
- On an isotropic noise walk, equilibrium rotation stayed at 3.24e-3 while `r` moved from 2.22 to 0.20.
  This refutes the stronger conjecture that epsilon acts as an effective-LR schedule there.
- The relaxation formula missed by 18.8% against a 15% limit, so the pre-registered kill fired and the
  real-network test was not run.

Evidence: [E4.0](experiments/E4_0_epsilon_law/RESULTS.md). Strength: high for the algebra, toy model only
for the rest. It gives no route to faster training. Five searches did not find the crossover law in the
literature, which is weak evidence that it is new.

### 6. Sign updates give forward gradients no estimator advantage

For a fixed gradient, signal-to-noise per unit gradient is 1 for raw updates, `sqrt(2/pi) = 0.80` when
tangent-normalized, and `2/pi = 0.64` for sign updates. That means about `pi^2/4 = 2.47` times as many
samples for the same accuracy. Closed form and Monte Carlo agree across dense, sparse and heavy-tailed
gradients. Training effects such as step-size stability are untested.

Evidence: [E3.0](experiments/E3_0_forward_gradient_snr/RESULTS.md).

### 7. Checking prior art saved an experiment

The noise-edge gate for Muon was retired before any compute was spent. BulkBoost (2610.07497) already
has the half-batch noise estimator and the Marchenko-Pastur edge, and reports that *boosting* the
below-edge modes helps; those modes are 93.7-97.1% of the spectrum. Spectrally Targeted Muon (2610.10965)
reports that top-only orthogonalization falls short. The search log, including what was not searched, is
in [the prior-art note](notes/lit/prior_art_H1_H2.md).

## All hypotheses

| ID | Question | Status | Record |
|---|---|---|---|
| H1 | Does a REM closed form set attention temperature? | **Killed** | [hypothesis](hypotheses/H1_rem_attention_temperature.md), [E1.0](experiments/E1_0_rem_numeric/RESULTS.md), [E1.1](experiments/E1_1_recall/RESULTS.md) |
| H2 | Does gating Muon at a noise edge help at small batch? | **Retired on prior art**, not tested | [hypothesis](hypotheses/H2_noise_edge_spectral_gate.md), [prior art](notes/lit/prior_art_H1_H2.md) |
| H3 | Do sign updates improve forward-gradient estimates? | **No estimator advantage**, training not tested | [E3.0](experiments/E3_0_forward_gradient_snr/RESULTS.md) |
| H4 | Does Adam's epsilon act as an effective-LR schedule? | **Stopped**, kill fired | [hypothesis](hypotheses/H4_adam_epsilon_schedule_law.md), [E4.0](experiments/E4_0_epsilon_law/RESULTS.md) |
| H5 | Does entmax's gain come from training or from readout? | **Launched, result not recorded** | [hypothesis](hypotheses/H5_sparsity_train_or_inference.md), [E5.1](experiments/E5_1_swap/RESULTS.md) |
| H6 | In diagonal nets, is decay-then-off the same as a small init? | **Launched, partly read** | [hypothesis](hypotheses/H6_decay_budget_vs_small_init.md), [E6.0](experiments/E6_0_decay_budget/RESULTS.md) |
| H7 | Do attention normalizers help long contexts on real text? | **Ran, did not test its question** | [hypothesis](hypotheses/H7_gpu_attention_on_real_text.md), [E7](experiments/E7_gpu/RESULTS.md) |
| H8 | Why do models break past training length: position scheme, or a bug? | **Ready to run on GPUs** | [hypothesis](hypotheses/H8_long_context_failure.md), [code](experiments/E8_long_context/e8.py) |

[LEDGER.md](LEDGER.md) has the evidence for each, plus the queue of ideas not yet pre-registered.

## How the work is done

1. **Falsifiable first.** Each hypothesis states its mechanism, prediction, baseline and kill criterion
   before any run. Pre-registrations are changed only by dated amendments, made before the data they affect.
2. **Derive before training.** A formula gets a derivation or numeric identity check first.
3. **Equal tuning.** Every arm gets the same search. A win against an untuned baseline does not count.
4. **Paired seeds and spread.** There are no single-seed claims. H7 applies a Holm correction to its primary tests.
5. **Negative results stay.** A killed hypothesis keeps its entry and its reason.
6. **Prior art is searched and logged**, including what was not searched and how deeply each paper was read.

### Mistakes found and kept on the record

- **E7 positional design.** RoPE at `theta = 10000` was fixed despite a warning in our own survey, and that
  is the likely cause of finding 4.
- **E7 copy probe.** Every probe input was 2048 bytes long, so even short distances ran in the failing regime.
- **E7 first run.** It was lost when its instance was destroyed before the results were copied. Results now
  print on screen, and that run is not used anywhere.
- **E1.2 kill criterion.** It fires by construction, because SSMax's `beta` is exactly `c log n`, so it
  carries little information.
- **E1.1 Calib arm.** It failed to bootstrap, so it does not test the question it was added for.
- **E1.1 analysis script.** It prints "consistent" for a gain prediction where the values are actually a tie.
  The written verdict overrides it.
- **E5.1 design.** A both-layer swap changes the cross-attention scores too. This was caught before any
  run, and the primary measure was amended to the cross-only swap.

## Repository map

| Path | Contents |
|---|---|
| [`LEDGER.md`](LEDGER.md) | Every hypothesis, its status and its evidence, plus the idea queue |
| [`hypotheses/`](hypotheses/) | Pre-registrations, with derivations and amendments |
| [`experiments/`](experiments/) | One folder per experiment: code, recorded output and `RESULTS.md` |
| [`src/mai/attention.py`](src/mai/attention.py) | Attention normalizers: softmax, SSMax, entmax-1.5, scaled entmax, REM, Calib |
| [`tests/`](tests/) | Tests for the normalizers, the readout swap and the E7 pipeline |
| [`notes/lit/`](notes/lit/) | Literature surveys with paper IDs and reading depth |
| [`STATE.md`](STATE.md) | Working state for resuming the research |

## Reproducing

Requires Python 3.9+, `torch >= 2.3`, `numpy` and `scipy`. The tests have their own runners, so pytest is
not needed.

```bash
python tests/test_attention.py
```

```bash
python tests/test_swap.py
```

```bash
python tests/test_e7.py
```

The numeric checks are standalone scripts. For example:

```bash
python experiments/E1_0_rem_numeric/check.py
```

The GPU experiments check the machine, run a smoke stage and then the full run. They resume if rerun, and
print their results on screen when they finish.

```bash
bash experiments/E7_gpu/run.sh
```

```bash
bash experiments/E8_long_context/run.sh
```

GitHub Actions runs every test, plus a small E8 run, on each push.

The recorded runs used Colab CPU, except E7, which used 4x RTX 5090 on a rented machine. Wall-clock timings
from the shared Colab runtime are not clean.

## Limits of this evidence

- E1 to E6 are small synthetic or toy problems, so their rankings may not transfer to language models.
- E7's useful results come only from `N = 256`, one positional scheme and 4 seeds. All its arms selected the
  top of the learning-rate grid, as did E1.1, so tuning may be incomplete.
- Some paper numbers come from AI-generated summaries rather than the PDFs. The notes mark each one, and they
  should be rechecked before being cited.
- E7 checkpoints were deleted after each job, so those models cannot be examined further.

## What is next

1. **Find out why long contexts fail.** Run H8. Normalizers can only be compared on real text once a setup survives
   past training length.
2. **Read out H5 and H6**, rerunning them if their files are gone.
3. **Move to different mathematics.** Candidates from the surveys are 1-D Wasserstein logit losses and how
   contrastive temperature should scale with the number of negatives.

## License

[MIT](LICENSE)
