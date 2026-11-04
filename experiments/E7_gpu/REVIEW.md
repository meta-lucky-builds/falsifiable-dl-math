# E7 independent review

Reviewed by reading only: `hypotheses/H7_gpu_attention_on_real_text.md`, `experiments/E7_gpu/e7.py`,
`analyze.py`, `run.sh`, `requirements.txt`, `src/mai/attention.py`, `tests/test_e7.py`. No code was run.
Line numbers refer to the files as read. Numbers marked "estimate" are hand calculations.

**Count: 0 block the run, 1 wrong result, 6 minor.**

No defect found stops the run. The maths, alignment, fairness and statistics held up on reading.

## Wrong result

### W1. "Float32" evaluation actually runs matrix products in TF32 (10-bit mantissa)

- **Where:** `e7.py:928` (`torch.backends.cuda.matmul.allow_tf32 = True`), against the claim in `e7.py:11-13` and
  H7 Amendment 1 (H7:88, "Validation and test run in float32").
- **Failure:** this flag lets cuBLAS round float32 matmul inputs to a 10-bit mantissa on Ampere and newer, which
  covers all three target GPUs (RTX 3090, RTX 5090, H200). It is on for the whole worker. So in validation and test,
  every float32 product is TF32: `q @ k`, `w @ v`, all Linear layers and the output head `F.linear`. Only sort,
  cumsum, softmax and LayerNorm are true float32. The CPU tests and the mini run cannot see this, because TF32 does
  not exist there. The reason given for float32 (bf16 "too coarse ... for 0.005 bits/byte") applies in part to TF32,
  and the report would state a method that did not run.
- **Size (estimate, unverified):** small. Rounding moves logits by roughly 5e-3, which costs about 1e-5 nats. That is
  over 100 times below the 0.005 threshold, and all arms share it. I expect no outcome to flip. The defect is that the
  registered statement is false, and the effect has not been measured.
- **Fix:** delete line 928, or set it to `False`. Line 929 (`cudnn`) has no effect here, since the model has no
  convolutions. Training may slow slightly, because `q @ k`, `w @ v` and the head are float32 there. The Linear layers
  are bf16 under autocast and are unaffected. The smoke projection will show the cost. If the owner wants to keep
  TF32 for training, set the flag to `False` on entry to `val_loss` and `evaluate`, and back to `True` after. The
  alternative is an amendment before the run saying "float32, with TF32 matmul". Editing `e7.py` changes its hash, so
  the smoke stage reruns on its own.

## Minor

### M1. Repeated `torch.compile` in one worker can reach dynamo's recompile limit, so the cost table can favour early arms

- **Where:** `e7.py:538`, with workers living across jobs (`e7.py:895-917`).
- **Failure:** a worker runs about 7 jobs in the full run (28 jobs, 4 GPUs), each compiling a new model. The no-grad
  check (`e7.py:539`) and the training step (`e7.py:605`) are separate grad modes, and arms differ in `readout`, so
  each job can add cache entries to the same code objects. If guards miss, a code object can pass the default limit (8,
  as I recall it). Dynamo then logs a warning and silently runs those frames eagerly, while the job still reports
  `compiled=True`. Results are unaffected, only speed. But jobs are ordered softmax first and `entmax15_scaled` last,
  so the registered cost report (H7:72) could overstate the cost of later arms. Smoke compiles only 4 jobs
  (`compile_seeds=[2000]`), so it cannot show this. Unverified: it depends on the torch version.
- **Fix:** call `torch._dynamo.reset()` just before `torch.compile` in `try_compile`. It also releases old graphs.

### M2. A resumed job reports `compiled=False` even though it trained compiled

- **Where:** `e7.py:592-596`, read by `analyze.py:150-154`.
- **Failure:** a job that crashes during evaluation resumes with `step == steps`, skips compile and records
  `compiled=False`. The report then warns "inconsistent compiled across jobs" although nothing differed. An owner who
  cannot debug may distrust a valid run. The smoke report always shows this warning for a different reason, since
  only seed 2000 compiles.
- **Fix:** store `compiled` in the checkpoint `state`, and restore it when no training steps remain.

### M3. The code hash ignores `src/mai/attention.py`, where `entmax15` lives

- **Where:** `e7.py:652`, `e7.py:1026`.
- **Failure:** after a change to `attention.py`, the smoke output counts as current and is not rerun, so the new code
  gets no GPU check. A resumed full run also mixes old and new `entmax15` results, and `analyze.py:150` does not warn.
- **Fix:** hash `e7.py` and `attention.py` together, in both places.

### M4. An idle orphan worker never notices the launcher has gone

- **Where:** `e7.py:898-902`. Only `Beat` checks the parent, and only while a job runs.
- **Failure:** the launcher dies (for example SIGHUP on a dropped SSH session without tmux) while a worker waits at
  the tuning barrier or at the end. That worker polls indefinitely. After a rerun it can claim a job, write an attempt,
  and kill itself at its first beat about 20 s later. The launcher clears attempts only at its own start
  (`e7.py:1045-1047`), so this burns one of the job's two attempts. A real crash on the same job then marks it failed
  and costs a further rerun. Training state is not harmed.
- **Fix:** in the idle branch, exit if `os.getppid() != ppid`.

### M5. Smoke never runs a full-size evaluation batch

- **Where:** `e7.py:84` (`eval_windows=8`), `e7.py:405`.
- **Failure:** at N = 2048 smoke uses at most 8 windows, against 64 in the full run, so peak memory is an eighth of
  the real figure. The real GPU OOM path through `batched` is never exercised. My estimate of the real peak for
  entmax at batch 64 is 10-12 GB (one block of z is 0.8 GB, about 11 such tensors live, sort indices add 1.6 GB). It
  should fit in 24 GB, and OOM halving exists, so this is a coverage gap and not an expected failure. It also makes the
  evaluation-time projection a straight extrapolation from small batches.
- **Fix:** set smoke `eval_windows=64`. That is one full batch per N, costs little, and improves the projection.

### M6. Verdicts are stated without qualification when seeds are missing

- **Where:** `analyze.py:85` (intersection of seeds), `analyze.py:99-103`, `analyze.py:237-242`.
- **Failure:** if a final job stays failed, the tests run on n = 2 or 3 and the report can print "Kill for transfer
  of E1.1 ... FIRED", with the missing job shown only as a separate warning. H7 Amendment 2 sets the outcomes for 4
  seeds, and its own text says a miss at low seed count may only mean too little data. The report is labelled partial
  by `run.sh`, but `report.txt` itself does not qualify the verdict.
- **Fix:** when `n < len(final_seeds)`, append "PROVISIONAL: n of 4 seeds" to the outcome and to the kill line.

## Pre-registration text (not code, no effect on the run)

- H7:100: "about 7 with 3" seeds. The strictest Holm level is 0.05/3 = 0.0167, which needs t = 5.3 at 2 df. A t of 7
  corresponds to 0.01. The "near 4" for 4 seeds is about 3.7. The argument for 4 seeds still holds.
- H7:114: "the same 30000 steps" is stale after Amendment 2 (20000).
- H7:84-85: "adds no new chance of a false positive to the family" holds for the composite swap claim, which needs P3.
  It does not hold if P3b is read as a claim of its own, because when P3 is truly non-null P3b is tested at a full
  0.05. Read P3b only inside the swap rule.

## Checked and believed correct

**GPU and compile**
- Evaluation and validation use the plain `model`, never `fwd`, with no autocast and under `no_grad`.
- `set_readout` acts on the plain model, the compiled object is dropped by then, and no dropout makes `eval()` moot.
- Checkpoints save `model.state_dict()`, so no `_orig_mod.` prefix appears; the optimizer holds the same parameters.
- Nested autocast-off: bf16 q, k, v are cast up before the core, and the dtype guard at `e7.py:157` catches failure.
- Autocast does not touch float64 RoPE tables; the weight cast cache is cleared on outer exit, before `backward`.
- `entmax15` has only static shapes under compile; `int(...)` at `e7.py:163` is gated-only, and gated is never compiled.
- Fused AdamW is legal here: parameters are on the device first, and `step` is restored to the device on load.
- GradScaler disabled: all calls are no-ops and `{}` round-trips; the float16 branch has correct unscale/skip/update.
- `torch.Generator(device="cuda")` with `randint` on `cuda:0` is valid, and the seed cannot collide across steps.
- `CUDA_VISIBLE_DEVICES`: empty is rejected, UUID lists pass through, each worker sees index 0.
- OOM recovery: handled outside the except block, sums are order-free, and batch size cannot change a result.

**Mathematics**
- RoPE: q and k rotate at their own absolute positions, including `cos[rows_from:]`; v is not rotated.
- Mask: `tril(a)` on a `(b-a, b)` block gives row `a+i` columns up to `a+i`, and `n = a+i+1` matches it exactly.
- Block and `rows_from` slicing, and the output concatenation order, are correct.
- SSMax and scaled beta broadcast correctly; initial beta equals softmax's at n = 256.
- Gate: cut at `n <= 256` per row, including a cut inside a block, in all layers; reuse of softmax L(256) and T is exact.
- `entmax15`: Peters support formula, Newton step sign, implicit gradient, denominator >= 1, x = 0 at the row max.
- Float32 support selection stays safe at 2048 columns: spurious support needs error above about a/rho, and rounding
  is far below that.
- L, T, targets: input `[e-N, e)`, targets `e-keep+1 .. e`; T sees `[pos-256, pos)`; bits = nats / ln 2 in double.
- Copy probe: the last 4 rows hold s3 to s6 and predict s4 to s7; D fits; the same string per window at every D.
- Tied head, GPT-2 residual init, cosine and warmup schedule, 2.7M parameter count: all correct.

**Fairness**
- Initial weights: built on CPU after `manual_seed`, normalizer parameters use no RNG, so they match by seed and device.
- Batch stream is a pure function of (seed, step); no other randomness exists in training.
- Weight decay is the same on shared parameters; normalizer parameters, LayerNorm and bias are excluded.
- Windows and probes are fixed, and identical across jobs by the hash; precision is the same for every arm.
- Selection reads only `val_loss_final`; tuning jobs never call `evaluate`; final seeds differ from the tuning seed.

**Statistics, against the registered text**
- `ttest_rel(x, y, alternative="less")` with x the claimed-lower arm; gain = b - a is positive when the claim holds.
- P1, P2, P3 pairs and per-seed mean over {1024, 2048} are as registered.
- Holm step-down is correct, with a monotone adjustment and NaN treated as 1.
- P3b is gated on the Holm result for P3, tested at 0.05, and the swap rule is P3 and (small rise or P3b).
- Rise sign is correct; three outcomes use "at least 0.005"; the kill fires on P1 Not supported.
- G rule looks only at N > 256; an edge of the grid is reported; zero-variance differences give "Not supported".

**Rented machine**
- `run.sh` is safe with `set -u` and an empty array, and handles a relative `--out`; files in the folder are LF only.
- scipy is checked and installed before anything runs; `analyze.py` runs in smoke, so an old scipy fails early.
- Smoke directory: removed on a config or `e7.py` change, reused otherwise; a rerun of finished smoke costs about a
  data load. Full and smoke have separate directories and locks; children do not inherit the lock.
- Heartbeat: `beat` runs per batch and per step. I see no path that holds a claim for 1800 s without a beat, including
  compile and the N = 2048 entmax batch (estimate: seconds to tens of seconds each). A long container pause could kill
  healthy workers, at the price of one retry from the checkpoint.
- Disk: about 33 MB per live checkpoint, at most 4 at a time, deleted after each job; the 3 GiB check is enough.
- Resume: atomic writes, the final checkpoint protects evaluation, partial evaluation results are kept per metric.
- Queue: a claim is atomic, pid reuse is guarded by start time, zombies count as dead, attempts reset on rerun.

## Could not verify without running

1. Whether `torch.compile` traces this model fully on the owner's torch. Slicing `self.blocks[:-1]` (`e7.py:236`)
   builds a new ModuleList and may cause a graph break. That would cost speed only. Smoke covers crashes, not speed.
2. Whether compiled gradients match plain ones. `try_compile` compares only a no-grad loss, and training uses a
   separate grad-mode graph that nothing compares. I have no reason to expect a mismatch. A cheap guard: in smoke,
   compare the gradient norm of compiled and plain models on one batch.
3. Whether M1's recompile limit is actually reached, and its current default value.
4. Real peak memory at N = 2048 (M5), including internal GPU sort buffers, and real evaluation time.
5. Whether a real allocation failure always matches `is_oom`. A message such as `CUBLAS_STATUS_ALLOC_FAILED` would
   not match and would count as a crash. Unlikely at the estimated peak.
6. The size of W1's effect on L, which is my estimate only.
7. Whether future torch versions still honour `allow_tf32` (a deprecation warning is possible), and whether the
   installed build supports the GPU. Preflight catches the second.
8. Whether `mattmahoney.net` is reachable from the machine. The failure message gives manual steps.
