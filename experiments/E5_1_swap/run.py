"""E5.1: train-by-readout swap. Pre-registered in hypotheses/H5_sparsity_train_or_inference.md.

Model, task, steps and schedule are E1.1's, imported from experiments/E1_1_recall/run.py.
Train with T in {softmax, entmax15, rem}. Each trained model is then read out at N = 8, 64
and 256 with I in {softmax, entmax15, rem} on both attention layers, plus entmax15 on the
cross layer only for softmax-trained models, each with and without a head refit.

CHOICES THE PRE-REGISTRATION LEAVES OPEN OR GETS WRONG (all fixed before any run):
- Retrieval. The pre-registration says retrieval depends only on scores, not the readout.
  That holds for the cross normalizer only. Swapping the SELF normalizer changes the memory,
  and therefore the cross scores. So the retrieval used for the reading is measured on the
  model as trained (T on both layers), whatever readout it currently carries, which makes it
  independent of the readout. A side effect: the both-layer swap, and so A[soft, ent], moves
  scores as well as the readout, and a refit cannot undo that because the scores are frozen.
  The cross-only readout is the arm without this effect. Each evaluation row also logs
  retrieval as read (`ret_best`), so analyze.py can show how far the scores moved.
- Refit learning rate: fixed at 1e-3 with E1.1's schedule shape and clip, for every (T, I).
  It is not tuned, and it does not follow T's selected rate, so T cannot affect the refit.
- LR selection: each arm's best rate per noise, unconditionally. The pre-registered gate
  ("if softmax or entmax15 beats 0.003") is printed but not applied. This differs from the
  gate only if REM alone prefers a higher rate at a noise level.
- Best head is per example (the needle is top in at least one head), matching E1.1's
  needle_mass_best_head.

SEEDS. Final 200-204 and tuning 300, disjoint from E1.1's 0 and 100-104. Data streams are
keyed by (kind, seed, noise, n), never by T or I, so test data for a seed is identical across
T and I, and each refit in a seed sees the same training batches. Kinds "train", "refit",
"val" and "test" are separate streams.

BUDGET (estimate, not measured; --smoke prints a measured projection). Assumes 70 ms per
training or refit step, and per forward batch 1.5 s at N=256, 0.12 s at N=64, 0.02 s at N=8.
    LR check: 3 arms x 2 noise x 3 lr x 1 seed = 18 runs x 176 s               =  3170 core-s
    training: 30 runs x 175 s                                                  =  5250 core-s
    refits:   10 per (seed, noise) x 10 (seed, noise) x 300 steps x 70 ms      =  2100 core-s
    eval:     23 passes (20 readout rows, 3 retrieval) x 10 x 16 batches x 1.64 s = 6030 core-s
Total about 16500 core-s, so 2.3 hours on two workers. If forward at N=256 is 3 s on the
shared runtime, the total is about 3.1 hours. Both fit the 4 hour budget. If the smoke
projection does not, lower --test-batches: steps are pre-registered.

Usage: python run.py --smoke
       nohup python run.py --out results.jsonl > run.log 2>&1 &     (rerun to resume)
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# Loaded by path: this file is also called run.py, so `import run` would find this file.
_spec = importlib.util.spec_from_file_location("e1_1_run", Path(__file__).resolve().parents[1] / "E1_1_recall" / "run.py")
e11 = importlib.util.module_from_spec(_spec)
sys.modules["e1_1_run"] = e11
_spec.loader.exec_module(e11)
RecallModel, make_batch, data_gen, stable_seed, clean = (e11.RecallModel, e11.make_batch, e11.data_gen,
                                                         e11.stable_seed, e11.clean)

ARMS = ("softmax", "entmax15", "rem")
# readout name -> (self normalizer, cross normalizer)
READOUTS = {"softmax": ("softmax", "softmax"), "entmax15": ("entmax15", "entmax15"), "rem": ("rem", "rem"),
            "entmax15_cross": ("softmax", "entmax15")}  # softmax-trained models only
ROW_KEYS = ("acc", "loss", "needle_mass", "needle_mass_best_head", "ret_best", "forward_ms_per_batch")

# Head refit. Trained: only what sits strictly after the cross-attention scores.
#   ca.v, ca.out      the cross value and output projections
#   ff2, norms.4      the feed-forward after cross-attention and its input norm
#   norms.5, head     the final norm and the classifier
# Frozen: everything else, because all of it feeds the cross scores z = ca.q(q) . ca.k(mem).
#   key_in, query_type, norms.2, ca.q    build the query side of the score
#   key_in, val_in, sa (q, k, v and out), ff1, norms.0, norms.1, norms.3, ca.k    build the key side
# sa.v and sa.out are value and output projections too, but they feed the memory and hence
# ca.k, so training them would change the scores. Only the cross layer's are refit.
REFIT_PREFIXES = ("ca.v.", "ca.out.", "ff2.", "norms.4.", "norms.5.", "head.")

FULL = dict(e11.FULL, factors=[1, 8, 32], noises=[0.6, 0.9], lrs=[3e-3, 6e-3, 1e-2], tune_seeds=[300],
            final_seeds=[200, 201, 202, 203, 204], val_batches=32, test_batches=16,
            refit_steps=300, refit_warmup=30, refit_lr=1e-3)
SMOKE = dict(FULL, steps=30, warmup=5, refit_steps=10, refit_warmup=3, noises=[0.6], lrs=[3e-3, 6e-3],
             final_seeds=[200], val_batches=1, test_batches=1)


def set_normalizer(m, name, kappa):
    """Put attention module m in the state a fresh module constructed with `name` would have.
    Only rem carries state, a kappa buffer that exists only if constructed as rem."""
    if name not in ARMS:
        raise ValueError(f"{name!r} has normalizer parameters or is unknown; it cannot be swapped")
    if name == "rem" and "kappa" not in m._buffers:
        m.register_buffer("kappa", torch.full((m.h, 1, 1), float(kappa)))
    elif name != "rem" and "kappa" in m._buffers:
        del m._buffers["kappa"]
    m.normalizer = name


def view(model, readout, cfg):
    """A copy of the model carrying the readout. The original is never changed."""
    m = copy.deepcopy(model)
    sa_name, ca_name = READOUTS[readout]
    set_normalizer(m.sa, sa_name, cfg["kappa"])
    set_normalizer(m.ca, ca_name, cfg["kappa"])
    return m


def readouts_for(arm):
    return ARMS + ("entmax15_cross",) if arm == "softmax" else ARMS


def fit(model, params, cfg, g, noise, lr, steps, warmup):
    """E1.1's training loop: same batch, schedule, clip and skipping of non-finite gradients."""
    opt = torch.optim.Adam(params, lr=lr)
    step_times, grad_norms, tail_loss, nonfinite = [], [], [], 0
    for t in range(steps):
        t0 = time.perf_counter()
        for group in opt.param_groups:
            group["lr"] = lr * min(1.0, (t + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * t / steps))
        keys, vals, q, y, _ = make_batch(g, cfg["batch"], cfg["train_n"], noise, cfg)
        loss = F.cross_entropy(model(keys, vals, q), y)
        opt.zero_grad()
        loss.backward()
        norm = float(nn.utils.clip_grad_norm_(params, cfg["clip"]))
        if math.isfinite(norm):
            opt.step()
            grad_norms.append(norm)
        else:
            nonfinite += 1
        step_times.append(time.perf_counter() - t0)
        if t >= steps - 100:
            tail_loss.append(loss.item())
    step_times.sort()
    grad_norms.sort()
    return dict(step_ms_median=clean(1e3 * step_times[len(step_times) // 2]),
                grad_norm_median=clean(grad_norms[len(grad_norms) // 2]) if grad_norms else None,
                nonfinite_steps=nonfinite, train_loss_tail=clean(sum(tail_loss) / max(len(tail_loss), 1)))


def train_model(cfg, arm, seed, lr, noise):
    torch.manual_seed(stable_seed("init", seed))  # same init across arms, as in E1.1
    model = RecallModel(cfg, arm)
    g = data_gen("train", seed, noise, cfg["train_n"], cfg)
    return model, fit(model, list(model.parameters()), cfg, g, noise, lr, cfg["steps"], cfg["warmup"])


def frozen_snapshot(model):
    return {k: v.clone() for k, v in model.state_dict().items() if not k.startswith(REFIT_PREFIXES)}


def check_frozen(model, snapshot):
    state = model.state_dict()
    for k, v in snapshot.items():
        if not torch.equal(state[k], v):
            raise AssertionError(f"frozen tensor {k} changed during refit")


def refit(model, cfg, seed, noise):
    """Train only REFIT_PREFIXES, in place, on `model`, which already carries the new readout."""
    params = []
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith(REFIT_PREFIXES))
        if p.requires_grad:
            params.append(p)
    snapshot = frozen_snapshot(model)
    g = data_gen("refit", seed, noise, cfg["train_n"], cfg)  # same batches for every (T, I) in a seed
    info = fit(model, params, cfg, g, noise, cfg["refit_lr"], cfg["refit_steps"], cfg["refit_warmup"])
    check_frozen(model, snapshot)
    return info


@torch.no_grad()
def eval_pass(model, cfg, kind, seed, noise, n, n_batches):
    """Accuracy, needle mass and retrieval as the model currently reads out. The retrieval
    fields come from raw cross scores (B, heads, 1, n), but those depend on the self normalizer."""
    g = data_gen(kind, seed, noise, n, cfg)
    sums, seen, fwd = defaultdict(float), 0, 0.0
    model.ca.capture = True
    heads = model.ca.h
    for _ in range(n_batches):
        keys, vals, q, y, idx = make_batch(g, cfg["eval_batch"], n, noise, cfg)
        t0 = time.perf_counter()
        logits = model(keys, vals, q)
        fwd += time.perf_counter() - t0
        size = y.numel()
        seen += size
        sums["loss"] += F.cross_entropy(logits, y, reduction="sum").item()
        sums["acc"] += (logits.argmax(-1) == y).sum().item()
        best = (keys * q[:, None]).sum(-1).argmax(-1)
        sums["matched_filter_acc"] += (vals[torch.arange(size), best] == y).sum().item()
        z, w = model.ca.last
        needle = w[:, :, 0].gather(-1, idx[:, None, None].expand(-1, heads, 1))
        sums["needle_mass"] += needle.mean().item() * size
        sums["needle_mass_best_head"] += needle.max(1).values.mean().item() * size
        hit = z[:, :, 0].argmax(-1) == idx[:, None]  # (B, heads)
        sums["ret_best"] += hit.any(1).sum().item()
        for h in range(heads):
            sums[f"ret_head{h}"] += hit[:, h].sum().item()
    model.ca.capture, model.ca.last = False, None
    res = {k: clean(v / seen) for k, v in sums.items()}
    res["forward_ms_per_batch"] = clean(1e3 * fwd / n_batches)
    return res


def retrieval(model, trained_arm, cfg, kind, seed, noise, n, n_batches):
    """Retrieval of the scores as trained: `trained_arm` on both layers, whatever readout the
    model carries. Independent of the readout and of any refit, since those scores are frozen.
    Also returns matched-filter accuracy, which depends on data alone."""
    res = eval_pass(view(model, trained_arm, cfg), cfg, kind, seed, noise, n, n_batches)
    return {k: v for k, v in res.items() if k.startswith("ret_") or k == "matched_filter_acc"}


def run_cell(task):
    try:
        return _run_cell(task)
    except Exception:
        return dict(task={k: task[k] for k in ("arm", "seed", "lr", "noise", "stage")},
                    error=traceback.format_exc())


def _run_cell(task):
    cfg, arm, seed, lr, noise = task["cfg"], task["arm"], task["seed"], task["lr"], task["noise"]
    start = time.perf_counter()
    model, info = train_model(cfg, arm, seed, lr, noise)
    val = eval_pass(model, cfg, "val", seed, noise, cfg["train_n"], cfg["val_batches"])
    rec = dict(arm=arm, seed=seed, lr=lr, noise=noise, stage=task["stage"], cfg_hash=task["cfg_hash"],
               val_acc=val["acc"], val_loss=val["loss"], lr_val_acc=task.get("lr_val_acc"), **info)
    if task["stage"] == "final":  # the only code path that evaluates longer lengths or test data
        lengths = [cfg["train_n"] * f for f in cfg["factors"]]
        batches = cfg["test_batches"]
        rec["retrieval"] = {str(n): retrieval(model, arm, cfg, "test", seed, noise, n, batches) for n in lengths}
        rec["evals"], rec["refit"] = {}, {}
        for readout in readouts_for(arm):
            plain = view(model, readout, cfg)
            fitted = copy.deepcopy(plain)
            rec["refit"][readout] = refit(fitted, cfg, seed, noise)
            rec["evals"][readout] = {}
            for name, m in (("none", plain), ("refit", fitted)):
                rows = {str(n): eval_pass(m, cfg, "test", seed, noise, n, batches) for n in lengths}
                rec["evals"][readout][name] = {n: {k: row[k] for k in ROW_KEYS} for n, row in rows.items()}
    rec["wall_s"] = clean(time.perf_counter() - start)
    return rec


def init_worker():
    e11.init_worker()


def run_tasks(tasks, workers, out, records):
    """Run the cells and append each as it finishes. Returns the number of failed cells."""
    if not tasks:
        return 0
    failed = 0
    if workers > 1:
        pool = mp.get_context("spawn").Pool(workers, initializer=init_worker)
        results = pool.imap_unordered(run_cell, tasks, chunksize=1)
    else:
        init_worker()
        pool, results = None, map(run_cell, tasks)
    for i, rec in enumerate(results, 1):
        if "error" in rec:
            failed += 1
            print(f"[{i}/{len(tasks)}] FAILED {rec['task']}\n{rec['error']}", flush=True)
            continue
        e11.append_line(out, rec)
        records[(rec["arm"], rec["seed"], rec["lr"], rec["noise"])] = rec
        print(f"[{i}/{len(tasks)}] {rec['stage']} {rec['arm']} seed={rec['seed']} lr={rec['lr']} "
              f"noise={rec['noise']} val_acc={rec['val_acc']} step_ms={rec['step_ms_median']} "
              f"wall={rec['wall_s']}s nonfinite={rec['nonfinite_steps']}", flush=True)
    if pool is not None:
        pool.close()
        pool.join()
    return failed


def select_lrs(records, cfg):
    """Best mean validation accuracy at N = 8, per arm and noise. Ties go to lower validation
    loss, then the smaller lr. Reads nothing but validation fields."""
    chosen = {}
    for arm in ARMS:
        for noise in cfg["noises"]:
            table = {}
            for lr in cfg["lrs"]:
                rows = [records[(arm, s, lr, noise)] for s in cfg["tune_seeds"]]
                losses = [r["val_loss"] if r["val_loss"] is not None else math.inf for r in rows]
                table[lr] = (sum(r["val_acc"] or 0.0 for r in rows) / len(rows), sum(losses) / len(rows))
            best = max(cfg["lrs"], key=lambda v: (round(table[v][0], 6), -table[v][1], -v))
            chosen[(arm, noise)] = (best, {str(v): table[v][0] for v in cfg["lrs"]})
    return chosen


def print_projection(records, smoke_wall, workers):
    """Timings were measured with the same number of workers running, so contention between
    them is already in the numbers: wall time is core time / workers."""
    total = 0.0
    print(f"\nProjection of the full run from this smoke run ({workers} workers, smoke wall {smoke_wall / 60:.1f} min):")
    for arm in ARMS:
        r = next(v for v in records.values() if v["arm"] == arm and v["stage"] == "final")
        rows = [row for by_refit in r["evals"].values() for by_n in by_refit.values() for row in by_n.items()]
        forward = defaultdict(list)
        for n, row in rows:
            forward[n].append(row["forward_ms_per_batch"] / 1e3)
        forward = {n: sum(v) / len(v) for n, v in forward.items()}
        refit_ms = sum(v["step_ms_median"] for v in r["refit"].values()) / len(r["refit"])
        n_readouts = len(r["evals"])
        train = FULL["steps"] * r["step_ms_median"] / 1e3
        val = FULL["val_batches"] * forward[str(FULL["train_n"])]
        refits = FULL["refit_steps"] * refit_ms / 1e3 * n_readouts
        evals = sum(forward.values()) * FULL["test_batches"] * (2 * n_readouts + 1)
        cell = (train + val) * len(FULL["noises"]) * len(FULL["lrs"]) * len(FULL["tune_seeds"])
        cell += (train + val + refits + evals) * len(FULL["noises"]) * len(FULL["final_seeds"])
        total += cell
        print(f"  {arm:10s} step={r['step_ms_median']}ms refit_step={refit_ms:.0f}ms "
              f"forward_s_by_n={ {n: round(v, 3) for n, v in forward.items()} } total={cell / 60:.0f} core-min")
    print(f"  TOTAL {total / 60:.0f} core-min: {total / 3600 / workers:.1f} hours at {workers} workers. Budget: 4 hours.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--test-batches", type=int, default=None)
    ap.add_argument("--final-seeds", type=int, default=None)
    args = ap.parse_args()
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    started = time.perf_counter()

    cfg = dict(SMOKE if args.smoke else FULL)
    if args.steps:
        cfg["steps"] = args.steps
    if args.test_batches:
        cfg["test_batches"] = args.test_batches
    if args.final_seeds:
        cfg["final_seeds"] = [200 + i for i in range(args.final_seeds)]
        if args.final_seeds < 5 and not args.smoke:
            print("WARNING: the pre-registration requires five seeds", flush=True)
    here = Path(__file__).resolve().parent
    out = Path(args.out) if args.out else here / ("results_smoke.jsonl" if args.smoke else "results.jsonl")
    if args.smoke:  # always re-measure
        out.unlink(missing_ok=True)
    # Seed lists and workers are excluded, so that adding seeds extends an existing run.
    hashed = {k: v for k, v in cfg.items() if k not in ("tune_seeds", "final_seeds")}
    cfg_hash = hashlib.sha1(json.dumps(hashed, sort_keys=True).encode()).hexdigest()[:12]
    records = e11.load_records(out, cfg_hash)
    print(f"cfg_hash={cfg_hash} out={out} resumed_cells={len(records)} workers={args.workers}", flush=True)

    def base(arm, seed, lr, noise, stage):
        return dict(cfg=cfg, cfg_hash=cfg_hash, arm=arm, seed=seed, lr=lr, noise=noise, stage=stage)

    tune = [base(a, s, lr, nz, "tune") for s in cfg["tune_seeds"] for nz in cfg["noises"] for lr in cfg["lrs"]
            for a in ARMS if (a, s, lr, nz) not in records]
    if run_tasks(tune, args.workers, out, records):
        sys.exit("LR check cells failed; rerun to retry them. Final stage not started.")

    chosen = select_lrs(records, cfg)
    for (arm, noise), (lr, scores) in chosen.items():
        print(f"selected {arm} noise={noise} lr={lr} val_acc_by_lr={scores}", flush=True)
    for noise in cfg["noises"]:
        fired = any(chosen[(a, noise)][0] != cfg["lrs"][0] for a in ("softmax", "entmax15"))
        print(f"pre-registered LR gate at noise={noise}: {'fired' if fired else 'not fired'}", flush=True)
    # Seed first, so that stopping early leaves whole seeds, which are what the pairing needs.
    final = [dict(base(a, s, chosen[(a, nz)][0], nz, "final"), lr_val_acc=chosen[(a, nz)][1])
             for s in cfg["final_seeds"] for nz in cfg["noises"] for a in ARMS
             if (a, s, chosen[(a, nz)][0], nz) not in records]
    if run_tasks(final, args.workers, out, records):
        sys.exit("final cells failed; rerun to retry them.")
    print("done", flush=True)
    if args.smoke:
        print_projection(records, time.perf_counter() - started, args.workers)


if __name__ == "__main__":
    main()
