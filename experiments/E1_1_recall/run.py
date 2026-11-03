"""E1.1: associative recall, length extrapolation. Pre-registered in hypotheses/H1_*.md.

TASK. An example is a set of N (key, value) pairs and a query key; the target is the
value paired with the query. Each pair is ONE token, key embedding + value embedding, and
there is no positional encoding of any kind. The model is therefore permutation equivariant
over pairs, and position cannot solve the task, by construction. Keys are fresh random unit
vectors in every example, so nothing can be memorized; only a learned similarity works.
(In 2506.16640's token-level K V K V task, no-PE models fail at 2x because they cannot tell
which value follows which key. The pair token removes that unrelated failure.)

MODEL. One self-attention block over pairs, then one cross-attention block from the query.
The cross row has exactly n = N candidates and one needle. The self rows are the
"no needle" rows that H1 lists as a risk. Head dim 32 = key dim, so one head can represent
the full key similarity; a smaller head would cap the signal below the task's own limit.

NOISE KNOB. Stored keys and the query get independent noise of norm about eta:
    k_obs = k + eta e,  q_obs = k_needle + eta e',  e ~ N(0, I/key_dim).
Chosen over near-duplicates because it moves sigma/mu in a way H1's formula can be read
against. For a matched filter mu is about 1 and sigma about (1 + eta^2)/sqrt(32), so with
L = log N + log 9, the E1.0 ratio mu / (sigma sqrt(2L)) is:
    eta      N=8    N=256
    0.0      1.93   1.44    above threshold everywhere
    0.6      1.42   1.06    reaches threshold at the longest length
    0.9      1.07   0.79    near threshold in training, below it at long lengths
Below threshold, length failure is about signal (H1 detection limit), so each length also
logs matched-filter accuracy as the ceiling reference.

FAVOURS REM, by design of this knob: independent keys give independent, near-Gaussian
distractor scores and one needle, which are H1's own assumptions. --dup-frac F makes a
fraction F of distractors near-copies of other distractors (clustered scores, where
n = count is the wrong effective n; 2610.34682). Run it to a separate --out before crediting
any REM win. Also, this short step budget gives learned-temperature arms little time to
converge, while REM has nothing to learn.

SELECTION. Stage "tune" trains arm x noise x lr on tune seeds and evaluates ONLY validation
data at the training length. Long lengths are not even computed for it. Stage "final" trains
on disjoint seeds at the selected lr and tests all lengths. Same grid and steps for all arms.
Validation and test data, training batches and shared init are keyed by seed, not arm.

BUDGET (estimate, not measured; --smoke prints a measured projection). Assumes 10 ms/step
(15 for calib) on one Colab thread, and forward batch at N=256 of 0.4 s (1.5 s for calib).
    tune:  7 arms x 3 noise x 3 lr x 2 seeds = 126 runs x 2500 steps, about 27 s each = 3400 core-s
    final: 7 arms x 3 noise x 5 seeds        = 105 runs, about 36 s each               = 3800 core-s
Total about 7200 core-s. Two workers take 60 min on two real cores and about 90 min if the
2 vCPUs are hyperthreads of one core (about 1.3x throughput). If the smoke projection is
far over 90 min, lower --steps or --tune-seeds. Same flags apply to every arm.
Train-length accuracy must approach the matched filter for any verdict to mean anything;
analyze.py checks this. If it does not, raise --steps rather than trusting the result.

Usage: python run.py --smoke
       nohup python run.py --out results.jsonl > run.log 2>&1 &     (rerun to resume)
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
import zlib
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mai.attention import NORMALIZERS, MultiHeadAttention, rem_beta, row_audit  # noqa: E402

KAPPA = math.log(9.0)  # target mass 0.9, as in E1.0; fixed and untuned
FULL = dict(train_n=8, factors=[1, 2, 4, 8, 16, 32], d_model=64, n_heads=2, d_ff=128, key_dim=32,
            n_values=16, batch=64, steps=2500, warmup=100, clip=1.0, lrs=[3e-4, 1e-3, 3e-3],
            noises=[0.0, 0.6, 0.9], tune_seeds=[0, 1], final_seeds=[100, 101, 102, 103, 104],
            val_batches=32, test_batches=16, eval_batch=64, kappa=KAPPA, calib_p=0.9, dup_frac=0.0)
SMOKE = dict(FULL, factors=[1, 4, 32], steps=30, warmup=5, lrs=[1e-3], noises=[0.6], tune_seeds=[0],
             final_seeds=[100], val_batches=1, test_batches=1)
AUDIT_ROWS = 64


def stable_seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def data_gen(kind, seed, noise, n, cfg):
    return torch.Generator().manual_seed(stable_seed(kind, seed, noise, n, cfg["dup_frac"]))


def make_batch(g, batch, n, noise, cfg):
    kd = cfg["key_dim"]
    keys = F.normalize(torch.randn(batch, n, kd, generator=g), dim=-1)
    vals = torch.randint(0, cfg["n_values"], (batch, n), generator=g)
    idx = torch.randint(0, n, (batch,), generator=g)
    rows = torch.arange(batch)
    if cfg["dup_frac"] > 0:  # distractors only; the needle key stays unique
        chosen = torch.rand(batch, n, generator=g) < cfg["dup_frac"]
        src = torch.randint(0, n, (batch, n), generator=g)
        chosen &= src != idx[:, None]
        chosen[rows, idx] = False
        copies = keys.gather(1, src[..., None].expand(-1, -1, kd))
        copies = copies + 0.1 * torch.randn(batch, n, kd, generator=g) / math.sqrt(kd)
        keys = F.normalize(torch.where(chosen[..., None], copies, keys), dim=-1)
    k_obs = keys + noise * torch.randn(batch, n, kd, generator=g) / math.sqrt(kd)
    q_obs = keys[rows, idx] + noise * torch.randn(batch, kd, generator=g) / math.sqrt(kd)
    return k_obs, vals, q_obs, vals[rows, idx], idx


class RecallModel(nn.Module):
    def __init__(self, cfg, arm):
        super().__init__()
        d = cfg["d_model"]
        kw = dict(normalizer=arm, train_n=cfg["train_n"], kappa=cfg["kappa"], calib_p=cfg["calib_p"])
        self.key_in = nn.Linear(cfg["key_dim"], d)
        self.val_in = nn.Embedding(cfg["n_values"], d)
        self.query_type = nn.Parameter(0.02 * torch.randn(d))
        self.sa = MultiHeadAttention(d, cfg["n_heads"], **kw)
        self.ca = MultiHeadAttention(d, cfg["n_heads"], **kw)
        self.ff1 = nn.Sequential(nn.Linear(d, cfg["d_ff"]), nn.GELU(), nn.Linear(cfg["d_ff"], d))
        self.ff2 = nn.Sequential(nn.Linear(d, cfg["d_ff"]), nn.GELU(), nn.Linear(cfg["d_ff"], d))
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(6))
        self.head = nn.Linear(d, cfg["n_values"])

    def forward(self, keys, vals, q):
        ln = self.norms
        x = self.key_in(keys) + self.val_in(vals)
        x = x + self.sa(ln[0](x))
        x = x + self.ff1(ln[1](x))
        y = (self.key_in(q) + self.query_type)[:, None]
        mem = ln[3](x)
        y = y + self.ca(ln[2](y), mem, mem)
        y = y + self.ff2(ln[4](y))
        return self.head(ln[5](y))[:, 0]


def clean(x):
    x = float(x)
    return float(f"{x:.5g}") if math.isfinite(x) else None


def audit_layer(m, cfg):
    """Per head: quantiles and a row sample of what E1.2 needs, from the first test batch."""
    z, w = m.last
    stats = row_audit(z)
    full = torch.ones_like(z, dtype=torch.bool)
    columns = dict(mean=stats["mean"], std=stats["std"], exkurt=stats["exkurt"],
                   mu_over_sigma=stats["mu_over_sigma"],
                   beta_rem=rem_beta(z, full, cfg["kappa"])[0],
                   beta_arm=m.row_beta(z).detach().expand_as(stats["mean"]),
                   top_mass=w.max(-1, keepdim=True).values)
    names = list(columns)
    probs = torch.tensor([0.1, 0.5, 0.9])
    out = {}
    for h in range(z.shape[1]):
        cols = {k: v[:, h].reshape(-1) for k, v in columns.items()}
        n_rows = cols["mean"].numel()
        pick = torch.linspace(0, n_rows - 1, min(n_rows, AUDIT_ROWS)).long()
        out[f"head{h}"] = dict(
            columns=names,
            rows=[[clean(cols[k][i]) for k in names] for i in pick.tolist()],
            q10_50_90={k: [clean(q) for q in torch.nanquantile(cols[k], probs).tolist()] for k in names})
    return out


@torch.no_grad()
def evaluate(model, cfg, kind, seed, noise, n, n_batches, audit=False):
    g = data_gen(kind, seed, noise, n, cfg)
    sums, seen, fwd, audit_out = defaultdict(float), 0, 0.0, None
    model.sa.capture = model.ca.capture = True
    for b in range(n_batches):
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
        for name, m in (("self", model.sa), ("cross", model.ca)):
            w = m.last[1]
            sums[f"{name}_top_mass"] += w.max(-1).values.mean().item() * size
            sums[f"{name}_entropy"] += -(w * w.clamp(min=1e-12).log()).sum(-1).mean().item() * size
        w = model.ca.last[1]  # (B, h, 1, n)
        needle = w[:, :, 0].gather(-1, idx[:, None, None].expand(-1, w.shape[1], 1))
        sums["needle_mass"] += needle.mean().item() * size
        sums["needle_mass_best_head"] += needle.max(1).values.mean().item() * size
        if audit and b == 0:
            audit_out = dict(self=audit_layer(model.sa, cfg), cross=audit_layer(model.ca, cfg))
    model.sa.capture = model.ca.capture = False
    model.sa.last = model.ca.last = None
    res = {k: clean(v / seen) for k, v in sums.items()}
    res["forward_ms_per_batch"] = clean(1e3 * fwd / n_batches)
    return res, audit_out


def run_cell(task):
    try:
        return _run_cell(task)
    except Exception:
        return dict(task={k: task[k] for k in ("arm", "seed", "lr", "noise", "stage")},
                    error=traceback.format_exc()), None


def _run_cell(task):
    cfg, arm, seed, lr, noise = task["cfg"], task["arm"], task["seed"], task["lr"], task["noise"]
    start = time.perf_counter()
    torch.manual_seed(stable_seed("init", seed))
    model = RecallModel(cfg, arm)
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    g = data_gen("train", seed, noise, cfg["train_n"], cfg)
    steps, step_times, grad_norms, tail_loss, nonfinite = cfg["steps"], [], [], [], 0
    for t in range(steps):
        t0 = time.perf_counter()
        for group in opt.param_groups:
            group["lr"] = lr * min(1.0, (t + 1) / cfg["warmup"]) * 0.5 * (1 + math.cos(math.pi * t / steps))
        keys, vals, q, y, _ = make_batch(g, cfg["batch"], cfg["train_n"], noise, cfg)
        loss = F.cross_entropy(model(keys, vals, q), y)
        opt.zero_grad()
        loss.backward()
        norm = float(nn.utils.clip_grad_norm_(params, cfg["clip"]))
        if math.isfinite(norm):
            opt.step()
            grad_norms.append(norm)
        else:  # skipped and counted, so a bad gradient cannot poison the weights
            nonfinite += 1
        step_times.append(time.perf_counter() - t0)
        if t >= steps - 100:
            tail_loss.append(loss.item())
    step_times.sort()
    grad_norms.sort()
    val, _ = evaluate(model, cfg, "val", seed, noise, cfg["train_n"], cfg["val_batches"])
    rec = dict(arm=arm, seed=seed, lr=lr, noise=noise, stage=task["stage"], cfg_hash=task["cfg_hash"],
               n_params=sum(p.numel() for p in params),
               n_extra_params=model.sa.n_extra_params() + model.ca.n_extra_params(),
               step_ms_mean=clean(1e3 * sum(step_times) / len(step_times)),
               step_ms_median=clean(1e3 * step_times[len(step_times) // 2]),
               grad_norm_median=clean(grad_norms[len(grad_norms) // 2]) if grad_norms else None,
               nonfinite_steps=nonfinite, train_loss_tail=clean(sum(tail_loss) / max(len(tail_loss), 1)),
               val_acc=val["acc"], val_loss=val["loss"], lr_val_acc=task.get("lr_val_acc"),
               learned=dict(self=model.sa.learned(), cross=model.ca.learned()))
    audit = None
    if task["stage"] == "final":  # the only code path that ever evaluates longer lengths
        by_n, audit_by_n = {}, {}
        for f in cfg["factors"]:
            n = cfg["train_n"] * f
            by_n[str(n)], audit_by_n[str(n)] = evaluate(model, cfg, "test", seed, noise, n,
                                                        cfg["test_batches"], audit=True)
        rec["by_n"] = by_n
        audit = dict(arm=arm, seed=seed, lr=lr, noise=noise, learned=rec["learned"], by_n=audit_by_n)
    rec["wall_s"] = clean(time.perf_counter() - start)
    return rec, audit


def init_worker():
    torch.set_num_threads(1)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)  # survive the launching shell going away


def load_records(path, cfg_hash):
    records = {}
    if not path.exists():
        return records
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            print(f"skipping a damaged line in {path}", flush=True)
            continue
        if r["cfg_hash"] != cfg_hash:
            sys.exit(f"{path} holds runs from a different config; use another --out")
        records[(r["arm"], r["seed"], r["lr"], r["noise"])] = r
    return records


def append_line(path, obj):
    with open(path, "ab+") as f:
        if f.tell() > 0:  # a killed process may have left a partial last line
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
        f.write((json.dumps(obj) + "\n").encode())
        f.flush()
        os.fsync(f.fileno())


def run_tasks(tasks, workers, out, audit_out, records):
    """Run the cells, append as each finishes. An audit line is written before its result
    line, so a recorded cell always has its audit. Returns the number of failed cells."""
    if not tasks:
        return 0
    failed = 0
    if workers > 1:
        pool = mp.get_context("spawn").Pool(workers, initializer=init_worker)
        results = pool.imap_unordered(run_cell, tasks, chunksize=1)
    else:
        init_worker()
        pool, results = None, map(run_cell, tasks)
    for i, (rec, audit) in enumerate(results, 1):
        if "error" in rec:
            failed += 1
            print(f"[{i}/{len(tasks)}] FAILED {rec['task']}\n{rec['error']}", flush=True)
            continue
        if audit is not None:
            append_line(audit_out, audit)
        append_line(out, rec)
        records[(rec["arm"], rec["seed"], rec["lr"], rec["noise"])] = rec
        print(f"[{i}/{len(tasks)}] {rec['stage']} {rec['arm']} seed={rec['seed']} lr={rec['lr']} "
              f"noise={rec['noise']} val_acc={rec['val_acc']} step_ms={rec['step_ms_median']} "
              f"wall={rec['wall_s']}s nonfinite={rec['nonfinite_steps']}", flush=True)
    if pool is not None:
        pool.close()
        pool.join()
    return failed


def select_lrs(records, cfg):
    """Best mean validation accuracy over tune seeds at the training length. Ties go to
    lower validation loss, then smaller lr. Reads nothing but validation fields."""
    chosen = {}
    for arm in NORMALIZERS:
        for noise in cfg["noises"]:
            table = {}
            for lr in cfg["lrs"]:
                rows = [records[(arm, s, lr, noise)] for s in cfg["tune_seeds"]]
                losses = [r["val_loss"] if r["val_loss"] is not None else math.inf for r in rows]
                table[lr] = (sum(r["val_acc"] or 0.0 for r in rows) / len(rows), sum(losses) / len(rows))
            best = max(cfg["lrs"], key=lambda v: (round(table[v][0], 6), -table[v][1], -v))
            chosen[(arm, noise)] = (best, {str(v): table[v][0] for v in cfg["lrs"]})
    return chosen


def print_projection(records, smoke_cfg):
    full, total = FULL, 0.0
    longest = max(f * smoke_cfg["train_n"] for f in smoke_cfg["factors"])
    print("\nProjection of the full run from this smoke run (one thread, uncontended):")
    for arm in NORMALIZERS:
        r = next(v for v in records.values() if v["arm"] == arm and v["stage"] == "final")
        train = full["steps"] * r["step_ms_median"] / 1e3
        forward = r["by_n"][str(longest)]["forward_ms_per_batch"] / 1e3
        evals = sum(forward * (f * full["train_n"] / longest) ** 2 * full["test_batches"] for f in full["factors"])
        n_noise = len(full["noises"])
        cell = train * n_noise * len(full["lrs"]) * len(full["tune_seeds"])
        cell += (train + evals) * n_noise * len(full["final_seeds"])
        total += cell
        print(f"  {arm:16s} params={r['n_params']} extra={r['n_extra_params']} "
              f"step={r['step_ms_median']}ms forward@{longest}={forward:.2f}s total={cell / 60:.1f} core-min")
    print(f"  TOTAL {total / 60:.0f} core-min: {total / 120:.0f} min at 2 real cores, "
          f"{total / 78:.0f} min at 1.3x hyperthread throughput. Audit cost not included.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--tune-seeds", type=int, default=None)
    ap.add_argument("--final-seeds", type=int, default=None)
    ap.add_argument("--dup-frac", type=float, default=0.0)
    args = ap.parse_args()
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)

    cfg = dict(SMOKE if args.smoke else FULL, dup_frac=args.dup_frac)
    if args.steps:
        cfg["steps"] = args.steps
    if args.tune_seeds:
        cfg["tune_seeds"] = list(range(args.tune_seeds))
    if args.final_seeds:
        cfg["final_seeds"] = [100 + i for i in range(args.final_seeds)]
    if args.final_seeds and args.final_seeds < 5 and not args.smoke:
        print("WARNING: the pre-registration requires at least 5 seeds", flush=True)
    workers = args.workers or (1 if args.smoke else 2)
    here = Path(__file__).resolve().parent
    out = Path(args.out) if args.out else here / ("results_smoke.jsonl" if args.smoke else "results.jsonl")
    audit_out = out.with_name(out.stem + "_audit.jsonl")
    if args.smoke:  # always re-measure
        out.unlink(missing_ok=True)
        audit_out.unlink(missing_ok=True)
    # Seed lists and workers are excluded, so that adding seeds extends an existing run.
    hashed = {k: v for k, v in cfg.items() if k not in ("tune_seeds", "final_seeds")}
    cfg_hash = hashlib.sha1(json.dumps(hashed, sort_keys=True).encode()).hexdigest()[:12]
    records = load_records(out, cfg_hash)
    print(f"cfg_hash={cfg_hash} out={out} resumed_cells={len(records)} workers={workers}", flush=True)

    def base(arm, seed, lr, noise, stage):
        return dict(cfg=cfg, cfg_hash=cfg_hash, arm=arm, seed=seed, lr=lr, noise=noise, stage=stage)

    tune = [base(a, s, lr, nz, "tune") for a in NORMALIZERS for nz in cfg["noises"] for lr in cfg["lrs"]
            for s in cfg["tune_seeds"] if (a, s, lr, nz) not in records]
    if run_tasks(tune, workers, out, audit_out, records):
        sys.exit("tuning cells failed; rerun to retry them. Final stage not started.")

    chosen = select_lrs(records, cfg)
    for (arm, noise), (lr, scores) in chosen.items():
        print(f"selected {arm} noise={noise} lr={lr} val_acc_by_lr={scores}", flush=True)
    final = []
    for (arm, noise), (lr, scores) in chosen.items():
        for s in cfg["final_seeds"]:
            if (arm, s, lr, noise) not in records:
                final.append(dict(base(arm, s, lr, noise, "final"), lr_val_acc=scores))
    if run_tasks(final, workers, out, audit_out, records):
        sys.exit("final cells failed; rerun to retry them.")
    print("done", flush=True)
    if args.smoke:
        print_projection(records, cfg)


if __name__ == "__main__":
    main()
