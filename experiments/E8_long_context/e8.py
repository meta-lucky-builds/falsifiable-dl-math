#!/usr/bin/env python3
"""E8 / H8: why byte-level models break past their training length.

Pre-registration: hypotheses/H8_long_context_failure.md. Linux only, like E7.

    python e8.py            full run on all visible GPUs; rerun to resume
    python e8.py --smoke    all arms and code paths on real data, few steps; prints a projection
    python e8.py --mini     small model, synthetic data, CPU

E7's data, training loop, job queue and launcher run unchanged, and E7 itself stays as recorded.
install() lists the only things E8 replaces in them.

Choices the pre-registration leaves open:
- Rows depend only on earlier columns, so one pass over windows of length max(Ns) gives every L(N).
- Lw limits attention in every layer, so a row's reach through the stack is more than W. Lw can
  therefore differ from T, and Lw is what E(N) uses, because it has the same targets as L(N).
- The windows are a function of the base config, so every arm, control included, reads the same ones.
- Model weights are kept in models/ after each job, so a surprising result can be examined further.
"""
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "E7_gpu"))
import e7  # noqa: E402

ARM_SPECS = {
    "rope_1e4": dict(theta=1e4, alibi=False, ctx_factor=1),
    "rope_5e5": dict(theta=5e5, alibi=False, ctx_factor=1),
    "alibi": dict(theta=None, alibi=True, ctx_factor=1),
    "rope_1e4_alibi": dict(theta=1e4, alibi=True, ctx_factor=1),
    "control_ctx4x": dict(theta=1e4, alibi=False, ctx_factor=4),
}
PROBE_SEED_E8 = 8003
LR = 1.2e-3
E7_MAKE_CONFIG = e7.make_config  # install() replaces e7.make_config with the function below


def make_config(mode):
    cfg = dict(E7_MAKE_CONFIG(mode), lrs=[], lr=LR, final_seeds=[3000, 3001, 3002])
    if mode == "smoke":
        cfg.update(final_seeds=[3000, 3001], compile_seeds=[3000], record_windows=16,
                   Ds=[32, 64, 128, 200, 400, 800, 1600],
                   inject_crash=dict(job="final_rope_1e4_s3000", step=15))
    elif mode == "mini":
        cfg.update(final_seeds=[3000, 3001], record_windows=8, Ds=[16, 24, 48, 96, 200],
                   inject_crash=dict(job="final_rope_1e4_s3000", step=15))
    else:
        cfg.update(record_windows=256, Ds=[32, 64, 128, 200, 400, 800, 1600])
    return cfg


def arm_config(cfg, arm):
    f = ARM_SPECS[arm]["ctx_factor"]
    return dict(cfg, ctx=cfg["ctx"] * f, batch=max(1, cfg["batch"] // f))


def alibi_slopes(h):
    return torch.tensor([2.0 ** (-8.0 * k / h) for k in range(1, h + 1)]).view(h, 1, 1)


def distance_edges(cfg):
    """Bucket edges for the distance profile, as multiples of the base training context."""
    w = cfg["ctx"]
    return [0, w // 16, w // 4, w, 2 * w, 4 * w, 8 * w]


# ---------------------------------------------------------------- model

class Recorder:
    """Accumulates attention statistics for one layer. Sums run over batch and head."""

    def __init__(self, T, keep, far_from, edges, device):
        self.T, self.keep, self.far_from = T, keep, far_from
        self.far = torch.zeros(T, dtype=torch.float64, device=device)
        self.row_count = torch.zeros(T, dtype=torch.float64, device=device)
        n = len(edges) - 1
        self.mass = torch.zeros(n, dtype=torch.float64, device=device)
        self.logit = torch.zeros(n, dtype=torch.float64, device=device)
        self.count = torch.zeros(n, dtype=torch.float64, device=device)
        inner = torch.tensor(edges[1:-1], device=device)
        self.bucket = torch.bucketize(torch.arange(T, device=device), inner, right=True)

    def add(self, w, rel, mask, dist, rows):
        total = w.shape[0] * w.shape[1]
        ws = w.sum((0, 1))
        self.far.index_add_(0, rows, (ws * (dist >= self.far_from)).sum(-1).double())
        self.row_count.index_add_(0, rows, torch.full_like(rows, total, dtype=torch.float64))
        tail = rows >= self.T - self.keep
        if tail.any():
            sel = mask[tail]
            bucket = self.bucket[dist[tail][sel]]
            self.mass.index_add_(0, bucket, ws[tail][sel].double())
            self.logit.index_add_(0, bucket, rel.sum((0, 1))[tail][sel].double())
            self.count.index_add_(0, bucket, torch.full_like(bucket, total, dtype=torch.float64))

    def result(self, bin_width):
        far = self.far / self.row_count
        mass = self.mass / self.mass.sum()
        clean = lambda t: [None if math.isnan(x) else x for x in t.tolist()]
        # 0/0 gives NaN below: a bucket with no column in it
        return dict(far=bin_rows(far, bin_width), mass=mass.tolist(), logit=clean(self.logit / self.count),
                    density=clean(self.mass / self.count))


class E8Attention(e7.Attention):
    """E7's softmax attention, plus an optional ALiBi bias, distance limit and recorder. With none of
    them this computes exactly what E7 computes, which tests/test_e8.py checks."""

    def __init__(self, d, h, train_n, alibi):
        super().__init__(d, h, "softmax", train_n)
        self.register_buffer("slopes", alibi_slopes(h) if alibi else None)
        self.window = None
        self.recorder = None

    def _weights(self, z, mask, n, sparse):
        if sparse:
            raise ValueError("E8 is softmax only")
        dist = rows = None
        if self.slopes is not None or self.window is not None or self.recorder is not None:
            rows = n.long().flatten() - 1
            dist = rows[:, None] - torch.arange(mask.shape[1], device=z.device)
        if self.window is not None:
            mask = mask & (dist < self.window)
        zmax = z.masked_fill(~mask, -math.inf).max(-1, keepdim=True).values.detach()
        logits = self._beta(n) * (z - zmax)
        if self.slopes is not None:
            logits = logits - self.slopes * dist.float()
        top = logits.masked_fill(~mask, -math.inf).max(-1, keepdim=True).values.detach()
        rel = logits - top
        w = F.softmax(rel.masked_fill(~mask, -math.inf), dim=-1)
        if self.recorder is not None:
            self.recorder.add(w, rel, mask, dist, rows)
        return w


class E8Block(e7.Block):
    def __init__(self, cfg, alibi):
        nn.Module.__init__(self)  # same layer order as e7.Block, so a seed gives the same weights
        d = cfg["d_model"]
        self.ln1 = nn.LayerNorm(d)
        self.attn = E8Attention(d, cfg["n_head"], cfg["ctx"], alibi)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))


class E8Model(e7.E7Model):
    def __init__(self, cfg, arm):
        nn.Module.__init__(self)
        spec = ARM_SPECS[arm]
        self.theta = spec["theta"]
        self.dh = cfg["d_model"] // cfg["n_head"]
        self.tok = nn.Embedding(256, cfg["d_model"])
        self.blocks = nn.ModuleList(E8Block(cfg, spec["alibi"]) for _ in range(cfg["n_layer"]))
        self.ln = nn.LayerNorm(cfg["d_model"])
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        res = 0.02 / math.sqrt(2 * cfg["n_layer"])
        for blk in self.blocks:
            nn.init.normal_(blk.attn.out.weight, 0.0, res)
            nn.init.normal_(blk.ff[2].weight, 0.0, res)

    def position_tables(self, T, device):
        """RoPE tables, or the identity rotation for an arm with no RoPE."""
        if self.theta is None:
            shape = (T, self.dh // 2)
            return torch.ones(shape, device=device), torch.zeros(shape, device=device)
        freqs = self.theta ** (-torch.arange(0, self.dh, 2, dtype=torch.float64, device=device) / self.dh)
        return e7.rope_tables(T, self.dh, device, freqs)

    def forward(self, x, keep=None):
        T = x.shape[1]
        cos, sin = self.position_tables(T, x.device)
        h = self.tok(x)
        for blk in self.blocks[:-1]:
            h = blk(h, cos, sin)
        h = self.blocks[-1](h, cos, sin, 0 if keep is None else T - keep)
        with torch.autocast(x.device.type, enabled=False):
            return F.linear(self.ln(h.float()), self.tok.weight)

    def set_window(self, window):
        for blk in self.blocks:
            blk.attn.window = window

    def set_recorders(self, make=None):
        """make() returns a new Recorder for each layer; no argument removes them."""
        for blk in self.blocks:
            blk.attn.recorder = make() if make else None
        return [blk.attn.recorder for blk in self.blocks]


# ---------------------------------------------------------------- evaluation

def bin_rows(values, width):
    """Means over consecutive bins. Input is per row; its length must be a multiple of width."""
    values = torch.as_tensor(values, dtype=torch.float64)
    if values.numel() % width:
        raise ValueError(f"{values.numel()} rows do not divide into bins of {width}")
    return values.view(-1, width).mean(1).tolist()


@torch.no_grad()
def per_row_bits(model, data, ends, T, bs0, bs_state, key, beat=lambda: None):
    """Bits per byte for each row of length-T windows, averaged over the windows."""
    def fn(i, j):
        x, y = e7.window_batch(data, ends[i:j], T, T)
        ce = F.cross_entropy(model(x).reshape(-1, 256).double(), y.reshape(-1), reduction="none")
        return (ce.view(x.shape[0], T).sum(0).cpu(),)
    total, = e7.batched(len(ends), bs0, fn, bs_state, key, beat)
    return total / len(ends) / e7.LOG2


def make_copy_probes(test, ends, D, seed=PROBE_SEED_E8):
    """Input length D + 3: the string starts at 0, and its first 4 bytes repeat so that the last 4
    rows predict bytes 4 to 7 of it, teacher-forced. test: uint8 array. D below 12 would overlap."""
    if D < 12:
        raise ValueError(f"distance {D} does not fit")
    L = D + 3
    strings = e7.PROBE_ALPHABET[np.random.default_rng(seed).integers(
        0, len(e7.PROBE_ALPHABET), (len(ends), 8))]
    x = np.stack([test[e - L:e] for e in ends])
    x[:, 0:8] = strings
    x[:, D - 4:D + 3] = strings[:, :7]
    return x, strings[:, 4:8].copy()


def evaluate(model, arm, cfg, job_cfg, rt, partial, beat=lambda: None):
    """All test-set numbers for one model. Progress survives a crash through `partial`."""
    test, device = rt.data["test"], rt.device
    test_np = test.cpu().numpy()
    ends_np = e7.eval_ends(test_np.size, cfg["eval_windows"], cfg)
    ends = torch.from_numpy(ends_np).to(device)
    T, keep, tokens, W = max(cfg["Ns"]), cfg["keep"], cfg["eval_tokens"], job_cfg["ctx"]
    bs0 = max(1, tokens // T)
    res = json.loads(partial.read_text()) if partial.exists() else dict(metrics={}, seconds=0.0)
    bs_state = {}

    def metric(name, fn):
        if name in res["metrics"]:
            return
        t0 = time.perf_counter()
        res["metrics"][name] = fn()
        res["seconds"] += time.perf_counter() - t0
        e7.atomic_text(partial, json.dumps(res))
        e7.log(f"  {arm} {name} done")

    # one batch-size key for all three passes, since they have the same shape
    metric("L_bins", lambda: bin_rows(per_row_bits(model, test, ends, T, bs0, bs_state, "rows", beat), keep))

    def windowed():
        model.set_window(W)
        try:
            return bin_rows(per_row_bits(model, test, ends, T, bs0, bs_state, "rows", beat), keep)
        finally:
            model.set_window(None)
    metric("Lw_bins", windowed)

    metric("T", lambda: e7.eval_truncated(model, test, ends, keep, W, max(1, tokens // W), bs_state, beat))

    def recorded(length):
        step = max(1, len(ends) // cfg["record_windows"])
        edges = distance_edges(cfg)
        recorders = model.set_recorders(lambda: Recorder(length, keep, W, edges, device))
        try:
            per_row_bits(model, test, ends[::step], length, max(1, tokens // length), bs_state,
                         f"rec{length}", beat)
            return dict(edges=edges, layers=[r.result(keep) for r in recorders])
        finally:
            model.set_recorders()
    metric("attention", lambda: recorded(T))
    metric("attention_short", lambda: recorded(cfg["Ns"][1]))  # its last bin is the rows C3 reads

    for D in cfg["Ds"]:
        def copy(D=D):
            x, y = make_copy_probes(test_np, ends_np, D)
            return e7.eval_copy(model, torch.from_numpy(x), torch.from_numpy(y), device,
                                max(1, tokens // (D + 3)), bs_state, beat)
        metric(f"copy_{D}", copy)

    sha = hashlib.sha1(ends_np.tobytes())
    return res["metrics"], res["seconds"], sha.hexdigest()


def run_job(job, out, cfg, rt, beat=lambda: None):
    jid, t0, arm = job["id"], time.time(), job["arm"]
    p = e7.paths(out)
    job_cfg = arm_config(cfg, arm)
    e7.log(f"{jid}: start, lr {job['lr']}, context {job_cfg['ctx']}, batch {job_cfg['batch']}")
    model, state, compiled = e7.train_loop(job, job_cfg, rt, p["ckpt"] / f"{jid}.pt", beat,
                                           crash_marker=p["queue"] / "crash_injected")
    times = state["times"]
    median = statistics.median(times[min(20, len(times) // 2):]) if times else float("nan")
    res = dict(job, ctx=job_cfg["ctx"], batch=job_cfg["batch"],
               params=sum(p_.numel() for p_ in model.parameters()), step_ms=median * 1e3,
               tokens_per_s=job_cfg["batch"] * job_cfg["ctx"] / median if times else float("nan"),
               nonfinite=state["nonfinite"], val_log=state["val_log"], compiled=compiled,
               resumed_from=state["resumed_from"], **e7.env_info(rt))
    res["val_loss_final"] = e7.val_loss(model, job_cfg, rt, {}, beat)
    models = out / "models"
    models.mkdir(exist_ok=True)
    tmp = models / f".{jid}.{os.getpid()}.tmp"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, models / f"{jid}.pt")
    res["eval"], res["eval_seconds"], res["eval_hash"] = evaluate(
        model, arm, cfg, job_cfg, rt, p["ckpt"] / f"{jid}.eval.json", beat)
    res["wall_seconds"] = time.time() - t0
    e7.log(f"{jid}: done, val {res['val_loss_final']:.4f}, step {res['step_ms']:.1f} ms")
    return res


# ---------------------------------------------------------------- what E7's launcher needs

def all_jobs(cfg):
    return [dict(id=f"final_{arm}_s{seed}", kind="final", arm=arm, seed=seed, lr=cfg["lr"])
            for arm in ARM_SPECS for seed in cfg["final_seeds"]]


def visible_jobs(out, cfg):
    return all_jobs(cfg)


def settled(out, cfg):
    return all(e7.is_done(out, j["id"]) or e7.is_failed(out, j["id"]) for j in all_jobs(cfg))


def ensure_selection(out, cfg):
    """E8 has no tuning. E7's launcher expects this file, so it records the fixed rate."""
    path = e7.paths(out)["selection"]
    if not path.exists():
        e7.atomic_text(path, json.dumps({arm: dict(lr=cfg["lr"]) for arm in ARM_SPECS}))


def code_sha():
    sha = hashlib.sha1()
    for path in (Path(__file__).resolve(), REPO / "experiments" / "E7_gpu" / "e7.py",
                 REPO / "src" / "mai" / "attention.py"):
        sha.update(path.read_bytes())
    return sha.hexdigest()


def print_projection(out, cfg, n_slots):
    full = make_config("full")
    results = [json.loads(p.read_text()) for p in e7.paths(out)["jobs"].glob("*.json")]
    lines = [f"Measured projection of the full run on this machine, {n_slots} workers:"]
    total, longest = 0.0, 0.0
    for arm in ARM_SPECS:
        rows = [r for r in results if r["arm"] == arm]
        train = full["steps"] * min(r["step_ms"] for r in rows) / 1e3
        ev = max(r["eval_seconds"] for r in rows) * full["eval_windows"] / cfg["eval_windows"]
        total += len(full["final_seeds"]) * (train + ev)
        longest = max(longest, train + ev)
        lines.append(f"  {arm:16s} train {train / 3600:6.2f} h, evaluation {ev / 3600:6.2f} h per job")
    lines.append(f"  total {total / 3600:.1f} GPU-hours; about {max(total / n_slots, longest) / 3600:.1f} h "
                 f"on {n_slots} workers")
    e7.atomic_text(out / "projection.txt", "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def install():
    """Everything E8 replaces in E7's machinery. Not done at import, so tests can compare to E7."""
    e7.__file__ = str(Path(__file__).resolve())  # the launcher spawns workers from this file
    e7.ARMS = tuple(ARM_SPECS)
    e7.make_config = make_config
    e7.E7Model = E8Model
    e7.run_job = run_job
    e7.visible_jobs = visible_jobs
    e7.settled = settled
    e7.ensure_selection = ensure_selection
    e7.code_sha = code_sha
    e7.print_projection = print_projection


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    install()
    if "--out" not in argv and "--preflight" not in argv:
        mode = "_smoke" if "--smoke" in argv else "_mini" if "--mini" in argv else ""
        argv += ["--out", str(REPO / "runs" / f"e8{mode}")]
    return e7.main(argv)


if __name__ == "__main__":
    sys.exit(main())
