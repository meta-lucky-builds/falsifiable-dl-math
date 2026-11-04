#!/usr/bin/env python3
"""E7 / H7: attention normalizers on long contexts in a byte-level model of enwik8.

Pre-registration: hypotheses/H7_gpu_attention_on_real_text.md. Linux only (/proc, fcntl).

    python e7.py            full run on all visible GPUs; rerun to resume
    python e7.py --smoke    all arms and code paths on real data, few steps; prints a projection
    python e7.py --mini     small model, synthetic data, CPU

Choices the pre-registration leaves open, all applied to every arm:
- Scores, normalizer and w @ v run in float32 inside autocast-off, and so does the output head.
  bf16 has an 8-bit mantissa, which is too coarse for entmax's cumsum and for 0.005 bits/byte.
- Validation and test run in float32 with no autocast; training uses bf16 as registered.
- Attention rows are processed in blocks. A block with rows [a, b) needs only columns [0, b), so
  this avoids about 44% of the score and sort work and bounds peak memory at N = 2048.
- All N-windows share their end points, so L(N) has identical targets at every N, and T(N), which
  sees only the preceding 256 bytes, is one number per readout. At N = 256 that context reaches
  before the window start, so T(256) sees slightly more than L(256), and G(256) is not a gain.
- Copy probe: each input is `probe_len` bytes. With E = probe_len - 3, the 8-byte string starts at
  E - D, its first 4 bytes occupy [E-4, E), and the last 4 rows predict bytes 4 to 7 of the string,
  teacher-forced. Every D uses the same window length, so only the distance changes.
- Batches are a pure function of (seed, step), so the batch RNG has no state to checkpoint, and
  nothing else in training consumes randomness.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from mai.attention import entmax15  # noqa: E402

ARMS = ("softmax", "ssmax", "entmax15", "entmax15_scaled")
# softmax comes first: gated reuses its numbers wherever every row has n <= ctx.
READOUTS = {"softmax": ("softmax", "entmax15", "ssmax", "gated"), "ssmax": ("ssmax",),
            "entmax15": ("entmax15",), "entmax15_scaled": ("entmax15_scaled",)}
NORM_PARAMS = ("s", "delta", "b_raw", "gamma_raw")
GAMMA_BOUND = 4.0  # as in src/mai/attention.py
EVAL_SEED, VAL_SEED, PROBE_SEED = 7001, 7002, 7003
PROBE_ALPHABET = np.frombuffer(
    b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", dtype=np.uint8)
MAX_ATTEMPTS = 2  # one retry
MAX_SLOT_CRASHES = 5
ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"
ENWIK8_SIZE = 100_000_000
LOG2 = math.log(2.0)

FULL = dict(
    name="full", data="enwik8", data_size=ENWIK8_SIZE,
    n_layer=6, n_head=6, d_model=384, ctx=256, batch=64, steps=30000, warmup=1000,
    wd=0.1, betas=[0.9, 0.99], clip=1.0, lrs=[3e-4, 6e-4, 1.2e-3],
    tune_seed=1000, final_seeds=[2000, 2001, 2002, 2003, 2004],
    Ns=[256, 512, 1024, 2048], primary_Ns=[1024, 2048], keep=64,
    Ds=[128, 512, 1024, 1900], probe_len=2048, eval_windows=2048,
    val_windows=2048, val_every=3000, ckpt_every=2000, eval_tokens=131072,
    compile_seeds=None, inject_crash=None, respawn_delay=30.0, poll=5.0)


def make_config(mode):
    if mode == "full":
        return dict(FULL)
    if mode == "smoke":
        return dict(FULL, name="smoke", steps=24, warmup=5, lrs=[3e-4, 1.2e-3],
                    final_seeds=[2000, 2001], eval_windows=8, val_windows=8, val_every=12,
                    ckpt_every=10, compile_seeds=[2000],
                    inject_crash=dict(job="final_softmax_s2000", step=15),
                    respawn_delay=2.0, poll=2.0)
    if mode == "mini":
        return dict(FULL, name="mini", data="mini", data_size=300_000,
                    n_layer=2, n_head=2, d_model=32, ctx=32, batch=8, steps=40, warmup=5,
                    lrs=[3e-4, 1.2e-3], final_seeds=[2000, 2001],
                    Ns=[32, 64, 128, 256], primary_Ns=[128, 256], keep=8,
                    Ds=[16, 48, 128, 230], probe_len=256, eval_windows=16, val_windows=16,
                    val_every=20, ckpt_every=10, eval_tokens=2048, compile_seeds=[],
                    inject_crash=dict(job="final_softmax_s2000", step=15),
                    respawn_delay=1.0, poll=0.5)
    raise ValueError(mode)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- model

def rope_tables(T, dh, device, freqs=None):
    """cos and sin of (position * frequency), shape (T, dh // 2). Any T works."""
    if freqs is None:
        freqs = 10000.0 ** (-torch.arange(0, dh, 2, dtype=torch.float64, device=device) / dh)
    angle = torch.arange(T, dtype=torch.float64, device=device)[:, None] * freqs
    return angle.cos().float(), angle.sin().float()


def rope(x, cos, sin):
    """Rotate consecutive pairs of the last dim. x: (..., T, dh); cos, sin: (T, dh // 2)."""
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2)


class Attention(nn.Module):
    """Causal multi-head attention; every arm and readout uses this one path. Linear layers come
    first and normalizer parameters are constants, so a seed gives the same weights in all arms."""

    def __init__(self, d, h, arm, train_n):
        super().__init__()
        if d % h or (d // h) % 2:
            raise ValueError("head dim must be a whole, even number for RoPE")
        self.h, self.dh, self.readout, self.train_n = h, d // h, arm, train_n
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d)
        log_n = math.log(train_n)
        self.s0 = 1.0 / log_n  # also the fixed s of the SSMax readout on softmax-trained models

        def per_head(value):
            return nn.Parameter(torch.full((h, 1, 1), float(value)))

        if arm == "ssmax":
            self.s = per_head(self.s0)
        elif arm == "entmax15_scaled":
            self.delta = per_head(0.0)
            self.b_raw = per_head(math.log(math.expm1(1.0 / log_n)))
            self.gamma_raw = per_head(math.atanh(1.0 / GAMMA_BOUND))

    def _beta(self, n):
        base = 1.0 / math.sqrt(self.dh)
        if self.readout in ("softmax", "entmax15", "gated"):
            return base
        if self.readout == "ssmax":
            return getattr(self, "s", self.s0) * n.log() * base
        b, gamma = F.softplus(self.b_raw), GAMMA_BOUND * torch.tanh(self.gamma_raw)
        return (self.delta + b * n.clamp(min=2.0).log().pow(gamma)) * base

    def normalize(self, z, mask, n):
        """Same formula as MultiHeadAttention.normalize in src/mai/attention.py."""
        if z.dtype != torch.float32:  # also catches a compiler ignoring the nested autocast-off
            raise RuntimeError(f"attention core must run in float32, got {z.dtype}")
        if self.readout != "gated":
            return self._weights(z, mask, n, self.readout in ("entmax15", "entmax15_scaled"))
        # Per row: softmax for n <= train_n, entmax15 beyond. n rises along the rows, and the cut
        # can fall inside a block. A block wholly on one side takes the same path as that readout.
        m = int((n <= self.train_n).sum())
        if m == n.shape[0] or m == 0:
            return self._weights(z, mask, n, m == 0)
        return torch.cat((self._weights(z[..., :m, :], mask[:m], n[:m], False),
                          self._weights(z[..., m:, :], mask[m:], n[m:], True)), -2)

    def _weights(self, z, mask, n, sparse):
        zmax = z.masked_fill(~mask, -math.inf).max(-1, keepdim=True).values.detach()
        logits = self._beta(n) * (z - zmax)
        if sparse:
            return entmax15(logits, mask)
        top = logits.masked_fill(~mask, -math.inf).max(-1, keepdim=True).values.detach()
        return F.softmax((logits - top).masked_fill(~mask, -math.inf), dim=-1)

    def forward(self, x, cos, sin, rows_from=0):
        """Returns output rows [rows_from, T) only, each attending to columns up to itself."""
        B, T, _ = x.shape
        h, dh, R = self.h, self.dh, T - rows_from
        q = self.q(x[:, rows_from:]).view(B, R, h, dh).transpose(1, 2)
        k = self.k(x).view(B, T, h, dh).transpose(1, 2)
        v = self.v(x).view(B, T, h, dh).transpose(1, 2)
        with torch.autocast(x.device.type, enabled=False):
            q = rope(q.float(), cos[rows_from:], sin[rows_from:])
            k = rope(k.float(), cos, sin)
            v = v.float()
            parts, step = [], max(32, T // 8)
            for a in range(rows_from, T, step):
                b = min(T, a + step)
                z = q[:, :, a - rows_from:b - rows_from] @ k[:, :, :b].transpose(-1, -2)
                mask = torch.ones(b - a, b, dtype=torch.bool, device=x.device).tril(a)
                n = torch.arange(a + 1, b + 1, dtype=torch.float32, device=x.device)[:, None]
                parts.append(self.normalize(z, mask, n) @ v[:, :, :b])
            o = torch.cat(parts, 2).transpose(1, 2).reshape(B, R, h * dh)
        return self.out(o)


class Block(nn.Module):
    def __init__(self, cfg, arm):
        super().__init__()
        d = cfg["d_model"]
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attention(d, cfg["n_head"], arm, cfg["ctx"])
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, cos, sin, rows_from=0):
        y = self.attn(self.ln1(x), cos, sin, rows_from)
        x = x[:, rows_from:] + y
        return x + self.ff(self.ln2(x))


class E7Model(nn.Module):
    def __init__(self, cfg, arm):
        super().__init__()
        self.dh = cfg["d_model"] // cfg["n_head"]
        self.tok = nn.Embedding(256, cfg["d_model"])
        self.blocks = nn.ModuleList(Block(cfg, arm) for _ in range(cfg["n_layer"]))
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

    def forward(self, x, keep=None):
        """Logits (B, T, 256), or for only the last `keep` rows, which saves the last layer."""
        T = x.shape[1]
        cos, sin = rope_tables(T, self.dh, x.device)
        h = self.tok(x)
        for blk in self.blocks[:-1]:
            h = blk(h, cos, sin)
        h = self.blocks[-1](h, cos, sin, 0 if keep is None else T - keep)
        with torch.autocast(x.device.type, enabled=False):
            return F.linear(self.ln(h.float()), self.tok.weight)

    def set_readout(self, readout):
        for blk in self.blocks:
            if readout == "entmax15_scaled" and not hasattr(blk.attn, "b_raw"):
                raise ValueError("entmax15_scaled needs learned parameters")
            if readout == "gated" and (hasattr(blk.attn, "s") or hasattr(blk.attn, "b_raw")):
                raise ValueError("the gated readout is for softmax-trained models only")
            blk.attn.readout = readout

    def learned(self):
        out = {}
        for blk in self.blocks:
            a = blk.attn
            values = {}
            if hasattr(a, "s"):
                values["s"] = a.s
            if hasattr(a, "b_raw"):
                values = dict(delta=a.delta, b=F.softplus(a.b_raw),
                              gamma=GAMMA_BOUND * torch.tanh(a.gamma_raw))
            for name, value in values.items():
                out.setdefault(name, []).append(value.detach().flatten().tolist())
        return out


def make_optimizer(model, lr, cfg, device):
    """Weight decay on matrices. Per-head normalizer parameters are not matrices, despite 3 dims."""
    decay, rest = [], []
    for name, p in model.named_parameters():
        is_norm = name.rsplit(".", 1)[-1] in NORM_PARAMS
        (decay if p.dim() >= 2 and not is_norm else rest).append(p)
    groups = [dict(params=decay, weight_decay=cfg["wd"]), dict(params=rest, weight_decay=0.0)]
    return torch.optim.AdamW(groups, lr=lr, betas=tuple(cfg["betas"]), fused=device == "cuda")


def lr_at(step, lr, cfg):
    w, T = cfg["warmup"], cfg["steps"]
    if step < w:
        return lr * (step + 1) / w
    return lr * 0.5 * (1.0 + math.cos(math.pi * (step - w) / max(1, T - w)))


# ---------------------------------------------------------------- data

def prepare_data(cfg, data_dir):
    """Runs in the launcher only, before any worker starts."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_path(cfg, data_dir)
    if path.exists() and path.stat().st_size == cfg["data_size"]:
        return path
    if path.exists():
        log(f"{path} has the wrong size; preparing it again")
    if cfg["data"] == "mini":
        rng = np.random.default_rng(424242)
        letters = np.frombuffer(b"abcdefghijklmnopqrstuvwxyz", dtype=np.uint8)
        words = [letters[rng.integers(0, 26, rng.integers(2, 9))].tobytes() for _ in range(300)]
        probs = 1.0 / np.arange(1, 301)
        probs /= probs.sum()
        text = b" ".join(words[i] for i in rng.choice(300, cfg["data_size"], p=probs))
        atomic_bytes(path, text[:cfg["data_size"]])
        return path
    zpath = data_dir / f"enwik8.zip.{os.getpid()}.tmp"
    tmp = data_dir / f"enwik8.{os.getpid()}.tmp"
    error = None
    for attempt in range(3):
        try:
            log(f"downloading {ENWIK8_URL}, attempt {attempt + 1}")
            with urllib.request.urlopen(ENWIK8_URL, timeout=60) as r, open(zpath, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            with zipfile.ZipFile(zpath) as z, z.open("enwik8") as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            size = tmp.stat().st_size
            if size != ENWIK8_SIZE:
                raise IOError(f"unzipped size is {size}, expected {ENWIK8_SIZE}")
            os.replace(tmp, path)
            zpath.unlink()
            return path
        except Exception as e:  # network, bad zip or bad size: all worth a retry
            error = e
            log(f"download failed: {e}")
            time.sleep(5 * (attempt + 1))
    for p in (zpath, tmp):
        if p.exists():
            p.unlink()
    raise SystemExit(
        f"ERROR: could not get enwik8: {error}\n"
        f"Get {ENWIK8_URL} yourself, unzip it, and put the {ENWIK8_SIZE}-byte file at {path}.\n"
        f"Then run the same command again.")


def data_path(cfg, data_dir):
    return data_dir / ("enwik8" if cfg["data"] == "enwik8" else "e7_mini.bin")


def load_data(cfg, data_dir, device):
    raw = np.fromfile(data_path(cfg, data_dir), dtype=np.uint8)
    if raw.size != cfg["data_size"]:
        raise RuntimeError(f"data has {raw.size} bytes, expected {cfg['data_size']}")
    a, b = int(0.9 * raw.size), int(0.95 * raw.size)
    t = torch.from_numpy(raw)
    return dict(train=t[:a].to(device), val=t[a:b].to(device), test=t[b:].to(device))


def get_batch(data, B, T, gen, seed, step):
    """The batch is a function of (seed, step) alone, so arms share it and resume reproduces it."""
    gen.manual_seed(seed * 1_000_003 + step)
    starts = torch.randint(0, data.numel() - T, (B,), device=data.device, generator=gen)
    w = data[starts[:, None] + torch.arange(T + 1, device=data.device)].long()
    return w[:, :-1], w[:, 1:]


def eval_ends(n_text, n_windows, cfg, seed=EVAL_SEED):
    """One stratified random end per window. Input is text[e-N:e], last target is text[e]."""
    lo = max(max(cfg["Ns"]), cfg["probe_len"], cfg["ctx"] + cfg["keep"])
    if n_text - lo < n_windows:
        raise ValueError("text too short for the requested windows")
    edges = np.linspace(lo, n_text, n_windows + 1).astype(np.int64)
    return np.random.default_rng(seed).integers(edges[:-1], edges[1:])


def window_batch(data, ends, N, keep):
    """Input is the N bytes before each end; targets are those of the last `keep` positions."""
    x = data[ends[:, None] - N + torch.arange(N, device=ends.device)].long()
    y = data[ends[:, None] - keep + 1 + torch.arange(keep, device=ends.device)].long()
    return x, y


def trunc_positions(ends, keep):
    """Absolute target positions of the last `keep` positions of each window, flattened."""
    return (ends[:, None] - keep + 1 + torch.arange(keep, device=ends.device)).flatten()


def trunc_batch(data, pos, ctx):
    """For each target position, the `ctx` bytes before it, and the target."""
    return data[pos[:, None] - ctx + torch.arange(ctx, device=pos.device)].long(), data[pos].long()


def make_probes(test, ends, cfg, seed=PROBE_SEED):
    """test: uint8 array. Returns ({D: inputs}, targets), with the same string per window at all D."""
    L = cfg["probe_len"]
    E = L - 3
    strings = PROBE_ALPHABET[np.random.default_rng(seed).integers(0, len(PROBE_ALPHABET),
                                                                  (len(ends), 8))]
    base = np.stack([test[e - L:e] for e in ends])
    probes = {}
    for D in cfg["Ds"]:
        if not 12 <= D <= E:
            raise ValueError(f"distance {D} does not fit")
        x = base.copy()
        x[:, E - D:E - D + 8] = strings
        x[:, E - 4:E + 3] = strings[:, :7]
        probes[D] = x
    return probes, strings[:, 4:8].copy()


# ---------------------------------------------------------------- evaluation

def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def batched(n, bs0, fn, bs_state, key, beat=lambda: None):
    """Sum the tuples fn(i, j) over [0, n), halving and remembering the batch size on OOM."""
    bs, i, total = bs_state.get(key, bs0), 0, None
    while i < n:
        j, oom = min(n, i + bs), False
        try:
            r = fn(i, j)
        except RuntimeError as e:
            if not is_oom(e) or bs == 1:
                raise
            oom = True
        if oom:  # outside the except block, so the traceback no longer pins GPU tensors
            bs = max(1, bs // 2)
            bs_state[key] = bs
            log(f"out of memory at {key}; batch size is now {bs}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        total = r if total is None else tuple(a + b for a, b in zip(total, r))
        i = j
        beat()
    return total


def ce_sum(logits, y):
    return F.cross_entropy(logits.reshape(-1, 256).double(), y.reshape(-1), reduction="sum").item()


@torch.no_grad()
def eval_windows(model, data, ends, N, keep, bs0, bs_state, beat=lambda: None):
    """Bits per byte on the last `keep` positions of length-N windows."""
    def fn(i, j):
        x, y = window_batch(data, ends[i:j], N, keep)
        return (ce_sum(model(x, keep=keep), y),)
    total, = batched(len(ends), bs0, fn, bs_state, f"N={N}", beat)
    return total / (len(ends) * keep) / LOG2


@torch.no_grad()
def eval_truncated(model, data, ends, keep, ctx, bs0, bs_state, beat=lambda: None):
    """Bits per byte on the same positions, each seeing only the ctx bytes before it."""
    pos = trunc_positions(ends, keep)

    def fn(i, j):
        x, y = trunc_batch(data, pos[i:j], ctx)
        return (ce_sum(model(x, keep=1), y[:, None]),)
    total, = batched(pos.numel(), bs0, fn, bs_state, "T", beat)
    return total / pos.numel() / LOG2


@torch.no_grad()
def eval_copy(model, x, targets, device, bs0, bs_state, beat=lambda: None):
    """x, targets: CPU uint8 tensors. Returns teacher-forced byte accuracy and all-4 accuracy."""
    def fn(i, j):
        t = targets[i:j].to(device).long()
        correct = model(x[i:j].to(device).long(), keep=4).argmax(-1) == t
        return correct.sum().item(), correct.all(-1).sum().item()
    byte, all4 = batched(x.shape[0], bs0, fn, bs_state, f"probe{x.shape[1]}", beat)
    return dict(byte=byte / (4 * x.shape[0]), all4=all4 / x.shape[0])


def val_loss(model, cfg, rt, bs_state, beat=lambda: None):
    val = rt.data["val"]
    ends = torch.from_numpy(eval_ends(val.numel(), cfg["val_windows"], cfg, VAL_SEED)).to(rt.device)
    ctx = cfg["ctx"]
    return eval_windows(model, val, ends, ctx, ctx, max(1, cfg["eval_tokens"] // ctx), bs_state, beat)


def evaluate(model, arm, cfg, rt, partial, beat=lambda: None):
    """All test-set numbers for one model. Progress survives a crash through `partial`."""
    test, device = rt.data["test"], rt.device
    test_np = test.cpu().numpy()
    ends_np = eval_ends(test_np.size, cfg["eval_windows"], cfg)
    probes, targets = make_probes(test_np, ends_np, cfg)
    sha = hashlib.sha1(ends_np.tobytes() + targets.tobytes())
    for D in cfg["Ds"]:
        sha.update(probes[D].tobytes())
    ends = torch.from_numpy(ends_np).to(device)
    targets = torch.from_numpy(targets)
    tokens, keep, ctx = cfg["eval_tokens"], cfg["keep"], cfg["ctx"]
    res = json.loads(partial.read_text()) if partial.exists() else dict(readouts={}, seconds=0.0)
    bs_state = {}

    def metric(r, group, name, fn):
        slot = res["readouts"].setdefault(r, {}).setdefault(group, {})
        if str(name) in slot:
            return
        t0 = time.perf_counter()
        slot[str(name)] = fn()
        res["seconds"] += time.perf_counter() - t0
        atomic_text(partial, json.dumps(res))
        log(f"  {r} {group} {name}: {slot[str(name)]}")

    for r in READOUTS[arm]:
        model.set_readout(r)
        for N in cfg["Ns"]:
            if r == "gated" and N <= ctx:  # every row has n <= ctx: this is softmax, exactly
                metric(r, "L", N, lambda: res["readouts"]["softmax"]["L"][str(N)])
                continue
            metric(r, "L", N, lambda: eval_windows(model, test, ends, N, keep,
                                                   max(1, tokens // N), bs_state, beat))
        if r == "gated":
            metric(r, "T", "all", lambda: res["readouts"]["softmax"]["T"]["all"])
        else:
            metric(r, "T", "all", lambda: eval_truncated(model, test, ends, keep, ctx,
                                                         max(1, tokens // ctx), bs_state, beat))
        for D in cfg["Ds"]:
            metric(r, "copy", D, lambda: eval_copy(
                model, torch.from_numpy(probes[D]), targets, device,
                max(1, tokens // cfg["probe_len"]), bs_state, beat))
    model.set_readout(arm)
    return res["readouts"], res["seconds"], sha.hexdigest()


# ---------------------------------------------------------------- training

class Runtime:
    def __init__(self, device, data):
        self.device, self.data = device, data
        self.amp_dtype = None
        if device == "cuda":
            self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def amp(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast("cuda", dtype=self.amp_dtype)


def step_loss(fwd, x, y):
    return F.cross_entropy(fwd(x).reshape(-1, 256), y.reshape(-1))


def try_compile(model, x, y, rt):
    """Returns (callable, compiled?). Any failure, or a wrong loss, falls back to the plain model."""
    try:
        compiled = torch.compile(model)
        with torch.no_grad(), rt.amp():
            a, b = step_loss(compiled, x, y).item(), step_loss(model, x, y).item()
        if not abs(a - b) <= 1e-2:
            raise RuntimeError(f"compiled loss {a} differs from plain loss {b}")
        log("torch.compile: active")
        return compiled, True
    except Exception as e:
        log(f"torch.compile: failed, using the plain model. {type(e).__name__}: {e}")
        return model, False


def backward(fwd, x, y, rt, scaler):
    with rt.amp():
        loss = step_loss(fwd, x, y)
    scaler.scale(loss).backward()
    return loss


def save_checkpoint(path, model, opt, scaler, state):
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                        **state), f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def train_loop(job, cfg, rt, ckpt, beat=lambda: None, stop_after=None, crash_marker=None):
    """Train or resume. stop_after simulates a crash: it returns without saving anything extra."""
    device, seed = rt.device, job["seed"]
    torch.manual_seed(seed)
    model = E7Model(cfg, job["arm"]).to(device)
    opt = make_optimizer(model, job["lr"], cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=rt.amp_dtype == torch.float16)
    state = dict(step=0, nonfinite=0, times=[], val_log=[], resumed_from=0)
    if ckpt.exists():
        try:
            ck = torch.load(ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            scaler.load_state_dict(ck["scaler"])
            state = {k: ck[k] for k in state}
            state["resumed_from"] = state["step"]
            log(f"{job['id']}: resumed at step {state['step']}")
        except Exception as e:  # replaced atomically, so this needs disk damage; costs only time
            log(f"{job['id']}: unreadable checkpoint, starting over. {e}")
            torch.manual_seed(seed)
            model = E7Model(cfg, job["arm"]).to(device)
            opt = make_optimizer(model, job["lr"], cfg, device)
            state = dict(step=0, nonfinite=0, times=[], val_log=[], resumed_from=0)
    train, gen, bs_state = rt.data["train"], torch.Generator(device=device), {}
    B, T, steps = cfg["batch"], cfg["ctx"], cfg["steps"]
    fwd, compiled = model, False
    want_compile = device == "cuda" and (cfg["compile_seeds"] is None or seed in cfg["compile_seeds"])
    if want_compile and state["step"] < steps:
        x, y = get_batch(train, B, T, gen, seed, state["step"])
        fwd, compiled = try_compile(model, x, y, rt)
    start = state["step"]
    for step in range(start, steps):
        t0 = time.perf_counter()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, job["lr"], cfg)
        x, y = get_batch(train, B, T, gen, seed, step)
        opt.zero_grad(set_to_none=True)
        try:
            loss = backward(fwd, x, y, rt, scaler)
        except Exception as e:
            if fwd is model or is_oom(e):
                raise
            log(f"compiled step failed, using the plain model. {type(e).__name__}: {e}")
            fwd, compiled = model, False
            opt.zero_grad(set_to_none=True)
            loss = backward(fwd, x, y, rt, scaler)
        scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
        if math.isfinite(norm.item()) and math.isfinite(loss.item()):
            scaler.step(opt)
        else:
            state["nonfinite"] += 1
        scaler.update()
        state["times"].append(time.perf_counter() - t0)
        done = step + 1
        state["step"] = done
        beat()
        if done % 500 == 0:
            log(f"{job['id']}: step {done}/{steps} loss {loss.item():.4f}")
        if done % cfg["val_every"] == 0 and done < steps:
            state["val_log"].append([done, val_loss(model, cfg, rt, bs_state, beat)])
            log(f"{job['id']}: step {done} val {state['val_log'][-1][1]:.4f}")
        if done % cfg["ckpt_every"] == 0 and done < steps:
            save_checkpoint(ckpt, model, opt, scaler, state)
        crash = cfg.get("inject_crash")
        if crash and crash_marker and crash["job"] == job["id"] and crash["step"] == done \
                and not crash_marker.exists():
            crash_marker.touch()
            log("injected crash for the resume test")
            os._exit(7)
        if stop_after == done:
            return model, state, compiled
    if start < steps:
        save_checkpoint(ckpt, model, opt, scaler, state)  # evaluation never repeats training
    return model, state, compiled


def env_info(rt):
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
    except Exception:
        commit = None
    return dict(git=commit, torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0) if rt.device == "cuda" else "cpu",
                amp=str(rt.amp_dtype), code_sha=hashlib.sha1(Path(__file__).read_bytes()).hexdigest())


def run_job(job, out, cfg, rt, beat=lambda: None):
    jid, t0 = job["id"], time.time()
    log(f"{jid}: start, lr {job['lr']}")
    model, state, compiled = train_loop(job, cfg, rt, paths(out)["ckpt"] / f"{jid}.pt", beat,
                                        crash_marker=paths(out)["queue"] / "crash_injected")
    times = state["times"]
    median = statistics.median(times[min(20, len(times) // 2):]) if times else float("nan")
    res = dict(job, params=sum(p.numel() for p in model.parameters()), step_ms=median * 1e3,
               tokens_per_s=cfg["batch"] * cfg["ctx"] / median if times else float("nan"),
               nonfinite=state["nonfinite"], val_log=state["val_log"], learned=model.learned(),
               compiled=compiled, resumed_from=state["resumed_from"], **env_info(rt))
    res["val_loss_final"] = val_loss(model, cfg, rt, {}, beat)
    if job["kind"] == "final":
        partial = paths(out)["ckpt"] / f"{jid}.eval.json"
        res["eval"], res["eval_seconds"], res["eval_hash"] = evaluate(model, job["arm"], cfg, rt,
                                                                      partial, beat)
        res["selection"] = json.loads(paths(out)["selection"].read_text())
    res["wall_seconds"] = time.time() - t0
    log(f"{jid}: done, val {res['val_loss_final']:.4f}, step {res['step_ms']:.1f} ms")
    return res


# ---------------------------------------------------------------- queue

def paths(out):
    q = out / "queue"
    return dict(jobs=out / "jobs", ckpt=out / "ckpt", logs=out / "logs", queue=q, claims=q / "claims",
                attempts=q / "attempts", failed=q / "failed", errors=q / "errors",
                selection=out / "selection.json", config=out / "config.json")


def make_dirs(out):
    p = paths(out)
    for key in ("jobs", "ckpt", "logs", "claims", "attempts", "failed", "errors"):
        p[key].mkdir(parents=True, exist_ok=True)


def atomic_bytes(path, data):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_text(path, text):
    atomic_bytes(path, text.encode())


def tune_jobs(cfg):
    return [dict(id=f"tune_{arm}_lr{lr}_s{cfg['tune_seed']}", kind="tune", arm=arm,
                 seed=cfg["tune_seed"], lr=lr) for arm in ARMS for lr in cfg["lrs"]]


def final_jobs(cfg, selection):
    return [dict(id=f"final_{arm}_s{seed}", kind="final", arm=arm, seed=seed,
                 lr=selection[arm]["lr"]) for arm in ARMS for seed in cfg["final_seeds"]]


def is_done(out, jid):
    return (paths(out)["jobs"] / f"{jid}.json").exists()


def is_failed(out, jid):
    return (paths(out)["failed"] / f"{jid}.json").exists()


def select_lrs(results):
    """Lowest final validation loss per arm; ties go to the lower rate. NaN never wins."""
    selection = {}
    for r in results:
        selection.setdefault(r["arm"], dict(table={}))["table"][str(r["lr"])] = r["val_loss_final"]
    for arm, s in selection.items():
        lrs = sorted(s["table"], key=float)
        finite = [lr for lr in lrs if math.isfinite(s["table"][lr])]
        best = min(finite, key=lambda lr: (s["table"][lr], float(lr))) if finite else lrs[0]
        s["lr"] = float(best)
        s["edge"] = best in (lrs[0], lrs[-1])
    return selection


def ensure_selection(out, cfg):
    """Written once all tuning results exist. Deterministic, so a repeated write is harmless."""
    p = paths(out)
    if p["selection"].exists():
        return
    jobs = tune_jobs(cfg)
    if all(is_done(out, j["id"]) for j in jobs):
        results = [json.loads((p["jobs"] / f"{j['id']}.json").read_text()) for j in jobs]
        atomic_text(p["selection"], json.dumps(select_lrs(results), indent=1, sort_keys=True))


def visible_jobs(out, cfg):
    """Tuning first. Final jobs do not exist until the selection does: that is the barrier."""
    p = paths(out)["selection"]
    return tune_jobs(cfg) + (final_jobs(cfg, json.loads(p.read_text())) if p.exists() else [])


def settled(out, cfg):
    """True when nothing is left that any worker could ever run."""
    tune = tune_jobs(cfg)
    if not all(is_done(out, j["id"]) or is_failed(out, j["id"]) for j in tune):
        return False
    if not all(is_done(out, j["id"]) for j in tune):
        return True  # a tuning job finally failed, so no selection and no final jobs
    ensure_selection(out, cfg)
    return all(is_done(out, j["id"]) or is_failed(out, j["id"]) for j in visible_jobs(out, cfg))


def proc_start(pid):
    """Start time of a live process, None if dead or a zombie. Guards against pid reuse."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None
    return None if fields[0] == "Z" else fields[19]


def try_claim(out, jid, slot):
    path = paths(out)["claims"] / jid
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        pid = os.getpid()
        json.dump(dict(pid=pid, start=proc_start(pid), host=socket.gethostname(), slot=slot), f)
    return True


def release(out, jid):
    try:
        (paths(out)["claims"] / jid).unlink()
    except FileNotFoundError:
        pass


def claim_status(path, stale_sec):
    """Returns (status, pid), status being live, dead or hung. Reads only; never deletes."""
    try:
        age = time.time() - path.stat().st_mtime
        info = json.loads(path.read_text())
    except FileNotFoundError:
        return "gone", None
    except (OSError, ValueError):  # created, not yet written, or the writer died in between
        return ("dead" if age > 60 else "live"), None
    pid = info.get("pid")
    if info.get("host") != socket.gethostname() or proc_start(pid) != info.get("start"):
        return "dead", pid
    return ("hung" if age > stale_sec else "live"), pid


def reclaim_stale(out, stale_sec):
    """Launcher only, so there is a single remover. Frees claims of dead workers, kills hung ones."""
    freed = []
    for path in sorted(paths(out)["claims"].iterdir()):
        status, pid = claim_status(path, stale_sec)
        if status == "hung":
            log(f"worker {pid} made no progress on {path.name} for {stale_sec}s; killing it")
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        if status in ("dead", "hung"):
            release(out, path.name)
            freed.append(path.name)
    return freed


def cleanup_job(out, jid):
    p = paths(out)["ckpt"]
    for path in (p / f"{jid}.pt", p / f"{jid}.eval.json"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def read_attempts(out, jid):
    try:
        return int((paths(out)["attempts"] / jid).read_text())
    except (FileNotFoundError, ValueError):
        return 0


def mark_failed(out, jid, why):
    atomic_text(paths(out)["failed"] / f"{jid}.json", json.dumps(dict(job=jid, why=why)))
    log(f"{jid}: FAILED after {MAX_ATTEMPTS} attempts")


def claim_next(out, cfg, slot):
    """Atomically claims the next runnable job, counting it as an attempt; None if there is none."""
    for job in visible_jobs(out, cfg):
        jid = job["id"]
        if is_done(out, jid) or is_failed(out, jid) or not try_claim(out, jid, slot):
            continue
        if is_done(out, jid):  # finished between the check and the claim, or crashed before cleanup
            cleanup_job(out, jid)
            release(out, jid)
            continue
        if is_failed(out, jid):
            release(out, jid)
            continue
        attempts = read_attempts(out, jid)
        if attempts >= MAX_ATTEMPTS:  # the previous holders crashed without recording anything
            mark_failed(out, jid, "worker crashed on every attempt; see logs/")
            release(out, jid)
            continue
        atomic_text(paths(out)["attempts"] / jid, str(attempts + 1))
        return job
    return None


class Beat:
    """Called from the main thread, so a stuck GPU call stops the heartbeat, as it should."""

    def __init__(self, path, ppid):
        self.path, self.ppid, self.last = path, ppid, time.time()

    def __call__(self):
        now = time.time()
        if now - self.last < 20.0:
            return
        self.last = now
        if os.getppid() != self.ppid:  # the launcher is gone: a rerun must find a free GPU
            log("launcher is gone; exiting")
            os._exit(9)
        try:
            os.utime(self.path)
        except OSError:
            pass


def worker_loop(out, cfg, slot, job_fn, poll=None):
    """job_fn(job, beat) returns the result dict. Returns the process exit code. After a job error
    the worker exits, so the launcher's respawn delay lets another GPU take the retry."""
    poll = cfg["poll"] if poll is None else poll
    ppid = os.getppid()
    while True:
        ensure_selection(out, cfg)
        job = claim_next(out, cfg, slot)
        if job is None:
            if settled(out, cfg):
                return 0
            time.sleep(poll)
            continue
        jid = job["id"]
        try:
            result = job_fn(job, Beat(paths(out)["claims"] / jid, ppid))
        except Exception:
            text = traceback.format_exc()
            log(f"{jid}: error\n{text}")
            attempts = read_attempts(out, jid)
            atomic_text(paths(out)["errors"] / f"{jid}.{attempts}.txt", text)
            if attempts >= MAX_ATTEMPTS:
                mark_failed(out, jid, text[-2000:])
            release(out, jid)
            return 1
        atomic_text(paths(out)["jobs"] / f"{jid}.json", json.dumps(result, indent=1))
        cleanup_job(out, jid)
        release(out, jid)


# ---------------------------------------------------------------- processes

def worker_main(args):
    out = Path(args.out)
    cfg = json.loads(paths(out)["config"].read_text())
    device = args.device
    torch.set_num_threads(2 if device == "cuda" else max(1, (os.cpu_count() or 2) // args.n_slots))
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    rt = Runtime(device, load_data(cfg, Path(args.data_dir), device))
    log(f"worker {args.slot}: pid {os.getpid()}, device {device}, amp {rt.amp_dtype}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if rt.amp_dtype == torch.float16:
        log("WARNING: bf16 is unsupported here; using float16 with GradScaler")
    code = worker_loop(out, cfg, args.slot, lambda job, beat: run_job(job, out, cfg, rt, beat))
    log(f"worker {args.slot}: exit {code}")
    return code


def preflight():
    log(f"python {sys.version.split()[0]}, torch {torch.__version__}, CUDA build {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("ERROR: torch cannot see a GPU. Check nvidia-smi, and that torch is a CUDA build.\n"
              "To validate on a CPU machine, use: bash experiments/E7_gpu/run.sh --mini",
              file=sys.stderr)
        return 1
    count = torch.cuda.device_count()
    log(f"visible GPUs: {count}")
    for i in range(count):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 2**30
        try:
            d = torch.device(f"cuda:{i}")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                x = torch.randn(4, 8, 8, device=d)
                x = (x @ x.transpose(-1, -2)).float().sort(-1).values
            if not torch.isfinite(x).all():
                raise RuntimeError("non-finite result")
        except Exception as e:
            print(f"ERROR: GPU {i} ({name}) failed a test computation: {e}\nIf the message mentions "
                  "a kernel image or an unsupported architecture, this torch build is too old for "
                  "this GPU; upgrade torch with: pip install --upgrade torch", file=sys.stderr)
            return 1
        log(f"GPU {i}: {name}, {mem:.0f} GiB, bf16 {torch.cuda.is_bf16_supported()}, test ok")
    free = shutil.disk_usage(REPO).free / 2**30
    if free < 3.0:
        print(f"ERROR: only {free:.1f} GiB free at {REPO}; at least 3 GiB is needed.", file=sys.stderr)
        return 1
    return 0


def gpu_ids():
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [i.strip() for i in env.split(",") if i.strip()]
    return [str(i) for i in range(torch.cuda.device_count())]


def print_projection(out, cfg, n_slots):
    """Project the full run from the measured smoke jobs. Writes projection.txt too."""
    full = make_config("full")
    results = [json.loads(p.read_text()) for p in paths(out)["jobs"].glob("*.json")]
    lines = [f"Measured projection of the full run on this machine, {n_slots} workers:"]
    tune_total, final_total, longest = 0.0, 0.0, 0.0
    for arm in ARMS:
        rows = [r for r in results if r["arm"] == arm]
        step = min(r["step_ms"] for r in rows) / 1e3
        evals = [r["eval_seconds"] for r in rows if r["kind"] == "final"]
        ev = max(evals) * full["eval_windows"] / cfg["eval_windows"]
        train = full["steps"] * step
        tune_total += len(full["lrs"]) * train
        final_total += len(full["final_seeds"]) * (train + ev)
        longest = max(longest, train + ev)
        lines.append(f"  {arm:16s} step {step * 1e3:7.1f} ms, train {train / 3600:6.2f} h, "
                     f"final evaluation {ev / 3600:6.2f} h per job")
    wall = max(tune_total / n_slots, longest) + max(final_total / n_slots, longest)
    lines.append(f"  total {(tune_total + final_total) / 3600:.1f} GPU-hours; about "
                 f"{wall / 3600:.1f} h on {n_slots} workers")
    atomic_text(out / "projection.txt", "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def verify_small_run(out, cfg):
    """Smoke and mini only: the injected crash must have been resumed from a checkpoint."""
    jid = cfg["inject_crash"]["job"]
    result = json.loads((paths(out)["jobs"] / f"{jid}.json").read_text())
    if not result["resumed_from"]:
        marker = paths(out)["queue"] / "crash_injected"
        raise SystemExit(f"ERROR: {jid} did not resume from a checkpoint (marker {marker.exists()})")
    log(f"resume verified: {jid} continued from step {result['resumed_from']}")


def launch(args, mode):
    import fcntl
    cfg = make_config(mode)
    out = Path(args.out or REPO / "runs" / {"full": "e7", "smoke": "e7_smoke", "mini": "e7_mini"}[mode])
    out = out.resolve()
    data_dir = Path(args.data_dir or REPO / "data").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(out.parent / f".{out.name}.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"ERROR: another launcher is already using {out}", file=sys.stderr)
        return 1
    sha = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()
    stored_cfg = out / "config.json"
    if stored_cfg.exists():
        same = json.loads(stored_cfg.read_text()) == json.loads(json.dumps(cfg))
        code_file = out / "code_sha.txt"
        if mode != "full" and (not same or not code_file.exists() or code_file.read_text() != sha):
            log(f"{mode} output is from other code or settings; removing {out}")
            shutil.rmtree(out)
        elif not same:
            print(f"ERROR: {out} holds a run with a different configuration. Move it away, or "
                  f"pass --out with another directory.", file=sys.stderr)
            return 1
    make_dirs(out)
    p = paths(out)
    atomic_text(p["config"], json.dumps(cfg, indent=1))
    atomic_text(out / "code_sha.txt", sha)
    prepare_data(cfg, data_dir)

    # A rerun gives failed jobs a new chance, and frees whatever a dead run left behind.
    for key in ("failed", "attempts"):
        for path in p[key].iterdir():
            path.unlink()
    reclaim_stale(out, args.stale_sec)

    device = args.device or ("cpu" if mode == "mini" else "cuda")
    if device == "cuda":
        ids = gpu_ids()
        if not ids:
            print("ERROR: no GPU is visible. Use --device cpu for a CPU run.", file=sys.stderr)
            return 1
    else:
        ids = [""]
    slots = [dict(index=i, gpu=gpu, proc=None, crashes=0, start_at=0.0, finished=False, gave_up=False)
             for i, gpu in enumerate(g for g in ids for _ in range(args.per_gpu))]
    log(f"run {cfg['name']}: output {out}, {len(slots)} workers on GPUs {ids}")

    def spawn(slot):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=slot["gpu"], PYTHONUNBUFFERED="1",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        logfile = open(p["logs"] / f"worker_{slot['index']}.log", "a")
        slot["proc"] = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", "--slot", str(slot["index"]),
             "--n-slots", str(len(slots)), "--out", str(out), "--data-dir", str(data_dir),
             "--device", device], env=env, stdout=logfile, stderr=subprocess.STDOUT,
            start_new_session=True)
        logfile.close()

    def stop(*_):
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, stop)

    last_report = 0.0
    try:
        while True:
            now = time.time()
            for slot in slots:
                if slot["finished"] or slot["gave_up"]:
                    continue
                if slot["proc"] is None:
                    if now >= slot["start_at"]:
                        spawn(slot)
                    continue
                code = slot["proc"].poll()
                if code is None:
                    continue
                slot["proc"] = None
                if code == 0:
                    slot["finished"] = True
                    continue
                slot["crashes"] += 1
                tail = f"see {p['logs'] / ('worker_%d.log' % slot['index'])}"
                log(f"worker {slot['index']} (GPU {slot['gpu']}) exited with code {code}, "
                    f"crash {slot['crashes']}; {tail}")
                if slot["crashes"] >= MAX_SLOT_CRASHES:
                    slot["gave_up"] = True
                    log(f"worker {slot['index']} crashed too often; other workers continue")
                else:
                    slot["start_at"] = now + cfg["respawn_delay"]
            reclaim_stale(out, args.stale_sec)
            if all(s["finished"] or s["gave_up"] for s in slots):
                break
            if now - last_report > 60.0:
                last_report = now
                names = visible_jobs(out, cfg)
                done = sum(is_done(out, j["id"]) for j in names)
                log(f"progress: {done}/{len(names)} visible jobs done, of "
                    f"{len(cfg['lrs']) * len(ARMS) + len(cfg['final_seeds']) * len(ARMS)}; "
                    f"running {sorted(x.name for x in p['claims'].iterdir())}")
            time.sleep(1.0)
    finally:
        for slot in slots:
            if slot["proc"] is not None and slot["proc"].poll() is None:
                slot["proc"].terminate()

    ensure_selection(out, cfg)
    failed = sorted(x.stem for x in p["failed"].glob("*.json"))
    expected = visible_jobs(out, cfg)
    missing = [j["id"] for j in expected if not is_done(out, j["id"])]
    if not p["selection"].exists():
        missing.append("all final jobs: tuning did not finish")
    if failed or missing:
        log(f"FAILED jobs: {failed}. Not finished: {missing}. Tracebacks: {p['errors']}. "
            f"Logs: {p['logs']}. Running the same command again resumes and retries them.")
        return 1
    log(f"all {len(expected)} jobs done: {p['jobs']}")
    if mode != "full":
        verify_small_run(out, cfg)
    if mode == "smoke":
        print_projection(out, cfg, len(slots))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true", help="few steps on real data, all code paths")
    ap.add_argument("--mini", action="store_true", help="small model, synthetic data, CPU")
    ap.add_argument("--preflight", action="store_true", help="check the GPUs and exit")
    ap.add_argument("--device", choices=("cuda", "cpu"), help="default cuda; cpu for --mini")
    ap.add_argument("--per-gpu", type=int, default=1, help="workers per GPU")
    ap.add_argument("--out", help="output directory; default runs/e7 and similar")
    ap.add_argument("--data-dir", help="default data/ at the repo root")
    ap.add_argument("--stale-sec", type=float, default=1800.0,
                    help="kill a live worker whose claim saw no progress for this long")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--slot", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--n-slots", type=int, default=1, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.per_gpu < 1:
        ap.error("--per-gpu must be at least 1")
    if args.preflight:
        return preflight()
    if args.worker:
        return worker_main(args)
    if args.smoke and args.mini:
        ap.error("choose one of --smoke and --mini")
    return launch(args, "smoke" if args.smoke else "mini" if args.mini else "full")


if __name__ == "__main__":
    sys.exit(main())
