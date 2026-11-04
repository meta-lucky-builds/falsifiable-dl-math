"""Tests for experiments/E7_gpu. Run with `pytest tests/test_e7.py` or `python tests/test_e7.py`.
The __main__ runner needs no pytest. CPU only; the mini run makes up most of the time."""
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "E7_gpu"))
sys.path.insert(0, str(ROOT / "src"))
import analyze  # noqa: E402
import e7  # noqa: E402
from mai.attention import GAMMA_BOUND, MultiHeadAttention  # noqa: E402

CFG = e7.make_config("mini")


def build(arm, seed=0, cfg=CFG):
    torch.manual_seed(seed)
    return e7.E7Model(cfg, arm).eval()


def all_settings():
    for arm in e7.ARMS:
        for readout in e7.READOUTS[arm]:
            yield arm, readout


def test_causal_masking_all_arms_and_readouts():
    x = torch.randint(0, 256, (2, 150))
    for arm, readout in all_settings():
        model = build(arm)
        model.set_readout(readout)
        with torch.no_grad():
            base = model(x)
            for t in (0, 31, 32, 100):  # includes rows on both sides of a block boundary
                y = x.clone()
                y[:, t + 1:] = (y[:, t + 1:] + 7) % 256
                assert torch.allclose(model(y)[:, :t + 1], base[:, :t + 1], atol=2e-5), (arm, readout, t)
                assert not torch.allclose(model(y)[:, t + 1:], base[:, t + 1:], atol=2e-5), (arm, readout, t)


def test_keep_rows_match_full_forward():
    x = torch.randint(0, 256, (2, 150))
    for arm in e7.ARMS:
        model = build(arm)
        with torch.no_grad():
            full = model(x)
            for keep in (1, 4, 64, 150):
                assert torch.allclose(model(x, keep=keep), full[:, -keep:], atol=2e-5), (arm, keep)


def test_rope_period_and_tables():
    freqs = torch.tensor([2 * math.pi / 16, 2 * math.pi / 8], dtype=torch.float64)
    cos, sin = e7.rope_tables(100, 4, "cpu", freqs)
    x = torch.randn(3, 2, 100, 4)
    r = e7.rope(x[:, :, :1].expand_as(x), cos, sin)
    assert torch.allclose(r[:, :, 0], x[:, :, 0])
    assert torch.allclose(r[:, :, 16], x[:, :, 0], atol=1e-5)  # common period of both pairs
    assert torch.allclose(r[:, :, 3 * 16], x[:, :, 0], atol=1e-5)
    assert not torch.allclose(r[:, :, 5], x[:, :, 0], atol=1e-3)
    # The default table: position 0 is the identity, and pair i rotates at 10000^(-2i/dh).
    cos, sin = e7.rope_tables(4000, 16, "cpu")
    assert torch.allclose(cos[0], torch.ones(8)) and torch.allclose(sin[0], torch.zeros(8))
    assert abs(cos[3000, 2].item() - math.cos(3000 * 10000 ** (-4 / 16))) < 1e-5
    # A rotation preserves the norm of a vector at any position.
    y = torch.randn(5, 4000, 16)
    assert torch.allclose(e7.rope(y, cos, sin).norm(dim=-1), y.norm(dim=-1), atol=1e-4)


def test_rope_score_depends_only_on_position_difference_past_training_length():
    dh = 16
    cos, sin = e7.rope_tables(4000, dh, "cpu")
    q, k = torch.randn(dh, dtype=torch.float32), torch.randn(dh, dtype=torch.float32)

    def score(p, r):
        return (e7.rope(q, cos[p], sin[p]) * e7.rope(k, cos[r], sin[r])).sum().item()

    for shift in (1, 300, 2000, 3500):
        assert abs(score(100 + shift, 40 + shift) - score(100, 40)) < 1e-4, shift
    assert abs(score(100, 40) - score(100, 41)) > 1e-5


def test_model_runs_past_training_length():
    model = build("entmax15")
    with torch.no_grad():
        assert torch.isfinite(model(torch.randint(0, 256, (1, 700)), keep=4)).all()


def _perturbed(arm, d=16, h=2, train_n=256):
    torch.manual_seed(5)
    attn = e7.Attention(d, h, arm, train_n)
    for name, p in attn.named_parameters(recurse=False):
        p.data.add_(0.2 * torch.randn_like(p))
    ref = MultiHeadAttention(d, h, normalizer=arm, train_n=train_n)
    ref.load_state_dict(attn.state_dict())
    return attn, ref


def test_normalizers_match_attention_py():
    assert GAMMA_BOUND == e7.GAMMA_BOUND
    for arm in e7.ARMS:
        init, ref0 = e7.Attention(16, 2, arm, 256), MultiHeadAttention(16, 2, normalizer=arm, train_n=256)
        for name, p in init.named_parameters(recurse=False):  # same initial normalizer values
            assert torch.allclose(p, dict(ref0.named_parameters(recurse=False))[name]), (arm, name)
        attn, ref = _perturbed(arm)
        z = torch.randn(2, 2, 40, 40) * 8.0
        mask = torch.ones(40, 40, dtype=torch.bool).tril()
        n = torch.arange(1, 41, dtype=torch.float32)[:, None]
        with torch.no_grad():
            got = attn.normalize(z, mask, n)
            want = ref.normalize(z, mask[None, None])
        assert torch.allclose(got, want, atol=1e-6), arm
        assert torch.allclose(got.sum(-1), torch.ones(2, 2, 40), atol=1e-4), arm


def test_block_path_equals_full_matrix_through_attention_py():
    T, d, h = 100, 16, 2  # block size is 32, so this crosses several blocks
    x = torch.randn(2, T, d)
    cos, sin = e7.rope_tables(T, d // h, "cpu")
    mask = torch.ones(T, T, dtype=torch.bool).tril()[None, None]
    for arm in e7.ARMS:
        attn, ref = _perturbed(arm)
        with torch.no_grad():
            q = e7.rope(ref.q(x).view(2, T, h, d // h).transpose(1, 2), cos, sin)
            k = e7.rope(ref.k(x).view(2, T, h, d // h).transpose(1, 2), cos, sin)
            v = ref.v(x).view(2, T, h, d // h).transpose(1, 2)
            w = ref.normalize(q @ k.transpose(-1, -2), mask)
            want = ref.out((w @ v).transpose(1, 2).reshape(2, T, d))
            got = attn(x, cos, sin)
            assert torch.allclose(got, want, atol=2e-5), arm
            assert torch.allclose(attn(x, cos, sin, 90), want[:, 90:], atol=2e-5), arm


def test_readout_swap_equals_freshly_built_model():
    x = torch.randint(0, 256, (2, 120))
    for readout in ("entmax15", "ssmax"):
        trained = build("softmax", seed=3)
        with torch.no_grad():
            for p in trained.parameters():  # as if trained; a shared state, not the shared init
                p.add_(0.01 * torch.randn_like(p))
        trained.set_readout(readout)
        fresh = build(readout, seed=9)
        info = fresh.load_state_dict(trained.state_dict(), strict=False)
        assert not info.unexpected_keys and all(k.endswith(".attn.s") for k in info.missing_keys)
        with torch.no_grad():
            assert torch.allclose(trained(x), fresh(x), atol=1e-5), readout


def _gated_and(other, seed=4, cfg=CFG):
    gated = build("softmax", seed, cfg)
    gated.set_readout("gated")
    return gated, build(other, seed, cfg)


def test_gated_equals_softmax_exactly_at_training_length():
    gated, soft = _gated_and("softmax")
    ctx = CFG["ctx"]
    with torch.no_grad():
        for T, keep in ((ctx, None), (ctx, 8), (ctx - 7, None)):
            x = torch.randint(0, 256, (3, T))
            assert torch.equal(gated(x, keep=keep), soft(x, keep=keep)), (T, keep)


def test_gated_equals_softmax_on_short_rows_at_longer_n():
    gated, soft = _gated_and("softmax")
    ctx = CFG["ctx"]
    for T in (2 * ctx, 128, 256):
        x = torch.randint(0, 256, (2, T))
        with torch.no_grad():
            a, b = gated(x), soft(x)
        assert torch.allclose(a[:, :ctx], b[:, :ctx], atol=1e-6), T
        assert not torch.allclose(a[:, ctx:], b[:, ctx:], atol=1e-5), T


def test_gated_equals_entmax15_on_long_rows():
    # One layer, so that the input to the attention is the same for both readouts. With more layers
    # long rows also read the softmax-derived values of short rows, as the gate intends.
    cfg = dict(CFG, n_layer=1)
    gated, ent = _gated_and("entmax15", cfg=cfg)
    ctx = CFG["ctx"]
    x = torch.randint(0, 256, (2, 3 * ctx))
    with torch.no_grad():
        a, b = gated(x), ent(x)
    assert torch.allclose(a[:, ctx:], b[:, ctx:], atol=1e-6)
    assert not torch.allclose(a[:, :ctx], b[:, :ctx], atol=1e-5)


def test_gate_cut_inside_a_block():
    # The normalizer itself, with rows 21 to 44 valid counts around a gate at 32.
    attn = e7.Attention(16, 2, "softmax", 32)
    z = torch.randn(2, 2, 24, 44) * 8.0
    mask = torch.ones(24, 44, dtype=torch.bool).tril(20)
    n = torch.arange(21, 45, dtype=torch.float32)[:, None]
    attn.readout = "softmax"
    soft = attn.normalize(z, mask, n)
    attn.readout = "entmax15"
    ent = attn.normalize(z, mask, n)
    attn.readout = "gated"
    got = attn.normalize(z, mask, n)
    m = 12  # rows with n = 21 to 32
    assert torch.allclose(got[:, :, :m], soft[:, :, :m], atol=1e-6)
    assert torch.allclose(got[:, :, m:], ent[:, :, m:], atol=1e-6)
    # keep starts a block on one side of the gate and crosses it.
    gated, _ = _gated_and("softmax")
    x = torch.randint(0, 256, (2, 2 * CFG["ctx"]))
    with torch.no_grad():
        assert torch.allclose(gated(x, keep=40), gated(x)[:, -40:], atol=2e-5)


def test_gated_is_only_for_softmax_trained_models():
    for arm in ("ssmax", "entmax15_scaled"):
        try:
            build(arm).set_readout("gated")
        except ValueError:
            continue
        raise AssertionError(arm)


def test_shared_initial_weights_across_arms():
    base = build("softmax", seed=11).state_dict()
    for arm in e7.ARMS[1:]:
        other = build(arm, seed=11).state_dict()
        for name, value in base.items():
            assert torch.equal(value, other[name]), (arm, name)
    assert not torch.equal(build("softmax", seed=12).tok.weight, base["tok.weight"])


def _text(n=20000):
    return torch.from_numpy(np.random.default_rng(1).integers(0, 256, n, dtype=np.uint8))


def test_truncated_and_full_windows_use_the_same_targets():
    text = _text()
    ends = torch.from_numpy(e7.eval_ends(text.numel(), CFG["eval_windows"], CFG))
    keep, ctx = CFG["keep"], CFG["ctx"]
    pos = e7.trunc_positions(ends, keep)
    x_t, y_t = e7.trunc_batch(text, pos, ctx)
    assert x_t.shape == (ends.numel() * keep, ctx)
    for N in CFG["Ns"]:
        x, y = e7.window_batch(text, ends, N, keep)
        assert torch.equal(y.flatten(), y_t), N
        assert x.shape == (ends.numel(), N)
        # The last row of a truncated context is the input byte at that position of the N window,
        # and the input of the N window at the last position is one before its last target.
        assert torch.equal(x[:, -keep:].flatten(), x_t[:, -1]), N
        assert torch.equal(x[:, -1], text[ends - 1].long()), N
    x, _ = e7.window_batch(text, ends, max(CFG["Ns"]), keep)
    assert torch.equal(x[:, -ctx:], x_t[keep - 1::keep]), "the last context is the window's tail"


def test_eval_windows_and_probes_are_fixed_and_well_formed():
    text = _text().numpy()
    a = e7.eval_ends(text.size, CFG["eval_windows"], CFG)
    b = e7.eval_ends(text.size, CFG["eval_windows"], CFG)
    assert np.array_equal(a, b) and a.min() >= CFG["probe_len"] and a.max() < text.size
    p1, t1 = e7.make_probes(text, a, CFG)
    p2, t2 = e7.make_probes(text, a, CFG)
    assert np.array_equal(t1, t2)
    L, E = CFG["probe_len"], CFG["probe_len"] - 3
    for D in CFG["Ds"]:
        assert np.array_equal(p1[D], p2[D])
        x = p1[D]
        assert x.shape == (len(a), L)
        assert np.array_equal(x[:, E - D:E - D + 4], x[:, E - 4:E]), "the first 4 bytes are repeated"
        assert np.array_equal(x[:, E - D + 4:E - D + 7], x[:, E:E + 3]), "teacher forcing"
        assert np.array_equal(x[:, E - D + 4:E - D + 8], t1), "the targets are the string's last 4"
        assert np.isin(t1, e7.PROBE_ALPHABET).all()
        untouched = np.ones(L, dtype=bool)
        untouched[E - D:E - D + 8] = False
        untouched[E - 4:] = False
        real = np.stack([text[e - L:e] for e in a])
        assert np.array_equal(x[:, untouched], real[:, untouched]), "the rest is real text"


def test_oom_halves_the_batch_and_counts_everything_once():
    seen = []

    def fn(i, j):
        if j - i > 4:
            raise RuntimeError("CUDA out of memory. Tried to allocate")
        seen.extend(range(i, j))
        return (j - i,)
    state = {}
    assert e7.batched(23, 16, fn, state, "N") == (23,)
    assert seen == list(range(23)) and state["N"] == 4
    try:
        e7.batched(3, 1, lambda i, j: 1 / 0, {}, "N")
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("other errors must not be swallowed")


def test_compile_failure_falls_back():
    real = torch.compile
    model = build("softmax")
    rt = e7.Runtime("cpu", {})
    x, y = torch.randint(0, 256, (2, 16)), torch.randint(0, 256, (2, 16))

    def broken(_):
        raise RuntimeError("no compiler")
    torch.compile = broken
    try:
        fwd, compiled = e7.try_compile(model, x, y, rt)
    finally:
        torch.compile = real
    assert fwd is model and not compiled


def _runtime(cfg):
    n = cfg["data_size"]
    raw = torch.from_numpy(np.random.default_rng(2).integers(97, 123, n, dtype=np.uint8))
    a, b = int(0.9 * n), int(0.95 * n)
    return e7.Runtime("cpu", dict(train=raw[:a], val=raw[a:b], test=raw[b:]))


def test_resume_reproduces_the_uninterrupted_run_exactly():
    cfg = dict(CFG, steps=24, ckpt_every=10, val_every=8, inject_crash=None)
    rt = _runtime(cfg)
    torch.set_num_threads(1)
    for arm in e7.ARMS:
        job = dict(id=f"t_{arm}", kind="tune", arm=arm, seed=5, lr=1e-3)
        with tempfile.TemporaryDirectory() as tmp:
            whole, _, _ = e7.train_loop(job, cfg, rt, Path(tmp) / "a.pt")
            # Stop at 15 with no extra save, like a crash: the last checkpoint is at step 10.
            e7.train_loop(job, cfg, rt, Path(tmp) / "b.pt", stop_after=15)
            resumed, state, _ = e7.train_loop(job, cfg, rt, Path(tmp) / "b.pt")
            assert state["resumed_from"] == 10 and state["step"] == 24, arm
            for (name, p), (_, q) in zip(whole.named_parameters(), resumed.named_parameters()):
                assert torch.equal(p, q), (arm, name)
            assert (Path(tmp) / "b.pt").exists(), "the final checkpoint protects the evaluation stage"


def test_batches_shared_by_seed_and_step():
    data = _text()
    gen = torch.Generator()
    a = e7.get_batch(data, 4, 16, gen, 2000, 7)[0]
    e7.get_batch(data, 4, 16, gen, 2000, 8)
    assert torch.equal(a, e7.get_batch(data, 4, 16, gen, 2000, 7)[0])
    assert not torch.equal(a, e7.get_batch(data, 4, 16, gen, 2001, 7)[0])
    x, y = e7.get_batch(data, 4, 16, gen, 2000, 7)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_weight_decay_groups_exclude_normalizer_parameters():
    model = build("entmax15_scaled")
    decay, rest = e7.make_optimizer(model, 1e-3, CFG, "cpu").param_groups
    assert decay["weight_decay"] == CFG["wd"] and rest["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    named = dict(model.named_parameters())
    for name in ("blocks.0.attn.delta", "blocks.0.attn.b_raw", "blocks.0.attn.gamma_raw"):
        assert any(named[name] is p for p in rest["params"]), name
    assert any(named["tok.weight"] is p for p in decay["params"])


WORKER = """
import json, os, sys, time
sys.path.insert(0, {e7dir!r})
import e7
from pathlib import Path
out = Path({out!r})
cfg = json.loads((out / "config.json").read_text())

def fn(job, beat):
    fd = os.open(out / "runs.log", os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    os.write(fd, (job["id"] + "\\n").encode())
    os.close(fd)
    time.sleep(0.05)
    return dict(job, val_loss_final=abs(job["lr"] - 6e-4))

sys.exit(e7.worker_loop(out, cfg, {slot}, fn, poll=0.1))
"""


def _queue_dir(tmp, cfg=CFG):
    out = Path(tmp) / "q"
    e7.make_dirs(out)
    (out / "config.json").write_text(json.dumps(cfg))
    return out


def test_two_concurrent_workers_run_each_job_exactly_once_across_the_barrier():
    with tempfile.TemporaryDirectory() as tmp:
        out = _queue_dir(tmp)
        procs = [subprocess.Popen(
            [sys.executable, "-c", WORKER.format(e7dir=str(ROOT / "experiments" / "E7_gpu"),
                                                 out=str(out), slot=i)])
            for i in range(2)]
        assert all(p.wait(timeout=120) == 0 for p in procs)
        ran = (out / "runs.log").read_text().split()
        expected = [j["id"] for j in e7.tune_jobs(CFG)]
        selection = json.loads((out / "selection.json").read_text())
        expected += [j["id"] for j in e7.final_jobs(CFG, selection)]
        assert sorted(ran) == sorted(expected)
        assert len(set(ran)) == len(ran)
        n_tune = len(e7.tune_jobs(CFG))
        assert all(r.startswith("tune_") for r in ran[:n_tune]), "tuning runs before final jobs"
        assert not list(e7.paths(out)["claims"].iterdir())


def test_stale_claims_are_reclaimed_and_live_ones_kept():
    with tempfile.TemporaryDirectory() as tmp:
        out = _queue_dir(tmp)
        claims = e7.paths(out)["claims"]
        job = e7.tune_jobs(CFG)[0]["id"]

        assert e7.try_claim(out, job, 0)
        assert not e7.try_claim(out, job, 1), "a held claim cannot be taken"
        assert e7.reclaim_stale(out, 1800) == [], "a live owner keeps its claim"

        # An owner that has exited.
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        info = json.loads((claims / job).read_text())
        (claims / job).write_text(json.dumps(dict(info, pid=dead.pid)))
        assert e7.reclaim_stale(out, 1800) == [job]
        assert e7.try_claim(out, job, 1), "a reclaimed job can be claimed again"

        # A reused pid: the same number, but another process start time.
        (claims / job).write_text(json.dumps(dict(info, start="1")))
        assert e7.reclaim_stale(out, 1800) == [job]

        # A live owner with no heartbeat for too long is killed, then reclaimed.
        hung = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        (claims / job).write_text(json.dumps(dict(info, pid=hung.pid, start=e7.proc_start(hung.pid))))
        old = time.time() - 4000
        os.utime(claims / job, (old, old))
        assert e7.reclaim_stale(out, 1800) == [job]
        assert hung.wait(timeout=20) != 0

        # A claim that was never written, once it is old.
        (claims / job).write_text("")
        assert e7.reclaim_stale(out, 1800) == []
        os.utime(claims / job, (old, old))
        assert e7.reclaim_stale(out, 1800) == [job]


def test_retry_once_then_fail_and_failed_tuning_stops_the_barrier():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = dict(CFG, lrs=[1e-3], final_seeds=[2000])
        out = _queue_dir(tmp, cfg)
        calls = []

        def fn(job, beat):
            calls.append(job["id"])
            if job["arm"] == "ssmax":
                raise ValueError("always fails")
            return dict(job, val_loss_final=1.0)

        codes = []
        while True:
            codes.append(e7.worker_loop(out, cfg, 0, fn, poll=0.0))
            if codes[-1] == 0:
                break
            assert len(codes) < 5
        target = "tune_ssmax_lr0.001_s1000"
        assert calls.count(target) == e7.MAX_ATTEMPTS and e7.is_failed(out, target)
        assert not (out / "selection.json").exists()
        assert not any(c.startswith("final_") for c in calls), "no final job without a selection"
        assert e7.settled(out, cfg)


def test_selection_reads_validation_only_and_is_deterministic():
    rows = [dict(arm="softmax", lr=lr, val_loss_final=v, eval="must not be read")
            for lr, v in ((3e-4, 2.0), (6e-4, 1.5), (1.2e-3, 1.5))]
    s = e7.select_lrs(rows)["softmax"]
    assert s["lr"] == 6e-4 and not s["edge"]
    assert e7.select_lrs(rows[::-1])["softmax"]["lr"] == 6e-4
    rows[2]["val_loss_final"] = 1.0
    assert e7.select_lrs(rows)["softmax"]["edge"]
    rows[2]["val_loss_final"] = float("nan")
    assert e7.select_lrs(rows)["softmax"]["lr"] == 6e-4


def test_holm_known_example():
    adjusted, reject = analyze.holm([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])
    assert reject == [True, False, False]
    adjusted, reject = analyze.holm([0.001, 0.02, 0.04])
    assert np.allclose(adjusted, [0.003, 0.04, 0.04]) and reject == [True, True, True]
    adjusted, reject = analyze.holm([float("nan"), 0.001, 0.5])
    assert np.allclose(adjusted, [1.0, 0.003, 1.0]) and reject == [False, True, False]


def test_paired_test_direction():
    a, b = {1: 1.0, 2: 1.1, 3: 0.9}, {1: 2.0, 2: 2.3, 3: 1.8}
    assert analyze.paired(a, b)["p"] < 0.05 < analyze.paired(b, a)["p"]
    assert analyze.paired(a, b)["gain"] > 0
    assert analyze.paired({1: 1.0}, {1: 2.0})["n"] == 1


def test_p3b_is_gatekept_by_p3_and_the_amended_rule():
    better = {1: 1.0, 2: 1.1, 3: 0.9}
    worse = {1: 2.0, 2: 2.3, 3: 1.8}
    closed = analyze.gatekept_p3b(False, better, worse)  # would be significant if it were tested
    assert not closed["tested"] and not closed["sig"] and math.isnan(closed["p"])
    opened = analyze.gatekept_p3b(True, better, worse)
    assert opened["tested"] and opened["sig"] and opened["p"] < 0.05
    against = analyze.gatekept_p3b(True, worse, better)
    assert against["tested"] and not against["sig"]
    assert not analyze.gatekept_p3b(True, {1: 1.0}, {1: 2.0})["sig"], "one seed gives no test"
    for small_rise in (False, True):  # no P3, no success, whatever else holds
        for p3b in (False, True):
            assert not analyze.swap_succeeds(False, small_rise, p3b)
    assert analyze.swap_succeeds(True, True, False) and analyze.swap_succeeds(True, False, True)
    assert not analyze.swap_succeeds(True, False, False)


def test_full_config_follows_amendment_2():
    full = e7.make_config("full")
    assert full["d_model"] == 192 and full["steps"] == 20000
    assert full["final_seeds"] == [2000, 2001, 2002, 2003]
    assert len(e7.tune_jobs(full)) + len(e7.final_jobs(full, {a: dict(lr=1e-3) for a in e7.ARMS})) == 28
    for arm in e7.ARMS:
        params = sum(p.numel() for p in e7.E7Model(full, arm).parameters())
        assert 2_500_000 <= params <= 3_000_000, (arm, params)
    assert e7.make_config("smoke")["d_model"] == 192, "smoke measures the model of the full run"
    assert e7.make_config("smoke")["steps"] == 24 and e7.make_config("mini")["d_model"] == 32


def test_three_outcomes():
    assert analyze.outcome(True, 0.002, 4) == "Supported"
    assert analyze.outcome(True, 0.02, 4) == "Supported"
    assert analyze.outcome(False, 0.02, 4) == "Inconclusive"
    assert analyze.outcome(False, 0.005, 4) == "Inconclusive", "at least 0.005"
    assert analyze.outcome(False, 0.004999, 4) == "Not supported"
    assert analyze.outcome(False, -0.02, 4) == "Not supported"
    assert analyze.outcome(False, 0.02, 1) == "No data"
    assert analyze.outcome(False, float("nan"), 0) == "No data"


def _synthetic_report(p1, p2, p3, drop=None):
    """A full-config report with chosen per-seed gains on mean L for P1, P2 and P3. `drop` is an
    (arm, seed) pair whose job is left out."""
    cfg = e7.make_config("full")
    seeds = cfg["final_seeds"]
    base = dict(zip(seeds, (2.00, 2.03, 1.98, 2.01)))

    def numbers(offsets):
        return dict(L={str(N): base[s] + offsets[i] for N in cfg["Ns"]}, T={"all": 2.5},
                    copy={str(D): dict(byte=0.5, all4=0.2) for D in cfg["Ds"]})

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        e7.make_dirs(out)
        (out / "config.json").write_text(json.dumps(cfg))
        for i, s in enumerate(seeds):
            for arm, offsets in (("softmax", [0.0] * 4), ("entmax15", [-g for g in p1]),
                                 ("ssmax", [g - h for g, h in zip(p2, p1)]),
                                 ("entmax15_scaled", [0.0] * 4)):
                if (arm, s) == drop:
                    continue
                ev = {arm: numbers(offsets)}
                if arm == "softmax":
                    ev = {r: numbers(offsets) for r in e7.READOUTS[arm]}
                    ev["entmax15"] = numbers([-g for g in p3])
                job = dict(id=f"final_{arm}_s{s}", kind="final", arm=arm, seed=s, lr=6e-4, eval=ev,
                           params=2715264, step_ms=10.0, tokens_per_s=1000.0, nonfinite=0)
                (out / "jobs" / f"{job['id']}.json").write_text(json.dumps(job))
        lines = analyze.report(out)[0]
    table = {label: next(x for x in lines if x.startswith(f"  {label} ")) for label in ("P1", "P2", "P3")}
    return table, "\n".join(lines)


def test_report_outcomes_through_the_real_tests():
    significant = [0.0200, 0.0201, 0.0199, 0.0200]  # far past any Holm threshold, with n = 4
    large_noisy = [0.2, -0.15, 0.2, -0.15]  # mean 0.025, t about 0.25
    small_noisy = [0.002, -0.001, 0.002, -0.001]  # mean 0.0005, t about 0.6

    # P2 is entmax15 against ssmax, so its gain is entered directly.
    table, text = _synthetic_report(significant, large_noisy, small_noisy)
    assert table["P1"].endswith(" Supported"), table["P1"]
    assert table["P2"].endswith(" Inconclusive"), table["P2"]
    assert table["P3"].endswith(" Not supported"), table["P3"]
    assert "Kill for transfer of E1.1 (P1 Not supported): NOT FIRED. P1 is Supported" in text
    assert "Model: 6 layers, 6 heads, width 192, parameters [2715264]" in text
    assert "Scale caveat:" in text and "than it would be at 10M" in text
    assert "missing" not in text, "4 seeds is the expected count"

    table, text = _synthetic_report(small_noisy, large_noisy, small_noisy)
    assert table["P1"].endswith(" Not supported"), table["P1"]
    assert "(P1 Not supported): FIRED." in text

    table, text = _synthetic_report(large_noisy, significant, small_noisy)
    assert table["P1"].endswith(" Inconclusive"), table["P1"]
    assert "(P1 Not supported): NOT FIRED. P1 is Inconclusive" in text


def test_verdicts_are_provisional_when_seeds_are_missing():
    significant = [0.0200, 0.0201, 0.0199, 0.0200]
    large_noisy = [0.2, -0.15, 0.2, -0.15]
    small_noisy = [0.002, -0.001, 0.002, -0.001]
    table, text = _synthetic_report(significant, large_noisy, small_noisy)
    assert "PROVISIONAL" not in text, "nothing is missing"
    # entmax15 is in P1 and P2 but not in P3, so only those two lose a seed.
    table, text = _synthetic_report(significant, large_noisy, small_noisy, drop=("entmax15", 2003))
    assert table["P1"].endswith("PROVISIONAL: 3 of 4 seeds"), table["P1"]
    assert table["P2"].endswith("PROVISIONAL: 3 of 4 seeds"), table["P2"]
    assert "PROVISIONAL" not in table["P3"], table["P3"]
    kill = next(x for x in text.splitlines() if "Kill for transfer" in x)
    assert kill.endswith("PROVISIONAL: 3 of 4 seeds"), kill
    assert analyze.provisional(4, 4) == "" and analyze.provisional(0, 4).endswith("0 of 4 seeds")


def test_compiled_flag_survives_a_resume_with_no_steps_left():
    cfg = dict(CFG, steps=12, ckpt_every=10, val_every=100, inject_crash=None)
    rt = _runtime(cfg)
    job = dict(id="t_flag", kind="tune", arm="softmax", seed=5, lr=1e-3)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "c.pt"
        _, state, compiled = e7.train_loop(job, cfg, rt, ckpt)
        assert not compiled and not state["compiled"]
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
        assert ck["compiled"] is False, "the flag is part of the checkpoint"
        ck["compiled"] = True  # as written by a run that trained compiled on a GPU
        torch.save(ck, ckpt)
        _, state, compiled = e7.train_loop(job, cfg, rt, ckpt)
        assert compiled is True and state["compiled"] is True and state["resumed_from"] == 12


def test_code_hash_covers_e7_and_attention():
    import hashlib
    first = (ROOT / "experiments" / "E7_gpu" / "e7.py").read_bytes()
    second = (ROOT / "src" / "mai" / "attention.py").read_bytes()
    assert e7.code_sha() == hashlib.sha1(first + second).hexdigest()
    assert e7.code_sha() != hashlib.sha1(first).hexdigest()
    assert e7.env_info(e7.Runtime("cpu", {}))["code_sha"] == e7.code_sha()


def test_idle_worker_exits_when_the_launcher_is_gone():
    real = os.getppid
    with tempfile.TemporaryDirectory() as tmp:
        cfg = dict(CFG, lrs=[1e-3], final_seeds=[2000])
        out = _queue_dir(tmp, cfg)
        for job in e7.tune_jobs(cfg):  # every job held by a live owner: this worker can only wait
            assert e7.try_claim(out, job["id"], 1)

        def never(job, beat):
            raise AssertionError("no job may run")

        values = iter([111])  # the launcher's pid at the start, then something else
        os.getppid = lambda: next(values, 222)
        try:
            assert e7.worker_loop(out, cfg, 0, never, poll=0.0) == 9
        finally:
            os.getppid = real


def test_is_oom_recognizes_gpu_allocation_failures():
    for text in ("CUDA out of memory. Tried to allocate 2.00 GiB",
                 "CUDA error: out of memory",
                 "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`",
                 "DefaultCPUAllocator: can't allocate: out of memory"):
        assert e7.is_oom(RuntimeError(text)), text
    assert e7.is_oom(torch.cuda.OutOfMemoryError("x"))
    for text in ("CUDA error: device-side assert triggered", "shape mismatch"):
        assert not e7.is_oom(RuntimeError(text)), text


def test_grad_norm_check_with_plain_callables():
    rt = e7.Runtime("cpu", {})
    model = build("softmax")
    before = [p.detach().clone() for p in model.parameters()]
    x, y = torch.randint(0, 256, (2, 16)), torch.randint(0, 256, (2, 16))
    same = lambda inp: model(inp)  # noqa: E731
    half = lambda inp: 0.5 * model(inp) + 0.5 * model(inp).detach()  # noqa: E731
    near = lambda inp: 0.999 * model(inp) + 0.001 * model(inp).detach()  # noqa: E731

    agree, a, b = e7.compare_grad_norms(model, same, model.parameters(), x, y, rt)
    assert agree and a > 0 and abs(a - b) <= 1e-6 * a
    agree, a, b = e7.compare_grad_norms(model, half, model.parameters(), x, y, rt)
    assert not agree and abs(b / a - 0.5) < 1e-3
    assert e7.compare_grad_norms(model, near, model.parameters(), x, y, rt)[0], "within 1 percent"

    # No trace on the training state: no gradients, and parameters untouched.
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before))

    # try_compile end to end, with a stand-in for the compiler: the same loss but other gradients.
    real = torch.compile
    try:
        torch.compile = lambda m: half
        fwd, compiled = e7.try_compile(model, x, y, rt)
        assert fwd is model and not compiled, "a gradient mismatch falls back"
        assert all(p.grad is None for p in model.parameters())
        torch.compile = lambda m: same
        fwd, compiled = e7.try_compile(model, x, y, rt)
        assert fwd is same and compiled
        assert all(p.grad is None for p in model.parameters())
    finally:
        torch.compile = real


def test_mini_end_to_end_with_analysis():
    """The launcher, a crash and resume, real training and evaluation, and analyze.py."""
    with tempfile.TemporaryDirectory() as tmp:
        out, data = Path(tmp) / "mini", Path(tmp) / "data"
        run = subprocess.run(
            [sys.executable, str(ROOT / "experiments" / "E7_gpu" / "e7.py"), "--mini", "--out", str(out),
             "--data-dir", str(data)], capture_output=True, text=True, timeout=240)
        assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
        assert "resume verified" in run.stdout, run.stdout[-3000:]
        assert analyze.main(["--out", str(out)]) == 0
        lines = (out / "report.txt").read_text().splitlines()
        text = "\n".join(lines)
        assert len(lines) <= 150, len(lines)
        for needle in ("P1 entmax15", "P3 softmax", "Readout swap:", "Kill for transfer", "G(N) rule"):
            assert needle in text, needle

        finals = [json.loads(p.read_text()) for p in (out / "jobs").glob("final_*.json")]
        assert len(finals) == 4 * len(CFG["final_seeds"])
        assert len({j["eval_hash"] for j in finals}) == 1, "the same windows and probes for all"
        for j in finals:
            for readout, numbers in j["eval"].items():
                assert set(numbers) == {"L", "T", "copy"}
                assert all(math.isfinite(v) for v in numbers["L"].values()), (j["id"], readout)
                assert len(numbers["L"]) == 4 and len(numbers["copy"]) == 4
            assert set(j["eval"]) == set(e7.READOUTS[j["arm"]])
            assert j["params"] > 0 and j["step_ms"] > 0
        for j in finals:
            if j["arm"] == "softmax":  # gated reuses softmax where every row has n <= ctx
                g, s = j["eval"]["gated"], j["eval"]["softmax"]
                assert g["T"] == s["T"] and g["L"][str(CFG["ctx"])] == s["L"][str(CFG["ctx"])]
        assert "P3b" in text
        rows = (out / "results_compact.csv").read_text().splitlines()
        assert len(rows) == 1 + len(e7.tune_jobs(CFG)) + 2 * (4 + 1 + 1 + 1)
        assert not list(e7.paths(out)["ckpt"].iterdir()), "checkpoints are deleted after results"


if __name__ == "__main__":
    failures = 0
    start = time.time()
    for name, fn in sorted(globals().copy().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            try:
                fn()
                print(f"ok   {name} ({time.time() - t0:.1f}s)", flush=True)
            except Exception:
                failures += 1
                print(f"FAIL {name}\n{traceback.format_exc()}", flush=True)
    print(f"{failures} failures, {time.time() - start:.0f}s")
    sys.exit(1 if failures else 0)
