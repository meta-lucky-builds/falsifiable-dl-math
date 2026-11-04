"""Tests for E8. `python tests/test_e8.py`. The __main__ runner needs no pytest. CPU only.

The most important ones check that the long-row path E8 is meant to diagnose behaves as claimed,
so a failure in the real run cannot come from the measurement."""
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "E8_long_context"))
import e8  # noqa: E402
import e7  # noqa: E402  # e8 puts it on the path; install() is not called here, so this E7 is pristine

# E7 has an analyze.py too, and it is earlier on the path, so load this one by its location
_spec = importlib.util.spec_from_file_location(
    "e8_analyze", ROOT / "experiments" / "E8_long_context" / "analyze.py")
analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze)

CFG = dict(e7.make_config("mini"), ctx=32, keep=8)


def model(arm="rope_1e4", seed=0, cfg=CFG):
    torch.manual_seed(seed)
    return e8.E8Model(cfg, arm).eval()


def tokens(T, B=3, seed=1):
    return torch.randint(0, 256, (B, T), generator=torch.Generator().manual_seed(seed))


def test_rope_arm_equals_e7():
    """Same seed, same weights, same logits, at lengths in and past the training length."""
    torch.manual_seed(0)
    old = e7.E7Model(CFG, "softmax").eval()
    new = model()
    old_names = dict(old.state_dict())
    new_names = {k: v for k, v in new.state_dict().items()}
    assert set(old_names) == set(new_names), set(old_names) ^ set(new_names)
    for k in old_names:
        assert torch.equal(old_names[k], new_names[k]), k
    with torch.no_grad():
        for T in (8, 32, 100, 256):
            x = tokens(T)
            assert torch.equal(old(x), new(x)), T


def test_row_depends_only_on_earlier_columns():
    """Row r of a long window equals the last row of a window of length r + 1. Basis of the per-row pass."""
    for arm in e8.ARM_SPECS:
        m = model(arm)
        x = tokens(200)
        with torch.no_grad():
            full = m(x)
            for r in (0, 7, 31, 32, 63, 64, 150, 199):
                assert (m(x[:, :r + 1])[:, -1] - full[:, r]).abs().max() < 2e-3, (arm, r)


def test_relative_positions_only():
    """A one-layer model with a distance limit sees only the last W bytes, wherever they sit. So RoPE
    depends on distance alone, including at positions past the training length."""
    cfg = dict(CFG, n_layer=1)
    for arm in ("rope_1e4", "rope_5e5", "alibi", "rope_1e4_alibi"):
        m = model(arm, cfg=cfg)
        W = 16
        m.set_window(W)
        tail = tokens(W)
        with torch.no_grad():
            short = m(tail)[:, -1]
            for pad in (1, 40, 300):
                x = torch.cat((tokens(pad, seed=2), tail), 1)
                assert (m(x)[:, -1] - short).abs().max() < 2e-3, (arm, pad)


def test_keep_matches_full_pass():
    m = model()
    x = tokens(128)
    with torch.no_grad():
        assert (m(x, keep=8) - m(x)[:, -8:]).abs().max() < 1e-4


def test_window_changes_only_rows_past_it():
    m = model()
    x = tokens(128)
    with torch.no_grad():
        a = m(x)
        m.set_window(32)
        b = m(x)
        m.set_window(None)
        c = m(x)
    assert torch.equal(a, c)
    assert torch.equal(a[:, :32], b[:, :32])
    assert (a[:, 40:] - b[:, 40:]).abs().max() > 1e-4


def test_window_mass_and_alibi_values():
    """The weights themselves, against a direct calculation at a row past the window."""
    a = e8.E8Attention(32, 2, 32, alibi=True)
    assert abs(a.slopes[0].item() - 2.0 ** -4) < 1e-7 and abs(a.slopes[1].item() - 2.0 ** -8) < 1e-7
    a.window = 3
    z = torch.randn(1, 2, 1, 10)
    mask = torch.ones(1, 10, dtype=torch.bool)
    n = torch.tensor([[10.0]])
    w = a._weights(z, mask, n, False)
    d = torch.arange(9, -1, -1, dtype=torch.float32)
    for h in range(2):
        logit = z[0, h, 0] / math.sqrt(a.dh) - a.slopes[h, 0, 0] * d
        logit[d >= 3] = -math.inf
        assert (F.softmax(logit, -1) - w[0, h, 0]).abs().max() < 1e-5
    assert (w[..., :7] == 0).all()


def test_no_rope_arm_is_position_free_except_bias():
    m = model("alibi")
    cos, sin = m.position_tables(9, "cpu")
    assert (cos == 1).all() and (sin == 0).all()


def test_theta_changes_only_rope():
    a, b = model("rope_1e4"), model("rope_5e5")
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    ca, _ = a.position_tables(4, "cpu")
    cb, _ = b.position_tables(4, "cpu")
    assert torch.equal(ca[:, 0], cb[:, 0]) and not torch.equal(ca[:, -1], cb[:, -1])


def test_per_row_bits_agrees_with_e7():
    """Summed per-row loss against E7's evaluation of the same windows and rows."""
    m = model()
    data = torch.randint(0, 256, (3000,), generator=torch.Generator().manual_seed(3), dtype=torch.long).byte()
    ends = torch.tensor([500, 900, 1500, 2500])
    rows = e8.per_row_bits(m, data, ends, 64, 2, {}, "t")
    want = e7.eval_windows(m, data, ends, 64, 64, 2, {})
    assert abs(rows.mean().item() - want) < 1e-6
    want_tail = e7.eval_windows(m, data, ends, 64, 8, 2, {})
    assert abs(rows[-8:].mean().item() - want_tail) < 1e-3


def test_oom_halving_keeps_the_sum():
    calls = []

    def fn(i, j):
        calls.append(j - i)
        if j - i > 2:
            raise RuntimeError("CUDA error: out of memory")
        return (torch.tensor([float(j - i), 1.0]) * 1.0,)
    total, = e7.batched(5, 4, fn, {}, "k")
    assert total[0].item() == 5.0


def test_bins():
    assert e8.bin_rows(torch.arange(8.0), 4) == [1.5, 5.5]
    try:
        e8.bin_rows(torch.arange(7.0), 4)
    except ValueError:
        return
    raise AssertionError("uneven bins must fail")


def test_recorder_against_brute_force():
    T, keep, far_from = 64, 8, 32
    m = model()
    x = tokens(T)
    edges = e8.distance_edges(CFG)
    recorders = m.set_recorders(lambda: e8.Recorder(T, keep, far_from, edges, "cpu"))
    with torch.no_grad():
        m(x)
    m.set_recorders()
    # recompute the first layer's weights directly, from the first attention's inputs
    a = m.blocks[0].attn
    cos, sin = m.position_tables(T, "cpu")
    h = a.h
    with torch.no_grad():
        y = m.blocks[0].ln1(m.tok(x))
        q = e7.rope(a.q(y).view(3, T, h, a.dh).transpose(1, 2), cos, sin)
        k = e7.rope(a.k(y).view(3, T, h, a.dh).transpose(1, 2), cos, sin)
        z = (q @ k.transpose(-1, -2)) / math.sqrt(a.dh)
        z = z.masked_fill(~torch.ones(T, T, dtype=torch.bool).tril(), -math.inf)
        w = z.softmax(-1)
    dist = torch.arange(T)[:, None] - torch.arange(T)
    far = (w * (dist >= far_from)).sum(-1).mean((0, 1))
    got = recorders[0].result(keep)
    assert np.allclose(got["far"], e8.bin_rows(far, keep), atol=1e-5), got["far"]
    bucket = torch.bucketize(torch.arange(T), torch.tensor(edges[1:-1]), right=True)
    tail = torch.zeros(len(edges) - 1, dtype=torch.float64)
    for r in range(T - keep, T):
        for c in range(r + 1):
            tail[bucket[r - c]] += w[:, :, r, c].sum().double()
    assert np.allclose(got["mass"], (tail / tail.sum()).tolist(), atol=1e-5), got["mass"]
    assert abs(sum(got["mass"]) - 1.0) < 1e-9


def test_copy_probe_layout():
    text = np.frombuffer(bytes(range(256)) * 40, dtype=np.uint8)
    ends = np.array([2100, 3000])
    for D in (12, 32, 200):
        x, y = e8.make_copy_probes(text, ends, D)
        assert x.shape == (2, D + 3)
        L = D + 3
        for w in range(2):
            s = np.concatenate((x[w, :4], y[w]))  # the planted string: start, then the targets
            assert (x[w, :8] == s).all()
            # the last 4 rows read bytes 3 to 6 of the string and predict bytes 4 to 7
            assert (x[w, L - 4:] == s[3:7]).all()
            assert (x[w, L - 7:L - 4] == s[:3]).all()
            # the string's start sits D positions before the repeat ends at row E = D
            assert (x[w, D - 4:D] == x[w, :4]).all()


def test_config():
    full = e8.make_config("full")
    assert full["final_seeds"] == [3000, 3001, 3002] and full["lr"] == 1.2e-3 and not full["lrs"]
    assert not set(full["final_seeds"]) & set(e7.FULL["final_seeds"])
    assert e8.arm_config(full, "control_ctx4x")["ctx"] == 1024
    assert e8.arm_config(full, "control_ctx4x")["batch"] == 16
    assert all(N % full["keep"] == 0 for N in full["Ns"])
    assert sum(p.numel() for p in e8.E8Model(full, "alibi").parameters()) == 2715264


def fake_run(arm, seed, L_far, Lw_far, far=0.9, ctx=32, val=1.0):
    bins = 32
    L_bins = [2.0] * bins
    Lw_bins = [2.0] * bins
    for i in range(ctx // 8, bins):
        L_bins[i], Lw_bins[i] = L_far, Lw_far
    layer = dict(far=[0.0] * (ctx // 8) + [far] * (bins - ctx // 8), mass=[1.0 / 6] * 6, logit=[0.0] * 6)
    ev = dict(L_bins=L_bins, Lw_bins=Lw_bins, T=2.0, attention=dict(edges=[0, 2, 8, 32, 64, 128, 256],
                                                                  layers=[layer] * 2))
    for D in (16, 24):
        ev[f"copy_{D}"] = dict(byte=0.1, all4=0.0)
    return dict(arm=arm, seed=seed, ctx=ctx, batch=8, step_ms=1.0, tokens_per_s=1.0, nonfinite=0,
                val_loss_final=val, eval=ev, eval_hash="abcdefgh")


FAKE_CFG = dict(name="full", keep=8, Ns=[32, 64, 128, 256], Ds=[16, 24], final_seeds=[1, 2, 3])


def fake_runs(base=(5.0, 2.0), control_far=(2.0, 2.0), survive=True):
    runs = {}
    runs["rope_1e4"] = [fake_run("rope_1e4", s, *base) for s in FAKE_CFG["final_seeds"]]
    runs["alibi"] = [fake_run("alibi", s, 2.0, 2.0, far=0.1) if survive else fake_run("alibi", s, 3.0, 2.0)
                     for s in FAKE_CFG["final_seeds"]]
    # control trains at 128, so N = 128 is in range; its loss at N = 128 is what C1 reads
    runs["control_ctx4x"] = [fake_run("control_ctx4x", s, *control_far, ctx=128) for s in FAKE_CFG["final_seeds"]]
    return runs


def test_rules_all_pass():
    lines, v = analyze.rules(FAKE_CFG, fake_runs())
    assert v["C1"] and v["C2"] and v["C3"]
    assert v["survive"]["alibi"] and not v["survive"]["rope_1e4"]
    assert any("SUPPORTED" in x and "NOT" not in x for x in lines)
    assert any("normalizer comparison" in x for x in lines)


def test_rules_control_failure_stops():
    runs = fake_runs()
    for r in runs["control_ctx4x"]:
        r["eval"]["L_bins"] = [4.0] * 32
    lines, v = analyze.rules(FAKE_CFG, runs)
    assert not v["C1"] and "survive" not in v
    assert any("bug is the leading" in x for x in lines)


def test_rules_no_survivor():
    lines, v = analyze.rules(FAKE_CFG, fake_runs(survive=False))
    assert not v["survive"]["alibi"]
    assert any("NOT SUPPORTED" in x for x in lines)
    assert any("train at length" in x for x in lines)


def test_rules_seed_outlier_fails_survival():
    runs = fake_runs()
    r = runs["alibi"][0]
    i = analyze.bin_index(256, 8)
    r["eval"]["L_bins"][i] = r["eval"]["Lw_bins"][i] + 0.051
    assert analyze.survives(FAKE_CFG, runs["alibi"]) is False


def test_rules_useful_requires_interval():
    runs = fake_runs()
    for r, gain in zip(runs["alibi"], (0.02, 0.021, 0.019)):
        i = analyze.bin_index(256, 8)
        r["eval"]["L_bins"][i] = r["eval"]["Lw_bins"][i] - gain
    ok, N, m, lo, hi = analyze.useful(FAKE_CFG, runs["alibi"])
    assert ok and N == 256 and hi < 0
    runs["alibi"][0]["eval"]["L_bins"][i] = runs["alibi"][0]["eval"]["Lw_bins"][i] + 0.1
    assert not analyze.useful(FAKE_CFG, runs["alibi"])[0]


def test_report_renders():
    text = analyze.report(FAKE_CFG, fake_runs())
    for needle in ("C1 positive control", "Loss per row", "Copy probe", "PARTIAL"):
        assert needle in text, needle


def test_mini_end_to_end():
    """The whole run through the real launcher, with an injected crash and a resume."""
    tmp = Path(tempfile.mkdtemp())
    try:
        out = tmp / "out"
        cmd = [sys.executable, str(ROOT / "experiments" / "E8_long_context" / "e8.py"), "--mini",
               "--out", str(out), "--data-dir", str(tmp / "data")]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        jobs = sorted((out / "jobs").glob("*.json"))
        assert len(jobs) == 10, jobs
        res = {p.stem: json.loads(p.read_text()) for p in jobs}
        assert res["final_rope_1e4_s3000"]["resumed_from"] > 0
        assert len({v["eval_hash"] for v in res.values()}) == 1
        assert res["final_control_ctx4x_s3000"]["ctx"] == 128
        for v in res.values():
            assert not (v["ctx"] == 32 and abs(v["eval"]["L_bins"][3] - v["eval"]["Lw_bins"][3]) > 1e-3), v["id"]
        assert (out / "models" / "final_alibi_s3001.pt").exists()
        r = subprocess.run([sys.executable, str(ROOT / "experiments" / "E8_long_context" / "analyze.py"),
                            "--out", str(out)], capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr[-2000:]
        text = (out / "report.txt").read_text()
        assert "NOT A RESULT" in text and "C1 positive control" in text and "CHECK FAILED" not in text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    names = [k for k in list(globals()) if k.startswith("test_")]
    if len(sys.argv) > 1:
        names = [n for n in names if sys.argv[1] in n]
    failed = 0
    for name in names:
        try:
            globals()[name]()
            print("ok  ", name, flush=True)
        except Exception:
            failed += 1
            print("FAIL", name, flush=True)
            traceback.print_exc()
    print(f"{len(names) - failed} of {len(names)} passed")
    sys.exit(1 if failed else 0)
