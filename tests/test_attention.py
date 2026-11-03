"""Tests for src/mai/attention.py. Run with `pytest tests/test_attention.py` or
`python tests/test_attention.py`. The __main__ runner needs no pytest."""
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mai.attention import (NORMALIZERS, MultiHeadAttention, calib_beta, entmax15, rem_beta,  # noqa: E402
                           row_audit, score_stats)

D, H = 8, 2
DH = D // H
KAPPA = math.log(9.0)


def make(name, dtype=torch.float64, **kw):
    return MultiHeadAttention(D, H, normalizer=name, **kw).to(dtype)


def degenerate_batch(dtype=torch.float64):
    """Rows (T dim): ordinary, identical, n=1, n=2, n=2 tied, n=3, all masked, huge scale,
    tiny scale, top tie, big offset, identical at an offset, NaN hidden under the mask."""
    torch.manual_seed(0)
    S = 6
    rows = [torch.randn(S) for _ in range(13)]
    rows[1] = torch.full((S,), 2.5)
    rows[4] = torch.full((S,), -1.0)
    rows[7] = torch.randn(S) * 1e6
    rows[8] = torch.randn(S) * 1e-8
    rows[9] = torch.tensor([3.0, 3.0, 1.0, 0.0, -1.0, 0.5])
    rows[10] = torch.randn(S) + 1e4
    rows[11] = torch.full((S,), 1e4)
    rows[12] = torch.randn(S)
    rows[12][4:] = float("nan")
    valid = [S, S, 1, 2, 2, 3, 0, S, S, S, S, S, 4]
    mask = torch.zeros(len(rows), S, dtype=torch.bool)
    for i, v in enumerate(valid):
        mask[i, torch.randperm(S)[:v] if i != 12 else torch.arange(4)] = True
    z = torch.stack(rows).to(dtype)
    # Shape (B=1, h, T, S), with head 2 a differently scaled copy of head 1.
    z = torch.stack([z, 0.5 * z + 1.0])[None]
    mask = torch.stack([mask, mask])[None]
    return z, mask


def test_sums_to_one_and_zero_on_masked():
    for dtype, tol in ((torch.float64, 1e-9), (torch.float32, 2e-4)):
        z, mask = degenerate_batch(dtype)
        has = mask.any(-1)
        for name in NORMALIZERS:
            w = make(name, dtype).normalize(z, mask)
            assert torch.isfinite(w).all(), name
            assert (w >= 0).all(), name
            assert (w[~mask] == 0).all(), name
            total = w.sum(-1)
            assert ((total[has] - 1).abs() < tol).all(), (name, dtype, total)
            assert (total[~has] == 0).all(), name


def test_finite_gradients_on_degenerate_rows():
    for dtype in (torch.float64, torch.float32):
        z0, mask = degenerate_batch(dtype)
        torch.manual_seed(1)
        coef = torch.randn(z0.shape, dtype=dtype)
        for name in NORMALIZERS:
            for kw in ({"learn_kappa": True}, {}) if name == "rem" else ({},):
                m = make(name, dtype, **kw)
                z = z0.clone().requires_grad_(True)
                (m.normalize(z, mask) * coef).sum().backward()
                assert torch.isfinite(z.grad).all(), (name, kw, dtype)
                for pname, p in m.named_parameters(recurse=False):
                    assert torch.isfinite(p.grad).all(), (name, pname, dtype)


def test_module_forward_backward_with_padding():
    torch.manual_seed(2)
    x = torch.randn(3, 5, D)
    x[1] = 1.0  # identical tokens produce identical scores
    pad = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
    rowmask = pad[:, None, :].expand(-1, 5, -1).clone()
    rowmask[0, 2] = False  # one all-masked row
    for name in NORMALIZERS:
        for mask in (None, pad, rowmask):
            m = MultiHeadAttention(D, H, normalizer=name)
            out = m(x, mask=mask)
            assert out.shape == x.shape and torch.isfinite(out).all(), name
            out.pow(2).sum().backward()
            for pname, p in m.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), (name, pname)


def test_masked_row_equals_truncated_row():
    """Stats and outputs may depend only on unmasked entries."""
    torch.manual_seed(3)
    keep = 4
    short = torch.randn(1, H, 5, keep, dtype=torch.float64)
    junk = 1e3 * torch.randn(1, H, 5, 3, dtype=torch.float64)
    long = torch.cat([short, junk], -1)
    mask = torch.zeros_like(long, dtype=torch.bool)
    mask[..., :keep] = True
    for name in NORMALIZERS:
        m = make(name)
        a = m.normalize(long, mask)[..., :keep]
        b = m.normalize(short)
        assert (a - b).abs().max() < 1e-10, name


def test_parameter_init_is_shared_across_arms():
    ref = None
    for name in NORMALIZERS:
        torch.manual_seed(4)
        m = MultiHeadAttention(D, H, normalizer=name)
        w = torch.cat([m.q.weight.flatten(), m.k.weight.flatten(), m.v.weight.flatten(), m.out.weight.flatten()])
        ref = w if ref is None else ref
        assert torch.equal(ref, w), name
    assert make("rem").n_extra_params() == 0 and make("calib").n_extra_params() == 0
    assert make("ssmax").n_extra_params() == H and make("entmax15_scaled").n_extra_params() == 3 * H


def test_rem_matches_preregistered_formula():
    torch.manual_seed(5)
    n = 200
    # Above threshold: a clear needle. Below threshold: no needle, so the clamp is active.
    z_noise = torch.randn(4, n, dtype=torch.float64)
    z_noise[0, 0] = 30.0
    z_noise[1, 0] = 9.0
    beta, _ = rem_beta(z_noise, torch.ones_like(z_noise, dtype=torch.bool), KAPPA)
    branches = set()
    for i in range(4):
        row = z_noise[i]
        top = row.argmax()
        rest = torch.cat([row[:top], row[top + 1:]])
        mu, s2 = row[top] - rest.mean(), rest.var()
        L = math.log(n) + KAPPA
        d = mu * mu - 2 * s2 * L
        expected = (mu - torch.sqrt(d.clamp(min=0.0))) / s2  # literal H1 formula
        assert abs(beta[i, 0] - expected) < 1e-9 * expected.abs().clamp(min=1.0), i
        branches.add(bool(d > 0))
    assert branches == {True, False}, "test must exercise both branches"


def test_rem_limits():
    # sigma -> 0: beta -> L / mu, finite. Here n = 5 with the 4 others identical.
    z = torch.tensor([[7.0, 2.0, 2.0, 2.0, 2.0]], dtype=torch.float64)
    beta, _ = rem_beta(z, torch.ones_like(z, dtype=torch.bool), KAPPA)
    assert abs(beta.item() - (math.log(5) + KAPPA) / 5.0) < 1e-12
    # n = 2 has no variance estimate, so it takes the same limit.
    z = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    beta, _ = rem_beta(z, torch.ones_like(z, dtype=torch.bool), KAPPA)
    assert abs(beta.item() - (math.log(2) + KAPPA) / 2.0) < 1e-12
    # Identical scores give beta = 0.
    z = torch.full((1, 6), 3.0, dtype=torch.float64)
    beta, _ = rem_beta(z, torch.ones_like(z, dtype=torch.bool), KAPPA)
    assert beta.item() == 0.0


def test_rem_scale_and_shift_invariance():
    torch.manual_seed(6)
    z = torch.randn(3, H, 4, 50, dtype=torch.float64)
    z[..., 0] += 6.0
    mask = torch.rand(z.shape) > 0.2
    m = make("rem")
    w0 = m.normalize(z, mask)
    b0 = m.row_beta(z, mask)
    for c in (1e-3, 0.5, 7.0, 1e3):
        for t in (0.0, -50.0, 1e3):
            zz = c * z + t
            assert (m.normalize(zz, mask) - w0).abs().max() < 1e-7, (c, t)
            assert (m.row_beta(zz, mask) * c / b0.clamp(min=1e-12) - 1).abs().max() < 1e-7, (c, t)


def test_beta_is_detached_for_plugin_rules():
    z = torch.randn(1, H, 2, 9, requires_grad=True)
    assert not make("rem", torch.float32).row_beta(z).requires_grad
    assert not make("calib", torch.float32).row_beta(z).requires_grad


def test_learned_kappa_gets_a_gradient_and_keeps_the_value():
    torch.manual_seed(7)
    z = torch.randn(4, H, 3, 40)
    z[..., 0] += 12.0  # above threshold
    fixed = MultiHeadAttention(D, H, normalizer="rem")
    learn = MultiHeadAttention(D, H, normalizer="rem", learn_kappa=True)
    assert (fixed.normalize(z) - learn.normalize(z)).abs().max() < 1e-6
    (learn.normalize(z)[..., 0]).sum().backward()
    assert learn.kappa.grad.abs().min() > 0 and torch.isfinite(learn.kappa.grad).all()
    assert (fixed.row_beta(z) - learn.row_beta(z).detach()).abs().max() < 1e-6


def _ref_entmax(z):
    """Independent solve: bisection on the KKT stationarity condition, with no sorting."""
    x = z / 2.0
    lo, hi = x.max() - 1.0, x.max()  # F(lo) >= 1 and F(hi) = 0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (np.clip(x - mid, 0, None) ** 2).sum() > 1.0:
            lo = mid
        else:
            hi = mid
    return np.clip(x - 0.5 * (lo + hi), 0, None) ** 2


def _objective(p, z):
    return p @ z + (4.0 / 3.0) * (1.0 - (p ** 1.5).sum())


def test_entmax_matches_slow_reference():
    rng = np.random.default_rng(8)
    for trial in range(200):
        n = int(rng.integers(1, 40))
        z = rng.normal(size=n) * float(rng.choice([0.01, 0.3, 1.0, 5.0, 100.0]))
        out = entmax15(torch.tensor(z)[None]).numpy()[0]
        ref = _ref_entmax(z)
        assert np.abs(out - ref).max() < 1e-8, (trial, n)
        # First-principles check: p maximizes p.z + H_1.5(p) on the simplex (concave).
        best = _objective(ref, z)
        for _ in range(20):
            r = rng.dirichlet(np.ones(n))
            t = float(rng.uniform(0.001, 1.0))
            assert _objective((1 - t) * ref + t * r, z) <= best + 1e-9, (trial, n)


def test_entmax_mask_and_known_value():
    z = torch.tensor([[1.0, 0.0, 99.0, -99.0]], dtype=torch.float64)
    mask = torch.tensor([[True, True, False, False]])
    w = entmax15(z, mask)[0]
    tau = (1 - math.sqrt(7)) / 4  # solves (0.5 - t)^2 + t^2 = 1
    assert abs(w[0].item() - (0.5 - tau) ** 2) < 1e-10 and abs(w[1].item() - tau ** 2) < 1e-10
    assert w[2].item() == 0 and w[3].item() == 0


def test_entmax_gradcheck():
    torch.manual_seed(9)
    z = (2.0 * torch.randn(3, 7, dtype=torch.float64)).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda t: entmax15(t), (z,), eps=1e-6, atol=1e-4, rtol=1e-3)
    mask = torch.rand(3, 7) > 0.3
    mask[:, 0] = True
    assert torch.autograd.gradcheck(lambda t: entmax15(t, mask), (z,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_calib_reaches_target_mass():
    torch.manual_seed(10)
    for dtype, tol in ((torch.float64, 1e-4), (torch.float32, 2e-3)):
        for p in (0.5, 0.9, 0.99):
            z = torch.randn(4, H, 5, 64, dtype=dtype)
            mask = torch.rand(z.shape) > 0.3
            m = make("calib", dtype, calib_p=p)
            zf = z.masked_fill(~mask, -math.inf)
            # Rows whose top two nearly tie can hit the spread cap by design; test the rest.
            top2 = zf.topk(2, dim=-1).values
            clear = (top2[..., 0] - top2[..., 1]) > 0.01 * score_stats(z, mask)["spread"][..., 0]
            top = m.normalize(z, mask).max(-1).values
            assert (top[clear] - p).abs().max() < tol, (dtype, p, top)
            # Smallest such beta: 2% less must miss the target.
            beta = calib_beta(z, mask, p, 20)
            lower = torch.softmax(0.98 * beta * (zf - zf.max(-1, keepdim=True).values), -1).max(-1).values
            assert (lower[clear] < p).all(), (dtype, p)


def test_calib_edge_cases():
    m = make("calib", calib_p=0.9)
    z = torch.tensor([[[[2.0, 0.0]]]], dtype=torch.float64).expand(1, H, 1, 2)
    assert (m.normalize(z).max(-1).values - 0.9).abs().max() < 1e-6  # n = 2
    m = make("calib", calib_p=0.2)
    z = torch.randn(1, H, 1, 3, dtype=torch.float64)
    assert (m.normalize(z) - 1.0 / 3).abs().max() < 1e-12  # p < 1/n: already met at beta = 0
    m = make("calib", calib_p=0.9)
    z = torch.randn(2, H, 3, 32, dtype=torch.float64)
    a, b = m.normalize(z), m.normalize(250.0 * z - 40.0)
    assert (a - b).abs().max() < 1e-3  # scale and shift invariant, to bisection precision


def test_ssmax_matches_formula():
    torch.manual_seed(11)
    m = make("ssmax")
    m.s.data = torch.tensor([0.3, 0.8], dtype=torch.float64).view(H, 1, 1)
    z = torch.randn(2, H, 3, 12, dtype=torch.float64)
    mask = torch.rand(z.shape) > 0.4
    mask[..., 0] = True
    w = m.normalize(z, mask)
    for b in range(2):
        for h in range(H):
            for t in range(3):
                idx = mask[b, h, t]
                n = int(idx.sum())
                x = z[b, h, t][idx] / math.sqrt(DH)
                s = m.s.data[h, 0, 0].item()
                if n > 1:
                    powers = torch.tensor(float(n), dtype=x.dtype) ** (s * x)  # eq. 2: n^(s z)
                    expected = powers / powers.sum()
                else:
                    expected = torch.ones(1, dtype=x.dtype)
                assert (w[b, h, t][idx] - expected).abs().max() < 1e-10, (b, h, t)


def test_scaled_entmax_matches_formula():
    torch.manual_seed(12)
    m = make("entmax15_scaled")
    m.delta.data.fill_(0.2)
    m.b_raw.data.fill_(0.4)
    m.gamma_raw.data.fill_(-0.1)
    z = torch.randn(2, H, 3, 12, dtype=torch.float64)
    mask = torch.rand(z.shape) > 0.4
    mask[..., 0] = True
    w = m.normalize(z, mask)
    b = math.log1p(math.exp(0.4))
    gamma = 4.0 * math.tanh(-0.1)
    n = mask.sum(-1, keepdim=True).double().clamp(min=2.0)
    scale = 0.2 + b * n.log() ** gamma
    assert (w - entmax15(scale * z / math.sqrt(DH), mask)).abs().max() < 1e-10


def test_softmax_and_learned_scale_formula():
    torch.manual_seed(13)
    z = torch.randn(2, H, 3, 9, dtype=torch.float64)
    w = make("softmax").normalize(z)
    assert (w - torch.softmax(z / math.sqrt(DH), -1)).abs().max() < 1e-12
    m = make("learned_scale")
    m.c.data.fill_(2.5)
    assert (m.normalize(z) - torch.softmax(2.5 * z / math.sqrt(DH), -1)).abs().max() < 1e-12


def test_score_stats_and_audit():
    z = torch.tensor([[5.0, 1.0, 2.0, 3.0, 4.0, 99.0]], dtype=torch.float64)
    mask = torch.tensor([[True, True, True, True, True, False]])
    s = score_stats(z, mask)
    rest = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    assert s["n"].item() == 5 and s["zmax"].item() == 5.0
    assert abs(s["mean"].item() - rest.mean().item()) < 1e-12
    assert abs(s["var"].item() - rest.var().item()) < 1e-12
    a = row_audit(z, mask)
    m2 = ((rest - 2.5) ** 2).mean()
    m4 = ((rest - 2.5) ** 4).mean()
    assert abs(a["exkurt"].item() - (m4 / m2 ** 2 - 3).item()) < 1e-12
    assert abs(a["mu_over_sigma"].item() - 2.5 / rest.std().item()) < 1e-12


if __name__ == "__main__":
    failed = 0
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
