# E6.0: checks for H6 (decay on then off vs small init, diagonal nets x = u^2 - v^2).
# Plain script, numpy and scipy only. Exit code 1 if a pre-registered kill fires.
import math, sys
import numpy as np
from scipy import stats

N, D, K, C0 = 30, 100, 3, 0.25
KILLS = []


def design(seed, noise=0.0, n_val=0, n_test=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, D))
    xs = np.zeros(D)
    idx = rng.choice(D, K, replace=False)
    xs[idx] = rng.choice([-1.0, 1.0], K)
    y = A @ xs + noise * rng.standard_normal(N)
    out = dict(A=A, y=y, xs=xs)
    for name, m in (("val", n_val), ("test", n_test)):
        if m:
            B = rng.standard_normal((m, D))
            out[name] = (B, B @ xs + noise * rng.standard_normal(m))
    return out


def constant(budget, tau):
    return lambda t: budget / tau if t < tau else 0.0


def early_burst(budget, tau):
    return lambda t: 5 * budget / tau if t < tau / 5 else 0.0


def late_burst(budget, tau):
    return lambda t: 5 * budget / tau if 4 * tau / 5 <= t < tau else 0.0


def ramp(budget, tau):
    return lambda t: 2 * budget / tau * (1 - t / tau) if t < tau else 0.0


def no_decay(t):
    return 0.0


def run(A, y, eta, alpha, c0, max_steps, tol, ckpt=None):
    """Gradient descent. Returns (x, u, v, steps, converged, ckpt_values)."""
    u = np.full(D, math.sqrt(c0))
    v = u.copy()
    n = len(y)
    ynorm = np.linalg.norm(y)
    At = A.T
    next_ckpt = 1
    vals = []
    for k in range(max_steps):
        x = u * u - v * v
        r = A @ x - y
        a = eta * alpha(k * eta)
        if ckpt is not None and k + 1 >= next_ckpt:
            vals.append((k, ckpt(x)))
            next_ckpt = max(k + 2, int(next_ckpt * 1.25))
        if tol is not None and a == 0.0 and np.linalg.norm(r) <= tol * ynorm:
            return x, u, v, k, True, vals
        g = At @ r / n
        z = 2 * eta * g
        u = u * (1 - a - z)
        v = v * (1 - a + z)
    return u * u - v * v, u, v, max_steps, False, vals


def closed_form(A, y, c):
    """Limit x = 2c sinh(A^T lam), A x = y. Damped Newton on the convex dual."""
    At = A.T
    lam = np.zeros(A.shape[0])

    def dual(l):
        s = At @ l
        with np.errstate(over="ignore"):
            return np.sum(2 * c * np.cosh(s)) - l @ y

    for _ in range(500):
        s = At @ lam
        r = A @ (2 * c * np.sinh(s)) - y
        if np.linalg.norm(r) <= 1e-11 * np.linalg.norm(y):
            return 2 * c * np.sinh(s)
        H = (A * (2 * c * np.cosh(s))) @ At
        step = np.linalg.solve(H + 1e-14 * np.trace(H) / len(H) * np.eye(len(H)), r)
        f0, t = dual(lam), 1.0
        slope = r @ step
        while t > 1e-12 and not dual(lam - t * step) <= f0 - 1e-4 * t * slope:
            t *= 0.5
        lam = lam - t * step
    raise RuntimeError("closed form did not converge")


def check(name, value, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + " = %.3e %s" % (value, detail), flush=True)
    return ok


def part_a():
    print("Part A: exact recursions", flush=True)
    d = design(0)
    A, y = d["A"], d["y"]
    eta, u, v = 0.01, np.full(D, math.sqrt(C0)), np.full(D, math.sqrt(C0))
    logc, phi = np.log(u * v), np.log(u / v)
    alpha = constant(1.5, 5.0)
    worst_c = worst_phi = worst_x = 0.0
    for k in range(2000):
        g = A.T @ (A @ (u * u - v * v) - y) / N
        a, z = eta * alpha(k * eta), 2 * eta * g
        logc = logc + np.log((1 - a) ** 2 - z * z)
        phi = phi + np.log((1 - a - z) / (1 - a + z))
        u, v = u * (1 - a - z), v * (1 - a + z)
        worst_c = max(worst_c, np.max(np.abs(logc - np.log(u * v))))
        worst_phi = max(worst_phi, np.max(np.abs(phi - np.log(u / v))))
        x = u * u - v * v
        worst_x = max(worst_x, np.max(np.abs(2 * np.exp(logc) * np.sinh(phi) - x)) / max(np.max(np.abs(x)), 1e-12))
    ok = check("A1_c_recursion", worst_c, worst_c < 1e-9)
    ok &= check("A2_phi_recursion", worst_phi, worst_phi < 1e-9)
    ok &= check("A3_x_equals_2c_sinh_phi", worst_x, worst_x < 1e-9)
    if not ok:
        KILLS.append("A_identities")


def part_b():
    print("Part B: schedule invariance, budget 1.5", flush=True)
    budget, tau = 1.5, 5.0
    c_pred = C0 * math.exp(-2 * budget)
    scheds = dict(constant=constant, early=early_burst, late=late_burst, ramp=ramp)
    etas = [0.01, 0.005, 0.0025]
    table = {}
    for seed in range(3):
        d = design(seed)
        A, y = d["A"], d["y"]
        xcf = closed_form(A, y, c_pred)
        for name, sched in scheds.items():
            errs = []
            for eta in etas:
                x, _, _, steps, conv, _ = run(A, y, eta, sched(budget, tau), C0, int(1500 / eta), 1e-10)
                errs.append(np.linalg.norm(x - xcf) / np.linalg.norm(xcf) if conv else math.nan)
            table[(seed, name)] = errs
            print(f"  design={seed} {name:9s} rel err at eta {etas}: " + " ".join(f"{e:.2e}" for e in errs), flush=True)
    fine = [v[-1] for v in table.values()]
    slopes = [np.polyfit(np.log(etas), np.log(v), 1)[0] for v in table.values() if all(np.isfinite(e) and e > 0 for e in v)]
    worst = max(fine) if all(np.isfinite(fine)) else math.inf
    ok = check("B1_worst_err_smallest_eta", worst, worst <= 0.02, "limit 0.02")
    ms = float(np.median(slopes)) if slopes else math.nan
    ok &= check("B2_median_slope", ms, ms >= 0.5, f"min {min(slopes) if slopes else math.nan:.2f}, limit 0.5")
    if not ok:
        KILLS.append("B_invariance")


def part_c():
    print("Part C: steps to the same limit, noiseless, eta = 0.01", flush=True)
    eta, tol, max_steps = 0.01, 1e-6, 400000
    for budget in (1.5, 3.0):
        c_t = C0 * math.exp(-2 * budget)
        ratios, void = [], 0
        small_steps, best_steps = [], []
        for seed in range(100, 120):
            d = design(seed)
            A, y = d["A"], d["y"]
            xcf = closed_form(A, y, c_t)
            xs, _, _, s_small, conv, _ = run(A, y, eta, no_decay, c_t, max_steps, tol)
            if not conv or np.linalg.norm(xs - xcf) > 0.02 * np.linalg.norm(xcf):
                void += 1
                continue
            best = None
            for sched in (constant, early_burst):
                for tau in (1.0, 5.0, 25.0):
                    x, _, _, st, conv, _ = run(A, y, eta, sched(budget, tau), C0, max_steps, tol)
                    if conv and np.linalg.norm(x - xcf) <= 0.02 * np.linalg.norm(xcf):
                        best = st if best is None else min(best, st)
            if best is None:
                void += 1
                continue
            ratios.append(best / s_small)
            small_steps.append(s_small)
            best_steps.append(best)
        if not ratios:
            print(f"  budget={budget}: all designs void, so this comparison is void", flush=True)
            continue
        med = float(np.median(ratios))
        print(f"  budget={budget}: designs used {len(ratios)}, void {void}; median steps small init {np.median(small_steps):.0f}, "
              f"best decay {np.median(best_steps):.0f}; ratio median {med:.3f}, min {min(ratios):.3f}, max {max(ratios):.3f}", flush=True)
        if not check(f"C_ratio_budget_{budget}", med, med >= 0.8, "kill below 0.8"):
            KILLS.append(f"C_speed_budget_{budget}")
        if void > 5:
            print(f"  WARNING: {void} of 20 designs void at budget {budget}; treat this comparison as unreliable", flush=True)


def part_d():
    print("Part D: noisy test error, chosen on validation", flush=True)
    eta, steps = 0.01, 40000
    grid = [1.0, 0.25, 0.06, 0.015, 0.004, 0.001]
    diffs, small_all, decay_all, picks = [], [], [], []
    for seed in range(200, 220):
        d = design(seed, noise=0.1, n_val=30, n_test=1000)
        results = {}
        for arm in ("small", "decay"):
            best = (math.inf, math.nan, None)
            for c in grid:
                if arm == "decay" and c >= C0:
                    if c > C0:
                        continue
                    alpha, c0 = no_decay, C0
                elif arm == "decay":
                    alpha, c0 = constant(0.5 * math.log(C0 / c), 5.0), C0
                else:
                    alpha, c0 = no_decay, c
                Bv, yv = d["val"]
                Bt, yt = d["test"]
                ck = lambda x: (float(np.mean((Bv @ x - yv) ** 2)), float(np.mean((Bt @ x - yt) ** 2)))
                _, _, _, _, _, vals = run(d["A"], d["y"], eta, alpha, c0, steps, None, ck)
                for k, (ve, te) in vals:
                    if arm == "decay" and alpha is not no_decay and k * eta < 5.0:
                        continue  # the arm is decay-then-off, so only checkpoints after the budget is spent
                    if ve < best[0]:
                        best = (ve, te, c)
            results[arm] = best
        small_all.append(results["small"][1])
        decay_all.append(results["decay"][1])
        diffs.append(results["decay"][1] - results["small"][1])
        picks.append((results["small"][2], results["decay"][2]))
    m, s = float(np.mean(diffs)), float(np.std(diffs, ddof=1))
    h = stats.t.ppf(0.975, len(diffs) - 1) * s / math.sqrt(len(diffs))
    print(f"  test MSE small init {np.mean(small_all):.5f}, decay {np.mean(decay_all):.5f}; noise floor 0.01")
    print(f"  decay - small = {m:+.5f} [{m - h:+.5f}, {m + h:+.5f}] over {len(diffs)} designs", flush=True)
    print(f"  chosen c, small init: {sorted(p[0] for p in picks)}")
    print(f"  chosen c, decay:      {sorted(p[1] for p in picks)}", flush=True)
    if m + h < 0:
        KILLS.append("D_decay_better")
        print("FAIL D_no_advantage: decay better, interval excludes zero")
    else:
        print("PASS D_no_advantage: decay is not better with an interval excluding zero")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_d()
    print("KILLS:", KILLS if KILLS else "none", flush=True)
    sys.exit(1 if KILLS else 0)
