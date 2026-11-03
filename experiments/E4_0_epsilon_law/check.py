# E4.0: float64 checks for H4 (Adam epsilon in the scale-invariant schedule law).
# Part A: exact identities on a synthetic scale-invariant logistic loss (numpy).
# Part B: the same one-step identities against torch.optim.AdamW on a BN MLP (skipped if no torch).
# Part C: mean-field closed forms on an isotropic noise walk (approximate, with tolerances).
# Plain script. Exit code 1 if anything fails. Last line of stdout is E4_0_JSON=...
import json, math, sys
import numpy as np

RES = {}
FAILED = []


def check(name, value, ok, detail=""):
    RES[name] = dict(value=float(value), ok=bool(ok))
    if not ok:
        FAILED.append(name)
    print(("PASS " if ok else "FAIL ") + name + " = %.3e %s" % (value, detail))


# ---------- Part A: problem and runners ----------
class Problem:
    def __init__(self, d=50, n=512, batch=64, T=2000, seed=0):
        rng = np.random.default_rng(seed)
        self.d = d
        self.X = rng.standard_normal((n, d))
        truth = rng.standard_normal(d)
        y = np.sign(self.X @ truth)
        flip = rng.random(n) < 0.1
        y[flip] = -y[flip]
        self.y = y
        self.idx = rng.integers(0, n, size=(T, batch))
        self.w0 = rng.standard_normal(d)
        self.w0 /= np.linalg.norm(self.w0)

    def gbar(self, u, t, scale=1.0):
        # Tangent gradient at unit u of mean log(1+exp(-y x.u)): scale-free by construction.
        b = self.idx[t]
        Xb, yb = self.X[b], self.y[b]
        z = yb * (Xb @ u)
        h = (-(yb / (1.0 + np.exp(z))))[:, None] * Xb
        h = h.mean(axis=0)
        return scale * (h - (h @ u) * u)


def adamw_direct(P, w0, etas, lams, eps, b1, b2, T, scale=1.0):
    W = np.zeros((T + 1, P.d))
    W[0] = w0
    PS, MH, VH, GB = [np.zeros((T, P.d)) for _ in range(4)]
    m = np.zeros(P.d)
    v = np.zeros(P.d)
    for t in range(T):
        w = W[t]
        r = np.linalg.norm(w)
        gb = P.gbar(w / r, t, scale)
        g = gb / r
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** (t + 1))
        vh = v / (1 - b2 ** (t + 1))
        p = mh / (np.sqrt(vh) + eps)
        W[t + 1] = (1 - etas[t] * lams[t]) * w - etas[t] * p
        PS[t], MH[t], VH[t], GB[t] = p, mh, vh, gb
    return dict(W=W, p=PS, mh=MH, vh=VH, gb=GB)


def closed_run(P, w0, etas, lams, eps, b1, b2, T):
    # Identity C: no radius anywhere; epsilon enters only through E_t = eps eta_t / a_t.
    a = 1 - etas * lams
    r0 = np.linalg.norm(w0)
    u = w0 / r0
    Psi = etas[0] / (a[0] * r0)
    mt = np.zeros(P.d)
    vt = np.zeros(P.d)
    rho = 1.0
    U = np.zeros((T + 1, P.d))
    PSI = np.zeros(T + 1)
    U[0], PSI[0] = u, Psi
    for t in range(T):
        gb = P.gbar(u, t)
        mt = b1 * rho * mt + (1 - b1) * gb
        vt = b2 * rho ** 2 * vt + (1 - b2) * gb * gb
        E = eps * etas[t] / a[t]
        D = np.sqrt(vt / (1 - b2 ** (t + 1))) + E / Psi
        p = (mt / (1 - b1 ** (t + 1))) / D
        y = u - Psi * p
        N = np.linalg.norm(y)
        u = y / N
        Psi = (etas[t + 1] / etas[t]) / a[t + 1] * Psi / N
        rho = a[t] * N
        U[t + 1], PSI[t + 1] = u, Psi
    return U, PSI


def schedule(kind, eta0, T):
    t = np.arange(T + 1)
    if kind == "const":
        return eta0 * np.ones(T + 1)
    if kind == "cosine":
        return eta0 * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * t / T)))
    if kind == "step":
        return eta0 * np.where(t < T // 2, 1.0, 0.1)
    raise ValueError(kind)


def part_a():
    b1, b2 = 0.9, 0.999
    lam0 = 0.05
    P = Problem()
    EPS = [0.0, 1e-8, 1e-3, 1e-1, 10.0]
    # K, D, R, C across epsilons and schedules
    wK = wD = wR = wCp = wCu = 0.0
    for kind in ["const", "cosine", "step"]:
        for eps in EPS:
            # Large eps barely moves u at small eta, so scale eta with (1 + eps) to exercise it.
            T = 300
            etas = schedule(kind, 0.01 * (1 + eps), T)
            lams = lam0 / (1 + eps) * np.ones(T + 1)
            a = 1 - etas * lams
            R = adamw_direct(P, P.w0, etas, lams, eps, b1, b2, T)
            W = R["W"]
            r = np.linalg.norm(W, axis=1)
            U = W / r[:, None]
            Phi = etas / (a * r ** 2)
            Psi = etas / (a * r)
            cosines = np.sum(U[:-1] * U[1:], axis=1)
            bud_rot = bud_rad = 0.0
            for t in range(T):
                p = R["p"][t]
                N = np.linalg.norm(U[t] - Psi[t] * p)
                B = (etas[t + 1] / etas[t]) / (a[t] * a[t + 1])
                Bt = (etas[t + 1] / etas[t]) / a[t + 1]
                wK = max(wK, abs(Phi[t + 1] / (B * Phi[t] / N ** 2) - 1))
                wK = max(wK, abs(Psi[t + 1] / (Bt * Psi[t] / N) - 1))
                c = U[t] @ p
                s = p @ p
                wK = max(wK, abs(N ** 2 - (1 - 2 * Psi[t] * c + Psi[t] ** 2 * s)))
                bud_rot += -math.log(cosines[t])
                bud_rad += -math.log(1 - Psi[t] * c)
                if eps > 0:
                    sv = np.sqrt(R["vh"][t])
                    alpha = sv / (sv + eps)
                    pbar = R["mh"][t] / sv
                    wD = max(wD, np.max(np.abs(p - alpha * pbar)) / np.max(np.abs(p)))
                    # 1 - alpha written as eps/(sv+eps): subtracting from 1 loses digits at small eps
                    wD = max(wD, np.max(np.abs(p - (eps / (sv + eps)) * R["mh"][t] / eps)) / np.max(np.abs(p)))
            bud = math.log(r[T] / r[0]) - np.sum(np.log(a[:T])) + bud_rad
            wR = max(wR, abs(bud_rot - bud))
            Uc, Pc = closed_run(P, P.w0, etas, lams, eps, b1, b2, T)
            wCp = max(wCp, np.max(np.abs(Pc / Psi - 1)))
            wCu = max(wCu, np.max(np.linalg.norm(Uc - U, axis=1)))
    check("A1_kinematics_K", wK, wK < 1e-11)
    check("A2_decomposition_D", wD, wD < 1e-11)
    check("A3_rotation_budget_R", wR, wR < 1e-10)
    check("A4_closure_C_Psi", wCp, wCp < 1e-8)
    check("A4_closure_C_u", wCu, wCu < 1e-8)

    # S: symmetry group
    T = 300
    eta, lam, eps = 0.01, 0.05, 1e-3
    etas = schedule("cosine", eta, T)
    lams = lam * np.ones(T + 1)
    A = adamw_direct(P, P.w0, etas, lams, eps, b1, b2, T)
    c = 7.3
    Bc = adamw_direct(P, c * P.w0, c * etas, lams / c, eps / c, b1, b2, T)
    rA = np.linalg.norm(A["W"], axis=1)
    rB = np.linalg.norm(Bc["W"], axis=1)
    du = np.max(np.linalg.norm(A["W"] / rA[:, None] - Bc["W"] / rB[:, None], axis=1))
    dr = np.max(np.abs(rB / (c * rA) - 1))
    check("A5_symmetry_scale_u", du, du < 1e-8)
    check("A5_symmetry_scale_r", dr, dr < 1e-8)
    s = 10.0
    Bs = adamw_direct(P, P.w0, etas, lams, eps * s, b1, b2, T, scale=s)
    dw = np.max(np.linalg.norm(A["W"] - Bs["W"], axis=1))
    check("A6_symmetry_lossscale", dw, dw < 1e-8)

    # Limits
    T = 30
    eta, lam = 0.01, 0.05
    etas = eta * np.ones(T + 1)
    lams = lam * np.ones(T + 1)
    W0 = adamw_direct(P, P.w0, etas, lams, 0.0, b1, b2, T)["W"]
    grid = np.array([1e-4, 1e-5, 1e-6, 1e-7])
    err = [np.max(np.linalg.norm(adamw_direct(P, P.w0, etas, lams, e, b1, b2, T)["W"] - W0, axis=1)) for e in grid]
    sl = np.polyfit(np.log(grid), np.log(err), 1)[0]
    check("A7_limit_eps0_slope", sl, 0.8 <= sl <= 1.2, "expect +1")

    def sgd_ema(b, etap, lamp):
        w = P.w0.copy()
        out = [w.copy()]
        m = np.zeros(P.d)
        for t in range(T):
            r = np.linalg.norm(w)
            g = P.gbar(w / r, t) / r
            m = b * m + (1 - b) * g
            w = (1 - etap * lamp) * w - etap * m / (1 - b ** (t + 1))
            out.append(w.copy())
        return np.array(out)

    Wref = sgd_ema(b1, eta, lam)
    grid = np.array([1e2, 1e3, 1e4, 1e5])
    err = []
    for e in grid:
        Wa = adamw_direct(P, P.w0, e * etas, lams / e, e, b1, b2, T)["W"]
        err.append(np.max(np.linalg.norm(Wa - Wref, axis=1)))
    sl = np.polyfit(np.log(grid), np.log(err), 1)[0]
    check("A8_limit_epsinf_slope", sl, -1.2 <= sl <= -0.8, "expect -1, to SGD with eta/eps, lam*eps")

    # Theorem 2.1 of the paper on the large-eps reference at b1 = 0
    Wr0 = sgd_ema(0.0, eta, lam)
    a = 1 - eta * lam
    r = np.linalg.norm(Wr0, axis=1)
    Phi = eta / (a * r ** 2)
    worst = 0.0
    for t in range(T):
        gb = P.gbar(Wr0[t] / r[t], t)
        pred = (1 / a ** 2) * Phi[t] / (1 + Phi[t] ** 2 * (gb @ gb))
        worst = max(worst, abs(Phi[t + 1] / pred - 1))
    check("A9_paper_thm2_1_at_large_eps", worst, worst < 1e-11)


# ---------- Part B: real torch.optim.AdamW ----------
def part_b():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("SKIP part B: torch not installed. Part B is NOT verified.")
        RES["B_skipped"] = dict(value=1.0, ok=True)
        return
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    eta, lam, b1, b2 = 0.01, 0.05, 0.9, 0.999
    a = 1 - eta * lam
    worst = dict(step=0.0, N=0.0, Phi=0.0, Psi=0.0, inv=0.0, tan=0.0)
    for eps in [1e-8, 1e-3, 1.0]:
        fc1 = nn.Linear(20, 16, bias=False)
        bn = nn.BatchNorm1d(16, eps=0.0, affine=False, track_running_stats=False)
        fc2 = nn.Linear(16, 3)
        params = list(fc1.parameters()) + list(fc2.parameters())
        opt = torch.optim.AdamW(params, lr=eta, betas=(b1, b2), eps=eps, weight_decay=lam)
        X = torch.randn(256, 20)
        y = torch.randint(0, 3, (256,))

        def grads_at(w, xb, yb):
            fc1.weight.data.copy_(w)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(fc2(nn.functional.softplus(bn(fc1(xb)))), yb)
            loss.backward()
            return fc1.weight.grad.detach().clone()

        m = torch.zeros_like(fc1.weight)
        v = torch.zeros_like(fc1.weight)
        for t in range(200):
            b = torch.randint(0, 256, (64,))
            xb, yb = X[b], y[b]
            w = fc1.weight.detach().clone()
            if t % 50 == 0:
                g0 = grads_at(w, xb, yb)
                w3 = w.clone()
                w3[0] *= 3.0
                g3 = grads_at(w3, xb, yb)
                worst["inv"] = max(worst["inv"], float((3 * g3[0] - g0[0]).norm() / g0[0].norm()))
                worst["inv"] = max(worst["inv"], float((g3[1:] - g0[1:]).norm() / g0[1:].norm()))
                worst["tan"] = max(worst["tan"], float((w[0] @ g0[0]).abs() / (w[0].norm() * g0[0].norm())))
            g = grads_at(w, xb, yb)  # leaves the unscaled weights and matching .grad in place
            opt.step()
            w1 = fc1.weight.detach().clone()
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            s = t + 1
            p = (m / (1 - b1 ** s)) / (torch.sqrt(v / (1 - b2 ** s)) + eps)
            worst["step"] = max(worst["step"], float((a * w - eta * p - w1).abs().max() / w1.abs().max()))
            for i in range(w.shape[0]):
                r = w[i].norm()
                r1 = w1[i].norm()
                u = w[i] / r
                Psi = eta / (a * r)
                N = (u - Psi * p[i]).norm()
                worst["N"] = max(worst["N"], float((r1 / (a * r * N) - 1).abs()))
                Phi, Phi1 = eta / (a * r ** 2), eta / (a * r1 ** 2)
                worst["Phi"] = max(worst["Phi"], float((Phi1 / ((1 / a ** 2) * Phi / N ** 2) - 1).abs()))
                worst["Psi"] = max(worst["Psi"], float(((eta / (a * r1)) / ((1 / a) * Psi / N) - 1).abs()))
    check("B1_torch_step_matches_formula", worst["step"], worst["step"] < 1e-12)
    check("B2_torch_N", worst["N"], worst["N"] < 1e-11)
    check("B3_torch_Phi_law", worst["Phi"], worst["Phi"] < 1e-11)
    check("B4_torch_Psi_law", worst["Psi"], worst["Psi"] < 1e-11)
    check("B5_bn_scale_invariance", worst["inv"], worst["inv"] < 1e-9)
    check("B6_bn_tangent_gradient", worst["tan"], worst["tan"] < 1e-9)


# ---------- Part C: mean-field closed forms on an isotropic noise walk ----------
class Walk:
    def __init__(self, M, d, eta, lam, eps, gb, b1, b2, r0, seed):
        self.rng = np.random.default_rng(seed)
        self.d, self.M = d, M
        self.eta, self.eps, self.gb, self.b1, self.b2 = eta, eps, gb, b1, b2
        self.a = 1 - eta * lam
        w = self.rng.standard_normal((M, d))
        self.w = r0 * w / np.linalg.norm(w, axis=1, keepdims=True)
        self.m = np.zeros((M, d))
        self.v = (gb / r0) ** 2 * np.ones((M, d))

    def step(self):
        r = np.linalg.norm(self.w, axis=1, keepdims=True)
        u = self.w / r
        xi = self.rng.standard_normal((self.M, self.d))
        xi = xi - np.sum(xi * u, axis=1, keepdims=True) * u
        g = self.gb * xi / r
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        self.w = self.a * self.w - self.eta * self.m / (np.sqrt(self.v) + self.eps)
        r1 = np.linalg.norm(self.w, axis=1, keepdims=True)
        th2 = np.mean(np.sum((self.w / r1 - u) ** 2, axis=1))
        return float(np.mean(r1)), float(th2)


def part_c():
    d, M, eta, lam, gb, b1, b2 = 1000, 8, 1e-3, 0.1, 0.01, 0.9, 0.99
    R0 = math.sqrt(eta * d / (2 * lam - eta * lam ** 2))
    th_pred = math.sqrt(2 * eta * lam * (1 - b1) / (1 + b1))
    for x in [0.0, 1.0, 10.0]:
        eps = x * (1 + x) * gb / R0
        req = x * gb / eps if x > 0 else R0
        W = Walk(M, d, eta, lam, eps, gb, b1, b2, req, seed=int(10 * x) + 1)
        for _ in range(10000):
            W.step()
        rs, ths = [], []
        for _ in range(3000):
            r, th2 = W.step()
            rs.append(r)
            ths.append(th2)
        rm = float(np.mean(rs))
        th = math.sqrt(float(np.mean(ths)))
        e = abs(rm / req - 1)
        check("C1_r_eq_x%g" % x, e, e < 0.05, "r=%.4f pred=%.4f" % (rm, req))
        e = abs(th / th_pred - 1)
        check("C2_theta_eq_x%g" % x, e, e < 0.05, "theta=%.4e pred=%.4e (independent of eps)" % (th, th_pred))
        W.w = 1.3 * W.w
        ys = []
        for _ in range(9000):
            r, _th = W.step()
            ys.append(math.log(r / rm))
        ys = np.array(ys)
        lo = int(np.argmax(ys < 0.2))
        hi = int(np.argmax(ys < 0.06))
        if hi <= lo + 50:
            hi = len(ys)
        slope = np.polyfit(np.arange(lo, hi), np.log(np.clip(ys[lo:hi], 1e-9, None)), 1)[0]
        om_pred = 2 * eta * lam * (1 + 2 * x) / (1 + x)
        e = abs(-slope / om_pred - 1)
        check("C3_omega_x%g" % x, e, e < 0.15, "omega=%.3e pred=%.3e" % (-slope, om_pred))

    # Transit law without decay
    T_poly = lambda z: z ** 2 + (4.0 / 3) * z ** 3 + z ** 4 / 2
    eta, gb, eps, M = 1e-3, 0.01, 0.01, 4
    rstar = gb / eps
    x0 = 0.5
    W = Walk(M, d, eta, 0.0, eps, gb, b1, b2, x0 * rstar, seed=99)
    targets = [1.0, 2.0, 3.0]
    times = [int((T_poly(z) - T_poly(x0)) * rstar ** 2 / (eta ** 2 * d)) for z in targets]
    k = 0
    for t in range(1, times[-1] + 1):
        r, _ = W.step()
        if t == times[k]:
            e = abs((r / rstar) / targets[k] - 1)
            check("C4_transit_x%g" % targets[k], e, e < 0.05, "t=%d x=%.3f" % (t, r / rstar))
            k += 1


if __name__ == "__main__":
    print("Part A: exact identities")
    part_a()
    print("Part B: torch.optim.AdamW")
    part_b()
    print("Part C: mean-field closed forms")
    part_c()
    print("FAILED: %s" % (FAILED if FAILED else "none"))
    print("E4_0_JSON=" + json.dumps(dict(failed=FAILED, results=RES)))
    sys.exit(1 if FAILED else 0)
