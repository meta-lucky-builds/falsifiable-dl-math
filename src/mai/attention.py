"""Multi-head attention with a selectable row normalizer, for H1 / E1.1.

Every normalizer has the form  w = pi(beta_row * z),  with z = q.k the raw score,
beta_row an inverse temperature in raw-score units, and pi either softmax or
entmax-1.5. `row_beta` therefore gives a directly comparable temperature for all
arms, which is what the E1.2 audit needs.

    softmax          softmax(z / sqrt(d))
    learned_scale    softmax(c_h z / sqrt(d))                      c_h per head
    ssmax            softmax(s_h log n z / sqrt(d))                2501.19399 eq. 11
    entmax15         entmax15(z / sqrt(d))
    entmax15_scaled  entmax15((delta + b (log n)^gamma) z/sqrt(d)) 2506.16640 eq. 8,
                     b = softplus(.), gamma = 4 tanh(.), all per head. This is the
                     paper's direct-scalar variant, not its content-dependent one.
    rem              beta*(mu_hat, sigma_hat, n, kappa), detached  H1
    calib            bisected beta with top mass = p, detached     E1.0 RESULTS

Conventions: scores are (B, heads, T, S). Masks are bool with True = valid. A row
with no valid entry returns all zeros, because no distribution exists on it.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NORMALIZERS = ("softmax", "learned_scale", "ssmax", "entmax15", "entmax15_scaled", "rem", "calib")

# Plug-in rules keep beta * (zmax - zmin) below this. The cap is scale-invariant and
# only binds on rows whose top two scores nearly tie (gap/spread below about 1e-3),
# where beta would otherwise diverge and dominate the batch gradient.
SPREAD_CAP = 1e4
GAMMA_BOUND = 4.0
_TINY = 1e-30


def _prepare(z, mask):
    """Remove masked values from z, and make empty rows safe to compute on."""
    mask = torch.ones_like(z, dtype=torch.bool) if mask is None else mask.expand_as(z)
    empty = ~mask.any(-1, keepdim=True)
    mask = mask | empty
    z = z.masked_fill(~mask, 0.0).masked_fill(empty, 0.0)
    return z, mask, empty


@torch.no_grad()
def score_stats(z, mask):
    """Row statistics from unmasked entries only. All outputs keep a singleton last dim.

    mean and var exclude exactly one instance of the row max (ties keep the others).
    var uses the m - 1 denominator and is 0 when m < 2, where m = n - 1.
    `degenerate` marks rows where beta is meaningless: n < 2, or spread at float noise.
    """
    n = mask.sum(-1, keepdim=True).to(z.dtype)
    zmax, top = z.masked_fill(~mask, -math.inf).max(-1, keepdim=True)
    zmin = z.masked_fill(~mask, math.inf).min(-1, keepdim=True).values
    rest = mask.clone()
    rest.scatter_(-1, top, torch.zeros_like(top, dtype=torch.bool))
    m = rest.sum(-1, keepdim=True).to(z.dtype)
    mean = z.masked_fill(~rest, 0.0).sum(-1, keepdim=True) / m.clamp(min=1.0)
    dev = (z - mean).masked_fill(~rest, 0.0)
    var = dev.pow(2).sum(-1, keepdim=True) / (m - 1.0).clamp(min=1.0)
    spread = zmax - zmin
    ref = z.masked_fill(~mask, 0.0).abs().max(-1, keepdim=True).values
    floor = 8.0 * torch.finfo(z.dtype).eps * ref + 1e-12
    degenerate = (n < 2) | (spread <= floor)
    return dict(n=n, m=m, zmax=zmax, top=top, rest=rest, mean=mean, dev=dev, var=var,
                mu=zmax - mean, spread=spread, degenerate=degenerate)


@torch.no_grad()
def row_audit(z, mask=None):
    """E1.2 row statistics, using the same estimates as rem_beta. NaN where undefined."""
    z, mask, _ = _prepare(z, mask)
    s = score_stats(z, mask)
    pos = s["var"] > 0
    nan = z.new_full(s["var"].shape, math.nan)
    m2 = s["dev"].pow(2).sum(-1, keepdim=True) / s["m"].clamp(min=1.0)
    m4 = s["dev"].pow(4).sum(-1, keepdim=True) / s["m"].clamp(min=1.0)
    kurt = torch.where(pos & (s["m"] >= 4), m4 / m2.clamp(min=_TINY).pow(2) - 3.0, nan)
    ratio = torch.where(pos, s["mu"] / s["var"].clamp(min=_TINY).sqrt(), nan)
    return dict(mean=s["mean"], std=s["var"].sqrt(), exkurt=kurt, mu_over_sigma=ratio, n=s["n"])


@torch.no_grad()
def rem_beta(z, mask, kappa):
    """H1 beta* per row. Returns (beta, d beta / d kappa). Assumes each row has a valid entry.

    With d = mu^2 - 2 sigma^2 L:
      d > 0:  beta = (mu - sqrt(d)) / sigma^2 = 2 L / (mu + sqrt(d)). The second form is
              algebraically identical and stays finite as sigma -> 0, including n = 2.
      d <= 0: beta = mu / sigma^2, the clamped branch; sigma^2 >= mu^2 / (2L) > 0 here.
    Identical scores and n < 2 give beta = 0, so the row is uniform. Nothing takes a
    gradient through sqrt, so sqrt(0) cannot produce a NaN.
    """
    s = score_stats(z, mask)
    kappa = torch.as_tensor(kappa, dtype=z.dtype, device=z.device)
    L = (torch.log(s["n"].clamp(min=1.0)) + kappa).clamp(min=0.0)
    mu, var = s["mu"], s["var"]
    disc = mu * mu - 2.0 * var * L
    root = disc.clamp(min=0.0).sqrt()
    above = disc > 0
    beta = torch.where(above, 2.0 * L / (mu + root).clamp(min=_TINY), mu / var.clamp(min=_TINY))
    beta = beta.min(SPREAD_CAP / s["spread"].clamp(min=_TINY))
    beta = beta.masked_fill(s["degenerate"], 0.0)
    # d beta / d L = 1 / sqrt(d) above threshold and 0 below. Bounded near threshold.
    slope = 1.0 / root.max(0.01 * mu).clamp(min=_TINY)
    slope = slope.masked_fill(~above | s["degenerate"] | (L <= 0), 0.0)
    return beta, slope


@torch.no_grad()
def calib_beta(z, mask, p, iters=20):
    """Smallest beta whose softmax(beta z) puts mass >= p on the row max. No distribution assumption.

    Mass rises monotonically in beta, and these bounds bracket the crossing, with
    T = log((n-1) p / (1-p)), g the top gap and D the spread:  T/D <= beta <= T/g.
    T <= 0 means p <= 1/n, so beta = 0 already meets the target. Bisection is on log beta
    because D/g can be large. A top tie or the spread cap can make p unreachable; the
    result is then the capped beta, which is the sharpest allowed.
    """
    s = score_stats(z, mask)
    rest = s["rest"]
    d = (s["zmax"] - z).masked_fill(~rest, 0.0)
    target = torch.log(s["m"].clamp(min=1.0)) + math.log(p / (1.0 - p))
    active = ~s["degenerate"] & (target > 0)
    spread = s["spread"].clamp(min=_TINY)
    gap = d.masked_fill(~rest, math.inf).min(-1, keepdim=True).values.clamp(min=_TINY)
    cap = SPREAD_CAP / spread
    one = torch.ones_like(target)
    hi = torch.where(active, (target / gap).min(cap), one)
    lo = torch.where(active, (target / spread).min(hi), one)
    log_lo, log_hi = lo.log(), hi.log()
    limit = (1.0 - p) / p
    for _ in range(iters):
        mid = 0.5 * (log_lo + log_hi)
        others = (torch.exp(-mid.exp() * d) * rest).sum(-1, keepdim=True)
        reached = others <= limit
        log_hi = torch.where(reached, mid, log_hi)
        log_lo = torch.where(reached, log_lo, mid)
    return log_hi.exp().masked_fill(~active, 0.0)


def entmax15(logits, mask=None):
    """Exact alpha = 1.5 entmax along the last dim; differentiable and mask-aware.

    Solution: p_i = (x_i - tau)_+^2 with x = logits/2 and sum_i p_i = 1. The support and
    tau0 come from the sort-based solver of Peters et al. 2019, without gradients. tau
    then takes one Newton step on F(tau) = sum_S (x_i - tau)^2 - 1, written in terms of
    the differentiable x. The value is unchanged, since F(tau0) = 0, and the gradient
    equals the implicit-function result d tau / d x_j = (x_j - tau) / sum_S (x_i - tau).
    The denominator is >= 1 by Cauchy-Schwarz, so unlike differentiating sqrt(delta)
    this cannot produce inf or 0 * inf. Each row must have a valid entry.
    """
    mask = torch.ones_like(logits, dtype=torch.bool) if mask is None else mask.expand_as(logits)
    top = logits.masked_fill(~mask, -math.inf).max(-1, keepdim=True).values.detach()
    x = (logits - top) / 2.0
    with torch.no_grad():
        xs = x.masked_fill(~mask, -math.inf).sort(-1, descending=True).values
        valid = torch.isfinite(xs)
        x0 = xs.masked_fill(~valid, 0.0)
        rho = torch.arange(1, x.shape[-1] + 1, dtype=x.dtype, device=x.device)
        mean = x0.cumsum(-1) / rho
        mean_sq = (x0 * x0).cumsum(-1) / rho
        delta = ((1.0 - rho * (mean_sq - mean * mean)) / rho).clamp(min=0.0)
        tau_all = mean - delta.sqrt()
        k = ((tau_all <= xs) & valid).sum(-1, keepdim=True).clamp(min=1)
        tau0 = tau_all.gather(-1, k - 1).clamp(max=-1e-12)  # the row max (x = 0) stays in support
        support = (x.detach() > tau0) & mask
        denom = (x.detach() - tau0).masked_fill(~support, 0.0).sum(-1, keepdim=True).clamp(min=1e-6)
    residual = (x - tau0).masked_fill(~support, 0.0).pow(2).sum(-1, keepdim=True) - 1.0
    tau = tau0 + residual / (2.0 * denom)
    return (x - tau).clamp(min=0.0).pow(2).masked_fill(~support, 0.0)


class MultiHeadAttention(nn.Module):
    """Linear layers come first and normalizer parameters are constants, so a given seed
    gives identical shared weights across arms, which pairs the initializations.

    train_n sets the initial s, delta, b and gamma so that ssmax and entmax15_scaled start
    with the same beta as softmax and entmax15 at the training length (the reciprocal-of-
    log-n init of 2501.19399 arms e and f). Neither arm gets a head start.
    """

    def __init__(self, d_model, n_heads, normalizer="softmax", train_n=8, kappa=math.log(9.0),
                 learn_kappa=False, calib_p=0.9, calib_iters=20):
        super().__init__()
        if normalizer not in NORMALIZERS:
            raise ValueError(f"unknown normalizer {normalizer!r}")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.h, self.dh, self.normalizer = n_heads, d_model // n_heads, normalizer
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.calib_p, self.calib_iters, self.learn_kappa = calib_p, calib_iters, learn_kappa
        self.capture, self.last = False, None

        def per_head(value):
            return nn.Parameter(torch.full((n_heads, 1, 1), float(value)))

        log_train_n = math.log(max(train_n, 2))
        if normalizer == "learned_scale":
            self.c = per_head(1.0)
        elif normalizer == "ssmax":
            self.s = per_head(1.0 / log_train_n)
        elif normalizer == "entmax15_scaled":
            self.delta = per_head(0.0)
            self.b_raw = per_head(math.log(math.expm1(1.0 / log_train_n)))  # softplus^-1
            self.gamma_raw = per_head(math.atanh(1.0 / GAMMA_BOUND))  # gamma = 1 at init
        elif normalizer == "rem":
            if learn_kappa:
                self.kappa = per_head(kappa)
            else:
                self.register_buffer("kappa", torch.full((n_heads, 1, 1), float(kappa)))

    def n_extra_params(self):
        return sum(p.numel() for p in self.parameters(recurse=False))

    def learned(self):
        """Normalizer parameters in their interpretable form, per head."""
        out = {n: p.detach().flatten().tolist() for n, p in self.named_parameters(recurse=False)}
        if self.normalizer == "entmax15_scaled":
            out["b"] = F.softplus(self.b_raw.detach()).flatten().tolist()
            out["gamma"] = (GAMMA_BOUND * torch.tanh(self.gamma_raw.detach())).flatten().tolist()
        return out

    def _scale(self, z, mask):
        n = mask.sum(-1, keepdim=True).to(z.dtype)
        base = 1.0 / math.sqrt(self.dh)
        name = self.normalizer
        if name in ("softmax", "entmax15"):
            return z.new_full(n.shape, base)
        if name == "learned_scale":
            return self.c * base
        if name == "ssmax":
            return self.s * torch.log(n.clamp(min=1.0)) * base  # n = 1 gives 0; the row is trivial
        if name == "entmax15_scaled":
            # n floored at 2: (log 1)^gamma is singular for gamma <= 0, and a one-token
            # row has weight 1 for any scale, so nothing is lost.
            log_n = torch.log(n.clamp(min=2.0))
            b, gamma = F.softplus(self.b_raw), GAMMA_BOUND * torch.tanh(self.gamma_raw)
            return (self.delta + b * log_n.pow(gamma)) * base
        if name == "rem":
            beta, slope = rem_beta(z, mask, self.kappa)
            if self.learn_kappa:  # beta stays detached from z; only kappa carries a gradient
                beta = beta + (self.kappa - self.kappa.detach()) * slope
            return beta
        return calib_beta(z, mask, self.calib_p, self.calib_iters)

    def row_beta(self, z, mask=None):
        """Inverse temperature per row (or per head), in raw-score units."""
        z, mask, _ = _prepare(z, mask)
        return self._scale(z, mask)

    def normalize(self, z, mask=None):
        z, mask, empty = _prepare(z, mask)
        beta = self._scale(z, mask)
        # Centre at the detached row max first: all rules are shift invariant, and this
        # avoids both precision loss and overflow in beta * z when scores carry a big offset.
        zmax = z.masked_fill(~mask, -math.inf).max(-1, keepdim=True).values.detach()
        logits = beta * (z - zmax)
        if self.normalizer in ("entmax15", "entmax15_scaled"):
            w = entmax15(logits, mask)
        else:
            top = logits.masked_fill(~mask, -math.inf).max(-1, keepdim=True).values.detach()
            w = F.softmax((logits - top).masked_fill(~mask, -math.inf), dim=-1)
        return w.masked_fill(empty, 0.0)

    def forward(self, query, key=None, value=None, mask=None):
        """mask: None, (B, S), or (B, T, S), with True = valid. Returns (B, T, d_model)."""
        key = query if key is None else key
        value = key if value is None else value
        B, T, S = query.shape[0], query.shape[1], key.shape[1]
        q = self.q(query).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k(key).view(B, S, self.h, self.dh).transpose(1, 2)
        v = self.v(value).view(B, S, self.h, self.dh).transpose(1, 2)
        z = q @ k.transpose(-1, -2)
        if mask is not None:
            mask = mask[:, None, None, :] if mask.dim() == 2 else mask[:, None, :, :]
        w = self.normalize(z, mask)
        if self.capture:
            self.last = (z.detach(), w.detach())
        return self.out((w @ v).transpose(1, 2).reshape(B, T, self.h * self.dh))
