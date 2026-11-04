"""Analysis for E5.1. Usage: python analyze.py results.jsonl

Reading, fixed in hypotheses/H5_sparsity_train_or_inference.md, at N = 256 with refit. By the
pre-run amendment there, A[soft, ent] in the reading is the cross-only swap, which alone keeps
the cross scores fixed. The both-layer share is printed for comparison and not read:
  share = (A[soft, ent] - A[soft, soft]) / (A[ent, ent] - A[soft, soft])
  Inference dominates  share's interval above 0.5 and retrieval rates differ by under 0.01
  Training dominates   share's interval below 0.5 and T=ent has higher retrieval, with a
                       paired interval excluding zero
  Mixed or unresolved  otherwise
  Premise failed       A[ent, ent] - A[soft, soft] at N = 256 has an interval including zero
                       at both noise levels; the share is then not interpreted

Where the pre-registration needs a detail, this script uses:
- Each noise level gets its own reading. Nothing is pooled across noise.
- "Differ by under 0.01" is the absolute mean paired difference, since the pre-registration
  asks for an interval only in the training clause.
- Retrieval is best head (per example), on the scores as trained: run.py pins the self
  normalizer to T, because a swapped self normalizer changes the cross scores.
- The premise uses refit accuracy, the A of the primary measure. Without refit, the same gap
  is printed beside it and does not enter the reading.

SHARE ESTIMATOR. Per-seed shares with a t interval, unless some seed's denominator is below
NEAR_ZERO, where a per-seed ratio is dominated by test noise or has a flipped sign. Then:
ratio of means with a percentile bootstrap over seeds. If over 2.5% of resamples have a
denominator at or below zero, the ratio is unbounded at this level and the interval is
(-inf, inf). The output states which estimator was used.
"""
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

from scipy import stats

_spec = importlib.util.spec_from_file_location("e1_1_analyze", Path(__file__).resolve().parents[1] / "E1_1_recall" / "analyze.py")
e11a = importlib.util.module_from_spec(_spec)
sys.modules["e1_1_analyze"] = e11a
_spec.loader.exec_module(e11a)
mean_sd, paired = e11a.mean_sd, e11a.paired

SOFT, ENT, REM = "softmax", "entmax15", "rem"
ARMS = (SOFT, ENT, REM)
CROSS = "entmax15_cross"
NEAR_ZERO = 0.02  # about one standard error of a paired accuracy difference on 1024 test examples
N_BOOT = 10000
MIN_SEEDS = 5
BASE_LR = 3e-3


def load(path):
    final, tune = {}, {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["stage"] == "final":
                final[(r["arm"], r["seed"], r["noise"])] = r
            else:
                tune[(r["arm"], r["seed"], r["lr"], r["noise"])] = r
    return list(final.values()), list(tune.values())


def fmt(d):
    return f"{d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] n={d['k']}"


def share_summary(num, den, seed=0):
    """num, den: {seed: value}. Returns the share estimate used and how it was computed."""
    seeds = sorted(set(num) & set(den))
    k = len(seeds)
    if k < 2:
        return dict(k=k, mean=math.nan, lo=math.nan, hi=math.nan, method="n/a: fewer than 2 seeds")
    n_vals, d_vals = [num[s] for s in seeds], [den[s] for s in seeds]
    if min(d_vals) >= NEAR_ZERO:
        m, sd = mean_sd([a / b for a, b in zip(n_vals, d_vals)])
        half = stats.t.ppf(0.975, k - 1) * sd / math.sqrt(k)
        return dict(k=k, mean=m, lo=m - half, hi=m + half, method="per-seed share, t interval")
    mean_den = sum(d_vals) / k
    point = sum(n_vals) / k / mean_den if mean_den > 0 else math.nan
    rng, ratios, undefined = random.Random(seed), [], 0
    for _ in range(N_BOOT):
        pick = [rng.randrange(k) for _ in range(k)]
        d = sum(d_vals[i] for i in pick)
        if d <= 0:
            undefined += 1
        else:
            ratios.append(sum(n_vals[i] for i in pick) / d)
    method = f"ratio of means, bootstrap over seeds (min denominator {min(d_vals):+.4f} < {NEAR_ZERO})"
    if undefined > 0.025 * N_BOOT or not ratios:
        return dict(k=k, mean=point, lo=-math.inf, hi=math.inf,
                    method=method + f"; {undefined}/{N_BOOT} resamples undefined, so unbounded")
    ratios.sort()
    return dict(k=k, mean=point, lo=ratios[int(0.025 * len(ratios))], hi=ratios[int(0.975 * (len(ratios) - 1))],
                method=method)


def includes_zero(d):
    return d["lo"] <= 0 <= d["hi"]


def premise_failed(gaps):
    """gaps: the A[ent, ent] - A[soft, soft] interval per noise level."""
    return all(includes_zero(g) for g in gaps)


def read_noise(share, ret):
    """share: the interval from share_summary. ret: paired T=ent minus T=soft retrieval."""
    if share["lo"] > 0.5 and abs(ret["mean"]) < 0.01:
        return "INFERENCE DOMINATES"
    if share["hi"] < 0.5 and ret["lo"] > 0:
        return "TRAINING DOMINATES"
    return "MIXED OR UNRESOLVED"


def main(path):
    final, tune = load(path)
    if not final:
        sys.exit(f"no final runs in {path}")
    noises = sorted({r["noise"] for r in final})
    lengths = sorted({int(n) for r in final for n in r["retrieval"]})
    seeds = sorted({r["seed"] for r in final})
    print(f"{len(final)} final runs, {len(tune)} LR check runs, seeds {seeds}")
    if len(seeds) < MIN_SEEDS:
        print(f"WARNING: fewer than the pre-registered {MIN_SEEDS} seeds")

    def A(arm, readout, kind, noise, n, field="acc"):
        return {r["seed"]: r["evals"][readout][kind][str(n)][field] for r in final
                if r["arm"] == arm and r["noise"] == noise and readout in r["evals"]}

    def ret(arm, noise, n, field="ret_best"):
        return {r["seed"]: r["retrieval"][str(n)][field] for r in final if r["arm"] == arm and r["noise"] == noise}

    def fixed_head(arm, noise, n):
        """Rate of the better single head, per seed."""
        return {r["seed"]: max(v for k, v in r["retrieval"][str(n)].items() if k.startswith("ret_head"))
                for r in final if r["arm"] == arm and r["noise"] == noise}

    def share(num_readout, kind, noise, n):
        base = A(SOFT, SOFT, kind, noise, n)
        swap, full = A(SOFT, num_readout, kind, noise, n), A(ENT, ENT, kind, noise, n)
        common = set(base) & set(swap) & set(full)
        return share_summary({s: swap[s] - base[s] for s in common}, {s: full[s] - base[s] for s in common})

    print("\n== LR selection (tuning validation accuracy at N = 8) ==")
    for noise in noises:
        chosen = {}
        for arm in ARMS:
            rows = [r for r in final if r["arm"] == arm and r["noise"] == noise]
            if rows:
                chosen[arm] = rows[0]["lr"]
                print(f"  noise={noise} {arm:10s} lr={rows[0]['lr']} val_acc_by_lr={rows[0]['lr_val_acc']}")
        fired = any(chosen.get(a, BASE_LR) != BASE_LR for a in (SOFT, ENT))
        note = "; REM alone moved, which the gate would not have allowed" if not fired and chosen.get(REM, BASE_LR) != BASE_LR else ""
        print(f"  noise={noise} pre-registered gate {'fired' if fired else 'not fired'}{note}")

    print("\n== Run health ==")
    print(f"  non-finite training steps: {sum(r['nonfinite_steps'] for r in final + tune)}, "
          f"non-finite refit steps: {sum(v['nonfinite_steps'] for r in final for v in r['refit'].values())}")
    bad = sum(len({r["retrieval"][str(n)]["matched_filter_acc"] for r in final if r["seed"] == s and r["noise"] == nz}) > 1
              for s in seeds for nz in noises for n in lengths)
    print(f"  pairing: matched-filter accuracy depends on data alone, and differs across T in {bad} (seed, noise, N) "
          f"cells; {'OK' if not bad else 'TEST DATA IS NOT PAIRED, DO NOT TRUST THIS RUN'}")

    print("\n== Accuracy and needle mass, mean (sd) ==")
    print("noise     N  T          I               refit  matched   acc              needle_best      ret_as_read")
    for noise in noises:
        for n in lengths:
            mf, _ = mean_sd(list(ret(SOFT, noise, n, "matched_filter_acc").values()))
            for arm in ARMS:
                for readout in (*ARMS, CROSS):
                    for kind in ("none", "refit"):
                        acc = A(arm, readout, kind, noise, n)
                        if not acc:
                            continue
                        m, sd = mean_sd(list(acc.values()))
                        nm, nsd = mean_sd(list(A(arm, readout, kind, noise, n, "needle_mass_best_head").values()))
                        rr, _ = mean_sd(list(A(arm, readout, kind, noise, n, "ret_best").values()))
                        print(f"{noise:5} {n:5d}  {arm:10s} {readout:15s} {kind:6s} {mf:7.4f}   "
                              f"{m:.4f} ({sd:.4f})  {nm:.4f} ({nsd:.4f})  {rr:.4f}")

    print("\n== Retrieval on the scores as trained, mean (sd) ==")
    for noise in noises:
        for n in lengths:
            for arm in ARMS:
                m, sd = mean_sd(list(ret(arm, noise, n).values()))
                fm, fsd = mean_sd(list(fixed_head(arm, noise, n).values()))
                print(f"  noise={noise} N={n:3d} T={arm:10s} best head (per example) {m:.4f} ({sd:.4f})  "
                      f"better single head {fm:.4f} ({fsd:.4f})")

    print("\n== Retrieval differences, paired over seeds ==")
    for noise in noises:
        for n in lengths:
            print(f"  noise={noise} N={n:3d} T=ent - T=soft {fmt(paired(ret(ENT, noise, n), ret(SOFT, noise, n)))}"
                  f"   T=ent - T=rem {fmt(paired(ret(ENT, noise, n), ret(REM, noise, n)))}"
                  f"   T=rem - T=soft {fmt(paired(ret(REM, noise, n), ret(SOFT, noise, n)))}")

    print("\n== Effect of the self normalizer on cross scores, T=soft (retrieval as read minus as trained) ==")
    for noise in noises:
        for n in lengths:
            base = ret(SOFT, noise, n)
            print(f"  noise={noise} N={n:3d} I=ent both layers {fmt(paired(A(SOFT, ENT, 'none', noise, n, 'ret_best'), base))}"
                  f"   I=ent cross only {fmt(paired(A(SOFT, CROSS, 'none', noise, n, 'ret_best'), base))}")

    print("\n== Premise check: A[ent, ent] - A[soft, soft] at N = 256 ==")
    gaps = {}
    for noise in noises:
        gaps[noise] = paired(A(ENT, ENT, "refit", noise, 256), A(SOFT, SOFT, "refit", noise, 256))
        plain = paired(A(ENT, ENT, "none", noise, 256), A(SOFT, SOFT, "none", noise, 256))
        print(f"  noise={noise} refit {fmt(gaps[noise])}   no refit (not used) {fmt(plain)}")
    failed = premise_failed(list(gaps.values()))
    print(f"  premise {'FAILED: interval includes zero at every noise level' if failed else 'holds'}")

    print("\n== Inference share (A[soft, ent] - A[soft, soft]) / (A[ent, ent] - A[soft, soft]) ==")
    for n in (256, 64):
        for noise in noises:
            s = share(CROSS, "refit", noise, n)
            print(f"  N={n:3d} noise={noise} cross only, refit: {s['mean']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}] n={s['k']}, {s['method']}")
            for label, readout, kind in (("cross only, no refit", CROSS, "none"), ("both layers, refit", ENT, "refit"),
                                         ("both layers, no refit", ENT, "none")):
                s = share(readout, kind, noise, n)
                print(f"    not used in the reading, {label}: {s['mean']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}] {s['method']}")

    print("\n== Reading (N = 256, refit, retrieval best head) ==")
    if failed:
        print("  PREMISE FAILED: E1.1's entmax gain did not replicate, and the share is not interpreted.")
        return
    for noise in noises:
        s = share(CROSS, "refit", noise, 256)
        r = paired(ret(ENT, noise, 256), ret(SOFT, noise, 256))
        print(f"  noise={noise}: {read_noise(s, r)}  (share [{s['lo']:+.3f}, {s['hi']:+.3f}], {s['method']}; "
              f"retrieval T=ent - T=soft {fmt(r)})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
