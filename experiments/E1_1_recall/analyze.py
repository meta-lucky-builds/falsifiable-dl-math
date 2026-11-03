"""Analysis for E1.1. Usage: python analyze.py results.jsonl

Kill rule (H1, pre-registered): REM is not better than the best baseline at the longest two
lengths, with paired 95% intervals excluding zero in its favor. A tie is reported as a tie.

The pre-registration has no noise levels, so its rule is applied at each noise level:
  KILL FIRED      the rule fires at every noise level
  PARTIAL         the rule fires at some levels only. Not a win: three levels give three
                  chances, with no multiplicity correction, so confirm on fresh seeds
  SURVIVED        REM wins at both lengths at every noise level
"Best baseline" is the highest mean accuracy at that noise and length, among all six other arms.
Picking the best after seeing the means leans against REM, which is the safe direction.
"""
import json
import math
import sys
from collections import defaultdict

from scipy import stats

ARMS = ("softmax", "learned_scale", "ssmax", "entmax15", "entmax15_scaled", "rem", "calib")
BASELINES = tuple(a for a in ARMS if a != "rem")
METRICS = ("acc", "loss", "needle_mass", "cross_top_mass", "cross_entropy", "self_top_mass", "self_entropy",
           "forward_ms_per_batch")
MIN_SEEDS = 5
LEARNED_FRACTION = 0.8  # train-length accuracy below this share of the matched filter means "not learned"


def load(path):
    final, tune = {}, 0
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["stage"] == "final":
                final[(r["arm"], r["seed"], r["noise"])] = r
            else:
                tune += 1
    return list(final.values()), tune


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return math.nan, math.nan
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else math.nan
    return m, sd


def paired(a, b):
    """Mean of a - b over common seeds, with a t-based 95% interval. a, b: {seed: value}."""
    seeds = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in seeds if a[s] is not None and b[s] is not None]
    k = len(diffs)
    if k < 2:
        return dict(k=k, mean=diffs[0] if k else math.nan, lo=math.nan, hi=math.nan, label="n/a")
    m, sd = mean_sd(diffs)
    half = stats.t.ppf(0.975, k - 1) * sd / math.sqrt(k)
    label = "WIN" if m - half > 0 else "LOSS" if m + half < 0 else "TIE"
    return dict(k=k, mean=m, lo=m - half, hi=m + half, label=label)


def show(d):
    return f"{d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] n={d['k']} {d['label']}"


def main(path):
    records, n_tune = load(path)
    if not records:
        sys.exit("no final-stage records yet")
    table = defaultdict(dict)  # (noise, n, arm, metric) -> {seed: value}
    oracle = defaultdict(dict)  # (noise, n) -> {seed: matched filter accuracy}
    for r in records:
        for n, res in r["by_n"].items():
            for metric in METRICS:
                table[(r["noise"], int(n), r["arm"], metric)][r["seed"]] = res.get(metric)
            oracle[(r["noise"], int(n))][r["seed"]] = res["matched_filter_acc"]
    noises = sorted({r["noise"] for r in records})
    lengths = sorted({int(n) for r in records for n in r["by_n"]})
    print(f"{len(records)} final runs, {n_tune} tuning runs. Lengths {lengths}. Noise levels {noises}.")

    print("\nModel size and cost per arm:")
    for arm in ARMS:
        rows = [r for r in records if r["arm"] == arm]
        if rows:
            bad = sum(r["nonfinite_steps"] for r in rows)
            print(f"  {arm:16s} params={rows[0]['n_params']} extra={rows[0]['n_extra_params']} "
                  f"step_ms={mean_sd([r['step_ms_median'] for r in rows])[0]:.1f} "
                  f"grad_norm={mean_sd([r['grad_norm_median'] for r in rows])[0]:.2f} nonfinite_steps={bad}")

    print("\nSelected learning rates (validation at the training length only):")
    for noise in noises:
        for arm in ARMS:
            rows = [r for r in records if r["arm"] == arm and r["noise"] == noise]
            if rows:
                print(f"  noise={noise} {arm:16s} lr={rows[0]['lr']} val_acc_by_lr={rows[0]['lr_val_acc']}")

    verdicts = {}
    provisional = False
    for noise in noises:
        print(f"\n===== noise {noise} =====")
        for n in lengths:
            mf = mean_sd(list(oracle[(noise, n)].values()))[0]
            print(f"\nN={n} ({n // lengths[0]}x). Matched filter acc {mf:.4f}.")
            print(f"  {'arm':16s} {'acc':>8s} {'sd':>7s} {'loss':>7s} {'needle':>7s} {'cross_top':>9s} "
                  f"{'cross_H':>7s} {'self_top':>8s} {'self_H':>7s} {'fwd_ms':>8s}")
            for arm in ARMS:
                m = {k: mean_sd(list(table[(noise, n, arm, k)].values()))[0] for k in METRICS}
                sd = mean_sd(list(table[(noise, n, arm, "acc")].values()))[1]
                print(f"  {arm:16s} {m['acc']:8.4f} {sd:7.4f} {m['loss']:7.3f} {m['needle_mass']:7.3f} "
                      f"{m['cross_top_mass']:9.3f} {m['cross_entropy']:7.3f} {m['self_top_mass']:8.3f} "
                      f"{m['self_entropy']:7.3f} {m['forward_ms_per_batch']:8.1f}")
            for base in BASELINES:
                d = paired(table[(noise, n, "rem", "acc")], table[(noise, n, base, "acc")])
                print(f"  rem - {base:16s} {show(d)}")

        # Is the task learned at all at the training length? Otherwise a verdict is empty.
        n0 = lengths[0]
        mf0 = mean_sd(list(oracle[(noise, n0)].values()))[0]
        for arm in ARMS:
            acc0 = mean_sd(list(table[(noise, n0, arm, "acc")].values()))[0]
            if not acc0 >= LEARNED_FRACTION * mf0:
                print(f"\n  WARNING: {arm} is at {acc0:.3f} at the training length, against matched filter "
                      f"{mf0:.3f}. It has not learned the task; raise --steps before trusting this level.")

        print("\nKill test at the longest two lengths:")
        labels = []
        for n in lengths[-2:]:
            means = {b: mean_sd(list(table[(noise, n, b, "acc")].values()))[0] for b in BASELINES}
            means = {b: v for b, v in means.items() if not math.isnan(v)}
            if not means:
                labels.append("n/a")
                continue
            best = max(means, key=means.get)
            d = paired(table[(noise, n, "rem", "acc")], table[(noise, n, best, "acc")])
            labels.append(d["label"])
            provisional = provisional or d["k"] < MIN_SEEDS
            print(f"  N={n}: best baseline {best}. rem - best = {show(d)}")
        if "n/a" in labels:
            verdicts[noise] = "NOT EVALUABLE"
        elif all(x == "WIN" for x in labels):
            verdicts[noise] = "not fired"
        else:
            verdicts[noise] = "FIRED (worse)" if "LOSS" in labels else "FIRED (tie)"
        print(f"  noise {noise}: kill {verdicts[noise]}")

    print("\n===== prediction: REM gain over SSMax is largest at high noise =====")
    for n in lengths[-2:]:
        gains = []
        for noise in noises:
            d = paired(table[(noise, n, "rem", "acc")], table[(noise, n, "ssmax", "acc")])
            gains.append(d["mean"])
            print(f"  N={n} noise={noise}: rem - ssmax = {show(d)}")
        known = [(g, nz) for g, nz in zip(gains, noises) if not math.isnan(g)]
        if known:
            at = max(known)[1]
            print(f"  N={n}: largest gain is at noise {at}; "
                  f"{'consistent with' if at == noises[-1] else 'against'} the prediction")

    print("\n===== PRE-REGISTERED KILL VERDICT =====")
    for noise, v in verdicts.items():
        print(f"  noise {noise}: {v}")
    states = set(verdicts.values())
    if "NOT EVALUABLE" in states:
        overall = "NOT EVALUABLE"
    elif all(v.startswith("FIRED") for v in verdicts.values()):
        tie = all(v == "FIRED (tie)" for v in verdicts.values())
        overall = "KILL FIRED" + (": all ties, so a parameter-free rule that matches, not a win" if tie else "")
    elif states == {"not fired"}:
        overall = "SURVIVED at every noise level"
    else:
        overall = "PARTIAL: fired at some noise levels only. Not a win; confirm on fresh seeds"
    print(f"  overall: {overall}")
    if provisional:
        print(f"  PROVISIONAL: fewer than {MIN_SEEDS} paired seeds in at least one comparison")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results.jsonl")
