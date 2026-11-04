#!/usr/bin/env python3
"""Report for E8. Applies the reading rules fixed in hypotheses/H8_long_context_failure.md.

    python analyze.py --out runs/e8

Writes report.txt and results_compact.csv in that directory. A partial run gets a partial report.
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from scipy import stats

BASE, CONTROL = "rope_1e4", "control_ctx4x"
SURVIVE_MEAN, SURVIVE_MAX = 0.02, 0.05
SHORT_COST, CONTROL_MARGIN, REPRO_MIN, FAR_MIN, USEFUL_MIN = 0.03, 0.03, 1.0, 0.5, 0.005


def load(out):
    cfg = json.loads((out / "config.json").read_text())
    runs = {}
    for path in sorted((out / "jobs").glob("*.json")):
        r = json.loads(path.read_text())
        runs.setdefault(r["arm"], []).append(r)
    return cfg, runs


def bin_index(N, keep):
    return N // keep - 1


def L(r, N, keep):
    return r["eval"]["L_bins"][bin_index(N, keep)]


def Lw(r, N, keep):
    return r["eval"]["Lw_bins"][bin_index(N, keep)]


def excess(r, N, keep):
    return L(r, N, keep) - Lw(r, N, keep)


def far_mass(r, N, keep):
    """Mean over layers at the bin of rows [N - keep, N)."""
    return statistics.mean(x["far"][bin_index(N, keep)] for x in r["eval"]["attention"]["layers"])


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def interval(xs):
    """Mean and two-sided 95% t interval; the interval is NaN with fewer than 2 values."""
    m = mean(xs)
    if len(xs) < 2:
        return m, float("nan"), float("nan")
    half = stats.t.ppf(0.975, len(xs) - 1) * statistics.stdev(xs) / len(xs) ** 0.5
    return m, m - half, m + half


def far_Ns(cfg, r):
    return [N for N in cfg["Ns"] if N > r["ctx"]]


def survives(cfg, rows):
    """None if there is nothing to read. Applies to rows past each run's training context."""
    keep = cfg["keep"]
    Ns = far_Ns(cfg, rows[0])
    if not Ns:
        return None
    return all(mean([excess(r, N, keep) for r in rows]) <= SURVIVE_MEAN
               and max(excess(r, N, keep) for r in rows) <= SURVIVE_MAX for N in Ns)


def useful(cfg, rows):
    """Returns (is it useful, the N, mean, low, high) for the N with the lowest excess."""
    keep = cfg["keep"]
    best = None
    for N in far_Ns(cfg, rows[0]):
        m, lo, hi = interval([excess(r, N, keep) for r in rows])
        if best is None or m < best[1]:
            best = (N, m, lo, hi)
    if best is None:
        return False, None, float("nan"), float("nan"), float("nan")
    N, m, lo, hi = best
    return m < -USEFUL_MIN and hi < 0, N, m, lo, hi


def rules(cfg, runs):
    """Reads the pre-registered rules. Returns (lines, verdict dict)."""
    keep, Ns = cfg["keep"], cfg["Ns"]
    short, n2 = Ns[0], Ns[1]
    v, out = {}, []
    base, control = runs.get(BASE), runs.get(CONTROL)
    if not base or not control:
        return ["Rules not read: the base or control arm has no finished runs."], dict(complete=False)
    base_short = mean([L(r, short, keep) for r in base])
    ctl_N = control[0]["ctx"]
    ctl = mean([L(r, ctl_N, keep) for r in control])
    v["C1"] = ctl <= base_short + CONTROL_MARGIN
    out.append(f"C1 positive control: control L({ctl_N}) {ctl:.3f}, limit {base_short + CONTROL_MARGIN:.3f} "
               f"(base L({short}) {base_short:.3f} + {CONTROL_MARGIN}): {'PASS' if v['C1'] else 'FAIL'}")
    e2 = mean([excess(r, n2, keep) for r in base])
    v["C2"] = e2 >= REPRO_MIN
    out.append(f"C2 reproduction: base E({n2}) {e2:+.3f}, needs at least {REPRO_MIN}: "
               f"{'PASS' if v['C2'] else 'FAIL'}")
    f2 = mean([far_mass(r, n2, keep) for r in base])
    v["C3"] = f2 >= FAR_MIN
    out.append(f"C3 mechanism: base far mass at N = {n2}, mean over layers, {f2:.3f}, needs at least "
               f"{FAR_MIN}: {'SUPPORTED' if v['C3'] else 'NOT SUPPORTED'}")
    if not v["C1"]:
        out.append("C1 FAILED, so a bug is the leading explanation. Stop here: the rest is not read.")
        v["complete"] = True
        return out, v
    if not v["C2"]:
        out.append("C2 FAILED: E7's failure did not reproduce. That discrepancy comes first.")
    out.append("")
    v["survive"], v["cost"], v["useful"] = {}, {}, {}
    for arm, rows in runs.items():
        s = survives(cfg, rows)
        v["survive"][arm] = s
        ms = mean([L(r, short, keep) for r in rows])
        v["cost"][arm] = ms > base_short + SHORT_COST
        u, uN, um, ulo, uhi = useful(cfg, rows)
        v["useful"][arm] = bool(s) and u
        note = "survives" if s else "fails"
        cost = f", short-context cost ({ms:.3f})" if v["cost"][arm] else ""
        use = ""
        if s:
            use = f"; best E({uN}) {um:+.3f} [{ulo:+.3f}, {uhi:+.3f}]: " + (
                "long context useful" if u else "long context not shown useful")
        out.append(f"  {arm:16s} {note}{cost}{use}")
    out.append("")
    linear = [a for a, rows in runs.items() if a in ("alibi", "rope_1e4_alibi") and v["survive"].get(a)]
    base_fails = not v["survive"].get(BASE)
    if v["C1"] and v["C2"] and v["C3"] and base_fails and linear:
        out.append(f"E7's claim about RoPE: SUPPORTED. The base fails, and linear decay survives in {linear}.")
        if "rope_5e5" in runs:
            out.append("  rope_5e5 survives, so theta is the setting responsible." if v["survive"].get("rope_5e5")
                       else "  rope_5e5 fails, so raising theta alone is not enough.")
    else:
        out.append("E7's claim about RoPE: NOT SUPPORTED by these rules.")
    candidates = [a for a, s in v["survive"].items() if s and a != CONTROL and not v["cost"][a]]
    any_use = any(v["useful"].values())
    if candidates:
        out.append(f"Next, per H8: pre-register a normalizer comparison on one of {candidates}, with fresh seeds.")
    else:
        out.append("Next, per H8: no arm survives without cost. Do not compare normalizers; train at length.")
    if not any_use:
        out.append("No arm shows useful long context. If the copy probe is near chance past W, this model size "
                   "gains nothing from long context here, so a comparison needs a retrieval task or a larger model.")
    v["complete"] = True
    return out, v


def table(title, header, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    fmt = lambda row: "  ".join(str(x).rjust(w) if i else str(x).ljust(w)
                                for i, (x, w) in enumerate(zip(row, widths)))
    return [title, fmt(header), fmt(["-" * w for w in widths])] + [fmt(r) for r in rows] + [""]


def report(cfg, runs):
    keep, Ns, arms = cfg["keep"], cfg["Ns"], list(runs)
    lines = [f"E8 report: {cfg['name']}", ""]
    if cfg["name"] != "full":
        lines += ["NOT A RESULT: this is a validation run, and its numbers carry no meaning.", ""]
    n = sum(len(v) for v in runs.values())
    expected = len(cfg["final_seeds"]) * 5
    lines.append(f"Runs finished: {n} of {expected}." + ("" if n == expected else " THIS REPORT IS PARTIAL."))
    bad = sum(r["nonfinite"] for rows in runs.values() for r in rows)
    lines.append(f"Non-finite training steps: {bad}. Evaluation windows: "
                 f"{sorted({r['eval_hash'][:8] for rows in runs.values() for r in rows})} (one value means all alike).")
    # rows inside the training context are unchanged by the limit, so this must be exactly zero
    worst = max((abs(excess(r, Ns[0], keep)) for rows in runs.values() for r in rows), default=0.0)
    lines.append(f"Check, |E({Ns[0]})| must be near zero: largest {worst:.2e}"
                 + ("" if worst < 1e-3 else "  CHECK FAILED"))
    lines.append("")

    head = ["arm"] + [f"L({N})" for N in Ns] + [f"E({N})" for N in Ns[1:]] + ["T", "val"]
    rows = []
    for a in arms:
        rs = runs[a]
        rows.append([a] + [f"{mean([L(r, N, keep) for r in rs]):.3f}" for N in Ns]
                    + [f"{mean([excess(r, N, keep) for r in rs]):+.3f}" for N in Ns[1:]]
                    + [f"{mean([r['eval']['T'] for r in rs]):.3f}",
                       f"{mean([r['val_loss_final'] for r in rs]):.3f}"])
    lines += table("Bits per byte, mean over seeds. E(N) = L(N) - Lw(N), paired on the same targets.", head, rows)

    lines.append("Reading rules, fixed in the pre-registration")
    verdict_lines, _ = rules(cfg, runs)
    lines += verdict_lines + [""]

    n_bins = max(Ns) // keep
    shown = [i for i in range(n_bins) if i < 12 or (i + 1) % 4 == 0]
    head = ["row bin end"] + [str(a) for a in arms]
    rows = [[(i + 1) * keep] + [f"{mean([r['eval']['L_bins'][i] for r in runs[a]]):.2f}" for a in arms]
            for i in shown]
    lines += table("Loss per row, bits per byte. A jump just past the training length is the failure.", head, rows)

    rows = []
    for a in arms:
        n_layers = len(runs[a][0]["eval"]["attention"]["layers"])
        for N in Ns[1:]:
            per_layer = [mean([r["eval"]["attention"]["layers"][k]["far"][bin_index(N, keep)] for r in runs[a]])
                         for k in range(n_layers)]
            rows.append([a, N] + [f"{x:.3f}" for x in per_layer])
    depth = len(rows[0]) - 2 if rows else 0
    lines += table("Attention mass on distances at or past the arm's training context, by layer.",
                   ["arm", "N"] + [f"layer{k}" for k in range(depth)], rows)

    def bucket_table(title, key, spec):
        some = next((r for rs in runs.values() for r in rs), None)
        if some is None:
            return []
        edges = some["eval"]["attention"]["edges"]
        head = ["arm"] + [f"{edges[i]}-{edges[i + 1]}" for i in range(len(edges) - 1)]
        out = []
        for a in arms:
            layers = [x for r in runs[a] for x in r["eval"]["attention"]["layers"]]
            cells = []
            for i in range(len(edges) - 1):
                vals = [x[key][i] for x in layers if x[key][i] is not None]
                cells.append(format(mean(vals), spec))
            out.append([a] + cells)
        return table(title, head, out)

    lines += bucket_table(f"Where attention mass goes on the last {keep} rows, by distance, mean over layers.",
                          "mass", ".3f")
    lines += bucket_table("Mean logit relative to the row's top, by distance. Far above near means aliasing.",
                          "logit", ".2f")

    Ds = cfg["Ds"]
    head = ["arm"] + [f"D={D}" for D in Ds]
    rows = [[a] + [f"{mean([r['eval'][f'copy_{D}']['byte'] for r in runs[a]]):.3f}" for D in Ds] for a in arms]
    lines += table("Copy probe, byte accuracy. Guessing in the alphabet gives about 0.016.", head, rows)

    rows = [[a, runs[a][0]["ctx"], runs[a][0]["batch"], f"{mean([r['step_ms'] for r in runs[a]]):.1f}",
             f"{mean([r['tokens_per_s'] for r in runs[a]]):.0f}"] for a in arms]
    lines += table("Cost", ["arm", "context", "batch", "step ms", "tokens/s"], rows)
    return "\n".join(lines) + "\n"


def write_csv(path, cfg, runs):
    keep, Ds = cfg["keep"], cfg["Ds"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["arm", "seed", "ctx", "step_ms", "nonfinite", "val", "T", "L_by_bin", "Lw_by_bin",
                    "far_mean_layers_by_bin", "copy_byte_by_D"])
        for arm, rows in runs.items():
            for r in rows:
                e = r["eval"]
                layers = e["attention"]["layers"]
                far = [statistics.mean(x["far"][i] for x in layers) for i in range(len(e["L_bins"]))]
                w.writerow([arm, r["seed"], r["ctx"], f"{r['step_ms']:.2f}", r["nonfinite"],
                            f"{r['val_loss_final']:.5f}", f"{e['T']:.5f}",
                            "|".join(f"{x:.4f}" for x in e["L_bins"]),
                            "|".join(f"{x:.4f}" for x in e["Lw_bins"]),
                            "|".join(f"{x:.3f}" for x in far),
                            "|".join(f"{e[f'copy_{D}']['byte']:.4f}" for D in Ds)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    cfg, runs = load(out)
    if not runs:
        print(f"ERROR: no finished jobs in {out / 'jobs'}", file=sys.stderr)
        return 1
    text = report(cfg, runs)
    (out / "report.txt").write_text(text)
    write_csv(out / "results_compact.csv", cfg, runs)
    print(f"report: {out / 'report.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
