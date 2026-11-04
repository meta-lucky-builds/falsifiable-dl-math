#!/usr/bin/env python3
"""Writes report.txt and results_compact.csv from an E7 output directory. Tolerates partial runs."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
ARMS = ("softmax", "ssmax", "entmax15", "entmax15_scaled")
THRESHOLD = 0.005  # bits per byte, from the pre-registration
ALPHA = 0.05
# (label, trained arm, readout) for the three primary tests
PRIMARY = (("P1 entmax15 < softmax", ("entmax15", "entmax15"), ("softmax", "softmax")),
           ("P2 entmax15 < ssmax", ("entmax15", "entmax15"), ("ssmax", "ssmax")),
           ("P3 softmax model, entmax15 readout < softmax readout",
            ("softmax", "entmax15"), ("softmax", "softmax")))


def holm(ps, alpha=ALPHA):
    """Holm step-down. Returns (adjusted p-values, reject flags), in the input order. NaN counts as 1."""
    clean = [1.0 if p is None or not math.isfinite(p) else float(p) for p in ps]
    adjusted, running = [0.0] * len(clean), 0.0
    for rank, i in enumerate(sorted(range(len(clean)), key=lambda i: clean[i])):
        running = max(running, min(1.0, (len(clean) - rank) * clean[i]))
        adjusted[i] = running
    return adjusted, [a <= alpha for a in adjusted]


def load(out):
    cfg = json.loads((out / "config.json").read_text())
    jobs = [json.loads(p.read_text()) for p in sorted((out / "jobs").glob("*.json"))]
    path = out / "selection.json"
    selection = json.loads(path.read_text()) if path.exists() else {}
    failed = sorted(p.stem for p in (out / "queue" / "failed").glob("*.json"))
    return cfg, [j for j in jobs if j["kind"] == "tune"], [j for j in jobs if j["kind"] == "final"], \
        selection, failed


class Table:
    """Numbers per seed for (arm, readout), all read through here so missing data stays missing."""

    def __init__(self, cfg, finals):
        self.cfg = cfg
        self.by = {}
        for j in finals:
            for readout, numbers in j.get("eval", {}).items():
                self.by.setdefault((j["arm"], readout), {})[j["seed"]] = numbers
        self.ctx = cfg["Ns"][0]

    def L(self, key, N):
        return {s: e["L"][str(N)] for s, e in self.by.get(key, {}).items() if str(N) in e.get("L", {})}

    def T(self, key):
        return {s: e["T"]["all"] for s, e in self.by.get(key, {}).items() if "all" in e.get("T", {})}

    def G(self, key, N):
        t, n = self.T(key), self.L(key, N)
        return {s: t[s] - n[s] for s in t if s in n}

    def copy(self, key, D, kind):
        return {s: e["copy"][str(D)][kind] for s, e in self.by.get(key, {}).items()
                if str(D) in e.get("copy", {})}

    def primary(self, key):
        """Per-seed mean L over the primary N, only for seeds that have every one of them."""
        rows = [self.L(key, N) for N in self.cfg["primary_Ns"]]
        seeds = set(rows[0]).intersection(*rows[1:])
        return {s: float(np.mean([r[s] for r in rows])) for s in seeds}


def mean_sd(values):
    a = np.array(list(values.values()) if isinstance(values, dict) else values, dtype=float)
    if a.size == 0:
        return "n/a"
    return f"{a.mean():.4f} ({a.std(ddof=1):.4f})" if a.size > 1 else f"{a.mean():.4f} (-)"


def paired(a, b):
    """One-sided paired t-test that a < b. Returns n, mean gain b - a, and p."""
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return dict(n=0, gain=float("nan"), p=float("nan"))
    x, y = np.array([a[s] for s in seeds]), np.array([b[s] for s in seeds])
    gain = float((y - x).mean())
    if len(seeds) < 2:
        return dict(n=len(seeds), gain=gain, p=float("nan"))
    with np.errstate(all="ignore"):
        p = float(stats.ttest_rel(x, y, alternative="less").pvalue)
    return dict(n=len(seeds), gain=gain, p=p)


def outcome(significant, gain, n):
    """The three outcomes of H7 Amendment 2. Too few paired seeds is no outcome at all."""
    if n < 2 or not math.isfinite(gain):
        return "No data"
    if significant:
        return "Supported"
    return "Inconclusive" if gain >= THRESHOLD else "Not supported"


def gatekept_p3b(p3_significant, gated, softmax, alpha=ALPHA):
    """P3b: gated < softmax readout. Tested only if P3 is significant, per the H7 amendment."""
    if not p3_significant:
        return dict(tested=False, sig=False, n=0, gain=float("nan"), p=float("nan"))
    r = paired(gated, softmax)
    return dict(r, tested=True, sig=math.isfinite(r["p"]) and r["p"] <= alpha)


def swap_succeeds(p3_significant, small_rise, p3b_significant):
    """The amended rule: P3, and either a small L rise at the training length or P3b."""
    return p3_significant and (small_rise or p3b_significant)


def interval(a, b):
    """Paired mean a - b with a two-sided 95% t interval, uncorrected."""
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return "n/a"
    d = np.array([a[s] - b[s] for s in seeds])
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return f"{d.mean():+.3f} [{d.mean() - half:+.3f},{d.mean() + half:+.3f}]"


def fmt_p(p):
    return "n/a" if not math.isfinite(p) else f"{p:.4f}"


def warnings_for(cfg, tunes, finals, selection, failed):
    w = []
    for arm, s in sorted(selection.items()):
        if s.get("edge"):
            w.append(f"{arm} selected lr {s['lr']:g}, an edge of the grid")
    bad = [(j["id"], j["nonfinite"]) for j in tunes + finals if j.get("nonfinite")]
    if bad:
        w.append(f"non-finite skipped steps in {len(bad)} jobs: " + ", ".join(f"{i}={n}" for i, n in bad[:6]))
    want = {(a, s) for a in ARMS for s in cfg["final_seeds"]}
    missing = sorted(want - {(j["arm"], j["seed"]) for j in finals})
    if missing:
        w.append(f"{len(missing)} final jobs missing: " + ", ".join(f"{a}/{s}" for a, s in missing[:8]))
    if failed:
        w.append(f"{len(failed)} jobs finally failed: " + ", ".join(failed[:6]))
    if cfg["name"] != "full":
        w.append(f"configuration '{cfg['name']}' with {len(cfg['final_seeds'])} seeds: "
                 f"a check of the code, not a result")
    for field in ("torch", "gpu", "amp", "git", "code_sha", "compiled"):
        values = {str(j.get(field)) for j in tunes + finals}
        if len(values) > 1:
            shown = ", ".join(sorted(v[:12] for v in values)[:4])
            w.append(f"inconsistent {field} across jobs: {shown}")
    hashes = {j.get("eval_hash") for j in finals}
    if len(hashes) > 1:
        w.append("evaluation windows or probes differ across jobs: results are not comparable")
    return w


def report(out):
    cfg, tunes, finals, selection, failed = load(out)
    t = Table(cfg, finals)
    Ns, Ds, ctx = cfg["Ns"], cfg["Ds"], cfg["Ns"][0]
    SWAPS = (("softmax", "softmax"), ("softmax", "entmax15"), ("softmax", "ssmax"), ("softmax", "gated"))
    trained = [(a, a) for a in ARMS]
    lines = [f"E7 / H7 report. Configuration {cfg['name']}: {len(finals)} final and {len(tunes)} tuning "
             f"jobs. L, T, G in bits per byte.",
             f"Model: {cfg['n_layer']} layers, {cfg['n_head']} heads, width {cfg['d_model']}, parameters "
             f"{sorted({j['params'] for j in tunes + finals})}. {cfg['steps']} steps, "
             f"{len(cfg['final_seeds'])} final seeds.",
             "Scale caveat: prior claims for these methods come from larger models, so a null here is "
             "weaker evidence against them than it would be at 10M.", ""]

    warns = warnings_for(cfg, tunes, finals, selection, failed)
    lines.append("WARNINGS" + ("" if warns else ": none"))
    lines += [f"  ! {x}" for x in warns[:14]]

    lines += ["", "SELECTED LEARNING RATES (validation loss at context "
              f"{cfg['ctx']}; * = selected)", "  arm              " +
              "".join(f"{lr:>12g}" for lr in cfg["lrs"])]
    for arm in ARMS:
        s = selection.get(arm, {})
        cells = []
        for lr in cfg["lrs"]:
            v = s.get("table", {}).get(str(lr))
            mark = "*" if s.get("lr") == lr else " "
            cells.append(f"{'n/a' if v is None else format(v, '.4f')}{mark}".rjust(12))
        lines.append(f"  {arm:16s} " + "".join(cells))

    lines += ["", "PER ARM, mean (sd) over seeds", "  arm              N     L                "
              "T                G"]
    for arm, r in trained:
        for N in Ns:
            lines.append(f"  {arm:16s} {N:<5d} {mean_sd(t.L((arm, r), N)):16s} "
                         f"{mean_sd(t.T((arm, r))):16s} {mean_sd(t.G((arm, r), N))}")

    def copy_rows(keys, label=lambda k: k[0]):
        rows = ["  " + " " * 18 + "".join(f"D={D:<14d}" for D in Ds)]
        for key in keys:
            cells = []
            for D in Ds:
                byte, all4 = t.copy(key, D, "byte"), t.copy(key, D, "all4")
                text = "n/a" if not byte else f"{np.mean(list(byte.values())):.3f}/" \
                                              f"{np.mean(list(all4.values())):.3f}"
                cells.append(f"{text:16s}")
            rows.append(f"  {label(key):18s}" + "".join(cells))
        return rows

    lines += ["", "COPY PROBE, accuracy per byte / all 4 correct, mean over seeds"] + copy_rows(trained)

    lines += ["", f"READOUT SWAP on softmax-trained models, no retraining, mean (sd)",
              "  readout          N     L                T                G"]
    for key in SWAPS:
        for N in Ns:
            lines.append(f"  {key[1]:16s} {N:<5d} {mean_sd(t.L(key, N)):16s} "
                         f"{mean_sd(t.T(key)):16s} {mean_sd(t.G(key, N))}")
    lines += ["", "COPY PROBE by readout"] + copy_rows(SWAPS, lambda k: k[1])

    results = [dict(label=label, **paired(t.primary(a), t.primary(b))) for label, a, b in PRIMARY]
    adjusted, rejected = holm([r["p"] for r in results])
    lines += ["", f"PRIMARY TESTS, one-sided paired t on mean L over N in {cfg['primary_Ns']}, Holm at "
              f"family level {ALPHA}", "  test                                                   "
              "n  gain      p        Holm p   outcome"]
    for r, adj, rej in zip(results, adjusted, rejected):
        r["sig"], r["adj"] = rej, adj
        r["outcome"] = outcome(rej, r["gain"], r["n"])
        lines.append(f"  {r['label']:53s} {r['n']:<2d} {r['gain']:+.5f}  {fmt_p(r['p']):8s} "
                     f"{adj:<8.4f} {r['outcome']}")
    lines.append(f"  Supported: significant under Holm. Inconclusive: not significant, mean gain at "
                 f"least {THRESHOLD}; needs new")
    lines.append("  seeds, in a separate confirmatory run not pooled with these. Not supported: not "
                 f"significant, gain under {THRESHOLD}.")
    p1, _, p3 = results

    lines += ["", "READING RULES, as fixed in the pre-registration"]
    lines.append("  Kill for transfer of E1.1 (P1 Not supported): " + {
        "Supported": "NOT FIRED. P1 is Supported.",
        "Inconclusive": "NOT FIRED. P1 is Inconclusive: it needs new seeds to settle.",
        "Not supported": "FIRED. The synthetic entmax gain does not transfer to this real-text setup "
                         "at this scale.",
        "No data": "NOT JUDGED. Too few paired seeds for P1."}[p1["outcome"]])

    rise = paired(t.L(("softmax", "entmax15"), ctx), t.L(("softmax", "softmax"), ctx))
    rise_value = -rise["gain"]  # gain is softmax minus entmax15, so the rise is its negative
    small_rise = rise["n"] > 0 and rise_value < THRESHOLD
    p3b = gatekept_p3b(p3["sig"], t.primary(("softmax", "gated")), t.primary(("softmax", "softmax")))
    verdict = "SUCCEEDS" if swap_succeeds(p3["sig"], small_rise, p3b["sig"]) else "DOES NOT SUCCEED"
    lines.append(f"  Readout swap: {verdict}.")
    lines.append(f"    P3 significant: {'yes' if p3['sig'] else 'no'}. L({ctx}) rise with the entmax15 "
                 f"readout: {rise_value:+.5f}, under {THRESHOLD}: {'yes' if small_rise else 'no'}.")
    if p3b["tested"]:
        lines.append(f"    P3b, row-level gated < softmax readout, one-sided paired t at {ALPHA}: n "
                     f"{p3b['n']}, gain {p3b['gain']:+.5f}, p {fmt_p(p3b['p'])}: "
                     f"{'holds' if p3b['sig'] else 'does not hold'}.")
    else:
        lines.append("    P3b, row-level gated < softmax readout: not tested, as P3 is not significant.")

    for name, r in [(r["label"][:2], r) for r in results] + [("P3b", p3b)]:
        if r["sig"]:
            size = "real but negligible" if r["gain"] < THRESHOLD else "above the threshold"
            lines.append(f"  Practical threshold, {name}: gain {r['gain']:.5f} is {size}.")
    if not any(r["sig"] for r in results):
        lines.append(f"  Practical threshold ({THRESHOLD} bits/byte): no significant gain to judge.")

    longer = [N for N in Ns if N > ctx]
    positive = [(a, N) for a, r in trained for N in longer
                if t.G((a, r), N) and np.mean(list(t.G((a, r), N).values())) > 0]
    if positive:
        shown = ", ".join(f"{a}@{N}" for a, N in positive[:8])
        lines.append(f"  G(N) rule: NOT FIRED. Mean G > 0 for {shown}.")
    else:
        lines.append("  G(N) rule: FIRED. Long context carries no usable information at this scale, "
                     "the loss comparison is")
        lines.append("    uninformative, and only the copy probe can be read.")
    lines.append(f"    (Checked at N > {ctx}. T({ctx}) sees at least as much context as L({ctx}), so "
                 f"G({ctx}) measures no gain.)")

    lines += ["  Cost: arm, L at the shortest N, median step ms, tokens per second"]
    for arm in ARMS:
        rows = [j for j in finals if j["arm"] == arm]
        step = f"{np.median([j['step_ms'] for j in rows]):.1f}" if rows else "n/a"
        rate = f"{np.median([j['tokens_per_s'] for j in rows]):.0f}" if rows else "n/a"
        lines.append(f"    {arm:16s} L({ctx}) {mean_sd(t.L((arm, arm), ctx)):16s} {step:>8s} ms "
                     f"{rate:>9s} tok/s")

    lines += ["", "COPY PROBE differences from softmax, accuracy per byte, paired 95% interval, "
              "secondary and uncorrected"]
    base = ("softmax", "softmax")
    for key in (("ssmax", "ssmax"), ("entmax15", "entmax15"), ("entmax15_scaled", "entmax15_scaled"),
                ("softmax", "entmax15"), ("softmax", "ssmax"), ("softmax", "gated")):
        name = key[0] if key[0] != "softmax" else f"swap {key[1]}"
        lines.append(f"  {name:16s}")
        lines[-1] += " ".join(f"D={D}: {interval(t.copy(key, D, 'byte'), t.copy(base, D, 'byte'))}"
                              for D in Ds[:2])
        lines.append(" " * 18 + " ".join(f"D={D}: {interval(t.copy(key, D, 'byte'), t.copy(base, D, 'byte'))}"
                                         for D in Ds[2:]))

    lines += ["", "LEARNED NORMALIZER PARAMETERS, mean (sd) over layers, heads and seeds"]
    for arm in ("ssmax", "entmax15_scaled"):
        rows = [j["learned"] for j in finals if j["arm"] == arm and j.get("learned")]
        for name in sorted(rows[0]) if rows else []:
            values = np.array([r[name] for r in rows], dtype=float)
            lines.append(f"  {arm:16s} {name:6s} {values.mean():+.4f} ({values.std():.4f})")
    lines += ["", f"NOTES: Fixed SSMax uses s = 1/log({cfg['ctx']}) with n the "
              f"row's valid count, as in src/mai/attention.py,",
              f"so it equals softmax only on rows with n = {cfg['ctx']}, not on the shorter rows of "
              f"an N = {cfg['ctx']} window."]
    return lines, cfg, tunes, finals


def write_csv(path, cfg, tunes, finals):
    Ns, Ds = cfg["Ns"], cfg["Ds"]
    head = ["id", "kind", "arm", "seed", "lr", "readout", "val", "step_ms", "tokens_per_s", "nonfinite"]
    head += [f"L{N}" for N in Ns] + ["T"] + [f"G{N}" for N in Ns]
    head += [f"copy_byte_{D}" for D in Ds] + [f"copy_all4_{D}" for D in Ds]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for j in tunes + finals:
            common = [j["id"], j["kind"], j["arm"], j["seed"], j["lr"]]
            tail = [j.get("val_loss_final"), j.get("step_ms"), j.get("tokens_per_s"), j.get("nonfinite")]
            if j["kind"] == "tune":
                w.writerow(common + [""] + tail)
                continue
            for readout, e in j.get("eval", {}).items():
                L = [e["L"].get(str(N)) for N in Ns]
                T = e.get("T", {}).get("all")
                G = [T - x if T is not None and x is not None else None for x in L]
                cb = [e["copy"].get(str(D), {}).get("byte") for D in Ds]
                ca = [e["copy"].get(str(D), {}).get("all4") for D in Ds]
                w.writerow(common + [readout] + tail + L + [T] + G + cb + ca)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "runs" / "e7"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    if not (out / "config.json").exists():
        print(f"ERROR: no E7 run in {out}", file=sys.stderr)
        return 1
    lines, cfg, tunes, finals = report(out)
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    write_csv(out / "results_compact.csv", cfg, tunes, finals)
    print(f"report:  {out / 'report.txt'} ({len(lines)} lines)")
    print(f"results: {out / 'results_compact.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
