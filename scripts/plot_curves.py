"""Training-curve figure from seg.train metrics.jsonl files (any number of runs).

usage (S3DF, CPU is enough):
    python scripts/plot_curves.py --run "<metrics.jsonl>|<label>" [--run ...] --title "..." --out curves.png
Top panel: train loss vs step (50-step means). Bottom panel: val peak F1 (solid), precision (dashed) and
recall (dotted) at each validation's best threshold vs step; the best step (max val peak F1) is starred and
annotated with F1 and threshold. Only the last training segment of a file is used (after the last "start").
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read(path):
    ev = [json.loads(l) for l in open(path) if l.strip()]
    starts = [i for i, e in enumerate(ev) if e.get("event") == "start"]
    seg = ev[starts[-1]:] if starts else ev
    tr = [(e["step"], e["loss"]) for e in seg if e.get("event") == "train"]
    va = [e for e in seg if e.get("event") == "val"]
    start = seg[0] if seg and seg[0].get("event") == "start" else {}
    return tr, va, start


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="metrics.jsonl|label")
    ap.add_argument("--title", default="training curves")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    summary = []
    for k, spec in enumerate(a.run):
        path, label = (spec.split("|", 1) + [None])[:2]
        label = label or path
        c = colors[k % len(colors)]
        tr, va, _ = read(path)
        if tr:
            ax1.plot([s for s, _ in tr], [l for _, l in tr], color=c, lw=1, label=label)
        if va:
            st = [e["step"] for e in va]
            ax2.plot(st, [e["peak_f1"] for e in va], "-o", color=c, ms=3, label="%s peak F1" % label)
            ax2.plot(st, [e.get("peak_p", float("nan")) for e in va], "--", color=c, lw=1, alpha=.7)
            ax2.plot(st, [e.get("peak_r", float("nan")) for e in va], ":", color=c, lw=1, alpha=.7)
            b = max(va, key=lambda e: e["peak_f1"])
            ax2.plot([b["step"]], [b["peak_f1"]], "*", color=c, ms=16, mec="k")
            ax2.annotate("best step %d: F1 %.3f @thr %.3g" % (b["step"], b["peak_f1"], b.get("best_thr", float("nan"))),
                         (b["step"], b["peak_f1"]), textcoords="offset points", xytext=(8, -14 - 12 * k),
                         fontsize=9, color=c)
            summary.append("%s: %d steps, %d vals, best val peak F1 %.4f at step %d"
                           % (label, tr[-1][0] if tr else 0, len(va), b["peak_f1"], b["step"]))
    ax1.set_ylabel("train loss (50-step mean)")
    ax1.set_yscale("log")
    ax1.grid(alpha=.3)
    ax1.legend(fontsize=9)
    ax2.set_ylabel("val peak metric (tol 3 px)\nsolid F1, dashed P, dotted R")
    ax2.set_xlabel("optimizer step")
    lo = min([e["peak_f1"] for spec in a.run for e in read(spec.split("|")[0])[1]] or [0])
    ax2.set_ylim(max(0.0, min(0.5, lo - 0.1)), 1.02)
    ax2.grid(alpha=.3)
    ax2.legend(fontsize=9, loc="lower right")
    fig.suptitle(a.title + "\n" + "\n".join(summary), fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("\n".join(summary))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
