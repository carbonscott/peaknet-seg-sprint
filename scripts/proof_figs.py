"""Evidence figures for the proof report, made on S3DF from real logs (matplotlib, Agg).

  python scripts/proof_figs.py text  --in FILE|- --title T --out x.png [--tail N] [--width 150]
      render a terminal capture / log excerpt as a PNG (header: host, time with zone, source path)
  python scripts/proof_figs.py curves --run NAME=/path/metrics.jsonl [...] --out curves.png
      validation peak F1 and training loss vs step for each run (seg/train.py metrics.jsonl)
  python scripts/proof_figs.py leaderboard --lb results/leaderboard/result.json --out lb.png
"""
import argparse
import json
import os
import socket
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def stamp():
    return "%s  %s" % (socket.gethostname(), time.strftime("%Y-%m-%d %H:%M:%S %Z"))


def cmd_text(a):
    src = sys.stdin.read() if a.inp == "-" else open(a.inp, errors="replace").read()
    lines = src.expandtabs(4).splitlines()
    if a.grep:
        lines = [l for l in lines if any(g in l for g in a.grep)]
    if a.tail:
        lines = lines[-a.tail:]
    lines = [l if len(l) <= a.width else l[:a.width - 1] + "…" for l in lines]
    n = max(1, len(lines))
    h = 0.9 + 0.165 * n
    fig = plt.figure(figsize=(a.width * 0.074 + 0.4, h), dpi=a.dpi)
    fig.patch.set_facecolor("#111418")
    fig.text(0.01, 1 - 0.28 / h, a.title, color="#9fe870", fontsize=11, family="monospace", va="top", weight="bold")
    fig.text(0.01, 1 - 0.55 / h, "captured on %s   source: %s" % (stamp(), a.source or (a.inp if a.inp != "-" else "stdin")),
             color="#8899aa", fontsize=8.5, family="monospace", va="top")
    fig.text(0.01, 1 - 0.85 / h, "\n".join(lines), color="#e6e6e6", fontsize=9, family="monospace", va="top",
             linespacing=1.32)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi, facecolor=fig.get_facecolor())
    print(a.out)


def read_jsonl(p):
    out = []
    for l in open(p):
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out


def cmd_curves(a):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2), dpi=100)
    for spec in a.run:
        name, p = spec.split("=", 1)
        ev = read_jsonl(p)
        v = [e for e in ev if e.get("event") == "val"]
        t = [e for e in ev if e.get("event") == "train"]
        if v:
            ax1.plot([e["step"] for e in v], [e["peak_f1"] for e in v], "o-", ms=3, label=name)
        if t:
            ax2.plot([e["step"] for e in t], [e["loss"] for e in t], lw=1, label=name)
        best = [e for e in ev if e.get("event") == "best"]
        if best:
            b = best[-1]
            ax1.annotate("%.3f" % b.get("peak_f1", float("nan")), (b.get("step", 0), b.get("peak_f1", 0)), fontsize=8,
                         xytext=(4, 4), textcoords="offset points")
    ax1.set_xlabel("step"); ax1.set_ylabel("val peak F1 (3 px, best thr)"); ax1.set_title("validation peak F1 during training")
    ax2.set_xlabel("step"); ax2.set_ylabel("training loss"); ax2.set_yscale("log"); ax2.set_title("training loss")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(a.title or "training curves from seg/train.py metrics.jsonl", fontsize=12, weight="bold")
    fig.text(0.99, 0.005, "made on %s from: %s" % (stamp(), "; ".join(s.split("=", 1)[1] for s in a.run))[:260],
             ha="right", fontsize=6.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(a.out)
    print(a.out)


def cmd_leaderboard(a):
    d = json.load(open(a.lb))
    ent = d["entries"][::-1]
    names = [e["statement"] for e in ent]
    fig, ax = plt.subplots(figsize=(11, 0.6 * len(ent) + 1.6), dpi=100)
    y = range(len(ent))
    ax.barh([i + 0.2 for i in y], [e["peak_f1"] for e in ent], 0.4, label="test peak F1 (3 px)", color="#2b7bba")
    ax.barh([i - 0.2 for i in y], [e["pixel_iou"] for e in ent], 0.4, label="test pixel IoU", color="#f0a030")
    for i, e in enumerate(ent):
        ax.text(e["peak_f1"] + 0.005, i + 0.2, "%.4f" % e["peak_f1"], va="center", fontsize=9)
        ax.text(e["pixel_iou"] + 0.005, i - 0.2, "%.4f" % e["pixel_iou"], va="center", fontsize=9)
    ax.set_yticks(list(y)); ax.set_yticklabels(names); ax.set_xlim(0, 1.08)
    ax.legend(loc="lower right", fontsize=9); ax.grid(axis="x", alpha=0.3)
    ax.set_title("Leaderboard: whole test split (%s images)" % ent[-1].get("n_test_images", "?"), weight="bold")
    fig.text(0.99, 0.01, "from %s, made on %s" % (a.lb, stamp()), ha="right", fontsize=7, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(a.out)
    print(a.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("text")
    t.add_argument("--in", dest="inp", required=True)
    t.add_argument("--title", required=True)
    t.add_argument("--source")
    t.add_argument("--out", required=True)
    t.add_argument("--tail", type=int, default=0)
    t.add_argument("--grep", nargs="*")
    t.add_argument("--width", type=int, default=150)
    t.add_argument("--dpi", type=int, default=90)
    c = sub.add_parser("curves")
    c.add_argument("--run", nargs="+", required=True)
    c.add_argument("--title")
    c.add_argument("--out", required=True)
    lb = sub.add_parser("leaderboard")
    lb.add_argument("--lb", required=True)
    lb.add_argument("--out", required=True)
    a = ap.parse_args()
    {"text": cmd_text, "curves": cmd_curves, "leaderboard": cmd_leaderboard}[a.cmd](a)


if __name__ == "__main__":
    main()
