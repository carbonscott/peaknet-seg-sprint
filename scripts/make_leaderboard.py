"""Rebuild results/leaderboard/result.json and the README.md leaderboard table from the
model statements' result.json files (numbers copied exactly, sorted by test peak F1).

usage (repo root): python scripts/make_leaderboard.py
"""
import json
import os
import re

MODELS = ["peaknet_baseline", "hf_segformer", "hf_upernet", "lucid_segformer", "lucid_unet", "best_model"]
BEGIN, END = "<!-- leaderboard:begin -->", "<!-- leaderboard:end -->"


def rows():
    out = []
    for s in MODELS:
        p = "results/%s/result.json" % s
        if not os.path.isfile(p):
            continue
        d = json.load(open(p))
        ev = d["eval"]
        out.append({
            "statement": s,
            "model": d.get("model", ""),
            "peak_f1": ev["peak"]["f1"],
            "peak_precision": ev["peak"].get("precision"),
            "peak_recall": ev["peak"].get("recall"),
            "pixel_iou": ev["pixel"]["iou"],
            "pixel_f1": ev["pixel"].get("f1"),
            "n_test_images": ev.get("n_images"),
            "peak_tol_px": ev.get("peak_tol_px"),
            "threshold": ev.get("threshold", ev.get("thr")),
            "train_gpus": d.get("train", {}).get("gpus"),
            "train_wallclock_min": d.get("train", {}).get("wallclock_min"),
            "upstream": [u.get("url") for u in d.get("upstream", [])],
            "checkpoint": d.get("checkpoint"),
            "eval_log": ev.get("remote_log"),
        })
    out.sort(key=lambda r: r["peak_f1"], reverse=True)
    return out


def fmt(x, nd=4):
    return "-" if x is None else ("%.*f" % (nd, x) if isinstance(x, float) else str(x))


def table(rs):
    lines = ["| Rank | Statement | Model | Test peak F1 (3 px) | Peak P | Peak R | Pixel IoU | Test images |",
             "|---:|---|---|---:|---:|---:|---:|---:|"]
    for i, r in enumerate(rs, 1):
        lines.append("| %d | `%s` | %s | %s | %s | %s | %s | %s |" % (
            i, r["statement"], r["model"].replace("|", "/"), fmt(r["peak_f1"]), fmt(r["peak_precision"]),
            fmt(r["peak_recall"]), fmt(r["pixel_iou"]), fmt(r["n_test_images"])))
    return "\n".join(lines)


def main():
    rs = rows()
    os.makedirs("results/leaderboard", exist_ok=True)
    d = {"statement": "leaderboard",
         "metric": "test peak F1 at 3 px tolerance (eval.peak.f1); pixel IoU second",
         "source": "copied from results/<statement>/result.json on main by scripts/make_leaderboard.py",
         "entries": rs}
    json.dump(d, open("results/leaderboard/result.json", "w"), indent=1)
    open("results/leaderboard/leaderboard.md", "w").write(table(rs) + "\n")
    block = "%s\n%s\n\nRebuilt by `python scripts/make_leaderboard.py` from each model's `results/<statement>/result.json`. Whole test split, peak tolerance 3 px.\n%s" % (BEGIN, table(rs), END)
    readme = open("README.md").read()
    if BEGIN in readme:
        readme = re.sub(re.escape(BEGIN) + ".*?" + re.escape(END), lambda m: block, readme, flags=re.S)
    else:
        readme = readme.replace("(filled in by the `leaderboard` statement)", block)
        if BEGIN not in readme:
            readme += "\n## Leaderboard\n\n" + block + "\n"
    open("README.md", "w").write(readme)
    print(table(rs))


if __name__ == "__main__":
    main()
