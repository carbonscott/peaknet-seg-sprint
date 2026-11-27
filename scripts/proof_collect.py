"""Collect S3DF evidence figures for the proof report in one pass (run on S3DF, repo root,
with results/<statement>/result.json copied in). Writes PNGs + evidence_auto.json into --out.

  python scripts/proof_collect.py --out /sdf/.../figures/h8/proof [--jobs 47394816 ...]

Per model with a result: training curve (metrics.jsonl next to train.remote_log, or
train.metrics_jsonl), eval-log excerpt, checkpoint listing. Global: squeue + sacct capture,
all-model curves, leaderboard chart. Captions record the source paths.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

MODELS = ["peaknet_baseline", "hf_segformer", "hf_upernet", "lucid_segformer", "lucid_unet", "best_model"]
HERE = os.path.dirname(os.path.abspath(__file__))


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout
    except Exception as e:
        return "error: %r" % e


def figs(*args):
    r = subprocess.run([sys.executable, os.path.join(HERE, "proof_figs.py")] + list(args), capture_output=True, text=True)
    if r.returncode:
        print("proof_figs failed:", args[:3], r.stderr[-400:], flush=True)
        return False
    return True


def metrics_path(d):
    tr = d.get("train", {})
    cands = [tr.get("metrics_jsonl")] if tr.get("metrics_jsonl") else []
    if tr.get("remote_log"):
        cands.append(os.path.join(os.path.dirname(tr["remote_log"]), "metrics.jsonl"))
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", nargs="*", default=[])
    ap.add_argument("--live", nargs="*", default=[], help="NAME=metrics.jsonl of runs in progress (curves figure)")
    ap.add_argument("--logs", nargs="*", default=[], help="NAME=train.log of runs in progress (log-tail figures)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    F = []

    def add(name, section, caption):
        F.append({"file": "results/proof_report/" + name, "remote": os.path.join(a.out, name), "section": section, "caption": caption})

    # cluster: squeue + sacct
    t = os.path.join(a.out, "squeue.txt")
    txt = sh("date; echo; squeue -u cwang31 -o '%.10i %.10j %.8T %.10M %.10L %R'; echo; squeue -s -u cwang31 -o '%.18i %.9P %.12j %.10M %N'")
    open(t, "w").write(txt)
    if figs("text", "--in", t, "--title", "squeue -u cwang31 (allocations and running job steps)", "--source", "squeue", "--out", os.path.join(a.out, "squeue.png")):
        add("squeue.png", "cluster", "<b>squeue</b> on S3DF: the sprint's ada allocations and the job steps running when the report was built.")
    if a.jobs:
        t = os.path.join(a.out, "sacct.txt")
        open(t, "w").write(sh("sacct -j %s --format=JobID%%-16,JobName%%-12,Start,Elapsed,State%%-10,AllocTRES%%-60 | head -70" % ",".join(a.jobs)))
        if figs("text", "--in", t, "--title", "sacct: job steps of the sprint allocations (training/eval runs)", "--source", "sacct -j " + ",".join(a.jobs),
                "--out", os.path.join(a.out, "sacct.png"), "--width", "170"):
            add("sacct.png", "cluster", "<b>sacct</b> for allocation(s) %s: every training and evaluation step with its start, elapsed time and resources." % ", ".join(a.jobs))

    if os.path.isfile("results/split/result.json"):
        if figs("split", "--out", os.path.join(a.out, "split.png")):
            add("split.png", "cluster", "<b>Dataset and split</b>: images per set and experiment (run-level split, no run in two sets) and three random test images with their labelled peaks, read from the Zarr stores on S3DF.")
    live = [r for r in a.live if os.path.isfile(r.split("=", 1)[1])]
    if live:
        if figs("curves", "--run", *live, "--title", "training runs in progress (snapshot)", "--out", os.path.join(a.out, "curves_live.png")):
            add("curves_live.png", "curves", "<b>Training in progress</b> (snapshot when the page was built): validation peak F1 and loss from the live <code>metrics.jsonl</code> of each crew's run.")
    for spec in a.logs:
        name, p = spec.split("=", 1)
        if os.path.isfile(p):
            safe = name.replace("/", "_")
            if figs("text", "--in", p, "--tail", "12", "--title", "%s: training log (tail, run in progress)" % name, "--source", p,
                    "--out", os.path.join(a.out, "trainlog_%s.png" % safe), "--width", "170"):
                add("trainlog_%s.png" % safe, "logs", "<b>%s</b> live training log tail <code>%s</code>." % (name, p))
    runs = []
    for m in MODELS:
        p = "results/%s/result.json" % m
        if not os.path.isfile(p):
            p = "results/proof_report/pending_%s.json" % m
        if not os.path.isfile(p):
            continue
        d = json.load(open(p))
        mp = metrics_path(d)
        if mp:
            runs.append("%s=%s" % (m, mp))
            if figs("curves", "--run", "%s=%s" % (m, mp), "--title", "%s: training curves" % m, "--out", os.path.join(a.out, "curves_%s.png" % m)):
                add("curves_%s.png" % m, m, "<b>%s</b> training curves from <code>%s</code> (val peak F1 at the val-picked threshold; training loss)." % (m, mp))
        el = d.get("eval", {}).get("remote_log")
        if el and os.path.isfile(el):
            if figs("text", "--in", el, "--tail", "14", "--title", "%s: evaluation log (tail)" % m, "--source", el, "--out", os.path.join(a.out, "evallog_%s.png" % m)):
                add("evallog_%s.png" % m, m, "<b>%s</b> evaluation log tail <code>%s</code>: the whole-test metrics JSON printed by the eval command." % (m, el))
        ck = d.get("checkpoint")
        if ck and os.path.exists(ck):
            t = os.path.join(a.out, "ckpt_%s.txt" % m)
            open(t, "w").write(sh("ls -la --time-style=+%%Y-%%m-%%dT%%H:%%M %s; echo; ls -la --time-style=+%%Y-%%m-%%dT%%H:%%M %s | head -20; sha256sum %s | cut -c1-64"
                                  % (ck, os.path.dirname(ck), ck)))
            if figs("text", "--in", t, "--title", "%s: durable checkpoint" % m, "--source", ck, "--out", os.path.join(a.out, "ckpt_%s.png" % m), "--width", "160"):
                add("ckpt_%s.png" % m, m, "<b>%s</b> checkpoint on durable S3DF storage (listing + sha256)." % m)
    if len(runs) > 1:
        if figs("curves", "--run", *runs, "--title", "all models: validation peak F1 and training loss", "--out", os.path.join(a.out, "curves_all.png")):
            add("curves_all.png", "curves", "<b>All models</b>: validation peak F1 and training loss vs step, from each run's <code>metrics.jsonl</code>.")
    if os.path.isfile("results/leaderboard/result.json"):
        if figs("leaderboard", "--lb", "results/leaderboard/result.json", "--out", os.path.join(a.out, "leaderboard.png")):
            add("leaderboard.png", "leaderboard", "<b>Leaderboard</b> chart rendered on S3DF from <code>results/leaderboard/result.json</code>.")
    json.dump({"figures": F}, open(os.path.join(a.out, "evidence_auto.json"), "w"), indent=1)
    print(json.dumps([f["file"] for f in F]))


if __name__ == "__main__":
    main()
