"""Write results/<statement>/result.json from a seg.evaluate metrics JSON (fetched from S3DF).

python3 scripts/lucid_result.py --statement lucid_segformer --metrics /tmp/m.json --exp lseg_b \
    --model "..." --upstream-url ... --upstream-ref ... --code a.py b.yaml --jobid 47394816.N \
    --wallclock 72 --train-images 15989 --config configs/x.yaml --proof results/x/a.png ... --notes "..."
Numbers are copied verbatim from the metrics JSON; paths point at the durable S3DF copies.
"""
import argparse
import json
import os

DUR = "/sdf/data/lcls/ds/prj/prjcwang31/results/peaknet-seg-sprint"

ap = argparse.ArgumentParser()
ap.add_argument("--statement", required=True)
ap.add_argument("--metrics", required=True)
ap.add_argument("--exp", required=True)
ap.add_argument("--agent", default="h5")
ap.add_argument("--model", required=True)
ap.add_argument("--upstream-url", nargs="+", required=True)
ap.add_argument("--upstream-ref", nargs="+", required=True)
ap.add_argument("--code", nargs="+", required=True)
ap.add_argument("--jobid", required=True)
ap.add_argument("--gpus", type=int, default=1)
ap.add_argument("--wallclock", type=float, required=True)
ap.add_argument("--train-images", type=int, required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--eval-cmd", required=True)
ap.add_argument("--proof", nargs="+", required=True)
ap.add_argument("--notes", default="")
ap.add_argument("--extra", default=None, help="JSON object merged into the top level")
a = ap.parse_args()

m = json.load(open(a.metrics))
assert m["split"] == "test", m["split"]
d = {
    "statement": a.statement,
    "model": a.model,
    "upstream": [{"url": u, "ref": r} for u, r in zip(a.upstream_url, a.upstream_ref)],
    "code": a.code,
    "train": {"slurm_jobid": a.jobid, "gpus": a.gpus, "wallclock_min": a.wallclock,
              "train_images": a.train_images, "config": a.config,
              "remote_log": "%s/logs/%s/%s/train.log" % (DUR, a.agent, a.exp),
              "metrics_jsonl": "%s/logs/%s/%s/metrics.jsonl" % (DUR, a.agent, a.exp)},
    "checkpoint": "%s/checkpoints/%s/%s/best.pt" % (DUR, a.agent, a.exp),
    "eval": {"split": "test", "n_images": m["n_images"], "peak_tol_px": m["peak_tol_px"],
             "threshold": m["threshold"], "threshold_from": m.get("threshold_from"),
             "tta": m.get("tta"), "tile": m.get("tile"),
             "pixel": {k: m["pixel"][k] for k in ("iou", "f1", "precision", "recall")},
             "peak": {k: m["peak"][k] for k in ("f1", "precision", "recall")},
             "counts": {"peak": {k: m["peak"].get(k) for k in ("tp", "fp", "fn")},
                        "pixel": {k: m["pixel"].get(k) for k in ("tp", "fp", "fn")}},
             "eval_cmd": a.eval_cmd,
             "remote_log": "%s/evals/%s/%s/eval.log" % (DUR, a.agent, a.exp),
             "remote_metrics": "%s/evals/%s/%s/metrics.json" % (DUR, a.agent, a.exp)},
    "proof": a.proof,
    "notes": a.notes,
}
if a.extra:
    d.update(json.loads(a.extra))
os.makedirs("results/%s" % a.statement, exist_ok=True)
json.dump(d, open("results/%s/result.json" % a.statement, "w"), indent=1)
print(json.dumps(d["eval"]["peak"]), d["eval"]["n_images"])
