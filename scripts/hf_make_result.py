"""Write results/<statement>/result.json from a seg.evaluate metrics JSON (fetched from S3DF).

usage: python scripts/hf_make_result.py --statement hf_segformer --agent h3 --exp seg_b2_a \
         --metrics local_metrics_test.json --model "SegFormer mit-b2 ..." --jobid 47394816.40 \
         --gpus 1 --wallclock 90 --upstream https://huggingface.co/nvidia/mit-b2=transformers-4.57.6 \
         --code seg/models/hf_segformer.py,configs/hf_segformer.yaml --proof results/hf_segformer/a.png \
         --notes "..."
"""
import argparse
import json
import os

D = "/sdf/data/lcls/ds/prj/prjcwang31/results/peaknet-seg-sprint"


def main():
    ap = argparse.ArgumentParser()
    for k in ("statement", "agent", "exp", "metrics", "model", "jobid", "upstream", "code", "proof", "notes"):
        ap.add_argument("--" + k, required=k != "notes", default="")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--wallclock", type=float, required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--eval-tag", default="test")
    ap.add_argument("--thr-from", default="sweep on the whole val split (metrics_val_<tag>.json)")
    ap.add_argument("--extra", default=None, help="JSON object merged into the top level")
    a = ap.parse_args()
    m = json.load(open(a.metrics))
    split = json.load(open("results/split/result.json"))
    assert m["split"] == "test" and m["n_images"] == split["n_images"]["test"], (m["n_images"], split["n_images"])
    ups = []
    for u in a.upstream.split(","):
        url, ref = u.split("=", 1)
        ups.append({"url": url, "ref": ref})
    argv = [x for x in (m.get("argv") or []) if not x.endswith("evaluate.py")]
    ev_cmd = "python -m seg.evaluate " + " ".join(argv) if argv else "python -m seg.evaluate"
    base = "%s/%%s/%s/%s" % (D, a.agent, a.exp)
    r = {
        "statement": a.statement,
        "model": a.model,
        "upstream": ups,
        "code": a.code.split(","),
        "train": {"slurm_jobid": a.jobid, "gpus": a.gpus, "wallclock_min": a.wallclock,
                  "train_images": split["n_images"]["train"],
                  "config": a.config or "configs/%s.yaml" % a.statement,
                  "remote_log": base % "logs" + "/train.log",
                  "metrics_jsonl": base % "logs" + "/metrics.jsonl"},
        "checkpoint": base % "checkpoints" + "/best.pt",
        "eval": {"split": "test", "n_images": m["n_images"], "peak_tol_px": m["peak_tol_px"],
                 "threshold": m["threshold"], "threshold_from": a.thr_from,
                 "tta": m.get("tta", False),
                 "pixel": {k: m["pixel"][k] for k in ("iou", "f1", "precision", "recall")},
                 "peak": {k: m["peak"][k] for k in ("f1", "precision", "recall")},
                 "eval_cmd": ev_cmd,
                 "remote_log": base % "evals" + "/eval_%s.log" % a.eval_tag,
                 "metrics_json": base % "evals" + "/metrics_%s.json" % a.eval_tag},
        "proof": a.proof.split(","),
        "notes": a.notes,
    }
    if a.extra:
        r.update(json.loads(a.extra))
    os.makedirs("results/%s" % a.statement, exist_ok=True)
    with open("results/%s/result.json" % a.statement, "w") as fh:
        json.dump(r, fh, indent=1)
        fh.write("\n")
    print(json.dumps(r["eval"], indent=1))


if __name__ == "__main__":
    main()
