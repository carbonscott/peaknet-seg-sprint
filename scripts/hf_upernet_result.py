"""Write results/hf_upernet/result.json from a whole-test metrics JSON (fetched from S3DF).

usage (repo root): python3 scripts/hf_upernet_result.py <local metrics.json> <exp> <slurm_step> <wallclock_min> \
                       [--model "..."] [--notes "..."] [--proof results/hf_upernet/x.png ...] [--train-images N]
Paths in the result point at the durable copies under /sdf/.../results/peaknet-seg-sprint/{checkpoints,evals,logs}/h4/<exp>/.
"""
import argparse
import json

D = "/sdf/data/lcls/ds/prj/prjcwang31/results/peaknet-seg-sprint"

ap = argparse.ArgumentParser()
ap.add_argument("metrics")
ap.add_argument("exp")
ap.add_argument("jobid")
ap.add_argument("wallclock_min", type=float)
ap.add_argument("--tag", default="test")
ap.add_argument("--model", default="UperNet (HF transformers) + ConvNeXt-tiny backbone, ADE20k-pretrained, "
                                     "1-ch log1p/std adapter, PixelShuffle(4) full-res head, focal+dice")
ap.add_argument("--config", default="configs/hf_upernet.yaml")
ap.add_argument("--gpus", type=int, default=1)
ap.add_argument("--train-images", type=int, default=15989)
ap.add_argument("--proof", nargs="+", default=[])
ap.add_argument("--notes", default="")
ap.add_argument("--eval-cmd", default=None)
ap.add_argument("--val-pick", default=None, help="local copy of val_pick_<tag>.json")
a = ap.parse_args()

m = json.load(open(a.metrics))
assert m["split"] == "test", m["split"]
ev_dir = "%s/evals/h4/%s" % (D, a.exp)
cmd = a.eval_cmd or ("python -m seg.evaluate --model hf_upernet --config %s --ckpt %s/checkpoints/h4/%s/best.pt "
                     "--split test --out %s/metrics_%s.json" % (a.config, D, a.exp, ev_dir, a.tag))
res = {
    "statement": "hf_upernet",
    "model": a.model,
    "upstream": [
        {"url": "https://huggingface.co/openmmlab/upernet-convnext-tiny", "ref": "transformers 4.57.6"},
        {"url": "https://github.com/huggingface/transformers", "ref": "v4.57.6 (UperNetForSemanticSegmentation)"},
    ],
    "code": ["seg/models/hf_upernet.py", a.config, "seg/train.py", "seg/losses.py", "seg/infer.py",
             "seg/evaluate.py", "seg/data.py", "scripts/hf_upernet_launch.sh", "scripts/hf_upernet_eval.sh",
             "scripts/hf_upernet_evalstep.sh"],
    "train": {"slurm_jobid": a.jobid, "gpus": a.gpus, "wallclock_min": a.wallclock_min,
              "train_images": a.train_images, "config": a.config,
              "remote_log": "%s/logs/h4/%s/train.log" % (D, a.exp),
              "metrics_jsonl": "%s/logs/h4/%s/metrics.jsonl" % (D, a.exp)},
    "checkpoint": "%s/checkpoints/h4/%s/best.pt" % (D, a.exp),
    "eval": {"split": "test", "n_images": m["n_images"], "peak_tol_px": m["peak_tol_px"],
             "threshold": m.get("threshold"),
             "pixel": {k: m["pixel"][k] for k in ("iou", "f1", "precision", "recall")},
             "peak": {k: m["peak"][k] for k in ("f1", "precision", "recall")},
             "counts": {"peak": {k: m["peak"].get(k) for k in ("tp", "fp", "fn")},
                        "pixel": {k: m["pixel"].get(k) for k in ("tp", "fp", "fn")}},
             "eval_cmd": cmd,
             "remote_log": "%s/eval_%s.log" % (ev_dir, a.tag),
             "metrics_json": "%s/metrics_%s.json" % (ev_dir, a.tag),
             "threshold_from": "val (first 400 val images; sweep 0.5..0.999 over best.pt/last.pt and "
                               "PK_MASK_SATURATED 0/1; see val_pick_%s.json)" % a.tag,
             "val_pick": json.load(open(a.val_pick)) if a.val_pick else None},
    "proof": a.proof,
    "notes": a.notes,
}
import os
os.makedirs("results/hf_upernet", exist_ok=True)
json.dump(res, open("results/hf_upernet/result.json", "w"), indent=1)
print(json.dumps(res["eval"], indent=1))
