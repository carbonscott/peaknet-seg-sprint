"""Write results/lucid_unet/result.json from a finished run's real logs (fetched from S3DF).
usage: python3 scripts/lucid_unet_result.py <exp> <slurm_step> <metrics.jsonl> <test_metrics.json> <proof.png>... [--notes "..."]
stdlib only. Paths in the JSON point at the durable copies under results/peaknet-seg-sprint.
"""
import json
import sys

args = sys.argv[1:]
notes = ""
if "--notes" in args:
    i = args.index("--notes")
    notes = args[i + 1]
    args = args[:i] + args[i + 2:]
exp, step, mjsonl, tmet = args[:4]
proofs = args[4:]
DUR = "/sdf/data/lcls/ds/prj/prjcwang31/results/peaknet-seg-sprint"
ev = [json.loads(l) for l in open(mjsonl) if l.strip().startswith("{")]
start = next(e for e in ev if e.get("event") == "start")
data = next(e for e in ev if e.get("event") == "data")
end = next((e for e in ev if e.get("event") == "end"), None)
vals = [e for e in ev if e.get("event") == "val"]
wall = end["wallclock_min"] if end else round(ev[-1]["time"] / 60, 1)
t = json.load(open(tmet))
split = json.load(open("results/split/result.json"))
assert t["n_images"] == split["n_images"]["test"], (t["n_images"], split["n_images"]["test"])
m = start["cfg"]["model"]
tr = start["cfg"]["train"]
r = {
    "statement": "lucid_unet",
    "model": "lucidrains x-unet XUnet 2D (dim %d, mults %s, attn at 1/8 + mid), 1-ch asinh adapter, "
             "full-res head, %s loss" % (m["dim"], m["dim_mults"], start["cfg"]["loss"]["type"]),
    "upstream": [{"url": "https://github.com/lucidrains/x-unet", "ref": "x-unet 0.5.0 (PyPI)"},
                 {"url": "https://github.com/lucidrains/x-unet/blob/main/x_unet/x_unet.py",
                  "ref": "XUnet class used unmodified; attention -> SDPA, 1-frame Conv3d -> conv2d at runtime"}],
    "code": ["seg/models/lucid_unet.py", "configs/lucid_unet.yaml", "seg/train.py", "seg/losses.py",
             "seg/data.py", "seg/infer.py", "seg/evaluate.py", "scripts/lucid_unet_run.sh"],
    "train": {"slurm_jobid": step, "gpus": int(start.get("world", 1)), "wallclock_min": wall,
              "train_images": int(data["train_images"]), "config": "configs/lucid_unet.yaml",
              "remote_log": "%s/logs/h6/%s/train.log" % (DUR, exp),
              "metrics_jsonl": "%s/logs/h6/%s/metrics.jsonl" % (DUR, exp),
              "steps": end["steps"] if end else vals[-1]["step"], "crop": tr["crop"],
              "batch_size": tr["batch_size"], "lr": tr["lr"],
              "best_val": end["best"] if end else None},
    "checkpoint": "%s/checkpoints/h6/%s/best.pt" % (DUR, exp),
    "eval": {"split": "test", "n_images": t["n_images"], "peak_tol_px": 3, "threshold": t["threshold"],
             "threshold_source": "val (ckpt thr from seg.train sweep)",
             "pixel": {k: t["pixel"][k] for k in ("iou", "f1", "precision", "recall")},
             "peak": {k: t["peak"][k] for k in ("f1", "precision", "recall")},
             "counts": {"peak_tp": t["peak"].get("tp"), "peak_fp": t["peak"].get("fp"), "peak_fn": t["peak"].get("fn")},
             "tile": t.get("tile"), "tta": t.get("tta"),
             "eval_cmd": "python -m seg.evaluate " + " ".join(t["argv"]) if isinstance(t.get("argv"), list) else str(t.get("argv")),
             "remote_log": "%s/evals/h6/%s/eval.log" % (DUR, exp),
             "metrics_json": "%s/evals/h6/%s/test_metrics.json" % (DUR, exp)},
    "proof": proofs,
    "notes": notes,
}
open("results/lucid_unet/result.json", "w").write(json.dumps(r, indent=1) + "\n")
print(json.dumps(r["eval"]["peak"]), r["eval"]["pixel"]["iou"])
