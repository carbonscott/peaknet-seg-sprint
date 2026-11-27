"""Write results/best_model/result.json from an S3DF eval JSON (seg.evaluate or seg.evaluate_ensemble output).

usage: python scripts/make_best_model.py --eval-json local_copy.json --spec spec.json
spec.json holds everything that is not a metric: model, based_on, upstream, code, train{...}, checkpoint,
eval_cmd, eval_remote_log, proof[...], notes. Metrics are copied verbatim from the eval JSON.
Refuses to write if n_images != split test size or if another model statement on disk has a higher peak F1.
"""
import argparse
import json
import os

MODELS = ["peaknet_baseline", "hf_segformer", "hf_upernet", "lucid_segformer", "lucid_unet"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ev = json.load(open(a.eval_json))
    sp = json.load(open(a.spec))
    ntest = json.load(open("results/split/result.json"))["n_images"]["test"]
    assert ev["n_images"] == ntest, "eval covers %d images, test has %d" % (ev["n_images"], ntest)
    assert ev.get("split", "test") == "test"
    f1 = ev["peak"]["f1"]
    for m in MODELS:
        p = "results/%s/result.json" % m
        if os.path.isfile(p):
            o = json.load(open(p))["eval"]["peak"]["f1"]
            if o > f1 and not a.force:
                raise SystemExit("best_model F1 %.4f < %s %.4f" % (f1, m, o))
    r = {"statement": "best_model", "model": sp["model"], "based_on": sp["based_on"], "upstream": sp["upstream"],
         "code": sp["code"], "train": sp["train"], "checkpoint": sp["checkpoint"],
         "eval": {"split": "test", "n_images": ev["n_images"], "peak_tol_px": 3,
                  "threshold": ev.get("threshold"), "threshold_from": ev.get("threshold_from", "ckpt (val)"),
                  "tta": ev.get("tta", False),
                  "pixel": {k: ev["pixel"][k] for k in ("iou", "f1", "precision", "recall")},
                  "peak": {k: ev["peak"][k] for k in ("f1", "precision", "recall")},
                  "counts": {"pixel": {k: ev["pixel"][k] for k in ("tp", "fp", "fn")},
                             "peak": {k: ev["peak"][k] for k in ("tp", "fp", "fn")}},
                  "eval_cmd": sp["eval_cmd"], "remote_log": sp["eval_remote_log"],
                  "metrics_json": sp.get("eval_metrics_json")},
         "proof": sp["proof"], "notes": sp.get("notes", "")}
    for k in ("members", "weights"):
        if k in ev:
            r["eval"][k] = ev[k]
    if "val" in ev:
        r["eval"]["val_threshold_sweep_images"] = ev["val"]["n_images"]
    os.makedirs("results/best_model", exist_ok=True)
    json.dump(r, open("results/best_model/result.json", "w"), indent=1)
    print("wrote results/best_model/result.json: peak F1 %.4f" % f1)


if __name__ == "__main__":
    main()
