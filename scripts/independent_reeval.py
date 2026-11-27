"""Independent read-back of a model's test metrics, for the proof report.

Shares no code path with seg/evaluate.py or seg/data.py: it reads the ground-truth labels
straight from the Zarr stores, reads the saved probability maps (<store>__<idx>.npy written by
`python -m seg.evaluate ... --save-probs DIR` or the ensemble script), thresholds them, and
scores them with seg.viz.match_peaks (max-cardinality bipartite matching on a KD-tree, not the
Hungarian code in seg/evaluate.py) plus its own pixel counts. Micro-averaged over the split.

usage: python scripts/independent_reeval.py --probs DIR --thr T [--split test] [--procs 8] --out reeval.json
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seg.viz import match_peaks  # noqa: E402

DATA = "/sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet10k/peaknet10k"
_A = {}


def _one(item):
    import zarr
    store, i = item
    z = zarr.open(os.path.join(_A["root"], store), "r")
    gt = np.asarray(z["labels"][i]) == _A["peak_label"]
    p = os.path.join(_A["probs"], "%s__%d.npy" % (store, i))
    if not os.path.exists(p):
        return store, i, None
    prob = np.load(p).astype(np.float32)
    if prob.shape != gt.shape:
        return store, i, "shape %s vs %s" % (prob.shape, gt.shape)
    pred = prob > _A["thr"]
    pc, gc, pairs = match_peaks(pred, gt, _A["tol"])
    tp = len(pairs)
    return store, i, dict(ptp=tp, pfp=len(pc) - tp, pfn=len(gc) - tp,
                          xtp=int((pred & gt).sum()), xfp=int((pred & ~gt).sum()), xfn=int((~pred & gt).sum()))


def prf(tp, fp, fn):
    p = 1.0 if tp + fp == 0 else tp / (tp + fp)
    r = 1.0 if tp + fn == 0 else tp / (tp + fn)
    return p, r, (0.0 if p + r == 0 else 2 * p * r / (p + r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--split-json", default="results/split/result.json")
    ap.add_argument("--root", default=DATA)
    ap.add_argument("--tol", type=float, default=3.0)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out")
    a = ap.parse_args()
    sp = json.load(open(a.split_json))
    _A.update(root=a.root, probs=a.probs, thr=a.thr, tol=a.tol, peak_label=sp.get("peak_label", 1))
    import zarr
    items = []
    for s in sp["stores"][a.split]:
        n = zarr.open(os.path.join(a.root, s), "r")["labels"].shape[0]
        items += [(s, i) for i in range(n)]
    t0 = time.time()
    print("host=%s SLURM_JOB_ID=%s probs=%s thr=%.4f images=%d" % (os.uname().nodename, os.environ.get("SLURM_JOB_ID"),
                                                                  a.probs, a.thr, len(items)), flush=True)
    tot = dict(ptp=0, pfp=0, pfn=0, xtp=0, xfp=0, xfn=0)
    missing, bad, n = [], [], 0
    with Pool(a.procs) as pool:
        for store, i, r in pool.imap_unordered(_one, items, chunksize=4):
            if r is None:
                missing.append("%s__%d" % (store, i))
            elif isinstance(r, str):
                bad.append("%s__%d %s" % (store, i, r))
            else:
                n += 1
                for k in tot:
                    tot[k] += r[k]
            if (n + len(missing)) % 200 == 0:
                print("  %d/%d" % (n + len(missing), len(items)), flush=True)
    pp, pr, pf = prf(tot["ptp"], tot["pfp"], tot["pfn"])
    xp, xr, xf = prf(tot["xtp"], tot["xfp"], tot["xfn"])
    xden = tot["xtp"] + tot["xfp"] + tot["xfn"]
    res = {"split": a.split, "n_images_expected": len(items), "n_images_scored": n, "n_missing": len(missing),
           "missing_examples": missing[:5], "bad": bad[:5], "thr": a.thr, "peak_tol_px": a.tol,
           "peak": {"f1": pf, "precision": pp, "recall": pr, "tp": tot["ptp"], "fp": tot["pfp"], "fn": tot["pfn"]},
           "pixel": {"iou": 1.0 if xden == 0 else tot["xtp"] / xden, "f1": xf, "precision": xp, "recall": xr},
           "scorer": "scripts/independent_reeval.py (zarr labels + seg.viz.match_peaks)", "minutes": (time.time() - t0) / 60}
    print(json.dumps(res), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
