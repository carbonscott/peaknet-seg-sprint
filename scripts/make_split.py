"""Scan the peaknet10k v2 Zarr stores and write the fixed run-level train/val/test split.

usage (on S3DF, shared venv):
    python scripts/make_split.py --out <dir>      # writes <dir>/scan.json and <dir>/split.json

Stores are grouped by run ({exp}_{run}); every store of one run goes to one set (no leakage).
Label ids are sampled from the first and last image of every store.
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np
import zarr

DATA = "/sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet10k/peaknet10k"
NAME = re.compile(r"^(.+)_peaknet\.(\d{4})\.zarr$")

# Run-level assignment (10 runs in 2 experiments). Chosen so that test holds >= 1000 images from
# both experiments and val/test are each roughly 8-10% of the images.
TEST_RUNS = ["mfxl1025422_r0309", "mfxl1027522_r0028"]
VAL_RUNS = ["mfxl1025422_r0310"]


def scan(root, sample_labels=True):
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "*_peaknet.*.zarr"))):
        b = os.path.basename(p)
        m = NAME.match(b)
        if not m or ".v3." in b:
            continue
        g = zarr.open(p, "r")
        ish, lsh = g["images"].shape, g["labels"].shape
        row = dict(store=b, run=m.group(1), n=int(ish[0]), H=int(ish[1]), W=int(ish[2]),
                   label_shape=list(lsh), label_dtype=str(g["labels"].dtype))
        if sample_labels:
            vals, pos = set(), []
            for i in sorted({0, ish[0] - 1}):
                lab = g["labels"][i]
                u = np.unique(lab)
                vals.update(int(x) for x in u)
                pos.append(int((lab > 0).sum()))
            row["label_values"] = sorted(vals)
            row["pos_px_sample"] = pos
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DATA)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    rows = scan(a.root)
    json.dump(rows, open(os.path.join(a.out, "scan.json"), "w"), indent=1)
    print("scanned %d stores in %.0fs" % (len(rows), time.time() - t0), flush=True)

    stores = {"train": [], "val": [], "test": []}
    n = {"train": 0, "val": 0, "test": 0}
    for r in rows:
        k = "test" if r["run"] in TEST_RUNS else "val" if r["run"] in VAL_RUNS else "train"
        stores[k].append(r["store"])
        n[k] += r["n"]
    labels = sorted({v for r in rows for v in r.get("label_values", [])})
    tot = sum(n.values())
    runs = {k: sorted({r["run"] for r in rows if r["store"] in set(stores[k])}) for k in stores}
    shapes = sorted({(r["H"], r["W"]) for r in rows})
    man = dict(statement="split", root=a.root, stores=stores, n_images=n, runs=runs,
               label_values=labels, peak_label=1, shapes=[list(s) for s in shapes],
               fractions={k: round(v / tot, 4) for k, v in n.items()})
    json.dump(man, open(os.path.join(a.out, "split.json"), "w"), indent=1)
    print(json.dumps(dict(n_images=n, fractions=man["fractions"], runs=runs, labels=labels,
                          shapes=man["shapes"])), flush=True)


if __name__ == "__main__":
    main()
