"""Diagnose false-positive peaks of a checkpoint on a few val images.
For each FP component: size, max prob, distance to nearest TRUE peak centroid (fragments sit
within a few px of a true peak; unlabeled real peaks / artifacts sit far away), raw image
value at the FP centroid relative to the image median/MAD.
usage: python scripts/fp_diag.py --model hf_segformer --ckpt best.pt --thr 0.95 --n 6 --out diag.json
"""
import argparse
import json

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree

from seg.data import PeakDataset, load_split
from seg.evaluate import peak_details
from seg.infer import predict_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--thr", type=float, default=0.95)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import importlib
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = importlib.import_module("seg.models." + a.model).build_model(ck["cfg"])
    sd = ck.get("state_dict", ck.get("model"))
    model.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()})
    model = model.cuda().eval()
    ds = PeakDataset(load_split()["val"])
    idxs = np.linspace(0, len(ds) - 1, a.n).astype(int)
    rows, summ = [], {"fp": 0, "fp_near6": 0, "fp_size1": 0, "tp": 0, "fn": 0}
    for i in idxs:
        it = ds[int(i)]
        img = it["image"].numpy().reshape(it["label"].shape)
        lab = it["label"].numpy() == 1
        with torch.no_grad():
            prob = predict_full(model, img)
        prob = np.asarray(prob)
        pred = prob > a.thr
        det = peak_details(pred.astype(np.int64), lab.astype(np.int64))
        tc = np.concatenate([np.reshape(det["matched_true"], (-1, 2)), np.reshape(det["missed"], (-1, 2))])
        tree = cKDTree(tc) if len(tc) else None
        comp, n = ndimage.label(pred, structure=np.ones((3, 3)))
        med = np.median(img); mad = np.median(np.abs(img - med)) * 1.4826 + 1e-3
        for (y, x) in det["false_pos"]:
            yy, xx = int(round(y)), int(round(x))
            c = comp[yy, xx]
            size = int((comp == c).sum()) if c else 0
            d = float(tree.query([y, x])[0]) if tree is not None else 1e9
            z = float((img[max(0, yy - 1):yy + 2, max(0, xx - 1):xx + 2].max() - med) / mad)
            rows.append({"img": int(i), "y": yy, "x": xx, "size": size, "pmax": float(prob[comp == c].max()) if c else None,
                         "d_true": round(d, 1), "z_raw": round(z, 1)})
            summ["fp"] += 1; summ["fp_near6"] += d <= 6; summ["fp_size1"] += size <= 1
        summ["tp"] += len(det["matched_true"]); summ["fn"] += len(det["missed"])
        print(it["meta"].get("identifier", i), "tp", len(det["matched_true"]), "fn", len(det["missed"]), "fp", len(det["false_pos"]), flush=True)
    zs = np.array([r["z_raw"] for r in rows]) if rows else np.zeros(0)
    summ["fp_zraw_pct"] = np.percentile(zs, [10, 50, 90]).round(1).tolist() if len(zs) else []
    print(json.dumps(summ))
    json.dump({"summary": summ, "fps": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
