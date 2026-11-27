"""Mask post-processing for peak predictions, plus a val sweep to choose it.

postprocess(prob, thr, min_size=0, close=0, min_peak_prob=None) -> bool mask
  - threshold at thr
  - close>0: binary closing with a (2*close+1)^2 square (merges fragments of one peak)
  - min_size: drop 8-connected components with fewer pixels
  - min_peak_prob: drop components whose max prob is below this (hysteresis-like)

Sweep (GPU; chooses parameters on VAL only):
  python -m seg.postproc --member smp_unet:/sdf/.../best.pt [--member ...] [--tta] --val-images 200 --out pp.json
"""
import json
import sys
import time

import numpy as np
from scipy import ndimage

from seg.evaluate import finish_peak, finish_pixel, match_centroids, peak_centroids, pixel_counts

_EIGHT = np.ones((3, 3), bool)


def postprocess(prob, thr, min_size=0, close=0, min_peak_prob=None):
    m = prob > thr
    if close:
        m = ndimage.binary_closing(m, structure=np.ones((2 * close + 1, 2 * close + 1), bool))
    if min_size or min_peak_prob is not None:
        lab, n = ndimage.label(m, structure=_EIGHT)
        if n:
            keep = np.ones(n + 1, bool)
            keep[0] = False
            if min_size:
                keep &= np.bincount(lab.ravel(), minlength=n + 1) >= min_size
            if min_peak_prob is not None:
                mx = ndimage.maximum(prob, lab, index=np.arange(n + 1))
                keep &= np.asarray(mx) >= min_peak_prob
                keep[0] = False
            m = keep[lab]
    return m


def grid(thrs):
    out = []
    for t in thrs:
        for ms in (0, 2, 3, 4, 6):
            for c in (0, 1):
                out.append(dict(thr=t, min_size=ms, close=c))
    return out


def main(argv=None):
    import argparse
    import torch
    from seg.data import load_split
    from seg.evaluate_ensemble import ens_prob, load_member, make_loader
    from seg.data import PeakDataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", action="append", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--val-images", type=int, default=200)
    ap.add_argument("--thrs", default="0.5,0.7,0.8,0.9,0.95,0.98")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    split = load_split()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members = [load_member(s, dev) for s in a.member]
    weights = [float(w) for w in a.weights.split(",")] if a.weights else [1.0] * len(members)
    n = len(PeakDataset(split[a.split]))
    idxs = np.linspace(0, n - 1, min(a.val_images, n)).round().astype(int).tolist()
    G = grid([float(t) for t in a.thrs.split(",")])
    C = np.zeros((len(G), 6), np.int64)
    fp_stats = []
    t0 = time.time()
    for k, it in enumerate(make_loader(split[a.split], idxs)):
        img = np.asarray(it["image"], dtype=np.float32)
        img = img[0] if img.ndim == 3 else img
        tb = np.asarray(it["label"]) == 1
        prob = ens_prob(members, weights, img, a.tta, a.tile)
        tc = peak_centroids(tb)
        for gi, g in enumerate(G):
            m = postprocess(prob, **g)
            C[gi] += np.array(pixel_counts(m, tb, cls=True) + match_centroids(peak_centroids(m), tc)[:3])
        if k < 60:  # characterise false positives at thr 0.9, no postproc
            m = prob > 0.9
            pc = peak_centroids(m)
            _, _, _, pairs = match_centroids(pc, tc)
            matched = {i for i, _ in pairs}
            lab, nl = ndimage.label(m, structure=_EIGHT)
            sizes = np.bincount(lab.ravel(), minlength=nl + 1)[1:]
            mx = np.asarray(ndimage.maximum(prob, lab, index=np.arange(1, nl + 1))) if nl else np.zeros(0)
            from scipy.spatial import cKDTree
            d = cKDTree(tc).query(pc)[0] if len(tc) and len(pc) else np.full(len(pc), np.inf)
            for i in range(len(pc)):
                if i not in matched:
                    fp_stats.append((int(sizes[i]), float(mx[i]), float(d[i])))
        if (k + 1) % 50 == 0:
            print("[postproc] %d images %.0f s" % (k + 1, time.time() - t0), file=sys.stderr, flush=True)
    res = []
    for g, c in zip(G, C):
        r = dict(g)
        r.update(peak=finish_peak(*c[3:]), pixel=finish_pixel(*c[:3]))
        res.append(r)
    res.sort(key=lambda r: -r["peak"]["f1"])
    fs = np.array(fp_stats) if fp_stats else np.zeros((0, 3))
    summ = {}
    if len(fs):
        summ = {"n_fp": len(fs), "size_hist": np.bincount(np.minimum(fs[:, 0].astype(int), 10)).tolist(),
                "dist_to_gt_quantiles": np.quantile(fs[:, 2], [0.1, 0.25, 0.5, 0.75, 0.9]).round(2).tolist(),
                "frac_within_6px": float((fs[:, 2] <= 6).mean()), "maxprob_median": float(np.median(fs[:, 1]))}
    out = {"split": a.split, "n_images": len(idxs), "members": a.member, "tta": a.tta, "top": res[:15],
           "baseline": [r for r in res if r["min_size"] == 0 and r["close"] == 0], "fp_at_0.9": summ}
    s = json.dumps(out, indent=1)
    print(s)
    if a.out:
        open(a.out, "w").write(s + "\n")


if __name__ == "__main__":
    main()
