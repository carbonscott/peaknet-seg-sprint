"""One-pass VAL grid for any checkpoint(s): flip-TTA x saturated-pixel mask x threshold (+ min component size).

Loads the model(s) once, reads each val image once, runs the 4 flips once, and scores every combination with
seg.evaluate.Accumulator (the team metric). Prints a ranked table and the seg.evaluate command that reproduces
the best combination reachable by seg.evaluate (tta, PK_MASK_SATURATED, thr) on the whole test split.

usage (from $PK/work/<you>, on a GPU step):
    python scripts/val_grid.py --member peaknet_baseline:/sdf/.../best.pt[:configs/x.yaml] [--member ...]
        [--n 300] [--tile 1024] [--masks 0,1,2] [--out grid.json]
Several --member = mean-prob ensemble (as seg.evaluate_ensemble).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402

THRS = [0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999]


def main(argv=None):
    import torch
    from scipy import ndimage
    from seg import infer
    from seg.data import PeakDataset, load_split
    from seg.evaluate import Accumulator
    from seg.evaluate_ensemble import load_member

    ap = argparse.ArgumentParser()
    ap.add_argument("--member", action="append", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=300, help="evenly spaced images of the split (0 = all)")
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--masks", default="0,1,2", help="saturated-mask dilations to try (0 = off)")
    ap.add_argument("--min-sizes", default="0,2,3", help="drop components smaller than this (post-proc, info)")
    ap.add_argument("--thrs", default=",".join(str(t) for t in THRS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members = [load_member(s, dev) for s in a.member]
    members = [m["model"] for m in members]
    thrs = [float(t) for t in a.thrs.split(",")]
    masks = [int(m) for m in a.masks.split(",")]
    msz = [int(m) for m in a.min_sizes.split(",")]
    combos = [(tta, mk, ms) for tta in (False, True) for mk in masks for ms in msz]
    acc = {c: Accumulator(thrs) for c in combos}

    stores = load_split()[a.split]
    ds = PeakDataset(stores)
    n = len(ds) if not a.n else min(a.n, len(ds))
    idxs = np.linspace(0, len(ds) - 1, n).round().astype(int).tolist()
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idxs), batch_size=None, num_workers=a.workers,
                                     prefetch_factor=4 if a.workers else None)
    eight = np.ones((3, 3), bool)
    t0 = time.time()
    flips = [(), (-1,), (-2,), (-2, -1)]
    for k, it in enumerate(dl):
        x = it["image"].float()
        x = (x if x.dim() == 4 else x[None] if x.dim() == 3 else x[None, None]).to(dev)
        tb = it["label"].numpy() == 1
        with torch.no_grad():
            ps = []
            for d in flips:
                q = 0
                for m in members:
                    xi = torch.flip(x, d) if d else x
                    pi = infer.predict_prob(m, xi, tile=a.tile, mask_saturated=0)
                    q = q + (torch.flip(pi, d) if d else pi)
                ps.append(q / len(members))
            probs = {False: ps[0], True: sum(ps) / 4}
            for tta in (False, True):
                for mk in masks:
                    p = infer.suppress_saturated(probs[tta], x, dilate=mk) if mk else probs[tta]
                    p = p[0, 0].float().cpu().numpy()
                    for ms in msz:
                        if ms:
                            # zero components (at the lowest thr) smaller than ms px
                            lab, nl = ndimage.label(p > min(thrs), structure=eight)
                            if nl:
                                sz = np.bincount(lab.ravel(), minlength=nl + 1)
                                small = sz < ms
                                small[0] = False
                                q = np.where(small[lab], 0.0, p)
                            else:
                                q = p
                        else:
                            q = p
                        acc[(tta, mk, ms)].add_prob(q, tb)
        if (k + 1) % 50 == 0:
            print("[val_grid] %d/%d images %.0f s" % (k + 1, n, time.time() - t0), file=sys.stderr, flush=True)

    rows = []
    for c in combos:
        for t in thrs:
            r = acc[c].result(t)
            rows.append(dict(tta=c[0], mask_saturated=c[1], min_size=c[2], thr=t, peak_f1=r["peak"]["f1"],
                             peak_p=r["peak"]["precision"], peak_r=r["peak"]["recall"], pix_iou=r["pixel"]["iou"]))
    rows.sort(key=lambda r: -r["peak_f1"])
    plain = [r for r in rows if r["min_size"] == 0]  # reachable with seg.evaluate
    base = max((r for r in plain if not r["tta"] and r["mask_saturated"] == 0), key=lambda r: r["peak_f1"])
    best = plain[0]
    cmd = "%spython -m seg.evaluate --model %s --ckpt %s --split test --thr %s%s%s" % (
        "PK_MASK_SATURATED=%d " % best["mask_saturated"] if best["mask_saturated"] else "",
        a.member[0].split(":")[0], a.member[0].split(":")[1], best["thr"], " --tta" if best["tta"] else "",
        " --tile %d" % a.tile if a.tile else "")
    out = dict(split=a.split, n_images=n, members=a.member, seconds=round(time.time() - t0, 1),
               baseline_no_tta_no_mask=base, best_evaluate_reachable=best, best_overall=rows[0],
               evaluate_cmd=cmd if len(a.member) == 1 else None, top=rows[:25])
    s = json.dumps(out, indent=1)
    print(s)
    if a.out:
        open(a.out, "w").write(s + "\n")


if __name__ == "__main__":
    main()
