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

THRS = [0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9998, 0.9999]


def _score(job):
    """CPU worker: rebuild sparse prob maps, apply min-size, count with seg.evaluate.Accumulator."""
    from scipy import ndimage
    from seg.evaluate import Accumulator
    shape, tb_idx, items, thrs, msz = job
    tb = np.zeros(shape, bool)
    tb.flat[tb_idx] = True
    eight = np.ones((3, 3), bool)
    out = {}
    for (tta, mk), idx, vals in items:
        p = np.zeros(shape, np.float32)
        p.flat[idx] = vals
        for ms in msz:
            q = p
            if ms:  # zero components (at the lowest thr) smaller than ms px
                lab, nl = ndimage.label(p > min(thrs), structure=eight)
                if nl:
                    small = np.bincount(lab.ravel(), minlength=nl + 1) < ms
                    small[0] = False
                    q = np.where(small[lab], 0.0, p)
            acc = Accumulator(thrs)
            acc.add_prob(q, tb)
            out[(tta, mk, ms)] = np.stack([acc.c[t] for t in acc.thr])
    return out


def main(argv=None):
    import multiprocessing as mp
    import torch
    from seg import infer
    from seg.data import PeakDataset, load_split
    from seg.evaluate import finish_peak, finish_pixel
    from seg.evaluate_ensemble import load_member

    ap = argparse.ArgumentParser()
    ap.add_argument("--member", action="append", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=300, help="evenly spaced images of the split (0 = all)")
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--masks", default="0,1,2", help="saturated-mask dilations to try (0 = off)")
    ap.add_argument("--min-sizes", default="0", help="e.g. 0,2,3: drop components smaller than this (post-proc, "
                    "info only; not reachable with seg.evaluate; each value multiplies CPU cost)")
    ap.add_argument("--procs", type=int, default=6, help="CPU processes for metric counting")
    ap.add_argument("--thrs", default=",".join(str(t) for t in THRS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    pool = mp.get_context("fork").Pool(a.procs)  # before CUDA init; workers only use numpy/scipy

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members = [load_member(s, dev) for s in a.member]
    members = [m["model"] for m in members]
    thrs = [float(t) for t in a.thrs.split(",")]
    masks = [int(m) for m in a.masks.split(",")]
    msz = [int(m) for m in a.min_sizes.split(",")]
    masks = masks if 0 in masks else [0] + masks  # the no-mask baseline is always scored
    msz = msz if 0 in msz else [0] + msz
    combos = [(tta, mk, ms) for tta in (False, True) for mk in masks for ms in msz]
    C = {c: np.zeros((len(thrs), 6), np.int64) for c in combos}
    pending = []

    def drain(block_until):
        while len(pending) > block_until:
            for c, v in pending.pop(0).get().items():
                C[c] += v

    stores = load_split()[a.split]
    ds = PeakDataset(stores)
    n = len(ds) if not a.n else min(a.n, len(ds))
    idxs = np.linspace(0, len(ds) - 1, n).round().astype(int).tolist()
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idxs), batch_size=None, num_workers=a.workers,
                                     prefetch_factor=4 if a.workers else None)
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
            items = []
            lo = min(thrs)
            for tta in (False, True):
                for mk in masks:
                    p = infer.suppress_saturated(probs[tta], x, dilate=mk) if mk else probs[tta]
                    p = p[0, 0].float().cpu().numpy()
                    idx = np.flatnonzero(p > lo)  # only pixels above the lowest thr matter
                    items.append(((tta, mk), idx, p.flat[idx]))
        pending.append(pool.apply_async(_score, ((tb.shape, np.flatnonzero(tb), items, thrs, msz),)))
        drain(2 * a.procs)
        if (k + 1) % 50 == 0:
            print("[val_grid] %d/%d images %.0f s" % (k + 1, n, time.time() - t0), file=sys.stderr, flush=True)

    drain(0)
    pool.close()
    rows = []
    for c in combos:
        for ti, t in enumerate(thrs):
            cc = C[c][ti]
            r = {"pixel": finish_pixel(*cc[:3]), "peak": finish_peak(*cc[3:])}
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
