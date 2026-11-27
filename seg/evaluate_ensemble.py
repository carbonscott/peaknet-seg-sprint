"""Evaluate an ensemble of checkpoints (mean of sigmoid probs, optional flip TTA) with the team metric.

The threshold is chosen on VAL (evenly spaced subset), then the WHOLE test split is evaluated at that
threshold with seg.evaluate.Accumulator (the same counting code as `python -m seg.evaluate`).

usage:
    python -m seg.evaluate_ensemble \
        --member hf_segformer:/sdf/.../best.pt --member smp_unet:/sdf/.../best.pt[:configs/smp_unet.yaml] \
        [--weights 1,1] [--tta] [--tile 1024] [--val-images 400] [--thr 0.5] \
        --out ens_test.json [--save-overlays DIR --n-overlays 4] [--limit N]
A single --member with --tta is "checkpoint + TTA + val-picked threshold".
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np

from seg.evaluate import Accumulator, _load_ckpt, _load_yaml, _state_dict, _strip_prefix

SWEEP = [round(x, 3) for x in np.arange(0.1, 0.951, 0.05)] + [0.97, 0.98, 0.99]


def load_member(spec, device):
    import torch
    parts = spec.split(":")
    name, ckpt = parts[0], parts[1]
    cfg = _load_yaml(parts[2]) if len(parts) > 2 and parts[2] else None
    ck = _load_ckpt(ckpt)
    if not cfg:
        cfg = ck.get("cfg", {}) if isinstance(ck, dict) else {}
    mod = importlib.import_module("seg.models.%s" % name)
    model = mod.build_model(cfg)
    miss, unexp = model.load_state_dict(_strip_prefix(_state_dict(ck)), strict=False)
    if miss:
        raise SystemExit("%s: checkpoint does not match the model (missing %s ...)" % (spec, miss[:5]))
    model.to(device).eval()
    thr = ck.get("thr") if isinstance(ck, dict) else None
    return dict(name=name, ckpt=ckpt, model=model, ckpt_thr=thr, val=ck.get("val") if isinstance(ck, dict) else None)


def make_loader(stores, idxs=None, workers=6):
    import torch
    from seg.data import PeakDataset
    ds = PeakDataset(stores)
    if idxs is not None:
        ds = torch.utils.data.Subset(ds, idxs)
    return torch.utils.data.DataLoader(ds, batch_size=1, num_workers=workers, collate_fn=lambda b: b[0])


def ens_prob(members, weights, img, tta, tile, mask_saturated=None):
    import torch
    from seg.infer import predict_prob
    dev = next(members[0]["model"].parameters()).device
    x = torch.as_tensor(img, dtype=torch.float32).reshape(1, 1, *img.shape[-2:]).to(dev)
    acc = 0
    for m, w in zip(members, weights):
        acc = acc + w * predict_prob(m["model"], x, tile=tile, tta=tta, mask_saturated=mask_saturated)
    return (acc / sum(weights))[0, 0].float().cpu().numpy()


def run(members, weights, loader, thrs, peak_label, tta, tile, log_every=100, overlays=None, n_overlays=0, ms=None):
    acc = Accumulator(thrs)
    t0 = time.time()
    for i, it in enumerate(loader):
        img = np.asarray(it["image"], dtype=np.float32)
        img = img[0] if img.ndim == 3 else img
        tb = np.asarray(it["label"]) == peak_label
        prob = ens_prob(members, weights, img, tta, tile, ms)
        acc.add_prob(prob, tb)
        if overlays is not None and len(overlays) < n_overlays and i % 97 == 0:
            overlays.append((img, tb, prob, it["meta"]))
        if (i + 1) % log_every == 0:
            r = acc.result(thrs[len(thrs) // 2])
            print("[ensemble] %d images %.0f s; thr %.2f peak F1 %.4f" % (i + 1, time.time() - t0, thrs[len(thrs) // 2],
                                                                         r["peak"]["f1"]), file=sys.stderr, flush=True)
    return acc


def main(argv=None):
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", action="append", required=True, help="model:ckpt[:config]")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--mask-saturated", type=int, default=None, help="dilation px for saturated-pixel suppression (0=off; default env PK_MASK_SATURATED)")
    ap.add_argument("--split-json", default="results/split/result.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--val-images", type=int, default=400, help="val images (evenly spaced) for the thr sweep")
    ap.add_argument("--thr", type=float, default=None, help="skip the val sweep and use this threshold")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-overlays", default=None)
    ap.add_argument("--n-overlays", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    from seg.data import load_split
    split = load_split(a.split_json)
    peak_label = int(json.load(open(a.split_json)).get("peak_label", 1))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members = [load_member(s, dev) for s in a.member]
    weights = [float(w) for w in a.weights.split(",")] if a.weights else [1.0] * len(members)
    assert len(weights) == len(members)
    t0 = time.time()
    out = {"members": [dict(name=m["name"], ckpt=m["ckpt"], ckpt_thr=m["ckpt_thr"]) for m in members],
           "weights": weights, "tta": a.tta, "tile": a.tile, "mask_saturated": a.mask_saturated, "argv": sys.argv if argv is None else argv}

    if a.thr is None:
        from seg.data import PeakDataset
        nval = len(PeakDataset(split["val"]))
        idxs = np.linspace(0, nval - 1, min(a.val_images, nval)).round().astype(int).tolist()
        vacc = run(members, weights, make_loader(split["val"], idxs, a.workers), SWEEP, peak_label, a.tta, a.tile, ms=a.mask_saturated)
        thr = vacc.best_threshold()
        out["val"] = {"n_images": vacc.n, "threshold": thr, "sweep": {str(t): vacc.result(t) for t in SWEEP}}
        print("[ensemble] val: best thr %.2f peak F1 %.4f (%.0f s)" % (thr, vacc.result(thr)["peak"]["f1"],
                                                                     time.time() - t0), file=sys.stderr, flush=True)
    else:
        thr = a.thr
    stores = split[a.split]
    idxs = list(range(a.limit)) if a.limit else None
    ovl = [] if a.save_overlays else None
    tacc = run(members, weights, make_loader(stores, idxs, a.workers), [thr], peak_label, a.tta, a.tile,
               overlays=ovl, n_overlays=a.n_overlays, ms=a.mask_saturated)
    res = tacc.result(thr)
    out.update({"split": a.split, "n_images": tacc.n, "peak_tol_px": tacc.tol, "threshold": thr,
                "threshold_from": "val" if a.thr is None else "given", "pixel": res["pixel"], "peak": res["peak"],
                "seconds": round(time.time() - t0, 1)})
    if ovl:
        try:
            from seg import viz
            os.makedirs(a.save_overlays, exist_ok=True)
            names = "+".join(m["name"] for m in members) + (" +TTA" if a.tta else "")
            for img, tb, prob, meta in ovl:
                st, ix = meta.get("store", "img"), int(meta.get("index", 0))
                viz.save_overlay(img, tb, prob > thr, os.path.join(a.save_overlays, "%s__%d.png" % (st, ix)),
                                 title="best_model [%s]  %s[%d]  (test)" % (names, st, ix),
                                 extra="thr=%.2f (val)  test totals: peak F1 %.4f, pixel IoU %.4f"
                                 % (thr, res["peak"]["f1"], res["pixel"]["iou"]), tol=3.0, cls=True)
        except Exception as e:
            print("[ensemble] overlays failed: %r" % (e,), file=sys.stderr)
    s = json.dumps(out, indent=1, default=str)
    print(s)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(s + "\n")
    return out


if __name__ == "__main__":
    main()
