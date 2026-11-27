"""Shared segmentation metrics for the PeakNet sprint (team-wide protocol).

Import-time dependencies: numpy and scipy only (the merge check imports this file).
torch and the model code are imported lazily inside the CLI.

    pixel_metrics(pred, target, cls=1) -> {tp, fp, fn, precision, recall, f1, iou}
    peak_metrics(pred, target, cls=1, tol=3.0) -> {tp, fp, fn, precision, recall, f1}

Both take (H, W) or (B, H, W) label/binary masks; counts are summed over the batch
(micro average). 0/0 counts as 1.0 for precision, recall and IoU; F1 is 0 when
precision + recall is 0.

Peaks are 8-connected components of (mask == cls), each represented by its centroid. A
predicted and a true peak match when their centroids are at most `tol` px apart; matching
is one-to-one and maximises the number of matched pairs (maximum bipartite matching,
Hopcroft-Karp; equivalent to Hungarian with 0/1 costs, which is what F1 cares about).

CLI (see `python -m seg.evaluate -h`):
    python -m seg.evaluate --model hf_segformer --config configs/hf_segformer.yaml \
        --ckpt /sdf/.../best.pt --split test --out /sdf/.../metrics.json
    python -m seg.evaluate --preds /sdf/.../preds_dir --split test --out metrics.json
"""
import json
import sys

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree

PEAK_TOL_PX = 3.0
_EIGHT = np.ones((3, 3), dtype=bool)


def _ratio(a, b):
    return 1.0 if b == 0 else float(a) / float(b)


def _f1(p, r):
    return 0.0 if p + r == 0 else 2.0 * p * r / (p + r)


def _as_batch(x):
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[None]
    if x.ndim != 3:
        raise ValueError("expected (H, W) or (B, H, W), got shape %r" % (x.shape,))
    return x


def finish_pixel(tp, fp, fn):
    tp, fp, fn = int(tp), int(fp), int(fn)
    p, r = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": _f1(p, r),
            "iou": _ratio(tp, tp + fp + fn)}


def finish_peak(tp, fp, fn):
    tp, fp, fn = int(tp), int(fp), int(fn)
    p, r = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": _f1(p, r)}


def pixel_counts(pred, target, cls=1):
    """(tp, fp, fn) pixel counts for class `cls`, summed over the batch."""
    p = np.asarray(pred) == cls
    t = np.asarray(target) == cls
    if p.shape != t.shape:
        raise ValueError("pred %r and target %r shapes differ" % (p.shape, t.shape))
    tp = int(np.count_nonzero(p & t))
    return tp, int(np.count_nonzero(p)) - tp, int(np.count_nonzero(t)) - tp


def pixel_metrics(pred, target, cls=1):
    return finish_pixel(*pixel_counts(pred, target, cls))


def peak_centroids(mask2d):
    """Centroids (N, 2) of the 8-connected components of a 2D boolean mask."""
    mask2d = np.asarray(mask2d, dtype=bool)
    lab, n = ndimage.label(mask2d, structure=_EIGHT)
    if n == 0:
        return np.zeros((0, 2))
    # mean pixel coordinate per component (== ndimage.center_of_mass of the binary mask, ~50x faster)
    idx = np.flatnonzero(lab)
    lv = lab.ravel()[idx]
    y, x = np.divmod(idx, lab.shape[1])
    cnt = np.bincount(lv, minlength=n + 1)[1:].astype(np.float64)
    cy = np.bincount(lv, weights=y, minlength=n + 1)[1:] / cnt
    cx = np.bincount(lv, weights=x, minlength=n + 1)[1:] / cnt
    return np.stack([cy, cx], axis=1)


def match_centroids(pc, tc, tol=PEAK_TOL_PX):
    """One-to-one matching of predicted to true centroids within `tol` px.

    Returns (tp, fp, fn, pairs) where pairs is a list of (pred_idx, true_idx)."""
    npred, ntrue = len(pc), len(tc)
    if npred == 0 or ntrue == 0:
        return 0, npred, ntrue, []
    cand = cKDTree(pc).query_ball_tree(cKDTree(tc), r=float(tol) + 1e-9)
    rows = np.repeat(np.arange(npred), [len(c) for c in cand])
    cols = np.fromiter((j for c in cand for j in c), dtype=np.int64, count=len(rows))
    if len(rows) == 0:
        return 0, npred, ntrue, []
    g = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(npred, ntrue))
    m = maximum_bipartite_matching(g, perm_type="column")  # m[i] = true idx matched to pred i, or -1
    pairs = [(i, int(j)) for i, j in enumerate(m) if j >= 0]
    tp = len(pairs)
    return tp, npred - tp, ntrue - tp, pairs


def peak_counts(pred, target, cls=1, tol=PEAK_TOL_PX):
    """(tp, fp, fn) peak counts, summed over the batch."""
    p = _as_batch(pred) == cls
    t = _as_batch(target) == cls
    if p.shape != t.shape:
        raise ValueError("pred %r and target %r shapes differ" % (p.shape, t.shape))
    TP = FP = FN = 0
    for pi, ti in zip(p, t):
        tp, fp, fn, _ = match_centroids(peak_centroids(pi), peak_centroids(ti), tol)
        TP, FP, FN = TP + tp, FP + fp, FN + fn
    return TP, FP, FN


def peak_metrics(pred, target, cls=1, tol=PEAK_TOL_PX):
    return finish_peak(*peak_counts(pred, target, cls, tol))


def peak_details(pred2d, target2d, cls=1, tol=PEAK_TOL_PX):
    """For figures: centroids of matched/false/missed peaks in one image.

    Returns dict(matched_pred, matched_true, false_pos, missed) of (N,2) (y, x) arrays."""
    pc = peak_centroids(np.asarray(pred2d) == cls)
    tc = peak_centroids(np.asarray(target2d) == cls)
    _, _, _, pairs = match_centroids(pc, tc, tol)
    pi = np.array([a for a, _ in pairs], dtype=np.int64)
    ti = np.array([b for _, b in pairs], dtype=np.int64)
    fp = np.setdiff1d(np.arange(len(pc)), pi)
    fn = np.setdiff1d(np.arange(len(tc)), ti)
    return {"matched_pred": pc[pi], "matched_true": tc[ti], "false_pos": pc[fp], "missed": tc[fn]}


class Accumulator:
    """Streaming micro-averaged pixel + peak counts, optionally for several thresholds."""

    def __init__(self, thresholds=(0.5,), tol=PEAK_TOL_PX):
        self.thr = [float(t) for t in thresholds]
        self.tol = tol
        self.c = {t: np.zeros(6, dtype=np.int64) for t in self.thr}  # px tp fp fn, peak tp fp fn
        self.n = 0

    def add_prob(self, prob, target_bin):
        """prob: (H,W) float in [0,1]; target_bin: (H,W) bool (True = peak)."""
        prob = np.asarray(prob)
        target_bin = np.asarray(target_bin, dtype=bool)
        tc = peak_centroids(target_bin)
        for t in self.thr:
            pb = prob > t
            px = pixel_counts(pb, target_bin, cls=True)
            pk = match_centroids(peak_centroids(pb), tc, self.tol)[:3]
            self.c[t] += np.array(px + pk, dtype=np.int64)
        self.n += 1

    def result(self, t):
        c = self.c[float(t)]
        return {"pixel": finish_pixel(*c[:3]), "peak": finish_peak(*c[3:])}

    def best_threshold(self):
        return max(self.thr, key=lambda t: (self.result(t)["peak"]["f1"], -abs(t - 0.5)))


# ----------------------------------------------------------------------------- CLI

def _load_yaml(path):
    if not path:
        return {}
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_ckpt(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _state_dict(ck):
    if isinstance(ck, dict):
        for k in ("model", "state_dict", "model_state_dict", "ema"):
            if k in ck and isinstance(ck[k], dict):
                return ck[k]
    return ck


def _strip_prefix(sd):
    out = {}
    for k, v in sd.items():
        for pre in ("module.", "_orig_mod."):
            while k.startswith(pre):
                k = k[len(pre):]
        out[k] = v
    return out


def _predict_fn(args):
    """Returns f(image2d np.float32) -> prob2d np.float32, plus info dict."""
    import importlib
    import torch

    cfg = _load_yaml(args.config)
    ck = _load_ckpt(args.ckpt)
    if isinstance(ck, dict) and isinstance(ck.get("cfg"), dict) and not cfg:
        cfg = ck["cfg"]
    mod = importlib.import_module("seg.models.%s" % args.model)
    model = mod.build_model(cfg)
    missing, unexpected = model.load_state_dict(_strip_prefix(_state_dict(ck)), strict=False)
    if missing or unexpected:
        print("[evaluate] load_state_dict: %d missing, %d unexpected keys" % (len(missing), len(unexpected)),
              file=sys.stderr)
        if len(missing) > 0 and not args.allow_missing:
            raise SystemExit("checkpoint does not match the model (missing %s ...)" % missing[:5])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()
    info = {"ckpt_thr": ck.get("thr") if isinstance(ck, dict) else None, "cfg": cfg}
    amp = dev.type == "cuda" and not args.no_amp

    infer = None
    try:
        infer = importlib.import_module("seg.infer")
    except ImportError:
        pass

    @torch.no_grad()
    def f(img):
        if infer is not None and hasattr(infer, "predict_full") and not args.plain:
            kw = {}
            if args.tile:
                kw["tile"] = args.tile
            if args.tta:
                kw["tta"] = True
            with torch.autocast(dev.type, enabled=amp, dtype=torch.bfloat16):
                p = infer.predict_full(model, img, **kw)
            if hasattr(p, "detach"):
                p = p.detach().float().cpu().numpy()
            return np.asarray(p, dtype=np.float32)
        x = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))[None, None].to(dev)
        H, W = x.shape[-2:]
        ph, pw = (-H) % 32, (-W) % 32
        if ph or pw:
            x = torch.nn.functional.pad(x, (0, pw, 0, ph))
        with torch.autocast(dev.type, enabled=amp, dtype=torch.bfloat16):
            y = model(x)
        if args.tta:
            with torch.autocast(dev.type, enabled=amp, dtype=torch.bfloat16):
                y2 = model(torch.flip(x, (-1,))).flip(-1)
                y3 = model(torch.flip(x, (-2,))).flip(-2)
            y = (torch.sigmoid(y.float()) + torch.sigmoid(y2.float()) + torch.sigmoid(y3.float())) / 3
        else:
            y = torch.sigmoid(y.float())
        return y[0, 0, :H, :W].cpu().numpy()

    return f, info


def _iter_split(args, split=None, every=1):
    from seg import data as D
    sp = D.load_split(args.split_json)
    stores = sp[split or args.split]
    if args.limit_stores:
        stores = stores[: args.limit_stores]
    kw = {}
    if args.data_root:
        kw["root"] = args.data_root
    ds = D.PeakDataset(stores, **kw)
    n = len(ds) if not args.limit else min(len(ds), args.limit)
    idx = list(range(0, n, max(1, int(every))))
    if args.workers > 0:
        import torch
        from torch.utils.data import DataLoader, Subset
        it_src = DataLoader(Subset(ds, idx), batch_size=None, shuffle=False, num_workers=args.workers,
                            prefetch_factor=4, persistent_workers=False)
    else:
        it_src = (ds[i] for i in idx)
    for i, it in enumerate(it_src):
        if hasattr(it.get("image"), "numpy"):
            it = dict(it, image=it["image"].numpy(), label=it["label"].numpy())
        img = np.asarray(it["image"], dtype=np.float32)
        if img.ndim == 3:
            img = img[0]
        yield i, img, np.asarray(it["label"]), it.get("meta", {})


def _parse_thr(s):
    if s is None:
        return None
    return [float(x) for x in s.split(",")]


def main(argv=None):
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Evaluate a checkpoint (or saved predictions) on a split.")
    ap.add_argument("--model", help="seg.models.<model> with build_model(cfg)")
    ap.add_argument("--config", help="configs/<model>.yaml (default: cfg stored in the checkpoint)")
    ap.add_argument("--ckpt", help="checkpoint .pt (state dict under model/state_dict, optional thr, cfg)")
    ap.add_argument("--preds", help="instead of a model: dir of <store>__<idx>.npy probability maps")
    ap.add_argument("--split", default="test")
    ap.add_argument("--split-json", default="results/split/result.json")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--thr", default=None, help="threshold(s), comma separated; default: ckpt thr or 0.5")
    ap.add_argument("--val-thr", default=None,
                    help="comma list: sweep these on the VAL split first, pick the best peak F1, then evaluate "
                         "--split at that single threshold (one process, one model load)")
    ap.add_argument("--val-every", type=int, default=1, help="with --val-thr: use every k-th val image")
    ap.add_argument("--tol", type=float, default=PEAK_TOL_PX)
    ap.add_argument("--peak-label", type=int, default=None, help="default: split's peak_label")
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--tta", action="store_true", help="flip TTA (mean of probs)")
    ap.add_argument("--plain", action="store_true", help="ignore seg.infer; pad-to-32 full-image forward")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--workers", type=int, default=6, help="DataLoader workers for reading stores")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N images only")
    ap.add_argument("--limit-stores", type=int, default=0, help="debug: first N stores only")
    ap.add_argument("--save-probs", default=None, help="dir: save prob maps as float16 .npy")
    ap.add_argument("--save-overlays", default=None, help="dir: annotated PNGs for a few images (seg.viz)")
    ap.add_argument("--n-overlays", type=int, default=4)
    ap.add_argument("--out", default=None, help="write the metrics JSON here")
    args = ap.parse_args(argv)

    peak_label = args.peak_label
    if peak_label is None:
        try:
            peak_label = int(json.load(open(args.split_json)).get("peak_label", 1))
        except Exception:
            peak_label = 1

    if args.preds:
        import os
        info = {"ckpt_thr": None}

        def predict(img, meta):
            return np.load(os.path.join(args.preds, "%s__%d.npy" % (meta["store"], int(meta["index"])))).astype(np.float32)
    else:
        if not (args.model and args.ckpt):
            ap.error("give --model and --ckpt (or --preds)")
        f, info = _predict_fn(args)

        def predict(img, meta):
            return f(img)

    thrs = _parse_thr(args.thr)
    val_info = None
    if args.val_thr:
        vthrs = _parse_thr(args.val_thr)
        vacc = Accumulator(vthrs, tol=args.tol)
        tv = time.time()
        for i, img, lab, meta in _iter_split(args, "val", args.val_every):
            vacc.add_prob(predict(img, meta), lab == peak_label)
            if (i + 1) % 100 == 0:
                print("[evaluate] val %d images, %.1f s" % (i + 1, time.time() - tv), file=sys.stderr, flush=True)
        vbest = vacc.best_threshold()
        val_info = {"n_images": vacc.n, "every": args.val_every, "picked_thr": vbest,
                    "by_threshold": {str(t): vacc.result(t) for t in vthrs}, "seconds": round(time.time() - tv, 1)}
        print("[evaluate] val-picked thr %.4g (val peak F1 %.4f)" % (vbest, vacc.result(vbest)["peak"]["f1"]),
              file=sys.stderr, flush=True)
        thrs = [vbest]
    if thrs is None:
        thrs = [float(info["ckpt_thr"])] if info.get("ckpt_thr") is not None else [0.5]
    acc = Accumulator(thrs, tol=args.tol)
    overlays = []
    t0 = time.time()
    for i, img, lab, meta in _iter_split(args):
        prob = predict(img, meta)
        if prob.shape != lab.shape:
            raise SystemExit("prediction shape %r != label shape %r (predict at native resolution)"
                             % (prob.shape, lab.shape))
        acc.add_prob(prob, lab == peak_label)
        if args.save_probs:
            import os
            os.makedirs(args.save_probs, exist_ok=True)
            np.save(os.path.join(args.save_probs, "%s__%d.npy" % (meta.get("store", "img"), int(meta.get("index", i)))),
                    prob.astype(np.float16))
        if args.save_overlays and len(overlays) < args.n_overlays and i % 97 == 0:
            overlays.append((img, lab == peak_label, prob, meta))
        if (i + 1) % 100 == 0:
            r = acc.result(thrs[0])
            print("[evaluate] %d images, %.1f s, thr %.3g peak f1 %.4f pixel iou %.4f"
                  % (i + 1, time.time() - t0, thrs[0], r["peak"]["f1"], r["pixel"]["iou"]), file=sys.stderr, flush=True)

    best = acc.best_threshold()
    res = acc.result(best)
    out = {"split": args.split, "n_images": acc.n, "peak_tol_px": args.tol, "peak_label": peak_label,
           "threshold": best, "pixel": res["pixel"], "peak": res["peak"],
           "model": args.model, "ckpt": args.ckpt, "preds": args.preds, "tta": bool(args.tta), "tile": args.tile,
           "threshold_from": ("val sweep (--val-thr)" if val_info else ("--thr" if args.thr else "checkpoint thr (val) or 0.5")),
           "seconds": round(time.time() - t0, 1), "argv": sys.argv if argv is None else argv}
    if val_info:
        out["val"] = val_info
    if len(thrs) > 1:
        out["by_threshold"] = {str(t): acc.result(t) for t in thrs}
        out["warning"] = "threshold picked on THIS split; for test reporting pick the threshold on val"
    if args.save_overlays and overlays:
        try:
            from seg import viz
            import os
            os.makedirs(args.save_overlays, exist_ok=True)
            for img, tb, prob, meta in overlays:
                st, ix = meta.get("store", "img"), int(meta.get("index", 0))
                viz.save_overlay(img, tb, prob > best, os.path.join(args.save_overlays, "%s__%d.png" % (st, ix)),
                                 title="%s  %s[%d]  (%s)" % (args.model or "preds", st, ix, args.split),
                                 extra="thr=%.3g  split totals: peak F1 %.4f, pixel IoU %.4f"
                                 % (best, res["peak"]["f1"], res["pixel"]["iou"]), tol=args.tol, cls=True)
        except Exception as e:  # figures must never kill an eval
            print("[evaluate] overlays failed: %r" % (e,), file=sys.stderr)
    s = json.dumps(out, indent=1, default=str)
    print(s)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(s + "\n")
    return out


if __name__ == "__main__":
    main()
