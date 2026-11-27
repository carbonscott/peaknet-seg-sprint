"""Annotated overlay figures for peak segmentation (shared by every model crew).

One figure per image: the full detector image with every peak circled (matched = green,
missed = red, false = yellow), plus zoomed crops around the most informative peaks with the
ground-truth mask (cyan outline) and the predicted mask (magenta outline). The image id and the
metrics are written on the figure.

Python use (numpy/scipy/matplotlib only):

    from seg.viz import save_overlay
    save_overlay(image, gt, pred, "overlay.png", title="hf_segformer  mfxl..._r0309.0000[3]",
                 extra="thr=0.42", tol=3.0, cls=1)

    image: (H,W) float detector image; gt, pred: (H,W) int/bool masks (pixels == cls are peaks,
    or pred may already be boolean). Returns the dict of metrics written on the figure.

    save_overlay_grid([(image, gt, pred, "id"), ...], "grid.png", title=...) puts several
    images (full view + 2 crops each) on one figure.

CLI (from saved predictions):

    python -m seg.viz --npz pred_0001.npz --out overlay.png --title "model / image id"
        npz keys: image, label (or gt), pred (mask) or prob (+ --thr); optional "id".
    python -m seg.viz --npz a.npz b.npz c.npz --out grid.png   (several -> grid)
"""
import argparse
import os

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree

GREEN, RED, YELLOW = "#22dd22", "#ff2222", "#ffdd00"
GT_EDGE, PRED_EDGE = "cyan", "magenta"
_EIGHT = np.ones((3, 3), bool)


# ----------------------------------------------------------------------------- matching
def centroids(mask):
    """Centroids (N,2) [y,x] of the 8-connected components of a boolean mask."""
    lab, n = ndimage.label(mask, structure=_EIGHT)
    if n == 0:
        return np.zeros((0, 2))
    return np.asarray(ndimage.center_of_mass(mask, lab, np.arange(1, n + 1)), float).reshape(-1, 2)


def match_peaks(pred_mask, gt_mask, tol=3.0):
    """One-to-one matching of predicted to true peak centroids within tol px.

    Maximum-cardinality bipartite matching (same tp count as Hungarian with a tol cutoff).
    Returns (pc, gc, pairs) with pairs an (M,2) int array of (pred_idx, gt_idx)."""
    pc, gc = centroids(pred_mask), centroids(gt_mask)
    if len(pc) == 0 or len(gc) == 0:
        return pc, gc, np.zeros((0, 2), int)
    sd = cKDTree(pc).sparse_distance_matrix(cKDTree(gc), tol, output_type="coo_matrix")
    # sparse_distance_matrix drops exact-zero distances; add them back explicitly
    rows, cols = list(sd.row), list(sd.col)
    for i, js in enumerate(cKDTree(gc).query_ball_point(pc, r=1e-9)):
        for j in js:
            rows.append(i)
            cols.append(j)
    if not rows:
        return pc, gc, np.zeros((0, 2), int)
    g = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(pc), len(gc)))
    g.data[:] = 1
    m = maximum_bipartite_matching(g, perm_type="column")  # m[pred] = gt or -1
    pairs = np.array([(i, j) for i, j in enumerate(m) if j >= 0], int).reshape(-1, 2)
    return pc, gc, pairs


def _ratio(a, b):
    return 1.0 if b == 0 else a / b


def image_metrics(pred_mask, gt_mask, tol=3.0, match=None):
    pc, gc, pairs = match if match is not None else match_peaks(pred_mask, gt_mask, tol)
    tp = len(pairs)
    fp, fn = len(pc) - tp, len(gc) - tp
    p, r = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    ptp = int(np.logical_and(pred_mask, gt_mask).sum())
    pfp = int(np.logical_and(pred_mask, ~gt_mask).sum())
    pfn = int(np.logical_and(~pred_mask, gt_mask).sum())
    return {"peak_tp": tp, "peak_fp": fp, "peak_fn": fn, "peak_precision": p, "peak_recall": r,
            "peak_f1": 0.0 if p + r == 0 else 2 * p * r / (p + r),
            "pixel_iou": _ratio(ptp, ptp + pfp + pfn)}


# ----------------------------------------------------------------------------- drawing
def _display(image, q=(1, 99.5)):
    """Robust display scaling: clip to percentiles of the unsaturated pixels, then sqrt."""
    a = np.nan_to_num(np.asarray(image, np.float32))
    s = a[::4, ::4] if a.size > 250000 else a
    s = s[s < 60000]
    lo, hi = np.percentile(s, q) if s.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1
    return np.sqrt(np.clip((a - lo) / (hi - lo), 0, 1))


def _as_mask(m, cls):
    m = np.asarray(m)
    return m.astype(bool) if m.dtype == bool else (m == cls)


def _classify(pc, gc, pairs):
    mp = set(pairs[:, 0].tolist()) if len(pairs) else set()
    mg = set(pairs[:, 1].tolist()) if len(pairs) else set()
    matched = gc[sorted(mg)] if mg else np.zeros((0, 2))
    missed = gc[[j for j in range(len(gc)) if j not in mg]] if len(gc) > len(mg) else np.zeros((0, 2))
    false = pc[[i for i in range(len(pc)) if i not in mp]] if len(pc) > len(mp) else np.zeros((0, 2))
    return matched, missed, false


def _circles(ax, pts, color, r, lw):
    from matplotlib.patches import Circle
    for y, x in pts:
        ax.add_patch(Circle((x, y), r, fill=False, ec=color, lw=lw))


def _pick_crops(matched, missed, false, n, half, shape):
    """Crop centres: misses and false peaks first (most informative), then matches.
    Skip centres already covered by an earlier crop."""
    order = [("missed", p) for p in missed] + [("false", p) for p in false] + [("matched", p) for p in matched]
    # interleave kinds so a figure shows a bit of each
    kinds = {"missed": [], "false": [], "matched": []}
    for k, p in order:
        kinds[k].append(p)
    seq, i = [], 0
    while any(i < len(v) for v in kinds.values()):
        for k in ("missed", "false", "matched"):
            if i < len(kinds[k]):
                seq.append((k, kinds[k][i]))
        i += 1
    out = []
    for k, (y, x) in seq:
        if any(abs(y - cy) < half * 0.7 and abs(x - cx) < half * 0.7 for _, (cy, cx) in out):
            continue
        cy = int(np.clip(round(y), half, max(half, shape[0] - half)))
        cx = int(np.clip(round(x), half, max(half, shape[1] - half)))
        out.append((k, (cy, cx)))
        if len(out) == n:
            break
    return out


def _draw_full(ax, disp, matched, missed, false, r):
    ax.imshow(disp, cmap="gray", interpolation="antialiased")
    _circles(ax, matched, GREEN, r, 1.3)
    _circles(ax, missed, RED, r, 1.3)
    _circles(ax, false, YELLOW, r, 1.3)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_crop(ax, raw, gt, pred, matched, missed, false, kind, c, half):
    y0, x0 = c[0] - half, c[1] - half
    sl = (slice(max(0, y0), y0 + 2 * half), slice(max(0, x0), x0 + 2 * half))
    y0, x0 = sl[0].start, sl[1].start
    ax.imshow(_display(raw[sl], (1, 99.9)), cmap="gray", interpolation="nearest")
    for m, col in ((gt[sl], GT_EDGE), (pred[sl], PRED_EDGE)):
        if m.any():
            ax.contour(m.astype(float), levels=[0.5], colors=col, linewidths=0.9)
    for pts, col in ((matched, GREEN), (missed, RED), (false, YELLOW)):
        if len(pts):
            sel = pts[(pts[:, 0] >= y0) & (pts[:, 0] < y0 + 2 * half) & (pts[:, 1] >= x0) & (pts[:, 1] < x0 + 2 * half)]
            _circles(ax, sel - [y0, x0], col, 4.5, 1.6)
    ax.set_title("%s @ (y=%d, x=%d)" % (kind, c[0], c[1]), fontsize=8,
                 color={"matched": "green", "missed": "red", "false": "#b08800"}[kind])
    ax.set_xticks([])
    ax.set_yticks([])


def _legend(fig, y=0.01):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", mfc="none", mec=c, ms=10, mew=2, label=l)
         for c, l in ((GREEN, "matched peak (TP)"), (RED, "missed peak (FN)"), (YELLOW, "false peak (FP)"))]
    h += [Line2D([], [], color=GT_EDGE, lw=2, label="ground-truth mask"),
          Line2D([], [], color=PRED_EDGE, lw=2, label="predicted mask")]
    fig.legend(handles=h, loc="lower center", ncol=5, fontsize=9, frameon=False, bbox_to_anchor=(0.5, y))


def _fmt(m, tol):
    return ("peak F1 %.3f  P %.3f  R %.3f  (TP %d  FP %d  FN %d, tol %g px)   pixel IoU %.3f"
            % (m["peak_f1"], m["peak_precision"], m["peak_recall"], m["peak_tp"], m["peak_fp"],
               m["peak_fn"], tol, m["pixel_iou"]))


def save_overlay(image, gt, pred, out_png, title="", extra="", tol=3.0, cls=1, n_crops=6,
                 crop_half=40, dpi=90):
    """Write one annotated overlay PNG. Returns the per-image metrics dict."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = np.asarray(image).squeeze()
    gt, pred = _as_mask(np.asarray(gt).squeeze(), cls), _as_mask(np.asarray(pred).squeeze(), cls)
    match = match_peaks(pred, gt, tol)
    met = image_metrics(pred, gt, tol, match)
    matched, missed, false = _classify(*match)
    disp = _display(image)
    r = max(8, image.shape[0] / 90)

    fig = plt.figure(figsize=(21, 10.5), dpi=dpi)
    gs = fig.add_gridspec(2, 5, width_ratios=[2.1, 2.1, 1, 1, 1], left=0.01, right=0.99,
                          top=0.90, bottom=0.07, wspace=0.05, hspace=0.15)
    ax = fig.add_subplot(gs[:, :2])
    _draw_full(ax, disp, matched, missed, false, r)
    ax.set_title("full image %dx%d: %d true peaks, %d predicted" % (image.shape[0], image.shape[1],
                 len(match[1]), len(match[0])), fontsize=10)
    crops = _pick_crops(matched, missed, false, n_crops, crop_half, image.shape)
    for k in range(n_crops):
        a = fig.add_subplot(gs[k // 3, 2 + k % 3])
        if k < len(crops):
            _draw_crop(a, image, gt, pred, matched, missed, false, crops[k][0], crops[k][1], crop_half)
        else:
            a.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.985)
    fig.text(0.5, 0.925, _fmt(met, tol) + (("   " + extra) if extra else ""), ha="center", fontsize=11)
    _legend(fig)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return met


def save_overlay_grid(samples, out_png, title="", extra="", tol=3.0, cls=1, crop_half=40, dpi=80):
    """Several images on one figure: per row, the full image plus its two most informative
    crops. samples: list of (image, gt, pred, image_id). Returns (per-image metrics, summed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(samples)
    fig = plt.figure(figsize=(16, 5.2 * n + 1.2), dpi=dpi)
    gs = fig.add_gridspec(n, 3, width_ratios=[1.6, 1, 1], left=0.01, right=0.99,
                          top=1 - 0.9 / (5.2 * n + 1.2), bottom=0.5 / (5.2 * n + 1.2), wspace=0.04, hspace=0.18)
    mets, tot = [], {"peak_tp": 0, "peak_fp": 0, "peak_fn": 0}
    for i, (image, gt, pred, iid) in enumerate(samples):
        image = np.asarray(image).squeeze()
        g, p = _as_mask(np.asarray(gt).squeeze(), cls), _as_mask(np.asarray(pred).squeeze(), cls)
        match = match_peaks(p, g, tol)
        met = image_metrics(p, g, tol, match)
        mets.append(met)
        for k in tot:
            tot[k] += met[k]
        matched, missed, false = _classify(*match)
        disp = _display(image)
        ax = fig.add_subplot(gs[i, 0])
        _draw_full(ax, disp, matched, missed, false, max(8, image.shape[0] / 90))
        ax.set_title("%s   peak F1 %.3f (TP %d FP %d FN %d)  IoU %.3f" % (
            iid, met["peak_f1"], met["peak_tp"], met["peak_fp"], met["peak_fn"], met["pixel_iou"]), fontsize=9)
        crops = _pick_crops(matched, missed, false, 2, crop_half, image.shape)
        for k in range(2):
            a = fig.add_subplot(gs[i, 1 + k])
            if k < len(crops):
                _draw_crop(a, image, g, p, matched, missed, false, crops[k][0], crops[k][1], crop_half)
            else:
                a.axis("off")
    tp, fp, fn = tot["peak_tp"], tot["peak_fp"], tot["peak_fn"]
    P, R = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    tot.update(peak_precision=P, peak_recall=R, peak_f1=0.0 if P + R == 0 else 2 * P * R / (P + R))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.text(0.5, 1 - 0.65 / (5.2 * n + 1.2), "these %d images: peak F1 %.3f  P %.3f  R %.3f  (TP %d FP %d FN %d, tol %g px)%s"
             % (n, tot["peak_f1"], P, R, tp, fp, fn, tol, ("   " + extra) if extra else ""), ha="center", fontsize=10)
    _legend(fig, y=0.0)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return mets, tot


# ----------------------------------------------------------------------------- CLI
def _load_npz(path, thr):
    z = np.load(path, allow_pickle=False)
    image = z["image"]
    gt = z["label"] if "label" in z else z["gt"]
    if "pred" in z:
        pred = z["pred"]
    else:
        pred = (z["prob"] > thr).astype(np.int64)
    iid = str(z["id"]) if "id" in z else os.path.basename(path)
    return image, gt, pred, iid


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", nargs="+", required=True, help="saved prediction file(s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--extra", default="", help="extra text after the metrics line (e.g. thr, ckpt)")
    ap.add_argument("--thr", type=float, default=0.5, help="threshold when the npz holds prob")
    ap.add_argument("--tol", type=float, default=3.0)
    ap.add_argument("--cls", type=int, default=1)
    a = ap.parse_args(argv)
    if len(a.npz) == 1:
        image, gt, pred, iid = _load_npz(a.npz[0], a.thr)
        met = save_overlay(image, gt, pred, a.out, title=a.title or iid, extra=a.extra, tol=a.tol, cls=a.cls)
    else:
        met = save_overlay_grid([_load_npz(p, a.thr) for p in a.npz], a.out, title=a.title,
                                extra=a.extra, tol=a.tol, cls=a.cls)[1]
    import json
    print(json.dumps({"out": a.out, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in met.items()}}))


if __name__ == "__main__":
    main()
