"""Peak object detection (statement `detection`): Hugging Face RT-DETRv2 on boxes derived
from the segmentation masks.

Labels: one box per 8-connected component of (label == peak_label), its bounding box padded
by `box_pad` px on every side (peaks are 1-10 px, so boxes are ~7-16 px).

Training: crops of `crop` px (most centred near a peak) upscaled by `scale` (tiny peaks need
it: RT-DETR's finest feature stride is 8) -> RTDetrV2ForObjectDetection (PekingU/rtdetr_v2_r18vd,
COCO-pretrained, 1-class head re-initialised) with its own loss (VFL + L1 + GIoU + denoising).
Inference: sliding window over the whole image (tile `crop`, overlap `overlap`), each tile
keeps detections whose centre lies in its core, then NMS. Metrics: COCO-style AP at IoU 0.5
(101-point interpolation, single class), and precision/recall at the score threshold picked
on val (best F1).

usage:
  python -m seg.detection train --config configs/detection.yaml --out RUNDIR
  python -m seg.detection eval  --config configs/detection.yaml --ckpt RUNDIR/best.pt --split test \
        --out metrics.json [--thr T] [--fig-dir DIR] [--limit N]
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
from scipy import ndimage

DATA = "/sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet10k/peaknet10k"
DEFAULTS = dict(
    data_root=DATA, split="results/split/result.json", peak_label=1,
    hf_model="PekingU/rtdetr_v2_r18vd", box_pad=3, crop=256, scale=2, overlap=32,
    crops_per_image=4, peak_centred=0.75, images_per_batch=4, workers=8,
    lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, minutes=40, max_iters=100000,
    val_every=400, val_images=120, score_floor=0.05, nms_iou=0.3, max_det_per_tile=100,
    seed=0, train_limit=None, amp=False, final_val_limit=None,
)


# ----------------------------------------------------------------------------- data
def load_cfg(path):
    cfg = dict(DEFAULTS)
    if path:
        import yaml
        cfg.update(yaml.safe_load(open(path)) or {})
    return cfg


def split_stores(cfg, which):
    return json.load(open(cfg["split"]))["stores"][which]


def index_images(stores, root):
    import zarr
    out = []
    for s in stores:
        z = zarr.open(os.path.join(root, s), "r")
        out += [(s, i) for i in range(z["images"].shape[0])]
    return out


_ZCACHE = {}


def read_image(root, store, i):
    import zarr
    key = (root, store)
    if key not in _ZCACHE:
        if len(_ZCACHE) > 64:
            _ZCACHE.clear()
        _ZCACHE[key] = zarr.open(os.path.join(root, store), "r")
    z = _ZCACHE[key]
    return np.asarray(z["images"][i], np.float32), np.asarray(z["labels"][i])


def mask_boxes(mask, pad):
    """(N,4) float boxes x0,y0,x1,y1 (pixel edges) of 8-connected components, padded."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    if n == 0:
        return np.zeros((0, 4), np.float32)
    sl = ndimage.find_objects(lab)
    b = np.array([[s[1].start - pad, s[0].start - pad, s[1].stop + pad, s[0].stop + pad] for s in sl], np.float32)
    H, W = mask.shape
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, W)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
    return b


def normalize(img):
    """Detector input in [0,1]: clip to [p1, p99.9] of unsaturated pixels, then sqrt."""
    s = img[::4, ::4]
    s = s[s < 60000]
    lo, hi = np.percentile(s, [1, 99.9]) if s.size else (0.0, 1.0)
    hi = max(hi, lo + 1e-3)
    return np.sqrt(np.clip((img - lo) / (hi - lo), 0, 1)).astype(np.float32)


class CropDataset:
    """Map-style: item = list of (crop (C,C) float32 in [0,1], boxes (N,4) x0y0x1y1 in crop px)."""

    def __init__(self, items, cfg, train=True):
        self.items, self.cfg, self.train = items, cfg, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        c = self.cfg
        store, i = self.items[k]
        img, lab = read_image(c["data_root"], store, i)
        x = normalize(img)
        boxes = mask_boxes(lab == c["peak_label"], c["box_pad"])
        H, W = x.shape
        C = c["crop"]
        out = []
        for _ in range(c["crops_per_image"]):
            if len(boxes) and random.random() < c["peak_centred"]:
                b = boxes[random.randrange(len(boxes))]
                cy, cx = (b[1] + b[3]) / 2, (b[0] + b[2]) / 2
                y0 = int(np.clip(cy - random.uniform(0.1, 0.9) * C, 0, H - C))
                x0 = int(np.clip(cx - random.uniform(0.1, 0.9) * C, 0, W - C))
            else:
                y0, x0 = random.randint(0, H - C), random.randint(0, W - C)
            crop = x[y0:y0 + C, x0:x0 + C]
            bb = boxes - np.array([x0, y0, x0, y0], np.float32)
            ctr = np.stack([(bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2], 1) if len(bb) else np.zeros((0, 2))
            keep = (ctr[:, 0] >= 0) & (ctr[:, 0] < C) & (ctr[:, 1] >= 0) & (ctr[:, 1] < C) if len(bb) else np.zeros(0, bool)
            bb = bb[keep].clip(0, C)
            if self.train:  # flips
                if random.random() < 0.5:
                    crop = crop[:, ::-1]
                    bb = bb[:, [2, 1, 0, 3]] * [-1, 1, -1, 1] + [C, 0, C, 0] if len(bb) else bb
                if random.random() < 0.5:
                    crop = crop[::-1, :]
                    bb = bb[:, [0, 3, 2, 1]] * [1, -1, 1, -1] + [0, C, 0, C] if len(bb) else bb
            out.append((np.ascontiguousarray(crop), np.asarray(bb, np.float32)))
        return out


def collate(batch):
    return [c for item in batch for c in item]


def to_model_inputs(crops, cfg, device):
    """List of (crop, boxes) -> pixel_values (B,3,S,S) and RT-DETR labels (normalised cxcywh)."""
    import torch
    import torch.nn.functional as F
    C, s = cfg["crop"], cfg["scale"]
    x = torch.from_numpy(np.stack([c for c, _ in crops]))[:, None].to(device)
    x = F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
    labels = []
    for _, b in crops:
        b = torch.from_numpy(b).float()
        if len(b):
            cxcywh = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                                  b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1) / C
        else:
            cxcywh = torch.zeros((0, 4))
        labels.append({"class_labels": torch.zeros(len(b), dtype=torch.long, device=device),
                       "boxes": cxcywh.to(device)})
    return x, labels


# ----------------------------------------------------------------------------- model
def build_model(cfg):
    from transformers import RTDetrV2ForObjectDetection
    return RTDetrV2ForObjectDetection.from_pretrained(cfg["hf_model"], num_labels=1, ignore_mismatched_sizes=True)


def tile_starts(n, t, stride):
    if n <= t:
        return [0]
    k = int(math.ceil((n - t) / stride)) + 1
    return sorted(set(int(round(v)) for v in np.linspace(0, n - t, k)))


def detect_image(model, img, cfg, device, batch=96):
    """Full-image detection by tiling. Returns boxes (N,4) x0y0x1y1 in image px, scores (N,)."""
    import torch
    import torch.nn.functional as F
    from torchvision.ops import nms
    x = normalize(img)
    H, W = x.shape
    T, ov, s = cfg["crop"], cfg["overlap"], cfg["scale"]
    ys, xs = tile_starts(H, T, T - ov), tile_starts(W, T, T - ov)
    pos = [(y, xx) for y in ys for xx in xs]
    xt = torch.from_numpy(x).to(device)
    allb, alls = [], []
    for k in range(0, len(pos), batch):
        pp = pos[k:k + batch]
        t = torch.stack([xt[y:y + T, xx:xx + T] for y, xx in pp])[:, None]
        t = F.interpolate(t, scale_factor=s, mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.get("amp", False) and device != "cpu"):
            o = model(pixel_values=t)
        sc = o.logits.float().sigmoid()[..., 0]  # (B,Q)
        bx = o.pred_boxes.float()  # (B,Q,4) cxcywh normalised
        for j, (y, xx) in enumerate(pp):
            m = sc[j] > cfg["score_floor"]
            if not m.any():
                continue
            q, b = sc[j][m], bx[j][m] * T
            top = q.argsort(descending=True)[:cfg["max_det_per_tile"]]
            q, b = q[top], b[top]
            cx, cy = b[:, 0] + xx, b[:, 1] + y
            # core region of this tile (avoid duplicates from overlaps)
            lo_x = xx + (ov / 2 if xx > 0 else -1e9)
            hi_x = xx + T - (ov / 2 if xx + T < W else -1e9)
            lo_y = y + (ov / 2 if y > 0 else -1e9)
            hi_y = y + T - (ov / 2 if y + T < H else -1e9)
            keep = (cx >= lo_x) & (cx < hi_x) & (cy >= lo_y) & (cy < hi_y)
            if not keep.any():
                continue
            b, q, cx, cy = b[keep], q[keep], cx[keep], cy[keep]
            allb.append(torch.stack([cx - b[:, 2] / 2, cy - b[:, 3] / 2, cx + b[:, 2] / 2, cy + b[:, 3] / 2], 1))
            alls.append(q)
    if not allb:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    b, q = torch.cat(allb), torch.cat(alls)
    k = nms(b, q, cfg["nms_iou"])
    return b[k].cpu().numpy(), q[k].cpu().numpy()


# ----------------------------------------------------------------------------- metrics
def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def match_dets(pb, ps, gb, iou_thr=0.5):
    """COCO-style greedy matching by descending score. Returns tp flags (N,) aligned with
    the score-sorted order, the sorted scores, and the matched-GT flags."""
    order = np.argsort(-ps, kind="stable")
    pb, ps = pb[order], ps[order]
    iou = box_iou(pb, gb)
    used = np.zeros(len(gb), bool)
    tp = np.zeros(len(pb), bool)
    for i in range(len(pb)):
        if len(gb) == 0:
            break
        cand = np.where(~used, iou[i], -1)
        j = int(cand.argmax())
        if cand[j] >= iou_thr:
            used[j] = True
            tp[i] = True
    return tp, ps, used, order


class DetAccumulator:
    def __init__(self, iou_thr=0.5):
        self.iou_thr, self.scores, self.tps, self.n_gt, self.n_img = iou_thr, [], [], 0, 0

    def add(self, pb, ps, gb):
        tp, s, _, _ = match_dets(pb, ps, gb, self.iou_thr)
        self.scores.append(s)
        self.tps.append(tp)
        self.n_gt += len(gb)
        self.n_img += 1

    def _sorted(self):
        s = np.concatenate(self.scores) if self.scores else np.zeros(0)
        t = np.concatenate(self.tps) if self.tps else np.zeros(0, bool)
        o = np.argsort(-s, kind="stable")
        return s[o], t[o]

    def ap(self):
        s, t = self._sorted()
        if self.n_gt == 0:
            return 1.0 if len(s) == 0 else 0.0
        ctp, cfp = np.cumsum(t), np.cumsum(~t)
        rec = ctp / self.n_gt
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        prec = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec  # envelope
        rs = np.linspace(0, 1, 101)
        idx = np.searchsorted(rec, rs, side="left")
        return float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx]))

    def at(self, thr):
        s, t = self._sorted()
        k = s >= thr
        tp, fp = int(t[k].sum()), int((~t[k]).sum())
        fn = self.n_gt - tp
        p = 1.0 if tp + fp == 0 else tp / (tp + fp)
        r = 1.0 if self.n_gt == 0 else tp / self.n_gt
        return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
                "f1": 0.0 if p + r == 0 else 2 * p * r / (p + r)}

    def best_threshold(self):
        best = (0.0, 0.5)
        for thr in np.round(np.arange(0.05, 0.96, 0.025), 3):
            f = self.at(thr)["f1"]
            if f > best[0]:
                best = (f, float(thr))
        return best[1]


# ----------------------------------------------------------------------------- figure
def save_det_figure(img, gb, pb, ps, thr, out_png, title, iou_thr=0.5, n_crops=6, half=40):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from seg.viz import _display
    k = ps >= thr
    pb, ps = pb[k], ps[k]
    tp, ps_s, used, order = match_dets(pb, ps, gb, iou_thr)
    pb_s = pb[order]
    groups = [("matched", gb[used], "#22dd22"), ("missed", gb[~used], "#ff2222"), ("false", pb_s[~tp], "#ffdd00")]
    fig = plt.figure(figsize=(21, 10.5), dpi=90)
    gs = fig.add_gridspec(2, 5, width_ratios=[2.1, 2.1, 1, 1, 1], left=0.01, right=0.99, top=0.90, bottom=0.07,
                          wspace=0.05, hspace=0.15)
    ax = fig.add_subplot(gs[:, :2])
    ax.imshow(_display(img), cmap="gray")
    r = max(8, img.shape[0] / 90)
    for _, bs, col in groups:
        for b in bs:
            ax.add_patch(plt.Circle(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), r, fill=False, ec=col, lw=1.3))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("full image %dx%d: %d true boxes, %d predicted (score >= %.3f)" % (img.shape[0], img.shape[1], len(gb), len(pb), thr), fontsize=10)
    cents = [(kind, b) for kind, bs, _ in (groups[1], groups[2], groups[0]) for b in bs[:2]]
    cents += [(kind, b) for kind, bs, _ in (groups[0], groups[1], groups[2]) for b in bs[2:]]
    for k2 in range(n_crops):
        a = fig.add_subplot(gs[k2 // 3, 2 + k2 % 3])
        if k2 >= len(cents):
            a.axis("off")
            continue
        kind, b = cents[k2]
        cy, cx = int((b[1] + b[3]) / 2), int((b[0] + b[2]) / 2)
        y0, x0 = max(0, cy - half), max(0, cx - half)
        a.imshow(_display(img[y0:y0 + 2 * half, x0:x0 + 2 * half], (1, 99.9)), cmap="gray", interpolation="nearest")
        for bb in gb:
            a.add_patch(Rectangle((bb[0] - x0 - .5, bb[1] - y0 - .5), bb[2] - bb[0], bb[3] - bb[1], fill=False, ec="cyan", lw=1.2))
        for bb, ok in zip(pb_s, tp):
            a.add_patch(Rectangle((bb[0] - x0 - .5, bb[1] - y0 - .5), bb[2] - bb[0], bb[3] - bb[1], fill=False,
                                  ec="#22dd22" if ok else "#ffdd00", lw=1.2, ls="--"))
        a.set_xlim(-.5, min(2 * half, img.shape[1] - x0) - .5); a.set_ylim(min(2 * half, img.shape[0] - y0) - .5, -.5)
        a.set_title("%s @ (y=%d, x=%d)" % (kind, cy, cx), fontsize=8,
                    color={"matched": "green", "missed": "red", "false": "#b08800"}[kind])
        a.set_xticks([]); a.set_yticks([])
    ntp = int(tp.sum())
    p = 1.0 if len(pb) == 0 else ntp / len(pb)
    rr = 1.0 if len(gb) == 0 else ntp / len(gb)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.985)
    fig.text(0.5, 0.925, "this image: box precision %.3f  recall %.3f  (TP %d  FP %d  FN %d at IoU>=%.1f)   crops: cyan = GT box, dashed green = TP box, dashed yellow = FP box"
             % (p, rr, ntp, len(pb) - ntp, len(gb) - ntp, iou_thr), ha="center", fontsize=10.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", mfc="none", mec=c, ms=10, mew=2, label=l) for c, l in
         (("#22dd22", "matched peak box (TP)"), ("#ff2222", "missed peak box (FN)"), ("#ffdd00", "false box (FP)"))]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.01))
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=90)
    plt.close(fig)


# ----------------------------------------------------------------------------- train / eval
def evaluate_items(model, items, cfg, device, log=None, fig_dir=None, fig_every=0, thr_for_fig=0.5, title=""):
    acc = DetAccumulator(0.5)
    t0 = time.time()
    for n, (store, i) in enumerate(items):
        img, lab = read_image(cfg["data_root"], store, i)
        gb = mask_boxes(lab == cfg["peak_label"], cfg["box_pad"])
        pb, ps = detect_image(model, img, cfg, device)
        acc.add(pb, ps, gb)
        if fig_dir and fig_every and n % fig_every == 0:
            save_det_figure(img, gb, pb, ps, thr_for_fig, os.path.join(fig_dir, "det_%s_%03d.png" % (store.replace(".zarr", ""), i)),
                            "%s  %s[%d]" % (title, store, i))
        if log and (n + 1) % 200 == 0:
            log("  eval %d/%d images, %.1f img/s" % (n + 1, len(items), (n + 1) / (time.time() - t0)))
    return acc


def train(a):
    import torch
    cfg = load_cfg(a.config)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "train.log"), "a")

    def log(msg):
        line = "%s %s" % (time.strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log("cfg %s" % json.dumps(cfg))
    log("SLURM_JOB_ID=%s SLURM_STEP_ID=%s host=%s device=%s %s" % (os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_STEP_ID"),
        os.uname().nodename, device, torch.cuda.get_device_name(0) if device == "cuda" else ""))
    tr = index_images(split_stores(cfg, "train"), cfg["data_root"])
    va = index_images(split_stores(cfg, "val"), cfg["data_root"])
    if cfg["train_limit"]:
        random.shuffle(tr); tr = tr[:cfg["train_limit"]]
    rv = random.Random(1); va = sorted(rv.sample(va, min(cfg["val_images"], len(va))))
    log("train images %d, val subset %d" % (len(tr), len(va)))
    model = build_model(cfg).to(device)
    bb = [p for n, p in model.named_parameters() if "backbone" in n]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n]
    opt = torch.optim.AdamW([{"params": bb, "lr": cfg["lr_backbone"]}, {"params": rest, "lr": cfg["lr"]}],
                            weight_decay=cfg["weight_decay"])
    dl = torch.utils.data.DataLoader(CropDataset(tr, cfg, True), batch_size=cfg["images_per_batch"], shuffle=True,
                                     num_workers=cfg["workers"], collate_fn=collate, drop_last=True, persistent_workers=True)
    t_start, it, best, ep = time.time(), 0, -1.0, 0
    budget = cfg["minutes"] * 60
    total_iters = min(cfg["max_iters"], 10 ** 9)
    done = False
    while not done:
        ep += 1
        for crops in dl:
            model.train()
            x, labels = to_model_inputs(crops, cfg, device)
            frac = min(1.0, (time.time() - t_start) / budget)
            for g, base in zip(opt.param_groups, (cfg["lr_backbone"], cfg["lr"])):
                g["lr"] = base * (min(1.0, (it + 1) / 200)) * (0.5 * (1 + math.cos(math.pi * frac)) * 0.95 + 0.05)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["amp"] and device == "cuda"):
                out = model(pixel_values=x, labels=labels)
            loss = out.loss
            opt.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                opt.step()
            it += 1
            if it % 50 == 0:
                ld = {k: round(float(v), 3) for k, v in out.loss_dict.items() if "aux" not in k and "dn" not in k}
                log("ep %d it %d loss %.3f %s lr %.2e elapsed %.1f min" % (ep, it, float(loss), ld, opt.param_groups[1]["lr"], (time.time() - t_start) / 60))
            last = (time.time() - t_start) >= budget or it >= total_iters
            if it % cfg["val_every"] == 0 or last:
                model.eval()
                acc = evaluate_items(model, va, cfg, device)
                ap, thr = acc.ap(), acc.best_threshold()
                at = acc.at(thr)
                log("VAL it %d AP50 %.4f best_thr %.3f P %.4f R %.4f F1 %.4f (n_img %d n_gt %d)" % (it, ap, thr, at["precision"], at["recall"], at["f1"], acc.n_img, acc.n_gt))
                if ap > best:
                    best = ap
                    torch.save({"model": model.state_dict(), "cfg": cfg, "thr": thr, "it": it, "val_ap50": ap}, os.path.join(a.out, "best.pt"))
                    log("  saved best.pt (val AP50 %.4f)" % ap)
            if last:
                done = True
                break
    # full-val threshold pick for the best checkpoint
    ck = torch.load(os.path.join(a.out, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()
    vall = index_images(split_stores(cfg, "val"), cfg["data_root"])
    if cfg["final_val_limit"]:
        vall = vall[:cfg["final_val_limit"]]
    acc = evaluate_items(model, vall, cfg, device, log=log)
    ck["thr"] = acc.best_threshold()
    ck["val_full"] = {"ap50": acc.ap(), **acc.at(ck["thr"]), "n_images": acc.n_img}
    torch.save(ck, os.path.join(a.out, "best.pt"))
    log("FULL VAL best ckpt it %d: %s thr %.3f" % (ck["it"], json.dumps(ck["val_full"]), ck["thr"]))
    log("done in %.1f min" % ((time.time() - t_start) / 60))


def evaluate(a):
    import torch
    cfg = load_cfg(a.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    cfg.update({k: v for k, v in ck.get("cfg", {}).items() if k in ("box_pad", "crop", "scale", "overlap", "hf_model")})
    if a.split_file:
        cfg["split"] = a.split_file
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    thr = a.thr if a.thr is not None else ck.get("thr", 0.5)
    items = index_images(split_stores(cfg, a.split), cfg["data_root"])
    if a.limit:
        items = items[:a.limit]
    t0 = time.time()
    print("SLURM_JOB_ID=%s SLURM_STEP_ID=%s host=%s ckpt=%s thr=%.3f n=%d" % (os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_STEP_ID"),
          os.uname().nodename, a.ckpt, thr, len(items)), flush=True)
    acc = evaluate_items(model, items, cfg, device, log=lambda m: print(m, flush=True), fig_dir=a.fig_dir,
                         fig_every=a.fig_every, thr_for_fig=thr, title="RT-DETRv2-r18 peak detector, %s split" % a.split)
    res = {"split": a.split, "n_images": acc.n_img, "n_gt_boxes": acc.n_gt, "map50": acc.ap(), "score_thr": thr,
           **acc.at(thr), "iou_thr": 0.5, "ckpt": a.ckpt, "minutes": (time.time() - t0) / 60}
    print(json.dumps(res), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--config")
    t.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--config")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--split", default="test")
    e.add_argument("--split-file")
    e.add_argument("--out")
    e.add_argument("--thr", type=float)
    e.add_argument("--limit", type=int)
    e.add_argument("--fig-dir")
    e.add_argument("--fig-every", type=int, default=0)
    a = ap.parse_args()
    if os.environ.get("PK_NO_CUDNN") == "1":
        import torch
        torch.backends.cudnn.enabled = False
    train(a) if a.cmd == "train" else evaluate(a)


if __name__ == "__main__":
    main()
