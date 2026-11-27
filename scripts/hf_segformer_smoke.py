"""Smoke trainer for hf_segformer (until the shared seg/train.py lands).

Loads a few stores into RAM, trains on random crops (half centred near a peak), and reports
val peak F1 (seg.evaluate.peak_metrics, tol 3) on full images of held-out stores.
usage: python scripts/hf_segformer_smoke.py --train s1,s2 --val s3 --iters 600 --out DIR
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import zarr

from seg.evaluate import peak_metrics, pixel_metrics
from seg.models.hf_segformer import build_model

DATA = "/sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet10k/peaknet10k"


def load(stores):
    ims, lbs = [], []
    for s in stores:
        z = zarr.open(os.path.join(DATA, s), "r")
        ims += list(z["images"][:].astype(np.float32))
        lbs += list((z["labels"][:] == 1).astype(np.float32))
    return ims, lbs


def crop_batch(ims, lbs, bs, c, rng):
    X, Y = [], []
    for _ in range(bs):
        i = rng.integers(len(ims))
        im, lb = ims[i], lbs[i]
        H, W = im.shape
        pos = np.argwhere(lb > 0) if rng.random() < 0.5 else []
        if len(pos):
            cy, cx = pos[rng.integers(len(pos))]
            y0 = int(np.clip(cy - rng.integers(c), 0, H - c))
            x0 = int(np.clip(cx - rng.integers(c), 0, W - c))
        else:
            y0, x0 = rng.integers(H - c + 1), rng.integers(W - c + 1)
        X.append(im[y0:y0 + c, x0:x0 + c])
        Y.append(lb[y0:y0 + c, x0:x0 + c])
    return torch.from_numpy(np.stack(X))[:, None], torch.from_numpy(np.stack(Y))[:, None]


def loss_fn(logits, y, pos_weight, dice_w):
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor(pos_weight, device=y.device))
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    return bce + dice_w * dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--pretrained", default="nvidia/mit-b2")
    ap.add_argument("--first_stride", type=int, default=4)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--pos_weight", type=float, default=10.0)
    ap.add_argument("--dice_w", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    t0 = time.time()
    tr = load(a.train.split(","))
    va = load(a.val.split(","))
    print("loaded %d train, %d val images in %.0fs" % (len(tr[0]), len(va[0]), time.time() - t0), flush=True)
    dev = "cuda"
    model = build_model({"model": {"pretrained": a.pretrained, "first_stride": a.first_stride}}).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.05)
    model.train()
    t0 = time.time()
    for it in range(1, a.iters + 1):
        x, y = crop_batch(*tr, a.bs, a.crop, rng)
        x, y = x.to(dev), y.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = loss_fn(logits.float(), y, a.pos_weight, a.dice_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % 50 == 0:
            print("it %d loss %.4f %.2fs/it mem %.1fGB" % (it, loss.item(), (time.time() - t0) / it,
                                                          torch.cuda.max_memory_allocated() / 1e9), flush=True)
    model.eval()
    probs = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for im in va[0]:
            probs.append(torch.sigmoid(model(torch.from_numpy(im)[None, None].to(dev)).float())[0, 0].cpu().numpy())
    tgt = np.stack(va[1]).astype(np.int64)
    res = {}
    for thr in (0.3, 0.5, 0.7, 0.9):
        pred = (np.stack(probs) > thr).astype(np.int64)
        pk, px = peak_metrics(pred, tgt), pixel_metrics(pred, tgt)
        res[thr] = {"peak_f1": pk["f1"], "peak_p": pk["precision"], "peak_r": pk["recall"], "pix_iou": px["iou"]}
        print("val thr %.1f %s" % (thr, json.dumps(res[thr])), flush=True)
    json.dump(res, open(os.path.join(a.out, "smoke_val.json"), "w"), indent=1)
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
