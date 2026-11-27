"""Smoke test for seg/models/lucid_segformer.py on a few stores (not a reported result).

python scripts/lucid_segformer_smoke.py --stores A.zarr B.zarr --iters 300 --bs 8 --crop 512
Peak-centred 512 crops, BCE(pos_weight)+soft dice, AMP bf16; prints loss, GPU memory, it/s, and
peak F1 (seg.evaluate.peak_metrics, tol 3) on 4 held-out full images at a few thresholds.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import zarr

from seg.evaluate import peak_metrics
from seg.models.lucid_segformer import build_model

DATA = "/sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet10k/peaknet10k"


def crop(img, lab, c, rng):
    H, W = img.shape
    ys, xs = np.nonzero(lab == 1)
    if len(ys) and rng.random() < 0.7:
        k = rng.integers(len(ys))
        y0 = int(np.clip(ys[k] - rng.integers(c), 0, H - c))
        x0 = int(np.clip(xs[k] - rng.integers(c), 0, W - c))
    else:
        y0, x0 = rng.integers(H - c + 1), rng.integers(W - c + 1)
    return img[y0:y0 + c, x0:x0 + c], lab[y0:y0 + c, x0:x0 + c]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", nargs="+", required=True)
    ap.add_argument("--config", default="configs/lucid_segformer.yaml")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--pos-weight", type=float, default=20.0)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    imgs, labs = [], []
    for s in a.stores:
        z = zarr.open(os.path.join(DATA, s), "r")
        n = z["images"].shape[0]
        for i in range(0, n, 4):
            imgs.append(z["images"][i])
            labs.append(z["labels"][i])
    hold = list(range(len(imgs) - 4, len(imgs)))
    tr = list(range(len(imgs) - 4))
    print("images", len(imgs), "train", len(tr), flush=True)
    dev = torch.device("cuda")
    m = build_model(yaml.safe_load(open(a.config))).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.iters, pct_start=0.1)
    pw = torch.tensor(a.pos_weight, device=dev)
    t0 = time.time()
    for it in range(a.iters):
        bx, by = [], []
        for _ in range(a.bs):
            k = tr[rng.integers(len(tr))]
            x, y = crop(imgs[k], labs[k], a.crop, rng)
            bx.append(x)
            by.append(y == 1)
        x = torch.from_numpy(np.stack(bx)).float()[:, None].to(dev)
        y = torch.from_numpy(np.stack(by)).float()[:, None].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m(x)
        lo = lo.float()
        bce = F.binary_cross_entropy_with_logits(lo, y, pos_weight=pw)
        p = torch.sigmoid(lo)
        dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
        loss = bce + dice
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 25 == 0 or it == a.iters - 1:
            print("it %d loss %.4f bce %.4f dice %.4f mem %.1fGB %.2f it/s" % (
                it, loss.item(), bce.item(), dice.item(), torch.cuda.max_memory_allocated() / 1e9,
                (it + 1) / (time.time() - t0)), flush=True)
    m.eval()
    probs = []
    t = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for k in hold:
            x = torch.from_numpy(imgs[k]).float()[None, None].to(dev)
            probs.append(torch.sigmoid(m(x).float())[0, 0].cpu().numpy())
    print("full-image infer %.2fs/img, mem %.1fGB" % ((time.time() - t) / len(hold),
                                                       torch.cuda.max_memory_allocated() / 1e9))
    for thr in (0.3, 0.5, 0.7, 0.9):
        pred = np.stack([p > thr for p in probs]).astype(np.int64)
        tgt = np.stack([(labs[k] == 1) for k in hold]).astype(np.int64)
        print("thr", thr, peak_metrics(pred, tgt), flush=True)


if __name__ == "__main__":
    main()
