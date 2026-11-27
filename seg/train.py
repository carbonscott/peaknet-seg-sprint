"""Generic trainer for binary peak segmentation (shared by every model statement).

usage (single GPU):
    python -m seg.train --model hf_segformer --config configs/hf_segformer.yaml --out $RUN
usage (DDP):
    torchrun --standalone --nproc_per_node=2 -m seg.train --model ... --config ... --out $RUN
options:
    --time-min 60          wall-clock training budget (LR schedule anneals to it; val+save at the end)
    --set train.lr=1e-4    override any config key (dotted path, yaml-parsed value); repeatable
    --init-ckpt best.pt    start from another run's weights (fine-tuning)
    --smoke                tiny run: 3 train stores, 8 val images, 20 steps

Config (yaml): the whole dict is passed to seg.models.<model>.build_model(cfg). Keys used here:
    train: {crop: 512, batch_size: 16, crops_per_image: 4, pos_frac: 0.7, lr: 3e-4, weight_decay: 0.01,
            max_steps: 20000, warmup: 300, val_every: 1000, val_images: 200, num_workers: 8,
            amp: true, grad_clip: 1.0, gain_jitter: 0.2, rot90: true, val_tile: null,
            thresholds: [0.2, ..., 0.9], seed: 0}
    loss:  see seg/losses.py
Outputs in --out: train.log lines on stdout, metrics.jsonl, best.pt {state_dict, cfg, thr, step, val, model},
last.pt. best = highest val peak F1 (tol 3 px) over the threshold sweep; thr saved with it.
"""
import argparse
import copy
import importlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml

from seg.infer import predict_prob
from seg.losses import build_loss

DATA_ROOT = "/sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet10k/peaknet10k"
DEFAULT_THR = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ----------------------------------------------------------------------------- config
def set_dotted(cfg, key, val):
    d = cfg
    ks = key.split(".")
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    d[ks[-1]] = yaml.safe_load(val)


def load_cfg(path, sets):
    cfg = yaml.safe_load(open(path)) if path else {}
    cfg = cfg or {}
    for s in sets or []:
        k, v = s.split("=", 1)
        set_dotted(cfg, k, v)
    return cfg


# ----------------------------------------------------------------------------- data
def get_split(path="results/split/result.json"):
    try:
        from seg.data import load_split
        return load_split(path)
    except Exception:
        return json.load(open(path))["stores"]


class _ZarrImages(torch.utils.data.Dataset):
    """Fallback (only if seg.data is unavailable): one item per image, dict(image, label, meta)."""

    def __init__(self, stores, root=DATA_ROOT):
        import zarr
        self.root, self.stores, self.index = root, list(stores), []
        for si, s in enumerate(self.stores):
            n = zarr.open(os.path.join(root, s), "r")["images"].shape[0]
            self.index += [(si, i) for i in range(n)]
        self._z = {}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        import zarr
        si, i = self.index[k]
        if si not in self._z:
            self._z[si] = zarr.open(os.path.join(self.root, self.stores[si]), "r")
        z = self._z[si]
        return dict(image=z["images"][i].astype(np.float32), label=z["labels"][i].astype(np.int64),
                    meta=dict(store=self.stores[si], index=i))


def make_base_dataset(stores):
    try:
        from seg.data import PeakDataset
        return PeakDataset(stores)
    except ImportError:
        print("[train] seg.data not importable; using fallback zarr reader", flush=True)
        return _ZarrImages(stores)


def _hw(a):
    a = np.asarray(a)
    return a[0] if a.ndim == 3 else a


class CropDataset(torch.utils.data.Dataset):
    """Wraps an image dataset; each item = `k` random crops of one image (peak-biased)."""

    def __init__(self, base, crop=512, k=4, pos_frac=0.7, peak_label=1, rot90=True, gain_jitter=0.2):
        self.base, self.crop, self.k, self.pos_frac = base, crop, k, pos_frac
        self.peak_label, self.rot90, self.gain_jitter = peak_label, rot90, gain_jitter

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        it = self.base[i]
        img = _hw(it["image"]).astype(np.float32)
        lab = (_hw(it["label"]) == self.peak_label).astype(np.float32)
        c = self.crop
        H, W = img.shape
        if H < c or W < c:
            ph, pw = max(c - H, 0), max(c - W, 0)
            img = np.pad(img, ((0, ph), (0, pw)))
            lab = np.pad(lab, ((0, ph), (0, pw)))
            H, W = img.shape
        pos = np.argwhere(lab > 0) if self.pos_frac > 0 else None
        xs, ys = [], []
        for _ in range(self.k):
            if pos is not None and len(pos) and random.random() < self.pos_frac:
                py, px = pos[random.randrange(len(pos))]
                y0 = int(np.clip(py - random.randrange(c), 0, H - c))
                x0 = int(np.clip(px - random.randrange(c), 0, W - c))
            else:
                y0, x0 = random.randint(0, H - c), random.randint(0, W - c)
            a = img[y0:y0 + c, x0:x0 + c]
            b = lab[y0:y0 + c, x0:x0 + c]
            if random.random() < 0.5:
                a, b = a[:, ::-1], b[:, ::-1]
            if random.random() < 0.5:
                a, b = a[::-1], b[::-1]
            if self.rot90:
                r = random.randrange(4)
                a, b = np.rot90(a, r), np.rot90(b, r)
            if self.gain_jitter:
                g = math.exp(random.uniform(-self.gain_jitter, self.gain_jitter))
                a = np.where(a >= 60000, a, a * g)
            xs.append(np.ascontiguousarray(a))
            ys.append(np.ascontiguousarray(b))
        return torch.from_numpy(np.stack(xs))[:, None], torch.from_numpy(np.stack(ys))


def crop_collate(batch):
    return torch.cat([b[0] for b in batch]), torch.cat([b[1] for b in batch])


# ----------------------------------------------------------------------------- metrics
@torch.no_grad()
def validate(model, val_ds, idxs, thrs, device, tile=None, peak_label=1):
    """Team metric (seg.evaluate.Accumulator) over val images idxs at full resolution.
    Returns counts array (len(thrs), 6): pixel tp,fp,fn ; peak tp,fp,fn."""
    from seg.evaluate import Accumulator
    model.eval()
    acc = Accumulator(thrs)
    for i in idxs:
        it = val_ds[i]
        img = torch.from_numpy(_hw(it["image"]).astype(np.float32))[None, None].to(device)
        prob = predict_prob(model, img, tile=tile)[0, 0].float().cpu().numpy()
        acc.add_prob(prob, _hw(it["label"]) == peak_label)
    model.train()
    return np.stack([acc.c[float(t)] for t in thrs]).astype(np.float64)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# ----------------------------------------------------------------------------- main
def unwrap(m):
    return m.module if hasattr(m, "module") else m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="results/split/result.json")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--time-min", type=float, default=None)
    ap.add_argument("--init-ckpt", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_cfg(a.config, a.set)
    tc = cfg.setdefault("train", {})
    g = lambda k, d: tc.get(k, d)  # noqa: E731

    ddp = "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = 0, 1, 0
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed = g("seed", 0) + rank
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    is0 = rank == 0
    os.makedirs(a.out, exist_ok=True)

    def log(**kw):
        if is0:
            kw["time"] = round(time.time() - t0, 1)
            print(json.dumps(kw), flush=True)
            with open(os.path.join(a.out, "metrics.jsonl"), "a") as f:
                f.write(json.dumps(kw) + "\n")

    t0 = time.time()
    split = get_split(a.split)
    tr_stores, va_stores = list(split["train"]), list(split["val"])
    if a.smoke:
        tr_stores, va_stores = tr_stores[:3], va_stores[:1]
        tc.update(max_steps=min(g("max_steps", 20), 20), val_every=10, val_images=8, warmup=5)
    time_budget = (a.time_min or g("time_min", 1e9)) * 60.0
    log(event="start", model=a.model, config=a.config, world=world, device=str(device),
        n_train_stores=len(tr_stores), n_val_stores=len(va_stores), cfg=cfg)

    # model
    mod = importlib.import_module("seg.models.%s" % a.model)
    model = mod.build_model(cfg)
    if a.init_ckpt:
        sd = torch.load(a.init_ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        miss = model.load_state_dict(sd, strict=False)
        log(event="init_ckpt", path=a.init_ckpt, missing=len(miss.missing_keys), unexpected=len(miss.unexpected_keys))
    model.to(device)
    nparam = sum(p.numel() for p in model.parameters())
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local] if device.type == "cuda" else None,
                                                          find_unused_parameters=bool(g("find_unused", False)))

    # data
    k = int(g("crops_per_image", 4))
    bs = int(g("batch_size", 16))
    per_load = max(1, bs // k)
    tr_base = make_base_dataset(tr_stores)
    va_base = make_base_dataset(va_stores)
    ds = CropDataset(tr_base, crop=int(g("crop", 512)), k=k, pos_frac=float(g("pos_frac", 0.7)),
                     peak_label=int(g("peak_label", 1)), rot90=bool(g("rot90", True)),
                     gain_jitter=float(g("gain_jitter", 0.2)))
    gen = torch.Generator().manual_seed(seed)
    sampler = torch.utils.data.RandomSampler(ds, replacement=True, num_samples=10 ** 8, generator=gen)
    nw = int(g("num_workers", 8))
    dl = torch.utils.data.DataLoader(ds, batch_size=per_load, sampler=sampler, num_workers=nw,
                                     collate_fn=crop_collate, pin_memory=True, drop_last=True,
                                     persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None,
                                     worker_init_fn=lambda w: (random.seed(seed * 1000 + w),
                                                               np.random.seed(seed * 1000 + w)))
    nv = min(int(g("val_images", 200)), len(va_base))
    all_v = np.linspace(0, len(va_base) - 1, nv).round().astype(int).tolist()
    my_v = all_v[rank::world]
    thrs = list(g("thresholds", DEFAULT_THR))
    log(event="data", train_images=len(tr_base), val_images_total=len(va_base), val_images_used=nv,
        crops_per_step=per_load * k * world, params=nparam)

    # optim
    base = unwrap(model)
    lr = float(g("lr", 3e-4))
    wd = float(g("weight_decay", 0.01))
    groups = base.param_groups(lr) if hasattr(base, "param_groups") else [{"params": [p for p in base.parameters() if p.requires_grad], "lr": lr}]
    for gr in groups:
        gr.setdefault("lr", lr)
        gr["base_lr"] = gr["lr"]
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
    crit = build_loss(cfg.get("loss", {}))
    max_steps = int(g("max_steps", 20000))
    warm = int(g("warmup", 300))
    val_every = int(g("val_every", 1000))
    amp = bool(g("amp", True)) and device.type == "cuda"
    clip = float(g("grad_clip", 1.0))
    val_tile = g("val_tile", None)

    def progress(step):
        return min(1.0, max(step / max_steps, (time.time() - t0) / time_budget))

    best = {"f1": -1.0}
    step, run_loss, n_loss = 0, 0.0, 0
    t_last = time.time()
    val_cost = 0.0
    model.train()
    it = iter(dl)
    done = False
    while not done:
        x, y = next(it)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pr = progress(step)
        f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * pr))
        for gr in opt.param_groups:
            gr["lr"] = gr["base_lr"] * max(f, 0.01)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(x)
        if hasattr(out, "logits"):
            out = out.logits
        out = out.float()
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        loss = crit(out, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        step += 1
        run_loss += float(loss.detach())
        n_loss += 1
        if step % 50 == 0:
            dt = time.time() - t_last
            log(event="train", step=step, loss=run_loss / n_loss, lr=opt.param_groups[0]["lr"],
                steps_per_s=round(50 / dt, 3), mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if amp else 0)
            run_loss, n_loss, t_last = 0.0, 0, time.time()
        # stop when steps or time (keeping room for one final validation) run out
        out_of_time = (time.time() - t0) + val_cost * 1.2 > time_budget
        last = step >= max_steps or out_of_time
        if ddp:
            flag = torch.tensor([1.0 if last else 0.0], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            last = bool(flag.item())
        if step % val_every == 0 or last:
            tv = time.time()
            c = validate(model, va_base, my_v, thrs, device, tile=val_tile, peak_label=int(g("peak_label", 1)))
            if ddp:
                ct = torch.from_numpy(c).to(device)
                dist.all_reduce(ct)
                c = ct.cpu().numpy()
            res = []
            for kk, t in enumerate(thrs):
                p, r, f1 = prf(*c[kk, 3:])
                pp, prr, pf = prf(*c[kk, :3])
                iou = c[kk, 0] / (c[kk, 0] + c[kk, 1] + c[kk, 2]) if c[kk, :3].sum() else 1.0
                res.append(dict(thr=t, peak_f1=f1, peak_p=p, peak_r=r, pix_f1=pf, pix_iou=iou))
            bi = max(range(len(res)), key=lambda q: res[q]["peak_f1"])
            val_cost = time.time() - tv
            log(event="val", step=step, best_thr=res[bi]["thr"], peak_f1=res[bi]["peak_f1"],
                peak_p=res[bi]["peak_p"], peak_r=res[bi]["peak_r"], pix_iou=res[bi]["pix_iou"],
                sweep=res, val_s=round(val_cost, 1))
            if is0:
                state = {"state_dict": unwrap(model).state_dict(), "cfg": cfg, "model": a.model,
                         "thr": res[bi]["thr"], "step": step, "val": res[bi]}
                torch.save(state, os.path.join(a.out, "last.pt"))
                if res[bi]["peak_f1"] > best["f1"]:
                    best = {"f1": res[bi]["peak_f1"], "step": step, "thr": res[bi]["thr"]}
                    torch.save(state, os.path.join(a.out, "best.pt"))
                    log(event="best", **best)
            if ddp:
                dist.barrier()
            done = last
    log(event="end", steps=step, best=best, wallclock_min=round((time.time() - t0) / 60, 2))
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
