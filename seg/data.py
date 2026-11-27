"""Shared loader for the peaknet10k Zarr v2 stores (owner: h1).

    from seg.data import load_split, PeakDataset, pad_collate
    sp = load_split()                                   # {"train": [...], "val": [...], "test": [...]}
    ds = PeakDataset(sp["train"], crop=512, pos_frac=0.9)   # training: random crops
    ds = PeakDataset(sp["test"])                        # evaluation: full native-resolution images

Item = {"image": float32 tensor (1,H,W) (raw detector values, may be negative),
        "label": int64 tensor (H,W) (0 background, 1 peak),
        "meta": {"store", "index", "identifier", "H", "W", "y0", "x0"}}

Normalization is the model wrapper's job (see normalize_image for a shared default).
Images are 1920x1920 or 1667x1665; full images have mixed sizes, so batch them with
pad_collate (pads to a common size, multiple of `pad_multiple`) or use batch_size=1.

Store lookup: if `stage_dir` (default /lscratch/cwang31/pkseg) holds a complete copy of a store,
it is used; otherwise the shared path under `root`. See NOTES.md for staging.
"""
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_ROOT = "/sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet10k/peaknet10k"
STAGE_DIR = "/lscratch/cwang31/pkseg"
SPLIT_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "split", "result.json")


def load_split(path=SPLIT_JSON):
    """Return {"train": [...], "val": [...], "test": [...]} store basenames."""
    return json.load(open(path))["stores"]


def store_path(name, root=DATA_ROOT, stage_dir=STAGE_DIR):
    if stage_dir:
        p = os.path.join(stage_dir, name)
        if os.path.isfile(os.path.join(p, ".complete")):
            return p
    return os.path.join(root, name)


def store_len(path):
    return int(json.load(open(os.path.join(path, "images", ".zarray")))["shape"][0])


def normalize_image(x, mode="log1p"):
    """Shared default input normalization (numpy or torch, any shape).
    log1p: clip negatives to 0, log1p, then divide by a fixed scale (~ log1p of a bright peak)."""
    if mode is None or mode == "none":
        return x
    if isinstance(x, torch.Tensor):
        return torch.log1p(x.clamp(min=0)) / 8.0
    return np.log1p(np.clip(x, 0, None)) / 8.0


class PeakDataset(Dataset):
    """Images of a list of stores.

    crop: None (full image) or int/(h,w) random crop (padded with 0 if the image is smaller).
    pos_frac: with this probability a crop is placed to contain a random peak pixel (if any).
    max_images: keep only the first N items (smoke tests). seed: crop RNG seed (None = random).
    """

    def __init__(self, stores, root=DATA_ROOT, crop=None, pos_frac=0.0, stage_dir=STAGE_DIR,
                 max_images=None, seed=None, peak_label=1):
        self.stores = list(stores)
        self.paths = [store_path(s, root, stage_dir) for s in self.stores]
        self.index = [(si, i) for si, p in enumerate(self.paths) for i in range(store_len(p))]
        if max_images:
            self.index = self.index[:max_images]
        self.crop = (crop, crop) if isinstance(crop, int) else crop
        self.pos_frac = pos_frac
        self.peak_label = peak_label
        self.rng = random.Random(seed)
        self._open = {}
        self._ids = {}

    def __len__(self):
        return len(self.index)

    def _group(self, si):
        g = self._open.get(si)
        if g is None:
            import zarr  # zarr 2.18; opened lazily so each DataLoader worker has its own handles
            g = self._open[si] = zarr.open(self.paths[si], "r")
        return g

    def _ident(self, si, g):
        ids = self._ids.get(si)
        if ids is None:  # the metadata chunk holds all 40 dicts: decode once per store
            try:
                ids = [m.get("identifier", "") for m in g["metadata"][:]]
            except Exception:
                ids = [""] * g["images"].shape[0]
            self._ids[si] = ids
        return ids

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_open"] = {}
        return d

    def _crop_origin(self, lab, H, W, ch, cw):
        if self.pos_frac and self.rng.random() < self.pos_frac:
            ys, xs = np.nonzero(lab == self.peak_label)
            if len(ys):
                k = self.rng.randrange(len(ys))
                y0 = self.rng.randint(max(0, ys[k] - ch + 1), min(ys[k], max(0, H - ch)))
                x0 = self.rng.randint(max(0, xs[k] - cw + 1), min(xs[k], max(0, W - cw)))
                return y0, x0
        return self.rng.randint(0, max(0, H - ch)), self.rng.randint(0, max(0, W - cw))

    def __getitem__(self, k):
        si, i = self.index[k]
        g = self._group(si)
        img = np.asarray(g["images"][i], dtype=np.float32)
        lab = np.asarray(g["labels"][i], dtype=np.int64)
        H, W = img.shape
        y0 = x0 = 0
        if self.crop:
            ch, cw = self.crop
            y0, x0 = self._crop_origin(lab, H, W, ch, cw)
            img = img[y0:y0 + ch, x0:x0 + cw]
            lab = lab[y0:y0 + ch, x0:x0 + cw]
            if img.shape != (ch, cw):
                pi = np.zeros((ch, cw), np.float32)
                pl = np.zeros((ch, cw), np.int64)
                pi[:img.shape[0], :img.shape[1]] = img
                pl[:lab.shape[0], :lab.shape[1]] = lab
                img, lab = pi, pl
        ident = self._ident(si, g)[i]
        meta = dict(store=self.stores[si], index=i, identifier=ident, H=H, W=W, y0=y0, x0=x0)
        return dict(image=torch.from_numpy(np.ascontiguousarray(img))[None],
                    label=torch.from_numpy(np.ascontiguousarray(lab)), meta=meta)


def pad_collate(items, pad_multiple=32):
    """Batch items of different sizes: zero-pad image and label to a common (H,W), a multiple of
    pad_multiple. meta keeps each item's own H, W; crop predictions back with [:H, :W]."""
    H = max(it["image"].shape[-2] for it in items)
    W = max(it["image"].shape[-1] for it in items)
    H = -(-H // pad_multiple) * pad_multiple
    W = -(-W // pad_multiple) * pad_multiple
    img = torch.zeros(len(items), 1, H, W, dtype=torch.float32)
    lab = torch.zeros(len(items), H, W, dtype=torch.int64)
    for b, it in enumerate(items):
        h, w = it["label"].shape
        img[b, :, :h, :w] = it["image"]
        lab[b, :h, :w] = it["label"]
    return dict(image=img, label=lab, meta=[it["meta"] for it in items])


def _bench(argv=None):
    """python -m seg.data --stores a.zarr b.zarr [--root R] [--n 40]: time image+label reads."""
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", nargs="+", required=True)
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--stage-dir", default=STAGE_DIR)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args(argv)
    ds = PeakDataset(a.stores, root=a.root, stage_dir=a.stage_dir, max_images=a.n)
    t = time.time()
    pos = 0
    for k in range(len(ds)):
        pos += int((ds[k]["label"] == 1).sum())
    dt = time.time() - t
    print(json.dumps(dict(paths=ds.paths[:2], n=len(ds), sec=round(dt, 2),
                          img_per_s=round(len(ds) / dt, 1), peak_px=pos)))


if __name__ == "__main__":
    _bench()
