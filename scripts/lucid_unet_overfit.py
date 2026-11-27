"""Sanity: can lucid_unet overfit one real 256 crop with peaks? (CPU ok). Prints loss and probs."""
import numpy as np
import torch
import yaml

from seg.data import PeakDataset, load_split
from seg.losses import build_loss
from seg.models.lucid_unet import build_model

torch.manual_seed(0)
cfg = yaml.safe_load(open("configs/lucid_unet.yaml"))
ds = PeakDataset(load_split()["train"][:3])
for i in range(len(ds)):
    it = ds[i]
    lab = np.asarray(it["label"]) == 1
    if lab.sum() > 20:
        break
ys, xs = np.nonzero(lab)
cy, cx = int(ys[len(ys) // 2]), int(xs[len(xs) // 2])
img = np.asarray(it["image"]).reshape(lab.shape)
y0, x0 = max(0, cy - 128), max(0, cx - 128)
x = torch.from_numpy(img[y0:y0 + 256, x0:x0 + 256].copy())[None, None].float()
t = torch.from_numpy(lab[y0:y0 + 256, x0:x0 + 256].copy())[None, None].float()
print("crop peak px", int(t.sum()), "img range", float(x.min()), float(x.max()))
m = build_model(cfg)
crit = build_loss(cfg["loss"])
opt = torch.optim.AdamW(m.parameters(), 6e-4)
for s in range(81):
    y = m(x)
    loss = crit(y, t)
    opt.zero_grad(); loss.backward(); opt.step()
    if s % 10 == 0:
        p = torch.sigmoid(y.detach())
        print("step %d loss %.4f  mean p(peak) %.3f  max p(bg) %.3f" % (s, loss.item(), p[t > 0].mean(), p[t == 0].max()), flush=True)
