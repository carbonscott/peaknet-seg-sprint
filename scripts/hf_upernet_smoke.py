"""CPU/GPU smoke test for seg/models/hf_upernet.py on a real store crop.
usage: python scripts/hf_upernet_smoke.py [stem_stride] [crop]"""
import sys, time, zarr, torch, numpy as np
sys.path.insert(0, ".")
from seg.models.hf_upernet import build_model
st = int(sys.argv[1]) if len(sys.argv) > 1 else 4
crop = int(sys.argv[2]) if len(sys.argv) > 2 else 512
root = "/sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet10k/peaknet10k/"
z = zarr.open(root + "mfxl1025422_r0309_peaknet.0000.zarr", "r")
im = torch.from_numpy(z["images"][0][:crop, :crop].astype(np.float32))[None, None]
lb = torch.from_numpy((z["labels"][0][:crop, :crop] == 1).astype(np.float32))[None, None]
m = build_model({"model": {"stem_stride": st}})
dev = "cuda" if torch.cuda.is_available() else "cpu"
m.to(dev).train(); im, lb = im.to(dev), lb.to(dev)
t = time.time()
out = m(im)
loss = torch.nn.functional.binary_cross_entropy_with_logits(out, lb)
loss.backward()
print("stride", st, "in", tuple(im.shape), "out", tuple(out.shape), "loss %.4f" % loss.item(),
      "params %.1fM" % (sum(p.numel() for p in m.parameters()) / 1e6), "fwd+bwd %.2fs" % (time.time() - t), dev)
m.eval()
with torch.no_grad():
    o2 = m(torch.rand(1, 1, 1665, 1667, device=dev) * 100)
print("odd-size eval out", tuple(o2.shape))
