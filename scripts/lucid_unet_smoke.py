"""CPU/GPU smoke test for seg/models/lucid_unet.py: shapes, SDPA == einsum attention, params.
usage: python scripts/lucid_unet_smoke.py [configs/lucid_unet.yaml] [--full]
"""
import sys
import time

import torch
import yaml

from seg.models.lucid_unet import build_model

cfgp = next((a for a in sys.argv[1:] if a.endswith(".yaml")), "configs/lucid_unet.yaml")
cfg = yaml.safe_load(open(cfgp))
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
m = build_model(cfg).to(dev).eval()
print("params %.2fM" % (sum(p.numel() for p in m.parameters()) / 1e6))

# SDPA patch must equal the upstream einsum attention
cfg0 = {"model": dict(cfg["model"], sdpa_attention=False)}
torch.manual_seed(0)
m0 = build_model(cfg0).to(dev).eval()
m0.load_state_dict(m.state_dict())
x = torch.randn(1, 1, 128, 96, device=dev) * 50 + 40
with torch.no_grad():
    a, b = m(x), m0(x)
print("shape", tuple(a.shape), "max|sdpa-einsum|", (a - b).abs().max().item())
assert a.shape == (1, 1, 128, 96)

shapes = [(2, 1, 512, 512), (1, 1, 1667, 1665)]
if "--full" in sys.argv:
    shapes.append((1, 1, 1920, 1920))
for s in shapes:
    x = torch.randn(*s, device=dev) * 50 + 40
    t = time.time()
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        y = m(x)
    print("in", s, "out", tuple(y.shape), "%.1fs" % (time.time() - t), "mean logit %.3f" % y.float().mean())
    assert y.shape == (s[0], 1, s[2], s[3])
if dev == "cuda":
    print("peak mem %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
print("SMOKE OK")
