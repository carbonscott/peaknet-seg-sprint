"""Tiny GPU/cuDNN diagnostic: free memory, cudnn version, conv2d and conv3d forward."""
import time
import os
import torch
if os.environ.get("NO_CUDNN"):
    torch.backends.cudnn.enabled = False
t = time.time()
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0), "init %.0fs" % (time.time() - t), flush=True)
print("mem_get_info GB", [round(v / 1e9, 1) for v in torch.cuda.mem_get_info()], flush=True)
print("cudnn", torch.backends.cudnn.version(), flush=True)
for name, conv, shp in [("conv2d", torch.nn.Conv2d(1, 8, 3).cuda(), (1, 1, 64, 64)),
                        ("conv3d", torch.nn.Conv3d(1, 8, (1, 7, 7)).cuda(), (1, 1, 1, 64, 64))]:
    try:
        y = conv(torch.randn(*shp, device="cuda"))
        torch.cuda.synchronize()
        print(name, "OK", tuple(y.shape), flush=True)
    except Exception as e:
        print(name, "FAIL", repr(e)[:200], flush=True)
if os.environ.get("BENCH"):
    import sys
    sys.argv = ["x", "--train", "--bs=8"]
    sys.path.insert(0, os.getcwd())
    exec(open("scripts/lucid_unet_smoke.py").read())
