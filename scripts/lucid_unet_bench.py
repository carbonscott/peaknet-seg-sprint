"""Bench lucid_unet fwd+bwd at crop 512: conv2d fast path on vs off (same weights), + output diff."""
import sys
import time

import torch
import yaml

from seg.models.lucid_unet import build_model

bs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
cfg = yaml.safe_load(open("configs/lucid_unet.yaml"))
torch.backends.cudnn.benchmark = True
res = {}
x = torch.randn(bs, 1, 512, 512, device="cuda") * 50 + 40
tgt = (torch.rand(bs, 1, 512, 512, device="cuda") < 1e-4).float()
for fast in (True, False):
    torch.manual_seed(0)
    m = build_model({"model": dict(cfg["model"], conv2d_fastpath=fast)}).cuda()
    if fast:
        sd = m.state_dict()
    else:
        m.load_state_dict(sd)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        res[fast] = m.eval()(x[:1]).float()
    m.train()
    opt = torch.optim.AdamW(m.parameters(), 1e-4)
    torch.cuda.reset_peak_memory_stats()
    for i in range(8):
        if i == 3:
            torch.cuda.synchronize(); t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = m(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y.float(), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    print("fastpath=%s bs=%d: %.3f s/it  peak %.1f GB" % (fast, bs, (time.time() - t) / 5,
          torch.cuda.max_memory_allocated() / 1e9), flush=True)
    del m, opt
    torch.cuda.empty_cache()
print("max|fast-slow| logits (bf16):", (res[True] - res[False]).abs().max().item())
