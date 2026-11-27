"""Independent inference pass for the proof report (h8): re-run the final best_model checkpoint(s)
on the whole test split WITHOUT seg/evaluate.py, seg/infer.py or seg/data.py.

Images come straight from the Zarr stores; each member is built with its own
seg.models.<name>.build_model(cfg) (the model contract) and called directly; padding to /32
(reflect), flip TTA, member averaging and the optional saturated-pixel mask are implemented
here. Probability maps are written as float16 <store>__<idx>.npy, then scored by
scripts/independent_reeval.py (its own matcher).

usage: python scripts/independent_infer.py --member smp_unet:/sdf/.../best.pt [--member ...] \
          [--tta] [--mask-saturated D] --out-probs DIR [--split test] [--limit N]
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = "/sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet10k/peaknet10k"


def load_member(spec, device):
    import torch
    import yaml
    parts = spec.split(":")
    name, ck_path = parts[0], parts[1]
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = yaml.safe_load(open(parts[2])) if len(parts) > 2 else ck.get("cfg", {})
    model = importlib.import_module("seg.models.%s" % name).build_model(cfg)
    sd = None
    for k in ("model", "state_dict", "model_state_dict", "ema"):
        if isinstance(ck, dict) and isinstance(ck.get(k), dict):
            sd = ck[k]
            break
    sd = {k.replace("module.", "", 1).replace("_orig_mod.", "", 1): v for k, v in (sd or ck).items()}
    r = model.load_state_dict(sd, strict=False)
    print("member %s %s: missing %d unexpected %d, ckpt thr %s" % (name, ck_path, len(r.missing_keys), len(r.unexpected_keys), ck.get("thr")), flush=True)
    return model.to(device).eval()


def forward(model, x):
    import torch
    import torch.nn.functional as F
    H, W = x.shape[-2:]
    ph, pw = (-H) % 32, (-W) % 32
    xp = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
        y = model(xp)
    y = y.logits if hasattr(y, "logits") else y
    return torch.sigmoid(y.float()[..., :H, :W])


def main():
    import torch
    import torch.nn.functional as F
    import zarr
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", action="append", required=True)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--mask-saturated", type=int, default=0)
    ap.add_argument("--out-probs", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--split-json", default="results/split/result.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reverse", action="store_true", help="walk the split backwards (second worker)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--f32", action="store_true", help="save float32 maps (needed for thresholds very close to 1)")
    ap.add_argument("--meet", help="stop when this other worker's output dir already has the image")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("host=%s SLURM_JOB_ID=%s SLURM_STEP_ID=%s device=%s members=%s tta=%s mask_saturated=%d" % (
        os.uname().nodename, os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_STEP_ID"), dev, a.member, a.tta, a.mask_saturated), flush=True)
    members = [load_member(m, dev) for m in a.member]
    sp = json.load(open(a.split_json))
    os.makedirs(a.out_probs, exist_ok=True)
    flips = [(), (-1,), (-2,), (-2, -1)] if a.tta else [()]
    n, t0 = 0, time.time()
    items = []
    for s in sp["stores"][a.split]:
        items += [(s, i) for i in range(zarr.open(os.path.join(DATA, s), "r")["images"].shape[0])]
    if a.reverse:
        items = items[::-1]
    for s, i in items:
            z = zarr.open(os.path.join(DATA, s), "r")
            out = os.path.join(a.out_probs, "%s__%d.npy" % (s, i))
            if (a.skip_existing and os.path.exists(out)) or (a.meet and os.path.exists(os.path.join(a.meet, os.path.basename(out)))):
                print("met existing output at %s__%d after %d images; stopping" % (s, i, n), flush=True)
                break
            img = np.asarray(z["images"][i], np.float32)
            x = torch.from_numpy(img)[None, None].to(dev)
            with torch.no_grad():
                acc = 0
                for m in members:
                    for d in flips:
                        q = forward(m, torch.flip(x, d) if d else x)
                        acc = acc + (torch.flip(q, d) if d else q)
                p = acc / (len(members) * len(flips))
                if a.mask_saturated:
                    k = 2 * a.mask_saturated + 1
                    sat = F.max_pool2d((x >= 65535).float(), k, stride=1, padding=a.mask_saturated) > 0
                    p = p.masked_fill(sat, 0.0)
            np.save(out + ".tmp.npy", p[0, 0].cpu().numpy().astype(np.float32 if a.f32 else np.float16))
            os.replace(out + ".tmp.npy", out)
            n += 1
            if n % 100 == 0:
                print("  %d images, %.2f img/s" % (n, n / (time.time() - t0)), flush=True)
            if a.limit and n >= a.limit:
                break
    print("done %d images in %.1f min" % (n, (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
