"""Shared full-resolution inference and input helpers.

Model contract (ORDER 2): model(x (B,1,H,W) float32 raw detector values) -> logits (B,1,H,W).
Wrappers normalize their own input; `norm_log` below is the suggested pointwise normalization
(pointwise, so random crops and full images see the same transform).

    from seg.infer import predict_full, norm_log
    prob = predict_full(model, image_hw, device="cuda", tile=None, tta=False)  # np.float32 (H,W) in [0,1]
"""
import numpy as np

import torch
import torch.nn.functional as F


def norm_log(x, center=4.0, scale=2.0, sat=60000.0):
    """Pointwise signed log transform: background (~40-60 ADU) -> ~0, peaks (100s-1000s) -> 1-3.
    Values >= sat (saturated / bad pixels) are zeroed. Works on any tensor shape."""
    x = torch.where(x >= sat, torch.zeros_like(x), x)
    y = torch.sign(x) * torch.log1p(x.abs())
    return (y - center) / scale


def pad_to_multiple(x, m=32, mode="constant"):
    """x (B,C,H,W) -> padded x, (H,W). Pads bottom/right."""
    H, W = x.shape[-2:]
    ph, pw = (-H) % m, (-W) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode=mode)
    return x, (H, W)


def _as_bchw(image):
    t = torch.as_tensor(np.asarray(image) if not torch.is_tensor(image) else image).float()
    while t.dim() < 4:
        t = t.unsqueeze(0)
    return t


@torch.no_grad()
def _forward(model, x, amp=True, multiple=32):
    xp, (H, W) = pad_to_multiple(x, multiple)
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=amp and x.device.type == "cuda"):
        out = model(xp)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if hasattr(out, "logits"):
        out = out.logits
    out = out.float()
    if out.shape[-2:] != xp.shape[-2:]:
        out = F.interpolate(out, size=xp.shape[-2:], mode="bilinear", align_corners=False)
    return out[..., :H, :W]


@torch.no_grad()
def predict_logits(model, x, tile=None, overlap=64, amp=True, multiple=32):
    """x (B,1,H,W) on device -> logits (B,1,H,W). tile=None: whole image at once.
    tile=int: sliding window of tile x tile with `overlap`, averaged logits."""
    if not tile or (x.shape[-2] <= tile and x.shape[-1] <= tile):
        return _forward(model, x, amp, multiple)
    B, C, H, W = x.shape
    out = torch.zeros((B, 1, H, W), device=x.device)
    cnt = torch.zeros((1, 1, H, W), device=x.device)
    step = tile - overlap
    ys = list(range(0, max(H - tile, 0) + 1, step))
    xs = list(range(0, max(W - tile, 0) + 1, step))
    if ys[-1] + tile < H:
        ys.append(H - tile)
    if xs[-1] + tile < W:
        xs.append(W - tile)
    for y0 in ys:
        for x0 in xs:
            y0c, x0c = max(y0, 0), max(x0, 0)
            patch = x[..., y0c:y0c + tile, x0c:x0c + tile]
            lo = _forward(model, patch, amp, multiple)
            out[..., y0c:y0c + tile, x0c:x0c + tile] += lo
            cnt[..., y0c:y0c + tile, x0c:x0c + tile] += 1
    return out / cnt.clamp(min=1)


SAT_VALUE = 65535.0


def saturated_mask_px():
    """Default dilation (px) for saturated-pixel suppression: env PK_MASK_SATURATED (unset/0 = off)."""
    import os
    try:
        return int(os.environ.get("PK_MASK_SATURATED", "0") or 0)
    except ValueError:
        return 0


def suppress_saturated(prob, x, dilate=1, sat=SAT_VALUE):
    """Zero prob where the raw image x >= sat, with the mask dilated by `dilate` px.
    prob, x: (B,1,H,W) tensors on the same device. dilate < 0 disables."""
    if dilate is None or dilate < 0:
        return prob
    m = (x >= sat).float()
    if dilate > 0:
        m = F.max_pool2d(m, 2 * dilate + 1, stride=1, padding=dilate)
    return prob * (1 - m)


@torch.no_grad()
def predict_prob(model, x, tile=None, overlap=64, amp=True, tta=False, multiple=32, mask_saturated=None):
    """x (B,1,H,W) tensor on the model's device -> sigmoid prob (B,1,H,W).
    tta=True averages probabilities over identity, h-flip, v-flip and hv-flip.
    mask_saturated: None -> env PK_MASK_SATURATED (default off); 0/False -> off; int d >= 1 -> zero prob
    where image >= 65535, mask dilated by d px (True means 1)."""
    if not tta:
        p = torch.sigmoid(predict_logits(model, x, tile, overlap, amp, multiple))
    else:
        acc = 0
        flips = [(), (-1,), (-2,), (-2, -1)]
        for d in flips:
            xi = torch.flip(x, d) if d else x
            q = torch.sigmoid(predict_logits(model, xi, tile, overlap, amp, multiple))
            acc = acc + (torch.flip(q, d) if d else q)
        p = acc / len(flips)
    ms = saturated_mask_px() if mask_saturated is None else int(mask_saturated)
    if ms:
        p = suppress_saturated(p, x, dilate=ms)
    return p


@torch.no_grad()
def predict_full(model, image, device=None, tile=None, overlap=64, amp=True, tta=False, multiple=32,
                 mask_saturated=None):
    """One image (H,W) / (1,H,W) numpy or tensor -> np.float32 prob map (H,W). device defaults to the model's.
    mask_saturated: see predict_prob (None -> env PK_MASK_SATURATED, default off)."""
    if device is None:
        device = next(model.parameters()).device
    x = _as_bchw(image).to(device)
    p = predict_prob(model, x, tile, overlap, amp, tta, multiple, mask_saturated)
    return p[0, 0].float().cpu().numpy()
