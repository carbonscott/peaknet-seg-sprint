"""lucidrains segformer-pytorch (Segformer) wrapped for 1-channel Bragg-peak segmentation.

Upstream: https://github.com/lucidrains/segformer-pytorch (pip segformer-pytorch 0.0.6), used unchanged.
Glue only:
  * input adapter: log1p of the non-negative signal, then per-image standardisation; channels=1,
    so no 1->3 replication is needed (the model trains from scratch; lucidrains ships no weights).
  * padding to a multiple of 32 inside forward, cropped back after.
  * full-resolution head: upstream Segformer predicts at 1/4 resolution, but Bragg peaks are 1-2 px.
    We ask it for 16 = 4x4 output channels and apply nn.PixelShuffle(4) (sub-pixel upsampling), then
    an optional tiny residual conv refinement that also sees the normalised image at full resolution.

Contract (ORDER 2): build_model(cfg) -> nn.Module; forward(x (B,1,H,W) float32) -> logits (B,1,H,W).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from segformer_pytorch import Segformer

DEFAULTS = dict(
    dims=[32, 64, 160, 256],          # MiT-b0-like widths
    heads=[1, 2, 5, 8],
    ff_expansion=[8, 8, 4, 4],
    reduction_ratio=[8, 4, 2, 1],
    num_layers=2,
    decoder_dim=256,
    refine=True,
    refine_ch=16,
    prior=0.01,                        # initial peak probability (bias init)
    norm_eps=1.0,
    sat=60000.0,                       # pixels >= sat are saturated/flagged (65535 lines at panel edges) -> 0
    norm="per_image",                  # per_image: log1p + per-image standardisation (lseg_a/b)
                                       # pointwise: seg.infer.norm_log (keeps absolute brightness; tiles == crops)
)


def normalize(x, eps=1.0, sat=60000.0):
    """x (B,1,H,W) raw detector counts -> log1p(relu(x)) standardised per image.
    Saturated / flagged pixels (>= sat, e.g. the 65535 lines at panel edges) are zeroed first: without
    this the model learns 'very bright = peak' and fires on every saturated line (val precision 0.1)."""
    x = x.float()
    x = torch.where(x >= sat, torch.zeros_like(x), x)
    x = torch.log1p(torch.clamp(x, min=0.0))
    m = x.mean(dim=(-2, -1), keepdim=True)
    s = x.std(dim=(-2, -1), keepdim=True)
    return (x - m) / (s + 1e-3 * eps)


class LucidSegformerPeak(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        c = dict(DEFAULTS)
        c.update((cfg or {}).get("model", cfg or {}) if isinstance(cfg, dict) else {})
        self.c = c
        self.seg = Segformer(
            dims=tuple(c["dims"]), heads=tuple(c["heads"]), ff_expansion=tuple(c["ff_expansion"]),
            reduction_ratio=tuple(c["reduction_ratio"]), num_layers=c["num_layers"],
            channels=1, decoder_dim=c["decoder_dim"], num_classes=16,
        )
        self.shuffle = nn.PixelShuffle(4)
        bias = -math.log((1 - c["prior"]) / c["prior"])
        last = self.seg.to_segmentation[-1]
        nn.init.constant_(last.bias, bias)
        self.refine = None
        if c["refine"]:
            r = c["refine_ch"]
            self.refine = nn.Sequential(
                nn.Conv2d(2, r, 3, padding=1), nn.GELU(),
                nn.Conv2d(r, r, 3, padding=1), nn.GELU(),
                nn.Conv2d(r, 1, 1),
            )
            nn.init.zeros_(self.refine[-1].weight)
            nn.init.zeros_(self.refine[-1].bias)

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, None]
        H, W = x.shape[-2:]
        ph, pw = (-H) % 32, (-W) % 32
        if self.c["norm"] == "pointwise":
            from seg.infer import norm_log
            x = norm_log(x.float(), sat=self.c["sat"])
        else:
            x = normalize(x, self.c["norm_eps"], self.c["sat"])
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        y = self.shuffle(self.seg(x))                  # (B,1,Hp,Wp) logits
        if self.refine is not None:
            y = y + self.refine(torch.cat([y, x.to(y.dtype)], dim=1))
        return y[..., :H, :W]


def build_model(cfg=None):
    return LucidSegformerPeak(cfg)
