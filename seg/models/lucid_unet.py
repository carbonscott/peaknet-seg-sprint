"""lucid_unet: lucidrains x-unet (XUnet, 2D) wrapped for binary Bragg-peak segmentation.

Upstream: https://github.com/lucidrains/x-unet (pip x-unet 0.5.0). The architecture is used
as-is; this file is glue only:
  * input adapter: 1-channel detector image -> robust per-image normalisation
    (median subtraction, fixed-scale asinh), XUnet built with channels=1, out_dim=1;
  * pad H, W to a multiple of 32 (XUnet downsamples 4x by 2) and crop logits back;
  * run x-unet's 1-frame Conv3d layers as conv2d (same weights and maths, much faster);
  * swap x-unet's einsum attention for torch scaled_dot_product_attention (same maths,
    O(N) memory), so the 1/8- and 1/16-resolution attention fits at full 1920x1920.

Model contract (ORDER 2): build_model(cfg) -> nn.Module; forward(x (B,1,H,W) float32)
-> logits (B,1,H,W).
"""
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_ARCH = dict(
    dim=32,
    dim_mults=(1, 2, 4, 8),
    num_blocks_per_stage=(2, 2, 2, 2),
    num_self_attn_per_stage=(0, 0, 0, 1),
    nested_unet_depths=(0, 0, 0, 0),
    consolidate_upsample_fmaps=True,
    use_convnext=False,
    attn_heads=8,
    attn_dim_head=32,
)


def _conv2d_forward(self, x):
    """Drop-in for an nn.Conv3d whose frame kernel is 1 on a 1-frame input: run it as conv2d
    (same weights, same maths; cuDNN conv3d is much slower than conv2d)."""
    if x.shape[2] != 1:
        return F.conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    y = F.conv2d(x[:, :, 0], self.weight[:, :, 0], self.bias, self.stride[1:], self.padding[1:],
                 self.dilation[1:], self.groups)
    return y.unsqueeze(2)


def _sdpa_attention_forward(self, x):
    """Drop-in for x_unet.Attention.forward using F.scaled_dot_product_attention."""
    from einops import rearrange
    f, h, w = x.shape[-3:]
    x = self.norm(x)
    q, k, v = self.to_qkv(x).chunk(3, dim=1)
    q, k, v = (rearrange(t, "b (h c) ... -> b h (...) c", h=self.heads) for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
    out = rearrange(out, "b h (f x y) d -> b (h d) f x y", f=f, x=h, y=w)
    return self.to_out(out)


class LucidUNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from x_unet import XUnet
        from x_unet.x_unet import Attention

        mcfg = dict(cfg.get("model", cfg))
        arch = dict(DEFAULT_ARCH)
        for k in DEFAULT_ARCH:
            if k in mcfg:
                v = mcfg[k]
                arch[k] = tuple(v) if isinstance(v, list) else v
        self.net = XUnet(channels=1, out_dim=1, **arch)
        if mcfg.get("sdpa_attention", True):
            for m in self.net.modules():
                if isinstance(m, Attention):
                    m.forward = types.MethodType(_sdpa_attention_forward, m)
        if mcfg.get("conv2d_fastpath", True):
            for m in self.net.modules():
                if type(m) is nn.Conv3d and m.kernel_size[0] == 1 and m.stride[0] == 1 \
                        and m.padding[0] == 0 and m.padding_mode == "zeros":
                    m.forward = types.MethodType(_conv2d_forward, m)
        self.pad_mult = int(mcfg.get("pad_multiple", 32))
        self.norm_scale = float(mcfg.get("norm_scale", 32.0))  # detector ADU per asinh unit
        self.norm_div = float(mcfg.get("norm_div", 3.0))
        self.sat_zero = float(mcfg.get("sat_zero", 0) or 0)  # e.g. 60000; 0 = off (xu_a..xu_d trained off)
        # bias the head toward "background" so early training is stable on ~1e-5 positives
        prior = float(mcfg.get("init_prior", 0.01))
        last = self.net.final_conv[-1]
        if getattr(last, "bias", None) is not None:
            nn.init.constant_(last.bias, float(torch.log(torch.tensor(prior / (1 - prior)))))

    def normalize(self, x):
        b = x.shape[0]
        flat = x.reshape(b, -1)
        sub = flat[:, :: max(1, flat.shape[1] // 65536)]
        med = sub.median(dim=1).values.view(b, 1, 1, 1)
        if self.sat_zero:  # saturated / bad pixels (65535 lines at panel edges) -> background
            x = torch.where(x >= self.sat_zero, med.expand_as(x), x)
        return torch.asinh((x - med) / self.norm_scale) / self.norm_div

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        H, W = x.shape[-2:]
        x = self.normalize(x.float())
        m = self.pad_mult
        ph, pw = (-H) % m, (-W) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="constant", value=0.0)
        y = self.net(x)
        if y.ndim == 5:  # (B, C, 1, H, W) if upstream keeps the frame axis
            y = y[:, :, 0]
        return y[..., :H, :W]


def build_model(cfg):
    return LucidUNet(cfg)
