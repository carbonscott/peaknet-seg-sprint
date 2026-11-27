"""Hugging Face SegFormer (nvidia/mit-b*) as a 1-class Bragg-peak segmenter.

Glue only: the network is transformers.SegformerForSemanticSegmentation loaded from the Hub
weights. We add (1) a per-image robust normalization of the 1-channel detector image,
(2) a 1x1 conv 1->3 channel adapter (initialised to replicate the channel),
(3) padding to a multiple of 32 and bilinear upsampling of the decoder logits (1/4 res, or
1/2 res with model.first_stride=2) back to the input size.

Contract (ORDER 2): build_model(cfg) -> nn.Module; forward(x (B,1,H,W) float32) -> logits (B,1,H,W).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def robust_norm(x, clip=(-10.0, 1e4)):
    """Per-image (x - median) / (1.4826 * MAD), then asinh. x: (B,1,H,W)."""
    B = x.shape[0]
    flat = x.reshape(B, -1).float()
    # subsample for speed on 1920x1920 images; the median of 1/16 of pixels is plenty
    sub = flat[:, ::16] if flat.shape[1] > 65536 else flat
    med = sub.median(dim=1, keepdim=True).values
    mad = (sub - med).abs().median(dim=1, keepdim=True).values
    z = (flat - med) / (1.4826 * mad + 1e-3)
    z = torch.nan_to_num(z, nan=0.0, posinf=clip[1], neginf=clip[0]).clamp(*clip)
    return torch.asinh(z).reshape_as(x)


class HFSegformerPeak(nn.Module):
    def __init__(self, pretrained="nvidia/mit-b2", first_stride=4, pad_to=32, drop_path=None, enc_lr_mult=0.25):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        kw = dict(num_labels=1, ignore_mismatched_sizes=True)
        conf = SegformerConfig.from_pretrained(pretrained)
        if first_stride != 4:
            strides = list(conf.strides)
            strides[0] = int(first_stride)
            kw["strides"] = strides
        if drop_path is not None:
            kw["drop_path_rate"] = float(drop_path)
        self.net = SegformerForSemanticSegmentation.from_pretrained(pretrained, **kw)
        self.adapter = nn.Conv2d(1, 3, kernel_size=1)
        with torch.no_grad():
            self.adapter.weight.fill_(1.0)
            self.adapter.bias.zero_()
        self.pad_to = pad_to
        self.enc_lr_mult = enc_lr_mult

    def param_groups(self, lr):
        """seg.train hook: pretrained encoder at lr*enc_lr_mult, decoder + adapter at lr."""
        enc = [p for p in self.net.segformer.parameters() if p.requires_grad]
        ids = {id(p) for p in enc}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in ids]
        return [{"params": enc, "lr": lr * self.enc_lr_mult}, {"params": rest, "lr": lr}]

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, None]
        H, W = x.shape[-2:]
        z = robust_norm(x)
        ph = (-H) % self.pad_to
        pw = (-W) % self.pad_to
        if ph or pw:
            z = F.pad(z, (0, pw, 0, ph), value=0.0)
        out = self.net(pixel_values=self.adapter(z)).logits  # (B,1,h/4,w/4)
        out = F.interpolate(out, size=z.shape[-2:], mode="bilinear", align_corners=False)
        return out[..., :H, :W]


def build_model(cfg):
    m = cfg.get("model", cfg) if isinstance(cfg, dict) else {}
    return HFSegformerPeak(pretrained=m.get("pretrained", "nvidia/mit-b2"),
                           first_stride=m.get("first_stride", 4),
                           pad_to=m.get("pad_to", 32),
                           drop_path=m.get("drop_path"),
                           enc_lr_mult=m.get("enc_lr_mult", 0.25))
