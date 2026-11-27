"""hf_upernet: Hugging Face UperNetForSemanticSegmentation (ConvNeXt backbone) for Bragg-peak segmentation.

Upstream: transformers UperNetForSemanticSegmentation + openmmlab/upernet-convnext-{tiny,small,base}
weights from huggingface.co (ADE20k-trained). Glue only:
  - input adapter: 1-channel detector image -> log1p -> per-image standardisation -> repeated to 3 channels
  - head swapped to 1 class (binary peak logit), auxiliary FCN head off by default
  - optional stem stride change (patch_size 4 -> 2) so the decoder runs at 1/2 instead of 1/4
    resolution; Bragg peaks are only a few pixels wide. Pretrained 4x4 stem weights are kept.
  - pad H, W to a multiple of 32 (reflect), crop logits back.

Contract (ORDER 2): build_model(cfg) -> nn.Module; forward(x (B,1,H,W) float32) -> logits (B,1,H,W).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULTS = {
    "pretrained": "openmmlab/upernet-convnext-tiny",
    "stem_stride": 4,          # 4 = stock ConvNeXt; 2 = higher-res features (4x compute)
    "aux_head": False,
    "pad_multiple": 32,
    "clip_max": 65000.0,       # 65535 = saturated/bad pixels; clamp before log
}


def normalize(x, clip_max=65000.0):
    """(B,1,H,W) raw ADU -> log1p, per-image zero mean / unit std."""
    x = torch.log1p(x.clamp(min=0.0, max=clip_max))
    m = x.mean(dim=(2, 3), keepdim=True)
    s = x.std(dim=(2, 3), keepdim=True).clamp(min=1e-3)
    return (x - m) / s


class HFUperNetPeak(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import UperNetForSemanticSegmentation

        c = dict(DEFAULTS)
        m = cfg.get("model")
        c.update(m if isinstance(m, dict) else cfg)
        self.c = c
        kw = dict(num_labels=1, ignore_mismatched_sizes=True, use_auxiliary_head=bool(c["aux_head"]))
        if c.get("from_scratch"):
            from transformers import UperNetConfig
            conf = UperNetConfig.from_pretrained(c["pretrained"], **{k: v for k, v in kw.items()
                                                                       if k != "ignore_mismatched_sizes"})
            self.net = UperNetForSemanticSegmentation(conf)
        else:
            self.net = UperNetForSemanticSegmentation.from_pretrained(c["pretrained"], **kw)
        st = int(c["stem_stride"])
        if st != 4:
            pe = self.net.backbone.embeddings.patch_embeddings
            pe.stride = (st, st)
            # keep 4x4 kernel; pad so output = ceil(H/st)
            pe.padding = ((4 - st) // 2, (4 - st) // 2) if st < 4 else (0, 0)

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, None]
        B, _, H, W = x.shape
        x = normalize(x, self.c["clip_max"])
        p = int(self.c["pad_multiple"])
        ph, pw = (-H) % p, (-W) % p
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        x = x.expand(-1, 3, -1, -1)
        out = self.net(pixel_values=x)
        logits = out.logits
        if self.training and getattr(out, "auxiliary_logits", None) is not None:
            self.aux_logits = out.auxiliary_logits[..., :H, :W]
        return logits[..., :H, :W]


def build_model(cfg: dict) -> nn.Module:
    return HFUperNetPeak(cfg or {})
