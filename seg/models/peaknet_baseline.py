"""peaknet_baseline: the user's PeakNet (ConvNeXtV2 backbone + BiFPN + seg head) for Bragg-peak segmentation.

Upstream: github.com/carbonscott/peaknet (peaknet.modeling.convnextv2_bifpn_net.PeakNet), installed
editable in the shared env from /sdf/home/c/cwang31/codes/peaknet. The architecture and its default
hyper-parameters are taken unchanged from peaknet/train/hydra_config/train_config/base.yaml
(ConvNeXtV2 [96,192,384,768] x [3,3,9,3], 1-channel stem, BiFPN 256 features x 1 block, base_level 4,
seg head 256 ch, bilinear 4x upsample). Glue only:
  - seg head num_classes 2 -> 1 (one binary peak logit; upstream uses 2-class softmax focal loss,
    which is equivalent to a sigmoid of the logit difference)
  - input normalisation: "log1p_std" (clamp, log1p, per-image standardisation; default) or
    "instance" (upstream InstanceNorm: per-image mean/std of the raw image)
  - pad H, W to a multiple of 32 (reflect), crop logits back to (H, W).
  - optional patch_size 2 (stem stride 2 instead of 4): same layers, every feature map at 2x resolution,
    seg-head upsampling factors scaled to match; ~4x compute.

Contract (ORDER 2): build_model(cfg) -> nn.Module; forward(x (B,1,H,W) float32) -> logits (B,1,H,W).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULTS = {
    "patch_size": 4,          # ConvNeXtV2 stem stride; 2 = features and logits at 1/2 res (peaks are 1-2 px)
    "hidden_sizes": [96, 192, 384, 768],
    "depths": [3, 3, 9, 3],
    "drop_path_rate": 0.0,
    "bifpn_features": 256,
    "bifpn_blocks": 1,
    "bifpn_base_level": 4,
    "head_channels": 256,
    "head_groups": 32,
    "uses_learned_upsample": False,
    "norm": "log1p_std",
    "clip_max": 65000.0,
    "pad_multiple": 32,
}


def normalize(x, how="log1p_std", clip_max=65000.0):
    if how == "instance":  # upstream peaknet.tensor_transforms.InstanceNorm
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        s = x.std(dim=(1, 2, 3), keepdim=True)
        return (x - m) / (s + 1e-6)
    x = torch.log1p(x.clamp(min=0.0, max=clip_max))
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    s = x.std(dim=(1, 2, 3), keepdim=True).clamp(min=1e-3)
    return (x - m) / s


def build_peaknet(c):
    from transformers.models.convnextv2.configuration_convnextv2 import ConvNextV2Config
    from peaknet.modeling.bifpn_config import BiFPNConfig, BiFPNBlockConfig, BNConfig, FusionConfig
    from peaknet.modeling.convnextv2_bifpn_net import PeakNet, PeakNetConfig, SegHeadConfig

    ps = int(c.get("patch_size", 4))
    backbone = ConvNextV2Config(num_channels=1, patch_size=ps, num_stages=4,
                                hidden_sizes=list(c["hidden_sizes"]), depths=list(c["depths"]),
                                hidden_act="gelu", initializer_range=0.02, layer_norm_eps=1e-12,
                                drop_path_rate=float(c["drop_path_rate"]), image_size=256,
                                out_features=["stage1", "stage2", "stage3", "stage4"])
    bifpn = BiFPNConfig(num_blocks=int(c["bifpn_blocks"]),
                        block=BiFPNBlockConfig(relu_inplace=False, down_scale_factor=0.5, up_scale_factor=2,
                                               num_features=int(c["bifpn_features"]), num_levels=4,
                                               base_level=int(c["bifpn_base_level"]),
                                               bn=BNConfig(eps=1e-5, momentum=0.1), fusion=FusionConfig(eps=1e-5)))
    head = SegHeadConfig(up_scale_factor=[ps, 2 * ps, 4 * ps, 8 * ps], num_groups=int(c["head_groups"]),
                         out_channels=int(c["head_channels"]), num_classes=1, base_scale_factor=2,
                         uses_learned_upsample=bool(c["uses_learned_upsample"]))
    return PeakNet(PeakNetConfig(backbone=backbone, bifpn=bifpn, seg_head=head))


class PeakNetBinary(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = dict(DEFAULTS)
        m = cfg.get("model") if isinstance(cfg, dict) else None
        c.update(m if isinstance(m, dict) else (cfg or {}))
        self.c = c
        self.net = build_peaknet(c)

    def forward(self, x):
        if x.dim() == 3:
            x = x[:, None]
        B, _, H, W = x.shape
        x = normalize(x.float(), self.c["norm"], self.c["clip_max"])
        p = int(self.c["pad_multiple"])
        ph, pw = (-H) % p, (-W) % p
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        y = self.net(x)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y.float(), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return y[..., :H, :W]


def build_model(cfg):
    return PeakNetBinary(cfg or {})
