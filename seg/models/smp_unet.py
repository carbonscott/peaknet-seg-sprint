"""segmentation-models-pytorch U-Net (full-resolution decoder) for best_model candidates.

Upstream: github.com/qubvel-org/segmentation_models.pytorch 0.6.0, encoder weights from
huggingface.co/smp-hub (e.g. resnet34.imagenet). Glue only: 1-channel input (smp folds the RGB stem
into 1 channel), pointwise log normalization (seg.infer.norm_log), padding to a multiple of 32.

cfg["model"]: {arch: Unet|UnetPlusPlus|FPN..., encoder: resnet34, weights: imagenet|null,
               decoder_channels: [256,128,64,32,16], encoder_lr_mult: 1.0}
"""
import torch.nn as nn
import torch.nn.functional as F

from seg.infer import norm_log


class SmpSeg(nn.Module):
    def __init__(self, arch="Unet", encoder="resnet34", weights="imagenet", decoder_channels=None,
                 encoder_lr_mult=1.0):
        super().__init__()
        import segmentation_models_pytorch as smp
        kw = dict(encoder_name=encoder, encoder_weights=weights, in_channels=1, classes=1)
        if decoder_channels and arch in ("Unet", "UnetPlusPlus"):
            kw["decoder_channels"] = list(decoder_channels)
        self.net = getattr(smp, arch)(**kw)
        self.encoder_lr_mult = encoder_lr_mult

    def forward(self, x):
        H, W = x.shape[-2:]
        ph, pw = (-H) % 32, (-W) % 32
        x = norm_log(x)
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        return self.net(x)[..., :H, :W]

    def param_groups(self, lr):
        enc = list(self.net.encoder.parameters())
        ids = {id(p) for p in enc}
        rest = [p for p in self.parameters() if id(p) not in ids]
        return [{"params": enc, "lr": lr * self.encoder_lr_mult}, {"params": rest, "lr": lr}]


def build_model(cfg):
    m = dict(cfg.get("model", {}))
    return SmpSeg(arch=m.get("arch", "Unet"), encoder=m.get("encoder", "resnet34"),
                  weights=m.get("weights", "imagenet"), decoder_channels=m.get("decoder_channels"),
                  encoder_lr_mult=float(m.get("encoder_lr_mult", 1.0)))
