"""Losses for sparse binary peak segmentation (logits (B,1,H,W), target (B,H,W) or (B,1,H,W) in {0,1}).

Config (cfg["loss"]): {"type": "focal_dice" | "bce_dice" | "bce" | "focal" | "dice",
                       "focal_alpha": 0.25, "focal_gamma": 2.0, "pos_weight": 1.0,
                       "w_main": 1.0, "w_dice": 1.0}
"""
import torch
import torch.nn.functional as F


def _prep(logits, target):
    if target.dim() == 3:
        target = target.unsqueeze(1)
    return logits.float(), target.to(logits.device).float()


def bce_loss(logits, target, pos_weight=1.0):
    logits, target = _prep(logits, target)
    pw = torch.tensor(float(pos_weight), device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)


def focal_loss(logits, target, alpha=0.25, gamma=2.0):
    """Sigmoid focal loss (Lin et al. 2017), mean over pixels."""
    logits, target = _prep(logits, target)
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    loss = ce * (1 - p_t) ** gamma
    if alpha is not None and alpha >= 0:
        loss = (alpha * target + (1 - alpha) * (1 - target)) * loss
    return loss.mean()


def dice_loss(logits, target, eps=1.0):
    """Soft dice over the whole batch (robust when most crops have few positives)."""
    logits, target = _prep(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    return 1 - (2 * inter + eps) / (p.sum() + target.sum() + eps)


class SegLoss(torch.nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = dict(cfg or {})
        self.type = cfg.get("type", "focal_dice")
        self.alpha = cfg.get("focal_alpha", 0.25)
        self.gamma = cfg.get("focal_gamma", 2.0)
        self.pos_weight = cfg.get("pos_weight", 1.0)
        self.w_main = cfg.get("w_main", 1.0)
        self.w_dice = cfg.get("w_dice", 1.0)

    def forward(self, logits, target):
        t = self.type
        main = 0.0
        if t.startswith("focal"):
            main = focal_loss(logits, target, self.alpha, self.gamma)
        elif t.startswith("bce"):
            main = bce_loss(logits, target, self.pos_weight)
        elif t != "dice":
            raise ValueError("unknown loss type %r" % t)
        loss = self.w_main * main if t != "dice" else 0.0
        if t.endswith("dice"):
            loss = loss + self.w_dice * dice_loss(logits, target)
        return loss


def build_loss(cfg):
    return SegLoss(cfg)
