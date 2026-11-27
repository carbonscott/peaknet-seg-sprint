"""Tiny conv net used only to smoke-test seg/train.py and seg/evaluate.py (not a statement model)."""
import torch.nn as nn

from seg.infer import norm_log


class Tiny(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, ch, 3, padding=1), nn.ReLU(), nn.Conv2d(ch, ch, 3, padding=1),
                                 nn.ReLU(), nn.Conv2d(ch, 1, 1))

    def forward(self, x):
        return self.net(norm_log(x))


def build_model(cfg):
    return Tiny(cfg.get("model", {}).get("ch", 16))
