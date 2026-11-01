"""Student 2: a MobileNetV2 (Sandler et al., 2018) for 224x224 flare forecasting.
Inverted residual blocks shrink 224 -> 7 pixels; global average pooling gives the [B, 1280] that feeds the heads.
Shape check, from downstream_examples/solar_flare_forcasting:
    conda run -n suryaenv python experiments/student/student224_mobilenet.py
"""

from collections import OrderedDict

import torch
import torch.nn as nn


def conv_bn(c_in, c_out, kernel=1, stride=1, groups=1, act=True):
    """Conv -> BatchNorm (-> ReLU6). groups=c_in makes it depthwise: one filter per channel."""
    layers = [nn.Conv2d(c_in, c_out, kernel, stride, kernel // 2, groups=groups, bias=False), nn.BatchNorm2d(c_out)]
    return nn.Sequential(*layers, nn.ReLU6(inplace=True)) if act else nn.Sequential(*layers)


class InvertedResidual(nn.Module):
    """Expand (1x1) -> depthwise 3x3 -> project (1x1, no ReLU), plus a skip when the shape is unchanged."""

    def __init__(self, c_in, c_out, stride, expand):
        super().__init__()
        hidden = c_in * expand
        self.block = nn.Sequential(
            *([conv_bn(c_in, hidden)] if expand != 1 else []),                # widen:  c_in -> c_in * expand
            conv_bn(hidden, hidden, kernel=3, stride=stride, groups=hidden),  # spatial mixing, each channel on its own
            conv_bn(hidden, c_out, act=False),                                # narrow: linear, a ReLU here would lose information
        )
        self.skip = stride == 1 and c_in == c_out

    def forward(self, x):
        return x + self.block(x) if self.skip else self.block(x)


# (expansion, output channels, blocks, stride of the first block): the MobileNetV2 paper's table
STAGES = [(1, 16, 1, 1),     # -> [B,  16, 112, 112]
          (6, 24, 2, 2),     # -> [B,  24,  56,  56]
          (6, 32, 3, 2),     # -> [B,  32,  28,  28]
          (6, 64, 4, 2),     # -> [B,  64,  14,  14]
          (6, 96, 3, 1),     # -> [B,  96,  14,  14]
          (6, 160, 3, 2),    # -> [B, 160,   7,   7]
          (6, 320, 1, 1)]    # -> [B, 320,   7,   7]


class StudentMobileNet(nn.Module):
    """[B, 13 channels, 2 times, 224, 224] -> flare logit [B, 1]; the 2 times enter as 26 channels.
    feature_dim: adds a feature head (1280 = teacher CLS size) for feature KD; None for logit KD.
    """

    def __init__(self, in_channels=26, feature_dim=None):
        super().__init__()
        layers = [("stem", conv_bn(in_channels, 32, kernel=3, stride=2))]    # -> [B, 32, 112, 112]
        c_in = 32
        for k, (t, c, n, s) in enumerate(STAGES, 1):
            layers.append((f"stage {k}", nn.Sequential(*[InvertedResidual(c_in if i == 0 else c, c, s if i == 0 else 1, t)
                                                          for i in range(n)])))
            c_in = c
        layers += [("conv 1x1", conv_bn(c_in, 1280)),                         # -> [B, 1280, 7, 7]
                   ("global average pool", nn.AdaptiveAvgPool2d(1)),          # -> [B, 1280, 1, 1]
                   ("flatten", nn.Flatten())]                                 # -> [B, 1280]
        self.backbone = nn.Sequential(OrderedDict(layers))
        self.classifier = nn.Linear(1280, 1)                                  # flare logit
        self.feature_head = nn.Linear(1280, feature_dim) if feature_dim else None

    def embed(self, ts):
        """Shared [B, 1280] features that both heads read."""
        return self.backbone(ts.flatten(1, 2))                                # [B, 13, 2, H, W] -> [B, 26, H, W]

    def forward(self, ts):
        """Flare logit [B, 1]."""
        return self.classifier(self.embed(ts))

    def features(self, ts):
        """Flare logit [B, 1] and projected features [B, feature_dim], for feature KD."""
        h = self.embed(ts)
        return self.classifier(h), self.feature_head(h)


if __name__ == "__main__":
    model = StudentMobileNet(feature_dim=1280)
    x = torch.randn(2, 13, 2, 224, 224)
    h = x.flatten(1, 2)
    print(f"{'input':34s}{tuple(x.shape)}\n{'time steps -> channels':34s}{tuple(h.shape)}")
    for name, layer in model.backbone.named_children():
        h = layer(h)
        print(f"{name:34s}{tuple(h.shape)}")
    logits, feats = model.features(x)
    print(f"{'classification head':34s}{tuple(logits.shape)}\n{'feature head':34s}{tuple(feats.shape)}")

    n = lambda m: sum(p.numel() for p in m.parameters())
    print(f"\nparameters: backbone {n(model.backbone) / 1e6:.2f} M, classifier {n(model.classifier)}, "
          f"feature head {n(model.feature_head) / 1e6:.2f} M")
