"""Student 1: a simple CNN for 224x224 flare forecasting (distillation-plan.txt, "Simple CNN").
Shape check, from downstream_examples/solar_flare_forcasting:
    conda run -n suryaenv python experiments/student/student224_cnn.py
"""

import torch
import torch.nn as nn


def conv_block(c_in, c_out, kernel=3, stride=1, padding=1):
    """Conv -> BatchNorm -> ReLU."""
    return nn.Sequential(nn.Conv2d(c_in, c_out, kernel, stride, padding, bias=False),
                         nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class StudentCNN(nn.Module):
    """[B, 13 channels, 2 times, 224, 224] -> flare logit [B, 1] and projected feature [B, 1280].
    The 2 times enter as 26 channels. The feature head is the projection g, used only by feature KD.
    """

    def __init__(self, in_channels=26, feature_dim=1280):
        super().__init__()
        self.backbone = nn.Sequential(
            conv_block(in_channels, 64), nn.MaxPool2d(2),        # -> [B,  64, 112, 112]
            conv_block(64, 128), nn.MaxPool2d(2),                # -> [B, 128,  56,  56]
            conv_block(128, 256), nn.MaxPool2d(2),               # -> [B, 256,  28,  28]
            conv_block(256, 512), nn.MaxPool2d(2),               # -> [B, 512,  14,  14]
            conv_block(512, 512), nn.MaxPool2d(2),               # -> [B, 512,   7,   7]
            conv_block(512, 512, kernel=3, stride=3, padding=0), # -> [B, 512,   2,   2]
            # no conv_block here because 1. conv_block has activation and batchnorm and it does not make sense to have
            # batchnorm on 1x1 feature map, as a single value has no variance and mean and batch norm is about mean and variance.
            nn.Conv2d(512, 512, kernel_size=2, stride=2),        # -> [B, 512,   1,   1]
            nn.Flatten(),                                        # -> [B, 512]
        )
        self.classifier = nn.Linear(512, 1)                      # output 1: flare logit
        self.feature_head = nn.Linear(512, feature_dim)          # output 2: projection to the teacher's CLS size

    def forward(self, ts):
        h = self.backbone(ts.flatten(1, 2))                      # [B, 13, 2, H, W] -> [B, 26, H, W] -> [B, 512]
        return self.classifier(h), self.feature_head(h)


if __name__ == "__main__":
    model = StudentCNN()
    x = torch.randn(2, 13, 2, 224, 224)
    h = x.flatten(1, 2)
    print(f"{'input':34s}{tuple(x.shape)}\n{'time steps -> channels':34s}{tuple(h.shape)}")
    for layer in model.backbone:
        h = layer(h)
        print(f"{'conv block' if isinstance(layer, nn.Sequential) else layer.__class__.__name__:34s}{tuple(h.shape)}")
    logit, feature = model(x)
    print(f"{'classification head':34s}{tuple(logit.shape)}\n{'feature head':34s}{tuple(feature.shape)}")

    n = lambda m: sum(p.numel() for p in m.parameters())
    print(f"\nparameters: backbone {n(model.backbone) / 1e6:.2f} M, classifier {n(model.classifier)}, "
          f"feature head {n(model.feature_head) / 1e6:.2f} M")
