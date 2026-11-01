"""Student 1: a plain CNN for 224x224 flare forecasting, plus the logit- and feature-KD losses.
Shape check, from downstream_examples/solar_flare_forcasting:
    conda run -n suryaenv python experiments/student/student224_cnn.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(c_in, c_out, kernel=3, stride=1, padding=1):
    """Conv -> BatchNorm -> ReLU."""
    return nn.Sequential(nn.Conv2d(c_in, c_out, kernel, stride, padding, bias=False),
                         nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class StudentCNN(nn.Module):
    """[B, 13 channels, 2 times, 224, 224] -> flare logit [B, 1]; the 2 times enter as 26 channels.
    feature_dim: adds a feature head (1280 = teacher CLS size) for feature KD; None for logit KD.
    """

    def __init__(self, in_channels=26, feature_dim=None):
        super().__init__()
        self.backbone = nn.Sequential(
            conv_block(in_channels, 64), nn.MaxPool2d(2),        # -> [B,  64, 112, 112]
            conv_block(64, 128), nn.MaxPool2d(2),                # -> [B, 128,  56,  56]
            conv_block(128, 256), nn.MaxPool2d(2),               # -> [B, 256,  28,  28]
            conv_block(256, 512), nn.MaxPool2d(2),               # -> [B, 512,  14,  14]
            conv_block(512, 512), nn.MaxPool2d(2),               # -> [B, 512,   7,   7]
            conv_block(512, 512, kernel=3, stride=3, padding=0), # -> [B, 512,   2,   2]
            nn.Conv2d(512, 512, kernel_size=2, stride=2),        # -> [B, 512,   1,   1]
            nn.Flatten(),                                        # -> [B, 512]
        )
        self.classifier = nn.Linear(512, 1)                      # flare logit
        self.feature_head = nn.Linear(512, feature_dim) if feature_dim else None

    def embed(self, ts):
        """Shared [B, 512] features that both heads read."""
        return self.backbone(ts.flatten(1, 2))                   # [B, 13, 2, H, W] -> [B, 26, H, W]

    def forward(self, ts):
        """Flare logit [B, 1]."""
        return self.classifier(self.embed(ts))

    def features(self, ts):
        """Flare logit [B, 1] and projected features [B, feature_dim], for feature KD."""
        h = self.embed(ts)
        return self.classifier(h), self.feature_head(h)


def logit_kd_loss(student_logits, teacher_logits, labels, pos_weight, alpha=1.0, beta=1.0):
    """alpha * weighted BCE(labels) + beta * MSE(teacher logits); returns the loss and its parts, for logging.
    pos_weight: weight of a flare sample (non-flares / flares in train).
    """
    labels = labels.float().view_as(student_logits)
    teacher_logits = teacher_logits.float().view_as(student_logits)
    l_wbce = F.binary_cross_entropy_with_logits(student_logits, labels, pos_weight=pos_weight)
    l_logit = F.mse_loss(student_logits, teacher_logits)
    return alpha * l_wbce + beta * l_logit, {"wbce": l_wbce.item(), "logit_mse": l_logit.item()}


def feature_kd_loss(student_logits, student_features, teacher_features, labels, pos_weight, alpha=1.0, beta=1.0):
    """alpha * weighted BCE(labels) + beta * mean(1 - cosine(student, teacher features)); returns the loss and its parts.
    Features are [B, 1280]; the cosine term is 0 when they point the same way, 2 when opposite.
    """
    labels = labels.float().view_as(student_logits)
    l_wbce = F.binary_cross_entropy_with_logits(student_logits, labels, pos_weight=pos_weight)
    l_feature = (1 - F.cosine_similarity(student_features, teacher_features.float(), dim=1)).mean()
    return alpha * l_wbce + beta * l_feature, {"wbce": l_wbce.item(), "feature_cos": l_feature.item()}


if __name__ == "__main__":
    model = StudentCNN(feature_dim=1280)
    x = torch.randn(2, 13, 2, 224, 224)
    h = x.flatten(1, 2)
    print(f"{'input':34s}{tuple(x.shape)}\n{'time steps -> channels':34s}{tuple(h.shape)}")
    for layer in model.backbone:
        h = layer(h)
        print(f"{layer.__class__.__name__ if not isinstance(layer, nn.Sequential) else 'conv block':34s}{tuple(h.shape)}")
    logits, feats = model.features(x)
    print(f"{'classification head':34s}{tuple(logits.shape)}\n{'feature head':34s}{tuple(feats.shape)}")

    n = lambda m: sum(p.numel() for p in m.parameters())
    print(f"\nparameters: backbone {n(model.backbone) / 1e6:.2f} M, classifier {n(model.classifier)}, "
          f"feature head {n(model.feature_head) / 1e6:.2f} M")

    loss, parts = logit_kd_loss(logits, torch.randn(2, 1), torch.tensor([1.0, 0.0]), torch.tensor(7.79))
    print(f"loss {loss.item():.3f}  parts {parts}")
    loss, parts = feature_kd_loss(logits, feats, torch.randn(2, 1280), torch.tensor([1.0, 0.0]), torch.tensor(7.79))
    print(f"feature KD loss {loss.item():.3f}  parts {parts}")
