"""Student 2: a TinyViT-5M (Wu et al., 2022) for 224x224 flare forecasting (distillation-plan.txt, "TinyViT-5M"), built with timm.
Settings of tiny_vit_5m_224 in the official code (microsoft/Cream, TinyViT/models/tiny_vit.py):
embed_dims [64, 128, 160, 320], depths [2, 2, 6, 2], num_heads [2, 4, 5, 10], window_sizes [7, 7, 14, 7].
Stage 1 is convolutional (MBConv blocks); stages 2-4 are attention blocks. At 224 the windows of stages 3 and 4
cover the whole map, so their attention is global. Average pooling + LayerNorm give the [B, 320] that feeds the heads.
Trained from scratch: no ImageNet weights, since the 26 input channels are not RGB.
Shape check, from downstream_examples/solar_flare_forcasting:
    conda run -n suryaenv python experiments/student/student224_tinyvit.py
"""

import timm
import torch
import torch.nn as nn


class StudentTinyViT(nn.Module):
    """[B, 13 channels, 2 times, 224, 224] -> flare logit [B, 1] and projected feature [B, 1280].
    The 2 times enter as 26 channels. The feature head is the projection g, used only by feature KD.
    """

    def __init__(self, in_channels=26, feature_dim=1280):
        super().__init__()
        # num_classes=0: timm's head stops after average pooling + LayerNorm and returns [B, 320]
        self.backbone = timm.create_model("tiny_vit_5m_224", pretrained=False, in_chans=in_channels, num_classes=0)
        width = self.backbone.num_features                                    # 320
        self.classifier = nn.Linear(width, 1)                                 # output 1: flare logit
        self.feature_head = nn.Linear(width, feature_dim)                     # output 2: projection to the teacher's CLS size

    def forward(self, ts):
        h = self.backbone(ts.flatten(1, 2))                                   # [B, 13, 2, H, W] -> [B, 26, H, W] -> [B, 320]
        return self.classifier(h), self.feature_head(h)


if __name__ == "__main__":
    model = StudentTinyViT()
    x = torch.randn(2, 13, 2, 224, 224)
    h = x.flatten(1, 2)
    print(f"{'input':34s}{tuple(x.shape)}\n{'time steps -> channels':34s}{tuple(h.shape)}")

    backbone = model.backbone
    h = backbone.patch_embed(h)
    print(f"{'patch embed (two stride-2 convs)':34s}{tuple(h.shape)}")
    for k, stage in enumerate(backbone.stages, 1):
        h = stage(h)
        print(f"{f'stage {k} ' + ('(MBConv)' if k == 1 else '(merge + attention)'):34s}{tuple(h.shape)}")
    h = backbone.head(h)
    print(f"{'average pool + LayerNorm':34s}{tuple(h.shape)}")

    logit, feature = model(x)
    print(f"{'classification head':34s}{tuple(logit.shape)}\n{'feature head':34s}{tuple(feature.shape)}")

    n = lambda m: sum(p.numel() for p in m.parameters())
    print(f"\nparameters: backbone {n(model.backbone) / 1e6:.2f} M, classifier {n(model.classifier)}, "
          f"feature head {n(model.feature_head) / 1e6:.2f} M")
