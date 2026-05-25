"""Weight-shared Siamese U-Net for change-style damage segmentation.

Pre and post tiles pass through a shared ResNet-34 encoder. Feature maps at each
pyramid scale are concatenated channel-wise then projected back to the original
channel count via a 1x1 conv before the standard smp Unet decoder.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class SiameseUNet(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", classes=5):
        super().__init__()
        base = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
        )
        self.encoder = base.encoder
        self.decoder = base.decoder
        self.segmentation_head = base.segmentation_head

        enc_channels = self.encoder.out_channels  # tuple, e.g. (3, 64, 64, 128, 256, 512)
        self.fusers = nn.ModuleList(
            [nn.Conv2d(c * 2, c, kernel_size=1) for c in enc_channels]
        )

    def forward(self, pre, post):
        f_pre = self.encoder(pre)
        f_post = self.encoder(post)
        fused = [fuse(torch.cat([a, b], dim=1)) for fuse, a, b in zip(self.fusers, f_pre, f_post)]
        decoded = self.decoder(*fused)
        return self.segmentation_head(decoded)

    def load_stage1_encoder(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        sd = ckpt.get("model", ckpt)
        enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        missing, unexpected = self.encoder.load_state_dict(enc_sd, strict=False)
        print(
            f"Loaded Stage-1 encoder weights: {len(enc_sd)} tensors "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
