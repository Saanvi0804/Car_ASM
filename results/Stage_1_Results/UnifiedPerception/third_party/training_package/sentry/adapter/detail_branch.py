"""Lightweight CNN detail branch for fine-grained spatial features.

Provides local edge/texture features that ViT global attention misses.
Runs in parallel with the ViT backbone on the same input image.
Produces multi-scale features at P3 (1/8), P4 (1/16), P5 (1/32) strides.

Inspired by DEIMv2/STA (arXiv:2509.20787) detail branch design.

Usage:
    branch = DetailBranch(out_channels=64)
    d3, d4, d5 = branch(images)  # images: (B, 3, 640, 640)
    # d3: (B, 64, 80, 80), d4: (B, 64, 40, 40), d5: (B, 64, 20, 20)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DetailBranch(nn.Module):
    """Ultra-lightweight CNN for fine-grained spatial features at 3 scales.

    Architecture: 3 stride-2 convs (640→320→160→80) for P3,
    then 2 additional stride-2 convs for P4 (40) and P5 (20).

    Uses standard 3×3 convolutions (not depthwise-separable).
    Conv + BN only, NO activation on output — matches ConvNeXt adapter pattern.
    YOLO neck applies its own activations.

    Args:
        out_channels: Number of output channels per scale. Default 64.
    """

    def __init__(self, out_channels: int = 64) -> None:
        super().__init__()

        # Stem: 3 stride-2 convs → 640/8 = 80 (P3 scale)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # No activation on final stem output
        )

        # P4: stride-2 from P3 → 80/2 = 40
        self.down_p4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # No activation — YOLO neck applies its own
        )

        # P5: stride-2 from P4 → 40/2 = 20
        self.down_p5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # No activation
        )

        total_params = sum(p.numel() for p in self.parameters())
        logger.info("DetailBranch: %d params, out_channels=%d", total_params, out_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract multi-scale detail features.

        Args:
            x: Input image tensor (B, 3, H, W). Must be the SAME augmented+normalized
               tensor fed to the ViT backbone (ensures train/infer consistency).

        Returns:
            Tuple of (d3, d4, d5):
                d3: (B, out_ch, H/8, W/8) — P3 scale, fine detail
                d4: (B, out_ch, H/16, W/16) — P4 scale
                d5: (B, out_ch, H/32, W/32) — P5 scale, coarse
        """
        d3 = self.stem(x)       # (B, 64, H/8, W/8)
        d4 = self.down_p4(d3)   # (B, 64, H/16, W/16)
        d5 = self.down_p5(d4)   # (B, 64, H/32, W/32)
        return d3, d4, d5
