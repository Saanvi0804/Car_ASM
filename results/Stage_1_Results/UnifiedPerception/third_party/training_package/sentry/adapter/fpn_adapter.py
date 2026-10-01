"""FPN Channel Adapter — projects backbone multi-scale features to target channel dims.

Bridges the channel dimension mismatch between a backbone (e.g. ConvNeXt-Tiny)
and a detection head (e.g. YOLO26 neck/PAN).

Example:
    ConvNeXt-Tiny outputs:          YOLO26 neck expects:
      stage3: (192, 80, 80)    →     P3: (256, 80, 80)
      stage4: (384, 40, 40)    →     P4: (256, 40, 40)
      stage5: (768, 20, 20)    →     P5: (512, 20, 20)

    Spatial dimensions match; only channels differ.
    This adapter uses 1x1 convolutions to project channels.

Usage:
    adapter = FPNChannelAdapter(
        in_channels=[192, 384, 768],
        out_channels=[256, 256, 512],
    )
    adapted = adapter(backbone_features)
    # adapted["p3"] = (B, 256, 80, 80)
    # adapted["p4"] = (B, 256, 40, 40)
    # adapted["p5"] = (B, 512, 20, 20)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FPNChannelAdapter(nn.Module):
    """Adapts multi-scale backbone features to target channel dimensions via 1x1 conv.

    Each level gets an independent 1x1 Conv2d + BatchNorm projection.
    No spatial resampling — spatial dims must already match between
    backbone output and head input.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: list[int],
        level_names: list[str] | None = None,
    ) -> None:
        """
        Args:
            in_channels: Channel dims from backbone at each scale level.
            out_channels: Channel dims expected by the detection head at each level.
            level_names: Names for the output keys (default: ["p3", "p4", "p5", ...]).
        """
        super().__init__()

        if len(in_channels) != len(out_channels):
            raise ValueError(
                f"in_channels ({len(in_channels)}) and out_channels ({len(out_channels)}) "
                "must have the same length"
            )

        self.num_levels = len(in_channels)
        self.level_names = level_names or [f"p{i + 3}" for i in range(self.num_levels)]

        self.projections = nn.ModuleList()
        for i, (c_in, c_out) in enumerate(zip(in_channels, out_channels)):
            if c_in == c_out:
                self.projections.append(nn.Identity())
                logger.info(
                    "  %s: passthrough (%d == %d)", self.level_names[i], c_in, c_out
                )
            else:
                self.projections.append(
                    nn.Sequential(
                        nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
                        nn.BatchNorm2d(c_out),
                    )
                )
                logger.info(
                    "  %s: Conv1x1 %d → %d", self.level_names[i], c_in, c_out
                )

        total_params = sum(p.numel() for p in self.parameters())
        logger.info("FPNChannelAdapter: %d levels, %d params", self.num_levels, total_params)

    def forward(
        self, features: list[torch.Tensor] | dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Project multi-scale features to target channels.

        Args:
            features: Either a list of tensors (one per scale level, coarse-to-fine
                      or fine-to-coarse) or a dict with stage keys.

        Returns:
            Dict mapping level names ("p3", "p4", "p5") to adapted tensors.
        """
        if isinstance(features, dict):
            # Extract tensors from dict, sorted by stage name.
            # Skip stages that aren't needed (e.g. stage1=raw input, stage2=too high-res).
            # ConvNeXt outputs stage1-5; YOLO needs the last 3 (stage3, stage4, stage5).
            sorted_keys = sorted(features.keys())
            # Take the LAST num_levels stages
            selected_keys = sorted_keys[-self.num_levels:]
            feat_list = [features[k] for k in selected_keys]
        else:
            feat_list = features

        if len(feat_list) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} feature levels, got {len(feat_list)}"
            )

        adapted = {}
        for i, (proj, feat) in enumerate(zip(self.projections, feat_list)):
            adapted[self.level_names[i]] = proj(feat)

        return adapted

    @staticmethod
    def for_convnext_to_yolo26(device: str = "cuda") -> "FPNChannelAdapter":
        """Factory: creates adapter for ConvNeXt-Tiny → YOLO26s neck.

        ConvNeXt-Tiny stages:        YOLO26s neck expects:
          stage3: 192 channels         P3: 256 channels  (80x80 at 640px)
          stage4: 384 channels         P4: 256 channels  (40x40)
          stage5: 768 channels         P5: 512 channels  (20x20)
        """
        adapter = FPNChannelAdapter(
            in_channels=[192, 384, 768],
            out_channels=[256, 256, 512],
            level_names=["p3", "p4", "p5"],
        )
        return adapter.to(device)

    @staticmethod
    def for_custom(
        in_channels: list[int],
        out_channels: list[int],
        device: str = "cuda",
    ) -> "FPNChannelAdapter":
        """Factory: creates adapter for arbitrary channel mappings."""
        adapter = FPNChannelAdapter(
            in_channels=in_channels,
            out_channels=out_channels,
        )
        return adapter.to(device)
