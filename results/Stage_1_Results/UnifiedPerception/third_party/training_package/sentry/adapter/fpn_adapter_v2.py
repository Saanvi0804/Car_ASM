"""FPN Spatial Adapter v2 — ConvNeXt→YOLO26 with 3×3 depthwise spatial mixing.

Improvement over FPNChannelAdapter (v1) which uses only 1×1 Conv for channel
projection. v2 adds a 3×3 depthwise conv before the 1×1 projection, giving
the adapter spatial context — it looks at a 3×3 neighborhood per channel
before remapping to YOLO's channel layout.

This specifically helps medium object detection where boundary features
matter. Pure 1×1 channel remapping (v1) cannot capture spatial patterns.

v1 (FPNChannelAdapter, 542K):   C_in → 1×1 Conv+BN → C_out
v2 (FPNSpatialAdapter, ~557K):  C_in → 3×3 DWConv+BN+GELU → 1×1 Conv+BN → C_out

Both output the same tensor shapes — drop-in replacement for YOLO26 neck injection.
No activation on final output — YOLO neck applies its own.

Usage:
    adapter = FPNSpatialAdapter.for_convnext_to_yolo26(device="cuda")
    adapted = adapter(backbone_features)
    # adapted["p3"] = (B, 256, 80, 80)
    # adapted["p4"] = (B, 256, 40, 40)
    # adapted["p5"] = (B, 512, 20, 20)

Reference: BRAINSTORM_LOG.md Decision 89 (Phase 2), docs/ADAPTER_GUIDE.md
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FPNSpatialAdapter(nn.Module):
    """ConvNeXt→YOLO26 adapter v2 with depthwise spatial mixing.

    Per scale:
        3×3 DWConv (groups=C_in, spatial mixing) + BN + GELU
        → 1×1 Conv (channel projection) + BN
        → output (no activation — YOLO neck applies its own)

    Args:
        in_channels: Channel dims from backbone at each scale level.
        out_channels: Channel dims expected by YOLO26 at each level.
        level_names: Names for output keys (default: ["p3", "p4", "p5"]).
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: list[int],
        level_names: list[str] | None = None,
    ) -> None:
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
            block = nn.Sequential(
                # Spatial mixing: 3×3 depthwise conv (each channel sees 3×3 neighborhood)
                nn.Conv2d(c_in, c_in, kernel_size=3, padding=1, groups=c_in, bias=False),
                nn.BatchNorm2d(c_in),
                nn.GELU(),
                # Channel projection: 1×1 conv (same as v1)
                nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
                nn.BatchNorm2d(c_out),
                # No activation — YOLO neck applies its own
            )
            self.projections.append(block)
            logger.info(
                "  %s: DWConv3x3(%d) + Conv1x1(%d→%d)",
                self.level_names[i], c_in, c_in, c_out,
            )

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "FPNSpatialAdapter (v2): %d levels, %d params (%.1fK)",
            self.num_levels, total_params, total_params / 1e3,
        )

    def forward(
        self, features: list[torch.Tensor] | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Project multi-scale features with spatial mixing + channel projection.

        Args:
            features: Either a list of tensors (one per scale, fine-to-coarse
                      or coarse-to-fine) or a dict with stage keys.

        Returns:
            Dict mapping level names ("p3", "p4", "p5") to adapted tensors.
        """
        if isinstance(features, dict):
            sorted_keys = sorted(features.keys())
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
    def for_convnext_to_yolo26(device: str = "cuda") -> "FPNSpatialAdapter":
        """Factory: creates v2 adapter for ConvNeXt-Tiny → YOLO26s neck.

        ConvNeXt-Tiny stages:        YOLO26s neck expects:
          stage3: 192 channels         P3: 256 channels  (80×80 at 640px)
          stage4: 384 channels         P4: 256 channels  (40×40)
          stage5: 768 channels         P5: 512 channels  (20×20)
        """
        adapter = FPNSpatialAdapter(
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
    ) -> "FPNSpatialAdapter":
        """Factory: creates v2 adapter for arbitrary channel mappings."""
        adapter = FPNSpatialAdapter(
            in_channels=in_channels,
            out_channels=out_channels,
        )
        return adapter.to(device)
