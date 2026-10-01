"""ViT→YOLO26 Adapter v2 — improved scale conversion and spatial mixing.

Changes from v1 (ViTYOLOAdapter):
    1. ViTDetScaleConversion_v2: adds 3×3 Conv+BN+GELU after deconv/conv for
       spatial refinement at each scale
    2. BiFusion_v2: adds 3×3 DWConv+BN+GELU before 1×1 projection — spatial
       context in the fused (ViT+Detail) features before channel projection
    3. Reuses: ViTFeatureExtractor, MultiLayerFusion, DetailBranch from v1 files

v1 ViTYOLOAdapter: 2.6M params
v2 ViTYOLOAdapter_v2: ~3.0-3.2M params (additional spatial refinement layers)

Both output identical tensor shapes — drop-in replacement.

Usage:
    adapter = ViTYOLOAdapter_v2.for_dinov2_to_yolo26s(imgsz=640)
    p3, p4, p5 = adapter(hidden_states, images)

Reference: BRAINSTORM_LOG.md Decision 89, docs/ADAPTER_DESIGN_ViT_YOLO26.md
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse components from v1 — no duplication
from sentry.adapter.vit_feature_extractor import ViTFeatureExtractor
from sentry.adapter.vit_yolo_adapter import MultiLayerFusion
from sentry.adapter.detail_branch import DetailBranch

logger = logging.getLogger(__name__)

_PATCH_SIZES = {"dinov2": 14, "dinov3": 16}
_YOLO26S_CHANNELS = [256, 256, 512]  # P3, P4, P5


class ViTDetScaleConversion_v2(nn.Module):
    """ViTDet-style scale conversion with spatial refinement.

    v1: ConvTranspose2d/Conv2d only (raw scale change)
    v2: ConvTranspose2d/Conv2d → 3×3 Conv+BN+GELU (spatial refinement after scale change)

    The 3×3 conv after scale conversion lets the adapter learn to clean up
    deconvolution artifacts and refine spatial patterns at each scale.

    Args:
        in_dim: Input feature dimension from ViT (384 for ViT-S)
    """

    def __init__(self, in_dim: int = 384) -> None:
        super().__init__()
        # P3: upsample 2× + depthwise spatial refinement
        self.up_p3 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, in_dim, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.GELU(),
            # Depthwise 3×3: spatial refinement without cross-channel mixing
            # Cleans up deconvolution checkerboard artifacts
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False),
            nn.BatchNorm2d(in_dim),
        )
        # P4: channel refinement + depthwise spatial refinement
        self.proj_p4 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.GELU(),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False),
            nn.BatchNorm2d(in_dim),
        )
        # P5: downsample 2× + depthwise spatial refinement
        self.down_p5 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.GELU(),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False),
            nn.BatchNorm2d(in_dim),
        )

        total = sum(p.numel() for p in self.parameters())
        logger.info("  ViTDetScaleConversion_v2: %d params (%.1fK)", total, total / 1e3)

    def forward(
        self, fused_maps: list[torch.Tensor], target_sizes: list[tuple[int, int]] | None = None,
    ) -> list[torch.Tensor]:
        """Convert 3 fused maps to P3/P4/P5 with spatial refinement.

        Args:
            fused_maps: list of 3 tensors, each (B, D, H_p, W_p)
            target_sizes: Optional exact target sizes for non-power-of-2 grids (DINOv2).

        Returns:
            [P3, P4, P5] at correct YOLO strides
        """
        p3 = self.up_p3(fused_maps[0])
        p4 = self.proj_p4(fused_maps[1])
        p5 = self.down_p5(fused_maps[2])

        # For non-power-of-2 grids (DINOv2: 45×45 → deconv gives 90×90, need 80×80)
        if target_sizes is not None:
            results = [p3, p4, p5]
            for i, (t_h, t_w) in enumerate(target_sizes):
                if results[i].shape[2] != t_h or results[i].shape[3] != t_w:
                    results[i] = F.interpolate(
                        results[i], size=(t_h, t_w), mode="bilinear", align_corners=False,
                    )
            return results

        return [p3, p4, p5]


class BiFusion_v2(nn.Module):
    """Bi-Fusion v2: spatial mixing before channel projection + optional post-refinement.

    v1: Concat(ViT, Detail) → 1×1 Conv+BN
    v2: Concat(ViT, Detail) → 3×3 DWConv+BN+GELU → 1×1 Conv+BN [→ 3×3 Conv+BN if post-refine]

    The 3×3 DWConv gives spatial context to the fused features before
    projecting to YOLO's channel layout.

    The optional post-refinement (ViTDet-style) adds a 3×3 Conv+BN AFTER
    the 1×1 projection for per-level spatial boundary refinement. This
    specifically helps small/medium object localization precision.

    Args:
        vit_channels: ViT feature dimension (384 for ViT-S)
        detail_channels: Detail branch output channels (64 or 128)
        out_channels: Per-scale output channels for YOLO26s [256, 256, 512]
        use_post_refinement: If True, adds 3×3 Conv+BN after 1×1 projection.
            Turn-key: False skips it (falls back to v2-base behavior).
    """

    def __init__(
        self,
        vit_channels: int = 384,
        detail_channels: int = 64,
        out_channels: list[int] | None = None,
        use_post_refinement: bool = True,
    ) -> None:
        super().__init__()
        out_channels = out_channels or _YOLO26S_CHANNELS
        in_ch = vit_channels + detail_channels
        self.use_post_refinement = use_post_refinement

        self.projections = nn.ModuleList()
        self.level_names = ["p3", "p4", "p5"]

        for i, out_ch in enumerate(out_channels):
            self.projections.append(
                nn.Sequential(
                    # Spatial mixing on fused features
                    nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
                    nn.BatchNorm2d(in_ch),
                    nn.GELU(),
                    # Channel projection (same as v1)
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    # No activation — YOLO neck applies its own
                )
            )
            logger.info(
                "  BiFusion_v2 %s: DWConv3x3(%d) + Conv1x1(%d→%d)",
                self.level_names[i], in_ch, in_ch, out_ch,
            )

        # Optional per-level 3×3 depthwise refinement after projection
        # Uses depthwise conv (not full 3×3) to stay within PVA <5M budget
        # Still provides spatial boundary refinement at each scale
        if use_post_refinement:
            self.post_refine = nn.ModuleList()
            for i, out_ch in enumerate(out_channels):
                self.post_refine.append(
                    nn.Sequential(
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1,
                                  groups=out_ch, bias=False),  # depthwise
                        nn.BatchNorm2d(out_ch),
                        # No activation — YOLO neck applies its own
                    )
                )
                logger.info(
                    "  BiFusion_v2 %s: +post-refine Conv3x3(%d→%d)",
                    self.level_names[i], out_ch, out_ch,
                )
        else:
            self.post_refine = None
            logger.info("  BiFusion_v2: post-refinement OFF (v2-base behavior)")

    def forward(
        self,
        vit_features: list[torch.Tensor],
        detail_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Fuse ViT and detail features with spatial mixing, project to YOLO channels."""
        outputs = {}
        for i, (vit_f, det_f, proj, name) in enumerate(
            zip(vit_features, detail_features, self.projections, self.level_names)
        ):
            fused = torch.cat([vit_f, det_f], dim=1)
            out = proj(fused)
            if self.post_refine is not None:
                out = self.post_refine[i](out)
            outputs[name] = out
        return outputs


class ViTYOLOAdapter_v2(nn.Module):
    """ViT→YOLO26 adapter v2 with spatial refinement.

    Architecture (same 5-component pipeline as v1, with v2 upgrades on C and E):
        [A] ViTFeatureExtractor  — extract layers [3,6,9,12] (reused from v1)
        [B] MultiLayerFusion     — learnable per-scale weights (reused from v1)
        [C] ViTDetScaleConversion_v2 — deconv/conv + 3×3 spatial refinement (NEW)
        [D] DetailBranch         — lightweight CNN (reused from v1)
        [E] BiFusion_v2          — DWConv spatial mixing + 1×1 projection (NEW)

    Args:
        backbone_type: "dinov2" or "dinov3"
        imgsz: Input image size (must be multiple of 32)
        layer_indices: ViT layers to extract (default [3,6,9,12])
        vit_dim: ViT hidden dimension (384 for ViT-S)
        detail_channels: Detail branch output channels. v2 default=128, v1=64.
        out_channels: YOLO26s target channels [256, 256, 512]
        use_post_refinement: Add 3×3 Conv+BN after BiFusion projection (ViTDet-style).
            True (default) = improved boundary precision. False = v2-base behavior.
        device: Target device
    """

    def __init__(
        self,
        backbone_type: Literal["dinov2", "dinov3"] = "dinov2",
        imgsz: int = 640,
        layer_indices: list[int] | None = None,
        vit_dim: int = 384,
        detail_channels: int = 128,
        out_channels: list[int] | None = None,
        use_post_refinement: bool = True,
        device: str = "cuda",
    ) -> None:
        super().__init__()

        if imgsz % 32 != 0:
            raise ValueError(f"imgsz={imgsz} must be a multiple of 32")

        self.backbone_type = backbone_type
        self.imgsz = imgsz
        self.vit_dim = vit_dim
        out_channels = out_channels or _YOLO26S_CHANNELS

        patch_size = _PATCH_SIZES[backbone_type]
        self.H_p, self.W_p = ViTFeatureExtractor.compute_grid_size(imgsz, patch_size)

        self.target_sizes = [
            (imgsz // 8, imgsz // 8),
            (imgsz // 16, imgsz // 16),
            (imgsz // 32, imgsz // 32),
        ]

        # [A] Multi-layer ViT feature extraction (reused from v1)
        self.feature_extractor = ViTFeatureExtractor(
            backbone_type=backbone_type,
            layer_indices=layer_indices or [3, 6, 9, 12],
            feature_dim=vit_dim,
        )

        # [B] Per-scale learnable fusion (reused from v1)
        num_layers = len(self.feature_extractor.layer_indices)
        self.fusion = MultiLayerFusion(num_layers=num_layers, num_scales=3)

        # [C] v2: ViTDet-style scale conversion WITH spatial refinement
        self.scale_conversion = ViTDetScaleConversion_v2(in_dim=vit_dim)

        # [D] Lightweight CNN detail branch (reused from v1)
        self.detail_branch = DetailBranch(out_channels=detail_channels)

        # [E] v2: Bi-Fusion with DWConv spatial mixing + optional post-refinement
        self.bi_fusion = BiFusion_v2(
            vit_channels=vit_dim,
            detail_channels=detail_channels,
            out_channels=out_channels,
            use_post_refinement=use_post_refinement,
        )

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "ViTYOLOAdapter_v2: backbone=%s, imgsz=%d, grid=%dx%d, "
            "total_params=%d (%.1fK), trainable=%d (%.1fK)",
            backbone_type, imgsz, self.H_p, self.W_p,
            total, total / 1e3, trainable, trainable / 1e3,
        )

    def forward(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        images: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the full v2 adapter pipeline.

        Args:
            hidden_states: ViT hidden states from model(output_hidden_states=True).
            images: Input image tensor (B, 3, H, W). H/W can vary per batch
                when multi-scale training is active.

        Returns:
            Dict with keys "p3", "p4", "p5" at YOLO26s expected shapes.
        """
        # Compute grid size dynamically from actual input (supports multi-scale)
        actual_imgsz = images.shape[-1]  # H (assumes square)
        patch_size = _PATCH_SIZES[self.backbone_type]
        H_p = actual_imgsz // patch_size
        W_p = actual_imgsz // patch_size

        # Compute target sizes dynamically from actual input
        target_sizes = [
            (actual_imgsz // 8, actual_imgsz // 8),    # P3
            (actual_imgsz // 16, actual_imgsz // 16),  # P4
            (actual_imgsz // 32, actual_imgsz // 32),  # P5
        ]

        # A: Extract multi-layer features (uses dynamic H_p, W_p)
        layer_features = self.feature_extractor(hidden_states, H_p, W_p)

        # B: Fuse layers per scale
        fused = self.fusion(layer_features)

        # C (v2): Scale conversion with spatial refinement (uses dynamic target_sizes)
        aligned = self.scale_conversion(fused, target_sizes=target_sizes)

        # D: Detail branch on raw pixels (naturally adapts to input size)
        d3, d4, d5 = self.detail_branch(images)

        # E (v2): Bi-Fusion with DWConv spatial mixing
        outputs = self.bi_fusion(aligned, (d3, d4, d5))

        return outputs

    @staticmethod
    def for_dinov2_to_yolo26s(
        imgsz: int = 640, detail_channels: int = 128,
        use_post_refinement: bool = True, device: str = "cuda",
    ) -> "ViTYOLOAdapter_v2":
        """Factory: creates v2 adapter for DINOv2 ViT-S → YOLO26s."""
        adapter = ViTYOLOAdapter_v2(
            backbone_type="dinov2",
            imgsz=imgsz,
            vit_dim=384,
            detail_channels=detail_channels,
            out_channels=_YOLO26S_CHANNELS,
            use_post_refinement=use_post_refinement,
        )
        return adapter.to(device)

    @staticmethod
    def for_dinov3_to_yolo26s(
        imgsz: int = 640, detail_channels: int = 128,
        use_post_refinement: bool = True, device: str = "cuda",
    ) -> "ViTYOLOAdapter_v2":
        """Factory: creates v2 adapter for DINOv3 ViT-S → YOLO26s."""
        adapter = ViTYOLOAdapter_v2(
            backbone_type="dinov3",
            imgsz=imgsz,
            vit_dim=384,
            detail_channels=detail_channels,
            out_channels=_YOLO26S_CHANNELS,
            use_post_refinement=use_post_refinement,
        )
        return adapter.to(device)
