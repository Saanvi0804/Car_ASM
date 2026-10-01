"""ViT→YOLO26 Adapter — bridges DINOv2/DINOv3 ViT features to YOLO26s neck.

Combines:
    - Multi-layer ViT feature extraction (layers [3,6,9,12])
    - Learnable per-scale fusion weights (softmax, 12 scalars)
    - ViTDet-style learned scale conversion (ConvTranspose2d/Conv2d, NOT bilinear)
    - Lightweight CNN detail branch for local features
    - Bi-Fusion: concat ViT+CNN features → 1×1 Conv+BN → YOLO channels

Output: P3 (B,256,80,80), P4 (B,256,40,40), P5 (B,512,20,20)
These inject into YOLO26s neck at y[4], y[6], y[10] — identical to ConvNeXt adapter.

Reference: docs/ADAPTER_DESIGN_ViT_YOLO26.md (Rev 3)

Usage:
    adapter = ViTYOLOAdapter(backbone_type="dinov2", imgsz=640)
    outputs = adapter(hidden_states, images)  # returns dict {"p3", "p4", "p5"}
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentry.adapter.vit_feature_extractor import ViTFeatureExtractor
from sentry.adapter.detail_branch import DetailBranch

logger = logging.getLogger(__name__)

# Patch sizes per backbone
_PATCH_SIZES = {"dinov2": 14, "dinov3": 16}

# YOLO26s expected channel layout (scale-specific — do NOT use for n/m/l/x)
_YOLO26S_CHANNELS = [256, 256, 512]  # P3, P4, P5


class MultiLayerFusion(nn.Module):
    """Learnable weighted combination of multi-layer ViT features per target scale.

    For each of 3 target scales (P3, P4, P5), learns a softmax-normalized
    weight vector over the extracted ViT layers. Initialized uniform.

    Params: num_scales × num_layers scalars (e.g., 3×4 = 12).
    """

    def __init__(self, num_layers: int = 4, num_scales: int = 3) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.scale_weights = nn.Parameter(
            torch.ones(num_scales, num_layers) / num_layers
        )

    def forward(self, layer_features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Fuse multi-layer features into per-scale representations.

        Args:
            layer_features: List of L tensors, each (B, D, H_p, W_p)

        Returns:
            List of num_scales tensors, each (B, D, H_p, W_p)
        """
        # Stack: (B, L, D, H_p, W_p)
        stacked = torch.stack(layer_features, dim=1)
        fused = []
        for s in range(self.num_scales):
            w = F.softmax(self.scale_weights[s], dim=0)  # (L,) normalized
            # Weighted sum over layers: (B, D, H_p, W_p)
            fused_s = (stacked * w[None, :, None, None, None]).sum(dim=1)
            fused.append(fused_s)
        return fused


class ViTDetScaleConversion(nn.Module):
    """ViTDet Simple Feature Pyramid — learned scale conversion.

    Replaces bilinear interpolation with learned ConvTranspose2d (upsample)
    and stride-2 Conv2d (downsample). Creates genuinely different representations
    at each scale instead of blurry resized copies.

    From ViTDet paper (Li et al. 2022):
    "We apply a set of convolutions or deconvolutions in parallel
     to produce multi-scale feature maps."

    CRITICAL FIX: Round 3 proved bilinear creates resized copies that fail
    on YOLO PAN neck. ViTDet deconv creates genuine multi-scale features.

    Args:
        in_dim: Input feature dimension from ViT (384 for ViT-S)
    """

    def __init__(self, in_dim: int = 384) -> None:
        super().__init__()
        # P3: upsample 2× via learned transposed convolution
        self.up_p3 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, in_dim, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(in_dim),
        )
        # P4: same scale, 1×1 conv for channel refinement
        self.proj_p4 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_dim),
        )
        # P5: downsample 2× via learned strided convolution
        self.down_p5 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
        )

        total = sum(p.numel() for p in self.parameters())
        logger.info("  ViTDetScaleConversion: %d params (%.1fK)", total, total / 1e3)

    def forward(
        self, fused_maps: list[torch.Tensor], target_sizes: list[tuple[int, int]] | None = None,
    ) -> list[torch.Tensor]:
        """Convert 3 fused maps (all H_p×W_p) to P3/P4/P5 at YOLO strides.

        Args:
            fused_maps: list of 3 tensors, each (B, D, H_p, W_p)
            target_sizes: Optional [(P3_h,P3_w), (P4_h,P4_w), (P5_h,P5_w)].
                If provided, output is resized to exact targets after deconv/conv.
                Needed for DINOv2 where ConvTranspose2d(45→90) ≠ target 80.
                DINOv3 (40→80) is exact and doesn't need this.

        Returns:
            [P3, P4, P5] at correct YOLO strides
        """
        p3 = self.up_p3(fused_maps[0])      # Learned upsample 2×
        p4 = self.proj_p4(fused_maps[1])     # Channel refinement (same spatial)
        p5 = self.down_p5(fused_maps[2])     # Learned downsample 2×

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


class BiFusion(nn.Module):
    """Concatenate ViT semantic features with CNN detail features, project to YOLO channels.

    Per scale: concat(vit_feat, detail_feat) → 1×1 Conv + BN → target channels.
    NO activation — matches ConvNeXt adapter pattern. YOLO neck applies its own.

    Args:
        vit_channels: ViT feature dimension (384 for ViT-S)
        detail_channels: Detail branch output channels (64 default)
        out_channels: Per-scale output channels for YOLO26s [256, 256, 512]
    """

    def __init__(
        self,
        vit_channels: int = 384,
        detail_channels: int = 64,
        out_channels: list[int] | None = None,
    ) -> None:
        super().__init__()
        out_channels = out_channels or _YOLO26S_CHANNELS
        in_ch = vit_channels + detail_channels

        self.projections = nn.ModuleList()
        self.level_names = ["p3", "p4", "p5"]

        for i, out_ch in enumerate(out_channels):
            self.projections.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                )
            )
            logger.info(
                "  BiFusion %s: %d → %d (1×1 Conv + BN, no activation)",
                self.level_names[i], in_ch, out_ch,
            )

    def forward(
        self,
        vit_features: list[torch.Tensor],
        detail_features: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Fuse ViT and detail features, project to YOLO channels."""
        outputs = {}
        for i, (vit_f, det_f, proj, name) in enumerate(
            zip(vit_features, detail_features, self.projections, self.level_names)
        ):
            fused = torch.cat([vit_f, det_f], dim=1)
            outputs[name] = proj(fused)
        return outputs


class ViTYOLOAdapter(nn.Module):
    """Full adapter: ViT features + Detail Branch → YOLO26s P3/P4/P5.

    Architecture:
        Image → ViT (frozen, multi-layer extraction)
              → MultiLayerFusion (12 learnable scalars)
              → Bilinear scale alignment (parameter-free)
        Image → DetailBranch (trainable CNN)
              → BiFusion (concat + 1×1 Conv + BN)
              → P3/P4/P5 for YOLO26s neck injection

    Args:
        backbone_type: "dinov2" or "dinov3"
        imgsz: Input image size (default 640, must be multiple of 32)
        layer_indices: ViT layers to extract (default [3,6,9,12])
        vit_dim: ViT hidden dimension (384 for ViT-S)
        detail_channels: Detail branch output channels (64)
        out_channels: YOLO26s target channels [256, 256, 512]
        device: Target device
    """

    def __init__(
        self,
        backbone_type: Literal["dinov2", "dinov3"] = "dinov2",
        imgsz: int = 640,
        layer_indices: list[int] | None = None,
        vit_dim: int = 384,
        detail_channels: int = 64,
        out_channels: list[int] | None = None,
        device: str = "cuda",
    ) -> None:
        super().__init__()

        if imgsz % 32 != 0:
            raise ValueError(
                f"imgsz={imgsz} must be a multiple of 32 for YOLO P5 stride alignment. "
                f"TAL small-box expansion depends on correct stride grids."
            )

        self.backbone_type = backbone_type
        self.imgsz = imgsz
        self.vit_dim = vit_dim
        out_channels = out_channels or _YOLO26S_CHANNELS

        patch_size = _PATCH_SIZES[backbone_type]
        self.H_p, self.W_p = ViTFeatureExtractor.compute_grid_size(imgsz, patch_size)

        # Target spatial sizes for YOLO26s at this imgsz
        self.target_sizes = [
            (imgsz // 8, imgsz // 8),    # P3: 80×80 for imgsz=640
            (imgsz // 16, imgsz // 16),  # P4: 40×40
            (imgsz // 32, imgsz // 32),  # P5: 20×20
        ]

        # Component A: Multi-layer ViT feature extraction
        self.feature_extractor = ViTFeatureExtractor(
            backbone_type=backbone_type,
            layer_indices=layer_indices or [3, 6, 9, 12],
            feature_dim=vit_dim,
        )

        # Component B: Per-scale learnable fusion
        num_layers = len(self.feature_extractor.layer_indices)
        self.fusion = MultiLayerFusion(num_layers=num_layers, num_scales=3)

        # Component C: ViTDet-style learned scale conversion (NOT bilinear)
        self.scale_conversion = ViTDetScaleConversion(in_dim=vit_dim)

        # Component D: Lightweight CNN detail branch
        self.detail_branch = DetailBranch(out_channels=detail_channels)

        # Component E: Bi-Fusion (concat + project)
        self.bi_fusion = BiFusion(
            vit_channels=vit_dim,
            detail_channels=detail_channels,
            out_channels=out_channels,
        )

        # Log parameter counts
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "ViTYOLOAdapter: backbone=%s, imgsz=%d, grid=%dx%d, "
            "total_params=%d (%.1fK), trainable=%d (%.1fK)",
            backbone_type, imgsz, self.H_p, self.W_p,
            total, total / 1e3, trainable, trainable / 1e3,
        )

    def forward(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        images: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the full adapter pipeline.

        Args:
            hidden_states: ViT hidden states from model(output_hidden_states=True).
                Tuple of (num_layers+1) tensors, each (B, total_tokens, D).
            images: Input image tensor (B, 3, imgsz, imgsz). SAME tensor fed to ViT.
                Used by detail branch for local features.

        Returns:
            Dict with keys "p3", "p4", "p5":
                p3: (B, 256, imgsz/8, imgsz/8)   e.g. (B, 256, 80, 80)
                p4: (B, 256, imgsz/16, imgsz/16)  e.g. (B, 256, 40, 40)
                p5: (B, 512, imgsz/32, imgsz/32)  e.g. (B, 512, 20, 20)
        """
        # A: Extract multi-layer features from ViT
        layer_features = self.feature_extractor(hidden_states, self.H_p, self.W_p)
        # layer_features: list of 4 tensors, each (B, D, H_p, W_p)

        # B: Fuse layers per target scale
        fused = self.fusion(layer_features)
        # fused: list of 3 tensors, each (B, D, H_p, W_p)

        # C: ViTDet-style learned scale conversion (NOT bilinear)
        # Creates genuinely different representations at each scale via
        # learned ConvTranspose2d (upsample) and stride-2 Conv2d (downsample)
        # For DINOv2 (45×45): deconv gives 90×90, need resize to 80×80
        # For DINOv3 (40×40): deconv gives exact 80×80, no resize needed
        aligned = self.scale_conversion(fused, target_sizes=self.target_sizes)

        # D: Detail branch — lightweight CNN on same input image
        d3, d4, d5 = self.detail_branch(images)

        # E: Bi-Fusion — concat ViT+CNN, project to YOLO channels
        outputs = self.bi_fusion(aligned, (d3, d4, d5))

        return outputs

    @staticmethod
    def for_dinov2_to_yolo26s(
        imgsz: int = 640, device: str = "cuda",
    ) -> "ViTYOLOAdapter":
        """Factory: creates adapter for DINOv2 ViT-S → YOLO26s."""
        adapter = ViTYOLOAdapter(
            backbone_type="dinov2",
            imgsz=imgsz,
            vit_dim=384,
            out_channels=_YOLO26S_CHANNELS,
        )
        return adapter.to(device)

    @staticmethod
    def for_dinov3_to_yolo26s(
        imgsz: int = 640, device: str = "cuda",
    ) -> "ViTYOLOAdapter":
        """Factory: creates adapter for DINOv3 ViT-S → YOLO26s."""
        adapter = ViTYOLOAdapter(
            backbone_type="dinov3",
            imgsz=imgsz,
            vit_dim=384,
            out_channels=_YOLO26S_CHANNELS,
        )
        return adapter.to(device)
