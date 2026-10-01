"""Feature Alignment Loss — Knowledge Distillation for adapter training.

Computes MSE between adapter output features (P3/P4/P5) and teacher
YOLO26s backbone features, providing direct gradient to the adapter
without flowing through the frozen YOLO neck+head.

Teacher: fisheye-tuned YOLO26s (best.pt from baseline training).
Fixed throughout training — never updated, always torch.no_grad().

Combined loss:
    L_total = L_detect + alpha * L_align
    alpha = max(1.0 - epoch/total_epochs, 0.1)  # progressive decay

Early training: alignment dominates → fast convergence on feature distribution
Late training: detection dominates → fine-tunes for actual detection quality

Reference: docs/ADAPTER_GUIDE.md Section 9, docs/ADAPTER_DESIGN_ViT_YOLO26.md Section 4.1

Usage:
    teacher = TeacherFeatureExtractor("runs/baseline/.../best.pt", device="cuda")
    loss_fn = FeatureAlignmentLoss()

    # In training loop:
    with torch.no_grad():
        teacher_features = teacher(images)  # {P3, P4, P5}
    L_align = loss_fn(adapter_output, teacher_features)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# YOLO26s backbone save indices where P3/P4/P5 are produced
_P3_IDX = 4
_P4_IDX = 6
_P5_IDX = 10
_NECK_START = 11


class TeacherFeatureExtractor(nn.Module):
    """Extracts P3/P4/P5 backbone features from a native YOLO26s model.

    Runs the YOLO backbone layers (0-10) only, returning the feature maps
    at the save indices that the YOLO neck reads from.

    The teacher is FROZEN and runs under torch.no_grad() during training.
    BN layers are in eval mode (use running statistics).

    Args:
        yolo_model_path: Path to YOLO26s weights (e.g., fisheye-tuned best.pt)
        device: Target device
    """

    def __init__(self, yolo_model_path: str, device: str = "cuda") -> None:
        super().__init__()
        from ultralytics import YOLO

        logger.info("Loading teacher YOLO26s from %s", yolo_model_path)
        yolo = YOLO(yolo_model_path)
        self._model = yolo.model.to(device)
        self._layers = self._model.model
        self._save_indices = self._model.save

        # Freeze everything and set to eval (BN uses running stats)
        for p in self._model.parameters():
            p.requires_grad = False
        self._model.eval()

        logger.info(
            "Teacher loaded: %d layers, save_indices=%s",
            len(self._layers), sorted(self._save_indices),
        )

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract backbone P3/P4/P5 features from images.

        Args:
            images: (B, 3, H, W) normalized images (same as fed to ViT).
                NOTE: YOLO expects [0,1] range images, NOT ImageNet-normalized.
                The caller must provide images in the correct range for YOLO.

        Returns:
            Dict with "P3", "P4", "P5" tensors at the teacher's backbone resolution.
        """
        y = [None] * len(self._layers)
        x = images

        # Run backbone layers only (0 to NECK_START-1)
        for i in range(_NECK_START):
            m = self._layers[i]
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            if i in self._save_indices:
                y[i] = x

        return {
            "P3": y[_P3_IDX],   # (B, 256, 80, 80) for YOLO26s
            "P4": y[_P4_IDX],   # (B, 256, 40, 40)
            "P5": y[_P5_IDX],   # (B, 512, 20, 20)
        }


class FeatureAlignmentLoss(nn.Module):
    """MSE loss between adapter output and teacher backbone features.

    Uses F.mse_loss with mean reduction — naturally normalizes per spatial size
    and per channel count. No additional per-scale weighting initially.

    If P5 gradient dominates (512ch vs 256ch for P3/P4), add 0.5× weight on P5.

    Args:
        p5_weight: Weight for P5 MSE term (default 1.0, reduce to 0.5 if needed)
    """

    def __init__(self, p5_weight: float = 1.0) -> None:
        super().__init__()
        self.p5_weight = p5_weight

    def forward(
        self,
        adapter_output: dict[str, torch.Tensor],
        teacher_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute feature alignment loss.

        Args:
            adapter_output: {"p3": (B,256,80,80), "p4": (B,256,40,40), "p5": (B,512,20,20)}
            teacher_features: {"P3": (B,256,80,80), "P4": (B,256,40,40), "P5": (B,512,20,20)}

        Returns:
            Scalar MSE loss tensor with gradient.
        """
        loss_p3 = F.mse_loss(adapter_output["p3"], teacher_features["P3"])
        loss_p4 = F.mse_loss(adapter_output["p4"], teacher_features["P4"])
        loss_p5 = F.mse_loss(adapter_output["p5"], teacher_features["P5"])

        return loss_p3 + loss_p4 + self.p5_weight * loss_p5


def compute_alpha(epoch: int, total_epochs: int, min_alpha: float = 0.1) -> float:
    """Progressive decay schedule for feature alignment weight.

    Early: alpha ≈ 1.0 → alignment dominates (fast convergence)
    Late:  alpha → min_alpha → detection loss dominates (fine-tuning)

    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total training epochs
        min_alpha: Minimum alpha (never fully removes alignment signal)

    Returns:
        Float in [min_alpha, 1.0]
    """
    return max(1.0 - epoch / total_epochs, min_alpha)
