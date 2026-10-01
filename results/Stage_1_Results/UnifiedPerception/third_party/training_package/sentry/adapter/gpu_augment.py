"""GPU-based augmentation and dataset utilities for v2 training.

Components:
    - gpu_hsv_jitter: HSV color jitter on GPU tensors (replaces cv2-based CPU version)
    - gpu_horizontal_flip: Batch horizontal flip with label adjustment on GPU
    - MosaicDataset: Dataset wrapper that stitches 4 images into 1 tile
    - collate_fn_v2: Collate with optional multi-scale random resize per batch

When --gpu-augment is disabled, the dataset's CPU augmentation is used instead
(v1 behavior). When --no-mosaic / --no-multi-scale, those features are skipped.

Does NOT require kornia or torchvision.transforms.v2 — uses pure torch ops.

Usage:
    from sentry.adapter.gpu_augment import gpu_hsv_jitter, gpu_horizontal_flip
    from sentry.adapter.gpu_augment import MosaicDataset, collate_fn_v2
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ─── GPU Augmentation Functions ───


def _rgb_to_hsv(img: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor to HSV. Input/output: (B, 3, H, W) float in [0, 1]."""
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    maxc = img.max(dim=1).values
    minc = img.min(dim=1).values
    diff = maxc - minc + 1e-8

    # Hue [0, 1]
    h = torch.zeros_like(maxc)
    mask_r = (maxc == r) & (diff > 1e-7)
    mask_g = (maxc == g) & (diff > 1e-7) & ~mask_r
    mask_b = ~mask_r & ~mask_g & (diff > 1e-7)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6.0 / 6.0
    h[mask_g] = ((b[mask_g] - r[mask_g]) / diff[mask_g] + 2.0) / 6.0
    h[mask_b] = ((r[mask_b] - g[mask_b]) / diff[mask_b] + 4.0) / 6.0

    # Saturation [0, 1]
    s = torch.where(maxc > 1e-7, diff / (maxc + 1e-8), torch.zeros_like(maxc))

    # Value [0, 1]
    v = maxc

    return torch.stack([h, s, v], dim=1)


def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """Convert HSV tensor to RGB. Input/output: (B, 3, H, W) float in [0, 1]."""
    h, s, v = hsv[:, 0] * 6.0, hsv[:, 1], hsv[:, 2]  # h in [0, 6)
    i = h.long() % 6
    f = h - h.long().float()
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    # Build RGB based on hue sector
    r = torch.where(i == 0, v, torch.where(i == 1, q, torch.where(i == 2, p,
          torch.where(i == 3, p, torch.where(i == 4, t, v)))))
    g = torch.where(i == 0, t, torch.where(i == 1, v, torch.where(i == 2, v,
          torch.where(i == 3, q, torch.where(i == 4, p, p)))))
    b = torch.where(i == 0, p, torch.where(i == 1, p, torch.where(i == 2, t,
          torch.where(i == 3, v, torch.where(i == 4, v, q)))))

    return torch.stack([r, g, b], dim=1).clamp(0.0, 1.0)


@torch.no_grad()
def gpu_hsv_jitter(
    images: torch.Tensor,
    h_gain: float = 0.015,
    s_gain: float = 0.7,
    v_gain: float = 0.4,
) -> torch.Tensor:
    """Apply HSV color jitter on GPU tensors.

    Args:
        images: (B, 3, H, W) float32 in [0, 1] range, RGB order
        h_gain, s_gain, v_gain: Jitter ranges (matches Ultralytics defaults)

    Returns:
        Augmented images, same shape and range.
    """
    B = images.shape[0]
    device = images.device

    # Random gains per image
    r = torch.rand(B, 3, device=device) * 2.0 - 1.0  # [-1, 1]
    gains = r * torch.tensor([h_gain, s_gain, v_gain], device=device) + 1.0  # [1-gain, 1+gain]

    hsv = _rgb_to_hsv(images)
    # Apply gains: h modulo 1.0, s and v clamped to [0, 1]
    hsv[:, 0] = (hsv[:, 0] * gains[:, 0].view(B, 1, 1)) % 1.0
    hsv[:, 1] = (hsv[:, 1] * gains[:, 1].view(B, 1, 1)).clamp(0.0, 1.0)
    hsv[:, 2] = (hsv[:, 2] * gains[:, 2].view(B, 1, 1)).clamp(0.0, 1.0)

    return _hsv_to_rgb(hsv)


@torch.no_grad()
def gpu_horizontal_flip(
    images: torch.Tensor,
    targets: torch.Tensor,
    p: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random horizontal flip on GPU with label adjustment.

    Args:
        images: (B, 3, H, W) float32
        targets: (N, 6) [batch_idx, class, cx, cy, w, h] normalized 0-1
        p: Flip probability per image

    Returns:
        (flipped_images, adjusted_targets)
    """
    B = images.shape[0]
    device = images.device

    # Generate per-image flip mask
    flip_mask = torch.rand(B, device=device) < p  # (B,) bool

    # Flip selected images
    if flip_mask.any():
        images = images.clone()
        images[flip_mask] = images[flip_mask].flip(-1)  # flip W dimension

        # Adjust targets: flip cx = 1.0 - cx for flipped images
        if targets.shape[0] > 0:
            targets = targets.clone()
            for i in range(B):
                if flip_mask[i]:
                    mask = targets[:, 0] == i
                    targets[mask, 2] = 1.0 - targets[mask, 2]  # cx = 1 - cx

    return images, targets


# ─── Mosaic Dataset Wrapper ───


class MosaicDataset(Dataset):
    """Dataset wrapper that stitches 4 random images into one mosaic tile.

    Each __getitem__ call samples 4 images from the base dataset, places them
    in a 2×2 grid with random center point, and merges their labels.

    Args:
        base_dataset: YOLODetectionDataset instance
        imgsz: Target output image size (default 640)

    Usage:
        mosaic = MosaicDataset(base_dataset, imgsz=640)
        mosaic.set_enabled(True)   # mosaic ON
        mosaic.set_enabled(False)  # passthrough to base dataset (last 10 epochs)
    """

    def __init__(self, base_dataset: Dataset, imgsz: int = 640) -> None:
        self.base = base_dataset
        self.imgsz = imgsz
        self.enabled = True
        logger.info("MosaicDataset: wrapping %d images, imgsz=%d", len(base_dataset), imgsz)

    def __len__(self) -> int:
        return len(self.base)

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable mosaic. When disabled, delegates to base dataset."""
        self.enabled = enabled

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled:
            return self.base[idx]

        s = self.imgsz
        # Random center point for the mosaic
        cx = random.randint(s // 4, 3 * s // 4)
        cy = random.randint(s // 4, 3 * s // 4)

        # Sample 4 indices (include current + 3 random)
        indices = [idx] + [random.randint(0, len(self.base) - 1) for _ in range(3)]

        # Create mosaic canvas
        mosaic_img = torch.full((3, s, s), 114.0 / 255.0)  # gray fill
        all_labels = []

        # Quadrant placement: top-left, top-right, bottom-left, bottom-right
        placements = [
            (0, 0, cx, cy),        # top-left
            (cx, 0, s, cy),        # top-right
            (0, cy, cx, s),        # bottom-left
            (cx, cy, s, s),        # bottom-right
        ]

        for i, (x1, y1, x2, y2) in enumerate(placements):
            img, labels = self.base[indices[i]]
            # img: (3, H, W), labels: (N, 5) [class, cx, cy, w, h]

            qw, qh = x2 - x1, y2 - y1
            if qw <= 0 or qh <= 0:
                continue

            # Resize image to fit quadrant
            img_resized = F.interpolate(
                img.unsqueeze(0), size=(qh, qw), mode="bilinear", align_corners=False,
            ).squeeze(0)

            # Place in mosaic
            mosaic_img[:, y1:y2, x1:x2] = img_resized

            # Adjust labels to mosaic coordinates (normalized 0-1)
            if labels.shape[0] > 0:
                adj_labels = labels.clone()
                # Transform from image-local normalized to mosaic-global normalized
                adj_labels[:, 1] = (labels[:, 1] * qw + x1) / s  # cx
                adj_labels[:, 2] = (labels[:, 2] * qh + y1) / s  # cy
                adj_labels[:, 3] = labels[:, 3] * qw / s          # w
                adj_labels[:, 4] = labels[:, 4] * qh / s          # h

                # Filter: keep boxes with center inside [0, 1] and minimum size
                cx_ok = (adj_labels[:, 1] > 0.005) & (adj_labels[:, 1] < 0.995)
                cy_ok = (adj_labels[:, 2] > 0.005) & (adj_labels[:, 2] < 0.995)
                size_ok = (adj_labels[:, 3] > 0.005) & (adj_labels[:, 4] > 0.005)
                valid = cx_ok & cy_ok & size_ok
                if valid.any():
                    all_labels.append(adj_labels[valid])

        if all_labels:
            merged_labels = torch.cat(all_labels, dim=0)
        else:
            merged_labels = torch.zeros((0, 5), dtype=torch.float32)

        return mosaic_img, merged_labels


# ─── Multi-Scale Collate Function ───


def collate_fn_v2(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    multi_scale: bool = False,
    scale_choices: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """V2 collate with optional multi-scale resize.

    Args:
        batch: List of (image, labels) from dataset
        multi_scale: If True, randomly resize all images to one of scale_choices
        scale_choices: List of sizes (must be multiples of 32).
            Default: [480, 512, 544, 576, 608, 640]

    Returns:
        (images, targets) where targets has batch_idx prepended
    """
    if scale_choices is None:
        scale_choices = [480, 512, 544, 576, 608, 640]

    images = torch.stack([b[0] for b in batch])

    # Multi-scale: randomly resize entire batch to one size
    if multi_scale:
        target_sz = random.choice(scale_choices)
        if target_sz != images.shape[-1]:
            images = F.interpolate(images, size=(target_sz, target_sz), mode="bilinear", align_corners=False)

    # Build targets with batch index (same as v1 collate)
    labels_list = []
    for i, (_, labels) in enumerate(batch):
        if labels.shape[0] > 0:
            batch_idx = torch.full((labels.shape[0], 1), i, dtype=torch.float32)
            labels_list.append(torch.cat([batch_idx, labels], dim=1))

    if labels_list:
        targets = torch.cat(labels_list, dim=0)
    else:
        targets = torch.zeros((0, 6), dtype=torch.float32)

    return images, targets
