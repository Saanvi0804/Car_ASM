"""Multi-layer ViT feature extractor for DINOv2 and DINOv3.

Extracts intermediate features from ViT transformer blocks,
drops CLS and register tokens, and reshapes flat tokens to 2D spatial grids.

Supports:
    - DINOv2 ViT-S/14: 1 CLS token, no registers → skip=1
    - DINOv3 ViT-S/16: 1 CLS + 4 register tokens → skip=5

Usage:
    extractor = ViTFeatureExtractor(backbone_type="dinov2", layer_indices=[3, 6, 9, 12])
    features = extractor(model_output, H_p=45, W_p=45)
    # features: list of 4 tensors, each (B, D, H_p, W_p)
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Token skip counts per backbone type
_SKIP_TOKENS = {
    "dinov2": 1,   # 1 CLS token only
    "dinov3": 5,   # 1 CLS + 4 register tokens (Darcet et al.)
}


class ViTFeatureExtractor(nn.Module):
    """Extract and reshape intermediate ViT features for multi-scale detection.

    Takes HuggingFace model output (with output_hidden_states=True) and:
    1. Selects features from specified transformer block indices
    2. Drops CLS and register tokens (backbone-specific skip count)
    3. Reshapes flat (B, N, D) tokens to spatial (B, D, H_p, W_p) grids

    Args:
        backbone_type: "dinov2" (skip=1) or "dinov3" (skip=5)
        layer_indices: Which transformer block outputs to extract.
            For 12-block ViT-S: [3, 6, 9, 12] extracts every 3rd block.
            HF hidden_states[0] = embeddings, hidden_states[i] = after block i.
        feature_dim: ViT hidden dimension (384 for ViT-S, 768 for ViT-B)
    """

    def __init__(
        self,
        backbone_type: Literal["dinov2", "dinov3"] = "dinov2",
        layer_indices: list[int] | None = None,
        feature_dim: int = 384,
    ) -> None:
        super().__init__()

        if backbone_type not in _SKIP_TOKENS:
            raise ValueError(
                f"Unknown backbone_type '{backbone_type}'. "
                f"Supported: {list(_SKIP_TOKENS.keys())}"
            )

        self.backbone_type = backbone_type
        self.skip_tokens = _SKIP_TOKENS[backbone_type]
        self.layer_indices = layer_indices or [3, 6, 9, 12]
        self.feature_dim = feature_dim

        logger.info(
            "ViTFeatureExtractor: backbone=%s, skip=%d tokens, layers=%s, dim=%d",
            backbone_type, self.skip_tokens, self.layer_indices, feature_dim,
        )

    def forward(
        self, hidden_states: tuple[torch.Tensor, ...], H_p: int, W_p: int,
    ) -> list[torch.Tensor]:
        """Extract and reshape intermediate features.

        Args:
            hidden_states: Tuple of tensors from model(output_hidden_states=True).
                hidden_states[0] = embedding output
                hidden_states[i] = output of transformer block i (1-indexed)
                Each tensor: (B, total_tokens, D) where total_tokens = skip + H_p*W_p
            H_p: Expected patch grid height (e.g., 45 for DINOv2@640, 40 for DINOv3@640)
            W_p: Expected patch grid width

        Returns:
            List of tensors, each (B, D, H_p, W_p), one per layer_index.
        """
        features = []
        expected_patches = H_p * W_p

        for idx in self.layer_indices:
            if idx >= len(hidden_states):
                raise IndexError(
                    f"Layer index {idx} out of range. Model has {len(hidden_states)} "
                    f"hidden states (0=embeddings, 1..{len(hidden_states)-1}=blocks)."
                )

            # Get block output and drop CLS / register tokens
            block_output = hidden_states[idx]  # (B, skip + N, D)
            tokens = block_output[:, self.skip_tokens:, :]  # (B, N, D)

            B, N, D = tokens.shape
            if N != expected_patches:
                raise ValueError(
                    f"Token count mismatch at layer {idx}: got {N} patches, "
                    f"expected {expected_patches} (H_p={H_p} × W_p={W_p}). "
                    f"Skipped {self.skip_tokens} special tokens from {block_output.shape[1]} total. "
                    f"Check input resolution and backbone patch size."
                )

            # Reshape to spatial grid: (B, N, D) → (B, D, H_p, W_p)
            feat = tokens.reshape(B, H_p, W_p, D).permute(0, 3, 1, 2).contiguous()
            features.append(feat)

        return features

    @staticmethod
    def compute_grid_size(imgsz: int, patch_size: int) -> tuple[int, int]:
        """Compute the patch grid size for a given input resolution.

        Args:
            imgsz: Input image size (assumes square)
            patch_size: ViT patch size (14 for DINOv2, 16 for DINOv3)

        Returns:
            (H_p, W_p) — patch grid dimensions

        Note:
            DINOv2@640: 640//14 = 45 (10px silently truncated)
            DINOv3@640: 640//16 = 40 (exact)
        """
        grid = imgsz // patch_size
        return grid, grid
