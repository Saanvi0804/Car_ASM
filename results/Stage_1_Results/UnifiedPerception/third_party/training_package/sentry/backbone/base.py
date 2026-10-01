"""Abstract base class for all backbone feature extractors."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseBackbone(ABC, nn.Module):
    """Base interface that every backbone must implement.

    Subclasses extract features from raw images and expose them as
    a dict of named tensors (cls_token, patch_tokens, feature_maps, etc.).
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the backbone on a batch of images.

        Args:
            images: (B, C, H, W) tensor, typically ImageNet-normalised.

        Returns:
            For ViT backbones:
                {"cls_token": (B, D), "patch_tokens": (B, N, D)}
            For CNN backbones:
                {"feature_maps": {"stage1": tensor, "stage2": tensor, ...}}
        """

    @abstractmethod
    def get_feature_dim(self) -> int:
        """Return the primary feature dimensionality of this backbone."""

    # ------------------------------------------------------------------
    # Concrete verification helper -- not abstract, works for any child.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def verify(self, device: str = "cuda") -> None:
        """Load the model onto *device*, run a dummy forward pass, and log shapes."""
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA unavailable, falling back to CPU for verification")
            device = "cpu"

        self.to(device).eval()

        dummy = torch.randn(1, 3, 224, 224, device=device)
        outputs = self.forward(dummy)

        logger.info("Backbone verification on device=%s", device)
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                logger.info("  %-15s shape=%s  dtype=%s", key, tuple(value.shape), value.dtype)
            elif isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    logger.info(
                        "  %s/%-10s shape=%s  dtype=%s",
                        key,
                        sub_key,
                        tuple(sub_val.shape),
                        sub_val.dtype,
                    )
        logger.info("  feature_dim=%d", self.get_feature_dim())
