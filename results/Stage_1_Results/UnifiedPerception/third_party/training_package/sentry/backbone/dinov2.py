"""DINOv2 ViT-S/14 backbone loaded via torch.hub."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from sentry.backbone.base import BaseBackbone

logger = logging.getLogger(__name__)

_FEATURE_DIM = 384  # ViT-Small embedding dimension


class DINOv2Backbone(BaseBackbone):
    """DINOv2 ViT-S/14 feature extractor.

    Loads the pretrained model from ``facebookresearch/dinov2`` via
    :func:`torch.hub.load` and returns both the [CLS] token and
    spatial patch tokens.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading DINOv2 ViT-S/14 (device=%s)", device)

        self.model: nn.Module = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            pretrained=True,
        )
        self.model.to(device).eval()

        # Freeze backbone weights by default
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract CLS and patch tokens from *images*.

        Returns:
            {"cls_token": (B, 384), "patch_tokens": (B, N, 384)}
        """
        # get_intermediate_layers with n=1 returns a list with one tensor
        # of shape (B, 1+N, D) when reshape=False.  We split CLS vs patches.
        outputs = self.model.get_intermediate_layers(
            images,
            n=1,
            reshape=False,
            return_class_token=True,
        )
        # outputs is a list of (patch_tokens, cls_token) tuples
        patch_tokens, cls_token = outputs[0]

        return {
            "cls_token": cls_token,          # (B, 384)
            "patch_tokens": patch_tokens,    # (B, N, 384)
        }

    def get_feature_dim(self) -> int:
        return _FEATURE_DIM
