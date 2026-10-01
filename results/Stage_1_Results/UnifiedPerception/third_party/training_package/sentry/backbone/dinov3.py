"""DINOv3 backbone variants loaded via HuggingFace transformers."""

from __future__ import annotations

import logging
from typing import Any

import torch

from sentry.backbone.base import BaseBackbone

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModel
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


_MODEL_MAP: dict[str, dict[str, Any]] = {
    "dinov3_vits16": {
        "hf_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "arch": "vit",
        "feature_dim": 384,
    },
    "dinov3_convnext_tiny": {
        "hf_name": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        "arch": "convnext",
        "feature_dim": 768,  # last stage channel count for ConvNeXt-Tiny
    },
}


class DINOv3Backbone(BaseBackbone):
    """DINOv3 feature extractor supporting ViT and ConvNeXt variants.

    The variant is selected via ``config["name"]``:
        - ``"dinov3_vits16"``  -- ViT-S/16, returns cls + patch tokens
        - ``"dinov3_convnext_tiny"`` -- ConvNeXt-Tiny, returns multi-scale feature maps
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "The `transformers` package is required for DINOv3 backbones. "
                "Install it with: pip install transformers"
            )

        name: str = config["name"]
        if name not in _MODEL_MAP:
            raise ValueError(
                f"Unknown DINOv3 variant '{name}'. "
                f"Supported: {list(_MODEL_MAP.keys())}"
            )

        meta = _MODEL_MAP[name]
        self._arch: str = meta["arch"]
        self._feature_dim: int = meta["feature_dim"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading DINOv3 '%s' from '%s' (device=%s)", name, meta["hf_name"], device)

        self.model = AutoModel.from_pretrained(meta["hf_name"])
        self.model.to(device).eval()

        for param in self.model.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if self._arch == "vit":
            return self._forward_vit(images)
        return self._forward_convnext(images)

    def _forward_vit(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """ViT variant: returns cls_token and patch_tokens.

        Returns:
            {"cls_token": (B, D), "patch_tokens": (B, N, D)}
        """
        outputs = self.model(pixel_values=images, output_hidden_states=False)
        last_hidden = outputs.last_hidden_state  # (B, 1+N, D)
        cls_token = last_hidden[:, 0]             # (B, D)
        patch_tokens = last_hidden[:, 1:]         # (B, N, D)
        return {"cls_token": cls_token, "patch_tokens": patch_tokens}

    def _forward_convnext(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """ConvNeXt variant: returns multi-scale feature maps.

        Returns:
            {"feature_maps": {"stage1": (B,C1,H1,W1), "stage2": ..., ...}}
        """
        outputs = self.model(pixel_values=images, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of tensors per stage
        feature_maps: dict[str, torch.Tensor] = {
            f"stage{i + 1}": feat
            for i, feat in enumerate(hidden_states)
        }
        return {"feature_maps": feature_maps}

    def get_feature_dim(self) -> int:
        return self._feature_dim
