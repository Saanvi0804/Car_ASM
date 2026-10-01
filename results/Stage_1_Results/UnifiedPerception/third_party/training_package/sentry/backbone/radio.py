"""NVIDIA RADIO backbone loaded via torch.hub (NVlabs/RADIO)."""

from __future__ import annotations

import logging
from typing import Any

import torch

from sentry.backbone.base import BaseBackbone

logger = logging.getLogger(__name__)

_MODEL_MAP: dict[str, dict[str, Any]] = {
    "radio_b": {
        "hf_name": "nvidia/C-RADIOv3-B",
        "version": "c-radio_v3-b",
        "arch": "vit",
        "feature_dim": 768,
        "params_m": 98,
    },
    "radio_l": {
        "hf_name": "nvidia/C-RADIOv3-L",
        "version": "c-radio_v3-l",
        "arch": "vit",
        "feature_dim": 1024,
        "params_m": 320,
    },
    "radio_h": {
        "hf_name": "nvidia/C-RADIOv4-H",
        "version": "c-radio_v4-h",
        "arch": "vit",
        "feature_dim": 1280,
        "params_m": 631,
    },
    "e_radio": {
        "hf_name": "nvidia/E-RADIO",
        "version": "e-radio_v2",
        "arch": "hybrid",
        "feature_dim": 1280,
        "params_m": 400,
    },
}


class RADIOBackbone(BaseBackbone):
    """NVIDIA RADIO (AM-RADIO) vision foundation model.

    Loads C-RADIO (commercial license) or E-RADIO (efficient hybrid)
    via torch.hub from NVlabs/RADIO.

    Supported variants via config["name"]:
        - "radio_b"  -- C-RADIOv3-B (98M params, ViT-B/16)
        - "radio_l"  -- C-RADIOv3-L (320M params, ViT-L/16)
        - "radio_h"  -- C-RADIOv4-H (631M params, ViT-H/16)
        - "e_radio"  -- E-RADIO (400M params, hybrid CNN-ViT)
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        name: str = config["name"]
        if name not in _MODEL_MAP:
            raise ValueError(
                f"Unknown RADIO variant '{name}'. "
                f"Supported: {list(_MODEL_MAP.keys())}"
            )

        meta = _MODEL_MAP[name]
        self._arch: str = meta["arch"]
        self._feature_dim: int = meta["feature_dim"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Loading RADIO '%s' from '%s' (~%dM params, device=%s)",
            name, meta["hf_name"], meta["params_m"], device,
        )

        self.model = torch.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version=meta["version"],
            progress=True,
        )
        self.model.to(device).eval()

        for param in self.model.parameters():
            param.requires_grad = False

        logger.info("RADIO model loaded successfully")

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract features from images.

        RADIO returns a tuple: (summary, spatial)
            - summary: (B, C) global feature vector
            - spatial: (B, T, D) spatial token features

        Returns:
            {"cls_token": (B, D), "patch_tokens": (B, T, D)}
        """
        summary, spatial = self.model(images)
        return {
            "cls_token": summary,
            "patch_tokens": spatial,
        }

    def get_feature_dim(self) -> int:
        return self._feature_dim
