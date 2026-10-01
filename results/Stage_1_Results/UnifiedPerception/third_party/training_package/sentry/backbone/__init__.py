"""Backbone registry and factory.

Usage::

    from sentry.backbone import build_backbone

    backbone = build_backbone({"name": "dinov2_vits14"})
"""

from __future__ import annotations

from typing import Type

from sentry.backbone.base import BaseBackbone
from sentry.backbone.dinov2 import DINOv2Backbone
from sentry.backbone.dinov3 import DINOv3Backbone
from sentry.backbone.radio import RADIOBackbone

BACKBONE_REGISTRY: dict[str, Type[BaseBackbone]] = {
    "dinov2_vits14": DINOv2Backbone,
    "dinov3_vits16": DINOv3Backbone,
    "dinov3_convnext_tiny": DINOv3Backbone,
    "radio_b": RADIOBackbone,
    "radio_l": RADIOBackbone,
    "radio_h": RADIOBackbone,
    "e_radio": RADIOBackbone,
}


def build_backbone(config: dict) -> BaseBackbone:
    """Instantiate a backbone from a config dict.

    Args:
        config: Must contain a ``"name"`` key that maps to a registered
            backbone class.  The entire dict is forwarded to the constructor.

    Returns:
        An initialised :class:`BaseBackbone` subclass.

    Raises:
        ValueError: If ``config["name"]`` is not in the registry.
    """
    name: str = config["name"]
    if name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'. "
            f"Available: {list(BACKBONE_REGISTRY.keys())}"
        )
    cls = BACKBONE_REGISTRY[name]
    return cls(config)
