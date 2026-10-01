"""
heads — registry of perception heads for the standalone UnifiedPerception package.

This package ships ONLY two heads:
    - detection : Pradipt's Adapter-v2 + YOLO26 object detector (unchanged).
    - action    : the whole-scene DINOv2-CLS action recognizer (a K400 linear
                  probe over the shared CLS tokens).

The per-person action head (action_perperson) is intentionally NOT included.

To add a new head:
    1. Drop `heads/<name>.py` implementing the Head protocol
       (see perception_types.Head). It MUST export a top-level `HEAD` instance.
    2. Add the import + registry entry below.
"""
from __future__ import annotations

from .detection import HEAD as DETECTION_HEAD
from .action import HEAD as ACTION_HEAD


# Order matters — heads that depend on prior head output must be registered
# AFTER their dependency. detection runs first so downstream consumers (and the
# combined overlay) can use its boxes; action is whole-scene and independent.
REGISTRY = {
    "detection": DETECTION_HEAD,
    "action":    ACTION_HEAD,
}


def get(name: str):
    """Look up a head by name."""
    if name not in REGISTRY:
        raise KeyError(f"no such head: {name!r} (available: {list(REGISTRY)})")
    return REGISTRY[name]


def list_heads() -> list[str]:
    return list(REGISTRY)
