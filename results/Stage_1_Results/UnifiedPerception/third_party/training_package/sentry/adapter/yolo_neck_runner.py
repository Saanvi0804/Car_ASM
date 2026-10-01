"""Run YOLO26 neck+head using externally provided backbone features.

Injects adapted FPN features (P3, P4, P5) into the YOLO26 save list
at the correct indices, then runs only the neck (layers 11-22) and
detect head (layer 23), skipping the YOLO backbone entirely.

YOLO26s forward dataflow:
    Backbone (layers 0-10):
        Layer  4 (C3k2)  → saved → P3 (256, 80, 80)
        Layer  6 (C3k2)  → saved → P4 (256, 40, 40)
        Layer 10 (C2PSA) → saved → P5 (512, 20, 20)

    Neck (layers 11-22):
        Layer 11: Upsample(P5)
        Layer 12: Concat(↑, y[6]=P4)
        Layer 13: C3k2 → saved
        Layer 14: Upsample
        Layer 15: Concat(↑, y[4]=P3)
        Layer 16: C3k2 → saved → feeds Detect
        Layer 17: Conv (downsample)
        Layer 18: Concat(↑, y[13])
        Layer 19: C3k2 → saved → feeds Detect
        Layer 20: Conv (downsample)
        Layer 21: Concat(↑, y[10]=P5)
        Layer 22: C3k2 → saved → feeds Detect

    Head (layer 23):
        Detect(y[16], y[19], y[22])
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Backbone layer indices where the neck reads features
_P3_IDX = 4
_P4_IDX = 6
_P5_IDX = 10

# First neck layer (everything before this is backbone)
_NECK_START = 11


class YOLONeckRunner(nn.Module):
    """Runs YOLO26 neck+head on externally provided FPN features.

    Usage:
        runner = YOLONeckRunner("yolo26s.pt")
        detections = runner(adapted_features)
        # adapted_features = {"p3": tensor, "p4": tensor, "p5": tensor}
    """

    def __init__(self, model_variant: str = "yolo26s.pt") -> None:
        super().__init__()

        logger.info("Loading YOLO26 model for neck extraction: %s", model_variant)
        yolo = YOLO(model_variant)
        self._yolo_model = yolo.model
        self._layers = yolo.model.model
        self._save_indices = yolo.model.save
        self._num_layers = len(self._layers)

        logger.info(
            "YOLONeckRunner: %d layers total, neck starts at layer %d, "
            "save indices: %s",
            self._num_layers, _NECK_START, sorted(self._save_indices),
        )

    def forward(
        self, features: dict[str, torch.Tensor]
    ) -> list[torch.Tensor]:
        """Run neck+head on adapted FPN features.

        Args:
            features: Dict with keys "p3", "p4", "p5" containing
                tensors matching YOLO26 expected shapes:
                  p3: (B, 256, 80, 80)
                  p4: (B, 256, 40, 40)
                  p5: (B, 512, 20, 20)

        Returns:
            Raw detection output from the YOLO Detect head.
        """
        # Pre-populate the save list with our injected features
        y: list[torch.Tensor | None] = [None] * self._num_layers

        y[_P3_IDX] = features["p3"]
        y[_P4_IDX] = features["p4"]
        y[_P5_IDX] = features["p5"]

        # Start from P5 as the input to the first neck layer (Upsample)
        x = features["p5"]

        # Run neck + detect head (layers 11 onwards)
        for i in range(_NECK_START, self._num_layers):
            m = self._layers[i]

            # Resolve input: either previous output or saved feature(s)
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    # List of sources (e.g. Concat takes [-1, 6])
                    x = [x if j == -1 else y[j] for j in m.f]

            x = m(x)

            # Save output if this layer is in the save set
            if i in self._save_indices:
                y[i] = x

        return x
