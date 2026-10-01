"""
perception_types — shared type definitions for the sentry-mode perception pipeline.

Defines the stable contract between the backbone and the heads:

  - BackboneOutput: what a backbone produces from an mp4.
  - Head:           protocol every head implements.
  - HeadResult:     what a head returns.

See PERCEPTION_ARCHITECTURE.md for design rationale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


# =============================================================================
# Backbone output
# =============================================================================

@dataclass
class BackboneOutput:
    """Standardized backbone output.

    Any backbone (DINOv2 today; CLIP, EVA, etc. in the future) MUST return
    an instance of this class. Heads pick which fields they need.

    Fields
    ------
    cls              : torch.Tensor  [T, D]         — last-layer CLS token per frame
    hidden_states    : dict[int, torch.Tensor]      — {layer_idx: [T, N+1, D]}
                                                    where N = num_patches, +1 for CLS
    patch_tokens     : torch.Tensor  [T, N, D]      — last-layer patch tokens (no CLS)
    metadata         : dict[str, Any]               — provenance + model info
    frames           : Optional[torch.Tensor] [T, C, H, W] — raw preprocessed frames
                                                    (kept for heads that need pixels,
                                                    e.g. an OCR head; None if not needed)
    """

    cls: Any                                  # torch.Tensor
    hidden_states: dict[int, Any] = field(default_factory=dict)
    patch_tokens: Optional[Any] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    frames: Optional[Any] = None

    def has(self, feature: str) -> bool:
        """Check if a requested feature is present + non-empty."""
        val = getattr(self, feature, None)
        if val is None:
            return False
        if isinstance(val, dict):
            return len(val) > 0
        return True


# =============================================================================
# Head result
# =============================================================================

@dataclass
class HeadResult:
    """Standardized head output.

    Every head returns an instance of this. The orchestrator collects them
    into a single JSON per input clip.

    Fields
    ------
    head_name        : str          — matches Head.name
    primary          : Any          — the "main" result. For detection, list of
                                     per-frame bbox dicts. For action, top-1
                                     class name (str).
    confidence       : float        — scalar confidence for the primary result.
                                     For detection: mean per-frame conf. For
                                     action: top-1 prob.
    details          : dict         — head-specific extras (top-k, per-frame
                                     breakdowns, class counts, whatever).
    annotated_mp4    : Optional[str] — path to an overlay mp4 the head produced,
                                     or None if this head doesn't produce one.
    error            : Optional[str] — set if predict() failed. When set,
                                     other fields may be missing.
    """

    head_name: str
    primary: Any = None
    confidence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    annotated_mp4: Optional[str] = None
    error: Optional[str] = None

    def to_json(self) -> dict:
        """JSON-serializable version (drops tensors, keeps everything else)."""
        return {
            "head_name": self.head_name,
            "primary": _jsonify(self.primary),
            "confidence": float(self.confidence),
            "details": _jsonify(self.details),
            "annotated_mp4": self.annotated_mp4,
            "error": self.error,
        }


def _jsonify(x):
    """Best-effort conversion of nested tensor/numpy stuff into JSON primitives."""
    import numpy as np
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    # torch tensor?
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "tolist"):
        return x.detach().cpu().tolist()
    return str(x)


# =============================================================================
# Head protocol
# =============================================================================

@runtime_checkable
class Head(Protocol):
    """Structural protocol every head implements. See PERCEPTION_ARCHITECTURE.md."""

    #: Unique name (registry key, lowercase). E.g. "detection", "action".
    name: str

    #: List of BackboneOutput field names this head reads. Used by the
    #: orchestrator to sanity-check that the backbone provides them.
    required_features: list[str]

    def predict(
        self,
        features: BackboneOutput,
        prior_results: Optional[dict[str, "HeadResult"]] = None,
    ) -> HeadResult:
        """Consume features → HeadResult. MUST NOT run a backbone forward pass
        for downstream heads that reuse shared features.

        `prior_results` is a dict of HeadResults from heads that ran earlier
        in the pipeline (registry order). Heads that depend on outputs from
        another head (e.g. a per-person action head depending on detection
        bboxes) read them from here. Heads that don't need cross-head data
        can ignore this kwarg.
        """
        ...

    def annotate(
        self, mp4_path: str, features: BackboneOutput, out_path: str
    ) -> Optional[str]:
        """Optional: produce an overlay mp4. Return path or None if not applicable."""
        ...
