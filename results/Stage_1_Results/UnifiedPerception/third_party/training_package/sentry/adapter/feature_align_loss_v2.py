"""Feature Alignment Loss v2 — adaptive per-scale weighting.

Three modes for weighting the MSE alignment loss across P3/P4/P5 scales:

- adaptive (v2 default): Loss-ratio rebalancing. Tracks each scale's convergence
  rate (current_loss / initial_loss). Scales that converge slower get higher
  weight automatically. Self-correcting — if large objects start struggling,
  their weight increases back up. No learnable params.

- fixed: Hardcoded weights [1.5, 1.0, 0.5] — small object emphasis.

- uniform: All weights 1.0 — identical to v1 FeatureAlignmentLoss behavior.

Detection loss is UNAFFECTED by these weights. Only the alignment term
(L_align in L_total = L_detect + alpha * L_align) is reweighted. Alpha
decays from 1.0→0.1 over training, so alignment is <10% of total loss
by late training — large object detection cannot be harmed.

Usage:
    loss_fn = FeatureAlignmentLoss_v2(mode="adaptive")
    loss = loss_fn(adapter_output, teacher_features)

Reference: BRAINSTORM_LOG.md Decision 89, Approach 5 (loss-ratio rebalancing)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse alpha decay from v1
from sentry.adapter.feature_align_loss import compute_alpha  # noqa: F401

logger = logging.getLogger(__name__)


class FeatureAlignmentLoss_v2(nn.Module):
    """MSE alignment loss with adaptive per-scale weighting.

    All internal state (weights, initial_losses) is created on the same
    device as the input tensors on first forward call. No register_buffer
    — avoids CPU/GPU device mismatch issues.

    Args:
        mode: "adaptive" (v2 default), "fixed", or "uniform" (v1 behavior).
        init_weights: Weights for "fixed" mode [P3, P4, P5]. Default [1.5, 1.0, 0.5].
        alpha: Rebalancing aggressiveness for adaptive mode. Default 1.5.
        ema_momentum: EMA factor for weight updates. Default 0.5.
    """

    def __init__(
        self,
        mode: str = "adaptive",
        init_weights: list[float] | None = None,
        alpha: float = 1.5,
        ema_momentum: float = 0.5,
    ) -> None:
        super().__init__()

        if mode not in ("adaptive", "fixed", "uniform"):
            raise ValueError(f"Unknown mode '{mode}'. Use 'adaptive', 'fixed', or 'uniform'.")

        self.mode = mode
        self.alpha = alpha
        self.ema_momentum = ema_momentum
        self._init_weights_list = init_weights or [1.5, 1.0, 0.5]

        # State created lazily on first forward (on correct device)
        self._weights: torch.Tensor | None = None
        self._initial_losses: torch.Tensor | None = None
        self._last_ratios: tuple[float, float, float] = (1.0, 1.0, 1.0)

        logger.info("FeatureAlignmentLoss_v2: mode=%s, weights=[%.2f, %.2f, %.2f] (sum=%.1f)",
                     mode, *self._init_weights_list, sum(self._init_weights_list))

    def _ensure_state(self, device: torch.device) -> None:
        """Create state tensors on the correct device (called once on first forward)."""
        if self._weights is not None:
            return

        if self.mode == "uniform":
            self._weights = torch.ones(3, device=device)
        elif self.mode == "fixed":
            w = torch.tensor(self._init_weights_list, dtype=torch.float32, device=device)
            self._weights = w / w.sum() * 3.0
        else:  # adaptive
            self._weights = torch.ones(3, device=device)
            self._initial_losses = None  # filled on first forward with losses

    def forward(
        self,
        adapter_output: dict[str, torch.Tensor],
        teacher_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute weighted feature alignment loss."""
        loss_p3 = F.mse_loss(adapter_output["p3"], teacher_features["P3"])
        loss_p4 = F.mse_loss(adapter_output["p4"], teacher_features["P4"])
        loss_p5 = F.mse_loss(adapter_output["p5"], teacher_features["P5"])

        device = loss_p3.device
        self._ensure_state(device)

        if self.mode == "adaptive":
            losses = torch.stack([loss_p3, loss_p4, loss_p5])

            # Record initial losses on first call
            if self._initial_losses is None:
                self._initial_losses = losses.detach().clone()
                return loss_p3 + loss_p4 + loss_p5

            # Compute convergence ratios: current / initial
            # High ratio = slow convergence = needs more weight
            with torch.no_grad():
                ratios = losses.detach() / (self._initial_losses + 1e-8)
                ratios = ratios.clamp(0.1, 10.0)
                self._last_ratios = (ratios[0].item(), ratios[1].item(), ratios[2].item())
                target_weights = ratios ** self.alpha
                target_weights = target_weights / target_weights.sum() * 3.0
                # EMA update (all on same device)
                self._weights = self.ema_momentum * self._weights + (1.0 - self.ema_momentum) * target_weights

        w = self._weights
        return w[0] * loss_p3 + w[1] * loss_p4 + w[2] * loss_p5

    def get_weights(self) -> tuple[float, float, float]:
        """Return current weights for logging."""
        if self._weights is None:
            return (1.0, 1.0, 1.0)
        return (self._weights[0].item(), self._weights[1].item(), self._weights[2].item())

    def get_ratios(self) -> tuple[float, float, float]:
        """Return current loss ratios (current/initial) for diagnostics."""
        return self._last_ratios
