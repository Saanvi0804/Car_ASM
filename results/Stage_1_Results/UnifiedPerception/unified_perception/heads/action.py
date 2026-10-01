"""
action head — whole-scene K400 action recognizer over the shared DINOv2 CLS.

Consumes the per-frame CLS tokens from the shared BackboneOutput (no separate
DINOv2 forward) and runs a trained K400 LINEAR PROBE to produce a top-K
prediction. The probe is a single Linear over aggregated CLS: average when the
Linear's input dim == D (the default checkpoint), or concat when it == D*T.

The checkpoint is env-overridable (no code change) via SENTRY_ACTION_CKPT:

    SENTRY_ACTION_CKPT=/path/to/best.pth \
        python -m unified_perception --source clip.mp4 --output-dir out

Optional sliding-window mode (SENTRY_ACTION_SLIDING=1) classifies overlapping
windows over the ~target-fps-subsampled CLS sequence and reports the most
frequent label plus a per-window timeline (used for the time-aligned overlay).
"""
from __future__ import annotations

import os as _os
import sys
from pathlib import Path
from typing import Optional

from ..perception_types import BackboneOutput, HeadResult


def _repo_root() -> Path:
    """UnifiedPerception/ root (this file is at <root>/unified_perception/heads/action.py)."""
    return Path(__file__).resolve().parents[2]


# Self-contained: all action assets live under <root>/assets/ (vendored). Each is
# still env-overridable if you want to point elsewhere.
_ASSETS = _repo_root() / "assets"
DEFAULT_HEAD_CKPT   = _os.environ.get("SENTRY_ACTION_CKPT",
                                      str(_ASSETS / "k400_outputs" / "best.pth"))
DEFAULT_CLASSES_TXT = _os.environ.get("SENTRY_CLASSES_TXT",
                                      str(_ASSETS / "kinetics400" / "annotations" / "classes.txt"))
DEFAULT_TRAIN_DIR   = _os.environ.get("SENTRY_TRAIN_DIR",
                                      str(_ASSETS / "kinetics400" / "train"))


def _load_class_names(classes_txt: str, train_dir: str) -> list[str]:
    """Alphabetized K400 class list — must match the training dataset index."""
    p = Path(classes_txt)
    if p.exists():
        return [l.strip() for l in p.read_text().splitlines() if l.strip()]
    # Fallback: derive from train subdirs (same ordering the trainer uses)
    t = Path(train_dir)
    if t.exists():
        return sorted(d.name for d in t.iterdir() if d.is_dir())
    raise FileNotFoundError(f"classes not found at {classes_txt} or {train_dir}")


# =============================================================================
# Linear-probe head loading
# =============================================================================

class _LoadedHead:
    """A loaded linear-probe action head: a bare ``nn.Linear`` classifier
    reconstructed from checkpoint weights. Consumes AGGREGATED CLS
    (``[1, in_dim]``) — average if ``in_dim == D``, concat if ``in_dim == D*T``.
    """

    def __init__(self, model, num_classes, classes, device, num_frames=8, in_dim=None):
        self.model = model
        self.num_classes = num_classes
        self.classes = classes
        self.device = device
        self.num_frames = num_frames or 8
        self.in_dim = in_dim
        self.aggregation = None  # set at classify time

    def _fit_frames(self, cls_tokens):
        """Subsample/pad a [T_all, D] CLS stack to exactly self.num_frames."""
        import numpy as np
        import torch
        T_all, D = cls_tokens.shape
        T = self.num_frames
        if T_all == T:
            return cls_tokens, D, T
        if T_all > T:
            idx = np.linspace(0, T_all - 1, T, dtype=int)
            return cls_tokens[idx], D, T
        pad = cls_tokens[-1:].expand(T - T_all, D)
        return torch.cat([cls_tokens, pad], dim=0), D, T

    def classify(self, cls_tokens, topk: int = 5):
        """cls_tokens: torch tensor [T, D] → list[(label, prob)] top-k."""
        import torch
        seq, D, T = self._fit_frames(cls_tokens)
        seq = seq.to(self.device)
        with torch.no_grad():
            if self.in_dim == D:
                self.aggregation = "average"
                agg = seq.mean(dim=0, keepdim=True)       # [1, D]
            elif self.in_dim == D * T:
                self.aggregation = "concat"
                agg = seq.reshape(1, -1)                  # [1, T*D]
            else:
                raise RuntimeError(
                    f"linear head in_dim={self.in_dim} incompatible with "
                    f"cls shape [T={T}, D={D}]")
            logits = self.model(agg)
            probs = torch.softmax(logits, dim=-1)[0]
        k = min(topk, int(probs.shape[-1]))
        top_probs, top_idx = torch.topk(probs, k=k)
        return [(self.classes[int(i)], float(p)) for i, p in zip(top_idx, top_probs)]

    def describe(self) -> dict:
        return {
            "kind": "linear",
            "aggregation": self.aggregation,
            "num_classes": self.num_classes,
        }


def load_action_head(head_ckpt: str, classes_txt: str, train_dir: str,
                     device: str = "cuda", num_frames_hint: int = 8) -> _LoadedHead:
    """Load the K400 linear-probe action head.

    Reconstructs the single classifier ``Linear`` from the checkpoint weights
    (the checkpoint stores ``{fc.weight, fc.bias}``). Aggregation is inferred at
    classify time from the Linear's input dim (average if ``== D``, concat if
    ``== D*T``).
    """
    import torch
    import torch.nn as nn
    if not Path(head_ckpt).exists():
        raise FileNotFoundError(head_ckpt)
    state = torch.load(head_ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict):
        sd = state.get("state_dict", state.get("model_state_dict", state))
    else:
        sd = state

    classes = _load_class_names(classes_txt, train_dir)

    # Find the classifier Linear (2-D weight). Prefer one matching our class
    # count; otherwise tolerate common class counts (K400/SSv2/UCF101).
    fc_w = fc_b = None
    for k, v in sd.items():
        if hasattr(v, "ndim") and v.ndim == 2 and v.shape[0] == len(classes):
            fc_w = v
            fc_b = sd.get(k.replace("weight", "bias"), torch.zeros(v.shape[0]))
            break
    if fc_w is None:
        for k, v in sd.items():
            if hasattr(v, "ndim") and v.ndim == 2 and int(v.shape[0]) in (400, 174, 101):
                fc_w = v
                fc_b = sd.get(k.replace("weight", "bias"), torch.zeros(v.shape[0]))
                break
    if fc_w is None:
        raise RuntimeError(
            f"{head_ckpt}: no classifier Linear found "
            f"(expected a 2-D weight with {len(classes)} rows)")

    out_dim, in_dim = int(fc_w.shape[0]), int(fc_w.shape[1])
    lin = nn.Linear(in_dim, out_dim).to(device).eval()
    lin.load_state_dict({"weight": fc_w.to(device), "bias": fc_b.to(device)})
    return _LoadedHead(lin, out_dim, classes, device,
                       num_frames=num_frames_hint, in_dim=in_dim)


# =============================================================================
# Head implementation
# =============================================================================

class ActionHead:
    name = "action"
    required_features = ["cls"]

    def __init__(
        self,
        head_ckpt: str = DEFAULT_HEAD_CKPT,
        classes_txt: str = DEFAULT_CLASSES_TXT,
        train_dir: str = DEFAULT_TRAIN_DIR,
        topk: int = 5,
        device: str = "cuda",
    ):
        self.head_ckpt   = head_ckpt
        self.classes_txt = classes_txt
        self.train_dir   = train_dir
        self.topk        = topk
        self.device      = device
        self._loaded: Optional[_LoadedHead] = None  # lazy
        # Sliding-window mode (opt-in via env). Default off = single uniform
        # prediction over the whole clip's CLS sequence.
        self.sliding    = _os.environ.get("SENTRY_ACTION_SLIDING", "0") == "1"
        self.target_fps = float(_os.environ.get("SENTRY_ACTION_FPS", "5"))
        self.hop        = max(1, int(_os.environ.get("SENTRY_ACTION_HOP", "1")))
        self.topn       = max(1, int(_os.environ.get("SENTRY_ACTION_TOPN", "3")))

    def _ensure_loaded(self):
        if self._loaded is None:
            self._loaded = load_action_head(
                self.head_ckpt, self.classes_txt, self.train_dir, self.device)

    def predict(self, features: BackboneOutput) -> HeadResult:
        cls = features.cls
        if cls is None or cls.numel() == 0:
            return HeadResult(head_name=self.name, error="no CLS tokens in features")

        try:
            self._ensure_loaded()
        except FileNotFoundError as e:
            return HeadResult(
                head_name=self.name,
                error=f"head checkpoint not found: {e}. Set SENTRY_ACTION_CKPT to a "
                      f"K400 linear-probe .pth."
            )
        except Exception as e:
            return HeadResult(head_name=self.name, error=f"head load failed: {e}")

        # Classify the whole-clip CLS sequence, either as one window or by sliding.
        try:
            if self.sliding:
                return self._predict_sliding(features, cls)
            top_k = self._loaded.classify(cls, self.topk)
        except Exception as e:
            return HeadResult(head_name=self.name, error=f"action classify failed: {e}")

        return HeadResult(
            head_name=self.name,
            primary=top_k[0][0],
            confidence=float(top_k[0][1]),
            details={
                "top_k":     top_k,
                **self._loaded.describe(),
                "head_ckpt": self.head_ckpt,
            },
            annotated_mp4=None,
        )

    def _predict_sliding(self, features: BackboneOutput, cls) -> HeadResult:
        """Slide a window over the ~target_fps-subsampled CLS sequence, classify
        each window, and aggregate per-window top-1 labels into a top-N
        frequency report + a per-window timeline (for the time-aligned overlay).
        Each window is an ordinary classify() call.
        """
        from collections import Counter
        meta = features.metadata or {}
        src_fps = float(meta.get("fps") or 0.0)
        bstride = int(meta.get("stride") or 1)      # CLS idx -> raw frame = i*bstride
        win = int(self._loaded.num_frames)          # window size (checkpoint num_frames)
        T_all = int(cls.shape[0])

        # Subsample the CLS sequence to ~target_fps.
        stride = max(1, round(src_fps / self.target_fps)) if src_fps > 0 else 1
        sampled = list(range(0, T_all, stride))

        # Overlapping windows of `win` consecutive sampled frames, hop in sampled frames.
        if len(sampled) < win:
            windows = [sampled]                     # short clip -> one (padded) window
        else:
            windows = [sampled[k:k + win]
                       for k in range(0, len(sampled) - win + 1, self.hop)]

        timeline = []                               # [center_raw_frame, label, conf]
        confs_by_label: dict = {}
        counts: Counter = Counter()
        for widx in windows:
            tk = self._loaded.classify(cls[widx], self.topk)
            label, conf = tk[0][0], float(tk[0][1])
            center_raw = int(widx[len(widx) // 2] * bstride)
            timeline.append([center_raw, label, conf])
            counts[label] += 1
            confs_by_label.setdefault(label, []).append(conf)

        def mean_conf(lbl):
            v = confs_by_label.get(lbl) or [0.0]
            return sum(v) / len(v)
        ranked = sorted(counts, key=lambda l: (counts[l], mean_conf(l)), reverse=True)
        freq_topN = [[l, int(counts[l]), round(mean_conf(l), 4)] for l in ranked[:self.topn]]
        top_k = [(l, mean_conf(l)) for l in ranked[:self.topk]]
        primary = ranked[0] if ranked else None

        return HeadResult(
            head_name=self.name,
            primary=primary,
            confidence=float(mean_conf(primary)) if primary else 0.0,
            details={
                "mode":          "sliding",
                "top_k":         top_k,
                "freq_topN":     freq_topN,
                "num_windows":   len(windows),
                "window":        win,
                "hop":           self.hop,
                "target_fps":    self.target_fps,
                "src_fps":       src_fps,
                "sample_stride": stride,
                "timeline":      timeline,
                **self._loaded.describe(),
                "head_ckpt":     self.head_ckpt,
            },
            annotated_mp4=None,
        )

    def annotate(
        self, source_mp4: str, head_result, out_path: str
    ) -> Optional[str]:
        """Burn the top-1 K400 label + confidence as a banner over every frame.

        This is the standalone action-only mp4 (no bboxes). The unified
        pipeline separately produces a combined mp4 (bboxes + banner) via
        _maybe_combine_overlays, which layers the banner on top of the
        detection-annotated mp4.
        """
        if head_result is None or not head_result.primary:
            return None
        import subprocess
        label = head_result.primary
        conf = head_result.confidence or 0.0
        text = f"{label} ({conf:.2f})".replace("'", "")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", source_mp4,
            "-vf",
            f"drawtext=text='{text}':x=(w-tw)/2:y=20:fontsize=48:"
            f"fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=10",
            "-c:v", "libx264", "-crf", "20", out_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[action] annotate ffmpeg failed:\n{proc.stderr[-400:]}",
                  file=sys.stderr)
            return None
        head_result.annotated_mp4 = out_path
        return out_path


# =============================================================================
# Module-level export
# =============================================================================

HEAD = ActionHead()
