"""
detection head — wraps Pradipt's Adapter-v2 + YOLO26 detector.

Two modes:

  mode="shared"     — consumes hidden_states from a BackboneOutput. Skips the
                      DINOv2 forward inside Pradipt's SharedBackbonePipeline
                      by subclassing it and adding a `predict_from_features()`
                      method that starts from line 319 of his predict() (i.e.
                      right where he calls `self.adapter(backbone_features,
                      img_tensor)`). Zero source-file modifications.

  mode="subprocess" — shells out to Pradipt's run_shared_backbone.py CLI as-is.
                      Runs its own DINOv2 (double forward), but is the
                      exact validated pipeline.

Default: `shared`. Fallback to subprocess is available via `mode="subprocess"`
or by setting the env var SENTRY_DETECTION_MODE=subprocess.

Upstream update path:
    - Pradipt changes his adapter signature → update `_predict_shared_frame`.
    - Pradipt changes the checkpoint keys → update `_load_pradipt_impl`.
    - Nothing outside `heads/detection.py` needs to change.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from ..perception_types import BackboneOutput, HeadResult


def _repo_root() -> Path:
    """UnifiedPerception/ root (this file is at <root>/unified_perception/heads/detection.py)."""
    return Path(__file__).resolve().parents[2]


def _find_training_package():
    """Self-contained: default to the vendored training_package inside this repo
    (<root>/third_party/training_package). Override with SENTRY_TRAINING_PACKAGE
    only if you deliberately want an external one."""
    env = os.environ.get("SENTRY_TRAINING_PACKAGE")
    if env:
        return env
    return str(_repo_root() / "third_party" / "training_package")

DEFAULT_TRAINING_PACKAGE = _find_training_package()


# =============================================================================
# Path resolution
# =============================================================================

def _resolve_paths(training_package: str, adapter_type: str) -> dict:
    tp = Path(training_package)
    if not tp.exists():
        raise FileNotFoundError(f"training_package not found: {tp}")
    if adapter_type == "vit":
        adapter = tp / "runs/vit_dinov2_v2_coco/adapter_best.pt"
    elif adapter_type == "convnext":
        adapter = tp / "runs/convnext_v2_coco/adapter_best.pt"
    else:
        raise ValueError(f"unknown adapter_type: {adapter_type}")
    yolo = tp / "yolo26s.pt"
    for p in (adapter, yolo):
        if not p.exists():
            raise FileNotFoundError(f"missing: {p}")
    return {
        "adapter_ckpt": str(adapter),
        "yolo_weights": str(yolo),
        "runner":       str(tp / "scripts/run_shared_backbone.py"),
        "python":       str(tp / ".venv/bin/python"),
        "training_package_dir": str(tp),
    }


# =============================================================================
# Layer 2 — upstream import + subclass
# =============================================================================

_PRADIPT_IMPL_CACHE: dict[str, Any] = {}


def _load_pradipt_impl(training_package: str, adapter_type: str, device: str = "cuda"):
    """Import Pradipt's SharedBackbonePipeline, subclass to add features-in method.

    We do NOT modify his file. We import his class and add ONE new method
    (`predict_from_features`) that mirrors his `predict()` but starts from
    the point AFTER the DINOv2 forward.
    """
    key = f"{training_package}|{adapter_type}|{device}"
    if key in _PRADIPT_IMPL_CACHE:
        return _PRADIPT_IMPL_CACHE[key]

    paths = _resolve_paths(training_package, adapter_type)
    tp = Path(training_package)
    for p in (tp, tp / "scripts"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    import torch  # noqa: F401
    from run_shared_backbone import SharedBackbonePipeline  # type: ignore

    class SharedBackboneDetector(SharedBackbonePipeline):
        """Adds predict_from_features() alongside Pradipt's predict()."""

        def predict_from_features(
            self,
            hidden_states,     # tuple/list of [1, N+1, D] tensors per layer
            img_tensor,        # [1, C, H, W] preprocessed image tensor
            original_hw=None,  # (H0, W0) of the source frame, for bbox scaling
        ):
            """Skip DINOv2 forward — feed features directly to adapter+YOLO26.

            Mirrors the tail of SharedBackbonePipeline.predict() starting at
            the adapter call (his lines 316/320 onward). This includes YOLO26
            neck+head + postprocessing.

            Parameters
            ----------
            hidden_states : tuple[torch.Tensor]
                Per-layer DINOv2 outputs, EXACTLY as returned by
                `self.vit_model(pixel_values=..., output_hidden_states=True).hidden_states`.
                Must be on `self.device`.
            img_tensor : torch.Tensor
                Preprocessed image tensor [1, C, H, W] on `self.device`.
                Used by ViTYOLOAdapter_v2's detail branch.
            original_hw : (int, int) | None
                Original frame (H, W) for bbox scale-back. If None, bboxes are
                left in imgsz coordinates.
            """
            import torch

            with torch.no_grad():
                # ─── Adapter (matches Pradipt's line 316 for convnext / 320 for vit)
                if self.adapter_type == "convnext":
                    adapted = self.adapter(hidden_states)
                elif self.adapter_type == "vit":
                    adapted = self.adapter(hidden_states, img_tensor)
                else:
                    raise ValueError(f"unknown adapter_type: {self.adapter_type}")

                # ─── YOLO26 neck+head (matches Pradipt's lines 322-334)
                self.detect_head.training = False
                self.detect_head.end2end = True
                y = [None] * len(self.yolo_layers)
                y[4]  = adapted["p3"]
                y[6]  = adapted["p4"]
                y[10] = adapted["p5"]
                x = adapted["p5"]
                for i in range(11, len(self.yolo_layers)):
                    m = self.yolo_layers[i]
                    if m.f != -1:
                        if isinstance(m.f, int):
                            x = y[m.f]
                        else:
                            x = [x if j == -1 else y[j] for j in m.f]
                    x = m(x)
                    if i in self.save_indices:
                        y[i] = x

                # ─── Postprocess (matches Pradipt's lines 340-352)
                raw_output = x
                postprocessed = raw_output[0] if isinstance(raw_output, tuple) else raw_output
                preds = postprocessed[0]
                mask = preds[:, 4] >= self.conf_thres
                preds = preds[mask].cpu().float().numpy()

            # Rescale bboxes to original frame size if provided
            if original_hw is not None and len(preds) > 0:
                H0, W0 = original_hw
                sx, sy = W0 / self.imgsz, H0 / self.imgsz
                preds[:, [0, 2]] *= sx
                preds[:, [1, 3]] *= sy

            # Package detections
            detections = []
            for p in preds:
                x1, y1, x2, y2, conf, cls_id = p
                cls_id = int(cls_id)
                detections.append({
                    "bbox":      [float(x1), float(y1), float(x2), float(y2)],
                    "conf":      float(conf),
                    "class_id":  cls_id,
                    "class_name": self.class_names.get(cls_id, f"cls_{cls_id}"),
                })
            return detections

    impl = SharedBackboneDetector(
        adapter_checkpoint=paths["adapter_ckpt"],
        yolo_weights=paths["yolo_weights"],
        adapter_type=adapter_type,
    )
    _PRADIPT_IMPL_CACHE[key] = impl
    return impl


# =============================================================================
# Layer 1 — head implementation
# =============================================================================

class DetectionHead:
    name = "detection"
    required_features = ["hidden_states", "frames"]

    def __init__(
        self,
        training_package: str = DEFAULT_TRAINING_PACKAGE,
        adapter_type: str = "vit",  # DINOv2 ViT-S — same backbone as action-recog, true shared backbone
        conf_thres: float = 0.25,
        mode: str | None = None,   # "shared" (default) or "subprocess"
    ):
        self.training_package = training_package
        self.adapter_type = adapter_type
        self.conf_thres = conf_thres
        # For convnext adapter (SOL config, DINOv3-ConvNeXt backbone), we must
        # use subprocess mode because our shared venv only has transformers
        # pinned for DINOv2 + action-recog. Subprocess runs in Pradipt's own venv.
        default_mode = "subprocess" if adapter_type == "convnext" else "shared"
        self.mode = mode or os.environ.get("SENTRY_DETECTION_MODE", default_mode)

    def predict(self, features: BackboneOutput) -> HeadResult:
        source = features.metadata.get("source")
        if not source:
            return HeadResult(head_name=self.name, error="no source in features.metadata")
        if self.mode == "subprocess":
            return self._predict_subprocess(source)
        return self._predict_shared(features)

    # ---- SHARED backbone path (V2, default) ----

    def _predict_shared(self, features: BackboneOutput) -> HeadResult:
        try:
            impl = _load_pradipt_impl(self.training_package, self.adapter_type)
            impl.conf_thres = self.conf_thres
        except Exception as e:
            return HeadResult(
                head_name=self.name,
                error=f"shared-mode impl load failed ({e}). Try mode='subprocess'."
            )

        import torch
        # Sanity checks: caller must give us hidden_states + frames
        if not features.hidden_states:
            return HeadResult(
                head_name=self.name,
                error="features.hidden_states empty. Re-extract with backbone.extract_features "
                      "(default returns hidden_states of all layers)."
            )
        if features.frames is None:
            return HeadResult(
                head_name=self.name,
                error="features.frames is None. Re-extract with keep_frames=True (needed "
                      "for the ViT adapter's detail branch)."
            )

        # dict -> ordered tuple to match Pradipt's expectation:
        #   list(hf_output.hidden_states) is a tuple of length (num_layers + 1)
        hs_dict = features.hidden_states
        max_idx = max(hs_dict.keys())
        hs_tuple = tuple(hs_dict[i] for i in range(max_idx + 1))
        device = "cuda"
        # Move to device
        hs_tuple = tuple(h.to(device) for h in hs_tuple)
        frames = features.frames.to(device)  # [T, C, H, W]
        T = frames.shape[0]

        all_frames_dets: list[list[dict]] = []
        try:
            for t in range(T):
                # Slice per-frame hidden states: each h is [T, N+1, D] → [1, N+1, D]
                per_frame_hs = tuple(h[t : t + 1] for h in hs_tuple)
                per_frame_img = frames[t : t + 1]  # [1, C, H, W]
                dets = impl.predict_from_features(per_frame_hs, per_frame_img)
                all_frames_dets.append(dets)
        except Exception as e:
            return HeadResult(
                head_name=self.name,
                error=f"predict_from_features raised: {e!r}. Falling back to mode='subprocess' "
                      f"is safe — call again with mode='subprocess'."
            )

        n_frames = len(all_frames_dets)
        n_persons_per_frame = [sum(1 for d in dets if d["class_name"] == "person")
                               for dets in all_frames_dets]
        avg_persons = sum(n_persons_per_frame) / n_frames if n_frames else 0.0
        person_confs = [d["conf"] for dets in all_frames_dets
                        for d in dets if d["class_name"] == "person"]
        mean_person_conf = sum(person_confs) / len(person_confs) if person_confs else 0.0

        return HeadResult(
            head_name=self.name,
            primary=all_frames_dets,
            confidence=float(mean_person_conf),
            details={
                "mode": "shared",
                "num_frames_sampled": n_frames,
                "avg_persons_per_frame": avg_persons,
                "adapter_type": self.adapter_type,
                "conf_thres": self.conf_thres,
                "note": "bboxes are in imgsz coordinates (typically 224). To draw on "
                        "the original video, scale by (orig_w/224, orig_h/224).",
            },
            annotated_mp4=None,  # shared path doesn't produce mp4 (yet)
        )

    # ---- SUBPROCESS fallback (V1) ----

    def _predict_subprocess(self, mp4_path: str) -> HeadResult:
        import re
        import os as _os
        out_dir = Path(mp4_path).parent / "_detection_head_scratch"
        out_dir.mkdir(exist_ok=True)
        paths = _resolve_paths(self.training_package, self.adapter_type)
        # Subprocess mode needs Pradipt's venv interpreter. If it's absent
        # (e.g. training_package re-extracted without rebuilding the venv), fail
        # with an actionable message instead of a raw ENOENT on the interpreter.
        if not _os.path.exists(paths["python"]):
            return HeadResult(
                head_name=self.name,
                error=(f"subprocess mode needs Pradipt's venv at {paths['python']} "
                       f"but it's missing. Rebuild it (scripts/setup_perception.sh, "
                       f"clear the pradipt_venv done-marker first), or use the default "
                       f"shared/vit mode (unset SENTRY_DETECTION_MODE, adapter_type='vit')."),
            )
        cmd = [
            paths["python"], paths["runner"],
            "--source", mp4_path,
            "--adapter-checkpoint", paths["adapter_ckpt"],
            "--yolo-weights", paths["yolo_weights"],
            "--adapter-type", self.adapter_type,
            "--conf-thres", str(self.conf_thres),
            "--no-display", "--save-video",
            "--output-dir", str(out_dir),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return HeadResult(head_name=self.name, error=proc.stderr[-1500:])

        rows = re.findall(r"Frame (\d+)/\d+ — (\d+) detections — ([\d.]+) FPS", proc.stdout)
        samples = [{"frame": int(f), "n_dets": int(d), "fps": float(fps)} for f, d, fps in rows]
        avg_dets = sum(s["n_dets"] for s in samples) / len(samples) if samples else 0.0

        return HeadResult(
            head_name=self.name,
            primary=samples,
            confidence=avg_dets,
            details={
                "mode": "subprocess",
                "training_package": self.training_package,
                "adapter_type": self.adapter_type,
                "note": "person-only stats: run scripts/detector_stats.py separately",
            },
            annotated_mp4=str(out_dir / "shared_backbone_output.mp4"),
        )

    def annotate(
        self,
        source_mp4: str,
        head_result: "HeadResult",
        out_path: str,
        image_size: int = 640,
        resize_short_side: int = 640,
    ) -> Optional[str]:
        """Draw per-frame bboxes onto the source video → annotated mp4.

        Bboxes returned by predict_from_features are in imgsz (224) space.
        We invert the DINOv2 preprocessing (Resize(256, short-side) +
        CenterCrop(224)) to map them back to source resolution.
        """
        try:
            import cv2
        except ImportError:
            print("[detection] cv2 not available — skipping annotate", file=sys.stderr)
            return None

        # If subprocess mode already produced its own annotated mp4, use that.
        if head_result and head_result.annotated_mp4 and Path(head_result.annotated_mp4).exists():
            return head_result.annotated_mp4

        primary = head_result.primary if head_result else None
        # Sanity: shared-mode primary is list-of-lists-of-dicts. Subprocess-mode
        # primary is list-of-dicts (per-frame counters). Only draw for the former.
        if not primary or not isinstance(primary[0], list):
            return None

        cap = cv2.VideoCapture(source_mp4)
        if not cap.isOpened():
            print(f"[detection] annotate: could not open {source_mp4}", file=sys.stderr)
            return None
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Invert simple squash: preprocessed frame is image_size x image_size,
        # so src = preproc * (src_dim / image_size), independent for x and y.
        sx_scale = src_w / image_size
        sy_scale = src_h / image_size

        def unproject(x: float, y: float) -> tuple[int, int]:
            return int(round(x * sx_scale)), int(round(y * sy_scale))

        # Encoder
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (src_w, src_h))
        if not writer.isOpened():
            print(f"[detection] annotate: writer failed to open {out_path}", file=sys.stderr)
            cap.release()
            return None

        # Colors: BGR
        COLORS = {
            "person": (0, 255, 0),
            "car":    (255, 128, 0),
            "truck":  (255, 128, 0),
            "bus":    (255, 128, 0),
        }
        DEFAULT_COLOR = (0, 255, 255)

        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dets = primary[idx] if idx < len(primary) else []
            # DEMO_PERSON_ONLY=1 filters the VISUAL overlay to person-class only
            # (drops COCO-domain-gap noise like "skateboard" on a jogger's feet).
            # The full detection primary + perception.json are unaffected.
            import os as _os
            _person_only = _os.environ.get("DEMO_PERSON_ONLY") == "1"
            for d in dets:
                cls_name = d.get("class_name", "?")
                if _person_only and cls_name != "person":
                    continue
                x1, y1, x2, y2 = d["bbox"]
                conf = d.get("conf", 0.0)
                p1 = unproject(x1, y1)
                p2 = unproject(x2, y2)
                color = COLORS.get(cls_name, DEFAULT_COLOR)
                cv2.rectangle(frame, p1, p2, color, thickness=2)
                label = f"{cls_name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (p1[0], p1[1] - th - 6),
                              (p1[0] + tw + 4, p1[1]), color, -1)
                cv2.putText(frame, label, (p1[0] + 2, p1[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            writer.write(frame)
            idx += 1

        cap.release()
        writer.release()
        head_result.annotated_mp4 = out_path
        return out_path


# =============================================================================
# Module-level export
# =============================================================================

HEAD = DetectionHead()
