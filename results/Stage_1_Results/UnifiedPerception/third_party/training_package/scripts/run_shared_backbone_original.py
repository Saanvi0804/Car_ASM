#!/usr/bin/env python3
"""Run detection on video using the shared backbone pipeline:

    DINOv3 ConvNeXt-T → FPN Adapter → YOLO26 Neck+Head → Detections

This is the "production mode" inference — shared backbone extracts features once,
adapter bridges to YOLO format, and the YOLO neck+head produces detections.

Usage:
    # Run on test video with trained adapter
    python scripts/run_shared_backbone.py \
        --source test_data/sample_video_1280_800_Undistorted.avi \
        --adapter-checkpoint runs/adapter_round2/adapter_best.pt \
        --yolo-weights runs/detect/runs/baseline/yolo26s_fisheye8k/weights/best.pt

    # Save output video without display
    python scripts/run_shared_backbone.py \
        --source test_data/sample_video_1280_800_Undistorted.avi \
        --adapter-checkpoint runs/adapter_round2/adapter_best.pt \
        --yolo-weights runs/detect/runs/baseline/yolo26s_fisheye8k/weights/best.pt \
        --no-display --save-video
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentry.backbone import build_backbone, BACKBONE_REGISTRY
from sentry.adapter.fpn_adapter import FPNChannelAdapter
from sentry.adapter.fpn_adapter_v2 import FPNSpatialAdapter
from sentry.adapter.vit_yolo_adapter import ViTYOLOAdapter
from sentry.adapter.vit_yolo_adapter_v2 import ViTYOLOAdapter_v2

logger = logging.getLogger(__name__)

# ImageNet normalization for ConvNeXt
_IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Detection class names (FishEye8K)
_FISHEYE_CLASSES = {0: "Pedestrian", 1: "Bike", 2: "Car", 3: "Bus", 4: "Truck"}
# COCO class names (if using COCO-pretrained YOLO)
_COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus",
    7: "truck", 24: "backpack", 26: "handbag", 43: "knife",
    34: "baseball bat", 76: "scissors",
}

# Colors per class (BGR)
_COLORS = [
    (147, 20, 255),   # Pedestrian — magenta
    (0, 0, 255),      # Bike — red
    (255, 144, 30),   # Car — blue
    (0, 255, 255),    # Bus — yellow
    (0, 165, 255),    # Truck — orange
]


class SharedBackbonePipeline:
    """Full inference pipeline: Backbone → Adapter → YOLO neck+head.

    Supports two adapter types:
        - "convnext": FPNChannelAdapter (3×1×1 Conv, 542K) with DINOv3 ConvNeXt-T backbone
        - "vit": ViTYOLOAdapter (multi-layer+detail+bi-fusion, 577K) with DINOv3 ViT-S backbone

    Adapter type can be auto-detected from checkpoint metadata or set explicitly.
    """

    # Adapter type → compatible backbone mapping
    _ADAPTER_BACKBONE_MAP = {
        "convnext": "dinov3_convnext_tiny",
        "vit": None,  # ViT backbone loaded directly (not via backbone registry)
    }

    # Backbone configs for ViT adapter (same as train_vit_adapter.py)
    _VIT_BACKBONE_CONFIGS = {
        "dinov2": {
            "hf_name": "facebook/dinov2-small",
            "patch_size": 14,
            "skip_tokens": 1,
            "feature_dim": 384,
        },
        "dinov3": {
            "hf_name": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "patch_size": 16,
            "skip_tokens": 5,
            "feature_dim": 384,
        },
    }

    def __init__(
        self,
        adapter_checkpoint: str,
        adapter_type: str = "auto",
        yolo_weights: str = "yolo26s.pt",
        backbone_name: str = "dinov3_convnext_tiny",
        imgsz: int = 640,
        conf_thres: float = 0.25,
        device: str = "cuda",
        depth_checkpoint: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.device = torch.device(device)
        self.imgsz = imgsz
        self.conf_thres = conf_thres

        # ─── Load checkpoint and determine adapter type ───
        logger.info("Loading adapter checkpoint: %s", adapter_checkpoint)
        ckpt = torch.load(adapter_checkpoint, map_location=self.device, weights_only=True)

        # Auto-detect adapter type from checkpoint metadata
        ckpt_adapter_class = ckpt.get("adapter_class", None)
        ckpt_backbone_type = ckpt.get("backbone_type", None)

        _VIT_ADAPTER_CLASSES = {"ViTYOLOAdapter", "ViTYOLOAdapter_v2"}
        _CONVNEXT_ADAPTER_CLASSES = {"FPNChannelAdapter", "FPNSpatialAdapter"}

        if adapter_type == "auto":
            if ckpt_adapter_class in _VIT_ADAPTER_CLASSES:
                adapter_type = "vit"
                logger.info("  Auto-detected adapter type: vit (%s)", ckpt_adapter_class)
            elif ckpt_adapter_class in _CONVNEXT_ADAPTER_CLASSES:
                adapter_type = "convnext"
                logger.info("  Auto-detected adapter type: convnext (%s)", ckpt_adapter_class)
            elif ckpt_adapter_class is not None:
                adapter_type = "convnext"
                logger.info("  Unknown adapter_class '%s' — defaulting to convnext", ckpt_adapter_class)
            else:
                adapter_type = "convnext"
                logger.info("  No adapter_class in checkpoint — defaulting to convnext")
        else:
            # Explicit adapter type — validate against checkpoint
            if ckpt_adapter_class in _VIT_ADAPTER_CLASSES and adapter_type != "vit":
                raise ValueError(
                    f"Mismatch: --adapter-type={adapter_type} but checkpoint contains "
                    f"'{ckpt_adapter_class}' (ViT adapter). "
                    f"Use --adapter-type=vit or --adapter-type=auto."
                )
            if ckpt_adapter_class in _CONVNEXT_ADAPTER_CLASSES and adapter_type == "vit":
                raise ValueError(
                    f"Mismatch: --adapter-type=vit but checkpoint contains "
                    f"'{ckpt_adapter_class}' (ConvNeXt adapter). "
                    f"Use --adapter-type=convnext or --adapter-type=auto."
                )
            # Legacy checkpoint (no adapter_class) + explicit vit → warn
            if ckpt_adapter_class is None and adapter_type == "vit":
                raise ValueError(
                    f"Mismatch: --adapter-type=vit but checkpoint has no adapter_class metadata. "
                    f"This looks like a ConvNeXt adapter checkpoint. "
                    f"Use --adapter-type=convnext or --adapter-type=auto."
                )
            logger.info("  Explicit adapter type: %s", adapter_type)

        self.adapter_type = adapter_type

        # ─── Build backbone + adapter based on type ───
        if adapter_type == "convnext":
            self._init_convnext(ckpt, backbone_name, device)
        elif adapter_type == "vit":
            vit_backbone_type = ckpt_backbone_type or "dinov3"
            self._init_vit(ckpt, vit_backbone_type, imgsz, device)
        else:
            raise ValueError(f"Unknown adapter_type '{adapter_type}'. Use 'convnext', 'vit', or 'auto'.")

        logger.info("  Adapter epoch %d, train_loss %.4f", ckpt["epoch"], ckpt["loss"])

        # ─── YOLO neck+head (same for both adapter types) ───
        logger.info("Loading YOLO26 neck+head from %s...", yolo_weights)
        yolo = YOLO(yolo_weights)
        self.yolo_model = yolo.model.to(self.device)
        self.yolo_layers = self.yolo_model.model
        self.save_indices = self.yolo_model.save
        self.yolo_model.eval()

        self.detect_head = self.yolo_layers[-1]
        self.class_names = self.yolo_model.names if hasattr(self.yolo_model, "names") else _FISHEYE_CLASSES

        # Normalization tensors
        self.mean = _IMG_MEAN.to(self.device)
        self.std = _IMG_STD.to(self.device)

        # ─── Optional: Depth adapter + decoder (shared backbone) ───
        self.depth_adapter = None
        self.depth_decoder = None
        if depth_checkpoint is not None:
            from sentry.adapter.depth_adapter import DepthFPNAdapter
            from sentry.depth.depth_decoder import LightweightDepthDecoder

            depth_ckpt = torch.load(depth_checkpoint, map_location=self.device, weights_only=False)
            depth_strategy = depth_ckpt.get("depth_strategy", "option_c")
            max_depth = depth_ckpt.get("max_depth", 20.0)

            if adapter_type == "convnext":
                self.depth_adapter = DepthFPNAdapter.for_convnext(
                    strategy=depth_strategy, device=device)
            else:
                from sentry.adapter.depth_adapter import ViTDepthAdapter
                # Detect backbone type from checkpoint
                depth_backbone = depth_ckpt.get("backbone", "dinov2")
                self.depth_adapter = ViTDepthAdapter(
                    backbone_type=depth_backbone, imgsz=self.imgsz,
                    vit_dim=384, detail_channels=64, out_channels=256,
                    strategy=depth_strategy,
                ).to(self.device)
            self.depth_adapter.load_state_dict(depth_ckpt["adapter_state_dict"])
            self.depth_adapter.eval()

            min_depth = depth_ckpt.get("min_depth", 0.1)
            depth_mode = depth_ckpt.get("depth_mode", "log")
            self.depth_decoder = LightweightDepthDecoder(
                in_channels=256, max_depth=max_depth,
                min_depth=min_depth, depth_mode=depth_mode,
            )
            self.depth_decoder.load_state_dict(depth_ckpt["decoder_state_dict"])
            self.depth_decoder.to(self.device).eval()
            self.max_depth = max_depth

            logger.info("Depth loaded: epoch %d, strategy=%s, max_depth=%.1fm",
                        depth_ckpt.get("epoch", -1), depth_strategy, max_depth)

        logger.info("Pipeline ready. Type: %s, Classes: %s, Depth: %s",
                     adapter_type, self.class_names, "ON" if self.depth_adapter else "OFF")

    def _init_convnext(self, ckpt, backbone_name, device):
        """Initialize ConvNeXt backbone + FPN adapter (Round 2 style)."""
        if backbone_name not in BACKBONE_REGISTRY:
            raise ValueError(f"Unknown backbone '{backbone_name}'. Available: {list(BACKBONE_REGISTRY.keys())}")

        logger.info("  Loading ConvNeXt backbone: %s", backbone_name)
        self.backbone = build_backbone({"name": backbone_name, "device": device})
        self.backbone.to(self.device).eval()

        if ckpt.get("adapter_class") == "FPNSpatialAdapter":
            self.adapter = FPNSpatialAdapter.for_convnext_to_yolo26(device=device)
            logger.info("  Using adapter v2: FPNSpatialAdapter")
        else:
            self.adapter = FPNChannelAdapter.for_convnext_to_yolo26(device=device)
            logger.info("  Using adapter v1: FPNChannelAdapter")
        self.adapter.load_state_dict(ckpt["adapter_state_dict"])
        self.adapter.eval()
        self.vit_model = None  # Not used for convnext

    def _init_vit(self, ckpt, vit_backbone_type, imgsz, device):
        """Initialize ViT backbone + ViTYOLOAdapter (Round 3 style)."""
        from transformers import AutoModel

        if vit_backbone_type not in self._VIT_BACKBONE_CONFIGS:
            raise ValueError(
                f"Unknown ViT backbone type '{vit_backbone_type}'. "
                f"Available: {list(self._VIT_BACKBONE_CONFIGS.keys())}"
            )

        cfg = self._VIT_BACKBONE_CONFIGS[vit_backbone_type]
        logger.info("  Loading ViT backbone: %s", cfg["hf_name"])
        self.vit_model = AutoModel.from_pretrained(cfg["hf_name"])
        self.vit_model.to(self.device).eval()
        for p in self.vit_model.parameters():
            p.requires_grad = False

        if ckpt.get("adapter_class") == "ViTYOLOAdapter_v2":
            self.adapter = ViTYOLOAdapter_v2(
                backbone_type=vit_backbone_type,
                imgsz=imgsz,
                vit_dim=cfg["feature_dim"],
            )
            logger.info("  Using adapter v2: ViTYOLOAdapter_v2")
        else:
            self.adapter = ViTYOLOAdapter(
                backbone_type=vit_backbone_type,
                imgsz=imgsz,
                vit_dim=cfg["feature_dim"],
            )
            logger.info("  Using adapter v1: ViTYOLOAdapter")
        self.adapter.load_state_dict(ckpt["adapter_state_dict"])
        self.adapter.to(self.device).eval()
        self.backbone = None  # Not used for vit

    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> tuple[list[dict], np.ndarray | None]:
        """Run detection (+ optional depth) on a single BGR frame.

        Backbone runs ONCE. Features are shared between detection and depth.

        Returns:
            (detections, depth_map) where:
                detections: list of dicts with "bbox", "conf", "class_id", "class_name", "depth_m"
                depth_map: (H, W) numpy float32 in meters, or None if no depth checkpoint
        """
        h0, w0 = frame.shape[:2]

        # Preprocess: resize, BGR→RGB, normalize to [0,1]
        img = cv2.resize(frame, (self.imgsz, self.imgsz))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # ImageNet normalize
        img_norm = (img_tensor - self.mean) / self.std

        # ─── Backbone (runs ONCE) ───
        if self.adapter_type == "convnext":
            backbone_out = self.backbone(img_norm)
            backbone_features = backbone_out["feature_maps"]  # shared between detection + depth
            adapted = self.adapter(backbone_features)
        elif self.adapter_type == "vit":
            vit_output = self.vit_model(pixel_values=img_norm, output_hidden_states=True)
            backbone_features = vit_output.hidden_states  # shared
            adapted = self.adapter(backbone_features, img_tensor)

        # 3. YOLO neck+head (eval mode)
        self.detect_head.training = False
        y = [None] * len(self.yolo_layers)
        y[4] = adapted["p3"]
        y[6] = adapted["p4"]
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

        # Postprocess: YOLO26 e2e returns (postprocessed, raw)
        raw_output = x
        if isinstance(raw_output, tuple):
            postprocessed = raw_output[0]
        else:
            postprocessed = raw_output

        # Parse detections
        preds = postprocessed[0]  # (max_det, 6): x1, y1, x2, y2, conf, class
        mask = preds[:, 4] >= self.conf_thres
        preds = preds[mask].cpu().float().numpy()

        # ─── Depth from SAME backbone features (if depth checkpoint loaded) ───
        depth_map_np = None
        if self.depth_adapter is not None:
            if self.adapter_type == "convnext":
                depth_adapted = self.depth_adapter(backbone_features)
            else:
                # ViTDepthAdapter handles everything internally:
                # extraction, fusion, scale conversion, detail branch, bifusion
                depth_adapted = self.depth_adapter(backbone_features, img_tensor)
            depth_pred = self.depth_decoder(depth_adapted["p3"], depth_adapted["p4"], depth_adapted["p5"])
            depth_map_np = depth_pred[0, 0].cpu().numpy()  # (imgsz, imgsz)

        # Scale boxes from imgsz back to original frame size
        scale_x = w0 / self.imgsz
        scale_y = h0 / self.imgsz

        detections = []
        for p in preds:
            x1, y1, x2, y2, conf, cls_id = p
            cls_id = int(cls_id)

            # Sample depth at bbox center (in imgsz coordinates)
            depth_m = None
            if depth_map_np is not None:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cx = max(0, min(cx, self.imgsz - 1))
                cy = max(0, min(cy, self.imgsz - 1))
                depth_m = float(depth_map_np[cy, cx])

            detections.append({
                "bbox": [
                    int(x1 * scale_x), int(y1 * scale_y),
                    int(x2 * scale_x), int(y2 * scale_y),
                ],
                "conf": float(conf),
                "class_id": cls_id,
                "class_name": self.class_names.get(cls_id, f"cls_{cls_id}"),
                "depth_m": depth_m,
            })

        # Resize depth map to original frame size for visualization
        if depth_map_np is not None:
            depth_map_np = cv2.resize(depth_map_np, (w0, h0))

        return detections, depth_map_np


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw bounding boxes, labels, and depth on frame."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["conf"]
        cls_id = det["class_id"]
        name = det["class_name"]
        depth_m = det.get("depth_m")

        color = _COLORS[cls_id % len(_COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{name}: {conf:.2f}"
        if depth_m is not None:
            label += f" | {depth_m:.1f}m"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def run(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = SharedBackbonePipeline(
        adapter_checkpoint=args.adapter_checkpoint,
        adapter_type=args.adapter_type,
        yolo_weights=args.yolo_weights,
        backbone_name=args.backbone,
        imgsz=args.imgsz,
        conf_thres=args.conf_thres,
        device=device,
        depth_checkpoint=args.depth_checkpoint,
    )

    # Open video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        logger.error("Cannot open source: %s", args.source)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Source: %s (%dx%d, %.1f FPS, %d frames)", args.source, w, h, fps, total_frames)

    # Video writer
    writer = None
    if args.save_video:
        out_path = Path(args.output_dir) / f"shared_backbone_output.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        logger.info("Saving output to: %s", out_path)

    frame_count = 0
    total_time = 0
    max_frames = args.frames if args.frames > 0 else float("inf")

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        detections, depth_map = pipeline.predict(frame)
        t1 = time.time()
        total_time += (t1 - t0)
        frame_count += 1

        # Draw detections + depth labels
        annotated = draw_detections(frame.copy(), detections)

        # FPS overlay
        cur_fps = frame_count / total_time if total_time > 0 else 0
        cv2.putText(
            annotated,
            f"FPS: {cur_fps:.1f} | Dets: {len(detections)} | Frame: {frame_count}/{total_frames}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

        if writer:
            writer.write(annotated)

        if not args.no_display:
            cv2.imshow("SentryMode — Shared Backbone", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

        if frame_count % 50 == 0:
            logger.info(
                "Frame %d/%d — %d detections — %.1f FPS",
                frame_count, total_frames, len(detections), cur_fps,
            )

    cap.release()
    if writer:
        writer.release()
    if not args.no_display:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # No GUI support (WSL/headless)

    avg_fps = frame_count / total_time if total_time > 0 else 0
    logger.info("Done: %d frames, %.1f avg FPS, %.1fs total", frame_count, avg_fps, total_time)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SentryMode — Shared Backbone Inference")
    parser.add_argument("--source", type=str, required=True, help="Path to video or image")
    parser.add_argument(
        "--adapter-checkpoint", type=str, default="runs/adapter_round2/adapter_best.pt",
        help="Path to trained adapter checkpoint (.pt file)",
    )
    parser.add_argument(
        "--adapter-type", type=str, default="auto", choices=["auto", "convnext", "vit"],
        help="Adapter type: 'auto' (detect from checkpoint metadata), "
             "'convnext' (FPNChannelAdapter, 542K, for ConvNeXt backbone), "
             "'vit' (ViTYOLOAdapter, 577K, for DINOv2/v3 ViT backbone). "
             "If set explicitly, validates against checkpoint metadata and errors on mismatch.",
    )
    parser.add_argument(
        "--yolo-weights", type=str, default="runs/detect/runs/baseline/yolo26s_fisheye8k/weights/best.pt",
        help="Path to YOLO26 weights for neck+head (fisheye-tuned or COCO)",
    )
    parser.add_argument(
        "--backbone", type=str, default="dinov3_convnext_tiny",
        help="Backbone name (only used when --adapter-type=convnext). "
             "For --adapter-type=vit, backbone is auto-selected from checkpoint metadata.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--frames", type=int, default=0, help="Max frames to process (0 = all)")
    parser.add_argument("--no-display", action="store_true", help="Disable live display window")
    parser.add_argument("--save-video", action="store_true", help="Save annotated output video")
    parser.add_argument("--output-dir", type=str, default="output/", help="Output directory for saved video")
    parser.add_argument(
        "--depth-checkpoint", type=str, default=None,
        help="Path to depth adapter+decoder checkpoint (depth_best.pt). "
             "If provided, depth estimation runs from the SAME backbone pass as detection. "
             "Each detection gets a depth value sampled at bbox center.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
