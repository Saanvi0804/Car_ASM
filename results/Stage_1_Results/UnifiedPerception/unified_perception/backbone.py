"""
backbone — shared DINOv2 feature extractor for the sentry-mode perception pipeline.

Runs the frozen DINOv2 backbone ONCE per video, returning a BackboneOutput
(see perception_types.py) that all downstream heads consume. This is the
"call DINOv2 once" architecture — subsequent heads (detection, action,
alert, etc.) never re-run the backbone.

Design
------
- Model loaded lazily and cached at module level (singleton per size).
- Backbone stays frozen; no gradients.
- Extracts CLS + patch tokens from the last layer AND all requested
  intermediate layer hidden states. Heads pick what they need.
- Frames are the standard DINOv2 preprocessing: resize→center crop→normalize.

CLI
---
    python backbone.py --source render.mp4 --num-frames 8 --out /tmp/feats.pt

Programmatic
------------
    from skills.backbone import extract_features
    feats = extract_features(mp4_path, num_frames=8)  # BackboneOutput

Backbone alignment (as of 2026-07-03)
-------------------------------------
Default is ``vits14`` (DINOv2 ViT-S/14, 384-dim). This matches
Pradipt's Adapter-v2 checkpoint AND the retrained action-recog head.
Both heads consume features from the same backbone forward pass.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .perception_types import BackboneOutput


# =============================================================================
# Model cache (singleton per model_name)
# =============================================================================

_BACKBONE_CACHE: dict[str, Any] = {}


def _hf_name(model_name: str) -> str:
    """Resolve short size (vits14, vitb14, vitl14, vitg14) to HuggingFace repo id."""
    aliases = {
        "vits14": "facebook/dinov2-small",
        "vitb14": "facebook/dinov2-base",
        "vitl14": "facebook/dinov2-large",
        "vitg14": "facebook/dinov2-giant",
    }
    if model_name in aliases:
        return aliases[model_name]
    if model_name.startswith("facebook/") or "/" in model_name:
        return model_name
    raise ValueError(f"Unknown DINOv2 backbone: {model_name}")


def _load_backbone(model_name: str = "vits14", device: str = "cuda"):
    """Lazy-load + cache the DINOv2 backbone. Frozen, eval mode."""
    import torch
    from transformers import AutoModel

    key = f"{model_name}:{device}"
    if key in _BACKBONE_CACHE:
        return _BACKBONE_CACHE[key]

    hf_id = _hf_name(model_name)
    print(f"[backbone] Loading {model_name} ({hf_id}) on {device}", file=sys.stderr)
    model = AutoModel.from_pretrained(hf_id).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    _BACKBONE_CACHE[key] = model
    return model


# =============================================================================
# Frame sampling + preprocessing
# =============================================================================

def _sample_frames(
    mp4_path: str, num_frames: int, image_size: int = 640, stride: int = 1
):
    """Uniformly sample num_frames from an mp4, return BOTH raw and normalized frames.

    Returns:
        frames_raw:  [T, C, image_size, image_size] float32 in [0, 1]
                     (matches Pradipt's `img_tensor` — for the adapter's
                     detail branch, which needs low-level pixel values).
        frames_norm: [T, C, image_size, image_size] float32, ImageNet-normalized
                     (for the DINOv2 backbone forward).

    - num_frames = -1 : use ALL frames (stride applies).
    - num_frames > 0  : uniformly sample num_frames indices.
    - stride > 1      : subsamples by stride BEFORE selecting indices.
    """
    import numpy as np
    import torch
    import decord
    from torchvision import transforms

    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(mp4_path, num_threads=1)
    total = len(vr)

    if num_frames == -1:
        indices = np.arange(0, total, stride)
    elif total < num_frames:
        indices = np.arange(total)
        indices = np.pad(indices, (0, num_frames - total), mode="edge")
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).permute(0, 3, 1, 2)  # [T, C, H, W] uint8

    # Resize (squash) + convert-to-float. This gives us the "raw [0,1]"
    # frames that the ViT-adapter's detail branch expects (Pradipt line 306).
    raw_tfm = transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ConvertImageDtype(torch.float32),
    ])
    frames_raw = torch.stack([raw_tfm(f) for f in frames])

    # Normalize on top of raw for the DINOv2 backbone forward.
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    frames_norm = torch.stack([normalize(f) for f in frames_raw])

    return frames_raw, frames_norm


# =============================================================================
# Main API
# =============================================================================

def extract_features(
    mp4_path: str,
    num_frames: int = 8,
    model_name: str = "vits14",
    device: str = "cuda",
    image_size: int = 640,   # Match Pradipt's adapter (built at imgsz=640)
    hidden_layer_indices: list[int] | None = None,
    keep_frames: bool = False,
    stride: int = 1,
) -> BackboneOutput:
    """Extract DINOv2 features from a single mp4.

    Parameters
    ----------
    mp4_path             : str
        Path to the input mp4.
    num_frames           : int
        Uniformly sample this many frames from the video (default: 8).
    model_name           : str
        Which DINOv2 to use — vits14/vitb14/vitl14/vitg14 or a HF repo id.
    device               : str
        "cuda" or "cpu". Default cuda.
    image_size           : int
        Square DINOv2 preprocessing size. This package supports 640 (default,
        matches the detection adapter; DINOv2 patch grid 45x45) or 644 (a clean
        multiple of the ViT/14 patch size -> 46x46 grid). See ALLOWED_IMAGE_SIZES
        in orchestrator.py; the detection adapter is baked for 640.
    hidden_layer_indices : list[int] | None
        Which intermediate layer indices to keep. If None, keeps ALL layers
        (memory permitting). Pradipt's Adapter-v2 wants indices [3, 6, 9, 12]
        for ViT-S/14.
    keep_frames          : bool
        If True, keep the preprocessed frames tensor in the BackboneOutput.
        Useful for heads that need pixel-level input (e.g. rendering
        annotated overlays). Increases memory.

    Returns
    -------
    BackboneOutput
    """
    import torch

    model = _load_backbone(model_name, device)
    frames_raw, frames_norm = _sample_frames(mp4_path, num_frames, image_size, stride=stride)
    frames_raw = frames_raw.to(device)
    frames_norm = frames_norm.to(device)

    with torch.no_grad():
        out = model(frames_norm, output_hidden_states=True)
    # out.last_hidden_state: [T, N+1, D]  (N patches + 1 CLS)
    # out.hidden_states:     tuple of length num_layers+1 (embedding + each layer)
    lhs = out.last_hidden_state
    cls = lhs[:, 0]                        # [T, D]
    patch_tokens = lhs[:, 1:]              # [T, N, D]

    # Collect requested (or all) hidden states
    if hidden_layer_indices is None:
        hidden_states = {i: h for i, h in enumerate(out.hidden_states)}
    else:
        hidden_states = {
            i: out.hidden_states[i]
            for i in hidden_layer_indices
            if 0 <= i < len(out.hidden_states)
        }

    # Move to CPU to keep GPU memory clean for downstream heads
    cls = cls.detach().cpu()
    patch_tokens = patch_tokens.detach().cpu()
    hidden_states = {i: h.detach().cpu() for i, h in hidden_states.items()}

    # Source frame rate — used by the action head's sliding-window mode to
    # subsample to a target fps. Best-effort; 0.0 if unavailable.
    try:
        import decord as _decord
        fps = float(_decord.VideoReader(mp4_path, num_threads=1).get_avg_fps())
    except Exception:
        fps = 0.0

    metadata = {
        "source":         mp4_path,
        "model_name":     model_name,
        "hf_id":          _hf_name(model_name),
        "num_frames":     num_frames,
        "image_size":     image_size,
        "feature_dim":    int(cls.shape[-1]),
        "num_patches":    int(patch_tokens.shape[1]),
        "num_layers":     len(out.hidden_states),
        "fps":            fps,
        "stride":         int(stride),   # CLS index i -> raw frame i*stride
    }

    return BackboneOutput(
        cls=cls,
        hidden_states=hidden_states,
        patch_tokens=patch_tokens,
        metadata=metadata,
        # frames = RAW [0,1] tensor (unnormalized). This is what Pradipt's
        # ViT-adapter detail branch consumes (see run_shared_backbone.py:316
        # — he passes `img_tensor` NOT `img_norm` to the adapter).
        frames=(frames_raw.detach().cpu() if keep_frames else None),
    )


# =============================================================================
# CLI
# =============================================================================

def _cli():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--source", required=True, help="Input mp4")
    p.add_argument("--num-frames", type=int, default=8)
    p.add_argument("--model-name", default="vits14",
                   help="vits14 (default) | vitb14 | vitl14 | vitg14 | facebook/...")
    p.add_argument("--device", default="cuda")
    p.add_argument("--layers", nargs="*", type=int, default=None,
                   help="Optional: only keep these hidden layer indices "
                        "(e.g. --layers 3 6 9 12 for Pradipt's adapter).")
    p.add_argument("--keep-frames", action="store_true",
                   help="Include the preprocessed frames tensor in the output.")
    p.add_argument("--out", help="Optional .pt path to save the BackboneOutput dict")
    args = p.parse_args()

    if not Path(args.source).exists():
        sys.exit(f"[backbone] source not found: {args.source}")

    feats = extract_features(
        args.source,
        num_frames=args.num_frames,
        model_name=args.model_name,
        device=args.device,
        hidden_layer_indices=args.layers,
        keep_frames=args.keep_frames,
    )
    print(f"[backbone] cls shape:       {tuple(feats.cls.shape)}")
    print(f"[backbone] patch_tokens:    {tuple(feats.patch_tokens.shape)}")
    print(f"[backbone] hidden_states:   {list(feats.hidden_states.keys())}")
    print(f"[backbone] feature_dim:     {feats.metadata['feature_dim']}")
    print(f"[backbone] backbone:        {feats.metadata['hf_id']}")

    if args.out:
        import torch
        torch.save(feats.__dict__, args.out)
        print(f"[backbone] saved -> {args.out}")


if __name__ == "__main__":
    _cli()
