"""
UnifiedPerception — a standalone package for the shared-DINOv2 perception flow.

One backbone forward per clip, then two heads:
    - detection : Pradipt's Adapter-v2 + YOLO26 object detector (unchanged).
    - action    : the whole-scene DINOv2-CLS action recognizer.

Supported input dimensions: 640x640 (default) or 644x644 (ALLOWED_IMAGE_SIZES).

Quick start
-----------
    from unified_perception import analyze
    result = analyze("clip.mp4", "/tmp/out", image_size=640)
"""
from __future__ import annotations


def _ensure_local_env() -> None:
    """Keep ALL runtime state inside this folder — no dependency on sibling
    directories or a shared home cache. Safe to call repeatedly; only fills in
    values the caller hasn't already set.

    - HF_HOME / TORCH_HOME / YOLO_CONFIG_DIR -> <root>/.cache/* so any model
      downloads (DINOv2, YOLO settings) land inside the package folder.
    - ffmpeg: if no system binary is on PATH, expose the one bundled by the
      imageio-ffmpeg wheel (installed by setup.sh) as `ffmpeg` in <root>/.cache/bin.
    """
    import os
    import shutil
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cache = root / ".cache"
    os.environ.setdefault("HF_HOME", str(cache / "hf"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(cache / "ultralytics"))

    # Create the cache dirs so downstream libs (HF, torch.hub, ultralytics) write
    # in-folder instead of falling back to a system/tmp location.
    for _var in ("HF_HOME", "TORCH_HOME", "YOLO_CONFIG_DIR"):
        try:
            Path(os.environ[_var]).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    if shutil.which("ffmpeg") is None:
        try:
            import imageio_ffmpeg  # bundled static ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            bindir = cache / "bin"
            bindir.mkdir(parents=True, exist_ok=True)
            link = bindir / "ffmpeg"
            if not link.exists():
                try:
                    link.symlink_to(exe)
                except Exception:
                    shutil.copy2(exe, link)
            os.environ["PATH"] = str(bindir) + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass  # overlays will simply be skipped if ffmpeg is truly unavailable


_ensure_local_env()

from .perception_types import BackboneOutput, HeadResult, Head
from .backbone import extract_features
from .heads import REGISTRY, get as get_head, list_heads
from .orchestrator import (
    ALLOWED_IMAGE_SIZES,
    analyze,
    analyze_render_dir,
)

__version__ = "0.1.0"

__all__ = [
    "analyze",
    "analyze_render_dir",
    "extract_features",
    "REGISTRY",
    "get_head",
    "list_heads",
    "BackboneOutput",
    "HeadResult",
    "Head",
    "ALLOWED_IMAGE_SIZES",
    "__version__",
]
