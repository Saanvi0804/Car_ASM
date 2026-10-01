"""
orchestrator — the UnifiedPerception flow (standalone package).

Single entry point: `analyze(mp4)` runs the shared DINOv2 backbone ONCE and
dispatches to the two registered heads — object `detection` (unchanged) and the
whole-scene `action` recognizer. Returns a single JSON with per-head results.

For multi-cam renders (a directory with N cam mp4s), use
`analyze_render_dir(render_dir)` — it runs each cam through analyze() then
optionally stitches the annotated mp4s into a 2x2 grid.

Design
------
- Backbone runs ONCE per input mp4, at image_size 640 or 644 (ALLOWED_IMAGE_SIZES).
- Heads registered in `heads/__init__.py::REGISTRY` are called sequentially.
- One head failing does NOT block the others (its result gets an error field).
- Optional per-head annotation + a combined "boxes + action label" overlay mp4.

CLI (see cli.py)
----------------
    # Single mp4
    python -m unified_perception --source /path/to/render.mp4 \\
        --output-dir /tmp/perception/one_clip --image-size 640

    # Multi-cam render dir (auto-finds fisheye cams, stitches 2x2 grid)
    python -m unified_perception --render-dir /path/to/render_dir \\
        --output-dir /tmp/perception/compound

    # Clean 46x46 DINOv2 patch grid for the action head:
    python -m unified_perception --source X.mp4 --output-dir /tmp/o --image-size 644

Programmatic
------------
    from unified_perception import analyze
    result = analyze("wave.mp4", "/tmp/out", image_size=640)
    # result = {
    #     "source":    "wave.mp4",
    #     "backbone":  {model_name, num_frames, feature_dim, image_size, ...},
    #     "heads": {
    #         "detection": HeadResult.to_json(),
    #         "action":    HeadResult.to_json(),
    #     },
    #     "combined_mp4": "/tmp/out/annotated_with_label.mp4",  # if made
    # }
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .backbone import extract_features
from .heads import REGISTRY
from .perception_types import BackboneOutput, HeadResult

# Supported square input dimensions for the shared DINOv2 forward.
#   640 -> matches the detection adapter (DINOv2 ViT/14 patch grid 45x45).
#   644 -> clean multiple of the ViT/14 patch size -> 46x46 grid (cleaner CLS for
#          the whole-scene action head; the detection adapter is baked for 640, so
#          at 644 the detection head reports an error while action still runs).
ALLOWED_IMAGE_SIZES = (640, 644)


def _validate_image_size(image_size: int) -> int:
    if image_size not in ALLOWED_IMAGE_SIZES:
        raise ValueError(
            f"image_size must be one of {ALLOWED_IMAGE_SIZES}, got {image_size}"
        )
    return image_size


# =============================================================================
# Summary helpers (keep terminal output compact — full results go to JSON)
# =============================================================================

def _summarize_primary(head_name: str, primary, details=None) -> str:
    """One-line summary of a head's primary. Detection primary is per-frame
    list-of-lists; collapse to counts. Action primary is a string label."""
    if primary is None:
        return "primary=None"
    if head_name == "detection" and isinstance(primary, list):
        n_frames = len(primary)
        total = sum(len(f) for f in primary if isinstance(f, list))
        classes: dict[str, int] = {}
        for f in primary:
            if not isinstance(f, list):
                continue
            for d in f:
                c = d.get("class_name", "?")
                classes[c] = classes.get(c, 0) + 1
        top = ", ".join(f"{c}={n}" for c, n in sorted(
            classes.items(), key=lambda x: -x[1])[:5])
        return f"frames={n_frames} dets={total} [{top}]"
    base = f"primary={primary}"
    if details and details.get("freq_topN"):
        top = ", ".join(f"{l} x{c}" for l, c, *_ in details["freq_topN"])
        base += f" | top{len(details['freq_topN'])} by freq: {top}"
    return base


# =============================================================================
# Single-clip analysis
# =============================================================================

def _unique_output_dir(output_dir: str) -> Path:
    """Collision-safe output directory.

    If `output_dir` already contains perception artifacts (perception.json or
    any camera sub-dir), auto-append a timestamp suffix so identical NL
    requests don't overwrite prior runs.

    Opt out with env var REUSE_OUTPUT_DIR=1 (for the "restart in-place" case).
    """
    import os as _os
    from datetime import datetime as _dt
    p = Path(output_dir)
    if _os.environ.get("REUSE_OUTPUT_DIR", "0") == "1":
        return p
    if p.exists() and p.is_dir():
        # Heuristic: any file OR any child dir means this path was used before
        try:
            has_content = any(p.iterdir())
        except Exception:
            has_content = False
        if has_content:
            suffix = _dt.now().strftime("%Y%m%d_%H%M%S")
            new = p.with_name(f"{p.name}_{suffix}")
            print(f"[unified_perception] output_dir already has content; using: {new}",
                  file=sys.stderr)
            return new
    return p


def analyze(
    mp4: str,
    output_dir: str,
    num_frames: int = -1,  # -1 = all frames (with stride); good for per-frame detection.
    model_name: str = "vits14",
    skip: Optional[list[str]] = None,
    stride: int = 1,       # subsample by stride when num_frames=-1
    reuse_output_dir: bool = False,  # skip uniqueness check (caller already handled it)
    image_size: int = 640,  # square input dim for the shared DINOv2 forward: 640 or 644
) -> dict:
    """Extract features once, run all registered heads (detection + action).

    Parameters
    ----------
    mp4         : str          input mp4 path
    output_dir  : str          where per-head artifacts (annotated mp4s, JSON) land
    num_frames  : int          how many frames to sample (default 8)
    model_name  : str          backbone size (default vits14, aligned across heads)
    skip        : list[str]    head names to skip, e.g. ["action"] to run only detection
    image_size  : int          square input dimension, one of ALLOWED_IMAGE_SIZES
                               (640 or 644). 640 keeps the detection adapter happy;
                               644 gives a clean 46x46 DINOv2 patch grid.
    """
    image_size = _validate_image_size(image_size)
    src = Path(mp4)
    if not src.exists():
        raise FileNotFoundError(mp4)
    out = Path(output_dir) if reuse_output_dir else _unique_output_dir(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    skip = set(skip or [])

    # 1. Single backbone forward (keep_frames=True so detection's adapter
    #    detail branch can consume raw pixels)
    feats = extract_features(
        mp4, num_frames=num_frames, model_name=model_name,
        keep_frames=True, stride=stride, image_size=image_size,
    )
    print(f"[unified_perception] backbone done (image_size={image_size}): {feats.metadata}",
          file=sys.stderr)

    # 2. Run each head — in registry order (detection, then action). Heads may
    # accept a `prior_results` kwarg to read earlier heads' outputs; heads that
    # don't (like this package's detection + action) are called single-arg.
    per_head_results: dict[str, dict] = {}
    for name, head in REGISTRY.items():
        if name in skip:
            per_head_results[name] = {"head_name": name, "error": "skipped"}
            continue
        try:
            # Backward-compat: try prior_results kwarg first; fall back to
            # single-arg call for heads written before Perception V2.
            try:
                result = head.predict(feats, prior_results=per_head_results)
            except TypeError:
                result = head.predict(feats)
        except Exception as e:
            result = HeadResult(head_name=name, error=f"predict raised: {e!r}")
        # Give heads a chance to produce their own annotated mp4. Detection's
        # annotate takes image_size (to invert the square preprocessing back to
        # source coords); other heads don't — fall back to the 3-arg form.
        ann_target = out / f"{name}_annotated.mp4"
        if hasattr(head, "annotate"):
            try:
                try:
                    ann = head.annotate(mp4, result, str(ann_target), image_size=image_size)
                except TypeError:
                    ann = head.annotate(mp4, result, str(ann_target))
                if ann:
                    result.annotated_mp4 = ann
            except Exception as e:
                print(f"[unified_perception] {name} annotate raised: {e!r}",
                      file=sys.stderr)

        per_head_results[name] = result.to_json()
        # Compact per-head summary (no giant primary dumps)
        primary_summary = _summarize_primary(name, result.primary, result.details)
        conf_str = f"{result.confidence:.3f}" if result.confidence is not None else "n/a"
        print(f"[unified_perception] {name}: {primary_summary}, conf={conf_str}",
              file=sys.stderr)

    # 3. Optional: combine annotations (burn action label onto detection mp4)
    combined_mp4 = _maybe_combine_overlays(mp4, per_head_results, out)

    result_json = {
        "source": mp4,
        "backbone": feats.metadata,
        "heads": per_head_results,
        "combined_mp4": combined_mp4,
        "output_dir": str(out),   # actual path used (may differ from arg after uniquify)
    }
    (out / "perception.json").write_text(json.dumps(result_json, indent=2, default=str))
    return result_json


# =============================================================================
# Multi-cam render dir
# =============================================================================

def _find_cameras(render_dir: str) -> list[Path]:
    rd = Path(render_dir)
    cams = sorted(rd.glob("scn_*/camera_*fisheye*.mp4"))
    if not cams:
        cams = sorted(rd.glob("camera_*.mp4"))
    return cams


def analyze_render_dir(
    render_dir: str,
    output_dir: str,
    num_frames: int = -1,
    model_name: str = "vits14",
    skip: Optional[list[str]] = None,
    stitch_grid: bool = True,
    stride: int = 1,
    image_size: int = 640,
) -> dict:
    """Run analyze() on every camera mp4 in a render dir, optionally stitch grid."""
    image_size = _validate_image_size(image_size)
    cams = _find_cameras(render_dir)
    if not cams:
        raise FileNotFoundError(f"no camera mp4s under {render_dir}")
    print(f"[unified_perception] {len(cams)} cams in {render_dir}", file=sys.stderr)

    # Collision-safe output: if the requested dir already has perception artifacts,
    # auto-append a timestamp suffix so identical NL requests don't overwrite.
    out_root = _unique_output_dir(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    per_cam = []
    for c in cams:
        cam_name = c.stem.replace("camera_", "").replace("_fisheye_200fov", "")
        # Pass reuse=True so the per-cam sub-dirs don't get another timestamp
        # suffix — we already uniquified the parent.
        cam_out = out_root / cam_name
        per_cam.append({
            "cam": cam_name,
            "result": analyze(str(c), str(cam_out),
                              num_frames=num_frames, model_name=model_name,
                              skip=skip, stride=stride,
                              reuse_output_dir=True, image_size=image_size),
        })

    grid_mp4 = ""
    if stitch_grid:
        annotated = []
        labels = []
        for pc in per_cam:
            # Prefer combined_mp4 (has detection + action label), else detection's annotated
            m = pc["result"].get("combined_mp4") or \
                pc["result"].get("heads", {}).get("detection", {}).get("annotated_mp4")
            if m and Path(m).exists():
                annotated.append(m)
                labels.append(pc["cam"].upper())
        if len(annotated) >= 2:
            grid_mp4 = str(out_root / "annotated_grid.mp4")
            _stitch_grid(annotated, grid_mp4, labels)

    action_summary = {
        pc["cam"]: pc["result"].get("heads", {}).get("action", {}).get("primary")
        for pc in per_cam
    }

    combined = {
        "render_dir":     render_dir,
        "per_cam":        per_cam,
        "grid_mp4":       grid_mp4,
        "action_summary": action_summary,
        "output_dir":     str(out_root),
    }
    (out_root / "perception_all.json").write_text(
        json.dumps(combined, indent=2, default=str)
    )
    return combined


# =============================================================================
# Helpers: combine per-head overlays, stitch grid
# =============================================================================

def _maybe_combine_overlays(
    src_mp4: str, per_head_results: dict, out: Path
) -> Optional[str]:
    """If detection produced an annotated mp4 AND the action head has a primary
    label, burn the whole-scene action label as a top banner and produce a
    single "combined" mp4.
    """
    det = per_head_results.get("detection") or {}
    ann = det.get("annotated_mp4")
    if not ann or not Path(ann).exists():
        return None

    act = per_head_results.get("action") or {}

    # Sliding-window action: burn a time-varying label from the per-window
    # timeline (label changes across the clip). Falls through to the single
    # static label below when there's no timeline.
    act_details = act.get("details") or {}
    timeline = act_details.get("timeline")
    src_fps = float(act_details.get("src_fps") or 0.0)
    if timeline and src_fps > 0 and not act.get("error"):
        seg = _segmented_label_overlay(ann, timeline, src_fps, out)
        if seg:
            return seg

    if act.get("primary") and not act.get("error"):
        label, conf, source = act["primary"], act.get("confidence"), "action"
    else:
        return ann  # detection-only overlay is still useful

    text = f"{source}: {label} ({conf:.2f})" if conf else f"{source}: {label}"
    combined = str(out / "annotated_with_label.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", ann,
        "-vf",
        f"drawtext=text='{text.replace(chr(39), '')}':x=(w-tw)/2:y=20:"
        f"fontsize=48:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=10",
        "-c:v", "libx264", "-crf", "20", combined,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[unified_perception] label overlay failed:\n{proc.stderr[-500:]}",
              file=sys.stderr)
        return ann
    return combined


def _segmented_label_overlay(ann: str, timeline: list, fps: float, out: Path) -> Optional[str]:
    """Burn a time-varying action label onto `ann` from the sliding-window
    timeline: each window's label shows from its center time until the next
    window's, merging consecutive identical labels. Single ffmpeg pass."""
    tl = sorted((t for t in timeline if t), key=lambda x: x[0])
    if not tl:
        return None
    segs: list = []
    for i, item in enumerate(tl):
        frame, label = item[0], item[1]
        t0 = float(frame) / fps
        t1 = (float(tl[i + 1][0]) / fps) if i + 1 < len(tl) else 1e9
        if segs and segs[-1][2] == label:
            segs[-1][1] = t1
        else:
            segs.append([t0, t1, label])

    def _esc(s):
        return str(s).replace("\\", "").replace("'", "").replace(":", r"\:")

    parts = [
        f"drawtext=text='{_esc('action ' + str(label))}':x=(w-tw)/2:y=20:"
        f"fontsize=44:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=10:"
        f"enable='between(t,{t0:.3f},{t1:.3f})'"
        for t0, t1, label in segs
    ]
    combined = str(out / "annotated_with_label.mp4")
    cmd = ["ffmpeg", "-y", "-i", ann, "-vf", ",".join(parts),
           "-c:v", "libx264", "-crf", "20", combined]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[unified_perception] segmented overlay failed:\n{proc.stderr[-500:]}",
              file=sys.stderr)
        return None
    return combined


def _stitch_grid(annotated_mp4s: list[str], out_mp4: str, labels: list[str]) -> str:
    """2x2 grid stitch with per-cam labels."""
    vids = annotated_mp4s[:4] + [annotated_mp4s[-1]] * max(0, 4 - len(annotated_mp4s))
    lbls = labels[:4] + [labels[-1]] * max(0, 4 - len(labels))

    parts = []
    for i, l in enumerate(lbls):
        parts.append(
            f"[{i}:v]drawtext=text='{l}':x=10:y=10:fontsize=32:"
            f"fontcolor=white:box=1:boxcolor=black@0.5[v{i}]"
        )
    parts += [
        "[v0][v1]hstack=inputs=2[top]",
        "[v2][v3]hstack=inputs=2[bot]",
        "[top][bot]vstack=inputs=2[out]",
    ]
    cmd = ["ffmpeg", "-y"]
    for v in vids:
        cmd += ["-i", v]
    cmd += ["-filter_complex", "; ".join(parts),
            "-map", "[out]", "-c:v", "libx264", "-crf", "23", out_mp4]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[unified_perception] grid stitch failed:\n{proc.stderr[-500:]}",
              file=sys.stderr)
        return ""
    return out_mp4


# The command-line interface lives in cli.py (run `python -m unified_perception`).
