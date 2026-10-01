"""
cli — command-line entry point for the UnifiedPerception package.

Run with:
    python -m unified_perception --source clip.mp4 --output-dir /tmp/out
    python -m unified_perception --render-dir render_dir --output-dir /tmp/out
or, if installed, via the `unified-perception` console script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .orchestrator import (
    ALLOWED_IMAGE_SIZES,
    analyze,
    analyze_render_dir,
    _summarize_primary,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="unified_perception",
        description="UnifiedPerception: shared DINOv2 backbone -> object "
                    "detection + whole-scene action recognition.",
    )
    src_grp = p.add_mutually_exclusive_group(required=True)
    src_grp.add_argument("--source", help="Single mp4")
    src_grp.add_argument("--render-dir", help="Multi-cam render dir")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-frames", type=int, default=-1,
                   help="Uniformly sample this many frames. -1 (default) = all frames "
                        "(with stride). The action head subsamples internally.")
    p.add_argument("--stride", type=int, default=1,
                   help="Frame stride when num_frames=-1. stride=1 = every frame, "
                        "stride=6 = every 6th frame (~5fps from 30fps).")
    p.add_argument("--model-name", default="vits14",
                   help="DINOv2 size: vits14 (default) | vitb14 | vitl14 | vitg14 | facebook/...")
    p.add_argument("--image-size", type=int, default=640, choices=list(ALLOWED_IMAGE_SIZES),
                   help="Square input dimension for the shared DINOv2 forward. "
                        "640 (default) matches the detection adapter; 644 gives a clean "
                        "46x46 DINOv2 patch grid for the action head (the detection "
                        "adapter is baked for 640 and will report an error at 644).")
    p.add_argument("--skip", nargs="*", action="append", default=None,
                   help="Head names to skip (detection | action). Accepts both "
                        "'--skip detection action' AND '--skip detection --skip action'.")
    p.add_argument("--no-grid", action="store_true", help="Don't stitch 2x2 grid")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    # --skip uses append+nargs=* -> list-of-lists (or None). Flatten so both
    # `--skip a b` and `--skip a --skip b` accumulate correctly.
    skip = [h for grp in (args.skip or []) for h in grp]

    if args.source:
        result = analyze(
            args.source, args.output_dir,
            num_frames=args.num_frames, model_name=args.model_name,
            skip=skip, stride=args.stride, image_size=args.image_size,
        )
    else:
        result = analyze_render_dir(
            args.render_dir, args.output_dir,
            num_frames=args.num_frames, model_name=args.model_name,
            skip=skip, stitch_grid=not args.no_grid, stride=args.stride,
            image_size=args.image_size,
        )

    # Compact summary — full per-frame details live in perception.json.
    summary = {
        "action_summary": result.get("action_summary"),
        "grid_mp4":       result.get("grid_mp4"),
        "combined_mp4":   result.get("combined_mp4"),
    }
    if "heads" in result:
        summary["heads"] = {
            name: {
                "summary":    _summarize_primary(name, r.get("primary"), r.get("details")),
                "confidence": r.get("confidence"),
                "error":      r.get("error"),
            }
            for name, r in result["heads"].items()
        }
    actual_out = result.get("output_dir") or args.output_dir
    if args.source:
        summary["details_json"] = str(Path(actual_out) / "perception.json")
    else:
        summary["details_json"] = str(Path(actual_out) / "perception_all.json")
    summary["output_dir"] = actual_out
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
