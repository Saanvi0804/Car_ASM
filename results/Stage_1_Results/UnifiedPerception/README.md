# UnifiedPerception — Usage Guide

A **fully self-contained** perception package. One DINOv2 backbone forward per
clip feeds two heads:

- **`detection`** — Adapter-v2 + YOLO26 object detector (vendored).
- **`action`** — whole-scene DINOv2-CLS action recognizer (K400 labels).

Everything it needs (code, checkpoints, even the virtual environment) lives
inside this folder. Nothing outside `UnifiedPerception/` is required at runtime.

---

## 1. Requirements

1. Linux with an **NVIDIA GPU + driver** (the pinned wheels are CUDA 12.4; the
   heads load on `cuda`, so a GPU is required).
2. `python3` available (validated on Python 3.12).
3. Internet access **once**, only for `setup.sh` (downloads pip wheels + the
   DINOv2 weights into the local cache). After that it runs offline.
4. No system `ffmpeg` needed — a static one is installed into the venv.

---

## 2. One-time setup

1. Enter the folder:
   ```bash
   cd UnifiedPerception
   ```
2. Build the self-contained environment:
   ```bash
   bash setup.sh
   ```
   This creates `./.venv`, installs the pinned deps from `requirements.txt`,
   installs the package (`pip install -e .`), points all caches at `./.cache/`,
   and pre-downloads `facebook/dinov2-small`.
   - Pick a specific interpreter: `PYTHON_BIN=python3.12 bash setup.sh`
   - Skip the DINOv2 pre-fetch: `SKIP_PREDOWNLOAD=1 bash setup.sh`
3. Activate the environment (required before every run):
   ```bash
   source .venv/bin/activate
   ```

---

## 3. Run on a single clip

1. Basic run (default input size 640×640):
   ```bash
   python -m unified_perception --source /path/to/clip.mp4 --output-dir out/clip1
   ```
2. Watch the terminal: it prints a compact JSON summary (top action label,
   detection counts, and the path to the full `perception.json`).
3. Find results in the output dir (see section 6).

---

## 4. Run on a multi-cam render directory

1. Point `--render-dir` at a folder of camera mp4s (auto-finds
   `scn_*/camera_*fisheye*.mp4`, else `camera_*.mp4`):
   ```bash
   python -m unified_perception --render-dir /path/to/render_dir --output-dir out/scene1
   ```
2. Each camera is analyzed, then the annotated clips are stitched into a 2×2
   grid (`annotated_grid.mp4`). Disable stitching with `--no-grid`.

---

## 5. Common options

1. `--image-size {640,644}` — square input size (see section 8). Default `640`.
2. `--skip action` — run **detection only**.
3. `--skip detection` — run **action only**.
4. `--model-name vitb14|vitl14|vitg14` — larger DINOv2 backbone (default `vits14`).
5. `--num-frames N` — sample exactly N frames (default `-1` = all frames, with stride).
6. `--stride K` — take every K-th frame when `--num-frames -1` (e.g. `--stride 6` ≈ 5 fps from 30 fps).
7. `--no-grid` — skip the 2×2 grid stitch (render-dir mode only).

Full flag list: `python -m unified_perception --help`.

---

## 6. What you get in `--output-dir`

Single-clip run:

1. `perception.json` — full result: backbone metadata + per-head output
   (detection boxes per frame, action top-k, confidences).
2. `detection_annotated.mp4` — boxes drawn on the source video.
3. `action_annotated.mp4` — top-1 action label banner on the source video.
4. `annotated_with_label.mp4` — combined overlay (boxes + action label), when both heads succeed.

Render-dir run:

5. `<cam>/…` — one sub-folder per camera with the files above.
6. `perception_all.json` — per-cam summary + action-per-cam.
7. `annotated_grid.mp4` — 2×2 stitched overlay.

> Note: if the output dir already has content, a timestamp suffix is appended so
> prior runs are never overwritten. Set `REUSE_OUTPUT_DIR=1` to write in place.

---

## 7. Use it from Python

1. Activate the venv, then:
   ```python
   from unified_perception import analyze

   result = analyze("clip.mp4", "out/clip1", image_size=640)
   print("action:", result["heads"]["action"]["primary"])
   print("json:  ", result["output_dir"] + "/perception.json")
   ```
2. Multi-cam: `from unified_perception import analyze_render_dir`.
3. Inspect available heads: `from unified_perception import REGISTRY` → `["detection", "action"]`.

---

## 8. Input dimension: 640 vs 644

1. `--image-size 640` (default) — DINOv2 patch grid 45×45. **Matches the
   detection adapter**, so the full pipeline (detection + action) works.
2. `--image-size 644` — `644 = 14 × 46`, a clean multiple of the ViT/14 patch
   size → 46×46 grid, better for the action/CLS head.
3. Caveat: the detection adapter is baked for 640, so at **644 the detection
   head reports an error** (non-fatal — it's caught and the action head still
   runs). Use 644 for action-focused runs; keep 640 when you need boxes.

---

## 9. Configuration (all optional)

Defaults already resolve inside this folder. Override only to point at an
external resource.

| Variable | Default (in-folder) | Purpose |
|----------|---------------------|---------|
| `SENTRY_TRAINING_PACKAGE` | `third_party/training_package` | Detection code + weights |
| `SENTRY_ACTION_CKPT` | `assets/k400_outputs/best.pth` | Action head checkpoint (K400 linear probe) |
| `SENTRY_CLASSES_TXT` | `assets/kinetics400/annotations/classes.txt` | K400 class list |
| `HF_HOME` / `TORCH_HOME` / `YOLO_CONFIG_DIR` | `.cache/*` | Model/download caches |
| `SENTRY_DETECTION_MODE` | `shared` | `shared` (in-process) or `subprocess` |
| `SENTRY_ACTION_SLIDING` | `0` | `1` = sliding-window action prediction |
| `REUSE_OUTPUT_DIR` | `0` | `1` = write in place (no timestamp suffix) |

Example — swap the action checkpoint for one run:
```bash
SENTRY_ACTION_CKPT=/abs/path/other_best.pth \
python -m unified_perception --source clip.mp4 --output-dir out/x
```

---

## 10. Troubleshooting

1. **`ModuleNotFoundError` / wrong Python** — you forgot `source .venv/bin/activate`.
2. **CUDA / torch install issues in `setup.sh`** — ensure the host has a CUDA 12.4
   capable driver; otherwise edit the `--extra-index-url` in `requirements.txt`
   to a matching wheel (e.g. `cpu`).
3. **Detection errors at `--image-size 644`** — expected; the adapter is 640-only.
   Re-run at 640 for boxes, or use `--skip detection`.
4. **First run is slow / needs network** — that's the one-time DINOv2 download
   into `./.cache/hf`. Pre-do it in `setup.sh` (default) so later runs are offline.
5. **No overlays produced** — a head errored (check `perception.json` → the head's
   `error` field) or the clip had no detections/action label.

---

## 11. Folder map

```
UnifiedPerception/
├── setup.sh / requirements.txt / pyproject.toml   # env + packaging
├── .venv/  .cache/                                # created by setup.sh (local)
├── assets/                                        # vendored checkpoints
│   ├── k400_outputs/best.pth
│   └── kinetics400/annotations/classes.txt
├── third_party/                                   # vendored external code
│   └── training_package/{sentry,scripts,runs,yolo26s.pt,configs}
└── unified_perception/                            # the importable package
    ├── __init__.py __main__.py cli.py
    ├── orchestrator.py backbone.py perception_types.py
    └── heads/{__init__.py, detection.py, action.py}
```
