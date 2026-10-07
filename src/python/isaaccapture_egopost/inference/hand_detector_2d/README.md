<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-OpenMDW-1.1
-->

# Hand boxes and the optional 2D detector

[← EgoHand README](../../README.md) · [Detector CLI reference](scripts/README.md)

EgoHand needs a 2D box for each hand you want to reconstruct. **RT-DETR is one
optional source of those boxes.** Use your preferred 2D detector or ground-truth
annotations by exporting the saved-box format below; no RT-DETR weights or
RT-DETR model access are needed for that path.

All commands run from `ego-reconstruction/EgoHand/`.

## Use your own detector or ground-truth boxes

Arrange your JSON files under a root directory, using the input video's stem
(filename without `.mp4`):

```text
bboxes/
└── my_video/
    ├── my_video_left_bbox.json
    └── my_video_right_bbox.json
```

Each file maps a zero-based, six-digit frame ID to a **square box in source-video
pixels**, ordered `[center_x, center_y, side_length]`:

```json
{
  "000000": [640.0, 360.0, 180.0],
  "000001": [642.0, 358.0, 182.0]
}
```

Use the anatomical left/right hand labels. Coordinates must match the decoded
input video, including any rotation or resize you applied before inference.
For an `xyxy` box, convert it with
`center_x = (x_min + x_max) / 2`, `center_y = (y_min + y_max) / 2`, and
`side_length = max(x_max - x_min, y_max - y_min)`; include any desired hand
padding in that side length. Values must be finite, with a positive side length.

Omit frames with no box instead of inserting a zero or null box. Missing frames
and missing hand files are treated as no detection; each video must have a file
for at least one requested hand. Frame IDs refer to decode order from the start
of the video, not timestamps.

The supplied [`saved_bboxes.yaml`](configs/saved_bboxes.yaml) contains:

```yaml
detector_backend: saved_bboxes
saved_bbox_path: "${BBOX_DIR}"
```

Set `BBOX_DIR` to the parent directory (`bboxes/`, not `bboxes/my_video/`):

```bash
export BBOX_DIR="/path/to/bboxes"

PYOPENGL_PLATFORM=egl PYTHONPATH="$(pwd)" .venv/bin/python \
  inference/scripts/run_egohand_inference_pipeline.py \
  --input /path/to/my_video.mp4 \
  --bbox_detector_config inference/hand_detector_2d/configs/saved_bboxes.yaml \
  --intrinsics_json /path/to/intrinsics.json \
  --perspective_cropping \
  --draw_skeleton --no_mesh \
  --dtype bfloat16 \
  --output_dir outputs/my_run
```

Keep the output root separate from `BBOX_DIR` to preserve your input box files.
Perspective cropping uses source-frame boxes directly. For standard pinhole
cropping, omit `--perspective_cropping`; the pipeline maps boxes internally if
pixel-aspect rectification is enabled. Saved source-frame boxes with distorted
cameras require perspective cropping. Do not pre-rectify the saved boxes.

## Use the bundled RT-DETR detector

The bundled [`rtdetr.yaml`](configs/rtdetr.yaml) enables tracking and offline
bbox smoothing. It needs no edits to use the hosted detector weights.

Pass it to `--bbox_detector_config` in the reconstruction command to detect
hands on the fly. The default ONNX weights download from the
[RT-DETR Hand Detector-v1](https://huggingface.co/nvidia/RT-DETR-Hand-Detector-v1)
repository and are cached locally.

For a local ONNX model, save your own detector YAML with:

```yaml
detector_backend: rtdetr
rtdetr_config:
  weights: "${RTDETR_WEIGHT_PATH}"
  bbox_smoothing: true
tracking: true
```

Export `RTDETR_WEIGHT_PATH=/path/to/hand_detector.onnx` and pass the path to
that YAML via `--bbox_detector_config`. Configs support environment-variable
substitution. Local detector weights bypass the RT-DETR download; the EgoHand
checkpoint is configured separately with `--egohand_ckpt`.

Bbox smoothing requires whole-clip detection, so live reconstruction with this
config cannot use `--frame_batch_size`. Disabling bbox smoothing with
`bbox_smoothing: false` permits batching, but live RT-DETR tracking then runs
independently within each batch. For whole-clip tracking with bounded decoding
memory, generate boxes with `run_hand_detector.py` as described below, then
reconstruct using `saved_bboxes.yaml`.

See the [config reference](scripts/README.md#bbox-detector-config-yaml-schema)
for thresholds, tracking, padding, and ONNX batch size.

## Known limitation: hands entering or leaving the frame

The bundled detector and tracker assume both hands are visible in frame.
When a hand disappears and re-enters partway through, the tracker can misassign or duplicate boxes.
This is not currently flagged in the results or in `--postprocessing_qc`.
If your footage includes hands entering/leaving the frame and this matters
for your use case, inspect the bbox overlay video, or supply your own
boxes/ground truth via the
[saved-box path](#use-your-own-detector-or-ground-truth-boxes) instead.

## Run detection separately

This is useful when you want to inspect or reuse detections before reconstruction.
For a second pass using `--perspective_cropping`, disable pixel-aspect
rectification in the detector pass:

```bash
PYTHONPATH="$(pwd)" .venv/bin/python \
  inference/hand_detector_2d/scripts/run_hand_detector.py \
  --input "$INPUT_VIDEO_PATH" \
  --output_dir "$BBOX_OUTPUT_PATH" \
  --bbox_detector_config inference/hand_detector_2d/configs/rtdetr.yaml \
  --intrinsics_json "$INTRINSICS_JSON_PATH" \
  --no-rectify_pixel_aspect_ratio
```

The script writes the same per-hand JSON format described above. Set
`BBOX_DIR="$BBOX_OUTPUT_PATH"`, then use the [saved-box reconstruction command](#use-your-own-detector-or-ground-truth-boxes).
That command uses `--perspective_cropping`. Use the same video and calibration
in both passes, with a separate reconstruction output directory. Perspective cropping
skips full-frame pixel-aspect rectification, so disabling it in the detector
pass matches live detection on source images. Saved boxes are always mapped
back to source coordinates; rectifying the detector input can change detections
when it resizes the images, but does not itself create a coordinate mismatch.

For standard pinhole cropping, omit `--perspective_cropping` and use the same
pixel-aspect rectification setting in both passes to match live detector
preprocessing (rectification is on by default). Match `--max_seconds` if you
want both passes to cover the same frames.

The standalone script runs RT-DETR only. Custom detectors feed the reconstruction
pipeline through exported JSONs. See the [standalone CLI reference](scripts/README.md)
for its full options.
