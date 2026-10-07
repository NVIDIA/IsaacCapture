<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-OpenMDW-1.1
-->

# `run_egohand_inference_pipeline.py`

End-to-end EgoHand inference on video files, run directly on decoded video
frames: 2D hand bounding boxes, from either the optional RT-DETR detector
or precomputed bounding-box files, feed `EgoHandPredictor` inference against
an EgoHand checkpoint; the model type is
auto-detected from the checkpoint, no flag needed.

`inference/docker/egohand_inference.sh` is a thin wrapper around this script for
running it in a container -- see `inference/docker/README.md` for the
Docker-specific flags (`--image`, `--ckpt`, etc.) and which of the flags
below it hardcodes.

## Usage

Single video with live RT-DETR detection:
```bash
python inference/scripts/run_egohand_inference_pipeline.py \
    --input /path/to/video.mp4 \
    --output_dir outputs/infer_video/ \
    --intrinsics_json /path/to/intrinsics.json \
    --bbox_detector_config inference/hand_detector_2d/configs/rtdetr.yaml
```

Folder of MP4s:
```bash
python inference/scripts/run_egohand_inference_pipeline.py \
    --input /path/to/videos/ \
    --output_dir outputs/infer_video/ \
    --intrinsics_json /path/to/intrinsics.json \
    --bbox_detector_config inference/hand_detector_2d/configs/rtdetr.yaml
```

Reading precomputed bboxes in two stages (see
`inference/hand_detector_2d/scripts/README.md`). First generate source-frame
boxes with pixel-aspect rectification explicitly disabled:
```bash
python inference/hand_detector_2d/scripts/run_hand_detector.py \
    --input /path/to/video.mp4 \
    --output_dir /path/to/bboxes/ \
    --bbox_detector_config inference/hand_detector_2d/configs/rtdetr.yaml \
    --intrinsics_json /path/to/intrinsics.json \
    --no-rectify_pixel_aspect_ratio
```

Then run inference with perspective cropping. The bbox input path is read from
`saved_bboxes.yaml` (`saved_bbox_path: "${BBOX_DIR}"`), with `${BBOX_DIR}`
resolved from the environment at load time:
```bash
BBOX_DIR=/path/to/bboxes python inference/scripts/run_egohand_inference_pipeline.py \
    --input /path/to/video.mp4 \
    --bbox_detector_config inference/hand_detector_2d/configs/saved_bboxes.yaml \
    --output_dir outputs/infer_video/ \
    --intrinsics_json /path/to/intrinsics.json \
    --perspective_cropping
```

Keep `--no-rectify_pixel_aspect_ratio` in the first stage to match the source
images used by live detection with perspective cropping. Without it,
rectification can change detections when it resizes the input, although saved
boxes are still mapped back to source coordinates. With the supplied NVIDIA
calibration, rectification is a no-op. Perspective cropping uses source-frame
boxes and supports distorted cameras.
Standard pinhole cropping can omit `--perspective_cropping` if pixel-aspect
settings match the detector pass. Saved source-frame boxes cannot be used with
full-image undistortion; that mode requires a live detector.

Every example above omits `--egohand_ckpt` to use the default HF-downloaded
checkpoint; pass it explicitly (`--egohand_ckpt /path/to/checkpoint.ckpt`) to
use a local one instead.

Metric output depends on the intrinsics in every camera mode: the predicted
camera is converted to a translation with `compute_transl_full_cam(pred_cam,
bbx_xys, K)`, so depth scales with `fx`. Without `--intrinsics_json`,
this falls back to `estimate_K`'s `focal = max(W, H)`, which puts every hand
at the wrong distance whenever the true focal differs from that heuristic --
invisibly, because the overlay renders through the same wrong
`K`. Checkpoints trained with `camera_mode: sam3db_perspective` also feed `K`
to the network, so a wrong one perturbs the prediction as well; those get an
explicit warning when `--intrinsics_json` is omitted.

## Arguments

### Required

| Flag | Description |
| ---- | ----------- |
| `--input` | MP4 file or folder of MP4 files |
| `--output_dir` | Output root; per-video results go into `<output_dir>/<video_stem>/` |
| `--bbox_detector_config` | Path to a bbox detector config YAML. `main()` raises if this is omitted. Supports `${ENV_VAR}` substitution for per-video paths (e.g. `saved_bboxes.yaml`'s `${BBOX_DIR}`). `rtdetr` (live detection) or `saved_bboxes` (read precomputed JSONs) -- full schema for both in `inference/hand_detector_2d/scripts/README.md#bbox-detector-config-yaml-schema` |
| `--intrinsics_json` | Camera calibration in the [shared JSON format](../../README.md#camera-input-formats). Required with `--perspective_cropping`. Omitting it falls back to an estimated focal length and reduces 3D accuracy -- see [known limitation](../../README.md#known-limitation-running-without-calibration) |

### Detection

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--hand` | `both` | `left`, `right`, or `both` |

**With `saved_bboxes`, `--output_dir` must NOT be the same path as (or share
a `<video_stem>/` subfolder with) `saved_bboxes.yaml`'s
`saved_bbox_path`/`BBOX_DIR`.** `--rectify_pixel_aspect_ratio` is on by
default, so this applies to any `saved_bboxes` run whose calibration has
`fx != fy` unless `--no-rectify_pixel_aspect_ratio` is passed. When a run
actually rectifies (`fx != fy`), it rewrites `<output_dir>/<video_stem>/<video_stem>_
<hand>_bbox.json` in place, mapping it from the corrected model frame back
to the source frame -- correct when that file is one this same run's live
detector just wrote. But `SavedBBoxDetector`'s input JSON at
`<saved_bbox_path>/<video_stem>/<video_stem>_<hand>_bbox.json` is already in
the source frame (`reports_in_source_frame = True`); if `--output_dir` points
at the same tree, this rewrite step silently double-applies the frame
mapping to that *input* file and corrupts it on disk for any future run that
reads it back. The current run's own results are unaffected (the file is
read before this happens), but the corruption is silent and persists. Keep
`--output_dir` and the bbox-detector-stage's `--output_dir`/`BBOX_DIR`
pointed at different roots.

### Camera / geometry

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--perspective_cropping` | off | Project each detected hand region into a virtual pinhole view using `--intrinsics_json` |
| `--rectify_pixel_aspect_ratio` / `--no-rectify_pixel_aspect_ratio` | on | For calibrations with `fx != fy`, resize each frame (and adjust its intrinsics) to what a square-pixel sensor of focal `max(fx, fy)` would have captured, before detection/inference -- EgoHand hand checkpoints are trained only on square-pixel cameras. No effect with `--perspective_cropping` or when `fx == fy`. Overlays, bbox JSONs and 3D outputs are all expressed in the original frame. For PINHOLE input, when pixel-aspect correction is applied, per-frame files written by `--save_npzs` retain `pred_keypoints_2d` normalized relative to the crop in the corrected image. See the warning below `--bbox_detector_config`/`--output_dir` if using `saved_bboxes` |

### Model / compute

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--egohand_ckpt` | resolved via `nvidia/EgoHand-v1` on HF | EgoHand checkpoint (`.ckpt`); downloads the file named by `--egohand_ckpt_hf_path` from [huggingface.co/nvidia/EgoHand-v1](https://huggingface.co/nvidia/EgoHand-v1) and caches locally if not given |
| `--egohand_ckpt_hf_path` | `egohand_s060000.ckpt` | Filename within `nvidia/EgoHand-v1` to fetch for `--egohand_ckpt`'s default. Only consulted when `--egohand_ckpt` is not given -- lets you pick a different HF-hosted checkpoint without a local copy or a code change |
| `--batch_size` | `32` | Batch size for EgoHand inference. RT-DETR takes its own batch size via the `rtdetr_config.batch_size` key in the detector YAML |
| `--dino_compile` | off | Compile each DINOv3 block (`torch.compile`, inductor, fullgraph) for eval/no-grad calls at the exact `--batch_size`; training, grad-enabled calls, and partial/tail batches keep the unmodified eager path automatically. Opt-in speedup for throughput-sensitive batch inference (e.g. long videos) where most batches are full-size -- skip it for small one-off runs, since `torch.compile` warm-up cost can exceed the savings. Compiled output can differ from eager by up to ~1e-2 on final joints/vertices (bf16/inductor numerics); validate once against an eager baseline before relying on it downstream |
| `--device` | `cuda` | Torch device |
| `--dtype` | `float32` | `float32`, `float16`, or `bfloat16` |

### Postprocessing

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--postprocessing_smooth` / `--no-postprocessing_smooth` | on | Adaptively smooth SOMA rotations, camera translation and shape using video timestamps; rebuild joints and vertices and re-project 2D keypoints through the original camera. Filtering spans the complete clip before QC and export, including across `--frame_batch_size` boundaries. Missing predictions and large timestamp gaps split tracks. See `inference/postprocessing/adaptive_smoothing.py`. Distinct from bbox tracking, which processes the 2D boxes before inference |
| `--postprocessing_qc` / `--no-postprocessing_qc` | off | Run 3D quality control once per complete video and write `quality_control.json`. Predictions and overlays are kept even when QC reports `drop=true`. Requires `--hand both` and `--intrinsics_json`. See the [postprocessing README](../postprocessing/README.md) |
| `--qc_max_bad_frac` | `0.02` | Allowed fraction of flagged frame-hands for QC, in `[0, 1]` (default: up to 2%). Coverage checks always apply. Only used with `--postprocessing_qc` |
| `--soma_hand_assets` | resolved via `soma.get_assets_dir()` | SOMAHand asset folder; downloads from [huggingface.co/nvidia/SOMA-X](https://huggingface.co/nvidia/SOMA-X) and caches locally if not given |

### Output control

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--no_save_videos` | off (videos saved) | Skip `.mp4` overlay outputs |
| `--no_mesh` | off | Skip mesh rendering in overlays. Mesh overlays support PINHOLE input |
| `--save_crop_overlays` | off | Save `crop_overlay_<hand>.mp4` (and images, per `--save_images`). Mainly a debugging aid for bbox/crop alignment -- the crop window follows each frame's raw detected bbox, which `--postprocessing_smooth` never touches, so it can look jittery even though the mesh/pose inside it is smoothed. `bbox_viz.mp4`/`fullimg_overlay.mp4` are unaffected either way. Only for PINHOLE input without `--perspective_cropping` |
| `--draw_skeleton` / `--no-draw_skeleton` | on | Draw the 2D joint skeleton overlay |
| `--save_images` | off | Save per-frame `.jpg` outputs |
| `--save_npzs` | off | Save per-frame full SOMA `.npz` outputs |
| `--per_frame_results` | off | Write one result `.npz` per frame under `<video>/<hand>/results/` (legacy layout). Default: a single `<video>/<hand>_results.npz` per hand for the whole clip |
| `--perf_output` | `<output_dir>/<video_stem>/perf.json` | Where to write perf metrics JSON |

### Long-video / memory

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--max_seconds` | none (full video) | Limit inference to the first N seconds of each video |
| `--frame_batch_size` | none (single batch = whole clip) | Stream video in batches of this many frames to limit decoded-image memory. **Disables overlay rendering when set** because batch images are released after inference. Predictions are retained for the complete clip; adaptive smoothing also retains bboxes and cameras and runs once after all batches, before QC and exports. With live RT-DETR `bbox_smoothing: true`, this option is rejected because bbox filtering must finish before cropping. With `bbox_smoothing: false` and `tracking: true`, tracking runs independently within each batch. For whole-clip tracking with bounded decoding memory, generate boxes with `run_hand_detector.py` first and use `saved_bboxes.yaml` for reconstruction. |

## Output layout

```
<output_dir>/
  <video_stem>/
    <video_stem>_left_bbox.json      # written only by a live (non-saved_bboxes) detector backend
    <video_stem>_right_bbox.json
    left_results.npz                  # SOMA, whole-clip (or --per_frame_results layout)
    right_results.npz
    quality_control.json             # only with --postprocessing_qc
    bbox_viz.mp4                      # unless --no_save_videos
    crop_overlay_left.mp4             # only with --save_crop_overlays (only for PINHOLE input without --perspective_cropping)
    crop_overlay_right.mp4            # only with --save_crop_overlays (only for PINHOLE input without --perspective_cropping)
    fullimg_overlay.mp4
    perf.json                         # unless --perf_output overrides the path
```

### Result arrays

`left_results.npz` and `right_results.npz` contain the following arrays. `N`
is the number of predictions for that hand, not the video length; use
`frame_names[i]` to locate row `i` in the source video. Missing predictions
are omitted, so the two hands may have different frame lists. An empty hand
file contains only `num_frames=0` and an empty `frame_names` array.

| Array | Shape | Meaning and units |
| --- | --- | --- |
| `num_frames` | scalar | Number of saved predictions. |
| `frame_names` | `(N,)` | Zero-based source frame IDs, stored as strings. |
| `hand_poses` | `(N, 25, 3, 3)` | Joint rotation matrices in the SOMA pose convention. |
| `identity_coeffs` | `(N, 20)` or `(N, 5)` | Dimensionless SOMA or MHR identity coefficients. |
| `global_scale` | `(N,)` | Dimensionless uniform scale. |
| `scale_params` | `(N, 26)` | Optional MHR scale parameters. |
| `vertices` | `(N, Vh, 3)` | Wrist-relative mesh vertices, metres; `Vh` depends on the hand model. |
| `joints` | `(N, 25, 3)` | Wrist-relative SOMA-25 joint positions, metres. |
| `cam_t` | `(N, 3)` | Wrist translation in camera coordinates, metres. |
| `pred_kp2d_crop_presence` | `(N, 25)` | Optional model score in `[0, 1]` for each joint being inside the model crop. |
| `pred_kp2d_sigma` | `(N, 25, 2)` | Optional RLE uncertainty scales for x/y in normalized model-crop coordinates; one unit is the crop side length. |

Geometry uses camera axes: x right, y down, z forward. Camera-space joints
are `joints + cam_t[:, None, :]`; perspective-cropped predictions are already
rotated back to the source camera axes. The SOMA-25 order is wrist `0`, thumb
`1–4`, index `5–9`, middle `10–14`, ring `15–19`, and little finger `20–24`,
with each finger ordered from base to tip. See the exact hierarchy in
[`soma_constants.py`](../postprocessing/soma_constants.py).

The two optional joint signals come from the direct 2D head and remain raw
even when pose smoothing is enabled. Sigma describes the model crop (the
virtual crop with perspective cropping), not source pixels or smoothed 3D
positions. Neither field is calibrated 3D confidence or an occlusion label:
an occluded joint can still have high crop presence. Checkpoints without the
2D head omit both fields; no confidence values are fabricated.

`--per_frame_results` writes the same prediction fields without the leading
`N` axis; frame IDs are in filenames rather than `frame_names`.
`--save_npzs` additionally writes full SOMA files. Their joint signals have
shapes `(1, 25)` and `(1, 25, 2)`; `pred_projection_valid` is a `(1, 25)`
boolean mask for geometrically valid 2D projection, not visibility.
