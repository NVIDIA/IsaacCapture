<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-OpenMDW-1.1
-->

# `run_hand_detector.py`

Stand-alone RT-DETRv2 hand bbox detection: video in, per-hand bbox JSON out.

This is the detection half of `run_egohand_inference_pipeline.py` pulled out
into its own entrypoint, so bbox detection can be run and reviewed as a
separate preprocessing stage instead of an in-process flag. Dependencies:
detection, the native tracker, the JSON writing, and `--intrinsics_json`'s
pixel-aspect-ratio correction need only this folder plus
`numpy`/`opencv`/`pillow`/`pyyaml`/`onnxruntime`/`huggingface_hub` (the last
only actually imported for the HF-download weights fallback) -- no torch, no
rest of the EgoHand repo -- so the folder can be copied out and run on its own.
`pixel_aspect.py` duplicates the square-pixel resize math from
`inference/camera/square_pixel.py` rather than importing it, so
rectification uses the same resize math without importing the rest of the
pipeline. Rectification is on by default here; the pipeline skips it with
`--perspective_cropping`. The detector Docker image also uses this self-contained
folder; see [the Docker guide](../../docker/README.md) for its wrapper (`detector.sh`).

## Usage

Run these commands from `src/python/isaaccapture_egopost/`. This example uses source-frame boxes with
perspective cropping, which supports the distorted-camera calibration supplied
for the NVIDIA evaluation videos.

```bash
python inference/hand_detector_2d/scripts/run_hand_detector.py \
    --input /path/to/video.mp4 \
    --output_dir /path/to/bboxes/ \
    --bbox_detector_config inference/hand_detector_2d/configs/rtdetr.yaml \
    --intrinsics_json /path/to/intrinsics.json \
    --no-rectify_pixel_aspect_ratio
```

Then, the EgoHand pass over the same video (omits `--egohand_ckpt` to use the
default HF-downloaded checkpoint; pass it explicitly to use a local one) --
use the same video and calibration in both stages (see below):
```bash
BBOX_DIR=/path/to/bboxes python inference/scripts/run_egohand_inference_pipeline.py \
    --input /path/to/video.mp4 \
    --bbox_detector_config inference/hand_detector_2d/configs/saved_bboxes.yaml \
    --intrinsics_json /path/to/intrinsics.json \
    --perspective_cropping \
    --output_dir /path/to/results/
```

For this two-stage perspective-cropping workflow, pass
`--no-rectify_pixel_aspect_ratio` when generating bboxes. The EgoHand pass skips
full-frame pixel-aspect rectification with `--perspective_cropping`, so this
keeps detection on the same source images as live detection in that mode.

Without the flag, the detector defaults to rectification. If the calibration's
`fx` and `fy` differ beyond the resize tolerance, detection runs on resized
images, which can change the boxes and downstream predictions. The JSON boxes
are still mapped back to source coordinates; omitting the flag does not by
itself create a coordinate mismatch or guarantee worse accuracy. For the
supplied NVIDIA calibration, rectification is a no-op, so the flag does not
change the results. FISHEYE624/KB4 calibrations require the flag because the
standalone rectification helper does not support them.

For standard pinhole cropping, omit `--perspective_cropping` and use the same
pixel-aspect setting in both passes to match live detection. Saved source-frame
boxes with distorted-camera calibration require `--perspective_cropping`;
the default full-image undistortion path requires a live detector on the
undistorted frames.

The JSONs this script writes are exactly what `saved_bbox.SavedBBoxDetector`
reads:

```
<output_dir>/<video_stem>/<video_stem>_{left,right}_bbox.json
```

and `saved_bboxes.yaml`'s `saved_bbox_path` is `${BBOX_DIR}`, i.e. the
`--output_dir` used here (`os.path.expandvars` resolves it at load time); the
second pass appends the video stem itself.

The standalone detector and live pipeline detection accept the same RT-DETR
config YAML. Only `detector_backend: rtdetr` is accepted by this script.
The reconstruction pass that reads its output uses a separate
`saved_bboxes.yaml` config, as shown above.

## Keep the two stages consistent

- **The video, and the calibration.** Use the same video and calibration in
  both stages. For perspective cropping, pass
  `--no-rectify_pixel_aspect_ratio` in the detector stage as shown above.
  For standard pinhole cropping, match the pixel-aspect settings to reproduce
  live detector preprocessing. Saved boxes are always in source coordinates,
  including when detection used rectified frames. Cropping or resizing the
  source video between stages requires transforming the saved boxes too.
- **`--max_seconds`.** Frame IDs are the decode index `{i:06d}` from the
  start of the video, the same convention the pipeline uses. Truncation changes
  which frames are covered without shifting their IDs. Whole-clip tracking and
  smoothing can also change boxes on shared frames when the detector pass is
  truncated.

Keep the reconstruction `--output_dir` separate from the detector output
(`saved_bbox_path`/`BBOX_DIR`), including their `<video_stem>/` subfolders.
When standard pinhole reconstruction actually resizes frames for pixel-aspect
rectification, its output rewrite can corrupt existing source-coordinate bbox
JSONs if those directories overlap. Perspective cropping skips this resize and
rewrite. See the [pipeline CLI reference](../../scripts/README.md#arguments)
for the full mechanism.

Frames are decoded in batches and only the detections are kept, so a long
video costs one batch of pixels rather than the whole clip. Tracking runs on
the accumulated detections after decoding finishes, so it sees the full sequence.

## Arguments

### Required

| Flag | Description |
| ---- | ----------- |
| `--input` | MP4 file or folder of MP4 files |
| `--output_dir` | Root for the bbox JSONs; one `<video_stem>/` subfolder per video. Pass this same path as `BBOX_DIR` to the `saved_bboxes.yaml` second pass |
| `--bbox_detector_config` | Bbox detector YAML, the same format `run_egohand_inference_pipeline.py` takes. Must set `detector_backend: rtdetr`. Everything about the detector comes from this file -- weights, score threshold, padding, ONNX batch size, whether to track. See `inference/hand_detector_2d/configs/rtdetr.yaml` |

### Optional

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--intrinsics_json` | none | Calibration for pixel-aspect rectification. Only read when rectification is enabled; without it, detection uses source images. The perspective-cropping reconstruction stage requires calibration independently. |
| `--hand` | `both` | `left`, `right`, or `both` |
| `--device` | `cuda` | Torch-style device, e.g. `cuda:1` or `cpu` |
| `--rectify_pixel_aspect_ratio` / `--no-rectify_pixel_aspect_ratio` | on | Resize non-square-pixel images before detection, then map boxes back to source coordinates. Use `--no-rectify_pixel_aspect_ratio` when the second stage uses `--perspective_cropping`. For standard pinhole cropping, match both stages to reproduce live detector preprocessing. No effect without calibration or when the focal-length ratio is within the resize tolerance of 1. |
| `--max_seconds` | none (full video) | Only process the first N seconds, matching the pipeline's `--max_seconds` |
| `--frame_batch_size` | `256` | Frames decoded into RAM at once. Caps peak memory on long clips; tracking still sees the whole accumulated sequence regardless. Distinct from the ONNX forward-pass batch, which the YAML sets via the `rtdetr_config.batch_size` key. Unlike `run_egohand_inference_pipeline.py`'s `--frame_batch_size`, this one has no rendering side effect -- this script never renders anything |

## Bbox detector config YAML schema

This script and `run_egohand_inference_pipeline.py` use separate config loaders
with the same YAML schema and `os.path.expandvars` substitution for `${ENV_VAR}`.
The pipeline supports `rtdetr` (live detection) and `saved_bboxes` (precomputed
JSONs); this script accepts only `rtdetr`.

### `detector_backend: rtdetr`

```yaml
detector_backend: rtdetr
rtdetr_config:
  weights: /path/to/rtdetr.onnx
  rtdetr_hf_weights: hand_detector.onnx
  batch_size: 8
  score_thr: 0.5
  padding: 0.3
  clip: false
  lr_order: true
  drop_gap_filled: false
  bbox_smoothing: false
  bbox_size_scale: 0.8
  bbox_size_alpha_grow: 0.12
  bbox_size_alpha_shrink: 0.06
  bbox_max_size_change: 0.01
tracking: true
```

`tracking` is a top-level boolean: `true` enables native temporal tracking;
`false` uses independent per-frame detections. Omission defaults to `false`.
Bbox smoothing requires `tracking: true`; for per-frame detection, also set
`bbox_smoothing: false` if your config enables it. Remove the obsolete
`rtdetr_config.tracker` key from older configs; it is no longer accepted.

This standalone script accumulates detections across decoding batches and tracks
the complete processed clip once. Live RT-DETR in the reconstruction pipeline
instead tracks each `--frame_batch_size` batch independently when bbox smoothing
is disabled; tracker state is not carried across batches. For whole-clip tracking
on long videos, run this script first and reconstruct using `saved_bboxes.yaml`.

Every `rtdetr_config` field is optional; defaults are listed in the table below
and defined in `inference/hand_detector_2d/rtdetr/detector.py:RTDetrConfig`.
The exception is `rtdetr_hf_weights`, which is a pseudo-field popped out of the YAML
mapping before it ever reaches `RTDetrConfig` -- by
`inference/hand_detector_2d/configs/rtdetr.py` for the full pipeline's rtdetr
backend, and by this script's own `load_rtdetr_config` for the standalone
path. It only controls `weights`'s default, it isn't a real `RTDetrConfig`
attribute:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `weights` | resolved via `nvidia/RT-DETR-Hand-Detector-v1` on HF | Path to the RT-DETRv2 `.onnx`; downloads the file named by `rtdetr_hf_weights` from [huggingface.co/nvidia/RT-DETR-Hand-Detector-v1](https://huggingface.co/nvidia/RT-DETR-Hand-Detector-v1) and caches locally if not given |
| `rtdetr_hf_weights` | `hand_detector.onnx` | Filename within `nvidia/RT-DETR-Hand-Detector-v1` to fetch for `weights`'s default. Only consulted when `weights` is not given -- lets you pick a different HF-hosted RT-DETR model without a local copy or a code change |
| `batch_size` | `8` | Frames per ONNX forward pass. Small on purpose -- the model is launch-bound, so bigger batches only cost memory |
| `score_thr` | `0.5` | Minimum detection confidence. Ignored under the native tracker, which needs sub-threshold candidates for its rescue rule |
| `padding` | `0.3` | Fractional bbox padding, EgoHand convention (`side *= 1 + padding`). Ignored under the native tracker, which applies its own |
| `clip` | `false` | Clip per-frame detections to image bounds before conversion to square crops. Ignored with tracking: detector candidates are always clipped, but final tracked boxes can extend outside the image. |
| `lr_order` | `true` | Native tracker only. Force the smaller-x box to be the left hand on frames where both are present |
| `drop_gap_filled` | `false` | Native tracker only. Emit boxes only on frames that carried a real detection |
| `bbox_smoothing` | `false` | Enable offline smoothing and tighter crops; requires `tracking: true`. The supplied RT-DETR YAMLs enable it explicitly. Set `false` to reproduce the previous native boxes. |
| `bbox_size_scale` | `0.8` | Target side relative to the native square, in `(0, 1]`; coverage and rate constraints can enlarge it. |
| `bbox_size_alpha_grow` | `0.12` | Size growth EMA weight at 20 fps, in `(0, 1]`; smaller values smooth more. |
| `bbox_size_alpha_shrink` | `0.06` | Size shrink EMA weight at 20 fps, in `(0, 1]`. |
| `bbox_max_size_change` | `0.01` | Nonnegative fractional growth limit per 20-fps interval. Shrinkage uses the reciprocal ratio. Zero holds a constant size within each track segment. |

Unknown `rtdetr_config` keys, invalid batch sizes, and invalid smoothing controls
are rejected during config validation. The standalone CLI reports these errors
with the config path and exits; the config API raises `ValueError`.

### Offline bbox smoothing

The filter runs after native association and geometry rules, before sparse
emission and EgoHand cropping. It rejects isolated reversals, smooths centers
adaptively, and smooths size in log space. A forward/backward size pass spreads
necessary growth over earlier frames and slows contraction. At 20 fps, adjacent
side lengths differ by at most a factor of `1.01`; elapsed timestamps scale
this bound and the EMA weights at other frame rates. Missing boxes, long-gap
reacquisition and large center jumps reset state; reset transitions are exempt.
An entirely undetected hand still emits no boxes.

The crop guard encloses the central `1 / PAD` of the native square, plus 1.5%
of its side per edge and compensation for center displacement. This is a
heuristic, not a guarantee of finger coverage: native sizes already include
minimum-size and motion rules. The guard uses the original size even when a
spike was rejected, so coverage can take priority over tightness. Smoother
trajectories alone do not demonstrate improved hand localization.

This is **offline, whole-clip processing**. Direct detector `run()` callers
must provide `frame_times={image_id: timestamp_seconds}` for the complete clip.
EgoHand supplies these timestamps even when pose smoothing is disabled. EgoHand's
`--frame_batch_size` is rejected when live bbox smoothing is enabled, before
existing outputs are modified. For large clips, generate bbox JSONs first with
this script, then run batched EgoHand with `saved_bboxes.yaml` as shown above.
This script's decode batching and the ONNX `batch_size` remain supported: both
accumulate detections before the filter runs once on the complete track.

### `detector_backend: saved_bboxes`

```yaml
detector_backend: saved_bboxes
saved_bbox_path: "${BBOX_DIR}"
```

| Field | Description |
| ----- | ----------- |
| `saved_bbox_path` | Root directory containing `<video_stem>/<video_stem>_{left,right}_bbox.json` (this script's own `--output_dir` layout). Almost always set to the `${BBOX_DIR}` placeholder and resolved via an environment variable at load time rather than hardcoded, so the same config file works for any bbox directory |

Only usable by `run_egohand_inference_pipeline.py` -- rejected by this script,
which requires `detector_backend: rtdetr` (it *produces* the bbox JSONs
`saved_bboxes` reads, so accepting `saved_bboxes` here would be circular).
