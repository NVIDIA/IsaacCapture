<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-OpenMDW-1.1
-->

# Optional hand postprocessing

[← EgoHand README](../../README.md)

Use quality control to inspect a reconstructed clip, or combine hand results
with an external camera trajectory for world-coordinate work. All commands run
from `src/python/isaaccapture_egopost/`.

## 3D quality control

QC checks both hands for implausible motion, geometry and size inconsistencies,
handedness errors, off-frame wrists, and missing predictions. It is **off by
default** and produces a report; a drop verdict does not delete predictions or
requested visualizations.

Add these options to your normal reconstruction command:

```bash
--postprocessing_qc --qc_max_bad_frac 0.02
```

The CLI requires `--hand both` (the default) and real camera calibration via
`--intrinsics_json`. Any [supported camera model](../../README.md#camera-input-formats)
can be used. The estimated-intrinsics fallback is not sufficient for QC.

Each video receives `<output_dir>/<video_stem>/quality_control.json`, with a
keep/drop verdict, rule counts, and source frame IDs:

- `frames[hand][rule]` lists the zero-based frame IDs flagged by that rule.
  Rules with no flagged frames are omitted.
- `missing_frames[hand]` lists frames with no prediction or non-finite 3D
  joints, including missing frames at the start and end of the processed video.
  Its length matches `missing[hand]`. Missing does not mean occluded.

The CLI allows up to 2% flagged frame-hand predictions by default; tune
`--qc_max_bad_frac` within `[0, 1]` for your data.
Delivery and frame-coverage checks still apply regardless of this tolerance.

These checks are heuristics. Review the report alongside the overlays when
choosing a threshold: a smooth but misplaced hand can pass a geometry check.
For Python use, rule definitions, and per-frame flags, see
[`clip_quality_control.py`](clip_quality_control.py). Its direct Python API
defaults to zero tolerance, whereas the inference CLI defaults to `0.02`.
For Python reports, use `verdict.to_json(with_frames=True)` to include frame
lists. Without an expected frame count, coverage is limited to the observed
frame span; `missing_frames` is `{}` when coverage cannot be determined.

## World coordinates and camera-pose refinement

EgoHand exports camera-relative hands. To place them in a world frame, supply
an external **camera-to-world** trajectory, for example from
[NVIDIA ViPE](https://github.com/nv-tlabs/vipe). Camera-pose error can introduce
jitter when composing that trajectory with the hands.

[`world_coordinate_projection.py`](world_coordinate_projection.py) uses both
hands jointly to refine the camera poses and reports metrics for the correction.
This is a separate, optional step after reconstruction:

```bash
PYTHONPATH="$(pwd)" .venv/bin/python \
  inference/postprocessing/world_coordinate_projection.py \
  --left "$OUTPUT_PATH/$VIDEO_NAME/left_results.npz" \
  --right "$OUTPUT_PATH/$VIDEO_NAME/right_results.npz" \
  --pose "$CAMERA_POSE_NPZ_PATH" \
  --out "$OUTPUT_PATH/$VIDEO_NAME/pose_stabilized.npz"
```

### Inputs

The camera trajectory NPZ accepts ViPE's layout:

```python
{
    "data": ...,  # (F, 4, 4), camera-to-world transforms
    "inds": ...,  # (F,), source frame indices; used to sort poses
}
```

It also accepts a bare `(F, 4, 4)` array under `data`, `poses`, or `arr_0`
without `inds`. The hand NPZ files use the standard EgoHand `joints` and
`cam_t` layout. The tool does not estimate a trajectory from video.

Both hand arrays must contain the same source frames as the camera poses.
The tool sorts poses by `inds` when present and otherwise keeps their stored
order. Arrange both hand arrays in that pose order before running. EgoHand
exports can omit different frames for each hand; use their `frame_names` to
check alignment. The tool pairs rows by position and checks equal frame counts,
but does not read or validate hand `frame_names`. Equal array lengths alone do
not guarantee frame alignment.

### Outputs and interpretation

| NPZ field | Contents |
| --- | --- |
| `poses` | Refined camera-to-world transforms, `(F, 4, 4)`. |
| `poses_raw` | Original transforms for comparison. |
| `left_cam`, `right_cam` | Hand joints in the refined camera frame, in metres. |
| `inds` | Sorted input pose indices, or sequential row indices `0..F-1` when the pose input has no `inds`. |
| `metrics` | JSON with quality metrics and parameters. |

Compose `poses` with `left_cam` / `right_cam` to obtain world coordinates; the
Python result provides `to_world()` for this operation.

The pose-fitting stage preserves the camera-frame hand keypoints. The default
final smoothing (`--smooth_post 1.0`) can also adjust the keypoints; set
`--smooth_post 0` to preserve their original image reprojection. Refinement
assumes some jitter is shared by both hands and caused by camera motion.
Inspect the reported common versus differential motion metrics before adopting
the corrected trajectory. It is not guaranteed to remain physically plausible.

See the [module documentation](world_coordinate_projection.py) for
`--sigma_smooth`, `--sigma_keep`, `--mode`, and other controls.

<a id="temporal-smoothing-during-inference"></a>

## Adaptive hand smoothing

The video runner enables adaptive SOMA parameter smoothing by default. Use
`--no-postprocessing_smooth` for raw predictions. The filter smooths hand
rotations on SO(3), camera translation and shape, then rebuilds joints and
vertices with the matching SOMA hand model. Saved parameters, geometry and
overlays therefore describe the same filtered hand.

Smoothing uses physical time (`frame_index / fps`) and adapts to local motion.
Missing or non-finite predictions and timestamp gaps split tracks. With
`--frame_batch_size`, raw predictions are collected across batches and filtered
once for the complete video before QC and export. Filtered joints are reprojected
using the corresponding pinhole or fisheye camera model. Filter defaults are
experimental; see [`adaptive_smoothing.py`](adaptive_smoothing.py).

Python callers retain `EgoHandPredictor(..., smooth=True)` and must supply
`frame_times={image_id: timestamp_in_seconds}` to `run()` or
`EgoHandInferencePipeline.run()`. Timestamps must be strictly increasing in image
order. Python smoothing remains off by default.

This is separate from both detector box tracking and the world-coordinate
refinement above.
