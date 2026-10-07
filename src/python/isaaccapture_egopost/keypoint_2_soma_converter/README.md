<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-OpenMDW-1.1
-->

# 21-point hand tracks to SOMA

Fit an undistorted video and frame-aligned, camera-space 21-point hand tracks
(MANO keypoint convention) to the native SOMA hand model with the MHR identity
backend. The result contains **25 native SOMA joints**, mesh reconstruction
parameters, and optional skeleton videos. This tool consumes existing 3D tracks;
it does not detect hands or estimate 3D keypoints from video alone.

## Sample: input 21 points → fitted SOMA 25 joints

**Left:** original input 21-point skeleton. **Right:** fitted native SOMA
25-joint skeleton, including the additional metacarpal joints. Both panels use
the same video frames and camera intrinsics. Green denotes the left hand;
orange denotes the right hand. Magenta rings on the fitted panel mark the
original input root (joint 0) for each hand. The input panel is not refitted
or smoothed.

![Last-frame comparison: input 21-point skeleton left, fitted SOMA 25-joint skeleton right](sample/comparison.jpg)

Last frame (index 570, at 19.00 seconds) of a 571-frame clip at 30 FPS,
1280×720 per panel. Only the screenshot is included in the repository.
To run fitting, supply your own undistorted video and matching landmarks below.
A fresh checkout cannot reproduce this exact clip without those original inputs.
The main command uses GrabCut; step 4 also provides the Transformers SAM3 variant.

## 1. Set up Python

From the repository root:

```bash
cd EgoHand/keypoint_2_soma_converter
```

Run the remaining commands from this directory.

Use Linux and an NVIDIA CUDA GPU. The default preset targets `cuda:0` with the
Triton renderer. The supplied requirements use PyTorch 2.7.1 and torchvision
0.22.1 with CUDA 12.8;
your NVIDIA driver must support that runtime. The sample was tested on an
RTX 6000 Ada. A separate CUDA toolkit installation is not needed for this preset.

Create a Conda environment:

```bash
conda env create -f environment.yml
conda activate soma-fit
```

Or install into an existing Python 3.12 environment:

```bash
python -m pip install -r requirements.txt
```

These are alternative setup routes. The requirements install this package and
pin the main numerical dependencies used for the sample. They are not a complete
transitive lock file. `imageio-ffmpeg` supplies the FFmpeg executable used to
create browser-compatible comparison videos; no workstation-specific FFmpeg
path is required. The setup installs the official `py-soma-x==0.3.0` package
and its dependencies, including `cholespy`. GrabCut needs no segmentation
checkpoint. Both setup routes include Transformers 5.18.0 and the SAM3 adapter
dependencies. SAM3 model weights are downloaded only when that backend is used.

Confirm the installation:

```bash
python -m soma_sequence_fit --help
```

The commands are `assets`, `fit`, and `compare`.

## 2. Download the native model assets

```bash
python -m soma_sequence_fit assets --out assets
```

This downloads the native SOMA/MHR files from `nvidia/SOMA-X`, revision
`v0.3.0`. Internet access is needed for the initial download. Model weights are
not committed; expected hashes are recorded in `src/soma_sequence_fit/assets.json`.
If you already have these assets, point `--assets` at their directory instead.

## 3. Prepare the landmark NPZ

The video must already be undistorted, and NPZ frame `t` must correspond to decoded video frame `t`.
The loader rejects mismatched frame counts. Dataset-specific decoding and
camera rectification happen upstream of this package.

| NPZ field | Shape | Meaning |
|---|---|---|
| `points3d` | `(T, 2, 21, 3)` | Anatomical left, then right; camera-space 3D points |
| `confidence` | `(T, 2, 21)` | Optional values in `[0, 1]`; zero marks missing points |
| `K` | `(3, 3)` or `(T, 3, 3)` | Pinhole camera intrinsics in source-video pixels |
| `timestamps` | `(T,)` | Optional strictly increasing seconds; otherwise derived from FPS |
| `units` | scalar string | `m`, `cm`, or `mm` |
| `coordinate_system` | scalar string | `opencv_camera` |
| `distortion` | optional | Must be zero; undistort upstream otherwise |

Camera axes are x right, y down, z forward. Points must include metric depth
and camera-space translation, not just root-relative hand coordinates. The
21-point order is wrist; thumb base-to-tip; then index, middle, ring, and little
finger base-to-tip, with four points per finger.

Example NPZ creation, after preparing arrays `points`, `confidence`, `K`, and `fps`:

```python
import numpy as np

np.savez_compressed(
    "landmarks.npz",
    points3d=points,  # (T, 2, 21, 3)
    confidence=confidence,  # (T, 2, 21)
    K=K,
    timestamps=np.arange(len(points)) / fps,
    units=np.array("m"),
    coordinate_system=np.array("opencv_camera"),
    distortion=np.zeros(5, np.float32),
)
```

## 4. Fit a video and its matching tracks

```bash
python -m soma_sequence_fit fit \
  --video /path/to/undistorted_video.mp4 \
  --points /path/to/landmarks.npz \
  --assets assets \
  --out outputs/sample \
  --overlay
```

Replace both input paths with your files. The video and landmarks must describe
the same frames. This uses the default fitting settings listed below. When
rerunning, choose a new output directory, such as `outputs/sample_2`, and use it in
step 5. The fit command will not overwrite an existing nonempty directory.
Omit `--overlay` if you only need the parameters and reports.
The overlay shows **all 25 fitted native SOMA joints**, connected using the
native skeleton hierarchy, with no filled masks or meshes. Magenta rings
label the original input root (joint 0) of each hand for comparison. These
reference markers do not change the fitted joints or exported parameters.

To fit with **Transformers SAM3**, first obtain access to `facebook/sam3` and
run `hf auth login`, then use:

```bash
python -m soma_sequence_fit fit \
  --video /path/to/undistorted_video.mp4 \
  --points /path/to/landmarks.npz \
  --assets assets \
  --mask-backend sam3 \
  --sam3-checkpoint facebook/sam3 \
  --out outputs/sample_sam3 \
  --overlay
```

For an already downloaded model, replace `facebook/sam3` with the local
Transformers model directory. This run also uses the default Triton renderer
and fitting settings. In step 5, replace `outputs/sample` with
`outputs/sample_sam3` in both the `--fit` and `--out` paths to visualize this run.

## 5. Generate the side-by-side comparison

```bash
python -m soma_sequence_fit compare \
  --video /path/to/undistorted_video.mp4 \
  --points /path/to/landmarks.npz \
  --fit outputs/sample/soma_parameters.npz \
  --out outputs/sample/comparison.mp4
```

Open `outputs/sample/comparison.mp4`. The command also writes
`outputs/sample/comparison.jpg` from the last frame. It uses the original input points
for the left panel and `joints25` plus `joint_parent_ids` from the fit for the
right panel. It checks frame indices, timestamps, intrinsics and joint shapes.
This visualization step does not run fitting or require a GPU. If the comparison
video or poster already exists, choose a new `--out` filename.

The screenshot above was generated by this procedure using the original test
clip. Reproducing that exact result requires its original video and landmarks,
which are not included. Floating-point optimization can vary slightly between
runs, even with the same settings; hardware and dependency changes can also
affect the result.

## Defaults and overrides

The default identity backend is `mhr`. Defaults are fixed; explicit flags
change only that invocation. No configuration edits are required.

| Setting | Default |
|---|---|
| Device / renderer / masks | `cuda:0` / `triton` / `grabcut` |
| Image size / pose components | `96` / `72` |
| Keyframes / residual candidates / added keyframes | `8` / `12` / `2` |
| Batch / keypoint batch / sequence batch | `8` / `128` / `64` |
| Initialization / pose / shape / final steps | `80` / `20` / `40` / `100` |
| Rounds / surface refinement / sequence polish | `2` / `30` / `20` |
| Wrist source / bone optimization | `palm` / enabled |
| Overlay | off unless `--overlay` is supplied |

For example, append `--device cuda:1 --batch-size 16` to the fit command.
`--no-optimize-bones` disables bone-scale optimization. CPU fitting requires
`--device cpu --renderer soft` and is substantially slower. See all options:

```bash
python -m soma_sequence_fit fit --help
```

### Shape consistency and jitter reduction

Identity, global scale and bone-scale parameters are shared across all frames
of each hand within a clip. They are not re-estimated independently per frame.
Pose and camera-space translation vary over time.

Whole-sequence fitting includes temporal penalties on rotation, translation
and mesh motion, including abrupt changes in acceleration. The defaults use
`--temporal-weight 1`, `--temporal-target smooth_observations`, a ±67 ms local
window, and at most 2 mm correction of the temporal observation target. Mesh
and root jerk weights are `0.05` and `0.1`. Temporal links cross renderer batch
boundaries but stop at missing frames or gaps over 100 ms.

Shape sharing is enforced. Smoothness is a weighted objective alongside
landmark and image agreement; it does not guarantee zero jitter. The 2 mm
setting bounds target correction, not output jitter. Each hand/clip has its
own shared shape; shape is not linked across different recordings.

### Segmentation and cache reuse

GrabCut is the default. Supplied binary masks can be used with
`--mask-backend files --mask-dir DIR`, laid out as `DIR/left/000000.png` and
`DIR/right/000000.png` at full video resolution. Use 8-bit grayscale PNGs with
`0` for background and `255` for foreground. The loader thresholds at `>127`,
so masks stored with values `0` and `1` are read as empty.

`--ignore-dir DIR` uses the same layout, resolution and encoding; white (`255`)
pixels exclude occluder regions from image fitting.

SAM3 uses Hugging Face Transformers. Transformers and torchvision are direct
project dependencies, included by both setup routes above and by `pip install .`.

Select `--mask-backend sam3`. The default model is `facebook/sam3`; obtain access
on Hugging Face and authenticate with `hf auth login` before the first download.
For offline use, pass `--sam3-checkpoint /path/to/sam3_directory`, containing the
Transformers model, processor and tokenizer files. Standalone `.pt` checkpoints
are not supported. Model weights retain their upstream license terms.

The adapter uses CUDA, bfloat16 image inference and a cached `hand` text prompt;
fitting remains float32. Detection and mask thresholds are both 0.5. Masks are
resized to the source video resolution before left/right association.
The first fit command in step 4 uses GrabCut and does not download SAM3 weights.

Rendering uses `--renderer triton` by default. `--renderer auto` selects Triton
on CUDA when available, otherwise the portable `soft` renderer. Explicit
`--renderer soft` supports CPU fitting at image sizes up to 128.

The documented command generates fresh observations. `--reuse-observations DIR`
checks input hashes, image size, seed and preprocessing code, but does not
validate changes to the requested mask backend, mask/ignore files, or SAM3
checkpoint. Do not reuse observations when those change.

## Outputs

| File | Contents |
|---|---|
| `soma_parameters.npz` | Shared identity/scale and per-frame native pose, translation, 25 joints and reconstruction metadata |
| `report.json` | Actual configuration, input/source/asset hashes, fitting diagnostics |
| `timing_breakdown.json` | Preprocessing, fitting and export timings |
| `metric_preservation.json` | Input-to-fit dimension diagnostics |
| `overlay.mp4` | Optional fitted SOMA 25-joint skeleton |
| `comparison.mp4` / `.jpg` | Optional side-by-side video and last-frame screenshot from `compare` |

The NPZ contains `identity_coeffs`, `global_scale`, `scale_params`,
`rotation_matrices`, `poses`, `global_translation`, `joints25`, `joint_parent_ids`,
`joint_names`, semantic `observations21`, validity flags, intrinsics and timestamps.
Translations and joints are in meters. Native root position is
`global_translation == joints25[..., 0, :]`; the source wrist convention need
not match the native root. Apply identity/scale parameters exactly once when
reconstructing the native model. Optimization logs, checkpoints and newly
generated observations are written under the output directory. With
`--reuse-observations DIR`, observations are read directly from `DIR` and are
not copied into the new output directory.

## Package contents

| Module | Responsibility |
|---|---|
| `__main__.py` | Commands and default settings |
| `data.py`, `geometry.py` | Input validation, video loading and camera geometry |
| `preprocess.py`, `segmentation.py` | Observations, masks and frame selection |
| `model.py` | Asset download and adapter for the installed SOMA hand model |
| `fitting.py`, `losses.py` | Optimization, size constraints and temporal objectives |
| `render.py`, `fast_render.py` | Surface rendering and the optional Triton implementation |
| `export.py`, `visualization.py` | Result files and skeleton comparisons |

- `src/`: fitting pipeline, comparison renderer, SOMA adapter and asset manifest.
- `pyproject.toml`, `requirements.txt`, `environment.yml`: installation metadata and setup.
- `sample/comparison.jpg`: the last-frame comparison screenshot.
- [`../../LICENSE.md`](../../LICENSE.md): the EgoHand OpenMDW-1.1 license.
