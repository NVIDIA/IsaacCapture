<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-OpenMDW-1.1
-->

# EgoHand inference in Docker

Two Docker images, each a thin `docker build`/`docker run` wrapper around one
Python entrypoint script. For the full argument list of each script, see the
component README:

| Wrapper | Wraps | Component README |
| ------- | ----- | ----------------- |
| `detector.sh` + `Dockerfile.detector` | `inference/hand_detector_2d/scripts/run_hand_detector.py` (RT-DETRv2 bbox detection only) | `inference/hand_detector_2d/scripts/README.md` |
| `egohand_inference.sh` + `Dockerfile.egohand` | `inference/scripts/run_egohand_inference_pipeline.py` (bbox detection + EgoHand hand reconstruction) | `inference/scripts/README.md` |

Both images bake in the source; checkpoints and input videos are mounted at
run time unless you bundle them at build time.

DINOv3 is built into the image from `src/python/isaaccapture_egopost/requirements-dinov3.txt`; building
needs GitHub access, inference does not.

## Prerequisites

- **Docker with the NVIDIA Container Toolkit.** `run` uses `docker run --gpus
  all`, which fails without the toolkit installed and configured for Docker.
  Verify with `docker run --rm --gpus all ubuntu nvidia-smi`.
- **Output file ownership.** The images run as a non-root `app` user, and
  `run` passes `--user "$(id -u):$(id -g)"`, so everything written to `--out`
  (npz, json, mp4, `perf.json`) is owned by you on the host. If you call
  `docker run` yourself, pass the same `--user` flag to get that behavior.
- **Disk space.** ~19 GB for the image, plus ~10 GB free in `/tmp` (or `$TMPDIR`)
  if you bundle the EgoHand checkpoint at build time.
- **Network access to Hugging Face, if not bundling/mounting weights.** See
  "HuggingFace fallback" below.

## `egohand_inference.sh` -- EgoHand hand reconstruction

The image entrypoint is `inference/scripts/run_egohand_inference_pipeline.py`:
RT-DETR hand-bbox detection (on the fly, or read from precomputed JSONs via
`--bbox-dir`) followed by `inference/pipeline/pipeline.py:
EgoHandInferencePipeline`, run directly on decoded video frames.

### Build

```bash
inference/docker/egohand_inference.sh build <image> [options]
```

`<image>` is the Docker image name/tag to create (required positional argument).

First build takes ~15 min (pip deps). Source-only changes rebuild in <1 min
because the pip layer is cached and only the `COPY` layers re-run.

The script stages a minimal build context in a temp dir, so large local dirs
(`outputs/`, `inputs/`, dataset symlinks) are never sent to the Docker daemon.

#### Bundling weights at build time

Weights can be baked into the image so `run` needs fewer flags:

```bash
inference/docker/egohand_inference.sh build <image> \
    --egohand-ckpt             /path/to/egohand.ckpt \
    --hand-detector-weight /path/to/rtdetr.onnx
```

| Flag | Effect |
| ---- | ------ |
| `--egohand-ckpt <path>` | Bundle EgoHand checkpoint (see "HuggingFace fallback" below for what happens when omitted) |
| `--hand-detector <rtdetr\|saved_bboxes>` | Which detector the weight is for (default: `rtdetr`, the only option that takes a weight). `saved_bboxes` needs no weight -- bbox detection is precomputed and supplied at run time via `--bbox-dir`; `--hand-detector-weight` is rejected with this value |
| `--hand-detector-weight <path>` | Bundle detector weight (not needed if every run will use `--bbox-dir`) |

Camera intrinsics are never bundled at build time -- always pass
`--intrinsics-json` at `run` time (for both the on-the-fly and `--bbox-dir`
paths). This model needs the real calibration to produce correct depth/mesh
scale.

### Run

```bash
inference/docker/egohand_inference.sh run \
    --image                <image> \
    --ckpt                 /path/to/egohand.ckpt \
    --hand-detector-weight /path/to/rtdetr.onnx \
    --input                /path/to/clip.mp4 \
    --intrinsics-json      /path/to/intrinsics.json \
    --out                  outputs/inference
```

`--image`, `--input`, and `--intrinsics-json` are required by this wrapper
(fails immediately, before touching Docker, if any is missing).
`--intrinsics-json` is never bundleable at build time (see above) -- always
pass it here. `--ckpt` and `--hand-detector-weight` are optional; see
"HuggingFace fallback" below.

`--input` accepts a single `.mp4` file or a folder of `.mp4` files; per-video
results land in `<out>/<video_stem>/`. The model type is auto-detected from
the checkpoint.

This wrapper is a sample invocation, not a general-purpose CLI: it hardcodes
`--hand both --device cuda --batch_size 32 --dtype bfloat16 --draw_skeleton
--no_mesh` (see the script's `usage()`). For other flags, edit the script or
run `docker run` yourself with `python inference/scripts/run_egohand_inference_pipeline.py ...`
and the mounts in "Mount layout inside the container"; see
`inference/scripts/README.md` for the full argument list.

#### Two steps: detect, then reconstruct from saved boxes (`--bbox-dir`)

```bash
# 1. Detect hand bboxes only
inference/docker/detector.sh run \
    --image           <detector-image> \
    --input           /path/to/clip.mp4 \
    --intrinsics-json /path/to/intrinsics.json \
    --out             /path/to/bboxes

# 2. Reconstruct from the saved boxes
inference/docker/egohand_inference.sh run \
    --image           <image> \
    --ckpt            /path/to/egohand.ckpt \
    --input           /path/to/clip.mp4 \
    --bbox-dir        /path/to/bboxes \
    --intrinsics-json /path/to/intrinsics.json \
    --perspective-cropping \
    --out             outputs/inference
```

`--bbox-dir` skips on-the-fly detection and reads the per-hand bbox JSONs that
`detector.sh run` writes. It is mutually exclusive with
`--hand-detector-weight`; no detector weight is loaded. See
`inference/hand_detector_2d/scripts/README.md` for the JSON layout.

These wrappers run the same pipeline as the host commands, so check the
[main README](../../README.md#reconstruct-with-your-own-2d-bboxes) for the
running rules, such as when `--perspective-cropping` is required for saved
boxes and how camera calibration is used
([camera formats](../../README.md#camera-input-formats)).

#### HuggingFace fallback

Checkpoints and detector weights (`--ckpt`, `--hand-detector-weight`,
`--weight`) resolve in this order: mounted at run time, bundled at build time,
downloaded from Hugging Face. Passing none is not an error; the run only fails
if the download fails. Optionally, pass `--hf-token "$HF_TOKEN"` (either
wrapper) to forward a token as `HF_TOKEN` in the container, e.g. if you hit
anonymous rate limits.

## `detector.sh` -- bbox detection only

The image entrypoint is `inference/hand_detector_2d/scripts/run_hand_detector.py`:
RT-DETRv2 hand-bbox detection only -- no EgoHand model, no SOMA, no torch
even loaded. Pairs with `egohand_inference.sh run --bbox-dir` to feed the JSONs
this produces into a full EgoHand inference run without re-detecting, so
detection can be run, reviewed, and cached as its own preprocessing stage.

### Build

```bash
inference/docker/detector.sh build <image> [--weight /path/to/rtdetr.onnx]
```

`--weight` optionally bundles an RT-DETRv2 ONNX weight into the image. The
HuggingFace fallback downloads `nvidia/RT-DETR-Hand-Detector-v1`.

### Run

```bash
inference/docker/detector.sh run \
    --image           <image> \
    --input           /path/to/clip.mp4 \
    --out             /path/to/bboxes \
    --intrinsics-json /path/to/intrinsics.json
```

`--image`, `--input`, `--out`, `--intrinsics-json` are required. `--weight`
is optional (see "HuggingFace fallback" above). Writes
`<out>/<video_stem>/<video_stem>_{left,right}_bbox.json` -- pass that same
`<out>` path as `--bbox-dir` to a `egohand_inference.sh run`.

`--tracking` and `--no-tracking` override the shared tracking setting only when
passed explicitly; `--no-tracking` also disables bbox smoothing. Use the same
input video and `--intrinsics-json` for both stages; see
`inference/hand_detector_2d/scripts/README.md` for camera preprocessing details.

## Mount layout inside the container

| Flag | Container path | Mode |
| ---- | -------------- | ---- |
| `--ckpt` | `/workspace/inputs/<filename>` | ro |
| `--weight` (`detector.sh`) / `--hand-detector-weight` (`egohand_inference.sh`) | `/workspace/inputs/<filename>` | ro |
| `--intrinsics-json` (either wrapper) | `/workspace/inputs/<filename>` | ro |
| `--bbox-dir` (`egohand_inference.sh`) | `/workspace/inputs/bbox` | ro |
| `--input` (file) | `/workspace/inputs/video/<filename>` | ro |
| `--input` (folder) | `/workspace/inputs/video` | ro |
| `--out` | `/workspace/outputs/bbox` (`detector.sh`) or `/workspace/outputs/inference` (`egohand_inference.sh`) | rw |
