<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-OpenMDW-1.1
-->

<div align="center" style="text-align: center;">
<h1 align="center" style="text-align: center;">EgoHand</h1>
<p align="center" style="text-align: center;"><em>Monocular 3D Hand Reconstruction from Egocentric Video</em></p>
<p align="center" style="text-align: center;">
  <a href="https://arxiv.org/abs/2610.30293">
    <img src="https://img.shields.io/badge/arXiv-2610.30293-b31b1b.svg"
         alt="arXiv: 2610.30293">
  </a>
  <a href="https://huggingface.co/nvidia/EgoHand-v1">
    <img src="https://img.shields.io/badge/Hugging_Face-EgoHand--v1-ffd21e"
         alt="Hugging Face: EgoHand-v1">
  </a>
  <a href="../../../LICENSES/LicenseRef-OpenMDW-1.1.txt">
    <img src="https://img.shields.io/badge/License-OpenMDW--1.1-green"
         alt="License: OpenMDW-1.1">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.12-blue" alt="Python 3.12">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-ee4c2c" alt="PyTorch">
  </a>
  <a href="https://developer.nvidia.com/cuda/toolkit">
    <img src="https://img.shields.io/badge/CUDA-GPU-76b900" alt="CUDA GPU">
  </a>
</p>
<p align="center" style="text-align: center;">
  <a href="docs/showcase/egohand_showcase.mp4"><img src="docs/showcase/egohand_showcase.gif" width="800" alt="EgoHand showcase: a 3-row, 4-column video grid becomes skeleton overlays, a hand close-up, and an orbiting 3D mesh view."></a><br>
</p>
</div>

## 📰 News

- **[2026-10-21]** 🚀 **EgoHand v1** is released for monocular 3D hand pose
  estimation using the [SOMA-Hand representation](https://github.com/NVlabs/SOMA-X).
  Get started with [pretrained models](https://huggingface.co/nvidia/EgoHand-v1),
  the [inference example](#example),
  [fine-tuning EgoHand with GEM-X](https://github.com/NVlabs/GEM-X), and the
  [paper](https://arxiv.org/abs/2610.30293).

<a id="overview"></a>

## 🔎 Overview

EgoHand is a monocular 3D hand reconstruction model developed by NVIDIA that recovers 25-joint hand poses and meshes from egocentric video using [SOMA-Hand](https://github.com/NVlabs/SOMA-X). Trained on 14 million diverse images, it delivers state-of-the-art accuracy with approximately 9 ms inference latency, enabling scalable annotation for dexterous robot policy learning.


The modular pipeline includes hand detection, adaptive temporal smoothing, and world-coordinate conversion using camera trajectories. Released under OpenMDW for commercial use.

![EgoHand inference pipeline.](docs/egohand_pipeline.svg)

<a id="setup"></a>

## 🛠️ Setup

### System requirements

| Component | Requirement |
| --- | --- |
| OS | Ubuntu 24.04 |
| GPU | NVIDIA RTX 6000 Ada, L40, L20, or RTX PRO 6000 Blackwell |
| Memory | 32 GB |
| Storage | 25 GB |

### Clone and install

```bash
git clone https://github.com/NVIDIA/IsaacCapture.git
cd IsaacCapture/src/python/isaaccapture_egopost

uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch
uv pip install --python .venv/bin/python -r requirements.txt

# Install DINOv3
uv pip install --python .venv/bin/python --no-deps -r requirements-dinov3.txt
.venv/bin/python -c "from dinov3.models.vision_transformer import DinoVisionTransformer"
```

### Prepare your inputs

| Input | What you need |
| --- | --- |
| **Video** | An `.mp4` file, or a folder of `.mp4` files. |
| **Camera calibration** | A JSON following the [camera input format](#camera-input-formats). See the [example](#example) for a sample calibration file. |
| **2D hand bounding boxes** | Boxes from your own detector, or the optional [RT-DETR Hand Detector-v1](https://huggingface.co/nvidia/RT-DETR-Hand-Detector-v1). See [Optional features](#optional-features). |
| **EgoHand checkpoint** | [EgoHand-v1](https://huggingface.co/nvidia/EgoHand-v1) |
| **SOMA-X hand assets** | Required files from [SOMA-X](https://huggingface.co/nvidia/SOMA-X): `SOMAHand.npz`, `SOMA_neutral.npz`, `SOMA_template_rig.usda`, `SOMA_procedural_transforms.json`, `MHR/mhr_model_lod1.pt`, `MHR/base_body_lod1.obj`, and `MHR/SOMA_wrap_lod1.obj`. |


Set `HF_TOKEN` to your Hugging Face token for higher download rate limits.

<!-- #### Hugging Face access checklist

Before your first run, make sure you have access to every repo you'll need:

- [EgoHand-v1](https://huggingface.co/nvidia/EgoHand-v1)
- [RT-DETR Hand Detector-v1](https://huggingface.co/nvidia/RT-DETR-Hand-Detector-v1)
- [SOMA-X](https://huggingface.co/nvidia/SOMA-X)

If a repo's page shows a request-access prompt, request access first. Then
create a token with **Read** access at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
Run `.venv/bin/hf auth login` to authenticate -- this persists across terminals and
reboots, unlike `export HF_TOKEN=...`, which only lasts for that shell
session. Verify with `.venv/bin/hf auth whoami`. See
[Hugging Face authentication](inference/hand_detector_2d/README.md#authenticate-with-hugging-face)
for troubleshooting, or the
[Docker guide](inference/docker/README.md#huggingface-fallback-and-authentication)
for container-specific instructions.

Need a sample video? Track 3 of the
[NVIDIA Video-to-Data Challenge](https://huggingface.co/datasets/nvidia/video_to_data_challenge)
provides egocentric videos. The [example below](#example)
downloads an RGB video from `track_3/public` and uses the included calibration. -->


### Example

Download an egocentric video from the [NVIDIA Video-to-Data Challenge](https://huggingface.co/datasets/nvidia/video_to_data_challenge) with the [camera calibration](docs/nvidia_v2d_ego_cam_a.json).

```bash
# Download the video from Huggingface.
.venv/bin/hf download nvidia/video_to_data_challenge \
  --repo-type dataset \
  --local-dir nv_v2d_challenge \
  track_3/public/videos/chunk-000/observation.images.ego_cam_a/episode_000045.mp4
```

```bash
PYOPENGL_PLATFORM=egl PYTHONPATH="$(pwd)" .venv/bin/python \
  inference/scripts/run_egohand_inference_pipeline.py \
  --input nv_v2d_challenge/track_3/public/videos/chunk-000/observation.images.ego_cam_a/episode_000045.mp4 \
  --bbox_detector_config inference/hand_detector_2d/configs/rtdetr.yaml \
  --intrinsics_json docs/nvidia_v2d_ego_cam_a.json \
  --perspective_cropping \
  --draw_skeleton --no_mesh \
  --dtype bfloat16 \
  --output_dir outputs/nv_v2d_challenge_example
```

**Example output:**

<p align="center">
  <img src="docs/nvidia_challenge_public_episode_000045.jpg" width="480" alt="NVIDIA public episode_000045 skeleton overlay with perspective cropping">
</p>


Results are saved under `--output_dir`:

```text
<output_dir>/
└── <video_stem>/
    ├── <video_stem>_left_bbox.json   # 2D detector output
    ├── <video_stem>_right_bbox.json
    ├── left_results.npz              # Per-hand poses, joints, vertices, translation
    ├── right_results.npz
    ├── perf.json                     # Model loading and per-video stage timings
    ├── fullimg_overlay.mp4           # Reconstruction over the source video
    ├── bbox_viz.mp4                  # Hand bounding boxes
    └── quality_control.json          # Only with --postprocessing_qc
```

See the [CLI reference](inference/scripts/README.md#output-layout) for the full
output layout and per-frame export options.


<a id="model-performance"></a>

## ⚡ Model performance

Use `--perf_output` to record pipeline latency. See the [performance guide](inference/perf/README.md) for
measurement details and stage definitions.

The tables below show reference timings for the example, measured in
milliseconds per frame.

- **Model inference latency** is the total time spent on the model itself: preparing
  the input batch, running the forward pass, and preparing the outputs.
- **Full pipeline latency** adds input preprocessing (such as cropping) and
  output smoothing on top of model inference, giving the end-to-end latency of the pipeline.

#### NVIDIA L40

| Stage | Perspective cropping | Without perspective cropping |
| --- | ---: | ---: |
| `preprocessing` | 6.861 | 3.891 |
| `batch_assembly` | 0.287 | 0.279 |
| `model_forward` | 15.172 | 15.167 |
| `post_forward` | 2.887 | 2.880 |
| `adaptive_smoothing` | 2.898 | 2.744 |
| **Model inference latency** | **18.346** | **18.326** |
| **Full pipeline latency** | 28.376 | 25.465 |

#### NVIDIA RTX PRO 6000 Blackwell:

| Stage | Perspective cropping | Without perspective cropping |
| --- | ---: | ---: |
| `preprocessing` | 6.971 | 2.721 |
| `batch_assembly` | 0.365 | 0.364 |
| `model_forward` | 6.870 | 6.894 |
| `post_forward` | 1.777 | 1.802 |
| `adaptive_smoothing` | 1.856 | 1.776 |
| **Model inference latency** | **9.012** | **9.060** |
| **Full pipeline latency** | 18.006 | 13.876 |

<a id="usage"></a>

## 🚀 Detailed Usage

### Camera input formats

Save your camera calibration in a JSON file and provide its path using
`--intrinsics_json`. For a pinhole camera, use the following format:

```json
{
  "model": "PINHOLE", "image_width": 1920, "image_height": 1080,
  "camera_matrix": {
    "matrix_3x3": [
      [1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0],
      [0.0, 0.0, 1.0]
    ]
  },
  "distortion_coefficients": []
}
```

Supported camera models and their `distortion_coefficients` are listed below.

| Camera model | `distortion_coefficients` | Description |
| --- | --- | --- |
| `PINHOLE` | `[]` | Pinhole camera without lens distortion. |
| `KB4` | `[k1, k2, k3, k4]` | OpenCV equidistant fisheye. |
| `FISHEYE624` | `[k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3]` | Project Aria fisheye, used in [HOT3D](https://facebookresearch.github.io/hot3d/). |
| `RATIONAL_POLYNOMIAL` | `[k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tau_x, tau_y]` | Rational polynomial with 14 coefficients, used in the [NVIDIA Video-to-Data Challenge](https://huggingface.co/datasets/nvidia/video_to_data_challenge). |

If camera intrinsics are unavailable, omit `--intrinsics_json`. The pipeline estimates the focal length as (`max(width, height)`), which may reduce reconstruction accuracy. Using calibrated camera intrinsics is recommended.

<a id="optional-features"></a>

## ✨ Optional features

| Feature | When to use it | Enable it / learn more |
| --- | --- | --- |
| **2D detector** | Run RT-DETR or reuse saved boxes from your detector or ground truth. | Set `--bbox_detector_config` to [`rtdetr.yaml`](inference/hand_detector_2d/configs/rtdetr.yaml) or [`saved_bboxes.yaml`](inference/hand_detector_2d/configs/saved_bboxes.yaml). For saved boxes, export the [per-hand JSON format](inference/hand_detector_2d/README.md#use-your-own-detector-or-ground-truth-boxes). See [detector setup](inference/hand_detector_2d/README.md#use-the-bundled-rt-detr-detector) for details. |
| **Perspective cropping** | Recommended for all cameras. Maps each hand to a virtual pinhole view, correcting lens distortion. | Add `--perspective_cropping` with the [shared camera JSON](#camera-input-formats) in `--intrinsics_json`.<br>Without the flag: detect and crop on original `PINHOLE` frames; for other models, undistort the full image first, then detect and crop on it. |
| **Adaptive smoothing** | Reduce temporal jitter while keeping exported geometry consistent. | On by default; disable with `--no-postprocessing_smooth`. [Smoothing guide](inference/postprocessing/README.md#adaptive-hand-smoothing) |
| **3D quality report** | Per-video report on motion, hand geometry, and missing predictions. | Add `--postprocessing_qc`. [QC guide](inference/postprocessing/README.md#3d-quality-control) |
| **World-coordinate projection and stabilization** | You have a camera-to-world trajectory and want to refine it using both hands. | Run after inference. [Inputs, command, and outputs](inference/postprocessing/README.md#world-coordinates-and-camera-pose-refinement) |


<a id="docker"></a>

## 🐳 Docker

See the [Docker guide](inference/docker/README.md) for a combined inference
image or separate detector and reconstruction images.


## 📚 Citation

If you use EgoHand in your work, please cite the following paper:

```bibtex
@misc{EgoHand2026,
  title={EgoHand: Monocular 3D Hand Reconstruction from Egocentric Video},
  author={Chen, Chao-Yeh and Chen, Tiffany and Duan, Lin and Hu, Tianyi and
    Huang, Yu-Hsiang and Iqbal, Umar and Li, Jiefeng and Lim, Edy and
    Liu, Yanchen and Pang, Yihan and Qiu, Hang and Saito, Jun and Su, Shao
    and Wang, Qingsi and Xie, Xiufeng},
  journal={arXiv:2610.30293},
  year={2026}
}
```


<a id="license"></a>

## 📄 License

EgoHand is licensed under [OpenMDW-1.1](../../../LICENSES/LicenseRef-OpenMDW-1.1.txt). See
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for third-party licenses.

**NOTICE AND DISCLAIMER:** This software automatically retrieves, accesses, or
interacts with external materials. These materials are not distributed with
the software and are governed by their own terms, conditions, and licenses.
You are solely responsible for identifying, reviewing, and complying with
those requirements, and for verifying the security, integrity, and suitability
of the materials for your use case.

This software is provided "AS IS", without warranty of any kind. The author
makes no representations or warranties regarding external materials and
assumes no liability for any losses, damages, liabilities, or legal
consequences arising from your use of, or inability to use, the software or
those materials. Use the software and external materials at your own risk.
