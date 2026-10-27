<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sensing AVS1 receiver

The receiver connects to the robot's AVS1/TCP sender, decodes HEVC with NVDEC,
and publishes decoded GPU rings to a local CUDA IPC socket. It opens no window.

Use the [host setup and receive commands](../../README.md). The default config
is `../../configs/s56-stereo-2560x1984-30-receiver.yaml` relative to this directory.
The default profile publishes `head.left` and `head.right` NV12 streams at
`/tmp/isaac-camera-streamer.sock`. Match receiver dimensions to the sender profile.

Run `camera_receiver.py --help` using the host environment for overrides such as
`--host`, `--port`, `--socket`, `--gpu-id` and diagnostic settings.
The `codec` extension is built separately for this host; do not copy it from Orin.
CUDA IPC consumers must run on the same machine and compatible GPU.
