<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Example remaps

`cam4.remap` and `cam5.remap` are example sender-side wrist undistortion maps for
the default camera config. They use approximate SHW5G fisheye intrinsics from a
reference rig; the wrist values reuse averaged head intrinsics. They are suitable
for demonstrating the viewing pipeline, and have no per-unit calibration accuracy
or stereo extrinsics. Calibrate your own cameras for metric work.

The files are copied as real assets into this example and tracked with Git LFS.
The sender keeps its calibration assets under `argus_sender/remaps/`. The config resolves
`../remaps/cam4.remap` relative to `argus_sender/configs/cameras.yaml`.

| Property | Example value |
| --- | --- |
| Calibration image size | 640x496 |
| Captured NV12 source | 1280x992, unrotated |
| Remap lookup grid | 1280x960 |
| Sender output | 640x480 NV12 |
| Filtering | Bilinear lookup followed by 2x box downsampling |
| Projection | Wrist-left balance=0 pinhole target, shared by both wrist maps |

The generator follows the reference setup's projection: compute a balance=0
undistorted 640x496 target, center-crop its principal point to 640x480, then scale
the target K and lookup grid by 2. Each camera's source K is also scaled by 2;
fisheye D is unchanged. The sender performs the box downsampling. Changing this
to a 640x480 lookup grid changes the filtering and image quality.

To regenerate on a host with Python, NumPy and OpenCV:

```bash
python3 -m venv /tmp/camera-remaps
/tmp/camera-remaps/bin/pip install 'numpy>=1.26,<2' 'opencv-python-headless>=4.9,<4.12'
/tmp/camera-remaps/bin/python argus_sender/remaps/generate_example.py
```

Pass `--calibration /path/to/calibration.json --output /path/to/remaps` for another
calibration. The example generator expects `image_size: [640, 496]` and
`cameras.wrist_left` / `cameras.wrist_right`, each with a 3x3 `K` and four-element
fisheye `D`. For different sensor geometry, generate compatible RMAPs and update
the sender config together. Device IDs in this example are 4 and 5.

`manifest.json` records calibration provenance, geometry and SHA256 checksums.
The RMAP v1 binary layout is little endian: `RMAP`, uint32 grid width, uint32 grid
height, float32 balance, then row-major float32 X and Y arrays. Coordinates address
the captured source; the file size is `16 + width * height * 8` bytes. The sender
checks geometry and rejects maps with excessive out-of-source coordinates.
