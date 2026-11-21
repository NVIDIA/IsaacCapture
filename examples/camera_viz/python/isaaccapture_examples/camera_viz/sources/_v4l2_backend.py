# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check OpenCV's V4L2 support without opening a camera or initializing CUDA."""

import sys
from types import ModuleType


def load_v4l2() -> ModuleType:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "V4L2 capture requires opencv-python. Run camera_viz.sh setup "
            'or install "opencv-python>=4.5,<5" in the camera environment.'
        ) from exc

    if not cv2.videoio_registry.hasBackend(cv2.CAP_V4L2):
        raise RuntimeError(
            f"OpenCV {cv2.__version__} has no available V4L2 backend. "
            "Run camera_viz.sh setup to install the supported OpenCV wheel, "
            "or use a Linux OpenCV build with WITH_V4L=ON."
        )
    return cv2


def main() -> int:
    try:
        cv2 = load_v4l2()
    except Exception as exc:
        print(f"V4L2 check failed: {exc}", file=sys.stderr)
        return 1
    print(f"OpenCV {cv2.__version__}: V4L2 backend available")
    return 0


if __name__ == "__main__":
    sys.exit(main())
