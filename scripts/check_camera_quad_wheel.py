# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check optional camera-helper delivery in an actual project wheel.

Run against wheels built with BUILD_OVXR_CAMERA_QUADS ON and OFF, including
ON -> OFF -> ON in the same build directory to detect stale packaged data.
"""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument(
        "--camera-quads", action=argparse.BooleanOptionalAction, required=True
    )
    args = parser.parse_args()
    prefix = "isaaccapture/ovxr_session_manager/native/"
    expected = {
        prefix + "libisaaccapture_camera_panels.so",
    }
    with ZipFile(args.wheel) as wheel:
        members = {
            name
            for name in wheel.namelist()
            if name.startswith(prefix) and not name.endswith("/")
        }
        wanted = expected if args.camera_quads else set()
        if members != wanted:
            raise RuntimeError(
                f"Camera helper package differs: expected {sorted(wanted)}, got {sorted(members)}"
            )
        if any("/openxr_layer/" in name for name in wheel.namelist()):
            raise RuntimeError("Wheel contains obsolete OpenXR API-layer packaging")
    print(
        json.dumps(
            {
                "wheel": str(args.wheel),
                "sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
                "camera_quads": args.camera_quads,
                "result": "PASS",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
