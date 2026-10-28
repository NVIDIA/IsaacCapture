# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fractions import Fraction
import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1])
actual_signatures = {
    "ColorLeft": sys.argv[2],
    "ColorRight": sys.argv[3],
}
try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(
        f"Invalid requested profile manifest {manifest_path}: {error}"
    ) from error

if manifest.get("version") != 1:
    raise SystemExit(
        f"Unsupported requested profile manifest version in {manifest_path}"
    )
streams = manifest.get("streams")
expected_names = set(actual_signatures)
if not isinstance(streams, dict) or set(streams) != expected_names:
    raise SystemExit(
        f"Requested profile manifest must contain exactly {sorted(expected_names)}"
    )

codec_names = {"mjpg": "mjpeg", "h264": "h264", "h265": "hevc"}
for stream_name, signature in actual_signatures.items():
    requested = streams[stream_name]
    if not isinstance(requested, dict):
        raise SystemExit(f"Invalid requested profile for {stream_name}")
    requested_format = requested.get("format")
    if requested_format not in codec_names:
        raise SystemExit(
            f"Invalid requested format for {stream_name}: {requested_format!r}"
        )
    for field in ("width", "height", "fps"):
        if type(requested.get(field)) is not int or requested[field] <= 0:
            raise SystemExit(
                f"Invalid requested {field} for {stream_name}: {requested.get(field)!r}"
            )
    parts = signature.split(",")
    if len(parts) != 4:
        raise SystemExit(f"Invalid ffprobe signature for {stream_name}: {signature!r}")
    codec, width, height, rate = parts
    try:
        actual_width = int(width)
        actual_height = int(height)
        actual_rate = Fraction(rate)
    except (ValueError, ZeroDivisionError) as error:
        raise SystemExit(
            f"Invalid ffprobe signature for {stream_name}: {signature!r}"
        ) from error
    if actual_rate <= 0:
        raise SystemExit(f"Invalid ffprobe rate for {stream_name}: {rate!r}")
    expected_media = (
        codec_names[requested_format],
        requested["width"],
        requested["height"],
    )
    actual_media = (codec, actual_width, actual_height)
    if actual_media != expected_media:
        raise SystemExit(
            f"{stream_name} recorded profile {signature!r} does not match request "
            f"{requested_format},{requested['width']},{requested['height']},{requested['fps']}/1"
        )
    # Elementary-stream VUI clocks may use multiple ticks per frame. MCAP's
    # declared profile and raw device timestamps determine capture FPS.
print(
    "Requested codec/resolution match ColorLeft and ColorRight: OK; "
    "capture FPS is checked from MCAP metadata and device timestamps"
)
