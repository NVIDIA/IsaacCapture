# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import tomllib
from pathlib import Path

path = Path(sys.argv[1])
try:
    with path.open("rb") as stream:
        document = tomllib.load(stream)
except (OSError, tomllib.TOMLDecodeError) as error:
    raise SystemExit(f"Invalid record configuration {path}: {error}") from error

if set(document) != {"record"} or not isinstance(document["record"], dict):
    raise SystemExit(f"{path}: the only top-level entry must be a [record] table")
record = document["record"]
scalar_options = {
    "duration": (int, "--duration"),
    "output": (str, "--output"),
    "format": (str, "--format"),
    "width": (int, "--width"),
    "height": (int, "--height"),
    "fps": (int, "--fps"),
    "device_uid": (str, "--device-uid"),
    "reconnect_timeout": (int, "--reconnect-timeout"),
    "reconnect_interval_ms": (int, "--reconnect-interval-ms"),
    "mcap_media": (str, "--mcap-media"),
    "preset": (str, "--preset"),
    "plugin": (str, "--plugin"),
}
boolean_options = {
    "record_device_diagnostics": (
        "--record-device-diagnostics=on",
        "--record-device-diagnostics=off",
    ),
    "imu": ("--imu", "--no-imu"),
    "audio": ("--audio", "--no-audio"),
    "preview": ("--preview", "--no-preview"),
    "keep_media_sidecars": ("--keep-media-sidecars", "--no-keep-media-sidecars"),
}
allowed = set(scalar_options) | set(boolean_options) | {"plugin_options"}
unknown = sorted(set(record) - allowed)
if unknown:
    raise SystemExit(f"{path}: unknown [record] option(s): {', '.join(unknown)}")

arguments = []
for name, (expected_type, option) in scalar_options.items():
    if name not in record:
        continue
    value = record[name]
    if type(value) is not expected_type:
        raise SystemExit(f"{path}: record.{name} must be {expected_type.__name__}")
    if isinstance(value, str) and (not value or "\0" in value or "\n" in value):
        raise SystemExit(
            f"{path}: record.{name} must be a non-empty single-line string"
        )
    arguments.extend((option, str(value)))

for name, (enabled, disabled) in boolean_options.items():
    if name not in record:
        continue
    value = record[name]
    if type(value) is not bool:
        raise SystemExit(f"{path}: record.{name} must be true or false")
    arguments.append(enabled if value else disabled)

plugin_options = record.get("plugin_options", [])
if not isinstance(plugin_options, list) or any(
    type(value) is not str for value in plugin_options
):
    raise SystemExit(f"{path}: record.plugin_options must be an array of strings")
for value in plugin_options:
    if not value.startswith("--") or "\0" in value or "\n" in value:
        raise SystemExit(
            f"{path}: each record.plugin_options entry must be a single-line --option"
        )
    arguments.extend(("--plugin-option", value))

sys.stdout.buffer.write(b"\0".join(value.encode() for value in arguments))
if arguments:
    sys.stdout.buffer.write(b"\0")
