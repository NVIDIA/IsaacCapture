<!--
SPDX-FileCopyrightText: Copyright (c) 2026 HTC Corporation. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vive SE3 Tracker Plugin

Pushes each VIVE Ultimate Tracker as its own generic SE3 (6-DoF) pose stream
(`se3_tracker.fbs`), one tensor collection per tracker — raw per-tracker poses
for the consumer to use directly.

## Data flow

```
Ultimate Trackers -> dongle -> VIVEHub tracker_server -> /tmp/vut.sock (VUT SDK)
    -> this plugin -> Se3TrackerPose per device -> tensor collections
        "vive_tracker_<serial>"  (e.g. vive_tracker_SN0001)
    -> core::Se3Tracker(collection_id) readers / MCAP recording / replay
```

Collections are created lazily on the first pose from each device and named by
the tracker's physical **serial number** — the only stable identity. See
"Identifying trackers" below.

## Reference frame

Poses are forwarded verbatim in the frame the VIVEHub daemon reports: the
SteamVR/OpenVR right-handed convention (X right, Y up, -Z forward), origin
defined by the tracker_server's own calibration. Per `se3_tracker.fbs` this is a
producer-defined reference frame (this producer is not XR-sourced); axis
conventions match OpenXR, but the origin is **not** the OpenXR session base
space — align downstream if cross-device consistency is needed.

## Timestamps

VUT pose timestamps are host `CLOCK_MONOTONIC` nanoseconds (`vut_types.h`) — the
same clock `SchemaPusher` documents as the local common clock — so valid samples
carry the VUT sample time verbatim (no push-loop resampling bias). The VUT SDK
does not expose the dongle's raw clock; the local common clock value is passed
as the documented best-effort substitute.

Push policy per device, per ~90 Hz tick:

| condition | action |
|---|---|
| new sample since last push | push `is_valid=true` at the VUT sample time |
| unchanged & fresh | push nothing (readers retain last-known; no duplicate timestamps in MCAP) |
| stale (> `VIVE_SE3_STALE_MS`) | push `is_valid=false` + identity filler every tick, stamped now |

## Configuration (environment)

| variable | default | meaning |
|---|---|---|
| `VIVE_VUT_SOCKET` | `/tmp/vut.sock` | VIVEHub VUT daemon socket path |
| `VIVE_SE3_STALE_MS` | `250` (ms; 1..3600000, else default) | staleness threshold before a tracker is pushed invalid |
| `VIVE_SE3_COLLECTIONS_FILE` | `$XDG_RUNTIME_DIR/vive_se3_collections.txt` (else `/tmp/…`) | where live collection ids are advertised |
| `VIVE_SE3_SYNTHETIC` | (unset) | `1` = fake trackers, no VIVEHub (smoke test) |

No role or naming knobs: collections are always named by serial.

## Identifying trackers (serial from the wire)

Requires the **VUT SDK (VIVEHub >= 1.0.1)**, whose pose event carries the
tracker's serial (`Pose::serial`). The plugin names each collection by that
serial:

- pose with a serial → `vive_tracker_<serial>` (e.g. `vive_tracker_SN0001`);
- pose with an empty serial (`""`) → `vive_tracker_<device_id>` (logged fallback).

The name is fixed on the **first** pose from a device (a live tensor collection
cannot be renamed). The serial is the stable physical identity, so an MCAP
recorded today still identifies exactly which tracker each channel came from.
Mapping a tracker to a role (which one is which) belongs downstream, wherever the
poses are consumed.

> This plugin builds against the VUT SDK; VIVEHub **1.0.1** is the first release
> shipping it (1.0.0 predates the rename and will not build or connect). Run
> VIVEHub 1.0.1 or newer.

## Running

```bash
# v1.0.1 (or newer) VIVEHub tracker_server running + trackers paired; CloudXR runtime up.
./vive_se3_tracker_plugin
```

Verify with the stock SE3 reader (per collection):

```bash
./se3_printer vive_tracker_SN0001   # a serial the plugin prints on startup
                                          # (or read one from the collections file)
```

## Collection auto-discovery

While running, the plugin writes its live collection ids (one per line) to a
per-user file under `$XDG_RUNTIME_DIR` (falling back to `/tmp`), overridable via
`VIVE_SE3_COLLECTIONS_FILE`. It is rewritten as collections appear and removed on
exit (including on Ctrl+C / SIGTERM). This lets readers attach without knowing
device_ids in advance — `record_se3_vive.py` resolves the same path so it can run
with no arguments. Stale files from a crashed run are cleared on the next startup.

## Recording / replay (Python examples)

Both run with no arguments once the plugin is up, in a venv set up as in
[Running an example](../../../examples/README.md#running-an-example):

```bash
uv pip install -e ./examples/mcap_record_replay
# records 10 s of every advertised collection to ./recordings/<timestamp>.mcap
python -m isaaccapture_examples.mcap_record_replay.record_se3_vive
# replays the newest recording, auto-discovering collections + capture rate,
# and serves a viser 3D view at http://localhost:8080
python -m isaaccapture_examples.mcap_record_replay.replay_se3_vive
```

Under the hood this is just the standard MCAP tooling — a `core::Se3Tracker(cid)`
per collection registered with a `DeviceIOSession`; the live/replay factories
handle serialization (`Se3TrackerRecordingTraits`, `ReplaySe3TrackerImpl`).
