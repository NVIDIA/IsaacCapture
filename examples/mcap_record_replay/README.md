<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MCAP record / replay

Record DeviceIO tracking to an MCAP file and replay it into a viser 3D view.
Live viewers are included for watching without recording.

```bash
uv pip install -e ./examples/mcap_record_replay
python -m isaaccapture_examples.mcap_record_replay.record_hand      # 5 s
python -m isaaccapture_examples.mcap_record_replay.replay_hand      # newest take
```

Recordings are written to `./recordings/` relative to where you run the
command; a replay given no path picks the newest matching recording there
(e.g. `replay_hand` picks the newest `hands_*.mcap`) and exits if none
match -- no silent fallback to a wrong-type recording.

`uv run` (instead of an activated venv) needs an explicit `--python 3.11` --
without it, `uv` may resolve against the system Python and fail to match the
lockfile.

The live and replay viewers bind every interface, so a browser on another
machine can reach them at `http://<this-host>:8080`. Pass `--host 127.0.0.1` to
keep a viewer local. Replay repeats until Ctrl+C; no loop flag is needed.

| Channel | Live | Record | Replay |
| --- | --- | --- | --- |
| Hands (OpenXR or SOMA) | `live_hand` | `record_hand` | `replay_hand` |
| Controllers | `live_controller` | `record_controller` | `replay_controller` |
| Full body (FullBodyPose or SOMA) | `live_full_body` | `record_full_body` | `replay_full_body` |
| Raw hand joint SE3 poses | `live_joint_se3_pose` | `record_joint_se3_pose` | `replay_joint_se3_pose` |
| VIVE SE3 trackers | — | `record_se3_vive` | `replay_se3_vive` |

`record_*` takes an optional duration in seconds and an optional output path.
Recording needs a live OpenXR runtime; replay needs only the file.

The body commands default to `--body-schema full-body-pose`, including
controllers. The hand commands default to `--hand-schema openxr-hand-pose`.
Select SOMA with `--body-schema soma` or `--hand-schema soma`. These options
and the `--soma-*` options below match the DeviceIO live viewer.

SOMA body mode uses only the body stream by default, including any available
finger joints. Pass `--controllers` to include controller recording, or
`--no-controllers` to disable it for FullBodyPose. Standalone hand streams use
the separate hand commands; they are not extracted from or fused with the body.

## SOMA body

Install the consumer and the existing demo publisher into your configured
development environment:

```bash
uv pip install -e ./examples/mcap_record_replay
uv pip install -e ./examples/soma_body_publisher
```

Start the live-only viewer, then connect a CloudXR client:

```bash
uv run --no-sync python -m isaaccapture_examples.mcap_record_replay.live_full_body \
  --accept-eula --body-schema soma --soma-body-representation joint-rotations
```

In another terminal, start the publisher after the runtime is running:

```bash
source ~/.cloudxr/run/cloudxr.env
uv run --no-sync python -m isaaccapture_examples.soma_body_publisher \
  --body-representation joint-rotations --loop
```

Record the received SOMA data in a third terminal:

```bash
uv run --no-sync python -m isaaccapture_examples.mcap_record_replay.record_full_body \
  10 recordings/soma_body_joint-rotations_demo.mcap \
  --accept-eula --body-schema soma --soma-body-representation joint-rotations
```

Stop the live viewer and publisher before replaying on the same Viser port.
Replay does not need a runtime or client:

```bash
uv run --no-sync python -m isaaccapture_examples.mcap_record_replay.replay_full_body \
  recordings/soma_body_joint-rotations_demo.mcap \
  --body-schema soma --soma-body-representation joint-rotations
```

For evaluated joint poses, use `--soma-body-representation joint-poses` on
the viewer, recorder, and replay commands, and `--body-representation joint-poses`
on the publisher. Use a separate output file, such as
`recordings/soma_body_joint-poses_demo.mcap`. Both paths render
SOMA's native 77-joint topology, including available finger joints. Unavailable
joints and their connecting bones are hidden.

Recording stores the received schema unchanged. Joint rotations use the
existing `SomaBodySource` FK helper with a prepared neutral model when viewing;
joint poses map directly without loading a model or running FK.

`--soma-body-collection-id` selects the input collection and MCAP channel base.
It defaults to `soma_body_demo`, matching the publisher's `--collection-id`.
Recordings contain
`<collection>/soma_body_joint_rotations` and its `_tracked` channel, or the
corresponding `soma_body_joint_poses` pair. Replay consumes the tracked channel.
Without a file argument, replay selects the newest recording matching its
representation under `./recordings/`.

Replay advances one tracked sample per update. Viewer pacing defaults to the
selected channel's average capture rate; `--rate` overrides it (`0` is unpaced).
This does not resample or synchronize streams. FullBodyPose and OpenXR hand
replay retain their existing 60 Hz pacing and duration-based loop.

## SOMA hands

Install the existing hand demo publisher alongside the MCAP example:

```bash
uv pip install -e ./examples/mcap_record_replay
uv pip install -e ./examples/soma_hand_publisher
```

Start the live-only viewer and connect a CloudXR client:

```bash
uv run --no-sync python -m isaaccapture_examples.mcap_record_replay.live_hand \
  --accept-eula --hand-schema soma --soma-hand-representation joint-rotations
```

After the runtime is running, start the publisher in a second terminal:

```bash
source ~/.cloudxr/run/cloudxr.env
uv run --no-sync python -m isaaccapture_examples.soma_hand_publisher \
  --hand-representation joint-rotations --loop
```

Record in a third terminal. Stop the viewer and publisher before replaying on
the same Viser port:

```bash
uv run --no-sync python -m isaaccapture_examples.mcap_record_replay.record_hand \
  10 recordings/soma_hand_joint-rotations_demo.mcap \
  --accept-eula --hand-schema soma --soma-hand-representation joint-rotations

uv run --no-sync python -m isaaccapture_examples.mcap_record_replay.replay_hand \
  recordings/soma_hand_joint-rotations_demo.mcap \
  --hand-schema soma --soma-hand-representation joint-rotations
```

For evaluated poses, use `--soma-hand-representation joint-poses` on the
viewer, recorder, and replay commands, and `--hand-representation joint-poses`
on the publisher. Use a separate output file. The viewer renders each hand's
native 25-joint Wrist/finger skeleton, with left green and right blue. Joint rotations
use `SomaHandSource` FK against prepared neutral models; joint poses map directly
without loading models. Recording does not run FK in either mode.

`--soma-left-hand-collection-id` and `--soma-right-hand-collection-id` default to
`soma_hand_left_demo` and `soma_hand_right_demo`, matching the publisher's
`--left-collection-id` and `--right-collection-id`. Each collection is recorded
separately with its anatomical `hand_side` intact. Replay reads its
`soma_hand_joint_rotations_tracked` or `soma_hand_joint_poses_tracked` channel.
Without a file argument, it selects the newest
`soma_hand_<representation>_*.mcap` under `./recordings/`.

Left and right trackers advance independently once per replay update, following
the existing native behavior. Viewer pacing defaults to the busier selected
hand's average capture rate; `--rate` overrides it. Missing or exhausted hands
are hidden, and each pass ends when the longer channel finishes. No timestamp
synchronization, resampling, or body/hand fusion is performed.

A C++ recorder lives in `cpp/`. Docs:
`docs/source/references/mcap_record_replay.rst`.
