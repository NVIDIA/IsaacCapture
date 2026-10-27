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
| Hands | `live_hand` | `record_hand` | `replay_hand` |
| Controllers | `live_controller` | `record_controller` | `replay_controller` |
| Full body | `live_full_body` | `record_full_body` | `replay_full_body` |
| Raw hand joint SE3 poses | `live_joint_se3_pose` | `record_joint_se3_pose` | `replay_joint_se3_pose` |
| VIVE SE3 trackers | — | `record_se3_vive` | `replay_se3_vive` |

`record_*` takes an optional duration in seconds and an optional output path.
Recording needs a live OpenXR runtime; replay needs only the file.

A C++ recorder lives in `cpp/`. Docs:
`docs/source/references/mcap_record_replay.rst`.
