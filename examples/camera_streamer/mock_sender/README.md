<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# mock_sender

Camera-free stand-in for `argus_sender`. It serves AVS1 HEVC over TCP so the
receiver, CUDA IPC consumers and anything downstream can run without a Jetson
or cameras. Like the real sender it is the TCP server, serves one receiver at a
time, and starts each connection with a new session at an IDR.

Requires `ffmpeg` with `libx265` and `drawtext`. Test streams (a moving pattern
with a burned-in `name frame` counter, IDR every 2 s, no B-frames) are encoded
once into `~/.cache/camera_streamer_mock_sender`.

```bash
python3 mock_sender.py                       # head 2560x1984, wrist1/2 640x480, 30 fps
python3 mock_sender.py --stream head:1280x960@30 --stream wrist:640x480@15
./../camera_receiver/run.sh --config mock_receiver.yaml   # receiver for the default streams
```

Streams are mono and appear in the order given; the receiver config must list
the same names, sizes and order. Streams at different rates share one base-rate
tick, as in the real sender.

| Option | Effect |
| --- | --- |
| `--pause NAME@START+SECONDS` | Stop one stream; it resumes at the next IDR. |
| `--disconnect-at SECONDS` | Close the receiver connection after that long. |
| `--clock-offset-ms`, `--clock-drift-ppm` | Offset (default 0, as with chrony-synced clocks) and drift of the synthetic sender clock. |
| `--sof-lag-ms` | Synthetic sensor start-of-frame to capture-ready lag (default 20, range 0-500). |
| `--timing-ext` | Emit the AVS1 timing extension: per frame a qualified sensor capture instant on the sender's monotonic clock, the sender's stated uncertainty and anchor age. The receiver in this tree parses it; an unmodified receiver rejects the bundle flag. |
| `--sm-offset-ms`, `--sm-drift-ppm` | Offset and drift of the sender's monotonic clock from this host's. The UDP clock responder and the timing extension are both on this clock. |
| `--sm-step AT_S:DELTA_MS` | Step the sender's monotonic clock this long after the receiver connects; repeatable. |
| `--sensor-uncertainty-us` | The bound the extension states for the sensor-to-sender-clock mapping (default 500). |
| `--timing-invalid START_S+DURATION_S` | Mark the timing records not valid (no capture claim) in that window; repeatable. |
| `--ground-truth FILE` | Append one JSON line per published frame: the true capture instant and the stamps, on this host's monotonic clock, for checking a consumer's mapped capture time. |
| `--clock-port PORT` | Serve the sender clock responder on this UDP port (0, the default, disables it). `--clock-bind` (127.0.0.1). Alive for the whole process, independent of media connections. |
| `--clock-mute START_S+DURATION_S` | Stop answering clock requests in that window (seconds after the mock starts); repeatable. |
| `--clock-loss F`, `--clock-reorder F` | Leave a fraction of requests unanswered, or send a fraction of replies after the next one. |

Timing fields come from the synthetic clock only. Every frame carries a
synthetic sensor start-of-frame TSC, in a TSC domain of the mock's own, so the
receiver derives `sensor_sof_ns` (capture-ready minus the lag) as it does for
Argus. Argus frame timestamps are never marked valid, so `sof_delta` stays
`n/a`. Nothing here measures real sensor timing. Frame numbers count published
frames per stream.

## Clock responder

`--clock-port` serves one UDP port, stateless and unauthenticated, as the real sender does. A datagram of exactly
13 bytes, `Q`, u32 sequence, i64 t1 (little-endian), is answered to its source with 29 bytes: `R`, sequence, t1, t2,
t3, with t2 taken at receipt and t3 just before the send, in nanoseconds on the sender's monotonic clock
(`--sm-offset-ms`, `--sm-drift-ppm`, `--sm-step`), the clock `sensor_capture_sm_ns` is on. Anything else is
ignored silently.

Tests: `python3 -m unittest discover -s mock_sender/tests` and
`python3 -m unittest tests.test_mock_sender_receiver_wire`, both from this
directory's parent. The full mock-to-CUDA-IPC pipeline (needs a GPU):
`camera_receiver/.venv/bin/python -m unittest tests.test_mock_to_ipc_pipeline`.
