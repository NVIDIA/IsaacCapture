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
| `--timing-ext`, `--sensor-uncertainty-us` | Send the AVS1 timing record on every bundle: the synthetic sensor start of frame (capture-ready minus the lag) on the mock's wall clock, with the stated uncertainty (default 500 us). The mock's wall clock is this host's clock plus `--clock-offset-ms` and `--clock-drift-ppm`. |

Timing fields come from the synthetic clock only. Every frame carries a
synthetic sensor start-of-frame TSC, in a TSC domain of the mock's own, so the
receiver derives `sensor_sof_ns` (capture-ready minus the lag) as it does for
Argus. Argus frame timestamps are never marked valid, so `sof_delta` stays
`n/a`. Nothing here measures real sensor timing. Frame numbers count published
frames per stream.

Tests: `python3 -m unittest discover -s mock_sender/tests` and
`python3 -m unittest tests.test_mock_sender_receiver_wire`, both from this
directory's parent. The full mock-to-CUDA-IPC pipeline (needs a GPU):
`camera_receiver/.venv/bin/python -m unittest tests.test_mock_to_ipc_pipeline`.
