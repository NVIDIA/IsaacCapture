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
| `--clock-offset-ms`, `--clock-drift-ppm` | Offset and drift of the synthetic sender clock. |

Timing fields come from the synthetic clock only. SOF/TSC and Argus frame
timestamps are never marked valid, so consumers see `sof_delta=n/a`, and
capture-to-consumer latencies include the clock offset. Nothing here measures
real sensor timing. Frame numbers count published frames per stream.

Tests: `python3 -m unittest discover -s mock_sender/tests` and
`python3 -m unittest tests.test_mock_sender_receiver_wire`, both from this
directory's parent.
