<!--
SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: Apache-2.0
-->

# LiveKit camera preview and recording

Camera video can reach the PC through a LiveKit room instead of through CaptureServer. CaptureServer only
says where the video is; the browser (preview) and the backend (recording) each join the room themselves.
The publisher is the Orin's sender, or `tools/livekit_orin_emulator.py` when no Orin is available.

Run every command from `examples/egocap`.

## 1. Describe the streams

Write a YAML file naming the room, the publisher's identity and one entry per published track. The field
reference is in [`capture_server/README.md`](capture_server/README.md#the---livekit-streams-file).

```yaml
livekit:
  url: wss://LIVEKIT_HOST
  room: sensing-preview
  publisher: orin-sensing
  timestamp_basis: capture_unix_us
  streams:
    - {name: Head left, track: sensing-left, width: 1280, height: 720, fps: 30}
```

- `track` must equal the name the publisher gives the track (the emulator's `livekit=NAME`).
- `publisher` must equal the identity in the publisher's join token.
- Preview works with any `timestamp_basis`. Recording needs `capture_unix_us` and `width`, `height` and
  `fps`; recording is refused while any offered stream lacks them.

Start CaptureServer with the file (add `--mock` for synthetic tracking):

```bash
uv run capture-server --mock --livekit-streams livekit.yaml
```

## 2. Create the publisher's join token

The publisher joins with a token for the room and identity from the file, with publish permission. With the
LiveKit CLI (`lk`), using the key pair of the LiveKit server:

```bash
lk token create --api-key "$LIVEKIT_API_KEY" --api-secret "$LIVEKIT_API_SECRET" \
  --join --room sensing-preview --identity orin-sensing --valid-for 24h > orin-token.txt
```

The CLI flags are not exercised by this repository's tests; check `lk token create --help` for your version.
The token file holds only the token. Keep it private.

## 3. Run the emulator (no Orin)

`tools/livekit_orin_emulator.py` joins the room and publishes one test-pattern track per `--add-stream`.

```bash
uv run python tools/livekit_orin_emulator.py \
  --livekit-url wss://LIVEKIT_HOST \
  --livekit-token-file orin-token.txt \
  --add-stream sensor=0,livekit=sensing-left \
  --width 1280 --height 720 --fps 30
```

`--add-stream sensor=N,livekit=NAME` is repeatable. `--timestamp-basis` takes `capture_unix_us` (default),
`processing_unix_us` or `monotonic_us`; set the same value in the streams file. Frames go through a software
encoder, so this checks the PC side and the metadata, not the Jetson encoder.

## 4. Preview and record on the PC

Give the backend the LiveKit server's key pair, then start it as in the root [`README.md`](README.md):

```bash
export LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=...
uv run uvicorn egocap_web.app:app --app-dir backend --port 8000
```

- The browser asks `GET /api/livekit-token` for a subscribe-only token, only for a room the current session
  offers, and shows the stream in the camera grid. Without the key pair the backend answers 503 and the page
  reports the preview as unavailable.
- Recording joins the room with its own subscribe-only token signed with the same key pair. It starts only
  when every offered stream is recordable (see step 1) and stops with a reason if a frame cannot be recorded
  honestly. The backend must run on the machine whose clock CaptureServer reports.
