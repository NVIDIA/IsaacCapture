<!--
SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: Apache-2.0
-->

# LiveKit camera preview and recording

Camera video can reach the PC through a LiveKit room instead of through CaptureServer. CaptureServer only
says where the video is; the browser (preview) and the backend (recording) each join the room themselves.
The publisher is the Orin's sender, or `tools/livekit_orin_emulator.py` when no Orin is available.

Recording LiveKit video and running the emulator need the optional `livekit` package: `uv sync --extra livekit`
(included in `--extra dev`). Without it the backend still starts and the browser preview works, but the Record
button is blocked for LiveKit streams with a message naming the extra.

Run every command from `examples/egocap`.

## 1. Describe the streams

Write a YAML file naming the room, the publisher's identity and one entry per published track. The field
reference is in [`capture_server/README.md`](capture_server/README.md#the---livekit-streams-file).

```yaml
livekit:
  url: wss://LIVEKIT_HOST
  room: sensing-preview
  publisher: orin-sensing
  timestamp_basis: monotonic_us
  streams:
    - {name: Head left, track: sensing-left, width: 1280, height: 720, fps: 30}
```

- `track` must equal the name the publisher gives the track (the emulator's `livekit=NAME`).
- `publisher` must equal the identity in the publisher's join token.
- `timestamp_basis` (default `monotonic_us`) must equal what the publisher stamps; the values are listed in the
  [streams file reference](capture_server/README.md#the---livekit-streams-file). Preview works with any of them.
  Recording needs a monotonic basis, `monotonic_us` or `capture_monotonic_us`, and `width`, `height` and `fps`;
  recording is refused while any offered stream lacks them.

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

`--add-stream sensor=N,livekit=NAME` is repeatable. `--timestamp-basis` takes `monotonic_us` (default),
`capture_monotonic_us`, `capture_unix_us` or `processing_unix_us`; set the same value in the streams file. The Orin's
sender (`--timestamp-basis=` on the sensing app) takes `monotonic_us` (default), `capture_monotonic_us` or
`processing_unix_us`. Both publishers answer the PC's clock-sync requests, which recording relies on. Frames go through a software
encoder, so this checks the PC side and the metadata, not the Jetson encoder.

## 4. Preview and record on the PC

Give the backend the LiveKit server's key pair, then start it as in the root [`README.md`](README.md):

```bash
export LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=...
uv run uvicorn egocap_web.app:app --app-dir backend --port 8000
```

- The browser asks `GET /api/livekit-token` for a viewer token, only for a room the current session
  offers, and shows the stream in the camera grid. The token can subscribe and send data messages (the clock-sync
  exchange needs them) but cannot publish media. The route is unauthenticated, so anyone who can reach the
  backend can send data messages into the room, so the publisher must act only on well-formed
  `isaaccapture.clock-sync` requests. Both publishers do: a packet is answered only if its topic is
  `isaaccapture.clock-sync` and it is exactly a 13-byte request starting with `Q`; everything else is ignored, and a
  reply goes only to the requester. The native sender keeps at most 16 pending requests and drops the oldest when
  more arrive. Neither has a per-second rate limit, and the emulator has no bound on pending replies, so keep the
  LiveKit room and the backend on a trusted network. Without the key pair the backend answers 503 and the page
  reports the preview as unavailable.
- Recording joins the room with its own viewer-capability token signed with the same key pair. It starts only
  when every offered stream is recordable (see step 1) and stops with a reason if a frame cannot be recorded
  honestly. The backend must run on the machine whose clock CaptureServer reports: recording is refused unless
  CaptureServer's URL is a loopback address.
- Recording maps the publisher's monotonic timestamps onto the PC's clock by sending the publisher clock-sync
  requests (data messages on topic `isaaccapture.clock-sync`) and reading its replies. The publisher must stay
  in the room and answer them; without replies the mapping goes stale.
