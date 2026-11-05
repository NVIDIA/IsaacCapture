<!--
SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: Apache-2.0
-->

# Live camera video over LiveKit

The camera host (an Orin running the SENSING publisher) publishes hardware-encoded H.264 into a LiveKit room.
CaptureServer says where that video is: it lists the room's cameras in the session's existing `streams` list, next to
its other cameras, from a small file you give it. The egocap backend passes the list on and issues viewer
tokens; the browser then connects to LiveKit itself to receive the video. No video passes through CaptureServer or
the backend, and CaptureServer never joins the room.

```text
Orin: SIPL -> hardware H.264 -> LiveKit publisher --> LiveKit server --> browser <video>
PC:   CaptureServer (--livekit-streams: where the streams are) -> egocap backend (list + viewer token)
```

**Recording.** A LiveKit stream is saved with a take only when its entry gives `width`, `height` and `fps` and
declares `timestamp_basis: capture_unix_us`, meaning the publisher stamps each frame with the mapped capture time of
the sensor event. If any offered stream does not qualify, recording is refused with that stream's name (remove it
from the streams file, or make it recordable); preview is unaffected. The native Orin sender stamps
`processing_unix_us` (host time after capture and conversion), so with it recording is refused until the sensor's
capture time is mapped. The recorder joins the room itself (it needs the same `LIVEKIT_API_KEY` and
`LIVEKIT_API_SECRET`) and checks each stream's first frame against the configured size before a take starts. The
recorder converts the publisher's Unix time to this machine's monotonic clock and applies one coarse check: a frame
that is more than 2 s old on arrival, or stamped in the future, ends the take. That catches gross clock disagreement
only; it does not prove the two machines' clocks agree to milliseconds, and nothing detects a clock step. Alignment
with tracking also assumes the backend and CaptureServer share one machine's clock. A take that passes is not
thereby proven aligned.

**Not yet exercised on the Orin.** The publisher, the browser's acceptance of its H.264 profile and the end-to-end
latency have not been run or measured. Report those as untested until someone has run them.

## 1. LiveKit server (on the PC or the LAN)

Run a LiveKit server reachable from both the Orin and the browser, and note its API key and secret. For a quick local
server, `livekit-server --dev --bind 0.0.0.0` uses the development key pair printed at start-up. Browsers need a
secure context for WebRTC on anything but `localhost`, so use a `wss://` URL with a certificate the browser trusts
when the browser is not on the server's own machine. Open the server's signalling and media ports between the machines.

## 2. Tell CaptureServer where the streams are

Write a file naming the room and its tracks (this is all CaptureServer needs; it never connects to LiveKit):

```yaml
livekit:
  url: wss://LIVEKIT_HOST          # what consumers connect to
  room: sensing-preview
  publisher: orin-sensing          # the publisher's participant identity
  streams:                         # up to eight
    - {name: Head left, track: sensing-left, width: 2560, height: 1984, fps: 30}
    - {name: Head right, track: sensing-right}
```

`width`, `height` and `fps` are optional and only describe what to expect; leave them out when unknown. Start
CaptureServer with `--livekit-streams FILE` in addition to its usual options. The entries reach the backend in the
session start message, so they appear once the backend's capture session has started. Offering a stream does not mean
frames are arriving.

## 3. Backend credentials

The backend signs viewer tokens with the server's key pair, read from its environment only:

```bash
export LIVEKIT_API_KEY=...      # the LiveKit server's API key
export LIVEKIT_API_SECRET=...   # its secret
```

`GET /api/livekit-token?room=ROOM` returns a token that can join that room and subscribe, nothing more. It answers 404
for a room the session does not offer and 503 when the key pair is not set. Like the camera preview routes it is open
to anyone who can reach the backend, so keep the backend on a trusted network. Missing credentials stop the video but
the stream list is still shown.

## 4. Orin publisher

Create a token file for the publisher: a LiveKit token for this room, with identity equal to `publisher` above,
allowed to publish (for example with LiveKit's `lk token create`; see its `--help`). Then add the LiveKit options to
the usual `camera_plugin_sensing` invocation, keeping the rig's own platform and configuration arguments:

```bash
camera_plugin_sensing ... \
  --livekit-url=wss://LIVEKIT_HOST --livekit-token-file=join.token \
  --add-stream=sensor=0,livekit=sensing-left \
  --add-stream=sensor=1,livekit=sensing-right
```

The `livekit=` names must equal the file's `track` values. Start the publisher by hand; closing the browser
disconnects only that viewer, and the Orin keeps capturing until you stop it.

## 5. Expected result and troubleshooting

Opening egocap shows the named tiles next to the tracking view. A tile reads "Waiting for camera" until its track is
published, "No video frames" if a published track stops producing frames, and "Disconnected" while the room
connection is down; the page reconnects on its own.

| Symptom | Check |
| --- | --- |
| No tiles at all | CaptureServer was started with `--livekit-streams`, and the backend's capture session has started. |
| "LiveKit preview unavailable: ... must be set" | `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` in the backend's environment. |
| Tiles never leave "Connecting" | The `url` is reachable from the browser; `wss://` certificate trusted; signalling and media ports open. |
| Connected, tile stays "Waiting for camera" | The publisher's token room and identity match `room` and `publisher`; `livekit=` names match `track`. |
| "Waiting for video" or black picture | The browser may not support the encoder's H.264 profile; check the browser console. |

## 5. No Orin: the emulator

`tools/livekit_orin_emulator.py` publishes test-pattern tracks with the same per-frame metadata (`user_timestamp`,
`frame_id`, 8-byte little-endian capture counter) in place of the Orin. It uses the SDK's software encoder, so it
exercises the PC side and the metadata contract but not the Jetson hardware encoder, its H.264 profile or the real
sender's pacing.

```bash
uv run python tools/livekit_orin_emulator.py \
  --livekit-url=wss://LIVEKIT_HOST --livekit-token-file=join.token \
  --add-stream=sensor=0,livekit=sensing-left --add-stream=sensor=1,livekit=sensing-right \
  --width=1280 --height=720 --fps=30 --timestamp-basis=capture_unix_us
```

`--timestamp-basis=processing_unix_us` declares what the current native sender sends: processing time, which is not
a capture time and is not recorded. `capture_unix_us` (the default) is honest here because the emulator is the
camera: it stamps the frame when it generates it. `monotonic_us` sends the publisher's monotonic clock, also not
recordable.

## 6. The whole stack on one machine, with the emulator

This runs everything except the Orin on one machine, against a local LiveKit server in development mode (key
`devkey`, secret `secret`). Each command runs in its own terminal, from `examples/egocap/`.

```bash
livekit-server --dev --bind 127.0.0.1                    # signalling on ws://127.0.0.1:7880

# pub.token: a token for room sensing-preview, identity orin-sensing, allowed to publish (lk token create, or any
# LiveKit token tool, signed with devkey/secret)
uv run python tools/livekit_orin_emulator.py --livekit-url=ws://127.0.0.1:7880 --livekit-token-file=pub.token \
  --add-stream=sensor=0,livekit=sensing-left --add-stream=sensor=1,livekit=sensing-right \
  --width=640 --height=480 --fps=15 --timestamp-basis=capture_unix_us

uv run capture-server --mock --livekit-streams streams.yaml
LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret uv run uvicorn egocap_web.app:app --app-dir backend --port 8000
```

`streams.yaml` names the room, the publisher and both tracks, with `width: 640`, `height: 480`, `fps: 15` and
`timestamp_basis: capture_unix_us` (section 2). Open `http://127.0.0.1:8000/`: both tiles show live video, and
Record saves a take with the two cameras next to the mock tracking. A take recorded this way came out KEPT, with 91
frames per camera over 6 s at the configured size and rate and increasing capture times.

This exercises the PC side only. It does not test the Jetson encoder, the sender's pacing, or clock agreement
between two machines.
