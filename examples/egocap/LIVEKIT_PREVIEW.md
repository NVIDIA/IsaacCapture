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

Recording LiveKit video and running the emulator need the optional `livekit` package: `uv sync --extra livekit`
(included in `--extra dev`). Without it the backend still starts and the browser preview works, but the Record
button is blocked for LiveKit streams with a message naming the extra.

```text
Orin: SIPL -> hardware H.264 -> LiveKit publisher --> LiveKit server --> browser <video>
PC:   CaptureServer (--livekit-streams: where the streams are) -> egocap backend (list + viewer token)
```

**Recording.** A LiveKit stream is saved with a take only when its entry gives `width`, `height` and `fps` and
declares `timestamp_basis: capture_monotonic_us`: the publisher stamps each frame with the sensor's capture event on
the publisher's own monotonic clock. If any offered stream does not qualify, recording is refused with that stream's
name (remove it from the streams file, or make it recordable); preview is unaffected. The native Orin sender
declares processing time by default, so with it recording is refused unless it is started with
`--timestamp-basis=capture_monotonic_us` and the streams file declares that basis.

There is one way to place the publisher's clock on this PC's, the four-timestamp exchange CloudXR uses. The recorder
joins the room itself (it needs the same `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`), and once a second sends the
publisher a small request as a LiveKit data message on topic `isaaccapture.clock-sync`: little-endian `Q`, a
sequence number and `t1` (this PC's monotonic ns); the publisher answers `R`, the sequence, `t1`, `t2` (its monotonic
ns on receipt) and `t3` (its monotonic ns when replying); the recorder stamps `t4` on the reply. The round trip with
the publisher's processing removed is `(t4 - t1) - (t3 - t2)` and the offset is `((t2 - t1) + (t3 - t4)) / 2`. The
offset assumes both directions took equally long, so the mapping's uncertainty is half the round trip and nothing
smaller is claimed. The best recent exchange is used; a mapping is withheld when its last exchange is more than 5 s
old; and a new exchange that disagrees with the current mapping by more than the two uncertainties plus a 200 ppm
drift allowance is a discontinuity. A take cannot start until the publisher has answered at least once, and ends if
the mapping goes missing, goes stale or jumps; frames already recorded keep the mapping they were made under, and
the newest mapping epochs, their uncertainties and the budget are saved in the kept episode's `timing` attributes. A
take also ends when the mapping's uncertainty exceeds the budget, 5 ms by default and set in milliseconds with
`EGOCAP_CLOCK_UNCERTAINTY_BUDGET_MS`; the default is a policy choice, not a measurement. The recorder
also applies one coarse check: a frame more than 2 s old on arrival, or stamped in the future, ends the take. A take
that passes is not thereby proven aligned with tracking, which also assumes the backend and CaptureServer share one
machine's clock. Browser display latency is not shown: it needs the same exchange between the PC and the browser and
an exact match of the displayed frame to its capture stamp, neither of which exists yet.

**Recording from the real Orin.** The native sender answers the clock-sync requests with its own monotonic clock, and
with `--timestamp-basis=capture_monotonic_us` publishes the sensor's capture event on that clock. It converts the
backend's sensor timestamp through one anchor pair (the source clock and `CLOCK_MONOTONIC`, read back to back and
refreshed once a second, never per frame): SIPL's capture counter as ticks of the ARM generic counter at its
`CNTFRQ` rate, Argus's sensor timestamp as `CLOCK_MONOTONIC_RAW` nanoseconds. Both units are assumptions, so every
frame is checked: a capture time later than the frame's ready time, or more than 2 s before it, fails the stream
rather than being published. The plausibility check only catches gross errors, so this option is not a qualified capture basis for
synchronized recording until the target sensor's clock, event and encoder correlation are verified. None of this has run on an Orin; the first run shows whether the assumed units hold, and
real alignment still needs the evidence below. The sender's only capture-event clock is the SIPL frame capture counter (`frameCaptureTSC`), which NVIDIA's public SIPL documentation describes as
raw ARM generic-timer ticks, not nanoseconds (the sender carries it as `capture_tsc`, an unscaled value). The deployed
Orin's SIPL headers are not available here, so nothing about its units or event is verified. The evidence required,
from a supported platform API and not from fitting measurements:

1. The counter's definition, units and event (which instant of the frame it marks) from the SIPL image-metadata
   documentation of the deployed release, and the TSC's definition and rate in the L4T documentation of the
   deployed JetPack. Record the exact header, SDK and L4T versions.
2. A contemporaneous counter and monotonic-clock reading from a supported API, so that a counter value can be placed
   on `CLOCK_MONOTONIC` through one offset that is re-measured at a low rate, never per frame. Convert in the sender
   and publish that time with the `capture_monotonic_us` basis, keeping the raw counter in the frame's user data.
3. A physical event seen by both the camera and a tracking device as the test of real alignment, with the residual
   offset, drift and jitter measured. The frame-age check only catches gross disagreement.

`SENSING_TSC_PROBE=1` prints the capture counter beside the host clocks twice a second, off by default, as separate
uncalibrated samples. The counter's units are unconfirmed, so they must not be subtracted, fitted or used as a mapping.

**Exercised once on an Orin.** The author ran the publisher on an AGX Orin with two SHW5G cameras (2560x1984, 30 fps,
H.264) into a LiveKit server on the Orin, and a PC browser showed the video. Over the run each track published about
2,100 frames with none replaced and one capture missed on one sensor. This was a single short run on dev keys and plain
`ws://`, not a qualification. Still unmeasured or unverified: end-to-end latency, behaviour when the encoder drops a
frame, and real sensor-to-tracking alignment. Recording from the real Orin stays refused.

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
  --width=1280 --height=720 --fps=30 --timestamp-basis=capture_monotonic_us
```

`--timestamp-basis=capture_monotonic_us` (the default) is the recordable basis and is honest here because the emulator
is the camera: it stamps the frame on its own emulated monotonic clock when it generates it, and answers the PC's
clock-sync requests with that same clock. `--monotonic-offset-s`, `--clock-drift-ppm` and `--pipeline-delay-ms` give
that clock a different boot time, a slow drift and a capture-to-publish delay, so the mapping has something real to
find. `processing_unix_us` declares what the current native sender sends (processing time, not recorded);
`capture_unix_us` and `monotonic_us` exist for tests of refusal and are not recordable.

## 6. The whole stack on one machine, with the emulator

This runs everything except the Orin on one machine, against a local LiveKit server in development mode (key
`devkey`, secret `secret`). Each command runs in its own terminal, from `examples/egocap/`.

```bash
livekit-server --dev --bind 127.0.0.1                    # signalling on ws://127.0.0.1:7880

# pub.token: a token for room sensing-preview, identity orin-sensing, allowed to publish (lk token create, or any
# LiveKit token tool, signed with devkey/secret)
uv run python tools/livekit_orin_emulator.py --livekit-url=ws://127.0.0.1:7880 --livekit-token-file=pub.token \
  --add-stream=sensor=0,livekit=sensing-left --add-stream=sensor=1,livekit=sensing-right \
  --width=640 --height=480 --fps=15 --timestamp-basis=capture_monotonic_us

uv run capture-server --mock --livekit-streams streams.yaml
LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret uv run uvicorn egocap_web.app:app --app-dir backend --port 8000
```

`streams.yaml` names the room, the publisher and both tracks, with `width: 640`, `height: 480`, `fps: 15` and
`timestamp_basis: capture_monotonic_us` (section 2). Open `http://127.0.0.1:8000/`: both tiles show live video, and
Record saves a take with the two cameras next to the mock tracking. A take recorded this way came out KEPT, with 91
frames per camera over 6 s at the configured size and rate and increasing capture times.

This exercises the PC side only. It does not test the Jetson encoder, the sender's pacing, or clock agreement
between two machines.
