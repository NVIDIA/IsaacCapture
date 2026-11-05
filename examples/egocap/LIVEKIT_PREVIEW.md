<!--
SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: Apache-2.0
-->

# Live camera preview over LiveKit

The camera host (an Orin running the SENSING publisher) publishes hardware-encoded H.264 into a LiveKit room. The
browser subscribes to those tracks itself and decodes them, so the egocap backend only tells it where the room is and
hands it a subscribe-only token. CaptureServer keeps doing tracking and control.

```text
Orin: SIPL -> hardware H.264 -> LiveKit publisher --> LiveKit server --> browser <video>
PC:   egocap backend (station config + viewer token)     CaptureServer (tracking, control)
```

**These tiles are preview-only.** They are not requested from CaptureServer and are not recorded; pressing Record
saves only the cameras the station already lists under `video_sources`. Every LiveKit tile says so.

**Not yet exercised on the Orin.** The publisher, the browser's acceptance of its H.264 profile and the end-to-end
latency have not been run or measured. Report those as untested until someone has run them.

## 1. LiveKit server (on the PC or the LAN)

Run a LiveKit server reachable from both the Orin and the browser, and note its API key and secret. For a quick local
server, `livekit-server --dev --bind 0.0.0.0` uses the development key pair printed at start-up. Browsers need a
secure context for WebRTC on anything but `localhost`, so use a `wss://` URL with a certificate the browser trusts
when the browser is not on the server's own machine. Open the server's signalling and media ports between the machines.

## 2. Backend credentials and station file

The backend signs viewer tokens with the server's key pair, read from its environment only:

```bash
export LIVEKIT_API_KEY=...      # the LiveKit server's API key
export LIVEKIT_API_SECRET=...   # its secret; never put this in the station file
```

Add a `livekit_preview` section to the station YAML that `EGOCAP_STATION_CONFIG` points at:

```yaml
livekit_preview:
  url: wss://LIVEKIT_HOST          # what the browser connects to
  room: sensing-preview
  publisher: orin-sensing          # the publisher's participant identity
  tracks:                          # up to four tiles
    - name: Head left              # label shown on the tile
      track: sensing-left          # the published track name
    - name: Head right
      track: sensing-right
```

Then start the backend as usual (`uv run uvicorn egocap_web.app:app --app-dir backend --port 8000`). The endpoint is
`GET /api/livekit-preview`: 404 when the section is absent, 503 when the key pair is not set. The token lets a viewer
join the room and subscribe, nothing more; like the camera preview routes it is open to anyone who can reach the
backend, so keep the backend on a trusted network.

## 3. Orin publisher

Create a token file for the publisher: a LiveKit token for this room, with identity equal to the station's `publisher`,
allowed to publish (for example with LiveKit's `lk token create`; see its `--help`). Then add the LiveKit options to
the usual `camera_plugin_sensing` invocation, keeping the rig's own platform and configuration arguments:

```bash
camera_plugin_sensing ... \
  --livekit-url=wss://LIVEKIT_HOST --livekit-token-file=join.token \
  --add-stream=sensor=0,livekit=sensing-left \
  --add-stream=sensor=1,livekit=sensing-right
```

The `livekit=` names must equal the station's `track` values. Start the publisher by hand; closing the browser
disconnects only that viewer, and the Orin keeps capturing until you stop it.

## 4. Expected result and troubleshooting

Opening egocap shows the named tiles next to the tracking view, each marked preview-only. A tile reads "Waiting for
camera" until its track is published, "No video frames" if a published track stops producing frames, and
"Disconnected" while the room connection is down; the page reconnects on its own.

| Symptom | Check |
| --- | --- |
| "LiveKit preview unavailable: ... must be set" | `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` in the backend's environment. |
| Tiles never leave "Connecting" | The `url` is reachable from the browser; `wss://` certificate trusted; signalling and media ports open. |
| Connected, tile stays "Waiting for camera" | The publisher's token room and identity match `room` and `publisher`; `livekit=` names match `track`. |
| "Waiting for video" or black picture | The browser may not support the encoder's H.264 profile; check the browser console. |
