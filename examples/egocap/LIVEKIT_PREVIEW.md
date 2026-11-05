<!--
SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: Apache-2.0
-->

# Live camera video over LiveKit

The camera host (an Orin running the SENSING publisher) publishes hardware-encoded H.264 into a LiveKit room.
CaptureServer says where that video is: it offers the room's streams in the `livekit_streams` list of the existing
session message, taken from a small file you give it. The egocap backend passes the list on and issues viewer
tokens; the browser then connects to LiveKit itself to receive the video. No video passes through CaptureServer or
the backend, and CaptureServer never joins the room.

```text
Orin: SIPL -> hardware H.264 -> LiveKit publisher --> LiveKit server --> browser <video>
PC:   CaptureServer (--livekit-streams: where the streams are) -> egocap backend (list + viewer token)
```

**The tiles are display-only for now and say so.** Saving these streams in a recording is planned but not built, so
pressing Record does not save them.

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
    - {id: head_left, name: Head left, track: sensing-left, width: 2560, height: 1984, fps: 30}
    - {id: head_right, name: Head right, track: sensing-right}
```

`width`, `height` and `fps` are optional and only describe what to expect; leave them out when unknown. Start
CaptureServer with `--livekit-streams FILE` in addition to its usual options. The list reaches the backend in the
session start message, so it appears once the backend's capture session has started. Offering a stream does not mean
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
