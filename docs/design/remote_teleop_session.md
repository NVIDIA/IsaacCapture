<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Remote Teleop Session: design

## 1. Goal and non-goals

`TeleopSession` is the Isaac Capture Python class that a robot app uses to read devices: each `step()` reads them once
and runs the app's retargeting pipeline, which turns device data into robot commands. In live mode (`SessionMode.LIVE`)
it reads them through a local OpenXR runtime (a program implementing OpenXR, the standard VR/AR device API).

Remote mode (`SessionMode.REMOTE`) lets the robot app run on a different machine from the one the headset connects to
through NVIDIA CloudXR (an OpenXR runtime streaming to headsets). A Teleop Session Server there reads the devices and
streams their data to the robot app, which keeps its loop; only its configuration changes. User documentation:
`docs/source/teleop/teleop_session.rst`.

Non-goals: video or audio from the robot; changing how the robot is commanded; serving several robot apps at once.

## 2. Terms

| Term | Meaning |
|---|---|
| robot app, robot machine | The program that runs `TeleopSession` and commands the robot; its machine. |
| server machine | The Linux machine the headset connects to; it runs the CloudXR runtime and the Teleop Session Server. |
| tracker, replay impl | An object reading one kind of device data, e.g. head poses; its replay implementation, which decodes records of MCAP (Isaac Capture's recording format) instead of reading OpenXR. |
| Teleop Session Server | One process on the server machine that serves one robot app at a time. It attaches to or starts the CloudXR runtime, except with `--no-launch-cloudxr-runtime`. Once a headset has connected, it opens one OpenXR session (connection to the runtime) and keeps it until it exits; the robot app's trackers read through it. The test-only `--placeholder-openxr` opens none. |
| supervisor | A program that restarts the Teleop Session Server when it exits, such as a systemd service or a shell script. |
| plugin | A program that reads a device, such as gloves, and pushes its data into the OpenXR runtime. Operators start plugins on the server machine, by hand or with a script; the robot app cannot request them. |
| link | The connection between a robot app and the Teleop Session Server. |
| sample, sample stream | One tracker reading, e.g. a head pose; samples as MCAP records sent over the link, not to a file. |
| topic | A named series of records, `<tracker name>/<channel>`; a hand tracker has channels `left_hand` and `right_hand`. |
| setup | The records that declare a topic and its schema (the layout of its records). |
| link message; HELLO, HELLO_ACK, STATUS, READY, ERROR, SETUP, SAMPLES | A unit on the link: a 24-byte link message header (type, payload size, send time), then a payload. HELLO: the robot app's request; HELLO_ACK: its acceptance; STATUS: progress and health; READY: ready to stream; ERROR: a refusal or failure, which closes the link; SETUP: setup records; SAMPLES: one tick's samples; PING, PONG, HEARTBEAT: liveness and clock (section 6). |
| tick; tick time, rate, age; stale | One `RemoteSessionHost.tick()` for the robot app the server serves: one `DeviceIOSession.update()`, reading each of its trackers once, then one SAMPLES link message. Tick time: the server machine's monotonic clock at its start; rate: ticks per second; age: now minus the newest SAMPLES' tick time, on the robot machine's clock; stale: age above the limit. |

## 3. How a robot app uses it

On the server machine, run `python -m isaaccapture.teleop_session_server`; in the robot app:

```python
from isaaccapture.teleop_session_manager import SessionMode, TeleopSession, TeleopSessionConfig

config = TeleopSessionConfig(
    app_name="my_robot",
    pipeline=pipeline,                        # unchanged
    mode=SessionMode.REMOTE,
    remote_endpoint="server-machine:47600",
)
with TeleopSession(config) as session:
    while running:
        send_to_robot(session.step())         # step() never waits for the network
```

The endpoint (`remote_endpoint`) is `host[:port]`, `tcp://host[:port]` or `ws://host[:port][/path]` (default ports: TCP
47600, WebSocket 47601). Entering the `TeleopSession` connects, sends HELLO and returns at READY: the headset has
stayed connected for about a second. It must get there within `remote_connect_timeout_s` (60 s; `None`: no limit).
`plugins` must be empty in remote mode (`ValueError`): plugins run on the server machine, started by hand or with a
script.

`step()` uses the newest samples (the server ticks at `remote_tick_rate_hz`, 90 Hz) and reports their tick age in
`last_step_info.remote_tick_age_s`. Above `remote_max_tick_age_s` (0.25 s) they are stale: trackers report no data
until fresh ones arrive.

Both machines run Linux; the robot machine needs no CloudXR or OpenXR runtime. Remote mode carries head, hand,
controller and full-body trackers, and plugin trackers that read a `collection_id` (the name under which a plugin pushes
data into the OpenXR runtime), except `PluginDeviceStatusTracker` and trackers with metadata (e.g. `TactileTracker`),
whose metadata the replay impl reads before any sample arrives. It refuses keyboard, message-channel (two-way headset
messages) and tactile (`TactileSource`) inputs and the options that need local OpenXR or recording. Sinks (pipeline
outputs toward a device, such as haptics) run, but their data is dropped.

## 4. Components

```
 robot machine                          server machine (Linux)
 robot app                              Teleop Session Server (one process)
   TeleopSession (remote mode)            listeners: one robot app at a time
     RemoteSession                          │
     RemoteSessionClient ◄════ link ════► RemoteSessionHost: the robot app's trackers
                                            │ read through the server's OpenXR session
                                            ▼
                                          CloudXR runtime ◄══ headset; plugins (started by the operator)
```

- `src/core/remote_link/`: wire format, TCP and WebSocket transports, `Link` (one end of a link), and the clock offset
  (the estimated difference between the machines' monotonic clocks).
- `src/core/mcap/cpp/inc/mcap/sample_stream.hpp`, `tracker_channels.hpp`: `TrackerSampleStreamWriter`,
  `TrackerSampleStreamReader`, and `TrackerSampleSource`, which feeds a replay impl from a file or a sample stream.
- `src/core/deviceio_session/` (`cpp/inc/deviceio_session/remote_session*.hpp`, `python/session_bindings.cpp`):
  `RemoteSession` (like `DeviceIOSession`, the C++ session that updates every tracker once per `update()`, but over
  replay impls), `RemoteSessionClient`, `RemoteSessionHost` (the robot app's and the server's link ends).
- `src/python/isaaccapture/teleop_session_manager/` (`teleop_session.py`, `_remote.py`): remote mode.
- `src/python/isaaccapture/teleop_session_server/`: `__main__.py` (the server's loop: CloudXR, listeners, the OpenXR
  session, one robot app at a time), `_session.py` (HELLO and one robot app's connection).

Python orchestrates and C++ carries the data; C++ threads never call into Python.

## 5. Data path

```
 Teleop Session Server (server machine)               robot app (robot machine)
 trackers ─► DeviceIOSession.update()                 TeleopSession.step() ─► RemoteSession.update()
                  │ records into                                                    │ updates
                  ▼                                                                 ▼
          TrackerSampleStreamWriter                TrackerSampleStreamReader ─► replay impls ─► pipeline
                  │                                            ▲
                  ▼                                            │
          RemoteSessionHost.tick() ══ SETUP, SAMPLES ══► RemoteSessionClient
```

**Robot app, on entry.** For each tracker of the pipeline's sources (its inputs) and of `TeleopSessionConfig.trackers`,
`RemoteSession.create` builds a replay impl reading one shared `TrackerSampleStreamReader`. HELLO carries the topics the
impls read and a descriptor per tracker: type, constructor parameters (e.g. a `collection_id`) and vendor (the backend
that supplies the data).

**Teleop Session Server.** It builds the trackers from HELLO, and `RemoteSessionHost` runs a `DeviceIOSession` over
them on the server's OpenXR session, recording the requested topics into a `TrackerSampleStreamWriter`. It ticks at
the robot app's tick rate; each tick sends setup for new topics before its SAMPLES, even an empty one, so the data does
not turn stale. A plugin tracker, and a full-body tracker from the Noitom or Xsens vendor, records only new samples, so
when the plugin's data leaves the OpenXR runtime, which a local reader sees as no data, the host adds a record without
data to the tick.

**Robot app, per `step()`.** The `TrackerSampleStreamReader` keeps each topic's newest record. `RemoteSession.update()`
takes a snapshot of those records, empty when they are stale; each replay impl reads its topics from it through a
`TrackerSampleSource`. A topic's first read checks its schema: an incompatible one raises; one that only gained fields
is accepted.

**Time.** Only the tick time is converted to the robot machine's clock, through the clock offset, and only for the
tick age. Each record holds a sample's payload and timestamps; replay impls hand out only the payload, so
the robot app never sees samples' timestamps, and a device time inside a payload stays on the server machine's clock.

## 6. Wire protocol

`messages.hpp` and `message_header.hpp` in `src/core/remote_link/cpp/inc/remote_link/` define the format.

| Link message | Payload |
|---|---|
| HELLO | JSON: protocol number, Isaac Capture version, app name (only logged), tick rate, trackers, topics; a non-empty `plugins` list is refused, and so is a HELLO over 1 MiB, with over 256 trackers or 1024 topics, or with a text over 1024 characters |
| HELLO_ACK | JSON: `session_id`, `server_version` |
| STATUS, READY | JSON: stage, message, headset state |
| ERROR | Text, shown to the user as is |
| SETUP, SAMPLES | Setup; one tick's samples |
| PING, PONG, HEARTBEAT | PONG: when the robot app's PING was sent and received; others empty |

**Delivery.** Every link message except SAMPLES is delivered in order. The sender keeps at most one unsent SAMPLES,
merging newer ones into it (newest record per topic), so a slow network drops older samples instead of queueing them.
The other link messages queue; once 64 MiB of them are unsent, the peer has stopped reading and the link is lost.

**Liveness and clock.** Each side sends a HEARTBEAT after 500 ms without sending and declares the link lost after 3 s
without receiving (the peer timeout). The PING/PONG exchange with the shortest recent round trip gives the clock offset.

## 7. The Teleop Session Server

The server is one process, `python -m isaaccapture.teleop_session_server`, serving one robot app at a time.

**For as long as the server runs** it holds:

- the CloudXR runtime, which it attaches to or starts, and whose health it checks every second; with
  `--no-launch-cloudxr-runtime` it does neither and uses the OpenXR runtime already configured;
- its listeners: TCP (`--listen`) and, with `--ws-listen`, WebSocket;
- one OpenXR session, opened as soon as the runtime reports a headset and retried every second until then (while
  `xrGetSystem` fails with `XR_ERROR_FORM_FACTOR_UNAVAILABLE`), whether or not a robot app is connected. An OpenXR
  session's extensions (optional OpenXR features) are fixed when it is created, so the server requests those of every
  tracker kind it serves: head, hands, controllers, full body for each body vendor, and every plugin tracker it
  accepts, plus `XR_BD_body_tracking`, which CloudXR provides. A HELLO whose trackers need any other is refused;
- the capture of native output (`capture_native_output()`): what native code, such as the OpenXR runtime, writes
  straight to the process's stdout and stderr goes to a capture file, `<timestamp>.isaaccapture.<pid>.native.log` in
  the log directory (`ISAACCAPTURE_LOG_DIR`, by default `/tmp/isaaccapture-<uid>/logs`). Python's own output, such as
  the server's log records, stays on the terminal. `ISAACCAPTURE_NATIVE_CAPTURE=off` or `ISAACCAPTURE_LOGGING=off`
  turns the capture off.

**Per robot app**, from when the server serves it until it leaves, the server has:

- a `RemoteSessionHost` on the connection, created when the server serves the robot app; a held robot app (see "One
  robot app at a time") has none, and its connection is not read. The server waits up to 5 s for HELLO and answers
  with HELLO_ACK, or ERROR with the reason;
- the robot app's trackers, started on the OpenXR session once it exists and updated at the robot app's tick rate,
  clamped to 1-240 Hz;
- STATUS on every change and, once the trackers run, about once a second: stage `connecting_openxr` while there is
  no OpenXR session yet, `waiting_for_headset` until the headset has stayed connected for 1.1 s, then READY (stage
  `ready`) and `running`.

When the robot app leaves, `RemoteSessionHost.close()` drops its trackers; the OpenXR session stays for the next one.

**Plugins.** The server never starts, stops, finds or monitors plugins, and a robot app cannot request them: a HELLO
that lists plugins is refused. Operators start them on the server machine, by hand or with a script, e.g. the Manus
glove plugin (`src/plugins/manus`). A robot app's plugin trackers read whatever the running plugins push.

**One robot app at a time.** A robot app that connects while another is served is held, its connection not yet read,
for up to 4 s: longer than the 3 s peer timeout, so a robot app that reconnects after its own link dropped is served.
If the served one leaves in that time, the held one is served; otherwise it gets ERROR `busy: ...`. One that connects
while another is held gets the busy ERROR at once. The served robot app keeps ticking meanwhile: refusals run beside
the loop, and while 8 are in progress a further connection is closed unanswered, which its robot app retries.

**Failures.** When the CloudXR health check fails, or the OpenXR session cannot be opened for a reason other than a
missing headset, or reports a failure, the server sends ERROR to the robot app it serves (`shutting down` for CloudXR,
the OpenXR failure otherwise), closes its link and exits with code 1, for a supervisor to restart it. When starting or
updating a robot app's trackers fails, or serving it fails in any other way, that robot app gets ERROR and its
connection ends; the server and its OpenXR session keep running.

## 8. Status reporting

`get_status()` reports the server machine's OpenXR session from the newest STATUS: `openxr/runtime` and
`openxr/headset`. `openxr/controller/left` and `/right` come from the controller samples the robot app receives, as in
live mode. Plugins do not appear in it, and a plugin that stops raises no `PluginCrashException`: the server does not
run plugins (section 7).

## 9. Errors and timeouts as the robot app sees them

| Situation | Result |
|---|---|
| A config, input or tracker that remote mode refuses; a `wss://` endpoint | `ValueError`, before any network traffic |
| Nothing accepts the connection, or the Teleop Session Server closes or resets it before sending anything (e.g. killed while the connection waited, or 8 refusals in progress) | Retried about every 0.5 s |
| No READY within `remote_connect_timeout_s` | `TimeoutError` naming the last STATUS stage, or `server did not answer` |
| ERROR: busy, shutting down, refused HELLO, OpenXR, tracker or other failure (a killed server sends none) | `RuntimeError` with its text, not retried; to wait out busy or a restart, retry entering |
| Failed name lookup, `tcp://` to a WebSocket port or vice versa, or a close after partial data | `RuntimeError` or `TimeoutError`, at once |
| Link lost after HELLO_ACK; incompatible topic schema | `RuntimeError` from entering or `step()`: `Remote session lost: ...` or `MCAP schema mismatch on '<topic>': ...` |

## 10. Security model

The link has no authentication and no encryption: any machine that reaches its port can receive tracker data. It
cannot start programs there: plugins are started on the server machine itself. Run the Teleop Session Server on a
trusted network, open its ports only toward the robot machine, and, since `wss://` is not supported, encrypt the link
with a tunnel (VPN, SSH or stunnel).

## 11. Testing

No automated test needs a headset or an OpenXR runtime. C++ `RemoteSession` tests write head and hand records into a
sample stream; tests over a link stream `KeyboardTracker` data, which needs no OpenXR (`TeleopSession` itself refuses
it), and pedal and Noitom full-body data from a fake tensor runtime (`tests/cpp/core/support/tensor_test_support`).
C++ tests use an in-memory transport, `socketpair()` or loopback. Teleop Session Server tests use its test-only
`--placeholder-openxr` flag (no OpenXR session; READY at once), or run its loop in process with a scripted OpenXR
session, CloudXR runtime and robot apps on a virtual clock.

## 12. Decisions and reasons

**A sample stream, not MCAP files.** The server records with live mode's MCAP code, into memory, and the robot
app reads with the replay impls that replay files: no tracker needed new decoding or recording logic, and the schema
check that guards old recordings also guards a robot app and Teleop Session Server on different versions.

**`step()` never waits.** Waiting would tie the robot app's loop to network jitter; staleness already tells the pipeline
that data is missing.

**The robot app's peer timeout starts with the first message,** since a connection can wait up to 4 s while another
robot app is served.

**Connecting is retried only when nothing accepts the connection or it closes before any data.** That close is a typed
`PeerClosedError` (`RemotePeerClosedError` in Python), since matching text could mistake an ERROR for a close. Anything
else is a refusal or would fail again.

**One process, one OpenXR session.** The OpenXR session lives as long as the server, so a robot app that connects
does not wait for one to open. Alternative: a child process per robot app, with its own OpenXR session and trackers.
Not chosen: with one robot app at a time its isolation gains little, the two processes would need their own
coordination (liveness, teardown, a child that outlives its parent), and every reconnect would reopen the OpenXR
session. The cost: a crash ends the server, for its supervisor to restart.

**Plugins outside the server.** Operators start plugins; the server never starts, stops, finds or monitors them.
Alternative: the robot app lists plugins in HELLO, and the server starts them for that connection. Not chosen: it
couples the robot app to plugin names and paths that exist only on the server machine; anyone who reaches the
unauthenticated port could start programs there with any arguments; and every reconnect would restart the devices. A
robot app's plugin trackers report no data while a plugin is not running, as in live mode.

**The stop path.** On SIGINT or SIGTERM, the server refuses the held robot app and those in its listen backlog with
the `shutting down` ERROR, while fewer than 8 refusals are in progress (it closes the others unanswered, and their
robot apps retry), and closes its listeners, so a robot app that connects later is refused by the kernel and
retries until a restarted server serves it. It sends the same ERROR to the robot app it serves and closes that link
gracefully, so the ERROR arrives instead of a reset, and closes that robot app's trackers before the OpenXR session
they read through, even when refusing the others fails. Then it exits with code 0. A killed server sends nothing; the
robot app sees the link close.

**Sinks are not flushed.** Flushing a sink writes through a tracker to a device; a `RemoteSession` has no such tracker,
so flushing would fail and could make `step()` raise.

**Version check.** A different protocol number is refused; a different Isaac Capture version only logs a warning, since
the per-topic schema check already rejects unreadable data.

## 13. Not done yet

- Authentication and TLS on the link; `wss://` on the robot side.
- Device status that plugins report (`PluginDeviceStatusTracker`, e.g. Manus gloves) in `get_status()`.
- Sink data, e.g. haptics, toward the server machine's devices.
- A replay or synthetic Teleop Session Server mode, for headset-free tests of head and hand data over a link.
- Typed exceptions for refusals.
- Remote mode in `examples/`.
- Reconnecting after a lost link; message-channel inputs.
- Trackers with metadata (`TactileTracker`): reading the metadata once it arrives on the stream.
