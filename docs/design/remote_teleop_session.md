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
| main process, session process | The Teleop Session Server process you start; a child process per robot app, with its own OpenXR session (connection to the runtime), trackers and plugins (programs that read devices such as gloves). |
| link | The connection between a robot app and its session process. |
| sample, sample stream | One tracker reading, e.g. a head pose; samples as MCAP records sent over the link, not to a file. |
| topic | A named series of records, `<tracker name>/<channel>`; a hand tracker has channels `left_hand` and `right_hand`. |
| setup | The records that declare a topic and its schema (the layout of its records). |
| link message; HELLO, HELLO_ACK, STATUS, READY, ERROR, SETUP, SAMPLES | A unit on the link: a 24-byte link message header (type, payload size, send time), then a payload. HELLO: the robot app's request; HELLO_ACK: its acceptance; STATUS: progress and health; READY: ready to stream; ERROR: a refusal or failure, which closes the link; SETUP: setup records; SAMPLES: one tick's samples; PING, PONG, HEARTBEAT: liveness and clock (section 6). |
| tick; tick time, rate, age; stale | One pass of the session process's loop: a `DeviceIOSession.update()` reading every tracker once, then one SAMPLES. Tick time: the server machine's monotonic clock at its start; rate: ticks per second; age: now minus the newest SAMPLES' tick time, on the robot machine's clock; stale: age above the limit. |

## 3. How a robot app uses it

On the server machine, run `python -m isaaccapture.teleop_session_server`; in the robot app:

```python
from isaaccapture.teleop_session_manager import PluginConfig, SessionMode, TeleopSession, TeleopSessionConfig

config = TeleopSessionConfig(
    app_name="my_robot",
    pipeline=pipeline,                        # unchanged
    plugins=[PluginConfig(...)],              # started by the Teleop Session Server
    mode=SessionMode.REMOTE,
    remote_endpoint="server-machine:47600",
)
with TeleopSession(config) as session:
    while running:
        send_to_robot(session.step())         # step() never waits for the network
```

The endpoint (`remote_endpoint`) is `host[:port]`, `tcp://host[:port]` or `ws://host[:port][/path]` (default ports: TCP
47600, WebSocket 47601). Entering the `TeleopSession` connects, sends HELLO and returns at READY: the session process
has started each requested plugin installed there, and the headset has stayed connected for about a second. It must get
there within `remote_connect_timeout_s` (60 s; `None`: no limit).

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
 robot app                              Teleop Session Server
   TeleopSession (remote mode)            main process ── starts, one at a time ─┐
     RemoteSession                                                               ▼
     RemoteSessionClient ◄════ link ════► session process: RemoteSessionHost,
                                            OpenXR session, trackers, plugins
                                              │ reads
                                              ▼
                                            CloudXR runtime ◄══ headset, plugins
```

- `src/core/remote_link/`: wire format, TCP and WebSocket transports, `Link` (one end of a link), and the clock offset
  (the estimated difference between the machines' monotonic clocks).
- `src/core/mcap/cpp/inc/mcap/sample_stream.hpp`, `tracker_channels.hpp`: `TrackerSampleStreamWriter`,
  `TrackerSampleStreamReader`, and `TrackerSampleSource`, which feeds a replay impl from a file or a sample stream.
- `src/core/deviceio_session/` (`cpp/inc/deviceio_session/remote_session*.hpp`, `python/session_bindings.cpp`):
  `RemoteSession` (like `DeviceIOSession`, the C++ session that updates every tracker once per `update()`, but over
  replay impls), `RemoteSessionClient`, `RemoteSessionHost` (the robot app's and the session process's link ends).
- `src/python/isaaccapture/teleop_session_manager/` (`teleop_session.py`, `_remote.py`): remote mode.
- `src/python/isaaccapture/teleop_session_server/`: `__main__.py` (main process), `_session.py` (session process).

Python orchestrates and C++ carries the data; C++ threads never call into Python.

## 5. Data path

```
 session process (server machine)                     robot app (robot machine)
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

**Session process.** It builds the trackers from HELLO, and `RemoteSessionHost` runs a `DeviceIOSession` over them,
recording the requested topics into a `TrackerSampleStreamWriter`. It ticks at the robot app's tick rate; each tick
sends setup for new topics before its SAMPLES, even an empty one, so the data does not turn stale. A plugin tracker,
and a full-body tracker from the Noitom or Xsens vendor, records only new samples, so when the plugin's data leaves the
OpenXR runtime, which a local reader sees as no data, the host adds a record without data to the tick.

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
| HELLO | JSON: protocol number, Isaac Capture version, app name, tick rate, trackers, topics, plugins |
| HELLO_ACK | JSON: `session_id`, `server_version` |
| STATUS, READY | JSON: stage, message, plugin process states, headset state |
| ERROR | Text, shown to the user as is |
| SETUP, SAMPLES | Setup; one tick's samples |
| PING, PONG, HEARTBEAT | PONG: when the robot app's PING was sent and received; others empty |

**Delivery.** Every link message except SAMPLES is delivered in order. The sender keeps at most one unsent SAMPLES,
merging newer ones into it (newest record per topic), so a slow network drops older samples instead of queueing them.

**Liveness and clock.** Each side sends a HEARTBEAT after 500 ms without sending and declares the link lost after 3 s
without receiving (the peer timeout). The PING/PONG exchange with the shortest recent round trip gives the clock offset.

## 7. Teleop Session Server processes

The **main process** attaches to or starts the CloudXR runtime and listens on TCP and, with `--ws-listen`, WebSocket.
For each accepted connection, when not busy, it starts a **session process** in a new process group, with the socket
and a serving pipe that tells whether it still serves its robot app.

The session process answers HELLO with HELLO_ACK, or ERROR with the reason; opens an OpenXR session, retrying until a
headset connects; starts the `DeviceIOSession` and the requested plugins; ticks; and sends READY once the headset has
stayed connected for a second. When the link ends, it stops its plugins and exits.

**Busy and handover.** While a session process serves a robot app, the main process refuses others with ERROR
`busy: ...`; a connection that dropped without closing counts as connected until the 3 s peer timeout. A robot app
that arrives while the previous session process still exists (a handover) waits for it to exit. A process whose robot
app has left gets up to 10 s, or the teardown time it reported when that is longer, and is then killed. If it still
serves its robot app after 1.5 s, or is still starting after 10 s, the newcomer gets the busy ERROR.

## 8. Status reporting

`get_status()` reports the server machine's plugins and OpenXR session from the newest STATUS: `plugin/<plugin_root_id>`
(its `PluginConfig` id) with process state and pid, and `openxr/runtime` and `openxr/headset`. `openxr/controller/left`
and `/right` come from the controller samples the robot app receives, as in live mode.

## 9. Errors and timeouts as the robot app sees them

| Situation | Result |
|---|---|
| A config, input or tracker that remote mode refuses; a `wss://` endpoint | `ValueError`, before any network traffic |
| Nothing accepts the connection, or the Teleop Session Server closes or resets it before sending anything (e.g. killed while the connection waited) | Retried about every 0.5 s |
| No READY within `remote_connect_timeout_s` | `TimeoutError` naming the last STATUS stage, or `server did not answer` |
| ERROR: busy, shutting down (stopped or killed), refused HELLO, plugin, OpenXR or other failure | `RuntimeError` with its text, not retried; to wait out busy or a restart, retry entering |
| Failed name lookup, `tcp://` to a WebSocket port or vice versa, or a close after partial data | `RuntimeError` or `TimeoutError`, at once |
| Link lost after HELLO_ACK; incompatible topic schema | `RuntimeError` from entering or `step()`: `Remote session lost: ...` or `MCAP schema mismatch on '<topic>': ...` |

## 10. Security model

The link has no authentication and no encryption: any machine that reaches its port can start the server machine's
plugins with any arguments and receive tracker data. Run the Teleop Session Server on a trusted network, open its ports
only toward the robot machine, and, since `wss://` is not supported, encrypt the link with a tunnel (VPN, SSH or
stunnel).

## 11. Testing

No automated test needs a headset or an OpenXR runtime. C++ `RemoteSession` tests write head and hand records into a
sample stream; tests over a link stream `KeyboardTracker` data, which needs no OpenXR (`TeleopSession` itself refuses
it), and pedal and Noitom full-body data from a fake tensor runtime (`tests/cpp/core/support/tensor_test_support`).
C++ tests use an in-memory transport, `socketpair()` or loopback. Teleop Session Server tests use its test-only
`--placeholder-openxr` flag (no OpenXR session; READY at once) or scripted OpenXR and plugins.

## 12. Decisions and reasons

**A sample stream, not MCAP files.** The session process records with live mode's MCAP code, into memory, and the robot
app reads with the replay impls that replay files: no tracker needed new decoding or recording logic, and the schema
check that guards old recordings also guards a robot app and Teleop Session Server on different versions.

**`step()` never waits.** Waiting would tie the robot app's loop to network jitter; staleness already tells the pipeline
that data is missing.

**The robot app's peer timeout starts with the first message,** since a connection can wait over 3 s in the main
process's listen backlog or a handover.

**Connecting is retried only when nothing accepts the connection or it closes before any data.** That close is a typed
`PeerClosedError` (`RemotePeerClosedError` in Python), since matching text could mistake an ERROR for a close. Anything
else is a refusal or would fail again.

**The stop path.** On SIGINT or SIGTERM, the main process refuses robot apps in its listen backlog with the
`shutting down` ERROR, closes its listeners and sends SIGTERM to the session process, which sends the same ERROR, stops
its plugins and closes the link gracefully, so the ERROR arrives instead of a reset. The main process waits for it
for 10 s, or for the teardown time it reported when that is longer, so that each plugin gets its `plugin.yaml`
`shutdown_timeout_seconds`; a second signal ends the wait. Then the main process kills the session process's process
group, in case a plugin outlived it.

**Session processes start from the main thread.** Each asks Linux for SIGTERM when the main process dies
(`PR_SET_PDEATHSIG`), since a killed main process sends none. Linux ties that signal to the starting thread, not the
process, so only the main thread starts session processes; the main process's own code starts no threads.

**A serving pipe.** Once it serves its robot app, a session process writes its teardown time into it: its plugins'
shutdown timeouts added up, since they stop one after another, plus a margin. It closes the pipe when its robot app
leaves, before the slow teardown, so a robot app that reconnects at once is served.

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
