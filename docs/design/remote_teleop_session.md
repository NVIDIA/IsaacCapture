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

Robot state travels the other way on the same link. A robot app that names a robot description hands the readings
of its robot (joint positions, a gripper, status readings, faults, any other data series the description declares) to
its `TeleopSession` as numbers; the library encodes them and sends them up the link. On the server machine a
`RobotStateStore`, hosted by the Teleop Session Server, keeps the newest record of each series, and a reader such as an
agent's harness imports `isaaccapture.robot_state` and asks for them by name. Robot state is sent only when both ends
agree to it while the link is set up; a robot app without a description, and a robot app or a Teleop Session Server
built without robot state, exchange the same link messages as without it. User documentation:
`docs/source/teleop/robot_state.rst`.

Non-goals: video or audio from the robot; changing how the robot is commanded; serving several robot apps at once.
Robot state adds none of these to the library: an agent's commands and their answers, who controls the robot, stop, the
handover between an agent and a person, and camera pictures stay in the application.

## 2. Terms

| Term | Meaning |
|---|---|
| robot app, robot machine | The program that runs `TeleopSession` and commands the robot; its machine. |
| server machine | The Linux machine the headset connects to; it runs the CloudXR runtime and the Teleop Session Server. |
| tracker, replay impl | An object reading one kind of device data, e.g. head poses; its replay implementation, which decodes records of MCAP (Isaac Capture's recording format) instead of reading OpenXR. |
| Teleop Session Server | One process on the server machine that serves one robot app at a time. It attaches to or starts the CloudXR runtime, except with `--no-launch-cloudxr-runtime`; a runtime it starts keeps running after it exits. Once a headset has connected, it opens one OpenXR session (connection to the runtime) and keeps it until it exits; the robot app's trackers read through it. The test-only `--placeholder-openxr` opens none. |
| supervisor | A program that restarts the Teleop Session Server when it exits, such as a systemd service or a shell script. |
| plugin | A program that reads a device, such as gloves, and pushes its data into the OpenXR runtime. Operators start plugins on the server machine, by hand or with a script; the robot app cannot request them. |
| link | The connection between a robot app and the Teleop Session Server. |
| sample, sample stream | One tracker reading, e.g. a head pose; samples as MCAP records sent over the link, not to a file. |
| topic | A named series of records, `<tracker name>/<channel>`; a hand tracker has channels `left_hand` and `right_hand`. In remote mode a tracker's name is its position in HELLO, e.g. `0/left_hand`. |
| setup | The records that declare a topic and its schema (the layout of its records). |
| link message; HELLO, HELLO_ACK, STATUS, READY, ERROR, SETUP, SAMPLES | A unit on the link: a 24-byte link message header (magic, version, type, channel, sequence number, payload size, send time), then a payload. HELLO: the robot app's request; HELLO_ACK: its acceptance; STATUS: progress and health; READY: the headset has been tracked for about a second (SAMPLES flow from the first tick, before it); ERROR: a refusal or failure, which closes the link; SETUP: setup records; SAMPLES: one tick's samples; PING, PONG, HEARTBEAT: liveness and clock (section 6). |
| tick; tick time, rate, age; stale | One `RemoteSessionHost.tick()` for the robot app the server serves: one `DeviceIOSession.update()`, reading each of its trackers once, then one SAMPLES link message. Tick time: the server machine's monotonic clock at its start; rate: ticks per second; age: now minus the newest SAMPLES' tick time, on the robot machine's clock; stale: age above the limit. |
| robot state | Readings of the robot, sent from the robot app to the server machine: joint positions (with velocities and efforts where the robot has them), a gripper's opening, status readings, faults, and any other series the robot declares. |
| data series (series), record | A named sequence of small records over time, e.g. the series `arm`: one record of joint positions per reading, some tens a second. It is not video. In code a series is a sample stream and its name a topic; a robot-state topic is the series name itself, e.g. `arm`. |
| shape | The layout of a series' records. Standard shapes: `joints` (the existing `JointStateOutputRecord`; a gripper is one more named joint), `status` (`RobotStatusRecord`: named readings and faults) and `description`. Any other shape is one the robot defines in its description; its records are JSON objects. |
| robot description | What a robot is: its series, the joints of each joints series with name, kind and limits, and its grippers. An integrator writes it once as a TOML file; it travels as the one record of the reserved series `robot_description`. |
| reader | A program on the server machine that imports `isaaccapture.robot_state` to read robot state, e.g. an agent's harness. |
| robot session | The time one robot app that sends robot state is connected to the Teleop Session Server. A new one empties the store. |
| ROBOT_STATE_SETUP, ROBOT_STATE_SAMPLES, ROBOT_STATE_RECORDS | The link messages of robot state, all from the robot app to the server (section 6). ROBOT_STATE_SETUP declares the series; ROBOT_STATE_SAMPLES carries records of series of which only the newest record matters; ROBOT_STATE_RECORDS carries records that must all arrive, in order (the description). |
| reading time, reading age, fresh | Reading time: the robot machine's monotonic clock when the robot app took the reading. Reading age: now minus the reading time, both on the server machine's clock. A record is fresh while its age is at most its series' `stale_after_s` (0: it never goes stale). These belong to robot state; tick time and tick age belong to SAMPLES. |

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
been tracked (`openxr/headset` CONNECTED) for about a second; tracked means awake, not necessarily worn. It must get
there within `remote_connect_timeout_s` (60 s; `None`: no limit). `plugins` must be empty in remote mode
(`ValueError`): plugins run on the server machine, started by hand or with a script.

`step()` uses the newest samples (the server ticks at `remote_tick_rate_hz`, 90 Hz) and reports their tick age in
`last_step_info.remote_tick_age_s` (`None` when the step got no fresh samples). Above `remote_max_tick_age_s` (0.25 s)
they are stale: trackers report no data until fresh ones arrive.

Both machines run Linux; the robot machine needs no CloudXR or OpenXR runtime. Remote mode carries head, hand,
controller and full-body trackers, and plugin trackers that read a `collection_id` (the name under which a plugin pushes
data into the OpenXR runtime). It raises `ValueError` for trackers with metadata (e.g. `TactileTracker`), whose metadata
the replay impl reads before any sample arrives, and, on entry, for any other tracker type. It leaves out
`PluginDeviceStatusTracker` and a tracker only a sink uses, without an error. It refuses keyboard, message-channel
(two-way headset messages) and tactile (`TactileSource`) inputs and the options that need local OpenXR or recording.
Sinks (pipeline outputs toward a device, such as haptics) run, but their data is dropped.

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
- `src/core/deviceio_session/` (`cpp/inc/deviceio_session/remote_session*.hpp`, `python/session_bindings.cpp`,
  `py_utils/inc/deviceio_py_utils/session.hpp`): `RemoteSession` (like `DeviceIOSession`, the C++ session that updates
  every tracker once per `update()`, but over replay impls), `RemoteSessionClient`, `RemoteSessionHost` (the robot
  app's and the server's link ends). Python's `RemoteSession` holds the C++ `RemoteSession` and its
  `RemoteSessionClient`; the server uses the `RemoteSessionListener` binding.
- `src/python/isaaccapture/teleop_session_manager/` (`teleop_session.py`, `_remote.py`): remote mode.
- `src/python/isaaccapture/teleop_session_server/`: `__main__.py` (the server's loop: CloudXR, listeners, the OpenXR
  session, one robot app at a time), `_session.py` (HELLO and one robot app's connection), `_common.py` (default
  ports, the busy and shutting-down texts, logging setup).

Python orchestrates and C++ carries the data; C++ threads never call into Python.

Robot state adds one path in the other direction, from the robot app to a reader:

```
 robot machine                                 server machine (Linux)
 robot app                                     Teleop Session Server (one process)
   TeleopSession.publish_robot_state()           RemoteSessionHost
     robot_state.records    numbers to a record    │ payloads of the robot-state link messages
     RobotStatePublisher    record to MCAP bytes   ▼
     RemoteSessionClient ═════ link ═══════════► RobotStateStore ◄── RobotState.from_store(): a reader in the server
                                                   │ snapshots
                                                   ▼
                                                 RobotStateLocalService ══ private local socket ══► reader program
                                                                                                    RobotState.connect()
```

- `src/core/schema/fbs/robot_description.fbs`, `robot_status.fbs`: the description and status records. Joints use the
  existing `joint_state.fbs`.
- `src/core/robot_state_publisher/`: `RobotStatePublisher`, the robot end. It turns declared series into MCAP Schema and
  Channel records and each record into an MCAP Message record, and builds and merges the payloads of the robot-state
  link messages.
- `src/core/robot_state_store/`: `RobotStateStore`, the server end. Per series it keeps the declaration, the newest
  record, its times and the clock sample that came with it, and it exports and imports snapshots. Its Python binding is
  `isaaccapture.robot_state._robot_state`.
- `src/core/remote_link/`: the three link message types. `src/core/deviceio_session/`: `RemoteSessionClient` sends
  them (`set_robot_state`, `start_robot_state`, `publish_robot_record`); `RemoteSessionHost` hands their payloads to
  the store attached with `attach_robot_state`.
- `src/python/isaaccapture/robot_state/`: `description.py` (the description file), `records.py` (numbers to record
  bytes and back), `reader.py` (`RobotState`, what a reader calls), `_local.py` (`RobotStateLocalService` and the
  connection between a reader program and the server).
- `src/python/isaaccapture/teleop_session_manager/`: `TeleopSessionConfig.robot_description`,
  `TeleopSession.publish_robot_state`, `publish_robot_record` and `robot_state_active`.
  `src/python/isaaccapture/teleop_session_server/`: the server owns the store and the `RobotStateLocalService`.

The two C++ modules carry bytes: they include no schema header and call no MCAP library, and read and write the three
MCAP record kinds themselves, so neither links `deviceio_session`, which holds the one MCAP implementation file. Typed
code (joints, status, description) is Python on top of the schema bindings.

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
data to that tick. It watches every full-body tracker this way, whatever its vendor.

**Robot app, per `step()`.** The `TrackerSampleStreamReader` keeps each topic's newest record. `RemoteSession.update()`
first raises the server's ERROR or a lost link, then takes a snapshot of those records, empty when they are stale; each
replay impl reads its topics from it through a `TrackerSampleSource`. A topic's first read after its setup arrives
checks its schema (entering reads once, right after READY): an incompatible one raises; one that only gained fields is
accepted.

**Time.** Only the tick time is converted to the robot machine's clock, through the clock offset, and only for the
tick age. Each record holds a sample's payload and timestamps; replay impls hand out only the payload, so
the robot app never sees samples' timestamps, and a device time inside a payload stays on the server machine's clock.

### Robot state

```
 robot app (robot machine)                               Teleop Session Server (server machine)
 publish_robot_state("arm", numbers)
        │ checks them against the description
        ▼
 records.encode_reading ─► RobotStatePublisher
                                  │
                                  ▼
                           RemoteSessionClient ══ ROBOT_STATE_SETUP, ══► RemoteSessionHost ─► RobotStateStore
                                                  ROBOT_STATE_SAMPLES,                              │
                                                  ROBOT_STATE_RECORDS                               ▼
                                                                                   RobotState (reader): joints("arm")
```

**The description file.** `TeleopSessionConfig.robot_description` names a TOML file (or holds a loaded
`RobotDescription`). It lists the robot's series; for a joints series each joint with name, kind (`revolute`,
`prismatic`, `continuous`, `gripper`) and limits; the readings of a status series; the fields of any shape the robot
defines; and its grippers. `isaaccapture.robot_state.description` reads it with `tomllib` and refuses every key it does
not know, with the file's path and the reason (`RobotDescriptionError`). A six-joint arm and a two-arm robot differ only
in their files; neither needs library code.

| Shape | Record | Schema sent in ROBOT_STATE_SETUP | Delivery |
|---|---|---|---|
| `description` (series `robot_description`) | `core.RobotDescriptionRecord` | its FlatBuffers binary schema | every record, in order |
| `joints` | `core.JointStateOutputRecord`, one entry per joint in file order | its FlatBuffers binary schema | newest only |
| `status` | `core.RobotStatusRecord`: readings in file order, then the faults present at that moment | its FlatBuffers binary schema | newest only |
| one the robot defines, e.g. `force` | a JSON object with exactly the fields the file lists | a JSON Schema generated from those fields, named `<robot>.<shape>` | newest only |

The link and the store carry any schema name, schema encoding, schema bytes and message encoding untouched, so a
Teleop Session Server reads a shape it has never seen without being rebuilt: the reader decodes `json` into a dict and
hands any other encoding out as bytes with its schema.

**Robot app, on entry.** The description is loaded before any connection, so its errors come first. HELLO offers robot
state. When HELLO_ACK agrees, the robot app sends ROBOT_STATE_SETUP for every series and then the description, in
ROBOT_STATE_RECORDS, before it waits for READY. When HELLO_ACK does not agree (a Teleop Session Server built without
robot state ignores the offer), the robot app logs one warning and sends no robot-state link message. Entering returns
at READY, as without robot state.

**Robot app, per reading.** `publish_robot_state(series, values, ...)` can be called from any thread while the session
is entered and does not depend on `step()`. Python checks the values against the description (the count, the kinds, the
optional velocities and efforts), adds the joint or reading names and the reading time, and builds the record; C++ wraps
it in an MCAP Message record and sends it at once, with the robot link's clock offset in the payload. The call returns
False, sending nothing, when the server did not agree or the link is down; `step()` raises the lost link as without
robot state.

**Teleop Session Server.** `RemoteSessionHost` hands each robot-state payload to the attached `RobotStateStore` with
the time it arrived. The store applies records in arrival order and keeps, per series, the newest one. It never calls
Python, so it can be filled on the link's receive thread. A record on a channel that no ROBOT_STATE_SETUP declared, or
from a payload built before the one the series' record came in, is dropped and counted. A payload the store cannot
parse is dropped and logged, once per robot session; the session goes on. A robot app that agrees robot state starts a
robot session, which empties the store; when it leaves, the store keeps its series and records but reports that no
robot app is connected.

**Reader.** `RobotState` is the one class a reader uses: `description()`, `series()`, `joints(name)`, `gripper(name)`,
`status(name)`, `record(name)`, and on each reading its age and whether it is fresh. Inside the server's process,
`RobotState.from_store()` (or `teleop_session_server.hosted_robot_state()`) reads the server's store directly. In a
program of its own, `RobotState.connect()` keeps a store of its own and fills it with snapshots of the server's. A
reader program and the Teleop Session Server exchange robot state over a local socket that belongs to the isaaccapture
package. Its format is private and may change in any release; it is not an interface. Use the calls of
`isaaccapture.robot_state`.

A read returns None when the series has no record in this robot session, when the record is not fresh (unless the
caller passes `allow_stale=True`), while no robot app is connected, and while the server cannot be reached. A reader
never raises because the server is not running; `server_reachable` and `connected` say which of the two is missing.
A joint the robot app marked as not valid reads as nan in position, velocity and effort: its record carries 0, which
a reader could take for a position.

**Time.** A record carries its reading time on the robot machine's monotonic clock, and each payload carries the robot
link's clock offset at the moment it was built (the robot app PINGs; the server sends no PING). The store
converts the reading time to the server machine's clock with that offset and gives the reading age against the
monotonic clock of the machine it is on; a reader program runs on the server machine, so its clock is the same one.
Until the robot app's first PONG the offset is a one-way estimate: the age is then too large by the shortest one-way
delay, never too small, and `clock_two_way` on the reading says so. The age is never less than the time since the
record arrived, so a reading time ahead of the robot machine's clock cannot keep a record fresh. The reading time
itself stays on the robot machine's clock and is not comparable with a time taken on the server machine.

## 6. Wire protocol

`messages.hpp` and `message_header.hpp` in `src/core/remote_link/cpp/inc/remote_link/` define the format.

| Link message | Payload |
|---|---|
| HELLO | JSON: protocol number, Isaac Capture version, app name (only logged), tick rate, trackers, topics, and `"robot_state": {"version": 1}` only from a robot app with a robot description. The server refuses one it cannot serve, e.g. with another protocol number, an unknown tracker type or one that needs an OpenXR extension the session lacks, a non-empty `plugins` list, over 1 MiB, over 256 trackers or 1024 topics, or a text over 1024 characters, or a `robot_state` that is not an object with an integer `version` of at least 1 |
| HELLO_ACK | JSON: `session_id`, `server_version`, and `"robot_state": {"version": 1}` only when HELLO offered robot state and the server has a store |
| STATUS, READY | JSON: stage, message, headset state |
| ERROR | Text, shown to the user as is |
| SETUP, SAMPLES | SETUP: MCAP schema and channel records. SAMPLES: tick sequence number and tick time, then the MCAP message records since the last SAMPLES |
| PING, PONG, HEARTBEAT | PONG: when the robot app's PING was sent and received; others empty |
| ROBOT_STATE_SETUP (40) | Robot app to server. MCAP schema and channel records of the declared series; each channel's metadata holds the series' `shape`, `stale_after_ns`, `delivery` (`latest` or `in_order`) and `rate_hz` |
| ROBOT_STATE_SAMPLES (41) | Robot app to server. A 32-byte prefix, then MCAP message records of series delivered newest only |
| ROBOT_STATE_RECORDS (42) | Robot app to server. The same prefix, then MCAP message records of series delivered in order (the description) |

**Delivery.** Every link message except SAMPLES (and ROBOT_STATE_SAMPLES, below) is delivered in order, and a SAMPLES
never overtakes one queued before
it, so SETUP arrives before the samples that use it. SAMPLES go to the socket only while less than 64 KiB is unsent
there (`TCP_NOTSENT_LOWAT`); meanwhile the sender keeps at most one unsent SAMPLES, merging newer ones into it (newest
record per topic), so a slow network drops older samples instead of queueing them. The other link messages queue; a
send blocked for 3 s, or 64 MiB of them unsent, means the peer has stopped reading, and the link is lost. A payload is
at most 16 MiB.

**Liveness and clock.** Each side sends a HEARTBEAT after 500 ms without sending and declares the link lost after 3 s
without receiving (the peer timeout; the robot app's starts with the first message, section 12). The robot app sends
PINGs, 10 at 50 ms intervals and then one a second; the exchange with the shortest of the last 32 round trips gives the
clock offset, and a one-way estimate stands in until the first PONG.

**Robot state: agreement.** The robot-state link messages are sent only after HELLO and HELLO_ACK both carried
`robot_state`. Types 43 to 47 are kept free for robot state.

| Robot app | Teleop Session Server | Result |
|---|---|---|
| no robot description | any | HELLO and HELLO_ACK hold the same bytes as without robot state; no robot-state link message |
| robot description | built without robot state | HELLO has the key, which that server ignores; HELLO_ACK lacks it; the robot app logs one warning, sends no robot-state link message, and its publish calls return False |
| built without robot state | has a store | neither HELLO nor HELLO_ACK has the key; the store is not attached |
| robot description | has a store | both have the key; the server attaches the store before it sends HELLO_ACK; the robot app sends ROBOT_STATE_SETUP, then the description, right after HELLO_ACK |

A server asked for a version above 1 answers with version 1; the robot app takes robot state as agreed only for an
answer of exactly 1. A peer built without robot state drops a link message type it does not know, and it never
receives one of the three.

**Robot state: payloads.** The MCAP records are the ones a file holds: a Schema and a Channel record per series in
ROBOT_STATE_SETUP (channel id and schema id count from 1 in declaration order), and Message records with the series'
record counter, the reading time as log time and the time of the publish call as publish time, both on the robot
machine's monotonic clock. The prefix of ROBOT_STATE_SAMPLES and ROBOT_STATE_RECORDS, little-endian:

| Offset | Type | Field |
|---|---|---|
| 0 | u64 | count of payloads of this type the robot app has built, from 1 |
| 8 | i64 | robot machine's monotonic clock when the payload was built, ns |
| 16 | i64 | clock offset, robot machine minus server machine, ns |
| 24 | u32 | flags: bit 0, the offset is valid; bit 1, it comes from a two-way exchange; other bits 0 |
| 28 | u32 | 0 |

**Robot state: delivery.** ROBOT_STATE_SETUP and ROBOT_STATE_RECORDS are delivered in order, so the declarations and
the description arrive whole. ROBOT_STATE_SAMPLES is delivered as SAMPLES is: while the socket cannot take it the
sender keeps one unsent payload and merges a newer one into it, keeping the newer prefix and the last record of each
series, so no series is lost and a slow network drops older readings instead of queueing them. Each publish call sends
its record at once; records of one series leave in the order they were published, also from several threads. A record
is at most 1 MiB and a robot app declares at most 256 series; the 16 MiB payload limit applies to a merged
ROBOT_STATE_SAMPLES too.

## 7. The Teleop Session Server

The server is one process, `python -m isaaccapture.teleop_session_server`, serving one robot app at a time.

**For as long as the server runs** it holds:

- the CloudXR runtime, which it attaches to or starts, and whose health it checks every second; a runtime it starts
  keeps running after the server exits. With `--no-launch-cloudxr-runtime` it does neither and uses the OpenXR runtime
  already configured;
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
  turns the capture off;
- one `RobotStateStore` and, from the first robot app that agreed robot state until the server stops, a
  `RobotStateLocalService` that hands snapshots of the store to reader programs (see "Robot state" below). A server
  that never serves robot state creates no socket.

**Per robot app**, from when the server serves it until it leaves, the server has:

- a `RemoteSessionHost` on the connection, created when the server serves the robot app; a held robot app (see "One
  robot app at a time") has none, and its connection is not read. The server answers HELLO with HELLO_ACK, or ERROR
  with the reason; a connection that sends no HELLO within 5 s is closed unanswered;
- the robot app's trackers, started on the OpenXR session once it exists and updated at the robot app's tick rate,
  clamped to 1-240 Hz;
- STATUS on every change and, once the trackers run, about once a second: stage `connecting_openxr` while there is
  no OpenXR session yet, `waiting_for_headset` until `openxr/headset` has been CONNECTED for 1.1 s (longer than the
  1 s a tracking loss takes to show), then READY (stage `ready`), sent once, and `running`. After READY a headset that
  disconnects changes only the STATUS headset state;
- when its HELLO offers robot state: the server's store, attached to its `RemoteSessionHost` before HELLO_ACK is sent,
  which starts a robot session.

When the robot app leaves, `RemoteSessionHost.close()` drops its trackers; the OpenXR session stays for the next one.
`close()` also ends the robot session, once: the store then reports that no robot app is connected.

**Plugins.** The server never starts, stops, finds or monitors plugins, and a robot app cannot request them: a HELLO
that lists plugins is refused. Operators start them on the server machine, by hand or with a script, e.g. the Manus
glove plugin (`src/plugins/manus`). A robot app's plugin trackers read whatever the running plugins push.

**Headset.** The server's OpenXR session draws nothing, so the headset gets no video: its web client must have
Headless ticked, and CloudXR must start with `TELEOP_CLIENT_WARMUP_END_TIMEOUT_MS=0`, or the client drops the stream
after about 30 s (user doc, "Headset settings"). Video from the robot (section 1) would change this.

**One robot app at a time.** A robot app that connects while another is served is held, its connection not yet read,
for up to 4 s: longer than the 3 s peer timeout, so a robot app that reconnects after its own link dropped is served.
If the served one leaves in that time, the held one is served; otherwise it gets ERROR `busy: ...`. One that connects
while another is held gets the busy ERROR at once. The served robot app keeps ticking meanwhile: refusals run beside
the loop, and while 8 are in progress a further connection is closed unanswered, which its robot app retries.

**Failures.** When the CloudXR health check fails, or the OpenXR session cannot be opened for a reason other than a
missing headset, or reports a failure, the server sends ERROR to the robot app it serves (`shutting down` for CloudXR,
the OpenXR failure otherwise), refuses the waiting ones as when stopped (section 12), closes its links and exits with
code 1, for a supervisor to restart it. Any other error in its loop also ends it, after the same `shutting down`
ERRORs. Before serving, it exits with code 1 when a listener cannot bind or CloudXR is unavailable, and with code 2
off Linux. When starting or updating a robot app's trackers fails, e.g. on a plugin sample that is not valid data for
its tracker (a wrong `collection_id`), or serving it fails in any other way, that robot app gets ERROR and its
connection ends; the server and its OpenXR session keep running.

**Robot state.** The server hosts the receiving side and does not implement it: the store is a library class
(`RobotStateStore`), filled by `RemoteSessionHost` on the link's receive thread, and nothing in the server's loop reads
or writes robot state. The server creates the store, gives it to each robot app's connection, and starts the
`RobotStateLocalService` at the first agreement; that service runs on a thread of its own, beside the loop, as the
refusals do. Together with the separate link message types (section 6), this keeps moving the receiving side into a
program of its own a change of where the class runs (section 12).

The reader socket is a Unix socket, by default `$XDG_RUNTIME_DIR/isaaccapture/robot_state.sock`, or
`/tmp/isaaccapture-<uid>/robot_state.sock` when `XDG_RUNTIME_DIR` is unset or not a directory; `--robot-state-socket`
names another path. Its directory must belong to the server's user and be closed to group and others (the server creates
it with mode 0700), the socket has mode 0600, and a client of another user is closed. The service serves at most 16
reader programs at a time, one connection each; it closes a further one and logs one warning, and that reader reports
`server_reachable` False and reads None until another has closed. When the socket cannot be opened, e.g. because another
Teleop Session Server serves robot state at that path, the server logs a warning and goes on: its robot app still sends,
and `hosted_robot_state()` still reads, but reader programs see no server. Until the first robot app has agreed robot
state the socket does not exist, so a reader program reports `server_reachable` False although the server runs.

**Robot state when the headset or CloudXR goes away.** A headset that disconnects and connects again leaves the server,
the link and the store running: after READY it changes only the STATUS headset state, and the robot app keeps
publishing. When CloudXR crashes, the server exits (see "Failures"). Whether that is intended is undecided. It follows
from the server's rule to stop on a failed CloudXR health check or a failed OpenXR session, which suits a server that
carries headset data only. With robot state hosted in the same process it means: from the crash until a
supervisor has started the server again and the robot app has entered a new session, no robot state reaches the server
machine; a reader program sees `server_reachable` False and reads None; the robot app's `step()` raises, and its
publish calls return False. The robot app does not reconnect by itself (section 13), and entering again waits for
READY, which needs CloudXR and a tracked headset. The alternative, a server that treats a CloudXR crash as "no human
input" and keeps its link and store, is not built.

## 8. Status reporting

`get_status()` reports the server machine's OpenXR session from the newest STATUS: `openxr/runtime` and
`openxr/headset`. As in live mode, `openxr/headset` is CONNECTED while CloudXR reports a tracked head pose and
DISCONNECTED about a second after it stops; a headset taken off but awake stays CONNECTED, so neither it nor READY
shows that an operator is present. While no STATUS arrives, both keep the last one, until the link is lost after 3 s.
With a controller input source, `openxr/controller/left` and `/right` come from the controller samples the robot app
receives, as in live mode, so they turn DISCONNECTED about a second after the samples turn stale. Plugins do not
appear in it, and a plugin that stops raises no `PluginCrashException`: the server does not run plugins (section 7).

## 9. Errors and timeouts as the robot app sees them

| Situation | Result |
|---|---|
| A config, input or tracker that remote mode refuses; a `wss://` endpoint | `ValueError`, before any network traffic |
| Nothing accepts the connection, or the Teleop Session Server closes or resets it before sending anything (e.g. killed while the connection waited, or 8 refusals in progress) | Retried about every 0.5 s (an unreachable host's attempt takes up to 2 s) |
| No READY within `remote_connect_timeout_s` | `TimeoutError`: `Teleop Session Server at <endpoint> was not ready after <t> s: <stage>`, the stage being the last STATUS stage, `no status from the server yet`, or `server did not answer` (with the last error, if any) |
| ERROR: busy, shutting down, refused HELLO, OpenXR, tracker or other failure (a killed server sends none) | `RuntimeError` with its text, from entering or `step()`, not retried; to wait out busy or a restart, retry entering |
| Failed name lookup, `tcp://` to a WebSocket port or vice versa, or a close after partial data | `RuntimeError`, at once (at a busy server, after its hold of up to 4 s) |
| Link lost after HELLO_ACK; incompatible topic schema | `RuntimeError` from entering or `step()`: `Remote session lost: ...` or `MCAP schema mismatch on '<topic>': ...` |
| `remote_connect_timeout_s=None`, and the server machine vanishes after accepting the connection, before answering | Nothing until TCP gives up, many minutes later (over `ws://` maybe never) |
| Ctrl-C while entering | `KeyboardInterrupt` within about 0.1 s, or after the TCP connect attempt in progress (at most 2 s, plus any host name lookup) |

## 10. Security model

The link has no authentication and no encryption: any machine that reaches its port can receive tracker data. It cannot
start programs there: plugins are started on the server machine itself. It can take the one robot-app slot, so the real
robot app gets `busy`, and read any `collection_id`. HELLO is bounded (section 6), and a plugin sample that is not valid
data for its tracker ends that robot app's connection instead of crashing the server. Run the Teleop Session Server on a
trusted network, open its ports only toward the robot machine, and, since `wss://` is not supported, encrypt the link
with a tunnel (VPN, SSH or stunnel).

Robot state widens what such a machine can do. Any machine that reaches the server's port and gets the robot-app slot
can offer robot state in its HELLO and then write a robot description and records into the store; a reader takes them
as the robot's, and nothing in the store or the reader tells them from the real robot app's. Doing so starts a robot
session, which empties what the real robot app sent. While the real robot app is served, the intruder is refused as
`busy`; it gets the slot whenever the robot app is not connected. A reader must therefore not let robot state alone
authorise an action on the robot. The link carries no robot state toward the robot app, so a machine on the link
cannot read robot state through it; tracker data it can read as described above. What it sends is bounded: the store
keeps one record per series and at most 256 series, refuses a record or a schema above 1 MiB, a payload is at most
16 MiB, and a payload the store cannot parse is dropped without ending the server; the reader verifies a FlatBuffers
record before decoding it.

On the server machine, robot state can be read by programs of the server's user (and by root): the reader socket's
directory has mode 0700, the socket mode 0600, and the service checks the user id of each client and closes one of
another user. The socket only hands out snapshots; a client cannot write robot state through it. It is not reachable
from another machine.

## 11. Testing

No automated test needs a headset or an OpenXR runtime. C++ `RemoteSession` tests write head and hand records into a
sample stream; tests over a link stream `KeyboardTracker` data, which needs no OpenXR (`TeleopSession` itself refuses
it), and pedal and Noitom full-body data from a fake tensor runtime (`tests/cpp/core/support/tensor_test_support`).
C++ tests use an in-memory transport, `socketpair()` or loopback. Teleop Session Server tests use its test-only
`--placeholder-openxr` flag (no OpenXR session; READY at once), or run its loop in process with a scripted OpenXR
session, CloudXR runtime and robot apps on a virtual clock.

Robot-state tests need no robot either. The publisher's and the store's C++ tests check their MCAP bytes against the
MCAP library, in both directions. Session tests run client, host and store over a `socketpair()` and through the
Python bindings over loopback. `TeleopSession` and Teleop Session Server tests cover both sides of the agreement: a
HELLO and a HELLO_ACK without the key hold the same bytes as without robot state, and a peer that does not agree gets
no robot-state link message.
Reader tests fill a store by hand and run the reader socket under a temporary folder. The example
(`examples/robot_state`) is the end-to-end test: a stand-in robot publishes every series of a description through a
remote session to a Teleop Session Server started with `--placeholder-openxr`, and a stand-in harness reads it back
through `RobotState.connect()`, once for a six-joint arm and once for a two-arm robot. No robot-state test runs on a
robot, with a headset or with CloudXR.

## 12. Decisions and reasons

**A sample stream, not MCAP files.** The server records with live mode's MCAP code, into memory, and the robot
app reads with the replay impls that replay files: no tracker needed new decoding or recording logic, and the schema
check that guards old recordings also guards a robot app and Teleop Session Server on different versions.

**`step()` never waits.** Waiting would tie the robot app's loop to network jitter; staleness already tells the pipeline
that data is missing.

**The robot app's peer timeout starts with the first message,** since a connection can wait up to 4 s while another
robot app is served.

**Connecting is retried only when nothing accepts the connection or it closes before any data.** That close is a typed
`PeerClosedError` (`RemotePeerClosedError` in Python), since matching text could mistake an ERROR for a close; a
connect attempt that nothing accepted cannot carry an ERROR and is recognised by its text. Anything else is a refusal or
would fail again.

**One process, one OpenXR session.** The OpenXR session lives as long as the server, so a robot app that connects
does not wait for one to open. Alternative: a child process per robot app, with its own OpenXR session and trackers.
Not chosen: with one robot app at a time its isolation gains little, the two processes would need their own
coordination (liveness, teardown, a child that outlives its parent), and every reconnect would reopen the OpenXR
session. The cost: a crash ends the server, for its supervisor to restart. Data a robot app selects cannot cause one:
plugin trackers check each sample before decoding it, so a wrong `collection_id` fails only that robot app.

**Plugins outside the server.** Operators start plugins; the server never starts, stops, finds or monitors them.
Alternative: the robot app lists plugins in HELLO, and the server starts them for that connection. Not chosen: it
couples the robot app to plugin names and paths that exist only on the server machine; anyone who reaches the
unauthenticated port could start programs there with any arguments; and every reconnect would restart the devices. A
robot app's plugin trackers report no data while a plugin is not running, as in live mode.

**The stop path.** On SIGINT or SIGTERM, and on a failure exit (section 7), the server refuses the held robot app and
those in its listen backlog with the `shutting down` ERROR, while fewer than 8 refusals are in progress (it closes the
others unanswered, and their robot apps retry), and closes its listeners, so a robot app that connects later is refused
by the kernel and retries until a restarted server serves it. It sends the same ERROR to the robot app it serves and
closes that link gracefully, so the ERROR arrives instead of a reset, and closes that robot app's trackers before the
OpenXR session they read through, even when refusing the others fails. It waits for the refusals in progress (each up to
about 3 s) and exits with code 0 (1 after a failure). A killed server sends nothing; the robot app sees the link close.

**Sinks are not flushed.** Flushing a sink writes through a tracker to a device; a `RemoteSession` has no such tracker,
so flushing would fail and could make `step()` raise.

**Version check.** A different protocol number in HELLO is refused; a different Isaac Capture version only logs a
warning, since the per-topic schema check already rejects unreadable data. A different link header version is not
refused with an ERROR: the server cannot read the HELLO and closes the connection unanswered, so the robot app retries
until `remote_connect_timeout_s` and reports `server did not answer`.

**Robot state rides the remote teleop session, hosted by the Teleop Session Server.** The link already joins the two
machines and already has a clock offset, liveness and a connect-time handshake. The receiving side sits in the Teleop
Session Server for the time being, and two rules keep moving it into a program of its own cheap: it is a library class
(`RobotStateStore`) that the server hosts, not code in the server's loop; and robot state has its own link message
types and is never mixed into SETUP or SAMPLES. The move is then the same class in another small program and a second
link from the robot app; the description file, the records and the calls of an application and of a reader stay as
they are. The cost of hosting: robot state shares the server's lifetime, CloudXR crash included (section 7).

**Agreed in HELLO and HELLO_ACK, not by a new protocol number.** A robot app or a Teleop Session Server built without
robot state must see no difference, and a different protocol number is refused. A key that a server built without
robot state ignores, answered only by a server that has a store, gives both ends the same answer without refusing
either.

**The application hands over numbers.** The robot app names a series and passes numbers in the order of its
description file; the library checks the count, adds the names and the reading time, encodes and sends. It names no
socket, port, link message type or encoding, and neither does a reader.

**A description file and three standard shapes.** A new robot needs a file, not library code. Joints use the existing
`JointStateOutputRecord`, where a gripper is one more named joint, so robot state and the retargeting engine's joint
outputs share one record. Status and the description are new FlatBuffers tables in files of their own; no existing
schema changes. Anything else is a JSON record with fields the file declares, carried with a JSON Schema, so neither
end is rebuilt for a new shape. The file is TOML: every supported Python reads it with the standard library, so the
reader and the test environments need no extra package, and the repository's tracker list is TOML too.

**Each series carries its own freshness and delivery.** `TrackerSampleStreamReader` has one age limit for all topics
and keeps no channel metadata; the description is sent once and must never go stale, while an arm's joints go stale in
half a second. So each channel's metadata holds `stale_after_ns` and `delivery`, and the store keeps them. The
description goes in ROBOT_STATE_RECORDS because the link keeps only one unsent payload per newest-only type.

**MCAP records without the MCAP library.** The one MCAP implementation file lives in `deviceio_session`, and
`TrackerSampleStreamWriter` needs an MCAP writer. The publisher and the store read and write the Schema, Channel and
Message records themselves, so they depend on the logger only and can move to another program without
`deviceio_session`. Their tests check the bytes against the MCAP library.

**The robot app's clock offset travels with the data.** Only the robot app PINGs, so the server has no two-way offset
of its own. Each payload carries the robot link's offset; the server sends no PING and `Link` is unchanged. Ages are
then right to half the difference between the two directions' delays once the offset is two-way, and too large, never
too small, until then.

**A stale or unknown reading is None.** A reader that gets a value may act on it, so a read returns None for a record
that is not fresh, while no robot app is connected and while the server cannot be reached; `allow_stale=True` asks for
the record anyway. The store keeps a departed robot app's records, and a series that never goes stale would otherwise
read as fresh for ever.

**A private local socket between a reader program and the server.** Robot state is a few kilobytes some tens of times a
second, so a local connection is enough. The calls of `isaaccapture.robot_state` are the interface; what carries them is
not documented as a format, so it can change without touching a reader. A reader that cannot import the package would be
served by a small bridge program, which is not built, not by a second documented format. A reader and a server of
different package versions do not exchange robot state: the reader refuses a snapshot it cannot read, logs one warning
and reads None.

**A bad robot-state payload does not end the session.** The host drops it and logs it once per robot session. The
headset data on the same link is what keeps the robot moving; a fault in robot state must not stop it.

## 13. Not done yet

- Authentication and TLS on the link; `wss://` on the robot side.
- Device status that plugins report (`PluginDeviceStatusTracker`, e.g. Manus gloves) in `get_status()`.
- Sink data, e.g. haptics, toward the server machine's devices.
- A replay or synthetic Teleop Session Server mode, for headset-free tests of head and hand data over a link.
- Typed exceptions for refusals.
- Remote mode in `examples/`.
- Reconnecting after a lost link; message-channel inputs.
- Trackers with metadata (`TactileTracker`): reading the metadata once it arrives on the stream.
- MCAP recording (`mcap_config`) in remote mode.

Robot state:

- Keeping the Teleop Session Server alive through a CloudXR crash. Whether its exit is intended is undecided
  (section 7); until it is decided, robot state stops reaching the server machine with it.
- Publishing before READY. The description is sent at HELLO_ACK, but the application can publish only once the session
  is entered, which needs a tracked headset.
- Authentication or encryption of robot state, as of the link.
- Robot state toward the robot app, feedback to the headset's controllers, and a reader on the robot machine.
- A history of records (the store keeps the newest per series) and a reading time per joint.
- A gripper's position in space (a gripper is its opening, as a named joint) and standard shapes for force and pose.
- A C++ loader of the description file and typed C++ helpers: a C++ robot app builds its record bytes itself.
- FlatBuffers shapes of a robot's own in the description file (the link and the store carry them) and decoding such a
  record through its binary schema.
- Grading the robot app's declared schema against the server's with the schema compatibility check; the reader verifies
  each buffer and decodes by name.
- More than one robot app at a time.
- A bridge program for a reader that cannot import the package (section 12).
- Recording robot state to MCAP, alone or with the headset's data.
- The example on a real robot.
- An agent's commands and their answers, who controls the robot, stop, the handover and camera pictures are not
  planned for the library (section 1).
