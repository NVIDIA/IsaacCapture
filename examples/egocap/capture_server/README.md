<!--
SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: Apache-2.0
-->

# CaptureServer

A standalone WebSocket server that exposes IsaacCapture head, controller and
hand tracking, plus camera JPEG frames, to the Ego Capture UI web backend. It has no
knowledge of episodes, operators, tasks, QC, Zarr or MP4 -- see
[below](#python-api-and-protocol-reference) for the full protocol and public Python API this
package implements. It is installed and run as part of `egocap`'s single `pyproject.toml`, not
as its own separately installable package.

The server runs in one of two modes, chosen once at setup and never by a client:

- **Mock** (`--mock`, or `ServerConfig(mock=True)`, the library default): synthetic
  motion and camera frames, no hardware. Needs no `ISAAC_ROOT`, IsaacCapture,
  CloudXR, ADB or DepthAI.
- **Real** (no `--mock`, or `ServerConfig(mock=False)`): IsaacCapture tracking and
  OAK-D camera frames. It never falls back to mock data; a missing SDK, runtime or
  device fails with a clear error.

The mode reported to clients is derived from the same flag.

## Setup (mock)

```sh
cd egocap
uv sync --extra dev
uv run pytest tests/capture_server -q
uv run capture-server --mock
```

No `ISAAC_ROOT` or external checkout is needed to
install, run, or test this package in mock mode.

## Setup (real)

Real tracking needs a built IsaacCapture checkout; camera-only sessions need none
(`ISAAC_ROOT` is read only when a session requests tracking).

**IsaacCapture.** Export `ISAAC_ROOT` as the root of the IsaacCapture checkout this example lives
in, once its Python bindings are built. The server imports `isaaccapture` only from
`$ISAAC_ROOT/build/python_package/Release` and refuses a package loaded from anywhere else; it
never uses a bundled copy. Validated with `isaaccapture` 1.7+local (built from IsaacCapture commit
`22026461c`), which also provides CloudXR runtime 6.3.0 and needs the project's Python 3.12.

**CloudXR runtime.** Start the CloudXR service first, using the same build and the
project's Python, so an unrelated globally installed SDK is not launched:

```sh
export ISAAC_ROOT="/path/to/your/IsaacCapture/checkout"
PYTHONPATH="$ISAAC_ROOT/build/python_package/Release" \
    uv run python -m isaaccapture.cloudxr.service start --setup-oob --usb-local
```

`--usb-local` routes headset traffic over the USB cable (needs `adb` and `coturn`),
which is required when the headset's Wi-Fi network cannot reach the PC. The headset
must enter its VR session before poses are valid; until then the head pose is an
untracked placeholder and the status reports the headset as disconnected. The server
attaches to the running service and leaves it running. If no service is running it
reports an error instead of starting one itself; the one exception is a race, where the
SDK launcher it attaches through could start a detached service if the service exits
between our liveness check and the attach.

**OAK-D cameras.** Install the optional extra (DepthAI 3.6.1) and name each camera by
its stable device ID (the OAK-D's MXID, e.g. from `dai.Device.getAllAvailableDevices()`):

```sh
uv sync --extra oak
ISAAC_ROOT="/path/to/your/IsaacCapture/checkout" uv run capture-server \
    --camera 'front=YOUR_DEVICE_ID' --camera-usb-speed HIGH
```

`--camera NAME=DEVICE_ID` is repeatable and sets the logical names clients request;
only configured names are accepted, and one physical device cannot serve two names.
`--camera-width/--camera-height/--camera-fps/--camera-quality` set the shared stream
settings (default 1280x720, 30 fps, JPEG quality 80). `--camera-usb-speed HIGH` caps
the link at USB 2; some host ports or cables cannot boot an OAK-D at USB 3. The camera
runs on-device NV12 to MJPEG; each frame's own intrinsics are reported in the camera
status when the device provides them.

**Failure policy.** A camera that produces no frame for max(5 s, five frame periods)
counts as failed. A required camera failure ends the session; an optional one reports
a degraded camera. There is no automatic reconnection: retry by starting a new session.
If a release times out the server refuses further sessions (`BUSY`) until it is
restarted, because a timeout does not prove the device was released.

## Validation status

Implemented and exercised on a Pico 4 Ultra and an OAK-D-W through the normal server
and client path: head and controller tracking at 60 Hz with valid, changing poses;
camera JPEGs at 30 fps with real scene content; a combined tracking and camera
session; stop and restart; a missing-device error.

Implemented but not exercised on hardware: hand joints, right-controller motion,
unplugging a device mid-capture (and so the required/optional camera policy), and more
than one physical camera. Not validated: writing a real captured stream to episode files through
the recording module (the server knows nothing about episodes or files, so that is a
separate integration, not a CaptureServer feature). Manus plugin requests are wired
through the plugin's required/optional discovery, but the tested IsaacCapture build has no
Manus plugin in its bundled set and live glove data has not been observed; the plugin's
SDK initialization is also a blocking vendor call. Timestamps
are host-monotonic at the moment of polling, not hardware exposure times.

## Python API and protocol reference

### CaptureClient

Import `CaptureClient`, `CaptureStarted`, `CaptureStopped`, `CaptureStatus`, `CaptureStatusType`, `FrameDescription`, `FieldDescriptor`, `CaptureFrame` and `CameraImage` from `capture_server.client`. Control and descriptor values are Python objects; frame and image payloads remain binary. Import the other public types and enums below from `capture_server.protocol`.

```python
CaptureClient(url: str)
```

```python
async def start(
    self,
    *,
    sources: Sequence[SourceType],
    cameras: Sequence[CameraSelection] = (),
    plugins: Sequence[PluginType] = (),
    on_started: Callback[CaptureStarted] | None = None,
    on_status: Callback[CaptureStatus] | None = None,
    on_capture_frame: Callback[CaptureFrame] | None = None,
    on_capture_image: Callback[CameraImage] | None = None,
    on_stopped: Callback[CaptureStopped] | None = None,
) -> None: ...
```

```python
async def stop(self) -> None: ...
```

`Callback[T]` means `Callable[[T], None | Awaitable[None]]`: a synchronous or asynchronous function taking one value of type T. All start arguments are keyword-only. The client exposes two operations, start and stop.

#### Constructor and start arguments

| Argument | Type and accepted values | Meaning |
|---|---|---|
| Constructor `url` | Nonempty `str`, a `ws://` or `wss://` station URL. | Identifies the remote capture server. Construction creates an idle client without connecting. |
| `sources` | Sequence of unique `SourceType`: `HEAD=1`, `CONTROLLERS=2`, `HANDS=3`. Required; an empty sequence is allowed. | Chooses tracking outputs. CONTROLLERS selects both controller sides; HANDS selects both hand sides. An empty sequence supports camera-only capture. |
| `cameras` | Sequence of `CameraSelection`, default empty; at most three, with unique names. | Selects station-configured cameras by logical name and type. |
| `plugins` | Sequence of unique `PluginType`: `MANUS=1`; default empty. | Enables these tracking plugins. Manus uses its HUMAN dataset and is optional: missing plugin discovery does not itself require session failure. This selection does not require connected gloves or filter hands by vendor. |
| Five `on_...` arguments | `Callback[T] \| None`, default None; argument types appear below. | Registers notifications. None discards that notification; it does not disable acquisition. |

An empty `sources` sequence together with any enabled plugin is rejected as `INVALID_CONFIG`: an enabled plugin needs IC actually entered for discovery to run, but the real IC graph-construction API rejects an empty output mapping outright, so that combination has no valid native graph to build. Plugin requests are otherwise independent of output subscriptions — a session can enable Manus without selecting `HANDS`, and vice versa; only the specific zero-sources-with-an-enabled-plugin combination is rejected, not any particular source choice. Camera-only capture (empty `sources`, no enabled plugin) remains supported and never enters IC.

start selects a 60 Hz tracking polling rate and headset device index 0. Camera dimensions, frame rate, JPEG settings and required/optional acquisition policy belong to the station configuration. The accepted settings are available through on\_started. A remote application supplies no device paths, process settings or station-local configuration.

`await client.start(...)` opens the connection, submits the selections and returns None after the server initiates startup and the client validates the accepted configuration and stream descriptions. The on\_started callback is attempted before start returns. This means IC startup has been initiated for a tracking request; the actual capture session has not begun yet. It does not mean that IC context entry or a first sampling step has succeeded. Callbacks can run before start returns. The client generates the session ID and manages connection and operation deadlines internally.

The lifecycle separates three milestones:

| Notification | Meaning |
|---|---|
| `on_started(CaptureStarted)` | IC startup has started, and the configuration and complete binary field layout are available. Capture has not begun. For camera-only requests, camera startup has started without launching IC tracking. |
| `on_status` with `type=CaptureStatusType.BEGUN` | The actual capture session has begun. Native context entry and the first step have succeeded for tracking requests; frames and images may now follow. Individual devices can still be absent. |
| `on_stopped(CaptureStopped)` | The attempt has ended and connection cleanup is complete. This also occurs if startup fails before capture begins. |

CaptureStarted and CaptureStopped bracket an accepted startup attempt. CaptureStatusType.BEGUN marks acquisition within that attempt through on\_status. No tracking or image callback occurs before that notification.

Calling start while the client is starting, capturing or stopping raises RuntimeError and preserves the active attempt. After on\_stopped returns, the client is idle and can start another session with new selections and callbacks.

#### stop()

`await client.stop()` requests cancellation of the client's current session, including during startup. It takes no arguments and returns None after sending or scheduling the request. It is idempotent and does nothing when idle or already stopping.

Completion arrives through on\_stopped after cleanup and connection closure. stop does not wait for that callback, so a callback can safely await stop. Keep the asyncio loop running until completion; do not await the completion event inside another callback.

#### Example

```python
import asyncio
```

```python
from capture_server.client import CaptureClient, CaptureStopped, CaptureFrame
from capture_server.protocol import SourceType
```

```python
async def capture() -> None:
    """Capture head and controller tracking for ten seconds."""
    finished = asyncio.Event()
    client = CaptureClient("ws://capture-station:8082")
```

```python
    def on_frame(frame: CaptureFrame) -> None:
        """Observe frame arrival without decoding its tracking payload."""
        print(frame.sequence, frame.capture_time_ns, len(frame.payload))
```

```python
    def on_stopped(result: CaptureStopped) -> None:
        """Report the outcome and release the application's wait."""
        if result.error_message is not None:
            print(result.error_message)
        finished.set()
```

```python
    await client.start(
        sources=(SourceType.HEAD, SourceType.CONTROLLERS),
        on_capture_frame=on_frame,
        on_stopped=on_stopped,
    )
```

```python
    # The application owns capture duration and waits for final cleanup.
    try:
        await asyncio.wait_for(finished.wait(), timeout=10)
    except TimeoutError:
        await client.stop()
        await finished.wait()
```

```python
asyncio.run(capture())
```

The example's ten-second limit is application policy. It is not a client constructor or start parameter.

#### Callback order and lifetime

| Callback | Argument type | Meaning and timing |
|---|---|---|
| `on_started` | `CaptureStarted` | Once when IC startup is initiated, with resolved configuration, clocks and complete stream descriptions. The actual capture session has not begun; camera-only requests report camera startup instead. |
| `on_status` | `CaptureStatus` | One consistent struct for startup progress, acquisition establishment and subsequent observation changes; type distinguishes these events. |
| `on_capture_frame` | `CaptureFrame` | A Python frame object with stream/sequence/timestamps and the unchanged binary tracking payload. No tracking values are decoded. |
| `on_capture_image` | `CameraImage` | A Python image object with stream/sequence/timestamps and a complete encoded JPEG. |
| `on_stopped` | `CaptureStopped` | Exactly once for an accepted attempt, after earlier callbacks and connection cleanup. No later data callback belongs to that attempt. |

Callbacks execute sequentially on the asyncio loop used by start. Asynchronous callbacks are awaited. Return promptly; expensive recording or display work belongs in an application-owned bounded queue. Slow processing can cause dropped frames, visible as sequence gaps and final counters.

Successful initialization delivers on\_started, then on\_status with one or more STARTING values, then on\_status with BEGUN once, followed by data and any STATUS_CHANGE values, and finally on\_stopped. These names are CaptureStatusType enum members. BEGUN precedes every frame or image callback; acquisition can begin while individual devices remain absent. Startup failure ends with on\_stopped without BEGUN or data. Identical replayed initialization and begun wire messages are validated without repeating their notifications or resetting sequences.

Every on\_status call receives CaptureStatus with the same members. STARTING includes startup_stage, a readable message and an optional observation. BEGUN and STATUS_CHANGE include a complete observation and set startup_stage and message to None. Status changes concern inventory, state, reasons, diagnostics or camera metadata, not every pose or timestamp tick. Transport keepalive does not call on\_status.

The client derives running from the session lifecycle:

| Callback value | `running` | Meaning |
|---|---|---|
| `CaptureStarted` or `CaptureStatus(type=STARTING)` | `False` | Startup has started; acquisition has not begun. |
| `CaptureStatus(type=BEGUN)` | `True` | Acquisition is established and samples may follow. |
| `CaptureStatus(type=STATUS_CHANGE)` | `True` | Acquisition has begun and this attempt has not ended. |
| `CaptureStopped` | `False` | The attempt has ended, including startup failure or unconfirmed completion after connection loss. |

running describes acquisition lifecycle, not current data freshness or device connectivity. True does not guarantee that every requested device is present or that a valid sample is arriving. Requesting stop does not itself confirm that capture has stopped: running remains true for a begun attempt until it ends, while data callbacks are suppressed. On connection loss, False describes the client's ended attempt, not proof of remote hardware cleanup. Final completion is reported through on\_stopped; the client does not also emit a extra status notification with running=False.

```python
from capture_server.client import CaptureStatus, CaptureStatusType


def on_status(event: CaptureStatus) -> None:
    """Display startup progress and observe acquisition state changes."""
    print("Running:", event.running)
    if event.type == CaptureStatusType.STARTING:
        print(event.startup_stage.name, event.message)
    elif event.type == CaptureStatusType.BEGUN:
        print("Acquisition started")
    if event.status is not None:
        print(event.status.devices)
```

Register this function with `start(on_status=on_status, ...)`. It receives startup progress even when no device observation is available yet. STATUS_CHANGE is only delivered after BEGUN; on\_stopped remains the separate completion notification. Calling stop suppresses further frame/image callbacks while legitimate lifecycle events can still arrive before completion.

#### Validation and exceptions

| Condition | Result |
|---|---|
| Invalid URL, selections or callback arguments | ValueError before a session is accepted. |
| start while active | RuntimeError; the existing attempt continues. |
| Initial command rejection | RequestRejectedError from start; no on\_stopped because no session was accepted. |
| Initial connection/acceptance deadline expires | TimeoutError from start, with connection cleanup. |
| Connection failure before acceptance | OSError or its connection-error subclass from start. |
| Invalid incoming metadata before acceptance | ProtocolError from start, with connection cleanup. |
| stop() called before acceptance | StoppedBeforeAcceptedError from start, with connection cleanup. |
| Server termination after acceptance | on\_stopped carries the server's reason and final observations. |
| Transport or decoding failure after acceptance | Bounded cleanup, then on\_stopped; reason is None unless a server terminal reply is received. |
| A callback other than on\_stopped raises | The client attempts Stop and cleanup, then on\_stopped includes the diagnostic and any confirmed server outcome. |
| on\_stopped raises | The exception is reported to the asyncio loop's exception handler; the callback is not repeated. |

Cancelling start()'s own await (the caller's own task is genuinely cancelled) before acceptance re-raises asyncio.CancelledError after connection cleanup. Calling stop() before acceptance is a distinct, non-cancellation outcome: the task running start() was never itself cancelled, so it raises StoppedBeforeAcceptedError instead, never a bare CancelledError. Cancellation after acceptance initiates cleanup and completion notification. Acceptance is the boundary between a start exception and an on\_stopped completion; an accepted attempt always has one completion notification.

`RequestRejectedError` is an Exception with the following additional members. `str(error)` returns its readable error\_message.

| Member | Type and values | Meaning |
|---|---|---|
| `session_id` | `str` | Identifies the rejected attempt. |
| `reason` | `RejectionReason`: `INVALID_CONFIG=1`, `BUSY=2`, `UNKNOWN_SESSION=3`. | Unsupported configuration, occupied station, or unknown target session, respectively. |
| `error_message` | Nonempty `str`. | Explanation for display; applications use reason rather than parsing this text. |

`ProtocolError` is a ValueError subclass for invalid received metadata, framing, layout or image content. Its exception string describes the violation; it adds no public members.

### Python value types

Callback values are immutable snapshots. Sequence members are tuples; NumPy members are read-only arrays. Retaining a value after callback return is safe: later messages do not overwrite it. Optional members are present as None when unavailable, including optional fields omitted on the wire.

The following tables specify every public member. These are received value objects, rather than additional client operations or request builders.

#### CameraSelection

A TypedDict supplied in start's cameras argument. Both keys are required.

| Key | Type and values | Meaning |
|---|---|---|
| `name` | Nonempty `str`. | Logical camera name configured on the station, such as camera\_left. An unknown name or duplicate name is rejected. |
| `type` | `CameraType`: `OAK_D_COLOR=1`. | Selects the OAK-D color-camera kind; it must match the station's named camera. |

Example: `cameras=({"name": "camera_left", "type": CameraType.OAK_D_COLOR},)`. CameraSelection contains only name and type.

#### CaptureStarted

| Member | Type and values | Meaning |
|---|---|---|
| `session_id` | `str` | Server-accepted identity generated by the client for this attempt. |
| `running` | `bool`, False. | IC startup has been initiated, but the actual capture session has not begun. Camera-only startup has the same lifecycle meaning. |
| `mode` | `CaptureMode`: `MOCK=0`, `LIVE=1`. | Synthetic or live acquisition, chosen by the station. |
| `effective_config` | `CaptureConfiguration` | Accepted selections with all defaults resolved. |
| `clock` | `CaptureClock` | Timestamp domain, unit and sampling meaning. |
| `streams` | `tuple[FrameDescription \| CameraStream, ...]` | Complete fixed stream registry with the binary tracking layout and camera formats. A descriptor announces an expected output, not a connected device. |

The client validates the wire SessionInit notification and converts it to CaptureStarted before invoking on\_started. This notification confirms that the server has initiated IC startup, not that the native capture session is established. Its tracking stream supplies every field name, wire scalar type, shape and byte offset, plus the payload size. The client validates the descriptor once and delivers subsequent payloads unchanged. A backend can forward the layout and binary payloads to a frontend without decoding and re-encoding tracking values. Initialization describes expected outputs independently of the availability observations delivered through on\_status.

#### CaptureConfiguration

| Member | Type | Meaning |
|---|---|---|
| `tracking` | `TrackingConfiguration` | Selected tracking outputs and polling configuration. |
| `plugins` | `tuple[PluginConfiguration, ...]` | Resolved plugin configuration. |
| `cameras` | `tuple[CameraSelection, ...]` | Accepted logical camera selections. Each name/type mapping is read-only in this snapshot. |

#### TrackingConfiguration and HeadsetConfiguration

| Type.member | Type and values | Meaning |
|---|---|---|
| `TrackingConfiguration.sources` | `tuple[SourceType, ...]`: `HEAD=1`, `CONTROLLERS=2`, `HANDS=3`. | Unique selected outputs; empty means no tracking stream. |
| `TrackingConfiguration.rate_hz` | Positive finite `float`, at most 120; start uses 60. | Polling cadence, not a promise of sensor exposure frequency or unique new samples. |
| `TrackingConfiguration.headset` | `HeadsetConfiguration` | Headset selection. |
| `HeadsetConfiguration.device_index` | `int`, 0 only. | First station headset. Boolean values are not indices. |

#### PluginConfiguration

| Member | Type and values | Meaning |
|---|---|---|
| `plugin` | `PluginType`: `MANUS=1`. | Plugin kind. |
| `enabled` | `bool` | Whether the station launches it; true for a plugin selected through start. |
| `required` | `bool` | Whether failure to discover the plugin fails startup; false for start selections. It does not require connected gloves. |
| `datasets` | `tuple[ManusDataset, ...]`: `HUMAN=1`. | Requested live human-skeleton dataset. start selects HUMAN. |

#### CaptureClock

| Member | Type and values | Meaning |
|---|---|---|
| `domain` | `ClockDomain`: `CAPTURE_HOST_MONOTONIC=1`. | Elapsed-time clock on the capture station, independent of the client's clock and calendar time. |
| `unit` | `TimeUnit`: `NANOSECONDS=1`. | One billion units per second. |
| `sample_time_semantics` | `SampleTimeSemantics`: `HOST_POLL=1`. | Timestamp marks host sampling, rather than a hardware exposure timestamp. |

Differences between timestamps on this clock measure station-side intervals. Comparing a station timestamp directly with a client timestamp does not measure network delay or absolute sample age. A progressing stream can still contain older sensor observations.

#### FrameDescription

An immutable description of the tracking frame format. CaptureStarted.streams supplies it once through on_started. Each CaptureFrame.description references that same object; the library does not rebuild it for each frame, and a relay sends it only when establishing the session, not with every payload. It contains fixed interpretation metadata, not changing sequence numbers, timestamps or tracking measurements.

| Member | Type and values | Meaning |
|---|---|---|
| `stream_id` | `int`, 0. | Associates tracking frames with this descriptor. |
| `kind` | `StreamKind`: `TRACKING=1`. | Descriptor kind. |
| `name` | `str`, tracking. | Logical stream name. |
| `schema` | `TrackingSchema`: `IC_TRACKING_V1=1`. | Fixed head/controller/hand schema. |
| `sources` | `tuple[SourceType, ...]`: `HEAD=1`, `CONTROLLERS=2`, `HANDS=3`. | Outputs selected for this stream. |
| `rate_hz` | Positive finite `float`, at most 120. | Requested polling cadence. |
| `pose_representation` | `PoseRepresentation`: `POSITION_QUATERNION=1`. | Poses contain a translation vector and rotation quaternion. |
| `quaternion_order` | `QuaternionOrder`: `XYZW=1`. | Quaternion components are x, y, z, w. |
| `position_unit` | `PositionUnit`: `METERS=1`. | Units for positions and joint radii. |
| `axes` | `Axes`: `RIGHT_HANDED_Y_UP_MINUS_Z_FORWARD=1`. | Right-handed coordinates, +Y up and −Z forward. |
| `reference_space` | Nonempty `str`, such as session\_floor. | Identifies the shared session coordinate space for these poses. |
| `hand_joint_order` | `HandJointOrder`: `OPENXR_26=1`. | Joint indexing of the hand arrays in the binary payload. |
| `payload_size` | `int`, 1995. | Exact tracking payload size, excluding the transport header. |
| `fields` | `tuple[FieldDescriptor, ...]` | Complete binary field descriptions, including presence and validity flags, with offsets measured within CaptureFrame.payload. |

#### FieldDescriptor

Each descriptor in CaptureStarted.streams describes one field in the binary tracking payload. Cache this layout from on_started before handling frames. The layout includes all head, controller and hand slots, including outputs that were not selected or are currently absent. It does not change within the session.

| Member | Type and values | Meaning |
|---|---|---|
| `name` | `str` | Dotted field name, such as head.position or hand_left.joint_orientations. |
| `dtype` | `WireDType`: `UINT8=1`, `FLOAT32_LE=2`, `FLOAT64_LE=3`. | Encoded scalar type: unsigned byte, little-endian 32-bit float or little-endian 64-bit float. Presence and validity flags are bytes with values 0=false and 1=true. |
| `shape` | `tuple[int, ...]` | Array dimensions; () means a scalar. Arrays use C order, with the last index varying fastest. |
| `offset` | Nonnegative `int`. | Byte offset from the start of CaptureFrame.payload. The 32-byte transport header has already been removed; do not add its size. |

Examples of the descriptors supplied through on_started:

| Name | Python enum and numeric dtype | Shape | Byte offset | Meaning |
|---|---|---|---|---|
| `head.present` | `WireDType.UINT8=1` | `()` | 0 | Whether the head output exists in this sample. |
| `head.position` | `WireDType.FLOAT32_LE=2` | `(3,)` | 1 | Head position in meters. |
| `head.orientation` | `WireDType.FLOAT32_LE=2` | `(4,)` | 13 | Head rotation as an xyzw quaternion. |
| `head.is_valid` | `WireDType.UINT8=1` | `()` | 29 | Whether the head pose is usable. |
| `hand_left.joint_positions` | `WireDType.FLOAT32_LE=2` | `(26,3)` | 278 | All 26 joint positions in session space. |
| `hand_left.joint_orientations` | `WireDType.FLOAT32_LE=2` | `(26,4)` | 590 | All 26 joint rotations as xyzw quaternions. |
| `hand_left.joint_radii` | `WireDType.FLOAT32_LE=2` | `(26,)` | 1006 | Joint radii in meters. |
| `hand_left.joint_valid` | `WireDType.UINT8=1` | `(26,)` | 1110 | Per-joint usability flags. |
| `hand_right.joint_positions` | `WireDType.FLOAT32_LE=2` | `(26,3)` | 1137 | Right-hand joint positions. |
| `hand_right.joint_orientations` | `WireDType.FLOAT32_LE=2` | `(26,4)` | 1449 | Right-hand joint rotations. |

These rows illustrate the layout; on_started supplies the complete tuple of fields. The full field table and meanings appear in Appendix A. A scalar occupies one dtype-sized value; an array occupies the product of its dimensions times the dtype size. Consumers must honor little-endian encoding and the supplied offsets: fields can be unaligned. Browser consumers can use DataView with little-endian reads rather than assuming every field can be exposed as an aligned typed-array view.

Both hands retain positions and orientations, in the declared 26-joint order. Frame callbacks never construct NumPy pose arrays or Python head/controller/hand sample objects. A consumer that needs individual values decodes them downstream using this descriptor.

#### CameraStream and ImageOrientation

| Type.member | Type and values | Meaning |
|---|---|---|
| `CameraStream.stream_id` | `int`, 1–3. | Camera stream identifier, assigned in selection order. |
| `CameraStream.kind` | `StreamKind`: `IMAGE=2`. | Descriptor kind. |
| `CameraStream.name` | `str` | Selected logical camera name. |
| `CameraStream.type` | `CameraType`: `OAK_D_COLOR=1`. | Color-camera kind. |
| `CameraStream.required` | `bool` | Whether terminal camera failure ends the entire capture. Chosen by the station. |
| `CameraStream.width` | Positive `int`. | Encoded image width in pixels. |
| `CameraStream.height` | Positive `int`. | Encoded image height in pixels. |
| `CameraStream.fps` | Positive finite `float`. | Configured camera frame rate. |
| `CameraStream.codec` | `ImageCodec`: `JPEG=1`. | Each image contains one complete JPEG. |
| `CameraStream.jpeg_quality` | `int`, 1–100. | Configured encoder quality. |
| `CameraStream.orientation` | `ImageOrientation` | Describes the encoded pixel orientation. |
| `CameraStream.intrinsics` | Read-only float64 NumPy array \[3,3\], or None. | Camera calibration matrix for these pixels; None means unknown. Later CameraStatus can provide it. |
| `ImageOrientation.rotation_degrees` | `int`, 0. | No additional pixel rotation is required. |
| `ImageOrientation.mirrored` | `bool`, false. | Pixels are not mirrored. |

The intrinsics matrix has rows \[fx, 0, cx\], \[0, fy, cy\], \[0, 0, 1\]: fx/fy are focal lengths in pixels and cx/cy are the principal point. Intrinsics do not describe a camera-to-tracking-space transform.

#### CaptureStatus

An immutable Python status value delivered to on\_status. All three event types use this same struct. It has no wire-message discriminator or protocol-version member; the client validates the incoming protocol messages and maps them into CaptureStatus before dispatch.

| Member | Type and values | Meaning |
|---|---|---|
| `session_id` | `str` | Current accepted attempt. |
| `type` | `CaptureStatusType`: `STARTING=1`, `BEGUN=2`, `STATUS_CHANGE=3`. | Startup progress, acquisition establishment, or a subsequent observation change. BEGUN is delivered once. |
| `running` | `bool`: False for STARTING; True for BEGUN and STATUS_CHANGE. | Whether acquisition has begun and this attempt has not ended. Derived consistently from the lifecycle; not an independently mutable flag. |
| `startup_stage` | `StartupStage`: `INITIALIZING=1`, `WAITING_FOR_HEADSET=2`, `CONNECTING=3`, `STARTING_SOURCES=4`; or None. | The current startup operation for STARTING. None for BEGUN and STATUS_CHANGE. It describes progress, not guaranteed device readiness. |
| `message` | Nonempty `str` for STARTING; None otherwise. | Readable explanation of the startup operation or wait. Device/provider diagnostics remain in status. |
| `status` | `StatusSnapshot \| None` | Complete observations. Can be None during STARTING before observations exist; required for BEGUN and STATUS_CHANGE. |

CaptureStatusType is a Python IntEnum independent of the wire MessageType enum. It is not a nested wrapper around the three protocol event classes. Startup progress has running=False even when status is None. Completion is delivered through CaptureStopped rather than a fourth CaptureStatusType value.

#### StatusSnapshot

A provider is software supplying observations, such as the OpenXR runtime or a tracking plugin. A device is an inventory entry belonging to a provider. Inventory can include disconnected devices and does not enumerate every output slot.

| Member | Type | Meaning |
|---|---|---|
| `updated_at_ns` | Nonnegative `int`. | Station monotonic time when observations were refreshed, not client receipt time. |
| `providers` | `tuple[ProviderStatus, ...]` | Monitored software providers. |
| `devices` | `tuple[DeviceStatus, ...]` | Complete monitored IC device inventory. |
| `cameras` | `tuple[CameraStatus, ...]` | Camera observations managed by the server. |

Snapshots are complete replacements, not changes to merge into the old inventory. A timestamp change alone does not trigger STATUS_CHANGE. Startup progress can still produce STARTING notifications while a startup_stage persists. A running provider does not prove connected hardware; readable tracking data does not prove its vendor.

#### ProviderStatus

| Member | Type and values | Meaning |
|---|---|---|
| `id` | Nonempty `str`. | Opaque identity, stable within the session. |
| `type` | `ProviderType`: `PLUGIN=1`, `OPENXR_RUNTIME=2`. | Plugin process or runtime provider. |
| `plugin` | `PluginType`: `MANUS=1`, or None. | Plugin kind; None for the runtime. |
| `name` | `str` | Human-readable provider name. |
| `status` | `ProviderState`: `INITIALIZING=1`, `AVAILABLE=2`, `FAILED=3`, `STOPPED=4`. | Preparing, operational, failed, or ended. AVAILABLE concerns software readiness. |
| `reason` | `StatusReason`, values 0–23 in the enum reference below. | Machine-readable explanation of the state. |
| `error` | `str`, possibly empty. | Additional readable diagnosis. |
| `pid` | Positive `int` or None. | Observed plugin process ID; diagnostic only. |
| `exit_code` | `int` or None. | Observed process exit code; not a StatusReason value. |
| `term_signal` | Positive `int` or None. | Observed termination signal number. |

#### DeviceStatus

| Member | Type and values | Meaning |
|---|---|---|
| `id` | Nonempty `str`. | Opaque identity, stable within the session. |
| `path` | `str` | Provider-relative logical path, such as /hand/left; not a filesystem path. |
| `type` | `str` | Native manifest device category, such as hand. |
| `description` | `str` | Human-readable device description. |
| `provider_id` | `str` | Matches the owning ProviderStatus.id. |
| `status` | `DeviceState`: `INITIALIZING=1`, `UNKNOWN=2`, `CONNECTED=3`, `DISCONNECTED=4`, `DEGRADED=5`, `FAILED=6`, `UNAVAILABLE=7`, `DISABLED=8`. | Preparing, not yet known, connected, disconnected, partially usable, failed, prevented by provider availability, or disabled by configuration. |
| `reason` | `StatusReason`, values 0–23 below. | Explains this device's state; can differ from its provider's reason. |
| `error` | `str`, possibly empty. | Additional readable diagnosis. |

#### CameraStatus

| Member | Type and values | Meaning |
|---|---|---|
| `name` | `str` | Selected logical camera name. |
| `stream_id` | `int`, 1–3. | Corresponding CameraStream identifier. |
| `status` | `DeviceState`: `INITIALIZING=1`, `UNKNOWN=2`, `CONNECTED=3`, `DISCONNECTED=4`, `DEGRADED=5`, `FAILED=6`, `UNAVAILABLE=7`, `DISABLED=8`. | Server-observed camera acquisition condition, with the meanings above. |
| `reason` | `StatusReason`, values 0–23 below. | Explains the camera condition; DEVICE\_ERROR=11 describes an acquisition error. |
| `error` | `str`, possibly empty. | Readable camera diagnosis. |
| `intrinsics` | Read-only float64 NumPy array \[3,3\], or None. | Latest calibration for the descriptor's image dimensions. None means unknown. |

An optional camera can remain unavailable while other streams continue. A required camera's terminal failure completes the capture with StopReason.CAMERA\_ERROR.

#### CaptureFrame

An immutable Python dataclass containing ordinary metadata fields and a binary payload. The library parses the transport header to expose its integers, validates the registered stream, sequence and payload length, and passes the tracking payload through unchanged. It does not unpack poses, joints, buttons, presence or validity fields, convert tracking values to NumPy/Python arrays, or serialize them as JSON. There is no automatic payload decoder or decoded-sample member.

| Member | Type and values | Meaning |
|---|---|---|
| `session_id` | `str` | Session owning the frame. |
| `description` | `FrameDescription` | The same immutable layout object supplied in CaptureStarted.streams for this stream. |
| `stream_id` | `int`, 0. | Tracking stream descriptor supplied through on_started. |
| `sequence` | Nonnegative `int`. | Per-stream sample counter starting at zero. Gaps identify samples not delivered to the application. |
| `capture_time_ns` | Nonnegative `int`. | Station monotonic host sampling time, interpreted using CaptureClock. |
| `receive_time_ns` | Nonnegative `int`. | Station-side time the server assembled this sample for transport, not the adapter's own sampling time (that is `capture_time_ns`) and not the time the remote client received the frame. |
| `payload` | Immutable `bytes`, exactly 1995 bytes for IC_TRACKING_V1. | Original packed tracking payload, without the 32-byte transport header. Retaining the frame after callback return is safe. |

CaptureFrame.description is a shared reference, not a per-frame copy or another wire message. Access changing metadata directly, such as frame.sequence; there is no nested metadata/header object or separate FrameMetadata type. The callback identifies tracking versus image data, so no redundant kind field is needed in Python. The transport kind is still validated internally.

CaptureFrame.payload can be forwarded as a binary WebSocket message without decoding or re-encoding its values. A browser/backend protocol must supply the corresponding session/stream identity, sequence and timestamps alongside that payload and deliver the layout before the first frame. Those metadata are not embedded in the payload itself. A relay does not convert the payload to JSON numbers or base64. Retaining binary does not imply a guarantee that networking libraries make no buffer copies.

Presence and validity remain encoded in the payload. An absent or unselected group has present=0; a present group can still be invalid or untracked. Downstream consumers check the group and per-joint validity before using measurements. Placeholder values are not observations. Device status remains separate from sample presence and validity.

#### CameraImage

| Member or method | Type | Meaning |
|---|---|---|
| `session_id` | `str` | Session owning the image. |
| `stream_id` | `int`, 1–3. | Camera stream descriptor supplied through on_started. |
| `sequence` | Nonnegative `int`. | Per-camera sample counter starting at zero. |
| `capture_time_ns` | Nonnegative `int`. | Station monotonic image sampling timestamp, interpreted using CaptureClock. |
| `receive_time_ns` | Nonnegative `int`. | Station-side time the server assembled this sample for transport, not the adapter's own sampling time (that is `capture_time_ns`) and not client receipt time. |
| `name` | `str` | Logical camera name resolved from the stream registry. |
| `jpeg` | `bytes` | Complete immutable JPEG, without a binary transport header. |
| `decode()` | Returns writable NumPy uint8 array \[height,width,3\]. | Decodes a new owned RGB image and validates its dimensions against CameraStream. Raises ProtocolError for invalid content or mismatched dimensions. Takes no arguments. |

JPEG delivery performs no automatic pixel decoding. decode is an explicit, synchronous image-only operation; relays forward jpeg without calling it. It does not rotate, mirror or transform the pixels. Camera metadata fields are direct Python attributes, with no FrameMetadata wrapper. No image callback occurs while a camera is unavailable.

#### CaptureStopped and StreamCounters

CaptureStopped is the client's completion value. A non-None reason confirms a server terminal reply. A None reason means the connection or client failed before terminal confirmation; it never implies a fabricated IC failure.

| Type.member | Type and values | Meaning |
|---|---|---|
| `CaptureStopped.session_id` | `str` | Completed attempt. |
| `CaptureStopped.running` | `bool`, False. | The attempt has ended, including when no server terminal reply could be confirmed. |
| `CaptureStopped.reason` | `StopReason`: `REQUESTED=1`, `OWNER_DISCONNECTED=4`, `SERVER_SHUTDOWN=6`, `IC_ERROR=7`, `IC_EXITED=8`, `CAMERA_ERROR=9`; or None. | Successful requested cleanup, owner loss, server shutdown, IC operation/cleanup error, unexpected acquisition-worker end, or fatal camera error. None means unconfirmed local completion. |
| `CaptureStopped.error_message` | Nonempty `str` or None. | Readable failure explanation. None for normal closure; can include a local callback failure while preserving the actual server reason. |
| `CaptureStopped.status` | `StatusSnapshot \| None` | Server's final available observation when confirmed; otherwise the last observed snapshot, which may be old. None means no observation exists. |
| `CaptureStopped.streams` | `tuple[StreamCounters, ...] \| None` | Confirmed final counters, possibly an empty tuple. None means final counters are unknown. |
| `StreamCounters.stream_id` | `int`, 0–3. | Initialized stream being counted. |
| `StreamCounters.produced` | Nonnegative `int`. | Complete samples generated. |
| `StreamCounters.sent` | Nonnegative `int`. | Completed server sends, not confirmation that an application recorded the samples. |
| `StreamCounters.dropped` | Nonnegative `int`. | Generated samples discarded before completed send, including shutdown queue disposal. |

Confirmed final counters satisfy produced = sent + dropped. An empty tuple means no initialized producer counters; None means no confirmed accounting. StopReason explains why the session ended. StatusReason explains a particular provider, device or camera condition and remains inside its status record.

For camera streams specifically, `produced` counts samples this adapter actually received from its native queue, not samples the camera generated. A size-one, non-blocking native encoded-frame queue, internal to the camera SDK and upstream of this adapter, can be overwritten by a newer frame before this adapter polls it; that overwritten frame never reaches the adapter at all, so it is invisible to `produced`/`sent`/`dropped` and does not appear as a sequence gap in `CameraImage.sequence` either (that counter only advances on frames this adapter actually delivers). The native SDK exposes `getSequenceNum()` on each delivered frame, which would detect such a gap, but folding that into `produced`/`dropped` would redefine `produced` from "generated by this adapter" to "generated by the camera," a protocol-semantics change outside this document's current contract. This queue is distinct from the server's own per-stream outbox (downstream of every adapter, tracking and camera alike), whose size-one drop-oldest eviction behavior is separately counted in `produced`/`sent`/`dropped` — only the native upstream encoder queue's overwrite, described above, is unaccounted for.

### Python enums

All enums are IntEnum classes. Supply enum members to the Python API; incoming numeric fields are converted to their corresponding enum. Ordinary quantities such as widths, rates, device indices and process exit codes are not enums. Unlisted values are unsupported; unused StopReason values 2, 3 and 5 are reserved.

| Python enum | Members and numeric values |
|---|---|
| `MessageType` | `BEGIN=1`, `STOP=2`, `SESSION_STARTING=3`, `SESSION_BEGUN=4`, `SESSION_STATUS=5`, `SESSION_STOPPED=6`, `CAPTURE_FRAME=7`, `CAPTURE_IMAGE=8`, `SESSION_INIT=9`, `REQUEST_REJECTED=10` |
| `CaptureStatusType` | `STARTING=1`, `BEGUN=2`, `STATUS_CHANGE=3` |
| `SourceType` | `HEAD=1`, `CONTROLLERS=2`, `HANDS=3` |
| `PluginType` | `MANUS=1` |
| `ManusDataset` | `HUMAN=1` |
| `CaptureMode` | `MOCK=0`, `LIVE=1` |
| `StartupStage` | `INITIALIZING=1`, `WAITING_FOR_HEADSET=2`, `CONNECTING=3`, `STARTING_SOURCES=4` |
| `StopReason` | `REQUESTED=1`, `OWNER_DISCONNECTED=4`, `SERVER_SHUTDOWN=6`, `IC_ERROR=7`, `IC_EXITED=8`, `CAMERA_ERROR=9` |
| `RejectionReason` | `INVALID_CONFIG=1`, `BUSY=2`, `UNKNOWN_SESSION=3` |
| `StreamKind` | `TRACKING=1`, `IMAGE=2` |
| `BinaryKind` | `CAPTURE_FRAME=1`, `CAPTURE_IMAGE=2` |
| `TrackingSchema` | `IC_TRACKING_V1=1` |
| `WireDType` | `UINT8=1`, `FLOAT32_LE=2`, `FLOAT64_LE=3` |
| `CameraType` | `OAK_D_COLOR=1` |
| `ImageCodec` | `JPEG=1` |
| `ClockDomain` | `CAPTURE_HOST_MONOTONIC=1` |
| `TimeUnit` | `NANOSECONDS=1` |
| `SampleTimeSemantics` | `HOST_POLL=1` |
| `PoseRepresentation` | `POSITION_QUATERNION=1` |
| `QuaternionOrder` | `XYZW=1` |
| `PositionUnit` | `METERS=1` |
| `Axes` | `RIGHT_HANDED_Y_UP_MINUS_Z_FORWARD=1` |
| `HandJointOrder` | `OPENXR_26=1` |
| `ProviderType` | `PLUGIN=1`, `OPENXR_RUNTIME=2` |
| `ProviderState` | `INITIALIZING=1`, `AVAILABLE=2`, `FAILED=3`, `STOPPED=4` |
| `DeviceState` | `INITIALIZING=1`, `UNKNOWN=2`, `CONNECTED=3`, `DISCONNECTED=4`, `DEGRADED=5`, `FAILED=6`, `UNAVAILABLE=7`, `DISABLED=8` |
| `StatusReason` | `NONE=0`, `PROVIDER_INITIALIZING=1`, `PROVIDER_FAILED=2`, `PROVIDER_STOPPED=3`, `NO_REPORT=4`, `STALE_REPORT=5`, `MALFORMED_REPORT=6`, `UNSUPPORTED_SCHEMA_VERSION=7`, `NO_HARDWARE_SIGNAL=8`, `CALIBRATION_FAILED=9`, `NO_CURRENT_DATA=10`, `DEVICE_ERROR=11`, `DISABLED_BY_CONFIGURATION=12`, `PROCESS_EXITED=13`, `PROCESS_SIGNALED=14`, `PROCESS_OBSERVATION_ERROR=15`, `STARTUP_FAILED=16`, `RUNTIME_UPDATE_FAILED=17`, `OPENXR_SESSION_READY=18`, `EXTERNAL_OPENXR_HEALTH_UNKNOWN=19`, `OPENXR_FORM_FACTOR_UNAVAILABLE=20`, `OPENXR_SESSION_LOST=21`, `OPENXR_INSTANCE_LOST=22`, `OPENXR_POLL_ERROR=23` |

StatusReason meanings are:

| StatusReason members | Meaning |
|---|---|
| `NONE=0` | No additional reason. |
| `PROVIDER_INITIALIZING=1`, `PROVIDER_FAILED=2`, `PROVIDER_STOPPED=3` | The associated provider is preparing, failed or stopped. |
| `NO_REPORT=4`, `STALE_REPORT=5` | No device report yet, or the latest report is too old. |
| `MALFORMED_REPORT=6`, `UNSUPPORTED_SCHEMA_VERSION=7` | A report cannot be interpreted or uses an unsupported schema. |
| `NO_HARDWARE_SIGNAL=8` | No connection/presence signal for the hardware. |
| `CALIBRATION_FAILED=9` | Reported calibration failure. |
| `NO_CURRENT_DATA=10` | Hardware may exist, but current usable data is unavailable. |
| `DEVICE_ERROR=11` | Device/acquisition failure with details in error. |
| `DISABLED_BY_CONFIGURATION=12` | Configuration disables this device. |
| `PROCESS_EXITED=13`, `PROCESS_SIGNALED=14` | Provider process ended normally at the OS level or was terminated by a signal; consult exit\_code or term\_signal. |
| `PROCESS_OBSERVATION_ERROR=15` | Provider process health could not be observed. |
| `STARTUP_FAILED=16` | Native provider/runtime initialization failed. |
| `RUNTIME_UPDATE_FAILED=17` | Updating native runtime data raised an error. |
| `OPENXR_SESSION_READY=18` | The owned OpenXR session is ready; this does not certify every output is tracked. |
| `EXTERNAL_OPENXR_HEALTH_UNKNOWN=19` | An externally owned OpenXR session's health is not known. |
| `OPENXR_FORM_FACTOR_UNAVAILABLE=20` | The requested OpenXR device form factor is unavailable. |
| `OPENXR_SESSION_LOST=21`, `OPENXR_INSTANCE_LOST=22` | OpenXR session or instance was lost. |
| `OPENXR_POLL_ERROR=23` | Polling OpenXR state failed. |

### Appendix A: Protocol reference

The protocol uses version 1. JSON commands and lifecycle notifications accompany binary tracking and image messages on one owner WebSocket. This appendix specifies every message, field and encoded layout.

Wire SessionInit and its tracking descriptors contain the byte layout required to interpret the tracking payload. The library validates that metadata and exposes CaptureStarted through on\_started, with the Python FrameDescription and FieldDescriptor values documented above, including all binary field offsets and the payload size. The distinct wire SessionStarting, SessionBegun and SessionStatus messages map to the single Python CaptureStatus struct delivered through on\_status. They retain their wire names and numeric types.

| Wire MessageType | Python CaptureStatusType | Python running | Python startup_stage/message |
|---|---|---|---|
| SESSION_STARTING=3 | STARTING=1 | False | Copy the wire stage and message. |
| SESSION_BEGUN=4 | BEGUN=2 | True | Both None. |
| SESSION_STATUS=5 | STATUS_CHANGE=3 | True | Both None. |

The Python CaptureStatus type and running members are derived values, not additional wire fields. Python startup_stage corresponds to wire SessionStarting.stage; the wire key remains stage. Session IDs and complete observations are preserved, including absent startup observations.

The client library parses the binary transport header into direct CaptureFrame or CameraImage integer fields before callbacks run. The tracking payload itself remains an unchanged binary blob. Header parsing does not decode any tracking samples.

running is a Python-only lifecycle convenience on callback values. It is derived from SessionInit, SessionStarting, SessionBegun and session termination; it adds no JSON field or binary payload field. StatusSnapshot remains the native provider/device/camera observation and carries no session running flag.

#### Integer enums and wire types

Every closed protocol choice is a typed Python `IntEnum` at both peers and a JSON integer or binary integer on the wire. Do not send enum names or string values. Assign values explicitly; never use `auto()`, reorder values or reuse a published value. Reject booleans, floats and unknown integers where an enum is expected. These are server protocol assignments, not purported native IC enum numbers: translate native IC enum members explicitly at the adapter boundary.

JSON object keys, session IDs, provider/device IDs, device paths, stream names, reference-space identifiers, plugin manifest device type/description and human-readable diagnostics remain strings. They identify resources or carry native metadata; they are not protocol dispatch enums. Native IC status enums currently use string values internally; those strings do not cross as protocol enum values. Returned resource IDs are opaque: clients compare and return them without parsing their contents or resolving them on their own computer.

The [Python enum reference](#python-enums) lists every enum member and numeric wire value. Each message-field table also names its enum and accepted values.

BinaryKind is the compact binary-header discriminator; MessageType discriminates JSON messages. They are separate enum types. Version remains integer 1. The 32-byte framing and 1995-byte tracking payload below remain fixed; controller scalars remain float64.

#### How IC selects and observes devices

IC's Python package is `isaaccapture`.

The application chooses source nodes and connects their outputs into an OutputCombiner. TeleopSession discovers the graph's DeviceIO source leaves and collects their tracker instances. HeadSource owns a HeadTracker, ControllersSource owns a ControllerTracker for left/right controllers, and HandsSource owns a generic HandTracker for left/right hands. Session entry gathers their required OpenXR extensions, establishes the OpenXR session, creates DeviceIOSession and starts configured plugins. Each synchronous step updates DeviceIO, refreshes status, polls the selected sources and computes outputs.

Choosing HandsSource does not select Manus. The native hand implementation enumerates XDev candidates, tries non-display devices before display-device candidates, and retains default session hand trackers as a final fallback. At each update it selects the first active successful candidate independently per hand. Neither HandsSource's HandInput nor the current tracking payload contains a device/provider identity. A valid hand can therefore be an OpenXR headset hand when no Manus hand is active.

Manus is configured separately as a plugin. The server maps a Manus plugin request to IC PluginConfig (`plugin_name` is the manifest name `manus_hand_plugin`, matched exactly by the IC plugin manager; the search path is the package's bundled plugins plus `$ISAAC_ROOT/install/plugins`, where IC's `cmake --install --component manus` places the plugin) and launches its human dataset. The plugin injects 26-joint hands into OpenXR. Its manifest declares left/right hand devices; its status publisher reports their actual hardware/data state. This protocol supports the human dataset only; it does not expose raw flex sensors, haptics or calibration commands.

IC PluginConfig.required means discovery of the plugin is required. It does not mean both physical gloves must be connected before session entry returns. A missing required plugin or source initialization error can raise during startup. A running plugin with missing gloves can coexist with a successfully entered session. No automatic switch from a requested plugin to another plugin is promised.

Device existence, provider health and sample usability are distinct:

- Manifest inventory describes devices the configured plugin can report, including disconnected devices. It is not proof that physical hardware is attached.
- Provider status describes plugin/OpenXR lifecycle.
- Device status describes connected/disconnected/degraded/unknown and the native reason.
- Optional TensorGroup presence and its validity flags describe the current sample.

The IC status inventory includes configured Manus manifest devices and the OpenXR headset. Do not fabricate independent controller inventory/status entries merely because the tracking schema contains controller outputs. Controllers' current availability is expressed by their optional outputs and pose validity. An unknown device ID lookup returns None; an unknown/stale device report is not proof of disconnection.

#### Connection and message inventory

The capture host is the computer running CaptureServer and talking to the hardware. For example, an Orin connected to the Pico headset and OAK-D cameras is the capture host; a laptop receiving the WebSocket stream is the client. A host-monotonic timestamp comes from the capture computer's elapsed-time clock. It is not a calendar date and cannot be directly subtracted from a different computer's clock.

A session is one lifetime of a fixed capture configuration, from Begin through SessionStopped. A stream is a sequence of samples of one advertised format: stream 0 contains tracking, and each camera has its own image stream. A descriptor is the metadata that tells a client how to interpret one stream's bytes. The owner is the one connected client allowed to configure the server; ownership lasts until that WebSocket disconnects, including between sessions.

The client connects to a supplied server WebSocket URL, which may identify a station anywhere on the network. Server address, port and transport deployment are not Begin fields. The protocol requires no shared filesystem, local hardware tools, process access or configuration access on the client. One owner connection is accepted, including while idle. A second connection closes with 1013. Compression is disabled. Standard WebSocket Ping/Pong is transport keepalive.

One JSON object is one text message. One complete header and payload is one binary message. There is no per-frame acknowledgement, base64, pickle or extra length prefix.

| Direction | Discriminator | Message |
|---|---|---|
| Client → server | 1 | Begin |
| Client → server | 2 | Stop |
| Server → client | 9 | SessionInit |
| Server → client | 3 | SessionStarting |
| Server → client | 4 | SessionBegun |
| Server → client | 5 | SessionStatus |
| Server → client | 6 | SessionStopped |
| Server → client | 10 | RequestRejected |
| Server → client | BinaryKind 1 | CaptureFrame |
| Server → client | BinaryKind 2 | CaptureImage |

There are ten message families. SessionInit describes the resolved session configuration and complete decoding layout before hardware initialization. SessionStarting reports progress and observed availability. SessionStatus carries IC's changing provider/device state without treating absence as terminal or confusing status with pose validity. No application recording command or file path is accepted.

Every JSON message has these envelope fields. An envelope is the small common part of a message that allows the receiver to route it before reading message-specific content.

| Field | Allowed contents | What it does |
|---|---|---|
| `type` | `MessageType`: `BEGIN=1`, `STOP=2`, `SESSION_STARTING=3`, `SESSION_BEGUN=4`, `SESSION_STATUS=5`, `SESSION_STOPPED=6`, `SESSION_INIT=9`, `REQUEST_REJECTED=10`. Binary families use the header's BinaryKind instead. | Selects the parser and action. Each message accepts only its own type value. The human-readable names in this document are explanations, not wire values. |
| `version` | Not an enum: exactly integer 1; not boolean true. | Identifies the protocol contract used to interpret all remaining fields. A receiver must reject another version rather than guess its meaning. |
| `session_id` | Nonempty string, at most 128 characters; a new UUID string is recommended for each new session. | Associates configuration, status, samples and terminal replies with one session. A reply for another ID must not reset the active session. This is a client-chosen correlation ID, not a hardware identifier. |

#### Begin: configure sources, plugins and cameras

```json
{
  "type": 1,
  "version": 1,
  "session_id": "7bd8ec63-c561-4b50-a8ca-58f38000f53f",
  "tracking": {
    "sources": [1, 2, 3],
    "rate_hz": 60,
    "headset": {"device_index": 0}
  },
  "plugins": [
    {"plugin": 1, "enabled": true, "required": true, "datasets": [1]}
  ],
  "cameras": []
}
```

`tracking.sources` is a unique array of SourceType integers. Omitted source types are not constructed or polled. Controllers and hands each select both left/right native outputs; they do not select serial numbers or a hardware vendor. An empty list permits camera-only capture. `rate_hz` defaults to 60 and must be finite, positive and at most 120. It controls polling; it does not configure headset exposure or guarantee fresh hardware at that frequency.

`headset` is an optional object controlling how the capture computer connects its headset to CloudXR. CloudXR supplies the OpenXR runtime used by these IC sources. Omission uses the server's configured defaults. These settings do not choose individual controllers or gloves.

Starting that connection means waiting for the OpenXR session to become usable. CaptureServer attaches to a CloudXR service that the station operator starts separately (for example with the SDK's out-of-band setup over ADB and a USB cable); starting the service and connecting the headset are the station's own responsibility, and a missing service fails the session with `IC_ERROR`. The station owns all runtime compatibility settings. The client selects a headset by index and observes connection progress; it never supplies a runtime profile, executable, device-tool setting or server configuration file.

| Begin field | Allowed contents/default | Meaning and effect |
|---|---|---|
| `tracking` | Required object. | Describes the IC output channels the client wants. Camera-only clients still supply this object with `sources:[]`. |
| `tracking.sources` | `SourceType`: required unique array drawn from `HEAD=1`, `CONTROLLERS=2`, `HANDS=3`; `[]` is allowed. | Constructs the corresponding IC sources. `[1,2]` requests head plus both controllers without requesting hand samples. `[1,2,3]` also requests both hands. This selects data channels, not particular physical device serial numbers. An empty array together with any enabled `plugins` entry is rejected as `INVALID_CONFIG`; see the `plugins` row below. |
| `tracking.rate_hz` | Not an enum: finite number greater than 0 and at most 120; default 60. | Requests how often the server polls the selected IC sources. At 60 Hz it attempts one combined tracking snapshot about every 16.7 ms. The device can update more slowly or temporarily provide no usable data. |
| `tracking.headset` | Object; omitted means device\_index=0. | Contains the remote headset selector below. A supplied non-object such as false or an array is invalid. |
| `tracking.headset.device_index` | Not an enum: integer 0 only; default 0. Reject booleans, null, negative integers and all other values. | Selects the first headset connected to the server. The index is server-local; the client supplies neither a device serial number nor connection-tool settings. Only the first headset is supported. |
| `plugins` | Array of plugin requests; default `[]`. | Requests supporting IC plugin processes. Without a Manus request, the server does not launch Manus merely because HANDS was selected. Any enabled entry here together with an empty `tracking.sources` is rejected as `INVALID_CONFIG`: an enabled plugin needs IC actually entered for discovery to run, but the real IC graph-construction API rejects an empty output mapping outright. |
| `cameras` | Array of 0–3 camera requests; default `[]`. | Selects OAK-D image streams separately from IC tracking. |

Headset selection and connection belong to the station, not the client. The station's separately
started CloudXR service performs the headset, out-of-band and ADB setup; CaptureServer samples the
runtime, accepts only `device_index=0`, and reports the headset's actual status in each snapshot.
The client does not enumerate devices, access ADB, supply a serial or configure a connection
transport. An absent CloudXR service fails the session with `IC_ERROR`; waiting progress while a
headset joins is shown through device status. An index is not a persistent hardware identity: a
later session's first device can be different.

`plugins` defaults to an empty array. Each entry selects a defined PluginType and supported output settings. A plugin kind occurs at most once. The server assigns IC's internal instance namespace and returns observed provider IDs as opaque identifiers; clients do not choose or construct those IDs. `enabled` and `required` retain the native PluginConfig semantics. The station owns plugin installation, execution and calibration defaults. The plugin list is independent of the tracking source list: enabling an injector does not subscribe its data automatically.

A plugin is a supporting program started by IC, rather than a physical device. Manus's plugin receives glove data and injects hand skeletons into OpenXR, where HandsSource can read them. An instance is one configured launch of that program; its ID gives its status records a namespace so records from different providers do not collide.

| Plugin request field | Allowed contents/default | Meaning and effect |
|---|---|---|
| `plugin` | `PluginType`: required, currently only `MANUS=1`. | Selects the server's installed Manus adapter. It is not an executable name or path supplied by the client. |
| `enabled` | Boolean; default true. | True permits launching the requested plugin; false skips it. A disabled entry does not imply its hardware is absent. |
| `required` | Boolean; default false, matching native PluginConfig. | True makes missing plugin discovery a startup error. False permits IC to skip a plugin it cannot find. Neither value makes physical glove connection a prerequisite for SessionBegun. |
| `datasets` | `ManusDataset`: unique nonempty array, currently `HUMAN=1` only, so `[1]`; default `[1]`. | HUMAN enables the 26-joint hand-skeleton injection path. A dataset here is a category of live plugin output, not a recorded training dataset or filesystem directory. |

HANDS selects generic OpenXR hand outputs. A client requiring Manus hardware checks Manus device status. HandsSource does not provide per-sample provider identity or an exclusive Manus filter, so a valid generic hand sample does not certify Manus origin. Recording requirements remain client policy.

`cameras` defaults to an empty array and contains up to three requests:

```json
{
  "name": "camera_left",
  "type": 1
}
```

Each camera request contains exactly name and type. The name selects a logical camera configured on the remote station, such as camera\_left. The server owns that name's mapping to hardware and its capture settings. The client supplies no device index, serial, local path, dimensions, frame rate, codec, encoder quality or required flag. Names have no built-in anatomical meaning; any agreed station-defined camera name is valid. Unknown names, duplicate names and type mismatches are rejected rather than silently selecting another camera. Two different configured names must not resolve to the same physical camera within a session.

SessionInit returns the server-selected dimensions, frame rate, codec, encoder quality and acquisition policy in that camera's stream descriptor. The client uses these returned settings to interpret images; it does not negotiate them through Begin. Camera settings are station-specific. Live OAK-D cameras encode JPEG on-device. No stereo/depth/IMU, H.264 or inter-frame data is included.

Camera required is a server-selected acquisition policy returned in SessionInit: failure to open a required camera fails startup; an optional missing camera retains its descriptor and can report absence/recovery. The client does not set this policy in Begin. It is distinct from IC PluginConfig.required and does not introduce a tracking-readiness policy.

| Camera request field | Allowed contents | Meaning and effect |
|---|---|---|
| `name` | Required nonempty unique string matching a station-configured logical camera, such as `camera_left`. | Selects which named camera the server should capture. It is not an arbitrary label to assign to whichever hardware is found first. The client knows the logical name without needing access to the server's hardware tools or configuration files. |
| `type` | `CameraType`: required, currently `OAK_D_COLOR=1`. | Selects and checks the camera acquisition family expected for that name. It does not enable depth or stereo outputs. This nested type is a CameraType; the message envelope's type is separately a MessageType. |

Reject unknown keys and unsupported enum values. Structurally valid unsupported settings are configuration rejections, distinct from malformed envelopes. All session configuration is fixed until Stop; a changed configuration needs a new session ID.

#### SessionInit: announce layout and expected outputs

After validating the request and resolving defaults, the server initiates IC startup and sends SessionInit before native capture-session entry or hardware acquisition. For camera-only requests, it initiates camera startup instead. SessionInit announces mode, effective\_config, clock and the complete stream descriptors once. It confirms that startup has been initiated without certifying connected hardware, completed native initialization or successful acquisition. The Python client exposes this notification as on\_started(CaptureStarted). Invalid or busy requests receive RequestRejected without a SessionInit. Once SessionInit accepts the session, startup failure ends it with SessionStopped.

A successful session follows Begin → SessionInit → SessionStarting (one or more progress updates) → SessionBegun → binary samples and SessionStatus → SessionStopped. Startup failure instead ends with SessionStopped and no binary samples. SessionInit is server-to-client metadata, not another client command or hardware polling operation.

The schema and configuration determine decoding before hardware connects. Tracking includes slots for head, both controllers and both hands; sources identifies selected channels. Camera names resolve to server-configured formats; SessionInit announces their codec and dimensions before opening hardware. These are expected outputs, not a confirmed connected-device inventory. A configured camera that is absent still has a known intended format; hardware discovery must not silently change it.

The optimistic aspect is the attempt to start those outputs. Keep stream IDs, fields, dtypes, shapes, offsets and payload sizes fixed regardless of later physical-device availability. An absent tracking group keeps its slots with present=0 and validity flags=0; its numeric placeholders must not be used as measurements. Partial hand validity retains the hand group with per-joint validity. An absent optional camera retains a descriptor but emits no JPEG until recovery.

Native initialization errors can still fail startup, including a missing required plugin, an unsupported required IC extension or a required camera failure. A disconnected glove reported by a successfully running plugin does not itself fail the IC session. An optimistic layout must not hide initialization exceptions or silently substitute another acquisition format.

This is a complete SessionInit for head, controllers and hands:

```json
{
  "type": 9,
  "version": 1,
  "session_id": "session-1",
  "mode": 1,
  "effective_config": {
    "tracking": {
      "sources": [
        1,
        2,
        3
      ],
      "rate_hz": 60,
      "headset": {
        "device_index": 0
      }
    },
    "plugins": [
      {
        "plugin": 1,
        "enabled": true,
        "required": true,
        "datasets": [
          1
        ]
      }
    ],
    "cameras": []
  },
  "clock": {
    "domain": 1,
    "unit": 1,
    "sample_time_semantics": 1
  },
  "streams": [
    {
      "stream_id": 0,
      "kind": 1,
      "name": "tracking",
      "schema": 1,
      "sources": [
        1,
        2,
        3
      ],
      "rate_hz": 60,
      "pose_representation": 1,
      "quaternion_order": 1,
      "position_unit": 1,
      "axes": 1,
      "reference_space": "session_floor",
      "hand_joint_order": 1,
      "layout": {
        "payload_size": 1995,
        "fields": [
          {"name":"head.present","dtype":1,"shape":[],"offset":0},
          {"name":"head.position","dtype":2,"shape":[3],"offset":1},
          {"name":"head.orientation","dtype":2,"shape":[4],"offset":13},
          {"name":"head.is_valid","dtype":1,"shape":[],"offset":29},
          {"name":"head.is_tracked","dtype":1,"shape":[],"offset":30},
          {"name":"controller_left.present","dtype":1,"shape":[],"offset":31},
          {"name":"controller_left.grip_position","dtype":2,"shape":[3],"offset":32},
          {"name":"controller_left.grip_orientation","dtype":2,"shape":[4],"offset":44},
          {"name":"controller_left.grip_is_valid","dtype":1,"shape":[],"offset":60},
          {"name":"controller_left.aim_position","dtype":2,"shape":[3],"offset":61},
          {"name":"controller_left.aim_orientation","dtype":2,"shape":[4],"offset":73},
          {"name":"controller_left.aim_is_valid","dtype":1,"shape":[],"offset":89},
          {"name":"controller_left.primary_click","dtype":3,"shape":[],"offset":90},
          {"name":"controller_left.secondary_click","dtype":3,"shape":[],"offset":98},
          {"name":"controller_left.thumbstick_x","dtype":3,"shape":[],"offset":106},
          {"name":"controller_left.thumbstick_y","dtype":3,"shape":[],"offset":114},
          {"name":"controller_left.thumbstick_click","dtype":3,"shape":[],"offset":122},
          {"name":"controller_left.menu_click","dtype":3,"shape":[],"offset":130},
          {"name":"controller_left.squeeze_value","dtype":3,"shape":[],"offset":138},
          {"name":"controller_left.trigger_value","dtype":3,"shape":[],"offset":146},
          {"name":"controller_right.present","dtype":1,"shape":[],"offset":154},
          {"name":"controller_right.grip_position","dtype":2,"shape":[3],"offset":155},
          {"name":"controller_right.grip_orientation","dtype":2,"shape":[4],"offset":167},
          {"name":"controller_right.grip_is_valid","dtype":1,"shape":[],"offset":183},
          {"name":"controller_right.aim_position","dtype":2,"shape":[3],"offset":184},
          {"name":"controller_right.aim_orientation","dtype":2,"shape":[4],"offset":196},
          {"name":"controller_right.aim_is_valid","dtype":1,"shape":[],"offset":212},
          {"name":"controller_right.primary_click","dtype":3,"shape":[],"offset":213},
          {"name":"controller_right.secondary_click","dtype":3,"shape":[],"offset":221},
          {"name":"controller_right.thumbstick_x","dtype":3,"shape":[],"offset":229},
          {"name":"controller_right.thumbstick_y","dtype":3,"shape":[],"offset":237},
          {"name":"controller_right.thumbstick_click","dtype":3,"shape":[],"offset":245},
          {"name":"controller_right.menu_click","dtype":3,"shape":[],"offset":253},
          {"name":"controller_right.squeeze_value","dtype":3,"shape":[],"offset":261},
          {"name":"controller_right.trigger_value","dtype":3,"shape":[],"offset":269},
          {"name":"hand_left.present","dtype":1,"shape":[],"offset":277},
          {"name":"hand_left.joint_positions","dtype":2,"shape":[26,3],"offset":278},
          {"name":"hand_left.joint_orientations","dtype":2,"shape":[26,4],"offset":590},
          {"name":"hand_left.joint_radii","dtype":2,"shape":[26],"offset":1006},
          {"name":"hand_left.joint_valid","dtype":1,"shape":[26],"offset":1110},
          {"name":"hand_right.present","dtype":1,"shape":[],"offset":1136},
          {"name":"hand_right.joint_positions","dtype":2,"shape":[26,3],"offset":1137},
          {"name":"hand_right.joint_orientations","dtype":2,"shape":[26,4],"offset":1449},
          {"name":"hand_right.joint_radii","dtype":2,"shape":[26],"offset":1865},
          {"name":"hand_right.joint_valid","dtype":1,"shape":[26],"offset":1969}
        ]
      }
    }
  ]
}
```

| SessionInit field | Python enum and accepted values, or ordinary type | Meaning |
|---|---|---|
| `mode` | `CaptureMode`: `MOCK=0`, `LIVE=1`. | Identifies whether samples come from synthetic generators or the live IC/camera adapters. It is chosen when launching the server, not by a remote Begin request. |
| `effective_config` | Object with the resolved `tracking`, `plugins` and `cameras` fields described above. | Tells the client which configuration was accepted, including default values. It is configuration, not evidence that every requested physical device is connected. |
| `clock` | Object containing the next three fields. | Explains how to interpret binary-header timestamps. |
| `clock.domain` | `ClockDomain`: `CAPTURE_HOST_MONOTONIC=1`. | Times use the capture computer's elapsed-time clock; they do not use Unix/calendar time or the client's elapsed-time clock. |
| `clock.unit` | `TimeUnit`: `NANOSECONDS=1`. | An interval of 1,000,000,000 timestamp units means one second. |
| `clock.sample_time_semantics` | `SampleTimeSemantics`: `HOST_POLL=1`. | The tracking timestamp describes the host's source-sampling step. It does not claim to be a sensor exposure time or an SDK callback timestamp. |
| `streams` | Array of tracking and camera descriptor objects. | Defines the stream IDs, formats and tracking byte layout the client must register from SessionInit before any binary messages. An empty array is valid only when no streams were requested. |

The client validates and caches this decoding contract. A layout/format change requires a new session. Unknown camera intrinsics can initially be null; later measured intrinsics are metadata, not a byte-layout change.

#### SessionStarting: report startup progress and observations

Send at least one SessionStarting after SessionInit and before SessionBegun, including when startup completes immediately:

```json
{"type":3,"version":1,"session_id":"session-1","stage":1,"message":"Initializing capture sources","status":null}
```

Send progress on stage changes and approximately once per second while a stage persists. Stage changes must follow actual headset/source initialization evidence. Startup must be cancellable.

| SessionStarting field | Python enum and accepted values, or ordinary type | Meaning |
|---|---|---|
| `stage` | `StartupStage`: `INITIALIZING=1`, `WAITING_FOR_HEADSET=2`, `CONNECTING=3`, `STARTING_SOURCES=4`. | INITIALIZING prepares the request/resources; WAITING\_FOR\_HEADSET means the required headset connection has not arrived; CONNECTING means connection setup is in progress; STARTING\_SOURCES means the IC trackers/plugins or cameras are being initialized. A stage is progress information, not a guarantee of physical device readiness. |
| `message` | Nonempty string; no enum. | Gives a readable explanation of the current wait or operation. Clients display it but use stage for program decisions. |
| `status` | Latest complete observed status object, or null if none is available yet. | Reports the current evidence about providers, devices and cameras during startup. INITIALIZING and UNKNOWN are legitimate answers; declaring a device in inventory is not evidence that it is connected. |

SessionStarting does not repeat or negotiate the layout. SessionInit describes expected outputs; SessionStarting.status reports observations and can remain INITIALIZING or UNKNOWN until IC has evidence. No binary sample is sent before SessionBegun.

IC context entry builds INITIALIZING inventory. After successful entry, providers become AVAILABLE, the owned OpenXR headset is CONNECTED / OPENXR\_SESSION\_READY, and Manus devices remain UNKNOWN / NO\_REPORT until an actual plugin report is read. The first step updates DeviceIO and refreshes the reports, but can still return no glove report. Preserve that uncertainty. get\_status reads a cache rather than polling hardware, so progress must retain the cached observation timestamp. The server completes its initial step during startup, then sends SessionBegun without waiting for every selected device to connect.

#### SessionBegun: confirm acquisition

SessionBegun confirms initialized session resources and a successful first IC step when tracking is requested. For camera-only capture, it confirms initialization according to the declared required/optional camera policies. It does not certify that every device is physically connected or every pose is valid. Missing gloves, inactive controllers or untracked hands do not by themselves prevent SessionBegun. Tracker creation errors, unsupported required OpenXR extensions, a missing required plugin or a required camera startup failure prevent establishment.

```json
{
  "type": 4,
  "version": 1,
  "session_id": "session-1",
  "status": {
    "updated_at_ns": 123000,
    "providers": [],
    "devices": [],
    "cameras": []
  }
}
```

The empty arrays illustrate the status envelope; real replies contain the current observed inventory. Do not replace observed devices with empty arrays or treat an empty array as proof of hardware absence. SessionBegun does not repeat or change mode, configuration, clock or descriptors; those were established by SessionInit. It authorizes delivery of samples in that format.

| SessionBegun field | Python enum and accepted values, or ordinary type | Meaning |
|---|---|---|
| `status` | Latest complete observed status object. | Confirms acquisition establishment with the current IC/camera observations. It may still contain UNKNOWN/NO\_REPORT. No new decoding descriptor is negotiated here. |

#### SessionStatus: report an observation change

One status schema is reused across the lifecycle: SessionStarting carries startup observations, SessionBegun carries establishment observations, SessionStatus carries subsequent changes, and SessionStopped carries the final available observation. Snapshots are complete, not deltas. After SessionBegun, the server sends SessionStatus when inventory, state, reason, diagnostic or camera metadata changes; a changed updated\_at\_ns alone does not trigger another message. WebSocket Ping/Pong supplies transport keepalive; tracking/image arrival and progression reveal source activity. The snapshot timestamp is observation time on the capture computer, never socket-send time.

| SessionStatus field | Python enum and accepted values, or ordinary type | Meaning |
|---|---|---|
| `status` | Required complete StatusSnapshot object. | Replaces the client's cached provider/device/camera observation for this session. Missing devices in the new inventory are not retained as stale entries from the old snapshot. |

```json
{
  "type":5,
  "version":1,
  "session_id":"session-1",
  "status":{
    "updated_at_ns":124000,
    "providers":[
      {"id":"provider-1","type":1,"plugin":1,"name":"manus_hand_plugin","status":2,"reason":0,"error":"","pid":1234,"exit_code":null,"term_signal":null}
    ],
    "devices":[
      {"id":"device-1","path":"/hand/left","type":"hand","description":"Left hand","provider_id":"provider-1","status":4,"reason":8,"error":""},
      {"id":"device-2","path":"/hand/right","type":"hand","description":"Right hand","provider_id":"provider-1","status":4,"reason":8,"error":""}
    ],
    "cameras":[]
  }
}
```

The provider/device state, paths, descriptions and diagnostics reflect native ProviderStatus and DeviceStatus observations. Convert type/state/reason enum members through the explicit wire tables. The server assigns opaque wire provider/device IDs and maintains the mapping to native IC IDs; provider\_id uses those same wire IDs. Add provider.plugin from the server's plugin association so a remote client can recognize Manus without parsing an internal namespace or a display name. Preserve UNKNOWN, DISABLED and DEGRADED; never reduce all states to a connected boolean. Process diagnostics are optional, nullable observations; clients need no process access and must not require them for ordinary capture.

A provider is the software component supplying devices, such as the OpenXR runtime or a Manus plugin process. A device is a particular inventory entry belonging to that provider. A snapshot is a complete observation at one time: the next snapshot replaces the previous one rather than listing only changes.

| Status field | Python enum and accepted values, or ordinary type | Meaning |
|---|---|---|
| `updated_at_ns` | Not an enum: nonnegative integer nanoseconds. | The time IC observed/refreshed this snapshot on the capture computer. It is distinct from when the client received it. |
| `providers` | Array of provider records. | Lists software providers in the native monitored inventory. A running provider does not imply all its devices are connected. |
| `devices` | Array of device records. | Lists devices declared by monitored providers, including devices currently disconnected or unknown. |
| `cameras` | Array of server-owned camera status records. | Reports OAK-D acquisition state separately from IC's own inventory. |
| Provider `id` | Nonempty opaque string; no enum. | Stable within the session and used by device.provider\_id to associate a device with its provider. Assigned by the server; clients do not construct or parse it. |
| Provider `type` | `ProviderType`: `PLUGIN=1`, `OPENXR_RUNTIME=2`. | Distinguishes a plugin process from the OpenXR runtime. |
| Provider `plugin` | `PluginType`: `MANUS=1` for the supported Manus plugin; null for an OpenXR runtime provider. | Identifies the plugin kind without requiring the client to understand the server's internal names. An available Manus provider still does not prove the origin of a generic hand sample. |
| Provider `name` | String; no enum. | Human-readable name supplied by native provider metadata. |
| Provider `status` | `ProviderState`: `INITIALIZING=1`, `AVAILABLE=2`, `FAILED=3`, `STOPPED=4`. | INITIALIZING is being prepared; AVAILABLE means the provider is operational, not that its hardware is present; FAILED means it encountered a reported failure; STOPPED means its lifetime ended. |
| Provider/device/camera `reason` | `StatusReason`: `NONE=0`, `PROVIDER_INITIALIZING=1`, `PROVIDER_FAILED=2`, `PROVIDER_STOPPED=3`, `NO_REPORT=4`, `STALE_REPORT=5`, `MALFORMED_REPORT=6`, `UNSUPPORTED_SCHEMA_VERSION=7`, `NO_HARDWARE_SIGNAL=8`, `CALIBRATION_FAILED=9`, `NO_CURRENT_DATA=10`, `DEVICE_ERROR=11`, `DISABLED_BY_CONFIGURATION=12`, `PROCESS_EXITED=13`, `PROCESS_SIGNALED=14`, `PROCESS_OBSERVATION_ERROR=15`, `STARTUP_FAILED=16`, `RUNTIME_UPDATE_FAILED=17`, `OPENXR_SESSION_READY=18`, `EXTERNAL_OPENXR_HEALTH_UNKNOWN=19`, `OPENXR_FORM_FACTOR_UNAVAILABLE=20`, `OPENXR_SESSION_LOST=21`, `OPENXR_INSTANCE_LOST=22`, `OPENXR_POLL_ERROR=23`. | Explains the state using a stable machine-readable code. The reason meanings are expanded below; an error string supplements the code without replacing it. |
| Provider/device/camera `error` | String, possibly empty; no enum. | Native or adapter diagnostic text for display/logging. Do not parse it to decide whether a device is connected. |
| Provider `pid` | Optional diagnostic; not an enum: positive process ID integer or null. | Identifies an observed server-side process. A client may display it but never needs local process access; omission and null both mean unavailable information. |
| Provider `exit_code` | Optional diagnostic; not an enum: integer or null. | Reports an observed process exit status. Omission/null means unavailable information, not success. |
| Provider `term_signal` | Optional diagnostic; not an enum: positive signal-number integer or null. | Identifies an operating-system signal that terminated the process, when observed. A client need not interpret OS signal numbers to use provider.status and reason. |
| Device `id` | Nonempty opaque string; no enum. | Server-assigned identifier for one observed device across snapshots in a session. Its contents do not encode a contractually parseable plugin root or hardware serial. |
| Device `path` | String; no enum. | Provider-relative device identifier such as `/hand/left`. This is a logical device path, not a file to open on the client. |
| Device `type` | String; no protocol enum. | Preserves the plugin manifest's device-category metadata. Unlike provider.type, this is an externally defined label and is not used as a protocol discriminator. |
| Device `description` | String; no enum. | Human-readable native device description. |
| Device `provider_id` | String matching a provider record's id; no enum. | Associates the device with the software provider responsible for its status. This association does not identify the provider of a generic HandInput sample. |
| Device/camera `status` | `DeviceState`: `INITIALIZING=1`, `UNKNOWN=2`, `CONNECTED=3`, `DISCONNECTED=4`, `DEGRADED=5`, `FAILED=6`, `UNAVAILABLE=7`, `DISABLED=8`. | Reports the observed device state. Detailed meanings follow below. |
| Camera `name` | Nonempty string matching a requested logical camera; no enum. | Associates the observation with the camera selected in Begin. No device index, hardware serial or catalogue lookup is required. |
| Camera `stream_id` | Not an enum: integer 1–3 matching the stream assigned in SessionInit. | Associates the observation with its camera descriptor and subsequent JPEG messages. Assignment does not certify successful camera startup. |
| Camera `intrinsics` | Optional null or a finite numeric array of shape \[3,3\]; no enum. | Supplies a known camera projection matrix for the transmitted dimensions. Omission/null means no calibration observation is supplied. Updating it changes metadata, not the image byte format. |

| DeviceState | Interpretation |
|---|---|
| `INITIALIZING=1` | The device's provider/session is still being prepared; no conclusion about current usable data follows. |
| `UNKNOWN=2` | The monitor lacks a trustworthy current report. This is uncertainty, not a declaration that the device is absent. |
| `CONNECTED=3` | The provider reports the device connected according to its own rules. Still check sample validity and whether receipt/sample progression has stalled. On a remote connection those checks do not establish absolute sample age. |
| `DISCONNECTED=4` | The provider reports no connected hardware signal for this declared device. |
| `DEGRADED=5` | Hardware may be present, but calibration, current data or another requirement is impaired. |
| `FAILED=6` | A device-specific failure is reported. |
| `UNAVAILABLE=7` | The device cannot currently be supplied, for example because its provider failed or stopped. |
| `DISABLED=8` | Configuration intentionally disabled this output; it is not evidence of missing hardware. |

StatusReason NONE means no additional failure/wait reason. PROVIDER\_INITIALIZING, PROVIDER\_FAILED and PROVIDER\_STOPPED attribute device state to the owning provider. NO\_REPORT means no status report arrived; STALE\_REPORT means the last report is too old; MALFORMED\_REPORT means its contents could not be interpreted; UNSUPPORTED\_SCHEMA\_VERSION means the monitor cannot interpret that report version. NO\_HARDWARE\_SIGNAL means the provider sees no hardware connection. CALIBRATION\_FAILED means calibration failed; NO\_CURRENT\_DATA means the device has no current usable data; DEVICE\_ERROR describes another device-specific fault. DISABLED\_BY\_CONFIGURATION explains an intentional disable.

PROCESS\_EXITED means an observed process ended; PROCESS\_SIGNALED means it was terminated by an operating-system signal; PROCESS\_OBSERVATION\_ERROR means process health could not be determined. STARTUP\_FAILED means initialization failed; RUNTIME\_UPDATE\_FAILED means an active update raised. OPENXR\_SESSION\_READY means the OpenXR session is ready, not that every pose is tracked; EXTERNAL\_OPENXR\_HEALTH\_UNKNOWN means IC does not own enough of an externally supplied session to certify its health. OPENXR\_FORM\_FACTOR\_UNAVAILABLE means the requested headset system is unavailable. OPENXR\_SESSION\_LOST and OPENXR\_INSTANCE\_LOST mean the corresponding OpenXR resource was lost. OPENXR\_POLL\_ERROR means observing OpenXR state failed. Preserve native state/reason combinations rather than inventing a new combination from these descriptions.

OAK-D status is separate from native IC inventory: cameras entries contain name, stream\_id, DeviceState status, StatusReason reason, display error and optional intrinsics (null or a finite 3×3 camera matrix as explained below). These are server-owned camera observations using the same normalized wire states; do not present them as IC get\_status() output. An absent optional camera emits no image until recovered. An announced stream without a current observation remains unknown; a stream descriptor is not a connected-state assertion. A later measured intrinsics value is camera metadata, not a byte-layout change.

A missing optional plugin may be skipped by IC and therefore have no inventory entry. Effective config records the request; absence from the observed inventory is not a synthesized native DISCONNECTED record. A missing required plugin fails establishment. When native startup fails, retain the last available status snapshot in the terminal reply.

#### Where StatusReason appears

StatusReason explains an observed provider/device/camera state; it is not the session's terminal StopReason. Its wire field is named reason inside each record of a status snapshot:

| Field path | Python enum and accepted values | Purpose |
|---|---|---|
| `status.providers[i].reason` | `StatusReason`: the explicitly assigned members 0–23 in the enum table. | Explains provider lifecycle/health, such as STARTUP\_FAILED=16, PROCESS\_EXITED=13 or OPENXR\_POLL\_ERROR=23. |
| `status.devices[i].reason` | `StatusReason`: the same members 0–23. | Explains a device condition, such as NO\_HARDWARE\_SIGNAL=8, NO\_CURRENT\_DATA=10 or PROVIDER\_FAILED=2. |
| `status.cameras[i].reason` | `StatusReason`: the same members 0–23. | Explains server-observed camera state using the same meanings, for example DEVICE\_ERROR=11. These camera observations are not claimed as native IC output. |

These nested reason fields use StatusReason; top-level SessionStopped.reason instead uses StopReason. For example, a device record with status=4 and reason=8 means DeviceState.DISCONNECTED and StatusReason.NO\_HARDWARE\_SIGNAL; its error string explains the observation in words. That condition alone does not end the session.

Native IC StatusReason names and meanings map to wire integers 0–23. Native Python values are strings; protocol values are explicit integers. Preserve different provider/device reasons: a provider can report STARTUP\_FAILED while its device reports UNAVAILABLE / PROVIDER\_FAILED. Do not infer typed reasons by parsing diagnostic text or force an OS process-exit code into StatusReason.

IC\_EXITED is a server-owned outcome for unexpected end of the acquisition worker; the synchronous adapter is a thread, not a separate IC process. It is distinct from a plugin's PROCESS\_EXITED observation and from a caught IC exception. Camera failures use CAMERA\_ERROR only when they end the session. Optional native process exit\_code/term\_signal remain diagnostics, not enum members.

#### Tracking descriptor and payload

One combined stream, ID 0, carries the selected public IC outputs in a stable layout. Unselected slots remain absent placeholders; they are not fabricated IC samples. No tracking stream is advertised for camera-only capture.

The descriptor is JSON metadata inside SessionInit.streams. The tracking sample payload is a binary blob inside each CaptureFrame. Do not send the sample as JSON numbers, repeat the descriptor per frame, or require the client to infer offsets from Python object layout. The descriptor supplies every field's name, numeric dtype, shape and byte offset, plus the total payload size, before any binary message arrives. The Python client validates and caches that descriptor from SessionInit, exposes it through on_started, and, after SessionBegun, delivers subsequent blobs unchanged. A downstream consumer uses the descriptor to decode values when needed. A new session establishes a new descriptor; the layout never changes within a session.

The SessionInit example above contains the complete tracking descriptor and field layout. The field explanations below apply to that streams entry.

Dotted field names preserve the public IC group/field identity. Dtype codes are WireDType values. Arrays use C order. All multibyte values are explicitly little-endian, with no implicit padding; never use host-native NumPy byte order. Float32 IC arrays remain float32; controller Python scalar floats remain float64. This is serialization, not an in-memory TensorGroup dump.

| Tracking descriptor field | Python enum and accepted values, or ordinary type | Meaning |
|---|---|---|
| `stream_id` | Not an enum: integer 0 for tracking. | Associates a binary message's header with this descriptor. |
| `kind` | `StreamKind`: `TRACKING=1` for this descriptor; `IMAGE=2` for camera descriptors. | Selects the kind of decoder to use for the stream. |
| `name` | String, `tracking` for this stream; no enum. | Provides a readable stream label. Routing uses stream\_id, not this label. |
| `schema` | `TrackingSchema`: `IC_TRACKING_V1=1`. | Identifies the complete field contract, including presence/validity meanings and the 1995-byte layout. A client rejects an unsupported schema rather than interpreting a similar-looking blob. |
| `sources` | `SourceType`: unique array drawn from `HEAD=1`, `CONTROLLERS=2`, `HANDS=3`. | Records the source channels selected for this tracking stream. Unselected channels retain inert slots in the fixed binary layout. |
| `rate_hz` | Not an enum: finite number greater than 0 and at most 120. | Advertises the accepted polling rate. It does not certify that every device generated a new sample at that rate. |
| `pose_representation` | `PoseRepresentation`: `POSITION_QUATERNION=1`. | Each pose uses a 3-number position and a 4-number orientation, rather than a 4×4 matrix or three rotation angles. |
| `quaternion_order` | `QuaternionOrder`: `XYZW=1`. | Gives the order of orientation components: x, y, z, then w. A quaternion represents rotation; it is not a position or Euler-angle tuple. |
| `position_unit` | `PositionUnit`: `METERS=1`. | Defines the scale of positions and hand radii; a position difference of 1 means one meter. |
| `axes` | `Axes`: `RIGHT_HANDED_Y_UP_MINUS_Z_FORWARD=1`. | Defines coordinate directions: positive Y is up and negative Z is forward, using a right-handed coordinate system. This tells the client how to interpret motion and avoid swapping or reflecting axes accidentally. |
| `reference_space` | Nonempty string identifying the session coordinate frame; no enum. | Identifies the common origin/frame in which the positions and orientations are expressed. For example, session\_floor denotes a floor-based session space. It does not locate the origin in a surveyed building coordinate system or guarantee the same origin after reconnect. |
| `hand_joint_order` | `HandJointOrder`: `OPENXR_26=1`. | Defines which anatomical joint each array index means. Both hands use the same 26-index order described below. |
| `layout.payload_size` | Not an enum: integer 1995 for this schema. | Gives the exact binary sample size excluding the 32-byte header. The decoder rejects a different size. |
| `layout.fields` | Array of field records. | Gives the byte-level recipe for extracting every named value from the blob. Field records contain the next four fields. |
| Field `name` | String identifying a declared schema field, such as head.position; no enum. | Names the extracted value using IC's output group and field identity. This is not an extra string stored inside each binary sample. |
| Field `dtype` | `WireDType`: `UINT8=1`, `FLOAT32_LE=2`, `FLOAT64_LE=3`. | Selects an unsigned 1-byte integer, a 4-byte floating-point value, or an 8-byte floating-point value. LE means least-significant byte first. Presence/validity use UINT8; positions/quaternions/radii use FLOAT32\_LE; controller input scalars use FLOAT64\_LE. |
| Field `shape` | Not an enum: array of positive integer dimensions, or `[]` for one scalar. | `[]` means one value, `[3]` means three values, and `[26,3]` means 26 rows of 3 values. C order means each row's values are adjacent before the next row begins. |
| Field `offset` | Not an enum: nonnegative integer byte offset within the 1995-byte payload. | Gives where the field starts, counted from the first sample byte after the header. It is not counted from the start of the whole WebSocket message. |

For example, head.position has dtype FLOAT32\_LE=2, shape \[3\] and offset 1. The client skips the 32-byte header, skips one payload byte, then reads three little-endian float32 values from payload bytes 1–12. The next field, head.orientation, starts at payload offset 13. In general, a field occupies the product of its dimensions times its dtype size; a scalar occupies one dtype-sized value. The descriptor must have no overlaps, out-of-bounds fields or missing required schema fields. Clients validate it once before accepting frames; they do not infer correctness merely because its declared total size is 1995.

Head, each controller and each hand has independent presence. For a selected source, absent OptionalTensorGroup maps to present=0; an existing group maps to present=1 and retains its native flags. For absent/unselected groups, zero numeric values and validity flags and use identity xyzw quaternions as inert placeholders. Presence, not placeholder content, determines absence.

Head presence does not imply tracking: IC explicitly permits a present readable head with is\_valid=true and is\_tracked=false. Each controller retains grip/aim and their independent pose validity, plus every input scalar. Each hand retains 26 OpenXR joints, radii and joint validity. OpenXR order is palm=0, wrist=1, four thumb joints=2–5, then five each for index=6–10, middle=11–15, ring=16–20 and little=21–25.

The following table explains the values packed into the binary fields. These are measurements or boolean flags, not enum fields. A flag uses UINT8 with accepted values 0=false and 1=true. Quaternions use xyzw order; an identity orientation is \[0,0,0,1\]. Validity flags govern whether numerical values may be used; a decoder must not infer validity from plausible-looking coordinates.

| Binary field or field family | Values and shape | Meaning |
|---|---|---|
| `head.present` | Scalar 0 or 1. | Indicates whether IC returned a HeadInput group for this step. 0 also covers an unselected head source. |
| `head.position` | Three float32 values, meters. | Position of the headset in the declared session reference space. |
| `head.orientation` | Four float32 values, xyzw quaternion. | Orientation of the headset in that same reference space. |
| `head.is_valid` | Scalar 0 or 1. | Whether the present head pose is readable according to IC. A readable placeholder can still be untracked. |
| `head.is_tracked` | Scalar 0 or 1. | Whether IC reports active head tracking. A client showing live head tracking needs presence, validity, this flag and sufficiently recent sample arrival. |
| `controller_left.present`, `controller_right.present` | Independent scalar 0 or 1 per side. | Indicates whether that side has a current ControllerInput group. One controller can be present while the other is absent. |
| Controller `grip_position`, `grip_orientation` | Three float32 position values and four float32 quaternion values per side. | The pose associated with holding the controller. A client attaching a virtual held object would normally consider this pose. |
| Controller `grip_is_valid` | Scalar 0 or 1 per side. | Governs use of the grip pose; it does not validate the aim pose. |
| Controller `aim_position`, `aim_orientation` | Three float32 position values and four float32 quaternion values per side. | The pose defining the controller's pointing ray. It can differ from the grip pose and is useful for pointing/selecting. |
| Controller `aim_is_valid` | Scalar 0 or 1 per side. | Governs use of the aim pose independently of grip validity. |
| Controller `primary_click`, `secondary_click` | Float64 scalars, normally 0.0 or 1.0 for released/pressed. | The runtime's primary and secondary button states. The generic server preserves these names rather than assigning device-specific A/B/X/Y labels. |
| Controller `thumbstick_x`, `thumbstick_y` | Float64 scalars, conventionally in \[-1,1\]. | Horizontal and vertical stick deflection, with 0 near the neutral position. Preserve the runtime's supplied sign and values. |
| Controller `thumbstick_click`, `menu_click` | Float64 scalars, normally 0.0 or 1.0. | Whether the stick button or menu button is pressed. Runtime/device support can differ; a zero alone does not certify that the device has that button. |
| Controller `squeeze_value`, `trigger_value` | Float64 scalars, conventionally in \[0,1\]. | Analog grip/squeeze and trigger activation. These are input measurements, not automatically interpreted recording commands. Preserve IC values without server-side clamping or application actions. |
| `hand_left.present`, `hand_right.present` | Independent scalar 0 or 1 per side. | Indicates whether IC returned that side's HandInput. This does not identify which physical hand-tracking device supplied it. |
| Hand `joint_positions` | Float32 array \[26,3\], meters. | Session-space position of each joint in the declared OpenXR order. These positions are not offsets relative to the wrist. |
| Hand `joint_orientations` | Float32 array \[26,4\], xyzw quaternions. | Session-space orientation of each joint, in the same index order as positions. |
| Hand `joint_radii` | Float32 array \[26\], meters. | Runtime-estimated radius associated with each joint, useful for visualizing or modeling its spatial extent. It is not a confidence score. |
| Hand `joint_valid` | UINT8 array \[26\], each value 0 or 1. | Whether each joint has a valid position and orientation according to IC. Some joints can be invalid while the hand group remains present. This is not a separate per-joint actively-tracked flag. |

Preserve session-space positions/quaternions. Do not select grip over aim, rename buttons, remove the palm, make hands wrist-relative, generate recorder matrices, or apply an application-specific alignment. The client performs any desired downstream conversion. Hand presence/validity does not provide provider identity, and joint validity is not a separate active-tracking flag.

##### Tracking field offsets

Each row is one field inside the 1995-byte tracking payload. Offset 0 starts immediately after the 32-byte binary header. Dtype is Python `WireDType`: `UINT8=1` (1 byte), `FLOAT32_LE=2` (4 bytes), or `FLOAT64_LE=3` (8 bytes). Empty shape `[]` means one scalar; all arrays use C order.

| Field name | Python enum member / wire dtype | Shape | Byte offset | Bytes |
|---|---|---|---|---|
| `head.present` | `WireDType.UINT8=1` | `[]` | 0 | 1 |
| `head.position` | `WireDType.FLOAT32_LE=2` | `[3]` | 1 | 12 |
| `head.orientation` | `WireDType.FLOAT32_LE=2` | `[4]` | 13 | 16 |
| `head.is_valid` | `WireDType.UINT8=1` | `[]` | 29 | 1 |
| `head.is_tracked` | `WireDType.UINT8=1` | `[]` | 30 | 1 |
| `controller_left.present` | `WireDType.UINT8=1` | `[]` | 31 | 1 |
| `controller_left.grip_position` | `WireDType.FLOAT32_LE=2` | `[3]` | 32 | 12 |
| `controller_left.grip_orientation` | `WireDType.FLOAT32_LE=2` | `[4]` | 44 | 16 |
| `controller_left.grip_is_valid` | `WireDType.UINT8=1` | `[]` | 60 | 1 |
| `controller_left.aim_position` | `WireDType.FLOAT32_LE=2` | `[3]` | 61 | 12 |
| `controller_left.aim_orientation` | `WireDType.FLOAT32_LE=2` | `[4]` | 73 | 16 |
| `controller_left.aim_is_valid` | `WireDType.UINT8=1` | `[]` | 89 | 1 |
| `controller_left.primary_click` | `WireDType.FLOAT64_LE=3` | `[]` | 90 | 8 |
| `controller_left.secondary_click` | `WireDType.FLOAT64_LE=3` | `[]` | 98 | 8 |
| `controller_left.thumbstick_x` | `WireDType.FLOAT64_LE=3` | `[]` | 106 | 8 |
| `controller_left.thumbstick_y` | `WireDType.FLOAT64_LE=3` | `[]` | 114 | 8 |
| `controller_left.thumbstick_click` | `WireDType.FLOAT64_LE=3` | `[]` | 122 | 8 |
| `controller_left.menu_click` | `WireDType.FLOAT64_LE=3` | `[]` | 130 | 8 |
| `controller_left.squeeze_value` | `WireDType.FLOAT64_LE=3` | `[]` | 138 | 8 |
| `controller_left.trigger_value` | `WireDType.FLOAT64_LE=3` | `[]` | 146 | 8 |
| `controller_right.present` | `WireDType.UINT8=1` | `[]` | 154 | 1 |
| `controller_right.grip_position` | `WireDType.FLOAT32_LE=2` | `[3]` | 155 | 12 |
| `controller_right.grip_orientation` | `WireDType.FLOAT32_LE=2` | `[4]` | 167 | 16 |
| `controller_right.grip_is_valid` | `WireDType.UINT8=1` | `[]` | 183 | 1 |
| `controller_right.aim_position` | `WireDType.FLOAT32_LE=2` | `[3]` | 184 | 12 |
| `controller_right.aim_orientation` | `WireDType.FLOAT32_LE=2` | `[4]` | 196 | 16 |
| `controller_right.aim_is_valid` | `WireDType.UINT8=1` | `[]` | 212 | 1 |
| `controller_right.primary_click` | `WireDType.FLOAT64_LE=3` | `[]` | 213 | 8 |
| `controller_right.secondary_click` | `WireDType.FLOAT64_LE=3` | `[]` | 221 | 8 |
| `controller_right.thumbstick_x` | `WireDType.FLOAT64_LE=3` | `[]` | 229 | 8 |
| `controller_right.thumbstick_y` | `WireDType.FLOAT64_LE=3` | `[]` | 237 | 8 |
| `controller_right.thumbstick_click` | `WireDType.FLOAT64_LE=3` | `[]` | 245 | 8 |
| `controller_right.menu_click` | `WireDType.FLOAT64_LE=3` | `[]` | 253 | 8 |
| `controller_right.squeeze_value` | `WireDType.FLOAT64_LE=3` | `[]` | 261 | 8 |
| `controller_right.trigger_value` | `WireDType.FLOAT64_LE=3` | `[]` | 269 | 8 |
| `hand_left.present` | `WireDType.UINT8=1` | `[]` | 277 | 1 |
| `hand_left.joint_positions` | `WireDType.FLOAT32_LE=2` | `[26,3]` | 278 | 312 |
| `hand_left.joint_orientations` | `WireDType.FLOAT32_LE=2` | `[26,4]` | 590 | 416 |
| `hand_left.joint_radii` | `WireDType.FLOAT32_LE=2` | `[26]` | 1006 | 104 |
| `hand_left.joint_valid` | `WireDType.UINT8=1` | `[26]` | 1110 | 26 |
| `hand_right.present` | `WireDType.UINT8=1` | `[]` | 1136 | 1 |
| `hand_right.joint_positions` | `WireDType.FLOAT32_LE=2` | `[26,3]` | 1137 | 312 |
| `hand_right.joint_orientations` | `WireDType.FLOAT32_LE=2` | `[26,4]` | 1449 | 416 |
| `hand_right.joint_radii` | `WireDType.FLOAT32_LE=2` | `[26]` | 1865 | 104 |
| `hand_right.joint_valid` | `WireDType.UINT8=1` | `[26]` | 1969 | 26 |

#### Camera descriptor and binary transport

Each accepted OAK-D camera has a stable stream ID 1–3 in Begin list order, including initially absent optional cameras:

```json
{
  "stream_id":1,"kind":2,"name":"camera_left",
  "type":1,"required":true,"width":640,"height":480,"fps":30,
  "codec":1,"jpeg_quality":90,
  "orientation":{"rotation_degrees":0,"mirrored":false},"intrinsics":null
}
```

Intrinsics is null or a finite 3×3 matrix for the transmitted dimensions. Images are initially unrotated/unmirrored. A CaptureImage payload is one independently decodable complete JPEG, never a delta. Codec does not prescribe the client's decoded BGR/RGB representation.

The camera descriptor repeats name and type from the accepted request and adds the server-resolved capture settings. It is announced in SessionInit. Any hardware-measured calibration not yet known is null; an updated calibration observation can be supplied through camera status without changing the decoding layout. These returned fields describe how to route and interpret the images:

| Camera descriptor field | Python enum and accepted values, or ordinary type | Meaning |
|---|---|---|
| `stream_id` | Not an enum: integer 1–3. | Identifies this camera in binary headers and status records for the duration of the session. |
| `kind` | `StreamKind`: `IMAGE=2` for cameras; `TRACKING=1` is reserved for the tracking stream. | Tells the client this stream contains encoded images rather than tracking blobs. |
| `name` | Nonempty string matching the accepted camera request; no enum. | Identifies the logical camera selected in Begin. |
| `type` | `CameraType`: `OAK_D_COLOR=1`. | Identifies the acquisition path used by the server. This is separate from kind, which identifies the stream as IMAGE. |
| `width`, `height` | Not enums: positive integers in pixels. | Give the dimensions of every decoded image in this stream. These are server-selected settings, not client request fields. |
| `fps` | Not an enum: finite positive number. | Gives the server-configured camera frame rate. It is separate from the tracking polling rate and does not guarantee images arrive at that rate during disconnection or congestion. |
| `codec` | `ImageCodec`: `JPEG=1`. | Selects the decoder for each complete image payload. |
| `jpeg_quality` | Not an enum: integer 1–100. | Reports the server-selected JPEG encoder setting. Higher values generally retain more detail and produce larger images; it is not a guaranteed accuracy percentage. |
| `required` | Boolean chosen by the server. | States whether failure to open this configured camera prevents session establishment. False permits unavailable status and no images while other streams run. It is not a declaration that the client must record this camera. |
| `orientation.rotation_degrees` | Not an enum: integer 0. | Advertises no server-applied image rotation. A client must not assume a hidden quarter-turn correction. |
| `orientation.mirrored` | Boolean false. | Advertises no server-applied mirror operation. |
| `intrinsics` | Null or a finite numeric array of shape \[3,3\]; not an enum. | A camera projection matrix for these image dimensions. Its focal-length terms express how rays project into pixels, and its principal-point terms locate the image's optical center in pixel coordinates. Null means the server supplies no such calibration; it must not be interpreted as an identity camera matrix. |

Each WebSocket binary message consists of a 32-byte header followed immediately by a payload. For CaptureFrame, that payload is the packed tracking blob defined by SessionInit. For CaptureImage, it is the JPEG bytes. The header itself contains no JSON and no field-name strings.

Both binary families use `struct.Struct("<4sBBHQQQ")`:

| Offset | Type | Content |
|---|---|---|
| 0 | 4 bytes | ASCII GTC1, a fixed framing identifier. |
| 4 | uint8 | Version 1. |
| 5 | uint8 | `BinaryKind`: `CAPTURE_FRAME=1`, `CAPTURE_IMAGE=2`. |
| 6 | uint16 LE | Advertised stream ID. |
| 8 | uint64 LE | Per-stream generated sequence, starting at 0. |
| 16 | uint64 LE | Sample/capture-host monotonic timestamp, ns. |
| 24 | uint64 LE | Capture-host receive/assembly timestamp, ns. |
| 32 | bytes | Exactly 1995 tracking bytes or one complete JPEG. |

Tracking messages are 2027 bytes. The session ID is supplied by the active SessionBegun association; do not repeat it per sample. No binary data precedes SessionBegun or follows its matching SessionStopped. A rejection for another ID does not disturb the active stream.

##### CaptureFrame

CaptureFrame uses BinaryKind.CAPTURE\_FRAME=1, stream\_id=0, and exactly 1995 payload bytes after the 32-byte header. The payload contains all fields in the tracking-offset table. Source presence and validity determine which values are usable; absent groups do not shorten the message. Every frame is a complete snapshot rather than a delta against a previous frame.

##### CaptureImage

CaptureImage uses BinaryKind.CAPTURE\_IMAGE=2 and the corresponding camera stream\_id in 1–3. The payload is one complete JPEG whose decoded dimensions match that stream's SessionInit descriptor. A client decodes each image independently, checks dimensions and does not require a previous image. An unavailable camera sends no image rather than an empty JPEG or a fabricated black frame.

##### Transport limits and closure

| Transport setting | Value and interpretation |
|---|---|
| Inbound control-message limit | 64 KiB per complete JSON message. An oversized message closes with WebSocket code 1009. |
| Tracking message size | Exactly 2027 bytes, including its header. |
| JPEG payload ceiling | 2 MiB; the complete binary image message is at most 2 MiB plus 32 bytes. The client configures its receive limit for the complete message. |
| Compression | WebSocket per-message compression is disabled; JPEG is already encoded image data. |
| Ping interval / timeout | 20 seconds / 20 seconds. Ping/Pong tests transport liveness, not device availability or sample age. |
| Send deadline | 2 seconds. A stalled send closes the connection and triggers local cleanup; the server does not cancel an in-flight send and reuse an uncertain transport. |
| Close timeout | 2 seconds for bounded connection shutdown. |
| Second owner | The server closes the second connection with code 1013 without disturbing the first owner. |
| Invalid transport/envelope | Malformed JSON, wrong-direction messages, client binary input, invalid session IDs or unsupported envelope versions close with code 1002. Valid identified request rejections use RequestRejected instead. |

Outgoing sample queues are bounded and discard oldest unsent samples under pressure. Sequence gaps and final dropped counts expose generated samples that do not complete delivery. A successful send means bytes are handed to the transport; it does not acknowledge application processing or recording.

The stream ID is an ordinary integer, not an enum: 0 identifies tracking and 1–3 identify the cameras advertised for this session. Sequence is an ordinary unsigned integer in 0–2^64−1, not a status code. It counts generated samples per stream, so a jump can expose samples discarded before delivery. The two timestamps are also ordinary unsigned integers in 0–2^64−1: sample time describes the sampling step, while receive/assembly time describes when the server assembled the data for transport. Their difference is not network latency to the client. Recent arrival and actual recent capture are distinct, especially across a distant network.

IC's public HeadInput/ControllerInput/HandInput contain no per-device capture timestamps. The tracking timestamp identifies the host sampling step and preserves its GraphTime/returned context. It is not socket-send time or sensor exposure time. The wire clock declaration uses HOST\_POLL; a camera timestamp in this contract does not certify exposure-time alignment.

A remote client has another monotonic epoch, meaning its elapsed-time clock starts from a different origin. The client can measure time since the last receipt using its own clock, detect progressing or repeated server sample timestamps, and measure intervals between server timestamps. None of those alone establishes the absolute age of a sample: a queue of old but progressing samples can still arrive continuously. Actual capture-to-client sample age requires an independently established clock relationship. This protocol supplies no clock synchronization or exposure-time alignment, and clients must not subtract a server timestamp directly from their local clock or label recent receipt as proof of low latency.

#### Stop, terminal replies and ownership

```json
{"type":2,"version":1,"session_id":"session-1"}
```

Stop cancels startup or active capture, closes session resources, discards queued samples and sends SessionStopped. The owner socket remains usable for a new session. SessionStopped includes the final available status snapshot:

```json
{
  "type":6,"version":1,"session_id":"session-1","reason":1,"error_message":null,
  "status":{"updated_at_ns":125000,"providers":[],"devices":[],"cameras":[]},
  "streams":[{"stream_id":0,"produced":120,"sent":117,"dropped":3}]
}
```

SessionStopped is terminal for an accepted session: once sent, no further samples or live status for that session follow. It is never used to reject a command while leaving that same session running. reason uses StopReason; error\_message is null for normal closure or a readable failure explanation. Native provider/device StatusReason remains inside the final status snapshot. Never send tracebacks.

Stop contains only the common envelope: MessageType.STOP=2, version=1 and the target session\_id. It needs no stream list because it ends the entire session, including tracking, plugins and cameras.

| SessionStopped field | Python enum and accepted values, or ordinary type | Meaning |
|---|---|---|
| `reason` | `StopReason`: `REQUESTED=1`, `OWNER_DISCONNECTED=4`, `SERVER_SHUTDOWN=6`, `IC_ERROR=7`, `IC_EXITED=8`, `CAMERA_ERROR=9`. | REQUESTED follows successful Stop cleanup; OWNER\_DISCONNECTED records lost ownership; SERVER\_SHUTDOWN records an orderly server exit; IC\_ERROR means an IC operation or cleanup raised; IC\_EXITED means the acquisition worker unexpectedly ended; CAMERA\_ERROR means a fatal camera failure ended the session. |
| `error_message` | Null or a nonempty string; no enum. | Readable failure explanation, including the affected camera or process when useful. Normal requested Stop, clean shutdown and owner loss use null. Clients never parse this text as a code. |
| `status` | Last complete status object, or null if none was ever available. | Preserves the last observed state for diagnosis. It is not an acknowledgement that the client received every final sample. |
| `streams` | Array of final counter records; may be empty. | Reports delivery accounting for each stream that was initialized. These are counters, not stream decoding descriptors. |
| Counter `stream_id` | Not an enum: integer matching an initialized stream. | Selects which stream these counters describe. |
| Counter `produced` | Not an enum: nonnegative integer. | Counts complete samples created before the session ended. |
| Counter `sent` | Not an enum: nonnegative integer. | Counts completed server send operations, not successful client recording or processing. |
| Counter `dropped` | Not an enum: nonnegative integer. | Counts generated samples discarded before completed send, including queued data discarded at shutdown. Final produced must equal sent+dropped. |

Final counters satisfy produced=sent+dropped after queued data is discarded. Empty streams is valid if startup failed before sample producers start. Final status can be null only if no observation was available. Preserve native failed/stopped states without inventing a hardware observation. If cleanup raises during an otherwise requested Stop, report IC\_ERROR rather than REQUESTED. If cleanup raises after an earlier acquisition failure, keep the original stop reason and include the additional cleanup diagnostic in error\_message.

#### Lifecycle mapping and request rejection

SessionInit, StartupStage and StopReason are server protocol concepts; they are not native IC enums. ProviderState, DeviceState and StatusReason retain the native observation meanings. The mapping is:

| Server event | IC operation or observation | Wire result |
|---|---|---|
| Accept a valid Begin and initiate startup | Resolve the fixed schema and station configuration and initiate IC startup without entering its native capture context; camera-only requests initiate camera startup. | SessionInit once; Python on_started receives CaptureStarted. |
| Initialize resources | Construct TeleopSession and enter its context; native status can be INITIALIZING or UNKNOWN. | SessionStarting with stage/message and cached observations. |
| Establish acquisition | Context entry and the first step succeed; missing physical devices can remain absent/unknown. | SessionBegun, then binary delivery. |
| Observe a change | Read the refreshed public IC status after stepping, or an observed camera change. | SessionStatus with the shared snapshot schema. |
| Stop normally | Exit native context and finish cleanup. | SessionStopped / REQUESTED. |
| Native operation raises | Construction, context entry, step, plugin health check or cleanup raises. | Cleanup and SessionStopped / IC\_ERROR, preserving native diagnostics/status. |

StartupStage describes server operations, not an extra device-state machine. Use WAITING\_FOR\_HEADSET or CONNECTING only when the server's connection API supplies evidence for that stage; a long-running IC context entry alone does not establish either condition.

Native step checks plugin health at frame 0 and every 60 frames. check\_health failures refresh observed process state and re-raise; DeviceIO.update exceptions mark RUNTIME\_UPDATE\_FAILED and re-raise. The server must not swallow these public API exceptions and claim uninterrupted native capture. Device-only loss without an exception can continue to appear as disconnected/degraded/absent outputs.

##### RequestRejected

RequestRejected is a reply about one command, not a session transition. It leaves any active session and decoding layout unchanged, even if the rejected command uses the active session\_id. No counters or device snapshot are attached.

```json
{"type":10,"version":1,"session_id":"session-1","request_type":1,"reason":1,"error_message":"Begin configuration differs from the active session"}
```

| RequestRejected field | Python enum and accepted values, or ordinary type | Meaning |
|---|---|---|
| `request_type` | `MessageType`: `BEGIN=1`, `STOP=2`. | Identifies which client command was rejected. |
| `reason` | `RejectionReason`: `INVALID_CONFIG=1`, `BUSY=2`, `UNKNOWN_SESSION=3`. | INVALID\_CONFIG rejects invalid/unsupported settings or changed configuration under an active/completed ID; BUSY rejects a new Begin while another session owns resources; UNKNOWN\_SESSION rejects Stop for an unknown ID. These are command outcomes, not IC device-status reasons. |
| `error_message` | Nonempty string; no enum. | Explains the rejection for display without requiring text parsing to identify its category. |

| Event | Response |
|---|---|
| New Begin while idle | Start resources, progress, then SessionBegun or startup failure. |
| Identical Begin for the active ID | Replay SessionInit, then current SessionStarting if still starting or SessionBegun and current status if running. Do not reallocate resources, reset sequences or reinstall the same decoding layout. |
| Same active ID with changed config | RequestRejected / INVALID\_CONFIG; active session continues. |
| Different-ID Begin while busy | RequestRejected / BUSY for that requested ID; active session continues. |
| Stop for active ID | Clean up, then SessionStopped / REQUESTED, or IC\_ERROR if cleanup raises. At most the current in-flight send finishes. |
| Duplicate Stop for most recently completed ID | Replay its cached terminal reply. |
| Stop for an unknown ID | RequestRejected / UNKNOWN\_SESSION for that ID; active session continues. |
| Reuse of a completed Begin ID | RequestRejected / INVALID\_CONFIG. Keep a bounded tombstone/reply cache; clients must generate unique IDs. |
| Inactive controllers/hands or disconnected/degraded gloves | Continue IC stepping and truthful status/optional output delivery. The client decides application readiness. |
| Provider/device failure observed without an IC exception | Report the observed status. If a subsequent native step/plugin health check raises, clean up and send SessionStopped / IC\_ERROR. |
| IC session construction/entry/step raises | Clean up and send StopReason.IC\_ERROR with the diagnostic error\_message and last available native status, if transport is usable. |
| IC acquisition worker unexpectedly returns/exits | Clean up and send StopReason.IC\_EXITED with a readable error\_message. Intentional Stop is not unexpected exit. |
| Required camera startup or active acquisition fails terminally | Clean up and send StopReason.CAMERA\_ERROR with the camera name and diagnosis in error\_message. Optional camera failures can remain camera status updates while other streams run. |
| Disconnect or stalled send | Clean up locally and release owner. No resume/replay across connections. |
| SIGINT/SIGTERM | Stop admission, bounded cleanup and best-effort SERVER\_SHUTDOWN, then close and exit. |

Malformed JSON, unknown/wrong-direction message type, binary client input, invalid/missing session\_id or unsupported envelope version closes with 1002; oversize input closes with 1009. A valid Begin envelope whose configuration cannot be supported receives RequestRejected / INVALID\_CONFIG without closing the owner socket. Rejections do not create a session and never emit SessionStopped. Once SessionInit has accepted the session, later acquisition initialization failure is terminal and uses SessionStopped.

If the previously-active session's own resource release failed or timed out, the server keeps rejecting every subsequent Begin with BUSY -- not just a Begin for a different ID -- even after that release might have genuinely finished in the background, since nothing re-awaits it. Restarting the server is the documented recovery for this case.

#### Example: Manus requested, only Pico head/controllers available

The client requests HEAD, CONTROLLERS and HANDS and enables the Manus human plugin.

1. If the required Manus plugin is not installed/discoverable, native IC startup raises; send SessionStopped with StopReason.IC\_ERROR and the native diagnostic text in error\_message. Pico availability does not satisfy a missing required plugin.
1. If the plugin starts and the OpenXR trackers can be established, session startup can succeed without gloves. Send SessionBegun; do not wait for all selected devices to become connected.
1. Manus left/right status reports DISCONNECTED / NO\_HARDWARE\_SIGNAL when the plugin sees no glove IDs. A present glove without current usable skeleton reports DEGRADED / NO\_CURRENT\_DATA; provider failure takes precedence as UNAVAILABLE.
1. Head/controllers can continue yielding usable outputs. Hands are absent if no runtime candidate is active, but may contain fallback headset hands. Do not infer Manus origin from present hands.
1. A client whose application requires gloves can remain unready or send Stop after inspecting status. The generic server does not silently impose that application's policy or terminate Pico-only tracking.
1. When gloves appear, normal native status and subsequent samples update without changing the stream layout. The current generic hand API still does not certify per-sample Manus provenance.

Launching the Manus plugin, observing connected gloves and preferring non-display OpenXR devices do not establish per-sample provider identity. Applications requiring exclusive Manus-origin samples need explicit source identity or selection support; generic HandsSource data does not supply that guarantee.
