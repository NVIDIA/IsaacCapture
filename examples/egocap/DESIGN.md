# Ego Capture UI

## 1. Ego Capture UI

Ego Capture UI is an IsaacCapture example for recording human demonstrations that robot-learning
systems can later train on. A typical station has a compute host, a headset, hand-tracking gloves,
a head-mounted camera, and two wrist cameras. The system records headset and controller poses,
hand tracking, and camera video as a series of episodes. Each episode is one take: an operator
starts recording, then keeps the take or discards it.

A browser page is the operator's interface to the station. It talks to a backend service over the
network, and the backend connects to CaptureServer, a standalone process that owns the capture
hardware and tracking session. The browser, backend, and CaptureServer run as independent
processes, whether on one machine or split across a local network.

![System overview: a capture station's headset, gloves and cameras connect to CaptureServer, which connects to the backend, which serves the browser and writes local recorded output.](assets/design/system-overview.svg)

*The capture station connects to CaptureServer; the backend holds the one connection to
CaptureServer, serves the browser, and writes recorded sessions to local storage.*

## 2. Capture station and data collection

### Operator workflow and station setup

A station belongs to one operator working at a time. A deployment's station file supplies that
station's label, its initial task instruction, the default operator name, the recording output
location, and the cameras it uses. At startup, the backend reads the station's camera names and
types from the file and requests those streams from CaptureServer. An unconfigured deployment uses
a fixed development default that requests no cameras; a configured-but-invalid station file blocks
new recording until it is fixed.

Before recording, the operator identifies the station and confirms its camera and tracking
assignment, selects or enters the task being demonstrated, and confirms the live preview — camera
video and tracked pose — looks correct. Starting an episode requires every required source to be
producing a current sample, not merely configured, so a camera or tracking source that has gone
stale blocks a new take rather than starting one against outdated data.

Recording itself is a short loop: start an episode, perform the demonstration, then either keep
the take or discard it. The operator can edit the task description between takes without
interrupting the session, and can rotate to a fresh batch of episodes when moving on to a new
block of work. A kept take is checked automatically once it finishes writing, and any problem
found is reported to the operator.

Offline analysis library functions give day's-end oversight, run as ad-hoc scripts rather than a
live page or CLI: a per-station activity log tracks active/inactive time through the day, and a
dataset-reporting function counts each station's valid versus excluded episodes and their recorded
duration from the saved session metadata, attributing each episode to the day it was actually
recorded rather than the session folder's own date.

Maintenance library functions, likewise ad-hoc scripts rather than a live page or CLI, correct a
misfiled task or studio label and relocate a session saved in the wrong place: one relocates a
session to its correct studio location, another updates task and studio labels across a range of
episodes. Both update a session's bookkeeping; its recorded video and tracking stay as captured.

### Timing and calibration requirements

A usable recording depends on more than a working camera and a working headset. Each camera's
capture latency needs to be known so its frames line up with tracking samples taken at the same
moment. The head camera needs calibrated intrinsics and extrinsics so its images can be related to
the headset's own tracked pose, and each wrist camera needs its own intrinsics calibration for the
same reason. The hand-tracking gloves need a calibration specific to the operator wearing them,
since hand geometry differs from person to person; a station used by more than one operator tracks
calibration per operator rather than assuming one calibration fits whoever is currently recording.

Station setup supplies and validates these calibrations once; recording uses whatever calibration
is currently in place without re-checking it for each take. Usable training data requires a
station whose calibration is current and correct.

### Data sources

A session records three kinds of data at once. Headset and controller tracking comes from the
station's tracking system through CaptureServer's IsaacCapture connection: position and
orientation for the head and both controllers, plus each controller's button and trigger state,
sampled together so the three poses stay consistent with each other within one tick.

Hand tracking comes from the hand-tracking gloves, reported as a full per-joint hand skeleton for
each hand when it is being tracked. The recorded hand-tracking row for a tick is dropped entirely
if either hand's wrist goes untracked — both hands together, not just the affected one — rather
than filled with a stale or fabricated pose; a live view can still show whichever hand remains
tracked, and the headset and controller rows for that tick are still recorded as usual.

A typical station uses one head camera and two wrist cameras. CaptureServer streams the supported
camera set requested by the client, so a station can be configured with a different camera set
without changing how the rest of the system records or stores it. Every recorded sample carries
its own capture time, the station's own sampling moment, and receive time, the station-side moment
the sample was assembled for transport; downstream tools read timing directly from these recorded
values.

### Saved data

A kept episode is written to the station's local storage as one MP4 video file per recorded
camera, numeric tracking data — headset and controller pose, hand skeleton — stored in Zarr, and
metadata: the task description and the operator who recorded it.

An automatic check runs once a take finishes recording and marks it kept, flagged, or rejected,
with the result and reason in the session status the backend reports back. A measured tracking
problem flags the take but keeps it, letting downstream consumers decide for themselves whether to
use it. A measured camera failure rejects the take and removes its media, since a recording
missing its camera content is not useful; a storage-integrity failure also rejects the take, but
keeps its data and reports its path for inspection. A take interrupted mid-recording, including by
a station restart, is retained as a pending episode that blocks a new take at that station. The
recording service exposes `recover_episode`/`discard_pending` to resolve it, but the browser page
has no command that calls either one yet -- clearing a pending episode today needs a direct call
into the service, not the page.

Whichever of the station's camera calibration files are actually available from setup — head
intrinsics and extrinsics, and the wrist cameras' intrinsics and calibration offsets — are
preserved with the recording session alongside its episodes, so a take stays tied to the
calibration active when it was captured rather than whatever calibration is live on the station
later. Camera latency and glove calibration remain station-setup requirements this copying does
not archive; a station missing an optional calibration file can still record, just without that
sidecar present. Keeping a take commits all of this to the local record; discarding one removes
its recorded media, so a discarded take's video and tracking never appear in the saved session.

### Downstream dataset use

A session's recorded episodes are the input to downstream training and dataset-conversion tools
outside this design's own scope. Those tools depend on a stable output contract: consistent source
names, file layout, image dimensions, coordinate frames, units, and joint order from one session to
the next. Reconfiguring how a camera or tracking source is captured does not change the name or
convention its recorded data carries, since a downstream reader keys on that contract rather than
on how the data was produced -- the mapping that preserves this for hand tracking (OpenXR's wire
convention to the saved joint order) is implemented but not yet verified against real hardware; see
`recording/README.md` for the current status. Video encoding is held to a looser standard: a file's pixels need not
match a different capture path exactly, since image dimensions, camera identity, and the alignment
between a frame and its tracking sample are what actually matter for compatibility, not
byte-for-byte pixel equality.

## 3. Web interface

### System overview

The web interface is two cooperating processes with a clean split of ownership. CaptureServer
owns the capture hardware: it holds the tracking session and reads the station's cameras, exposing
them through one hardware-facing interface independent of any particular recording application.
The backend owns the operator-facing side: it holds the one connection to CaptureServer for as
long as it runs, tracks the current recording session and its episodes, writes the local files,
and serves the browser page. A station swaps in different tracking or camera hardware without
changing how the backend records or serves a session.

A browser reaches the backend over the local network. Sending a recording command requires an
operator name, recorded with the take for attribution, and a control token issued once per backend
run, which authorizes the command. The WebSocket connection itself accepts only pages from an
origin the backend recognizes. Together, origin admission, the control token, and operator
attribution form the access model for a trusted, single-operator, single-station deployment.

![Browser connections: the backend reaches the browser over one WebSocket for commands, status and tracking, a separate MJPEG camera-preview path, and an optional embedded 3D view.](assets/design/browser-connections.svg)

*Three independent paths connect the backend to the browser: a WebSocket for commands, status and
tracking; MJPEG camera preview; and an optional embedded 3D view.*

### Data Recording Page

The data recording page is the operator's primary screen, and the one place a demonstration
session is actually run from start to finish. It shows the live camera grid and the current
tracked pose, the task description, and the recording controls: start, stop, discard, and rotate
to a new batch, available as on-screen buttons and keyboard shortcuts alike, usable with hands busy
demonstrating a task. Buttons and keyboard shortcuts use the same readiness checks. A command that
cannot run — no operator entered, no camera ready, no task set — is rejected with the specific
reason, and a take's outcome is
reported once its check actually completes, not assumed from the stop command alone.

If a browser tab reconnects mid-session, it is brought back to the same live state, since the
recording session itself lives in the backend, not in any one tab. The task description opens into
an editable draft between takes, with explicit Save and Cancel actions, without pausing the live
preview; saving applies the new text to the next episode that starts, while an episode already in
progress keeps the task text it was started with.

Pressing Space outside any text field, or clicking a toggle, turns voice control on. Listening
stays on across takes until it is turned off or the operator says "stop listening".

A fixed set of phrases invokes the same start/stop/discard/task command handlers and eligibility
checks the buttons and keyboard shortcuts use. A spoken `task <text>` opens the same draft editor,
with its own spoken save/cancel phrases; a spoken discard opens the same explicit confirmation the
Discard button does.

The browser performs the speech recognition itself, through its own speech-recognition API,
sending microphone audio to its own recognition service, and sends the resulting command or task
text to the backend. An ordinary pause in speech keeps listening on; denied microphone access, no
reachable recognition service, or an unsupported configuration stops it, with the reason shown next
to the toggle.

### Calibration

Producing usable data requires calibration, owned by the capture station rather than the recording
page. Hand-tracking calibration, specific to the operator wearing the gloves, is produced on the
capture host and supplied to the tracking connection as part of station setup; a session uses
whatever calibration the tracking connection has been given. Camera calibration — intrinsics and
extrinsics for the head camera, and intrinsics plus a calibration-offsets file for the wrist
cameras — is captured once during station setup and copied into each session by the backend, not
read by the recording page itself, so a take can be related back to the calibration active when it
was recorded.

### WebSocket connection

One WebSocket connection carries everything the recording page needs apart from the camera images
themselves: a status snapshot on connect and on every meaningful change, a steady heartbeat so a
tab can tell a healthy-but-quiet connection from a stalled one, sampled tracking data while a
session is live, and a reply to every command the page sends. Each reply identifies the command it
answers. The page's own commands — set the task, start or stop an episode, discard a take, rotate
to a new batch — travel back over that same connection.

Every connected browser tab shares the one backend-owned session, so a second viewer sees the same
live state a first viewer does, including a command sent from a different tab; this is what lets
one operator record from one tab while a second tab, open on another screen, simply watches. Each
sample on the connection carries its own capture and receive time, so a viewer can tell a fresh
reading from one that has gone stale; a camera or tracking source that stops updating is shown as
stale rather than left on its last good value.

### Viser interface

An optional embedded 3D view renders the tracked head, both controllers, and each hand's tracked
skeleton when present, from the same pose data the WebSocket already carries. It gives an operator
a second, spatial way to confirm tracking looks right before or during a take — a controller
drifting out of its expected position, or a hand losing tracking mid-reach, is often easier to
notice as a moving 3D figure than as a column of changing numbers. The view renders the
WebSocket's own tracking state directly, marking a hand absent or invalid exactly as the
connection reports it. A deployment turns the view on or off independently of live preview, the
WebSocket connection, and recording. The view opens with a fixed camera position and look-at point
chosen to frame the operator's space.

### MJPEG image streaming

Camera preview travels a separate path from the WebSocket: tracking updates many times a second
and needs to arrive promptly, while a camera image is comparatively large, so keeping the two
channels separate keeps a burst of image data from delaying the tracking data an operator relies
on in the moment. The supported OAK-D color cameras encode images on-device, and CaptureServer
forwards those encoded bytes to the backend unchanged; the backend streams the current image for
each camera to the browser as a continuous sequence. Camera preview publishes the latest JPEG per
stream, so a slow viewer sees the most current image without falling behind. Recording draws from
a separate bounded queue, keeping the two paths' load independent. Capture or recording failures
are reported in session status. The backend records the received images preserving their received
geometry — the camera's own framing, with no additional cropping, rotation, or rectification — and
encodes the recorded video as its own MP4 step, separate from the camera's own on-device JPEG
encoding.

### CaptureServer

CaptureServer exposes one request-driven capture interface for both mocked data and a real
tracking session with real cameras: a client selects the sources and cameras it wants, and
CaptureServer satisfies that exact request. A station requests a different camera set, or a
development environment runs entirely without hardware, through that same interface.

A capture attempt reports its own startup progress and becomes ready once the underlying tracking
session is actually producing samples, letting a client follow a slow connection to completion.
That readiness reflects the capture session itself; the backend separately and continuously checks
whether each required source is currently producing a fresh sample before starting a new episode.

A capture session stays open between takes: starting or stopping an episode reuses the same
hardware connection, so multiple episodes are recorded, kept, or discarded within one continuous
session. The interface description — what tracking and cameras were requested, and what format
each stream delivers — stays fixed for the session's duration, so a client decodes every sample
against one known shape. Ending a session reports stream counters and the session outcome for each
requested stream, giving the client a definite account of how the session ended.

### Backend recording responsibilities

The backend is the one place that turns raw capture data into a recorded session. It holds one
active recording context at a time, shared by every connected browser tab rather than
reconstructed per tab. The recording session stays in the backend when a browser disconnects;
reconnecting tabs receive its current state. Rotating to a new batch ends the current batch; the
next take starts a new recording session, independent of the underlying capture connection, which
keeps running under it the whole time. The backend writes accepted samples to local storage, runs
the automatic check once a take stops, and reports the outcome — kept, flagged, or rejected — back
to every connected viewer, with the specific reason behind any failure.

### Recordings browser (PoC demo, not part of the final design)

A separate, optional browser page lists and downloads the files a station has already recorded,
reading directly from the recording output root. It is off by default; enabling it serves every
file under that root to anyone who can reach the backend, with no login, token, or origin check of
its own, so a deployment turns it on only where that exposure is acceptable. It is independent of
the recording page: it never reconnects to CaptureServer and never affects a live session.

## 4. Assumptions and scope

This design serves a single operator working at one station at a time. The browser, backend, and
CaptureServer run close together, on the same machine or across one local network, with latency
suited to that setting. The backend's recording output and the calibration assets a session needs
are reachable from the backend's own host; a cross-host timestamp comparison relies on whatever
clock relationship the deployment itself establishes between hosts.

Recorded data is written to the station's own local storage, which a deployment manages like any
other local output — moving, backing up, or uploading it is the deployment's own responsibility.
Camera and tracking timestamps mark each source's own host-sampling moment at host-sampling
resolution, giving each sample's own timing as captured rather than a hardware exposure time or a
frame-exact synchronization across sensors.

A station's calibration — camera latency, head camera intrinsics/extrinsics, wrist camera
intrinsics and calibration offsets, and glove calibration — is current and correctly applied when
a session starts; station setup owns producing and verifying it. Recorded video preserves the
camera's own received pixels; a downstream consumer applies whatever geometric correction it needs
using the recorded calibration assets.

Viewing a station's live state requires network access to the backend. Issuing a recording command
additionally requires an operator name and a control token issued for that backend run, and the
WebSocket connection itself admits only a recognized origin — attribution and basic access control
sized for a trusted, single-operator, single-station deployment. Each station's deployment is
independent; coordinating or comparing data across stations is a downstream consumer's
responsibility.
