<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Remote Assistance: LiveKit prototype

Video and robot control pass through LiveKit between Robot Site and Edge Compute.
CloudXR connects the operator's headset to Edge Compute. NGINX is a later addition.

## Multiple Site Process Setup

Run **Robot** and **Camera Televiz** independently or together; the **Essential**
processes are always needed for the headset workflow described here. These roles
are intended for multiple computers, but running everything on one computer is
also valid and lets you verify the bridges before introducing network issues.

Run each foreground process in its own terminal. Process numbers identify the
terminals at each site; they are not a strict startup order. Start LiveKit first.
For Robot operation, start MuJoCo (when used), SONIC and both bridges before
starting the Pico manager. For Camera Televiz, start the sender and viewer.

| Site | Process | Group | Purpose |
|---|---|---|---|
| Edge Compute | 0 | Essential | LiveKit SFU |
| Edge Compute | 1 | Essential | CloudXR / Pico teleop streamer |
| Edge Compute | 2 | Robot | Edge control bridge |
| Edge Compute | 3 | Camera Televiz | Camera receiver and display |
| Robot Site | 0 | Robot | MuJoCo simulator (simulation only) |
| Robot Site | 1 | Robot | SONIC policy deployment / ZMQ manager |
| Robot Site | 2 | Robot | Robot control bridge |
| Robot Site | 3 | Camera Televiz | Camera frame sender |

### One-time preparation

The commands assume your existing GR00T, CloudXR and SONIC environments work.
On each machine that runs a bridge or sender, install the Remote Assistance
package from its IsaacTeleop checkout:

```bash
cd ~/IsaacTeleop
uv venv --python 3.12 examples/remote_assistance/.venv
uv pip install --python examples/remote_assistance/.venv/bin/python \
  -e 'examples/remote_assistance[camera,test]'
```

For Camera Televiz on Edge Compute, prepare the viewer environment:

```bash
cd ~/IsaacTeleop
examples/camera_viz/camera_viz.sh setup
uv pip install --python examples/camera_viz/.venv/bin/python \
  -e examples/remote_assistance
```

For MuJoCo on Robot Site, run `bash install_scripts/install_mujoco_sim.sh`
from `~/GR00T-WholeBodyControl` if its environment is not already installed.
NVIDIA Container Toolkit configuration and any Docker daemon restart belong to
machine preparation, before starting LiveKit or SONIC containers. They are not
part of each session's startup.

**Current configuration:** the supplied YAML files use localhost and development
credentials for the single-computer test. The process placement below is also the
intended multiple-computer layout, but moving machines requires configuring a
reachable SFU address/media ports and credentials first. Run one SFU on Edge
Compute; Robot Site connects to that same SFU. Keep the ZMQ endpoints local to
each site. See the network configuration notes under Delivery and Lifecycle.

### Edge Compute

#### Process 0 — Essential — LiveKit SFU

```bash
cd ~/IsaacTeleop
docker compose -f examples/remote_assistance/compose.yaml up -d
```

This starts the `isaac-remote-assistance-livekit-1` container in the background.
Run it on **Edge Compute only**, once for both Robot and Camera Televiz. `up -d`
leaves an unchanged running service in place; it does not need a dedicated
terminal. Check its status with:

```bash
docker compose -f examples/remote_assistance/compose.yaml ps
```

#### Process 1 — Essential — CloudXR / Pico Teleop Streamer

```bash
cd ~/GR00T-WholeBodyControl
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py \
  --manager --input-source isaac-teleop \
  --port 5558 --zmq_feedback_port 5559
```

Connect the headset's CloudXR.js web client to this Edge Compute host as in the
existing workflow. This process supplies the shared XR workflow even when only
Camera Televiz is being used. The Robot bridges are optional for camera viewing.
A headless camera probe or `--mode window` viewer does not require this XR process.

#### Process 2 — Robot — Edge Control Bridge

```bash
cd ~/IsaacTeleop
examples/remote_assistance/.venv/bin/python -m remote_assistance bridge \
  examples/remote_assistance/session.yaml --role edge
```

Forwards the local Pico manager's ZMQ output to LiveKit and returns robot feedback
to the manager. This bridge carries control data; it is not the camera sender.

#### Process 3 — Camera Televiz — Receiver and Display

```bash
cd ~/IsaacTeleop/examples/camera_viz
source .venv/bin/activate
source ~/.cloudxr/run/cloudxr.env
./camera_viz.sh run configs/livekit-webcam.yaml --mode xr
```

Receives the Robot Site video through LiveKit and displays it through
Televiz/CloudXR. Use `--mode window` for a local monitor instead of XR.

### Robot Site

#### Process 0 — Robot — MuJoCo Simulator (Optional)

Use this process for simulation; omit it when using a physical robot.

```bash
cd ~/GR00T-WholeBodyControl
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

#### Process 1 — Robot — SONIC Policy Deployment / ZMQ Manager

**On the Robot Site host**, enter the development container (this step is required):

```bash
cd ~/GR00T-WholeBodyControl/gear_sonic_deploy
export TensorRT_ROOT=$HOME/TensorRT
./docker/run-ros2-dev.sh
```

Wait for the container shell before continuing. Its prompt normally looks like
`root@<hostname>:/workspace/g1_deploy#`. A `(g1_deploy)` prefix alone only means
the environment script was sourced; it does not mean you are inside Docker.
If the launcher fails or returns to the host prompt, resolve that error first.

**Inside the container**, run:

```bash
cd /workspace/g1_deploy
source scripts/setup_env.sh
./deploy.sh --cp policy/low_latency/model \
  --obs-config policy/low_latency/observation_config.yaml \
  --input-type zmq_manager sim
```

If CMake reports that `/home/.../gear_sonic_deploy/build` differs from
`/workspace/g1_deploy/build`, the build was launched on the host using the
container's cache. Enter the container and rerun the commands above; do not
remove the cache or install the missing container dependencies on the host.

The deployment script builds the project. Use `real` in place of `sim` only for
your physical-robot workflow. SONIC subscribes on local port **5556** and publishes
feedback on **5557**. The development container must share the Robot Site host's
network, as in the existing setup. The Pico manager runs on Edge Compute, not here.

#### Process 2 — Robot — Robot Control Bridge

```bash
cd ~/IsaacTeleop
examples/remote_assistance/.venv/bin/python -m remote_assistance bridge \
  examples/remote_assistance/session.yaml --role robot
```

Receives control through LiveKit, republishes it to local SONIC, and sends SONIC
feedback back to Edge Compute. This is the initial Teleop Remote Session adapter.

#### Process 3 — Camera Televiz — Camera Frame Sender

```bash
cd ~/IsaacTeleop
examples/remote_assistance/.venv/bin/python -m remote_assistance camera \
  examples/camera_viz/configs/livekit-webcam.yaml
```

Captures the local V4L2 camera and publishes video to LiveKit. Only this process
should own the camera device; stop a direct-camera viewer before starting it.
Camera capture and robot control remain separate processes in the same room.

Capture size and rate come from `livekit-webcam.yaml` (currently 2560×1440 at
30 fps, MJPG). For 1080p60, set width/height/fps to 1920/1080/60. Keep sender and
viewer configurations consistent and restart both after changing the mode.

## Delivery and Lifecycle

### Control routing and status

| Local endpoint | Producer | Consumer |
|---|---|---|
| Edge: 5558 | Pico manager | Edge bridge |
| Robot: 5556 | Robot bridge | SONIC |
| Robot: 5557 | SONIC feedback | Robot bridge |
| Edge: 5559 | Edge bridge | Pico manager feedback reader |

Separate ports ensure the one-computer setup traverses LiveKit. Do not leave an
old Pico manager publishing directly on 5556. Routes in `session.yaml` preserve
SONIC's single-frame ZMQ payloads: `pose`, `planner`, `command`, `manager_state`
and `g1_debug`. `goal` and `execution_state` routes are available for future
hybrid-compute adapters; no Local Brain or Cloud Brain model is implemented.
Retargeting stays on the edge. This adapter does not make the core
`TeleopSession.step()` API remote.

Every five seconds the bridges report peer readiness, `control=waiting`,
`active` or `suspended`, and per-topic `to_livekit/` and `to_zmq/` counters.
Check increasing target counts toward SONIC and `g1_debug` counts toward the
manager. These are forwarding counts, not execution acknowledgments.

### Input loss and delivery guarantees

- Each endpoint accepts only its configured peer and topic allowlist. JWTs
  must match the configured room and identity. Distinct components use distinct
  identities so they do not evict each other from the room.
- Pose, planner and observation updates use lossy packets and coalesce by topic
  between discrete commands. Commands and goals use reliable packets. The local
  outbox preserves target-before-command submission order; separate reliable and
  lossy channels do not guarantee that order remotely. Atomic mode transitions
  require a future application acknowledgment/dependency contract.
  Reliable packets are best-effort transport, not application acknowledgments.
  The existing local ZMQ PUB/SUB handoff also does not acknowledge execution.
- Messages are split into at most 1100-byte chunks plus a 36-byte header,
  below LiveKit's recommended lossy packet size. Reassembly is bounded to
  32 messages, 128 KiB each. Incomplete, expired, duplicate and old messages
  are discarded. State samples do not accumulate behind old samples.
- Receivers issue short-lived random leases. A returned packet is accepted
  only before that lease expires on the receiver's monotonic clock; no clock
  synchronization is required. The bridge separately bounds local queue age.
  These are transport-admission deadlines, not timestamps at sensor exposure
  or a guarantee of maximum physical actuation age. Reconnection clears leases.
- The robot bridge waits silently for initial targets and reports `control=suspended`
  after 300 ms without pose/planner input. It never generates SONIC stop commands
  on timeout or shutdown: `stop=true` exits the policy process. Fresh targets
  resume forwarding automatically; explicit operator stop commands pass unchanged.
  Feedback and command messages do not refresh target freshness.
- Suspension delegates behavior to the existing local controller: streamed motion
  finishes its buffered frames and clamps to the last reference frame; planner
  input defaults to IDLE after its own one-second timeout. The policy continues
  running. This is not an immediate physical pose freeze or a guarantee of balance.
  Validate loss and recovery in MuJoCo before physical use. A dedicated controller
  hold/idle contract, authority/arming rules, and command acknowledgments remain
  future work. The transport must not fabricate fresh timestamps on stale targets.
- Video reception has capacity one and a latest-frame mailbox. The viewer
  blanks the feed after one second without a fresh frame. Sender restarts are
  supported by track resubscription; terminal connection failures require
  restarting the affected process. Temporary reconnects are handled by the SDK.
- Video frame IDs and capture-thread monotonic timestamps travel in LiveKit
  frame metadata. Capture time is measured **after `VideoCapture.read()`**,
  not at sensor exposure. Monotonic timestamps from different computers cannot
  be subtracted to measure one-way latency.

The current raw-frame SDK adapter introduces host-memory copies and an extra
H.264 encode/decode between the robot and edge. The edge still performs the
CloudXR encoding. These costs are part of the experiment; no zero-copy or
hardware-encoding performance claim is implied by selecting `auto`.

### Camera behavior

Video uses H.264, a single encoding (no simulcast), and the configured bitrate
(currently 15 Mbit/s). The sender rejects a different negotiated capture mode.
`video_encoder: auto` lets the SDK select its backend; `nvenc` and `software`
are explicit alternatives whose availability depends on the host.

The viewer currently uses `video_decoder: software` because the tested NVDEC
path produced frozen pixels despite reporting fresh frames. This sets the SDK's
process-wide `LK_DISABLE_NVDEC` before connection, while leaving CUDA available
for Televiz. Changing to `auto` requires a fresh viewer process.

### Network configuration for separate computers

The supplied `livekit-local.yaml` is for local testing. Before moving Robot Site
to another machine, configure a routable SFU address and media ports. The current
ports are 7880/TCP for signaling, 7881/TCP for media fallback and 7882/UDP for media.
Both sites must connect to the same SFU and room.

Replace development keys with generated server credentials and room-scoped JWTs.
Set `development: false`, set `url` to the reachable SFU, and set `token_env` to
the environment variable containing each process's token. `edge.livekit` and
`robot.livekit` can override shared settings in `session.yaml`. Camera sender,
viewer and probe need distinct identities/tokens. Use TLS for remote signaling;
NGINX can be added later for TLS termination. WebRTC media connectivity and TURN,
when needed, remain separate requirements.

### Suspension, resumption and shutdown

In MuJoCo, test removing the headset: the robot bridge should report
`control=suspended` and increment `input_suspensions`, while SONIC keeps running.
Restore input and verify `control=active` without restarting SONIC. The prototype
does not blend target jumps or require an explicit resume gesture. Explicit
operator stop still terminates SONIC, as in the original workflow.

Stop foreground processes with Ctrl-C in their own terminals. Bridge shutdown
does not stop SONIC; shut down the controller through its normal operator workflow.
To stop the shared SFU, run on Edge Compute:

```bash
cd ~/IsaacTeleop
docker compose -f examples/remote_assistance/compose.yaml down
```

Stopping the SFU interrupts both video and control. Temporary reconnects are
handled by the SDK; terminal connection failures require restarting the affected
process. A future Authority Enforcer will handle ownership, arming and freshness
between the robot bridge and SONIC. Controller-local hold/idle behavior remains
necessary when the transport or authority process is unavailable.

### Optional verification

For a headless camera check, add `--synthetic` to the sender command and run:

```bash
cd ~/IsaacTeleop
examples/remote_assistance/.venv/bin/python -m remote_assistance probe \
  examples/camera_viz/configs/livekit-webcam.yaml --seconds 15 --synthetic
```

This verifies changing decoded pixels against frame IDs. Omit `--synthetic`
from both commands to probe a physical camera. Stop the extra subscriber before
latency measurements. Optical light-to-display measurements include the full
pipeline; packet RTT and received frame rate do not measure that latency.

Run automated tests from `~/IsaacTeleop`:

```bash
examples/remote_assistance/.venv/bin/python -m pytest examples/remote_assistance/tests -q
LIVEKIT_TEST_URL=ws://127.0.0.1:7880 \
  examples/remote_assistance/.venv/bin/python -m pytest \
  examples/remote_assistance/tests/test_integration.py -q
```

Integration tests use isolated rooms and test ZMQ endpoints, never live SONIC
command ports. They cover media, message transport, input loss and recovery.
They do not validate physical balance.

References: [LiveKit Python SDK](https://github.com/livekit/python-sdks),
[data packets](https://docs.livekit.io/transport/data/packets/),
[server ports](https://docs.livekit.io/transport/self-hosting/ports-firewall/).
