<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Remote Assistance: LiveKit prototype

Video and robot control pass through LiveKit between Robot Site and Edge Compute.
CloudXR connects the operator's headset to Edge Compute. The network profile uses
NGINX for TLS signaling; media and control data still pass through LiveKit.

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
| Robot Site | 0 | Robot | MuJoCo simulator (simulation only) |
| Robot Site | 1 | Robot | SONIC policy deployment / ZMQ manager |
| Robot Site | 2 | Robot | Robot control bridge |
| Robot Site | 3 | Camera Televiz | Camera frame sender |
| Edge Compute | 0 | Essential | LiveKit SFU |
| Edge Compute | 1 | Robot | Edge control bridge |
| Edge Compute | 2 | Camera Televiz | Camera receiver and display |
| Edge Compute | 3 | Essential | CloudXR / Pico teleop streamer |

### One-time preparation

Complete the following preparation before starting the site processes.

#### Robot Site — GR00T, SONIC and model checkpoints

On the **Robot Site host**, install Git LFS and clone GR00T-WholeBodyControl
if it is not already present:

```bash
sudo apt update
sudo apt install -y git git-lfs python3-venv
git lfs install
cd ~
git clone --recurse-submodules https://github.com/NVlabs/GR00T-WholeBodyControl.git
cd ~/GR00T-WholeBodyControl
git lfs pull
git submodule update --init --recursive
```

For an existing checkout, skip the clone and run the last three commands.
Keep any existing feature branch used by your working deployment.

Next, complete the upstream
[Installation (Deployment) guide](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html)
on Robot Site: install the CUDA/TensorRT prerequisites, then follow one complete
setup path through environment setup and `just build`. The process commands in
this README use the **Docker (ROS2 Development Environment)** path: configure
Docker access and NVIDIA Container Toolkit, set `TensorRT_ROOT` on the host,
launch `gear_sonic_deploy/docker/run-ros2-dev.sh`, and build inside the container.
If choosing native deployment instead, follow the guide's dependency installer,
environment setup and build instructions on the host, and run SONIC there.
Do not mix host and container build directories.

**Thor compatibility:** the linked guide currently specifies TensorRT 10.13 for
x86_64 and 10.7 with JetPack 6 for Orin; it does not establish a supported SONIC
configuration for Thor with JetPack 7.1/CUDA 13. Confirm the Thor-specific
TensorRT/container combination with the GR00T maintainers before policy
deployment. Our LiveKit camera/bridge installation does not validate SONIC
inference on Thor.

Download the **model checkpoints on Robot Site** by following the
[Model Card](https://nvlabs.github.io/GR00T-WholeBodyControl/model_card.html).
The launch commands below select the low-latency variant. From the GR00T root,
use a dedicated download environment:

```bash
cd ~/GR00T-WholeBodyControl
python3 -m venv .venv_models
.venv_models/bin/python -m pip install huggingface_hub
.venv_models/bin/python download_from_hf.py --low-latency
```

Use the matching encoder, decoder and observation configuration together. Check
that the files referenced by our launch commands are present:

```bash
ls gear_sonic_deploy/policy/low_latency/model_encoder.onnx \
   gear_sonic_deploy/policy/low_latency/model_decoder.onnx \
   gear_sonic_deploy/policy/low_latency/observation_config.yaml \
   gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx
```

#### Robot Site — MuJoCo environment (simulation only)

Before activating `.venv_sim` in Robot Site Process 0, install it from the
**GR00T repository root**:

```bash
cd ~/GR00T-WholeBodyControl
bash install_scripts/install_mujoco_sim.sh
source .venv_sim/bin/activate
```

Use `install_mujoco_sim.sh` for this environment; `install_pico.sh` creates the
teleoperation environment instead. The simulator installer recreates `.venv_sim`,
so run it during preparation, not every session. Omit this preparation and
Process 0 when using a physical robot. Successful installation on Thor must
still be followed by a local simulator startup check.

#### Both sites — Remote Assistance package

On each machine that runs a bridge or sender, install the Remote Assistance
package from its IsaacTeleop checkout:

```bash
cd ~/IsaacTeleop
uv venv --python 3.12 examples/remote_assistance/.venv
uv pip install --python examples/remote_assistance/.venv/bin/python \
  -e 'examples/remote_assistance[camera,test]'
```

#### Edge Compute — XR and camera viewer

Use your working GR00T Pico/IsaacTeleop and CloudXR environment on Edge Compute
for Process 3. For Camera Televiz, prepare the viewer environment:

```bash
cd ~/IsaacTeleop
examples/camera_viz/camera_viz.sh setup
uv pip install --python examples/camera_viz/.venv/bin/python \
  -e examples/remote_assistance
```

NVIDIA Container Toolkit configuration and any Docker daemon restart belong to
machine preparation, before starting LiveKit or SONIC containers. They are not
part of each session's startup.

**Choose a profile:** the checked-in YAML files use localhost and development
credentials for the single-computer test. For two computers, first follow
[Network setup](#network-setup-for-two-computers) below to generate private site
bundles, then use those bundles in the same processes. Run one SFU on Edge Compute;
Robot Site connects to it. Keep ZMQ endpoints local to each site.

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

**Two computers — run on the Robot Site host (Thor):**

```bash
cd ~/IsaacTeleop
source "$HOME/remote-assistance-robot/credentials.env"
examples/remote_assistance/.venv/bin/python -m remote_assistance bridge \
  "$HOME/remote-assistance-robot/session.yaml" --role robot
```

Source credentials in each new terminal. The generated YAML selects the Edge
server; sourcing credentials alone does not override a localhost YAML.

**Single-computer development test only:**

```bash
cd ~/IsaacTeleop
examples/remote_assistance/.venv/bin/python -m remote_assistance bridge \
  examples/remote_assistance/session.yaml --role robot
```

Receives control through LiveKit, republishes it to local SONIC, and sends SONIC
feedback back to Edge Compute. This is the initial Teleop Remote Session adapter.

#### Process 3 — Camera Televiz — Camera Frame Sender

**Two computers — run on the Robot Site host (Thor):**

```bash
cd ~/IsaacTeleop
source "$HOME/remote-assistance-robot/credentials.env"
examples/remote_assistance/.venv/bin/python -m remote_assistance camera \
  "$HOME/remote-assistance-robot/camera.yaml"
```

Source credentials in each new terminal. The generated YAML selects the Edge
server; sourcing credentials alone does not override a localhost YAML.

**Single-computer development test only:**

```bash
cd ~/IsaacTeleop
examples/remote_assistance/.venv/bin/python -m remote_assistance camera \
  examples/camera_viz/configs/livekit-webcam.yaml
```

Captures the local V4L2 camera and publishes video to LiveKit. Only this process
should own the camera device; stop a direct-camera viewer before starting it.
Camera capture and robot control remain separate processes in the same room.

Capture size and rate come from the selected camera YAML. Choose a mode supported
by that camera, keep sender and viewer configurations consistent, and restart
both after changing the mode.

### Edge Compute

**For two computers, use the generated server profile in Network setup step 3
and the bridge/viewer commands in step 9.** The localhost commands below are
for single-computer development; Process 3 applies to both setups.

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

#### Process 1 — Robot — Edge Control Bridge

```bash
cd ~/IsaacTeleop
examples/remote_assistance/.venv/bin/python -m remote_assistance bridge \
  examples/remote_assistance/session.yaml --role edge
```

Forwards the local Pico manager's ZMQ output to LiveKit and returns robot feedback
to the manager. This bridge carries control data; it is not the camera sender.

#### Process 2 — Camera Televiz — Receiver and Display

```bash
cd ~/IsaacTeleop/examples/camera_viz
source .venv/bin/activate
source ~/.cloudxr/run/cloudxr.env
./camera_viz.sh run configs/livekit-webcam.yaml --mode xr
```

Receives the Robot Site video through LiveKit and displays it through
Televiz/CloudXR. Use `--mode window` for a local monitor instead of XR.

#### Process 3 — Essential — CloudXR / Pico Teleop Streamer

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

If both bridges remain `ready=True` with empty Edge `counters` and `forwarded`,
their readiness heartbeats work but Edge has sent no application messages.
Check Pico Manager and its ZMQ publisher on port 5558 first. It waits for the
first Isaac Teleop body sample before binding that port, then publishes
`manager_state` even before robot activation. Missing body tracking, a stopped
manager or a mismatched ZMQ endpoint/topic can explain this pattern; these
counters alone cannot identify which. A single non-increasing `wrong_peer`
count does not establish a persistent peer mismatch. Capture Pico Manager's
startup output alongside both bridge logs when investigating intermittent starts.

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

### Experimental Thor hardware camera encoding

The opt-in `gstreamer` backend uses `nvv4l2h264enc` or `nvv4l2h265enc` and
publishes complete encoded frames through LiveKit, without a second sender-side
encode. It requires **LiveKit 1.1.20**: the Python adapter uses its private native
FFI because this SDK version has no public encoded-source wrapper. Missing NVIDIA
plugins or unsupported capture modes fail explicitly; there is no software fallback.

On Thor (Ubuntu 24.04 / Python 3.12), install the GStreamer Python bindings:

```bash
sudo apt install python3-gi gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0
gst-inspect-1.0 nvv4l2h264enc
gst-inspect-1.0 nvv4l2h265enc
```

Stop the current camera sender to release the camera. From `~/IsaacTeleop`,
create a separate profile from the working network bundle:

```bash
examples/remote_assistance/.venv/bin/python - <<'PYCONFIG'
from pathlib import Path
import yaml
bundle = Path.home() / "remote-assistance-robot"
cfg = yaml.safe_load((bundle / "camera.yaml").read_text())
cfg["livekit"].update(video_encoder="gstreamer", codec="h264", bitrate_mbps=15)
cfg["cameras"][0].update(type="v4l2", device="/dev/video4", fourcc="YUYV",
                         width=1280, height=720, fps=30)
(bundle / "camera-gstreamer.yaml").write_text(yaml.safe_dump(cfg))
PYCONFIG
source "$HOME/remote-assistance-robot/credentials.env"
PYTHONPATH="/usr/lib/python3/dist-packages${PYTHONPATH:+:$PYTHONPATH}" \
  examples/remote_assistance/.venv/bin/python -m remote_assistance camera \
  "$HOME/remote-assistance-robot/camera-gstreamer.yaml" \
  2>&1 | tee "$HOME/remote-assistance-gstreamer.log"
```

For H.264, keep the existing Edge receiver running. For H.265, change
`livekit.codec` to `h265` in this experimental sender profile and set
`livekit.video_decoder: auto` in the generated Edge camera profile. Stop and
restart the Edge viewer in a fresh process with `LK_DISABLE_NVDEC` unset, then
restart the sender. H.265 passed the desktop GPU pixel test with automatic
decoding; the software decoder produced no frames. This does not establish
long-duration decoder stability. Restore `video_decoder: software` when
returning to the established H.264 configuration.
Use the same scene, resolution, frame rate and bitrate for a useful comparison.
The tested D435i advertises **1280×720 at 30 fps**, **1920×1080 at 30 fps**, and
**960×540 at 60 fps**. Hardware encoding cannot enable 720p60 on a camera that
does not advertise it; use another camera for that mode.

Logs report encoded fps, encoded payload Mbps, native LiveKit acceptance and
queue drops. These are not receiver fps, wire bandwidth or encoder latency.
Overflow/rejection discards dependent frames until a keyframe; keyframe requests
are forwarded to the encoder. Bitrate feedback is applied only when the plugin
supports changes while playing; otherwise a warning indicates fixed bitrate.
Capture fps stays fixed, including when LiveKit requests a lower rate. This is
an experimental LAN path, not complete congestion adaptation. Color conversion
still uses CPU `videoconvert`; this is not a zero-copy pipeline.

To roll back, stop this sender and launch the original `camera.yaml` profile.

### Sensing Astra S56 on Orin (GMSL2 / Argus)

Camera selection stays in `cameras[].type`; encoding stays in
`livekit.video_encoder`. Existing V4L2 and DepthAI profiles remain supported:

| Camera source | `type` | `video_encoder` | Capture and encode |
|---|---|---|---|
| USB YUYV camera | `v4l2` | `gstreamer` | V4L2 → conversion → Jetson H.264/H.265 |
| USB camera / synthetic test | `v4l2` / `synthetic` | `auto` / `software` | Existing raw-frame LiveKit encoder |
| OAK | `depthai` | `depthai` | DepthAI → OAK onboard H.264 |
| S56 | `argus` | `gstreamer` | Argus ISP → NVMM → Jetson H.264/H.265 |

GStreamer source adapters in `gst_sources.py` supply a pipeline prefix and
configure its camera element. They share the encoder, keyframe recovery,
timestamps, feedback and LiveKit publisher. DepthAI retains its own capture
adapter and shares that encoded publisher. Adding a camera source does not
require changing the LiveKit transport. The prototype still publishes one
camera per sender process.

**One-time preparation on Orin:** install the matching Sensing driver/ISP
package and select its device-tree overlay using the vendor guide. On the
tested JetPack 6.2 / L4T 36.4.3 system, those files were already installed.
Ubuntu 22.04's system Python 3.10 GI bindings cannot be imported into the bridge's
Python 3.12 venv. Build matching bindings using the optional extra:

```bash
cd ~/IsaacTeleop
sudo apt install -y libgirepository1.0-dev libcairo2-dev pkg-config python3-dev \
  gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0
uv pip install --python examples/remote_assistance/.venv/bin/python \
  -e 'examples/remote_assistance[gstreamer]'
```

**Initialize the camera after boot** using the installed vendor package:

```bash
cd ~/SG10A_AGON_G2M_A1_AGX_ORIN_S56x1_SHF3Lx6_JP6.2_L4TR36.4.3
sudo bash ./load_modules.sh 30
sudo systemctl restart nvargus-daemon
v4l2-ctl --list-devices
```

The package documents S56 connections on J27 or J29 and a synchronization jumper
between pins 2 and 4 of J19. Check the guide for your board. Select `60` instead
of `30` when testing 60 fps; the external trigger must match the requested rate.
Loading the vendor script reinitializes the camera drivers, so stop camera
senders before running it.

Bundle generation includes matching sender/viewer presets:

| Profile | Output | Sensor mode | Status |
|---|---|---|---|
| `camera-s56-1080p30.yaml` | 1920×1080, 30 fps | 0 | Short LiveKit test verified |
| `camera-s56-720p60.yaml` | 1280×720, 60 fps | 0 (ISP scaling) | Experimental |
| `camera-s56-1080p60.yaml` | 1920×1080, 60 fps | 0 | Experimental |
| `camera-s56-2560x1984-30.yaml` | 2560×1984, 30 fps | 1 | Experimental |

For an existing bundle, generate presets on **Edge Compute** and transfer them
with the usual helper. This preserves its current credentials and base camera
configuration. Update the Robot Site code before selecting these new profiles.

```bash
cd ~/IsaacTeleop
examples/remote_assistance/.venv/bin/python -m remote_assistance.camera_profiles
bash examples/remote_assistance/deploy_robot_bundle.sh
```

On **Robot Site**, stop the previous sender, then launch:

```bash
cd ~/IsaacTeleop
source "$HOME/remote-assistance-robot/credentials.env"
# Use the active Orin desktop's DISPLAY and XAUTHORITY; these were :1 and
# /run/user/1000/gdm/Xauthority on the tested Marvin session.
export DISPLAY=:1
export XAUTHORITY=/run/user/1000/gdm/Xauthority
examples/remote_assistance/.venv/bin/python -m remote_assistance camera \
  "$HOME/remote-assistance-robot/camera-s56-1080p30.yaml"
```

Initial captures intermittently returned `NvBufSurfaceFromFd Failed`, even with
the active display. Longer Argus event/frame waits (10/15 seconds) allowed
repeated captures. The adapter allows 20 seconds for its first encoded frame,
then retains the normal 10-second stall watchdog. S56 mode programming takes
several seconds; allow startup to complete. Headless operation and long-duration
restart reliability remain unverified. Check standalone capture before debugging
LiveKit when this error appears. Do not hardcode session-specific display
settings into the camera backend. No CPU color conversion is requested
in the Argus path; this does not establish zero-copy behavior inside the SDK.

On **Edge Compute**, restart the viewer with the matching profile:

```bash
cd ~/IsaacTeleop/examples/camera_viz
source "$HOME/remote-assistance-network/edge/credentials.env"
unset LK_DISABLE_NVDEC
./camera_viz.sh run "$HOME/remote-assistance-network/edge/camera-s56-1080p30.yaml" --mode xr
```

S56 presets use `sensor_id: 0` and `flip_method: 2` (180°) for the tested mounting.
Adjust these fields for another sensor or orientation; `sensor_mode: -1` allows
Argus mode selection. Use `flip_method: 0` for no rotation. Start with H.264;
H.265 compatibility for this camera has not been tested. To switch back, stop
this sender and select the matching OAK or V4L2 profiles on the two sites.

A short Orin→LiveKit→Edge test on 2026-10-14 settled at 30 encoded fps near
15 Mbps, with zero native rejections or capture-queue drops. The automatic
NVIDIA decoder delivered 649 distinct 1920×1080 images (about 26.1 fps over the
observed receive interval). Sender acceptance does not establish loss-free
delivery or 30 fps at the viewer. The received image confirmed the 180° correction.
The operator subsequently confirmed the S56 camera working in the full stack.
One native LiveKit SDK panic occurred near shutdown when the receiver ended
first; two follow-up sender shutdowns completed cleanly. For now, stop the camera
sender before the viewer. Subscriber-disconnect recovery remains a known issue
to investigate; these short checks do not establish production reliability.

### Network setup for two computers

This profile supports Linux hosts with direct IPv4 connectivity over a LAN or
VPN. Both computers must be able to reach the Edge Compute address. Being on
the same Wi-Fi SSID does not prove this: guest networks may isolate clients or
block UDP. If that happens, use an approved network/VPN or ask the network team
for connectivity; NGINX cannot bypass client isolation. Internet NAT traversal
and TURN deployment are not included in this profile.

**0. Set the site IP addresses and Robot Site username.** On each computer, run
`ip -brief -4 address` and identify its address on the network connecting the
two sites (for example, the Wi-Fi interface). Once both addresses are known,
set these variables on **both Edge Compute and Robot Site**, using the same
values on each system. Replace `[linux_username_on_robot_site]` with your Linux
login username on Robot Site (without the brackets). You can find that username
by running `whoami` in a host terminal on Robot Site:

```bash
# Replace these example addresses with your computers' current addresses.
export EDGE_IP=10.29.91.247
export ROBOT_IP=10.29.91.174
export ROBOT_USER="[linux_username_on_robot_site]"
```

The following setup commands reuse these variables. They apply to the current
terminal and its child processes; repeat this block in each new terminal where
you run commands that use them. If an address changes, update the variables on
both systems and the firewall rules; if the Edge address changes, regenerate
the bundles as described in step 1. Changing a variable alone does not update
existing configurations.

**1. Generate or refresh the bundles on Edge Compute.** Install the package as above
and ensure `openssl` is available:

```bash
cd ~/IsaacTeleop
examples/remote_assistance/.venv/bin/python -m remote_assistance.network_setup \
  --edge-ip "$EDGE_IP" \
  --output "$HOME/remote-assistance-network" \
  --hours 24
```

This creates two private directories:

- `edge/`: LiveKit and NGINX configurations, TLS keys/certificates, edge bridge,
  viewer and probe tokens, and application YAML files.
- Both sites also receive matching `camera-oak-*.yaml` and `camera-s56-*.yaml` presets.
- `robot/`: robot bridge and camera tokens, application YAML files and the public
  test CA certificate. It contains no server signing key or TLS private key.

The generator copies the current camera settings, selects a unique room, and
issues separate participant tokens. Tokens expire after 24 hours by default
(`--hours` accepts 1–168); test TLS certificates expire after seven days.
Files are private and ignored by Git. **Never commit the generated bundles.**
If a previous bundle exists, it is automatically backed up and replaced. Stop
running clients and the network server before regenerating. Camera settings come
from `--camera-config`; edits made only in the old bundle must be reapplied.
Continue with step 2, then restart services and source the new `credentials.env`
in each client terminal.

**2. Transfer only `robot/` to Robot Site.** With Robot bridge and camera clients
stopped, run this single command on Edge Compute after step 1:

```bash
bash examples/remote_assistance/deploy_robot_bundle.sh
```

The script uses the exported `ROBOT_USER` and `ROBOT_IP` variables from step 0.
It defaults to `~/remote-assistance-network`; pass another bundle directory as
its only argument if needed. SSH may prompt for your Robot Site password.
No separate Thor terminal or script installation on Thor is needed.

The script streams the Robot configuration/credential files and generated camera presets over SSH into
a private staging directory. After transfer succeeds, it preserves any existing
bundle at `~/remote-assistance-robot.backup.<unique suffix>/robot` and installs
the new one at `~/remote-assistance-robot`. Failed uploads leave the existing
bundle intact and report the staging path. Retry this command after resolving
the error. Run only one transfer at a time. Restart clients only after success,
using the new `credentials.env`. Server keys remain on Edge Compute.

Do not run the generator independently on both machines: they need matching
server credentials, room and CA. The generated `credentials.env` works after
relocation, and selects its adjacent CA certificate through `SSL_CERT_FILE`.
Source it only in the dedicated bridge/camera terminals; it changes that
process's certificate trust. No system-wide CA installation or TLS verification
bypass is required. The headset still uses the existing CloudXR connection and
certificate; it does not connect to this LiveKit signaling endpoint.

**3. Start the network server on Edge Compute.** This profile includes NGINX
for TLS signaling.
First stop the localhost SFU when intentionally switching profiles (this
interrupts its current sessions):

```bash
# Process 0 — Essential — LiveKit SFU
cd ~/IsaacTeleop
docker compose -f examples/remote_assistance/compose.yaml down
docker compose -f "$HOME/remote-assistance-network/edge/compose.yaml" up -d --force-recreate
```

`--force-recreate` works on first launch and reloads current certificates and
credentials on later launches. Repeating it interrupts existing LiveKit sessions.

The generated deployment uses these default ports on Edge Compute:

| Port | Purpose | Accessible from Robot Site? |
|---|---|---|
| TCP 8443 | NGINX HTTPS/WSS signaling | Yes |
| UDP 7882 | LiveKit WebRTC media and data | Yes |
| TCP 7881 | LiveKit WebRTC fallback | Yes |
| TCP 7880 | LiveKit HTTP behind NGINX | No; loopback only |

Permit the three client-facing ports through the host firewall and network
policy. ZMQ 5556–5559 stays local and need not be opened across sites. NGINX
proxies signaling only; encrypted WebRTC video/control travels directly between
each client and the SFU. The server advertises the selected Edge IP for media,
so using a different interface's address can break media even if signaling works.

**4. Configure the firewall on Edge Compute.** In a host terminal, first check
whether Ubuntu's UFW is active:

```bash
sudo ufw status verbose
```

If it reports `Status: active`, allow the Robot Site to reach the three ports:

```bash
sudo ufw allow proto tcp from "$ROBOT_IP" to "$EDGE_IP" port 8443
sudo ufw allow proto udp from "$ROBOT_IP" to "$EDGE_IP" port 7882
sudo ufw allow proto tcp from "$ROBOT_IP" to "$EDGE_IP" port 7881
sudo ufw status numbered
```

These rules take effect immediately and persist across reboots; no reload is
needed. Update them if either computer's address changes. To remove a rule later,
use the same command with `ufw delete allow` instead of `ufw allow`.

If UFW reports `Status: inactive`, it is not blocking these connections: proceed
to step 5. Do not enable it just for this test; enabling a firewall requires
accounting for existing SSH and CloudXR access too. If another firewall manager
or corporate policy controls the computer, use that system's equivalent rules.

**5. Check the server from Edge Compute.** Run in an Edge Compute terminal:

```bash
curl --noproxy '*' --connect-timeout 5 \
  --cacert "$HOME/remote-assistance-network/edge/ca.crt" \
  "https://${EDGE_IP}:8443/"
```

Expect `OK`, confirming TCP/TLS reachability, not media connectivity. If curl
reports error **60** after refreshing a bundle, rerun step 3 to recreate the
server with the current certificates, then retry. For error **77**, check that
the Edge CA file exists and is readable. Do not bypass verification with `-k`.

**6. Configure the firewall on Robot Site.** In a host terminal on **Robot Site**,
check UFW's status and outgoing policy:

```bash
sudo ufw status verbose
```

If UFW is inactive, or its outgoing policy is `allow`, continue to step 7.
No new inbound server ports are needed with the usual UFW policy of allowing
outgoing traffic and established replies. If its outgoing policy is `deny`,
permit outbound access explicitly:

```bash
sudo ufw allow out proto tcp to "$EDGE_IP" port 8443
sudo ufw allow out proto udp to "$EDGE_IP" port 7882
sudo ufw allow out proto tcp to "$EDGE_IP" port 7881
```

These commands configure the computer's firewall only. Network isolation or
upstream restrictions require the network administrator's help. The generated
containers use host networking, so the Edge rules are host input rules, not Docker
port-forwarding rules. See [Ubuntu's UFW documentation](https://ubuntu.com/server/docs/firewalls/).

**7. Check reachability from Robot Site**, before starting robot control:

```bash
curl --noproxy '*' --connect-timeout 5 \
  --cacert "$HOME/remote-assistance-robot/ca.crt" \
  "https://${EDGE_IP}:8443/"
```

The command should return `OK`, confirming TCP/TLS reachability (not media
connectivity). On Robot Site, use `~/remote-assistance-robot/ca.crt`, not the
Edge Compute path `~/remote-assistance-network/edge/ca.crt`.

If curl reports error **77**, check that the robot certificate exists and is
readable. If missing, rerun **Step 2’s robot-bundle transfer from Edge Compute**,
then retry. A timeout suggests routing, firewall or Wi-Fi isolation; certificate
verification errors suggest a mismatched bundle/IP or expired certificate.
Do not use `curl -k` to bypass verification.

**8. Start the processes on Robot Site first.** Keep the Edge network server
from step 3 running. MuJoCo (process 0) is required when using the simulator;
omit it when using a real robot. SONIC (process 1) is required on
Robot Site for this robot deployment workflow. Start it and the robot bridge
below; add the camera sender when using Camera Televiz. Run each process in its
own terminal and leave it running while you move to Edge Compute in step 9.

For simulation, start MuJoCo in a **Robot Site host** terminal:

```bash
# Process 0 — Robot — MuJoCo sim2sim (simulation only)
cd ~/GR00T-WholeBodyControl
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
```

In a separate **Robot Site host** terminal, enter the SONIC development container:

```bash
# Process 1 — Robot — SONIC Policy Deployment / ZMQ Manager
cd ~/GR00T-WholeBodyControl/gear_sonic_deploy
export TensorRT_ROOT=$HOME/TensorRT
./docker/run-ros2-dev.sh
```

Wait for the container shell. Then, **inside the container**, run:

```bash
# Process 1 — Robot — SONIC Policy Deployment / ZMQ Manager (inside container)
cd /workspace/g1_deploy
source scripts/setup_env.sh
./deploy.sh --cp policy/low_latency/model \
  --obs-config policy/low_latency/observation_config.yaml \
  --input-type zmq_manager sim
```

Use `real` instead of `sim` for your physical-robot workflow. Keep this terminal
running and open a separate host terminal for the robot bridge:

```bash
# Process 2 — Robot — robot bridge
cd ~/IsaacTeleop
source "$HOME/remote-assistance-robot/credentials.env"
examples/remote_assistance/.venv/bin/python -m remote_assistance bridge \
  "$HOME/remote-assistance-robot/session.yaml" --role robot
```

```bash
# Process 3 — Camera Televiz — camera sender (separate Robot Site host terminal)
cd ~/IsaacTeleop
source "$HOME/remote-assistance-robot/credentials.env"
examples/remote_assistance/.venv/bin/python -m remote_assistance camera \
  "$HOME/remote-assistance-robot/camera.yaml"
```

**Camera sender uses `camera`, not `bridge --role edge`.** The latter is a
control-data process that belongs on Edge Compute. A connection attempt to
`ws://127.0.0.1:7880` indicates the localhost development profile was selected.

Adjust `device` in the Robot Site camera YAML for that machine's camera before
starting the sender. Match width/height/fps in both camera YAML files to a
supported capture mode. For a camera-free network test, add `--synthetic` to
the sender and use the probe in step 9 instead of the viewer.

**9. Start the processes on Edge Compute.** The network server (process 0) is
already running from step 3. For Robot control, start the edge bridge (process 1)
below; for Camera Televiz, start the viewer (process 2). Then start CloudXR / Pico
Teleop Streamer (process 3) with the command below. Run each process in its own
terminal.

```bash
# Process 1 — Robot — edge bridge
cd ~/IsaacTeleop
source "$HOME/remote-assistance-network/edge/credentials.env"
examples/remote_assistance/.venv/bin/python -m remote_assistance bridge \
  "$HOME/remote-assistance-network/edge/session.yaml" --role edge
```

```bash
# Process 2 — Camera Televiz — viewer
cd ~/IsaacTeleop/examples/camera_viz
source .venv/bin/activate
source ~/.cloudxr/run/cloudxr.env
source "$HOME/remote-assistance-network/edge/credentials.env"
./camera_viz.sh run "$HOME/remote-assistance-network/edge/camera.yaml" --mode xr
```

For the camera-free network test, run this probe instead of the viewer. The
probe does not require CloudXR / Pico Teleop Streamer:

```bash
cd ~/IsaacTeleop
source "$HOME/remote-assistance-network/edge/credentials.env"
examples/remote_assistance/.venv/bin/python -m remote_assistance probe \
  "$HOME/remote-assistance-network/edge/probe.yaml" --seconds 15 --synthetic
```

Finally, start the XR input process in a separate Edge Compute terminal:

```bash
# Process 3 — Essential — CloudXR / Pico Teleop Streamer
cd ~/GR00T-WholeBodyControl
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py \
  --manager --input-source isaac-teleop \
  --port 5558 --zmq_feedback_port 5559
```

After it starts, the teleoperator can connect the headset's CloudXR.js web client
to the CloudXR runtime on Edge Compute, using the existing headset connection
workflow. Keep this process and the selected bridges/viewer running while
operating. Process 3 is not required for the headless probe or window-only camera
viewing.

Verify changing pixels, then test robot control with MuJoCo and observe bridge
counters in both directions. A stable optical baseline on one computer does
not establish latency or loss behavior over Wi-Fi. Test headset removal and
network interruption in simulation before physical deployment.

**Stop the network server on Edge Compute** with:

```bash
docker compose -f "$HOME/remote-assistance-network/edge/compose.yaml" down
```

This stops both NGINX and LiveKit, not the Pico/SONIC processes. The localhost
Compose shutdown command does not stop the separate network profile.

### Pico Manager waits forever after a body-tracking extension error

If startup reports `no support found for requested extension: XR_FB_body_tracking`
followed by this repeated message:

> **`waiting for Isaac Teleop body data (connect the headset to CloudXR)...`**

Check which environment started the shared CloudXR runtime. The camera_viz and SONIC teleop environments
can contain different native CloudXR builds even when both report the same
`isaaccapture` package version. Attaching to a runtime without the requested
extension prevents Pico Manager from creating its OpenXR session; reconnecting
the headset alone cannot fix that initialization failure.

For the tested mismatch, the teleop environment has the required extensions:

1. Stop Pico Manager and the Edge camera viewer with Ctrl+C. Leave LiveKit,
   the bridges and the Thor camera sender running.
2. Stop CloudXR from Edge Compute:

   ```bash
   cd ~/GR00T-WholeBodyControl
   .venv_teleop/bin/python -m isaaccapture.cloudxr.service stop
   ```

   If stopping times out, inspect the service logs before proceeding.
3. Start Pico Manager using the Edge Process 3 command **before** the viewer,
   so its environment launches the compatible CloudXR runtime.
4. Restart the viewer and reconnect the headset. Confirm the extension error
   is gone and body data arrives before attempting robot activation.

Align the native CloudXR builds in both environments for a lasting fix;
this recovery sequence addresses the tested startup-order mismatch.

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

### Experimental OAK-D-W onboard H.264 sender

This path uses the OAK's RVC2 encoder, then forwards encoded frames through Thor
into LiveKit without re-encoding. It requires DepthAI, not a V4L2 device node.
The IMX378 supports a 1080p60 sensor mode; 720p60 uses ISP downscaling.
Both profiles have been tested with the connected camera. See
[IMX378 modes](https://docs.luxonis.com/hardware/sensors/IMX378).

First synchronize the sender code to Thor; copying YAML alone is insufficient.
For a direct source transfer, run on **Edge Compute** (using the Robot Site
IP and username from network setup). These commands replace the named source
files; preserve any independent edits on Thor first:

```bash
cd ~/IsaacTeleop
scp examples/remote_assistance/remote_assistance/{camera,gst_camera,encoded_source,oak_camera}.py \
  "${ROBOT_USER}@${ROBOT_IP}:~/IsaacTeleop/examples/remote_assistance/remote_assistance/"
scp examples/remote_assistance/pyproject.toml \
  "${ROBOT_USER}@${ROBOT_IP}:~/IsaacTeleop/examples/remote_assistance/"
```

`KeyError: 'ENCODER_BACKEND_DEPTHAI'` means the running sender lacks the OAK
routing code. `depthai` is our capture backend, which publishes through LiveKit's
`PRE_ENCODED` backend; it is not a LiveKit encoder enum. Transfer the files above,
install the optional SDK below, and restart the sender.

On **Robot Site**, connect the OAK with a USB 3 data cable. Stop the existing
camera sender, then install the optional SDK in the transport environment:

```bash
cd ~/IsaacTeleop
examples/remote_assistance/.venv/bin/python -m pip install -e 'examples/remote_assistance[oak]'
```

If DepthAI reports USB permissions errors, install the
[Luxonis Linux USB rules](https://docs.luxonis.com/software/depthai/manual-install/)
and reconnect the camera. Run the sender on the host as your normal user.

#### Select an OAK profile

Network setup step 1 creates all four profiles on **Edge Compute**, in both
`~/remote-assistance-network/edge/` and `~/remote-assistance-network/robot/`.
Step 2 (`deploy_robot_bundle.sh`) creates or replaces
`~/remote-assistance-robot/` on **Robot Site**, including these profiles and its
credentials. It is a transferred bundle, not a directory created by cloning.

| Filename | Encoded resolution | FPS |
| --- | --- | --- |
| `camera-oak-720p60.yaml` | 1280×720 | 60 |
| `camera-oak-1080p60.yaml` | 1920×1080 | 60 |
| `camera-oak-4k30.yaml` | 3840×2160 | 30 |
| `camera-oak-2016x1520-60.yaml` | 2016×1520 | 60 |

For an **existing bundle**, add or reset all four presets on Edge without
rotating tokens or changing `camera.yaml`:

```bash
cd ~/IsaacTeleop
examples/remote_assistance/.venv/bin/python -m remote_assistance.camera_profiles
bash examples/remote_assistance/deploy_robot_bundle.sh
```

The transfer uses `ROBOT_USER` and `ROBOT_IP` from step 0. Stop Robot Site clients
before transferring; the helper backs up and replaces their bundle. This helper
copies configurations, not source code or dependencies. Existing token expiry
still applies. For a nondefault bundle path, pass `--bundle PATH` to the profile
generator and `PATH` to the transfer script.

Run **one sender and one viewer**, selecting the same filename on both sites.
For example, on Robot Site:

```bash
cd ~/IsaacTeleop
source "$HOME/remote-assistance-robot/credentials.env"
examples/remote_assistance/.venv/bin/python -m remote_assistance camera \
  "$HOME/remote-assistance-robot/camera-oak-720p60.yaml"
```

On Edge Compute (start Pico Manager first when using robot control):

```bash
cd ~/IsaacTeleop/examples/camera_viz
source .venv/bin/activate
source ~/.cloudxr/run/cloudxr.env
source "$HOME/remote-assistance-network/edge/credentials.env"
unset LK_DISABLE_NVDEC
./camera_viz.sh run "$HOME/remote-assistance-network/edge/camera-oak-720p60.yaml" --mode xr
```

To switch modes, stop these two camera processes and change the filename in both
commands. No Python editing, credential regeneration or server restart is needed.
Profiles preserve the template's camera name, room, identity settings and display
layout; they select H.264, 15 Mbps, and automatic decoding. Generated presets
replace any previous presets of the same names.

This experiment uses fixed 15 Mbps and no B-frames. It repeats keyframes every
0.5 seconds and waits for a fresh IDR after queue loss. DepthAI v2 does not expose
runtime encoder control here, so LiveKit bitrate/fps feedback and keyframe
requests cannot directly adjust the OAK encoder. Use a network with sufficient
capacity; adaptive encoding and H.265 remain separate follow-up work. A camera
`mxid` may be specified when more than one OAK is attached.


#### OAK higher-resolution experiments

The sender also accepts **3840×2160 at 30 fps** and **2016×1520 at 60 fps**.
The second profile uses the IMX378 2024×1520 sensor mode, cropping eight columns:
the [OAK H.264 encoder requires widths divisible by 32](https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/video_encoder).
Exact 2024×1520 encoding is not supported by this backend. 4K60 is also rejected.
Transfer the updated `oak_camera.py` to Robot Site before running these profiles.

Select the matching profiles from the table above; all four are generated and
transferred together.

Operator tests on 2026-10-12 used OAK-D-W/IMX378 onboard H.264, Thor as
Robot Site, and the Edge headset viewer:

| Encoded resolution | Reported sender fps | Payload Mbps |
| --- | --- | --- |
| 1280×720 | 60.0 | 14.88–15.31 |
| 1920×1080 | 60.0 | 14.68–15.31 |
| 3840×2160 | 30.0–30.2 | 14.85–16.37 |
| 2016×1520 | 59.2 initially, then 60.0–60.2 | 14.58–15.32 |

All supplied log samples reported zero LiveKit rejections and zero application
queue drops. The operator reported high visual quality in the headset. These
are sender measurements and subjective viewing results, not measurements of
network packet loss, decoded frame rate, or end-to-end latency.
