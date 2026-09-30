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

**Choose a profile:** the checked-in YAML files use localhost and development
credentials for the single-computer test. For two computers, first follow
[Network setup](#network-setup-for-two-computers) below to generate private site
bundles, then use those bundles in the same processes. Run one SFU on Edge Compute;
Robot Site connects to it. Keep ZMQ endpoints local to each site.

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

Capture size and rate come from `livekit-webcam.yaml` (currently 1920×1080 at
60 fps, MJPG). For 1440p30, set width/height/fps to 2560/1440/30. Keep sender and
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

**1. Generate the bundles once on Edge Compute.** Install the package as above
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
- `robot/`: robot bridge and camera tokens, application YAML files and the public
  test CA certificate. It contains no server signing key or TLS private key.

The generator copies the current camera settings, selects a unique room, and
issues separate participant tokens. Tokens expire after 24 hours by default
(`--hours` accepts 1–168); test TLS certificates expire after seven days.
Files are private and ignored by Git. **Never commit the generated bundles.**
The command refuses to overwrite an existing output directory. To renew tokens
or change the Edge IP, generate a new bundle directory, stop the old network
profile, transfer the new robot bundle, and restart server and clients together.
This rotates credentials and the test CA. Token expiry affects new joins;
it is not an immediate disconnect/revocation mechanism for existing sessions.

**2. Transfer only `robot/` to Robot Site**, using an approved secure transfer
method such as SSH/SCP. Run this on Edge Compute using the variables from step 0:

```bash
scp -r "$HOME/remote-assistance-network/robot" \
  "${ROBOT_USER}@${ROBOT_IP}:~/remote-assistance-robot"
```

Do not run the generator independently on both machines: they need matching
server credentials, room and CA. The generated `credentials.env` works after
relocation, and selects its adjacent CA certificate through `SSL_CERT_FILE`.
Source it only in the dedicated bridge/camera terminals; it changes that
process's certificate trust. No system-wide CA installation or TLS verification
bypass is required. The headset still uses the existing CloudXR connection and
certificate; it does not connect to this LiveKit signaling endpoint.

**3. Start the network server on Edge Compute.** First stop the localhost SFU
when intentionally switching profiles (this interrupts its current sessions):

```bash
cd ~/IsaacTeleop
docker compose -f examples/remote_assistance/compose.yaml down
docker compose -f "$HOME/remote-assistance-network/edge/compose.yaml" up -d
```

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

**5. Configure the firewall on Robot Site.** In a host terminal on **Robot Site**,
check UFW's status and outgoing policy:

```bash
sudo ufw status verbose
```

If UFW is inactive, or its outgoing policy is `allow`, continue to step 6.
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

**6. Check reachability from Robot Site**, before starting robot control:

```bash
curl --connect-timeout 5 \
  --cacert "$HOME/remote-assistance-robot/ca.crt" \
  "https://${EDGE_IP}:8443/"
```

A response confirms TCP/TLS reachability, not media connectivity. A timeout
suggests routing, firewall or Wi-Fi isolation; a certificate error suggests the
wrong bundle/IP or an expired certificate. Do not use `curl -k` to bypass it.

**7. Use the generated configurations in the existing process layout.** Start
Pico, SONIC and MuJoCo exactly as above; only the bridge/camera commands change.
On Edge Compute, in separate terminals:

```bash
# Process 2 — Robot — edge bridge
cd ~/IsaacTeleop
source "$HOME/remote-assistance-network/edge/credentials.env"
examples/remote_assistance/.venv/bin/python -m remote_assistance bridge \
  "$HOME/remote-assistance-network/edge/session.yaml" --role edge
```

```bash
# Process 3 — Camera Televiz — viewer
cd ~/IsaacTeleop/examples/camera_viz
source .venv/bin/activate
source ~/.cloudxr/run/cloudxr.env
source "$HOME/remote-assistance-network/edge/credentials.env"
./camera_viz.sh run "$HOME/remote-assistance-network/edge/camera.yaml" --mode xr
```

On Robot Site, in separate terminals:

```bash
# Process 2 — Robot — robot bridge
cd ~/IsaacTeleop
source "$HOME/remote-assistance-robot/credentials.env"
examples/remote_assistance/.venv/bin/python -m remote_assistance bridge \
  "$HOME/remote-assistance-robot/session.yaml" --role robot
```

```bash
# Process 3 — Camera Televiz — camera sender
cd ~/IsaacTeleop
source "$HOME/remote-assistance-robot/credentials.env"
examples/remote_assistance/.venv/bin/python -m remote_assistance camera \
  "$HOME/remote-assistance-robot/camera.yaml"
```

Adjust `device` in the Robot Site camera YAML for that machine's camera. Match
width/height/fps in both camera YAML files to a supported capture mode. For a
camera-free network test, add `--synthetic` to the sender and run this on Edge
Compute instead of the viewer:

```bash
cd ~/IsaacTeleop
source "$HOME/remote-assistance-network/edge/credentials.env"
examples/remote_assistance/.venv/bin/python -m remote_assistance probe \
  "$HOME/remote-assistance-network/edge/probe.yaml" --seconds 15 --synthetic
```

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
