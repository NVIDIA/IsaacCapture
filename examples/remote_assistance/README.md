<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Remote Assistance media transport

Run a camera sender on a **client** computer and receive it on a **host** computer.
LiveKit is the included transport plugin. The transport core is separately
installable and discovers plugins through Python entry points. Applications can
install another plugin without rebuilding the core or installing LiveKit.

This example demonstrates video and opaque message transport. It does not start a
robot controller or a headset service. Camera capture/encoding is an example-side
adapter; the transport consumes complete encoded pictures. Adopting the camera
plugin reader is a separate integration.

## One-time preparation — both sites

Clone the IsaacCapture repo to both the host and client sites first and conduct this first time preparation. Set IC_DIR to the path of installation. Then run this installation on both sites.

Use Python 3.11 or newer and `uv`. Complete all three steps on **both sites**.

**1. Set the checkout path and create the environment.**

```bash
export IC_DIR="$HOME/IsaacCapture"
export TRANSPORT=livekit
cd $IC_DIR
uv venv --python 3.12 examples/remote_assistance/.venv
```

Create this environment once. If you recreate `.venv`, repeat both installation
steps below; recreating it removes its installed packages.

**2. Install the transport core and selected plugin.**

```bash
examples/remote_assistance/.venv/bin/python examples/remote_assistance/install_transport_plugins.py \
  --transport "${TRANSPORT:?Select TRANSPORT}" \
  --output "/tmp/transport-wheels-$(date +%Y%m%d-%H%M%S)" \
  --python examples/remote_assistance/.venv/bin/python
```

Run this helper with the virtual environment’s Python. The system `python3` on
some clients is Python 3.10 and lacks `tomllib`, which the installer requires.
No system Python upgrade is needed.

**3. Install the example package and verify both installations.**

The plugin installer does **not** install `isaaccapture-remote-assistance`.
This separate, required package supplies the `remote_assistance.network_setup`
command and the camera, receiver, viewer, and message entrypoints.

```bash
uv pip install --python examples/remote_assistance/.venv/bin/python \
  examples/remote_assistance
examples/remote_assistance/.venv/bin/python -m isaac_media_transport.transport_plugins
examples/remote_assistance/.venv/bin/python -m remote_assistance.network_setup --help
```

The first check should list the selected plugin; the second should print network
setup usage without a traceback. Neither check needs an existing bundle or starts
a service. If it reports `No module named 'remote_assistance'`, rerun the example
package installation immediately above using this same interpreter.

Native CUDA libraries are separate; a sender does not need the host's GPU decoder
packages. These packages are not added to the main Isaac Capture wheel by this example.

For a camera-free smoke test, use `camera-synthetic.yaml`. It generates changing
CPU frames, uses software H.264, and needs no Jetson or CUDA. For Sensing stereo on
Orin, use `camera-s56-stereo.yaml` and complete the next two prerequisites.

### Host Setup

#### GPU decode and CUDA IPC

CUDA IPC is not required but highly recommended to reduce latency on the host.

For the Sensing preset, install the NVIDIA driver and supported CUDA toolkit.
Select the CuPy package matching the installed CUDA major; install only one variant:

```bash
# CUDA 13 example; use cupy-cuda12x instead for a compatible CUDA 12 installation.
cd $IC_DIR
uv pip install --python examples/remote_assistance/.venv/bin/python \
  'PyNvVideoCodec==2.0.2' 'cupy-cuda13x>=13,<14'
```

**Build the native LiveKit receive extension on the host.** This is required for
the supplied Sensing preset (`decoded_memory: cuda`); installing the Python
packages above does not build it. The camera-free CPU preset does not need it.

Install the Rust/Cargo and C/C++ build prerequisites described in the
[LiveKit plugin instructions](../../src/plugins/transport_livekit/README.md), then run:

```bash
cd $IC_DIR
examples/remote_assistance/.venv/bin/python src/plugins/transport_livekit/build_livekit_cuda.py \
  --build-dir "$HOME/.cache/isaac-livekit-native" \
  --output "$HOME/.local/lib/liblivekit_ffi.so" \
  --timing --playout-control
ls -lh "$HOME/.local/lib/liblivekit_ffi.so"
```

The Sensing template already sets `livekit_cuda_library` to this output path.
Its immediate-playout setting requires `--playout-control`. Complete this step
before starting the receiver. The client sender does not need this extension.
If you choose another output path, set `livekit_cuda_library` in the camera
template before generating bundles, or update the existing host `camera.yaml`.

For the diagnostic desktop viewer:

```bash
uv pip install --python examples/remote_assistance/.venv/bin/python \
  'examples/remote_assistance[viewer]'
```

The diagnostic viewer copies decoded pixels to CPU for OpenCV. It is useful for
image checks, but is not a GPU display path or an optical-latency reference.
A GPU application can consume `IpcReceiver` directly; see the
[core interface](../../src/core/media_transport/README.md).

#### GPU desktop and XR / CloudXR viewer

For low-latency desktop or headset viewing, install the existing GPU renderer
on the host:

```bash
cd $IC_DIR
examples/camera_viz/camera_viz.sh setup
uv pip install --python examples/camera_viz/.venv/bin/python src/core/media_transport
```

Follow the [camera viewer setup guide](../camera_viz/README.md) for the CUDA,
Vulkan and CloudXR prerequisites. This is video-only: no robot policy or Pico
Manager is required. The IPC adapter uses the core package, not a second
LiveKit connection or decoder.

### Client Setup

#### Sensing / Argus capture

In applikcations with the Sensing cameras running on Argus adapter board, use the
camera vendor's driver package matching the board and JetPack release.
After each reboot, run its module/PWM preparation at **30 Hz** for the supplied
30 fps preset resolutions, or at **60 Hz** for 60 fps resolutions. Then restart
`nvargus-daemon`. Confirm video devices with `v4l2-ctl --list-devices`.
Close other processes using the same sensors.

```bash
sudo apt install -y libgirepository1.0-dev libcairo2-dev pkg-config python3-dev \
  gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0
cd $IC_DIR
uv pip install --python examples/remote_assistance/.venv/bin/python \
  'examples/remote_assistance[gstreamer]'
```

The stereo preset uses sensor 0 for the left eye and sensor 1 for the right eye.
Verify your wiring by covering each lens: left/right are viewed looking forward
from behind the camera. Swap the preset’s `left.sensor_id` and `right.sensor_id`
if needed, regenerate and transfer the client bundle, then restart the sender.
Changing the template alone does not update an already generated bundle.
Check that assignment against the actual mounting. Change resolution, frame rate,
sensor mode, and PWM together when choosing another sensor mode.


## Network setup for two computers

### Exports for Both Sites

Remember to export the installed path for IsaacCapture on both sites:

```bash
export IC_DIR="$HOME/IsaacCapture"
```

Use a direct cable or a network allowing communication between the computers.
On both sites, identify the addresses of the interfaces connecting them:

```bash
ip -brief -4 address
```

Then transfer the results into export commands and apply on both sites. For example:

```bash
export HOST_IP=192.168.90.1   # Replace with the host's address on the shared network.
export CLIENT_IP=192.168.90.2 # Replace with the client's address on that network.
export CLIENT_USER=[your_client username]
```

### Host Site Setup

Install Docker with Compose and `openssl` on the host. Set the variables below
in the terminal used for generation and transfer. Repeat the relevant exports in
new terminals; shell variables do not carry between terminals.

Choose your transport method with the following export:

```bash
export TRANSPORT=livekit
export BUNDLE="$HOME/remote-assistance-$TRANSPORT"
```

Required - select the camera template from the remote_assistance folder:

```bash
# For a camera-free test, select camera-synthetic.yaml instead.
export CAMERA_CONFIG="$IC_DIR/examples/remote_assistance/camera-s56-stereo.yaml"
export LATENCY_CONFIG="$IC_DIR/examples/remote_assistance/latency.yaml"
```

Then generate the bundles with this command:

```bash
cd $IC_DIR
examples/remote_assistance/.venv/bin/python -m remote_assistance.network_setup \
  --transport "$TRANSPORT" --host-ip "${HOST_IP:?Set HOST_IP}" \
  --output "$BUNDLE" --camera-config "$CAMERA_CONFIG" \
  --session-config examples/remote_assistance/session.yaml \
  --latency-config "$LATENCY_CONFIG"
```

Generation creates `host/` and `client/`, with scoped tokens and TLS certificates.
Existing bundles are backed up. Tokens last 24 hours by default; certificates last
seven days. Regeneration requires transferring the new client bundle, recreating
the server containers, and restarting clients with the new credentials.
Generated bundles contain credentials and must never be committed.

Start the host's network server:

```bash
docker compose -f "$BUNDLE/host/compose.yaml" up -d --force-recreate
```

#### Transfer the bundle and check signaling

Use the same host setup terminal so `BUNDLE`, `CLIENT_USER`, and the addresses
are still set:

```bash
ssh "${CLIENT_USER:?Set CLIENT_USER}@${CLIENT_IP:?Set CLIENT_IP}" \
  'mkdir -p "$HOME/remote-assistance-client" && chmod 700 "$HOME/remote-assistance-client"'
scp -r "$BUNDLE/client/." "$CLIENT_USER@$CLIENT_IP:remote-assistance-client/"
```

Test the LiveKit bridge using this command:

```bash
curl --fail --silent --show-error --noproxy '*' --connect-timeout 5 \
  --write-out '\nLiveKit signaling HTTP status: %{http_code}\n' \
  --cacert "$BUNDLE/host/ca.crt" \
  "https://${HOST_IP:?Set HOST_IP}:8443/"
```

Expect `OK` followed by `LiveKit signaling HTTP status: 200` on its own line.
A successful TLS check confirms
signaling reachability; it does not prove that UDP media can pass. Continue on
the client below and run its TLS check before starting the camera sender.

### Firewall Setup

Configure these rules once for the selected addresses and ports. Repeat this
setup if those values or your firewall policy change.
These Ubuntu UFW commands allow this connection through specific ports while
keeping the rest of the firewall in place. They use the default generated ports;
if you changed ports during generation, use those values here too.

| Host port | Purpose |
|---|---|
| TCP 8443 | Verified TLS signaling |
| TCP 7881 | RTC fallback when UDP cannot connect |
| UDP 7882 | Video and data media traffic |

TCP 7880 is the server's loopback-only HTTP backend; no external opening is needed.
The client also needs TCP 22 for the SSH/SCP bundle transfer.

#### On the host

Use the `HOST_IP` and `CLIENT_IP` values selected in Host Site Setup. Both
variables must be set in the terminal running these commands. If `ufw` is not
installed, install it with `sudo apt install -y ufw` first.

```bash
sudo ufw status verbose
sudo ufw allow in proto tcp from "$CLIENT_IP" to "$HOST_IP" port 8443
sudo ufw allow in proto tcp from "$CLIENT_IP" to "$HOST_IP" port 7881
sudo ufw allow in proto udp from "$CLIENT_IP" to "$HOST_IP" port 7882
# Permit server media/ICE traffic and bundle transfer even with restricted egress.
sudo ufw allow out proto udp from "$HOST_IP" port 7882 to "$CLIENT_IP"
sudo ufw allow out proto tcp from "$HOST_IP" to "$CLIENT_IP" port 22
sudo ufw status verbose
```

#### On the client

Use the same selected `HOST_IP` and `CLIENT_IP` values in the client terminal.
Both variables must be set there too. If `ufw` is not installed, install it with
`sudo apt install -y ufw` first.

```bash
sudo ufw status verbose
sudo ufw allow out proto tcp from "$CLIENT_IP" to "$HOST_IP" port 8443
sudo ufw allow out proto tcp from "$CLIENT_IP" to "$HOST_IP" port 7881
sudo ufw allow out proto udp from "$CLIENT_IP" to "$HOST_IP" port 7882
# LiveKit uses dynamic client UDP ports; allow traffic from the host's media port.
sudo ufw allow in proto udp from "$HOST_IP" port 7882 to "$CLIENT_IP"
sudo ufw allow in proto tcp from "$HOST_IP" to "$CLIENT_IP" port 22
sudo ufw status verbose
```

With standard UFW connection tracking, TCP replies to the client's outgoing
connections are allowed automatically. The client does not listen on TCP 8443
or 7881. An active UFW applies these rules immediately. If its status is
`inactive`, UFW is already allowing traffic; there is no need to enable it for
this example. Earlier explicit deny rules may take precedence: inspect
`sudo ufw status numbered` if traffic remains blocked.

These commands configure UFW on these computers. A separate network firewall,
VPN policy, or Wi-Fi client isolation must also allow communication between them.
See the [LiveKit port reference](https://docs.livekit.io/transport/self-hosting/ports-firewall/)
and [Ubuntu UFW instructions](https://ubuntu.com/server/docs/security-firewall/).

### Client Site Processes

On the client, set `IC_DIR` to its checkout and repeat `HOST_IP` for the network
check.

```bash
export IC_DIR="$HOME/IsaacCapture"
export HOST_IP=192.168.90.1 # Same host address selected above.
export CLIENT_BUNDLE="$HOME/remote-assistance-client"
```

The bundle transfer from the host supplied configuration, not source code or packages.
Test the LiveKit bridge using this command:

```bash
curl --fail --silent --show-error --noproxy '*' --connect-timeout 5 \
  --write-out '\nLiveKit signaling HTTP status: %{http_code}\n' \
  --cacert "$CLIENT_BUNDLE/ca.crt" \
  "https://${HOST_IP:?Set HOST_IP}:8443/"
```

Expect `OK` followed by `LiveKit signaling HTTP status: 200` on its own line.
This checks signaling reachability, not UDP video delivery. HTTP errors return a
nonzero exit status; `000` means no HTTP response was received.

#### Camera Sender
Next start the camera sender process:

```bash
source "$CLIENT_BUNDLE/credentials.env"
cd $IC_DIR
examples/remote_assistance/.venv/bin/python -m remote_assistance camera \
  "$CLIENT_BUNDLE/camera.yaml"
```

The Sensing sender reports encoded fps separately for `cam.left` and `cam.right`.
Both should reach about 30 fps. A zero encoded count is a capture problem even
if the room connection succeeds. Allow startup bandwidth estimation and decoder
initialization to settle before judging the frame rate; the first usable stereo
pairs can arrive substantially later than the first encoded frame.

### Host Site Processes

#### Camera Reciever
For the CUDA preset, start a receiver in one terminal:

```bash
# repeat the exports from before in a new terminal
export IC_DIR="$HOME/IsaacCapture"
export TRANSPORT=livekit
export BUNDLE="$HOME/remote-assistance-$TRANSPORT"
cd $IC_DIR

# source the env and start the receiver
source "$BUNDLE/host/credentials.env"
examples/remote_assistance/.venv/bin/python -m remote_assistance receiver \
  "$BUNDLE/host/camera.yaml"
```

#### Camera Viewer — desktop or XR

Keep the sender and CUDA receiver running. Choose either viewer below.
For the CUDA preset, use the GPU desktop viewer to keep frames on the GPU.
In another host terminal:

```bash
# repeat the exports from before in a new terminal
export IC_DIR="$HOME/IsaacCapture"
export TRANSPORT=livekit
export BUNDLE="$HOME/remote-assistance-$TRANSPORT"
cd $IC_DIR

examples/camera_viz/.venv/bin/python -m isaaccapture_examples.camera_viz \
  "$BUNDLE/host/camera.yaml" --source ipc --mode window
```

This uses the same GPU renderer and latency markers as the XR route below.
Use Ctrl+C to stop it. Complete the GPU viewer prerequisites above first.

The **OpenCV diagnostic viewer** is an alternative for image inspection. It
copies frames to CPU, converts their colors and joins stereo images before
submission, adding work that the GPU viewer avoids. Use the GPU viewer for
latency comparisons. To run the diagnostic viewer instead, stop the GPU viewer
and use the same exports:

```bash
source "$BUNDLE/host/credentials.env"
examples/remote_assistance/.venv/bin/python -m remote_assistance viewer \
  "$BUNDLE/host/camera.yaml" --ipc --scale 1.0
```

For the synthetic CPU preset, skip the separate receiver and omit `--ipc` from
this diagnostic command. `--headless` counts received samples without a window.
`--seconds 30` bounds any example process for a smoke test. Press `q` while the
diagnostic window has focus, close its window, or use Ctrl+C to stop it.

For **XR / CloudXR**, stop the desktop viewer and run this alternative using
the same host camera YAML and receiver socket:

```bash
export IC_DIR="$HOME/IsaacCapture"
export TRANSPORT=livekit
export BUNDLE="$HOME/remote-assistance-$TRANSPORT"
cd $IC_DIR
examples/camera_viz/.venv/bin/python -m isaaccapture_examples.camera_viz \
  "$BUNDLE/host/camera.yaml" --source ipc --mode xr
```

Connect the headset using the CloudXR connection instructions printed by the
viewer. It starts the CloudXR runtime and presents the left and right tracks
to their corresponding eyes, keeping decoded frames on the GPU. Headset-to-host
reachability and CloudXR's own network requirements are separate from the
client-to-host LiveKit firewall rules above; see the linked viewer guide.
Use Ctrl+C to stop the XR viewer. This IPC route requires the CUDA receiver;
it does not apply to the synthetic CPU preset.

## Optional opaque-message test

Opaque messages let applications send their own control commands, status updates,
or sensor metadata alongside video without the transport interpreting the payload.

This optional test checks bidirectional message delivery between the two sites.
Keep the LiveKit server running and start both commands within the same 30-second
window. Open a new terminal on each **physical computer** and use its block below;
adjust the checkout and bundle paths if you installed elsewhere.

```bash
# Host computer
export IC_DIR="$HOME/IsaacCapture"
export TRANSPORT=livekit
export BUNDLE="$HOME/remote-assistance-$TRANSPORT"
cd $IC_DIR
source "$BUNDLE/host/credentials.env"
examples/remote_assistance/.venv/bin/python -m remote_assistance messages \
  "$BUNDLE/host/session.yaml" --role host --seconds 30
```

```bash
# Client computer (Orin)
export IC_DIR="$HOME/IsaacCapture"
export CLIENT_BUNDLE="$HOME/remote-assistance-client"
cd $IC_DIR
source "$CLIENT_BUNDLE/credentials.env"
examples/remote_assistance/.venv/bin/python -m remote_assistance messages \
  "$CLIENT_BUNDLE/session.yaml" --role client --seconds 30
```

This exchanges demonstration status messages only. Robot command schemas,
controller arming, watchdog actions and recovery policy belong to the application.

## Transport plugins and timing

See the [transport contract](../../src/core/media_transport/README.md) for payload
ownership, timestamp units, feedback, and external-plugin registration. See the
[LiveKit plugin](../../src/plugins/transport_livekit/README.md) for SDK dependencies
and optional native receive support.

### One measurement policy for both sites

Instrumentation settings live in **`$IC_DIR/examples/remote_assistance/latency.yaml`**,
not in camera templates. Use the same file with S56, SHW5G, or other camera
configurations and with any transport that supplies the stage markers. Camera
resolution, sensor IDs and encoding settings still belong in the camera template.

```yaml
enabled: true                # The single master switch
file_enabled: false          # Retain JSONL stage records and qualified results
terminal_enabled: true       # Periodic viewer reports
hud_enabled: false           # Persistent XR panel
report_interval_s: 5
run_id: ""                   # Bundle generation assigns a shared unique ID
log_dir: /tmp
clock:
  client: "${CLIENT_USER}@${CLIENT_IP}"
  client_python: python3
  max_age_s: 2
  max_uncertainty_ms: 5
  drift_ppm: 1000
```

The review preset enables online measurements and terminal reports by default.
To turn all instrumentation off, change **only `enabled` to `false`**. Select any
combination of file, terminal and HUD outputs. HUD-only mode sets `hud_enabled:
true`, `terminal_enabled: false`, and `file_enabled: false`; the master remains
true. File logging and the HUD remain off in the shipped preset.

On the host, set `CLIENT_USER` and `CLIENT_IP` as in Network setup and verify
non-interactive SSH access (or put the actual `user@address` in `clock.client`):

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 "$CLIENT_USER@$CLIENT_IP" true
```

The host must trust the client's SSH host key and have a working login key. Do
not disable host-key verification. The helper requires only the client's Python
standard library and does not require sudo or change either computer's clock.

Generate the bundles using `--latency-config "$LATENCY_CONFIG"`, transfer the
client bundle, and restart the processes with the new credentials. Generation
resolves the SSH address and places the same `latency.yaml` and run ID in both
bundles. The camera YAML references that sidecar by a relative path. Templates
and already generated bundles are separate: editing a template does not change
a running process. Use a new generated run ID when changing test conditions.

Start the sender, receiver (for CUDA IPC) and chosen viewer normally. **No
separate latency-monitor terminal is required.** The viewer starts clock probes
and joins markers in the background. An SSH or telemetry failure marks latency
unavailable and retries; it does not stop video.

### What the partial photon-to-photon total measures

The primary total spans **camera frame available to the sender → viewer
submission begins**, using the same endpoints as the prototype's online
comparison. It estimates the bridge's measurable portion of the path, including
encoding, transport, decoding, pairing, IPC and viewer preparation. It excludes
sensor exposure/readout before the capture callback and physical display/XR
presentation after submission. The earlier approximately 36 ms result is a
measured baseline, not a value this code assumes or guarantees.

| Available stage markers | What they help locate |
| --- | --- |
| `capture_available`, `encoder_submit`, `encoded_ready`, `access_unit_ready` | Capture handoff and encoding on the client |
| `publish_begin`, `transport_submit`, `transport_available` | Application/transport handoff across the clock-mapped network boundary |
| `decode_begin`, `decode_ready`, `pair_ready` | Host decoding and stereo coordination |
| `ipc_export_begin`, `ipc_export_ready`, `ipc_import_begin`, `ipc_copy_ready` | CUDA IPC handoff |
| `viewer_take`, `viewer_ready`, `viewer_submit_begin`, `viewer_submit_end` | Viewer preparation and submission-call cost |

A span covering missing intermediate markers is labelled a **partial span**, not
attributed to one stage. Totals require matching capture and viewer endpoints.
Submission-end timing is diagnostic and is not added to the primary total.
Native transport diagnostics, where supplied, overlap other spans and must not
be summed with them. The synthetic software path can measure the total without
exposing native encoding/network sub-stages. Capture plugins must timestamp the
actual frame-availability boundary and preserve frame identity for comparable
measurements; configuration alone cannot invent missing markers.

The clock estimator reuses authenticated four-timestamp SSH probes. It bounds
the client-to-host monotonic-clock offset using round-trip measurements and an
explicit relative drift allowance. It does **not** assume symmetric network
latency. Stale mappings, excessive uncertainty, disagreeing clock samples,
process restarts and incomplete frame joins invalidate results. The displayed
estimate is conditional on that drift allowance; inspect bounds and uncertainty
in recorded results when comparing small differences. This is not a guarantee
of exposure synchronization or full optical latency.

### Outputs and overhead

- **Terminal:** the final viewer periodically reports each track's partial total,
  rolling mean/p95 (up to 300 samples per interval), sample count and available stage intervals. Missing or stale
  totals say unavailable rather than zero.
- **HUD:** the IPC/XR viewer shows qualified per-eye partial totals in a persistent
  head-locked panel, with the measured clock uncertainty shown as ± milliseconds. It does not add a HUD to OpenCV or headless operation.
- **Files:** each process writes `umb-<run_id>-<role>.jsonl` under its site's
  `log_dir`. The viewer also writes `umb-<run_id>-online.jsonl` with per-frame
  qualified intervals, uncertainty and summaries. Headless mode has no viewer
  submission, so it cannot produce this total.

File logging is optional: online joins use bounded same-user memory buffers and
Unix sockets, with metadata fetched over SSH. No pixels travel through the
measurement channel. Buffer overruns or restarts discard incomplete joins;
telemetry failures do not block media. Enabling instrumentation still costs
clock reads, metadata processing and periodic probes; HUD rendering and file
I/O add their own costs. Recheck overhead and the optical baseline on your
hardware before choosing production defaults.

Clock qualification and event collection live in the transport core, independent
of camera SDKs and transport implementations. This example's stage schema is
video-specific. Audio, tactile and control streams can reuse the clock mapping
and collection strategy, but need their own frame/message identities and clearly
defined start/end stages; they are not measured by this example yet.
