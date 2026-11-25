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

Use Python 3.11 or newer and `uv`. Run this in each site's checkout:

```bash
export IC_DIR="$HOME/IsaacCapture"
export TRANSPORT=livekit
cd "${IC_DIR:?Set IC_DIR to this site's checkout}"
uv venv --python 3.12 examples/remote_assistance/.venv
python3 examples/remote_assistance/install_transport_plugins.py \
  --transport "${TRANSPORT:?Select TRANSPORT}" \
  --output "/tmp/transport-wheels-$(date +%Y%m%d-%H%M%S)" \
  --python examples/remote_assistance/.venv/bin/python
uv pip install --python examples/remote_assistance/.venv/bin/python examples/remote_assistance
examples/remote_assistance/.venv/bin/python -m isaac_media_transport.transport_plugins
```

Installation builds only the requested plugin and common core. Native CUDA
libraries are separate; a sender does not need the host's GPU decoder packages.
The packages are not added to the main Isaac Capture wheel by this example.

For a camera-free smoke test, use `camera-synthetic.yaml`. It generates changing
CPU frames, uses software H.264, and needs no Jetson or CUDA. For Sensing stereo on
Orin, use `camera-s56-stereo.yaml` and complete the next two prerequisites.

### Client — Sensing / Argus capture

Use the camera vendor's driver package matching the board and JetPack release.
After each reboot, run its module/PWM preparation at **30 Hz** for the supplied
2560×1984 at 30 fps preset, then restart `nvargus-daemon`. Confirm video devices
with `v4l2-ctl --list-devices`; device enumeration alone does not prove capture.
Close other processes using the same sensors.

```bash
sudo apt install -y libgirepository1.0-dev libcairo2-dev pkg-config python3-dev \
  gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0
cd "${IC_DIR:?Set IC_DIR}"
uv pip install --python examples/remote_assistance/.venv/bin/python \
  'examples/remote_assistance[gstreamer]'
```

The stereo preset uses sensor 1 for the left eye and sensor 0 for the right eye.
Check that assignment against the actual mounting. Change resolution, frame rate,
sensor mode, and PWM together when choosing another sensor mode.

### Host — optional GPU decode and CUDA IPC

For the Sensing preset, install the NVIDIA driver and supported CUDA toolkit.
Select the CuPy package matching the installed CUDA major; install only one variant:

```bash
# CUDA 13 example; use cupy-cuda12x instead for a compatible CUDA 12 installation.
cd "${IC_DIR:?Set IC_DIR}"
uv pip install --python examples/remote_assistance/.venv/bin/python \
  'PyNvVideoCodec==2.0.2' 'cupy-cuda13x>=13,<14'
```

Build the optional encoded-receive extension following the
[LiveKit plugin instructions](../../src/plugins/transport_livekit/README.md).
Place the resulting library at `~/.local/lib/liblivekit_ffi.so`, or set
`livekit_cuda_library` in the camera template to its actual host path. The Sensing
preset requests immediate playout, so build with `--playout-control`.

For the diagnostic desktop viewer:

```bash
uv pip install --python examples/remote_assistance/.venv/bin/python \
  'examples/remote_assistance[viewer]'
```

The diagnostic viewer copies decoded pixels to CPU for OpenCV. It is useful for
image checks, but is not a GPU display path or an optical-latency reference.
A GPU application can consume `IpcReceiver` directly; see the
[core interface](../../src/core/media_transport/README.md).

## Network setup for two computers

Use a direct cable or a network allowing communication between the computers.
On both sites, identify the addresses of the interfaces connecting them:

```bash
ip -brief -4 address
```

### Host Site Setup

Install Docker with Compose and `openssl` on the host. Set the variables below
in the terminal used for generation and transfer. Repeat the relevant exports in
new terminals; shell variables do not carry between terminals.

```bash
export IC_DIR="$HOME/IsaacCapture"
export TRANSPORT=livekit
export BUNDLE="$HOME/remote-assistance-$TRANSPORT"
export HOST_IP=192.168.90.1   # Replace with the host's address on the shared network.
export CLIENT_IP=192.168.90.2 # Replace with the client's address on that network.
export CLIENT_USER=your_client_username
export CAMERA_CONFIG="$IC_DIR/examples/remote_assistance/camera-s56-stereo.yaml"
# For a camera-free test, select camera-synthetic.yaml instead.

cd "$IC_DIR"
examples/remote_assistance/.venv/bin/python -m remote_assistance.network_setup \
  --transport "$TRANSPORT" --host-ip "${HOST_IP:?Set HOST_IP}" \
  --output "$BUNDLE" --camera-config "$CAMERA_CONFIG" \
  --session-config examples/remote_assistance/session.yaml
```

Generation creates `host/` and `client/`, with scoped tokens and TLS certificates.
Existing bundles are backed up. Tokens last 24 hours by default; certificates last
seven days. Regeneration requires transferring the new client bundle, recreating
the server containers, and restarting clients with the new credentials.
Generated bundles contain credentials and must never be committed.

Transfer the client bundle and start the server:

```bash
ssh "${CLIENT_USER:?Set CLIENT_USER}@${CLIENT_IP:?Set CLIENT_IP}" \
  'mkdir -p "$HOME/remote-assistance-client" && chmod 700 "$HOME/remote-assistance-client"'
scp -r "$BUNDLE/client/." "$CLIENT_USER@$CLIENT_IP:remote-assistance-client/"
docker compose -f "$BUNDLE/host/compose.yaml" up -d --force-recreate
curl --noproxy '*' --connect-timeout 5 --cacert "$BUNDLE/host/ca.crt" \
  "https://${HOST_IP:?Set HOST_IP}:8443/"
```

Expect an HTTP response from LiveKit. Permit TCP 8443 (TLS signaling), TCP 7881
(RTC fallback), and UDP 7882 (media) from the client on the host firewall.
HTTP 7880 is bound to loopback. A successful TLS check alone does not prove UDP
media reachability. Default ports can be overridden during generation.

### Client Site Processes

On the client, set `IC_DIR` to its checkout and repeat `HOST_IP` for the network
check. The bundle transfer supplies configuration, not source code or packages;
install this same source revision on both computers before testing.

```bash
export IC_DIR="$HOME/IsaacCapture"
export HOST_IP=192.168.90.1 # Same host address selected above.
export CLIENT_BUNDLE="$HOME/remote-assistance-client"
curl --noproxy '*' --connect-timeout 5 --cacert "$CLIENT_BUNDLE/ca.crt" \
  "https://${HOST_IP:?Set HOST_IP}:8443/"
source "$CLIENT_BUNDLE/credentials.env"
cd "$IC_DIR"
examples/remote_assistance/.venv/bin/python -m remote_assistance camera \
  "$CLIENT_BUNDLE/camera.yaml"
```

The Sensing sender reports encoded fps separately for `cam.left` and `cam.right`.
Both should reach about 30 fps. A zero encoded count is a capture problem even
if the room connection succeeds. Allow startup bandwidth estimation and decoder
initialization to settle before judging the frame rate; the first usable stereo
pairs can arrive substantially later than the first encoded frame.

### Host Site Processes

For the CUDA preset, start a receiver in one terminal:

```bash
export IC_DIR="$HOME/IsaacCapture"
export TRANSPORT=livekit
export BUNDLE="$HOME/remote-assistance-$TRANSPORT"
source "$BUNDLE/host/credentials.env"
cd "$IC_DIR"
examples/remote_assistance/.venv/bin/python -m remote_assistance receiver \
  "$BUNDLE/host/camera.yaml"
```

In another terminal, repeat the exports and start the diagnostic viewer:

```bash
export IC_DIR="$HOME/IsaacCapture"
export TRANSPORT=livekit
export BUNDLE="$HOME/remote-assistance-$TRANSPORT"
source "$BUNDLE/host/credentials.env"
cd "$IC_DIR"
examples/remote_assistance/.venv/bin/python -m remote_assistance viewer \
  "$BUNDLE/host/camera.yaml" --ipc --scale 0.5
```

For the synthetic CPU preset, skip the separate receiver and omit `--ipc` from
the viewer command. `--headless` counts received samples without a window.
`--seconds 30` bounds any example process for a smoke test. Press `q` while the
viewer window has focus, close its window, or use Ctrl+C to stop it.

### Optional opaque-message test

In another terminal on each site, source that site's `credentials.env`, then run:

```bash
# Host
examples/remote_assistance/.venv/bin/python -m remote_assistance messages \
  "$BUNDLE/host/session.yaml" --role host --seconds 30
```

```bash
# Client
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

Tracing remains opt-in. Add this under `livekit` in the camera template **before**
generating and transferring bundles:

```yaml
latency:
  enabled: true
  run_id: my-trial
  log_dir: /tmp
```

The same run ID is used on both sites; the processes choose distinct sender,
receiver and viewer filenames. The desktop viewer records submission around
`imshow`; that does not mean the display has scanned out the image. Headless
operation has no viewer-submission markers. These events are local-clock stage markers, not
an automatically qualified cross-machine or photon-to-photon latency. Clock
qualification and optical endpoints require separate measurement.
