<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Quick Start

This guide walks through running a teleoperation session with an XR headset using CloudXR. By the
end you will have the retargeting pipeline processing live hand/controller data and printing gripper
commands to the terminal.

No headset? See [Synthetic XR Teleoperation](synthetic_xr.md) for desktop controller inputs and examples.

> ##### Steps
> 
> * [1. Check out code base (for examples)](#check-out-code-base-for-examples)
> * [2. Install the `isaaccapture` pip package](#install-the-isaaccapture-pip-package)
> * [3. Configure CloudXR (optional)](#configure-cloudxr-optional)
> * [4. Whitelist ports for Firewall](#whitelist-ports-for-firewall)
> * [5. Connect an XR headset](#connect-an-xr-headset)
> * [6. Run a teleop example](#run-a-teleop-example)
> * [Next steps](#next-steps)
> * [More Information](#more-information)

<a id="check-out-code-base"></a>

## 1. Check out code base (for examples)

Clone the repository and enter the project directory:

```bash
git clone https://github.com/NVIDIA/IsaacCapture.git
cd IsaacCapture
```

As a quick start guide, we don’t need to build the code base from source. However, we still need
to clone the repository for a couple quick samples to run.

<a id="install-isaaccapture-pip-package"></a>

## 2. Install the `isaaccapture` pip package

#### NOTE
Upgrading from 1.5 or earlier? The import package and the distribution were both
renamed from `isaacteleop` in 1.6. Installing `isaaccapture` automatically
replaces the legacy package and keeps old imports working temporarily.
See [Migrating to isaaccapture](../development/migration.md).

Install the package into a virtual environment. Without one, `pip install` fails on Ubuntu 24.04
with `externally-managed-environment` and `uv pip install` with `No virtual environment
found`. In a new terminal, from the repository root, create and activate one with
[uv](https://docs.astral.sh/uv/), which downloads the requested Python if the system lacks it
(Python 3.11 to 3.13 are supported). If `uv` is missing, install it first as shown in
[One time setup](../development/build_from_source/index.md#one-time-setup).

```bash
# --seed adds pip, so plain pip install also works in this venv.
uv venv --python 3.11 --seed .venv
source .venv/bin/activate
```

To use an existing venv, or a conda environment other than `base` (`uv` treats `base` as a
system Python), activate it and skip `uv venv`. Activate the environment again in every new
terminal you use for the steps below.

Install the latest stable release from PyPI:

```bash
uv pip install 'isaaccapture[cloudxr,retargeters]~=1.0'
```

Pre-release builds are published between stable releases. Picking one up takes the NVIDIA index
and an opt-in to pre-release versions:

```bash
uv pip install 'isaaccapture[cloudxr,retargeters]~=1.0' \
      --extra-index-url https://pypi.nvidia.com --prerelease=allow
```

The two commands above track the newest 1.x release. To follow this version of the documentation
(1.8.x) exactly, pin to the series it describes instead:

```default
uv pip install 'isaaccapture[cloudxr,retargeters]~=1.8.0' \
      --extra-index-url [https://pypi.nvidia.com](https://pypi.nvidia.com) --prerelease=allow
```

Instead of installing the package from PyPI, you can build from source and install the local wheel.
See [Build from Source](../development/build_from_source/index.md) for more details.

<a id="run-cloudxr-server"></a>

## 3. Configure CloudXR (optional)

The teleop examples in this guide bring CloudXR up for you when they connect —
you do **not** need to start the runtime in a separate terminal or source any
environment file. If nothing is serving yet, the example starts a CloudXR
service in the background and prints how to stop it; that service outlives the
example, so the headset stays connected from one run to the next.
The first launch downloads the CloudXR Web Client SDK and asks you to review and
accept the EULA on the terminal; answer the prompt once and the acceptance is
remembered for subsequent runs.

<!-- TODO: ``Quest3`` is the profile every WebXR client uses — Quest, PICO, and the
desktop-browser IWER emulator — so the name reads as a hardware restriction it
does not impose. Either rename the profile or add a generic alias (e.g.
``webxr``) in the runtime, then update the default and the values listed below. -->

The CloudXR runtime uses the `Quest3` device profile by default. The name is
narrower than the profile: `Quest3` serves every WebXR client — Quest, PICO,
and the desktop-browser IWER emulator — so PICO users should keep it. Apple
Vision Pro connects natively and needs `auto-native` instead; the default is
not derived from the connected headset, so set it yourself. The profile is set
on the *runtime*, so applications inherit whatever the runtime they connect to
was started with. To override it, or any other setting, write a `KEY=value`
env file and start the service with it:

```bash
echo 'NV_DEVICE_PROFILE=auto-native' > custom.env
python -m isaaccapture.cloudxr.service start --cloudxr-env-config ./custom.env
```

The same `--cloudxr-env-config` flag is available on the teleop examples under
`examples/teleop`, which register CloudXR’s launcher arguments through
`CloudXRLauncher.add_launcher_arguments()` — but it applies only when the
example is the one starting the runtime. With a service already running the
example attaches to it instead, and prints which settings it had to ignore (see
[CloudXR Service](../references/cloudxr.md)). (The ROS 2 example takes the equivalent
`cloudxr_env_config` ROS parameter instead.)
To inspect the resolved settings after startup:

```bash
cat ~/.cloudxr/run/cloudxr.env
```

#### NOTE
To manage the service yourself — check what is running, stop it, or use
launch modes like `--host-client` and `--setup-oob` — see
[CloudXR Service](../references/cloudxr.md).

#### Environment variables

| Variable                     | Default   | Description                                      | Values                                                   |
|------------------------------|-----------|--------------------------------------------------|----------------------------------------------------------|
| `NV_DEVICE_PROFILE`          | `Quest3`  | Device profile                                   | `auto-webrtc`, `auto-native`, `Quest3`, `AppleVisionPro` |
| `NV_CXR_ENABLE_PUSH_DEVICES` | `true`    | Push device overseer for hand tracking           | `true`, `false`                                          |
| `NV_CXR_FILE_LOGGING`        | `true`    | File-based logging (disable to print to console) | `true`, `false`                                          |

On Jetson Orin the launcher selects the experimental CloudXR runtime and
main-thread join automatically. You normally do not need to set
`ISAAC_TELEOP_CLOUDXR_EXP` or `ISAAC_TELEOP_CLOUDXR_JOIN_MAIN`; those are
overrides for debugging or forcing the stable path. See
[CloudXR Service](../references/cloudxr.md).

<a id="whitelist-firewall-ports"></a>

## 4. Whitelist ports for Firewall

CloudXR requires certain network ports to be open. Depending on your firewall configuration, you
might need to whitelist them manually.

### Meta Quest and Pico headsets

For **Quest and Pico headsets** (web client), at the minimum, you need to whitelist the ports
for the CloudXR runtime and wss proxy:

```bash
sudo ufw allow 47998/udp
sudo ufw allow 49100,48322/tcp
```

If you are running the web client from source (dev server), open both
ports:

```bash
sudo ufw allow 8080,8443/tcp
```

### Vision Pro client

For **Vision Pro client**, you need to whitelist the ports for the CloudXR runtime and wss proxy:

```bash
sudo ufw allow 48010,48322/tcp
sudo ufw allow 47998:48000,48005,48008,48012/udp
```

Please see the [CloudXR network setup](https://docs.nvidia.com/cloudxr-sdk/latest/requirement/network_setup.html#ports-and-firewalls) for more details for other network configurations (such as
running the CloudXR runtime and wss proxy in containerized environment; or using Vision Pro client).

<a id="connect-xr-headset"></a>

## 5. Connect an XR headset

<a id="connect-quest-pico"></a>

### Meta Quest, PICO headset, or desktop browser

No physical headset required for a quick test: open
[nvidia.github.io/IsaacCapture/client](https://nvidia.github.io/IsaacCapture/client/main/)
in a **desktop browser** — IWER (Immersive Web Emulator Runtime) loads automatically and
emulates a Meta Quest 3 headset.

For a real headset, open the same URL in your **Meta Quest or PICO browser**.

#### NOTE
The default client shows an in-XR control panel with connection information and controls
useful during development. To hide it, open **Advanced settings → In-XR comfort** before
connecting and set **Control panel when entering XR (saved per teleop application)** to
**Hide control panel**. Use the small **Show** control to reopen it in XR. See
[Web client for camera visualization](../references/camera_streaming.md#camera-streaming-web-client).

#### IMPORTANT
If using a physical headset, make sure it is updated to the latest firmware before connecting.
Older firmware may ship an outdated WebXR runtime that fails to connect or streams with reduced
functionality.

#### NOTE
Out of the box, Quest headsets stream at 72 FPS and 25 Mbps (Pico 4 Ultra at 90 FPS
and 100 Mbps). You can raise both under **Advanced settings** in the control panel,
up to 120 FPS and 200 Mbps, but on a typical 5 GHz Wi-Fi link the higher bitrates
saturate the connection and you get reprojection judder within a few minutes. The
defaults are the values that held stable over long sessions in our testing.

The client remembers your settings between sessions. A saved profile other than
`Custom` picks up the current recommended values on the next load, so updated
defaults reach you without clearing browser storage. Switch the profile to
`Custom` if you want your manual values to stick.

#### NOTE
When a teleop example starts the CloudXR service, it hosts the matching web client at
`https://<your-ip>:48322/client/` by default and prints that URL. The GitHub Pages URL above
remains available; pass `--no-host-client` to disable local hosting. Port 48322 is already
whitelisted in step [4. Whitelist ports for Firewall](#whitelist-firewall-ports). See [CloudXR Service](../references/cloudxr.md) for service
hosting and existing-service behavior.

### CloudXR web client

![CloudXR web client usage instruction](_static/cloudxr-web-client-howto.png)

**Figure:** CloudXR web client usage instruction

### Privacy warning

![Browser privacy warning for self-signed certificate](_static/cloudxr_accept_cert_not_private.png)

**Figure:** Browser privacy warning for self-signed certificate

### Certificate accepted

![Certificate accepted page](_static/cloudxr_accept_cert_accepted.png)

**Figure:** Certificate accepted page

As illustrated in the figure above, there are 3 steps to connect to your headset:

1. Enter the IP address of the workstation running CloudXR
2. Accept the self-signed SSL certificate, which was created automatically during [3. Configure CloudXR (optional)](#run-cloudxr-server):
   - Click the **Click https://<ip>:48322/ to accept cert** link that appears on the page.
   - In the new tab, you will see a **“Your connection is not private”** warning. Click **Advanced**, then **Proceed to <ip> (unsafe)**.
   - Once accepted, the page will show **Certificate Accepted**. Navigate back to the CloudXR.js client page.

#### IMPORTANT
**Jetson Orin:** when the CloudXR runtime runs on Orin, set **Video Codec** to
**H.264** in the CloudXR web client.

1. Click **Connect** to begin teleoperation.

#### NOTE
On PICO 4 Ultra, the web client refreshes after session ends and connection
failures by default, keeping the last error visible until a stream connects.
See [web client refresh troubleshooting](../references/cloudxr.md#cloudxr-web-client-refresh)
for refresh settings and retry behavior.

For advanced usage and troubleshooting of CloudXR, see the [CloudXR documentation](https://docs.nvidia.com/cloudxr-sdk/latest/index.html) for more
details.

### Offline / air-gapped use

On **first run**, the launcher syncs the published web client into
`~/.cloudxr/static-client/` (override with `TELEOP_WEB_CLIENT_STATIC_DIR`):
`index.html`, `bundle.js`, and `bundle.emulator.js`. Subsequent runs
are offline once those files are cached.

For a **true air-gapped machine**, copy the full `build/` output (or the
matching directory from [nvidia.github.io/IsaacCapture/client](https://nvidia.github.io/IsaacCapture/client/main/)) into
`~/.cloudxr/static-client/` on the air-gapped host before the first run.

The source code for the web client is in the [deps/cloudxr/webxr_client/](https://github.com/NVIDIA/IsaacCapture/tree/main/deps/cloudxr/webxr_client) directory.
To build the web client from source, see [Build WebXR Client](../development/build_from_source/webxr.md).

<a id="connect-apple-vision-pro"></a>

### Apple Vision Pro

For Apple Vision Pro, you will need to build and install the Isaac XR Teleop Sample Client. Follow
the instructions in the [Isaac XR Teleop Sample Client for Apple Vision Pro](https://github.com/isaac-sim/isaac-xr-teleop-sample-client-apple) repository to build
and install the sample client on your Apple Vision Pro.

#### NOTE
You will need v3.0.0 or newer of the [Isaac XR Teleop Sample Client for Apple Vision Pro](https://github.com/isaac-sim/isaac-xr-teleop-sample-client-apple)
to connect to Isaac Teleop.

<a id="run-teleop-example"></a>

## 6. Run a teleop example

Install the teleop examples into the environment from step 2, then run the
simplified gripper retargeting example. This demonstrates the full
pipeline: reading XR controller input via CloudXR, retargeting it through the
`GripperRetargeter`, and printing the resulting gripper command values:

```bash
uv pip install -e ./examples/teleop
python -m isaaccapture_examples.teleop.gripper_retargeting_example_simple
```

Once running, squeeze the **right controller trigger** to control the gripper.
`GripperRetargeter` emits binary commands: `+1.00` means open and `-1.00`
means closed. With the default configuration, trigger values above `0.5`
close the gripper; values at or below `0.5` open it. The analog trigger value
is thresholded by the retargeter, so intermediate output values are not expected.
You should see periodic status output:

```text
============================================================
Gripper Retargeting - Squeeze triggers to control grippers
============================================================

Gripper command: +1.00 = open, -1.00 = closed (trigger > 0.5).
Running until Ctrl+C.
[  0.5s] Right: 1.00
[  1.0s] Right: -1.00
[  1.5s] Right: 1.00
...
```

The example runs until you press **Ctrl+C**. To stop automatically after a
chosen duration, measured after session startup, pass `--duration`:

```bash
python -m isaaccapture_examples.teleop.gripper_retargeting_example_simple --duration 20
```

`--duration 0` also runs until Ctrl+C. Both a timed run and Ctrl+C announce
that the session is closing, then print `Session closed.` after cleanup.
Stream or cleanup errors still propagate; a browser stream error is not the
expected completion message. For troubleshooting, retain the Python traceback,
CloudXR runtime logs, and browser error and stop-reason codes.

For proportional controller input, read `ControllerInputIndex.TRIGGER_VALUE`
or use `SO101GripperRetargeter` for SO-101 jaw closedness in `[0, 1]`
(`0` open, `1` closed); see [Retargeters: SO-101 (5-DOF arm)](../references/retargeting/so101.md).

To try other examples, run another module of `isaaccapture_examples.teleop`
the same way; [examples/teleop/README.md](https://github.com/NVIDIA/IsaacCapture/blob/main/examples/teleop/README.md) lists them all. For instance:

- `se3_retargeting_example` — maps hand or controller poses to
  end-effector poses (absolute or relative), with the same `--duration` and Ctrl+C behavior
- `dex_bimanual_example` — bimanual dexterous hand retargeting
- `gripper_retargeting_example` — full gripper example with more
  configuration options

## Next steps

![Isaac Lab](_static/isaaclab.jpg)

**Teleoperation in Isaac Lab**

Follow instructions in [Teleoperation and Imitation Learning with Isaac Lab Mimic](https://isaac-sim.github.io/IsaacLab/develop/source/overview/imitation-learning/teleop_imitation.html#teleoperation-imitation-learning) to know
more about how to collect demonstrations with Isaac Lab and how to augment them with Isaac
Lab Mimic and train imitation learning policies.

If you are new to Isaac Lab, follow instructions in [Isaac Lab Quick Start](https://isaac-sim.github.io/IsaacLab/develop/source/setup/quickstart.html) to get started.

![Isaac ROS](_static/isaacros.png)

**Teleoperation with Isaac ROS**

For a complete [Isaac ROS](https://nvidia-isaac-ros.github.io) pipeline, follow [Teleoperation with Isaac GR00T and Unitree G1](https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/real-robot-workflow/real-teleop.html)
— an end-to-end workflow that combines Isaac Teleop, CloudXR, and ROS 2 to teleoperate a
Unitree G1 humanoid. You validate the setup in MuJoCo, then deploy on real hardware over a
Jetson AGX Thor, as a precursor to data collection and imitation learning.

*ROS is a trademark of Open Robotics.*

## More Information

- [Teleop Session](teleop_session.md) — learn how `TeleopSession` works and how to build
  custom retargeting pipelines
- [Televiz](../development/televiz.md) — visualize robot camera and sensor feeds in an XR headset with
  the Televiz compositor (and share a single XR session with your teleop pipeline)
- [Build from Source](../development/build_from_source/index.md) — build the C++ core, Python bindings, and plugins
  from source

<!-- References -->
