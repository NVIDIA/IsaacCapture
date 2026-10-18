<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Synthetic XR Teleoperation

Run Franka stacking through virtual XR controllers and the native WebXR → CloudXR → retargeting → Lab IK path. Start with the [illustrated tutorial](../../docs/source/teleop/synthetic_xr.rst). The browser shows streamed simulation; task logic does not animate the robot locally.

## Supported setup

This integration targets **one local Linux x86-64 simulation environment** with Isaac Capture built from this branch, Lab `cb9628bee016c532feb5fc36c3b012fe707bed1b`, task `IsaacContrib-Stack-Cube-Franka-IK-Abs`, and CloudXR `6.3.0`. See [VALIDATION.txt](VALIDATION.txt) for exact executed identities and results.

You need a supported NVIDIA GPU/driver, a graphical desktop, Chrome with its normal sandbox enabled, and the product's CloudXR network and license prerequisites. The Python driver and observer share a local filesystem. Keep CDP and observations on loopback; no headset or ADB is required.

### Install in a fresh environment

Use this candidate's complete source checkout or source archive as `CAPTURE_ROOT`. The public prebuilt client does not contain this feature. Install Git, Git LFS, [uv](https://docs.astral.sh/uv/getting-started/installation/), Node.js 20 or later, and the [Isaac Capture source-build prerequisites](../../docs/source/development/build_from_source/index.rst). Use a new directory for Lab:

```bash
export CAPTURE_ROOT=/absolute/path/to/IsaacCapture
export LAB_ROOT=/absolute/path/to/IsaacLab
export CLOUDXR_INSTALL_DIR=/absolute/path/to/cloudxr
export SYNTHETIC_RUN=/absolute/path/to/a/new/run

git clone https://github.com/isaac-sim/IsaacLab.git "$LAB_ROOT"
git -C "$LAB_ROOT" checkout cb9628bee016c532feb5fc36c3b012fe707bed1b
cd "$LAB_ROOT"
uv sync --locked --extra teleop --extra isaacsim --extra ov
source .venv/bin/activate
```

The pinned Lab checkout manages its own native dependencies and explicit dependency overrides. Use its locked environment rather than combining packages from an existing installation. Install this branch's SDK **after** Lab setup; another `uv sync` would restore Lab's older SDK selection.

Build the SDK wheels, or use the matching prebuilt Linux/Python 3.12 wheels distributed with this candidate:

```bash
cd "$CAPTURE_ROOT"
cmake -B build -DISAAC_TELEOP_PYTHON_VERSION=3.12 \
  -DBUILD_VIZ=OFF -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF \
  -DENABLE_CLOUDXR_EXP_BUNDLE=OFF -DENABLE_CLOUDXR_BUNDLE_CHECK=ON \
  -DCMAKE_INSTALL_PREFIX="$CAPTURE_ROOT/install"
cmake --build build --parallel 6
cmake --install build
uv pip install 'isaaccapture[retargeters-lite,cloudxr,ui]==1.7+local' \
  --find-links="$CAPTURE_ROOT/install/wheels"
uv pip install './examples/synthetic_xr[franka]'
```

Both the `isaaccapture` wheel and its `isaacteleop` transition wheel must be available in `install/wheels`. The feature's `[franka]` extra supplies its browser transport and numerical dependencies. This recipe builds the SDK without Televiz; the Franka example uses Lab's Kit viewport.

Build the client with its committed dependency lockfile:

```bash
cd "$CAPTURE_ROOT"
printf 'CXR_HOST_VOLUME_PATH=%s\n' "$CLOUDXR_INSTALL_DIR" > deps/cloudxr/.env
source scripts/setup_cloudxr_env.sh
scripts/download_cloudxr_sdk.sh
cd deps/cloudxr/webxr_client
npm ci
npm run build
cd "$CAPTURE_ROOT"
```

Keep `deps/cloudxr/webxr_client/build/`. In each subsequent terminal, activate `$LAB_ROOT/.venv/bin/activate` and set the same four paths above.

### Trust the local service certificate

For localhost development on Linux, generate the service's certificate before opening Chrome:

```bash
python -c 'import os; from pathlib import Path; from isaaccapture.cloudxr.wss import cert_paths_from_dir, ensure_certificate; ensure_certificate(cert_paths_from_dir(Path(os.environ["CLOUDXR_INSTALL_DIR"]) / "certs"))'
```

Import `server.crt` as a trusted self-signed **server** certificate using Chrome's certificate manager. Alternatively, with `libnss3-tools` installed, follow [Chromium's Linux certificate instructions](https://chromium.googlesource.com/chromium/src/+/main/docs/linux/cert_management.md):

```bash
nss_dir="$HOME/.local/share/pki/nssdb"
if [[ -d "$HOME/.pki/nssdb" ]]; then nss_dir="$HOME/.pki/nssdb"; fi
mkdir -p "$nss_dir"
if [[ ! -f "$nss_dir/cert9.db" ]]; then
  certutil -N --empty-password -d "sql:$nss_dir"
fi
certutil -A -d "sql:$nss_dir" -n synthetic-xr-localhost -t 'P,,' \
  -i "$CLOUDXR_INSTALL_DIR/certs/server.crt"
```

Trust only the certificate generated on your host and keep `server.key` private. Restart the dedicated browser after changing trust. The example uses `https://localhost`; a remote hostname needs a certificate valid for that hostname. To remove this trust later, run `certutil -D -d "sql:$nss_dir" -n synthetic-xr-localhost`.

## 1. Start the service

Check that another service does not own the configured CloudXR ports. Run in a dedicated terminal:

```bash
export TELEOP_WEB_CLIENT_STATIC_DIR="$CAPTURE_ROOT/deps/cloudxr/webxr_client/build"
export TELEOP_OOB_HUB_ONLY=1
python -m isaaccapture.cloudxr.service run \
  --cloudxr-install-dir "$CLOUDXR_INSTALL_DIR" \
  --cloudxr-env-config "$LAB_ROOT/source/isaaclab_teleop/isaaclab_teleop/cloudxrjs-cloudxr.env" \
  --setup-oob --host-client
```

Accept the CloudXR license when prompted. For unattended use after accepting its terms, the service supports `--accept-eula`. Hub-only mode starts OOB without USB/ADB. Leave this terminal running; use its printed host/port if different from the defaults below.

## 2. Start Lab and the observer

In a second terminal, load the environment generated by the running service. Do not start a competing CloudXR runtime:

```bash
set -a
source "$CLOUDXR_INSTALL_DIR/run/cloudxr.env"
set +a
SYNTHETIC_XR_VISIBLE=1 python -m isaaccapture_examples.synthetic_xr.franka.observe_record_demos \
  --lab-root "$LAB_ROOT" \
  --output "$SYNTHETIC_RUN/observations" \
  --task IsaacContrib-Stack-Cube-Franka-IK-Abs \
  --dataset_file "$SYNTHETIC_RUN/franka.hdf5" \
  --num_demos 1 --xr --viz kit --cloudxr_env none --no-auto_launch_cloudxr env.seed=42
```

The observation directory must not exist. Accept Kit's license prompt if this environment has not done so. After accepting its terms, unattended runs can set `OMNI_KIT_ACCEPT_EULA=YES`. Resolve any first-launch desktop or Kit setup dialogs before starting the task; decline optional telemetry if you do not want it. Wait for Lab's XR/scene initialization and the loopback endpoint `http://127.0.0.1:8765/state`. No robot motion starts until the task driver starts teleoperation.

## 3. Connect the browser

Use a fresh dedicated profile and the local service certificate configured in the prerequisite setup:

```bash
google-chrome --user-data-dir="$SYNTHETIC_RUN/browser-profile" \
  --remote-debugging-address=127.0.0.1 --remote-debugging-port=9222 \
  'https://localhost:48322/client/?syntheticXR=1&immersiveMode=vr&oobEnable=1&serverIP=localhost&port=48322&codec=h264#/sim/isaaclab'
```

Click **CONNECT** and wait for the Synthetic XR panel to say **Connected**. Keep exactly one `syntheticXR=1` tab. The script starts teleoperation and moves the controllers; you do not need to select or manually move a controller.

If Lab and the browser cannot create GPU contexts together, add `--use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader --disable-gpu-compositing` and append `&perEyeWidth=512&perEyeHeight=448` before the URL's `#` fragment. This is a functional fallback, not a performance baseline. Resolve browser sandbox or certificate errors using your host/browser installation guidance; disabling those protections is not part of the supported setup recipe.

## 4. Run stacking

In a third terminal with the same paths and Python environment:

```bash
python -m isaaccapture_examples.synthetic_xr.franka.stack_logic \
  --observation-dir "$SYNTHETIC_RUN/observations" \
  --output "$SYNTHETIC_RUN/logic"
```

A completed run contains a native successful HDF5 episode, `observations/exported-episode-1.json`, and a task `result.json` with `status: PASS`. The driver waits for native export, native environment closure and a zero worker exit status recorded by the observer supervisor. `stages.jsonl`, command receipts and `controller-acks.jsonl` expose the requested and received actions. `--stop-after approach` or `--stop-after grasp` is a partial diagnostic run and does not claim task completion.

## Application interface

Import `FrankaTaskSession` from `isaaccapture_examples.synthetic_xr.franka.session`. Its context attaches to the configured observer, starts teleoperation, and releases synthetic inputs on exit. By default it requests observer shutdown on exit; it does not own or terminate your browser or CloudXR service.

| Operation | Contract |
| --- | --- |
| `observe()` | Fresh schema-v1 dictionary; rejects stale feedback, a restarted observer, or a mismatched observation directory. |
| `move_tool(position, quaternion, seconds=2)` | Franka `ee_frame` target in world metres/Z-up; quaternion `[x,y,z,w]`, unit length. Interpolates XR input and checks native acknowledgement and tool convergence. |
| `set_gripper(closed=True)` | Boolean grip decision, converted to XR trigger; this does not write robot joints. |
| `wait(seconds)` | Wait in simulation seconds with a bounded wall-clock timeout. |
| `mark(name)` | Save a named observation to the task trace. |
| `wait_for_export(timeout=30)` | Wait for the native successful episode and orderly observer completion. Does not set a success flag. Use `--num_demos 1`. |

Observation fields: `run_id` identifies the observer, `episode_id` changes on reset, `steps` counts physics steps, `simulation_dt` is seconds/step, and `timestamp` is host Unix time. `ee_positions_world[0][0]` and `ee_quaternions_world[0][0]` describe the tool. `cubes[name]` contains `position_world[0]`, `quaternion_world[0]`, and `linear_velocity_world[0]`. Positions are metres; velocities are metres/second; all quaternions are xyzw. `joint_names` maps `joints[0]`. Select additional scene objects with the observer's `--objects name ...`; the `cubes` field name is retained for schema-v1 compatibility.

The adapter owns the `panda_hand` → tool offset (0.1034 m along local Z) and XR/frame transforms. It remains specific to this Franka configuration. Targets are checked for finite values and the declared workspace; this is not a collision-avoidance or reachability planner. Native episode completion ends movement immediately, including any remaining interpolation; inspect the returned observation before issuing another action. Only one task driver may attach to an observer. Use a new observation directory for every run; a failed attachment does not take over an existing driver.

For your own task, use the shipped [stack_logic.py](python/isaaccapture_examples/synthetic_xr/franka/stack_logic.py) as the runnable starting point and replace its decisions. Define a trusted Python callback `is_success(observation) -> bool` and start the observer with `--success my_task:is_success`; make `my_task` importable in the Lab environment. The callback receives a detached snapshot after each step. Use `episode_id` to reset any history. Exceptions and non-boolean returns abort the run. The callback is application code, not a sandbox.

The native recorder still requires its configured consecutive success steps (`--num_success_steps`, default 10), records state/actions, marks success and exports. An application callback changes the success criterion, not the motion pipeline. Include any required grasp/lift history, settling and release checks. A driver finishing its commands does not establish success. Without `--success`, the original native stacking criterion remains active.

## Stop and troubleshoot

The session context releases controller input and requests observer shutdown on exceptions. The observer supervisor bounds the complete run to 900 seconds by default (`--session-timeout`); timeout or forced shutdown remains a failure. For an externally interrupted driver, create `$SYNTHETIC_RUN/observations/stop` to stop the observer, or interrupt its terminal. Close the dedicated browser and interrupt the service terminal when finished. Once a session attaches, the observer stops if it receives no driver heartbeat for 30 seconds (`--driver-timeout`). Session operations refresh this heartbeat. Long application computations must call `observe()` periodically or configure an appropriate timeout. This watchdog stops the simulation; it is not a real-robot safety system. A crashed driver cannot perform normal browser cleanup.

- **Stale observation / no XR frame:** check Lab's terminal, CONNECT state and the matching client build; do not reuse a previous run's files.
- **Target not acknowledged:** inspect command/native receipts and connection state. The driver fails instead of silently moving the robot directly.
- **No native export:** inspect the success callback or native criterion. A log saying the script ended is insufficient.
- **Observer hook/configuration mismatch:** use the supported Lab revision. The adapter depends on native recording hooks and Franka scene names.

This release covers simulated controller input. Physical headset sensing, other robots, hand/body/glove input and real-robot safety are separate scopes.
