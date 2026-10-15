<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Synthetic XR Teleoperation — Preview

Automate Franka cube stacking through virtual XR controllers and the native WebXR/CloudXR/retargeting/IK path. No physical headset or manual demonstration is required. CloudXR host requirements still apply.

Start with the [illustrated workflow and bring-your-own-task guide](../../docs/source/teleop/synthetic_xr.rst). This README contains the setup and run commands.

## Franka stacking

This integration adds read-only observation hooks around Lab's `record_demos.py`. It does not replace the native task pipeline or write joint actions. `stack_logic` reads cube and tool poses, converts targets into XR controller poses and grip/trigger input, and waits for matching native actions.

The integration is built for the prepared Lab revision `cb9628bee016c532feb5fc36c3b012fe707bed1b`, its `IsaacContrib-Stack-Cube-Franka-IK-Abs` task, CloudXR 6.3.0 and the installed Isaac Teleop 1.5 backend. The new client is based on Isaac Capture `e07a2cc8c3fcbc8d0aac9f9bdc7919a792983e44`. These component identities are distinct; a new-client check is not a fresh validation of every current SDK backend. See `VALIDATION.txt` for the measured result and limits before treating another combination as supported.

In the configured Lab Python environment, install the optional example dependencies:

```bash
python -m pip install -e './examples/synthetic_xr[franka]'
```

Build this checkout's WebXR client using `docs/source/development/build_from_source/webxr.rst`. Serve its `build/` directory using `TELEOP_WEB_CLIENT_STATIC_DIR` and the service's `--host-client` option; the public prebuilt client does not include this preview. Start the existing CloudXR service with its Lab WebXR profile and the newly built client. For automated setup, use OOB hub-only mode; no ADB is required. Do not start a competing runtime if Lab or the service already owns one.

In that Lab environment, run the native recording script through the observation wrapper, using an output directory that does not already exist:

```bash
SYNTHETIC_XR_VISIBLE=1 python -m isaaccapture_examples.synthetic_xr.franka.observe_record_demos \
  --lab-root /absolute/path/to/IsaacLab \
  --output /absolute/path/to/run/observations \
  --task IsaacContrib-Stack-Cube-Franka-IK-Abs \
  --dataset_file /absolute/path/to/run/franka.hdf5 \
  --mcap_record_path /absolute/path/to/run/controllers.mcap \
  --num_demos 1 --xr --viz kit --cloudxr_env none --no-auto_launch_cloudxr env.seed=42
```

Open a dedicated desktop browser profile with loopback remote debugging enabled, using the actual client URL and `#/sim/isaaclab` application fragment:

```bash
google-chrome --user-data-dir=/tmp/synthetic-xr-profile \
  --remote-debugging-address=127.0.0.1 --remote-debugging-port=9222 \
  'https://HOST:PORT/client/?syntheticXR=1&immersiveMode=vr#/sim/isaaclab'
```

If GPU context creation fails while running Lab, the validated fallback uses `--use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader --disable-gpu-compositing` and client query parameters `perEyeWidth=512&perEyeHeight=448`. This is a functional configuration, not a rendering-performance baseline.

Connect the client. Keep exactly one tab with `syntheticXR=1` in this browser. Run the task logic in a second terminal:

```bash
python -m isaaccapture_examples.synthetic_xr.franka.stack_logic \
  --cdp-url http://127.0.0.1:9222 \
  --feedback-url http://127.0.0.1:8765/state \
  --output /absolute/path/to/run/logic
```

The script starts teleoperation and supplies controller movement and grip/trigger input automatically. You do not need to select a controller or move it using the panel.

The script uses the public `window.syntheticXR` API via CDP. CDP is needed only for this external Python driver; interactive desktop controls need no debugging port. Feedback is served on loopback and must be fresh. The script stops after one native successful export and checks settled cube geometry. Stage and native-action acknowledgement traces are retained in the output directories.

The observer requires Lab's `create_environment`, `create_environment_config`, `setup_teleop_device` and `process_success_condition` integration hooks plus the task's cube/tool names. It fails if those hooks or shapes are incompatible; it does not silently substitute a task. Use `--stop-after approach` or `--stop-after grasp` to inspect intermediate stages without claiming a completed episode.

## Optional: inspect desktop controller input

Build this checkout's WebXR client using `docs/source/development/build_from_source/webxr.rst`, then serve its `build/` directory through the existing CloudXR service's `--host-client` option and `TELEOP_WEB_CLIENT_STATIC_DIR`. Do not use the public prebuilt client for this preview; it does not contain this branch's controls.

From the repository root:

```bash
python -m pip install -e ./examples/synthetic_xr
synthetic-xr --client-url 'https://HOST:PORT/client/'
```

Replace the URL with the service's printed URL, retaining its query and application fragment. Connect, open the synthetic input panel and select a controller. For applications that require a start command, use **Start teleoperation**, which invokes the existing client handler. Change position or orientation, engage Grip / clutch, or change Trigger. Run controller sweep generates a nine-second sequence and releases both buttons when complete.

Record input and Save input trace export a synthetic command trace. Load trace and Replay once reproduce it through IWER. The existing browser recorder and native MCAP use different formats and injection boundaries. See `docs/source/teleop/synthetic_xr.rst` for the API and trace contract.

## Optional: native controller MCAP

Install this checkout's existing `examples/mcap_record_replay` package first. Start its controller recording with a connected desktop client and run the sweep during capture:

```bash
python -m isaaccapture_examples.mcap_record_replay.record_controller 15 controllers.mcap
python -m isaaccapture_examples.synthetic_xr.replay_once controllers.mcap
```

The second command opens a native replay viewer on loopback port 8090, plays once, leaves the viewer open for five seconds, then closes. No live XR session is required during replay.

For the supported Isaac Teleop 1.5 compatibility stack, select the installed SDK's original example explicitly:

```bash
python -m isaaccapture_examples.synthetic_xr.replay_once controllers.mcap \
  --legacy-example-dir /absolute/path/to/IsaacCapture/examples/mcap_record_replay/python
```

This invokes that version's native replay pipeline; it does not emulate MCAP decoding or alias the newer SDK to the older one. Its replay loop polls at 60 Hz: a lower-rate recording can finish yielding payloads before the duration timer expires, leaving an invalid/empty view until exit. This preview verifies payload order and values, not reproduction of original sample timing.

## Scope

Controller input only in this preview. Generated hand/body inputs and SO101 portability are separate follow-ups. No headset tracking, glove protocol or physical robot qualification is implied. Stop input releases synthetic buttons; use the application stop control for task shutdown. Simulation ground truth drives the stacking example, so its output is synthetic demonstration data.
