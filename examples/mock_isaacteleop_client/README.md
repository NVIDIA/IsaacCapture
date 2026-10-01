<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Mock IsaacTeleop client

A pure-Python drop-in shim for `isaacteleop`: no CloudXR runtime, no OpenXR, no native plugins,
no headset, no browser tab. Lets a real consumer of `isaacteleop` (e.g. Guman's
`IsaacTeleopBackend`, `groot/core/robotics/teleop/xr/backends/isaac_teleop_backend.py`) run
end-to-end against synthetic head/controller/hand/joint-state data.

See [`../../docs/mock-isaacteleop-client-design.md`](../../docs/mock-isaacteleop-client-design.md)
for the design rationale, and `GumanIsaacTeleopUsage.md` (a sibling research doc, not checked
into this repo) for the complete, code-grounded spec of the real `isaacteleop` surface this
package implements against.

## Install

```bash
cd examples/mock_isaacteleop_client
uv pip install -e .
```

Install this **instead of** the real `isaacteleop`/`isaaccapture` package (or put it earlier on
`PYTHONPATH`) in whatever environment should run headless.

## Run the example

```bash
cd examples/mock_isaacteleop_client
uv run python/guman_shaped_usage_example.py
```

Prints head/controller/hand/joint-state data at ~1 Hz, built exactly the way Guman's
`_build_session_config()`/`build_snapshot()` build and read it (construct source nodes, wire an
`OutputCombiner`, enter `TeleopSession`, call `step()` once per tick).

## What's covered

Every `isaacteleop` symbol Guman's backend imports: `oxr` (side-effect only), `HeadSource`,
`ControllersSource`, `HandsSource`, `JointStateSource`, `HapticSink`, `OutputCombiner`,
`BaseRetargeter`, `PluginConfig`, `TeleopSessionConfig`, `TeleopSession`, `CloudXRLauncher`,
`haptic_glove_device`, `TactileVectorToFingerPower`. See `isaacteleop/` for the implementation and
`tests/python/examples/mock_isaacteleop_client/` for the test suite.

## What's not covered

`SessionMode.REPLAY`, `McapRecordingConfig`/`McapReplayConfig`, `oxr_handles` (external session
sharing), `teleop_control_pipeline`, and any plugin health-check signal — none of these are on
Guman's actual call path. Hand/body motion is deterministic and plausibly-shaped, not a
biomechanical simulation.
