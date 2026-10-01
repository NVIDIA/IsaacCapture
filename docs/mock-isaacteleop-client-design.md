<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Mock IsaacTeleop client: design

## Problem

A real consumer of `isaacteleop` — e.g. Guman's `IsaacTeleopBackend`
(`groot/core/robotics/teleop/xr/backends/isaac_teleop_backend.py` in the `gr00t` repo) — needs a
real CloudXR runtime, a real OpenXR session, and a connected headset to run at all. Development,
demos, and automated testing of such a consumer all pay for that even when the thing actually
under test is downstream logic, not tracking itself.

## Two architectures considered

**Native, OpenXR-runtime-level** (the first approach taken on this branch, since superseded):
extend `src/plugins/controller_synthetic_hands`'s pattern — a native plugin that opens its own
`OpenXRSession` and injects synthetic poses into the runtime via push devices. This was abandoned
once it became clear it doesn't solve the actual problem: every path through this codebase that
opens an OpenXR session still needs a *real, responding OpenXR runtime* underneath (CloudXR's own
runtime in this repo's case — there's no Monado-style headless runtime already wired up; CI's
`CXR_PYTHON_GPU_TESTS`/`CXR_NATIVE_GPU_TESTS` lists deliberately exclude anything that needs a
live session to succeed, per `deps/cloudxr/docker-compose.test.yaml`'s own comment that such a
wait "waits forever" with no client connected). And `TeleopSession.__enter__`
(`teleop_session_manager/teleop_session.py:_enter_resources`) unconditionally opens a real
`oxr.OpenXRSession` in live mode regardless of what the pipeline contains — so even a pipeline of
pure external inputs still needs a real session open underneath it. Building a software OpenXR
runtime ICD to satisfy that is a much larger undertaking than this problem needs, and reusing the
real `HeadSource`/`ControllersSource`/`FullBodySource` conversion nodes (which was the main
appeal of that approach) doesn't actually buy anything if a real OpenXR session is still required
to drive them.

**Pure-Python package-level shim (what this branch implements):** a real consumer like Guman
never touches `isaacteleop`'s DeviceIO/OpenXR/compute-graph internals directly — it only
constructs source-node objects, wires them into an `OutputCombiner`, enters a `TeleopSession`
context manager, and calls `session.step()` once per tick to get back a `dict[str, TensorGroup]`
keyed by its pipeline's output names. That entire contract is reproducible with no native code,
no OpenXR, no CloudXR runtime at all — see `examples/mock_isaacteleop_client/`.

## What "complete" means here

This mock does not attempt to cover every symbol in the real `isaacteleop` package — it covers
exactly the surface a real consumer's code actually imports and calls, verified by reading that
consumer's source directly (Guman's `isaac_teleop_backend.py`) rather than guessing from names:

- Every import: `oxr` (side-effect only), `HeadSource`/`ControllersSource`/`HandsSource`/
  `JointStateSource`/`HapticSink`, `OutputCombiner`/`BaseRetargeter`, `PluginConfig`/
  `TeleopSessionConfig`/`TeleopSession`, `haptic_glove_device`, `TactileVectorToFingerPower`,
  `CloudXRLauncher`.
- The exact session lifecycle: construct a `CloudXRLauncher` (accepted, inert — `.stop()` is a
  no-op), build a pipeline, enter `TeleopSession` as a context manager, `step()` once per tick
  with no arguments, read the result via `dict.get(name)`.
- The exact tensor-type field layout for `HeadPose`, `ControllerInput`, and `HandInput` (field
  names, index-enum members, shapes, dtypes) — so a consumer's existing decode helpers (which
  index by `*Index` enum member, check `.is_none`, then check an inner `*_is_valid` field) work
  unmodified against synthetic data.
- Generic, positionally-indexed `JointStateSource` groups (one float per named joint) for
  plugin-fed channels like Manus sensor data or exoskeleton joint angles — a consumer only needs
  the right *count*, not real values, from these.
- Plugins (`PluginConfig`) are accepted into `TeleopSessionConfig` and never launched,
  health-checked, or validated — a consumer has no code path that reacts differently to "plugin
  didn't start" versus "that output's group is empty," so there's nothing to simulate there.

Everything else in the real `isaacteleop` surface (`SessionMode.REPLAY`, MCAP recording/replay,
`oxr_handles` session sharing, async/pipelined execution modes) is left out because no code path
in the consumer this was built against (Guman's backend) exercises it.

## Architecture

```
examples/mock_isaacteleop_client/
  pyproject.toml              # dist name "isaacteleop-mock"; package root is literally "isaacteleop"
  python/isaacteleop/
    __init__.py
    oxr.py                    # side-effect-only stub
    cloudxr.py                # CloudXRLauncher: accepts kwargs, .stop() is a no-op
    teleop_session_manager.py # PluginConfig, TeleopSessionConfig, TeleopSession
    haptic_devices/glove.py   # haptic_glove_device -> object with set_tactile()
    retargeters/tactile_retargeters.py  # TactileVectorToFingerPower, same shape
    retargeting_engine/
      deviceio_source_nodes.py  # HeadSource, ControllersSource, HandsSource, JointStateSource, HapticSink
      interface/                # OutputSelector, OutputCombiner, BaseRetargeter, type aliases
      tensor_types.py           # HeadPoseIndex, ControllerInputIndex, HandInputIndex, ...
    _synthetic.py              # TensorGroup + SyntheticPoseGenerator (internal, not imported by consumers)
  python/guman_shaped_usage_example.py   # end-to-end demo, same usage shape as Guman's backend
```

The package root is `isaacteleop/` (not namespaced under `isaaccapture_examples/` the way other
examples in this repo are) deliberately — the whole point is that it's importable as the bare
`isaacteleop` name, so installing it ahead of (or instead of) the real `isaacteleop`/`isaaccapture`
package on `PYTHONPATH` makes a consumer run against synthetic data with zero code changes on its
side.

`SyntheticPoseGenerator.sample(name)` (`_synthetic.py`) dispatches purely on the output-name
*pattern* passed to `TeleopSession.step()` — `"head"`, `"controller_{left,right}"`,
`"hand_{left,right}"`, or (via a `name -> joint_count` table built from every
`JointStateSource` in the pipeline at construction time) any other name. It never needs to know
what kind of source node produced that name. Motion is a deterministic function of elapsed wall
time only (slowly circling head, controllers tracing a small reach pattern with a breathing
trigger value, hands gently curling) — plausibly shaped, not biomechanically simulated; the goal
is exercising a consumer's real decode/control logic with zero hardware, not a realistic avatar.

## Verification

Unlike the abandoned native approach (which would have needed a full CMake+pybind11 build to run
at all), this is pure Python and was run directly in the sandbox this branch was built in:
`uv run --extra dev pytest` in `tests/python/examples/mock_isaacteleop_client/` passes 11/11, and
`examples/mock_isaacteleop_client/python/guman_shaped_usage_example.py` was run end-to-end and
confirmed to print live head/controller data at the expected cadence. The CMake/ctest wiring
(`tests/python/examples/mock_isaacteleop_client/CMakeLists.txt`, mirroring the `camera_viz` test
leaf's pattern) was written but **not verified via an actual `cmake` configure/build** — that
would require a full build of this repo, out of scope for the time available here.

## Out of scope / follow-up

- **Other consumers' call shapes.** This was built against exactly one real consumer's source
  (Guman's `isaac_teleop_backend.py`). A different consumer touching a different corner of
  `isaacteleop` (e.g. `FullBodySource`, `SessionMode.REPLAY`) would need its own symbol audit
  before this shim covers it — follow the same method (read the consumer's actual imports/calls,
  don't guess from names).
- **`FullBodySource`** — not used by Guman's backend today, so not implemented here, though
  `tensor_types.FullBodyInputIndex` is stubbed in for completeness per the schema reference.
- **Realistic motion** — see above; shape/plumbing correctness is the goal, not a motion model.
- **MCAP recording/replay through this shim** — `TeleopSessionConfig.mcap_config` isn't modeled;
  a consumer that needs record/replay of its own synthetic run would need to add that itself
  (e.g. writing `step()`'s output directly, bypassing this shim's `TeleopSession`).
