<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Implementation and verification report

## Implemented

- Standalone Viser panels: left selection inspector, right properties/files,
  floating top toolbar, and central 3D scene.
- Exact FlatBuffer load/export, schema validation, canonical OpenXR-order
  default shape, optional bounded atomic local writes, and browser downloads.
- Sharpa `TeleopSession` integration for OpenXR hands and 22 raw glove joints,
  two configurable tactile trackers, optional plugin launch, and MCAP replay.
- URDF forward kinematics, hand skeleton, tactile skinning, force colors,
  left/right display mirroring, selection, box selection, inspector edits, and
  translate/rotate gizmos.
- Explicit session state machine, producer/renderer separation with a one-slot
  latest mailbox, frozen pause view, local-shape override, and single-editor
  lease.

## Design decisions and existing-code changes

- OpenXR defines the 26 ordinals and runtime poses, but no universal bind pose.
  The new-file pose and hierarchy are documented tool-defined editor seeds.
- `TactileTaxel` has no unbound representation: all four indices must be valid,
  including zero-weight slots. Added grid taxels therefore repeat wrist index 1
  and use weights `[1, 0, 0, 0]`.
- Producer metadata fixes one `TactileShape` for a tactile collection/session.
  The tool caches it and validates each `TactileData.taxel_force` length; a
  changed producer layout requires a new collection/session.
- The only functional existing-source change is a generic verifier-backed
  `Serialized<T>::from_bytes()` binding, strictly required to load exact schema
  bytes. Parent example/test CMake lists changed only for installation/testing.

## Files changed

- `examples/tactile_visualizer/`: executable package, model/source adapters,
  URDF config, README, and this report.
- `tests/python/examples/tactile_visualizer/`: deterministic model, mock-source,
  config, and Viser construction tests.
- `src/core/schema/python/schema_serialized.h` and its tactile-shape test:
  verified standalone FlatBuffer decoding.
- Parent example/test `CMakeLists.txt`: register the new package and tests.

## Automated verification

An editable native build and the focused suite were run locally:

```bash
PYTHONPATH=examples/tactile_visualizer/python \
  .venv-tactile/bin/python -m pytest -q \
  tests/python/core/schema/test_tactile_shape.py \
  tests/python/examples/tactile_visualizer
# 33 passed
```

The native schema extension was built and installed into the test environment.
`SKIP=check-copyright-year pre-commit run --all-files` passed every enabled hook.
A three-second mock-mode server smoke run produced no application error; it was
terminated by the test timeout as intended.

The tests are hardware-free. They do not claim that a particular Sharpa URDF,
plugin build, browser/GPU, or physical sensor has been verified.

## Manual target-system checklist

1. Open offline mode, load a known shape, expand representative joints/taxels,
   export it, and byte-compare or decode the result.
2. In mock mode, confirm palm/fist motion, URDF FK, skeleton motion, taxel
   skinning, force colors, hand mirroring, pause/resume, and timeout behavior.
3. Verify click, taxel/Shift-joint rectangle selection, local/world gizmo axes,
   numeric edits, rejected invalid weights/indices, and one-editor enforcement
   with two browsers.
4. With the target Sharpa installation, verify the SDK config, plugin path,
   RAW-to-URDF mapping, tactile collection IDs, and physical hand direction.
5. Record and replay one session; confirm MCAP channel labels and final-frame
   behavior. Disconnect the producer and confirm that the frozen view remains
   editable.

## Known integration limits

- The default bind pose is an editor seed, not a standardized OpenXR bind pose
  or a glove calibration.
- Viser 1.1.1 transform-control events do not expose mouse buttons. The toolbar
  uses explicit Translate/Rotate modes rather than pretending to implement
  left-button/right-button semantics.
- Pointer, rendering, and physical-device behavior require the manual checks
  above; automated tests exercise only deterministic Python boundaries.
- Viser 1.1.1 exposes no folder-expansion callback. Shape leaf controls are
  instantiated immediately but remain visually hidden in collapsed folders.
- Live hardware, vendor URDF alignment, browser pointer ergonomics, and
  physical force calibration have not been verified.
