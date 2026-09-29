<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Haptic Feedback Examples

Runnable demos for the Isaac Teleop haptic device-output path: a tactile-shaped
signal feeds a per-device retargeter and a `HapticSink`, which `TeleopSession`
flushes to an `IHapticDevice` adapter each frame.

| Example | What it demonstrates |
| --- | --- |
| `controller_haptic_example.py` | Pull a controller trigger to rumble that same controller — the minimal in-process `HapticSink` → `ControllerHapticDevice` wiring. |
| `hand_pinch_haptic_example.py` | Pinch a fingertip toward the thumb to vibrate a haptic glove — the cross-process path (`HapticSink` → `PushTensorHapticDevice`, pushing a `HapticCommand` to a glove plugin such as Manus). |

```bash
uv run controller_haptic_example.py     # motion-controller rumble
uv run hand_pinch_haptic_example.py      # haptic glove (needs a glove plugin running)
```

Both connect through the CloudXR / OpenXR runtime, so start the runtime first.
The full architecture, run instructions, and how to add a new haptic device are
in the official documentation:

**Haptic Feedback** — <https://nvidia.github.io/IsaacCapture/main/device/haptic_feedback.html>
(source: [`docs/source/device/haptic_feedback.rst`](../../../docs/source/device/haptic_feedback.rst))
