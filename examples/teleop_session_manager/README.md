<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TeleopSessionManager examples

Four standalone demos of the `TeleopSession` API. In a venv set up as in
[Running an example](../README.md#running-an-example), install once, then run
whichever one you want:

```bash
uv pip install -e ./examples/teleop_session_manager
python -m isaaccapture_examples.teleop_session_manager.teleop_session_example
```

| Module | What it shows |
| --- | --- |
| `teleop_session_example` | Velocity from hand-wrist and controller motion, driven by `session.step()` with the synthetic hands plugin |
| `teleop_controls_simple_example` | The simplified controls helper, with the state machine in `teleop_controls_simple_helper` |
| `external_inputs_example` | Feeding a session from an input source outside DeviceIO |
| `message_channel_example` | Sending and receiving on a session message channel |

## See also

- Module docs: `docs/source/references/teleop_session.rst`
- Retargeting engine: `src/core/retargeting_engine/`
