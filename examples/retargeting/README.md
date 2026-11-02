<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Retargeting engine examples

Standalone demos of the retargeting engine. In a venv set up as in
[Running an example](../README.md#running-an-example), install once, then run
whichever one you want:

```bash
uv pip install -e ./examples/retargeting
python -m isaaccapture_examples.retargeting.sources_example
```

| Module | What it shows |
| --- | --- |
| `sources_example` | Wiring DeviceIO sources into a retargeting graph |
| `dual_source_teleop_example` | Two input sources driving one session |
| `example_retargeters` | `GainOffsetRetargeter`, a minimal custom retargeter |
| `multi_retargeter_tuning_demo_imgui` | Tuning several retargeters live, with an ImGui panel |
| `sharpa_hand_retargeter_demo` | Bimanual Sharpa hands; `--synthetic` needs no headset. Needs `isaaccapture[grounding]` |
| `wuji_hand_retargeter_demo` | Wuji hand, in `synthetic` / `replay` / `drive` modes. Needs `isaaccapture[wuji]` |

Docs: `docs/source/references/retargeting/`.
