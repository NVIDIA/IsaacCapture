<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Hand and glove acceptance

Checks that a hand-tracking integration — a glove plugin such as Manus or Wuji, or a
headset's own optical hands — delivers `core.HandPoseRecord` data a retargeter can use:
26 OpenXR joints per hand, in metres, in the OpenXR joint frame, with the fingers where
their names say.

| Directory | What it is | Needs |
|---|---|---|
| [`capture/`](capture/README.md) | records one scripted take and its label sidecar | the built isaacteleop wheel, CloudXR, a headset |
| [`checker/`](checker/README.md) | reads a take and returns `pass` / `fail` / `retake` / `insufficient_data` | `mcap`, `flatbuffers`; no isaacteleop |
| [`oracle/`](oracle/README.md) | generates synthetic recordings with known defects | the checker's venv |

`capture` imports the checker to label and audit what it wrote. The checker never imports
isaacteleop, and the oracle shares no code with the checker.

## Recording a take

```bash
acceptance/hand/checker/setup_env.sh
acceptance/hand/capture/setup_env.sh

# The headset's optical hands: no plugin.
acceptance/hand/capture/record.sh quest3 --accept-eula

# A glove. The plugin has to be built and installed into install/plugins first.
acceptance/hand/capture/record.sh manus --plugin manus_hand_plugin --accept-eula
WUJI_GLOVE_WRIST_SOURCE=hand_tracking \
  acceptance/hand/capture/record.sh wuji --plugin wuji_glove_plugin --accept-eula
```

Open the printed viser URL and press start. The script runs itself: each of the ten steps
is spoken, counted down with three short tones, opened by a beep and closed by a tick.
Nothing is pressed during the take. Then check it:

```bash
acceptance/hand/checker/.venv/bin/python -m hand_acceptance.cli TAKE-hand.mcap
```

Hand over the whole take directory; its four files repeat its name.

## Thresholds

Every tolerance is a placeholder until real glove recordings set it. Synthetic data
proves that a measurement is correct; it cannot say what a real hand looks like.
