<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Hand oracle

Synthetic hand recordings with known defects, each paired in `fixtures_index.json` with
the verdict and failing check the checker must produce.

```bash
../checker/setup_env.sh    # once; pins flatc and emits the bindings
./generate.sh              # writes fixtures/, verifies them, fails if the index moved
```

| File | What |
|---|---|
| `hand_model.py` | the hand: bone lengths, forward kinematics, poses, pinch targets, mirroring |
| `script.py` | the ten-step script at 60 Hz, with the options each fixture varies |
| `mcap_io.py` | writes `core.HandPoseRecord` MCAP and the label sidecar |
| `generate_fixtures.py` | the fixture list and each defect's transformation |
| `verify_fixtures.py` | reads each fixture back and confirms its defect is present |
| `fixtures_index.json` | committed; the recordings are not |

Generation is byte-deterministic, so a change to the generator shows up as a changed
index. The sets are golden (26 and 21 joints), benign variants that must pass, device
defects that must fail, and a shallow fist that must ask for a retake.
