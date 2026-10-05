<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LeRobot example datasets

Minimal examples for recording, visualizing and analyzing
[LeRobot](https://github.com/huggingface/lerobot) datasets from Isaac Teleop.

```bash
uv pip install -e ./examples/lerobot          # add [viz] for the rerun viewer
python -m isaaccapture_examples.lerobot.record
```

| Module | What it does |
| --- | --- |
| `record` | Records a LeRobot-format dataset from live human data — head and hand positions only, for demonstration |
| `visualize` | Plots a dataset with [rerun](https://rerun.io) (needs the `viz` extra) |
| `analyze` | Parses and summarizes a recorded dataset |

Datasets are written to `./local_datasets/` relative to where you run the
command, and the other two modules read from the same place. `record` always
creates a new dataset, so remove an old one before re-running:

```bash
rm -rf local_datasets
```

> The SO-101 teleoperation example referenced from the Isaac Teleop docs lives
> in the [LeRobot repository](https://github.com/huggingface/lerobot) under
> `examples/isaac_teleop_to_so101/`, not here.
