# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
import numpy as np
import rerun as rr

# Load dataset
dataset = LeRobotDataset(
    repo_id="teleop/tracking_demo", root=Path("local_datasets/teleop_tracking")
)

# Start Rerun
rr.init("teleop_tracking", spawn=True)

# Loop through frames/samples
for i in range(len(dataset)):
    rr.set_time("frame_index", sequence=dataset[i]["frame_index"].item())
    rr.set_time("timestamp", timestamp=dataset[i]["timestamp"].item())

    obs_head = dataset[i]["observation.head"]
    obs_left_hand = dataset[i]["observation.left_hand"]
    obs_right_hand = dataset[i]["observation.right_hand"]

    head_data = np.array(obs_head)
    left_hand_data = np.array(obs_left_hand)
    right_hand_data = np.array(obs_right_hand)

    rr.log("head", rr.Scalars(head_data))
    rr.log("left_hand", rr.Scalars(left_hand_data))
    rr.log("right_hand", rr.Scalars(right_hand_data))
