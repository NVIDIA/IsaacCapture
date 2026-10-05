# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
import glob
import os

# List folders under local_datasets matching "teleop_tracking_20251218_115049"
pattern = "local_datasets/teleop_tracking_*"
candidates = [p for p in glob.glob(pattern) if os.path.isdir(p)]
if not candidates:
    raise RuntimeError(f"No matching folders found for pattern: {pattern}")
# Sort by folder name (lexicographical, which works with timestamp naming)
candidates.sort()
latest_folder = candidates[-1]
print(f"Playing back from folder: {latest_folder}")

# Load dataset
dataset = LeRobotDataset(repo_id="teleop/tracking_demo", root=Path(latest_folder))


print("\n=== Timestamp Analysis ===")

# Get timestamps for first episode
episode_idx = 0
episode_info = dataset.meta.episodes[episode_idx]
start_idx = episode_info["dataset_from_index"]
end_idx = episode_info["dataset_to_index"]

timestamps = []
frame_indices = []
for idx in range(start_idx, end_idx):
    sample = dataset[idx]
    timestamps.append(sample["timestamp"].item())
    frame_indices.append(sample["frame_index"].item())

print(f"Episode {episode_idx} timestamps (all frames):")
for frame_idx, ts in zip(frame_indices, timestamps):
    if frame_idx % 60 == 0:
        print(f"  Frame {frame_idx:4d}: {ts:.4f}s")

# Calculate FPS from timestamps
fps = dataset.meta.info["fps"]
print(f"\nDataset FPS: {fps}")
if len(timestamps) > 1:
    avg_dt = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
    measured_fps = 1.0 / avg_dt
    print(f"Measured FPS from timestamps: {measured_fps:.2f}")
