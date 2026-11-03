# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# Named poses in the same frame.
std_msgs/Header header
string[] name
geometry_msgs/Pose[] pose
bool[] is_valid
