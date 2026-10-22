// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <schema/orbbec_ego_calibration_generated.h>

#include <string>

namespace plugins::ego
{
// Parse the EGO SDK's cam0.T_cam_imu contract, filling only missing IMU extrinsics.
// Invalid input leaves calibration unchanged; raw YAML is never rewritten.
bool populate_imu_calibration_from_yaml(const std::string& yaml, core::OrbbecEgoCalibrationT& calibration);
} // namespace plugins::ego
