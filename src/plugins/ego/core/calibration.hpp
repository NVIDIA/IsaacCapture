// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "inc/ego_camera/ego_camera.hpp"

#include <libobsensor/ObSensor.hpp>

#include <memory>
#include <string>

namespace plugins::ego
{
bool populate_stereo_calibration_from_yaml(const std::string& yaml, core::EgoCalibrationT& calibration);
std::string raw_data(const std::shared_ptr<ob::Device>& device, OBPropertyID id);
std::string json_escape(const std::string& input);
} // namespace plugins::ego
