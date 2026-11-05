// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "camera_capture.hpp"
#include "sensing_types.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace plugins
{
namespace sensing
{

/// Opens the chosen backend for the given camera ids (SIPL pipeline indices, or Argus device indices).
/// @throws std::runtime_error if the backend is not built in or cannot open the cameras.
std::unique_ptr<CameraCapture> make_capture(CameraBackend backend,
                                            const SensingConfig& config,
                                            const std::vector<uint32_t>& camera_ids);

} // namespace sensing
} // namespace plugins
