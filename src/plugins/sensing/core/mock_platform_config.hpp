// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Mock backend only. Answers SiplCamera::query from the vendor platform JSON
// itself, so the pipeline-index renumbering and link-mask behaviour can be
// exercised with no SIPL library.

#pragma once

#include "sipl_camera.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace plugins
{
namespace sensing
{

/**
 * @brief Resolve a platform config to its sensors, as INvSIPLCameraQuery would.
 *
 * Module `m` of the named config is enabled when its transport's mask has a
 * non-zero hex digit at the module's link index, i.e.
 * `(masks[transportIndex] >> (4 * linkIndex)) & 0xF`. An empty mask list keeps
 * every module. `SensorInfo::id` is the position among the enabled sensors, not
 * the JSON sensorInfo.id.
 *
 * SENSING_MOCK_SCALE_DIV=N divides the reported width and height (kept even),
 * which keeps tests fast without touching the vendor JSON.
 *
 * @throws std::runtime_error when the file or the named config is missing.
 */
std::vector<SensorInfo> mock_query_platform_config(const std::string& platform_config_json,
                                                   const std::string& platform_config_name,
                                                   const std::vector<uint32_t>& link_masks);

} // namespace sensing
} // namespace plugins
