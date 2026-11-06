// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "producer_status.hpp"

#include <string_view>

namespace plugins
{
namespace sensing
{

/**
 * @brief Turn a camera-start failure into what an operator should do about it.
 *
 * The capture stack reports failures as status codes and driver log lines ("Init failed with SIPLStatus 10"). A
 * camera that is simply unpowered or unplugged produces the same failure as several others, so the message names
 * the physical checks first and keeps the original text for anyone who needs it.
 */
Fault diagnose_camera_failure(std::string_view what);

} // namespace sensing
} // namespace plugins
