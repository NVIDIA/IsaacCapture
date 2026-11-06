/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_CAPTURE_PRIORITY_POLICY_H
#define ARGUS_SENDER_CAPTURE_PRIORITY_POLICY_H

#include "Config.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace argus_sender
{

size_t selectPriorityStream(const AppConfig& config, const std::vector<uint32_t>& streamFps);

uint64_t secondaryCaptureReleaseDelayNs(uint64_t framePeriodNs);

uint64_t prioritySyncToleranceNs(uint64_t framePeriodNs);

} // namespace argus_sender

#endif // ARGUS_SENDER_CAPTURE_PRIORITY_POLICY_H
