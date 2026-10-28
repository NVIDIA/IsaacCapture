/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_CAPTURE_DOMAIN_PLAN_H
#define ARGUS_SENDER_CAPTURE_DOMAIN_PLAN_H

#include "Config.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace argus_sender
{

struct CaptureEndpointDescriptor
{
    size_t streamIndex = 0;
    size_t cameraIndex = 0;
    StreamType streamType = STREAM_TYPE_MONO;
    uint32_t modeIndex = 0;
    uint32_t sourceWidth = 0;
    uint32_t sourceHeight = 0;
    uint64_t framePeriodNs = 0;
};

struct CaptureDomainPlan
{
    std::vector<std::vector<size_t>> domains;
    std::vector<size_t> endpointDomains;
};

bool buildCaptureDomainPlan(const std::vector<CaptureEndpointDescriptor>& endpoints,
                            CaptureDomainPlan* plan,
                            std::string* error);

} // namespace argus_sender

#endif // ARGUS_SENDER_CAPTURE_DOMAIN_PLAN_H
