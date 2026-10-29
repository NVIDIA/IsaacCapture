/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_OUTPUT_GEOMETRY_H
#define ARGUS_SENDER_OUTPUT_GEOMETRY_H

#include "Config.h"

#include <cstdint>
#include <string>
#include <vector>

namespace argus_sender
{

struct ImageSize
{
    uint32_t width = 0;
    uint32_t height = 0;
};

struct NormalizedClip
{
    float left = 0.0F;
    float top = 0.0F;
    float right = 1.0F;
    float bottom = 1.0F;
};

struct ResolvedOutputGeometry
{
    // Wire/output dimensions (the size shipped in AVS1 stream descriptors).
    uint32_t width = 0;
    uint32_t height = 0;
    // Argus capture (ISP output) dimensions. Equal to width/height unless the
    // stream carries a sender-side remap, which captures at the remap's source
    // resolution and resamples down to width x height itself.
    uint32_t captureWidth = 0;
    uint32_t captureHeight = 0;
    std::vector<NormalizedClip> clips;
};

bool resolveOutputGeometry(const OutputConfig& output,
                           const std::vector<ImageSize>& sources,
                           ResolvedOutputGeometry* resolved,
                           std::string* error);

} // namespace argus_sender

#endif // ARGUS_SENDER_OUTPUT_GEOMETRY_H
