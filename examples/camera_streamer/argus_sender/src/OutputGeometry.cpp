/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "OutputGeometry.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace argus_sender
{
namespace
{

bool fail(std::string* error, const std::string& message)
{
    if (error)
        *error = message;
    return false;
}

uint32_t evenRounded(double value)
{
    uint32_t rounded = static_cast<uint32_t>(std::floor(value + 0.5));
    if (rounded < 2)
        rounded = 2;
    if (rounded % 2 != 0)
        ++rounded;
    return rounded;
}

bool aspectMatches(double left, double right)
{
    const double scale = std::max(std::fabs(left), std::fabs(right));
    return scale > 0.0 && std::fabs(left - right) / scale <= 0.001;
}

NormalizedClip centerCrop(double sourceAspect, double targetAspect)
{
    NormalizedClip clip;
    if (aspectMatches(sourceAspect, targetAspect))
        return clip;
    if (targetAspect > sourceAspect)
    {
        const float retainedHeight = static_cast<float>(sourceAspect / targetAspect);
        clip.top = (1.0F - retainedHeight) * 0.5F;
        clip.bottom = 1.0F - clip.top;
    }
    else
    {
        const float retainedWidth = static_cast<float>(targetAspect / sourceAspect);
        clip.left = (1.0F - retainedWidth) * 0.5F;
        clip.right = 1.0F - clip.left;
    }
    return clip;
}

} // namespace

bool resolveOutputGeometry(const OutputConfig& output,
                           const std::vector<ImageSize>& sources,
                           ResolvedOutputGeometry* resolved,
                           std::string* error)
{
    if (!resolved)
        return fail(error, "resolved geometry output is null");
    *resolved = ResolvedOutputGeometry();
    if (sources.empty())
        return fail(error, "at least one source resolution is required");
    for (size_t i = 0; i < sources.size(); ++i)
    {
        if (sources[i].width == 0 || sources[i].height == 0)
            return fail(error, "source resolution is zero");
    }

    uint32_t width = output.width;
    uint32_t height = output.height;
    const double referenceAspect = static_cast<double>(sources[0].width) / sources[0].height;
    if (width == 0 && height == 0)
    {
        width = sources[0].width;
        height = sources[0].height;
        for (size_t i = 1; i < sources.size(); ++i)
        {
            if (sources[i].width != width || sources[i].height != height)
            {
                return fail(error, "native stereo resolutions differ; specify output width or height");
            }
        }
    }
    else if (width != 0 && height == 0)
    {
        height = evenRounded(static_cast<double>(width) / referenceAspect);
    }
    else if (width == 0 && height != 0)
    {
        width = evenRounded(static_cast<double>(height) * referenceAspect);
    }
    if (width == 0 || height == 0 || width % 2 != 0 || height % 2 != 0)
        return fail(error, "resolved NV12 dimensions must be nonzero and even");

    resolved->width = width;
    resolved->height = height;
    // With a sender-side remap the ISP delivers the remap's source frame and
    // the resample to width x height happens in the remap itself, so the
    // aspect contract applies to the CAPTURE dimensions, not the output.
    resolved->captureWidth = output.remap.enabled() ? output.remap.captureWidth : width;
    resolved->captureHeight = output.remap.enabled() ? output.remap.captureHeight : height;

    const double targetAspect = static_cast<double>(resolved->captureWidth) / resolved->captureHeight;
    // An inferred dimension preserves the reference aspect before NV12 even-pixel rounding.
    // Remap capture dimensions remain an explicit aspect contract.
    const bool inferredDimension = (output.width == 0) != (output.height == 0);
    const double strictAspect = inferredDimension && !output.remap.enabled() ? referenceAspect : targetAspect;
    resolved->clips.reserve(sources.size());
    for (size_t i = 0; i < sources.size(); ++i)
    {
        const double sourceAspect = static_cast<double>(sources[i].width) / sources[i].height;
        if (output.resizeMode == RESIZE_STRICT && !aspectMatches(sourceAspect, strictAspect))
        {
            std::ostringstream message;
            message << "strict resize rejects source " << sources[i].width << "x" << sources[i].height << " -> "
                    << resolved->captureWidth << "x" << resolved->captureHeight << " aspect-ratio change";
            return fail(error, message.str());
        }
        resolved->clips.push_back(output.resizeMode == RESIZE_CENTER_CROP ? centerCrop(sourceAspect, targetAspect) :
                                                                            NormalizedClip());
    }
    return true;
}

} // namespace argus_sender
