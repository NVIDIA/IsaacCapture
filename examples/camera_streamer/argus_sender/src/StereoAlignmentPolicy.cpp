/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "StereoAlignmentPolicy.h"

namespace argus_sender
{
namespace
{

uint64_t absoluteDeltaNs(int64_t deltaNs)
{
    if (deltaNs >= 0)
        return static_cast<uint64_t>(deltaNs);
    return static_cast<uint64_t>(-(deltaNs + 1)) + 1ULL;
}

} // namespace

bool stereoAlignmentDeltaNs(const FrameTiming& left, const FrameTiming& right, int64_t* deltaNs)
{
    if (!deltaNs)
        return false;
    if (left.hasSensorSofTimestampTsc && right.hasSensorSofTimestampTsc)
    {
        *deltaNs = signedDeltaNs(right.sensorSofTimestampTscNs, left.sensorSofTimestampTscNs);
        return true;
    }
    if (!left.hasSensorTimestamp || !right.hasSensorTimestamp)
        return false;
    *deltaNs = signedDeltaNs(right.sensorTimestampNs, left.sensorTimestampNs);
    return true;
}

StereoAlignmentAction chooseStereoAlignmentAction(const FrameTiming& left,
                                                  const FrameTiming& right,
                                                  uint64_t framePeriodNs,
                                                  uint32_t dropsAlreadyApplied,
                                                  int64_t* deltaNs)
{
    int64_t measuredDeltaNs = 0;
    if (!stereoAlignmentDeltaNs(left, right, &measuredDeltaNs))
        return STEREO_ALIGNMENT_KEEP;
    if (deltaNs)
        *deltaNs = measuredDeltaNs;
    if (framePeriodNs == 0 || dropsAlreadyApplied >= kStereoMaxAlignmentDrops ||
        absoluteDeltaNs(measuredDeltaNs) <= framePeriodNs / 2)
    {
        return STEREO_ALIGNMENT_KEEP;
    }
    return measuredDeltaNs > 0 ? STEREO_ALIGNMENT_DROP_LEFT : STEREO_ALIGNMENT_DROP_RIGHT;
}

} // namespace argus_sender
