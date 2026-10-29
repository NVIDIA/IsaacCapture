/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_STEREO_ALIGNMENT_POLICY_H
#define ARGUS_SENDER_STEREO_ALIGNMENT_POLICY_H

#include "StereoTiming.h"

#include <cstdint>

namespace argus_sender
{

const uint32_t kStereoMaxAlignmentDrops = 4;

enum StereoAlignmentAction
{
    STEREO_ALIGNMENT_KEEP,
    STEREO_ALIGNMENT_DROP_LEFT,
    STEREO_ALIGNMENT_DROP_RIGHT
};

bool stereoAlignmentDeltaNs(const FrameTiming& left, const FrameTiming& right, int64_t* deltaNs);

StereoAlignmentAction chooseStereoAlignmentAction(const FrameTiming& left,
                                                  const FrameTiming& right,
                                                  uint64_t framePeriodNs,
                                                  uint32_t dropsAlreadyApplied,
                                                  int64_t* deltaNs = NULL);

} // namespace argus_sender

#endif // ARGUS_SENDER_STEREO_ALIGNMENT_POLICY_H
