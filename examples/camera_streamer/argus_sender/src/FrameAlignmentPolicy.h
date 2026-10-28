/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_FRAME_ALIGNMENT_POLICY_H
#define ARGUS_SENDER_FRAME_ALIGNMENT_POLICY_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace argus_sender
{

const uint32_t kMaxFrameAlignmentDrops = 4;

struct FrameAlignmentSample
{
    uint64_t sofTscNs = 0;
    uint64_t framePeriodNs = 0;
    bool valid = false;
};

enum FrameAlignmentDecision
{
    FRAME_ALIGNMENT_UNAVAILABLE,
    FRAME_ALIGNMENT_KEEP,
    FRAME_ALIGNMENT_REACQUIRE
};

FrameAlignmentDecision chooseFrameToReacquire(const std::vector<FrameAlignmentSample>& samples,
                                              size_t* sampleIndex,
                                              uint64_t* spanNs = NULL);

// Selects the valid candidate whose SOF TSC is closest to referenceTscNs.
// A tie is resolved in favor of the newer candidate.
bool chooseClosestTscCandidate(const std::vector<uint64_t>& candidateTscNs,
                               uint64_t referenceTscNs,
                               size_t* candidateIndex,
                               int64_t* selectedDeltaNs = NULL);

// Computes the midpoint reference only when every timestamp is valid and the
// complete set fits within maxSpanNs.
bool computeSynchronizedTscReference(const std::vector<uint64_t>& timestamps,
                                     uint64_t maxSpanNs,
                                     uint64_t* referenceTscNs,
                                     uint64_t* spanNs = NULL);

} // namespace argus_sender

#endif // ARGUS_SENDER_FRAME_ALIGNMENT_POLICY_H
