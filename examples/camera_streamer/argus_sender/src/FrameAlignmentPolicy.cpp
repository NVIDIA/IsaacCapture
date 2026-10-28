/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "FrameAlignmentPolicy.h"

#include <limits>

namespace argus_sender
{

FrameAlignmentDecision chooseFrameToReacquire(const std::vector<FrameAlignmentSample>& samples,
                                              size_t* sampleIndex,
                                              uint64_t* spanNs)
{
    if (sampleIndex)
        *sampleIndex = std::numeric_limits<size_t>::max();
    if (spanNs)
        *spanNs = 0;
    if (samples.empty())
        return FRAME_ALIGNMENT_KEEP;

    uint64_t newestTscNs = 0;
    uint64_t oldestTscNs = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i < samples.size(); ++i)
    {
        if (!samples[i].valid || samples[i].sofTscNs == 0 || samples[i].framePeriodNs == 0)
        {
            return FRAME_ALIGNMENT_UNAVAILABLE;
        }
        if (samples[i].sofTscNs > newestTscNs)
            newestTscNs = samples[i].sofTscNs;
        if (samples[i].sofTscNs < oldestTscNs)
            oldestTscNs = samples[i].sofTscNs;
    }
    if (spanNs)
        *spanNs = newestTscNs - oldestTscNs;

    size_t oldestOutOfWindow = std::numeric_limits<size_t>::max();
    uint64_t largestAgeNs = 0;
    for (size_t i = 0; i < samples.size(); ++i)
    {
        const uint64_t ageNs = newestTscNs - samples[i].sofTscNs;
        if (ageNs > samples[i].framePeriodNs / 2 &&
            (oldestOutOfWindow == std::numeric_limits<size_t>::max() || ageNs > largestAgeNs))
        {
            oldestOutOfWindow = i;
            largestAgeNs = ageNs;
        }
    }
    if (oldestOutOfWindow == std::numeric_limits<size_t>::max())
        return FRAME_ALIGNMENT_KEEP;
    if (sampleIndex)
        *sampleIndex = oldestOutOfWindow;
    return FRAME_ALIGNMENT_REACQUIRE;
}

bool chooseClosestTscCandidate(const std::vector<uint64_t>& candidateTscNs,
                               uint64_t referenceTscNs,
                               size_t* candidateIndex,
                               int64_t* selectedDeltaNs)
{
    if (candidateIndex)
        *candidateIndex = std::numeric_limits<size_t>::max();
    if (selectedDeltaNs)
        *selectedDeltaNs = 0;
    if (referenceTscNs == 0)
        return false;

    size_t bestIndex = std::numeric_limits<size_t>::max();
    uint64_t bestDistanceNs = std::numeric_limits<uint64_t>::max();
    uint64_t bestTscNs = 0;
    for (size_t i = 0; i < candidateTscNs.size(); ++i)
    {
        const uint64_t candidate = candidateTscNs[i];
        if (candidate == 0)
            continue;
        const uint64_t distance = candidate >= referenceTscNs ? candidate - referenceTscNs : referenceTscNs - candidate;
        if (bestIndex == std::numeric_limits<size_t>::max() || distance < bestDistanceNs ||
            (distance == bestDistanceNs && candidate > bestTscNs))
        {
            bestIndex = i;
            bestDistanceNs = distance;
            bestTscNs = candidate;
        }
    }
    if (bestIndex == std::numeric_limits<size_t>::max())
        return false;

    if (candidateIndex)
        *candidateIndex = bestIndex;
    if (selectedDeltaNs)
    {
        const uint64_t positiveLimit = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
        if (bestTscNs >= referenceTscNs)
        {
            const uint64_t delta = bestTscNs - referenceTscNs;
            *selectedDeltaNs = delta > positiveLimit ? std::numeric_limits<int64_t>::max() : static_cast<int64_t>(delta);
        }
        else
        {
            const uint64_t delta = referenceTscNs - bestTscNs;
            *selectedDeltaNs = delta > positiveLimit ? std::numeric_limits<int64_t>::min() : -static_cast<int64_t>(delta);
        }
    }
    return true;
}

bool computeSynchronizedTscReference(const std::vector<uint64_t>& timestamps,
                                     uint64_t maxSpanNs,
                                     uint64_t* referenceTscNs,
                                     uint64_t* spanNs)
{
    if (referenceTscNs)
        *referenceTscNs = 0;
    if (spanNs)
        *spanNs = 0;
    if (!referenceTscNs || timestamps.empty())
        return false;

    uint64_t earliest = std::numeric_limits<uint64_t>::max();
    uint64_t latest = 0;
    for (size_t i = 0; i < timestamps.size(); ++i)
    {
        if (timestamps[i] == 0)
            return false;
        if (timestamps[i] < earliest)
            earliest = timestamps[i];
        if (timestamps[i] > latest)
            latest = timestamps[i];
    }
    const uint64_t valueSpanNs = latest - earliest;
    if (spanNs)
        *spanNs = valueSpanNs;
    if (valueSpanNs > maxSpanNs)
        return false;
    *referenceTscNs = earliest + valueSpanNs / 2ULL;
    return true;
}

} // namespace argus_sender
