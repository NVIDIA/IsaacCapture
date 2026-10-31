/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_MIXED_RATE_SCHEDULER_H
#define ARGUS_SENDER_MIXED_RATE_SCHEDULER_H

#include <cstdint>
#include <vector>

namespace argus_sender
{

class MixedRateScheduler
{
public:
    explicit MixedRateScheduler(const std::vector<uint32_t>& streamFps);

    uint32_t baseFps() const;
    std::vector<bool> next();

private:
    uint32_t m_baseFps;
    std::vector<uint32_t> m_streamFps;
    std::vector<uint64_t> m_accumulators;
};

} // namespace argus_sender

#endif // ARGUS_SENDER_MIXED_RATE_SCHEDULER_H
