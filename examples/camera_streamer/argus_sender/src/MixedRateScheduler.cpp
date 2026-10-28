/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "MixedRateScheduler.h"

#include <algorithm>

namespace argus_sender
{

MixedRateScheduler::MixedRateScheduler(const std::vector<uint32_t>& streamFps) : m_baseFps(0), m_streamFps(streamFps)
{
    for (size_t i = 0; i < streamFps.size(); ++i)
        m_baseFps = std::max(m_baseFps, streamFps[i]);
    m_accumulators.resize(streamFps.size());
    for (size_t i = 0; i < streamFps.size(); ++i)
    {
        m_accumulators[i] = m_baseFps >= streamFps[i] ? m_baseFps - streamFps[i] : 0;
    }
}

uint32_t MixedRateScheduler::baseFps() const
{
    return m_baseFps;
}

std::vector<bool> MixedRateScheduler::next()
{
    std::vector<bool> due(m_streamFps.size(), false);
    if (m_baseFps == 0)
        return due;
    for (size_t i = 0; i < m_streamFps.size(); ++i)
    {
        m_accumulators[i] += m_streamFps[i];
        if (m_accumulators[i] >= m_baseFps)
        {
            m_accumulators[i] -= m_baseFps;
            due[i] = true;
        }
    }
    return due;
}

} // namespace argus_sender
