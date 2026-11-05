// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace video_stage
{

/// Bounded per-stage duration samples, summarised as percentiles at shutdown; nothing is logged per frame.
class DurationStats
{
public:
    explicit DurationStats(size_t capacity = 1U << 16) : m_capacity(capacity)
    {
        m_samples.reserve(capacity);
    }

    void add(int64_t ns)
    {
        if (m_samples.size() < m_capacity)
            m_samples.push_back(ns);
        else
            ++m_overflow;
    }

    size_t count() const
    {
        return m_samples.size();
    }
    size_t overflow() const
    {
        return m_overflow;
    }

    /// The q-th percentile (0..100) in nanoseconds; 0 when empty.
    int64_t percentile(double q) const
    {
        if (m_samples.empty())
            return 0;
        std::vector<int64_t> sorted(m_samples);
        std::sort(sorted.begin(), sorted.end());
        const size_t index =
            std::min(sorted.size() - 1, static_cast<size_t>(q / 100.0 * static_cast<double>(sorted.size())));
        return sorted[index];
    }

private:
    size_t m_capacity;
    size_t m_overflow = 0;
    std::vector<int64_t> m_samples;
};

} // namespace video_stage
