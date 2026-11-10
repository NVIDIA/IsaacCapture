// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
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
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_samples.size() < m_capacity)
            m_samples.push_back(ns);
        else
            ++m_overflow;
    }

    size_t count() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_samples.size();
    }
    size_t overflow() const
    {
        return m_overflow;
    }

    /// The q-th percentile (0..100) in nanoseconds; 0 when empty.
    int64_t percentile(double q) const
    {
        std::vector<int64_t> sorted;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            sorted = m_samples;
        }
        if (sorted.empty())
            return 0;
        std::sort(sorted.begin(), sorted.end());
        const size_t index =
            std::min(sorted.size() - 1, static_cast<size_t>(q / 100.0 * static_cast<double>(sorted.size())));
        return sorted[index];
    }

private:
    size_t m_capacity;
    size_t m_overflow = 0;
    mutable std::mutex m_mutex;
    std::vector<int64_t> m_samples;
};

struct PipelineStats
{
    // Written by the processing thread and read by the statistics thread, so the counters are atomic.
    std::atomic<uint64_t> records{ 0 };
    std::atomic<uint64_t> pictures{ 0 };
    std::atomic<uint64_t> dropped_no_surface{ 0 };
    std::atomic<uint64_t> gaps{ 0 }; ///< records whose frame number did not increase
    std::atomic<uint64_t> metadata_lost{ 0 }; ///< pictures whose metadata entry had been overwritten
    DurationStats queue_age_ns; ///< from intake to the start of processing
    DurationStats decode_ns; ///< from handing an access unit to the decoder to its picture appearing
    DurationStats decode_lag; ///< records handed to the decoder after this picture's own record, before it appeared
                              ///< (frames)
    DurationStats process_encode_ns;
    DurationStats receive_to_output_ns;
};

} // namespace video_stage
