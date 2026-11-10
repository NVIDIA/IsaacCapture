// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <optional>

namespace video_stage
{

/// What is known about a frame that went into the encoder.
struct InFlightFrame
{
    std::uint32_t sequence = 0;
    std::uint64_t capture_tsc = 0;
    /// The application timestamp to publish with the frame, in microseconds.
    std::uint64_t stamp_us = 0;
    /// The stamp given to the encoder with the input, in microseconds: the key a returned unit is matched by.
    std::int64_t submitted_us = 0;
};

/**
 * @brief Ties each encoded unit back to the frame it came from, and says when it cannot.
 *
 * Stamp mode credits a unit to a frame only when the stamp the encoder hands back equals one submitted; older frames
 * the encoder never returned are skipped and counted, and a unit matching nothing is reported as not identified (the
 * oldest in-flight entry is dropped so the queue stays bounded). Fifo mode assumes one unit per input in order and
 * cannot be verified; it is an explicit choice for approximate timing. Worker thread only.
 */
class FrameCorrelator
{
public:
    enum class Mode
    {
        Stamp,
        Fifo,
    };

    struct Result
    {
        InFlightFrame frame;
        bool identified = true;
        /// Inputs older than the match that the encoder did not return a unit for.
        std::uint64_t skipped = 0;
    };

    explicit FrameCorrelator(Mode mode, std::size_t max_in_flight = 12) : m_mode(mode), m_max(max_in_flight)
    {
    }

    void submitted(const InFlightFrame& frame)
    {
        m_in_flight.push_back(frame);
        // The encoder holds a bounded number of frames, so a longer queue is stale metadata.
        while (m_in_flight.size() > m_max)
            m_in_flight.pop_front();
    }

    /// The frame a unit stamped `unit_stamp_us` came from; nullopt when nothing is in flight.
    std::optional<Result> returned(std::int64_t unit_stamp_us)
    {
        if (m_in_flight.empty())
            return std::nullopt;
        Result result;
        if (m_mode == Mode::Stamp)
        {
            const auto match = std::find_if(m_in_flight.begin(), m_in_flight.end(),
                                            [&](const InFlightFrame& f) { return f.submitted_us == unit_stamp_us; });
            if (match != m_in_flight.end())
            {
                result.skipped = static_cast<std::uint64_t>(std::distance(m_in_flight.begin(), match));
                result.frame = *match;
                m_in_flight.erase(m_in_flight.begin(), std::next(match));
                return result;
            }
            result.identified = false;
        }
        result.frame = m_in_flight.front();
        m_in_flight.pop_front();
        return result;
    }

    bool empty() const
    {
        return m_in_flight.empty();
    }
    std::size_t in_flight() const
    {
        return m_in_flight.size();
    }

private:
    Mode m_mode;
    std::size_t m_max;
    std::deque<InFlightFrame> m_in_flight;
};

} // namespace video_stage
