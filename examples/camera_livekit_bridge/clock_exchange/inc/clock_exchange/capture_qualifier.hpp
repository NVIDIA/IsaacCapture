// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <clock_exchange/clock_estimator.hpp>

#include <cstdint>
#include <optional>

namespace camera_bridge
{

enum class CaptureVerdict
{
    /// The capture time is mapped onto the host's monotonic clock with a stated uncertainty inside the budget.
    Qualified,
    /// The sender marked the frame's timing record not valid, or sent none.
    SenderInvalid,
    /// No current mapping of the sender clock (none yet, or the replies stopped).
    NoMapping,
    /// A new mapping epoch has too few exchanges behind it (after a session restart or a clock step).
    Warmup,
    /// Exchange uncertainty plus the sender's stated sensor uncertainty is over the budget.
    OverBudget,
    /// The mapped capture time is after the frame was ready, or implausibly long before it.
    ImplausibleAge,
};

const char* to_string(CaptureVerdict verdict);

struct CaptureInput
{
    bool sender_valid = false;
    /// The sensor capture instant on the sender's monotonic clock, as the sender transported it.
    int64_t capture_sm_ns = 0;
    /// The sender's stated bound on its sensor-to-sender-clock mapping.
    int64_t upstream_uncertainty_ns = 0;
    /// The host's CLOCK_MONOTONIC when the frame was ready to encode.
    int64_t ready_monotonic_ns = 0;
};

struct CaptureResult
{
    CaptureVerdict verdict = CaptureVerdict::SenderInvalid;
    /// The capture instant on the host's monotonic clock; meaningful only when qualified.
    int64_t capture_bridge_ns = 0;
    /// Exchange uncertainty plus the sender's upstream uncertainty.
    int64_t combined_uncertainty_ns = 0;
    uint64_t epoch = 0;
};

/**
 * @brief Decides, per frame, whether a capture time may be claimed, and converts it to the host's monotonic clock.
 *
 * It never substitutes another time for a capture time it cannot vouch for: the answer is a verdict, and only a
 * qualified one carries a time. The checks are validity, a current mapping with enough exchanges behind it,
 * upstream plus mapping uncertainty inside the budget, and a physically possible capture-to-ready age. A learned
 * age level is not a check: it cannot tell a clock step from a stall. A step of the sender clock is seen at the
 * next exchange (a discontinuity starts a new epoch and warm-up); frames in between are exposed for at most one
 * exchange interval. Not thread-safe.
 */
class CaptureQualifier
{
public:
    CaptureQualifier(int64_t budget_ns = 5'000'000, std::size_t warmup_exchanges = 3, int64_t max_age_ns = 1'000'000'000)
        : m_budget_ns(budget_ns), m_warmup(warmup_exchanges), m_max_age_ns(max_age_ns)
    {
    }

    CaptureResult qualify(const CaptureInput& input,
                          const std::optional<ClockMapping>& mapping,
                          std::size_t accepted_in_epoch)
    {
        CaptureResult result;
        if (!input.sender_valid)
        {
            result.verdict = CaptureVerdict::SenderInvalid;
            return result;
        }
        if (!mapping)
        {
            result.verdict = CaptureVerdict::NoMapping;
            return result;
        }
        result.epoch = mapping->epoch;
        result.combined_uncertainty_ns = mapping->uncertainty_ns + input.upstream_uncertainty_ns;
        if (accepted_in_epoch < m_warmup)
        {
            result.verdict = CaptureVerdict::Warmup;
            return result;
        }
        if (result.combined_uncertainty_ns > m_budget_ns)
        {
            result.verdict = CaptureVerdict::OverBudget;
            return result;
        }
        const int64_t capture = mapping->to_local_ns(input.capture_sm_ns);
        const int64_t age = input.ready_monotonic_ns - capture;
        if (age < 0 || age > m_max_age_ns)
        {
            result.verdict = CaptureVerdict::ImplausibleAge;
            return result;
        }
        result.verdict = CaptureVerdict::Qualified;
        result.capture_bridge_ns = capture;
        return result;
    }

private:
    int64_t m_budget_ns;
    std::size_t m_warmup;
    int64_t m_max_age_ns;
};

inline const char* to_string(CaptureVerdict verdict)
{
    switch (verdict)
    {
    case CaptureVerdict::Qualified:
        return "qualified";
    case CaptureVerdict::SenderInvalid:
        return "sender_invalid";
    case CaptureVerdict::NoMapping:
        return "no_mapping";
    case CaptureVerdict::Warmup:
        return "warmup";
    case CaptureVerdict::OverBudget:
        return "over_budget";
    case CaptureVerdict::ImplausibleAge:
        return "implausible_age";
    }
    return "unknown";
}

} // namespace camera_bridge
