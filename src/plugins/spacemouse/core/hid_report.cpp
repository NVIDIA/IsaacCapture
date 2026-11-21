// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/spacemouse/hid_report.hpp"

#include <algorithm>

namespace plugins
{
namespace spacemouse
{

namespace
{

constexpr uint8_t kTranslationReport = 1;
constexpr uint8_t kRotationReport = 2;
constexpr uint8_t kButtonReport = 3;
constexpr std::size_t kAxesReportSize = 7; // report ID + three 16-bit axes
constexpr std::size_t kCombinedReportSize = 13; // report ID + six 16-bit axes
constexpr double kFullDeflectionCounts = 350.0;

float axis(const uint8_t* bytes)
{
    const auto raw = static_cast<int16_t>(static_cast<uint16_t>(bytes[0]) | (static_cast<uint16_t>(bytes[1]) << 8));
    return static_cast<float>(std::clamp(raw / kFullDeflectionCounts, -1.0, 1.0));
}

std::array<float, 3> axes(const uint8_t* bytes)
{
    return { axis(bytes), axis(bytes + 2), axis(bytes + 4) };
}

} // namespace

void SpaceMouseState::clear()
{
    translation.fill(0.0F);
    rotation.fill(0.0F);
    std::fill(buttons.begin(), buttons.end(), 0);
}

bool apply_report(SpaceMouseState& state, const uint8_t* report, std::size_t size, bool combined_report)
{
    if (size == 0)
        return false;

    switch (report[0])
    {
    case kTranslationReport:
        if (combined_report)
        {
            if (size < kCombinedReportSize)
                return false;
            state.translation = axes(report + 1);
            state.rotation = axes(report + 7);
            return true;
        }
        if (size < kAxesReportSize)
            return false;
        state.translation = axes(report + 1);
        return true;
    case kRotationReport:
        if (combined_report || size < kAxesReportSize)
            return false;
        state.rotation = axes(report + 1);
        return true;
    case kButtonReport:
        if (size < 2)
            return false;
        state.buttons.assign(std::min((size - 1) * 8, kMaxButtons), 0);
        for (std::size_t bit = 0; bit < state.buttons.size(); ++bit)
            state.buttons[bit] = (report[1 + bit / 8] >> (bit % 8)) & 1U;
        return true;
    default:
        return false;
    }
}

} // namespace spacemouse
} // namespace plugins
