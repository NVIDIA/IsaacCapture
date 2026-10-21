// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "xdev_hand_tracker_selection.hpp"

#include <algorithm>
#include <cctype>
#include <string>

namespace core::detail
{

namespace
{

std::string ascii_lower(std::string_view value)
{
    std::string result(value);
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return result;
}

bool contains(const std::string& value, std::string_view needle)
{
    return value.find(needle) != std::string::npos;
}

} // namespace

XDevHandTrackerSelection classify_xdev_hand_tracker(std::string_view name, std::string_view serial)
{
    const std::string lower_name = ascii_lower(name);
    const std::string lower_serial = ascii_lower(serial);

    const bool is_display = contains(lower_name, "displaydevice") || contains(lower_name, "display device") ||
                            contains(lower_serial, "displaydevice") || contains(lower_serial, "display device") ||
                            contains(lower_name, "head device") || contains(lower_serial, "head device");
    const bool is_explicit_hand_tracker = contains(lower_name, "hand tracker") ||
                                          contains(lower_serial, "hand tracker") || contains(lower_serial, "push-l") ||
                                          contains(lower_serial, "push-r");

    XDevHandTrackerPriority priority = XDevHandTrackerPriority::unclassified_fallback;
    if (is_explicit_hand_tracker)
    {
        priority = XDevHandTrackerPriority::preferred;
    }
    else if (is_display)
    {
        priority = XDevHandTrackerPriority::display_fallback;
    }

    const bool left_hint =
        contains(lower_name, "left") || contains(lower_serial, "push-l") || contains(lower_serial, "head device (0)");
    const bool right_hint =
        contains(lower_name, "right") || contains(lower_serial, "push-r") || contains(lower_serial, "head device (1)");

    // An explicit side identifies the only meaningful hand for this XDev. Generic hand/display
    // devices are still tried for both hands.
    const bool has_single_side_hint = left_hint != right_hint;
    return { priority, !has_single_side_hint || left_hint, !has_single_side_hint || right_hint };
}

} // namespace core::detail
