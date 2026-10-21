// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

namespace core::detail
{

enum class XDevHandTrackerPriority
{
    preferred,
    display_fallback,
    unclassified_fallback,
};

struct XDevHandTrackerSelection
{
    XDevHandTrackerPriority priority;
    bool left;
    bool right;
};

XDevHandTrackerSelection classify_xdev_hand_tracker(std::string_view name, std::string_view serial);

} // namespace core::detail
