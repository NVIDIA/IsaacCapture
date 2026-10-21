// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tracker.hpp"

#include <schema/serialized.hpp>

#include <array>
#include <optional>

namespace core
{

struct HandPose;

// Abstract base interface for hand tracker implementations.
class IHandTrackerImpl : public ITrackerImpl
{
public:
    virtual const Serialized<HandPose>& get_left_hand() const = 0;
    virtual const Serialized<HandPose>& get_right_hand() const = 0;
    // A null payload can be a recorded tracking loss, not channel exhaustion.
    // Unknown for live trackers and before the first replay update.
    virtual std::optional<std::array<bool, 2>> get_replay_record_availability() const
    {
        return std::nullopt;
    }
};

} // namespace core
