// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/hand_tracker_base.hpp>
#include <schema/hand_generated.h>

namespace core
{

// Tracks both left and right hands via XR_EXT_hand_tracking.
class HandTracker : public ITracker
{
public:
    std::string_view get_name() const override
    {
        return TRACKER_NAME;
    }

    // Query methods:
    // - the handle is empty when the hand is inactive.
    // - when it is non-empty, nested fields in HandPose are safe to read.
    const Serialized<HandPose>& get_left_hand(const ITrackerSession& session) const;
    const Serialized<HandPose>& get_right_hand(const ITrackerSession& session) const;

private:
    static constexpr const char* TRACKER_NAME = "HandTracker";
};

} // namespace core
