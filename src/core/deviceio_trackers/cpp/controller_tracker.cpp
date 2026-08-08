// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/deviceio_trackers/controller_tracker.hpp"

namespace core
{

// ============================================================================
// ControllerTracker Public Interface
// ============================================================================

const Serialized<ControllerSnapshot>& ControllerTracker::get_left_controller(const ITrackerSession& session) const
{
    return static_cast<const IControllerTrackerImpl&>(session.get_tracker_impl(*this)).get_left_controller();
}

const Serialized<ControllerSnapshot>& ControllerTracker::get_right_controller(const ITrackerSession& session) const
{
    return static_cast<const IControllerTrackerImpl&>(session.get_tracker_impl(*this)).get_right_controller();
}

void ControllerTracker::apply_left_haptic_feedback(const ITrackerSession& session,
                                                   float amplitude,
                                                   float frequency_hz,
                                                   float duration_s) const
{
    static_cast<const IControllerTrackerImpl&>(session.get_tracker_impl(*this))
        .apply_left_haptic_feedback(amplitude, frequency_hz, duration_s);
}

void ControllerTracker::apply_right_haptic_feedback(const ITrackerSession& session,
                                                    float amplitude,
                                                    float frequency_hz,
                                                    float duration_s) const
{
    static_cast<const IControllerTrackerImpl&>(session.get_tracker_impl(*this))
        .apply_right_haptic_feedback(amplitude, frequency_hz, duration_s);
}

std::string ControllerTracker::get_interaction_profile(const ITrackerSession& session) const
{
    return static_cast<const IControllerTrackerImpl&>(session.get_tracker_impl(*this)).get_interaction_profile();
}

} // namespace core
