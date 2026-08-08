// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/controller_tracker_base.hpp>
#include <schema/controller_generated.h>

#include <string>

namespace core
{

// Tracks both left and right controllers via XR_NVX1_action_context.
// Each instance creates its own action context, so multiple ControllerTracker
// instances can coexist on the same XrSession.
class ControllerTracker : public ITracker
{
public:
    std::string_view get_name() const override
    {
        return TRACKER_NAME;
    }

    // Query methods:
    // - the handle is empty when the controller is inactive.
    // - when it is non-empty, nested fields in ControllerSnapshot are safe to read.
    const Serialized<ControllerSnapshot>& get_left_controller(const ITrackerSession& session) const;
    const Serialized<ControllerSnapshot>& get_right_controller(const ITrackerSession& session) const;

    /// Drive the left/right controller's haptic actuator for one frame.
    ///
    /// Per-side methods mirror :meth:`get_left_controller` /
    /// :meth:`get_right_controller`, matching how DeviceIO names bilateral
    /// hardware (the side is chosen by which method you call, not a parameter).
    ///
    /// Bridges Isaac Teleop's haptic feedback flow (see
    /// isaaccapture.haptic_devices.controller.ControllerHapticDevice) to the
    /// underlying runtime impl. Vendor neutral at this layer; concrete impls
    /// (e.g. the live OpenXR impl) translate to runtime-specific calls.
    ///
    /// `amplitude` is in [0, 1]; `amplitude == 0` requests an explicit stop
    /// instead of a zero-amplitude pulse. `frequency_hz == 0` selects the
    /// runtime's default frequency; `duration_s == 0` selects the runtime's
    /// shortest supported pulse.
    void apply_left_haptic_feedback(const ITrackerSession& session,
                                    float amplitude,
                                    float frequency_hz,
                                    float duration_s) const;
    void apply_right_haptic_feedback(const ITrackerSession& session,
                                     float amplitude,
                                     float frequency_hz,
                                     float duration_s) const;

    /// The interaction profile bound for the controllers; see
    /// :cpp:func:`IControllerTrackerImpl::get_interaction_profile`.
    std::string get_interaction_profile(const ITrackerSession& session) const;

private:
    static constexpr const char* TRACKER_NAME = "ControllerTracker";
};

} // namespace core
