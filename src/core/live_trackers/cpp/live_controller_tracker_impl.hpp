// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/controller_tracker_base.hpp>
#include <log_bridge/logger.hpp>
#include <mcap/tracker_channels.hpp>
#include <oxr_utils/oxr_funcs.hpp>
#include <oxr_utils/oxr_session_handles.hpp>
#include <oxr_utils/oxr_time.hpp>
#include <schema/controller_generated.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace core
{

using ControllerMcapChannels = McapTrackerChannels<ControllerSnapshotRecord>;

class LiveControllerTrackerImpl : public IControllerTrackerImpl
{
public:
    static std::vector<std::string> required_extensions()
    {
        return { XR_NVX1_ACTION_CONTEXT_EXTENSION_NAME };
    }
    static std::unique_ptr<ControllerMcapChannels> create_mcap_channels(mcap::McapWriter& writer,
                                                                        std::string_view base_name);

    LiveControllerTrackerImpl(const OpenXRSessionHandles& handles, std::unique_ptr<ControllerMcapChannels> mcap_channels);
    ~LiveControllerTrackerImpl() = default;

    LiveControllerTrackerImpl(const LiveControllerTrackerImpl&) = delete;
    LiveControllerTrackerImpl& operator=(const LiveControllerTrackerImpl&) = delete;
    LiveControllerTrackerImpl(LiveControllerTrackerImpl&&) = delete;
    LiveControllerTrackerImpl& operator=(LiveControllerTrackerImpl&&) = delete;

    void update(int64_t monotonic_time_ns) override;
    const Serialized<ControllerSnapshot>& get_left_controller() const override;
    const Serialized<ControllerSnapshot>& get_right_controller() const override;
    void apply_left_haptic_feedback(float amplitude, float frequency_hz, float duration_s) const override;
    void apply_right_haptic_feedback(float amplitude, float frequency_hz, float duration_s) const override;
    std::string get_interaction_profile() const override;

private:
    std::string query_interaction_profile() const;

public:
private:
    // Internal side selector for the shared haptic implementation. The public
    // surface stays split (apply_left/right) to match get_left/right_controller.
    enum class Side
    {
        Left,
        Right
    };
    void apply_haptic_feedback(Side side, float amplitude, float frequency_hz, float duration_s) const;

    const OpenXRCoreFunctions core_funcs_;
    XrTimeConverter time_converter_;
    std::shared_ptr<spdlog::logger> logger_ = isaaccapture::Logger::get("isaaccapture.core.LiveControllerTrackerImpl");

    XrInstance instance_;
    XrSession session_;

    // Cached on sync: querying per call would cost an IPC round trip per frame.
    std::string interaction_profile_;
    XrSpace base_space_;

    XrPath left_hand_path_;
    XrPath right_hand_path_;

    ActionContextFunctions action_ctx_funcs_;
    XrInstanceActionContextPtr instance_action_context_;
    XrSessionActionContextPtr session_action_context_;

    XrActionSetPtr action_set_;
    XrAction grip_pose_action_;
    XrAction aim_pose_action_;
    XrAction primary_click_action_;
    XrAction secondary_click_action_;
    XrAction thumbstick_action_;
    XrAction thumbstick_click_action_;
    XrAction menu_click_action_;
    XrAction squeeze_value_action_;
    XrAction trigger_value_action_;
    // Output action that drives controller rumble. One action with two
    // subaction paths (left + right) bound to /user/hand/{left,right}/output/haptic
    // matches how the input actions above subaction onto the same paths.
    XrAction haptic_action_;

    XrSpacePtr left_grip_space_;
    XrSpacePtr right_grip_space_;
    XrSpacePtr left_aim_space_;
    XrSpacePtr right_aim_space_;

    // The snapshots published each frame, encoded from locals in update().
    Serialized<ControllerSnapshot> left_tracked_;
    Serialized<ControllerSnapshot> right_tracked_;
    int64_t last_update_time_ = 0;

    // Once-per-side log gates for OpenXR haptic call failures. Indexed by
    // side: [0]=left, [1]=right. `mutable` is required because
    // `apply_haptic_feedback` is `const` (the impl object is treated as
    // immutable from the public interface; the side effect lives in the
    // runtime), but we still want to log the first failure per side.
    mutable std::array<std::atomic<bool>, 2> apply_haptic_error_logged_{ { false, false } };
    mutable std::array<std::atomic<bool>, 2> stop_haptic_error_logged_{ { false, false } };

    std::unique_ptr<ControllerMcapChannels> mcap_channels_;
};

} // namespace core
