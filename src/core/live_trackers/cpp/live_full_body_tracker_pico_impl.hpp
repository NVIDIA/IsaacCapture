// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/full_body_tracker_base.hpp>
#include <log_bridge/logger.hpp>
#include <mcap/tracker_channels.hpp>
#include <oxr_utils/oxr_funcs.hpp>
#include <oxr_utils/oxr_session_handles.hpp>
#include <oxr_utils/oxr_time.hpp>
#include <schema/full_body_generated.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace core
{

using FullBodyMcapChannels = McapTrackerChannels<FullBodyPoseRecord>;

// Live full-body impl for the "body.pico-xr" vendor: sources joints directly from
// the native PICO XR_BD_body_tracking extension.
//
// Supports limp-mode: if body tracking hardware is unavailable, the constructor
// succeeds but body_tracker_ remains XR_NULL_HANDLE and update() returns empty data.
class LiveFullBodyTrackerPicoImpl : public IFullBodyTrackerImpl
{
public:
    static std::vector<std::string> required_extensions()
    {
        return { "XR_BD_body_tracking" };
    }
    static std::unique_ptr<FullBodyMcapChannels> create_mcap_channels(mcap::McapWriter& writer,
                                                                      std::string_view base_name);

    LiveFullBodyTrackerPicoImpl(const OpenXRSessionHandles& handles, std::unique_ptr<FullBodyMcapChannels> mcap_channels);
    ~LiveFullBodyTrackerPicoImpl();

    LiveFullBodyTrackerPicoImpl(const LiveFullBodyTrackerPicoImpl&) = delete;
    LiveFullBodyTrackerPicoImpl& operator=(const LiveFullBodyTrackerPicoImpl&) = delete;
    LiveFullBodyTrackerPicoImpl(LiveFullBodyTrackerPicoImpl&&) = delete;
    LiveFullBodyTrackerPicoImpl& operator=(LiveFullBodyTrackerPicoImpl&&) = delete;

    void update(int64_t monotonic_time_ns) override;
    const Serialized<FullBodyPose>& get_body_pose() const override;

private:
    OpenXRSessionHandles handles_;
    XrTimeConverter time_converter_;
    XrSpace base_space_;
    std::shared_ptr<spdlog::logger> logger_ = isaaccapture::Logger::get("isaaccapture.core.LiveFullBodyTrackerPicoImpl");
    XrBodyTrackerBD body_tracker_;
    // The snapshot published each frame, encoded from a local in update().
    Serialized<FullBodyPose> tracked_;
    int64_t last_update_time_ = 0;

    PFN_xrCreateBodyTrackerBD pfn_create_body_tracker_;
    PFN_xrDestroyBodyTrackerBD pfn_destroy_body_tracker_;
    PFN_xrLocateBodyJointsBD pfn_locate_body_joints_;

    std::unique_ptr<FullBodyMcapChannels> mcap_channels_;
};

} // namespace core
