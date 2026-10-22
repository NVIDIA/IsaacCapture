// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/hand_tracker_base.hpp>
#include <log_bridge/logger.hpp>
#include <mcap/tracker_channels.hpp>
#include <oxr_utils/oxr_funcs.hpp>
#include <oxr_utils/oxr_session_handles.hpp>
#include <oxr_utils/oxr_time.hpp>
#include <schema/hand_generated.h>

#include <XR_MNDX_xdev_space.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core
{

using HandMcapChannels = McapTrackerChannels<HandPoseRecord>;

class LiveHandTrackerImpl : public IHandTrackerImpl
{
public:
    static std::vector<std::string> required_extensions();
    static std::unique_ptr<HandMcapChannels> create_mcap_channels(mcap::McapWriter& writer, std::string_view base_name);

    LiveHandTrackerImpl(const OpenXRSessionHandles& handles, std::unique_ptr<HandMcapChannels> mcap_channels);
    ~LiveHandTrackerImpl();

    LiveHandTrackerImpl(const LiveHandTrackerImpl&) = delete;
    LiveHandTrackerImpl& operator=(const LiveHandTrackerImpl&) = delete;
    LiveHandTrackerImpl(LiveHandTrackerImpl&&) = delete;
    LiveHandTrackerImpl& operator=(LiveHandTrackerImpl&&) = delete;

    void update(int64_t monotonic_time_ns) override;
    const Serialized<HandPose>& get_left_hand() const override;
    const Serialized<HandPose>& get_right_hand() const override;

private:
    void initialize_xdev_hand_trackers(const OpenXRSessionHandles& handles);
    bool try_create_xdev_hand_tracker(XrSession session,
                                      XrXDevIdMNDX xdev_id,
                                      XrHandEXT hand,
                                      std::vector<XrHandTrackerEXT>& trackers);
    bool try_create_default_hand_tracker(XrSession session, XrHandEXT hand, std::vector<XrHandTrackerEXT>& trackers);
    void destroy_hand_trackers(std::vector<XrHandTrackerEXT>& trackers);
    void destroy_xdev_list();
    void update_hand(const std::vector<XrHandTrackerEXT>& trackers, XrTime time, std::optional<HandPoseT>& tracked);
    bool try_update_hand(XrHandTrackerEXT tracker, XrTime time, std::optional<HandPoseT>& tracked);

    XrTimeConverter time_converter_;
    XrSpace base_space_;
    std::shared_ptr<spdlog::logger> logger_ = isaaccapture::Logger::get("isaaccapture.core.LiveHandTrackerImpl");

    std::vector<XrHandTrackerEXT> left_hand_trackers_;
    std::vector<XrHandTrackerEXT> right_hand_trackers_;
    XrXDevListMNDX xdev_list_;

    // The snapshots published each frame, encoded from locals in update().
    Serialized<HandPose> left_tracked_;
    Serialized<HandPose> right_tracked_;
    int64_t last_update_time_ = 0;

    PFN_xrCreateHandTrackerEXT pfn_create_hand_tracker_;
    PFN_xrDestroyHandTrackerEXT pfn_destroy_hand_tracker_;
    PFN_xrLocateHandJointsEXT pfn_locate_hand_joints_;
    PFN_xrCreateXDevListMNDX pfn_create_xdev_list_;
    PFN_xrDestroyXDevListMNDX pfn_destroy_xdev_list_;
    PFN_xrEnumerateXDevsMNDX pfn_enumerate_xdevs_;
    PFN_xrGetXDevPropertiesMNDX pfn_get_xdev_properties_;

    std::unique_ptr<HandMcapChannels> mcap_channels_;
};

} // namespace core
