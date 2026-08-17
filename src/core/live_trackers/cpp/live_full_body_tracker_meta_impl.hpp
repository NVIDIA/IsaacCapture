// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/full_body_tracker_base.hpp>
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

using FullBodyMcapChannels = McapTrackerChannels<FullBodyPoseRecord, FullBodyPose>;

// Live full-body impl for the "body.quest-cloudxr" vendor: sources the full 84-joint
// skeleton from XR_META_body_tracking_full_body, then reduces it to the canonical
// 24-joint BodyJoint layout (schema/fbs/full_body.fbs) before publishing.
//
// This reduction is the same joint selection CloudXR.js used to perform in the
// browser (PICO_TO_META_JOINT_MAP) before the Quest full-body skeleton was sent
// down the wire in full; it now lives here, immediately upstream of the
// orientation-convention retargeting in isaacteleop.retargeting_engine, matching
// every other vendor's contract of publishing vendor-neutral 24-joint data.
//
// Supports limp-mode: if body tracking hardware is unavailable, the constructor
// succeeds but body_tracker_ remains XR_NULL_HANDLE and update() returns empty data.
class LiveFullBodyTrackerMetaImpl : public IFullBodyTrackerImpl
{
public:
    static std::vector<std::string> required_extensions()
    {
        return { "XR_FB_body_tracking", "XR_META_body_tracking_full_body" };
    }
    static std::unique_ptr<FullBodyMcapChannels> create_mcap_channels(mcap::McapWriter& writer,
                                                                      std::string_view base_name);

    LiveFullBodyTrackerMetaImpl(const OpenXRSessionHandles& handles, std::unique_ptr<FullBodyMcapChannels> mcap_channels);
    ~LiveFullBodyTrackerMetaImpl();

    LiveFullBodyTrackerMetaImpl(const LiveFullBodyTrackerMetaImpl&) = delete;
    LiveFullBodyTrackerMetaImpl& operator=(const LiveFullBodyTrackerMetaImpl&) = delete;
    LiveFullBodyTrackerMetaImpl(LiveFullBodyTrackerMetaImpl&&) = delete;
    LiveFullBodyTrackerMetaImpl& operator=(LiveFullBodyTrackerMetaImpl&&) = delete;

    void update(int64_t monotonic_time_ns) override;
    const FullBodyPoseTrackedT& get_body_pose() const override;

private:
    XrTimeConverter time_converter_;
    XrSpace base_space_;
    XrBodyTrackerFB body_tracker_;
    FullBodyPoseTrackedT tracked_;
    int64_t last_update_time_ = 0;

    PFN_xrCreateBodyTrackerFB pfn_create_body_tracker_;
    PFN_xrDestroyBodyTrackerFB pfn_destroy_body_tracker_;
    PFN_xrLocateBodyJointsFB pfn_locate_body_joints_;

    std::unique_ptr<FullBodyMcapChannels> mcap_channels_;
};

} // namespace core
