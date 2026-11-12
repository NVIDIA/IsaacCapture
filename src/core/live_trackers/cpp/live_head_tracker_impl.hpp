// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/head_tracker_base.hpp>
#include <mcap/tracker_channels.hpp>
#include <oxr_utils/oxr_funcs.hpp>
#include <oxr_utils/oxr_session_handles.hpp>
#include <oxr_utils/oxr_time.hpp>
#include <schema/head_generated.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace core
{

using HeadMcapChannels = McapTrackerChannels<HeadPoseRecord>;

class LiveHeadTrackerImpl : public IHeadTrackerImpl
{
public:
    static std::vector<std::string> required_extensions()
    {
        return {};
    }
    static std::unique_ptr<HeadMcapChannels> create_mcap_channels(mcap::McapWriter& writer, std::string_view base_name);

    LiveHeadTrackerImpl(const OpenXRSessionHandles& handles, std::unique_ptr<HeadMcapChannels> mcap_channels);

    LiveHeadTrackerImpl(const LiveHeadTrackerImpl&) = delete;
    LiveHeadTrackerImpl& operator=(const LiveHeadTrackerImpl&) = delete;
    LiveHeadTrackerImpl(LiveHeadTrackerImpl&&) = delete;
    LiveHeadTrackerImpl& operator=(LiveHeadTrackerImpl&&) = delete;

    void update(int64_t monotonic_time_ns) override;
    const Serialized<HeadPose>& get_head() const override;

private:
    const OpenXRCoreFunctions core_funcs_;
    OpenXRSessionHandles handles_;
    XrTimeConverter time_converter_;
    XrSpace base_space_;
    XrSpacePtr view_space_;
    // The snapshot published each frame, encoded from a local in update().
    Serialized<HeadPose> tracked_;
    int64_t last_update_time_ = 0;
    std::unique_ptr<HeadMcapChannels> mcap_channels_;
};

} // namespace core
