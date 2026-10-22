// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/hand_tracker_base.hpp>
#include <log_bridge/logger.hpp>
#include <mcap/tracker_channels.hpp>
#include <schema/hand_generated.h>

#include <cstdint>
#include <memory>
#include <string_view>

namespace core
{

using HandMcapViewers = McapTrackerViewers<HandPoseRecord>;

class ReplayHandTrackerImpl : public IHandTrackerImpl
{
public:
    ReplayHandTrackerImpl(std::unique_ptr<mcap::McapReader> reader,
                          std::string_view base_name,
                          const RecordedSchemas& recorded);

    ReplayHandTrackerImpl(const ReplayHandTrackerImpl&) = delete;
    ReplayHandTrackerImpl& operator=(const ReplayHandTrackerImpl&) = delete;
    ReplayHandTrackerImpl(ReplayHandTrackerImpl&&) = delete;
    ReplayHandTrackerImpl& operator=(ReplayHandTrackerImpl&&) = delete;

    void update(int64_t monotonic_time_ns) override;
    const Serialized<HandPose>& get_left_hand() const override;
    const Serialized<HandPose>& get_right_hand() const override;

private:
    Serialized<HandPose> left_tracked_;
    Serialized<HandPose> right_tracked_;
    std::unique_ptr<HandMcapViewers> mcap_viewers_;
    std::shared_ptr<spdlog::logger> logger_;
    bool warned_no_left_data_ = false;
    bool warned_no_right_data_ = false;
};

} // namespace core
