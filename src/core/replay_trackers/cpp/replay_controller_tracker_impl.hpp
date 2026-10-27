// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/controller_tracker_base.hpp>
#include <log_bridge/logger.hpp>
#include <mcap/tracker_channels.hpp>
#include <schema/controller_generated.h>

#include <cstdint>
#include <memory>
#include <string_view>

namespace core
{

using ControllerMcapViewers = McapTrackerViewers<ControllerSnapshotRecord>;

class ReplayControllerTrackerImpl : public IControllerTrackerImpl
{
public:
    ReplayControllerTrackerImpl(TrackerSampleSource source, std::string_view base_name);

    ReplayControllerTrackerImpl(const ReplayControllerTrackerImpl&) = delete;
    ReplayControllerTrackerImpl& operator=(const ReplayControllerTrackerImpl&) = delete;
    ReplayControllerTrackerImpl(ReplayControllerTrackerImpl&&) = delete;
    ReplayControllerTrackerImpl& operator=(ReplayControllerTrackerImpl&&) = delete;

    void update(int64_t monotonic_time_ns) override;
    const Serialized<ControllerSnapshot>& get_left_controller() const override;
    const Serialized<ControllerSnapshot>& get_right_controller() const override;
    // Replay sessions do not drive hardware — haptic feedback is a no-op here.
    void apply_left_haptic_feedback(float /*amplitude*/, float /*frequency_hz*/, float /*duration_s*/) const override
    {
    }
    void apply_right_haptic_feedback(float /*amplitude*/, float /*frequency_hz*/, float /*duration_s*/) const override
    {
    }

private:
    Serialized<ControllerSnapshot> left_tracked_;
    Serialized<ControllerSnapshot> right_tracked_;
    std::unique_ptr<ControllerMcapViewers> mcap_viewers_;
    std::shared_ptr<spdlog::logger> logger_;
    bool warned_no_left_data_ = false;
    bool warned_no_right_data_ = false;
};

} // namespace core
