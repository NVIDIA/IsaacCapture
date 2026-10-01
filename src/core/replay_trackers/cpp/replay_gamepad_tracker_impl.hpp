// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/gamepad_tracker_base.hpp>
#include <log_bridge/logger.hpp>
#include <mcap/tracker_channels.hpp>
#include <schema/gamepad_generated.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace core
{

using GamepadMcapViewers = McapTrackerViewers<GamepadOutputRecord>;

// Replays recorded gamepad state; the physical device is never opened.
class ReplayGamepadTrackerImpl : public IGamepadTrackerImpl
{
public:
    ReplayGamepadTrackerImpl(std::unique_ptr<mcap::McapReader> reader,
                             std::string_view base_name,
                             const RecordedSchemas& recorded);

    ReplayGamepadTrackerImpl(const ReplayGamepadTrackerImpl&) = delete;
    ReplayGamepadTrackerImpl& operator=(const ReplayGamepadTrackerImpl&) = delete;
    ReplayGamepadTrackerImpl(ReplayGamepadTrackerImpl&&) = delete;
    ReplayGamepadTrackerImpl& operator=(ReplayGamepadTrackerImpl&&) = delete;

    void update(int64_t monotonic_time_ns) override;
    const Serialized<GamepadOutput>& get_data() const override;

private:
    Serialized<GamepadOutput> tracked_;
    std::unique_ptr<GamepadMcapViewers> mcap_viewers_;
    std::string base_name_;
    std::shared_ptr<spdlog::logger> logger_;
    bool warned_no_data_ = false;
};

} // namespace core
