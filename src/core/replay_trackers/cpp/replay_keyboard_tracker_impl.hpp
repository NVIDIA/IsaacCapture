// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/keyboard_tracker_base.hpp>
#include <log_bridge/logger.hpp>
#include <mcap/tracker_channels.hpp>
#include <schema/keyboard_generated.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace core
{

class KeyboardInputState;

using KeyboardMcapViewers = McapTrackerViewers<KeyboardOutputRecord>;

// Replays recorded keyboard state. Live provider input is drained and discarded so a surface
// left attached during replay cannot inject keys; that is reported once.
class ReplayKeyboardTrackerImpl : public IKeyboardTrackerImpl
{
public:
    ReplayKeyboardTrackerImpl(std::unique_ptr<mcap::McapReader> reader,
                              std::string_view base_name,
                              const RecordedSchemas& recorded,
                              std::shared_ptr<KeyboardInputState> state);

    ReplayKeyboardTrackerImpl(const ReplayKeyboardTrackerImpl&) = delete;
    ReplayKeyboardTrackerImpl& operator=(const ReplayKeyboardTrackerImpl&) = delete;
    ReplayKeyboardTrackerImpl(ReplayKeyboardTrackerImpl&&) = delete;
    ReplayKeyboardTrackerImpl& operator=(ReplayKeyboardTrackerImpl&&) = delete;

    void update(int64_t monotonic_time_ns) override;
    const Serialized<KeyboardOutput>& get_data() const override;

private:
    Serialized<KeyboardOutput> tracked_;
    std::unique_ptr<KeyboardMcapViewers> mcap_viewers_;
    std::shared_ptr<KeyboardInputState> state_;
    std::string base_name_;
    std::shared_ptr<spdlog::logger> logger_;
    bool warned_no_data_ = false;
    bool warned_live_input_ = false;
};

} // namespace core
