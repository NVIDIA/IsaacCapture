// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/gamepad_tracker_base.hpp>
#include <log_bridge/logger.hpp>
#include <mcap/tracker_channels.hpp>
#include <schema/gamepad_generated.h>

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace core
{

using GamepadMcapChannels = McapTrackerChannels<GamepadOutputRecord>;

// In-process gamepad: reads the Linux joystick-API device once per frame. Unlike the
// OpenXR-backed impls it uses no extensions and never touches the session handles.
class LiveGamepadTrackerImpl : public IGamepadTrackerImpl
{
public:
    static std::vector<std::string> required_extensions()
    {
        return {};
    }
    // Minimum time between attempts to (re)open a missing device [ns].
    static constexpr int64_t REOPEN_INTERVAL_NS = 1'000'000'000;

    static std::unique_ptr<GamepadMcapChannels> create_mcap_channels(mcap::McapWriter& writer,
                                                                     std::string_view base_name);

    //! @param device_path Joystick device to read; empty discovers one (see discover_gamepad_device()).
    LiveGamepadTrackerImpl(std::string device_path, std::unique_ptr<GamepadMcapChannels> mcap_channels);
    ~LiveGamepadTrackerImpl() override;

    LiveGamepadTrackerImpl(const LiveGamepadTrackerImpl&) = delete;
    LiveGamepadTrackerImpl& operator=(const LiveGamepadTrackerImpl&) = delete;
    LiveGamepadTrackerImpl(LiveGamepadTrackerImpl&&) = delete;
    LiveGamepadTrackerImpl& operator=(LiveGamepadTrackerImpl&&) = delete;

    void update(int64_t monotonic_time_ns) override;
    const Serialized<GamepadOutput>& get_data() const override;

private:
    bool open_device();
    void close_device();
    // Drains every pending joystick event; returns false when the device went away.
    bool read_events();

    std::string configured_path_;
    std::string open_path_;
    int device_fd_ = -1;
    int64_t next_open_attempt_ns_ = 0;
    bool warned_missing_ = false;
    std::set<uint16_t> pressed_buttons_;
    std::vector<float> axes_;
    Serialized<GamepadOutput> tracked_;
    std::unique_ptr<GamepadMcapChannels> mcap_channels_;
    std::shared_ptr<spdlog::logger> logger_ = isaaccapture::Logger::get("isaaccapture.core.LiveGamepadTrackerImpl");
};

} // namespace core
