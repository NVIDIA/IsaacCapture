// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/spacemouse_tracker_base.hpp>
#include <log_bridge/logger.hpp>
#include <mcap/tracker_channels.hpp>
#include <schema/spacemouse_generated.h>

#include <array>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace core
{

using SpaceMouseMcapChannels = McapTrackerChannels<SpaceMouseOutputRecord>;

// In-process SpaceMouse: reads the 3Dconnexion HID device once per frame. Unlike the
// OpenXR-backed impls it uses no extensions and never touches the session handles.
class LiveSpaceMouseTrackerImpl : public ISpaceMouseTrackerImpl
{
public:
    static std::vector<std::string> required_extensions()
    {
        return {};
    }
    // Minimum time between attempts to (re)open a missing device [ns].
    static constexpr int64_t REOPEN_INTERVAL_NS = 1'000'000'000;

    static std::unique_ptr<SpaceMouseMcapChannels> create_mcap_channels(mcap::McapWriter& writer,
                                                                        std::string_view base_name);

    //! @param device_path HID device to read; empty discovers one (see discover_spacemouse_device()).
    //! @param combined_report Report layout of an explicit @p device_path.
    LiveSpaceMouseTrackerImpl(std::string device_path,
                              bool combined_report,
                              std::unique_ptr<SpaceMouseMcapChannels> mcap_channels);
    ~LiveSpaceMouseTrackerImpl() override;

    LiveSpaceMouseTrackerImpl(const LiveSpaceMouseTrackerImpl&) = delete;
    LiveSpaceMouseTrackerImpl& operator=(const LiveSpaceMouseTrackerImpl&) = delete;
    LiveSpaceMouseTrackerImpl(LiveSpaceMouseTrackerImpl&&) = delete;
    LiveSpaceMouseTrackerImpl& operator=(LiveSpaceMouseTrackerImpl&&) = delete;

    void update(int64_t monotonic_time_ns) override;
    const Serialized<SpaceMouseOutput>& get_data() const override;

private:
    bool open_device();
    void close_device();
    // Drains every pending HID report; returns false when the device went away.
    bool read_reports();

    std::string configured_path_;
    bool configured_combined_report_;
    std::string open_path_;
    bool combined_report_ = false;
    int device_fd_ = -1;
    int64_t next_open_attempt_ns_ = 0;
    bool warned_missing_ = false;
    std::array<float, 3> translation_{};
    std::array<float, 3> rotation_{};
    std::set<uint16_t> pressed_buttons_;
    Serialized<SpaceMouseOutput> tracked_;
    std::unique_ptr<SpaceMouseMcapChannels> mcap_channels_;
    std::shared_ptr<spdlog::logger> logger_ = isaaccapture::Logger::get("isaaccapture.core.LiveSpaceMouseTrackerImpl");
};

} // namespace core
