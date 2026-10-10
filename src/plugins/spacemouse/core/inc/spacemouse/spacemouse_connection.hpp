// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hid_report.hpp"

#include <log_bridge/logger.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace plugins
{
namespace spacemouse
{

//! Why the device is or is not open.
enum class ConnectionStatus
{
    Connected, //!< The device is open and being read.
    NoDevice, //!< No SpaceMouse is connected (or the configured path does not exist).
    OpenFailed, //!< A SpaceMouse is connected but cannot be opened, usually for lack of permission.
};

struct SpaceMouseConnectionConfig
{
    //! Device node to read; unset discovers the first validated model and rediscovers after a disconnect.
    std::optional<std::filesystem::path> device_path;
    //! Report layout of an explicit @ref device_path (3Dconnexion Universal Receiver).
    bool combined_report = false;
    std::filesystem::path hidraw_class_dir = "/sys/class/hidraw";
    std::filesystem::path dev_dir = "/dev";
    //! Minimum time between attempts to open a missing device.
    int64_t retry_interval_ns = 1'000'000'000;
    //! Deflected axes are zeroed after this long with no report, such as a wireless link lost mid-motion.
    //! A deflected SpaceMouse Compact reports every 16 ms even when held still (24.5 ms worst gap measured).
    int64_t stale_axes_timeout_ns = 250'000'000;
};

/*!
 * @brief Reads one SpaceMouse's raw HID node, reopening it after a disconnect.
 *
 * poll() opens the device when due, drains every pending report into state(), and closes it once it
 * goes away. A closed device can report no more releases or motion, so its state is cleared, and an
 * open one whose axes go stale has them zeroed.
 */
class SpaceMouseConnection
{
public:
    explicit SpaceMouseConnection(SpaceMouseConnectionConfig config);
    ~SpaceMouseConnection();

    SpaceMouseConnection(const SpaceMouseConnection&) = delete;
    SpaceMouseConnection& operator=(const SpaceMouseConnection&) = delete;

    void poll(int64_t now_ns);

    ConnectionStatus status() const
    {
        return status_;
    }
    //! Why the last open failed; empty unless status() is OpenFailed.
    const std::string& error() const
    {
        return error_;
    }
    const SpaceMouseState& state() const
    {
        return state_;
    }

private:
    void open_device();
    void close_device();
    //! Drains pending reports; false once the device is gone.
    bool read_reports(int64_t now_ns);
    //! Zeroes deflected axes that have had no report for stale_axes_timeout_ns.
    void expire_stale_axes(int64_t now_ns);

    SpaceMouseConnectionConfig config_;
    int fd_ = -1;
    std::filesystem::path open_path_;
    bool combined_report_ = false;
    std::optional<int64_t> next_open_ns_;
    ConnectionStatus status_ = ConnectionStatus::NoDevice;
    bool status_logged_ = false; //!< The current status() has been logged once.
    std::string error_;
    SpaceMouseState state_;
    int64_t last_report_ns_ = 0;
    std::shared_ptr<spdlog::logger> logger_ =
        isaaccapture::Logger::get("isaaccapture.plugins.spacemouse.SpaceMouseConnection");
};

} // namespace spacemouse
} // namespace plugins
