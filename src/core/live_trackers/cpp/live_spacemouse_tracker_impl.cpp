// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "live_spacemouse_tracker_impl.hpp"

#include <deviceio_trackers/spacemouse_tracker.hpp>
#include <mcap/recording_traits.hpp>
#include <schema/serialized.hpp>
#include <schema/spacemouse_bfbs_generated.h>
#include <schema/timestamp_generated.h>

#include <algorithm>
#include <optional>
#include <utility>

// Reading /dev/hidrawN is Linux-only; elsewhere the live SpaceMouse reports no device.
#ifdef __linux__
#    include <cerrno>
#    include <cstring>
#    include <fcntl.h>
#    include <unistd.h>
#endif

namespace core
{

namespace
{

// Combined (Universal Receiver) reports are 13 bytes: report ID + 6 translation bytes +
// 6 rotation bytes. Separate reports are 7 bytes: report ID + 6 axis bytes.
constexpr size_t kCombinedReportSize = 13;
constexpr size_t kSeparateReportSize = 7;
constexpr double kAxisScale = 350.0;

// Two bytes, little-endian, to a signed 16-bit integer.
int16_t to_int16(uint8_t low, uint8_t high)
{
    return static_cast<int16_t>(static_cast<uint16_t>(low) | (static_cast<uint16_t>(high) << 8));
}

// Normalize to [-1, 1]: 350 raw counts is full deflection.
[[maybe_unused]] float convert_axis(uint8_t low, uint8_t high)
{
    return static_cast<float>(std::clamp(static_cast<double>(to_int16(low, high)) / kAxisScale, -1.0, 1.0));
}

void close_fd([[maybe_unused]] int fd)
{
#ifdef __linux__
    close(fd);
#endif
}

} // namespace

std::unique_ptr<SpaceMouseMcapChannels> LiveSpaceMouseTrackerImpl::create_mcap_channels(mcap::McapWriter& writer,
                                                                                        std::string_view base_name)
{
    return std::make_unique<SpaceMouseMcapChannels>(
        writer, base_name,
        std::vector<std::string>(SpaceMouseRecordingTraits::recording_channels.begin(),
                                 SpaceMouseRecordingTraits::recording_channels.end()));
}

LiveSpaceMouseTrackerImpl::LiveSpaceMouseTrackerImpl(std::string device_path,
                                                     bool combined_report,
                                                     std::unique_ptr<SpaceMouseMcapChannels> mcap_channels)
    : configured_path_(std::move(device_path)),
      configured_combined_report_(combined_report),
      mcap_channels_(std::move(mcap_channels))
{
}

LiveSpaceMouseTrackerImpl::~LiveSpaceMouseTrackerImpl()
{
    // A session ending is not a disconnect: close quietly.
    if (device_fd_ >= 0)
        close_fd(device_fd_);
}

void LiveSpaceMouseTrackerImpl::update(int64_t monotonic_time_ns)
{
    // Invalidate first, publish last: the encode below is the only writer.
    tracked_.reset();

    if (device_fd_ < 0 && monotonic_time_ns >= next_open_attempt_ns_)
    {
        next_open_attempt_ns_ = monotonic_time_ns + REOPEN_INTERVAL_NS;
        open_device();
    }

    // Read on the host clock, so every timestamp field is the monotonic time.
    const DeviceDataTimestamp timestamp(monotonic_time_ns, monotonic_time_ns, monotonic_time_ns);

    if (device_fd_ >= 0 && !read_reports())
        close_device();

    if (device_fd_ < 0)
    {
        // No SpaceMouse this frame: nothing to report.
        tracked_ = publish_and_record<SpaceMouseOutputRecord>(mcap_channels_.get(), 0, timestamp, nullptr);
        return;
    }

    SpaceMouseOutputT native;
    native.translation.assign(translation_.begin(), translation_.end());
    native.rotation.assign(rotation_.begin(), rotation_.end());
    native.pressed_buttons.assign(pressed_buttons_.begin(), pressed_buttons_.end());
    native.is_valid = true;
    tracked_ = publish_and_record(mcap_channels_.get(), 0, timestamp, &native);
}

const Serialized<SpaceMouseOutput>& LiveSpaceMouseTrackerImpl::get_data() const
{
    return tracked_;
}

bool LiveSpaceMouseTrackerImpl::open_device()
{
#ifndef __linux__
    if (!warned_missing_)
    {
        logger_->warn("Live SpaceMouse input needs Linux hidraw; no SpaceMouse will be read");
        warned_missing_ = true;
    }
    return false;
#else
    std::optional<SpaceMouseDevice> device =
        configured_path_.empty() ?
            discover_spacemouse_device() :
            std::optional<SpaceMouseDevice>(SpaceMouseDevice{ configured_path_, configured_combined_report_ });
    if (!device)
    {
        if (!warned_missing_)
        {
            logger_->info("No SpaceMouse found under /sys/class/hidraw; waiting for one to connect");
            warned_missing_ = true;
        }
        return false;
    }

    const int fd = open(device->device_path.c_str(), O_RDONLY | O_NONBLOCK);
    if (fd < 0)
    {
        if (!warned_missing_)
        {
            // EACCES is the common case: /dev/hidraw* is root-only without a udev rule.
            logger_->warn("Cannot open SpaceMouse {}: {} (reading /dev/hidraw* needs a udev rule granting access)",
                          device->device_path, std::strerror(errno));
            warned_missing_ = true;
        }
        return false;
    }

    translation_.fill(0.0F);
    rotation_.fill(0.0F);
    pressed_buttons_.clear();

    device_fd_ = fd;
    open_path_ = device->device_path;
    combined_report_ = device->combined_report;
    warned_missing_ = false;
    logger_->info("Opened SpaceMouse {}{}", open_path_, combined_report_ ? " (combined report)" : "");
    return true;
#endif
}

void LiveSpaceMouseTrackerImpl::close_device()
{
    close_fd(device_fd_);
    device_fd_ = -1;
    logger_->info("SpaceMouse {} disconnected", open_path_);
    open_path_.clear();
    // A closed device can no longer report releases or motion -- forget its state so a stale
    // button doesn't stick and stale nonzero axes don't keep commanding motion.
    translation_.fill(0.0F);
    rotation_.fill(0.0F);
    pressed_buttons_.clear();
}

bool LiveSpaceMouseTrackerImpl::read_reports()
{
#ifndef __linux__
    return false;
#else
    const size_t report_size = combined_report_ ? kCombinedReportSize : kSeparateReportSize;
    while (true)
    {
        uint8_t buffer[kCombinedReportSize];
        const ssize_t n = read(device_fd_, buffer, report_size);
        if (n < 0)
        {
            if (errno == EAGAIN || errno == EWOULDBLOCK)
                return true;
            if (errno == EINTR)
                continue;
            // I/O error (ENODEV on unplug): the device is gone.
            return false;
        }
        if (n == 0)
            return false; // end of file: the device is gone

        const auto size = static_cast<size_t>(n);
        const uint8_t report_id = buffer[0];
        if (report_id == 1 && size >= kSeparateReportSize)
        {
            translation_ = { convert_axis(buffer[1], buffer[2]), convert_axis(buffer[3], buffer[4]),
                             convert_axis(buffer[5], buffer[6]) };
            if (combined_report_ && size >= kCombinedReportSize)
            {
                rotation_ = { convert_axis(buffer[7], buffer[8]), convert_axis(buffer[9], buffer[10]),
                              convert_axis(buffer[11], buffer[12]) };
            }
        }
        else if (report_id == 2 && !combined_report_ && size >= kSeparateReportSize)
        {
            rotation_ = { convert_axis(buffer[1], buffer[2]), convert_axis(buffer[3], buffer[4]),
                          convert_axis(buffer[5], buffer[6]) };
        }
        else if (report_id == 3 && size >= 2)
        {
            // Button report: buffer[1] is a bitmask (bit i = button i currently held).
            pressed_buttons_.clear();
            for (uint16_t bit = 0; bit < 8; ++bit)
            {
                if ((buffer[1] & (1U << bit)) != 0U)
                    pressed_buttons_.insert(bit);
            }
        }
    }
#endif
}

} // namespace core
