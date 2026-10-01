// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "live_gamepad_tracker_impl.hpp"

#include <deviceio_trackers/gamepad_tracker.hpp>
#include <mcap/recording_traits.hpp>
#include <schema/gamepad_bfbs_generated.h>
#include <schema/serialized.hpp>
#include <schema/timestamp_generated.h>

#include <algorithm>
#include <optional>
#include <utility>

// The joystick API is Linux-only; elsewhere the live gamepad reports no device.
#ifdef __linux__
#    include <linux/joystick.h>
#    include <sys/ioctl.h>

#    include <cerrno>
#    include <cstring>
#    include <fcntl.h>
#    include <unistd.h>
#endif

namespace core
{

namespace
{

constexpr double kMaxAxisValue = 32767.0;
// Fallback axis count when JSIOCGAXES is unavailable -- covers the common left/right-stick +
// trigger + dpad layout (8 axes) reported by most Xbox-style gamepads under the xpad driver.
[[maybe_unused]] constexpr uint8_t kDefaultAxisCount = 8;

[[maybe_unused]] float normalize_axis(int16_t raw_value)
{
    return static_cast<float>(std::clamp(static_cast<double>(raw_value) / kMaxAxisValue, -1.0, 1.0));
}

void close_fd([[maybe_unused]] int fd)
{
#ifdef __linux__
    close(fd);
#endif
}

} // namespace

std::unique_ptr<GamepadMcapChannels> LiveGamepadTrackerImpl::create_mcap_channels(mcap::McapWriter& writer,
                                                                                  std::string_view base_name)
{
    return std::make_unique<GamepadMcapChannels>(
        writer, base_name,
        std::vector<std::string>(
            GamepadRecordingTraits::recording_channels.begin(), GamepadRecordingTraits::recording_channels.end()));
}

LiveGamepadTrackerImpl::LiveGamepadTrackerImpl(std::string device_path, std::unique_ptr<GamepadMcapChannels> mcap_channels)
    : configured_path_(std::move(device_path)), mcap_channels_(std::move(mcap_channels))
{
}

LiveGamepadTrackerImpl::~LiveGamepadTrackerImpl()
{
    // A session ending is not a disconnect: close quietly.
    if (device_fd_ >= 0)
        close_fd(device_fd_);
}

void LiveGamepadTrackerImpl::update(int64_t monotonic_time_ns)
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

    if (device_fd_ >= 0 && !read_events())
        close_device();

    if (device_fd_ < 0)
    {
        // No gamepad this frame: nothing to report.
        tracked_ = publish_and_record<GamepadOutputRecord>(mcap_channels_.get(), 0, timestamp, nullptr);
        return;
    }

    GamepadOutputT native;
    native.pressed_buttons.assign(pressed_buttons_.begin(), pressed_buttons_.end());
    native.axes = axes_;
    native.is_valid = true;
    tracked_ = publish_and_record(mcap_channels_.get(), 0, timestamp, &native);
}

const Serialized<GamepadOutput>& LiveGamepadTrackerImpl::get_data() const
{
    return tracked_;
}

bool LiveGamepadTrackerImpl::open_device()
{
#ifndef __linux__
    if (!warned_missing_)
    {
        logger_->warn("Live gamepad input needs the Linux joystick API; no gamepad will be read");
        warned_missing_ = true;
    }
    return false;
#else
    std::optional<std::string> path =
        configured_path_.empty() ? discover_gamepad_device() : std::optional<std::string>(configured_path_);
    if (!path)
    {
        if (!warned_missing_)
        {
            logger_->info("No gamepad found under /dev/input/by-path; waiting for one to connect");
            warned_missing_ = true;
        }
        return false;
    }

    const int fd = open(path->c_str(), O_RDONLY | O_NONBLOCK);
    if (fd < 0)
    {
        if (!warned_missing_)
        {
            logger_->warn("Cannot open gamepad {}: {}", *path, std::strerror(errno));
            warned_missing_ = true;
        }
        return false;
    }

    uint8_t axis_count = kDefaultAxisCount;
    ioctl(fd, JSIOCGAXES, &axis_count);
    axes_.assign(axis_count, 0.0F);
    pressed_buttons_.clear();

    device_fd_ = fd;
    open_path_ = *path;
    warned_missing_ = false;
    logger_->info("Opened gamepad {} ({} axes)", open_path_, static_cast<int>(axis_count));
    return true;
#endif
}

void LiveGamepadTrackerImpl::close_device()
{
    close_fd(device_fd_);
    device_fd_ = -1;
    logger_->info("Gamepad {} disconnected", open_path_);
    open_path_.clear();
    // A closed device can no longer report releases or the sticks returning to center -- forget
    // its state so nothing stale keeps commanding motion.
    pressed_buttons_.clear();
    std::fill(axes_.begin(), axes_.end(), 0.0F);
}

bool LiveGamepadTrackerImpl::read_events()
{
#ifndef __linux__
    return false;
#else
    while (true)
    {
        js_event event;
        const ssize_t n = read(device_fd_, &event, sizeof(event));
        if (n == static_cast<ssize_t>(sizeof(event)))
        {
            const auto type = static_cast<uint8_t>(event.type & ~JS_EVENT_INIT);
            if (type == JS_EVENT_AXIS && event.number < axes_.size())
            {
                axes_[event.number] = normalize_axis(event.value);
            }
            else if (type == JS_EVENT_BUTTON)
            {
                if (event.value != 0)
                    pressed_buttons_.insert(event.number);
                else
                    pressed_buttons_.erase(event.number);
            }
            continue;
        }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
            return true;
        if (n < 0 && errno == EINTR)
            continue;
        // End of file, a partial event, or an I/O error (ENODEV on unplug): the device is gone.
        return false;
    }
#endif
}

} // namespace core
