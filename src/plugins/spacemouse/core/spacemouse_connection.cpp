// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/spacemouse/spacemouse_connection.hpp"

#include "inc/spacemouse/device_discovery.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <utility>

namespace plugins
{
namespace spacemouse
{

namespace
{

// Larger than any SpaceMouse report: hidraw returns one report per read.
constexpr std::size_t kReadBufferSize = 64;

} // namespace

SpaceMouseConnection::SpaceMouseConnection(SpaceMouseConnectionConfig config) : config_(std::move(config))
{
}

SpaceMouseConnection::~SpaceMouseConnection()
{
    if (fd_ >= 0)
        ::close(fd_);
}

void SpaceMouseConnection::poll(int64_t now_ns)
{
    if (fd_ < 0 && (!next_open_ns_ || now_ns >= *next_open_ns_))
    {
        next_open_ns_ = now_ns + config_.retry_interval_ns;
        open_device();
        last_report_ns_ = now_ns;
    }
    if (fd_ >= 0 && !read_reports(now_ns))
        close_device();
    if (fd_ >= 0)
        expire_stale_axes(now_ns);
}

void SpaceMouseConnection::open_device()
{
    const std::optional<DeviceLocation> device =
        config_.device_path ?
            std::optional<DeviceLocation>(DeviceLocation{ *config_.device_path, config_.combined_report }) :
            find_device(config_.hidraw_class_dir, config_.dev_dir);
    const bool report = !status_logged_;
    const ConnectionStatus previous = status_;
    if (!device)
    {
        status_ = ConnectionStatus::NoDevice;
        error_.clear();
        if (report || previous != status_)
            logger_->info("No SpaceMouse found; waiting for one to connect");
        status_logged_ = true;
        return;
    }

    const int fd = ::open(device->path.c_str(), O_RDONLY | O_NONBLOCK);
    if (fd < 0)
    {
        const int open_errno = errno;
        status_ = open_errno == ENOENT ? ConnectionStatus::NoDevice : ConnectionStatus::OpenFailed;
        error_ = status_ == ConnectionStatus::OpenFailed ? device->path.string() + ": " + std::strerror(open_errno) : "";
        if (report || previous != status_)
        {
            if (open_errno == ENOENT)
                logger_->info("No SpaceMouse at {}; waiting for it to connect", device->path.string());
            else if (open_errno == EACCES)
                logger_->warn("Cannot open {} ({}): install the SpaceMouse udev rule (see the plugin README)",
                              device->path.string(), std::strerror(open_errno));
            else
                logger_->warn("Cannot open {}: {}", device->path.string(), std::strerror(open_errno));
        }
        status_logged_ = true;
        return;
    }

    fd_ = fd;
    open_path_ = device->path;
    combined_report_ = device->combined_report;
    status_ = ConnectionStatus::Connected;
    error_.clear();
    state_.clear();
    logger_->info("Opened SpaceMouse {}{}", open_path_.string(), combined_report_ ? " (combined report)" : "");
}

void SpaceMouseConnection::close_device()
{
    ::close(fd_);
    fd_ = -1;
    status_ = ConnectionStatus::NoDevice;
    state_.clear();
    logger_->info("SpaceMouse {} disconnected", open_path_.string());
    open_path_.clear();
}

bool SpaceMouseConnection::read_reports(int64_t now_ns)
{
    uint8_t buffer[kReadBufferSize];
    while (true)
    {
        const ssize_t n = ::read(fd_, buffer, sizeof(buffer));
        if (n > 0)
        {
            apply_report(state_, buffer, static_cast<std::size_t>(n), combined_report_);
            last_report_ns_ = now_ns;
            continue;
        }
        if (n < 0 && errno == EINTR)
            continue;
        // EAGAIN: nothing pending. Anything else (ENODEV on unplug, or end of file) means the device is gone.
        return n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK);
    }
}

void SpaceMouseConnection::expire_stale_axes(int64_t now_ns)
{
    const auto deflected = [](const std::array<float, 3>& axes)
    { return std::any_of(axes.begin(), axes.end(), [](float value) { return value != 0.0f; }); };
    if (now_ns - last_report_ns_ < config_.stale_axes_timeout_ns ||
        !(deflected(state_.translation) || deflected(state_.rotation)))
        return;
    state_.translation = {};
    state_.rotation = {};
    logger_->warn("SpaceMouse {} sent no report for {} ms while deflected; zeroing its axes", open_path_.string(),
                  (now_ns - last_report_ns_) / 1'000'000);
}

} // namespace spacemouse
} // namespace plugins
