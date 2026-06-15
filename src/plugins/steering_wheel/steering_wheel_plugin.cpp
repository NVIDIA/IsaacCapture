// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "steering_wheel_plugin.hpp"

#include <flatbuffers/flatbuffers.h>
#include <linux/joystick.h>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/os_time.hpp>
#include <schema/steering_wheel_generated.h>
#include <sys/ioctl.h>
#include <sys/select.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

namespace plugins
{
namespace steering_wheel
{

namespace
{

constexpr size_t kJsEventSize = sizeof(js_event);
constexpr double kMaxAxisValue = 32767.0;
constexpr size_t kMaxFlatbufferSize = 1024;

float normalize_axis(int16_t raw_value)
{
    return static_cast<float>(std::max(-1.0, std::min(1.0, static_cast<double>(raw_value) / kMaxAxisValue)));
}

} // namespace

SteeringWheelPlugin::SteeringWheelPlugin(const std::string& device_path,
                                         const std::string& collection_id,
                                         SteeringWheelAxisMapping axis_mapping)
    : device_path_(device_path),
      axis_mapping_(axis_mapping),
      session_(
          std::make_shared<core::OpenXRSession>("SteeringWheelPlugin", core::SchemaPusher::get_required_extensions())),
      pusher_(session_->get_handles(),
              core::SchemaPusherConfig{ .collection_id = collection_id,
                                        .max_flatbuffer_size = kMaxFlatbufferSize,
                                        .tensor_identifier = "steering_wheel",
                                        .localized_name = "Steering Wheel",
                                        .app_name = "SteeringWheelPlugin" })
{
    if (!open_device())
        throw std::runtime_error("SteeringWheelPlugin: Failed to open " + device_path + " (" + strerror(errno) + ")");
}

SteeringWheelPlugin::~SteeringWheelPlugin()
{
    close_device();
}

void SteeringWheelPlugin::update()
{
    if (device_fd_ < 0)
    {
        open_device();
        if (device_fd_ < 0)
        {
            push_current_state();
            return;
        }
    }

    fd_set read_fds;
    struct timeval timeout = { 0, 0 };

    while (true)
    {
        FD_ZERO(&read_fds);
        FD_SET(device_fd_, &read_fds);
        timeout = { 0, 0 };

        int ret = select(device_fd_ + 1, &read_fds, nullptr, nullptr, &timeout);
        if (ret < 0)
        {
            if (errno == EINTR)
                return;
            close_device();
            push_current_state();
            return;
        }
        if (ret == 0 || !FD_ISSET(device_fd_, &read_fds))
        {
            break;
        }

        js_event event;
        ssize_t n = read(device_fd_, &event, kJsEventSize);
        if (n != static_cast<ssize_t>(kJsEventSize))
        {
            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
                break;
            close_device();
            push_current_state();
            return;
        }

        const uint8_t event_type = event.type & ~JS_EVENT_INIT;
        if (event_type == JS_EVENT_AXIS)
        {
            if (event.number < axes_.size())
            {
                axes_[event.number] = normalize_axis(event.value);
            }
        }
        else if (event_type == JS_EVENT_BUTTON)
        {
            if (event.number < buttons_.size())
            {
                buttons_[event.number] = event.value ? 1 : 0;
            }
        }
    }

    push_current_state();
}

bool SteeringWheelPlugin::open_device()
{
    if (device_fd_ >= 0)
        return true;

    int fd = open(device_path_.c_str(), O_RDONLY | O_NONBLOCK);
    if (fd < 0)
        return false;

    device_fd_ = fd;
    resize_state_from_device();
    std::cout << "SteeringWheelPlugin: Opened " << device_path_ << std::endl;
    return true;
}

void SteeringWheelPlugin::close_device()
{
    if (device_fd_ < 0)
        return;

    close(device_fd_);
    device_fd_ = -1;
}

void SteeringWheelPlugin::push_current_state()
{
    core::SteeringWheelOutputT out;
    out.steering = axis_value(axis_mapping_.steering_axis);
    out.throttle = axis_value(axis_mapping_.throttle_axis);
    out.brake = axis_value(axis_mapping_.brake_axis);
    out.clutch = axis_value(axis_mapping_.clutch_axis);
    out.buttons = buttons_;
    out.hat_x = hat_[0];
    out.hat_y = hat_[1];

    const auto sample_time_ns = core::os_monotonic_now_ns();

    flatbuffers::FlatBufferBuilder builder(kMaxFlatbufferSize);
    auto offset = core::SteeringWheelOutput::Pack(builder, &out);
    builder.Finish(offset);
    pusher_.push_buffer(builder.GetBufferPointer(), builder.GetSize(), sample_time_ns, sample_time_ns);
}

float SteeringWheelPlugin::axis_value(int axis_index) const
{
    if (axis_index < 0)
        return 0.0f;
    const auto index = static_cast<size_t>(axis_index);
    return index < axes_.size() ? axes_[index] : 0.0f;
}

void SteeringWheelPlugin::resize_state_from_device()
{
    unsigned char axis_count = 0;
    unsigned char button_count = 0;
    if (ioctl(device_fd_, JSIOCGAXES, &axis_count) < 0)
    {
        axis_count = 0;
    }
    if (ioctl(device_fd_, JSIOCGBUTTONS, &button_count) < 0)
    {
        button_count = 0;
    }

    axes_.assign(axis_count, 0.0f);
    buttons_.assign(button_count, 0);
}

} // namespace steering_wheel
} // namespace plugins
