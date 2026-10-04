// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "keyboard_plugin.hpp"

#include <flatbuffers/flatbuffers.h>
#include <linux/input.h>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/os_time.hpp>
#include <schema/keyboard_generated.h>
#include <sys/ioctl.h>
#include <sys/select.h>

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <unistd.h>

namespace plugins
{
namespace keyboard
{

namespace
{

constexpr size_t kInputEventSize = sizeof(input_event);
constexpr size_t kMaxFlatbufferSize = 2048;

} // namespace

KeyboardPlugin::KeyboardPlugin(const std::string& device_path, const std::string& collection_id)
    : device_path_(device_path),
      session_(std::make_shared<core::OpenXRSession>("KeyboardPlugin", core::SchemaPusher::get_required_extensions())),
      pusher_(session_->get_handles(),
              core::SchemaPusherConfig{ .collection_id = collection_id,
                                        .max_flatbuffer_size = kMaxFlatbufferSize,
                                        .tensor_identifier = "keyboard",
                                        .localized_name = "Keyboard",
                                        .app_name = "KeyboardPlugin" })
{
    if (!open_device())
        throw std::runtime_error("KeyboardPlugin: Failed to open " + device_path + " (" + strerror(errno) + ")");
}

KeyboardPlugin::~KeyboardPlugin()
{
    if (device_fd_ >= 0)
        close_device();
}

void KeyboardPlugin::update()
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
            // If there is no data to read (ret == 0) or the device file descriptor is not set in
            // the read set, break out of the loop; this means there's no new event available.
            break;
        }

        input_event event;
        ssize_t n = read(device_fd_, &event, kInputEventSize);
        if (n != static_cast<ssize_t>(kInputEventSize))
        {
            if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
                break;
            close_device();
            push_current_state();
            return;
        }

        if (event.type == EV_SYN)
        {
            if (event.code == SYN_DROPPED)
            {
                // The kernel-side event queue overran; everything until the next
                // SYN_REPORT is incomplete and must be discarded rather than applied.
                awaiting_syn_report_ = true;
            }
            else if (event.code == SYN_REPORT && awaiting_syn_report_)
            {
                // Only leave recovery mode on a successful resync -- if EVIOCGKEY
                // failed, pressed_keys_ is still whatever it was before the drop and
                // must not be trusted yet; keep discarding events until the next
                // SYN_REPORT gives us another chance to resynchronize.
                if (resync_pressed_keys())
                    awaiting_syn_report_ = false;
            }
            continue;
        }

        if (awaiting_syn_report_)
            continue;

        // value: 0 = release, 1 = press, 2 = autorepeat (ignored -- key is already tracked as held).
        if (event.type == EV_KEY && event.value != 2)
        {
            if (event.value != 0)
                pressed_keys_.insert(event.code);
            else
                pressed_keys_.erase(event.code);
        }
    }

    push_current_state();
}

bool KeyboardPlugin::open_device()
{
    assert(device_fd_ < 0);

    int fd = open(device_path_.c_str(), O_RDONLY | O_NONBLOCK);
    if (fd < 0)
        return false;

    device_fd_ = fd;
    awaiting_syn_report_ = false;
    resync_pressed_keys();
    std::cout << "KeyboardPlugin: Opened " << device_path_ << std::endl;
    return true;
}

void KeyboardPlugin::close_device()
{
    assert(device_fd_ >= 0);

    close(device_fd_);
    device_fd_ = -1;
    awaiting_syn_report_ = false;
    // A closed device can no longer report releases -- forget everything it
    // last reported as held so a stale key doesn't stick "pressed" forever.
    pressed_keys_.clear();
}

bool KeyboardPlugin::resync_pressed_keys()
{
    unsigned char bitmap[(KEY_CNT + 7) / 8] = {};
    if (ioctl(device_fd_, EVIOCGKEY(sizeof(bitmap)), bitmap) < 0)
        return false;

    pressed_keys_.clear();
    for (uint16_t code = 0; code < KEY_CNT; ++code)
    {
        if (bitmap[code / 8] & (1u << (code % 8)))
            pressed_keys_.insert(code);
    }
    return true;
}

void KeyboardPlugin::push_current_state()
{
    core::KeyboardOutputT out;
    out.pressed_keys.assign(pressed_keys_.begin(), pressed_keys_.end());
    out.is_valid = true;

    auto sample_time_ns = core::os_monotonic_now_ns();

    flatbuffers::FlatBufferBuilder builder(kMaxFlatbufferSize);
    auto offset = core::KeyboardOutput::Pack(builder, &out);
    builder.Finish(offset);
    pusher_.push_buffer(builder.GetBufferPointer(), builder.GetSize(), sample_time_ns, sample_time_ns);
}

} // namespace keyboard
} // namespace plugins
