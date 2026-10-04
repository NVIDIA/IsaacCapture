// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pusherio/schema_pusher.hpp>

#include <cstdint>
#include <memory>
#include <set>
#include <string>

namespace core
{
class OpenXRSession;
}

namespace plugins
{
namespace keyboard
{

/*!
 * @brief Reads a Linux evdev keyboard device (e.g. /dev/input/event3), tracks
 *        the set of currently-held key codes, and pushes KeyboardOutput via
 *        OpenXR SchemaPusher. Carries no semantic mapping -- keys are reported
 *        as-is (evdev key codes, see linux/input-event-codes.h).
 */
class KeyboardPlugin
{
public:
    KeyboardPlugin(const std::string& device_path, const std::string& collection_id);
    ~KeyboardPlugin();

    void update();

private:
    bool open_device();
    void close_device();
    void push_current_state();
    // Rebuilds pressed_keys_ from the kernel's authoritative EVIOCGKEY bitmap --
    // used on open and after a SYN_DROPPED (evdev client-side buffer overrun) so a
    // dropped release event can't leave a key stuck "pressed" forever. Returns
    // false on ioctl failure, in which case pressed_keys_ is left untouched --
    // callers recovering from a drop must not treat that as resynchronized.
    bool resync_pressed_keys();

    std::string device_path_;
    int device_fd_ = -1;
    // Set between a SYN_DROPPED and the next SYN_REPORT: per the evdev protocol,
    // events in that window are incomplete and must be discarded rather than
    // applied individually.
    bool awaiting_syn_report_ = false;

    std::set<uint16_t> pressed_keys_;

    std::shared_ptr<core::OpenXRSession> session_;
    core::SchemaPusher pusher_;
};

} // namespace keyboard
} // namespace plugins
