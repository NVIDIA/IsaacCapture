// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"

#include <log_bridge/logger.hpp>

#include <cstddef>
#include <cstring>
#include <revohuman.h>
#include <string>

namespace plugins
{
namespace revohuman
{

inline RhHand to_rh(Hand hand)
{
    return hand == Hand::Left ? RH_HAND_LEFT : RH_HAND_RIGHT;
}

//! A fixed-size C SDK text field, NUL-terminated unless it fills the field.
template <std::size_t N>
std::string sdk_text(const char (&chars)[N])
{
    return std::string(chars, strnlen(chars, N));
}

//! "E1101 TransportDisconnected" for this thread's last C SDK error; the SDK's full text goes to the debug log.
inline std::string sdk_error()
{
    const RhErrorCode code = rh_last_error_code();
    std::string head;
    char buf[512];
    if (rh_last_error(buf, sizeof(buf)) > 0)
    {
        isaaccapture::Logger::get("isaaccapture.plugins.revohuman.sdk")->debug("SDK error: {}", buf);
        head = std::string(buf).substr(0, std::string(buf).find(':'));
    }
    if (head.empty())
    {
        const char* name = rh_error_class_name(code);
        head = "E" + std::to_string(code) + (name != nullptr ? std::string(" ") + name : "");
    }
    return head;
}

} // namespace revohuman
} // namespace plugins
