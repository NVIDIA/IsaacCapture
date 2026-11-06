// SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "realhand_ffg_glove_protocol.hpp"

#include <array>
#include <chrono>
#include <optional>
#include <string>
#include <vector>

namespace plugins::realhand_ffg_glove
{

class SerialGlove
{
public:
    SerialGlove(const std::string& port, int baudrate, bool check_existing_users = true);
    ~SerialGlove();

    SerialGlove(const SerialGlove&) = delete;
    SerialGlove& operator=(const SerialGlove&) = delete;

    void send(Command command, const std::vector<uint8_t>& payload = {});
    std::vector<Frame> read_available(int timeout_ms);
    const std::string& port() const
    {
        return port_;
    }
    int baudrate() const
    {
        return baudrate_;
    }

private:
    int fd_ = -1;
    std::string port_;
    int baudrate_ = 0;
    FrameParser parser_;
};

std::vector<std::string> discover_serial_ports();
bool serial_port_is_open(const std::string& port);

} // namespace plugins::realhand_ffg_glove
