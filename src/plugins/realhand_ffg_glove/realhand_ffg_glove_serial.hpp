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
    SerialGlove(const std::string& port, int baudrate);
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
std::optional<VersionInfo> probe_glove(SerialGlove& glove, int timeout_ms = 250);

} // namespace plugins::realhand_ffg_glove
