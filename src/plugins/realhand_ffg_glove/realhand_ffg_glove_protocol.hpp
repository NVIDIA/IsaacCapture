// SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace plugins::realhand_ffg_glove
{

inline constexpr uint8_t kFrameHeader = 0x5D;
inline constexpr std::size_t kNumSensors = 21;
inline constexpr std::array<int, 4> kDefaultBaudrates = { 2'000'000, 460'800, 1'000'000, 921'600 };

enum class Command : uint8_t
{
    Version = 0x01,
    SetFlag = 0x02,
    Position = 0x03,
    ForceFeedback = 0x04,
    A3Position = 0xA3,
    A6Position = 0xA6,
    A7Force = 0xA7,
};

enum class HandSide
{
    Left,
    Right,
};

struct Frame
{
    Command command;
    std::vector<uint8_t> payload;
};

struct VersionInfo
{
    std::string version;
    HandSide side;
};

class FrameParser
{
public:
    std::optional<Frame> process(uint8_t byte);
    void reset();

private:
    enum class State
    {
        Header,
        Command,
        Length,
        Payload,
        Checksum,
    };

    State state_ = State::Header;
    uint8_t command_ = 0;
    uint8_t payload_size_ = 0;
    uint8_t checksum_ = 0;
    std::vector<uint8_t> payload_;
};

std::vector<uint8_t> pack_frame(Command command, const std::vector<uint8_t>& payload = {});
std::optional<VersionInfo> decode_version(const Frame& frame);
std::optional<std::array<float, kNumSensors>> decode_positions(const Frame& frame);

} // namespace plugins::realhand_ffg_glove
