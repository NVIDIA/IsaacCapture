// SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "realhand_ffg_glove_protocol.hpp"

#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace plugins::realhand_ffg_glove
{

namespace
{

constexpr float kDegreesToRadians = 0.01745329251994329577f;

template <typename T>
T read_little_endian(const uint8_t* bytes)
{
    std::array<uint8_t, sizeof(T)> native{};
    if constexpr (std::endian::native == std::endian::little)
    {
        std::memcpy(native.data(), bytes, sizeof(T));
    }
    else
    {
        for (std::size_t i = 0; i < sizeof(T); ++i)
        {
            native[i] = bytes[sizeof(T) - 1 - i];
        }
    }
    T value{};
    std::memcpy(&value, native.data(), sizeof(T));
    return value;
}

} // namespace

void FrameParser::reset()
{
    state_ = State::Header;
    command_ = 0;
    payload_size_ = 0;
    checksum_ = 0;
    payload_.clear();
}

std::optional<Frame> FrameParser::process(uint8_t byte)
{
    switch (state_)
    {
    case State::Header:
        if (byte == kFrameHeader)
        {
            checksum_ = byte;
            state_ = State::Command;
        }
        return std::nullopt;
    case State::Command:
        command_ = byte;
        checksum_ = static_cast<uint8_t>(checksum_ + byte);
        state_ = State::Length;
        return std::nullopt;
    case State::Length:
        payload_size_ = byte;
        checksum_ = static_cast<uint8_t>(checksum_ + byte);
        payload_.clear();
        payload_.reserve(payload_size_);
        state_ = payload_size_ == 0 ? State::Checksum : State::Payload;
        return std::nullopt;
    case State::Payload:
        payload_.push_back(byte);
        checksum_ = static_cast<uint8_t>(checksum_ + byte);
        if (payload_.size() == payload_size_)
        {
            state_ = State::Checksum;
        }
        return std::nullopt;
    case State::Checksum:
        if (byte == checksum_)
        {
            Frame frame{ static_cast<Command>(command_), payload_ };
            reset();
            return frame;
        }
        reset();
        if (byte == kFrameHeader)
        {
            checksum_ = byte;
            state_ = State::Command;
        }
        return std::nullopt;
    }
    reset();
    return std::nullopt;
}

std::vector<uint8_t> pack_frame(Command command, const std::vector<uint8_t>& payload)
{
    if (payload.size() > std::numeric_limits<uint8_t>::max())
    {
        return {};
    }
    std::vector<uint8_t> frame;
    frame.reserve(payload.size() + 4);
    frame.push_back(kFrameHeader);
    frame.push_back(static_cast<uint8_t>(command));
    frame.push_back(static_cast<uint8_t>(payload.size()));
    frame.insert(frame.end(), payload.begin(), payload.end());

    uint8_t checksum = 0;
    for (uint8_t byte : frame)
    {
        checksum = static_cast<uint8_t>(checksum + byte);
    }
    frame.push_back(checksum);
    return frame;
}

std::optional<VersionInfo> decode_version(const Frame& frame)
{
    if (frame.command != Command::Version || frame.payload.size() < 5 || frame.payload[4] > 1)
    {
        return std::nullopt;
    }
    const uint32_t packed = read_little_endian<uint32_t>(frame.payload.data());
    char digits[16]{};
    std::snprintf(digits, sizeof(digits), "%05u", packed);
    const std::string text(digits);
    VersionInfo result;
    result.version = std::to_string(text[0] - '0') + "." + std::to_string(std::stoi(text.substr(1, 2))) + "." +
                     std::to_string(std::stoi(text.substr(3, 2)));
    result.side = frame.payload[4] == 0 ? HandSide::Left : HandSide::Right;
    return result;
}

std::optional<std::array<float, kNumSensors>> decode_positions(const Frame& frame)
{
    std::array<float, kNumSensors> result{};
    if (frame.command == Command::Position || frame.command == Command::A3Position)
    {
        if (frame.payload.size() != kNumSensors * sizeof(float))
        {
            return std::nullopt;
        }
        for (std::size_t i = 0; i < kNumSensors; ++i)
        {
            const float degrees = read_little_endian<float>(frame.payload.data() + i * sizeof(float));
            if (!std::isfinite(degrees))
            {
                return std::nullopt;
            }
            result[i] = degrees * kDegreesToRadians;
        }
        return result;
    }
    if (frame.command == Command::A6Position)
    {
        if (frame.payload.size() != kNumSensors * sizeof(int16_t))
        {
            return std::nullopt;
        }
        for (std::size_t i = 0; i < kNumSensors; ++i)
        {
            const int16_t hundredth_degrees = read_little_endian<int16_t>(frame.payload.data() + i * sizeof(int16_t));
            result[i] = static_cast<float>(hundredth_degrees) * 0.01f * kDegreesToRadians;
        }
        return result;
    }
    return std::nullopt;
}

} // namespace plugins::realhand_ffg_glove
