// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <camera_core/types.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace core::camera
{

// Changes only when a reader and a plugin of different versions could misread each other.
// A field or a message added to the schema does not change it.
inline constexpr std::uint32_t WIRE_PROTOCOL_VERSION = 1;

// No message is larger than this, so a receive buffer of this size always fits one.
inline constexpr std::size_t MAX_WIRE_MESSAGE_BYTES = 64 * 1024;

// Reader to plugin, first message on a connection.
struct HelloMessage
{
    std::uint32_t protocol_version = WIRE_PROTOCOL_VERSION;
    std::string reader_name; // for the plugin's log
};

// Plugin to reader, in answer to HelloMessage.
struct WelcomeMessage
{
    std::uint32_t protocol_version = WIRE_PROTOCOL_VERSION;
    std::vector<GroupDescription> groups; // what the plugin publishes
    std::string error; // why the plugin refuses the reader, if it does
};

// Reader to plugin: start sending a group's frame sets. One subscription per connection.
struct SubscribeMessage
{
    std::string group;
    ReaderOptions options;
};

// Plugin to reader, in answer to SubscribeMessage. On success one file descriptor per stream
// travels with it, in stream order, for the stream's pool.
struct SubscribedMessage
{
    std::string error; // why the subscription was refused, if it was
    GroupDescription group;
};

// Reader to plugin: done with a frame set.
struct ReleaseMessage
{
    std::uint64_t lease_id = 0;
};

// What decode_wire_message returns. A frame set travels as a FrameEvent. std::monostate is a
// message from a newer peer that this build does not know; ignore it.
using WireMessage =
    std::variant<std::monostate, HelloMessage, WelcomeMessage, SubscribeMessage, SubscribedMessage, FrameEvent, ReleaseMessage>;

/**
 * @brief Bytes that are not a message, or a message this build cannot represent.
 */
class WireError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

/**
 * @brief Turns messages into bytes. It reuses one buffer, so after the first few messages
 * encoding allocates nothing. Not thread-safe.
 */
class WireEncoder
{
public:
    WireEncoder();
    ~WireEncoder();

    WireEncoder(const WireEncoder&) = delete;
    WireEncoder& operator=(const WireEncoder&) = delete;

    /**
     * @return The message's bytes, valid until the next call on this encoder.
     * @throws std::length_error if the message is larger than MAX_WIRE_MESSAGE_BYTES.
     */
    std::span<const std::byte> encode(const HelloMessage& message);
    std::span<const std::byte> encode(const WelcomeMessage& message);
    std::span<const std::byte> encode(const SubscribeMessage& message);
    std::span<const std::byte> encode(const SubscribedMessage& message);
    std::span<const std::byte> encode(const FrameEvent& message);
    std::span<const std::byte> encode(const ReleaseMessage& message);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief Read one message. The bytes come from another process and are checked in full.
 * @throws WireError if they are not a message, or name more streams, planes or images than a
 * group can have, or a format this build does not know.
 */
WireMessage decode_wire_message(std::span<const std::byte> bytes);

} // namespace core::camera
