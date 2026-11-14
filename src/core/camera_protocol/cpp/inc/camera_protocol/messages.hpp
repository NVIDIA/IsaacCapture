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
inline constexpr std::uint32_t PROTOCOL_VERSION = 1;

// No message is larger than this, so a receive buffer of this size always fits one.
inline constexpr std::size_t MAX_MESSAGE_BYTES = 64 * 1024;

// Reader to plugin, the first message on a connection. A connection carries one subscription,
// so the reader says here what it came for.
struct HandshakeRequest
{
    std::uint32_t protocol_version = PROTOCOL_VERSION;
    std::string reader_name; // for the plugin's log
    std::string group; // the group to read; empty to ask only what the plugin publishes
    ReaderOptions options; // for the group
};

// Plugin to reader, the answer to HandshakeRequest.
struct HandshakeResponse
{
    std::uint32_t protocol_version = PROTOCOL_VERSION;
    std::string error; // why the plugin refuses the reader, if it does; the connection then ends

    // The group that was asked for, with one file descriptor per stream travelling with the
    // message, in stream order, for the stream's pool. Or every group the plugin publishes,
    // when the request named none.
    std::vector<GroupDescription> groups;
};

// Reader to plugin: done with a frame set.
struct ReleaseMessage
{
    std::uint64_t lease_id = 0;
};

// What decode_message returns. A frame set travels as a FrameEvent. std::monostate is a
// message from a newer peer that this build does not know; ignore it.
using ProtocolMessage = std::variant<std::monostate, HandshakeRequest, HandshakeResponse, FrameEvent, ReleaseMessage>;

/**
 * @brief Bytes that are not a message, or a message this build cannot represent.
 */
class ProtocolError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

/**
 * @brief Turns messages into bytes. It reuses one buffer, so after the first few messages
 * encoding allocates nothing. Not thread-safe.
 */
class MessageEncoder
{
public:
    MessageEncoder();
    ~MessageEncoder();

    MessageEncoder(const MessageEncoder&) = delete;
    MessageEncoder& operator=(const MessageEncoder&) = delete;

    /**
     * @return The message's bytes, valid until the next call on this encoder.
     * @throws std::length_error if the message is larger than MAX_MESSAGE_BYTES.
     */
    std::span<const std::byte> encode(const HandshakeRequest& message);
    std::span<const std::byte> encode(const HandshakeResponse& message);
    std::span<const std::byte> encode(const FrameEvent& message);
    std::span<const std::byte> encode(const ReleaseMessage& message);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief Read one message. The bytes come from another process and are checked in full.
 * @throws ProtocolError if they are not a message, or name more streams, planes or images
 * than a group can have, or a format this build does not know.
 */
ProtocolMessage decode_message(std::span<const std::byte> bytes);

} // namespace core::camera
