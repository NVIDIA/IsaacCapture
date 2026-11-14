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

// The newest and the oldest protocol version this build speaks. A version is added when an
// operation one side depends on is added; a field or a message that only informs needs none.
// The handshake settles which version a connection uses.
inline constexpr std::uint32_t PROTOCOL_VERSION = 1;
inline constexpr std::uint32_t OLDEST_PROTOCOL_VERSION = 1;

// No message is larger than this, so a receive buffer of this size always fits one.
inline constexpr std::size_t MAX_MESSAGE_BYTES = 64 * 1024;

// Reader to plugin, the first message on a connection. A connection carries one subscription,
// so the reader says here what it came for.
struct HandshakeRequest
{
    std::uint32_t protocol_version = PROTOCOL_VERSION; // the newest version the reader speaks
    std::string reader_name; // for the plugin's log
    std::string group; // the group to read; empty to ask only what the plugin publishes
    ReaderOptions options; // for the group

    // Set by decode_message when the request asks for something this build does not know,
    // such as a delivery order added later. The plugin refuses the reader with it.
    std::string unsupported;
};

// A group as a handshake response describes it.
struct GroupEntry
{
    GroupDescription group;

    // Set by decode_message when the group uses a codec, a pixel format or a kind of memory
    // this build does not know. `group` then holds its id only: the group exists and can be
    // listed, and it cannot be read by this build.
    std::string unsupported;
};

// Plugin to reader, the answer to HandshakeRequest and the first message the plugin sends.
struct HandshakeResponse
{
    // The version the connection uses: the older of the reader's and the plugin's newest.
    std::uint32_t protocol_version = PROTOCOL_VERSION;
    std::string error; // why the plugin refuses the reader, if it does; the connection then ends

    // The group that was asked for, with one file descriptor per stream travelling with the
    // message, in stream order, for the stream's pool. Or every group the plugin publishes,
    // when the request named none.
    std::vector<GroupEntry> groups;
};

// Reader to plugin: done with a frame set.
struct ReleaseMessage
{
    std::uint64_t lease_id = 0;
};

// Reader to plugin, on a newest-order subscription: hold the newest frame set after
// `after_sequence` for me, now or as soon as there is one.
struct AcquireMessage
{
    std::uint64_t after_sequence = 0; // the last frame set the reader took; 0 for none
};

// What decode_message returns. A frame set travels as a FrameEvent. std::monostate is a
// message this build does not know and whose sender marked it as one that may be ignored.
using ProtocolMessage =
    std::variant<std::monostate, HandshakeRequest, HandshakeResponse, FrameEvent, ReleaseMessage, AcquireMessage>;

/**
 * @brief Bytes that are not a message, or a message this build cannot act on.
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
     * @throws std::invalid_argument if the message names more streams, images or planes than a
     * group can have. Nothing is cut short to make it fit.
     * @throws std::length_error if the message is larger than MAX_MESSAGE_BYTES.
     */
    std::span<const std::byte> encode(const HandshakeRequest& message);
    std::span<const std::byte> encode(const HandshakeResponse& message);
    std::span<const std::byte> encode(const FrameEvent& message);
    std::span<const std::byte> encode(const ReleaseMessage& message);
    std::span<const std::byte> encode(const AcquireMessage& message);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

/**
 * @brief Read one message. The bytes come from another process: their structure is verified
 * and every count is bounded here. Whether a group's description makes sense (sizes, planes,
 * strides) is not checked here; problem_with() does that where the description is used.
 * @throws ProtocolError if they are not a message, name more streams, planes or images than a
 * group can have, or are an operation this build does not know.
 */
ProtocolMessage decode_message(std::span<const std::byte> bytes);

} // namespace core::camera
