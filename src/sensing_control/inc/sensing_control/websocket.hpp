// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace plugins
{
namespace sensing
{
namespace websocket
{

/// RFC 6455 opcodes this server understands.
enum Opcode : std::uint8_t
{
    kContinuation = 0x0,
    kText = 0x1,
    kBinary = 0x2,
    kClose = 0x8,
    kPing = 0x9,
    kPong = 0xA,
};

/// SHA-1 of `data` as 20 raw bytes; used only for the opening handshake, never for security.
std::string sha1(std::string_view data);

std::string base64_encode(std::string_view data);

/// The Sec-WebSocket-Accept value for a client's Sec-WebSocket-Key.
std::string accept_key(std::string_view client_key);

/// A parsed client upgrade request.
struct Handshake
{
    bool ok = false;
    /// HTTP status to answer with when !ok (400, 401, 404 or 426), and a plain-text reason.
    int status = 400;
    std::string reason;
    std::string key;
    /// The bearer token from `Authorization: Bearer ...` or the `token=` query parameter; empty when absent.
    std::string token;
    std::string path;
};

/// Parse the request head (everything up to and including the blank line).
Handshake parse_handshake(std::string_view head);

/// The server's answer to a successful or failed handshake.
std::string handshake_response(const Handshake& handshake);

/// Compare two secrets without leaking where they differ.
bool constant_time_equal(std::string_view a, std::string_view b);

struct Frame
{
    bool fin = true;
    std::uint8_t opcode = kText;
    std::string payload;
};

enum class ParseStatus
{
    NeedMore,
    Complete,
    Error,
};

struct ParseResult
{
    ParseStatus status = ParseStatus::NeedMore;
    std::size_t consumed = 0;
    Frame frame;
    /// WebSocket close code to send when status is Error (1002 protocol error, 1009 too big).
    std::uint16_t close_code = 1002;
    std::string error;
};

/// Parse one client frame from the front of `buffer`. Client frames must be masked.
ParseResult parse_frame(std::string_view buffer, std::size_t max_payload);

/// Encode one unmasked server frame.
std::string encode_frame(std::uint8_t opcode, std::string_view payload, bool fin = true);

/// A close frame carrying `code` and a short reason.
std::string encode_close(std::uint16_t code, std::string_view reason);

} // namespace websocket
} // namespace sensing
} // namespace plugins
