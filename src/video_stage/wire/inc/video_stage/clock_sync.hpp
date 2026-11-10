// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// The publisher's half of the PC's four-timestamp clock exchange: parse a request, build the reply. All times are
// CLOCK_MONOTONIC nanoseconds, the clock the PC maps. Wire format (little-endian), shared with
// examples/egocap/backend/egocap_web/livekit_clock_sync.py and the egocap emulator:
//   request, 13 bytes: 'Q', u32 sequence, i64 t1 (the requester's clock)
//   reply,   29 bytes: 'R', u32 sequence, i64 t1 (both echoed), i64 t2 (receipt), i64 t3 (reply send)
// No LiveKit type appears here, so the format is testable on its own.

#pragma once

#include <cstddef>
#include <cstdint>
#include <ctime>
#include <optional>
#include <vector>

namespace plugins
{
namespace sensing
{

inline constexpr const char* kClockSyncTopic = "isaaccapture.clock-sync";
inline constexpr std::size_t kClockSyncRequestSize = 13;
inline constexpr std::size_t kClockSyncReplySize = 29;
/// Byte offset of t3 in a reply, so it can be written as late as possible, just before the send.
inline constexpr std::size_t kClockSyncReplyT3Offset = 21;

struct ClockSyncRequest
{
    std::uint32_t sequence = 0;
    std::int64_t t1 = 0;
};

inline std::int64_t monotonic_ns()
{
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<std::int64_t>(ts.tv_sec) * 1'000'000'000LL + ts.tv_nsec;
}

inline void put_le(std::vector<std::uint8_t>& out, std::size_t at, std::uint64_t value, std::size_t bytes)
{
    for (std::size_t i = 0; i < bytes; ++i)
        out[at + i] = static_cast<std::uint8_t>((value >> (8 * i)) & 0xFF);
}

inline std::uint64_t get_le(const std::vector<std::uint8_t>& in, std::size_t at, std::size_t bytes)
{
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < bytes; ++i)
        value |= static_cast<std::uint64_t>(in[at + i]) << (8 * i);
    return value;
}

/// The request in `data`, or nullopt for anything that is not exactly a request.
inline std::optional<ClockSyncRequest> parse_clock_sync_request(const std::vector<std::uint8_t>& data)
{
    if (data.size() != kClockSyncRequestSize || data[0] != 'Q')
        return std::nullopt;
    ClockSyncRequest request;
    request.sequence = static_cast<std::uint32_t>(get_le(data, 1, 4));
    request.t1 = static_cast<std::int64_t>(get_le(data, 5, 8));
    return request;
}

inline std::vector<std::uint8_t> build_clock_sync_reply(const ClockSyncRequest& request, std::int64_t t2, std::int64_t t3)
{
    std::vector<std::uint8_t> reply(kClockSyncReplySize);
    reply[0] = 'R';
    put_le(reply, 1, request.sequence, 4);
    put_le(reply, 5, static_cast<std::uint64_t>(request.t1), 8);
    put_le(reply, 13, static_cast<std::uint64_t>(t2), 8);
    put_le(reply, kClockSyncReplyT3Offset, static_cast<std::uint64_t>(t3), 8);
    return reply;
}

/// The requester's half of the same exchange, for a peer that sends requests (the PC video stage over its control
/// link).
inline std::vector<std::uint8_t> build_clock_sync_request(const ClockSyncRequest& request)
{
    std::vector<std::uint8_t> out(kClockSyncRequestSize);
    out[0] = 'Q';
    put_le(out, 1, request.sequence, 4);
    put_le(out, 5, static_cast<std::uint64_t>(request.t1), 8);
    return out;
}

struct ClockSyncReply
{
    std::uint32_t sequence = 0;
    std::int64_t t1 = 0;
    std::int64_t t2 = 0;
    std::int64_t t3 = 0;
};

/// The reply in `data`, or nullopt for anything that is not exactly a reply.
inline std::optional<ClockSyncReply> parse_clock_sync_reply(const std::vector<std::uint8_t>& data)
{
    if (data.size() != kClockSyncReplySize || data[0] != 'R')
        return std::nullopt;
    ClockSyncReply reply;
    reply.sequence = static_cast<std::uint32_t>(get_le(data, 1, 4));
    reply.t1 = static_cast<std::int64_t>(get_le(data, 5, 8));
    reply.t2 = static_cast<std::int64_t>(get_le(data, 13, 8));
    reply.t3 = static_cast<std::int64_t>(get_le(data, kClockSyncReplyT3Offset, 8));
    return reply;
}

/// Writes t3 into an already built reply.
inline void set_clock_sync_reply_t3(std::vector<std::uint8_t>& reply, std::int64_t t3)
{
    put_le(reply, kClockSyncReplyT3Offset, static_cast<std::uint64_t>(t3), 8);
}

} // namespace sensing
} // namespace plugins
