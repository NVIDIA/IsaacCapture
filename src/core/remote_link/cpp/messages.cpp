// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/remote_link/messages.hpp"

#include "byte_order.hpp"
#include "inc/remote_link/message_header.hpp"

#include <algorithm>
#include <string>

namespace core
{

using remote_link_detail::read_le;
using remote_link_detail::write_le;

const char* to_string(LinkMessageType type)
{
    switch (type)
    {
    case LinkMessageType::Hello:
        return "HELLO";
    case LinkMessageType::HelloAck:
        return "HELLO_ACK";
    case LinkMessageType::Heartbeat:
        return "HEARTBEAT";
    case LinkMessageType::Error:
        return "ERROR";
    case LinkMessageType::Ping:
        return "PING";
    case LinkMessageType::Pong:
        return "PONG";
    case LinkMessageType::Status:
        return "STATUS";
    case LinkMessageType::Setup:
        return "SETUP";
    case LinkMessageType::Samples:
        return "SAMPLES";
    case LinkMessageType::Ready:
        return "READY";
    }
    return "UNKNOWN";
}

std::vector<uint8_t> encode_pong(const PongPayload& pong)
{
    std::vector<uint8_t> out(kPongPayloadSize);
    write_le<int64_t>(out.data(), pong.t1_ns);
    write_le<int64_t>(out.data() + 8, pong.t2_ns);
    return out;
}

PongPayload decode_pong(std::span<const uint8_t> payload)
{
    if (payload.size() != kPongPayloadSize)
    {
        throw LinkMessageDecodeError("remote link: PONG payload of " + std::to_string(payload.size()) + " bytes");
    }
    return PongPayload{ read_le<int64_t>(payload.data()), read_le<int64_t>(payload.data() + 8) };
}

void encode_samples_prefix(const SamplesPrefix& prefix, uint8_t* out16)
{
    write_le<uint64_t>(out16, prefix.tick_seq);
    write_le<int64_t>(out16 + 8, prefix.tick_time_ns);
}

std::vector<uint8_t> encode_samples_payload(const SamplesPrefix& prefix, std::span<const uint8_t> records)
{
    std::vector<uint8_t> out(kSamplesPrefixSize + records.size());
    encode_samples_prefix(prefix, out.data());
    std::copy(records.begin(), records.end(), out.begin() + kSamplesPrefixSize);
    return out;
}

SamplesPrefix decode_samples_prefix(std::span<const uint8_t> payload)
{
    if (payload.size() < kSamplesPrefixSize)
    {
        throw LinkMessageDecodeError("remote link: SAMPLES payload of " + std::to_string(payload.size()) +
                                     " bytes is shorter than its prefix");
    }
    return SamplesPrefix{ read_le<uint64_t>(payload.data()), read_le<int64_t>(payload.data() + 8) };
}

} // namespace core
