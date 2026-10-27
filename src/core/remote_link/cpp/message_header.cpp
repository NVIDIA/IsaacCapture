// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/remote_link/message_header.hpp"

#include "byte_order.hpp"

#include <algorithm>
#include <string>

namespace core
{

using remote_link_detail::read_le;
using remote_link_detail::write_le;

void encode_header(const LinkMessageHeader& header, uint8_t* out24)
{
    write_le<uint32_t>(out24, kRemoteLinkMagic);
    out24[4] = header.type;
    out24[5] = kRemoteLinkVersion;
    write_le<uint16_t>(out24 + 6, header.channel);
    write_le<uint32_t>(out24 + 8, header.seq);
    write_le<uint32_t>(out24 + 12, header.payload_size);
    write_le<int64_t>(out24 + 16, header.t_send_ns);
}

LinkMessageHeader decode_header(const uint8_t* in24)
{
    if (read_le<uint32_t>(in24) != kRemoteLinkMagic)
    {
        throw LinkMessageDecodeError("remote link: bad message magic");
    }
    if (in24[5] != kRemoteLinkVersion)
    {
        throw LinkMessageDecodeError("remote link: unsupported protocol version " + std::to_string(in24[5]));
    }
    LinkMessageHeader header;
    header.type = in24[4];
    header.channel = read_le<uint16_t>(in24 + 6);
    header.seq = read_le<uint32_t>(in24 + 8);
    header.payload_size = read_le<uint32_t>(in24 + 12);
    header.t_send_ns = read_le<int64_t>(in24 + 16);
    if (header.payload_size > kMaxPayloadSize)
    {
        throw LinkMessageDecodeError("remote link: payload of " + std::to_string(header.payload_size) +
                                     " bytes exceeds the limit");
    }
    return header;
}

std::vector<uint8_t> encode_link_message(uint8_t type, std::span<const uint8_t> payload)
{
    if (payload.size() > kMaxPayloadSize)
    {
        throw std::invalid_argument("remote link: payload of " + std::to_string(payload.size()) +
                                    " bytes exceeds the limit");
    }
    std::vector<uint8_t> message(kHeaderSize + payload.size());
    LinkMessageHeader header;
    header.type = type;
    header.payload_size = static_cast<uint32_t>(payload.size());
    encode_header(header, message.data());
    std::copy(payload.begin(), payload.end(), message.begin() + kHeaderSize);
    return message;
}

void stamp_header(uint8_t* message, uint32_t seq, int64_t t_send_ns)
{
    write_le<uint32_t>(message + 8, seq);
    write_le<int64_t>(message + 16, t_send_ns);
}

} // namespace core
