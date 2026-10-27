// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace core
{

// Link message header, 24 bytes little-endian: magic u32 @0 | type u8 @4 | version u8 @5 | channel u16 @6 | seq u32 @8
// | payload_size u32 @12 | t_send_ns i64 @16.
constexpr uint32_t kRemoteLinkMagic = 0x30535452; // "RTS0"
constexpr uint8_t kRemoteLinkVersion = 0;
constexpr size_t kHeaderSize = 24;
constexpr uint32_t kMaxPayloadSize = 16u << 20;

struct LinkMessageHeader
{
    uint8_t type = 0;
    uint16_t channel = 0;
    uint32_t seq = 0;
    uint32_t payload_size = 0;
    //! Sender CLOCK_MONOTONIC, stamped immediately before the transport send.
    int64_t t_send_ns = 0;
};

class LinkMessageDecodeError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

void encode_header(const LinkMessageHeader& header, uint8_t* out24);
//! Validates magic, version and payload size only; any type value is accepted.
LinkMessageHeader decode_header(const uint8_t* in24);
//! Header (seq and t_send zero) followed by `payload`. Throws std::invalid_argument above kMaxPayloadSize.
std::vector<uint8_t> encode_link_message(uint8_t type, std::span<const uint8_t> payload);
//! Overwrites seq and t_send of an encoded link message in place.
void stamp_header(uint8_t* message, uint32_t seq, int64_t t_send_ns);

} // namespace core
