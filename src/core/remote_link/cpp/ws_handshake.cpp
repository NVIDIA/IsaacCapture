// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "ws_handshake.hpp"

#include <vector>

namespace core::remote_link_detail
{

namespace
{

constexpr char kAcceptGuid[] = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

uint32_t rotl(uint32_t value, int bits)
{
    return (value << bits) | (value >> (32 - bits));
}

} // namespace

std::array<uint8_t, 20> sha1(const std::string& text)
{
    std::array<uint32_t, 5> h{ 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0 };
    std::vector<uint8_t> msg(text.begin(), text.end());
    const uint64_t bit_length = static_cast<uint64_t>(msg.size()) * 8;
    msg.push_back(0x80);
    while (msg.size() % 64 != 56)
    {
        msg.push_back(0);
    }
    for (int i = 7; i >= 0; --i)
    {
        msg.push_back(static_cast<uint8_t>(bit_length >> (8 * i)));
    }
    for (size_t chunk = 0; chunk < msg.size(); chunk += 64)
    {
        std::array<uint32_t, 80> w{};
        for (size_t i = 0; i < 16; ++i)
        {
            const uint8_t* p = &msg[chunk + 4 * i];
            w[i] = (uint32_t{ p[0] } << 24) | (uint32_t{ p[1] } << 16) | (uint32_t{ p[2] } << 8) | uint32_t{ p[3] };
        }
        for (size_t i = 16; i < 80; ++i)
        {
            w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        }
        uint32_t a = h[0];
        uint32_t b = h[1];
        uint32_t c = h[2];
        uint32_t d = h[3];
        uint32_t e = h[4];
        for (size_t i = 0; i < 80; ++i)
        {
            uint32_t f = 0;
            uint32_t k = 0;
            if (i < 20)
            {
                f = (b & c) | (~b & d);
                k = 0x5A827999;
            }
            else if (i < 40)
            {
                f = b ^ c ^ d;
                k = 0x6ED9EBA1;
            }
            else if (i < 60)
            {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8F1BBCDC;
            }
            else
            {
                f = b ^ c ^ d;
                k = 0xCA62C1D6;
            }
            const uint32_t t = rotl(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = rotl(b, 30);
            b = a;
            a = t;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
    }
    std::array<uint8_t, 20> digest{};
    for (size_t i = 0; i < digest.size(); ++i)
    {
        digest[i] = static_cast<uint8_t>(h[i / 4] >> (24 - 8 * (i % 4)));
    }
    return digest;
}

std::string base64(const uint8_t* data, size_t size)
{
    static constexpr char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((size + 2) / 3 * 4);
    for (size_t i = 0; i < size; i += 3)
    {
        uint32_t n = uint32_t{ data[i] } << 16;
        if (i + 1 < size)
        {
            n |= uint32_t{ data[i + 1] } << 8;
        }
        if (i + 2 < size)
        {
            n |= uint32_t{ data[i + 2] };
        }
        out += kAlphabet[(n >> 18) & 63];
        out += kAlphabet[(n >> 12) & 63];
        out += i + 1 < size ? kAlphabet[(n >> 6) & 63] : '=';
        out += i + 2 < size ? kAlphabet[n & 63] : '=';
    }
    return out;
}

std::string websocket_accept_key(const std::string& key)
{
    const std::array<uint8_t, 20> digest = sha1(key + kAcceptGuid);
    return base64(digest.data(), digest.size());
}

} // namespace core::remote_link_detail
