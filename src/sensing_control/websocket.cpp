// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/sensing_control/websocket.hpp"

#include <algorithm>
#include <array>
#include <cctype>

namespace plugins
{
namespace sensing
{
namespace websocket
{

namespace
{

constexpr std::string_view kGuid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

std::uint32_t rotl(std::uint32_t value, int bits)
{
    return (value << bits) | (value >> (32 - bits));
}

std::string lower(std::string_view text)
{
    std::string out(text);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return std::tolower(c); });
    return out;
}

std::string trim(std::string_view text)
{
    std::size_t begin = 0;
    std::size_t end = text.size();
    while (begin < end && (text[begin] == ' ' || text[begin] == '\t'))
        ++begin;
    while (end > begin && (text[end - 1] == ' ' || text[end - 1] == '\t' || text[end - 1] == '\r'))
        --end;
    return std::string(text.substr(begin, end - begin));
}

/// The value of one `name=value` pair of a query string, percent-decoding nothing: tokens are URL-safe.
std::string query_value(std::string_view query, std::string_view name)
{
    std::size_t pos = 0;
    while (pos <= query.size())
    {
        const std::size_t amp = query.find('&', pos);
        const std::string_view pair = query.substr(pos, amp == std::string_view::npos ? amp : amp - pos);
        const std::size_t eq = pair.find('=');
        if (eq != std::string_view::npos && pair.substr(0, eq) == name)
            return std::string(pair.substr(eq + 1));
        if (amp == std::string_view::npos)
            break;
        pos = amp + 1;
    }
    return {};
}

} // namespace

std::string sha1(std::string_view data)
{
    std::array<std::uint32_t, 5> h = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0 };

    std::string message(data);
    const std::uint64_t bit_length = static_cast<std::uint64_t>(data.size()) * 8;
    message.push_back(static_cast<char>(0x80));
    while (message.size() % 64 != 56)
        message.push_back('\0');
    for (int shift = 56; shift >= 0; shift -= 8)
        message.push_back(static_cast<char>((bit_length >> shift) & 0xFF));

    for (std::size_t offset = 0; offset < message.size(); offset += 64)
    {
        std::array<std::uint32_t, 80> w{};
        for (int i = 0; i < 16; ++i)
        {
            const auto* p = reinterpret_cast<const unsigned char*>(message.data() + offset + 4 * i);
            w[i] = (std::uint32_t(p[0]) << 24) | (std::uint32_t(p[1]) << 16) | (std::uint32_t(p[2]) << 8) | p[3];
        }
        for (int i = 16; i < 80; ++i)
            w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

        std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (int i = 0; i < 80; ++i)
        {
            std::uint32_t f = 0;
            std::uint32_t k = 0;
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
            const std::uint32_t temp = rotl(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = rotl(b, 30);
            b = a;
            a = temp;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
    }

    std::string digest;
    for (const std::uint32_t word : h)
        for (int shift = 24; shift >= 0; shift -= 8)
            digest.push_back(static_cast<char>((word >> shift) & 0xFF));
    return digest;
}

std::string base64_encode(std::string_view data)
{
    static constexpr char kAlphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((data.size() + 2) / 3 * 4);
    for (std::size_t i = 0; i < data.size(); i += 3)
    {
        const std::uint32_t b0 = static_cast<unsigned char>(data[i]);
        const std::uint32_t b1 = i + 1 < data.size() ? static_cast<unsigned char>(data[i + 1]) : 0;
        const std::uint32_t b2 = i + 2 < data.size() ? static_cast<unsigned char>(data[i + 2]) : 0;
        const std::uint32_t triple = (b0 << 16) | (b1 << 8) | b2;
        out.push_back(kAlphabet[(triple >> 18) & 0x3F]);
        out.push_back(kAlphabet[(triple >> 12) & 0x3F]);
        out.push_back(i + 1 < data.size() ? kAlphabet[(triple >> 6) & 0x3F] : '=');
        out.push_back(i + 2 < data.size() ? kAlphabet[triple & 0x3F] : '=');
    }
    return out;
}

std::string accept_key(std::string_view client_key)
{
    std::string joined(client_key);
    joined.append(kGuid);
    return base64_encode(sha1(joined));
}

bool constant_time_equal(std::string_view a, std::string_view b)
{
    // Length is not secret here: it is compared first, and only the content is hidden.
    if (a.size() != b.size())
        return false;
    unsigned char difference = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        difference |= static_cast<unsigned char>(a[i]) ^ static_cast<unsigned char>(b[i]);
    return difference == 0;
}

Handshake parse_handshake(std::string_view head)
{
    Handshake result;
    const std::size_t line_end = head.find("\r\n");
    if (line_end == std::string_view::npos)
    {
        result.reason = "malformed request";
        return result;
    }
    const std::string_view request_line = head.substr(0, line_end);
    // "GET <target> HTTP/1.1"
    const std::size_t first_space = request_line.find(' ');
    const std::size_t second_space = request_line.rfind(' ');
    if (first_space == std::string_view::npos || second_space == first_space ||
        request_line.substr(0, first_space) != "GET" || request_line.substr(second_space + 1) != "HTTP/1.1")
    {
        result.reason = "expected a GET request over HTTP/1.1";
        return result;
    }
    const std::string_view target = request_line.substr(first_space + 1, second_space - first_space - 1);
    const std::size_t question = target.find('?');
    result.path = std::string(target.substr(0, question));
    const std::string_view query = question == std::string_view::npos ? std::string_view() : target.substr(question + 1);

    bool upgrade = false;
    bool connection_upgrade = false;
    std::string version;
    std::string bearer;
    std::size_t pos = line_end + 2;
    while (pos < head.size())
    {
        const std::size_t end = head.find("\r\n", pos);
        const std::string_view line = head.substr(pos, end == std::string_view::npos ? end : end - pos);
        if (line.empty())
            break;
        const std::size_t colon = line.find(':');
        if (colon != std::string_view::npos)
        {
            const std::string name = lower(trim(line.substr(0, colon)));
            const std::string value = trim(line.substr(colon + 1));
            if (name == "upgrade")
                upgrade = lower(value) == "websocket";
            else if (name == "connection")
                connection_upgrade = lower(value).find("upgrade") != std::string::npos;
            else if (name == "sec-websocket-key")
                result.key = value;
            else if (name == "sec-websocket-version")
                version = value;
            else if (name == "authorization" && lower(value).rfind("bearer ", 0) == 0)
                bearer = trim(std::string_view(value).substr(7));
        }
        if (end == std::string_view::npos)
            break;
        pos = end + 2;
    }

    if (!upgrade || !connection_upgrade || result.key.empty())
    {
        result.status = 426;
        result.reason = "this is a WebSocket endpoint";
        return result;
    }
    if (version != "13")
    {
        result.status = 426;
        result.reason = "only WebSocket version 13 is supported";
        return result;
    }
    result.token = bearer.empty() ? query_value(query, "token") : bearer;
    result.ok = true;
    return result;
}

std::string handshake_response(const Handshake& handshake)
{
    if (handshake.ok)
    {
        return "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               "Sec-WebSocket-Accept: " +
               accept_key(handshake.key) + "\r\n\r\n";
    }
    const char* phrase = handshake.status == 401 ? "Unauthorized" :
                         handshake.status == 404 ? "Not Found" :
                         handshake.status == 426 ? "Upgrade Required" :
                                                   "Bad Request";
    std::string response = "HTTP/1.1 " + std::to_string(handshake.status) + " " + phrase + "\r\n";
    if (handshake.status == 426)
        response += "Upgrade: websocket\r\n";
    response += "Content-Type: text/plain\r\nContent-Length: " + std::to_string(handshake.reason.size()) +
                "\r\nConnection: close\r\n\r\n" + handshake.reason;
    return response;
}

ParseResult parse_frame(std::string_view buffer, std::size_t max_payload)
{
    ParseResult result;
    if (buffer.size() < 2)
        return result;
    const auto byte = [&](std::size_t i) { return static_cast<unsigned char>(buffer[i]); };

    const unsigned char b0 = byte(0);
    const unsigned char b1 = byte(1);
    if ((b0 & 0x70) != 0)
    {
        result.status = ParseStatus::Error;
        result.error = "reserved bits set";
        return result;
    }
    if ((b1 & 0x80) == 0)
    {
        result.status = ParseStatus::Error;
        result.error = "client frames must be masked";
        return result;
    }
    const std::uint8_t opcode = b0 & 0x0F;
    const bool control = (opcode & 0x08) != 0;

    std::uint64_t length = b1 & 0x7F;
    std::size_t header = 2;
    if (length == 126)
    {
        if (buffer.size() < 4)
            return result;
        length = (std::uint64_t(byte(2)) << 8) | byte(3);
        header = 4;
    }
    else if (length == 127)
    {
        if (buffer.size() < 10)
            return result;
        length = 0;
        for (int i = 0; i < 8; ++i)
            length = (length << 8) | byte(2 + i);
        header = 10;
    }
    if (control && (length > 125 || (b0 & 0x80) == 0))
    {
        result.status = ParseStatus::Error;
        result.error = "invalid control frame";
        return result;
    }
    if (length > max_payload)
    {
        result.status = ParseStatus::Error;
        result.close_code = 1009;
        result.error = "message too big";
        return result;
    }
    const std::size_t total = header + 4 + static_cast<std::size_t>(length);
    if (buffer.size() < total)
        return result;

    const unsigned char mask[4] = { byte(header), byte(header + 1), byte(header + 2), byte(header + 3) };
    result.frame.fin = (b0 & 0x80) != 0;
    result.frame.opcode = opcode;
    result.frame.payload.resize(static_cast<std::size_t>(length));
    for (std::size_t i = 0; i < length; ++i)
        result.frame.payload[i] = static_cast<char>(byte(header + 4 + i) ^ mask[i % 4]);
    result.consumed = total;
    result.status = ParseStatus::Complete;
    return result;
}

std::string encode_frame(std::uint8_t opcode, std::string_view payload, bool fin)
{
    std::string out;
    out.push_back(static_cast<char>((fin ? 0x80 : 0x00) | (opcode & 0x0F)));
    if (payload.size() < 126)
    {
        out.push_back(static_cast<char>(payload.size()));
    }
    else if (payload.size() <= 0xFFFF)
    {
        out.push_back(126);
        out.push_back(static_cast<char>(payload.size() >> 8));
        out.push_back(static_cast<char>(payload.size() & 0xFF));
    }
    else
    {
        out.push_back(127);
        for (int shift = 56; shift >= 0; shift -= 8)
            out.push_back(static_cast<char>((static_cast<std::uint64_t>(payload.size()) >> shift) & 0xFF));
    }
    out.append(payload);
    return out;
}

std::string encode_close(std::uint16_t code, std::string_view reason)
{
    std::string payload;
    payload.push_back(static_cast<char>(code >> 8));
    payload.push_back(static_cast<char>(code & 0xFF));
    payload.append(reason.substr(0, 120));
    return encode_frame(kClose, payload);
}

} // namespace websocket
} // namespace sensing
} // namespace plugins
