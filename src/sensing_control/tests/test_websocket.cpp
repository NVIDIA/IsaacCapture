// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <sensing_control/websocket.hpp>

#include <cstdint>
#include <string>

using namespace plugins::sensing::websocket;

namespace
{

std::string hex(const std::string& bytes)
{
    static constexpr char kDigits[] = "0123456789abcdef";
    std::string out;
    for (const unsigned char c : bytes)
    {
        out.push_back(kDigits[c >> 4]);
        out.push_back(kDigits[c & 0x0F]);
    }
    return out;
}

/// A client frame: masked, as RFC 6455 requires.
std::string client_frame(std::uint8_t opcode, const std::string& payload, bool fin = true, bool masked = true)
{
    const unsigned char mask[4] = { 0x37, 0xfa, 0x21, 0x3d };
    std::string out;
    out.push_back(static_cast<char>((fin ? 0x80 : 0) | opcode));
    const std::uint8_t mask_bit = masked ? 0x80 : 0;
    if (payload.size() < 126)
    {
        out.push_back(static_cast<char>(mask_bit | payload.size()));
    }
    else if (payload.size() <= 0xFFFF)
    {
        out.push_back(static_cast<char>(mask_bit | 126));
        out.push_back(static_cast<char>(payload.size() >> 8));
        out.push_back(static_cast<char>(payload.size() & 0xFF));
    }
    else
    {
        out.push_back(static_cast<char>(mask_bit | 127));
        for (int shift = 56; shift >= 0; shift -= 8)
            out.push_back(static_cast<char>((static_cast<std::uint64_t>(payload.size()) >> shift) & 0xFF));
    }
    if (masked)
        out.append(reinterpret_cast<const char*>(mask), 4);
    for (std::size_t i = 0; i < payload.size(); ++i)
        out.push_back(static_cast<char>(masked ? (static_cast<unsigned char>(payload[i]) ^ mask[i % 4]) : payload[i]));
    return out;
}

std::string upgrade_request(const std::string& target = "/",
                            const std::string& extra_headers = "",
                            const std::string& version = "13")
{
    return "GET " + target +
           " HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: keep-alive, Upgrade\r\n"
           "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: " +
           version + "\r\n" + extra_headers + "\r\n";
}

} // namespace

TEST_CASE("sha1 matches the standard test vectors", "[websocket]")
{
    CHECK(hex(sha1("abc")) == "a9993e364706816aba3e25717850c26c9cd0d89d");
    CHECK(hex(sha1("")) == "da39a3ee5e6b4b0d3255bfef95601890afd80709");
    // Two blocks after padding.
    CHECK(hex(sha1("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")) ==
          "84983e441c3bd26ebaae4aa1f95129e5e54670f1");
}

TEST_CASE("base64 and the RFC 6455 accept key", "[websocket]")
{
    CHECK(base64_encode("") == "");
    CHECK(base64_encode("f") == "Zg==");
    CHECK(base64_encode("fo") == "Zm8=");
    CHECK(base64_encode("foo") == "Zm9v");
    CHECK(accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
}

TEST_CASE("constant_time_equal", "[websocket]")
{
    CHECK(constant_time_equal("secret", "secret"));
    CHECK(constant_time_equal("", ""));
    CHECK_FALSE(constant_time_equal("secret", "secreT"));
    CHECK_FALSE(constant_time_equal("secret", "secre"));
    CHECK_FALSE(constant_time_equal("", "x"));
}

TEST_CASE("parse_frame decodes the three length forms", "[websocket]")
{
    SECTION("short")
    {
        const auto r = parse_frame(client_frame(kText, "hello"), 1 << 20);
        REQUIRE(r.status == ParseStatus::Complete);
        CHECK(r.frame.fin);
        CHECK(r.frame.opcode == kText);
        CHECK(r.frame.payload == "hello");
        CHECK(r.consumed == 2 + 4 + 5);
    }
    SECTION("125 is the last short form, 126 uses 16 bits")
    {
        const std::string p125(125, 'a');
        const std::string p126(126, 'b');
        const auto a = parse_frame(client_frame(kText, p125), 1 << 20);
        REQUIRE(a.status == ParseStatus::Complete);
        CHECK(a.consumed == 2 + 4 + 125);
        const auto b = parse_frame(client_frame(kText, p126), 1 << 20);
        REQUIRE(b.status == ParseStatus::Complete);
        CHECK(b.frame.payload == p126);
        CHECK(b.consumed == 4 + 4 + 126);
    }
    SECTION("65535 is the 16-bit maximum, 65536 uses 64 bits")
    {
        const std::string p16(65535, 'c');
        const std::string p64(65536, 'd');
        const auto a = parse_frame(client_frame(kBinary, p16), 1 << 20);
        REQUIRE(a.status == ParseStatus::Complete);
        CHECK(a.consumed == 4 + 4 + 65535);
        const auto b = parse_frame(client_frame(kBinary, p64), 1 << 20);
        REQUIRE(b.status == ParseStatus::Complete);
        CHECK(b.frame.payload == p64);
        CHECK(b.consumed == 10 + 4 + 65536);
    }
}

TEST_CASE("parse_frame waits for more bytes and leaves trailing bytes", "[websocket]")
{
    const std::string frame = client_frame(kText, std::string(300, 'x'));
    for (const std::size_t cut : { std::size_t(0), std::size_t(1), std::size_t(3), std::size_t(5), frame.size() - 1 })
        CHECK(parse_frame(std::string_view(frame).substr(0, cut), 1 << 20).status == ParseStatus::NeedMore);

    const std::string two = client_frame(kText, "one") + client_frame(kText, "two");
    const auto first = parse_frame(two, 1 << 20);
    REQUIRE(first.status == ParseStatus::Complete);
    CHECK(first.frame.payload == "one");
    const auto second = parse_frame(std::string_view(two).substr(first.consumed), 1 << 20);
    REQUIRE(second.status == ParseStatus::Complete);
    CHECK(second.frame.payload == "two");
}

TEST_CASE("parse_frame reports fragments through fin and continuation", "[websocket]")
{
    const auto first = parse_frame(client_frame(kText, "hel", /*fin=*/false), 1 << 20);
    REQUIRE(first.status == ParseStatus::Complete);
    CHECK_FALSE(first.frame.fin);
    CHECK(first.frame.opcode == kText);
    const auto last = parse_frame(client_frame(kContinuation, "lo", /*fin=*/true), 1 << 20);
    REQUIRE(last.status == ParseStatus::Complete);
    CHECK(last.frame.fin);
    CHECK(last.frame.opcode == kContinuation);
}

TEST_CASE("parse_frame errors", "[websocket]")
{
    SECTION("an unmasked client frame is a protocol error")
    {
        const auto r = parse_frame(client_frame(kText, "hi", true, /*masked=*/false), 1 << 20);
        CHECK(r.status == ParseStatus::Error);
        CHECK(r.close_code == 1002);
    }
    SECTION("reserved bits")
    {
        std::string frame = client_frame(kText, "hi");
        frame[0] = static_cast<char>(frame[0] | 0x40);
        const auto r = parse_frame(frame, 1 << 20);
        CHECK(r.status == ParseStatus::Error);
        CHECK(r.close_code == 1002);
    }
    SECTION("an oversized payload is 1009, decided from the header alone")
    {
        const std::string frame = client_frame(kText, std::string(200, 'a'));
        const auto r = parse_frame(std::string_view(frame).substr(0, 4), 100);
        CHECK(r.status == ParseStatus::Error);
        CHECK(r.close_code == 1009);
        const auto exact = parse_frame(client_frame(kText, std::string(100, 'a')), 100);
        CHECK(exact.status == ParseStatus::Complete);
    }
    SECTION("a 64-bit length beyond the limit is 1009 without waiting for the payload")
    {
        std::string header;
        header.push_back(static_cast<char>(0x81));
        header.push_back(static_cast<char>(0x80 | 127));
        header.append(std::string("\x00\x00\x00\x01\x00\x00\x00\x00", 8));
        const auto r = parse_frame(header, 1 << 20);
        CHECK(r.status == ParseStatus::Error);
        CHECK(r.close_code == 1009);
    }
    SECTION("control frames may not be fragmented or longer than 125 bytes")
    {
        CHECK(parse_frame(client_frame(kPing, "x", /*fin=*/false), 1 << 20).status == ParseStatus::Error);
        CHECK(parse_frame(client_frame(kClose, std::string(126, 'x')), 1 << 20).status == ParseStatus::Error);
        CHECK(parse_frame(client_frame(kPing, std::string(125, 'x')), 1 << 20).status == ParseStatus::Complete);
    }
}

TEST_CASE("encode_frame produces unmasked server frames", "[websocket]")
{
    CHECK(encode_frame(kText, "Hello") == std::string("\x81\x05Hello"));
    CHECK(encode_frame(kText, "a", /*fin=*/false) == std::string("\x01\x01"
                                                                 "a"));

    const std::string mid = encode_frame(kText, std::string(126, 'a'));
    CHECK(mid.size() == 4 + 126);
    CHECK(static_cast<unsigned char>(mid[1]) == 126);
    CHECK(static_cast<unsigned char>(mid[2]) == 0);
    CHECK(static_cast<unsigned char>(mid[3]) == 126);

    const std::string big = encode_frame(kBinary, std::string(70000, 'a'));
    CHECK(big.size() == 10 + 70000);
    CHECK(static_cast<unsigned char>(big[1]) == 127);
    CHECK(static_cast<unsigned char>(big[7]) == 1);
    CHECK(static_cast<unsigned char>(big[8]) == 0x11);
    CHECK(static_cast<unsigned char>(big[9]) == 0x70);

    const std::string close = encode_close(1009, "big");
    CHECK(close == std::string("\x88\x05\x03\xf1"
                               "big",
                               7));
    CHECK(encode_close(1000, std::string(500, 'r')).size() == 2 + 2 + 120);
}

TEST_CASE("parse_handshake accepts a valid upgrade", "[websocket]")
{
    const auto h = parse_handshake(upgrade_request("/chan", "Authorization: Bearer s3cret\r\n"));
    REQUIRE(h.ok);
    CHECK(h.key == "dGhlIHNhbXBsZSBub25jZQ==");
    CHECK(h.token == "s3cret");
    CHECK(h.path == "/chan");

    const std::string response = handshake_response(h);
    CHECK(response.rfind("HTTP/1.1 101 Switching Protocols\r\n", 0) == 0);
    CHECK(response.find("Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n") != std::string::npos);
}

TEST_CASE("parse_handshake finds the token in the header or the query", "[websocket]")
{
    SECTION("header names and the Bearer scheme are case-insensitive")
    {
        const auto h = parse_handshake(upgrade_request("/", "AUTHORIZATION: bearer abc123\r\n"));
        REQUIRE(h.ok);
        CHECK(h.token == "abc123");
    }
    SECTION("query parameter")
    {
        const auto h = parse_handshake(upgrade_request("/?x=1&token=qtok&y=2"));
        REQUIRE(h.ok);
        CHECK(h.token == "qtok");
        CHECK(h.path == "/");
    }
    SECTION("the header wins over the query")
    {
        const auto h = parse_handshake(upgrade_request("/?token=q", "Authorization: Bearer h\r\n"));
        REQUIRE(h.ok);
        CHECK(h.token == "h");
    }
    SECTION("no token leaves it empty and the handshake still parses")
    {
        const auto h = parse_handshake(upgrade_request());
        REQUIRE(h.ok);
        CHECK(h.token.empty());
    }
}

TEST_CASE("parse_handshake refuses what is not a WebSocket upgrade", "[websocket]")
{
    SECTION("a plain GET is 426 with an Upgrade header in the answer")
    {
        const auto h = parse_handshake("GET / HTTP/1.1\r\nHost: x\r\n\r\n");
        CHECK_FALSE(h.ok);
        CHECK(h.status == 426);
        const std::string response = handshake_response(h);
        CHECK(response.rfind("HTTP/1.1 426 Upgrade Required\r\n", 0) == 0);
        CHECK(response.find("Upgrade: websocket\r\n") != std::string::npos);
        CHECK(response.find("Content-Length: " + std::to_string(h.reason.size())) != std::string::npos);
    }
    SECTION("a missing key is 426")
    {
        const auto h = parse_handshake(
            "GET / HTTP/1.1\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n");
        CHECK(h.status == 426);
        CHECK_FALSE(h.ok);
    }
    SECTION("a wrong version is 426 and says so")
    {
        const auto h = parse_handshake(upgrade_request("/", "", "8"));
        CHECK_FALSE(h.ok);
        CHECK(h.status == 426);
        CHECK(h.reason.find("13") != std::string::npos);
    }
    SECTION("bad request lines are 400")
    {
        CHECK(parse_handshake("POST / HTTP/1.1\r\n\r\n").status == 400);
        CHECK(parse_handshake("GET / HTTP/1.0\r\n\r\n").status == 400);
        CHECK(parse_handshake("GET\r\n\r\n").status == 400);
        CHECK(parse_handshake("garbage").status == 400);
        CHECK_FALSE(parse_handshake("GET / HTTP/1.0\r\n\r\n").ok);
    }
    SECTION("status phrases")
    {
        Handshake h;
        h.status = 401;
        h.reason = "no";
        CHECK(handshake_response(h).rfind("HTTP/1.1 401 Unauthorized\r\n", 0) == 0);
        h.status = 404;
        CHECK(handshake_response(h).rfind("HTTP/1.1 404 Not Found\r\n", 0) == 0);
        h.status = 400;
        CHECK(handshake_response(h).rfind("HTTP/1.1 400 Bad Request\r\n", 0) == 0);
    }
}
