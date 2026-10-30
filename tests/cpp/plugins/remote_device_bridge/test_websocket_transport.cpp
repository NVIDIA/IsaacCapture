// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <remote_plugin_session/protocol_frame.hpp>
#include <remote_plugin_session/remote_address.hpp>
#include <remote_plugin_session/tcp_transport.hpp>
#include <remote_plugin_session/websocket_transport.hpp>
#include <sys/socket.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace remote_plugin_session;
namespace protocol = remote_plugin_protocol;
using Role = WebSocketTransport::Role;

namespace
{

constexpr std::chrono::milliseconds kWait{ 2000 };
constexpr std::chrono::milliseconds kShort{ 300 };

std::pair<TcpSocket, TcpSocket> socket_pair()
{
    int fds[2];
    REQUIRE(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0);
    return { TcpSocket(fds[0]), TcpSocket(fds[1]) };
}

//! A wire frame whose WebSocket message (header + payload) is exactly `message_size` bytes.
std::vector<uint8_t> frame_of_size(size_t message_size)
{
    std::vector<uint8_t> payload(message_size - protocol::kHeaderSize);
    for (size_t i = 0; i < payload.size(); ++i)
    {
        payload[i] = static_cast<uint8_t>(i * 31 + 7);
    }
    return protocol::encode_frame(protocol::FrameType::SchemaSample, 3, payload);
}

//! Hand-built client frame (masked, length < 65536) for driving the server role directly.
std::vector<uint8_t> raw_frame(uint8_t opcode, bool fin, const std::vector<uint8_t>& payload, bool masked = true)
{
    std::vector<uint8_t> out{ static_cast<uint8_t>((fin ? 0x80 : 0) | opcode) };
    const uint8_t mask_bit = masked ? 0x80 : 0;
    if (payload.size() < 126)
    {
        out.push_back(static_cast<uint8_t>(mask_bit | payload.size()));
    }
    else
    {
        out.insert(out.end(), { static_cast<uint8_t>(mask_bit | 126), static_cast<uint8_t>(payload.size() >> 8),
                                static_cast<uint8_t>(payload.size()) });
    }
    const uint8_t key[4] = { 0x12, 0x34, 0x56, 0x78 };
    if (masked)
    {
        out.insert(out.end(), key, key + 4);
    }
    for (size_t i = 0; i < payload.size(); ++i)
    {
        out.push_back(masked ? payload[i] ^ key[i % 4] : payload[i]);
    }
    return out;
}

void raw_send(TcpSocket& socket, const std::string& text)
{
    REQUIRE(socket.send_all(reinterpret_cast<const uint8_t*>(text.data()), text.size()));
}

void raw_send(TcpSocket& socket, const std::vector<uint8_t>& bytes)
{
    REQUIRE(socket.send_all(bytes.data(), bytes.size()));
}

std::vector<uint8_t> raw_recv(TcpSocket& socket, size_t size)
{
    std::vector<uint8_t> bytes(size);
    REQUIRE(socket.recv_all(bytes.data(), size, kWait) == TcpSocket::RecvResult::Ok);
    return bytes;
}

//! Reads the server's HTTP response head byte by byte.
std::string raw_recv_head(TcpSocket& socket)
{
    std::string text;
    while (text.size() < 4 || text.compare(text.size() - 4, 4, "\r\n\r\n") != 0)
    {
        text.push_back(static_cast<char>(raw_recv(socket, 1)[0]));
    }
    return text;
}

// RFC 6455 section 1.3 sample handshake.
const std::string kSampleUpgrade =
    "GET /chat HTTP/1.1\r\nHost: server.example.com\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
    "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nOrigin: http://example.com\r\nSec-WebSocket-Version: 13\r\n\r\n";

//! A server transport upgraded by a raw client using the RFC sample key.
std::unique_ptr<WebSocketTransport> upgraded_server(TcpSocket server_end, TcpSocket& raw_client)
{
    auto server = std::make_unique<WebSocketTransport>(std::move(server_end), Role::Server);
    raw_send(raw_client, kSampleUpgrade);
    REQUIRE(server->wait_ready(kWait) == ReceiveResult::Ok);
    const std::string response = raw_recv_head(raw_client);
    REQUIRE(response.rfind("HTTP/1.1 101", 0) == 0);
    REQUIRE(response.find("Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n") != std::string::npos);
    return server;
}

struct Connected
{
    std::unique_ptr<WebSocketTransport> client;
    std::unique_ptr<WebSocketTransport> server;
};

Connected connected_pair()
{
    auto [client_end, server_end] = socket_pair();
    Connected pair;
    pair.server = std::make_unique<WebSocketTransport>(std::move(server_end), Role::Server);
    std::thread client(
        [&] { pair.client = std::make_unique<WebSocketTransport>(std::move(client_end), Role::Client, "h:1", "/x"); });
    REQUIRE(pair.server->wait_ready(kWait) == ReceiveResult::Ok);
    client.join();
    REQUIRE(pair.client);
    return pair;
}

} // namespace

TEST_CASE("parse_remote selects TCP or WebSocket", "[remote_device_bridge][unit]")
{
    const RemoteAddress tcp = parse_remote("10.0.0.2:5000", 47500);
    REQUIRE_FALSE(tcp.ws_path);
    REQUIRE(tcp.endpoint.host == "10.0.0.2");
    REQUIRE(tcp.endpoint.port == 5000);

    const RemoteAddress ws = parse_remote("ws://bridge.local:8080/teleop", 47500);
    REQUIRE(ws.ws_path);
    REQUIRE(ws.endpoint.host == "bridge.local");
    REQUIRE(ws.endpoint.port == 8080);
    REQUIRE(*ws.ws_path == "/teleop");

    const RemoteAddress bare = parse_remote("ws://bridge.local", 47500);
    REQUIRE(bare.endpoint.port == 47500);
    REQUIRE(*bare.ws_path == "/");

    REQUIRE_THROWS_AS(parse_remote("wss://bridge.local", 47500), std::invalid_argument);
    REQUIRE_THROWS_AS(parse_remote("ws:///teleop", 47500), std::invalid_argument);
}

TEST_CASE("parse_endpoint validates ports and accepts IPv6 hosts", "[remote_device_bridge][unit]")
{
    for (const char* bad : { "host:", "host:0", "host:70000", "host:80x", "[::1", "[::1]x" })
    {
        INFO(bad);
        REQUIRE_THROWS_AS(parse_endpoint(bad, 47500), std::invalid_argument);
    }
    const Endpoint bracketed = parse_endpoint("[::1]:8080", 47500);
    REQUIRE(bracketed.host == "::1");
    REQUIRE(bracketed.port == 8080);
    const Endpoint bare_v6 = parse_endpoint("fe80::1", 47500);
    REQUIRE(bare_v6.host == "fe80::1");
    REQUIRE(bare_v6.port == 47500);
    REQUIRE(parse_endpoint("[::1]", 47500).port == 47500);
}

TEST_CASE("WebSocketTransport round-trips frames of every length class", "[remote_device_bridge][unit]")
{
    // 7-bit, 16-bit and 64-bit WebSocket lengths, both directions (client frames are masked).
    const size_t message_size = GENERATE(size_t{ 25 }, 125, 126, 65535, 65536, 1024 * 1024);
    Connected pair = connected_pair();
    const std::vector<uint8_t> sent = frame_of_size(message_size);

    for (auto [from, to] :
         { std::pair{ pair.client.get(), pair.server.get() }, std::pair{ pair.server.get(), pair.client.get() } })
    {
        bool send_ok = false;
        std::thread sender([&, from = from] { send_ok = from->send(sent.data(), sent.size()); });
        std::vector<uint8_t> received;
        REQUIRE(to->receive(received, kWait) == ReceiveResult::Ok);
        sender.join();
        REQUIRE(send_ok);
        REQUIRE(received == sent);
    }
}

TEST_CASE("WebSocketTransport server role handles control frames and fragments", "[remote_device_bridge][unit]")
{
    auto [raw, server_end] = socket_pair();
    auto server = upgraded_server(std::move(server_end), raw);
    std::vector<uint8_t> frame;

    SECTION("ping is answered with a pong carrying the same payload")
    {
        raw_send(raw, raw_frame(0x9, true, { 'h', 'i' }));
        REQUIRE(server->receive(frame, kShort) == ReceiveResult::Timeout);
        REQUIRE(raw_recv(raw, 4) == std::vector<uint8_t>{ 0x8A, 0x02, 'h', 'i' });
    }
    SECTION("a fragmented message with an interleaved ping is reassembled")
    {
        const std::vector<uint8_t> sent = frame_of_size(300);
        raw_send(raw, raw_frame(0x2, false, { sent.begin(), sent.begin() + 100 }));
        raw_send(raw, raw_frame(0x9, true, {}));
        raw_send(raw, raw_frame(0x0, true, { sent.begin() + 100, sent.end() }));
        REQUIRE(server->receive(frame, kWait) == ReceiveResult::Ok);
        REQUIRE(frame == sent);
        REQUIRE(raw_recv(raw, 2) == std::vector<uint8_t>{ 0x8A, 0x00 });
    }
    SECTION("peer close is echoed and reported as closed")
    {
        raw_send(raw, raw_frame(0x8, true, { 0x03, 0xE8 }));
        REQUIRE(server->receive(frame, kWait) == ReceiveResult::Closed);
        REQUIRE(raw_recv(raw, 4) == std::vector<uint8_t>{ 0x88, 0x02, 0x03, 0xE8 });
    }
    SECTION("an unmasked client frame is malformed")
    {
        raw_send(raw, raw_frame(0x2, true, frame_of_size(protocol::kHeaderSize), false));
        REQUIRE(server->receive(frame, kWait) == ReceiveResult::Malformed);
    }
    SECTION("text frames are malformed")
    {
        raw_send(raw, raw_frame(0x1, true, { 'x' }));
        REQUIRE(server->receive(frame, kWait) == ReceiveResult::Malformed);
    }
    SECTION("a message that is not exactly one wire frame is malformed")
    {
        raw_send(raw, raw_frame(0x2, true, { 0x01 }));
        REQUIRE(server->receive(frame, kWait) == ReceiveResult::Malformed);
    }
    SECTION("TCP EOF is reported as closed")
    {
        raw.shutdown();
        REQUIRE(server->receive(frame, kWait) == ReceiveResult::Closed);
    }
}

TEST_CASE("WebSocketTransport rejects bad handshakes", "[remote_device_bridge][unit]")
{
    auto [peer, own_end] = socket_pair();
    SECTION("server answers garbage with 400 and refuses to send")
    {
        WebSocketTransport server(std::move(own_end), Role::Server);
        raw_send(peer, "hello there\r\n\r\n");
        std::vector<uint8_t> frame;
        REQUIRE(server.receive(frame, kWait) == ReceiveResult::Malformed);
        REQUIRE(raw_recv_head(peer).rfind("HTTP/1.1 400", 0) == 0);
        const std::vector<uint8_t> sent = frame_of_size(protocol::kHeaderSize);
        REQUIRE_FALSE(server.send(sent.data(), sent.size()));
    }
    SECTION("client throws on a non-101 response")
    {
        raw_send(peer, "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n");
        REQUIRE_THROWS_AS(WebSocketTransport(std::move(own_end), Role::Client, "h:1", "/"), std::runtime_error);
    }
    SECTION("client throws on a wrong Sec-WebSocket-Accept")
    {
        raw_send(peer,
                 "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                 "Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n");
        REQUIRE_THROWS_AS(WebSocketTransport(std::move(own_end), Role::Client, "h:1", "/"), std::runtime_error);
    }
}

TEST_CASE("connect_transport speaks ws:// to a listening WebSocketTransport", "[remote_device_bridge][unit]")
{
    TcpSocket listener = TcpSocket::listen(Endpoint{ "127.0.0.1", 0 }, 1);
    const RemoteAddress remote = parse_remote("ws://127.0.0.1:" + std::to_string(listener.local_port()) + "/t", 1);
    std::unique_ptr<ITransport> client;
    std::thread connector([&] { client = connect_transport(remote, kWait); });
    std::unique_ptr<ITransport> server;
    while (!server)
    {
        server = WebSocketTransport::accept(listener, kShort);
    }
    REQUIRE(server->wait_ready(kWait) == ReceiveResult::Ok);
    connector.join();
    REQUIRE(client);
    std::vector<uint8_t> frame;

    const std::vector<uint8_t> sent = frame_of_size(200);
    REQUIRE(client->send(sent.data(), sent.size()));
    REQUIRE(server->receive(frame, kWait) == ReceiveResult::Ok);
    REQUIRE(frame == sent);
    client->shutdown();
    REQUIRE(server->receive(frame, kWait) == ReceiveResult::Closed);
}

TEST_CASE("TcpTransport delimits frames by the header and reports broken peers", "[remote_device_bridge][unit]")
{
    auto [peer, transport_end] = socket_pair();
    TcpTransport transport(std::move(transport_end));
    std::vector<uint8_t> frame;

    SECTION("a header-only frame arrives on its own")
    {
        const std::vector<uint8_t> sent = protocol::encode_frame(protocol::FrameType::Heartbeat, 0, {});
        raw_send(peer, sent);
        REQUIRE(transport.receive(frame, kWait) == ReceiveResult::Ok);
        REQUIRE(frame == sent);
    }
    SECTION("a frame with a payload arrives whole")
    {
        const std::vector<uint8_t> sent = frame_of_size(protocol::kHeaderSize + 100);
        raw_send(peer, sent);
        REQUIRE(transport.receive(frame, kWait) == ReceiveResult::Ok);
        REQUIRE(frame == sent);
    }
    SECTION("nothing within the timeout")
    {
        REQUIRE(transport.receive(frame, kShort) == ReceiveResult::Timeout);
    }
    SECTION("a bad magic is malformed")
    {
        std::vector<uint8_t> sent = protocol::encode_frame(protocol::FrameType::Heartbeat, 0, {});
        sent[0] ^= 0xFF;
        raw_send(peer, sent);
        REQUIRE(transport.receive(frame, kWait) == ReceiveResult::Malformed);
    }
    SECTION("a header whose payload never arrives counts as closed")
    {
        const std::vector<uint8_t> sent = frame_of_size(protocol::kHeaderSize + 100);
        raw_send(peer, std::vector<uint8_t>(sent.begin(), sent.begin() + protocol::kHeaderSize));
        REQUIRE(transport.receive(frame, kWait) == ReceiveResult::Closed);
    }
    SECTION("peer EOF is closed")
    {
        peer.shutdown();
        REQUIRE(transport.receive(frame, kWait) == ReceiveResult::Closed);
    }
}
