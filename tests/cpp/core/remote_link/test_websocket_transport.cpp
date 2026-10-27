// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "test_support.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <remote_link/message_header.hpp>
#include <remote_link/websocket_transport.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace core;
using namespace remote_link_test;
using Role = WebSocketTransport::Role;

namespace
{

//! Hand-built WebSocket frame for driving a transport from a raw socket. Lengths of any size class.
std::vector<uint8_t> raw_frame(uint8_t opcode, bool fin, const std::vector<uint8_t>& payload, bool masked = true)
{
    std::vector<uint8_t> out{ static_cast<uint8_t>((fin ? 0x80 : 0) | opcode) };
    const uint8_t mask_bit = masked ? 0x80 : 0;
    const uint64_t size = payload.size();
    if (size < 126)
    {
        out.push_back(static_cast<uint8_t>(mask_bit | size));
    }
    else if (size <= 0xFFFF)
    {
        out.insert(out.end(), { static_cast<uint8_t>(mask_bit | 126), static_cast<uint8_t>(size >> 8),
                                static_cast<uint8_t>(size) });
    }
    else
    {
        out.push_back(static_cast<uint8_t>(mask_bit | 127));
        for (int i = 7; i >= 0; --i)
        {
            out.push_back(static_cast<uint8_t>(size >> (8 * i)));
        }
    }
    const uint8_t key[4] = { 0x12, 0x34, 0x56, 0x78 };
    if (masked)
    {
        out.insert(out.end(), key, key + 4);
    }
    for (size_t i = 0; i < payload.size(); ++i)
    {
        out.push_back(masked ? static_cast<uint8_t>(payload[i] ^ key[i % 4]) : payload[i]);
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

//! A client transport upgraded by a real server role; `raw_server` gets a duplicate of the server's descriptor for
//! writing hand-built WebSocket frames to the client.
std::unique_ptr<WebSocketTransport> upgraded_client(TcpSocket& raw_server)
{
    auto [client_end, server_end] = socket_pair();
    raw_server = TcpSocket(::dup(server_end.fd()));
    WebSocketTransport server(std::move(server_end), Role::Server);
    std::unique_ptr<WebSocketTransport> client;
    std::thread connector([&, end = std::move(client_end)]() mutable
                          { client = std::make_unique<WebSocketTransport>(std::move(end), Role::Client, "h:1", "/x"); });
    REQUIRE(server.wait_ready(kWait) == ReceiveResult::Ok);
    connector.join();
    REQUIRE(client);
    return client;
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
    std::thread connector(
        [&] { pair.client = std::make_unique<WebSocketTransport>(std::move(client_end), Role::Client, "h:1", "/x"); });
    REQUIRE(pair.server->wait_ready(kWait) == ReceiveResult::Ok);
    connector.join();
    REQUIRE(pair.client);
    return pair;
}

} // namespace

TEST_CASE("RemoteLink: WebSocket server answers the RFC 6455 sample handshake", "[remote_link][unit]")
{
    auto [raw, server_end] = socket_pair();
    auto server = upgraded_server(std::move(server_end), raw);
    REQUIRE(server->wait_ready(std::chrono::milliseconds{ 0 }) == ReceiveResult::Ok);
}

TEST_CASE("RemoteLink: WebSocket server upgrades inside the first receive", "[remote_link][unit]")
{
    auto [raw, server_end] = socket_pair();
    WebSocketTransport server(std::move(server_end), Role::Server);
    const std::vector<uint8_t> early = message_of_size(kHeaderSize + 4);
    REQUIRE_FALSE(server.send(early.data(), early.size()));
    REQUIRE_FALSE(server.wait_writable(std::chrono::milliseconds{ 0 }));
    raw_send(raw, kSampleUpgrade);
    raw_send(raw, raw_frame(0x2, true, early));
    std::vector<uint8_t> message;
    REQUIRE(server.receive(message, kWait) == ReceiveResult::Ok);
    REQUIRE(message == early);
    REQUIRE(raw_recv_head(raw).rfind("HTTP/1.1 101", 0) == 0);
    REQUIRE(server.wait_writable(std::chrono::milliseconds{ 0 }));
}

TEST_CASE("RemoteLink: WebSocket round-trips every length class both ways", "[remote_link][unit]")
{
    // WebSocket payload lengths 24 (smallest link message), 125 / 126 (7-bit / 16-bit boundary), 65535 / 65536
    // (16-bit / 64-bit boundary) and 1 MiB.
    const size_t message_size =
        GENERATE(kHeaderSize, size_t{ 125 }, size_t{ 126 }, size_t{ 65535 }, size_t{ 65536 }, size_t{ 1 } << 20);
    Connected pair = connected_pair();
    const std::vector<uint8_t> sent = message_of_size(message_size);
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

TEST_CASE("RemoteLink: WebSocket accepts a zero-length data frame only as part of a message", "[remote_link][unit]")
{
    auto [raw, server_end] = socket_pair();
    auto server = upgraded_server(std::move(server_end), raw);
    const std::vector<uint8_t> sent = message_of_size(kHeaderSize + 8);
    raw_send(raw, raw_frame(0x2, false, {}));
    raw_send(raw, raw_frame(0x0, true, sent));
    std::vector<uint8_t> message;
    REQUIRE(server->receive(message, kWait) == ReceiveResult::Ok);
    REQUIRE(message == sent);

    raw_send(raw, raw_frame(0x2, true, {}));
    REQUIRE(server->receive(message, kWait) == ReceiveResult::Malformed);
}

TEST_CASE("RemoteLink: WebSocket masking rules", "[remote_link][unit]")
{
    const std::vector<uint8_t> sent = message_of_size(kHeaderSize + 6);

    SECTION("client frames are masked")
    {
        auto [client_end, raw] = socket_pair();
        TcpSocket raw_dup(::dup(raw.fd()));
        std::unique_ptr<WebSocketTransport> client;
        std::thread connector(
            [&, end = std::move(client_end)]() mutable
            { client = std::make_unique<WebSocketTransport>(std::move(end), Role::Client, "h:1", "/x"); });
        WebSocketTransport server(std::move(raw), Role::Server);
        REQUIRE(server.wait_ready(kWait) == ReceiveResult::Ok);
        connector.join();
        REQUIRE(client->send(sent.data(), sent.size()));
        const std::vector<uint8_t> head = raw_recv(raw_dup, 2);
        REQUIRE(head[0] == 0x82);
        REQUIRE(head[1] == (0x80 | sent.size()));
        const std::vector<uint8_t> mask = raw_recv(raw_dup, 4);
        std::vector<uint8_t> body = raw_recv(raw_dup, sent.size());
        for (size_t i = 0; i < body.size(); ++i)
        {
            body[i] ^= mask[i % 4];
        }
        REQUIRE(body == sent);
    }
    SECTION("server frames are not masked")
    {
        auto [raw, server_end] = socket_pair();
        auto server = upgraded_server(std::move(server_end), raw);
        REQUIRE(server->send(sent.data(), sent.size()));
        REQUIRE(raw_recv(raw, 2) == std::vector<uint8_t>{ 0x82, static_cast<uint8_t>(sent.size()) });
        REQUIRE(raw_recv(raw, sent.size()) == sent);
    }
    SECTION("the server rejects an unmasked frame")
    {
        auto [raw, server_end] = socket_pair();
        auto server = upgraded_server(std::move(server_end), raw);
        raw_send(raw, raw_frame(0x2, true, sent, false));
        std::vector<uint8_t> message;
        REQUIRE(server->receive(message, kWait) == ReceiveResult::Malformed);
    }
    SECTION("the client rejects a masked frame")
    {
        TcpSocket raw;
        auto client = upgraded_client(raw);
        raw_send(raw, raw_frame(0x2, true, sent, true));
        std::vector<uint8_t> message;
        REQUIRE(client->receive(message, kWait) == ReceiveResult::Malformed);
    }
    SECTION("the client accepts an unmasked frame")
    {
        TcpSocket raw;
        auto client = upgraded_client(raw);
        raw_send(raw, raw_frame(0x2, true, sent, false));
        std::vector<uint8_t> message;
        REQUIRE(client->receive(message, kWait) == ReceiveResult::Ok);
        REQUIRE(message == sent);
    }
}

TEST_CASE("RemoteLink: WebSocket control frames", "[remote_link][unit]")
{
    auto [raw, server_end] = socket_pair();
    auto server = upgraded_server(std::move(server_end), raw);
    std::vector<uint8_t> message;

    SECTION("PING is answered with a PONG carrying the same payload")
    {
        raw_send(raw, raw_frame(0x9, true, { 'h', 'i' }));
        REQUIRE(server->receive(message, std::chrono::milliseconds{ 100 }) == ReceiveResult::Timeout);
        REQUIRE(raw_recv(raw, 4) == std::vector<uint8_t>{ 0x8A, 0x02, 'h', 'i' });
    }
    SECTION("an unsolicited PONG is ignored")
    {
        const std::vector<uint8_t> sent = message_of_size(kHeaderSize);
        raw_send(raw, raw_frame(0xA, true, { 1 }));
        raw_send(raw, raw_frame(0x2, true, sent));
        REQUIRE(server->receive(message, kWait) == ReceiveResult::Ok);
        REQUIRE(message == sent);
    }
    SECTION("a fragmented message with an interleaved PING is reassembled")
    {
        const std::vector<uint8_t> sent = message_of_size(300);
        raw_send(raw, raw_frame(0x2, false, { sent.begin(), sent.begin() + 100 }));
        raw_send(raw, raw_frame(0x9, true, {}));
        raw_send(raw, raw_frame(0x0, false, { sent.begin() + 100, sent.begin() + 200 }));
        raw_send(raw, raw_frame(0x0, true, { sent.begin() + 200, sent.end() }));
        REQUIRE(server->receive(message, kWait) == ReceiveResult::Ok);
        REQUIRE(message == sent);
        REQUIRE(raw_recv(raw, 2) == std::vector<uint8_t>{ 0x8A, 0x00 });
    }
    SECTION("CLOSE is echoed and reported as Closed")
    {
        raw_send(raw, raw_frame(0x8, true, { 0x03, 0xE8, 'b', 'y', 'e' }));
        REQUIRE(server->receive(message, kWait) == ReceiveResult::Closed);
        REQUIRE(raw_recv(raw, 4) == std::vector<uint8_t>{ 0x88, 0x02, 0x03, 0xE8 });
    }
    SECTION("TCP EOF is reported as Closed")
    {
        raw.shutdown();
        REQUIRE(server->receive(message, kWait) == ReceiveResult::Closed);
    }
}

TEST_CASE("RemoteLink: WebSocket reports a message cut off as Truncated", "[remote_link][unit]")
{
    auto [raw, server_end] = socket_pair();
    auto server = upgraded_server(std::move(server_end), raw);
    const std::vector<uint8_t> sent = message_of_size(kHeaderSize + 100);
    const std::vector<uint8_t> frame = raw_frame(0x2, true, sent);
    const std::vector<uint8_t> first_fragment = raw_frame(0x2, false, { sent.begin(), sent.begin() + 50 });

    SECTION("TCP EOF after the first byte of a frame")
    {
        raw_send(raw, std::vector<uint8_t>(frame.begin(), frame.begin() + 1));
    }
    SECTION("TCP EOF in the middle of a payload")
    {
        raw_send(raw, std::vector<uint8_t>(frame.begin(), frame.begin() + 20));
    }
    SECTION("TCP EOF between the fragments of a message")
    {
        raw_send(raw, first_fragment);
    }
    SECTION("CLOSE between the fragments of a message")
    {
        raw_send(raw, first_fragment);
        raw_send(raw, raw_frame(0x8, true, { 0x03, 0xE8 }));
    }
    raw.shutdown();
    std::vector<uint8_t> message;
    REQUIRE(server->receive(message, kWait) == ReceiveResult::Truncated);
}

TEST_CASE("RemoteLink: WebSocket malformed input", "[remote_link][unit]")
{
    auto [raw, server_end] = socket_pair();
    auto server = upgraded_server(std::move(server_end), raw);
    std::vector<uint8_t> message;
    std::vector<uint8_t> bytes;

    SECTION("text frame")
    {
        bytes = raw_frame(0x1, true, { 'x' });
    }
    SECTION("unknown data opcode")
    {
        bytes = raw_frame(0x3, true, message_of_size(kHeaderSize));
    }
    SECTION("reserved control opcode")
    {
        bytes = raw_frame(0xB, true, {});
    }
    SECTION("RSV bit set")
    {
        bytes = raw_frame(0x2, true, message_of_size(kHeaderSize));
        bytes[0] |= 0x40;
    }
    SECTION("fragmented control frame")
    {
        bytes = raw_frame(0x9, false, {});
    }
    SECTION("continuation without a start")
    {
        bytes = raw_frame(0x0, true, message_of_size(kHeaderSize));
    }
    SECTION("new data frame inside a fragmented message")
    {
        const std::vector<uint8_t> sent = message_of_size(kHeaderSize + 10);
        bytes = raw_frame(0x2, false, { sent.begin(), sent.begin() + 5 });
        const std::vector<uint8_t> second = raw_frame(0x2, true, { sent.begin() + 5, sent.end() });
        bytes.insert(bytes.end(), second.begin(), second.end());
    }
    SECTION("message shorter than a header")
    {
        bytes = raw_frame(0x2, true, { 0x01 });
    }
    SECTION("message length disagrees with its header")
    {
        std::vector<uint8_t> sent = message_of_size(kHeaderSize + 10);
        sent.pop_back();
        bytes = raw_frame(0x2, true, sent);
    }
    SECTION("bad wire magic")
    {
        std::vector<uint8_t> sent = message_of_size(kHeaderSize);
        sent[0] = 0;
        bytes = raw_frame(0x2, true, sent);
    }
    SECTION("declared length above the limit")
    {
        bytes = { 0x82, 0x80 | 127 };
        const uint64_t size = kHeaderSize + kMaxPayloadSize + 1;
        for (int i = 7; i >= 0; --i)
        {
            bytes.push_back(static_cast<uint8_t>(size >> (8 * i)));
        }
        bytes.insert(bytes.end(), { 0, 0, 0, 0 });
    }
    raw_send(raw, bytes);
    REQUIRE(server->receive(message, kWait) == ReceiveResult::Malformed);
}

TEST_CASE("RemoteLink: WebSocket rejects bad handshakes", "[remote_link][unit]")
{
    auto [peer, own_end] = socket_pair();
    SECTION("the server answers a non-upgrade request with 400")
    {
        WebSocketTransport server(std::move(own_end), Role::Server);
        raw_send(peer, "GET / HTTP/1.1\r\nHost: x\r\n\r\n");
        REQUIRE(server.wait_ready(kWait) == ReceiveResult::Malformed);
        REQUIRE(raw_recv_head(peer).rfind("HTTP/1.1 400", 0) == 0);
        const std::vector<uint8_t> sent = message_of_size(kHeaderSize);
        REQUIRE_FALSE(server.send(sent.data(), sent.size()));
    }
    SECTION("the server reports a silent client as Timeout")
    {
        WebSocketTransport server(std::move(own_end), Role::Server);
        REQUIRE(server.wait_ready(std::chrono::milliseconds{ 20 }) == ReceiveResult::Timeout);
    }
    SECTION("the client throws on a non-101 response")
    {
        raw_send(peer, "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n");
        REQUIRE_THROWS_AS(WebSocketTransport(std::move(own_end), Role::Client, "h:1", "/"), std::runtime_error);
    }
    SECTION("the client throws on a wrong Sec-WebSocket-Accept")
    {
        raw_send(peer,
                 "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                 "Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n");
        REQUIRE_THROWS_AS(WebSocketTransport(std::move(own_end), Role::Client, "h:1", "/"), std::runtime_error);
    }
    SECTION("the client throws when the server never answers")
    {
        REQUIRE_THROWS_AS(
            WebSocketTransport(std::move(own_end), Role::Client, "h:1", "/", std::chrono::milliseconds{ 20 }),
            std::runtime_error);
    }
    SECTION("the client throws at once on an answer that is not HTTP")
    {
        raw_send(peer, message_of_size(kHeaderSize, 11));
        REQUIRE_THROWS_AS(WebSocketTransport(std::move(own_end), Role::Client, "h:1", "/"), std::runtime_error);
    }
}

TEST_CASE("RemoteLink: a WebSocket server refuses a peer that sends no HTTP request at once", "[remote_link][unit]")
{
    auto [peer, own_end] = socket_pair();
    WebSocketTransport server(std::move(own_end), Role::Server);
    // A tcp:// robot: HELLO, then a HEARTBEAT every 100 ms, so no read ever stalls.
    std::atomic<bool> stop{ false };
    std::jthread robot(
        [&]
        {
            const std::vector<uint8_t> hello = message_of_size(kHeaderSize + 20, 1);
            const std::vector<uint8_t> heartbeat = message_of_size(kHeaderSize, 11);
            peer.send_all(hello.data(), hello.size());
            while (!stop && peer.send_all(heartbeat.data(), heartbeat.size()))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });
            }
        });
    const auto start = std::chrono::steady_clock::now();
    REQUIRE(server.wait_ready(kWait) == ReceiveResult::Malformed);
    REQUIRE(std::chrono::steady_clock::now() - start < std::chrono::milliseconds{ 500 });
    stop = true;
}

TEST_CASE("RemoteLink: a WebSocket server gives a trickled request 3 s after its first byte", "[remote_link][unit]")
{
    auto [peer, own_end] = socket_pair();
    WebSocketTransport server(std::move(own_end), Role::Server);
    std::atomic<bool> stop{ false };
    std::jthread client(
        [&]
        {
            const std::string header = "GET / HTTP/1.1\r\nX-Pad: ";
            peer.send_all(reinterpret_cast<const uint8_t*>(header.data()), header.size());
            const uint8_t pad = 'x';
            while (!stop && peer.send_all(&pad, 1))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });
            }
        });
    const auto start = std::chrono::steady_clock::now();
    REQUIRE(server.wait_ready(std::chrono::milliseconds{ 10'000 }) == ReceiveResult::Closed);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    stop = true;
    CHECK(elapsed >= std::chrono::milliseconds{ 2900 });
    CHECK(elapsed < std::chrono::milliseconds{ 4500 });
}

TEST_CASE("RemoteLink: connect_transport speaks ws:// to an adopted server transport", "[remote_link][unit]")
{
    Listener listener(Endpoint{ "127.0.0.1", 0 });
    const RemoteAddress remote = parse_remote("ws://127.0.0.1:" + std::to_string(listener.port()) + "/teleop");
    std::unique_ptr<ITransport> client;
    std::thread connector([&] { client = connect_transport(remote, kWait); });
    std::optional<Listener::Accepted> accepted = listener.accept(kWait);
    REQUIRE(accepted.has_value());
    std::unique_ptr<ITransport> server = adopt_transport(accepted->fd, true);
    REQUIRE(server->wait_ready(kWait) == ReceiveResult::Ok);
    connector.join();
    REQUIRE(client);

    const std::vector<uint8_t> sent = message_of_size(kHeaderSize + 200);
    std::vector<uint8_t> received;
    REQUIRE(client->send(sent.data(), sent.size()));
    REQUIRE(server->receive(received, kWait) == ReceiveResult::Ok);
    REQUIRE(received == sent);
    REQUIRE(server->send(sent.data(), sent.size()));
    REQUIRE(client->receive(received, kWait) == ReceiveResult::Ok);
    REQUIRE(received == sent);
    client->shutdown();
    REQUIRE(server->receive(received, kWait) == ReceiveResult::Closed);
}

TEST_CASE("RemoteLink: WebSocket shutdown_write ends the peer's stream and keeps receiving", "[remote_link][unit]")
{
    Connected pair = connected_pair();
    const std::vector<uint8_t> sent = message_of_size(kHeaderSize + 100);
    std::vector<uint8_t> received;

    REQUIRE(pair.server->send(sent.data(), sent.size()));
    pair.server->shutdown_write();
    REQUIRE(pair.client->receive(received, kWait) == ReceiveResult::Ok);
    REQUIRE(received == sent);
    REQUIRE(pair.client->receive(received, kWait) == ReceiveResult::Closed);

    REQUIRE(pair.client->send(sent.data(), sent.size()));
    REQUIRE(pair.server->receive(received, kWait) == ReceiveResult::Ok);
    REQUIRE(received == sent);
}

TEST_CASE("RemoteLink: a WebSocket client started without waiting upgrades once the server answers", "[remote_link][unit]")
{
    auto [client_end, server_end] = socket_pair();
    const RemoteAddress remote{ Endpoint{ "127.0.0.1", 1 }, true, "/teleop" };
    std::unique_ptr<WebSocketTransport> client = WebSocketTransport::start_client(std::move(client_end), remote);
    const std::vector<uint8_t> sent = message_of_size(kHeaderSize + 50);
    REQUIRE_FALSE(client->wait_upgraded(std::chrono::milliseconds{ 20 }));
    REQUIRE(client->wait_ready(std::chrono::milliseconds{ 0 }) == ReceiveResult::Timeout);
    REQUIRE_FALSE(client->send(sent.data(), sent.size()));

    WebSocketTransport server(std::move(server_end), Role::Server);
    REQUIRE(server.wait_ready(kWait) == ReceiveResult::Ok);
    REQUIRE(client->wait_upgraded(kWait));
    REQUIRE(client->wait_upgraded(std::chrono::milliseconds{ 0 }));
    REQUIRE(client->wait_ready(std::chrono::milliseconds{ 0 }) == ReceiveResult::Ok);

    std::vector<uint8_t> received;
    REQUIRE(client->send(sent.data(), sent.size()));
    REQUIRE(server.receive(received, kWait) == ReceiveResult::Ok);
    REQUIRE(received == sent);
    REQUIRE(server.send(sent.data(), sent.size()));
    REQUIRE(client->receive(received, kWait) == ReceiveResult::Ok);
    REQUIRE(received == sent);
}

TEST_CASE("RemoteLink: a WebSocket client started without waiting reports a refused upgrade", "[remote_link][unit]")
{
    auto [client_end, raw_server] = socket_pair();
    const RemoteAddress remote{ Endpoint{ "::1", 47601 }, true, "/teleop" };
    std::unique_ptr<WebSocketTransport> client = WebSocketTransport::start_client(std::move(client_end), remote);
    const std::string request = raw_recv_head(raw_server);
    REQUIRE(request.rfind("GET /teleop HTTP/1.1\r\n", 0) == 0);
    REQUIRE(request.find("\r\nHost: [::1]:47601\r\n") != std::string::npos);
    raw_send(raw_server, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
    SECTION("wait_upgraded throws the refusal")
    {
        try
        {
            client->wait_upgraded(kWait);
            FAIL("no exception");
        }
        catch (const std::runtime_error& e)
        {
            REQUIRE(std::string(e.what()) == "WebSocket upgrade to [::1]:47601/teleop refused: HTTP/1.1 404 Not Found");
        }
    }
    SECTION("wait_ready reports it as Malformed")
    {
        REQUIRE(client->wait_ready(kWait) == ReceiveResult::Malformed);
    }
}

namespace
{

//! What wait_upgraded() threw: whether it was a PeerClosedError, and its text.
std::pair<bool, std::string> upgrade_failure(WebSocketTransport& client)
{
    try
    {
        client.wait_upgraded(kWait);
    }
    catch (const PeerClosedError& e)
    {
        return { true, e.what() };
    }
    catch (const std::runtime_error& e)
    {
        return { false, e.what() };
    }
    FAIL("wait_upgraded did not throw");
    return {};
}

} // namespace

TEST_CASE("RemoteLink: a WebSocket client reports a server that closes before answering as PeerClosedError",
          "[remote_link][unit]")
{
    auto [client_end, raw_server] = socket_pair();
    const RemoteAddress remote{ Endpoint{ "127.0.0.1", 47601 }, true, "/" };
    std::unique_ptr<WebSocketTransport> client = WebSocketTransport::start_client(std::move(client_end), remote);
    raw_recv_head(raw_server);
    raw_server = TcpSocket();
    SECTION("wait_upgraded throws PeerClosedError")
    {
        const auto [peer_closed, text] = upgrade_failure(*client);
        CHECK(peer_closed);
        CHECK(text == "WebSocket upgrade to 127.0.0.1:47601/: connection closed before the answer");
    }
    SECTION("wait_ready reports it as Closed")
    {
        CHECK(client->wait_ready(kWait) == ReceiveResult::Closed);
    }
}

TEST_CASE("RemoteLink: a WebSocket client reports an answer it cannot use as a plain error", "[remote_link][unit]")
{
    const auto [answer, expected] = GENERATE(table<std::string, std::string>({
        { "HTTP/1.1 404 Not Found\r\n\r\n", "WebSocket upgrade to 127.0.0.1:47601/ refused: HTTP/1.1 404 Not Found" },
        { "HTTP/1.1 10", "WebSocket upgrade to 127.0.0.1:47601/: no valid HTTP response" }, // breaks off
        { "RTS0", "WebSocket upgrade to 127.0.0.1:47601/: no valid HTTP response" }, // a tcp:// server's bytes
    }));
    auto [client_end, raw_server] = socket_pair();
    const RemoteAddress remote{ Endpoint{ "127.0.0.1", 47601 }, true, "/" };
    std::unique_ptr<WebSocketTransport> client = WebSocketTransport::start_client(std::move(client_end), remote);
    raw_recv_head(raw_server);
    raw_send(raw_server, answer);
    raw_server = TcpSocket();
    const auto [peer_closed, text] = upgrade_failure(*client);
    CHECK_FALSE(peer_closed);
    CHECK(text == expected);
}

TEST_CASE("RemoteLink: a ws:// client on a TCP port is refused, not closed", "[remote_link][unit]")
{
    Listener listener(Endpoint{ "127.0.0.1", 0 });
    const RemoteAddress remote = parse_remote("ws://127.0.0.1:" + std::to_string(listener.port()));
    std::unique_ptr<WebSocketTransport> client =
        WebSocketTransport::start_client(TcpSocket::connect(remote.endpoint, kWait), remote);
    std::optional<Listener::Accepted> accepted = listener.accept(kWait);
    REQUIRE(accepted.has_value());
    std::unique_ptr<ITransport> server = adopt_transport(accepted->fd, false);
    std::vector<uint8_t> message;
    REQUIRE(server->receive(message, kWait) == ReceiveResult::Malformed);
    // Closed with the rest of the request unread, which resets the connection after the answer.
    server.reset();
    const auto [peer_closed, text] = upgrade_failure(*client);
    CHECK_FALSE(peer_closed);
    CHECK(text == "WebSocket upgrade to " + endpoint_text(remote.endpoint) + "/ refused: HTTP/1.1 400 Bad Request");
}
