// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "test_support.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <remote_link/message_header.hpp>
#include <remote_link/tcp_transport.hpp>
#include <remote_link/websocket_transport.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <memory>
#include <optional>
#include <pthread.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace core;
using namespace remote_link_test;

TEST_CASE("RemoteLink: parse_endpoint", "[remote_link][unit]")
{
    Endpoint e = parse_endpoint("10.0.0.2:5000", 1);
    REQUIRE(e.host == "10.0.0.2");
    REQUIRE(e.port == 5000);

    e = parse_endpoint("robot.local", 47600);
    REQUIRE(e.host == "robot.local");
    REQUIRE(e.port == 47600);

    e = parse_endpoint("[::1]:47600", 1);
    REQUIRE(e.host == "::1");
    REQUIRE(e.port == 47600);

    e = parse_endpoint("[fe80::1]", 9);
    REQUIRE(e.host == "fe80::1");
    REQUIRE(e.port == 9);

    e = parse_endpoint("::1", 9);
    REQUIRE(e.host == "::1");
    REQUIRE(e.port == 9);

    e = parse_endpoint("127.0.0.1:0", 9);
    REQUIRE(e.port == 0);

    for (const char* bad : { "", ":5000", "host:", "host:x", "host:65536", "host:-1", "[::1", "[::1]x", "[]:1" })
    {
        INFO(bad);
        REQUIRE_THROWS_AS(parse_endpoint(bad, 1), std::invalid_argument);
    }
}

TEST_CASE("RemoteLink: endpoint_text brackets IPv6 literals only", "[remote_link][unit]")
{
    CHECK(endpoint_text(Endpoint{ "127.0.0.1", 47600 }) == "127.0.0.1:47600");
    CHECK(endpoint_text(Endpoint{ "robot.local", 1 }) == "robot.local:1");
    CHECK(endpoint_text(Endpoint{ "::1", 47600 }) == "[::1]:47600");
    CHECK(endpoint_text(Endpoint{ "fe80::1%eth0", 9 }) == "[fe80::1%eth0]:9");
    for (const Endpoint& endpoint : { Endpoint{ "10.0.0.2", 5000 }, Endpoint{ "::1", 0 }, Endpoint{ "h", 65535 } })
    {
        const Endpoint parsed = parse_endpoint(endpoint_text(endpoint), 1);
        CHECK(parsed.host == endpoint.host);
        CHECK(parsed.port == endpoint.port);
    }
}

TEST_CASE("RemoteLink: parse_remote", "[remote_link][unit]")
{
    RemoteAddress r = parse_remote("10.0.0.2:5000");
    REQUIRE_FALSE(r.websocket);
    REQUIRE(r.endpoint.host == "10.0.0.2");
    REQUIRE(r.endpoint.port == 5000);

    r = parse_remote("tcp://server");
    REQUIRE_FALSE(r.websocket);
    REQUIRE(r.endpoint.host == "server");
    REQUIRE(r.endpoint.port == kDefaultTcpPort);
    REQUIRE(kDefaultTcpPort == 47600);

    r = parse_remote("server");
    REQUIRE(r.endpoint.port == 47600);

    r = parse_remote("ws://server");
    REQUIRE(r.websocket);
    REQUIRE(r.endpoint.port == kDefaultWebSocketPort);
    REQUIRE(kDefaultWebSocketPort == 47601);
    REQUIRE(r.path == "/");

    r = parse_remote("ws://server:8080/teleop/x");
    REQUIRE(r.websocket);
    REQUIRE(r.endpoint.host == "server");
    REQUIRE(r.endpoint.port == 8080);
    REQUIRE(r.path == "/teleop/x");

    r = parse_remote("tcp://[::1]:47600");
    REQUIRE(r.endpoint.host == "::1");
    REQUIRE(r.endpoint.port == 47600);

    r = parse_remote("ws://[::1]/p");
    REQUIRE(r.websocket);
    REQUIRE(r.endpoint.host == "::1");
    REQUIRE(r.endpoint.port == 47601);
    REQUIRE(r.path == "/p");

    try
    {
        parse_remote("wss://server");
        FAIL("wss:// must be rejected");
    }
    catch (const std::invalid_argument& e)
    {
        REQUIRE(std::string(e.what()) ==
                "wss:// is not supported yet; use ws:// through an encrypted tunnel (VPN, SSH or stunnel) between the "
                "two machines");
    }
    REQUIRE_THROWS_AS(parse_remote("http://server"), std::invalid_argument);
    REQUIRE_THROWS_AS(parse_remote("ws:///path"), std::invalid_argument);
    REQUIRE_THROWS_AS(parse_remote("tcp://server/path"), std::invalid_argument);
}

TEST_CASE("RemoteLink: TcpTransport carries whole link messages over a stream", "[remote_link][unit]")
{
    auto [a_end, b_end] = socket_pair();
    TcpTransport a(std::move(a_end));
    TcpTransport b(std::move(b_end));

    const size_t message_size =
        GENERATE(kHeaderSize, kHeaderSize + 1, size_t{ 65536 }, size_t{ 1 } << 20, kHeaderSize + kMaxPayloadSize);
    const std::vector<uint8_t> sent = message_of_size(message_size);
    bool send_ok = false;
    std::thread sender(
        [&]
        {
            // Two messages back to back: the receiver must split them at the header's payload size.
            send_ok = a.send(sent.data(), sent.size()) && a.send(sent.data(), sent.size());
        });
    std::vector<uint8_t> received;
    REQUIRE(b.receive(received, kWait) == ReceiveResult::Ok);
    REQUIRE(received == sent);
    REQUIRE(b.receive(received, kWait) == ReceiveResult::Ok);
    REQUIRE(received == sent);
    sender.join();
    REQUIRE(send_ok);
    REQUIRE(b.receive(received, std::chrono::milliseconds{ 20 }) == ReceiveResult::Timeout);
}

TEST_CASE("RemoteLink: TcpTransport reports timeout, EOF, truncation and bad headers", "[remote_link][unit]")
{
    auto [raw, own_end] = socket_pair();
    TcpTransport transport(std::move(own_end));
    std::vector<uint8_t> message;

    SECTION("timeout before the first byte")
    {
        REQUIRE(transport.receive(message, std::chrono::milliseconds{ 20 }) == ReceiveResult::Timeout);
    }
    SECTION("peer EOF")
    {
        raw.shutdown();
        REQUIRE(transport.receive(message, kWait) == ReceiveResult::Closed);
    }
    SECTION("peer EOF in the middle of a header")
    {
        const std::vector<uint8_t> full = message_of_size(kHeaderSize);
        REQUIRE(raw.send_all(full.data(), 4));
        raw.shutdown();
        REQUIRE(transport.receive(message, kWait) == ReceiveResult::Truncated);
    }
    SECTION("peer EOF in the middle of a payload")
    {
        const std::vector<uint8_t> full = message_of_size(kHeaderSize + 100);
        REQUIRE(raw.send_all(full.data(), kHeaderSize + 10));
        raw.shutdown();
        REQUIRE(transport.receive(message, kWait) == ReceiveResult::Truncated);
    }
    SECTION("bad magic")
    {
        std::vector<uint8_t> bad = message_of_size(kHeaderSize);
        bad[0] = 'X';
        REQUIRE(raw.send_all(bad.data(), bad.size()));
        REQUIRE(transport.receive(message, kWait) == ReceiveResult::Malformed);
    }
    SECTION("oversized payload")
    {
        std::vector<uint8_t> bad = message_of_size(kHeaderSize);
        LinkMessageHeader header = decode_header(bad.data());
        header.payload_size = kMaxPayloadSize + 1;
        encode_header(header, bad.data());
        REQUIRE(raw.send_all(bad.data(), bad.size()));
        REQUIRE(transport.receive(message, kWait) == ReceiveResult::Malformed);
    }
    SECTION("shutdown unblocks a waiting receive")
    {
        std::thread closer(
            [&]
            {
                std::this_thread::sleep_for(std::chrono::milliseconds{ 50 });
                transport.shutdown();
            });
        const auto start = std::chrono::steady_clock::now();
        REQUIRE(transport.receive(message, std::chrono::milliseconds{ 5000 }) == ReceiveResult::Closed);
        closer.join();
        REQUIRE(std::chrono::steady_clock::now() - start < std::chrono::milliseconds{ 2000 });
        const std::vector<uint8_t> out = message_of_size(kHeaderSize);
        REQUIRE_FALSE(transport.send(out.data(), out.size()));
    }
}

TEST_CASE("RemoteLink: TcpTransport answers an HTTP request in place of the first message with 400", "[remote_link][unit]")
{
    auto [raw, own_end] = socket_pair();
    TcpTransport transport(std::move(own_end));
    std::vector<uint8_t> message;
    const std::string request =
        "GET / HTTP/1.1\r\nHost: 127.0.0.1:47600\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n";
    const auto send_request = [&raw = raw, &request]
    { REQUIRE(raw.send_all(reinterpret_cast<const uint8_t*>(request.data()), request.size())); };
    uint8_t byte = 0;

    SECTION("the first message")
    {
        send_request();
        REQUIRE(transport.receive(message, kWait) == ReceiveResult::Malformed);
        const std::string answer = "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        std::vector<uint8_t> received(answer.size());
        REQUIRE(raw.recv_all(received.data(), received.size(), kWait) == TcpSocket::RecvResult::Ok);
        CHECK(std::string(received.begin(), received.end()) == answer);
        CHECK(raw.recv_all(&byte, 1, kWait) == TcpSocket::RecvResult::Closed);
        const std::vector<uint8_t> out = message_of_size(kHeaderSize);
        CHECK_FALSE(transport.send(out.data(), out.size()));
    }
    SECTION("not after a message")
    {
        const std::vector<uint8_t> first = message_of_size(kHeaderSize + 10, 1);
        REQUIRE(raw.send_all(first.data(), first.size()));
        REQUIRE(transport.receive(message, kWait) == ReceiveResult::Ok);
        send_request();
        REQUIRE(transport.receive(message, kWait) == ReceiveResult::Malformed);
        CHECK(raw.recv_all(&byte, 1, kShort) == TcpSocket::RecvResult::Timeout);
    }
    SECTION("not other bytes")
    {
        std::vector<uint8_t> bad = message_of_size(kHeaderSize);
        bad[0] = 'X';
        REQUIRE(raw.send_all(bad.data(), bad.size()));
        REQUIRE(transport.receive(message, kWait) == ReceiveResult::Malformed);
        CHECK(raw.recv_all(&byte, 1, kShort) == TcpSocket::RecvResult::Timeout);
    }
}

TEST_CASE("RemoteLink: TcpSocket wait_writable, release and peer EOF on send", "[remote_link][unit]")
{
    auto [a, b] = socket_pair();
    REQUIRE(a.wait_writable(std::chrono::milliseconds{ 0 }));
    const int fd = b.release();
    REQUIRE_FALSE(b.valid());
    REQUIRE(fd >= 0);
    ::close(fd);
    const std::vector<uint8_t> out = message_of_size(kHeaderSize + 10);
    // The first send after the peer closed may still be accepted by the kernel; a later one fails (no SIGPIPE).
    bool failed = false;
    for (int i = 0; i < 10 && !failed; ++i)
    {
        failed = !a.send_all(out.data(), out.size());
    }
    REQUIRE(failed);
}

TEST_CASE("RemoteLink: Listener accepts a TCP connection on an ephemeral port", "[remote_link][unit]")
{
    Listener listener(Endpoint{ "127.0.0.1", 0 });
    REQUIRE(listener.port() != 0);
    REQUIRE_FALSE(listener.accept(std::chrono::milliseconds{ 10 }).has_value());

    std::unique_ptr<ITransport> client;
    std::thread connector(
        [&] { client = connect_transport(parse_remote("127.0.0.1:" + std::to_string(listener.port())), kWait); });
    std::optional<Listener::Accepted> accepted = listener.accept(kWait);
    connector.join();
    REQUIRE(accepted.has_value());
    REQUIRE(accepted->fd >= 0);
    REQUIRE(accepted->peer.rfind("127.0.0.1:", 0) == 0);
    REQUIRE(client);

    std::unique_ptr<ITransport> server = adopt_transport(accepted->fd, false);
    REQUIRE(server->wait_ready(std::chrono::milliseconds{ 0 }) == ReceiveResult::Ok);
    const std::vector<uint8_t> sent = message_of_size(kHeaderSize + 300);
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

TEST_CASE("RemoteLink: Listener binds IPv6 loopback when available", "[remote_link][unit]")
{
    std::optional<Listener> listener;
    try
    {
        listener.emplace(Endpoint{ "::1", 0 });
    }
    catch (const std::runtime_error&)
    {
        SKIP("no IPv6 loopback on this host");
    }
    std::unique_ptr<ITransport> client;
    std::thread connector(
        [&] { client = connect_transport(parse_remote("[::1]:" + std::to_string(listener->port())), kWait); });
    std::optional<Listener::Accepted> accepted = listener->accept(kWait);
    connector.join();
    REQUIRE(accepted.has_value());
    REQUIRE(client);
    CHECK_THAT(accepted->peer, Catch::Matchers::StartsWith("[::1]:"));
    ::close(accepted->fd);
}

TEST_CASE("RemoteLink: IPv6 endpoints in error texts and peer names are in brackets", "[remote_link][unit]")
{
    std::optional<Listener> listener;
    try
    {
        listener.emplace(Endpoint{ "::1", 0 });
    }
    catch (const std::runtime_error&)
    {
        SKIP("no IPv6 loopback on this host");
    }
    const uint16_t port_number = listener->port();
    const std::string port = std::to_string(port_number);

    TcpSocket client = TcpSocket::connect(Endpoint{ "::1", port_number }, kWait);
    CHECK(client.peer_name() == "[::1]:" + port);
    std::optional<Listener::Accepted> accepted = listener->accept(kWait);
    REQUIRE(accepted.has_value());
    ::close(accepted->fd);

    REQUIRE_THROWS_WITH(
        Listener(Endpoint{ "::1", port_number }), Catch::Matchers::StartsWith("cannot listen on [::1]:" + port + ": "));
    listener->close();
    REQUIRE_THROWS_WITH(TcpSocket::connect(Endpoint{ "::1", port_number }, kWait),
                        Catch::Matchers::StartsWith("cannot connect to [::1]:" + port + ": "));
}

TEST_CASE("RemoteLink: IPv4 endpoints in error texts and peer names are unchanged", "[remote_link][unit]")
{
    Listener listener(Endpoint{ "127.0.0.1", 0 });
    const uint16_t port_number = listener.port();
    const std::string port = std::to_string(port_number);
    TcpSocket client = TcpSocket::connect(Endpoint{ "127.0.0.1", port_number }, kWait);
    CHECK(client.peer_name() == "127.0.0.1:" + port);
    std::optional<Listener::Accepted> accepted = listener.accept(kWait);
    REQUIRE(accepted.has_value());
    CHECK_THAT(accepted->peer, Catch::Matchers::StartsWith("127.0.0.1:"));
    ::close(accepted->fd);

    REQUIRE_THROWS_WITH(Listener(Endpoint{ "127.0.0.1", port_number }),
                        Catch::Matchers::StartsWith("cannot listen on 127.0.0.1:" + port + ": "));
    listener.close();
    REQUIRE_THROWS_WITH(TcpSocket::connect(Endpoint{ "127.0.0.1", port_number }, kWait),
                        Catch::Matchers::StartsWith("cannot connect to 127.0.0.1:" + port + ": "));
}

TEST_CASE("RemoteLink: a closed Listener refuses connections", "[remote_link][unit]")
{
    Listener listener(Endpoint{ "127.0.0.1", 0 });
    const std::string endpoint = "127.0.0.1:" + std::to_string(listener.port());
    listener.close();
    listener.close();
    REQUIRE_THROWS_AS(listener.accept(std::chrono::milliseconds{ 0 }), std::runtime_error);
    REQUIRE_THROWS_WITH(
        connect_transport(parse_remote(endpoint), kWait), Catch::Matchers::ContainsSubstring("Connection refused"));
}

namespace
{

void ignore_signal(int)
{
}

} // namespace

TEST_CASE("RemoteLink: signals do not lengthen a timed wait", "[remote_link][unit]")
{
    // Handled without SA_RESTART, as Python handles SIGINT: every one interrupts poll().
    struct sigaction action = {};
    action.sa_handler = ignore_signal;
    sigemptyset(&action.sa_mask);
    struct sigaction previous = {};
    REQUIRE(sigaction(SIGUSR1, &action, &previous) == 0);
    Listener listener(Endpoint{ "127.0.0.1", 0 });
    const pthread_t waiter = pthread_self();
    std::atomic<bool> stop{ false };
    std::jthread signaller(
        [&]
        {
            // For 1 s at most, so a wait that each signal restarts still ends.
            for (int i = 0; i < 20 && !stop; ++i)
            {
                pthread_kill(waiter, SIGUSR1);
                std::this_thread::sleep_for(std::chrono::milliseconds{ 50 });
            }
        });
    const auto start = std::chrono::steady_clock::now();
    const std::optional<Listener::Accepted> accepted = listener.accept(std::chrono::milliseconds{ 500 });
    const auto elapsed = std::chrono::steady_clock::now() - start;
    stop = true;
    signaller.join();
    sigaction(SIGUSR1, &previous, nullptr);
    CHECK_FALSE(accepted.has_value());
    CHECK(elapsed >= std::chrono::milliseconds{ 450 });
    CHECK(elapsed < std::chrono::milliseconds{ 900 });
}

TEST_CASE("RemoteLink: connect_transport fails with nothing listening", "[remote_link][unit]")
{
    uint16_t port = 0;
    {
        Listener listener(Endpoint{ "127.0.0.1", 0 });
        port = listener.port();
    }
    REQUIRE_THROWS_AS(connect_transport(parse_remote("127.0.0.1:" + std::to_string(port)), kWait), std::runtime_error);
}

TEST_CASE("RemoteLink: TcpTransport shutdown_write ends the peer's stream and keeps receiving", "[remote_link][unit]")
{
    auto [a_end, b_end] = socket_pair();
    TcpTransport a(std::move(a_end));
    TcpTransport b(std::move(b_end));
    const std::vector<uint8_t> sent = message_of_size(kHeaderSize + 100);
    std::vector<uint8_t> received;

    REQUIRE(a.send(sent.data(), sent.size()));
    a.shutdown_write();
    REQUIRE(b.receive(received, kWait) == ReceiveResult::Ok);
    REQUIRE(received == sent);
    REQUIRE(b.receive(received, kWait) == ReceiveResult::Closed);

    REQUIRE(b.send(sent.data(), sent.size()));
    REQUIRE(a.receive(received, kWait) == ReceiveResult::Ok);
    REQUIRE(received == sent);
    REQUIRE_FALSE(a.send(sent.data(), sent.size()));
}
