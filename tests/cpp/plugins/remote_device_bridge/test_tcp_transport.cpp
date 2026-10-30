// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <remote_plugin_session/protocol_frame.hpp>
#include <remote_plugin_session/tcp_transport.hpp>
#include <sys/socket.h>

#include <chrono>
#include <cstdint>
#include <utility>
#include <vector>

using namespace remote_plugin_session;
namespace protocol = remote_plugin_protocol;

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

//! A wire frame of exactly `message_size` bytes.
std::vector<uint8_t> frame_of_size(size_t message_size)
{
    return protocol::encode_frame(
        protocol::FrameType::SchemaSample, 3, std::vector<uint8_t>(message_size - protocol::kHeaderSize, 0x5A));
}

void raw_send(TcpSocket& socket, const std::vector<uint8_t>& bytes)
{
    REQUIRE(socket.send_all(bytes.data(), bytes.size()));
}

} // namespace

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
