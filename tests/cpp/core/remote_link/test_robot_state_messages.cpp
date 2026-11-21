// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The robot-state link message kinds: their numbers and names, and how a Link carries them.

#include "test_support.hpp"

#include <catch2/catch_test_macros.hpp>
#include <remote_link/link.hpp>
#include <remote_link/messages.hpp>
#include <remote_link/tcp_transport.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using namespace core;
using namespace remote_link_test;

namespace
{

//! What a Link's handler received, in order.
struct Received
{
    void add(ReceivedLinkMessage&& message)
    {
        std::lock_guard<std::mutex> lock(mutex);
        messages.push_back(std::move(message));
    }

    size_t count()
    {
        std::lock_guard<std::mutex> lock(mutex);
        return messages.size();
    }

    std::mutex mutex;
    std::vector<ReceivedLinkMessage> messages;
};

std::vector<uint8_t> bytes(const std::string& text)
{
    return std::vector<uint8_t>(text.begin(), text.end());
}

} // namespace

TEST_CASE("RemoteLink: robot-state message type ids and names", "[remote_link][unit]")
{
    REQUIRE(static_cast<int>(LinkMessageType::RobotStateSetup) == 40);
    REQUIRE(static_cast<int>(LinkMessageType::RobotStateSamples) == 41);
    REQUIRE(static_cast<int>(LinkMessageType::RobotStateRecords) == 42);
    REQUIRE(std::string(to_string(LinkMessageType::RobotStateSetup)) == "ROBOT_STATE_SETUP");
    REQUIRE(std::string(to_string(LinkMessageType::RobotStateSamples)) == "ROBOT_STATE_SAMPLES");
    REQUIRE(std::string(to_string(LinkMessageType::RobotStateRecords)) == "ROBOT_STATE_RECORDS");
    // The numbers kept free for later robot-state kinds have no name yet.
    REQUIRE(std::string(to_string(static_cast<LinkMessageType>(43))) == "UNKNOWN");
    REQUIRE(std::string(to_string(static_cast<LinkMessageType>(47))) == "UNKNOWN");
}

TEST_CASE("RemoteLink: Link carries robot-state messages and does not count them as unknown", "[remote_link][unit]")
{
    auto [robot_socket, server_socket] = socket_pair();
    Received received;
    Link robot(std::make_unique<TcpTransport>(std::move(robot_socket)), LinkOptions{}, [](ReceivedLinkMessage&&) {});
    Link server(std::make_unique<TcpTransport>(std::move(server_socket)), LinkOptions{},
                [&received](ReceivedLinkMessage&& message) { received.add(std::move(message)); });
    robot.start();
    server.start();

    robot.send_reliable(LinkMessageType::RobotStateSetup, bytes("setup"));
    robot.send_reliable(LinkMessageType::RobotStateRecords, bytes("records"));
    REQUIRE(wait_until([&] { return received.count() == 2; }));
    robot.send_latest(LinkMessageType::RobotStateSamples, bytes("samples"));
    REQUIRE(wait_until([&] { return received.count() == 3; }));
    robot.send_reliable(static_cast<LinkMessageType>(43), bytes("later"));
    REQUIRE(wait_until([&] { return received.count() == 4; }));

    {
        std::lock_guard<std::mutex> lock(received.mutex);
        REQUIRE(received.messages[0].type == LinkMessageType::RobotStateSetup);
        REQUIRE(received.messages[0].payload == bytes("setup"));
        REQUIRE(received.messages[1].type == LinkMessageType::RobotStateRecords);
        REQUIRE(received.messages[1].payload == bytes("records"));
        REQUIRE(received.messages[2].type == LinkMessageType::RobotStateSamples);
        REQUIRE(received.messages[2].payload == bytes("samples"));
        REQUIRE(static_cast<int>(received.messages[3].type) == 43);
    }
    // Only the kind without a name counts as unknown.
    REQUIRE(server.stats().unknown_received == 1);
    REQUIRE(server.alive());
    REQUIRE(robot.alive());
    robot.close();
    server.close();
}
