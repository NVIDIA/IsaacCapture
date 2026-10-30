// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <remote_plugin_session/remote_address.hpp>

#include <stdexcept>

using namespace remote_plugin_session;

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

TEST_CASE("parse_bridge_remote defaults to the bridge port of the scheme", "[remote_device_bridge][unit]")
{
    REQUIRE(parse_bridge_remote("10.0.0.2").endpoint.port == kDefaultBridgePort);
    REQUIRE(parse_bridge_remote("ws://10.0.0.2").endpoint.port == kDefaultBridgeWsPort);
    REQUIRE(parse_bridge_remote("ws://10.0.0.2:9000/bridge").endpoint.port == 9000);
}
