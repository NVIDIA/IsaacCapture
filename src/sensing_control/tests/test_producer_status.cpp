// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <sensing_control/producer_status.hpp>

using namespace plugins::sensing;

namespace
{
StreamStatus stream(const std::string& track)
{
    StreamStatus s;
    s.track = track;
    s.direct_host = "10.0.0.1";
    s.direct_port = 1111;
    s.direct_connected = true;
    return s;
}
} // namespace

TEST_CASE("a new status is Starting with the release and no fault", "[producer_status]")
{
    ProducerStatus status("rel-1");
    const auto s = status.snapshot();
    CHECK(s.state == ProducerState::Starting);
    CHECK_FALSE(s.fault.has_value());
    CHECK(s.release == "rel-1");
    CHECK(s.streams.empty());
}

TEST_CASE("state names", "[producer_status]")
{
    CHECK(std::string(to_string(ProducerState::Starting)) == "starting");
    CHECK(std::string(to_string(ProducerState::Streaming)) == "streaming");
    CHECK(std::string(to_string(ProducerState::CameraFault)) == "camera_fault");
    CHECK(std::string(to_string(ProducerState::Error)) == "error");
    CHECK(std::string(to_string(ProducerState::Stopping)) == "stopping");
}

TEST_CASE("a fault is kept only for the failing states and cleared on leaving them", "[producer_status]")
{
    ProducerStatus status("r");
    const Fault fault{ "code", "message" };

    status.set_state(ProducerState::CameraFault, fault);
    REQUIRE(status.snapshot().fault.has_value());
    CHECK(status.snapshot().fault->code == "code");

    status.set_state(ProducerState::Error, fault);
    CHECK(status.snapshot().fault.has_value());

    status.set_state(ProducerState::Streaming, fault); // a fault passed with a healthy state is ignored
    CHECK(status.snapshot().state == ProducerState::Streaming);
    CHECK_FALSE(status.snapshot().fault.has_value());

    status.set_state(ProducerState::CameraFault, fault);
    status.set_state(ProducerState::Stopping);
    CHECK_FALSE(status.snapshot().fault.has_value());

    status.set_state(ProducerState::CameraFault); // a failing state with no fault stays empty
    CHECK_FALSE(status.snapshot().fault.has_value());
}

TEST_CASE("set_direct_target changes one stream and reports unknown names", "[producer_status]")
{
    ProducerStatus status("r");
    status.set_streams({ stream("left"), stream("right") });

    CHECK(status.set_direct_target("left", "192.168.1.2", 5600));
    auto s = status.snapshot();
    CHECK(s.streams[0].direct_host == "192.168.1.2");
    CHECK(s.streams[0].direct_port == 5600);
    CHECK_FALSE(s.streams[0].direct_connected);
    CHECK(s.streams[1].direct_host == "10.0.0.1");
    CHECK(s.streams[1].direct_connected);

    CHECK(status.set_direct_target("left", "", 5600)); // an empty host clears the port too
    s = status.snapshot();
    CHECK(s.streams[0].direct_host.empty());
    CHECK(s.streams[0].direct_port == 0);

    CHECK_FALSE(status.set_direct_target("nope", "h", 1));
}
