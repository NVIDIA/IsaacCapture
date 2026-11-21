// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// NoDataSession: a tracker of every kind is held and reports no data.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <deviceio_session/no_data_session.hpp>
#include <deviceio_trackers/controller_tracker.hpp>
#include <deviceio_trackers/full_body_tracker.hpp>
#include <deviceio_trackers/hand_tracker.hpp>
#include <deviceio_trackers/haptic_command_reader_tracker.hpp>
#include <deviceio_trackers/head_tracker.hpp>
#include <deviceio_trackers/joint_state_tracker.hpp>
#include <deviceio_trackers/keyboard_tracker.hpp>
#include <deviceio_trackers/message_channel_tracker.hpp>
#include <deviceio_trackers/se3_tracker.hpp>
#include <deviceio_trackers/tactile_tracker.hpp>
#include <log_bridge/logger.hpp>
#include <spdlog/sinks/ostream_sink.h>

#include <array>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace
{

core::MessageChannelTracker make_channel()
{
    return core::MessageChannelTracker(std::array<uint8_t, core::MessageChannelTracker::CHANNEL_UUID_SIZE>{});
}

} // namespace

TEST_CASE("NoDataSession: a tracker of every kind reports no data", "[no_data][session]")
{
    core::HeadTracker head;
    core::HandTracker hands;
    core::ControllerTracker controllers;
    core::FullBodyTracker body;
    core::KeyboardTracker keyboard;
    core::HapticCommandReaderTracker haptics("haptics");
    core::Se3Tracker se3("se3");
    core::JointStateTracker joints("joints");

    auto session = core::NoDataSession::run({ &head, &hands, &controllers, &body, &keyboard, &haptics, &se3, &joints });

    CHECK_FALSE(head.get_head(*session));
    CHECK_FALSE(hands.get_left_hand(*session));
    CHECK_FALSE(hands.get_right_hand(*session));
    CHECK_FALSE(controllers.get_left_controller(*session));
    CHECK_FALSE(controllers.get_right_controller(*session));
    CHECK_FALSE(body.get_body_pose(*session));
    CHECK_FALSE(keyboard.get_data(*session));
    CHECK_FALSE(haptics.get_data(*session));
    CHECK_FALSE(se3.get_data(*session));
    CHECK_FALSE(joints.get_data(*session));
}

TEST_CASE("NoDataSession: a message channel is disconnected, has no messages and refuses to send",
          "[no_data][session]")
{
    core::MessageChannelTracker channel = make_channel();
    auto session = core::NoDataSession::run({ &channel });

    CHECK(channel.get_status(*session) == core::MessageChannelStatus::DISCONNECTED);

    // An empty batch, encoded as a live channel's empty drain is: a handle whose list is absent.
    const auto& messages = channel.get_messages(*session);
    REQUIRE(messages);
    CHECK(messages->data() == nullptr);

    CHECK_THROWS_MATCHES(
        channel.send_message(*session, { 1, 2, 3 }), std::runtime_error,
        Catch::Matchers::Message("NoDataSession: MessageChannelTracker::send_message: channel is not open"));
    // The refused message is not kept.
    CHECK(channel.get_messages(*session)->data() == nullptr);
    CHECK(channel.get_status(*session) == core::MessageChannelStatus::DISCONNECTED);
}

TEST_CASE("NoDataSession: a tracker with metadata has no data and no metadata, and logs nothing", "[no_data][session]")
{
    // The logger of the impl such a tracker gets; Logger::get returns the same instance.
    auto logger = isaaccapture::Logger::get("isaaccapture.core.ReplayTactileTrackerImpl");
    std::ostringstream lines;
    logger->sinks().push_back(std::make_shared<spdlog::sinks::ostream_sink_mt>(lines));

    core::TactileTracker tactile("tactile/right");
    auto session = core::NoDataSession::run({ &tactile });

    CHECK_FALSE(tactile.get_data(*session));
    CHECK_FALSE(tactile.get_metadata(*session));
    CHECK(lines.str().empty());

    logger->warn("seen");
    CHECK_FALSE(lines.str().empty());
    logger->sinks().pop_back();
}

TEST_CASE("NoDataSession: every kind in one session, next to each other", "[no_data][session]")
{
    core::HeadTracker head;
    core::MessageChannelTracker first = make_channel();
    core::MessageChannelTracker second = make_channel();
    core::TactileTracker left("tactile/left");
    core::TactileTracker right("tactile/right");

    auto session = core::NoDataSession::run({ &first, &head, &left, &second, &right });

    CHECK_FALSE(head.get_head(*session));
    CHECK(first.get_status(*session) == core::MessageChannelStatus::DISCONNECTED);
    CHECK(second.get_status(*session) == core::MessageChannelStatus::DISCONNECTED);
    CHECK_FALSE(left.get_metadata(*session));
    CHECK_FALSE(right.get_data(*session));
}

TEST_CASE("NoDataSession: refuses a null tracker, a tracker listed twice and a tracker it was not given",
          "[no_data][session]")
{
    core::HeadTracker head;
    core::HeadTracker other;
    core::MessageChannelTracker channel = make_channel();

    CHECK_THROWS_MATCHES(core::NoDataSession::run({ &head, nullptr }), std::invalid_argument,
                         Catch::Matchers::Message("NoDataSession: tracker pointer is null"));
    CHECK_THROWS_MATCHES(core::NoDataSession::run({ &head, &head }), std::invalid_argument,
                         Catch::Matchers::Message("NoDataSession: tracker 'HeadTracker' is listed twice"));
    CHECK_THROWS_MATCHES(
        core::NoDataSession::run({ &channel, &channel }), std::invalid_argument,
        Catch::Matchers::Message("NoDataSession: tracker 'MessageChannelTracker' is listed twice"));

    auto session = core::NoDataSession::run({ &head });
    CHECK_THROWS_MATCHES(other.get_head(*session), std::runtime_error,
                         Catch::Matchers::Message("Tracker implementation not found for tracker: HeadTracker"));

    auto empty = core::NoDataSession::run({});
    CHECK_THROWS_AS(head.get_head(*empty), std::runtime_error);
}
