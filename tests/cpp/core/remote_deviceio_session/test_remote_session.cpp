// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Integration tests for RemoteSession: record through McapTrackerChannels into a sample stream,
// as a live DeviceIOSession does when it records into one, feed the stream to a RemoteSession,
// and read the data back through the typed tracker facades.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <deviceio_session/remote_session.hpp>
#include <deviceio_session/remote_session_host.hpp>
#include <deviceio_trackers/controller_tracker.hpp>
#include <deviceio_trackers/full_body_tracker.hpp>
#include <deviceio_trackers/hand_tracker.hpp>
#include <deviceio_trackers/head_tracker.hpp>
#include <deviceio_trackers/joint_se3_pose_tracker.hpp>
#include <deviceio_trackers/message_channel_tracker.hpp>
#include <deviceio_trackers/tactile_tracker.hpp>
#include <mcap/recording_traits.hpp>
#include <mcap/sample_stream.hpp>
#include <mcap/tracker_channels.hpp>
#include <mcap/writer.hpp>
#include <oxr_utils/os_time.hpp>
#include <remote_link/messages.hpp>
#include <schema/controller_generated.h>
#include <schema/full_body_generated.h>
#include <schema/hand_generated.h>
#include <schema/head_generated.h>
#include <schema/joint_se3_pose_generated.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using HeadChannels = core::McapTrackerChannels<core::HeadPoseRecord>;
using HandChannels = core::McapTrackerChannels<core::HandPoseRecord>;
using ControllerChannels = core::McapTrackerChannels<core::ControllerSnapshotRecord>;
using FullBodyChannels = core::McapTrackerChannels<core::FullBodyPoseRecord>;
using JointSe3PoseChannels = core::McapTrackerChannels<core::JointSe3PoseOutputRecord>;

template <typename Array>
std::vector<std::string> to_string_vec(const Array& names)
{
    return std::vector<std::string>(names.begin(), names.end());
}

//! The server half: records the way a DeviceIOSession recording into a stream does.
struct Server
{
    // Declared first so it outlives the writer, which writes the end of the recording on close.
    core::TrackerSampleStreamWriter out;
    mcap::McapWriter writer;

    Server()
    {
        writer.open(out, core::TrackerSampleStreamWriter::writer_options());
    }

    uint64_t next_tick_seq = 1;

    //! Hands one tick's records to `stream`, with a tick time `age_ns` ago.
    void send(core::TrackerSampleStreamReader& stream, int64_t age_ns = 0)
    {
        stream.add_setup(out.take_setup());
        stream.add_samples(out.take_samples(), next_tick_seq++, core::os_monotonic_now_ns() - age_ns);
    }
};

core::DeviceDataTimestamp stamp()
{
    const int64_t now = core::os_monotonic_now_ns();
    return core::DeviceDataTimestamp(now, now, now);
}

void write_head(HeadChannels& ch, float x)
{
    core::HeadPoseT data;
    data.is_valid = true;
    data.pose = std::make_shared<core::Pose>(core::Point(x, 0.0f, 0.0f), core::Quaternion(0.0f, 0.0f, 0.0f, 1.0f));
    ch.write(0, core::pack_record<core::HeadPoseRecord>(&data, stamp()));
}

void write_hand(HandChannels& ch, size_t channel_index, float wrist_radius)
{
    const core::Pose pose(core::Point(0.0f, 0.0f, 0.0f), core::Quaternion(0.0f, 0.0f, 0.0f, 1.0f));
    core::HandPoseT data;
    data.joints = std::make_shared<core::HandJoints>();
    data.joints->mutable_poses()->Mutate(core::HandJoint_WRIST, core::HandJointPose(pose, true, wrist_radius));
    ch.write(channel_index, core::pack_record<core::HandPoseRecord>(&data, stamp()));
}

void write_controller(ControllerChannels& ch, size_t channel_index, float trigger_value)
{
    const core::ControllerInputState inputs(false, false, false, false, 0.0f, 0.0f, 0.0f, trigger_value);
    core::ControllerSnapshotT data;
    data.inputs = std::make_shared<core::ControllerInputState>(inputs);
    ch.write(channel_index, core::pack_record<core::ControllerSnapshotRecord>(&data, stamp()));
}

void write_full_body(FullBodyChannels& ch)
{
    core::FullBodyPoseT data;
    data.all_joint_poses_tracked = true;
    ch.write(0, core::pack_record<core::FullBodyPoseRecord>(&data, stamp()));
}

//! The tracked channel only, which is all replay and remote read of a pusher-fed tracker.
void write_joint_se3_pose(JointSe3PoseChannels& ch, float x)
{
    core::JointSe3PoseOutputT data;
    data.type = core::JointType_HAND_RAW;
    data.joints.emplace_back(core::JointName_HAND_RAW_THUMB_TIP,
                             core::Pose(core::Point(x, 0.0f, 0.0f), core::Quaternion(0.0f, 0.0f, 0.0f, 1.0f)));
    ch.write(1, core::pack_record<core::JointSe3PoseOutputRecord>(&data, stamp()));
}

} // namespace

TEST_CASE("RemoteSession: head and hands round-trip through the stream", "[remote][session]")
{
    Server server;
    HeadChannels head_ch(server.writer, "0", to_string_vec(core::HeadRecordingTraits::recording_channels));
    HandChannels hand_ch(server.writer, "1", to_string_vec(core::HandRecordingTraits::recording_channels));

    core::HeadTracker head;
    core::HandTracker hands;
    auto stream = std::make_shared<core::TrackerSampleStreamReader>();
    auto session = core::RemoteSession::run({ stream, { { &head, "0" }, { &hands, "1" } } });

    session->update();
    CHECK_FALSE(session->is_fresh());
    CHECK_FALSE(head.get_head(*session));

    write_head(head_ch, 1.5f);
    core::HandPoseT left;
    hand_ch.write(0, core::pack_record<core::HandPoseRecord>(&left, stamp()));
    // An inactive right hand is a record with no payload.
    hand_ch.write(1, core::pack_record<core::HandPoseRecord>(nullptr, stamp()));
    server.send(*stream);

    session->update();
    CHECK(session->is_fresh());
    const auto& pose = head.get_head(*session);
    REQUIRE(pose);
    CHECK(pose->pose()->position().x() == 1.5f);
    CHECK(hands.get_left_hand(*session));
    CHECK_FALSE(hands.get_right_hand(*session));
}

TEST_CASE("RemoteSession: every tracker kind reports no data until its channels are declared", "[remote][session]")
{
    Server server;
    core::HeadTracker head;
    core::HandTracker hands;
    core::ControllerTracker controllers;
    core::FullBodyTracker body;
    core::JointSe3PoseTracker tips("manus_sensors_left");
    auto stream = std::make_shared<core::TrackerSampleStreamReader>();
    core::RemoteSessionConfig config;
    config.stream = stream;
    config.tracker_names = { { &head, "0" }, { &hands, "1" }, { &controllers, "2" }, { &body, "3" }, { &tips, "4" } };
    auto session = core::RemoteSession::run(config);

    const auto check_no_data = [&]
    {
        CHECK_FALSE(session->is_fresh());
        CHECK_FALSE(session->tick_age_ns().has_value());
        CHECK_FALSE(head.get_head(*session));
        CHECK_FALSE(hands.get_left_hand(*session));
        CHECK_FALSE(hands.get_right_hand(*session));
        CHECK_FALSE(controllers.get_left_controller(*session));
        CHECK_FALSE(controllers.get_right_controller(*session));
        CHECK_FALSE(body.get_body_pose(*session));
        CHECK_FALSE(tips.get_data(*session));
    };

    // The stream holds no Schema, Channel or Message record, as before the server's session exists.
    for (int update = 0; update < 3; ++update)
    {
        CAPTURE(update);
        REQUIRE_NOTHROW(session->update());
        check_no_data();
    }

    HeadChannels head_ch(server.writer, "0", to_string_vec(core::HeadRecordingTraits::recording_channels));
    HandChannels hand_ch(server.writer, "1", to_string_vec(core::HandRecordingTraits::recording_channels));
    ControllerChannels controller_ch(
        server.writer, "2", to_string_vec(core::ControllerRecordingTraits::recording_channels));
    FullBodyChannels body_ch(server.writer, "3", to_string_vec(core::FullBodyRecordingTraits::recording_channels));
    JointSe3PoseChannels tips_ch(
        server.writer, "4", to_string_vec(core::JointSe3PoseRecordingTraits::recording_channels));

    // Declared channels with no record yet.
    stream->add_setup(server.out.take_setup());
    REQUIRE_NOTHROW(session->update());
    check_no_data();

    write_head(head_ch, 1.5f);
    write_hand(hand_ch, 0, 0.25f);
    write_hand(hand_ch, 1, 0.75f);
    write_controller(controller_ch, 0, 0.25f);
    write_controller(controller_ch, 1, 0.75f);
    write_full_body(body_ch);
    write_joint_se3_pose(tips_ch, 3.0f);
    server.send(*stream);

    REQUIRE_NOTHROW(session->update());
    CHECK(session->is_fresh());
    CHECK(session->tick_age_ns().has_value());
    REQUIRE(head.get_head(*session));
    CHECK(head.get_head(*session)->pose()->position().x() == 1.5f);
    REQUIRE(hands.get_left_hand(*session));
    CHECK(hands.get_left_hand(*session)->joints()->poses()->Get(core::HandJoint_WRIST)->radius() == 0.25f);
    REQUIRE(hands.get_right_hand(*session));
    CHECK(hands.get_right_hand(*session)->joints()->poses()->Get(core::HandJoint_WRIST)->radius() == 0.75f);
    REQUIRE(controllers.get_left_controller(*session));
    CHECK(controllers.get_left_controller(*session)->inputs()->trigger_value() == 0.25f);
    REQUIRE(controllers.get_right_controller(*session));
    CHECK(controllers.get_right_controller(*session)->inputs()->trigger_value() == 0.75f);
    REQUIRE(body.get_body_pose(*session));
    CHECK(body.get_body_pose(*session)->all_joint_poses_tracked());
    REQUIRE(tips.get_data(*session));
    CHECK(tips.get_data(*session)->joints()->Get(0)->pose().position().x() == 3.0f);
}

TEST_CASE("RemoteSession: every update sees the newest samples", "[remote][session]")
{
    Server server;
    HeadChannels head_ch(server.writer, "0", to_string_vec(core::HeadRecordingTraits::recording_channels));
    core::HeadTracker head;
    auto stream = std::make_shared<core::TrackerSampleStreamReader>();
    auto session = core::RemoteSession::run({ stream, { { &head, "0" } } });

    write_head(head_ch, 1.0f);
    server.send(*stream);
    write_head(head_ch, 2.0f);
    server.send(*stream);

    session->update();
    CHECK(head.get_head(*session)->pose()->position().x() == 2.0f);

    // No new samples: the newest are published again while they are fresh.
    session->update();
    CHECK(head.get_head(*session)->pose()->position().x() == 2.0f);
}

TEST_CASE("RemoteSession: a pusher-fed tracker keeps its last sample until a new one arrives", "[remote][session]")
{
    Server server;
    HeadChannels head_ch(server.writer, "0", to_string_vec(core::HeadRecordingTraits::recording_channels));
    JointSe3PoseChannels tips_ch(
        server.writer, "1", to_string_vec(core::JointSe3PoseRecordingTraits::recording_channels));

    core::HeadTracker head;
    core::JointSe3PoseTracker tips("manus_sensors_left");
    auto stream = std::make_shared<core::TrackerSampleStreamReader>();
    auto session = core::RemoteSession::run({ stream, { { &head, "0" }, { &tips, "1" } } });

    write_head(head_ch, 1.0f);
    write_joint_se3_pose(tips_ch, 3.0f);
    server.send(*stream);
    session->update();
    REQUIRE(tips.get_data(*session));

    // The next update brings head data but no new tips sample, as when the plugin pushed none.
    write_head(head_ch, 2.0f);
    server.send(*stream);
    session->update();
    const auto& kept = tips.get_data(*session);
    REQUIRE(kept);
    CHECK(kept->joints()->Get(0)->pose().position().x() == 3.0f);
}

TEST_CASE("RemoteSession: a stale stream reports no data", "[remote][session]")
{
    Server server;
    HeadChannels head_ch(server.writer, "0", to_string_vec(core::HeadRecordingTraits::recording_channels));
    core::HeadTracker head;
    auto stream = std::make_shared<core::TrackerSampleStreamReader>();
    auto session = core::RemoteSession::run({ stream, { { &head, "0" } }, 250'000'000 });

    write_head(head_ch, 1.0f);
    server.send(*stream, 1'000'000'000);
    session->update();
    CHECK_FALSE(session->is_fresh());
    CHECK_FALSE(head.get_head(*session));

    write_head(head_ch, 2.0f);
    server.send(*stream);
    session->update();
    CHECK(session->is_fresh());
    REQUIRE(head.get_head(*session));
}

TEST_CASE("RemoteSession: rejects a config without a stream", "[remote][session]")
{
    core::HeadTracker head;
    CHECK_THROWS_AS(core::RemoteSession::run({ nullptr, { { &head, "0" } } }), std::invalid_argument);
}

TEST_CASE("RemoteSession: trackers subscribe to the topics they read", "[remote][session]")
{
    core::HeadTracker head;
    core::HandTracker hands;
    auto stream = std::make_shared<core::TrackerSampleStreamReader>();
    auto session = core::RemoteSession::run({ stream, { { &head, "1" }, { &hands, "0" } } });

    CHECK(stream->subscribed_topics() == std::vector<std::string>{ "0/left_hand", "0/right_hand", "1/head" });
}

TEST_CASE("RemoteSession: an incompatible declared schema fails the first update, before any record", "[remote][session]")
{
    Server server;
    // "0/head" declared with the hand record's schema.
    HandChannels wrong(server.writer, "0", { "head" });
    wrong.write(0, core::pack_record<core::HandPoseRecord>(nullptr, stamp()));

    core::HeadTracker head;
    auto stream = std::make_shared<core::TrackerSampleStreamReader>();
    auto session = core::RemoteSession::run({ stream, { { &head, "0" } } });

    // Only the setup arrives; no samples do.
    stream->add_setup(server.out.take_setup());
    CHECK_THROWS_WITH(session->update(), Catch::Matchers::ContainsSubstring("MCAP schema mismatch on '0/head'"));
    CHECK_FALSE(session->is_fresh());
}

TEST_CASE("RemoteSession: rejects a MessageChannelTracker before subscribing anything", "[remote][session]")
{
    core::HeadTracker head;
    core::MessageChannelTracker channel(std::array<uint8_t, core::MessageChannelTracker::CHANNEL_UUID_SIZE>{});
    auto stream = std::make_shared<core::TrackerSampleStreamReader>();

    CHECK_THROWS_MATCHES(core::RemoteSession::run({ stream, { { &head, "0" }, { &channel, "1" } } }),
                         std::invalid_argument,
                         Catch::Matchers::Message("MessageChannelTracker is not supported over a remote session"));
    CHECK(stream->subscribed_topics().empty());
}

TEST_CASE("RemoteSession: rejects a tracker with metadata before subscribing anything", "[remote][session]")
{
    core::HeadTracker head;
    core::TactileTracker tactile("tactile/right");
    auto stream = std::make_shared<core::TrackerSampleStreamReader>();

    CHECK_THROWS_MATCHES(core::RemoteSession::run({ stream, { { &head, "0" }, { &tactile, "1" } } }),
                         std::invalid_argument,
                         Catch::Matchers::Message("TactileTracker is not supported over a remote session: its "
                                                  "metadata is read before any sample arrives"));
    CHECK(stream->subscribed_topics().empty());
}

TEST_CASE("RemoteSession: tick_age_ns is the published samples' tick age", "[remote][session]")
{
    constexpr int64_t kAge = 50'000'000;
    constexpr int64_t kMaxAge = 250'000'000;

    Server server;
    HeadChannels head_ch(server.writer, "0", to_string_vec(core::HeadRecordingTraits::recording_channels));
    core::HeadTracker head;
    auto stream = std::make_shared<core::TrackerSampleStreamReader>();
    auto session = core::RemoteSession::run({ stream, { { &head, "0" } }, kMaxAge });

    session->update();
    CHECK_FALSE(session->tick_age_ns().has_value());

    write_head(head_ch, 1.0f);
    server.send(*stream, kAge);
    session->update();
    REQUIRE(session->tick_age_ns().has_value());
    CHECK(*session->tick_age_ns() >= kAge);
    CHECK(*session->tick_age_ns() <= kMaxAge);

    // A newer tick whose mapped tick time is older still replaces it; by then it is stale.
    write_head(head_ch, 2.0f);
    server.send(*stream, 2 * kMaxAge);
    session->update();
    CHECK_FALSE(session->is_fresh());
    CHECK_FALSE(session->tick_age_ns().has_value());
    CHECK_FALSE(head.get_head(*session));

    write_head(head_ch, 3.0f);
    server.send(*stream);
    session->update();
    CHECK(session->is_fresh());
    REQUIRE(session->tick_age_ns().has_value());
    CHECK(*session->tick_age_ns() < kMaxAge);
    CHECK(head.get_head(*session)->pose()->position().x() == 3.0f);
}

TEST_CASE("RemoteSession: ticks are ordered by tick_seq", "[remote][session]")
{
    Server server;
    HeadChannels head_ch(server.writer, "0", to_string_vec(core::HeadRecordingTraits::recording_channels));
    core::HeadTracker head;
    auto stream = std::make_shared<core::TrackerSampleStreamReader>();
    auto session = core::RemoteSession::run({ stream, { { &head, "0" } } });

    write_head(head_ch, 1.0f);
    stream->add_setup(server.out.take_setup());
    const auto first = server.out.take_samples();
    write_head(head_ch, 2.0f);
    const auto second = server.out.take_samples();

    const int64_t now = core::os_monotonic_now_ns();
    REQUIRE(stream->add_samples(second, 2, now - 10'000'000));
    // A later tick time by the mapped clock, but an older tick.
    CHECK_FALSE(stream->add_samples(first, 1, now));
    session->update();
    CHECK(head.get_head(*session)->pose()->position().x() == 2.0f);
}

TEST_CASE("RemoteSession: merged SAMPLES payloads keep the newer prefix and every channel's newest record",
          "[remote][session]")
{
    Server server;
    HeadChannels head_ch(server.writer, "0", to_string_vec(core::HeadRecordingTraits::recording_channels));
    JointSe3PoseChannels tips_ch(
        server.writer, "1", to_string_vec(core::JointSe3PoseRecordingTraits::recording_channels));

    write_head(head_ch, 1.0f);
    write_joint_se3_pose(tips_ch, 3.0f);
    const auto setup = server.out.take_setup();
    const auto older = core::encode_samples_payload({ 1, 100 }, server.out.take_samples());
    // No new tips sample in the newer tick.
    write_head(head_ch, 2.0f);
    const auto newer = core::encode_samples_payload({ 2, 200 }, server.out.take_samples());

    const std::vector<uint8_t> merged = core::merge_samples_payloads(older, newer);
    const core::SamplesPrefix prefix = core::decode_samples_prefix(merged);
    CHECK(prefix.tick_seq == 2);
    CHECK(prefix.tick_time_ns == 200);

    core::HeadTracker head;
    core::JointSe3PoseTracker tips("manus_sensors_left");
    auto stream = std::make_shared<core::TrackerSampleStreamReader>();
    auto session = core::RemoteSession::run({ stream, { { &head, "0" }, { &tips, "1" } } });
    stream->add_setup(setup);
    REQUIRE(stream->add_samples(std::span<const uint8_t>(merged).subspan(core::kSamplesPrefixSize), prefix.tick_seq,
                                core::os_monotonic_now_ns()));
    session->update();
    REQUIRE(head.get_head(*session));
    CHECK(head.get_head(*session)->pose()->position().x() == 2.0f);
    REQUIRE(tips.get_data(*session));
    CHECK(tips.get_data(*session)->joints()->Get(0)->pose().position().x() == 3.0f);

    CHECK_THROWS(core::merge_samples_payloads({ 1, 2, 3 }, newer));
    CHECK_THROWS(core::merge_samples_payloads(older, { 1, 2, 3 }));
}
