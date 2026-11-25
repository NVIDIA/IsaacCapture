// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sdk_doubles.hpp"

#include <avatar/avatar_hand_tracking_plugin.hpp>
#include <catch2/catch_test_macros.hpp>

namespace plugins::avatar
{
struct AvatarTrackerTestAccess
{
    static auto& gloves(AvatarTracker& tracker)
    {
        return tracker.m_gloves;
    }
    static auto& injectors(AvatarTracker& tracker)
    {
        return tracker.m_injectors;
    }
    static auto& pushers(AvatarTracker& tracker)
    {
        return tracker.m_joint_pushers;
    }
};
} // namespace plugins::avatar
using plugins::avatar::AvatarTrackerTestAccess;

namespace
{
using plugins::avatar::AvatarTracker;
using namespace std::chrono_literals;
void check(bool value, const char* message)
{
    INFO(message);
    REQUIRE(value);
}
struct Rig
{
    std::unique_ptr<AvatarTracker> tracker;
    Rig()
    {
        tracker_test_double::created.clear();
        tracker_test_double::destroyed.clear();
        tracker_test_double::pushed.clear();
        tracker_test_double::attempts.clear();
        tracker_test_double::fail.clear();
        tracker_test_double::push_fail.clear();
        plugin_utils::PluginDeviceStatusPublisher::latest.clear();
        for (int i = 0; i < 2; ++i)
        {
            avatar::AvatarSDK::devices[i] = std::make_shared<avatar::Device>();
            avatar::AvatarSDK::devices[i]->side = (avatar::DeviceSide)i;
            valid(i, 1);
        }
        plugins::avatar::AvatarPluginConfig config;
        config.haptic = false;
        config.monitoring_plugin_root_id = "probe";
        tracker = std::make_unique<AvatarTracker>(config);
    }
    ~Rig()
    {
        tracker.reset();
        avatar::AvatarSDK::devices = {};
    }
    void valid(int side, int64_t stamp)
    {
        auto& frames = avatar::AvatarSDK::devices[side]->frames;
        for (int c = 0; c < 3; ++c)
        {
            auto& f = frames[c];
            f = {};
            f.which = c == 0 ? avatar::AvatarDataFrame::kSkeleton :
                      c == 1 ? avatar::AvatarDataFrame::kRaw :
                               avatar::AvatarDataFrame::kRobot;
            f.skeleton.header.stamp = { stamp, 0 };
            f.raw.header.stamp = { stamp, 0 };
            f.robot.header.stamp = { stamp, 0 };
            f.skeleton.landmark.resize(25);
            f.raw.joint.position.resize(22);
            f.robot.joint.position.resize(22);
        }
    }
    void pulse()
    {
        for (auto& d : avatar::AvatarSDK::devices)
            d->pulse();
    }
    void update()
    {
        pulse();
        tracker->update();
    }
    const plugin_utils::PluginDeviceStatusEntry& status(const std::string& path)
    {
        for (auto& s : plugin_utils::PluginDeviceStatusPublisher::latest)
            if (s.path == path)
                return s;
        throw std::runtime_error("missing status " + path);
    }
    void healthy(const std::string& path)
    {
        check(status(path).state == core::PluginDeviceState_CONNECTED, "expected connected dataset");
    }
    void no_data(const std::string& path)
    {
        check(status(path).state != core::PluginDeviceState_CONNECTED, "empty dataset remained connected");
        check(status(path).reason == core::PluginDeviceReason_NO_CURRENT_DATA,
              "empty dataset missing NO_CURRENT_DATA reason");
    }
    void allow_retry(int side, int category)
    {
        AvatarTrackerTestAccess::gloves(*tracker)[side].datasets[category].output = {};
    }
};
void startup_isolation()
{
    Rig r;
    tracker_test_double::fail.insert("avatar_raw_left");
    r.update();
    check(AvatarTrackerTestAccess::gloves(*r.tracker)[0].ready, "RAW setup failure blocked physical readiness");
    for (auto& d : avatar::AvatarSDK::devices)
        for (int n : d->fetches)
            check(n > 0, "output failure skipped SDK fetch");
    check(tracker_test_double::pushed["human_left"] > 0, "RAW setup failure blocked HUMAN output");
    check(tracker_test_double::pushed["avatar_robot_left"] > 0, "RAW setup failure blocked ROBOT output");
    check(tracker_test_double::pushed["human_right"] > 0, "left failure blocked right HUMAN output");
    check(tracker_test_double::pushed["avatar_raw_right"] > 0, "left failure blocked right RAW output");
    r.healthy("/hand/left");
    r.healthy("/glove/avatar_robot_left");
    check(r.status("/glove/avatar_raw_left").reason == core::PluginDeviceReason_DEVICE_ERROR,
          "failed RAW setup not exposed");
    tracker_test_double::fail.clear();
    r.allow_retry(0, 1);
    r.update();
    check(tracker_test_double::pushed["avatar_raw_left"] > 0, "RAW did not recover");
    r.healthy("/glove/avatar_raw_left");
}
void reconnect_isolation()
{
    Rig r;
    r.update();
    int right = tracker_test_double::pushed["human_right"];
    auto& left = AvatarTrackerTestAccess::gloves(*r.tracker)[0];
    left.last_ingress = std::chrono::steady_clock::now() - 4s;
    avatar::AvatarSDK::devices[1]->pulse();
    r.tracker->update();
    check(!left.ready, "stale physical ingress still ready");
    check(!AvatarTrackerTestAccess::injectors(*r.tracker)[0], "disconnect retained HUMAN output");
    check(tracker_test_double::pushed["human_right"] > right, "disconnect blocked other side");
    tracker_test_double::fail.insert("avatar_raw_left");
    r.valid(0, 2);
    r.update();
    check(left.ready, "RAW creation failure blocked reconnect readiness");
    check(tracker_test_double::created["human_left"] == 2, "HUMAN failed to recreate on reconnect");
    check(tracker_test_double::created["avatar_robot_left"] == 2, "ROBOT failed to recreate on reconnect");
    r.healthy("/hand/left");
    r.healthy("/glove/avatar_robot_left");
}
void invalid_payloads(bool wrong_case)
{
    Rig r;
    r.update();
    int right = tracker_test_double::pushed["human_right"];
    r.valid(0, 2);
    auto& frames = avatar::AvatarSDK::devices[0]->frames;
    for (auto& f : frames)
    {
        if (wrong_case)
            f.which = avatar::AvatarDataFrame::PAYLOAD_NOT_SET;
        else
        {
            f.skeleton.landmark.clear();
            f.raw.joint.position.clear();
            f.robot.joint.position.clear();
        }
    }
    r.update();
    check(!AvatarTrackerTestAccess::injectors(*r.tracker)[0], "unusable HUMAN retained injector");
    check(!AvatarTrackerTestAccess::pushers(*r.tracker)[0][0], "unusable RAW retained pusher");
    check(!AvatarTrackerTestAccess::pushers(*r.tracker)[0][1], "unusable ROBOT retained pusher");
    r.no_data("/hand/left");
    r.no_data("/glove/avatar_raw_left");
    r.no_data("/glove/avatar_robot_left");
    check(tracker_test_double::pushed["human_right"] > right, "invalid left payload blocked right side");
    // Rejected frames must not become timestamp baselines, even after freshness expires.
    for (auto& dataset : AvatarTrackerTestAccess::gloves(*r.tracker)[0].datasets)
        dataset.freshness.invalidate();
    r.valid(0, 2);
    r.update();
    check(tracker_test_double::created["human_left"] == 2, "valid HUMAN did not recover");
    check(tracker_test_double::created["avatar_raw_left"] == 2, "valid RAW did not recover");
    check(tracker_test_double::created["avatar_robot_left"] == 2, "valid ROBOT did not recover");
    r.healthy("/hand/left");
    r.healthy("/glove/avatar_raw_left");
    r.healthy("/glove/avatar_robot_left");
}
void cached_same_handle()
{
    Rig r;
    r.update();
    auto& left = AvatarTrackerTestAccess::gloves(*r.tracker)[0];
    left.last_ingress = std::chrono::steady_clock::now() - 4s;
    avatar::AvatarSDK::devices[1]->pulse();
    r.tracker->update();
    check(!left.ready, "stale ingress stayed ready");
    auto handle = left.device;
    r.update();
    check(left.device == handle, "same handle unexpectedly changed");
    check(left.ready, "fresh physical ingress did not recover");
    check(!AvatarTrackerTestAccess::injectors(*r.tracker)[0], "unchanged cached HUMAN resurrected after outage");
    check(!AvatarTrackerTestAccess::pushers(*r.tracker)[0][0], "unchanged cached RAW resurrected after outage");
    r.no_data("/hand/left");
    r.no_data("/glove/avatar_raw_left");
    auto& frames = avatar::AvatarSDK::devices[0]->frames;
    frames[1].raw.header.stamp = { 2, 0 };
    r.update();
    check(!AvatarTrackerTestAccess::injectors(*r.tracker)[0], "advancing RAW revived cached HUMAN");
    check(bool(AvatarTrackerTestAccess::pushers(*r.tracker)[0][0]), "fresh RAW did not recover independently");
    r.no_data("/hand/left");
    r.healthy("/glove/avatar_raw_left");
    frames[0].skeleton.header.stamp = { 2, 0 };
    r.update();
    check(bool(AvatarTrackerTestAccess::injectors(*r.tracker)[0]), "fresh HUMAN failed to recover");
    r.healthy("/hand/left");
}
void delivery_failures()
{
    for (auto key : { "human_left", "avatar_raw_left", "avatar_robot_left" })
    {
        Rig r;
        r.update();
        auto before = tracker_test_double::pushed;
        tracker_test_double::push_fail.insert(key);
        r.update();
        check(tracker_test_double::destroyed[key] > 0, "failed delivery kept output alive");
        check(tracker_test_double::pushed["human_right"] > before["human_right"], "delivery failure blocked other side");
        for (auto other : { "human_left", "avatar_raw_left", "avatar_robot_left" })
            if (std::string(other) != key)
                check(tracker_test_double::pushed[other] > before[other], "delivery failure blocked healthy dataset");
        std::string path = std::string(key) == "human_left"      ? "/hand/left" :
                           std::string(key) == "avatar_raw_left" ? "/glove/avatar_raw_left" :
                                                                   "/glove/avatar_robot_left";
        check(r.status(path).reason == core::PluginDeviceReason_DEVICE_ERROR, "failed delivery missing error health");
        tracker_test_double::push_fail.clear();
        r.allow_retry(0, std::string(key) == "human_left" ? 0 : std::string(key) == "avatar_raw_left" ? 1 : 2);
        r.update();
        check(tracker_test_double::pushed[key] > before[key], "failed delivery did not recover");
        r.healthy(path);
    }
}

} // namespace
TEST_CASE("Tracker startup isolates output creation failures", "[sharpa_avatar][integration]")
{
    startup_isolation();
}
TEST_CASE("Tracker reconnect isolates output creation failures", "[sharpa_avatar][integration]")
{
    reconnect_isolation();
}
TEST_CASE("Tracker health and delivery reject empty payloads", "[sharpa_avatar][integration]")
{
    invalid_payloads(false);
}
TEST_CASE("Tracker health and delivery reject wrong payload cases", "[sharpa_avatar][integration]")
{
    invalid_payloads(true);
}
TEST_CASE("Tracker retains timestamp baseline across same-handle outages", "[sharpa_avatar][integration]")
{
    cached_same_handle();
}
TEST_CASE("Tracker isolates output delivery failures", "[sharpa_avatar][integration]")
{
    delivery_failures();
}

TEST_CASE("Tracker startup rejects empty payloads before creating outputs", "[sharpa_avatar][integration]")
{
    Rig r;
    for (auto& frame : avatar::AvatarSDK::devices[0]->frames)
    {
        frame.skeleton.landmark.clear();
        frame.raw.joint.position.clear();
        frame.robot.joint.position.clear();
    }
    r.update();
    r.no_data("/hand/left");
    r.no_data("/glove/avatar_raw_left");
    r.no_data("/glove/avatar_robot_left");
    check(!AvatarTrackerTestAccess::injectors(*r.tracker)[0], "empty startup created a HUMAN injector");
    check(!AvatarTrackerTestAccess::pushers(*r.tracker)[0][0], "empty startup created a RAW pusher");
    check(!AvatarTrackerTestAccess::pushers(*r.tracker)[0][1], "empty startup created a ROBOT pusher");
    r.valid(0, 1);
    r.update();
    r.healthy("/hand/left");
    r.healthy("/glove/avatar_raw_left");
    r.healthy("/glove/avatar_robot_left");
}
