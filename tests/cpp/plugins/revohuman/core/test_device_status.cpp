// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <revohuman_plugin/device_status.hpp>

#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace plugins::revohuman;

namespace
{

// Mirrors _VALID_PLUGIN_DEVICE_STATE_REASON_PAIRS in
// src/python/isaaccapture/teleop_session_manager/status_monitor.py; any other pair makes
// TeleopSession reject the whole status report.
const std::set<std::pair<core::PluginDeviceState, core::PluginDeviceReason>> kAcceptedPairs = {
    { core::PluginDeviceState_UNKNOWN, core::PluginDeviceReason_NO_HARDWARE_SIGNAL },
    { core::PluginDeviceState_UNKNOWN, core::PluginDeviceReason_NO_CURRENT_DATA },
    { core::PluginDeviceState_CONNECTED, core::PluginDeviceReason_NONE },
    { core::PluginDeviceState_DISCONNECTED, core::PluginDeviceReason_NO_HARDWARE_SIGNAL },
    { core::PluginDeviceState_DEGRADED, core::PluginDeviceReason_CALIBRATION_FAILED },
    { core::PluginDeviceState_DEGRADED, core::PluginDeviceReason_NO_CURRENT_DATA },
    { core::PluginDeviceState_DEGRADED, core::PluginDeviceReason_DEVICE_ERROR },
    { core::PluginDeviceState_FAILED, core::PluginDeviceReason_DEVICE_ERROR },
    { core::PluginDeviceState_DISABLED, core::PluginDeviceReason_DISABLED_BY_CONFIGURATION },
};

enum Row
{
    kHandRow = 0,
    kGloveRow = 1,
    kTactileRow = 2,
};

HandStatusInput streaming()
{
    HandStatusInput input;
    input.health.hand_active = true;
    input.health.wrist_tracked = true;
    input.health.glove_live = true;
    input.health.glove_seen = true;
    input.health.tactile_live = true;
    input.health.tactile_seen = true;
    return input;
}

//! Glove and tactile have both been seen and are silent now.
HandStatusInput lost()
{
    HandStatusInput input;
    input.health.glove_seen = true;
    input.health.tactile_seen = true;
    return input;
}

std::vector<std::optional<Source::GloveNote>> all_notes()
{
    return { std::nullopt, Source::GloveNote{ Source::GloveNote::Kind::Waiting, "waiting" },
             Source::GloveNote{ Source::GloveNote::Kind::NotReady, "gate closed" },
             Source::GloveNote{ Source::GloveNote::Kind::Ended, "ended" } };
}

} // namespace

TEST_CASE("every health maps onto a state/reason pair TeleopSession accepts", "[revohuman][status]")
{
    const std::vector<TactileSkip> skips = {
        TactileSkip::None,         TactileSkip::NoShape,      TactileSkip::NoTactile,
        TactileSkip::Invalid,      TactileSkip::Uncalibrated, TactileSkip::CountMismatch,
        TactileSkip::NoSampleTime, TactileSkip::StaleSample,  TactileSkip::NonMonotonicSample
    };
    std::size_t checked = 0;
    for (const Hand hand : { Hand::Left, Hand::Right })
    {
        for (int bits = 0; bits < (1 << 7); ++bits)
        {
            for (const TactileSkip skip : skips)
            {
                for (const auto& note : all_notes())
                {
                    HandStatusInput input;
                    input.health.hand_active = (bits & 0x01) != 0;
                    input.health.wrist_tracked = (bits & 0x02) != 0;
                    input.health.kinematics_missing = (bits & 0x04) != 0;
                    input.health.glove_live = (bits & 0x08) != 0;
                    input.health.glove_seen = (bits & 0x10) != 0;
                    input.health.tactile_live = (bits & 0x20) != 0;
                    input.health.tactile_seen = (bits & 0x40) != 0;
                    input.health.tactile_skip = skip;
                    input.note = note;
                    for (const auto& row : make_hand_status(hand, input))
                    {
                        INFO(row.path << " bits=" << bits << " skip=" << tactile_skip_name(skip));
                        CHECK(kAcceptedPairs.count({ row.state, row.reason }) == 1);
                        ++checked;
                    }
                }
            }
        }
    }
    CHECK(checked == 2 * 128 * 9 * 4 * 3);
}

TEST_CASE("rows carry the device paths of plugin.yaml", "[revohuman][status]")
{
    const auto left = make_hand_status(Hand::Left, streaming());
    CHECK(left[kHandRow].path == "/hand/left");
    CHECK(left[kGloveRow].path == "/glove/revohuman_left");
    CHECK(left[kTactileRow].path == "/tactile/revohuman_left");

    const auto right = make_hand_status(Hand::Right, streaming());
    CHECK(right[kHandRow].path == "/hand/right");
    CHECK(right[kGloveRow].path == "/glove/revohuman_right");
    CHECK(right[kTactileRow].path == "/tactile/revohuman_right");

    for (const auto& row : left)
    {
        CHECK(row.state == core::PluginDeviceState_CONNECTED);
        CHECK(row.reason == core::PluginDeviceReason_NONE);
    }
}

TEST_CASE("a device that never gave data is unknown, one that stopped is disconnected", "[revohuman][status]")
{
    const auto waiting = make_hand_status(Hand::Right, HandStatusInput{});
    for (const auto& row : waiting)
    {
        INFO(row.path);
        CHECK(row.state == core::PluginDeviceState_UNKNOWN);
        CHECK(row.reason == core::PluginDeviceReason_NO_HARDWARE_SIGNAL);
    }

    const auto stopped = make_hand_status(Hand::Right, lost());
    for (const auto& row : stopped)
    {
        INFO(row.path);
        CHECK(row.state == core::PluginDeviceState_DISCONNECTED);
        CHECK(row.reason == core::PluginDeviceReason_NO_HARDWARE_SIGNAL);
    }
}

TEST_CASE("what the source says about a glove overrides its last sample", "[revohuman][status]")
{
    HandStatusInput input = streaming();

    input.note = Source::GloveNote{ Source::GloveNote::Kind::Ended, "glove stream ended: unplugged" };
    auto rows = make_hand_status(Hand::Left, input);
    CHECK(rows[kGloveRow].state == core::PluginDeviceState_FAILED);
    CHECK(rows[kGloveRow].reason == core::PluginDeviceReason_DEVICE_ERROR);
    CHECK(rows[kGloveRow].error == "glove stream ended: unplugged");

    input.note = Source::GloveNote{ Source::GloveNote::Kind::NotReady, "gate closed" };
    rows = make_hand_status(Hand::Left, input);
    CHECK(rows[kGloveRow].state == core::PluginDeviceState_DEGRADED);
    CHECK(rows[kGloveRow].reason == core::PluginDeviceReason_NO_CURRENT_DATA);

    // Before the first sample a closed gate is still "waiting", not a degraded glove.
    input = HandStatusInput{};
    input.note = Source::GloveNote{ Source::GloveNote::Kind::NotReady, "gate closed" };
    rows = make_hand_status(Hand::Left, input);
    CHECK(rows[kGloveRow].state == core::PluginDeviceState_UNKNOWN);
    CHECK(rows[kGloveRow].reason == core::PluginDeviceReason_NO_CURRENT_DATA);

    input.note = Source::GloveNote{ Source::GloveNote::Kind::Waiting, "waiting for the glove: BUSY" };
    rows = make_hand_status(Hand::Left, input);
    CHECK(rows[kGloveRow].state == core::PluginDeviceState_UNKNOWN);
    CHECK(rows[kGloveRow].reason == core::PluginDeviceReason_NO_HARDWARE_SIGNAL);
    CHECK(rows[kGloveRow].error == "waiting for the glove: BUSY");

    // Looked for again after it streamed: unplugged, so disconnected.
    input = lost();
    input.note = Source::GloveNote{ Source::GloveNote::Kind::Waiting, "waiting for the glove: unplugged" };
    rows = make_hand_status(Hand::Left, input);
    CHECK(rows[kGloveRow].state == core::PluginDeviceState_DISCONNECTED);
    CHECK(rows[kGloveRow].reason == core::PluginDeviceReason_NO_HARDWARE_SIGNAL);
    CHECK(rows[kGloveRow].error == "waiting for the glove: unplugged");
}

TEST_CASE("the hand row says why no tracked hand is injected", "[revohuman][status]")
{
    HandStatusInput input = streaming();
    input.health.wrist_tracked = false;
    auto rows = make_hand_status(Hand::Left, input);
    CHECK(rows[kHandRow].state == core::PluginDeviceState_DEGRADED);
    CHECK(rows[kHandRow].reason == core::PluginDeviceReason_NO_CURRENT_DATA);
    CHECK(rows[kHandRow].error == "no tracked wrist pose from the host");

    input.health.hand_active = false;
    input.health.kinematics_missing = true;
    rows = make_hand_status(Hand::Left, input);
    CHECK(rows[kHandRow].state == core::PluginDeviceState_DEGRADED);
    CHECK(rows[kHandRow].error == "glove samples carry no joint poses");
}

TEST_CASE("tactile reports why it is not pushed while the glove is live", "[revohuman][status]")
{
    HandStatusInput input = streaming();
    input.health.tactile_live = false;

    input.health.tactile_skip = TactileSkip::NoShape;
    CHECK(make_hand_status(Hand::Left, input)[kTactileRow].state == core::PluginDeviceState_DISABLED);
    CHECK(make_hand_status(Hand::Left, input)[kTactileRow].error.find("taxel layout") != std::string::npos);

    // A P1 glove has no tactile sensors: tactile is off, not broken, whether or not the glove is live.
    input.health.tactile_skip = TactileSkip::NoTactile;
    for (const bool live : { true, false })
    {
        input.health.glove_live = live;
        const auto off = make_hand_status(Hand::Left, input)[kTactileRow];
        CHECK(off.state == core::PluginDeviceState_DISABLED);
        CHECK(off.reason == core::PluginDeviceReason_DISABLED_BY_CONFIGURATION);
        CHECK(off.error == "this glove has no tactile sensors");
    }
    input.health.glove_live = true;

    input.health.tactile_skip = TactileSkip::Uncalibrated;
    auto row = make_hand_status(Hand::Left, input)[kTactileRow];
    CHECK(row.state == core::PluginDeviceState_DEGRADED);
    CHECK(row.reason == core::PluginDeviceReason_CALIBRATION_FAILED);

    input.health.tactile_skip = TactileSkip::CountMismatch;
    row = make_hand_status(Hand::Left, input)[kTactileRow];
    CHECK(row.state == core::PluginDeviceState_FAILED);
    CHECK(row.reason == core::PluginDeviceReason_DEVICE_ERROR);

    input.health.tactile_skip = TactileSkip::StaleSample;
    row = make_hand_status(Hand::Left, input)[kTactileRow];
    CHECK(row.state == core::PluginDeviceState_DEGRADED);
    CHECK(row.reason == core::PluginDeviceReason_NO_CURRENT_DATA);

    // Without a live glove the last skip reason is stale: tactile is simply gone.
    input.health.glove_live = false;
    input.health.tactile_skip = TactileSkip::CountMismatch;
    row = make_hand_status(Hand::Left, input)[kTactileRow];
    CHECK(row.state == core::PluginDeviceState_DISCONNECTED);
    CHECK(row.reason == core::PluginDeviceReason_NO_HARDWARE_SIGNAL);
}
