// SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <oxr_utils/math.hpp>
#include <realhand_ffg_glove/realhand_ffg_glove_plugin.hpp>
#include <realhand_ffg_glove/realhand_ffg_glove_skeleton.hpp>
#include <realhand_ffg_test_support/calibration.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numbers>
#include <unistd.h>

#include <yaml-cpp/yaml.h>

using namespace plugins::realhand_ffg_glove;
using namespace realhand_ffg_test_support;

namespace
{
void check(bool value, const char* message)
{
    INFO(message);
    REQUIRE(value);
}

float distance(const XrVector3f& a, const XrVector3f& b)
{
    return std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z));
}

bool same(const XrHandJointLocationEXT& a, const XrHandJointLocationEXT& b)
{
    return distance(a.pose.position, b.pose.position) < 1e-6f &&
           std::abs(a.pose.orientation.x - b.pose.orientation.x) < 1e-6f &&
           std::abs(a.pose.orientation.y - b.pose.orientation.y) < 1e-6f &&
           std::abs(a.pose.orientation.z - b.pose.orientation.z) < 1e-6f &&
           std::abs(a.pose.orientation.w - b.pose.orientation.w) < 1e-6f;
}

void expect_invalid(const YAML::Node& data, const CalibrationFile& file)
{
    file.save(data);
    bool rejected = false;
    try
    {
        HumanHandSkeleton model(file.path.string());
    }
    catch (const std::exception&)
    {
        rejected = true;
    }
    check(rejected, "reject invalid calibration");
}
} // namespace

TEST_CASE("Human skeleton geometry, calibration, and wrist continuity", "[realhand_ffg_skeleton]")
{
    {
        RealHandFFGGloveConfig config;
        const auto& left = config.wrist.left_aim_to_wrist;
        const auto& right = config.wrist.right_aim_to_wrist;
        check(config.wrist.mode == plugin_utils::WristSourceMode::Auto, "mount defaults preserve wrist source selection");
        check(distance(left.position, { 0.020534770647f, -0.047368722071f, 0.192378854628f }) < 1e-7f,
              "default PICO bracket translation");
        check(distance(right.position, { -left.position.x, left.position.y, left.position.z }) < 1e-7f,
              "default bracket positions are mirrored");
        const auto& q = left.orientation;
        check(std::abs(q.x + 0.422424918198f) < 1e-7f && std::abs(q.y - 0.014580024139f) < 1e-7f &&
                  std::abs(q.z - 0.024960358040f) < 1e-7f && std::abs(q.w - 0.905936858676f) < 1e-7f,
              "default PICO bracket orientation");
        check(std::abs(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w - 1.0f) < 1e-6f,
              "default bracket rotation is normalized");
        for (const auto point : { XrVector3f{ 0, 0, 0 }, XrVector3f{ 0.03f, 0.04f, -0.15f } })
        {
            const auto a = oxr_utils::multiply_poses(left, { { 0, 0, 0, 1 }, point });
            const auto b = oxr_utils::multiply_poses(right, { { 0, 0, 0, 1 }, { -point.x, point.y, point.z } });
            check(distance(b.position, { -a.position.x, a.position.y, a.position.z }) < 1e-6f,
                  "default bracket rotations preserve mirrored hand geometry");
        }
        config.wrist.left_aim_to_wrist = { { 0, 0, 0, 1 }, { 0, 0, 0 } };
        check(config.wrist.left_aim_to_wrist.position.z == 0 && config.wrist.right_aim_to_wrist.position.z > 0,
              "each bracket offset remains independently overridable");
    }
    {
        WristPoseContinuity left, right;
        XrPosef pose{ { 0, 0, 0, 1 }, { 0, 0, 0 } };
        check(left.accept({ pose, true, true }, 1'000'000'000), "initial valid wrist accepted");
        check(right.accept({ pose, true, true }, 1'000'000'000), "independent right wrist accepted");
        pose.position.x = 0.243f;
        check(!left.accept({ pose, true, true }, 1'012'000'000), "reject recorded-scale 24 cm controller teleport");
        check(right.accept({ { { 0, 0, 0, -1 }, { 0, 0, 0 } }, true, true }, 1'012'000'000),
              "left dropout and quaternion sign do not affect the right hand");
        check(!left.accept({ pose, true, true }, 1'024'000'000), "first stable sample remains inactive");
        check(!left.accept({ pose, true, true }, 1'036'000'000), "second stable sample remains inactive");
        check(left.accept({ pose, true, true }, 1'048'000'000), "reacquire without changing the wrist pose");
        check(!left.accept({ pose, false, false }, 1'060'000'000),
              "missing controller does not publish a local-origin hand");
        check(!left.accept({ pose, true, true }, 1'072'000'000), "tracking loss requires reacquisition");
        check(!left.accept({ pose, true, true }, 1'072'000'000), "nonmonotonic timestamps rejected");
        pose.position.x = std::numeric_limits<float>::quiet_NaN();
        check(!left.accept({ pose, true, true }, 1'084'000'000), "nonfinite wrist rejected");
        WristPoseContinuity fast;
        pose = { { 0, 0, 0, 1 }, { 0, 0, 0 } };
        check(fast.accept({ pose, true, true }, 2'000'000'000), "fast motion baseline");
        pose.position.x = 0.04f;
        check(fast.accept({ pose, true, true }, 2'012'000'000), "ordinary fast motion passes unchanged");
        pose.orientation = { 1, 0, 0, 0 };
        check(!fast.accept({ pose, true, true }, 2'024'000'000), "controller orientation teleport rejected");
    }
    CalibrationFile file;
    file.save(calibration());
    const HumanHandSkeleton model(file.path.string());
    const HumanHandSkeleton large(file.path.string(), 1.5f);
    std::array<float, kNumSensors> sensors{};
    const auto open = *model.reconstruct(sensors, HandSide::Left);
    check(open.size() == 26, "all OpenXR joints populated");
    for (std::size_t base : { 6u, 11u, 16u, 21u })
    {
        const auto& root = open[base].pose.position;
        const auto& knuckle = open[base + 1].pose.position;
        const float fraction = (root.x * knuckle.x + root.y * knuckle.y + root.z * knuckle.z) /
                               (knuckle.x * knuckle.x + knuckle.y * knuckle.y + knuckle.z * knuckle.z);
        const XrVector3f projected{ fraction * knuckle.x, fraction * knuckle.y, fraction * knuckle.z };
        check(fraction > 0.0f && fraction < 1.0f && distance(root, projected) < 1e-6f,
              "neutral metacarpal root lies between the wrist and its knuckle");
    }
    for (float span : { -0.2f, 0.2f, 0.0f })
    {
        auto data = calibration(6.2f);
        for (const auto* name : { "left", "right" })
        {
            for (std::size_t sensor = 5; sensor < kNumSensors; sensor += 4)
            {
                data["hands"][name]["spread"][sensor] = 6.2f + span;
            }
        }
        file.save(data);
        const HumanHandSkeleton lateral_model(file.path.string());
        for (const auto side : { HandSide::Left, HandSide::Right })
        {
            sensors.fill(6.2f);
            const auto neutral = *lateral_model.reconstruct(sensors, side);
            const float mirror = side == HandSide::Left ? 1.0f : -1.0f;
            for (std::size_t finger = 0; finger < 4; ++finger)
            {
                const auto sensor = 5 + 4 * finger;
                const auto tip = 10 + 5 * finger;
                float last_x = 1.0f;
                for (int step = -10; step <= 10; ++step)
                {
                    sensors.fill(6.2f);
                    const float delta = step * 0.02f;
                    // Equivalent anatomical motion has opposite raw lateral deltas on the two gloves.
                    sensors[sensor] = std::remainder(6.2f + mirror * delta, 2.0f * std::numbers::pi_v<float>);
                    const auto moved = *lateral_model.reconstruct(sensors, side);
                    const float x = mirror * (moved[tip].pose.position.x - neutral[tip].pose.position.x);
                    check(step == 0 ? std::abs(x) < 1e-6f : x * delta < 0.0f,
                          "side-normalized lateral delta moves toward little-finger side for either hand");
                    check(x <= last_x + 1e-6f, "lateral sweep is monotonic through zero and phase wrap");
                    last_x = x;
                    check(std::abs(moved[tip].pose.position.y - neutral[tip].pose.position.y) < 1e-6f,
                          "sideways motion does not introduce flexion");
                    for (std::size_t joint = 0; joint < moved.size(); ++joint)
                    {
                        if (joint < tip - 4 || joint > tip)
                        {
                            check(same(neutral[joint], moved[joint]), "lateral input never moves another finger");
                        }
                    }
                }
            }
        }
    }
    for (int step = 0; step <= 20; ++step)
    {
        sensors.fill(step / 20.0f);
        const auto left = *model.reconstruct(sensors, HandSide::Left);
        auto right_sensors = sensors;
        for (std::size_t sensor = 5; sensor < kNumSensors; sensor += 4)
        {
            right_sensors[sensor] = -sensors[sensor];
        }
        const auto right = *model.reconstruct(right_sensors, HandSide::Right);
        const auto scaled = *large.reconstruct(sensors, HandSide::Left);
        for (std::size_t joint = 0; joint < left.size(); ++joint)
        {
            const auto& p = left[joint].pose.position;
            const auto& q = left[joint].pose.orientation;
            check(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z), "finite positions");
            check(std::abs(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w - 1.0f) < 1e-5f, "unit rotations");
            check(left[joint].locationFlags ==
                      (XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT),
                  "estimated skeleton is valid but not independently tracked");
            check(left[joint].radius > 0.0f, "positive radii");
            check(std::abs(p.x + right[joint].pose.position.x) < 1e-6f &&
                      std::abs(p.y - right[joint].pose.position.y) < 1e-6f &&
                      std::abs(p.z - right[joint].pose.position.z) < 1e-6f,
                  "symmetric left/right anatomy");
            check(std::abs(q.x - right[joint].pose.orientation.x) < 1e-6f &&
                      std::abs(q.y + right[joint].pose.orientation.y) < 1e-6f &&
                      std::abs(q.z + right[joint].pose.orientation.z) < 1e-6f &&
                      std::abs(q.w - right[joint].pose.orientation.w) < 1e-6f,
                  "mirrored rotations");
            check(distance(scaled[joint].pose.position, { p.x * 1.5f, p.y * 1.5f, p.z * 1.5f }) < 1e-6f,
                  "hand size scales geometry consistently");
        }
        for (std::size_t base : { 2u, 6u, 11u, 16u, 21u })
        {
            const auto end = base == 2 ? 5u : base + 4;
            if (base != 2)
            {
                const float length = distance(left[base].pose.position, left[base + 1].pose.position);
                const auto knuckle =
                    oxr_utils::multiply_poses(left[base].pose, { { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, -length } });
                check(distance(knuckle.position, left[base + 1].pose.position) < 1e-6f,
                      "metacarpal minus-Z axis points along its bone toward the knuckle");
            }
            for (std::size_t joint = base; joint < end; ++joint)
            {
                check(std::abs(distance(left[joint].pose.position, left[joint + 1].pose.position) -
                               distance(open[joint].pose.position, open[joint + 1].pose.position)) < 1e-6f,
                      "bone lengths remain rigid across flexion");
            }
        }
    }
    for (std::size_t sensor = 0; sensor < kNumSensors; ++sensor)
    {
        sensors.fill(0.0f);
        sensors[sensor] = 0.7f;
        const auto moved = *model.reconstruct(sensors, HandSide::Left);
        const std::size_t digit = sensor < 5 ? 0 : 1 + (sensor - 5) / 4;
        const std::size_t first = digit == 0 ? 2 : 6 + (digit - 1) * 5;
        const std::size_t last = digit == 0 ? 5 : first + 4;
        for (std::size_t joint = 0; joint < moved.size(); ++joint)
        {
            if (joint < first || joint > last)
            {
                check(same(open[joint], moved[joint]), "each sensor affects only its own digit");
            }
        }
        const bool unused = sensor == 3 || (sensor >= 5 && (sensor - 5) % 4 == 2);
        if (!unused)
        {
            check(!same(open[last], moved[last]), "every supported sensor has motion response");
        }
    }
    for (float sign : { -1.0f, 1.0f })
    {
        file.save(calibration(6.0f, sign * 2.7f));
        const HumanHandSkeleton wrapped(file.path.string());
        sensors.fill(6.0f + sign * 2.7f);
        for (std::size_t sensor = 5; sensor < kNumSensors; sensor += 4)
        {
            sensors[sensor] = 6.0f;
        }
        const auto boundary = *wrapped.reconstruct(sensors, HandSide::Left);
        for (float overshoot : { 0.01f, 0.2f, 0.5f })
        {
            sensors.fill(std::remainder(6.0f + sign * (2.7f + overshoot), 2.0f * std::numbers::pi_v<float>));
            for (std::size_t sensor = 5; sensor < kNumSensors; sensor += 4)
            {
                sensors[sensor] = 6.0f;
            }
            const auto saturated = *wrapped.reconstruct(sensors, HandSide::Left);
            for (std::size_t joint = 2; joint <= 5; ++joint)
            {
                check(same(boundary[joint], saturated[joint]), "thumb stays curled beyond wrapped endpoint");
            }
            for (std::size_t finger = 0; finger < 4; ++finger)
            {
                const auto joint = 6 + finger * 5;
                check(std::abs(boundary[joint + 1].pose.orientation.x - saturated[joint + 1].pose.orientation.x) < 1e-5f,
                      "MCP saturation across angular wrap");
            }
        }
    }
    const XrPosef wrist{ { 0.0f, 0.0f, 1.0f, 0.0f }, { 1.0f, 2.0f, 3.0f } };
    const auto placed = place_hand(open, wrist, true, true);
    const auto cached = place_hand(open, wrist, true, false);
    const auto absent = place_hand(placed, wrist, false, true);
    for (std::size_t joint = 0; joint < open.size(); ++joint)
    {
        const auto p = open[joint].pose.position;
        check(distance(placed[joint].pose.position, { 1.0f - p.x, 2.0f - p.y, 3.0f + p.z }) < 1e-6f,
              "global wrist rigid transform");
        check((placed[joint].locationFlags & XR_SPACE_LOCATION_POSITION_TRACKED_BIT) != 0, "live wrist tracked");
        check(cached[joint].locationFlags == open[joint].locationFlags, "cached wrist not tracked");
        check(absent[joint].locationFlags == open[joint].locationFlags, "missing wrist clears stale tracked flags");
    }
    auto data = calibration();
    data["hands"].remove("right");
    file.save(data);
    const HumanHandSkeleton left_only(file.path.string());
    sensors.fill(0.0f);
    check(!left_only.reconstruct(sensors, HandSide::Right), "never borrow opposite side calibration");
    check(!HumanHandSkeleton().reconstruct(sensors, HandSide::Left), "uncalibrated hands inactive");
    sensors[0] = std::numeric_limits<float>::quiet_NaN();
    check(!model.reconstruct(sensors, HandSide::Left), "nonfinite input does not publish a hand");
    data = calibration();
    data["format"] = "realhand-ffg-glove-raw-calibration-v2";
    expect_invalid(data, file);
    data = calibration();
    data["hands"]["left"].remove("thumb_opposition");
    expect_invalid(data, file);
    data = calibration();
    data["hands"]["left"]["fist"][6] = 0.0f;
    expect_invalid(data, file);
    data = calibration();
    data["hands"]["right"]["open"][0] = std::numeric_limits<float>::infinity();
    expect_invalid(data, file);
    data = calibration();
    data["hands"]["left"]["thumb_opposition"][0] = 0.049f;
    expect_invalid(data, file);
    auto mapping = YAML::Load(R"(
format: realhand-ffg-thumb-quadratic-v1
input_origin: [0, 0, 0, 0]
input_scale: [1, 1, 1, 1]
input_min: [0, 0, 0, 0]
input_max: [0.049, 1, 1, 1]
base_position: [0.025, -0.005, -0.020]
lengths: [0.035, 0.030, 0.026]
coefficients:
  - [-0.85, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  - [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  - [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  - [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
)");
    data["hands"]["left"]["thumb_mapping"] = mapping;
    file.save(data);
    const HumanHandSkeleton multivariate(file.path.string());
    sensors.fill(0.0f);
    check(multivariate.reconstruct(sensors, HandSide::Left).has_value(),
          "multivariate thumb fit does not require a legacy single-pose span");
    for (std::size_t sensor = 0; sensor < 4; ++sensor)
    {
        data["hands"]["left"]["thumb_mapping"]["input_max"][sensor] = 0.0f;
    }
    expect_invalid(data, file);
    std::printf("OK: 26-joint geometry, independence, wrap, placement, and calibration checks passed\n");
}


TEST_CASE("Cached controller wrist is inactive until stable tracked samples return", "[realhand_ffg_skeleton]")
{
    constexpr std::int64_t frame_time_ns = 11'111'111;
    WristPoseContinuity continuity;
    plugin_utils::WristSample wrist{ .valid = true, .tracked = true };
    REQUIRE(continuity.accept(wrist, 1'000'000'000));
    wrist.tracked = false;
    for (std::int64_t frame = 1; frame <= 270; ++frame)
        REQUIRE_FALSE(continuity.accept(wrist, 1'000'000'000 + frame * frame_time_ns));
    wrist.tracked = true;
    for (std::int64_t frame = 271; frame <= 273; ++frame)
        REQUIRE_FALSE(continuity.accept(wrist, 1'000'000'000 + frame * frame_time_ns));
    REQUIRE(continuity.accept(wrist, 1'000'000'000 + 274 * frame_time_ns));
}
