// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the generated SteeringWheel FlatBuffer types.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <schema/steering_wheel_generated.h>

#define VT(field) (field + 2) * 2

static_assert(core::SteeringWheelOutput::VT_STEERING == VT(0));
static_assert(core::SteeringWheelOutput::VT_THROTTLE == VT(1));
static_assert(core::SteeringWheelOutput::VT_BRAKE == VT(2));
static_assert(core::SteeringWheelOutput::VT_CLUTCH == VT(3));
static_assert(core::SteeringWheelOutput::VT_BUTTONS == VT(4));
static_assert(core::SteeringWheelOutput::VT_HAT_X == VT(5));
static_assert(core::SteeringWheelOutput::VT_HAT_Y == VT(6));

static_assert(core::SteeringWheelOutputRecord::VT_DATA == VT(0));
static_assert(core::SteeringWheelOutputRecord::VT_TIMESTAMP == VT(1));

TEST_CASE("SteeringWheelOutputT default construction", "[steering_wheel][native]")
{
    core::SteeringWheelOutputT output;

    CHECK(output.steering == 0.0f);
    CHECK(output.throttle == 0.0f);
    CHECK(output.brake == 0.0f);
    CHECK(output.clutch == 0.0f);
    CHECK(output.buttons.empty());
    CHECK(output.hat_x == 0);
    CHECK(output.hat_y == 0);
}

TEST_CASE("SteeringWheelOutput serialization and deserialization", "[steering_wheel][serialize]")
{
    flatbuffers::FlatBufferBuilder builder;

    core::SteeringWheelOutputT output;
    output.steering = -0.25f;
    output.throttle = 0.8f;
    output.brake = 0.1f;
    output.clutch = 0.0f;
    output.buttons = { 1, 0, 1 };
    output.hat_x = -1;
    output.hat_y = 1;

    auto offset = core::SteeringWheelOutput::Pack(builder, &output);
    builder.Finish(offset);

    const auto* deserialized = flatbuffers::GetRoot<core::SteeringWheelOutput>(builder.GetBufferPointer());
    CHECK(deserialized->steering() == Catch::Approx(-0.25f));
    CHECK(deserialized->throttle() == Catch::Approx(0.8f));
    CHECK(deserialized->brake() == Catch::Approx(0.1f));
    CHECK(deserialized->buttons()->size() == 3);
    CHECK(deserialized->buttons()->Get(0) == 1);
    CHECK(deserialized->hat_x() == -1);
    CHECK(deserialized->hat_y() == 1);
}

TEST_CASE("SteeringWheelOutputTrackedT defaults to null data", "[steering_wheel][tracked]")
{
    core::SteeringWheelOutputTrackedT tracked;

    CHECK(tracked.data == nullptr);
}
