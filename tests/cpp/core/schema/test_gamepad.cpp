// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the generated Gamepad FlatBuffer types.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <schema/gamepad_generated.h>

#include <memory>

#define VT(field) (field + 2) * 2

static_assert(core::GamepadSnapshot::VT_INPUTS == VT(0));

static_assert(core::GamepadSnapshotRecord::VT_DATA == VT(0));
static_assert(core::GamepadSnapshotRecord::VT_TIMESTAMP == VT(1));

// 14 bools padded to the floats' alignment, then 6 floats.
static_assert(sizeof(core::GamepadInputState) == 16 + 6 * 4);

TEST_CASE("GamepadSnapshotT default construction", "[gamepad][native]")
{
    core::GamepadSnapshotT snapshot;

    CHECK(snapshot.inputs == nullptr);
}

TEST_CASE("GamepadSnapshot serialization and deserialization", "[gamepad][serialize]")
{
    flatbuffers::FlatBufferBuilder builder;

    core::GamepadSnapshotT snapshot;
    core::GamepadInputState inputs;
    inputs.mutate_west_click(true);
    inputs.mutate_dpad_up_click(true);
    inputs.mutate_thumbstick_left_y(1.0f);
    inputs.mutate_trigger_right_value(0.5f);
    snapshot.inputs = std::make_shared<core::GamepadInputState>(inputs);

    builder.Finish(core::GamepadSnapshot::Pack(builder, &snapshot));

    const auto* deserialized = flatbuffers::GetRoot<core::GamepadSnapshot>(builder.GetBufferPointer());
    REQUIRE(deserialized->inputs() != nullptr);
    CHECK(deserialized->inputs()->west_click());
    CHECK(deserialized->inputs()->dpad_up_click());
    CHECK_FALSE(deserialized->inputs()->south_click());
    CHECK(deserialized->inputs()->thumbstick_left_y() == Catch::Approx(1.0f));
    CHECK(deserialized->inputs()->trigger_right_value() == Catch::Approx(0.5f));
}
