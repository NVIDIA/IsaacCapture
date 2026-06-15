// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the generated VehicleControl FlatBuffer types.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <schema/vehicle_control_generated.h>

#define VT(field) (field + 2) * 2

static_assert(core::VehicleControlCommand::VT_SEQUENCE == VT(0));
static_assert(core::VehicleControlCommand::VT_STEER == VT(1));
static_assert(core::VehicleControlCommand::VT_ACCEL == VT(2));
static_assert(core::VehicleControlCommand::VT_THROTTLE == VT(3));
static_assert(core::VehicleControlCommand::VT_BRAKE == VT(4));

static_assert(core::VehicleControlCommandRecord::VT_DATA == VT(0));
static_assert(core::VehicleControlCommandRecord::VT_TIMESTAMP == VT(1));

TEST_CASE("VehicleControlCommandT default construction", "[vehicle_control][native]")
{
    core::VehicleControlCommandT command;

    CHECK(command.sequence == 0);
    CHECK(command.steer == 0.0f);
    CHECK(command.accel == 0.0f);
    CHECK(command.throttle == 0.0f);
    CHECK(command.brake == 0.0f);
}

TEST_CASE("VehicleControlCommand serialization and deserialization", "[vehicle_control][serialize]")
{
    flatbuffers::FlatBufferBuilder builder;

    core::VehicleControlCommandT command;
    command.sequence = 42;
    command.steer = -0.5f;
    command.accel = 0.75f;
    command.throttle = 0.75f;
    command.brake = 0.0f;

    auto offset = core::VehicleControlCommand::Pack(builder, &command);
    builder.Finish(offset);

    const auto* deserialized = flatbuffers::GetRoot<core::VehicleControlCommand>(builder.GetBufferPointer());
    CHECK(deserialized->sequence() == 42);
    CHECK(deserialized->steer() == Catch::Approx(-0.5f));
    CHECK(deserialized->accel() == Catch::Approx(0.75f));
    CHECK(deserialized->throttle() == Catch::Approx(0.75f));
    CHECK(deserialized->brake() == Catch::Approx(0.0f));
}

TEST_CASE("VehicleControlCommandRecord can omit data", "[vehicle_control][record]")
{
    flatbuffers::FlatBufferBuilder builder;

    core::VehicleControlCommandRecordBuilder record_builder(builder);
    builder.Finish(record_builder.Finish());

    const auto* record = flatbuffers::GetRoot<core::VehicleControlCommandRecord>(builder.GetBufferPointer());
    CHECK(record->data() == nullptr);
}
