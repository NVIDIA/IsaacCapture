// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the generated TactileData FlatBuffer message.

#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <openxr/openxr.h>
#include <schema/serialized.hpp>
#include <schema/tactile_generated.h>
#include <schema/timestamp_generated.h>

#include <memory>
#include <vector>

// VT_<FIELD> == (id + 2) * 2.
#define VT(field) (field + 2) * 2
static_assert(core::TactileData::VT_TAXEL_FORCE == VT(0));
static_assert(core::TactileDataRecord::VT_DATA == VT(0));
static_assert(core::TactileDataRecord::VT_TIMESTAMP == VT(1));

// Producers copy XrVector3f arrays straight into taxel_force.
static_assert(sizeof(core::Point) == sizeof(XrVector3f));

namespace
{

core::TactileDataT make_frame()
{
    core::TactileDataT native;
    native.taxel_force = { core::Point(0.0f, 0.0f, 1.0f), core::Point(0.5f, -0.5f, 2.0f) };
    return native;
}

} // namespace

TEST_CASE("TactileData round-trips every field", "[unit][tactile]")
{
    const auto data = core::pack<core::TactileData>(make_frame());

    REQUIRE(data);
    REQUIRE(data->taxel_force() != nullptr);
    REQUIRE(data->taxel_force()->size() == 2);
    CHECK(data->taxel_force()->Get(1)->x() == 0.5f);
    CHECK(data->taxel_force()->Get(1)->z() == 2.0f);
}

TEST_CASE("TactileData leaves an empty frame's vector absent", "[unit][tactile]")
{
    const auto data = core::pack<core::TactileData>(core::TactileDataT{});

    REQUIRE(data);
    CHECK(data->taxel_force() == nullptr);
}

TEST_CASE("TactileDataRecord verifies and carries its timestamp", "[unit][tactile]")
{
    core::TactileDataRecordT record;
    record.data = std::make_shared<core::TactileDataT>(make_frame());
    record.timestamp = std::make_shared<core::DeviceDataTimestamp>(1, 2, 3);

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::TactileDataRecord::Pack(builder, &record));

    flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
    REQUIRE(core::VerifyTactileDataRecordBuffer(verifier));
    const auto* read = core::GetTactileDataRecord(builder.GetBufferPointer());
    CHECK(read->timestamp()->sample_time_local_common_clock() == 2);
    CHECK(read->data()->taxel_force()->size() == 2);
}
