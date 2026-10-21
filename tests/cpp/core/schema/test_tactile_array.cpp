// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the generated TactileDataArray FlatBuffer message.

#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <openxr/openxr.h>
#include <schema/serialized.hpp>
#include <schema/tactile_array_generated.h>
#include <schema/timestamp_generated.h>

#include <memory>
#include <vector>

// VT_<FIELD> == (id + 2) * 2.
#define VT(field) (field + 2) * 2
// Field renames change generated VT_* symbols without changing field IDs.
static_assert(core::TactileDataArray::VT_TAXEL_FORCES == VT(0));
static_assert(core::TactileDataArrayRecord::VT_DATA == VT(0));
static_assert(core::TactileDataArrayRecord::VT_TIMESTAMP == VT(1));

// Producers copy XrVector3f arrays straight into taxel_forces.
static_assert(sizeof(core::Point) == sizeof(XrVector3f));

namespace
{

core::TactileDataArrayT make_frame()
{
    core::TactileDataArrayT native;
    native.taxel_forces = { core::Point(0.0f, 0.0f, 1.0f), core::Point(0.5f, -0.5f, 2.0f) };
    return native;
}

} // namespace

TEST_CASE("TactileDataArray round-trips every field", "[unit][tactile]")
{
    const auto data = core::pack<core::TactileDataArray>(make_frame());

    REQUIRE(data);
    REQUIRE(data->taxel_forces() != nullptr);
    REQUIRE(data->taxel_forces()->size() == 2);
    CHECK(data->taxel_forces()->Get(1)->x() == 0.5f);
    CHECK(data->taxel_forces()->Get(1)->z() == 2.0f);
}

TEST_CASE("TactileDataArray leaves an empty frame's vector absent", "[unit][tactile]")
{
    const auto data = core::pack<core::TactileDataArray>(core::TactileDataArrayT{});

    REQUIRE(data);
    CHECK(data->taxel_forces() == nullptr);
}

TEST_CASE("TactileDataArrayRecord verifies and carries its timestamp", "[unit][tactile]")
{
    core::TactileDataArrayRecordT record;
    record.data = std::make_shared<core::TactileDataArrayT>(make_frame());
    record.timestamp = std::make_shared<core::DeviceDataTimestamp>(1, 2, 3);

    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::TactileDataArrayRecord::Pack(builder, &record));

    flatbuffers::Verifier verifier(builder.GetBufferPointer(), builder.GetSize());
    REQUIRE(core::VerifyTactileDataArrayRecordBuffer(verifier));
    const auto* read = core::GetTactileDataArrayRecord(builder.GetBufferPointer());
    CHECK(read->timestamp()->sample_time_local_common_clock() == 2);
    CHECK(read->data()->taxel_forces()->size() == 2);
}
