// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the generated SpaceMouse FlatBuffer types.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <schema/spacemouse_generated.h>

#include <memory>

#define VT(field) (field + 2) * 2

static_assert(core::SpaceMouseOutput::VT_TRANSLATION == VT(0));
static_assert(core::SpaceMouseOutput::VT_ROTATION == VT(1));
static_assert(core::SpaceMouseOutput::VT_BUTTONS == VT(2));
static_assert(core::SpaceMouseOutput::VT_CONNECTED == VT(3));

static_assert(core::SpaceMouseOutputRecord::VT_DATA == VT(0));
static_assert(core::SpaceMouseOutputRecord::VT_TIMESTAMP == VT(1));

TEST_CASE("SpaceMouseOutputT default construction", "[spacemouse][native]")
{
    core::SpaceMouseOutputT output;

    CHECK(output.translation == nullptr);
    CHECK(output.rotation == nullptr);
    CHECK(output.buttons.empty());
    CHECK_FALSE(output.connected);
}

TEST_CASE("SpaceMouseOutput serialization and deserialization", "[spacemouse][serialize]")
{
    flatbuffers::FlatBufferBuilder builder;

    core::SpaceMouseOutputT output;
    output.translation = std::make_shared<core::Point>(0.5f, -0.25f, 1.0f);
    output.rotation = std::make_shared<core::Point>(-1.0f, 0.0f, 0.75f);
    output.buttons = { 1, 0 };
    output.connected = true;

    auto offset = core::SpaceMouseOutput::Pack(builder, &output);
    builder.Finish(offset);

    const auto* deserialized = flatbuffers::GetRoot<core::SpaceMouseOutput>(builder.GetBufferPointer());
    REQUIRE(deserialized->translation() != nullptr);
    CHECK(deserialized->translation()->x() == Catch::Approx(0.5f));
    CHECK(deserialized->translation()->y() == Catch::Approx(-0.25f));
    CHECK(deserialized->translation()->z() == Catch::Approx(1.0f));
    REQUIRE(deserialized->rotation() != nullptr);
    CHECK(deserialized->rotation()->x() == Catch::Approx(-1.0f));
    CHECK(deserialized->rotation()->z() == Catch::Approx(0.75f));
    REQUIRE(deserialized->buttons()->size() == 2);
    CHECK(deserialized->buttons()->Get(0) == 1);
    CHECK(deserialized->buttons()->Get(1) == 0);
    CHECK(deserialized->connected());
}
