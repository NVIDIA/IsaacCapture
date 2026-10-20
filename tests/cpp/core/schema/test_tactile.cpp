// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the TactileMetadata and TactileData schemas and their validation.

#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <schema/serialized.hpp>
#include <schema/tactile_validation.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

// Structs are baked into recorded layouts; see src/core/schema/README.md.
static_assert(sizeof(core::TactileSegment) == 28);
static_assert(sizeof(core::TactileTaxel) == 52);
static_assert(sizeof(core::TactileShear) == 8);

namespace
{

core::Pose identity_at(float x, float y, float z)
{
    return core::Pose(core::Point(x, y, z), core::Quaternion(0, 0, 0, 1));
}

core::TactileTaxel taxel(const core::Pose& pose, std::array<uint16_t, 4> segments, std::array<float, 4> weights)
{
    return core::TactileTaxel(pose, flatbuffers::span<const uint16_t, 4>(segments.data(), 4),
                              flatbuffers::span<const float, 4>(weights.data(), 4));
}

// A left glove: wrist and index tip, a taxel on the tip and one skinned across both.
core::TactileMetadataT make_metadata()
{
    core::TactileMetadataT native;
    native.device = std::make_shared<core::TactileDeviceT>();
    native.device->vendor = "Example";
    native.device->side = core::BodySide_LEFT;
    native.device->nominal_rate_hz = 100.0f;

    native.shape = std::make_shared<core::TactileShapeT>();
    native.shape->segments = { core::TactileSegment(identity_at(0, 0, 0)),
                               core::TactileSegment(identity_at(0, 0.15f, 0)) };
    native.shape->taxels = { taxel(identity_at(0, 0.16f, 0.01f), { 1, 0, 0, 0 }, { 1, 0, 0, 0 }),
                             taxel(identity_at(0, 0.08f, 0.01f), { 0, 1, 0, 0 }, { 0.5f, 0.5f, 0, 0 }) };
    native.shape->taxel_areas = { 1e-5f, 1e-5f };

    native.binding = std::make_shared<core::SkeletonBindingT>();
    native.binding->profile = std::make_shared<core::SkeletonProfileRefT>();
    native.binding->profile->id = "openxr.hand";
    native.binding->profile->version = 1;
    native.binding->segment_names = { "WRIST", "INDEX_TIP" };

    native.channels = std::make_shared<core::TactileChannelsT>();
    native.channels->normal_force = true;
    native.channels->shear_force = true;
    native.channels->taxel_state = true;
    return native;
}

std::string validate(const core::TactileMetadataT& native)
{
    return core::validate_tactile_metadata(*core::pack<core::TactileMetadata>(native));
}

// Error from validating make_metadata() after `edit`.
std::string validate_edited(const std::function<void(core::TactileMetadataT&)>& edit)
{
    auto native = make_metadata();
    edit(native);
    return validate(native);
}

core::TactileDataT make_frame()
{
    core::TactileDataT frame;
    frame.normal_force = { 1.5f, 0.25f };
    frame.shear_force = { core::TactileShear(0.1f, 0), core::TactileShear(0, -0.1f) };
    frame.taxel_state = { core::TaxelState_VALID, core::TaxelState_SATURATED };
    frame.sequence = 7;
    return frame;
}

std::string validate_frame(const core::TactileDataT& frame, const core::TactileMetadataT& metadata = make_metadata())
{
    return core::validate_tactile_data(
        *core::pack<core::TactileData>(frame), *core::pack<core::TactileMetadata>(metadata));
}

} // namespace

TEST_CASE("TactileMetadata round-trips and verifies as its own root", "[unit][tactile]")
{
    const auto metadata = core::pack<core::TactileMetadata>(make_metadata());
    const auto bytes = metadata.buffer();
    flatbuffers::Verifier verifier(bytes.data(), bytes.size());
    REQUIRE(core::VerifyTactileMetadataBuffer(verifier));
    CHECK(std::string(core::TactileMetadata::GetFullyQualifiedName()) == "core.TactileMetadata");

    CHECK(metadata->device()->side() == core::BodySide_LEFT);
    CHECK(*metadata->device()->nominal_rate_hz() == 100.0f);
    const auto* second = metadata->shape()->taxels()->Get(1);
    CHECK(second->segments()->Get(1) == 1);
    CHECK(second->weights()->Get(0) == 0.5f);
    CHECK(metadata->binding()->segment_names()->Get(1)->str() == "INDEX_TIP");
    CHECK(core::validate_tactile_metadata(*metadata).empty());
}

TEST_CASE("validate_tactile_metadata rejects malformed shapes", "[unit][tactile]")
{
    using M = core::TactileMetadataT;
    CHECK_FALSE(validate_edited([](M& m) { m.shape.reset(); }).empty());
    CHECK_FALSE(validate_edited([](M& m) { m.shape->segments.clear(); }).empty());
    CHECK_FALSE(validate_edited([](M& m) { m.shape->taxels.clear(); }).empty());
    CHECK_FALSE(validate_edited(
                    [](M& m) {
                        m.shape->segments[0] =
                            core::TactileSegment(core::Pose(core::Point(0, 0, 0), core::Quaternion(0, 0, 0, 2)));
                    })
                    .empty());
    // Weights that do not sum to 1, a negative weight, an index past the segments, a segment
    // weighted twice, and a zero-weight slot naming a segment other than 0.
    CHECK_FALSE(validate_edited(
                    [](M& m) {
                        m.shape->taxels[1] = taxel(identity_at(0, 0, 0), { 0, 1, 0, 0 }, { 0.5f, 0.4f, 0, 0 });
                    })
                    .empty());
    CHECK_FALSE(validate_edited(
                    [](M& m) {
                        m.shape->taxels[1] = taxel(identity_at(0, 0, 0), { 0, 1, 0, 0 }, { 1.5f, -0.5f, 0, 0 });
                    })
                    .empty());
    CHECK_FALSE(validate_edited(
                    [](M& m) {
                        m.shape->taxels[0] = taxel(identity_at(0, 0, 0), { 2, 0, 0, 0 }, { 1, 0, 0, 0 });
                    })
                    .empty());
    CHECK_FALSE(validate_edited(
                    [](M& m) {
                        m.shape->taxels[1] = taxel(identity_at(0, 0, 0), { 1, 1, 0, 0 }, { 0.5f, 0.5f, 0, 0 });
                    })
                    .empty());
    CHECK_FALSE(validate_edited(
                    [](M& m) {
                        m.shape->taxels[0] = taxel(identity_at(0, 0, 0), { 1, 1, 0, 0 }, { 1, 0, 0, 0 });
                    })
                    .empty());
    CHECK_FALSE(validate_edited([](M& m) { m.shape->taxel_areas = { 1e-5f }; }).empty());
    CHECK_FALSE(validate_edited([](M& m) { m.shape->taxel_areas = { 1e-5f, 0.0f }; }).empty());
}

TEST_CASE("validate_tactile_metadata checks channels, binding and extensions", "[unit][tactile]")
{
    using M = core::TactileMetadataT;
    CHECK_FALSE(validate_edited([](M& m) { m.channels.reset(); }).empty());
    CHECK_FALSE(validate_edited([](M& m) { m.channels = std::make_shared<core::TactileChannelsT>(); }).empty());
    CHECK(validate_edited([](M& m) { m.binding.reset(); }).empty());
    CHECK(validate_edited([](M& m) { m.binding->segment_names = { "WRIST" }; }).find("binding:") == 0);
    CHECK(validate_edited([](M& m) { m.device->side = core::BodySide_UNSPECIFIED; }).find("side") != std::string::npos);

    const auto extension = [](const std::string& id)
    {
        auto native = std::make_shared<core::TactileExtensionT>();
        native->id = id;
        native->version = 1;
        return native;
    };
    CHECK(validate_edited([&](M& m) { m.extensions.push_back(extension("com.example.glove")); }).empty());
    CHECK_FALSE(validate_edited([&](M& m) { m.extensions.push_back(extension("glove")); }).empty());
    CHECK_FALSE(validate_edited(
                    [&](M& m)
                    {
                        m.extensions.push_back(extension("com.example.glove"));
                        m.extensions.push_back(extension("com.example.glove"));
                    })
                    .empty());
}

TEST_CASE("validate_tactile_data matches the declared channels to the taxel count", "[unit][tactile]")
{
    CHECK(validate_frame(make_frame()).empty());

    auto missing = make_frame();
    missing.normal_force.clear();
    CHECK(validate_frame(missing).find("normal_force") == 0);

    auto short_shear = make_frame();
    short_shear.shear_force.pop_back();
    CHECK(validate_frame(short_shear).find("shear_force") == 0);

    auto undeclared = make_frame();
    undeclared.raw = { 1, 2 };
    CHECK(validate_frame(undeclared).find("raw") == 0);

    auto with_raw = make_metadata();
    with_raw.channels->raw_values_per_taxel = 3;
    auto raw = make_frame();
    raw.raw = { 1, 2, 3, 4, 5, 6 };
    CHECK(validate_frame(raw, with_raw).empty());
    raw.raw.pop_back();
    CHECK(validate_frame(raw, with_raw).find("raw") == 0);
}

TEST_CASE("TactileData round-trips through its record", "[unit][tactile]")
{
    core::TactileDataRecordT native;
    native.data = std::make_shared<core::TactileDataT>(make_frame());
    native.timestamp = std::make_shared<core::DeviceDataTimestamp>(1, 2, 3);
    const auto record = core::pack<core::TactileDataRecord>(native);
    const auto* data = record->data();
    REQUIRE(data != nullptr);
    CHECK(data->shear_force()->Get(1)->y() == -0.1f);
    CHECK(data->taxel_state()->Get(1) == core::TaxelState_SATURATED);
    CHECK(*data->sequence() == 7u);
    CHECK(record->timestamp()->sample_time_local_common_clock() == 2);
}
