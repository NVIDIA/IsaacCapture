// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <schema/tactile_generated.h>
#include <schema/tactile_metadata_generated.h>
#include <schema/tactile_validation.hpp>

#include <memory>
#include <string>

namespace
{

std::shared_ptr<core::TactileMeasurementT> force_measurement()
{
    auto measurement = std::make_shared<core::TactileMeasurementT>();
    measurement->quantity = core::TactileQuantity_FORCE;
    measurement->unit = core::TactileUnit_NEWTON;
    measurement->component_count = 3;
    measurement->frame = core::TactileValueFrame_TAXEL;
    return measurement;
}

std::shared_ptr<core::TactileTaxelT> taxel(std::string id, uint32_t node_index)
{
    auto value = std::make_shared<core::TactileTaxelT>();
    value->taxel_id = std::move(id);
    value->pose = std::make_shared<core::Pose>();
    value->skin.emplace_back(node_index, 1.0F);
    return value;
}

core::TactileMetadataT metadata()
{
    core::TactileMetadataT value;
    value.schema_version = 1;
    value.shape_frame_id = "glove";
    value.measurement = force_measurement();
    auto node = std::make_shared<core::TactileNodeT>();
    node->node_id = "index_distal";
    node->bind_pose = std::make_shared<core::Pose>();
    value.nodes.push_back(std::move(node));
    value.taxels.push_back(taxel("taxel-0", 0));
    value.taxels.push_back(taxel("taxel-1", 0));
    return value;
}

} // namespace

TEST_CASE("Tactile metadata accepts a right-handed three-axis force layout", "[unit][schema]")
{
    auto native = metadata();
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::TactileMetadata::Pack(builder, &native));
    const auto* encoded = flatbuffers::GetRoot<core::TactileMetadata>(builder.GetBufferPointer());

    CHECK_NOTHROW(core::validate_tactile_metadata(*encoded));
    CHECK(encoded->measurement()->component_count() == 3);
    CHECK(encoded->taxels()->size() == 2);
}

TEST_CASE("Tactile frames must match metadata dimensions", "[unit][schema]")
{
    auto metadata_native = metadata();
    flatbuffers::FlatBufferBuilder metadata_builder;
    metadata_builder.Finish(core::TactileMetadata::Pack(metadata_builder, &metadata_native));
    const auto* encoded_metadata = flatbuffers::GetRoot<core::TactileMetadata>(metadata_builder.GetBufferPointer());

    core::TactileDataT data_native;
    data_native.values = { 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F };
    flatbuffers::FlatBufferBuilder data_builder;
    data_builder.Finish(core::TactileData::Pack(data_builder, &data_native));
    const auto* encoded_data = flatbuffers::GetRoot<core::TactileData>(data_builder.GetBufferPointer());
    CHECK_NOTHROW(core::validate_tactile_data(*encoded_metadata, *encoded_data));

    data_native.values.pop_back();
    flatbuffers::FlatBufferBuilder short_builder;
    short_builder.Finish(core::TactileData::Pack(short_builder, &data_native));
    const auto* short_data = flatbuffers::GetRoot<core::TactileData>(short_builder.GetBufferPointer());
    CHECK_THROWS_AS(core::validate_tactile_data(*encoded_metadata, *short_data), std::invalid_argument);
}

TEST_CASE("Tactile skinning rejects invalid node references and weights", "[unit][schema]")
{
    auto native = metadata();
    native.taxels[0]->skin[0] = core::TactileSkinWeight(1, 1.0F);
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(core::TactileMetadata::Pack(builder, &native));
    const auto* encoded = flatbuffers::GetRoot<core::TactileMetadata>(builder.GetBufferPointer());
    CHECK_THROWS_AS(core::validate_tactile_metadata(*encoded), std::invalid_argument);
}
