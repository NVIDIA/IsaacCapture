// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Semantic validation for tactile metadata and frames. FlatBuffers verification covers
// memory safety; these checks cover the cross-field invariants of the protocol.

#pragma once

#include <schema/tactile_binding_profile_generated.h>
#include <schema/tactile_generated.h>
#include <schema/tactile_metadata_generated.h>
#include <schema/tactile_profiles.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace core
{

inline void validate_tactile_measurement(const TactileMeasurement& measurement)
{
    switch (measurement.quantity())
    {
    case TactileQuantity_FORCE:
        if ((measurement.component_count() != 1 && measurement.component_count() != 3) ||
            measurement.unit() != TactileUnit_NEWTON || measurement.frame() != TactileValueFrame_TAXEL)
        {
            throw std::invalid_argument("FORCE requires 1 or 3 NEWTON components in the TAXEL frame");
        }
        break;
    case TactileQuantity_PRESSURE:
        if (measurement.component_count() != 1 || measurement.unit() != TactileUnit_PASCAL ||
            measurement.frame() != TactileValueFrame_NONE)
        {
            throw std::invalid_argument("PRESSURE requires one PASCAL component with no vector frame");
        }
        break;
    case TactileQuantity_RAW:
        if (measurement.component_count() != 1 || measurement.unit() != TactileUnit_COUNT ||
            measurement.frame() != TactileValueFrame_NONE)
        {
            throw std::invalid_argument("RAW requires one COUNT component with no vector frame");
        }
        break;
    default:
        throw std::invalid_argument("measurement quantity must be a tactile schema v1 quantity");
    }
}

inline bool equals(const flatbuffers::String* value, std::string_view expected)
{
    return value != nullptr && value->string_view() == expected;
}

inline bool equals(const flatbuffers::String* left, const flatbuffers::String* right)
{
    return left != nullptr && right != nullptr && left->string_view() == right->string_view();
}

inline void validate_tactile_binding_profile(const SemanticProfile& profile, size_t node_count)
{
    const auto* payload = profile.payload();
    if (!equals(profile.payload_schema(), TACTILE_BINDING_PROFILE_SCHEMA) || payload == nullptr)
    {
        throw std::invalid_argument("known tactile binding profile has the wrong payload schema");
    }
    flatbuffers::Verifier verifier(payload->data(), payload->size());
    if (!verifier.VerifyBuffer<TactileBindingProfile>(nullptr))
    {
        throw std::invalid_argument("tactile binding profile payload is not a valid FlatBuffer");
    }
    const auto* binding = flatbuffers::GetRoot<TactileBindingProfile>(payload->data());

    switch (binding->kind())
    {
    case TactileBindingKind_OPENXR_HAND:
    {
        if (!equals(profile.profile_id(), TACTILE_OPENXR_HAND_PROFILE_ID) || binding->openxr_hand() == nullptr ||
            binding->urdf() != nullptr || binding->isaaccapture_pose() != nullptr)
        {
            throw std::invalid_argument("OpenXR hand profile id, kind, and payload must agree");
        }
        const auto* indices = binding->openxr_hand()->node_joint_indices();
        if (indices == nullptr || indices->size() != node_count)
        {
            throw std::invalid_argument("OpenXR hand profile must map every tactile node");
        }
        std::unordered_set<uint8_t> joints;
        for (uint8_t joint : *indices)
        {
            if (joint >= 26 || !joints.insert(joint).second)
            {
                throw std::invalid_argument("OpenXR hand profile joint ordinals must be unique and less than 26");
            }
        }
        break;
    }
    case TactileBindingKind_URDF:
    {
        if (!equals(profile.profile_id(), TACTILE_URDF_PROFILE_ID) || binding->urdf() == nullptr ||
            binding->openxr_hand() != nullptr || binding->isaaccapture_pose() != nullptr)
        {
            throw std::invalid_argument("URDF profile id, kind, and payload must agree");
        }
        const auto* mappings = binding->urdf()->nodes();
        if (mappings == nullptr || mappings->size() != node_count)
        {
            throw std::invalid_argument("URDF profile must map every tactile node");
        }
        std::unordered_set<uint32_t> mapped_nodes;
        for (const auto* mapping : *mappings)
        {
            if (mapping == nullptr || mapping->node_index() >= node_count || mapping->entity_name() == nullptr ||
                mapping->entity_name()->size() == 0 || !mapped_nodes.insert(mapping->node_index()).second)
            {
                throw std::invalid_argument("URDF node mappings must be unique, in range, and named");
            }
        }
        bool resource_found = false;
        if (const auto* resources = profile.resources(); resources != nullptr)
        {
            for (const auto* resource : *resources)
            {
                if (resource != nullptr && equals(resource->logical_id(), binding->urdf()->resource_id()))
                {
                    resource_found = true;
                    if (resource->sha256() == nullptr || resource->sha256()->size() != 32)
                    {
                        throw std::invalid_argument("URDF content requires a 32-byte SHA-256 digest");
                    }
                    if (resource->inline_data() != nullptr && resource->inline_data()->size() != 0 &&
                        resource->inline_data()->size() != resource->byte_size())
                    {
                        throw std::invalid_argument("embedded URDF size does not match its content descriptor");
                    }
                }
            }
        }
        if (!resource_found)
        {
            throw std::invalid_argument("URDF profile resource_id does not resolve to profile content");
        }
        break;
    }
    case TactileBindingKind_ISAACCAPTURE_POSE:
    {
        const auto* pose = binding->isaaccapture_pose();
        if (!equals(profile.profile_id(), TACTILE_ISAACCAPTURE_POSE_PROFILE_ID) || pose == nullptr ||
            binding->openxr_hand() != nullptr || binding->urdf() != nullptr)
        {
            throw std::invalid_argument("IsaacCapture pose profile id, kind, and payload must agree");
        }
        if (pose->pose_schema() == nullptr || pose->pose_schema()->size() == 0 ||
            ((pose->tracker_name() == nullptr || pose->tracker_name()->size() == 0) &&
             (pose->collection_id() == nullptr || pose->collection_id()->size() == 0)) ||
            pose->node_pose_indices() == nullptr || pose->node_pose_indices()->size() != node_count)
        {
            throw std::invalid_argument("IsaacCapture pose profile must identify a stream and map every tactile node");
        }
        break;
    }
    default:
        throw std::invalid_argument("tactile binding profile kind must not be NONE");
    }
}

inline void validate_tactile_metadata(const TactileMetadata& metadata)
{
    if (metadata.schema_version() == 0)
    {
        throw std::invalid_argument("schema_version must be positive");
    }
    if (metadata.measurement() == nullptr)
    {
        throw std::invalid_argument("measurement is required");
    }
    validate_tactile_measurement(*metadata.measurement());

    const auto* nodes = metadata.nodes();
    const size_t node_count = nodes != nullptr ? nodes->size() : 0;
    std::unordered_set<std::string> node_ids;
    if (nodes != nullptr)
    {
        for (const auto* node : *nodes)
        {
            if (node == nullptr || node->node_id() == nullptr || node->node_id()->size() == 0 ||
                node->bind_pose() == nullptr)
            {
                throw std::invalid_argument("every node requires a non-empty node_id and bind_pose");
            }
            if (!node_ids.insert(node->node_id()->str()).second)
            {
                throw std::invalid_argument("node_id values must be unique");
            }
        }
    }

    const auto* taxels = metadata.taxels();
    if (taxels == nullptr || taxels->size() == 0)
    {
        throw std::invalid_argument("at least one taxel is required");
    }
    std::unordered_set<std::string> taxel_ids;
    for (const auto* taxel : *taxels)
    {
        if (taxel == nullptr || taxel->taxel_id() == nullptr || taxel->taxel_id()->size() == 0 || taxel->pose() == nullptr)
        {
            throw std::invalid_argument("every taxel requires a non-empty taxel_id and pose");
        }
        if (!taxel_ids.insert(taxel->taxel_id()->str()).second)
        {
            throw std::invalid_argument("taxel_id values must be unique");
        }

        const auto* skin = taxel->skin();
        if (skin == nullptr || skin->size() == 0)
        {
            continue;
        }
        float sum = 0.0F;
        std::unordered_set<uint32_t> influenced_nodes;
        for (const auto* influence : *skin)
        {
            if (influence->node_index() >= node_count)
            {
                throw std::invalid_argument("taxel skin references an out-of-range node");
            }
            if (!std::isfinite(influence->weight()) || influence->weight() <= 0.0F)
            {
                throw std::invalid_argument("taxel skin weights must be finite and positive");
            }
            if (!influenced_nodes.insert(influence->node_index()).second)
            {
                throw std::invalid_argument("taxel skin must not repeat a node");
            }
            sum += influence->weight();
        }
        if (std::abs(sum - 1.0F) > 1.0e-5F)
        {
            throw std::invalid_argument("taxel skin weights must sum to 1");
        }
    }

    std::unordered_set<std::string> profile_ids;
    if (const auto* profiles = metadata.profiles(); profiles != nullptr)
    {
        for (const auto* profile : *profiles)
        {
            if (profile == nullptr || profile->profile_id() == nullptr || profile->profile_id()->size() == 0)
            {
                throw std::invalid_argument("every semantic profile requires a profile_id");
            }
            if (!profile_ids.insert(profile->profile_id()->str()).second)
            {
                throw std::invalid_argument("profile_id values must be unique");
            }
            if (equals(profile->profile_id(), TACTILE_OPENXR_HAND_PROFILE_ID) ||
                equals(profile->profile_id(), TACTILE_URDF_PROFILE_ID) ||
                equals(profile->profile_id(), TACTILE_ISAACCAPTURE_POSE_PROFILE_ID))
            {
                validate_tactile_binding_profile(*profile, node_count);
            }
        }
    }
}

inline void validate_tactile_data(const TactileMetadata& metadata, const TactileData& data)
{
    validate_tactile_metadata(metadata);
    const size_t taxel_count = metadata.taxels()->size();
    const size_t expected_values = taxel_count * metadata.measurement()->component_count();
    if (data.values() == nullptr || data.values()->size() != expected_values)
    {
        throw std::invalid_argument("tactile value count does not match metadata");
    }
    if (data.status() != nullptr && data.status()->size() != 0 && data.status()->size() != taxel_count)
    {
        throw std::invalid_argument("taxel status count must be zero or match metadata");
    }
}

} // namespace core
