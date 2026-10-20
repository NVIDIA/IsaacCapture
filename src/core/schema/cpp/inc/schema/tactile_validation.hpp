// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The checks a TactileMetadata and each TactileData published under it must pass.
// docs/source/references/tactile_schema.rst lists the rules.

#pragma once

#include <schema/skeleton_binding.hpp>
#include <schema/tactile_generated.h>
#include <schema/tactile_metadata_generated.h>

#include <cmath>
#include <cstddef>
#include <set>
#include <string>

namespace core
{

//! Largest deviation from 1 a taxel's weights may sum to.
inline constexpr float TACTILE_WEIGHT_SUM_TOLERANCE = 1e-5f;
//! Largest deviation from 1 the norm of a bind or taxel orientation may have.
inline constexpr float TACTILE_QUATERNION_NORM_TOLERANCE = 1e-3f;

namespace detail
{

inline bool is_rigid_pose(const Pose& pose)
{
    const auto& p = pose.position();
    const auto& q = pose.orientation();
    if (!std::isfinite(p.x()) || !std::isfinite(p.y()) || !std::isfinite(p.z()))
    {
        return false;
    }
    const float norm = std::sqrt(q.x() * q.x() + q.y() * q.y() + q.z() * q.z() + q.w() * q.w());
    return std::isfinite(norm) && std::fabs(norm - 1.0f) <= TACTILE_QUATERNION_NORM_TOLERANCE;
}

inline std::string check_vector_size(const char* name, size_t present, size_t expected, bool declared)
{
    if (!declared)
    {
        return present == 0 ? "" : std::string(name) + " is filled but not declared in channels";
    }
    if (present != expected)
    {
        return std::string(name) + " has " + std::to_string(present) + " values, expected " + std::to_string(expected);
    }
    return "";
}

} // namespace detail

/*!
 * @brief Checks `metadata` against the rules for a tactile device description.
 * @return Empty when valid, otherwise the first rule it breaks.
 */
inline std::string validate_tactile_metadata(const TactileMetadata& metadata)
{
    const auto* shape = metadata.shape();
    if (shape == nullptr)
    {
        return "shape is missing";
    }
    const auto* segments = shape->segments();
    const size_t segment_count = segments != nullptr ? segments->size() : 0;
    if (segment_count == 0)
    {
        return "shape has no segments";
    }
    for (size_t s = 0; s < segment_count; ++s)
    {
        if (!detail::is_rigid_pose(segments->Get(static_cast<flatbuffers::uoffset_t>(s))->bind_pose()))
        {
            return "segments[" + std::to_string(s) + "].bind_pose is not finite with a unit orientation";
        }
    }

    const auto* taxels = shape->taxels();
    const size_t taxel_count = taxels != nullptr ? taxels->size() : 0;
    if (taxel_count == 0)
    {
        return "shape has no taxels";
    }
    for (size_t i = 0; i < taxel_count; ++i)
    {
        const auto* taxel = taxels->Get(static_cast<flatbuffers::uoffset_t>(i));
        const std::string at = "taxels[" + std::to_string(i) + "]";
        if (!detail::is_rigid_pose(taxel->pose()))
        {
            return at + ".pose is not finite with a unit orientation";
        }
        float sum = 0.0f;
        std::set<uint16_t> weighted;
        for (flatbuffers::uoffset_t k = 0; k < 4; ++k)
        {
            const float weight = taxel->weights()->Get(k);
            const uint16_t segment = taxel->segments()->Get(k);
            if (!std::isfinite(weight) || weight < 0.0f)
            {
                return at + ".weights[" + std::to_string(k) + "] is negative or not finite";
            }
            if (weight == 0.0f)
            {
                if (segment != 0)
                {
                    return at + ".segments[" + std::to_string(k) + "] has zero weight but is not 0";
                }
                continue;
            }
            if (segment >= segment_count)
            {
                return at + ".segments[" + std::to_string(k) + "] is " + std::to_string(segment) + ", past the " +
                       std::to_string(segment_count) + " segments";
            }
            if (!weighted.insert(segment).second)
            {
                return at + " weights segment " + std::to_string(segment) + " twice";
            }
            sum += weight;
        }
        if (std::fabs(sum - 1.0f) > TACTILE_WEIGHT_SUM_TOLERANCE)
        {
            return at + ".weights sum to " + std::to_string(sum) + ", not 1";
        }
    }

    if (const auto* areas = shape->taxel_areas())
    {
        if (areas->size() != taxel_count)
        {
            return "taxel_areas has " + std::to_string(areas->size()) + " values for " + std::to_string(taxel_count) +
                   " taxels";
        }
        for (const float area : *areas)
        {
            if (!std::isfinite(area) || area <= 0.0f)
            {
                return "taxel_areas must be finite and positive";
            }
        }
    }

    const auto* channels = metadata.channels();
    if (channels == nullptr)
    {
        return "channels is missing";
    }
    if (!channels->normal_force() && !channels->shear_force() && channels->raw_values_per_taxel() == 0)
    {
        return "channels declares no force or raw data";
    }

    if (const auto* binding = metadata.binding())
    {
        const BodySide side = metadata.device() != nullptr ? metadata.device()->side() : BodySide_UNSPECIFIED;
        const std::string error = validate_skeleton_binding(*binding, segment_count, side);
        if (!error.empty())
        {
            return "binding: " + error;
        }
    }

    if (const auto* extensions = metadata.extensions())
    {
        std::set<std::string> ids;
        for (const auto* extension : *extensions)
        {
            const std::string id = extension->id() != nullptr ? extension->id()->str() : std::string();
            if (!is_valid_registry_id(id))
            {
                return "extension id '" + id + "' is not a registry id";
            }
            if (!ids.insert(id).second)
            {
                return "extension id '" + id + "' appears twice";
            }
        }
    }
    return "";
}

/*!
 * @brief Checks that `data` fills exactly the channels `metadata` declares, one entry per taxel.
 *
 * Precondition: `metadata` passes validate_tactile_metadata().
 * @return Empty when valid, otherwise what is wrong.
 */
inline std::string validate_tactile_data(const TactileData& data, const TactileMetadata& metadata)
{
    const size_t taxel_count = metadata.shape()->taxels()->size();
    const auto* channels = metadata.channels();
    const auto size = [](const auto* vector) -> size_t { return vector != nullptr ? vector->size() : 0; };

    std::string error =
        detail::check_vector_size("normal_force", size(data.normal_force()), taxel_count, channels->normal_force());
    if (error.empty())
    {
        error = detail::check_vector_size("shear_force", size(data.shear_force()), taxel_count, channels->shear_force());
    }
    if (error.empty())
    {
        error = detail::check_vector_size("raw", size(data.raw()), taxel_count * channels->raw_values_per_taxel(),
                                          channels->raw_values_per_taxel() > 0);
    }
    if (error.empty())
    {
        error = detail::check_vector_size("taxel_state", size(data.taxel_state()), taxel_count, channels->taxel_state());
    }
    return error;
}

} // namespace core
