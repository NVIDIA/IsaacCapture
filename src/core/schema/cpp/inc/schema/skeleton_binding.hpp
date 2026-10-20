// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The registered skeleton profiles, and the checks every SkeletonBinding must pass.
// docs/source/references/skeleton_binding.rst is the registry's specification.

#pragma once

#include <schema/full_body_generated.h>
#include <schema/hand_generated.h>
#include <schema/skeleton_binding_generated.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace core
{

//! XR_EXT_hand_tracking joints, named as HandJoint names them.
inline constexpr std::string_view SKELETON_PROFILE_OPENXR_HAND = "openxr.hand";
//! XR_BD_body_tracking joints, named as BodyJoint names them.
inline constexpr std::string_view SKELETON_PROFILE_OPENXR_BODY_BD = "openxr.body.bd";
//! Link frames of the URDF the binding's asset holds, named as the URDF names them.
inline constexpr std::string_view SKELETON_PROFILE_URDF_LINK = "urdf.link";

//! A registered skeleton profile.
struct SkeletonProfile
{
    std::string_view id;
    uint32_t version;
    //! The segment names the profile defines; empty when they come from the asset.
    std::vector<std::string> segment_names;
    //! Whether a binding needs a device side of LEFT or RIGHT.
    bool requires_side;
    //! Whether a binding names its segments in an asset.
    bool requires_asset;
};

//! Every registered profile.
inline const std::vector<SkeletonProfile>& skeleton_profiles()
{
    static const std::vector<SkeletonProfile> profiles = []
    {
        std::vector<std::string> hand_joints;
        for (const HandJoint joint : EnumValuesHandJoint())
        {
            if (joint != HandJoint_NUM_JOINTS)
            {
                hand_joints.emplace_back(EnumNameHandJoint(joint));
            }
        }
        std::vector<std::string> body_joints;
        for (const BodyJoint joint : EnumValuesBodyJoint())
        {
            if (joint != BodyJoint_NUM_JOINTS)
            {
                body_joints.emplace_back(EnumNameBodyJoint(joint));
            }
        }
        return std::vector<SkeletonProfile>{
            { SKELETON_PROFILE_OPENXR_HAND, 1, std::move(hand_joints), true, false },
            { SKELETON_PROFILE_OPENXR_BODY_BD, 1, std::move(body_joints), false, false },
            { SKELETON_PROFILE_URDF_LINK, 1, {}, false, true },
        };
    }();
    return profiles;
}

//! The registered profile `id` at `version`, or null when it is not registered.
inline const SkeletonProfile* find_skeleton_profile(std::string_view id, uint32_t version)
{
    const auto& profiles = skeleton_profiles();
    const auto it =
        std::find_if(profiles.begin(), profiles.end(),
                     [&](const SkeletonProfile& profile) { return profile.id == id && profile.version == version; });
    return it != profiles.end() ? &*it : nullptr;
}

//! Whether `id` is a registry id: two or more dot-separated tokens, each a lowercase letter
//! followed by lowercase letters, digits and '_'.
inline bool is_valid_registry_id(std::string_view id)
{
    size_t tokens = 0;
    size_t start = 0;
    while (start <= id.size())
    {
        const size_t end = std::min(id.find('.', start), id.size());
        const std::string_view token = id.substr(start, end - start);
        if (token.empty() || token[0] < 'a' || token[0] > 'z')
        {
            return false;
        }
        for (const char c : token)
        {
            if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_'))
            {
                return false;
            }
        }
        ++tokens;
        start = end + 1;
    }
    return tokens >= 2;
}

//! Whether `digest` is an RFC 6920 SHA-256 name: "ni:///sha-256;" and 43 base64url characters.
inline bool is_valid_asset_digest(std::string_view digest)
{
    constexpr std::string_view prefix = "ni:///sha-256;";
    if (digest.size() != prefix.size() + 43 || digest.substr(0, prefix.size()) != prefix)
    {
        return false;
    }
    return std::all_of(
        digest.begin() + prefix.size(), digest.end(),
        [](char c)
        { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '_'; });
}

/*!
 * @brief Checks `binding` against the device it binds and against its profile.
 * @param segment_count Number of segments of the bound device.
 * @param side Side of the bound device.
 * @return Empty when valid, otherwise what is wrong. A profile that is not registered is valid
 *         and reads as a generic binding.
 */
inline std::string validate_skeleton_binding(const SkeletonBinding& binding, size_t segment_count, BodySide side)
{
    const auto* names = binding.segment_names();
    const size_t name_count = names != nullptr ? names->size() : 0;
    if (name_count != segment_count)
    {
        return "segment_names has " + std::to_string(name_count) + " names for " + std::to_string(segment_count) +
               " segments";
    }
    std::set<std::string_view> seen;
    for (size_t i = 0; i < name_count; ++i)
    {
        const auto* name = names->Get(static_cast<flatbuffers::uoffset_t>(i));
        if (name == nullptr || name->size() == 0)
        {
            return "segment_names[" + std::to_string(i) + "] is empty";
        }
        if (!seen.insert(std::string_view(name->c_str(), name->size())).second)
        {
            return "segment_names repeats '" + name->str() + "'";
        }
    }

    const auto* asset = binding.asset();
    if (asset != nullptr && (asset->digest() == nullptr ||
                             !is_valid_asset_digest(std::string_view(asset->digest()->c_str(), asset->digest()->size()))))
    {
        return "asset.digest is not an ni:///sha-256 name";
    }

    const auto* profile = binding.profile();
    if (profile == nullptr)
    {
        return asset != nullptr ? "an asset needs a profile that names segments in it" : "";
    }
    const std::string id = profile->id() != nullptr ? profile->id()->str() : std::string();
    if (!is_valid_registry_id(id))
    {
        return "profile.id '" + id + "' is not a registry id";
    }
    if (profile->version() == 0)
    {
        return "profile.version must be at least 1";
    }

    const SkeletonProfile* known = find_skeleton_profile(id, profile->version());
    if (known == nullptr)
    {
        return "";
    }
    if (known->requires_asset && asset == nullptr)
    {
        return "profile " + id + " needs an asset";
    }
    if (!known->requires_asset && asset != nullptr)
    {
        return "profile " + id + " takes no asset";
    }
    if (known->requires_side && side != BodySide_LEFT && side != BodySide_RIGHT)
    {
        return "profile " + id + " needs a side of LEFT or RIGHT";
    }
    if (!known->segment_names.empty())
    {
        for (const auto* name : *names)
        {
            if (std::find(known->segment_names.begin(), known->segment_names.end(), name->str()) ==
                known->segment_names.end())
            {
                return "'" + name->str() + "' is not a segment of profile " + id;
            }
        }
    }
    return "";
}

} // namespace core
