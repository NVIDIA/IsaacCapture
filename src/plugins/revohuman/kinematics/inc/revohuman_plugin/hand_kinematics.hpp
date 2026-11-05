// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <revohuman_plugin/pose_math.hpp>
#include <revohuman_plugin/source.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace plugins
{
namespace revohuman
{

//! Mechanical hand model.
enum class HandModel
{
    Dv1, //!< PTV1 gloves.
    P, //!< P1 gloves. The SDK has no taxel layout for it: they have no tactile sensors.
};

//! The hand model of a product as the SDK identified it ("P1", "PTV1", ...), from the SDK's own table
//! (rh_variant_hand_model). Throws std::runtime_error for a product the SDK has no model for, and when
//! the build or the loaded SDK has no such table; there is no default model.
HandModel hand_model_of_product(const std::string& product_code);
//! Whether this build's C SDK has that table; without it no product has a hand model.
bool sdk_has_product_hand_models();

//! OpenXR hand joints from the SDK's point set for @p model (rh_hand_points), through the reviewed
//! OpenXR mapping. Throws std::runtime_error when the build or the loaded SDK has no point set matching
//! the reviewed mapping; there is no fallback to another model.
std::shared_ptr<const KinematicsProvider> make_hand_kinematics(HandModel model);

//! Where the taxels of a glove are, from the SDK's taxel layout (rh_hand_model_taxel_layout).
struct SdkTaxels
{
    std::vector<Vec3> positions; //!< Meters, in the WRIST frame of make_hand_kinematics(model).
    std::vector<Vec3> normals; //!< Outward unit normals, same frame.
    std::vector<std::uint32_t> links; //!< SDK link each taxel is fixed to (rh_hand_model_fk order).
    std::string source; //!< The layout's source note, e.g. "cuff-surface-v2@4e7da46".
};

//! The SDK's taxels at joint angles @p angles (all valid), one per value of a tactile frame.
//! Throws std::runtime_error when make_hand_kinematics(model) would, for a model without a layout (P),
//! and when the build or the loaded SDK has no taxel layout.
SdkTaxels sdk_taxels(HandModel model, Hand hand, const std::array<double, kGloveJointCount>& angles);

//! TactileMetadata built from the SDK's taxel layout, ready for LoadedShape.
struct SdkTactileShape
{
    std::vector<std::uint8_t> metadata;
    std::string layout_source;
};

//! TactileMetadata of a glove: one segment per OpenXR joint (joint_ids 0..25) bound at all-zero joint
//! angles, and each taxel rigidly on the OpenXR joint that moves with its SDK link, with +Z along its
//! outward normal. Skinned with make_hand_kinematics(model) it follows the SDK's taxels at any joint
//! angles: every OpenXR joint that carries taxels is rigid on their SDK link. Throws as sdk_taxels() does.
SdkTactileShape make_sdk_tactile_shape(HandModel model, Hand hand);

} // namespace revohuman
} // namespace plugins
