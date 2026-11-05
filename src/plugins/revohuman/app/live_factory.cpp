// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Built when the RevoHuman C SDK library was found.

#include "revohuman_plugin.hpp"

#include <revohuman_plugin/hand_kinematics.hpp>
#include <revohuman_plugin/live_source.hpp>

#include <stdexcept>
#include <utility>

namespace plugins
{
namespace revohuman
{

namespace
{

//! Each glove gets the hand model the SDK gives its identified product.
KinematicsFactory kinematics()
{
    return [](const std::string& product_code)
    {
        // An SDK whose model differs from the reviewed one makes make_hand_kinematics throw; the caller
        // (LiveSource::open_hand) catches it and gives up on just this glove, logging the reason at error level.
        return make_hand_kinematics(hand_model_of_product(product_code));
    };
}

} // namespace

std::optional<std::string> live_source_unavailable_reason(const LiveSourceConfig&)
{
    // Neither the SDK hand points nor the hand model has a fallback: a build or SDK without either refuses
    // to start. The model of each glove is known only once it is open, so DV1 stands in here.
    try
    {
        make_hand_kinematics(HandModel::Dv1);
    }
    catch (const std::exception& e)
    {
        return e.what();
    }
    if (!sdk_has_product_hand_models())
        return "this build's C SDK cannot tell the hand model of a glove's product (no rh_variant_hand_model); "
               "rebuild with -DREVOHUMAN_SDK_ROOT pointing at a C SDK that has it";
    return std::nullopt;
}

std::unique_ptr<Source> make_live_source(const LiveSourceConfig& config)
{
    return std::make_unique<LiveSource>(config, kinematics());
}

TactileShape make_tactile_shape(Hand hand)
{
    // Built before any glove is open. Only DV1 has a taxel layout: a P1 glove has no tactile sensors, and the
    // pipeline pushes this metadata only once a glove with tactile sensors is identified.
    auto built = make_sdk_tactile_shape(HandModel::Dv1, hand);
    return { std::make_shared<const LoadedShape>(std::move(built.metadata)), "C SDK taxel layout " + built.layout_source };
}

} // namespace revohuman
} // namespace plugins
