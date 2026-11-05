// SPDX-FileCopyrightText: Copyright (c) 2026 BrainCo. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/revohuman_plugin/device_status.hpp"

#include <utility>

namespace plugins
{
namespace revohuman
{

namespace
{

using core::PluginDeviceReason;
using core::PluginDeviceReason_CALIBRATION_FAILED;
using core::PluginDeviceReason_DEVICE_ERROR;
using core::PluginDeviceReason_DISABLED_BY_CONFIGURATION;
using core::PluginDeviceReason_NO_CURRENT_DATA;
using core::PluginDeviceReason_NO_HARDWARE_SIGNAL;
using core::PluginDeviceReason_NONE;
using core::PluginDeviceState;
using core::PluginDeviceState_CONNECTED;
using core::PluginDeviceState_DEGRADED;
using core::PluginDeviceState_DISABLED;
using core::PluginDeviceState_DISCONNECTED;
using core::PluginDeviceState_FAILED;
using core::PluginDeviceState_UNKNOWN;

DeviceStatusRow row(std::string path, PluginDeviceState state, PluginDeviceReason reason, std::string error = {})
{
    return DeviceStatusRow{ .path = std::move(path), .state = state, .reason = reason, .error = std::move(error) };
}

//! A device that has given data before and stopped is DISCONNECTED; one that never did is UNKNOWN.
PluginDeviceState lost_or_unknown(bool seen)
{
    return seen ? PluginDeviceState_DISCONNECTED : PluginDeviceState_UNKNOWN;
}

DeviceStatusRow hand_row(std::string path, const HandHealth& h)
{
    if (h.hand_active && h.wrist_tracked)
    {
        return row(std::move(path), PluginDeviceState_CONNECTED, PluginDeviceReason_NONE);
    }
    if (h.hand_active)
    {
        return row(std::move(path), PluginDeviceState_DEGRADED, PluginDeviceReason_NO_CURRENT_DATA,
                   "no tracked wrist pose from the host");
    }
    if (h.kinematics_missing)
    {
        return row(std::move(path), PluginDeviceState_DEGRADED, PluginDeviceReason_NO_CURRENT_DATA,
                   "glove samples carry no joint poses");
    }
    return row(std::move(path), lost_or_unknown(h.glove_seen), PluginDeviceReason_NO_HARDWARE_SIGNAL);
}

DeviceStatusRow glove_row(std::string path, const HandHealth& h, const std::optional<Source::GloveNote>& note)
{
    // What the source says about the glove overrides a recently cached healthy sample.
    if (note)
    {
        switch (note->kind)
        {
        case Source::GloveNote::Kind::Ended:
            return row(std::move(path), PluginDeviceState_FAILED, PluginDeviceReason_DEVICE_ERROR, note->text);
        case Source::GloveNote::Kind::Waiting:
            // A glove that streamed before and is looked for again was lost, not never there.
            return row(std::move(path), lost_or_unknown(h.glove_seen), PluginDeviceReason_NO_HARDWARE_SIGNAL, note->text);
        case Source::GloveNote::Kind::NotReady:
            return row(std::move(path), h.glove_seen ? PluginDeviceState_DEGRADED : PluginDeviceState_UNKNOWN,
                       PluginDeviceReason_NO_CURRENT_DATA, note->text);
        }
    }
    if (h.glove_live)
    {
        return row(std::move(path), PluginDeviceState_CONNECTED, PluginDeviceReason_NONE);
    }
    if (!h.glove_seen)
    {
        return row(std::move(path), PluginDeviceState_UNKNOWN, PluginDeviceReason_NO_HARDWARE_SIGNAL);
    }
    return row(std::move(path), PluginDeviceState_DISCONNECTED, PluginDeviceReason_NO_HARDWARE_SIGNAL,
               "no recent glove sample");
}

DeviceStatusRow tactile_row(std::string path, const HandHealth& h)
{
    if (h.tactile_live)
    {
        return row(std::move(path), PluginDeviceState_CONNECTED, PluginDeviceReason_NONE);
    }
    switch (h.tactile_skip)
    {
    case TactileSkip::NoShape:
        return row(std::move(path), PluginDeviceState_DISABLED, PluginDeviceReason_DISABLED_BY_CONFIGURATION,
                   "no TactileMetadata: the C SDK gave no taxel layout (see the start-up log)");
    case TactileSkip::NoTactile:
        return row(std::move(path), PluginDeviceState_DISABLED, PluginDeviceReason_DISABLED_BY_CONFIGURATION,
                   "this glove has no tactile sensors");
    case TactileSkip::Uncalibrated:
        if (h.glove_live)
        {
            return row(std::move(path), PluginDeviceState_DEGRADED, PluginDeviceReason_CALIBRATION_FAILED,
                       tactile_skip_name(h.tactile_skip));
        }
        break;
    case TactileSkip::CountMismatch:
        if (h.glove_live)
        {
            return row(std::move(path), PluginDeviceState_FAILED, PluginDeviceReason_DEVICE_ERROR,
                       "taxel count differs from the TactileMetadata");
        }
        break;
    case TactileSkip::None:
        break;
    default:
        if (h.glove_live)
        {
            return row(std::move(path), PluginDeviceState_DEGRADED, PluginDeviceReason_NO_CURRENT_DATA,
                       tactile_skip_name(h.tactile_skip));
        }
        break;
    }
    return row(std::move(path), lost_or_unknown(h.tactile_seen), PluginDeviceReason_NO_HARDWARE_SIGNAL);
}

} // namespace

std::array<DeviceStatusRow, 3> make_hand_status(Hand hand, const HandStatusInput& input)
{
    const std::string side = hand == Hand::Left ? "left" : "right";
    std::string hand_path = "/hand/" + side;
    std::string glove_path = "/glove/revohuman_" + side;
    std::string tactile_path = "/tactile/revohuman_" + side;
    return { hand_row(std::move(hand_path), input.health), glove_row(std::move(glove_path), input.health, input.note),
             tactile_row(std::move(tactile_path), input.health) };
}

} // namespace revohuman
} // namespace plugins
