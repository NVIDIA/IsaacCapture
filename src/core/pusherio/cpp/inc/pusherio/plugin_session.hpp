// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hand_tracking_push_channel.hpp"
#include "schema_pusher.hpp"
#include "wrist_tracking_source.hpp"

#include <deviceio_base/tracker.hpp>

#include <memory>

namespace core
{

/*!
 * @brief Transport-neutral capabilities a plugin may use during a session.
 *
 * Concrete sessions translate these capabilities into transport prerequisites
 * during construction and reject channel types that were not declared.
 */
struct PluginSessionRequirements
{
    bool schema_push = false;
    bool hand_tracking_push = false;
    bool wrist_tracking_pull = false;
};

/*!
 * @brief Multiplexed transport-neutral pull channel for typed DeviceIO trackers and optional sources.
 *
 * One update establishes the snapshot boundary for every tracker on the channel.
 * The pull channel must outlive every source created from it.
 */
class IPluginPullChannel : public ITrackerSession
{
public:
    virtual ~IPluginPullChannel() = default;

    virtual void update() = 0;

    //! Returns null when the requested source mode is unavailable after capability negotiation.
    virtual std::unique_ptr<IWristTrackingSource> create_wrist_tracking_source(const WristTrackingSourceConfig& config) = 0;
};

/*!
 * @brief Session abstraction that creates operation-specific plugin channels.
 *
 * The caller must keep the session alive until all pull and push channels
 * created from it have been destroyed.
 */
class IPluginSession
{
public:
    virtual ~IPluginSession() = default;

    virtual std::unique_ptr<IPluginPullChannel> create_pull_channel() = 0;
    virtual std::unique_ptr<ISchemaPushChannel> create_schema_push_channel(const SchemaPusherConfig& config) = 0;
    virtual std::unique_ptr<IHandTrackingPushChannel> create_hand_tracking_push_channel(XrHandEXT hand) = 0;
};

using PluginSessionHandle = std::shared_ptr<IPluginSession>;

} // namespace core
