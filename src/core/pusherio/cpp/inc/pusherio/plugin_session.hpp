// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hand_tracking_push_channel.hpp"
#include "schema_push_channel.hpp"
#include "wrist_tracking_source.hpp"

#include <deviceio_base/tracker.hpp>

#include <functional>
#include <memory>
#include <optional>
#include <vector>

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
    //! Wrist source mode the pull channel will serve; empty when wrist tracking is not pulled.
    std::optional<WristTrackingSourceMode> wrist_tracking_pull;
};

/*!
 * @brief Multiplexed transport-neutral pull channel for typed DeviceIO trackers and optional sources.
 *
 * One update establishes the snapshot boundary for every tracker on the channel.
 * The session owns the pull channel; it must outlive every source created from it.
 */
class IPluginPullChannel : public ITrackerSession
{
public:
    virtual void update() = 0;

    //! Throws std::logic_error unless wrist tracking pull was declared with config.mode. Never returns null.
    virtual std::unique_ptr<IWristTrackingSource> create_wrist_tracking_source(const WristTrackingSourceConfig& config) = 0;
};

/*!
 * @brief Session abstraction that creates operation-specific plugin channels.
 *
 * The caller must keep the session alive until all push channels and wrist
 * sources created from it have been destroyed. The create_* methods throw
 * std::logic_error for a channel type that was not declared and never return null.
 */
class IPluginSession
{
public:
    virtual ~IPluginSession() = default;

    virtual IPluginPullChannel& pull_channel() = 0;
    virtual std::unique_ptr<ISchemaPushChannel> create_schema_push_channel(const SchemaPusherConfig& config) = 0;
    virtual std::unique_ptr<IHandTrackingPushChannel> create_hand_tracking_push_channel(XrHandEXT hand) = 0;
};

//! Creates a plugin session for the given requirements; the trackers join its pull channel.
using PluginSessionFactory =
    std::function<std::unique_ptr<IPluginSession>(const PluginSessionRequirements&, std::vector<std::shared_ptr<ITracker>>)>;

} // namespace core
