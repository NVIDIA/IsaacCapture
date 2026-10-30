// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "remote_address.hpp"
#include "transport.hpp"

#include <pusherio/plugin_session.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <string>

namespace remote_plugin_session
{

namespace detail
{
class RemotePluginSessionImpl;
}

/*!
 * @brief Plugin session that forwards every channel over an ITransport to a remote_device_bridge.
 *
 * Same contract as the OpenXR plugin session: the session owns its pull channel
 * (opened with the session, with the wrist source HELLO declared) and the owner
 * destroys all push channels before the session. After the transport
 * drops, pushes become silent no-ops and channels stay safe to destroy.
 * Each push channel has a small latest-wins queue; on a stalled link the oldest
 * samples of that channel are dropped, other channels are unaffected. A channel
 * the bridge refuses (ERROR on its channel id) turns into a no-op; the session
 * and its other channels continue. The pull channel exposes no trackers; wrist
 * poses are the latest bridge samples (no interpolation).
 */
class RemotePluginSession final : public core::IPluginSession
{
public:
    struct Stats
    {
        uint64_t sent_hand = 0;
        uint64_t sent_schema = 0;
        uint64_t wrist_rx = 0;
        uint64_t heartbeats = 0;
        uint64_t dropped_queue = 0; //!< samples evicted by the per-channel latest-wins queues
        uint64_t tx_bytes = 0;
        std::map<uint16_t, uint64_t> dropped_by_channel; //!< open push channels only
        std::map<uint16_t, std::string> refused_channels; //!< channel id -> the bridge's reason
    };

    //! Connects over TCP, sends HELLO and waits for HELLO_ACK; throws std::runtime_error on any failure.
    RemotePluginSession(std::string host, uint16_t port, std::string app_name, core::PluginSessionRequirements requirements);
    //! Same handshake over an already connected transport.
    RemotePluginSession(std::unique_ptr<ITransport> transport,
                        std::string app_name,
                        core::PluginSessionRequirements requirements);
    ~RemotePluginSession() override;

    RemotePluginSession(const RemotePluginSession&) = delete;
    RemotePluginSession& operator=(const RemotePluginSession&) = delete;
    RemotePluginSession(RemotePluginSession&&) = delete;
    RemotePluginSession& operator=(RemotePluginSession&&) = delete;

    core::IPluginPullChannel& pull_channel() override;
    std::unique_ptr<core::ISchemaPushChannel> create_schema_push_channel(const core::SchemaPusherConfig& config) override;
    std::unique_ptr<core::IHandTrackingPushChannel> create_hand_tracking_push_channel(XrHandEXT hand) override;

    //! False once the transport failed; channel creation then throws and pushes are dropped.
    bool connected() const;
    //! Estimated bridge-to-local clock offset in ns (local ~= bridge + offset); 0 before the first frame.
    int64_t clock_offset_ns() const;
    //! Cumulative counters; also logged periodically as one "5s:" line.
    Stats stats() const;

private:
    std::shared_ptr<detail::RemotePluginSessionImpl> m_impl;
    std::unique_ptr<core::IPluginPullChannel> m_pull_channel;
};

//! Connects each session it builds to `remote`. The remote session exposes no trackers, so the factory throws
//! std::logic_error when handed any.
core::PluginSessionFactory make_remote_plugin_session_factory(std::string app_name, RemoteAddress remote);

} // namespace remote_plugin_session
