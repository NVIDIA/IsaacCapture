// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
 * @brief Plugin session that forwards every channel over TCP to a remote_device_bridge.
 *
 * Same contract as OpenXRPluginSession: the owner destroys all channels before
 * the session and may call create_pull_channel() once. After the transport
 * drops, pushes become silent no-ops and channels stay safe to destroy.
 * Each push channel has a latest-wins queue (4 samples / 8 MiB); on a stalled
 * link the oldest samples of that channel are dropped, other channels are
 * unaffected. A channel the bridge refuses (ERROR on its channel id) turns
 * into a no-op; the session and its other channels continue. P0: the pull
 * channel exposes no trackers; wrist poses are the latest bridge samples (no
 * interpolation).
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

    std::unique_ptr<core::IPluginPullChannel> create_pull_channel() override;
    std::unique_ptr<core::ISchemaPushChannel> create_schema_push_channel(const core::SchemaPusherConfig& config) override;
    std::unique_ptr<core::IHandTrackingPushChannel> create_hand_tracking_push_channel(XrHandEXT hand) override;

    //! False once the transport failed; channel creation then throws and pushes are dropped.
    bool connected() const;
    //! Estimated bridge-to-local clock offset in ns (local ~= bridge + offset); 0 before the first frame.
    int64_t clock_offset_ns() const;
    //! Cumulative counters; also logged every 5 s as one "[remote] 5s:" line.
    Stats stats() const;

private:
    std::shared_ptr<detail::RemotePluginSessionImpl> m_impl;
};

} // namespace remote_plugin_session
