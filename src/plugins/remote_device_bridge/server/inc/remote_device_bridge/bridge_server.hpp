// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pusherio/plugin_session.hpp>
#include <remote_plugin_session/bridge_limits.hpp>
#include <remote_plugin_session/tcp_socket.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace remote_device_bridge
{

enum class ClockSource
{
    TwoWay, //!< PING/PONG min-RTT offset; one-way only for a v0 client or when no PONG answers
    OneWay, //!< frame-header minimum only, no PINGs (A/B baseline)
};

struct BridgeOptions
{
    remote_plugin_session::Endpoint listen{ "0.0.0.0", 47500 };
    //! Second listener speaking WebSocket (same frames, same --max-clients budget); off when unset.
    std::optional<remote_plugin_session::Endpoint> ws_listen;
    //! Overrides the HELLO app name when non-empty.
    std::string app_name;
    int wrist_rate_hz = 90;
    int max_clients = 8;
    //! Samples whose mapped time is older than this are dropped instead of pushed; 0 disables.
    int max_sample_age_ms = 250;
    //! OPEN_SCHEMA with a larger max_flatbuffer_size is refused before it reaches the runtime.
    size_t max_schema_bytes = remote_plugin_session::kDefaultMaxSchemaBytes;
    ClockSource clock_source = ClockSource::TwoWay;
    //! One CSV row per pushed hand sample (remote_plugin_session::SampleCsv); off when empty.
    std::string stats_csv;
    //! Period of each client's status line; not a flag (tests shorten it).
    std::chrono::milliseconds stats_period{ 5000 };
};

//! Creates the session a client's channels are mirrored onto. Called in the forked child once HELLO arrived;
//! the bridge binary passes one that builds an OpenXRPluginSession.
using SessionFactory = std::function<std::unique_ptr<core::IPluginSession>(
    const std::string& app_name, const core::PluginSessionRequirements& requirements)>;

/*!
 * @brief Accepts remote plugin sessions and mirrors each onto its own OpenXRPluginSession.
 *
 * The accept loop itself never touches OpenXR or creates threads: every client is served by a forked child
 * process (the OpenXR loader allows one XrInstance per process), and within it every OpenXR call happens on
 * one thread.
 */
class BridgeServer
{
public:
    BridgeServer(BridgeOptions options, SessionFactory session_factory);
    ~BridgeServer();
    BridgeServer(const BridgeServer&) = delete;
    BridgeServer& operator=(const BridgeServer&) = delete;

    //! Bound ports (useful with port 0); ws_port() is 0 without --ws-listen.
    uint16_t port() const;
    uint16_t ws_port() const;

    //! Accept loop; returns once stop becomes true and every client has been reaped.
    void run(const std::atomic<bool>& stop);

    //! Hand push-device owners shared with the forked children (defined in the .cpp).
    struct HandOwners;

private:
    struct ClientRecord;

    void reap_clients(bool wait);
    void admit_client(std::unique_ptr<remote_plugin_session::ITransport> client,
                      const std::string& peer,
                      const std::atomic<bool>& stop);

    BridgeOptions m_options;
    SessionFactory m_session_factory;
    remote_plugin_session::TcpSocket m_listener;
    remote_plugin_session::TcpSocket m_ws_listener;
    HandOwners* m_hand_owners = nullptr;
    std::vector<uint8_t> m_bridge_full_ack;
    std::list<ClientRecord> m_clients;
    uint32_t m_accepted = 0;
};

} // namespace remote_device_bridge
