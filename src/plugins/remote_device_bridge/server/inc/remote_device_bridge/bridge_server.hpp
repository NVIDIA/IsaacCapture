// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pusherio/plugin_session.hpp>
#include <remote_plugin_session/bridge_limits.hpp>
#include <remote_plugin_session/remote_address.hpp>
#include <remote_plugin_session/tcp_socket.hpp>
#include <remote_plugin_session/transport.hpp>

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

class SampleCsv;

namespace detail
{
//! Hand push-device owners shared with the forked children (defined in the .cpp).
struct HandOwners;
struct HandOwnersUnmap
{
    void operator()(HandOwners* owners) const noexcept;
};
} // namespace detail

enum class ClockSource
{
    TwoWay, //!< PING/PONG min-RTT offset; one-way only when no PONG answers
    OneWay, //!< frame-header minimum only, no PINGs (A/B baseline)
};

struct BridgeOptions
{
    remote_plugin_session::Endpoint listen{ "0.0.0.0", remote_plugin_session::kDefaultBridgePort };
    //! Second listener speaking WebSocket (same frames, same --max-clients budget).
    std::optional<remote_plugin_session::Endpoint> ws_listen;
    //! Overrides the HELLO app name.
    std::optional<std::string> app_name;
    int wrist_rate_hz = 90;
    int max_clients = 8;
    //! Samples whose mapped time is older than this are dropped instead of pushed.
    std::optional<std::chrono::milliseconds> max_sample_age{ 250 };
    //! OPEN_SCHEMA with a larger max_flatbuffer_size is refused before it reaches the runtime.
    size_t max_schema_bytes = remote_plugin_session::kDefaultMaxSchemaBytes;
    ClockSource clock_source = ClockSource::TwoWay;
    //! One CSV row per pushed hand sample (SampleCsv).
    std::optional<std::string> stats_csv;
    //! Period of each client's status line; not a flag (tests shorten it).
    std::chrono::milliseconds stats_period{ 5000 };
};

//! Returns the factory for a client's session, given the app name; called in the forked child once HELLO arrived.
using SessionFactory = std::function<core::PluginSessionFactory(const std::string& app_name)>;

/*!
 * @brief Accepts remote plugin sessions and mirrors each onto a session from the SessionFactory.
 *
 * Every client is served by its own forked child process (see this plugin's AGENTS.md for why).
 */
class BridgeServer
{
public:
    BridgeServer(BridgeOptions options, SessionFactory session_factory);
    ~BridgeServer();

    //! Bound ports (useful with port 0).
    uint16_t port() const;
    std::optional<uint16_t> ws_port() const;

    //! Accept loop; returns once stop becomes true and every client has been reaped.
    void run(const std::atomic<bool>& stop);

private:
    struct ClientRecord;

    //! Logs and forgets clients that have exited, without blocking.
    void reap_finished();
    //! Asks every child to stop and blocks until all are reaped (SIGKILL after a grace period).
    void stop_and_reap_all();
    void on_reaped(const ClientRecord& client, int status);
    //! Frees every hand `client` claimed.
    void release_hands(const ClientRecord& client);
    void admit_client(std::unique_ptr<remote_plugin_session::ITransport> client,
                      const std::string& peer,
                      const std::atomic<bool>& stop);

    BridgeOptions m_options;
    SessionFactory m_session_factory;
    remote_plugin_session::TcpSocket m_listener;
    remote_plugin_session::TcpSocket m_ws_listener;
    std::unique_ptr<detail::HandOwners, detail::HandOwnersUnmap> m_hand_owners;
    std::unique_ptr<SampleCsv> m_csv; //!< --stats-csv, opened before the first fork
    std::list<ClientRecord> m_clients;
    uint32_t m_accepted = 0;
};

} // namespace remote_device_bridge
