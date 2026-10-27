// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "deviceio_session.hpp"

#include <remote_link/link.hpp>
#include <remote_link/transport.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace core
{

class TrackerSampleStreamWriter;

/*!
 * @brief Merges an unsent SAMPLES payload with a newer one: the newer prefix, and per MCAP channel the newest record.
 *
 * SAMPLES payloads are deltas, so a channel only `older` wrote keeps its record. Throws on a malformed payload.
 */
std::vector<uint8_t> merge_samples_payloads(std::vector<uint8_t> older, std::vector<uint8_t> newer);

/*!
 * @brief The server end of a remote session: owns the Link, the sample stream writer and the DeviceIOSession.
 *
 * tick() records one DeviceIOSession update and sends it as one SAMPLES link message; one the link could not send
 * yet is merged into the next one. Call start_session(), tick() and close() from one thread; the other methods are
 * thread-safe.
 */
class RemoteSessionHost
{
public:
    //! Adopts an accepted descriptor and starts the Link, which answers PINGs and sends heartbeats. A WebSocket
    //! upgrade runs on the link's receive thread.
    static std::unique_ptr<RemoteSessionHost> adopt(int fd, bool websocket, LinkOptions link_options = {});
    //! The same over a connected transport; send_pings and name of `link_options` are set here.
    static std::unique_ptr<RemoteSessionHost> adopt(std::unique_ptr<ITransport> transport, LinkOptions link_options = {});

    ~RemoteSessionHost();
    RemoteSessionHost(const RemoteSessionHost&) = delete;
    RemoteSessionHost& operator=(const RemoteSessionHost&) = delete;

    //! The robot's HELLO JSON, or nullopt after `timeout` (negative: no limit). Throws std::runtime_error once the
    //! link is lost.
    std::optional<std::string> wait_hello(std::chrono::milliseconds timeout);

    void send_hello_ack(std::string json);
    void send_status(std::string json);
    void send_ready(std::string json);
    //! Queued like the other messages; close() delivers it before ending the connection.
    void send_error(std::string text);

    /*!
     * @brief Creates the DeviceIOSession, recording `tracker_names` into the stream, which keeps only `topics`
     *        (empty: every topic).
     * @throws std::logic_error when a session exists; whatever DeviceIOSession::run() throws.
     */
    void start_session(const std::vector<std::shared_ptr<ITracker>>& trackers,
                       const std::vector<std::pair<const ITracker*, std::string>>& tracker_names,
                       const OpenXRSessionHandles& handles,
                       const VendorConfig& vendor_config,
                       const std::vector<std::string>& topics);

    //! One DeviceIOSession::update(), sent as one SAMPLES link message after a SETUP for the channels it declared.
    //! Sends nothing once the link is down. Throws what update() throws.
    void tick();

    bool alive() const;
    //! Empty while alive.
    std::string close_reason() const;
    //! Idempotent: ends the DeviceIOSession, then closes the link so that the robot reads every queued message.
    void close();

private:
    struct State;

    RemoteSessionHost(std::unique_ptr<ITransport> transport, LinkOptions link_options);

    void handle(ReceivedLinkMessage&& message);
    void send_text(LinkMessageType type, const std::string& text);

    // The writer outlives the session recording into it.
    std::unique_ptr<TrackerSampleStreamWriter> writer_;
    std::unique_ptr<DeviceIOSession> session_;
    uint64_t next_tick_seq_ = 1;
    std::unique_ptr<State> state_;
    // Declared last, so it is closed (its threads joined) before the state its handler writes goes away.
    std::unique_ptr<Link> link_;
};

} // namespace core
