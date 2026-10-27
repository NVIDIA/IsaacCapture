// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "clock_offset.hpp"
#include "messages.hpp"
#include "transport.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace core
{

struct LinkOptions
{
    std::chrono::milliseconds heartbeat_period{ 500 }; //!< HEARTBEAT after this long without any send
    std::chrono::milliseconds peer_timeout{ 3000 }; //!< dead after this long without any receive
    std::chrono::milliseconds receive_poll{ 100 };
    std::chrono::milliseconds close_drain{ 500 }; //!< how long close() lets queued reliable messages go out
    bool send_pings = false; //!< drive a PingScheduler for a two-way clock estimate
    //! peer_timeout and the PING schedule start with the first received message instead of start(): until then,
    //! including the transport handshake, only close() ends the wait.
    bool peer_timeout_from_first_message = false;
    std::string name = "link"; //!< prefix for log lines
};

struct ReceivedLinkMessage
{
    LinkMessageType type; //!< may hold a value outside the enum
    std::vector<uint8_t> payload;
    int64_t remote_send_ns; //!< header t_send, peer clock
    int64_t local_receive_ns;
};

//! Runs on the receive thread; must not block or call into Python. An exception kills the link.
using LinkMessageHandler = std::function<void(ReceivedLinkMessage&&)>;
//! Combines an unsent queued payload with a newer one; runs on the send_latest() caller's thread.
using MergeFn = std::function<std::vector<uint8_t>(std::vector<uint8_t> queued, std::vector<uint8_t> incoming)>;

struct ClockEstimate
{
    int64_t offset_ns = 0; //!< local - remote
    bool two_way = false;
    bool valid = false; //!< any sample yet
    int64_t rtt_min_ns = -1; //!< -1 until two_way
};

struct LinkStats
{
    uint64_t messages_sent = 0; //!< including HEARTBEAT, PING and PONG
    uint64_t messages_received = 0;
    uint64_t latest_replaced = 0; //!< unsent latest-wins payloads replaced or merged
    uint64_t unknown_received = 0; //!< types outside LinkMessageType, passed to the handler
};

/*!
 * @brief One connection: a sender and a receiver thread over an ITransport, heartbeats, liveness and clock offset.
 *
 * Reliable messages go out in FIFO order and are never dropped while the link lives. Latest-wins messages keep one
 * unsent slot per type and are sent only while the transport is writable, never ahead of a reliable message queued
 * before them. PING/PONG and HEARTBEAT are handled
 * here and never reach the handler. A dead link stays dead; nothing reconnects.
 */
class Link
{
public:
    Link(std::unique_ptr<ITransport> transport, LinkOptions options, LinkMessageHandler handler);
    ~Link();
    Link(const Link&) = delete;
    Link& operator=(const Link&) = delete;

    //! Starts the threads. The sender waits for transport->wait_ready(), which the receiver runs first.
    void start();

    //! Queued before start() is allowed. Throws std::invalid_argument above kMaxPayloadSize; ignored once dead.
    void send_reliable(LinkMessageType type, std::vector<uint8_t> payload);
    //! Replaces the unsent payload of this type, or merges with it when `merge` is set. If `merge` throws, both
    //! payloads are dropped and the exception propagates.
    void send_latest(LinkMessageType type, std::vector<uint8_t> payload, MergeFn merge = {});

    //! False once the link died or close() ran.
    bool alive() const;
    //! Empty while alive.
    std::string close_reason() const;
    //! True once the link died because the connection broke (the peer closed or reset it, or a send failed) before
    //! any byte of a message arrived. False while alive, after close(), and for any other cause, a message cut off
    //! included.
    bool peer_closed_before_any_message() const;
    ClockEstimate clock() const;
    LinkStats stats() const;

    //! Idempotent: sends queued reliable messages (once the transport handshake is done) for up to close_drain,
    //! shuts the transport down, joins.
    void close();
    /*!
     * @brief close(), but the peer reads everything sent: after the drain the transport is half-closed and input is
     *        read and discarded until the peer closes or `linger` passes.
     *
     * Closing a TCP socket with unread input resets the connection, which can discard data the peer has not read,
     * such as a final ERROR.
     */
    void close_gracefully(std::chrono::milliseconds linger);

private:
    struct LatestSlot
    {
        uint8_t type;
        std::optional<std::vector<uint8_t>> payload;
    };

    void sender_loop();
    void receiver_loop();
    void handle(std::vector<uint8_t>& message, int64_t now_ns);
    ReceiveResult wait_transport_ready();
    void start_pings(int64_t now_ns);
    bool send_now(std::vector<uint8_t>& message);
    //! `connection_broke`: end of stream or a reset between messages, or a failed send, as opposed to a message cut
    //! off, malformed data or silence.
    void fail(const std::string& reason, bool connection_broke = false);
    void enqueue_reliable(std::vector<uint8_t> message);
    void close_impl(std::optional<std::chrono::milliseconds> linger);

    std::unique_ptr<ITransport> m_transport;
    const LinkOptions m_options;
    LinkMessageHandler m_handler;

    mutable std::mutex m_mutex;
    std::condition_variable m_cv;
    std::deque<std::vector<uint8_t>> m_reliable;
    std::vector<LatestSlot> m_latest;
    size_t m_latest_cursor = 0;
    bool m_started = false;
    bool m_ready = false; //!< transport handshake done
    bool m_closing = false;
    bool m_lingering = false; //!< close_gracefully(): half-closed, input is discarded
    bool m_sender_done = false;
    bool m_pings_started = false; //!< wakes the sender for a PING schedule the receiver started
    bool m_closed = false; //!< close() finished
    bool m_alive = true;
    std::string m_close_reason;
    bool m_peer_closed_before_any_message = false;
    LinkStats m_stats;

    mutable std::mutex m_clock_mutex;
    ClockOffsetEstimator m_clock;
    std::optional<PingScheduler> m_pings;

    std::atomic<int64_t> m_last_receive_ns{ 0 };
    int64_t m_last_send_ns = 0; //!< sender thread only
    uint32_t m_seq = 0; //!< sender thread only

    std::mutex m_close_mutex;
    std::thread m_sender;
    std::thread m_receiver;
};

} // namespace core
