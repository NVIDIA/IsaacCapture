// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <remote_link/link.hpp>
#include <remote_link/transport.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

namespace core
{

class TrackerSampleStreamReader;
class WebSocketTransport;

enum class RemoteReadyState
{
    Pending,
    Ready,
};

//! No HELLO_ACK arrived before the connect timeout. The Python bindings raise it as TimeoutError.
class RemoteConnectTimeoutError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

/*!
 * @brief The robot end of a remote session: a Link to a Teleop Session Server that feeds a sample stream.
 *
 * SETUP and SAMPLES link messages go into `stream`, which a RemoteSession reads; tick times are mapped onto this
 * machine's clock. Every method is thread-safe. Once the server's ERROR text arrived, every failure reports it,
 * even after the link died for another reason. The link's peer timeout applies from the server's first message, so
 * a server that accepted the connection but has not taken it up yet is waited for.
 */
class RemoteSessionClient
{
public:
    /*!
     * @brief Connects to `endpoint` (see parse_remote()) and queues `hello_json` as HELLO; wait_hello_ack() waits
     *        for the answer. Over ws:// the HELLO goes out once wait_hello_ack() or wait_ready() completed the
     *        WebSocket upgrade.
     * @param connect_timeout Bounds the TCP connect, not the host name lookup; negative waits without limit.
     * @param link_options Link timing; send_pings, peer_timeout_from_first_message and name are set here.
     * @throws std::invalid_argument for a malformed endpoint or no stream; std::runtime_error when connecting fails.
     *         Its text starts with "cannot connect to " when nothing accepted the connection, and it is a
     *         TcpConnectTimeoutError when the connect timed out, a PeerClosedError when the ws:// upgrade request
     *         could not be sent.
     */
    static std::unique_ptr<RemoteSessionClient> start(const std::string& endpoint,
                                                      std::string hello_json,
                                                      std::shared_ptr<TrackerSampleStreamReader> stream,
                                                      std::chrono::milliseconds connect_timeout,
                                                      LinkOptions link_options = {});

    /*!
     * @brief start(), then wait_hello_ack().
     * @param timeout Bounds connecting and the wait for HELLO_ACK together; negative waits without limit.
     * @throws as start(); RemoteConnectTimeoutError when no HELLO_ACK arrives in time; as wait_hello_ack().
     */
    static std::unique_ptr<RemoteSessionClient> connect(const std::string& endpoint,
                                                        std::string hello_json,
                                                        std::shared_ptr<TrackerSampleStreamReader> stream,
                                                        std::chrono::milliseconds timeout,
                                                        LinkOptions link_options = {});

    //! The same over a transport that is already connected.
    static std::unique_ptr<RemoteSessionClient> connect(std::unique_ptr<ITransport> transport,
                                                        std::string hello_json,
                                                        std::shared_ptr<TrackerSampleStreamReader> stream,
                                                        std::chrono::milliseconds timeout,
                                                        LinkOptions link_options = {});

    ~RemoteSessionClient();
    RemoteSessionClient(const RemoteSessionClient&) = delete;
    RemoteSessionClient& operator=(const RemoteSessionClient&) = delete;

    /*!
     * @brief True once HELLO_ACK arrived; false after `timeout` (negative: no limit). A ws:// connection completes
     *        its WebSocket upgrade here first.
     * @throws std::runtime_error with the server's ERROR text, a failed upgrade (its text starts with
     *         "WebSocket upgrade to "), or "Remote session ended before HELLO_ACK: <reason>". It is a
     *         PeerClosedError when the server closed or reset the connection before sending any byte of a message
     *         (over ws://, also before answering the upgrade), e.g. one that crashed or was killed before it took the
     *         connection up; start() may then succeed.
     */
    bool wait_hello_ack(std::chrono::milliseconds timeout);

    //! Ready once READY arrived; Pending after `timeout` (negative: no limit). Completes a pending WebSocket
    //! upgrade first, as wait_hello_ack() does. Throws as wait_hello_ack() while upgrading, then as throw_if_failed().
    RemoteReadyState wait_ready(std::chrono::milliseconds timeout);
    //! The newest STATUS or READY JSON.
    std::optional<std::string> latest_status() const;
    std::string hello_ack() const;
    //! Throws std::runtime_error with the server's ERROR text, or "Remote session lost: <reason>" once the link is
    //! down (close() included), or while the WebSocket upgrade waits for wait_hello_ack() or wait_ready().
    void throw_if_failed() const;
    //! False once the link is down (close() included).
    bool alive() const;
    ClockEstimate clock() const;
    //! Idempotent; the stream receives nothing after it returns.
    void close();

private:
    struct State;

    RemoteSessionClient(std::unique_ptr<ITransport> transport,
                        std::shared_ptr<TrackerSampleStreamReader> stream,
                        LinkOptions link_options);

    void handle(ReceivedLinkMessage&& message);
    //! Completes a pending WebSocket upgrade within `timeout`, then starts the link; false while it is pending.
    bool finish_upgrade(std::chrono::milliseconds timeout);
    void throw_if_failed(const std::string& lost_prefix) const;
    //! Waits for `done` (checked under the state mutex) until `timeout`; throws on failure.
    template <typename Done>
    bool wait_for(std::chrono::milliseconds timeout, const std::string& lost_prefix, Done done) const;

    std::unique_ptr<State> state_;
    std::mutex upgrade_mutex_;
    //! A ws:// transport (owned by link_) whose upgrade is not complete; the link starts once it is. Changed under
    //! upgrade_mutex_.
    std::atomic<WebSocketTransport*> upgrade_{ nullptr };
    // Declared last, so it is closed (its threads joined) before the state its handler writes goes away.
    std::unique_ptr<Link> link_;
};

} // namespace core
