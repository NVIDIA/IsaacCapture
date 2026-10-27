// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/deviceio_session/remote_session_client.hpp"

#include <log_bridge/logger.hpp>
#include <mcap/sample_stream.hpp>
#include <remote_link/messages.hpp>
#include <remote_link/tcp_transport.hpp>
#include <remote_link/websocket_transport.hpp>

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace core
{

namespace
{

// Link death notifies nobody, so waits wake this often to check it.
constexpr std::chrono::milliseconds kLivenessPoll{ 20 };

constexpr const char* kLostPrefix = "Remote session lost: ";
constexpr const char* kHelloAckLostPrefix = "Remote session ended before HELLO_ACK: ";

std::string to_text(const std::vector<uint8_t>& payload)
{
    return std::string(payload.begin(), payload.end());
}

//! What is left of `timeout` (negative: no limit) since `since`.
std::chrono::milliseconds remaining(std::chrono::milliseconds timeout, std::chrono::steady_clock::time_point since)
{
    if (timeout.count() < 0)
    {
        return timeout;
    }
    const auto spent = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - since);
    return std::max(std::chrono::milliseconds{ 0 }, timeout - spent);
}

} // namespace

struct RemoteSessionClient::State
{
    std::shared_ptr<TrackerSampleStreamReader> stream;
    mutable std::mutex mutex;
    std::condition_variable cv;
    std::optional<std::string> hello_ack;
    std::optional<std::string> status;
    std::optional<std::string> error;
    bool ready = false;
    bool closed = false; //!< close() ran
};

RemoteSessionClient::RemoteSessionClient(std::unique_ptr<ITransport> transport,
                                         std::shared_ptr<TrackerSampleStreamReader> stream,
                                         LinkOptions link_options)
    : state_(std::make_unique<State>())
{
    if (!stream)
    {
        throw std::invalid_argument("RemoteSessionClient: no stream");
    }
    state_->stream = std::move(stream);
    link_options.send_pings = true;
    // A server busy handing over from the previous session accepts the connection before it answers.
    link_options.peer_timeout_from_first_message = true;
    link_options.name = "RemoteSessionClient";
    link_ = std::make_unique<Link>(std::move(transport), std::move(link_options),
                                   [this](ReceivedLinkMessage&& message) { handle(std::move(message)); });
}

RemoteSessionClient::~RemoteSessionClient()
{
    close();
}

std::unique_ptr<RemoteSessionClient> RemoteSessionClient::start(const std::string& endpoint,
                                                                std::string hello_json,
                                                                std::shared_ptr<TrackerSampleStreamReader> stream,
                                                                std::chrono::milliseconds connect_timeout,
                                                                LinkOptions link_options)
{
    if (!stream)
    {
        throw std::invalid_argument("RemoteSessionClient: no stream");
    }
    const RemoteAddress remote = parse_remote(endpoint);
    TcpSocket socket = TcpSocket::connect(remote.endpoint, connect_timeout);
    std::unique_ptr<ITransport> transport;
    WebSocketTransport* upgrade = nullptr;
    if (remote.websocket)
    {
        std::unique_ptr<WebSocketTransport> websocket = WebSocketTransport::start_client(std::move(socket), remote);
        upgrade = websocket.get();
        transport = std::move(websocket);
    }
    else
    {
        transport = std::make_unique<TcpTransport>(std::move(socket));
    }
    std::unique_ptr<RemoteSessionClient> client(
        new RemoteSessionClient(std::move(transport), std::move(stream), std::move(link_options)));
    client->link_->send_reliable(LinkMessageType::Hello, std::vector<uint8_t>(hello_json.begin(), hello_json.end()));
    if (upgrade != nullptr)
    {
        client->upgrade_ = upgrade;
    }
    else
    {
        client->link_->start();
    }
    return client;
}

std::unique_ptr<RemoteSessionClient> RemoteSessionClient::connect(const std::string& endpoint,
                                                                  std::string hello_json,
                                                                  std::shared_ptr<TrackerSampleStreamReader> stream,
                                                                  std::chrono::milliseconds timeout,
                                                                  LinkOptions link_options)
{
    const auto begin = std::chrono::steady_clock::now();
    auto client = start(endpoint, std::move(hello_json), std::move(stream), timeout, std::move(link_options));
    if (!client->wait_hello_ack(remaining(timeout, begin)))
    {
        throw RemoteConnectTimeoutError("No HELLO_ACK from the Teleop Session Server before the connect timeout");
    }
    isaaccapture::Logger::get("isaaccapture.core.RemoteSessionClient")
        ->info("RemoteSessionClient: connected to {}: {}", endpoint, client->hello_ack());
    return client;
}

std::unique_ptr<RemoteSessionClient> RemoteSessionClient::connect(std::unique_ptr<ITransport> transport,
                                                                  std::string hello_json,
                                                                  std::shared_ptr<TrackerSampleStreamReader> stream,
                                                                  std::chrono::milliseconds timeout,
                                                                  LinkOptions link_options)
{
    std::unique_ptr<RemoteSessionClient> client(
        new RemoteSessionClient(std::move(transport), std::move(stream), std::move(link_options)));
    client->link_->send_reliable(LinkMessageType::Hello, std::vector<uint8_t>(hello_json.begin(), hello_json.end()));
    client->link_->start();
    if (!client->wait_hello_ack(timeout))
    {
        throw RemoteConnectTimeoutError("No HELLO_ACK from the Teleop Session Server before the connect timeout");
    }
    return client;
}

bool RemoteSessionClient::wait_hello_ack(std::chrono::milliseconds timeout)
{
    const auto begin = std::chrono::steady_clock::now();
    if (!finish_upgrade(timeout))
    {
        return false;
    }
    return wait_for(remaining(timeout, begin), kHelloAckLostPrefix, [this] { return state_->hello_ack.has_value(); });
}

bool RemoteSessionClient::finish_upgrade(std::chrono::milliseconds timeout)
{
    std::lock_guard<std::mutex> lock(upgrade_mutex_);
    WebSocketTransport* const upgrade = upgrade_.load();
    if (upgrade == nullptr)
    {
        return true;
    }
    throw_if_failed(kHelloAckLostPrefix);
    bool upgraded = false;
    try
    {
        upgraded = upgrade->wait_upgraded(timeout);
    }
    catch (const std::runtime_error&)
    {
        // close() shuts the transport down, which breaks the answer: report the close.
        throw_if_failed(kHelloAckLostPrefix);
        throw;
    }
    if (!upgraded)
    {
        return false;
    }
    // Under the state mutex, so a close() either sees the started link or happened before this check.
    std::lock_guard<std::mutex> state_lock(state_->mutex);
    if (state_->closed)
    {
        throw std::runtime_error(std::string(kHelloAckLostPrefix) + "closed");
    }
    upgrade_ = nullptr;
    link_->start();
    return true;
}

void RemoteSessionClient::handle(ReceivedLinkMessage&& message)
{
    switch (message.type)
    {
    case LinkMessageType::Setup:
        state_->stream->add_setup(message.payload);
        return;
    case LinkMessageType::Samples:
    {
        const SamplesPrefix prefix = decode_samples_prefix(message.payload);
        const ClockEstimate clock = link_->clock();
        // Every received header adds a clock sample before the handler runs, so this does not happen.
        if (!clock.valid)
        {
            return;
        }
        state_->stream->add_samples(std::span<const uint8_t>(message.payload).subspan(kSamplesPrefixSize),
                                    prefix.tick_seq, prefix.tick_time_ns + clock.offset_ns);
        return;
    }
    case LinkMessageType::HelloAck:
    case LinkMessageType::Status:
    case LinkMessageType::Ready:
    case LinkMessageType::Error:
        break;
    default:
        return; // messages of later protocol versions
    }

    std::string text = to_text(message.payload);
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        switch (message.type)
        {
        case LinkMessageType::HelloAck:
            state_->hello_ack = std::move(text);
            break;
        case LinkMessageType::Ready:
            state_->ready = true;
            state_->status = std::move(text);
            break;
        case LinkMessageType::Status:
            state_->status = std::move(text);
            break;
        default: // Error: the first one is the cause
            if (!state_->error)
            {
                state_->error = text.empty() ? "The Teleop Session Server ended the session" : std::move(text);
            }
            break;
        }
    }
    state_->cv.notify_all();
}

void RemoteSessionClient::throw_if_failed(const std::string& lost_prefix) const
{
    // Read before the error: by the time the link is down, an ERROR received before that is stored.
    const bool alive = link_->alive();
    const std::string reason = alive ? std::string() : link_->close_reason();
    const bool unanswered = !alive && link_->peer_closed_before_any_message();
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->error)
    {
        throw std::runtime_error(*state_->error);
    }
    if (unanswered)
    {
        throw PeerClosedError(lost_prefix + reason);
    }
    if (!alive)
    {
        throw std::runtime_error(lost_prefix + reason);
    }
}

void RemoteSessionClient::throw_if_failed() const
{
    throw_if_failed(kLostPrefix);
    if (upgrade_.load() != nullptr)
    {
        // Nothing else completes it: until then no HELLO is sent and nothing is received.
        throw std::runtime_error("The WebSocket upgrade is not complete yet; call wait_hello_ack() or wait_ready() first");
    }
}

template <typename Done>
bool RemoteSessionClient::wait_for(std::chrono::milliseconds timeout, const std::string& lost_prefix, Done done) const
{
    const bool bounded = timeout.count() >= 0;
    const auto deadline = std::chrono::steady_clock::now() + (bounded ? timeout : std::chrono::milliseconds{ 0 });
    while (true)
    {
        throw_if_failed(lost_prefix);
        std::unique_lock<std::mutex> lock(state_->mutex);
        if (done())
        {
            return true;
        }
        auto wait = kLivenessPoll;
        if (bounded)
        {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline)
            {
                return false;
            }
            wait = std::min(wait, std::chrono::ceil<std::chrono::milliseconds>(deadline - now));
        }
        state_->cv.wait_for(lock, wait);
    }
}

RemoteReadyState RemoteSessionClient::wait_ready(std::chrono::milliseconds timeout)
{
    const auto begin = std::chrono::steady_clock::now();
    if (!finish_upgrade(timeout))
    {
        return RemoteReadyState::Pending;
    }
    return wait_for(remaining(timeout, begin), kLostPrefix, [this] { return state_->ready; }) ?
               RemoteReadyState::Ready :
               RemoteReadyState::Pending;
}

std::optional<std::string> RemoteSessionClient::latest_status() const
{
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->status;
}

std::string RemoteSessionClient::hello_ack() const
{
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->hello_ack.value_or(std::string());
}

bool RemoteSessionClient::alive() const
{
    return link_->alive();
}

ClockEstimate RemoteSessionClient::clock() const
{
    return link_->clock();
}

void RemoteSessionClient::close()
{
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->closed = true;
    }
    state_->cv.notify_all();
    link_->close();
}

} // namespace core
