// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/remote_link/link.hpp"

#include "inc/remote_link/message_header.hpp"

#include <log_bridge/logger.hpp>
#include <oxr_utils/os_time.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace core
{

namespace
{

// Upper bound on a sender wait while latest-wins messages wait for the transport to drain.
constexpr std::chrono::milliseconds kWritablePoll{ 2 };
// How long a failed send leaves the receiver to end the link: on a broken connection it does so at once.
constexpr std::chrono::milliseconds kSendFailureVerdict{ 250 };

int64_t to_ns(std::chrono::milliseconds duration)
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
}

std::shared_ptr<spdlog::logger> logger()
{
    return isaaccapture::Logger::get("isaaccapture.remote_link.Link");
}

const char* to_text(ReceiveResult result)
{
    switch (result)
    {
    case ReceiveResult::Ok:
        return "ok";
    case ReceiveResult::Timeout:
        return "timed out";
    case ReceiveResult::Closed:
        return "connection closed";
    case ReceiveResult::Malformed:
        return "malformed data";
    case ReceiveResult::Truncated:
        return "incomplete message";
    }
    return "unknown";
}

bool is_known(uint8_t type)
{
    switch (static_cast<LinkMessageType>(type))
    {
    case LinkMessageType::Hello:
    case LinkMessageType::HelloAck:
    case LinkMessageType::Heartbeat:
    case LinkMessageType::Error:
    case LinkMessageType::Ping:
    case LinkMessageType::Pong:
    case LinkMessageType::Status:
    case LinkMessageType::Setup:
    case LinkMessageType::Samples:
    case LinkMessageType::Ready:
        return true;
    }
    return false;
}

} // namespace

Link::Link(std::unique_ptr<ITransport> transport, LinkOptions options, LinkMessageHandler handler)
    : m_transport(std::move(transport)), m_options(std::move(options)), m_handler(std::move(handler))
{
    if (!m_transport)
    {
        throw std::invalid_argument("Link requires a transport");
    }
    if (m_options.heartbeat_period.count() <= 0 || m_options.peer_timeout.count() <= 0 ||
        m_options.receive_poll.count() <= 0 || m_options.close_drain.count() < 0)
    {
        throw std::invalid_argument("Link options require positive periods");
    }
}

Link::~Link()
{
    close();
}

void Link::start()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_started)
    {
        throw std::logic_error("Link::start called twice");
    }
    if (m_closing || !m_alive)
    {
        throw std::logic_error("Link::start called after close");
    }
    m_started = true;
    const int64_t now = os_monotonic_now_ns();
    m_last_receive_ns = now;
    m_last_send_ns = now;
    if (m_options.send_pings && !m_options.peer_timeout_from_first_message)
    {
        std::lock_guard<std::mutex> clock_lock(m_clock_mutex);
        m_pings.emplace(now);
    }
    m_receiver = std::thread(&Link::receiver_loop, this);
    m_sender = std::thread(&Link::sender_loop, this);
}

void Link::send_reliable(LinkMessageType type, std::vector<uint8_t> payload)
{
    enqueue_reliable(encode_link_message(static_cast<uint8_t>(type), payload));
}

void Link::enqueue_reliable(std::vector<uint8_t> message)
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_alive || m_closing)
        {
            return;
        }
        m_reliable.push_back(std::move(message));
    }
    m_cv.notify_all();
}

void Link::send_latest(LinkMessageType type, std::vector<uint8_t> payload, MergeFn merge)
{
    if (payload.size() > kMaxPayloadSize)
    {
        throw std::invalid_argument("remote link: payload of " + std::to_string(payload.size()) +
                                    " bytes exceeds the limit");
    }
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_alive || m_closing)
        {
            return;
        }
        const uint8_t raw_type = static_cast<uint8_t>(type);
        auto slot = std::find_if(
            m_latest.begin(), m_latest.end(), [raw_type](const LatestSlot& s) { return s.type == raw_type; });
        if (slot == m_latest.end())
        {
            m_latest.push_back(LatestSlot{ raw_type, std::nullopt });
            slot = std::prev(m_latest.end());
        }
        if (slot->payload)
        {
            ++m_stats.latest_replaced;
            if (merge)
            {
                std::vector<uint8_t> queued = std::move(*slot->payload);
                slot->payload.reset();
                payload = merge(std::move(queued), std::move(payload));
                if (payload.size() > kMaxPayloadSize)
                {
                    throw std::invalid_argument("remote link: merged payload exceeds the limit");
                }
            }
        }
        slot->payload = std::move(payload);
    }
    m_cv.notify_all();
}

bool Link::alive() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_alive;
}

std::string Link::close_reason() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_close_reason;
}

bool Link::peer_closed_before_any_message() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_peer_closed_before_any_message;
}

ClockEstimate Link::clock() const
{
    std::lock_guard<std::mutex> lock(m_clock_mutex);
    ClockEstimate estimate;
    estimate.valid = m_clock.has_sample();
    estimate.two_way = m_clock.two_way();
    estimate.offset_ns = m_clock.offset_ns();
    estimate.rtt_min_ns = estimate.two_way ? m_clock.rtt_min_ns() : -1;
    return estimate;
}

LinkStats Link::stats() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_stats;
}

void Link::fail(const std::string& reason, bool connection_broke)
{
    bool closing = false;
    bool unanswered = false;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_alive)
        {
            return;
        }
        m_alive = false;
        // While lingering, the peer closing (or going quiet) is the expected end.
        m_close_reason = m_lingering ? "closed" : reason;
        closing = m_closing;
        unanswered = connection_broke && !closing && m_stats.messages_received == 0;
        m_peer_closed_before_any_message = unanswered;
    }
    m_transport->shutdown();
    m_cv.notify_all();
    if (unanswered)
    {
        // The owner reports it: a robot app connects again to a server that closed before answering.
        logger()->debug("[{}] link lost before any message: {}", m_options.name, reason);
    }
    else if (!closing)
    {
        logger()->warn("[{}] link lost: {}", m_options.name, reason);
    }
}

bool Link::send_now(std::vector<uint8_t>& message)
{
    const int64_t now = os_monotonic_now_ns();
    stamp_header(message.data(), m_seq++, now);
    if (!m_transport->send(message.data(), message.size()))
    {
        {
            // The receiver reads what the peer sent before the connection broke, so it tells a peer that sent part
            // of a message from one that sent nothing.
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait_for(lock, kSendFailureVerdict, [this] { return !m_alive || m_closing; });
        }
        // Counted as a broken connection: the other cause, a send stalled past the send timeout, needs the peer
        // to leave a full socket buffer unread.
        fail("send failed", true);
        return false;
    }
    m_last_send_ns = now;
    std::lock_guard<std::mutex> lock(m_mutex);
    ++m_stats.messages_sent;
    return true;
}

void Link::sender_loop()
{
    std::unique_lock<std::mutex> lock(m_mutex);
    // Also when closing: what is queued goes out once the handshake is done; close() bounds the wait by close_drain.
    m_cv.wait(lock, [this] { return !m_alive || m_ready; });
    // Set when the transport was not writable: latest-wins items then wait kWritablePoll before the next try.
    bool latest_blocked = false;
    while (m_alive && m_ready)
    {
        const auto latest_pending = [this] {
            return std::any_of(
                m_latest.begin(), m_latest.end(), [](const LatestSlot& s) { return s.payload.has_value(); });
        };
        int64_t due_ns = m_last_send_ns + to_ns(m_options.heartbeat_period);
        {
            std::lock_guard<std::mutex> clock_lock(m_clock_mutex);
            if (m_pings)
            {
                due_ns = std::min(due_ns, m_pings->next_due_ns());
            }
        }
        auto wait = std::chrono::nanoseconds(std::max<int64_t>(0, due_ns - os_monotonic_now_ns()));
        if (latest_blocked)
        {
            wait = std::min<std::chrono::nanoseconds>(wait, kWritablePoll);
        }
        m_cv.wait_for(lock, wait,
                      [&] {
                          return !m_alive || m_closing || !m_reliable.empty() || m_pings_started ||
                                 (!latest_blocked && latest_pending());
                      });
        m_pings_started = false;
        if (!m_alive)
        {
            break;
        }
        std::deque<std::vector<uint8_t>> reliable;
        reliable.swap(m_reliable);
        const bool closing = m_closing;
        const bool has_latest = !closing && latest_pending();
        lock.unlock();

        bool ok = true;
        for (auto& message : reliable)
        {
            if (!(ok = send_now(message)))
            {
                break;
            }
        }
        if (closing)
        {
            // Latest-wins items are dropped on close; only the reliable queue drains.
            lock.lock();
            if (!ok || m_reliable.empty())
            {
                break;
            }
            continue;
        }

        latest_blocked = false;
        if (ok && has_latest)
        {
            if (!m_transport->wait_writable(std::chrono::milliseconds{ 0 }))
            {
                latest_blocked = true;
            }
            else
            {
                std::vector<std::vector<uint8_t>> latest;
                {
                    std::lock_guard<std::mutex> relock(m_mutex);
                    // A reliable message queued since the swap goes first: it may be what the latest payload
                    // depends on (SETUP before the SAMPLES that use its channels).
                    const size_t count = m_closing || !m_reliable.empty() ? 0 : m_latest.size();
                    for (size_t i = 0; i < count; ++i)
                    {
                        LatestSlot& slot = m_latest[(m_latest_cursor + i) % count];
                        if (slot.payload)
                        {
                            latest.push_back(encode_link_message(slot.type, *slot.payload));
                            slot.payload.reset();
                        }
                    }
                    if (count > 0)
                    {
                        m_latest_cursor = (m_latest_cursor + 1) % count;
                    }
                }
                for (auto& message : latest)
                {
                    if (!(ok = send_now(message)))
                    {
                        break;
                    }
                }
            }
        }

        if (ok)
        {
            PingScheduler::Action action = PingScheduler::Action::None;
            {
                std::lock_guard<std::mutex> clock_lock(m_clock_mutex);
                if (m_pings)
                {
                    action = m_pings->poll(os_monotonic_now_ns());
                }
            }
            if (action == PingScheduler::Action::Ping)
            {
                std::vector<uint8_t> ping = encode_link_message(static_cast<uint8_t>(LinkMessageType::Ping), {});
                ok = send_now(ping);
            }
            else if (action == PingScheduler::Action::GiveUp)
            {
                logger()->info("[{}] no PONG from the peer; keeping the one-way clock estimate", m_options.name);
            }
        }

        if (ok && os_monotonic_now_ns() - m_last_send_ns >= to_ns(m_options.heartbeat_period))
        {
            std::vector<uint8_t> heartbeat = encode_link_message(static_cast<uint8_t>(LinkMessageType::Heartbeat), {});
            send_now(heartbeat);
        }
        lock.lock();
    }
    m_sender_done = true;
    lock.unlock();
    m_cv.notify_all();
}

ReceiveResult Link::wait_transport_ready()
{
    if (!m_options.peer_timeout_from_first_message)
    {
        return m_transport->wait_ready(m_options.peer_timeout);
    }
    // No limit: the handshake ends with the peer's answer, a failure, or close() shutting the transport down.
    while (true)
    {
        const ReceiveResult result = m_transport->wait_ready(m_options.receive_poll);
        if (result != ReceiveResult::Timeout || !alive())
        {
            return result;
        }
    }
}

void Link::start_pings(int64_t now_ns)
{
    {
        std::lock_guard<std::mutex> clock_lock(m_clock_mutex);
        m_pings.emplace(now_ns);
    }
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_pings_started = true;
    }
    m_cv.notify_all();
}

void Link::receiver_loop()
{
    const ReceiveResult ready = wait_transport_ready();
    if (ready != ReceiveResult::Ok)
    {
        fail(std::string("transport handshake failed: ") + to_text(ready), ready == ReceiveResult::Closed);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_ready = true;
    }
    m_cv.notify_all();
    m_last_receive_ns = os_monotonic_now_ns();

    // Whether peer_timeout applies yet.
    bool received = !m_options.peer_timeout_from_first_message;
    std::vector<uint8_t> message;
    while (true)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_alive)
            {
                return;
            }
        }
        const ReceiveResult result = m_transport->receive(message, m_options.receive_poll);
        const int64_t now = os_monotonic_now_ns();
        if (result == ReceiveResult::Timeout)
        {
            if (received && now - m_last_receive_ns > to_ns(m_options.peer_timeout))
            {
                fail("no messages for " + std::to_string((now - m_last_receive_ns) / 1'000'000) + " ms");
                return;
            }
            continue;
        }
        if (result != ReceiveResult::Ok)
        {
            fail(std::string("receive failed: ") + to_text(result), result == ReceiveResult::Closed);
            return;
        }
        if (!received)
        {
            received = true;
            if (m_options.send_pings)
            {
                start_pings(now);
            }
        }
        try
        {
            handle(message, now);
        }
        catch (const std::exception& e)
        {
            fail(std::string("receive failed: ") + e.what());
            return;
        }
    }
}

void Link::handle(std::vector<uint8_t>& message, int64_t now_ns)
{
    if (message.size() < kHeaderSize)
    {
        throw LinkMessageDecodeError("remote link: message shorter than a header");
    }
    const LinkMessageHeader header = decode_header(message.data());
    if (message.size() != kHeaderSize + header.payload_size)
    {
        throw LinkMessageDecodeError("remote link: message length does not match its header");
    }
    m_last_receive_ns = now_ns;
    const bool known = is_known(header.type);
    bool lingering = false;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        ++m_stats.messages_received;
        if (!known)
        {
            ++m_stats.unknown_received;
        }
        lingering = m_lingering;
    }
    if (lingering)
    {
        return; // close_gracefully() reads only to let the peer finish
    }
    {
        std::lock_guard<std::mutex> lock(m_clock_mutex);
        m_clock.add_sample(header.t_send_ns, now_ns);
    }
    const std::span<const uint8_t> payload(message.data() + kHeaderSize, header.payload_size);
    switch (static_cast<LinkMessageType>(header.type))
    {
    case LinkMessageType::Ping:
        enqueue_reliable(encode_link_message(
            static_cast<uint8_t>(LinkMessageType::Pong), encode_pong(PongPayload{ header.t_send_ns, now_ns })));
        return;
    case LinkMessageType::Pong:
    {
        const PongPayload pong = decode_pong(payload);
        std::lock_guard<std::mutex> lock(m_clock_mutex);
        if (m_pings && m_pings->active())
        {
            m_clock.add_two_way_sample(pong.t1_ns, pong.t2_ns, header.t_send_ns, now_ns);
            m_pings->on_pong();
        }
        return;
    }
    case LinkMessageType::Heartbeat:
        return;
    default:
        break;
    }
    if (m_handler)
    {
        m_handler(ReceivedLinkMessage{ static_cast<LinkMessageType>(header.type),
                                       std::vector<uint8_t>(payload.begin(), payload.end()), header.t_send_ns, now_ns });
    }
}

void Link::close()
{
    close_impl(std::nullopt);
}

void Link::close_gracefully(std::chrono::milliseconds linger)
{
    close_impl(linger);
}

void Link::close_impl(std::optional<std::chrono::milliseconds> linger)
{
    std::lock_guard<std::mutex> close_lock(m_close_mutex);
    std::unique_lock<std::mutex> lock(m_mutex);
    if (m_closed)
    {
        return;
    }
    m_closing = true;
    m_cv.notify_all();
    if (m_started)
    {
        m_cv.wait_for(lock, m_options.close_drain, [this] { return m_sender_done || !m_alive; });
    }
    // Half-close only after the sender finished: nothing may be sent after it.
    if (linger && m_sender_done && m_ready && m_alive)
    {
        m_lingering = true;
        lock.unlock();
        m_transport->shutdown_write();
        lock.lock();
        // The receiver discards input until the peer closes (fail() then ends the wait) or the linger passes.
        m_cv.wait_for(lock, *linger, [this] { return !m_alive; });
    }
    if (m_alive)
    {
        m_alive = false;
        m_close_reason = "closed";
    }
    lock.unlock();
    m_cv.notify_all();
    m_transport->shutdown();
    if (m_sender.joinable())
    {
        m_sender.join();
    }
    if (m_receiver.joinable())
    {
        m_receiver.join();
    }
    lock.lock();
    m_closed = true;
    logger()->debug("[{}] link closed: {}", m_options.name, m_close_reason);
}

} // namespace core
