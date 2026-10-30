// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/remote_plugin_session/remote_plugin_session.hpp"

#include "inc/remote_plugin_session/clock_offset_estimator.hpp"
#include "inc/remote_plugin_session/protocol_frame.hpp"
#include "inc/remote_plugin_session/protocol_payloads.hpp"
#include "inc/remote_plugin_session/remote_address.hpp"
#include "inc/remote_plugin_session/tcp_socket.hpp"
#include "inc/remote_plugin_session/tcp_transport.hpp"

#include <log_bridge/logger.hpp>
#include <oxr_utils/os_time.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>

namespace remote_plugin_session
{

namespace protocol = remote_plugin_protocol;

namespace
{

std::shared_ptr<spdlog::logger> logger()
{
    return isaaccapture::Logger::get("isaaccapture.plugins.remote_device_bridge.RemotePluginSession");
}

// The bridge answers HELLO only after its OpenXR session is up, which waits for a headset; on CloudXR that
// routinely takes longer than a few seconds.
constexpr std::chrono::milliseconds kHandshakeTimeout{ 30000 };
constexpr std::chrono::milliseconds kHeartbeatPeriod{ 500 };
constexpr std::chrono::milliseconds kPeerTimeout{ 3000 };
constexpr std::chrono::milliseconds kReceivePoll{ 100 };
constexpr std::chrono::seconds kStatsPeriod{ 5 };
// Destruction lets the sender flush its CLOSE frames this long before the transport is cut under it.
constexpr std::chrono::milliseconds kShutdownDrain{ 500 };
// Latest-wins per channel: on a stalled link each channel keeps at most this many (or this much) of its own
// newest samples; a fast channel can never evict a slow one.
constexpr size_t kMaxQueuedSamplesPerChannel = 4;
constexpr size_t kMaxQueuedBytesPerChannel = 8u * 1024u * 1024u;

using FrameBytes = std::vector<uint8_t>; // one encoded wire message, header included

struct ChannelQueue
{
    protocol::FrameType type = protocol::FrameType::HandSample;
    std::deque<FrameBytes> frames;
    size_t bytes = 0;
    uint64_t dropped = 0;
};

} // namespace

class detail::RemotePluginSessionImpl
{
public:
    RemotePluginSessionImpl(std::unique_ptr<ITransport> transport,
                            const std::string& app_name,
                            core::PluginSessionRequirements requirements)
        : m_requirements(requirements), m_transport(std::move(transport))
    {
        if (!m_transport)
        {
            throw std::invalid_argument("RemotePluginSession: transport is null");
        }

        protocol::Hello hello;
        hello.app_name = app_name;
        hello.requirements = requirements;
        FrameBytes hello_frame = protocol::encode_frame(
            protocol::FrameType::Hello, protocol::kSessionChannelId, protocol::encode_hello(hello));
        if (!send_now(hello_frame))
        {
            throw std::runtime_error("RemotePluginSession: failed to send HELLO");
        }

        FrameBytes bytes;
        const ReceiveResult result = m_transport->receive(bytes, kHandshakeTimeout);
        if (result != ReceiveResult::Ok)
        {
            throw std::runtime_error(result == ReceiveResult::Timeout ?
                                         "RemotePluginSession: timed out waiting for HELLO_ACK" :
                                         "RemotePluginSession: connection lost while waiting for HELLO_ACK");
        }
        const protocol::Frame frame = protocol::split_frame(std::move(bytes));
        if (frame.header.type != protocol::FrameType::HelloAck)
        {
            throw std::runtime_error("RemotePluginSession: expected HELLO_ACK, received frame type " +
                                     std::to_string(static_cast<int>(frame.header.type)));
        }
        const protocol::HelloAck ack = protocol::decode_hello_ack(frame.payload);
        if (!ack.accepted)
        {
            throw std::runtime_error("RemotePluginSession: bridge rejected the session: " + ack.reason);
        }
        m_clock.add_sample(frame.header.t_send_monotonic_ns, core::os_monotonic_now_ns());

        m_connected.store(true);
        try
        {
            m_sender = std::thread(
                [this]
                {
                    sender_loop();
                    {
                        std::lock_guard<std::mutex> lock(m_queue_mutex);
                        m_sender_done = true;
                    }
                    m_queue_cv.notify_all();
                });
            m_receiver = std::thread(&RemotePluginSessionImpl::receiver_loop, this);
        }
        catch (...)
        {
            shutdown();
            throw;
        }
    }

    ~RemotePluginSessionImpl()
    {
        shutdown();
    }

    const core::PluginSessionRequirements& requirements() const
    {
        return m_requirements;
    }

    bool connected() const
    {
        return m_connected.load();
    }

    int64_t clock_offset_ns() const
    {
        std::lock_guard<std::mutex> lock(m_clock_mutex);
        return m_clock.offset_ns();
    }

    RemotePluginSession::Stats stats() const
    {
        RemotePluginSession::Stats stats;
        stats.sent_hand = m_sent_hand.load();
        stats.sent_schema = m_sent_schema.load();
        stats.wrist_rx = m_wrist_rx.load();
        stats.heartbeats = m_heartbeats.load();
        stats.tx_bytes = m_tx_bytes.load();
        std::lock_guard<std::mutex> lock(m_queue_mutex);
        stats.dropped_queue = m_dropped_samples;
        for (const auto& [channel_id, queue] : m_sample_queues)
        {
            stats.dropped_by_channel[channel_id] = queue.dropped;
        }
        stats.refused_channels = m_refused_channels;
        return stats;
    }

    void require_connected() const
    {
        if (!m_connected.load())
        {
            throw std::runtime_error("RemotePluginSession: transport is disconnected");
        }
    }

    //! Allocates a channel id and queues its OPEN_* frame; ids are never reused within a session.
    uint16_t open_channel(protocol::FrameType type, std::vector<uint8_t> payload)
    {
        require_connected();
        uint16_t channel_id = 0;
        {
            std::lock_guard<std::mutex> lock(m_state_mutex);
            if (m_next_channel_id == 0)
            {
                throw std::runtime_error("RemotePluginSession: channel ids exhausted");
            }
            channel_id = m_next_channel_id++;
            m_open_channels.insert(channel_id);
        }
        enqueue_control(type, channel_id, std::move(payload));
        return channel_id;
    }

    void send_control(protocol::FrameType type, uint16_t channel_id, std::vector<uint8_t> payload)
    {
        require_connected();
        enqueue_control(type, channel_id, std::move(payload));
    }

    void close_channel(uint16_t channel_id)
    {
        {
            std::lock_guard<std::mutex> lock(m_state_mutex);
            if (m_open_channels.erase(channel_id) == 0)
            {
                return;
            }
        }
        {
            // Pending samples must not follow CLOSE onto the wire (the bridge would see an unknown channel).
            std::lock_guard<std::mutex> lock(m_queue_mutex);
            if (const auto it = m_sample_queues.find(channel_id); it != m_sample_queues.end())
            {
                m_queued_samples -= it->second.frames.size();
                m_sample_queues.erase(it);
            }
        }
        if (m_connected.load())
        {
            enqueue_control(protocol::FrameType::Close, channel_id, {});
        }
    }

    //! Per-channel latest-wins queue; control frames are never dropped.
    void push_sample(protocol::FrameType type, uint16_t channel_id, FrameBytes frame)
    {
        if (!m_connected.load())
        {
            return;
        }
        {
            std::lock_guard<std::mutex> lock(m_queue_mutex);
            if (m_refused_channels.count(channel_id) != 0)
            {
                return;
            }
            ChannelQueue& queue = m_sample_queues[channel_id];
            queue.type = type;
            while (!queue.frames.empty() && (queue.frames.size() >= kMaxQueuedSamplesPerChannel ||
                                             queue.bytes + frame.size() > kMaxQueuedBytesPerChannel))
            {
                queue.bytes -= queue.frames.front().size();
                queue.frames.pop_front();
                ++queue.dropped;
                ++m_dropped_samples;
                --m_queued_samples;
            }
            queue.bytes += frame.size();
            queue.frames.push_back(std::move(frame));
            ++m_queued_samples;
        }
        m_queue_cv.notify_one();
    }

    core::WristTrackingSample latest_wrist(XrHandEXT hand) const
    {
        std::lock_guard<std::mutex> lock(m_wrist_mutex);
        return m_wrist[hand == XR_HAND_LEFT_EXT ? 0 : 1];
    }

    void shutdown()
    {
        if (m_shutdown.exchange(true))
        {
            return;
        }

        std::set<uint16_t> remaining;
        {
            std::lock_guard<std::mutex> lock(m_state_mutex);
            remaining.swap(m_open_channels);
        }
        if (m_connected.load())
        {
            for (auto it = remaining.rbegin(); it != remaining.rend(); ++it)
            {
                enqueue_control(protocol::FrameType::Close, *it, {});
            }
        }

        {
            std::unique_lock<std::mutex> lock(m_queue_mutex);
            m_stop = true;
            m_queue_cv.notify_all();
            if (m_sender.joinable())
            {
                // A sender stalled inside send() cannot finish on its own; the transport shutdown below cuts it off.
                m_queue_cv.wait_for(lock, kShutdownDrain, [this] { return m_sender_done; });
            }
        }
        m_connected.store(false);
        m_transport->shutdown();
        if (m_sender.joinable())
        {
            m_sender.join();
        }
        if (m_receiver.joinable())
        {
            m_receiver.join();
        }
    }

private:
    void enqueue_control(protocol::FrameType type, uint16_t channel_id, const std::vector<uint8_t>& payload)
    {
        FrameBytes frame = protocol::encode_frame(type, channel_id, payload);
        {
            std::lock_guard<std::mutex> lock(m_queue_mutex);
            m_control_queue.push_back(std::move(frame));
        }
        m_queue_cv.notify_one();
    }

    //! Stamps seq/t_send (and a PONG's pong_sent_ns) and sends one frame. Constructor thread during the handshake,
    //! sender thread after.
    bool send_now(FrameBytes& frame)
    {
        const int64_t t_send_ns = core::os_monotonic_now_ns();
        protocol::stamp_header(frame, m_next_seq++, t_send_ns);
        if (protocol::decode_header(frame).type == protocol::FrameType::Pong)
        {
            protocol::set_pong_sent_time(frame, t_send_ns);
        }
        if (!m_transport->send(frame.data(), frame.size()))
        {
            return false;
        }
        m_tx_bytes.fetch_add(frame.size());
        return true;
    }

    //! Waits until `deadline` for work. Control frames first and in order, then one sample per channel
    //! (round-robin) so a 180 Hz channel cannot starve a 60 Hz one. Empty when the deadline passed.
    std::vector<FrameBytes> next_batch(std::chrono::steady_clock::time_point deadline)
    {
        std::vector<FrameBytes> batch;
        std::unique_lock<std::mutex> lock(m_queue_mutex);
        m_queue_cv.wait_until(
            lock, deadline,
            [this] { return m_stop || !m_connected.load() || !m_control_queue.empty() || m_queued_samples > 0; });
        for (FrameBytes& frame : m_control_queue)
        {
            batch.push_back(std::move(frame));
        }
        m_control_queue.clear();
        if (m_stop)
        {
            return batch;
        }
        for (auto& [channel_id, queue] : m_sample_queues)
        {
            if (queue.frames.empty())
            {
                continue;
            }
            queue.bytes -= queue.frames.front().size();
            batch.push_back(std::move(queue.frames.front()));
            queue.frames.pop_front();
            --m_queued_samples;
            (queue.type == protocol::FrameType::HandSample ? m_sent_hand : m_sent_schema).fetch_add(1);
        }
        return batch;
    }

    void log_stats()
    {
        const RemotePluginSession::Stats now = stats();
        const RemotePluginSession::Stats& last = m_logged_stats;
        logger()->info(
            "[remote] 5s: sent hand={} schema={} wrist_rx={} hb={} | dropped queue={} | tx_bytes={} | offset_ns={} | "
            "connected={:d}",
            now.sent_hand - last.sent_hand, now.sent_schema - last.sent_schema, now.wrist_rx - last.wrist_rx,
            now.heartbeats - last.heartbeats, now.dropped_queue - last.dropped_queue, now.tx_bytes - last.tx_bytes,
            clock_offset_ns(), m_connected.load());
        m_logged_stats = now;
    }

    void on_disconnect(const char* reason)
    {
        if (!m_connected.exchange(false))
        {
            return;
        }
        if (!m_stop)
        {
            logger()->warn("[RemotePluginSession] disconnected: {}", reason);
        }
        m_transport->shutdown();
        m_queue_cv.notify_all();
    }

    //! The bridge refused one channel: its pushes become no-ops, the session and other channels continue.
    void refuse_channel(uint16_t channel_id, const std::string& reason)
    {
        logger()->warn("[RemotePluginSession] bridge refused channel {}: {}", channel_id, reason);
        std::lock_guard<std::mutex> lock(m_queue_mutex);
        m_refused_channels[channel_id] = reason;
        if (const auto it = m_sample_queues.find(channel_id); it != m_sample_queues.end())
        {
            m_queued_samples -= it->second.frames.size();
            m_sample_queues.erase(it);
        }
    }

    //! Sends are done outside the queue lock: a stalled send must not block the plugin's push().
    void sender_loop()
    {
        auto last_send = std::chrono::steady_clock::now();
        auto next_stats = last_send + kStatsPeriod;
        while (true)
        {
            std::vector<FrameBytes> batch = next_batch(std::min(last_send + kHeartbeatPeriod, next_stats));
            if (!m_connected.load())
            {
                return;
            }
            if (batch.empty() && m_stop)
            {
                return;
            }
            const auto now = std::chrono::steady_clock::now();
            if (batch.empty() && now - last_send >= kHeartbeatPeriod)
            {
                batch.push_back(protocol::encode_frame(protocol::FrameType::Heartbeat, protocol::kSessionChannelId, {}));
                m_heartbeats.fetch_add(1);
            }
            for (FrameBytes& frame : batch)
            {
                if (!send_now(frame))
                {
                    on_disconnect("send failed");
                    return;
                }
                last_send = std::chrono::steady_clock::now();
            }
            if (now >= next_stats)
            {
                log_stats();
                next_stats += kStatsPeriod;
            }
        }
    }

    void receiver_loop()
    {
        auto last_receive = std::chrono::steady_clock::now();
        while (m_connected.load() && !m_stop)
        {
            FrameBytes bytes;
            switch (m_transport->receive(bytes, kReceivePoll))
            {
            case ReceiveResult::Ok:
                last_receive = std::chrono::steady_clock::now();
                handle_frame(std::move(bytes), core::os_monotonic_now_ns());
                break;
            case ReceiveResult::Timeout:
                if (std::chrono::steady_clock::now() - last_receive > kPeerTimeout)
                {
                    on_disconnect("no frames from the bridge within the peer timeout");
                    return;
                }
                break;
            case ReceiveResult::Closed:
                on_disconnect("connection closed by the bridge");
                return;
            case ReceiveResult::Malformed:
                on_disconnect("malformed frame from the bridge");
                return;
            }
        }
    }

    void handle_frame(FrameBytes bytes, int64_t t_recv_ns)
    {
        try
        {
            const protocol::Frame frame = protocol::split_frame(std::move(bytes));
            {
                std::lock_guard<std::mutex> lock(m_clock_mutex);
                m_clock.add_sample(frame.header.t_send_monotonic_ns, t_recv_ns);
            }
            switch (frame.header.type)
            {
            case protocol::FrameType::WristSample:
            {
                // sample_time_ns is dropped: core::WristTrackingSample has no timestamp field and P0 serves the
                // latest sample.
                const protocol::WristSample sample = protocol::decode_wrist_sample(frame.payload);
                m_wrist_rx.fetch_add(1);
                std::lock_guard<std::mutex> lock(m_wrist_mutex);
                m_wrist[sample.hand == XR_HAND_LEFT_EXT ? 0 : 1] = sample.sample;
                break;
            }
            case protocol::FrameType::Ping:
            {
                // Queued as control (never dropped); send_now() sets pong_sent_ns when the sender thread sends it.
                const int64_t t1_ns = protocol::decode_ping(frame.payload);
                enqueue_control(protocol::FrameType::Pong, protocol::kSessionChannelId,
                                protocol::encode_pong(protocol::Pong{ t1_ns, t_recv_ns, 0 }));
                if (!std::exchange(m_answered_ping, true))
                {
                    logger()->info("[remote] clock: answering bridge PINGs (two-way offset)");
                }
                break;
            }
            case protocol::FrameType::Error:
            {
                const std::string message = protocol::decode_error(frame.payload);
                if (frame.header.channel_id != protocol::kSessionChannelId)
                {
                    refuse_channel(frame.header.channel_id, message);
                    break;
                }
                logger()->error("[RemotePluginSession] bridge error: {}", message);
                on_disconnect("bridge reported an error");
                break;
            }
            case protocol::FrameType::Heartbeat:
            case protocol::FrameType::HelloAck:
                break;
            default:
                logger()->warn(
                    "[RemotePluginSession] ignoring unexpected frame type {}", static_cast<int>(frame.header.type));
                break;
            }
        }
        catch (const protocol::DecodeError& e)
        {
            logger()->error("[RemotePluginSession] {}", e.what());
            on_disconnect("undecodable frame from the bridge");
        }
    }

    core::PluginSessionRequirements m_requirements;
    std::unique_ptr<ITransport> m_transport;

    std::atomic<bool> m_connected{ false };
    std::atomic<bool> m_stop{ false };
    std::atomic<bool> m_shutdown{ false };

    std::mutex m_state_mutex;
    uint16_t m_next_channel_id = 1;
    std::set<uint16_t> m_open_channels;

    mutable std::mutex m_queue_mutex;
    std::condition_variable m_queue_cv;
    std::deque<FrameBytes> m_control_queue;
    std::map<uint16_t, ChannelQueue> m_sample_queues;
    std::map<uint16_t, std::string> m_refused_channels;
    bool m_sender_done = false;
    size_t m_queued_samples = 0;
    uint64_t m_dropped_samples = 0;
    uint32_t m_next_seq = 0; // handshake on the constructor thread, then the sender thread only

    std::atomic<uint64_t> m_sent_hand{ 0 };
    std::atomic<uint64_t> m_sent_schema{ 0 };
    std::atomic<uint64_t> m_wrist_rx{ 0 };
    std::atomic<uint64_t> m_heartbeats{ 0 };
    std::atomic<uint64_t> m_tx_bytes{ 0 };
    bool m_answered_ping = false; // receiver thread only
    RemotePluginSession::Stats m_logged_stats; // sender thread only

    mutable std::mutex m_wrist_mutex;
    core::WristTrackingSample m_wrist[2];

    mutable std::mutex m_clock_mutex;
    ClockOffsetEstimator m_clock;

    std::thread m_sender;
    std::thread m_receiver;
};

namespace
{

class RemoteHandTrackingPushChannel final : public core::IHandTrackingPushChannel
{
public:
    RemoteHandTrackingPushChannel(std::shared_ptr<detail::RemotePluginSessionImpl> impl, uint16_t channel_id)
        : m_impl(std::move(impl)), m_channel_id(channel_id)
    {
    }

    ~RemoteHandTrackingPushChannel() override
    {
        m_impl->close_channel(m_channel_id);
    }

    void push(const XrHandJointLocationEXT* joint_locations, int64_t sample_time_local_common_clock_ns) override
    {
        if (joint_locations == nullptr)
        {
            throw std::invalid_argument("Hand tracking push requires joint locations");
        }
        if (!m_impl->connected())
        {
            return;
        }
        m_impl->push_sample(
            protocol::FrameType::HandSample, m_channel_id,
            protocol::encode_hand_sample_frame(m_channel_id, joint_locations, sample_time_local_common_clock_ns));
    }

private:
    std::shared_ptr<detail::RemotePluginSessionImpl> m_impl;
    uint16_t m_channel_id;
};

class RemoteSchemaPushChannel final : public core::ISchemaPushChannel
{
public:
    RemoteSchemaPushChannel(std::shared_ptr<detail::RemotePluginSessionImpl> impl, uint16_t channel_id)
        : m_impl(std::move(impl)), m_channel_id(channel_id)
    {
    }

    ~RemoteSchemaPushChannel() override
    {
        m_impl->close_channel(m_channel_id);
    }

    void push_buffer(const uint8_t* buffer,
                     size_t size,
                     int64_t sample_time_local_common_clock_ns,
                     int64_t sample_time_raw_device_clock_ns) override
    {
        if (buffer == nullptr && size != 0)
        {
            throw std::invalid_argument("push_buffer: buffer is null for a non-empty sample");
        }
        if (!m_impl->connected())
        {
            return;
        }
        m_impl->push_sample(
            protocol::FrameType::SchemaSample, m_channel_id,
            protocol::encode_schema_sample_frame(
                m_channel_id, buffer, size, sample_time_local_common_clock_ns, sample_time_raw_device_clock_ns));
    }

private:
    std::shared_ptr<detail::RemotePluginSessionImpl> m_impl;
    uint16_t m_channel_id;
};

class RemoteWristTrackingSource final : public core::IWristTrackingSource
{
public:
    explicit RemoteWristTrackingSource(std::shared_ptr<detail::RemotePluginSessionImpl> impl) : m_impl(std::move(impl))
    {
    }

    // P0 returns the latest bridge sample; the requested time is not interpolated.
    core::WristTrackingSample query(XrHandEXT hand, int64_t /*sample_time_local_common_clock_ns*/) override
    {
        return m_impl->latest_wrist(hand);
    }

private:
    std::shared_ptr<detail::RemotePluginSessionImpl> m_impl;
};

class RemotePluginPullChannel final : public core::IPluginPullChannel
{
public:
    explicit RemotePluginPullChannel(std::shared_ptr<detail::RemotePluginSessionImpl> impl)
        : m_impl(std::move(impl)), m_channel_id(m_impl->open_channel(protocol::FrameType::OpenPull, {}))
    {
        // The bridge streams WRIST_SAMPLEs on this channel when HELLO declared a wrist source.
        if (m_impl->requirements().wrist_tracking_pull)
        {
            m_wrist_source.emplace(m_impl);
        }
    }

    ~RemotePluginPullChannel() override
    {
        m_impl->close_channel(m_channel_id);
    }

    // The receiver thread keeps the wrist cache current; nothing to poll here.
    void update() override
    {
    }

    const core::ITrackerImpl& get_tracker_impl(const core::ITracker&) const override
    {
        throw std::runtime_error("RemotePluginSession pull channel exposes no trackers (P0)");
    }

    core::IWristTrackingSource& wrist_tracking_source() override
    {
        if (!m_wrist_source)
        {
            throw std::logic_error(
                "RemotePluginSession: wrist tracking pull was not declared in PluginSessionRequirements");
        }
        return *m_wrist_source;
    }

private:
    std::shared_ptr<detail::RemotePluginSessionImpl> m_impl;
    uint16_t m_channel_id;
    std::optional<RemoteWristTrackingSource> m_wrist_source;
};

} // namespace

RemotePluginSession::RemotePluginSession(std::string host,
                                         uint16_t port,
                                         std::string app_name,
                                         core::PluginSessionRequirements requirements)
    : RemotePluginSession(
          std::make_unique<TcpTransport>(TcpSocket::connect(Endpoint{ std::move(host), port }, kConnectTimeout)),
          std::move(app_name),
          requirements)
{
}

RemotePluginSession::RemotePluginSession(std::unique_ptr<ITransport> transport,
                                         std::string app_name,
                                         core::PluginSessionRequirements requirements)
    : m_impl(std::make_shared<detail::RemotePluginSessionImpl>(std::move(transport), app_name, requirements)),
      m_pull_channel(std::make_unique<RemotePluginPullChannel>(m_impl))
{
}

RemotePluginSession::~RemotePluginSession()
{
    m_impl->shutdown();
}

core::IPluginPullChannel& RemotePluginSession::pull_channel()
{
    return *m_pull_channel;
}

std::unique_ptr<core::ISchemaPushChannel> RemotePluginSession::create_schema_push_channel(
    const core::SchemaPusherConfig& config)
{
    if (!m_impl->requirements().schema_push)
    {
        throw std::logic_error("RemotePluginSession: schema push was not declared in PluginSessionRequirements");
    }
    const uint16_t channel_id =
        m_impl->open_channel(protocol::FrameType::OpenSchema, protocol::encode_open_schema(config));
    return std::make_unique<RemoteSchemaPushChannel>(m_impl, channel_id);
}

std::unique_ptr<core::IHandTrackingPushChannel> RemotePluginSession::create_hand_tracking_push_channel(XrHandEXT hand)
{
    if (!m_impl->requirements().hand_tracking_push)
    {
        throw std::logic_error("RemotePluginSession: hand-tracking push was not declared in PluginSessionRequirements");
    }
    const uint16_t channel_id = m_impl->open_channel(protocol::FrameType::OpenHand, protocol::encode_open_hand(hand));
    return std::make_unique<RemoteHandTrackingPushChannel>(m_impl, channel_id);
}

bool RemotePluginSession::connected() const
{
    return m_impl->connected();
}

int64_t RemotePluginSession::clock_offset_ns() const
{
    return m_impl->clock_offset_ns();
}

RemotePluginSession::Stats RemotePluginSession::stats() const
{
    return m_impl->stats();
}

core::PluginSessionFactory make_remote_plugin_session_factory(std::string app_name, RemoteAddress remote)
{
    return [app_name = std::move(app_name), remote = std::move(remote)](
               const core::PluginSessionRequirements& requirements,
               std::vector<std::shared_ptr<core::ITracker>> trackers) -> std::unique_ptr<core::IPluginSession>
    {
        if (!trackers.empty())
        {
            throw std::logic_error("RemotePluginSession exposes no trackers; " + app_name + " passed some");
        }
        return std::make_unique<RemotePluginSession>(connect_transport(remote), app_name, requirements);
    };
}

} // namespace remote_plugin_session
