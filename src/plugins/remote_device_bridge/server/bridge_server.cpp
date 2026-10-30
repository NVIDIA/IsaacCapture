// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/remote_device_bridge/bridge_server.hpp"

#include "inc/remote_device_bridge/latency_stats.hpp"

#include <log_bridge/logger.hpp>
#include <oxr_utils/os_time.hpp>
#include <remote_plugin_session/bridge_limits.hpp>
#include <remote_plugin_session/clock_offset_estimator.hpp>
#include <remote_plugin_session/protocol_frame.hpp>
#include <remote_plugin_session/protocol_payloads.hpp>
#include <remote_plugin_session/tcp_socket.hpp>
#include <remote_plugin_session/tcp_transport.hpp>
#include <remote_plugin_session/websocket_transport.hpp>
#include <sys/mman.h>
#include <sys/wait.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <poll.h>
#include <set>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace remote_device_bridge
{

namespace
{

using remote_plugin_session::ClockOffsetEstimator;
using remote_plugin_session::ITransport;
using remote_plugin_session::PingScheduler;
using remote_plugin_session::ReceiveResult;
namespace protocol = remote_plugin_protocol;

std::shared_ptr<spdlog::logger> logger()
{
    return isaaccapture::Logger::get("isaaccapture.plugins.remote_device_bridge.BridgeServer");
}

constexpr std::chrono::milliseconds kAcceptPoll{ 200 };
constexpr std::chrono::milliseconds kHandshakeTimeout{ 3000 };
constexpr std::chrono::milliseconds kReceivePoll{ 100 };
constexpr std::chrono::nanoseconds kHeartbeatPeriod{ 500'000'000 };
constexpr std::chrono::nanoseconds kPeerTimeout{ 3'000'000'000 };
// A refused client gets "bridge full" as soon as its transport is up (ws://: after the upgrade, waited for this
// long), then the connection lingers this long so the ACK is delivered and a late HELLO is read, not reset.
constexpr std::chrono::milliseconds kLingerWait{ 1000 };
// Refusal children alive at once; past this a refused client gets a best-effort ACK and an immediate close.
constexpr size_t kMaxRefusing = 16;
constexpr std::chrono::milliseconds kReapPoll{ 100 };
constexpr std::chrono::seconds kStopGrace{ 3 };

bool is_sample(protocol::FrameType type)
{
    return type == protocol::FrameType::HandSample || type == protocol::FrameType::SchemaSample;
}

struct Received
{
    protocol::Frame frame;
    int64_t t_recv_ns;
};

//! HandOwners::owner index of a hand.
size_t hand_index(XrHandEXT hand)
{
    return hand == XR_HAND_LEFT_EXT ? 0 : 1;
}

const char* hand_side(size_t hand_index)
{
    return hand_index == 0 ? "left" : "right";
}

} // namespace

//! Bridge client id owning each hand's push device (0 = free). Lives in a MAP_SHARED page created before any
//! fork, so every child sees the others' claims; the parent frees a crashed child's claims when it reaps it.
struct detail::HandOwners
{
    std::array<std::atomic<uint32_t>, 2> owner{};

    //! Empty when `client_id` now owns the hand, else the current owner.
    std::optional<uint32_t> claim(size_t hand, uint32_t client_id)
    {
        uint32_t expected = 0;
        if (owner[hand].compare_exchange_strong(expected, client_id))
        {
            return std::nullopt;
        }
        return expected;
    }

    void release(size_t hand, uint32_t client_id)
    {
        uint32_t expected = client_id;
        owner[hand].compare_exchange_strong(expected, 0);
    }
};
static_assert(std::atomic<uint32_t>::is_always_lock_free, "hand owners are shared across processes");

void detail::HandOwnersUnmap::operator()(HandOwners* owners) const noexcept
{
    munmap(owners, sizeof(HandOwners));
}

namespace
{

//! Refuses one OPEN_*: answered with ERROR on that channel id, the session and its other channels continue.
class ChannelRefused : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

//! The handshake did not produce a session; the reason is already logged.
class HandshakeRefused : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

//! An owner outside this bridge (a local plugin) makes xrCreatePushDeviceNV fail with XR_ERROR_RUNTIME_FAILURE (-2,
//! seen 2026-09-29; XR_NVX1_device_interface has no dedicated code). plugin_utils throws it as a std::system_error
//! in the "XrResult" category. Any other result stays a session error.
bool push_device_owned_elsewhere(const std::exception& e)
{
    const auto* error = dynamic_cast<const std::system_error*>(&e);
    return error != nullptr && std::strcmp(error->code().category().name(), "XrResult") == 0 &&
           error->code().value() == XR_ERROR_RUNTIME_FAILURE &&
           std::string(error->what()).rfind("xrCreatePushDeviceNV(", 0) == 0;
}

//! One accepted connection: owns the mirrored plugin session and its channels.
class ClientSession
{
public:
    //! Reads HELLO, creates the session and answers HELLO_ACK; throws HandshakeRefused when the client is refused.
    ClientSession(std::unique_ptr<ITransport> transport,
                  uint32_t client_id,
                  const BridgeOptions& options,
                  const SessionFactory& session_factory,
                  detail::HandOwners& hand_owners,
                  const SampleCsv* csv)
        : m_transport(std::move(transport)),
          m_client_id(client_id),
          m_tag("[bridge c" + std::to_string(client_id) + "]"),
          m_options(options),
          m_hand_owners(hand_owners),
          m_csv(csv)
    {
        // The reader starts last, so a throw here leaves only members whose destructors close the transport.
        handshake(session_factory);
    }

    ~ClientSession()
    {
        m_done = true;
        m_transport->shutdown();
        m_reader.join();

        // Push channels close before the session, which owns the pull channel and wrist source (m_session is
        // declared before the channel maps, so it is destroyed last).
        while (!m_hand_channels.empty())
        {
            close_channel(m_hand_channels.begin()->first);
        }
        while (!m_schema_channels.empty())
        {
            close_channel(m_schema_channels.begin()->first);
        }
    }

    void run(const std::atomic<bool>& stop)
    {
        const auto period = std::chrono::nanoseconds(1'000'000'000 / m_options.wrist_rate_hz);
        auto next_tick = std::chrono::steady_clock::now() + period;
        auto next_stats = std::chrono::steady_clock::now() + m_options.stats_period;

        while (!stop.load() && !m_done.load())
        {
            {
                std::unique_lock<std::mutex> lock(m_queue_mutex);
                m_queue_cv.wait_until(lock, next_tick, [this] { return !m_queue.empty() || m_reader_closed.load(); });
            }
            if (!apply_queued())
            {
                return;
            }

            if (m_reader_closed.load())
            {
                // The reader never pushes after raising the flag, so one more pass picks up the client's last
                // frames (typically CLOSE for the pull channel) that landed between the swap above and the check.
                if (!apply_queued())
                {
                    return;
                }
                if (m_reader_error.empty())
                {
                    logger()->info("{} client disconnected", m_tag);
                }
                else
                {
                    fail(m_reader_error);
                }
                return;
            }

            const int64_t now_ns = core::os_monotonic_now_ns();
            if (now_ns - m_last_recv_ns > kPeerTimeout.count())
            {
                logger()->warn("{} no frames from the client within the peer timeout; closing", m_tag);
                return;
            }

            const auto now = std::chrono::steady_clock::now();
            if (now >= next_tick)
            {
                tick(now_ns);
                next_tick += period;
                if (next_tick < now)
                {
                    next_tick = now + period;
                }
            }
            poll_pings(now_ns);
            if (!m_done.load() && now_ns - m_last_send_ns > kHeartbeatPeriod.count())
            {
                send(protocol::FrameType::Heartbeat, protocol::kSessionChannelId, {});
            }
            if (now >= next_stats)
            {
                log_stats();
                next_stats += m_options.stats_period;
            }
        }
    }

private:
    struct HandChannel
    {
        std::unique_ptr<core::IHandTrackingPushChannel> channel;
        size_t hand = 0; //!< HandOwners index
        uint64_t pushes = 0;
    };

    struct SchemaChannel
    {
        std::unique_ptr<core::ISchemaPushChannel> channel;
        std::string collection_id;
        uint64_t pushes = 0;
    };

    struct Counters
    {
        uint64_t rx_hand = 0;
        uint64_t rx_schema = 0;
        uint64_t applied = 0;
        uint64_t coalesced = 0;
        uint64_t stale = 0;
        uint64_t tx_wrist = 0;
    };

    [[noreturn]] void refuse_handshake(const std::string& reason)
    {
        logger()->warn("{} {}; dropping connection", m_tag, reason);
        throw HandshakeRefused(reason);
    }

    void handshake(const SessionFactory& session_factory)
    {
        std::vector<uint8_t> bytes;
        if (m_transport->receive(bytes, kHandshakeTimeout) != ReceiveResult::Ok)
        {
            refuse_handshake("bad HELLO (no frame)");
        }
        protocol::Frame frame;
        protocol::Hello hello;
        try
        {
            frame = protocol::split_frame(std::move(bytes));
            if (frame.header.type != protocol::FrameType::Hello)
            {
                refuse_handshake("bad HELLO (expected HELLO, got frame type " +
                                 std::to_string(static_cast<int>(frame.header.type)) + ")");
            }
            hello = protocol::decode_hello(frame.payload);
        }
        catch (const protocol::DecodeError& e)
        {
            refuse_handshake(std::string("bad HELLO (") + e.what() + ")");
        }
        note_received(frame.header, core::os_monotonic_now_ns());

        const std::string app_name =
            m_options.app_name.value_or(!hello.app_name.empty() ? hello.app_name : std::string("remote_device_bridge"));
        logger()->info(
            "{} HELLO from '{}' (schema_push={:d} hand_tracking_push={:d} wrist_tracking_pull={:d}); app name '{}'",
            m_tag, hello.app_name, hello.requirements.schema_push, hello.requirements.hand_tracking_push,
            hello.requirements.wrist_tracking_pull.has_value(), app_name);

        logger()->info("{} creating plugin session", m_tag);
        try
        {
            m_session = session_factory(app_name)(hello.requirements, {});
        }
        catch (const std::exception& e)
        {
            logger()->error("{} plugin session creation failed: {}", m_tag, e.what());
            send(protocol::FrameType::HelloAck, protocol::kSessionChannelId,
                 protocol::encode_hello_ack(
                     protocol::HelloAck{ false, std::string("plugin session creation failed: ") + e.what() }));
            throw HandshakeRefused(e.what());
        }
        logger()->info("{} plugin session ready", m_tag);
        m_wrist_declared = hello.requirements.wrist_tracking_pull.has_value();

        send(protocol::FrameType::HelloAck, protocol::kSessionChannelId,
             protocol::encode_hello_ack(protocol::HelloAck{ true, "" }));
        if (m_done.load())
        {
            refuse_handshake("HELLO_ACK not delivered");
        }
        if (m_options.clock_source == ClockSource::TwoWay)
        {
            m_pings.emplace(core::os_monotonic_now_ns());
        }
        // Last: nothing may throw once the reader runs.
        m_reader = std::thread(&ClientSession::reader_loop, this);
    }

    void reader_loop()
    {
        while (!m_done.load())
        {
            std::vector<uint8_t> bytes;
            const ReceiveResult result = m_transport->receive(bytes, kReceivePoll);
            if (result == ReceiveResult::Timeout)
            {
                continue;
            }
            if (result == ReceiveResult::Malformed)
            {
                m_reader_error = "malformed message from the transport";
                break;
            }
            if (result != ReceiveResult::Ok)
            {
                break;
            }
            Received received{ {}, core::os_monotonic_now_ns() };
            try
            {
                received.frame = protocol::split_frame(std::move(bytes));
            }
            catch (const protocol::DecodeError& e)
            {
                m_reader_error = std::string("bad frame: ") + e.what();
                break;
            }
            {
                std::lock_guard<std::mutex> lock(m_queue_mutex);
                m_queue.push_back(std::move(received));
            }
            m_queue_cv.notify_one();
        }
        m_reader_closed.store(true);
        m_queue_cv.notify_all();
    }

    //! Swaps out the queue and applies it: control frames in order, samples coalesced to the newest per channel
    //! (a slow OpenXR push must not turn into a growing backlog). False once the session has failed.
    bool apply_queued()
    {
        std::deque<Received> batch;
        {
            std::lock_guard<std::mutex> lock(m_queue_mutex);
            batch.swap(m_queue);
        }
        std::map<uint16_t, size_t> newest;
        for (size_t i = 0; i < batch.size(); ++i)
        {
            if (is_sample(batch[i].frame.header.type))
            {
                newest[batch[i].frame.header.channel_id] = i;
            }
        }
        for (size_t i = 0; i < batch.size(); ++i)
        {
            const protocol::FrameHeader& header = batch[i].frame.header;
            if (is_sample(header.type))
            {
                ++(header.type == protocol::FrameType::HandSample ? m_counts.rx_hand : m_counts.rx_schema);
                if (newest[header.channel_id] != i)
                {
                    ++m_counts.coalesced;
                    note_received(header, batch[i].t_recv_ns);
                    continue;
                }
            }
            apply(batch[i]);
            if (m_done.load())
            {
                return false;
            }
        }
        return true;
    }

    void note_received(const protocol::FrameHeader& header, int64_t t_recv_ns)
    {
        m_clock.add_sample(header.t_send_monotonic_ns, t_recv_ns);
        m_last_recv_ns = t_recv_ns;
    }

    //! Set while OPEN_PULL is open and HELLO declared a wrist source; owned by m_session.
    core::IWristTrackingSource* wrist_source() const
    {
        return m_pull && m_wrist_declared ? &m_pull->channel->wrist_tracking_source() : nullptr;
    }

    void apply(const Received& received)
    {
        const protocol::FrameHeader& header = received.frame.header;
        const std::vector<uint8_t>& payload = received.frame.payload;
        note_received(header, received.t_recv_ns);

        try
        {
            switch (header.type)
            {
            case protocol::FrameType::OpenHand:
            {
                const XrHandEXT open_hand = protocol::decode_open_hand(payload);
                require_free_channel(header.channel_id);
                const size_t hand = hand_index(open_hand);
                const char* const side = hand_side(hand);
                // The runtime allows one push device per hand; refuse a second owner before it gets there.
                if (const std::optional<uint32_t> owner = m_hand_owners.claim(hand, m_client_id))
                {
                    throw ChannelRefused(std::string("hand ") + side + " is already owned by bridge client c" +
                                         std::to_string(*owner) + "; channel refused");
                }
                try
                {
                    auto channel = m_session->create_hand_tracking_push_channel(open_hand);
                    m_hand_channels.emplace(header.channel_id, HandChannel{ std::move(channel), hand });
                }
                catch (const std::exception& e)
                {
                    m_hand_owners.release(hand, m_client_id);
                    if (!push_device_owned_elsewhere(e))
                    {
                        throw;
                    }
                    logger()->warn("{} {}", m_tag, e.what());
                    throw ChannelRefused(std::string("hand ") + side +
                                         " is already owned by another OpenXR session (runtime refused the push "
                                         "device); channel refused");
                }
                logger()->info("{} channel {}: hand {} opened", m_tag, header.channel_id, side);
                break;
            }
            case protocol::FrameType::OpenSchema:
            {
                const core::SchemaPusherConfig config = protocol::decode_open_schema(payload);
                require_free_channel(header.channel_id);
                // The runtime tensor is max_flatbuffer_size bytes whatever the sample holds.
                if (const std::optional<std::string> error = remote_plugin_session::schema_cap_error(
                        config.collection_id, config.max_flatbuffer_size, m_options.max_schema_bytes))
                {
                    throw ChannelRefused(*error);
                }
                auto channel = m_session->create_schema_push_channel(config);
                m_schema_channels.emplace(header.channel_id, SchemaChannel{ std::move(channel), config.collection_id });
                logger()->info("{} channel {}: schema '{}' opened", m_tag, header.channel_id, config.collection_id);
                break;
            }
            case protocol::FrameType::OpenPull:
            {
                if (m_pull)
                {
                    throw std::runtime_error("only one pull channel per session");
                }
                require_free_channel(header.channel_id);
                // The session built the pull channel, and the wrist source HELLO declared, when it was created.
                m_pull = PullChannel{ &m_session->pull_channel(), header.channel_id };
                logger()->info("{} channel {}: pull channel opened{}", m_tag, header.channel_id,
                               wrist_source() != nullptr ? " with wrist source" : "");
                break;
            }
            case protocol::FrameType::Close:
                close_channel(header.channel_id);
                break;
            case protocol::FrameType::HandSample:
            {
                const auto it = m_hand_channels.find(header.channel_id);
                if (it == m_hand_channels.end())
                {
                    if (m_refused_channels.count(header.channel_id) != 0)
                    {
                        break; // sent before the client saw the refusal
                    }
                    throw std::runtime_error("HAND_SAMPLE on unknown channel " + std::to_string(header.channel_id));
                }
                const protocol::HandSample sample = protocol::decode_hand_sample(payload);
                const int64_t t_ns = m_clock.remote_to_local(sample.sample_time_ns);
                const int64_t now_ns = core::os_monotonic_now_ns();
                if (stale(t_ns, now_ns))
                {
                    break;
                }
                it->second.channel->push(sample.joints.data(), t_ns);
                ++it->second.pushes;
                ++m_counts.applied;
                if (m_csv &&
                    !m_csv->write({ core::os_monotonic_now_ns(), received.t_recv_ns, sample.sample_time_ns, t_ns,
                                    now_ns - t_ns, hand_side(it->second.hand), header.seq,
                                    index_curl_m(sample.joints.data()) }) &&
                    !m_csv_failed)
                {
                    m_csv_failed = true;
                    logger()->warn(
                        "{} stats CSV row not written ({}); later rows may be missing", m_tag, std::strerror(errno));
                }
                break;
            }
            case protocol::FrameType::SchemaSample:
            {
                const auto it = m_schema_channels.find(header.channel_id);
                if (it == m_schema_channels.end())
                {
                    if (m_refused_channels.count(header.channel_id) != 0)
                    {
                        break;
                    }
                    throw std::runtime_error("SCHEMA_SAMPLE on unknown channel " + std::to_string(header.channel_id));
                }
                const protocol::SchemaSample sample = protocol::decode_schema_sample(payload);
                const int64_t t_ns = m_clock.remote_to_local(sample.sample_time_local_common_clock_ns);
                if (stale(t_ns, core::os_monotonic_now_ns()))
                {
                    break;
                }
                it->second.channel->push_buffer(
                    sample.bytes.data(), sample.bytes.size(), t_ns, sample.sample_time_raw_device_clock_ns);
                ++it->second.pushes;
                ++m_counts.applied;
                break;
            }
            case protocol::FrameType::Pong:
            {
                const protocol::Pong pong = protocol::decode_pong(payload);
                if (!m_pings || !m_pings->active())
                {
                    break;
                }
                m_pings->on_pong();
                m_clock.add_two_way_sample(
                    pong.ping_sent_ns, pong.ping_received_ns, pong.pong_sent_ns, received.t_recv_ns);
                if (!std::exchange(m_logged_pong, true))
                {
                    logger()->info(
                        "{} clock: first PONG, rtt_ms={:g}; using two-way offset", m_tag, m_clock.rtt_last_ns() / 1e6);
                }
                break;
            }
            case protocol::FrameType::Heartbeat:
                break;
            case protocol::FrameType::Hello:
                throw std::runtime_error("HELLO after the handshake");
            default:
                throw std::runtime_error("unexpected frame type " + std::to_string(static_cast<int>(header.type)));
            }
        }
        catch (const ChannelRefused& e)
        {
            refuse_channel(header.channel_id, e.what());
        }
        catch (const std::exception& e)
        {
            fail(e.what());
        }
    }

    //! True (and counted) when the mapped sample time is older than max_sample_age. Records the age either way.
    bool stale(int64_t mapped_t_ns, int64_t now_ns)
    {
        m_ages_ns.push_back(now_ns - mapped_t_ns);
        if (!m_options.max_sample_age ||
            !remote_plugin_session::should_drop_stale(
                mapped_t_ns, now_ns, std::chrono::nanoseconds(*m_options.max_sample_age).count()))
        {
            return false;
        }
        ++m_counts.stale;
        return true;
    }

    void require_free_channel(uint16_t channel_id) const
    {
        if (channel_id == protocol::kSessionChannelId || m_hand_channels.count(channel_id) != 0 ||
            m_schema_channels.count(channel_id) != 0 || m_refused_channels.count(channel_id) != 0 ||
            (m_pull && channel_id == m_pull->id))
        {
            throw std::runtime_error("channel id " + std::to_string(channel_id) + " is already in use");
        }
    }

    //! Logs each push channel's push count as it closes.
    void close_channel(uint16_t channel_id)
    {
        if (const auto hand = m_hand_channels.find(channel_id); hand != m_hand_channels.end())
        {
            const uint64_t pushes = hand->second.pushes;
            const size_t hand_index = hand->second.hand;
            m_hand_channels.erase(hand);
            m_hand_owners.release(hand_index, m_client_id);
            logger()->info("{} hand channel {} closed after {} pushes", m_tag, channel_id, pushes);
            return;
        }
        if (const auto schema = m_schema_channels.find(channel_id); schema != m_schema_channels.end())
        {
            const uint64_t pushes = schema->second.pushes;
            const std::string collection_id = schema->second.collection_id;
            m_schema_channels.erase(schema);
            logger()->info("{} schema channel {} ({}) closed after {} pushes", m_tag, channel_id, collection_id, pushes);
            return;
        }
        if (m_pull && channel_id == m_pull->id)
        {
            m_pull.reset();
            logger()->info("{} channel {} (pull) closed", m_tag, channel_id);
            return;
        }
        if (m_refused_channels.erase(channel_id) != 0)
        {
            return;
        }
        logger()->warn("{} CLOSE for unknown channel {}", m_tag, channel_id);
    }

    void poll_pings(int64_t now_ns)
    {
        if (!m_pings)
        {
            return;
        }
        switch (m_pings->poll(now_ns))
        {
        case PingScheduler::Action::Ping:
            send(protocol::FrameType::Ping, protocol::kSessionChannelId,
                 protocol::encode_ping(core::os_monotonic_now_ns()));
            break;
        case PingScheduler::Action::GiveUp:
            logger()->warn("{} clock: no PONG, falling back to one-way", m_tag);
            break;
        case PingScheduler::Action::None:
            break;
        }
    }

    void tick(int64_t now_ns)
    {
        if (!m_pull)
        {
            return;
        }
        try
        {
            m_pull->channel->update();
            if (core::IWristTrackingSource* const wrist = wrist_source())
            {
                for (const XrHandEXT hand : { XR_HAND_LEFT_EXT, XR_HAND_RIGHT_EXT })
                {
                    protocol::WristSample sample;
                    sample.sample_time_ns = now_ns;
                    sample.hand = hand;
                    sample.sample = wrist->query(hand, now_ns);
                    send(protocol::FrameType::WristSample, m_pull->id, protocol::encode_wrist_sample(sample));
                    if (m_done.load())
                    {
                        return;
                    }
                    ++m_counts.tx_wrist;
                }
            }
        }
        catch (const std::exception& e)
        {
            fail(e.what());
        }
    }

    //! A failed send ends the session (sets m_done).
    void send(protocol::FrameType type, uint16_t channel_id, const std::vector<uint8_t>& payload)
    {
        std::vector<uint8_t> frame = protocol::encode_frame(type, channel_id, payload);
        m_last_send_ns = core::os_monotonic_now_ns();
        protocol::stamp_header(frame, m_next_seq++, m_last_send_ns);
        if (!m_transport->send(frame.data(), frame.size()))
        {
            m_done.store(true);
        }
    }

    void refuse_channel(uint16_t channel_id, const std::string& message)
    {
        logger()->warn("{} channel {} refused: {}", m_tag, channel_id, message);
        m_refused_channels.insert(channel_id);
        send(protocol::FrameType::Error, channel_id, protocol::encode_error(message));
    }

    void fail(const std::string& message)
    {
        logger()->error("{} error: {}; closing connection", m_tag, message);
        send(protocol::FrameType::Error, protocol::kSessionChannelId, protocol::encode_error(message));
        m_done.store(true);
    }

    //! Deltas since the previous line; all counters live on the run thread.
    void log_stats()
    {
        logger()->info("{} 5s: rx hand={} schema={} | applied={} coalesced={} stale={} | tx wrist={} | offset_ns={}{}",
                       m_tag, m_counts.rx_hand - m_logged.rx_hand, m_counts.rx_schema - m_logged.rx_schema,
                       m_counts.applied - m_logged.applied, m_counts.coalesced - m_logged.coalesced,
                       m_counts.stale - m_logged.stale, m_counts.tx_wrist - m_logged.tx_wrist, m_clock.offset_ns(),
                       format_clock_stats(std::move(m_ages_ns), m_clock));
        m_logged = m_counts;
        m_ages_ns.clear();
    }

    std::unique_ptr<ITransport> m_transport;
    const uint32_t m_client_id;
    const std::string m_tag;
    const BridgeOptions& m_options;
    detail::HandOwners& m_hand_owners;

    std::unique_ptr<core::IPluginSession> m_session;
    std::map<uint16_t, HandChannel> m_hand_channels;
    std::map<uint16_t, SchemaChannel> m_schema_channels;
    struct PullChannel
    {
        core::IPluginPullChannel* channel; //!< owned by m_session
        uint16_t id;
    };
    std::optional<PullChannel> m_pull; //!< set while OPEN_PULL is open
    bool m_wrist_declared = false;
    std::set<uint16_t> m_refused_channels; //!< until the client's CLOSE

    ClockOffsetEstimator m_clock;
    std::optional<PingScheduler> m_pings; //!< set for --clock-source=two-way
    bool m_logged_pong = false;
    std::vector<int64_t> m_ages_ns; //!< since the last status line
    bool m_csv_failed = false; //!< warned once
    const SampleCsv* m_csv = nullptr; //!< the parent's --stats-csv file, inherited across fork(); null without it
    uint32_t m_next_seq = 0;
    int64_t m_last_send_ns = 0;
    int64_t m_last_recv_ns = 0;
    Counters m_counts;
    Counters m_logged; //!< m_counts at the previous status line

    std::thread m_reader;
    std::string m_reader_error; //!< why the reader stopped, if not a clean close; written before m_reader_closed
    std::mutex m_queue_mutex;
    std::condition_variable m_queue_cv;
    std::deque<Received> m_queue;
    std::atomic<bool> m_reader_closed{ false };
    std::atomic<bool> m_done{ false };
};

//! Runs one client to completion in the forked child; false after a fatal error.
bool serve_client(std::unique_ptr<ITransport> transport,
                  uint32_t client_id,
                  const BridgeOptions& options,
                  const SessionFactory& session_factory,
                  detail::HandOwners& hand_owners,
                  const SampleCsv* csv,
                  const std::atomic<bool>& stop)
{
    bool ok = true;
    try
    {
        ClientSession session(std::move(transport), client_id, options, session_factory, hand_owners, csv);
        session.run(stop);
    }
    catch (const HandshakeRefused&)
    {
    }
    catch (const std::exception& e)
    {
        logger()->error("[bridge c{}] fatal: {}", client_id, e.what());
        ok = false;
    }
    logger()->info("[bridge c{}] session closed", client_id);
    return ok;
}

std::vector<uint8_t> bridge_full_ack()
{
    return protocol::encode_frame(protocol::FrameType::HelloAck, protocol::kSessionChannelId,
                                  protocol::encode_hello_ack(protocol::HelloAck{ false, "bridge full" }));
}

//! Runs in a refusal child: HELLO_ACK first, then read until the client hangs up (it does once it reads the
//! ACK) or kLingerWait passes, so the close is a FIN even when the HELLO arrives late.
void send_bridge_full(ITransport& client, std::vector<uint8_t> ack)
{
    if (client.wait_ready(kLingerWait) != ReceiveResult::Ok)
    {
        return;
    }
    protocol::stamp_header(ack, 0, core::os_monotonic_now_ns());
    if (!client.send(ack.data(), ack.size()))
    {
        return;
    }
    const auto deadline = std::chrono::steady_clock::now() + kLingerWait;
    for (std::vector<uint8_t> bytes;;)
    {
        const auto left =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
        if (left <= std::chrono::milliseconds{ 0 })
        {
            return;
        }
        const ReceiveResult result = client.receive(bytes, left);
        if (result == ReceiveResult::Closed || result == ReceiveResult::Malformed)
        {
            return;
        }
    }
}

double uptime_s(std::chrono::steady_clock::time_point started)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
}

} // namespace

struct BridgeServer::ClientRecord
{
    uint32_t id = 0;
    std::string peer;
    std::chrono::steady_clock::time_point started;
    pid_t pid = 0;
    bool refusal = false; //!< only sends "bridge full"; not counted against --max-clients
};

BridgeServer::BridgeServer(BridgeOptions options, SessionFactory session_factory)
    : m_options(std::move(options)), m_session_factory(std::move(session_factory))
{
    if (!m_session_factory)
    {
        throw std::invalid_argument("BridgeServer requires a session factory");
    }
    if (m_options.wrist_rate_hz <= 0 || m_options.max_clients <= 0 ||
        (m_options.max_sample_age && m_options.max_sample_age->count() <= 0))
    {
        throw std::invalid_argument("BridgeOptions: wrist_rate_hz, max_clients and max_sample_age must be positive");
    }
    if (m_options.max_schema_bytes == 0)
    {
        throw std::invalid_argument("BridgeOptions: max_schema_bytes must be positive");
    }
    m_listener = remote_plugin_session::TcpSocket::listen(m_options.listen, m_options.max_clients);
    if (m_options.ws_listen)
    {
        m_ws_listener = remote_plugin_session::TcpSocket::listen(*m_options.ws_listen, m_options.max_clients);
        logger()->info(
            "remote_device_bridge listening for ws:// on {}:{}", m_options.ws_listen->host, m_ws_listener.local_port());
    }
    if (m_options.stats_csv)
    {
        m_csv = std::make_unique<SampleCsv>(*m_options.stats_csv);
    }
    logger()->info(
        "remote_device_bridge listening on {}:{} (wrist {} Hz, process per client, max {}, max sample age {}, max "
        "schema {} bytes)",
        m_options.listen.host, m_listener.local_port(), m_options.wrist_rate_hz, m_options.max_clients,
        m_options.max_sample_age ? std::to_string(m_options.max_sample_age->count()) + " ms" : std::string("off"),
        m_options.max_schema_bytes);
    logger()->info("remote_device_bridge clock={}{}",
                   m_options.clock_source == ClockSource::TwoWay ? "two-way" : "one-way",
                   m_options.stats_csv ? " stats-csv=" + *m_options.stats_csv : std::string());
    void* const shared =
        mmap(nullptr, sizeof(detail::HandOwners), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (shared == MAP_FAILED)
    {
        throw std::runtime_error(std::string("mmap(hand owners): ") + std::strerror(errno));
    }
    m_hand_owners.reset(new (shared) detail::HandOwners());
}

BridgeServer::~BridgeServer() = default;

uint16_t BridgeServer::port() const
{
    return m_listener.local_port();
}

std::optional<uint16_t> BridgeServer::ws_port() const
{
    if (!m_ws_listener.valid())
    {
        return std::nullopt;
    }
    return m_ws_listener.local_port();
}

void BridgeServer::release_hands(const ClientRecord& client)
{
    for (size_t hand = 0; hand < m_hand_owners->owner.size(); ++hand)
    {
        m_hand_owners->release(hand, client.id);
    }
}

void BridgeServer::on_reaped(const ClientRecord& client, int status)
{
    if (client.refusal)
    {
        return;
    }
    release_hands(client);
    logger()->info("[bridge] client pid {} exited ({}) after {:g} s", client.pid,
                   WIFEXITED(status) ? "status " + std::to_string(WEXITSTATUS(status)) :
                                       "signal " + std::to_string(WTERMSIG(status)),
                   uptime_s(client.started));
}

void BridgeServer::reap_finished()
{
    for (auto it = m_clients.begin(); it != m_clients.end();)
    {
        int status = 0;
        const pid_t reaped = waitpid(it->pid, &status, WNOHANG);
        if (reaped == 0)
        {
            ++it;
            continue;
        }
        if (reaped < 0)
        {
            // The child is gone without a status (ECHILD); drop it so it stops counting against max_clients.
            logger()->warn("[bridge] waitpid(client pid {}) failed: {}", it->pid, std::strerror(errno));
            release_hands(*it);
        }
        else
        {
            on_reaped(*it, status);
        }
        it = m_clients.erase(it);
    }
}

void BridgeServer::stop_and_reap_all()
{
    for (const ClientRecord& client : m_clients)
    {
        kill(client.pid, SIGTERM);
    }
    // A child still inside the session factory (the OpenXR one waits for a headset) never sees the stop flag, so
    // the grace period ends in SIGKILL.
    const auto grace_end = std::chrono::steady_clock::now() + kStopGrace;
    for (const ClientRecord& client : m_clients)
    {
        int status = 0;
        pid_t reaped = waitpid(client.pid, &status, WNOHANG);
        while (reaped == 0)
        {
            if (std::chrono::steady_clock::now() >= grace_end)
            {
                logger()->warn("[bridge] client pid {} killed after {} s grace", client.pid, kStopGrace.count());
                kill(client.pid, SIGKILL);
                reaped = waitpid(client.pid, &status, 0);
                break;
            }
            std::this_thread::sleep_for(kReapPoll);
            reaped = waitpid(client.pid, &status, WNOHANG);
        }
        if (reaped == client.pid)
        {
            on_reaped(client, status);
        }
    }
    m_clients.clear();
}

void BridgeServer::admit_client(std::unique_ptr<ITransport> client, const std::string& peer, const std::atomic<bool>& stop)
{
    const uint32_t id = ++m_accepted;
    const auto refusing = static_cast<size_t>(
        std::count_if(m_clients.begin(), m_clients.end(), [](const ClientRecord& c) { return c.refusal; }));
    const size_t active = m_clients.size() - refusing;
    if (active >= static_cast<size_t>(m_options.max_clients))
    {
        logger()->warn("[bridge] client {} refused: bridge full ({})", peer, active);
        if (refusing >= kMaxRefusing)
        {
            std::vector<uint8_t> ack = bridge_full_ack();
            protocol::stamp_header(ack, 0, core::os_monotonic_now_ns());
            client->send(ack.data(), ack.size());
            return;
        }
        // A child answers so the accept loop never waits on a slow client (the ws:// upgrade, the linger).
        const pid_t pid = fork();
        if (pid == 0)
        {
            m_listener.close();
            m_ws_listener.close();
            send_bridge_full(*client, bridge_full_ack());
            _exit(0);
        }
        if (pid < 0)
        {
            logger()->error("[bridge] fork failed ({}); closing refused client {}", std::strerror(errno), peer);
            return;
        }
        m_clients.push_back(ClientRecord{ id, peer, std::chrono::steady_clock::now(), pid, true });
        return;
    }

    const pid_t pid = fork();
    if (pid < 0)
    {
        logger()->error("[bridge] fork failed ({}); dropping client {}", std::strerror(errno), peer);
        return;
    }
    if (pid == 0)
    {
        m_listener.close();
        m_ws_listener.close();
        const bool ok =
            serve_client(std::move(client), id, m_options, m_session_factory, *m_hand_owners, m_csv.get(), stop);
        // _exit skips stdio teardown: flush the logger's file sink and stdout first.
        logger()->flush();
        std::fflush(nullptr);
        _exit(ok ? 0 : 1);
    }
    logger()->info("[bridge] client {} started (pid {}, c{})", peer, pid, id);
    m_clients.push_back(ClientRecord{ id, peer, std::chrono::steady_clock::now(), pid });
}

void BridgeServer::run(const std::atomic<bool>& stop)
{
    while (!stop.load())
    {
        reap_finished();
        std::array<pollfd, 2> listeners{ pollfd{ m_listener.fd(), POLLIN, 0 }, pollfd{ m_ws_listener.fd(), POLLIN, 0 } };
        const nfds_t count = m_ws_listener.valid() ? 2 : 1;
        const int ready = ::poll(listeners.data(), count, static_cast<int>(kAcceptPoll.count()));
        if (ready < 0 && errno != EINTR)
        {
            throw std::runtime_error(std::string("poll(accept): ") + std::strerror(errno));
        }
        if (ready <= 0)
        {
            continue;
        }
        for (nfds_t i = 0; i < count; ++i)
        {
            if ((listeners[i].revents & POLLIN) == 0)
            {
                continue;
            }
            std::string peer;
            std::unique_ptr<ITransport> client =
                i == 0 ? remote_plugin_session::TcpTransport::accept(m_listener, std::chrono::milliseconds{ 0 }, &peer) :
                         remote_plugin_session::WebSocketTransport::accept(
                             m_ws_listener, std::chrono::milliseconds{ 0 }, &peer);
            if (client)
            {
                admit_client(std::move(client), i == 0 ? peer : "ws://" + peer, stop);
            }
        }
    }
    stop_and_reap_all();
}

} // namespace remote_device_bridge
