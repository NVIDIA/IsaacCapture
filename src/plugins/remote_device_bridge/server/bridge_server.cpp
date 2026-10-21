// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/remote_device_bridge/bridge_server.hpp"

#include <log_bridge/logger.hpp>
#include <oxr_utils/os_time.hpp>
#include <remote_plugin_session/bridge_limits.hpp>
#include <remote_plugin_session/clock_offset_estimator.hpp>
#include <remote_plugin_session/latency_stats.hpp>
#include <remote_plugin_session/tcp_socket.hpp>
#include <remote_plugin_session/websocket_transport.hpp>
#include <remote_plugin_session/wire_codec.hpp>
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
namespace wire = remote_plugin_session::wire;

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

bool is_sample(wire::FrameType type)
{
    return type == wire::FrameType::HandSample || type == wire::FrameType::SchemaSample;
}

struct Received
{
    wire::Frame frame;
    int64_t t_recv_ns;
};

} // namespace

//! Bridge client id owning each hand's push device (0 = free). Lives in a MAP_SHARED page created before any
//! fork, so every child sees the others' claims; the parent frees a crashed child's claims when it reaps it.
struct BridgeServer::HandOwners
{
    std::array<std::atomic<uint32_t>, 2> owner{};

    //! 0 when `client_id` now owns the hand, else the current owner.
    uint32_t claim(size_t hand, uint32_t client_id)
    {
        uint32_t expected = 0;
        owner[hand].compare_exchange_strong(expected, client_id);
        return expected;
    }

    void release(size_t hand, uint32_t client_id)
    {
        uint32_t expected = client_id;
        owner[hand].compare_exchange_strong(expected, 0);
    }
};
static_assert(std::atomic<uint32_t>::is_always_lock_free, "hand owners are shared across processes");

namespace
{

//! Refuses one OPEN_*: answered with ERROR on that channel id, the session and its other channels continue.
class ChannelRefused : public std::runtime_error
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

//! One accepted connection: owns the mirrored OpenXR session and its channels.
class ClientSession
{
public:
    ClientSession(std::unique_ptr<ITransport> transport,
                  uint32_t client_id,
                  const BridgeOptions& options,
                  const SessionFactory& session_factory,
                  BridgeServer::HandOwners& hand_owners)
        : m_transport(std::move(transport)),
          m_client_id(client_id),
          m_tag("[bridge c" + std::to_string(client_id) + "]"),
          m_options(options),
          m_session_factory(session_factory),
          m_hand_owners(hand_owners)
    {
        if (!m_options.stats_csv.empty())
        {
            m_csv = std::make_unique<remote_plugin_session::SampleCsv>(m_options.stats_csv);
        }
    }

    ~ClientSession()
    {
        m_done = true;
        m_transport->shutdown();
        if (m_reader.joinable())
        {
            m_reader.join();
        }

        // Teardown order mirrors OpenXRPluginSession's contract: channels before the session.
        while (!m_hand_channels.empty())
        {
            close_channel(m_hand_channels.begin()->first);
        }
        while (!m_schema_channels.empty())
        {
            close_channel(m_schema_channels.begin()->first);
        }
        m_wrist_source.reset();
        m_pull_channel.reset();
        m_session.reset();
    }

    ClientSession(const ClientSession&) = delete;
    ClientSession& operator=(const ClientSession&) = delete;

    //! Reads HELLO, creates the OpenXR session and answers HELLO_ACK. False means the client was refused.
    bool handshake()
    {
        std::vector<uint8_t> bytes;
        wire::Frame frame;
        wire::Hello hello;
        try
        {
            if (m_transport->receive(bytes, kHandshakeTimeout) != ReceiveResult::Ok)
            {
                throw wire::DecodeError("no frame");
            }
            frame = wire::split_frame(std::move(bytes));
            if (frame.header.type != wire::FrameType::Hello)
            {
                throw wire::DecodeError(std::string("expected HELLO, got ") + wire::to_string(frame.header.type));
            }
            hello = wire::decode_hello(frame.payload);
        }
        catch (const wire::DecodeError& e)
        {
            logger()->warn("{} bad HELLO ({}); dropping connection", m_tag, e.what());
            return false;
        }
        note_received(frame.header, core::os_monotonic_now_ns());

        if (hello.proto_version > wire::kHelloVersion)
        {
            send(wire::FrameType::HelloAck, wire::kSessionChannelId,
                 wire::encode(
                     wire::HelloAck{ false, "unsupported protocol version " + std::to_string(hello.proto_version) }));
            return false;
        }

        std::string app_name = !m_options.app_name.empty() ? m_options.app_name :
                               !hello.app_name.empty()     ? hello.app_name :
                                                             std::string("remote_device_bridge");
        logger()->info(
            "{} HELLO from '{}' (schema_push={:d} hand_tracking_push={:d} wrist_tracking_pull={:d}); OpenXR app "
            "name '{}'",
            m_tag, hello.app_name, hello.requirements.schema_push, hello.requirements.hand_tracking_push,
            hello.requirements.wrist_tracking_pull, app_name);

        logger()->info("{} creating OpenXR plugin session (waits for a headset)", m_tag);
        try
        {
            m_session = m_session_factory(app_name, hello.requirements);
        }
        catch (const std::exception& e)
        {
            logger()->error("{} OpenXR session creation failed: {}", m_tag, e.what());
            send(wire::FrameType::HelloAck, wire::kSessionChannelId,
                 wire::encode(wire::HelloAck{ false, std::string("OpenXR session creation failed: ") + e.what() }));
            return false;
        }
        logger()->info("{} OpenXR plugin session ready", m_tag);

        if (!send(wire::FrameType::HelloAck, wire::kSessionChannelId, wire::encode(wire::HelloAck{ true, "" })))
        {
            return false;
        }
        m_reader = std::thread(&ClientSession::reader_loop, this);
        if (m_options.clock_source == ClockSource::TwoWay)
        {
            if (hello.proto_version >= 1)
            {
                m_pings.emplace(core::os_monotonic_now_ns());
            }
            else
            {
                logger()->warn("{} WARN clock: HELLO v0 client cannot answer PING, using one-way", m_tag);
            }
        }
        return true;
    }

    void run(const std::atomic<bool>& stop)
    {
        const auto period = std::chrono::nanoseconds(1'000'000'000 / std::max(1, m_options.wrist_rate_hz));
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
                apply_queued();
                logger()->info("{} client disconnected", m_tag);
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
                send(wire::FrameType::Heartbeat, wire::kSessionChannelId, {});
            }
            if (now >= next_stats)
            {
                log_stats();
                next_stats += m_options.stats_period;
            }
        }
    }

private:
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
            if (result != ReceiveResult::Ok)
            {
                break;
            }
            Received received{ {}, core::os_monotonic_now_ns() };
            try
            {
                received.frame = wire::split_frame(std::move(bytes));
            }
            catch (const wire::DecodeError&)
            {
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
            const wire::FrameHeader& header = batch[i].frame.header;
            if (is_sample(header.type))
            {
                ++(header.type == wire::FrameType::HandSample ? m_rx_hand : m_rx_schema);
                if (newest[header.channel_id] != i)
                {
                    ++m_coalesced;
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

    void note_received(const wire::FrameHeader& header, int64_t t_recv_ns)
    {
        m_clock.add_sample(header.t_send_monotonic_ns, t_recv_ns);
        m_last_recv_ns = t_recv_ns;
    }

    void apply(const Received& received)
    {
        const wire::FrameHeader& header = received.frame.header;
        const std::vector<uint8_t>& payload = received.frame.payload;
        note_received(header, received.t_recv_ns);

        try
        {
            switch (header.type)
            {
            case wire::FrameType::OpenHand:
            {
                const wire::OpenHand open = wire::decode_open_hand(payload);
                require_free_channel(header.channel_id);
                const char* const side = open.hand == XR_HAND_LEFT_EXT ? "left" : "right";
                const size_t hand = open.hand == XR_HAND_LEFT_EXT ? 0 : 1;
                // The runtime allows one push device per hand; refuse a second owner before it gets there.
                if (const uint32_t owner = m_hand_owners.claim(hand, m_client_id); owner != 0)
                {
                    throw ChannelRefused(std::string("hand ") + side + " is already owned by bridge client c" +
                                         std::to_string(owner) + "; channel refused");
                }
                try
                {
                    auto channel = m_session->create_hand_tracking_push_channel(open.hand);
                    m_hand_channels[header.channel_id] = { std::move(channel), 0, side, hand };
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
            case wire::FrameType::OpenSchema:
            {
                const core::SchemaPusherConfig config = wire::decode_open_schema(payload);
                require_free_channel(header.channel_id);
                // The runtime tensor is max_flatbuffer_size bytes whatever the sample holds.
                if (const std::string error = remote_plugin_session::schema_cap_error(
                        config.collection_id, config.max_flatbuffer_size, m_options.max_schema_bytes);
                    !error.empty())
                {
                    throw ChannelRefused(error);
                }
                m_schema_channels[header.channel_id].channel = m_session->create_schema_push_channel(config);
                logger()->info("{} channel {}: schema '{}' opened", m_tag, header.channel_id, config.collection_id);
                break;
            }
            case wire::FrameType::OpenPull:
            {
                if (m_pull_channel)
                {
                    throw std::runtime_error("only one pull channel per session");
                }
                require_free_channel(header.channel_id);
                m_pull_channel = m_session->create_pull_channel();
                m_pull_channel_id = header.channel_id;
                logger()->info("{} channel {}: pull channel opened", m_tag, header.channel_id);
                break;
            }
            case wire::FrameType::OpenWrist:
            {
                const core::WristTrackingSourceConfig config = wire::decode_open_wrist(payload);
                if (!m_pull_channel || header.channel_id != m_pull_channel_id)
                {
                    throw std::runtime_error("OPEN_WRIST requires the open pull channel id");
                }
                m_wrist_source = m_pull_channel->create_wrist_tracking_source(config);
                if (!m_wrist_source)
                {
                    throw std::runtime_error("wrist tracking source unavailable for the requested mode");
                }
                logger()->info("{} channel {}: wrist source opened", m_tag, header.channel_id);
                break;
            }
            case wire::FrameType::Close:
                close_channel(header.channel_id);
                break;
            case wire::FrameType::HandSample:
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
                const wire::HandSample sample = wire::decode_hand_sample(payload);
                const int64_t t_ns = m_clock.remote_to_local(sample.sample_time_ns);
                const int64_t now_ns = core::os_monotonic_now_ns();
                if (stale(t_ns, now_ns))
                {
                    break;
                }
                it->second.channel->push(sample.joints.data(), t_ns);
                ++it->second.pushes;
                ++m_applied;
                if (m_csv)
                {
                    m_csv->write({ core::os_monotonic_now_ns(), received.t_recv_ns, sample.sample_time_ns, t_ns,
                                   now_ns - t_ns, it->second.side, header.seq,
                                   remote_plugin_session::index_curl_m(sample.joints.data()) });
                }
                break;
            }
            case wire::FrameType::SchemaSample:
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
                const wire::SchemaSample sample = wire::decode_schema_sample(payload);
                // OpenXRSchemaPushChannel copies into a max_flatbuffer_size buffer without checking.
                if (sample.bytes.size() > it->second.channel->config().max_flatbuffer_size)
                {
                    throw std::runtime_error("SCHEMA_SAMPLE of " + std::to_string(sample.bytes.size()) +
                                             " bytes exceeds the channel's max_flatbuffer_size");
                }
                const int64_t t_ns = m_clock.remote_to_local(sample.sample_time_local_common_clock_ns);
                if (stale(t_ns, core::os_monotonic_now_ns()))
                {
                    break;
                }
                it->second.channel->push_buffer(
                    sample.bytes.data(), sample.bytes.size(), t_ns, sample.sample_time_raw_device_clock_ns);
                ++it->second.pushes;
                ++m_applied;
                break;
            }
            case wire::FrameType::Pong:
            {
                const wire::Pong pong = wire::decode_pong(payload);
                if (!m_pings || !m_pings->active())
                {
                    break;
                }
                m_pings->on_pong();
                m_clock.add_two_way_sample(pong.t1_ns, pong.t2_ns, pong.t3_ns, received.t_recv_ns);
                if (!std::exchange(m_logged_pong, true))
                {
                    logger()->info(
                        "{} clock: first PONG, rtt_ms={:g}; using two-way offset", m_tag, m_clock.rtt_last_ns() / 1e6);
                }
                break;
            }
            case wire::FrameType::Heartbeat:
            case wire::FrameType::Hello:
                break;
            default:
                throw std::runtime_error(std::string("unexpected frame ") + wire::to_string(header.type));
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

    //! True (and counted) when the mapped sample time is older than --max-sample-age-ms. Records the age either way.
    bool stale(int64_t mapped_t_ns, int64_t now_ns)
    {
        m_ages_ns.push_back(now_ns - mapped_t_ns);
        const int64_t max_age_ns = static_cast<int64_t>(m_options.max_sample_age_ms) * 1'000'000;
        if (!remote_plugin_session::should_drop_stale(mapped_t_ns, now_ns, max_age_ns))
        {
            return false;
        }
        ++m_stale;
        return true;
    }

    void require_free_channel(uint16_t channel_id) const
    {
        if (channel_id == wire::kSessionChannelId || m_hand_channels.count(channel_id) != 0 ||
            m_schema_channels.count(channel_id) != 0 || m_refused_channels.count(channel_id) != 0 ||
            (m_pull_channel && channel_id == m_pull_channel_id))
        {
            throw std::runtime_error("channel id " + std::to_string(channel_id) + " is already in use");
        }
    }

    //! Logs the per-channel push count: it is the evidence the loopback runbook reads.
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
            const std::string collection_id = schema->second.channel->config().collection_id;
            m_schema_channels.erase(schema);
            logger()->info("{} schema channel {} ({}) closed after {} pushes", m_tag, channel_id, collection_id, pushes);
            return;
        }
        if (m_pull_channel && channel_id == m_pull_channel_id)
        {
            m_wrist_source.reset();
            m_pull_channel.reset();
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
            send(wire::FrameType::Ping, wire::kSessionChannelId, wire::encode(wire::Ping{ core::os_monotonic_now_ns() }));
            break;
        case PingScheduler::Action::GiveUp:
            logger()->warn("{} WARN clock: no PONG, falling back to one-way", m_tag);
            break;
        case PingScheduler::Action::None:
            break;
        }
    }

    void tick(int64_t now_ns)
    {
        if (!m_pull_channel)
        {
            return;
        }
        try
        {
            m_pull_channel->update();
            if (m_wrist_source)
            {
                for (const bool is_left : { true, false })
                {
                    wire::WristSample sample;
                    sample.sample_time_ns = now_ns;
                    sample.is_left = is_left;
                    sample.sample = m_wrist_source->query(is_left, now_ns);
                    if (!send(wire::FrameType::WristSample, m_pull_channel_id, wire::encode(sample)))
                    {
                        return;
                    }
                    ++m_tx_wrist;
                }
            }
        }
        catch (const std::exception& e)
        {
            fail(e.what());
        }
    }

    bool send(wire::FrameType type, uint16_t channel_id, const std::vector<uint8_t>& payload)
    {
        std::vector<uint8_t> frame = wire::encode_frame(type, channel_id, payload);
        m_last_send_ns = core::os_monotonic_now_ns();
        wire::stamp_header(frame.data(), m_next_seq++, m_last_send_ns);
        if (!m_transport->send(frame.data(), frame.size()))
        {
            m_done.store(true);
            return false;
        }
        return true;
    }

    void refuse_channel(uint16_t channel_id, const std::string& message)
    {
        logger()->warn("{} channel {} refused: {}", m_tag, channel_id, message);
        m_refused_channels.insert(channel_id);
        send(wire::FrameType::Error, channel_id, wire::encode(wire::Error{ message }));
    }

    void fail(const std::string& message)
    {
        logger()->error("{} error: {}; closing connection", m_tag, message);
        send(wire::FrameType::Error, wire::kSessionChannelId, wire::encode(wire::Error{ message }));
        m_done.store(true);
    }

    //! Deltas since the previous line; all counters live on the run thread.
    void log_stats()
    {
        logger()->info("{} 5s: rx hand={} schema={} | applied={} coalesced={} stale={} | tx wrist={} | offset_ns={}{}",
                       m_tag, m_rx_hand - m_logged[0], m_rx_schema - m_logged[1], m_applied - m_logged[2],
                       m_coalesced - m_logged[3], m_stale - m_logged[4], m_tx_wrist - m_logged[5], m_clock.offset_ns(),
                       remote_plugin_session::format_clock_stats(m_ages_ns, m_clock));
        m_logged = { m_rx_hand, m_rx_schema, m_applied, m_coalesced, m_stale, m_tx_wrist };
        m_ages_ns.clear();
    }

    template <typename Channel>
    struct CountedChannel
    {
        std::unique_ptr<Channel> channel;
        uint64_t pushes = 0;
        const char* side = ""; //!< hand channels: "left" / "right"
        size_t hand = 0; //!< hand channels: HandOwners index
    };

    std::unique_ptr<ITransport> m_transport;
    const uint32_t m_client_id;
    const std::string m_tag;
    const BridgeOptions& m_options;
    const SessionFactory& m_session_factory;
    BridgeServer::HandOwners& m_hand_owners;

    std::unique_ptr<core::IPluginSession> m_session;
    std::map<uint16_t, CountedChannel<core::IHandTrackingPushChannel>> m_hand_channels;
    std::map<uint16_t, CountedChannel<core::ISchemaPushChannel>> m_schema_channels;
    std::unique_ptr<core::IPluginPullChannel> m_pull_channel;
    std::unique_ptr<core::IWristTrackingSource> m_wrist_source;
    uint16_t m_pull_channel_id = 0;
    std::set<uint16_t> m_refused_channels; //!< until the client's CLOSE

    ClockOffsetEstimator m_clock;
    std::optional<PingScheduler> m_pings; //!< set for --clock-source=two-way with a v1 client
    bool m_logged_pong = false;
    std::vector<int64_t> m_ages_ns; //!< since the last 5 s line
    std::unique_ptr<remote_plugin_session::SampleCsv> m_csv;
    uint32_t m_next_seq = 0;
    int64_t m_last_send_ns = 0;
    int64_t m_last_recv_ns = 0;
    uint64_t m_rx_hand = 0;
    uint64_t m_rx_schema = 0;
    uint64_t m_applied = 0;
    uint64_t m_coalesced = 0;
    uint64_t m_stale = 0;
    uint64_t m_tx_wrist = 0;
    std::array<uint64_t, 6> m_logged{};

    std::thread m_reader;
    std::mutex m_queue_mutex;
    std::condition_variable m_queue_cv;
    std::deque<Received> m_queue;
    std::atomic<bool> m_reader_closed{ false };
    std::atomic<bool> m_done{ false };
};

//! Runs one client to completion in the forked child.
void serve_client(std::unique_ptr<ITransport> transport,
                  uint32_t client_id,
                  const BridgeOptions& options,
                  const SessionFactory& session_factory,
                  BridgeServer::HandOwners& hand_owners,
                  const std::atomic<bool>& stop)
{
    try
    {
        ClientSession session(std::move(transport), client_id, options, session_factory, hand_owners);
        if (session.handshake())
        {
            session.run(stop);
        }
    }
    catch (const std::exception& e)
    {
        logger()->error("[bridge c{}] fatal: {}", client_id, e.what());
    }
    logger()->info("[bridge c{}] session closed", client_id);
}

//! Runs in a refusal child: HELLO_ACK first, then read until the client hangs up (it does once it reads the
//! ACK) or kLingerWait passes, so the close is a FIN even when the HELLO arrives late.
void send_bridge_full(ITransport& client, std::vector<uint8_t> ack)
{
    if (client.wait_ready(kLingerWait) != ReceiveResult::Ok)
    {
        return;
    }
    wire::stamp_header(ack.data(), 0, core::os_monotonic_now_ns());
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
    if (m_options.wrist_rate_hz <= 0 || m_options.max_clients <= 0 || m_options.max_sample_age_ms < 0)
    {
        throw std::invalid_argument("--wrist-rate-hz and --max-clients must be positive, --max-sample-age-ms >= 0");
    }
    if (m_options.max_schema_bytes == 0)
    {
        throw std::invalid_argument("--max-schema-bytes must be a positive integer (bytes); default 32768");
    }
    m_listener = remote_plugin_session::TcpSocket::listen(m_options.listen, m_options.max_clients);
    if (m_options.ws_listen)
    {
        m_ws_listener = remote_plugin_session::TcpSocket::listen(*m_options.ws_listen, m_options.max_clients);
        logger()->info(
            "remote_device_bridge listening for ws:// on {}:{}", m_options.ws_listen->host, m_ws_listener.local_port());
    }
    if (!m_options.stats_csv.empty())
    {
        remote_plugin_session::SampleCsv::create(m_options.stats_csv);
    }
    m_bridge_full_ack = wire::encode_frame(
        wire::FrameType::HelloAck, wire::kSessionChannelId, wire::encode(wire::HelloAck{ false, "bridge full" }));
    logger()->info(
        "remote_device_bridge listening on {}:{} (wrist {} Hz, process per client, max {}, max sample age {} ms, max "
        "schema {} bytes)",
        m_options.listen.host, m_listener.local_port(), m_options.wrist_rate_hz, m_options.max_clients,
        m_options.max_sample_age_ms, m_options.max_schema_bytes);
    logger()->info("remote_device_bridge clock={}{}",
                   m_options.clock_source == ClockSource::TwoWay ? "two-way" : "one-way",
                   m_options.stats_csv.empty() ? "" : " stats-csv=" + m_options.stats_csv);
    // Mapped last so a throw above cannot leak it.
    void* const shared = mmap(nullptr, sizeof(HandOwners), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (shared == MAP_FAILED)
    {
        throw std::runtime_error(std::string("mmap(hand owners): ") + std::strerror(errno));
    }
    m_hand_owners = new (shared) HandOwners();
}

BridgeServer::~BridgeServer()
{
    if (m_hand_owners != nullptr)
    {
        m_hand_owners->~HandOwners();
        munmap(m_hand_owners, sizeof(HandOwners));
    }
}

uint16_t BridgeServer::port() const
{
    return m_listener.local_port();
}

uint16_t BridgeServer::ws_port() const
{
    return m_ws_listener.valid() ? m_ws_listener.local_port() : 0;
}

//! Logs and forgets finished clients; with `wait` it first asks children to stop and blocks for all of them.
void BridgeServer::reap_clients(bool wait)
{
    if (wait)
    {
        for (const ClientRecord& client : m_clients)
        {
            kill(client.pid, SIGTERM);
        }
    }
    // A child still inside the OpenXRPluginSession constructor (waiting for a headset) never sees the stop flag,
    // so the grace period ends in SIGKILL.
    const auto grace_end = std::chrono::steady_clock::now() + kStopGrace;
    for (auto it = m_clients.begin(); it != m_clients.end();)
    {
        int status = 0;
        pid_t reaped = waitpid(it->pid, &status, WNOHANG);
        while (wait && reaped == 0)
        {
            if (std::chrono::steady_clock::now() >= grace_end)
            {
                logger()->warn("[bridge] client pid {} killed after {} s grace", it->pid, kStopGrace.count());
                kill(it->pid, SIGKILL);
                reaped = waitpid(it->pid, &status, 0);
                break;
            }
            std::this_thread::sleep_for(kReapPoll);
            reaped = waitpid(it->pid, &status, WNOHANG);
        }
        const bool gone = reaped == it->pid;
        if (gone && !it->refusal)
        {
            for (size_t hand = 0; hand < m_hand_owners->owner.size(); ++hand)
            {
                m_hand_owners->release(hand, it->id);
            }
            logger()->info("[bridge] client pid {} exited ({}) after {:g} s", it->pid,
                           WIFEXITED(status) ? "status " + std::to_string(WEXITSTATUS(status)) :
                                               "signal " + std::to_string(WTERMSIG(status)),
                           uptime_s(it->started));
        }
        it = gone ? m_clients.erase(it) : std::next(it);
    }
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
            std::vector<uint8_t> ack = m_bridge_full_ack;
            wire::stamp_header(ack.data(), 0, core::os_monotonic_now_ns());
            client->send(ack.data(), ack.size());
            return;
        }
        // A child answers so the accept loop never waits on a slow client (the ws:// upgrade, the linger).
        const pid_t pid = fork();
        if (pid == 0)
        {
            m_listener.close();
            m_ws_listener.close();
            send_bridge_full(*client, m_bridge_full_ack);
            _exit(0);
        }
        if (pid < 0)
        {
            logger()->error("[bridge] fork failed; closing refused client {}", peer);
            return;
        }
        m_clients.push_back(ClientRecord{ id, peer, std::chrono::steady_clock::now(), pid, true });
        return;
    }

    // fork() is safe here because this process never creates threads or OpenXR objects: the child is a
    // clean process that owns exactly one OpenXRPluginSession, like a local plugin process would. Threads
    // are not an option: the OpenXR loader allows one XrInstance per process.
    const pid_t pid = fork();
    if (pid < 0)
    {
        logger()->error("[bridge] fork failed; dropping client {}", peer);
        return;
    }
    if (pid == 0)
    {
        m_listener.close();
        m_ws_listener.close();
        serve_client(std::move(client), id, m_options, m_session_factory, *m_hand_owners, stop);
        // _exit skips stdio teardown: flush the logger's file sink and stdout first.
        logger()->flush();
        std::fflush(nullptr);
        _exit(0);
    }
    logger()->info("[bridge] client {} started (pid {}, c{})", peer, pid, id);
    m_clients.push_back(ClientRecord{ id, peer, std::chrono::steady_clock::now(), pid });
}

void BridgeServer::run(const std::atomic<bool>& stop)
{
    while (!stop.load())
    {
        reap_clients(false);
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
    reap_clients(true);
}

} // namespace remote_device_bridge
