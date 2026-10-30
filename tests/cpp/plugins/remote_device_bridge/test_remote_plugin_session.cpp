// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <oxr_utils/os_time.hpp>
#include <remote_plugin_session/clock_offset_estimator.hpp>
#include <remote_plugin_session/protocol_frame.hpp>
#include <remote_plugin_session/protocol_payloads.hpp>
#include <remote_plugin_session/remote_plugin_session.hpp>
#include <remote_plugin_session/tcp_socket.hpp>
#include <remote_plugin_session/tcp_transport.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using namespace remote_plugin_session;
namespace protocol = remote_plugin_protocol;

namespace
{

constexpr std::chrono::milliseconds kWait{ 2000 };
const core::WristTrackingSourceConfig kWristConfig{
    .mode = core::WristTrackingSourceMode::Auto,
    .left_aim_to_wrist = { { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } },
    .right_aim_to_wrist = { { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } }
};

bool send_frame(ITransport& transport,
                protocol::FrameType type,
                uint16_t channel_id,
                uint32_t seq,
                const std::vector<uint8_t>& payload)
{
    std::vector<uint8_t> frame = protocol::encode_frame(type, channel_id, payload);
    protocol::stamp_header(frame, seq, core::os_monotonic_now_ns());
    return transport.send(frame.data(), frame.size());
}

//! Removes and returns the oldest frame of `type`.
std::optional<protocol::Frame> take_frame(std::deque<protocol::Frame>& frames, protocol::FrameType type)
{
    const auto it = std::find_if(
        frames.begin(), frames.end(), [type](const protocol::Frame& frame) { return frame.header.type == type; });
    if (it == frames.end())
    {
        return std::nullopt;
    }
    protocol::Frame frame = std::move(*it);
    frames.erase(it);
    return frame;
}

//! Minimal in-process bridge: answers HELLO, records every frame, can inject frames and drop the socket.
class FakeBridge
{
public:
    explicit FakeBridge(bool accept = true, std::string reason = "")
        : m_listener(TcpSocket::listen(Endpoint{ "127.0.0.1", 0 }, 1)), m_accept(accept), m_reason(std::move(reason))
    {
        m_thread = std::thread(&FakeBridge::serve, this);
    }

    ~FakeBridge()
    {
        m_stop = true;
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_client)
            {
                m_client->shutdown();
            }
        }
        m_thread.join();
    }

    uint16_t port() const
    {
        return m_listener.local_port();
    }

    std::optional<protocol::Frame> wait_for(protocol::FrameType type, std::chrono::milliseconds timeout = kWait)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (true)
        {
            if (auto frame = take_frame(m_frames, type))
            {
                return frame;
            }
            if (m_cv.wait_until(lock, deadline) == std::cv_status::timeout)
            {
                return std::nullopt;
            }
        }
    }

    bool send(protocol::FrameType type, uint16_t channel_id, const std::vector<uint8_t>& payload)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_client && send_frame(*m_client, type, channel_id, m_seq++, payload);
    }

    void drop_connection()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_client)
        {
            m_client->shutdown();
        }
    }

    bool client_closed(std::chrono::milliseconds timeout = kWait)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_cv.wait_for(lock, timeout, [this] { return m_closed; });
    }

private:
    void serve()
    {
        std::unique_ptr<ITransport> client;
        while (!m_stop && !client)
        {
            client = TcpTransport::accept(m_listener, std::chrono::milliseconds(50));
        }
        if (!client)
        {
            return;
        }

        std::vector<uint8_t> bytes;
        if (client->receive(bytes, kWait) != ReceiveResult::Ok)
        {
            return;
        }
        protocol::Frame hello = protocol::split_frame(std::move(bytes));
        if (hello.header.type != protocol::FrameType::Hello)
        {
            return;
        }
        ITransport* transport = client.get();
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_client = std::move(client);
            m_frames.push_back(std::move(hello));
            send_frame(*m_client, protocol::FrameType::HelloAck, protocol::kSessionChannelId, m_seq++,
                       protocol::encode_hello_ack(protocol::HelloAck{ m_accept, m_reason }));
        }
        m_cv.notify_all();

        while (!m_stop)
        {
            const ReceiveResult result = transport->receive(bytes, std::chrono::milliseconds(50));
            if (result == ReceiveResult::Timeout)
            {
                continue;
            }
            std::lock_guard<std::mutex> lock(m_mutex);
            if (result == ReceiveResult::Ok)
            {
                m_frames.push_back(protocol::split_frame(std::move(bytes)));
            }
            else
            {
                m_closed = true;
                m_cv.notify_all();
                return;
            }
            m_cv.notify_all();
        }
    }

    TcpSocket m_listener;
    bool m_accept;
    std::string m_reason;
    std::atomic<bool> m_stop{ false };
    std::thread m_thread;

    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::unique_ptr<ITransport> m_client;
    std::deque<protocol::Frame> m_frames;
    uint32_t m_seq = 0;
    bool m_closed = false;
};

//! In-memory ITransport standing in for a stalled TCP link: while held, send() blocks like a write into a full
//! socket; released frames are recorded. receive() serves the accepting HELLO_ACK and times out afterwards.
class BlockingTransport final : public ITransport
{
public:
    BlockingTransport()
    {
        m_inbox.push_back(protocol::encode_frame(protocol::FrameType::HelloAck, protocol::kSessionChannelId,
                                                 protocol::encode_hello_ack(protocol::HelloAck{ true, "" })));
        protocol::stamp_header(m_inbox.back(), 0, core::os_monotonic_now_ns());
    }

    bool send(const uint8_t* frame, size_t size) override
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_blocked = m_held;
        m_cv.wait(lock, [this] { return !m_held || m_closed; });
        m_blocked = false;
        if (m_closed)
        {
            return false;
        }
        m_sent.push_back(protocol::split_frame(std::vector<uint8_t>(frame, frame + size)));
        m_cv.notify_all();
        return true;
    }

    ReceiveResult receive(std::vector<uint8_t>& frame, std::chrono::milliseconds timeout) override
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (!m_cv.wait_for(lock, timeout, [this] { return !m_inbox.empty() || m_closed; }))
        {
            return ReceiveResult::Timeout;
        }
        if (m_closed)
        {
            return ReceiveResult::Closed;
        }
        frame = std::move(m_inbox.front());
        m_inbox.pop_front();
        return ReceiveResult::Ok;
    }

    void shutdown() override
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_closed = true;
        m_cv.notify_all();
    }

    void hold(bool held)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_held = held;
        m_cv.notify_all();
    }

    //! True while a send() is waiting for hold(false).
    bool blocked() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_blocked;
    }

    std::optional<protocol::Frame> wait_for(protocol::FrameType type, std::chrono::milliseconds timeout = kWait)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (true)
        {
            if (auto frame = take_frame(m_sent, type))
            {
                return frame;
            }
            if (m_cv.wait_until(lock, deadline) == std::cv_status::timeout)
            {
                return std::nullopt;
            }
        }
    }

private:
    mutable std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_held = false;
    bool m_blocked = false;
    bool m_closed = false;
    std::deque<protocol::Frame> m_sent;
    std::deque<std::vector<uint8_t>> m_inbox;
};

std::array<XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT> make_joints(float seed)
{
    std::array<XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT> joints{};
    for (size_t i = 0; i < joints.size(); ++i)
    {
        const float f = seed + 0.001f * static_cast<float>(i);
        joints[i].locationFlags = XR_SPACE_LOCATION_POSITION_VALID_BIT | static_cast<uint64_t>(i) << 32;
        joints[i].pose = { { f, -f, 2.0f * f, 1.0f - f }, { 3.0f * f, 4.0f * f, -5.0f * f } };
        joints[i].radius = 0.01f + f;
    }
    return joints;
}

class DummyTracker final : public core::ITracker
{
public:
    std::string_view get_name() const override
    {
        return "dummy";
    }
};

bool wait_until(const std::function<bool()>& predicate, std::chrono::milliseconds timeout = kWait)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!predicate())
    {
        if (std::chrono::steady_clock::now() > deadline)
        {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return true;
}

} // namespace

TEST_CASE("RemotePluginSession performs the HELLO handshake", "[remote_device_bridge][unit]")
{
    FakeBridge bridge;
    RemotePluginSession session(
        "127.0.0.1", bridge.port(), "test_app", { .schema_push = true, .hand_tracking_push = true });
    REQUIRE(session.connected());

    const auto hello = bridge.wait_for(protocol::FrameType::Hello);
    REQUIRE(hello.has_value());
    REQUIRE(hello->header.channel_id == protocol::kSessionChannelId);
    REQUIRE(hello->header.seq == 0);
    const protocol::Hello decoded = protocol::decode_hello(hello->payload);
    REQUIRE(decoded.app_name == "test_app");
    REQUIRE(decoded.requirements.schema_push);
    REQUIRE(decoded.requirements.hand_tracking_push);
    REQUIRE_FALSE(decoded.requirements.wrist_tracking_pull);
}

TEST_CASE("RemotePluginSession fails fast when the bridge refuses or is absent", "[remote_device_bridge][unit]")
{
    SECTION("HELLO_ACK with accepted=0 throws with the reason")
    {
        FakeBridge bridge(false, "no OpenXR runtime");
        try
        {
            RemotePluginSession session("127.0.0.1", bridge.port(), "test_app", {});
            FAIL("expected std::runtime_error");
        }
        catch (const std::runtime_error& e)
        {
            REQUIRE(std::string(e.what()).find("no OpenXR runtime") != std::string::npos);
        }
    }
    SECTION("nothing listening throws")
    {
        TcpSocket probe = TcpSocket::listen(Endpoint{ "127.0.0.1", 0 }, 1);
        const uint16_t free_port = probe.local_port();
        probe.close();
        REQUIRE_THROWS_AS(RemotePluginSession("127.0.0.1", free_port, "test_app", {}), std::runtime_error);
    }
    SECTION("a null transport is rejected")
    {
        REQUIRE_THROWS_AS(RemotePluginSession(std::unique_ptr<ITransport>(), "test_app", {}), std::invalid_argument);
    }
}

TEST_CASE("RemotePluginSession accepts an already connected transport", "[remote_device_bridge][unit]")
{
    FakeBridge bridge;
    RemotePluginSession session(
        std::make_unique<TcpTransport>(TcpSocket::connect(Endpoint{ "127.0.0.1", bridge.port() }, kWait)),
        "via_transport", { .hand_tracking_push = true });
    REQUIRE(session.connected());
    const auto hello = bridge.wait_for(protocol::FrameType::Hello);
    REQUIRE(hello.has_value());
    REQUIRE(protocol::decode_hello(hello->payload).app_name == "via_transport");
}

TEST_CASE("RemotePluginSession enforces declared requirements", "[remote_device_bridge][unit]")
{
    FakeBridge bridge;
    RemotePluginSession session(
        "127.0.0.1", bridge.port(), "test_app", { .schema_push = false, .hand_tracking_push = true });

    REQUIRE_THROWS_AS(session.create_schema_push_channel(core::SchemaPusherConfig{}), std::logic_error);
    core::IPluginPullChannel& pull = session.pull_channel();
    REQUIRE_THROWS_AS(pull.wrist_tracking_source(), std::logic_error);
    const DummyTracker tracker;
    REQUIRE_THROWS_AS(pull.get_tracker_impl(tracker), std::runtime_error);
    REQUIRE(bridge.wait_for(protocol::FrameType::OpenPull).has_value());
}

TEST_CASE("RemotePluginSession rejects hand-tracking push when it was not declared", "[remote_device_bridge][unit]")
{
    FakeBridge bridge;
    RemotePluginSession session(
        "127.0.0.1", bridge.port(), "test_app", { .schema_push = true, .hand_tracking_push = false });

    REQUIRE_THROWS_AS(session.create_hand_tracking_push_channel(XR_HAND_LEFT_EXT), std::logic_error);
    REQUIRE_FALSE(bridge.wait_for(protocol::FrameType::OpenHand, std::chrono::milliseconds(200)).has_value());
}

TEST_CASE("RemotePluginSession forwards OPEN_HAND, HAND_SAMPLE and CLOSE bit-exact", "[remote_device_bridge][unit]")
{
    FakeBridge bridge;
    RemotePluginSession session("127.0.0.1", bridge.port(), "test_app", { .hand_tracking_push = true });

    uint16_t channel_id = 0;
    {
        const auto channel = session.create_hand_tracking_push_channel(XR_HAND_RIGHT_EXT);
        const auto open = bridge.wait_for(protocol::FrameType::OpenHand);
        REQUIRE(open.has_value());
        REQUIRE(open->header.channel_id != protocol::kSessionChannelId);
        REQUIRE(protocol::decode_open_hand(open->payload) == XR_HAND_RIGHT_EXT);
        channel_id = open->header.channel_id;

        const auto joints = make_joints(0.25f);
        channel->push(joints.data(), 123456789LL);

        const auto sample = bridge.wait_for(protocol::FrameType::HandSample);
        REQUIRE(sample.has_value());
        REQUIRE(sample->header.channel_id == channel_id);
        const protocol::HandSample decoded = protocol::decode_hand_sample(sample->payload);
        REQUIRE(decoded.sample_time_ns == 123456789LL);
        for (size_t i = 0; i < joints.size(); ++i)
        {
            REQUIRE(std::memcmp(&decoded.joints[i], &joints[i], sizeof(XrHandJointLocationEXT)) == 0);
        }
    }

    const auto close = bridge.wait_for(protocol::FrameType::Close);
    REQUIRE(close.has_value());
    REQUIRE(close->header.channel_id == channel_id);
    REQUIRE(close->payload.empty());
}

TEST_CASE("RemotePluginSession forwards schema channels", "[remote_device_bridge][unit]")
{
    FakeBridge bridge;
    RemotePluginSession session("127.0.0.1", bridge.port(), "test_app", { .schema_push = true });

    const core::SchemaPusherConfig config{ .collection_id = "collection",
                                           .max_flatbuffer_size = 64,
                                           .tensor_identifier = "tensor",
                                           .localized_name = "Localized",
                                           .app_name = "test_app" };
    auto channel = session.create_schema_push_channel(config);

    const auto open = bridge.wait_for(protocol::FrameType::OpenSchema);
    REQUIRE(open.has_value());
    REQUIRE(protocol::decode_open_schema(open->payload).localized_name == "Localized");

    const std::vector<uint8_t> bytes = { 1, 2, 3, 4, 5 };
    channel->push_buffer(bytes.data(), bytes.size(), 77, 88);
    const auto sample = bridge.wait_for(protocol::FrameType::SchemaSample);
    REQUIRE(sample.has_value());
    REQUIRE(sample->header.channel_id == open->header.channel_id);
    const protocol::SchemaSample decoded = protocol::decode_schema_sample(sample->payload);
    REQUIRE(decoded.bytes == bytes);
    REQUIRE(decoded.sample_time_local_common_clock_ns == 77);
    REQUIRE(decoded.sample_time_raw_device_clock_ns == 88);
}

TEST_CASE("RemotePluginSession exposes bridge wrist samples through the pull channel", "[remote_device_bridge][unit]")
{
    FakeBridge bridge;
    core::WristTrackingSourceConfig config;
    config.mode = core::WristTrackingSourceMode::HandTracking;
    config.left_aim_to_wrist = { { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.5f, 0.0f, 0.0f } };
    config.right_aim_to_wrist = { { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } };
    RemotePluginSession session("127.0.0.1", bridge.port(), "test_app", { .wrist_tracking_pull = config });

    const auto hello = bridge.wait_for(protocol::FrameType::Hello);
    REQUIRE(hello.has_value());
    const auto declared = protocol::decode_hello(hello->payload).requirements.wrist_tracking_pull;
    REQUIRE(declared.has_value());
    REQUIRE(declared->mode == core::WristTrackingSourceMode::HandTracking);
    REQUIRE(declared->left_aim_to_wrist.position.x == 0.5f);
    const auto open_pull = bridge.wait_for(protocol::FrameType::OpenPull);
    REQUIRE(open_pull.has_value());

    core::IPluginPullChannel& pull = session.pull_channel();
    core::IWristTrackingSource& source = pull.wrist_tracking_source();
    pull.update();
    core::WristTrackingSample before = source.query(XR_HAND_LEFT_EXT, 0);
    REQUIRE_FALSE(before.valid);
    REQUIRE_FALSE(before.tracked);

    protocol::WristSample wrist;
    wrist.sample_time_ns = 1000;
    wrist.hand = XR_HAND_LEFT_EXT;
    wrist.sample.valid = true;
    wrist.sample.tracked = true;
    wrist.sample.pose.position = { 0.1f, 1.2f, -0.4f };
    REQUIRE(bridge.send(
        protocol::FrameType::WristSample, open_pull->header.channel_id, protocol::encode_wrist_sample(wrist)));

    REQUIRE(wait_until([&] { return source.query(XR_HAND_LEFT_EXT, 0).valid; }));
    const core::WristTrackingSample left = source.query(XR_HAND_LEFT_EXT, 0);
    REQUIRE(left.tracked);
    REQUIRE(left.pose.position.y == 1.2f);
    REQUIRE_FALSE(source.query(XR_HAND_RIGHT_EXT, 0).valid);
}

TEST_CASE("RemotePluginSession answers every PING with a PONG", "[remote_device_bridge][unit]")
{
    FakeBridge bridge;
    RemotePluginSession session("127.0.0.1", bridge.port(), "test_app", { .hand_tracking_push = true });
    REQUIRE(bridge.wait_for(protocol::FrameType::Hello).has_value());

    for (int64_t t1 : { 111LL, 222LL, 333LL })
    {
        const int64_t before = core::os_monotonic_now_ns();
        REQUIRE(bridge.send(protocol::FrameType::Ping, protocol::kSessionChannelId, protocol::encode_ping(t1)));
        const auto frame = bridge.wait_for(protocol::FrameType::Pong);
        const int64_t after = core::os_monotonic_now_ns();
        REQUIRE(frame.has_value());
        REQUIRE(frame->header.channel_id == protocol::kSessionChannelId);
        const protocol::Pong pong = protocol::decode_pong(frame->payload);
        REQUIRE(pong.ping_sent_ns == t1);
        // Same host, same clock: before <= ping_received <= pong_sent == header t_send <= after.
        REQUIRE(before <= pong.ping_received_ns);
        REQUIRE(pong.ping_received_ns <= pong.pong_sent_ns);
        REQUIRE(pong.pong_sent_ns == frame->header.t_send_monotonic_ns);
        REQUIRE(pong.pong_sent_ns <= after);

        ClockOffsetEstimator clock;
        clock.add_two_way_sample(before, pong.ping_received_ns, pong.pong_sent_ns, after);
        REQUIRE(clock.rtt_last_ns() >= 0);
        REQUIRE(clock.rtt_last_ns() <= after - before);
    }
}

TEST_CASE("RemotePluginSession keeps the session when the bridge refuses one channel", "[remote_device_bridge][unit]")
{
    FakeBridge bridge;
    RemotePluginSession session(
        "127.0.0.1", bridge.port(), "test_app", { .schema_push = true, .hand_tracking_push = true });
    core::SchemaPusherConfig config;
    config.collection_id = "tactile";
    config.max_flatbuffer_size = 40960;
    auto refused = session.create_schema_push_channel(config);
    auto hand = session.create_hand_tracking_push_channel(XR_HAND_LEFT_EXT);
    const auto open_schema = bridge.wait_for(protocol::FrameType::OpenSchema);
    const auto open_hand = bridge.wait_for(protocol::FrameType::OpenHand);
    REQUIRE(open_schema.has_value());
    REQUIRE(open_hand.has_value());

    SECTION("ERROR on a channel id refuses only that channel")
    {
        const std::string reason =
            "schema 'tactile' sample size 40960 bytes exceeds bridge cap 32768 bytes (runtime "
            "limit); channel refused";
        REQUIRE(bridge.send(protocol::FrameType::Error, open_schema->header.channel_id, protocol::encode_error(reason)));
        REQUIRE(wait_until([&] { return session.stats().refused_channels.count(open_schema->header.channel_id) == 1; }));
        REQUIRE(session.stats().refused_channels.at(open_schema->header.channel_id) == reason);
        REQUIRE(session.connected());

        const std::vector<uint8_t> bytes(16, 0x5A);
        refused->push_buffer(bytes.data(), bytes.size(), 1, 1);
        const auto joints = make_joints(0.3f);
        hand->push(joints.data(), 42);
        const auto sample = bridge.wait_for(protocol::FrameType::HandSample);
        REQUIRE(sample.has_value());
        REQUIRE(sample->header.channel_id == open_hand->header.channel_id);
        REQUIRE_FALSE(bridge.wait_for(protocol::FrameType::SchemaSample, std::chrono::milliseconds(200)).has_value());
        REQUIRE(session.connected());

        refused.reset(); // CLOSE still goes out so the bridge forgets the refused id
        const auto close = bridge.wait_for(protocol::FrameType::Close);
        REQUIRE(close.has_value());
        REQUIRE(close->header.channel_id == open_schema->header.channel_id);
    }
    SECTION("ERROR on channel 0 ends the session")
    {
        REQUIRE(bridge.send(protocol::FrameType::Error, protocol::kSessionChannelId, protocol::encode_error("fatal")));
        REQUIRE(wait_until([&] { return !session.connected(); }));
        REQUIRE(session.stats().refused_channels.empty());
    }
}

TEST_CASE("RemotePluginSession survives the bridge dropping the socket", "[remote_device_bridge][unit]")
{
    FakeBridge bridge;
    std::unique_ptr<core::IHandTrackingPushChannel> channel;
    {
        RemotePluginSession session("127.0.0.1", bridge.port(), "test_app",
                                    { .hand_tracking_push = true, .wrist_tracking_pull = kWristConfig });
        channel = session.create_hand_tracking_push_channel(XR_HAND_LEFT_EXT);
        REQUIRE(bridge.wait_for(protocol::FrameType::OpenHand).has_value());

        bridge.drop_connection();
        REQUIRE(wait_until([&] { return !session.connected(); }));

        const auto joints = make_joints(0.5f);
        REQUIRE_NOTHROW(channel->push(joints.data(), 1));
        REQUIRE_NOTHROW(session.pull_channel().update());
        REQUIRE_THROWS_AS(session.create_hand_tracking_push_channel(XR_HAND_RIGHT_EXT), std::runtime_error);
    }
    // Channels outliving a dead session must still be destroyable.
    REQUIRE_NOTHROW(channel->push(make_joints(0.5f).data(), 2));
    REQUIRE_NOTHROW(channel.reset());
}

TEST_CASE("RemotePluginSession never reconnects after the link drops", "[remote_device_bridge][unit]")
{
    TcpSocket listener = TcpSocket::listen(Endpoint{ "127.0.0.1", 0 }, 4);
    std::unique_ptr<ITransport> bridge_side;
    std::thread accept_once(
        [&]
        {
            bridge_side = TcpTransport::accept(listener, kWait);
            std::vector<uint8_t> hello;
            if (bridge_side && bridge_side->receive(hello, kWait) == ReceiveResult::Ok)
            {
                send_frame(*bridge_side, protocol::FrameType::HelloAck, protocol::kSessionChannelId, 0,
                           protocol::encode_hello_ack(protocol::HelloAck{ true, "" }));
            }
        });
    RemotePluginSession session("127.0.0.1", listener.local_port(), "test_app", { .hand_tracking_push = true });
    accept_once.join();
    REQUIRE(bridge_side);
    auto channel = session.create_hand_tracking_push_channel(XR_HAND_LEFT_EXT);

    bridge_side->shutdown();
    REQUIRE(wait_until([&] { return !session.connected(); }));
    const auto joints = make_joints(0.4f);
    for (int i = 0; i < 20; ++i)
    {
        channel->push(joints.data(), i);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    // A new HELLO needs a new session (a restarted plugin); nothing connects back on its own.
    REQUIRE(TcpTransport::accept(listener, std::chrono::milliseconds(500)) == nullptr);
    REQUIRE_FALSE(session.connected());
    REQUIRE_THROWS_AS(session.create_hand_tracking_push_channel(XR_HAND_RIGHT_EXT), std::runtime_error);
}

TEST_CASE("RemotePluginSession closes remaining channels and the socket on destruction", "[remote_device_bridge][unit]")
{
    FakeBridge bridge;
    std::unique_ptr<core::IHandTrackingPushChannel> channel;
    {
        RemotePluginSession session("127.0.0.1", bridge.port(), "test_app", { .hand_tracking_push = true });
        channel = session.create_hand_tracking_push_channel(XR_HAND_LEFT_EXT);
        REQUIRE(bridge.wait_for(protocol::FrameType::OpenHand).has_value());
    }
    REQUIRE(bridge.wait_for(protocol::FrameType::Close).has_value());
    REQUIRE(bridge.client_closed());
    REQUIRE_NOTHROW(channel.reset());
}

TEST_CASE("RemotePluginSession keeps the newest samples per channel while the link is stalled",
          "[remote_device_bridge][unit]")
{
    auto transport = std::make_unique<BlockingTransport>();
    BlockingTransport& link = *transport;
    RemotePluginSession session(std::move(transport), "test_app", { .schema_push = true, .hand_tracking_push = true });
    core::SchemaPusherConfig filler_config;
    filler_config.collection_id = "filler";
    auto filler = session.create_schema_push_channel(filler_config);
    auto fast = session.create_hand_tracking_push_channel(XR_HAND_LEFT_EXT); // "A": 180 Hz
    auto slow = session.create_hand_tracking_push_channel(XR_HAND_RIGHT_EXT); // "B": 60 Hz
    REQUIRE(link.wait_for(protocol::FrameType::OpenSchema).has_value());
    const auto open_fast = link.wait_for(protocol::FrameType::OpenHand);
    const auto open_slow = link.wait_for(protocol::FrameType::OpenHand);
    REQUIRE(open_fast.has_value());
    REQUIRE(open_slow.has_value());
    const uint16_t fast_id = open_fast->header.channel_id;
    const uint16_t slow_id = open_slow->header.channel_id;

    // Stall the link: the sender thread blocks inside send() on the filler sample, so every hand sample pushed
    // from here on can only wait in its channel's queue.
    link.hold(true);
    const std::vector<uint8_t> filler_bytes = { 0xAB };
    filler->push_buffer(filler_bytes.data(), filler_bytes.size(), 0, 0);
    REQUIRE(wait_until([&] { return link.blocked(); }));

    const auto joints = make_joints(0.1f);
    constexpr int kFastPushes = 180;
    constexpr int kSlowPushes = kFastPushes / 3;
    for (int i = 0; i < kFastPushes; ++i)
    {
        fast->push(joints.data(), 1000 + i);
        if (i % 3 == 0)
        {
            slow->push(joints.data(), 2000 + i / 3);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    const RemotePluginSession::Stats stalled = session.stats();
    REQUIRE(stalled.dropped_by_channel.at(fast_id) > 0);
    REQUIRE(stalled.dropped_by_channel.at(slow_id) > 0);
    REQUIRE(stalled.dropped_by_channel.at(fast_id) > stalled.dropped_by_channel.at(slow_id));
    REQUIRE(stalled.dropped_queue >= stalled.dropped_by_channel.at(fast_id) + stalled.dropped_by_channel.at(slow_id));
    REQUIRE(session.connected());

    link.hold(false);
    std::vector<int64_t> fast_times;
    std::vector<int64_t> slow_times;
    const auto deadline = std::chrono::steady_clock::now() + kWait;
    while (std::chrono::steady_clock::now() < deadline &&
           (fast_times.empty() || fast_times.back() != 1000 + kFastPushes - 1 || slow_times.empty() ||
            slow_times.back() != 2000 + kSlowPushes - 1))
    {
        const auto frame = link.wait_for(protocol::FrameType::HandSample, std::chrono::milliseconds(200));
        if (!frame.has_value())
        {
            continue;
        }
        (frame->header.channel_id == fast_id ? fast_times : slow_times)
            .push_back(protocol::decode_hand_sample(frame->payload).sample_time_ns);
    }

    // Each channel delivered its newest sample, in order, from a queue that never held more than 4.
    REQUIRE_FALSE(fast_times.empty());
    REQUIRE_FALSE(slow_times.empty());
    REQUIRE(fast_times.back() == 1000 + kFastPushes - 1);
    REQUIRE(slow_times.back() == 2000 + kSlowPushes - 1);
    REQUIRE(fast_times.size() <= 4);
    REQUIRE(slow_times.size() <= 4);
    REQUIRE(std::is_sorted(fast_times.begin(), fast_times.end()));
    REQUIRE(std::is_sorted(slow_times.begin(), slow_times.end()));
    REQUIRE(session.stats().sent_hand == fast_times.size() + slow_times.size());
}

TEST_CASE("RemotePluginSession drops a channel's pending samples when the channel closes", "[remote_device_bridge][unit]")
{
    auto transport = std::make_unique<BlockingTransport>();
    BlockingTransport& link = *transport;
    RemotePluginSession session(std::move(transport), "test_app", { .schema_push = true, .hand_tracking_push = true });
    core::SchemaPusherConfig filler_config;
    filler_config.collection_id = "filler";
    auto filler = session.create_schema_push_channel(filler_config);
    REQUIRE(link.wait_for(protocol::FrameType::OpenSchema).has_value());

    // Stall the sender on a filler sample. A paused TCP peer is not enough: with 8 MiB socket buffers the kernel
    // absorbs the filler and the hand samples leave before the CLOSE.
    link.hold(true);
    const std::vector<uint8_t> filler_bytes = { 0xCD };
    filler->push_buffer(filler_bytes.data(), filler_bytes.size(), 0, 0);
    REQUIRE(wait_until([&] { return link.blocked(); }));
    {
        auto channel = session.create_hand_tracking_push_channel(XR_HAND_LEFT_EXT);
        const auto joints = make_joints(0.2f);
        for (int i = 0; i < 3; ++i)
        {
            channel->push(joints.data(), i);
        }
    }
    link.hold(false);

    // OPEN_HAND and CLOSE go out in order; the samples queued behind the stalled filler never follow the CLOSE.
    const auto open = link.wait_for(protocol::FrameType::OpenHand);
    REQUIRE(open.has_value());
    const auto close = link.wait_for(protocol::FrameType::Close);
    REQUIRE(close.has_value());
    REQUIRE(close->header.channel_id == open->header.channel_id);
    REQUIRE_FALSE(link.wait_for(protocol::FrameType::HandSample, std::chrono::milliseconds(200)).has_value());
}
