// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Both ends of a remote session over loopback: a RemoteSessionHost recording a live DeviceIOSession, and a
// RemoteSessionClient feeding a RemoteSession. The server's KeyboardTracker never touches the OpenXR session
// handles, so placeholder handles stand in for a runtime; plugin-fed trackers read a fake tensor runtime.

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <deviceio_base/tracker_vendor.hpp>
#include <deviceio_session/deviceio_session.hpp>
#include <deviceio_session/remote_session.hpp>
#include <deviceio_session/remote_session_client.hpp>
#include <deviceio_session/remote_session_host.hpp>
#include <deviceio_trackers/full_body_tracker.hpp>
#include <deviceio_trackers/generic_3axis_pedal_tracker.hpp>
#include <deviceio_trackers/keyboard_tracker.hpp>
#include <mcap/sample_stream.hpp>
#include <oxr_utils/os_time.hpp>
#include <oxr_utils/oxr_session_handles.hpp>
#include <remote_link/message_header.hpp>
#include <remote_link/messages.hpp>
#include <remote_link/reject.hpp>
#include <remote_link/tcp_transport.hpp>
#include <remote_link/websocket_transport.hpp>
#include <schema/full_body_generated.h>
#include <schema/keyboard_generated.h>
#include <schema/pedals_generated.h>
#include <schema/serialized.hpp>
#include <sys/socket.h>
#if defined(XR_USE_TIMESPEC)
#    include <tensor_test_support/fake_tensor_runtime.hpp>
#endif

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace
{

using namespace std::chrono_literals;
using Catch::Matchers::Message;
using Catch::Matchers::StartsWith;

constexpr std::chrono::milliseconds kWait{ 5000 };
constexpr uint16_t KEY_W = 17;
constexpr uint16_t KEY_A = 30;
constexpr uint16_t KEY_K = 37;

const std::string kHello = R"({"protocol": 1})";
const std::string kHelloAck = R"({"session_id": "test", "server_version": "0"})";
const std::string kWaiting = R"({"stage": "waiting_for_headset"})";
const std::string kReady = R"({"stage": "ready"})";

// Non-null values the keyboard impl never dereferences.
core::OpenXRSessionHandles placeholder_handles()
{
    core::OpenXRSessionHandles handles;
    handles.instance = reinterpret_cast<XrInstance>(1);
    handles.session = reinterpret_cast<XrSession>(1);
    handles.space = reinterpret_cast<XrSpace>(1);
    return handles;
}

bool wait_until(const std::function<bool()>& predicate, std::chrono::milliseconds timeout = kWait)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!predicate())
    {
        if (std::chrono::steady_clock::now() > deadline)
        {
            return false;
        }
        std::this_thread::sleep_for(2ms);
    }
    return true;
}

bool fails(const core::RemoteSessionClient& client)
{
    try
    {
        client.throw_if_failed();
        return false;
    }
    catch (const std::runtime_error&)
    {
        return true;
    }
}

std::vector<uint8_t> bytes(const std::string& text)
{
    return std::vector<uint8_t>(text.begin(), text.end());
}

void send_raw(core::ITransport& transport, core::LinkMessageType type, const std::string& payload)
{
    std::vector<uint8_t> message = core::encode_link_message(static_cast<uint8_t>(type), bytes(payload));
    core::stamp_header(message.data(), 0, core::os_monotonic_now_ns());
    transport.send(message.data(), message.size());
}

std::string tcp_endpoint(const core::Listener& listener, bool websocket = false)
{
    return std::string(websocket ? "ws://" : "tcp://") + "127.0.0.1:" + std::to_string(listener.port());
}

//! The keys `keyboard` holds in `session`, or nullopt when it reports no data.
std::optional<std::vector<uint16_t>> held_keys(const core::KeyboardTracker& keyboard, const core::ITrackerSession& session)
{
    const auto& data = keyboard.get_data(session);
    if (!data)
    {
        return std::nullopt;
    }
    std::vector<uint16_t> held;
    if (const auto* keys = data->held_keys())
    {
        held.assign(keys->begin(), keys->end());
    }
    return held;
}

//! The robot end: keyboards "0", "1", ... read through a RemoteSession that a RemoteSessionClient feeds.
struct Robot
{
    explicit Robot(int keyboard_count = 1, int64_t max_tick_age_ns = 250'000'000)
    {
        std::vector<std::pair<const core::ITracker*, std::string>> names;
        for (int i = 0; i < keyboard_count; ++i)
        {
            keyboards.push_back(std::make_unique<core::KeyboardTracker>());
            names.emplace_back(keyboards.back().get(), std::to_string(i));
        }
        session = core::RemoteSession::run({ stream, names, max_tick_age_ns });
    }

    std::vector<std::string> topics() const
    {
        return stream->subscribed_topics();
    }

    void connect(const std::string& endpoint, core::LinkOptions options = {})
    {
        client = core::RemoteSessionClient::connect(endpoint, kHello, stream, kWait, std::move(options));
    }

    //! Updates until keyboard `index` holds exactly `keys`.
    bool wait_for_keys(size_t index, const std::vector<uint16_t>& keys)
    {
        return wait_until(
            [&]
            {
                session->update();
                return held_keys(*keyboards[index], *session) == keys;
            });
    }

    std::vector<std::unique_ptr<core::KeyboardTracker>> keyboards;
    std::shared_ptr<core::TrackerSampleStreamReader> stream = std::make_shared<core::TrackerSampleStreamReader>();
    std::unique_ptr<core::RemoteSession> session;
    std::unique_ptr<core::RemoteSessionClient> client;
};

//! The server end: accepts one connection on its own thread and runs `script` with its RemoteSessionHost.
class Server
{
public:
    using Script = std::function<void(core::RemoteSessionHost& host, const std::atomic<bool>& stop)>;

    //! `accept_delay` leaves the connection in the listen backlog for that long, as a server busy handing over
    //! from its previous session does.
    Server(bool websocket, Script script, std::chrono::milliseconds accept_delay = 0ms) : m_websocket(websocket)
    {
        m_thread = std::jthread([this, script = std::move(script), accept_delay] { run(script, accept_delay); });
    }

    ~Server()
    {
        stop();
    }

    std::string endpoint() const
    {
        return tcp_endpoint(m_listener, m_websocket);
    }

    //! Sets the script's stop flag and waits until the host is closed.
    void stop()
    {
        m_stop = true;
        if (m_thread.joinable())
        {
            m_thread.join();
        }
    }

    //! What the script threw, if anything.
    std::string error() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_error;
    }

private:
    void run(const Script& script, std::chrono::milliseconds accept_delay)
    {
        try
        {
            std::this_thread::sleep_for(accept_delay);
            const std::optional<core::Listener::Accepted> accepted = m_listener.accept(kWait);
            if (!accepted)
            {
                throw std::runtime_error("no connection");
            }
            const auto host = core::RemoteSessionHost::adopt(accepted->fd, m_websocket);
            script(*host, m_stop);
            host->close();
        }
        catch (const std::exception& e)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_error = e.what();
        }
    }

    bool m_websocket;
    core::Listener m_listener{ core::Endpoint{ "127.0.0.1", 0 } };
    std::atomic<bool> m_stop{ false };
    mutable std::mutex m_mutex;
    std::string m_error;
    std::jthread m_thread;
};

//! A server session with one keyboard ("0") holding W.
struct KeyboardServer
{
    std::shared_ptr<core::KeyboardTracker> keyboard = std::make_shared<core::KeyboardTracker>();
    std::shared_ptr<core::KeyboardProvider> provider;

    //! HELLO, HELLO_ACK, the session, then STATUS.
    void start(core::RemoteSessionHost& host, const std::vector<std::string>& topics)
    {
        if (host.wait_hello(kWait) != kHello)
        {
            throw std::runtime_error("no HELLO");
        }
        host.send_hello_ack(kHelloAck);
        host.start_session({ keyboard }, { { keyboard.get(), "0" } }, placeholder_handles(), {}, topics);
        provider = keyboard->create_provider();
        provider->key_down(KEY_W);
        host.send_status(kWaiting);
    }

    //! A tick, so a SAMPLES link message, every 5 ms while `running()` holds and the link lives.
    static void run(core::RemoteSessionHost& host, const std::function<bool()>& running)
    {
        while (running() && host.alive())
        {
            host.tick();
            std::this_thread::sleep_for(5ms);
        }
    }
};

//! The usual script: READY at once, then ticks until the test stops the server.
Server::Script serve_until_stopped(const Robot& robot)
{
    return [&robot](core::RemoteSessionHost& host, const std::atomic<bool>& stop)
    {
        KeyboardServer server;
        server.start(host, robot.topics());
        host.send_ready(kReady);
        KeyboardServer::run(host, [&] { return !stop; });
        host.close();
    };
}

/*!
 * Wraps a transport; while held, send() blocks and wait_writable() reports false, as on a congested network, so
 * latest-wins link messages wait in the Link. Records the tick_seq of every SAMPLES it sends.
 */
class HoldableTransport final : public core::ITransport
{
public:
    explicit HoldableTransport(std::unique_ptr<core::ITransport> inner) : m_inner(std::move(inner))
    {
    }

    bool send(const uint8_t* message, size_t size) override
    {
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_cv.wait(lock, [this] { return !m_held || m_shut; });
            if (m_shut)
            {
                return false;
            }
            const core::LinkMessageHeader header = core::decode_header(message);
            if (header.type == static_cast<uint8_t>(core::LinkMessageType::Samples))
            {
                m_samples.push_back(core::decode_samples_prefix(
                                        std::span<const uint8_t>(message + core::kHeaderSize, header.payload_size))
                                        .tick_seq);
            }
        }
        return m_inner->send(message, size);
    }

    core::ReceiveResult receive(std::vector<uint8_t>& message, std::chrono::milliseconds timeout) override
    {
        return m_inner->receive(message, timeout);
    }

    void shutdown() override
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_shut = true;
        }
        m_cv.notify_all();
        m_inner->shutdown();
    }

    void shutdown_write() override
    {
        m_inner->shutdown_write();
    }

    bool wait_writable(std::chrono::milliseconds timeout) override
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_held)
            {
                return false;
            }
        }
        return m_inner->wait_writable(timeout);
    }

    void hold(bool held)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_held = held;
        }
        m_cv.notify_all();
    }

    std::vector<uint64_t> samples_sent() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_samples;
    }

private:
    std::unique_ptr<core::ITransport> m_inner;
    mutable std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_held = false;
    bool m_shut = false;
    std::vector<uint64_t> m_samples;
};

} // namespace

TEST_CASE("RemoteSessionClient: a RemoteSession reads the server's keyboard over TCP and WebSocket",
          "[remote][session][network]")
{
    const bool websocket = GENERATE(false, true);
    Robot robot;
    Server server(websocket, serve_until_stopped(robot));

    robot.connect(server.endpoint());
    CHECK(robot.client->hello_ack() == kHelloAck);
    REQUIRE(robot.client->wait_ready(kWait) == core::RemoteReadyState::Ready);
    CHECK(robot.client->latest_status() == kReady);

    REQUIRE(robot.wait_for_keys(0, { KEY_W }));
    CHECK(robot.session->is_fresh());
    REQUIRE(robot.session->tick_age_ns().has_value());
    CHECK(std::llabs(*robot.session->tick_age_ns()) < 250'000'000);
    CHECK(robot.stream->dropped_messages() == 0);
    REQUIRE(wait_until([&] { return robot.client->clock().two_way; }));
    CHECK(robot.client->alive());
    CHECK_NOTHROW(robot.client->throw_if_failed());

    server.stop();
    CHECK(server.error().empty());
}

TEST_CASE("RemoteSessionClient: wait_ready reports Pending until READY arrives", "[remote][session][network]")
{
    Robot robot;
    std::atomic<bool> send_ready{ false };
    Server server(false,
                  [&](core::RemoteSessionHost& host, const std::atomic<bool>& stop)
                  {
                      KeyboardServer keyboard;
                      keyboard.start(host, robot.topics());
                      KeyboardServer::run(host, [&] { return !stop && !send_ready; });
                      host.send_ready(kReady);
                      KeyboardServer::run(host, [&] { return !stop; });
                      host.close();
                  });

    robot.connect(server.endpoint());
    const auto start = std::chrono::steady_clock::now();
    CHECK(robot.client->wait_ready(100ms) == core::RemoteReadyState::Pending);
    CHECK(std::chrono::steady_clock::now() - start >= 100ms);
    REQUIRE(wait_until([&] { return robot.client->latest_status() == kWaiting; }));
    // SAMPLES flow before READY.
    REQUIRE(robot.wait_for_keys(0, { KEY_W }));

    send_ready = true;
    CHECK(robot.client->wait_ready(kWait) == core::RemoteReadyState::Ready);
    CHECK(robot.client->latest_status() == kReady);
    server.stop();
    CHECK(server.error().empty());
}

TEST_CASE("RemoteSessionClient: the server's ERROR text is the failure", "[remote][session][network]")
{
    const bool websocket = GENERATE(false, true);
    Robot robot;

    SECTION("instead of HELLO_ACK")
    {
        Server server(websocket,
                      [](core::RemoteSessionHost& host, const std::atomic<bool>&)
                      {
                          if (host.wait_hello(kWait))
                          {
                              host.send_error("unknown plugin 'nope'");
                          }
                      });
        CHECK_THROWS_MATCHES(robot.connect(server.endpoint()), std::runtime_error, Message("unknown plugin 'nope'"));
    }
    SECTION("while running, and after the link closed")
    {
        std::atomic<bool> fail{ false };
        Server server(websocket,
                      [&](core::RemoteSessionHost& host, const std::atomic<bool>& stop)
                      {
                          KeyboardServer keyboard;
                          keyboard.start(host, robot.topics());
                          host.send_ready(kReady);
                          KeyboardServer::run(host, [&] { return !stop && !fail; });
                          host.send_error("headset lost");
                          host.close();
                      });
        robot.connect(server.endpoint());
        REQUIRE(robot.client->wait_ready(kWait) == core::RemoteReadyState::Ready);
        fail = true;
        REQUIRE(wait_until([&] { return fails(*robot.client); }));
        CHECK_THROWS_MATCHES(robot.client->throw_if_failed(), std::runtime_error, Message("headset lost"));

        // The server closed after the ERROR; the link is down now, and the ERROR is still the reason.
        REQUIRE(wait_until([&] { return !robot.client->alive(); }));
        CHECK_THROWS_MATCHES(robot.client->throw_if_failed(), std::runtime_error, Message("headset lost"));
        CHECK_THROWS_MATCHES(robot.client->wait_ready(0ms), std::runtime_error, Message("headset lost"));
        server.stop();
        CHECK(server.error().empty());
    }
}

TEST_CASE("RemoteSessionClient: the session is lost when the server closes", "[remote][session][network]")
{
    Robot robot;
    Server server(false, serve_until_stopped(robot));
    robot.connect(server.endpoint());
    REQUIRE(robot.client->wait_ready(kWait) == core::RemoteReadyState::Ready);
    REQUIRE(robot.wait_for_keys(0, { KEY_W }));

    server.stop();
    REQUIRE(wait_until([&] { return !robot.client->alive(); }));
    CHECK_THROWS_MATCHES(robot.client->throw_if_failed(), std::runtime_error,
                         Message("Remote session lost: receive failed: connection closed"));
    CHECK_THROWS_AS(robot.client->wait_ready(0ms), std::runtime_error);
    // Without new SAMPLES the stream goes stale, and the keyboard reports no data.
    REQUIRE(wait_until(
        [&]
        {
            robot.session->update();
            return !robot.session->is_fresh();
        }));
    CHECK_FALSE(held_keys(*robot.keyboards[0], *robot.session));
    CHECK(server.error().empty());
}

TEST_CASE("RemoteSessionClient: a silent server is lost after the peer timeout", "[remote][session][network]")
{
    core::Listener listener(core::Endpoint{ "127.0.0.1", 0 });
    std::atomic<bool> done{ false };
    // Answers the HELLO, then reads and never sends.
    std::jthread server(
        [&]
        {
            const std::optional<core::Listener::Accepted> accepted = listener.accept(kWait);
            if (!accepted)
            {
                return;
            }
            const auto transport = core::adopt_transport(accepted->fd, false);
            std::vector<uint8_t> message;
            while (!done)
            {
                const core::ReceiveResult result = transport->receive(message, 100ms);
                if (result == core::ReceiveResult::Timeout)
                {
                    continue;
                }
                if (result != core::ReceiveResult::Ok)
                {
                    return;
                }
                if (core::decode_header(message.data()).type == static_cast<uint8_t>(core::LinkMessageType::Hello))
                {
                    send_raw(*transport, core::LinkMessageType::HelloAck, kHelloAck);
                }
            }
        });

    Robot robot;
    core::LinkOptions options;
    options.heartbeat_period = 100ms;
    options.peer_timeout = 300ms;
    robot.connect(tcp_endpoint(listener), options);
    const auto start = std::chrono::steady_clock::now();
    REQUIRE(wait_until([&] { return !robot.client->alive(); }));
    CHECK(std::chrono::steady_clock::now() - start >= 200ms);
    CHECK_THROWS_WITH(robot.client->throw_if_failed(), StartsWith("Remote session lost: no messages for "));
    done = true;
}

TEST_CASE("RemoteSessionClient: connect gives up without HELLO_ACK", "[remote][session][network]")
{
    core::Listener listener(core::Endpoint{ "127.0.0.1", 0 });
    std::atomic<bool> done{ false };
    std::jthread server(
        [&]
        {
            const std::optional<core::Listener::Accepted> accepted = listener.accept(kWait);
            if (!accepted)
            {
                return;
            }
            const auto transport = core::adopt_transport(accepted->fd, false);
            std::vector<uint8_t> message;
            while (!done && transport->receive(message, 100ms) != core::ReceiveResult::Closed)
            {
            }
        });

    Robot robot;
    CHECK_THROWS_MATCHES(core::RemoteSessionClient::connect(tcp_endpoint(listener), kHello, robot.stream, 200ms),
                         core::RemoteConnectTimeoutError,
                         Message("No HELLO_ACK from the Teleop Session Server before the connect timeout"));
    done = true;
}

TEST_CASE("RemoteSessionClient: connect reports a missing server and bad arguments", "[remote][session][network]")
{
    uint16_t port = 0;
    {
        core::Listener listener(core::Endpoint{ "127.0.0.1", 0 });
        port = listener.port();
    }
    Robot robot;
    CHECK_THROWS_WITH(robot.connect("127.0.0.1:" + std::to_string(port)),
                      StartsWith("cannot connect to 127.0.0.1:" + std::to_string(port)));
    CHECK_THROWS_AS(robot.connect("wss://127.0.0.1:1"), std::invalid_argument);
    CHECK_THROWS_AS(core::RemoteSessionClient::connect("127.0.0.1:1", kHello, nullptr, kWait), std::invalid_argument);
}

TEST_CASE("RemoteSessionClient: connect fails with the text of a refusal", "[remote][session][network]")
{
    const bool websocket = GENERATE(false, true);
    const std::string refusal = "busy: another robot app is connected to this Teleop Session Server";
    core::Listener listener(core::Endpoint{ "127.0.0.1", 0 });
    std::jthread rejecter(
        [&]
        {
            if (const std::optional<core::Listener::Accepted> accepted = listener.accept(kWait))
            {
                core::reject_remote_connection(accepted->fd, websocket, refusal);
            }
        });
    Robot robot;
    CHECK_THROWS_MATCHES(robot.connect(tcp_endpoint(listener, websocket)), std::runtime_error, Message(refusal));
}

TEST_CASE("RemoteSessionHost: wait_hello times out, then fails once the robot is gone", "[remote][session][network]")
{
    core::Listener listener(core::Endpoint{ "127.0.0.1", 0 });
    std::unique_ptr<core::ITransport> robot;
    std::jthread connector([&] { robot = core::connect_transport(core::parse_remote(tcp_endpoint(listener)), kWait); });
    const std::optional<core::Listener::Accepted> accepted = listener.accept(kWait);
    REQUIRE(accepted.has_value());
    connector.join();
    REQUIRE(robot);

    const auto host = core::RemoteSessionHost::adopt(accepted->fd, false);
    CHECK_FALSE(host->wait_hello(50ms).has_value());
    CHECK(host->alive());
    CHECK(host->close_reason().empty());
    CHECK_THROWS_AS(host->tick(), std::logic_error);

    robot->shutdown();
    CHECK_THROWS_MATCHES(
        host->wait_hello(kWait), std::runtime_error, Message("Remote session lost: receive failed: connection closed"));
    CHECK_FALSE(host->alive());
}

TEST_CASE("RemoteSessionHost: SAMPLES the link cannot send yet merge into one", "[remote][session][network]")
{
    int fds[2];
    REQUIRE(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, fds) == 0);
    auto holdable = std::make_unique<HoldableTransport>(std::make_unique<core::TcpTransport>(core::TcpSocket(fds[1])));
    HoldableTransport& link = *holdable;
    const auto host = core::RemoteSessionHost::adopt(std::move(holdable));

    // A generous tick age limit: the merged SAMPLES waits in the link while it is held.
    Robot robot(2, 5'000'000'000);
    std::jthread handshake(
        [&]
        {
            try
            {
                if (host->wait_hello(kWait))
                {
                    host->send_hello_ack(kHelloAck);
                }
            }
            catch (const std::exception&)
            {
            }
        });
    robot.client = core::RemoteSessionClient::connect(
        std::make_unique<core::TcpTransport>(core::TcpSocket(fds[0])), kHello, robot.stream, kWait);
    handshake.join();

    auto keyboard0 = std::make_shared<core::KeyboardTracker>();
    auto keyboard1 = std::make_shared<core::KeyboardTracker>();
    host->start_session({ keyboard0, keyboard1 }, { { keyboard0.get(), "0" }, { keyboard1.get(), "1" } },
                        placeholder_handles(), {}, robot.topics());
    const auto provider0 = keyboard0->create_provider();
    const auto provider1 = keyboard1->create_provider();
    provider0->key_down(KEY_W);
    uint64_t ticks = 0;
    REQUIRE(wait_until(
        [&]
        {
            host->tick();
            ++ticks;
            robot.session->update();
            return held_keys(*robot.keyboards[0], *robot.session) == std::vector<uint16_t>{ KEY_W };
        }));
    // Nothing left to send once the last tick's SAMPLES went out.
    REQUIRE(wait_until(
        [&]
        {
            const auto sent = link.samples_sent();
            return !sent.empty() && sent.back() == ticks;
        }));

    link.hold(true);
    const std::vector<uint64_t> before = link.samples_sent();
    provider1->key_down(KEY_K);
    host->tick();
    provider0->key_up(KEY_W);
    host->tick();
    provider0->key_down(KEY_A);
    host->tick();
    std::this_thread::sleep_for(50ms);
    CHECK(link.samples_sent() == before);

    link.hold(false);
    REQUIRE(robot.wait_for_keys(0, { KEY_A }));
    REQUIRE(robot.wait_for_keys(1, { KEY_K }));
    std::vector<uint64_t> expected = before;
    expected.push_back(ticks + 3);
    CHECK(link.samples_sent() == expected);
    CHECK(robot.stream->dropped_messages() == 0);
    host->close();
}

TEST_CASE("RemoteSessionClient: connect waits past the peer timeout for a server that takes the connection up late",
          "[remote][session][network]")
{
    const bool websocket = GENERATE(false, true);
    Robot robot;
    Server server(websocket, serve_until_stopped(robot), 800ms);
    core::LinkOptions options;
    options.heartbeat_period = 50ms;
    options.peer_timeout = 200ms;
    const auto start = std::chrono::steady_clock::now();
    robot.connect(server.endpoint(), options);
    CHECK(std::chrono::steady_clock::now() - start >= 800ms);
    CHECK(robot.client->hello_ack() == kHelloAck);
    REQUIRE(robot.client->wait_ready(kWait) == core::RemoteReadyState::Ready);
    REQUIRE(robot.wait_for_keys(0, { KEY_W }));
    // The server's heartbeats keep the link up from its first message on.
    std::this_thread::sleep_for(400ms);
    CHECK(robot.client->alive());
    server.stop();
    CHECK(server.error().empty());
}

TEST_CASE("RemoteSessionClient: start returns before HELLO_ACK, which wait_hello_ack waits for",
          "[remote][session][network]")
{
    const bool websocket = GENERATE(false, true);
    Robot robot;
    Server server(websocket, serve_until_stopped(robot), 300ms);
    const auto start = std::chrono::steady_clock::now();
    robot.client = core::RemoteSessionClient::start(server.endpoint(), kHello, robot.stream, kWait);
    CHECK(std::chrono::steady_clock::now() - start < 300ms);
    CHECK_FALSE(robot.client->wait_hello_ack(20ms));
    CHECK(robot.client->hello_ack().empty());
    REQUIRE(robot.client->wait_hello_ack(kWait));
    CHECK(robot.client->hello_ack() == kHelloAck);
    CHECK(robot.client->wait_hello_ack(0ms));
    REQUIRE(robot.client->wait_ready(kWait) == core::RemoteReadyState::Ready);
    REQUIRE(robot.wait_for_keys(0, { KEY_W }));
    server.stop();
    CHECK(server.error().empty());
}

TEST_CASE("RemoteSessionClient: wait_ready alone completes what start began", "[remote][session][network]")
{
    const bool websocket = GENERATE(false, true);
    Robot robot;
    Server server(websocket, serve_until_stopped(robot));
    robot.client = core::RemoteSessionClient::start(server.endpoint(), kHello, robot.stream, kWait);
    if (websocket)
    {
        // Until a wait completes the upgrade, no HELLO goes out and nothing arrives.
        CHECK_THROWS_WITH(robot.client->throw_if_failed(), StartsWith("The WebSocket upgrade is not complete yet"));
    }
    REQUIRE(robot.client->wait_ready(kWait) == core::RemoteReadyState::Ready);
    CHECK(robot.client->hello_ack() == kHelloAck);
    CHECK_NOTHROW(robot.client->throw_if_failed());
    REQUIRE(robot.wait_for_keys(0, { KEY_W }));
    server.stop();
    CHECK(server.error().empty());
}

TEST_CASE("RemoteSessionClient: start reports a connection nothing accepts; wait_hello_ack a refused upgrade",
          "[remote][session][network]")
{
    Robot robot;
    SECTION("nothing accepts")
    {
        uint16_t port = 0;
        {
            core::Listener listener(core::Endpoint{ "127.0.0.1", 0 });
            port = listener.port();
        }
        CHECK_THROWS_WITH(
            core::RemoteSessionClient::start("127.0.0.1:" + std::to_string(port), kHello, robot.stream, kWait),
            StartsWith("cannot connect to 127.0.0.1:" + std::to_string(port)));
    }
    SECTION("the WebSocket upgrade is refused")
    {
        core::Listener listener(core::Endpoint{ "127.0.0.1", 0 });
        std::jthread refuser(
            [&]
            {
                if (const std::optional<core::Listener::Accepted> accepted = listener.accept(kWait))
                {
                    core::TcpSocket socket(accepted->fd);
                    const std::string answer = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
                    socket.send_all(reinterpret_cast<const uint8_t*>(answer.data()), answer.size());
                    uint8_t byte = 0;
                    while (socket.recv_all(&byte, 1, kWait) == core::TcpSocket::RecvResult::Ok)
                    {
                    }
                }
            });
        const std::string endpoint = tcp_endpoint(listener, true) + "/teleop";
        const auto client = core::RemoteSessionClient::start(endpoint, kHello, robot.stream, kWait);
        CHECK_THROWS_MATCHES(client->wait_hello_ack(kWait), std::runtime_error,
                             Message("WebSocket upgrade to 127.0.0.1:" + std::to_string(listener.port()) +
                                     "/teleop refused: HTTP/1.1 404 Not Found"));
        client->close();
    }
}

TEST_CASE("RemoteSessionClient: close ends a wait for HELLO_ACK on another thread", "[remote][session][network]")
{
    const bool websocket = GENERATE(false, true);
    // Never accepted: the connection stays in the listen backlog, so no upgrade or HELLO_ACK comes.
    core::Listener listener(core::Endpoint{ "127.0.0.1", 0 });
    Robot robot;
    const auto client = core::RemoteSessionClient::start(tcp_endpoint(listener, websocket), kHello, robot.stream, kWait);
    std::string error;
    std::jthread waiter(
        [&]
        {
            try
            {
                client->wait_hello_ack(-1ms);
            }
            catch (const std::runtime_error& e)
            {
                error = e.what();
            }
        });
    std::this_thread::sleep_for(100ms);
    const auto start = std::chrono::steady_clock::now();
    client->close();
    waiter.join();
    CHECK(std::chrono::steady_clock::now() - start < 1s);
    CHECK(error == "Remote session ended before HELLO_ACK: closed");
}

namespace
{

//! Runs `wait`, which must throw: whether it threw a PeerClosedError, and its text.
std::pair<bool, std::string> failure_of(const std::function<void()>& wait)
{
    try
    {
        wait();
    }
    catch (const core::PeerClosedError& e)
    {
        return { true, e.what() };
    }
    catch (const std::runtime_error& e)
    {
        return { false, e.what() };
    }
    FAIL("no exception");
    return {};
}

//! Accepts one robot, reads its HELLO (over ws://, after the upgrade), sends the first 4 bytes of a message and
//! ends the connection: with a reset, or with end of stream once the robot closed its end.
void send_part_of_a_message(core::Listener& listener, bool websocket, bool reset)
{
    const std::optional<core::Listener::Accepted> accepted = listener.accept(kWait);
    if (!accepted)
    {
        return;
    }
    core::TcpSocket raw(::dup(accepted->fd));
    std::unique_ptr<core::ITransport> transport = core::adopt_transport(accepted->fd, websocket);
    std::vector<uint8_t> hello;
    if (transport->receive(hello, kWait) != core::ReceiveResult::Ok)
    {
        return;
    }
    // A WebSocket frame head announcing 24 unmasked bytes, then 2 of them; or 4 bytes of a link message header.
    std::vector<uint8_t> part{ 0x82, 24, 0, 0 };
    if (!websocket)
    {
        const std::vector<uint8_t> heartbeat =
            core::encode_link_message(static_cast<uint8_t>(core::LinkMessageType::Heartbeat), {});
        part.assign(heartbeat.begin(), heartbeat.begin() + 4);
    }
    raw.send_all(part.data(), part.size());
    if (reset)
    {
        const linger abort{ 1, 0 };
        ::setsockopt(raw.fd(), SOL_SOCKET, SO_LINGER, &abort, sizeof(abort));
        transport.reset();
        return;
    }
    raw.shutdown_write();
    uint8_t byte = 0;
    while (raw.recv_all(&byte, 1, kWait) == core::TcpSocket::RecvResult::Ok)
    {
    }
}

} // namespace

TEST_CASE("RemoteSessionClient: a server that closes before sending anything is a PeerClosedError",
          "[remote][session][network]")
{
    const bool websocket = GENERATE(false, true);
    Robot robot;
    core::Listener listener(core::Endpoint{ "127.0.0.1", 0 });
    const std::string expected = websocket ? "WebSocket upgrade to 127.0.0.1:" + std::to_string(listener.port()) +
                                                 "/: connection closed before the answer" :
                                             "Remote session ended before HELLO_ACK: receive failed: connection closed";
    std::unique_ptr<core::RemoteSessionClient> client;
    SECTION("closed once accepted")
    {
        std::jthread closer(
            [&]
            {
                if (const std::optional<core::Listener::Accepted> accepted = listener.accept(kWait))
                {
                    // After the robot's first byte, so its HELLO or upgrade request went out before the close.
                    core::TcpSocket socket(accepted->fd);
                    uint8_t byte = 0;
                    socket.recv_all(&byte, 1, kWait);
                }
            });
        client = core::RemoteSessionClient::start(tcp_endpoint(listener, websocket), kHello, robot.stream, kWait);
    }
    SECTION("reset in the listen backlog")
    {
        client = core::RemoteSessionClient::start(tcp_endpoint(listener, websocket), kHello, robot.stream, kWait);
        std::this_thread::sleep_for(100ms);
        listener.close();
    }
    const auto [peer_closed, text] = failure_of([&] { client->wait_hello_ack(kWait); });
    CHECK(peer_closed);
    CHECK(text == expected);
}

TEST_CASE("RemoteSessionClient: a server that sent anything before closing is no PeerClosedError",
          "[remote][session][network]")
{
    const bool websocket = GENERATE(false, true);
    Robot robot;
    std::pair<bool, std::string> failure;
    std::string expected;
    SECTION("a refusal")
    {
        expected = "busy: another robot app is connected to this Teleop Session Server";
        core::Listener listener(core::Endpoint{ "127.0.0.1", 0 });
        std::jthread rejecter(
            [&]
            {
                if (const std::optional<core::Listener::Accepted> accepted = listener.accept(kWait))
                {
                    core::reject_remote_connection(accepted->fd, websocket, expected);
                }
            });
        const auto client =
            core::RemoteSessionClient::start(tcp_endpoint(listener, websocket), kHello, robot.stream, kWait);
        failure = failure_of([&] { client->wait_hello_ack(kWait); });
    }
    SECTION("part of a message, then end of stream")
    {
        expected = "Remote session ended before HELLO_ACK: receive failed: incomplete message";
        core::Listener listener(core::Endpoint{ "127.0.0.1", 0 });
        std::jthread server([&] { send_part_of_a_message(listener, websocket, false); });
        const auto client =
            core::RemoteSessionClient::start(tcp_endpoint(listener, websocket), kHello, robot.stream, kWait);
        failure = failure_of([&] { client->wait_hello_ack(kWait); });
    }
    SECTION("part of a message, then a reset")
    {
        expected = "Remote session ended before HELLO_ACK: receive failed: incomplete message";
        core::Listener listener(core::Endpoint{ "127.0.0.1", 0 });
        std::jthread server([&] { send_part_of_a_message(listener, websocket, true); });
        const auto client =
            core::RemoteSessionClient::start(tcp_endpoint(listener, websocket), kHello, robot.stream, kWait);
        failure = failure_of([&] { client->wait_hello_ack(kWait); });
    }
    SECTION("a STATUS before HELLO_ACK")
    {
        expected = "Remote session ended before HELLO_ACK: receive failed: connection closed";
        Server server(websocket,
                      [](core::RemoteSessionHost& host, const std::atomic<bool>&)
                      {
                          host.wait_hello(kWait);
                          host.send_status(kWaiting);
                      });
        const auto client = core::RemoteSessionClient::start(server.endpoint(), kHello, robot.stream, kWait);
        failure = failure_of([&] { client->wait_hello_ack(kWait); });
        server.stop();
    }
    SECTION("the link lost after HELLO_ACK")
    {
        expected = "Remote session lost: receive failed: connection closed";
        Server server(websocket,
                      [](core::RemoteSessionHost& host, const std::atomic<bool>& stop)
                      {
                          host.wait_hello(kWait);
                          host.send_hello_ack(kHelloAck);
                          wait_until([&] { return stop.load(); });
                      });
        robot.connect(server.endpoint());
        server.stop();
        failure = failure_of([&] { robot.client->wait_ready(kWait); });
    }
    CHECK_FALSE(failure.first);
    CHECK(failure.second == expected);
}

TEST_CASE("RemoteSessionClient: tcp:// on a WebSocket port is malformed data, no PeerClosedError",
          "[remote][session][network]")
{
    Robot robot;
    core::Listener listener(core::Endpoint{ "127.0.0.1", 0 });
    std::jthread server(
        [&]
        {
            if (const std::optional<core::Listener::Accepted> accepted = listener.accept(kWait))
            {
                // Answers the robot's binary HELLO with 400 Bad Request.
                core::adopt_transport(accepted->fd, true)->wait_ready(kWait);
            }
        });
    const auto client = core::RemoteSessionClient::start(tcp_endpoint(listener), kHello, robot.stream, kWait);
    const auto [peer_closed, text] = failure_of([&] { client->wait_hello_ack(kWait); });
    CHECK_FALSE(peer_closed);
    CHECK(text == "Remote session ended before HELLO_ACK: receive failed: malformed data");
}

#if defined(XR_USE_TIMESPEC)

namespace
{

using tensor_test_support::FakeTensorRuntime;

//! A plugin-fed tracker type: the server reads its collection through the fake tensor runtime, the robot the
//! same tracker type through a RemoteSession.
struct PluginTrackerCase
{
    std::string collection;
    uint32_t sample_size;
    std::vector<uint8_t> payload;
    std::shared_ptr<core::ITracker> server_tracker;
    core::VendorConfig vendors;
    std::unique_ptr<core::ITracker> robot_tracker;
    std::function<bool(const core::ITracker&, const core::ITrackerSession&)> has_data;
};

PluginTrackerCase pedal_case()
{
    PluginTrackerCase c;
    c.collection = "pedals";
    auto tracker = std::make_shared<core::Generic3AxisPedalTracker>(c.collection);
    c.sample_size = static_cast<uint32_t>(tracker->max_flatbuffer_size());
    core::Generic3AxisPedalOutputT native;
    native.left_pedal = 0.5f;
    const auto packed = core::pack<core::Generic3AxisPedalOutput>(native);
    c.payload.assign(packed.buffer().begin(), packed.buffer().end());
    c.server_tracker = tracker;
    c.robot_tracker = std::make_unique<core::Generic3AxisPedalTracker>(c.collection);
    c.has_data = [](const core::ITracker& t, const core::ITrackerSession& s)
    { return static_cast<bool>(static_cast<const core::Generic3AxisPedalTracker&>(t).get_data(s)); };
    return c;
}

PluginTrackerCase noitom_case()
{
    PluginTrackerCase c;
    c.collection = "noitom_mocap";
    c.sample_size = 4096;
    core::FullBodyPoseT native;
    native.joints = std::make_unique<core::BodyJoints>();
    native.all_joint_poses_tracked = true;
    const auto packed = core::pack<core::FullBodyPose>(native);
    c.payload.assign(packed.buffer().begin(), packed.buffer().end());
    c.server_tracker = std::make_shared<core::FullBodyTracker>();
    c.vendors.tracker_vendors.push_back({ c.server_tracker.get(), core::TrackerVendor{ "body.noitom", {} } });
    c.robot_tracker = std::make_unique<core::FullBodyTracker>();
    c.has_data = [](const core::ITracker& t, const core::ITrackerSession& s)
    { return static_cast<bool>(static_cast<const core::FullBodyTracker&>(t).get_body_pose(s)); };
    return c;
}

//! What the plugin does next. The server thread applies each step before its next tick, so only that thread
//! touches the fake runtime.
enum PluginStep
{
    kPublish = 1,
    kGoQuiet,
    kVanish,
    kReturn,
};

} // namespace

TEST_CASE("RemoteSessionHost: a plugin tracker reads no data once its device goes away, as in a local session",
          "[remote][session][network]")
{
    PluginTrackerCase c = GENERATE(0, 1) == 0 ? pedal_case() : noitom_case();
    CAPTURE(c.collection);

    auto stream = std::make_shared<core::TrackerSampleStreamReader>();
    auto session = core::RemoteSession::run({ stream, { { c.robot_tracker.get(), "0" } } });
    const std::vector<std::string> topics = stream->subscribed_topics();
    std::atomic<int> requested{ 0 };
    std::atomic<int> ticks{ 0 };

    Server server(false,
                  [&](core::RemoteSessionHost& host, const std::atomic<bool>& stop)
                  {
                      FakeTensorRuntime runtime;
                      // The session's trackers call into the runtime until the host is closed.
                      struct CloseHostFirst
                      {
                          core::RemoteSessionHost& host;
                          ~CloseHostFirst()
                          {
                              host.close();
                          }
                      } close_host_first{ host };

                      if (host.wait_hello(kWait) != kHello)
                      {
                          throw std::runtime_error("no HELLO");
                      }
                      host.send_hello_ack(kHelloAck);
                      host.start_session({ c.server_tracker }, { { c.server_tracker.get(), "0" } }, runtime.handles(),
                                         c.vendors, topics);
                      host.send_ready(kReady);
                      for (int applied = 0; !stop && host.alive();)
                      {
                          for (; applied < requested; ++applied)
                          {
                              const int step = applied + 1;
                              if (step == kPublish || step == kReturn)
                              {
                                  runtime.create(c.collection, c.sample_size);
                                  runtime.push(c.collection, c.payload, core::os_monotonic_now_ns());
                              }
                              else if (step == kVanish)
                              {
                                  runtime.destroy(c.collection);
                              }
                          }
                          host.tick();
                          ++ticks;
                          std::this_thread::sleep_for(5ms);
                      }
                  });

    const auto client = core::RemoteSessionClient::connect(server.endpoint(), kHello, stream, kWait);
    REQUIRE(client->wait_ready(kWait) == core::RemoteReadyState::Ready);
    // Fresh as well, so that no data cannot come from a stale stream instead.
    const auto reads = [&](bool data)
    {
        return wait_until(
            [&]
            {
                session->update();
                return session->is_fresh() && c.has_data(*c.robot_tracker, *session) == data;
            });
    };

    CHECK(reads(false));
    requested = kPublish;
    REQUIRE(reads(true));

    // A plugin that publishes nothing new keeps its last sample.
    requested = kGoQuiet;
    const int quiet_from = ticks;
    REQUIRE(wait_until([&] { return ticks >= quiet_from + 10; }));
    CHECK(reads(true));

    requested = kVanish;
    CHECK(reads(false));

    requested = kReturn;
    CHECK(reads(true));

    server.stop();
    CHECK(server.error().empty());
}

#endif
