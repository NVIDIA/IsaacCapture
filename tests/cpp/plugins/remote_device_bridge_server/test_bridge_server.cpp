// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// End-to-end tests of the real BridgeServer (fork per client) over loopback TCP. The bridge runs in a forked
// process so the test's client threads never exist in a process that forks; its plugin session is a fake that
// logs every call, so no OpenXR runtime is involved. Bridge and fake output share one log file.

#include <catch2/catch_test_macros.hpp>
#include <oxr_utils/os_time.hpp>
#include <remote_device_bridge/bridge_server.hpp>
#include <remote_plugin_session/protocol_frame.hpp>
#include <remote_plugin_session/protocol_payloads.hpp>
#include <remote_plugin_session/remote_plugin_session.hpp>
#include <remote_plugin_session/tcp_socket.hpp>
#include <remote_plugin_session/tcp_transport.hpp>
#include <remote_plugin_session/websocket_transport.hpp>
#include <sys/wait.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <poll.h>
#include <regex>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace remote_plugin_session;
using remote_device_bridge::BridgeOptions;
using remote_device_bridge::BridgeServer;
namespace protocol = remote_plugin_protocol;

namespace
{

using Ms = std::chrono::milliseconds;
constexpr Ms kWait{ 5000 };
constexpr int64_t kMs = 1'000'000;

//! Read by the fake session in the bridge process; set before BridgeProcess forks.
struct FakeBehavior
{
    int first_push_delay_ms = 0;
    XrResult hand_create_result = XR_SUCCESS; //!< xrCreatePushDeviceNV result the fake reports
};
FakeBehavior g_fake;

std::atomic<bool> g_bridge_stop{ false };

void on_sigterm(int)
{
    g_bridge_stop.store(true);
}

//! Same category name and what() text as plugin_utils' CheckXrResult.
class XrResultCategory final : public std::error_category
{
public:
    const char* name() const noexcept override
    {
        return "XrResult";
    }
    std::string message(int value) const override
    {
        return std::to_string(value);
    }
};

void log_event(const std::string& line)
{
    std::cout << "[fake pid=" << getpid() << "] " << line << std::endl;
}

const char* side(XrHandEXT hand)
{
    return hand == XR_HAND_LEFT_EXT ? "left" : "right";
}

class FakeHandChannel final : public core::IHandTrackingPushChannel
{
public:
    explicit FakeHandChannel(XrHandEXT hand) : m_hand(hand)
    {
        log_event(std::string("open hand ") + side(hand));
    }
    ~FakeHandChannel() override
    {
        log_event(std::string("close hand ") + side(m_hand));
    }
    void push(const XrHandJointLocationEXT*, int64_t sample_time_ns) override
    {
        if (m_pushes++ == 0 && g_fake.first_push_delay_ms > 0)
        {
            std::this_thread::sleep_for(Ms(g_fake.first_push_delay_ms));
        }
        log_event(std::string("push hand ") + side(m_hand) + " t=" + std::to_string(sample_time_ns));
    }

private:
    XrHandEXT m_hand;
    uint64_t m_pushes = 0;
};

class FakeSchemaChannel final : public core::ISchemaPushChannel
{
public:
    explicit FakeSchemaChannel(core::SchemaPusherConfig config) : m_config(std::move(config))
    {
        log_event("open schema " + m_config.collection_id);
    }
    ~FakeSchemaChannel() override
    {
        log_event("close schema " + m_config.collection_id);
    }
    void push_buffer(const uint8_t*, size_t size, int64_t, int64_t) override
    {
        log_event("push schema " + m_config.collection_id + " size=" + std::to_string(size));
    }

private:
    core::SchemaPusherConfig m_config;
};

class FakeWristSource final : public core::IWristTrackingSource
{
public:
    core::WristTrackingSample query(XrHandEXT, int64_t) override
    {
        return {};
    }
};

//! Created with the session, like the OpenXR one; the wrist source exists only when HELLO declared it.
class FakePullChannel final : public core::IPluginPullChannel
{
public:
    explicit FakePullChannel(bool has_wrist) : m_has_wrist(has_wrist)
    {
    }
    void update() override
    {
    }
    const core::ITrackerImpl& get_tracker_impl(const core::ITracker&) const override
    {
        throw std::runtime_error("no trackers");
    }
    core::IWristTrackingSource& wrist_tracking_source() override
    {
        if (!m_has_wrist)
        {
            throw std::logic_error("wrist tracking pull was not declared");
        }
        return m_wrist;
    }

private:
    bool m_has_wrist;
    FakeWristSource m_wrist;
};

class FakeSession final : public core::IPluginSession
{
public:
    FakeSession(const std::string& app_name, const core::PluginSessionRequirements& requirements)
        : m_pull(requirements.wrist_tracking_pull.has_value())
    {
        log_event("session open app=" + app_name);
    }
    ~FakeSession() override
    {
        log_event("session closed");
    }
    core::IPluginPullChannel& pull_channel() override
    {
        return m_pull;
    }
    std::unique_ptr<core::ISchemaPushChannel> create_schema_push_channel(const core::SchemaPusherConfig& config) override
    {
        return std::make_unique<FakeSchemaChannel>(config);
    }
    std::unique_ptr<core::IHandTrackingPushChannel> create_hand_tracking_push_channel(XrHandEXT hand) override
    {
        if (XR_FAILED(g_fake.hand_create_result))
        {
            static const XrResultCategory category;
            throw std::system_error(g_fake.hand_create_result, category,
                                    std::string("xrCreatePushDeviceNV(") + side(hand) + ") failed with XrResult");
        }
        return std::make_unique<FakeHandChannel>(hand);
    }

private:
    FakePullChannel m_pull;
};

bool wait_until(const std::function<bool()>& predicate, Ms timeout = kWait)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!predicate())
    {
        if (std::chrono::steady_clock::now() > deadline)
        {
            return false;
        }
        std::this_thread::sleep_for(Ms(10));
    }
    return true;
}

size_t count(const std::string& text, const std::string& needle)
{
    size_t n = 0;
    for (size_t pos = text.find(needle); pos != std::string::npos; pos = text.find(needle, pos + needle.size()))
    {
        ++n;
    }
    return n;
}

//! Sum of every "<key>=<n>" on the bridge's status lines.
uint64_t status_sum(const std::string& log, const std::string& key)
{
    const std::regex pattern(" 5s: (?:.*[ |])?" + key + "=([0-9]+)");
    uint64_t sum = 0;
    std::istringstream lines(log);
    for (std::string line; std::getline(lines, line);)
    {
        std::smatch match;
        if (std::regex_search(line, match, pattern))
        {
            sum += std::stoull(match[1]);
        }
    }
    return sum;
}

//! Sample times the fake pushed for one hand, in push order.
std::vector<int64_t> pushed_times(const std::string& log, const char* hand_side)
{
    const std::regex pattern(std::string("push hand ") + hand_side + " t=(-?[0-9]+)");
    std::vector<int64_t> times;
    for (std::sregex_iterator it(log.begin(), log.end(), pattern), end; it != end; ++it)
    {
        times.push_back(std::stoll((*it)[1]));
    }
    return times;
}

//! The bridge in its own process, logging to a temp file. The destructor stops it like Ctrl-C would.
class BridgeProcess
{
public:
    explicit BridgeProcess(BridgeOptions options = {})
    {
        options.listen = Endpoint{ "127.0.0.1", 0 };
        std::string path = (std::filesystem::temp_directory_path() / "remote_device_bridge_test_XXXXXX").string();
        const int log_fd = mkstemp(path.data());
        REQUIRE(log_fd >= 0);
        close(log_fd);
        m_log_path = path;

        int ready[2];
        REQUIRE(pipe(ready) == 0);
        std::cout.flush();
        std::cerr.flush();
        std::fflush(nullptr);
        m_pid = fork();
        REQUIRE(m_pid >= 0);
        if (m_pid == 0)
        {
            close(ready[0]);
            run_bridge(options, ready[1]);
        }
        close(ready[1]);
        pollfd fd{ ready[0], POLLIN, 0 };
        std::array<uint16_t, 2> ports{};
        if (poll(&fd, 1, static_cast<int>(kWait.count())) == 1 &&
            read(ready[0], ports.data(), sizeof(ports)) == sizeof(ports))
        {
            m_port = ports[0];
            m_ws_port = ports[1];
        }
        close(ready[0]);
        REQUIRE(m_port != 0);
    }

    ~BridgeProcess()
    {
        kill(m_pid, SIGTERM);
        int status = 0;
        const bool exited = wait_until([&] { return waitpid(m_pid, &status, WNOHANG) == m_pid; }, Ms(10000));
        if (!exited)
        {
            kill(m_pid, SIGKILL);
            waitpid(m_pid, &status, 0);
        }
        if (std::uncaught_exceptions() > 0 || !exited)
        {
            std::cerr << "---- bridge log ----\n" << log() << "--------------------" << std::endl;
        }
        std::filesystem::remove(m_log_path);
    }

    BridgeProcess(const BridgeProcess&) = delete;
    BridgeProcess& operator=(const BridgeProcess&) = delete;

    uint16_t port() const
    {
        return m_port;
    }

    //! 0 unless the options set ws_listen.
    uint16_t ws_port() const
    {
        return m_ws_port;
    }

    std::string log() const
    {
        std::ifstream file(m_log_path);
        std::stringstream text;
        text << file.rdbuf();
        return text.str();
    }

    size_t count(const std::string& needle) const
    {
        return ::count(log(), needle);
    }

    bool wait_log(const std::string& needle, size_t times = 1, Ms timeout = kWait) const
    {
        return wait_until([&] { return count(needle) >= times; }, timeout);
    }

    //! pid of the bridge child serving the client that sent HELLO with `app_name`.
    pid_t client_pid(const std::string& app_name) const
    {
        const std::regex pattern("\\[fake pid=([0-9]+)\\] session open app=" + app_name + "\n");
        const std::string text = log();
        std::smatch match;
        return std::regex_search(text, match, pattern) ? static_cast<pid_t>(std::stoi(match[1])) : 0;
    }

private:
    [[noreturn]] void run_bridge(const BridgeOptions& options, int ready_fd)
    {
        const int log_fd = open(m_log_path.c_str(), O_WRONLY | O_APPEND);
        dup2(log_fd, STDOUT_FILENO);
        dup2(log_fd, STDERR_FILENO);
        close(log_fd);
        std::signal(SIGTERM, on_sigterm);
        std::array<uint16_t, 2> ports{};
        try
        {
            BridgeServer server(options,
                                [](const std::string& app_name) -> core::PluginSessionFactory
                                {
                                    return [app_name](const core::PluginSessionRequirements& requirements,
                                                      std::vector<std::shared_ptr<core::ITracker>>)
                                    { return std::make_unique<FakeSession>(app_name, requirements); };
                                });
            ports = { server.port(), server.ws_port().value_or(0) };
            (void)!write(ready_fd, ports.data(), sizeof(ports));
            close(ready_fd);
            server.run(g_bridge_stop);
        }
        catch (const std::exception& e)
        {
            std::cerr << "bridge failed: " << e.what() << std::endl;
            (void)!write(ready_fd, ports.data(), sizeof(ports));
        }
        std::cout.flush();
        std::cerr.flush();
        _exit(0);
    }

    std::string m_log_path;
    pid_t m_pid = -1;
    uint16_t m_port = 0;
    uint16_t m_ws_port = 0;
};

//! Speaks the wire protocol by hand: optionally no HELLO, any capture time, no PONGs. Heartbeats keep the bridge's
//! 3 s peer timeout away.
class RawClient
{
public:
    explicit RawClient(uint16_t port, bool send_hello = true)
        : m_transport(std::make_unique<TcpTransport>(TcpSocket::connect(Endpoint{ "127.0.0.1", port }, kWait)))
    {
        if (!send_hello)
        {
            return;
        }
        protocol::Hello hello;
        hello.app_name = "raw";
        hello.requirements = { .schema_push = true,
                               .hand_tracking_push = true,
                               .wrist_tracking_pull = core::WristTrackingSourceConfig{} };
        send(protocol::FrameType::Hello, protocol::kSessionChannelId, protocol::encode_hello(hello));
        m_heartbeat = std::thread(
            [this]
            {
                while (!m_stop.load())
                {
                    send(protocol::FrameType::Heartbeat, protocol::kSessionChannelId, {});
                    std::this_thread::sleep_for(Ms(300));
                }
            });
    }

    ~RawClient()
    {
        disconnect();
    }

    bool send(protocol::FrameType type, uint16_t channel_id, const std::vector<uint8_t>& payload)
    {
        return send_frame(protocol::encode_frame(type, channel_id, payload));
    }

    bool send_hand_sample(uint16_t channel_id, int64_t capture_ns)
    {
        const std::array<XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT> joints{};
        return send_frame(protocol::encode_hand_sample_frame(channel_id, joints.data(), capture_ns));
    }

    //! Next frame of `type`; frames of other types are counted and skipped.
    std::optional<protocol::Frame> wait_for(protocol::FrameType type, Ms timeout = kWait)
    {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline)
        {
            std::vector<uint8_t> bytes;
            const auto left = std::chrono::duration_cast<Ms>(deadline - std::chrono::steady_clock::now());
            const ReceiveResult result = m_transport->receive(bytes, std::max(left, Ms(1)));
            if (result == ReceiveResult::Timeout)
            {
                continue;
            }
            if (result != ReceiveResult::Ok)
            {
                return std::nullopt;
            }
            protocol::Frame frame = protocol::split_frame(std::move(bytes));
            m_receive_ns[frame.header.type].push_back(core::os_monotonic_now_ns());
            if (frame.header.type == type)
            {
                return frame;
            }
        }
        return std::nullopt;
    }

    //! Receive times of every frame of `type` seen so far.
    const std::vector<int64_t>& received(protocol::FrameType type)
    {
        return m_receive_ns[type];
    }

    void disconnect()
    {
        m_stop.store(true);
        if (m_heartbeat.joinable())
        {
            m_heartbeat.join();
        }
        m_transport->shutdown();
    }

private:
    bool send_frame(std::vector<uint8_t> frame)
    {
        std::lock_guard<std::mutex> lock(m_send_mutex);
        protocol::stamp_header(frame, m_seq++, core::os_monotonic_now_ns());
        return m_transport->send(frame.data(), frame.size());
    }

    std::unique_ptr<ITransport> m_transport;
    std::mutex m_send_mutex;
    uint32_t m_seq = 0;
    std::map<protocol::FrameType, std::vector<int64_t>> m_receive_ns;
    std::atomic<bool> m_stop{ false };
    std::thread m_heartbeat;
};

bool accepted(RawClient& client)
{
    const auto ack = client.wait_for(protocol::FrameType::HelloAck);
    return ack.has_value() && protocol::decode_hello_ack(ack->payload).accepted;
}

core::SchemaPusherConfig schema_config(const std::string& collection_id, size_t max_flatbuffer_size)
{
    return { .collection_id = collection_id,
             .max_flatbuffer_size = max_flatbuffer_size,
             .tensor_identifier = "tensor",
             .localized_name = collection_id,
             .app_name = "test" };
}

std::unique_ptr<RemotePluginSession> connect(uint16_t port, const std::string& app_name)
{
    return std::make_unique<RemotePluginSession>(
        std::make_unique<TcpTransport>(TcpSocket::connect(Endpoint{ "127.0.0.1", port }, kWait)), app_name,
        core::PluginSessionRequirements{
            .schema_push = true, .hand_tracking_push = true, .wrist_tracking_pull = core::WristTrackingSourceConfig{} });
}

const std::array<XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT> kJoints{};

//! Pushes to `channel` every 10 ms on a thread until destroyed.
class Streamer
{
public:
    explicit Streamer(core::IHandTrackingPushChannel& channel)
        : m_thread(
              [this, &channel]
              {
                  while (!m_stop.load())
                  {
                      channel.push(kJoints.data(), core::os_monotonic_now_ns());
                      std::this_thread::sleep_for(Ms(10));
                  }
              })
    {
    }
    ~Streamer()
    {
        m_stop.store(true);
        m_thread.join();
    }

private:
    std::atomic<bool> m_stop{ false };
    std::thread m_thread;
};

} // namespace

TEST_CASE("Bridge opens plugin-session channels only on OPEN and releases them on CLOSE and disconnect",
          "[remote_device_bridge][unit]")
{
    BridgeProcess bridge;
    RawClient client(bridge.port());
    REQUIRE(accepted(client));
    REQUIRE(bridge.wait_log("session open app=raw"));

    std::this_thread::sleep_for(Ms(300));
    REQUIRE(bridge.count("] open ") == 0);

    REQUIRE(client.send(protocol::FrameType::OpenHand, 1, protocol::encode_open_hand(XR_HAND_LEFT_EXT)));
    REQUIRE(bridge.wait_log("open hand left"));
    REQUIRE(client.send(protocol::FrameType::Close, 1, {}));
    REQUIRE(bridge.wait_log("close hand left"));

    REQUIRE(client.send(protocol::FrameType::OpenHand, 2, protocol::encode_open_hand(XR_HAND_RIGHT_EXT)));
    REQUIRE(client.send(protocol::FrameType::OpenSchema, 3, protocol::encode_open_schema(schema_config("sensors", 256))));
    REQUIRE(client.send(protocol::FrameType::OpenPull, 4, {}));
    REQUIRE(bridge.wait_log("channel 4: pull channel opened with wrist source"));

    // Disconnect without CLOSE: the bridge releases everything, push channels before the session (which owns the
    // pull channel).
    client.disconnect();
    REQUIRE(bridge.wait_log("session closed"));
    const std::string log = bridge.log();
    REQUIRE(count(log, "close hand right") == 1);
    REQUIRE(count(log, "close schema sensors") == 1);
    REQUIRE(log.find("close hand right") < log.find("session closed"));
    REQUIRE(log.find("close schema sensors") < log.find("session closed"));
}

TEST_CASE("Bridge maps the sender's capture time by the clock offset only", "[remote_device_bridge][unit]")
{
    BridgeProcess bridge;
    RawClient client(bridge.port());
    REQUIRE(accepted(client));
    REQUIRE(client.send(protocol::FrameType::OpenHand, 1, protocol::encode_open_hand(XR_HAND_LEFT_EXT)));
    REQUIRE(bridge.wait_log("open hand left"));

    // Same host, same CLOCK_MONOTONIC: the offset is ~0, so the pushed time is the capture time give or take
    // the loopback delay.
    const int64_t capture_ns = core::os_monotonic_now_ns() - 100 * kMs;
    REQUIRE(client.send_hand_sample(1, capture_ns));
    REQUIRE(bridge.wait_log("push hand left"));
    const std::vector<int64_t> pushed = pushed_times(bridge.log(), "left");
    REQUIRE(pushed.size() == 1);
    INFO("pushed - capture = " << pushed[0] - capture_ns << " ns");
    REQUIRE(std::abs(pushed[0] - capture_ns) < 5 * kMs);
}

TEST_CASE("Killing one client process leaves the bridge and the other clients running", "[remote_device_bridge][unit]")
{
    BridgeProcess bridge;
    auto a = connect(bridge.port(), "client_a");
    auto b = connect(bridge.port(), "client_b");
    auto a_left = a->create_hand_tracking_push_channel(XR_HAND_LEFT_EXT);
    auto b_right = b->create_hand_tracking_push_channel(XR_HAND_RIGHT_EXT);
    REQUIRE(bridge.wait_log("open hand left"));
    REQUIRE(bridge.wait_log("open hand right"));
    const pid_t a_pid = bridge.client_pid("client_a");
    REQUIRE(a_pid > 0);
    REQUIRE(bridge.client_pid("client_b") != a_pid);

    REQUIRE(kill(a_pid, SIGKILL) == 0);
    REQUIRE(bridge.wait_log("client pid " + std::to_string(a_pid) + " exited (signal 9)"));
    REQUIRE(wait_until([&] { return !a->connected(); }));

    const size_t before = bridge.count("push hand right");
    {
        Streamer streaming(*b_right);
        REQUIRE(bridge.wait_log("push hand right", before + 10));
    }
    REQUIRE(b->connected());

    // The parent is still accepting, and the dead child's hand is free again.
    auto c = connect(bridge.port(), "client_c");
    auto c_left = c->create_hand_tracking_push_channel(XR_HAND_LEFT_EXT);
    REQUIRE(bridge.wait_log("open hand left", 2));
    REQUIRE(c->stats().refused_channels.empty());
}

TEST_CASE("A second client opening an owned hand is refused and the owner keeps streaming", "[remote_device_bridge][unit]")
{
    BridgeProcess bridge;
    auto a = connect(bridge.port(), "client_a");
    auto b = connect(bridge.port(), "client_b");
    auto a_left = a->create_hand_tracking_push_channel(XR_HAND_LEFT_EXT);
    REQUIRE(bridge.wait_log("open hand left"));
    Streamer streaming(*a_left);
    REQUIRE(bridge.wait_log("push hand left", 5));

    auto b_left = b->create_hand_tracking_push_channel(XR_HAND_LEFT_EXT);
    REQUIRE(wait_until([&] { return b->stats().refused_channels.size() == 1; }));
    const std::string reason = b->stats().refused_channels.begin()->second;
    REQUIRE(reason == "hand left is already owned by bridge client c1; channel refused");
    REQUIRE(bridge.wait_log("refused: hand left is already owned by bridge client c1; channel refused"));
    REQUIRE(b->connected());

    // The refused client's session and other channels work; the owner never noticed.
    auto b_right = b->create_hand_tracking_push_channel(XR_HAND_RIGHT_EXT);
    b_left->push(kJoints.data(), core::os_monotonic_now_ns());
    b_right->push(kJoints.data(), core::os_monotonic_now_ns());
    REQUIRE(bridge.wait_log("push hand right"));
    const size_t before = bridge.count("push hand left");
    REQUIRE(bridge.wait_log("push hand left", before + 10));
    REQUIRE(a->connected());
    REQUIRE(a->stats().refused_channels.empty());
    REQUIRE(bridge.count("open hand left") == 1);
    REQUIRE(bridge.count("closing connection") == 0);
}

TEST_CASE("A hand the runtime refuses is a channel refusal, not a session error", "[remote_device_bridge][unit]")
{
    g_fake.hand_create_result = XR_ERROR_RUNTIME_FAILURE;
    BridgeProcess bridge;
    g_fake.hand_create_result = XR_SUCCESS;
    auto client = connect(bridge.port(), "client");
    auto left = client->create_hand_tracking_push_channel(XR_HAND_LEFT_EXT);
    REQUIRE(wait_until([&] { return client->stats().refused_channels.size() == 1; }));
    REQUIRE(client->stats().refused_channels.begin()->second ==
            "hand left is already owned by another OpenXR session (runtime refused the push device); channel refused");
    auto schema = client->create_schema_push_channel(schema_config("after", 64));
    REQUIRE(bridge.wait_log("open schema after"));
    REQUIRE(client->connected());
}

TEST_CASE("Any other runtime failure opening a hand ends the session", "[remote_device_bridge][unit]")
{
    XrResult result = XR_SUCCESS;
    SECTION("XR_ERROR_VALIDATION_FAILURE")
    {
        result = XR_ERROR_VALIDATION_FAILURE;
    }
    SECTION("XR_ERROR_SESSION_LOST")
    {
        result = XR_ERROR_SESSION_LOST;
    }
    g_fake.hand_create_result = result;
    BridgeProcess bridge;
    g_fake.hand_create_result = XR_SUCCESS;
    RawClient client(bridge.port());
    REQUIRE(accepted(client));
    REQUIRE(client.send(protocol::FrameType::OpenHand, 1, protocol::encode_open_hand(XR_HAND_LEFT_EXT)));

    const std::string message = "xrCreatePushDeviceNV(left) failed with XrResult: " + std::to_string(result);
    const auto error = client.wait_for(protocol::FrameType::Error);
    REQUIRE(error.has_value());
    REQUIRE(error->header.channel_id == protocol::kSessionChannelId);
    REQUIRE(protocol::decode_error(error->payload) == message);
    REQUIRE(bridge.wait_log("[bridge c1] error: " + message + "; closing connection"));
    REQUIRE(bridge.wait_log("session closed"));
    REQUIRE(bridge.count("refused") == 0);
}

TEST_CASE("A schema channel above 32768 bytes is refused at open and the session keeps working",
          "[remote_device_bridge][unit]")
{
    BridgeOptions options;
    options.max_schema_bytes = 32768;
    BridgeProcess bridge(options);
    auto client = connect(bridge.port(), "client");
    auto too_big = client->create_schema_push_channel(schema_config("tactile", 32769));
    auto at_cap = client->create_schema_push_channel(schema_config("at_cap", 32768));
    auto left = client->create_hand_tracking_push_channel(XR_HAND_LEFT_EXT);

    REQUIRE(wait_until([&] { return client->stats().refused_channels.size() == 1; }));
    const std::string reason = client->stats().refused_channels.begin()->second;
    REQUIRE(reason ==
            "schema 'tactile' sample size 32769 bytes exceeds bridge cap 32768 bytes (runtime limit); channel refused");
    REQUIRE(bridge.wait_log("refused: " + reason));
    REQUIRE(bridge.wait_log("open schema at_cap"));
    REQUIRE(bridge.wait_log("open hand left"));
    REQUIRE(bridge.count("open schema tactile") == 0);

    const std::vector<uint8_t> bytes(100, 0x11);
    too_big->push_buffer(bytes.data(), bytes.size(), core::os_monotonic_now_ns(), 0);
    at_cap->push_buffer(bytes.data(), bytes.size(), core::os_monotonic_now_ns(), 0);
    left->push(kJoints.data(), core::os_monotonic_now_ns());
    REQUIRE(bridge.wait_log("push schema at_cap size=100"));
    REQUIRE(bridge.wait_log("push hand left"));

    too_big.reset(); // CLOSE for the refused id is accepted quietly
    at_cap->push_buffer(bytes.data(), bytes.size(), core::os_monotonic_now_ns(), 0);
    REQUIRE(bridge.wait_log("push schema at_cap size=100", 2));
    REQUIRE(client->connected());
    REQUIRE(bridge.count("closing connection") == 0);
    REQUIRE(bridge.count("CLOSE for unknown channel") == 0);
}

TEST_CASE("The 9th client gets HELLO_ACK 'bridge full' within 500 ms", "[remote_device_bridge][unit]")
{
    BridgeOptions options;
    options.max_clients = 8;
    BridgeProcess bridge(options);
    std::vector<std::unique_ptr<RemotePluginSession>> clients;
    for (int i = 0; i < 8; ++i)
    {
        clients.push_back(connect(bridge.port(), "client" + std::to_string(i)));
    }
    REQUIRE(bridge.wait_log("session open app=", 8));

    SECTION("a client that sends HELLO")
    {
        const auto start = std::chrono::steady_clock::now();
        try
        {
            connect(bridge.port(), "ninth");
            FAIL("expected the bridge to refuse the 9th client");
        }
        catch (const std::runtime_error& e)
        {
            const auto took = std::chrono::steady_clock::now() - start;
            REQUIRE(took < Ms(500));
            REQUIRE(std::string(e.what()) == "RemotePluginSession: bridge rejected the session: bridge full");
        }
    }
    SECTION("a client that never sends HELLO")
    {
        const auto start = std::chrono::steady_clock::now();
        RawClient silent(bridge.port(), false);
        const auto ack = silent.wait_for(protocol::FrameType::HelloAck, Ms(1000));
        REQUIRE(ack.has_value());
        const auto took = std::chrono::steady_clock::now() - start;
        REQUIRE(took < Ms(500));
        const protocol::HelloAck decoded = protocol::decode_hello_ack(ack->payload);
        REQUIRE_FALSE(decoded.accepted);
        REQUIRE(decoded.reason == "bridge full");
    }
    REQUIRE(bridge.wait_log("refused: bridge full (8)"));
    for (const auto& client : clients)
    {
        REQUIRE(client->connected());
    }
}

TEST_CASE("A refused client whose HELLO is 400 ms late still gets 'bridge full'", "[remote_device_bridge][unit]")
{
    BridgeOptions options;
    options.max_clients = 1;
    options.ws_listen = Endpoint{ "127.0.0.1", 0 };
    BridgeProcess bridge(options);
    auto first = connect(bridge.port(), "first");
    REQUIRE(bridge.wait_log("session open app=first"));

    protocol::Hello hello;
    hello.app_name = "late";
    std::vector<uint8_t> hello_frame =
        protocol::encode_frame(protocol::FrameType::Hello, protocol::kSessionChannelId, protocol::encode_hello(hello));
    protocol::stamp_header(hello_frame, 0, core::os_monotonic_now_ns());

    std::unique_ptr<ITransport> late;
    std::vector<uint8_t> bytes;
    SECTION("tcp: the ACK is there before the HELLO is sent")
    {
        TcpSocket socket = TcpSocket::connect(Endpoint{ "127.0.0.1", bridge.port() }, kWait);
        std::this_thread::sleep_for(Ms(400));
        late = std::make_unique<TcpTransport>(std::move(socket));
        REQUIRE(late->receive(bytes, Ms(0)) == ReceiveResult::Ok);
    }
    SECTION("ws: the upgrade itself arrives 400 ms late")
    {
        REQUIRE(bridge.ws_port() != 0);
        TcpSocket socket = TcpSocket::connect(Endpoint{ "127.0.0.1", bridge.ws_port() }, kWait);
        std::this_thread::sleep_for(Ms(400));
        late = std::make_unique<WebSocketTransport>(
            std::move(socket), "127.0.0.1:" + std::to_string(bridge.ws_port()), "/", kWait);
    }
    REQUIRE(late->send(hello_frame.data(), hello_frame.size()));
    if (bytes.empty())
    {
        REQUIRE(late->receive(bytes, kWait) == ReceiveResult::Ok);
    }
    const protocol::Frame frame = protocol::split_frame(std::move(bytes));
    REQUIRE(frame.header.type == protocol::FrameType::HelloAck);
    const protocol::HelloAck ack = protocol::decode_hello_ack(frame.payload);
    REQUIRE_FALSE(ack.accepted);
    REQUIRE(ack.reason == "bridge full");
    // The refusal child read the late HELLO and closed; nothing else follows the ACK.
    REQUIRE(late->receive(bytes, Ms(2000)) == ReceiveResult::Closed);
    REQUIRE(bridge.wait_log("refused: bridge full (1)"));
    REQUIRE(first->connected());
}

TEST_CASE("Bridge pushes the newest sample per pass, drops stale ones and applies every control frame",
          "[remote_device_bridge][unit]")
{
    BridgeOptions options;
    options.max_sample_age = Ms(250);
    options.stats_period = Ms(500);
    g_fake.first_push_delay_ms = 200; // holds the bridge's run thread so the burst below queues up
    BridgeProcess bridge(options);
    g_fake.first_push_delay_ms = 0;

    RawClient client(bridge.port());
    REQUIRE(accepted(client));
    REQUIRE(client.send(protocol::FrameType::OpenHand, 1, protocol::encode_open_hand(XR_HAND_LEFT_EXT)));
    REQUIRE(bridge.wait_log("open hand left"));

    REQUIRE(client.send_hand_sample(1, core::os_monotonic_now_ns()));
    std::this_thread::sleep_for(Ms(50));
    // 10 samples 10 ms apart with two control frames in the middle. The newest is applied ~150 ms after its
    // capture, inside the 250 ms staleness limit.
    const int64_t newest_ns = core::os_monotonic_now_ns();
    for (int i = 0; i < 10; ++i)
    {
        REQUIRE(client.send_hand_sample(1, newest_ns - (9 - i) * 10 * kMs));
        if (i == 4)
        {
            REQUIRE(client.send(
                protocol::FrameType::OpenSchema, 2, protocol::encode_open_schema(schema_config("control", 64))));
            REQUIRE(client.send(protocol::FrameType::OpenPull, 3, {}));
        }
    }

    REQUIRE(bridge.wait_log("push hand left", 2));
    REQUIRE(bridge.wait_log("open schema control"));
    REQUIRE(bridge.wait_log("pull channel opened"));
    REQUIRE(wait_until([&] { return status_sum(bridge.log(), "coalesced") == 9; }));
    const std::vector<int64_t> pushed = pushed_times(bridge.log(), "left");
    REQUIRE(pushed.size() == 2);
    INFO("pushed - newest = " << pushed[1] - newest_ns << " ns");
    REQUIRE(std::abs(pushed[1] - newest_ns) < 5 * kMs); // the newest, not one 10 ms older

    REQUIRE(client.send_hand_sample(1, core::os_monotonic_now_ns() - 1000 * kMs));
    REQUIRE(wait_until([&] { return status_sum(bridge.log(), "stale") == 1; }));
    REQUIRE(pushed_times(bridge.log(), "left").size() == 2);
    REQUIRE(status_sum(bridge.log(), "rx hand") == 12);
    REQUIRE(status_sum(bridge.log(), "applied") == 2);
}

TEST_CASE("With --clock-source=one-way the bridge never PINGs and streams on the one-way offset",
          "[remote_device_bridge][unit]")
{
    BridgeOptions options;
    options.clock_source = remote_device_bridge::ClockSource::OneWay;
    BridgeProcess bridge(options);
    RawClient client(bridge.port());
    REQUIRE(accepted(client));
    REQUIRE(client.send(protocol::FrameType::OpenHand, 1, protocol::encode_open_hand(XR_HAND_RIGHT_EXT)));
    REQUIRE(client.send_hand_sample(1, core::os_monotonic_now_ns()));
    REQUIRE(bridge.wait_log("push hand right"));

    // Past the PING burst and the 2 s PONG timeout.
    REQUIRE_FALSE(client.wait_for(protocol::FrameType::Ping, Ms(2500)).has_value());
    REQUIRE(client.received(protocol::FrameType::Ping).empty());
    REQUIRE(bridge.count("clock: no PONG") == 0);
}

TEST_CASE("A client that never answers PING gets 10 PINGs at 50 ms, one at 1 Hz, then one-way",
          "[remote_device_bridge][unit]")
{
    BridgeProcess bridge;
    RawClient client(bridge.port());
    REQUIRE(accepted(client));
    REQUIRE_FALSE(client.wait_for(protocol::FrameType::Error, Ms(3000)).has_value());

    const std::vector<int64_t>& pings = client.received(protocol::FrameType::Ping);
    REQUIRE(pings.size() == 11);
    for (size_t i = 1; i < 10; ++i)
    {
        INFO("gap " << i << ": " << (pings[i] - pings[i - 1]) / kMs << " ms");
        REQUIRE(pings[i] - pings[i - 1] > 30 * kMs);
        REQUIRE(pings[i] - pings[i - 1] < 90 * kMs);
    }
    REQUIRE(pings[10] - pings[9] > 900 * kMs);
    REQUIRE(pings[10] - pings[9] < 1100 * kMs);
    REQUIRE(bridge.count("clock: no PONG, falling back to one-way") == 1);
}

TEST_CASE("The 5 s status line carries age, RTT, offset and clock", "[remote_device_bridge][unit]")
{
    BridgeProcess bridge;
    auto client = connect(bridge.port(), "status");
    auto left = client->create_hand_tracking_push_channel(XR_HAND_LEFT_EXT);
    {
        Streamer streaming(*left);
        REQUIRE(bridge.wait_log(" 5s: ", 1, Ms(8000)));
    }
    const std::string log = bridge.log();
    const std::regex pattern(
        "\\[bridge c1\\] 5s: rx hand=[0-9]+ schema=0 \\| applied=[0-9]+ coalesced=[0-9]+ "
        "stale=[0-9]+ \\| tx wrist=[1-9][0-9]* \\| offset_ns=-?[0-9]+ \\| age_ms p50=-?[0-9.]+ "
        "p99=-?[0-9.]+ max=-?[0-9.]+ \\| rtt_ms min=[0-9.]+ last=[0-9.]+ \\| "
        "offset_ms=-?[0-9.]+ clock=two-way\n");
    std::smatch match;
    REQUIRE(std::regex_search(log, match, pattern));
}
