// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Robot state over a remote session: a RemoteSessionClient publishes records of declared series, and a
// RemoteSessionHost hands them to an attached RobotStateStore. Both ends share a socket pair; no tracker runs.

#include <catch2/catch_test_macros.hpp>
#include <deviceio_session/remote_session_client.hpp>
#include <deviceio_session/remote_session_host.hpp>
#include <mcap/sample_stream.hpp>
#include <oxr_utils/os_time.hpp>
#include <remote_link/message_header.hpp>
#include <remote_link/messages.hpp>
#include <remote_link/tcp_transport.hpp>
#include <remote_link/transport.hpp>
#include <robot_state_publisher/robot_state_publisher.hpp>
#include <robot_state_store/robot_state_store.hpp>
#include <sys/socket.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

using namespace std::chrono_literals;

constexpr std::chrono::milliseconds kWait{ 5000 };

const std::string kHello = R"({"protocol": 1, "robot_state": {"version": 1}})";
const std::string kHelloAck = R"({"session_id": "test", "server_version": "0", "robot_state": {"version": 1}})";
const std::string kStatus = R"({"stage": "waiting_for_headset"})";

constexpr uint8_t kSetup = static_cast<uint8_t>(core::LinkMessageType::RobotStateSetup);
constexpr uint8_t kSamples = static_cast<uint8_t>(core::LinkMessageType::RobotStateSamples);
constexpr uint8_t kRecords = static_cast<uint8_t>(core::LinkMessageType::RobotStateRecords);

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

std::vector<uint8_t> bytes(const std::string& text)
{
    return std::vector<uint8_t>(text.begin(), text.end());
}

//! A series of a shape the robot defines; `in_order` makes it one whose every record is delivered.
core::RobotSeriesDeclaration series(const std::string& name, bool in_order = false)
{
    core::RobotSeriesDeclaration declaration;
    declaration.name = name;
    declaration.shape = "test";
    declaration.schema_name = "test." + name;
    declaration.schema_encoding = "jsonschema";
    declaration.schema_data = bytes("{}");
    declaration.message_encoding = "json";
    declaration.stale_after_ns = 500'000'000;
    declaration.in_order = in_order;
    declaration.rate_hz = 20.0;
    return declaration;
}

//! The newest record of `name` as text; nullopt without one.
std::optional<std::string> latest_text(const core::RobotStateStore& store, const std::string& name)
{
    const core::RobotRecord record = store.latest(name, core::os_monotonic_now_ns());
    if (!record.data)
    {
        return std::nullopt;
    }
    return std::string(record.data->begin(), record.data->end());
}

/*!
 * Wraps a transport and records the type of every link message it sends. While held, send() blocks and
 * wait_writable() reports false, as on a congested network, so latest-wins link messages wait in the Link.
 */
class RecordingTransport final : public core::ITransport
{
public:
    explicit RecordingTransport(std::unique_ptr<core::ITransport> inner) : m_inner(std::move(inner))
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
            m_sent.push_back(core::decode_header(message).type);
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

    //! The robot-state link messages sent, by type, in order.
    std::vector<uint8_t> robot_state_sent() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<uint8_t> sent;
        std::copy_if(m_sent.begin(), m_sent.end(), std::back_inserter(sent),
                     [](uint8_t type) { return type == kSetup || type == kSamples || type == kRecords; });
        return sent;
    }

private:
    std::unique_ptr<core::ITransport> m_inner;
    mutable std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_held = false;
    bool m_shut = false;
    std::vector<uint8_t> m_sent;
};

//! A connected RemoteSessionClient and RemoteSessionHost. With `store` the host attaches it before HELLO_ACK,
//! as the Teleop Session Server does for a robot app that offers robot state.
struct Session
{
    explicit Session(std::shared_ptr<core::RobotStateStore> store = nullptr)
    {
        int fds[2];
        REQUIRE(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, fds) == 0);
        host = core::RemoteSessionHost::adopt(std::make_unique<core::TcpTransport>(core::TcpSocket(fds[1])));
        auto recording =
            std::make_unique<RecordingTransport>(std::make_unique<core::TcpTransport>(core::TcpSocket(fds[0])));
        link = recording.get();
        std::jthread handshake(
            [&]
            {
                try
                {
                    if (host->wait_hello(kWait))
                    {
                        if (store)
                        {
                            host->attach_robot_state(store);
                        }
                        host->send_hello_ack(kHelloAck);
                    }
                }
                catch (const std::exception&)
                {
                }
            });
        client = core::RemoteSessionClient::connect(std::move(recording), kHello, stream, kWait);
    }

    //! True once a STATUS the host sends reached the client: the session is still up.
    bool still_up()
    {
        host->send_status(kStatus);
        return wait_until([&] { return client->latest_status() == kStatus; }) && host->alive() && client->alive();
    }

    std::shared_ptr<core::TrackerSampleStreamReader> stream = std::make_shared<core::TrackerSampleStreamReader>();
    std::unique_ptr<core::RemoteSessionHost> host;
    RecordingTransport* link = nullptr; //!< the client's transport, owned by its Link
    std::unique_ptr<core::RemoteSessionClient> client;
};

void send_raw(core::ITransport& transport, core::LinkMessageType type, const std::vector<uint8_t>& payload)
{
    std::vector<uint8_t> message = core::encode_link_message(static_cast<uint8_t>(type), payload);
    core::stamp_header(message.data(), 0, core::os_monotonic_now_ns());
    REQUIRE(transport.send(message.data(), message.size()));
}

} // namespace

TEST_CASE("RemoteSessionClient: robot state reaches the store the host attached", "[remote][session][robot_state]")
{
    const auto store = std::make_shared<core::RobotStateStore>();
    Session session(store);
    CHECK(store->connected());
    CHECK(store->session_count() == 1);

    session.client->set_robot_state({ series("robot_description", true), series("arm"), series("status") });
    CHECK_FALSE(session.client->robot_state_active());
    session.client->start_robot_state();
    CHECK(session.client->robot_state_active());
    REQUIRE(wait_until([&] { return store->series().size() == 3; }));
    const std::vector<core::RobotSeriesInfo> declared = store->series();
    CHECK(declared[0].name == "robot_description");
    CHECK(declared[0].in_order);
    CHECK(declared[1].name == "arm");
    CHECK_FALSE(declared[1].in_order);
    CHECK(declared[1].shape == "test");
    CHECK(declared[1].schema_name == "test.arm");
    CHECK(declared[1].schema_encoding == "jsonschema");
    CHECK(declared[1].schema_data == bytes("{}"));
    CHECK(declared[1].message_encoding == "json");
    CHECK(declared[1].stale_after_ns == 500'000'000);
    CHECK(declared[1].rate_hz == 20.0);
    CHECK(declared[2].name == "status");

    const int64_t sample_time_ns = core::os_monotonic_now_ns();
    CHECK(session.client->publish_robot_record("robot_description", bytes("what the robot is"), sample_time_ns));
    CHECK(session.client->publish_robot_record("arm", bytes("joints 1"), sample_time_ns));
    REQUIRE(wait_until([&] { return latest_text(*store, "arm") == "joints 1"; }));
    CHECK(latest_text(*store, "robot_description") == "what the robot is");
    CHECK_FALSE(latest_text(*store, "status").has_value());

    const core::RobotRecord arm = store->latest("arm", core::os_monotonic_now_ns());
    CHECK(arm.sequence == 1);
    CHECK(arm.sample_time_ns == sample_time_ns);
    CHECK(arm.publish_time_ns >= sample_time_ns);
    CHECK(arm.fresh);
    // The robot's clock estimate came with the record: both ends share this machine's clock.
    REQUIRE(arm.age_ns.has_value());
    CHECK(*arm.age_ns >= 0);
    CHECK(*arm.age_ns < 2'000'000'000);
    CHECK(arm.receive_age_ns >= 0);

    CHECK(session.client->publish_robot_record("arm", bytes("joints 2"), core::os_monotonic_now_ns()));
    REQUIRE(wait_until([&] { return latest_text(*store, "arm") == "joints 2"; }));
    CHECK(store->latest("arm", core::os_monotonic_now_ns()).sequence == 2);
    CHECK(store->dropped_records() == 0);
    CHECK(session.still_up());

    session.client->close();
    CHECK_FALSE(session.client->robot_state_active());
    CHECK_FALSE(session.client->publish_robot_record("arm", bytes("late"), core::os_monotonic_now_ns()));
    session.host->close();
    CHECK_FALSE(store->connected());
    // What the robot app sent stays readable after it left.
    CHECK(latest_text(*store, "arm") == "joints 2");
}

TEST_CASE("RemoteSessionClient: robot state is not sent before start_robot_state()", "[remote][session][robot_state]")
{
    const auto store = std::make_shared<core::RobotStateStore>();
    Session session(store);
    session.client->set_robot_state({ series("arm") });

    CHECK_FALSE(session.client->publish_robot_record("arm", bytes("early"), core::os_monotonic_now_ns()));
    // A series nobody declared is not looked at either.
    CHECK_FALSE(session.client->publish_robot_record("leg", bytes("early"), core::os_monotonic_now_ns()));
    REQUIRE(session.still_up());
    CHECK(session.link->robot_state_sent().empty());
    CHECK(store->series().empty());

    session.client->start_robot_state();
    session.client->start_robot_state();
    REQUIRE(wait_until([&] { return store->series().size() == 1; }));
    CHECK(session.client->publish_robot_record("arm", bytes("joints"), core::os_monotonic_now_ns()));
    REQUIRE(wait_until([&] { return latest_text(*store, "arm") == "joints"; }));
    // One SETUP, however often robot state is started.
    CHECK(session.link->robot_state_sent() == std::vector<uint8_t>{ kSetup, kSamples });
}

TEST_CASE("RemoteSessionClient: robot state calls refuse what the publisher refuses", "[remote][session][robot_state]")
{
    const auto store = std::make_shared<core::RobotStateStore>();
    Session session(store);

    session.client->set_robot_state({ series("arm") });
    // A refused set keeps the series declared before.
    CHECK_THROWS_AS(session.client->set_robot_state({ series("leg"), series("leg") }), std::invalid_argument);
    CHECK_THROWS_AS(session.client->set_robot_state({ series("") }), std::invalid_argument);
    session.client->start_robot_state();
    REQUIRE(wait_until([&] { return store->series().size() == 1; }));
    CHECK(store->series()[0].name == "arm");

    CHECK_THROWS_AS(session.client->set_robot_state({ series("leg") }), std::logic_error);
    CHECK_THROWS_AS(
        session.client->publish_robot_record("leg", bytes("x"), core::os_monotonic_now_ns()), std::invalid_argument);
    const std::vector<uint8_t> too_large(core::kMaxRobotRecordBytes + 1);
    CHECK_THROWS_AS(
        session.client->publish_robot_record("arm", too_large, core::os_monotonic_now_ns()), std::invalid_argument);
    CHECK(session.client->publish_robot_record("arm", bytes("joints"), core::os_monotonic_now_ns()));
    REQUIRE(wait_until([&] { return latest_text(*store, "arm") == "joints"; }));
    CHECK(session.still_up());
}

TEST_CASE("RemoteSessionClient: robot state without declared series starts and sends nothing",
          "[remote][session][robot_state]")
{
    const auto store = std::make_shared<core::RobotStateStore>();
    Session session(store);
    session.client->start_robot_state();
    CHECK(session.client->robot_state_active());
    REQUIRE(session.still_up());
    CHECK(session.link->robot_state_sent().empty());
    CHECK(store->series().empty());
}

TEST_CASE("RemoteSessionClient: robot state samples the link cannot send yet merge into one",
          "[remote][session][robot_state]")
{
    const auto store = std::make_shared<core::RobotStateStore>();
    Session session(store);
    session.client->set_robot_state({ series("arm"), series("wrist") });
    session.client->start_robot_state();
    REQUIRE(wait_until([&] { return store->series().size() == 2; }));

    session.link->hold(true);
    const int64_t now_ns = core::os_monotonic_now_ns();
    CHECK(session.client->publish_robot_record("arm", bytes("arm 1"), now_ns));
    CHECK(session.client->publish_robot_record("wrist", bytes("wrist 1"), now_ns));
    CHECK(session.client->publish_robot_record("arm", bytes("arm 2"), now_ns));
    CHECK(session.client->publish_robot_record("arm", bytes("arm 3"), now_ns));
    std::this_thread::sleep_for(50ms);
    CHECK(session.link->robot_state_sent() == std::vector<uint8_t>{ kSetup });
    CHECK_FALSE(latest_text(*store, "arm").has_value());

    session.link->hold(false);
    REQUIRE(wait_until([&] { return latest_text(*store, "arm") == "arm 3"; }));
    // One link message carried the newest record of each series, so no series is lost.
    CHECK(latest_text(*store, "wrist") == "wrist 1");
    CHECK(session.link->robot_state_sent() == std::vector<uint8_t>{ kSetup, kSamples });
    CHECK(store->latest("arm", core::os_monotonic_now_ns()).sequence == 1);
    CHECK(store->latest("wrist", core::os_monotonic_now_ns()).sequence == 1);
    CHECK(store->dropped_records() == 0);
}

TEST_CASE("RemoteSessionClient: robot state records of an in-order series all arrive, in order",
          "[remote][session][robot_state]")
{
    const auto store = std::make_shared<core::RobotStateStore>();
    Session session(store);
    session.client->set_robot_state({ series("robot_description", true), series("arm") });
    session.client->start_robot_state();
    REQUIRE(wait_until([&] { return store->series().size() == 2; }));

    session.link->hold(true);
    const int64_t now_ns = core::os_monotonic_now_ns();
    // A newest-only record published first leaves after the in-order ones queued behind it.
    CHECK(session.client->publish_robot_record("arm", bytes("arm"), now_ns));
    for (int i = 1; i <= 5; ++i)
    {
        CHECK(session.client->publish_robot_record(
            "robot_description", bytes("description " + std::to_string(i)), now_ns));
    }
    session.link->hold(false);

    REQUIRE(wait_until([&] { return latest_text(*store, "arm") == "arm"; }));
    CHECK(latest_text(*store, "robot_description") == "description 5");
    CHECK(store->latest("robot_description", core::os_monotonic_now_ns()).sequence == 5);
    CHECK(store->dropped_records() == 0);
    CHECK(session.link->robot_state_sent() ==
          std::vector<uint8_t>{ kSetup, kRecords, kRecords, kRecords, kRecords, kRecords, kSamples });
}

TEST_CASE("RemoteSessionClient: robot state published on two threads arrives in publish order",
          "[remote][session][robot_state]")
{
    constexpr int kPerThread = 200;
    const auto store = std::make_shared<core::RobotStateStore>();
    Session session(store);
    session.client->set_robot_state({ series("log", true), series("arm") });
    session.client->start_robot_state();
    REQUIRE(wait_until([&] { return store->series().size() == 2; }));

    {
        const auto publish = [&](const std::string& who)
        {
            for (int i = 1; i <= kPerThread; ++i)
            {
                const std::vector<uint8_t> record = bytes(who + " " + std::to_string(i));
                session.client->publish_robot_record("log", record, core::os_monotonic_now_ns());
                session.client->publish_robot_record("arm", record, core::os_monotonic_now_ns());
            }
        };
        std::jthread first(publish, "first");
        std::jthread second(publish, "second");
    }

    // The store drops a record that arrives behind a later one of its series.
    REQUIRE(wait_until(
        [&] { return store->latest("log", core::os_monotonic_now_ns()).sequence == uint64_t{ 2 * kPerThread }; }));
    const std::string last = "first " + std::to_string(kPerThread);
    const std::string other_last = "second " + std::to_string(kPerThread);
    const std::optional<std::string> log = latest_text(*store, "log");
    CHECK((log == last || log == other_last));
    REQUIRE(wait_until([&] { return latest_text(*store, "arm") == log; }));
    CHECK(store->dropped_records() == 0);
    CHECK(session.still_up());
}

TEST_CASE("RemoteSessionHost: robot state is dropped without a store", "[remote][session][robot_state]")
{
    Session session;
    session.client->set_robot_state({ series("robot_description", true), series("arm") });
    session.client->start_robot_state();
    CHECK(session.client->publish_robot_record("robot_description", bytes("what"), core::os_monotonic_now_ns()));
    CHECK(session.client->publish_robot_record("arm", bytes("joints 1"), core::os_monotonic_now_ns()));
    REQUIRE(wait_until([&] { return session.link->robot_state_sent().size() == 3; }));
    REQUIRE(session.still_up());

    // A store attached afterwards never saw the SETUP: its records are on channels it does not know.
    const auto store = std::make_shared<core::RobotStateStore>();
    session.host->attach_robot_state(store);
    CHECK(store->connected());
    CHECK(session.client->publish_robot_record("arm", bytes("joints 2"), core::os_monotonic_now_ns()));
    REQUIRE(wait_until([&] { return store->dropped_records() == 1; }));
    CHECK(store->series().empty());
    CHECK(session.still_up());
}

TEST_CASE("RemoteSessionHost: robot state a malformed payload does not end the session", "[remote][session][robot_state]")
{
    int fds[2];
    REQUIRE(socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, fds) == 0);
    const auto host = core::RemoteSessionHost::adopt(std::make_unique<core::TcpTransport>(core::TcpSocket(fds[1])));
    core::TcpTransport robot{ core::TcpSocket(fds[0]) };
    const auto store = std::make_shared<core::RobotStateStore>();
    host->attach_robot_state(store);
    const uint64_t generation = store->generation();

    send_raw(robot, core::LinkMessageType::RobotStateSetup, bytes("not MCAP records"));
    send_raw(robot, core::LinkMessageType::RobotStateSamples, bytes("abc"));
    send_raw(robot, core::LinkMessageType::RobotStateRecords, {});
    send_raw(robot, core::LinkMessageType::RobotStateSamples,
             core::encode_robot_samples_payload(core::RobotSamplesPrefix{ 1, 0, 0, true, true }, bytes("cut off")));

    // Well-formed ones behind them are taken.
    core::RobotStatePublisher publisher;
    publisher.declare_series(series("arm"));
    send_raw(robot, core::LinkMessageType::RobotStateSetup, publisher.take_setup());
    const int64_t now_ns = core::os_monotonic_now_ns();
    publisher.publish("arm", bytes("joints"), now_ns, now_ns);
    send_raw(robot, core::LinkMessageType::RobotStateSamples,
             core::encode_robot_samples_payload(
                 core::RobotSamplesPrefix{ 2, now_ns, 0, true, true }, publisher.take_latest_records()));
    REQUIRE(wait_until([&] { return latest_text(*store, "arm") == "joints"; }));
    CHECK(store->generation() > generation);
    CHECK(store->series().size() == 1);
    CHECK(host->alive());

    // The rest of the session goes on.
    send_raw(robot, core::LinkMessageType::Hello, bytes(kHello));
    CHECK(host->wait_hello(kWait) == kHello);
}

TEST_CASE("RemoteSessionHost: robot state ends the store's session once per attach", "[remote][session][robot_state]")
{
    const auto store = std::make_shared<core::RobotStateStore>();
    CHECK_FALSE(store->connected());
    {
        Session first(store);
        CHECK(store->connected());
        CHECK(store->session_count() == 1);
        first.host->close();
        CHECK_FALSE(store->connected());

        // The next robot app's session: a second close() and the destructor of the first host leave it alone.
        store->begin_session();
        first.host->close();
        CHECK(store->connected());
        first.host->attach_robot_state(store);
        CHECK(store->session_count() == 2);
        first.host.reset();
        CHECK(store->connected());
        CHECK(store->session_count() == 2);
    }
    store->end_session();

    Session second(store);
    CHECK(store->session_count() == 3);
    // Detaching ends the session; a store attached in its place begins one of its own.
    second.host->attach_robot_state(nullptr);
    CHECK_FALSE(store->connected());
    second.host->attach_robot_state(store);
    CHECK(store->connected());
    CHECK(store->session_count() == 4);
    const auto other = std::make_shared<core::RobotStateStore>();
    second.host->attach_robot_state(other);
    CHECK_FALSE(store->connected());
    CHECK(other->connected());
    second.host->attach_robot_state(other);
    CHECK(other->session_count() == 2);
    second.host.reset();
    CHECK_FALSE(other->connected());
    CHECK(store->session_count() == 4);
}
