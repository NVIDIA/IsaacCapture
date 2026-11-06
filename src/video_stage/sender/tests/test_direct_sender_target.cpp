// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Retargeting a DirectSender at run time: the connection moves, the new destination starts at a keyframe, an empty
// host idles, and none of it can hang the caller or the destructor.

#include <arpa/inet.h>
#include <catch2/catch_test_macros.hpp>
#include <netinet/in.h>
#include <sys/socket.h>
#include <video_stage/avs1.hpp>
#include <video_stage/direct_sender.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <poll.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using plugins::sensing::DirectSender;
using plugins::sensing::DirectSenderConfig;
using plugins::sensing::EncodedVideoFrame;
using namespace std::chrono_literals;

namespace
{

using Clock = std::chrono::steady_clock;

/// A loopback receiver that accepts the sender's DATA and CTRL connections and reads AVS1 records from DATA.
class Peer
{
public:
    explicit Peer(int backlog = 8)
    {
        m_listen = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        REQUIRE(::bind(m_listen, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
        REQUIRE(::listen(m_listen, backlog) == 0);
        socklen_t len = sizeof(addr);
        ::getsockname(m_listen, reinterpret_cast<sockaddr*>(&addr), &len);
        m_port = ntohs(addr.sin_port);
    }

    ~Peer()
    {
        ::close(m_listen);
        close_connections();
    }

    std::uint16_t port() const
    {
        return m_port;
    }

    /// Accepts the DATA and CTRL connections; true when both hellos arrived.
    bool accept_pair(int timeout_ms = 5000)
    {
        for (int i = 0; i < 2; ++i)
        {
            pollfd pfd{ m_listen, POLLIN, 0 };
            if (::poll(&pfd, 1, timeout_ms) <= 0)
                return false;
            const int fd = ::accept(m_listen, nullptr, nullptr);
            if (fd < 0)
                return false;
            char hello[4];
            if (!read_exact(fd, reinterpret_cast<std::uint8_t*>(hello), 4, timeout_ms))
            {
                ::close(fd);
                return false;
            }
            const std::string tag(hello, 4);
            if (tag == "DATA")
                m_data = fd;
            else if (tag == "CTRL")
                m_control = fd;
            else
            {
                ::close(fd);
                return false;
            }
        }
        return m_data >= 0 && m_control >= 0;
    }

    /// One AVS1 record's bytes after the length prefix; nullopt on EOF or timeout.
    std::optional<std::vector<std::uint8_t>> read_record(int timeout_ms = 5000)
    {
        std::uint8_t prefix[4];
        if (m_data < 0 || !read_exact(m_data, prefix, 4, timeout_ms))
            return std::nullopt;
        std::vector<std::uint8_t> body(video_stage::avs1::read_length(prefix));
        if (!read_exact(m_data, body.data(), body.size(), timeout_ms))
            return std::nullopt;
        return body;
    }

    /// True if the DATA connection was closed by the sender within the timeout (records still in flight are drained).
    bool data_closed(int timeout_ms = 5000)
    {
        const auto until = Clock::now() + std::chrono::milliseconds(timeout_ms);
        while (Clock::now() < until)
        {
            pollfd pfd{ m_data, POLLIN, 0 };
            if (::poll(&pfd, 1, 100) > 0)
            {
                std::uint8_t buffer[65536];
                const ssize_t n = ::recv(m_data, buffer, sizeof(buffer), 0);
                if (n <= 0)
                    return true;
            }
        }
        return false;
    }

    /// Whether any new connection arrives within the timeout.
    bool any_connection(int timeout_ms)
    {
        pollfd pfd{ m_listen, POLLIN, 0 };
        return ::poll(&pfd, 1, timeout_ms) > 0;
    }

    /// The first record of the DATA stream must carry an IDR: the PC cannot start at a delta frame.
    bool first_record_is_keyframe()
    {
        const auto body = read_record();
        if (!body)
            return false;
        std::string error;
        const auto record = video_stage::avs1::parse(body->data(), body->size(), error);
        return record && video_stage::avs1::contains_idr(record->payload, record->payload_size);
    }

    void close_connections()
    {
        for (int* fd : { &m_data, &m_control })
        {
            if (*fd >= 0)
                ::close(*fd);
            *fd = -1;
        }
    }

    /// Fills the accept queue with half-open clients so later SYNs are dropped and a connect hangs.
    void saturate(std::vector<int>& clients)
    {
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = htons(m_port);
        for (int i = 0; i < 8; ++i)
        {
            const int c = ::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
            ::connect(c, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
            clients.push_back(c);
        }
    }

private:
    static bool read_exact(int fd, std::uint8_t* out, std::size_t size, int timeout_ms)
    {
        std::size_t done = 0;
        while (done < size)
        {
            pollfd pfd{ fd, POLLIN, 0 };
            if (::poll(&pfd, 1, timeout_ms) <= 0)
                return false;
            const ssize_t n = ::recv(fd, out + done, size - done, 0);
            if (n <= 0)
                return false;
            done += static_cast<std::size_t>(n);
        }
        return true;
    }

    int m_listen = -1;
    int m_data = -1;
    int m_control = -1;
    std::uint16_t m_port = 0;
};

/// Plays the encoder: a unit every few milliseconds, a keyframe every tenth and whenever the sender asks.
class Feeder
{
public:
    Feeder(DirectSender& sender, std::size_t unit_bytes = 2000) : m_sender(sender), m_unit_bytes(unit_bytes)
    {
        m_thread = std::thread(
            [this]
            {
                std::uint32_t id = 0;
                bool force_key = true;
                while (!m_stop)
                {
                    force_key |= m_sender.take_keyframe_request();
                    const bool key = force_key || (id % 10 == 0);
                    force_key = false;
                    std::vector<std::uint8_t> unit = { 0, 0, 0, 1, static_cast<std::uint8_t>(key ? 0x65 : 0x41), 0x88 };
                    unit.resize(m_unit_bytes, 0x5a);
                    EncodedVideoFrame frame;
                    frame.data = unit.data();
                    frame.size = unit.size();
                    frame.keyframe = key;
                    frame.frame_id = ++id;
                    frame.user_timestamp_us = id * 1000ULL;
                    m_sender.submit(frame);
                    std::this_thread::sleep_for(5ms);
                }
            });
    }
    ~Feeder()
    {
        m_stop = true;
        m_thread.join();
    }

private:
    DirectSender& m_sender;
    std::size_t m_unit_bytes;
    std::atomic<bool> m_stop{ false };
    std::thread m_thread;
};

DirectSenderConfig config_for(const std::string& host, std::uint16_t port)
{
    DirectSenderConfig config;
    config.host = host;
    config.port = port;
    config.width = 64;
    config.height = 48;
    config.max_queued_bytes = 512U << 20;
    config.max_queue_age = std::chrono::hours(1);
    return config;
}

template <typename Pred>
bool wait_for(Pred pred, std::chrono::milliseconds limit = 5s)
{
    const auto until = Clock::now() + limit;
    while (Clock::now() < until)
    {
        if (pred())
            return true;
        std::this_thread::sleep_for(5ms);
    }
    return pred();
}

} // namespace

TEST_CASE("the configured target is reported by host() and port()", "[direct_sender_target]")
{
    Peer a;
    DirectSender sender(config_for("127.0.0.1", a.port()));
    CHECK(sender.host() == "127.0.0.1");
    CHECK(sender.port() == a.port());
    REQUIRE(a.accept_pair());
    CHECK(wait_for([&] { return sender.connected(); }));
}

TEST_CASE("retargeting moves the connection and the new peer starts at a keyframe", "[direct_sender_target]")
{
    Peer a;
    Peer b;
    DirectSender sender(config_for("127.0.0.1", a.port()));
    Feeder feeder(sender);

    REQUIRE(a.accept_pair());
    CHECK(a.first_record_is_keyframe());
    REQUIRE(wait_for([&] { return sender.connected(); }));

    sender.set_target("127.0.0.1", b.port());
    CHECK(sender.host() == "127.0.0.1");
    CHECK(sender.port() == b.port());

    REQUIRE(b.accept_pair());
    CHECK(b.first_record_is_keyframe());
    CHECK(a.data_closed()); // the old destination was left
    REQUIRE(wait_for([&] { return sender.connected(); }));

    // The new peer keeps receiving, and the old one gets nothing more and no reconnect.
    for (int i = 0; i < 5; ++i)
        CHECK(b.read_record());
    CHECK_FALSE(a.any_connection(300));
}

TEST_CASE("setting the same peer again reconnects it from a keyframe", "[direct_sender_target]")
{
    Peer a;
    DirectSender sender(config_for("127.0.0.1", a.port()));
    Feeder feeder(sender);
    REQUIRE(a.accept_pair());
    REQUIRE(a.first_record_is_keyframe());

    sender.set_target("127.0.0.1", a.port());
    REQUIRE(a.data_closed());
    a.close_connections();
    REQUIRE(a.accept_pair());
    CHECK(a.first_record_is_keyframe());
}

TEST_CASE("an empty host idles the sender", "[direct_sender_target]")
{
    Peer a;
    DirectSender sender(config_for("127.0.0.1", a.port()));
    Feeder feeder(sender);
    REQUIRE(a.accept_pair());
    REQUIRE(wait_for([&] { return sender.connected(); }));

    sender.set_target("", 1234);
    CHECK(sender.host().empty());
    CHECK(sender.port() == 0);
    CHECK(wait_for([&] { return !sender.connected(); }));
    CHECK(a.data_closed());

    // Idle: no reconnect, and submitted units are not queued or counted as sent.
    CHECK_FALSE(a.any_connection(500));
    const auto sent = sender.sent();
    std::this_thread::sleep_for(200ms);
    CHECK(sender.sent() == sent);
    CHECK_FALSE(sender.connected());

    // A target set later wakes it again.
    Peer b;
    sender.set_target("127.0.0.1", b.port());
    REQUIRE(b.accept_pair());
    CHECK(b.first_record_is_keyframe());
    CHECK(wait_for([&] { return sender.connected(); }));
}

TEST_CASE("a sender created without a target idles until one is set", "[direct_sender_target]")
{
    DirectSender sender(config_for("", 0));
    Feeder feeder(sender);
    std::this_thread::sleep_for(300ms);
    CHECK_FALSE(sender.connected());
    CHECK(sender.sent() == 0);
    CHECK(sender.host().empty());

    Peer a;
    sender.set_target("127.0.0.1", a.port());
    REQUIRE(a.accept_pair());
    CHECK(a.first_record_is_keyframe());
}

TEST_CASE("set_target while a send is blocked does not hang and the new peer gets video", "[direct_sender_target]")
{
    Peer stalled;
    Peer b;
    DirectSender sender(config_for("127.0.0.1", stalled.port()));
    REQUIRE(stalled.accept_pair());

    // Units far larger than the socket buffers, to a peer that never reads: the worker ends up blocked in send().
    const std::vector<std::uint8_t> unit = []
    {
        std::vector<std::uint8_t> bytes(1U << 20, 0x5a);
        bytes[0] = 0;
        bytes[1] = 0;
        bytes[2] = 0;
        bytes[3] = 1;
        bytes[4] = 0x65;
        return bytes;
    }();
    for (int i = 0; i < 100; ++i)
    {
        EncodedVideoFrame frame;
        frame.data = unit.data();
        frame.size = unit.size();
        frame.keyframe = (i == 0);
        frame.frame_id = static_cast<std::uint32_t>(i);
        sender.submit(frame);
    }
    std::this_thread::sleep_for(500ms);
    REQUIRE(sender.sent() < 100);

    const auto start = Clock::now();
    sender.set_target("127.0.0.1", b.port());
    CHECK(Clock::now() - start < 1s);

    Feeder feeder(sender);
    REQUIRE(b.accept_pair());
    CHECK(b.first_record_is_keyframe());
}

TEST_CASE("destruction is bounded after a retarget", "[direct_sender_target]")
{
    const auto time_destruction = [](std::unique_ptr<DirectSender> sender)
    {
        const auto start = Clock::now();
        sender.reset();
        return Clock::now() - start;
    };

    SECTION("to a peer that is connected")
    {
        Peer a;
        Peer b;
        auto sender = std::make_unique<DirectSender>(config_for("127.0.0.1", a.port()));
        REQUIRE(a.accept_pair());
        sender->set_target("127.0.0.1", b.port());
        REQUIRE(b.accept_pair());
        std::this_thread::sleep_for(200ms);
        CHECK(time_destruction(std::move(sender)) < 2s);
    }
    SECTION("to a peer whose connect never completes")
    {
        Peer a;
        Peer hung(0);
        std::vector<int> clients;
        hung.saturate(clients);
        auto sender = std::make_unique<DirectSender>(config_for("127.0.0.1", a.port()));
        REQUIRE(a.accept_pair());
        sender->set_target("127.0.0.1", hung.port());
        std::this_thread::sleep_for(500ms); // let the worker reach its connect
        CHECK(time_destruction(std::move(sender)) < 2s);
        for (const int fd : clients)
            ::close(fd);
    }
    SECTION("while idle after an empty host")
    {
        Peer a;
        auto sender = std::make_unique<DirectSender>(config_for("127.0.0.1", a.port()));
        REQUIRE(a.accept_pair());
        sender->set_target("", 0);
        std::this_thread::sleep_for(300ms);
        CHECK(time_destruction(std::move(sender)) < 2s);
    }
}

TEST_CASE("retargeting away from a connect that never completes is prompt", "[direct_sender_target]")
{
    Peer hung(0);
    std::vector<int> clients;
    hung.saturate(clients);
    Peer b;
    DirectSender sender(config_for("127.0.0.1", hung.port()));
    Feeder feeder(sender);
    std::this_thread::sleep_for(500ms);

    sender.set_target("127.0.0.1", b.port());
    REQUIRE(b.accept_pair(3000)); // well inside the 5 s connect timeout
    CHECK(b.first_record_is_keyframe());
    for (const int fd : clients)
        ::close(fd);
}

namespace
{

/// A loopback port nothing listens on: bound once to learn a free number, then released.
std::uint16_t closed_port()
{
    const int s = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    REQUIRE(::bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
    socklen_t len = sizeof(addr);
    ::getsockname(s, reinterpret_cast<sockaddr*>(&addr), &len);
    ::close(s);
    return ntohs(addr.sin_port);
}

bool starts_with(const std::string& text, const std::string& prefix)
{
    return text.compare(0, prefix.size(), prefix) == 0;
}

} // namespace

TEST_CASE("a connect to a closed port is reported by last_error", "[direct_sender_target][last_error]")
{
    const auto port = closed_port();
    DirectSender sender(config_for("127.0.0.1", port));
    REQUIRE(wait_for([&] { return !sender.last_error().empty(); }));
    const auto error = sender.last_error();
    CHECK(starts_with(error, "cannot connect to 127.0.0.1:" + std::to_string(port) + ": "));
    CHECK_FALSE(sender.connected());
}

TEST_CASE("a successful connection leaves last_error empty", "[direct_sender_target][last_error]")
{
    Peer a;
    DirectSender sender(config_for("127.0.0.1", a.port()));
    REQUIRE(a.accept_pair());
    REQUIRE(wait_for([&] { return sender.connected(); }));
    CHECK(sender.last_error().empty());
}

TEST_CASE("set_target to a reachable peer clears a connect error", "[direct_sender_target][last_error]")
{
    Peer a;
    DirectSender sender(config_for("127.0.0.1", closed_port()));
    REQUIRE(wait_for([&] { return !sender.last_error().empty(); }));

    sender.set_target("127.0.0.1", a.port());
    REQUIRE(a.accept_pair());
    REQUIRE(wait_for([&] { return sender.connected(); }));
    CHECK(sender.last_error().empty());
}

TEST_CASE("set_target to an empty host clears last_error and it stays clear", "[direct_sender_target][last_error]")
{
    DirectSender sender(config_for("127.0.0.1", closed_port()));
    REQUIRE(wait_for([&] { return !sender.last_error().empty(); }));

    sender.set_target("", 0);
    CHECK(sender.last_error().empty());
    std::this_thread::sleep_for(300ms);
    CHECK(sender.last_error().empty());
}

TEST_CASE("set_target to another closed port reports the new destination", "[direct_sender_target][last_error]")
{
    DirectSender sender(config_for("127.0.0.1", closed_port()));
    REQUIRE(wait_for([&] { return !sender.last_error().empty(); }));

    const auto second = closed_port();
    sender.set_target("127.0.0.1", second);
    REQUIRE(wait_for(
        [&] { return starts_with(sender.last_error(), "cannot connect to 127.0.0.1:" + std::to_string(second)); }));
}

TEST_CASE("a retarget away from a peer that then closes leaves no stale control error",
          "[direct_sender_target][last_error]")
{
    Peer a;
    Peer b;
    DirectSender sender(config_for("127.0.0.1", a.port()));
    REQUIRE(a.accept_pair());
    REQUIRE(wait_for([&] { return sender.connected(); }));

    sender.set_target("127.0.0.1", b.port());
    a.close_connections();
    REQUIRE(b.accept_pair());
    REQUIRE(wait_for([&] { return sender.connected(); }));
    std::this_thread::sleep_for(500ms);
    CHECK(sender.last_error().find("closed the control connection") == std::string::npos);
    CHECK(sender.last_error().empty());
}
