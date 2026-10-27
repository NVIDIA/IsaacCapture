// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "test_support.hpp"

#include <arpa/inet.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <netinet/in.h>
#include <oxr_utils/os_time.hpp>
#include <remote_link/link.hpp>
#include <remote_link/message_header.hpp>
#include <remote_link/messages.hpp>
#include <remote_link/tcp_transport.hpp>
#include <remote_link/websocket_transport.hpp>
#include <sys/socket.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using namespace core;
using namespace remote_link_test;

namespace
{

//! Shared state of an in-memory duplex connection.
struct Pipe
{
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::vector<uint8_t>> queue[2]; //!< queue[i]: messages waiting for end i
    bool closed = false;
    bool write_closed[2] = { false, false }; //!< write_closed[i]: end i half-closed its sending direction
    bool held[2] = { false, false };
    bool blocked[2] = { false, false };
};

/*!
 * One end of a Pipe. While held, send() blocks like a write into a full socket and wait_writable() reports false,
 * so latest-wins messages stay queued in the Link. Shutdown delivers what was already sent, then Closed;
 * shutdown_write() does that in one direction.
 */
class BlockingTransport final : public ITransport
{
public:
    BlockingTransport(std::shared_ptr<Pipe> pipe, int index) : m_pipe(std::move(pipe)), m_index(index)
    {
    }

    bool send(const uint8_t* message, size_t size) override
    {
        std::unique_lock<std::mutex> lock(m_pipe->mutex);
        m_pipe->blocked[m_index] = m_pipe->held[m_index];
        m_pipe->cv.notify_all();
        m_pipe->cv.wait(lock, [this] { return !m_pipe->held[m_index] || m_pipe->closed; });
        m_pipe->blocked[m_index] = false;
        if (m_pipe->closed || m_pipe->write_closed[m_index])
        {
            return false;
        }
        m_pipe->queue[1 - m_index].emplace_back(message, message + size);
        m_pipe->cv.notify_all();
        return true;
    }

    ReceiveResult receive(std::vector<uint8_t>& message, std::chrono::milliseconds timeout) override
    {
        std::unique_lock<std::mutex> lock(m_pipe->mutex);
        auto& inbox = m_pipe->queue[m_index];
        const auto ended = [&] { return m_pipe->closed || m_pipe->write_closed[1 - m_index]; };
        if (!m_pipe->cv.wait_for(lock, timeout, [&] { return !inbox.empty() || ended(); }))
        {
            return ReceiveResult::Timeout;
        }
        if (inbox.empty())
        {
            return ReceiveResult::Closed;
        }
        message = std::move(inbox.front());
        inbox.pop_front();
        return ReceiveResult::Ok;
    }

    void shutdown() override
    {
        std::lock_guard<std::mutex> lock(m_pipe->mutex);
        m_pipe->closed = true;
        m_pipe->cv.notify_all();
    }

    void shutdown_write() override
    {
        std::lock_guard<std::mutex> lock(m_pipe->mutex);
        m_pipe->write_closed[m_index] = true;
        m_pipe->cv.notify_all();
    }

    bool wait_writable(std::chrono::milliseconds timeout) override
    {
        std::unique_lock<std::mutex> lock(m_pipe->mutex);
        return m_pipe->cv.wait_for(lock, timeout, [this] { return !m_pipe->held[m_index] || m_pipe->closed; });
    }

private:
    std::shared_ptr<Pipe> m_pipe;
    int m_index;
};

struct Decoded
{
    LinkMessageHeader header;
    std::vector<uint8_t> payload;
};

//! The test's side of a pipe: reads and writes raw link messages.
class Peer
{
public:
    explicit Peer(std::shared_ptr<Pipe> pipe) : m_pipe(pipe), m_transport(pipe, 1)
    {
    }

    //! Next message of `type`; other types (e.g. HEARTBEAT) are skipped unless `type` is nullopt.
    std::optional<Decoded> next(std::optional<LinkMessageType> type, std::chrono::milliseconds timeout = kWait)
    {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        std::vector<uint8_t> raw;
        while (std::chrono::steady_clock::now() < deadline)
        {
            const auto remaining =
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
            if (m_transport.receive(raw, remaining) != ReceiveResult::Ok)
            {
                return std::nullopt;
            }
            Decoded decoded{ decode_header(raw.data()), std::vector<uint8_t>(raw.begin() + kHeaderSize, raw.end()) };
            REQUIRE(raw.size() == kHeaderSize + decoded.header.payload_size);
            if (!type || decoded.header.type == static_cast<uint8_t>(*type))
            {
                return decoded;
            }
        }
        return std::nullopt;
    }

    ReceiveResult receive(std::vector<uint8_t>& raw, std::chrono::milliseconds timeout = kWait)
    {
        return m_transport.receive(raw, timeout);
    }

    void send(uint8_t type, const std::vector<uint8_t>& payload = {})
    {
        std::vector<uint8_t> message = encode_link_message(type, payload);
        stamp_header(message.data(), m_seq++, os_monotonic_now_ns());
        REQUIRE(m_transport.send(message.data(), message.size()));
    }

    void send(LinkMessageType type, const std::vector<uint8_t>& payload = {})
    {
        send(static_cast<uint8_t>(type), payload);
    }

    //! Holds or releases the Link's sends (end 0).
    void hold_link(bool held)
    {
        std::lock_guard<std::mutex> lock(m_pipe->mutex);
        m_pipe->held[0] = held;
        m_pipe->cv.notify_all();
    }

    bool link_send_blocked()
    {
        std::lock_guard<std::mutex> lock(m_pipe->mutex);
        return m_pipe->blocked[0];
    }

    void shutdown()
    {
        m_transport.shutdown();
    }

private:
    std::shared_ptr<Pipe> m_pipe;
    BlockingTransport m_transport;
    uint32_t m_seq = 0;
};

//! Messages delivered to a Link's handler.
class Inbox
{
public:
    LinkMessageHandler handler()
    {
        return [this](ReceivedLinkMessage&& message)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_messages.push_back(std::move(message));
            m_cv.notify_all();
        };
    }

    std::optional<ReceivedLinkMessage> wait(std::chrono::milliseconds timeout = kWait)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (!m_cv.wait_for(lock, timeout, [this] { return !m_messages.empty(); }))
        {
            return std::nullopt;
        }
        ReceivedLinkMessage message = std::move(m_messages.front());
        m_messages.pop_front();
        return message;
    }

    size_t size()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_messages.size();
    }

private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::deque<ReceivedLinkMessage> m_messages;
};

struct Fixture
{
    std::shared_ptr<Pipe> pipe = std::make_shared<Pipe>();
    Peer peer{ pipe };
    Inbox inbox;
    std::unique_ptr<Link> link;

    explicit Fixture(LinkOptions options = quiet_options())
    {
        link = std::make_unique<Link>(std::make_unique<BlockingTransport>(pipe, 0), options, inbox.handler());
    }

    //! Long heartbeat and peer timeout so only what the test sends goes over the pipe.
    static LinkOptions quiet_options()
    {
        LinkOptions options;
        options.heartbeat_period = std::chrono::milliseconds{ 60'000 };
        options.peer_timeout = std::chrono::milliseconds{ 60'000 };
        options.receive_poll = std::chrono::milliseconds{ 10 };
        options.close_drain = std::chrono::milliseconds{ 2'000 };
        options.name = "test";
        return options;
    }
};

//! A client socket whose small receive buffer is set before connecting, so the window stays small.
TcpSocket small_receive_buffer_client(uint16_t port)
{
    const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    REQUIRE(fd >= 0);
    TcpSocket socket(fd);
    const int receive_buffer = 4096;
    REQUIRE(setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &receive_buffer, sizeof(receive_buffer)) == 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    REQUIRE(::connect(fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0);
    return socket;
}

std::vector<uint8_t> bytes(const std::string& text)
{
    return std::vector<uint8_t>(text.begin(), text.end());
}

//! The robot end of a TCP loopback connection whose server end sent the first `sent` bytes of a message, then reset
//! it.
std::unique_ptr<ITransport> reset_after(size_t sent)
{
    Listener listener(Endpoint{ "127.0.0.1", 0 });
    std::unique_ptr<ITransport> robot_end =
        connect_transport(parse_remote("tcp://127.0.0.1:" + std::to_string(listener.port())), kWait);
    const std::optional<Listener::Accepted> accepted = listener.accept(kWait);
    REQUIRE(accepted.has_value());
    TcpSocket server_end(accepted->fd);
    const std::vector<uint8_t> message = message_of_size(kHeaderSize + 100);
    REQUIRE(server_end.send_all(message.data(), sent));
    const linger abort{ 1, 0 };
    REQUIRE(::setsockopt(server_end.fd(), SOL_SOCKET, SO_LINGER, &abort, sizeof(abort)) == 0);
    return robot_end;
}

std::string text(const std::vector<uint8_t>& data)
{
    return std::string(data.begin(), data.end());
}

} // namespace

TEST_CASE("RemoteLink: Link sends reliable messages in FIFO order and never drops them", "[remote_link][unit]")
{
    Fixture f;
    f.link->send_reliable(LinkMessageType::Hello, bytes("queued before start"));
    f.link->start();
    f.peer.hold_link(true);
    constexpr int kCount = 500;
    for (int i = 0; i < kCount; ++i)
    {
        f.link->send_reliable(LinkMessageType::Status, bytes(std::to_string(i)));
    }
    REQUIRE(wait_until([&] { return f.peer.link_send_blocked(); }));
    f.peer.hold_link(false);

    const auto hello = f.peer.next(LinkMessageType::Hello);
    REQUIRE(hello);
    REQUIRE(text(hello->payload) == "queued before start");
    uint32_t last_seq = hello->header.seq;
    for (int i = 0; i < kCount; ++i)
    {
        const auto status = f.peer.next(LinkMessageType::Status);
        REQUIRE(status);
        REQUIRE(text(status->payload) == std::to_string(i));
        REQUIRE(status->header.seq == last_seq + 1);
        REQUIRE(status->header.t_send_ns > 0);
        last_seq = status->header.seq;
    }
    REQUIRE(f.link->alive());
    REQUIRE(wait_until([&] { return f.link->stats().messages_sent == kCount + 1; }));
}

TEST_CASE("RemoteLink: Link latest-wins keeps only the newest unsent payload per type", "[remote_link][unit]")
{
    Fixture f;
    f.link->start();
    f.peer.hold_link(true);
    f.link->send_latest(LinkMessageType::Samples, bytes("samples 1"));
    f.link->send_latest(LinkMessageType::Samples, bytes("samples 2"));
    f.link->send_latest(LinkMessageType::Status, bytes("status"));
    f.link->send_latest(LinkMessageType::Samples, bytes("samples 3"));
    REQUIRE(f.link->stats().latest_replaced == 2);
    // The transport reports not writable, so nothing was taken off the slots.
    std::this_thread::sleep_for(std::chrono::milliseconds{ 30 });
    REQUIRE_FALSE(f.peer.link_send_blocked());
    f.peer.hold_link(false);

    const auto samples = f.peer.next(LinkMessageType::Samples);
    REQUIRE(samples);
    REQUIRE(text(samples->payload) == "samples 3");
    const auto status = f.peer.next(LinkMessageType::Status);
    REQUIRE(status);
    REQUIRE(text(status->payload) == "status");
    REQUIRE_FALSE(f.peer.next(LinkMessageType::Samples, std::chrono::milliseconds{ 100 }));

    // Once writable, each new payload goes out on its own.
    f.link->send_latest(LinkMessageType::Samples, bytes("samples 4"));
    const auto next = f.peer.next(LinkMessageType::Samples);
    REQUIRE(next);
    REQUIRE(text(next->payload) == "samples 4");
}

TEST_CASE("RemoteLink: Link merges a latest-wins payload with the unsent one", "[remote_link][unit]")
{
    Fixture f;
    f.link->start();
    f.peer.hold_link(true);
    std::vector<std::pair<std::string, std::string>> calls;
    const MergeFn merge = [&calls](std::vector<uint8_t> queued, std::vector<uint8_t> incoming)
    {
        calls.emplace_back(text(queued), text(incoming));
        queued.push_back('+');
        queued.insert(queued.end(), incoming.begin(), incoming.end());
        return queued;
    };
    f.link->send_latest(LinkMessageType::Samples, bytes("a"), merge);
    REQUIRE(calls.empty());
    f.link->send_latest(LinkMessageType::Samples, bytes("b"), merge);
    f.link->send_latest(LinkMessageType::Samples, bytes("c"), merge);
    REQUIRE(calls == std::vector<std::pair<std::string, std::string>>{ { "a", "b" }, { "a+b", "c" } });
    f.peer.hold_link(false);

    const auto samples = f.peer.next(LinkMessageType::Samples);
    REQUIRE(samples);
    REQUIRE(text(samples->payload) == "a+b+c");
    REQUIRE_FALSE(f.peer.next(LinkMessageType::Samples, std::chrono::milliseconds{ 100 }));
}

TEST_CASE("RemoteLink: Link reliable messages pass a blocked latest-wins slot", "[remote_link][unit]")
{
    Fixture f;
    f.link->start();
    f.peer.hold_link(true);
    f.link->send_latest(LinkMessageType::Samples, bytes("samples"));
    f.link->send_reliable(LinkMessageType::Status, bytes("status"));
    REQUIRE(wait_until([&] { return f.peer.link_send_blocked(); }));
    f.peer.hold_link(false);
    const auto first = f.peer.next(std::nullopt);
    REQUIRE(first);
    REQUIRE(first->header.type == static_cast<uint8_t>(LinkMessageType::Status));
    REQUIRE(f.peer.next(LinkMessageType::Samples));
}

TEST_CASE("RemoteLink: Link sends HEARTBEAT when idle and stays alive on the peer's", "[remote_link][unit]")
{
    LinkOptions options = Fixture::quiet_options();
    options.heartbeat_period = std::chrono::milliseconds{ 50 };
    options.peer_timeout = std::chrono::milliseconds{ 300 };
    Fixture f(options);
    f.link->start();
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 12; ++i)
    {
        const auto heartbeat = f.peer.next(std::nullopt);
        REQUIRE(heartbeat);
        REQUIRE(heartbeat->header.type == static_cast<uint8_t>(LinkMessageType::Heartbeat));
        REQUIRE(heartbeat->payload.empty());
        f.peer.send(LinkMessageType::Heartbeat);
    }
    // 12 heartbeats at 50 ms span more than the 300 ms peer timeout.
    REQUIRE(std::chrono::steady_clock::now() - start > options.peer_timeout);
    REQUIRE(f.link->alive());
    REQUIRE(f.inbox.size() == 0);
}

TEST_CASE("RemoteLink: Link dies after the peer timeout", "[remote_link][unit]")
{
    LinkOptions options = Fixture::quiet_options();
    options.peer_timeout = std::chrono::milliseconds{ 150 };
    Fixture f(options);
    f.link->start();
    REQUIRE(f.link->alive());
    REQUIRE(f.link->close_reason().empty());
    REQUIRE(wait_until([&] { return !f.link->alive(); }));
    REQUIRE(f.link->close_reason().rfind("no messages for ", 0) == 0);
    REQUIRE(f.link->close_reason().find(" ms") != std::string::npos);

    // Dead links drop new messages instead of throwing.
    f.link->send_reliable(LinkMessageType::Status, bytes("late"));
    f.link->send_latest(LinkMessageType::Samples, bytes("late"));
    f.link->close();
    REQUIRE(f.link->close_reason().rfind("no messages for ", 0) == 0);
}

TEST_CASE("RemoteLink: Link dies when the peer closes", "[remote_link][unit]")
{
    Fixture f;
    f.link->start();
    f.peer.shutdown();
    REQUIRE(wait_until([&] { return !f.link->alive(); }));
    REQUIRE(f.link->close_reason() == "receive failed: connection closed");
}

TEST_CASE("RemoteLink: Link tells a peer that closed before any message from other deaths", "[remote_link][unit]")
{
    SECTION("end of stream first")
    {
        Fixture f;
        f.link->start();
        CHECK_FALSE(f.link->peer_closed_before_any_message());
        f.peer.shutdown();
        REQUIRE(wait_until([&] { return !f.link->alive(); }));
        CHECK(f.link->close_reason() == "receive failed: connection closed");
        CHECK(f.link->peer_closed_before_any_message());
    }
    SECTION("end of stream over a socket")
    {
        auto [robot_end, server_end] = socket_pair();
        Inbox inbox;
        Link link(std::make_unique<TcpTransport>(std::move(robot_end)), Fixture::quiet_options(), inbox.handler());
        link.send_reliable(LinkMessageType::Hello, bytes("{}"));
        link.start();
        server_end = TcpSocket();
        REQUIRE(wait_until([&] { return !link.alive(); }));
        CHECK(link.peer_closed_before_any_message());
    }
    SECTION("a failed send first")
    {
        //! Every send fails; nothing ever arrives.
        class SendFails final : public ITransport
        {
        public:
            bool send(const uint8_t*, size_t) override
            {
                return false;
            }
            ReceiveResult receive(std::vector<uint8_t>&, std::chrono::milliseconds timeout) override
            {
                std::this_thread::sleep_for(std::min(timeout, std::chrono::milliseconds{ 10 }));
                return ReceiveResult::Timeout;
            }
            void shutdown() override
            {
            }
        };
        Inbox inbox;
        Link link(std::make_unique<SendFails>(), Fixture::quiet_options(), inbox.handler());
        link.send_reliable(LinkMessageType::Hello, bytes("{}"));
        link.start();
        REQUIRE(wait_until([&] { return !link.alive(); }));
        CHECK(link.close_reason() == "send failed");
        CHECK(link.peer_closed_before_any_message());
    }
    SECTION("a failed send before the receiver read part of a message")
    {
        //! Every send fails; the receiver finds part of a message a little later.
        class LateTruncation final : public ITransport
        {
        public:
            bool send(const uint8_t*, size_t) override
            {
                return false;
            }
            ReceiveResult receive(std::vector<uint8_t>&, std::chrono::milliseconds) override
            {
                std::this_thread::sleep_for(std::chrono::milliseconds{ 50 });
                return ReceiveResult::Truncated;
            }
            void shutdown() override
            {
            }
        };
        Inbox inbox;
        Link link(std::make_unique<LateTruncation>(), Fixture::quiet_options(), inbox.handler());
        link.send_reliable(LinkMessageType::Hello, bytes("{}"));
        link.start();
        REQUIRE(wait_until([&] { return !link.alive(); }));
        CHECK(link.close_reason() == "receive failed: incomplete message");
        CHECK_FALSE(link.peer_closed_before_any_message());
    }
    SECTION("a reset before the first send")
    {
        // The first send fails; the bytes that arrived before the reset decide.
        for (const size_t sent : { size_t{ 0 }, size_t{ 4 } })
        {
            for (int attempt = 0; attempt < 10; ++attempt)
            {
                CAPTURE(sent, attempt);
                Inbox inbox;
                Link link(reset_after(sent), Fixture::quiet_options(), inbox.handler());
                link.send_reliable(LinkMessageType::Hello, bytes("{}"));
                link.start();
                REQUIRE(wait_until([&] { return !link.alive(); }));
                if (sent == 0)
                {
                    CHECK(link.peer_closed_before_any_message());
                }
                else
                {
                    CHECK(link.close_reason() == "receive failed: incomplete message");
                    CHECK_FALSE(link.peer_closed_before_any_message());
                }
            }
        }
    }
    SECTION("end of stream after a message")
    {
        Fixture f;
        f.link->start();
        f.peer.send(LinkMessageType::Heartbeat);
        REQUIRE(wait_until([&] { return f.link->stats().messages_received == 1; }));
        f.peer.shutdown();
        REQUIRE(wait_until([&] { return !f.link->alive(); }));
        CHECK(f.link->close_reason() == "receive failed: connection closed");
        CHECK_FALSE(f.link->peer_closed_before_any_message());
    }
    SECTION("part of a message, then end of stream")
    {
        // Part of the header, then the header and part of the payload.
        for (const size_t sent : { size_t{ 4 }, kHeaderSize + 10 })
        {
            CAPTURE(sent);
            auto [robot_end, server_end] = socket_pair();
            Inbox inbox;
            Link link(std::make_unique<TcpTransport>(std::move(robot_end)), Fixture::quiet_options(), inbox.handler());
            link.start();
            const std::vector<uint8_t> message = message_of_size(kHeaderSize + 100);
            REQUIRE(server_end.send_all(message.data(), sent));
            server_end.shutdown();
            REQUIRE(wait_until([&] { return !link.alive(); }));
            CHECK(link.close_reason() == "receive failed: incomplete message");
            CHECK_FALSE(link.peer_closed_before_any_message());
        }
    }
    SECTION("malformed data first")
    {
        auto [robot_end, server_end] = socket_pair();
        Inbox inbox;
        Link link(std::make_unique<TcpTransport>(std::move(robot_end)), Fixture::quiet_options(), inbox.handler());
        link.start();
        const std::string answer = "HTTP/1.1 400 Bad Request\r\n\r\n"; // a WebSocket server's answer to tcp://
        REQUIRE(server_end.send_all(reinterpret_cast<const uint8_t*>(answer.data()), answer.size()));
        REQUIRE(wait_until([&] { return !link.alive(); }));
        CHECK(link.close_reason() == "receive failed: malformed data");
        CHECK_FALSE(link.peer_closed_before_any_message());
    }
    SECTION("close()")
    {
        Fixture f;
        f.link->start();
        f.link->close();
        CHECK(f.link->close_reason() == "closed");
        CHECK_FALSE(f.link->peer_closed_before_any_message());
    }
}

TEST_CASE("RemoteLink: Link answers PING with PONG and passes neither to the handler", "[remote_link][unit]")
{
    Fixture f;
    f.link->start();
    f.peer.send(LinkMessageType::Ping);
    const auto pong = f.peer.next(LinkMessageType::Pong);
    REQUIRE(pong);
    const PongPayload payload = decode_pong(pong->payload);
    REQUIRE(payload.t1_ns > 0);
    REQUIRE(payload.t2_ns >= payload.t1_ns);
    REQUIRE(pong->header.t_send_ns >= payload.t2_ns);
    f.peer.send(LinkMessageType::Pong, encode_pong(PongPayload{ 1, 2 })); // unsolicited: ignored
    f.peer.send(LinkMessageType::Heartbeat);
    f.peer.send(LinkMessageType::Status, bytes("after"));
    const auto status = f.inbox.wait();
    REQUIRE(status);
    REQUIRE(status->type == LinkMessageType::Status);
    REQUIRE(f.inbox.size() == 0);
    REQUIRE_FALSE(f.link->clock().two_way);
}

TEST_CASE("RemoteLink: PING/PONG between two Links yields a two-way clock estimate", "[remote_link][unit]")
{
    auto pipe = std::make_shared<Pipe>();
    LinkOptions robot_options = Fixture::quiet_options();
    robot_options.send_pings = true;
    Inbox robot_inbox;
    Inbox server_inbox;
    Link robot(std::make_unique<BlockingTransport>(pipe, 0), robot_options, robot_inbox.handler());
    Link server(std::make_unique<BlockingTransport>(pipe, 1), Fixture::quiet_options(), server_inbox.handler());
    REQUIRE_FALSE(robot.clock().valid);
    REQUIRE(robot.clock().rtt_min_ns == -1);
    server.start();
    robot.start();

    REQUIRE(wait_until([&] { return robot.clock().two_way; }));
    const ClockEstimate estimate = robot.clock();
    REQUIRE(estimate.valid);
    REQUIRE(estimate.rtt_min_ns >= 0);
    // Same process, same clock: the offset is bounded by half the round trip.
    REQUIRE(std::abs(estimate.offset_ns) <= estimate.rtt_min_ns / 2 + 1);
    // The server answered PINGs but sent none, so it only has the one-way estimate.
    REQUIRE(wait_until([&] { return server.clock().valid; }));
    REQUIRE_FALSE(server.clock().two_way);
    REQUIRE(robot_inbox.size() == 0);
    REQUIRE(server_inbox.size() == 0);

    server.send_reliable(LinkMessageType::Ready, bytes("{}"));
    const auto ready = robot_inbox.wait();
    REQUIRE(ready);
    REQUIRE(ready->type == LinkMessageType::Ready);
    REQUIRE(ready->local_receive_ns >= ready->remote_send_ns);
    robot.close();
    server.close();
}

TEST_CASE("RemoteLink: Link passes unknown message types to the handler", "[remote_link][unit]")
{
    Fixture f;
    f.link->start();
    const int64_t before = os_monotonic_now_ns();
    f.peer.send(uint8_t{ 200 }, bytes("future"));
    const auto message = f.inbox.wait();
    REQUIRE(message);
    REQUIRE(static_cast<uint8_t>(message->type) == 200);
    REQUIRE(text(message->payload) == "future");
    REQUIRE(message->remote_send_ns >= before);
    REQUIRE(message->local_receive_ns >= message->remote_send_ns);
    REQUIRE(f.link->stats().unknown_received == 1);
    REQUIRE(f.link->alive());
    REQUIRE(f.link->clock().valid);
}

TEST_CASE("RemoteLink: Link dies on a malformed message or a throwing handler", "[remote_link][unit]")
{
    SECTION("malformed")
    {
        Fixture f;
        f.link->start();
        std::vector<uint8_t> message = encode_link_message(32, bytes("x"));
        message.push_back(0); // longer than its header says
        auto transport = std::make_shared<BlockingTransport>(f.pipe, 1);
        REQUIRE(transport->send(message.data(), message.size()));
        REQUIRE(wait_until([&] { return !f.link->alive(); }));
        REQUIRE(f.link->close_reason().rfind("receive failed: ", 0) == 0);
    }
    SECTION("handler throws")
    {
        auto pipe = std::make_shared<Pipe>();
        Peer peer(pipe);
        Link link(std::make_unique<BlockingTransport>(pipe, 0), Fixture::quiet_options(),
                  [](ReceivedLinkMessage&&) { throw std::runtime_error("boom"); });
        link.start();
        peer.send(LinkMessageType::Status);
        REQUIRE(wait_until([&] { return !link.alive(); }));
        REQUIRE(link.close_reason() == "receive failed: boom");
    }
}

TEST_CASE("RemoteLink: Link close drains queued reliable messages", "[remote_link][unit]")
{
    Fixture f;
    f.link->start();
    f.peer.hold_link(true);
    f.link->send_reliable(LinkMessageType::Status, bytes("1"));
    f.link->send_reliable(LinkMessageType::Status, bytes("2"));
    f.link->send_latest(LinkMessageType::Samples, bytes("dropped on close"));
    f.link->send_reliable(LinkMessageType::Error, bytes("bye"));
    REQUIRE(wait_until([&] { return f.peer.link_send_blocked(); }));
    // Releases the transport only once close() has started: from then on send_latest() is ignored, so re-queueing
    // the held SAMPLES stops counting as a replacement.
    std::atomic<bool> close_started{ false };
    std::thread release(
        [&]
        {
            close_started = wait_until(
                [&]
                {
                    const uint64_t replaced = f.link->stats().latest_replaced;
                    f.link->send_latest(LinkMessageType::Samples, bytes("dropped on close"));
                    return f.link->stats().latest_replaced == replaced;
                });
            f.peer.hold_link(false);
        });
    f.link->close();
    release.join();
    REQUIRE(close_started);
    REQUIRE_FALSE(f.link->alive());
    REQUIRE(f.link->close_reason() == "closed");

    std::vector<std::string> received;
    while (const auto message = f.peer.next(std::nullopt, std::chrono::milliseconds{ 200 }))
    {
        REQUIRE(message->header.type != static_cast<uint8_t>(LinkMessageType::Samples));
        received.push_back(text(message->payload));
    }
    REQUIRE(received == std::vector<std::string>{ "1", "2", "bye" });
    f.link->close(); // idempotent
}

TEST_CASE("RemoteLink: Link close right after start still sends what was queued", "[remote_link][unit]")
{
    // The receiver may not have run the transport handshake yet when close() starts.
    for (int i = 0; i < 50; ++i)
    {
        Fixture f;
        f.link->send_reliable(LinkMessageType::Error, bytes("bye"));
        f.link->start();
        f.link->close();
        const auto error = f.peer.next(LinkMessageType::Error, std::chrono::milliseconds{ 500 });
        REQUIRE(error);
        REQUIRE(text(error->payload) == "bye");
    }
}

TEST_CASE("RemoteLink: Link close gives up after close_drain on a stuck transport", "[remote_link][unit]")
{
    LinkOptions options = Fixture::quiet_options();
    options.close_drain = std::chrono::milliseconds{ 100 };
    Fixture f(options);
    f.link->start();
    f.peer.hold_link(true);
    f.link->send_reliable(LinkMessageType::Status, bytes("stuck"));
    REQUIRE(wait_until([&] { return f.peer.link_send_blocked(); }));
    const auto start = std::chrono::steady_clock::now();
    f.link->close();
    REQUIRE(std::chrono::steady_clock::now() - start < std::chrono::milliseconds{ 1'500 });
    REQUIRE_FALSE(f.link->alive());
}

TEST_CASE("RemoteLink: Link close without start and invalid use", "[remote_link][unit]")
{
    Fixture f;
    f.link->close();
    REQUIRE_FALSE(f.link->alive());
    REQUIRE_THROWS_AS(f.link->start(), std::logic_error);

    Fixture g;
    g.link->start();
    REQUIRE_THROWS_AS(g.link->start(), std::logic_error);
    REQUIRE_THROWS_AS(g.link->send_reliable(LinkMessageType::Samples, std::vector<uint8_t>(kMaxPayloadSize + 1)),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(Link(nullptr, LinkOptions{}, {}), std::invalid_argument);
}

TEST_CASE("RemoteLink: Link over TCP and WebSocket loopback", "[remote_link][unit]")
{
    const bool websocket = GENERATE(false, true);
    Listener listener(Endpoint{ "127.0.0.1", 0 });
    const std::string scheme = websocket ? "ws://" : "tcp://";
    const RemoteAddress remote = parse_remote(scheme + "127.0.0.1:" + std::to_string(listener.port()));

    std::unique_ptr<ITransport> client_transport;
    std::thread connector([&] { client_transport = connect_transport(remote, kWait); });
    std::optional<Listener::Accepted> accepted = listener.accept(kWait);
    REQUIRE(accepted.has_value());
    // The server Link runs the upgrade itself (receiver thread) and holds its sends until it is done.
    Inbox server_inbox;
    Link server(adopt_transport(accepted->fd, websocket), Fixture::quiet_options(), server_inbox.handler());
    server.send_reliable(LinkMessageType::HelloAck, bytes("queued before the upgrade"));
    server.start();
    connector.join();
    REQUIRE(client_transport);

    LinkOptions robot_options = Fixture::quiet_options();
    robot_options.send_pings = true;
    Inbox robot_inbox;
    Link robot(std::move(client_transport), robot_options, robot_inbox.handler());
    robot.start();
    robot.send_reliable(LinkMessageType::Hello, bytes("{\"protocol\": 1}"));

    const auto hello = server_inbox.wait();
    REQUIRE(hello);
    REQUIRE(hello->type == LinkMessageType::Hello);
    REQUIRE(text(hello->payload) == "{\"protocol\": 1}");
    const auto ack = robot_inbox.wait();
    REQUIRE(ack);
    REQUIRE(ack->type == LinkMessageType::HelloAck);

    std::vector<uint8_t> big(1 << 20);
    for (size_t i = 0; i < big.size(); ++i)
    {
        big[i] = static_cast<uint8_t>(i * 13);
    }
    server.send_latest(LinkMessageType::Samples, big);
    const auto samples = robot_inbox.wait();
    REQUIRE(samples);
    REQUIRE(samples->type == LinkMessageType::Samples);
    REQUIRE(samples->payload == big);
    REQUIRE(wait_until([&] { return robot.clock().two_way; }));

    server.send_reliable(LinkMessageType::Error, bytes("server stopping"));
    server.close();
    const auto error = robot_inbox.wait();
    REQUIRE(error);
    REQUIRE(error->type == LinkMessageType::Error);
    REQUIRE(text(error->payload) == "server stopping");
    REQUIRE(wait_until([&] { return !robot.alive(); }));
    REQUIRE_FALSE(robot.close_reason().empty());
}

TEST_CASE("RemoteLink: Link never sends a latest-wins payload ahead of a reliable message queued before it",
          "[remote_link][unit]")
{
    Fixture f;
    f.link->start();
    constexpr int kCount = 2000;
    const MergeFn keep_newer = [](std::vector<uint8_t>, std::vector<uint8_t> incoming) { return incoming; };
    // As a host queues SETUP for new channels, then the SAMPLES that use them.
    std::thread producer(
        [&]
        {
            for (int i = 0; i < kCount; ++i)
            {
                f.link->send_reliable(LinkMessageType::Setup, bytes(std::to_string(i)));
                f.link->send_latest(LinkMessageType::Samples, bytes(std::to_string(i)), keep_newer);
            }
        });
    int last_setup = -1;
    int last_samples = -1;
    int overtaken = 0;
    while (last_samples < kCount - 1)
    {
        const auto message = f.peer.next(std::nullopt);
        if (!message)
        {
            break;
        }
        const int index = std::stoi(text(message->payload));
        if (message->header.type == static_cast<uint8_t>(LinkMessageType::Setup))
        {
            last_setup = index;
        }
        else if (message->header.type == static_cast<uint8_t>(LinkMessageType::Samples))
        {
            overtaken += index > last_setup ? 1 : 0;
            last_samples = index;
        }
    }
    producer.join();
    REQUIRE(last_samples == kCount - 1);
    REQUIRE(last_setup == kCount - 1);
    REQUIRE(overtaken == 0);
}

TEST_CASE("RemoteLink: Link close_gracefully half-closes, then discards input until the peer closes",
          "[remote_link][unit]")
{
    Fixture f;
    f.link->start();
    f.peer.send(LinkMessageType::Status, bytes("before"));
    REQUIRE(f.inbox.wait());
    f.link->send_reliable(LinkMessageType::Error, bytes("bye"));

    const auto start = std::chrono::steady_clock::now();
    // jthread: joined even when a REQUIRE below throws.
    std::jthread closer([&] { f.link->close_gracefully(std::chrono::milliseconds{ 10'000 }); });
    std::vector<uint8_t> raw;
    std::vector<uint8_t> types;
    ReceiveResult result;
    while ((result = f.peer.receive(raw)) == ReceiveResult::Ok)
    {
        types.push_back(decode_header(raw.data()).type);
        if (types.back() == static_cast<uint8_t>(LinkMessageType::Error))
        {
            REQUIRE(text(std::vector<uint8_t>(raw.begin() + kHeaderSize, raw.end())) == "bye");
        }
    }
    REQUIRE(result == ReceiveResult::Closed);
    REQUIRE(types == std::vector<uint8_t>{ static_cast<uint8_t>(LinkMessageType::Error) });

    // Input while lingering is read but not handed over.
    const uint64_t received = f.link->stats().messages_received;
    f.peer.send(LinkMessageType::Status, bytes("late"));
    REQUIRE(wait_until([&] { return f.link->stats().messages_received == received + 1; }));
    REQUIRE(f.link->alive());

    f.peer.shutdown();
    closer.join();
    REQUIRE(std::chrono::steady_clock::now() - start < std::chrono::milliseconds{ 5'000 });
    REQUIRE_FALSE(f.link->alive());
    REQUIRE(f.link->close_reason() == "closed");
    REQUIRE(f.inbox.size() == 0);
    f.link->close_gracefully(std::chrono::milliseconds{ 10'000 }); // idempotent
}

TEST_CASE("RemoteLink: Link close_gracefully gives up after the linger", "[remote_link][unit]")
{
    Fixture f;
    f.link->start();
    const auto start = std::chrono::steady_clock::now();
    f.link->close_gracefully(std::chrono::milliseconds{ 100 });
    const auto elapsed = std::chrono::steady_clock::now() - start;
    REQUIRE(elapsed >= std::chrono::milliseconds{ 100 });
    REQUIRE(elapsed < std::chrono::milliseconds{ 5'000 });
    REQUIRE(f.link->close_reason() == "closed");
    std::vector<uint8_t> raw;
    REQUIRE(f.peer.receive(raw) == ReceiveResult::Closed);
}

TEST_CASE("RemoteLink: Link close_gracefully without start or after death closes at once", "[remote_link][unit]")
{
    SECTION("never started")
    {
        Fixture f;
        const auto start = std::chrono::steady_clock::now();
        f.link->close_gracefully(std::chrono::milliseconds{ 10'000 });
        REQUIRE(std::chrono::steady_clock::now() - start < std::chrono::milliseconds{ 5'000 });
        REQUIRE(f.link->close_reason() == "closed");
    }
    SECTION("already dead")
    {
        Fixture f;
        f.link->start();
        f.peer.shutdown();
        REQUIRE(wait_until([&] { return !f.link->alive(); }));
        const auto start = std::chrono::steady_clock::now();
        f.link->close_gracefully(std::chrono::milliseconds{ 10'000 });
        REQUIRE(std::chrono::steady_clock::now() - start < std::chrono::milliseconds{ 5'000 });
        REQUIRE(f.link->close_reason() == "receive failed: connection closed");
    }
}

TEST_CASE("RemoteLink: Link close_gracefully over TCP and WebSocket loopback", "[remote_link][unit]")
{
    const bool websocket = GENERATE(false, true);
    Listener listener(Endpoint{ "127.0.0.1", 0 });
    const std::string scheme = websocket ? "ws://" : "tcp://";
    const RemoteAddress remote = parse_remote(scheme + "127.0.0.1:" + std::to_string(listener.port()));

    std::unique_ptr<ITransport> client_transport;
    std::thread connector([&] { client_transport = connect_transport(remote, kWait); });
    std::optional<Listener::Accepted> accepted = listener.accept(kWait);
    REQUIRE(accepted.has_value());
    Inbox server_inbox;
    Link server(adopt_transport(accepted->fd, websocket), Fixture::quiet_options(), server_inbox.handler());
    server.start();
    connector.join();
    REQUIRE(client_transport);

    LinkOptions robot_options = Fixture::quiet_options();
    robot_options.send_pings = true;
    Inbox robot_inbox;
    Link robot(std::move(client_transport), robot_options, robot_inbox.handler());
    robot.start();
    robot.send_reliable(LinkMessageType::Hello, bytes("{}"));
    REQUIRE(server_inbox.wait());

    server.send_reliable(LinkMessageType::Error, bytes("server stopping"));
    const auto start = std::chrono::steady_clock::now();
    server.close_gracefully(std::chrono::milliseconds{ 10'000 });
    // The robot saw end of stream and closed its end, which ended the linger.
    REQUIRE(std::chrono::steady_clock::now() - start < std::chrono::milliseconds{ 5'000 });
    REQUIRE(server.close_reason() == "closed");

    const auto error = robot_inbox.wait();
    REQUIRE(error);
    REQUIRE(error->type == LinkMessageType::Error);
    REQUIRE(text(error->payload) == "server stopping");
    REQUIRE(wait_until([&] { return !robot.alive(); }));
    REQUIRE(robot.close_reason() == "receive failed: connection closed");
}

TEST_CASE("RemoteLink: close_gracefully delivers a send backlog despite unread input", "[remote_link][unit]")
{
    Listener listener(Endpoint{ "127.0.0.1", 0 });
    TcpTransport client(small_receive_buffer_client(listener.port()));
    std::optional<Listener::Accepted> accepted = listener.accept(kWait);
    REQUIRE(accepted.has_value());

    // The handler parks the receive thread on the first message, so what follows stays unread in the socket.
    std::mutex mutex;
    std::condition_variable cv;
    bool released = false;
    std::atomic<int> handled{ 0 };
    const auto release = [&]
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            released = true;
        }
        cv.notify_all();
    };
    Link server(adopt_transport(accepted->fd, false), Fixture::quiet_options(),
                [&](ReceivedLinkMessage&&)
                {
                    ++handled;
                    std::unique_lock<std::mutex> lock(mutex);
                    cv.wait(lock, [&] { return released; });
                });
    struct ReleaseOnExit
    {
        std::function<void()> release;
        ~ReleaseOnExit()
        {
            release();
        }
    } release_on_exit{ release };
    server.start();

    const std::vector<uint8_t> filler =
        encode_link_message(static_cast<uint8_t>(LinkMessageType::Status), std::vector<uint8_t>(1024, 7));
    for (int i = 0; i < 8; ++i)
    {
        REQUIRE(client.send(filler.data(), filler.size()));
    }
    REQUIRE(wait_until([&] { return handled == 1; }));

    // More than the robot's receive buffer holds, so the ERROR behind it is still in the server's send queue at
    // close; less than TCP_NOTSENT_LOWAT, so the sends complete.
    const std::vector<uint8_t> backlog(48 * 1024, 9);
    server.send_reliable(LinkMessageType::Status, backlog);
    server.send_reliable(LinkMessageType::Error, bytes("server stopping"));
    const auto start = std::chrono::steady_clock::now();
    std::jthread closer([&] { server.close_gracefully(std::chrono::milliseconds{ 10'000 }); });
    std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });
    release();
    // The robot reads late, after the server's close has run.
    std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });

    std::vector<uint8_t> message;
    REQUIRE(client.receive(message, kWait) == ReceiveResult::Ok);
    REQUIRE(message.size() == kHeaderSize + backlog.size());
    REQUIRE(client.receive(message, kWait) == ReceiveResult::Ok);
    REQUIRE(decode_header(message.data()).type == static_cast<uint8_t>(LinkMessageType::Error));
    REQUIRE(text(std::vector<uint8_t>(message.begin() + kHeaderSize, message.end())) == "server stopping");
    REQUIRE(client.receive(message, kWait) == ReceiveResult::Closed);
    client.shutdown();
    closer.join();
    REQUIRE(std::chrono::steady_clock::now() - start < std::chrono::milliseconds{ 5'000 });
    REQUIRE(server.close_reason() == "closed");
}

TEST_CASE("RemoteLink: Link with peer_timeout_from_first_message waits for the first message without limit",
          "[remote_link][unit]")
{
    LinkOptions options = Fixture::quiet_options();
    options.peer_timeout = std::chrono::milliseconds{ 100 };
    options.peer_timeout_from_first_message = true;
    Fixture f(options);
    f.link->start();
    std::this_thread::sleep_for(std::chrono::milliseconds{ 400 });
    REQUIRE(f.link->alive());

    // From the first message on, the peer timeout applies.
    f.peer.send(LinkMessageType::Heartbeat);
    const auto quiet_since = std::chrono::steady_clock::now();
    REQUIRE(wait_until([&] { return !f.link->alive(); }));
    REQUIRE(std::chrono::steady_clock::now() - quiet_since >= options.peer_timeout);
    REQUIRE(f.link->close_reason().rfind("no messages for ", 0) == 0);
}

TEST_CASE("RemoteLink: Link with peer_timeout_from_first_message pings once the peer has spoken", "[remote_link][unit]")
{
    LinkOptions options = Fixture::quiet_options();
    options.send_pings = true;
    options.peer_timeout_from_first_message = true;
    Fixture f(options);
    f.link->start();
    REQUIRE_FALSE(f.peer.next(LinkMessageType::Ping, std::chrono::milliseconds{ 300 }));

    // The quiet options' 60 s heartbeat would hold the first PING back if the sender were not woken.
    f.peer.send(LinkMessageType::Heartbeat);
    const auto ping = f.peer.next(LinkMessageType::Ping, std::chrono::milliseconds{ 500 });
    REQUIRE(ping);
    f.peer.send(LinkMessageType::Pong, encode_pong(PongPayload{ ping->header.t_send_ns, os_monotonic_now_ns() }));
    REQUIRE(wait_until([&] { return f.link->clock().two_way; }));
}

TEST_CASE("RemoteLink: Link with peer_timeout_from_first_message waits for the transport handshake without limit",
          "[remote_link][unit]")
{
    auto [client_end, server_end] = socket_pair();
    const RemoteAddress remote{ Endpoint{ "127.0.0.1", 1 }, true, "/" };
    LinkOptions options = Fixture::quiet_options();
    options.peer_timeout = std::chrono::milliseconds{ 100 };
    Inbox inbox;
    SECTION("with the option, the link waits for the upgrade")
    {
        options.peer_timeout_from_first_message = true;
        Link link(WebSocketTransport::start_client(std::move(client_end), remote), options, inbox.handler());
        link.send_reliable(LinkMessageType::Hello, bytes("hello"));
        link.start();
        std::this_thread::sleep_for(std::chrono::milliseconds{ 400 });
        REQUIRE(link.alive());

        WebSocketTransport server(std::move(server_end), WebSocketTransport::Role::Server);
        REQUIRE(server.wait_ready(kWait) == ReceiveResult::Ok);
        std::vector<uint8_t> message;
        REQUIRE(server.receive(message, kWait) == ReceiveResult::Ok);
        REQUIRE(decode_header(message.data()).type == static_cast<uint8_t>(LinkMessageType::Hello));
        std::vector<uint8_t> ready = encode_link_message(static_cast<uint8_t>(LinkMessageType::Ready), bytes("{}"));
        stamp_header(ready.data(), 0, os_monotonic_now_ns());
        REQUIRE(server.send(ready.data(), ready.size()));
        const auto received = inbox.wait();
        REQUIRE(received);
        REQUIRE(received->type == LinkMessageType::Ready);
    }
    SECTION("without it, the handshake fails after the peer timeout")
    {
        Link link(WebSocketTransport::start_client(std::move(client_end), remote), options, inbox.handler());
        link.start();
        REQUIRE(wait_until([&] { return !link.alive(); }));
        REQUIRE(link.close_reason() == "transport handshake failed: timed out");
    }
}
