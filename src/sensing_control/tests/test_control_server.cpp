// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <arpa/inet.h>
#include <catch2/catch_test_macros.hpp>
#include <netinet/in.h>
#include <sensing_control/control_server.hpp>
#include <sensing_control/websocket.hpp>
#include <sys/socket.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <poll.h>
#include <string>
#include <thread>
#include <unistd.h>

using namespace plugins::sensing;
using namespace std::chrono_literals;
namespace ws = plugins::sensing::websocket;

namespace
{

constexpr const char* kToken = "test-token-123";

/// A received server frame (server frames are unmasked).
struct ServerFrame
{
    std::uint8_t opcode = 0;
    std::string payload;
};

/// A bare TCP client that speaks just enough WebSocket to drive the server.
class RawClient
{
public:
    explicit RawClient(std::uint16_t port)
    {
        m_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        REQUIRE(::connect(m_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
    }
    ~RawClient()
    {
        if (m_fd >= 0)
            ::close(m_fd);
    }
    RawClient(const RawClient&) = delete;
    RawClient& operator=(const RawClient&) = delete;

    void send_raw(const std::string& bytes)
    {
        std::size_t done = 0;
        while (done < bytes.size())
        {
            const ssize_t n = ::send(m_fd, bytes.data() + done, bytes.size() - done, MSG_NOSIGNAL);
            REQUIRE(n > 0);
            done += static_cast<std::size_t>(n);
        }
    }

    static std::string upgrade(const std::string& target, const std::string& auth_header)
    {
        return "GET " + target +
               " HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n" +
               auth_header + "\r\n";
    }

    /// Handshake with the right token; returns the response head.
    std::string handshake(const std::string& token = kToken)
    {
        send_raw(upgrade("/", "Authorization: Bearer " + token + "\r\n"));
        return read_head();
    }

    /// Reads up to the end of an HTTP head; for a refusal also reads the body that follows. Empty on EOF/timeout.
    std::string read_head()
    {
        while (m_buffer.find("\r\n\r\n") == std::string::npos)
            if (!fill(3000))
                break;
        const auto end = m_buffer.find("\r\n\r\n");
        if (end == std::string::npos)
            return {};
        std::string head = m_buffer.substr(0, end + 4);
        m_buffer.erase(0, end + 4);
        return head;
    }

    /// Reads whatever remains until the peer closes (or the timeout), for HTTP bodies.
    std::string read_until_eof(int timeout_ms = 3000)
    {
        while (fill(timeout_ms))
        {
        }
        std::string out = std::move(m_buffer);
        m_buffer.clear();
        return out;
    }

    void send_frame(std::uint8_t opcode, const std::string& payload, bool fin = true, bool masked = true)
    {
        const unsigned char mask[4] = { 1, 2, 3, 4 };
        std::string out;
        out.push_back(static_cast<char>((fin ? 0x80 : 0) | opcode));
        const std::uint8_t mask_bit = masked ? 0x80 : 0;
        if (payload.size() < 126)
        {
            out.push_back(static_cast<char>(mask_bit | payload.size()));
        }
        else if (payload.size() <= 0xFFFF)
        {
            out.push_back(static_cast<char>(mask_bit | 126));
            out.push_back(static_cast<char>(payload.size() >> 8));
            out.push_back(static_cast<char>(payload.size() & 0xFF));
        }
        else
        {
            out.push_back(static_cast<char>(mask_bit | 127));
            for (int shift = 56; shift >= 0; shift -= 8)
                out.push_back(static_cast<char>((static_cast<std::uint64_t>(payload.size()) >> shift) & 0xFF));
        }
        if (masked)
            out.append(reinterpret_cast<const char*>(mask), 4);
        for (std::size_t i = 0; i < payload.size(); ++i)
            out.push_back(static_cast<char>(masked ? (static_cast<unsigned char>(payload[i]) ^ mask[i % 4]) : payload[i]));
        send_raw(out);
    }

    /// The next server frame, or nullopt on EOF/timeout.
    std::optional<ServerFrame> read_frame(int timeout_ms = 3000)
    {
        while (true)
        {
            if (m_buffer.size() >= 2)
            {
                const auto b = [&](std::size_t i) { return static_cast<unsigned char>(m_buffer[i]); };
                std::uint64_t length = b(1) & 0x7F;
                std::size_t header = 2;
                if (length == 126 && m_buffer.size() >= 4)
                {
                    length = (std::uint64_t(b(2)) << 8) | b(3);
                    header = 4;
                }
                else if (length == 127 && m_buffer.size() >= 10)
                {
                    length = 0;
                    for (int i = 0; i < 8; ++i)
                        length = (length << 8) | b(2 + i);
                    header = 10;
                }
                const bool header_known =
                    !((b(1) & 0x7F) == 126 && m_buffer.size() < 4) && !((b(1) & 0x7F) == 127 && m_buffer.size() < 10);
                if (header_known && m_buffer.size() >= header + length)
                {
                    ServerFrame frame;
                    frame.opcode = b(0) & 0x0F;
                    frame.payload = m_buffer.substr(header, static_cast<std::size_t>(length));
                    m_buffer.erase(0, header + static_cast<std::size_t>(length));
                    return frame;
                }
            }
            if (!fill(timeout_ms))
                return std::nullopt;
        }
    }

    /// The close code of the next frame, which must be a close frame; 0 otherwise.
    std::uint16_t read_close_code(int timeout_ms = 3000)
    {
        const auto frame = read_frame(timeout_ms);
        if (!frame || frame->opcode != ws::kClose || frame->payload.size() < 2)
            return 0;
        return static_cast<std::uint16_t>((static_cast<unsigned char>(frame->payload[0]) << 8) |
                                          static_cast<unsigned char>(frame->payload[1]));
    }

    /// True if the peer closes the TCP connection (EOF or reset) within the timeout.
    bool peer_closes(int timeout_ms = 3000)
    {
        const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        while (std::chrono::steady_clock::now() < until)
        {
            pollfd pfd{ m_fd, POLLIN, 0 };
            if (::poll(&pfd, 1, 100) > 0)
            {
                char buffer[4096];
                const ssize_t n = ::recv(m_fd, buffer, sizeof(buffer), 0);
                if (n <= 0)
                    return true;
            }
        }
        return false;
    }

private:
    bool fill(int timeout_ms)
    {
        pollfd pfd{ m_fd, POLLIN, 0 };
        if (::poll(&pfd, 1, timeout_ms) <= 0)
            return false;
        char buffer[4096];
        const ssize_t n = ::recv(m_fd, buffer, sizeof(buffer), 0);
        if (n <= 0)
            return false;
        m_buffer.append(buffer, static_cast<std::size_t>(n));
        return true;
    }

    int m_fd = -1;
    std::string m_buffer;
};

ControlServerConfig config()
{
    ControlServerConfig c;
    c.bind_address = "127.0.0.1";
    c.port = 0;
    c.token = kToken;
    return c;
}

std::unique_ptr<ControlServer> make_server(ControlServerConfig c = config())
{
    return std::make_unique<ControlServer>(c, [](const std::string& text) { return "echo:" + text; });
}

bool wait_clients(const ControlServer& server, std::size_t count)
{
    const auto until = std::chrono::steady_clock::now() + 3s;
    while (std::chrono::steady_clock::now() < until)
    {
        if (server.client_count() == count)
            return true;
        std::this_thread::sleep_for(5ms);
    }
    return server.client_count() == count;
}

} // namespace

TEST_CASE("an empty token is refused at construction", "[control_server]")
{
    ControlServerConfig c = config();
    c.token.clear();
    CHECK_THROWS_AS(make_server(c), std::runtime_error);
}

TEST_CASE("a bind address that is not IPv4 is refused", "[control_server]")
{
    ControlServerConfig c = config();
    c.bind_address = "not-an-address";
    CHECK_THROWS_AS(make_server(c), std::runtime_error);
}

TEST_CASE("a port already in use is refused with the reason", "[control_server]")
{
    auto first = make_server();
    ControlServerConfig c = config();
    c.port = first->port();
    // SO_REUSEADDR does not allow two listeners on the same port.
    try
    {
        make_server(c);
        FAIL("expected the second bind to fail");
    }
    catch (const std::runtime_error& e)
    {
        CHECK(std::string(e.what()).find("cannot listen") != std::string::npos);
    }
}

TEST_CASE("read_secret_file takes the first line, trimmed", "[control_server]")
{
    const auto dir = std::filesystem::temp_directory_path();
    const auto path = dir / ("sensing_control_secret_" + std::to_string(::getpid()) + ".txt");
    {
        std::ofstream(path) << "  hunter2 \t\r\nsecond line\n";
    }
    CHECK(read_secret_file(path.string()) == "hunter2");
    {
        std::ofstream(path) << "  \n";
    }
    CHECK_THROWS_AS(read_secret_file(path.string()), std::runtime_error);
    std::filesystem::remove(path);
    CHECK_THROWS_AS(read_secret_file(path.string()), std::runtime_error);
}

TEST_CASE("a good handshake is accepted with the RFC accept key", "[control_server]")
{
    auto server = make_server();
    RawClient client(server->port());
    const std::string head = client.handshake();
    CHECK(head.rfind("HTTP/1.1 101", 0) == 0);
    CHECK(head.find("s3pPLMBiTxaQ9kYGzzhZRbK+xOo=") != std::string::npos);
    CHECK(wait_clients(*server, 1));
}

TEST_CASE("the token may be sent as a query parameter", "[control_server]")
{
    auto server = make_server();
    RawClient client(server->port());
    client.send_raw(RawClient::upgrade(std::string("/?token=") + kToken, ""));
    CHECK(client.read_head().rfind("HTTP/1.1 101", 0) == 0);
}

TEST_CASE("a wrong or absent token is 401 and the connection ends", "[control_server]")
{
    auto server = make_server();

    SECTION("wrong token")
    {
        RawClient client(server->port());
        const std::string head = client.handshake("wrong-token");
        CHECK(head.rfind("HTTP/1.1 401", 0) == 0);
        CHECK(client.peer_closes());
    }
    SECTION("same length, one byte off")
    {
        RawClient client(server->port());
        std::string almost = kToken;
        almost.back() ^= 1;
        CHECK(client.handshake(almost).rfind("HTTP/1.1 401", 0) == 0);
    }
    SECTION("no token")
    {
        RawClient client(server->port());
        client.send_raw(RawClient::upgrade("/", ""));
        CHECK(client.read_head().rfind("HTTP/1.1 401", 0) == 0);
    }
    CHECK(server->client_count() == 0);
}

TEST_CASE("a plain HTTP GET is 426", "[control_server]")
{
    auto server = make_server();
    RawClient client(server->port());
    client.send_raw("GET / HTTP/1.1\r\nHost: x\r\n\r\n");
    const std::string head = client.read_head();
    CHECK(head.rfind("HTTP/1.1 426", 0) == 0);
    CHECK(head.find("Upgrade: websocket") != std::string::npos);
    CHECK(client.read_until_eof().find("WebSocket endpoint") != std::string::npos);
}

TEST_CASE("a malformed request line is 400", "[control_server]")
{
    auto server = make_server();
    RawClient client(server->port());
    client.send_raw("POST / HTTP/1.1\r\n\r\n");
    CHECK(client.read_head().rfind("HTTP/1.1 400", 0) == 0);
}

TEST_CASE("an oversized handshake is refused", "[control_server]")
{
    auto server = make_server();
    RawClient client(server->port());
    client.send_raw("GET / HTTP/1.1\r\nX-Pad: " + std::string(16 * 1024, 'a'));
    CHECK(client.read_head().rfind("HTTP/1.1 400", 0) == 0);
}

TEST_CASE("ping frames are answered with pong carrying the payload", "[control_server]")
{
    auto server = make_server();
    RawClient client(server->port());
    REQUIRE(client.handshake().rfind("HTTP/1.1 101", 0) == 0);
    client.send_frame(ws::kPing, "beat");
    const auto frame = client.read_frame();
    REQUIRE(frame);
    CHECK(frame->opcode == ws::kPong);
    CHECK(frame->payload == "beat");
}

TEST_CASE("a text message goes through the handler and the reply comes back", "[control_server]")
{
    auto server = make_server();
    RawClient client(server->port());
    REQUIRE(client.handshake().rfind("HTTP/1.1 101", 0) == 0);

    SECTION("a single frame")
    {
        client.send_frame(ws::kText, "hello");
        const auto frame = client.read_frame();
        REQUIRE(frame);
        CHECK(frame->opcode == ws::kText);
        CHECK(frame->payload == "echo:hello");
    }
    SECTION("fragments are reassembled")
    {
        client.send_frame(ws::kText, "he", /*fin=*/false);
        client.send_frame(ws::kContinuation, "ll", /*fin=*/false);
        client.send_frame(ws::kContinuation, "o", /*fin=*/true);
        const auto frame = client.read_frame();
        REQUIRE(frame);
        CHECK(frame->payload == "echo:hello");
    }
    SECTION("a message that needs a 16-bit length is served")
    {
        const std::string big(5000, 'z');
        client.send_frame(ws::kText, big);
        const auto frame = client.read_frame();
        REQUIRE(frame);
        CHECK(frame->payload == "echo:" + big);
    }
    SECTION("several messages in order")
    {
        for (int i = 0; i < 5; ++i)
            client.send_frame(ws::kText, std::to_string(i));
        for (int i = 0; i < 5; ++i)
        {
            const auto frame = client.read_frame();
            REQUIRE(frame);
            CHECK(frame->payload == "echo:" + std::to_string(i));
        }
    }
    SECTION("a client close frame is answered with close 1000 and the connection ends")
    {
        client.send_frame(ws::kClose, std::string("\x03\xe8", 2));
        CHECK(client.read_close_code() == 1000);
        CHECK(client.peer_closes());
        CHECK(wait_clients(*server, 0));
    }
}

TEST_CASE("broadcast reaches every connected client", "[control_server]")
{
    auto server = make_server();
    RawClient a(server->port());
    RawClient b(server->port());
    REQUIRE(a.handshake().rfind("HTTP/1.1 101", 0) == 0);
    REQUIRE(b.handshake().rfind("HTTP/1.1 101", 0) == 0);
    REQUIRE(wait_clients(*server, 2));

    server->broadcast("{\"event\":1}");
    server->broadcast("{\"event\":2}");
    for (RawClient* client : { &a, &b })
    {
        const auto first = client->read_frame();
        const auto second = client->read_frame();
        REQUIRE(first);
        REQUIRE(second);
        CHECK(first->opcode == ws::kText);
        CHECK(first->payload == "{\"event\":1}");
        CHECK(second->payload == "{\"event\":2}");
    }
}

TEST_CASE("a client that has not handshaken does not receive broadcasts", "[control_server]")
{
    auto server = make_server();
    RawClient pending(server->port());
    RawClient open(server->port());
    REQUIRE(open.handshake().rfind("HTTP/1.1 101", 0) == 0);
    server->broadcast("x");
    REQUIRE(open.read_frame());
    CHECK_FALSE(pending.read_frame(300));
}

TEST_CASE("an oversized message closes the connection with 1009", "[control_server]")
{
    ControlServerConfig c = config();
    c.max_message_bytes = 256;
    auto server = make_server(c);

    SECTION("one frame")
    {
        RawClient client(server->port());
        REQUIRE(client.handshake().rfind("HTTP/1.1 101", 0) == 0);
        client.send_frame(ws::kText, std::string(1000, 'a'));
        CHECK(client.read_close_code() == 1009);
        CHECK(client.peer_closes());
    }
    SECTION("fragments that add up to too much")
    {
        RawClient client(server->port());
        REQUIRE(client.handshake().rfind("HTTP/1.1 101", 0) == 0);
        client.send_frame(ws::kText, std::string(200, 'a'), /*fin=*/false);
        client.send_frame(ws::kContinuation, std::string(200, 'a'), /*fin=*/false);
        CHECK(client.read_close_code() == 1009);
    }
    SECTION("a message exactly at the limit is served")
    {
        RawClient client(server->port());
        REQUIRE(client.handshake().rfind("HTTP/1.1 101", 0) == 0);
        client.send_frame(ws::kText, std::string(256, 'a'));
        const auto frame = client.read_frame();
        REQUIRE(frame);
        CHECK(frame->opcode == ws::kText);
    }
}

TEST_CASE("a binary frame closes the connection with 1003", "[control_server]")
{
    auto server = make_server();
    RawClient client(server->port());
    REQUIRE(client.handshake().rfind("HTTP/1.1 101", 0) == 0);
    client.send_frame(ws::kBinary, "\x01\x02");
    CHECK(client.read_close_code() == 1003);
    CHECK(client.peer_closes());
}

TEST_CASE("an unmasked client frame closes the connection with 1002", "[control_server]")
{
    auto server = make_server();
    RawClient client(server->port());
    REQUIRE(client.handshake().rfind("HTTP/1.1 101", 0) == 0);
    client.send_frame(ws::kText, "hi", true, /*masked=*/false);
    CHECK(client.read_close_code() == 1002);
    CHECK(client.peer_closes());
    CHECK(wait_clients(*server, 0));
}

TEST_CASE("max_clients is enforced and a slot is reusable", "[control_server]")
{
    ControlServerConfig c = config();
    c.max_clients = 2;
    auto server = make_server(c);

    auto first = std::make_unique<RawClient>(server->port());
    RawClient second(server->port());
    REQUIRE(first->handshake().rfind("HTTP/1.1 101", 0) == 0);
    REQUIRE(second.handshake().rfind("HTTP/1.1 101", 0) == 0);
    REQUIRE(wait_clients(*server, 2));

    {
        RawClient third(server->port());
        third.send_raw(RawClient::upgrade("/", std::string("Authorization: Bearer ") + kToken + "\r\n"));
        CHECK(third.peer_closes());
    }
    CHECK(server->client_count() == 2);

    first.reset(); // a departing client frees a slot
    REQUIRE(wait_clients(*server, 1));
    RawClient fourth(server->port());
    CHECK(fourth.handshake().rfind("HTTP/1.1 101", 0) == 0);
}

TEST_CASE("destroying the server is bounded with idle and half-open clients", "[control_server]")
{
    auto server = make_server();
    RawClient idle(server->port());
    REQUIRE(idle.handshake().rfind("HTTP/1.1 101", 0) == 0);
    RawClient half_open(server->port()); // connected, never sends the request
    RawClient partial(server->port());
    partial.send_raw("GET / HTTP/1.1\r\nHost: x\r\n"); // an unfinished head
    REQUIRE(wait_clients(*server, 1));
    std::this_thread::sleep_for(100ms);

    const auto start = std::chrono::steady_clock::now();
    server.reset();
    CHECK(std::chrono::steady_clock::now() - start < 2s);
}

TEST_CASE("destroying an idle server is prompt", "[control_server]")
{
    auto server = make_server();
    const auto start = std::chrono::steady_clock::now();
    server.reset();
    CHECK(std::chrono::steady_clock::now() - start < 1s);
}

TEST_CASE("a client that never reads is cut off instead of buffered for", "[control_server]")
{
    auto server = make_server();
    RawClient client(server->port());
    REQUIRE(client.handshake().rfind("HTTP/1.1 101", 0) == 0);
    REQUIRE(wait_clients(*server, 1));

    // Far more than the 1 MiB output cap plus the socket buffers; the client reads none of it.
    const std::string chunk(32 * 1024, 'x');
    for (int i = 0; i < 600; ++i)
    {
        server->broadcast(chunk);
        if (i % 16 == 0)
            std::this_thread::sleep_for(10ms);
    }
    CHECK(wait_clients(*server, 0));
}
