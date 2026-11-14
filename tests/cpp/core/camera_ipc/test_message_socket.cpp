// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <camera_ipc/host_pool.hpp>
#include <camera_ipc/message_socket.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sys/socket.h>
#include <sys/un.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace core::camera;
using namespace std::chrono_literals;

namespace
{

// A directory for socket files that is removed with everything in it.
class TemporaryDirectory
{
public:
    TemporaryDirectory()
    {
        std::string pattern = (std::filesystem::temp_directory_path() / "camera_ipc_XXXXXX").string();
        REQUIRE(::mkdtemp(pattern.data()) != nullptr);
        m_path = pattern;
    }

    ~TemporaryDirectory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(m_path, ignored);
    }

    std::string socket_path() const
    {
        return (m_path / "endpoint.sock").string();
    }

private:
    std::filesystem::path m_path;
};

std::span<const std::byte> bytes_of(std::string_view text)
{
    return std::as_bytes(std::span(text.data(), text.size()));
}

std::string text_of(std::span<const std::byte> bytes, std::size_t count)
{
    return { reinterpret_cast<const char*>(bytes.data()), count };
}

struct Connection
{
    TemporaryDirectory directory;
    MessageListener listener{ directory.socket_path() };
    MessageSocket client = MessageSocket::connect(directory.socket_path());
    MessageSocket server = accept();

    MessageSocket accept()
    {
        auto accepted = listener.accept(1s);
        REQUIRE(accepted.has_value());
        return std::move(*accepted);
    }
};

} // namespace

TEST_CASE("messages arrive whole and in order", "[unit][camera_ipc][socket]")
{
    Connection connection;
    REQUIRE(connection.client.send(bytes_of("first")) == SendStatus::SENT);
    REQUIRE(connection.client.send(bytes_of("second message")) == SendStatus::SENT);

    std::array<std::byte, 64> buffer;
    ReceivedMessage received = connection.server.receive(buffer, 1s);
    REQUIRE(received.status == ReceiveStatus::MESSAGE);
    REQUIRE(text_of(buffer, received.bytes) == "first");
    REQUIRE(received.descriptors.empty());

    received = connection.server.receive(buffer, 1s);
    REQUIRE(received.status == ReceiveStatus::MESSAGE);
    REQUIRE(text_of(buffer, received.bytes) == "second message");

    // And the other way.
    REQUIRE(connection.server.send(bytes_of("reply")) == SendStatus::SENT);
    received = connection.client.receive(buffer, 1s);
    REQUIRE(text_of(buffer, received.bytes) == "reply");
}

TEST_CASE("descriptors travel with a message", "[unit][camera_ipc][socket]")
{
    Connection connection;
    HostPool left(2, 4096, "camera_ipc_test_left");
    HostPool right(2, 4096, "camera_ipc_test_right");
    left.slot(1)[5] = std::byte{ 0x33 };
    right.slot(0)[9] = std::byte{ 0x44 };

    const std::array<int, 2> descriptors = { left.descriptor(), right.descriptor() };
    REQUIRE(connection.server.send(bytes_of("pools"), descriptors) == SendStatus::SENT);

    std::array<std::byte, 16> buffer;
    ReceivedMessage received = connection.client.receive(buffer, 1s);
    REQUIRE(received.status == ReceiveStatus::MESSAGE);
    REQUIRE(text_of(buffer, received.bytes) == "pools");
    REQUIRE(received.descriptors.size() == 2);

    // The receiver's descriptors are its own, arrive in order and map the same memory.
    REQUIRE(received.descriptors[0].get() != left.descriptor());
    REQUIRE(received.descriptors[1].get() != right.descriptor());
    const HostPoolView left_view(std::move(received.descriptors[0]), 2, 4096);
    const HostPoolView right_view(std::move(received.descriptors[1]), 2, 4096);
    REQUIRE(left_view.slot(1)[5] == std::byte{ 0x33 });
    REQUIRE(right_view.slot(0)[9] == std::byte{ 0x44 });
}

TEST_CASE("send never blocks when the peer does not read", "[unit][camera_ipc][socket]")
{
    Connection connection;
    const std::vector<std::byte> payload(1024, std::byte{ 1 });

    SendStatus status = SendStatus::SENT;
    int sent = 0;
    while (status == SendStatus::SENT && sent < 1'000'000)
    {
        status = connection.client.send(payload);
        sent += status == SendStatus::SENT ? 1 : 0;
    }
    REQUIRE(status == SendStatus::WOULD_BLOCK);
    REQUIRE(sent > 0);

    // Room appears as soon as the peer reads.
    std::array<std::byte, 1024> buffer;
    REQUIRE(connection.server.receive(buffer, 1s).status == ReceiveStatus::MESSAGE);
    REQUIRE(connection.client.send(payload) == SendStatus::SENT);
}

TEST_CASE("a closed peer is reported, not thrown", "[unit][camera_ipc][socket]")
{
    TemporaryDirectory directory;
    MessageListener listener(directory.socket_path());
    std::optional<MessageSocket> server;
    {
        MessageSocket client = MessageSocket::connect(directory.socket_path());
        server = listener.accept(1s);
        REQUIRE(server.has_value());
        REQUIRE(client.send(bytes_of("last words")) == SendStatus::SENT);
    }

    // What was sent before the peer closed is still delivered.
    std::array<std::byte, 32> buffer;
    ReceivedMessage received = server->receive(buffer, 1s);
    REQUIRE(received.status == ReceiveStatus::MESSAGE);
    REQUIRE(text_of(buffer, received.bytes) == "last words");

    REQUIRE(server->receive(buffer, 1s).status == ReceiveStatus::CLOSED);
    REQUIRE(server->send(bytes_of("anyone?")) == SendStatus::CLOSED);
}

TEST_CASE("receive and accept time out when nothing arrives", "[unit][camera_ipc][socket]")
{
    Connection connection;
    std::array<std::byte, 8> buffer;
    REQUIRE(connection.server.receive(buffer, 10ms).status == ReceiveStatus::TIMEOUT);
    REQUIRE(connection.server.receive(buffer, 0ms).status == ReceiveStatus::TIMEOUT);
    REQUIRE_FALSE(connection.listener.accept(10ms).has_value());
}

TEST_CASE("a listener replaces a stale socket file but not a live one", "[unit][camera_ipc][socket]")
{
    TemporaryDirectory directory;
    const std::string path = directory.socket_path();
    {
        MessageListener live(path);
        try
        {
            MessageListener second(path);
            FAIL("a second listener took over a live endpoint");
        }
        catch (const std::system_error& error)
        {
            REQUIRE(error.code().value() == EADDRINUSE);
        }
        // The live listener still works after the refused attempt.
        MessageSocket client = MessageSocket::connect(path);
        REQUIRE(live.accept(1s).has_value());
    }
    REQUIRE_FALSE(std::filesystem::exists(path)); // a listener removes its file

    // A process that died without cleaning up leaves the file behind.
    {
        sockaddr_un address = {};
        address.sun_family = AF_UNIX;
        std::memcpy(address.sun_path, path.data(), path.size());
        const UniqueFd dead(::socket(AF_UNIX, SOCK_SEQPACKET, 0));
        REQUIRE(::bind(dead.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0);
    }
    REQUIRE(std::filesystem::exists(path));
    MessageListener replacement(path);
    MessageSocket client = MessageSocket::connect(path);
    REQUIRE(replacement.accept(1s).has_value());
}

TEST_CASE("the lock decides who listens, not the socket file", "[unit][camera_ipc][socket]")
{
    TemporaryDirectory directory;
    const std::string path = directory.socket_path();
    {
        MessageListener live(path);
        // Even with the socket file gone, the path still belongs to the live listener.
        REQUIRE(std::filesystem::remove(path));
        REQUIRE_THROWS_AS(MessageListener(path), std::system_error);
        // A refused attempt does not loosen the owner's lock, even from the same process.
        REQUIRE_THROWS_AS(MessageListener(path), std::system_error);
        REQUIRE_FALSE(std::filesystem::exists(path));
    }
    // The lock file stays for the next listener, which takes the path over.
    REQUIRE(std::filesystem::exists(path + ".lock"));
    MessageListener next(path);
    MessageSocket client = MessageSocket::connect(path);
    REQUIRE(next.accept(1s).has_value());
}

TEST_CASE("of listeners that start at the same moment exactly one wins", "[unit][camera_ipc][socket]")
{
    TemporaryDirectory directory;
    const std::string path = directory.socket_path();

    std::array<std::unique_ptr<MessageListener>, 8> contenders;
    std::atomic<bool> start{ false };
    std::vector<std::thread> threads;
    for (auto& contender : contenders)
    {
        threads.emplace_back(
            [&contender, &start, &path]
            {
                while (!start.load())
                {
                    std::this_thread::yield();
                }
                try
                {
                    contender = std::make_unique<MessageListener>(path);
                }
                catch (const std::system_error&)
                {
                    // Lost to another contender.
                }
            });
    }
    start = true;
    for (auto& thread : threads)
    {
        thread.join();
    }

    const auto is_listening = [](const std::unique_ptr<MessageListener>& contender) { return contender != nullptr; };
    REQUIRE(std::count_if(contenders.begin(), contenders.end(), is_listening) == 1);

    // The winner owns the socket file: a client reaches it.
    MessageListener& winner = **std::find_if(contenders.begin(), contenders.end(), is_listening);
    MessageSocket client = MessageSocket::connect(path);
    REQUIRE(winner.accept(1s).has_value());
}

TEST_CASE("a path that holds something other than a socket is left alone", "[unit][camera_ipc][socket]")
{
    TemporaryDirectory directory;
    const std::string path = directory.socket_path();
    std::ofstream(path) << "not a socket";

    try
    {
        MessageListener listener(path);
        FAIL("a listener replaced a file that is not a socket");
    }
    catch (const std::system_error& error)
    {
        REQUIRE(error.code().value() == EEXIST);
    }
    REQUIRE(std::filesystem::is_regular_file(path));
    REQUIRE(std::filesystem::file_size(path) == 12);
}

TEST_CASE("only the owner can connect to a listener", "[unit][camera_ipc][socket]")
{
    TemporaryDirectory directory;
    const MessageListener listener(directory.socket_path());

    const std::filesystem::file_status status = std::filesystem::status(directory.socket_path());
    REQUIRE(std::filesystem::is_socket(status));
    REQUIRE(status.permissions() == (std::filesystem::perms::owner_read | std::filesystem::perms::owner_write));
}

TEST_CASE("misuse of a message socket is rejected", "[unit][camera_ipc][socket]")
{
    Connection connection;
    REQUIRE_THROWS_AS(connection.client.send({}), std::invalid_argument);

    const std::vector<int> too_many(MAX_DESCRIPTORS_PER_MESSAGE + 1, 0);
    REQUIRE_THROWS_AS(connection.client.send(bytes_of("x"), too_many), std::invalid_argument);

    // A message that does not fit the receiver's buffer is an error, not a silent cut.
    REQUIRE(connection.client.send(bytes_of("sixteen bytes!!!")) == SendStatus::SENT);
    std::array<std::byte, 4> small;
    REQUIRE_THROWS_AS(connection.server.receive(small, 1s), std::length_error);
    // Also with no room at all, which must not read as a closed connection.
    REQUIRE(connection.client.send(bytes_of("x")) == SendStatus::SENT);
    REQUIRE_THROWS_AS(connection.server.receive({}, 1s), std::length_error);

    TemporaryDirectory directory;
    REQUIRE_THROWS_AS(MessageSocket::connect(directory.socket_path()), std::system_error);
    REQUIRE_THROWS_AS(MessageSocket::connect(std::string(200, 'a')), std::invalid_argument);
    REQUIRE_THROWS_AS(MessageListener(""), std::invalid_argument);
}
