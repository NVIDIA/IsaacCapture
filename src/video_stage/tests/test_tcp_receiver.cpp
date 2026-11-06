// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <arpa/inet.h>
#include <catch2/catch_test_macros.hpp>
#include <netinet/in.h>
#include <sys/socket.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <tcp_receiver.hpp>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace std::chrono_literals;
using video_stage::TcpReceiver;

namespace
{

constexpr uint16_t kStopPort = 5741;
constexpr uint16_t kRacePort = 5742;
constexpr uint16_t kSessionPort = 5743;

int connect_to(uint16_t port)
{
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (fd < 0 || ::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0)
    {
        if (fd >= 0)
            ::close(fd);
        return -1;
    }
    return fd;
}

/// Opens a DATA and a CTRL connection with their hellos; the descriptors are closed on destruction.
struct Sender
{
    explicit Sender(uint16_t port)
    {
        data = connect_to(port);
        if (data >= 0)
            (void)::send(data, "DATA", 4, MSG_NOSIGNAL);
        control = connect_to(port);
        if (control >= 0)
            (void)::send(control, "CTRL", 4, MSG_NOSIGNAL);
    }
    ~Sender()
    {
        for (const int fd : { data, control })
            if (fd >= 0)
                ::close(fd);
    }
    Sender(const Sender&) = delete;
    Sender& operator=(const Sender&) = delete;
    int data = -1;
    int control = -1;
};

} // namespace

TEST_CASE("a session is accepted and carries a record and control bytes", "[tcp_receiver]")
{
    TcpReceiver receiver(kSessionPort);
    Sender sender(kSessionPort);
    REQUIRE(sender.data >= 0);
    REQUIRE(sender.control >= 0);
    receiver.accept_session();

    const uint8_t record[] = { 0, 0, 0, 3, 'a', 'b', 'c' }; // 4-byte length prefix, then the record
    REQUIRE(::send(sender.data, record, sizeof(record), MSG_NOSIGNAL) == static_cast<ssize_t>(sizeof(record)));
    std::vector<uint8_t> out;
    REQUIRE(receiver.read_record(out));
    CHECK(out == std::vector<uint8_t>{ 'a', 'b', 'c' });

    const uint8_t ping[] = { 7 };
    REQUIRE(receiver.send_control(ping, 1));
    uint8_t got = 0;
    REQUIRE(::recv(sender.control, &got, 1, 0) == 1);
    CHECK(got == 7);
    receiver.shutdown_session();
    CHECK_FALSE(receiver.read_record(out)); // a shut-down session ends cleanly
}

TEST_CASE("stop_listening ends a blocked accept_session", "[tcp_receiver]")
{
    TcpReceiver receiver(kStopPort);
    auto accepted = std::async(std::launch::async,
                               [&]
                               {
                                   try
                                   {
                                       receiver.accept_session();
                                       return false;
                                   }
                                   catch (const std::exception&)
                                   {
                                       return true;
                                   }
                               });
    std::this_thread::sleep_for(200ms);
    REQUIRE(accepted.wait_for(0ms) == std::future_status::timeout); // blocked, with no sender
    receiver.stop_listening();
    REQUIRE(accepted.wait_for(3s) == std::future_status::ready);
    CHECK(accepted.get()); // it threw instead of returning a session
}

TEST_CASE("stop_listening ends an accept blocked on a silent connection", "[tcp_receiver]")
{
    TcpReceiver receiver(kStopPort + 10);
    const int silent = connect_to(kStopPort + 10); // connects, never sends its hello
    REQUIRE(silent >= 0);
    auto accepted = std::async(std::launch::async,
                               [&]
                               {
                                   try
                                   {
                                       receiver.accept_session();
                                   }
                                   catch (const std::exception&)
                                   {
                                   }
                               });
    std::this_thread::sleep_for(300ms); // the hello read is now pending (bounded at 2 s)
    const auto begin = std::chrono::steady_clock::now();
    receiver.stop_listening();
    REQUIRE(accepted.wait_for(1500ms) == std::future_status::ready);
    CHECK(std::chrono::steady_clock::now() - begin < 1500ms);
    ::close(silent);
}

TEST_CASE("shutdown_session from another thread while accept_session replaces descriptors", "[tcp_receiver][race]")
{
    TcpReceiver receiver(kRacePort);

    for (int round = 0; round < 200; ++round)
    {
        std::atomic<bool> done{ false };

        std::thread accepting(
            [&]
            {
                try
                {
                    receiver.accept_session();
                }
                catch (const std::exception&)
                {
                }
                done = true;
            });

        // A sender that keeps offering a DATA/CTRL pair until a session is up; a shutdown may eat a hello in flight.
        std::thread client(
            [&]
            {
                while (!done)
                {
                    Sender sender(kRacePort);
                    for (int i = 0; i < 20 && !done; ++i)
                        std::this_thread::sleep_for(1ms);
                }
            });

        std::thread shutting(
            [&]
            {
                while (!done)
                {
                    receiver.shutdown_session();
                    std::this_thread::yield();
                }
            });

        // Users of the session on other threads: none may touch a descriptor that is being replaced.
        std::thread user(
            [&]
            {
                uint8_t byte = 0;
                std::vector<uint8_t> record;
                while (!done)
                {
                    (void)receiver.pending_bytes();
                    (void)receiver.send_control(&byte, 1);
                    std::this_thread::yield();
                }
            });

        const auto deadline = std::chrono::steady_clock::now() + 15s;
        while (!done && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(1ms);
        if (!done)
            receiver.stop_listening(); // do not hang the suite if a session never completes
        accepting.join();
        client.join();
        shutting.join();
        user.join();
        REQUIRE(std::chrono::steady_clock::now() < deadline);
        receiver.shutdown_session(); // end any reader before the next round retires the descriptors
    }
}
