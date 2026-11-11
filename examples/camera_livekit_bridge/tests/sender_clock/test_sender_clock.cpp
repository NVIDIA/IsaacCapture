// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// SenderClockClient against a fake UDP responder that misbehaves on demand.

#include <arpa/inet.h>
#include <sender_clock/sender_clock_client.hpp>
#include <sys/socket.h>
#include <video_types/video_types.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace
{

using camera_bridge::monotonic_ns;
using camera_bridge::SenderClockClient;
using camera_bridge::SenderClockConfig;

int g_failures = 0;

void check(bool condition, const std::string& what)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << what << "\n";
        ++g_failures;
    }
}

enum class Mode
{
    Good,
    Mute,
    WrongSequence,
    WrongT1,
    WrongTag,
    Short,
    Long,
    Delayed, // answers every request only after the next one arrives, so every reply is stale
};

void put_le(std::uint8_t* out, std::uint64_t value, std::size_t bytes)
{
    for (std::size_t i = 0; i < bytes; ++i)
        out[i] = static_cast<std::uint8_t>((value >> (8 * i)) & 0xFF);
}

std::uint64_t get_le(const std::uint8_t* in, std::size_t bytes)
{
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < bytes; ++i)
        value |= static_cast<std::uint64_t>(in[i]) << (8 * i);
    return value;
}

/// A responder on the sender's clock = host clock + offset.
class FakeSender
{
public:
    explicit FakeSender(int64_t offset_ns) : m_offset(offset_ns)
    {
        m_fd = ::socket(AF_INET, SOCK_DGRAM, 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        ::bind(m_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
        socklen_t len = sizeof(addr);
        ::getsockname(m_fd, reinterpret_cast<sockaddr*>(&addr), &len);
        m_port = ntohs(addr.sin_port);
        timeval tv{ 0, 50'000 };
        ::setsockopt(m_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        m_thread = std::thread([this] { serve(); });
    }
    ~FakeSender()
    {
        m_stop = true;
        m_thread.join();
        ::close(m_fd);
    }
    std::uint16_t port() const
    {
        return m_port;
    }
    void set_mode(Mode mode)
    {
        m_mode = mode;
    }

private:
    void serve()
    {
        std::vector<std::uint8_t> held;
        sockaddr_in held_peer{};
        while (!m_stop)
        {
            std::uint8_t in[64];
            sockaddr_in peer{};
            socklen_t len = sizeof(peer);
            const ssize_t got = ::recvfrom(m_fd, in, sizeof(in), 0, reinterpret_cast<sockaddr*>(&peer), &len);
            if (got != 13)
                continue;
            const int64_t t2 = monotonic_ns() + m_offset;
            const Mode mode = m_mode;
            if (mode == Mode::Mute)
                continue;
            std::uint8_t out[40] = {};
            out[0] = mode == Mode::WrongTag ? 'X' : 'R';
            std::uint32_t seq = static_cast<std::uint32_t>(get_le(in + 1, 4));
            if (mode == Mode::WrongSequence)
                seq += 1;
            put_le(out + 1, seq, 4);
            std::int64_t t1 = static_cast<std::int64_t>(get_le(in + 5, 8));
            if (mode == Mode::WrongT1)
                t1 += 1;
            put_le(out + 5, static_cast<std::uint64_t>(t1), 8);
            put_le(out + 13, static_cast<std::uint64_t>(t2), 8);
            put_le(out + 21, static_cast<std::uint64_t>(monotonic_ns() + m_offset), 8);
            std::size_t size = 29;
            if (mode == Mode::Short)
                size = 28;
            if (mode == Mode::Long)
                size = 30;
            if (mode == Mode::Delayed)
            {
                if (!held.empty())
                    ::sendto(
                        m_fd, held.data(), held.size(), 0, reinterpret_cast<sockaddr*>(&held_peer), sizeof(held_peer));
                held.assign(out, out + size);
                held_peer = peer;
                continue;
            }
            ::sendto(m_fd, out, size, 0, reinterpret_cast<sockaddr*>(&peer), len);
        }
    }

    int m_fd = -1;
    std::uint16_t m_port = 0;
    int64_t m_offset;
    std::atomic<Mode> m_mode{ Mode::Good };
    std::atomic<bool> m_stop{ false };
    std::thread m_thread;
};

void wait_ms(int ms)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

SenderClockConfig fast_config(std::uint16_t port)
{
    SenderClockConfig config;
    config.host = "127.0.0.1";
    config.port = port;
    config.interval = std::chrono::milliseconds(20);
    config.reply_timeout = std::chrono::milliseconds(60);
    return config;
}

void test_measures_the_offset()
{
    const int64_t offset = 7'000'000'000;
    FakeSender sender(offset);
    SenderClockClient client(fast_config(sender.port()));
    client.start();
    wait_ms(400);
    const auto snap = client.snapshot();
    client.stop();
    check(snap.mapping.has_value(), "a mapping exists after replies");
    check(snap.accepted_in_epoch >= 3, "several exchanges accepted");
    if (snap.mapping)
    {
        const int64_t error = snap.mapping->offset_ns - offset;
        check(error > -2'000'000 && error < 2'000'000, "offset within 2 ms of the fake sender's");
    }
}

void test_unmatched_replies_are_not_measurements()
{
    for (Mode mode : { Mode::WrongSequence, Mode::WrongT1, Mode::WrongTag, Mode::Short, Mode::Long, Mode::Delayed })
    {
        FakeSender sender(1'000'000'000);
        sender.set_mode(mode);
        SenderClockClient client(fast_config(sender.port()));
        client.start();
        wait_ms(300);
        const auto snap = client.snapshot();
        client.stop();
        check(!snap.mapping && snap.exchanges == 0, "no measurement from a reply that does not answer the request (mode " +
                                                        std::to_string(static_cast<int>(mode)) + ")");
        check(snap.ignored > 0 || mode == Mode::Delayed, "the bad replies are counted as ignored");
    }
}

void test_loss_and_recovery()
{
    FakeSender sender(2'000'000'000);
    SenderClockClient client(fast_config(sender.port()));
    client.start();
    wait_ms(300);
    check(client.snapshot().mapping.has_value(), "mapping before the loss");
    sender.set_mode(Mode::Mute);
    wait_ms(200);
    const auto muted = client.snapshot();
    check(muted.timeouts > 0, "silence is counted as timeouts");
    sender.set_mode(Mode::Good);
    wait_ms(300);
    const auto after = client.snapshot();
    client.stop();
    check(after.exchanges > muted.exchanges, "measurements resume after the loss");
}

void test_reset_voids_the_outstanding_request()
{
    FakeSender sender(3'000'000'000);
    SenderClockClient client(fast_config(sender.port()));
    client.start();
    wait_ms(300);
    const auto before = client.snapshot();
    check(before.mapping.has_value() && before.accepted_in_epoch >= 3, "warm before the reset");
    client.begin_session(1);
    const auto reset = client.snapshot();
    check(!reset.mapping && reset.accepted_in_epoch == 0, "a new session discards every earlier measurement");
    check(reset.generation > before.generation, "the generation advanced");
    client.begin_session(1);
    check(client.snapshot().generation == reset.generation, "the same session does not reset again");
    wait_ms(300);
    check(client.snapshot().mapping.has_value(), "measurements resume in the new generation");
    client.stop();
}

void test_wrong_peer_is_ignored()
{
    FakeSender sender(1'000'000'000);
    FakeSender other(5'000'000'000);
    // The client asks `sender`; `other` cannot be heard because replies must come from the address asked.
    SenderClockClient client(fast_config(sender.port()));
    client.start();
    wait_ms(200);
    const auto snap = client.snapshot();
    client.stop();
    check(snap.mapping && snap.mapping->offset_ns < 2'000'000'000, "only the configured sender's clock is used");
}

} // namespace

int main()
{
    test_measures_the_offset();
    test_unmatched_replies_are_not_measurements();
    test_loss_and_recovery();
    test_reset_voids_the_outstanding_request();
    test_wrong_peer_is_ignored();
    if (g_failures != 0)
    {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "all sender_clock checks passed\n";
    return 0;
}
