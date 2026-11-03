// SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "realhand_ffg_glove_reader.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <termios.h>
#include <thread>
#include <unistd.h>

using namespace plugins::realhand_ffg_glove;
using namespace std::chrono_literals;

namespace
{
void check(bool condition, const char* message)
{
    if (!condition)
    {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::abort();
    }
}

class FakeGlove
{
public:
    explicit FakeGlove(HandSide side = HandSide::Left) : side_(side)
    {
        fd_ = ::posix_openpt(O_RDWR | O_NOCTTY | O_NONBLOCK);
        check(fd_ >= 0 && ::grantpt(fd_) == 0 && ::unlockpt(fd_) == 0, "create PTY");
        port = ::ptsname(fd_);
    }
    ~FakeGlove()
    {
        ::close(fd_);
    }
    void reply()
    {
        uint8_t bytes[1024];
        ssize_t size;
        while ((size = ::read(fd_, bytes, sizeof(bytes))) > 0)
        {
            for (ssize_t i = 0; i < size; ++i)
            {
                if (auto frame = parser_.process(bytes[i]))
                {
                    if (frame->command == Command::Version)
                    {
                        ++version_queries;
                        if (respond)
                        {
                            send(Command::Version, { 0x88, 0x4E, 0, 0, static_cast<uint8_t>(side_ == HandSide::Right) });
                        }
                    }
                    else if (frame->command == Command::Position)
                    {
                        ++position_queries;
                        if (respond)
                        {
                            std::vector<uint8_t> payload(kNumSensors * sizeof(float));
                            for (std::size_t index = 0; index < kNumSensors; ++index)
                            {
                                float angle = static_cast<float>(index + 1);
                                std::memcpy(payload.data() + index * sizeof(float), &angle, sizeof(float));
                            }
                            send(Command::Position, payload);
                        }
                    }
                }
            }
        }
    }
    std::string port;
    bool respond = true;
    int version_queries = 0;
    int position_queries = 0;

private:
    void send(Command command, const std::vector<uint8_t>& payload)
    {
        const auto bytes = pack_frame(command, payload);
        check(::write(fd_, bytes.data(), bytes.size()) == static_cast<ssize_t>(bytes.size()), "PTY reply");
    }
    int fd_ = -1;
    HandSide side_;
    FrameParser parser_;
};

void pump(GloveReader& reader, std::initializer_list<FakeGlove*> gloves, std::chrono::milliseconds duration)
{
    const auto until = GloveReader::Clock::now() + duration;
    while (GloveReader::Clock::now() < until)
    {
        const auto start = GloveReader::Clock::now();
        reader.update();
        check(GloveReader::Clock::now() - start < 100ms, "discovery must not wait 250ms for a silent device");
        for (auto* glove : gloves)
        {
            glove->reply();
        }
        std::this_thread::sleep_for(2ms);
    }
}

void test_cold_start_and_silent_port()
{
    FakeGlove left;
    FakeGlove silent;
    silent.respond = false;
    GloveReader reader({ left.port, silent.port }, { 460800, 921600 });
    pump(reader, { &left, &silent }, 650ms);
    check(reader.endpoints()[0] && reader.endpoints()[0]->has_positions, "cold start discovers and queries left glove");
    check(!reader.endpoints()[1], "one glove never populates opposite side");
    check(left.position_queries >= 25, "silent probes must not stall connected glove queries");
    check(reader.endpoints()[0]->positions[0] > 0, "live position data decoded");
    left.respond = false;
    pump(reader, { &left, &silent }, 1100ms);
    check(!reader.endpoints()[0], "stale glove disconnected");
    left.respond = true;
    pump(reader, { &left, &silent }, 2100ms);
    check(reader.endpoints()[0] && reader.endpoints()[0]->has_positions, "disconnected glove rediscovered");
}

void test_discovery_stops_with_two_gloves()
{
    FakeGlove left;
    FakeGlove right(HandSide::Right);
    FakeGlove unused;
    GloveReader reader({ left.port, right.port, unused.port }, { 460800 });
    pump(reader, { &left, &right, &unused }, 100ms);
    check(reader.endpoints()[0] && reader.endpoints()[1], "both sides identified");
    check(unused.version_queries == 0, "no extra serial ports probed after both gloves connect");
}

void test_busy_port_is_not_modified()
{
    FakeGlove device;
    const int owner = ::open(device.port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    check(owner >= 0, "open non-exclusive device owner");
    termios before{}, after{};
    check(::tcgetattr(owner, &before) == 0, "read original serial configuration");
    bool rejected = false;
    try
    {
        SerialGlove glove(device.port, 460800);
    }
    catch (const std::runtime_error&)
    {
        rejected = true;
    }
    check(rejected, "reject already open non-exclusive serial port");
    check(::tcgetattr(owner, &after) == 0 && ::cfgetospeed(&after) == ::cfgetospeed(&before), "busy port baud unchanged");
    device.reply();
    check(device.version_queries == 0, "no injected probe bytes on busy port");
    ::close(owner);
    {
        SerialGlove glove(device.port, 460800);
    }
    {
        SerialGlove glove(device.port, 460800);
    }
}

void test_writes_timeout_when_device_stalls()
{
    FakeGlove device;
    SerialGlove glove(device.port, 460800);
    bool timed_out = false;
    for (int i = 0; i < 10000 && !timed_out; ++i)
    {
        const auto start = GloveReader::Clock::now();
        try
        {
            glove.send(Command::ForceFeedback, std::vector<uint8_t>(255, 0));
        }
        catch (const std::runtime_error&)
        {
            timed_out = true;
        }
        check(GloveReader::Clock::now() - start < 100ms, "non-draining output must have a bounded write");
    }
    check(timed_out, "backpressure reaches write timeout");
}
} // namespace

void run_reader_tests()
{
    test_cold_start_and_silent_port();
    test_discovery_stops_with_two_gloves();
    test_busy_port_is_not_modified();
    test_writes_timeout_when_device_stalls();
}
