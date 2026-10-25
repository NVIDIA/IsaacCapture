// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// LiveKitVideoOutput against a fake track, with no LiveKit server: pacing, the bounded staging,
// ownership of the borrowed camera frame, the NV12 conversion and reduction, and failure handling.

#include "mock_test_support.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <core/livekit_video_output.hpp>

#include <condition_variable>
#include <cuda_runtime.h>
#include <mutex>
#include <vector>

using namespace plugins::sensing;
using namespace sensing_test;
using namespace std::chrono_literals;

namespace
{

struct Received
{
    VideoFrameInfo info;
    std::vector<uint8_t> nv12;
};

// A track that records what it is given, and can hold a submission open or make it fail.
class FakeSink : public VideoTrackSink
{
public:
    FakeSink(uint32_t width, uint32_t height) : m_width(width), m_height(height), m_buffer(width * height * 3 / 2)
    {
    }

    uint32_t width() const override
    {
        return m_width;
    }
    uint32_t height() const override
    {
        return m_height;
    }
    uint8_t* frame_buffer() override
    {
        return m_buffer.data();
    }

    void submit(const VideoFrameInfo& info) override
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        ++m_entered;
        m_changed.notify_all();
        m_changed.wait(lock, [&] { return !m_gated; });
        if (m_fail)
            throw std::runtime_error("injected sink failure");
        m_received.push_back({ info, m_buffer });
        m_changed.notify_all();
    }

    std::string describe() override
    {
        return "fake";
    }

    void gate(bool closed)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_gated = closed;
        m_changed.notify_all();
    }

    void fail()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_fail = true;
    }

    bool wait_entered(size_t count)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_changed.wait_for(lock, 5s, [&] { return m_entered >= count; });
    }

    bool wait_received(size_t count)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_changed.wait_for(lock, 5s, [&] { return m_received.size() >= count; });
    }

    std::vector<Received> received()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_received;
    }

private:
    uint32_t m_width;
    uint32_t m_height;
    std::vector<uint8_t> m_buffer;
    std::mutex m_mutex;
    std::condition_variable m_changed;
    bool m_gated = false;
    bool m_fail = false;
    size_t m_entered = 0;
    std::vector<Received> m_received;
};

// An RGBA frame in device memory that a test rewrites, standing in for the camera's mailbox slot.
class DeviceFrame
{
public:
    DeviceFrame(uint32_t width, uint32_t height) : m_width(width), m_height(height)
    {
        cudaMallocPitch(reinterpret_cast<void**>(&m_ptr), &m_pitch, static_cast<size_t>(width) * 4, height);
    }
    ~DeviceFrame()
    {
        cudaFree(m_ptr);
    }
    DeviceFrame(const DeviceFrame&) = delete;
    DeviceFrame& operator=(const DeviceFrame&) = delete;

    // Left half `left`, right half `right`.
    void fill(Rgb left, Rgb right)
    {
        std::vector<uint8_t> host(static_cast<size_t>(m_width) * m_height * 4);
        for (uint32_t y = 0; y < m_height; ++y)
        {
            for (uint32_t x = 0; x < m_width; ++x)
            {
                const Rgb c = x < m_width / 2 ? left : right;
                uint8_t* px = &host[(static_cast<size_t>(y) * m_width + x) * 4];
                px[0] = static_cast<uint8_t>(c.r);
                px[1] = static_cast<uint8_t>(c.g);
                px[2] = static_cast<uint8_t>(c.b);
                px[3] = 255;
            }
        }
        cudaMemcpy2D(m_ptr, m_pitch, host.data(), static_cast<size_t>(m_width) * 4, static_cast<size_t>(m_width) * 4,
                     m_height, cudaMemcpyHostToDevice);
    }
    void fill(Rgb colour)
    {
        fill(colour, colour);
    }

    FrameView view(uint64_t sequence, uint64_t timestamp_ns, uint64_t tsc_ns = 0) const
    {
        FrameView v;
        v.ptr = reinterpret_cast<uintptr_t>(m_ptr);
        v.pitch = m_pitch;
        v.width = m_width;
        v.height = m_height;
        v.timestamp_ns = timestamp_ns;
        v.capture_tsc_ns = tsc_ns;
        v.sequence = sequence;
        return v;
    }

private:
    uint32_t m_width;
    uint32_t m_height;
    uint8_t* m_ptr = nullptr;
    size_t m_pitch = 0;
};

constexpr Rgb kRed{ 255, 0, 0 };
constexpr Rgb kBlue{ 0, 0, 255 };
constexpr Rgb kGreen{ 0, 255, 0 };

// BT.601 limited range, the matrix the track assumes.
struct Yuv
{
    int y, u, v;
};

Yuv limited_601(Rgb c)
{
    return { static_cast<int>(16 + 0.257 * c.r + 0.504 * c.g + 0.098 * c.b + 0.5),
             static_cast<int>(128 - 0.148 * c.r - 0.291 * c.g + 0.439 * c.b + 0.5),
             static_cast<int>(128 + 0.439 * c.r - 0.368 * c.g - 0.071 * c.b + 0.5) };
}

Yuv nv12_at(const std::vector<uint8_t>& nv12, uint32_t width, uint32_t height, uint32_t x, uint32_t y)
{
    const size_t luma = static_cast<size_t>(width) * height;
    const size_t chroma = luma + static_cast<size_t>(y / 2) * width + (x / 2) * 2;
    return { nv12[static_cast<size_t>(y) * width + x], nv12[chroma], nv12[chroma + 1] };
}

bool matches(const Yuv& got, const Yuv& want, int tolerance = 2)
{
    return std::abs(got.y - want.y) <= tolerance && std::abs(got.u - want.u) <= tolerance &&
           std::abs(got.v - want.v) <= tolerance;
}

LiveKitOutputConfig config(uint32_t width, uint32_t height, uint32_t fps = 30, uint32_t div = 1)
{
    LiveKitOutputConfig c;
    c.source_width = width;
    c.source_height = height;
    c.fps = fps;
    c.scale_div = div;
    return c;
}

} // namespace

TEST_CASE("the reduction must divide the sensor exactly and leave even sizes", "[unit][gpu][sensing][livekit]")
{
    CHECK_NOTHROW(LiveKitVideoOutput::validate_geometry(2560, 1984, 1));
    CHECK_NOTHROW(LiveKitVideoOutput::validate_geometry(2560, 1984, 2)); // 1280x992
    CHECK_NOTHROW(LiveKitVideoOutput::validate_geometry(2560, 1984, 4)); // 640x496
    CHECK_THROWS_WITH(
        LiveKitVideoOutput::validate_geometry(2560, 1984, 3), Catch::Matchers::ContainsSubstring("exactly"));
    CHECK_THROWS_WITH(
        LiveKitVideoOutput::validate_geometry(2560, 1984, 0), Catch::Matchers::ContainsSubstring("at least 1"));
    // 1984 / 64 = 31, which is odd: NV12 cannot hold it, and it is refused rather than rounded.
    CHECK_THROWS_WITH(LiveKitVideoOutput::validate_geometry(2560, 1984, 64), Catch::Matchers::ContainsSubstring("even"));
    CHECK_THROWS_AS(LiveKitVideoOutput::validate_geometry(0, 0, 1), std::runtime_error);
}

TEST_CASE("a track whose size is not the output size is refused", "[unit][gpu][sensing][livekit]")
{
    CHECK_THROWS_AS(LiveKitVideoOutput(std::make_unique<FakeSink>(64, 48), config(64, 48, 30, 2)), std::runtime_error);
    CHECK_THROWS_AS(LiveKitVideoOutput(std::make_unique<FakeSink>(64, 48), config(64, 48, 0)), std::runtime_error);
}

TEST_CASE("frames are converted to BT.601 limited-range NV12 with the camera's identifiers",
          "[unit][gpu][sensing][livekit]")
{
    auto sink = std::make_unique<FakeSink>(64, 48);
    FakeSink& track = *sink;
    LiveKitVideoOutput output(std::move(sink), config(64, 48));

    DeviceFrame frame(64, 48);
    frame.fill(kRed, kBlue);
    output.submit(frame.view(7, 5'000'000'000ull, 123456789));
    REQUIRE(track.wait_received(1));

    const auto got = track.received().front();
    CHECK(got.info.frame_id == 7);
    CHECK(got.info.timestamp_us == 5'000'000);
    CHECK(got.info.capture_tsc_ns == 123456789);
    CHECK(matches(nv12_at(got.nv12, 64, 48, 8, 12), limited_601(kRed)));
    CHECK(matches(nv12_at(got.nv12, 64, 48, 56, 12), limited_601(kBlue)));
}

TEST_CASE("a reduced output averages blocks and keeps the picture's layout", "[unit][gpu][sensing][livekit]")
{
    auto sink = std::make_unique<FakeSink>(32, 24);
    FakeSink& track = *sink;
    LiveKitVideoOutput output(std::move(sink), config(64, 48, 30, 2));

    DeviceFrame frame(64, 48);
    frame.fill(kRed, kBlue);
    output.submit(frame.view(1, 1'000'000'000));
    REQUIRE(track.wait_received(1));

    const auto got = track.received().front();
    CHECK(got.nv12.size() == 32u * 24u * 3 / 2);
    CHECK(matches(nv12_at(got.nv12, 32, 24, 4, 12), limited_601(kRed)));
    CHECK(matches(nv12_at(got.nv12, 32, 24, 28, 12), limited_601(kBlue)));
}

TEST_CASE("the publication rate is held on the capture clock", "[unit][gpu][sensing][livekit]")
{
    auto sink = std::make_unique<FakeSink>(64, 48);
    FakeSink& track = *sink;
    LiveKitVideoOutput output(std::move(sink), config(64, 48, 30));

    DeviceFrame frame(64, 48);
    frame.fill(kRed);
    // Sixty captures over one second of capture time, offered to a 30 fps track.
    for (uint64_t i = 0; i < 60; ++i)
    {
        output.submit(frame.view(i + 1, 1'000'000'000ull + i * 16'666'667ull));
        std::this_thread::sleep_for(3ms); // let the worker keep up, so nothing is replaced
    }

    const auto stats = output.stats();
    CHECK(stats.offered == 60);
    CHECK(stats.snapshots + stats.paced_out == 60);
    CHECK(stats.snapshots >= 29);
    CHECK(stats.snapshots <= 31);
    CHECK(stats.paced_out >= 29);
    REQUIRE(track.wait_received(stats.snapshots - stats.replaced));
}

TEST_CASE("a stalled track cannot stall capture, and only the newest waiting frame is kept",
          "[unit][gpu][sensing][livekit]")
{
    auto sink = std::make_unique<FakeSink>(64, 48);
    FakeSink& track = *sink;
    track.gate(true);
    LiveKitVideoOutput output(std::move(sink), config(64, 48, 1000));

    DeviceFrame frame(64, 48);
    frame.fill(kRed);
    output.submit(frame.view(1, 1'000'000'000ull));
    REQUIRE(track.wait_entered(1)); // the worker holds frame 1 inside the stalled track

    // Fifty more captures arrive while it is stuck: the polling thread must not wait for any of them.
    const auto start = std::chrono::steady_clock::now();
    for (uint64_t i = 2; i <= 51; ++i)
        output.submit(frame.view(i, 1'000'000'000ull + i * 2'000'000ull));
    CHECK(std::chrono::steady_clock::now() - start < 2s);

    const auto stats = output.stats();
    CHECK(stats.replaced >= 48); // one slot waits, and each newer capture takes it over

    track.gate(false);
    REQUIRE(track.wait_received(2));
    std::this_thread::sleep_for(200ms);
    const auto got = track.received();
    REQUIRE(got.size() == 2); // frame 1, then only the newest
    CHECK(got[0].info.frame_id == 1);
    CHECK(got[1].info.frame_id == 51);
}

TEST_CASE("the camera frame may be rewritten as soon as submit returns", "[unit][gpu][sensing][livekit]")
{
    auto sink = std::make_unique<FakeSink>(64, 48);
    FakeSink& track = *sink;
    track.gate(true);
    LiveKitVideoOutput output(std::move(sink), config(64, 48, 1000));

    DeviceFrame source(64, 48);
    source.fill(kRed);
    output.submit(source.view(1, 1'000'000'000ull));
    REQUIRE(track.wait_entered(1)); // frame 1 is converted and held in the track

    // Frame 2 waits behind it. The camera then rewrites its slot, as it does after the next poll.
    source.fill(kBlue);
    output.submit(source.view(2, 1'010'000'000ull));
    source.fill(kGreen);

    track.gate(false);
    REQUIRE(track.wait_received(2));
    const auto got = track.received();
    CHECK(matches(nv12_at(got[0].nv12, 64, 48, 32, 24), limited_601(kRed)));
    // The snapshot was complete before submit returned, so the later overwrite cannot reach it.
    CHECK(matches(nv12_at(got[1].nv12, 64, 48, 32, 24), limited_601(kBlue)));
}

TEST_CASE("a failing track fails the output and later captures return at once", "[unit][gpu][sensing][livekit]")
{
    auto sink = std::make_unique<FakeSink>(64, 48);
    FakeSink& track = *sink;
    track.fail();
    LiveKitVideoOutput output(std::move(sink), config(64, 48, 1000));

    DeviceFrame frame(64, 48);
    frame.fill(kRed);
    output.submit(frame.view(1, 1'000'000'000ull));
    REQUIRE(wait_for([&] { return output.failed(); }));
    CHECK(output.failure().find("injected sink failure") != std::string::npos);

    const auto start = std::chrono::steady_clock::now();
    for (uint64_t i = 2; i < 100; ++i)
        output.submit(frame.view(i, 1'000'000'000ull + i * 2'000'000ull));
    CHECK(std::chrono::steady_clock::now() - start < 1s);
    CHECK(output.stats().published == 0);
}

TEST_CASE("destroying an output while a frame waits drops it and returns", "[unit][gpu][sensing][livekit]")
{
    DeviceFrame frame(64, 48);
    frame.fill(kRed);

    const auto start = std::chrono::steady_clock::now();
    {
        LiveKitVideoOutput output(std::make_unique<FakeSink>(64, 48), config(64, 48, 1000));
        for (uint64_t i = 1; i <= 20; ++i)
            output.submit(frame.view(i, 1'000'000'000ull + i * 2'000'000ull));
    }
    // The worker joined: nothing is left running against freed memory, and teardown is prompt.
    CHECK(std::chrono::steady_clock::now() - start < 3s);
}
