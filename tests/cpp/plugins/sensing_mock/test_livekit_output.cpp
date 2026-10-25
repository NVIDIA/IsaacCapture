// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// LiveKitVideoOutput against a fake track, with no LiveKit server: the hardware-encoded units it hands
// over, pacing, the bounded staging, ownership of the borrowed camera frame, the reduction, the
// receiving side's keyframe requests, refused frames, the end-of-stream drain and failure handling.

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
    EncodedVideoFrame info; ///< `data` is not valid after submit() returns; see `bytes`.
    std::vector<uint8_t> bytes;
};

// A track that records what it is given. It can hold a submission open, make it fail, refuse frames
// as the SDK can, and hand back the receiving side's requests.
class FakeSink : public VideoTrackSink
{
public:
    FakeSink(uint32_t width, uint32_t height) : m_width(width), m_height(height)
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

    bool submit(const EncodedVideoFrame& frame) override
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        ++m_entered;
        m_changed.notify_all();
        m_changed.wait(lock, [&] { return !m_gated; });
        if (m_fail)
            throw std::runtime_error("injected sink failure");
        if (m_refuse > 0)
        {
            --m_refuse;
            return false;
        }
        Received got;
        got.info = frame;
        got.bytes.assign(frame.data, frame.data + frame.size);
        got.info.data = nullptr;
        m_received.push_back(std::move(got));
        m_changed.notify_all();
        return true;
    }

    EncoderFeedback take_feedback() override
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        EncoderFeedback out = m_feedback;
        m_feedback = {};
        return out;
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

    /// The next `count` frames are refused, as the SDK may.
    void refuse(int count)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_refuse = count;
    }

    void request_keyframe()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_feedback.keyframe_requested = true;
    }

    void request_bitrate(uint32_t bps)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_feedback.target_bitrate_bps = bps;
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

    /// Every received unit, in order, as one Annex-B stream.
    std::vector<uint8_t> stream()
    {
        std::vector<uint8_t> out;
        for (const auto& unit : received())
            out.insert(out.end(), unit.bytes.begin(), unit.bytes.end());
        return out;
    }

private:
    uint32_t m_width;
    uint32_t m_height;
    std::mutex m_mutex;
    std::condition_variable m_changed;
    bool m_gated = false;
    bool m_fail = false;
    int m_refuse = 0;
    EncoderFeedback m_feedback;
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

// Mean luma the encoder should produce for a solid colour: BT.601 limited range.
double luma_601(Rgb c)
{
    return 16 + 0.257 * c.r + 0.504 * c.g + 0.098 * c.b;
}

LiveKitOutputConfig config(uint32_t width, uint32_t height, uint32_t fps = 30, uint32_t div = 1)
{
    LiveKitOutputConfig c;
    c.source_width = width;
    c.source_height = height;
    c.fps = fps;
    c.scale_div = div;
    c.bitrate_bps = 4'000'000;
    c.peak_bitrate_bps = 6'000'000;
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
    CHECK_THROWS_AS(
        LiveKitVideoOutput(std::make_unique<FakeSink>(320, 240), config(320, 240, 30, 2)), std::runtime_error);
    CHECK_THROWS_AS(LiveKitVideoOutput(std::make_unique<FakeSink>(320, 240), config(320, 240, 0)), std::runtime_error);
}

TEST_CASE("an encoder that cannot start fails construction, with no software fallback", "[unit][gpu][sensing][livekit]")
{
    // Far beyond what H.264 hardware encodes.
    CHECK_THROWS_WITH(LiveKitVideoOutput(std::make_unique<FakeSink>(16384, 16384), config(16384, 16384)),
                      Catch::Matchers::ContainsSubstring("h264_nvenc"));
}

TEST_CASE("frames come out hardware-encoded as decodable units with the camera's identifiers",
          "[unit][gpu][sensing][livekit]")
{
    auto sink = std::make_unique<FakeSink>(320, 240);
    FakeSink& track = *sink;
    LiveKitVideoOutput output(std::move(sink), config(320, 240));

    DeviceFrame frame(320, 240);
    frame.fill(kRed, kBlue);
    output.submit(frame.view(7, 5'000'000'000ull, 123456789));
    output.flush();
    REQUIRE(track.wait_received(1));

    const auto got = track.received();
    REQUIRE(got.size() == 1);
    CHECK(got[0].info.keyframe);
    CHECK(got[0].info.frame_id == 7);
    CHECK(got[0].info.timestamp_us == 5'000'000);
    CHECK(got[0].info.capture_tsc_ns == 123456789);

    // SPS and PPS travel with the IDR, so a receiver can start on this unit alone.
    std::vector<uint8_t> types;
    for (const auto& nal : split_annex_b(got[0].bytes))
        types.push_back(nal.type);
    REQUIRE(types.size() >= 3);
    CHECK(types[0] == kNalSps);
    CHECK(types[1] == kNalPps);

    const auto pictures = decode_annex_b(got[0].bytes);
    REQUIRE(pictures.size() == 1);
    CHECK(pictures[0].width == 320);
    CHECK(pictures[0].height == 240);
    CHECK(std::abs(pictures[0].luma_left - luma_601(kRed)) < 12);
    CHECK(std::abs(pictures[0].luma_right - luma_601(kBlue)) < 12);
    CHECK(output.describe().find("pre-encoded passthrough") != std::string::npos);
}

TEST_CASE("a reduced output is encoded at the reduced size and keeps the picture's layout",
          "[unit][gpu][sensing][livekit]")
{
    auto sink = std::make_unique<FakeSink>(320, 240);
    FakeSink& track = *sink;
    LiveKitVideoOutput output(std::move(sink), config(640, 480, 30, 2));

    DeviceFrame frame(640, 480);
    frame.fill(kRed, kBlue);
    output.submit(frame.view(1, 1'000'000'000));
    output.flush();
    REQUIRE(track.wait_received(1));

    const auto pictures = decode_annex_b(track.stream());
    REQUIRE(pictures.size() == 1);
    CHECK(pictures[0].width == 320);
    CHECK(pictures[0].height == 240);
    CHECK(std::abs(pictures[0].luma_left - luma_601(kRed)) < 12);
    CHECK(std::abs(pictures[0].luma_right - luma_601(kBlue)) < 12);
}

TEST_CASE("the publication rate is held on the capture clock, and every encoded frame is drained",
          "[unit][gpu][sensing][livekit]")
{
    auto sink = std::make_unique<FakeSink>(320, 240);
    FakeSink& track = *sink;
    LiveKitVideoOutput output(std::move(sink), config(320, 240, 30));

    DeviceFrame frame(320, 240);
    frame.fill(kRed);
    // Sixty captures over one second of capture time, offered to a 30 fps track.
    for (uint64_t i = 0; i < 60; ++i)
    {
        output.submit(frame.view(i + 1, 1'000'000'000ull + i * 16'666'667ull));
        std::this_thread::sleep_for(3ms); // let the worker keep up, so nothing is replaced
    }
    output.flush();

    const auto stats = output.stats();
    CHECK(stats.offered == 60);
    CHECK(stats.snapshots + stats.paced_out == 60);
    CHECK(stats.snapshots >= 29);
    CHECK(stats.snapshots <= 31);
    CHECK(stats.paced_out >= 29);
    // Each frame the encoder took is published after the drain, one unit per frame.
    CHECK(stats.published == stats.snapshots - stats.replaced - stats.encoder_busy);
    CHECK(track.received().size() == stats.published);
}

TEST_CASE("the end-of-stream drain publishes every unit with its own stamp", "[unit][gpu][sensing][livekit]")
{
    auto sink = std::make_unique<FakeSink>(320, 240);
    FakeSink& track = *sink;
    LiveKitVideoOutput output(std::move(sink), config(320, 240, 1000));

    DeviceFrame frame(320, 240);
    frame.fill(kGreen);
    for (uint64_t i = 1; i <= 5; ++i)
    {
        output.submit(frame.view(i, 2'000'000'000ull + i * 33'333'000ull, 1000 + i));
        std::this_thread::sleep_for(15ms);
    }
    output.flush();

    const auto got = track.received();
    REQUIRE(got.size() == 5);
    for (size_t i = 0; i < got.size(); ++i)
    {
        CHECK(got[i].info.frame_id == i + 1);
        CHECK(got[i].info.timestamp_us == (2'000'000'000ull + (i + 1) * 33'333'000ull) / 1000);
        CHECK(got[i].info.capture_tsc_ns == 1001 + i);
        CHECK(got[i].info.keyframe == (i == 0));
    }
    CHECK(decode_annex_b(track.stream()).size() == 5);
}

TEST_CASE("a keyframe request from the receiving side makes the next frame an IDR", "[unit][gpu][sensing][livekit]")
{
    auto sink = std::make_unique<FakeSink>(320, 240);
    FakeSink& track = *sink;
    LiveKitVideoOutput output(std::move(sink), config(320, 240, 1000));

    DeviceFrame frame(320, 240);
    frame.fill(kRed);
    for (uint64_t i = 1; i <= 3; ++i)
    {
        output.submit(frame.view(i, 1'000'000'000ull + i * 33'000'000ull));
        std::this_thread::sleep_for(40ms);
    }
    track.request_keyframe();
    for (uint64_t i = 4; i <= 6; ++i)
    {
        output.submit(frame.view(i, 1'000'000'000ull + i * 33'000'000ull));
        std::this_thread::sleep_for(40ms);
    }
    output.flush();

    const auto got = track.received();
    REQUIRE(got.size() == 6);
    std::vector<bool> keyframes;
    for (const auto& unit : got)
        keyframes.push_back(unit.info.keyframe);
    CHECK(keyframes == std::vector<bool>{ true, false, false, true, false, false });
    CHECK(output.stats().keyframes_requested == 1);

    // The re-sent IDR carries parameter sets again, so a late joiner can start on it.
    const auto nals = split_annex_b(got[3].bytes);
    REQUIRE(nals.size() >= 3);
    CHECK(nals[0].type == kNalSps);
    CHECK(nals[1].type == kNalPps);

    // And the whole stream, with the second IDR, decodes.
    CHECK(decode_annex_b(track.stream()).size() == 6);
}

TEST_CASE("a frame the SDK refuses is followed by a keyframe, and nothing between", "[unit][gpu][sensing][livekit]")
{
    auto sink = std::make_unique<FakeSink>(320, 240);
    FakeSink& track = *sink;
    LiveKitVideoOutput output(std::move(sink), config(320, 240, 1000));

    DeviceFrame frame(320, 240);
    frame.fill(kRed);
    output.submit(frame.view(1, 1'000'000'000ull));
    std::this_thread::sleep_for(60ms);
    REQUIRE(track.wait_received(1));

    track.refuse(1);
    for (uint64_t i = 2; i <= 6; ++i)
    {
        output.submit(frame.view(i, 1'000'000'000ull + i * 33'000'000ull));
        std::this_thread::sleep_for(60ms);
    }
    output.flush();

    // Frame 2 was refused; frame 3 was encoded as the IDR that repairs the stream.
    const auto got = track.received();
    std::vector<uint32_t> ids;
    for (const auto& unit : got)
        ids.push_back(unit.info.frame_id);
    CHECK(ids == std::vector<uint32_t>{ 1, 3, 4, 5, 6 });
    CHECK(got[1].info.keyframe);
    CHECK(output.stats().rejected == 1);
    CHECK(decode_annex_b(track.stream()).size() == 5);
}

TEST_CASE("a bitrate target the backend cannot apply does not disturb the stream", "[unit][gpu][sensing][livekit]")
{
    auto sink = std::make_unique<FakeSink>(320, 240);
    FakeSink& track = *sink;
    LiveKitVideoOutput output(std::move(sink), config(320, 240, 1000));

    DeviceFrame frame(320, 240);
    frame.fill(kRed);
    track.request_bitrate(500'000'000); // above any ceiling: clamped, and the mock encoder keeps its rate
    output.submit(frame.view(1, 1'000'000'000ull));
    output.flush();
    CHECK_FALSE(output.failed());
    CHECK(track.received().size() == 1);
}

TEST_CASE("a stalled track cannot stall capture, and only the newest waiting frame is kept",
          "[unit][gpu][sensing][livekit]")
{
    auto sink = std::make_unique<FakeSink>(320, 240);
    FakeSink& track = *sink;
    track.gate(true);
    LiveKitVideoOutput output(std::move(sink), config(320, 240, 1000));

    DeviceFrame frame(320, 240);
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
    auto sink = std::make_unique<FakeSink>(320, 240);
    FakeSink& track = *sink;
    track.gate(true);
    LiveKitVideoOutput output(std::move(sink), config(320, 240, 1000));

    DeviceFrame source(320, 240);
    source.fill(kRed);
    output.submit(source.view(1, 1'000'000'000ull));
    REQUIRE(track.wait_entered(1)); // frame 1 is encoded and held in the track

    // Frame 2 waits behind it. The camera then rewrites its slot, as it does after the next poll.
    source.fill(kBlue);
    output.submit(source.view(2, 1'010'000'000ull));
    source.fill(kGreen);

    track.gate(false);
    output.flush();
    REQUIRE(track.wait_received(2));
    const auto pictures = decode_annex_b(track.stream());
    REQUIRE(pictures.size() == 2);
    CHECK(std::abs(pictures[0].luma - luma_601(kRed)) < 12);
    // The snapshot was complete before submit returned, so the later overwrite cannot reach it.
    CHECK(std::abs(pictures[1].luma - luma_601(kBlue)) < 12);
}

TEST_CASE("a failing track fails the output and later captures return at once", "[unit][gpu][sensing][livekit]")
{
    auto sink = std::make_unique<FakeSink>(320, 240);
    FakeSink& track = *sink;
    track.fail();
    LiveKitVideoOutput output(std::move(sink), config(320, 240, 1000));

    DeviceFrame frame(320, 240);
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
    DeviceFrame frame(320, 240);
    frame.fill(kRed);

    const auto start = std::chrono::steady_clock::now();
    {
        LiveKitVideoOutput output(std::make_unique<FakeSink>(320, 240), config(320, 240, 1000));
        for (uint64_t i = 1; i <= 20; ++i)
            output.submit(frame.view(i, 1'000'000'000ull + i * 2'000'000ull));
    }
    // The worker joined: nothing is left running against freed memory, and teardown is prompt.
    CHECK(std::chrono::steady_clock::now() - start < 3s);
}
