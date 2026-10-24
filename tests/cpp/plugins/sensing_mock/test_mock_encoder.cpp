// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mock_test_support.hpp"

#include <catch2/catch_test_macros.hpp>
#include <core/jetson_encoder.hpp>

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

using namespace plugins::sensing;
using namespace sensing_test;
using namespace std::chrono_literals;

namespace
{

constexpr uint32_t kWidth = 64;
constexpr uint32_t kHeight = 48;

class Frame
{
public:
    Frame()
    {
        cudaMallocPitch(reinterpret_cast<void**>(&m_ptr), &m_pitch, kWidth * 4, kHeight);
        cudaMemset2D(m_ptr, m_pitch, 0x80, kWidth * 4, kHeight);
    }
    ~Frame()
    {
        cudaFree(m_ptr);
    }
    uintptr_t ptr() const
    {
        return reinterpret_cast<uintptr_t>(m_ptr);
    }
    size_t pitch() const
    {
        return m_pitch;
    }

private:
    uint8_t* m_ptr = nullptr;
    size_t m_pitch = 0;
};

EncoderConfig make_config()
{
    EncoderConfig config;
    config.width = kWidth;
    config.height = kHeight;
    config.fps = 30;
    config.bitrate_bps = 1'000'000;
    return config;
}

} // namespace

TEST_CASE("encoder rejects zero or odd geometry", "[unit][gpu][sensing]")
{
    auto config = make_config();
    config.width = 0;
    CHECK_THROWS_AS(JetsonEncoder(config), std::runtime_error);
    config = make_config();
    config.height = 47;
    CHECK_THROWS_AS(JetsonEncoder(config), std::runtime_error);
}

TEST_CASE("encoder holds frames back during warm-up, then emits them in order", "[unit][gpu][sensing]")
{
    Frame frame;
    JetsonEncoder encoder(make_config());

    REQUIRE(encoder.submit(frame.ptr(), frame.pitch(), 1'000'000'000));
    std::this_thread::sleep_for(100ms);
    CHECK(encoder.poll().empty()); // nothing yet: the encoder wants more input first

    REQUIRE(encoder.submit(frame.ptr(), frame.pitch(), 1'033'000'000));
    REQUIRE(encoder.submit(frame.ptr(), frame.pitch(), 1'066'000'000));
    REQUIRE(encoder.submit(frame.ptr(), frame.pitch(), 1'100'000'000));

    std::vector<EncodedUnit> units;
    REQUIRE(wait_for(
        [&]
        {
            for (auto unit = encoder.poll(); !unit.empty(); unit = encoder.poll())
                units.push_back(std::move(unit));
            return units.size() >= 2;
        }));

    // The first unit opens the stream: SPS, PPS, IDR slice.
    const auto first = split_annex_b(units[0].data);
    REQUIRE(first.size() == 3);
    CHECK(first[0].type == 0x67);
    CHECK(first[1].type == 0x68);
    CHECK(first[2].type == 0x65);
    CHECK(frame_of(first[2]) == 0);

    const auto second = split_annex_b(units[1].data);
    REQUIRE(second.size() == 1);
    CHECK(second[0].type == 0x41);
    CHECK(frame_of(second[0]) == 1);
}

TEST_CASE("encoder returns the capture stamp at microsecond resolution", "[unit][gpu][sensing]")
{
    Frame frame;
    JetsonEncoder encoder(make_config());

    // A V4L2 timeval carries microseconds, so the sub-microsecond part cannot come back.
    const int64_t stamp = 1'234'567'891'234;
    REQUIRE(encoder.submit(frame.ptr(), frame.pitch(), stamp));
    const auto tail = encoder.end_of_stream();
    REQUIRE_FALSE(tail.empty());
    CHECK(tail.timestamp_ns == 1'234'567'891'000);
    CHECK(tail.timestamp_ns != stamp);
}

TEST_CASE("encoder marks an IDR at every GOP boundary", "[unit][gpu][sensing]")
{
    Frame frame;
    auto config = make_config();
    config.gop = 3;
    JetsonEncoder encoder(config);
    // Seven frames against six input buffers: submit() may refuse one until the encoder frees a buffer.
    for (int i = 0; i < 7; ++i)
        REQUIRE(wait_for([&] { return encoder.submit(frame.ptr(), frame.pitch(), 1'000'000'000 + i * 33'000'000); }));

    const auto nals = split_annex_b(encoder.end_of_stream().data);
    std::vector<uint32_t> idr_frames;
    for (const auto& nal : nals)
    {
        if (nal.type == 0x65)
            idr_frames.push_back(frame_of(nal));
    }
    CHECK(idr_frames == std::vector<uint32_t>{ 0, 3, 6 });
}

TEST_CASE("encoder refuses frames when every input buffer is in flight", "[unit][gpu][sensing]")
{
    ScopedEnv latency("SENSING_MOCK_ENCODER_LATENCY_MS", "300");
    Frame frame;
    JetsonEncoder encoder(make_config());

    int accepted = 0;
    int rejected = 0;
    for (int i = 0; i < 10; ++i)
        (encoder.submit(frame.ptr(), frame.pitch(), 1'000'000'000 + i) ? accepted : rejected)++;

    // Six input buffers, and the encoder is too slow to return any in the loop.
    CHECK(accepted == 6);
    CHECK(rejected == 4);

    // Once the encoder frees one, submission works again.
    CHECK(wait_for([&] { return encoder.submit(frame.ptr(), frame.pitch(), 2'000'000'000); }, 2s));
}

TEST_CASE("end of stream returns the tail as one unit carrying the last stamp", "[unit][gpu][sensing]")
{
    Frame frame;
    JetsonEncoder encoder(make_config());
    for (int i = 0; i < 4; ++i)
        REQUIRE(encoder.submit(frame.ptr(), frame.pitch(), 1'000'000'000 + i * 1'000'000));

    const auto tail = encoder.end_of_stream();
    std::vector<uint32_t> frames;
    for (const auto& nal : split_annex_b(tail.data))
    {
        if (is_slice(nal))
            frames.push_back(frame_of(nal));
    }
    CHECK(frames == std::vector<uint32_t>{ 0, 1, 2, 3 });
    CHECK(tail.timestamp_ns == 1'003'000'000);

    CHECK(encoder.poll().empty());
    CHECK(encoder.end_of_stream().empty());
}
