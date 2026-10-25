// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mock_test_support.hpp"

#include <catch2/catch_test_macros.hpp>
#include <core/jetson_encoder.hpp>

#include <cuda_runtime.h>
#include <random>
#include <stdexcept>
#include <vector>

using namespace plugins::sensing;
using namespace sensing_test;
using namespace std::chrono_literals;

namespace
{

constexpr uint32_t kWidth = 320;
constexpr uint32_t kHeight = 240;

/// A solid gray RGBA frame on the device.
class Frame
{
public:
    Frame()
    {
        cudaMallocPitch(reinterpret_cast<void**>(&m_ptr), &m_pitch, kWidth * 4, kHeight);
        set(0x80);
    }
    ~Frame()
    {
        cudaFree(m_ptr);
    }
    void set(int gray)
    {
        cudaMemset2D(m_ptr, m_pitch, gray, kWidth * 4, kHeight);
    }
    /// Different random pixels for each seed: content no encoder can predict, so the bitrate target decides
    /// how large a unit is.
    void randomize(uint32_t seed)
    {
        std::mt19937 rng(seed);
        std::vector<uint8_t> host(static_cast<size_t>(kWidth) * kHeight * 4);
        for (auto& byte : host)
            byte = static_cast<uint8_t>(rng());
        cudaMemcpy2D(m_ptr, m_pitch, host.data(), kWidth * 4, kWidth * 4, kHeight, cudaMemcpyHostToDevice);
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
    config.bitrate_bps = 4'000'000;
    config.peak_bitrate_bps = 6'000'000;
    return config;
}

// Submits `count` frames, waiting out a full input pool, then returns the whole stream.
std::vector<uint8_t> encode_stream(JetsonEncoder& encoder, Frame& frame, int count, int64_t first_stamp_ns = 1'000'000'000)
{
    std::vector<uint8_t> stream;
    for (int i = 0; i < count; ++i)
    {
        REQUIRE(wait_for([&] { return encoder.submit(frame.ptr(), frame.pitch(), first_stamp_ns + i * 33'333'000); }));
        for (auto unit = encoder.poll(); !unit.empty(); unit = encoder.poll())
            stream.insert(stream.end(), unit.data.begin(), unit.data.end());
    }
    const auto tail = encoder.end_of_stream();
    stream.insert(stream.end(), tail.data.begin(), tail.data.end());
    return stream;
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

TEST_CASE("the stream opens with SPS, PPS and an IDR slice, High profile at Level 5.2", "[unit][gpu][sensing]")
{
    Frame frame;
    JetsonEncoder encoder(make_config());
    std::vector<Nal> nals;
    for (auto& nal : split_annex_b(encode_stream(encoder, frame, 3)))
    {
        if (nal.type != kNalSei) // the encoder may add SEI; it is not part of the shape under test
            nals.push_back(std::move(nal));
    }

    REQUIRE(nals.size() >= 4);
    CHECK(nals[0].type == kNalSps);
    CHECK(nals[1].type == kNalPps);
    CHECK(nals[2].type == kNalIdr);
    CHECK(nals[3].type == kNalSlice);
    REQUIRE(nals[0].payload.size() >= 3);
    CHECK(nals[0].payload[0] == 100); // profile_idc: High
    CHECK(nals[0].payload[2] == 52); // level_idc: 5.2
}

TEST_CASE("the output decodes, in order, at the configured size", "[unit][gpu][sensing]")
{
    Frame frame;
    JetsonEncoder encoder(make_config());

    // Four frames of rising brightness, so order and content can both be read back.
    std::vector<uint8_t> stream;
    for (int i = 0; i < 4; ++i)
    {
        frame.set(40 + 50 * i);
        REQUIRE(wait_for(
            [&] {
                return encoder.submit(frame.ptr(), frame.pitch(), int64_t{ 1'000'000'000 } + i * int64_t{ 33'333'000 });
            }));
        // The next frame is only written once the copy of this one is done: submit() has returned.
    }
    const auto tail = encoder.end_of_stream();
    stream.insert(stream.end(), tail.data.begin(), tail.data.end());

    const auto pictures = decode_annex_b(stream);
    REQUIRE(pictures.size() == 4);
    for (size_t i = 0; i < pictures.size(); ++i)
    {
        CHECK(pictures[i].width == static_cast<int>(kWidth));
        CHECK(pictures[i].height == static_cast<int>(kHeight));
        if (i > 0)
            CHECK(pictures[i].luma > pictures[i - 1].luma + 20);
    }
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

TEST_CASE("encoder marks an IDR at every GOP boundary, with parameter sets in front", "[unit][gpu][sensing]")
{
    Frame frame;
    auto config = make_config();
    config.gop = 3;
    JetsonEncoder encoder(config);

    std::vector<uint8_t> slice_types;
    size_t parameter_sets = 0;
    for (const auto& nal : split_annex_b(encode_stream(encoder, frame, 7)))
    {
        if (is_slice(nal))
            slice_types.push_back(nal.type);
        parameter_sets += nal.type == kNalSps ? 1 : 0;
    }
    CHECK(slice_types == std::vector<uint8_t>{ kNalIdr, kNalSlice, kNalSlice, kNalIdr, kNalSlice, kNalSlice, kNalIdr });
    CHECK(parameter_sets == 3);
}

TEST_CASE("end of stream returns the tail as one unit carrying the last stamp", "[unit][gpu][sensing]")
{
    Frame frame;
    JetsonEncoder encoder(make_config());
    for (int i = 0; i < 4; ++i)
        REQUIRE(encoder.submit(frame.ptr(), frame.pitch(), 1'000'000'000 + i * 1'000'000));

    const auto tail = encoder.end_of_stream();
    CHECK(tail.timestamp_ns == 1'003'000'000);
    size_t slices = 0;
    for (const auto& nal : split_annex_b(tail.data))
        slices += is_slice(nal) ? 1 : 0;
    CHECK(slices >= 1);

    CHECK(encoder.poll().empty());
    CHECK(encoder.end_of_stream().empty());
}

TEST_CASE("a lower bitrate target changes the encoded output and the stream still decodes", "[unit][gpu][sensing]")
{
    Frame frame;
    auto config = make_config();
    config.bitrate_bps = 4'000'000;
    config.peak_bitrate_bps = 6'000'000;
    JetsonEncoder encoder(config);

    constexpr int kBefore = 20;
    constexpr int kAfter = 40;
    std::vector<EncodedUnit> units;
    for (int i = 0; i < kBefore + kAfter; ++i)
    {
        if (i == kBefore)
            REQUIRE(encoder.set_bitrate(400'000));
        frame.randomize(1000 + i);
        REQUIRE(wait_for(
            [&] {
                return encoder.submit(frame.ptr(), frame.pitch(), int64_t{ 1'000'000'000 } + i * int64_t{ 33'333'000 });
            }));
        std::this_thread::sleep_for(5ms);
        for (auto unit = encoder.poll(); !unit.empty(); unit = encoder.poll())
            units.push_back(std::move(unit));
    }
    for (auto& unit : encoder.drain())
        units.push_back(std::move(unit));
    REQUIRE(units.size() == kBefore + kAfter);

    auto mean_size = [&](int from, int to)
    {
        double total = 0;
        for (int i = from; i < to; ++i)
            total += static_cast<double>(units[i].data.size());
        return total / (to - from);
    };
    const double before = mean_size(3, kBefore);
    const double after = mean_size(kBefore + 20, kBefore + kAfter); // after the rate control has settled
    INFO("before " << before << " bytes, after " << after << " bytes");
    CHECK(after < before * 0.4);

    std::vector<uint8_t> stream;
    for (const auto& unit : units)
        stream.insert(stream.end(), unit.data.begin(), unit.data.end());
    CHECK(decode_annex_b(stream).size() == units.size());
}

TEST_CASE("a keyframe request makes the next frame an IDR with parameter sets", "[unit][gpu][sensing]")
{
    Frame frame;
    JetsonEncoder encoder(make_config()); // GOP of 150: no IDR but the first, unless asked for
    std::vector<EncodedUnit> units;
    for (int i = 0; i < 6; ++i)
    {
        if (i == 4)
            encoder.request_keyframe();
        REQUIRE(wait_for(
            [&] {
                return encoder.submit(frame.ptr(), frame.pitch(), int64_t{ 1'000'000'000 } + i * int64_t{ 33'333'000 });
            }));
        std::this_thread::sleep_for(20ms);
    }
    for (auto& unit : encoder.drain())
        units.push_back(std::move(unit));
    REQUIRE(units.size() == 6);

    std::vector<bool> keyframes;
    for (const auto& unit : units)
        keyframes.push_back(unit.keyframe);
    CHECK(keyframes == std::vector<bool>{ true, false, false, false, true, false });
    const auto nals = split_annex_b(units[4].data);
    REQUIRE(nals.size() >= 3);
    CHECK(nals[0].type == kNalSps);
}
