// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <video_stage/avs1.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace avs1 = video_stage::avs1;

namespace
{

// Record offsets (after the 4-byte length prefix).
constexpr size_t kStream = 64;
constexpr size_t kSample = 64 + 40;
constexpr size_t kPayload = 64 + 40 + 72;

uint64_t be(const std::vector<uint8_t>& b, size_t off, size_t n)
{
    uint64_t v = 0;
    for (size_t i = 0; i < n; ++i)
        v = (v << 8) | b.at(off + i);
    return v;
}

void set_be(std::vector<uint8_t>& b, size_t off, size_t n, uint64_t v)
{
    for (size_t i = 0; i < n; ++i)
        b.at(off + n - 1 - i) = static_cast<uint8_t>(v >> (8 * i));
}

const std::vector<uint8_t> kAu = { 0, 0, 0, 1, 0x67, 1, 2, 0, 0, 0, 1, 0x68, 3, 0, 0, 1, 0x65, 9, 9, 9 };

avs1::Sample make_sample()
{
    avs1::Sample s;
    s.codec = avs1::Codec::H264;
    s.width = 1280;
    s.height = 720;
    s.fps_numerator = 60;
    s.bundle_sequence = 0x0102030405060708ULL;
    s.session_id = 0x1112131415161718ULL;
    s.frame_number = 0x2122232425262728ULL;
    s.frame_timestamp_ns = 0x3132333435363738ULL;
    return s;
}

// Encoded record without the length prefix.
std::vector<uint8_t> valid_record(const std::vector<uint8_t>& payload = kAu)
{
    std::vector<uint8_t> out;
    avs1::encode(make_sample(), payload.data(), payload.size(), out);
    return std::vector<uint8_t>(out.begin() + 4, out.end());
}

bool rejects(const std::vector<uint8_t>& rec)
{
    std::string error;
    const auto r = avs1::parse(rec.data(), rec.size(), error);
    return !r.has_value() && !error.empty();
}

} // namespace

TEST_CASE("AVS1 round trip keeps every field", "[video_stage_wire]")
{
    const avs1::Sample s = make_sample();
    std::vector<uint8_t> out;
    avs1::encode(s, kAu.data(), kAu.size(), out);
    REQUIRE(avs1::read_length(out.data()) == out.size() - 4);

    std::string error;
    const auto r = avs1::parse(out.data() + 4, out.size() - 4, error);
    REQUIRE(r.has_value());
    CHECK(r->sample.codec == avs1::Codec::H264);
    CHECK(r->sample.width == s.width);
    CHECK(r->sample.height == s.height);
    CHECK(r->sample.fps_numerator == s.fps_numerator);
    CHECK(r->sample.bundle_sequence == s.bundle_sequence);
    CHECK(r->sample.session_id == s.session_id);
    CHECK(r->sample.frame_number == s.frame_number);
    CHECK(r->sample.frame_timestamp_ns == s.frame_timestamp_ns);
    REQUIRE(r->payload_size == kAu.size());
    CHECK(std::memcmp(r->payload, kAu.data(), kAu.size()) == 0);
    CHECK(r->payload == out.data() + 4 + kPayload);
}

TEST_CASE("AVS1 encode follows WIRE_FORMAT.md byte for byte", "[video_stage_wire]")
{
    const avs1::Sample s = make_sample();
    std::vector<uint8_t> out;
    avs1::encode(s, kAu.data(), kAu.size(), out);
    REQUIRE(out.size() == 4 + 64 + 40 + 72 + kAu.size());
    CHECK(be(out, 0, 4) == out.size() - 4);

    const std::vector<uint8_t> r(out.begin() + 4, out.end());
    // HEADER "!4sHHIHHQQQQIIQ"
    CHECK(std::memcmp(r.data(), "AVS1", 4) == 0);
    CHECK(be(r, 4, 2) == 1);
    CHECK(be(r, 6, 2) == 0);
    CHECK(be(r, 8, 4) == 64);
    CHECK(be(r, 12, 2) == 1);
    CHECK(be(r, 14, 2) == 1);
    CHECK(be(r, 16, 8) == s.bundle_sequence);
    CHECK(be(r, 24, 8) == s.session_id);
    CHECK(be(r, 32, 8) == 0);
    CHECK(be(r, 40, 8) == 0);
    CHECK(be(r, 48, 4) == 112);
    CHECK(be(r, 52, 4) == kAu.size());
    CHECK(be(r, 56, 8) == 0);
    // STREAM "!HHBBHIIIIQHHI"
    CHECK(be(r, kStream + 0, 2) == 0);
    CHECK(be(r, kStream + 2, 2) == 1);
    CHECK(r[kStream + 4] == 2);
    CHECK(r[kStream + 5] == 1);
    CHECK(be(r, kStream + 6, 2) == 0);
    CHECK(be(r, kStream + 8, 4) == s.width);
    CHECK(be(r, kStream + 12, 4) == s.height);
    CHECK(be(r, kStream + 16, 4) == s.fps_numerator);
    CHECK(be(r, kStream + 20, 4) == 1);
    CHECK(be(r, kStream + 24, 8) != 0);
    CHECK(be(r, kStream + 32, 2) == 0);
    CHECK(be(r, kStream + 34, 2) == 1);
    CHECK(be(r, kStream + 36, 4) == 0);
    // SAMPLE "!HBBHHIIQQQQQQII"
    CHECK(be(r, kSample + 0, 2) == 0);
    CHECK(r[kSample + 2] == 0);
    CHECK(r[kSample + 3] == 0);
    CHECK(be(r, kSample + 4, 2) == 0x0005);
    CHECK(be(r, kSample + 6, 2) == 0);
    CHECK(be(r, kSample + 16, 8) == s.frame_number);
    CHECK(be(r, kSample + 32, 8) == s.frame_timestamp_ns);
    CHECK(be(r, kSample + 64, 4) == 0);
    CHECK(be(r, kSample + 68, 4) == kAu.size());
    CHECK(std::memcmp(r.data() + kPayload, kAu.data(), kAu.size()) == 0);
}

TEST_CASE("AVS1 read_length is big endian", "[video_stage_wire]")
{
    const uint8_t p[4] = { 0x01, 0x02, 0x03, 0x04 };
    CHECK(avs1::read_length(p) == 0x01020304U);
    const uint8_t q[4] = { 0xFF, 0xFF, 0xFF, 0xFF };
    CHECK(avs1::read_length(q) == 0xFFFFFFFFU);
}

TEST_CASE("AVS1 parse rejects malformed records", "[video_stage_wire]")
{
    const std::vector<uint8_t> good = valid_record();
    {
        std::string error;
        REQUIRE(avs1::parse(good.data(), good.size(), error).has_value());
    }

    struct Mutation
    {
        const char* name;
        size_t offset;
        size_t bytes;
        uint64_t value;
    };
    const Mutation mutations[] = {
        { "bad magic", 0, 1, 'X' },
        { "version 2", 4, 2, 2 },
        { "version 0", 4, 2, 0 },
        { "bundle flags", 6, 2, 1 },
        { "bundle flags TAI", 6, 2, 2 },
        { "header_bytes", 8, 4, 65 },
        { "stream_count 0", 12, 2, 0 },
        { "stream_count 2", 12, 2, 2 },
        { "sample_count 0", 14, 2, 0 },
        { "sample_count 2", 14, 2, 2 },
        { "descriptor_bytes small", 48, 4, 111 },
        { "descriptor_bytes large", 48, 4, 113 },
        { "payload_bytes small", 52, 4, kAu.size() - 1 },
        { "payload_bytes large", 52, 4, kAu.size() + 1 },
        { "sender field offset 32", 32, 8, 1 },
        { "sender field offset 40", 40, 8, 1 },
        { "sender field offset 56", 56, 8, 1 },
        { "stream absent", kStream + 2, 2, 0 },
        { "stream stereo flag", kStream + 2, 2, 3 },
        { "codec raw", kStream + 4, 1, 0 },
        { "codec hevc", kStream + 4, 1, 1 },
        { "codec 3", kStream + 4, 1, 3 },
        { "pixel format 0", kStream + 5, 1, 0 },
        { "pixel format 2", kStream + 5, 1, 2 },
        { "rotation 180", kStream + 6, 2, 180 },
        { "fps denominator 0", kStream + 20, 4, 0 },
        { "first_sample 1", kStream + 32, 2, 1 },
        { "stream sample_count 0", kStream + 34, 2, 0 },
        { "stream sample_count 2", kStream + 34, 2, 2 },
        { "stream reserved", kStream + 36, 4, 1 },
        { "sample unknown flag 0x10", kSample + 4, 2, 0x0015 },
        { "sample unknown flag 0x8000", kSample + 4, 2, 0x8005 },
        { "sample not present", kSample + 4, 2, 0x0004 },
        { "sample no timestamp valid", kSample + 4, 2, 0x0001 },
        { "sample flags clear", kSample + 4, 2, 0 },
        { "sample reserved", kSample + 6, 2, 1 },
        { "payload length 0", kSample + 68, 4, 0 },
        { "payload offset 1", kSample + 64, 4, 1 },
        { "payload length mismatch", kSample + 68, 4, kAu.size() - 1 },
    };
    for (const Mutation& m : mutations)
    {
        DYNAMIC_SECTION(m.name)
        {
            std::vector<uint8_t> rec = good;
            set_be(rec, m.offset, m.bytes, m.value);
            CHECK(rejects(rec));
        }
    }

    SECTION("optional sample flags are accepted")
    {
        std::vector<uint8_t> rec = good;
        set_be(rec, kSample + 4, 2, 0x000F);
        std::string error;
        CHECK(avs1::parse(rec.data(), rec.size(), error).has_value());
    }
    SECTION("record size does not match payload_bytes")
    {
        std::vector<uint8_t> rec = good;
        rec.push_back(0);
        CHECK(rejects(rec));
        rec = good;
        rec.pop_back();
        CHECK(rejects(rec));
    }
    SECTION("empty payload")
    {
        std::vector<uint8_t> rec(good.begin(), good.begin() + kPayload);
        set_be(rec, 52, 4, 0);
        set_be(rec, kSample + 68, 4, 0);
        CHECK(rejects(rec));
    }
    SECTION("short record")
    {
        CHECK(rejects(std::vector<uint8_t>{}));
        CHECK(rejects(std::vector<uint8_t>(good.begin(), good.begin() + 10)));
    }
}

TEST_CASE("AVS1 rejects every proper prefix of a valid record", "[video_stage_wire]")
{
    const std::vector<uint8_t> good = valid_record();
    for (size_t n = 0; n < good.size(); ++n)
    {
        std::string error;
        const auto r = avs1::parse(good.data(), n, error);
        INFO("prefix length " << n);
        REQUIRE_FALSE(r.has_value());
        REQUIRE_FALSE(error.empty());
    }
}

TEST_CASE("AVS1 parse survives single-byte corruption anywhere", "[video_stage_wire]")
{
    const std::vector<uint8_t> good = valid_record();
    uint32_t state = 12345;
    for (int i = 0; i < 4000; ++i)
    {
        state = state * 1664525U + 1013904223U;
        std::vector<uint8_t> rec = good;
        rec[(state >> 8) % rec.size()] ^= static_cast<uint8_t>(1U + (state >> 24) % 255U);
        std::string error;
        const auto r = avs1::parse(rec.data(), rec.size(), error);
        if (r)
        {
            REQUIRE(r->payload >= rec.data());
            REQUIRE(r->payload + r->payload_size <= rec.data() + rec.size());
        }
    }
}

TEST_CASE("AVS1 contains_idr", "[video_stage_wire]")
{
    SECTION("3-byte start code IDR")
    {
        const uint8_t au[] = { 0, 0, 1, 0x65, 1, 2, 3 };
        CHECK(avs1::contains_idr(au, sizeof(au)));
    }
    SECTION("4-byte start code IDR")
    {
        const uint8_t au[] = { 0, 0, 0, 1, 0x65, 1, 2, 3 };
        CHECK(avs1::contains_idr(au, sizeof(au)));
    }
    SECTION("SPS, PPS, IDR")
    {
        CHECK(avs1::contains_idr(kAu.data(), kAu.size()));
    }
    SECTION("IDR with other nal_ref_idc")
    {
        const uint8_t au[] = { 0, 0, 0, 1, 0x25, 1 };
        CHECK(avs1::contains_idr(au, sizeof(au)));
    }
    SECTION("SPS and PPS only")
    {
        const uint8_t au[] = { 0, 0, 0, 1, 0x67, 1, 2, 0, 0, 0, 1, 0x68, 3 };
        CHECK_FALSE(avs1::contains_idr(au, sizeof(au)));
    }
    SECTION("non-IDR slice")
    {
        const uint8_t au[] = { 0, 0, 0, 1, 0x41, 1, 2, 3 };
        CHECK_FALSE(avs1::contains_idr(au, sizeof(au)));
    }
    SECTION("empty and tiny inputs")
    {
        CHECK_FALSE(avs1::contains_idr(nullptr, 0));
        const uint8_t au[] = { 0, 0, 1 };
        CHECK_FALSE(avs1::contains_idr(au, sizeof(au)));
    }
}
