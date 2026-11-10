// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <video_stage/h264_access_units.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

using video_stage::AccessUnitRange;
using video_stage::split_access_units;

namespace
{

using Bytes = std::vector<std::uint8_t>;

const Bytes kStart4 = { 0, 0, 0, 1 };
const Bytes kStart3 = { 0, 0, 1 };

/// A NAL with the given start code, header byte and payload (payloads here never contain 00 00 0x).
Bytes nal(const Bytes& start, std::uint8_t header, const Bytes& payload)
{
    Bytes out = start;
    out.push_back(header);
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

// first_mb_in_slice is the Exp-Golomb code at the start of the slice data: '1' is 0, '010' is 1, '011' is 2.
const Bytes kFirstSlice = { 0x88, 0x84, 0x21 };
const Bytes kSecondSlice = { 0x48, 0x84, 0x21 };
const Bytes kThirdSlice = { 0x68, 0x84, 0x21 };

Bytes sps(const Bytes& start = kStart4)
{
    return nal(start, 0x67, { 0x64, 0x00, 0x1f, 0xac });
}
Bytes pps(const Bytes& start = kStart4)
{
    return nal(start, 0x68, { 0xeb, 0xc3, 0xcb });
}
Bytes idr(const Bytes& start = kStart4, const Bytes& slice = kFirstSlice)
{
    return nal(start, 0x65, slice);
}
Bytes p_slice(const Bytes& start = kStart3, const Bytes& slice = kFirstSlice)
{
    return nal(start, 0x41, slice);
}
Bytes sei(const Bytes& start = kStart3)
{
    return nal(start, 0x06, { 0x05, 0x02, 0xaa, 0xbb, 0x80 });
}
Bytes filler(const Bytes& start = kStart3)
{
    return nal(start, 0x0c, { 0xff, 0xff, 0x80 });
}

Bytes concat(std::initializer_list<Bytes> parts)
{
    Bytes out;
    for (const auto& part : parts)
        out.insert(out.end(), part.begin(), part.end());
    return out;
}

std::vector<AccessUnitRange> split(const Bytes& stream)
{
    return split_access_units(stream.data(), stream.size());
}

} // namespace

TEST_CASE("an empty or start-code-free input has no access units", "[h264_access_units]")
{
    CHECK(split_access_units(nullptr, 0).empty());
    CHECK(split(Bytes{}).empty());
    CHECK(split(Bytes{ 1, 2, 3, 4, 5, 6 }).empty());
    CHECK(split(Bytes{ 0, 0, 1 }).empty()); // a start code with no NAL after it
}

TEST_CASE("parameter sets stay with the IDR that follows", "[h264_access_units]")
{
    const Bytes a = concat({ sps(), pps(), idr() });
    const Bytes b = p_slice();
    const Bytes c = p_slice();
    const auto units = split(concat({ a, b, c }));

    REQUIRE(units.size() == 3);
    CHECK(units[0].offset == 0);
    CHECK(units[0].size == a.size());
    CHECK(units[0].idr);
    CHECK(units[1].offset == a.size());
    CHECK(units[1].size == b.size());
    CHECK_FALSE(units[1].idr);
    CHECK(units[2].offset == a.size() + b.size());
    CHECK(units[2].size == c.size());
    CHECK_FALSE(units[2].idr);
}

TEST_CASE("the units tile the stream exactly", "[h264_access_units]")
{
    const Bytes stream = concat({ sps(), pps(), idr(), p_slice(), sei(), p_slice(), p_slice() });
    const auto units = split(stream);
    REQUIRE(units.size() == 4);
    std::size_t expected_offset = 0;
    for (const auto& unit : units)
    {
        CHECK(unit.offset == expected_offset);
        expected_offset += unit.size;
    }
    CHECK(expected_offset == stream.size());
}

TEST_CASE("slices of one picture stay together", "[h264_access_units]")
{
    const Bytes first =
        concat({ sps(), pps(), idr(kStart4, kFirstSlice), idr(kStart3, kSecondSlice), idr(kStart3, kThirdSlice) });
    const Bytes second = concat({ p_slice(kStart3, kFirstSlice), p_slice(kStart3, kSecondSlice) });
    const auto units = split(concat({ first, second }));

    REQUIRE(units.size() == 2);
    CHECK(units[0].size == first.size());
    CHECK(units[0].idr);
    CHECK(units[1].offset == first.size());
    CHECK(units[1].size == second.size());
    CHECK_FALSE(units[1].idr);
}

TEST_CASE("a new first slice directly after a picture starts a new unit", "[h264_access_units]")
{
    const Bytes p1 = concat({ p_slice(kStart3, kFirstSlice), p_slice(kStart3, kSecondSlice) });
    const Bytes p2 = p_slice(kStart3, kFirstSlice);
    const auto units = split(concat({ p1, p2 }));
    REQUIRE(units.size() == 2);
    CHECK(units[0].size == p1.size());
    CHECK(units[1].size == p2.size());
}

TEST_CASE("SEI and an access unit delimiter before a picture belong to that picture", "[h264_access_units]")
{
    const Bytes p0 = p_slice();
    const Bytes aud = nal(kStart3, 0x09, { 0xf0 });
    const Bytes next = concat({ aud, sei(), p_slice() });
    const auto units = split(concat({ p0, next }));

    REQUIRE(units.size() == 2);
    CHECK(units[0].size == p0.size());
    CHECK(units[1].offset == p0.size());
    CHECK(units[1].size == next.size());
}

TEST_CASE("trailing filler, end of sequence and end of stream stay with the previous picture", "[h264_access_units]")
{
    SECTION("filler")
    {
        const Bytes first = concat({ p_slice(), filler() });
        const Bytes second = p_slice();
        const auto units = split(concat({ first, second }));
        REQUIRE(units.size() == 2);
        CHECK(units[0].size == first.size());
        CHECK(units[1].offset == first.size());
    }
    SECTION("end of sequence then the next IDR's parameter sets")
    {
        const Bytes first = concat({
            sps(),
            pps(),
            idr(),
            nal(kStart3, 0x0a, {}),
        });
        const Bytes second = concat({ sps(), pps(), idr() });
        const auto units = split(concat({ first, second }));
        REQUIRE(units.size() == 2);
        CHECK(units[0].size == first.size());
        CHECK(units[1].offset == first.size());
        CHECK(units[1].idr);
    }
    SECTION("end of stream at the very end")
    {
        const Bytes stream = concat({ p_slice(), nal(kStart4, 0x0b, {}) });
        const auto units = split(stream);
        REQUIRE(units.size() == 1);
        CHECK(units[0].size == stream.size());
    }
    SECTION("several fillers in a row")
    {
        const Bytes first = concat({ p_slice(), filler(), filler() });
        const auto units = split(concat({ first, p_slice() }));
        REQUIRE(units.size() == 2);
        CHECK(units[0].size == first.size());
    }
}

TEST_CASE("three- and four-byte start codes both work, and a four-byte code's leading zero begins its unit",
          "[h264_access_units]")
{
    const Bytes first = concat(
        { nal(kStart3, 0x67, { 0x64, 0x00, 0x1f }), nal(kStart3, 0x68, { 0xeb }), nal(kStart3, 0x65, kFirstSlice) });
    const Bytes second = p_slice(kStart4);
    const Bytes third = p_slice(kStart3);
    const auto units = split(concat({ first, second, third }));

    REQUIRE(units.size() == 3);
    CHECK(units[0].size == first.size());
    CHECK(units[1].offset == first.size()); // includes the zero byte of the 4-byte code
    CHECK(units[1].size == second.size());
    CHECK(units[2].offset == first.size() + second.size());
    CHECK(units[2].size == third.size());
}

TEST_CASE("only an IDR slice marks a unit as a keyframe", "[h264_access_units]")
{
    const auto units = split(concat({ sps(), pps(), idr(), p_slice(), idr(kStart4), p_slice() }));
    REQUIRE(units.size() == 4);
    CHECK(units[0].idr);
    CHECK_FALSE(units[1].idr);
    CHECK(units[2].idr);
    CHECK_FALSE(units[3].idr);
}

TEST_CASE("a single NAL is one unit", "[h264_access_units]")
{
    const Bytes stream = p_slice(kStart4);
    const auto units = split(stream);
    REQUIRE(units.size() == 1);
    CHECK(units[0].offset == 0);
    CHECK(units[0].size == stream.size());
}
