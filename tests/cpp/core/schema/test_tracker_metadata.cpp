// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for tracker metadata chunking: split_tracker_metadata and TrackerMetadataAssembler.

#include <catch2/catch_test_macros.hpp>
#include <flatbuffers/flatbuffers.h>
#include <schema/tracker_metadata.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// VT_<FIELD> == (id + 2) * 2.
#define VT(field) (field + 2) * 2
static_assert(core::TrackerMetadataChunk::VT_METADATA_HASH == VT(0));
static_assert(core::TrackerMetadataChunk::VT_TOTAL_SIZE == VT(1));
static_assert(core::TrackerMetadataChunk::VT_CHUNK_INDEX == VT(2));
static_assert(core::TrackerMetadataChunk::VT_CHUNK_COUNT == VT(3));
static_assert(core::TrackerMetadataChunk::VT_OFFSET == VT(4));
static_assert(core::TrackerMetadataChunk::VT_BYTES == VT(5));

namespace
{

using Status = core::TrackerMetadataAssembler::Status;

std::vector<uint8_t> make_metadata(size_t size, uint8_t seed = 0)
{
    std::vector<uint8_t> bytes(size);
    for (size_t i = 0; i < size; ++i)
    {
        bytes[i] = static_cast<uint8_t>(i * 31 + seed);
    }
    return bytes;
}

// A chunk built by hand, for headers split_tracker_metadata() would never produce.
core::Serialized<core::TrackerMetadataChunk> make_chunk(
    uint64_t hash, uint32_t total, uint32_t index, uint32_t count, uint32_t offset, const std::vector<uint8_t>& bytes)
{
    flatbuffers::FlatBufferBuilder builder;
    const auto payload = builder.CreateVector(bytes);
    builder.Finish(core::CreateTrackerMetadataChunk(builder, hash, total, index, count, offset, payload));
    return core::Serialized<core::TrackerMetadataChunk>::adopt(builder);
}

} // namespace

TEST_CASE("metadata collection sits next to the data collection", "[unit][tracker_metadata]")
{
    CHECK(core::tracker_metadata_collection_id("tactile/right") == "tactile/right/metadata");
}

TEST_CASE("split_tracker_metadata keeps every chunk within max_chunk_size", "[unit][tracker_metadata]")
{
    constexpr size_t max_chunk_size = 1024;
    constexpr size_t capacity = max_chunk_size - core::TRACKER_METADATA_CHUNK_OVERHEAD;
    // One byte short of, exactly, and one byte past a whole number of chunks.
    for (const size_t size : { 3 * capacity - 1, 3 * capacity, 3 * capacity + 1 })
    {
        const auto metadata = make_metadata(size);
        const auto chunks = core::split_tracker_metadata(metadata, max_chunk_size);

        REQUIRE(chunks.size() == (size + capacity - 1) / capacity);
        for (const auto& chunk : chunks)
        {
            CHECK(chunk.buffer().size() <= max_chunk_size);
            CHECK(chunk->total_size() == size);
            CHECK(chunk->chunk_count() == chunks.size());
            CHECK(chunk->metadata_hash() == core::tracker_metadata_hash(metadata));
        }
    }
}

TEST_CASE("split_tracker_metadata rejects what it cannot carry", "[unit][tracker_metadata]")
{
    CHECK_THROWS_AS(core::split_tracker_metadata({}), std::invalid_argument);
    CHECK_THROWS_AS(
        core::split_tracker_metadata(make_metadata(10), core::TRACKER_METADATA_CHUNK_OVERHEAD), std::invalid_argument);
    CHECK_THROWS_AS(
        core::split_tracker_metadata(make_metadata(10), core::TRACKER_METADATA_CHUNK_SIZE + 1), std::invalid_argument);
    const std::vector<uint8_t> oversized(core::TRACKER_METADATA_MAX_SIZE + 1);
    CHECK_THROWS_AS(core::split_tracker_metadata(oversized), std::invalid_argument);
}

TEST_CASE("assembler restores metadata from chunks in any order, with repeats", "[unit][tracker_metadata]")
{
    const auto metadata = make_metadata(5000);
    const auto chunks = core::split_tracker_metadata(metadata, 1024);
    REQUIRE(chunks.size() == 6);

    core::TrackerMetadataAssembler assembler;
    // Joins mid-loop, sees one chunk twice, then wraps around.
    for (const size_t index : { 3, 4, 4, 5, 0, 1 })
    {
        CHECK(assembler.add(*chunks[index]) == Status::Pending);
    }
    REQUIRE(assembler.add(*chunks[2]) == Status::Complete);
    CHECK(assembler.take() == metadata);

    // take() leaves it empty: the next pass is assembled from scratch.
    CHECK(assembler.add(*chunks[0]) == Status::Pending);
}

TEST_CASE("assembler restarts on chunks of other metadata", "[unit][tracker_metadata]")
{
    const auto first = core::split_tracker_metadata(make_metadata(3000, 1), 1024);
    const auto second_metadata = make_metadata(3000, 2);
    const auto second = core::split_tracker_metadata(second_metadata, 1024);
    REQUIRE(first.size() == second.size());

    core::TrackerMetadataAssembler assembler;
    for (size_t i = 0; i + 1 < first.size(); ++i)
    {
        CHECK(assembler.add(*first[i]) == Status::Pending);
    }
    // The producer changed: the partial first buffer must not be completed with its chunks.
    for (size_t i = 0; i + 1 < second.size(); ++i)
    {
        CHECK(assembler.add(*second[i]) == Status::Pending);
    }
    REQUIRE(assembler.add(*second.back()) == Status::Complete);
    CHECK(assembler.take() == second_metadata);
}

TEST_CASE("assembler rejects inconsistent chunks", "[unit][tracker_metadata]")
{
    core::TrackerMetadataAssembler assembler;
    const std::vector<uint8_t> payload(8, 1);

    SECTION("payload past total_size")
    {
        CHECK(assembler.add(*make_chunk(1, 10, 0, 2, 4, payload)) == Status::Rejected);
    }
    SECTION("index past chunk_count")
    {
        CHECK(assembler.add(*make_chunk(1, 16, 2, 2, 0, payload)) == Status::Rejected);
    }
    SECTION("total_size past the limit")
    {
        const auto total = static_cast<uint32_t>(core::TRACKER_METADATA_MAX_SIZE + 1);
        CHECK(assembler.add(*make_chunk(1, total, 0, 1, 0, payload)) == Status::Rejected);
    }
    SECTION("bytes that do not match the hash")
    {
        CHECK(assembler.add(*make_chunk(1, 8, 0, 1, 0, payload)) == Status::Rejected);
        CHECK(assembler.rejection().find("hash") != std::string::npos);
    }
}
