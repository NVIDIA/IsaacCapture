// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Wire contract for tracker metadata: a table fixed per producer, possibly larger than one tensor
// sample, so it travels as TrackerMetadataChunk samples on a companion collection.
//
// The producer pushes every chunk in a loop for as long as it runs, so a consumer that joins late
// or drops samples completes the buffer on a later pass.

#pragma once

#include "serialized.hpp"

#include <schema/tracker_metadata_generated.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace core
{

//! Chunk sample size producers use by default, and the largest one consumers accept. 128 KiB
//! samples failed in the CloudXR runtime's IPC; 16 KiB is the largest size in use (noitom_mocap).
inline constexpr size_t TRACKER_METADATA_CHUNK_SIZE = 16384;

//! Largest metadata buffer a consumer reassembles; bounds what a chunk header can make it allocate.
inline constexpr size_t TRACKER_METADATA_MAX_SIZE = 16 * 1024 * 1024;

//! Bytes of an encoded chunk that are not payload: root offset, vtable, scalars, vector length
//! and alignment. About 80 in practice.
inline constexpr size_t TRACKER_METADATA_CHUNK_OVERHEAD = 128;

//! Collection the metadata of the tracker reading `collection_id` is pushed on.
inline std::string tracker_metadata_collection_id(std::string_view collection_id)
{
    return std::string(collection_id) + "/metadata";
}

//! Name of the one tensor in a metadata collection.
inline constexpr std::string_view TRACKER_METADATA_TENSOR_IDENTIFIER = "metadata";

//! FNV-1a 64 over `bytes`.
inline uint64_t tracker_metadata_hash(std::span<const uint8_t> bytes)
{
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (const uint8_t byte : bytes)
    {
        hash ^= byte;
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

/*!
 * @brief Splits an encoded metadata buffer into chunks that each encode to at most `max_chunk_size` bytes.
 * @throws std::invalid_argument when `metadata` is empty or larger than TRACKER_METADATA_MAX_SIZE,
 *         or when `max_chunk_size` is outside the protocol's supported range.
 */
inline std::vector<Serialized<TrackerMetadataChunk>> split_tracker_metadata(
    std::span<const uint8_t> metadata, size_t max_chunk_size = TRACKER_METADATA_CHUNK_SIZE)
{
    if (metadata.empty() || metadata.size() > TRACKER_METADATA_MAX_SIZE)
    {
        throw std::invalid_argument("split_tracker_metadata: metadata size " + std::to_string(metadata.size()) +
                                    " is outside 1.." + std::to_string(TRACKER_METADATA_MAX_SIZE));
    }
    if (max_chunk_size <= TRACKER_METADATA_CHUNK_OVERHEAD)
    {
        throw std::invalid_argument("split_tracker_metadata: max_chunk_size must exceed " +
                                    std::to_string(TRACKER_METADATA_CHUNK_OVERHEAD));
    }
    if (max_chunk_size > TRACKER_METADATA_CHUNK_SIZE)
    {
        throw std::invalid_argument("split_tracker_metadata: max_chunk_size must not exceed " +
                                    std::to_string(TRACKER_METADATA_CHUNK_SIZE));
    }

    const size_t capacity = max_chunk_size - TRACKER_METADATA_CHUNK_OVERHEAD;
    const auto count = static_cast<uint32_t>((metadata.size() - 1) / capacity + 1);
    const uint64_t hash = tracker_metadata_hash(metadata);

    std::vector<Serialized<TrackerMetadataChunk>> chunks;
    chunks.reserve(count);
    flatbuffers::FlatBufferBuilder builder(max_chunk_size);
    for (uint32_t index = 0; index < count; ++index)
    {
        const size_t offset = index * capacity;
        const size_t size = std::min(capacity, metadata.size() - offset);
        const auto bytes = builder.CreateVector(metadata.data() + offset, size);
        builder.Finish(CreateTrackerMetadataChunk(
            builder, hash, static_cast<uint32_t>(metadata.size()), index, count, static_cast<uint32_t>(offset), bytes));
        assert(builder.GetSize() <= max_chunk_size && "TRACKER_METADATA_CHUNK_OVERHEAD is too small");
        chunks.push_back(Serialized<TrackerMetadataChunk>::adopt(builder));
    }
    return chunks;
}

/*!
 * @brief Reassembles a metadata buffer from chunks that arrive in any order and repeat.
 *
 * A chunk whose hash, total size or chunk count differs from the assembly in progress belongs to
 * other metadata and restarts the assembly. The finished buffer is checked against its hash.
 */
class TrackerMetadataAssembler
{
public:
    enum class Status
    {
        Pending,
        Complete,
        Rejected
    };

    //! Adds one chunk. After Complete, take() returns the buffer; after Rejected, rejection() says why.
    Status add(const TrackerMetadataChunk& chunk)
    {
        const auto* bytes = chunk.bytes();
        const uint32_t total = chunk.total_size();
        const uint32_t count = chunk.chunk_count();
        if (bytes == nullptr || bytes->size() == 0 || total > TRACKER_METADATA_MAX_SIZE || count == 0 ||
            count > total || chunk.chunk_index() >= count || uint64_t{ chunk.offset() } + bytes->size() > total)
        {
            return reject("chunk header does not fit its payload");
        }

        if (chunk.metadata_hash() != hash_ || total != bytes_.size() || count != received_.size())
        {
            hash_ = chunk.metadata_hash();
            bytes_.assign(total, 0);
            received_.assign(count, false);
            missing_ = count;
        }

        if (!received_[chunk.chunk_index()])
        {
            std::copy(bytes->begin(), bytes->end(), bytes_.begin() + chunk.offset());
            received_[chunk.chunk_index()] = true;
            --missing_;
        }
        if (missing_ > 0)
        {
            return Status::Pending;
        }
        if (tracker_metadata_hash(bytes_) != hash_)
        {
            reset();
            return reject("reassembled metadata does not match its hash");
        }
        return Status::Complete;
    }

    //! The buffer the last add() completed. Leaves the assembler empty.
    std::vector<uint8_t> take()
    {
        std::vector<uint8_t> out = std::move(bytes_);
        reset();
        return out;
    }

    //! Drops the assembly in progress.
    void reset()
    {
        hash_ = 0;
        bytes_.clear();
        received_.clear();
        missing_ = 0;
    }

    const std::string& rejection() const
    {
        return rejection_;
    }

private:
    Status reject(std::string reason)
    {
        rejection_ = std::move(reason);
        return Status::Rejected;
    }

    uint64_t hash_ = 0;
    std::vector<uint8_t> bytes_;
    std::vector<bool> received_;
    uint32_t missing_ = 0;
    std::string rejection_;
};

} // namespace core
