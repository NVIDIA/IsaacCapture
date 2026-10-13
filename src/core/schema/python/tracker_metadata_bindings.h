// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Python helpers for producers of tracker metadata (see schema/tracker_metadata.hpp). A Python
// producer pushes the chunks with a TensorPushTracker on tracker_metadata_collection_id(), a few
// per frame, in a loop.

#pragma once

#include <pybind11/pybind11.h>
#include <schema/tracker_metadata.hpp>

#include <cstdint>
#include <span>
#include <string>

namespace py = pybind11;

namespace core
{

inline void bind_tracker_metadata(py::module& m)
{
    m.attr("TRACKER_METADATA_CHUNK_SIZE") = TRACKER_METADATA_CHUNK_SIZE;
    m.attr("TRACKER_METADATA_TENSOR_IDENTIFIER") = std::string(TRACKER_METADATA_TENSOR_IDENTIFIER);

    m.def(
        "tracker_metadata_collection_id",
        [](const std::string& collection_id) { return tracker_metadata_collection_id(collection_id); },
        py::arg("collection_id"), "Collection the metadata of the tracker reading `collection_id` is pushed on.");

    m.def(
        "split_tracker_metadata",
        [](const py::bytes& metadata, size_t max_chunk_size)
        {
            const std::string bytes = metadata;
            const auto chunks = split_tracker_metadata(
                std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size()), max_chunk_size);
            py::list out;
            for (const auto& chunk : chunks)
            {
                const auto encoded = chunk.buffer();
                out.append(py::bytes(reinterpret_cast<const char*>(encoded.data()), encoded.size()));
            }
            return out;
        },
        py::arg("metadata"), py::arg("max_chunk_size") = TRACKER_METADATA_CHUNK_SIZE,
        "Split an encoded metadata table (e.g. TactileShape.to_bytes()) into chunk payloads of at most "
        "max_chunk_size bytes each. Raises ValueError on an empty or oversized table.");
}

} // namespace core
