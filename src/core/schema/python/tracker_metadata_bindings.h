// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Python helpers for producers of tracker metadata (see schema/tracker_metadata.hpp). A Python
// producer pushes the encoded table with a TensorPushTracker on tracker_metadata_collection_id(),
// periodically.

#pragma once

#include <pybind11/pybind11.h>
#include <schema/tracker_metadata.hpp>

#include <string>

namespace py = pybind11;

namespace core
{

inline void bind_tracker_metadata(py::module& m)
{
    m.attr("TRACKER_METADATA_SAMPLE_SIZE") = TRACKER_METADATA_SAMPLE_SIZE;
    m.attr("TRACKER_METADATA_TENSOR_IDENTIFIER") = std::string(TRACKER_METADATA_TENSOR_IDENTIFIER);

    m.def(
        "tracker_metadata_collection_id",
        [](const std::string& collection_id) { return tracker_metadata_collection_id(collection_id); },
        py::arg("collection_id"), "Collection the metadata of the tracker reading `collection_id` is pushed on.");
}

} // namespace core
