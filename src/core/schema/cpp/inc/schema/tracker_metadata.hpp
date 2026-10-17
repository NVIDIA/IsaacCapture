// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Wire contract for tracker metadata: a table fixed per producer, pushed as samples of a companion
// collection next to the data collection. The producer pushes it when it starts and periodically
// after that, so a consumer that joins late still receives it.

#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace core
{

//! Largest metadata sample a tracker accepts. A producer sizes its metadata collection to its
//! encoded table, up to this.
inline constexpr size_t TRACKER_METADATA_MAX_SIZE = 16384;

//! Collection the metadata of the tracker reading `collection_id` is pushed on.
inline std::string tracker_metadata_collection_id(std::string_view collection_id)
{
    return std::string(collection_id) + "/metadata";
}

//! Name of the one tensor in a metadata collection.
inline constexpr std::string_view TRACKER_METADATA_TENSOR_IDENTIFIER = "metadata";

} // namespace core
