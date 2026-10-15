// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Wire contract for tracker metadata: a table fixed per producer, pushed as samples of a companion
// collection next to the data collection. The producer republishes it for as long as it runs, so a
// consumer that joins late still receives it.

#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace core
{

//! Sample size of a metadata collection, and the largest one consumers accept. 128 KiB samples
//! failed in the CloudXR runtime's IPC; 16 KiB is the largest size in use (noitom_mocap).
inline constexpr size_t TRACKER_METADATA_SAMPLE_SIZE = 16384;

//! Collection the metadata of the tracker reading `collection_id` is pushed on.
inline std::string tracker_metadata_collection_id(std::string_view collection_id)
{
    return std::string(collection_id) + "/metadata";
}

//! Name of the one tensor in a metadata collection.
inline constexpr std::string_view TRACKER_METADATA_TENSOR_IDENTIFIER = "metadata";

} // namespace core
