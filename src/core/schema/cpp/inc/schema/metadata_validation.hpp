// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tactile_metadata_validation.hpp"

namespace core
{

//! Semantic verification after FlatBuffers verification; schemas without extra rules pass.
template <typename MetadataT>
[[nodiscard]] inline bool verify_metadata(const MetadataT&)
{
    return true;
}

template <>
[[nodiscard]] inline bool verify_metadata<TactileMetadata>(const TactileMetadata& metadata)
{
    return verify_tactile_metadata(metadata);
}

} // namespace core
