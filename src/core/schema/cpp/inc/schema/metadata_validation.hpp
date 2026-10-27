// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tactile_metadata_validation.hpp"

#include <stdexcept>
#include <string>
#include <string_view>

namespace core
{

//! Throws on invalid semantics after FlatBuffers verification; schemas without extra rules are a no-op.
template <typename MetadataT>
inline void verify_metadata(const MetadataT&)
{
}

template <>
inline void verify_metadata<TactileMetadata>(const TactileMetadata& metadata)
{
    verify_tactile_metadata(metadata);
}

//! Keep stream context here so live and replay report the same diagnostic.
template <typename MetadataT>
inline void verify_metadata(const MetadataT& metadata, std::string_view name)
{
    try
    {
        verify_metadata(metadata);
    }
    catch (const std::runtime_error& error)
    {
        throw std::runtime_error(std::string(name) + ": " + error.what());
    }
}

} // namespace core
