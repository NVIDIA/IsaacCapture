// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openxr_schema_push_channel.hpp"
#include "schema_push_channel.hpp"

namespace core
{

//! Pushes FlatBuffer schema data into a tensor collection on a caller-owned OpenXR
//! session. Plugins that use an IPluginSession hold ISchemaPushChannel instead.
using SchemaPusher = OpenXRSchemaPushChannel;

} // namespace core
