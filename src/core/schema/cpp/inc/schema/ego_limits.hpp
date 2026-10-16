// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace core
{

// Shared by EGO SchemaPushers and trackers. 32 KiB accommodates the
// 32-sample IMU batches and EGO calibration YAML while remaining below the
// CloudXR runtime's IPC tensor-payload limit.
inline constexpr size_t EGO_MAX_FLATBUFFER_SIZE = 32 * 1024;

} // namespace core
