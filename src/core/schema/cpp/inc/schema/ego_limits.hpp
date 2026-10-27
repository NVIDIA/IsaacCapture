// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace core
{

// Size of every EGO SchemaPusher tensor; the orbbec_ego_* entries in
// deviceio_trackers/trackers.toml must set the same max_flatbuffer_size. 32 KiB
// accommodates the 32-sample IMU batches and EGO calibration YAML while remaining
// below the CloudXR runtime's IPC tensor-payload limit.
inline constexpr size_t EGO_MAX_FLATBUFFER_SIZE = 32 * 1024;

} // namespace core
