// SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace plugins::ego
{

// Converts an SDK microsecond timestamp to nanoseconds; throws std::overflow_error
// when the result does not fit in int64.
int64_t sdk_timestamp_ns(uint64_t timestamp_us);

} // namespace plugins::ego
