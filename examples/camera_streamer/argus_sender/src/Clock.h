// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace argus_sender
{
enum SenderClockDomain
{
    SENDER_CLOCK_UNKNOWN = 0,
    SENDER_CLOCK_TAI = 1,
    SENDER_CLOCK_REALTIME = 2
};

uint64_t readSenderClockNs(uint32_t* clockDomain = nullptr);
uint64_t readTscNowNs();
}
