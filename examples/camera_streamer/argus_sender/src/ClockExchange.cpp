/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ClockExchange.h"

#include <ctime>

namespace argus_sender
{
namespace
{

uint64_t readLe(const uint8_t* data, int bytes)
{
    uint64_t value = 0;
    for (int i = bytes - 1; i >= 0; --i)
        value = (value << 8) | data[i];
    return value;
}

void appendLe(std::string* out, uint64_t value, int bytes)
{
    for (int i = 0; i < bytes; ++i)
        out->push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
}

} // namespace

bool parseClockRequest(const uint8_t* data, size_t size, uint32_t* seq, int64_t* t1)
{
    if (!data || size != kClockRequestBytes || data[0] != 'Q')
        return false;
    *seq = static_cast<uint32_t>(readLe(data + 1, 4));
    *t1 = static_cast<int64_t>(readLe(data + 5, 8));
    return true;
}

std::string buildClockReply(uint32_t seq, int64_t t1, int64_t t2, int64_t t3)
{
    std::string reply;
    reply.reserve(kClockReplyBytes);
    reply.push_back('R');
    appendLe(&reply, seq, 4);
    appendLe(&reply, static_cast<uint64_t>(t1), 8);
    appendLe(&reply, static_cast<uint64_t>(t2), 8);
    appendLe(&reply, static_cast<uint64_t>(t3), 8);
    return reply;
}

int64_t monotonicClockNs()
{
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

} // namespace argus_sender
