/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_CLOCK_EXCHANGE_H
#define ARGUS_SENDER_CLOCK_EXCHANGE_H

// The four-timestamp clock exchange payloads, little-endian: a request is 'Q', u32 seq, i64 t1 (the requester's send
// time) and the reply is 'R', u32 seq, i64 t1 echoed, i64 t2 (the responder's receive time) and i64 t3 (the responder's
// send time). The responder's times are on its CLOCK_MONOTONIC.

#include <cstddef>
#include <cstdint>
#include <string>

namespace argus_sender
{

const size_t kClockRequestBytes = 13;
const size_t kClockReplyBytes = 29;

// Parses a request; false unless `data` is exactly one well-formed request.
bool parseClockRequest(const uint8_t* data, size_t size, uint32_t* seq, int64_t* t1);

// Builds the reply to a request received at t2 and answered at t3.
std::string buildClockReply(uint32_t seq, int64_t t1, int64_t t2, int64_t t3);

// CLOCK_MONOTONIC in nanoseconds.
int64_t monotonicClockNs();

} // namespace argus_sender

#endif // ARGUS_SENDER_CLOCK_EXCHANGE_H
