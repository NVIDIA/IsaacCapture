/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_ENCODED_FRAME_SINK_H
#define ARGUS_SENDER_ENCODED_FRAME_SINK_H

#include <cstddef>
#include <cstdint>

namespace argus_sender
{

class EncodedFrameSink
{
public:
    virtual ~EncodedFrameSink()
    {
    }
    virtual bool enqueueEncoded(
        const uint8_t* data, size_t size, uint64_t frameSeq, uint64_t encodeDoneNs, uint32_t senderClockDomain) = 0;
    virtual bool hasError() const = 0;
};

} // namespace argus_sender

#endif // ARGUS_SENDER_ENCODED_FRAME_SINK_H
