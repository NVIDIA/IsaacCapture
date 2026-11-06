/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_PACKET_SINK_H
#define ARGUS_SENDER_PACKET_SINK_H

#include <cstdint>
#include <utility>
#include <vector>

namespace argus_sender
{

struct PacketStreamTiming
{
    bool present = false;
    bool encoded = false;
    uint64_t captureStartSteadyNs = 0;
    uint64_t acquireSpreadNs = 0;
    uint64_t copyDoneSteadyNs = 0;
    uint64_t processStartSteadyNs = 0;
    uint64_t processDoneSteadyNs = 0;
};

struct PacketSampleTiming
{
    bool present = false;
    uint64_t sensorSofTscNs = 0;
};

struct PacketTiming
{
    uint64_t captureStartSteadyNs = 0;
    uint64_t acquireSpreadNs = 0;
    uint64_t captureToEncodeStartNs = 0;
    uint64_t captureToProcessDoneNs = 0;
    std::vector<PacketStreamTiming> streams;
    // Flattened in stream-major, camera-minor order.
    std::vector<PacketSampleTiming> samples;
};

class PacketSink
{
public:
    virtual ~PacketSink()
    {
    }
    virtual bool enqueueFramed(std::vector<uint8_t>&& packet) = 0;
    virtual bool enqueueFramedWithTiming(std::vector<uint8_t>&& packet, const PacketTiming& timing)
    {
        (void)timing;
        return enqueueFramed(std::move(packet));
    }
    virtual bool hasError() const = 0;
};

} // namespace argus_sender

#endif // ARGUS_SENDER_PACKET_SINK_H
