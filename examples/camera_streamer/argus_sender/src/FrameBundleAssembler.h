/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_FRAME_BUNDLE_ASSEMBLER_H
#define ARGUS_SENDER_FRAME_BUNDLE_ASSEMBLER_H

#include "Avs1Wire.h"
#include "StereoTiming.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace argus_sender
{

class EncodedFrameSink;
class PacketSink;

struct BundleStreamInput
{
    bool present = false;
    uint64_t sequence = 0;
    std::vector<FrameTiming> timings;
};

class FrameBundleAssembler
{
public:
    FrameBundleAssembler(PacketSink* packetSink, uint64_t sessionId, const std::vector<Avs1StreamRecord>& streamTemplates);
    ~FrameBundleAssembler();

    EncodedFrameSink* encodedSink(size_t streamIndex, size_t cameraIndex);
    bool beginBundle(uint64_t bundleSequence,
                     const std::vector<BundleStreamInput>& streams,
                     uint64_t senderCaptureReadyNs,
                     uint64_t senderProcessSubmitNs,
                     uint32_t senderClockDomain,
                     uint64_t senderCaptureReadyTscNs);
    bool recordCopyDone(size_t streamIndex, uint64_t bundleSequence, uint64_t copyDoneSteadyNs);
    bool recordProcessStart(size_t streamIndex, uint64_t bundleSequence, uint64_t processStartSteadyNs);
    bool submitRaw(size_t streamIndex,
                   size_t cameraIndex,
                   uint64_t bundleSequence,
                   std::vector<uint8_t>&& payload,
                   uint64_t processDoneNs,
                   uint32_t senderClockDomain);
    bool hasError() const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace argus_sender

#endif // ARGUS_SENDER_FRAME_BUNDLE_ASSEMBLER_H
