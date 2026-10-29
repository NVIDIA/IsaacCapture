/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_STREAM_PROCESSOR_H
#define ARGUS_SENDER_STREAM_PROCESSOR_H

#include "Config.h"
#include "OutputGeometry.h"

#include <cstdint>
#include <memory>

namespace argus_sender
{

class FrameBundleAssembler;
struct CapturedFrameSet;

class StreamProcessor
{
public:
    StreamProcessor(const StreamConfig& config,
                    const ResolvedOutputGeometry& geometry,
                    size_t streamIndex,
                    bool kpi,
                    uint32_t logIntervalSeconds,
                    bool benchEncode);
    ~StreamProcessor();

    bool start(FrameBundleAssembler* assembler, uint32_t encoderFps);
    bool waitForInputBuffers();
    bool submitCopies(CapturedFrameSet* frames, uint64_t bundleSequence);
    bool finishSubmit();
    bool discardPendingCopies();
    bool drain();
    bool sendEos();
    void shutdown();
    bool hasError() const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace argus_sender

#endif // ARGUS_SENDER_STREAM_PROCESSOR_H
