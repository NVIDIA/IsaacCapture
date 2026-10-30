/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_CAPTURE_MANAGER_H
#define ARGUS_SENDER_CAPTURE_MANAGER_H

#include "Config.h"
#include "OutputGeometry.h"
#include "StereoTiming.h"

#include <Argus/Argus.h>
#include <EGL/egl.h>
#include <EGLStream/EGLStream.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace argus_sender
{

class DmaBuffer;

class CapturedDmaBuffer
{
public:
    CapturedDmaBuffer();
    ~CapturedDmaBuffer();
    CapturedDmaBuffer(CapturedDmaBuffer&& other) noexcept;
    CapturedDmaBuffer& operator=(CapturedDmaBuffer&& other) noexcept;

    DmaBuffer* get() const;
    DmaBuffer* release();
    void reset(DmaBuffer* buffer = NULL);

private:
    CapturedDmaBuffer(const CapturedDmaBuffer&);
    CapturedDmaBuffer& operator=(const CapturedDmaBuffer&);

    DmaBuffer* m_buffer;
};

struct CapturedFrameSet
{
    std::vector<Argus::UniqueObj<EGLStream::Frame>> frames;
    // HEVC streams capture directly into client-owned encoder DMABUFs. Each
    // lease returns its buffer to Argus unless ownership is transferred to
    // NVENC by StreamProcessor.
    std::vector<CapturedDmaBuffer> dmaBuffers;
    std::vector<FrameTiming> timings;
};

class CaptureManager
{
public:
    explicit CaptureManager(const AppConfig& config);
    ~CaptureManager();

    bool initialize(Argus::CameraProvider* provider, EGLDisplay display);
    bool startPriorityDomain(size_t priorityDomainIndex);
    bool startSecondaryDomains(size_t priorityDomainIndex);
    bool discardDomainWarmup(size_t domainIndex, uint32_t frameCount);
    bool discardWarmup(uint32_t frameCount);
    bool waitForDomainSynchronization(size_t domainIndex,
                                      uint32_t maxFrameSets,
                                      uint32_t requiredConsecutive,
                                      uint64_t toleranceNs);
    bool requestDomainCapture(size_t domainIndex);
    void prepareCaptured(std::vector<CapturedFrameSet>* captured) const;
    bool acquireDomain(size_t domainIndex, std::vector<CapturedFrameSet>* captured);
    bool acquireDomainClosestTo(size_t domainIndex, uint64_t referenceSofTscNs, std::vector<CapturedFrameSet>* captured);
    bool reacquireDomain(size_t domainIndex, std::vector<CapturedFrameSet>* captured);
    void recordCaptured(size_t streamIndex, const CapturedFrameSet& frames);
    bool stop();
    void releaseResources();
    bool hasError() const;

    size_t streamCount() const;
    size_t domainCount() const;
    size_t streamDomain(size_t streamIndex) const;
    size_t domainFor(size_t streamIndex, size_t cameraIndex) const;
    bool domainUsesContinuousFifo(size_t domainIndex) const;
    bool hasContinuousFifoDomains() const;
    void logFifoCandidateStats();
    const ResolvedOutputGeometry& geometry(size_t streamIndex) const;
    uint32_t nominalFps(size_t streamIndex) const;
    uint64_t framePeriodNs(size_t streamIndex) const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace argus_sender

#endif // ARGUS_SENDER_CAPTURE_MANAGER_H
