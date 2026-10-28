/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_FRAME_COPY_WORKER_H
#define ARGUS_SENDER_FRAME_COPY_WORKER_H

#include <cstdint>
#include <memory>
#include <vector>

namespace EGLStream
{
class Frame;
}

namespace argus_sender
{

class DmaBuffer;
struct NvRemapTable;

// Optional raw-payload production performed on the worker thread right after
// the VIC copy: either a tight NV12 pack of the whole buffer, or — when
// `remap` is set — a lookup-table resample into outWidth x outHeight (see
// NvRemap.h). Running this on the per-camera worker keeps the two wrist
// remaps parallel and off the coordinator thread.
struct RawPackJob
{
    uint32_t width = 0;
    uint32_t height = 0;
    const NvRemapTable* remap = nullptr;
    std::vector<uint8_t>* payload = nullptr;
};

class FrameCopyWorker
{
public:
    explicit FrameCopyWorker(const char* label);
    ~FrameCopyWorker();

    bool start();
    void shutdown();
    bool submit(EGLStream::Frame* frame, DmaBuffer* buffer, const RawPackJob* pack = nullptr);
    bool wait(uint64_t* completionSteadyNs = NULL);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace argus_sender

#endif // ARGUS_SENDER_FRAME_COPY_WORKER_H
