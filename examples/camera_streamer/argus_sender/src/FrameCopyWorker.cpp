/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "FrameCopyWorker.h"

#include "HevcEncoder.h"
#include "NvRemap.h"

#include <Argus/Argus.h>
#include <EGLStream/EGLStream.h>
#include <EGLStream/NV/ImageNativeBuffer.h>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <nvbufsurface.h>
#include <string>
#include <thread>

namespace argus_sender
{
namespace
{

bool copyFrameToBuffer(EGLStream::Frame* frame, DmaBuffer* buffer, const char* label)
{
    EGLStream::IFrame* iFrame = Argus::interface_cast<EGLStream::IFrame>(frame);
    if (!iFrame)
    {
        std::fprintf(stderr, "Failed to get %s EGLStream IFrame interface\n", label);
        return false;
    }

    EGLStream::NV::IImageNativeBuffer* iNativeBuffer =
        Argus::interface_cast<EGLStream::NV::IImageNativeBuffer>(iFrame->getImage());
    if (!iNativeBuffer)
    {
        std::fprintf(stderr, "%s IImageNativeBuffer not supported by Argus EGL image\n", label);
        return false;
    }

    if (iNativeBuffer->copyToNvBuffer(buffer->getFd(), EGLStream::NV::ROTATION_0) != Argus::STATUS_OK)
    {
        std::fprintf(stderr, "Failed to copy %s Argus EGL frame to raw processing DMABUF\n", label);
        return false;
    }
    return true;
}

// Produces the tight raw NV12 wire payload from the pitch-linear DMABUF:
// either a plain row pack of the whole buffer, or the LUT remap when the
// job carries a table. Runs on the worker thread.
bool packRawNv12(DmaBuffer* buffer,
                 const RawPackJob& pack,
                 std::vector<uint8_t>* scratchY,
                 std::vector<uint8_t>* scratchUV,
                 const char* label)
{
    if (!buffer || !pack.payload)
        return false;
    NvBufSurface* surface = NULL;
    if (NvBufSurfaceFromFd(buffer->getFd(), reinterpret_cast<void**>(&surface)) != 0 || !surface)
    {
        std::fprintf(stderr, "Failed to resolve %s raw NV12 NvBufSurface\n", label);
        return false;
    }
    if (NvBufSurfaceMap(surface, 0, -1, NVBUF_MAP_READ) != 0)
    {
        std::fprintf(stderr, "Failed to map %s raw NV12 NvBufSurface\n", label);
        return false;
    }
    bool ok = true;
    if (NvBufSurfaceSyncForCpu(surface, 0, -1) != 0)
    {
        std::fprintf(stderr, "Failed to synchronize %s raw NV12 for CPU\n", label);
        ok = false;
    }
    const NvBufSurfacePlaneParams& planes = surface->surfaceList[0].planeParams;
    if (ok && planes.num_planes < 2)
    {
        std::fprintf(stderr, "%s raw NV12 surface has %u plane(s), expected at least 2\n", label, planes.num_planes);
        ok = false;
    }
    const uint8_t* sourceY = NULL;
    const uint8_t* sourceUV = NULL;
    if (ok)
    {
        sourceY = static_cast<const uint8_t*>(surface->surfaceList[0].mappedAddr.addr[0]);
        sourceUV = static_cast<const uint8_t*>(surface->surfaceList[0].mappedAddr.addr[1]);
        if (!sourceY || !sourceUV)
        {
            std::fprintf(stderr, "%s raw NV12 planes are not CPU-mapped\n", label);
            ok = false;
        }
    }
    if (ok && pack.remap)
    {
        const NvRemapTable& table = *pack.remap;
        if (planes.width[0] < table.srcWidth || planes.height[0] < table.srcHeight)
        {
            std::fprintf(stderr, "%s remap expects a %ux%u source but the buffer is %ux%u\n", label, table.srcWidth,
                         table.srcHeight, planes.width[0], planes.height[0]);
            ok = false;
        }
        else
        {
            // NvBufSurface CPU mappings are uncached/write-combined: random
            // gathers there are an order of magnitude slower than cached
            // reads. Stream-copy just the source rows the taps use into
            // cached scratch memory first (sequential reads are fast), then
            // gather from the scratch.
            const uint32_t lumaRows = table.lumaRowLast - table.lumaRowFirst + 1;
            const uint32_t chromaRows = table.chromaRowLast - table.chromaRowFirst + 1;
            scratchY->resize(static_cast<size_t>(table.srcWidth) * lumaRows);
            scratchUV->resize(static_cast<size_t>(table.srcWidth) * chromaRows);
            for (uint32_t row = 0; row < lumaRows; ++row)
            {
                std::memcpy(scratchY->data() + static_cast<size_t>(row) * table.srcWidth,
                            sourceY + static_cast<size_t>(table.lumaRowFirst + row) * planes.pitch[0], table.srcWidth);
            }
            for (uint32_t row = 0; row < chromaRows; ++row)
            {
                std::memcpy(scratchUV->data() + static_cast<size_t>(row) * table.srcWidth,
                            sourceUV + static_cast<size_t>(table.chromaRowFirst + row) * planes.pitch[1], table.srcWidth);
            }
            pack.payload->resize(static_cast<size_t>(table.outWidth) * table.outHeight * 3U / 2U);
            nvRemapNv12(table, scratchY->data(), table.srcWidth, scratchUV->data(), table.srcWidth,
                        pack.payload->data(), table.lumaRowFirst, table.chromaRowFirst);
        }
    }
    else if (ok)
    {
        pack.payload->resize(static_cast<size_t>(pack.width) * pack.height * 3U / 2U);
        size_t destinationOffset = 0;
        for (uint32_t plane = 0; plane < 2 && ok; ++plane)
        {
            const uint32_t rows = plane == 0 ? pack.height : pack.height / 2;
            const uint32_t rowBytes = pack.width;
            const uint32_t availableRowBytes = planes.width[plane] * planes.bytesPerPix[plane];
            const uint8_t* source = plane == 0 ? sourceY : sourceUV;
            if (planes.height[plane] < rows || availableRowBytes < rowBytes || planes.pitch[plane] < rowBytes)
            {
                std::fprintf(stderr, "%s raw NV12 plane %u layout cannot provide tight %ux%u data\n", label, plane,
                             rowBytes, rows);
                ok = false;
                break;
            }
            for (uint32_t row = 0; row < rows; ++row)
            {
                std::memcpy(pack.payload->data() + destinationOffset,
                            source + static_cast<size_t>(row) * planes.pitch[plane], rowBytes);
                destinationOffset += rowBytes;
            }
        }
    }
    NvBufSurfaceUnMap(surface, 0, -1);
    if (!ok)
        pack.payload->clear();
    return ok;
}

} // namespace

class FrameCopyWorker::Impl
{
public:
    explicit Impl(const char* label)
        : m_label(label ? label : ""),
          m_stop(false),
          m_hasJob(false),
          m_completed(false),
          m_result(false),
          m_completionSteadyNs(0),
          m_frame(NULL),
          m_buffer(NULL),
          m_hasPack(false)
    {
    }

    ~Impl()
    {
        shutdown();
    }

    bool start()
    {
        if (m_thread.joinable())
            return true;

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = false;
            m_hasJob = false;
            m_completed = false;
            m_result = false;
            m_completionSteadyNs = 0;
            m_frame = NULL;
            m_buffer = NULL;
            m_hasPack = false;
        }
        try
        {
            m_thread = std::thread(&Impl::threadMain, this);
        }
        catch (...)
        {
            std::fprintf(stderr, "Failed to start %s frame-copy worker\n", m_label.c_str());
            return false;
        }
        return true;
    }

    void shutdown()
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_jobAvailable.notify_all();
        m_jobCompleted.notify_all();
        if (m_thread.joinable())
            m_thread.join();
    }

    bool submit(EGLStream::Frame* frame, DmaBuffer* buffer, const RawPackJob* pack)
    {
        if (!frame || !buffer || (pack && !pack->payload))
            return false;

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_stop || !m_thread.joinable() || m_hasJob || m_completed)
            {
                std::fprintf(stderr, "%s frame-copy worker is not ready for a new job\n", m_label.c_str());
                return false;
            }
            m_frame = frame;
            m_buffer = buffer;
            m_hasPack = pack != nullptr;
            if (pack)
                m_pack = *pack;
            m_hasJob = true;
        }
        m_jobAvailable.notify_one();
        return true;
    }

    bool wait(uint64_t* completionSteadyNs)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_jobCompleted.wait(lock, [this]() { return m_completed || m_stop; });
        if (!m_completed)
            return false;

        const bool result = m_result;
        if (completionSteadyNs)
            *completionSteadyNs = m_completionSteadyNs;
        m_completed = false;
        m_result = false;
        m_completionSteadyNs = 0;
        m_frame = NULL;
        m_buffer = NULL;
        m_hasPack = false;
        return result;
    }

private:
    void threadMain()
    {
        while (true)
        {
            EGLStream::Frame* frame = NULL;
            DmaBuffer* buffer = NULL;
            bool hasPack = false;
            RawPackJob pack;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_jobAvailable.wait(lock, [this]() { return m_hasJob || m_stop; });
                if (m_stop)
                    break;
                frame = m_frame;
                buffer = m_buffer;
                hasPack = m_hasPack;
                pack = m_pack;
            }

            bool result = copyFrameToBuffer(frame, buffer, m_label.c_str());
            if (result && hasPack)
                result = packRawNv12(buffer, pack, &m_scratchY, &m_scratchUV, m_label.c_str());
            const uint64_t completionSteadyNs = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                    .count());
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                m_result = result;
                m_completionSteadyNs = completionSteadyNs;
                m_hasJob = false;
                m_completed = true;
            }
            m_jobCompleted.notify_one();
        }
    }

    const std::string m_label;
    std::thread m_thread;
    std::mutex m_mutex;
    std::condition_variable m_jobAvailable;
    std::condition_variable m_jobCompleted;
    bool m_stop;
    bool m_hasJob;
    bool m_completed;
    bool m_result;
    uint64_t m_completionSteadyNs;
    EGLStream::Frame* m_frame;
    DmaBuffer* m_buffer;
    bool m_hasPack;
    RawPackJob m_pack;
    // Persistent cached staging for the remap source rows (worker thread only).
    std::vector<uint8_t> m_scratchY;
    std::vector<uint8_t> m_scratchUV;
};

FrameCopyWorker::FrameCopyWorker(const char* label) : m_impl(new Impl(label))
{
}

FrameCopyWorker::~FrameCopyWorker()
{
}

bool FrameCopyWorker::start()
{
    return m_impl->start();
}

void FrameCopyWorker::shutdown()
{
    m_impl->shutdown();
}

bool FrameCopyWorker::submit(EGLStream::Frame* frame, DmaBuffer* buffer, const RawPackJob* pack)
{
    return m_impl->submit(frame, buffer, pack);
}

bool FrameCopyWorker::wait(uint64_t* completionSteadyNs)
{
    return m_impl->wait(completionSteadyNs);
}

} // namespace argus_sender
