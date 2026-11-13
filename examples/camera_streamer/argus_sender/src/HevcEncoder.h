/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_HEVC_ENCODER_H
#define ARGUS_SENDER_HEVC_ENCODER_H

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <linux/videodev2.h>

#include <NvBufSurface.h>
#include <NvBuffer.h>
#include <atomic>
#include <cstdint>
#include <memory>

namespace argus_sender
{

class EncodedFrameSink;

struct HevcEncoderSettings
{
    uint32_t bitrateMbps = 50;
    bool kpi = true;
    uint32_t logIntervalSeconds = 1;
    bool benchEncode = false;
};

const uint32_t kEncoderOutputPlaneBuffers = 2;

// A capture backend that lends buffers to the encoder implements this; release() hands a buffer back to it.
class DmaBufferOwner
{
public:
    virtual ~DmaBufferOwner()
    {
    }
    virtual bool release() = 0;
};

class DmaBuffer : public NvBuffer
{
public:
    static DmaBuffer* create(uint32_t width,
                             uint32_t height,
                             NvBufSurfaceColorFormat colorFormat,
                             NvBufSurfaceLayout layout = NVBUF_LAYOUT_BLOCK_LINEAR);
    // Wraps a DMA-BUF this process does not own, for the encoder's input queue. The caller keeps `fd` valid until the
    // encoder has dequeued the buffer; destroying the wrapper closes and frees nothing.
    static DmaBuffer* createExternal(int fd);
    ~DmaBuffer();

    int getFd() const;
    bool mapEglImage();
    EGLImageKHR getEglImage() const;

    // The backend this buffer is on loan from, if any. releaseToOwner() hands it back and returns false when there is
    // no owner or the owner refuses.
    void setOwner(std::unique_ptr<DmaBufferOwner> owner);
    DmaBufferOwner* owner() const;
    bool hasOwner() const;
    bool releaseToOwner();

private:
    DmaBuffer();

    int m_fd;
    bool m_ownsSurface;
    NvBufSurface* m_surface;
    bool m_eglImageMapped;
    std::unique_ptr<DmaBufferOwner> m_owner;
};

class DirectHevcEncoder
{
public:
    DirectHevcEncoder(EncodedFrameSink* sink, const HevcEncoderSettings& settings);
    ~DirectHevcEncoder();

    bool initialize(uint32_t width, uint32_t height, uint32_t fps, uint32_t gop);
    bool queueInput(struct v4l2_buffer& v4l2Buf, DmaBuffer* buffer, uint64_t frameSeq = 0);
    bool dequeueInput(struct v4l2_buffer& v4l2Buf, DmaBuffer** buffer);
    bool sendEos(struct v4l2_buffer& v4l2Buf);
    bool hasError() const;
    void shutdown();

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

bool waitForEncoderInput(DirectHevcEncoder* encoder, struct v4l2_buffer* v4l2Buf, DmaBuffer** buffer);

} // namespace argus_sender

#endif // ARGUS_SENDER_HEVC_ENCODER_H
