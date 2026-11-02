/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_HEVC_ENCODER_H
#define ARGUS_SENDER_HEVC_ENCODER_H

#include <Argus/Argus.h>
#include <Argus/BufferStream.h>
#include <EGL/egl.h>
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

class DmaBuffer : public NvBuffer
{
public:
    static DmaBuffer* create(const Argus::Size2D<uint32_t>& size,
                             NvBufSurfaceColorFormat colorFormat,
                             NvBufSurfaceLayout layout = NVBUF_LAYOUT_BLOCK_LINEAR);
    ~DmaBuffer();

    int getFd() const;
    bool mapEglImage();
    EGLImageKHR getEglImage() const;

    // Bind this client-owned DMABUF to the Argus Buffer that writes it.
    // A bound buffer is acquired by the application until releaseToArgus().
    bool bindArgusBuffer(Argus::IBufferOutputStream* stream, Argus::Buffer* buffer);
    bool markArgusAcquired(Argus::Buffer* buffer);
    bool releaseToArgus();
    bool isArgusBound() const;
    Argus::Buffer* getArgusBuffer() const;

private:
    DmaBuffer();

    int m_fd;
    NvBufSurface* m_surface;
    bool m_eglImageMapped;
    Argus::IBufferOutputStream* m_argusStream;
    Argus::Buffer* m_argusBuffer;
    std::atomic<bool> m_argusAcquired;
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
