// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "HardwareEncoder.h"

#include "CudaConversion.h"

#include <NvVideoEncoder.h>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <v4l2_nv_extensions.h>

namespace zed_sender
{
namespace
{
void check(int rc, const char* operation)
{
    if (rc < 0)
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(rc));
}
NvBufSurface* surface(uint32_t width, uint32_t height, NvBufSurfaceColorFormat format, NvBufSurfaceLayout layout)
{
    NvBufSurfaceCreateParams p{};
    p.width = width;
    p.height = height;
    p.colorFormat = format;
    p.layout = layout;
    p.memType = NVBUF_MEM_SURFACE_ARRAY;
    NvBufSurface* result = nullptr;
    check(NvBufSurfaceCreate(&result, 1, &p), "NvBufSurfaceCreate");
    if (!result)
        throw std::runtime_error("NvBufSurfaceCreate returned no surface");
    result->numFilled = 1;
    return result;
}
void configureVic()
{
    NvBufSurfTransformConfigParams p{};
    p.compute_mode = NvBufSurfTransformCompute_VIC;
    check(NvBufSurfTransformSetSessionParams(&p), "NvBufSurfTransformSetSessionParams");
}
void dequeue(NvV4l2ElementPlane& plane, v4l2_buffer& buffer, NvBuffer*& output, const char* operation)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (plane.dqBuffer(buffer, &output, nullptr, 0) < 0)
    {
        if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
            throw std::runtime_error(std::string(operation) + ": " + std::strerror(errno));
        if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error(std::string(operation) + " timed out");
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
}
}

struct HardwareEncoder::Impl
{
    struct Eye
    {
        NvVideoEncoder* encoder = nullptr;
        NvBufSurface* output = nullptr;
    };
    std::array<Eye, 2> eyes{};
    NvBufSurface* input = nullptr;
    bool mapped = false;
    std::unique_ptr<CudaConversion> conversion;
    uint8_t* rgbaDevice = nullptr;
    uint32_t inputWidth = 0, inputHeight = 0, width = 0, height = 0;

    ~Impl()
    {
        // Synchronize even a failed CUDA submission before unmapping its destination.
        conversion.reset();
        // Release encoder imports before destroying their DMA-BUF surfaces.
        for (auto& eye : eyes)
        {
            if (eye.encoder)
            {
                eye.encoder->output_plane.setStreamStatus(false);
                eye.encoder->capture_plane.setStreamStatus(false);
                delete eye.encoder;
            }
            if (eye.output)
                NvBufSurfaceDestroy(eye.output);
        }
        if (mapped)
            NvBufSurfaceUnMapCudaBuffer(input, 0);
        if (input)
            NvBufSurfaceDestroy(input);
    }

    void initialize(uint32_t iw, uint32_t ih, uint32_t ow, uint32_t oh, uint32_t fps, uint32_t mbps, uint32_t gop)
    {
        inputWidth = iw;
        inputHeight = ih;
        width = ow;
        height = oh;
        configureVic();
        // CUDA performs the explicit BT.601 conversion; direct YUYV surface
        // transforms on this BSP produced incorrect colors in hardware tests.
        conversion.reset(new CudaConversion(iw, ih));
        input = surface(iw, ih, NVBUF_COLOR_FORMAT_RGBA, NVBUF_LAYOUT_PITCH);
        check(NvBufSurfaceMapCudaBuffer(input, 0), "NvBufSurfaceMapCudaBuffer (Thor/OpenRM required)");
        mapped = true;
        auto* cudaBuffer = static_cast<NvBufSurfaceCudaBuffer*>(input->surfaceList[0].mappedAddr.cudaPtr);
        if (!cudaBuffer || !cudaBuffer->dataPtr || input->surfaceList[0].pitch < iw * 4)
            throw std::runtime_error("invalid mapped CUDA RGBA surface");
        rgbaDevice = static_cast<uint8_t*>(cudaBuffer->dataPtr);
        for (size_t index = 0; index < eyes.size(); ++index)
        {
            auto& eye = eyes[index];
            // The UVC camera reports BT.601 limited range; VIC converts to the receiver's BT.709 convention.
            eye.output = surface(ow, oh, NVBUF_COLOR_FORMAT_NV12_709, NVBUF_LAYOUT_BLOCK_LINEAR);
            eye.encoder = NvVideoEncoder::createVideoEncoder(index ? "zed_right" : "zed_left", O_NONBLOCK);
            if (!eye.encoder)
                throw std::runtime_error("could not create Jetson hardware HEVC encoder");
            auto& e = *eye.encoder;
            check(e.setCapturePlaneFormat(V4L2_PIX_FMT_H265, ow, oh, 16 * 1024 * 1024), "capture format");
            check(e.setOutputPlaneFormat(V4L2_PIX_FMT_NV12M, ow, oh, V4L2_COLORSPACE_REC709, false), "output format");
            check(e.setBitrate(mbps * 1000000), "bitrate");
            check(e.setProfile(V4L2_MPEG_VIDEO_H265_PROFILE_MAIN), "profile");
            check(e.setRateControlMode(V4L2_MPEG_VIDEO_BITRATE_MODE_CBR), "CBR");
            check(e.setFrameRate(fps, 1), "FPS");
            check(e.setIFrameInterval(gop), "GOP");
            check(e.setIDRInterval(gop), "IDR interval");
            check(e.setNumBFrames(0), "disable B frames");
            check(e.setNumReferenceFrames(1), "reference frames");
            check(e.setHWPresetType(V4L2_ENC_HW_PRESET_ULTRAFAST), "hardware preset");
            check(e.setInsertSpsPpsAtIdrEnabled(true), "SPS/PPS");
            check(e.setInsertAudEnabled(true), "AUD");
            check(e.setInsertVuiEnabled(true), "VUI");
            check(e.output_plane.setupPlane(V4L2_MEMORY_DMABUF, 2, false, false), "output buffers");
            check(e.capture_plane.setupPlane(V4L2_MEMORY_MMAP, 4, true, false), "capture buffers");
            check(e.output_plane.setStreamStatus(true), "output stream on");
            check(e.capture_plane.setStreamStatus(true), "capture stream on");
            for (uint32_t i = 0; i < e.capture_plane.getNumBuffers(); ++i)
            {
                v4l2_buffer b{};
                v4l2_plane planes[VIDEO_MAX_PLANES]{};
                b.index = i;
                b.m.planes = planes;
                check(e.capture_plane.qBuffer(b, nullptr), "queue capture buffer");
            }
        }
    }

    void convert(const uint8_t* yuyv, size_t bytes, size_t stride)
    {
        if (!yuyv || stride < size_t(inputWidth) * 2 || bytes / inputHeight < stride)
            throw std::runtime_error("incomplete YUYV input");
        conversion->convert(yuyv, stride, rgbaDevice, input->surfaceList[0].pitch);
        configureVic();
        for (size_t i = 0; i < eyes.size(); ++i)
        {
            NvBufSurfTransformRect src{}, dst{};
            src.left = uint32_t(i) * inputWidth / 2;
            src.width = inputWidth / 2;
            src.height = inputHeight;
            dst.width = width;
            dst.height = height;
            NvBufSurfTransformParams p{};
            p.src_rect = &src;
            p.dst_rect = &dst;
            p.transform_flag = NVBUFSURF_TRANSFORM_CROP_SRC | NVBUFSURF_TRANSFORM_CROP_DST | NVBUFSURF_TRANSFORM_FILTER;
            p.transform_filter = NvBufSurfTransformInter_Bilinear;
            check(NvBufSurfTransform(input, eyes[i].output, &p), "VIC RGB to BT.709 NV12");
        }
    }

    void submit(Eye& eye)
    {
        auto* slot = eye.encoder->output_plane.getNthBuffer(0);
        v4l2_buffer b{};
        v4l2_plane planes[VIDEO_MAX_PLANES]{};
        b.m.planes = planes;
        for (uint32_t i = 0; i < slot->n_planes; ++i)
        {
            planes[i].m.fd = int(eye.output->surfaceList[0].bufferDesc);
            planes[i].bytesused = slot->planes[i].fmt.sizeimage;
            planes[i].length = slot->planes[i].fmt.sizeimage;
        }
        check(eye.encoder->output_plane.qBuffer(b, nullptr), "submit HEVC input");
    }

    std::vector<uint8_t> collect(Eye& eye)
    {
        v4l2_buffer b{};
        v4l2_plane planes[VIDEO_MAX_PLANES]{};
        b.m.planes = planes;
        NvBuffer* captured = nullptr;
        dequeue(eye.encoder->capture_plane, b, captured, "HEVC capture");
        if (!captured || !captured->planes[0].data || (b.flags & V4L2_BUF_FLAG_ERROR) ||
            planes[0].bytesused <= planes[0].data_offset || planes[0].bytesused > captured->planes[0].fmt.sizeimage)
            throw std::runtime_error("invalid HEVC access unit");
        const auto* data = captured->planes[0].data;
        std::vector<uint8_t> result(data + planes[0].data_offset, data + planes[0].bytesused);
        captured->planes[0].bytesused = 0;
        planes[0].bytesused = planes[0].data_offset = 0;
        b.flags = 0;
        check(eye.encoder->capture_plane.qBuffer(b, nullptr), "recycle HEVC output");
        return result;
    }

    void release(Eye& eye)
    {
        v4l2_buffer b{};
        v4l2_plane planes[VIDEO_MAX_PLANES]{};
        b.m.planes = planes;
        NvBuffer* released = nullptr;
        dequeue(eye.encoder->output_plane, b, released, "release HEVC input");
        if (b.index || (b.flags & V4L2_BUF_FLAG_ERROR))
            throw std::runtime_error("invalid HEVC input completion");
    }
};

HardwareEncoder::HardwareEncoder() = default;
HardwareEncoder::~HardwareEncoder() = default;
void HardwareEncoder::close() noexcept
{
    impl.reset();
}
void HardwareEncoder::open(uint32_t iw, uint32_t ih, uint32_t ow, uint32_t oh, uint32_t fps, uint32_t mbps, uint32_t gop)
{
    close();
    std::unique_ptr<Impl> candidate(new Impl);
    candidate->initialize(iw, ih, ow, oh, fps, mbps, gop);
    impl = std::move(candidate);
}
EncodedStereo HardwareEncoder::encode(const uint8_t* data, size_t bytes, size_t stride)
{
    if (!impl)
        throw std::logic_error("encoder is not open");
    try
    {
        impl->convert(data, bytes, stride);
        impl->submit(impl->eyes[0]);
        impl->submit(impl->eyes[1]);
        EncodedStereo pair{ impl->collect(impl->eyes[0]), impl->collect(impl->eyes[1]) };
        impl->release(impl->eyes[0]);
        impl->release(impl->eyes[1]);
        return pair;
    }
    catch (...)
    {
        close();
        throw;
    }
}
}
