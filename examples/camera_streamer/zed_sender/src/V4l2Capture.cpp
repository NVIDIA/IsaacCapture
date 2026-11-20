// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "V4l2Capture.h"

#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <stdexcept>
#include <system_error>
#include <time.h>
#include <unistd.h>

namespace zed_sender
{
namespace
{
int call(int fd, unsigned long request, void* arg)
{
    int rc;
    do
    {
        rc = ioctl(fd, request, arg);
    } while (rc < 0 && errno == EINTR);
    return rc;
}
void checked(int fd, unsigned long request, void* arg, const char* name)
{
    if (call(fd, request, arg) < 0)
        throw std::system_error(errno, std::generic_category(), name);
}
uint64_t now(clockid_t clock)
{
    timespec t{};
    if (clock_gettime(clock, &t))
        throw std::system_error(errno, std::generic_category(), "clock_gettime");
    return uint64_t(t.tv_sec) * 1000000000ULL + t.tv_nsec;
}
}

struct V4l2Capture::Impl
{
    struct Mapping
    {
        void* address = MAP_FAILED;
        size_t size = 0;
    };
    int fd = -1;
    bool streaming = false, saved = false, touched = false, allocated = false;
    v4l2_format originalFormat{};
    v4l2_streamparm originalParams{};
    std::vector<Mapping> mappings;
    uint32_t width = 0, height = 0, stride = 0;
};

V4l2Capture::V4l2Capture() : impl(new Impl)
{
}
V4l2Capture::~V4l2Capture()
{
    close();
}

void V4l2Capture::open(const std::string& device, uint32_t width, uint32_t height, uint32_t fps)
{
    if (device.empty() || !width || width % 4 || !height || !fps)
        throw std::invalid_argument("invalid side-by-side capture dimensions/FPS/device");
    close();
    try
    {
        impl->fd = ::open(device.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
        if (impl->fd < 0)
            throw std::system_error(errno, std::generic_category(), "open " + device);
        v4l2_capability caps{};
        checked(impl->fd, VIDIOC_QUERYCAP, &caps, "VIDIOC_QUERYCAP");
        const uint32_t flags = caps.capabilities & V4L2_CAP_DEVICE_CAPS ? caps.device_caps : caps.capabilities;
        if (!(flags & V4L2_CAP_VIDEO_CAPTURE) || !(flags & V4L2_CAP_STREAMING))
            throw std::runtime_error("select a streaming video capture node, not the metadata node");
        impl->originalFormat.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        impl->originalParams.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        checked(impl->fd, VIDIOC_G_FMT, &impl->originalFormat, "VIDIOC_G_FMT");
        checked(impl->fd, VIDIOC_G_PARM, &impl->originalParams, "VIDIOC_G_PARM");
        impl->saved = true;
        v4l2_format format{};
        format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        format.fmt.pix.width = width;
        format.fmt.pix.height = height;
        format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
        format.fmt.pix.field = V4L2_FIELD_NONE;
        impl->touched = true;
        checked(impl->fd, VIDIOC_S_FMT, &format, "VIDIOC_S_FMT");
        checked(impl->fd, VIDIOC_G_FMT, &format, "VIDIOC_G_FMT readback");
        const auto& pix = format.fmt.pix;
        if (pix.width != width || pix.height != height || pix.pixelformat != V4L2_PIX_FMT_YUYV ||
            pix.field != V4L2_FIELD_NONE || uint64_t(pix.bytesperline) < uint64_t(width) * 2 ||
            uint64_t(pix.bytesperline) * height > pix.sizeimage)
            throw std::runtime_error("camera did not accept the requested progressive YUYV mode/stride");
        const uint32_t encoding = pix.ycbcr_enc == V4L2_YCBCR_ENC_DEFAULT ?
                                      uint32_t(V4L2_MAP_YCBCR_ENC_DEFAULT(pix.colorspace)) :
                                      pix.ycbcr_enc;
        const uint32_t quantization = pix.quantization == V4L2_QUANTIZATION_DEFAULT ?
                                          uint32_t(V4L2_MAP_QUANTIZATION_DEFAULT(false, pix.colorspace, encoding)) :
                                          pix.quantization;
        if (encoding != V4L2_YCBCR_ENC_601 || quantization != V4L2_QUANTIZATION_LIM_RANGE)
            throw std::runtime_error("this ZED converter requires BT.601 limited-range YUYV");
        v4l2_streamparm params{};
        params.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        params.parm.capture.timeperframe.numerator = 1;
        params.parm.capture.timeperframe.denominator = fps;
        checked(impl->fd, VIDIOC_S_PARM, &params, "VIDIOC_S_PARM");
        checked(impl->fd, VIDIOC_G_PARM, &params, "VIDIOC_G_PARM readback");
        const auto interval = params.parm.capture.timeperframe;
        if (!interval.numerator || uint64_t(interval.numerator) * fps != interval.denominator)
            throw std::runtime_error("camera did not accept the requested FPS");
        impl->width = width;
        impl->height = height;
        impl->stride = pix.bytesperline;
        v4l2_requestbuffers request{};
        request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        request.memory = V4L2_MEMORY_MMAP;
        request.count = 4;
        checked(impl->fd, VIDIOC_REQBUFS, &request, "VIDIOC_REQBUFS");
        impl->allocated = true;
        if (!request.count)
            throw std::runtime_error("camera allocated no buffers");
        impl->mappings.resize(std::min(4U, request.count));
        for (uint32_t i = 0; i < impl->mappings.size(); ++i)
        {
            v4l2_buffer buffer{};
            buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buffer.memory = V4L2_MEMORY_MMAP;
            buffer.index = i;
            checked(impl->fd, VIDIOC_QUERYBUF, &buffer, "VIDIOC_QUERYBUF");
            if (buffer.length < uint64_t(impl->stride) * height)
                throw std::runtime_error("camera buffer is shorter than a frame");
            auto& mapping = impl->mappings[i];
            mapping.size = buffer.length;
            mapping.address = mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, impl->fd, buffer.m.offset);
            if (mapping.address == MAP_FAILED)
                throw std::system_error(errno, std::generic_category(), "mmap camera buffer");
            checked(impl->fd, VIDIOC_QBUF, &buffer, "VIDIOC_QBUF startup");
        }
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        impl->streaming = true;
        checked(impl->fd, VIDIOC_STREAMON, &type, "VIDIOC_STREAMON");
    }
    catch (...)
    {
        close();
        throw;
    }
}

bool V4l2Capture::readLatest(Frame& frame, int timeoutMs)
{
    if (timeoutMs < 0 || !impl->streaming)
        throw std::logic_error("invalid capture read");
    pollfd p{ impl->fd, POLLIN, 0 };
    const int status = poll(&p, 1, timeoutMs);
    if (status < 0 && errno == EINTR)
        return false;
    if (status < 0)
        throw std::system_error(errno, std::generic_category(), "poll camera");
    if (!status)
        return false;
    if (p.revents & (POLLERR | POLLHUP | POLLNVAL))
        throw std::runtime_error("camera disconnected or poll error");
    std::array<v4l2_buffer, 4> dequeued{};
    size_t count = 0;
    int selected = -1;
    uint64_t readyWall = 0, readyMono = 0;
    const size_t bytes = size_t(impl->stride) * impl->height;
    try
    {
        // Holding dequeued buffers makes the drain finite even while the sensor runs.
        while (count < impl->mappings.size())
        {
            auto& b = dequeued[count];
            b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            b.memory = V4L2_MEMORY_MMAP;
            if (call(impl->fd, VIDIOC_DQBUF, &b) < 0)
            {
                if (errno == EAGAIN)
                    break;
                throw std::system_error(errno, std::generic_category(), "VIDIOC_DQBUF");
            }
            ++count;
            if (b.index >= impl->mappings.size())
                throw std::runtime_error("invalid camera buffer index");
            if (b.bytesused > impl->mappings[b.index].size)
                throw std::runtime_error("oversized camera frame");
            if (!(b.flags & V4L2_BUF_FLAG_ERROR) && b.bytesused >= bytes)
            {
                selected = int(count - 1);
                readyMono = now(CLOCK_MONOTONIC);
                readyWall = now(CLOCK_REALTIME);
            }
        }
        if (selected >= 0)
        {
            const auto& b = dequeued[size_t(selected)];
            frame.bytes.resize(bytes);
            memcpy(frame.bytes.data(), impl->mappings[b.index].address, bytes);
            frame.width = impl->width;
            frame.height = impl->height;
            frame.stride = impl->stride;
            frame.sequence = b.sequence;
            frame.captureReadyRealtimeNs = readyWall;
            frame.captureReadyMonotonicNs = readyMono;
        }
    }
    catch (...)
    {
        for (size_t i = 0; i < count; ++i)
            call(impl->fd, VIDIOC_QBUF, &dequeued[i]);
        throw;
    }
    int error = 0;
    for (size_t i = 0; i < count; ++i)
        if (call(impl->fd, VIDIOC_QBUF, &dequeued[i]) < 0 && !error)
            error = errno;
    if (error)
        throw std::system_error(error, std::generic_category(), "VIDIOC_QBUF recycle");
    return selected >= 0;
}

void V4l2Capture::close() noexcept
{
    if (impl->fd >= 0 && impl->streaming)
    {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        call(impl->fd, VIDIOC_STREAMOFF, &type);
    }
    for (auto& mapping : impl->mappings)
        if (mapping.address != MAP_FAILED)
            munmap(mapping.address, mapping.size);
    impl->mappings.clear();
    if (impl->fd >= 0)
    {
        if (impl->allocated)
        {
            v4l2_requestbuffers request{};
            request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            request.memory = V4L2_MEMORY_MMAP;
            call(impl->fd, VIDIOC_REQBUFS, &request);
        }
        if (impl->saved && impl->touched)
        {
            call(impl->fd, VIDIOC_S_FMT, &impl->originalFormat);
            call(impl->fd, VIDIOC_S_PARM, &impl->originalParams);
        }
        ::close(impl->fd);
    }
    impl->fd = -1;
    impl->streaming = impl->saved = impl->touched = impl->allocated = false;
}
}
