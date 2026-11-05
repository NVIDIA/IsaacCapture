// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Argus buffer-stream capture into NvBufSurface DMA-BUFs. The Argus calls follow the Jetson Multimedia API's
// client-owned-buffer pattern. Not built or run by the author: no Argus SDK or camera was available.

#include "inc/sensing_core/argus_capture.hpp"

#include <Argus/Argus.h>
#include <Argus/BufferStream.h>
#include <Argus/EGLImage.h>
#include <EGL/egl.h>

#include <NvBufSurface.h>
#include <cstring>
#include <ctime>
#include <map>
#include <stdexcept>
#include <string>

namespace plugins
{
namespace sensing
{

namespace
{

void check(Argus::Status status, const std::string& what)
{
    if (status != Argus::STATUS_OK)
        throw std::runtime_error("Argus: " + what + " failed (status " + std::to_string(static_cast<int>(status)) + ")");
}

uint64_t clock_ns(clockid_t clock)
{
    timespec ts{};
    clock_gettime(clock, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ull + static_cast<uint64_t>(ts.tv_nsec);
}

// One NV12 DMA-BUF the camera writes into. Block-linear: only the hardware encoder reads it.
struct CaptureBuffer
{
    int fd = -1;
    NvBufSurface* surface = nullptr;
    bool egl_mapped = false;
    Argus::UniqueObj<Argus::Buffer> argus;

    ~CaptureBuffer()
    {
        argus.reset(); // the Argus buffer refers to the EGL image, so it goes first
        if (egl_mapped)
            NvBufSurfaceUnMapEglImage(surface, 0);
        if (surface)
            NvBufSurfaceDestroy(surface);
    }
};

} // namespace

struct ArgusCapture::Impl
{
    struct Camera
    {
        CameraInfo info;
        Argus::CameraDevice* device = nullptr;
        Argus::SensorMode* mode = nullptr;
        Argus::UniqueObj<Argus::CaptureSession> session;
        Argus::UniqueObj<Argus::OutputStream> stream;
        Argus::IBufferOutputStream* buffers = nullptr;
        Argus::UniqueObj<Argus::Request> request;
        std::vector<std::unique_ptr<CaptureBuffer>> pool;
    };

    Argus::UniqueObj<Argus::CameraProvider> provider;
    EGLDisplay display = EGL_NO_DISPLAY;
    std::vector<std::unique_ptr<Camera>> cameras;
    std::vector<CameraInfo> infos;
    std::map<uint32_t, Camera*> by_id;
    bool running = false;

    Impl(const ArgusConfig& config)
    {
        provider.reset(Argus::CameraProvider::create());
        auto* provider_iface = Argus::interface_cast<Argus::ICameraProvider>(provider);
        if (!provider_iface)
            throw std::runtime_error("Argus: no camera provider (is nvargus-daemon running?)");

        display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (display == EGL_NO_DISPLAY || !eglInitialize(display, nullptr, nullptr))
            throw std::runtime_error("Argus: EGL display initialisation failed");

        std::vector<Argus::CameraDevice*> devices;
        check(provider_iface->getCameraDevices(&devices), "getCameraDevices");
        for (uint32_t index : config.devices)
        {
            if (index >= devices.size())
                throw std::runtime_error("Argus: no camera device " + std::to_string(index) + " (found " +
                                         std::to_string(devices.size()) + ")");
            auto camera = std::make_unique<Camera>();
            camera->device = devices[index];
            camera->mode = sensor_mode(camera->device, config.sensor_mode);
            auto* mode_iface = Argus::interface_cast<Argus::ISensorMode>(camera->mode);
            const auto size = mode_iface->getResolution();
            const uint64_t period_ns = mode_iface->getFrameDurationRange().min();
            if (period_ns == 0)
                throw std::runtime_error("Argus: sensor mode has a zero frame duration");
            camera->info = { index,         "argus" + std::to_string(index),      size.width(),
                             size.height(), 1e9 / static_cast<double>(period_ns), PixelKind::DmaBufNv12 };
            open(*camera, provider_iface, config.buffer_count);
            infos.push_back(camera->info);
            by_id[index] = camera.get();
            cameras.push_back(std::move(camera));
        }
    }

    ~Impl()
    {
        stop();
        for (auto& camera : cameras)
        {
            camera->request.reset();
            camera->pool.clear(); // Argus buffers, then the DMA-BUFs behind them
            camera->stream.reset();
            camera->session.reset();
        }
        cameras.clear();
        provider.reset();
        if (display != EGL_NO_DISPLAY)
            eglTerminate(display);
    }

    static Argus::SensorMode* sensor_mode(Argus::CameraDevice* device, uint32_t index)
    {
        auto* properties = Argus::interface_cast<Argus::ICameraProperties>(device);
        std::vector<Argus::SensorMode*> modes;
        if (!properties || properties->getAllSensorModes(&modes) != Argus::STATUS_OK || index >= modes.size())
            throw std::runtime_error("Argus: sensor mode " + std::to_string(index) + " is not available");
        return modes[index];
    }

    // Session, buffer stream, DMA-BUF pool and repeating request for one camera.
    void open(Camera& camera, Argus::ICameraProvider* provider_iface, uint32_t buffer_count)
    {
        Argus::Status status = Argus::STATUS_OK;
        camera.session.reset(provider_iface->createCaptureSession(camera.device, &status));
        auto* session = Argus::interface_cast<Argus::ICaptureSession>(camera.session);
        if (!session)
            throw std::runtime_error("Argus: createCaptureSession failed (status " +
                                     std::to_string(static_cast<int>(status)) + ")");

        Argus::UniqueObj<Argus::OutputStreamSettings> settings(
            session->createOutputStreamSettings(Argus::STREAM_TYPE_BUFFER));
        auto* settings_iface = Argus::interface_cast<Argus::IOutputStreamSettings>(settings);
        auto* buffer_settings_iface = Argus::interface_cast<Argus::IBufferOutputStreamSettings>(settings);
        if (!settings_iface || !buffer_settings_iface)
            throw std::runtime_error("Argus: createOutputStreamSettings failed");
        settings_iface->setCameraDevice(camera.device);
        check(buffer_settings_iface->setBufferType(Argus::BUFFER_TYPE_EGL_IMAGE), "setBufferType");
        buffer_settings_iface->setMetadataEnable(true);

        camera.stream.reset(session->createOutputStream(settings.get()));
        camera.buffers = Argus::interface_cast<Argus::IBufferOutputStream>(camera.stream);
        if (!camera.buffers)
            throw std::runtime_error("Argus: createOutputStream failed");

        Argus::UniqueObj<Argus::BufferSettings> egl_settings(camera.buffers->createBufferSettings());
        auto* egl_iface = Argus::interface_cast<Argus::IEGLImageBufferSettings>(egl_settings);
        if (!egl_iface)
            throw std::runtime_error("Argus: createBufferSettings failed");
        check(egl_iface->setEGLDisplay(display), "setEGLDisplay");

        for (uint32_t i = 0; i < buffer_count; ++i)
            camera.pool.push_back(make_buffer(camera, egl_settings.get(), egl_iface));

        camera.request.reset(session->createRequest(Argus::CAPTURE_INTENT_PREVIEW));
        auto* request = Argus::interface_cast<Argus::IRequest>(camera.request);
        if (!request)
            throw std::runtime_error("Argus: createRequest failed");
        check(request->enableOutputStream(camera.stream.get()), "enableOutputStream");
        auto* source = Argus::interface_cast<Argus::ISourceSettings>(request->getSourceSettings());
        if (!source)
            throw std::runtime_error("Argus: no source settings");
        check(source->setSensorMode(camera.mode), "setSensorMode");
        // Hold the sensor mode's own frame rate: auto-exposure may otherwise stretch the frame period in dim light.
        const uint64_t period_ns = Argus::interface_cast<Argus::ISensorMode>(camera.mode)->getFrameDurationRange().min();
        check(source->setFrameDurationRange(Argus::Range<uint64_t>(period_ns, period_ns)), "setFrameDurationRange");
    }

    // One DMA-BUF handed to Argus as a capture buffer; it starts out released, ready to be written.
    std::unique_ptr<CaptureBuffer> make_buffer(Camera& camera,
                                               Argus::BufferSettings* egl_settings,
                                               Argus::IEGLImageBufferSettings* egl_iface)
    {
        auto buffer = std::make_unique<CaptureBuffer>();
        NvBufSurf::NvCommonAllocateParams params;
        std::memset(&params, 0, sizeof(params));
        params.memtag = NvBufSurfaceTag_CAMERA;
        params.width = camera.info.width;
        params.height = camera.info.height;
        params.colorFormat = NVBUF_COLOR_FORMAT_NV12;
        params.layout = NVBUF_LAYOUT_BLOCK_LINEAR;
        params.memType = NVBUF_MEM_SURFACE_ARRAY;
        if (NvBufSurf::NvAllocate(&params, 1, &buffer->fd) != 0 ||
            NvBufSurfaceFromFd(buffer->fd, reinterpret_cast<void**>(&buffer->surface)) != 0 || !buffer->surface)
            throw std::runtime_error("Argus: allocating a capture DMA-BUF failed");
        if (NvBufSurfaceMapEglImage(buffer->surface, 0) != 0)
            throw std::runtime_error("Argus: mapping a capture DMA-BUF to an EGL image failed");
        buffer->egl_mapped = true;
        check(egl_iface->setEGLImage(buffer->surface->surfaceList[0].mappedAddr.eglImage), "setEGLImage");

        buffer->argus.reset(camera.buffers->createBuffer(egl_settings));
        auto* iface = Argus::interface_cast<Argus::IBuffer>(buffer->argus);
        if (!iface)
            throw std::runtime_error("Argus: createBuffer failed");
        iface->setClientData(buffer.get());
        check(camera.buffers->releaseBuffer(buffer->argus.get()), "releaseBuffer (initial)");
        return buffer;
    }

    void start()
    {
        if (running)
            return;
        for (auto& camera : cameras)
        {
            auto* session = Argus::interface_cast<Argus::ICaptureSession>(camera->session);
            check(session->repeat(camera->request.get()), "repeat");
        }
        running = true;
    }

    void stop()
    {
        if (!running)
            return;
        running = false;
        for (auto& camera : cameras)
        {
            auto* session = Argus::interface_cast<Argus::ICaptureSession>(camera->session);
            session->stopRepeat();
            session->waitForIdle();
        }
    }
};

ArgusCapture::ArgusCapture(const ArgusConfig& config) : m_impl(std::make_shared<Impl>(config))
{
}

ArgusCapture::~ArgusCapture()
{
    // Outstanding leases keep the Impl, and with it the Argus objects, alive until they are released.
    m_impl->stop();
}

std::vector<CameraInfo> ArgusCapture::query(uint32_t sensor_mode)
{
    Argus::UniqueObj<Argus::CameraProvider> provider(Argus::CameraProvider::create());
    auto* provider_iface = Argus::interface_cast<Argus::ICameraProvider>(provider);
    if (!provider_iface)
        throw std::runtime_error("Argus: no camera provider (is nvargus-daemon running?)");
    std::vector<Argus::CameraDevice*> devices;
    check(provider_iface->getCameraDevices(&devices), "getCameraDevices");

    std::vector<CameraInfo> infos;
    for (uint32_t index = 0; index < devices.size(); ++index)
    {
        const auto* mode = Argus::interface_cast<Argus::ISensorMode>(Impl::sensor_mode(devices[index], sensor_mode));
        const auto size = mode->getResolution();
        const uint64_t period_ns = mode->getFrameDurationRange().min();
        infos.push_back({ index, "argus" + std::to_string(index), size.width(), size.height(),
                          period_ns ? 1e9 / static_cast<double>(period_ns) : 0.0, PixelKind::DmaBufNv12 });
    }
    return infos;
}

const std::vector<CameraInfo>& ArgusCapture::cameras() const
{
    return m_impl->infos;
}

void ArgusCapture::start()
{
    m_impl->start();
}

void ArgusCapture::stop()
{
    m_impl->stop();
}

std::optional<CaptureFrame> ArgusCapture::acquire_latest(uint32_t camera_id)
{
    const auto found = m_impl->by_id.find(camera_id);
    if (found == m_impl->by_id.end())
        throw std::runtime_error("Argus: no camera " + std::to_string(camera_id));
    Impl::Camera& camera = *found->second;

    // Take everything that has arrived and keep the newest: older frames are returned to the camera at once.
    Argus::Buffer* newest = nullptr;
    for (;;)
    {
        Argus::Status status = Argus::STATUS_OK;
        Argus::Buffer* buffer = camera.buffers->acquireBuffer(0, &status);
        if (!buffer)
        {
            if (status != Argus::STATUS_TIMEOUT)
                throw std::runtime_error("Argus: acquireBuffer failed (status " +
                                         std::to_string(static_cast<int>(status)) + ")");
            break;
        }
        if (newest)
            camera.buffers->releaseBuffer(newest);
        newest = buffer;
    }
    if (!newest)
        return std::nullopt;

    CaptureFrame frame;
    frame.camera_id = camera_id;
    frame.width = camera.info.width;
    frame.height = camera.info.height;
    frame.monotonic_ns = clock_ns(CLOCK_MONOTONIC);
    frame.unix_ns = clock_ns(CLOCK_REALTIME);

    auto* iface = Argus::interface_cast<Argus::IBuffer>(newest);
    const auto* data = static_cast<const CaptureBuffer*>(iface->getClientData());
    // The sensor's own timestamp, unscaled and in Argus's clock domain; the capture id numbers the frames.
    const auto* metadata = Argus::interface_cast<const Argus::ICaptureMetadata>(iface->getMetadata());
    if (metadata)
    {
        frame.sensor_time = metadata->getSensorTimestamp();
        frame.sequence = metadata->getCaptureId();
    }
    frame.pixels = DmaBufNv12Pixels{ data->fd };

    // The lease returns the buffer to the camera, and keeps the Argus objects alive until it does.
    frame.lease = FrameLease([impl = m_impl, stream = camera.buffers, newest]() { stream->releaseBuffer(newest); });
    return frame;
}

} // namespace sensing
} // namespace plugins
