// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Argus buffer-stream capture into NvBufSurface DMA-BUFs. The Argus calls follow the Jetson Multimedia API's
// client-owned-buffer pattern. Built and run on an AGX Orin by a reviewer before the acquisition threads and the
// bounded teardown were added; those have not been built or run.

#include "inc/sensing_core/argus_capture.hpp"

#include <Argus/Argus.h>
#include <Argus/BufferStream.h>
#include <Argus/EGLImage.h>
#include <EGL/egl.h>
#include <log_bridge/logger.hpp>

#include <NvBufSurface.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

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

// A blocking acquire returns as soon as a frame arrives; the wait only bounds how long a camera with no frames
// (an empty port) takes to notice a stop. The value needs validating on the Orin.
constexpr uint64_t kAcquireTimeoutNs = 100'000'000;
// Per camera, how long a stopped session may take to go idle before it is treated as hung.
constexpr uint64_t kIdleTimeoutNs = 3'000'000'000;
// Whole-teardown budget. An SDK call that never returns cannot be cancelled, so past this the process ends: no
// cleanup may run on state the camera hardware could still be writing to.
constexpr std::chrono::seconds kStopDeadline{ 20 };

std::shared_ptr<isaaccapture::Logger> logger()
{
    return isaaccapture::Logger::get("isaaccapture.plugins.sensing.ArgusCapture");
}

// The SDK call teardown is in, recorded before the call is made so a call that never returns is still named.
struct ActiveCall
{
    std::atomic<const char*> what{ nullptr };
    std::atomic<int64_t> camera{ -1 };
};
ActiveCall g_active_call;

[[noreturn]] void fatal(const char* what, int64_t camera_id)
{
    std::fprintf(stderr, "Argus: %s (camera %lld); ending the process\n", what, static_cast<long long>(camera_id));
    std::_Exit(EXIT_FAILURE);
}

// Ends the process if teardown does not finish in time. Scope it around the blocking SDK calls.
class StopWatchdog
{
public:
    explicit StopWatchdog(std::chrono::seconds limit)
        : m_thread(
              [this, limit]
              {
                  std::unique_lock<std::mutex> lock(m_mutex);
                  if (!m_cv.wait_for(lock, limit, [this] { return m_done; }))
                  {
                      const char* what = g_active_call.what.load();
                      fatal((std::string("teardown did not finish in ") + (what ? what : "an unknown step")).c_str(),
                            g_active_call.camera.load());
                  }
              })
    {
    }
    StopWatchdog(const StopWatchdog&) = delete;
    StopWatchdog& operator=(const StopWatchdog&) = delete;
    ~StopWatchdog()
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_done = true;
        }
        m_cv.notify_all();
        m_thread.join();
    }

private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_done = false;
    std::thread m_thread; // last: it starts using the members above
};

// Runs one blocking SDK call and names it in the log when it is slow, so a hang points at the exact call.
template <typename F>
auto timed(const char* what, int64_t camera_id, F&& call)
{
    g_active_call.camera.store(camera_id);
    g_active_call.what.store(what);
    const auto begin = std::chrono::steady_clock::now();
    struct Guard
    {
        const char* what;
        int64_t camera_id;
        std::chrono::steady_clock::time_point begin;
        ~Guard()
        {
            g_active_call.what.store(nullptr);
            const auto ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin);
            if (ms.count() >= 1000)
                logger()->warn("camera {}: {} took {} ms", camera_id, what, ms.count());
        }
    } guard{ what, camera_id, begin };
    return call();
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

struct ArgusCapture::Impl : std::enable_shared_from_this<ArgusCapture::Impl>
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
        /// Numbers frames whose capture has no metadata, so the consumer never mistakes them for one re-read frame.
        uint32_t fallback_sequence = 0;

        /// This camera's acquisition thread: it blocks on the buffer stream, so a camera that delivers nothing
        /// holds up no other camera. It keeps only the newest frame; a replaced one goes back to the camera.
        std::thread worker;
        std::mutex slot_mutex;
        std::optional<CaptureFrame> latest; ///< guarded by slot_mutex
        std::string error; ///< why the worker stopped, guarded by slot_mutex
    };

    // An initialised EGL display is released by this member whether construction completes or throws: a destructor
    // that never runs for a failed constructor cannot be the only place that calls eglTerminate.
    struct EglDisplay
    {
        EGLDisplay handle = EGL_NO_DISPLAY;
        EglDisplay() = default;
        EglDisplay(const EglDisplay&) = delete;
        EglDisplay& operator=(const EglDisplay&) = delete;
        ~EglDisplay()
        {
            terminate();
        }
        void terminate()
        {
            if (handle != EGL_NO_DISPLAY)
                eglTerminate(std::exchange(handle, EGL_NO_DISPLAY));
        }
    };

    Argus::UniqueObj<Argus::CameraProvider> provider;
    EglDisplay egl;
    std::vector<std::unique_ptr<Camera>> cameras;
    std::vector<CameraInfo> infos;
    std::map<uint32_t, Camera*> by_id;
    bool running = false;
    std::atomic<bool> stopping{ false };

    Impl(const ArgusConfig& config)
    {
        // Peak demand per camera: the encoder's pinned buffers, the output's pending frame, its worker's frame and
        // one fresh acquire.
        if (config.buffer_count < kEncoderInputBuffers + 3)
            throw std::runtime_error("Argus: buffer_count must be at least " + std::to_string(kEncoderInputBuffers + 3) +
                                     " so the camera never waits for a buffer the encoder holds");
        provider.reset(Argus::CameraProvider::create());
        auto* provider_iface = Argus::interface_cast<Argus::ICameraProvider>(provider);
        if (!provider_iface)
            throw std::runtime_error("Argus: no camera provider (is nvargus-daemon running?)");

        egl.handle = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (egl.handle == EGL_NO_DISPLAY || !eglInitialize(egl.handle, nullptr, nullptr))
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
            if (!mode_iface)
                throw std::runtime_error("Argus: sensor mode has no mode interface");
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
        // Destroying the Argus objects and the EGL display is SDK work too and can hang: one deadline covers all of it.
        const StopWatchdog watchdog(kStopDeadline);
        g_active_call.camera.store(-1);
        g_active_call.what.store("destroying the camera objects");
        for (auto& camera : cameras)
        {
            camera->request.reset();
            camera->pool.clear(); // Argus buffers, then the DMA-BUFs behind them
            camera->stream.reset();
            camera->session.reset();
        }
        cameras.clear();
        provider.reset();
        g_active_call.what.store("terminating the EGL display");
        egl.terminate(); // after the Argus objects above; the member's destructor then has nothing left to do
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
        check(egl_iface->setEGLDisplay(egl.handle), "setEGLDisplay");

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
        // Any failure below undoes everything started so far, so no thread or repeating session outlives it.
        std::vector<Camera*> started;
        stopping = false;
        try
        {
            for (auto& camera : cameras)
            {
                auto* session = Argus::interface_cast<Argus::ICaptureSession>(camera->session);
                check(session->repeat(camera->request.get()), "repeat");
                started.push_back(camera.get());
            }
            for (auto& camera : cameras)
            {
                camera->error.clear();
                camera->worker = std::thread([this, cam = camera.get()] { acquire_loop(*cam); });
            }
        }
        catch (...)
        {
            teardown(started);
            throw;
        }
        running = true;
    }

    void stop()
    {
        if (!running)
            return;
        running = false;
        std::vector<Camera*> all;
        for (auto& camera : cameras)
            all.push_back(camera.get());
        teardown(all);
    }

    // The one teardown path, for stop() and for a failed start(), under one deadline: end the acquisition threads and
    // return the frames they hold (a held frame's lease owns this Impl, so that cycle must always be broken here),
    // then stop the sessions.
    void teardown(const std::vector<Camera*>& sessions)
    {
        const StopWatchdog watchdog(kStopDeadline);
        // Acquisition threads end within one acquire timeout; none may still hold a buffer when the sessions stop.
        stopping = true;
        for (auto& camera : cameras)
            if (camera->worker.joinable())
                timed("joining the acquisition thread", camera->info.id,
                      [&]
                      {
                          camera->worker.join();
                          return 0;
                      });
        // Frames waiting in the slots go back to their streams first: an in-flight request may need a free buffer to
        // finish, and waitForIdle would wait on it. Buffers the encoder still holds stay with it.
        for (auto& camera : cameras)
        {
            std::optional<CaptureFrame> dropped;
            {
                std::lock_guard<std::mutex> lock(camera->slot_mutex);
                dropped = std::move(camera->latest);
                camera->latest.reset();
            }
            timed("returning a held frame", camera->info.id,
                  [&]
                  {
                      dropped.reset();
                      return 0;
                  });
        }
        // Stop every session before waiting on any, so one camera's slow stop does not delay the others'.
        for (Camera* camera : sessions)
        {
            auto* session = Argus::interface_cast<Argus::ICaptureSession>(camera->session);
            timed("stopRepeat", camera->info.id, [&] { return session->stopRepeat(); });
        }
        for (Camera* camera : sessions)
        {
            auto* session = Argus::interface_cast<Argus::ICaptureSession>(camera->session);
            const Argus::Status status =
                timed("waitForIdle", camera->info.id, [&] { return session->waitForIdle(kIdleTimeoutNs); });
            // The hardware may still own the session's buffers and a retry in this process would pile new sessions on
            // top of it. This executable is dedicated to the cameras, so a session that will not stop ends the
            // process and the service supervisor restarts it.
            if (status != Argus::STATUS_OK)
                fatal("a session did not go idle after stopRepeat", camera->info.id);
        }
    }

    // Worker thread: block for each frame, stamp it on arrival, and keep only the newest.
    void acquire_loop(Camera& camera)
    {
        try
        {
            while (!stopping.load())
            {
                Argus::Status status = Argus::STATUS_OK;
                Argus::Buffer* buffer = camera.buffers->acquireBuffer(kAcquireTimeoutNs, &status);
                if (!buffer)
                {
                    if (status == Argus::STATUS_TIMEOUT)
                        continue;
                    if (!stopping.load())
                        fail(camera,
                             "Argus: acquireBuffer failed (status " + std::to_string(static_cast<int>(status)) + ")");
                    return;
                }
                CaptureFrame frame = make_frame(camera, buffer);
                std::optional<CaptureFrame> replaced;
                {
                    std::lock_guard<std::mutex> lock(camera.slot_mutex);
                    replaced = std::move(camera.latest);
                    camera.latest = std::move(frame);
                }
                // `replaced` is destroyed here, outside the lock: its lease returns the buffer to the camera.
            }
        }
        catch (const std::exception& e)
        {
            fail(camera, e.what());
        }
    }

    void fail(Camera& camera, const std::string& what)
    {
        std::lock_guard<std::mutex> lock(camera.slot_mutex);
        if (camera.error.empty())
            camera.error = what;
    }

    // The frame for a buffer that has just arrived; its stamps are the arrival time.
    CaptureFrame make_frame(Camera& camera, Argus::Buffer* buffer)
    {
        CaptureFrame frame;
        frame.camera_id = camera.info.id;
        frame.width = camera.info.width;
        frame.height = camera.info.height;
        frame.monotonic_ns = clock_ns(CLOCK_MONOTONIC);
        frame.unix_ns = clock_ns(CLOCK_REALTIME);

        auto* iface = Argus::interface_cast<Argus::IBuffer>(buffer);
        const auto* data = static_cast<const CaptureBuffer*>(iface->getClientData());
        // The sensor's own timestamp, unscaled and in Argus's clock domain; the capture id numbers the frames.
        const auto* metadata = Argus::interface_cast<const Argus::ICaptureMetadata>(iface->getMetadata());
        if (metadata)
        {
            frame.sensor_time = metadata->getSensorTimestamp();
            frame.sequence = metadata->getCaptureId();
        }
        else
        {
            frame.sequence = ++camera.fallback_sequence;
        }
        frame.pixels = DmaBufNv12Pixels{ data->fd };

        // The lease returns the buffer to the camera, and keeps the Argus objects alive until it does.
        frame.lease = FrameLease([self = shared_from_this(), stream = camera.buffers, buffer]()
                                 { stream->releaseBuffer(buffer); });
        return frame;
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
        // A device without this mode cannot be opened with it, but must not hide the other cameras.
        auto* properties = Argus::interface_cast<Argus::ICameraProperties>(devices[index]);
        std::vector<Argus::SensorMode*> modes;
        if (!properties || properties->getAllSensorModes(&modes) != Argus::STATUS_OK || sensor_mode >= modes.size())
            continue;
        const auto* mode = Argus::interface_cast<Argus::ISensorMode>(modes[sensor_mode]);
        if (!mode)
            continue;
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

    // The camera's own thread keeps the newest frame; take it, if there is one.
    std::lock_guard<std::mutex> lock(camera.slot_mutex);
    if (!camera.error.empty())
        throw std::runtime_error(camera.error);
    std::optional<CaptureFrame> frame = std::move(camera.latest);
    camera.latest.reset();
    return frame;
}

} // namespace sensing
} // namespace plugins
