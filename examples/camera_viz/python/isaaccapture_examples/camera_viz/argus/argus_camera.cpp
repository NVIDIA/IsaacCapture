// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "argus_camera.hpp"

#include "yuv_to_rgba.cuh"

#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime_api.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace camera_viz::argus
{
namespace
{

void check_cuda(CUresult result, const char* what)
{
    if (result == CUDA_SUCCESS)
    {
        return;
    }
    const char* name = nullptr;
    const char* text = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &text);
    std::ostringstream oss;
    oss << what << " failed";
    if (name)
    {
        oss << ": " << name;
    }
    if (text)
    {
        oss << " (" << text << ")";
    }
    throw std::runtime_error(oss.str());
}

void check_runtime(cudaError_t result, const char* what)
{
    if (result != cudaSuccess)
    {
        std::ostringstream oss;
        oss << what << " failed: " << cudaGetErrorString(result);
        throw std::runtime_error(oss.str());
    }
}

void check_argus(Argus::Status status, const char* what)
{
    if (status != Argus::STATUS_OK)
    {
        std::ostringstream oss;
        oss << what << " failed with Argus status " << static_cast<int>(status);
        throw std::runtime_error(oss.str());
    }
}

constexpr uint32_t kCudaEglInfiniteTimeout = 0xffffffffu;

uint32_t acquire_timeout_us(uint32_t timeout_ms)
{
    if (timeout_ms == kCudaEglInfiniteTimeout)
    {
        return kCudaEglInfiniteTimeout;
    }
    constexpr uint32_t kMaxFiniteTimeoutMs = (kCudaEglInfiniteTimeout - 1U) / 1000U;
    if (timeout_ms > kMaxFiniteTimeoutMs)
    {
        return kCudaEglInfiniteTimeout - 1U;
    }
    return timeout_ms * 1000U;
}

bool is_acquire_timeout(CUresult result)
{
    return result == CUDA_ERROR_TIMEOUT || result == CUDA_ERROR_LAUNCH_TIMEOUT;
}

struct SharedCameraProvider
{
    std::mutex mutex;
    Argus::UniqueObj<Argus::CameraProvider> provider;
    Argus::ICameraProvider* iface = nullptr;
    std::vector<Argus::CameraDevice*> devices;
    size_t refs = 0;
};

SharedCameraProvider g_provider;

void retain_camera_provider(Argus::ICameraProvider*& iface, std::vector<Argus::CameraDevice*>& devices)
{
    std::lock_guard<std::mutex> guard(g_provider.mutex);
    if (g_provider.refs == 0)
    {
        g_provider.provider.reset(Argus::CameraProvider::create());
        g_provider.iface = Argus::interface_cast<Argus::ICameraProvider>(g_provider.provider);
        if (!g_provider.iface)
        {
            g_provider.provider.reset();
            throw std::runtime_error("failed to create Argus CameraProvider");
        }
        check_argus(g_provider.iface->getCameraDevices(&g_provider.devices), "getCameraDevices");
        if (g_provider.devices.empty())
        {
            g_provider.iface = nullptr;
            g_provider.provider.reset();
            throw std::runtime_error("Argus reported no camera devices");
        }
    }
    ++g_provider.refs;
    iface = g_provider.iface;
    devices = g_provider.devices;
}

void release_camera_provider()
{
    std::lock_guard<std::mutex> guard(g_provider.mutex);
    if (g_provider.refs == 0)
    {
        return;
    }
    --g_provider.refs;
    if (g_provider.refs == 0)
    {
        g_provider.devices.clear();
        g_provider.iface = nullptr;
        g_provider.provider.reset();
    }
}

uint64_t monotonic_ns()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool is_yuv420_planar(CUeglColorFormat fmt)
{
    return fmt == CU_EGL_COLOR_FORMAT_YUV420_PLANAR || fmt == CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER ||
           fmt == CU_EGL_COLOR_FORMAT_YUV420_PLANAR_709 || fmt == CU_EGL_COLOR_FORMAT_YUV420_PLANAR_2020;
}

bool is_yvu420_planar(CUeglColorFormat fmt)
{
    return fmt == CU_EGL_COLOR_FORMAT_YVU420_PLANAR || fmt == CU_EGL_COLOR_FORMAT_YVU420_PLANAR_ER ||
           fmt == CU_EGL_COLOR_FORMAT_YVU420_PLANAR_709 || fmt == CU_EGL_COLOR_FORMAT_YVU420_PLANAR_2020;
}

bool is_yuv420_semiplanar(CUeglColorFormat fmt)
{
    return fmt == CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR || fmt == CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER ||
           fmt == CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_709 || fmt == CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_2020;
}

bool is_yvu420_semiplanar(CUeglColorFormat fmt)
{
    return fmt == CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR || fmt == CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_ER ||
           fmt == CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_709 || fmt == CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_2020;
}

bool is_extended_range(CUeglColorFormat fmt)
{
    return fmt == CU_EGL_COLOR_FORMAT_YUV420_PLANAR_ER || fmt == CU_EGL_COLOR_FORMAT_YVU420_PLANAR_ER ||
           fmt == CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER || fmt == CU_EGL_COLOR_FORMAT_YVU420_SEMIPLANAR_ER;
}

YuvLayout layout_for(CUeglColorFormat fmt)
{
    if (is_yuv420_planar(fmt))
    {
        return YuvLayout::YUV420Planar;
    }
    if (is_yvu420_planar(fmt))
    {
        return YuvLayout::YVU420Planar;
    }
    if (is_yuv420_semiplanar(fmt))
    {
        return YuvLayout::YUV420SemiPlanar;
    }
    if (is_yvu420_semiplanar(fmt))
    {
        return YuvLayout::YVU420SemiPlanar;
    }
    std::ostringstream oss;
    oss << "unsupported Argus CUDA EGL color format " << static_cast<int>(fmt);
    throw std::runtime_error(oss.str());
}

bool is_planar(YuvLayout layout)
{
    return layout == YuvLayout::YUV420Planar || layout == YuvLayout::YVU420Planar;
}

YuvLayout swapped_uv_layout(YuvLayout layout)
{
    switch (layout)
    {
    case YuvLayout::YUV420Planar:
        return YuvLayout::YVU420Planar;
    case YuvLayout::YVU420Planar:
        return YuvLayout::YUV420Planar;
    case YuvLayout::YUV420SemiPlanar:
        return YuvLayout::YVU420SemiPlanar;
    case YuvLayout::YVU420SemiPlanar:
        return YuvLayout::YUV420SemiPlanar;
    }
    return layout;
}

cudaTextureObject_t texture_for_array(CUarray array)
{
    cudaResourceDesc resource_desc{};
    resource_desc.resType = cudaResourceTypeArray;
    resource_desc.res.array.array = reinterpret_cast<cudaArray_t>(array);

    cudaTextureDesc texture_desc{};
    texture_desc.addressMode[0] = cudaAddressModeClamp;
    texture_desc.addressMode[1] = cudaAddressModeClamp;
    texture_desc.filterMode = cudaFilterModePoint;
    texture_desc.readMode = cudaReadModeElementType;
    texture_desc.normalizedCoords = 0;

    cudaTextureObject_t texture = 0;
    check_runtime(cudaCreateTextureObject(&texture, &resource_desc, &texture_desc, nullptr), "cudaCreateTextureObject");
    return texture;
}

} // namespace

ArgusCamera::ArgusCamera(const ArgusConfig& config) : config_(config)
{
    if (config_.sensor_ids.empty() || config_.sensor_ids.size() > 2)
    {
        throw std::invalid_argument("ArgusConfig.sensor_ids must contain one or two sensor ids");
    }
    if (config_.width == 0 || config_.height == 0)
    {
        throw std::invalid_argument("ArgusConfig.width/height must be non-zero");
    }
    stereo_ = config_.sensor_ids.size() == 2;
    stream_count_ = config_.sensor_ids.size();
}

ArgusCamera::~ArgusCamera()
{
    stop();
}

void ArgusCamera::start()
{
    if (running_.load())
    {
        return;
    }
    initialize();
    running_.store(true);
    thread_ = std::thread(&ArgusCamera::producer_loop, this);
}

void ArgusCamera::stop()
{
    if (!running_.exchange(false))
    {
        cleanup();
        return;
    }

    if (i_capture_session_)
    {
        i_capture_session_->stopRepeat();
        i_capture_session_->waitForIdle();
    }
    for (size_t i = 0; i < stream_count_; ++i)
    {
        if (streams_[i].egl_stream)
        {
            streams_[i].egl_stream->disconnect();
        }
    }
    if (thread_.joinable())
    {
        thread_.join();
    }
    cleanup();
}

std::optional<FrameView> ArgusCamera::latest()
{
    throw_if_failed();
    std::lock_guard<std::mutex> guard(publish_mutex_);
    if (publish_idx_ < 0 || consumed_sequence_ == published_sequence_)
    {
        return std::nullopt;
    }
    consumed_sequence_ = published_sequence_;

    FrameView view;
    view.left_ptr = reinterpret_cast<uintptr_t>(buffers_[0][publish_idx_].ptr);
    view.left_pitch = buffers_[0][publish_idx_].pitch;
    view.width = config_.width;
    view.height = config_.height;
    view.timestamp_ns = published_timestamp_ns_;
    view.sequence = published_sequence_;
    view.stereo = stereo_;
    if (stereo_)
    {
        view.right_ptr = reinterpret_cast<uintptr_t>(buffers_[1][publish_idx_].ptr);
        view.right_pitch = buffers_[1][publish_idx_].pitch;
    }
    return view;
}

bool ArgusCamera::is_stereo() const
{
    return stereo_;
}

uint32_t ArgusCamera::width() const
{
    return config_.width;
}

uint32_t ArgusCamera::height() const
{
    return config_.height;
}

void ArgusCamera::initialize()
{
    cleanup();
    failed_.store(false);
    {
        std::lock_guard<std::mutex> guard(error_mutex_);
        failure_message_.clear();
    }

    check_cuda(cuInit(0), "cuInit");
    check_cuda(cuDeviceGet(&cu_device_, config_.gpu_id), "cuDeviceGet");
    check_cuda(cuDevicePrimaryCtxRetain(&cu_context_, cu_device_), "cuDevicePrimaryCtxRetain");
    cu_context_retained_ = true;
    check_cuda(cuCtxSetCurrent(cu_context_), "cuCtxSetCurrent");
    check_cuda(cuStreamCreate(&convert_stream_, CU_STREAM_NON_BLOCKING), "cuStreamCreate");

    retain_camera_provider(i_camera_provider_, camera_devices_);
    shared_provider_retained_ = true;

    std::vector<Argus::CameraDevice*> selected;
    selected.reserve(stream_count_);
    for (uint32_t sensor_id : config_.sensor_ids)
    {
        selected.push_back(camera_device(sensor_id));
    }

    capture_session_.reset(stream_count_ == 1 ? i_camera_provider_->createCaptureSession(selected[0]) :
                                                i_camera_provider_->createCaptureSession(selected));
    i_capture_session_ = Argus::interface_cast<Argus::ICaptureSession>(capture_session_);
    if (!i_capture_session_)
    {
        throw std::runtime_error("failed to create Argus CaptureSession");
    }

    for (size_t i = 0; i < stream_count_; ++i)
    {
        Argus::UniqueObj<Argus::OutputStreamSettings> settings(
            i_capture_session_->createOutputStreamSettings(Argus::STREAM_TYPE_EGL));
        auto* i_settings = Argus::interface_cast<Argus::IOutputStreamSettings>(settings);
        auto* i_egl_settings = Argus::interface_cast<Argus::IEGLOutputStreamSettings>(settings);
        if (!i_settings || !i_egl_settings)
        {
            throw std::runtime_error("failed to create Argus EGL OutputStreamSettings");
        }
        check_argus(i_settings->setCameraDevice(selected[i]), "setCameraDevice");
        check_argus(i_egl_settings->setPixelFormat(Argus::PIXEL_FMT_YCbCr_420_888), "setPixelFormat");
        check_argus(
            i_egl_settings->setResolution(Argus::Size2D<uint32_t>(config_.width, config_.height)), "setResolution");
        // Leave EGLDisplay unset for CUDA-only consumption. NVIDIA's cudaHistogram
        // sample uses a display-agnostic EGLStream for CUDA consumers; binding a
        // default EGL display can fail on headless/OpenXR-only Thor setups.
        check_argus(i_egl_settings->setMode(Argus::EGL_STREAM_MODE_MAILBOX), "setMode MAILBOX");

        streams_[i].output_stream.reset(i_capture_session_->createOutputStream(settings.get()));
        streams_[i].egl_stream = Argus::interface_cast<Argus::IEGLOutputStream>(streams_[i].output_stream);
        if (!streams_[i].egl_stream)
        {
            throw std::runtime_error("failed to create Argus EGL OutputStream");
        }
    }

    request_.reset(i_capture_session_->createRequest());
    auto* i_request = Argus::interface_cast<Argus::IRequest>(request_);
    if (!i_request)
    {
        throw std::runtime_error("failed to create Argus Request");
    }
    auto* i_source_settings = Argus::interface_cast<Argus::ISourceSettings>(i_request->getSourceSettings());
    if (!i_source_settings)
    {
        throw std::runtime_error("failed to get Argus ISourceSettings");
    }
    check_argus(i_source_settings->setSensorMode(sensor_mode_for(selected[0])), "setSensorMode");
    // Leave frame duration at the sensor-mode default. For SHW5G this mirrors
    // NVIDIA's cudaHistogram sample, which succeeds on the same Argus device.
    for (size_t i = 0; i < stream_count_; ++i)
    {
        check_argus(i_request->enableOutputStream(streams_[i].output_stream.get()), "enableOutputStream");
    }

    for (size_t eye = 0; eye < stream_count_; ++eye)
    {
        for (auto& buffer : buffers_[eye])
        {
            CUdeviceptr ptr = 0;
            check_cuda(cuMemAllocPitch(&ptr, &buffer.pitch, config_.width * 4, config_.height, 4), "cuMemAllocPitch");
            buffer.ptr = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(ptr));
        }
    }
}

void ArgusCamera::connect_cuda_consumers()
{
    for (size_t i = 0; i < stream_count_; ++i)
    {
        if (!streams_[i].egl_stream)
        {
            throw std::runtime_error("Argus EGL stream missing while connecting CUDA consumer");
        }
        if (!streams_[i].connection)
        {
            std::ostringstream label;
            label << "cuEGLStreamConsumerConnect stream " << i;
            check_cuda(cuEGLStreamConsumerConnect(&streams_[i].connection, streams_[i].egl_stream->getEGLStream()),
                       label.str().c_str());
        }
    }
}

void ArgusCamera::wait_for_streams_connected()
{
    for (size_t i = 0; i < stream_count_; ++i)
    {
        if (!streams_[i].egl_stream)
        {
            throw std::runtime_error("Argus EGL stream missing while waiting for connection");
        }
        std::ostringstream label;
        label << "waitUntilConnected stream " << i;
        check_argus(streams_[i].egl_stream->waitUntilConnected(), label.str().c_str());
    }
}

void ArgusCamera::set_failure(const std::string& message)
{
    {
        std::lock_guard<std::mutex> guard(error_mutex_);
        failure_message_ = message;
    }
    failed_.store(true);
}

void ArgusCamera::throw_if_failed() const
{
    if (!failed_.load())
    {
        return;
    }
    std::lock_guard<std::mutex> guard(error_mutex_);
    throw std::runtime_error(failure_message_.empty() ? "Argus producer failed" : failure_message_);
}

void ArgusCamera::cleanup()
{
    if (thread_.joinable())
    {
        thread_.join();
    }

    if (i_capture_session_)
    {
        i_capture_session_->stopRepeat();
        i_capture_session_->waitForIdle();
    }

    for (size_t i = 0; i < streams_.size(); ++i)
    {
        if (streams_[i].connection)
        {
            cuEGLStreamConsumerDisconnect(&streams_[i].connection);
            streams_[i].connection = nullptr;
        }
        if (streams_[i].egl_stream)
        {
            streams_[i].egl_stream->disconnect();
            streams_[i].egl_stream = nullptr;
        }
        streams_[i].output_stream.reset();
    }

    request_.reset();
    capture_session_.reset();
    i_capture_session_ = nullptr;
    camera_devices_.clear();
    i_camera_provider_ = nullptr;
    if (shared_provider_retained_)
    {
        release_camera_provider();
        shared_provider_retained_ = false;
    }

    for (auto& eye : buffers_)
    {
        for (auto& buffer : eye)
        {
            if (buffer.ptr)
            {
                cuMemFree(static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(buffer.ptr)));
                buffer.ptr = nullptr;
                buffer.pitch = 0;
            }
        }
    }

    if (convert_stream_)
    {
        cuStreamDestroy(convert_stream_);
        convert_stream_ = nullptr;
    }

    if (egl_initialized_)
    {
        eglTerminate(egl_display_);
        egl_initialized_ = false;
    }
    egl_display_ = EGL_NO_DISPLAY;

    if (cu_context_retained_)
    {
        cuDevicePrimaryCtxRelease(cu_device_);
        cu_context_retained_ = false;
        cu_context_ = nullptr;
    }

    {
        std::lock_guard<std::mutex> guard(publish_mutex_);
        publish_idx_ = -1;
        published_sequence_ = 0;
        consumed_sequence_ = 0;
        published_timestamp_ns_ = 0;
    }
}

void ArgusCamera::producer_loop()
{
    bool repeat_active = false;
    try
    {
        check_cuda(cuCtxSetCurrent(cu_context_), "cuCtxSetCurrent producer");
        connect_cuda_consumers();
        if (config_.repeat_capture)
        {
            const Argus::Status repeat_status = i_capture_session_->repeat(request_.get());
            if (repeat_status == Argus::STATUS_OK)
            {
                repeat_active = true;
            }
            else
            {
                std::cerr << "[argus] repeat() failed with Argus status " << static_cast<int>(repeat_status)
                          << "; falling back to per-frame capture()" << std::endl;
            }
        }

        while (running_.load())
        {
            const uint32_t write_idx = pick_write_index();
            std::array<AcquiredFrame, 2> acquired{};
            if (!repeat_active)
            {
                Argus::Status capture_status = Argus::STATUS_OK;
                const uint64_t capture_timeout_ns = 1000000000ULL;
                const uint32_t capture_id =
                    i_capture_session_->capture(request_.get(), capture_timeout_ns, &capture_status);
                if (capture_id == 0 || capture_status != Argus::STATUS_OK)
                {
                    check_argus(capture_status, "capture");
                    throw std::runtime_error("Argus capture request timed out before submission");
                }
            }

            bool ok = true;
            for (size_t eye = 0; eye < stream_count_; ++eye)
            {
                if (!acquire(streams_[eye], acquired[eye]))
                {
                    ok = false;
                    break;
                }
            }
            if (ok)
            {
                std::vector<cudaTextureObject_t> pending_textures;
                for (size_t eye = 0; eye < stream_count_; ++eye)
                {
                    auto textures = convert_frame(acquired[eye].frame, buffers_[eye][write_idx]);
                    pending_textures.insert(pending_textures.end(), textures.begin(), textures.end());
                }
                check_cuda(cuStreamSynchronize(convert_stream_), "cuStreamSynchronize");
                check_runtime(cudaGetLastError(), "Argus YUV to RGBA kernel");
                for (cudaTextureObject_t texture : pending_textures)
                {
                    if (texture)
                    {
                        check_runtime(cudaDestroyTextureObject(texture), "cudaDestroyTextureObject");
                    }
                }
                const uint64_t ts = monotonic_ns();
                for (size_t eye = 0; eye < stream_count_; ++eye)
                {
                    release(streams_[eye], acquired[eye]);
                }
                publish(write_idx, ts);
            }
            else
            {
                for (size_t eye = 0; eye < stream_count_; ++eye)
                {
                    release(streams_[eye], acquired[eye]);
                }
            }
        }
    }
    catch (const std::exception& e)
    {
        std::ostringstream oss;
        oss << "Argus producer error: " << e.what();
        std::cerr << "[argus] " << oss.str() << std::endl;
        set_failure(oss.str());
        running_.store(false);
        if (i_capture_session_)
        {
            i_capture_session_->stopRepeat();
            i_capture_session_->waitForIdle();
        }
    }
    catch (...)
    {
        const std::string msg = "Argus producer error: unknown exception";
        std::cerr << "[argus] " << msg << std::endl;
        set_failure(msg);
        running_.store(false);
        if (i_capture_session_)
        {
            i_capture_session_->stopRepeat();
            i_capture_session_->waitForIdle();
        }
    }
}

bool ArgusCamera::acquire(StreamState& stream, AcquiredFrame& out)
{
    if (!stream.connection)
    {
        return false;
    }
    const uint32_t timeout_us = acquire_timeout_us(config_.acquire_timeout_ms);
    CUresult result = cuEGLStreamConsumerAcquireFrame(&stream.connection, &out.resource, &out.stream, timeout_us);
    if (is_acquire_timeout(result))
    {
        return false;
    }
    if (result != CUDA_SUCCESS)
    {
        if (!running_.load())
        {
            return false;
        }
        check_cuda(result, "cuEGLStreamConsumerAcquireFrame");
    }
    result = cuGraphicsResourceGetMappedEglFrame(&out.frame, out.resource, 0, 0);
    check_cuda(result, "cuGraphicsResourceGetMappedEglFrame");
    return true;
}

void ArgusCamera::release(StreamState& stream, AcquiredFrame& acquired)
{
    if (acquired.resource && stream.connection)
    {
        cuEGLStreamConsumerReleaseFrame(&stream.connection, acquired.resource, &acquired.stream);
        acquired.resource = nullptr;
        acquired.stream = nullptr;
        std::memset(&acquired.frame, 0, sizeof(acquired.frame));
    }
}

std::vector<cudaTextureObject_t> ArgusCamera::convert_frame(const CUeglFrame& frame, DeviceBuffer& dest)
{
    std::vector<cudaTextureObject_t> pending_textures;
    if (frame.width != config_.width || frame.height != config_.height)
    {
        std::ostringstream oss;
        oss << "Argus frame size " << frame.width << "x" << frame.height << " does not match configured "
            << config_.width << "x" << config_.height;
        throw std::runtime_error(oss.str());
    }
    if (frame.cuFormat != CU_AD_FORMAT_UNSIGNED_INT8)
    {
        throw std::runtime_error("Argus frame is not unsigned 8-bit YUV");
    }

    YuvLayout layout = layout_for(frame.eglColorFormat);
    if (config_.swap_uv)
    {
        layout = swapped_uv_layout(layout);
    }
    const bool full_range = config_.full_range || is_extended_range(frame.eglColorFormat);
    const bool planar = is_planar(layout);
    const uint32_t required_planes = planar ? 3 : 2;
    if (frame.planeCount < required_planes)
    {
        std::ostringstream oss;
        oss << "Argus frame has " << frame.planeCount << " plane(s), expected at least " << required_planes;
        throw std::runtime_error(oss.str());
    }

    if (frame.frameType == CU_EGL_FRAME_TYPE_PITCH)
    {
        if (planar)
        {
            throw std::runtime_error(
                "planar Argus pitch frames are unsupported because CUeglFrame exposes only first-plane pitch");
        }
        const auto* y_plane = static_cast<const uint8_t*>(frame.frame.pPitch[0]);
        const auto* uv_or_u_plane = static_cast<const uint8_t*>(frame.frame.pPitch[1]);
        const int y_pitch = static_cast<int>(frame.pitch);
        const int uv_pitch = static_cast<int>(frame.pitch);
        launch_yuv420_pitch_to_rgba(y_plane, uv_or_u_plane, nullptr, y_pitch, uv_pitch, 0, config_.width,
                                    config_.height, dest.ptr, static_cast<int>(dest.pitch), layout, full_range,
                                    reinterpret_cast<cudaStream_t>(convert_stream_));
        return pending_textures;
    }

    if (frame.frameType == CU_EGL_FRAME_TYPE_ARRAY)
    {
        cudaTextureObject_t y_tex = texture_for_array(frame.frame.pArray[0]);
        pending_textures.push_back(y_tex);
        cudaTextureObject_t uv_or_u_tex = texture_for_array(frame.frame.pArray[1]);
        pending_textures.push_back(uv_or_u_tex);
        cudaTextureObject_t v_tex = 0;
        if (planar)
        {
            v_tex = texture_for_array(frame.frame.pArray[2]);
            pending_textures.push_back(v_tex);
        }
        launch_yuv420_array_to_rgba(y_tex, uv_or_u_tex, v_tex, config_.width, config_.height, dest.ptr,
                                    static_cast<int>(dest.pitch), layout, full_range,
                                    reinterpret_cast<cudaStream_t>(convert_stream_));
        return pending_textures;
    }

    throw std::runtime_error("unsupported Argus CUDA EGL frame type");
}

void ArgusCamera::publish(uint32_t write_idx, uint64_t timestamp_ns)
{
    std::lock_guard<std::mutex> guard(publish_mutex_);
    publish_idx_ = static_cast<int>(write_idx);
    published_timestamp_ns_ = timestamp_ns;
    ++published_sequence_;
}

uint32_t ArgusCamera::pick_write_index() const
{
    std::lock_guard<std::mutex> guard(publish_mutex_);
    if (publish_idx_ < 0)
    {
        return 0;
    }
    return static_cast<uint32_t>((publish_idx_ + 1) % 3);
}

Argus::CameraDevice* ArgusCamera::camera_device(uint32_t sensor_id) const
{
    if (sensor_id >= camera_devices_.size())
    {
        std::ostringstream oss;
        oss << "Argus sensor_id " << sensor_id << " is out of range; Argus reports " << camera_devices_.size()
            << " camera device(s)";
        throw std::out_of_range(oss.str());
    }
    return camera_devices_[sensor_id];
}

Argus::SensorMode* ArgusCamera::sensor_mode_for(Argus::CameraDevice* device) const
{
    auto* props = Argus::interface_cast<Argus::ICameraProperties>(device);
    if (!props)
    {
        throw std::runtime_error("failed to get Argus ICameraProperties");
    }
    std::vector<Argus::SensorMode*> modes;
    check_argus(props->getAllSensorModes(&modes), "getAllSensorModes");
    if (modes.empty())
    {
        throw std::runtime_error("Argus camera device has no sensor modes");
    }
    if (config_.sensor_mode >= modes.size())
    {
        std::ostringstream oss;
        oss << "Argus sensor_mode " << config_.sensor_mode << " is out of range; camera has " << modes.size()
            << " mode(s)";
        throw std::out_of_range(oss.str());
    }
    return modes[config_.sensor_mode];
}

} // namespace camera_viz::argus
