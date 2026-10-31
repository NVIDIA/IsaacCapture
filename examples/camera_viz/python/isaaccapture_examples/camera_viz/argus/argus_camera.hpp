// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Argus/Argus.h>
#include <EGL/egl.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cuda.h>
#include <cudaEGL.h>
#include <cuda_runtime.h>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace camera_viz::argus
{

struct ArgusConfig
{
    std::string name;
    std::vector<uint32_t> sensor_ids;
    uint32_t sensor_mode = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    double fps = 30.0;
    int gpu_id = 0;
    bool full_range = false;
    bool swap_uv = false;
    uint32_t acquire_timeout_ms = 0xffffffffu;
    bool repeat_capture = true;
};

struct FrameView
{
    uintptr_t left_ptr = 0;
    size_t left_pitch = 0;
    uintptr_t right_ptr = 0;
    size_t right_pitch = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t timestamp_ns = 0;
    uint64_t sequence = 0;
    bool stereo = false;
};

class ArgusCamera
{
public:
    explicit ArgusCamera(const ArgusConfig& config);
    ~ArgusCamera();

    ArgusCamera(const ArgusCamera&) = delete;
    ArgusCamera& operator=(const ArgusCamera&) = delete;

    void start();
    void stop();
    std::optional<FrameView> latest();

    bool is_stereo() const;
    uint32_t width() const;
    uint32_t height() const;

private:
    struct DeviceBuffer
    {
        uint8_t* ptr = nullptr;
        size_t pitch = 0;
    };

    struct StreamState
    {
        Argus::UniqueObj<Argus::OutputStream> output_stream;
        Argus::IEGLOutputStream* egl_stream = nullptr;
        CUeglStreamConnection connection = nullptr;
    };

    struct AcquiredFrame
    {
        CUgraphicsResource resource = nullptr;
        CUstream stream = nullptr;
        CUeglFrame frame{};
    };

    void initialize();
    void cleanup();
    void producer_loop();
    void connect_cuda_consumers();
    void wait_for_streams_connected();
    void set_failure(const std::string& message);
    void throw_if_failed() const;
    bool acquire(StreamState& stream, AcquiredFrame& out);
    void release(StreamState& stream, AcquiredFrame& acquired);
    std::vector<cudaTextureObject_t> convert_frame(const CUeglFrame& frame, DeviceBuffer& dest);
    void publish(uint32_t write_idx, uint64_t timestamp_ns);
    uint32_t pick_write_index() const;

    Argus::CameraDevice* camera_device(uint32_t sensor_id) const;
    Argus::SensorMode* sensor_mode_for(Argus::CameraDevice* device) const;

    ArgusConfig config_;
    bool stereo_ = false;

    EGLDisplay egl_display_ = EGL_NO_DISPLAY;
    bool egl_initialized_ = false;

    CUdevice cu_device_ = 0;
    CUcontext cu_context_ = nullptr;
    bool cu_context_retained_ = false;
    CUstream convert_stream_ = nullptr;

    Argus::ICameraProvider* i_camera_provider_ = nullptr;
    bool shared_provider_retained_ = false;
    Argus::UniqueObj<Argus::CaptureSession> capture_session_;
    Argus::ICaptureSession* i_capture_session_ = nullptr;
    Argus::UniqueObj<Argus::Request> request_;

    std::vector<Argus::CameraDevice*> camera_devices_;
    std::array<StreamState, 2> streams_;
    size_t stream_count_ = 0;

    std::array<std::array<DeviceBuffer, 3>, 2> buffers_{};

    std::atomic<bool> running_{ false };
    std::atomic<bool> failed_{ false };
    std::thread thread_;

    mutable std::mutex error_mutex_;
    std::string failure_message_;

    mutable std::mutex publish_mutex_;
    int publish_idx_ = -1;
    uint64_t published_sequence_ = 0;
    uint64_t consumed_sequence_ = 0;
    uint64_t published_timestamp_ns_ = 0;
};

} // namespace camera_viz::argus
