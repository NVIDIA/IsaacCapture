// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Mock backend for SiplCamera: synthetic frames from a CUDA pattern generator,
// no SIPL, NvSci or EGL. It keeps the contract the rest of the plugin relies on
// -- per-sensor capture threads, a three-slot RGBA mailbox with a reader lease,
// a TSC stamp shared by every sensor, and failures surfacing from latest() --
// and runs the real YUV-to-RGBA kernel on the synthetic NV12, so the conversion
// code is exercised too.
//
// Environment (read once, at construction):
//   SENSING_MOCK_FPS=N          override the platform config's frame rate
//   SENSING_MOCK_FAIL_AFTER=N   every sensor fails after N frames, as a link drop would
//   SENSING_MOCK_SCALE_DIV=N    shrink the reported geometry (see mock_platform_config.hpp)

#include "mock_nv12_source.cuh"
#include "mock_platform_config.hpp"
#include "sipl_camera.hpp"
#include "yuv_to_rgba.cuh"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace plugins
{
namespace sensing
{

namespace
{

// Arbitrary offset so nothing can mistake the TSC for CLOCK_MONOTONIC.
constexpr uint64_t kMockTscBaseNs = 1'000'000'000'000'000ull;

uint64_t monotonic_ns()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

double env_number(const char* name, double fallback)
{
    const char* value = std::getenv(name);
    return value ? std::strtod(value, nullptr) : fallback;
}

void check_cuda(cudaError_t err, const char* what)
{
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("mock SIPL: ") + what + ": " + cudaGetErrorString(err));
}

} // namespace

struct SiplCamera::Impl
{
    struct DeviceBuffer
    {
        uint8_t* ptr = nullptr;
        size_t pitch = 0;
    };

    struct Sensor
    {
        SensorInfo info;
        /// Synthetic ISP0 surface, in place of the NvSciBuf SIPL would fill.
        DeviceBuffer luma;
        DeviceBuffer chroma;
        cudaStream_t stream = nullptr;

        /// Triple-buffered RGBA8 mailbox. A reader leases the published slot
        /// until its next latest() call, so the producer never overwrites it.
        std::array<DeviceBuffer, 3> buffers{};
        mutable std::mutex publish_mutex;
        int publish_idx = -1;
        int lease_idx = -1;
        uint64_t published_sequence = 0;
        uint64_t consumed_sequence = 0;
        uint64_t published_timestamp_ns = 0;
        uint64_t published_capture_tsc_ns = 0;

        std::thread frame_thread;

        // Owns its device memory, so a constructor that throws part-way through
        // allocating does not leak what it had already allocated.
        ~Sensor()
        {
            cudaFree(luma.ptr);
            cudaFree(chroma.ptr);
            for (auto& buffer : buffers)
                cudaFree(buffer.ptr);
            if (stream)
                cudaStreamDestroy(stream);
        }
    };

    explicit Impl(const SiplConfig& cfg) : config(cfg)
    {
        if (config.link_masks.empty())
            throw std::runtime_error("link masks are required for a GMSL platform config");

        sensor_infos =
            mock_query_platform_config(config.platform_config_json, config.platform_config_name, config.link_masks);
        if (sensor_infos.empty())
        {
            throw std::runtime_error("no modules left in '" + config.platform_config_name +
                                     "' after applying the link masks -- check the mask against the config");
        }

        fps_override = env_number("SENSING_MOCK_FPS", 0.0);
        fail_after_frames = static_cast<uint64_t>(env_number("SENSING_MOCK_FAIL_AFTER", 0.0));

        check_cuda(cudaSetDevice(config.gpu_id), "cudaSetDevice");
        for (const auto& info : sensor_infos)
        {
            auto sensor = std::make_unique<Sensor>();
            sensor->info = info;
            allocate(*sensor);
            sensors.push_back(std::move(sensor));
        }
    }

    ~Impl()
    {
        stop();
    }

    void allocate(Sensor& sensor)
    {
        const size_t width = sensor.info.width;
        const size_t height = sensor.info.height;
        check_cuda(cudaMallocPitch(reinterpret_cast<void**>(&sensor.luma.ptr), &sensor.luma.pitch, width, height),
                   "cudaMallocPitch(Y)");
        check_cuda(cudaMallocPitch(reinterpret_cast<void**>(&sensor.chroma.ptr), &sensor.chroma.pitch, width, height / 2),
                   "cudaMallocPitch(UV)");
        for (auto& buffer : sensor.buffers)
        {
            check_cuda(cudaMallocPitch(reinterpret_cast<void**>(&buffer.ptr), &buffer.pitch, width * 4, height),
                       "cudaMallocPitch(RGBA)");
        }
        check_cuda(cudaStreamCreateWithFlags(&sensor.stream, cudaStreamNonBlocking), "cudaStreamCreate");
    }

    Sensor* find(uint32_t id)
    {
        for (auto& sensor : sensors)
        {
            if (sensor->info.id == id)
                return sensor.get();
        }
        return nullptr;
    }

    void set_failure(const std::string& message)
    {
        std::lock_guard<std::mutex> guard(failure_mutex);
        if (failure.empty())
            failure = message;
    }

    void throw_if_failed()
    {
        std::lock_guard<std::mutex> guard(failure_mutex);
        if (!failure.empty())
            throw std::runtime_error(failure);
    }

    uint32_t pick_write_index(const Sensor& sensor) const
    {
        std::lock_guard<std::mutex> guard(sensor.publish_mutex);
        if (sensor.publish_idx < 0)
            return 0;
        // Skip the published slot and any slot still leased to a reader. With
        // three slots one is always free, so this cannot spin.
        uint32_t idx = static_cast<uint32_t>((sensor.publish_idx + 1) % 3);
        if (static_cast<int>(idx) == sensor.lease_idx)
            idx = (idx + 1) % 3;
        return idx;
    }

    void publish(Sensor& sensor, uint32_t write_idx, uint64_t timestamp_ns, uint64_t capture_tsc_ns)
    {
        std::lock_guard<std::mutex> guard(sensor.publish_mutex);
        sensor.publish_idx = static_cast<int>(write_idx);
        sensor.published_timestamp_ns = timestamp_ns;
        sensor.published_capture_tsc_ns = capture_tsc_ns;
        ++sensor.published_sequence;
    }

    void capture_one(Sensor& sensor, uint64_t frame_index, uint64_t tsc_ns)
    {
        const uint32_t write_idx = pick_write_index(sensor);
        const DeviceBuffer& dest = sensor.buffers[write_idx];
        const int width = static_cast<int>(sensor.info.width);
        const int height = static_cast<int>(sensor.info.height);

        launch_mock_nv12_pattern(sensor.luma.ptr, static_cast<int>(sensor.luma.pitch), sensor.chroma.ptr,
                                 static_cast<int>(sensor.chroma.pitch), width, height, sensor.info.id,
                                 static_cast<uint32_t>(frame_index), sensor.stream);

        // Same call, layout and colour matrix as the SIPL path: full-range
        // BT.709 semi-planar, from the surface the ISP would have written.
        const YuvLayout layout = config.swap_uv ? YuvLayout::YVU420SemiPlanar : YuvLayout::YUV420SemiPlanar;
        launch_yuv420_pitch_to_rgba(sensor.luma.ptr, sensor.chroma.ptr, nullptr, static_cast<int>(sensor.luma.pitch),
                                    static_cast<int>(sensor.chroma.pitch), 0, width, height, dest.ptr,
                                    static_cast<int>(dest.pitch), layout, true, true, sensor.stream);
        check_cuda(cudaGetLastError(), "mock frame kernels");
        check_cuda(cudaStreamSynchronize(sensor.stream), "cudaStreamSynchronize");

        publish(sensor, write_idx, monotonic_ns(), tsc_ns);
    }

    void frame_loop(Sensor& sensor, std::chrono::steady_clock::time_point epoch)
    {
        try
        {
            check_cuda(cudaSetDevice(config.gpu_id), "cudaSetDevice frame_loop");
            const double fps = fps_override > 0.0 ? fps_override : std::max(sensor.info.fps, 1.0);
            const double period_ns = 1e9 / fps;
            for (uint64_t n = 0; running.load(); ++n)
            {
                std::this_thread::sleep_until(epoch + std::chrono::nanoseconds(static_cast<int64_t>(n * period_ns)));
                if (!running.load())
                    break;
                if (fail_after_frames != 0 && n >= fail_after_frames)
                    throw std::runtime_error("injected link failure (SENSING_MOCK_FAIL_AFTER)");
                // Every sensor derives its TSC from the same epoch, as the
                // deserializer's shared fsync does: frame n of each sensor
                // carries the same capture time.
                capture_one(sensor, n, kMockTscBaseNs + static_cast<uint64_t>(n * period_ns));
            }
        }
        catch (const std::exception& e)
        {
            std::ostringstream oss;
            oss << "SIPL sensor " << sensor.info.id << " capture error: " << e.what();
            std::cerr << "[sipl-mock] " << oss.str() << std::endl;
            set_failure(oss.str());
            running.store(false);
        }
    }

    void start()
    {
        if (running.exchange(true))
            return;
        const auto epoch = std::chrono::steady_clock::now();
        for (auto& sensor : sensors)
            sensor->frame_thread = std::thread([this, s = sensor.get(), epoch] { frame_loop(*s, epoch); });
    }

    void stop()
    {
        running.store(false);
        for (auto& sensor : sensors)
        {
            if (sensor->frame_thread.joinable())
                sensor->frame_thread.join();
        }
    }

    SiplConfig config;
    std::vector<SensorInfo> sensor_infos;
    std::vector<std::unique_ptr<Sensor>> sensors;
    std::atomic<bool> running{ false };
    double fps_override = 0.0;
    uint64_t fail_after_frames = 0;
    std::mutex failure_mutex;
    std::string failure;
};

SiplCamera::SiplCamera(const SiplConfig& config) : m_impl(std::make_unique<Impl>(config))
{
}

SiplCamera::~SiplCamera() = default;

std::vector<SensorInfo> SiplCamera::query(const std::string& platform_config_json,
                                          const std::string& platform_config_name,
                                          const std::vector<uint32_t>& link_masks)
{
    return mock_query_platform_config(platform_config_json, platform_config_name, link_masks);
}

const std::vector<SensorInfo>& SiplCamera::sensors() const
{
    return m_impl->sensor_infos;
}

void SiplCamera::start()
{
    m_impl->start();
}

void SiplCamera::stop()
{
    m_impl->stop();
}

std::optional<FrameView> SiplCamera::latest(uint32_t sensor_id)
{
    m_impl->throw_if_failed();
    Impl::Sensor* sensor = m_impl->find(sensor_id);
    if (!sensor)
        throw std::runtime_error("no SIPL pipeline with index " + std::to_string(sensor_id));

    std::lock_guard<std::mutex> guard(sensor->publish_mutex);
    if (sensor->publish_idx < 0 || sensor->consumed_sequence == sensor->published_sequence)
        return std::nullopt;
    sensor->consumed_sequence = sensor->published_sequence;

    // The caller now holds this slot; pick_write_index() skips it until the
    // next latest() moves the lease on.
    sensor->lease_idx = sensor->publish_idx;

    FrameView view;
    view.ptr = reinterpret_cast<uintptr_t>(sensor->buffers[sensor->publish_idx].ptr);
    view.pitch = sensor->buffers[sensor->publish_idx].pitch;
    view.width = sensor->info.width;
    view.height = sensor->info.height;
    view.timestamp_ns = sensor->published_timestamp_ns;
    view.capture_tsc_ns = sensor->published_capture_tsc_ns;
    view.sequence = sensor->published_sequence;
    return view;
}

} // namespace sensing
} // namespace plugins
