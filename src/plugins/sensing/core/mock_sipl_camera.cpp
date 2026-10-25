// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Mock backend for SiplCamera: frames from an animated GIF, no SIPL, NvSci or EGL.
// It keeps the contract the rest of the plugin relies on -- per-sensor capture
// threads, a three-slot RGBA mailbox with a reader lease, a TSC stamp shared by
// every sensor, and failures surfacing from latest() -- and runs the real
// YUV-to-RGBA kernel on a synthetic NV12 surface made from the GIF, so the
// conversion code is exercised too.
//
// What a sensor shows is chosen by module name, never by pipeline index (which the link mask renumbers):
//   S56C_HEAD     the robot, the two eyes offset against each other by a fixed disparity
//   SHF3L_LEFT    the waving hand
//   SHF3L_RIGHT   the waving hand, flipped left to right
//   SHW5G_MIX_L2 / SHW5G_MIX_L3   as SHF3L_LEFT / SHF3L_RIGHT
// A module not listed needs SENSING_MOCK_GIF or SENSING_MOCK_GIF_<module name>. Environment, in order:
//   SENSING_MOCK_GIF_<module name>=PATH     one module's GIF
//   SENSING_MOCK_GIF=PATH                   every sensor's GIF
//   SENSING_MOCK_FLIP_<module name>=1|0     flip one module, whatever its GIF
// A GIF named by the environment is shown as it is, with no eye offset. A GIF that cannot be loaded,
// the bundled one included, is an error. It is scaled to cover the sensor's configured size and
// centre-cropped, never stretched, and follows the capture clock: it advances with each frame's time,
// and the sequence number and stamps advance on every capture even while the GIF shows one image.
//
// Other environment (read once, at construction):
//   SENSING_MOCK_FPS=N          override the platform config's frame rate
//   SENSING_MOCK_FAIL_AFTER=N   every sensor fails after N frames, as a link drop would
//   SENSING_MOCK_SCALE_DIV=N    shrink the reported geometry (see mock_platform_config.hpp)

#include "mock_gif_source.hpp"
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
#include <map>
#include <memory>
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

// Total horizontal offset between the two eyes, as a fraction of the width: a 60 mm baseline seen at
// about half a metre through a 140 degree lens.
constexpr double kEyeDisparity = 0.02;

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

        std::shared_ptr<MockGifSource> gif;
        bool mirror = false;
        int shift_x = 0;

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
            choose_picture(info, *sensor);
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

    /// Sets the GIF a sensor shows, whether it is flipped and how far its picture is offset.
    void choose_picture(const SensorInfo& info, Sensor& sensor)
    {
        struct Look
        {
            const char* module;
            const char* gif;
            bool mirror;
            bool stereo;
        };
        static constexpr Look kLooks[] = {
            { "S56C_HEAD", "noto-robot.gif", false, true },
            { "SHF3L_LEFT", "noto-waving-hand.gif", false, false },
            { "SHF3L_RIGHT", "noto-waving-hand.gif", true, false },
            { "SHW5G_MIX_L2", "noto-waving-hand.gif", false, false },
            { "SHW5G_MIX_L3", "noto-waving-hand.gif", true, false },
        };
        const Look* look = nullptr;
        for (const Look& candidate : kLooks)
            look = info.name == candidate.module ? &candidate : look;

        const char* path = std::getenv(("SENSING_MOCK_GIF_" + info.name).c_str());
        if (!path)
            path = std::getenv("SENSING_MOCK_GIF");
        std::string bundled;
        if (!path)
        {
            if (!look)
            {
                throw std::runtime_error("no mock picture for module '" + info.name + "': set SENSING_MOCK_GIF_" +
                                         info.name + " or SENSING_MOCK_GIF");
            }
            bundled = std::string(SENSING_MOCK_ASSET_DIR) + "/" + look->gif;
            path = bundled.c_str();
            sensor.mirror = look->mirror;
            if (look->stereo)
            {
                // The sensors of one module are its left and right eyes, in that order. The left eye sees
                // the scene further right, so its picture moves right and the right eye's moves left.
                size_t eye = 0;
                for (const SensorInfo& other : sensor_infos)
                    eye += other.name == info.name && other.id < info.id ? 1 : 0;
                const int half = static_cast<int>(info.width * kEyeDisparity / 2);
                sensor.shift_x = eye == 0 ? half : -half;
            }
        }

        // Sensors showing the same file share one copy on the device.
        auto& slot = gif_cache[path];
        if (!slot)
            slot = std::make_shared<MockGifSource>(path);
        sensor.gif = slot;

        const std::string flip_name = "SENSING_MOCK_FLIP_" + info.name;
        if (const char* flip = std::getenv(flip_name.c_str()))
        {
            const std::string value = flip;
            if (value != "0" && value != "1")
                throw std::runtime_error(flip_name + " must be 0 or 1, got '" + value + "'");
            sensor.mirror = value == "1";
        }
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

    void capture_one(Sensor& sensor, double seconds, uint64_t tsc_ns)
    {
        const uint32_t write_idx = pick_write_index(sensor);
        const DeviceBuffer& dest = sensor.buffers[write_idx];
        const int width = static_cast<int>(sensor.info.width);
        const int height = static_cast<int>(sensor.info.height);

        launch_mock_gif_nv12(
            sensor.gif->frame(sensor.gif->frame_index_at(seconds)), static_cast<int>(sensor.gif->pitch()),
            static_cast<int>(sensor.gif->width()), static_cast<int>(sensor.gif->height()), sensor.luma.ptr,
            static_cast<int>(sensor.luma.pitch), sensor.chroma.ptr, static_cast<int>(sensor.chroma.pitch), width,
            height, sensor.mirror, sensor.shift_x, sensor.stream);

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
                capture_one(sensor, n * period_ns / 1e9, kMockTscBaseNs + static_cast<uint64_t>(n * period_ns));
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
    std::map<std::string, std::shared_ptr<MockGifSource>> gif_cache;
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
