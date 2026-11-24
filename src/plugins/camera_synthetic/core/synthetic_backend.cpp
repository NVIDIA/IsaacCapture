// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/camera_synthetic/synthetic_backend.hpp"

#include <camera/cuda_pool.hpp>
#include <camera/options.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <span>
#include <stdexcept>
#include <thread>
#include <vector>

namespace plugins::camera_synthetic
{

using core::camera::Backend;
using core::camera::CameraConfig;
using core::camera::CameraSource;
using core::camera::FrameOutput;
using core::camera::FrameTimeline;
using core::camera::GroupDescription;
using core::camera::ImageInfo;
using core::camera::MemoryKind;
using core::camera::OptionReader;
using core::camera::PixelFormat;
using core::camera::PlaneSize;
using core::camera::StreamDescription;
using core::camera::WritableImage;

namespace
{

std::int64_t now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

class SyntheticSource final : public CameraSource
{
public:
    explicit SyntheticSource(const CameraConfig& config)
    {
        OptionReader options(config);
        const std::uint32_t width = options.number("width", 1280, 16, 8192);
        const std::uint32_t height = options.number("height", 720, 16, 8192);
        const std::uint32_t fps = options.number("fps", 60, 1, 1000);
        const std::uint32_t streams = options.number("streams", 2, 1, core::camera::MAX_STREAMS_PER_GROUP);
        const std::string format_name = options.text("format", "bgra8");
        const std::string memory_name = options.text("memory", "host");
        const std::uint32_t gpu = options.number("gpu", 0, 0, 255);
        options.finish();

        PixelFormat format = PixelFormat::BGRA8;
        if (format_name == "rgba8")
        {
            format = PixelFormat::RGBA8;
        }
        else if (format_name == "gray8")
        {
            format = PixelFormat::GRAY8;
        }
        else if (format_name == "nv12")
        {
            format = PixelFormat::NV12;
        }
        else if (format_name != "bgra8")
        {
            options.fail("option 'format' is '" + format_name + "'; it takes bgra8, rgba8, gray8 or nv12");
        }
        if (memory_name == "gpu")
        {
            try
            {
                m_cuda_device = core::camera::cuda_device_uuid(gpu);
            }
            catch (const std::runtime_error& error)
            {
                options.fail(error.what());
            }
        }
        else if (memory_name != "host")
        {
            options.fail("option 'memory' is '" + memory_name + "'; it takes host or gpu");
        }
        else if (config.options.contains("gpu"))
        {
            options.fail("option 'gpu' names the device of memory=gpu; this camera's frames are in host memory");
        }

        m_period = std::chrono::nanoseconds(1'000'000'000 / fps);
        m_group.id = config.id;
        for (std::uint32_t i = 0; i < streams; ++i)
        {
            StreamDescription stream;
            // A pair reads better as left and right than as 0 and 1.
            stream.id = config.id + "/" + (streams == 2 ? (i == 0 ? "left" : "right") : std::to_string(i));
            stream.pixel_format = format;
            stream.width = width;
            stream.height = height;
            stream.plane_count = core::camera::plane_count_of(format);
            std::uint64_t offset = 0;
            for (std::uint32_t plane = 0; plane < stream.plane_count; ++plane)
            {
                const PlaneSize size = core::camera::plane_size_of(format, width, height, plane);
                stream.planes[plane] = { static_cast<std::uint32_t>(offset), static_cast<std::uint32_t>(size.row_bytes) };
                offset += size.row_bytes * size.rows;
            }
            stream.slot_bytes = offset;
            stream.frame_rate = fps;
            if (memory_name == "gpu")
            {
                stream.memory = MemoryKind::CUDA;
                stream.cuda_device = m_cuda_device;
            }
            m_group.streams.push_back(stream);
        }
        if (memory_name == "gpu")
        {
            m_staging.resize(m_group.streams.front().slot_bytes);
        }
    }

    ~SyntheticSource() override
    {
        stop();
    }

    GroupDescription describe() const override
    {
        return m_group;
    }

    void start(FrameOutput& output) override
    {
        if (m_thread.joinable())
        {
            return;
        }
        m_stopping = false;
        m_thread = std::thread([this, &output] { run(output); });
    }

    void stop() override
    {
        if (!m_thread.joinable())
        {
            return;
        }
        m_stopping = true;
        m_thread.join();
    }

private:
    void run(FrameOutput& output)
    {
        std::vector<ImageInfo> images(m_group.streams.size());
        auto tick = std::chrono::steady_clock::now();
        for (std::uint64_t frame = 0;; ++frame)
        {
            if (!sleep_until(tick))
            {
                return;
            }
            FrameTimeline timeline;
            timeline.capture_ns = now_ns();
            if (write(output, frame, images))
            {
                timeline.sdk_ready_ns = now_ns();
                output.publish(images, timeline);
            }
            tick += m_period;
            // After a stall, carry on from now instead of bursting to catch up.
            tick = std::max(tick, std::chrono::steady_clock::now() - m_period);
        }
    }

    // Sleeps until `tick` in steps short enough for stop() to be prompt at any frame rate.
    // False when the camera was stopped meanwhile. A flag and short sleeps, not a condition
    // variable: GCC 11's ThreadSanitizer misreads a wait on the steady clock as a double lock.
    bool sleep_until(std::chrono::steady_clock::time_point tick) const
    {
        constexpr std::chrono::milliseconds LONGEST_STEP(50);
        while (!m_stopping)
        {
            const auto now = std::chrono::steady_clock::now();
            if (now >= tick)
            {
                return true;
            }
            std::this_thread::sleep_until(std::min(tick, now + LONGEST_STEP));
        }
        return false;
    }

    // Writes one frame set. False when a stream has no free slot: the frame set is dropped.
    bool write(FrameOutput& output, std::uint64_t frame, std::vector<ImageInfo>& images)
    {
        for (std::size_t stream = 0; stream < images.size(); ++stream)
        {
            const std::optional<WritableImage> image = output.acquire(stream);
            if (!image)
            {
                for (std::size_t taken = 0; taken < stream; ++taken)
                {
                    output.abandon(taken, images[taken].slot);
                }
                return false;
            }
            // A camera in CUDA memory makes the image on the host and copies it to the device;
            // the copy has finished when the call returns, so the image is whole when published.
            const std::span<std::byte> pixels = m_staging.empty() ? image->data : std::span<std::byte>(m_staging);
            std::memset(pixels.data(), std::to_integer<int>(synthetic_fill(frame, stream)), pixels.size());
            std::memcpy(pixels.data(), &frame, sizeof(frame));
            if (!m_staging.empty())
            {
                core::camera::copy_to_cuda(m_cuda_device, image->cuda, pixels);
            }
            images[stream] = {};
            images[stream].slot = image->slot;
            images[stream].device_frame_id = frame;
        }
        return true;
    }

    GroupDescription m_group;
    core::camera::CudaDeviceUuid m_cuda_device = {};
    std::vector<std::byte> m_staging; // one image on the host, for a camera in CUDA memory
    std::chrono::nanoseconds m_period{};
    std::thread m_thread;
    std::atomic<bool> m_stopping{ false };
};

} // namespace

Backend synthetic_backend()
{
    return { "synthetic", "frames with known content at a set rate; needs no hardware",
             [](std::span<const CameraConfig> cameras)
             {
                 // Synthetic cameras have nothing in common, so there is nothing to open once.
                 std::vector<std::unique_ptr<CameraSource>> sources;
                 for (const CameraConfig& camera : cameras)
                 {
                     sources.push_back(std::make_unique<SyntheticSource>(camera));
                 }
                 return sources;
             } };
}

std::uint64_t synthetic_frame_number(std::span<const std::byte> image)
{
    std::uint64_t frame = 0;
    std::memcpy(&frame, image.data(), std::min(image.size(), sizeof(frame)));
    return frame;
}

} // namespace plugins::camera_synthetic
