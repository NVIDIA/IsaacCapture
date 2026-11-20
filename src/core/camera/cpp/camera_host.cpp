// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/camera/camera_host.hpp"

#include "inc/camera/cuda_pool.hpp"
#include "inc/camera/endpoint.hpp"
#include "inc/camera/host_pool.hpp"

#include <log_bridge/logger.hpp>

#include <chrono>
#include <stdexcept>
#include <utility>
#include <vector>

namespace core::camera
{

namespace
{

// One stream's slots, in the kind of memory the stream is described with.
struct StreamPool
{
    std::unique_ptr<HostPool> host;
    std::unique_ptr<CudaPool> cuda;

    std::uint32_t slot_count() const
    {
        return host ? host->slot_count() : cuda->slot_count();
    }

    int descriptor() const
    {
        return host ? host->descriptor() : cuda->descriptor();
    }
};

// One camera's way into the publisher: slots come from the camera's own pools.
class PoolOutput final : public FrameOutput
{
public:
    PoolOutput(Publisher& publisher, GroupId group, std::vector<StreamPool>& pools)
        : m_publisher(publisher), m_group(group), m_pools(pools)
    {
    }

    std::optional<WritableImage> acquire(std::size_t stream) override
    {
        const std::optional<std::uint32_t> slot = m_publisher.acquire_slot(m_group, stream);
        if (!slot)
        {
            return std::nullopt;
        }
        const StreamPool& pool = m_pools.at(stream);
        return pool.host ? WritableImage{ *slot, pool.host->slot(*slot), 0 } :
                           WritableImage{ *slot, {}, pool.cuda->slot(*slot) };
    }

    std::uint32_t slot_count(std::size_t stream) const override
    {
        return m_pools.at(stream).slot_count();
    }

    std::span<std::byte> slot_memory(std::size_t stream, std::uint32_t slot) override
    {
        const StreamPool& pool = m_pools.at(stream);
        if (!pool.host)
        {
            throw std::logic_error("the stream's slots are in CUDA memory: ask for slot_cuda_memory()");
        }
        return pool.host->slot(slot);
    }

    CudaAddress slot_cuda_memory(std::size_t stream, std::uint32_t slot) override
    {
        const StreamPool& pool = m_pools.at(stream);
        if (!pool.cuda)
        {
            throw std::logic_error("the stream's slots are in host memory: ask for slot_memory()");
        }
        return pool.cuda->slot(slot);
    }

    void claim(std::size_t stream, std::uint32_t slot) override
    {
        m_publisher.claim_slot(m_group, stream, slot);
    }

    void on_slot_free(SlotFreeCallback callback) override
    {
        m_publisher.set_slot_free_callback(m_group, std::move(callback));
    }

    void abandon(std::size_t stream, std::uint32_t slot) override
    {
        m_publisher.abandon_slot(m_group, stream, slot);
    }

    std::uint64_t publish(std::span<const ImageInfo> images, const FrameTimeline& timeline) override
    {
        return m_publisher.publish(m_group, images, timeline);
    }

    void restart() override
    {
        m_publisher.restart_group(m_group);
    }

private:
    Publisher& m_publisher;
    GroupId m_group;
    std::vector<StreamPool>& m_pools;
};

struct Camera
{
    std::unique_ptr<CameraSource> source;
    std::string id;
    GroupId group = 0;
    std::vector<StreamPool> pools; // one per stream
    std::unique_ptr<PoolOutput> output;
    PublisherStats logged; // the counters at the previous log_activity()
};

} // namespace

class CameraHost::Impl
{
public:
    explicit Impl(const std::string& endpoint_path)
        : endpoint(publisher, endpoint_path), logger(isaaccapture::Logger::get("isaaccapture.camera.CameraHost"))
    {
    }

    Publisher publisher;
    Endpoint endpoint;
    // Stable addresses: an output points at its camera's pools.
    std::vector<std::unique_ptr<Camera>> cameras;
    bool started = false;
    std::chrono::steady_clock::time_point activity_logged = std::chrono::steady_clock::now();
    std::shared_ptr<spdlog::logger> logger;
};

CameraHost::CameraHost(const std::string& endpoint_path) : m_impl(std::make_unique<Impl>(endpoint_path))
{
}

CameraHost::~CameraHost()
{
    stop();
}

void CameraHost::add_camera(std::unique_ptr<CameraSource> source, std::uint32_t slot_count)
{
    if (m_impl->started)
    {
        throw std::logic_error("cameras are added before the camera host starts");
    }
    GroupDescription description = source->describe();
    for (StreamDescription& stream : description.streams)
    {
        if (stream.slot_count == 0)
        {
            stream.slot_count = slot_count;
        }
    }
    // A description that is not usable is refused before any memory is allocated for it, and
    // a camera whose memory cannot be had is not published.
    if (const std::string problem = problem_with(description); !problem.empty())
    {
        throw std::invalid_argument(problem);
    }

    auto camera = std::make_unique<Camera>();
    camera->id = description.id;
    std::vector<int> descriptors;
    for (const StreamDescription& stream : description.streams)
    {
        StreamPool pool;
        if (stream.memory == MemoryKind::CUDA)
        {
            pool.cuda = std::make_unique<CudaPool>(stream.cuda_device, stream.slot_count, stream.slot_bytes);
        }
        else
        {
            pool.host = std::make_unique<HostPool>(stream.slot_count, stream.slot_bytes, "camera." + stream.id);
        }
        descriptors.push_back(pool.descriptor());
        camera->pools.push_back(std::move(pool));
    }
    const GroupId group = m_impl->publisher.add_group(description, source->backend_reserve());
    camera->group = group;
    m_impl->endpoint.set_group_memory(group, std::move(descriptors));
    camera->output = std::make_unique<PoolOutput>(m_impl->publisher, group, camera->pools);
    camera->source = std::move(source);

    const StreamDescription& first = description.streams.front();
    m_impl->logger->info("camera '{}': {} stream(s), {}x{} at {:.2f} fps, {} slots of {} bytes each in {}",
                         description.id, description.streams.size(), first.width, first.height, first.frame_rate,
                         first.slot_count, first.slot_bytes,
                         first.memory == MemoryKind::CUDA ? "CUDA memory on " + cuda_device_name(first.cuda_device) :
                                                            std::string("host memory"));
    m_impl->cameras.push_back(std::move(camera));
}

void CameraHost::start()
{
    if (m_impl->started)
    {
        return;
    }
    m_impl->started = true;
    try
    {
        m_impl->endpoint.start();
        for (const auto& camera : m_impl->cameras)
        {
            camera->source->start(*camera->output);
        }
    }
    catch (...)
    {
        // Nothing is left half started: the cameras that did start are stopped and readers
        // are not served, so start() can be called again.
        stop();
        throw;
    }
}

void CameraHost::stop()
{
    if (!m_impl->started)
    {
        return;
    }
    // Cameras first: nothing publishes into a pool or the endpoint once they have stopped.
    for (const auto& camera : m_impl->cameras)
    {
        camera->source->stop();
    }
    m_impl->endpoint.stop();
    m_impl->started = false;
}

std::size_t CameraHost::camera_count() const
{
    return m_impl->cameras.size();
}

std::size_t CameraHost::reader_count() const
{
    return m_impl->endpoint.reader_count();
}

const Publisher& CameraHost::publisher() const
{
    return m_impl->publisher;
}

void CameraHost::log_activity()
{
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - m_impl->activity_logged).count();
    m_impl->activity_logged = now;
    for (const auto& camera : m_impl->cameras)
    {
        const PublisherStats stats = m_impl->publisher.stats(camera->group);
        const std::uint64_t published = stats.published - camera->logged.published;
        const std::uint64_t dropped = stats.no_free_slot - camera->logged.no_free_slot;
        const std::uint64_t unsent = stats.send_failures - camera->logged.send_failures;
        camera->logged = stats;
        m_impl->logger->info("camera '{}': {} frame sets in {:.1f} s ({:.1f} per second), {} reader(s) connected",
                             camera->id, published, seconds, seconds > 0.0 ? published / seconds : 0.0,
                             m_impl->endpoint.reader_count());
        if (dropped != 0 || unsent != 0)
        {
            m_impl->logger->warn(
                "camera '{}': dropped {} frame set(s) for lack of a free slot; {} event(s) "
                "were not taken by a reader",
                camera->id, dropped, unsent);
        }
    }
}

} // namespace core::camera
