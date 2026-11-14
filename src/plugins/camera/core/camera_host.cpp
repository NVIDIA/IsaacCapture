// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/camera_plugin/camera_host.hpp"

#include <camera_endpoint/endpoint.hpp>
#include <camera_ipc/host_pool.hpp>
#include <log_bridge/logger.hpp>

#include <algorithm>
#include <chrono>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

namespace plugins::camera
{

using core::camera::Endpoint;
using core::camera::FrameTimeline;
using core::camera::GroupDescription;
using core::camera::GroupId;
using core::camera::HostPool;
using core::camera::ImageInfo;
using core::camera::Publisher;
using core::camera::StreamDescription;

namespace
{

// One camera's way into the publisher: slots come from the camera's own pools.
class PoolOutput final : public FrameOutput
{
public:
    PoolOutput(Publisher& publisher, GroupId group, std::vector<std::unique_ptr<HostPool>>& pools)
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
        return WritableImage{ *slot, m_pools.at(stream)->slot(*slot) };
    }

    std::uint32_t slot_count(std::size_t stream) const override
    {
        return m_pools.at(stream)->slot_count();
    }

    std::span<std::byte> slot_memory(std::size_t stream, std::uint32_t slot) override
    {
        return m_pools.at(stream)->slot(slot);
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
    std::vector<std::unique_ptr<HostPool>>& m_pools;
};

struct Camera
{
    std::unique_ptr<CameraSource> source;
    std::string id;
    GroupId group = 0;
    std::vector<std::unique_ptr<HostPool>> pools; // one per stream
    std::unique_ptr<PoolOutput> output;
    core::camera::PublisherStats logged; // the counters at the previous log_activity()
};

} // namespace

const Backend& find_backend(std::span<const Backend> backends, const std::string& name)
{
    std::string built;
    for (const Backend& backend : backends)
    {
        if (backend.name == name)
        {
            return backend;
        }
        built += (built.empty() ? "" : ", ") + backend.name;
    }
    throw std::invalid_argument("this camera plugin was built without a '" + name + "' backend (it has: " + built +
                                "). Backends are chosen when the plugin is built; see the plugin's README");
}

std::vector<std::unique_ptr<CameraSource>> open_cameras(std::span<const Backend> backends,
                                                        std::span<const CameraConfig> cameras)
{
    std::vector<std::unique_ptr<CameraSource>> sources;
    std::vector<std::string> opened;
    for (const CameraConfig& first : cameras)
    {
        if (std::find(opened.begin(), opened.end(), first.backend) != opened.end())
        {
            continue;
        }
        opened.push_back(first.backend);
        std::vector<CameraConfig> of_backend;
        std::copy_if(cameras.begin(), cameras.end(), std::back_inserter(of_backend),
                     [&](const CameraConfig& camera) { return camera.backend == first.backend; });
        std::vector<std::unique_ptr<CameraSource>> of_sources = find_backend(backends, first.backend).open(of_backend);
        if (of_sources.size() != of_backend.size())
        {
            throw std::logic_error("the '" + first.backend + "' backend was given " + std::to_string(of_backend.size()) +
                                   " camera(s) and opened " + std::to_string(of_sources.size()));
        }
        std::move(of_sources.begin(), of_sources.end(), std::back_inserter(sources));
    }
    return sources;
}

class CameraHost::Impl
{
public:
    explicit Impl(const std::string& endpoint_path)
        : endpoint(publisher, endpoint_path), logger(isaaccapture::Logger::get("isaaccapture.plugins.camera.CameraHost"))
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

    auto camera = std::make_unique<Camera>();
    camera->id = description.id;
    // Refuses a description that is not usable before any memory is allocated for it.
    const GroupId group = m_impl->publisher.add_group(description, source->backend_reserve());
    camera->group = group;
    std::vector<int> descriptors;
    for (const StreamDescription& stream : description.streams)
    {
        camera->pools.push_back(std::make_unique<HostPool>(stream.slot_count, stream.slot_bytes, "camera." + stream.id));
        descriptors.push_back(camera->pools.back()->descriptor());
    }
    m_impl->endpoint.set_group_memory(group, std::move(descriptors));
    camera->output = std::make_unique<PoolOutput>(m_impl->publisher, group, camera->pools);
    camera->source = std::move(source);

    const StreamDescription& first = description.streams.front();
    m_impl->logger->info("camera '{}': {} stream(s), {}x{} at {:.2f} fps, {} slots of {} bytes each", description.id,
                         description.streams.size(), first.width, first.height, first.frame_rate, first.slot_count,
                         first.slot_bytes);
    m_impl->cameras.push_back(std::move(camera));
}

void CameraHost::start()
{
    if (m_impl->started)
    {
        return;
    }
    m_impl->started = true;
    m_impl->endpoint.start();
    for (const auto& camera : m_impl->cameras)
    {
        camera->source->start(*camera->output);
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
        const core::camera::PublisherStats stats = m_impl->publisher.stats(camera->group);
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

} // namespace plugins::camera
