// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <camera_core/types.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace plugins::camera
{

// One camera as the configuration names it.
struct CameraConfig
{
    std::string id; // the group readers ask for
    std::string backend; // the backend that opens it
    std::map<std::string, std::string> options; // the backend's own settings
};

// A free slot of one stream, for the next image.
struct WritableImage
{
    std::uint32_t slot = 0;
    std::span<std::byte> data;
};

/**
 * @brief Where a camera's frames go. Safe to call from the camera's capture thread while
 * readers come and go.
 */
class FrameOutput
{
public:
    virtual ~FrameOutput() = default;

    /**
     * @brief A free slot of one stream to write the next image into.
     * @return Nothing when every slot is in use: drop the frame, and abandon() what was already
     * acquired for it.
     */
    virtual std::optional<WritableImage> acquire(std::size_t stream) = 0;

    // Gives back an acquired slot that will not be published.
    virtual void abandon(std::size_t stream, std::uint32_t slot) = 0;

    /**
     * @brief Publish one frame set: images[i] is stream i's, in a slot from acquire().
     * @param timeline capture_ns and sdk_ready_ns, on the host's monotonic clock.
     * @return The frame set's sequence number.
     */
    virtual std::uint64_t publish(std::span<const core::camera::ImageInfo> images,
                                  const core::camera::FrameTimeline& timeline) = 0;
};

/**
 * @brief One configured camera: one or more streams captured together. A backend implements
 * this for its SDK.
 */
class CameraSource
{
public:
    virtual ~CameraSource() = default;

    /**
     * @brief What the camera publishes: the group, its streams and the size of a slot. A
     * stream's slot_count may be left 0 to take the plugin's pool size.
     */
    virtual core::camera::GroupDescription describe() const = 0;

    // Start capturing into `output`, on a thread of the source's own.
    virtual void start(FrameOutput& output) = 0;

    // Stop capturing. The output is not called once this has returned.
    virtual void stop() = 0;
};

// One camera SDK as the plugin sees it.
struct Backend
{
    std::string name; // what a camera's configuration names
    std::string description;

    // Opens one camera. Throws std::invalid_argument for options the backend does not accept.
    std::function<std::unique_ptr<CameraSource>(const CameraConfig&)> open;

    // The cameras of this backend that are plugged in, each ready to be opened. Left empty by
    // a backend that has nothing to look for.
    std::function<std::vector<CameraConfig>()> discover = {};
};

} // namespace plugins::camera
