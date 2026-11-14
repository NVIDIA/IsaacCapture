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
 * @brief Where a camera's frames go: the pools of its streams and the way to publish into
 * them. Safe to call from the camera's capture thread while readers come and go.
 *
 * A camera gets the slot for an image in one of two ways, and uses one of them throughout:
 * - acquire(): the plugin hands out a free slot and the camera writes the image into it.
 * - claim(): an SDK that is given its buffers up front says which one it filled. The camera
 *   hands the slots' memory to the SDK before capturing (slot_count(), slot_memory()), asks
 *   to be told when a slot is free again (on_slot_free()), and names slots with claim().
 */
class FrameOutput
{
public:
    using SlotFreeCallback = std::function<void(std::size_t stream, std::uint32_t slot)>;

    virtual ~FrameOutput() = default;

    /**
     * @brief A free slot of one stream to write the next image into.
     * @return Nothing when every slot is in use: drop the frame, and abandon() what was already
     * acquired for it.
     * @throws std::logic_error after on_slot_free(): that camera names its slots with claim().
     */
    virtual std::optional<WritableImage> acquire(std::size_t stream) = 0;

    // The slots of one stream's pool, and the memory of one of them. The memory stays where it
    // is for as long as the output exists, so an SDK can be given it once.
    virtual std::uint32_t slot_count(std::size_t stream) const = 0;
    virtual std::span<std::byte> slot_memory(std::size_t stream, std::uint32_t slot) = 0;

    /**
     * @brief Say that the SDK filled this slot.
     * @throws std::invalid_argument if a reader or the newest frame set still holds the slot:
     * the SDK was handed a buffer that was not free.
     */
    virtual void claim(std::size_t stream, std::uint32_t slot) = 0;

    /**
     * @brief Be told when a published slot is held by no reader and is no longer the newest
     * frame set's, so that its buffer can go back to the SDK. Called with no lock of the
     * plugin held, on the thread that let go of the slot last: the plugin's own when a reader
     * gave a frame set back or went away, the camera's from inside publish() when a newer
     * frame set took its place. Set it before the first claim().
     */
    virtual void on_slot_free(SlotFreeCallback callback) = 0;

    // Gives back a slot from acquire() or claim() that will not be published.
    virtual void abandon(std::size_t stream, std::uint32_t slot) = 0;

    /**
     * @brief Publish one frame set: images[i] is stream i's, in a slot from acquire() or
     * claim(). Publishing says the images are complete, and readers may look at them from
     * this moment on: call it once everything that writes them has finished.
     * @param timeline capture_ns and sdk_ready_ns, on the host's monotonic clock.
     * @return The frame set's sequence number.
     */
    virtual std::uint64_t publish(std::span<const core::camera::ImageInfo> images,
                                  const core::camera::FrameTimeline& timeline) = 0;

    /**
     * @brief Say that the camera restarted. Readers see a new generation, and nothing from
     * before the restart is handed out any more. Frame sets readers still hold stay theirs,
     * and their slots out of use, until they are given back.
     */
    virtual void restart() = 0;
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

    /**
     * @brief Slots the camera needs for itself, on top of the readers' and the newest frame
     * set's: one to write the next image into, more for an SDK that keeps several captures in
     * flight. Readers are never given so many slots that the camera is left with fewer.
     */
    virtual std::uint32_t backend_reserve() const
    {
        return 1;
    }

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

    // Opens the cameras the configuration names for this backend, all in one call, and returns
    // one source per camera in the same order. What the SDK has open belongs to the backend,
    // not to a camera: an SDK that serves several cameras from one object, or keeps state for
    // the whole process, opens it once here and lets the sources it returns share it. Throws
    // std::invalid_argument for options the backend does not accept.
    std::function<std::vector<std::unique_ptr<CameraSource>>(std::span<const CameraConfig>)> open;

    // The cameras of this backend that are plugged in, each ready to be opened. Left empty by
    // a backend that has nothing to look for.
    std::function<std::vector<CameraConfig>()> discover = {};
};

} // namespace plugins::camera
