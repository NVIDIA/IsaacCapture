// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <camera_core/types.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace core::camera
{

namespace detail
{
class ReaderState;
}

/**
 * @brief A reader could not be opened or cannot go on: nothing listens on the endpoint, the
 * plugin refused the reader, or it sent something a reader cannot use.
 */
class CameraReaderError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

/**
 * @brief The camera plugin closed the connection or is gone.
 */
class CameraDisconnected : public CameraReaderError
{
public:
    using CameraReaderError::CameraReaderError;
};

/**
 * @brief next() was called by a reader that already holds as many frame sets as its window.
 * Give one back first, or open the reader with a larger window.
 */
class CameraWindowExhausted : public std::logic_error
{
public:
    using std::logic_error::logic_error;
};

// One group a plugin publishes, as list() reports it.
struct GroupListing
{
    GroupDescription group; // the id only, when `unsupported` is set

    // Why this build cannot read the group, for one that uses a codec, a pixel format or a
    // kind of memory added after it; empty for a group it can open.
    std::string unsupported;
};

// Which camera plugin to read from. Left empty, it is this user's plugin on this machine.
struct PluginLocation
{
    std::string endpoint; // a socket path; empty for the default endpoint of `instance`
    std::string instance; // a named plugin instance; empty for the default one
};

/**
 * @brief One image of a frame set. The memory is the plugin's pool, mapped read-only, and
 * stays as it is for as long as the FrameSet the view came from is held.
 */
struct ImageView
{
    const StreamDescription* stream = nullptr; // what the bytes are; owned by the reader
    ImageInfo info;
    std::span<const std::byte> data; // pixels: the whole slot; encoded video: the access unit

    /**
     * @brief The bytes of one plane of a pixel image, from its first row to the end of its last.
     * Rows are stream->planes[index].stride_bytes apart.
     * @throws std::out_of_range for a plane the image does not have.
     */
    std::span<const std::byte> plane(std::uint32_t index) const;
};

/**
 * @brief One frame set, held for this reader: the plugin does not reuse its slots until the
 * object is destroyed or release() is called, also when the reader it came from is gone
 * first. Hold at most as many as the reader's window.
 */
class FrameSet
{
public:
    FrameSet(FrameSet&& other) noexcept;
    FrameSet& operator=(FrameSet&& other) noexcept;
    FrameSet(const FrameSet&) = delete;
    FrameSet& operator=(const FrameSet&) = delete;
    ~FrameSet();

    std::uint64_t sequence() const
    {
        return m_event.sequence;
    }

    // Restarts of the camera since the plugin began publishing the group.
    std::uint32_t generation() const
    {
        return m_event.generation;
    }

    // Frame sets published since the previous one this reader was given, and not given to it.
    std::uint32_t skipped() const
    {
        return m_event.skipped;
    }

    const FrameTimeline& timeline() const
    {
        return m_event.timeline;
    }

    std::size_t image_count() const
    {
        return m_event.image_count;
    }

    /**
     * @brief The image of one stream, in the order of the group's streams.
     * @throws std::out_of_range for a stream the group does not have.
     * @throws std::logic_error after release().
     */
    ImageView image(std::size_t stream) const;

    // Gives the frame set back before the object is destroyed. Its views are no longer valid:
    // give it back only when nothing reads them any more.
    void release() noexcept;

private:
    friend class CameraReader;
    FrameSet(std::shared_ptr<detail::ReaderState> state, const FrameEvent& event);

    std::shared_ptr<detail::ReaderState> m_state;
    FrameEvent m_event;
};

/**
 * @brief Reads one group of cameras from the camera plugin.
 *
 * @code
 * CameraReader reader = CameraReader::open("head");
 * while (auto frames = reader.next(std::chrono::seconds(1)))
 * {
 *     std::span<const std::byte> left = frames->image(0).data;
 * }
 * @endcode
 */
class CameraReader
{
public:
    /**
     * @brief What a camera plugin publishes, including groups this build cannot read.
     * @throws CameraReaderError if no plugin answers or the plugin refuses this reader.
     */
    static std::vector<GroupListing> list(const PluginLocation& plugin = {});

    /**
     * @brief Subscribe to one group. A reader for a second group is a second CameraReader.
     * @throws CameraReaderError if no plugin answers, the plugin refuses the subscription, or
     * what it answers is not the group that was asked for in a form this build can read; the
     * message says why.
     */
    static CameraReader open(const std::string& group,
                             const ReaderOptions& options = {},
                             const PluginLocation& plugin = {});

    CameraReader(CameraReader&& other) noexcept;
    CameraReader& operator=(CameraReader&& other) noexcept;
    CameraReader(const CameraReader&) = delete;
    CameraReader& operator=(const CameraReader&) = delete;
    ~CameraReader();

    const GroupDescription& group() const;

    /**
     * @brief Wait up to `timeout` for a frame set.
     *
     * With newest order the call asks the plugin, which holds the newest frame set there is at
     * that moment, or the next one published: a reader that was busy gets what the camera saw
     * last, not what it saw when the reader stopped. A frame set that arrived for an earlier
     * call that had stopped waiting is given back and exchanged for the newest; that exchange
     * may outlast a short timeout by a few milliseconds.
     * With in order the plugin sends every frame set and this returns the next.
     *
     * @return The frame set, or nothing on timeout.
     * @throws CameraWindowExhausted if the reader already holds its window's worth.
     * @throws CameraDisconnected if the plugin is gone.
     * @throws CameraReaderError if the plugin sent something a reader cannot use.
     */
    std::optional<FrameSet> next(std::chrono::milliseconds timeout);

    /**
     * @brief Becomes readable when a frame set may be waiting; for waiting on several readers
     * at once. A newest-order reader is sent a frame set only after next() asked for one, so
     * call next() first, with a timeout of zero, and again when the descriptor is readable.
     */
    int descriptor() const;

private:
    explicit CameraReader(std::shared_ptr<detail::ReaderState> state);

    std::shared_ptr<detail::ReaderState> m_state;
};

} // namespace core::camera
