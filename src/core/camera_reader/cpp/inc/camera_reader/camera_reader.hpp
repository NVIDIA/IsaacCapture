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
 * object is destroyed or release() is called. Hold at most as many as the reader's window.
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

    // Changes when the camera restarted.
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

    // Gives the frame set back before the object is destroyed. Its views are no longer valid.
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
 * CameraReader reader = CameraReader::open(endpoint, "head");
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
     * @brief What the plugin behind `endpoint` publishes.
     * @throws CameraReaderError if nothing listens there or the plugin refuses this reader.
     */
    static std::vector<GroupDescription> list(const std::string& endpoint);

    /**
     * @brief Subscribe to one group. A reader for a second group is a second CameraReader.
     * @throws CameraReaderError if nothing listens on the endpoint or the plugin refuses the
     * subscription; the message says why.
     */
    static CameraReader open(const std::string& endpoint, const std::string& group, const ReaderOptions& options = {});

    CameraReader(CameraReader&& other) noexcept;
    CameraReader& operator=(CameraReader&& other) noexcept;
    CameraReader(const CameraReader&) = delete;
    CameraReader& operator=(const CameraReader&) = delete;
    ~CameraReader();

    const GroupDescription& group() const;

    /**
     * @brief Wait up to `timeout` for a frame set. With newest order this is the newest one the
     * plugin has sent; older ones waiting behind it are given back and counted as skipped.
     * @return The frame set, or nothing on timeout.
     * @throws CameraDisconnected if the plugin is gone.
     * @throws CameraReaderError if the plugin sent a frame set that does not match the group.
     */
    std::optional<FrameSet> next(std::chrono::milliseconds timeout);

    // Becomes readable when a frame set may be waiting; for waiting on several readers at once.
    int descriptor() const;

private:
    explicit CameraReader(std::shared_ptr<detail::ReaderState> state);

    std::shared_ptr<detail::ReaderState> m_state;
};

} // namespace core::camera
