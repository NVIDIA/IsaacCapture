// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// What PoseProvider reads from one stereo pair of one camera source, in host memory. Times are int64_t nanoseconds.
#pragma once

#include <pose_engine_api/engine_types.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace pose_provider
{

enum class Shutter
{
    Unknown,
    Global,
    Rolling,
};

/** @brief The acquisition event that a stream's sample times mark. */
enum class TimedEvent
{
    Unknown,
    ExposureStart,
    ExposureMidpoint,
    FrameEnd,
};

/** @brief Pinhole intrinsics of the delivered pixels. The pixel origin is the upper-left pixel center. */
struct Intrinsics
{
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    DistortionModel distortion = DistortionModel::None;
    std::vector<double> coefficients; //!< coefficient_count(distortion) values.
};

/**
 * @brief One stream of the selected pair. Its role comes from the source's pair declaration or, for a source that
 * declares none, from the input's configuration; never from stream order.
 */
struct StreamDescription
{
    std::string stream_id; //!< The source's identity for this stream.
    uint32_t width = 0;
    uint32_t height = 0;
    PixelFormat format = PixelFormat::Gray8;
    std::optional<double> rate_hz; //!< Nominal rate.
    Shutter shutter = Shutter::Unknown;
    TimedEvent timed_event = TimedEvent::Unknown;
    std::optional<Intrinsics> intrinsics;
    std::optional<Transform> T_rig_camera; //!< Optical frame into the rig frame that both eyes share.
};

/**
 * @brief The first event of a working input: the subscription is active and this holds for every later frame.
 *
 * An absent value is unknown, never zero or identity. Values are as the source reports them, converted only where
 * the conversion is exact; a value with no exact form here fails the input instead.
 */
struct SourceDescription
{
    std::string backend;
    std::string device_id; //!< Serial or device path.
    std::string session_id; //!< This source run; a new run fails the input.
    StreamDescription left;
    StreamDescription right;
};

/** @brief One stream's delivered image. */
struct StreamImage
{
    ImageLayout layout; //!< As delivered, including the actual stride.
    const std::byte* data = nullptr; //!< First byte of the first plane; null once released.
    std::size_t size_bytes = 0; //!< Readable bytes from `data`; 0 once released.
    std::optional<int64_t> local_common_clock_ns; //!< Sample time of the stream's timed event; absent if not given.
    std::optional<int64_t> raw_device_clock_ns; //!< The same sample time on the device clock; absent if not given.
};

/**
 * @brief One paired measurement and the source's lease on its images. Move-only; a frame may outlive its input.
 *
 * release(), destruction or assignment ends the lease without blocking. After release(), both eyes' data are null,
 * while the capture number, layouts and times stay readable. Release after the last read; a source may withhold further
 * frames while frames are held, and reports what it skips as FrameLoss.
 */
class StereoFrame
{
public:
    /** @brief Destroying the last copy of @p lease returns both images to the source; it must not block or throw. */
    StereoFrame(uint64_t sequence, StreamImage left, StreamImage right, std::shared_ptr<const void> lease) noexcept
        : sequence_(sequence), left_(left), right_(right), lease_(std::move(lease))
    {
    }

    StereoFrame(StereoFrame&& other) noexcept
        : sequence_(other.sequence_), left_(other.left_), right_(other.right_), lease_(std::move(other.lease_))
    {
        other.release();
    }

    StereoFrame& operator=(StereoFrame&& other) noexcept
    {
        if (this != &other)
        {
            release();
            sequence_ = other.sequence_;
            left_ = other.left_;
            right_ = other.right_;
            lease_ = std::move(other.lease_);
            other.release();
        }
        return *this;
    }

    StereoFrame(const StereoFrame&) = delete;
    StereoFrame& operator=(const StereoFrame&) = delete;

    ~StereoFrame()
    {
        release();
    }

    /** @brief The pair's capture number at the source, shared by both eyes. */
    uint64_t sequence() const
    {
        return sequence_;
    }

    const StreamImage& left() const
    {
        return left_;
    }

    const StreamImage& right() const
    {
        return right_;
    }

    /** @brief Ends the lease; repeated calls have no further effect. */
    void release() noexcept
    {
        left_.data = right_.data = nullptr;
        left_.size_bytes = right_.size_bytes = 0;
        lease_.reset();
    }

private:
    uint64_t sequence_ = 0;
    StreamImage left_;
    StreamImage right_;
    std::shared_ptr<const void> lease_;
};

/** @brief Captures missing before the next frame: count >= 1 capture numbers, starting at first_sequence. */
struct FrameLoss
{
    uint64_t first_sequence = 0;
    uint64_t count = 0;
};

/** @brief Delivery to this input ended. A last event. */
struct InputEnded
{
    bool requested = false; //!< True when detach() ended it; false when the source ended first.
};

/**
 * @brief A last event: the subscription failed, or the source was lost, restarted or changed its description. A source
 * whose stop looks the same as a crash reports both this way, and ends with InputEnded only after detach().
 */
struct InputFailed
{
    std::string reason; //!< Names the backend, device and stream where known, and what failed.
};

using CameraEvent = std::variant<SourceDescription, StereoFrame, FrameLoss, InputEnded, InputFailed>;

/**
 * @brief Pose Provider's input from one camera source.
 *
 * poll() yields SourceDescription first, then StereoFrame and FrameLoss in delivery order, and ends with exactly one
 * InputEnded or InputFailed, which may also come first. Until its first event the input may keep trying to subscribe;
 * after it, the input never reconnects or resubscribes on its own. Not thread-safe: one thread makes every call,
 * including destruction, and releases every frame. poll(), detach() and wait_fd() never wait on the source or throw;
 * failures arrive as InputFailed.
 */
class ICameraInput
{
public:
    /**
     * @brief Ends the subscription without detach() and without an event; frames already taken stay valid. It runs on
     * the caller's control thread, so any wait for the implementation's own thread must be short and bounded.
     */
    virtual ~ICameraInput() = default;

    /** @brief Moves the next event into @p out; false, leaving @p out unchanged, when none is available yet. */
    virtual bool poll(CameraEvent& out) = 0;

    /**
     * @brief Ends delivery to this input only: poll() then returns what was already accepted, then InputEnded, unless
     * the input fails first. Repeated calls, and calls after the last event, have no further effect.
     */
    virtual void detach() = 0;

    /**
     * @brief Readable whenever poll() has an event. It may also wake without one, but a poll() that returns false
     * leaves it unreadable. Valid until destruction; wait on it, never read or close it.
     */
    virtual int wait_fd() const = 0;
};

} // namespace pose_provider
