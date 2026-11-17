// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A scripted ICameraInput. The test queues events on a FakeCameraSource, which outlives the FakeCameraInput that the
// code under test owns and polls. Single-threaded; its descriptor is readable exactly while poll() has an event.
#pragma once

#include <pose_camera_input/camera_input.hpp>
#include <sys/eventfd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <poll.h>
#include <stdexcept>
#include <system_error>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <variant>
#include <vector>

namespace pose_provider::testing
{

/** @brief A frame to deliver. Each eye's data and size_bytes come from its bytes; the rest is delivered as given. */
struct FakeFrame
{
    uint64_t sequence = 0;
    StreamImage left;
    StreamImage right;
    std::vector<std::byte> left_bytes;
    std::vector<std::byte> right_bytes;
};

/**
 * @brief The test's side of a fake input: the event script and a record of what the input's user did.
 *
 * push() throws std::logic_error for an order no input could deliver: data before the description, a second
 * description, or anything after a queued last event. After detach(), everything but InputFailed is dropped.
 */
class FakeCameraSource
{
public:
    FakeCameraSource() : fd_(::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK))
    {
        if (fd_ < 0)
        {
            throw std::system_error(errno, std::generic_category(), "eventfd");
        }
    }

    ~FakeCameraSource()
    {
        ::close(fd_);
    }

    FakeCameraSource(const FakeCameraSource&) = delete;
    FakeCameraSource& operator=(const FakeCameraSource&) = delete;

    void push(SourceDescription description)
    {
        enqueue(std::move(description));
    }

    void push(FakeFrame frame)
    {
        enqueue(std::move(frame));
    }

    void push(FrameLoss loss)
    {
        enqueue(loss);
    }

    void push(InputFailed failure)
    {
        enqueue(std::move(failure));
    }

    /** @brief Queues the end of a source that stopped without detach(). */
    void push_end()
    {
        enqueue(InputEnded{ false });
    }

    /** @brief While set, the InputEnded that detach() makes pending is withheld, as by a source slow to drain. */
    void withhold_end(bool withhold)
    {
        withhold_end_ = withhold;
        sync_fd();
    }

    uint64_t frames_delivered() const
    {
        return delivered_;
    }

    uint64_t frames_released() const
    {
        return released_;
    }

    int detach_calls() const
    {
        return detach_calls_;
    }

    bool input_destroyed() const
    {
        return input_destroyed_;
    }

private:
    friend class FakeCameraInput;

    using Step = std::variant<SourceDescription, FakeFrame, FrameLoss, InputEnded, InputFailed>;

    // Owns both eyes' bytes and keeps the source alive; destroying it counts the release.
    struct Lease
    {
        Lease(std::shared_ptr<FakeCameraSource> owner, FakeFrame& frame)
            : source(std::move(owner)), left(std::move(frame.left_bytes)), right(std::move(frame.right_bytes))
        {
        }

        ~Lease()
        {
            ++source->released_;
        }

        std::shared_ptr<FakeCameraSource> source;
        std::vector<std::byte> left;
        std::vector<std::byte> right;
    };

    void enqueue(Step step)
    {
        if (closed_)
        {
            throw std::logic_error("FakeCameraSource: nothing follows the last event");
        }
        const bool description = std::holds_alternative<SourceDescription>(step);
        const bool failure = std::holds_alternative<InputFailed>(step);
        const bool last = failure || std::holds_alternative<InputEnded>(step);
        if (description ? described_ : !(described_ || last))
        {
            throw std::logic_error("FakeCameraSource: the description comes first, once");
        }
        described_ = described_ || description;
        closed_ = last;
        // The order is checked first, so a script is valid or invalid whether or not the input has detached.
        if (detached_ && !failure)
        {
            return;
        }
        queue_.push_back(std::move(step));
        sync_fd();
    }

    // An event is waiting: a queued step, or the InputEnded that detach() makes pending and withhold_end() allows.
    bool ready() const
    {
        return !finished_ && (!queue_.empty() || (detached_ && !withhold_end_));
    }

    // Readable exactly while ready().
    void sync_fd()
    {
        const bool want = ready();
        if (want && !signaled_)
        {
            const uint64_t one = 1;
            signaled_ = ::write(fd_, &one, sizeof(one)) == sizeof(one);
        }
        else if (!want && signaled_)
        {
            uint64_t count = 0;
            signaled_ = ::read(fd_, &count, sizeof(count)) != sizeof(count);
        }
    }

    bool take(CameraEvent& out, const std::shared_ptr<FakeCameraSource>& self)
    {
        if (!ready())
        {
            return false;
        }
        if (queue_.empty())
        {
            out = InputEnded{ true };
        }
        else
        {
            Step step = std::move(queue_.front());
            queue_.pop_front();
            std::visit(
                [&](auto& event)
                {
                    if constexpr (std::is_same_v<std::decay_t<decltype(event)>, FakeFrame>)
                    {
                        out = deliver(std::move(event), self);
                    }
                    else
                    {
                        out = std::move(event);
                    }
                },
                step);
        }
        delivered_ += std::holds_alternative<StereoFrame>(out) ? 1 : 0;
        finished_ = std::holds_alternative<InputEnded>(out) || std::holds_alternative<InputFailed>(out);
        sync_fd();
        return true;
    }

    static StereoFrame deliver(FakeFrame frame, std::shared_ptr<FakeCameraSource> self)
    {
        auto lease = std::make_shared<Lease>(std::move(self), frame);
        frame.left.data = lease->left.data();
        frame.left.size_bytes = lease->left.size();
        frame.right.data = lease->right.data();
        frame.right.size_bytes = lease->right.size();
        return StereoFrame(frame.sequence, frame.left, frame.right, std::move(lease));
    }

    void detach()
    {
        ++detach_calls_;
        detached_ = true;
        sync_fd();
    }

    int fd_;
    std::deque<Step> queue_;
    bool described_ = false;
    bool closed_ = false; // A last event is queued.
    bool finished_ = false; // The last event was taken.
    bool detached_ = false;
    bool withhold_end_ = false;
    bool signaled_ = false;
    bool input_destroyed_ = false;
    int detach_calls_ = 0;
    uint64_t delivered_ = 0;
    uint64_t released_ = 0;
};

class FakeCameraInput final : public ICameraInput
{
public:
    explicit FakeCameraInput(std::shared_ptr<FakeCameraSource> source) : source_(std::move(source))
    {
    }

    ~FakeCameraInput() override
    {
        source_->input_destroyed_ = true;
    }

    bool poll(CameraEvent& out) override
    {
        return source_->take(out, source_);
    }

    void detach() override
    {
        source_->detach();
    }

    int wait_fd() const override
    {
        return source_->fd_;
    }

private:
    std::shared_ptr<FakeCameraSource> source_;
};

/** @brief A Stereo source that PoseProvider's camera profile accepts, with the left eye at the rig origin. */
inline SourceDescription fake_stereo_description()
{
    SourceDescription description;
    description.backend = "oak";
    description.device_id = "fake-device";
    description.session_id = "fake-session";
    for (StreamDescription* eye : { &description.left, &description.right })
    {
        eye->width = 8;
        eye->height = 4;
        eye->format = PixelFormat::Gray8;
        eye->rate_hz = 30.0;
        eye->shutter = Shutter::Global;
        eye->timed_event = TimedEvent::ExposureMidpoint;
        eye->intrinsics = Intrinsics{ 100.0, 100.0, 3.5, 1.5, DistortionModel::None, {} };
        eye->T_rig_camera = Transform{};
    }
    description.left.stream_id = "fake/left";
    description.right.stream_id = "fake/right";
    description.right.T_rig_camera->translation_m = { 0.075, 0.0, 0.0 };
    return description;
}

/** @brief A frame matching fake_stereo_description(), with both times on both eyes and distinct bytes per eye. */
inline FakeFrame fake_frame(uint64_t sequence)
{
    StreamImage eye;
    eye.layout = ImageLayout{ 8, 4, PixelFormat::Gray8, 8 };
    eye.raw_device_clock_ns = static_cast<int64_t>(sequence) * 33'333'333;
    eye.local_common_clock_ns = *eye.raw_device_clock_ns + 1'000'000'000;

    FakeFrame frame;
    frame.sequence = sequence;
    frame.left = eye;
    frame.right = eye;
    frame.left_bytes.assign(32, std::byte{ 0x10 });
    frame.right_bytes.assign(32, std::byte{ 0x60 });
    return frame;
}

/** @brief Whether @p fd is readable now, without waiting. */
inline bool readable(int fd)
{
    pollfd entry{ fd, POLLIN, 0 };
    return ::poll(&entry, 1, 0) == 1 && (entry.revents & POLLIN) != 0;
}

} // namespace pose_provider::testing
