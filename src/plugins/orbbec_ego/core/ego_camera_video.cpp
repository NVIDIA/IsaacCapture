// SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#if defined(ORBBEC_EGO_ENABLE_PREVIEW)
#    include "preview.hpp"
#endif

#include "diagnostics.hpp"
#include "ego_camera_impl.hpp"
#include "inc/ego_camera/cancellation.hpp"
#include "inc/ego_camera/sdk_timestamp.hpp"

#include <oxr_utils/os_time.hpp>

#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>
#include <utility>

namespace plugins::ego
{

using detail::describe_profiles;
using detail::RecoverableCaptureError;
using detail::to_ob_frame;
using detail::to_ob_sensor;

namespace
{

constexpr size_t kMaxQueuedVideoFrameSets = 256;

OBFormat to_ob_format(core::OrbbecEgoPixelFormat format)
{
    switch (format)
    {
    case core::OrbbecEgoPixelFormat_Mjpg:
        return OB_FORMAT_MJPG;
    case core::OrbbecEgoPixelFormat_H264:
        return OB_FORMAT_H264;
    case core::OrbbecEgoPixelFormat_H265:
        return OB_FORMAT_H265;
    default:
        throw std::invalid_argument("Unsupported Ego pixel format");
    }
}

void print_profiles(const std::shared_ptr<ob::Device>& device, core::OrbbecEgoCameraStream stream)
{
    const auto profiles = device->getSensor(to_ob_sensor(stream))->getStreamProfileList();
    std::ostringstream profiles_message;
    profiles_message << "Available " << core::EnumNameOrbbecEgoCameraStream(stream) << " profiles:";
    for (uint32_t index = 0; index < profiles->getCount(); ++index)
    {
        const auto profile = profiles->getProfile(index);
        if (!profile->is<ob::VideoStreamProfile>())
            continue;
        const auto video = profile->as<ob::VideoStreamProfile>();
        profiles_message << " " << video->getWidth() << "x" << video->getHeight() << "@" << video->getFps() << " "
                         << ob::TypeHelper::convertOBFormatTypeToString(video->getFormat());
    }
    detail::log_warning() << profiles_message.str();
}

} // namespace

namespace detail
{

OBFrameType to_ob_frame(core::OrbbecEgoCameraStream stream)
{
    switch (stream)
    {
    case core::OrbbecEgoCameraStream_ColorLeft:
        return OB_FRAME_COLOR_LEFT;
    case core::OrbbecEgoCameraStream_ColorRight:
        return OB_FRAME_COLOR_RIGHT;
    default:
        throw std::invalid_argument("Unsupported Ego camera stream");
    }
}

OBSensorType to_ob_sensor(core::OrbbecEgoCameraStream stream)
{
    switch (stream)
    {
    case core::OrbbecEgoCameraStream_ColorLeft:
        return OB_SENSOR_COLOR_LEFT;
    case core::OrbbecEgoCameraStream_ColorRight:
        return OB_SENSOR_COLOR_RIGHT;
    default:
        throw std::invalid_argument("Unsupported Ego camera stream");
    }
}

bool has_sensor(const std::shared_ptr<ob::Device>& device, OBSensorType sensor)
{
    const auto sensors = device->getSensorList();
    for (uint32_t index = 0; index < sensors->getCount(); ++index)
    {
        if (sensors->getSensorType(index) == sensor)
            return true;
    }
    return false;
}

std::shared_ptr<ob::VideoStreamProfile> select_profile(const std::shared_ptr<ob::Device>& device,
                                                       const StreamConfig& stream,
                                                       const CaptureConfig& config)
{
    validate_stream_config(stream, config);
    const auto width = stream.width != 0 ? stream.width : config.width;
    const auto height = stream.height != 0 ? stream.height : config.height;
    const auto fps = stream.fps != 0 ? stream.fps : config.fps;
    const auto format = to_ob_format(stream.pixel_format);
    try
    {
        const auto profile = device->getSensor(to_ob_sensor(stream.camera))
                                 ->getStreamProfileList()
                                 ->getVideoStreamProfile(
                                     static_cast<int>(width), static_cast<int>(height), format, static_cast<int>(fps));
        detail::log_info()
            << "Selected " << core::EnumNameOrbbecEgoCameraStream(stream.camera) << " profile: " << profile->getWidth()
            << "x" << profile->getHeight() << "@" << profile->getFps() << " "
            << core::EnumNameOrbbecEgoPixelFormat(stream.pixel_format) << std::endl;
        return profile;
    }
    catch (const ob::Error& error)
    {
        print_profiles(device, stream.camera);
        throw std::runtime_error("No matching Ego profile for " +
                                 std::string(core::EnumNameOrbbecEgoCameraStream(stream.camera)) + ": " + error.what());
    }
}

std::shared_ptr<ob::Config> make_video_config(const ActiveProfiles& profiles, const std::vector<StreamConfig>& streams)
{
    auto config = std::make_shared<ob::Config>();
    for (const auto& [_, profile] : profiles)
        config->enableStream(profile);
    const bool has_interframe_video = std::any_of(streams.begin(), streams.end(),
                                                  [](const StreamConfig& stream)
                                                  {
                                                      return stream.pixel_format == core::OrbbecEgoPixelFormat_H264 ||
                                                             stream.pixel_format == core::OrbbecEgoPixelFormat_H265;
                                                  });
    // Single-eye capture must not wait for an unrequested color frame.
    // MJPG capture must retain an available eye even when its stereo partner is absent.
    config->setFrameAggregateOutputMode(profiles.size() == 1 ? OB_FRAME_AGGREGATE_OUTPUT_ANY_SITUATION :
                                        has_interframe_video ? OB_FRAME_AGGREGATE_OUTPUT_COLOR_FRAME_REQUIRE :
                                                               OB_FRAME_AGGREGATE_OUTPUT_ANY_SITUATION);
    return config;
}

std::string describe_profiles(const ActiveProfiles& profiles)
{
    std::ostringstream output;
    bool first = true;
    for (const auto& [stream, profile] : profiles)
    {
        if (!first)
            output << ", ";
        first = false;
        output << core::EnumNameOrbbecEgoCameraStream(stream) << "=" << profile->getWidth() << "x"
               << profile->getHeight() << "@" << profile->getFps() << " "
               << ob::TypeHelper::convertOBFormatTypeToString(profile->getFormat());
    }
    return output.str();
}

} // namespace detail

struct EgoCamera::Impl::VideoStartupReadiness
{
    struct ExpectedFrame
    {
        OBFrameType frame_type;
        uint32_t width;
        uint32_t height;
        uint32_t fps;
        OBFormat format;
        std::string name;
    };

    std::mutex mutex;
    std::condition_variable wake;
    std::vector<bool> seen;
    std::string error;
    std::atomic<bool> complete{ false };
    std::vector<ExpectedFrame> expected_frames;
};

std::shared_ptr<EgoCamera::Impl::VideoStartupReadiness> EgoCamera::Impl::prepare_video_startup_readiness() const
{
    std::vector<VideoStartupReadiness::ExpectedFrame> expected_frames;
    expected_frames.reserve(active_profiles_.size());
    for (const auto& [stream, profile] : active_profiles_)
    {
        expected_frames.push_back({ to_ob_frame(stream), profile->getWidth(), profile->getHeight(), profile->getFps(),
                                    profile->getFormat(), core::EnumNameOrbbecEgoCameraStream(stream) });
    }
    auto readiness = std::make_shared<VideoStartupReadiness>();
    readiness->seen.resize(expected_frames.size());
    readiness->expected_frames = std::move(expected_frames);
    return readiness;
}

void EgoCamera::Impl::start_initial_video(const std::shared_ptr<ob::Config>& pipeline_config,
                                          const std::shared_ptr<VideoStartupReadiness>& readiness)
{
    const auto& expected_frames = readiness->expected_frames;
    try
    {
        pipeline_->start(
            pipeline_config,
            [this, readiness, expected_frames](std::shared_ptr<ob::FrameSet> frame_set)
            {
                if (!frame_set || !accepting_callbacks_.load(std::memory_order_acquire))
                    return;
                const bool active_capture = capture_active_.load(std::memory_order_acquire);
                std::vector<bool> observed(expected_frames.size());
                std::string observation_error;
                if (!readiness->complete.load(std::memory_order_acquire))
                {
                    try
                    {
                        for (size_t index = 0; index < expected_frames.size(); ++index)
                        {
                            const auto& expected = expected_frames[index];
                            const auto raw_frame = frame_set->getFrame(expected.frame_type);
                            if (!raw_frame)
                                continue;
                            const auto frame = raw_frame->as<ob::VideoFrame>();
                            const auto profile =
                                frame ? frame->getStreamProfile()->as<ob::VideoStreamProfile>() : nullptr;
                            if (!frame || !profile || frame->getWidth() != expected.width ||
                                frame->getHeight() != expected.height || frame->getFormat() != expected.format ||
                                profile->getFps() != expected.fps)
                            {
                                observation_error = "SDK returned a different active profile for " + expected.name;
                                if (frame && profile)
                                {
                                    observation_error +=
                                        ": actual=" + std::to_string(frame->getWidth()) + "x" +
                                        std::to_string(frame->getHeight()) + "@" + std::to_string(profile->getFps()) +
                                        " " + ob::TypeHelper::convertOBFormatTypeToString(frame->getFormat());
                                }
                                break;
                            }
                            observed[index] = true;
                        }
                    }
                    catch (const std::exception& error)
                    {
                        observation_error = error.what();
                    }
                    catch (...)
                    {
                        observation_error = "unknown error while inspecting a frameset";
                    }
                }
                if (!observation_error.empty())
                {
                    std::lock_guard<std::mutex> lock(readiness->mutex);
                    readiness->error = std::move(observation_error);
                    readiness->wake.notify_one();
                    return;
                }
                if (!active_capture)
                {
                    if (readiness->complete.load(std::memory_order_relaxed))
                        return;
                    std::lock_guard<std::mutex> lock(readiness->mutex);
                    auto& seen = readiness->seen;
                    for (size_t index = 0; index < observed.size(); ++index)
                        seen[index] = seen[index] || observed[index];
                    if (std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }))
                    {
                        readiness->complete.store(true, std::memory_order_release);
                        readiness->wake.notify_one();
                    }
                    return;
                }
                receive_video_frame_set(std::move(frame_set));
            });
        video_pipeline_started_ = true;
    }
    catch (const ob::Error& error)
    {
        throw std::runtime_error("Ego uid=" + selected_device_uid_ + " firmware=" + selected_firmware_ +
                                 " rejected simultaneous profiles [" + describe_profiles(active_profiles_) +
                                 "]: " + error.what());
    }
}

void EgoCamera::Impl::wait_for_video_readiness(const std::shared_ptr<VideoStartupReadiness>& readiness)
{
    const auto& expected_frames = readiness->expected_frames;
    std::string readiness_error;
    {
        std::unique_lock<std::mutex> lock(readiness->mutex);
        const bool ready = wait_capture_ready(
            readiness->wake, lock, std::chrono::seconds(10),
            [&readiness] { return readiness->complete.load(std::memory_order_acquire) || !readiness->error.empty(); },
            config_.stop_requested);
        if (!ready)
        {
            readiness_error = "timed out waiting for";
            for (size_t index = 0; index < expected_frames.size(); ++index)
            {
                if (!readiness->seen[index])
                    readiness_error += " " + expected_frames[index].name;
            }
        }
        else
            readiness_error = readiness->error;
    }
    if (!readiness_error.empty())
    {
        throw std::runtime_error("Ego uid=" + selected_device_uid_ + " firmware=" + selected_firmware_ +
                                 " active profiles [" + describe_profiles(active_profiles_) +
                                 "] failed readiness: " + readiness_error);
    }
}

void EgoCamera::Impl::start_recovered_video(const std::shared_ptr<ob::Config>& pipeline_config)
{
    pipeline_ = std::make_unique<ob::Pipeline>(device_);
    pipeline_->start(pipeline_config, [this](std::shared_ptr<ob::FrameSet> frame_set)
                     { receive_video_frame_set(std::move(frame_set)); });
    video_pipeline_started_ = true;
}

void EgoCamera::Impl::wait_for_recovered_video_readiness()
{
    const auto ready = [this]
    {
        return std::all_of(streams_.begin(), streams_.end(),
                           [this](const StreamConfig& stream) { return stats_.at(stream.camera).epoch_frame_count > 0; });
    };
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!ready() && std::chrono::steady_clock::now() < deadline)
    {
        check_cancelled();
        drain_video_frames();
        drain_events();
        if (const auto failure = sink_->metadata_error(); !failure.empty())
            throw std::runtime_error("Ego metadata publication failed: " + failure);
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (!async_error_.empty())
            throw std::runtime_error(async_error_);
    }
    if (ready())
        return;
    std::string missing;
    for (const auto& stream : streams_)
    {
        if (stats_.at(stream.camera).epoch_frame_count == 0)
            missing += " " + std::string(core::EnumNameOrbbecEgoCameraStream(stream.camera));
    }
    throw std::runtime_error("Ego uid=" + selected_device_uid_ + " firmware=" + selected_firmware_ +
                             " active profiles [" + describe_profiles(active_profiles_) +
                             "] timed out waiting for video streams after reconnect:" + missing);
}

void EgoCamera::Impl::validate_video_liveness() const
{
    if (!capture_active_.load(std::memory_order_acquire))
        return;
    const int64_t now = core::os_monotonic_now_ns();
    for (const auto& stream : streams_)
    {
        if (video_liveness_.expired(stream.camera, now, capture_epoch_started_ns_))
        {
            throw RecoverableCaptureError("Ego " + std::string(core::EnumNameOrbbecEgoCameraStream(stream.camera)) +
                                          " stopped delivering frames for five seconds");
        }
    }
}

void EgoCamera::Impl::receive_video_frame_set(std::shared_ptr<ob::FrameSet> frame_set)
{
    if (!frame_set || !accepting_callbacks_.load(std::memory_order_acquire) ||
        !capture_active_.load(std::memory_order_acquire))
        return;
    const int64_t arrival_time_local_common_clock_ns = core::os_monotonic_now_ns();
    try
    {
        // Arrival proves liveness even when a slow consumer fills the queue.
        video_liveness_.observe(arrival_time_local_common_clock_ns, [&frame_set](auto stream)
                                { return static_cast<bool>(frame_set->getFrame(to_ob_frame(stream))); });
        {
            std::lock_guard<std::mutex> lock(video_queue_mutex_);
            if (!accepting_callbacks_.load(std::memory_order_relaxed))
                return;
            if (video_frame_sets_.size() >= kMaxQueuedVideoFrameSets)
            {
                ++auxiliary_stats_.dropped_video_frame_sets;
                return;
            }
            video_frame_sets_.emplace_back(std::move(frame_set), arrival_time_local_common_clock_ns);
        }
        video_queue_cv_.notify_one();
    }
    catch (const std::exception& error)
    {
        set_async_error(std::string("Video callback failed: ") + error.what());
    }
    catch (...)
    {
        set_async_error("Video callback failed: unknown error");
    }
}

void EgoCamera::Impl::drain_video_frames(bool wait_for_frame)
{
    std::deque<std::pair<std::shared_ptr<ob::FrameSet>, int64_t>> pending;
    uint64_t dropped = 0;
    {
        std::unique_lock<std::mutex> lock(video_queue_mutex_);
        if (wait_for_frame && video_frame_sets_.empty())
            video_queue_cv_.wait_for(lock, std::chrono::milliseconds(20));
        pending.swap(video_frame_sets_);
        dropped = auxiliary_stats_.dropped_video_frame_sets;
    }
    report_dropped_video_frame_sets(dropped);
    for (const auto& [frame_set, arrival_time_local_common_clock_ns] : pending)
        process_frame_set(frame_set, arrival_time_local_common_clock_ns);
}

// Overflow drops whole frame sets for as long as the host lags, so warn at most every five seconds.
void EgoCamera::Impl::report_dropped_video_frame_sets(uint64_t dropped)
{
    if (dropped == reported_dropped_video_frame_sets_)
        return;
    const int64_t now = core::os_monotonic_now_ns();
    if (reported_dropped_video_frame_sets_ != 0 && now - last_drop_warning_ns_ < 5'000'000'000)
        return;
    detail::log_warning() << "Ego video callback queue was full; " << dropped - reported_dropped_video_frame_sets_
                          << " more video frame sets lost (" << dropped << " total); recording continues";
    reported_dropped_video_frame_sets_ = dropped;
    last_drop_warning_ns_ = now;
}

void EgoCamera::Impl::process_frame_set(const std::shared_ptr<ob::FrameSet>& frame_set,
                                        int64_t arrival_time_local_common_clock_ns)
{
    if (!frame_set)
        return;
    for (const auto& stream : streams_)
    {
        const auto raw_frame = frame_set->getFrame(to_ob_frame(stream.camera));
        if (!raw_frame)
            continue;
        const auto frame = raw_frame->as<ob::VideoFrame>();
        if (!frame)
            throw std::runtime_error("Ego video frameset contained a non-video frame");

        const auto profile = frame->getStreamProfile()->as<ob::VideoStreamProfile>();
        const auto expected = active_profiles_.at(stream.camera);
        if (!profile || frame->getWidth() != expected->getWidth() || frame->getHeight() != expected->getHeight() ||
            frame->getFormat() != expected->getFormat() || profile->getFps() != expected->getFps())
        {
            throw std::runtime_error("Ego " + std::string(core::EnumNameOrbbecEgoCameraStream(stream.camera)) +
                                     " delivered a frame outside the selected active profile " +
                                     std::to_string(expected->getWidth()) + "x" +
                                     std::to_string(expected->getHeight()) + "@" + std::to_string(expected->getFps()) +
                                     " " + ob::TypeHelper::convertOBFormatTypeToString(expected->getFormat()));
        }
        CapturedFrame captured;
        captured.metadata.stream = stream.camera;
        captured.metadata.sequence_number = frame->getIndex();
        captured.metadata.width = frame->getWidth();
        captured.metadata.height = frame->getHeight();
        captured.metadata.fps = profile->getFps();
        captured.metadata.pixel_format = stream.pixel_format;
        captured.metadata.encoded_bytes = frame->getDataSize();
        captured.metadata.capture_epoch = capture_epoch_;
        captured.encoded_data.assign(frame->getData(), frame->getData() + frame->getDataSize());
        captured.sample_time_raw_device_clock_ns = sdk_timestamp_ns(frame->getTimeStampUs());
        captured.sample_time_local_common_clock_ns = arrival_time_local_common_clock_ns;
        auto& stats = stats_[stream.camera];
        if (stats.epoch_frame_count > 0 && captured.metadata.sequence_number <= stats.last_sequence)
        {
            throw std::runtime_error(
                "Ego " + std::string(core::EnumNameOrbbecEgoCameraStream(stream.camera)) +
                " delivered a non-increasing sequence: previous=" + std::to_string(stats.last_sequence) +
                " current=" + std::to_string(captured.metadata.sequence_number));
        }
        if (stats.epoch_frame_count > 0 && captured.sample_time_raw_device_clock_ns <= stats.last_device_timestamp_ns)
        {
            throw std::runtime_error("Ego " + std::string(core::EnumNameOrbbecEgoCameraStream(stream.camera)) +
                                     " delivered a non-increasing raw device timestamp: previous=" +
                                     std::to_string(stats.last_device_timestamp_ns) +
                                     " current=" + std::to_string(captured.sample_time_raw_device_clock_ns));
        }
        std::function<void(const CapturedFrame&)> accepted_frame;
#if defined(ORBBEC_EGO_ENABLE_PREVIEW)
        if (preview_)
            accepted_frame = [this](const CapturedFrame& accepted) { preview_->submit(accepted); };
#endif
        sink_->on_frame(captured, accepted_frame);

        if (stats.epoch_frame_count > 0 && captured.metadata.sequence_number > stats.last_sequence + 1)
        {
            const uint64_t missing = captured.metadata.sequence_number - stats.last_sequence - 1;
            stats.sequence_gaps += missing;
            detail::log_warning() << "Ego " << core::EnumNameOrbbecEgoCameraStream(stream.camera)
                                  << " sequence gap: previous=" << stats.last_sequence
                                  << " current=" << captured.metadata.sequence_number << " missing=" << missing;
        }
        stats.frame_count++;
        stats.epoch_frame_count++;
        stats.byte_count += captured.encoded_data.size();
        stats.last_sequence = captured.metadata.sequence_number;
        stats.last_device_timestamp_ns = captured.sample_time_raw_device_clock_ns;
    }
}

} // namespace plugins::ego
