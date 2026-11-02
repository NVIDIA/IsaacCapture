// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "stub_engine.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <optional>
#include <set>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>

namespace pose_provider
{
namespace
{

std::string trim(std::string_view text)
{
    const auto first = text.find_first_not_of(" \t\r");
    if (first == std::string_view::npos)
    {
        return {};
    }
    const auto last = text.find_last_not_of(" \t\r");
    return std::string(text.substr(first, last - first + 1));
}

bool finite(const std::array<double, 3>& values)
{
    return std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); });
}

bool finite_positive(double value)
{
    return std::isfinite(value) && value > 0.0;
}

bool rigid(const Transform& transform)
{
    double norm_squared = 0.0;
    for (double v : transform.rotation_xyzw)
    {
        norm_squared += v * v;
    }
    return finite(transform.translation_m) && std::isfinite(norm_squared) && std::abs(norm_squared - 1.0) < 1e-6;
}

std::string check_stream(const StreamSetup& stream, const EngineCapabilities& capabilities)
{
    const std::string name = "stream " + std::to_string(stream.id.value) + ": ";
    if (!canonical_size_bytes(stream.layout))
    {
        return name + "invalid image layout";
    }
    if (!capabilities.pixel_formats.contains(stream.layout.format))
    {
        return name + "unsupported pixel format";
    }
    if (!finite_positive(stream.rate_hz))
    {
        return name + "rate must be positive";
    }
    const CameraCalibration& calibration = stream.calibration;
    if (!finite_positive(calibration.fx) || !finite_positive(calibration.fy) || !std::isfinite(calibration.cx) ||
        !std::isfinite(calibration.cy))
    {
        return name + "invalid intrinsics";
    }
    if (!capabilities.distortion_models.contains(calibration.distortion))
    {
        return name + "unsupported distortion model";
    }
    if (calibration.coefficients.size() != coefficient_count(calibration.distortion) ||
        !std::all_of(calibration.coefficients.begin(), calibration.coefficients.end(),
                     [](double v) { return std::isfinite(v); }))
    {
        return name + "distortion coefficients are not finite or do not match the model";
    }
    if (!rigid(calibration.T_body_camera))
    {
        return name + "camera extrinsic is not a rigid transform";
    }
    return {};
}

std::string check_imu(const ImuSetup& imu)
{
    if (!rigid(imu.T_body_imu))
    {
        return "IMU extrinsic is not a rigid transform";
    }
    if (!finite_positive(imu.rate_hz))
    {
        return "IMU rate must be positive";
    }
    for (double noise : { imu.accel_noise_density, imu.gyro_noise_density, imu.accel_random_walk, imu.gyro_random_walk })
    {
        if (!std::isfinite(noise) || noise < 0.0)
        {
            return "IMU noise parameters must be non-negative";
        }
    }
    return {};
}

// Stands in for the engine library's reads of the pixels.
uint64_t read_pixels(const HostImage& image)
{
    uint64_t sum = 0;
    for (std::size_t i = 0; i < image.size_bytes; ++i)
    {
        sum += static_cast<uint8_t>(image.data[i]);
    }
    return sum;
}

} // namespace

std::string parse_stub_settings(const std::string& text, StubSettings& out)
{
    StubSettings parsed;
    std::size_t start = 0;
    while (start < text.size())
    {
        const std::size_t end = std::min(text.find_first_of(";\n", start), text.size());
        const std::string entry = trim(std::string_view(text).substr(start, end - start));
        start = end + 1;
        if (entry.empty())
        {
            continue;
        }
        const std::size_t equals = entry.find('=');
        if (equals == std::string::npos)
        {
            return "stub settings: expected key=value, got '" + entry + "'";
        }
        const std::string key = trim(std::string_view(entry).substr(0, equals));
        const std::string value = trim(std::string_view(entry).substr(equals + 1));
        uint64_t number = 0;
        const auto [end_of_number, error] = std::from_chars(value.data(), value.data() + value.size(), number);
        if (value.empty() || error != std::errc() || end_of_number != value.data() + value.size())
        {
            return "stub settings: '" + key + "' needs a non-negative integer";
        }
        if (key == "image_capacity")
        {
            parsed.image_capacity = number;
        }
        else if (key == "imu_capacity")
        {
            parsed.imu_capacity = number;
        }
        else if (key == "pose_capacity")
        {
            parsed.pose_capacity = number;
        }
        else if (key == "initializing_groups")
        {
            parsed.initializing_groups = number;
        }
        else
        {
            return "stub settings: unknown key '" + key + "'";
        }
    }
    if (parsed.image_capacity == 0 || parsed.imu_capacity == 0 || parsed.pose_capacity == 0)
    {
        return "stub settings: capacities must be positive";
    }
    out = parsed;
    return {};
}

std::unique_ptr<IPoseEngine> make_pose_engine()
{
    return std::make_unique<StubPoseEngine>();
}

StubPoseEngine::~StubPoseEngine()
{
    stop();
}

EngineCapabilities StubPoseEngine::capabilities() const
{
    return EngineCapabilities{
        { InputMode::Stereo, InputMode::StereoImu },
        { MemoryKind::Host },
        { PixelFormat::Gray8, PixelFormat::BGR888, PixelFormat::RGBA8, PixelFormat::NV12 },
        { DistortionModel::None, DistortionModel::RadTan5, DistortionModel::Rational8, DistortionModel::ThinPrism12,
          DistortionModel::Tilted14 },
    };
}

EngineIdentity StubPoseEngine::identity() const
{
    return EngineIdentity{ "pose_engine_stub", "1", "none", "none" };
}

std::string StubPoseEngine::validate_configuration(const ResolvedPoseSetup& setup) const
{
    const EngineCapabilities supported = capabilities();
    if (!supported.input_modes.contains(setup.input_mode))
    {
        return "unsupported input mode";
    }
    if (!supported.memory_kinds.contains(setup.memory_kind))
    {
        return "unsupported memory kind";
    }
    if (setup.engine_identity != identity())
    {
        return "the setup was resolved for a different engine";
    }
    if (setup.body_frame.empty())
    {
        return "the Body frame is not named";
    }
    if (setup.streams.size() != 2)
    {
        return "stereo input needs exactly two streams";
    }
    if (setup.streams[0].id == setup.streams[1].id)
    {
        return "stream IDs must differ";
    }
    for (const StreamSetup& stream : setup.streams)
    {
        if (std::string error = check_stream(stream, supported); !error.empty())
        {
            return error;
        }
    }
    if (setup.input_mode == InputMode::Stereo && setup.imu)
    {
        return "Stereo mode takes no IMU calibration";
    }
    if (setup.input_mode == InputMode::StereoImu)
    {
        if (!setup.imu)
        {
            return "StereoImu mode needs IMU calibration";
        }
        if (std::string error = check_imu(*setup.imu); !error.empty())
        {
            return error;
        }
    }
    StubSettings settings;
    return parse_stub_settings(setup.engine_settings, settings);
}

void StubPoseEngine::initialize(const ResolvedPoseSetup& setup)
{
    std::lock_guard lock(mutex_);
    if (state_ != State::Uninitialized)
    {
        throw std::logic_error("StubPoseEngine::initialize() needs an uninitialized engine");
    }
    if (std::string error = validate_configuration(setup); !error.empty())
    {
        throw std::invalid_argument(error);
    }
    parse_stub_settings(setup.engine_settings, settings_);
    setup_ = setup;
    state_ = State::Initialized;
}

void StubPoseEngine::start()
{
    std::lock_guard lock(mutex_);
    if (state_ != State::Initialized)
    {
        throw std::logic_error("StubPoseEngine::start() needs an initialized engine");
    }
    state_ = State::Running;
    worker_ = std::thread(&StubPoseEngine::run_worker, this);
}

bool StubPoseEngine::admitting() const
{
    return state_ == State::Running && !stopping_;
}

bool StubPoseEngine::matches_setup(const EngineImageGroup& group) const
{
    if (group.images.size() != setup_.streams.size())
    {
        return false;
    }
    std::set<StreamId> seen;
    for (const EngineImage& image : group.images)
    {
        const auto stream = std::find_if(setup_.streams.begin(), setup_.streams.end(),
                                         [&](const StreamSetup& candidate) { return candidate.id == image.stream; });
        if (stream == setup_.streams.end() || !seen.insert(image.stream).second)
        {
            return false;
        }
        const HostImage* host = std::get_if<HostImage>(&image.storage);
        if (host == nullptr || host->layout != stream->layout || host->data == nullptr || !host->hold ||
            host->size_bytes < canonical_size_bytes(stream->layout).value_or(0))
        {
            return false;
        }
    }
    return true;
}

// A rejected or dropped group's holds are released after this returns, outside the lock.
EnqueueResult StubPoseEngine::on_image_group(EngineImageGroup group)
{
    std::lock_guard lock(mutex_);
    if (!admitting())
    {
        ++health_.not_running_rejections;
        return EnqueueResult::RejectedNotRunning;
    }
    if (!matches_setup(group))
    {
        ++health_.invalid_input_rejections;
        return EnqueueResult::RejectedInvalidInput;
    }
    if (images_pending_ >= settings_.image_capacity)
    {
        ++health_.dropped_image_groups;
        return EnqueueResult::DroppedBackpressure;
    }
    ++images_pending_;
    inputs_.emplace_back(std::move(group));
    input_ready_.notify_one();
    return EnqueueResult::Accepted;
}

EnqueueResult StubPoseEngine::on_imu(const EngineImuSample& sample)
{
    std::lock_guard lock(mutex_);
    if (!admitting())
    {
        ++health_.not_running_rejections;
        return EnqueueResult::RejectedNotRunning;
    }
    if (setup_.input_mode != InputMode::StereoImu || !finite(sample.acceleration_mps2) ||
        !finite(sample.angular_velocity_radps))
    {
        ++health_.invalid_input_rejections;
        return EnqueueResult::RejectedInvalidInput;
    }
    if (imus_queued_ >= settings_.imu_capacity)
    {
        ++health_.dropped_imu_samples;
        if (!imu_gap_open_)
        {
            imu_gap_open_ = true;
            ++health_.imu_gaps;
        }
        return EnqueueResult::DroppedBackpressure;
    }
    if (imu_gap_open_)
    {
        inputs_.emplace_back(ImuGap{});
        imu_gap_open_ = false;
    }
    ++imus_queued_;
    inputs_.emplace_back(sample);
    input_ready_.notify_one();
    return EnqueueResult::Accepted;
}

void StubPoseEngine::finish_input()
{
    std::lock_guard lock(mutex_);
    if (state_ == State::Uninitialized || state_ == State::Initialized)
    {
        throw std::logic_error("StubPoseEngine::finish_input() needs a started engine");
    }
    if (state_ != State::Running)
    {
        return;
    }
    if (imu_gap_open_)
    {
        inputs_.emplace_back(ImuGap{});
        imu_gap_open_ = false;
    }
    state_ = State::Draining;
    input_ready_.notify_one();
}

bool StubPoseEngine::input_drained() const
{
    std::lock_guard lock(mutex_);
    const bool finished = state_ == State::Draining || state_ == State::Stopped;
    return finished && inputs_.empty() && !worker_busy_ && !input_discarded_;
}

bool StubPoseEngine::try_get_next_pose(EnginePose& out)
{
    std::lock_guard lock(mutex_);
    if (poses_.empty())
    {
        return false;
    }
    out = poses_.front();
    poses_.pop_front();
    pose_space_.notify_one();
    return true;
}

EngineHealth StubPoseEngine::health() const
{
    std::lock_guard lock(mutex_);
    return health_;
}

void StubPoseEngine::stop()
{
    {
        std::lock_guard lock(mutex_);
        if (state_ == State::Stopped)
        {
            return;
        }
        stopping_ = true;
    }
    input_ready_.notify_all();
    pose_space_.notify_all();
    if (worker_.joinable())
    {
        worker_.join();
    }

    std::deque<Input> abandoned;
    {
        std::lock_guard lock(mutex_);
        input_discarded_ = input_discarded_ || !inputs_.empty();
        abandoned.swap(inputs_);
        images_pending_ = 0;
        imus_queued_ = 0;
        state_ = State::Stopped;
    }
    // The worker has exited, so `abandoned` releases storage that nothing reads.
}

void StubPoseEngine::run_worker()
{
    std::unique_lock lock(mutex_);
    while (true)
    {
        input_ready_.wait(lock, [this] { return stopping_ || !inputs_.empty(); });
        if (stopping_)
        {
            return;
        }
        Input input = std::move(inputs_.front());
        inputs_.pop_front();
        worker_busy_ = true;
        const bool discard = health_.fatal_error;
        const bool is_image = std::holds_alternative<EngineImageGroup>(input);
        const bool is_imu = std::holds_alternative<EngineImuSample>(input);
        lock.unlock();

        std::optional<EnginePose> pose;
        if (!discard && is_image)
        {
            const EngineImageGroup& group = std::get<EngineImageGroup>(input);
            for (const EngineImage& image : group.images)
            {
                pixel_checksum_ += read_pixels(std::get<HostImage>(image.storage));
            }
            Transform placeholder;
            placeholder.translation_m = { static_cast<double>(groups_processed_), static_cast<double>(imu_processed_),
                                          0.0 };
            pose = EnginePose{ group.group_id, group.anchor, placeholder,
                               groups_processed_ < settings_.initializing_groups ? TrackingState::Initializing :
                                                                                   TrackingState::Tracking };
            ++groups_processed_;
        }
        else if (!discard && is_imu)
        {
            ++imu_processed_;
        }
        // Releases the group's storage after its last read, before the pose is published.
        input = ImuGap{};

        lock.lock();
        if (discard)
        {
            input_discarded_ = true;
        }
        else if (!is_image && !is_imu)
        {
            health_.fatal_error = true;
            health_.fatal_reason = "IMU samples were dropped and the stub adapter has no gap recovery";
        }
        if (is_imu)
        {
            --imus_queued_;
        }
        if (pose)
        {
            if (poses_.size() >= settings_.pose_capacity)
            {
                ++health_.output_stalls;
                pose_space_.wait(lock, [this] { return stopping_ || poses_.size() < settings_.pose_capacity; });
            }
            if (stopping_)
            {
                input_discarded_ = true;
                worker_busy_ = false;
                return;
            }
            poses_.push_back(*pose);
        }
        if (is_image)
        {
            --images_pending_;
        }
        worker_busy_ = false;
    }
}

} // namespace pose_provider
