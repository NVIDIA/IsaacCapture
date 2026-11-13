// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/pose_provider/pose_provider.hpp"

#include <log_bridge/logger.hpp>
#include <pose_undistort/undistort.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace pose_provider
{

namespace
{

constexpr const char* kLoggerName = "isaaccapture.plugins.pose_provider.PoseProvider";

// The engine has no wake handle, so an initialized engine is polled at this period.
constexpr std::chrono::milliseconds kEnginePollPeriod{ 5 };

// The one built-in camera profile, for any device of this backend; resolve_pose_setup() applies its rules.
constexpr const char* kProfileBackend = "oak";
constexpr const char* kBodyFrame = "left_optical";

// Accepts float-rounded rotations, which are renormalized; anything further off is not a rotation.
constexpr double kUnitTolerance = 1e-3;

const char* component_name(ProviderComponent component)
{
    switch (component)
    {
    case ProviderComponent::Provider:
        return "Provider";
    case ProviderComponent::Setup:
        return "Setup";
    case ProviderComponent::Engine:
        return "Engine";
    case ProviderComponent::CameraInput:
        return "CameraInput";
    }
    return "unknown";
}

std::string ms_text(std::chrono::milliseconds duration)
{
    return std::to_string(duration.count()) + " ms";
}

void check_timeout(const char* name, std::chrono::milliseconds timeout)
{
    if (timeout <= std::chrono::milliseconds::zero() || timeout > kMaxProviderTimeout)
    {
        throw std::invalid_argument(std::string(name) + " must be in (0, " + ms_text(kMaxProviderTimeout) + "], not " +
                                    ms_text(timeout));
    }
}

std::string health_text(const EngineHealth& health)
{
    return "dropped image groups " + std::to_string(health.dropped_image_groups) + ", dropped IMU samples " +
           std::to_string(health.dropped_imu_samples) + ", IMU gaps " + std::to_string(health.imu_gaps) +
           ", invalid-input rejections " + std::to_string(health.invalid_input_rejections) + ", not-running rejections " +
           std::to_string(health.not_running_rejections) + ", output stalls " + std::to_string(health.output_stalls);
}

using Vector = std::array<double, 3>;
using Quaternion = std::array<double, 4>; // x, y, z, w

Quaternion multiply(const Quaternion& a, const Quaternion& b)
{
    return { a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
             a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0], a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
             a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2] };
}

// Rotates v by the unit quaternion q.
Vector rotate(const Quaternion& q, const Vector& v)
{
    const Vector t{ 2 * (q[1] * v[2] - q[2] * v[1]), 2 * (q[2] * v[0] - q[0] * v[2]), 2 * (q[0] * v[1] - q[1] * v[0]) };
    return { v[0] + q[3] * t[0] + q[1] * t[2] - q[2] * t[1], v[1] + q[3] * t[1] + q[2] * t[0] - q[0] * t[2],
             v[2] + q[3] * t[2] + q[0] * t[1] - q[1] * t[0] };
}

// T_a_c = T_a_b * T_b_c.
Transform compose(const Transform& ab, const Transform& bc)
{
    const Vector moved = rotate(ab.rotation_xyzw, bc.translation_m);
    return Transform{ { ab.translation_m[0] + moved[0], ab.translation_m[1] + moved[1], ab.translation_m[2] + moved[2] },
                      multiply(ab.rotation_xyzw, bc.rotation_xyzw) };
}

Transform inverse(const Transform& ab)
{
    const Quaternion& q = ab.rotation_xyzw;
    const Quaternion conjugate{ -q[0], -q[1], -q[2], q[3] };
    const Vector moved = rotate(conjugate, ab.translation_m);
    return Transform{ { -moved[0], -moved[1], -moved[2] }, conjugate };
}

// @p pose with a unit rotation, or nullopt if it is not a rigid transform.
std::optional<Transform> rigid(const Transform& pose)
{
    double norm_squared = 0.0;
    for (double v : pose.rotation_xyzw)
    {
        norm_squared += v * v;
    }
    const double norm = std::sqrt(norm_squared);
    const bool finite =
        std::all_of(pose.translation_m.begin(), pose.translation_m.end(), [](double v) { return std::isfinite(v); });
    if (!finite || !std::isfinite(norm) || std::abs(norm - 1.0) > kUnitTolerance)
    {
        return std::nullopt;
    }
    Transform unit = pose;
    for (double& v : unit.rotation_xyzw)
    {
        v /= norm;
    }
    return unit;
}

// A stream that meets the profile's per-stream rules, with its pose in the rig renormalized.
struct CheckedStream
{
    std::string name; // "<role> eye '<stream_id>'", which starts every reason about the stream.
    const StreamDescription& description;
    Transform T_rig_camera;
};

// Throws std::invalid_argument naming the stream when it breaks one of the profile's per-stream rules.
CheckedStream check_stream(const char* role, const StreamDescription& stream)
{
    const std::string name = std::string(role) + " eye '" + stream.stream_id + "'";
    const auto reject = [&](const char* what) { return std::invalid_argument(name + " " + what); };
    if (stream.format != PixelFormat::Gray8)
    {
        throw reject("is not Gray8");
    }
    if (stream.shutter != Shutter::Global)
    {
        throw reject("has no global shutter");
    }
    if (stream.timed_event == TimedEvent::Unknown)
    {
        throw reject("does not state which event its sample times mark");
    }
    if (!stream.rate_hz || !std::isfinite(*stream.rate_hz) || *stream.rate_hz <= 0.0)
    {
        throw reject("has no positive nominal rate");
    }
    if (!stream.intrinsics)
    {
        throw reject("has no intrinsics");
    }
    if (!stream.T_rig_camera)
    {
        throw reject("has no pose in the rig");
    }
    const std::optional<Transform> pose = rigid(*stream.T_rig_camera);
    if (!pose)
    {
        throw reject("has a pose in the rig that is not a rigid transform");
    }
    return CheckedStream{ name, stream, *pose };
}

// The delivered lens, as given, placed at @p T_body_camera.
CameraCalibration lens(const Intrinsics& intrinsics, const Transform& T_body_camera)
{
    return CameraCalibration{ intrinsics.fx,         intrinsics.fy,           intrinsics.cx, intrinsics.cy,
                              intrinsics.distortion, intrinsics.coefficients, T_body_camera };
}

// Field of view in degrees across @p size pixels, from pixel edge to pixel edge.
double field_of_view_deg(double principal_point, double focal_length, uint32_t size)
{
    return (std::atan((principal_point + 0.5) / focal_length) +
            std::atan((static_cast<double>(size) - 0.5 - principal_point) / focal_length)) *
           180.0 / std::numbers::pi;
}

} // namespace

PoseProvider::PoseProvider(ProviderConfig config,
                           std::unique_ptr<IPoseEngine> engine,
                           std::unique_ptr<ICameraInput> camera_input)
    : config_(config), camera_(std::move(camera_input)), engine_(std::move(engine))
{
    if (!engine_)
    {
        throw std::invalid_argument("PoseProvider needs an engine");
    }
    check_timeout("startup_timeout", config_.startup_timeout);
    check_timeout("drain_timeout", config_.drain_timeout);
    startup_deadline_ = Clock::now() + config_.startup_timeout;
    status_.startup_timeout = config_.startup_timeout;
    status_.drain_timeout = config_.drain_timeout;
}

PoseProvider::~PoseProvider()
{
    try
    {
        abort();
    }
    catch (...)
    {
    }
}

void PoseProvider::update()
{
    try
    {
        ProviderState before = status_.state;
        while (status_.state != ProviderState::Finished)
        {
            step();
            if (status_.state == before)
            {
                break;
            }
            before = status_.state;
        }
    }
    catch (const std::exception& e)
    {
        fail(ProviderComponent::Provider, std::string("internal error: ") + e.what());
    }
    catch (...)
    {
        fail(ProviderComponent::Provider, "internal error");
    }
}

void PoseProvider::request_start()
{
    start_requested_ = true;
}

void PoseProvider::request_finish()
{
    finish_requested_ = true;
}

void PoseProvider::abort()
{
    if (status_.state != ProviderState::Finished)
    {
        fail(ProviderComponent::Provider, "aborted");
    }
}

bool PoseProvider::finished() const
{
    return status_.state == ProviderState::Finished;
}

ProviderStatus PoseProvider::status() const
{
    return status_;
}

int PoseProvider::wait_fd() const
{
    return camera_ ? camera_->wait_fd() : -1;
}

std::chrono::milliseconds PoseProvider::wait_timeout() const
{
    switch (status_.state)
    {
    case ProviderState::Preparing:
        // Nothing to poll before the engine is initialized. Rounded up, so a wake before the deadline cannot spin.
        return std::max(std::chrono::milliseconds::zero(),
                        std::chrono::ceil<std::chrono::milliseconds>(startup_deadline_ - Clock::now()));
    case ProviderState::Ready:
    case ProviderState::Running:
    case ProviderState::Draining:
        return kEnginePollPeriod;
    case ProviderState::Finished:
        break;
    }
    return std::chrono::milliseconds::zero();
}

ResolvedPoseSetup PoseProvider::resolve_pose_setup(const SourceDescription& source,
                                                   const EngineCapabilities& capabilities,
                                                   const EngineIdentity& identity)
{
    if (source.backend != kProfileBackend)
    {
        throw std::invalid_argument("no camera profile for this backend");
    }
    // Generic validation, before the engine's own and before the undistortion search.
    if (!capabilities.input_modes.contains(InputMode::Stereo) || !capabilities.memory_kinds.contains(MemoryKind::Host) ||
        !capabilities.pixel_formats.contains(PixelFormat::Gray8) ||
        !capabilities.distortion_models.contains(DistortionModel::None))
    {
        throw std::invalid_argument("the engine does not take Stereo host Gray8 images without distortion");
    }

    const std::array<CheckedStream, 2> checked{ check_stream("left", source.left), check_stream("right", source.right) };

    ResolvedPoseSetup setup;
    setup.input_mode = InputMode::Stereo;
    setup.memory_kind = MemoryKind::Host;
    setup.body_frame = kBodyFrame;
    setup.engine_identity = identity;
    // Body is the left eye's optical frame, wherever the rig's own reference is.
    const Transform T_body_rig = inverse(checked[0].T_rig_camera);
    for (uint32_t i = 0; i < checked.size(); ++i)
    {
        const StreamDescription& delivered = checked[i].description;
        StreamSetup stream;
        stream.id = StreamId{ i };
        stream.layout = ImageLayout{ delivered.width, delivered.height, PixelFormat::Gray8, delivered.width };
        stream.rate_hz = *delivered.rate_hz;
        try
        {
            stream.calibration =
                widest_undistorted_calibration(lens(*delivered.intrinsics, compose(T_body_rig, checked[i].T_rig_camera)),
                                               delivered.width, delivered.height);
        }
        catch (const std::invalid_argument& e)
        {
            throw std::invalid_argument(checked[i].name + ": " + e.what());
        }
        setup.streams.push_back(std::move(stream));
    }
    return setup;
}

void PoseProvider::step()
{
    switch (status_.state)
    {
    case ProviderState::Preparing:
        prepare();
        break;
    case ProviderState::Ready:
        await_start();
        break;
    case ProviderState::Running:
        run();
        break;
    case ProviderState::Draining:
        drain();
        break;
    case ProviderState::Finished:
        break;
    }
}

void PoseProvider::prepare()
{
    if (finish_requested_)
    {
        finish_run();
        return;
    }
    if (!take_camera_events())
    {
        return;
    }
    if (!source_)
    {
        if (Clock::now() >= startup_deadline_)
        {
            fail(ProviderComponent::CameraInput,
                 "no camera source description within the startup timeout (" + ms_text(config_.startup_timeout) + ")");
        }
        return;
    }

    EngineCapabilities capabilities;
    EngineIdentity identity;
    if (!call_engine("capabilities", [&] { capabilities = engine_->capabilities(); }) ||
        !call_engine("identity", [&] { identity = engine_->identity(); }))
    {
        return;
    }
    std::optional<ResolvedPoseSetup> setup;
    try
    {
        setup = resolve_pose_setup(*source_, capabilities, identity);
    }
    catch (const std::invalid_argument& e)
    {
        const std::string named = source_->backend.empty() ?
                                      "streams '" + source_->left.stream_id + "' and '" + source_->right.stream_id + "'" :
                                      source_->backend + " device '" + source_->device_id + "'";
        fail(ProviderComponent::Setup, named + ": " + e.what());
        return;
    }

    std::string rejection;
    if (!call_engine("validate_configuration", [&] { rejection = engine_->validate_configuration(*setup); }))
    {
        return;
    }
    if (!rejection.empty())
    {
        fail(ProviderComponent::Setup, "the engine rejected the setup: " + rejection);
        return;
    }
    if (!call_engine("initialize", [&] { engine_->initialize(*setup); }))
    {
        return;
    }
    setup_ = std::move(setup);
    enter(ProviderState::Ready);
}

void PoseProvider::await_start()
{
    if (!check_health())
    {
        return;
    }
    // A finish takes priority over a pending start, and never needs finish_input() before start().
    if (finish_requested_)
    {
        finish_run();
        return;
    }
    if (!take_camera_events())
    {
        return;
    }
    if (start_requested_)
    {
        if (call_engine("start", [&] { engine_->start(); }))
        {
            enter(ProviderState::Running);
        }
        return;
    }
    if (Clock::now() >= startup_deadline_)
    {
        fail(ProviderComponent::Provider,
             "not started within the startup timeout (" + ms_text(config_.startup_timeout) + ")");
    }
}

void PoseProvider::run()
{
    // Checked first, so that no new input is admitted once a finish is requested.
    if (finish_requested_)
    {
        drain_deadline_ = Clock::now() + config_.drain_timeout;
        enter(ProviderState::Draining);
        return;
    }
    if (!take_camera_events() || !take_estimates())
    {
        return;
    }
    check_health();
}

void PoseProvider::drain()
{
    if (!input_finished_)
    {
        if (!call_engine("finish_input", [&] { engine_->finish_input(); }))
        {
            return;
        }
        input_finished_ = true;
    }

    // Read before polling, so that no estimate can appear after a drained result.
    bool drained = false;
    if (!call_engine("input_drained", [&] { drained = engine_->input_drained(); }))
    {
        return;
    }
    if (!take_estimates() || !check_health())
    {
        return;
    }
    if (drained)
    {
        finish_run();
        return;
    }
    take_camera_events(); // A finish is requested, so no camera event fails the run here.
    if (Clock::now() >= drain_deadline_)
    {
        fail(ProviderComponent::Engine,
             "the engine did not drain within the drain timeout (" + ms_text(config_.drain_timeout) + ")");
    }
}

void PoseProvider::enter(ProviderState state)
{
    status_.state = state;
    auto logger = isaaccapture::Logger::get(kLoggerName);
    switch (state)
    {
    case ProviderState::Ready:
    {
        // Per eye: the applied focal length over the delivered one, and the applied field of view.
        const auto factor = [](const StreamDescription& eye, const StreamSetup& stream)
        { return stream.calibration.fx / eye.intrinsics->fx; };
        const auto horizontal = [](const StreamSetup& stream)
        { return field_of_view_deg(stream.calibration.cx, stream.calibration.fx, stream.layout.width); };
        const auto vertical = [](const StreamSetup& stream)
        { return field_of_view_deg(stream.calibration.cy, stream.calibration.fy, stream.layout.height); };
        const StreamSetup& left = setup_->streams[0];
        const StreamSetup& right = setup_->streams[1];
        logger->info(
            "Ready: {} device '{}', session '{}', Body {}; left eye '{}' as stream {}: undistortion factor {:.3f}, "
            "field of view {:.1f} x {:.1f} deg; right eye '{}' as stream {}: undistortion factor {:.3f}, field of "
            "view {:.1f} x {:.1f} deg",
            source_->backend, source_->device_id, source_->session_id, setup_->body_frame, source_->left.stream_id,
            left.id.value, factor(source_->left, left), horizontal(left), vertical(left), source_->right.stream_id,
            right.id.value, factor(source_->right, right), horizontal(right), vertical(right));
        break;
    }
    case ProviderState::Running:
        logger->info("Running");
        break;
    case ProviderState::Draining:
        logger->info("Draining: finishing input, then waiting up to {} for the engine", ms_text(config_.drain_timeout));
        break;
    case ProviderState::Preparing:
    case ProviderState::Finished:
        break;
    }
}

bool PoseProvider::take_camera_events()
{
    // Polling until poll() returns false leaves wait_fd() unreadable until the next event arrives.
    for (;;)
    {
        CameraEvent event;
        if (!camera_ || !camera_->poll(event))
        {
            return true;
        }
        if (auto* description = std::get_if<SourceDescription>(&event))
        {
            source_ = std::move(*description);
        }
        else if (std::holds_alternative<InputFailed>(event) || std::holds_alternative<InputEnded>(event))
        {
            const auto* failure = std::get_if<InputFailed>(&event);
            const std::string reason = failure ? "the camera input failed: " + failure->reason : "the camera input ended";
            // A source stopped by the same signal as this run must not fail the finish under way.
            if (!finish_requested_)
            {
                fail(ProviderComponent::CameraInput, reason);
                return false;
            }
            isaaccapture::Logger::get(kLoggerName)->info("After the finish request, {}", reason);
        }
        // Nothing admits frames: a StereoFrame is released unread as `event` goes out of scope.
    }
}

bool PoseProvider::take_estimates()
{
    for (;;)
    {
        EnginePose pose;
        bool taken = false;
        if (!call_engine("try_get_next_pose", [&] { taken = engine_->try_get_next_pose(pose); }))
        {
            return false;
        }
        if (!taken)
        {
            return true;
        }
        // No image group is admitted yet, so every estimate breaks the accepted-group order.
        fail(ProviderComponent::Engine,
             "estimate for image group " + std::to_string(pose.group_id) + ", which was never accepted");
        return false;
    }
}

bool PoseProvider::check_health()
{
    EngineHealth health;
    if (!call_engine("health", [&] { health = engine_->health(); }))
    {
        return false;
    }
    status_.engine_health = health;
    if (health.fatal_error)
    {
        fail(ProviderComponent::Engine, "engine fatal error: " + health.fatal_reason);
        return false;
    }
    return true;
}

bool PoseProvider::call_engine(const char* call, const std::function<void()>& work)
{
    try
    {
        work();
        return true;
    }
    catch (const std::exception& e)
    {
        fail(ProviderComponent::Engine, std::string(call) + ": " + e.what());
    }
    catch (...)
    {
        fail(ProviderComponent::Engine, std::string(call) + ": unknown exception");
    }
    return false;
}

void PoseProvider::fail(ProviderComponent component, std::string reason)
{
    if (status_.state == ProviderState::Finished)
    {
        return;
    }
    record_failure(component, std::move(reason));
    finish_run();
}

void PoseProvider::record_failure(ProviderComponent component, std::string reason)
{
    auto logger = isaaccapture::Logger::get(kLoggerName);
    if (status_.failed)
    {
        logger->warn("Later failure ({}), not recorded: {}", component_name(component), reason);
        return;
    }
    status_.failed = true;
    status_.failed_component = component;
    status_.failure_reason = std::move(reason);
    logger->error("Run failed ({}): {}", component_name(component), status_.failure_reason);
}

void PoseProvider::stop_engine()
{
    if (engine_stopped_)
    {
        return;
    }
    engine_stopped_ = true;
    try
    {
        engine_->stop();
    }
    catch (const std::exception& e)
    {
        record_failure(ProviderComponent::Engine, std::string("stop: ") + e.what());
    }
    catch (...)
    {
        record_failure(ProviderComponent::Engine, "stop: unknown exception");
    }
}

void PoseProvider::finish_run()
{
    // Engine calls here go straight to record_failure(): fail() would re-enter finish_run() before Finished is set.
    stop_engine();
    // Bounded blocking, like stop(): the input's destructor may wait briefly for its own thread.
    camera_.reset();
    // The contract does not promise health() before initialize(), and an engine that was never initialized has no
    // run to report on.
    if (setup_)
    {
        try
        {
            status_.engine_health = engine_->health();
            if (status_.engine_health.fatal_error && !status_.failed)
            {
                record_failure(ProviderComponent::Engine, "engine fatal error: " + status_.engine_health.fatal_reason);
            }
        }
        catch (const std::exception& e)
        {
            record_failure(ProviderComponent::Engine, std::string("health: ") + e.what());
        }
        catch (...)
        {
            record_failure(ProviderComponent::Engine, "health: unknown exception");
        }
    }

    status_.state = ProviderState::Finished;
    auto logger = isaaccapture::Logger::get(kLoggerName);
    if (status_.failed)
    {
        logger->error("Finished: failed ({}): {}; {}", component_name(*status_.failed_component),
                      status_.failure_reason, health_text(status_.engine_health));
    }
    else
    {
        logger->info("Finished; {}", health_text(status_.engine_health));
    }
}

} // namespace pose_provider
