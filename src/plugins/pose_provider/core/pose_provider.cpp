// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/pose_provider/pose_provider.hpp"

#include <deviceio_trackers/se3_tracker.hpp>
#include <flatbuffers/flatbuffers.h>
#include <log_bridge/logger.hpp>
#include <mcap/writer.hpp>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/os_time.hpp>
#include <plugin_utils/plugin_device_status_publisher.hpp>
#include <pose_undistort/undistort.hpp>
#include <pusherio/openxr_schema_push_channel.hpp>
#include <schema/se3_tracker_generated.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <variant>
#include <vector>

namespace pose_provider
{

namespace
{

constexpr const char* kLoggerName = "isaaccapture.plugins.pose_provider.PoseProvider";

// The engine has no wake handle, so an initialized engine is polled at this period.
constexpr std::chrono::milliseconds kEnginePollPeriod{ 5 };

// An update() at least this often lets the device status heartbeat keep the report within the reader's stale timeout.
constexpr auto kStatusHeartbeat = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::nanoseconds(plugin_utils::PluginDeviceStatusPublisher::HEARTBEAT_PERIOD_NS));

// The device in plugin.yaml that the device status reports.
constexpr const char* kDevicePath = "/se3_tracker/pose_provider";

static_assert(kMaxOutputCollectionBytes == XR_MAX_TENSOR_IDENTIFIER_SIZE - 1);

// The one built-in camera profile, for any device of this backend; resolve_pose_setup() applies its rules.
constexpr const char* kProfileBackend = "oak";
constexpr const char* kBodyFrame = "left_optical";

// Accepts float-rounded rotations, which are renormalized; anything further off is not a rotation.
constexpr double kUnitTolerance = 1e-3;

// Applied as ProviderStatus::prepared_image_groups. More than an engine's usual admission queue, so its backpressure,
// not this pool, normally drops a group.
constexpr std::size_t kPreparedGroups = 8;

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
    case ProviderComponent::InputPreparation:
        return "InputPreparation";
    case ProviderComponent::Output:
        return "Output";
    case ProviderComponent::RunLog:
        return "RunLog";
    }
    return "unknown";
}

// How the engine rejected an offered image group, or null when it accepted the group or dropped it, which its health
// counts.
const char* rejection_text(EnqueueResult result)
{
    switch (result)
    {
    case EnqueueResult::Accepted:
    case EnqueueResult::DroppedBackpressure:
        return nullptr;
    case EnqueueResult::RejectedInvalidInput:
        return "as invalid input";
    case EnqueueResult::RejectedNotRunning:
        return "as not running";
    }
    return "with an unknown result";
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

// "<role> eye '<stream_id>'", which starts every reason about the stream.
std::string stream_name(const char* role, const StreamDescription& stream)
{
    return std::string(role) + " eye '" + stream.stream_id + "'";
}

// A stream that meets the profile's per-stream rules, with its pose in the rig renormalized.
struct CheckedStream
{
    std::string name; // As stream_name() gives it.
    const StreamDescription& description;
    Transform T_rig_camera;
};

// Throws std::invalid_argument naming the stream when it breaks one of the profile's per-stream rules.
CheckedStream check_stream(const char* role, const StreamDescription& stream)
{
    const std::string name = stream_name(role, stream);
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

// The applied focal length over the delivered one.
double undistortion_factor(const StreamDescription& delivered, const StreamSetup& applied)
{
    return applied.calibration.fx / delivered.intrinsics->fx;
}

// The applied field of view in degrees, horizontal then vertical.
std::array<double, 2> field_of_view(const StreamSetup& applied)
{
    const CameraCalibration& c = applied.calibration;
    return { field_of_view_deg(c.cx, c.fx, applied.layout.width), field_of_view_deg(c.cy, c.fy, applied.layout.height) };
}

// One stream's preparation: the setup its frames must match and the map that undistorts them.
struct PreparedStream
{
    std::string name; // As stream_name() gives it.
    StreamId id;
    ImageLayout layout; // Prepared and packed, at the delivered size.
    UndistortMap map;
    std::size_t offset = 0; // Of the stream's pixels within a slot.
};

// Prepared storage for one image group, every stream packed. Held from preparation until the last copy of its hold
// goes, which the engine may release on any thread; the hold co-owns the slot.
struct Slot
{
    std::vector<std::byte> pixels;
    std::atomic<bool> held{ false };
};

using Slots = std::array<std::shared_ptr<Slot>, kPreparedGroups>;

// A slot with no hold, or null. The acquire load orders the engine's last reads of it before the next writes.
std::shared_ptr<Slot> free_slot(const Slots& slots)
{
    for (const std::shared_ptr<Slot>& slot : slots)
    {
        if (!slot->held.load(std::memory_order_acquire))
        {
            return slot;
        }
    }
    return nullptr;
}

// Captures missing before @p first when @p next is the next expected one, or 0 before the first frame. Throws
// std::invalid_argument when @p first goes back.
uint64_t missing_before(const std::optional<uint64_t>& next, uint64_t first, const char* what)
{
    if (!next)
    {
        return 0;
    }
    if (first < *next)
    {
        throw std::invalid_argument(std::string(what) + " at capture number " + std::to_string(first) +
                                    " does not follow the previous capture: expected " + std::to_string(*next) +
                                    " or later");
    }
    return first - *next;
}

// Throws std::invalid_argument naming the stream when @p image cannot be read as one of @p stream's frames.
void check_image(const PreparedStream& stream, const StreamImage& image)
{
    const ImageLayout& layout = image.layout;
    if (layout.format != PixelFormat::Gray8)
    {
        throw std::invalid_argument(stream.name + ": frame is not Gray8");
    }
    const std::optional<std::size_t> needed = canonical_size_bytes(layout);
    if (!needed || layout.width != stream.layout.width || layout.height != stream.layout.height)
    {
        throw std::invalid_argument(
            stream.name + ": frame is " + std::to_string(layout.width) + "x" + std::to_string(layout.height) +
            " with stride " + std::to_string(layout.stride_bytes) + ", not " + std::to_string(stream.layout.width) +
            "x" + std::to_string(stream.layout.height) + " with stride at least " + std::to_string(stream.layout.width));
    }
    const std::size_t readable = image.data != nullptr ? image.size_bytes : 0;
    if (readable < *needed)
    {
        throw std::invalid_argument(stream.name + ": frame has " + std::to_string(readable) +
                                    " readable bytes; its layout needs " + std::to_string(*needed));
    }
}

// The anchor stream's times. Throws std::invalid_argument naming the stream unless both are present, the device time
// is not negative, and both follow @p last.
AnchorTimes check_anchor(const std::string& name, const StreamImage& image, const std::optional<AnchorTimes>& last)
{
    if (!image.local_common_clock_ns || !image.raw_device_clock_ns)
    {
        throw std::invalid_argument(name + ": frame lacks a common-clock or device-clock time");
    }
    const AnchorTimes times{ *image.local_common_clock_ns, *image.raw_device_clock_ns };
    if (times.raw_device_clock_ns < 0)
    {
        throw std::invalid_argument(name + ": device-clock time " + std::to_string(times.raw_device_clock_ns) +
                                    " ns is negative");
    }
    if (last && (times.local_common_clock_ns <= last->local_common_clock_ns ||
                 times.raw_device_clock_ns <= last->raw_device_clock_ns))
    {
        throw std::invalid_argument(
            name + ": frame times (common " + std::to_string(times.local_common_clock_ns) + " ns, device " +
            std::to_string(times.raw_device_clock_ns) + " ns) do not both follow the previous frame's (" +
            std::to_string(last->local_common_clock_ns) + " ns, " + std::to_string(last->raw_device_clock_ns) + " ns)");
    }
    return times;
}

// The device status entry for @p status.
plugin_utils::PluginDeviceStatusEntry device_status(const ProviderStatus& status)
{
    plugin_utils::PluginDeviceStatusEntry entry{ .path = kDevicePath };
    if (status.failed)
    {
        entry.state = core::PluginDeviceState_FAILED;
        entry.reason = core::PluginDeviceReason_DEVICE_ERROR;
        entry.error = std::string(component_name(*status.failed_component)) + ": " + status.failure_reason;
    }
    else if (status.state == ProviderState::Preparing || status.state == ProviderState::Finished)
    {
        entry.state = core::PluginDeviceState_UNKNOWN;
        entry.reason = core::PluginDeviceReason_NO_CURRENT_DATA;
    }
    else
    {
        entry.state = core::PluginDeviceState_CONNECTED;
        entry.reason = core::PluginDeviceReason_NONE;
    }
    return entry;
}

// The plugin's own session: schema push only, on an OpenXR session that does not wait for a headset system.
class OpenXRPoseSession final : public core::IPluginSession
{
public:
    OpenXRPoseSession()
        : session_("PoseProviderPlugin",
                   core::OpenXRSchemaPushChannel::get_required_extensions(),
                   /*wait_for_system=*/false)
    {
    }

    core::IPluginPullChannel& pull_channel() override
    {
        throw std::logic_error("The Pose Provider session has no pull channel");
    }

    std::unique_ptr<core::ISchemaPushChannel> create_schema_push_channel(const core::SchemaPusherConfig& config) override
    {
        return std::make_unique<core::OpenXRSchemaPushChannel>(session_.get_handles(), config);
    }

    std::unique_ptr<core::IHandTrackingPushChannel> create_hand_tracking_push_channel(XrHandEXT) override
    {
        throw std::logic_error("The Pose Provider session has no hand tracking push");
    }

private:
    core::OpenXRSession session_;
};

std::unique_ptr<core::IPluginSession> open_plugin_session(const core::PluginSessionFactory& session_factory)
{
    if (!session_factory)
    {
        return std::make_unique<OpenXRPoseSession>();
    }
    auto session = session_factory(core::PluginSessionRequirements{ .schema_push = true }, {});
    if (!session)
    {
        throw std::runtime_error("the plugin session factory returned no session");
    }
    return session;
}

// The run log's channels, one JSON document per message.
constexpr const char* kSetupTopic = "setup";
constexpr const char* kFrameTopic = "frame";
constexpr const char* kAdmissionTopic = "admission";
constexpr const char* kLossTopic = "loss";
constexpr const char* kEstimateTopic = "estimate";
constexpr const char* kSummaryTopic = "summary";

const char* name(InputMode mode)
{
    switch (mode)
    {
    case InputMode::Stereo:
        return "Stereo";
    case InputMode::StereoImu:
        return "StereoImu";
    }
    return "unknown";
}

const char* name(MemoryKind kind)
{
    switch (kind)
    {
    case MemoryKind::Host:
        return "Host";
    }
    return "unknown";
}

const char* name(PixelFormat format)
{
    switch (format)
    {
    case PixelFormat::Gray8:
        return "Gray8";
    case PixelFormat::BGR888:
        return "BGR888";
    case PixelFormat::RGBA8:
        return "RGBA8";
    case PixelFormat::NV12:
        return "NV12";
    }
    return "unknown";
}

const char* name(DistortionModel model)
{
    switch (model)
    {
    case DistortionModel::None:
        return "None";
    case DistortionModel::RadTan5:
        return "RadTan5";
    case DistortionModel::Rational8:
        return "Rational8";
    case DistortionModel::ThinPrism12:
        return "ThinPrism12";
    case DistortionModel::Tilted14:
        return "Tilted14";
    }
    return "unknown";
}

const char* name(Shutter shutter)
{
    switch (shutter)
    {
    case Shutter::Unknown:
        return "Unknown";
    case Shutter::Global:
        return "Global";
    case Shutter::Rolling:
        return "Rolling";
    }
    return "unknown";
}

const char* name(TimedEvent event)
{
    switch (event)
    {
    case TimedEvent::Unknown:
        return "Unknown";
    case TimedEvent::ExposureStart:
        return "ExposureStart";
    case TimedEvent::ExposureMidpoint:
        return "ExposureMidpoint";
    case TimedEvent::FrameEnd:
        return "FrameEnd";
    }
    return "unknown";
}

const char* name(TrackingState state)
{
    switch (state)
    {
    case TrackingState::Initializing:
        return "Initializing";
    case TrackingState::Tracking:
        return "Tracking";
    case TrackingState::Lost:
        return "Lost";
    case TrackingState::Relocalizing:
        return "Relocalizing";
    }
    return "unknown";
}

const char* name(EnqueueResult result)
{
    switch (result)
    {
    case EnqueueResult::Accepted:
        return "Accepted";
    case EnqueueResult::DroppedBackpressure:
        return "DroppedBackpressure";
    case EnqueueResult::RejectedInvalidInput:
        return "RejectedInvalidInput";
    case EnqueueResult::RejectedNotRunning:
        return "RejectedNotRunning";
    }
    return "unknown";
}

// JSON is built by appending to one string. separate() puts the comma before every member or element but the first.
void separate(std::string& out)
{
    if (!out.empty() && out.back() != '{' && out.back() != '[' && out.back() != ':')
    {
        out += ',';
    }
}

void append(std::string& out, std::string_view text)
{
    out += '"';
    for (const char c : text)
    {
        if (c == '"' || c == '\\')
        {
            out += '\\';
            out += c;
        }
        else if (static_cast<unsigned char>(c) < 0x20)
        {
            std::array<char, 7> escaped{};
            std::snprintf(escaped.data(), escaped.size(), "\\u%04x", static_cast<unsigned>(c));
            out += escaped.data();
        }
        else
        {
            out += c;
        }
    }
    out += '"';
}

// Without it, a string literal would convert to bool.
void append(std::string& out, const char* text)
{
    append(out, std::string_view(text));
}

void append(std::string& out, bool value)
{
    out += value ? "true" : "false";
}

void append(std::string& out, std::nullopt_t)
{
    out += "null";
}

template <typename Number>
requires std::integral<Number> || std::floating_point<Number>
void append(std::string& out, Number value)
{
    if constexpr (std::floating_point<Number>)
    {
        if (!std::isfinite(value))
        {
            out += "null";
            return;
        }
    }
    std::array<char, 32> text{};
    char* end = std::to_chars(text.data(), text.data() + text.size(), value).ptr;
    out.append(text.data(), end);
}

template <typename Enum>
requires std::is_enum_v<Enum>
void append(std::string& out, Enum value)
{
    append(out, name(value));
}

template <typename T>
void append(std::string& out, const std::optional<T>& value)
{
    if (value)
    {
        append(out, *value);
        return;
    }
    out += "null";
}

template <typename T>
void field(std::string& out, std::string_view key, const T& value)
{
    separate(out);
    append(out, key);
    out += ':';
    append(out, value);
}

template <typename Values>
void field_array(std::string& out, std::string_view key, const Values& values)
{
    separate(out);
    append(out, key);
    out += ":[";
    for (const auto& value : values)
    {
        separate(out);
        append(out, value);
    }
    out += ']';
}

// Opens a member named @p key, or an array element when @p key is empty, with @p bracket.
void open_value(std::string& out, std::string_view key, char bracket)
{
    separate(out);
    if (!key.empty())
    {
        append(out, key);
        out += ':';
    }
    out += bracket;
}

void transform_field(std::string& out, std::string_view key, const Transform& transform)
{
    open_value(out, key, '{');
    field_array(out, "translation_m", transform.translation_m);
    field_array(out, "rotation_xyzw", transform.rotation_xyzw);
    out += '}';
}

// A pinhole lens: focal lengths, principal point and distortion.
void lens_fields(std::string& out,
                 double fx,
                 double fy,
                 double cx,
                 double cy,
                 DistortionModel model,
                 const std::vector<double>& coefficients)
{
    field(out, "fx", fx);
    field(out, "fy", fy);
    field(out, "cx", cx);
    field(out, "cy", cy);
    field(out, "distortion", model);
    field_array(out, "coefficients", coefficients);
}

// A stream image's two sample times, each null when absent.
void times_field(std::string& out, std::string_view key, const StreamImage& image)
{
    open_value(out, key, '{');
    field(out, "local_common_clock_ns", image.local_common_clock_ns);
    field(out, "raw_device_clock_ns", image.raw_device_clock_ns);
    out += '}';
}

mcap::ChannelId add_channel(mcap::McapWriter& writer, const char* topic)
{
    mcap::Channel channel(topic, "json", 0);
    writer.addChannel(channel);
    return channel.id;
}

// The run log's file, which must not exist yet. The MCAP writer does not check its output, so each write is checked
// here and the first failure is kept; nothing is written after it. The MCAP writer writes small pieces, which stdio
// buffers.
class RunLogFile final : public mcap::IWritable
{
public:
    // Throws std::runtime_error naming @p path when it cannot be created.
    explicit RunLogFile(std::filesystem::path path) : path_(std::move(path)), file_(std::fopen(path_.c_str(), "wbx"))
    {
        if (file_ == nullptr)
        {
            throw std::runtime_error(reason("created", errno));
        }
    }

    ~RunLogFile() override
    {
        if (file_ != nullptr)
        {
            std::fclose(file_);
        }
    }

    RunLogFile(const RunLogFile&) = delete;
    RunLogFile& operator=(const RunLogFile&) = delete;

    void handleWrite(const std::byte* data, uint64_t size) override
    {
        size_ += size;
        if (error_.empty() && std::fwrite(data, 1, size, file_) != size)
        {
            fail(reason("written", errno));
        }
    }

    // After each full chunk.
    void flush() override
    {
        if (error_.empty() && std::fflush(file_) != 0)
        {
            fail(reason("written", errno));
        }
    }

    // Syncs and closes the file, so that a write the disk did not take is reported. The MCAP writer's close() calls it.
    void end() override
    {
        if (file_ == nullptr)
        {
            return;
        }
        if (error_.empty() && (std::fflush(file_) != 0 || ::fsync(::fileno(file_)) != 0))
        {
            fail(reason("written", errno));
        }
        if (std::fclose(std::exchange(file_, nullptr)) != 0)
        {
            fail(reason("closed", errno));
        }
    }

    uint64_t size() const override
    {
        return size_;
    }

    // Keeps @p reason unless a failure is already kept.
    void fail(std::string reason)
    {
        if (error_.empty())
        {
            error_ = std::move(reason);
        }
    }

    // The first failure; empty while there is none.
    const std::string& error() const
    {
        return error_;
    }

private:
    std::string reason(const char* what, int error) const
    {
        return "run log '" + path_.string() + "' could not be " + what + ": " + std::generic_category().message(error);
    }

    std::filesystem::path path_;
    std::FILE* file_;
    uint64_t size_ = 0;
    std::string error_;
};

} // namespace

// Prepares camera frames as engine image groups: checks each stream against the setup, undistorts it into
// provider-owned packed storage and releases the frame, and counts the captures lost on the way. The control thread
// makes every call. It shares lens() and stream_name() with setup resolution, which is why it is in this file; to move
// it to its own, pass each stream's name and delivered lens in instead.
class PoseInputAdapter
{
public:
    // Builds each stream's map from its delivered lens to its applied calibration, and the storage.
    PoseInputAdapter(const SourceDescription& source, const ResolvedPoseSetup& setup);

    // The frame as the next image group, or nullopt when the engine holds every slot, which counts a dropped frame.
    // Captures missing before it count as lost. The frame is released on every path. Throws std::invalid_argument as
    // check_image(), missing_before() and check_anchor() do.
    std::optional<EngineImageGroup> prepare(StereoFrame frame, ProviderStatus& status);

    // Counts the reported captures, and any missing before them, as lost. A loss before the first frame is not counted:
    // it may cover captures from before admission opened. Throws std::invalid_argument when it goes back.
    void lose(const FrameLoss& loss, ProviderStatus& status);

    // Every capture from this capture number on comes after the frames prepared so far; absent before the first frame.
    std::optional<uint64_t> next_sequence() const
    {
        return next_sequence_;
    }

private:
    std::array<PreparedStream, 2> streams_; // Left, the anchor, then right, as resolve_pose_setup() lists them.
    Slots slots_;
    uint64_t next_group_id_ = 1; // 0 is never used, so an estimate left at the default group_id breaks the order.
    std::optional<uint64_t> next_sequence_;
    std::optional<AnchorTimes> last_anchor_;
};

PoseInputAdapter::PoseInputAdapter(const SourceDescription& source, const ResolvedPoseSetup& setup)
{
    const std::array<const StreamDescription*, 2> delivered{ &source.left, &source.right };
    const std::array<const char*, 2> roles{ "left", "right" };
    std::size_t offset = 0;
    for (std::size_t i = 0; i < streams_.size(); ++i)
    {
        const StreamSetup& applied = setup.streams.at(i);
        PreparedStream& stream = streams_[i];
        stream.name = stream_name(roles[i], *delivered[i]);
        stream.id = applied.id;
        stream.layout = applied.layout;
        stream.map = make_undistort_map(lens(*delivered[i]->intrinsics, applied.calibration.T_body_camera),
                                        applied.calibration, applied.layout.width, applied.layout.height);
        stream.offset = offset;
        offset += stream.map.x.size();
    }
    for (std::shared_ptr<Slot>& slot : slots_)
    {
        slot = std::make_shared<Slot>();
        slot->pixels.resize(offset);
    }
}

std::optional<EngineImageGroup> PoseInputAdapter::prepare(StereoFrame frame, ProviderStatus& status)
{
    const std::array<const StreamImage*, 2> images{ &frame.left(), &frame.right() };
    for (std::size_t i = 0; i < streams_.size(); ++i)
    {
        check_image(streams_[i], *images[i]);
    }
    const uint64_t lost = missing_before(next_sequence_, frame.sequence(), "frame");
    const AnchorTimes anchor = check_anchor(streams_[0].name, frame.left(), last_anchor_);
    status.frames_lost += lost;
    next_sequence_ = frame.sequence() + 1;
    last_anchor_ = anchor;

    const std::shared_ptr<Slot> slot = free_slot(slots_);
    if (!slot)
    {
        ++status.frames_dropped;
        return std::nullopt;
    }
    for (std::size_t i = 0; i < streams_.size(); ++i)
    {
        remap_gray8(
            streams_[i].map, images[i]->data, images[i]->layout.stride_bytes, slot->pixels.data() + streams_[i].offset);
    }
    frame.release();
    slot->held.store(true, std::memory_order_relaxed);
    const std::shared_ptr<const void> hold(
        slot->pixels.data(), [slot](const void*) { slot->held.store(false, std::memory_order_release); });

    EngineImageGroup group;
    group.group_id = next_group_id_++;
    group.fusion_time_ns = anchor.local_common_clock_ns;
    group.anchor = anchor;
    group.images.reserve(streams_.size());
    for (const PreparedStream& stream : streams_)
    {
        group.images.push_back(EngineImage{
            stream.id, HostImage{ stream.layout, slot->pixels.data() + stream.offset, stream.map.x.size(), hold } });
    }
    return group;
}

void PoseInputAdapter::lose(const FrameLoss& loss, ProviderStatus& status)
{
    if (!next_sequence_)
    {
        return;
    }
    status.frames_lost += missing_before(next_sequence_, loss.first_sequence, "loss report") + loss.count;
    next_sequence_ = loss.first_sequence + loss.count;
}

// The run's plugin session and the plugin's device status in it. The control thread makes every call.
class OutputSession
{
public:
    // Opens the session and, with @p plugin_root_id, the device status collection. Throws when either cannot open.
    OutputSession(const core::PluginSessionFactory& session_factory, const std::string& plugin_root_id);

    // Opens an se3_tracker collection. Throws when it cannot open.
    std::unique_ptr<core::ISchemaPushChannel> open_pose_collection(const std::string& collection_id);

    // Publishes @p status when it changed or a heartbeat is due. A failure is logged and ends status publication, so
    // that status never changes the run.
    void publish_status(const ProviderStatus& status);

private:
    std::unique_ptr<core::IPluginSession> session_;
    std::unique_ptr<plugin_utils::PluginDeviceStatusPublisher> status_; // Destroyed before the session.
};

OutputSession::OutputSession(const core::PluginSessionFactory& session_factory, const std::string& plugin_root_id)
    : session_(open_plugin_session(session_factory))
{
    if (!plugin_root_id.empty())
    {
        status_ = std::make_unique<plugin_utils::PluginDeviceStatusPublisher>(*session_, plugin_root_id);
    }
}

std::unique_ptr<core::ISchemaPushChannel> OutputSession::open_pose_collection(const std::string& collection_id)
{
    return session_->create_schema_push_channel(
        core::SchemaPusherConfig{ .collection_id = collection_id,
                                  .max_flatbuffer_size = core::Se3Tracker::DEFAULT_MAX_FLATBUFFER_SIZE,
                                  .tensor_identifier = std::string(core::Se3Tracker::TENSOR_IDENTIFIER),
                                  .localized_name = "Pose Provider" });
}

void OutputSession::publish_status(const ProviderStatus& status)
{
    if (!status_)
    {
        return;
    }
    try
    {
        status_->publish_if_changed({ device_status(status) }, core::os_monotonic_now_ns());
    }
    catch (const std::exception& e)
    {
        isaaccapture::Logger::get(kLoggerName)->warn("Device status not published, and no longer published: {}", e.what());
        status_.reset();
    }
}

// Publishes the run's estimates on one se3_tracker collection, in SlamWorld. The control thread makes every call.
class PoseOutput
{
public:
    explicit PoseOutput(std::unique_ptr<core::ISchemaPushChannel> channel) : channel_(std::move(channel))
    {
    }

    // Publishes the estimate with its anchor's times, then records it in @p status. Only a Tracking estimate is valid;
    // any other carries an identity pose. Throws when the push fails, leaving @p status unchanged.
    void publish(const EnginePose& pose, ProviderStatus& status);

private:
    std::unique_ptr<core::ISchemaPushChannel> channel_;
};

void PoseOutput::publish(const EnginePose& pose, ProviderStatus& status)
{
    const bool valid = pose.tracking_state == TrackingState::Tracking;
    const Transform placed = valid ? pose.T_slamworld_body : Transform{};
    const std::array<double, 3>& t = placed.translation_m;
    const std::array<double, 4>& q = placed.rotation_xyzw;
    const core::Pose sample(core::Point(static_cast<float>(t[0]), static_cast<float>(t[1]), static_cast<float>(t[2])),
                            core::Quaternion(static_cast<float>(q[0]), static_cast<float>(q[1]),
                                             static_cast<float>(q[2]), static_cast<float>(q[3])));
    flatbuffers::FlatBufferBuilder builder(core::Se3Tracker::DEFAULT_MAX_FLATBUFFER_SIZE);
    builder.Finish(core::CreateSe3TrackerPose(builder, &sample, valid));
    channel_->push_buffer(builder.GetBufferPointer(), builder.GetSize(), pose.anchor.local_common_clock_ns,
                          pose.anchor.raw_device_clock_ns);
    status.latest_tracking_state = pose.tracking_state;
    if (valid)
    {
        status.last_valid_measurement_time_ns = pose.anchor.local_common_clock_ns;
    }
    status.last_publication_time_ns = core::os_monotonic_now_ns();
}

// The run log: the run's setup, input decisions, published estimates and summary, one JSON document per MCAP
// message. Records collect in the MCAP writer's chunk, which is written to the file when it fills and at the close.
// The control thread makes every call.
class RunLog
{
public:
    // Creates @p path, which must not exist yet. Throws std::runtime_error naming it when it cannot.
    explicit RunLog(std::filesystem::path path);

    void setup(const ProviderConfig& config,
               const SourceDescription& source,
               const ResolvedPoseSetup& setup,
               const EngineCapabilities& capabilities);

    // A frame taken while Running, after preparation released it: its capture number and both streams' times, the
    // captures counted lost just before it, and its prepared group, absent when the engine held all prepared storage.
    void frame(const StereoFrame& frame, uint64_t lost_before, const std::optional<EngineImageGroup>& group);

    void admission(uint64_t group_id, EnqueueResult result);

    // A loss report taken while Running, with the captures it added to the lost count.
    void loss(const FrameLoss& loss, uint64_t counted);

    void estimate(const EnginePose& pose, int64_t publication_time_ns);

    // Writes the summary and closes the file. Returns a failure take_error() has not returned yet.
    std::optional<std::string> finish(const ProviderStatus& status,
                                      bool finish_requested,
                                      std::optional<uint64_t> next_sequence,
                                      uint64_t frames_after_cutoff);

    // The first failure to write or close the file, the first time it is asked for.
    std::optional<std::string> take_error();

private:
    void write(mcap::ChannelId channel); // json_ as the next message.

    RunLogFile file_;
    mcap::McapWriter writer_; // After file_, which it writes to until it is destroyed.
    mcap::ChannelId setup_channel_ = 0;
    mcap::ChannelId frame_channel_ = 0;
    mcap::ChannelId admission_channel_ = 0;
    mcap::ChannelId loss_channel_ = 0;
    mcap::ChannelId estimate_channel_ = 0;
    mcap::ChannelId summary_channel_ = 0;
    uint32_t sequence_ = 0; // Across channels, so it gives the order of every record.
    std::string json_; // Reused for each record.
    uint64_t published_ = 0;
    std::optional<EnginePose> last_published_;
    bool error_taken_ = false;
};

RunLog::RunLog(std::filesystem::path path) : file_(std::move(path))
{
    mcap::McapWriterOptions options("pose_provider");
    options.compression = mcap::Compression::None;
    writer_.open(file_, options);
    file_.flush(); // So that a full disk fails here, not at the first chunk.
    if (!file_.error().empty())
    {
        throw std::runtime_error(file_.error());
    }
    setup_channel_ = add_channel(writer_, kSetupTopic);
    frame_channel_ = add_channel(writer_, kFrameTopic);
    admission_channel_ = add_channel(writer_, kAdmissionTopic);
    loss_channel_ = add_channel(writer_, kLossTopic);
    estimate_channel_ = add_channel(writer_, kEstimateTopic);
    summary_channel_ = add_channel(writer_, kSummaryTopic);
}

void RunLog::setup(const ProviderConfig& config,
                   const SourceDescription& source,
                   const ResolvedPoseSetup& setup,
                   const EngineCapabilities& capabilities)
{
    const std::array<const StreamDescription*, 2> delivered{ &source.left, &source.right };
    json_ = "{";
    open_value(json_, "source", '{');
    field(json_, "backend", source.backend);
    field(json_, "device_id", source.device_id);
    field(json_, "session_id", source.session_id);
    open_value(json_, "streams", '[');
    for (const StreamDescription* stream : delivered)
    {
        open_value(json_, "", '{');
        field(json_, "stream_id", stream->stream_id);
        field(json_, "width", stream->width);
        field(json_, "height", stream->height);
        field(json_, "format", stream->format);
        field(json_, "rate_hz", stream->rate_hz);
        field(json_, "shutter", stream->shutter);
        field(json_, "timed_event", stream->timed_event);
        const Intrinsics& lens = *stream->intrinsics;
        open_value(json_, "intrinsics", '{');
        lens_fields(json_, lens.fx, lens.fy, lens.cx, lens.cy, lens.distortion, lens.coefficients);
        json_ += '}';
        transform_field(json_, "T_rig_camera", *stream->T_rig_camera);
        json_ += '}';
    }
    json_ += "]}";

    open_value(json_, "setup", '{');
    field(json_, "input_mode", setup.input_mode);
    field(json_, "memory_kind", setup.memory_kind);
    field(json_, "body_frame", setup.body_frame);
    field(json_, "engine_settings", setup.engine_settings);
    open_value(json_, "streams", '[');
    for (std::size_t i = 0; i < setup.streams.size(); ++i)
    {
        const StreamSetup& stream = setup.streams[i];
        const CameraCalibration& calibration = stream.calibration;
        open_value(json_, "", '{');
        field(json_, "id", stream.id.value);
        field(json_, "width", stream.layout.width);
        field(json_, "height", stream.layout.height);
        field(json_, "format", stream.layout.format);
        field(json_, "stride_bytes", stream.layout.stride_bytes);
        field(json_, "rate_hz", stream.rate_hz);
        open_value(json_, "calibration", '{');
        lens_fields(json_, calibration.fx, calibration.fy, calibration.cx, calibration.cy, calibration.distortion,
                    calibration.coefficients);
        transform_field(json_, "T_body_camera", calibration.T_body_camera);
        json_ += '}';
        field(json_, "undistortion_factor", undistortion_factor(*delivered.at(i), stream));
        field_array(json_, "field_of_view_deg", field_of_view(stream));
        json_ += '}';
    }
    json_ += "]}";

    open_value(json_, "engine", '{');
    field(json_, "adapter", setup.engine_identity.adapter);
    field(json_, "adapter_version", setup.engine_identity.adapter_version);
    field(json_, "engine_library", setup.engine_identity.engine_library);
    field(json_, "engine_library_version", setup.engine_identity.engine_library_version);
    field_array(json_, "input_modes", capabilities.input_modes);
    field_array(json_, "memory_kinds", capabilities.memory_kinds);
    field_array(json_, "pixel_formats", capabilities.pixel_formats);
    field_array(json_, "distortion_models", capabilities.distortion_models);
    json_ += '}';

    field(json_, "prepared_image_groups", kPreparedGroups);
    field(json_, "output_collection", config.output_collection);
    field(json_, "startup_timeout_ms", config.startup_timeout.count());
    field(json_, "drain_timeout_ms", config.drain_timeout.count());
    json_ += '}';
    write(setup_channel_);
}

void RunLog::frame(const StereoFrame& frame, uint64_t lost_before, const std::optional<EngineImageGroup>& group)
{
    json_ = "{";
    field(json_, "sequence", frame.sequence());
    times_field(json_, "left", frame.left());
    times_field(json_, "right", frame.right());
    field(json_, "lost_before", lost_before);
    field(json_, "group_id", group ? std::optional(group->group_id) : std::nullopt);
    field(json_, "fusion_time_ns", group ? std::optional(group->fusion_time_ns) : std::nullopt);
    json_ += '}';
    write(frame_channel_);
}

void RunLog::admission(uint64_t group_id, EnqueueResult result)
{
    json_ = "{";
    field(json_, "group_id", group_id);
    field(json_, "result", result);
    json_ += '}';
    write(admission_channel_);
}

void RunLog::loss(const FrameLoss& loss, uint64_t counted)
{
    json_ = "{";
    field(json_, "first_sequence", loss.first_sequence);
    field(json_, "count", loss.count);
    field(json_, "counted", counted);
    json_ += '}';
    write(loss_channel_);
}

void RunLog::estimate(const EnginePose& pose, int64_t publication_time_ns)
{
    json_ = "{";
    field(json_, "group_id", pose.group_id);
    field(json_, "tracking_state", pose.tracking_state);
    field(json_, "local_common_clock_ns", pose.anchor.local_common_clock_ns);
    field(json_, "raw_device_clock_ns", pose.anchor.raw_device_clock_ns);
    field(json_, "publication_time_ns", publication_time_ns);
    json_ += '}';
    write(estimate_channel_);
    ++published_;
    last_published_ = pose;
}

std::optional<std::string> RunLog::finish(const ProviderStatus& status,
                                          bool finish_requested,
                                          std::optional<uint64_t> next_sequence,
                                          uint64_t frames_after_cutoff)
{
    json_ = "{";
    field(json_, "outcome", status.failed ? "failed" : "finished");
    field(json_, "finish_requested", finish_requested);
    field(json_, "failed_component",
          status.failed_component ? std::optional(component_name(*status.failed_component)) : std::nullopt);
    field(json_, "failure_reason", status.failed ? std::optional(status.failure_reason) : std::nullopt);
    field(json_, "published_estimates", published_);
    if (last_published_)
    {
        open_value(json_, "last_published", '{');
        field(json_, "group_id", last_published_->group_id);
        field(json_, "local_common_clock_ns", last_published_->anchor.local_common_clock_ns);
        field(json_, "raw_device_clock_ns", last_published_->anchor.raw_device_clock_ns);
        json_ += '}';
    }
    else
    {
        field(json_, "last_published", std::nullopt);
    }
    field(json_, "frames_lost", status.frames_lost);
    field(json_, "frames_dropped", status.frames_dropped);
    field(json_, "next_sequence", next_sequence);
    field(json_, "frames_after_cutoff", frames_after_cutoff);
    const EngineHealth& health = status.engine_health;
    open_value(json_, "engine_health", '{');
    field(json_, "dropped_image_groups", health.dropped_image_groups);
    field(json_, "dropped_imu_samples", health.dropped_imu_samples);
    field(json_, "imu_gaps", health.imu_gaps);
    field(json_, "invalid_input_rejections", health.invalid_input_rejections);
    field(json_, "not_running_rejections", health.not_running_rejections);
    field(json_, "output_stalls", health.output_stalls);
    field(json_, "fatal_error", health.fatal_error);
    field(json_, "fatal_reason", health.fatal_reason);
    json_ += "}}";
    write(summary_channel_);
    writer_.close();
    return take_error();
}

std::optional<std::string> RunLog::take_error()
{
    if (error_taken_ || file_.error().empty())
    {
        return std::nullopt;
    }
    error_taken_ = true;
    return file_.error();
}

void RunLog::write(mcap::ChannelId channel)
{
    mcap::Message message;
    message.channelId = channel;
    message.sequence = sequence_++;
    message.logTime = static_cast<mcap::Timestamp>(core::os_monotonic_now_ns());
    message.publishTime = message.logTime;
    message.data = reinterpret_cast<const std::byte*>(json_.data());
    message.dataSize = json_.size();
    const mcap::Status status = writer_.write(message);
    if (!status.ok())
    {
        file_.fail("run log message not written: " + status.message);
    }
}

template <typename Work>
bool PoseProvider::call_engine(const char* call, Work&& work)
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

PoseProvider::PoseProvider(ProviderConfig config,
                           std::unique_ptr<IPoseEngine> engine,
                           std::unique_ptr<ICameraInput> camera_input,
                           core::PluginSessionFactory session_factory)
    : config_(std::move(config)),
      session_factory_(std::move(session_factory)),
      camera_(std::move(camera_input)),
      engine_(std::move(engine))
{
    if (!engine_)
    {
        throw std::invalid_argument("PoseProvider needs an engine");
    }
    check_timeout("startup_timeout", config_.startup_timeout);
    check_timeout("drain_timeout", config_.drain_timeout);
    if (config_.output_collection.size() > kMaxOutputCollectionBytes)
    {
        throw std::invalid_argument("output_collection must be at most " + std::to_string(kMaxOutputCollectionBytes) +
                                    " bytes, not " + std::to_string(config_.output_collection.size()));
    }
    startup_deadline_ = Clock::now() + config_.startup_timeout;
    status_.startup_timeout = config_.startup_timeout;
    status_.drain_timeout = config_.drain_timeout;
    status_.prepared_image_groups = kPreparedGroups;
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
        // A run log that failed while the run went on.
        if (std::optional<std::string> error = run_log_ ? run_log_->take_error() : std::nullopt)
        {
            fail(ProviderComponent::RunLog, std::move(*error));
        }
        if (session_)
        {
            session_->publish_status(status_);
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
    return camera_ && !camera_ended_ ? camera_->wait_fd() : -1;
}

std::chrono::milliseconds PoseProvider::wait_timeout() const
{
    switch (status_.state)
    {
    case ProviderState::Preparing:
    {
        // Nothing to poll before the engine is initialized. Rounded up, so a wake before the deadline cannot spin.
        const auto until_deadline =
            std::max(std::chrono::milliseconds::zero(),
                     std::chrono::ceil<std::chrono::milliseconds>(startup_deadline_ - Clock::now()));
        const bool reports_status = session_ && !config_.plugin_root_id.empty();
        return reports_status ? std::min(until_deadline, kStatusHeartbeat) : until_deadline;
    }
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
    // Before the session, so that a session failure reaches the summary too.
    if (!run_log_ && config_.episode_dir)
    {
        try
        {
            run_log_ = std::make_unique<RunLog>(*config_.episode_dir / kRunLogFileName);
        }
        catch (const std::exception& e)
        {
            fail(ProviderComponent::RunLog, e.what());
            return;
        }
    }
    // Before the camera wait, so that the device status covers Preparing and a missing runtime fails at once.
    if (!session_ && !config_.output_collection.empty())
    {
        try
        {
            session_ = std::make_unique<OutputSession>(session_factory_, config_.plugin_root_id);
        }
        catch (const std::exception& e)
        {
            fail(ProviderComponent::Output,
                 std::string("the plugin session or the device status could not open: ") + e.what());
            return;
        }
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
    input_ = std::make_unique<PoseInputAdapter>(*source_, *setup_);
    if (session_)
    {
        try
        {
            output_ = std::make_unique<PoseOutput>(session_->open_pose_collection(config_.output_collection));
        }
        catch (const std::exception& e)
        {
            fail(ProviderComponent::Output,
                 "output collection '" + config_.output_collection + "' could not open: " + e.what());
            return;
        }
    }
    if (run_log_)
    {
        run_log_->setup(config_, *source_, *setup_, capabilities);
    }
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
    // Each frame taken while Running is offered or dropped in the update that takes it, so no group is pending at the
    // cutoff. A camera input that ends after the deadline fails without finish_input().
    const bool late = Clock::now() >= drain_deadline_;
    if (!input_finished_)
    {
        if (!detached_)
        {
            camera_->detach();
            detached_ = true;
        }
        take_camera_events(); // A finish is requested, so no camera event fails the run here.
        if (camera_ended_ && !late)
        {
            const std::optional<uint64_t> next = input_->next_sequence();
            const std::string cutoff = next ? "capture number " + std::to_string(*next) : std::string("the first frame");
            isaaccapture::Logger::get(kLoggerName)
                ->info("Camera input ended; cutoff before {}; {} later frames released unread", cutoff,
                       frames_after_cutoff_);
            if (!call_engine("finish_input", [&] { engine_->finish_input(); }))
            {
                return;
            }
            input_finished_ = true;
        }
    }

    // Read before polling, so that no estimate can appear after a drained result.
    bool drained = false;
    if (input_finished_ && !call_engine("input_drained", [&] { drained = engine_->input_drained(); }))
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
    if (late)
    {
        const std::string limit = " within the drain timeout (" + ms_text(config_.drain_timeout) + ")";
        if (input_finished_)
        {
            fail(ProviderComponent::Engine, "the engine did not drain" + limit);
        }
        else
        {
            fail(ProviderComponent::CameraInput, "the camera input did not end" + limit);
        }
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
        const StreamSetup& left = setup_->streams[0];
        const StreamSetup& right = setup_->streams[1];
        const std::array<double, 2> left_view = field_of_view(left);
        const std::array<double, 2> right_view = field_of_view(right);
        logger->info(
            "Ready: {} device '{}', session '{}', Body {}; left eye '{}' as stream {}: undistortion factor {:.3f}, "
            "field of view {:.1f} x {:.1f} deg; right eye '{}' as stream {}: undistortion factor {:.3f}, field of "
            "view {:.1f} x {:.1f} deg",
            source_->backend, source_->device_id, source_->session_id, setup_->body_frame, source_->left.stream_id,
            left.id.value, undistortion_factor(source_->left, left), left_view[0], left_view[1], source_->right.stream_id,
            right.id.value, undistortion_factor(source_->right, right), right_view[0], right_view[1]);
        break;
    }
    case ProviderState::Running:
        logger->info("Running");
        break;
    case ProviderState::Draining:
        logger->info(
            "Draining: up to {} for the camera input to end and the engine to drain", ms_text(config_.drain_timeout));
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
        else if (std::holds_alternative<StereoFrame>(event) || std::holds_alternative<FrameLoss>(event))
        {
            // Admission is open only while Running. Elsewhere a frame is released unread as `event` goes out of scope,
            // and a loss costs no admitted frame.
            if (status_.state == ProviderState::Running)
            {
                if (!admit(event))
                {
                    return false;
                }
            }
            else if (status_.state == ProviderState::Draining && std::holds_alternative<StereoFrame>(event))
            {
                ++frames_after_cutoff_;
            }
        }
        else
        {
            camera_ended_ = true;
            const auto* failure = std::get_if<InputFailed>(&event);
            const std::string reason = failure ? "the camera input failed: " + failure->reason : "the camera input ended";
            // A source stopped by the same signal as this run must not fail the finish under way.
            if (!finish_requested_)
            {
                fail(ProviderComponent::CameraInput, reason);
                return false;
            }
            if (failure || !std::get<InputEnded>(event).requested)
            {
                isaaccapture::Logger::get(kLoggerName)->info("After the finish request, {}", reason);
            }
        }
    }
}

bool PoseProvider::admit(CameraEvent& event)
{
    auto* frame = std::get_if<StereoFrame>(&event);
    const uint64_t lost_before = status_.frames_lost;
    std::optional<EngineImageGroup> group;
    try
    {
        if (frame != nullptr)
        {
            group = input_->prepare(std::move(*frame), status_);
        }
        else
        {
            input_->lose(std::get<FrameLoss>(event), status_);
        }
    }
    catch (const std::invalid_argument& e)
    {
        fail(ProviderComponent::InputPreparation, e.what());
        return false;
    }
    // The frame was moved into preparation, which released it; a released frame keeps its capture number and times.
    if (run_log_ && frame != nullptr)
    {
        run_log_->frame(*frame, status_.frames_lost - lost_before, group);
    }
    else if (run_log_)
    {
        run_log_->loss(std::get<FrameLoss>(event), status_.frames_lost - lost_before);
    }
    if (!group)
    {
        return true;
    }
    const uint64_t group_id = group->group_id;
    EnqueueResult result = EnqueueResult::Accepted;
    if (!call_engine("on_image_group", [&] { result = engine_->on_image_group(std::move(*group)); }))
    {
        return false;
    }
    if (run_log_)
    {
        run_log_->admission(group_id, result);
    }
    if (const char* how = rejection_text(result))
    {
        fail(ProviderComponent::Engine, "the engine rejected image group " + std::to_string(group_id) + " " + how);
        return false;
    }
    if (result == EnqueueResult::Accepted)
    {
        accepted_groups_.push_back(group_id);
    }
    return true;
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
        // Accepted groups may go without an estimate, but an estimate never repeats a group or goes back.
        const auto next = std::lower_bound(accepted_groups_.begin(), accepted_groups_.end(), pose.group_id);
        if (next == accepted_groups_.end() || *next != pose.group_id)
        {
            fail(ProviderComponent::Engine, "estimate for image group " + std::to_string(pose.group_id) +
                                                ", which is not an accepted group awaiting its estimate");
            return false;
        }
        accepted_groups_.erase(accepted_groups_.begin(), next + 1);
        if (!output_)
        {
            continue;
        }
        try
        {
            output_->publish(pose, status_);
        }
        catch (const std::exception& e)
        {
            fail(ProviderComponent::Output,
                 "estimate for image group " + std::to_string(pose.group_id) + " not published: " + e.what());
            return false;
        }
        if (run_log_)
        {
            run_log_->estimate(pose, *status_.last_publication_time_ns);
        }
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
    // No estimate is taken after stop().
    output_.reset();
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
    // After every other step, so that the summary has the run's outcome. A write failure before it keeps the summary
    // out of the file; one at the close comes after it, so finish() returns it for the status.
    if (run_log_)
    {
        try
        {
            if (std::optional<std::string> error = run_log_->finish(
                    status_, finish_requested_, input_ ? input_->next_sequence() : std::nullopt, frames_after_cutoff_))
            {
                record_failure(ProviderComponent::RunLog, std::move(*error));
            }
        }
        catch (const std::exception& e)
        {
            record_failure(ProviderComponent::RunLog, std::string("run log summary not written: ") + e.what());
        }
        run_log_.reset();
    }

    status_.state = ProviderState::Finished;
    if (session_)
    {
        // The final report, then the session closes, before the last log line.
        session_->publish_status(status_);
        session_.reset();
    }
    auto logger = isaaccapture::Logger::get(kLoggerName);
    if (status_.failed)
    {
        logger->error("Finished: failed ({}): {}; frames lost {}, frames dropped {}; {}",
                      component_name(*status_.failed_component), status_.failure_reason, status_.frames_lost,
                      status_.frames_dropped, health_text(status_.engine_health));
    }
    else
    {
        logger->info("Finished; frames lost {}, frames dropped {}; {}", status_.frames_lost, status_.frames_dropped,
                     health_text(status_.engine_health));
    }
}

} // namespace pose_provider
