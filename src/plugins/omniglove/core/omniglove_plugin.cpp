// SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "inc/omniglove/omniglove_plugin.hpp"

#include <oxr_utils/math.hpp>
#include <oxr_utils/os_time.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace plugins::omniglove
{

namespace
{

constexpr auto kWorkerPeriod = std::chrono::milliseconds(2);
constexpr int64_t kStaleThresholdNs = 200'000'000;
constexpr XrSpaceLocationFlags kTrackedFlags =
    XR_SPACE_LOCATION_POSITION_TRACKED_BIT | XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT;

plugin_utils::WristSourceMode wrist_source_mode_from_env()
{
    const char* value = std::getenv("OMNIGLOVE_WRIST_SOURCE");
    if (value == nullptr || *value == '\0' || std::strcmp(value, "hand_tracking") == 0)
    {
        return plugin_utils::WristSourceMode::HandTracking;
    }
    if (std::strcmp(value, "auto") == 0)
    {
        return plugin_utils::WristSourceMode::Auto;
    }
    if (std::strcmp(value, "controller") == 0)
    {
        return plugin_utils::WristSourceMode::Controller;
    }
    isaaccapture::Logger::get("isaaccapture.plugins.omniglove.main")
        ->warn("unknown OMNIGLOVE_WRIST_SOURCE '{}', using 'hand_tracking'", value);
    return plugin_utils::WristSourceMode::HandTracking;
}

XrPosef pose_from_env(const char* name)
{
    XrPosef pose{ { 0.0F, 0.0F, 0.0F, 1.0F }, { 0.0F, 0.0F, 0.0F } };
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0')
    {
        return pose;
    }
    if (std::sscanf(value, "%f,%f,%f,%f,%f,%f,%f", &pose.position.x, &pose.position.y, &pose.position.z,
                    &pose.orientation.x, &pose.orientation.y, &pose.orientation.z, &pose.orientation.w) != 7)
    {
        throw std::runtime_error(std::string(name) + " must be px,py,pz,qx,qy,qz,qw");
    }
    const float values[] = { pose.position.x,    pose.position.y,    pose.position.z,   pose.orientation.x,
                             pose.orientation.y, pose.orientation.z, pose.orientation.w };
    if (!std::all_of(std::begin(values), std::end(values), [](float component) { return std::isfinite(component); }))
    {
        throw std::runtime_error(std::string(name) + " contains a non-finite value");
    }
    const float norm = std::sqrt(pose.orientation.x * pose.orientation.x + pose.orientation.y * pose.orientation.y +
                                 pose.orientation.z * pose.orientation.z + pose.orientation.w * pose.orientation.w);
    if (norm < 1.0e-6F)
    {
        throw std::runtime_error(std::string(name) + " contains a zero quaternion");
    }
    pose.orientation.x /= norm;
    pose.orientation.y /= norm;
    pose.orientation.z /= norm;
    pose.orientation.w /= norm;
    return pose;
}

const char* hand_name(bool is_left)
{
    return is_left ? "left" : "right";
}

} // namespace

void OmniglovePlugin::SessionDeleter::operator()(omniglove_session_t* session) const noexcept
{
    omniglove_session_destroy(session);
}

OmniglovePlugin::OmniglovePlugin(const std::string& plugin_root_id)
{
    m_logger->info("initializing with root {}", plugin_root_id);

    plugin_utils::WristSourceConfig wrist_config;
    wrist_config.mode = wrist_source_mode_from_env();
    if (wrist_config.mode != plugin_utils::WristSourceMode::HandTracking)
    {
        // Controller aim is not a wrist pose without an explicit mount calibration.
        for (const char* name : { "OMNIGLOVE_AIM_TO_WRIST_LEFT", "OMNIGLOVE_AIM_TO_WRIST_RIGHT" })
        {
            const char* value = std::getenv(name);
            if (value == nullptr || *value == '\0')
            {
                throw std::runtime_error(std::string(name) + " is required for controller and auto wrist sources");
            }
        }
    }
    wrist_config.left_aim_to_wrist = pose_from_env("OMNIGLOVE_AIM_TO_WRIST_LEFT");
    wrist_config.right_aim_to_wrist = pose_from_env("OMNIGLOVE_AIM_TO_WRIST_RIGHT");
    const char* wrist_mode = wrist_config.mode == plugin_utils::WristSourceMode::HandTracking ? "hand_tracking" :
                             wrist_config.mode == plugin_utils::WristSourceMode::Controller   ? "controller" :
                                                                                                "auto";
    m_logger->info("wrist source {}", wrist_mode);
    initialize_sdk();
    const auto wrist_requirements = plugin_utils::WristPoseSource::collect_requirements(wrist_config.mode);

    auto extensions = core::DeviceIOSession::get_required_extensions(wrist_requirements.trackers);
    extensions.push_back(XR_NVX1_DEVICE_INTERFACE_BASE_EXTENSION_NAME);
    extensions.insert(extensions.end(), wrist_requirements.extensions.begin(), wrist_requirements.extensions.end());

    m_session = std::make_shared<core::OpenXRSession>("Omniglove", extensions);
    const auto handles = m_session->get_handles();
    m_deviceio_session = core::DeviceIOSession::run(wrist_requirements.trackers, handles);
    m_wrist_source = std::make_unique<plugin_utils::WristPoseSource>(
        wrist_config, handles, m_deviceio_session.get(), wrist_requirements.controller_tracker);
    m_time_converter = std::make_unique<core::XrTimeConverter>(handles);

    start_device_streams();
    m_running.store(true, std::memory_order_release);
    try
    {
        m_worker = std::thread(&OmniglovePlugin::worker_thread, this);
    }
    catch (...)
    {
        m_running.store(false, std::memory_order_release);
        throw;
    }
    m_logger->info("initialized with {} glove stream(s)", m_streams.size());
}

OmniglovePlugin::~OmniglovePlugin()
{
    m_running.store(false, std::memory_order_release);
    if (m_worker.joinable())
    {
        m_worker.join();
    }
    m_left_injector.reset();
    m_right_injector.reset();
    m_sdk_session.reset();
    m_streams.clear();
}

bool OmniglovePlugin::is_running() const noexcept
{
    return m_running.load(std::memory_order_acquire);
}

bool OmniglovePlugin::has_failed() const noexcept
{
    return m_failed.load(std::memory_order_acquire);
}

void OmniglovePlugin::initialize_sdk()
{
    // Accept any SDK ABI version when its header and runtime agree. A new ABI
    // number is not itself an incompatibility; required API changes fail build.
    const uint32_t runtime_abi = omniglove_abi_version();
    if (runtime_abi != OMNIGLOVE_ABI_VERSION)
    {
        throw std::runtime_error("Omniglove header/runtime ABI mismatch: header=" +
                                 std::to_string(OMNIGLOVE_ABI_VERSION) + " runtime=" + std::to_string(runtime_abi));
    }
    m_logger->info(
        "Omniglove ABI {}.{}.{}", (runtime_abi >> 16U) & 0xffU, (runtime_abi >> 8U) & 0xffU, runtime_abi & 0xffU);
    m_logger->info("Omniglove SDK release {} ({})", OMNIGLOVE_SDK_RELEASE_VERSION, OMNIGLOVE_SDK_GIT_COMMIT);

    omniglove_session_t* session = nullptr;
    omniglove_init_report_t* report = nullptr;
    const omniglove_error_t error = omniglove_initialize(nullptr, &session, &report);

    if (report != nullptr)
    {
        const size_t size = omniglove_init_report_message(report, nullptr, 0);
        if (size > 1)
        {
            std::vector<char> message(size);
            omniglove_init_report_message(report, message.data(), message.size());
            m_logger->info("SDK discovery: {}", message.data());
        }
        omniglove_init_report_destroy(report);
    }

    if (error != OMNIGLOVE_ERROR_NONE || session == nullptr)
    {
        throw std::runtime_error(std::string("omniglove_initialize failed: ") + omniglove_error_name(error));
    }
    m_sdk_session.reset(session);
}

void OmniglovePlugin::start_device_streams()
{
    bool have_left = false;
    bool have_right = false;
    const size_t count = omniglove_session_device_count(m_sdk_session.get());

    for (size_t index = 0; index < count; ++index)
    {
        omniglove_device_t* device = omniglove_session_device_at(m_sdk_session.get(), index);
        if (device == nullptr)
        {
            m_logger->warn("ignoring null glove entry {}", index);
            continue;
        }
        const omniglove_hand_t hand = omniglove_device_hand(device);
        if (hand == OMNIGLOVE_HAND_UNKNOWN)
        {
            m_logger->warn("ignoring glove {} because its hand side is unknown", index);
            continue;
        }
        const bool is_left = hand == OMNIGLOVE_HAND_LEFT;
        bool& occupied = is_left ? have_left : have_right;
        if (occupied)
        {
            m_logger->warn("ignoring a second {} glove", hand_name(is_left));
            continue;
        }

        auto stream = std::make_unique<DeviceStream>();
        stream->device = device;
        stream->context = { this, is_left };

        omniglove_stream_config_t config{};
        config.struct_size = sizeof(config);
        config.kind = OMNIGLOVE_STREAM_POST_DATA;
        config.stream_mode = OMNIGLOVE_STREAM_MODE_DEFAULT;
        config.output_mode = OMNIGLOVE_STREAM_OUTPUT_DIRECT;
        config.source_interval_ms = 8;
        config.source_timeout_ms = 200;
        config.callback = &OmniglovePlugin::stream_callback;
        config.callback_user_data = &stream->context;
        config.callback_queue_capacity = 64;

        const auto configure_error = omniglove_device_stream_configure(device, &config);
        if (configure_error != OMNIGLOVE_STREAM_ERROR_NONE)
        {
            m_logger->error("could not configure {} POST_DATA stream: {}", hand_name(is_left),
                            static_cast<unsigned>(configure_error));
            continue;
        }
        const auto start_error = omniglove_device_stream_start(device, OMNIGLOVE_STREAM_POST_DATA);
        if (start_error != OMNIGLOVE_STREAM_ERROR_NONE)
        {
            m_logger->error(
                "could not start {} POST_DATA stream: {}", hand_name(is_left), static_cast<unsigned>(start_error));
            continue;
        }
        stream->started = true;
        occupied = true;
        m_logger->info("started {} glove at {} Hz", hand_name(is_left), omniglove_device_nominal_source_rate_hz(device));
        m_streams.push_back(std::move(stream));
    }

    if (m_streams.empty())
    {
        throw std::runtime_error("no left or right Omniglove POST_DATA stream could be started");
    }
}

void OmniglovePlugin::stream_callback(const omniglove_stream_sample_t* sample, void* user_data)
{
    if (sample == nullptr || user_data == nullptr)
    {
        return;
    }
    auto* context = static_cast<StreamContext*>(user_data);
    context->plugin->on_sample(*sample, context->is_left);
}

void OmniglovePlugin::on_sample(const omniglove_stream_sample_t& sample, bool is_left)
{
    if (sample.kind != OMNIGLOVE_STREAM_POST_DATA || sample.layout_version != OMNIGLOVE_STREAM_POST_LAYOUT_VERSION ||
        sample.data.postdata == nullptr || sample.sample_monotonic_ns > std::numeric_limits<int64_t>::max())
    {
        invalidate_hand(is_left);
        return;
    }

    HandFrame converted;
    const ConversionStatus status = convert_post_data(*sample.data.postdata, is_left, converted.joints);
    if (status != ConversionStatus::Success)
    {
        invalidate_hand(is_left);
        m_logger->debug("{} POST_DATA rejected: {}", hand_name(is_left), conversion_status_name(status));
        return;
    }
    converted.sample_monotonic_ns = sample.sample_monotonic_ns;
    converted.connection_generation = sample.connection_generation;
    converted.source_sequence = sample.source_sequence;
    converted.valid = true;

    std::lock_guard<std::mutex> lock(m_frame_mutex);
    (is_left ? m_left : m_right) = converted;
}

void OmniglovePlugin::invalidate_hand(bool is_left)
{
    std::lock_guard<std::mutex> lock(m_frame_mutex);
    (is_left ? m_left : m_right).valid = false;
}

void OmniglovePlugin::pump_hand(std::unique_ptr<plugin_utils::HandInjector>& injector,
                                XrHandEXT hand,
                                const HandFrame& frame,
                                uint64_t& last_pushed_ns)
{
    const bool is_left = hand == XR_HAND_LEFT_EXT;
    const int64_t now_ns = core::os_monotonic_now_ns();
    const bool fresh = frame.valid && frame.sample_monotonic_ns <= static_cast<uint64_t>(now_ns) &&
                       now_ns - static_cast<int64_t>(frame.sample_monotonic_ns) < kStaleThresholdNs;
    if (!fresh)
    {
        if (injector)
        {
            m_logger->info("{} hand inactive: SDK frame stale or invalid", hand_name(is_left));
        }
        injector.reset();
        last_pushed_ns = 0;
        return;
    }
    if (frame.sample_monotonic_ns == last_pushed_ns)
    {
        return;
    }

    const XrTime time = m_time_converter->convert_monotonic_ns_to_xrtime(static_cast<int64_t>(frame.sample_monotonic_ns));
    const plugin_utils::WristSample wrist = m_wrist_source->query(is_left, time);
    if (!wrist.valid)
    {
        if (injector)
        {
            m_logger->info("{} hand inactive: wrist pose unavailable", hand_name(is_left));
        }
        injector.reset();
        return;
    }

    if (!injector)
    {
        const auto handles = m_session->get_handles();
        injector = std::make_unique<plugin_utils::HandInjector>(handles.instance, handles.session, hand, handles.space);
        m_logger->info("{} hand active: connection generation {}, source sequence {}", hand_name(is_left),
                       frame.connection_generation, frame.source_sequence);
    }

    auto joints = frame.joints;
    for (auto& joint : joints)
    {
        if ((joint.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) == 0)
        {
            continue;
        }
        joint.pose = oxr_utils::multiply_poses(wrist.pose, joint.pose);
        if (wrist.tracked)
        {
            joint.locationFlags |= kTrackedFlags;
        }
    }
    injector->push(joints.data(), time);
    last_pushed_ns = frame.sample_monotonic_ns;
}

void OmniglovePlugin::worker_thread()
{
    while (m_running.load(std::memory_order_acquire))
    {
        try
        {
            m_deviceio_session->update();
            HandFrame left;
            HandFrame right;
            {
                std::lock_guard<std::mutex> lock(m_frame_mutex);
                left = m_left;
                right = m_right;
            }
            pump_hand(m_left_injector, XR_HAND_LEFT_EXT, left, m_left_last_pushed_ns);
            pump_hand(m_right_injector, XR_HAND_RIGHT_EXT, right, m_right_last_pushed_ns);
        }
        catch (const std::exception& error)
        {
            m_logger->error("update failed: {}", error.what());
            m_left_injector.reset();
            m_right_injector.reset();
            m_failed.store(true, std::memory_order_release);
            m_running.store(false, std::memory_order_release);
            return;
        }
        std::this_thread::sleep_for(kWorkerPeriod);
    }
}

} // namespace plugins::omniglove
