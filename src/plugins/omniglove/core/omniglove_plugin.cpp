// SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "inc/omniglove_plugin/omniglove_plugin.hpp"

#include <omniglove/omniglove.h>
#include <oxr_utils/math.hpp>
#include <oxr_utils/os_time.hpp>

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace plugins::omniglove
{

namespace
{

constexpr auto kSourceTimeout = std::chrono::milliseconds(200);
constexpr int64_t kStaleThresholdNs = std::chrono::duration_cast<std::chrono::nanoseconds>(kSourceTimeout).count();
constexpr uint32_t kSourceIntervalMs = 8;
constexpr uint32_t kCallbackQueueCapacity = 64;
constexpr XrSpaceLocationFlags kTrackedFlags =
    XR_SPACE_LOCATION_POSITION_TRACKED_BIT | XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT;

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
    wrist_config.mode =
        plugin_utils::wrist_source_mode_from_env("OMNIGLOVE_WRIST_SOURCE", plugin_utils::WristSourceMode::HandTracking);
    if (wrist_config.mode != plugin_utils::WristSourceMode::HandTracking)
    {
        wrist_config.left_aim_to_wrist = plugin_utils::pose_from_env("OMNIGLOVE_AIM_TO_WRIST_LEFT");
        wrist_config.right_aim_to_wrist = plugin_utils::pose_from_env("OMNIGLOVE_AIM_TO_WRIST_RIGHT");
    }
    const char* wrist_mode = wrist_config.mode == plugin_utils::WristSourceMode::HandTracking ? "hand_tracking" :
                             wrist_config.mode == plugin_utils::WristSourceMode::Controller   ? "controller" :
                                                                                                "auto";
    m_logger->info("wrist source {}", wrist_mode);
    initialize_sdk();
    const auto wrist_requirements = plugin_utils::WristPoseSource::collect_requirements(wrist_config.mode);

    auto extensions = core::DeviceIOSession::get_required_extensions(wrist_requirements.trackers);
    extensions.push_back(XR_NVX1_DEVICE_INTERFACE_BASE_EXTENSION_NAME);
    extensions.insert(extensions.end(), wrist_requirements.extensions.begin(), wrist_requirements.extensions.end());

    const auto status_extensions = plugin_utils::PluginDeviceStatusPublisher::get_required_extensions();
    extensions.insert(extensions.end(), status_extensions.begin(), status_extensions.end());
    m_session = std::make_unique<core::OpenXRSession>("Omniglove", extensions);
    const auto handles = m_session->get_handles();
    m_deviceio_session = core::DeviceIOSession::run(wrist_requirements.trackers, handles);
    m_wrist_source = std::make_unique<plugin_utils::WristPoseSource>(
        wrist_config, handles, m_deviceio_session.get(), wrist_requirements.controller_tracker);
    m_time_converter.emplace(handles);
    m_status_publisher.emplace(handles, plugin_root_id);

    start_device_streams();
    m_logger->info("initialized with {} glove stream(s)", m_streams.size());
}

OmniglovePlugin::~OmniglovePlugin() = default;

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

        // Own the callback context before starting the stream, including failure cleanup.
        m_streams.push_back(std::make_unique<StreamContext>(StreamContext{ this, is_left }));

        omniglove_stream_config_t config{};
        config.struct_size = sizeof(config);
        config.kind = OMNIGLOVE_STREAM_POST_DATA;
        config.stream_mode = OMNIGLOVE_STREAM_MODE_DEFAULT;
        config.output_mode = OMNIGLOVE_STREAM_OUTPUT_DIRECT;
        config.source_interval_ms = kSourceIntervalMs;
        config.source_timeout_ms = static_cast<uint32_t>(kSourceTimeout.count());
        config.callback = &OmniglovePlugin::stream_callback;
        config.callback_user_data = m_streams.back().get();
        config.callback_queue_capacity = kCallbackQueueCapacity;

        const auto configure_error = omniglove_device_stream_configure(device, &config);
        if (configure_error != OMNIGLOVE_STREAM_ERROR_NONE)
        {
            throw std::runtime_error(std::string("could not configure ") + hand_name(is_left) +
                                     " POST_DATA stream: " + std::to_string(static_cast<unsigned>(configure_error)));
        }
        const auto start_error = omniglove_device_stream_start(device, OMNIGLOVE_STREAM_POST_DATA);
        if (start_error != OMNIGLOVE_STREAM_ERROR_NONE)
        {
            throw std::runtime_error(std::string("could not start ") + hand_name(is_left) +
                                     " POST_DATA stream: " + std::to_string(static_cast<unsigned>(start_error)));
        }
        occupied = true;
        m_logger->info("started {} glove at {} Hz", hand_name(is_left), omniglove_device_nominal_source_rate_hz(device));
    }

    if (m_streams.empty())
    {
        throw std::runtime_error("no left or right Omniglove POST_DATA stream could be started");
    }
}

void OmniglovePlugin::stream_callback(const omniglove_stream_sample_t* sample, void* user_data)
{
    if (sample == nullptr)
    {
        return;
    }
    auto* context = static_cast<StreamContext*>(user_data);
    context->plugin->on_sample(*sample, context->is_left);
}

void OmniglovePlugin::on_sample(const omniglove_stream_sample_t& sample, bool is_left)
{
    // The SDK data union may only be read through the member selected by kind.
    const bool post_data_present = sample.kind == OMNIGLOVE_STREAM_POST_DATA && sample.data.postdata != nullptr;
    if (!post_data_present || sample.layout_version != OMNIGLOVE_STREAM_POST_LAYOUT_VERSION ||
        sample.sample_monotonic_ns > std::numeric_limits<int64_t>::max())
    {
        std::lock_guard<std::mutex> lock(m_frame_mutex);
        auto& hand = m_hands[is_left ? 0 : 1];
        hand.frame.reset();
        if (!hand.sample_error_logged)
        {
            m_logger->error("{} SDK sample contract mismatch: kind {}, layout {}, post data present {}, timestamp {}",
                            hand_name(is_left), static_cast<unsigned>(sample.kind), sample.layout_version,
                            post_data_present, sample.sample_monotonic_ns);
            hand.sample_error_logged = true;
        }
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
    converted.sample_monotonic_ns = static_cast<int64_t>(sample.sample_monotonic_ns);
    converted.connection_generation = sample.connection_generation;
    converted.source_sequence = sample.source_sequence;

    std::lock_guard<std::mutex> lock(m_frame_mutex);
    m_hands[is_left ? 0 : 1].frame = converted;
}

void OmniglovePlugin::invalidate_hand(bool is_left)
{
    std::lock_guard<std::mutex> lock(m_frame_mutex);
    m_hands[is_left ? 0 : 1].frame.reset();
}

void OmniglovePlugin::pump_hand(HandState& hand, bool is_left, const std::optional<HandFrame>& frame)
{
    auto& injector = hand.injector;
    const int64_t now_ns = core::os_monotonic_now_ns();
    const bool fresh =
        frame && frame->sample_monotonic_ns <= now_ns && now_ns - frame->sample_monotonic_ns < kStaleThresholdNs;
    if (!fresh)
    {
        if (injector)
        {
            m_logger->info("{} hand inactive: SDK frame stale or invalid", hand_name(is_left));
        }
        injector.reset();
        hand.state = core::PluginDeviceState_DISCONNECTED;
        hand.reason = core::PluginDeviceReason_NO_HARDWARE_SIGNAL;
        return;
    }
    if (frame->sample_monotonic_ns == hand.last_pushed_ns)
    {
        return;
    }

    const XrTime time = m_time_converter->convert_monotonic_ns_to_xrtime(frame->sample_monotonic_ns);
    const plugin_utils::WristSample wrist = m_wrist_source->query(is_left, time);
    if (!wrist.valid)
    {
        if (injector)
        {
            m_logger->info("{} hand inactive: wrist pose unavailable", hand_name(is_left));
        }
        injector.reset();
        hand.state = core::PluginDeviceState_DEGRADED;
        hand.reason = core::PluginDeviceReason_NO_CURRENT_DATA;
        return;
    }

    if (!injector)
    {
        const auto handles = m_session->get_handles();
        injector = std::make_unique<plugin_utils::HandInjector>(
            handles.instance, handles.session, is_left ? XR_HAND_LEFT_EXT : XR_HAND_RIGHT_EXT, handles.space);
        m_logger->info("{} hand active: connection generation {}, source sequence {}", hand_name(is_left),
                       frame->connection_generation, frame->source_sequence);
    }

    auto joints = frame->joints;
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
    hand.last_pushed_ns = frame->sample_monotonic_ns;
    hand.state = core::PluginDeviceState_CONNECTED;
    hand.reason = core::PluginDeviceReason_NONE;
}

void OmniglovePlugin::update()
{
    m_deviceio_session->update();
    std::array<std::optional<HandFrame>, 2> frames;
    {
        std::lock_guard<std::mutex> lock(m_frame_mutex);
        frames = { m_hands[0].frame, m_hands[1].frame };
    }
    pump_hand(m_hands[0], true, frames[0]);
    pump_hand(m_hands[1], false, frames[1]);
    m_status_publisher->publish_if_changed({ { "/hand/left", m_hands[0].state, m_hands[0].reason, {} },
                                             { "/hand/right", m_hands[1].state, m_hands[1].reason, {} } },
                                           core::os_monotonic_now_ns());
}

} // namespace plugins::omniglove
