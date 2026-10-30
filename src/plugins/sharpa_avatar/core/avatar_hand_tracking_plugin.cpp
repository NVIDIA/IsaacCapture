// SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/avatar/avatar_hand_tracking_plugin.hpp"

#include "avatar_glove_collection.hpp"

#include <flatbuffers/flatbuffers.h>
#include <log_bridge/logger.hpp>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/math.hpp>
#include <oxr_utils/os_time.hpp>
#include <oxr_utils/pose_conversions.hpp>
#include <plugin_utils/hand_injector.hpp>
#include <schema/joint_state_generated.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace plugins
{
namespace avatar
{

namespace
{

constexpr size_t kJointFlatbufferSize = 4096;
constexpr auto kGloveRetryInterval = std::chrono::seconds(2);
constexpr auto kOutputSetupRetryInterval = std::chrono::seconds(2);
constexpr auto kHapticCommandTimeout = std::chrono::milliseconds(200);
constexpr auto kHapticFailureRetryInterval = std::chrono::seconds(1);
constexpr auto kSdkHealthRefreshInterval = std::chrono::milliseconds(100);
constexpr auto kDonglePresenceInterval = std::chrono::seconds(1);
constexpr double kIngressCallbackMaxHz = 10.0;
constexpr uint16_t kAvatarDongleVendorId = 0x1915;
constexpr uint16_t kAvatarDongleProductId = 0x520F;
constexpr std::string_view kUsbDevicesRoot = "/sys/bus/usb/devices";

constexpr std::array<const char*, 2> kHandDevicePaths{ "/hand/left", "/hand/right" };
constexpr std::array<const char*, 2> kPhysicalGlovePaths{ "/glove/left", "/glove/right" };
constexpr std::array<const char*, 2> kRawDevicePaths{ "/glove/avatar_raw_left", "/glove/avatar_raw_right" };
constexpr std::array<const char*, 2> kRobotDevicePaths{ "/glove/avatar_robot_left", "/glove/avatar_robot_right" };

constexpr std::array<std::string_view, 22> kRawJointNames = {
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA",  "thumb_IP",      "index_MCP_AA",
    "index_MCP_FE", "index_PIP",    "index_DIP",    "middle_MCP_AA", "middle_MCP_FE", "middle_PIP",
    "middle_DIP",   "ring_MCP_AA",  "ring_MCP_FE",  "ring_PIP",      "ring_DIP",      "pinky_MCP_AA",
    "pinky_MCP_FE", "pinky_DIP1",   "pinky_DIP2",   "pinky_DIP3",
};

constexpr std::array<std::string_view, 22> kRobotJointNames = {
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA",  "thumb_IP",      "index_MCP_FE",
    "index_MCP_AA", "index_PIP",    "index_DIP",    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP",
    "middle_DIP",   "ring_MCP_FE",  "ring_MCP_AA",  "ring_PIP",      "ring_DIP",      "pinky_CMC",
    "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP",    "pinky_DIP",
};

const std::shared_ptr<spdlog::logger>& logger()
{
    static const auto instance = isaaccapture::Logger::get("isaaccapture.plugins.sharpa_avatar.AvatarTracker");
    return instance;
}

size_t side_index(::avatar::DeviceSide side)
{
    switch (side)
    {
    case ::avatar::DeviceSide::LEFT:
        return 0;
    case ::avatar::DeviceSide::RIGHT:
        return 1;
    }
    throw std::invalid_argument("Unsupported Avatar device side");
}

constexpr XrHandEXT xr_hand(::avatar::DeviceSide side)
{
    switch (side)
    {
    case ::avatar::DeviceSide::LEFT:
        return XR_HAND_LEFT_EXT;
    case ::avatar::DeviceSide::RIGHT:
        return XR_HAND_RIGHT_EXT;
    }
    return XR_HAND_LEFT_EXT;
}

size_t joint_category_index(::avatar::DeviceDataCategory category)
{
    switch (category)
    {
    case ::avatar::DeviceDataCategory::RAW:
        return 0;
    case ::avatar::DeviceDataCategory::ROBOT:
        return 1;
    case ::avatar::DeviceDataCategory::HUMAN:
        break;
    }
    throw std::invalid_argument("HUMAN data is not a joint stream");
}

size_t data_category_index(::avatar::DeviceDataCategory category)
{
    const auto it = std::find(kDataCategories.begin(), kDataCategories.end(), category);
    if (it == kDataCategories.end())
    {
        throw std::invalid_argument("Unsupported Avatar data category");
    }
    return static_cast<size_t>(std::distance(kDataCategories.begin(), it));
}

const char* collection_id(::avatar::DeviceSide side, ::avatar::DeviceDataCategory category)
{
    static constexpr std::array<std::array<const char*, 2>, 2> kCollectionIds = { {
        { AVATAR_RAW_LEFT_COLLECTION_ID, AVATAR_ROBOT_LEFT_COLLECTION_ID },
        { AVATAR_RAW_RIGHT_COLLECTION_ID, AVATAR_ROBOT_RIGHT_COLLECTION_ID },
    } };
    return kCollectionIds[side_index(side)][joint_category_index(category)];
}

::avatar::AvatarDataFrame& cached_joint_frame(GloveState& glove, ::avatar::DeviceDataCategory category)
{
    switch (category)
    {
    case ::avatar::DeviceDataCategory::RAW:
        return glove.raw_frame;
    case ::avatar::DeviceDataCategory::ROBOT:
        return glove.robot_frame;
    case ::avatar::DeviceDataCategory::HUMAN:
        break;
    }
    throw std::invalid_argument("HUMAN data has no joint frame");
}

const ::avatar::AvatarDataFrame& cached_joint_frame(const GloveState& glove, ::avatar::DeviceDataCategory category)
{
    switch (category)
    {
    case ::avatar::DeviceDataCategory::RAW:
        return glove.raw_frame;
    case ::avatar::DeviceDataCategory::ROBOT:
        return glove.robot_frame;
    case ::avatar::DeviceDataCategory::HUMAN:
        break;
    }
    throw std::invalid_argument("HUMAN data has no joint frame");
}

::avatar::AvatarDataFrame::PayloadCase payload_case(::avatar::DeviceDataCategory category)
{
    switch (category)
    {
    case ::avatar::DeviceDataCategory::RAW:
        return ::avatar::AvatarDataFrame::kRaw;
    case ::avatar::DeviceDataCategory::ROBOT:
        return ::avatar::AvatarDataFrame::kRobot;
    case ::avatar::DeviceDataCategory::HUMAN:
        return ::avatar::AvatarDataFrame::kSkeleton;
    }
    return ::avatar::AvatarDataFrame::PAYLOAD_NOT_SET;
}

const ::avatar::Hand& hand_payload(const ::avatar::AvatarDataFrame& frame, ::avatar::DeviceDataCategory category)
{
    switch (category)
    {
    case ::avatar::DeviceDataCategory::RAW:
        return frame.raw;
    case ::avatar::DeviceDataCategory::ROBOT:
        return frame.robot;
    case ::avatar::DeviceDataCategory::HUMAN:
        break;
    }
    throw std::invalid_argument("HUMAN data has no joint payload");
}

const std::array<std::string_view, 22>& joint_names(::avatar::DeviceDataCategory category)
{
    switch (category)
    {
    case ::avatar::DeviceDataCategory::RAW:
        return kRawJointNames;
    case ::avatar::DeviceDataCategory::ROBOT:
        return kRobotJointNames;
    case ::avatar::DeviceDataCategory::HUMAN:
        break;
    }
    throw std::invalid_argument("HUMAN data has no joint names");
}

constexpr std::string_view error_name(::avatar::ErrorCode error)
{
    using ErrorCode = ::avatar::ErrorCode;
    switch (error)
    {
    case ErrorCode::SUCCESS:
        return "SUCCESS";
    case ErrorCode::INVALID_INPUT_PARAMETER:
        return "INVALID_INPUT_PARAMETER";
    case ErrorCode::OPERATION_NOT_ALLOWED:
        return "OPERATION_NOT_ALLOWED";
    case ErrorCode::CONNECTION_FAILED:
        return "CONNECTION_FAILED";
    case ErrorCode::TCP_SEND_FAILED:
        return "TCP_SEND_FAILED";
    case ErrorCode::TCP_RECV_FAILED:
        return "TCP_RECV_FAILED";
    case ErrorCode::INVALID_RESPONSE:
        return "INVALID_RESPONSE";
    case ErrorCode::DEVICE_NOT_FOUND:
        return "DEVICE_NOT_FOUND";
    case ErrorCode::NO_VALID_DATA_RETURNED:
        return "NO_VALID_DATA_RETURNED";
    case ErrorCode::DEVICE_UNSUPPORTED_COMMAND:
        return "DEVICE_UNSUPPORTED_COMMAND";
    case ErrorCode::INVALID_HEADER:
        return "INVALID_HEADER";
    case ErrorCode::INCOMPLETE_PAYLOAD:
        return "INCOMPLETE_PAYLOAD";
    case ErrorCode::CRC_ERROR:
        return "CRC_ERROR";
    case ErrorCode::INVALID_PAYLOAD:
        return "INVALID_PAYLOAD";
    case ErrorCode::INNER_ERROR:
        return "INNER_ERROR";
    case ErrorCode::UPGRADE_IN_PROGRESS:
        return "UPGRADE_IN_PROGRESS";
    case ErrorCode::INVALID_FIRMWARE_FILE:
        return "INVALID_FIRMWARE_FILE";
    case ErrorCode::FIRMWARE_VERSION_INCOMPATIBLE:
        return "FIRMWARE_VERSION_INCOMPATIBLE";
    case ErrorCode::UNKNOWN_RSP_CODE:
        return "UNKNOWN_RSP_CODE";
    }
    return "UNRECOGNIZED_ERROR";
}

std::optional<std::string> physical_device_error(const ::avatar::DeviceInfo& info, const GloveState& glove)
{
    std::ostringstream message;
    bool has_error = false;
    for (const auto& [code, unused] : info.diagnostics)
    {
        (void)unused;
        if (code == static_cast<int>(::avatar::FaultCode::NORMAL))
        {
            continue;
        }
        message << (has_error ? ", " : "Avatar SDK fault: ") << "0x" << std::hex << std::uppercase << code;
        has_error = true;
    }
    if (!has_error && info.error_code != 0)
    {
        message << "Avatar SDK fault: 0x" << std::hex << std::uppercase << info.error_code;
        has_error = true;
    }
    if (glove.haptic_error)
    {
        message << (has_error ? "; " : "") << *glove.haptic_error;
        has_error = true;
    }
    return has_error ? std::optional<std::string>(message.str()) : std::nullopt;
}

// OpenXR slot -> HUMAN landmark index for avatar-sdk 1.7.3-17. Unsupported
// palm and little-finger intermediate/distal slots remain invalid.
constexpr std::array<std::optional<size_t>, XR_HAND_JOINT_COUNT_EXT> kOpenXRSlotSourceIndices = {
    std::nullopt, 0,  1,  3,  5,  6,  7,  8,  9,  10, 11,           12,           13,
    14,           15, 16, 17, 18, 19, 20, 21, 22, 23, std::nullopt, std::nullopt, 24,
};

} // anonymous namespace

// Wrist offsets for XR controllers: align the controller aim pose to a natural
// wrist orientation for each hand.
static constexpr XrPosef kLeftHandOffset = { { -0.70710678f, -0.5f, 0.0f, 0.5f }, { -0.1f, 0.02f, -0.02f } };
static constexpr XrPosef kRightHandOffset = { { -0.70710678f, 0.5f, 0.0f, 0.5f }, { 0.1f, 0.02f, -0.02f } };

AvatarSdkSession::AvatarSdkSession(const std::string& config_path)
{
    const std::string sdk_config = config_path.empty() ? AVATAR_SDK_CONFIG_PATH : config_path;
    const auto error = ::avatar::AvatarSDK::get_instance().initialize(sdk_config);
    if (error != ::avatar::ErrorCode::SUCCESS)
    {
        throw std::runtime_error("Avatar SDK initialize failed: " + std::string(error_name(error)) + " (" +
                                 std::to_string(static_cast<int>(error)) + ")");
    }
}

AvatarSdkSession::~AvatarSdkSession() noexcept
{
    try
    {
        ::avatar::AvatarSDK::get_instance().destroy();
    }
    catch (...)
    {
        // Destructors must not propagate vendor exceptions.
    }
}

::avatar::AvatarSDK& AvatarSdkSession::get()
{
    return ::avatar::AvatarSDK::get_instance();
}

void GloveIngressSignal::record() noexcept
{
    const auto now_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    m_last_event_ns.store(now_ns, std::memory_order_relaxed);
    m_generation.fetch_add(1, std::memory_order_release);
}

uint64_t GloveIngressSignal::generation() const noexcept
{
    return m_generation.load(std::memory_order_acquire);
}

std::optional<std::chrono::steady_clock::time_point> GloveIngressSignal::last_event() const noexcept
{
    const int64_t event_ns = m_last_event_ns.load(std::memory_order_relaxed);
    if (event_ns < 0)
    {
        return std::nullopt;
    }
    return std::chrono::steady_clock::time_point(
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::nanoseconds(event_ns)));
}

GloveState::~GloveState() noexcept
{
    reset();
}

void GloveState::reset() noexcept
{
    ingress.reset();
    if (device)
    {
        try
        {
            device->set_raw_human_frame_callback({});
        }
        catch (...)
        {
            // Continue teardown even if the vendor callback unregister fails.
        }
        try
        {
            device->stop();
        }
        catch (...)
        {
            // Device release must not escape a destructor.
        }
    }
    clear_current_data();
    device.reset();
    observed_ingress_generation = 0;
    last_ingress.reset();
    haptic_error.reset();
    output_setup_error.reset();
    last_output_setup_attempt.reset();
    started = false;
    ready = false;
}

void GloveState::clear_current_data() noexcept
{
    landmarks.clear();
    raw_frame = {};
    robot_frame = {};
    for (auto& dataset : datasets)
    {
        dataset.last_stamp.reset();
        dataset.last_fresh_data.reset();
        dataset.fetch_error.reset();
    }
}

AvatarTracker::AvatarTracker(AvatarPluginConfig config) : m_config(std::move(config)), m_sdk(m_config.sdk_config_path)
{
    logger()->info("Avatar SDK initialized.");
    logger()->info("datasets: human={} raw={} robot={} haptic={}", m_config.human ? "on" : "off",
                   m_config.raw ? "on" : "off", m_config.robot ? "on" : "off", m_config.haptic ? "on" : "off");
    initialize_openxr();
}

AvatarTracker::~AvatarTracker() noexcept
{
    for (const ::avatar::DeviceSide side : kDeviceSides)
    {
        reset_glove(side);
    }
}

void AvatarTracker::initialize_openxr()
{
    if (m_config.monitoring_plugin_root_id && m_config.monitoring_plugin_root_id->empty())
    {
        throw std::invalid_argument("AvatarTracker monitoring_plugin_root_id must not be empty");
    }

    plugin_utils::WristSourceConfig wrist_config;
    wrist_config.mode = plugin_utils::WristSourceMode::Auto;
    wrist_config.left_aim_to_wrist = kLeftHandOffset;
    wrist_config.right_aim_to_wrist = kRightHandOffset;
    auto wrist_requirements = plugin_utils::WristPoseSource::collect_requirements(wrist_config.mode);

    std::vector<std::shared_ptr<core::ITracker>> trackers = wrist_requirements.trackers;
    if (m_config.haptic)
    {
        m_haptic_reader = std::make_shared<core::HapticCommandReaderTracker>(AVATAR_GLOVE_HAPTIC_COLLECTION_ID);
        trackers.push_back(m_haptic_reader);
    }

    std::vector<std::string> extensions = core::DeviceIOSession::get_required_extensions(trackers);
    const auto add_extension = [&extensions](std::string_view extension)
    {
        if (std::find(extensions.begin(), extensions.end(), extension) == extensions.end())
        {
            extensions.emplace_back(extension);
        }
    };

    if (m_config.raw || m_config.robot)
    {
        for (const auto& extension : core::SchemaPusher::get_required_extensions())
        {
            add_extension(extension);
        }
    }
    if (m_config.human)
    {
        add_extension(XR_NVX1_DEVICE_INTERFACE_BASE_EXTENSION_NAME);
    }
    if (m_config.monitoring_plugin_root_id)
    {
        for (const auto& extension : plugin_utils::PluginDeviceStatusPublisher::get_required_extensions())
        {
            add_extension(extension);
        }
    }
    for (const char* extension : wrist_requirements.extensions)
    {
        add_extension(extension);
    }

    const bool wait_for_openxr_system = false;
    m_session = std::make_shared<core::OpenXRSession>(m_config.app_name, extensions, wait_for_openxr_system);
    m_handles = m_session->get_handles();

    if (m_config.monitoring_plugin_root_id)
    {
        m_status_publisher =
            std::make_unique<plugin_utils::PluginDeviceStatusPublisher>(m_handles, *m_config.monitoring_plugin_root_id);
    }

    m_time_converter.emplace(m_handles);

    m_deviceio_session = core::DeviceIOSession::run(trackers, m_handles);
    m_wrist_source = std::make_unique<plugin_utils::WristPoseSource>(
        wrist_config, m_handles, m_deviceio_session.get(), std::move(wrist_requirements.controller_tracker));
}

void AvatarTracker::start_glove_if_present(GloveState& glove, ::avatar::DeviceSide side)
{
    if (glove.device)
    {
        return;
    }

    glove.device = m_sdk.get().get_device(::avatar::DeviceType::GLOVE, side);
    if (!glove.device)
    {
        return;
    }

    glove.device->set_human_frame_build_enabled(false);
    glove.device->set_raw_human_callback_max_hz(kIngressCallbackMaxHz);
    glove.ingress = std::make_shared<GloveIngressSignal>();
    const std::shared_ptr<GloveIngressSignal> ingress = glove.ingress;
    glove.device->set_raw_human_frame_callback(
        [ingress](const ::avatar::AvatarDataFrame&, const ::avatar::AvatarDataFrame&) { ingress->record(); });

    const auto start_error = glove.device->start();
    if (start_error != ::avatar::ErrorCode::SUCCESS)
    {
        logger()->warn(
            "{} glove start failed: {} ({})", to_string(side), error_name(start_error), static_cast<int>(start_error));
        glove.reset();
        return;
    }
    glove.started = true;
}

void AvatarTracker::refresh_transport_state() noexcept
{
    try
    {
        m_transport_state = m_sdk.get().get_usb_connection_state();
    }
    catch (...)
    {
        m_transport_state = ::avatar::UsbConnectionState::ERROR;
    }

    if (!m_status_publisher)
    {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (!m_last_dongle_check || now - *m_last_dongle_check >= kDonglePresenceInterval)
    {
        m_dongle_presence = detect_usb_device_presence(kUsbDevicesRoot, kAvatarDongleVendorId, kAvatarDongleProductId);
        m_last_dongle_check = now;
    }
}

void AvatarTracker::refresh_sdk_inventory()
{
    const auto now = std::chrono::steady_clock::now();
    if (m_last_inventory_refresh && now - *m_last_inventory_refresh < kSdkHealthRefreshInterval)
    {
        return;
    }
    m_last_inventory_refresh = now;

    try
    {
        std::array<std::optional<::avatar::DeviceInfo>, 2> next_inventory;
        for (const auto& info : m_sdk.get().list_device_info(::avatar::DeviceType::GLOVE))
        {
            next_inventory[side_index(info.device_side)] = info;
        }
        m_inventory = std::move(next_inventory);
        m_inventory_available = true;
        for (size_t index = 0; index < m_inventory.size(); ++index)
        {
            m_ever_observed[index] = m_ever_observed[index] || m_inventory[index].has_value();
        }
        reconcile_glove_handles();
        m_inventory_error_logged = false;
    }
    catch (const std::exception& error)
    {
        m_inventory_available = false;
        if (!m_inventory_error_logged)
        {
            logger()->warn("Avatar SDK inventory refresh failed; tracking handles are retained: {}", error.what());
            m_inventory_error_logged = true;
        }
    }
    catch (...)
    {
        m_inventory_available = false;
        if (!m_inventory_error_logged)
        {
            logger()->warn("Avatar SDK inventory refresh failed; tracking handles are retained.");
            m_inventory_error_logged = true;
        }
    }
}

void AvatarTracker::reconcile_glove_handles()
{
    if (!m_inventory_available)
    {
        return;
    }
    for (const ::avatar::DeviceSide side : kDeviceSides)
    {
        const size_t index = side_index(side);
        GloveState& state = glove(side);
        if (m_inventory[index] || !state.device)
        {
            continue;
        }
        logger()->info("{} glove disappeared from the Avatar SDK inventory.", to_string(side));
        reset_glove(side);
    }
}

void AvatarTracker::reset_glove(::avatar::DeviceSide side) noexcept
{
    const size_t index = side_index(side);
    stop_haptics(side);
    destroy_output_resources(side);
    glove(side).reset();
    m_haptic_commands[index].discard();
    m_haptic_stopped[index] = true;
    m_last_haptic_sample_times[index].reset();
    m_haptic_retry_after[index].reset();
    m_haptic_error_logged[index] = false;
}

void AvatarTracker::try_connect_missing_gloves()
{
    if (!m_inventory_available)
    {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (m_last_glove_retry && now - *m_last_glove_retry < kGloveRetryInterval)
    {
        return;
    }
    m_last_glove_retry = now;

    for (const ::avatar::DeviceSide side : kDeviceSides)
    {
        const size_t index = side_index(side);
        GloveState& state = glove(side);
        if (!state.device && m_inventory[index] && m_inventory[index]->online)
        {
            start_glove_if_present(state, side);
        }
    }

    const auto has_device = [](const GloveState& state) { return state.device != nullptr; };
    const bool waiting = std::none_of(m_gloves.begin(), m_gloves.end(), has_device);
    if (waiting && !m_glove_wait_logged)
    {
        logger()->info("Waiting for an online glove...");
    }
    m_glove_wait_logged = waiting;
}

void AvatarTracker::refresh_glove_activity()
{
    const auto now = std::chrono::steady_clock::now();
    for (const ::avatar::DeviceSide side : kDeviceSides)
    {
        GloveState& state = glove(side);
        if (!state.started)
        {
            continue;
        }

        if (!state.ingress)
        {
            continue;
        }
        const uint64_t generation = state.ingress->generation();
        if (generation != state.observed_ingress_generation)
        {
            state.observed_ingress_generation = generation;
            state.last_ingress = state.ingress->last_event();
            state.ever_received_ingress = state.ever_received_ingress || state.last_ingress.has_value();
        }
        const bool ingress_fresh = state.last_ingress && now - *state.last_ingress < kAvatarDataTimeout;
        if (ingress_fresh && !state.ready)
        {
            if (state.last_output_setup_attempt && now - *state.last_output_setup_attempt < kOutputSetupRetryInterval)
            {
                continue;
            }
            state.last_output_setup_attempt = now;
            if (!try_create_output_resources(side))
            {
                continue;
            }
            state.ready = true;
            if (m_config.haptic)
            {
                stop_haptics(side);
            }
            logger()->info("{} glove connected and streaming.", to_string(side));
        }
        else if (!ingress_fresh && state.ready)
        {
            logger()->warn("{} glove stream is stale; withdrawing its outputs.", to_string(side));
            if (m_config.haptic)
            {
                stop_haptics(side);
            }
            destroy_output_resources(side);
            state.ready = false;
            state.clear_current_data();
        }
    }
}

bool AvatarTracker::try_create_output_resources(::avatar::DeviceSide side)
{
    const size_t index = side_index(side);
    GloveState& state = glove(side);
    try
    {
        std::unique_ptr<plugin_utils::HandInjector> injector;
        std::array<std::unique_ptr<core::SchemaPusher>, 2> joint_pushers;

        if (m_config.human)
        {
            injector = std::make_unique<plugin_utils::HandInjector>(
                m_handles.instance, m_handles.session, xr_hand(side), m_handles.space);
        }
        for (const ::avatar::DeviceDataCategory category : kJointDataCategories)
        {
            if (!dataset_enabled(category))
            {
                continue;
            }
            const std::string localized_name =
                "Avatar " + std::string(to_string(category)) + " " + std::string(to_string(side));
            joint_pushers[joint_category_index(category)] = std::make_unique<core::SchemaPusher>(
                m_handles, core::SchemaPusherConfig{ .collection_id = collection_id(side, category),
                                                     .max_flatbuffer_size = kJointFlatbufferSize,
                                                     .tensor_identifier = "joint_state",
                                                     .localized_name = localized_name });
        }

        m_injectors[index] = std::move(injector);
        m_joint_pushers[index] = std::move(joint_pushers);
        state.output_setup_error.reset();
        return true;
    }
    catch (const std::exception& error)
    {
        state.output_setup_error = "output setup failed: " + std::string(error.what());
        logger()->warn("{} glove output setup failed; this side will retry: {}", to_string(side), error.what());
    }
    catch (...)
    {
        state.output_setup_error = "output setup failed with an unknown error";
        logger()->warn("{} glove output setup failed; this side will retry.", to_string(side));
    }
    return false;
}

void AvatarTracker::destroy_output_resources(::avatar::DeviceSide side) noexcept
{
    const size_t index = side_index(side);
    m_injectors[index].reset();
    for (auto& pusher : m_joint_pushers[index])
    {
        pusher.reset();
    }
}

void AvatarTracker::refresh_data()
{
    for (const ::avatar::DeviceSide side : kDeviceSides)
    {
        GloveState& state = glove(side);
        if (!state.ready)
        {
            continue;
        }
        const auto now = std::chrono::steady_clock::now();
        const auto note_result = [&state, now](::avatar::DeviceDataCategory category, ::avatar::ErrorCode error,
                                               const ::avatar::Stamp* stamp)
        {
            const size_t category_idx = data_category_index(category);
            auto& dataset = state.datasets[category_idx];
            if (error != ::avatar::ErrorCode::SUCCESS)
            {
                dataset.fetch_error = error;
                return;
            }
            dataset.fetch_error.reset();
            if (stamp != nullptr && (!dataset.last_stamp || stamp->sec != dataset.last_stamp->sec ||
                                     stamp->nanosec != dataset.last_stamp->nanosec))
            {
                dataset.last_stamp = *stamp;
                dataset.last_fresh_data = now;
                dataset.ever_received_data = true;
            }
        };
        if (m_config.human)
        {
            ::avatar::AvatarDataFrame frame;
            const auto error = state.device->fetch_data(frame, ::avatar::DeviceDataCategory::HUMAN);
            if (error == ::avatar::ErrorCode::SUCCESS)
            {
                note_result(::avatar::DeviceDataCategory::HUMAN, error, &frame.skeleton.header.stamp);
                state.landmarks = std::move(frame.skeleton.landmark);
            }
            else
            {
                note_result(::avatar::DeviceDataCategory::HUMAN, error, nullptr);
                state.landmarks.clear();
            }
        }
        for (const ::avatar::DeviceDataCategory category : kJointDataCategories)
        {
            if (!dataset_enabled(category))
            {
                continue;
            }
            ::avatar::AvatarDataFrame frame;
            const auto error = state.device->fetch_data(frame, category);
            ::avatar::AvatarDataFrame& cached = cached_joint_frame(state, category);
            if (error == ::avatar::ErrorCode::SUCCESS)
            {
                note_result(category, error, &hand_payload(frame, category).header.stamp);
                cached = std::move(frame);
            }
            else
            {
                note_result(category, error, nullptr);
                cached = {};
            }
        }
    }
}

void AvatarTracker::publish_device_status()
{
    if (!m_status_publisher)
    {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    std::vector<plugin_utils::PluginDeviceStatusEntry> entries;
    entries.reserve(9);
    entries.push_back(make_dongle_status(m_dongle_presence, m_transport_state));

    for (const ::avatar::DeviceSide side : kDeviceSides)
    {
        const size_t index = side_index(side);
        const GloveState& state = glove(side);
        const bool inventory_present = m_inventory[index].has_value();
        const bool ingress_fresh = state.last_ingress && now - *state.last_ingress < kAvatarDataTimeout;
        entries.push_back(make_physical_glove_status(
            kPhysicalGlovePaths[index],
            PhysicalGloveHealthInput{
                .inventory_available = m_inventory_available,
                .inventory_present = inventory_present,
                .ever_observed = m_ever_observed[index],
                .sdk_online = inventory_present && m_inventory[index]->online,
                .ingress_fresh = ingress_fresh,
                .ever_received_ingress = state.ever_received_ingress,
                .device_error = inventory_present ? physical_device_error(*m_inventory[index], state) : std::nullopt,
            }));

        const auto make_input = [this, &state, index, ingress_fresh, inventory_present, now](
                                    ::avatar::DeviceDataCategory category, bool enabled)
        {
            const size_t category_idx = data_category_index(category);
            const auto& dataset = state.datasets[category_idx];
            const bool has_no_current_data = dataset.fetch_error == ::avatar::ErrorCode::NO_VALID_DATA_RETURNED;
            std::optional<std::string> fetch_error;
            if (state.output_setup_error)
            {
                fetch_error = state.output_setup_error;
            }
            else if (dataset.fetch_error && !has_no_current_data)
            {
                fetch_error = "Avatar SDK fetch_data failed: " + std::string(error_name(*dataset.fetch_error)) + " (" +
                              std::to_string(static_cast<int>(*dataset.fetch_error)) + ")";
            }
            return DatasetHealthInput{
                .enabled = enabled,
                .inventory_available = m_inventory_available,
                .inventory_present = inventory_present,
                .ever_observed = m_ever_observed[index],
                .ingress_fresh = ingress_fresh,
                .ever_received_ingress = state.ever_received_ingress,
                .data_fresh = dataset.last_fresh_data && now - *dataset.last_fresh_data < kAvatarDataTimeout,
                .ever_received_data = dataset.ever_received_data,
                .fetch_has_no_current_data = has_no_current_data,
                .fetch_error = std::move(fetch_error),
            };
        };

        entries.push_back(make_dataset_status(
            kHandDevicePaths[index], make_input(::avatar::DeviceDataCategory::HUMAN, m_config.human)));
        entries.push_back(
            make_dataset_status(kRawDevicePaths[index], make_input(::avatar::DeviceDataCategory::RAW, m_config.raw)));
        entries.push_back(make_dataset_status(
            kRobotDevicePaths[index], make_input(::avatar::DeviceDataCategory::ROBOT, m_config.robot)));
    }

    m_status_publisher->publish_if_changed(entries, core::os_monotonic_now_ns());
}

const GloveState& AvatarTracker::glove(::avatar::DeviceSide side) const
{
    return m_gloves[side_index(side)];
}

GloveState& AvatarTracker::glove(::avatar::DeviceSide side)
{
    return m_gloves[side_index(side)];
}

void AvatarTracker::update()
{
    refresh_transport_state();
    refresh_sdk_inventory();
    try_connect_missing_gloves();
    refresh_glove_activity();
    m_deviceio_session->update();
    refresh_data();
    process_haptics();
    try
    {
        publish_device_status();
        m_status_publish_error_logged = false;
    }
    catch (const std::exception& error)
    {
        if (!m_status_publish_error_logged)
        {
            logger()->warn("Device status publishing failed; tracking will continue: {}", error.what());
            m_status_publish_error_logged = true;
        }
    }
    catch (...)
    {
        if (!m_status_publish_error_logged)
        {
            logger()->warn("Device status publishing failed; tracking will continue.");
            m_status_publish_error_logged = true;
        }
    }
    if (m_config.raw || m_config.robot)
    {
        push_joint_frames();
    }
    if (m_config.human)
    {
        inject_hand_data();
    }
}

void AvatarTracker::process_haptics()
{
    if (!m_haptic_reader)
    {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    for (const ::avatar::DeviceSide side : kDeviceSides)
    {
        const size_t index = side_index(side);
        const auto& tracked = m_haptic_reader->get_data(*m_deviceio_session, to_string(side));
        HapticCommandDeliveryState& delivery = m_haptic_commands[index];
        delivery.observe(tracked);
        const core::HapticCommand* command = delivery.pending();
        if (command != nullptr)
        {
            const bool retry_ready = !m_haptic_retry_after[index] || now >= *m_haptic_retry_after[index];
            if (retry_ready)
            {
                if (command->values() == nullptr || command->values()->size() != kAvatarFingerCount)
                {
                    delivery.discard();
                }
                else
                {
                    std::array<float, kAvatarFingerCount> powers{};
                    for (size_t i = 0; i < kAvatarFingerCount; ++i)
                    {
                        powers[i] = command->values()->Get(i);
                    }
                    if (apply_haptic_command(side, powers))
                    {
                        delivery.acknowledge();
                        m_last_haptic_sample_times[index] = now;
                        m_haptic_stopped[index] =
                            std::all_of(powers.begin(), powers.end(),
                                        [](float power) { return !std::isfinite(power) || power <= 0.0f; });
                    }
                }
            }
        }
        if (!m_haptic_stopped[index] && m_last_haptic_sample_times[index] &&
            now - *m_last_haptic_sample_times[index] >= kHapticCommandTimeout &&
            (!m_haptic_retry_after[index] || now >= *m_haptic_retry_after[index]))
        {
            stop_haptics(side);
        }
    }
}

bool AvatarTracker::dataset_enabled(::avatar::DeviceDataCategory category) const
{
    switch (category)
    {
    case ::avatar::DeviceDataCategory::RAW:
        return m_config.raw;
    case ::avatar::DeviceDataCategory::ROBOT:
        return m_config.robot;
    case ::avatar::DeviceDataCategory::HUMAN:
        return m_config.human;
    }
    return false;
}

void AvatarTracker::push_joint_frames()
{
    for (const ::avatar::DeviceSide side : kDeviceSides)
    {
        for (const ::avatar::DeviceDataCategory category : kJointDataCategories)
        {
            const auto& pusher = m_joint_pushers[side_index(side)][joint_category_index(category)];
            if (pusher)
            {
                push_joint_frame(glove(side), side, category, *pusher);
            }
        }
    }
}

void AvatarTracker::push_joint_frame(const GloveState& glove,
                                     ::avatar::DeviceSide side,
                                     ::avatar::DeviceDataCategory category,
                                     core::SchemaPusher& pusher)
{
    const ::avatar::AvatarDataFrame& frame = cached_joint_frame(glove, category);
    if (frame.payload_case() != payload_case(category))
    {
        return;
    }
    const ::avatar::Hand& hand = hand_payload(frame, category);
    if (hand.joint.position.empty())
    {
        return;
    }

    core::JointStateOutputT output;
    output.device_id = collection_id(side, category);
    output.has_velocity = false;
    output.has_effort = false;
    output.ee_pose_valid = false;
    output.joints.reserve(hand.joint.position.size());
    const auto& names = joint_names(category);

    for (size_t i = 0; i < hand.joint.position.size(); ++i)
    {
        auto joint = std::make_shared<core::JointStateT>();
        joint->name = i < names.size() ? std::string(names[i]) : std::string{};
        joint->position = hand.joint.position[i];
        joint->valid = std::isfinite(joint->position);
        output.joints.push_back(std::move(joint));
    }

    const auto sample_time_ns = core::os_monotonic_now_ns();
    flatbuffers::FlatBufferBuilder builder(kJointFlatbufferSize);
    const auto offset = core::JointStateOutput::Pack(builder, &output);
    builder.Finish(offset);
    pusher.push_buffer(builder.GetBufferPointer(), builder.GetSize(), sample_time_ns, sample_time_ns);
}

bool AvatarTracker::apply_haptic_command(::avatar::DeviceSide side, const std::array<float, kAvatarFingerCount>& powers)
{
    if (!glove(side).ready)
    {
        return false;
    }
    return send_haptic_command(side, powers);
}

bool AvatarTracker::send_haptic_command(::avatar::DeviceSide side, const std::array<float, kAvatarFingerCount>& powers)
{
    GloveState& state = glove(side);
    if (!state.device)
    {
        return false;
    }

    try
    {
        std::string request = R"({"key":"set_vibration","intensity":[)";
        for (size_t i = 0; i < kAvatarFingerCount; ++i)
        {
            const float normalized = std::isfinite(powers[i]) ? std::clamp(powers[i], 0.0f, 1.0f) : 0.0f;
            if (i != 0)
            {
                request += ',';
            }
            request += std::to_string(static_cast<int>(std::lround(normalized * 255.0f)));
        }
        request += "]}";

        const auto [error, unused] = state.device->set_task(request);
        (void)unused;
        if (error != ::avatar::ErrorCode::SUCCESS)
        {
            state.haptic_error = "set_vibration failed: " + std::string(error_name(error)) + " (" +
                                 std::to_string(static_cast<int>(error)) + ")";
            const size_t index = side_index(side);
            m_haptic_retry_after[index] = std::chrono::steady_clock::now() + kHapticFailureRetryInterval;
            if (!m_haptic_error_logged[index])
            {
                m_haptic_error_logged[index] = true;
                logger()->warn(
                    "set_vibration failed for {} glove: {} ({}); further errors for this side will be silenced.",
                    to_string(side), error_name(error), static_cast<int>(error));
            }
            return false;
        }
        state.haptic_error.reset();
        const size_t index = side_index(side);
        m_haptic_retry_after[index].reset();
        m_haptic_error_logged[index] = false;
        return true;
    }
    catch (const std::exception& error)
    {
        state.haptic_error = "set_vibration threw: " + std::string(error.what());
        const size_t index = side_index(side);
        m_haptic_retry_after[index] = std::chrono::steady_clock::now() + kHapticFailureRetryInterval;
        if (!m_haptic_error_logged[index])
        {
            m_haptic_error_logged[index] = true;
            logger()->warn("set_vibration threw for {} glove: {}; further errors for this side will be silenced.",
                           to_string(side), error.what());
        }
    }
    catch (...)
    {
        state.haptic_error = "set_vibration threw an unknown error";
        const size_t index = side_index(side);
        m_haptic_retry_after[index] = std::chrono::steady_clock::now() + kHapticFailureRetryInterval;
        if (!m_haptic_error_logged[index])
        {
            m_haptic_error_logged[index] = true;
            logger()->warn(
                "set_vibration threw for {} glove; further errors for this side will be silenced.", to_string(side));
        }
    }
    return false;
}

void AvatarTracker::stop_haptics(::avatar::DeviceSide side) noexcept
{
    const size_t index = side_index(side);
    if (!m_config.haptic || !glove(side).device)
    {
        m_haptic_stopped[index] = true;
        m_last_haptic_sample_times[index].reset();
        m_haptic_retry_after[index].reset();
        return;
    }

    try
    {
        m_haptic_stopped[index] = send_haptic_command(side, {});
    }
    catch (...)
    {
        m_haptic_stopped[index] = false;
        m_haptic_retry_after[index] = std::chrono::steady_clock::now() + kHapticFailureRetryInterval;
    }
    if (m_haptic_stopped[index])
    {
        m_last_haptic_sample_times[index].reset();
    }
    else
    {
        m_last_haptic_sample_times[index] = std::chrono::steady_clock::now();
    }
}

void AvatarTracker::map_landmarks_to_openxr(const std::vector<::avatar::Pose>& landmarks,
                                            const XrPosef& root_pose,
                                            bool is_root_tracked,
                                            XrHandJointLocationEXT out_joints[XR_HAND_JOINT_COUNT_EXT]) const
{
    for (uint32_t j = 0; j < XR_HAND_JOINT_COUNT_EXT; ++j)
    {
        out_joints[j] = { 0 };
        const std::optional<size_t> source_index = kOpenXRSlotSourceIndices[j];
        if (!source_index || *source_index >= landmarks.size())
        {
            continue;
        }
        const auto& lp = landmarks[*source_index];

        XrPosef local_pose;
        local_pose.position.x = lp.position.x;
        local_pose.position.y = lp.position.y;
        local_pose.position.z = lp.position.z;
        local_pose.orientation.x = lp.orientation.x;
        local_pose.orientation.y = lp.orientation.y;
        local_pose.orientation.z = lp.orientation.z;
        local_pose.orientation.w = lp.orientation.w;

        out_joints[j].pose = oxr_utils::multiply_poses(root_pose, local_pose);
        out_joints[j].radius = 0.01f;
        out_joints[j].locationFlags = XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;

        if (is_root_tracked)
        {
            out_joints[j].locationFlags |=
                XR_SPACE_LOCATION_POSITION_TRACKED_BIT | XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT;
        }
    }
}

void AvatarTracker::inject_hand_data()
{
    const XrTime time = m_time_converter->os_monotonic_now();

    const auto process_hand = [this, time](::avatar::DeviceSide side)
    {
        const std::vector<::avatar::Pose>& landmarks = glove(side).landmarks;
        if (landmarks.empty())
        {
            return;
        }

        // No optical/controller wrist: inject wrist-relative at identity with
        // VALID-only flags so HandsSource still has a skeleton to draw.
        const plugin_utils::WristSample wrist = m_wrist_source->query(side == ::avatar::DeviceSide::LEFT, time);
        const XrPosef root = wrist.valid ? wrist.pose : oxr_utils::identity_posef();

        XrHandJointLocationEXT joints[XR_HAND_JOINT_COUNT_EXT];
        map_landmarks_to_openxr(landmarks, root, wrist.valid && wrist.tracked, joints);

        plugin_utils::HandInjector* injector = m_injectors[side_index(side)].get();
        if (injector != nullptr)
        {
            injector->push(joints, time);
        }
    };

    for (const ::avatar::DeviceSide side : kDeviceSides)
    {
        process_hand(side);
    }
}

} // namespace avatar
} // namespace plugins
