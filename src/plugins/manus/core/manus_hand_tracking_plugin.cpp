// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Bridges MANUS Core (integrated mode) to the IsaacTeleop DeviceIO layer for a
// pair of MANUS gloves.
//
// MANUS SDK callbacks fire on SDK threads and only cache state: skeleton joints,
// the glove ids discovered from the landscape, and raw flex-sensor transforms.
// All consumption happens on the plugin tick in update(), which drives three
// independently switchable datasets (see ManusPluginConfig):
//   human   - cached joints are anchored to a wrist pose from the OpenXR xdev
//             hand trackers, or a controller aim pose when xdev is unavailable,
//             and injected as XR_EXT_hand_tracking joints.
//   sensors - cached sensor transforms are pushed as JointSe3PoseOutput flatbuffers.
//   haptic  - inbound HapticCommands are forwarded to the glove finger motors.
//
// ManusTracker is a singleton because the SDK's C callbacks carry no user
// pointer and must reach the instance through instance().

#include "inc/manus/manus_hand_tracking_plugin.hpp"

#include "inc/manus/manus_glove_collection.hpp"

#include <flatbuffers/flatbuffers.h>
#include <oxr_utils/math.hpp>
#include <oxr_utils/os_time.hpp>
#include <oxr_utils/pose_conversions.hpp>
#include <schema/haptic_command_generated.h>
#include <schema/joint_se3_pose_generated.h>

#include <ManusSDK.h>
#include <ManusSDKTypeInitializers.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace plugins
{
namespace manus
{

namespace
{

SDKReturnCode get_raw_skeleton_node_count(uint32_t glove_id, uint32_t& node_count)
{
    return CoreSdk_GetRawSkeletonNodeCount(glove_id, &node_count);
}

// Must agree with JointSe3PoseTracker::DEFAULT_MAX_FLATBUFFER_SIZE on the consumer side
// (src/core/deviceio_trackers/trackers.toml).
constexpr size_t kSensorFlatbufferSize = 1024;
constexpr auto kSkeletonStaleThreshold = std::chrono::milliseconds(200);
// MANUS supplies 25 nodes. Injection expands them to OpenXR's 26 joints by
// deriving the palm from the final MANUS node.
constexpr size_t kMinimumUsableSkeletonNodes = static_cast<size_t>(XR_HAND_JOINT_COUNT_EXT) - 1;

int64_t steady_now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool has_current_usable_skeleton(const std::vector<SkeletonNode>& nodes,
                                 std::chrono::steady_clock::time_point stamp,
                                 std::chrono::steady_clock::time_point now)
{
    return nodes.size() >= kMinimumUsableSkeletonNodes && now - stamp <= kSkeletonStaleThreshold;
}

std::vector<unsigned char> read_calibration_file(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        throw std::runtime_error("Failed to open Manus calibration file: " + path);
    }

    const std::streamsize size = file.tellg();
    if (size <= 0 || static_cast<uint64_t>(size) > std::numeric_limits<uint32_t>::max())
    {
        throw std::runtime_error("Invalid Manus calibration file size: " + path);
    }

    std::vector<unsigned char> data(static_cast<size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(data.data()), size))
    {
        throw std::runtime_error("Failed to read Manus calibration file: " + path);
    }
    return data;
}

constexpr core::WristTrackingSourceConfig kWristSourceConfig{
    .mode = core::WristTrackingSourceMode::Auto,
    .left_aim_to_wrist = { { -0.70710678f, -0.5f, 0.0f, 0.5f }, { -0.1f, 0.02f, -0.02f } },
    .right_aim_to_wrist = { { -0.70710678f, 0.5f, 0.0f, 0.5f }, { 0.1f, 0.02f, -0.02f } }
};

} // anonymous namespace

ManusTracker& ManusTracker::instance(const ManusPluginConfig& config,
                                     const core::PluginSessionFactory& plugin_session_factory) noexcept(false)
{
    const ConstructionArgs args{ config, plugin_session_factory };
    return instance(&args);
}

ManusTracker& ManusTracker::instance(const ConstructionArgs* args)
{
    // The SDK callbacks pass null. They are registered while the first call constructs the
    // tracker, and the static's initialization guard makes their threads wait for it.
    static ManusTracker s(
        [args]() -> const ConstructionArgs&
        {
            assert(args != nullptr && "Manus SDK callbacks run only after instance(config, factory)");
            return *args;
        }());
    return s;
}

void ManusTracker::update()
{
    if (!m_plugin_session)
    {
        // The configured plugin session is unavailable, so no session-backed
        // pull, positioning, or output can be updated.
        return;
    }

    m_plugin_session->pull_channel().update();
    try
    {
        publish_device_status();
        m_status_publish_error_logged = false;
    }
    catch (const std::exception& error)
    {
        if (!m_status_publish_error_logged)
        {
            m_logger->warn("Device status publishing failed; tracking will continue: {}", error.what());
            m_status_publish_error_logged = true;
        }
    }
    catch (...)
    {
        if (!m_status_publish_error_logged)
        {
            m_logger->warn("Device status publishing failed; tracking will continue.");
            m_status_publish_error_logged = true;
        }
    }

    // Latest-wins per endpoint: the hardware only retains the most recent
    // vibration call, so dropping intermediate samples on a slow tick is fine.
    // The producer pushes an independent HapticCommand per hand on one
    // collection, so read each side separately -- reading a single latest sample
    // would let whichever hand was pushed last clobber the other. Non-5-finger
    // payloads are ignored (this plugin only drives 5-finger gloves).
    if (m_session_channels.haptic_reader)
    {
        for (const std::string_view endpoint : { std::string_view("left"), std::string_view("right") })
        {
            const auto& tracked = m_session_channels.haptic_reader->get_data(m_plugin_session->pull_channel(), endpoint);
            const core::HapticCommand* command = tracked.get();
            if (command != nullptr && command->values() != nullptr && command->values()->size() == kManusFingerCount)
            {
                std::array<float, kManusFingerCount> powers{};
                for (size_t i = 0; i < kManusFingerCount; ++i)
                {
                    powers[i] = command->values()->Get(i);
                }
                apply_haptic_command(endpoint == "left", powers);
            }
        }
    }

    if (m_config.sensors)
    {
        push_sensor_states();
    }

    if (m_config.human)
    {
        inject_hand_data();
    }
}

std::vector<SkeletonNode> ManusTracker::get_left_hand_nodes() const
{
    std::lock_guard<std::mutex> lock(m_skeleton_mutex);
    return m_left_hand_nodes;
}

std::vector<SkeletonNode> ManusTracker::get_right_hand_nodes() const
{
    std::lock_guard<std::mutex> lock(m_skeleton_mutex);
    return m_right_hand_nodes;
}

std::vector<NodeInfo> ManusTracker::get_left_node_info() const
{
    std::lock_guard<std::mutex> lock(m_skeleton_mutex);
    return m_left_node_info;
}

std::vector<NodeInfo> ManusTracker::get_right_node_info() const
{
    std::lock_guard<std::mutex> lock(m_skeleton_mutex);
    return m_right_node_info;
}

void ManusTracker::apply_haptic_command(bool is_left, const std::array<float, kManusFingerCount>& powers)
{
    uint32_t glove_id = 0;
    {
        std::lock_guard<std::mutex> lock(landscape_mutex);
        const auto& opt = is_left ? left_glove_id : right_glove_id;
        if (!opt.has_value())
        {
            // No glove connected on this side — silently no-op. Spamming the
            // log every frame while the glove is disconnected drowns out real
            // errors; the user already knows the glove is down because hand
            // tracking is unavailable.
            return;
        }
        glove_id = *opt;
    }

    // Clamp to [0, 1] — the Manus SDK does the same internally but
    // documenting the contract here lets retargeters with looser saturation
    // bounds wire up safely.
    std::array<float, kManusFingerCount> clamped{};
    for (size_t i = 0; i < clamped.size(); ++i)
    {
        // std::clamp passes NaN / ±Inf through unchanged, so sanitize first --
        // a non-finite power must never reach the SDK.
        clamped[i] = std::isfinite(powers[i]) ? std::clamp(powers[i], 0.0f, 1.0f) : 0.0f;
    }

    const SDKReturnCode rc = CoreSdk_VibrateFingersForGlove(glove_id, clamped.data());
    if (rc != SDKReturnCode::SDKReturnCode_Success)
    {
        const size_t slot = is_left ? 0 : 1;
        bool expected = false;
        if (m_haptic_error_logged[slot].compare_exchange_strong(expected, true))
        {
            m_logger->warn(
                "CoreSdk_VibrateFingersForGlove failed for {} glove (id={}, code={}); further errors for "
                "this side will be silenced.",
                is_left ? "left" : "right", glove_id, static_cast<int>(rc));
        }
    }
}

ManusTracker::ManusTracker(const ConstructionArgs& args) noexcept(false) : m_config(args.config)
{
    const core::PluginSessionFactory& plugin_session_factory = args.plugin_session_factory;
    if (m_config.monitoring_plugin_root_id && m_config.monitoring_plugin_root_id->empty())
    {
        throw std::invalid_argument("ManusTracker monitoring_plugin_root_id must not be empty");
    }

    const bool monitoring_enabled = m_config.monitoring_plugin_root_id.has_value();
    const bool needs_plugin_session = m_config.human || m_config.sensors || m_config.haptic || monitoring_enabled;
    if (needs_plugin_session && !plugin_session_factory)
    {
        throw std::invalid_argument("ManusTracker requires a plugin session factory for enabled session datasets");
    }

    if (!m_config.left_calibration_file.empty())
    {
        m_left_calibration_file = read_calibration_file(m_config.left_calibration_file);
    }
    if (!m_config.right_calibration_file.empty())
    {
        m_right_calibration_file = read_calibration_file(m_config.right_calibration_file);
    }

    // Registered before CoreSdk_InitializeIntegrated() so SDK-internal messages emitted
    // during init itself (not just the ones our own wrapper narrates below) are captured too.
    CoreSdk_RegisterCallbackForOnLog(OnLog);

    m_logger->info("Initializing SDK...");
    const SDKReturnCode t_InitializeResult = CoreSdk_InitializeIntegrated();
    if (t_InitializeResult != SDKReturnCode::SDKReturnCode_Success)
    {
        throw std::runtime_error("Failed to initialize Manus SDK, error code: " +
                                 std::to_string(static_cast<int>(t_InitializeResult)));
    }
    m_logger->info("SDK initialized successfully");
    m_logger->info("datasets: human={} sensors={} haptic={}", m_config.human ? "on" : "off",
                   m_config.sensors ? "on" : "off", m_config.haptic ? "on" : "off");

    RegisterCallbacks();

    CoordinateSystemVUH t_VUH;
    CoordinateSystemVUH_Init(&t_VUH);
    t_VUH.handedness = Side::Side_Right;
    t_VUH.up = AxisPolarity::AxisPolarity_PositiveY;
    t_VUH.view = AxisView::AxisView_ZToViewer;
    t_VUH.unitScale = 1.0f;

    m_logger->info("Setting up coordinate system (Y-up, right-handed, meters)...");
    const SDKReturnCode t_CoordinateResult = CoreSdk_InitializeCoordinateSystemWithVUH(t_VUH, true);

    if (t_CoordinateResult != SDKReturnCode::SDKReturnCode_Success)
    {
        throw std::runtime_error("Failed to initialize Manus SDK coordinate system, error code: " +
                                 std::to_string(static_cast<int>(t_CoordinateResult)));
    }
    m_logger->info("Coordinate system initialized successfully");

    ConnectToGloves();

    if (!needs_plugin_session)
    {
        m_logger->info("No session datasets enabled; running Manus-only (skeleton callbacks only).");
        return;
    }

    std::string error_msg = "Unknown error";
    bool success = false;

    try
    {
        std::vector<std::shared_ptr<core::ITracker>> trackers;
        if (m_config.haptic)
        {
            // The reader's buffer must be >= the producer's collection sample size; use the shared
            // default (matching the producer's PushTensorHapticDevice), not a Manus-specific size.
            m_session_channels.haptic_reader =
                std::make_shared<core::HapticCommandReaderTracker>(MANUS_GLOVE_COLLECTION_ID);
            trackers.push_back(m_session_channels.haptic_reader);
        }

        m_plugin_session = plugin_session_factory(
            core::PluginSessionRequirements{ .schema_push = m_config.sensors || monitoring_enabled,
                                             .hand_tracking_push = m_config.human,
                                             .wrist_tracking_pull =
                                                 m_config.human ? std::optional(kWristSourceConfig) : std::nullopt },
            std::move(trackers));

        if (monitoring_enabled)
        {
            m_session_channels.status_publisher = std::make_unique<plugin_utils::PluginDeviceStatusPublisher>(
                *m_plugin_session, *m_config.monitoring_plugin_root_id);
        }

        if (m_config.human)
        {
            m_session_channels.left_hand = m_plugin_session->create_hand_tracking_push_channel(XR_HAND_LEFT_EXT);
            m_session_channels.right_hand = m_plugin_session->create_hand_tracking_push_channel(XR_HAND_RIGHT_EXT);
        }

        if (m_config.sensors)
        {
            m_session_channels.left_sensor = m_plugin_session->create_schema_push_channel(
                core::SchemaPusherConfig{ .collection_id = MANUS_SENSORS_LEFT_COLLECTION_ID,
                                          .max_flatbuffer_size = kSensorFlatbufferSize,
                                          .tensor_identifier = "joint_se3_pose",
                                          .localized_name = "Manus Sensors Left" });
            m_session_channels.right_sensor = m_plugin_session->create_schema_push_channel(
                core::SchemaPusherConfig{ .collection_id = MANUS_SENSORS_RIGHT_COLLECTION_ID,
                                          .max_flatbuffer_size = kSensorFlatbufferSize,
                                          .tensor_identifier = "joint_se3_pose",
                                          .localized_name = "Manus Sensors Right" });
        }

        if (!m_config.human)
        {
            m_logger->info("Plugin session ready (human injection disabled).");
        }

        success = true;
    }
    catch (const std::exception& e)
    {
        error_msg = e.what();
    }

    if (!success)
    {
        m_logger->warn("Plugin session initialization failed: {}", error_msg);
        // Drop session-created objects before the session that owns their transport.
        m_session_channels = {};
        m_plugin_session.reset();

        if (monitoring_enabled)
        {
            shutdown_sdk();
            throw std::runtime_error("Managed Manus launch requires an OpenXR monitoring session: " + error_msg);
        }

        m_logger->warn(
            "Continuing in unmanaged Manus-only mode (no hand injection, sensor push, or session positioning).");
    }
}

ManusTracker::~ManusTracker()
{
    {
        std::lock_guard<std::mutex> lock(m_lifecycle_mutex);
        m_initialized = false;
    }

    shutdown_sdk();
}


void ManusTracker::shutdown_sdk()
{
    CoreSdk_RegisterCallbackForRawSkeletonStream(nullptr);
    CoreSdk_RegisterCallbackForLandscapeStream(nullptr);
    CoreSdk_RegisterCallbackForErgonomicsStream(nullptr);
    CoreSdk_RegisterCallbackForRawDeviceDataStream(nullptr);
    // Left registered through DisconnectFromGloves()/CoreSdk_ShutDown() so their own
    // shutdown-sequence log messages are still captured; unregistered last.
    DisconnectFromGloves();
    CoreSdk_ShutDown();
    CoreSdk_RegisterCallbackForOnLog(nullptr);
}

void ManusTracker::RegisterCallbacks()
{
    CoreSdk_RegisterCallbackForRawSkeletonStream(OnSkeletonStream);
    CoreSdk_RegisterCallbackForLandscapeStream(OnLandscapeStream);
    if (m_config.sensors)
    {
        CoreSdk_RegisterCallbackForRawDeviceDataStream(OnRawDeviceDataStream);
    }
}

void ManusTracker::ConnectToGloves() noexcept(false)
{
    bool connected = false;
    const int max_attempts = 30; // Maximum connection attempts
    const auto retry_delay = std::chrono::milliseconds(1000); // 1 second delay between attempts
    int attempts = 0;

    m_logger->info("Looking for Manus gloves...");

    while (!connected && attempts < max_attempts)
    {
        attempts++;

        if (const auto start_result = CoreSdk_LookForHosts(1, false); start_result != SDKReturnCode::SDKReturnCode_Success)
        {
            m_logger->warn("Failed to look for hosts (attempt {}/{})", attempts, max_attempts);
            std::this_thread::sleep_for(retry_delay);
            continue;
        }

        uint32_t number_of_hosts_found{};
        if (const auto number_result = CoreSdk_GetNumberOfAvailableHostsFound(&number_of_hosts_found);
            number_result != SDKReturnCode::SDKReturnCode_Success)
        {
            m_logger->warn("Failed to get number of available hosts (attempt {}/{})", attempts, max_attempts);
            std::this_thread::sleep_for(retry_delay);
            continue;
        }

        if (number_of_hosts_found == 0)
        {
            m_logger->warn("Failed to find hosts (attempt {}/{})", attempts, max_attempts);
            std::this_thread::sleep_for(retry_delay);
            continue;
        }

        std::vector<ManusHost> available_hosts(number_of_hosts_found);

        if (const auto hosts_result = CoreSdk_GetAvailableHostsFound(available_hosts.data(), number_of_hosts_found);
            hosts_result != SDKReturnCode::SDKReturnCode_Success)
        {
            m_logger->warn("Failed to get available hosts (attempt {}/{})", attempts, max_attempts);
            std::this_thread::sleep_for(retry_delay);
            continue;
        }

        if (const auto connect_result = CoreSdk_ConnectToHost(available_hosts[0]);
            connect_result == SDKReturnCode::SDKReturnCode_NotConnected)
        {
            m_logger->warn("Failed to connect to host (attempt {}/{})", attempts, max_attempts);
            std::this_thread::sleep_for(retry_delay);
            continue;
        }

        connected = true;
        is_connected = true;
        m_logger->info("Successfully connected to Manus host after {} attempts", attempts);
    }

    if (!connected)
    {
        m_logger->error("Failed to connect to Manus gloves after {} attempts", max_attempts);
        throw std::runtime_error("Failed to connect to Manus gloves");
    }
}

void ManusTracker::DisconnectFromGloves()
{
    if (is_connected)
    {
        CoreSdk_Disconnect();
        is_connected = false;
        m_logger->info("Disconnected from Manus gloves");
    }
}

bool ManusTracker::apply_glove_calibration(uint32_t glove_id, bool is_left)
{
    auto& calibration_file = is_left ? m_left_calibration_file : m_right_calibration_file;
    if (calibration_file.empty())
    {
        return true;
    }

    SetGloveCalibrationReturnCode result = SetGloveCalibrationReturnCode_Error;
    const SDKReturnCode rc = CoreSdk_SetGloveCalibration(
        glove_id, calibration_file.data(), static_cast<uint32_t>(calibration_file.size()), &result);
    if (rc != SDKReturnCode::SDKReturnCode_Success || result != SetGloveCalibrationReturnCode_Success)
    {
        m_logger->warn("Failed to apply {} glove calibration file (glove id={}, SDK code={}, result={})",
                       is_left ? "left" : "right", glove_id, static_cast<int>(rc), static_cast<int>(result));
        return false;
    }

    m_logger->debug("Applied {} glove calibration file to glove id={}", is_left ? "left" : "right", glove_id);
    return true;
}

void ManusTracker::OnSkeletonStream(const SkeletonStreamInfo* skeleton_stream_info)
{
    auto& tracker = instance(nullptr);
    std::lock_guard<std::mutex> instance_lock(tracker.m_lifecycle_mutex);
    if (!tracker.m_initialized)
    {
        return;
    }

    for (uint32_t i = 0; i < skeleton_stream_info->skeletonsCount; i++)
    {
        RawSkeletonInfo skeleton_info;
        if (CoreSdk_GetRawSkeletonInfo(i, &skeleton_info) != SDKReturnCode::SDKReturnCode_Success)
        {
            continue;
        }

        std::vector<SkeletonNode> nodes(skeleton_info.nodesCount);
        skeleton_info.publishTime = skeleton_stream_info->publishTime;
        const bool skeleton_read_succeeded = CoreSdk_GetRawSkeletonData(i, nodes.data(), skeleton_info.nodesCount) ==
                                             SDKReturnCode::SDKReturnCode_Success;

        uint32_t glove_id = skeleton_info.gloveId;

        // Check if glove ID matches any known glove
        bool is_left_glove, is_right_glove;
        {
            std::lock_guard<std::mutex> landscape_lock(tracker.landscape_mutex);
            is_left_glove = tracker.left_glove_id && glove_id == *tracker.left_glove_id;
            is_right_glove = tracker.right_glove_id && glove_id == *tracker.right_glove_id;
        }

        if (!is_left_glove && !is_right_glove)
        {
            tracker.m_logger->warn("Skipping data from unknown glove ID: {}", glove_id);
            continue;
        }

        std::string prefix = is_left_glove ? "left" : "right";

        // Save data for OpenXR Injection
        {
            std::lock_guard<std::mutex> lock(tracker.m_skeleton_mutex);
            if (!skeleton_read_succeeded)
            {
                tracker.m_skeleton_stamps[is_left_glove ? 0 : 1] = {};
                continue;
            }

            if (is_left_glove)
            {
                tracker.m_left_hand_nodes = nodes;
                tracker.m_skeleton_stamps[0] = std::chrono::steady_clock::now();
            }
            else if (is_right_glove)
            {
                tracker.m_right_hand_nodes = nodes;
                tracker.m_skeleton_stamps[1] = std::chrono::steady_clock::now();
            }
        }
    }
}

void ManusTracker::OnLandscapeStream(const Landscape* landscape)
{
    auto& tracker = instance(nullptr);
    std::lock_guard<std::mutex> instance_lock(tracker.m_lifecycle_mutex);
    if (!tracker.m_initialized)
    {
        return;
    }

    const auto& gloves = landscape->gloveDevices;

    std::lock_guard<std::mutex> landscape_lock(tracker.landscape_mutex);

    // We only support one left and one right glove
    if (gloves.gloveCount > 2)
    {
        tracker.m_logger->warn("Invalid number of gloves detected: {}", gloves.gloveCount);
        return;
    }

    // Extract glove IDs from landscape data
    bool left_present = false;
    bool right_present = false;
    for (uint32_t i = 0; i < gloves.gloveCount; i++)
    {
        const GloveLandscapeData& glove = gloves.gloves[i];
        if (glove.side == Side::Side_Left)
        {
            left_present = true;
            if (tracker.left_glove_id != glove.id)
            {
                tracker.left_glove_id = glove.id;
                tracker.m_calibration_failed[0] = !tracker.apply_glove_calibration(glove.id, true);
                tracker.m_logger->info("Left glove connected (ID {}); start streaming.", glove.id);
            }
            // Fetch bone topology once on connect
            uint32_t nc = 0;
            if (get_raw_skeleton_node_count(glove.id, nc) == SDKReturnCode::SDKReturnCode_Success && nc > 0)
            {
                std::lock_guard<std::mutex> sk(tracker.m_skeleton_mutex);
                tracker.m_left_node_info.resize(nc);
                if (CoreSdk_GetRawSkeletonNodeInfoArray(glove.id, tracker.m_left_node_info.data(), nc) !=
                    SDKReturnCode::SDKReturnCode_Success)
                    tracker.m_left_node_info.clear();
            }
        }
        else if (glove.side == Side::Side_Right)
        {
            right_present = true;
            if (tracker.right_glove_id != glove.id)
            {
                tracker.right_glove_id = glove.id;
                tracker.m_calibration_failed[1] = !tracker.apply_glove_calibration(glove.id, false);
                tracker.m_logger->info("Right glove connected (ID {}); start streaming.", glove.id);
            }
            uint32_t nc = 0;
            if (get_raw_skeleton_node_count(glove.id, nc) == SDKReturnCode::SDKReturnCode_Success && nc > 0)
            {
                std::lock_guard<std::mutex> sk(tracker.m_skeleton_mutex);
                tracker.m_right_node_info.resize(nc);
                if (CoreSdk_GetRawSkeletonNodeInfoArray(glove.id, tracker.m_right_node_info.data(), nc) !=
                    SDKReturnCode::SDKReturnCode_Success)
                    tracker.m_right_node_info.clear();
            }
        }
    }

    // Clear stale state for any glove that is no longer present in this landscape
    // update (i.e., disconnected). Resetting the IDs prevents OnSkeletonStream from
    // matching future packets to a dead glove, and clearing the node cache prevents
    // inject_hand_data() from replaying the last known stale pose indefinitely.
    {
        std::lock_guard<std::mutex> skeleton_lock(tracker.m_skeleton_mutex);
        if (!left_present && tracker.left_glove_id.has_value())
        {
            tracker.m_logger->warn("Left glove disconnected (ID {})", *tracker.left_glove_id);
            tracker.left_glove_id.reset();
            tracker.m_calibration_failed[0] = false;
            tracker.m_left_hand_nodes.clear();
            tracker.m_skeleton_stamps[0] = {};
            tracker.m_left_node_info.clear();
            {
                std::lock_guard<std::mutex> sensor_lock(tracker.m_sensor_mutex);
                tracker.m_sensor_count[0] = 0;
            }
        }
        if (!right_present && tracker.right_glove_id.has_value())
        {
            tracker.m_logger->warn("Right glove disconnected (ID {})", *tracker.right_glove_id);
            tracker.right_glove_id.reset();
            tracker.m_calibration_failed[1] = false;
            tracker.m_right_hand_nodes.clear();
            tracker.m_skeleton_stamps[1] = {};
            tracker.m_right_node_info.clear();
            {
                std::lock_guard<std::mutex> sensor_lock(tracker.m_sensor_mutex);
                tracker.m_sensor_count[1] = 0;
            }
        }
    }
}

void ManusTracker::OnLog(LogSeverity p_Severity, const char* p_Log, uint32_t p_Length) noexcept
{
    // Handed to the Manus SDK as a C callback, so nothing may escape: constructing the
    // std::string can throw bad_alloc, and Logger::get() allocates and touches spdlog's
    // registry on first use. Unwinding into the SDK's own frames is undefined behaviour,
    // and this fires from inside CoreSdk_InitializeIntegrated(). Dropping one vendor log
    // line is the acceptable outcome; there is nowhere to report the failure, because
    // logging is the thing that failed.
    try
    {
        // Deliberately does not go through instance(): this can fire synchronously from
        // CoreSdk_InitializeIntegrated(), called mid-constructor, before the function-local
        // static in instance() has finished constructing -- reentering that initialization
        // from the same thread is undefined behavior. Look the logger up directly by name
        // instead.
        //
        // Keep the SDK stream distinct from the plugin's own diagnostics so readers,
        // set_console_logger_name_filter() and set_logger_colors() can distinguish them.
        static const auto logger = isaaccapture::Logger::get("isaaccapture.plugins.manus.ManusSdk");
        const std::string message(p_Log, p_Length);

        // Inherit the SDK's own severity rather than collapsing everything to one level.
        switch (p_Severity)
        {
        case LogSeverity_Debug:
            logger->debug("{}", message);
            break;
        case LogSeverity_Info:
            logger->info("{}", message);
            break;
        case LogSeverity_Warn:
            logger->warn("{}", message);
            break;
        case LogSeverity_Error:
            logger->error("{}", message);
            break;
        default:
            logger->info("{}", message);
            break;
        }
    }
    catch (...)
    {
    }
}

void ManusTracker::OnRawDeviceDataStream(const RawDeviceDataInfo* raw_device_data_info)
{
    auto& tracker = instance(nullptr);
    std::lock_guard<std::mutex> instance_lock(tracker.m_lifecycle_mutex);
    if (!tracker.m_initialized || !tracker.m_config.sensors)
    {
        return;
    }

    for (uint32_t i = 0; i < raw_device_data_info->rawDeviceDataCount; ++i)
    {
        RawDeviceData raw{};
        if (CoreSdk_GetRawDeviceData(i, &raw) != SDKReturnCode::SDKReturnCode_Success)
        {
            continue;
        }

        bool is_left = false;
        bool is_right = false;
        {
            std::lock_guard<std::mutex> landscape_lock(tracker.landscape_mutex);
            is_left = tracker.left_glove_id && raw.id == *tracker.left_glove_id;
            is_right = tracker.right_glove_id && raw.id == *tracker.right_glove_id;
        }
        if (!is_left && !is_right)
        {
            continue;
        }

        const size_t side = is_left ? 0 : 1;
        if (raw.sensorCount == 0)
        {
            // Clear cached count so push_sensor_side stops emitting stale tips.
            std::lock_guard<std::mutex> sensor_lock(tracker.m_sensor_mutex);
            tracker.m_sensor_count[side] = 0;
            continue;
        }

        const uint32_t count = std::min(raw.sensorCount, static_cast<uint32_t>(kManusSensorCount));
        std::lock_guard<std::mutex> sensor_lock(tracker.m_sensor_mutex);
        tracker.m_sensor_count[side] = count;
        for (uint32_t j = 0; j < count; ++j)
        {
            tracker.m_sensor_transforms[side][j] = raw.sensorData[j];
        }
    }
}

void ManusTracker::push_sensor_states()
{
    if (m_session_channels.left_sensor)
    {
        push_sensor_side(true, *m_session_channels.left_sensor);
    }
    if (m_session_channels.right_sensor)
    {
        push_sensor_side(false, *m_session_channels.right_sensor);
    }
}

void ManusTracker::publish_device_status()
{
    if (!m_session_channels.status_publisher)
    {
        return;
    }

    std::vector<plugin_utils::PluginDeviceStatusEntry> entries;
    entries.reserve(2);

    if (!m_config.human)
    {
        for (const char* path : { "/hand/left", "/hand/right" })
        {
            entries.push_back(plugin_utils::PluginDeviceStatusEntry{
                .path = path,
                .state = core::PluginDeviceState_DISABLED,
                .reason = core::PluginDeviceReason_DISABLED_BY_CONFIGURATION,
            });
        }
    }
    else
    {
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> landscape_lock(landscape_mutex);
        std::lock_guard<std::mutex> skeleton_lock(m_skeleton_mutex);
        const auto append_side = [&entries](
                                     const char* path, bool present, bool calibration_failed, bool has_current_skeleton)
        {
            plugin_utils::PluginDeviceStatusEntry entry{ .path = path };
            if (!present)
            {
                entry.state = core::PluginDeviceState_DISCONNECTED;
                entry.reason = core::PluginDeviceReason_NO_HARDWARE_SIGNAL;
            }
            else if (calibration_failed)
            {
                entry.state = core::PluginDeviceState_DEGRADED;
                entry.reason = core::PluginDeviceReason_CALIBRATION_FAILED;
                entry.error = "glove calibration failed";
            }
            else if (!has_current_skeleton)
            {
                entry.state = core::PluginDeviceState_DEGRADED;
                entry.reason = core::PluginDeviceReason_NO_CURRENT_DATA;
                entry.error = "glove present but has no current usable skeleton data";
            }
            else
            {
                entry.state = core::PluginDeviceState_CONNECTED;
                entry.reason = core::PluginDeviceReason_NONE;
            }
            entries.push_back(std::move(entry));
        };
        append_side("/hand/left", left_glove_id.has_value(), m_calibration_failed[0],
                    has_current_usable_skeleton(m_left_hand_nodes, m_skeleton_stamps[0], now));
        append_side("/hand/right", right_glove_id.has_value(), m_calibration_failed[1],
                    has_current_usable_skeleton(m_right_hand_nodes, m_skeleton_stamps[1], now));
    }

    m_session_channels.status_publisher->publish_if_changed(entries, steady_now_ns());
}

void ManusTracker::push_sensor_side(bool is_left, core::ISchemaPushChannel& channel)
{
    const size_t side = is_left ? 0 : 1;
    uint32_t count = 0;
    std::array<ManusTransform, kManusSensorCount> transforms{};
    {
        std::lock_guard<std::mutex> sensor_lock(m_sensor_mutex);
        count = m_sensor_count[side];
        transforms = m_sensor_transforms[side];
    }

    // Hosts treat missing pushes as "no sensors"; only emit a full 5-tip pack.
    if (count < static_cast<uint32_t>(kManusSensorCount))
    {
        return;
    }

    std::vector<core::JointSe3Pose> joints;
    joints.reserve(static_cast<size_t>(kManusSensorCount));
    for (int sensor = 0; sensor < kManusSensorCount; ++sensor)
    {
        const ManusTransform& t = transforms[static_cast<size_t>(sensor)];
        joints.emplace_back(kManusSensorJoints[static_cast<size_t>(sensor)],
                            core::Pose(core::Point(t.position.x, t.position.y, t.position.z),
                                       core::Quaternion(t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w)));
    }

    const auto sample_time_ns = core::os_monotonic_now_ns();
    flatbuffers::FlatBufferBuilder builder(kSensorFlatbufferSize);
    const auto device_id =
        builder.CreateString(is_left ? MANUS_SENSORS_LEFT_COLLECTION_ID : MANUS_SENSORS_RIGHT_COLLECTION_ID);
    // Sort through the builder, not the object API: JointSe3PoseOutput::Pack() emits a keyed
    // struct vector with CreateVectorOfStructs and does NOT sort it, which would leave the
    // reader's LookupByKey searching an unsorted vector.
    const auto joints_offset = builder.CreateVectorOfSortedStructs(&joints);
    builder.Finish(core::CreateJointSe3PoseOutput(builder, core::JointType_HAND_RAW, joints_offset, device_id));
    channel.push_buffer(builder.GetBufferPointer(), builder.GetSize(), sample_time_ns, sample_time_ns);
}

void ManusTracker::inject_hand_data()
{
    std::vector<SkeletonNode> left_nodes;
    std::vector<SkeletonNode> right_nodes;
    std::array<std::chrono::steady_clock::time_point, 2> skeleton_stamps;

    {
        std::lock_guard<std::mutex> lock(m_skeleton_mutex);
        left_nodes = m_left_hand_nodes;
        right_nodes = m_right_hand_nodes;
        skeleton_stamps = m_skeleton_stamps;
    }

    const int64_t sample_time_ns = core::os_monotonic_now_ns();
    const auto now = std::chrono::steady_clock::now();

    auto process_hand =
        [&](const std::vector<SkeletonNode>& nodes, std::chrono::steady_clock::time_point stamp, bool is_left)
    {
        if (!has_current_usable_skeleton(nodes, stamp, now))
        {
            return;
        }

        XrHandJointLocationEXT joints[XR_HAND_JOINT_COUNT_EXT];
        XrPosef root_pose = { { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } };
        bool is_root_tracked = false;

        const core::WristTrackingSample wrist = m_plugin_session->pull_channel().wrist_tracking_source().query(
            is_left ? XR_HAND_LEFT_EXT : XR_HAND_RIGHT_EXT, sample_time_ns);
        if (wrist.valid)
        {
            if (is_left)
            {
                m_left_root_pose = wrist.pose;
            }
            else
            {
                m_right_root_pose = wrist.pose;
            }
            is_root_tracked = wrist.tracked;
        }

        root_pose = is_left ? m_left_root_pose : m_right_root_pose;
        uint32_t nodes_count = static_cast<uint32_t>(nodes.size());

        for (uint32_t j = 0; j < XR_HAND_JOINT_COUNT_EXT; j++)
        {
            // Determine source index in Manus array
            int manus_index = -1;

            if (j == XR_HAND_JOINT_PALM_EXT)
            {
                // OpenXR Palm -> Use Manus Palm (Last Index)
                if (nodes_count > 0)
                {
                    manus_index = nodes_count - 1;
                }
            }
            else if (j == XR_HAND_JOINT_WRIST_EXT)
            {
                // OpenXR Wrist -> Manus Wrist (Index 0)
                manus_index = 0;
            }
            else
            {
                // OpenXR Finger Joints (Indices 2..25) -> Manus Finger Joints (Indices 1..24)
                manus_index = j - 1;
            }

            if (manus_index >= 0 && manus_index < (int)nodes_count)
            {
                const auto& pos = nodes[manus_index].transform.position;
                const auto& rot = nodes[manus_index].transform.rotation;

                XrPosef local_pose;
                local_pose.position.x = pos.x;
                local_pose.position.y = pos.y;
                local_pose.position.z = pos.z;
                local_pose.orientation.x = rot.x;
                local_pose.orientation.y = rot.y;
                local_pose.orientation.z = rot.z;
                local_pose.orientation.w = rot.w;

                joints[j].pose = oxr_utils::multiply_poses(root_pose, local_pose);

                joints[j].radius = 0.01f;
                joints[j].locationFlags = XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;

                if (is_root_tracked)
                {
                    joints[j].locationFlags |=
                        XR_SPACE_LOCATION_POSITION_TRACKED_BIT | XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT;
                }
            }
            else
            {
                // Invalid joint if index out of bounds
                joints[j] = { 0 };
            }
        }

        if (is_left)
        {
            m_session_channels.left_hand->push(joints, sample_time_ns);
        }
        else
        {
            m_session_channels.right_hand->push(joints, sample_time_ns);
        }
    };

    process_hand(left_nodes, skeleton_stamps[0], true);
    process_hand(right_nodes, skeleton_stamps[1], false);
}

} // namespace manus
} // namespace plugins
