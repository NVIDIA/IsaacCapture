// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "manus_glove_collection.hpp"

#include <deviceio_trackers/haptic_command_reader_tracker.hpp>
#include <log_bridge/logger.hpp>
#include <openxr/openxr_platform.h>
#include <plugin_utils/plugin_device_status_publisher.hpp>
#include <pusherio/hand_tracking_push_channel.hpp>
#include <pusherio/plugin_session.hpp>
#include <pusherio/schema_pusher.hpp>

#include <ManusSDK.h>
#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace plugins
{
namespace manus
{

// Manus haptic gloves expose exactly five finger motors; the SDK's
// CoreSdk_VibrateFingersForGlove takes a fixed powers[5]. A glove with a
// different actuator count would change this and the values it consumes.
inline constexpr std::size_t kManusFingerCount = 5;

/// Runtime feature flags for manus_hand_plugin (parsed from --datasets=...).
struct ManusPluginConfig
{
    std::string app_name = "ManusHandPlugin";
    std::optional<std::string> monitoring_plugin_root_id;
    std::string left_calibration_file;
    std::string right_calibration_file;
    bool human = true; // hand tracking push
    bool sensors = true; // RawDeviceData -> SchemaPusher
    bool haptic = true; // inbound HapticCommandReaderTracker
};

using ManusPluginSessionFactory = std::function<std::unique_ptr<core::IPluginSession>()>;

class __attribute__((visibility("default"))) ManusTracker
{
public:
    /// Get the singleton instance. The first call constructs with ``config``;
    /// later calls (e.g. from Manus SDK callbacks) ignore all arguments.
    static ManusTracker& instance(const ManusPluginConfig& config = ManusPluginConfig{},
                                  ManusPluginSessionFactory plugin_session_factory = {},
                                  std::shared_ptr<core::HapticCommandReaderTracker> haptic_reader = {}) noexcept(false);

    void update();
    std::vector<SkeletonNode> get_left_hand_nodes() const;
    std::vector<SkeletonNode> get_right_hand_nodes() const;
    std::vector<NodeInfo> get_left_node_info() const;
    std::vector<NodeInfo> get_right_node_info() const;

    /// Vibrate the five finger motors of one haptic glove.
    ///
    /// `powers` is in Manus order [Thumb, Index, Middle, Ring, Pinky],
    /// values clamped to [0, 1]. Dispatched from `update()` once per frame
    /// off the latest HapticCommand the plugin received.
    ///
    /// No-ops (and logs at most once per side) when the glove is not
    /// connected, the glove reports no haptic support, or the SDK call
    /// returns a non-success code.
    ///
    /// Thread-safe — `landscape_mutex` guards the per-side glove id.
    void apply_haptic_command(bool is_left, const std::array<float, kManusFingerCount>& powers);

private:
    // Lifecycle
    ManusTracker(const ManusPluginConfig& config,
                 ManusPluginSessionFactory plugin_session_factory,
                 std::shared_ptr<core::HapticCommandReaderTracker> haptic_reader) noexcept(false);
    ~ManusTracker();

    ManusTracker(const ManusTracker&) = delete;
    ManusTracker& operator=(const ManusTracker&) = delete;
    ManusTracker(ManusTracker&&) = delete;
    ManusTracker& operator=(ManusTracker&&) = delete;
    void initialize() noexcept(false);
    void shutdown_sdk();

    // ManusSDK specific methods
    void RegisterCallbacks();
    void ConnectToGloves() noexcept(false);
    void DisconnectFromGloves();
    bool apply_glove_calibration(uint32_t glove_id, bool is_left);
    static void OnSkeletonStream(const SkeletonStreamInfo* skeleton_stream_info);
    static void OnLandscapeStream(const Landscape* landscape);
    static void OnRawDeviceDataStream(const RawDeviceDataInfo* raw_device_data_info);
    // Registered before CoreSdk_InitializeIntegrated() (see initialize()), so it may fire
    // synchronously from within that call, on the calling thread, before the ManusTracker
    // singleton (a function-local static in instance()) has finished constructing. Must
    // therefore never call instance() -- see the definition for how it gets its logger.
    static void OnLog(LogSeverity p_Severity, const char* p_Log, uint32_t p_Length) noexcept;

    void push_sensor_states();
    void push_sensor_side(bool is_left, core::SchemaPusher& pusher);
    void publish_device_status();

    // Hand-publishing methods
    void inject_hand_data();

    // -- Member Variables --

    ManusPluginConfig m_config;
    std::shared_ptr<spdlog::logger> m_logger = isaaccapture::Logger::get("isaaccapture.plugins.manus.ManusTracker");

    // Lifecycle
    std::mutex m_lifecycle_mutex;
    bool m_initialized = false;

    // ManusSDK State
    mutable std::mutex landscape_mutex;
    std::optional<uint32_t> left_glove_id;
    std::optional<uint32_t> right_glove_id;
    std::vector<unsigned char> m_left_calibration_file;
    std::vector<unsigned char> m_right_calibration_file;
    std::array<bool, 2> m_calibration_failed{ { false, false } };
    bool is_connected = false;

    // Haptic state — the per-side log-once flags use std::atomic to stay
    // quiet when many frames in a row fail (e.g. the glove was disconnected
    // mid-session). Only `apply_haptic_command` (non-const) writes here, so
    // no `mutable` is needed; const callers do not touch these flags.
    std::array<std::atomic<bool>, 2> m_haptic_error_logged{ { false, false } };

    // Flex-sensor cache (RawDeviceData). Indexed 0=left, 1=right.
    mutable std::mutex m_sensor_mutex;
    std::array<uint32_t, 2> m_sensor_count{ { 0, 0 } };
    std::array<std::array<ManusTransform, kManusSensorCount>, 2> m_sensor_transforms{};

    // Plugin session state
    ManusPluginSessionFactory m_plugin_session_factory;
    std::unique_ptr<core::IPluginSession> m_plugin_session;
    std::unique_ptr<core::IHandTrackingPushChannel> m_left_hand_channel;
    std::unique_ptr<core::IHandTrackingPushChannel> m_right_hand_channel;
    // Inbound HapticCommand tensor; collection identity in
    // inc/manus/manus_glove_collection.hpp. Read each frame in update().
    std::shared_ptr<core::HapticCommandReaderTracker> m_haptic_reader;
    core::IPluginPullChannel* m_pull_channel = nullptr; // owned by m_plugin_session
    std::unique_ptr<core::IWristTrackingSource> m_wrist_tracking_source;
    std::unique_ptr<core::SchemaPusher> m_left_sensor_pusher;
    std::unique_ptr<core::SchemaPusher> m_right_sensor_pusher;
    std::unique_ptr<plugin_utils::PluginDeviceStatusPublisher> m_status_publisher;
    bool m_status_publish_error_logged = false;

    // Persistent root poses (initialized to identity)
    XrPosef m_left_root_pose = { { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } };
    XrPosef m_right_root_pose = { { 0.0f, 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } };

    // Skeleton Data
    mutable std::mutex m_skeleton_mutex;
    std::vector<SkeletonNode> m_left_hand_nodes;
    std::vector<SkeletonNode> m_right_hand_nodes;
    std::array<std::chrono::steady_clock::time_point, 2> m_skeleton_stamps{};
    // Node topology (parent IDs) — populated once per glove connect
    std::vector<NodeInfo> m_left_node_info;
    std::vector<NodeInfo> m_right_node_info;
};

} // namespace manus
} // namespace plugins
