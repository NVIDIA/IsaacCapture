// SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "omniglove_hand_conversion.hpp"

#include <deviceio_session/deviceio_session.hpp>
#include <log_bridge/logger.hpp>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/oxr_time.hpp>
#include <plugin_utils/hand_injector.hpp>
#include <plugin_utils/plugin_device_status_publisher.hpp>
#include <plugin_utils/wrist_pose_source.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

struct omniglove_session;
struct omniglove_stream_sample;

namespace plugins::omniglove
{

struct SdkSessionDeleter
{
    void operator()(omniglove_session* session) const noexcept;
};
using SdkSession = std::unique_ptr<omniglove_session, SdkSessionDeleter>;

class OmniglovePlugin
{
public:
    static constexpr auto kSourceInterval = std::chrono::milliseconds(8);

    explicit OmniglovePlugin(const std::string& plugin_root_id);
    ~OmniglovePlugin();

    void update();

private:
    struct HandFrame
    {
        std::array<XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT> joints{};
        int64_t sample_monotonic_ns = 0;
        uint64_t connection_generation = 0;
        uint64_t source_sequence = 0;
    };

    struct CallbackState
    {
        std::optional<HandFrame> frame;
        int64_t last_received_ns = 0;
    };

    struct StreamContext
    {
        OmniglovePlugin* plugin;
        bool is_left;
    };

    struct HandState
    {
        std::optional<StreamContext> stream;
        std::unique_ptr<plugin_utils::HandInjector> injector;
        int64_t last_pushed_ns = 0;
        core::PluginDeviceState state = core::PluginDeviceState_DISCONNECTED;
        core::PluginDeviceReason reason = core::PluginDeviceReason_NO_HARDWARE_SIGNAL;
    };

    static constexpr std::size_t hand_index(bool is_left)
    {
        return is_left ? 0 : 1;
    }

    HandState& hand_state(bool is_left);
    static void stream_callback(const omniglove_stream_sample* sample, void* user_data) noexcept;

    void start_device_streams();
    void on_sample(const omniglove_stream_sample& sample, bool is_left);
    void pump_hand(HandState& hand, bool is_left, const CallbackState& input);

    std::shared_ptr<spdlog::logger> m_logger =
        isaaccapture::Logger::get("isaaccapture.plugins.omniglove.OmniglovePlugin");

    std::unique_ptr<core::OpenXRSession> m_session;
    std::unique_ptr<core::DeviceIOSession> m_deviceio_session;
    std::unique_ptr<plugin_utils::WristPoseSource> m_wrist_source;
    std::optional<core::XrTimeConverter> m_time_converter;
    std::optional<plugin_utils::PluginDeviceStatusPublisher> m_status_publisher;

    std::mutex m_frame_mutex;
    std::array<HandState, 2> m_hands;
    std::array<CallbackState, 2> m_inputs;
    std::exception_ptr m_callback_error;
    // Destroy the SDK first: its callbacks borrow the contexts, frames and mutex above.
    SdkSession m_sdk_session;
};

} // namespace plugins::omniglove
