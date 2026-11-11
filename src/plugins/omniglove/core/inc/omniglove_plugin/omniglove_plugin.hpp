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
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

struct omniglove_session;
struct omniglove_stream_sample;

namespace plugins::omniglove
{

class OmniglovePlugin
{
public:
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

    struct HandState
    {
        std::optional<HandFrame> frame;
        std::unique_ptr<plugin_utils::HandInjector> injector;
        int64_t last_pushed_ns = 0;
        core::PluginDeviceState state = core::PluginDeviceState_DISCONNECTED;
        core::PluginDeviceReason reason = core::PluginDeviceReason_NO_HARDWARE_SIGNAL;
        bool sample_error_logged = false;
    };

    struct StreamContext
    {
        OmniglovePlugin* plugin;
        bool is_left;
    };

    struct SessionDeleter
    {
        void operator()(omniglove_session* session) const noexcept;
    };

    static void stream_callback(const omniglove_stream_sample* sample, void* user_data);

    void initialize_sdk();
    void start_device_streams();
    void on_sample(const omniglove_stream_sample& sample, bool is_left);
    void invalidate_hand(bool is_left);
    void pump_hand(HandState& hand, bool is_left, const std::optional<HandFrame>& frame);

    std::shared_ptr<spdlog::logger> m_logger =
        isaaccapture::Logger::get("isaaccapture.plugins.omniglove.OmniglovePlugin");

    std::unique_ptr<core::OpenXRSession> m_session;
    std::unique_ptr<core::DeviceIOSession> m_deviceio_session;
    std::unique_ptr<plugin_utils::WristPoseSource> m_wrist_source;
    std::optional<core::XrTimeConverter> m_time_converter;
    std::optional<plugin_utils::PluginDeviceStatusPublisher> m_status_publisher;

    std::mutex m_frame_mutex;
    std::array<HandState, 2> m_hands;
    std::vector<std::unique_ptr<StreamContext>> m_streams;
    // Destroy the SDK first: its callbacks borrow the contexts, frames and mutex above.
    std::unique_ptr<omniglove_session, SessionDeleter> m_sdk_session;
};

} // namespace plugins::omniglove
