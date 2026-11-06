// SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "omniglove_hand_conversion.hpp"

#include <deviceio_session/deviceio_session.hpp>
#include <log_bridge/logger.hpp>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/oxr_time.hpp>
#include <plugin_utils/hand_injector.hpp>
#include <plugin_utils/wrist_pose_source.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace plugins::omniglove
{

class OmniglovePlugin
{
public:
    explicit OmniglovePlugin(const std::string& plugin_root_id);
    ~OmniglovePlugin();

    bool is_running() const noexcept;
    bool has_failed() const noexcept;

    OmniglovePlugin(const OmniglovePlugin&) = delete;
    OmniglovePlugin& operator=(const OmniglovePlugin&) = delete;
    OmniglovePlugin(OmniglovePlugin&&) = delete;
    OmniglovePlugin& operator=(OmniglovePlugin&&) = delete;

private:
    struct HandFrame
    {
        std::array<XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT> joints{};
        uint64_t sample_monotonic_ns = 0;
        uint64_t connection_generation = 0;
        uint64_t source_sequence = 0;
        bool valid = false;
    };

    struct StreamContext
    {
        OmniglovePlugin* plugin = nullptr;
        bool is_left = false;
    };

    struct DeviceStream
    {
        omniglove_device_t* device = nullptr;
        StreamContext context;
        bool started = false;
    };

    struct SessionDeleter
    {
        void operator()(omniglove_session_t* session) const noexcept;
    };

    static void stream_callback(const omniglove_stream_sample_t* sample, void* user_data);

    void initialize_sdk();
    void start_device_streams();
    void on_sample(const omniglove_stream_sample_t& sample, bool is_left);
    void invalidate_hand(bool is_left);
    void worker_thread();
    void pump_hand(std::unique_ptr<plugin_utils::HandInjector>& injector,
                   XrHandEXT hand,
                   const HandFrame& frame,
                   uint64_t& last_pushed_ns);

    std::shared_ptr<spdlog::logger> m_logger =
        isaaccapture::Logger::get("isaaccapture.plugins.omniglove.OmniglovePlugin");

    std::shared_ptr<core::OpenXRSession> m_session;
    std::unique_ptr<core::DeviceIOSession> m_deviceio_session;
    std::unique_ptr<plugin_utils::WristPoseSource> m_wrist_source;
    std::unique_ptr<plugin_utils::HandInjector> m_left_injector;
    std::unique_ptr<plugin_utils::HandInjector> m_right_injector;
    std::unique_ptr<core::XrTimeConverter> m_time_converter;

    // The SDK session is declared after callback contexts so it stops and
    // joins callback workers before those contexts are destroyed.
    std::vector<std::unique_ptr<DeviceStream>> m_streams;
    std::unique_ptr<omniglove_session_t, SessionDeleter> m_sdk_session;

    std::mutex m_frame_mutex;
    HandFrame m_left;
    HandFrame m_right;
    uint64_t m_left_last_pushed_ns = 0;
    uint64_t m_right_last_pushed_ns = 0;

    std::thread m_worker;
    std::atomic<bool> m_running{ false };
    std::atomic<bool> m_failed{ false };
};

} // namespace plugins::omniglove
