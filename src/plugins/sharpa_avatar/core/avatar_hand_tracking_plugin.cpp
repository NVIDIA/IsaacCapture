// SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "avatar_hand_tracking_plugin_impl.hpp"

#include <flatbuffers/flatbuffers.h>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/math.hpp>
#include <oxr_utils/os_time.hpp>
#include <oxr_utils/pose_conversions.hpp>
#include <plugin_utils/hand_injector.hpp>
#include <schema/joint_state_generated.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace plugins
{
namespace avatar
{

namespace
{

// Returns true if the OpenXR loader/runtime advertises the given extension.
// Safe to call before any XrInstance exists.
bool is_openxr_extension_supported(const char* ext_name)
{
    uint32_t count = 0;
    if (XR_FAILED(xrEnumerateInstanceExtensionProperties(nullptr, 0, &count, nullptr)))
    {
        return false;
    }
    std::vector<XrExtensionProperties> props(count, XrExtensionProperties{ XR_TYPE_EXTENSION_PROPERTIES });
    if (XR_FAILED(xrEnumerateInstanceExtensionProperties(nullptr, count, &count, props.data())))
    {
        return false;
    }
    return std::any_of(props.begin(), props.end(),
                       [ext_name](const XrExtensionProperties& p) { return std::string(p.extensionName) == ext_name; });
}

constexpr size_t kAvatarFingerCount = 5;
constexpr size_t kJointFlatbufferSize = 4096;
constexpr auto kAvatarDataTimeout = std::chrono::seconds(10);
constexpr auto kOpenXRRetryInterval = std::chrono::seconds(10);
constexpr char kAvatarSdkConfigPath[] = "/opt/avatar-sdk/share/sdk_config.json";
std::atomic<bool> g_avatar_sdk_in_use{ false };

// Maps the 26 OpenXR XrHandJointEXT slots onto Avatar HUMAN landmark indices.
//
// OpenXR order (XrHandJointEXT): 0 PALM, 1 WRIST, then per finger
//   THUMB:  2 METACARPAL, 3 PROXIMAL, 4 DISTAL, 5 TIP
//   INDEX:  6 METACARPAL, 7 PROXIMAL, 8 INTERMEDIATE, 9 DISTAL, 10 TIP
//   MIDDLE: 11..15, RING: 16..20, LITTLE: 21..25 (same 5-slot layout)
//
// Avatar HUMAN landmark order (config/sdk_config.json `human_joint_names`, 25 entries):
//   0 WRIST
//   1 thumb_CMC_FE, 2 thumb_CMC_AA, 3 thumb_MCP_FE, 4 thumb_MCP_AA, 5 thumb_IP, 6 thumb_tip
//   7 index_MCP_AA, 8 index_MCP_FE, 9 index_PIP, 10 index_DIP, 11 index_tip
//   12 middle(AA,FE,PIP,DIP,tip)   17 ring(AA,FE,PIP,DIP,tip)   22 pinky(AA,FE,tip)
//
// The Avatar and OpenXR skeletons do not have identical joint counts per finger,
// so this is a best-effort correspondence. Validate/adjust with the
// avatar_hand_tracker_printer tool. PALM has no Avatar source (reuses WRIST).
const std::array<int, XR_HAND_JOINT_COUNT_EXT>& openxr_to_avatar_map()
{
    // clang-format off
    static const std::array<int, XR_HAND_JOINT_COUNT_EXT> kMap = {
        0,   // XR_HAND_JOINT_PALM_EXT           -> WRIST (no dedicated palm landmark)
        0,   // XR_HAND_JOINT_WRIST_EXT          -> WRIST

        1,   // THUMB_METACARPAL                 -> thumb_CMC_FE
        3,   // THUMB_PROXIMAL                   -> thumb_MCP_FE
        5,   // THUMB_DISTAL                     -> thumb_IP
        6,   // THUMB_TIP                        -> thumb_tip

        7,   // INDEX_METACARPAL                 -> index_MCP_AA
        8,   // INDEX_PROXIMAL                   -> index_MCP_FE
        9,   // INDEX_INTERMEDIATE              -> index_PIP
        10,  // INDEX_DISTAL                     -> index_DIP
        11,  // INDEX_TIP                        -> index_tip

        12,  // MIDDLE_METACARPAL                -> middle_MCP_AA
        13,  // MIDDLE_PROXIMAL                  -> middle_MCP_FE
        14,  // MIDDLE_INTERMEDIATE            -> middle_PIP
        15,  // MIDDLE_DISTAL                    -> middle_DIP
        16,  // MIDDLE_TIP                       -> middle_tip

        17,  // RING_METACARPAL                  -> ring_MCP_AA
        18,  // RING_PROXIMAL                    -> ring_MCP_FE
        19,  // RING_INTERMEDIATE              -> ring_PIP
        20,  // RING_DISTAL                      -> ring_DIP
        21,  // RING_TIP                         -> ring_tip

        22,        // LITTLE_METACARPAL          -> pinky_MCP_AA
        23,        // LITTLE_PROXIMAL            -> pinky_MCP_FE
        23,        // LITTLE_INTERMEDIATE      -> pinky_MCP_FE (no pinky PIP landmark)
        24,        // LITTLE_DISTAL              -> pinky_tip
        24,        // LITTLE_TIP                 -> pinky_tip
    };
    // clang-format on
    return kMap;
}

} // anonymous namespace

// Wrist offsets for XR controllers: align the controller aim pose to a natural
// wrist orientation for each hand.
static constexpr XrPosef kLeftHandOffset = { { -0.70710678f, -0.5f, 0.0f, 0.5f }, { -0.1f, 0.02f, -0.02f } };
static constexpr XrPosef kRightHandOffset = { { -0.70710678f, 0.5f, 0.0f, 0.5f }, { 0.1f, 0.02f, -0.02f } };

AvatarSdkSession::AvatarSdkSession(const std::string& config_path)
{
    bool expected = false;
    if (!g_avatar_sdk_in_use.compare_exchange_strong(expected, true))
    {
        throw std::runtime_error("Only one AvatarTracker may own the process-wide Avatar SDK");
    }

    const std::string effective_path = config_path.empty() ? kAvatarSdkConfigPath : config_path;
    const auto error = ::avatar::AvatarSDK::get_instance().initialize(effective_path);
    if (error != ::avatar::ErrorCode::SUCCESS)
    {
        g_avatar_sdk_in_use.store(false);
        throw std::runtime_error("Avatar SDK initialize failed, error code: " + std::to_string(static_cast<int>(error)));
    }
    m_initialized = true;
}

AvatarSdkSession::~AvatarSdkSession()
{
    if (m_initialized)
    {
        ::avatar::AvatarSDK::get_instance().destroy();
        g_avatar_sdk_in_use.store(false);
    }
}

::avatar::AvatarSDK& AvatarSdkSession::get()
{
    return ::avatar::AvatarSDK::get_instance();
}

GloveState::~GloveState()
{
    reset();
}

void GloveState::reset()
{
    if (device)
    {
        if (started)
        {
            device->stop();
        }
        device->destroy();
    }
    device.reset();
    started = false;
    last_successful_fetch = {};
}

AvatarTracker::Impl::Impl(AvatarPluginConfig config) : m_config(std::move(config)), m_sdk(m_config.sdk_config_path)
{
    std::cout << "[Avatar] Initializing Avatar SDK..." << std::endl;
    std::cout << "[Avatar] datasets: human=" << (m_config.human ? "on" : "off")
              << " raw=" << (m_config.raw ? "on" : "off") << " robot=" << (m_config.robot ? "on" : "off")
              << " haptic=" << (m_config.haptic ? "on" : "off") << std::endl;
    connect_gloves();
    try_initialize_openxr();
}

AvatarTracker::Impl::~Impl()
{
    reset_openxr();
}

void AvatarTracker::Impl::try_initialize_openxr()
{
    m_last_openxr_retry = std::chrono::steady_clock::now();
    std::string error_msg = "Unknown error";
    bool success = false;

    try
    {
        // ControllerTracker is always available; HandTracker requires
        // XR_EXT_hand_tracking so only add it when advertised.
        m_controller_tracker = std::make_shared<core::ControllerTracker>();
        std::vector<std::shared_ptr<core::ITracker>> trackers = { m_controller_tracker };

        const bool hand_tracking_supported = is_openxr_extension_supported(XR_EXT_HAND_TRACKING_EXTENSION_NAME);
        if (m_config.human && hand_tracking_supported)
        {
            m_hand_tracker = std::make_shared<core::HandTracker>();
            trackers.push_back(m_hand_tracker);
        }
        else if (m_config.human)
        {
            std::cout << "[Avatar] " << XR_EXT_HAND_TRACKING_EXTENSION_NAME
                      << " not supported by runtime; HandTracker will not be created." << std::endl;
        }

        if (m_config.haptic)
        {
            m_haptic_reader = std::make_shared<core::HapticCommandReaderTracker>(AVATAR_GLOVE_HAPTIC_COLLECTION_ID);
            trackers.push_back(m_haptic_reader);
        }

        std::vector<std::string> extensions = core::DeviceIOSession::get_required_extensions(trackers);
        if (m_config.raw || m_config.robot)
        {
            for (const auto& ext : core::SchemaPusher::get_required_extensions())
            {
                if (std::find(extensions.begin(), extensions.end(), ext) == extensions.end())
                {
                    extensions.push_back(ext);
                }
            }
        }
        extensions.push_back(XR_NVX1_DEVICE_INTERFACE_BASE_EXTENSION_NAME);

        const bool xdev_extension_supported = is_openxr_extension_supported(XR_MNDX_XDEV_SPACE_EXTENSION_NAME);
        if (xdev_extension_supported)
        {
            extensions.push_back(XR_MNDX_XDEV_SPACE_EXTENSION_NAME);
        }
        else
        {
            std::cout << "[Avatar] " << XR_MNDX_XDEV_SPACE_EXTENSION_NAME
                      << " not supported; optical hand tracking unavailable, using controller fallback." << std::endl;
        }

        const bool wait_for_openxr_system = false;
        m_session = std::make_shared<core::OpenXRSession>(m_config.app_name, extensions, wait_for_openxr_system);
        m_handles = m_session->get_handles();

        m_time_converter.emplace(m_handles);

        if (m_config.human)
        {
            m_left_injector = std::make_unique<plugin_utils::HandInjector>(
                m_handles.instance, m_handles.session, XR_HAND_LEFT_EXT, m_handles.space);
            m_right_injector = std::make_unique<plugin_utils::HandInjector>(
                m_handles.instance, m_handles.session, XR_HAND_RIGHT_EXT, m_handles.space);
        }

        auto make_joint_pusher = [this](const char* collection_id, const char* localized_name)
        {
            return std::make_unique<core::SchemaPusher>(
                m_handles, core::SchemaPusherConfig{ .collection_id = collection_id,
                                                     .max_flatbuffer_size = kJointFlatbufferSize,
                                                     .tensor_identifier = "joint_state",
                                                     .localized_name = localized_name });
        };
        if (m_config.raw)
        {
            m_left_raw_pusher = make_joint_pusher(AVATAR_RAW_LEFT_COLLECTION_ID, "Avatar RAW Left");
            m_right_raw_pusher = make_joint_pusher(AVATAR_RAW_RIGHT_COLLECTION_ID, "Avatar RAW Right");
        }
        if (m_config.robot)
        {
            m_left_robot_pusher = make_joint_pusher(AVATAR_ROBOT_LEFT_COLLECTION_ID, "Avatar ROBOT Left");
            m_right_robot_pusher = make_joint_pusher(AVATAR_ROBOT_RIGHT_COLLECTION_ID, "Avatar ROBOT Right");
        }

        m_deviceio_session = core::DeviceIOSession::run(trackers, m_handles);

        if (xdev_extension_supported)
        {
            initialize_xdev_hand_trackers();
        }

        std::cout << "[Avatar] Initialized with wrist source: " << (m_xdev_available ? "HandTracking" : "Controllers")
                  << std::endl;

        success = true;
    }
    catch (const std::exception& e)
    {
        error_msg = e.what();
    }

    if (!success)
    {
        reset_openxr();
        std::cerr << "[Avatar] Warning: OpenXR initialization failed: " << error_msg << std::endl;
        std::cerr << "[Avatar] Continuing in Avatar-only mode and retrying OpenXR." << std::endl;
    }
}

void AvatarTracker::Impl::reset_openxr()
{
    cleanup_xdev_hand_trackers();
    m_deviceio_session.reset();
    m_left_raw_pusher.reset();
    m_right_raw_pusher.reset();
    m_left_robot_pusher.reset();
    m_right_robot_pusher.reset();
    m_left_injector.reset();
    m_right_injector.reset();
    m_haptic_reader.reset();
    m_hand_tracker.reset();
    m_controller_tracker.reset();
    m_time_converter.reset();
    m_session.reset();
    m_handles = {};
}

void AvatarTracker::Impl::connect_gloves()
{
    std::cout << "[Avatar] Waiting for glove discovery..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));

    auto& sdk = m_sdk.get();
    m_left.reset();
    m_right.reset();
    m_left.device = sdk.get_device(::avatar::DeviceType::GLOVE, ::avatar::DeviceSide::LEFT);
    m_right.device = sdk.get_device(::avatar::DeviceType::GLOVE, ::avatar::DeviceSide::RIGHT);

    start_glove_if_present(m_left, "LEFT");
    start_glove_if_present(m_right, "RIGHT");

    if (!m_left.started && !m_right.started)
    {
        std::cout << "[Avatar] No online gloves yet; plugin will keep looking." << std::endl;
    }
}

void AvatarTracker::Impl::start_glove_if_present(GloveState& glove, const char* label)
{
    if (glove.started || !glove.device)
    {
        return;
    }

    const auto init_error = glove.device->init("{}");
    if (init_error != ::avatar::ErrorCode::SUCCESS)
    {
        std::cerr << "[Avatar] " << label << " glove initialization failed, error code " << static_cast<int>(init_error)
                  << std::endl;
        glove.reset();
        return;
    }
    glove.device->set_human_frame_build_enabled(m_config.human);

    const auto start_error = glove.device->start();
    if (start_error != ::avatar::ErrorCode::SUCCESS)
    {
        std::cerr << "[Avatar] " << label << " glove start failed, error code " << static_cast<int>(start_error)
                  << std::endl;
        glove.reset();
        return;
    }
    glove.started = true;
    glove.last_successful_fetch = std::chrono::steady_clock::now();
    std::cout << "[Avatar] " << label << " glove connected and streaming." << std::endl;
}

void AvatarTracker::Impl::try_connect_missing_gloves()
{
    if (m_left.started && m_right.started)
    {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    if (m_last_glove_retry.time_since_epoch().count() != 0 && now - m_last_glove_retry < std::chrono::seconds(2))
    {
        return;
    }
    m_last_glove_retry = now;

    auto& sdk = m_sdk.get();
    if (!m_left.started)
    {
        m_left.reset();
        m_left.device = sdk.get_device(::avatar::DeviceType::GLOVE, ::avatar::DeviceSide::LEFT);
        start_glove_if_present(m_left, "LEFT");
    }
    if (!m_right.started)
    {
        m_right.reset();
        m_right.device = sdk.get_device(::avatar::DeviceType::GLOVE, ::avatar::DeviceSide::RIGHT);
        start_glove_if_present(m_right, "RIGHT");
    }
    if (!m_left.started && !m_right.started &&
        (m_last_glove_wait_log.time_since_epoch().count() == 0 || now - m_last_glove_wait_log >= std::chrono::seconds(10)))
    {
        m_last_glove_wait_log = now;
        std::cout << "[Avatar] Waiting for an online glove..." << std::endl;
    }
}

void AvatarTracker::Impl::refresh_data()
{
    for (GloveState* glove : { &m_left, &m_right })
    {
        if (!glove->device || !glove->started)
        {
            continue;
        }
        if (!glove->device->get_device_info().online)
        {
            {
                std::lock_guard<std::mutex> lock(m_data_mutex);
                glove->landmarks.clear();
                glove->raw_frame = {};
                glove->robot_frame = {};
            }
            glove->reset();
            continue;
        }

        const bool expects_data = m_config.human || m_config.raw || m_config.robot;
        bool fetched_any = false;
        if (m_config.human)
        {
            ::avatar::AvatarDataFrame frame;
            const bool fetched =
                glove->device->fetch_data(frame, ::avatar::DeviceDataCategory::HUMAN) == ::avatar::ErrorCode::SUCCESS;
            fetched_any = fetched_any || fetched;
            std::lock_guard<std::mutex> lock(m_data_mutex);
            if (fetched)
            {
                glove->landmarks = std::move(frame.skeleton.landmark);
            }
            else
            {
                glove->landmarks.clear();
            }
        }
        if (m_config.raw)
        {
            ::avatar::AvatarDataFrame frame;
            const bool fetched =
                glove->device->fetch_data(frame, ::avatar::DeviceDataCategory::RAW) == ::avatar::ErrorCode::SUCCESS;
            fetched_any = fetched_any || fetched;
            std::lock_guard<std::mutex> lock(m_data_mutex);
            if (fetched)
            {
                glove->raw_frame = std::move(frame);
            }
            else
            {
                glove->raw_frame = {};
            }
        }
        if (m_config.robot)
        {
            ::avatar::AvatarDataFrame frame;
            const bool fetched =
                glove->device->fetch_data(frame, ::avatar::DeviceDataCategory::ROBOT) == ::avatar::ErrorCode::SUCCESS;
            fetched_any = fetched_any || fetched;
            std::lock_guard<std::mutex> lock(m_data_mutex);
            if (fetched)
            {
                glove->robot_frame = std::move(frame);
            }
            else
            {
                glove->robot_frame = {};
            }
        }
        if (expects_data)
        {
            const auto now = std::chrono::steady_clock::now();
            if (fetched_any)
            {
                glove->last_successful_fetch = now;
            }
            else if (now - glove->last_successful_fetch >= kAvatarDataTimeout)
            {
                glove->reset();
            }
        }
    }
}

std::vector<AvatarLandmark> AvatarTracker::Impl::get_left_landmarks() const
{
    std::lock_guard<std::mutex> lock(m_data_mutex);
    std::vector<AvatarLandmark> landmarks;
    landmarks.reserve(m_left.landmarks.size());
    for (const auto& landmark : m_left.landmarks)
    {
        landmarks.push_back({ .position = { landmark.position.x, landmark.position.y, landmark.position.z },
                              .orientation = { landmark.orientation.w, landmark.orientation.x, landmark.orientation.y,
                                               landmark.orientation.z } });
    }
    return landmarks;
}

std::vector<AvatarLandmark> AvatarTracker::Impl::get_right_landmarks() const
{
    std::lock_guard<std::mutex> lock(m_data_mutex);
    std::vector<AvatarLandmark> landmarks;
    landmarks.reserve(m_right.landmarks.size());
    for (const auto& landmark : m_right.landmarks)
    {
        landmarks.push_back({ .position = { landmark.position.x, landmark.position.y, landmark.position.z },
                              .orientation = { landmark.orientation.w, landmark.orientation.x, landmark.orientation.y,
                                               landmark.orientation.z } });
    }
    return landmarks;
}

AvatarJointFrame AvatarTracker::Impl::get_left_raw_frame() const
{
    std::lock_guard<std::mutex> lock(m_data_mutex);
    if (m_left.raw_frame.payload_case() != ::avatar::AvatarDataFrame::kRaw)
    {
        return {};
    }
    return { m_left.raw_frame.raw.joint.name, m_left.raw_frame.raw.joint.position };
}

AvatarJointFrame AvatarTracker::Impl::get_right_raw_frame() const
{
    std::lock_guard<std::mutex> lock(m_data_mutex);
    if (m_right.raw_frame.payload_case() != ::avatar::AvatarDataFrame::kRaw)
    {
        return {};
    }
    return { m_right.raw_frame.raw.joint.name, m_right.raw_frame.raw.joint.position };
}

AvatarJointFrame AvatarTracker::Impl::get_left_robot_frame() const
{
    std::lock_guard<std::mutex> lock(m_data_mutex);
    if (m_left.robot_frame.payload_case() != ::avatar::AvatarDataFrame::kRobot)
    {
        return {};
    }
    return { m_left.robot_frame.robot.joint.name, m_left.robot_frame.robot.joint.position };
}

AvatarJointFrame AvatarTracker::Impl::get_right_robot_frame() const
{
    std::lock_guard<std::mutex> lock(m_data_mutex);
    if (m_right.robot_frame.payload_case() != ::avatar::AvatarDataFrame::kRobot)
    {
        return {};
    }
    return { m_right.robot_frame.robot.joint.name, m_right.robot_frame.robot.joint.position };
}

void AvatarTracker::Impl::update()
{
    std::lock_guard<std::mutex> update_lock(m_update_mutex);
    try_connect_missing_gloves();

    if (!m_deviceio_session)
    {
        const auto now = std::chrono::steady_clock::now();
        if (now - m_last_openxr_retry >= kOpenXRRetryInterval)
        {
            try_initialize_openxr();
        }
        refresh_data();
        return;
    }

    m_deviceio_session->update();
    refresh_data();
    if (m_haptic_reader)
    {
        for (const std::string_view endpoint : { std::string_view("left"), std::string_view("right") })
        {
            const auto& tracked = m_haptic_reader->get_data(*m_deviceio_session, endpoint);
            const core::HapticCommand* command = tracked.get();
            if (command != nullptr && command->values() != nullptr && command->values()->size() == kAvatarFingerCount)
            {
                std::vector<float> powers(kAvatarFingerCount);
                for (size_t i = 0; i < kAvatarFingerCount; ++i)
                {
                    powers[i] = command->values()->Get(i);
                }
                apply_haptic_command(endpoint == "left", powers);
            }
        }
    }
    if (m_config.raw || m_config.robot)
    {
        push_joint_frames();
    }
    if (m_config.human && m_left_injector && m_right_injector)
    {
        inject_hand_data();
    }
}

void AvatarTracker::Impl::push_joint_frames()
{
    if (m_left_raw_pusher)
    {
        push_joint_frame(true, false, *m_left_raw_pusher);
    }
    if (m_right_raw_pusher)
    {
        push_joint_frame(false, false, *m_right_raw_pusher);
    }
    if (m_left_robot_pusher)
    {
        push_joint_frame(true, true, *m_left_robot_pusher);
    }
    if (m_right_robot_pusher)
    {
        push_joint_frame(false, true, *m_right_robot_pusher);
    }
}

void AvatarTracker::Impl::push_joint_frame(bool is_left, bool is_robot, core::SchemaPusher& pusher)
{
    ::avatar::AvatarDataFrame frame;
    {
        std::lock_guard<std::mutex> lock(m_data_mutex);
        const GloveState& glove = is_left ? m_left : m_right;
        frame = is_robot ? glove.robot_frame : glove.raw_frame;
    }

    const ::avatar::Hand& hand = is_robot ? frame.robot : frame.raw;
    if (frame.payload_case() != (is_robot ? ::avatar::AvatarDataFrame::kRobot : ::avatar::AvatarDataFrame::kRaw) ||
        hand.joint.position.empty())
    {
        return;
    }

    core::JointStateOutputT output;
    output.device_id = is_left ? (is_robot ? AVATAR_ROBOT_LEFT_COLLECTION_ID : AVATAR_RAW_LEFT_COLLECTION_ID) :
                                 (is_robot ? AVATAR_ROBOT_RIGHT_COLLECTION_ID : AVATAR_RAW_RIGHT_COLLECTION_ID);
    output.has_velocity = false;
    output.has_effort = false;
    output.ee_pose_valid = false;
    output.joints.reserve(hand.joint.position.size());

    for (size_t i = 0; i < hand.joint.position.size(); ++i)
    {
        auto joint = std::make_shared<core::JointStateT>();
        joint->name = "joint_" + std::to_string(i);
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

void AvatarTracker::Impl::apply_haptic_command(bool is_left, const std::vector<float>& powers)
{
    if (powers.size() != kAvatarFingerCount)
    {
        return;
    }

    GloveState& glove = is_left ? m_left : m_right;
    if (!glove.device || !glove.started)
    {
        return;
    }

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

    const auto [ec, unused] = glove.device->set_task(request);
    (void)unused;
    if (ec != ::avatar::ErrorCode::SUCCESS)
    {
        const size_t side = is_left ? 0 : 1;
        if (!m_haptic_error_logged[side])
        {
            m_haptic_error_logged[side] = true;
            std::cerr << "[Avatar] set_vibration failed for " << (is_left ? "left" : "right")
                      << " glove; further errors for this side will be silenced." << std::endl;
        }
    }
}

void AvatarTracker::Impl::map_landmarks_to_openxr(const std::vector<::avatar::Pose>& landmarks,
                                                  const XrPosef& root_pose,
                                                  bool is_root_tracked,
                                                  XrHandJointLocationEXT out_joints[XR_HAND_JOINT_COUNT_EXT]) const
{
    const auto& map = openxr_to_avatar_map();
    const int count = static_cast<int>(landmarks.size());

    for (uint32_t j = 0; j < XR_HAND_JOINT_COUNT_EXT; ++j)
    {
        const int src = map[j];
        if (src < 0 || src >= count)
        {
            out_joints[j] = { 0 };
            continue;
        }

        const auto& lp = landmarks[static_cast<size_t>(src)];

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

void AvatarTracker::Impl::inject_hand_data()
{
    std::vector<::avatar::Pose> left_landmarks;
    std::vector<::avatar::Pose> right_landmarks;
    {
        std::lock_guard<std::mutex> lock(m_data_mutex);
        left_landmarks = m_left.landmarks;
        right_landmarks = m_right.landmarks;
    }

    const XrTime time = m_time_converter->os_monotonic_now();

    auto process_hand = [&](const std::vector<::avatar::Pose>& landmarks, bool is_left)
    {
        if (landmarks.empty())
        {
            return;
        }

        XrPosef wrist_pose;
        bool is_root_tracked = false;
        bool xdev_pose_valid = false;

        if (m_xdev_available)
        {
            XrHandTrackerEXT tracker = is_left ? m_native_left_hand_tracker : m_native_right_hand_tracker;
            bool xdev_tracked = false;
            if (update_xdev_hand(tracker, time, wrist_pose, xdev_tracked))
            {
                if (is_left)
                {
                    m_left_root_pose = wrist_pose;
                }
                else
                {
                    m_right_root_pose = wrist_pose;
                }
                is_root_tracked = xdev_tracked;
                xdev_pose_valid = true;
            }
        }

        if (!xdev_pose_valid && get_controller_wrist_pose(is_left, wrist_pose))
        {
            if (is_left)
            {
                m_left_root_pose = wrist_pose;
            }
            else
            {
                m_right_root_pose = wrist_pose;
            }
            is_root_tracked = true;
        }

        const XrPosef root_pose = is_left ? m_left_root_pose : m_right_root_pose;

        XrHandJointLocationEXT joints[XR_HAND_JOINT_COUNT_EXT];
        map_landmarks_to_openxr(landmarks, root_pose, is_root_tracked, joints);

        if (is_left)
        {
            m_left_injector->push(joints, time);
        }
        else
        {
            m_right_injector->push(joints, time);
        }
    };

    process_hand(left_landmarks, true);
    process_hand(right_landmarks, false);
}

// ============================================================================
//  Wrist positioning (Isaac Teleop hand plugin pattern)
// ============================================================================

void AvatarTracker::Impl::initialize_xdev_hand_trackers()
{
    auto load_func = [this](const char* name, PFN_xrVoidFunction* ptr) -> bool
    {
        XrResult result = m_handles.xrGetInstanceProcAddr(m_handles.instance, name, ptr);
        return XR_SUCCEEDED(result) && *ptr != nullptr;
    };

    if (!load_func("xrCreateXDevListMNDX", reinterpret_cast<PFN_xrVoidFunction*>(&m_pfn_create_xdev_list)) ||
        !load_func("xrDestroyXDevListMNDX", reinterpret_cast<PFN_xrVoidFunction*>(&m_pfn_destroy_xdev_list)) ||
        !load_func("xrEnumerateXDevsMNDX", reinterpret_cast<PFN_xrVoidFunction*>(&m_pfn_enumerate_xdevs)) ||
        !load_func("xrGetXDevPropertiesMNDX", reinterpret_cast<PFN_xrVoidFunction*>(&m_pfn_get_xdev_properties)))
    {
        std::cerr << "[Avatar] XR_MNDX_xdev_space not available, falling back to controllers" << std::endl;
        return;
    }

    if (!load_func("xrCreateHandTrackerEXT", reinterpret_cast<PFN_xrVoidFunction*>(&m_pfn_create_hand_tracker)) ||
        !load_func("xrDestroyHandTrackerEXT", reinterpret_cast<PFN_xrVoidFunction*>(&m_pfn_destroy_hand_tracker)) ||
        !load_func("xrLocateHandJointsEXT", reinterpret_cast<PFN_xrVoidFunction*>(&m_pfn_locate_hand_joints)))
    {
        std::cerr << "[Avatar] Hand tracking extension not available, falling back to controllers" << std::endl;
        return;
    }

    XrCreateXDevListInfoMNDX create_info{ XR_TYPE_CREATE_XDEV_LIST_INFO_MNDX };
    XrResult result = m_pfn_create_xdev_list(m_handles.session, &create_info, &m_xdev_list);
    if (XR_FAILED(result))
    {
        std::cerr << "[Avatar] Failed to create XDevList, falling back to controllers" << std::endl;
        return;
    }

    uint32_t xdev_count = 0;
    result = m_pfn_enumerate_xdevs(m_xdev_list, 0, &xdev_count, nullptr);
    if (XR_FAILED(result) || xdev_count == 0)
    {
        std::cerr << "[Avatar] No XDevs found, falling back to controllers" << std::endl;
        return;
    }

    std::vector<XrXDevIdMNDX> xdev_ids(xdev_count);
    result = m_pfn_enumerate_xdevs(m_xdev_list, xdev_count, &xdev_count, xdev_ids.data());
    if (XR_FAILED(result))
    {
        return;
    }

    // Runtime-specific serial naming (observed on Monado). May change across runtimes.
    XrXDevIdMNDX left_xdev_id = 0;
    XrXDevIdMNDX right_xdev_id = 0;
    std::vector<std::string> seen_serials;

    for (const auto& xdev_id : xdev_ids)
    {
        XrGetXDevInfoMNDX get_info{ XR_TYPE_GET_XDEV_INFO_MNDX };
        get_info.id = xdev_id;

        XrXDevPropertiesMNDX properties{ XR_TYPE_XDEV_PROPERTIES_MNDX };
        result = m_pfn_get_xdev_properties(m_xdev_list, &get_info, &properties);
        if (XR_FAILED(result))
        {
            continue;
        }

        std::string serial_str = properties.serial ? properties.serial : "";
        seen_serials.push_back(serial_str);

        if (serial_str == "Head Device (0)")
        {
            left_xdev_id = xdev_id;
        }
        else if (serial_str == "Head Device (1)")
        {
            right_xdev_id = xdev_id;
        }
    }

    if (left_xdev_id == 0 || right_xdev_id == 0)
    {
        std::string serials_list;
        for (const auto& s : seen_serials)
        {
            if (!serials_list.empty())
                serials_list += ", ";
            serials_list += '"';
            serials_list += s;
            serials_list += '"';
        }
        std::cerr << "[Avatar] Could not match optical hand-tracking XDevs by serial. "
                  << "Expected \"Head Device (0)\" (left) and \"Head Device (1)\" (right), "
                  << "but found: [" << serials_list << "]." << std::endl;
    }

    auto create_tracker = [this](XrXDevIdMNDX xdev_id, XrHandEXT hand, XrHandTrackerEXT& out_tracker) -> bool
    {
        if (xdev_id == 0)
        {
            return false;
        }

        XrCreateHandTrackerXDevMNDX xdev_create_info{ XR_TYPE_CREATE_HAND_TRACKER_XDEV_MNDX };
        xdev_create_info.xdevList = m_xdev_list;
        xdev_create_info.id = xdev_id;

        XrHandTrackerCreateInfoEXT create_info{ XR_TYPE_HAND_TRACKER_CREATE_INFO_EXT };
        create_info.next = &xdev_create_info;
        create_info.hand = hand;
        create_info.handJointSet = XR_HAND_JOINT_SET_DEFAULT_EXT;

        return XR_SUCCEEDED(m_pfn_create_hand_tracker(m_handles.session, &create_info, &out_tracker));
    };

    const bool left_ok = create_tracker(left_xdev_id, XR_HAND_LEFT_EXT, m_native_left_hand_tracker);
    const bool right_ok = create_tracker(right_xdev_id, XR_HAND_RIGHT_EXT, m_native_right_hand_tracker);

    if (left_ok && right_ok)
    {
        m_xdev_available = true;
    }
    else
    {
        std::cerr << "[Avatar] Failed to create native hand trackers, falling back to controllers" << std::endl;
        cleanup_xdev_hand_trackers();
    }
}

void AvatarTracker::Impl::cleanup_xdev_hand_trackers()
{
    if (m_native_left_hand_tracker != XR_NULL_HANDLE && m_pfn_destroy_hand_tracker)
    {
        m_pfn_destroy_hand_tracker(m_native_left_hand_tracker);
        m_native_left_hand_tracker = XR_NULL_HANDLE;
    }
    if (m_native_right_hand_tracker != XR_NULL_HANDLE && m_pfn_destroy_hand_tracker)
    {
        m_pfn_destroy_hand_tracker(m_native_right_hand_tracker);
        m_native_right_hand_tracker = XR_NULL_HANDLE;
    }
    if (m_xdev_list != XR_NULL_HANDLE && m_pfn_destroy_xdev_list)
    {
        m_pfn_destroy_xdev_list(m_xdev_list);
        m_xdev_list = XR_NULL_HANDLE;
    }
    m_xdev_available = false;
}

bool AvatarTracker::Impl::update_xdev_hand(XrHandTrackerEXT tracker,
                                           XrTime time,
                                           XrPosef& out_wrist_pose,
                                           bool& out_is_tracked)
{
    out_is_tracked = false;

    if (tracker == XR_NULL_HANDLE || !m_pfn_locate_hand_joints || time == 0)
    {
        return false;
    }

    XrHandJointsLocateInfoEXT locate_info{ XR_TYPE_HAND_JOINTS_LOCATE_INFO_EXT };
    locate_info.baseSpace = m_handles.space;
    locate_info.time = time;

    XrHandJointLocationEXT joint_locations[XR_HAND_JOINT_COUNT_EXT];

    XrHandJointLocationsEXT locations{ XR_TYPE_HAND_JOINT_LOCATIONS_EXT };
    locations.jointCount = XR_HAND_JOINT_COUNT_EXT;
    locations.jointLocations = joint_locations;

    XrResult result = m_pfn_locate_hand_joints(tracker, &locate_info, &locations);
    if (XR_FAILED(result) || !locations.isActive)
    {
        return false;
    }

    const auto& wrist = joint_locations[XR_HAND_JOINT_WRIST_EXT];
    const bool is_valid = (wrist.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) &&
                          (wrist.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT);

    if (is_valid)
    {
        out_wrist_pose = wrist.pose;
        out_is_tracked = (wrist.locationFlags & XR_SPACE_LOCATION_POSITION_TRACKED_BIT) &&
                         (wrist.locationFlags & XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT);
        return true;
    }

    return false;
}

bool AvatarTracker::Impl::get_controller_wrist_pose(bool is_left, XrPosef& out_wrist_pose)
{
    const auto& tracked = is_left ? m_controller_tracker->get_left_controller(*m_deviceio_session) :
                                    m_controller_tracker->get_right_controller(*m_deviceio_session);

    if (!tracked)
    {
        return false;
    }

    bool aim_valid = false;
    XrPosef raw_pose = oxr_utils::get_aim_pose(*tracked, aim_valid);

    if (!aim_valid)
    {
        return false;
    }

    XrPosef offset_pose = is_left ? kLeftHandOffset : kRightHandOffset;
    out_wrist_pose = oxr_utils::multiply_poses(raw_pose, offset_pose);
    return true;
}

AvatarTracker::AvatarTracker(AvatarPluginConfig config) : m_impl(std::make_unique<Impl>(std::move(config)))
{
}

AvatarTracker::~AvatarTracker() = default;

void AvatarTracker::update()
{
    m_impl->update();
}

std::vector<AvatarLandmark> AvatarTracker::get_left_landmarks() const
{
    return m_impl->get_left_landmarks();
}

std::vector<AvatarLandmark> AvatarTracker::get_right_landmarks() const
{
    return m_impl->get_right_landmarks();
}

AvatarJointFrame AvatarTracker::get_left_raw_frame() const
{
    return m_impl->get_left_raw_frame();
}

AvatarJointFrame AvatarTracker::get_right_raw_frame() const
{
    return m_impl->get_right_raw_frame();
}

AvatarJointFrame AvatarTracker::get_left_robot_frame() const
{
    return m_impl->get_left_robot_frame();
}

AvatarJointFrame AvatarTracker::get_right_robot_frame() const
{
    return m_impl->get_right_robot_frame();
}

} // namespace avatar
} // namespace plugins
