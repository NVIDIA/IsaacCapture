// SPDX-FileCopyrightText: Copyright (c) 2025-2026 Avatar SDK contributors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/avatar/avatar_hand_tracking_plugin.hpp"

#include "avatar_glove_collection.hpp"

#include <flatbuffers/flatbuffers.h>
#include <nlohmann/json.hpp>
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
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace plugins
{
namespace avatar
{

namespace
{

constexpr size_t kJointFlatbufferSize = 4096;
constexpr auto kAvatarDataTimeout = std::chrono::seconds(10);
constexpr auto kGloveRetryInterval = std::chrono::seconds(2);
constexpr auto kGloveWaitLogInterval = std::chrono::seconds(10);

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

constexpr plugin_utils::WristSide wrist_side(::avatar::DeviceSide side)
{
    switch (side)
    {
    case ::avatar::DeviceSide::LEFT:
        return plugin_utils::WristSide::Left;
    case ::avatar::DeviceSide::RIGHT:
        return plugin_utils::WristSide::Right;
    }
    return plugin_utils::WristSide::Left;
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

// Exact `human_joint_names` spellings from sdk_config.json. Null slots have no
// Avatar HUMAN counterpart and must remain invalid.
constexpr std::array<const char*, XR_HAND_JOINT_COUNT_EXT> kOpenXRSlotSources = {
    nullptr,
    "WRIST",
    "right_thumb_CMC_FE_link",
    "right_thumb_MCP_FE_link",
    "right_thumb_IP_link",
    "right_thumb_virtualtip",
    "right_index_MCP_AA_link",
    "right_index_MCP_FE_link",
    "right_index_PIP_link",
    "right_index_DIP_link",
    "right_index_virtualtip",
    "right_middle_MCP_AA_link",
    "right_middle_MCP_FE_link",
    "right_middle_PIP_link",
    "right_middle_DIP_link",
    "right_middle_virtualtip",
    "right_ring_MCP_AA_link",
    "right_ring_MCP_FE_link",
    "right_ring_PIP_link",
    "right_ring_DIP_link",
    "right_ring_virtualtip",
    "right_pinky_MCP_AA_link",
    "right_pinky_MCP_FE_link",
    nullptr,
    nullptr,
    "right_pinky_virtualtip",
};

std::unordered_map<std::string, size_t> load_human_landmark_indices(const std::string& config_path)
{
    std::ifstream file(config_path);
    if (!file)
    {
        throw std::runtime_error("Cannot open Avatar SDK config: " + config_path);
    }

    const auto names = nlohmann::json::parse(file).at("human_joint_names").get<std::vector<std::string>>();
    std::unordered_map<std::string, size_t> indices;
    for (size_t i = 0; i < names.size(); ++i)
    {
        if (!indices.emplace(names[i], i).second)
        {
            throw std::runtime_error("Duplicate human landmark name in Avatar SDK config: " + names[i]);
        }
    }
    for (const char* source : kOpenXRSlotSources)
    {
        if (source != nullptr && indices.find(source) == indices.end())
        {
            throw std::runtime_error("Avatar SDK config is missing human landmark: " + std::string(source));
        }
    }
    return indices;
}

} // anonymous namespace

// Wrist offsets for XR controllers: align the controller aim pose to a natural
// wrist orientation for each hand.
static constexpr XrPosef kLeftHandOffset = { { -0.70710678f, -0.5f, 0.0f, 0.5f }, { -0.1f, 0.02f, -0.02f } };
static constexpr XrPosef kRightHandOffset = { { -0.70710678f, 0.5f, 0.0f, 0.5f }, { 0.1f, 0.02f, -0.02f } };

AvatarSdkSession::AvatarSdkSession(const std::string& config_path)
    : m_config_path(config_path.empty() ? AVATAR_SDK_CONFIG_PATH : config_path)
{
    const auto error = ::avatar::AvatarSDK::get_instance().initialize(m_config_path);
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

const std::string& AvatarSdkSession::config_path() const
{
    return m_config_path;
}

GloveState::~GloveState() noexcept
{
    reset();
}

void GloveState::reset() noexcept
{
    try
    {
        if (device)
        {
            device->stop();
        }
    }
    catch (...)
    {
        // Reset remains safe when called from GloveState's destructor.
    }
    // AvatarSDK owns and destroys its cached devices with the SDK session.
    device.reset();
    last_successful_fetch.reset();
}

AvatarTracker::AvatarTracker(AvatarPluginConfig config) : m_config(std::move(config)), m_sdk(m_config.sdk_config_path)
{
    std::cout << "[Avatar] Avatar SDK initialized." << std::endl;
    std::cout << "[Avatar] datasets: human=" << (m_config.human ? "on" : "off")
              << " raw=" << (m_config.raw ? "on" : "off") << " robot=" << (m_config.robot ? "on" : "off")
              << " haptic=" << (m_config.haptic ? "on" : "off") << std::endl;
    if (m_config.human)
    {
        m_landmark_index = load_human_landmark_indices(m_sdk.config_path());
    }
    try_connect_missing_gloves();
    initialize_openxr();
}

void AvatarTracker::initialize_openxr()
{
    plugin_utils::WristSourceConfig wrist_config;
    wrist_config.mode = plugin_utils::WristSourceMode::Auto;
    wrist_config.aim_to_wrist[static_cast<size_t>(plugin_utils::WristSide::Left)] = kLeftHandOffset;
    wrist_config.aim_to_wrist[static_cast<size_t>(plugin_utils::WristSide::Right)] = kRightHandOffset;
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
    for (const char* extension : wrist_requirements.extensions)
    {
        add_extension(extension);
    }

    const bool wait_for_openxr_system = false;
    m_session = std::make_shared<core::OpenXRSession>(m_config.app_name, extensions, wait_for_openxr_system);
    m_handles = m_session->get_handles();

    m_time_converter.emplace(m_handles);

    if (m_config.human)
    {
        for (const ::avatar::DeviceSide side : kDeviceSides)
        {
            m_injectors[side_index(side)] = std::make_unique<plugin_utils::HandInjector>(
                m_handles.instance, m_handles.session, xr_hand(side), m_handles.space);
        }
    }

    auto make_joint_pusher = [this](::avatar::DeviceSide side, ::avatar::DeviceDataCategory category)
    {
        const std::string localized_name =
            "Avatar " + std::string(to_string(category)) + " " + std::string(to_string(side));
        return std::make_unique<core::SchemaPusher>(
            m_handles, core::SchemaPusherConfig{ .collection_id = collection_id(side, category),
                                                 .max_flatbuffer_size = kJointFlatbufferSize,
                                                 .tensor_identifier = "joint_state",
                                                 .localized_name = localized_name });
    };
    for (const ::avatar::DeviceDataCategory category : kJointDataCategories)
    {
        if (!dataset_enabled(category))
        {
            continue;
        }
        for (const ::avatar::DeviceSide side : kDeviceSides)
        {
            m_joint_pushers[side_index(side)][joint_category_index(category)] = make_joint_pusher(side, category);
        }
    }

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

    const auto init_error = glove.device->init("{}");
    if (init_error != ::avatar::ErrorCode::SUCCESS)
    {
        std::cerr << "[Avatar] " << to_string(side) << " glove initialization failed: " << error_name(init_error)
                  << " (" << static_cast<int>(init_error) << ")" << std::endl;
        glove.reset();
        return;
    }
    glove.device->set_human_frame_build_enabled(m_config.human);

    const auto start_error = glove.device->start();
    if (start_error != ::avatar::ErrorCode::SUCCESS)
    {
        std::cerr << "[Avatar] " << to_string(side) << " glove start failed: " << error_name(start_error) << " ("
                  << static_cast<int>(start_error) << ")" << std::endl;
        glove.reset();
        return;
    }
    glove.last_successful_fetch = std::chrono::steady_clock::now();
    std::cout << "[Avatar] " << to_string(side) << " glove connected and streaming." << std::endl;
}

void AvatarTracker::try_connect_missing_gloves()
{
    const auto has_device = [](const GloveState& state) { return state.device != nullptr; };
    if (std::all_of(m_gloves.begin(), m_gloves.end(), has_device))
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
        start_glove_if_present(glove(side), side);
    }
    if (std::none_of(m_gloves.begin(), m_gloves.end(), has_device) &&
        (!m_last_glove_wait_log || now - *m_last_glove_wait_log >= kGloveWaitLogInterval))
    {
        m_last_glove_wait_log = now;
        std::cout << "[Avatar] Waiting for an online glove..." << std::endl;
    }
}

void AvatarTracker::refresh_data()
{
    for (const ::avatar::DeviceSide side : kDeviceSides)
    {
        GloveState& state = glove(side);
        if (!state.device)
        {
            continue;
        }
        if (!state.device->get_device_info().online)
        {
            state.landmarks.clear();
            state.raw_frame = {};
            state.robot_frame = {};
            state.reset();
            continue;
        }

        const bool expects_data = m_config.human || m_config.raw || m_config.robot;
        bool fetched_any = false;
        if (m_config.human)
        {
            ::avatar::AvatarDataFrame frame;
            const bool fetched =
                state.device->fetch_data(frame, ::avatar::DeviceDataCategory::HUMAN) == ::avatar::ErrorCode::SUCCESS;
            fetched_any = fetched_any || fetched;
            if (fetched)
            {
                state.landmarks = std::move(frame.skeleton.landmark);
            }
            else
            {
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
            const bool fetched = state.device->fetch_data(frame, category) == ::avatar::ErrorCode::SUCCESS;
            fetched_any = fetched_any || fetched;
            ::avatar::AvatarDataFrame& cached = cached_joint_frame(state, category);
            if (fetched)
            {
                cached = std::move(frame);
            }
            else
            {
                cached = {};
            }
        }
        if (expects_data)
        {
            const auto now = std::chrono::steady_clock::now();
            if (fetched_any)
            {
                state.last_successful_fetch = now;
            }
            else if (!state.last_successful_fetch || now - *state.last_successful_fetch >= kAvatarDataTimeout)
            {
                state.reset();
            }
        }
    }
}

const GloveState& AvatarTracker::glove(::avatar::DeviceSide side) const
{
    return m_gloves[side_index(side)];
}

GloveState& AvatarTracker::glove(::avatar::DeviceSide side)
{
    return m_gloves[side_index(side)];
}

std::vector<AvatarLandmark> AvatarTracker::get_landmarks(::avatar::DeviceSide side) const
{
    const auto& source = glove(side).landmarks;
    std::vector<AvatarLandmark> landmarks;
    landmarks.reserve(source.size());
    for (const auto& landmark : source)
    {
        landmarks.push_back({ .position = { landmark.position.x, landmark.position.y, landmark.position.z },
                              .orientation = { landmark.orientation.w, landmark.orientation.x, landmark.orientation.y,
                                               landmark.orientation.z } });
    }
    return landmarks;
}

AvatarJointFrame AvatarTracker::get_joint_frame(::avatar::DeviceSide side, ::avatar::DeviceDataCategory category) const
{
    const GloveState& state = glove(side);
    const ::avatar::AvatarDataFrame& frame = cached_joint_frame(state, category);
    if (frame.payload_case() != payload_case(category))
    {
        return {};
    }
    const ::avatar::Hand& hand = hand_payload(frame, category);
    return { hand.joint.name, hand.joint.position };
}

void AvatarTracker::update()
{
    try_connect_missing_gloves();
    m_deviceio_session->update();
    refresh_data();
    if (m_haptic_reader)
    {
        for (const ::avatar::DeviceSide side : kDeviceSides)
        {
            const auto& tracked = m_haptic_reader->get_data(*m_deviceio_session, to_string(side));
            const core::HapticCommand* command = tracked.get();
            if (command != nullptr && command->values() != nullptr && command->values()->size() == kAvatarFingerCount)
            {
                std::array<float, kAvatarFingerCount> powers{};
                for (size_t i = 0; i < kAvatarFingerCount; ++i)
                {
                    powers[i] = command->values()->Get(i);
                }
                apply_haptic_command(side, powers);
            }
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

    for (size_t i = 0; i < hand.joint.position.size(); ++i)
    {
        auto joint = std::make_shared<core::JointStateT>();
        joint->name = i < hand.joint.name.size() ? hand.joint.name[i] : std::string{};
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

void AvatarTracker::apply_haptic_command(::avatar::DeviceSide side, const std::array<float, kAvatarFingerCount>& powers)
{
    GloveState& state = glove(side);
    if (!state.device)
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

    const auto [error, unused] = state.device->set_task(request);
    (void)unused;
    if (error != ::avatar::ErrorCode::SUCCESS)
    {
        const size_t index = side_index(side);
        if (!m_haptic_error_logged[index])
        {
            m_haptic_error_logged[index] = true;
            std::cerr << "[Avatar] set_vibration failed for " << to_string(side) << " glove: " << error_name(error)
                      << " (" << static_cast<int>(error) << "); further errors for this side will be silenced."
                      << std::endl;
        }
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
        const char* source = kOpenXRSlotSources[j];
        if (source == nullptr)
        {
            continue;
        }

        const auto source_it = m_landmark_index.find(source);
        if (source_it == m_landmark_index.end() || source_it->second >= landmarks.size())
        {
            continue;
        }
        const auto& lp = landmarks[source_it->second];

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

        const plugin_utils::WristSample wrist = m_wrist_source->query(wrist_side(side), time);
        if (!wrist.valid)
        {
            return;
        }

        XrHandJointLocationEXT joints[XR_HAND_JOINT_COUNT_EXT];
        map_landmarks_to_openxr(landmarks, wrist.pose, wrist.tracked, joints);

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
