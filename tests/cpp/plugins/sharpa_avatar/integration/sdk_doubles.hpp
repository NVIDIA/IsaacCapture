// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Synthetic SDK/transport boundaries for the real tracker orchestration tests.
// These record ownership, fetches and pushes; they do not emulate a live runtime.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
using XrTime = int64_t;
using XrHandEXT = int;
using XrSpace = int;
inline constexpr int XR_HAND_LEFT_EXT = 0, XR_HAND_RIGHT_EXT = 1, XR_HAND_JOINT_COUNT_EXT = 26;
inline constexpr int XR_SPACE_LOCATION_POSITION_VALID_BIT = 1, XR_SPACE_LOCATION_ORIENTATION_VALID_BIT = 2,
                     XR_SPACE_LOCATION_POSITION_TRACKED_BIT = 4, XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT = 8;
inline constexpr char XR_NVX1_DEVICE_INTERFACE_BASE_EXTENSION_NAME[] = "fakeXR";
struct XrQuaternionf
{
    float x = 0, y = 0, z = 0, w = 1;
};
struct XrVector3f
{
    float x = 0, y = 0, z = 0;
};
struct XrPosef
{
    XrQuaternionf orientation;
    XrVector3f position;
};
struct XrHandJointLocationEXT
{
    uint64_t locationFlags;
    XrPosef pose;
    float radius;
};
namespace tracker_test_double
{
inline std::map<std::string, int> created, destroyed, pushed, attempts;
inline std::set<std::string> fail, push_fail;
} // namespace tracker_test_double
namespace spdlog
{
class logger
{
public:
    template <class... T>
    void info(T&&...)
    {
    }
    template <class... T>
    void warn(T&&...)
    {
    }
    template <class... T>
    void debug(T&&...)
    {
    }
    template <class... T>
    void error(T&&...){};
};
} // namespace spdlog
namespace isaaccapture
{
struct Logger
{
    static auto get(const char*)
    {
        return std::make_shared<spdlog::logger>();
    }
};
} // namespace isaaccapture
namespace avatar
{
enum class DeviceSide
{
    LEFT,
    RIGHT
};
enum class DeviceDataCategory
{
    HUMAN,
    RAW,
    ROBOT
};
enum class DeviceType
{
    GLOVE
};
enum class ErrorCode
{
    SUCCESS,
    INVALID_INPUT_PARAMETER,
    OPERATION_NOT_ALLOWED,
    CONNECTION_FAILED,
    TCP_SEND_FAILED,
    TCP_RECV_FAILED,
    INVALID_RESPONSE,
    DEVICE_NOT_FOUND,
    NO_VALID_DATA_RETURNED,
    DEVICE_UNSUPPORTED_COMMAND,
    INVALID_HEADER,
    INCOMPLETE_PAYLOAD,
    CRC_ERROR,
    INVALID_PAYLOAD,
    INNER_ERROR,
    UPGRADE_IN_PROGRESS,
    INVALID_FIRMWARE_FILE,
    FIRMWARE_VERSION_INCOMPATIBLE,
    UNKNOWN_RSP_CODE
};
enum class FaultCode
{
    NORMAL
};
enum class UsbConnectionState
{
    CONNECTED,
    DISCONNECTED,
    CONNECTING,
    RECONNECTING,
    ERROR
};
struct Stamp
{
    int64_t sec = 0, nanosec = 0;
};
struct Header
{
    Stamp stamp;
};
struct Pose
{
    XrVector3f position;
    XrQuaternionf orientation;
};
struct Joint
{
    std::vector<float> position;
    std::vector<std::string> name;
};
struct Hand
{
    Header header;
    Joint joint;
};
struct Skeleton
{
    Header header;
    std::vector<Pose> landmark;
};
struct AvatarDataFrame
{
    enum PayloadCase
    {
        PAYLOAD_NOT_SET,
        kRaw,
        kRobot,
        kSkeleton
    };
    PayloadCase which = PAYLOAD_NOT_SET;
    Skeleton skeleton;
    Hand raw, robot;
    PayloadCase payload_case() const
    {
        return which;
    }
};
struct DeviceInfo
{
    DeviceSide device_side;
    bool online = true;
    std::map<int, int> diagnostics;
    int error_code = 0;
};
struct Device
{
    DeviceSide side;
    std::array<AvatarDataFrame, 3> frames;
    std::array<ErrorCode, 3> errors{};
    std::array<int, 3> fetches{};
    std::function<void(const AvatarDataFrame&, const AvatarDataFrame&)> callback;
    void set_human_frame_build_enabled(bool)
    {
    }
    void set_raw_human_callback_max_hz(double)
    {
    }
    void set_raw_human_frame_callback(std::function<void(const AvatarDataFrame&, const AvatarDataFrame&)> cb)
    {
        callback = std::move(cb);
    }
    ErrorCode start()
    {
        pulse();
        return ErrorCode::SUCCESS;
    }
    void stop()
    {
    }
    void pulse()
    {
        if (callback)
            callback(frames[0], frames[1]);
    }
    ErrorCode fetch_data(AvatarDataFrame& f, DeviceDataCategory c)
    {
        int i = (int)c;
        ++fetches[i];
        f = frames[i];
        return errors[i];
    }
    auto set_task(const std::string&)
    {
        return std::make_pair(ErrorCode::SUCCESS, 0);
    }
};
using DevicePtr = std::shared_ptr<Device>;
struct AvatarSDK
{
    inline static std::array<DevicePtr, 2> devices;
    static AvatarSDK& get_instance()
    {
        static AvatarSDK s;
        return s;
    }
    ErrorCode initialize(const std::string&)
    {
        return ErrorCode::SUCCESS;
    }
    void destroy()
    {
    }
    auto get_device(DeviceType, DeviceSide side)
    {
        return devices[(int)side];
    }
    UsbConnectionState get_usb_connection_state()
    {
        return UsbConnectionState::CONNECTED;
    }
    std::vector<DeviceInfo> list_device_info(DeviceType)
    {
        std::vector<DeviceInfo> v;
        for (int i = 0; i < 2; ++i)
            if (devices[i])
                v.push_back({ (DeviceSide)i });
        return v;
    }
};
} // namespace avatar
namespace flatbuffers
{
struct FlatBufferBuilder
{
    explicit FlatBufferBuilder(size_t)
    {
    }
    template <class T>
    void Finish(T)
    {
    }
    const uint8_t* GetBufferPointer()
    {
        static uint8_t b = 0;
        return &b;
    }
    size_t GetSize()
    {
        return 1;
    }
};
} // namespace flatbuffers
namespace core
{
inline int64_t os_monotonic_now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct OpenXRSessionHandles
{
    int instance = 1, session = 1, space = 1;
};
struct ITracker
{
    virtual ~ITracker() = default;
};
struct OpenXRSession
{
    OpenXRSession(const std::string&, const std::vector<std::string>&, bool)
    {
    }
    OpenXRSessionHandles get_handles()
    {
        return {};
    }
};
struct XrTimeConverter
{
    explicit XrTimeConverter(const OpenXRSessionHandles&)
    {
    }
    XrTime os_monotonic_now()
    {
        return os_monotonic_now_ns();
    }
};
template <class T>
using Serialized = std::shared_ptr<T>;
struct HapticValues
{
    size_t size() const
    {
        return 5;
    }
    float Get(size_t) const
    {
        return 0;
    }
};
struct HapticCommand
{
    const HapticValues* values() const
    {
        return nullptr;
    }
};
struct DeviceIOSession
{
    static std::vector<std::string> get_required_extensions(const std::vector<std::shared_ptr<ITracker>>&)
    {
        return {};
    }
    static auto run(const std::vector<std::shared_ptr<ITracker>>&, OpenXRSessionHandles)
    {
        return std::make_unique<DeviceIOSession>();
    }
    void update()
    {
    }
};
struct HapticCommandReaderTracker : ITracker
{
    inline static constexpr size_t DEFAULT_MAX_PAYLOAD_SIZE = 100;
    template <class... T>
    HapticCommandReaderTracker(T&&...)
    {
    }
    Serialized<HapticCommand> get_data(DeviceIOSession&, std::string_view)
    {
        return {};
    }
};
struct SchemaPusherConfig
{
    std::string collection_id;
    size_t max_flatbuffer_size = 0;
    std::string tensor_identifier, localized_name;
};
struct SchemaPusher
{
    std::string key;
    SchemaPusher(OpenXRSessionHandles, const SchemaPusherConfig& c) : key(c.collection_id)
    {
        ++tracker_test_double::attempts[key];
        if (tracker_test_double::fail.count(key))
            throw std::runtime_error("create failed " + key);
        ++tracker_test_double::created[key];
    }
    ~SchemaPusher()
    {
        ++tracker_test_double::destroyed[key];
    }
    static std::vector<std::string> get_required_extensions()
    {
        return {};
    }
    void push_buffer(const uint8_t*, size_t, int64_t, int64_t)
    {
        if (tracker_test_double::push_fail.count(key))
            throw std::runtime_error("push failed " + key);
        ++tracker_test_double::pushed[key];
    }
};
struct JointStateT
{
    std::string name;
    float position = 0;
    bool valid = false;
};
struct JointStateOutputT
{
    std::string device_id;
    bool has_velocity = false, has_effort = false, ee_pose_valid = false;
    std::vector<std::shared_ptr<JointStateT>> joints;
};
struct JointStateOutput
{
    static int Pack(flatbuffers::FlatBufferBuilder&, JointStateOutputT*)
    {
        return 0;
    }
};
enum PluginDeviceState
{
    PluginDeviceState_UNKNOWN,
    PluginDeviceState_DEGRADED,
    PluginDeviceState_CONNECTED,
    PluginDeviceState_DISCONNECTED,
    PluginDeviceState_FAILED,
    PluginDeviceState_DISABLED
};
enum PluginDeviceReason
{
    PluginDeviceReason_NONE,
    PluginDeviceReason_NO_CURRENT_DATA,
    PluginDeviceReason_DEVICE_ERROR,
    PluginDeviceReason_NO_HARDWARE_SIGNAL,
    PluginDeviceReason_DISABLED_BY_CONFIGURATION
};
} // namespace core
namespace oxr_utils
{
inline XrPosef identity_posef()
{
    return {};
}
inline XrPosef multiply_poses(XrPosef a, XrPosef)
{
    return a;
}
} // namespace oxr_utils
namespace plugin_utils
{
struct PluginDeviceStatusEntry
{
    std::string path;
    core::PluginDeviceState state = core::PluginDeviceState_UNKNOWN;
    core::PluginDeviceReason reason = core::PluginDeviceReason_NONE;
    std::string error;
};
struct PluginDeviceStatusPublisher
{
    inline static std::vector<PluginDeviceStatusEntry> latest;
    template <class... T>
    PluginDeviceStatusPublisher(T&&...)
    {
    }
    static std::vector<std::string> get_required_extensions()
    {
        return {};
    }
    void publish_if_changed(const std::vector<PluginDeviceStatusEntry>& v, int64_t)
    {
        latest = v;
    }
};
struct HandInjector
{
    std::string key;
    HandInjector(int, int, XrHandEXT hand, int) : key(hand == XR_HAND_LEFT_EXT ? "human_left" : "human_right")
    {
        ++tracker_test_double::attempts[key];
        if (tracker_test_double::fail.count(key))
            throw std::runtime_error("create failed " + key);
        ++tracker_test_double::created[key];
    }
    ~HandInjector()
    {
        ++tracker_test_double::destroyed[key];
    }
    void push(const XrHandJointLocationEXT*, XrTime)
    {
        if (tracker_test_double::push_fail.count(key))
            throw std::runtime_error("push failed " + key);
        ++tracker_test_double::pushed[key];
    }
};
enum class WristSourceMode
{
    Auto
};
struct WristSourceConfig
{
    WristSourceMode mode;
    XrPosef left_aim_to_wrist, right_aim_to_wrist;
};
struct WristSample
{
    bool valid = false, tracked = false;
    XrPosef pose;
};
struct WristPoseSource
{
    struct Requirements
    {
        std::vector<std::shared_ptr<core::ITracker>> trackers;
        std::vector<const char*> extensions;
        std::shared_ptr<core::ITracker> controller_tracker;
    };
    static Requirements collect_requirements(WristSourceMode)
    {
        return {};
    }
    template <class... T>
    WristPoseSource(T&&...)
    {
    }
    WristSample query(bool, XrTime)
    {
        return {};
    }
};
} // namespace plugin_utils
