// SPDX-FileCopyrightText: Copyright (c) 2026 Wuji Technology. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "wuji_glove_plugin.hpp"

#include <oxr_utils/math.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace plugins
{
namespace wuji_glove
{

namespace
{

// MediaPipe-21 landmark index -> OpenXR XrHandJointEXT slot.
// The 5 OpenXR slots not covered here (PALM + the four non-thumb METACARPALs)
// are left with POSITION/ORIENTATION_VALID cleared — the Wuji retargeter skips
// them, and other XR_EXT_hand_tracking consumers treat them as untracked.
//
// KEEP IN SYNC with the Python table OPENXR_TO_MEDIAPIPE_INDICES in
// src/python/isaacteleop/retargeters/wuji_hand_retargeter.py — same 21 entries, applied in the
// opposite direction (that one gathers OpenXR -> MediaPipe). A silent
// divergence is a joint-permutation bug nothing catches at runtime.
constexpr XrHandJointEXT kMpToXr[21] = {
    XR_HAND_JOINT_WRIST_EXT,
    XR_HAND_JOINT_THUMB_METACARPAL_EXT,
    XR_HAND_JOINT_THUMB_PROXIMAL_EXT,
    XR_HAND_JOINT_THUMB_DISTAL_EXT,
    XR_HAND_JOINT_THUMB_TIP_EXT,
    XR_HAND_JOINT_INDEX_PROXIMAL_EXT,
    XR_HAND_JOINT_INDEX_INTERMEDIATE_EXT,
    XR_HAND_JOINT_INDEX_DISTAL_EXT,
    XR_HAND_JOINT_INDEX_TIP_EXT,
    XR_HAND_JOINT_MIDDLE_PROXIMAL_EXT,
    XR_HAND_JOINT_MIDDLE_INTERMEDIATE_EXT,
    XR_HAND_JOINT_MIDDLE_DISTAL_EXT,
    XR_HAND_JOINT_MIDDLE_TIP_EXT,
    XR_HAND_JOINT_RING_PROXIMAL_EXT,
    XR_HAND_JOINT_RING_INTERMEDIATE_EXT,
    XR_HAND_JOINT_RING_DISTAL_EXT,
    XR_HAND_JOINT_RING_TIP_EXT,
    XR_HAND_JOINT_LITTLE_PROXIMAL_EXT,
    XR_HAND_JOINT_LITTLE_INTERMEDIATE_EXT,
    XR_HAND_JOINT_LITTLE_DISTAL_EXT,
    XR_HAND_JOINT_LITTLE_TIP_EXT,
};

constexpr XrSpaceLocationFlags kPoseValidFlags =
    XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
constexpr XrSpaceLocationFlags kPoseTrackedFlags =
    XR_SPACE_LOCATION_POSITION_TRACKED_BIT | XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT;

constexpr XrPosef kLeftAimToWrist = { { 0.26388208f, 0.17382305f, -0.06730102f, 0.94637327f },
                                      { -0.01391519f, -0.10860867f, 0.08197439f } };
constexpr XrPosef kRightAimToWrist = { { 0.26388208f, -0.17382305f, 0.06730102f, 0.94637327f },
                                       { 0.01391519f, -0.10860867f, 0.08197439f } };

constexpr float kDefaultJointRadius = 0.01f; // meters; SDK does not provide radius.
constexpr size_t kNumMediaPipeJoints = std::size(kMpToXr);

// Timing budget: the glove streams EMF poses at 120 Hz. The 16 ms pump
// (~62.5 Hz) always finds a fresher sample than the one it injected last, and
// the 200 ms staleness window is ~24 missed frames before a hand is dropped.
constexpr auto kFramePeriod = std::chrono::milliseconds(16);
constexpr auto kStaleThreshold = std::chrono::milliseconds(200);
constexpr auto kDiscoveryInterval = std::chrono::seconds(1);
constexpr auto kShutdownPollInterval = std::chrono::milliseconds(100);

// Re-express a wrist-relative joint pose from the Wuji skeleton basis in the
// OpenXR hand-joint basis.
//
// The SDK skeleton poses are FK link frames of wuji-sdk's default hand URDFs:
// the bone runs along -Z (matching XR_EXT_hand_tracking's "+Z points away from
// the fingertip"), and the palmar side is +Y on the right hand / -Y on the
// left (the left URDF is Y-mirrored). OpenXR puts the dorsal side at +Y for
// both hands, so the left hand already matches and the right hand differs by
// a 180° rotation about Z — whose conjugation simply negates the x and y
// components of both the position and the quaternion.
constexpr XrPosef remap_to_openxr_basis(const XrPosef& pose, bool is_left)
{
    if (is_left)
    {
        return pose;
    }
    XrPosef out = pose;
    out.position.x = -pose.position.x;
    out.position.y = -pose.position.y;
    out.orientation.x = -pose.orientation.x;
    out.orientation.y = -pose.orientation.y;
    return out;
}

constexpr bool remap_to_openxr_basis_is_correct()
{
    constexpr XrPosef input{ { 1.0f, 2.0f, 3.0f, 4.0f }, { 5.0f, 6.0f, 7.0f } };
    constexpr XrPosef left = remap_to_openxr_basis(input, true);
    constexpr XrPosef right = remap_to_openxr_basis(input, false);
    return left.orientation.x == 1.0f && left.orientation.y == 2.0f && left.orientation.z == 3.0f &&
           left.orientation.w == 4.0f && left.position.x == 5.0f && left.position.y == 6.0f &&
           left.position.z == 7.0f && right.orientation.x == -1.0f && right.orientation.y == -2.0f &&
           right.orientation.z == 3.0f && right.orientation.w == 4.0f && right.position.x == -5.0f &&
           right.position.y == -6.0f && right.position.z == 7.0f;
}

static_assert(remap_to_openxr_basis_is_correct());

// Convert a 21-joint Wuji skeleton into a 26-joint XrHandJointLocationEXT set,
// re-based into the OpenXR hand-joint basis. Returns false if the skeleton
// does not carry the expected 21 joints.
//
// Output poses stay wrist-relative with VALID-only flags; pump_hand() composes
// the fused wrist pose on top and decides the TRACKED bits.
bool convert_skeleton(const WujiHandSkeleton* frame,
                      bool is_left,
                      std::array<XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT>& out)
{
    if (frame == nullptr || frame->joints == nullptr || frame->joints_len < kNumMediaPipeJoints)
    {
        return false;
    }

    // Start fully untracked, then fill the 21 mapped joints.
    for (auto& j : out)
    {
        j = XrHandJointLocationEXT{};
        j.locationFlags = 0;
        j.radius = kDefaultJointRadius;
        j.pose.orientation = XrQuaternionf{ 0.0f, 0.0f, 0.0f, 1.0f };
    }

    for (size_t mp = 0; mp < kNumMediaPipeJoints; ++mp)
    {
        const WujiSkeletonJoint& src = frame->joints[mp];
        XrHandJointLocationEXT& dst = out[kMpToXr[mp]];
        const XrPosef sdk_pose{ XrQuaternionf{ src.pose.orientation.x, src.pose.orientation.y, src.pose.orientation.z,
                                               src.pose.orientation.w },
                                XrVector3f{ src.pose.position[0], src.pose.position[1], src.pose.position[2] } };
        dst.pose = remap_to_openxr_basis(sdk_pose, is_left);
        dst.radius = kDefaultJointRadius;
        dst.locationFlags = kPoseValidFlags;
    }
    return true;
}

// Optional aim-to-wrist offset override.
// WUJI_GLOVE_AIM_TO_WRIST_{LEFT,RIGHT}="px,py,pz,qx,qy,qz,qw" (meters + quat).
XrPosef pose_from_env(const char* name, const XrPosef& fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0')
    {
        return fallback;
    }
    XrPosef pose{};
    if (std::sscanf(value, "%f,%f,%f,%f,%f,%f,%f", &pose.position.x, &pose.position.y, &pose.position.z,
                    &pose.orientation.x, &pose.orientation.y, &pose.orientation.z, &pose.orientation.w) != 7)
    {
        std::cerr << "WujiGlovePlugin: could not parse " << name << " ('" << value
                  << "', want px,py,pz,qx,qy,qz,qw); using the built-in offset" << std::endl;
        return fallback;
    }
    // sscanf("%f") happily accepts "nan" and "inf"; a non-finite offset would
    // poison every injected joint pose, and NaN also slips past the norm check
    // below (every comparison against NaN is false).
    if (!std::isfinite(pose.position.x) || !std::isfinite(pose.position.y) || !std::isfinite(pose.position.z) ||
        !std::isfinite(pose.orientation.x) || !std::isfinite(pose.orientation.y) ||
        !std::isfinite(pose.orientation.z) || !std::isfinite(pose.orientation.w))
    {
        std::cerr << "WujiGlovePlugin: " << name << " ('" << value
                  << "') has non-finite values; using the built-in offset" << std::endl;
        return fallback;
    }
    const float norm = std::sqrt(pose.orientation.x * pose.orientation.x + pose.orientation.y * pose.orientation.y +
                                 pose.orientation.z * pose.orientation.z + pose.orientation.w * pose.orientation.w);
    if (norm < 1e-6f)
    {
        std::cerr << "WujiGlovePlugin: " << name << " has a zero quaternion; using the built-in offset" << std::endl;
        return fallback;
    }
    pose.orientation.x /= norm;
    pose.orientation.y /= norm;
    pose.orientation.z /= norm;
    pose.orientation.w /= norm;
    return pose;
}

// Wrist-source selection: WUJI_GLOVE_WRIST_SOURCE = auto (default) |
// hand_tracking | controller.
plugin_utils::WristSourceMode wrist_source_mode_from_env()
{
    const char* value = std::getenv("WUJI_GLOVE_WRIST_SOURCE");
    if (value == nullptr || *value == '\0' || std::strcmp(value, "auto") == 0)
    {
        return plugin_utils::WristSourceMode::Auto;
    }
    if (std::strcmp(value, "hand_tracking") == 0)
    {
        return plugin_utils::WristSourceMode::HandTracking;
    }
    if (std::strcmp(value, "controller") == 0)
    {
        return plugin_utils::WristSourceMode::Controller;
    }
    std::cerr << "WujiGlovePlugin: unknown WUJI_GLOVE_WRIST_SOURCE '" << value << "', using 'auto'" << std::endl;
    return plugin_utils::WristSourceMode::Auto;
}

// Resolve the glove's hand side explicitly via the device's "hand_side" GET
// (a NUL-terminated "left"/"right" string) — never inferred from frame contents.
std::optional<bool> query_is_left(WujiDevice* dev)
{
    char buf[16] = { 0 };
    size_t needed = 0;
    if (wuji_glove_get_hand_side(dev, buf, sizeof(buf), &needed) != WUJI_STATUS_OK)
    {
        return std::nullopt;
    }
    if (std::strcmp(buf, "left") == 0)
    {
        return true;
    }
    if (std::strcmp(buf, "right") == 0)
    {
        return false;
    }
    return std::nullopt;
}

const char* safe_err()
{
    const char* e = wuji_last_error();
    return e ? e : "(no error message)";
}

} // namespace

WujiGlovePlugin::WujiGlovePlugin(const std::string& plugin_root_id) noexcept(false) : m_root_id(plugin_root_id)
{
    std::cout << "Initializing WujiGlovePlugin with root: " << m_root_id << std::endl;

    // The glove itself is not an OpenXR upstream tracker — it is read
    // out-of-band via wuji_sdk. The tracker list carries only what the wrist
    // source needs (the controller tracker for the aim-pose fallback); the
    // OpenXR session exists for the push-device (injection) extension, the
    // wrist-source queries, and the XrTime base.
    plugin_utils::WristSourceConfig wrist_config;
    wrist_config.mode = wrist_source_mode_from_env();
    wrist_config.left_aim_to_wrist = pose_from_env("WUJI_GLOVE_AIM_TO_WRIST_LEFT", kLeftAimToWrist);
    wrist_config.right_aim_to_wrist = pose_from_env("WUJI_GLOVE_AIM_TO_WRIST_RIGHT", kRightAimToWrist);
    auto wrist_requirements = plugin_utils::WristPoseSource::collect_requirements(wrist_config.mode);

    std::vector<std::shared_ptr<core::ITracker>> trackers = wrist_requirements.trackers;
    auto extensions = core::DeviceIOSession::get_required_extensions(trackers);
    extensions.push_back(XR_NVX1_DEVICE_INTERFACE_BASE_EXTENSION_NAME);
    extensions.insert(extensions.end(), wrist_requirements.extensions.begin(), wrist_requirements.extensions.end());

    m_session = std::make_shared<core::OpenXRSession>("WujiGlove", extensions);
    const auto handles = m_session->get_handles();

    m_deviceio_session = core::DeviceIOSession::run(trackers, handles);
    m_time_converter.emplace(handles);
    m_wrist_source = std::make_unique<plugin_utils::WristPoseSource>(
        wrist_config, handles, m_deviceio_session.get(), wrist_requirements.controller_tracker);

    WujiInitOptions init_opts{};
    init_opts.log_level = 2; // warn only; the plugin reports connection and errors itself
    if (wuji_init(&init_opts) != WUJI_STATUS_OK)
    {
        throw std::runtime_error(std::string("WujiGlovePlugin: wuji_init failed: ") + safe_err());
    }

    m_running = true;
    try
    {
        m_worker_thread = std::thread(&WujiGlovePlugin::worker_thread, this);
        m_connection_thread = std::thread(&WujiGlovePlugin::connection_thread, this);
    }
    catch (...)
    {
        m_running = false;
        if (m_worker_thread.joinable())
        {
            m_worker_thread.join();
        }
        if (m_connection_thread.joinable())
        {
            m_connection_thread.join();
        }
        wuji_shutdown();
        throw;
    }
    std::cout << "WujiGlovePlugin initialized and running" << std::endl;
}

WujiGlovePlugin::~WujiGlovePlugin()
{
    std::cout << "Shutting down WujiGlovePlugin..." << std::endl;
    m_running = false;
    if (m_connection_thread.joinable())
    {
        m_connection_thread.join();
    }
    if (m_worker_thread.joinable())
    {
        m_worker_thread.join();
    }
    wuji_shutdown();
}

bool WujiGlovePlugin::is_running() const noexcept
{
    return m_running.load(std::memory_order_acquire);
}

bool WujiGlovePlugin::has_failed() const noexcept
{
    return m_failed.load(std::memory_order_acquire);
}

bool WujiGlovePlugin::connect_glove(GloveConnection& connection)
{
    WujiConnectTarget target{};
    target.kind = WUJI_CONNECT_TARGET_KIND_SN;
    target.value = connection.serial.c_str();

    WujiDevice* device = nullptr;
    if (wuji_connect(&target, connection.serial.c_str(), nullptr, &device) != WUJI_STATUS_OK)
    {
        std::cerr << "WujiGlovePlugin: wuji_connect(" << connection.serial << ") failed: " << safe_err() << std::endl;
        return false;
    }

    const std::optional<bool> is_left = query_is_left(device);
    if (!is_left.has_value())
    {
        std::cerr << "WujiGlovePlugin: could not determine hand_side for " << connection.serial << std::endl;
        wuji_dev_disconnect(device);
        wuji_dev_release(device);
        return false;
    }

    // The plugin assumes a single left/right glove pair: hand slots are keyed
    // by side only, so a second same-side glove would interleave
    // last-writer-wins into one injected hand. Drop the newcomer here, before
    // it subscribes (it must never touch the shared hand slot), and skip it in
    // future scans until the plugin restarts.
    const auto same_side = std::find_if(m_connections.begin(), m_connections.end(),
                                        [&connection, &is_left](const auto& other)
                                        {
                                            return other.get() != &connection && other->context &&
                                                   other->subscription != nullptr && other->context->is_left == *is_left;
                                        });
    if (same_side != m_connections.end())
    {
        std::cerr << "WujiGlovePlugin: second " << (*is_left ? "left" : "right") << " glove " << connection.serial
                  << " discovered while " << (*same_side)->serial << " is bound; ignoring " << connection.serial
                  << std::endl;
        connection.ignored = true;
        wuji_dev_disconnect(device);
        wuji_dev_release(device);
        return false;
    }

    auto context = std::make_unique<SubContext>();
    context->self = this;
    context->is_left = *is_left;

    WujiSub* subscription = nullptr;
    if (wuji_glove_subscribe_hand_skeleton(device, &WujiGlovePlugin::skeleton_callback, context.get(), &subscription) !=
        WUJI_STATUS_OK)
    {
        std::cerr << "WujiGlovePlugin: subscribe hand_skeleton failed for " << connection.serial << ": " << safe_err()
                  << std::endl;
        wuji_dev_disconnect(device);
        wuji_dev_release(device);
        return false;
    }

    connection.device = device;
    connection.subscription = subscription;
    connection.context = std::move(context);
    std::cout << "WujiGlovePlugin: connected " << connection.serial << " ("
              << (connection.context->is_left ? "left" : "right") << ")" << std::endl;
    return true;
}

void WujiGlovePlugin::disconnect_glove(GloveConnection& connection)
{
    if (connection.context)
    {
        invalidate_hand(connection.context->is_left);
    }
    if (connection.subscription != nullptr)
    {
        // Closing joins the SDK callback thread; the callback context must stay
        // alive until this returns.
        wuji_sub_close(connection.subscription);
        connection.subscription = nullptr;
    }
    if (connection.context)
    {
        // An in-flight callback may have re-validated the slot between the
        // invalidate above and the close; invalidate again now that the
        // callback thread is joined.
        invalidate_hand(connection.context->is_left);
    }
    connection.context.reset();
    if (connection.device != nullptr)
    {
        wuji_dev_disconnect(connection.device);
        wuji_dev_release(connection.device);
        connection.device = nullptr;
    }
}

void WujiGlovePlugin::discover_gloves()
{
    WujiDiscovered* list = nullptr;
    size_t count = 0;
    if (wuji_scan(&list, &count) != WUJI_STATUS_OK)
    {
        std::cerr << "WujiGlovePlugin: wuji_scan failed: " << safe_err() << std::endl;
        // The SDK does not document list/count contents on failure; free only
        // what is provably allocated.
        if (list != nullptr)
        {
            wuji_discovered_free(list, count);
        }
        return;
    }

    for (size_t i = 0; i < count && m_running; ++i)
    {
        if (list[i].device_id != WUJI_DEVICE_TYPE_WUJI_GLOVE || list[i].serial_number[0] == '\0')
        {
            continue;
        }

        const std::string serial = list[i].serial_number;
        auto found = std::find_if(m_connections.begin(), m_connections.end(),
                                  [&serial](const auto& connection) { return connection->serial == serial; });
        if (found == m_connections.end())
        {
            auto connection = std::make_unique<GloveConnection>();
            connection->serial = serial;
            m_connections.push_back(std::move(connection));
            found = std::prev(m_connections.end());
        }

        GloveConnection& connection = **found;
        if (connection.ignored)
        {
            continue;
        }
        if (connection.subscription == nullptr)
        {
            connect_glove(connection);
        }
    }
    wuji_discovered_free(list, count);
}

void WujiGlovePlugin::connection_thread()
{
    while (m_running)
    {
        for (auto& connection : m_connections)
        {
            if (connection->context && connection->context->terminal.load(std::memory_order_acquire))
            {
                std::cout << "WujiGlovePlugin: disconnected " << connection->serial << std::endl;
                disconnect_glove(*connection);
            }
        }

        discover_gloves();

        // Sleep in kShutdownPollInterval slices so shutdown stays responsive
        // while re-scanning every kDiscoveryInterval.
        for (auto waited = std::chrono::milliseconds(0); waited < kDiscoveryInterval && m_running;
             waited += kShutdownPollInterval)
        {
            std::this_thread::sleep_for(kShutdownPollInterval);
        }
    }

    for (auto& connection : m_connections)
    {
        disconnect_glove(*connection);
    }
    m_connections.clear();
}

// Runs on the wuji_sdk subscription worker thread. `frame` is valid only for the
// duration of this call.
void WujiGlovePlugin::skeleton_callback(WujiFrameKind kind, const WujiHandSkeleton* frame, void* user_data)
{
    if (user_data == nullptr)
    {
        return;
    }
    auto* ctx = static_cast<SubContext*>(user_data);
    if (kind == WUJI_FRAME_KIND_OK && frame != nullptr)
    {
        ctx->self->on_skeleton(frame, ctx->is_left);
    }
    else if (kind == WUJI_FRAME_KIND_END || kind == WUJI_FRAME_KIND_ERROR)
    {
        // Closing from this SDK callback thread would self-deadlock. Signal the
        // connection thread to own cleanup and resubscription instead.
        ctx->self->invalidate_hand(ctx->is_left);
        ctx->terminal.store(true, std::memory_order_release);
    }
}

void WujiGlovePlugin::on_skeleton(const WujiHandSkeleton* frame, bool is_left)
{
    std::array<XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT> joints{};
    if (!convert_skeleton(frame, is_left, joints))
    {
        return;
    }

    std::lock_guard<std::mutex> lock(m_frame_mutex);
    HandFrame& slot = is_left ? m_left : m_right;
    slot.joints = joints;
    slot.valid = true;
    slot.stamp = std::chrono::steady_clock::now();
}

void WujiGlovePlugin::invalidate_hand(bool is_left)
{
    std::lock_guard<std::mutex> lock(m_frame_mutex);
    HandFrame& slot = is_left ? m_left : m_right;
    slot.valid = false;
}

void WujiGlovePlugin::pump_hand(std::unique_ptr<plugin_utils::HandInjector>& injector,
                                XrHandEXT hand,
                                const HandFrame& frame,
                                XrTime time)
{
    // Treat data older than 200 ms as "hand absent": drop the injector so the
    // runtime reports isActive=false rather than a frozen pose.
    using namespace std::chrono;
    const bool fresh = frame.valid && (steady_clock::now() - frame.stamp) < kStaleThreshold;
    if (!fresh)
    {
        injector.reset();
        return;
    }
    if (!injector)
    {
        const auto handles = m_session->get_handles();
        injector = std::make_unique<plugin_utils::HandInjector>(handles.instance, handles.session, hand, handles.space);
    }

    // Fuse the device wrist pose: place the wrist-relative skeleton at the
    // fused wrist, and set TRACKED bits only while the wrist source is
    // actively tracked. With no wrist source available the skeleton stays
    // wrist-relative at the space origin with VALID-only flags (honest
    // degradation: consumers see the shape but know the pose is untracked).
    plugin_utils::WristSample wrist;
    if (m_wrist_source)
    {
        wrist = m_wrist_source->query(hand == XR_HAND_LEFT_EXT, time);
    }

    std::array<XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT> joints = frame.joints;
    if (wrist.valid)
    {
        for (auto& joint : joints)
        {
            if ((joint.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) == 0)
            {
                continue;
            }
            joint.pose = oxr_utils::multiply_poses(wrist.pose, joint.pose);
            if (wrist.tracked)
            {
                joint.locationFlags |= kPoseTrackedFlags;
            }
        }
    }
    injector->push(joints.data(), time);
}

void WujiGlovePlugin::worker_thread()
{
    while (m_running)
    {
        try
        {
            m_deviceio_session->update();
        }
        catch (const std::exception& e)
        {
            std::cerr << "WujiGlovePlugin update error: " << e.what() << std::endl;
            m_left_injector.reset();
            m_right_injector.reset();
            m_failed.store(true, std::memory_order_release);
            m_running.store(false, std::memory_order_release);
            return;
        }

        const XrTime time = m_time_converter->os_monotonic_now();

        HandFrame left_copy;
        HandFrame right_copy;
        {
            std::lock_guard<std::mutex> lock(m_frame_mutex);
            left_copy = m_left;
            right_copy = m_right;
        }

        pump_hand(m_left_injector, XR_HAND_LEFT_EXT, left_copy, time);
        pump_hand(m_right_injector, XR_HAND_RIGHT_EXT, right_copy, time);

        std::this_thread::sleep_for(kFramePeriod);
    }
}

} // namespace wuji_glove
} // namespace plugins
