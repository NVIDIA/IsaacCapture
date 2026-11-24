// SPDX-FileCopyrightText: Copyright (c) 2026 HTC Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <log_bridge/logger.hpp>
#include <pusherio/schema_pusher.hpp>
#include <vut/vut_client.h>
#include <vut/vut_types.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace core
{
class OpenXRSession;
}

namespace plugins
{
namespace vive_se3_tracker
{

// Collection ids are "<prefix><name>", one collection per tracker. The reader
// (core::Se3Tracker) must be constructed with the same collection_id; the tensor
// identifier inside the collection is fixed to core::Se3Tracker::TENSOR_IDENTIFIER.
constexpr const char* kCollectionPrefix = "vive_tracker_";

// Default VIVEHub VUT daemon socket (README: /tmp/vut.sock, per-user).
constexpr const char* kDefaultVutSocket = "/tmp/vut.sock";

// Basename of the file the plugin advertises its live collection ids in (one per
// line) so readers (e.g. record_se3_vive.py) can auto-discover them. It lives in
// $XDG_RUNTIME_DIR (per-user, mode-0700) when set; otherwise in a private
// per-user dir created under /tmp (see resolve_collections_file) — never a fixed
// world-writable /tmp path, which would be open to symlink pre-planting.
// Overridable wholesale via VIVE_SE3_COLLECTIONS_FILE. Rewritten on each new
// collection, removed on exit. Keep the resolution in sync with record_se3_vive.py.
constexpr const char* kCollectionsFileName = "vive_se3_collections.txt";

// A tracker pose older than this (vs the pose's own monotonic timestamp) is
// treated as stale -> pushed with is_valid=false. Overridable via
// VIVE_SE3_STALE_MS.
constexpr int64_t kDefaultStaleMs = 250;

// Upper bound for VIVE_SE3_STALE_MS (1 hour). Values <= 0 or above this are
// rejected in favor of the default: 0 (e.g. a non-numeric env) would mark every
// sample stale, and a huge value would overflow the * 1'000'000 to nanoseconds.
constexpr int64_t kMaxStaleMs = 3600000;

// Synthetic mode (VIVE_SE3_SYNTHETIC=1): generate slow circular motion for a
// few fake trackers instead of connecting to VIVEHub — no-hardware smoke test
// of the push -> read path (pair with se3_printer).

/*!
 * @brief Reads live Vive Ultimate Tracker poses from the VIVEHub `tracker_server`
 *        daemon (via the VUT SDK client) and pushes each tracker as its own SE3
 *        tracker stream (se3_tracker.fbs) — one tensor collection per device.
 *
 * Data flow: tracker_server (dongle -> trackers) -> /tmp/vut.sock -> vut::Client
 * -> on_vut_pose() (receiver thread) -> latest-pose store -> update() (push
 * thread) -> Se3TrackerPose -> per-device tensor collection -> core::Se3Tracker.
 *
 * Naming: each collection is named by the tracker's physical serial number —
 * "<prefix><serial>" (e.g. "vive_tracker_SN0001"), the only stable
 * identity. The serial arrives on the pose wire (VUT SDK >= 1.0.1, Pose::serial);
 * a pose with an empty serial falls back to "<prefix><device_id>". Mapping a
 * tracker to a role is a downstream concern, keyed by serial wherever the poses
 * are consumed.
 *
 * Timestamps: VUT pose timestamps are host CLOCK_MONOTONIC (vut_types.h), the
 * same clock SchemaPusher documents as the local common clock, so the VUT
 * sample time is passed through verbatim — no push-time resampling bias. The
 * dongle's raw clock is not exposed by the VUT SDK, so the same value is used
 * as the documented best-effort substitute.
 *
 * Per-device push policy (per ~90 Hz tick):
 * - new sample since last push -> push is_valid=true at the VUT sample time
 * - unchanged & fresh          -> push nothing (live readers retain last-known;
 *                                 avoids duplicate-timestamp samples in MCAP)
 * - stale                      -> push is_valid=false + identity filler at the
 *                                 current time (explicit invalidity beats
 *                                 silence)
 */
class ViveSe3TrackerPlugin
{
public:
    ViveSe3TrackerPlugin();
    ~ViveSe3TrackerPlugin();

    ViveSe3TrackerPlugin(const ViveSe3TrackerPlugin&) = delete;
    ViveSe3TrackerPlugin& operator=(const ViveSe3TrackerPlugin&) = delete;
    ViveSe3TrackerPlugin(ViveSe3TrackerPlugin&&) = delete;
    ViveSe3TrackerPlugin& operator=(ViveSe3TrackerPlugin&&) = delete;

    // Push pending tracker samples. Call at the desired rate (~90 Hz).
    void update();

private:
    // VUT receiver-thread callback: store the latest pose for a device_id.
    void on_vut_pose(const vut::Pose& p);

    // Synthetic pose generator: writes fake latest_poses_ entries (timestamps =
    // now, so they are never stale).
    void generate_synthetic_poses(int64_t now_ns);

    // Rewrite collections_file_ with the current set of live collection ids so
    // readers can auto-discover them without knowing serials in advance.
    void write_collections_file() const;

    // Per-device push state (pusher created lazily on first pose).
    struct DeviceStream
    {
        std::unique_ptr<core::SchemaPusher> pusher;
        std::string collection_id; // "vive_tracker_<serial>" (or device_id fallback)
        std::string serial; // serial the collection was named from
        int64_t last_pushed_ts_ns = -1;
        bool pushed_invalid = false; // logged transition into stale
    };
    // Create (or fetch) the stream for a device. serial comes from the pose wire
    // (new VUT SDK); an empty serial falls back to device_id naming. The name is
    // fixed at creation from the first pose's serial (collections can't rename);
    // if the daemon later reassigns this device_id to a different serial, the
    // stream is rebuilt so samples never land in the previous tracker's collection.
    DeviceStream& stream_for(uint32_t device_id, const std::string& serial);

    std::shared_ptr<spdlog::logger> logger_ =
        isaaccapture::Logger::get("isaaccapture.plugins.vive_se3_tracker.ViveSe3TrackerPlugin");

    std::shared_ptr<core::OpenXRSession> session_;

    // --- VIVEHub VUT client ---
    std::unique_ptr<vut::Client> vut_client_;
    bool synthetic_mode_ = false;
    int64_t start_time_ns_ = 0;
    int64_t stale_ns_ = kDefaultStaleMs * 1000000;
    std::optional<std::string> collections_file_; // live collection-id advertisement path

    // Latest pose per device_id, written by the VUT receiver thread, snapshotted
    // by the push thread. Guarded by pose_mutex_.
    struct LatestPose
    {
        float pos[3] = { 0.f, 0.f, 0.f };
        float quat[4] = { 0.f, 0.f, 0.f, 1.f };
        std::string serial; // tracker serial from the pose wire ("" if unknown)
        int64_t ts_ns = 0;
        bool have = false;
    };
    std::mutex pose_mutex_;
    std::unordered_map<uint32_t, LatestPose> latest_poses_;

    std::unordered_map<uint32_t, DeviceStream> streams_;
};

} // namespace vive_se3_tracker
} // namespace plugins
