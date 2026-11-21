// SPDX-FileCopyrightText: Copyright (c) 2026 HTC Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "vive_se3_tracker_plugin.hpp"

#include <deviceio_trackers/se3_tracker.hpp>
#include <flatbuffers/flatbuffers.h>
#include <log_bridge/logger.hpp>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/os_time.hpp>
#include <schema/se3_tracker_generated.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace plugins
{
namespace vive_se3_tracker
{

namespace
{

int64_t env_int(const char* name, int64_t fallback)
{
    const char* v = std::getenv(name);
    if (!v || !*v)
        return fallback;
    return std::strtoll(v, nullptr, 10);
}

// A directory is safe to place the advertisement file in only if it is a real
// directory (not a symlink) that we own with no group/other write access — so no
// other local user can pre-plant a symlink and redirect our truncating open.
bool is_secure_dir(const std::string& dir)
{
    struct stat st
    {
    };
    if (::lstat(dir.c_str(), &st) != 0)
        return false;
    if (!S_ISDIR(st.st_mode))
        return false; // e.g. a symlink or a regular file planted at that path
    if (st.st_uid != ::geteuid())
        return false; // not owned by us
    if (st.st_mode & (S_IWGRP | S_IWOTH))
        return false; // group/other-writable
    return true;
}

// Resolve where to write the collection-id advertisement file, securely:
//   1. VIVE_SE3_COLLECTIONS_FILE, if set, wins verbatim (caller's responsibility).
//   2. else $XDG_RUNTIME_DIR (systemd gives a per-user, mode-0700 dir).
//   3. else a private /tmp/vive_se3_tracker-<uid> we create 0700 and verify we own.
// Disables the advertisement if the private directory cannot be secured.
// Keep in sync with record_se3_vive.py.
std::string resolve_collections_file()
{
    const char* env_cf = std::getenv("VIVE_SE3_COLLECTIONS_FILE");
    if (env_cf && *env_cf)
        return env_cf;

    const char* xdg = std::getenv("XDG_RUNTIME_DIR");
    if (xdg && *xdg && is_secure_dir(xdg))
        return std::string(xdg) + "/" + kCollectionsFileName;

    const std::string dir = "/tmp/vive_se3_tracker-" + std::to_string(::geteuid());
    ::mkdir(dir.c_str(), 0700); // ignore EEXIST; is_secure_dir does the real check
    if (is_secure_dir(dir))
        return dir + "/" + kCollectionsFileName;

    isaaccapture::Logger::get("isaaccapture.plugins.vive_se3_tracker.ViveSe3TrackerPlugin")
        ->warn(
            "could not secure a per-user directory for the collections file; set VIVE_SE3_COLLECTIONS_FILE to a "
            "private path.");
    return {};
}

core::SchemaPusherConfig make_pusher_config(const std::string& collection_id)
{
    // Wire rendezvous (tensor identifier + buffer size) comes from the Se3Tracker facade —
    // the single source of truth shared with LiveSe3TrackerImpl; a mismatch is silent no-data.
    return core::SchemaPusherConfig{ .collection_id = collection_id,
                                     .max_flatbuffer_size = core::Se3Tracker::DEFAULT_MAX_FLATBUFFER_SIZE,
                                     .tensor_identifier = std::string(core::Se3Tracker::TENSOR_IDENTIFIER),
                                     .localized_name = "Vive Ultimate Tracker SE3",
                                     .app_name = "ViveSe3TrackerPlugin" };
}

} // namespace

ViveSe3TrackerPlugin::ViveSe3TrackerPlugin()
    : session_(
          std::make_shared<core::OpenXRSession>("ViveSe3TrackerPlugin", core::SchemaPusher::get_required_extensions())),
      start_time_ns_(core::os_monotonic_now_ns())
{
    // Clamp VIVE_SE3_STALE_MS before scaling to ns: a non-numeric env yields 0
    // (env_int), which would mark every sample stale, and a huge value would
    // overflow the * 1'000'000. Reject out-of-range values, use the default.
    const int64_t stale_ms = env_int("VIVE_SE3_STALE_MS", kDefaultStaleMs);
    if (stale_ms <= 0 || stale_ms > kMaxStaleMs)
    {
        if (std::getenv("VIVE_SE3_STALE_MS"))
            logger_->warn("ignoring invalid VIVE_SE3_STALE_MS={}; using {} ms", stale_ms, kDefaultStaleMs);
        stale_ns_ = kDefaultStaleMs * 1000000;
    }
    else
    {
        stale_ns_ = stale_ms * 1000000;
    }

    // Advertisement file, placed in a per-user directory (never a fixed
    // world-writable /tmp path — see resolve_collections_file).
    collections_file_ = resolve_collections_file();
    // Remove any stale advertisement from a previous run; recreated lazily as
    // collections come up. Prevents readers from subscribing to dead streams.
    std::remove(collections_file_.c_str());

    const char* synth = std::getenv("VIVE_SE3_SYNTHETIC");
    synthetic_mode_ = (synth && (synth[0] == '1' || synth[0] == 't' || synth[0] == 'T'));
    if (synthetic_mode_)
    {
        logger_->info("SYNTHETIC mode (VIVE_SE3_SYNTHETIC=1); not connecting to VIVEHub.");
        return;
    }

    const char* env_sock = std::getenv("VIVE_VUT_SOCKET");
    const std::string socket_path = (env_sock && *env_sock) ? env_sock : kDefaultVutSocket;

    logger_->info("connecting to VIVEHub VUT daemon at {} (stale threshold {} ms)", socket_path, stale_ns_ / 1000000);

    vut_client_ = std::make_unique<vut::Client>(socket_path);
    vut_client_->set_auto_reconnect(true, 200, 2000);
    vut_client_->on_connection_change([this](bool up)
                                      { logger_->info("VIVEHub daemon {}", up ? "CONNECTED" : "disconnected"); });
    vut_client_->on_pose([this](const vut::Pose& p) { on_vut_pose(p); });

    // Non-fatal: with auto-reconnect the supervisor keeps retrying if the daemon
    // isn't up yet. Poses simply won't flow until it connects and a tracker runs.
    if (vut_client_->connect("ViveSe3TrackerPlugin", vut::SUB_POSE) != 0)
        logger_->info("daemon not up yet; retrying in background.");

    // Best-effort device inventory (may be empty until trackers pair). The serial
    // — and thus the collection name — only arrives on the pose wire, so it is
    // resolved on the first pose, not here (list_devices carries no serial).
    std::vector<vut::Device> devs;
    if (vut_client_->list_devices(devs, 500) == 0 && !devs.empty())
    {
        logger_->info("devices reported by daemon:");
        for (const auto& d : devs)
            logger_->info("    id={} state={} name=\"{}\"", d.device_id, static_cast<int>(d.state), d.name);
    }
}

ViveSe3TrackerPlugin::~ViveSe3TrackerPlugin()
{
    // disconnect() stops the VUT SDK's I/O thread and JOINS it before returning
    // (SDK contract; verified: it does shutdown(fd) + std::thread::join()). The
    // pose callback runs only on that thread, so once this returns on_vut_pose()
    // can no longer fire and the members it touches are destroyed safely
    // afterwards. We never call disconnect() from inside the callback, so there
    // is no self-join.
    if (vut_client_)
        vut_client_->disconnect();
    if (!collections_file_.empty())
        std::remove(collections_file_.c_str());
}

// Sanitize a serial taken straight off the VUT wire before it becomes a
// collection id and a line in the advertisement file that record_se3_vive.py
// trusts. Bounds the read to the fixed buffer (no reliance on NUL-termination)
// and keeps only filename-safe characters — alphanumerics and '-' '_' '.',
// dropping everything else (newlines and separators included) so a malformed
// serial cannot inject or split lines.
static std::string sanitize_serial(const char* s, size_t maxlen)
{
    const size_t n = ::strnlen(s, maxlen);
    std::string out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
        const char c = s[i];
        const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' ||
                        c == '_' || c == '.';
        if (ok)
            out.push_back(c);
    }
    return out;
}

// Collection id for a tracker: its physical serial when present (the stable
// identity), else a device_id fallback for a pose that arrived with no serial.
static std::string make_collection_id(uint32_t device_id, const std::string& serial)
{
    if (!serial.empty())
        return kCollectionPrefix + serial;
    return kCollectionPrefix + std::to_string(device_id);
}

void ViveSe3TrackerPlugin::on_vut_pose(const vut::Pose& p)
{
    // Runs on the VUT receiver thread — copy only, no blocking. Pushing happens
    // on the update() thread; SchemaPusher is never touched from here.
    bool first = false;
    std::string serial;
    {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        LatestPose& lp = latest_poses_[p.device_id];
        first = !lp.have;
        lp.pos[0] = p.position[0];
        lp.pos[1] = p.position[1];
        lp.pos[2] = p.position[2];
        lp.quat[0] = p.orientation[0];
        lp.quat[1] = p.orientation[1];
        lp.quat[2] = p.orientation[2];
        lp.quat[3] = p.orientation[3];
        // Serial arrives on the pose wire (VUT SDK >= 1.0.1), "" if unknown.
        // Sanitized + length-bounded here (see sanitize_serial) before it names a
        // collection. Fixed on the first pose — collections can't be renamed later.
        lp.serial = sanitize_serial(p.serial, sizeof(p.serial));
        lp.ts_ns = static_cast<int64_t>(p.timestamp_ns);
        lp.have = true;
        serial = lp.serial;
    }

    // Log outside the lock so the push thread never blocks on this I/O.
    if (first)
    {
        logger_->info("first pose from device_id={}{} -> collection '{}'", p.device_id,
                      serial.empty() ? " (no serial on wire; naming by device_id)" : "",
                      make_collection_id(p.device_id, serial));
    }
}

ViveSe3TrackerPlugin::DeviceStream& ViveSe3TrackerPlugin::stream_for(uint32_t device_id, const std::string& serial)
{
    auto it = streams_.find(device_id);
    if (it != streams_.end())
    {
        // Same device_id but a different serial means the daemon reassigned this
        // id to another physical tracker (re-pair / dongle reset). Rebuild so the
        // new tracker gets its own collection instead of inheriting the old name.
        if (!serial.empty() && !it->second.serial.empty() && it->second.serial != serial)
        {
            logger_->warn("device_id={} serial changed from '{}' to '{}'; creating a new collection", device_id,
                          it->second.serial, serial);
            streams_.erase(it);
        }
        else
        {
            return it->second;
        }
    }

    DeviceStream stream;
    stream.serial = serial;
    stream.collection_id = make_collection_id(device_id, serial);
    stream.pusher =
        std::make_unique<core::SchemaPusher>(session_->get_handles(), make_pusher_config(stream.collection_id));
    logger_->info("created tensor collection '{}' for device_id={}", stream.collection_id, device_id);
    DeviceStream& ref = streams_.emplace(device_id, std::move(stream)).first->second;
    write_collections_file(); // advertise the updated live set to readers
    return ref;
}

void ViveSe3TrackerPlugin::write_collections_file() const
{
    if (collections_file_.empty())
        return;
    // Write to a temp then rename so a concurrent reader never sees a partial file.
    const std::string tmp = collections_file_ + ".tmp";
    {
        std::ofstream out(tmp, std::ios::trunc);
        if (!out)
        {
            logger_->warn("could not write collections file {}", tmp);
            return;
        }
        for (const auto& kv : streams_)
            out << kv.second.collection_id << "\n";
    }
    if (std::rename(tmp.c_str(), collections_file_.c_str()) != 0)
    {
        logger_->warn("could not update collections file {}", collections_file_);
        std::remove(tmp.c_str()); // don't leave the temp behind
    }
}

void ViveSe3TrackerPlugin::generate_synthetic_poses(int64_t now_ns)
{
    // Slow circle per device, phase-offset by device_id; orientation spins about Y.
    const double t = static_cast<double>(now_ns - start_time_ns_) / 1e9;

    // Fake a trio (device_ids 1,2,3) with synthetic serials so the collections
    // are named as they would be live — enough to smoke-test the
    // push->record->replay path with no hardware.
    std::lock_guard<std::mutex> lock(pose_mutex_);
    for (uint32_t dev = 1; dev <= 3; ++dev)
    {
        const double phase = t * 0.5 + dev * 2.0;
        LatestPose& lp = latest_poses_[dev];
        lp.pos[0] = static_cast<float>(0.3 * std::cos(phase));
        lp.pos[1] = 0.2f * dev;
        lp.pos[2] = static_cast<float>(0.3 * std::sin(phase));
        const double half = phase * 0.5;
        lp.quat[0] = 0.f;
        lp.quat[1] = static_cast<float>(std::sin(half));
        lp.quat[2] = 0.f;
        lp.quat[3] = static_cast<float>(std::cos(half));
        lp.serial = "SYNTH0000000" + std::to_string(dev);
        lp.ts_ns = now_ns;
        lp.have = true;
    }
}

void ViveSe3TrackerPlugin::update()
{
    if (synthetic_mode_)
        generate_synthetic_poses(core::os_monotonic_now_ns());

    // Snapshot under the lock, push outside it (pushes go through OpenXR).
    std::unordered_map<uint32_t, LatestPose> snapshot;
    {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        snapshot = latest_poses_;
    }

    const int64_t now_ns = core::os_monotonic_now_ns();

    for (const auto& kv : snapshot)
    {
        const uint32_t device_id = kv.first;
        const LatestPose& lp = kv.second;
        if (!lp.have)
            continue;

        DeviceStream& stream = stream_for(device_id, lp.serial);
        const bool stale = (now_ns - lp.ts_ns > stale_ns_);

        core::Se3TrackerPoseT out;
        int64_t sample_time_ns;

        if (!stale)
        {
            if (lp.ts_ns == stream.last_pushed_ts_ns)
                continue; // no new sample; live readers retain last-known

            // VUT forwards poses in the SteamVR/OpenVR right-handed frame (X right,
            // Y up, -Z forward) with the daemon's own origin — a producer-defined
            // reference frame per se3_tracker.fbs (documented in README.md); passed
            // verbatim. Origin alignment vs the consumer's frame is downstream tuning.
            out.pose = std::make_shared<core::Pose>(core::Point(lp.pos[0], lp.pos[1], lp.pos[2]),
                                                    core::Quaternion(lp.quat[0], lp.quat[1], lp.quat[2], lp.quat[3]));
            out.is_valid = true;
            // VUT timestamps are host CLOCK_MONOTONIC — the local common clock —
            // so the sample time passes through without conversion. No raw dongle
            // clock is exposed; same value as the documented substitute.
            sample_time_ns = lp.ts_ns;
            stream.last_pushed_ts_ns = lp.ts_ns;
            if (stream.pushed_invalid)
            {
                stream.pushed_invalid = false;
                logger_->info("device_id={} tracking recovered", device_id);
            }
        }
        else
        {
            // Identity pose is a filler consistent with "pose contents unspecified
            // when is_valid == false" (se3_tracker.fbs) — consumers gate on is_valid,
            // never on pose values. Pushed every tick while stale, stamped with the
            // current time.
            out.pose =
                std::make_shared<core::Pose>(core::Point(0.0f, 0.0f, 0.0f), core::Quaternion(0.0f, 0.0f, 0.0f, 1.0f));
            out.is_valid = false;
            sample_time_ns = now_ns;
            if (!stream.pushed_invalid)
            {
                stream.pushed_invalid = true;
                logger_->warn("device_id={} stale (last sample {} ms ago) -> pushing is_valid=false", device_id,
                              (now_ns - lp.ts_ns) / 1000000);
            }
        }

        flatbuffers::FlatBufferBuilder builder(core::Se3Tracker::DEFAULT_MAX_FLATBUFFER_SIZE);
        auto offset = core::Se3TrackerPose::Pack(builder, &out);
        builder.Finish(offset);

        stream.pusher->push_buffer(builder.GetBufferPointer(), builder.GetSize(), sample_time_ns, sample_time_ns);
    }
}

} // namespace vive_se3_tracker
} // namespace plugins
