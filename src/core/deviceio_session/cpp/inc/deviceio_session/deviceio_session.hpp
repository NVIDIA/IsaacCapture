// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/tracker.hpp>
#include <deviceio_base/tracker_vendor.hpp>
#include <log_bridge/logger.hpp>
#include <oxr_utils/oxr_funcs.hpp>
#include <oxr_utils/oxr_session_handles.hpp>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Forward declaration -- mcap::McapWriter is an implementation detail of DeviceIOSession.
// Consumers of deviceio_core do not need to link against mcap::mcap.
namespace mcap
{
class McapWriter;
} // namespace mcap

namespace core
{

/**
 * @brief MCAP recording configuration for DeviceIOSession.
 *
 * tracker_names maps each ITracker pointer to its MCAP channel base name.
 * Trackers not in the map receive no channel writer and skip recording.
 * Pass as std::optional<McapRecordingConfig> to DeviceIOSession::run();
 * std::nullopt disables recording.
 */
struct McapRecordingConfig
{
    std::string filename;
    std::vector<std::pair<const ITracker*, std::string>> tracker_names;
};

/**
 * @brief Per-session vendor selection for vendored trackers (live sessions only).
 *
 * tracker_vendors maps each ITracker pointer to the vendor (id + params) used to
 * source it. Trackers not listed fall back to their default vendor id. Vendor is a
 * live-session concern (native XR hardware vs. an external pushed-tensor plugin);
 * replay reads the recorded channel regardless of vendor. Pass to
 * DeviceIOSession::run() and get_required_extensions(); an empty config selects
 * every tracker's default vendor.
 */
struct VendorConfig
{
    std::vector<std::pair<const ITracker*, TrackerVendor>> tracker_vendors;
};

// OpenXR DeviceIO Session - manages trackers and optional MCAP recording.
// When a McapRecordingConfig is provided, the session owns and drives a
// mcap::McapWriter; each tracker impl registers its own channels and writes
// directly during update().
class DeviceIOSession : public ITrackerSession
{
public:
    // Static helper — required OpenXR extensions for the given trackers (live factory; not per-tracker API).
    // Vendored trackers resolve their extensions through the vendor id selected in vendor_config.
    static std::vector<std::string> get_required_extensions(const std::vector<std::shared_ptr<ITracker>>& trackers,
                                                            const VendorConfig& vendor_config = {});

    // Static helper — whether any of the trackers needs an OpenXR session. When false, run() accepts
    // default-constructed (null) handles, so a session of in-process devices needs no OpenXR runtime.
    static bool requires_openxr(const std::vector<std::shared_ptr<ITracker>>& trackers,
                                const VendorConfig& vendor_config = {});

    // Static factory - Create and initialize a session with trackers.
    // Optionally pass a McapRecordingConfig to enable automatic MCAP recording, and a
    // VendorConfig to select the vendor for any vendored trackers. Null handles are accepted only
    // when requires_openxr() is false; otherwise std::invalid_argument is thrown.
    static std::unique_ptr<DeviceIOSession> run(const std::vector<std::shared_ptr<ITracker>>& trackers,
                                                const OpenXRSessionHandles& handles,
                                                std::optional<McapRecordingConfig> recording_config = std::nullopt,
                                                VendorConfig vendor_config = {});

    // Destructor defined in .cpp where mcap::McapWriter is fully defined
    ~DeviceIOSession();

    /**
     * @brief Updates the session and all registered trackers.
     *
     * If recording is active, tracker implementations write MCAP samples
     * directly during this call.
     *
     * @throws std::runtime_error On critical tracker/runtime failures.
     * @note A thrown exception indicates a fatal condition; the application is
     *       expected to terminate rather than continue running.
     */
    void update();

    const ITrackerImpl& get_tracker_impl(const ITracker& tracker) const override
    {
        auto it = tracker_impls_.find(&tracker);
        if (it == tracker_impls_.end())
        {
            throw std::runtime_error("Tracker implementation not found for tracker: " + std::string(tracker.get_name()));
        }
        return *(it->second);
    }

private:
    DeviceIOSession(const std::vector<std::shared_ptr<ITracker>>& trackers,
                    const OpenXRSessionHandles& handles,
                    std::optional<McapRecordingConfig> recording_config,
                    VendorConfig vendor_config);

    const OpenXRSessionHandles handles_;
    std::shared_ptr<spdlog::logger> logger_ = isaaccapture::Logger::get("isaaccapture.core.DeviceIOSession");
    std::unordered_map<const ITracker*, std::unique_ptr<ITrackerImpl>> tracker_impls_;

    // Owned MCAP writer; null when recording is not configured.
    std::unique_ptr<mcap::McapWriter> mcap_writer_;
};

} // namespace core
