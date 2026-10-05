// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/deviceio_session/deviceio_session.hpp"

#include <live_trackers/live_deviceio_factory.hpp>
#include <mcap/writer.hpp>
#include <openxr/openxr.h>
#include <oxr_utils/os_time.hpp>

#include <stdexcept>

namespace core
{

// ============================================================================
// DeviceIOSession Implementation
// ============================================================================

namespace
{

// Identify a tracker in an error message by its name, tolerating a null pointer.
std::string tracker_name_for_error(const ITracker* tracker)
{
    return tracker ? std::string(tracker->get_name()) : std::string("<null>");
}

bool tracker_in_list(const std::vector<std::shared_ptr<ITracker>>& trackers, const ITracker* tracker_ptr)
{
    for (const auto& t : trackers)
    {
        if (t.get() == tracker_ptr)
            return true;
    }
    return false;
}

// Fully validate a vendor config against the session's tracker list before anything consumes it.
// DeviceIOSession is the single owner of vendor validation: the live factory assumes a validated
// config and treats an invalid one as undefined behavior. Two parts: the tracker-list presence
// check (done here since the session holds the list) and the dispatch-driven vendor-validity check
// (delegated to validate_vendor_selections(), which owns the vendor dispatch table).
void validate_vendor_config(const std::vector<std::shared_ptr<ITracker>>& trackers,
                            const std::vector<std::pair<const ITracker*, TrackerVendor>>& tracker_vendors)
{
    for (const auto& [tracker_ptr, vendor] : tracker_vendors)
    {
        if (!tracker_in_list(trackers, tracker_ptr))
        {
            throw std::invalid_argument("DeviceIOSession: vendor selection '" + vendor.id + "' references tracker '" +
                                        tracker_name_for_error(tracker_ptr) + "' that is not in the trackers list");
        }
    }
    validate_vendor_selections(tracker_vendors);
}

} // namespace

DeviceIOSession::DeviceIOSession(const std::vector<std::shared_ptr<ITracker>>& trackers,
                                 const OpenXRSessionHandles& handles,
                                 std::optional<McapRecordingConfig> recording_config,
                                 VendorConfig vendor_config)
    : handles_(handles)
{
    std::vector<std::pair<const ITracker*, std::string>> tracker_names;

    // Validate up front, before the MCAP writer opens below, so an invalid config leaves no
    // stray recording file on disk.
    validate_vendor_config(trackers, vendor_config.tracker_vendors);

    if (recording_config)
    {
        for (const auto& [tracker_ptr, name] : recording_config->tracker_names)
        {
            if (!tracker_in_list(trackers, tracker_ptr))
            {
                throw std::invalid_argument("DeviceIOSession: McapRecordingConfig references tracker '" + name +
                                            "' that is not in the trackers list");
            }
        }

        mcap_writer_ = std::make_unique<mcap::McapWriter>();
        mcap::McapWriterOptions options("teleop");
        options.compression = mcap::Compression::None;

        auto status = mcap_writer_->open(recording_config->filename, options);
        if (!status.ok())
        {
            throw std::runtime_error("DeviceIOSession: failed to open MCAP file '" + recording_config->filename +
                                     "': " + status.message);
        }
        logger_->info("DeviceIOSession: recording to {}", recording_config->filename);

        tracker_names = std::move(recording_config->tracker_names);
    }

    LiveDeviceIOFactory factory(handles_, mcap_writer_.get(), tracker_names, vendor_config.tracker_vendors);

    for (const auto& tracker : trackers)
    {
        if (!tracker)
        {
            throw std::invalid_argument("DeviceIOSession: null tracker in trackers list");
        }
        tracker_impls_.emplace(tracker.get(), factory.create_tracker_impl(*tracker));
    }
}

DeviceIOSession::~DeviceIOSession() = default;

std::vector<std::string> DeviceIOSession::get_required_extensions(const std::vector<std::shared_ptr<ITracker>>& trackers,
                                                                  const VendorConfig& vendor_config)
{
    // Validate here too, so an extension query rejects a bad vendor config just like construction.
    validate_vendor_config(trackers, vendor_config.tracker_vendors);
    return LiveDeviceIOFactory::get_required_extensions(trackers, vendor_config.tracker_vendors);
}

bool DeviceIOSession::requires_openxr(const std::vector<std::shared_ptr<ITracker>>& trackers,
                                      const VendorConfig& vendor_config)
{
    validate_vendor_config(trackers, vendor_config.tracker_vendors);
    return LiveDeviceIOFactory::requires_openxr(trackers, vendor_config.tracker_vendors);
}

std::unique_ptr<DeviceIOSession> DeviceIOSession::run(const std::vector<std::shared_ptr<ITracker>>& trackers,
                                                      const OpenXRSessionHandles& handles,
                                                      std::optional<McapRecordingConfig> recording_config,
                                                      VendorConfig vendor_config)
{
    const bool has_handles = handles.instance != XR_NULL_HANDLE && handles.session != XR_NULL_HANDLE &&
                             handles.space != XR_NULL_HANDLE && handles.xrGetInstanceProcAddr != nullptr;
    if (!has_handles && requires_openxr(trackers, vendor_config))
    {
        throw std::invalid_argument(
            "DeviceIOSession::run: OpenXR session handles are required (instance, session, space and "
            "xrGetInstanceProcAddr must be set) because at least one tracker needs OpenXR");
    }

    isaaccapture::Logger::get("isaaccapture.core.DeviceIOSession")
        ->info("DeviceIOSession: Creating session with {} trackers", trackers.size());

    return std::unique_ptr<DeviceIOSession>(
        new DeviceIOSession(trackers, handles, std::move(recording_config), std::move(vendor_config)));
}

void DeviceIOSession::update()
{
    const int64_t monotonic_ns = os_monotonic_now_ns();

    for (auto& kv : tracker_impls_)
    {
        kv.second->update(monotonic_ns);
    }
}

} // namespace core
