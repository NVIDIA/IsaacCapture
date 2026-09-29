// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <deviceio_base/tracker_vendor.hpp>

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mcap
{
class McapWriter;
} // namespace mcap

namespace core
{

class ITracker;
class ITrackerImpl;
class ControllerTracker;
class IControllerTrackerImpl;
class MessageChannelTracker;
class IMessageChannelTrackerImpl;
class FullBodyTracker;
class IFullBodyTrackerImpl;
class TensorPushTracker;
class ITensorPushTrackerImpl;
class HandTracker;
class IHandTrackerImpl;
class HeadTracker;
class IHeadTrackerImpl;
class HapticCommandReaderTracker;
class IHapticCommandReaderTrackerImpl;
class KeyboardTracker;
class IKeyboardTrackerImpl;
struct OpenXRSessionHandles;

// Forward decls for trackers declared in deviceio_trackers/trackers.toml. Generated at
// configure time; do not add a row here for a manifest tracker.
#include "generated_tracker_forward_decls.inc"

/**
 * @brief Factory for live OpenXR tracker implementations.
 *
 * Used by DeviceIOSession to construct OpenXR-backed tracker implementations.
 * When writer is non-null, each simple impl receives a typed McapTrackerChannels
 * for MCAP recording.
 */
class LiveDeviceIOFactory
{
public:
    /**
     * @brief Aggregate OpenXR extensions required by the given trackers for a live session.
     *
     * Each tracker resolves its required extensions through the dispatch table using the vendor
     * id selected in @p tracker_vendors (or its default vendor when unlisted).
     *
     * @pre @p tracker_vendors is a validated vendor config (see validate_vendor_selections()).
     *      Passing an invalid config is undefined behavior; DeviceIOSession validates before
     *      calling this.
     */
    static std::vector<std::string> get_required_extensions(
        const std::vector<std::shared_ptr<ITracker>>& trackers,
        const std::vector<std::pair<const ITracker*, TrackerVendor>>& tracker_vendors = {});

    /** Create tracker impl from a tracker instance using the same dispatch as extension discovery. */
    std::unique_ptr<ITrackerImpl> create_tracker_impl(const ITracker& tracker);

    // @pre @p tracker_vendors is a validated vendor config (see validate_vendor_selections()).
    // The factory assumes validity; passing an invalid config is undefined behavior.
    // DeviceIOSession validates before constructing the factory.
    LiveDeviceIOFactory(const OpenXRSessionHandles& handles,
                        mcap::McapWriter* writer,
                        const std::vector<std::pair<const ITracker*, std::string>>& tracker_names,
                        const std::vector<std::pair<const ITracker*, TrackerVendor>>& tracker_vendors = {});

    std::unique_ptr<IHeadTrackerImpl> create_head_tracker_impl(const HeadTracker* tracker);
    std::unique_ptr<IHandTrackerImpl> create_hand_tracker_impl(const HandTracker* tracker);
    std::unique_ptr<IControllerTrackerImpl> create_controller_tracker_impl(const ControllerTracker* tracker);
    std::unique_ptr<IMessageChannelTrackerImpl> create_message_channel_tracker_impl(const MessageChannelTracker* tracker);
    std::unique_ptr<IFullBodyTrackerImpl> create_full_body_tracker_pico_impl(const FullBodyTracker* tracker);
    std::unique_ptr<IFullBodyTrackerImpl> create_full_body_tracker_noitom_impl(const FullBodyTracker* tracker);
    std::unique_ptr<IFullBodyTrackerImpl> create_full_body_tracker_xsens_impl(const FullBodyTracker* tracker);
    std::unique_ptr<ITensorPushTrackerImpl> create_tensor_push_tracker_impl(const TensorPushTracker* tracker);
    std::unique_ptr<IHapticCommandReaderTrackerImpl> create_haptic_command_reader_tracker_impl(
        const HapticCommandReaderTracker* tracker);
    std::unique_ptr<IKeyboardTrackerImpl> create_keyboard_tracker_impl(const KeyboardTracker* tracker);
    // create_<name>_tracker_impl for every manifest tracker.
#include "generated_live_factory_declarations.inc"

private:
    // Per-tracker data resolved from the session config: MCAP channel base name (recording) and
    // vendor selection. A tracker appears only when it has one or the other.
    struct TrackerData
    {
        std::optional<std::string> name; // MCAP channel base name; absent -> not recorded.
        std::optional<TrackerVendor> vendor; // vendor selection; absent -> default vendor id.
    };

    bool should_record(const ITracker* tracker) const;
    std::string_view get_name(const ITracker* tracker) const;
    const TrackerVendor* find_vendor(const ITracker* tracker) const;

    const OpenXRSessionHandles& handles_;
    mcap::McapWriter* writer_;
    std::unordered_map<const ITracker*, TrackerData> tracker_data_;
};

/**
 * @brief Validate per-tracker vendor selections against the live vendor dispatch table.
 *
 * Rejects selections on tracker types that do not support vendors, unknown vendor ids, vendor
 * ids that belong to a different tracker type, unsupported vendor params, and duplicate entries.
 * Throws std::invalid_argument on the first violation. List-independent: the caller checks that
 * each selection references a tracker it owns.
 *
 * This owns the dispatch-driven vendor rules; the factory assumes a config that has passed here.
 * DeviceIOSession runs this (with its own tracker-list presence check) before opening any
 * recording output and before constructing the factory.
 */
void validate_vendor_selections(const std::vector<std::pair<const ITracker*, TrackerVendor>>& tracker_vendors);

} // namespace core
