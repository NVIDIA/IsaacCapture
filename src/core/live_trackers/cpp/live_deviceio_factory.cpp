// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/live_trackers/live_deviceio_factory.hpp"

#include "generated_live_includes.inc"
#include "live_controller_tracker_impl.hpp"
#include "live_full_body_tracker_noitom_impl.hpp"
#include "live_full_body_tracker_pico_impl.hpp"
#include "live_full_body_tracker_xsens_impl.hpp"
#include "live_hand_tracker_impl.hpp"
#include "live_haptic_command_reader_tracker_impl.hpp"
#include "live_head_tracker_impl.hpp"
#include "live_keyboard_tracker_impl.hpp"
#include "live_message_channel_tracker_impl.hpp"
#include "live_tensor_push_tracker_impl.hpp"

#include <deviceio_trackers/controller_tracker.hpp>
#include <deviceio_trackers/full_body_tracker.hpp>
#include <deviceio_trackers/hand_tracker.hpp>
#include <deviceio_trackers/haptic_command_reader_tracker.hpp>
#include <deviceio_trackers/head_tracker.hpp>
#include <deviceio_trackers/keyboard_tracker.hpp>
#include <deviceio_trackers/message_channel_tracker.hpp>
#include <deviceio_trackers/tensor_push_tracker.hpp>
#include <oxr_utils/oxr_time.hpp>

#include <cassert>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace core
{

namespace
{

// Pure type probe for a dispatch row: true when `tracker` is dynamically a TrackerT.
// Kept separate from try_add_extensions so callers that only need the type test (e.g.
// tracker_supports_vendors) do not run -- and then discard -- the extension collection.
template <typename TrackerT>
bool is_tracker_type(const ITracker& tracker)
{
    return dynamic_cast<const TrackerT*>(&tracker) != nullptr;
}

template <typename TrackerT, typename ImplT>
bool try_add_extensions(const ITracker& tracker, std::set<std::string>& out)
{
    if (!is_tracker_type<TrackerT>(tracker))
        return false;
    for (const auto& ext : ImplT::required_extensions())
        out.insert(ext);
    return true;
}

std::unique_ptr<ITrackerImpl> try_create_head_impl(LiveDeviceIOFactory& factory, const ITracker& tracker)
{
    auto* typed = dynamic_cast<const HeadTracker*>(&tracker);
    return typed ? factory.create_head_tracker_impl(typed) : nullptr;
}

std::unique_ptr<ITrackerImpl> try_create_hand_impl(LiveDeviceIOFactory& factory, const ITracker& tracker)
{
    auto* typed = dynamic_cast<const HandTracker*>(&tracker);
    return typed ? factory.create_hand_tracker_impl(typed) : nullptr;
}

std::unique_ptr<ITrackerImpl> try_create_controller_impl(LiveDeviceIOFactory& factory, const ITracker& tracker)
{
    auto* typed = dynamic_cast<const ControllerTracker*>(&tracker);
    return typed ? factory.create_controller_tracker_impl(typed) : nullptr;
}

std::unique_ptr<ITrackerImpl> try_create_message_channel_impl(LiveDeviceIOFactory& factory, const ITracker& tracker)
{
    auto* typed = dynamic_cast<const MessageChannelTracker*>(&tracker);
    return typed ? factory.create_message_channel_tracker_impl(typed) : nullptr;
}

std::unique_ptr<ITrackerImpl> try_create_full_body_pico_impl(LiveDeviceIOFactory& factory, const ITracker& tracker)
{
    auto* typed = dynamic_cast<const FullBodyTracker*>(&tracker);
    return typed ? factory.create_full_body_tracker_pico_impl(typed) : nullptr;
}

std::unique_ptr<ITrackerImpl> try_create_full_body_noitom_impl(LiveDeviceIOFactory& factory, const ITracker& tracker)
{
    auto* typed = dynamic_cast<const FullBodyTracker*>(&tracker);
    return typed ? factory.create_full_body_tracker_noitom_impl(typed) : nullptr;
}

std::unique_ptr<ITrackerImpl> try_create_full_body_xsens_impl(LiveDeviceIOFactory& factory, const ITracker& tracker)
{
    auto* typed = dynamic_cast<const FullBodyTracker*>(&tracker);
    return typed ? factory.create_full_body_tracker_xsens_impl(typed) : nullptr;
}

std::unique_ptr<ITrackerImpl> try_create_tensor_push_impl(LiveDeviceIOFactory& factory, const ITracker& tracker)
{
    auto* typed = dynamic_cast<const TensorPushTracker*>(&tracker);
    return typed ? factory.create_tensor_push_tracker_impl(typed) : nullptr;
}

std::unique_ptr<ITrackerImpl> try_create_haptic_command_reader_impl(LiveDeviceIOFactory& factory, const ITracker& tracker)
{
    auto* typed = dynamic_cast<const HapticCommandReaderTracker*>(&tracker);
    return typed ? factory.create_haptic_command_reader_tracker_impl(typed) : nullptr;
}

std::unique_ptr<ITrackerImpl> try_create_keyboard_impl(LiveDeviceIOFactory& factory, const ITracker& tracker)
{
    auto* typed = dynamic_cast<const KeyboardTracker*>(&tracker);
    return typed ? factory.create_keyboard_tracker_impl(typed) : nullptr;
}

#include "generated_live_try_create.inc"

using CollectExtensionsFn = bool (*)(const ITracker&, std::set<std::string>&);
using TypeMatchesFn = bool (*)(const ITracker&);
using TryCreateFn = std::unique_ptr<ITrackerImpl> (*)(LiveDeviceIOFactory&, const ITracker&);

struct TrackerDispatchEntry
{
    CollectExtensionsFn collect_extensions;
    // Pure type probe for this row's tracker type (see is_tracker_type); no side effects.
    TypeMatchesFn matches;
    TryCreateFn try_create;
    // Vendor routing (last so single-vendor rows can omit it): a default-initialized row is a
    // type's sole vendor; multi-vendor types set vendor_id per row and list the default vendor
    // first (see k_tracker_dispatch).
    std::string_view vendor_id = {};
};

// Build a dispatch row for a (tracker type, impl) pair, wiring up its extension
// collector, type probe, and impl builder. vendor_id defaults to the non-vendored
// sentinel; pass it to declare a vendored row.
template <typename TrackerT, typename ImplT>
constexpr TrackerDispatchEntry make_dispatch_entry(TryCreateFn try_create, std::string_view vendor_id = {})
{
    return TrackerDispatchEntry{ &try_add_extensions<TrackerT, ImplT>, &is_tracker_type<TrackerT>, try_create, vendor_id };
}

// One row per (tracker type, vendor). A tracker type may have several vendor rows; the first row
// for a type is the one chosen when no vendor is selected. Extension discovery and impl creation
// both scan this table in order: keep rows whose vendor id matches the selection (or, when
// unselected, every row and let the type-checked thunk pick the first match), then the row's
// type-checked thunk builds the concrete impl.
inline const TrackerDispatchEntry k_tracker_dispatch[] = {
    make_dispatch_entry<HeadTracker, LiveHeadTrackerImpl>(&try_create_head_impl),
    make_dispatch_entry<HandTracker, LiveHandTrackerImpl>(&try_create_hand_impl),
    make_dispatch_entry<ControllerTracker, LiveControllerTrackerImpl>(&try_create_controller_impl),
    make_dispatch_entry<MessageChannelTracker, LiveMessageChannelTrackerImpl>(&try_create_message_channel_impl),
    make_dispatch_entry<FullBodyTracker, LiveFullBodyTrackerPicoImpl>(&try_create_full_body_pico_impl, "body.pico-xr"),
    make_dispatch_entry<FullBodyTracker, LiveFullBodyTrackerNoitomImpl>(
        &try_create_full_body_noitom_impl, LiveFullBodyTrackerNoitomImpl::VENDOR_ID),
    make_dispatch_entry<FullBodyTracker, LiveFullBodyTrackerXsensImpl>(
        &try_create_full_body_xsens_impl, LiveFullBodyTrackerXsensImpl::VENDOR_ID),
    make_dispatch_entry<TensorPushTracker, LiveTensorPushTrackerImpl>(&try_create_tensor_push_impl),
    make_dispatch_entry<HapticCommandReaderTracker, LiveHapticCommandReaderTrackerImpl>(
        &try_create_haptic_command_reader_impl),
    make_dispatch_entry<KeyboardTracker, LiveKeyboardTrackerImpl>(&try_create_keyboard_impl),
// Manifest trackers are single-vendor, so their rows can sit last as a block.
#include "generated_live_dispatch_rows.inc"
};

// Find a tracker's vendor selection in the config, or nullptr when unlisted.
const TrackerVendor* find_tracker_vendor(const std::vector<std::pair<const ITracker*, TrackerVendor>>& tracker_vendors,
                                         const ITracker* tracker)
{
    for (const auto& [ptr, vendor] : tracker_vendors)
    {
        if (ptr == tracker)
            return &vendor;
    }
    return nullptr;
}

// True when a dispatch row may be selected for a tracker. With a vendor selected, only its
// row matches. With none, every row is a candidate: the scanning loop breaks on the first
// type match, so the first row for the tracker's type -- its default vendor -- wins.
bool row_selected(const TrackerDispatchEntry& row, const TrackerVendor* selected)
{
    return selected ? (row.vendor_id == selected->id) : true;
}

// No dispatch row produced an impl for a tracker (and its selected vendor); report why.
[[noreturn]] void throw_unresolved_tracker(const char* context, const ITracker& tracker, const TrackerVendor* selected)
{
    if (selected)
    {
        throw std::invalid_argument(std::string(context) + ": no live vendor '" + selected->id + "' for tracker '" +
                                    std::string(tracker.get_name()) + "'");
    }
    throw std::invalid_argument(std::string(context) + ": unsupported tracker type '" +
                                std::string(tracker.get_name()) + "'");
}

// True when a dispatch row offers this vendor id, i.e. it names a live vendor a tracker can select.
bool dispatch_has_vendor(std::string_view vendor_id)
{
    // An empty id is the non-vendored-row sentinel (vendor_id = {}), never a selectable
    // vendor. Reject it here so an empty TrackerVendor id is reported up front as an
    // unknown vendor id instead of matching those sentinel rows and failing later.
    if (vendor_id.empty())
        return false;
    for (const auto& row : k_tracker_dispatch)
    {
        if (row.vendor_id == vendor_id)
            return true;
    }
    return false;
}

// True when a tracker's type has at least one vendored dispatch row (a row with a
// non-empty vendor id). Derived from the table, not a hardcoded type: adding a
// vendored row for a new tracker type makes that type vendor-selectable here with
// no other change. Each row's `matches` predicate is the type probe (a pure
// dynamic_cast, no side effects). Null is treated as unsupported.
bool tracker_supports_vendors(const ITracker* tracker)
{
    if (!tracker)
        return false;
    for (const auto& row : k_tracker_dispatch)
    {
        if (!row.vendor_id.empty() && row.matches(*tracker))
            return true;
    }
    return false;
}

// True when a dispatch row offers `vendor_id` for this tracker's own type, i.e. the
// selection is valid for this specific tracker rather than merely for some other
// vendored type. Scoping the id to the tracker's type is what makes a cross-type
// pairing (a valid id belonging to a different type) fail here, with a precise
// error, instead of later during impl creation. Null tracker or empty id are never
// accepted.
bool tracker_accepts_vendor(const ITracker* tracker, std::string_view vendor_id)
{
    if (!tracker || vendor_id.empty())
        return false;
    for (const auto& row : k_tracker_dispatch)
    {
        if (row.vendor_id == vendor_id && row.matches(*tracker))
            return true;
    }
    return false;
}

// Identify a tracker in an error message by its name, tolerating a null pointer
// (validation can run before the tracker list is known to hold no nulls).
std::string tracker_name_for_error(const ITracker* tracker)
{
    return tracker ? std::string(tracker->get_name()) : std::string("<null>");
}

} // namespace

// Defined in this translation unit because it consults the private vendor dispatch table
// (k_tracker_dispatch above); the anonymous-namespace helpers it calls stay visible here. See the
// header for the full contract.
void validate_vendor_selections(const std::vector<std::pair<const ITracker*, TrackerVendor>>& tracker_vendors)
{
    std::unordered_set<const ITracker*> seen;
    for (const auto& [tracker, vendor] : tracker_vendors)
    {
        // Only vendored tracker types accept a vendor selection; reject any other so a
        // misassigned (silently-ignored) selection surfaces as an error.
        if (!tracker_supports_vendors(tracker))
        {
            throw std::invalid_argument("LiveDeviceIOFactory: vendor selection '" + vendor.id +
                                        "' provided for tracker '" + tracker_name_for_error(tracker) +
                                        "', whose type does not support vendors");
        }
        // Reject unknown vendor ids up front rather than when the impl is built.
        if (!dispatch_has_vendor(vendor.id))
        {
            throw std::invalid_argument("LiveDeviceIOFactory: unknown vendor id '" + vendor.id + "' for tracker '" +
                                        tracker_name_for_error(tracker) + "'");
        }
        // The id names a real vendor, but reject it unless it belongs to this
        // tracker's own type so a cross-type pairing fails here instead of later.
        if (!tracker_accepts_vendor(tracker, vendor.id))
        {
            throw std::invalid_argument("LiveDeviceIOFactory: vendor id '" + vendor.id +
                                        "' is not available for tracker '" + tracker_name_for_error(tracker) + "'");
        }
        if (vendor.id == LiveFullBodyTrackerNoitomImpl::VENDOR_ID)
        {
            LiveFullBodyTrackerNoitomImpl::validate_vendor(vendor);
        }
        else if (vendor.id == LiveFullBodyTrackerXsensImpl::VENDOR_ID)
        {
            LiveFullBodyTrackerXsensImpl::validate_vendor(vendor);
        }
        else if (!vendor.params.empty())
        {
            throw std::invalid_argument("LiveDeviceIOFactory: vendor params are not supported yet for tracker '" +
                                        tracker_name_for_error(tracker) + "' (vendor id '" + vendor.id + "')");
        }

        if (!seen.insert(tracker).second)
        {
            throw std::invalid_argument("LiveDeviceIOFactory: duplicate vendor selection for tracker '" +
                                        tracker_name_for_error(tracker) + "' (vendor id '" + vendor.id + "')");
        }
    }
}

std::vector<std::string> LiveDeviceIOFactory::get_required_extensions(
    const std::vector<std::shared_ptr<ITracker>>& trackers,
    const std::vector<std::pair<const ITracker*, TrackerVendor>>& tracker_vendors)
{
    std::set<std::string> all;

    // Precondition: DeviceIOSession has validated tracker_vendors (see @pre); this only resolves.

    // DeviceIOSession always owns an XrTimeConverter; match session requirements even with zero trackers.
    for (const auto& ext : XrTimeConverter::get_required_extensions())
        all.insert(ext);

    for (const auto& tracker : trackers)
    {
        if (!tracker)
            throw std::invalid_argument("LiveDeviceIOFactory: null tracker in trackers list");

        const TrackerVendor* selected = find_tracker_vendor(tracker_vendors, tracker.get());

        bool matched = false;
        for (const auto& dispatch : k_tracker_dispatch)
        {
            if (!row_selected(dispatch, selected))
                continue;
            if (dispatch.collect_extensions(*tracker, all))
            {
                matched = true;
                break;
            }
        }

        if (!matched)
            throw_unresolved_tracker("LiveDeviceIOFactory::get_required_extensions", *tracker, selected);
    }

    return { all.begin(), all.end() };
}

LiveDeviceIOFactory::LiveDeviceIOFactory(const OpenXRSessionHandles& handles,
                                         mcap::McapWriter* writer,
                                         const std::vector<std::pair<const ITracker*, std::string>>& tracker_names,
                                         const std::vector<std::pair<const ITracker*, TrackerVendor>>& tracker_vendors)
    : handles_(handles), writer_(writer)
{
    // Precondition: DeviceIOSession has validated tracker_vendors (see the ctor @pre); assume valid.

    for (const auto& [tracker, name] : tracker_names)
    {
        TrackerData& data = tracker_data_[tracker];
        if (data.name.has_value())
        {
            throw std::invalid_argument("LiveDeviceIOFactory: duplicate tracker pointer for channel name '" + name +
                                        "' (already mapped as '" + *data.name + "')");
        }
        data.name = name;
    }

    for (const auto& [tracker, vendor] : tracker_vendors)
    {
        tracker_data_[tracker].vendor = vendor;
    }
}

std::unique_ptr<ITrackerImpl> LiveDeviceIOFactory::create_tracker_impl(const ITracker& tracker)
{
    const TrackerVendor* selected = find_vendor(&tracker);

    for (const auto& dispatch : k_tracker_dispatch)
    {
        if (!row_selected(dispatch, selected))
            continue;
        if (std::unique_ptr<ITrackerImpl> impl = dispatch.try_create(*this, tracker))
        {
            return impl;
        }
    }
    throw_unresolved_tracker("LiveDeviceIOFactory::create_tracker_impl", tracker, selected);
}

bool LiveDeviceIOFactory::should_record(const ITracker* tracker) const
{
    auto it = tracker_data_.find(tracker);
    return writer_ && it != tracker_data_.end() && it->second.name.has_value();
}

std::string_view LiveDeviceIOFactory::get_name(const ITracker* tracker) const
{
    auto it = tracker_data_.find(tracker);
    assert(it != tracker_data_.end() && it->second.name.has_value() &&
           "get_name called for tracker without a channel name (call should_record first)");
    return *it->second.name;
}

const TrackerVendor* LiveDeviceIOFactory::find_vendor(const ITracker* tracker) const
{
    auto it = tracker_data_.find(tracker);
    return (it != tracker_data_.end() && it->second.vendor) ? &*it->second.vendor : nullptr;
}

std::unique_ptr<IHeadTrackerImpl> LiveDeviceIOFactory::create_head_tracker_impl(const HeadTracker* tracker)
{
    std::unique_ptr<HeadMcapChannels> channels;
    if (should_record(tracker))
    {
        channels = LiveHeadTrackerImpl::create_mcap_channels(*writer_, get_name(tracker));
    }
    return std::make_unique<LiveHeadTrackerImpl>(handles_, std::move(channels));
}

std::unique_ptr<IHandTrackerImpl> LiveDeviceIOFactory::create_hand_tracker_impl(const HandTracker* tracker)
{
    std::unique_ptr<HandMcapChannels> channels;
    if (should_record(tracker))
    {
        channels = LiveHandTrackerImpl::create_mcap_channels(*writer_, get_name(tracker));
    }
    return std::make_unique<LiveHandTrackerImpl>(handles_, std::move(channels));
}

std::unique_ptr<IControllerTrackerImpl> LiveDeviceIOFactory::create_controller_tracker_impl(const ControllerTracker* tracker)
{
    std::unique_ptr<ControllerMcapChannels> channels;
    if (should_record(tracker))
    {
        channels = LiveControllerTrackerImpl::create_mcap_channels(*writer_, get_name(tracker));
    }
    return std::make_unique<LiveControllerTrackerImpl>(handles_, std::move(channels));
}

std::unique_ptr<IMessageChannelTrackerImpl> LiveDeviceIOFactory::create_message_channel_tracker_impl(
    const MessageChannelTracker* tracker)
{
    std::unique_ptr<MessageChannelMcapChannels> channels;
    if (should_record(tracker))
    {
        channels = LiveMessageChannelTrackerImpl::create_mcap_channels(*writer_, get_name(tracker));
    }
    return std::make_unique<LiveMessageChannelTrackerImpl>(handles_, tracker, std::move(channels));
}

std::unique_ptr<IFullBodyTrackerImpl> LiveDeviceIOFactory::create_full_body_tracker_pico_impl(const FullBodyTracker* tracker)
{
    std::unique_ptr<FullBodyMcapChannels> channels;
    if (should_record(tracker))
    {
        channels = LiveFullBodyTrackerPicoImpl::create_mcap_channels(*writer_, get_name(tracker));
    }
    return std::make_unique<LiveFullBodyTrackerPicoImpl>(handles_, std::move(channels));
}

std::unique_ptr<IFullBodyTrackerImpl> LiveDeviceIOFactory::create_full_body_tracker_noitom_impl(const FullBodyTracker* tracker)
{
    std::unique_ptr<FullBodyMcapChannels> channels;
    if (should_record(tracker))
    {
        channels = LiveFullBodyTrackerPicoImpl::create_mcap_channels(*writer_, get_name(tracker));
    }

    const TrackerVendor* vendor = find_vendor(tracker);
    assert(vendor && vendor->id == LiveFullBodyTrackerNoitomImpl::VENDOR_ID);
    return std::make_unique<LiveFullBodyTrackerNoitomImpl>(handles_, *vendor, std::move(channels));
}

std::unique_ptr<IFullBodyTrackerImpl> LiveDeviceIOFactory::create_full_body_tracker_xsens_impl(const FullBodyTracker* tracker)
{
    std::unique_ptr<FullBodyMcapChannels> channels;
    if (should_record(tracker))
    {
        channels = LiveFullBodyTrackerPicoImpl::create_mcap_channels(*writer_, get_name(tracker));
    }

    const TrackerVendor* vendor = find_vendor(tracker);
    assert(vendor && vendor->id == LiveFullBodyTrackerXsensImpl::VENDOR_ID);
    return std::make_unique<LiveFullBodyTrackerXsensImpl>(handles_, *vendor, std::move(channels));
}

std::unique_ptr<ITensorPushTrackerImpl> LiveDeviceIOFactory::create_tensor_push_tracker_impl(const TensorPushTracker* tracker)
{
    return std::make_unique<LiveTensorPushTrackerImpl>(handles_, tracker);
}

std::unique_ptr<IHapticCommandReaderTrackerImpl> LiveDeviceIOFactory::create_haptic_command_reader_tracker_impl(
    const HapticCommandReaderTracker* tracker)
{
    return std::make_unique<LiveHapticCommandReaderTrackerImpl>(handles_, tracker);
}

std::unique_ptr<IKeyboardTrackerImpl> LiveDeviceIOFactory::create_keyboard_tracker_impl(const KeyboardTracker* tracker)
{
    std::unique_ptr<KeyboardMcapChannels> channels;
    if (should_record(tracker))
    {
        channels = LiveKeyboardTrackerImpl::create_mcap_channels(*writer_, get_name(tracker));
    }
    return std::make_unique<LiveKeyboardTrackerImpl>(tracker->input_state(), std::move(channels));
}

#include "generated_live_factory_methods.inc"

} // namespace core
