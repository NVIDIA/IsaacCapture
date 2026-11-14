// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/camera_core/publisher.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>

namespace core::camera
{

namespace
{

std::int64_t steady_now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

DeliveryOrder resolve_order(const ReaderOptions& options)
{
    if (options.order != DeliveryOrder::DEFAULT)
    {
        return options.order;
    }
    return options.window == 1 ? DeliveryOrder::NEWEST : DeliveryOrder::IN_ORDER;
}

// The bytes one row of a plane holds and the number of its rows.
struct PlaneSize
{
    std::uint64_t row_bytes = 0;
    std::uint64_t rows = 0;
};

PlaneSize plane_size_of(const StreamDescription& stream, std::uint32_t plane)
{
    const std::uint64_t width = stream.width;
    const std::uint64_t height = stream.height;
    switch (stream.pixel_format)
    {
    case PixelFormat::BGRA8:
    case PixelFormat::RGBA8:
        return { width * 4, height };
    case PixelFormat::GRAY8:
        return { width, height };
    case PixelFormat::NV12:
        // Luma at full size, then one interleaved chroma pair per two by two pixels.
        return plane == 0 ? PlaneSize{ width, height } : PlaneSize{ (width + 1) / 2 * 2, (height + 1) / 2 };
    }
    return {};
}

// A pixel stream names every plane of its format and every plane lies inside the slot; an
// encoded stream has no planes.
bool layout_is_valid(const StreamDescription& stream)
{
    if (stream.is_encoded())
    {
        return stream.plane_count == 0;
    }
    if (stream.width == 0 || stream.height == 0 || stream.plane_count != plane_count_of(stream.pixel_format))
    {
        return false;
    }
    for (std::uint32_t i = 0; i < stream.plane_count; ++i)
    {
        const PlaneSize size = plane_size_of(stream, i);
        const std::uint64_t stride = stream.planes[i].stride_bytes;
        // 64-bit sums of 32-bit fields, so nothing here can wrap.
        const std::uint64_t end = stream.planes[i].offset_bytes + stride * (size.rows - 1) + size.row_bytes;
        if (size.rows == 0 || stride < size.row_bytes || end > stream.slot_bytes)
        {
            return false;
        }
    }
    return true;
}

} // namespace

Publisher::Publisher(Clock clock) : m_clock(clock ? std::move(clock) : Clock(steady_now_ns))
{
}

GroupId Publisher::add_group(const GroupDescription& description, std::uint32_t backend_reserve)
{
    if (description.id.empty())
    {
        throw std::invalid_argument("camera group needs an id");
    }
    if (description.streams.empty() || description.streams.size() > MAX_STREAMS_PER_GROUP)
    {
        throw std::invalid_argument("camera group '" + description.id + "' needs between 1 and " +
                                    std::to_string(MAX_STREAMS_PER_GROUP) + " streams");
    }
    if (backend_reserve == 0)
    {
        throw std::invalid_argument("camera group '" + description.id + "' needs a backend reserve of at least 1");
    }

    auto group = std::make_unique<Group>();
    group->description = description;
    group->reserve = backend_reserve;
    group->slot_count = std::numeric_limits<std::uint32_t>::max();
    for (const StreamDescription& stream : description.streams)
    {
        if (stream.id.empty() || stream.slot_bytes == 0 || stream.slot_count == 0 ||
            stream.slot_count > MAX_SLOTS_PER_STREAM)
        {
            throw std::invalid_argument("camera group '" + description.id + "' has a stream without an id, a slot " +
                                        "size or between 1 and " + std::to_string(MAX_SLOTS_PER_STREAM) + " slots");
        }
        if (!layout_is_valid(stream))
        {
            throw std::invalid_argument("camera stream '" + stream.id + "' needs a size and every plane of its " +
                                        "pixel format inside the slot, or no planes at all when it is encoded");
        }
        group->slot_count = std::min(group->slot_count, stream.slot_count);
        group->slots.emplace_back(stream.slot_count);
    }
    // Before any reader, the ring keeps one frame set and the backend keeps its reserve.
    if (group->slot_count <= backend_reserve)
    {
        throw std::invalid_argument("camera group '" + description.id + "' has pools of " +
                                    std::to_string(group->slot_count) + " slots, too few for one frame set and a " +
                                    "backend reserve of " + std::to_string(backend_reserve));
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    for (const auto& existing : m_groups)
    {
        if (existing->description.id == description.id)
        {
            throw std::invalid_argument("camera group '" + description.id + "' already exists");
        }
    }
    m_groups.push_back(std::move(group));
    return static_cast<GroupId>(m_groups.size() - 1);
}

std::optional<std::uint32_t> Publisher::acquire_slot(GroupId group, std::size_t stream)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    Group& target = group_at(group);
    if (target.on_free)
    {
        throw std::logic_error("camera group '" + target.description.id +
                               "' has a slot-free callback, so its backend names slots with claim_slot");
    }
    std::vector<Slot>& slots = target.slots.at(stream);
    for (std::uint32_t i = 0; i < slots.size(); ++i)
    {
        if (slots[i].holds == 0 && !slots[i].writing)
        {
            slots[i].writing = true;
            return i;
        }
    }
    ++target.stats.no_free_slot;
    return std::nullopt;
}

void Publisher::claim_slot(GroupId group, std::size_t stream, std::uint32_t slot)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    Slot& state = slot_at(group_at(group), stream, slot);
    if (state.holds != 0 || state.writing)
    {
        throw std::invalid_argument("camera slot " + std::to_string(slot) + " is still in use and cannot be claimed");
    }
    state.writing = true;
}

void Publisher::abandon_slot(GroupId group, std::size_t stream, std::uint32_t slot)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    Slot& state = slot_at(group_at(group), stream, slot);
    if (!state.writing)
    {
        throw std::invalid_argument("camera slot " + std::to_string(slot) + " was not acquired and cannot be abandoned");
    }
    state.writing = false;
}

void Publisher::set_slot_free_callback(GroupId group, SlotFreeCallback callback)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    group_at(group).on_free = callback ? std::make_shared<const SlotFreeCallback>(std::move(callback)) : nullptr;
}

std::uint64_t Publisher::publish(GroupId group, std::span<const ImageInfo> images, const FrameTimeline& timeline)
{
    std::vector<FreedSlot> freed;
    std::uint64_t sequence = 0;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        Group& target = group_at(group);
        if (images.size() != target.description.streams.size())
        {
            throw std::invalid_argument("camera group '" + target.description.id + "' publishes " +
                                        std::to_string(target.description.streams.size()) + " images per frame set");
        }
        for (std::size_t i = 0; i < images.size(); ++i)
        {
            if (!slot_at(target, i, images[i].slot).writing)
            {
                throw std::invalid_argument("camera slot " + std::to_string(images[i].slot) +
                                            " was not acquired before it was published");
            }
            if (images[i].bytes_used > target.description.streams[i].slot_bytes)
            {
                throw std::invalid_argument("camera stream '" + target.description.streams[i].id + "' published " +
                                            std::to_string(images[i].bytes_used) + " bytes into a slot of " +
                                            std::to_string(target.description.streams[i].slot_bytes));
            }
        }

        FrameSet set;
        set.sequence = ++target.sequence;
        set.generation = target.generation;
        set.timeline = timeline;
        set.timeline.published_ns = m_clock();
        for (std::size_t i = 0; i < images.size(); ++i)
        {
            set.images[i] = images[i];
            slot_at(target, i, images[i].slot).writing = false; // the ring holds it from here
        }
        hold(target, set);
        target.ring.push_back(set);
        ++target.stats.published;
        sequence = set.sequence;

        trim_ring(target, freed);
        for (SubscriptionId id : target.subscriptions)
        {
            deliver(target, id, m_subscriptions.at(id), freed);
        }
    }
    notify(freed);
    return sequence;
}

void Publisher::restart_group(GroupId group)
{
    std::vector<FreedSlot> freed;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        Group& target = group_at(group);
        ++target.generation;
        while (!target.ring.empty())
        {
            unhold(target, target.ring.front(), freed);
            target.ring.pop_front();
        }
    }
    notify(freed);
}

SubscribeResult Publisher::subscribe(EventSink& sink, GroupId group, const ReaderOptions& options)
{
    SubscribeResult result;
    std::lock_guard<std::mutex> lock(m_mutex);
    if (group >= m_groups.size())
    {
        result.error = "unknown camera group";
        return result;
    }
    if (options.window == 0)
    {
        result.error = "a reader's window must be at least 1";
        return result;
    }
    Group& target = *m_groups[group];
    const DeliveryOrder order = resolve_order(options);

    // The ring keeps what in-order readers may still be waiting for; newest-only readers
    // need the newest frame set and nothing older.
    std::size_t history = order == DeliveryOrder::IN_ORDER ? options.window : 1;
    std::size_t windows = options.window;
    for (SubscriptionId id : target.subscriptions)
    {
        const Subscription& other = m_subscriptions.at(id);
        if (other.order == DeliveryOrder::IN_ORDER)
        {
            history = std::max<std::size_t>(history, other.window);
        }
        windows += other.window;
    }
    const std::size_t needed = history + windows + target.reserve;
    if (needed > target.slot_count)
    {
        result.error = "subscription does not fit the pool of camera group '" + target.description.id + "': it needs " +
                       std::to_string(needed) + " slots (" + std::to_string(history) + " ring history, " +
                       std::to_string(windows) + " for reader windows, " + std::to_string(target.reserve) +
                       " for the backend) and the pool has " + std::to_string(target.slot_count);
        return result;
    }

    Subscription subscription;
    subscription.group = group;
    subscription.window = options.window;
    subscription.order = order;
    subscription.sink = &sink;
    result.id = m_next_subscription++;
    m_subscriptions.emplace(result.id, std::move(subscription));
    target.subscriptions.push_back(result.id);
    return result;
}

void Publisher::release(SubscriptionId subscription, std::uint64_t lease_id)
{
    std::vector<FreedSlot> freed;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto found = m_subscriptions.find(subscription);
        if (found == m_subscriptions.end())
        {
            return;
        }
        const auto lease = found->second.leases.find(lease_id);
        if (lease == found->second.leases.end())
        {
            return;
        }
        Group& target = *m_groups[found->second.group];
        unhold(target, lease->second, freed);
        found->second.leases.erase(lease);
        deliver(target, subscription, found->second, freed);
    }
    notify(freed);
}

void Publisher::unsubscribe(SubscriptionId subscription)
{
    std::vector<FreedSlot> freed;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto found = m_subscriptions.find(subscription);
        if (found == m_subscriptions.end())
        {
            return;
        }
        Group& target = *m_groups[found->second.group];
        for (const auto& lease : found->second.leases)
        {
            unhold(target, lease.second, freed);
        }
        std::erase(target.subscriptions, subscription);
        m_subscriptions.erase(found);
        trim_ring(target, freed);
    }
    notify(freed);
}

std::size_t Publisher::group_count() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_groups.size();
}

GroupDescription Publisher::group_description(GroupId group) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return group_at(group).description;
}

std::optional<GroupId> Publisher::find_group(std::string_view id) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    for (std::size_t i = 0; i < m_groups.size(); ++i)
    {
        if (m_groups[i]->description.id == id)
        {
            return static_cast<GroupId>(i);
        }
    }
    return std::nullopt;
}

std::uint32_t Publisher::generation(GroupId group) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return group_at(group).generation;
}

PublisherStats Publisher::stats(GroupId group) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return group_at(group).stats;
}

Publisher::Group& Publisher::group_at(GroupId group)
{
    if (group >= m_groups.size())
    {
        throw std::out_of_range("unknown camera group");
    }
    return *m_groups[group];
}

const Publisher::Group& Publisher::group_at(GroupId group) const
{
    if (group >= m_groups.size())
    {
        throw std::out_of_range("unknown camera group");
    }
    return *m_groups[group];
}

Publisher::Slot& Publisher::slot_at(Group& group, std::size_t stream, std::uint32_t slot)
{
    if (stream >= group.slots.size() || slot >= group.slots[stream].size())
    {
        throw std::invalid_argument("camera group '" + group.description.id + "' has no stream " +
                                    std::to_string(stream) + " slot " + std::to_string(slot));
    }
    return group.slots[stream][slot];
}

std::size_t Publisher::history_of(const Group& group) const
{
    std::size_t history = 1; // the newest frame set is always kept
    for (SubscriptionId id : group.subscriptions)
    {
        const Subscription& subscription = m_subscriptions.at(id);
        if (subscription.order == DeliveryOrder::IN_ORDER)
        {
            history = std::max<std::size_t>(history, subscription.window);
        }
    }
    return history;
}

void Publisher::hold(Group& group, const FrameSet& set)
{
    for (std::size_t i = 0; i < group.slots.size(); ++i)
    {
        ++slot_at(group, i, set.images[i].slot).holds;
    }
}

void Publisher::unhold(Group& group, const FrameSet& set, std::vector<FreedSlot>& freed)
{
    for (std::size_t i = 0; i < group.slots.size(); ++i)
    {
        Slot& slot = slot_at(group, i, set.images[i].slot);
        if (--slot.holds == 0 && group.on_free)
        {
            freed.push_back({ group.on_free, i, set.images[i].slot });
        }
    }
}

void Publisher::trim_ring(Group& group, std::vector<FreedSlot>& freed)
{
    const std::size_t history = history_of(group);
    while (group.ring.size() > history)
    {
        unhold(group, group.ring.front(), freed);
        group.ring.pop_front();
    }
}

void Publisher::deliver(Group& group, SubscriptionId id, Subscription& subscription, std::vector<FreedSlot>& freed)
{
    while (subscription.leases.size() < subscription.window)
    {
        if (group.ring.empty() || group.ring.back().sequence <= subscription.cursor)
        {
            return;
        }
        const FrameSet* target = &group.ring.back();
        if (subscription.order == DeliveryOrder::IN_ORDER)
        {
            target = &*std::find_if(group.ring.begin(), group.ring.end(),
                                    [&](const FrameSet& set) { return set.sequence > subscription.cursor; });
        }

        FrameEvent event;
        event.lease_id = m_next_lease++;
        event.sequence = target->sequence;
        event.generation = target->generation;
        if (subscription.started)
        {
            const std::uint64_t skipped = target->sequence - subscription.cursor - 1;
            event.skipped =
                static_cast<std::uint32_t>(std::min<std::uint64_t>(skipped, std::numeric_limits<std::uint32_t>::max()));
        }
        event.image_count = static_cast<std::uint32_t>(group.slots.size());
        event.images = target->images;
        event.timeline = target->timeline;
        event.timeline.sent_ns = m_clock();

        hold(group, *target);
        if (!subscription.sink->send_event(id, event))
        {
            unhold(group, *target, freed);
            ++group.stats.send_failures;
            return;
        }
        ++group.stats.events_sent;
        subscription.leases.emplace(event.lease_id, *target);
        subscription.cursor = target->sequence;
        subscription.started = true;
    }
}

void Publisher::notify(const std::vector<FreedSlot>& freed)
{
    for (const FreedSlot& slot : freed)
    {
        (*slot.callback)(slot.stream, slot.slot);
    }
}

} // namespace core::camera
