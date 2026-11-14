// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/camera_core/publisher.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <random>
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

// Never 0, which stands for "not chosen yet".
std::uint64_t new_session_id()
{
    std::random_device source;
    for (;;)
    {
        const std::uint64_t id = (std::uint64_t{ source() } << 32) | source();
        if (id != 0)
        {
            return id;
        }
    }
}

} // namespace

Publisher::Publisher(Clock clock) : m_clock(clock ? std::move(clock) : Clock(steady_now_ns))
{
}

GroupId Publisher::add_group(const GroupDescription& description, std::uint32_t backend_reserve)
{
    if (const std::string problem = problem_with(description); !problem.empty())
    {
        throw std::invalid_argument(problem);
    }
    if (backend_reserve == 0)
    {
        throw std::invalid_argument("camera group '" + description.id + "' needs a backend reserve of at least 1");
    }

    auto group = std::make_unique<Group>();
    group->description = description;
    if (group->description.session_id == 0)
    {
        group->description.session_id = new_session_id();
    }
    group->reserve = backend_reserve;
    group->slot_count = std::numeric_limits<std::uint32_t>::max();
    for (const StreamDescription& stream : description.streams)
    {
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

    const std::unique_lock<std::shared_mutex> registry(m_registry);
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
    Group& target = group_at(group);
    const std::lock_guard<std::mutex> lock(target.mutex);
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
    Group& target = group_at(group);
    const std::lock_guard<std::mutex> lock(target.mutex);
    Slot& state = slot_at(target, stream, slot);
    if (state.holds != 0 || state.writing)
    {
        throw std::invalid_argument("camera slot " + std::to_string(slot) + " is still in use and cannot be claimed");
    }
    state.writing = true;
}

void Publisher::abandon_slot(GroupId group, std::size_t stream, std::uint32_t slot)
{
    Group& target = group_at(group);
    const std::lock_guard<std::mutex> lock(target.mutex);
    Slot& state = slot_at(target, stream, slot);
    if (!state.writing)
    {
        throw std::invalid_argument("camera slot " + std::to_string(slot) + " was not acquired and cannot be abandoned");
    }
    state.writing = false;
}

void Publisher::set_slot_free_callback(GroupId group, SlotFreeCallback callback)
{
    Group& target = group_at(group);
    const std::lock_guard<std::mutex> lock(target.mutex);
    target.on_free = callback ? std::make_shared<const SlotFreeCallback>(std::move(callback)) : nullptr;
}

std::uint64_t Publisher::publish(GroupId group, std::span<const ImageInfo> images, const FrameTimeline& timeline)
{
    std::vector<FreedSlot> freed;
    std::uint64_t sequence = 0;
    Group& target = group_at(group);
    {
        const std::lock_guard<std::mutex> lock(target.mutex);
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
        for (auto& [id, subscription] : target.subscriptions)
        {
            deliver(target, id, subscription, freed);
        }
    }
    notify(freed);
    return sequence;
}

void Publisher::restart_group(GroupId group)
{
    std::vector<FreedSlot> freed;
    Group& target = group_at(group);
    {
        const std::lock_guard<std::mutex> lock(target.mutex);
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
    if (options.window == 0)
    {
        result.error = "a reader's window must be at least 1";
        return result;
    }
    Group* target = nullptr;
    SubscriptionId id = 0;
    {
        // The subscription is known to the registry before its group can send it an event, so
        // a release that follows the very first event finds it.
        const std::unique_lock<std::shared_mutex> registry(m_registry);
        if (group >= m_groups.size())
        {
            result.error = "unknown camera group";
            return result;
        }
        target = m_groups[group].get();
        id = m_next_subscription++;
        m_subscription_groups.emplace(id, group);
    }

    {
        const std::lock_guard<std::mutex> lock(target->mutex);
        const DeliveryOrder order = resolved_order(options);

        // The ring keeps what in-order readers may still be waiting for; newest-only readers
        // need the newest frame set and nothing older.
        std::size_t history = order == DeliveryOrder::IN_ORDER ? options.window : 1;
        std::size_t windows = options.window;
        for (const auto& [other_id, other] : target->subscriptions)
        {
            if (other.order == DeliveryOrder::IN_ORDER)
            {
                history = std::max<std::size_t>(history, other.window);
            }
            windows += other.window;
        }
        const std::size_t needed = history + windows + target->reserve;
        if (needed <= target->slot_count)
        {
            Subscription subscription;
            subscription.window = options.window;
            subscription.order = order;
            subscription.sink = &sink;
            target->subscriptions.emplace(id, std::move(subscription));
            result.id = id;
            return result;
        }
        result.error = "subscription does not fit the pool of camera group '" + target->description.id + "': it needs " +
                       std::to_string(needed) + " slots (" + std::to_string(history) + " ring history, " +
                       std::to_string(windows) + " for reader windows, " + std::to_string(target->reserve) +
                       " for the backend) and the pool has " + std::to_string(target->slot_count);
    }
    const std::unique_lock<std::shared_mutex> registry(m_registry);
    m_subscription_groups.erase(id);
    return result;
}

void Publisher::activate(SubscriptionId subscription)
{
    Group* target = group_of(subscription);
    if (target == nullptr)
    {
        return;
    }
    const std::lock_guard<std::mutex> lock(target->mutex);
    const auto found = target->subscriptions.find(subscription);
    if (found != target->subscriptions.end())
    {
        found->second.active = true;
    }
}

bool Publisher::request_newest(SubscriptionId subscription, std::uint64_t after_sequence)
{
    Group* target = group_of(subscription);
    if (target == nullptr)
    {
        return false;
    }
    std::vector<FreedSlot> freed;
    {
        const std::lock_guard<std::mutex> lock(target->mutex);
        const auto found = target->subscriptions.find(subscription);
        if (found == target->subscriptions.end() || !found->second.active || found->second.order != DeliveryOrder::NEWEST)
        {
            return false;
        }
        found->second.wanted = true;
        found->second.cursor = after_sequence;
        found->second.started = after_sequence != 0;
        deliver(*target, subscription, found->second, freed);
    }
    notify(freed);
    return true;
}

void Publisher::release(SubscriptionId subscription, std::uint64_t lease_id)
{
    Group* target = group_of(subscription);
    if (target == nullptr)
    {
        return;
    }
    std::vector<FreedSlot> freed;
    {
        const std::lock_guard<std::mutex> lock(target->mutex);
        const auto found = target->subscriptions.find(subscription);
        if (found == target->subscriptions.end())
        {
            return;
        }
        const auto lease = found->second.leases.find(lease_id);
        if (lease == found->second.leases.end())
        {
            return;
        }
        unhold(*target, lease->second, freed);
        found->second.leases.erase(lease);
        deliver(*target, subscription, found->second, freed);
    }
    notify(freed);
}

void Publisher::unsubscribe(SubscriptionId subscription)
{
    Group* target = group_of(subscription);
    if (target == nullptr)
    {
        return;
    }
    std::vector<FreedSlot> freed;
    {
        // Once this lock is let go, no send_event runs or will start for the subscription.
        const std::lock_guard<std::mutex> lock(target->mutex);
        const auto found = target->subscriptions.find(subscription);
        if (found != target->subscriptions.end())
        {
            for (const auto& lease : found->second.leases)
            {
                unhold(*target, lease.second, freed);
            }
            target->subscriptions.erase(found);
            trim_ring(*target, freed);
        }
    }
    {
        const std::unique_lock<std::shared_mutex> registry(m_registry);
        m_subscription_groups.erase(subscription);
    }
    notify(freed);
}

std::size_t Publisher::group_count() const
{
    const std::shared_lock<std::shared_mutex> registry(m_registry);
    return m_groups.size();
}

GroupDescription Publisher::group_description(GroupId group) const
{
    return group_at(group).description;
}

std::optional<GroupId> Publisher::find_group(std::string_view id) const
{
    const std::shared_lock<std::shared_mutex> registry(m_registry);
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
    const Group& target = group_at(group);
    const std::lock_guard<std::mutex> lock(target.mutex);
    return target.generation;
}

PublisherStats Publisher::stats(GroupId group) const
{
    const Group& target = group_at(group);
    const std::lock_guard<std::mutex> lock(target.mutex);
    return target.stats;
}

Publisher::Group& Publisher::group_at(GroupId group) const
{
    const std::shared_lock<std::shared_mutex> registry(m_registry);
    if (group >= m_groups.size())
    {
        throw std::out_of_range("unknown camera group");
    }
    return *m_groups[group];
}

Publisher::Group* Publisher::group_of(SubscriptionId subscription) const
{
    const std::shared_lock<std::shared_mutex> registry(m_registry);
    const auto found = m_subscription_groups.find(subscription);
    return found == m_subscription_groups.end() ? nullptr : m_groups[found->second].get();
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

std::size_t Publisher::history_of(const Group& group)
{
    std::size_t history = 1; // the newest frame set is always kept
    for (const auto& [id, subscription] : group.subscriptions)
    {
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

void Publisher::deliver(Group& group, SubscriptionId id, Subscription& subscription, std::vector<FreedSlot>& freed) const
{
    if (!subscription.active)
    {
        return;
    }
    while (subscription.leases.size() < subscription.window)
    {
        // A newest-order reader is held a frame set only while it asks for one: whatever was
        // held ahead of time would be old by the time the reader came for it.
        if (subscription.order == DeliveryOrder::NEWEST && !subscription.wanted)
        {
            return;
        }
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
        event.lease_id = group.next_lease++;
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
        subscription.wanted = false;
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
