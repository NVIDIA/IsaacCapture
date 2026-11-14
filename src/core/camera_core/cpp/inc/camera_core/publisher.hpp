// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace core::camera
{

using GroupId = std::uint32_t;
using SubscriptionId = std::uint32_t; // 0 is never a valid subscription

/**
 * @brief Where a reader's frame events go: a socket connection or an in-process queue.
 */
class EventSink
{
public:
    virtual ~EventSink() = default;

    /**
     * @brief Hand one event to the reader. Called on the capture path with the group's lock
     * held, so it must not block, throw or call back into the publisher. Calls for one
     * subscription never overlap.
     * @return false when the event could not be handed over; the reader then gets a later one.
     */
    virtual bool send_event(SubscriptionId subscription, const FrameEvent& event) noexcept = 0;
};

// Counters of one group since it was added.
struct PublisherStats
{
    std::uint64_t published = 0; // frame sets published
    std::uint64_t events_sent = 0;
    std::uint64_t send_failures = 0;
    std::uint64_t no_free_slot = 0; // acquire_slot calls that found every slot in use
};

struct SubscribeResult
{
    SubscriptionId id = 0;
    std::string error; // why the subscription was refused, when id is 0

    explicit operator bool() const
    {
        return id != 0;
    }
};

/**
 * @brief Tracks which pool slots hold which frame sets and who may still read them.
 *
 * A group keeps a short ring of recent frame sets; a reader's window is how many are kept for
 * it at a time, unread and held together. A subscription is accepted only when
 * slots >= ring history + sum of reader windows + backend reserve. Events carry slot indices;
 * slot memory is never touched here. Thread-safe; no call blocks on a reader, and each group
 * has its own lock, so one camera never waits for another camera's readers.
 */
class Publisher
{
public:
    using Clock = std::function<std::int64_t()>;
    using SlotFreeCallback = std::function<void(std::size_t stream, std::uint32_t slot)>;

    /**
     * @param clock Host monotonic time in nanoseconds; defaults to std::chrono::steady_clock.
     */
    explicit Publisher(Clock clock = {});

    // ---- backend side ----------------------------------------------------------------------

    /**
     * @brief Declare a group and the size of its pools.
     * @param backend_reserve Slots the backend needs beyond the ring and the readers: one to
     * write ahead, more for an SDK that keeps several captures in flight in its own queue.
     * @throws std::invalid_argument if the description is not usable.
     */
    GroupId add_group(const GroupDescription& description, std::uint32_t backend_reserve = 1);

    /**
     * @brief Take a free slot to write the next image into.
     * @return The slot, or nothing when every slot is in use (the frame should be dropped).
     * @throws std::logic_error for a group that has a slot-free callback: its backend owns the
     * pool and names slots with claim_slot.
     */
    std::optional<std::uint32_t> acquire_slot(GroupId group, std::size_t stream);

    /**
     * @brief Mark a slot the SDK chose and filled, for a pool the SDK owns.
     * @throws std::invalid_argument if the slot is still held by the ring or a reader.
     */
    void claim_slot(GroupId group, std::size_t stream, std::uint32_t slot);

    /**
     * @brief Give back a slot taken with acquire_slot or claim_slot without publishing it.
     */
    void abandon_slot(GroupId group, std::size_t stream, std::uint32_t slot);

    /**
     * @brief Be told when a published slot is no longer held by the ring or any reader, so a
     * buffer the SDK owns can go back to it at once. Called outside the group's lock, on
     * the thread whose call released the last hold. A group with this callback hands out its
     * slots with claim_slot only, so a slot cannot be taken again before its callback has run.
     */
    void set_slot_free_callback(GroupId group, SlotFreeCallback callback);

    /**
     * @brief Publish one frame set. images[i] belongs to stream i and names a slot taken with
     * acquire_slot or claim_slot.
     * @param timeline capture_ns and sdk_ready_ns from the backend; the rest is filled in here.
     * @return The frame set's sequence number.
     * @throws std::invalid_argument if the images do not match the group or do not fit its slots.
     */
    std::uint64_t publish(GroupId group, std::span<const ImageInfo> images, const FrameTimeline& timeline);

    /**
     * @brief Tell readers the camera restarted: the generation changes and the ring is emptied.
     * Frame sets readers already hold stay valid until released.
     */
    void restart_group(GroupId group);

    // ---- reader side -----------------------------------------------------------------------

    /**
     * @brief Start sending a group's frame events to a sink, which must outlive the subscription.
     * @return The subscription, or the reason it does not fit the pool.
     */
    SubscribeResult subscribe(EventSink& sink, GroupId group, const ReaderOptions& options);

    /**
     * @brief Return a frame set. The reader's next one, if any is waiting, is sent at once.
     */
    void release(SubscriptionId subscription, std::uint64_t lease_id);

    /**
     * @brief End a subscription and take back every frame set it still holds.
     */
    void unsubscribe(SubscriptionId subscription);

    // ---- description -----------------------------------------------------------------------

    std::size_t group_count() const;
    GroupDescription group_description(GroupId group) const;
    std::optional<GroupId> find_group(std::string_view id) const;
    std::uint32_t generation(GroupId group) const;
    PublisherStats stats(GroupId group) const;

private:
    struct FrameSet
    {
        std::uint64_t sequence = 0;
        std::uint32_t generation = 0;
        FrameTimeline timeline;
        std::array<ImageInfo, MAX_STREAMS_PER_GROUP> images;
    };

    struct Subscription
    {
        std::uint32_t window = 1;
        DeliveryOrder order = DeliveryOrder::NEWEST;
        EventSink* sink = nullptr;
        std::uint64_t cursor = 0; // sequence of the last frame set sent to this reader
        bool started = false;
        std::map<std::uint64_t, FrameSet> leases; // sent and not yet released
    };

    struct Slot
    {
        std::uint32_t holds = 0; // the ring and every reader lease that names this slot
        bool writing = false; // taken by the backend and not yet published
    };

    // One camera. Everything that changes after add_group is guarded by `mutex`; one group's
    // lock is never held while another's is taken.
    struct Group
    {
        GroupDescription description; // fixed once the group is added
        std::uint32_t reserve = 1;
        std::uint32_t slot_count = 0; // the smallest pool of the group

        mutable std::mutex mutex;
        std::uint32_t generation = 0;
        std::uint64_t sequence = 0;
        std::uint64_t next_lease = 1;
        std::vector<std::vector<Slot>> slots; // per stream
        std::deque<FrameSet> ring;
        std::map<SubscriptionId, Subscription> subscriptions; // in the order they were made
        std::shared_ptr<const SlotFreeCallback> on_free;
        PublisherStats stats;
    };

    struct FreedSlot
    {
        std::shared_ptr<const SlotFreeCallback> callback;
        std::size_t stream = 0;
        std::uint32_t slot = 0;
    };

    Group& group_at(GroupId group) const;
    Group* group_of(SubscriptionId subscription) const;
    static Slot& slot_at(Group& group, std::size_t stream, std::uint32_t slot);
    static std::size_t history_of(const Group& group);
    static void hold(Group& group, const FrameSet& set);
    static void unhold(Group& group, const FrameSet& set, std::vector<FreedSlot>& freed);
    static void trim_ring(Group& group, std::vector<FreedSlot>& freed);
    void deliver(Group& group, SubscriptionId id, Subscription& subscription, std::vector<FreedSlot>& freed) const;
    static void notify(const std::vector<FreedSlot>& freed);

    Clock m_clock;

    // Which groups exist and which group a subscription belongs to: read on every call, changed
    // only when a group is added or a reader comes or goes. Held only to look a group up, never
    // together with a group's lock.
    mutable std::shared_mutex m_registry;
    std::vector<std::unique_ptr<Group>> m_groups; // never removed, so a group's address is stable
    std::map<SubscriptionId, GroupId> m_subscription_groups;
    SubscriptionId m_next_subscription = 1;
};

} // namespace core::camera
