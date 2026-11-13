// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <camera_core/publisher.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace core::camera;

namespace
{

// A reader that keeps every event it is sent.
struct RecordingSink : EventSink
{
    bool send_event(SubscriptionId, const FrameEvent& event) override
    {
        if (!accept)
        {
            return false;
        }
        events.push_back(event);
        return true;
    }

    std::vector<FrameEvent> events;
    bool accept = true;
};

GroupDescription make_group(std::uint32_t slots, std::size_t streams = 2, const std::string& id = "head")
{
    GroupDescription group;
    group.id = id;
    for (std::size_t i = 0; i < streams; ++i)
    {
        StreamDescription stream;
        stream.id = id + "/" + std::to_string(i);
        stream.width = 640;
        stream.height = 400;
        stream.stride_bytes = 640 * 4;
        stream.slot_bytes = 640 * 400 * 4;
        stream.slot_count = slots;
        stream.frame_rate = 60.0;
        group.streams.push_back(stream);
    }
    return group;
}

// Publishes one frame set the way a backend does: take a free slot per stream, then publish.
std::uint64_t publish_next(Publisher& publisher, GroupId group, const FrameTimeline& timeline = {})
{
    const std::size_t streams = publisher.group_description(group).streams.size();
    std::vector<ImageInfo> images(streams);
    for (std::size_t i = 0; i < streams; ++i)
    {
        const auto slot = publisher.acquire_slot(group, i);
        REQUIRE(slot.has_value());
        images[i].slot = *slot;
    }
    return publisher.publish(group, images, timeline);
}

std::vector<std::uint64_t> sequences(const RecordingSink& sink)
{
    std::vector<std::uint64_t> result;
    for (const FrameEvent& event : sink.events)
    {
        result.push_back(event.sequence);
    }
    return result;
}

} // namespace

TEST_CASE("a window of one always gets the newest frame set", "[camera_core][delivery]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(8));
    RecordingSink sink;
    const SubscribeResult reader = publisher.subscribe(sink, group, { .window = 1 });
    REQUIRE(reader);

    publish_next(publisher, group);
    REQUIRE(sequences(sink) == std::vector<std::uint64_t>{ 1 });
    REQUIRE(sink.events[0].skipped == 0);
    REQUIRE(sink.events[0].image_count == 2);

    // The reader still holds frame set 1, so the next ones wait.
    publish_next(publisher, group);
    publish_next(publisher, group);
    REQUIRE(sink.events.size() == 1);

    publisher.release(reader.id, sink.events[0].lease_id);
    REQUIRE(sequences(sink) == std::vector<std::uint64_t>{ 1, 3 });
    REQUIRE(sink.events[1].skipped == 1);
}

TEST_CASE("a larger window gets every frame set in order", "[camera_core][delivery]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(8));
    RecordingSink sink;
    const SubscribeResult reader = publisher.subscribe(sink, group, { .window = 3 });
    REQUIRE(reader);

    for (int i = 0; i < 4; ++i)
    {
        publish_next(publisher, group);
    }
    // Three are out, the fourth waits in the ring until the reader has room.
    REQUIRE(sequences(sink) == std::vector<std::uint64_t>{ 1, 2, 3 });

    publisher.release(reader.id, sink.events[0].lease_id);
    REQUIRE(sequences(sink) == std::vector<std::uint64_t>{ 1, 2, 3, 4 });
    REQUIRE(sink.events[3].skipped == 0);
}

TEST_CASE("an in-order reader that falls behind is told what it skipped", "[camera_core][delivery]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(8));
    RecordingSink sink;
    const SubscribeResult reader = publisher.subscribe(sink, group, { .window = 2 });
    REQUIRE(reader);

    for (int i = 0; i < 6; ++i)
    {
        publish_next(publisher, group);
    }
    REQUIRE(sequences(sink) == std::vector<std::uint64_t>{ 1, 2 });

    // The ring kept the last two (5 and 6); 3 and 4 are gone.
    publisher.release(reader.id, sink.events[0].lease_id);
    REQUIRE(sequences(sink) == std::vector<std::uint64_t>{ 1, 2, 5 });
    REQUIRE(sink.events[2].skipped == 2);

    publisher.release(reader.id, sink.events[1].lease_id);
    REQUIRE(sequences(sink) == std::vector<std::uint64_t>{ 1, 2, 5, 6 });
    REQUIRE(sink.events[3].skipped == 0);
}

TEST_CASE("newest order with a larger window skips to the newest frame set", "[camera_core][delivery]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(8));
    RecordingSink sink;
    const SubscribeResult reader = publisher.subscribe(sink, group, { .window = 2, .order = DeliveryOrder::NEWEST });
    REQUIRE(reader);

    for (int i = 0; i < 4; ++i)
    {
        publish_next(publisher, group);
    }
    REQUIRE(sequences(sink) == std::vector<std::uint64_t>{ 1, 2 });

    // A viewer that holds one frame and wants the newest must not be handed frame set 3.
    publisher.release(reader.id, sink.events[0].lease_id);
    REQUIRE(sequences(sink) == std::vector<std::uint64_t>{ 1, 2, 4 });
    REQUIRE(sink.events[2].skipped == 1);
}

TEST_CASE("a subscription that does not fit the pool is refused", "[camera_core][budget]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(4));
    RecordingSink first;
    RecordingSink second;

    // 1 ring history + 1 window + 1 backend reserve = 3 of 4 slots.
    REQUIRE(publisher.subscribe(first, group, { .window = 1 }));

    // An in-order reader with a window of two needs 2 ring history + 3 windows + 1 reserve = 6.
    const SubscribeResult refused = publisher.subscribe(second, group, { .window = 2 });
    REQUIRE_FALSE(refused);
    REQUIRE_THAT(refused.error, Catch::Matchers::ContainsSubstring("needs 6 slots"));
    REQUIRE_THAT(refused.error, Catch::Matchers::ContainsSubstring("pool has 4"));

    // A second newest-only reader needs 1 + 2 + 1 = 4 and fits.
    REQUIRE(publisher.subscribe(second, group, { .window = 1 }));

    REQUIRE_FALSE(publisher.subscribe(second, group, { .window = 0 }));
    REQUIRE_FALSE(publisher.subscribe(second, 99, { .window = 1 }));
}

TEST_CASE("the backend reserve counts against the pool", "[camera_core][budget]")
{
    Publisher publisher;
    // An SDK that keeps four captures in flight, on a pool of eight.
    const GroupId group = publisher.add_group(make_group(8), 4);
    RecordingSink sinks[4];

    // 1 ring history + 3 windows + 4 reserve = 8.
    for (int i = 0; i < 3; ++i)
    {
        REQUIRE(publisher.subscribe(sinks[i], group, { .window = 1 }));
    }
    const SubscribeResult refused = publisher.subscribe(sinks[3], group, { .window = 1 });
    REQUIRE_FALSE(refused);
    REQUIRE_THAT(refused.error, Catch::Matchers::ContainsSubstring("4 for the backend"));
}

TEST_CASE("a slot is reused only when neither the ring nor a reader holds it", "[camera_core][slots]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(3, 1));
    RecordingSink sink;
    const SubscribeResult reader = publisher.subscribe(sink, group, { .window = 1 });
    REQUIRE(reader);

    publish_next(publisher, group); // slot 0: held by the reader
    publish_next(publisher, group); // slot 1: held by the ring
    REQUIRE(sink.events.size() == 1);
    REQUIRE(sink.events[0].images[0].slot == 0);

    const auto third = publisher.acquire_slot(group, 0);
    REQUIRE(third == 2u);
    REQUIRE_FALSE(publisher.acquire_slot(group, 0).has_value());
    REQUIRE(publisher.stats().no_free_slot == 1);
    publisher.abandon_slot(group, 0, *third);

    // Releasing frame set 1 frees slot 0 and hands the reader frame set 2.
    publisher.release(reader.id, sink.events[0].lease_id);
    REQUIRE(sink.events.size() == 2);
    REQUIRE(sink.events[1].images[0].slot == 1);
    REQUIRE(publisher.acquire_slot(group, 0) == 0u);
}

TEST_CASE("the slot-free callback fires when the last holder lets go", "[camera_core][slots]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(4, 1));
    std::vector<std::pair<std::size_t, std::uint32_t>> freed;
    publisher.set_slot_free_callback(group,
                                     [&](std::size_t stream, std::uint32_t slot)
                                     {
                                         // Runs outside the publisher's lock, so calling back in must not deadlock.
                                         (void)publisher.stats();
                                         freed.emplace_back(stream, slot);
                                     });
    RecordingSink sink;
    const SubscribeResult reader = publisher.subscribe(sink, group, { .window = 1 });
    REQUIRE(reader);

    // The SDK owns the buffers and picks the slot it filled.
    const auto publish_slot = [&](std::uint32_t slot)
    {
        publisher.claim_slot(group, 0, slot);
        ImageInfo image;
        image.slot = slot;
        publisher.publish(group, { &image, 1 }, {});
    };

    publish_slot(0); // ring + reader
    publish_slot(1); // the ring lets slot 0 go, the reader still has it
    REQUIRE(freed.empty());
    REQUIRE_THROWS_AS(publisher.claim_slot(group, 0, 0), std::invalid_argument);

    publisher.release(reader.id, sink.events[0].lease_id); // reader moves on to slot 1
    REQUIRE(freed == std::vector<std::pair<std::size_t, std::uint32_t>>{ { 0, 0 } });

    publish_slot(2); // the ring lets slot 1 go, the reader still has it
    REQUIRE(freed.size() == 1);
    publisher.unsubscribe(reader.id);
    REQUIRE(freed == std::vector<std::pair<std::size_t, std::uint32_t>>{ { 0, 0 }, { 0, 1 } });

    // A slot the backend gives up itself needs no notification.
    publisher.claim_slot(group, 0, 0);
    publisher.abandon_slot(group, 0, 0);
    REQUIRE(freed.size() == 2);
}

TEST_CASE("unsubscribing returns every frame set the reader held", "[camera_core][slots]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(8, 1));
    RecordingSink sink;
    const SubscribeResult reader = publisher.subscribe(sink, group, { .window = 3 });
    REQUIRE(reader);
    for (int i = 0; i < 3; ++i)
    {
        publish_next(publisher, group);
    }
    REQUIRE(sink.events.size() == 3);

    publisher.unsubscribe(reader.id);
    // Only the newest frame set stays in the ring: seven of the eight slots are free again.
    for (int i = 0; i < 7; ++i)
    {
        REQUIRE(publisher.acquire_slot(group, 0).has_value());
    }
    REQUIRE_FALSE(publisher.acquire_slot(group, 0).has_value());

    // Releasing after the subscription ended is harmless.
    publisher.release(reader.id, sink.events[0].lease_id);
}

TEST_CASE("a frame set that could not be sent is not counted as held", "[camera_core][delivery]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(3, 1));
    RecordingSink sink;
    const SubscribeResult reader = publisher.subscribe(sink, group, { .window = 1 });
    REQUIRE(reader);

    sink.accept = false;
    publish_next(publisher, group);
    REQUIRE(publisher.stats().send_failures == 1);
    REQUIRE(sink.events.empty());

    sink.accept = true;
    publish_next(publisher, group);
    REQUIRE(sequences(sink) == std::vector<std::uint64_t>{ 2 });
    REQUIRE(sink.events[0].skipped == 0);
    REQUIRE(publisher.stats().events_sent == 1);

    // Frame set 1 left the ring and no reader holds it: its slot is free.
    REQUIRE(publisher.acquire_slot(group, 0) == 0u);
}

TEST_CASE("a restart changes the generation and empties the ring", "[camera_core][lifecycle]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(4, 1));
    RecordingSink sink;
    const SubscribeResult reader = publisher.subscribe(sink, group, { .window = 1 });
    REQUIRE(reader);

    publish_next(publisher, group);
    REQUIRE(sink.events[0].generation == 0);

    publisher.restart_group(group);
    REQUIRE(publisher.generation(group) == 1);

    // The frame set the reader holds stays valid; the sequence keeps counting.
    publisher.release(reader.id, sink.events[0].lease_id);
    REQUIRE(sink.events.size() == 1);
    publish_next(publisher, group);
    REQUIRE(sink.events.size() == 2);
    REQUIRE(sink.events[1].generation == 1);
    REQUIRE(sink.events[1].sequence == 2);
}

TEST_CASE("a group publishes one image per stream", "[camera_core][groups]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(4, 3, "rig"));
    REQUIRE(publisher.find_group("rig") == group);
    REQUIRE_FALSE(publisher.find_group("head").has_value());
    REQUIRE(publisher.group_count() == 1);

    RecordingSink sink;
    REQUIRE(publisher.subscribe(sink, group, { .window = 1 }));

    std::vector<ImageInfo> images(3);
    for (std::size_t i = 0; i < 3; ++i)
    {
        images[i].slot = *publisher.acquire_slot(group, i);
        images[i].device_frame_id = 100 + i;
    }
    images[1].flags = IMAGE_FLAG_KEYFRAME;
    images[1].bytes_used = 4096;
    images[2].flags = IMAGE_FLAG_SDK_WARNING;
    publisher.publish(group, images, {});

    REQUIRE(sink.events.size() == 1);
    const FrameEvent& event = sink.events[0];
    REQUIRE(event.image_count == 3);
    REQUIRE(event.images[0].device_frame_id == 100);
    REQUIRE(event.images[1].flags == IMAGE_FLAG_KEYFRAME);
    REQUIRE(event.images[1].bytes_used == 4096);
    REQUIRE(event.images[2].flags == IMAGE_FLAG_SDK_WARNING);
}

TEST_CASE("an encoded stream is described by its codec, not by a pixel format", "[camera_core][groups]")
{
    GroupDescription description = make_group(4, 1, "head.enc");
    StreamDescription& stream = description.streams[0];
    REQUIRE_FALSE(stream.is_encoded());
    stream.codec = Codec::H265;
    stream.stride_bytes = 0;
    stream.slot_bytes = 2 * 1024 * 1024; // room for one access unit

    Publisher publisher;
    const GroupId group = publisher.add_group(description);
    const StreamDescription stored = publisher.group_description(group).streams[0];
    REQUIRE(stored.is_encoded());
    REQUIRE(stored.codec == Codec::H265);
    REQUIRE(stored.slot_bytes == 2 * 1024 * 1024);
}

TEST_CASE("every frame set carries its timeline", "[camera_core][timeline]")
{
    std::int64_t now = 1000;
    Publisher publisher([&] { return now; });
    const GroupId group = publisher.add_group(make_group(4, 1));
    RecordingSink sink;
    const SubscribeResult reader = publisher.subscribe(sink, group, { .window = 1 });
    REQUIRE(reader);

    FrameTimeline timeline;
    timeline.capture_ns = 100;
    timeline.sdk_ready_ns = 900;
    publish_next(publisher, group, timeline);
    REQUIRE(sink.events[0].timeline.capture_ns == 100);
    REQUIRE(sink.events[0].timeline.sdk_ready_ns == 900);
    REQUIRE(sink.events[0].timeline.published_ns == 1000);
    REQUIRE(sink.events[0].timeline.sent_ns == 1000);

    // A frame set that waits for the reader is stamped as sent when it leaves.
    now = 2000;
    publish_next(publisher, group);
    now = 2500;
    publisher.release(reader.id, sink.events[0].lease_id);
    REQUIRE(sink.events[1].timeline.capture_ns == TIME_UNKNOWN);
    REQUIRE(sink.events[1].timeline.published_ns == 2000);
    REQUIRE(sink.events[1].timeline.sent_ns == 2500);
}

TEST_CASE("misuse by a backend is rejected", "[camera_core][validation]")
{
    Publisher publisher;
    REQUIRE_THROWS_AS(publisher.add_group(make_group(4, 0)), std::invalid_argument);
    REQUIRE_THROWS_AS(publisher.add_group(make_group(4, MAX_STREAMS_PER_GROUP + 1)), std::invalid_argument);
    REQUIRE_THROWS_AS(publisher.add_group(make_group(0)), std::invalid_argument);
    REQUIRE_THROWS_AS(publisher.add_group(make_group(MAX_SLOTS_PER_STREAM + 1)), std::invalid_argument);
    REQUIRE_THROWS_AS(publisher.add_group(make_group(4), 0), std::invalid_argument);

    const GroupId group = publisher.add_group(make_group(4));
    REQUIRE_THROWS_AS(publisher.add_group(make_group(4)), std::invalid_argument); // same id

    // Publishing needs one image per stream, each in a slot the backend took first.
    std::vector<ImageInfo> images(2);
    REQUIRE_THROWS_AS(publisher.publish(group, images, {}), std::invalid_argument);
    images[0].slot = *publisher.acquire_slot(group, 0);
    REQUIRE_THROWS_AS(publisher.publish(group, { images.data(), 1 }, {}), std::invalid_argument);
    REQUIRE_THROWS_AS(publisher.publish(group, images, {}), std::invalid_argument); // stream 1 not acquired
    REQUIRE(publisher.stats().published == 0);

    REQUIRE_THROWS_AS(publisher.abandon_slot(group, 1, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(publisher.acquire_slot(99, 0), std::out_of_range);
    REQUIRE_THROWS_AS(publisher.claim_slot(group, 0, 99), std::invalid_argument);
}

TEST_CASE("capture and a reader on different threads keep every frame set in order", "[camera_core][threads]")
{
    // A reader as the endpoint will drive it: events arrive on the capture thread, the
    // reader's own thread takes them and releases.
    struct QueueSink : EventSink
    {
        bool send_event(SubscriptionId, const FrameEvent& event) override
        {
            {
                std::lock_guard<std::mutex> lock(mutex);
                queue.push_back(event);
            }
            ready.notify_one();
            return true;
        }

        std::mutex mutex;
        std::condition_variable ready;
        std::deque<FrameEvent> queue;
    };

    constexpr std::uint64_t FRAME_SETS = 2000;
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(16));
    QueueSink sink;
    const SubscribeResult reader = publisher.subscribe(sink, group, { .window = 4 });
    REQUIRE(reader);

    std::atomic<std::uint64_t> received{ 0 };
    std::atomic<std::uint64_t> out_of_order{ 0 };
    std::thread consumer(
        [&]
        {
            std::uint64_t last = 0;
            while (last < FRAME_SETS)
            {
                FrameEvent event;
                {
                    std::unique_lock<std::mutex> lock(sink.mutex);
                    sink.ready.wait(lock, [&] { return !sink.queue.empty(); });
                    event = sink.queue.front();
                    sink.queue.pop_front();
                }
                out_of_order += event.sequence <= last ? 1 : 0;
                last = event.sequence;
                ++received;
                publisher.release(reader.id, event.lease_id);
            }
        });

    std::uint64_t dropped = 0;
    for (std::uint64_t i = 0; i < FRAME_SETS; ++i)
    {
        std::vector<ImageInfo> images(2);
        const auto left = publisher.acquire_slot(group, 0);
        const auto right = publisher.acquire_slot(group, 1);
        if (!left || !right)
        {
            // Cannot happen while the budget holds: a free slot is always left for the backend.
            ++dropped;
            continue;
        }
        images[0].slot = *left;
        images[1].slot = *right;
        publisher.publish(group, images, {});
    }
    consumer.join();

    REQUIRE(dropped == 0);
    REQUIRE(out_of_order == 0);
    REQUIRE(received > 0);
    REQUIRE(publisher.stats().published == FRAME_SETS);
}
