// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <camera_core/publisher.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <limits>
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
    bool send_event(SubscriptionId, const FrameEvent& event) noexcept override
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
        stream.plane_count = 1;
        stream.planes[0].stride_bytes = 640 * 4;
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

// A subscription as the endpoint sets one up: reserved, then, once the reader knows of it, active.
SubscribeResult open(Publisher& publisher, EventSink& sink, GroupId group, const ReaderOptions& options = {})
{
    const SubscribeResult result = publisher.subscribe(sink, group, options);
    if (result)
    {
        publisher.activate(result.id);
    }
    return result;
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

TEST_CASE("a newest-order reader gets the newest frame set when it asks", "[unit][camera_core][delivery]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(8));
    RecordingSink sink;
    const SubscribeResult reader = open(publisher, sink, group, { .window = 1 });
    REQUIRE(reader);

    // Nothing is held for a reader that has not asked.
    publish_next(publisher, group);
    publish_next(publisher, group);
    REQUIRE(sink.events.empty());

    REQUIRE(publisher.request_newest(reader.id, 0));
    REQUIRE(sequences(sink) == std::vector<std::uint64_t>{ 2 });
    REQUIRE(sink.events[0].skipped == 0);
    REQUIRE(sink.events[0].image_count == 2);

    // It asks again while it still holds frame set 2: the answer waits for room, and is the
    // newest frame set there is when the room comes.
    REQUIRE(publisher.request_newest(reader.id, 2));
    publish_next(publisher, group);
    publish_next(publisher, group);
    REQUIRE(sink.events.size() == 1);

    publisher.release(reader.id, sink.events[0].lease_id);
    REQUIRE(sequences(sink) == std::vector<std::uint64_t>{ 2, 4 });
    REQUIRE(sink.events[1].skipped == 1);
}

TEST_CASE("a newest-order reader that is waiting is sent a frame set as it is published", "[unit][camera_core][delivery]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(8));
    RecordingSink sink;
    const SubscribeResult reader = open(publisher, sink, group, { .window = 1 });
    REQUIRE(reader);

    REQUIRE(publisher.request_newest(reader.id, 0));
    REQUIRE(sink.events.empty());
    publish_next(publisher, group);
    REQUIRE(sequences(sink) == std::vector<std::uint64_t>{ 1 });

    // One request, one frame set.
    publisher.release(reader.id, sink.events[0].lease_id);
    publish_next(publisher, group);
    REQUIRE(sink.events.size() == 1);
}

TEST_CASE("a reader that paused gets the newest frame set, not the one from when it stopped",
          "[unit][camera_core][delivery]")
{
    std::int64_t now = 0;
    Publisher publisher([&] { return now; });
    const GroupId group = publisher.add_group(make_group(4, 1));
    RecordingSink sink;
    const SubscribeResult reader = open(publisher, sink, group, { .window = 1 });
    REQUIRE(reader);

    REQUIRE(publisher.request_newest(reader.id, 0));
    publish_next(publisher, group);
    publisher.release(reader.id, sink.events[0].lease_id);

    // The reader does something else for five frame periods, holding nothing and asking for
    // nothing. Nothing is put aside for it: every slot but the ring's one is free throughout.
    for (int i = 0; i < 5; ++i)
    {
        now += 10;
        publish_next(publisher, group);
        REQUIRE(sink.events.size() == 1);
        std::vector<std::uint32_t> free_slots;
        while (const auto slot = publisher.acquire_slot(group, 0))
        {
            free_slots.push_back(*slot);
        }
        REQUIRE(free_slots.size() == 3);
        for (const std::uint32_t slot : free_slots)
        {
            publisher.abandon_slot(group, 0, slot);
        }
    }

    // When it comes back it is handed the frame set published last, held from this moment.
    now += 3;
    REQUIRE(publisher.request_newest(reader.id, 1));
    REQUIRE(sequences(sink) == std::vector<std::uint64_t>{ 1, 6 });
    REQUIRE(sink.events[1].skipped == 4);
    REQUIRE(sink.events[1].timeline.published_ns == 50);
    REQUIRE(sink.events[1].timeline.sent_ns == 53);
}

TEST_CASE("a frame set given back unseen is sent again while it is still the newest", "[unit][camera_core][delivery]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(8));
    RecordingSink sink;
    const SubscribeResult reader = open(publisher, sink, group, { .window = 1 });
    REQUIRE(reader);

    REQUIRE(publisher.request_newest(reader.id, 0));
    publish_next(publisher, group);
    publisher.release(reader.id, sink.events[0].lease_id);

    // The reader asked, gave up waiting, and frame set 2 arrived for it later. It cannot tell
    // whether 2 is still the newest, so it gives it back and asks again, naming the last one
    // it took.
    REQUIRE(publisher.request_newest(reader.id, 1));
    publish_next(publisher, group);
    REQUIRE(sequences(sink) == std::vector<std::uint64_t>{ 1, 2 });
    publisher.release(reader.id, sink.events[1].lease_id);
    REQUIRE(publisher.request_newest(reader.id, 1));
    REQUIRE(sequences(sink) == std::vector<std::uint64_t>{ 1, 2, 2 });
    REQUIRE(sink.events[2].skipped == 0);
    REQUIRE(sink.events[2].lease_id != sink.events[1].lease_id);

    // Had a newer one been published meanwhile, that is the one it gets.
    publish_next(publisher, group);
    publisher.release(reader.id, sink.events[2].lease_id);
    REQUIRE(publisher.request_newest(reader.id, 1));
    REQUIRE(sink.events.back().sequence == 3);
    REQUIRE(sink.events.back().skipped == 1);
}

TEST_CASE("a subscription is sent nothing until it is activated", "[unit][camera_core][delivery]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(8));
    RecordingSink in_order;
    RecordingSink newest;
    const SubscribeResult ordered = publisher.subscribe(in_order, group, { .window = 2 });
    const SubscribeResult latest = publisher.subscribe(newest, group, { .window = 1 });
    REQUIRE(ordered);
    REQUIRE(latest);

    // Reserved: both count in the budget (2 ring history + 3 windows + 1 reserve), neither
    // is sent anything, and a request is refused.
    for (int i = 0; i < 3; ++i)
    {
        publish_next(publisher, group);
    }
    REQUIRE(in_order.events.empty());
    REQUIRE_FALSE(publisher.request_newest(latest.id, 0));
    RecordingSink third;
    REQUIRE_FALSE(publisher.subscribe(third, group, { .window = 3 }));

    // Active: the in-order reader starts at once with what the ring kept, which is what was
    // published while it was only reserved. It does not have to wait for the next publish.
    publisher.activate(ordered.id);
    publisher.activate(latest.id);
    REQUIRE(sequences(in_order) == std::vector<std::uint64_t>{ 2, 3 });
    REQUIRE(in_order.events[0].skipped == 0);
    REQUIRE(newest.events.empty());
    publish_next(publisher, group);
    REQUIRE(in_order.events.size() == 2); // its window of two is full

    // Asking is for newest-order readers.
    REQUIRE_FALSE(publisher.request_newest(ordered.id, 0));
    REQUIRE(publisher.request_newest(latest.id, 0));
    REQUIRE(sequences(newest) == std::vector<std::uint64_t>{ 4 });
    REQUIRE_FALSE(publisher.request_newest(99, 0));
}

TEST_CASE("a larger window gets every frame set in order", "[unit][camera_core][delivery]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(8));
    RecordingSink sink;
    const SubscribeResult reader = open(publisher, sink, group, { .window = 3 });
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

TEST_CASE("an in-order reader that falls behind is told what it skipped", "[unit][camera_core][delivery]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(8));
    RecordingSink sink;
    const SubscribeResult reader = open(publisher, sink, group, { .window = 2 });
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

TEST_CASE("newest order with a larger window skips to the newest frame set", "[unit][camera_core][delivery]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(8));
    RecordingSink sink;
    const SubscribeResult reader = open(publisher, sink, group, { .window = 2, .order = DeliveryOrder::NEWEST });
    REQUIRE(reader);

    // It holds two frame sets at a time and asks for a third.
    REQUIRE(publisher.request_newest(reader.id, 0));
    publish_next(publisher, group);
    REQUIRE(publisher.request_newest(reader.id, 1));
    publish_next(publisher, group);
    REQUIRE(sequences(sink) == std::vector<std::uint64_t>{ 1, 2 });
    REQUIRE(publisher.request_newest(reader.id, 2));
    publish_next(publisher, group);
    publish_next(publisher, group);
    REQUIRE(sink.events.size() == 2);

    // A viewer that holds one frame and wants the newest must not be handed frame set 3.
    publisher.release(reader.id, sink.events[0].lease_id);
    REQUIRE(sequences(sink) == std::vector<std::uint64_t>{ 1, 2, 4 });
    REQUIRE(sink.events[2].skipped == 1);
}

TEST_CASE("a subscription that does not fit the pool is refused", "[unit][camera_core][budget]")
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

TEST_CASE("the backend reserve counts against the pool", "[unit][camera_core][budget]")
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

TEST_CASE("a slot is reused only when neither the ring nor a reader holds it", "[unit][camera_core][slots]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(3, 1));
    RecordingSink sink;
    const SubscribeResult reader = open(publisher, sink, group, { .window = 1 });
    REQUIRE(reader);
    REQUIRE(publisher.request_newest(reader.id, 0));

    publish_next(publisher, group); // slot 0: held by the reader
    publish_next(publisher, group); // slot 1: held by the ring
    REQUIRE(sink.events.size() == 1);
    REQUIRE(sink.events[0].images[0].slot == 0);
    REQUIRE(publisher.request_newest(reader.id, 1)); // waits for the reader to have room

    const auto third = publisher.acquire_slot(group, 0);
    REQUIRE(third == 2u);
    REQUIRE_FALSE(publisher.acquire_slot(group, 0).has_value());
    REQUIRE(publisher.stats(group).no_free_slot == 1);
    publisher.abandon_slot(group, 0, *third);

    // Releasing frame set 1 frees slot 0 and hands the reader frame set 2.
    publisher.release(reader.id, sink.events[0].lease_id);
    REQUIRE(sink.events.size() == 2);
    REQUIRE(sink.events[1].images[0].slot == 1);
    REQUIRE(publisher.acquire_slot(group, 0) == 0u);
}

TEST_CASE("the slot-free callback fires when the last holder lets go", "[unit][camera_core][slots]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(4, 1));
    std::vector<std::pair<std::size_t, std::uint32_t>> freed;
    publisher.set_slot_free_callback(group,
                                     [&](std::size_t stream, std::uint32_t slot)
                                     {
                                         // Runs outside the publisher's lock, so calling back in must not deadlock.
                                         (void)publisher.stats(group);
                                         freed.emplace_back(stream, slot);
                                     });
    RecordingSink sink;
    const SubscribeResult reader = open(publisher, sink, group, { .window = 1 });
    REQUIRE(reader);
    REQUIRE(publisher.request_newest(reader.id, 0));

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

    REQUIRE(publisher.request_newest(reader.id, 1));
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

TEST_CASE("unsubscribing returns every frame set the reader held", "[unit][camera_core][slots]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(8, 1));
    RecordingSink sink;
    const SubscribeResult reader = open(publisher, sink, group, { .window = 3 });
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

TEST_CASE("a frame set that could not be sent is not counted as held", "[unit][camera_core][delivery]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(3, 1));
    RecordingSink sink;
    const SubscribeResult reader = open(publisher, sink, group, { .window = 1 });
    REQUIRE(reader);
    REQUIRE(publisher.request_newest(reader.id, 0));

    sink.accept = false;
    publish_next(publisher, group);
    REQUIRE(publisher.stats(group).send_failures == 1);
    REQUIRE(sink.events.empty());

    // The reader is still waiting, so the next frame set is tried too.
    sink.accept = true;
    publish_next(publisher, group);
    REQUIRE(sequences(sink) == std::vector<std::uint64_t>{ 2 });
    REQUIRE(sink.events[0].skipped == 0);
    REQUIRE(publisher.stats(group).events_sent == 1);

    // Frame set 1 left the ring and no reader holds it: its slot is free.
    REQUIRE(publisher.acquire_slot(group, 0) == 0u);
}

TEST_CASE("a restart changes the generation and empties the ring", "[unit][camera_core][lifecycle]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(4, 1));
    RecordingSink sink;
    const SubscribeResult reader = open(publisher, sink, group, { .window = 1 });
    REQUIRE(reader);
    REQUIRE(publisher.request_newest(reader.id, 0));

    publish_next(publisher, group);
    REQUIRE(sink.events[0].generation == 0);

    publisher.restart_group(group);
    REQUIRE(publisher.generation(group) == 1);

    // The frame set the reader holds stays valid, and its slot out of use, until it is given
    // back; the sequence keeps counting.
    REQUIRE(publisher.acquire_slot(group, 0) == 1u);
    publisher.abandon_slot(group, 0, 1);
    publisher.release(reader.id, sink.events[0].lease_id);
    REQUIRE(publisher.acquire_slot(group, 0) == 0u);
    publisher.abandon_slot(group, 0, 0);
    REQUIRE(sink.events.size() == 1);
    REQUIRE(publisher.request_newest(reader.id, 1));
    REQUIRE(sink.events.size() == 1); // the ring was emptied: nothing from before the restart
    publish_next(publisher, group);
    REQUIRE(sink.events.size() == 2);
    REQUIRE(sink.events[1].generation == 1);
    REQUIRE(sink.events[1].sequence == 2);
}

TEST_CASE("a group publishes one image per stream", "[unit][camera_core][groups]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(4, 3, "rig"));
    REQUIRE(publisher.find_group("rig") == group);
    REQUIRE_FALSE(publisher.find_group("head").has_value());
    REQUIRE(publisher.group_count() == 1);

    RecordingSink sink;
    const SubscribeResult reader = open(publisher, sink, group, { .window = 1 });
    REQUIRE(reader);
    REQUIRE(publisher.request_newest(reader.id, 0));

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

TEST_CASE("an encoded stream is described by its codec, not by a pixel format", "[unit][camera_core][groups]")
{
    GroupDescription description = make_group(4, 1, "head.enc");
    StreamDescription& stream = description.streams[0];
    REQUIRE_FALSE(stream.is_encoded());
    stream.codec = Codec::H265;
    stream.plane_count = 0;
    stream.slot_bytes = 2 * 1024 * 1024; // room for one access unit

    Publisher publisher;
    const GroupId group = publisher.add_group(description);
    const StreamDescription stored = publisher.group_description(group).streams[0];
    REQUIRE(stored.is_encoded());
    REQUIRE(stored.codec == Codec::H265);
    REQUIRE(stored.slot_bytes == 2 * 1024 * 1024);
}

TEST_CASE("every frame set carries its timeline", "[unit][camera_core][timeline]")
{
    std::int64_t now = 1000;
    Publisher publisher([&] { return now; });
    const GroupId group = publisher.add_group(make_group(4, 1));
    RecordingSink sink;
    const SubscribeResult reader = open(publisher, sink, group, { .window = 1 });
    REQUIRE(reader);
    REQUIRE(publisher.request_newest(reader.id, 0));

    FrameTimeline timeline;
    timeline.capture_ns = 100;
    timeline.sdk_ready_ns = 900;
    publish_next(publisher, group, timeline);
    REQUIRE(sink.events[0].timeline.capture_ns == 100);
    REQUIRE(sink.events[0].timeline.sdk_ready_ns == 900);
    REQUIRE(sink.events[0].timeline.published_ns == 1000);
    REQUIRE(sink.events[0].timeline.sent_ns == 1000);

    // A frame set that waits for the reader is stamped as sent when it leaves.
    REQUIRE(publisher.request_newest(reader.id, 1));
    now = 2000;
    publish_next(publisher, group);
    now = 2500;
    publisher.release(reader.id, sink.events[0].lease_id);
    REQUIRE(sink.events[1].timeline.capture_ns == TIME_UNKNOWN);
    REQUIRE(sink.events[1].timeline.published_ns == 2000);
    REQUIRE(sink.events[1].timeline.sent_ns == 2500);
}

TEST_CASE("misuse by a backend is rejected", "[unit][camera_core][validation]")
{
    Publisher publisher;
    REQUIRE_THROWS_AS(publisher.add_group(make_group(4, 0)), std::invalid_argument);
    REQUIRE_THROWS_AS(publisher.add_group(make_group(4, MAX_STREAMS_PER_GROUP + 1)), std::invalid_argument);
    REQUIRE_THROWS_AS(publisher.add_group(make_group(0)), std::invalid_argument);
    REQUIRE_THROWS_AS(publisher.add_group(make_group(MAX_SLOTS_PER_STREAM + 1)), std::invalid_argument);
    REQUIRE_THROWS_AS(publisher.add_group(make_group(4), 0), std::invalid_argument);

    // A pool holds at least one frame set for the ring on top of the backend's reserve.
    REQUIRE_THROWS_AS(publisher.add_group(make_group(1)), std::invalid_argument);
    REQUIRE_THROWS_AS(publisher.add_group(make_group(4), 4), std::invalid_argument);
    REQUIRE_NOTHROW(publisher.add_group(make_group(4, 2, "tight"), 3));

    // A pixel stream names every plane of its format; an encoded stream has none.
    GroupDescription nv12 = make_group(4, 1, "nv12");
    nv12.streams[0].pixel_format = PixelFormat::NV12;
    REQUIRE_THROWS_AS(publisher.add_group(nv12), std::invalid_argument);
    nv12.streams[0].plane_count = 2;
    nv12.streams[0].planes[1] = { .offset_bytes = 640 * 400, .stride_bytes = 640 };
    REQUIRE_NOTHROW(publisher.add_group(nv12));
    GroupDescription encoded_with_planes = make_group(4, 1, "enc");
    encoded_with_planes.streams[0].codec = Codec::H264;
    REQUIRE_THROWS_AS(publisher.add_group(encoded_with_planes), std::invalid_argument);
    GroupDescription no_size = make_group(4, 1, "empty");
    no_size.streams[0].width = 0;
    REQUIRE_THROWS_AS(publisher.add_group(no_size), std::invalid_argument);

    const GroupId group = publisher.add_group(make_group(4));
    REQUIRE_THROWS_AS(publisher.add_group(make_group(4)), std::invalid_argument); // same id

    // Publishing needs one image per stream, each in a slot the backend took first.
    std::vector<ImageInfo> images(2);
    REQUIRE_THROWS_AS(publisher.publish(group, images, {}), std::invalid_argument);
    images[0].slot = *publisher.acquire_slot(group, 0);
    REQUIRE_THROWS_AS(publisher.publish(group, { images.data(), 1 }, {}), std::invalid_argument);
    REQUIRE_THROWS_AS(publisher.publish(group, images, {}), std::invalid_argument); // stream 1 not acquired

    // An image cannot be larger than the slot it sits in.
    images[1].slot = *publisher.acquire_slot(group, 1);
    images[1].bytes_used = publisher.group_description(group).streams[1].slot_bytes + 1;
    REQUIRE_THROWS_AS(publisher.publish(group, images, {}), std::invalid_argument);
    REQUIRE(publisher.stats(group).published == 0);
    images[1].bytes_used = 0;
    REQUIRE(publisher.publish(group, images, {}) == 1); // the slots were still the backend's to publish

    REQUIRE_THROWS_AS(publisher.abandon_slot(group, 1, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(publisher.acquire_slot(99, 0), std::out_of_range);
    REQUIRE_THROWS_AS(publisher.claim_slot(group, 0, 99), std::invalid_argument);
}

TEST_CASE("a plane that does not lie inside the slot is rejected", "[unit][camera_core][validation]")
{
    Publisher publisher;
    const auto with = [](auto&& change)
    {
        GroupDescription group = make_group(4, 1);
        change(group.streams[0]);
        return group;
    };

    // A row is as long as the format needs, and the last row ends inside the slot.
    REQUIRE_THROWS_AS(
        publisher.add_group(with([](StreamDescription& stream) { stream.planes[0].stride_bytes = 640 * 4 - 1; })),
        std::invalid_argument);
    REQUIRE_THROWS_AS(publisher.add_group(with([](StreamDescription& stream) { stream.planes[0].offset_bytes = 1; })),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(
        publisher.add_group(with([](StreamDescription& stream) { stream.slot_bytes -= 1; })), std::invalid_argument);
    REQUIRE_THROWS_AS(
        publisher.add_group(with([](StreamDescription& stream) { stream.planes[0].stride_bytes = 0xFFFFFFFF; })),
        std::invalid_argument);

    // Padding between rows is fine as long as the last row still fits.
    const auto padded = [](StreamDescription& stream)
    {
        stream.planes[0].stride_bytes = 4096;
        stream.slot_bytes = 4096 * 399 + 640 * 4;
    };
    REQUIRE_NOTHROW(publisher.add_group(with(padded)));

    // NV12 with odd sides: chroma rows cover the rounded-up half, and both planes must fit.
    const auto nv12 = [](std::uint64_t slot_bytes)
    {
        return [slot_bytes](StreamDescription& stream)
        {
            stream.id = "nv12";
            stream.pixel_format = PixelFormat::NV12;
            stream.width = 5;
            stream.height = 3;
            stream.plane_count = 2;
            stream.planes[0] = { .offset_bytes = 0, .stride_bytes = 8 };
            stream.planes[1] = { .offset_bytes = 24, .stride_bytes = 8 };
            stream.slot_bytes = slot_bytes;
        };
    };
    REQUIRE_THROWS_AS(publisher.add_group(with(nv12(24 + 8 + 5))), std::invalid_argument); // chroma row needs 6
    GroupDescription odd = with(nv12(24 + 8 + 6));
    odd.id = "odd";
    REQUIRE_NOTHROW(publisher.add_group(odd));
}

TEST_CASE("a description is checked without sums that can wrap", "[unit][camera_core][validation]")
{
    const auto with = [](auto&& change)
    {
        GroupDescription group = make_group(4, 2);
        change(group);
        return problem_with(group);
    };
    REQUIRE(problem_with(make_group(4, 2)).empty());

    // The largest numbers the fields hold: a product of them would wrap 64 bits.
    const std::uint32_t most = std::numeric_limits<std::uint32_t>::max();
    REQUIRE_FALSE(with(
                      [&](GroupDescription& group)
                      {
                          StreamDescription& stream = group.streams[0];
                          stream.width = most;
                          stream.height = most;
                          stream.planes[0] = { .offset_bytes = most, .stride_bytes = most };
                          stream.slot_bytes = MAX_SLOT_BYTES;
                      })
                      .empty());
    REQUIRE_THAT(with([](GroupDescription& group) { group.streams[0].slot_bytes = MAX_SLOT_BYTES + 1; }),
                 Catch::Matchers::ContainsSubstring("slot size"));
    REQUIRE_THAT(with([](GroupDescription& group) { group.streams[1].id = group.streams[0].id; }),
                 Catch::Matchers::ContainsSubstring("two streams called"));
    REQUIRE_THAT(
        with([](GroupDescription& group) { group.streams[0].frame_rate = std::numeric_limits<double>::quiet_NaN(); }),
        Catch::Matchers::ContainsSubstring("frame rate"));
    REQUIRE_THAT(with([](GroupDescription& group) { group.streams[0].frame_rate = -1.0; }),
                 Catch::Matchers::ContainsSubstring("frame rate"));
    REQUIRE_THAT(with([](GroupDescription& group) { group.streams[0].plane_count = MAX_PLANES_PER_IMAGE + 1; }),
                 Catch::Matchers::ContainsSubstring("plane"));
    REQUIRE_THAT(with([](GroupDescription& group) { group.streams[0].id.clear(); }),
                 Catch::Matchers::ContainsSubstring("without an id"));
    REQUIRE_THAT(
        with([](GroupDescription& group) { group.id.clear(); }), Catch::Matchers::ContainsSubstring("needs an id"));

    // A frame rate that is not known is described as 0.
    REQUIRE(with([](GroupDescription& group) { group.streams[0].frame_rate = 0.0; }).empty());
}

TEST_CASE("every group has a session id", "[unit][camera_core][groups]")
{
    Publisher publisher;
    const GroupId head = publisher.add_group(make_group(4, 1, "head"));
    const GroupId wrist = publisher.add_group(make_group(4, 1, "wrist"));
    const std::uint64_t head_session = publisher.group_description(head).session_id;
    REQUIRE(head_session != 0);
    REQUIRE(publisher.group_description(wrist).session_id != 0);
    REQUIRE(publisher.group_description(wrist).session_id != head_session);

    // A restart of the camera is not a new session.
    publisher.restart_group(head);
    REQUIRE(publisher.group_description(head).session_id == head_session);

    // A backend that has an identity of its own for the run gives it.
    GroupDescription named = make_group(4, 1, "named");
    named.session_id = 42;
    REQUIRE(publisher.group_description(publisher.add_group(named)).session_id == 42);
}

TEST_CASE("a group with a slot-free callback hands out slots by claim only", "[unit][camera_core][slots]")
{
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(4, 1));
    REQUIRE(publisher.acquire_slot(group, 0) == 0u);
    publisher.abandon_slot(group, 0, 0);

    publisher.set_slot_free_callback(group, [](std::size_t, std::uint32_t) {});
    REQUIRE_THROWS_AS(publisher.acquire_slot(group, 0), std::logic_error);
    REQUIRE_NOTHROW(publisher.claim_slot(group, 0, 0));

    publisher.set_slot_free_callback(group, {});
    REQUIRE(publisher.acquire_slot(group, 0) == 1u);
}

TEST_CASE("a pool owned by the camera SDK never leaves the SDK short of its reserve", "[unit][camera_core][slots]")
{
    // The shape of a capture stack that is given its buffers up front and picks the one it
    // fills: ten per eye, of which the SDK needs four in hand to keep capturing.
    constexpr std::uint32_t BUFFERS = 10;
    constexpr std::uint32_t RESERVE = 4;
    Publisher publisher;
    const GroupId group = publisher.add_group(make_group(BUFFERS, 2), RESERVE);

    // What the SDK has to fill, per eye. A buffer goes back the moment nobody holds it.
    std::array<std::deque<std::uint32_t>, 2> with_sdk;
    for (std::deque<std::uint32_t>& buffers : with_sdk)
    {
        for (std::uint32_t buffer = 0; buffer < BUFFERS; ++buffer)
        {
            buffers.push_back(buffer);
        }
    }
    publisher.set_slot_free_callback(
        group, [&](std::size_t stream, std::uint32_t slot) { with_sdk[stream].push_back(slot); });

    // An encoder that reads every frame set and holds two at a time, and a viewer that wants
    // the newest. With the reserve they fit ten buffers; a recorder on top would not.
    RecordingSink encoder;
    RecordingSink viewer;
    RecordingSink recorder;
    const SubscribeResult encoding = open(publisher, encoder, group, { .window = 2, .order = DeliveryOrder::IN_ORDER });
    const SubscribeResult viewing = open(publisher, viewer, group, { .window = 1 });
    REQUIRE(encoding);
    REQUIRE(viewing);
    REQUIRE(publisher.request_newest(viewing.id, 0));
    REQUIRE_FALSE(publisher.subscribe(recorder, group, { .window = 2, .order = DeliveryOrder::IN_ORDER }));

    std::uint32_t state = 12345;
    const auto random = [&state]
    {
        state = state * 1664525u + 1013904223u;
        return state >> 16;
    };
    std::size_t encoder_released = 0;
    std::size_t viewer_released = 0;
    for (int frame = 0; frame < 5000; ++frame)
    {
        // The SDK fills one buffer per eye, and must find its reserve waiting.
        std::array<ImageInfo, 2> images;
        for (std::size_t eye = 0; eye < 2; ++eye)
        {
            REQUIRE(with_sdk[eye].size() >= RESERVE);
            images[eye].slot = with_sdk[eye].front();
            with_sdk[eye].pop_front();
            publisher.claim_slot(group, eye, images[eye].slot);
        }
        if (random() % 16 == 0)
        {
            // The eyes did not arrive as a pair: both are given up and go straight back.
            for (std::size_t eye = 0; eye < 2; ++eye)
            {
                publisher.abandon_slot(group, eye, images[eye].slot);
                with_sdk[eye].push_back(images[eye].slot);
            }
            continue;
        }
        publisher.publish(group, images, {});

        // The encoder is done with a frame set after a frame or two. The viewer keeps up,
        // except when it stalls for a while.
        while (encoder_released < encoder.events.size() && random() % 3 != 0)
        {
            publisher.release(encoding.id, encoder.events[encoder_released++].lease_id);
        }
        if (random() % 50 != 0)
        {
            while (viewer_released < viewer.events.size())
            {
                publisher.release(viewing.id, viewer.events[viewer_released++].lease_id);
            }
            publisher.request_newest(viewing.id, viewer.events.empty() ? 0 : viewer.events.back().sequence);
        }
    }

    // Both readers were served throughout, and what either did not get is accounted for.
    REQUIRE(publisher.stats(group).published > 4000);
    REQUIRE(viewer.events.size() > 1000);
    REQUIRE(encoder.events.size() > 1000);
    for (std::size_t i = 1; i < encoder.events.size(); ++i)
    {
        REQUIRE(encoder.events[i].sequence == encoder.events[i - 1].sequence + 1 + encoder.events[i].skipped);
    }
}

TEST_CASE("one camera does not wait for another camera's readers", "[unit][camera_core][threads]")
{
    using namespace std::chrono_literals;

    // A reader whose sink does not come back. A sink must not do this; it stands in for the
    // worst a reader of one camera can do to another camera.
    struct StuckSink : EventSink
    {
        bool send_event(SubscriptionId, const FrameEvent&) noexcept override
        {
            entered = true;
            const auto deadline = std::chrono::steady_clock::now() + 5s;
            while (!let_go && std::chrono::steady_clock::now() < deadline)
            {
                std::this_thread::sleep_for(1ms);
            }
            return true;
        }

        std::atomic<bool> entered{ false };
        std::atomic<bool> let_go{ false };
    };

    Publisher publisher;
    const GroupId head = publisher.add_group(make_group(4, 1, "head"));
    const GroupId wrist = publisher.add_group(make_group(4, 1, "wrist"));
    StuckSink stuck;
    RecordingSink wrist_reader;
    const SubscribeResult head_reading = open(publisher, stuck, head);
    REQUIRE(head_reading);
    REQUIRE(publisher.request_newest(head_reading.id, 0));
    const SubscribeResult wrist_reading = open(publisher, wrist_reader, wrist);
    REQUIRE(wrist_reading);
    REQUIRE(publisher.request_newest(wrist_reading.id, 0));

    std::thread head_capture(
        [&]
        {
            ImageInfo image;
            image.slot = publisher.acquire_slot(head, 0).value_or(0);
            publisher.publish(head, { &image, 1 }, {});
        });
    while (!stuck.entered)
    {
        std::this_thread::sleep_for(1ms);
    }

    // The head camera is inside its reader's sink. The wrist camera publishes all the same,
    // and readers can still come, go and ask what there is.
    const auto before = std::chrono::steady_clock::now();
    REQUIRE(publish_next(publisher, wrist) == 1);
    REQUIRE(wrist_reader.events.size() == 1);
    publisher.release(wrist_reading.id, wrist_reader.events[0].lease_id);
    REQUIRE(publisher.find_group("wrist") == wrist);
    REQUIRE(publisher.stats(wrist).events_sent == 1);
    RecordingSink late;
    const SubscribeResult joined = publisher.subscribe(late, wrist, {});
    REQUIRE(joined);
    publisher.unsubscribe(joined.id);
    REQUIRE(std::chrono::steady_clock::now() - before < 2s);
    REQUIRE_FALSE(stuck.let_go);

    stuck.let_go = true;
    head_capture.join();
    REQUIRE(publisher.stats(head).events_sent == 1);
}

TEST_CASE("capture and a reader on different threads keep every frame set in order", "[unit][camera_core][threads]")
{
    // A reader as the endpoint will drive it: events arrive on the capture thread, the
    // reader's own thread takes them and releases.
    struct QueueSink : EventSink
    {
        bool send_event(SubscriptionId, const FrameEvent& event) noexcept override
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
    const SubscribeResult reader = open(publisher, sink, group, { .window = 4 });
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
    REQUIRE(publisher.stats(group).published == FRAME_SETS);
}
