// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <camera_wire/messages.hpp>
#include <camera_wire_schema/camera_wire_generated.h>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

using namespace core::camera;
namespace wire = core::camera::wire;

namespace
{

GroupDescription stereo_group()
{
    GroupDescription group;
    group.id = "head";
    for (const char* eye : { "left", "right" })
    {
        StreamDescription stream;
        stream.id = std::string("head/") + eye;
        stream.pixel_format = PixelFormat::NV12;
        stream.width = 1920;
        stream.height = 1080;
        stream.plane_count = 2;
        stream.planes[0] = { .offset_bytes = 0, .stride_bytes = 2048 };
        stream.planes[1] = { .offset_bytes = 2048 * 1080, .stride_bytes = 2048 };
        stream.slot_bytes = 2048 * 1620;
        stream.slot_count = 8;
        stream.frame_rate = 59.94;
        group.streams.push_back(stream);
    }
    return group;
}

GroupDescription encoded_group()
{
    GroupDescription group;
    group.id = "head.h265";
    StreamDescription stream;
    stream.id = "head/left.h265";
    stream.codec = Codec::H265;
    stream.width = 1920;
    stream.height = 1080;
    stream.slot_bytes = 1 << 20;
    stream.slot_count = 16;
    stream.frame_rate = 60.0;
    group.streams.push_back(stream);
    return group;
}

void require_same(const GroupDescription& actual, const GroupDescription& expected)
{
    REQUIRE(actual.id == expected.id);
    REQUIRE(actual.streams.size() == expected.streams.size());
    for (std::size_t i = 0; i < expected.streams.size(); ++i)
    {
        const StreamDescription& a = actual.streams[i];
        const StreamDescription& e = expected.streams[i];
        REQUIRE(a.id == e.id);
        REQUIRE(a.codec == e.codec);
        REQUIRE(a.memory == e.memory);
        REQUIRE(a.width == e.width);
        REQUIRE(a.height == e.height);
        REQUIRE(a.plane_count == e.plane_count);
        for (std::uint32_t plane = 0; plane < e.plane_count; ++plane)
        {
            REQUIRE(a.planes[plane].offset_bytes == e.planes[plane].offset_bytes);
            REQUIRE(a.planes[plane].stride_bytes == e.planes[plane].stride_bytes);
        }
        REQUIRE(a.slot_bytes == e.slot_bytes);
        REQUIRE(a.slot_count == e.slot_count);
        REQUIRE(a.frame_rate == e.frame_rate);
        if (!e.is_encoded())
        {
            REQUIRE(a.pixel_format == e.pixel_format);
        }
    }
}

// What the other process receives: a copy of the bytes, decoded.
template <typename Message>
Message round_trip(WireEncoder& encoder, const Message& message)
{
    const std::span<const std::byte> bytes = encoder.encode(message);
    const std::vector<std::byte> received(bytes.begin(), bytes.end());
    WireMessage decoded = decode_wire_message(received);
    REQUIRE(std::holds_alternative<Message>(decoded));
    return std::get<Message>(std::move(decoded));
}

std::span<const std::byte> bytes_of(const flatbuffers::FlatBufferBuilder& builder)
{
    return { reinterpret_cast<const std::byte*>(builder.GetBufferPointer()), builder.GetSize() };
}

} // namespace

TEST_CASE("the handshake messages survive the wire", "[unit][camera_wire]")
{
    WireEncoder encoder;

    const HelloMessage hello = round_trip(encoder, HelloMessage{ .protocol_version = 7, .reader_name = "viewer" });
    REQUIRE(hello.protocol_version == 7);
    REQUIRE(hello.reader_name == "viewer");

    WelcomeMessage welcome;
    welcome.groups = { stereo_group(), encoded_group() };
    const WelcomeMessage received = round_trip(encoder, welcome);
    REQUIRE(received.protocol_version == WIRE_PROTOCOL_VERSION);
    REQUIRE(received.error.empty());
    REQUIRE(received.groups.size() == 2);
    require_same(received.groups[0], stereo_group());
    require_same(received.groups[1], encoded_group());

    const WelcomeMessage refused = round_trip(encoder, WelcomeMessage{ .error = "protocol version 7 is not supported" });
    REQUIRE(refused.error == "protocol version 7 is not supported");
    REQUIRE(refused.groups.empty());
}

TEST_CASE("a subscription and its answer survive the wire", "[unit][camera_wire]")
{
    WireEncoder encoder;

    const SubscribeMessage subscribe =
        round_trip(encoder, SubscribeMessage{ "head", { .window = 4, .order = DeliveryOrder::IN_ORDER } });
    REQUIRE(subscribe.group == "head");
    REQUIRE(subscribe.options.window == 4);
    REQUIRE(subscribe.options.order == DeliveryOrder::IN_ORDER);

    // What a reader that sets nothing asks for.
    const SubscribeMessage defaults = round_trip(encoder, SubscribeMessage{ "head", {} });
    REQUIRE(defaults.options.window == 1);
    REQUIRE(defaults.options.order == DeliveryOrder::DEFAULT);

    const SubscribedMessage accepted = round_trip(encoder, SubscribedMessage{ "", stereo_group() });
    REQUIRE(accepted.error.empty());
    require_same(accepted.group, stereo_group());

    const SubscribedMessage refused = round_trip(encoder, SubscribedMessage{ "does not fit the pool", {} });
    REQUIRE(refused.error == "does not fit the pool");
    REQUIRE(refused.group.streams.empty());
}

TEST_CASE("a frame set and its release survive the wire", "[unit][camera_wire]")
{
    WireEncoder encoder;

    FrameEvent event;
    event.lease_id = 0x1122334455667788;
    event.sequence = 123456789012;
    event.generation = 3;
    event.skipped = 2;
    event.image_count = 2;
    event.timeline = { .capture_ns = 1000, .sdk_ready_ns = 2000, .published_ns = 3000, .sent_ns = 4000 };
    event.images[0] = {
        .slot = 5, .flags = IMAGE_FLAG_SDK_WARNING, .bytes_used = 0, .device_frame_id = 77, .device_time_ns = 99
    };
    event.images[1] = { .slot = 6, .flags = IMAGE_FLAG_KEYFRAME, .bytes_used = 4096, .device_frame_id = 78 };

    const FrameEvent received = round_trip(encoder, event);
    REQUIRE(received.lease_id == event.lease_id);
    REQUIRE(received.sequence == event.sequence);
    REQUIRE(received.generation == 3);
    REQUIRE(received.skipped == 2);
    REQUIRE(received.image_count == 2);
    REQUIRE(received.timeline.capture_ns == 1000);
    REQUIRE(received.timeline.sdk_ready_ns == 2000);
    REQUIRE(received.timeline.published_ns == 3000);
    REQUIRE(received.timeline.sent_ns == 4000);
    REQUIRE(received.images[0].slot == 5);
    REQUIRE(received.images[0].flags == IMAGE_FLAG_SDK_WARNING);
    REQUIRE(received.images[0].device_frame_id == 77);
    REQUIRE(received.images[0].device_time_ns == 99);
    REQUIRE(received.images[1].slot == 6);
    REQUIRE(received.images[1].flags == IMAGE_FLAG_KEYFRAME);
    REQUIRE(received.images[1].bytes_used == 4096);
    REQUIRE(received.images[1].device_time_ns == TIME_UNKNOWN);

    // Times that are not known stay unknown.
    const FrameEvent bare = round_trip(encoder, FrameEvent{});
    REQUIRE(bare.image_count == 0);
    REQUIRE(bare.timeline.capture_ns == TIME_UNKNOWN);
    REQUIRE(bare.timeline.sent_ns == TIME_UNKNOWN);

    REQUIRE(round_trip(encoder, ReleaseMessage{ 42 }).lease_id == 42);
}

TEST_CASE("an encoder is reused from message to message", "[unit][camera_wire]")
{
    WireEncoder encoder;
    FrameEvent event;
    event.image_count = 2;
    for (std::uint64_t sequence = 1; sequence <= 100; ++sequence)
    {
        event.sequence = sequence;
        REQUIRE(round_trip(encoder, event).sequence == sequence);
    }
    // A small message after a large one is not padded to the large one's size.
    WelcomeMessage welcome;
    welcome.groups = { stereo_group(), encoded_group() };
    const std::size_t large = encoder.encode(welcome).size();
    REQUIRE(encoder.encode(ReleaseMessage{ 1 }).size() < large);
}

TEST_CASE("bytes that are not a message are rejected", "[unit][camera_wire]")
{
    REQUIRE_THROWS_AS(decode_wire_message({}), WireError);

    const std::vector<std::byte> garbage(64, std::byte{ 0xa5 });
    REQUIRE_THROWS_AS(decode_wire_message(garbage), WireError);

    WireEncoder encoder;
    WelcomeMessage welcome;
    welcome.groups = { stereo_group() };
    const std::span<const std::byte> bytes = encoder.encode(welcome);
    const std::vector<std::byte> whole(bytes.begin(), bytes.end());
    REQUIRE_THROWS_AS(decode_wire_message({ whole.data(), whole.size() / 2 }), WireError);

    // No cut of a message may be read out of bounds; each is either refused or read in full.
    for (std::size_t length = 0; length < whole.size(); ++length)
    {
        const std::vector<std::byte> cut(whole.begin(), whole.begin() + static_cast<std::ptrdiff_t>(length));
        try
        {
            (void)decode_wire_message(cut);
        }
        catch (const WireError&)
        {
        }
    }
}

TEST_CASE("a message that does not fit this build's limits is rejected", "[unit][camera_wire]")
{
    WireEncoder encoder;

    GroupDescription crowded = stereo_group();
    crowded.streams.resize(MAX_STREAMS_PER_GROUP + 1, crowded.streams.front());
    const std::span<const std::byte> bytes = encoder.encode(SubscribedMessage{ "", crowded });
    REQUIRE_THROWS_AS(decode_wire_message(bytes), WireError);

    // A pixel format added by a newer plugin cannot be read, so the group is refused as a whole.
    GroupDescription newer = stereo_group();
    newer.streams[1].pixel_format = static_cast<PixelFormat>(99);
    REQUIRE_THROWS_AS(decode_wire_message(encoder.encode(SubscribedMessage{ "", newer })), WireError);

    flatbuffers::FlatBufferBuilder builder;

    // More planes than an image can have.
    const std::vector<flatbuffers::Offset<wire::PlaneLayout>> planes(
        MAX_PLANES_PER_IMAGE + 1, wire::CreatePlaneLayout(builder, 0, 64));
    const auto stream =
        wire::CreateStream(builder, builder.CreateString("s"), wire::Codec_None, wire::PixelFormat_Gray8,
                           wire::MemoryKind_Host, 64, 64, builder.CreateVector(planes), 64 * 64, 4, 30.0);
    const auto group = wire::CreateGroup(builder, builder.CreateString("g"), builder.CreateVector(&stream, 1));
    builder.Finish(wire::CreateMessage(builder, wire::Body_Subscribed, wire::CreateSubscribed(builder, 0, group).Union()));
    REQUIRE_THROWS_AS(decode_wire_message(bytes_of(builder)), WireError);

    // More images than a group has streams.
    builder.Clear();
    const std::vector<flatbuffers::Offset<wire::Image>> images(MAX_STREAMS_PER_GROUP + 1, wire::CreateImage(builder));
    builder.Finish(wire::CreateMessage(
        builder, wire::Body_Frame, wire::CreateFrame(builder, 1, 1, 0, 0, 0, builder.CreateVector(images)).Union()));
    REQUIRE_THROWS_AS(decode_wire_message(bytes_of(builder)), WireError);

    // A body type with no body behind it.
    builder.Clear();
    builder.Finish(wire::CreateMessage(builder, wire::Body_Release, 0));
    REQUIRE_THROWS_AS(decode_wire_message(bytes_of(builder)), WireError);

    // The largest message a receive buffer has to hold.
    REQUIRE_THROWS_AS(
        encoder.encode(HelloMessage{ .reader_name = std::string(MAX_WIRE_MESSAGE_BYTES, 'x') }), std::length_error);
}

TEST_CASE("a message from a newer peer is ignored, not an error", "[unit][camera_wire]")
{
    flatbuffers::FlatBufferBuilder builder;

    // A union member this build does not have.
    const auto body = wire::CreateRelease(builder, 1);
    builder.Finish(wire::CreateMessage(builder, static_cast<wire::Body>(wire::Body_MAX + 1), body.Union()));
    REQUIRE(std::holds_alternative<std::monostate>(decode_wire_message(bytes_of(builder))));

    // No body at all.
    builder.Clear();
    builder.Finish(wire::CreateMessage(builder));
    REQUIRE(std::holds_alternative<std::monostate>(decode_wire_message(bytes_of(builder))));
}
