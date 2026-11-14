// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <camera_protocol/messages.hpp>
#include <catch2/catch_test_macros.hpp>
#include <schema/camera_protocol_generated.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

using namespace core::camera;

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
Message round_trip(MessageEncoder& encoder, const Message& message)
{
    const std::span<const std::byte> bytes = encoder.encode(message);
    const std::vector<std::byte> received(bytes.begin(), bytes.end());
    ProtocolMessage decoded = decode_message(received);
    REQUIRE(std::holds_alternative<Message>(decoded));
    return std::get<Message>(std::move(decoded));
}

std::span<const std::byte> bytes_of(const flatbuffers::FlatBufferBuilder& builder)
{
    return { reinterpret_cast<const std::byte*>(builder.GetBufferPointer()), builder.GetSize() };
}

} // namespace

TEST_CASE("a handshake survives the wire in both directions", "[unit][camera_protocol]")
{
    MessageEncoder encoder;

    // A reader that wants a group says so in its first message.
    HandshakeRequest request;
    request.protocol_version = 7;
    request.reader_name = "viewer";
    request.group = "head";
    request.options = { .window = 4, .order = DeliveryOrder::IN_ORDER };
    const HandshakeRequest asked = round_trip(encoder, request);
    REQUIRE(asked.protocol_version == 7);
    REQUIRE(asked.reader_name == "viewer");
    REQUIRE(asked.group == "head");
    REQUIRE(asked.options.window == 4);
    REQUIRE(asked.options.order == DeliveryOrder::IN_ORDER);

    // One that sets nothing asks what is published, and would read with the default window.
    const HandshakeRequest plain = round_trip(encoder, HandshakeRequest{});
    REQUIRE(plain.protocol_version == PROTOCOL_VERSION);
    REQUIRE(plain.group.empty());
    REQUIRE(plain.options.window == 1);
    REQUIRE(plain.options.order == DeliveryOrder::DEFAULT);

    HandshakeResponse response;
    response.groups = { stereo_group(), encoded_group() };
    const HandshakeResponse answered = round_trip(encoder, response);
    REQUIRE(answered.protocol_version == PROTOCOL_VERSION);
    REQUIRE(answered.error.empty());
    REQUIRE(answered.groups.size() == 2);
    require_same(answered.groups[0], stereo_group());
    require_same(answered.groups[1], encoded_group());

    const HandshakeResponse refused = round_trip(encoder, HandshakeResponse{ .error = "does not fit the pool" });
    REQUIRE(refused.error == "does not fit the pool");
    REQUIRE(refused.groups.empty());
}

TEST_CASE("a frame set and its release survive the wire", "[unit][camera_protocol]")
{
    MessageEncoder encoder;

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

TEST_CASE("an encoder is reused from message to message", "[unit][camera_protocol]")
{
    MessageEncoder encoder;
    FrameEvent event;
    event.image_count = 2;
    for (std::uint64_t sequence = 1; sequence <= 100; ++sequence)
    {
        event.sequence = sequence;
        REQUIRE(round_trip(encoder, event).sequence == sequence);
    }
    // A small message after a large one is not padded to the large one's size.
    HandshakeResponse response;
    response.groups = { stereo_group(), encoded_group() };
    const std::size_t large = encoder.encode(response).size();
    REQUIRE(encoder.encode(ReleaseMessage{ 1 }).size() < large);
}

TEST_CASE("bytes that are not a message are rejected", "[unit][camera_protocol]")
{
    REQUIRE_THROWS_AS(decode_message({}), ProtocolError);

    const std::vector<std::byte> garbage(64, std::byte{ 0xa5 });
    REQUIRE_THROWS_AS(decode_message(garbage), ProtocolError);

    MessageEncoder encoder;
    HandshakeResponse response;
    response.groups = { stereo_group() };
    const std::span<const std::byte> bytes = encoder.encode(response);
    const std::vector<std::byte> whole(bytes.begin(), bytes.end());
    REQUIRE_THROWS_AS(decode_message({ whole.data(), whole.size() / 2 }), ProtocolError);

    // No cut of a message may be read out of bounds; each is either refused or read in full.
    for (std::size_t length = 0; length < whole.size(); ++length)
    {
        const std::vector<std::byte> cut(whole.begin(), whole.begin() + static_cast<std::ptrdiff_t>(length));
        try
        {
            (void)decode_message(cut);
        }
        catch (const ProtocolError&)
        {
        }
    }
}

TEST_CASE("a message that does not fit this build's limits is rejected", "[unit][camera_protocol]")
{
    MessageEncoder encoder;

    GroupDescription crowded = stereo_group();
    crowded.streams.resize(MAX_STREAMS_PER_GROUP + 1, crowded.streams.front());
    REQUIRE_THROWS_AS(decode_message(encoder.encode(HandshakeResponse{ .groups = { crowded } })), ProtocolError);

    // A pixel format added by a newer plugin cannot be read, so the group is refused as a whole.
    GroupDescription newer = stereo_group();
    newer.streams[1].pixel_format = static_cast<PixelFormat>(99);
    REQUIRE_THROWS_AS(decode_message(encoder.encode(HandshakeResponse{ .groups = { newer } })), ProtocolError);
    REQUIRE_THROWS_AS(
        decode_message(encoder.encode(HandshakeRequest{ .options = { .order = static_cast<DeliveryOrder>(9) } })),
        ProtocolError);

    flatbuffers::FlatBufferBuilder builder;
    const auto response_with = [&builder](
                                   const std::vector<std::uint32_t>& offsets, const std::vector<std::uint32_t>& strides)
    {
        builder.Clear();
        const auto stream =
            core::CreateCameraStream(builder, builder.CreateString("s"), core::CameraCodec_None,
                                     core::CameraPixelFormat_Gray8, core::CameraMemoryKind_Host, 64, 64,
                                     builder.CreateVector(offsets), builder.CreateVector(strides), 64 * 64, 4, 30.0);
        const auto group = core::CreateCameraGroup(builder, builder.CreateString("g"), builder.CreateVector(&stream, 1));
        const auto response =
            core::CreateCameraHandshakeResponse(builder, PROTOCOL_VERSION, 0, builder.CreateVector(&group, 1));
        builder.Finish(
            core::CreateCameraMessage(builder, core::CameraMessageBody_CameraHandshakeResponse, response.Union()));
        return bytes_of(builder);
    };
    // One plane is fine; more planes than an image can have, or offsets without strides, are not.
    REQUIRE_NOTHROW(decode_message(response_with({ 0 }, { 64 })));
    REQUIRE_THROWS_AS(decode_message(response_with({ 0, 0, 0 }, { 64, 64, 64 })), ProtocolError);
    REQUIRE_THROWS_AS(decode_message(response_with({ 0, 0 }, { 64 })), ProtocolError);

    // More images than a group has streams.
    builder.Clear();
    const std::vector<flatbuffers::Offset<core::CameraImage>> images(
        MAX_STREAMS_PER_GROUP + 1, core::CreateCameraImage(builder));
    const auto frame = core::CreateCameraFrame(builder, 1, 1, 0, 0, -1, -1, -1, -1, builder.CreateVector(images));
    builder.Finish(core::CreateCameraMessage(builder, core::CameraMessageBody_CameraFrame, frame.Union()));
    REQUIRE_THROWS_AS(decode_message(bytes_of(builder)), ProtocolError);

    // A body type with no body behind it.
    builder.Clear();
    builder.Finish(core::CreateCameraMessage(builder, core::CameraMessageBody_CameraRelease, 0));
    REQUIRE_THROWS_AS(decode_message(bytes_of(builder)), ProtocolError);

    // The largest message a receive buffer has to hold.
    REQUIRE_THROWS_AS(
        encoder.encode(HandshakeRequest{ .reader_name = std::string(MAX_MESSAGE_BYTES, 'x') }), std::length_error);
}

TEST_CASE("a message from a newer peer is ignored, not an error", "[unit][camera_protocol]")
{
    flatbuffers::FlatBufferBuilder builder;

    // A union member this build does not have.
    const auto body = core::CreateCameraRelease(builder, 1);
    builder.Finish(core::CreateCameraMessage(
        builder, static_cast<core::CameraMessageBody>(core::CameraMessageBody_MAX + 1), body.Union()));
    REQUIRE(std::holds_alternative<std::monostate>(decode_message(bytes_of(builder))));

    // No body at all.
    builder.Clear();
    builder.Finish(core::CreateCameraMessage(builder));
    REQUIRE(std::holds_alternative<std::monostate>(decode_message(bytes_of(builder))));
}
