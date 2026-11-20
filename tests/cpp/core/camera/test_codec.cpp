// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <camera/messages.hpp>
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
    group.session_id = 0x0123456789abcdef;
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

GroupDescription gray_group()
{
    GroupDescription group;
    group.id = "wrist";
    StreamDescription stream;
    stream.id = "wrist/0";
    stream.pixel_format = PixelFormat::GRAY8;
    stream.width = 640;
    stream.height = 480;
    stream.plane_count = 1;
    stream.planes[0] = { .offset_bytes = 0, .stride_bytes = 640 };
    stream.slot_bytes = 640 * 480;
    stream.slot_count = 16;
    stream.frame_rate = 60.0;
    group.streams.push_back(stream);
    return group;
}

void require_same(const GroupDescription& actual, const GroupDescription& expected)
{
    REQUIRE(actual.id == expected.id);
    REQUIRE(actual.session_id == expected.session_id);
    REQUIRE(actual.streams.size() == expected.streams.size());
    for (std::size_t i = 0; i < expected.streams.size(); ++i)
    {
        const StreamDescription& a = actual.streams[i];
        const StreamDescription& e = expected.streams[i];
        REQUIRE(a.id == e.id);
        REQUIRE(a.pixel_format == e.pixel_format);
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
        REQUIRE(a.cuda_device == e.cuda_device);
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

// Finishes a message whose body the schema's own helpers would not build: a body type this
// build does not have, a missing body, or the mark that lets a receiver ignore it.
void finish_message(flatbuffers::FlatBufferBuilder& builder,
                    std::uint32_t body_type,
                    flatbuffers::Offset<void> body,
                    bool skippable = false)
{
    builder.Finish(core::CreateCameraMessage(builder, static_cast<core::CameraMessageBody>(body_type), body, skippable));
}

// A stream as a plugin describes it, with the kinds given as numbers so that ones this build
// does not have can be written.
flatbuffers::Offset<core::CameraStream> build_stream(flatbuffers::FlatBufferBuilder& builder,
                                                     const std::string& id,
                                                     std::uint32_t pixel_format = core::CameraPixelFormat_Gray8,
                                                     std::uint32_t memory = core::CameraMemoryKind_Host,
                                                     std::size_t plane_count = 1)
{
    const std::vector<core::CameraPlaneLayout> planes(plane_count, core::CameraPlaneLayout(0, 64));
    return core::CreateCameraStream(
        builder, builder.CreateString(id), static_cast<core::CameraPixelFormat>(pixel_format),
        static_cast<core::CameraMemoryKind>(memory), 64, 64, builder.CreateVectorOfStructs(planes), 64 * 64, 4, 30.0);
}

flatbuffers::Offset<core::CameraGroup> build_group(flatbuffers::FlatBufferBuilder& builder,
                                                   const std::string& id,
                                                   const std::vector<flatbuffers::Offset<core::CameraStream>>& streams)
{
    return core::CreateCameraGroup(builder, builder.CreateString(id), builder.CreateVector(streams), 9);
}

std::span<const std::byte> finish_response(flatbuffers::FlatBufferBuilder& builder,
                                           const std::vector<flatbuffers::Offset<core::CameraGroup>>& groups)
{
    const auto response = core::CreateCameraHandshakeResponse(builder, PROTOCOL_VERSION, 0, builder.CreateVector(groups));
    finish_message(builder, core::CameraMessageBody_CameraHandshakeResponse, response.Union());
    return bytes_of(builder);
}

HandshakeResponse response_with(std::initializer_list<GroupDescription> groups)
{
    HandshakeResponse response;
    for (const GroupDescription& group : groups)
    {
        response.groups.push_back({ group, {} });
    }
    return response;
}

} // namespace

TEST_CASE("a handshake survives the wire in both directions", "[unit][camera]")
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
    REQUIRE(asked.unsupported.empty());

    // One that sets nothing asks what is published, and would read with the default window.
    const HandshakeRequest plain = round_trip(encoder, HandshakeRequest{});
    REQUIRE(plain.protocol_version == PROTOCOL_VERSION);
    REQUIRE(plain.group.empty());
    REQUIRE(plain.options.window == 1);
    REQUIRE(plain.options.order == DeliveryOrder::DEFAULT);

    const HandshakeResponse answered = round_trip(encoder, response_with({ stereo_group(), gray_group() }));
    REQUIRE(answered.protocol_version == PROTOCOL_VERSION);
    REQUIRE(answered.error.empty());
    REQUIRE(answered.groups.size() == 2);
    REQUIRE(answered.groups[0].unsupported.empty());
    REQUIRE(answered.groups[1].unsupported.empty());
    require_same(answered.groups[0].group, stereo_group());
    require_same(answered.groups[1].group, gray_group());

    const HandshakeResponse refused = round_trip(encoder, HandshakeResponse{ .error = "does not fit the pool" });
    REQUIRE(refused.error == "does not fit the pool");
    REQUIRE(refused.groups.empty());
}

TEST_CASE("a frame set, its release and a request for one survive the wire", "[unit][camera]")
{
    MessageEncoder encoder;

    FrameEvent event;
    event.lease_id = 0x1122334455667788;
    event.sequence = 123456789012;
    event.generation = 3;
    event.skipped = 2;
    event.image_count = 2;
    event.timeline = { .capture_ns = 1000, .sdk_ready_ns = 2000, .published_ns = 3000, .sent_ns = 4000 };
    event.images[0] = { .slot = 5, .flags = IMAGE_FLAG_SDK_WARNING, .device_frame_id = 77, .device_time_ns = 99 };
    // A flag bit this build has no name for is passed on as it is.
    event.images[1] = { .slot = 6, .flags = 1u << 9, .device_frame_id = 78 };

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
    REQUIRE(received.images[1].flags == (1u << 9));
    REQUIRE(received.images[1].device_frame_id == 78);
    REQUIRE(received.images[1].device_time_ns == TIME_UNKNOWN);

    // Times that are not known stay unknown.
    const FrameEvent bare = round_trip(encoder, FrameEvent{});
    REQUIRE(bare.image_count == 0);
    REQUIRE(bare.timeline.capture_ns == TIME_UNKNOWN);
    REQUIRE(bare.timeline.sent_ns == TIME_UNKNOWN);

    REQUIRE(round_trip(encoder, ReleaseMessage{ 42 }).lease_id == 42);
    REQUIRE(round_trip(encoder, AcquireMessage{ 123456789012 }).after_sequence == 123456789012);
    REQUIRE(round_trip(encoder, AcquireMessage{}).after_sequence == 0);
}

TEST_CASE("the numbers of the protocol are the ones it was published with", "[unit][camera]")
{
    // A message's number, a flag's bit and a kind's value are read by builds that will never see
    // this source. Changing one here is changing the protocol.
    REQUIRE(core::CameraMessageBody_CameraHandshakeRequest == 1);
    REQUIRE(core::CameraMessageBody_CameraHandshakeResponse == 2);
    REQUIRE(core::CameraMessageBody_CameraFrameSet == 3);
    REQUIRE(core::CameraMessageBody_CameraRelease == 4);
    REQUIRE(core::CameraMessageBody_CameraAcquire == 5);
    REQUIRE(static_cast<std::uint32_t>(core::CameraFrameFlag_SdkWarning) == IMAGE_FLAG_SDK_WARNING);
    REQUIRE(IMAGE_FLAG_SDK_WARNING == 1u);
    REQUIRE(core::CameraPixelFormat_Bgra8 == 1);
    REQUIRE(core::CameraPixelFormat_Gray8 == 4);
    REQUIRE(core::CameraMemoryKind_Host == 1);
    REQUIRE(core::CameraDeliveryOrder_InOrder == 2);
    REQUIRE(sizeof(core::CameraPlaneLayout) == 8);
    REQUIRE(PROTOCOL_VERSION >= OLDEST_PROTOCOL_VERSION);
}

TEST_CASE("an encoder is reused from message to message", "[unit][camera]")
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
    const std::size_t large = encoder.encode(response_with({ stereo_group(), gray_group() })).size();
    REQUIRE(encoder.encode(ReleaseMessage{ 1 }).size() < large);
}

TEST_CASE("bytes that are not a message are rejected", "[unit][camera]")
{
    REQUIRE_THROWS_AS(decode_message({}), ProtocolError);

    const std::vector<std::byte> garbage(64, std::byte{ 0xa5 });
    REQUIRE_THROWS_AS(decode_message(garbage), ProtocolError);

    MessageEncoder encoder;
    const std::span<const std::byte> bytes = encoder.encode(response_with({ stereo_group() }));
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

TEST_CASE("the encoder refuses counts it cannot write, and cuts nothing short", "[unit][camera]")
{
    MessageEncoder encoder;

    GroupDescription crowded = stereo_group();
    crowded.streams.resize(MAX_STREAMS_PER_GROUP + 1, crowded.streams.front());
    REQUIRE_THROWS_AS(encoder.encode(response_with({ crowded })), std::invalid_argument);

    GroupDescription planes = stereo_group();
    planes.streams[0].plane_count = MAX_PLANES_PER_IMAGE + 1;
    REQUIRE_THROWS_AS(encoder.encode(response_with({ planes })), std::invalid_argument);

    FrameEvent event;
    event.image_count = MAX_STREAMS_PER_GROUP + 1;
    REQUIRE_THROWS_AS(encoder.encode(event), std::invalid_argument);

    // The largest message a receive buffer has to hold.
    REQUIRE_THROWS_AS(
        encoder.encode(HandshakeRequest{ .reader_name = std::string(MAX_MESSAGE_BYTES, 'x') }), std::length_error);

    // The encoder is still usable after refusing a message.
    REQUIRE(round_trip(encoder, ReleaseMessage{ 7 }).lease_id == 7);
}

TEST_CASE("a message that is broken is rejected", "[unit][camera]")
{
    flatbuffers::FlatBufferBuilder builder;

    // More images than a group has streams.
    const std::vector<flatbuffers::Offset<core::CameraFrameRef>> frames(
        MAX_STREAMS_PER_GROUP + 1, core::CreateCameraFrameRef(builder));
    const auto frame_set = core::CreateCameraFrameSet(builder, 1, 1, 0, 0, -1, -1, -1, -1, builder.CreateVector(frames));
    finish_message(builder, core::CameraMessageBody_CameraFrameSet, frame_set.Union());
    REQUIRE_THROWS_AS(decode_message(bytes_of(builder)), ProtocolError);

    // A body type with no body behind it.
    builder.Clear();
    finish_message(builder, core::CameraMessageBody_CameraRelease, 0);
    REQUIRE_THROWS_AS(decode_message(bytes_of(builder)), ProtocolError);
}

TEST_CASE("a description is decoded as it was sent; whether it makes sense is checked where it is used", "[unit][camera]")
{
    // The structure of these messages is fine. What they describe is not: rows longer than the
    // slot, a plane count that does not match the format, a slot size no pool can have.
    MessageEncoder encoder;
    GroupDescription rows = stereo_group();
    rows.streams[0].planes[0].stride_bytes = 16;
    GroupDescription planes = stereo_group();
    planes.streams[1].plane_count = 1;
    GroupDescription slots = stereo_group();
    slots.streams[0].slot_bytes = MAX_SLOT_BYTES + 1;
    GroupDescription wraps = stereo_group();
    wraps.streams[0].width = 0xFFFFFFFF;
    wraps.streams[0].height = 0xFFFFFFFF;
    wraps.streams[0].planes[0] = { .offset_bytes = 0xFFFFFFFF, .stride_bytes = 0xFFFFFFFF };

    for (const GroupDescription& broken : { rows, planes, slots, wraps })
    {
        const HandshakeResponse response = round_trip(encoder, response_with({ broken }));
        REQUIRE(response.groups.size() == 1);
        REQUIRE(response.groups[0].unsupported.empty());
        REQUIRE_FALSE(problem_with(response.groups[0].group).empty());
    }
    REQUIRE(problem_with(round_trip(encoder, response_with({ stereo_group() })).groups[0].group).empty());
}

TEST_CASE("a stream in CUDA memory is described with its device", "[unit][camera]")
{
    GroupDescription group = stereo_group();
    for (StreamDescription& stream : group.streams)
    {
        stream.memory = MemoryKind::CUDA;
        stream.cuda_device = { 0xb4, 0x37, 0x72, 0x62, 0xe0, 0x0d, 0x16, 0x89,
                               0xcd, 0xf9, 0xd9, 0xc0, 0x2b, 0x8a, 0xee, 0x17 };
    }
    MessageEncoder encoder;
    const HandshakeResponse response = round_trip(encoder, response_with({ group, stereo_group() }));
    REQUIRE(response.groups.size() == 2);
    REQUIRE(response.groups[0].unsupported.empty());
    require_same(response.groups[0].group, group);
    REQUIRE(problem_with(response.groups[0].group).empty());
    // A stream in host memory names no device.
    require_same(response.groups[1].group, stereo_group());

    // A device that is named in another form than this build's, or not at all, is not guessed at.
    flatbuffers::FlatBufferBuilder builder;
    const auto cuda_stream = [&builder](const std::vector<std::uint8_t>* device)
    {
        const std::vector<core::CameraPlaneLayout> planes(1, core::CameraPlaneLayout(0, 64));
        return core::CreateCameraStream(builder, builder.CreateString("head/0"), core::CameraPixelFormat_Gray8,
                                        core::CameraMemoryKind_Cuda, 64, 64, builder.CreateVectorOfStructs(planes),
                                        64 * 64, 4, 30.0, device != nullptr ? builder.CreateVector(*device) : 0);
    };
    const std::vector<std::uint8_t> short_name = { 1, 2, 3, 4 };
    for (const std::vector<std::uint8_t>* device : { &short_name, static_cast<const std::vector<std::uint8_t>*>(nullptr) })
    {
        builder.Clear();
        ProtocolMessage decoded =
            decode_message(finish_response(builder, { build_group(builder, "head", { cuda_stream(device) }) }));
        const std::vector<GroupEntry> groups = std::get<HandshakeResponse>(std::move(decoded)).groups;
        REQUIRE(groups.size() == 1);
        REQUIRE(groups[0].group.id == "head");
        REQUIRE(groups[0].group.streams.empty());
        REQUIRE(groups[0].unsupported.find("CUDA device") != std::string::npos);
    }
}

TEST_CASE("a group this build cannot read is listed as such, next to the ones it can", "[unit][camera]")
{
    flatbuffers::FlatBufferBuilder builder;
    const auto decode_groups = [&builder](const std::vector<flatbuffers::Offset<core::CameraGroup>>& groups)
    {
        ProtocolMessage decoded = decode_message(finish_response(builder, groups));
        REQUIRE(std::holds_alternative<HandshakeResponse>(decoded));
        return std::get<HandshakeResponse>(std::move(decoded)).groups;
    };

    // A newer plugin publishes one camera in a kind of memory added after this build and one
    // with a newer pixel format, next to a camera this build reads.
    builder.Clear();
    const std::vector<GroupEntry> groups = decode_groups({
        build_group(builder, "head", { build_stream(builder, "head/left"), build_stream(builder, "head/right") }),
        build_group(builder, "wrist",
                    { build_stream(builder, "wrist/0", core::CameraPixelFormat_Gray8, core::CameraMemoryKind_MAX + 1) }),
        build_group(builder, "hip", { build_stream(builder, "hip/0", core::CameraPixelFormat_MAX + 1) }),
    });
    REQUIRE(groups.size() == 3);
    REQUIRE(groups[0].unsupported.empty());
    REQUIRE(groups[0].group.streams.size() == 2);
    REQUIRE(groups[0].group.session_id == 9);
    REQUIRE(problem_with(groups[0].group).empty());
    REQUIRE(groups[1].group.id == "wrist");
    REQUIRE(groups[1].group.streams.empty());
    REQUIRE(groups[1].unsupported.find("kind of memory") != std::string::npos);
    REQUIRE(groups[2].unsupported.find("pixel format") != std::string::npos);

    // More planes or more streams than this build can describe.
    builder.Clear();
    REQUIRE_FALSE(
        decode_groups({ build_group(builder, "planar",
                                    { build_stream(builder, "planar/0", core::CameraPixelFormat_Gray8,
                                                   core::CameraMemoryKind_Host, MAX_PLANES_PER_IMAGE + 1) }) })[0]
            .unsupported.empty());
    builder.Clear();
    std::vector<flatbuffers::Offset<core::CameraStream>> many;
    for (std::size_t i = 0; i <= MAX_STREAMS_PER_GROUP; ++i)
    {
        many.push_back(build_stream(builder, "rig/" + std::to_string(i)));
    }
    const std::vector<GroupEntry> crowded = decode_groups({ build_group(builder, "rig", many) });
    REQUIRE(crowded[0].group.id == "rig");
    REQUIRE(crowded[0].unsupported.find("more streams") != std::string::npos);

    // A reader that asks for a delivery order added later is told so, not dropped unanswered.
    MessageEncoder encoder;
    const HandshakeRequest newer =
        round_trip(encoder, HandshakeRequest{ .group = "head", .options = { .order = static_cast<DeliveryOrder>(9) } });
    REQUIRE(newer.group == "head");
    REQUIRE(newer.unsupported.find("delivery order") != std::string::npos);
}

TEST_CASE("a message type this build does not know is ignored only when its sender says it may be", "[unit][camera]")
{
    flatbuffers::FlatBufferBuilder builder;
    const std::uint32_t unknown = core::CameraMessageBody_MAX + 1;

    // Something that only informs, such as a status a newer plugin sends: ignored.
    finish_message(builder, unknown, core::CreateCameraRelease(builder, 1).Union(), true);
    REQUIRE(std::holds_alternative<std::monostate>(decode_message(bytes_of(builder))));

    // An operation the peer counts on: carrying on as if it had not been sent would go wrong
    // quietly, so it is an error.
    builder.Clear();
    finish_message(builder, unknown, core::CreateCameraRelease(builder, 1).Union(), false);
    REQUIRE_THROWS_AS(decode_message(bytes_of(builder)), ProtocolError);

    // No body at all, marked or not.
    builder.Clear();
    builder.Finish(core::CreateCameraMessage(builder));
    REQUIRE_THROWS_AS(decode_message(bytes_of(builder)), ProtocolError);
    builder.Clear();
    finish_message(builder, core::CameraMessageBody_NONE, 0, true);
    REQUIRE(std::holds_alternative<std::monostate>(decode_message(bytes_of(builder))));

    // The mark changes nothing for a message this build does know.
    builder.Clear();
    finish_message(builder, core::CameraMessageBody_CameraRelease, core::CreateCameraRelease(builder, 5).Union(), true);
    const ProtocolMessage known = decode_message(bytes_of(builder));
    REQUIRE(std::get<ReleaseMessage>(known).lease_id == 5);
}

TEST_CASE("fields added by a newer peer are passed over, and fields an older peer lacks take their defaults",
          "[unit][camera]")
{
    flatbuffers::FlatBufferBuilder builder;
    // A table's fields sit two bytes apart in its vtable, so the slot after a table's last field
    // is where a newer schema's first appended field goes.
    constexpr flatbuffers::voffset_t NEXT = 2;

    // Newer reader: a handshake request with a field appended after `order`.
    {
        const auto name = builder.CreateString("newer");
        const auto group = builder.CreateString("head");
        const auto start = builder.StartTable();
        builder.AddElement<std::uint32_t>(core::CameraHandshakeRequest::VT_PROTOCOL_VERSION, PROTOCOL_VERSION + 1, 0);
        builder.AddOffset(core::CameraHandshakeRequest::VT_READER_NAME, name);
        builder.AddOffset(core::CameraHandshakeRequest::VT_GROUP_ID, group);
        builder.AddElement<std::uint32_t>(core::CameraHandshakeRequest::VT_WINDOW, 2, 1);
        builder.AddElement<std::uint32_t>(core::CameraHandshakeRequest::VT_ORDER, core::CameraDeliveryOrder_Newest, 0);
        builder.AddElement<std::uint64_t>(core::CameraHandshakeRequest::VT_ORDER + NEXT, 0xfeedULL, 0);
        const flatbuffers::Offset<void> request(builder.EndTable(start));
        finish_message(builder, core::CameraMessageBody_CameraHandshakeRequest, request);
        const ProtocolMessage decoded = decode_message(bytes_of(builder));
        const HandshakeRequest& asked = std::get<HandshakeRequest>(decoded);
        REQUIRE(asked.protocol_version == PROTOCOL_VERSION + 1);
        REQUIRE(asked.reader_name == "newer");
        REQUIRE(asked.group == "head");
        REQUIRE(asked.options.window == 2);
        REQUIRE(asked.options.order == DeliveryOrder::NEWEST);
        REQUIRE(asked.unsupported.empty());
    }

    // Newer plugin: a frame set with a field appended after `frames`, and a frame reference
    // with one appended after `device_time_ns`.
    {
        builder.Clear();
        auto start = builder.StartTable();
        builder.AddElement<std::uint32_t>(core::CameraFrameRef::VT_SLOT_INDEX, 3, 0);
        builder.AddElement<std::int64_t>(core::CameraFrameRef::VT_DEVICE_TIME_NS + NEXT, 555, 0);
        const flatbuffers::Offset<core::CameraFrameRef> frame(builder.EndTable(start));
        const auto frames = builder.CreateVector(&frame, 1);
        start = builder.StartTable();
        builder.AddElement<std::uint64_t>(core::CameraFrameSet::VT_LEASE_ID, 11, 0);
        builder.AddElement<std::uint64_t>(core::CameraFrameSet::VT_SEQUENCE, 22, 0);
        builder.AddOffset(core::CameraFrameSet::VT_FRAMES, frames);
        builder.AddElement<std::uint32_t>(core::CameraFrameSet::VT_FRAMES + NEXT, 7, 0);
        const flatbuffers::Offset<void> frame_set(builder.EndTable(start));
        finish_message(builder, core::CameraMessageBody_CameraFrameSet, frame_set);
        const ProtocolMessage decoded = decode_message(bytes_of(builder));
        const FrameEvent& event = std::get<FrameEvent>(decoded);
        REQUIRE(event.lease_id == 11);
        REQUIRE(event.sequence == 22);
        REQUIRE(event.image_count == 1);
        REQUIRE(event.images[0].slot == 3);
        REQUIRE(event.images[0].device_time_ns == TIME_UNKNOWN);
        REQUIRE(event.timeline.capture_ns == TIME_UNKNOWN);
    }

    // Older peer: a group written before `session_id` existed, in a message written before
    // `skippable` did. What is missing reads as its default.
    {
        builder.Clear();
        const auto stream = build_stream(builder, "head/left");
        const auto id = builder.CreateString("head");
        const auto streams = builder.CreateVector(&stream, 1);
        auto start = builder.StartTable();
        builder.AddOffset(core::CameraGroup::VT_ID, id);
        builder.AddOffset(core::CameraGroup::VT_STREAMS, streams);
        const flatbuffers::Offset<core::CameraGroup> group(builder.EndTable(start));
        const auto response =
            core::CreateCameraHandshakeResponse(builder, PROTOCOL_VERSION, 0, builder.CreateVector(&group, 1));
        start = builder.StartTable();
        builder.AddElement<std::uint8_t>(
            core::CameraMessage::VT_BODY_TYPE, core::CameraMessageBody_CameraHandshakeResponse, 0);
        builder.AddOffset(core::CameraMessage::VT_BODY, response.Union());
        builder.Finish(flatbuffers::Offset<core::CameraMessage>(builder.EndTable(start)));
        const ProtocolMessage decoded = decode_message(bytes_of(builder));
        const HandshakeResponse& answered = std::get<HandshakeResponse>(decoded);
        REQUIRE(answered.groups.size() == 1);
        REQUIRE(answered.groups[0].group.id == "head");
        REQUIRE(answered.groups[0].group.session_id == 0);
        REQUIRE(answered.groups[0].unsupported.empty());
    }
}
