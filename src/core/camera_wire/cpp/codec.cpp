// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/camera_wire/messages.hpp"

#include <camera_wire_schema/camera_wire_generated.h>

#include <algorithm>
#include <array>
#include <optional>

namespace core::camera
{

namespace
{

// The schema's enums carry the same numbers as the core's, so a value converts by a cast.
static_assert(static_cast<std::uint32_t>(PixelFormat::BGRA8) == wire::PixelFormat_Bgra8);
static_assert(static_cast<std::uint32_t>(PixelFormat::RGBA8) == wire::PixelFormat_Rgba8);
static_assert(static_cast<std::uint32_t>(PixelFormat::NV12) == wire::PixelFormat_Nv12);
static_assert(static_cast<std::uint32_t>(PixelFormat::GRAY8) == wire::PixelFormat_Gray8);
static_assert(static_cast<std::uint32_t>(Codec::NONE) == wire::Codec_None);
static_assert(static_cast<std::uint32_t>(Codec::H264) == wire::Codec_H264);
static_assert(static_cast<std::uint32_t>(Codec::H265) == wire::Codec_H265);
static_assert(static_cast<std::uint32_t>(MemoryKind::HOST) == wire::MemoryKind_Host);
static_assert(static_cast<std::uint32_t>(DeliveryOrder::DEFAULT) == wire::DeliveryOrder_Default);
static_assert(static_cast<std::uint32_t>(DeliveryOrder::NEWEST) == wire::DeliveryOrder_Newest);
static_assert(static_cast<std::uint32_t>(DeliveryOrder::IN_ORDER) == wire::DeliveryOrder_InOrder);

using Builder = flatbuffers::FlatBufferBuilder;

flatbuffers::Offset<wire::Group> build_group(Builder& builder, const GroupDescription& group)
{
    std::vector<flatbuffers::Offset<wire::Stream>> streams;
    streams.reserve(group.streams.size());
    for (const StreamDescription& stream : group.streams)
    {
        std::array<flatbuffers::Offset<wire::PlaneLayout>, MAX_PLANES_PER_IMAGE> planes;
        const std::size_t plane_count = std::min<std::size_t>(stream.plane_count, MAX_PLANES_PER_IMAGE);
        for (std::size_t i = 0; i < plane_count; ++i)
        {
            planes[i] = wire::CreatePlaneLayout(builder, stream.planes[i].offset_bytes, stream.planes[i].stride_bytes);
        }
        const auto id = builder.CreateString(stream.id);
        const auto plane_vector = builder.CreateVector(planes.data(), plane_count);
        streams.push_back(wire::CreateStream(builder, id, static_cast<wire::Codec>(stream.codec),
                                             static_cast<wire::PixelFormat>(stream.pixel_format),
                                             static_cast<wire::MemoryKind>(stream.memory), stream.width, stream.height,
                                             plane_vector, stream.slot_bytes, stream.slot_count, stream.frame_rate));
    }
    const auto id = builder.CreateString(group.id);
    const auto stream_vector = builder.CreateVector(streams);
    return wire::CreateGroup(builder, id, stream_vector);
}

std::string text_of(const flatbuffers::String* string)
{
    return string != nullptr ? string->str() : std::string();
}

std::optional<PixelFormat> pixel_format_of(wire::PixelFormat format)
{
    switch (format)
    {
    case wire::PixelFormat_Bgra8:
    case wire::PixelFormat_Rgba8:
    case wire::PixelFormat_Nv12:
    case wire::PixelFormat_Gray8:
        return static_cast<PixelFormat>(format);
    default:
        return std::nullopt;
    }
}

std::optional<Codec> codec_of(wire::Codec codec)
{
    switch (codec)
    {
    case wire::Codec_None:
    case wire::Codec_H264:
    case wire::Codec_H265:
        return static_cast<Codec>(codec);
    default:
        return std::nullopt;
    }
}

std::optional<DeliveryOrder> order_of(wire::DeliveryOrder order)
{
    switch (order)
    {
    case wire::DeliveryOrder_Default:
    case wire::DeliveryOrder_Newest:
    case wire::DeliveryOrder_InOrder:
        return static_cast<DeliveryOrder>(order);
    default:
        return std::nullopt;
    }
}

GroupDescription read_group(const wire::Group* group)
{
    GroupDescription result;
    if (group == nullptr)
    {
        return result;
    }
    result.id = text_of(group->id());
    const auto* streams = group->streams();
    if (streams == nullptr)
    {
        return result;
    }
    if (streams->size() > MAX_STREAMS_PER_GROUP)
    {
        throw WireError("camera group '" + result.id + "' has more streams than a group can have");
    }
    for (const wire::Stream* stream : *streams)
    {
        StreamDescription description;
        description.id = text_of(stream->id());
        const auto codec = codec_of(stream->codec());
        const auto pixel_format = pixel_format_of(stream->pixel_format());
        // An encoded stream has no pixel format to know.
        if (!codec || stream->memory() != wire::MemoryKind_Host || (*codec == Codec::NONE && !pixel_format))
        {
            throw WireError("camera stream '" + description.id +
                            "' uses a format or a kind of memory this build does not know");
        }
        description.codec = *codec;
        description.pixel_format = pixel_format.value_or(PixelFormat::BGRA8);
        description.memory = MemoryKind::HOST;
        description.width = stream->width();
        description.height = stream->height();
        if (const auto* planes = stream->planes())
        {
            if (planes->size() > MAX_PLANES_PER_IMAGE)
            {
                throw WireError("camera stream '" + description.id + "' has more planes than an image can have");
            }
            description.plane_count = planes->size();
            for (std::uint32_t i = 0; i < planes->size(); ++i)
            {
                description.planes[i] = { planes->Get(i)->offset_bytes(), planes->Get(i)->stride_bytes() };
            }
        }
        description.slot_bytes = stream->slot_bytes();
        description.slot_count = stream->slot_count();
        description.frame_rate = stream->frame_rate();
        result.streams.push_back(std::move(description));
    }
    return result;
}

FrameEvent read_frame(const wire::Frame& frame)
{
    FrameEvent event;
    event.lease_id = frame.lease_id();
    event.sequence = frame.sequence();
    event.generation = frame.generation();
    event.skipped = frame.skipped();
    if (const wire::Timeline* timeline = frame.timeline())
    {
        event.timeline = { timeline->capture_ns(), timeline->sdk_ready_ns(), timeline->published_ns(),
                           timeline->sent_ns() };
    }
    if (const auto* images = frame.images())
    {
        if (images->size() > MAX_STREAMS_PER_GROUP)
        {
            throw WireError("a camera frame set has more images than a group has streams");
        }
        event.image_count = images->size();
        for (std::uint32_t i = 0; i < images->size(); ++i)
        {
            const wire::Image* image = images->Get(i);
            event.images[i] = { image->slot(), image->flags(), image->bytes_used(), image->device_frame_id(),
                                image->device_time_ns() };
        }
    }
    return event;
}

} // namespace

struct WireEncoder::Impl
{
    Builder builder{ 1024 };

    template <typename Table>
    std::span<const std::byte> finish(wire::Body type, flatbuffers::Offset<Table> body)
    {
        builder.Finish(wire::CreateMessage(builder, type, body.Union()));
        if (builder.GetSize() > MAX_WIRE_MESSAGE_BYTES)
        {
            throw std::length_error("a camera wire message of " + std::to_string(builder.GetSize()) +
                                    " bytes is over the limit of " + std::to_string(MAX_WIRE_MESSAGE_BYTES));
        }
        return { reinterpret_cast<const std::byte*>(builder.GetBufferPointer()), builder.GetSize() };
    }
};

WireEncoder::WireEncoder() : m_impl(std::make_unique<Impl>())
{
}

WireEncoder::~WireEncoder() = default;

std::span<const std::byte> WireEncoder::encode(const HelloMessage& message)
{
    Builder& builder = m_impl->builder;
    builder.Clear();
    const auto name = builder.CreateString(message.reader_name);
    return m_impl->finish(wire::Body_Hello, wire::CreateHello(builder, message.protocol_version, name));
}

std::span<const std::byte> WireEncoder::encode(const WelcomeMessage& message)
{
    Builder& builder = m_impl->builder;
    builder.Clear();
    std::vector<flatbuffers::Offset<wire::Group>> groups;
    groups.reserve(message.groups.size());
    for (const GroupDescription& group : message.groups)
    {
        groups.push_back(build_group(builder, group));
    }
    const auto group_vector = builder.CreateVector(groups);
    const auto error = builder.CreateString(message.error);
    return m_impl->finish(
        wire::Body_Welcome, wire::CreateWelcome(builder, message.protocol_version, group_vector, error));
}

std::span<const std::byte> WireEncoder::encode(const SubscribeMessage& message)
{
    Builder& builder = m_impl->builder;
    builder.Clear();
    const auto group = builder.CreateString(message.group);
    return m_impl->finish(
        wire::Body_Subscribe, wire::CreateSubscribe(builder, group, message.options.window,
                                                    static_cast<wire::DeliveryOrder>(message.options.order)));
}

std::span<const std::byte> WireEncoder::encode(const SubscribedMessage& message)
{
    Builder& builder = m_impl->builder;
    builder.Clear();
    const auto error = builder.CreateString(message.error);
    const auto group = build_group(builder, message.group);
    return m_impl->finish(wire::Body_Subscribed, wire::CreateSubscribed(builder, error, group));
}

// The per-frame path: nothing here allocates once the builder has grown to a frame's size.
std::span<const std::byte> WireEncoder::encode(const FrameEvent& message)
{
    Builder& builder = m_impl->builder;
    builder.Clear();
    std::array<flatbuffers::Offset<wire::Image>, MAX_STREAMS_PER_GROUP> images;
    const std::size_t image_count = std::min<std::size_t>(message.image_count, MAX_STREAMS_PER_GROUP);
    for (std::size_t i = 0; i < image_count; ++i)
    {
        const ImageInfo& image = message.images[i];
        images[i] = wire::CreateImage(
            builder, image.slot, image.flags, image.bytes_used, image.device_frame_id, image.device_time_ns);
    }
    const auto image_vector = builder.CreateVector(images.data(), image_count);
    const auto timeline = wire::CreateTimeline(builder, message.timeline.capture_ns, message.timeline.sdk_ready_ns,
                                               message.timeline.published_ns, message.timeline.sent_ns);
    return m_impl->finish(
        wire::Body_Frame, wire::CreateFrame(builder, message.lease_id, message.sequence, message.generation,
                                            message.skipped, timeline, image_vector));
}

std::span<const std::byte> WireEncoder::encode(const ReleaseMessage& message)
{
    Builder& builder = m_impl->builder;
    builder.Clear();
    return m_impl->finish(wire::Body_Release, wire::CreateRelease(builder, message.lease_id));
}

WireMessage decode_wire_message(std::span<const std::byte> bytes)
{
    const auto* data = reinterpret_cast<const std::uint8_t*>(bytes.data());
    flatbuffers::Verifier verifier(data, bytes.size());
    if (bytes.empty() || !wire::VerifyMessageBuffer(verifier))
    {
        throw WireError("received bytes that are not a camera wire message");
    }
    const wire::Message* message = wire::GetMessage(data);
    // The verifier accepts a body type without a body.
    if (message->body_type() != wire::Body_NONE && message->body() == nullptr)
    {
        throw WireError("received a camera wire message without a body");
    }
    switch (message->body_type())
    {
    case wire::Body_Hello:
    {
        const wire::Hello* hello = message->body_as_Hello();
        return HelloMessage{ hello->protocol_version(), text_of(hello->reader_name()) };
    }
    case wire::Body_Welcome:
    {
        const wire::Welcome* welcome = message->body_as_Welcome();
        WelcomeMessage result;
        result.protocol_version = welcome->protocol_version();
        result.error = text_of(welcome->error());
        if (const auto* groups = welcome->groups())
        {
            for (const wire::Group* group : *groups)
            {
                result.groups.push_back(read_group(group));
            }
        }
        return result;
    }
    case wire::Body_Subscribe:
    {
        const wire::Subscribe* subscribe = message->body_as_Subscribe();
        const auto order = order_of(subscribe->order());
        if (!order)
        {
            throw WireError("a camera reader asked for a delivery order this build does not know");
        }
        return SubscribeMessage{ text_of(subscribe->group()), { subscribe->window(), *order } };
    }
    case wire::Body_Subscribed:
    {
        const wire::Subscribed* subscribed = message->body_as_Subscribed();
        return SubscribedMessage{ text_of(subscribed->error()), read_group(subscribed->group()) };
    }
    case wire::Body_Frame:
        return read_frame(*message->body_as_Frame());
    case wire::Body_Release:
        return ReleaseMessage{ message->body_as_Release()->lease_id() };
    default:
        return std::monostate{};
    }
}

} // namespace core::camera
