// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/camera_protocol/messages.hpp"

#include <schema/camera_protocol_generated.h>

#include <algorithm>
#include <array>
#include <optional>

namespace core::camera
{

namespace
{

// The schema's enums carry the same numbers as the core's, so a value converts by a cast.
static_assert(static_cast<std::uint32_t>(PixelFormat::BGRA8) == CameraPixelFormat_Bgra8);
static_assert(static_cast<std::uint32_t>(PixelFormat::RGBA8) == CameraPixelFormat_Rgba8);
static_assert(static_cast<std::uint32_t>(PixelFormat::NV12) == CameraPixelFormat_Nv12);
static_assert(static_cast<std::uint32_t>(PixelFormat::GRAY8) == CameraPixelFormat_Gray8);
static_assert(static_cast<std::uint32_t>(Codec::NONE) == CameraCodec_None);
static_assert(static_cast<std::uint32_t>(Codec::H264) == CameraCodec_H264);
static_assert(static_cast<std::uint32_t>(Codec::H265) == CameraCodec_H265);
static_assert(static_cast<std::uint32_t>(MemoryKind::HOST) == CameraMemoryKind_Host);
static_assert(static_cast<std::uint32_t>(DeliveryOrder::DEFAULT) == CameraDeliveryOrder_Default);
static_assert(static_cast<std::uint32_t>(DeliveryOrder::NEWEST) == CameraDeliveryOrder_Newest);
static_assert(static_cast<std::uint32_t>(DeliveryOrder::IN_ORDER) == CameraDeliveryOrder_InOrder);

using Builder = flatbuffers::FlatBufferBuilder;

flatbuffers::Offset<CameraGroup> build_group(Builder& builder, const GroupDescription& group)
{
    std::vector<flatbuffers::Offset<CameraStream>> streams;
    streams.reserve(group.streams.size());
    for (const StreamDescription& stream : group.streams)
    {
        std::array<std::uint32_t, MAX_PLANES_PER_IMAGE> offsets = {};
        std::array<std::uint32_t, MAX_PLANES_PER_IMAGE> strides = {};
        const std::size_t plane_count = std::min<std::size_t>(stream.plane_count, MAX_PLANES_PER_IMAGE);
        for (std::size_t i = 0; i < plane_count; ++i)
        {
            offsets[i] = stream.planes[i].offset_bytes;
            strides[i] = stream.planes[i].stride_bytes;
        }
        const auto id = builder.CreateString(stream.id);
        const auto offset_vector = builder.CreateVector(offsets.data(), plane_count);
        const auto stride_vector = builder.CreateVector(strides.data(), plane_count);
        streams.push_back(CreateCameraStream(
            builder, id, static_cast<CameraCodec>(stream.codec), static_cast<CameraPixelFormat>(stream.pixel_format),
            static_cast<CameraMemoryKind>(stream.memory), stream.width, stream.height, offset_vector, stride_vector,
            stream.slot_bytes, stream.slot_count, stream.frame_rate));
    }
    const auto id = builder.CreateString(group.id);
    const auto stream_vector = builder.CreateVector(streams);
    return CreateCameraGroup(builder, id, stream_vector);
}

std::string text_of(const flatbuffers::String* string)
{
    return string != nullptr ? string->str() : std::string();
}

std::optional<PixelFormat> pixel_format_of(CameraPixelFormat format)
{
    switch (format)
    {
    case CameraPixelFormat_Bgra8:
    case CameraPixelFormat_Rgba8:
    case CameraPixelFormat_Nv12:
    case CameraPixelFormat_Gray8:
        return static_cast<PixelFormat>(format);
    default:
        return std::nullopt;
    }
}

std::optional<Codec> codec_of(CameraCodec codec)
{
    switch (codec)
    {
    case CameraCodec_None:
    case CameraCodec_H264:
    case CameraCodec_H265:
        return static_cast<Codec>(codec);
    default:
        return std::nullopt;
    }
}

std::optional<DeliveryOrder> order_of(CameraDeliveryOrder order)
{
    switch (order)
    {
    case CameraDeliveryOrder_Default:
    case CameraDeliveryOrder_Newest:
    case CameraDeliveryOrder_InOrder:
        return static_cast<DeliveryOrder>(order);
    default:
        return std::nullopt;
    }
}

GroupDescription read_group(const CameraGroup& group)
{
    GroupDescription result;
    result.id = text_of(group.id());
    const auto* streams = group.streams();
    if (streams == nullptr)
    {
        return result;
    }
    if (streams->size() > MAX_STREAMS_PER_GROUP)
    {
        throw ProtocolError("camera group '" + result.id + "' has more streams than a group can have");
    }
    for (const CameraStream* stream : *streams)
    {
        StreamDescription description;
        description.id = text_of(stream->id());
        const auto codec = codec_of(stream->codec());
        const auto pixel_format = pixel_format_of(stream->pixel_format());
        // An encoded stream has no pixel format to know.
        if (!codec || stream->memory() != CameraMemoryKind_Host || (*codec == Codec::NONE && !pixel_format))
        {
            throw ProtocolError("camera stream '" + description.id +
                                "' uses a format or a kind of memory this build does not know");
        }
        description.codec = *codec;
        description.pixel_format = pixel_format.value_or(PixelFormat::BGRA8);
        description.memory = MemoryKind::HOST;
        description.width = stream->width();
        description.height = stream->height();

        const auto* offsets = stream->plane_offsets();
        const auto* strides = stream->plane_strides();
        const std::uint32_t plane_count = offsets != nullptr ? offsets->size() : 0;
        if (plane_count > MAX_PLANES_PER_IMAGE || plane_count != (strides != nullptr ? strides->size() : 0))
        {
            throw ProtocolError("camera stream '" + description.id + "' does not have one offset and one stride for " +
                                "each plane an image can have");
        }
        description.plane_count = plane_count;
        for (std::uint32_t i = 0; i < plane_count; ++i)
        {
            description.planes[i] = { offsets->Get(i), strides->Get(i) };
        }
        description.slot_bytes = stream->slot_bytes();
        description.slot_count = stream->slot_count();
        description.frame_rate = stream->frame_rate();
        result.streams.push_back(std::move(description));
    }
    return result;
}

FrameEvent read_frame(const CameraFrame& frame)
{
    FrameEvent event;
    event.lease_id = frame.lease_id();
    event.sequence = frame.sequence();
    event.generation = frame.generation();
    event.skipped = frame.skipped();
    event.timeline = { frame.capture_ns(), frame.sdk_ready_ns(), frame.published_ns(), frame.sent_ns() };
    if (const auto* images = frame.images())
    {
        if (images->size() > MAX_STREAMS_PER_GROUP)
        {
            throw ProtocolError("a camera frame set has more images than a group has streams");
        }
        event.image_count = images->size();
        for (std::uint32_t i = 0; i < images->size(); ++i)
        {
            const CameraImage* image = images->Get(i);
            event.images[i] = { image->slot(), image->flags(), image->bytes_used(), image->device_frame_id(),
                                image->device_time_ns() };
        }
    }
    return event;
}

} // namespace

struct MessageEncoder::Impl
{
    Builder builder{ 1024 };

    template <typename Table>
    std::span<const std::byte> finish(CameraMessageBody type, flatbuffers::Offset<Table> body)
    {
        builder.Finish(CreateCameraMessage(builder, type, body.Union()));
        if (builder.GetSize() > MAX_MESSAGE_BYTES)
        {
            throw std::length_error("a camera message of " + std::to_string(builder.GetSize()) +
                                    " bytes is over the limit of " + std::to_string(MAX_MESSAGE_BYTES));
        }
        return { reinterpret_cast<const std::byte*>(builder.GetBufferPointer()), builder.GetSize() };
    }
};

MessageEncoder::MessageEncoder() : m_impl(std::make_unique<Impl>())
{
}

MessageEncoder::~MessageEncoder() = default;

std::span<const std::byte> MessageEncoder::encode(const HandshakeRequest& message)
{
    Builder& builder = m_impl->builder;
    builder.Clear();
    const auto name = builder.CreateString(message.reader_name);
    const auto group = builder.CreateString(message.group);
    return m_impl->finish(
        CameraMessageBody_CameraHandshakeRequest,
        CreateCameraHandshakeRequest(builder, message.protocol_version, name, group, message.options.window,
                                     static_cast<CameraDeliveryOrder>(message.options.order)));
}

std::span<const std::byte> MessageEncoder::encode(const HandshakeResponse& message)
{
    Builder& builder = m_impl->builder;
    builder.Clear();
    std::vector<flatbuffers::Offset<CameraGroup>> groups;
    groups.reserve(message.groups.size());
    for (const GroupDescription& group : message.groups)
    {
        groups.push_back(build_group(builder, group));
    }
    const auto group_vector = builder.CreateVector(groups);
    const auto error = builder.CreateString(message.error);
    return m_impl->finish(CameraMessageBody_CameraHandshakeResponse,
                          CreateCameraHandshakeResponse(builder, message.protocol_version, error, group_vector));
}

// The per-frame path: nothing here allocates once the builder has grown to a frame's size.
std::span<const std::byte> MessageEncoder::encode(const FrameEvent& message)
{
    Builder& builder = m_impl->builder;
    builder.Clear();
    std::array<flatbuffers::Offset<CameraImage>, MAX_STREAMS_PER_GROUP> images;
    const std::size_t image_count = std::min<std::size_t>(message.image_count, MAX_STREAMS_PER_GROUP);
    for (std::size_t i = 0; i < image_count; ++i)
    {
        const ImageInfo& image = message.images[i];
        images[i] = CreateCameraImage(
            builder, image.slot, image.flags, image.bytes_used, image.device_frame_id, image.device_time_ns);
    }
    const auto image_vector = builder.CreateVector(images.data(), image_count);
    return m_impl->finish(CameraMessageBody_CameraFrame,
                          CreateCameraFrame(builder, message.lease_id, message.sequence, message.generation,
                                            message.skipped, message.timeline.capture_ns, message.timeline.sdk_ready_ns,
                                            message.timeline.published_ns, message.timeline.sent_ns, image_vector));
}

std::span<const std::byte> MessageEncoder::encode(const ReleaseMessage& message)
{
    Builder& builder = m_impl->builder;
    builder.Clear();
    return m_impl->finish(CameraMessageBody_CameraRelease, CreateCameraRelease(builder, message.lease_id));
}

ProtocolMessage decode_message(std::span<const std::byte> bytes)
{
    const auto* data = reinterpret_cast<const std::uint8_t*>(bytes.data());
    flatbuffers::Verifier verifier(data, bytes.size());
    if (bytes.empty() || !VerifyCameraMessageBuffer(verifier))
    {
        throw ProtocolError("received bytes that are not a camera message");
    }
    const CameraMessage* message = GetCameraMessage(data);
    // The verifier accepts a body type without a body.
    if (message->body_type() != CameraMessageBody_NONE && message->body() == nullptr)
    {
        throw ProtocolError("received a camera message without a body");
    }
    switch (message->body_type())
    {
    case CameraMessageBody_CameraHandshakeRequest:
    {
        const CameraHandshakeRequest* request = message->body_as_CameraHandshakeRequest();
        const auto order = order_of(request->order());
        if (!order)
        {
            throw ProtocolError("a camera reader asked for a delivery order this build does not know");
        }
        return HandshakeRequest{ request->protocol_version(),
                                 text_of(request->reader_name()),
                                 text_of(request->group()),
                                 { request->window(), *order } };
    }
    case CameraMessageBody_CameraHandshakeResponse:
    {
        const CameraHandshakeResponse* response = message->body_as_CameraHandshakeResponse();
        HandshakeResponse result;
        result.protocol_version = response->protocol_version();
        result.error = text_of(response->error());
        if (const auto* groups = response->groups())
        {
            for (const CameraGroup* group : *groups)
            {
                result.groups.push_back(read_group(*group));
            }
        }
        return result;
    }
    case CameraMessageBody_CameraFrame:
        return read_frame(*message->body_as_CameraFrame());
    case CameraMessageBody_CameraRelease:
        return ReleaseMessage{ message->body_as_CameraRelease()->lease_id() };
    default:
        return std::monostate{};
    }
}

} // namespace core::camera
