// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/camera_protocol/messages.hpp"

#include <schema/camera_protocol_generated.h>

#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace core::camera
{

namespace
{

// The schema's enums, flag bits and message numbers are the core's and the protocol's: a value
// converts by a cast, and a change to either side stops the build here.
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
static_assert(std::uint32_t{ IMAGE_FLAG_KEYFRAME } == std::uint32_t{ CameraFrameFlag_Keyframe });
static_assert(std::uint32_t{ IMAGE_FLAG_SDK_WARNING } == std::uint32_t{ CameraFrameFlag_SdkWarning });
static_assert(std::uint32_t{ CameraFrameFlag_Keyframe } == 1u && std::uint32_t{ CameraFrameFlag_SdkWarning } == 2u);
static_assert(CameraMessageBody_CameraHandshakeRequest == 1);
static_assert(CameraMessageBody_CameraHandshakeResponse == 2);
static_assert(CameraMessageBody_CameraFrameSet == 3);
static_assert(CameraMessageBody_CameraRelease == 4);
static_assert(CameraMessageBody_CameraAcquire == 5);

using Builder = flatbuffers::FlatBufferBuilder;

flatbuffers::Offset<CameraGroup> build_group(Builder& builder, const GroupDescription& group)
{
    if (group.streams.size() > MAX_STREAMS_PER_GROUP)
    {
        throw std::invalid_argument("camera group '" + group.id + "' has more streams than a group can have");
    }
    std::vector<flatbuffers::Offset<CameraStream>> streams;
    streams.reserve(group.streams.size());
    for (const StreamDescription& stream : group.streams)
    {
        if (stream.plane_count > MAX_PLANES_PER_IMAGE)
        {
            throw std::invalid_argument("camera stream '" + stream.id + "' has more planes than an image can have");
        }
        std::array<CameraPlaneLayout, MAX_PLANES_PER_IMAGE> planes;
        for (std::size_t i = 0; i < stream.plane_count; ++i)
        {
            planes[i] = CameraPlaneLayout(stream.planes[i].offset_bytes, stream.planes[i].stride_bytes);
        }
        const auto id = builder.CreateString(stream.id);
        const auto plane_vector = builder.CreateVectorOfStructs(planes.data(), stream.plane_count);
        streams.push_back(CreateCameraStream(builder, id, static_cast<CameraCodec>(stream.codec),
                                             static_cast<CameraPixelFormat>(stream.pixel_format),
                                             static_cast<CameraMemoryKind>(stream.memory), stream.width, stream.height,
                                             plane_vector, stream.slot_bytes, stream.slot_count, stream.frame_rate));
    }
    const auto id = builder.CreateString(group.id);
    const auto stream_vector = builder.CreateVector(streams);
    return CreateCameraGroup(builder, id, stream_vector, group.session_id);
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

// Why this build cannot represent a stream; nothing when it can. A newer plugin may publish
// such a stream next to ones this build reads, so this is a property of the group and not a
// broken message.
std::string unsupported_by_this_build(const CameraStream& stream, StreamDescription& description)
{
    const std::string name = "camera stream '" + description.id + "'";
    const auto codec = codec_of(stream.codec());
    if (!codec)
    {
        return name + " uses a codec this build does not know (" + std::to_string(stream.codec()) + ")";
    }
    if (stream.memory() != CameraMemoryKind_Host)
    {
        return name + " is in a kind of memory this build does not know (" + std::to_string(stream.memory()) + ")";
    }
    const auto pixel_format = pixel_format_of(stream.pixel_format());
    if (*codec == Codec::NONE && !pixel_format) // an encoded stream has no pixel format to know
    {
        return name + " uses a pixel format this build does not know (" + std::to_string(stream.pixel_format()) + ")";
    }
    const auto* planes = stream.planes();
    const std::uint32_t plane_count = planes != nullptr ? planes->size() : 0;
    if (plane_count > MAX_PLANES_PER_IMAGE)
    {
        return name + " has more planes than this build can describe";
    }
    description.codec = *codec;
    description.pixel_format = pixel_format.value_or(PixelFormat::BGRA8);
    description.memory = MemoryKind::HOST;
    description.plane_count = plane_count;
    for (std::uint32_t i = 0; i < plane_count; ++i)
    {
        description.planes[i] = { planes->Get(i)->offset_bytes(), planes->Get(i)->stride_bytes() };
    }
    return {};
}

GroupEntry read_group(const CameraGroup& group)
{
    GroupEntry entry;
    entry.group.id = text_of(group.id());
    entry.group.session_id = group.session_id();
    const auto* streams = group.streams();
    if (streams == nullptr)
    {
        return entry;
    }
    if (streams->size() > MAX_STREAMS_PER_GROUP)
    {
        entry.unsupported = "camera group '" + entry.group.id + "' has more streams than this build can read";
        return entry;
    }
    for (const CameraStream* stream : *streams)
    {
        StreamDescription description;
        description.id = text_of(stream->id());
        entry.unsupported = unsupported_by_this_build(*stream, description);
        if (!entry.unsupported.empty())
        {
            entry.group.streams.clear();
            return entry;
        }
        description.width = stream->width();
        description.height = stream->height();
        description.slot_bytes = stream->slot_bytes();
        description.slot_count = stream->slot_count();
        description.frame_rate = stream->frame_rate();
        entry.group.streams.push_back(std::move(description));
    }
    return entry;
}

FrameEvent read_frame_set(const CameraFrameSet& frame_set)
{
    FrameEvent event;
    event.lease_id = frame_set.lease_id();
    event.sequence = frame_set.sequence();
    event.generation = frame_set.generation();
    event.skipped = frame_set.skipped();
    event.timeline = { frame_set.capture_ns(), frame_set.sdk_ready_ns(), frame_set.published_ns(), frame_set.sent_ns() };
    if (const auto* frames = frame_set.frames())
    {
        if (frames->size() > MAX_STREAMS_PER_GROUP)
        {
            throw ProtocolError("a camera frame set has more images than a group has streams");
        }
        event.image_count = frames->size();
        for (std::uint32_t i = 0; i < frames->size(); ++i)
        {
            const CameraFrameRef* frame = frames->Get(i);
            event.images[i] = { frame->slot_index(), static_cast<std::uint32_t>(frame->flags()), frame->bytes_used(),
                                frame->device_frame_id(), frame->device_time_ns() };
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
    for (const GroupEntry& entry : message.groups)
    {
        groups.push_back(build_group(builder, entry.group));
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
    if (message.image_count > MAX_STREAMS_PER_GROUP)
    {
        throw std::invalid_argument("a camera frame set has more images than a group has streams");
    }
    std::array<flatbuffers::Offset<CameraFrameRef>, MAX_STREAMS_PER_GROUP> frames;
    for (std::size_t i = 0; i < message.image_count; ++i)
    {
        const ImageInfo& image = message.images[i];
        frames[i] = CreateCameraFrameRef(builder, image.slot, static_cast<CameraFrameFlag>(image.flags),
                                         image.bytes_used, image.device_frame_id, image.device_time_ns);
    }
    const auto frame_vector = builder.CreateVector(frames.data(), message.image_count);
    return m_impl->finish(CameraMessageBody_CameraFrameSet,
                          CreateCameraFrameSet(builder, message.lease_id, message.sequence, message.generation,
                                               message.skipped, message.timeline.capture_ns, message.timeline.sdk_ready_ns,
                                               message.timeline.published_ns, message.timeline.sent_ns, frame_vector));
}

std::span<const std::byte> MessageEncoder::encode(const ReleaseMessage& message)
{
    Builder& builder = m_impl->builder;
    builder.Clear();
    return m_impl->finish(CameraMessageBody_CameraRelease, CreateCameraRelease(builder, message.lease_id));
}

std::span<const std::byte> MessageEncoder::encode(const AcquireMessage& message)
{
    Builder& builder = m_impl->builder;
    builder.Clear();
    return m_impl->finish(CameraMessageBody_CameraAcquire, CreateCameraAcquire(builder, message.after_sequence));
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
    const auto type = static_cast<std::uint32_t>(message->body_type());
    if (type == CameraMessageBody_NONE || type > CameraMessageBody_MAX)
    {
        // From a newer peer. One that only informs can be ignored. Anything else is an
        // operation the peer counts on, and carrying on without it would go wrong quietly.
        if (message->skippable())
        {
            return std::monostate{};
        }
        throw ProtocolError("received a camera message of a type this build does not know (" + std::to_string(type) +
                            ") and cannot ignore");
    }
    // The verifier accepts a body type without a body.
    if (message->body() == nullptr)
    {
        throw ProtocolError("received a camera message without a body");
    }
    switch (message->body_type())
    {
    case CameraMessageBody_CameraHandshakeRequest:
    {
        const CameraHandshakeRequest* request = message->body_as_CameraHandshakeRequest();
        HandshakeRequest result;
        result.protocol_version = request->protocol_version();
        result.reader_name = text_of(request->reader_name());
        result.group = text_of(request->group_id());
        result.options.window = request->window();
        if (const auto order = order_of(request->order()))
        {
            result.options.order = *order;
        }
        else
        {
            result.unsupported = "the reader asked for a delivery order this build does not know (" +
                                 std::to_string(request->order()) + ")";
        }
        return result;
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
    case CameraMessageBody_CameraFrameSet:
        return read_frame_set(*message->body_as_CameraFrameSet());
    case CameraMessageBody_CameraRelease:
        return ReleaseMessage{ message->body_as_CameraRelease()->lease_id() };
    case CameraMessageBody_CameraAcquire:
        return AcquireMessage{ message->body_as_CameraAcquire()->after_sequence() };
    default:
        throw ProtocolError("received a camera message this build has no reader for");
    }
}

} // namespace core::camera
