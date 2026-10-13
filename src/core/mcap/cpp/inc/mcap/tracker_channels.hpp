// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "recorded_schemas.hpp"

#include <flatbuffers/flatbuffers.h>
#include <log_bridge/logger.hpp>
#include <mcap/reader.hpp>
#include <mcap/writer.hpp>
#include <schema/serialized.hpp>
#include <schema/timestamp_generated.h>
#include <schema_compat/schema_compat.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace core
{

inline std::string mcap_topic(std::string_view base_name, const std::string& sub_channel)
{
    return std::string(base_name) + "/" + sub_channel;
}

/**
 * @brief Type-safe MCAP channel writer for FlatBuffer record types.
 *
 * @tparam RecordT The FlatBuffer root written to the channels: a record wrapper (e.g.
 *                 HeadPoseRecord, encoded by pack_record()), or a table recorded as itself,
 *                 which has no timestamp and is written with an explicit log time.
 *                 Must expose BinarySchema.
 *
 * The factory creates a unique_ptr<McapTrackerChannels<...>> only when recording
 * is active and passes it to the impl. Impls null-check before calling write().
 */
template <typename RecordT>
class McapTrackerChannels
{
public:
    McapTrackerChannels(mcap::McapWriter& writer, std::string_view base_name, const std::vector<std::string>& sub_channels)
        : writer_(&writer)
    {
        std::string_view schema_text(
            reinterpret_cast<const char*>(RecordT::BinarySchema::data()), RecordT::BinarySchema::size());

        // flatc derives this name from the .fbs, and McapTrackerViewers matches the recorded
        // one against the same expression, so the two cannot be made to disagree by hand.
        mcap::Schema schema(RecordT::GetFullyQualifiedName(), "flatbuffer", std::string(schema_text));
        writer_->addSchema(schema);

        channel_ids_.reserve(sub_channels.size());
        for (const auto& sub : sub_channels)
        {
            mcap::Channel ch(mcap_topic(base_name, sub), "flatbuffer", schema.id);
            writer_->addChannel(ch);
            channel_ids_.push_back(ch.id);
        }
    }

    /*!
     * @brief Write a Record that is already encoded.
     *
     */
    void write(size_t channel_index, const Serialized<RecordT>& record)
    {
        const auto* timestamp = record ? record->timestamp() : nullptr;
        write(channel_index, record, timestamp != nullptr ? timestamp->available_time_local_common_clock() : 0);
    }

    //! Writes a root with no timestamp of its own, such as a table recorded as itself, at `log_time_ns`.
    void write(size_t channel_index, const Serialized<RecordT>& record, int64_t log_time_ns)
    {
        if (channel_index >= channel_ids_.size())
        {
            throw std::out_of_range(
                "McapTrackerChannels: write called with channel_index=" + std::to_string(channel_index) + " but only " +
                std::to_string(channel_ids_.size()) + " channels registered");
        }

        const std::span<const uint8_t> bytes = record.buffer();
        if (bytes.empty())
        {
            throw std::invalid_argument("McapTrackerChannels: write called with a record that owns no buffer");
        }

        mcap::Message msg;
        msg.channelId = channel_ids_[channel_index];
        msg.logTime = static_cast<mcap::Timestamp>(log_time_ns);
        msg.publishTime = msg.logTime;
        msg.sequence = sequence_++;
        msg.data = reinterpret_cast<const std::byte*>(bytes.data());
        msg.dataSize = bytes.size();
        auto status = writer_->write(msg);
        if (!status.ok())
        {
            isaaccapture::Logger::get("isaaccapture.core.McapTrackerChannels")->error("write failed: {}", status.message);
        }
    }

private:
    mcap::McapWriter* writer_;
    std::vector<mcap::ChannelId> channel_ids_;
    uint32_t sequence_ = 0;
};

//! The payload a Record wraps, as its object-API type. Every generated Record native
//! holds its payload in a `std::shared_ptr<PayloadT> data`, which is what lets the payload
//! be named from the Record type alone -- and so lets an absent one be spelled `nullptr`.
template <typename RecordT>
using record_payload_t = typename decltype(std::declval<typename RecordT::NativeTableType>().data)::element_type;

/*!
 * @brief Encodes a payload and its timestamp into the Record type MCAP stores.
 *
 * The encode lives here rather than on the writer so that McapTrackerChannels only ever
 * moves bytes: a caller that also publishes the payload keeps the record and narrows into
 * it, and one that needs the same record on two channels writes it twice.
 *
 * A null `data` records the timestamp alone -- an inactive device, or a frame marker on a
 * channel whose tracker drained nothing. The payload type comes from `RecordT`, so only
 * the Record has to be named at the call site.
 */
template <typename RecordT>
Serialized<RecordT> pack_record(const record_payload_t<RecordT>* data, const DeviceDataTimestamp& timestamp)
{
    using DataTableT = typename record_payload_t<RecordT>::TableType;

    flatbuffers::FlatBufferBuilder builder(256);

    flatbuffers::Offset<DataTableT> data_offset;
    if (data != nullptr)
    {
        data_offset = DataTableT::Pack(builder, data);
    }

    DeviceDataTimestamp ts = timestamp;
    typename RecordT::Builder record_builder(builder);
    if (data != nullptr)
    {
        record_builder.add_data(data_offset);
    }
    record_builder.add_timestamp(&ts);
    builder.Finish(record_builder.Finish());

    return Serialized<RecordT>::adopt(builder);
}

//! Address of the contained value, or null when there is none -- the pointer form an
//! optional payload takes at an API that spells absence with nullptr. `std::optional` has
//! no accessor for this, and `&*value` is only valid once you have already tested it.
template <typename T>
const T* value_ptr(const std::optional<T>& value)
{
    return value ? &*value : nullptr;
}

/*!
 * @brief Encode `native` once, record it when recording is on, and return the payload
 *        handle to publish.
 *
 * Recording wraps the payload in a Record, and the payload sits inside those bytes, so
 * there is no reason to encode it twice: with channels attached the published handle is
 * a view into the record that was just written. With recording off there is no Record to
 * write and the payload is encoded on its own. Either way `native` is encoded exactly
 * once, which is what lets it stay a local of `update()`.
 *
 * A null `native` means the device is inactive: an empty handle comes back, and the
 * record still goes out carrying only its timestamp.
 */
template <typename RecordT>
Serialized<typename record_payload_t<RecordT>::TableType> publish_and_record(McapTrackerChannels<RecordT>* channels,
                                                                             size_t channel_index,
                                                                             const DeviceDataTimestamp& timestamp,
                                                                             const record_payload_t<RecordT>* native)
{
    using DataTableT = typename record_payload_t<RecordT>::TableType;

    if (channels == nullptr)
    {
        return native != nullptr ? pack<DataTableT>(*native) : Serialized<DataTableT>();
    }

    const auto record = pack_record<RecordT>(native, timestamp);
    channels->write(channel_index, record);
    return record.narrow(record->data());
}

//! The payload a recorded root carries: the Record's `data`, or the root itself for a table
//! recorded without a Record wrapper. Shares the record's buffer either way.
template <typename DataTableT, typename RecordT>
Serialized<DataTableT> recorded_payload(const Serialized<RecordT>& record)
{
    if constexpr (std::is_same_v<RecordT, DataTableT>)
    {
        return record;
    }
    else
    {
        return record.narrow(record->data());
    }
}

/**
 * @brief Type-safe MCAP channel reader returning owning handles over the recorded records.
 *
 * @tparam RecordT The FlatBuffer record wrapper stored in MCAP (e.g. HeadPoseRecord).
 *
 * Read-side counterpart to McapTrackerChannels. Owns an McapReader and iterates
 * all registered sub-channels through a single LinearMessageView. Each message is
 * verified and copied into a buffer the returned handle owns before the iterator
 * advances, because the underlying FileReader buffer is only valid until the next
 * read() call.
 *
 * When the iterator yields a message for a channel other than the one requested
 * by the caller, that handle is buffered in the corresponding ChannelBuffer and
 * returned on a subsequent read() for that channel index.
 *
 * Every schema the recording registers for these topics is checked against the binary
 * schema RecordT was generated from, so a recording written by a build with a different
 * `.fbs` is reported rather than silently misread (see schema_compat.hpp). The check runs
 * to completion in the constructor: a recording this build cannot decode is refused there,
 * which is what keeps a schema failure from surfacing part-way through a replay and leaves
 * read() with none of it to do. It needs the whole schema list up front (RecordedSchemas),
 * so a recording that cannot produce one is refused as well.
 */
template <typename RecordT>
class McapTrackerViewers
{
public:
    McapTrackerViewers(const McapTrackerViewers&) = delete;
    McapTrackerViewers& operator=(const McapTrackerViewers&) = delete;
    McapTrackerViewers(McapTrackerViewers&&) = delete;
    McapTrackerViewers& operator=(McapTrackerViewers&&) = delete;

    /*!
     * @param recorded What the file declares. Required rather than read here, so that a caller
     *                 replaying one recording through several trackers reads it once: for a
     *                 file with no summary section that is the difference between one pass over
     *                 the data section and one per tracker. Read during construction only.
     */
    McapTrackerViewers(std::unique_ptr<mcap::McapReader> reader,
                       std::string_view base_name,
                       const std::vector<std::string>& sub_channels,
                       const RecordedSchemas& recorded)
        : reader_(std::move(reader))
    {
        for (const auto& sub : sub_channels)
        {
            channels_.push_back({ mcap_topic(base_name, sub), {} });
        }

        std::vector<mcap::SchemaId> graded;
        for (const RecordedSchemas::Channel& channel : recorded.channels())
        {
            if (!channel_index_of(channel.topic).has_value())
            {
                continue;
            }

            // What a channel says its message bytes are is the channel's own, so every one of
            // them is asked. The grade below belongs to the schema, and every channel a tracker
            // reads shares the one its writer registered, so it is asked once.
            enforce_schema_compat(check_message_encoding(channel), channel.topic);

            const mcap::SchemaId schema_id = channel.schema != nullptr ? channel.schema->id : 0;
            if (std::find(graded.begin(), graded.end(), schema_id) != graded.end())
            {
                continue;
            }
            graded.push_back(schema_id);

            enforce_schema_compat(compare_schema(channel), channel.topic);
        }

        auto on_problem = [](const mcap::Status& s) { throw std::runtime_error("McapTrackerViewers:" + s.message); };

        mcap::ReadMessageOptions options;
        options.topicFilter = [this](std::string_view t) { return channel_index_of(t).has_value(); };

        tracker_view_ = std::make_unique<TrackerView>(reader_->readMessages(on_problem, options));
    }

    /**
     * @brief Read the next record as an encoded handle.
     * @param channel_index Index into the sub_channels list passed at construction.
     * @return A handle owning the recorded bytes, empty when no more messages remain.
     *
     * The two levels of absence are the handle and its payload, not a wrapper around
     * either: an empty handle means the stream had nothing left, while a non-empty handle
     * whose `payload()` is null means a record was read for a tracker that was inactive.
     * A record that was read always yields a non-empty handle, so there is nothing for an
     * optional to say that the handle does not.
     *
     * The recorded root is a Record whose `data` is byte-for-byte the payload table
     * consumers read, so a caller narrows to it rather than unpacking: `record.narrow(...)`
     * shares this buffer instead of allocating a second one.
     */
    Serialized<RecordT> read(size_t channel_index)
    {
        if (channel_index >= channels_.size())
        {
            throw std::out_of_range("McapTrackerViewers: read called with channel_index=" + std::to_string(channel_index) +
                                    " but only " + std::to_string(channels_.size()) + " channels registered");
        }

        // Return a previously buffered record if one was stashed while
        // advancing the shared iterator on behalf of a different channel.
        if (!channels_[channel_index].buffer.empty())
        {
            auto result = std::move(channels_[channel_index].buffer.front());
            channels_[channel_index].buffer.pop_front();
            return result;
        }

        while (tracker_view_->it != tracker_view_->view.end())
        {
            const auto& msg_view = *(tracker_view_->it);
            auto idx = channel_index_of(msg_view.channel->topic);
            assert(idx.has_value() && "topic filter admitted a topic this tracker does not read");
            Serialized<RecordT> record = adopt_message(msg_view.message, *idx);

            ++(tracker_view_->it);

            // The requested channel; return the record.
            if (*idx == channel_index)
            {
                return record;
            }
            // Not the requested channel; stash for a future read(idx) call.
            channels_[*idx].buffer.push_back(std::move(record));
        }

        return Serialized<RecordT>();
    }

private:
    struct ChannelBuffer
    {
        std::string topic;
        std::deque<Serialized<RecordT>> buffer;
    };

    struct TrackerView
    {
        mcap::LinearMessageView view;
        mcap::LinearMessageView::Iterator it;

        explicit TrackerView(mcap::LinearMessageView&& v) : view(std::move(v)), it(view.begin())
        {
        }
    };

    //! Where `topic` sits in the sub_channels list, or nullopt when this tracker does not read it.
    std::optional<size_t> channel_index_of(std::string_view topic) const
    {
        for (size_t i = 0; i < channels_.size(); ++i)
        {
            if (channels_[i].topic == topic)
            {
                return i;
            }
        }
        return std::nullopt;
    }

    /*!
     * @brief Whether this channel's messages are FlatBuffers at all.
     *
     * The channel declares this, not its schema, so two channels naming one schema can still
     * disagree. Without the check the payloads reach adopt_message() and come back as a corrupt
     * FlatBuffer, which says nothing about what is actually wrong.
     */
    SchemaCompatResult check_message_encoding(const RecordedSchemas::Channel& channel) const
    {
        if (channel.message_encoding != "flatbuffer")
        {
            return { SchemaCompat::Incompatible,
                     "channel message encoding is '" + channel.message_encoding + "', reader expects 'flatbuffer'" };
        }
        return { SchemaCompat::Identical, {} };
    }

    SchemaCompatResult compare_schema(const RecordedSchemas::Channel& channel) const
    {
        const mcap::Schema* schema = channel.schema.get();
        if (schema == nullptr)
        {
            return { SchemaCompat::Incompatible, "channel carries no embedded schema" };
        }
        // How the bytes below are written, which is a separate field from the channel's own
        // encoding: a channel may carry json messages against a schema stated as a bfbs.
        if (schema->encoding != "flatbuffer")
        {
            return { SchemaCompat::Incompatible,
                     "schema encoding is '" + schema->encoding + "', reader expects 'flatbuffer'" };
        }
        // The same expression the writer names the schema with, so a recording made by any
        // build of this record type matches and one made for another type does not.
        if (schema->name != RecordT::GetFullyQualifiedName())
        {
            return { SchemaCompat::Incompatible, "schema is named '" + schema->name + "', reader expects '" +
                                                     RecordT::GetFullyQualifiedName() + "'" };
        }

        return check_schema_compat(
            std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(schema->data.data()), schema->data.size()),
            std::span<const uint8_t>(RecordT::BinarySchema::data(), RecordT::BinarySchema::size()));
    }

    // Verifies the recorded bytes and takes ownership of a copy of them. The copy is
    // needed either way: the iterator owns the message storage and reuses it on the next
    // advance, so the bytes cannot outlive this call by reference.
    Serialized<RecordT> adopt_message(const mcap::Message& msg, size_t channel_index) const
    {
        flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(msg.data), msg.dataSize);
        if (!verifier.VerifyBuffer<RecordT>())
        {
            throw std::runtime_error("McapTrackerViewers: corrupt FlatBuffer in channel " +
                                     std::to_string(channel_index) + " at sequence " + std::to_string(msg.sequence));
        }

        const auto* bytes = reinterpret_cast<const uint8_t*>(msg.data);
        return Serialized<RecordT>::adopt(std::vector<uint8_t>(bytes, bytes + msg.dataSize));
    }

    std::unique_ptr<mcap::McapReader> reader_;
    std::vector<ChannelBuffer> channels_;
    std::unique_ptr<TrackerView> tracker_view_;
};

} // namespace core
