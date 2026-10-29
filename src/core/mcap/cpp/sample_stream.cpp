// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/mcap/sample_stream.hpp"

#include <mcap/reader.hpp>

#include <algorithm>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace core
{

namespace
{

//! Opcode and body length precede every record body.
constexpr size_t kRecordPrefix = 1 + sizeof(uint64_t);

uint64_t read_le(const uint8_t* bytes, size_t width)
{
    uint64_t value = 0;
    for (size_t i = width; i-- > 0;)
    {
        value = (value << 8) | bytes[i];
    }
    return value;
}

/*!
 * @brief Calls `fn(opcode, record)` for each complete record at the start of `bytes`, where
 *        `record` spans the opcode, the length and the body.
 * @return Bytes consumed; less than `bytes.size()` when the last record is incomplete.
 */
template <typename Fn>
size_t for_each_record(std::span<const uint8_t> bytes, Fn&& fn)
{
    size_t offset = 0;
    while (bytes.size() - offset >= kRecordPrefix)
    {
        const uint64_t body = read_le(bytes.data() + offset + 1, sizeof(uint64_t));
        if (body > bytes.size() - offset - kRecordPrefix)
        {
            break;
        }
        fn(static_cast<mcap::OpCode>(bytes[offset]), bytes.subspan(offset, kRecordPrefix + body));
        offset += kRecordPrefix + body;
    }
    return offset;
}

mcap::Record as_mcap_record(std::span<const uint8_t> record)
{
    // The Parse* functions only read through `data`; mcap::Record just does not declare it const.
    return mcap::Record{ static_cast<mcap::OpCode>(record[0]), record.size() - kRecordPrefix,
                         const_cast<std::byte*>(reinterpret_cast<const std::byte*>(record.data() + kRecordPrefix)) };
}

void require_ok(const mcap::Status& status, const char* what)
{
    if (!status.ok())
    {
        throw std::runtime_error(std::string(what) + ": " + status.message);
    }
}

void append(std::vector<uint8_t>& out, std::span<const uint8_t> bytes)
{
    out.insert(out.end(), bytes.begin(), bytes.end());
}

void append_le(std::vector<uint8_t>& out, uint64_t value, size_t width)
{
    for (size_t i = 0; i < width; ++i)
    {
        out.push_back(static_cast<uint8_t>(value >> (8 * i)));
    }
}

//! The channel id a Message record's body starts with.
uint16_t message_channel_id(std::span<const uint8_t> record, const char* who)
{
    if (record.size() < kRecordPrefix + sizeof(uint16_t))
    {
        throw std::runtime_error(std::string(who) + ": Message record too short for a channel id");
    }
    return static_cast<uint16_t>(read_le(record.data() + kRecordPrefix, sizeof(uint16_t)));
}

} // namespace

std::vector<uint8_t> merge_samples_records(std::span<const uint8_t> older, std::span<const uint8_t> newer)
{
    struct Entry
    {
        uint16_t channel_id;
        std::span<const uint8_t> record;
    };
    std::vector<Entry> entries;
    const auto collect = [&entries](mcap::OpCode opcode, std::span<const uint8_t> record)
    {
        if (opcode != mcap::OpCode::Message)
        {
            throw std::runtime_error("merge_samples_records: a record other than a Message");
        }
        entries.push_back({ message_channel_id(record, "merge_samples_records"), record });
    };
    for (const std::span<const uint8_t> records : { older, newer })
    {
        if (for_each_record(records, collect) != records.size())
        {
            throw std::runtime_error("merge_samples_records: records end inside a record");
        }
    }

    std::unordered_map<uint16_t, size_t> last_index;
    for (size_t i = 0; i < entries.size(); ++i)
    {
        last_index[entries[i].channel_id] = i;
    }

    std::vector<uint8_t> merged;
    for (size_t i = 0; i < entries.size(); ++i)
    {
        if (last_index.at(entries[i].channel_id) == i)
        {
            append(merged, entries[i].record);
        }
    }
    return merged;
}

// ============================================================================
// TrackerSampleStreamWriter
// ============================================================================

mcap::McapWriterOptions TrackerSampleStreamWriter::writer_options()
{
    mcap::McapWriterOptions options("teleop");
    options.noChunking = true;
    options.compression = mcap::Compression::None;
    options.noMessageIndex = true;
    options.noSummary = true;
    return options;
}

void TrackerSampleStreamWriter::set_topic_filter(const std::vector<std::string>& topics)
{
    topic_filter_ = std::set<std::string, std::less<>>(topics.begin(), topics.end());
}

void TrackerSampleStreamWriter::end()
{
}

uint64_t TrackerSampleStreamWriter::size() const
{
    return size_;
}

std::vector<uint8_t> TrackerSampleStreamWriter::take_setup()
{
    return std::exchange(setup_, {});
}

std::vector<uint8_t> TrackerSampleStreamWriter::take_samples()
{
    return std::exchange(samples_, {});
}

bool TrackerSampleStreamWriter::add_record(std::string_view topic, std::span<const uint8_t> data, uint64_t log_time_ns)
{
    const auto channel = kept_topics_.find(topic);
    if (channel == kept_topics_.end())
    {
        return false;
    }
    // Message body: channel id, sequence, log time, publish time, then the data.
    constexpr size_t kHeader = sizeof(uint16_t) + sizeof(uint32_t) + 2 * sizeof(uint64_t);
    samples_.push_back(static_cast<uint8_t>(mcap::OpCode::Message));
    append_le(samples_, kHeader + data.size(), sizeof(uint64_t));
    append_le(samples_, channel->second, sizeof(uint16_t));
    append_le(samples_, 0, sizeof(uint32_t));
    append_le(samples_, log_time_ns, sizeof(uint64_t));
    append_le(samples_, log_time_ns, sizeof(uint64_t));
    append(samples_, data);
    return true;
}

void TrackerSampleStreamWriter::handleWrite(const std::byte* data, uint64_t size)
{
    size_ += size;
    if (ended_)
    {
        return;
    }
    const auto* bytes = reinterpret_cast<const uint8_t*>(data);
    pending_.insert(pending_.end(), bytes, bytes + size);
    consume_records();
}

void TrackerSampleStreamWriter::consume_records()
{
    size_t offset = 0;
    if (!magic_seen_)
    {
        if (pending_.size() < sizeof(mcap::Magic))
        {
            return;
        }
        if (!std::equal(std::begin(mcap::Magic), std::end(mcap::Magic), pending_.begin()))
        {
            throw std::runtime_error("TrackerSampleStreamWriter: output does not start with the MCAP magic");
        }
        magic_seen_ = true;
        offset = sizeof(mcap::Magic);
    }

    const auto split = [this](mcap::OpCode opcode, std::span<const uint8_t> record)
    {
        switch (opcode)
        {
        case mcap::OpCode::Schema:
            append(setup_, record);
            break;
        case mcap::OpCode::Channel:
        {
            mcap::Channel channel;
            require_ok(mcap::McapReader::ParseChannel(as_mcap_record(record), &channel),
                       "TrackerSampleStreamWriter: bad Channel record");
            if (topic_filter_.empty() || topic_filter_.contains(channel.topic))
            {
                kept_channels_.insert(channel.id);
                kept_topics_[channel.topic] = channel.id;
                append(setup_, record);
            }
            break;
        }
        case mcap::OpCode::Message:
            if (kept_channels_.contains(message_channel_id(record, "TrackerSampleStreamWriter")))
            {
                append(samples_, record);
            }
            break;
        case mcap::OpCode::Footer:
            ended_ = true;
            break;
        default:
            // Header, DataEnd and the summary carry nothing a stream reader needs.
            break;
        }
    };
    const size_t consumed = for_each_record(std::span<const uint8_t>(pending_).subspan(offset), split);

    if (ended_)
    {
        pending_.clear();
        return;
    }
    pending_.erase(pending_.begin(), pending_.begin() + static_cast<std::ptrdiff_t>(offset + consumed));
}

// ============================================================================
// TrackerSampleStreamReader
// ============================================================================

void TrackerSampleStreamReader::add_setup(std::span<const uint8_t> records)
{
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t consumed =
        for_each_record(records,
                        [this](mcap::OpCode opcode, std::span<const uint8_t> record)
                        {
                            if (opcode == mcap::OpCode::Schema)
                            {
                                auto schema = std::make_shared<mcap::Schema>();
                                require_ok(mcap::McapReader::ParseSchema(as_mcap_record(record), schema.get()),
                                           "TrackerSampleStreamReader: bad Schema record");
                                schemas_[schema->id] = std::move(schema);
                            }
                            else if (opcode == mcap::OpCode::Channel)
                            {
                                mcap::Channel channel;
                                require_ok(mcap::McapReader::ParseChannel(as_mcap_record(record), &channel),
                                           "TrackerSampleStreamReader: bad Channel record");
                                channel_topics_[channel.id] = channel.topic;

                                const auto schema = schemas_.find(channel.schemaId);
                                pending_.channels[channel.topic] =
                                    RecordedSchemas::Channel{ channel.topic, channel.messageEncoding,
                                                              schema != schemas_.end() ? schema->second : nullptr };
                            }
                        });
    if (consumed != records.size())
    {
        throw std::runtime_error("TrackerSampleStreamReader: setup ends inside a record");
    }
}

bool TrackerSampleStreamReader::add_samples(std::span<const uint8_t> records, uint64_t tick_seq, int64_t tick_time_ns)
{
    if (for_each_record(records, [](mcap::OpCode, std::span<const uint8_t>) {}) != records.size())
    {
        throw std::runtime_error("TrackerSampleStreamReader: samples end inside a record");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (newest_tick_seq_ && tick_seq <= *newest_tick_seq_)
    {
        return false;
    }

    for_each_record(
        records,
        [this](mcap::OpCode opcode, std::span<const uint8_t> record)
        {
            if (opcode != mcap::OpCode::Message)
            {
                return;
            }
            mcap::Message message;
            require_ok(mcap::McapReader::ParseMessage(as_mcap_record(record), &message),
                       "TrackerSampleStreamReader: bad Message record");

            const auto topic = channel_topics_.find(message.channelId);
            if (topic == channel_topics_.end())
            {
                ++dropped_;
                return;
            }

            const auto* data = reinterpret_cast<const uint8_t*>(message.data);
            pending_.records[topic->second] =
                Record{ std::make_shared<const std::vector<uint8_t>>(data, data + message.dataSize), next_sequence_++ };
        });
    newest_tick_seq_ = tick_seq;
    newest_tick_time_ns_ = tick_time_ns;
    return true;
}

void TrackerSampleStreamReader::subscribe(std::string topic)
{
    std::lock_guard<std::mutex> lock(mutex_);
    subscriptions_.insert(std::move(topic));
}

std::vector<std::string> TrackerSampleStreamReader::subscribed_topics() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return std::vector<std::string>(subscriptions_.begin(), subscriptions_.end());
}

bool TrackerSampleStreamReader::snapshot(int64_t now_ns, int64_t max_age_ns)
{
    std::lock_guard<std::mutex> lock(mutex_);
    published_.channels = pending_.channels;
    if (!newest_tick_time_ns_ || now_ns - *newest_tick_time_ns_ > max_age_ns)
    {
        published_.records.clear();
        published_tick_time_ns_.reset();
        return false;
    }
    published_.records = pending_.records;
    published_tick_time_ns_ = newest_tick_time_ns_;
    return true;
}

std::optional<int64_t> TrackerSampleStreamReader::published_tick_time_ns() const
{
    return published_tick_time_ns_;
}

TrackerSampleStreamReader::Record TrackerSampleStreamReader::latest(std::string_view topic) const
{
    const auto it = published_.records.find(topic);
    return it != published_.records.end() ? it->second : Record{};
}

const RecordedSchemas::Channel* TrackerSampleStreamReader::channel(std::string_view topic) const
{
    const auto it = published_.channels.find(topic);
    return it != published_.channels.end() ? &it->second : nullptr;
}

std::optional<int64_t> TrackerSampleStreamReader::newest_tick_time_ns() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return newest_tick_time_ns_;
}

uint64_t TrackerSampleStreamReader::dropped_messages() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return dropped_;
}

} // namespace core
