// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/robot_state_publisher/robot_state_publisher.hpp"

#include <log_bridge/logger.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace core
{

namespace
{

// MCAP record opcodes. Every record is opcode u8, body length u64, body; integers are little-endian.
constexpr uint8_t kOpSchema = 0x03;
constexpr uint8_t kOpChannel = 0x04;
constexpr uint8_t kOpMessage = 0x05;

//! Opcode and body length precede every record body.
constexpr size_t kRecordPrefix = 1 + sizeof(uint64_t);
//! Message body before the data: channel id, sequence, log time, publish time.
constexpr size_t kMessageHeader = sizeof(uint16_t) + sizeof(uint32_t) + 2 * sizeof(uint64_t);

// Flags of RobotSamplesPrefix on the wire.
constexpr uint32_t kFlagClockValid = 1u << 0;
constexpr uint32_t kFlagClockTwoWay = 1u << 1;

std::shared_ptr<spdlog::logger> logger()
{
    return isaaccapture::Logger::get("isaaccapture.robot_state.RobotStatePublisher");
}

void append_le(std::vector<uint8_t>& out, uint64_t value, size_t width)
{
    for (size_t i = 0; i < width; ++i)
    {
        out.push_back(static_cast<uint8_t>(value >> (8 * i)));
    }
}

uint64_t read_le(const uint8_t* bytes, size_t width)
{
    uint64_t value = 0;
    for (size_t i = width; i-- > 0;)
    {
        value = (value << 8) | bytes[i];
    }
    return value;
}

void append(std::vector<uint8_t>& out, std::span<const uint8_t> bytes)
{
    out.insert(out.end(), bytes.begin(), bytes.end());
}

//! u32 length, then the bytes: MCAP's string and byte-array form.
void append_sized(std::vector<uint8_t>& out, std::string_view text)
{
    append_le(out, text.size(), sizeof(uint32_t));
    out.insert(out.end(), text.begin(), text.end());
}

void append_record_start(std::vector<uint8_t>& out, uint8_t opcode, uint64_t body_size)
{
    out.push_back(opcode);
    append_le(out, body_size, sizeof(uint64_t));
}

//! The shortest decimal text that reads back to the same double, whatever the process locale.
std::string number_text(double value)
{
    char buffer[32];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
    return std::string(buffer, result.ptr);
}

void append_schema(std::vector<uint8_t>& out, uint16_t id, const RobotSeriesDeclaration& declaration)
{
    const uint64_t texts = declaration.schema_name.size() + declaration.schema_encoding.size();
    const uint64_t body = sizeof(uint16_t) + 3 * sizeof(uint32_t) + texts + declaration.schema_data.size();
    append_record_start(out, kOpSchema, body);
    append_le(out, id, sizeof(uint16_t));
    append_sized(out, declaration.schema_name);
    append_sized(out, declaration.schema_encoding);
    append_le(out, declaration.schema_data.size(), sizeof(uint32_t));
    append(out, declaration.schema_data);
}

void append_channel(std::vector<uint8_t>& out, uint16_t id, const RobotSeriesDeclaration& declaration)
{
    // Keys in sorted order, as MCAP writes a metadata map.
    const std::pair<std::string_view, std::string> metadata[] = {
        { "delivery", declaration.in_order ? "in_order" : "latest" },
        { "rate_hz", number_text(declaration.rate_hz) },
        { "shape", declaration.shape },
        { "stale_after_ns", std::to_string(declaration.stale_after_ns) },
    };
    uint64_t metadata_size = 0;
    for (const auto& [key, value] : metadata)
    {
        metadata_size += sizeof(uint32_t) + key.size() + sizeof(uint32_t) + value.size();
    }

    const uint64_t texts = declaration.name.size() + declaration.message_encoding.size();
    const uint64_t body = 2 * sizeof(uint16_t) + 3 * sizeof(uint32_t) + texts + metadata_size;
    append_record_start(out, kOpChannel, body);
    append_le(out, id, sizeof(uint16_t));
    append_le(out, id, sizeof(uint16_t)); // schema id: one Schema per Channel, same id
    append_sized(out, declaration.name);
    append_sized(out, declaration.message_encoding);
    append_le(out, metadata_size, sizeof(uint32_t));
    for (const auto& [key, value] : metadata)
    {
        append_sized(out, key);
        append_sized(out, value);
    }
}

void append_message(std::vector<uint8_t>& out,
                    uint16_t channel_id,
                    uint32_t sequence,
                    int64_t log_time_ns,
                    int64_t publish_time_ns,
                    std::span<const uint8_t> data)
{
    out.reserve(out.size() + kRecordPrefix + kMessageHeader + data.size());
    append_record_start(out, kOpMessage, kMessageHeader + data.size());
    append_le(out, channel_id, sizeof(uint16_t));
    append_le(out, sequence, sizeof(uint32_t));
    append_le(out, static_cast<uint64_t>(log_time_ns), sizeof(uint64_t));
    append_le(out, static_cast<uint64_t>(publish_time_ns), sizeof(uint64_t));
    append(out, data);
}

struct MessageRef
{
    uint16_t channel_id;
    std::span<const uint8_t> record; //!< opcode, length and body
};

//! Adds the Message records that follow the prefix of `payload` to `out`.
void collect_messages(std::span<const uint8_t> payload, std::vector<MessageRef>& out)
{
    if (payload.size() < kRobotSamplesPrefixSize)
    {
        throw std::runtime_error("merge_robot_samples_payloads: payload of " + std::to_string(payload.size()) +
                                 " bytes is shorter than its prefix");
    }
    std::span<const uint8_t> rest = payload.subspan(kRobotSamplesPrefixSize);
    while (!rest.empty())
    {
        if (rest.size() < kRecordPrefix)
        {
            throw std::runtime_error("merge_robot_samples_payloads: records end inside a record");
        }
        const uint64_t body = read_le(rest.data() + 1, sizeof(uint64_t));
        if (body > rest.size() - kRecordPrefix)
        {
            throw std::runtime_error("merge_robot_samples_payloads: records end inside a record");
        }
        if (rest[0] != kOpMessage)
        {
            throw std::runtime_error("merge_robot_samples_payloads: a record other than a Message");
        }
        if (body < kMessageHeader)
        {
            throw std::runtime_error("merge_robot_samples_payloads: Message record too short");
        }
        const size_t size = kRecordPrefix + static_cast<size_t>(body);
        const auto channel_id = static_cast<uint16_t>(read_le(rest.data() + kRecordPrefix, sizeof(uint16_t)));
        out.push_back({ channel_id, rest.first(size) });
        rest = rest.subspan(size);
    }
}

std::invalid_argument refused(const std::string& reason)
{
    return std::invalid_argument("RobotStatePublisher: " + reason);
}

//! The entry of `name` in a publisher's series map. Throws std::invalid_argument for an undeclared series.
template <typename SeriesMap>
auto& declared(SeriesMap& series, std::string_view name)
{
    const auto it = series.find(name);
    if (it == series.end())
    {
        throw refused("series '" + std::string(name) + "' is not declared");
    }
    return it->second;
}

} // namespace

std::vector<uint8_t> encode_robot_samples_payload(const RobotSamplesPrefix& prefix, std::span<const uint8_t> records)
{
    std::vector<uint8_t> out;
    out.reserve(kRobotSamplesPrefixSize + records.size());
    append_le(out, prefix.publish_seq, sizeof(uint64_t));
    append_le(out, static_cast<uint64_t>(prefix.robot_send_time_ns), sizeof(uint64_t));
    append_le(out, static_cast<uint64_t>(prefix.robot_minus_server_ns), sizeof(uint64_t));
    uint32_t flags = 0;
    if (prefix.clock_valid)
    {
        flags |= kFlagClockValid;
    }
    if (prefix.clock_two_way)
    {
        flags |= kFlagClockTwoWay;
    }
    append_le(out, flags, sizeof(uint32_t));
    append_le(out, 0, sizeof(uint32_t)); // reserved
    append(out, records);
    return out;
}

std::vector<uint8_t> merge_robot_samples_payloads(std::vector<uint8_t> older, std::vector<uint8_t> newer)
{
    std::vector<MessageRef> messages;
    collect_messages(older, messages);
    collect_messages(newer, messages);

    std::unordered_map<uint16_t, size_t> last_index;
    for (size_t i = 0; i < messages.size(); ++i)
    {
        last_index[messages[i].channel_id] = i;
    }

    std::vector<uint8_t> merged(newer.begin(), newer.begin() + kRobotSamplesPrefixSize);
    for (size_t i = 0; i < messages.size(); ++i)
    {
        if (last_index.at(messages[i].channel_id) == i)
        {
            append(merged, messages[i].record);
        }
    }
    return merged;
}

// ============================================================================
// RobotStatePublisher
// ============================================================================

void RobotStatePublisher::declare_series(RobotSeriesDeclaration declaration)
{
    if (declaration.name.empty())
    {
        throw refused("a series needs a name");
    }
    const std::string quoted = "series '" + declaration.name + "'";
    if (series_.contains(declaration.name))
    {
        throw refused(quoted + " is declared twice");
    }
    if (series_.size() >= kMaxRobotSeries)
    {
        throw refused(quoted + " is above the limit of " + std::to_string(kMaxRobotSeries) + " series");
    }
    if (declaration.stale_after_ns < 0)
    {
        throw refused(quoted + ": stale_after_ns must be zero or above");
    }
    if (!std::isfinite(declaration.rate_hz) || declaration.rate_hz < 0.0)
    {
        throw refused(quoted + ": rate_hz must be a finite number, zero or above");
    }

    Series series;
    series.channel_id = static_cast<uint16_t>(series_.size() + 1);
    series.in_order = declaration.in_order;
    append_schema(setup_, series.channel_id, declaration);
    append_channel(setup_, series.channel_id, declaration);
    logger()->debug("Series '{}' declared as channel {} (shape '{}', {})", declaration.name, series.channel_id,
                    declaration.shape, declaration.in_order ? "in order" : "newest only");
    series_.emplace(std::move(declaration.name), std::move(series));
}

bool RobotStatePublisher::has_series(std::string_view name) const
{
    return series_.contains(name);
}

bool RobotStatePublisher::in_order(std::string_view name) const
{
    return declared(series_, name).in_order;
}

void RobotStatePublisher::publish(std::string_view name,
                                  std::span<const uint8_t> data,
                                  int64_t sample_time_ns,
                                  int64_t publish_time_ns)
{
    Series& series = declared(series_, name);
    if (data.size() > kMaxRobotRecordBytes)
    {
        throw refused("record of " + std::to_string(data.size()) + " bytes for series '" + std::string(name) +
                      "' is above the limit of " + std::to_string(kMaxRobotRecordBytes) + " bytes");
    }

    ++series.sequence;
    if (series.in_order)
    {
        append_message(in_order_records_, series.channel_id, series.sequence, sample_time_ns, publish_time_ns, data);
        return;
    }
    series.unsent.clear();
    append_message(series.unsent, series.channel_id, series.sequence, sample_time_ns, publish_time_ns, data);
    series.unsent_order = ++publish_count_;
}

std::vector<uint8_t> RobotStatePublisher::take_setup()
{
    return std::exchange(setup_, {});
}

std::vector<uint8_t> RobotStatePublisher::take_latest_records()
{
    // By publish call: each unsent record has its own unsent_order.
    std::vector<std::pair<uint64_t, Series*>> unsent;
    for (auto& [name, series] : series_)
    {
        if (!series.unsent.empty())
        {
            unsent.emplace_back(series.unsent_order, &series);
        }
    }
    std::sort(unsent.begin(), unsent.end());

    std::vector<uint8_t> records;
    for (const auto& [order, series] : unsent)
    {
        append(records, series->unsent);
        series->unsent.clear();
    }
    return records;
}

std::vector<uint8_t> RobotStatePublisher::take_in_order_records()
{
    return std::exchange(in_order_records_, {});
}

} // namespace core
