// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/robot_state_store/robot_state_store.hpp"

#include <log_bridge/logger.hpp>

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace core
{

namespace
{

constexpr size_t kMaxSeries = 256;
//! Largest record data and largest schema data taken.
constexpr size_t kMaxBlobBytes = size_t{ 1 } << 20;

//! MCAP record opcodes; each record is the opcode, a u64 body length and the body.
constexpr uint8_t kOpSchema = 0x03;
constexpr uint8_t kOpChannel = 0x04;
constexpr uint8_t kOpMessage = 0x05;
constexpr size_t kRecordPrefix = 1 + sizeof(uint64_t);

//! Prefix of a samples payload: publish_seq u64, robot_send_time_ns i64, robot_minus_server_ns i64, flags u32, u32.
constexpr size_t kSamplesPrefixSize = 32;
constexpr uint32_t kPrefixClockValid = 1u << 0;
constexpr uint32_t kPrefixClockTwoWay = 1u << 1;

constexpr uint8_t kSnapshotMagic[4] = { 'R', 'S', 'S', 'N' };
//! Rises with every change of the snapshot layout.
constexpr uint16_t kSnapshotVersion = 1;
constexpr uint16_t kSnapshotHasDeclarations = 1u << 0;
constexpr uint16_t kSnapshotConnected = 1u << 1;
constexpr uint8_t kSnapshotHasRecord = 1u << 0;
constexpr uint8_t kSnapshotClockValid = 1u << 1;
constexpr uint8_t kSnapshotClockTwoWay = 1u << 2;

std::shared_ptr<spdlog::logger> logger()
{
    return isaaccapture::Logger::get("isaaccapture.robot_state.RobotStateStore");
}

//! Reads little-endian fields off the front of a byte span. Throws std::runtime_error past its end.
class ByteReader
{
public:
    ByteReader(std::span<const uint8_t> bytes, const char* what) : m_bytes(bytes), m_what(what)
    {
    }

    size_t remaining() const
    {
        return m_bytes.size() - m_offset;
    }

    std::span<const uint8_t> take(size_t size)
    {
        if (size > remaining())
        {
            throw std::runtime_error(std::string("RobotStateStore: ") + m_what + " ends inside a field");
        }
        const auto out = m_bytes.subspan(m_offset, size);
        m_offset += size;
        return out;
    }

    template <typename T>
    T number()
    {
        using U = std::make_unsigned_t<T>;
        const auto bytes = take(sizeof(T));
        U bits = 0;
        for (size_t i = 0; i < sizeof(T); ++i)
        {
            bits = static_cast<U>(bits | static_cast<U>(static_cast<U>(bytes[i]) << (8 * i)));
        }
        return static_cast<T>(bits);
    }

    //! A u32 length and that many bytes.
    std::span<const uint8_t> blob()
    {
        return take(number<uint32_t>());
    }

    std::string text()
    {
        const auto bytes = blob();
        return std::string(bytes.begin(), bytes.end());
    }

    std::span<const uint8_t> rest()
    {
        return take(remaining());
    }

private:
    std::span<const uint8_t> m_bytes;
    const char* m_what;
    size_t m_offset = 0;
};

template <typename T>
void put_number(std::vector<uint8_t>& out, T value)
{
    using U = std::make_unsigned_t<T>;
    const U bits = static_cast<U>(value);
    for (size_t i = 0; i < sizeof(T); ++i)
    {
        out.push_back(static_cast<uint8_t>(bits >> (8 * i)));
    }
}

void put_blob(std::vector<uint8_t>& out, std::span<const uint8_t> bytes)
{
    put_number<uint32_t>(out, static_cast<uint32_t>(bytes.size()));
    out.insert(out.end(), bytes.begin(), bytes.end());
}

void put_text(std::vector<uint8_t>& out, std::string_view text)
{
    put_number<uint32_t>(out, static_cast<uint32_t>(text.size()));
    out.insert(out.end(), text.begin(), text.end());
}

struct McapRecord
{
    uint8_t opcode = 0;
    std::span<const uint8_t> body;
};

//! Throws std::runtime_error when `bytes` end inside a record.
std::vector<McapRecord> split_records(std::span<const uint8_t> bytes, const char* what)
{
    std::vector<McapRecord> records;
    size_t offset = 0;
    while (offset < bytes.size())
    {
        ByteReader prefix(bytes.subspan(offset, std::min(kRecordPrefix, bytes.size() - offset)), what);
        const uint8_t opcode = prefix.number<uint8_t>();
        const uint64_t body = prefix.number<uint64_t>();
        if (body > bytes.size() - offset - kRecordPrefix)
        {
            throw std::runtime_error(std::string("RobotStateStore: ") + what + " ends inside a record");
        }
        records.push_back({ opcode, bytes.subspan(offset + kRecordPrefix, static_cast<size_t>(body)) });
        offset += kRecordPrefix + static_cast<size_t>(body);
    }
    return records;
}

struct SchemaDeclaration
{
    uint16_t id = 0;
    std::string name;
    std::string encoding;
    std::vector<uint8_t> data;
};

struct ChannelDeclaration
{
    uint16_t id = 0;
    uint16_t schema_id = 0;
    RobotSeriesInfo info; //!< without the schema's fields
};

//! A Schema or a Channel record of a setup payload, in payload order.
struct SetupItem
{
    bool is_schema = false;
    SchemaDeclaration schema;
    ChannelDeclaration channel;
};

//! Decimal integer text, zero or above; 0 for anything else.
int64_t parse_stale_after_ns(const std::string& text)
{
    int64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || value < 0)
    {
        return 0;
    }
    return value;
}

//! Decimal number text, finite and zero or above; 0 for anything else. Reads the same in every locale.
double parse_rate_hz(const std::string& text)
{
    double value = 0.0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || !std::isfinite(value) || value < 0.0)
    {
        return 0.0;
    }
    return value;
}

SchemaDeclaration parse_schema(std::span<const uint8_t> body)
{
    ByteReader reader(body, "a Schema record");
    SchemaDeclaration schema;
    schema.id = reader.number<uint16_t>();
    schema.name = reader.text();
    schema.encoding = reader.text();
    const auto data = reader.blob();
    if (schema.id == 0)
    {
        throw std::runtime_error("RobotStateStore: a Schema record with id 0");
    }
    if (data.size() > kMaxBlobBytes)
    {
        throw std::runtime_error("RobotStateStore: schema '" + schema.name + "' of " + std::to_string(data.size()) +
                                 " bytes is above the limit of " + std::to_string(kMaxBlobBytes));
    }
    schema.data.assign(data.begin(), data.end());
    return schema;
}

ChannelDeclaration parse_channel(std::span<const uint8_t> body)
{
    ByteReader reader(body, "a Channel record");
    ChannelDeclaration channel;
    channel.id = reader.number<uint16_t>();
    channel.schema_id = reader.number<uint16_t>();
    channel.info.name = reader.text();
    channel.info.message_encoding = reader.text();
    if (channel.info.name.empty())
    {
        throw std::runtime_error("RobotStateStore: a Channel record without a topic");
    }

    // Unknown keys are ignored; a missing or unreadable value keeps the default.
    ByteReader metadata(reader.blob(), "the metadata of a Channel record");
    while (metadata.remaining() > 0)
    {
        const std::string key = metadata.text();
        const std::string value = metadata.text();
        if (key == "shape")
        {
            channel.info.shape = value;
        }
        else if (key == "stale_after_ns")
        {
            channel.info.stale_after_ns = parse_stale_after_ns(value);
        }
        else if (key == "delivery")
        {
            channel.info.in_order = value == "in_order";
        }
        else if (key == "rate_hz")
        {
            channel.info.rate_hz = parse_rate_hz(value);
        }
    }
    return channel;
}

std::vector<SetupItem> parse_setup(std::span<const uint8_t> payload)
{
    std::vector<SetupItem> items;
    for (const McapRecord& record : split_records(payload, "the setup payload"))
    {
        SetupItem item;
        if (record.opcode == kOpSchema)
        {
            item.is_schema = true;
            item.schema = parse_schema(record.body);
        }
        else if (record.opcode == kOpChannel)
        {
            item.channel = parse_channel(record.body);
        }
        else
        {
            throw std::runtime_error("RobotStateStore: the setup payload holds a record other than Schema or Channel");
        }
        items.push_back(std::move(item));
    }
    return items;
}

bool same_declaration(const RobotSeriesInfo& a, const RobotSeriesInfo& b)
{
    return a.name == b.name && a.shape == b.shape && a.schema_name == b.schema_name &&
           a.schema_encoding == b.schema_encoding && a.schema_data == b.schema_data &&
           a.message_encoding == b.message_encoding && a.stale_after_ns == b.stale_after_ns &&
           a.in_order == b.in_order && a.rate_hz == b.rate_hz;
}

//! a - b, held at the ends of the range instead of overflowing.
int64_t saturating_sub(int64_t a, int64_t b)
{
    constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
    constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
    if (b > 0 && a < kMin + b)
    {
        return kMin;
    }
    if (b < 0 && a > kMax + b)
    {
        return kMax;
    }
    return a - b;
}

std::shared_ptr<const std::vector<uint8_t>> copy_bytes(std::span<const uint8_t> bytes)
{
    return std::make_shared<const std::vector<uint8_t>>(bytes.begin(), bytes.end());
}

} // namespace

// ============================================================================
// Impl
// ============================================================================

struct RobotStateStore::Impl
{
    struct Series
    {
        RobotSeriesInfo info;
        std::shared_ptr<const std::vector<uint8_t>> data; //!< null: no record
        uint64_t sequence = 0; //!< records taken on this series in this robot session
        int64_t sample_time_ns = 0;
        int64_t publish_time_ns = 0;
        int64_t receive_time_ns = 0;
        RobotClockSample clock;
        std::optional<uint64_t> payload_seq; //!< publish_seq of the samples payload the record came in
    };

    struct Schema
    {
        std::string name;
        std::string encoding;
        std::vector<uint8_t> data;
    };

    //! Declares `info` or replaces the series of its name; a changed declaration drops the record.
    size_t declare(RobotSeriesInfo info)
    {
        const auto found = index.find(info.name);
        if (found == index.end())
        {
            const size_t position = series.size();
            index.emplace(info.name, position);
            series.emplace_back().info = std::move(info);
            ++setup_generation;
            ++generation;
            return position;
        }
        Series& held = series[found->second];
        if (!same_declaration(held.info, info))
        {
            held.info = std::move(info);
            held.data.reset();
            held.payload_seq.reset();
            ++setup_generation;
            ++generation;
        }
        return found->second;
    }

    void put(Series& held,
             std::shared_ptr<const std::vector<uint8_t>> data,
             int64_t sample_time_ns,
             int64_t publish_time_ns,
             int64_t receive_time_ns,
             const RobotClockSample& clock)
    {
        held.data = std::move(data);
        ++held.sequence;
        held.sample_time_ns = sample_time_ns;
        held.publish_time_ns = publish_time_ns;
        held.receive_time_ns = receive_time_ns;
        held.clock = clock;
        ++generation;
    }

    mutable std::mutex mutex;
    bool connected = false;
    uint64_t session_count = 0;
    uint64_t generation = 0;
    uint64_t setup_generation = 1;
    uint64_t dropped = 0;
    bool drop_logged = false; //!< one warning per robot session
    std::vector<Series> series; //!< declaration order
    std::map<std::string, size_t, std::less<>> index; //!< series name -> position in `series`
    std::unordered_map<uint16_t, Schema> schemas; //!< as the setup payloads of this robot session declared them
    std::unordered_map<uint16_t, size_t> channel_series; //!< channel id -> position in `series`
};

// ============================================================================
// RobotStateStore
// ============================================================================

RobotStateStore::RobotStateStore() : m_impl(std::make_unique<Impl>())
{
}

RobotStateStore::~RobotStateStore() = default;

void RobotStateStore::begin_session()
{
    uint64_t session = 0;
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        m_impl->series.clear();
        m_impl->index.clear();
        m_impl->schemas.clear();
        m_impl->channel_series.clear();
        m_impl->drop_logged = false;
        m_impl->connected = true;
        session = ++m_impl->session_count;
        ++m_impl->setup_generation;
        ++m_impl->generation;
    }
    logger()->debug("robot state session {} began", session);
}

void RobotStateStore::end_session()
{
    uint64_t session = 0;
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        if (!m_impl->connected)
        {
            return;
        }
        m_impl->connected = false;
        session = m_impl->session_count;
        ++m_impl->generation;
    }
    logger()->debug("robot state session {} ended", session);
}

void RobotStateStore::add_setup(std::span<const uint8_t> payload)
{
    std::vector<SetupItem> items = parse_setup(payload);

    std::lock_guard<std::mutex> lock(m_impl->mutex);
    Impl& impl = *m_impl;

    // Checked in full before anything changes.
    std::unordered_map<uint16_t, const SchemaDeclaration*> new_schemas;
    std::set<std::string_view> new_names;
    size_t schema_count = impl.schemas.size();
    for (SetupItem& item : items)
    {
        if (item.is_schema)
        {
            if (!impl.schemas.contains(item.schema.id) && !new_schemas.contains(item.schema.id))
            {
                ++schema_count;
            }
            new_schemas[item.schema.id] = &item.schema;
            continue;
        }
        ChannelDeclaration& channel = item.channel;
        if (channel.schema_id != 0)
        {
            if (const auto fresh = new_schemas.find(channel.schema_id); fresh != new_schemas.end())
            {
                channel.info.schema_name = fresh->second->name;
                channel.info.schema_encoding = fresh->second->encoding;
                channel.info.schema_data = fresh->second->data;
            }
            else if (const auto held = impl.schemas.find(channel.schema_id); held != impl.schemas.end())
            {
                channel.info.schema_name = held->second.name;
                channel.info.schema_encoding = held->second.encoding;
                channel.info.schema_data = held->second.data;
            }
            else
            {
                throw std::runtime_error("RobotStateStore: the Channel of series '" + channel.info.name +
                                         "' names the undeclared Schema " + std::to_string(channel.schema_id));
            }
        }
        if (!impl.index.contains(channel.info.name))
        {
            new_names.insert(channel.info.name);
        }
    }
    if (impl.series.size() + new_names.size() > kMaxSeries || schema_count > kMaxSeries)
    {
        throw std::runtime_error("RobotStateStore: the setup payload would make more than " +
                                 std::to_string(kMaxSeries) + " series or schemas");
    }

    for (SetupItem& item : items)
    {
        if (item.is_schema)
        {
            impl.schemas[item.schema.id] = Impl::Schema{ std::move(item.schema.name), std::move(item.schema.encoding),
                                                         std::move(item.schema.data) };
        }
        else
        {
            impl.channel_series[item.channel.id] = impl.declare(std::move(item.channel.info));
        }
    }
}

void RobotStateStore::add_samples(std::span<const uint8_t> payload, int64_t receive_time_ns)
{
    if (payload.size() < kSamplesPrefixSize)
    {
        throw std::runtime_error("RobotStateStore: a samples payload of " + std::to_string(payload.size()) +
                                 " bytes is shorter than its prefix");
    }
    ByteReader prefix(payload.first(kSamplesPrefixSize), "the prefix of a samples payload");
    const uint64_t publish_seq = prefix.number<uint64_t>();
    prefix.number<int64_t>(); // robot_send_time_ns
    RobotClockSample clock;
    clock.robot_minus_server_ns = prefix.number<int64_t>();
    const uint32_t flags = prefix.number<uint32_t>();
    clock.valid = (flags & kPrefixClockValid) != 0;
    clock.two_way = clock.valid && (flags & kPrefixClockTwoWay) != 0;

    struct Message
    {
        uint16_t channel_id = 0;
        int64_t sample_time_ns = 0;
        int64_t publish_time_ns = 0;
        std::shared_ptr<const std::vector<uint8_t>> data;
    };
    std::vector<Message> messages;
    for (const McapRecord& record : split_records(payload.subspan(kSamplesPrefixSize), "a samples payload"))
    {
        if (record.opcode != kOpMessage)
        {
            throw std::runtime_error("RobotStateStore: a samples payload holds a record other than a Message");
        }
        // Message body: channel id, sequence, log time, publish time, then the data.
        ByteReader body(record.body, "a Message record");
        Message message;
        message.channel_id = body.number<uint16_t>();
        body.number<uint32_t>();
        message.sample_time_ns = static_cast<int64_t>(body.number<uint64_t>());
        message.publish_time_ns = static_cast<int64_t>(body.number<uint64_t>());
        const auto data = body.rest();
        if (data.size() > kMaxBlobBytes)
        {
            throw std::runtime_error("RobotStateStore: a record of " + std::to_string(data.size()) +
                                     " bytes is above the limit of " + std::to_string(kMaxBlobBytes));
        }
        message.data = copy_bytes(data);
        messages.push_back(std::move(message));
    }

    std::string dropped_reason;
    {
        std::lock_guard<std::mutex> lock(m_impl->mutex);
        Impl& impl = *m_impl;
        // A payload may hold several records of one in-order series: only the first is checked against the
        // payload the series' record came in.
        std::vector<bool> taken(impl.series.size(), false);
        for (Message& message : messages)
        {
            const auto channel = impl.channel_series.find(message.channel_id);
            const bool declared = channel != impl.channel_series.end();
            if (declared)
            {
                Impl::Series& held = impl.series[channel->second];
                if (!held.payload_seq || *held.payload_seq < publish_seq || taken[channel->second])
                {
                    taken[channel->second] = true;
                    held.payload_seq = publish_seq;
                    impl.put(held, std::move(message.data), message.sample_time_ns, message.publish_time_ns,
                             receive_time_ns, clock);
                    continue;
                }
            }

            ++impl.dropped;
            ++impl.generation;
            if (!impl.drop_logged)
            {
                impl.drop_logged = true;
                if (declared)
                {
                    const Impl::Series& held = impl.series[channel->second];
                    dropped_reason = "series '" + held.info.name + "' holds a record of payload " +
                                     std::to_string(*held.payload_seq) + ", this one came in payload " +
                                     std::to_string(publish_seq);
                }
                else
                {
                    dropped_reason = "channel " + std::to_string(message.channel_id) + " is not declared";
                }
            }
        }
    }
    if (!dropped_reason.empty())
    {
        logger()->warn(
            "dropped a robot state record: {} (further drops of this robot session are only counted)", dropped_reason);
    }
}

void RobotStateStore::declare_series(const RobotSeriesInfo& info)
{
    if (info.name.empty())
    {
        throw std::invalid_argument("RobotStateStore: a series needs a name");
    }
    if (info.schema_data.size() > kMaxBlobBytes)
    {
        throw std::invalid_argument("RobotStateStore: series '" + info.name + "': schema data of " +
                                    std::to_string(info.schema_data.size()) + " bytes is above the limit of " +
                                    std::to_string(kMaxBlobBytes));
    }
    if (info.stale_after_ns < 0 || !std::isfinite(info.rate_hz) || info.rate_hz < 0.0)
    {
        throw std::invalid_argument("RobotStateStore: series '" + info.name +
                                    "': stale_after_ns and rate_hz must be zero or above");
    }

    std::lock_guard<std::mutex> lock(m_impl->mutex);
    if (!m_impl->index.contains(info.name) && m_impl->series.size() >= kMaxSeries)
    {
        throw std::invalid_argument("RobotStateStore: series '" + info.name + "' would be more than " +
                                    std::to_string(kMaxSeries) + " series");
    }
    m_impl->declare(info);
}

void RobotStateStore::put_record(std::string_view name,
                                 std::span<const uint8_t> data,
                                 int64_t sample_time_ns,
                                 int64_t publish_time_ns,
                                 int64_t receive_time_ns,
                                 const RobotClockSample& clock)
{
    if (data.size() > kMaxBlobBytes)
    {
        throw std::invalid_argument("RobotStateStore: series '" + std::string(name) + "': a record of " +
                                    std::to_string(data.size()) + " bytes is above the limit of " +
                                    std::to_string(kMaxBlobBytes));
    }
    auto bytes = copy_bytes(data);
    RobotClockSample taken = clock;
    taken.two_way = clock.valid && clock.two_way;

    std::lock_guard<std::mutex> lock(m_impl->mutex);
    const auto found = m_impl->index.find(name);
    if (found == m_impl->index.end())
    {
        throw std::invalid_argument("RobotStateStore: series '" + std::string(name) + "' is not declared");
    }
    m_impl->put(m_impl->series[found->second], std::move(bytes), sample_time_ns, publish_time_ns, receive_time_ns, taken);
}

bool RobotStateStore::connected() const
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->connected;
}

uint64_t RobotStateStore::session_count() const
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->session_count;
}

uint64_t RobotStateStore::generation() const
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->generation;
}

uint64_t RobotStateStore::setup_generation() const
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->setup_generation;
}

uint64_t RobotStateStore::dropped_records() const
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    return m_impl->dropped;
}

std::vector<RobotSeriesInfo> RobotStateStore::series() const
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    std::vector<RobotSeriesInfo> out;
    out.reserve(m_impl->series.size());
    for (const Impl::Series& held : m_impl->series)
    {
        out.push_back(held.info);
    }
    return out;
}

std::optional<RobotSeriesInfo> RobotStateStore::series_info(std::string_view name) const
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    const auto found = m_impl->index.find(name);
    if (found == m_impl->index.end())
    {
        return std::nullopt;
    }
    return m_impl->series[found->second].info;
}

RobotRecord RobotStateStore::latest(std::string_view name, int64_t now_ns) const
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    const auto found = m_impl->index.find(name);
    if (found == m_impl->index.end() || !m_impl->series[found->second].data)
    {
        return RobotRecord{};
    }
    const Impl::Series& held = m_impl->series[found->second];

    RobotRecord record;
    record.data = held.data;
    record.sequence = held.sequence;
    record.sample_time_ns = held.sample_time_ns;
    record.publish_time_ns = held.publish_time_ns;
    record.receive_time_ns = held.receive_time_ns;
    record.receive_age_ns = std::max<int64_t>(0, saturating_sub(now_ns, held.receive_time_ns));
    if (held.clock.valid)
    {
        // The sample time on this machine's clock is the robot's minus (robot clock - this clock). A reading is
        // never younger than the time since it arrived, so a time stamp ahead of the robot's clock cannot keep it
        // fresh.
        const int64_t sample_time_here_ns = saturating_sub(held.sample_time_ns, held.clock.robot_minus_server_ns);
        record.age_ns = std::max(record.receive_age_ns, saturating_sub(now_ns, sample_time_here_ns));
    }
    record.clock_two_way = held.clock.two_way;
    const int64_t age_ns = record.age_ns ? *record.age_ns : record.receive_age_ns;
    record.fresh = held.info.stale_after_ns == 0 || age_ns <= held.info.stale_after_ns;
    return record;
}

// Snapshot layout, little-endian; text and bytes are a u32 length and the bytes:
//   magic[4], version u16, flags u16, session_count u64, generation u64, setup_generation u64, dropped u64,
//   series count u32, then per series: name, the declaration when the flags say so (shape, schema name, schema
//   encoding, schema data, message encoding, stale_after_ns i64, in_order u8, rate_hz as its u64 bits),
//   sequence u64, record flags u8, and with a record: sample, publish and receive time i64,
//   robot_minus_server_ns i64, data.
std::vector<uint8_t> RobotStateStore::export_snapshot(uint64_t known_setup_generation) const
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    const Impl& impl = *m_impl;
    const bool declarations = known_setup_generation != impl.setup_generation;

    std::vector<uint8_t> out;
    out.insert(out.end(), std::begin(kSnapshotMagic), std::end(kSnapshotMagic));
    put_number<uint16_t>(out, kSnapshotVersion);
    put_number<uint16_t>(out, static_cast<uint16_t>((declarations ? kSnapshotHasDeclarations : 0) |
                                                    (impl.connected ? kSnapshotConnected : 0)));
    put_number<uint64_t>(out, impl.session_count);
    put_number<uint64_t>(out, impl.generation);
    put_number<uint64_t>(out, impl.setup_generation);
    put_number<uint64_t>(out, impl.dropped);
    put_number<uint32_t>(out, static_cast<uint32_t>(impl.series.size()));
    for (const Impl::Series& held : impl.series)
    {
        put_text(out, held.info.name);
        if (declarations)
        {
            put_text(out, held.info.shape);
            put_text(out, held.info.schema_name);
            put_text(out, held.info.schema_encoding);
            put_blob(out, held.info.schema_data);
            put_text(out, held.info.message_encoding);
            put_number<int64_t>(out, held.info.stale_after_ns);
            put_number<uint8_t>(out, held.info.in_order ? 1 : 0);
            put_number<uint64_t>(out, std::bit_cast<uint64_t>(held.info.rate_hz));
        }
        put_number<uint64_t>(out, held.sequence);
        put_number<uint8_t>(out, static_cast<uint8_t>((held.data ? kSnapshotHasRecord : 0) |
                                                      (held.clock.valid ? kSnapshotClockValid : 0) |
                                                      (held.clock.two_way ? kSnapshotClockTwoWay : 0)));
        if (held.data)
        {
            put_number<int64_t>(out, held.sample_time_ns);
            put_number<int64_t>(out, held.publish_time_ns);
            put_number<int64_t>(out, held.receive_time_ns);
            put_number<int64_t>(out, held.clock.robot_minus_server_ns);
            put_blob(out, *held.data);
        }
    }
    return out;
}

void RobotStateStore::import_snapshot(std::span<const uint8_t> snapshot)
{
    static constexpr const char* kNotOurs = "RobotStateStore: not a snapshot this build of the store made";

    struct Entry
    {
        Impl::Series series;
        std::span<const uint8_t> data; //!< inside `snapshot`
        bool has_record = false;
    };

    if (snapshot.size() < sizeof(kSnapshotMagic) + sizeof(uint16_t) ||
        !std::equal(std::begin(kSnapshotMagic), std::end(kSnapshotMagic), snapshot.begin()))
    {
        throw std::runtime_error(kNotOurs);
    }
    ByteReader reader(snapshot.subspan(sizeof(kSnapshotMagic)), "a snapshot");
    if (reader.number<uint16_t>() != kSnapshotVersion)
    {
        throw std::runtime_error(kNotOurs);
    }
    const uint16_t flags = reader.number<uint16_t>();
    const bool declarations = (flags & kSnapshotHasDeclarations) != 0;
    const bool connected = (flags & kSnapshotConnected) != 0;
    const uint64_t session_count = reader.number<uint64_t>();
    const uint64_t generation = reader.number<uint64_t>();
    const uint64_t setup_generation = reader.number<uint64_t>();
    const uint64_t dropped = reader.number<uint64_t>();
    const uint32_t count = reader.number<uint32_t>();
    if (count > kMaxSeries)
    {
        throw std::runtime_error(kNotOurs);
    }

    std::vector<Entry> entries(count);
    std::map<std::string, size_t, std::less<>> index;
    for (size_t position = 0; position < entries.size(); ++position)
    {
        Entry& entry = entries[position];
        RobotSeriesInfo& info = entry.series.info;
        info.name = reader.text();
        if (declarations)
        {
            info.shape = reader.text();
            info.schema_name = reader.text();
            info.schema_encoding = reader.text();
            const auto schema_data = reader.blob();
            info.schema_data.assign(schema_data.begin(), schema_data.end());
            info.message_encoding = reader.text();
            info.stale_after_ns = reader.number<int64_t>();
            info.in_order = reader.number<uint8_t>() != 0;
            info.rate_hz = std::bit_cast<double>(reader.number<uint64_t>());
        }
        entry.series.sequence = reader.number<uint64_t>();
        const uint8_t record_flags = reader.number<uint8_t>();
        entry.has_record = (record_flags & kSnapshotHasRecord) != 0;
        entry.series.clock.valid = (record_flags & kSnapshotClockValid) != 0;
        entry.series.clock.two_way = (record_flags & kSnapshotClockTwoWay) != 0;
        if (entry.has_record)
        {
            entry.series.sample_time_ns = reader.number<int64_t>();
            entry.series.publish_time_ns = reader.number<int64_t>();
            entry.series.receive_time_ns = reader.number<int64_t>();
            entry.series.clock.robot_minus_server_ns = reader.number<int64_t>();
            entry.data = reader.blob();
        }
        if (!index.emplace(info.name, position).second)
        {
            throw std::runtime_error(kNotOurs);
        }
    }
    if (reader.remaining() != 0)
    {
        throw std::runtime_error(kNotOurs);
    }

    std::lock_guard<std::mutex> lock(m_impl->mutex);
    Impl& impl = *m_impl;
    if (!declarations)
    {
        if (setup_generation != impl.setup_generation)
        {
            throw std::runtime_error("RobotStateStore: a snapshot without declarations of setup generation " +
                                     std::to_string(setup_generation) + " does not fit this store's " +
                                     std::to_string(impl.setup_generation));
        }
        if (entries.size() != impl.series.size())
        {
            throw std::runtime_error(kNotOurs);
        }
        for (size_t position = 0; position < entries.size(); ++position)
        {
            if (entries[position].series.info.name != impl.series[position].info.name)
            {
                throw std::runtime_error(kNotOurs);
            }
        }
    }

    const bool same_session = !declarations && session_count == impl.session_count;
    std::vector<Impl::Series> series;
    series.reserve(entries.size());
    for (size_t position = 0; position < entries.size(); ++position)
    {
        Entry& entry = entries[position];
        if (!declarations)
        {
            entry.series.info = std::move(impl.series[position].info);
        }
        if (entry.has_record)
        {
            // An unchanged record keeps its bytes instead of a new copy.
            const Impl::Series* held = same_session ? &impl.series[position] : nullptr;
            entry.series.data =
                held && held->data && held->sequence == entry.series.sequence ? held->data : copy_bytes(entry.data);
        }
        series.push_back(std::move(entry.series));
    }

    impl.series = std::move(series);
    if (declarations)
    {
        impl.index = std::move(index);
    }
    impl.schemas.clear();
    impl.channel_series.clear();
    impl.connected = connected;
    impl.session_count = session_count;
    impl.generation = generation;
    impl.setup_generation = setup_generation;
    impl.dropped = dropped;
}

} // namespace core
