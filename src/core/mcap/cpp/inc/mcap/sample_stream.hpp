// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "recorded_schemas.hpp"

#include <mcap/writer.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace core
{

/*!
 * @brief Splits an unchunked MCAP recording into setup records and the samples of each tick.
 *
 * A sample stream carries tracker samples in the MCAP record format, with no file around them
 * and no history: the receiver keeps only the newest sample per topic. A DeviceIOSession records
 * through an McapWriter opened on this writable with writer_options(), so each record reaches
 * handleWrite() as the writer emits it. Schema and Channel records become setup; the Message
 * records written between two take_samples() calls are one tick's samples. Not thread-safe: the session's
 * update loop owns it.
 */
class TrackerSampleStreamWriter : public mcap::IWritable
{
public:
    //! Options to open the McapWriter with, so that no record is held back in a chunk.
    static mcap::McapWriterOptions writer_options();

    //! Keeps only messages on these topics; empty keeps every topic. Set before the first write.
    void set_topic_filter(const std::vector<std::string>& topics);

    void end() override;
    uint64_t size() const override;

    //! Schema and Channel records written since the last call, in write order.
    std::vector<uint8_t> take_setup();
    //! Message records written since the last call.
    std::vector<uint8_t> take_samples();

    /*!
     * @brief Adds a Message record on `topic` to the samples, after those the session wrote.
     * @return False, adding nothing, when `topic` has no kept channel: none declared yet, or filtered out.
     */
    bool add_record(std::string_view topic, std::span<const uint8_t> data, uint64_t log_time_ns);

protected:
    void handleWrite(const std::byte* data, uint64_t size) override;

private:
    void consume_records();

    std::vector<uint8_t> pending_;
    bool magic_seen_ = false;
    bool ended_ = false;
    uint64_t size_ = 0;
    std::set<std::string, std::less<>> topic_filter_;
    std::set<uint16_t> kept_channels_;
    std::map<std::string, uint16_t, std::less<>> kept_topics_;
    std::vector<uint8_t> setup_;
    std::vector<uint8_t> samples_;
};

/*!
 * @brief Merges the Message records of two ticks into one, keeping the last record per channel.
 *
 * Each tick's samples are a delta, so a channel that `newer` does not write keeps its record from `older`.
 * The result reads the same through TrackerSampleStreamReader::add_samples() as the two in order.
 * @throws std::runtime_error when either ends inside a record or holds a record other than
 *         a Message.
 */
std::vector<uint8_t> merge_samples_records(std::span<const uint8_t> older, std::span<const uint8_t> newer);

/*!
 * @brief The receiving end of a sample stream: the newest record per topic.
 *
 * add_setup() and add_samples() take what TrackerSampleStreamWriter produced and may run on a receive
 * thread. snapshot() publishes what has arrived to the session thread, which reads it through
 * latest() and channel(), so every tracker read during one session update sees the same state.
 * A topic keeps its newest record across ticks, as a tracker keeps its last sample until a new
 * one arrives.
 */
class TrackerSampleStreamReader
{
public:
    struct Record
    {
        //! Null when the topic has no record.
        std::shared_ptr<const std::vector<uint8_t>> bytes;
        //! Changes whenever a new record arrives on the topic.
        uint64_t sequence = 0;
    };

    /*!
     * @brief Takes Schema and Channel records.
     * @throws std::runtime_error when the bytes are not a sequence of complete records.
     */
    void add_setup(std::span<const uint8_t> records);

    /*!
     * @brief Takes the Message records of one tick.
     * @param tick_seq The sender's tick counter. Ticks are ordered by it alone.
     * @param tick_time_ns The tick time, mapped onto this machine's monotonic clock. Used only
     *                     for the tick age.
     * @return False, ignoring the records, when `tick_seq` is not newer than every tick taken.
     * @throws std::runtime_error when the bytes are not a sequence of complete records.
     */
    bool add_samples(std::span<const uint8_t> records, uint64_t tick_seq, int64_t tick_time_ns);

    //! Adds `topic` to subscribed_topics(). Thread-safe.
    void subscribe(std::string topic);

    //! The topics the stream's readers asked for, sorted and unique. Thread-safe.
    std::vector<std::string> subscribed_topics() const;

    /*!
     * @brief Publishes everything received so far to latest() and channel().
     * @return False, publishing no records, when the newest tick taken is stale: its tick time is
     *         more than max_age_ns before now_ns.
     */
    bool snapshot(int64_t now_ns, int64_t max_age_ns);

    //! Tick time of the records the last snapshot() published, or nullopt when it published none.
    //! Session thread only.
    std::optional<int64_t> published_tick_time_ns() const;

    //! The newest record on `topic` as of the last snapshot(). Session thread only.
    Record latest(std::string_view topic) const;

    //! What `topic` was declared with, or null before its Channel record arrived. Session thread only.
    const RecordedSchemas::Channel* channel(std::string_view topic) const;

    //! Tick time of the newest tick taken, by tick_seq, if any.
    std::optional<int64_t> newest_tick_time_ns() const;

    //! Messages dropped because their channel was never declared.
    uint64_t dropped_messages() const;

private:
    struct State
    {
        std::map<std::string, RecordedSchemas::Channel, std::less<>> channels;
        std::map<std::string, Record, std::less<>> records;
    };

    mutable std::mutex mutex_;
    std::unordered_map<uint16_t, mcap::SchemaPtr> schemas_;
    std::unordered_map<uint16_t, std::string> channel_topics_;
    State pending_;
    std::optional<uint64_t> newest_tick_seq_;
    std::optional<int64_t> newest_tick_time_ns_;
    uint64_t next_sequence_ = 1;
    uint64_t dropped_ = 0;
    std::set<std::string, std::less<>> subscriptions_;

    State published_;
    std::optional<int64_t> published_tick_time_ns_;
};

} // namespace core
