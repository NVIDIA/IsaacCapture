// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace core
{

//! Series one publisher takes.
constexpr size_t kMaxRobotSeries = 256;
//! Largest record one publish() takes.
constexpr size_t kMaxRobotRecordBytes = 1u << 20;
//! Bytes of a RobotSamplesPrefix on the wire.
constexpr size_t kRobotSamplesPrefixSize = 32;

//! One data series a robot sends: its topic and what its records are. The record bytes are opaque here.
struct RobotSeriesDeclaration
{
    std::string name; //!< topic
    std::string shape; //!< channel metadata "shape"
    std::string schema_name;
    std::string schema_encoding; //!< "flatbuffer", "jsonschema" or ""
    std::vector<uint8_t> schema_data;
    std::string message_encoding; //!< "flatbuffer" or "json"
    int64_t stale_after_ns = 0; //!< 0: never stale
    bool in_order = false; //!< true: every record is delivered, in ROBOT_STATE_RECORDS
    double rate_hz = 0.0; //!< 0: not stated
};

//! What precedes the Message records of a ROBOT_STATE_SAMPLES or ROBOT_STATE_RECORDS payload.
struct RobotSamplesPrefix
{
    uint64_t publish_seq = 0; //!< payloads of this kind made by this client, from 1
    int64_t robot_send_time_ns = 0; //!< robot CLOCK_MONOTONIC when the payload was made
    int64_t robot_minus_server_ns = 0; //!< the robot link's clock offset estimate (local - remote)
    bool clock_valid = false;
    bool clock_two_way = false;
};

//! Prefix followed by `records`.
std::vector<uint8_t> encode_robot_samples_payload(const RobotSamplesPrefix& prefix, std::span<const uint8_t> records);

/*!
 * @brief Merges two unsent ROBOT_STATE_SAMPLES payloads into one. Usable as core::MergeFn.
 *
 * Keeps the newer prefix and, per channel, the last Message record of the older records followed by the newer
 * ones, in the order of their last occurrence.
 * @throws std::runtime_error on a malformed payload: shorter than its prefix, ending inside a record, or holding
 *         a record other than a complete Message.
 */
std::vector<uint8_t> merge_robot_samples_payloads(std::vector<uint8_t> older, std::vector<uint8_t> newer);

/*!
 * @brief The robot end of robot state: turns declared series and record bytes into link message payload parts.
 *
 * declare_series() yields the MCAP Schema and Channel records of ROBOT_STATE_SETUP; publish() yields one MCAP
 * Message record, held until taken. Channel and schema ids count from 1 in declaration order. Not thread-safe.
 */
class RobotStatePublisher
{
public:
    //! Throws std::invalid_argument: empty name, a name declared twice, more than kMaxRobotSeries, a negative
    //! stale_after_ns, a rate_hz that is negative or not finite.
    void declare_series(RobotSeriesDeclaration declaration);

    bool has_series(std::string_view name) const;

    //! Throws std::invalid_argument for an undeclared series.
    bool in_order(std::string_view name) const;

    /*!
     * @brief Makes the Message record of one reading. A newest-only series keeps only its newest record until
     *        taken.
     * @param sample_time_ns Time of the reading, robot CLOCK_MONOTONIC: the record's log time.
     * @param publish_time_ns Time the reading was handed over, same clock: the record's publish time.
     * @throws std::invalid_argument for an undeclared series or data above kMaxRobotRecordBytes.
     */
    void publish(std::string_view name, std::span<const uint8_t> data, int64_t sample_time_ns, int64_t publish_time_ns);

    //! Schema and Channel records of the series declared since the last call; empty when none.
    std::vector<uint8_t> take_setup();

    //! The unsent Message record of each newest-only series, in the order of their publish calls; empty when none.
    std::vector<uint8_t> take_latest_records();

    //! Every unsent Message record of the in-order series, in publish order; empty when none.
    std::vector<uint8_t> take_in_order_records();

private:
    struct Series
    {
        uint16_t channel_id = 0;
        bool in_order = false;
        uint32_t sequence = 0; //!< records published on this series; wraps
        std::vector<uint8_t> unsent; //!< newest-only: the Message record not taken yet
        uint64_t unsent_order = 0; //!< publish_count_ when `unsent` was made
    };

    std::map<std::string, Series, std::less<>> series_;
    std::vector<uint8_t> setup_;
    std::vector<uint8_t> in_order_records_;
    uint64_t publish_count_ = 0;
};

} // namespace core
