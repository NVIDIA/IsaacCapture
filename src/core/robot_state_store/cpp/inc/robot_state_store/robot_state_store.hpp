// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace core
{

//! Topic of the series that carries the robot's description.
inline constexpr std::string_view kRobotDescriptionTopic = "robot_description";

//! One data series as its robot declared it. The record bytes are opaque here.
struct RobotSeriesInfo
{
    std::string name; //!< topic
    std::string shape; //!< "description", "joints", "status" or the robot's own shape name
    std::string schema_name;
    std::string schema_encoding;
    std::vector<uint8_t> schema_data;
    std::string message_encoding;
    int64_t stale_after_ns = 0; //!< 0: never stale
    bool in_order = false; //!< true: every record is delivered
    double rate_hz = 0.0; //!< 0: not stated
};

//! The robot's estimate of its clock against this machine's, as it came with a record.
struct RobotClockSample
{
    int64_t robot_minus_server_ns = 0;
    bool valid = false;
    bool two_way = false;
};

//! The newest record of a series and how old it is.
struct RobotRecord
{
    std::shared_ptr<const std::vector<uint8_t>> data; //!< null: the series has no record
    uint64_t sequence = 0; //!< records taken on this series in this robot session
    int64_t sample_time_ns = 0; //!< robot clock
    int64_t publish_time_ns = 0; //!< robot clock
    int64_t receive_time_ns = 0; //!< this machine's clock
    std::optional<int64_t> age_ns; //!< at least receive_age_ns; nullopt without a clock estimate
    int64_t receive_age_ns = 0; //!< time since the record arrived
    bool clock_two_way = false;
    bool fresh = false; //!< false without a record
};

/*!
 * @brief The server end of robot state: each series' declaration and its newest record.
 *
 * Takes the payloads of the robot-state link messages (MCAP Schema, Channel and Message records, read here
 * without the MCAP library) or declarations and records given directly. Each series keeps its own freshness
 * limit, and the age of a record is measured on this machine's clock through the clock estimate that came with
 * it, never as less than the time since the record arrived. Thread-safe. Never calls into Python; add_setup()
 * and add_samples() may run on a link's receive thread.
 */
class RobotStateStore
{
public:
    RobotStateStore();
    ~RobotStateStore();
    RobotStateStore(const RobotStateStore&) = delete;
    RobotStateStore& operator=(const RobotStateStore&) = delete;

    //! A robot app that sends robot state connected: drops every series and record, connected() turns true,
    //! session_count() and setup_generation() rise.
    void begin_session();

    //! It left: connected() turns false; series and records stay. Does nothing when not connected.
    void end_session();

    /*!
     * @brief Takes the payload of ROBOT_STATE_SETUP. A series declared again with other values loses its record.
     * @throws std::runtime_error, changing nothing, on bytes that are not complete Schema and Channel records, a
     *         Channel without a topic or with an undeclared Schema, or above 256 series.
     */
    void add_setup(std::span<const uint8_t> payload);

    /*!
     * @brief Takes the payload of ROBOT_STATE_SAMPLES or ROBOT_STATE_RECORDS, prefix included.
     *
     * Records are applied in order. A record on an undeclared channel is dropped, and so is one that came in a
     * payload whose publish_seq is not above that of the payload the series' record came in.
     * @param receive_time_ns This machine's monotonic clock when the link message arrived.
     * @throws std::runtime_error, changing nothing, on a short prefix, incomplete records, a record other than a
     *         Message, or a record above 1 MiB.
     */
    void add_samples(std::span<const uint8_t> payload, int64_t receive_time_ns);

    //! The same without the wire: declares or replaces a series. Throws std::invalid_argument for an empty name,
    //! schema data above 1 MiB, or above 256 series.
    void declare_series(const RobotSeriesInfo& info);

    //! The same without the wire: puts one record. Throws std::invalid_argument for an undeclared series or data
    //! above 1 MiB.
    void put_record(std::string_view name,
                    std::span<const uint8_t> data,
                    int64_t sample_time_ns,
                    int64_t publish_time_ns,
                    int64_t receive_time_ns,
                    const RobotClockSample& clock);

    bool connected() const;
    uint64_t session_count() const; //!< begin_session() calls
    uint64_t generation() const; //!< rises with every change
    uint64_t setup_generation() const; //!< from 1, never falls; rises with begin_session() and each series change
    uint64_t dropped_records() const; //!< Messages not taken: undeclared channel, or from an older payload

    //! Declaration order, the description's series included.
    std::vector<RobotSeriesInfo> series() const;
    std::optional<RobotSeriesInfo> series_info(std::string_view name) const;

    //! The newest record of `name` with its ages at `now_ns` (this machine's monotonic clock). A series that is
    //! not declared reads as one without a record.
    RobotRecord latest(std::string_view name, int64_t now_ns) const;

    //! Everything a reader in another program needs, in a private versioned format. The series declarations are
    //! left out exactly when `known_setup_generation` equals setup_generation().
    std::vector<uint8_t> export_snapshot(uint64_t known_setup_generation = 0) const;

    /*!
     * @brief Replaces this store's content with the snapshot's (keeps its series when the snapshot left them out).
     * @throws std::runtime_error, changing nothing, on bytes export_snapshot() of this build did not make, or on
     *         a snapshot without declarations whose setup_generation is not this store's.
     */
    void import_snapshot(std::span<const uint8_t> snapshot);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace core
