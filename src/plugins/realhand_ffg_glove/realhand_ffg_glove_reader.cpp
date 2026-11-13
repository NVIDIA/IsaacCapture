// SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "realhand_ffg_glove_reader.hpp"

#include <log_bridge/logger.hpp>

#include <algorithm>
#include <filesystem>
#include <utility>

namespace plugins::realhand_ffg_glove
{
namespace
{
constexpr auto kDiscoveryPeriod = std::chrono::seconds(2);
constexpr auto kProbeTimeout = std::chrono::milliseconds(250);
constexpr auto kProbeQueryPeriod = std::chrono::milliseconds(40);
constexpr auto kQueryPeriod = std::chrono::milliseconds(16);
constexpr auto kStaleTimeout = std::chrono::seconds(1);
auto logger()
{
    return isaaccapture::Logger::get("isaaccapture.plugins.realhand_ffg_glove.GloveReader");
}
} // namespace

GloveReader::GloveReader(std::vector<std::string> ports, std::vector<int> baudrates)
    : configured_ports_(std::move(ports)), baudrates_(std::move(baudrates))
{
    if (baudrates_.empty())
    {
        baudrates_.assign(kDefaultBaudrates.begin(), kDefaultBaudrates.end());
    }
    if (configured_ports_.empty() && discover_serial_ports().empty())
    {
        logger()->warn("No identified FFG USB device; specify verified glove ports with --ports=/dev/ttyUSB0");
    }
}

bool GloveReader::occupied(const std::string& port) const
{
    return std::any_of(endpoints_.begin(), endpoints_.end(),
                       [&port](const auto& endpoint)
                       {
                           std::error_code error;
                           return endpoint && std::filesystem::equivalent(port, endpoint->serial->port(), error);
                       });
}

void GloveReader::advance_probe()
{
    probe_.reset();
    if (++baud_index_ == baudrates_.size())
    {
        baud_index_ = 0;
        ++candidate_index_;
        checked_port_.reset();
    }
}

void GloveReader::discover(Clock::time_point now)
{
    if (endpoints_[0] && endpoints_[1])
    {
        probe_.reset();
        candidates_.clear();
        return;
    }
    if (!probe_ && candidate_index_ >= candidates_.size())
    {
        if (last_discovery_ && now - *last_discovery_ < kDiscoveryPeriod)
        {
            return;
        }
        last_discovery_ = now;
        candidates_ = configured_ports_.empty() ? discover_serial_ports() : configured_ports_;
        candidate_index_ = 0;
        baud_index_ = 0;
        checked_port_.reset();
        if (candidates_.empty())
        {
            for (std::size_t side = 0; side < endpoints_.size(); ++side)
            {
                if (!endpoints_[side])
                {
                    errors_[side] = "No identified FFG USB device. Specify glove ports with --ports=/dev/ttyUSB0";
                }
            }
            return;
        }
    }
    const auto& port = candidates_[candidate_index_];
    if (occupied(port))
    {
        ++candidate_index_;
        baud_index_ = 0;
        checked_port_.reset();
        return;
    }
    try
    {
        if (!probe_)
        {
            // Busyness belongs to the candidate, not each baud-rate attempt.
            if (!checked_port_ || *checked_port_ != port)
            {
                if (serial_port_is_open(port))
                {
                    ++candidate_index_;
                    baud_index_ = 0;
                    checked_port_.reset();
                    for (std::size_t side = 0; side < endpoints_.size(); ++side)
                        if (!endpoints_[side])
                            errors_[side] = "serial port is already in use: " + port;
                    return;
                }
                checked_port_ = port;
            }
            probe_ = std::make_unique<SerialGlove>(port, baudrates_[baud_index_], false);
            probe_started_ = now;
            probe_->send(Command::Version);
            probe_last_query_ = now;
        }
        // Probe incrementally; never wait for a silent port on the publish thread.
        for (const auto& frame : probe_->read_available(0))
        {
            if (auto version = decode_version(frame))
            {
                const std::size_t side = version->side == HandSide::Left ? 0 : 1;
                if (!endpoints_[side])
                {
                    auto endpoint = std::make_unique<Endpoint>();
                    endpoint->serial = std::move(probe_);
                    endpoint->version = *version;
                    endpoint->last_receive = now;
                    logger()->info("Connected {} glove {} on {} at {} baud", side == 0 ? "left" : "right",
                                   version->version, port, endpoint->serial->baudrate());
                    endpoints_[side] = std::move(endpoint);
                    errors_[side].clear();
                }
                else
                {
                    logger()->warn("Ignoring duplicate {} glove on {}", side == 0 ? "left" : "right", port);
                    probe_.reset();
                }
                ++candidate_index_;
                baud_index_ = 0;
                checked_port_.reset();
                return;
            }
        }
        if (now - probe_started_ >= kProbeTimeout)
        {
            advance_probe();
        }
        else if (now - probe_last_query_ >= kProbeQueryPeriod)
        {
            probe_->send(Command::Version);
            probe_last_query_ = now;
        }
    }
    catch (const std::exception& error)
    {
        for (std::size_t side = 0; side < endpoints_.size(); ++side)
        {
            if (!endpoints_[side])
            {
                errors_[side] = error.what();
            }
        }
        advance_probe();
    }
}

void GloveReader::drop_endpoint(std::size_t side, const std::string& error)
{
    logger()->warn("Disconnected {} glove: {}", side == 0 ? "left" : "right", error);
    endpoints_[side].reset();
    errors_[side] = error;
    last_discovery_.reset();
}

void GloveReader::poll_endpoint(std::size_t side, Clock::time_point now)
{
    auto& endpoint = endpoints_[side];
    if (!endpoint)
    {
        return;
    }
    try
    {
        // Empty optional timestamps mean "due now" without subtracting time_point::min().
        if (!endpoint->last_query || now - *endpoint->last_query >= kQueryPeriod)
        {
            endpoint->serial->send(Command::Position);
            endpoint->last_query = now;
        }
        for (const auto& frame : endpoint->serial->read_available(0))
        {
            if (auto positions = decode_positions(frame))
            {
                endpoint->last_receive = now;
                endpoint->positions = *positions;
                endpoint->has_positions = true;
                if (frame.command == Command::A6Position)
                {
                    endpoint->serial->send(Command::A7Force, std::vector<uint8_t>(5 * sizeof(float), 0));
                }
            }
        }
        if (now - endpoint->last_receive > kStaleTimeout)
        {
            drop_endpoint(side, "no position response for one second");
        }
    }
    catch (const std::exception& error)
    {
        drop_endpoint(side, error.what());
    }
}

void GloveReader::update(Clock::time_point now)
{
    poll_endpoint(0, now);
    poll_endpoint(1, now);
    discover(now);
}
} // namespace plugins::realhand_ffg_glove
