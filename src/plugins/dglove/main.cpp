// SPDX-FileCopyrightText: Copyright (c) 2026 mwang259. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <dglove_format/json.hpp>
#include <flatbuffers/idl.h>
#include <log_bridge/logger.hpp>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/os_time.hpp>
#include <pusherio/schema_pusher.hpp>
#include <schema/dglove_bfbs_generated.h>
#include <schema/dglove_generated.h>

#include <array>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <iostream>
#include <limits>
#include <memory>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace
{
constexpr size_t kMaxLine = 16384;
constexpr size_t kCapacity = 4096;
volatile std::sig_atomic_t stopped = 0;
void stop(int)
{
    stopped = 1;
}

void validate(const core::DGloveFrame& frame)
{
    if (frame.hand() != 1 && frame.hand() != 2)
        throw std::runtime_error("Invalid DGlove hand");
    const auto count = [](const auto* values) { return values ? values->size() : 0; };
    if (!frame.received_monotonic_ns() ||
        frame.received_monotonic_ns() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        throw std::runtime_error("Invalid receiver timestamp");
    const bool pose = frame.available() && frame.pose_valid();
    if (count(frame.mano_pose45()) != (pose ? 45 : 0) || count(frame.joints21_xyz_m()) != (pose ? 63 : 0) ||
        count(frame.palm_adc()) != (frame.available() ? 210 : 0))
        throw std::runtime_error("Invalid DGlove array lengths");
    for (const auto* values : { frame.mano_pose45(), frame.joints21_xyz_m() })
        if (values)
            for (float value : *values)
                if (!std::isfinite(value))
                    throw std::runtime_error("Non-finite DGlove pose");
    if (frame.palm_adc())
        for (uint32_t value : *frame.palm_adc())
            if (value > 4095)
                throw std::runtime_error("Invalid palm ADC");
    if (frame.available() &&
        (frame.schema_version() != 1 || !frame.glove_id() || frame.glove_id()->size() == 0 || !frame.session_id() ||
         frame.session_id()->size() == 0 || !frame.output_sequence() || !frame.frame_number() ||
         frame.mirror_x() != (frame.hand() == 1) || frame.device_sequence() > 255 || frame.status() > 255 ||
         !std::isfinite(frame.source_timestamp())))
        throw std::runtime_error("Invalid DGlove source metadata");
}
} // namespace

int main(int argc, char** argv)
try
{
    bool validate_only = false, roundtrip = false;
    std::string prefix = "dglove";
    double stale_after = 1.0;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--validate-only")
            validate_only = true;
        else if (arg == "--roundtrip-json")
            roundtrip = true;
        else if (arg.starts_with("--collection-prefix="))
            prefix = arg.substr(20);
        else if (arg.starts_with("--stale-after="))
            stale_after = std::stod(arg.substr(14));
        else if (arg == "--help")
        {
            std::cout << "dglove_plugin [--collection-prefix=dglove] [--stale-after=1]\n"
                         "              [--validate-only [--roundtrip-json]]\n";
            return 0;
        }
        else
            throw std::runtime_error("Unknown option: " + arg);
    }
    if (prefix.empty() || prefix.back() == '/' || !std::isfinite(stale_after) || stale_after <= 0 ||
        stale_after > 3600 || (roundtrip && !validate_only))
        throw std::runtime_error("Invalid collection prefix, timeout or validation options");
    flatbuffers::Parser parser;
    parser.builder_.ForceDefaults(true);
    parser.opts.strict_json = true;
    if (!parser.Deserialize(core::DGloveFrameRecordBinarySchema::data(), core::DGloveFrameRecordBinarySchema::size()) ||
        !parser.SetRootType("core.DGloveFrame"))
        throw std::runtime_error("Cannot load DGlove schema");
    std::signal(SIGINT, stop);
    std::signal(SIGTERM, stop);
    std::unique_ptr<core::OpenXRSession> xr;
    if (!validate_only)
        xr = std::make_unique<core::OpenXRSession>("DGlovePlugin", core::SchemaPusher::get_required_extensions(), false);
    std::array<std::unique_ptr<core::SchemaPusher>, 2> pushers;
    std::array<uint64_t, 2> last_sequences{}, last_times{};
    std::array<std::string, 2> sessions;
    const auto consume = [&](const std::string& line)
    {
        if (!parser.ParseJson(line.c_str()))
            throw std::runtime_error(parser.error_);
        const auto* buffer = parser.builder_.GetBufferPointer();
        const auto size = parser.builder_.GetSize();
        flatbuffers::Verifier verifier(buffer, size);
        if (size > kCapacity || !verifier.VerifyBuffer<core::DGloveFrame>())
            throw std::runtime_error("Invalid or oversized DGlove FlatBuffer");
        const auto& frame = *flatbuffers::GetRoot<core::DGloveFrame>(buffer);
        validate(frame);
        const auto side = frame.hand() - 1;
        if (frame.available())
        {
            if ((frame.session_id()->str() == sessions[side] && frame.output_sequence() <= last_sequences[side]) ||
                frame.received_monotonic_ns() <= last_times[side])
                throw std::runtime_error("DGlove local time or sequence moved backwards");
            sessions[side] = frame.session_id()->str();
            last_sequences[side] = frame.output_sequence();
            last_times[side] = frame.received_monotonic_ns();
        }
        if (roundtrip)
            std::cout << dglove::frame_json(buffer) << '\n';
        if (!frame.available())
        {
            pushers[side].reset();
            sessions[side].clear();
            last_sequences[side] = 0;
            return;
        }
        if (validate_only)
            return;
        const auto local_ns = static_cast<int64_t>(frame.received_monotonic_ns());
        const double age = static_cast<double>(core::os_monotonic_now_ns() - local_ns) * 1e-9;
        if (age > stale_after || age < -0.1)
            throw std::runtime_error("Stale DGlove IPC frame or receiver clock mismatch");
        if (!pushers[side])
            pushers[side] = std::make_unique<core::SchemaPusher>(
                xr->get_handles(), core::SchemaPusherConfig{ .collection_id = prefix + (side == 0 ? "/left" : "/right"),
                                                             .max_flatbuffer_size = kCapacity,
                                                             .tensor_identifier = "dglove",
                                                             .localized_name = "DGlove source data",
                                                             .app_name = "DGlovePlugin" });
        pushers[side]->push_buffer(buffer, size, local_ns, 0);
    };
    std::string pending;
    std::array<char, 4096> bytes{};
    while (!stopped)
    {
        pollfd input{ STDIN_FILENO, POLLIN, 0 };
        const int ready = poll(&input, 1, 50);
        if (ready < 0 && errno != EINTR)
            throw std::runtime_error("stdin poll failed");
        if (ready > 0)
        {
            const auto count = read(STDIN_FILENO, bytes.data(), bytes.size());
            if (count == 0)
            {
                if (!pending.empty())
                    throw std::runtime_error("Truncated IPC frame at EOF");
                break;
            }
            if (count < 0)
            {
                if (errno == EINTR)
                    continue;
                throw std::runtime_error("stdin read failed");
            }
            pending.append(bytes.data(), static_cast<size_t>(count));
            for (size_t end; (end = pending.find('\n')) != std::string::npos;)
            {
                if (end + 1 > kMaxLine)
                    throw std::runtime_error("IPC frame exceeds size limit");
                consume(pending.substr(0, end));
                pending.erase(0, end + 1);
            }
            if (pending.size() >= kMaxLine)
                throw std::runtime_error("IPC frame exceeds size limit");
        }
        for (size_t side = 0; side < pushers.size(); ++side)
            if (pushers[side] &&
                static_cast<double>(core::os_monotonic_now_ns() - static_cast<int64_t>(last_times[side])) * 1e-9 >
                    stale_after)
                pushers[side].reset();
    }
    return 0;
}
catch (const std::exception& exc)
{
    isaaccapture::Logger::get("isaaccapture.plugins.dglove")->error("{}", exc.what());
    return 1;
}
