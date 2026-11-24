// SPDX-FileCopyrightText: Copyright (c) 2026 mwang259. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <dayanglove_format/json.hpp>
#include <deviceio_session/deviceio_session.hpp>
#include <deviceio_trackers/dayanglove_tracker.hpp>
#include <log_bridge/logger.hpp>
#include <oxr/oxr_session.hpp>
#include <oxr_utils/os_time.hpp>
#include <schema/serialized.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{
volatile std::sig_atomic_t stopped = 0;
void stop(int)
{
    stopped = 1;
}
}

int main(int argc, char** argv)
try
{
    std::string prefix = "dayanglove";
    double seconds = 10;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg.starts_with("--collection-prefix="))
            prefix = arg.substr(20);
        else if (arg.starts_with("--seconds="))
            seconds = std::stod(arg.substr(10));
        else
            throw std::runtime_error("Unknown option: " + arg);
    }
    if (!std::isfinite(seconds) || seconds < 0 || prefix.empty() || prefix.back() == '/')
        throw std::runtime_error("Invalid duration or collection prefix");
    std::signal(SIGINT, stop);
    std::signal(SIGTERM, stop);
    const std::array readers{ std::make_shared<core::DAYANGloveTracker>(prefix + "/left"),
                              std::make_shared<core::DAYANGloveTracker>(prefix + "/right") };
    const std::vector<std::shared_ptr<core::ITracker>> trackers{ readers[0], readers[1] };
    core::OpenXRSession xr("DAYANGloveReader", core::DeviceIOSession::get_required_extensions(trackers), false);
    auto session = core::DeviceIOSession::run(trackers, xr.get_handles());
    std::array<uint64_t, 2> last_times{};
    std::array<bool, 2> active{};
    size_t received = 0;
    const auto start = std::chrono::steady_clock::now();
    while (!stopped &&
           (seconds == 0 || std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < seconds))
    {
        session->update();
        for (size_t side = 0; side < readers.size(); ++side)
        {
            const auto& data = readers[side]->get_data(*session);
            const bool fresh =
                data && data->available() &&
                static_cast<double>(core::os_monotonic_now_ns() - static_cast<int64_t>(data->received_monotonic_ns())) *
                        1e-9 <=
                    1.0;
            if (!fresh)
            {
                if (active[side])
                    std::cout << "{\"hand\":" << side + 1 << ",\"available\":false}\n";
                active[side] = false;
                continue;
            }
            if (data->hand() != side + 1)
                throw std::runtime_error("Hand collection routing mismatch");
            active[side] = true;
            if (data->received_monotonic_ns() == last_times[side])
                continue;
            last_times[side] = data->received_monotonic_ns();
            ++received;
            std::cout << dayanglove::frame_json(data.buffer().data()) << '\n';
        }
        std::cout.flush();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!received)
        throw std::runtime_error("No DAYANGlove frames received");
    return 0;
}
catch (const std::exception& exc)
{
    isaaccapture::Logger::get("isaaccapture.plugins.dayanglove.reader")->error("{}", exc.what());
    return 1;
}
