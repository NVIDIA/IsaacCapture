// SPDX-FileCopyrightText: Copyright (c) 2026 ORBBEC INC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <ego_mcap/merge.hpp>
#include <log_bridge/logger.hpp>

#include <iostream>
#include <map>
#include <string>

int main(int argc, char** argv)
{
    const auto usage = []
    { std::cout << "Usage: ego_mcap_merge --recording SESSION.mcap --media FRAGMENT.mcap --output FINAL.mcap\n"; };
    if (argc == 2 && std::string(argv[1]) == "--help")
    {
        usage();
        return 0;
    }
    std::map<std::string, std::string> options;
    for (int i = 1; i < argc; i += 2)
    {
        const std::string argument = argv[i];
        if (i + 1 >= argc || (argument != "--recording" && argument != "--media" && argument != "--output") ||
            !options.emplace(argument, argv[i + 1]).second)
        {
            std::cout << "Invalid or duplicate option: " << argument << '\n';
            usage();
            return 2;
        }
    }
    if (options.size() != 3)
    {
        usage();
        return 2;
    }
    try
    {
        plugins::ego::merge_recording(options.at("--recording"), options.at("--media"), options.at("--output"));
        return 0;
    }
    catch (const std::exception& error)
    {
        isaaccapture::Logger::get("isaaccapture.ego.merge")->error("{}", error.what());
        return 1;
    }
}
