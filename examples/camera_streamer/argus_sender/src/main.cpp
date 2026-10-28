/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "CaptureCoordinator.h"
#include "Config.h"
#include "Runtime.h"
#include "SenderSession.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv)
{
    // stdout is block-buffered when it is connected to journald instead of a
    // terminal. Flush each newline-terminated diagnostic promptly.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    signal(SIGINT, argus_sender::handleSignal);
    signal(SIGTERM, argus_sender::handleSignal);
    signal(SIGPIPE, SIG_IGN);

    argus_sender::CommandLine commandLine;
    if (!argus_sender::parseCommandLine(argc, argv, &commandLine))
    {
        argus_sender::printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    argus_sender::setVerboseLogging(commandLine.verbose);
    if (commandLine.listModes)
        return argus_sender::printCameraModes() ? EXIT_SUCCESS : EXIT_FAILURE;

    argus_sender::AppConfig config;
    std::string error;
    if (!argus_sender::loadConfigFile(commandLine.configPath, &config, &error))
    {
        std::fprintf(stderr, "Configuration error: %s\n", error.c_str());
        return EXIT_FAILURE;
    }
    if (commandLine.checkConfig)
    {
        std::printf("argus_sender: config ok; streams=%zu, bind=%s:%u\n", config.streams.size(),
                    config.network.bindHost.c_str(), config.network.port);
        return EXIT_SUCCESS;
    }
    return argus_sender::executeConfiguredSender(config) ? EXIT_SUCCESS : EXIT_FAILURE;
}
