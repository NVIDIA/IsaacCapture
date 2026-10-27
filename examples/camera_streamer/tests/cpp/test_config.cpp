/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Config.h"
#include "TestHarness.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>

namespace
{

bool loadText(const std::string& text, argus_sender::AppConfig* config, std::string* error)
{
    char path[] = "/tmp/argus_sender_config_XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0)
        return false;
    close(fd);
    {
        std::ofstream file(path);
        file << text;
    }
    const bool ok = argus_sender::loadConfigFile(path, config, error);
    std::remove(path);
    return ok;
}

const char* validConfig()
{
    return "version: 1\n"
           "network:\n"
           "  bind: 127.0.0.1\n"
           "  port: 55000\n"
           "runtime:\n"
           "  frames: 120\n"
           "  log_interval_s: 5\n"
           "alignment:\n"
           "  warmup_frames: 17\n"
           "  interval_frames: 29\n"
           "streams:\n"
           "  - id: front\n"
           "    type: stereo\n"
           "    capture: {mode: mailbox}\n"
           "    cameras:\n"
           "      - {device: 0, mode: 1}\n"
           "      - {device: 1, mode: 1}\n"
           "    output:\n"
           "      fps: 60\n"
           "      codec: hevc\n"
           "      width: 1280\n"
           "      resize_mode: center_crop\n"
           "      rotation: 180\n"
           "      bitrate_mbps: 25\n"
           "  - id: wrist\n"
           "    type: mono\n"
           "    cameras: [{device: 2, mode: 0}]\n"
           "    capture: {mode: fifo, fifo_length: 3}\n"
           "    output:\n"
           "      fps: 30\n"
           "      codec: raw_nv12\n";
}

} // namespace

int main()
{
    TestContext test("config");
    argus_sender::AppConfig config;
    std::string error;
    TEST_CHECK(test, loadText(validConfig(), &config, &error));
    TEST_CHECK_EQ(test, config.streams.size(), static_cast<size_t>(2));
    TEST_CHECK_EQ(test, config.alignment.warmupFrames, 17U);
    TEST_CHECK_EQ(test, config.alignment.intervalFrames, 29U);
    TEST_CHECK_EQ(test, config.runtime.logIntervalSeconds, 5U);
    TEST_CHECK_EQ(test, config.streams[0].type, argus_sender::STREAM_TYPE_STEREO);
    TEST_CHECK_EQ(test, config.streams[0].capture.mode, argus_sender::CAPTURE_MODE_MAILBOX);
    TEST_CHECK_EQ(test, config.streams[0].output.resizeMode, argus_sender::RESIZE_CENTER_CROP);
    TEST_CHECK_EQ(test, config.streams[0].output.height, static_cast<uint32_t>(0));
    TEST_CHECK_EQ(test, config.streams[1].output.codec, argus_sender::CODEC_RAW_NV12);
    TEST_CHECK_EQ(test, config.streams[1].output.fps, static_cast<uint32_t>(30));
    TEST_CHECK_EQ(test, config.streams[1].capture.mode, argus_sender::CAPTURE_MODE_FIFO);
    TEST_CHECK_EQ(test, config.streams[1].capture.fifoLength, 3U);

    {
        char program[] = "argus_sender";
        char configOption[] = "--config";
        char configPath[] = "configs/camera.yaml";
        char verboseOption[] = "--verbose";
        char checkOption[] = "--check-config";
        char* arguments[] = { program, configOption, configPath, verboseOption, checkOption };
        argus_sender::CommandLine commandLine;
        TEST_CHECK(test, argus_sender::parseCommandLine(5, arguments, &commandLine));
        TEST_CHECK_EQ(test, commandLine.configPath, std::string("configs/camera.yaml"));
        TEST_CHECK(test, commandLine.verbose);
        TEST_CHECK(test, commandLine.checkConfig);
    }

    for (const char* profile : { "s56-stereo-2560x1984-30.yaml", "s56-stereo-1080p60.yaml" })
    {
        argus_sender::AppConfig shared;
        TEST_CHECK(test, argus_sender::loadConfigFile(
                             std::string(CAMERA_STREAMER_CONFIG_DIR) + "/" + profile, &shared, &error));
        TEST_CHECK_EQ(test, shared.network.port, 5000U);
        TEST_CHECK_EQ(test, shared.streams.size(), static_cast<size_t>(1));
        TEST_CHECK_EQ(test, shared.streams[0].id, std::string("head"));
    }

    argus_sender::AppConfig invalid;
    TEST_CHECK(test, !loadText("version: 1\nstreams: []\nunknown: true\n", &invalid, &error));
    TEST_CHECK(test, error.find("unknown key") != std::string::npos);

    TEST_CHECK(test, !loadText("version: 1\n"
                               "streams:\n"
                               "  - {id: a, type: mono, cameras: [{device: 0}], output: {fps: 0}}\n"
                               "  - {id: b, type: mono, cameras: [{device: 1}], output: {fps: 30}}\n",
                               &invalid, &error));
    TEST_CHECK(test, error.find("cannot be mixed") != std::string::npos);

    TEST_CHECK(test, !loadText("version: 1\n"
                               "streams:\n"
                               "  - {id: a, type: mono, cameras: [{device: 0}], output: {fps: 30}}\n"
                               "  - {id: b, type: mono, cameras: [{device: 0}], output: {fps: 30}}\n",
                               &invalid, &error));
    TEST_CHECK(test, error.find("more than one stream") != std::string::npos);

    TEST_CHECK(test, !loadText("version: 1\n"
                               "streams:\n"
                               "  - id: bad_rotation\n"
                               "    type: mono\n"
                               "    cameras: [{device: 0}]\n"
                               "    output: {fps: 30, rotation: 90}\n",
                               &invalid, &error));
    TEST_CHECK(test, error.find("must be 0 or 180") != std::string::npos);

    TEST_CHECK(test, !loadText("version: 1\n"
                               "streams:\n"
                               "  - id: bad_capture_mode\n"
                               "    type: mono\n"
                               "    cameras: [{device: 0}]\n"
                               "    capture: {mode: newest}\n"
                               "    output: {fps: 30}\n",
                               &invalid, &error));
    TEST_CHECK(test, error.find("mailbox or fifo") != std::string::npos);

    TEST_CHECK(test, !loadText("version: 1\n"
                               "streams:\n"
                               "  - id: short_fifo\n"
                               "    type: mono\n"
                               "    cameras: [{device: 0}]\n"
                               "    capture: {mode: fifo, fifo_length: 2}\n"
                               "    output: {fps: 30}\n",
                               &invalid, &error));
    TEST_CHECK(test, error.find("at least 3") != std::string::npos);

    TEST_CHECK(test, !loadText("version: 1\n"
                               "streams:\n"
                               "  - id: mailbox_fifo_length\n"
                               "    type: mono\n"
                               "    cameras: [{device: 0}]\n"
                               "    capture: {mode: mailbox, fifo_length: 3}\n"
                               "    output: {fps: 30}\n",
                               &invalid, &error));
    TEST_CHECK(test, error.find("only valid") != std::string::npos);

    TEST_CHECK(test, !loadText("version: 1\n"
                               "streams:\n"
                               "  - id: nested_alignment\n"
                               "    type: mono\n"
                               "    cameras: [{device: 0}]\n"
                               "    output: {fps: 60}\n"
                               "    alignment: {warmup_frames: 1}\n",
                               &invalid, &error));
    TEST_CHECK(test, error.find("unknown key") != std::string::npos);

    // Sender-side remap: happy path (relative file resolves against the
    // config file's directory, i.e. /tmp for these mkstemp configs).
    argus_sender::AppConfig remapConfig;
    TEST_CHECK(test, loadText("version: 1\n"
                              "streams:\n"
                              "  - id: wrist\n"
                              "    type: mono\n"
                              "    cameras: [{device: 4}]\n"
                              "    output:\n"
                              "      fps: 60\n"
                              "      codec: raw_nv12\n"
                              "      width: 640\n"
                              "      height: 480\n"
                              "      remap:\n"
                              "        file: ../remaps/cam4.remap\n"
                              "        capture_width: 2560\n"
                              "        capture_height: 1984\n",
                              &remapConfig, &error));
    TEST_CHECK(test, remapConfig.streams[0].output.remap.enabled());
    TEST_CHECK_EQ(test, remapConfig.streams[0].output.remap.file, std::string("/tmp/../remaps/cam4.remap"));
    TEST_CHECK_EQ(test, remapConfig.streams[0].output.remap.captureWidth, 2560U);
    TEST_CHECK_EQ(test, remapConfig.streams[0].output.remap.captureHeight, 1984U);

    TEST_CHECK(test, !loadText("version: 1\n"
                               "streams:\n"
                               "  - id: remap_needs_raw\n"
                               "    type: mono\n"
                               "    cameras: [{device: 4}]\n"
                               "    output:\n"
                               "      fps: 60\n"
                               "      codec: hevc\n"
                               "      width: 640\n"
                               "      height: 480\n"
                               "      remap: {file: m.remap, capture_width: 2560, capture_height: 1984}\n",
                               &invalid, &error));
    TEST_CHECK(test, error.find("requires codec raw_nv12") != std::string::npos);

    TEST_CHECK(test, !loadText("version: 1\n"
                               "streams:\n"
                               "  - id: remap_needs_dims\n"
                               "    type: mono\n"
                               "    cameras: [{device: 4}]\n"
                               "    output:\n"
                               "      fps: 60\n"
                               "      codec: raw_nv12\n"
                               "      remap: {file: m.remap, capture_width: 2560, capture_height: 1984}\n",
                               &invalid, &error));
    TEST_CHECK(test, error.find("explicit output width and height") != std::string::npos);

    TEST_CHECK(test, !loadText("version: 1\n"
                               "streams:\n"
                               "  - id: remap_needs_capture\n"
                               "    type: mono\n"
                               "    cameras: [{device: 4}]\n"
                               "    output:\n"
                               "      fps: 60\n"
                               "      codec: raw_nv12\n"
                               "      width: 640\n"
                               "      height: 480\n"
                               "      remap: {file: m.remap}\n",
                               &invalid, &error));
    TEST_CHECK(test, error.find("capture_width/height") != std::string::npos);

    TEST_CHECK(test, !loadText("version: 1\n"
                               "streams:\n"
                               "  - id: remap_needs_file\n"
                               "    type: mono\n"
                               "    cameras: [{device: 4}]\n"
                               "    output:\n"
                               "      fps: 60\n"
                               "      codec: raw_nv12\n"
                               "      width: 640\n"
                               "      height: 480\n"
                               "      remap: {capture_width: 2560, capture_height: 1984}\n",
                               &invalid, &error));
    TEST_CHECK(test, error.find("file is required") != std::string::npos);

    TEST_CHECK(test, !loadText("version: 1\n"
                               "streams:\n"
                               "  - id: remap_unknown_key\n"
                               "    type: mono\n"
                               "    cameras: [{device: 4}]\n"
                               "    output:\n"
                               "      fps: 60\n"
                               "      codec: raw_nv12\n"
                               "      width: 640\n"
                               "      height: 480\n"
                               "      remap: {file: m.remap, capture_width: 2560, capture_height: 1984, supersample: 2}\n",
                               &invalid, &error));
    TEST_CHECK(test, error.find("unknown key") != std::string::npos);
    return test.finish();
}
