/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ARGUS_SENDER_CONFIG_H
#define ARGUS_SENDER_CONFIG_H

#include <cstdint>
#include <string>
#include <vector>

namespace argus_sender
{

enum StreamType
{
    STREAM_TYPE_MONO,
    STREAM_TYPE_STEREO
};

enum CodecType
{
    CODEC_HEVC,
    CODEC_RAW_NV12
};

enum ResizeMode
{
    RESIZE_STRICT,
    RESIZE_CENTER_CROP,
    RESIZE_STRETCH
};

enum CaptureMode
{
    CAPTURE_MODE_MAILBOX,
    CAPTURE_MODE_FIFO
};

struct CameraConfig
{
    uint32_t device = 0;
    uint32_t mode = 0;
};

// Optional sender-side lookup-table remap (raw_nv12 mono streams only).
// The stream captures at capture_width x capture_height, applies the RMAP
// file (possibly supersampled; see NvRemap.h), and ships the configured
// output width x height ready to record — consumers apply no further remap.
struct RemapConfig
{
    std::string file;
    uint32_t captureWidth = 0;
    uint32_t captureHeight = 0;

    bool enabled() const
    {
        return !file.empty();
    }
};

struct OutputConfig
{
    uint32_t fps = 0;
    CodecType codec = CODEC_HEVC;
    uint32_t width = 0;
    uint32_t height = 0;
    ResizeMode resizeMode = RESIZE_STRICT;
    // Presentation rotation advertised to AVS1 consumers. The sender keeps
    // captured pixels in native orientation; camera_receiver applies it.
    uint32_t rotation = 0;
    uint32_t bitrateMbps = 50;
    uint32_t gop = 0;
    RemapConfig remap;
};

struct CaptureConfig
{
    CaptureMode mode = CAPTURE_MODE_MAILBOX;
    uint32_t fifoLength = 3;
};

struct AlignmentConfig
{
    uint32_t warmupFrames = 60;
    uint32_t intervalFrames = 60;
};

struct StreamConfig
{
    std::string id;
    StreamType type = STREAM_TYPE_MONO;
    std::vector<CameraConfig> cameras;
    CaptureConfig capture;
    OutputConfig output;
};

struct NetworkConfig
{
    std::string bindHost = "0.0.0.0";
    uint16_t port = 5000;
    uint32_t reconnectDelayMs = 1000;
    std::string dumpWirePath;
};

struct RuntimeConfig
{
    uint64_t frames = 0;
    bool kpi = true;
    uint32_t logIntervalSeconds = 1;
    bool benchEncode = false;
};

struct AppConfig
{
    uint32_t version = 1;
    NetworkConfig network;
    RuntimeConfig runtime;
    AlignmentConfig alignment;
    std::vector<StreamConfig> streams;
};

struct CommandLine
{
    std::string configPath;
    bool listModes = false;
    bool checkConfig = false;
    bool verbose = false;
};

void printUsage(const char* argv0);
bool parseCommandLine(int argc, char** argv, CommandLine* commandLine);
bool loadConfigFile(const std::string& path, AppConfig* config, std::string* error);
bool validateConfig(const AppConfig& config, std::string* error);

const char* streamTypeName(StreamType type);
const char* captureModeName(CaptureMode mode);
const char* codecTypeName(CodecType codec);
const char* resizeModeName(ResizeMode mode);

} // namespace argus_sender

#endif // ARGUS_SENDER_CONFIG_H
