/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Config.h"

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <set>
#include <sstream>
#include <string>

#include <yaml-cpp/yaml.h>

namespace argus_sender
{
namespace
{

bool fail(std::string* error, const std::string& message)
{
    if (error)
        *error = message;
    return false;
}

bool requireMap(const YAML::Node& node, const std::string& path, std::string* error)
{
    if (!node || !node.IsMap())
        return fail(error, path + " must be a mapping");
    return true;
}

bool rejectUnknown(const YAML::Node& node, const std::set<std::string>& allowed, const std::string& path, std::string* error)
{
    if (!requireMap(node, path, error))
        return false;
    for (YAML::const_iterator it = node.begin(); it != node.end(); ++it)
    {
        const std::string key = it->first.as<std::string>();
        if (allowed.count(key) == 0)
            return fail(error, path + " has unknown key '" + key + "'");
    }
    return true;
}

template <typename T>
bool readOptional(const YAML::Node& node, const char* key, const std::string& path, T* value, std::string* error)
{
    if (!node[key])
        return true;
    try
    {
        *value = node[key].as<T>();
        return true;
    }
    catch (const YAML::Exception& e)
    {
        return fail(error, path + "." + key + ": " + e.what());
    }
}

bool readUint32(const YAML::Node& node, const char* key, const std::string& path, uint32_t* value, std::string* error)
{
    if (!node[key])
        return true;
    uint64_t parsed = 0;
    if (!readOptional(node, key, path, &parsed, error))
        return false;
    if (parsed > std::numeric_limits<uint32_t>::max())
        return fail(error, path + "." + key + " is outside uint32 range");
    *value = static_cast<uint32_t>(parsed);
    return true;
}

bool parseCamera(const YAML::Node& node, const std::string& path, CameraConfig* camera, std::string* error)
{
    const std::set<std::string> allowed = { "device", "mode" };
    if (!rejectUnknown(node, allowed, path, error))
        return false;
    if (!node["device"])
        return fail(error, path + ".device is required");
    return readUint32(node, "device", path, &camera->device, error) &&
           readUint32(node, "mode", path, &camera->mode, error);
}

bool parseRemap(const YAML::Node& node, const std::string& path, RemapConfig* remap, std::string* error)
{
    const std::set<std::string> allowed = { "file", "capture_width", "capture_height" };
    if (!rejectUnknown(node, allowed, path, error))
        return false;
    if (!node["file"])
        return fail(error, path + ".file is required");
    return readOptional(node, "file", path, &remap->file, error) &&
           readUint32(node, "capture_width", path, &remap->captureWidth, error) &&
           readUint32(node, "capture_height", path, &remap->captureHeight, error);
}

bool parseOutput(const YAML::Node& node, const std::string& path, OutputConfig* output, std::string* error)
{
    const std::set<std::string> allowed = { "fps",      "codec",        "width", "height", "resize_mode",
                                            "rotation", "bitrate_mbps", "gop",   "remap" };
    if (!rejectUnknown(node, allowed, path, error))
        return false;
    if (node["remap"] && !parseRemap(node["remap"], path + ".remap", &output->remap, error))
    {
        return false;
    }
    if (!readUint32(node, "fps", path, &output->fps, error) || !readUint32(node, "width", path, &output->width, error) ||
        !readUint32(node, "height", path, &output->height, error) ||
        !readUint32(node, "rotation", path, &output->rotation, error) ||
        !readUint32(node, "bitrate_mbps", path, &output->bitrateMbps, error) ||
        !readUint32(node, "gop", path, &output->gop, error))
    {
        return false;
    }

    std::string codec = "hevc";
    if (!readOptional(node, "codec", path, &codec, error))
        return false;
    if (codec == "hevc" || codec == "h265")
        output->codec = CODEC_HEVC;
    else if (codec == "raw" || codec == "raw_nv12" || codec == "nv12")
        output->codec = CODEC_RAW_NV12;
    else
        return fail(error, path + ".codec must be hevc or raw_nv12");

    std::string resize = "strict";
    if (!readOptional(node, "resize_mode", path, &resize, error))
        return false;
    if (resize == "strict")
        output->resizeMode = RESIZE_STRICT;
    else if (resize == "center_crop")
        output->resizeMode = RESIZE_CENTER_CROP;
    else if (resize == "stretch")
        output->resizeMode = RESIZE_STRETCH;
    else
    {
        return fail(error, path + ".resize_mode must be strict, center_crop, or stretch");
    }
    return true;
}

bool parseCapture(const YAML::Node& node, const std::string& path, CaptureConfig* capture, std::string* error)
{
    const std::set<std::string> allowed = { "mode", "fifo_length" };
    if (!rejectUnknown(node, allowed, path, error))
        return false;

    std::string mode = "mailbox";
    if (!readOptional(node, "mode", path, &mode, error) ||
        !readUint32(node, "fifo_length", path, &capture->fifoLength, error))
    {
        return false;
    }
    if (mode == "mailbox")
        capture->mode = CAPTURE_MODE_MAILBOX;
    else if (mode == "fifo")
        capture->mode = CAPTURE_MODE_FIFO;
    else
        return fail(error, path + ".mode must be mailbox or fifo");
    if (capture->mode == CAPTURE_MODE_MAILBOX && node["fifo_length"])
    {
        return fail(error, path + ".fifo_length is only valid when mode is fifo");
    }
    return true;
}

bool parseAlignment(const YAML::Node& node, const std::string& path, AlignmentConfig* alignment, std::string* error)
{
    const std::set<std::string> allowed = { "warmup_frames", "interval_frames" };
    if (!rejectUnknown(node, allowed, path, error))
        return false;
    return readUint32(node, "warmup_frames", path, &alignment->warmupFrames, error) &&
           readUint32(node, "interval_frames", path, &alignment->intervalFrames, error);
}

bool parseStream(const YAML::Node& node, size_t index, StreamConfig* stream, std::string* error)
{
    std::ostringstream pathBuilder;
    pathBuilder << "streams[" << index << "]";
    const std::string path = pathBuilder.str();
    const std::set<std::string> allowed = { "id", "type", "cameras", "capture", "output", "receiver" };
    if (!rejectUnknown(node, allowed, path, error))
        return false;
    if (node["receiver"] && !node["receiver"].IsMap())
        return fail(error, path + ".receiver must be a mapping");
    if (!node["id"] || !node["type"] || !node["cameras"] || !node["output"])
    {
        return fail(error, path + " requires id, type, cameras, and output");
    }
    if (!readOptional(node, "id", path, &stream->id, error))
        return false;

    std::string type;
    if (!readOptional(node, "type", path, &type, error))
        return false;
    if (type == "mono")
        stream->type = STREAM_TYPE_MONO;
    else if (type == "stereo")
        stream->type = STREAM_TYPE_STEREO;
    else
        return fail(error, path + ".type must be mono or stereo");

    const YAML::Node cameras = node["cameras"];
    if (!cameras.IsSequence())
        return fail(error, path + ".cameras must be a sequence");
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        CameraConfig camera;
        std::ostringstream cameraPath;
        cameraPath << path << ".cameras[" << i << "]";
        if (!parseCamera(cameras[i], cameraPath.str(), &camera, error))
            return false;
        stream->cameras.push_back(camera);
    }
    if (node["capture"] && !parseCapture(node["capture"], path + ".capture", &stream->capture, error))
    {
        return false;
    }
    if (!parseOutput(node["output"], path + ".output", &stream->output, error))
        return false;
    return true;
}

} // namespace

void printUsage(const char* argv0)
{
    std::printf(
        "Usage: %s --config PATH\n"
        "       %s --check-config --config PATH\n"
        "       %s --list-modes\n\n"
        "Options:\n"
        "  -c, --config PATH  YAML sender configuration\n"
        "      --check-config Validate configuration without opening cameras\n"
        "      --list-modes   Print Argus camera devices and sensor modes\n"
        "  -v, --verbose      Enable detailed diagnostic logging\n"
        "  -h, --help         Show this help\n",
        argv0, argv0, argv0);
}

bool parseCommandLine(int argc, char** argv, CommandLine* commandLine)
{
    if (!commandLine)
        return false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);
        if (arg == "-h" || arg == "--help")
        {
            printUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        if (arg == "--list-modes")
        {
            commandLine->listModes = true;
            continue;
        }
        if (arg == "--check-config")
        {
            commandLine->checkConfig = true;
            continue;
        }
        if (arg == "-v" || arg == "--verbose")
        {
            commandLine->verbose = true;
            continue;
        }
        if ((arg == "-c" || arg == "--config") && i + 1 < argc)
        {
            commandLine->configPath = argv[++i];
            continue;
        }
        std::fprintf(stderr, "Unknown or incomplete argument: %s\n", arg.c_str());
        return false;
    }
    if (!commandLine->listModes && commandLine->configPath.empty())
    {
        std::fprintf(stderr, "--config PATH is required\n");
        return false;
    }
    if (commandLine->listModes && !commandLine->configPath.empty())
    {
        std::fprintf(stderr, "--list-modes does not accept --config\n");
        return false;
    }
    if (commandLine->listModes && commandLine->checkConfig)
    {
        std::fprintf(stderr, "--list-modes and --check-config are mutually exclusive\n");
        return false;
    }
    return true;
}

bool loadConfigFile(const std::string& path, AppConfig* config, std::string* error)
{
    if (!config)
        return fail(error, "configuration output is null");
    *config = AppConfig();
    try
    {
        const YAML::Node root = YAML::LoadFile(path);
        const std::set<std::string> rootAllowed = { "version", "network",  "runtime", "alignment",
                                                    "streams", "receiver", "ipc" };
        if (!rejectUnknown(root, rootAllowed, "config", error))
            return false;
        if (!readUint32(root, "version", "config", &config->version, error))
            return false;

        if (root["network"])
        {
            const YAML::Node network = root["network"];
            const std::set<std::string> allowed = { "bind", "port", "reconnect_ms", "dump_wire" };
            if (!rejectUnknown(network, allowed, "network", error) ||
                !readOptional(network, "bind", "network", &config->network.bindHost, error))
            {
                return false;
            }
            uint32_t port = config->network.port;
            if (!readUint32(network, "port", "network", &port, error) ||
                !readUint32(network, "reconnect_ms", "network", &config->network.reconnectDelayMs, error) ||
                !readOptional(network, "dump_wire", "network", &config->network.dumpWirePath, error))
            {
                return false;
            }
            if (port > std::numeric_limits<uint16_t>::max())
                return fail(error, "network.port must be in 1..65535");
            config->network.port = static_cast<uint16_t>(port);
        }

        if (root["runtime"])
        {
            const YAML::Node runtime = root["runtime"];
            const std::set<std::string> allowed = { "frames", "kpi", "log_interval_s", "bench_encode" };
            if (!rejectUnknown(runtime, allowed, "runtime", error) ||
                !readOptional(runtime, "frames", "runtime", &config->runtime.frames, error) ||
                !readOptional(runtime, "kpi", "runtime", &config->runtime.kpi, error) ||
                !readUint32(runtime, "log_interval_s", "runtime", &config->runtime.logIntervalSeconds, error) ||
                !readOptional(runtime, "bench_encode", "runtime", &config->runtime.benchEncode, error))
            {
                return false;
            }
        }

        if (root["alignment"] && !parseAlignment(root["alignment"], "alignment", &config->alignment, error))
        {
            return false;
        }

        if (!root["streams"] || !root["streams"].IsSequence())
            return fail(error, "streams must be a sequence");
        for (size_t i = 0; i < root["streams"].size(); ++i)
        {
            StreamConfig stream;
            if (!parseStream(root["streams"][i], i, &stream, error))
                return false;
            config->streams.push_back(stream);
        }

        // Remap files resolve against the config file's directory so the
        // deployed tree works regardless of the service's working directory.
        const size_t lastSlash = path.find_last_of('/');
        const std::string configDir = lastSlash == std::string::npos ? "." : path.substr(0, lastSlash);
        for (size_t i = 0; i < config->streams.size(); ++i)
        {
            RemapConfig& remap = config->streams[i].output.remap;
            if (remap.enabled() && remap.file[0] != '/')
                remap.file = configDir + "/" + remap.file;
        }
    }
    catch (const YAML::Exception& e)
    {
        return fail(error, std::string("failed to parse '") + path + "': " + e.what());
    }
    return validateConfig(*config, error);
}

bool validateConfig(const AppConfig& config, std::string* error)
{
    if (config.version != 1)
        return fail(error, "config.version must be 1");
    if (config.network.bindHost.empty())
        return fail(error, "network.bind cannot be empty");
    if (config.network.port == 0)
        return fail(error, "network.port must be in 1..65535");
    if (config.streams.empty())
        return fail(error, "at least one stream is required");
    if (config.streams.size() > std::numeric_limits<uint16_t>::max())
        return fail(error, "too many streams for AVS1");

    std::set<std::string> ids;
    std::set<uint32_t> devices;
    bool sawNativeFps = false;
    bool sawExplicitFps = false;
    for (size_t i = 0; i < config.streams.size(); ++i)
    {
        const StreamConfig& stream = config.streams[i];
        std::ostringstream pathBuilder;
        pathBuilder << "streams[" << i << "]";
        const std::string path = pathBuilder.str();
        if (stream.id.empty())
            return fail(error, path + ".id cannot be empty");
        if (!ids.insert(stream.id).second)
            return fail(error, "duplicate stream id '" + stream.id + "'");
        const size_t expectedCameras = stream.type == STREAM_TYPE_STEREO ? 2 : 1;
        if (stream.cameras.size() != expectedCameras)
        {
            std::ostringstream message;
            message << path << " type " << streamTypeName(stream.type) << " requires " << expectedCameras << " camera(s)";
            return fail(error, message.str());
        }
        if (stream.type == STREAM_TYPE_STEREO && stream.cameras[0].mode != stream.cameras[1].mode)
        {
            return fail(error, path + " stereo cameras must currently use the same mode index");
        }
        for (size_t camera = 0; camera < stream.cameras.size(); ++camera)
        {
            if (!devices.insert(stream.cameras[camera].device).second)
            {
                std::ostringstream message;
                message << "Argus device " << stream.cameras[camera].device << " is assigned to more than one stream";
                return fail(error, message.str());
            }
        }
        if (stream.output.width != 0 && stream.output.width % 2 != 0)
            return fail(error, path + ".output.width must be even");
        if (stream.output.height != 0 && stream.output.height % 2 != 0)
            return fail(error, path + ".output.height must be even");
        if (stream.output.rotation != 0 && stream.output.rotation != 180)
            return fail(error, path + ".output.rotation must be 0 or 180");
        if (stream.output.codec == CODEC_HEVC && stream.output.bitrateMbps == 0)
            return fail(error, path + ".output.bitrate_mbps must be nonzero for HEVC");
        if (stream.output.remap.enabled())
        {
            const RemapConfig& remap = stream.output.remap;
            if (stream.output.codec != CODEC_RAW_NV12)
                return fail(error, path + ".output.remap requires codec raw_nv12");
            if (stream.type != STREAM_TYPE_MONO)
                return fail(error, path + ".output.remap requires a mono stream");
            if (stream.output.width == 0 || stream.output.height == 0)
            {
                return fail(error, path + ".output.remap requires explicit output width and height");
            }
            if (stream.output.resizeMode != RESIZE_STRICT)
                return fail(error, path + ".output.remap requires resize_mode strict");
            if (remap.captureWidth == 0 || remap.captureHeight == 0 || remap.captureWidth % 2 != 0 ||
                remap.captureHeight % 2 != 0)
            {
                return fail(error, path + ".output.remap.capture_width/height must be nonzero and even");
            }
        }
        if (stream.capture.mode == CAPTURE_MODE_FIFO && stream.capture.fifoLength < 3)
        {
            return fail(error, path + ".capture.fifo_length must be at least 3 for the two-frame candidate window");
        }
        if (stream.output.fps == 0)
            sawNativeFps = true;
        else
            sawExplicitFps = true;
    }
    if (sawNativeFps && sawExplicitFps)
    {
        return fail(error, "fps: 0 cannot be mixed with explicit FPS values; specify every stream rate");
    }
    return true;
}

const char* streamTypeName(StreamType type)
{
    return type == STREAM_TYPE_STEREO ? "stereo" : "mono";
}

const char* captureModeName(CaptureMode mode)
{
    return mode == CAPTURE_MODE_FIFO ? "fifo" : "mailbox";
}

const char* codecTypeName(CodecType codec)
{
    return codec == CODEC_RAW_NV12 ? "raw_nv12" : "hevc";
}

const char* resizeModeName(ResizeMode mode)
{
    switch (mode)
    {
    case RESIZE_CENTER_CROP:
        return "center_crop";
    case RESIZE_STRETCH:
        return "stretch";
    default:
        return "strict";
    }
}

} // namespace argus_sender
