// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <ego_camera/ego_camera.hpp>
#include <ego_camera/time_alignment.hpp>
#include <log_bridge/logger.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace
{

std::atomic<bool> stop_requested{ false };

void signal_handler(int signal)
{
    if (signal == SIGINT || signal == SIGTERM)
        stop_requested.store(true, std::memory_order_relaxed);
}

core::OrbbecEgoCameraStream parse_camera_name(const std::string& name)
{
    if (name == "ColorLeft")
        return core::OrbbecEgoCameraStream_ColorLeft;
    if (name == "ColorRight")
        return core::OrbbecEgoCameraStream_ColorRight;
    throw std::runtime_error("Unknown camera name: '" + name + "'. Expected ColorLeft or ColorRight.");
}

core::OrbbecEgoPixelFormat parse_pixel_format(const std::string& value)
{
    if (value == "mjpg" || value == "mjpeg")
        return core::OrbbecEgoPixelFormat_Mjpg;
    if (value == "h264")
        return core::OrbbecEgoPixelFormat_H264;
    if (value == "h265" || value == "hevc")
        return core::OrbbecEgoPixelFormat_H265;
    throw std::runtime_error("Unknown format '" + value + "'. Expected mjpg, h264, or h265.");
}

bool parse_on_off(const std::string& value)
{
    if (value == "on" || value == "true" || value == "1")
        return true;
    if (value == "off" || value == "false" || value == "0")
        return false;
    throw std::runtime_error("Expected on or off, got '" + value + "'");
}

plugins::ego::StreamConfig parse_stream_arg(const std::string& arg)
{
    plugins::ego::StreamConfig config{};
    bool has_camera = false;
    std::istringstream input(arg);
    std::string token;
    while (std::getline(input, token, ','))
    {
        const auto equals = token.find('=');
        if (equals == std::string::npos)
            throw std::runtime_error("Invalid key=value in --add-stream: '" + token + "'");
        const auto key = token.substr(0, equals);
        const auto value = token.substr(equals + 1);
        if (key == "camera")
        {
            config.camera = parse_camera_name(value);
            has_camera = true;
        }
        else if (key == "output")
        {
            config.output_path = value;
        }
        else if (key == "format")
            config.pixel_format = parse_pixel_format(value);
        else if (key == "width")
            config.width = std::stoul(value);
        else if (key == "height")
            config.height = std::stoul(value);
        else if (key == "fps")
            config.fps = std::stoul(value);
        else
        {
            throw std::runtime_error("Unknown key in --add-stream: '" + key + "'");
        }
    }
    if (!has_camera)
        throw std::runtime_error("--add-stream requires camera=<name>");
    return config;
}

plugins::ego::McapMediaMode parse_mcap_media_mode(const std::string& value)
{
    if (value == "sidecar")
        return plugins::ego::McapMediaMode::Sidecar;
    if (value == "embedded")
        return plugins::ego::McapMediaMode::Embedded;
    throw std::runtime_error("--mcap-media must be sidecar or embedded");
}

void print_usage(const char* program)
{
    std::cout << "Usage: " << program << " [options] --add-stream=camera=<name>[,output=<path>]\n\n"
              << "Streams (repeatable):\n"
              << "  --add-stream=camera=ColorLeft[,output=<path>][,format=mjpg|h264|h265,width=N,height=N,fps=N]\n"
              << "  --add-stream=camera=ColorRight[,output=<path>][,format=mjpg|h264|h265,width=N,height=N,fps=N]\n\n"
              << "Capture:\n"
              << "  --width=N --height=N --fps=N  0 selects the SDK default profile\n"
              << "  --bitrate=N --dynamic-bitrate=on|off\n"
              << "  --device-uid=UID               Select a specific Ego device\n"
              << "  --reconnect-timeout=N          Wait N seconds for the same physical device; 0 disables recovery "
                 "(default 30)\n"
              << "  --reconnect-interval-ms=N      Device re-enumeration interval (default 1000)\n"
              << "  --preview                      SDL side-by-side latest-frame preview\n"
              << "  --enable-imu --imu-rate=400|1000\n"
              << "  --accel-full-scale=<g> --gyro-full-scale=<dps>\n"
              << "  --enable-audio                 Capture PCM audio (embedded MCAP needs no WAV path)\n"
              << "  --audio-output=PATH.wav        Capture audio and write a WAV sidecar\n"
              << "Metadata (mutually exclusive):\n"
              << "  --mcap-filename=PATH            Write a local MCAP\n"
              << "  --collection-prefix=PREFIX      Publish metadata via OpenXR tensors\n"
              << "  --mcap-media=sidecar|embedded (default sidecar)\n"
              << "  --keep-media-sidecars           Keep raw video/WAV with embedded MCAP\n"
              << "  --mcap-media-spool=PATH         Embedded-media fragment for TeleopSession merger\n"
              << "  --record-device-diagnostics=on|off  Save detailed diagnostics in local MCAP (default on)\n"
              << "Device controls:\n"
              << "  --exposure=N --gain=N --white-balance=N --brightness=N\n"
              << "  --sharpness=N --saturation=N --contrast=N --power-frequency=N\n"
              << "  --set-property=SDK_PROPERTY_NAME=VALUE [--persist-controls]\n"
              << "  --calibration-output=PATH.json --list-capabilities\n"
              << "  --time-alignment=PATH          Validated EGO capture timing profile\n"
              << "Ego has no quality property; use bitrate/dynamic-bitrate. Ego exposes no Depth/IR/point cloud.\n"
              << "Advertised profiles are per sensor; capture validates the exact simultaneous combination without "
                 "fallback.\n"
              << "Encoded profiles above 30 FPS remain uncertified and are rejected after SDK profile resolution.\n"
              << "  --plugin-root-id=ID            PluginManager status collection identity\n";
}

} // namespace

int main(int argc, char** argv)
try
{
    plugins::ego::CaptureConfig capture_config;
    std::map<core::OrbbecEgoCameraStream, plugins::ego::StreamConfig> stream_map;
    bool list_capabilities = false;
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument(argv[index]);
        if (argument == "--help" || argument == "-h")
        {
            print_usage(argv[0]);
            return 0;
        }
        if (argument.rfind("--add-stream=", 0) == 0)
        {
            auto stream = parse_stream_arg(argument.substr(13));
            if (!stream_map.emplace(stream.camera, std::move(stream)).second)
                throw std::runtime_error("Duplicate camera in --add-stream");
        }
        else if (argument.rfind("--width=", 0) == 0)
            capture_config.width = std::stoul(argument.substr(8));
        else if (argument.rfind("--height=", 0) == 0)
            capture_config.height = std::stoul(argument.substr(9));
        else if (argument.rfind("--fps=", 0) == 0)
            capture_config.fps = std::stoul(argument.substr(6));
        else if (argument.rfind("--device-uid=", 0) == 0)
            capture_config.device_uid = argument.substr(13);
        else if (argument.rfind("--reconnect-timeout=", 0) == 0)
            capture_config.reconnect_timeout_seconds = std::stoul(argument.substr(20));
        else if (argument.rfind("--reconnect-interval-ms=", 0) == 0)
            capture_config.reconnect_interval_milliseconds = std::stoul(argument.substr(24));
        else if (argument.rfind("--bitrate=", 0) == 0)
            capture_config.bitrate = std::stoul(argument.substr(10));
        else if (argument.rfind("--dynamic-bitrate=", 0) == 0)
        {
            capture_config.dynamic_bitrate = parse_on_off(argument.substr(18));
            capture_config.dynamic_bitrate_set = true;
        }
        else if (argument == "--preview")
            capture_config.preview = true;
        else if (argument == "--enable-imu")
            capture_config.enable_imu = true;
        else if (argument == "--enable-audio")
            capture_config.enable_audio = true;
        else if (argument.rfind("--imu-rate=", 0) == 0)
            capture_config.imu_rate = std::stoul(argument.substr(11));
        else if (argument.rfind("--accel-full-scale=", 0) == 0)
            capture_config.accel_full_scale_g = std::stof(argument.substr(19));
        else if (argument.rfind("--gyro-full-scale=", 0) == 0)
            capture_config.gyro_full_scale_dps = std::stof(argument.substr(18));
        else if (argument.rfind("--audio-output=", 0) == 0)
        {
            capture_config.audio_output = argument.substr(15);
            capture_config.enable_audio = true;
        }
        else if (argument.rfind("--collection-prefix=", 0) == 0)
            capture_config.collection_prefix = argument.substr(20);
        else if (argument.rfind("--mcap-filename=", 0) == 0)
            capture_config.mcap_filename = argument.substr(16);
        else if (argument.rfind("--mcap-media=", 0) == 0)
            capture_config.mcap_media_mode = parse_mcap_media_mode(argument.substr(13));
        else if (argument == "--keep-media-sidecars")
            capture_config.keep_media_sidecars = true;
        else if (argument.rfind("--mcap-media-spool=", 0) == 0)
            capture_config.mcap_media_spool = argument.substr(19);
        else if (argument.rfind("--calibration-output=", 0) == 0)
            capture_config.calibration_output = argument.substr(21);
        else if (argument.rfind("--time-alignment=", 0) == 0)
        {
            capture_config.time_alignment = argument.substr(17);
            if (capture_config.time_alignment.empty())
                throw std::invalid_argument("--time-alignment requires a JSON profile path");
        }
        else if (argument == "--persist-controls")
            capture_config.persist_controls = true;
        else if (argument == "--list-capabilities")
            list_capabilities = true;
        else if (argument.rfind("--set-property=", 0) == 0)
        {
            const auto setting = argument.substr(15);
            const auto equals = setting.find('=');
            if (equals == std::string::npos)
                throw std::runtime_error("--set-property requires SDK_PROPERTY_NAME=VALUE");
            capture_config.properties.push_back({ setting.substr(0, equals), std::stod(setting.substr(equals + 1)) });
        }
        else if (argument.rfind("--exposure=", 0) == 0)
            capture_config.properties.push_back({ "OB_PROP_COLOR_EXPOSURE_INT", std::stod(argument.substr(11)) });
        else if (argument.rfind("--gain=", 0) == 0)
            capture_config.properties.push_back({ "OB_PROP_COLOR_GAIN_INT", std::stod(argument.substr(7)) });
        else if (argument.rfind("--white-balance=", 0) == 0)
            capture_config.properties.push_back({ "OB_PROP_COLOR_WHITE_BALANCE_INT", std::stod(argument.substr(16)) });
        else if (argument.rfind("--brightness=", 0) == 0)
            capture_config.properties.push_back({ "OB_PROP_COLOR_BRIGHTNESS_INT", std::stod(argument.substr(13)) });
        else if (argument.rfind("--sharpness=", 0) == 0)
            capture_config.properties.push_back({ "OB_PROP_COLOR_SHARPNESS_INT", std::stod(argument.substr(12)) });
        else if (argument.rfind("--saturation=", 0) == 0)
            capture_config.properties.push_back({ "OB_PROP_COLOR_SATURATION_INT", std::stod(argument.substr(13)) });
        else if (argument.rfind("--contrast=", 0) == 0)
            capture_config.properties.push_back({ "OB_PROP_COLOR_CONTRAST_INT", std::stod(argument.substr(11)) });
        else if (argument.rfind("--power-frequency=", 0) == 0)
            capture_config.properties.push_back(
                { "OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT", std::stod(argument.substr(18)) });
        else if (argument.rfind("--plugin-root-id=", 0) == 0)
            capture_config.plugin_root_id = argument.substr(17);
        else if (argument.rfind("--record-device-diagnostics=", 0) == 0)
            capture_config.record_device_diagnostics = parse_on_off(argument.substr(28));
        else
        {
            std::cerr << "Unknown option: " << argument << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    if (!capture_config.collection_prefix.empty() && !capture_config.mcap_filename.empty())
        throw std::runtime_error("--collection-prefix and --mcap-filename are mutually exclusive");
#if !EGO_ENABLE_PREVIEW
    if (capture_config.preview)
    {
        throw std::runtime_error("--preview requires a preview-enabled build. Reconfigure with -DEGO_ENABLE_PREVIEW=ON.");
    }
#endif
    if (capture_config.imu_rate != 400 && capture_config.imu_rate != 1000)
        throw std::runtime_error("--imu-rate must be 400 or 1000");
    if (capture_config.reconnect_interval_milliseconds == 0)
        throw std::runtime_error("--reconnect-interval-ms must be greater than zero");
    const bool embedded_media = capture_config.mcap_media_mode == plugins::ego::McapMediaMode::Embedded;
    if (embedded_media && capture_config.collection_prefix.empty() && capture_config.mcap_filename.empty())
        throw std::runtime_error("--mcap-media=embedded requires --mcap-filename or --collection-prefix");
    if (embedded_media && !capture_config.collection_prefix.empty() && capture_config.mcap_media_spool.empty())
        throw std::runtime_error("--mcap-media=embedded with --collection-prefix requires --mcap-media-spool=PATH");
    if (!embedded_media && capture_config.keep_media_sidecars)
        throw std::runtime_error("--keep-media-sidecars is only valid with --mcap-media=embedded");
    if (embedded_media && !capture_config.keep_media_sidecars && !capture_config.audio_output.empty())
        throw std::runtime_error(
            "--audio-output requires --keep-media-sidecars with --mcap-media=embedded; use --enable-audio");
    if (list_capabilities)
    {
        plugins::ego::EgoCamera::list_capabilities(capture_config);
        return 0;
    }
    if (stream_map.empty())
        throw std::runtime_error("At least one --add-stream is required.");

    std::vector<plugins::ego::StreamConfig> streams;
    streams.reserve(stream_map.size());
    for (auto& [_, stream] : stream_map)
    {
        if ((embedded_media && capture_config.keep_media_sidecars) || !embedded_media)
        {
            if (stream.output_path.empty())
                throw std::runtime_error(
                    "--add-stream requires output=<path> unless --mcap-media=embedded is used without --keep-media-sidecars");
        }
        plugins::ego::validate_stream_config(stream, capture_config);
        streams.push_back(std::move(stream));
    }

    plugins::ego::validate_time_alignment_profile(capture_config.time_alignment);
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    capture_config.stop_requested = [] { return stop_requested.load(std::memory_order_relaxed); };
    auto sink = plugins::ego::create_frame_sink(streams, capture_config);
    plugins::ego::EgoCamera camera(capture_config, streams, std::move(sink));
    const auto stats_interval = std::chrono::seconds(5);
    auto last_stats = std::chrono::steady_clock::now();
    while (!stop_requested.load(std::memory_order_relaxed))
    {
        camera.update();
        if (camera.preview_closed())
            break;
        const auto now = std::chrono::steady_clock::now();
        if (now - last_stats >= stats_interval)
        {
            camera.print_stats();
            last_stats = now;
        }
    }
    camera.close();
    camera.print_stats();
    return 0;
}
catch (const plugins::ego::CaptureCancelled& error)
{
    isaaccapture::Logger::get("isaaccapture.plugins.ego.main")->info("{}", error.what());
    return 130;
}
catch (const std::exception& error)
{
    isaaccapture::Logger::get("isaaccapture.plugins.ego.main")->error("{}: {}", argv[0], error.what());
    return 1;
}
