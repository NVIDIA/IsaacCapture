// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <camera_backend_synthetic/synthetic_backend.hpp>
#include <camera_reader/camera_reader.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace core::camera;

namespace
{

constexpr std::string_view USAGE =
    "Usage: camera_bench --endpoint=PATH [--group=ID] [--frames=N] [--window=N] [--order=newest|in_order]\n"
    "                    [--check-synthetic]\n"
    "\n"
    "Reads one group from a running camera plugin and prints where each frame set spent its time.\n"
    "Without --group it lists what the plugin publishes.\n"
    "  --frames=N          frame sets to read (default 300)\n"
    "  --window=N          frame sets held for this reader at a time (default 1)\n"
    "  --order=...         newest skips to the latest frame set, in_order reads every one\n"
    "  --check-synthetic   check every byte of every image against the synthetic backend's pattern\n";

// How long this reader waits for a frame set before it gives up on the camera.
constexpr std::chrono::seconds FRAME_TIMEOUT(5);

std::optional<std::string_view> flag_value(std::string_view arg, std::string_view name)
{
    if (arg.starts_with(name) && arg.size() > name.size() && arg[name.size()] == '=')
    {
        return arg.substr(name.size() + 1);
    }
    return std::nullopt;
}

std::uint32_t number_of(std::string_view flag, std::string_view text)
{
    std::uint32_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc() || end != text.data() + text.size() || value == 0)
    {
        throw std::invalid_argument(std::string(flag) + " takes a whole number above zero, not '" + std::string(text) +
                                    "'");
    }
    return value;
}

struct Arguments
{
    std::string endpoint;
    std::string group;
    std::uint32_t frames = 300;
    ReaderOptions options;
    bool check_synthetic = false;
    bool help = false;
};

Arguments parse(int argc, char** argv)
{
    Arguments arguments;
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            arguments.help = true;
        }
        else if (arg == "--check-synthetic")
        {
            arguments.check_synthetic = true;
        }
        else if (const auto endpoint = flag_value(arg, "--endpoint"))
        {
            arguments.endpoint = *endpoint;
        }
        else if (const auto group = flag_value(arg, "--group"))
        {
            arguments.group = *group;
        }
        else if (const auto frames = flag_value(arg, "--frames"))
        {
            arguments.frames = number_of("--frames", *frames);
        }
        else if (const auto window = flag_value(arg, "--window"))
        {
            arguments.options.window = number_of("--window", *window);
        }
        else if (const auto order = flag_value(arg, "--order"))
        {
            if (*order != "newest" && *order != "in_order")
            {
                throw std::invalid_argument("--order takes newest or in_order, not '" + std::string(*order) + "'");
            }
            arguments.options.order = *order == "newest" ? DeliveryOrder::NEWEST : DeliveryOrder::IN_ORDER;
        }
        else
        {
            throw std::invalid_argument("unknown argument '" + std::string(arg) + "'");
        }
    }
    if (!arguments.help && arguments.endpoint.empty())
    {
        throw std::invalid_argument("--endpoint is required");
    }
    return arguments;
}

const char* name_of(const StreamDescription& stream)
{
    if (stream.codec == Codec::H264)
    {
        return "h264";
    }
    if (stream.codec == Codec::H265)
    {
        return "h265";
    }
    switch (stream.pixel_format)
    {
    case PixelFormat::BGRA8:
        return "bgra8";
    case PixelFormat::RGBA8:
        return "rgba8";
    case PixelFormat::NV12:
        return "nv12";
    case PixelFormat::GRAY8:
        return "gray8";
    }
    return "unknown";
}

void print_group(const GroupDescription& group)
{
    std::printf("%s: %zu stream(s)\n", group.id.c_str(), group.streams.size());
    for (const StreamDescription& stream : group.streams)
    {
        std::printf("  %-24s %ux%u %s, %.2f fps, %u slots of %llu bytes\n", stream.id.c_str(), stream.width,
                    stream.height, name_of(stream), stream.frame_rate, stream.slot_count,
                    static_cast<unsigned long long>(stream.slot_bytes));
    }
}

std::int64_t now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// One step of a frame set's way from the sensor to this reader.
struct Stage
{
    const char* name;
    std::vector<std::int64_t> nanoseconds;

    // Skips a step whose start or end the camera did not report.
    void add(std::int64_t from, std::int64_t to)
    {
        if (from != TIME_UNKNOWN && to != TIME_UNKNOWN)
        {
            nanoseconds.push_back(to - from);
        }
    }

    void print()
    {
        if (nanoseconds.empty())
        {
            std::printf("  %-22s not reported by this camera\n", name);
            return;
        }
        std::sort(nanoseconds.begin(), nanoseconds.end());
        const auto at = [&](double fraction) {
            return nanoseconds[static_cast<std::size_t>(fraction * static_cast<double>(nanoseconds.size() - 1))] / 1000.0;
        };
        std::printf("  %-22s %10.1f %10.1f %10.1f %10.1f\n", name, at(0.5), at(0.9), at(0.99), at(1.0));
    }
};

// Whether every byte of the image is what the synthetic backend wrote for its frame number.
bool is_synthetic(const ImageView& image, std::size_t stream)
{
    const std::uint64_t frame = plugins::camera::synthetic_frame_number(image.data);
    const std::byte fill = plugins::camera::synthetic_fill(frame, stream);
    const auto rest = image.data.subspan(std::min(image.data.size(), sizeof(frame)));
    return image.info.device_frame_id == frame &&
           std::all_of(rest.begin(), rest.end(), [fill](std::byte value) { return value == fill; });
}

int run(const Arguments& arguments)
{
    if (arguments.group.empty())
    {
        for (const GroupDescription& group : CameraReader::list(arguments.endpoint))
        {
            print_group(group);
        }
        return 0;
    }

    CameraReader reader = CameraReader::open(arguments.endpoint, arguments.group, arguments.options);
    print_group(reader.group());

    std::array<Stage, 5> stages = { { { "capture -> sdk ready", {} },
                                      { "sdk ready -> published", {} },
                                      { "published -> sent", {} },
                                      { "sent -> reader", {} },
                                      { "capture -> reader", {} } } };
    std::uint64_t skipped = 0;
    std::uint64_t corrupt = 0;
    const std::int64_t started = now_ns();
    for (std::uint32_t i = 0; i < arguments.frames; ++i)
    {
        const std::optional<FrameSet> frames = reader.next(FRAME_TIMEOUT);
        const std::int64_t received = now_ns();
        if (!frames)
        {
            std::fprintf(stderr, "camera_bench: no frame set from '%s' for %lld seconds\n", arguments.group.c_str(),
                         static_cast<long long>(FRAME_TIMEOUT.count()));
            return 1;
        }
        const FrameTimeline& timeline = frames->timeline();
        stages[0].add(timeline.capture_ns, timeline.sdk_ready_ns);
        stages[1].add(timeline.sdk_ready_ns, timeline.published_ns);
        stages[2].add(timeline.published_ns, timeline.sent_ns);
        stages[3].add(timeline.sent_ns, received);
        stages[4].add(timeline.capture_ns, received);
        skipped += frames->skipped();
        if (arguments.check_synthetic)
        {
            for (std::size_t stream = 0; stream < frames->image_count(); ++stream)
            {
                corrupt += is_synthetic(frames->image(stream), stream) ? 0 : 1;
            }
        }
    }
    const double seconds = static_cast<double>(now_ns() - started) / 1e9;

    std::printf("read %u frame sets in %.2f s (%.1f per second), skipped %llu\n", arguments.frames, seconds,
                arguments.frames / seconds, static_cast<unsigned long long>(skipped));
    std::printf("  %-22s %10s %10s %10s %10s   (microseconds)\n", "", "p50", "p90", "p99", "max");
    for (Stage& stage : stages)
    {
        stage.print();
    }
    if (arguments.check_synthetic)
    {
        std::printf("synthetic pattern: %llu image(s) did not match\n", static_cast<unsigned long long>(corrupt));
    }
    return corrupt == 0 ? 0 : 1;
}

} // namespace

// Everything this tool prints is its result or its usage, for a terminal; it logs nothing.
int main(int argc, char** argv)
{
    Arguments arguments;
    try
    {
        arguments = parse(argc, argv);
    }
    catch (const std::exception& error)
    {
        std::cerr << "camera_bench: " << error.what() << "\n\n" << USAGE;
        return 2;
    }
    if (arguments.help)
    {
        std::cout << USAGE;
        return 0;
    }
    try
    {
        return run(arguments);
    }
    catch (const std::exception& error)
    {
        std::cerr << "camera_bench: " << error.what() << "\n";
        return 1;
    }
}
