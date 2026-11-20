// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "Avs1Wire.h"
#include "Clock.h"
#include "Config.h"
#include "HardwareEncoder.h"
#include "Runtime.h"
#include "TcpTransport.h"
#include "V4l2Capture.h"

#include <linux/videodev2.h>
#include <sys/ioctl.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <exception>
#include <fcntl.h>
#include <fstream>
#include <glob.h>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unistd.h>

namespace
{
uint64_t monotonicNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::vector<std::string> zedDevices()
{
    glob_t paths{};
    std::vector<std::string> result;
    if (glob("/sys/class/video4linux/video*/name", 0, nullptr, &paths) == 0)
        for (size_t i = 0; i < paths.gl_pathc; ++i)
        {
            std::string name;
            std::ifstream file(paths.gl_pathv[i]);
            std::getline(file, name);
            if (name.find("ZED") == std::string::npos)
                continue;
            const std::string path(paths.gl_pathv[i]);
            const size_t start = path.rfind('/', path.size() - 6) + 1;
            const std::string device = "/dev/" + path.substr(start, path.size() - 5 - start);
            const int fd = open(device.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
            if (fd < 0)
                continue;
            v4l2_capability caps{};
            if (ioctl(fd, VIDIOC_QUERYCAP, &caps) == 0)
            {
                const auto flags = caps.capabilities & V4L2_CAP_DEVICE_CAPS ? caps.device_caps : caps.capabilities;
                if ((flags & V4L2_CAP_VIDEO_CAPTURE) && (flags & V4L2_CAP_STREAMING))
                    result.push_back(device);
            }
            close(fd);
        }
    globfree(&paths);
    return result;
}

class CaptureWorker
{
public:
    CaptureWorker(const zed_sender::Config& c, const std::string& device)
    {
        camera.open(device, c.captureWidth * 2, c.captureHeight, c.captureFps);
        worker = std::thread(
            [this, c]
            {
                try
                {
                    zed_sender::V4l2Capture::Frame frame;
                    uint64_t last = monotonicNs();
                    while (!stop && !argus_sender::stopRequested())
                    {
                        if (!camera.readLatest(frame, 100))
                        {
                            if (monotonicNs() - last > uint64_t(c.timeoutMs) * 1000000)
                                throw std::runtime_error("ZED capture timed out");
                            continue;
                        }
                        last = monotonicNs();
                        std::lock_guard<std::mutex> lock(mutex);
                        std::swap(latest, frame);
                        available = true;
                        cv.notify_one();
                    }
                }
                catch (...)
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    error = std::current_exception();
                    cv.notify_one();
                }
            });
    }
    ~CaptureWorker()
    {
        stop = true;
        if (worker.joinable())
            worker.join();
    }
    bool next(zed_sender::V4l2Capture::Frame& frame)
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait_for(lock, std::chrono::milliseconds(100), [&] { return available || error; });
        if (error)
            std::rethrow_exception(error);
        if (!available)
            return false;
        std::swap(frame, latest);
        available = false;
        return true;
    }

private:
    zed_sender::V4l2Capture camera;
    std::thread worker;
    std::atomic<bool> stop{ false };
    std::mutex mutex;
    std::condition_variable cv;
    zed_sender::V4l2Capture::Frame latest;
    bool available = false;
    std::exception_ptr error;
};

class Socket
{
public:
    explicit Socket(int value) : fd(value)
    {
    }
    ~Socket()
    {
        if (fd >= 0)
            close(fd);
    }
    int fd;
};

uint32_t number(const std::string& text)
{
    size_t end = 0;
    if (text.empty() || text[0] == '-')
        throw std::runtime_error("expected nonnegative integer");
    const auto n = std::stoul(text, &end);
    if (end != text.size() || n > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("invalid integer");
    return uint32_t(n);
}
}

int main(int argc, char** argv)
{
    using namespace argus_sender;
    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);
    std::signal(SIGPIPE, SIG_IGN);
    setvbuf(stdout, nullptr, _IOLBF, 0);
    try
    {
        std::string path = "configs/zed.yaml", deviceOverride;
        bool checkOnly = false, list = false;
        uint32_t portOverride = 0, framesOverride = 0;
        bool hasFramesOverride = false;
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            auto value = [&]() -> std::string
            {
                if (++i >= argc)
                    throw std::runtime_error("missing value after " + arg);
                return argv[i];
            };
            if (arg == "--config")
                path = value();
            else if (arg == "--device")
                deviceOverride = value();
            else if (arg == "--port")
            {
                portOverride = number(value());
                if (!portOverride)
                    throw std::runtime_error("port must be nonzero");
            }
            else if (arg == "--frames")
            {
                framesOverride = number(value());
                hasFramesOverride = true;
            }
            else if (arg == "--check-config")
                checkOnly = true;
            else if (arg == "--list-devices")
                list = true;
            else if (arg == "--verbose")
                setVerboseLogging(true);
            else if (arg == "--help")
            {
                std::cout << "zed_sender --config FILE [--device PATH] [--port N] [--frames N]\n"
                             "           [--check-config] [--list-devices] [--verbose]\n";
                return 0;
            }
            else
                throw std::runtime_error("unknown argument " + arg);
        }
        if (list)
        {
            for (const auto& device : zedDevices())
                std::cout << device << '\n';
            return 0;
        }
        auto c = zed_sender::loadConfig(path);
        if (portOverride)
            c.port = portOverride;
        if (hasFramesOverride)
            c.frames = framesOverride;
        if (!deviceOverride.empty())
            c.device = deviceOverride;
        zed_sender::validateConfig(c);
        std::printf("ZED: capture %ux%u/eye @ %u, output %ux%u/eye @ %u, HEVC CBR %u Mbps/eye\n", c.captureWidth,
                    c.captureHeight, c.captureFps, c.width, c.height, c.fps, c.bitrateMbps);
        if (checkOnly)
        {
            std::cout << "Configuration valid (camera mode availability is checked on open).\n";
            return 0;
        }
        uint64_t sent = 0;
        bool appendDump = false;
        while (!stopRequested() && (!c.frames || sent < c.frames))
        {
            Socket socket(acceptTcpClient(c.bind, uint16_t(c.port), c.reconnectMs));
            if (socket.fd < 0)
                break;
            try
            {
                std::string device = c.device;
                if (device == "auto")
                {
                    const auto devices = zedDevices();
                    if (devices.size() != 1)
                        throw std::runtime_error(
                            "auto requires exactly one ZED video device; use --list-devices/--device");
                    device = devices.front();
                }
                // Acquire only after a receiver connects; each session starts fresh encoders/IDRs.
                CaptureWorker capture(c, device);
                zed_sender::HardwareEncoder encoder;
                encoder.open(c.captureWidth * 2, c.captureHeight, c.width, c.height, c.fps, c.bitrateMbps, c.gop);
                TcpPacketSenderSettings settings;
                settings.port = uint16_t(c.port);
                settings.dumpWirePath = c.dumpWire;
                settings.streamLabels = { "head" };
                settings.sampleLabels = { "head.left", "head.right" };
                settings.logIntervalSeconds = c.logInterval;
                TcpPacketSender sender(&socket.fd, settings, 1, uint16_t(c.port), "zed_sender", appendDump);
                if (!sender.start())
                    throw std::runtime_error("TCP sender start failed");
                appendDump = true;
                const uint64_t session = readSenderClockNs();
                uint64_t sequence = 0, previousBucket = std::numeric_limits<uint64_t>::max();
                uint64_t inputSequence = 0, intervalFrames = 0, intervalBytes = 0;
                uint32_t lastCameraSequence = 0;
                bool first = true;
                uint64_t statsStart = monotonicNs();
                zed_sender::V4l2Capture::Frame frame;
                while (!stopRequested() && !sender.hasError() && (!c.frames || sent < c.frames))
                {
                    if (!capture.next(frame))
                        continue;
                    inputSequence += first ? 0 : uint32_t(frame.sequence - lastCameraSequence);
                    lastCameraSequence = frame.sequence;
                    first = false;
                    const uint64_t bucket = inputSequence * c.fps / c.captureFps;
                    if (bucket == previousBucket)
                        continue;
                    previousBucket = bucket;
                    const uint64_t submit = readSenderClockNs();
                    auto encoded = encoder.encode(frame.bytes.data(), frame.bytes.size(), frame.stride);
                    const uint64_t done = readSenderClockNs();
                    Avs1Bundle bundle;
                    bundle.flags = AVS1_BUNDLE_SENDER_CLOCK_VALID;
                    bundle.sequence = ++sequence;
                    bundle.sessionId = session;
                    bundle.senderCaptureReadyNs = frame.captureReadyRealtimeNs;
                    bundle.senderTcpEnqueueNs = readSenderClockNs();
                    Avs1StreamRecord stream;
                    stream.flags = AVS1_STREAM_PRESENT | AVS1_STREAM_STEREO;
                    stream.width = c.width;
                    stream.height = c.height;
                    stream.fpsNumerator = c.fps;
                    stream.rotation = uint16_t(c.rotation);
                    stream.sequence = sequence;
                    for (unsigned eye = 0; eye < 2; ++eye)
                    {
                        Avs1SampleRecord sample;
                        sample.flags = AVS1_SAMPLE_PRESENT | AVS1_SAMPLE_SENDER_TIMING_VALID;
                        sample.cameraIndex = uint8_t(eye);
                        sample.role = eye ? AVS1_ROLE_RIGHT : AVS1_ROLE_LEFT;
                        // No Argus device/mode identity exists for USB. Keep these fields explicitly unspecified.
                        sample.deviceIndex = sample.sensorModeIndex = std::numeric_limits<uint32_t>::max();
                        sample.frameNumber = frame.sequence;
                        sample.senderCaptureReadyNs = frame.captureReadyRealtimeNs;
                        sample.senderProcessSubmitNs = submit;
                        sample.senderProcessDoneNs = done;
                        sample.payload = std::move(eye ? encoded.right : encoded.left);
                        stream.samples.push_back(std::move(sample));
                    }
                    bundle.streams.push_back(std::move(stream));
                    std::vector<uint8_t> packet;
                    const char* error = nullptr;
                    if (!buildAvs1WireRecord(bundle, &packet, &error))
                        throw std::runtime_error(error ? error : "AVS1 serialization failed");
                    intervalBytes += packet.size();
                    if (!sender.enqueueFramed(std::move(packet)))
                        break;
                    ++sent;
                    ++intervalFrames;
                    const uint64_t current = monotonicNs();
                    if (c.logInterval && current - statsStart >= uint64_t(c.logInterval) * 1000000000ULL)
                    {
                        const double seconds = (current - statsStart) / 1e9;
                        std::printf("zed_sender: %.2f FPS, %.2f Mbps combined, latest convert+encode %.2f ms\n",
                                    intervalFrames / seconds, intervalBytes * 8.0 / seconds / 1e6, (done - submit) / 1e6);
                        intervalBytes = intervalFrames = 0;
                        statsStart = current;
                    }
                }
                sender.shutdown(c.frames && sent >= c.frames);
            }
            catch (const std::exception& e)
            {
                std::cerr << "zed_sender session: " << e.what() << '\n';
                if (c.frames)
                    return 1;
            }
            if (!stopRequested() && (!c.frames || sent < c.frames))
                std::this_thread::sleep_for(std::chrono::milliseconds(c.reconnectMs));
        }
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "zed_sender: " << e.what() << '\n';
        return 1;
    }
}
