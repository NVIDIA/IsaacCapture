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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <glob.h>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace
{
uint64_t monotonicNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct ZedDevice
{
    std::string node, model, serial;
};

std::string readLine(const std::string& path)
{
    std::string text;
    std::ifstream file(path);
    std::getline(file, text);
    return text;
}

std::string canonical(const std::string& path)
{
    char buffer[PATH_MAX];
    return realpath(path.c_str(), buffer) ? std::string(buffer) : std::string();
}

// "2-3.2.3" -> root port "3" and downstream chain "2.3"; a hub's USB 2 and USB 3 halves share the chain.
void splitUsbName(const std::string& name, std::string& rootPort, std::string& chain)
{
    const std::string ports = name.substr(name.find('-') + 1);
    const size_t dot = ports.find('.');
    rootPort = ports.substr(0, dot);
    chain = dot == std::string::npos ? std::string() : ports.substr(dot + 1);
}

// The video interface reports a model-wide serial; the unit serial is on the camera's USB 2 HID device.
std::string unitSerial(const std::string& videoUsbName, const std::vector<std::string>& videoUsbNames)
{
    std::string videoRoot, videoChain;
    splitUsbName(videoUsbName, videoRoot, videoChain);
    const std::string bus = videoUsbName.substr(0, videoUsbName.find('-'));
    const std::string peer =
        canonical("/sys/bus/usb/devices/usb" + bus + "/" + bus + "-0:1.0/usb" + bus + "-port" + videoRoot + "/peer");
    const std::string peerPort =
        peer.substr(peer.rfind("-port") == std::string::npos ? peer.size() : peer.rfind("-port") + 5);
    std::vector<std::string> onChain, onPeer;
    glob_t paths{};
    if (glob("/sys/bus/usb/devices/*/idVendor", 0, nullptr, &paths) == 0)
        for (size_t i = 0; i < paths.gl_pathc; ++i)
        {
            const std::string directory = std::string(paths.gl_pathv[i]).substr(0, std::strlen(paths.gl_pathv[i]) - 9);
            const std::string name = directory.substr(directory.rfind('/') + 1);
            const std::string serial = readLine(directory + "/serial");
            if (readLine(paths.gl_pathv[i]) != "2b03" || serial.empty() || name.find('-') == std::string::npos ||
                std::find(videoUsbNames.begin(), videoUsbNames.end(), name) != videoUsbNames.end())
                continue;
            std::string root, chain;
            splitUsbName(name, root, chain);
            const bool sameChain = videoChain.empty() || chain == videoChain ||
                                   chain.compare(0, videoChain.size() + 1, videoChain + ".") == 0;
            if (!sameChain)
                continue;
            onChain.push_back(serial);
            if (!peerPort.empty() && root == peerPort)
                onPeer.push_back(serial);
        }
    globfree(&paths);
    if (onChain.size() == 1)
        return onChain.front();
    return onPeer.size() == 1 ? onPeer.front() : std::string();
}

std::vector<ZedDevice> zedDevices()
{
    glob_t paths{};
    std::vector<ZedDevice> result;
    std::vector<std::string> usbNames;
    if (glob("/sys/class/video4linux/video*/name", 0, nullptr, &paths) == 0)
        for (size_t i = 0; i < paths.gl_pathc; ++i)
        {
            const std::string name = readLine(paths.gl_pathv[i]);
            if (name.find("ZED") == std::string::npos)
                continue;
            const std::string path(paths.gl_pathv[i]);
            const size_t start = path.rfind('/', path.size() - 6) + 1;
            const std::string video = path.substr(start, path.size() - 5 - start);
            const std::string device = "/dev/" + video;
            const int fd = open(device.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
            if (fd < 0)
                continue;
            v4l2_capability caps{};
            if (ioctl(fd, VIDIOC_QUERYCAP, &caps) == 0)
            {
                const auto flags = caps.capabilities & V4L2_CAP_DEVICE_CAPS ? caps.device_caps : caps.capabilities;
                if ((flags & V4L2_CAP_VIDEO_CAPTURE) && (flags & V4L2_CAP_STREAMING))
                {
                    // .../<usb device>/<usb interface> is where the video node hangs.
                    const std::string interface = canonical("/sys/class/video4linux/" + video + "/device");
                    const std::string usbDevice = interface.substr(0, interface.rfind('/'));
                    usbNames.push_back(usbDevice.substr(usbDevice.rfind('/') + 1));
                    result.push_back({ device, name.substr(0, name.find(':')), std::string() });
                }
            }
            close(fd);
        }
    globfree(&paths);
    for (size_t i = 0; i < result.size(); ++i)
        result[i].serial = unitSerial(usbNames[i], usbNames);
    return result;
}

std::string resolveDevice(const zed_sender::CameraConfig& camera)
{
    if (camera.serial.empty() && camera.device != "auto")
        return camera.device;
    const auto devices = zedDevices();
    if (camera.serial.empty())
    {
        if (devices.size() != 1)
            throw std::runtime_error("auto requires exactly one ZED video device, found " +
                                     std::to_string(devices.size()) + "; set serial or device");
        return devices.front().node;
    }
    std::string attached;
    for (const auto& device : devices)
    {
        if (device.serial == camera.serial)
            return device.node;
        attached += (attached.empty() ? "" : ", ") + (device.serial.empty() ? device.node + " (unknown)" : device.serial);
    }
    throw std::runtime_error("not attached (attached ZED serials: " + (attached.empty() ? "none" : attached) + ")");
}

class CaptureWorker
{
public:
    CaptureWorker(const zed_sender::CameraConfig& c, const std::string& device)
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

// One session's shared wire state. Each camera sends its own bundle as soon as it is encoded;
// every bundle still describes all enabled cameras, so the receiver sees absent ones too.
class BundleSender
{
public:
    BundleSender(argus_sender::TcpPacketSender& tcp,
                 const std::vector<zed_sender::CameraConfig>& cameras,
                 uint64_t frameLimit,
                 std::atomic<uint64_t>& sent)
        : m_tcp(tcp),
          m_cameras(cameras),
          m_streamSequence(cameras.size(), 0),
          m_frameLimit(frameLimit),
          m_sent(sent),
          m_session(argus_sender::readSenderClockNs())
    {
    }

    // Returns the wire bytes sent, or 0 when the session is over.
    size_t send(size_t streamIndex,
                const zed_sender::V4l2Capture::Frame& frame,
                zed_sender::EncodedStereo&& encoded,
                uint64_t submitNs,
                uint64_t doneNs)
    {
        using namespace argus_sender;
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_failed || (m_frameLimit && m_sent >= m_frameLimit))
            return 0;
        Avs1Bundle bundle;
        bundle.flags = AVS1_BUNDLE_SENDER_CLOCK_VALID;
        bundle.sequence = ++m_sequence;
        bundle.sessionId = m_session;
        bundle.senderCaptureReadyNs = frame.captureReadyRealtimeNs;
        bundle.senderTcpEnqueueNs = readSenderClockNs();
        for (size_t i = 0; i < m_cameras.size(); ++i)
        {
            const auto& camera = m_cameras[i];
            const bool present = i == streamIndex;
            const bool stereo = camera.eye == zed_sender::Eye::Both;
            Avs1StreamRecord stream;
            stream.flags = uint16_t((present ? AVS1_STREAM_PRESENT : 0) | (stereo ? AVS1_STREAM_STEREO : 0));
            stream.width = camera.width;
            stream.height = camera.height;
            stream.fpsNumerator = camera.fps;
            stream.rotation = uint16_t(camera.rotation);
            stream.sequence = present ? ++m_streamSequence[i] : 0;
            for (unsigned eye = 0; eye < (stereo ? 2U : 1U); ++eye)
            {
                Avs1SampleRecord sample;
                sample.cameraIndex = uint8_t(eye);
                sample.role = uint8_t(stereo ? (eye ? AVS1_ROLE_RIGHT : AVS1_ROLE_LEFT) : AVS1_ROLE_MONO);
                // The camera's position in the sender config; the receiver names streams from it.
                sample.deviceIndex = camera.index;
                // USB capture has no sensor mode identity.
                sample.sensorModeIndex = std::numeric_limits<uint32_t>::max();
                if (present)
                {
                    sample.flags = AVS1_SAMPLE_PRESENT | AVS1_SAMPLE_SENDER_TIMING_VALID;
                    sample.frameNumber = frame.sequence;
                    sample.senderCaptureReadyNs = frame.captureReadyRealtimeNs;
                    sample.senderProcessSubmitNs = submitNs;
                    sample.senderProcessDoneNs = doneNs;
                    const bool right = stereo ? eye == 1 : camera.eye == zed_sender::Eye::Right;
                    sample.payload = std::move(right ? encoded.right : encoded.left);
                }
                stream.samples.push_back(std::move(sample));
            }
            bundle.streams.push_back(std::move(stream));
        }
        std::vector<uint8_t> packet;
        const char* error = nullptr;
        if (!buildAvs1WireRecord(bundle, &packet, &error))
            throw std::runtime_error(error ? error : "AVS1 serialization failed");
        const size_t bytes = packet.size();
        if (!m_tcp.enqueueFramed(std::move(packet)))
        {
            m_failed = true;
            return 0;
        }
        ++m_sent;
        return bytes;
    }

    bool finished()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_failed || (m_frameLimit && m_sent >= m_frameLimit);
    }

private:
    argus_sender::TcpPacketSender& m_tcp;
    const std::vector<zed_sender::CameraConfig> m_cameras;
    std::vector<uint64_t> m_streamSequence;
    const uint64_t m_frameLimit;
    std::atomic<uint64_t>& m_sent;
    const uint64_t m_session;
    std::mutex m_mutex;
    uint64_t m_sequence = 0;
    bool m_failed = false;
};

// Streams one camera for the session. A missing or failing camera is reported and retried
// without disturbing the others.
void runCamera(const zed_sender::CameraConfig camera,
               size_t streamIndex,
               const zed_sender::Config& config,
               BundleSender& out,
               const std::atomic<bool>& stop)
{
    using namespace argus_sender;
    const std::string label = camera.label();
    const uint64_t reminderNs = 30ULL * 1000000000ULL;
    bool unavailable = false;
    uint64_t lastReport = 0;
    auto active = [&] { return !stop && !stopRequested() && !out.finished(); };
    while (active())
    {
        try
        {
            const std::string device = resolveDevice(camera);
            CaptureWorker capture(camera, device);
            zed_sender::HardwareEncoder encoder;
            encoder.open(camera.captureWidth * 2, camera.captureHeight, camera.width, camera.height, camera.fps,
                         camera.bitrateKbps, camera.gop, camera.eye != zed_sender::Eye::Right,
                         camera.eye != zed_sender::Eye::Left);
            std::printf("zed_sender %s: %s (%s) on %s\n", label.c_str(),
                        unavailable ? "RECOVERED, streaming" : "streaming", camera.identity().c_str(), device.c_str());
            unavailable = false;
            uint64_t previousBucket = std::numeric_limits<uint64_t>::max();
            uint64_t inputSequence = 0, intervalFrames = 0, intervalBytes = 0;
            uint32_t lastCameraSequence = 0;
            bool first = true;
            uint64_t statsStart = monotonicNs();
            zed_sender::V4l2Capture::Frame frame;
            while (active())
            {
                if (!capture.next(frame))
                    continue;
                inputSequence += first ? 0 : uint32_t(frame.sequence - lastCameraSequence);
                lastCameraSequence = frame.sequence;
                first = false;
                const uint64_t bucket = inputSequence * camera.fps / camera.captureFps;
                if (bucket == previousBucket)
                    continue;
                previousBucket = bucket;
                const uint64_t submit = readSenderClockNs();
                auto encoded = encoder.encode(frame.bytes.data(), frame.bytes.size(), frame.stride);
                const uint64_t done = readSenderClockNs();
                const size_t bytes = out.send(streamIndex, frame, std::move(encoded), submit, done);
                if (!bytes)
                    break;
                intervalBytes += bytes;
                ++intervalFrames;
                const uint64_t current = monotonicNs();
                if (config.logInterval && current - statsStart >= uint64_t(config.logInterval) * 1000000000ULL)
                {
                    const double seconds = (current - statsStart) / 1e9;
                    std::printf("zed_sender %s: %.2f FPS, %.3f Mbps, latest convert+encode %.2f ms\n", label.c_str(),
                                intervalFrames / seconds, intervalBytes * 8.0 / seconds / 1e6, (done - submit) / 1e6);
                    intervalBytes = intervalFrames = 0;
                    statsStart = current;
                }
            }
        }
        catch (const std::exception& e)
        {
            if (!active())
                break;
            // Report on entry to the failed state, then remind on a timer while it lasts.
            const uint64_t current = monotonicNs();
            if (!unavailable || current - lastReport >= reminderNs)
            {
                std::fprintf(stderr,
                             "zed_sender %s: CAMERA UNAVAILABLE (%s): %s. Retrying every %u ms; other cameras keep "
                             "streaming.\n",
                             label.c_str(), camera.identity().c_str(), e.what(), config.cameraRetryMs);
                lastReport = current;
            }
            unavailable = true;
            const uint64_t deadline = current + uint64_t(config.cameraRetryMs) * 1000000ULL;
            while (active() && monotonicNs() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
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
                std::cout << device.node << "  " << device.model << "  serial "
                          << (device.serial.empty() ? "unknown" : device.serial) << '\n';
            return 0;
        }
        auto c = zed_sender::loadConfig(path);
        if (portOverride)
            c.port = portOverride;
        if (hasFramesOverride)
            c.frames = framesOverride;
        if (!deviceOverride.empty())
        {
            if (c.enabledCameras().size() != 1)
                throw std::runtime_error("--device needs a config with exactly one enabled camera");
            for (auto& camera : c.cameras)
                if (camera.enabled)
                {
                    camera.serial.clear();
                    camera.device = deviceOverride;
                }
        }
        zed_sender::validateConfig(c);
        const auto cameras = c.enabledCameras();
        for (const auto& camera : c.cameras)
        {
            if (!camera.enabled)
            {
                std::printf("ZED %s: disabled\n", camera.label().c_str());
                continue;
            }
            std::printf("ZED %s (%s): eye %s, capture %ux%u/eye @ %u, output %ux%u @ %u, HEVC CBR %u kbps/eye\n",
                        camera.label().c_str(), camera.identity().c_str(), zed_sender::eyeName(camera.eye),
                        camera.captureWidth, camera.captureHeight, camera.captureFps, camera.width, camera.height,
                        camera.fps, camera.bitrateKbps);
        }
        if (checkOnly)
        {
            std::cout << "Configuration valid (camera mode availability is checked on open).\n";
            return 0;
        }
        std::atomic<uint64_t> sent{ 0 };
        bool appendDump = false;
        while (!stopRequested() && (!c.frames || sent < c.frames))
        {
            Socket socket(acceptTcpClient(c.bind, uint16_t(c.port), c.reconnectMs));
            if (socket.fd < 0)
                break;
            try
            {
                TcpPacketSenderSettings settings;
                settings.port = uint16_t(c.port);
                settings.dumpWirePath = c.dumpWire;
                for (const auto& camera : cameras)
                {
                    settings.streamLabels.push_back(camera.label());
                    if (camera.eye == zed_sender::Eye::Both)
                    {
                        settings.sampleLabels.push_back(camera.label() + ".left");
                        settings.sampleLabels.push_back(camera.label() + ".right");
                    }
                    else
                        settings.sampleLabels.push_back(camera.label());
                }
                settings.logIntervalSeconds = c.logInterval;
                TcpPacketSender sender(
                    &socket.fd, settings, uint32_t(cameras.size()), uint16_t(c.port), "zed_sender", appendDump);
                if (!sender.start())
                    throw std::runtime_error("TCP sender start failed");
                appendDump = true;
                // Cameras are acquired only after a receiver connects; each session starts fresh encoders/IDRs.
                BundleSender out(sender, cameras, c.frames, sent);
                std::atomic<bool> stop{ false };
                std::vector<std::thread> workers;
                for (size_t i = 0; i < cameras.size(); ++i)
                    workers.emplace_back(runCamera, cameras[i], i, std::cref(c), std::ref(out), std::cref(stop));
                while (!stopRequested() && !sender.hasError() && !out.finished())
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                stop = true;
                for (auto& worker : workers)
                    worker.join();
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
