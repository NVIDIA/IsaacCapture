// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "manus_sdk.hpp"

#include <log_bridge/logger.hpp>
#include <manus_calibration/engine.hpp>
#include <manus_calibration/export_writer.hpp>
#include <manus_calibration/protocol.hpp>
#include <sys/prctl.h>
#include <sys/socket.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <poll.h>
#include <string>
#include <string_view>
#include <unistd.h>

using namespace plugins::manus::calibration;

namespace
{

constexpr const char* kWorkerVersion = "1";
constexpr int kPollIntervalMs = 50;

static_assert(ATOMIC_BOOL_LOCK_FREE == 2, "lock-free atomic bool is required for signal safety");

std::atomic<bool> g_stop_requested{ false };

void signal_handler(int)
{
    g_stop_requested.store(true, std::memory_order_relaxed);
}

class SteadyClock final : public Clock
{
public:
    double now_s() override
    {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
};

struct Options
{
    int ipc_fd = -1;
    double discovery_timeout_s = 15.0;
};

void print_usage()
{
    // Deliberate terminal output: usage text for a person running the executable by hand.
    std::fputs("usage: manus_glove_calibrator --ipc-fd=N [--discovery-timeout=SECONDS]\n", stderr);
}

std::optional<Options> parse_options(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg = argv[i];
        char* end = nullptr;
        if (arg.rfind("--ipc-fd=", 0) == 0)
        {
            const std::string value(arg.substr(9));
            const long fd = std::strtol(value.c_str(), &end, 10);
            if (value.empty() || *end != '\0' || fd < 3)
            {
                return std::nullopt;
            }
            options.ipc_fd = static_cast<int>(fd);
        }
        else if (arg.rfind("--discovery-timeout=", 0) == 0)
        {
            const std::string value(arg.substr(20));
            const double seconds = std::strtod(value.c_str(), &end);
            if (value.empty() || *end != '\0' || !(seconds > 0.0) || seconds > 3600.0)
            {
                return std::nullopt;
            }
            options.discovery_timeout_s = seconds;
        }
        else
        {
            return std::nullopt;
        }
    }
    return options.ipc_fd >= 0 ? std::optional<Options>(options) : std::nullopt;
}

// Newline-delimited message transport over the inherited socket.
class Ipc
{
public:
    explicit Ipc(int fd) : m_fd(fd)
    {
    }

    // Writes one event line. A broken peer is remembered so the caller can cancel.
    void send(const Event& event)
    {
        if (m_broken)
        {
            return;
        }
        const std::string line = serialize_event(event) + "\n";
        size_t sent = 0;
        while (sent < line.size())
        {
            // MSG_NOSIGNAL: a vanished supervisor must surface as an error, not as SIGPIPE.
            const ssize_t n = ::send(m_fd, line.data() + sent, line.size() - sent, MSG_NOSIGNAL);
            if (n < 0 && errno == EINTR)
            {
                continue;
            }
            if (n <= 0)
            {
                m_broken = true;
                return;
            }
            sent += static_cast<size_t>(n);
        }
    }

    bool broken() const
    {
        return m_broken;
    }

    // Reads what is available and passes each complete line to `on_line`. Returns false at end of stream.
    template <typename LineHandler>
    bool read_lines(LineHandler&& on_line)
    {
        char chunk[4096];
        const ssize_t n = ::recv(m_fd, chunk, sizeof(chunk), 0);
        if (n < 0)
        {
            return errno == EINTR || errno == EAGAIN;
        }
        if (n == 0)
        {
            return false;
        }
        for (ssize_t i = 0; i < n; ++i)
        {
            if (chunk[i] != '\n')
            {
                append(chunk[i], on_line);
                continue;
            }
            if (m_discarding)
            {
                m_discarding = false;
            }
            else
            {
                on_line(std::string_view(m_line));
            }
            m_line.clear();
        }
        return true;
    }

    int fd() const
    {
        return m_fd;
    }

private:
    template <typename LineHandler>
    void append(char c, LineHandler& on_line)
    {
        if (m_discarding)
        {
            return;
        }
        m_line.push_back(c);
        if (m_line.size() > kMaxMessageBytes)
        {
            // An oversized line is dropped whole, up to its newline, and reported once.
            m_discarding = true;
            m_line.assign(kMaxMessageBytes + 1, ' ');
            on_line(std::string_view(m_line));
            m_line.clear();
        }
    }

    int m_fd;
    std::string m_line;
    bool m_discarding = false;
    bool m_broken = false;
};

} // namespace

int main(int argc, char** argv)
{
    const std::optional<Options> options = parse_options(argc, argv);
    if (!options)
    {
        print_usage();
        return 2;
    }
    const auto logger = isaaccapture::Logger::get("isaaccapture.plugins.manus.GloveCalibrator");

    std::signal(SIGTERM, signal_handler);
    std::signal(SIGINT, signal_handler);
    std::signal(SIGPIPE, SIG_IGN);

    // The supervisor owns this process: if it dies, cancel rather than keep the dongles open.
    const pid_t parent = getppid();
    prctl(PR_SET_PDEATHSIG, SIGTERM);
    if (getppid() != parent)
    {
        g_stop_requested.store(true);
    }

    ManusGloveSdk sdk;
    PosixExportWriter writer;
    SteadyClock clock;
    Ipc ipc(options->ipc_fd);
    EngineConfig config;
    config.discovery_timeout_s = options->discovery_timeout_s;
    CalibrationEngine engine(
        sdk, writer, clock, [&ipc](const Event& event) { ipc.send(event); }, config);

    ipc.send(ReadyEvent{ kWorkerVersion, sdk.version() });
    logger->info("Calibration worker ready");

    bool stream_open = true;
    while (!engine.finished())
    {
        if (g_stop_requested.load(std::memory_order_relaxed) || ipc.broken())
        {
            engine.request_cancel();
        }

        if (stream_open)
        {
            pollfd watched{ ipc.fd(), POLLIN, 0 };
            const int ready = ::poll(&watched, 1, kPollIntervalMs);
            if (ready > 0)
            {
                stream_open = ipc.read_lines(
                    [&](std::string_view line)
                    {
                        ParsedCommand parsed = parse_command(line);
                        if (parsed.rejection)
                        {
                            ipc.send(*parsed.rejection);
                        }
                        else if (parsed.command)
                        {
                            engine.handle(*parsed.command);
                        }
                    });
                // End of stream means the supervisor is gone: cancel and clean up.
                if (!stream_open)
                {
                    logger->warn("IPC closed by the supervisor; cancelling");
                    engine.request_cancel();
                }
            }
        }
        else
        {
            ::poll(nullptr, 0, kPollIntervalMs);
        }
        engine.tick();
    }

    logger->info("Calibration worker finished");
    return engine.outcome() == Outcome::Failed ? 1 : 0;
}
