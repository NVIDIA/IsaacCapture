// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Configures process-local C++ logging sinks. Records either forward to the
// session leader or fall back to matching stderr and rotating-file sinks;
// helpers keep formatting, levels, paths, and file handling aligned with Python.

#include "sink_config.hpp"

#include "inc/log_bridge/logger.hpp"
#include "socket_sink.hpp"

#include <spdlog/common.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#ifndef _WIN32
#    include <sys/stat.h>

#    include <fcntl.h>
#    include <unistd.h>
#    include <vector>
#else
#    include <process.h>
#endif

namespace isaaccapture::detail
{
namespace
{

constexpr std::size_t kFileMaxBytes = 10 * 1024 * 1024; // 10 MiB
constexpr std::size_t kFileBackupCount = 5;
// Match Python's LINE_FORMAT; %* renders Python-style level names. Only the console
// drops the date: log files need it across midnight, rotation and processes.
constexpr const char* kFilePattern = "[%Y-%m-%d %H:%M:%S.%e] [%*] [%n] [pid:%P] %v";
constexpr const char* kConsolePattern = "[%H:%M:%S.%e] [%*] [%n] [pid:%P] %v";

// Map spdlog levels to Python's display names.
std::string_view python_level_name(spdlog::level::level_enum level)
{
    switch (level)
    {
    case spdlog::level::trace:
        return "TRACE";
    case spdlog::level::debug:
        return "DEBUG";
    case spdlog::level::info:
        return "INFO";
    case spdlog::level::warn:
        return "WARNING";
    case spdlog::level::err:
        return "ERROR";
    case spdlog::level::critical:
        return "CRITICAL";
    default:
        return "INFO";
    }
}

// Right-padded to 5 and never truncated, matching Python's own "%(levelname)-5s".
class PythonLevelFormatter : public spdlog::custom_flag_formatter
{
public:
    void format(const spdlog::details::log_msg& msg, const std::tm&, spdlog::memory_buf_t& dest) override
    {
        const std::string_view name = python_level_name(msg.level);
        dest.append(name.data(), name.data() + name.size());
        for (std::size_t pad = name.size(); pad < 5; ++pad)
        {
            dest.push_back(' ');
        }
    }

    std::unique_ptr<custom_flag_formatter> clone() const override
    {
        return std::make_unique<PythonLevelFormatter>();
    }
};

// Each sink owns its formatter.
std::unique_ptr<spdlog::pattern_formatter> make_formatter(const char* pattern)
{
    auto formatter = std::make_unique<spdlog::pattern_formatter>();
    formatter->add_flag<PythonLevelFormatter>('*').set_pattern(pattern);
    return formatter;
}

#ifndef _WIN32
// Expand "~" and "~/..."; reject unsupported "~user".
std::filesystem::path expand_user(const std::string& raw)
{
    if (raw.empty() || raw[0] != '~')
    {
        return raw;
    }
    if (raw.size() > 1 && raw[1] != '/')
    {
        return {};
    }
    const char* home = std::getenv("HOME");
    if (home == nullptr || home[0] == '\0')
    {
        return {};
    }
    return raw.size() == 1 ? std::filesystem::path(home) : std::filesystem::path(home) / raw.substr(2);
}

// Replace an unsafe entry and reserve the filename before spdlog opens it.
void reserve_log_file(const spdlog::filename_t& filename)
{
    struct ::stat existing
    {
    };
    if (::lstat(filename.c_str(), &existing) == 0)
    {
        if (S_ISREG(existing.st_mode) && existing.st_uid == ::getuid())
        {
            return; // our own file being reopened; leave it and its mode alone
        }
        ::unlink(filename.c_str());
    }
    const int reserved = ::open(filename.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    if (reserved >= 0)
    {
        ::close(reserved);
    }
}

// Reapply owner-only permissions and close-on-exec after each spdlog open.
void harden_log_file(const spdlog::filename_t&, std::FILE* file)
{
    const int fd = ::fileno(file);
    if (fd < 0)
    {
        return;
    }
    ::fchmod(fd, 0600);
    const int flags = ::fcntl(fd, F_GETFD);
    if (flags >= 0)
    {
        ::fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
    }
}
#endif

int current_pid()
{
#ifndef _WIN32
    return static_cast<int>(::getpid());
#else
    return static_cast<int>(::_getpid());
#endif
}

// Reserve a per-writer path distinct from Python and spdlog rotation names.
std::filesystem::path unique_log_path(const std::filesystem::path& dir, const std::string& stem)
{
    constexpr int kMaxAttempts = 16;
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt)
    {
        const std::string suffix = attempt == 0 ? std::string() : "-" + std::to_string(attempt);
        auto candidate = dir / (stem + suffix + ".log");
        // Reserve atomically to close the exists/open race.
#ifndef _WIN32
        const int reserved = ::open(candidate.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
        if (reserved >= 0)
        {
            ::close(reserved);
            return candidate;
        }
#else
        // No mode to set on this platform: the file inherits the directory's ACL.
        if (std::FILE* reserved = std::fopen(candidate.string().c_str(), "wx"))
        {
            std::fclose(reserved);
            return candidate;
        }
#endif
    }
    return {}; // Console only; see local_sinks().
}

// Local time, filename-safe (no ':' or ' '): YYYYMMDD-HHMMSS.
std::string current_timestamp()
{
    const std::time_t now = std::time(nullptr);
    std::tm tm_buf{};
#ifndef _WIN32
    ::localtime_r(&now, &tm_buf);
#else
    ::localtime_s(&tm_buf, &now);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tm_buf);
    return std::string(buf);
}

// Match Python's per-user log directory.
std::filesystem::path log_dir()
{
    // Treat an empty override as unset.
    if (const char* override_dir = std::getenv("ISAACCAPTURE_LOG_DIR"); override_dir != nullptr && override_dir[0] != '\0')
    {
#ifndef _WIN32
        if (auto expanded = expand_user(override_dir); !expanded.empty())
        {
            return expanded;
        }
#else
        return std::filesystem::path(override_dir);
#endif
    }
#ifndef _WIN32
    return "/tmp/isaaccapture-" + std::to_string(static_cast<unsigned>(::getuid())) + "/logs";
#else
    // Match Python's tempfile.gettempdir(), which is already per-user.
    std::error_code temp_ec;
    const auto temp_dir = std::filesystem::temp_directory_path(temp_ec);
    return (temp_ec ? std::filesystem::path(".") : temp_dir) / "isaaccapture" / "logs";
#endif
}

// Python's str.strip().lower(), so both halves read a variable identically.
std::string normalized_env(std::string_view text)
{
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())))
    {
        text.remove_prefix(1);
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())))
    {
        text.remove_suffix(1);
    }

    std::string name(text);
    std::transform(
        name.begin(), name.end(), name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return name;
}

// Parse Python's names and numeric thresholds; invalid values fall back to info.
spdlog::level::level_enum console_level()
{
    const char* raw = std::getenv("ISAACCAPTURE_LOG_LEVEL");
    if (raw == nullptr)
    {
        return spdlog::level::info;
    }

    const std::string name = normalized_env(raw);

    static constexpr std::pair<std::string_view, spdlog::level::level_enum> kNames[] = {
        { "trace", spdlog::level::trace },  { "debug", spdlog::level::debug }, { "info", spdlog::level::info },
        { "warning", spdlog::level::warn }, { "error", spdlog::level::err },   { "critical", spdlog::level::critical },
    };
    for (const auto& [candidate, level] : kNames)
    {
        if (name == candidate)
        {
            return level;
        }
    }

    std::string_view digits(name);
    if (!digits.empty() && (digits.front() == '+' || digits.front() == '-'))
    {
        digits.remove_prefix(1);
    }
    if (digits.empty() || !std::all_of(digits.begin(), digits.end(), [](char c) { return c >= '0' && c <= '9'; }))
    {
        return spdlog::level::info;
    }

    std::string_view numeric_name(name);
    if (!numeric_name.empty() && numeric_name.front() == '+')
    {
        numeric_name.remove_prefix(1);
    }
    int numeric = 0;
    const auto [end, ec] = std::from_chars(numeric_name.data(), numeric_name.data() + numeric_name.size(), numeric);
    if (ec == std::errc::result_out_of_range && end == numeric_name.data() + numeric_name.size())
    {
        return !numeric_name.empty() && numeric_name.front() == '-' ? spdlog::level::trace : spdlog::level::off;
    }
    if (ec == std::errc{} && end == numeric_name.data() + numeric_name.size())
    {
        if (numeric <= 5)
        {
            return spdlog::level::trace;
        }
        if (numeric <= 10)
        {
            return spdlog::level::debug;
        }
        if (numeric <= 20)
        {
            return spdlog::level::info;
        }
        if (numeric <= 30)
        {
            return spdlog::level::warn;
        }
        if (numeric <= 40)
        {
            return spdlog::level::err;
        }
        if (numeric <= 50)
        {
            return spdlog::level::critical;
        }
        return spdlog::level::off;
    }
    return spdlog::level::info;
}

} // namespace

const std::vector<spdlog::sink_ptr>& local_sinks()
{
    static const std::vector<spdlog::sink_ptr> sinks = []
    {
        // Logging off: every record goes to stderr unfiltered; no file, no forwarding.
        if (!logging_enabled())
        {
            auto console = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
            console->set_level(spdlog::level::trace);
            console->set_formatter(make_formatter(kConsolePattern));
            return std::vector<spdlog::sink_ptr>{ console };
        }

        // Forward to the session leader when its socket is reachable.
        if (auto socket_path = forwarding_socket_path(); !socket_path.empty())
        {
            auto forward = std::make_shared<SocketForwardSink>(std::move(socket_path));
            forward->set_level(spdlog::level::trace); // the receiver's own logger does the filtering
            return std::vector<spdlog::sink_ptr>{ forward };
        }

        // Build the infallible stderr sink before optional file setup.
        auto console = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
        console->set_level(console_level());
        console->set_formatter(make_formatter(kConsolePattern));

        auto dir = log_dir();
#ifndef _WIN32
        // Track newly created components for owner-only permissions.
        std::vector<std::filesystem::path> missing;
        {
            std::error_code exists_ec;
            std::filesystem::path probe = dir;
            while (!probe.empty() && probe != probe.parent_path() && !std::filesystem::exists(probe, exists_ec))
            {
                missing.push_back(probe);
                probe = probe.parent_path();
            }
        }
#endif
        std::error_code dir_ec;
        std::filesystem::create_directories(dir, dir_ec);
#ifndef _WIN32
        // Preserve permissions on existing operator-provided directories.
        for (const auto& component : missing)
        {
            std::error_code perms_ec;
            std::filesystem::permissions(
                component, std::filesystem::perms::owner_all, std::filesystem::perm_options::replace, perms_ec);
        }
#endif

        // Give each C++ writer a timestamped, PID-qualified path.
        auto filename =
            unique_log_path(dir, current_timestamp() + ".isaaccapture." + std::to_string(current_pid()) + ".cpp");
        if (filename.empty())
        {
            return std::vector<spdlog::sink_ptr>{ console };
        }

        try
        {
            spdlog::file_event_handlers events;
#ifndef _WIN32
            // Vet the name before opening, then harden the opened descriptor.
            events.before_open = reserve_log_file;
            events.after_open = harden_log_file;
#endif
            auto file = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                filename.string(), kFileMaxBytes, kFileBackupCount, false, events);
            file->set_level(spdlog::level::trace); // always captures everything.
            file->set_formatter(make_formatter(kFilePattern));
            return std::vector<spdlog::sink_ptr>{ console, file };
        }
        catch (const spdlog::spdlog_ex&)
        {
            // File setup failure leaves the console sink.
            return std::vector<spdlog::sink_ptr>{ console };
        }
    }();
    return sinks;
}

} // namespace isaaccapture::detail

namespace isaaccapture
{

bool logging_enabled()
{
    const char* raw = std::getenv("ISAACCAPTURE_LOGGING");
    return raw == nullptr || detail::normalized_env(raw) != "off";
}

} // namespace isaaccapture
