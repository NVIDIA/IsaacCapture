// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/manus_calibration/export_writer.hpp"

#include <sys/stat.h>

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

namespace plugins::manus::calibration
{

namespace
{

std::string errno_text(const std::string& what)
{
    return what + ": " + std::strerror(errno);
}

} // namespace

bool PosixExportWriter::directory_usable(const std::string& dir)
{
    struct stat info
    {
    };
    return ::stat(dir.c_str(), &info) == 0 && S_ISDIR(info.st_mode) && ::access(dir.c_str(), W_OK | X_OK) == 0;
}

std::optional<std::string> PosixExportWriter::write(const std::string& dir,
                                                    const std::string& basename,
                                                    const std::vector<uint8_t>& bytes)
{
    // The basename is built by the engine, but a separator here would escape the output directory.
    if (basename.empty() || basename.find('/') != std::string::npos)
    {
        return "invalid export file name";
    }
    const std::string final_path = dir + "/" + basename;
    const std::string part_path = final_path + ".part";

    // O_EXCL: never overwrite an earlier file, and never write through a planted symlink.
    const int fd = ::open(part_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0)
    {
        return errno_text("cannot create " + part_path);
    }

    size_t written = 0;
    while (written < bytes.size())
    {
        const ssize_t n = ::write(fd, bytes.data() + written, bytes.size() - written);
        if (n < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            const std::string error = errno_text("cannot write " + part_path);
            ::close(fd);
            ::unlink(part_path.c_str());
            return error;
        }
        written += static_cast<size_t>(n);
    }
    if (::fsync(fd) != 0)
    {
        const std::string error = errno_text("cannot flush " + part_path);
        ::close(fd);
        ::unlink(part_path.c_str());
        return error;
    }
    if (::close(fd) != 0)
    {
        const std::string error = errno_text("cannot close " + part_path);
        ::unlink(part_path.c_str());
        return error;
    }
    // link() fails when the final name exists, where rename() would silently replace it.
    if (::link(part_path.c_str(), final_path.c_str()) != 0)
    {
        const std::string error = errno_text("cannot publish " + final_path);
        ::unlink(part_path.c_str());
        return error;
    }
    ::unlink(part_path.c_str());

    // The new name is only durable once the directory entry is flushed.
    const int dir_fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dir_fd >= 0)
    {
        ::fsync(dir_fd);
        ::close(dir_fd);
    }
    return std::nullopt;
}

} // namespace plugins::manus::calibration
