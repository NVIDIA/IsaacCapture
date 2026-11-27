// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <posix_utils/file_descriptor.hpp>
#include <sys/wait.h>

#include <cerrno>
#include <fcntl.h>
#include <type_traits>
#include <unistd.h>

using isaaccapture::posix_utils::FileDescriptor;
static_assert(!std::is_copy_constructible_v<FileDescriptor>);
static_assert(!std::is_copy_assignable_v<FileDescriptor>);

TEST_CASE("descriptor reset closes the old fd and destruction closes its replacement", "[unit]")
{
    int pipe_fds[2];
    REQUIRE(::pipe(pipe_fds) == 0);
    {
        FileDescriptor descriptor(pipe_fds[0]);
        descriptor.reset(pipe_fds[1]);
        CHECK(::fcntl(pipe_fds[0], F_GETFD) == -1);
        CHECK(errno == EBADF);
        descriptor.reset(descriptor.get());
        CHECK(::fcntl(pipe_fds[1], F_GETFD) >= 0);
    }
    CHECK(::fcntl(pipe_fds[1], F_GETFD) == -1);
    CHECK(errno == EBADF);
}

TEST_CASE("descriptor zero is owned without changing the parent standard input", "[unit]")
{
    const auto child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0)
    {
        if (::dup2(STDOUT_FILENO, STDIN_FILENO) != STDIN_FILENO)
            ::_exit(1);
        {
            FileDescriptor descriptor(STDIN_FILENO);
        }
        ::_exit(::fcntl(STDIN_FILENO, F_GETFD) == -1 && errno == EBADF ? 0 : 2);
    }
    int status = 0;
    REQUIRE(::waitpid(child, &status, 0) == child);
    REQUIRE(WIFEXITED(status));
    CHECK(WEXITSTATUS(status) == 0);
}
