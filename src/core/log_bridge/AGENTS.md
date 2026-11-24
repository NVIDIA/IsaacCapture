<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `log_bridge` (C++ half of the logging system)

**CRITICAL (non-optional):** Before editing this package, complete the mandatory
`AGENTS.md` preflight in [`../../../AGENTS.md`](../../../AGENTS.md) and read
[`../AGENTS.md`](../AGENTS.md). The cross-tree rules for *using* loggers are in
the repo root file under "Logging"; this file is about changing the machinery.

## One C++ library, socket-only Python integration

`cpp/` builds the static `log_bridge::log_bridge_core` library and includes no
pybind11 or `Python.h`. Every C++ logger uses `local_sinks()`: a reachable
`ISAACCAPTURE_LOG_SOCKET` selects `SocketForwardSink`, including loopback from
extensions loaded in the leader; otherwise the process owns matching console
and rotating-file sinks. Keep socket forwarding as the sole C++ route into
Python logging.

**The experimental Windows build compiles this library.** Keep every POSIX
facility in `sink_config.cpp` and `socket_sink.cpp` behind `#ifndef _WIN32`
with its Windows body beside it: MSVC has no `<sys/socket.h>`, `<sys/un.h>`,
`<poll.h>` or `<unistd.h>`, and a Linux-only build cannot show the breakage.
The shared `secure_directory` module keeps its Windows implementation beside
the POSIX implementation behind the same guard.

`logger.hpp` is the only public header. `isaaccapture::Logger::get()` is memoized
on the name, so repeated calls are cheap and always yield the same instance.
Every logger starts at trace so no record is dropped before sink/handler
filtering. Do not add source-kind thresholds: the console follows
`ISAACCAPTURE_LOG_LEVEL` (info by default), while the file always captures
trace and above.

**`local_sinks()` must not throw.** It is the function-local static behind
`Logger::get()`, which every call site in this tree treats as infallible — and
a static whose initializer throws stays uninitialised, so the next logging call
re-runs it and throws again. The console sink is therefore built before
anything that can fail, directory creation reports failure without throwing
for operating-system errors, and the file sink is wrapped: an unwritable log directory costs the
file and nothing else.

**A log file is vetted twice, and `after_open` alone is not enough.** Every
rotation reaches `file_helper::open(name, truncate=true)`, which truncates
through `fopen(name, "wb")` with no `O_NOFOLLOW` — so a symlink standing at the
base name has its target truncated *before* `after_open` can look at the
descriptor. `before_open` (`reserve_log_file`) is what vets the name;
`after_open` (`harden_log_file`) only re-applies 0600 and `FD_CLOEXEC`, which
that same reopen would otherwise leave at the umask. Keep both, and do not
assume a check on the opened file can undo what opening it already did.

Neither hook may log — `sink_it_` holds the sink's mutex across them, so
logging from there deadlocks. Neither can veto, either (`before_open` returns
`void`), which is why it clears the name rather than refusing it. Do **not**
"fix" that by redirecting a failed check to `/dev/null`: that was tried, and it
is one-way (`/dev/null` reports size 0, so the sink never rotates again, so
nothing re-checks), silent, and triggered by an operator `chmod` as readily as
by an attacker.

**Known gap, not an oversight to re-report:** the rotating file sink's
descriptor can land on fd 0/1/2 when the host left one closed.
`socket_sink.cpp`'s `move_above_std()` works because `::socket()` hands over a
descriptor it then owns; `file_helper::open()` offers no such moment — it calls
`before_open` once, then retries `fopen` and calls `after_open` only on the
iteration that succeeds. Closing this needs an RAII guard on the per-record
logging path, which would pin `/dev/null` onto the host's fd 0/1/2 while held.

## Socket forwarding is the only C++ route into Python

Each target that statically links `log_bridge_core` owns its own
`local_sinks()` and spdlog registry. The shared socket contract, not shared C++
state, unifies their records: the leader publishes the address, in-process
extensions loop back through it, and child processes inherit it.

The local-sink fallback is why **no two log-file names may be derivable from
the same inputs**. A rotating file tolerates exactly one writer, and on Windows — or on
POSIX whenever `ensure_receiver()` cannot bind — the Python half and every
extension module's `local_sinks()` all want one in the same process at the same
moment. `_file.py` owns `<timestamp>.isaaccapture.<pid>.log`; C++ takes `.cpp`
and, past the first writer, a `-1`/`-2` suffix (`unique_log_path()`). Keep them
disjoint, and keep any new suffix out of the shape `rotating_file_sink` gives
its own backups.

## Private directory creation

- Prevent symlink attacks and time-of-check/time-of-use races by validating
  each opened parent before creating children. Reject `..` before normalization
  to prevent hidden path traversal. Create directories privately from the start;
  sticky shared ancestors do not make a writable final directory safe.
- Traversal must not require read permission on searchable ancestors; use
  traversal-only descriptors where supported and test as a non-root caller.

## The env-var contract is shared with Python, not parallel to it

`sink_config.cpp` and `socket_sink.cpp` read `ISAACCAPTURE_LOGGING`, `ISAACCAPTURE_LOG_DIR`,
`ISAACCAPTURE_LOG_LEVEL` and `ISAACCAPTURE_LOG_SOCKET` with the same meaning the
Python half gives them, and `SocketForwardSink` speaks the same wire format as
`logging_config/_forwarding.py`. A standalone C++ process with no interpreter
therefore forwards exactly like a Python child does. If you change the frame
layout, the field names, or how a variable is interpreted, change both halves in
the same commit or they silently stop understanding each other.

## Never log between `fork()` and `execvp()`

`plugin_manager/cpp/plugin.cpp` launches every plugin through that window, where
only async-signal-safe calls are legal. `Logger::get()` allocates and touches
spdlog's global registry, so it is unsafe there. Do not route the child's
failure reports through a logger — **nor through `std::cerr`**, whose
`ostream::sentry` flushes `cout` first and so copies whatever the *host* had
buffered there into wherever fd 1 now points. The child writes a fixed error
record to a close-on-exec pipe; the parent turns it into a normal ERROR log.

The same window points the **child's** fd 1 and fd 2 at
`ISAACCAPTURE_NATIVE_CAPTURE_FILE`, which is how a plugin's non-logger output
stays off the terminal without the parent touching its own descriptors. The
child's descriptors are ours to set; the host's are not. `getenv` is *not*
async-signal-safe, so the path is read into a `const char*` **before** `fork()`.
The file remains the place to inspect non-logger output from a launched plugin.

## Related

- Python half: [`../../python/isaaccapture/logging_config/AGENTS.md`](../../python/isaaccapture/logging_config/AGENTS.md)
- spdlog is fetched in `deps/third_party/CMakeLists.txt`; when
  `BUILD_PLUGIN_OAK_CAMERA` is on it is built against vcpkg's fmt so DepthAI
  and this tree share one fmt. The reason is written there.
