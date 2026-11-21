<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `logging_config` (Python half of the logging system)

**CRITICAL (non-optional):** Before editing this package, complete the mandatory
`AGENTS.md` preflight in [`../../../../AGENTS.md`](../../../../AGENTS.md) (read
every applicable `AGENTS.md` on your paths, not just this file). The
cross-tree rules for *using* loggers are in that file under "Logging"; this
file is about changing the machinery itself.

## Module scope runs on every `import isaaccapture`

`isaaccapture/__init__.py` imports this package eagerly and then calls
`install()`. Anything at module scope here therefore executes before any caller
can guard it, on every platform the tree is built for — including the
experimental Windows build, which runs `import isaaccapture` during pybind11
stub generation.

- **Guard every POSIX-only facility at the point of use**, not at the call
  site. `_POSIX` in `_core.py` and `_HAS_UNIX_SOCKETS` in `_forwarding.py`
  exist for this. `os.getuid`, `os.O_NOFOLLOW`, `socket.AF_UNIX` and
  `socketserver.UnixStreamServer` are all absent on Windows.
- **A `class` statement evaluates its bases at import.** CPython defines
  `UnixStreamServer` inside `if hasattr(socket, "AF_UNIX")`, so a subclass of
  it must sit inside an equivalent guard — skipping the *call* is not enough.
- An unguarded POSIX call here is a **build** failure, not a runtime one: stub
  generation imports the package and fails the Windows job outright.
- **Nothing `install()` reaches may raise.** Whatever it raises is raised by
  `import isaaccapture` in the host application. An unusable log directory costs
  the handler that needed it and is reported through the handlers already
  attached; it must never cost the import.
- **Keep handlers usable through application `atexit` callbacks.** Cleanup
  registered during import can run before a callback the host registered
  earlier. Empty-file cleanup may close the file handler only when it is also
  removing that empty file; leave a non-empty handler for `logging.shutdown()`.

## Leader and forwarding children

`ISAACCAPTURE_LOG_SOCKET` decides which half of `_setup.install()` a process
takes. The leader owns the real console and file handlers and publishes a
receiver; every process that inherits the variable gets a forwarding handler
**and nothing else** — no console handler, no file handler, no fallback.

- That design assumes **the publisher outlives the processes that inherit the
  variable.** Any process deliberately spawned to outlive its launcher must
  drop `ISAACCAPTURE_LOG_SOCKET` from the child environment, or it will ship
  records to a receiver that stops answering the moment the launcher exits and
  has nothing of its own to fall back on. `cloudxr/background.py` does this;
  `cloudxr/service/_service.py`'s runtime worker deliberately does not, because
  it is tied to the service's lifetime.
- The C++ half reads the same variable, so dropping it covers both — but only
  for a process that has a Python half to do the dropping. A standalone plugin
  executable has none, so `socket_sink.cpp` verifies the address for itself.
- **Socket forwarding is the only C++ route into this tree.** Extensions loaded
  in the leader loop back through its receiver; child processes inherit the
  same address. Do not add a separate in-process sink.
- **`socket_path()` verifies the address rather than trusting it**, and unsets
  the variable when nothing is listening, so an address left behind by a dead
  leader cannot hand a process a forwarding handler and no other. That covers
  startup only — a leader that dies mid-run still takes its children's records
  with it.

## Native fd capture

`_native_fd.py` handles output that no logger can reach: the CloudXR/Monado
OpenXR runtime and the Manus SDK write formatted lines straight to fd 1/2 and
export no log hook, so the descriptor is the only seam.

- **Never redirect fd 1 or fd 2 outside a scope.** A library must not alter its
  host process's descriptors. `install()` opens the capture file and publishes
  its path; it does *not* `dup2`. Rebinding happens only inside
  `_native_fd.scoped()`, which `capture_native_output()` exposes and which
  `TeleopSession` wraps around native construction and teardown.
  `ISAACCAPTURE_NATIVE_CAPTURE=off` turns that off entirely; there is no
  process-wide mode, and adding one back is the specific regression this module
  was rewritten to remove.
- **Capture files, never pipes.** A pipe blocks writes past its 64 KiB capacity
  until a reader drains it, and a drain thread in this process needs the GIL
  while the native call doing the writing holds it — `oxr_bindings.cpp` releases
  none, so the two deadlock. A write to a file needs nothing else to run. Do not
  "simplify" this back to a pipe. The same reasoning rules out a pty.
- **A process isaaccapture launches is not the host.** Its descriptors are ours
  to point at the capture file, and that is how out-of-process vendor output is
  kept off the terminal without touching the host: `plugin.cpp` opens
  `ISAACCAPTURE_NATIVE_CAPTURE_FILE` between `fork()` and `execvp()`, and the
  Python spawn sites pass `_native_fd.capture_fd()`.
- **The published path is part of the contract.** `ISAACCAPTURE_NATIVE_CAPTURE_FILE`
  is read by C++ that cannot re-derive the name (it carries a timestamp and the
  leader's pid). Change the name or the format in both halves at once.
- **fd 1/2 are left closed if the host left them closed.** `_move_above_std()`
  relocates our own descriptors clear of 0/1/2 rather than pinning `/dev/null`
  onto a closed std fd, which would itself be a change to the host's state.
  **Decide that by the interpreter's streams, never by `os.dup()` succeeding.**
  The kernel hands out the lowest free number, so a descriptor the host started
  with closed is occupied by an ordinary file of its long before a scope opens —
  the leader's own log file, often enough. `_stdio_stream()` answers the
  question that matters, "does this interpreter write through that number".

## The package `__init__` exports six functions, and that is the whole surface

`set_console_level`, `set_console_logger_name_filter`,
`set_console_content_filter`, `set_console_format`, `set_logger_colors` and
`set_propagate_to_root`. `install()` is deliberately outside `__all__` and must
be called only by the package bootstrap.

Everything else this package defines is reached through its own module —
`_core.LINE_FORMAT`, `_core.TRACE`, `_native_api.capture_native_output`, and so
on — **including by code elsewhere in this tree**, which is why `wss.py`,
`teleop_session.py` and `cloudxr/service/_service.py` import from `._core` and
`._native_api` rather than from the package. Do not "tidy" those into a package
import: a name in `__all__` is an interface this package then has to keep, and
the native-capture helpers in particular read as an invitation to do something
`TeleopSession` already does for every site in this tree.

Raising one back up is a one-line change if a host application ever turns out to
need it. The reverse is not, once anything outside this repository imports it.

## Related

- C++ half: [`../../../core/log_bridge/AGENTS.md`](../../../core/log_bridge/AGENTS.md)
