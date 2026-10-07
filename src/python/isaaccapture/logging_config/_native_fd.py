# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture raw fd 1/2 output without permanently changing host descriptors.

Host fds are rebound only inside :func:`scoped`; launched child processes may
write directly to the published capture file.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from typing import TextIO

from ._core import (
    TRACE,
    _move_above_std,
    ensure_log_dir,
    logging_enabled,
    root_logger,
)

_CAPTURED_FDS = (1, 2)

# Match CPython's stdout/stderr error handling.
_FALLBACK_ERRORS = {1: "surrogateescape", 2: "backslashreplace"}

_MAX_BYTES = 10 * 1024 * 1024  # 10 MiB
_BACKUP_COUNT = 5
_CHECK_INTERVAL = 0.05

#: Capture path inherited by children without a Python interpreter.
CAPTURE_FILE_ENV = "ISAACCAPTURE_NATIVE_CAPTURE_FILE"

#: ``off`` disables :func:`scoped` in this process; anything else leaves it on.
CAPTURE_MODE_ENV = "ISAACCAPTURE_NATIVE_CAPTURE"

_lock = threading.RLock()

_sink_path: str | None = None
_sink_fd: int | None = None

# Keep separate restore and TextIOWrapper descriptors at stable numbers.
_saved_raw: dict[int, int] = {}
_saved_stream_fd: dict[int, int] = {}
_saved: dict[int, TextIO] = {}
_active_fds: list[int] = []
_active_inheritable: dict[int, bool] = {}

_depth = 0

_pre_scope_streams: dict[int, TextIO | None] = {}
_pre_scope_handler_stream: TextIO | None = None

# Persist every byte and mirror it only at TRACE.
_mirror_thread: threading.Thread | None = None
_mirror_stop = threading.Event()
_echo = False

_sink_warned = False


def _no_sink(reason: str) -> None:
    """Report capture failure once through the configured logger tree."""
    global _sink_warned
    if _sink_warned:
        return
    _sink_warned = True
    root_logger.warning(
        "Native output capture disabled: %s. Lines written straight to fd 1 or "
        "fd 2 -- the OpenXR runtime's and the vendor SDKs' own diagnostics -- "
        "will not be recorded. Set ISAACCAPTURE_LOG_DIR to a directory you can "
        "write.",
        reason,
    )


def enabled() -> bool:
    """Whether :func:`scoped` may rebind this process's fd 1 and fd 2."""
    mode = (os.environ.get(CAPTURE_MODE_ENV) or "").strip().lower()
    return logging_enabled() and mode != "off"


def _discard_if_empty(sink_path: str, sink_fd: int, owner_pid: int) -> None:
    """Remove the creator's empty file only if path and fd still match."""
    if os.getpid() != owner_pid:
        return
    try:
        opened = os.fstat(sink_fd)
        current = os.lstat(sink_path)
        if (
            opened.st_size == 0
            and stat.S_ISREG(current.st_mode)
            and (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino)
        ):
            os.unlink(sink_path)
    except OSError:
        return  # Best-effort atexit cleanup; the file may already be gone or open.


def ensure_sink() -> str | None:
    """Open and publish one capture file for both descriptors; never raise.

    A file preserves interleaving and avoids pipe backpressure while native
    calls hold the GIL.
    """
    global _sink_path, _sink_fd
    if _sink_path is not None:
        return _sink_path
    if not logging_enabled():
        return None
    with _lock:
        if _sink_path is not None:
            return _sink_path
        try:
            directory = ensure_log_dir()
        except OSError as exc:
            _no_sink(f"cannot use the log directory ({exc})")
            return None
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        path = str(directory / f"{timestamp}.isaaccapture.{os.getpid()}.native.log")
        try:
            fd = os.open(
                path,
                # Refuse collisions and symlinks in operator-provided directories.
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_APPEND
                | getattr(os, "O_NOFOLLOW", 0)
                # Preserve raw bytes on Windows.
                | getattr(os, "O_BINARY", 0),
                0o600,
            )
        except OSError as exc:
            _no_sink(f"cannot create {path} ({exc})")
            return None
        try:
            sink_fd = _move_above_std(fd)
        except OSError as exc:
            try:
                os.unlink(path)
            except OSError:
                pass
            _no_sink(f"cannot reserve a descriptor for {path} ({exc})")
            return None
        _sink_fd = sink_fd
        _sink_path = path
        # Children cannot derive the timestamped path themselves.
        os.environ[CAPTURE_FILE_ENV] = path
        atexit.register(_discard_if_empty, path, sink_fd, os.getpid())
        _start_mirror()
        return _sink_path


def capture_path() -> str | None:
    """Path of this session's capture file, or ``None`` if there is none yet."""
    return _sink_path


def capture_fd() -> int | None:
    """Return this module's append fd for child stdio, or ``None``."""
    ensure_sink()
    return _sink_fd


def _text_options(fd: int) -> dict[str, str]:
    """Copy safe encoding options from the interpreter stream for *fd*."""
    stream = _stdio_stream(fd)
    try:
        encoding = getattr(stream, "encoding", None)
        errors = getattr(stream, "errors", None)
    except Exception:  # noqa: BLE001 -- these attributes belong to the host
        encoding = errors = None
    return {
        "encoding": encoding if isinstance(encoding, str) else "utf-8",
        "errors": errors if isinstance(errors, str) else _FALLBACK_ERRORS[fd],
    }


def _ensure_saved_slots() -> list[int]:
    """Reserve stable restore slots for interpreter-owned std descriptors."""
    capturable = []
    for fd in _CAPTURED_FDS:
        if _stdio_stream(fd) is None:
            continue
        if fd in _saved_raw:
            capturable.append(fd)
            continue
        try:
            raw = _move_above_std(os.dup(fd))
        except OSError:
            continue  # closed, or otherwise not duplicable; nothing to capture
        stream_fd = None
        try:
            stream_fd = _move_above_std(os.dup(fd))
            stream = os.fdopen(
                stream_fd, "w", buffering=1, closefd=False, **_text_options(fd)
            )
        except (OSError, ValueError, LookupError):
            os.close(raw)
            if stream_fd is not None:
                os.close(stream_fd)
            continue
        _saved_raw[fd] = raw
        _saved_stream_fd[fd] = stream_fd
        _saved[fd] = stream
        capturable.append(fd)
    return capturable


def _follows(stream: TextIO | None, fd: int) -> bool:
    """Whether writes to *stream* pass through *fd*."""
    try:
        return stream.fileno() == fd
    except Exception:  # noqa: BLE001 -- host stream implementations are unrestricted
        return False


def _stdio_stream(fd: int) -> TextIO | None:
    """Return the current or original interpreter stream backed by *fd*."""
    current, original = (
        (sys.stdout, sys.__stdout__) if fd == 1 else (sys.stderr, sys.__stderr__)
    )
    return next((s for s in (current, original) if _follows(s, fd)), None)


def _set_handler_stream(handler: logging.StreamHandler, stream: TextIO) -> None:
    try:
        handler.setStream(stream)
    except Exception:  # noqa: BLE001 -- restoring host descriptors wins
        handler.acquire()
        try:
            handler.stream = stream
        finally:
            handler.release()


def _flush_host_streams() -> None:
    """Flush Python streams before rebinding or restoring their descriptors."""
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        try:
            stream.flush()
        except Exception:  # noqa: BLE001 -- restoring host descriptors wins
            pass


def _begin(console_handler: logging.StreamHandler | None) -> None:
    """Point fd 1 and fd 2 at the capture file and keep Python's streams on the
    terminal. Callers hold ``_lock``; :func:`_end` undoes exactly this.
    """
    global _active_fds, _active_inheritable, _pre_scope_handler_stream
    _active_fds = []
    _active_inheritable = {}
    sink = capture_fd()
    if sink is None:
        return
    candidates = _ensure_saved_slots()
    if not candidates:
        return

    _flush_host_streams()

    # Refresh stable slots in case the host rebound its descriptors.
    capturable = []
    for fd in candidates:
        try:
            inheritable = os.get_inheritable(fd)
            os.dup2(fd, _saved_raw[fd], inheritable=False)
            os.dup2(fd, _saved_stream_fd[fd], inheritable=False)
        except OSError:
            continue
        capturable.append(fd)
        _active_inheritable[fd] = inheritable

    # Move Python streams first so they stay on the terminal without a race.
    _pre_scope_streams.clear()
    _pre_scope_handler_stream = None
    try:
        if 1 in capturable and _follows(sys.stdout, 1):
            _pre_scope_streams[1] = sys.stdout
            sys.stdout = _saved[1]
        if 2 in capturable and _follows(sys.stderr, 2):
            _pre_scope_streams[2] = sys.stderr
            sys.stderr = _saved[2]
        if (
            console_handler is not None
            and 2 in capturable
            and _follows(console_handler.stream, 2)
        ):
            _pre_scope_handler_stream = console_handler.stream
            _set_handler_stream(console_handler, _saved[2])
    except BaseException:
        _end(console_handler)
        raise

    rebound = []
    for fd in capturable:
        try:
            os.dup2(sink, fd, inheritable=_active_inheritable[fd])
        except OSError:
            # Restore both partially rebound descriptors and Python streams.
            _active_fds = rebound
            _end(console_handler)
            return
        rebound.append(fd)
    _active_fds = rebound


def _end(console_handler: logging.StreamHandler | None) -> None:
    """Put fd 1, fd 2 and Python's streams back exactly as :func:`_begin` found
    them. Callers hold ``_lock``.
    """
    global _active_fds, _active_inheritable, _pre_scope_handler_stream
    for fd in _active_fds:
        try:
            os.dup2(_saved_raw[fd], fd, inheritable=_active_inheritable[fd])
        except OSError:
            # Keep restoring the other descriptor and Python stream objects.
            pass
    if 1 in _pre_scope_streams:
        sys.stdout = _pre_scope_streams[1]
    if 2 in _pre_scope_streams:
        sys.stderr = _pre_scope_streams[2]
    if console_handler is not None and _pre_scope_handler_stream is not None:
        _set_handler_stream(console_handler, _pre_scope_handler_stream)
    _pre_scope_streams.clear()
    _pre_scope_handler_stream = None
    _active_fds = []
    _active_inheritable = {}


@contextlib.contextmanager
def scoped(
    console_handler: logging.StreamHandler | None = None,
) -> Iterator[str | None]:
    """Capture process-wide raw fd writes for a reentrant scoped block."""
    global _depth
    if not enabled():
        yield None
        return
    with _lock:
        if _depth == 0:
            _begin(console_handler)
        _depth += 1
        path = _sink_path
    try:
        yield path
    finally:
        with _lock:
            _depth -= 1
            if _depth == 0:
                # Always restore, even if flushing a host stream fails.
                try:
                    _flush_host_streams()
                finally:
                    _end(console_handler)


def _write_all(fd: int, data: bytes) -> None:
    """``os.write`` until *data* is gone; a tty or a full disk can short-write."""
    while data:
        data = data[os.write(fd, data) :]


def _echo_target() -> int | None:
    """Return the real stderr fd without feeding capture output back into itself."""
    if 2 in _active_fds:
        return _saved_stream_fd.get(2)
    return 2 if _stdio_stream(2) is not None else None


def _echo_chunk(chunk: bytes) -> None:
    target = _echo_target()
    if _echo and target is not None:
        _write_all(target, chunk)


def _shift_backups(path: str) -> None:
    """Make room for ``path.1``, dropping the oldest bounded backup."""
    oldest = f"{path}.{_BACKUP_COUNT}"
    try:
        os.unlink(oldest)
    except FileNotFoundError:
        pass
    for index in range(_BACKUP_COUNT - 1, 0, -1):
        source = f"{path}.{index}"
        try:
            os.replace(source, f"{path}.{index + 1}")
        except FileNotFoundError:
            pass


def _rotate_capture(sink_path: str, sink_fd: int, reader) -> None:
    """Copy-truncate the shared inode so existing child descriptors keep working."""
    size = os.fstat(sink_fd).st_size
    if size < _MAX_BYTES:
        return

    # Echo the visible snapshot before truncation.
    if reader.tell() > size:
        reader.seek(0)
    while reader.tell() < size:
        chunk = reader.read(min(65536, size - reader.tell()))
        if not chunk:
            break
        _echo_chunk(chunk)

    temp_fd = -1
    temp_path = ""
    try:
        created_fd, temp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(sink_path)}.",
            suffix=".tmp",
            dir=os.path.dirname(sink_path),
        )
        try:
            temp_fd = _move_above_std(created_fd)
        except OSError:
            # _move_above_std closes every low descriptor it consumed.
            temp_fd = -1
            raise
        reader.seek(max(0, size - _MAX_BYTES))
        remaining = min(size, _MAX_BYTES)
        while remaining:
            chunk = reader.read(min(65536, remaining))
            if not chunk:
                break
            _write_all(temp_fd, chunk)
            remaining -= len(chunk)
        os.close(temp_fd)
        temp_fd = -1

        _shift_backups(sink_path)
        os.replace(temp_path, f"{sink_path}.1")
        temp_path = ""
        os.ftruncate(sink_fd, 0)
        reader.seek(0)
    except OSError:
        # Preserve the size bound even when backup creation fails.
        try:
            os.ftruncate(sink_fd, 0)
            reader.seek(0)
        except OSError:
            reader.seek(0, os.SEEK_END)
    finally:
        if temp_fd >= 0:
            try:
                os.close(temp_fd)
            except OSError:
                pass
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def _mirror(sink_path: str, sink_fd: int) -> None:
    """Bound the capture file and tail new bytes to the terminal at TRACE."""
    reader_fd = -1
    try:
        reader_fd = _move_above_std(
            os.open(
                sink_path,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0),
            )
        )
        # Refuse a replacement at the same path.
        opened, original = os.fstat(reader_fd), os.fstat(sink_fd)
        if (opened.st_dev, opened.st_ino) != (original.st_dev, original.st_ino):
            return
        stream = os.fdopen(reader_fd, "rb", buffering=0)
        reader_fd = -1
        with stream as sink:
            sink.seek(0, os.SEEK_END)
            while not _mirror_stop.is_set():
                chunk = sink.read(65536)
                if chunk:
                    _echo_chunk(chunk)
                if os.fstat(sink_fd).st_size >= _MAX_BYTES:
                    _rotate_capture(sink_path, sink_fd, sink)
                if not chunk:
                    _mirror_stop.wait(_CHECK_INTERVAL)
    except (OSError, ValueError):
        return  # Best-effort; must never affect the capture or the host.
    finally:
        if reader_fd >= 0:
            os.close(reader_fd)


def _start_mirror() -> None:
    """Start the capture maintenance and optional mirror thread once."""
    global _mirror_thread
    with _lock:
        if _mirror_thread is not None or _sink_path is None or _sink_fd is None:
            return
        thread = threading.Thread(
            target=_mirror,
            args=(_sink_path, _sink_fd),
            name="isaaccapture-native-capture",
            daemon=True,
        )
        try:
            thread.start()
        except RuntimeError:
            # Import-time setup must not fail if a thread cannot start.
            return
        _mirror_thread = thread
        atexit.register(_stop_mirror, thread, os.getpid())


def _stop_mirror(thread: threading.Thread, owner_pid: int) -> None:
    """Stop the creator's thread before empty-file cleanup runs."""
    if os.getpid() != owner_pid:
        return
    _mirror_stop.set()
    try:
        thread.join(timeout=1)
    except RuntimeError:
        pass


def follow_console_level(level: int) -> None:
    """Persist capture always and mirror it only when the console is at TRACE."""
    global _echo
    _echo = level <= TRACE
    ensure_sink()
