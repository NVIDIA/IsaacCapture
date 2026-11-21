# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one rotating log file the session leader persists every record to."""

from __future__ import annotations

import atexit
import logging
import os
import stat
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ._core import (
    DATE_FORMAT,
    LINE_FORMAT,
    TRACE,
    _move_above_std,
    ensure_log_dir,
    root_logger,
)

_MAX_BYTES = 10 * 1024 * 1024  # 10 MiB
_BACKUP_COUNT = 5

_lock = threading.Lock()
_handler: logging.Handler | None = None


class _PrivateRotatingFileHandler(RotatingFileHandler):
    """Create rotating files exclusively, without symlinks, at mode 0600."""

    def _open(self):
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_APPEND
            | getattr(os, "O_NOFOLLOW", 0)
            # TextIOWrapper owns newline conversion; keep the fd binary on Windows.
            | getattr(os, "O_BINARY", 0)
        )
        fd = _move_above_std(os.open(self.baseFilename, flags, 0o600))
        try:
            return open(
                fd,
                self.mode,
                encoding=self.encoding,
                errors=getattr(self, "errors", None),
            )
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            raise


def _discard_if_empty(path: Path, identity: tuple[int, int], owner_pid: int) -> None:
    """Remove the creator's unchanged empty file without ending later atexit logging."""
    if os.getpid() != owner_pid:
        return
    handler = _handler
    if handler is not None:
        handler.acquire()
    try:
        if handler is not None:
            handler.flush()
        current = os.lstat(path)
        if (
            stat.S_ISREG(current.st_mode)
            and (current.st_dev, current.st_ino) == identity
            and current.st_size == 0
        ):
            if handler is not None:
                handler.close()
            current = os.lstat(path)
            if (
                stat.S_ISREG(current.st_mode)
                and (current.st_dev, current.st_ino) == identity
                and current.st_size == 0
            ):
                os.unlink(path)
    except OSError:
        return  # Best-effort atexit cleanup; nothing here may raise.
    finally:
        if handler is not None:
            handler.release()


def ensure_handler() -> logging.Handler:
    """Create the process's timestamped, PID-qualified ``TRACE`` file once."""
    global _handler
    if _handler is not None:
        return _handler
    with _lock:
        if _handler is not None:
            return _handler
        directory = ensure_log_dir()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        path = directory / f"{timestamp}.isaaccapture.{os.getpid()}.log"
        handler = _PrivateRotatingFileHandler(
            path,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(LINE_FORMAT, datefmt=DATE_FORMAT))
        handler.setLevel(TRACE)
        root_logger.addHandler(handler)
        # Preserve identity so exit cleanup cannot unlink a replacement.
        opened = os.fstat(handler.stream.fileno())
        atexit.register(
            _discard_if_empty, path, (opened.st_dev, opened.st_ino), os.getpid()
        )
        _handler = handler
        return _handler
