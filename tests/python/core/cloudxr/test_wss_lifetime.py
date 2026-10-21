# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proxy failure must retire every owned logging attachment."""

import asyncio
import logging

import pytest

from isaaccapture.cloudxr import wss


def test_proxy_failure_removes_all_owned_handlers(monkeypatch, tmp_path):
    loggers = [
        logging.getLogger(name)
        for name in (
            wss.log.name,
            "isaaccapture.cloudxr.oob_teleop_adb",
            "isaaccapture.cloudxr.oob_teleop_env",
            "isaaccapture.cloudxr.oob_teleop_hub",
            "isaaccapture.cloudxr.oob_teleop_lifecycle",
            "isaaccapture.cloudxr.wss.session_failure",
        )
    ]
    original = [(tuple(log.handlers), log.level, log.propagate) for log in loggers]

    def unavailable_certificate():
        raise RuntimeError("certificate unavailable")

    monkeypatch.setattr(wss, "default_cert_paths", unavailable_certificate)

    async def fail_proxy():
        future = asyncio.get_running_loop().create_future()
        await wss.run(tmp_path / "proxy.log", future)

    try:
        for _ in range(2):
            with pytest.raises(RuntimeError, match="certificate unavailable"):
                asyncio.run(fail_proxy())
        for log, (handlers, _, _) in zip(loggers, original, strict=True):
            assert tuple(log.handlers) == handlers
    finally:
        # Also retire leaked attachments on the pre-fix test run.
        for log, (handlers, level, propagate) in zip(loggers, original, strict=True):
            for handler in tuple(log.handlers):
                if handler not in handlers:
                    log.removeHandler(handler)
                    handler.close()
            log.setLevel(level)
            log.propagate = propagate
