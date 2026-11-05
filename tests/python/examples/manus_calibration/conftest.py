# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from repo_paths import repo_root  # noqa: E402

sys.path.insert(0, str(repo_root() / "examples" / "manus_calibration" / "python"))
sys.path.insert(
    0, str(repo_root() / "src" / "plugins" / "manus" / "calibration" / "python")
)

from isaaccapture_manus_calibration import protocol as proto  # noqa: E402
from isaaccapture_manus_calibration.worker import (  # noqa: E402
    AttemptState,
    CalibrationAttempt,
    CalibrationOutcome,
    CalibrationWorker,
)


class Folder:
    """The plain output folder under test: what calibrations left in it."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def saved_attempts(self) -> list[str]:
        """One entry per attempt that left .mcal files, so a pair counts once."""
        if not self.directory.is_dir():
            return []
        names = sorted(p.name for p in self.directory.glob("*.mcal"))
        return sorted({name.split("_")[0] for name in names})

    def leftovers(self) -> list[str]:
        """Anything in the folder that is not a finished .mcal file, such as a half-written one."""
        if not self.directory.is_dir():
            return []
        return sorted(
            p.name for p in self.directory.iterdir() if not p.name.endswith(".mcal")
        )


FAKE_WORKER = Path(__file__).with_name("fake_worker.py")
BOTH = (proto.Side.LEFT, proto.Side.RIGHT)


@pytest.fixture
def commands_log(tmp_path: Path) -> Path:
    return tmp_path / "commands.log"


@pytest.fixture
def make_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, commands_log: Path):
    """Build a CalibrationWorker around the scripted fake, with short supervision timeouts."""

    def factory(
        scenario: dict | None = None, **settings
    ) -> tuple[CalibrationWorker, Folder]:
        wrapper = tmp_path / "fake-calibrator"
        wrapper.write_text(
            f"#!{sys.executable}\nimport runpy\nrunpy.run_path({str(FAKE_WORKER)!r}, run_name='__main__')\n"
        )
        wrapper.chmod(0o755)
        monkeypatch.setenv("FAKE_WORKER_SCENARIO", json.dumps(scenario or {}))
        monkeypatch.setenv("FAKE_WORKER_LOG", str(commands_log))
        folder = Folder(tmp_path / "calibrations")
        timeouts = {
            "ready_timeout_s": 5,
            "cancel_grace_s": 2,
            "exit_grace_s": 2,
            "term_grace_s": 1,
            "kill_wait_s": 2,
        }
        return CalibrationWorker(
            wrapper, folder.directory, **{**timeouts, **settings}
        ), folder

    return factory


def commands(path: Path) -> list[str]:
    """Command types the fake worker received, in order."""
    return path.read_text().split() if path.exists() else []


async def drive(
    attempt: CalibrationAttempt,
    sides=BOTH,
    *,
    until: type | None = None,
    timeout: float = 20.0,
) -> CalibrationOutcome:
    """Start ``attempt`` and confirm every prompt until it settles."""
    await attempt.start(sides)
    confirmed: set[str] = set()
    async with asyncio.timeout(timeout):
        while attempt.state is not AttemptState.DONE:
            prompt = attempt.prompt
            if prompt is not None and prompt.token not in confirmed:
                confirmed.add(prompt.token)
                try:
                    await attempt.confirm(prompt.token)
                except Exception:
                    # A refused or abandoned confirm is part of what these drivers exercise; the loop
                    # ends on the attempt reaching DONE, and the caller inspects the outcome.
                    pass
            await asyncio.sleep(0.01)
    return await attempt.outcome()


async def wait_for(predicate, timeout: float = 10.0) -> None:
    async with asyncio.timeout(timeout):
        # The predicate reads plain attributes, so there is no event to await.
        while not predicate():  # noqa: ASYNC110
            await asyncio.sleep(0.01)
