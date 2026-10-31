# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercises the `__enter__`/`__exit__` pairing guard in the two
acquisition adapters directly: an object whose own `__enter__()` raised
must never see a matching `__exit__()` call, while an object that *did*
enter must still be released. See isaac_capture.py's `close()` and
depthai.py's `close()` for the guard these tests cover."""

from __future__ import annotations

from functools import partial
from unittest.mock import patch

import pytest

from capture_server import protocol as p
from capture_server.acquisition.depthai import DepthAIAdapter
from capture_server.acquisition.errors import CameraSourceError, SourceError
from capture_server.acquisition.isaac_capture import IsaacCaptureAdapter, TrackingSelection
from capture_server.mock import mock_depthai, mock_isaac_capture


async def test_isaac_capture_skips_session_exit_when_session_enter_raised():
    """`startup_error` fails TeleopSession.__enter__() specifically, after
    CloudXRLauncher.__enter__() has already succeeded -- close() must
    still release the launcher it entered, but must not call __exit__ on
    the session whose own __enter__() never returned."""
    selection = TrackingSelection(
        head=True, controller_left=False, controller_right=False, hand_left=False, hand_right=False
    )
    adapter = IsaacCaptureAdapter(
        sdk=mock_isaac_capture,
        session_factory=partial(
            mock_isaac_capture.TeleopSession, scenario="demo", seed=1, faults={"startup_error"}
        ),
        selection=selection,
        rate_hz=30,
    )

    with (
        patch.object(mock_isaac_capture.TeleopSession, "__exit__") as session_exit,
        patch.object(mock_isaac_capture.NoopContext, "__exit__") as launcher_exit,
    ):
        with pytest.raises(SourceError):
            await adapter.enter()

        await adapter.close()

        launcher_exit.assert_called_once()
        session_exit.assert_not_called()


async def test_depthai_skips_pipeline_exit_when_pipeline_enter_raised():
    """`camera_required_failure` fails Pipeline.__enter__() itself -- close()
    must not call __exit__ on a pipeline that never successfully entered."""
    camera = p.CameraStreamConfig(
        stream_id=1,
        name="cam-1",
        type=p.CameraType.OAK_D_COLOR,
        required=True,
        width=1280,
        height=720,
        fps=30.0,
        jpeg_quality=80,
    )
    adapter = DepthAIAdapter(
        sdk=mock_depthai,
        pipeline_factory=partial(
            mock_depthai.Pipeline, device_id="mock-oakd-1", faults={"camera_required_failure"}
        ),
        camera=camera,
    )

    with patch.object(mock_depthai.Pipeline, "__exit__") as pipeline_exit:
        with pytest.raises(CameraSourceError):
            await adapter.enter()

        await adapter.close()

        pipeline_exit.assert_not_called()
