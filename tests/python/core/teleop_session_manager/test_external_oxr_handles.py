# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for external OpenXR handle support in TeleopSession.

Verifies that:
  - TeleopSessionConfig accepts an optional oxr_handles field.
  - TeleopSession.__enter__ passes external handles directly to
    DeviceIOSession.run() and skips OpenXRSession.create().
  - TeleopSession.__enter__ falls back to creating its own OpenXR session
    when no external handles are provided.

All OpenXR / DeviceIO native calls are mocked so no OpenXR runtime is required.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from isaaccapture.teleop_session_manager import TeleopSession, TeleopSessionConfig

if TYPE_CHECKING:
    from isaaccapture.oxr import OpenXRSessionHandles


# ============================================================================
# Helpers
# ============================================================================


def _make_stub_handles(
    instance: int = 0xABCD,
    session: int = 0x1234,
    space: int = 0x5678,
    proc_addr: int = 0x9ABC,
) -> OpenXRSessionHandles:
    """Create a lightweight stub that looks like OpenXRSessionHandles."""
    handles = MagicMock()
    handles.instance = instance
    handles.session = session
    handles.space = space
    handles.proc_addr = proc_addr
    return handles


def _make_empty_pipeline():
    """Create a mock pipeline with no leaf nodes (no trackers)."""
    pipeline = MagicMock()
    pipeline.get_leaf_nodes.return_value = []
    return pipeline


@contextmanager
def _mock_deviceio_and_oxr():
    """Patch DeviceIOSession.run and the OpenXRSession constructor to avoid native calls.

    Yields a namespace object with attributes:
        - deviceio_run:      the mock replacing DeviceIOSession.run
        - oxr_cls:           the mock replacing the OpenXRSession class
        - mock_dio_session:  the fake DeviceIOSession instance
        - mock_oxr_session:  the fake OpenXRSession instance
    """
    mock_dio_session = MagicMock()
    mock_dio_session.__enter__ = MagicMock(return_value=mock_dio_session)
    mock_dio_session.__exit__ = MagicMock(return_value=False)

    mock_oxr_session = MagicMock()
    mock_oxr_session.__enter__ = MagicMock(return_value=mock_oxr_session)
    mock_oxr_session.__exit__ = MagicMock(return_value=False)
    mock_oxr_session.get_handles.return_value = _make_stub_handles()

    with (
        patch(
            "isaaccapture.deviceio.DeviceIOSession.run", return_value=mock_dio_session
        ) as dio_run,
        patch(
            "isaaccapture.oxr.OpenXRSession", return_value=mock_oxr_session
        ) as oxr_cls,
    ):
        ns = MagicMock()
        ns.deviceio_run = dio_run
        ns.oxr_cls = oxr_cls
        ns.mock_dio_session = mock_dio_session
        ns.mock_oxr_session = mock_oxr_session
        yield ns


# ============================================================================
# TeleopSessionConfig.oxr_handles
# ============================================================================


class TestTeleopSessionConfigOxrHandles:
    """Tests for the oxr_handles field on TeleopSessionConfig."""

    def test_oxr_handles_defaults_to_none(self):
        """oxr_handles is None when not explicitly set."""
        config = TeleopSessionConfig(
            app_name="test",
            pipeline=_make_empty_pipeline(),
        )
        assert config.oxr_handles is None

    def test_oxr_handles_accepts_stub(self):
        """oxr_handles stores the provided value."""
        handles = _make_stub_handles()
        config = TeleopSessionConfig(
            app_name="test",
            pipeline=_make_empty_pipeline(),
            oxr_handles=handles,
        )
        assert config.oxr_handles is handles


# ============================================================================
# TeleopSession.__enter__ with external handles
# ============================================================================


class TestTeleopSessionExternalHandles:
    """Tests for TeleopSession when external OpenXR handles are provided."""

    def test_factory_receives_aggregated_extensions_before_deviceio(self):
        handles = _make_stub_handles()
        tracker = MagicMock()
        factory = MagicMock(return_value=handles)
        config = TeleopSessionConfig(
            app_name="test",
            pipeline=_make_empty_pipeline(),
            trackers=[tracker],
            oxr_session_factory=factory,
        )
        with (
            _mock_deviceio_and_oxr() as mocks,
            patch(
                "isaaccapture.deviceio.DeviceIOSession.get_required_extensions",
                return_value=["XR_EXT_hand_tracking", "XR_NVX1_push_tensor"],
            ) as extensions,
        ):
            factory.side_effect = lambda required: (
                mocks.deviceio_run.assert_not_called() or handles
            )
            with TeleopSession(config):
                assert tracker in extensions.call_args.args[0]
                factory.assert_called_once_with(extensions.return_value)
                assert mocks.deviceio_run.call_args.args[1] is handles
                mocks.oxr_cls.assert_not_called()

    def test_factory_failure_does_not_start_deviceio(self):
        config = TeleopSessionConfig(
            app_name="test",
            pipeline=_make_empty_pipeline(),
            oxr_session_factory=MagicMock(side_effect=RuntimeError("viz failed")),
        )
        with _mock_deviceio_and_oxr() as mocks:
            with pytest.raises(RuntimeError, match="viz failed"):
                with TeleopSession(config):
                    pass
            mocks.deviceio_run.assert_not_called()

    @pytest.mark.parametrize("field", ["oxr_handles", "joint_publisher"])
    def test_factory_rejects_competing_session_owners(self, field):
        with pytest.raises(ValueError, match="oxr_session_factory requires LIVE"):
            TeleopSessionConfig(
                app_name="test",
                pipeline=_make_empty_pipeline(),
                oxr_session_factory=MagicMock(),
                **{field: MagicMock()},
            )

    def test_skips_oxr_session_creation(self):
        """When oxr_handles is set, OpenXRSession.create() must not be called."""
        external_handles = _make_stub_handles()
        config = TeleopSessionConfig(
            app_name="test",
            pipeline=_make_empty_pipeline(),
            oxr_handles=external_handles,
        )

        with _mock_deviceio_and_oxr() as mocks:
            session = TeleopSession(config)
            session.__enter__()
            try:
                mocks.oxr_cls.assert_not_called()
            finally:
                session.__exit__(None, None, None)

    def test_passes_external_handles_to_deviceio(self):
        """External handles are forwarded to DeviceIOSession.run()."""
        external_handles = _make_stub_handles()
        config = TeleopSessionConfig(
            app_name="test",
            pipeline=_make_empty_pipeline(),
            oxr_handles=external_handles,
        )

        with _mock_deviceio_and_oxr() as mocks:
            session = TeleopSession(config)
            session.__enter__()
            try:
                mocks.deviceio_run.assert_called_once()
                _, call_kwargs = mocks.deviceio_run.call_args
                if call_kwargs:
                    actual_handles = call_kwargs.get(
                        "handles", mocks.deviceio_run.call_args[0][1]
                    )
                else:
                    actual_handles = mocks.deviceio_run.call_args[0][1]
                assert actual_handles is external_handles
            finally:
                session.__exit__(None, None, None)

    def test_oxr_session_attribute_remains_none(self):
        """When using external handles, session.oxr_session stays None."""
        config = TeleopSessionConfig(
            app_name="test",
            pipeline=_make_empty_pipeline(),
            oxr_handles=_make_stub_handles(),
        )

        with _mock_deviceio_and_oxr():
            session = TeleopSession(config)
            session.__enter__()
            try:
                assert session.oxr_session is None
            finally:
                session.__exit__(None, None, None)

    def test_deviceio_session_is_set(self):
        """DeviceIO session is created even when using external handles."""
        config = TeleopSessionConfig(
            app_name="test",
            pipeline=_make_empty_pipeline(),
            oxr_handles=_make_stub_handles(),
        )

        with _mock_deviceio_and_oxr():
            session = TeleopSession(config)
            session.__enter__()
            try:
                assert session.deviceio_session is not None
            finally:
                session.__exit__(None, None, None)


# ============================================================================
# TeleopSession.__enter__ without external handles (standalone fallback)
# ============================================================================


class TestTeleopSessionStandaloneFallback:
    """Tests for TeleopSession when no external handles are provided."""

    def test_creates_own_oxr_session(self):
        """Without oxr_handles, OpenXRSession.create() is called."""
        config = TeleopSessionConfig(
            app_name="test",
            pipeline=_make_empty_pipeline(),
        )

        with _mock_deviceio_and_oxr() as mocks:
            session = TeleopSession(config)
            session.__enter__()
            try:
                mocks.oxr_cls.assert_called_once()
            finally:
                session.__exit__(None, None, None)

    def test_uses_handles_from_own_session(self):
        """Standalone mode passes handles from the internally-created session."""
        config = TeleopSessionConfig(
            app_name="test",
            pipeline=_make_empty_pipeline(),
        )

        with _mock_deviceio_and_oxr() as mocks:
            session = TeleopSession(config)
            session.__enter__()
            try:
                mocks.deviceio_run.assert_called_once()
                actual_handles = mocks.deviceio_run.call_args[0][1]
                # Should be the handles returned by the mock OXR session
                expected = mocks.mock_oxr_session.get_handles.return_value
                assert actual_handles is expected
            finally:
                session.__exit__(None, None, None)

    def test_oxr_session_attribute_is_set(self):
        """In standalone mode, session.oxr_session is populated."""
        config = TeleopSessionConfig(
            app_name="test",
            pipeline=_make_empty_pipeline(),
        )

        with _mock_deviceio_and_oxr():
            session = TeleopSession(config)
            session.__enter__()
            try:
                assert session.oxr_session is not None
            finally:
                session.__exit__(None, None, None)
