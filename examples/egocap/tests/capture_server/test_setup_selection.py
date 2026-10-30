"""Setup-time selection of the acquisition implementation (mock vs real) from ServerConfig.mock.

Import-isolation checks run in a fresh interpreter, because what a process has already imported
is the thing under test.
"""

import subprocess
import sys

import pytest

from capture_server import protocol as p
from capture_server.acquisition import SourceError
from capture_server.launch import build_mock_source_factory, build_real_source_factory, build_source_factory
from capture_server.server import CaptureServer, ServerConfig

SDK_MODULES = ("isaaccapture", "isaacteleop", "depthai")


def _run_isolated(code: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run ``code`` in a fresh interpreter with ISAAC_ROOT removed, so only the code under test can read it."""
    import os

    env = {k: v for k, v in os.environ.items() if k != "ISAAC_ROOT"}
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=False)


def _never(*_args):
    raise AssertionError("no session is begun in these tests")


def test_server_config_defaults_to_the_mock():
    """Library callers and tests keep the mock unless a setup site explicitly chooses real acquisition."""
    assert ServerConfig().mock is True


@pytest.mark.parametrize("mock,expected", [(True, p.CaptureMode.MOCK), (False, p.CaptureMode.LIVE)])
def test_reported_capture_mode_is_derived_from_the_setup_flag(mock, expected):
    """The server has no separate mode setting: the capture mode always agrees with ServerConfig.mock."""
    server = CaptureServer(ServerConfig(mock=mock), source_factory=_never)
    assert server._mode is expected


def test_capture_server_no_longer_accepts_an_independent_mode():
    """A caller cannot construct a server whose reported mode disagrees with its selected implementation."""
    with pytest.raises(TypeError):
        CaptureServer(ServerConfig(), mode=p.CaptureMode.LIVE, source_factory=_never)  # type: ignore[call-arg]


def test_server_selects_its_own_implementation_from_the_setup_flag():
    """With no source_factory the server chooses the mock or real factory itself, once, at construction."""
    assert CaptureServer(ServerConfig(mock=True))._mode is p.CaptureMode.MOCK
    assert CaptureServer(ServerConfig(mock=False))._mode is p.CaptureMode.LIVE


def test_a_mock_factory_cannot_be_paired_with_real_setup():
    """mock=False with a launch-built mock factory is rejected, so LIVE can never serve synthetic data."""
    with pytest.raises(ValueError, match="for mock acquisition but ServerConfig.mock=False"):
        CaptureServer(ServerConfig(mock=False), source_factory=build_mock_source_factory(scenario="demo", seed=0))


def test_mock_setup_selects_the_mock_factory_and_real_setup_fails_clearly():
    """mock=True builds the tagged mock factory; mock=False the tagged real one."""
    assert build_source_factory(ServerConfig(mock=True)).mock is True
    real = build_source_factory(ServerConfig(mock=False))
    assert real.mock is False and callable(build_real_source_factory(ServerConfig(mock=False)))
    camera = p.CameraStreamConfig(
        stream_id=1,
        name="cam",
        type=p.CameraType.OAK_D_COLOR,
        required=True,
        width=1280,
        height=720,
        fps=30.0,
        jpeg_quality=80,
    )
    with pytest.raises(ValueError, match="no station device"):  # real cameras come only from station config
        real(p.BeginRequest(tracking=p.TrackingConfig(sources=())), (camera,))


async def test_unknown_mock_scenario_is_rejected_and_mock_options_reach_only_the_mock():
    """Mock-only options are validated and applied only on the mock path."""
    with pytest.raises(ValueError, match="unknown mock scenario"):
        build_source_factory(ServerConfig(mock=True), scenario="not-a-scenario")
    begin = p.BeginRequest(tracking=p.TrackingConfig(sources=(p.SourceType.HEAD,)))
    faulted = build_source_factory(
        ServerConfig(mock=True), scenario="demo", seed=4, faults=frozenset({"startup_error"})
    )
    acq = faulted(begin, ())
    try:
        with pytest.raises(SourceError):  # the forwarded fault actually reached the mock IsaacCapture
            await acq.enter()
    finally:
        await acq.close()


def test_mock_setup_needs_no_isaac_root_and_loads_no_hardware_sdk():
    """With ISAAC_ROOT unset, building the mock factory imports neither hardware SDK."""
    result = _run_isolated(
        "import sys\n"
        "from capture_server.launch import build_source_factory\n"
        "from capture_server.server import ServerConfig\n"
        "build_source_factory(ServerConfig(mock=True))\n"
        f"loaded = [m for m in {SDK_MODULES!r} if m in sys.modules]\n"
        "assert not loaded, loaded\n"
        "print('OK')\n"
    )
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr


def test_real_setup_never_loads_the_mock_sdk_files():
    """Choosing real acquisition does not import the mock SDKs, so mock data cannot be reached by accident."""
    result = _run_isolated(
        "import sys\n"
        "from capture_server.launch import build_source_factory\n"
        "from capture_server.server import ServerConfig\n"
        "build_source_factory(ServerConfig(mock=False))\n"
        "loaded = [m for m in sys.modules if m.startswith('capture_server.mock')]\n"
        "assert not loaded, loaded\n"
        "print('OK')\n"
    )
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr
