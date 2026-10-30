"""Resolve the real IsaacCapture SDK from ``ISAAC_ROOT`` for live tracking.

``ISAAC_ROOT`` is a separate, built IsaacCapture checkout; the importable package (with its native
libraries) is its ``build/python_package/Release/isaaccapture`` tree. No bundled copy and no
other installed copy may satisfy this dependency: the loaded package must originate from that tree.
Only the real path calls this; mock setup never reads ``ISAAC_ROOT`` or imports the SDK.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from .tensor_indices import ControllerInputIndex, HandInputIndex, HeadInputIndex

__all__ = ["IsaacCaptureUnavailable", "load_isaac_capture"]

_log = logging.getLogger(__name__)

_BUILD_SUBDIR = Path("build/python_package/Release")


class IsaacCaptureUnavailable(RuntimeError):
    """The real IsaacCapture SDK cannot be used; the message says what to fix."""


_loaded: SimpleNamespace | None = None


def load_isaac_capture() -> SimpleNamespace:
    """Import the IsaacCapture build under ``$ISAAC_ROOT`` and return the names the adapter composes.

    Cached after the first success. The returned namespace has the adapter's ``sdk`` surface
    (``HeadSource``, ``ControllersSource``, ``HandsSource``, ``OutputCombiner``, ``PluginConfig``,
    ``TeleopSessionConfig``, ``CloudXRLauncher``) plus ``TeleopSession``, ``version`` and ``package_dir``.
    """
    global _loaded
    if _loaded is not None:
        return _loaded
    root_text = os.environ.get("ISAAC_ROOT")
    if not root_text:
        raise IsaacCaptureUnavailable(
            "ISAAC_ROOT is not set; point it at a built IsaacCapture checkout to use real tracking"
        )
    root = Path(root_text).expanduser().resolve()
    build_dir = root / _BUILD_SUBDIR
    package_dir = build_dir / "isaaccapture"
    if not (package_dir / "__init__.py").is_file():
        raise IsaacCaptureUnavailable(
            f"no built isaaccapture package at {package_dir}; build IsaacCapture in ISAAC_ROOT={root}"
        )
    if "isaaccapture" not in sys.modules and str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    try:
        package = importlib.import_module("isaaccapture")
    except ImportError as exc:
        raise IsaacCaptureUnavailable(f"cannot import isaaccapture from {package_dir}: {exc}") from exc
    # Importing the package root already loads some of its native pieces, so this rejects a wrong-origin copy
    # before this module imports any of the submodules it needs, not before all SDK side effects.
    origin = Path(package.__file__).resolve()
    if package_dir not in origin.parents:
        raise IsaacCaptureUnavailable(
            f"isaaccapture was imported from {origin}, not from ISAAC_ROOT={root}; "
            "another copy is already loaded or shadows it"
        )
    try:
        teleop = importlib.import_module("isaaccapture.teleop_session_manager")
        sources = importlib.import_module("isaaccapture.retargeting_engine.deviceio_source_nodes")
        interface = importlib.import_module("isaaccapture.retargeting_engine.interface")
        indices = importlib.import_module("isaaccapture.retargeting_engine.tensor_types.indices")
        launcher = importlib.import_module("isaaccapture.cloudxr.launcher")
        runtime = importlib.import_module("isaaccapture.cloudxr.runtime")
    except ImportError as exc:  # missing/unloadable native library or module
        raise IsaacCaptureUnavailable(f"cannot import isaaccapture from {package_dir}: {exc}") from exc

    for local, real_name in (
        (HeadInputIndex, "HeadInputIndex"),
        (ControllerInputIndex, "ControllerInputIndex"),
        (HandInputIndex, "HandInputIndex"),
    ):
        real = getattr(indices, real_name)
        wrong = [m.name for m in local if getattr(real, m.name, None) != m.value]
        if wrong:
            raise IsaacCaptureUnavailable(
                f"isaaccapture {package.__version__} {real_name} differs from the adapter's for {wrong}"
            )

    def plugin_config(**kwargs):
        """The SDK ``PluginConfig``; an empty search path becomes the bundled plugins plus ``install/plugins``."""
        # The SDK drops only search paths that do not exist, so listing both is safe. The package bundles
        # just the plugins built into the wheel (gamepad, spacemouse); out-of-tree plugins such as Manus are
        # placed by IsaacCapture's own ``cmake --install --component manus`` under the checkout's
        # ``install/plugins``, which is why a request that searched the bundled directory alone never found them.
        kwargs["search_paths"] = kwargs.get("search_paths") or [
            teleop.BUNDLED_PLUGINS_DIR,
            root / "install" / "plugins",
        ]
        return teleop.PluginConfig(**kwargs)

    def session_config(**kwargs):
        """The SDK ``TeleopSessionConfig`` with synchronous stepping: the adapter loop owns the one cadence."""
        return teleop.TeleopSessionConfig(
            retargeting_execution=teleop.RetargetingExecutionConfig(mode=teleop.RetargetingExecutionMode.SYNC),
            **kwargs,
        )

    class AttachOnlyCloudXR:
        """CloudXR is owned by a separately started service: attach to it, do not start one.

        The liveness check below makes a missing service an error. The SDK constructor re-probes and would
        start a detached one itself only if the service exited between the two checks (a narrow race).
        """

        @staticmethod
        def launch_context(args):
            """Return a launcher attached to the running CloudXR service; ``args`` is unused (SDK defaults)."""
            run_dir = os.path.join(os.path.expanduser("~/.cloudxr"), "run")
            if not runtime.is_runtime_live(run_dir):
                raise IsaacCaptureUnavailable(
                    f"no CloudXR runtime is serving {run_dir}; start one first with "
                    "`python -m isaaccapture.cloudxr.service start` (capture-server attaches, leaves it running)"
                )
            return launcher.CloudXRLauncher(run_embedded=False)

    _loaded = SimpleNamespace(
        HeadSource=sources.HeadSource,
        ControllersSource=sources.ControllersSource,
        HandsSource=sources.HandsSource,
        OutputCombiner=interface.OutputCombiner,
        PluginConfig=plugin_config,
        TeleopSessionConfig=session_config,
        CloudXRLauncher=AttachOnlyCloudXR,
        TeleopSession=teleop.TeleopSession,
        version=package.__version__,
        package_dir=package_dir,
    )
    _log.info("using isaaccapture %s from %s", package.__version__, package_dir)
    return _loaded
