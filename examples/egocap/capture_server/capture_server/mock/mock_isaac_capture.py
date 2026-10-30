# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""A synthetic approximation of the public IsaacCapture API subset the
production :mod:`capture_server.acquisition.isaac_capture` adapter calls,
inspected against an IsaacCapture checkout at pinned commit
``b418850e9ec47067b7c0c63f35944bfa0cc5ce37`` (reconfirm before relying on
this file for anything beyond this package's own needs):

- ``teleop_session_manager/teleop_session.py`` -- ``TeleopSession``,
  ``__enter__``/``__exit__``/``step()``/``get_status()``. ``step()`` takes
  only optional keyword arguments and returns ``Dict[str, TensorGroup]``;
  it is synchronous and unpaced -- "session fields such as frame_count...
  are intentionally owned by the application thread that calls step()".
- ``teleop_session_manager/status.py`` -- ``ProviderType``/``ProviderState``/
  ``DeviceState``/``StatusReason`` (native ``str`` enums) and
  ``ProviderStatus``/``DeviceStatus``/``StatusSnapshot`` dataclasses,
  reproduced here field-for-field (native ``ProviderStatus`` has no
  ``plugin`` field -- the production adapter infers that mapping).
- ``retargeting_engine/interface/tensor_group.py`` -- ``OptionalTensorGroup``,
  reproduced here (its ``is_none``/``__getitem__``/``__setitem__`` contract,
  minus the DLPack/tensor-type-validation machinery no caller the
  production adapter uses needs).
- ``retargeting_engine/tensor_types/standard_types.py`` -- the
  ``HeadInput``/``ControllerInput``/``HandInput`` field order reproduced on
  :class:`HeadInputIndex`/:class:`ControllerInputIndex`/
  :class:`HandInputIndex` below.
- ``retargeting_engine/tensor_types/indices.py`` -- :class:`HandJointIndex`,
  reproduced with the same 26 OpenXR joint names/order.
- ``retargeting_engine/deviceio_source_nodes/{head,controllers,
  hands}_source.py`` and ``retargeting_engine/interface/output_combiner.py``
  -- :class:`HeadSource`/:class:`ControllersSource`/:class:`HandsSource`'s
  ``name``-constructed, ``.output(selector)``-wired public shape and
  :class:`OutputCombiner`'s plain ``{leaf_key: source.output(selector)}``
  dict, matching ``examples/deviceio_live_view/.../deviceio_viser.py``'s
  ``build_all_human_pipeline`` reference construction pattern -- not a
  graph engine; this mock only inspects which source type/selector each
  leaf key names.
- ``cloudxr/launcher.py`` -- ``CloudXRLauncher.launch_context``/
  ``NoopContext``; this mock always behaves like the real SDK's own
  no-launched-runtime ``NoopContext`` path, since nothing here ever starts
  a real CloudXR runtime.

This module approximates ONLY that subset -- not a graph engine, not
unrelated retargeters/recording/visualization/async-pipelined execution,
and not a protocol-shaped replacement (native status enums are strings;
``capture_server.protocol``'s are integers -- the production adapter does
that mapping, not this file). It must not import ``capture_server.protocol``
and does not.

Deterministic hand/head motion math below is adapted from an earlier
Pico-mock model, trimmed to the head/controller/26-joint-hand subset
``TeleopSession.step()`` actually produces, and folded into
:class:`_Operator` below. All scenario/fault state lives on the
:class:`TeleopSession` instance that owns it -- there is no
process-global mutable registry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from functools import lru_cache
from typing import Any

import numpy as np

from capture_server.acquisition.clock import Clock, MonotonicClock
from capture_server.acquisition.tensor_indices import (
    ControllerInputIndex,
    HandInputIndex,
    HandJointIndex,
    HeadInputIndex,
)

__all__ = [
    "MANUS_PLUGIN_NAME",
    "CloudXRLauncher",
    "ControllerInputIndex",
    "ControllersSource",
    "DeviceState",
    "DeviceStatus",
    "HandInputIndex",
    "HandJointIndex",
    "HandsSource",
    "HeadInputIndex",
    "HeadSource",
    "KNOWN_FAULTS",
    "NoopContext",
    "OptionalTensorGroup",
    "OutputCombiner",
    "PluginConfig",
    "ProviderState",
    "ProviderStatus",
    "ProviderType",
    "SCENARIOS",
    "StatusReason",
    "StatusSnapshot",
    "TeleopSession",
    "TeleopSessionConfig",
]

# Named IC-side faults, scoped to this file: all
# scenario state belongs to the created mock API/session instance; no
# process-global mutable fault registry. Each TeleopSession owns its own
# active set, settable at construction and live via set_faults() -- a test
# that wants to flip a fault on an already-running session keeps an
# explicit handle to that session's own instance, not a shared registry.
KNOWN_FAULTS = frozenset(
    {
        "startup_error",
        "provider_failed",
        "provider_recovered",
        # Distinct from both of the above: "startup_error" fails the whole
        # native entry; "provider_failed" degrades an already-loaded Manus
        # provider post-entry. This fault instead fails ONLY the Manus
        # plugin's own load step during __enter__ -- per the
        # PluginRequest.required policy, a required plugin failing to load
        # must fail session establishment (IC_ERROR), an optional one must
        # not. Meaningless unless the configured plugins actually include
        # an enabled Manus PluginConfig.
        "manus_plugin_load_failed",
        "sample_freeze",
        "fatal_acquisition_error",
        # Per-sample group presence/validity -- independent of device
        # inventory (get_status()) and of the static tracking selection
        # (whether a group is in config.pipeline.outputs at all): a
        # selected, registered device can still report this specific
        # sample as absent or present-but-invalid.
        "controller_left_absent",
        "controller_right_absent",
        "hand_left_absent",
        "hand_right_absent",
        "controller_pose_invalid",
        "hand_pose_invalid",
        # Distinct from hand_pose_invalid (every joint invalid, the whole
        # hand untracked): a mixed validity mask where only some joints
        # are untracked -- a present, partially-usable hand, not an
        # all-or-nothing one.
        "hand_pose_partially_invalid",
    }
)

# --------------------------------------------------------------------------- native status types
# Reproduced field-for-field from teleop_session_manager/status.py.


class ProviderType(str, Enum):
    PLUGIN = "plugin"
    OPENXR_RUNTIME = "openxr_runtime"


class ProviderState(str, Enum):
    INITIALIZING = "initializing"
    AVAILABLE = "available"
    FAILED = "failed"
    STOPPED = "stopped"


class DeviceState(str, Enum):
    INITIALIZING = "initializing"
    UNKNOWN = "unknown"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"


class StatusReason(str, Enum):
    NONE = "none"
    PROVIDER_INITIALIZING = "provider_initializing"
    PROVIDER_FAILED = "provider_failed"
    PROVIDER_STOPPED = "provider_stopped"
    NO_REPORT = "no_report"
    STALE_REPORT = "stale_report"
    MALFORMED_REPORT = "malformed_report"
    UNSUPPORTED_SCHEMA_VERSION = "unsupported_schema_version"
    NO_HARDWARE_SIGNAL = "no_hardware_signal"
    CALIBRATION_FAILED = "calibration_failed"
    NO_CURRENT_DATA = "no_current_data"
    DEVICE_ERROR = "device_error"
    DISABLED_BY_CONFIGURATION = "disabled_by_configuration"
    PROCESS_EXITED = "process_exited"
    PROCESS_SIGNALED = "process_signaled"
    PROCESS_OBSERVATION_ERROR = "process_observation_error"
    STARTUP_FAILED = "startup_failed"
    RUNTIME_UPDATE_FAILED = "runtime_update_failed"
    OPENXR_SESSION_READY = "openxr_session_ready"
    EXTERNAL_OPENXR_HEALTH_UNKNOWN = "external_openxr_health_unknown"
    OPENXR_FORM_FACTOR_UNAVAILABLE = "openxr_form_factor_unavailable"
    OPENXR_SESSION_LOST = "openxr_session_lost"
    OPENXR_INSTANCE_LOST = "openxr_instance_lost"
    OPENXR_POLL_ERROR = "openxr_poll_error"


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    id: str
    type: ProviderType
    name: str
    status: ProviderState
    reason: StatusReason
    error: str
    pid: int | None = None
    exit_code: int | None = None
    term_signal: int | None = None


@dataclass(frozen=True, slots=True)
class DeviceStatus:
    id: str
    path: str
    type: str
    description: str
    provider_id: str
    status: DeviceState
    reason: StatusReason
    error: str


@dataclass(frozen=True, slots=True)
class StatusSnapshot:
    updated_at_ns: int
    providers: tuple[ProviderStatus, ...]
    devices: tuple[DeviceStatus, ...]


_OPENXR_PROVIDER_ID = "provider-openxr"
_MANUS_PROVIDER_ID = "provider-manus"
#: Mirrors the real Manus manifest name (src/plugins/manus/plugin.yaml: ``name: manus_hand_plugin``). The
#: real plugin manager resolves ``PluginConfig.plugin_name`` by exact match against that field and names the
#: provider after it, so the mock accepts and reports exactly that spelling -- a mock that accepted a looser
#: name would pass while the live request went undiscovered. A test asserts this equals the adapter's
#: ``MANUS_PLUGIN_NAME``; this file must not import the adapter (it would pull in ``capture_server.protocol``).
MANUS_PLUGIN_NAME = "manus_hand_plugin"
_HEADSET_DEVICE_ID = "device-headset"
_LEFT_HAND_DEVICE_ID = "device-hand-left"
_RIGHT_HAND_DEVICE_ID = "device-hand-right"
_LEFT_CONTROLLER_DEVICE_ID = "device-controller-left"
_RIGHT_CONTROLLER_DEVICE_ID = "device-controller-right"


# --------------------------------------------------------------------------- native tensor groups
def _joint_radius(name: str) -> float:
    if "TIP" in name:
        return 0.005
    if "DISTAL" in name:
        return 0.007
    if "INTERMEDIATE" in name:
        return 0.008
    if "PROXIMAL" in name:
        return 0.009
    if "METACARPAL" in name:
        return 0.010
    if name == "WRIST":
        return 0.014
    if name == "PALM":
        return 0.045
    raise AssertionError(f"unrecognized hand joint name: {name!r}")  # pragma: no cover


HAND_JOINT_RADII: tuple[float, ...] = tuple(_joint_radius(j.name) for j in HandJointIndex)
assert len(HAND_JOINT_RADII) == 26


class OptionalTensorGroup:
    """Reproduces retargeting_engine/interface/tensor_group.py's
    ``OptionalTensorGroup``: starts absent (``is_none`` True); writing any
    index via ``__setitem__`` makes it present; reading while absent
    raises ``ValueError``, matching the real class exactly (minus its
    DLPack/snapshot/tensor-type-validation machinery, which no caller the
    production adapter uses needs)."""

    def __init__(self, size: int) -> None:
        self._values: list[Any] = [None] * size
        self._is_none = True

    @property
    def is_none(self) -> bool:
        return self._is_none

    def set_none(self) -> None:
        self._is_none = True

    def __getitem__(self, index: IntEnum) -> Any:
        if self._is_none:
            raise ValueError("cannot read from an absent OptionalTensorGroup")
        return self._values[int(index)]

    def __setitem__(self, index: IntEnum, value: Any) -> None:
        self._is_none = False
        self._values[int(index)] = value


# --------------------------------------------------------------------------- selectors/config
# Reproduced from retargeting_engine/deviceio_source_nodes/{head,controllers,
# hands}_source.py's public construction/selector surface: each source node
# is constructed with a channel ``name`` and exposes ``.output(selector)``
# for wiring into an OutputCombiner -- examples/deviceio_live_view/python/
# isaaccapture_examples/deviceio_live_view/deviceio_viser.py:
# build_all_human_pipeline is the reference construction pattern.
class HeadSource:
    def __init__(self, name: str) -> None:
        self.name = name

    def output(self, selector: str) -> tuple["HeadSource", str]:
        return (self, selector)


class ControllersSource:
    LEFT = "left"
    RIGHT = "right"

    def __init__(self, name: str) -> None:
        self.name = name

    def output(self, selector: str) -> tuple["ControllersSource", str]:
        return (self, selector)


class HandsSource:
    LEFT = "left"
    RIGHT = "right"

    def __init__(self, name: str) -> None:
        self.name = name

    def output(self, selector: str) -> tuple["HandsSource", str]:
        return (self, selector)


class OutputCombiner:
    """Reproduces retargeting_engine/interface/output_combiner.py's public
    construction shape: a plain ``{leaf_key: source.output(selector)}``
    dict, keyed exactly like ``step()``'s returned result dict. The real
    constructor rejects an empty ``output_mapping`` with ``ValueError`` --
    reproduced here too, rather than silently accepting a graph shape the
    native API has no equivalent for."""

    def __init__(self, outputs: dict[str, tuple[Any, str]]) -> None:
        if not outputs:
            raise ValueError("OutputCombiner requires at least one output mapping entry")
        self.outputs = outputs


@dataclass(frozen=True)
class PluginConfig:
    """Reproduces the real ``PluginConfig``'s public construction shape
    (teleop_session_manager/config.py), field-for-field."""

    plugin_name: str
    plugin_root_id: str
    search_paths: list[Any] = field(default_factory=list)
    enabled: bool = True
    plugin_args: list[str] = field(default_factory=list)
    required: bool = False


@dataclass(frozen=True)
class TeleopSessionConfig:
    """Reproduces the real ``TeleopSessionConfig``'s public construction
    shape (``app_name``/``pipeline``/``plugins``) -- not a graph engine:
    this mock only inspects ``pipeline.outputs`` to know which leaf keys
    and source-node types to synthesize samples for."""

    app_name: str
    pipeline: OutputCombiner
    plugins: list[PluginConfig] = field(default_factory=list)


# --------------------------------------------------------------------------- CloudXRLauncher
class NoopContext:
    """Mirrors the real SDK's own no-launched-runtime context -- this mock
    never starts a real CloudXR runtime, so it always behaves this way."""

    @property
    def owns_runtime(self) -> bool:
        return False

    def __enter__(self) -> "NoopContext":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        return None


class CloudXRLauncher:
    """Only the entry point the production adapter calls."""

    @staticmethod
    def launch_context(*_args: Any, **_kwargs: Any) -> NoopContext:
        return NoopContext()


# --------------------------------------------------------------------------- deterministic motion
# A reduced motion model, scoped to the head/controller/26-joint-hand
# subset TeleopSession.step() produces. No body skeleton, motion-tracker
# channels, or Sharpa-exo/Manus-flex-sensor extras -- none of those map
# onto this SDK subset. The rotation/quaternion helpers, the
# keyframe/scenario blending system and the per-finger forward-kinematics
# hand model produce exactly the 26-joint OpenXR layout (palm, wrist,
# thumb x4, then index/middle/ring/little x5) this mock's tensor groups
# use.
#
# All poses are [x, y, z, qx, qy, qz, qw] in a Y-up, -Z-forward, +X-right
# frame with the origin on the floor under the operator (a head ~1.65 m up).
# Nothing here is captured from hardware: values are physically plausible
# (constant bone lengths, smooth min-jerk motion between keyframes, small
# deterministic tremor), not real. Output is a pure function of elapsed
# time, so it does not depend on poll rate; the same seed always gives the
# same operator.

HEAD_HEIGHT_M = 1.65
SCENARIOS = ("idle", "demo", "reach", "grasp", "point", "wave", "calibrate", "record_cycle")


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _mat_to_quat(m: np.ndarray) -> np.ndarray:
    t = float(np.trace(m))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    return -q if q[3] < 0 else q


def _pose7(pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
    return np.concatenate([pos, _mat_to_quat(rot)])


def _min_jerk(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x**3 * (10 - 15 * x + 6 * x * x)


def _kf(lp, rp, lc=0.05, rc=0.05, trig=(0.0, 0.0), grip=(0.0, 0.0), spread=0.0, tilt=0.5):
    def five(c):
        return np.full(5, c, dtype=np.float64) if np.isscalar(c) else np.asarray(c, dtype=np.float64)

    return {
        "pos": (np.asarray(lp, dtype=np.float64), np.asarray(rp, dtype=np.float64)),
        "curl": (five(lc), five(rc)),
        "trig": np.asarray(trig, dtype=np.float64),
        "grip": np.asarray(grip, dtype=np.float64),
        "spread": spread,
        "tilt": tilt,
    }


_KF = {
    "rest": _kf((-0.20, 1.05, -0.35), (0.20, 1.05, -0.35)),
    "reach": _kf((-0.15, 1.25, -0.62), (0.15, 1.25, -0.62), 0.1, 0.1, spread=0.2, tilt=0.2),
    "grasp": _kf((-0.10, 1.15, -0.45), (0.10, 1.15, -0.45), 0.85, 0.85, (1.0, 1.0), (1.0, 1.0), tilt=0.8),
    "point": _kf((-0.20, 1.05, -0.35), (0.12, 1.30, -0.55), 0.05, [0.7, 0.0, 0.9, 0.9, 0.9], (0.0, 0.0), tilt=0.3),
    "wave": _kf((-0.20, 1.05, -0.35), (0.25, 1.62, -0.30), 0.05, 0.0, spread=0.6, tilt=0.0),
    "calibrate": _kf((-0.14, 1.45, -0.42), (0.14, 1.45, -0.42), 0.0, 0.0, spread=0.5, tilt=-0.2),
}
_PLAN = {
    "idle": [("rest", 10.0)],
    "reach": [("rest", 2.0), ("reach", 4.0)],
    "grasp": [("reach", 2.5), ("grasp", 3.0), ("rest", 2.0)],
    "point": [("rest", 2.0), ("point", 4.0)],
    "wave": [("rest", 2.0), ("wave", 5.0)],
    "calibrate": [("rest", 2.0), ("calibrate", 6.0)],
    "record_cycle": [("rest", 2.0), ("reach", 2.5), ("grasp", 3.0), ("reach", 2.5), ("rest", 2.0)],
    "demo": [("rest", 3.0), ("reach", 3.0), ("grasp", 3.0), ("point", 3.0), ("wave", 4.0),
             ("calibrate", 3.0)],
}  # fmt: skip
_TRANSITION_S = 0.9


def _blend(a, b, u):
    return {
        "pos": tuple(a["pos"][i] * (1 - u) + b["pos"][i] * u for i in range(2)),
        "curl": tuple(a["curl"][i] * (1 - u) + b["curl"][i] * u for i in range(2)),
        "trig": a["trig"] * (1 - u) + b["trig"] * u,
        "grip": a["grip"] * (1 - u) + b["grip"] * u,
        "spread": a["spread"] * (1 - u) + b["spread"] * u,
        "tilt": a["tilt"] * (1 - u) + b["tilt"] * u,
    }


_FINGER_LEN = {
    "index": (0.055, 0.045, 0.026, 0.020, 0.016),
    "middle": (0.052, 0.050, 0.029, 0.021, 0.017),
    "ring": (0.048, 0.046, 0.027, 0.020, 0.016),
    "little": (0.044, 0.036, 0.020, 0.018, 0.015),
}
_FINGER_S = {"index": -0.030, "middle": -0.010, "ring": 0.010, "little": 0.030}
_FINGER_MAX = (math.radians(85), math.radians(100), math.radians(70))


@dataclass(frozen=True)
class _HandPose:
    wrist: np.ndarray  # (7,)
    controller: np.ndarray  # (7,)
    joints: np.ndarray  # (26, 7)
    curl: np.ndarray  # (5,)
    spread: float
    trigger: float
    grip: float


def _hand(side: int, wrist: np.ndarray, r_hand: np.ndarray, curl: np.ndarray, spread: float) -> np.ndarray:
    joints = np.zeros((26, 7), dtype=np.float64)

    def put(i, local, rot_local=np.eye(3)):
        joints[i] = _pose7(wrist + r_hand @ local, r_hand @ rot_local)

    put(1, np.zeros(3))
    put(0, np.array([0.0, 0.0, -0.045]))
    t_dir = np.array([-0.55 * side, -0.15, -0.8])
    t_dir /= np.linalg.norm(t_dir)
    base = np.array([-0.030 * side, -0.010, -0.025])
    lens = (0.045, 0.038, 0.030, 0.020)
    ang = curl[0] * math.radians(60)
    pos, d = base, t_dir
    for k in range(4):
        put(2 + k, pos, _rot_x(-ang * min(k, 2) / 2))
        bend = _rot_x(-ang / 2 * (1 if k else 0.3)) @ d
        pos = pos + d * lens[k]
        d = bend / np.linalg.norm(bend)
    for fi, name in enumerate(("index", "middle", "ring", "little")):
        lm, lp, li, ld, lt = _FINGER_LEN[name]
        s = _FINGER_S[name] * (1 + 0.8 * spread) * side
        c = float(curl[1 + fi])
        yaw = _rot_y(-math.radians(10) * spread * (fi - 1.5) * side)
        a1, a2, a3 = (c * m for m in _FINGER_MAX)
        base_i = 6 + 5 * fi
        pos = np.array([s, 0.0, -0.012])
        put(base_i, pos, yaw)
        pos = pos + yaw @ np.array([0.0, 0.0, -lm])
        cum = 0.0
        for j, (ln, da) in enumerate(((lp, a1), (li, a2), (ld, a3)), start=1):
            cum += da
            rot = yaw @ _rot_x(-cum + da)
            put(base_i + j, pos, rot)
            seg = yaw @ _rot_x(-cum) @ np.array([0.0, 0.0, -ln])
            pos = pos + seg
        put(base_i + 4, pos, yaw @ _rot_x(-cum))
    return joints


class _Operator:
    """Time-driven synthetic operator. ``scenario`` is one of :data:`SCENARIOS`."""

    def __init__(self, scenario: str = "demo", seed: int = 0):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
        self.scenario, self.seed = scenario, int(seed)
        rng = np.random.default_rng(self.seed)
        self._ph = rng.uniform(0, 2 * np.pi, (16, 3))
        self._fr = rng.uniform(0.6, 2.6, (16, 3))
        plan = _PLAN[scenario]
        self._plan = plan
        self._period = sum(d for _, d in plan)

    def _segment(self, t: float) -> tuple[str, str, float, float]:
        lt = t % self._period
        acc = 0.0
        for i, (name, dwell) in enumerate(self._plan):
            if lt < acc + dwell:
                prev = self._plan[i - 1][0]
                return prev, name, lt - acc, lt
            acc += dwell
        return self._plan[-1][0], self._plan[-1][0], 0.0, lt

    def _keyframe(self, t: float):
        prev, cur, ts, lt = self._segment(t)
        u = _min_jerk(ts / _TRANSITION_S) if self.scenario != "idle" else 1.0
        return _blend(_KF[prev], _KF[cur], u), cur, ts, lt

    def _tremor(self, ch: int, t: float, amp: float) -> np.ndarray:
        return amp * np.sin(2 * np.pi * self._fr[ch] * t + self._ph[ch])

    def _hand_pose(self, side: int, t: float) -> _HandPose:
        kf, cur, ts, _ = self._keyframe(t)
        i = 0 if side < 0 else 1
        pos = kf["pos"][i].copy()
        if cur == "wave" and side > 0:
            pos[0] += 0.10 * math.sin(2 * math.pi * 1.6 * ts) * _min_jerk(ts / 1.2)
        pos = pos + self._tremor(i, t, 0.002)
        head_yaw = self._head_yaw(t)
        r_hand = _rot_y(head_yaw) @ _rot_x(-kf["tilt"] * 0.4) @ _rot_z(self._tremor(2 + i, t, 0.02)[0])
        pos = np.array([0.0, 0.0, 0.0]) + _rot_y(head_yaw) @ pos
        curl = np.clip(kf["curl"][i] + self._tremor(4 + i, t, 0.01)[0], 0, 1)
        joints = _hand(side, pos, r_hand, curl, float(kf["spread"]))
        ctrl = _pose7(pos + r_hand @ np.array([0.0, -0.02, 0.05]), r_hand @ _rot_x(-0.6))
        return _HandPose(
            joints[1].copy(), ctrl, joints, curl, float(kf["spread"]), float(kf["trig"][i]),
            float(kf["grip"][i]),
        )  # fmt: skip

    def _head_yaw(self, t: float) -> float:
        return math.radians(7.0) * math.sin(2 * math.pi * 0.07 * t + self._ph[8, 0])

    @lru_cache(maxsize=4096)
    def _core(self, k: int):
        t = k * 1e-6
        yaw = self._head_yaw(t)
        pitch = math.radians(4.0) * math.sin(2 * math.pi * 0.11 * t + self._ph[9, 0])
        sway = self._tremor(10, t, 0.004)
        head_pos = np.array([sway[0], HEAD_HEIGHT_M + 0.01 * math.sin(2 * math.pi * 0.23 * t), sway[2]])
        head = _pose7(head_pos, _rot_y(yaw) @ _rot_x(pitch))
        return head, self._hand_pose(-1, t), self._hand_pose(+1, t)

    def core(self, t: float):
        return self._core(int(round(t * 1e6)))

    def inputs(self, t: float) -> dict:
        _, left, right = self.core(t)
        _, cur, ts, lt = self._keyframe(t)
        axis_l = axis_r = (0.0, 0.0)
        if self.scenario in ("demo", "record_cycle") and cur in ("reach", "point"):
            a = 2 * math.pi * 0.4 * t
            axis_l = (0.65 * math.sin(a), 0.65 * math.cos(a))
        if self.scenario == "wave" and cur == "wave":
            axis_r = (0.0, 0.5 * math.sin(2 * math.pi * 0.5 * ts))
        pressed = {k: False for k in ("A", "B", "X", "Y", "left_menu", "right_menu", "left_click", "right_click")}
        if self.scenario in ("record_cycle", "demo"):
            pressed["A"] = 1.0 <= lt % 12 < 1.3
            pressed["B"] = 9.0 <= lt % 12 < 9.3
            pressed["X"] = 5.0 <= lt % 12 < 5.2
            pressed["Y"] = 6.0 <= lt % 12 < 6.2
            pressed["left_menu"] = 11.0 <= lt % 12 < 11.2
        if self.scenario == "demo":
            pressed["right_click"] = cur == "point" and 1.0 <= ts < 1.2
        return {
            "left_trigger": left.trigger, "right_trigger": right.trigger,
            "left_grip": left.grip, "right_grip": right.grip,
            "left_axis": axis_l, "right_axis": axis_r, **pressed,
        }  # fmt: skip


# --------------------------------------------------------------------------- TeleopSession
class TeleopSession:
    """Approximates ``teleop_session_manager.teleop_session.TeleopSession``'s
    construction/context/``step()``/``get_status()`` shape. All scenario/
    fault state lives on this instance; faults are read live from
    ``self._faults`` each call (settable via :meth:`set_faults`) so an
    already-running session reflects a fault flipped on it mid-capture --
    an explicitly injected, instance-owned set, never process-global
    module state."""

    _FATAL_AFTER_STEPS = 30

    def __init__(
        self,
        config: TeleopSessionConfig,
        *,
        scenario: str = "demo",
        seed: int = 0,
        faults: set[str] | frozenset[str] | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.config = config
        self.frame_count = 0
        self._operator = _Operator(scenario=scenario, seed=seed)
        self._faults: set[str] = set(faults) if faults else set()
        self._clock = clock or MonotonicClock()
        self._entered = False
        self._enter_t0: float | None = None
        self._frozen_t: float | None = None
        self._worker_exited = False
        # Set during __enter__ only for an OPTIONAL Manus plugin that
        # could not be discovered/loaded -- a required one instead raises
        # and never reaches a fully-entered state at all. Matches the real
        # SDK's TeleopSession._resolve_plugin, which silently SKIPS an
        # undiscoverable optional plugin (no exception, no monitored
        # inventory entry for it at all): an optional skipped plugin means
        # absent inventory, not a reported failure.
        # Read by get_status() to omit Manus's provider/device entries
        # entirely, distinct from "provider_failed" below (a plugin that
        # DID launch and later failed, which genuinely is FAILED/
        # UNAVAILABLE -- a different scenario this flag must not touch).
        self._manus_undiscovered = False

    def set_faults(self, names: set[str] | frozenset[str] | None) -> None:
        """Replace this session's active fault set (``None``/empty means
        none). Live: takes effect on the next ``step()``/``get_status()``
        call, including on an already-entered session."""
        self._faults = set() if names is None else set(names)

    def active_faults(self) -> set[str]:
        return set(self._faults)

    def __enter__(self) -> "TeleopSession":
        if "startup_error" in self._faults:
            raise RuntimeError("mock startup_error fault: native entry failed")
        if "manus_plugin_load_failed" in self._faults:
            manus_plugin = next(
                (
                    plugin
                    for plugin in self.config.plugins
                    if plugin.plugin_name == MANUS_PLUGIN_NAME and plugin.enabled
                ),
                None,
            )
            if manus_plugin is not None:
                if manus_plugin.required:
                    raise RuntimeError("mock manus_plugin_load_failed fault: required Manus plugin failed to load")
                self._manus_undiscovered = True
        self._enter_t0 = self._clock.monotonic()
        self._entered = True
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self._entered = False
        return None

    def _elapsed(self) -> float:
        assert self._enter_t0 is not None, "TeleopSession.step() called before __enter__"
        now = self._clock.monotonic()
        if "sample_freeze" in self._faults:
            if self._frozen_t is None:
                self._frozen_t = now
            now = self._frozen_t
        else:
            self._frozen_t = None
        return now - self._enter_t0

    def step(
        self,
        *,
        external_inputs: dict[str, Any] | None = None,
        graph_time: Any | None = None,
        execution_events: Any | None = None,
    ) -> dict[str, Any]:
        """Produce the configured output mapping at call-time elapsed time.
        Synchronous and unpaced, matching the real API: no sleep, no
        background thread, no rate limiter -- the caller decides when to
        invoke it."""
        if not self._entered:
            raise RuntimeError("TeleopSession.step() called before __enter__")
        self.frame_count += 1
        if "fatal_acquisition_error" in self._faults and self.frame_count >= self._FATAL_AFTER_STEPS:
            # Real, typed exit evidence for the caller to check via
            # get_status() -- StatusReason.PROCESS_EXITED, below -- not
            # something inferred from this exception's own message text.
            self._worker_exited = True
            raise RuntimeError("mock fatal_acquisition_error fault: acquisition worker exited")

        t = self._elapsed()
        head_pose, left_hand, right_hand = self._operator.core(t)
        inputs = self._operator.inputs(t)

        result: dict[str, OptionalTensorGroup] = {}
        for leaf_key, (source_node, selector) in self.config.pipeline.outputs.items():
            if isinstance(source_node, HeadSource):
                result[leaf_key] = self._head_group(head_pose)
            elif isinstance(source_node, ControllersSource):
                side = "left" if selector == ControllersSource.LEFT else "right"
                if f"controller_{side}_absent" in self._faults:
                    # Registered/selected but absent THIS sample -- stays
                    # is_none=True, distinct from never being selected at
                    # all (which would simply not appear in this dict).
                    result[leaf_key] = OptionalTensorGroup(len(ControllerInputIndex))
                    continue
                hand = left_hand if side == "left" else right_hand
                valid = "controller_pose_invalid" not in self._faults
                result[leaf_key] = self._controller_group(hand, inputs, side, valid=valid)
            elif isinstance(source_node, HandsSource):
                side = "left" if selector == HandsSource.LEFT else "right"
                if f"hand_{side}_absent" in self._faults:
                    # Registered/selected but absent THIS sample -- stays
                    # is_none=True, same pattern as controller_*_absent.
                    # Independent of every other group: the other hand,
                    # head and controllers keep reporting normally.
                    result[leaf_key] = OptionalTensorGroup(len(HandInputIndex))
                    continue
                hand = left_hand if side == "left" else right_hand
                if "hand_pose_invalid" in self._faults:
                    joint_valid = np.zeros(26, dtype=np.uint8)
                elif "hand_pose_partially_invalid" in self._faults:
                    # Deterministic mixed mask: every fingertip joint
                    # (HandJointIndex.*_TIP) is untracked, everything else
                    # (palm/wrist/metacarpal/proximal/intermediate/distal)
                    # stays valid -- a present, partially-usable hand,
                    # never all-or-nothing.
                    joint_valid = np.array([0 if "TIP" in j.name else 1 for j in HandJointIndex], dtype=np.uint8)
                else:
                    joint_valid = np.ones(26, dtype=np.uint8)
                result[leaf_key] = self._hand_group(hand, joint_valid=joint_valid)
            else:
                raise ValueError(f"unsupported source node in pipeline.outputs: {source_node!r}")
        return result

    @staticmethod
    def _head_group(head_pose: np.ndarray) -> OptionalTensorGroup:
        group = OptionalTensorGroup(len(HeadInputIndex))
        group[HeadInputIndex.POSITION] = head_pose[:3].astype(np.float32)
        group[HeadInputIndex.ORIENTATION] = head_pose[3:].astype(np.float32)
        group[HeadInputIndex.IS_VALID] = True
        group[HeadInputIndex.IS_TRACKED] = True
        return group

    @staticmethod
    def _controller_group(
        hand: _HandPose, inputs: dict[str, Any], side: str, *, valid: bool = True
    ) -> OptionalTensorGroup:
        if side == "left":
            primary, secondary, menu, click = (
                inputs["X"],
                inputs["Y"],
                inputs["left_menu"],
                inputs["left_click"],
            )
            axis = inputs["left_axis"]
        else:
            primary, secondary, menu, click = (
                inputs["A"],
                inputs["B"],
                inputs["right_menu"],
                inputs["right_click"],
            )
            axis = inputs["right_axis"]
        grip, aim = hand.wrist, hand.controller
        group = OptionalTensorGroup(len(ControllerInputIndex))
        group[ControllerInputIndex.GRIP_POSITION] = grip[:3].astype(np.float32)
        group[ControllerInputIndex.GRIP_ORIENTATION] = grip[3:].astype(np.float32)
        group[ControllerInputIndex.GRIP_IS_VALID] = valid
        group[ControllerInputIndex.AIM_POSITION] = aim[:3].astype(np.float32)
        group[ControllerInputIndex.AIM_ORIENTATION] = aim[3:].astype(np.float32)
        group[ControllerInputIndex.AIM_IS_VALID] = valid
        group[ControllerInputIndex.PRIMARY_CLICK] = float(primary)
        group[ControllerInputIndex.SECONDARY_CLICK] = float(secondary)
        group[ControllerInputIndex.THUMBSTICK_X] = float(axis[0])
        group[ControllerInputIndex.THUMBSTICK_Y] = float(axis[1])
        group[ControllerInputIndex.THUMBSTICK_CLICK] = float(click)
        group[ControllerInputIndex.MENU_CLICK] = float(menu)
        group[ControllerInputIndex.SQUEEZE_VALUE] = float(hand.grip)
        group[ControllerInputIndex.TRIGGER_VALUE] = float(hand.trigger)
        return group  # fmt: skip

    @staticmethod
    def _hand_group(hand: _HandPose, *, joint_valid: np.ndarray) -> OptionalTensorGroup:
        group = OptionalTensorGroup(len(HandInputIndex))
        group[HandInputIndex.JOINT_POSITIONS] = hand.joints[:, :3].astype(np.float32)
        group[HandInputIndex.JOINT_ORIENTATIONS] = hand.joints[:, 3:].astype(np.float32)
        group[HandInputIndex.JOINT_RADII] = np.asarray(HAND_JOINT_RADII, dtype=np.float32)
        group[HandInputIndex.JOINT_VALID] = joint_valid
        return group

    def get_status(self) -> StatusSnapshot:
        """Return IC-shaped provider/device status. Camera status is not
        part of this SDK's status surface -- the production DepthAI
        adapter reports that independently; the shared mapper merges both
        into one protocol ``StatusSnapshot``."""
        active = self._faults
        openxr_state, openxr_reason = ProviderState.AVAILABLE, StatusReason.NONE
        manus_state, manus_reason = ProviderState.AVAILABLE, StatusReason.NONE
        left_state = right_state = DeviceState.CONNECTED
        left_reason = right_reason = StatusReason.NONE
        if "provider_failed" in active:
            manus_state, manus_reason = ProviderState.FAILED, StatusReason.STARTUP_FAILED
            left_state = right_state = DeviceState.UNAVAILABLE
            left_reason = right_reason = StatusReason.PROVIDER_FAILED
        if self._worker_exited:
            # Real, typed exit evidence -- IsaacCaptureAdapter checks this
            # (never this fault's step()-raised exception text) to decide
            # SourceExited vs a plain SourceError.
            openxr_state, openxr_reason = ProviderState.FAILED, StatusReason.PROCESS_EXITED
        providers = [
            ProviderStatus(
                id=_OPENXR_PROVIDER_ID, type=ProviderType.OPENXR_RUNTIME, name="OpenXR Runtime",
                status=openxr_state, reason=openxr_reason, error="",
            ),
        ]  # fmt: skip
        devices = [
            DeviceStatus(
                id=_HEADSET_DEVICE_ID, path="/head/headset", type="headset",
                description="Pico headset", provider_id=_OPENXR_PROVIDER_ID,
                status=DeviceState.CONNECTED, reason=StatusReason.NONE, error="",
            ),
        ]  # fmt: skip
        # Controller DEVICE inventory is a separate, more static concept
        # from per-sample presence/validity (controller_*_absent/
        # controller_pose_invalid, in step() above): it reflects whether
        # this side was registered/selected for the session at all, not
        # whether its pose happens to be usable on any given sample --
        # declaring the device here does not depend on either fault.
        # Owned by the OpenXR provider, like the headset (OpenXR is what
        # reports controller poses, unlike the plugin-provided Manus
        # hands below). Derived from the configured ControllersSource +
        # its LEFT/RIGHT selector directly -- NOT from the leaf label a
        # particular graph happens to key its OutputCombiner dict with
        # (an application is free to call that key anything; the
        # selector, not the label, is what the real SDK actually
        # selected). set() dedupes in case the same selector is somehow
        # wired to more than one leaf.
        selected_sides = {
            selector
            for source_node, selector in self.config.pipeline.outputs.values()
            if isinstance(source_node, ControllersSource)
        }
        if ControllersSource.LEFT in selected_sides:
            devices.append(
                DeviceStatus(
                    id=_LEFT_CONTROLLER_DEVICE_ID, path="/controller/left", type="controller",
                    description="Left controller", provider_id=_OPENXR_PROVIDER_ID,
                    status=DeviceState.CONNECTED, reason=StatusReason.NONE, error="",
                )
            )  # fmt: skip
        if ControllersSource.RIGHT in selected_sides:
            devices.append(
                DeviceStatus(
                    id=_RIGHT_CONTROLLER_DEVICE_ID, path="/controller/right", type="controller",
                    description="Right controller", provider_id=_OPENXR_PROVIDER_ID,
                    status=DeviceState.CONNECTED, reason=StatusReason.NONE, error="",
                )
            )  # fmt: skip
        # No Manus inventory unless the configured graph's plugin selection
        # actually requested/enabled it: no inventory entry unless the
        # native-style plugin config requested/launched it. Independent of
        # which tracking groups were
        # selected: a session can request Manus without selecting hand
        # output (it still shows up, degraded/failed/etc.) and selecting
        # HANDS does not request Manus by itself -- without a Manus
        # request, the server does not launch Manus merely because HANDS
        # was selected. A missing/unrequested plugin has no inventory
        # entry, never a synthesized disconnected one.
        # An optional Manus plugin this session's __enter__ could not
        # discover (manus_plugin_load_failed fault, required=False) gets
        # the SAME treatment -- real TeleopSession._resolve_plugin skips
        # it silently, so it is exactly as absent from inventory as an
        # unrequested one. Independent of generic hand tracking: hand_
        # left/hand_right sample presence (step()) is driven entirely by
        # the configured pipeline outputs, not by Manus provenance.
        manus_enabled = (
            any(plugin.plugin_name == MANUS_PLUGIN_NAME and plugin.enabled for plugin in self.config.plugins)
            and not self._manus_undiscovered
        )
        if manus_enabled:
            providers.append(
                ProviderStatus(
                    id=_MANUS_PROVIDER_ID, type=ProviderType.PLUGIN, name=MANUS_PLUGIN_NAME,
                    status=manus_state, reason=manus_reason, error="",
                )
            )  # fmt: skip
            devices.append(
                DeviceStatus(
                    id=_LEFT_HAND_DEVICE_ID, path="/hand/left", type="hand",
                    description="Left hand", provider_id=_MANUS_PROVIDER_ID,
                    status=left_state, reason=left_reason, error="",
                )
            )  # fmt: skip
            devices.append(
                DeviceStatus(
                    id=_RIGHT_HAND_DEVICE_ID, path="/hand/right", type="hand",
                    description="Right hand", provider_id=_MANUS_PROVIDER_ID,
                    status=right_state, reason=right_reason, error="",
                )
            )  # fmt: skip
        return StatusSnapshot(
            updated_at_ns=int(self._clock.monotonic() * 1e9),
            providers=tuple(providers),
            devices=tuple(devices),
        )
