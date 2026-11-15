# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent program's settings (configs/agent.yaml) and APAI's configuration.

load_settings() reads and checks the file. build_apai_config() writes an arm
profile and a camera profile into the run folder and loads them with APAI's
own loader; the arm's type is robot_link, the obstacle grid is required and
the joint speed is the task's. APAI and the experiment are imported inside
functions, so the module imports without them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .arm_adapter import ARM_TYPE, ArmSettings
from .self_mask import EXPERIMENT_DIR_ENV

STACK_MINIMAL = "minimal"
STACK_FULL = "full"
STACKS = (STACK_MINIMAL, STACK_FULL)
LOG_LEVELS = ("debug", "info", "warning", "error")
# The camera a run uses: the camera connection, APAI's mock camera, or a
# folder of recorded stereo pairs ("recorded:<folder>").
CAMERA_LINK = "link"
CAMERA_MOCK = "mock"
CAMERA_RECORDED = "recorded:"
# The profile type of camera/stereo_camera.py.
LINK_STEREO = "link_stereo"
# Set in the environment, each of these changes what APAI builds behind the
# launcher's back; the program refuses to start with one of them.
REFUSED_ENV = (
    "APAI_ROBOT",
    "APAI_BASE",
    "APAI_START_STOPPED",
    "APAI_OBSERVED_FINGER_GATE",
    "APAI_OCCUPANCY",
)
# Where APAI writes a store: the environment variable and the name in the
# agent program's store folder. (Its learned-notes library is only read.)
APAI_STORES = (
    ("APAI_BELIEFS_PATH", "beliefs.json"),
    ("APAI_ARM_HOLD_DIR", "arm_holds"),
    ("APAI_REST_LOG", "rest_readings.jsonl"),
    ("APAI_EPISODIC_PATH", "episodic.json"),
    ("APAI_GRASP_MEMORY_PATH", "grasp_memory.json"),
    ("APAI_ENVELOPE_PATH", "envelope.json"),
    ("APAI_PROGRAMS_PATH", "programs.jsonl"),
)


class ConfigError(ValueError):
    """agent.yaml, the environment or APAI's configuration cannot be used."""


@dataclass(frozen=True)
class LinkSettings:
    socket_path: str = ""  # "" is the robot link service's default path
    connect_timeout_s: float = 10.0
    state_max_age_s: float = 0.5
    expected_profile_id: str = ""  # "" accepts any profile id
    # "" names the calibration by the hashes of the camera block's two files.
    calibration_id: str = ""
    allow_unnamed_calibration: bool = False


@dataclass(frozen=True)
class ThorSettings:
    """Copies of the Thor's settings that are not in the WELCOME."""

    first_wait_s: float = 0.5
    settle_timeout_s: float = 2.0
    agent_max_joint_speed: float = 0.4


@dataclass(frozen=True)
class MoveSettings:
    start_tol_rad: float = 0.02
    fresh_wait_s: float = 0.6
    start_grid_age_s: float = 1.0
    start_grid_wait_s: float = 3.0
    expires_in_s: float = 2.0
    cancel_wait_s: float = 2.0


@dataclass(frozen=True)
class PermitSettings:
    reach_setpoints: int = 25
    renew_s: float = 0.05
    life_s: float = 1.0


@dataclass(frozen=True)
class JawSettings:
    fresh_s: float = 1.0
    result_timeout_s: float = 12.0


@dataclass(frozen=True)
class TaskSettings:
    timed_return: bool = False
    failed_steps_limit: int = 3
    max_joint_speed: float = 0.1
    # How long the end of a task waits for the RESULT of its owner request.
    end_wait_s: float = 5.0
    # How long a start waits for the ROBOT_STATE that shows the agent as owner.
    owner_wait_s: float = 2.0
    # How long the start of the program waits for a usable obstacle grid.
    grid_wait_s: float = 20.0


@dataclass(frozen=True)
class CameraSettings:
    bridge_python: str = ""
    camera_pr: str = ""  # "" takes AGENTIC_REMOTE_CAMERA_PR
    receiver_socket: str = "/tmp/camera_receiver_avs1_cuda_ipc.sock"
    streams: tuple[str, str] = ("head.left", "head.right")
    calibration_dir: str = ""
    hand_eye_json: str = ""
    max_skew_ms: float = 5.0
    depth_backend: str = "ffs"
    fps: float = 7.0


@dataclass(frozen=True)
class ExperimentSettings:
    stack: str = STACK_FULL
    # APAI's own profiles the minimal stack builds on.
    arm_profile: str = "mock"
    llm: str = "mock"


@dataclass(frozen=True)
class RigSettings:
    """Values of the rig's arm profile on the Thor, copied (see agent.yaml)."""

    home_q: tuple[float, ...] | None = None
    park_q: tuple[float, ...] | None = None
    body_mask_dilate_px: int | None = None
    motion_bounds: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class AgentSettings:
    path: Path
    state_dir: Path
    log_level: str = "info"
    link: LinkSettings = field(default_factory=LinkSettings)
    thor: ThorSettings = field(default_factory=ThorSettings)
    move: MoveSettings = field(default_factory=MoveSettings)
    permits: PermitSettings = field(default_factory=PermitSettings)
    jaws: JawSettings = field(default_factory=JawSettings)
    task: TaskSettings = field(default_factory=TaskSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)
    experiment: ExperimentSettings = field(default_factory=ExperimentSettings)
    rig: RigSettings = field(default_factory=RigSettings)

    @property
    def minimal(self) -> bool:
        return self.experiment.stack == STACK_MINIMAL

    def as_dict(self) -> dict[str, Any]:
        """The resolved values, as plain data."""
        out: dict[str, Any] = {
            "path": str(self.path),
            "state_dir": str(self.state_dir),
            "log_level": self.log_level,
        }
        for name in _SECTIONS:
            block = getattr(self, name)
            out[name] = {
                item.name: _plain(getattr(block, item.name)) for item in fields(block)
            }
        return out


_SECTIONS: dict[str, type] = {
    "link": LinkSettings,
    "thor": ThorSettings,
    "move": MoveSettings,
    "permits": PermitSettings,
    "jaws": JawSettings,
    "task": TaskSettings,
    "camera": CameraSettings,
    "experiment": ExperimentSettings,
    "rig": RigSettings,
}
_TOP = ("state_dir", "log_level", *_SECTIONS)
# Keys whose value is a path; a relative one is taken from the file's folder.
_PATHS = {
    "camera": ("bridge_python", "camera_pr", "calibration_dir", "hand_eye_json"),
    "link": ("socket_path",),
}
# May be 0.
_MAY_BE_ZERO = {"max_skew_ms"}


def load_settings(path: str | os.PathLike[str]) -> AgentSettings:
    """Read and check agent.yaml; ConfigError says what is wrong."""
    file = Path(path).expanduser()
    try:
        data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as problem:
        raise ConfigError(f"cannot read {file}: {problem}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{file} must hold a mapping")
    unknown = sorted(set(data) - set(_TOP))
    if unknown:
        raise ConfigError(f"{file}: unknown key(s) {', '.join(unknown)}")
    folder = file.resolve().parent
    blocks: dict[str, Any] = {}
    for name, kind in _SECTIONS.items():
        blocks[name] = _block(name, kind, data.get(name) or {}, folder)
    level = str(data.get("log_level", "info")).lower()
    if level not in LOG_LEVELS:
        raise ConfigError(f"log_level must be one of {', '.join(LOG_LEVELS)}")
    state_dir = _path(str(data.get("state_dir") or "../.state"), folder)
    settings = AgentSettings(
        path=file, state_dir=Path(state_dir), log_level=level, **blocks
    )
    _check(settings)
    return settings


def _block(name: str, kind: type, given: Any, folder: Path) -> Any:
    if not isinstance(given, dict):
        raise ConfigError(f"{name} must be a mapping")
    known = {item.name: item for item in fields(kind)}
    unknown = sorted(set(given) - set(known))
    if unknown:
        raise ConfigError(f"{name}: unknown key(s) {', '.join(unknown)}")
    values: dict[str, Any] = {}
    for key, value in given.items():
        default = known[key].default
        where = f"{name}.{key}"
        if key in ("home_q", "park_q"):
            value = None if value is None else _numbers(where, value)
        elif key == "motion_bounds":
            if value is not None and not isinstance(value, dict):
                raise ConfigError(f"{where} must be a mapping")
        elif key == "streams":
            if not _is_names(value, 2):
                raise ConfigError(f"{where} must be two stream names")
            value = tuple(value)
        elif isinstance(default, bool):
            if not isinstance(value, bool):
                raise ConfigError(f"{where} must be true or false")
        elif isinstance(default, int) or key == "body_mask_dilate_px":
            if value is not None or key != "body_mask_dilate_px":
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ConfigError(f"{where} must be a whole number above 0")
        elif isinstance(default, float):
            low_ok = key in _MAY_BE_ZERO
            if not _is_number(value) or value < 0 or (value == 0 and not low_ok):
                raise ConfigError(f"{where} must be a finite number above 0")
            value = float(value)
        elif isinstance(default, str):
            value = "" if value is None else value
            if not isinstance(value, str):
                raise ConfigError(f"{where} must be text")
            if value and key in _PATHS.get(name, ()):
                value = _path(value, folder)
        values[key] = value
    return kind(**values)


def _check(settings: AgentSettings) -> None:
    """The rules that span keys."""
    if settings.experiment.stack not in STACKS:
        raise ConfigError(f"experiment.stack must be one of {', '.join(STACKS)}")
    speed, limit = settings.task.max_joint_speed, settings.thor.agent_max_joint_speed
    if speed > limit:
        raise ConfigError(
            f"task.max_joint_speed is {speed:g} rad/s, above the Thor's agent "
            f"limit thor.agent_max_joint_speed of {limit:g} rad/s"
        )
    home, park = settings.rig.home_q, settings.rig.park_q
    if home is not None and park is not None and len(home) != len(park):
        raise ConfigError("rig.home_q and rig.park_q differ in length")


def arm_settings(settings: AgentSettings, calibration_id: str = "") -> ArmSettings:
    """What the arm adapter takes of the settings."""
    link, thor, move = settings.link, settings.thor, settings.move
    permits, jaws = settings.permits, settings.jaws
    return ArmSettings(
        expected_profile_id=link.expected_profile_id,
        calibration_id=calibration_id,
        allow_unnamed_calibration=link.allow_unnamed_calibration,
        connect_timeout_s=link.connect_timeout_s,
        state_max_age_s=link.state_max_age_s,
        first_wait_s=thor.first_wait_s,
        settle_timeout_s=thor.settle_timeout_s,
        agent_max_joint_speed=thor.agent_max_joint_speed,
        start_tol_rad=move.start_tol_rad,
        fresh_wait_s=move.fresh_wait_s,
        start_grid_age_s=move.start_grid_age_s,
        start_grid_wait_s=move.start_grid_wait_s,
        expires_in_s=move.expires_in_s,
        cancel_wait_s=move.cancel_wait_s,
        reach_setpoints=permits.reach_setpoints,
        renew_s=permits.renew_s,
        life_s=permits.life_s,
        fresh_s=jaws.fresh_s,
        result_timeout_s=jaws.result_timeout_s,
    )


def refused_environment(environ: Mapping[str, str] | None = None) -> None:
    """ConfigError when the environment would change what APAI builds."""
    env = os.environ if environ is None else environ
    found = [name for name in REFUSED_ENV if str(env.get(name, "")).strip()]
    if found:
        raise ConfigError(
            f"{', '.join(found)} is set in the environment: APAI would build "
            "another robot, start stopped or change its obstacle check; unset "
            "it and start again"
        )


def store_folder(settings: AgentSettings) -> Path:
    """Where APAI's stores of this example live (not in APAI's checkout)."""
    return settings.state_dir / "apai"


def redirect_apai_stores(
    settings: AgentSettings, environ: dict[str, str] | None = None
) -> Path:
    """Point every APAI store that is not set yet into the store folder."""
    env = os.environ if environ is None else environ
    folder = store_folder(settings)
    folder.mkdir(parents=True, exist_ok=True)
    for name, leaf in APAI_STORES:
        env.setdefault(name, str(folder / leaf))
    return folder


def camera_kind(camera: str) -> tuple[str, str]:
    """("link" | "mock" | "recorded", the recorded folder or "")."""
    if camera in (CAMERA_LINK, CAMERA_MOCK):
        return camera, ""
    if camera.startswith(CAMERA_RECORDED) and camera[len(CAMERA_RECORDED) :]:
        return "recorded", camera[len(CAMERA_RECORDED) :]
    raise ConfigError(
        f"--camera must be link, mock or recorded:<folder>, not {camera!r}"
    )


def calibration_id(settings: AgentSettings, camera: str = CAMERA_LINK) -> str:
    """The name of the camera calibration this program uses.

    link.calibration_id when it is set; else, for a stereo camera, the hashes
    of the calibration manifest and the hand-eye file; "" for the mock camera.
    """
    if settings.link.calibration_id:
        return settings.link.calibration_id
    if camera_kind(camera)[0] == CAMERA_MOCK:
        return ""
    block = settings.camera
    if not block.calibration_dir or not block.hand_eye_json:
        raise ConfigError(
            "camera.calibration_dir and camera.hand_eye_json are needed for the "
            "stereo camera"
        )
    from .camera import stereo_camera

    try:
        return stereo_camera.calibration_id(block.calibration_dir, block.hand_eye_json)
    except Exception as problem:  # CameraError: APAI's class or a RuntimeError
        raise ConfigError(str(problem)) from None


def experiment_dir() -> Path:
    """The experiment folder; ConfigError when the variable does not name one."""
    folder = os.environ.get(EXPERIMENT_DIR_ENV, "")
    if not folder or not Path(folder).is_dir():
        raise ConfigError(
            f"experiment.stack is full: {EXPERIMENT_DIR_ENV} must name the "
            f"experiment folder (it is {folder!r})"
        )
    return Path(folder).resolve()


def build_apai_config(
    settings: AgentSettings,
    run_dir: str | os.PathLike[str],
    *,
    camera: str = CAMERA_LINK,
    environ: Mapping[str, str] | None = None,
) -> Any:
    """APAI's configuration (a Cfg) for this run; ConfigError when refused.

    Writes arm_profile.yaml and, for a stereo camera, camera_profile.yaml into
    the run folder. Nothing is connected or started.
    """
    refused_environment(environ)
    kind, recorded = camera_kind(camera)
    try:
        from apai.config import Cfg, load_demo_config
    except ImportError as missing:
        raise ConfigError(f"APAI cannot be imported: {missing}") from None
    folder = Path(run_dir)
    folder.mkdir(parents=True, exist_ok=True)
    arm_file = folder / "arm_profile.yaml"
    arm_file.write_text(yaml.safe_dump(_arm_profile(settings), sort_keys=False))
    camera_name = CAMERA_MOCK
    if kind != CAMERA_MOCK:
        camera_file = folder / "camera_profile.yaml"
        profile = _camera_profile(settings, folder, recorded)
        camera_file.write_text(yaml.safe_dump(profile, sort_keys=False))
        camera_name = str(camera_file.with_suffix(""))
    llm = settings.experiment.llm if settings.minimal else "codex_astra"
    try:
        data = load_demo_config(
            camera=camera_name, arm=str(arm_file.with_suffix("")), llm=llm
        ).as_dict()
        views = [data, *(arm["resolved"] for arm in data["arms"] if "resolved" in arm)]
        if not settings.minimal:
            _merge_experiment(data, views, kind)
        for view in views:
            if settings.minimal:
                view["occupancy"]["enabled"] = True
                view["occupancy"]["required"] = True
            view["safety"]["max_joint_vel"] = settings.task.max_joint_speed
            view["memory"]["experience_path"] = str(
                store_folder(settings) / "experience.json"
            )
        _refuse(data, views, settings)
    except ConfigError:
        raise
    except Exception as problem:
        raise ConfigError(
            f"APAI's configuration cannot be built: {type(problem).__name__}: {problem}"
        ) from problem
    return Cfg(data)


def _arm_profile(settings: AgentSettings) -> dict[str, Any]:
    rig = settings.rig
    if settings.minimal:
        parent = settings.experiment.arm_profile
    else:
        parent = str(experiment_dir() / "configs" / "rebot_workstation")
    profile: dict[str, Any] = {"extends": parent, "type": ARM_TYPE}
    for name in ("home_q", "park_q"):
        if getattr(rig, name) is not None:
            profile[name] = list(getattr(rig, name))
    if rig.body_mask_dilate_px is not None:
        profile["body_mask"] = {"dilate_px": rig.body_mask_dilate_px}
    if rig.motion_bounds is not None:
        profile["motion_bounds"] = _plain(rig.motion_bounds)
    return profile


def _camera_profile(
    settings: AgentSettings, run_dir: Path, recorded: str
) -> dict[str, Any]:
    block = settings.camera
    if not block.calibration_dir or not block.hand_eye_json:
        raise ConfigError(
            "camera.calibration_dir and camera.hand_eye_json are needed for the "
            "stereo camera"
        )
    profile: dict[str, Any] = {
        "type": LINK_STEREO,
        "name": "head",
        "width": 1280,
        "height": 992,
        "fps": block.fps,
        "calibration_dir": block.calibration_dir,
        "extrinsics": {"hand_eye_json": block.hand_eye_json},
        "depth_backend": block.depth_backend,
        "frame_timeout_s": 10,
        "max_frame_age_s": 3,
        "map_depth": True,
        "fuse_beliefs": False,
    }
    if recorded:
        profile["recorded_dir"] = str(Path(recorded).expanduser().resolve())
        return profile
    if not block.bridge_python:
        raise ConfigError(
            "camera.bridge_python (the camera consumer's interpreter) is needed "
            "for the camera connection"
        )
    bridge: dict[str, Any] = {
        "python": block.bridge_python,
        "receiver_socket": block.receiver_socket,
        "run_dir": str(run_dir),
        "streams": list(block.streams),
        "max_skew_ms": block.max_skew_ms,
    }
    if block.camera_pr:
        bridge["camera_pr"] = block.camera_pr
    profile["bridge"] = bridge
    return profile


def _merge_experiment(
    data: dict[str, Any], views: list[dict[str, Any]], camera: str
) -> None:
    """The experiment's own files over APAI's, read from its folder."""
    from apai.config import _deep_merge, _resolve_paths, load_profile

    root = experiment_dir()
    configs = root / "configs"
    runtime = yaml.safe_load((configs / "thor_runtime.yaml").read_text())
    cloud = yaml.safe_load((configs / "thor_cloud.yaml").read_text())
    cloud["llm"]["codex_bin"] = str(root / cloud["llm"]["codex_bin"])
    overlay = _resolve_paths(runtime["overrides"], root)
    for view in views:
        _deep_merge(view, overlay)
        view["llm"] = cloud["llm"]
        view["memory"].update(persist_beliefs=False, persist_episodic=False)
    depth = load_profile("cameras", str(configs / "sensing_thor")).as_dict()
    for view in views:
        for profile in [view.get("camera", {}), *view.get("cameras", [])]:
            if profile.get("type") != LINK_STEREO:
                continue
            if camera == CAMERA_LINK:
                profile["depth_backend"] = depth["depth_backend"]
            profile["ffs"] = depth.get("ffs", {})
    for profile in data["cameras"]:
        profile["map_depth"] = True
        profile["fuse_beliefs"] = False
        profile["detector"] = data["detector"]
    data["camera"] = data["cameras"][0]


def _refuse(
    data: dict[str, Any], views: list[dict[str, Any]], settings: AgentSettings
) -> None:
    """What the first run does not start with."""
    from apai.config import Cfg
    from apai.perception.grounding import Extrinsics

    if data.get("robot_mode") or len(data["arms"]) != 1:
        raise ConfigError("the agent program drives one arm and no base")
    if data["arm"].get("type") != ARM_TYPE:
        raise ConfigError(f"the arm profile's type must be {ARM_TYPE}")
    limit = settings.thor.agent_max_joint_speed
    for view in views:
        if not view["occupancy"].get("enabled") or not view["occupancy"]["required"]:
            raise ConfigError("the obstacle grid must be enabled and required")
        speed = view["safety"]["max_joint_vel"]
        if not _is_number(speed) or not 0 < speed <= limit:
            raise ConfigError(
                f"the joint speed must be a finite number above 0 and at most "
                f"{limit:g} rad/s, not {speed!r}"
            )
    extrinsics = Extrinsics.from_config(Cfg(data["camera"]["extrinsics"]))
    if not extrinsics.calibrated:
        raise ConfigError(
            "the camera mounting is not calibrated: "
            f"{getattr(extrinsics, 'calibration_error', 'no transform')}"
        )


def summary(settings: AgentSettings, cfg: Any, camera: str) -> dict[str, Any]:
    """What --check prints: the resolved values and the calibration id."""
    return {
        "configuration": "valid",
        "settings": settings.as_dict(),
        "calibration_id": calibration_id(settings, camera),
        "camera": camera,
        "arm_type": cfg.arm.type,
        "camera_type": cfg.camera.type,
        "llm_type": cfg.llm.get("type"),
        "max_joint_vel_rad_s": cfg.safety.max_joint_vel,
        "obstacle_grid": {
            "required": cfg.occupancy.required,
            "max_age_s": cfg.occupancy.get("max_age_s"),
        },
        "motion_bounds": _plain(cfg.arm.get("motion_bounds")),
    }


def file_hashes(folder: Path, modules: Mapping[str, Any]) -> dict[str, str]:
    """SHA-256 of every imported module file that lies under the folder."""
    root = folder.resolve()
    found: dict[str, str] = {}
    for module in list(modules.values()):
        file = getattr(module, "__file__", None)
        if not file:
            continue
        path = Path(file).resolve()
        if root in path.parents:
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                digest = "unreadable"
            found[str(path.relative_to(root))] = digest
    return dict(sorted(found.items()))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def _path(text: str, folder: Path) -> str:
    path = Path(os.path.expandvars(text)).expanduser()
    return str(path if path.is_absolute() else (folder / path).resolve())


def _plain(value: Any) -> Any:
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def _numbers(where: str, value: Any) -> tuple[float, ...]:
    if not isinstance(value, list) or not value or not all(map(_is_number, value)):
        raise ConfigError(f"{where} must be a list of finite numbers")
    return tuple(float(item) for item in value)


def _is_names(value: Any, count: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == count
        and all(isinstance(item, str) and item for item in value)
    )
