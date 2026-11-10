# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Operator-adjustable station setup kept by the backend, and the operator tools it knows about.

Unlike the deployment settings (settings_routes.py), these are the operator's to change from the page:
camera type and roles, task catalog, and preselected task. They live in one JSON file under a hidden
``.egocap/`` directory in the recording output
(``EGOCAP_OPERATOR_SETTINGS_FILE`` overrides the location) so every browser sees the same value and it survives a
restart. A change needs the run's control token, the same credential that admits recording commands, so a stray
page cannot relabel a station.

The tools list is a registry only. Each entry says whether this build can launch it; none can yet, and the page
shows them disabled with that reason rather than hiding them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .recording_command import control_token_valid
from .task_files import is_catalog_name

logger = logging.getLogger("isaaccapture.egocap.operator_settings")

router = APIRouter()

_NO_STORE = {"Cache-Control": "no-store"}

OPERATOR_SETTINGS_FILE_ENV = "EGOCAP_OPERATOR_SETTINGS_FILE"

#: The camera types a station can be built with: GMSL cameras, or USB
#: OAK-D cameras read by CaptureServer. The value is a label for the operator and for tools; it does not itself
#: open any camera (CaptureServer and the station file own that).
CAMERA_TYPES = ("sensing_gmsl", "oak_d")

#: The camera roles a station fills, whatever the camera type: the operator's view and each wrist.
CAMERA_ROLES = ("ego", "wrist_left", "wrist_right")

_NOT_WIRED = "not wired in this build"

#: Operator tools the page offers. ``applies_to`` names the camera type a tool is for, or ``None`` for all.
TOOLS: tuple[dict[str, Any], ...] = (
    {
        "id": "headset_calibration",
        "name": "Headset calibration",
        "description": "Intrinsics and extrinsics of the headset cameras.",
        "applies_to": None,
        "available": False,
        "unavailable_reason": _NOT_WIRED,
    },
    {
        "id": "glove_calibration",
        "name": "Glove calibration",
        "description": "Hand-tracking glove calibration against the headset controllers.",
        "applies_to": None,
        "available": False,
        "unavailable_reason": _NOT_WIRED,
    },
    {
        "id": "fisheye_rectification",
        "name": "Fisheye rectification",
        "description": (
            "Rectification maps for the three SENSING GMSL cameras. Only fisheye lenses need this; "
            "rectilinear cameras do not."
        ),
        "applies_to": "sensing_gmsl",
        "available": False,
        "unavailable_reason": _NOT_WIRED,
    },
)


class OperatorSettingsError(ValueError):
    """The stored settings file exists but cannot be used (not JSON, not an object, or an unknown value)."""


@dataclass(frozen=True)
class OperatorSettings:
    """The operator's station setup; every field is optional so a fresh station starts with nothing asserted."""

    camera_type: str | None = None
    #: Which offered camera (a CaptureServer camera name) fills each role; unassigned is
    #: ``None``. Every role is always present so the page can show the gaps.
    camera_roles: dict[str, str | None] = field(default_factory=lambda: {role: None for role in CAMERA_ROLES})
    #: The catalog the recording panel's task picker offers, by file name; ``None`` means free text only.
    task_file: str | None = None
    #: The task (by its catalog name) the picker preselects when the current task is still unset.
    preselected_task: str | None = None

    def to_json(self) -> dict[str, Any]:
        """The wire form, with the accepted camera types and roles so the page needs no second copy of either."""
        return {
            "camera_type": self.camera_type,
            "camera_types": list(CAMERA_TYPES),
            "camera_roles": dict(self.camera_roles),
            "roles": list(CAMERA_ROLES),
            "task_file": self.task_file,
            "preselected_task": self.preselected_task,
        }

    def stored(self) -> dict[str, Any]:
        """The fields a PUT may change, as written to the file."""
        return {
            "camera_type": self.camera_type,
            "camera_roles": dict(self.camera_roles),
            "task_file": self.task_file,
            "preselected_task": self.preselected_task,
        }

    def updated(self, changes: dict[str, Any]) -> OperatorSettings:
        """These settings with the fields named in ``changes`` replaced; a PUT names only what it changes. A
        ``camera_roles`` change may itself name only some roles; the rest keep their assignment."""
        merged = self.stored()
        merged.update({key: changes[key] for key in merged if key in changes and key != "camera_roles"})
        # A catalog change must not preserve a task from its predecessor.
        if merged["task_file"] != self.task_file and "preselected_task" not in changes:
            merged["preselected_task"] = None
        if "camera_roles" in changes:
            roles = changes["camera_roles"]
            if not isinstance(roles, dict):
                raise OperatorSettingsError("camera_roles must be an object of role to camera name or null")
            merged["camera_roles"] = {**self.camera_roles, **roles}
        return _parse(merged)


def _parse_roles(raw: Any) -> dict[str, str | None]:
    """Role assignments: known roles only, each a non-empty camera name or null, no camera in two roles."""
    if raw is None:
        return {role: None for role in CAMERA_ROLES}
    if not isinstance(raw, dict):
        raise OperatorSettingsError("camera_roles must be an object of role to camera name or null")
    unknown = sorted(set(raw) - set(CAMERA_ROLES))
    if unknown:
        raise OperatorSettingsError(
            f"unknown camera role(s): {', '.join(unknown)}; roles are {', '.join(CAMERA_ROLES)}"
        )
    roles: dict[str, str | None] = {}
    for role in CAMERA_ROLES:
        value = raw.get(role)
        if value is not None and (not isinstance(value, str) or value.strip() == ""):
            raise OperatorSettingsError(f"camera_roles.{role} must be a camera name or null")
        roles[role] = value
    assigned = [value for value in roles.values() if value is not None]
    if len(assigned) != len(set(assigned)):
        raise OperatorSettingsError("a camera cannot fill two roles")
    return roles


def _optional_text(raw: dict[str, Any], key: str, *, check=None, what: str = "") -> str | None:
    """Validate a nullable text field using its optional domain predicate."""
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or (check is not None and not check(value)):
        raise OperatorSettingsError(f"{key} must be {what or 'a string'} or null")
    return value


def _parse(raw: Any) -> OperatorSettings:
    """A stored or submitted object as settings; an unknown value is an error, an unknown key is ignored."""
    if not isinstance(raw, dict):
        raise OperatorSettingsError("operator settings must be an object")
    camera_type = raw.get("camera_type")
    if camera_type is not None and camera_type not in CAMERA_TYPES:
        raise OperatorSettingsError(f"camera_type must be one of {', '.join(CAMERA_TYPES)} or null")
    return OperatorSettings(
        camera_type=camera_type,
        camera_roles=_parse_roles(raw.get("camera_roles")),
        task_file=_optional_text(raw, "task_file", check=is_catalog_name, what="a plain *.csv file name"),
        preselected_task=_optional_text(
            raw, "preselected_task", check=lambda v: v.strip() != "", what="a task name"
        ),
    )


class OperatorSettingsStore:
    """One JSON file holding the operator settings; absent means defaults, and writes replace it atomically."""

    def __init__(self, path: Path) -> None:
        """Bind storage and its mutation lock for this backend process."""
        self.path = path
        self.lock = threading.RLock()

    def load(self) -> OperatorSettings:
        """The stored settings, or the defaults when nothing has been saved yet."""
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return OperatorSettings()
        except UnicodeError as exc:
            raise OperatorSettingsError("operator settings are not valid UTF-8") from exc
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise OperatorSettingsError(f"{self.path} is not valid JSON: {exc}") from exc
        return _parse(raw)

    def save(self, settings: OperatorSettings) -> None:
        """Write the settings, through a temporary file so a crash never leaves a half-written one."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Unique files prevent overlapping writers from sharing or replacing each other's staging file.
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.path.parent, prefix=f".{self.path.name}.", delete=False
        ) as handle:
            tmp = Path(handle.name)
            try:
                handle.write(json.dumps(settings.stored(), indent=2) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()
                os.replace(tmp, self.path)
            finally:
                tmp.unlink(missing_ok=True)

    def update(self, changes: dict[str, Any], root: Path | None, imported: Path | None) -> OperatorSettings:
        """Validate and persist a partial change under the same lock as catalog removal."""
        from .task_files import TaskFileError, read_available_task_file

        # The whole transaction stays in one thread so cancellation cannot release its lock early.
        with self.lock:
            try:
                current = self.load()
            except (OperatorSettingsError, UnicodeError):
                logger.warning("Replacing invalid operator settings at %s", self.path)
                current = OperatorSettings()
            settings = current.updated(changes)
            # Validate task choices against the same imported-first catalog that the picker reads.
            if settings.task_file is not None and root is not None:
                try:
                    catalog = read_available_task_file(root, imported, settings.task_file)
                except TaskFileError as exc:
                    raise OperatorSettingsError(str(exc)) from exc
                if settings.preselected_task is not None and not any(
                    task.name == settings.preselected_task for task in catalog.tasks
                ):
                    raise OperatorSettingsError("preselected_task must name a task in the selected catalog")
            elif settings.task_file is None and settings.preselected_task is not None:
                raise OperatorSettingsError("preselected_task needs a selected catalog")
            self.save(settings)
            return settings


def operator_settings_path(recording_output_root: Path) -> Path:
    """Resolve the override or a hidden config directory excluded from the recordings browser."""
    override = os.environ.get(OPERATOR_SETTINGS_FILE_ENV)
    return (
        Path(override).expanduser() if override else recording_output_root / ".egocap" / "operator_settings.json"
    )


def _store(request: Request) -> OperatorSettingsStore | None:
    return getattr(request.app.state, "operator_settings_store", None)


@router.get("/api/operator-settings")
async def read_operator_settings(request: Request) -> JSONResponse:
    """Return saved settings, or defaults with a repair warning if the saved data is invalid."""
    store = _store(request)
    if store is None:
        return JSONResponse(
            {"detail": "operator settings are not available yet"}, status_code=503, headers=_NO_STORE
        )
    try:
        settings = await asyncio.to_thread(store.load)
    except OperatorSettingsError:
        # Defaults keep the form usable so an authenticated save can repair invalid storage.
        logger.warning("Invalid operator settings at %s; showing defaults for repair", store.path)
        return JSONResponse(
            {**OperatorSettings().to_json(), "warning": "Saved settings are invalid. Save to replace them."},
            headers=_NO_STORE,
        )
    return JSONResponse(settings.to_json(), headers=_NO_STORE)


@router.put("/api/operator-settings")
async def write_operator_settings(request: Request) -> JSONResponse:
    """Change the operator settings. The body carries ``control_token`` and the fields to change (the rest keep
    their stored value); a wrong token is 403 before the values are even looked at, and an invalid value is 400
    with the reason."""
    store = _store(request)
    recording_context = getattr(request.app.state, "recording_context", None)
    if store is None or recording_context is None:
        return JSONResponse(
            {"detail": "operator settings are not available yet"}, status_code=503, headers=_NO_STORE
        )
    try:
        body = await request.json()
    except ValueError:
        return JSONResponse({"detail": "body must be JSON"}, status_code=400, headers=_NO_STORE)
    if not isinstance(body, dict):
        return JSONResponse({"detail": "body must be an object"}, status_code=400, headers=_NO_STORE)
    if not control_token_valid(body.get("control_token"), recording_context.control_token):
        return JSONResponse({"detail": "control token rejected"}, status_code=403, headers=_NO_STORE)
    try:
        settings = await asyncio.to_thread(
            store.update,
            body,
            getattr(request.app.state, "task_file_root", None),
            getattr(request.app.state, "imported_task_file_root", None),
        )
    except OperatorSettingsError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=400, headers=_NO_STORE)
    except OSError as exc:
        return JSONResponse(
            {"detail": f"could not save operator settings: {exc}"}, status_code=503, headers=_NO_STORE
        )
    return JSONResponse(settings.to_json(), headers=_NO_STORE)


@router.get("/api/tools")
async def list_tools() -> JSONResponse:
    """The operator tools this build knows about, each with whether it can be launched."""
    return JSONResponse({"tools": list(TOOLS)}, headers=_NO_STORE)
