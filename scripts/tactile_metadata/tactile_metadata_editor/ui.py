# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Viser forms and an ordered, local editor event boundary."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import logging
import math
import queue
import secrets
import threading
import time
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from .domain import (
    Candidate, Device, Document, FrozenMap, Issue, ModelConfig, Segment, Skinning,
    Snapshot, Taxel, thaw,
)
from .state import METADATA_GROUPS, State, replace_device, _index as _row_index

LOGGER = logging.getLogger("isaaccapture.tools.tactile_metadata.ui")
PAGE_SIZE = 25
VISER_VERSION = "1.1.1"


def _json(value):
    return json.dumps(thaw(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _parse_json(value):
    def reject(text):
        raise ValueError(f"Non-finite JSON number: {text}")

    parsed = json.loads(value, parse_constant=reject)
    _json(parsed)
    return parsed


def _number(value):
    result = _parse_json(value) if isinstance(value, str) else value
    if result is not None and (isinstance(result, bool) or not isinstance(result, (int, float))):
        raise ValueError("Enter a JSON number or null")
    return result


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _weight_sum(row):
    if not isinstance(row, tuple) or len(row) != 4 or not all(_finite(value) for value in row):
        return None
    try:
        total = sum(row)
    except OverflowError:
        return None
    return total if _finite(total) else None


def _weight_display(value):
    # Validate before multiplying: invalid raw weights must remain editable.
    if not _finite(value) or not 0 <= value <= 1:
        return None
    return (int(255 * value), 70, int(255 * (1 - value))), max(30, int(255 * value))


def _force_magnitude(force):
    if not isinstance(force, (tuple, list)) or len(force) != 3 or not all(_finite(value) for value in force):
        return None
    try:
        value = math.hypot(*force)
    except OverflowError:
        return None
    return value if _finite(value) else None


def _source_values(device):
    from .document import raw_values

    try:
        return raw_values(device, output_order=True)
    except ValueError:
        return raw_values(device)


def _hand_geometry_validity(hand):
    from .compiler import FLOAT32_MAX, float32, quaternion_norm_error
    from .profiles.hand import JOINT_NAMES, JOINT_VALUES

    unavailable = dict.fromkeys(JOINT_NAMES, False)
    if not isinstance(hand, FrozenMap) or hand.get("valid") is not True:
        return unavailable
    positions, orientations, flags = (hand.get(name) for name in ("positions", "orientations_xyzw", "is_valid"))
    if any(not isinstance(rows, tuple) or len(rows) != len(JOINT_NAMES) for rows in (positions, orientations, flags)):
        return unavailable
    for name, index in JOINT_VALUES.items():
        position, orientation = positions[index], orientations[index]
        if (flags[index] is not True or not isinstance(position, tuple) or len(position) != 3
                or not isinstance(orientation, tuple) or len(orientation) != 4):
            continue
        pose = (*position, *orientation)
        if not all(_finite(value) and abs(value) <= FLOAT32_MAX for value in pose):
            continue
        target = tuple(float32(value) for value in pose)
        unavailable[name] = quaternion_norm_error(target) <= float32(1e-4)
    return {name: valid and unavailable["WRIST"] for name, valid in unavailable.items()}


def _taxel_hand_dependencies(device, taxel):
    from .mappings.lbs import validate_slots

    row = _row_index(taxel.taxel_index, min(len(device.skinning.indices), len(device.skinning.weights)))
    if row is None:
        return None
    indices, weights = device.skinning.indices[row], device.skinning.weights[row]
    if validate_slots(indices, weights, len(device.segments)):
        return None
    names = tuple(device.segments[int(index)].semantic for index, weight in zip(indices, weights) if weight > 0)
    return tuple(dict.fromkeys(("WRIST", *names))) if all(isinstance(name, str) for name in names) else None


def _retain_live_points(identities, points, validity, previous):
    identities_out, points_out, valid_out = [], [], []
    for identity, point, valid in zip(identities, points, validity):
        valid = bool(valid and point is not None and len(point) == 3 and all(_finite(value) for value in point))
        value = point if valid else previous.get(identity)
        if value is not None:
            identities_out.append(identity)
            points_out.append(tuple(float(coordinate) for coordinate in value))
            valid_out.append(valid)
    return tuple(identities_out), tuple(points_out), tuple(valid_out)


def _live_geometry_masks(side, hand, *, frozen=False):
    # Latest validity gates retained coordinates before asynchronous rebuilding.
    available = None if frozen else _hand_geometry_validity(hand)
    joints = {name: bool(valid and (frozen or available.get(name, False)))
              for name, valid in zip(side.joint_names, side.joint_valid)}
    taxels = {}
    for identifier, valid in zip(side.taxel_ids, side.taxel_valid):
        dependencies = side.taxel_dependencies.get(identifier)
        taxels[identifier] = bool(valid and (frozen or available.get("WRIST", False) and dependencies is not None
                                           and all(available.get(name, False) for name in dependencies)))
    return taxels, joints


def _array(value):
    result = _parse_json(value) if isinstance(value, str) else value
    if not isinstance(result, (list, tuple)):
        raise ValueError("Enter a JSON array")
    return result


def _common(values):
    values = tuple(values)
    if not values:
        return "", "No selected values"
    first = _json(values[0])
    if any(_json(value) != first for value in values[1:]):
        return "", "Mixed values — enter an explicit new value"
    return first, ""


def _replace_device_candidate(snapshot, device, label):
    return Candidate(snapshot.revision, replace_device(snapshot.document, device), label)


def _restore(document, model):
    if model is None or document.model is None:
        return document
    from .profiles import profile_for

    configured = model.with_config(document.model)
    return replace(document, devices=tuple(
        profile_for(device).restore_mounts(device, configured, document.bind_configuration)
        for device in document.devices
    ))


def _pose_candidate(snapshot, device_id, entity_id, pose, kind, model=None):
    device = snapshot.document.device(device_id)
    if kind == "segment":
        row = next(index for index, item in enumerate(device.segments) if item.id == entity_id)
        poses = list(device.segment_bind_poses)
        if row >= len(poses):
            raise ValueError("Missing B row; repair the complete B array first")
        poses[row] = pose
        device = replace(device, segment_bind_poses=poses, segments=tuple(
            replace(item, mount_pose=None) if item.id == entity_id else item
            for item in device.segments
        ))
        document = _restore(replace_device(snapshot.document, device), model)
    else:
        device = replace(device, taxels=tuple(
            replace(item, bind_pose=pose, surface_attachment=None,
                    generation_issue=None, provenance="manual")
            if item.id == entity_id else item for item in device.taxels
        ))
        document = replace_device(snapshot.document, device)
    return Candidate(snapshot.revision, document, f"Move {kind}")


@dataclass
class _Delivery:
    payload: dict
    ready: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None


class _BridgeSession:
    """Only admission and acknowledgements run in HTTP request threads."""

    def __init__(self, client_id, origin, inbox):
        self.client_id = str(client_id)
        self.origin = origin
        self.token = secrets.token_urlsafe(32)
        self.inbox = inbox
        self.epoch = secrets.token_hex(16)
        self.active = True
        self._lock = threading.Lock()
        self._next = 1
        self._deliveries = {}

    def revoke(self):
        with self._lock:
            self.active = False
            for delivery in self._deliveries.values():
                if not delivery.ready.is_set():
                    delivery.result = {"seq": delivery.payload["seq"], "revision": -1,
                                       "error": "Editor disconnected"}
                    delivery.ready.set()

    def admit(self, origin, payload):
        with self._lock:
            if (not self.active or origin != self.origin
                    or str(payload.get("client_id")) != self.client_id
                    or not isinstance(payload.get("token"), str)
                    or not secrets.compare_digest(payload["token"], self.token)
                    or payload.get("epoch") != self.epoch):
                raise PermissionError("Editor identity or Origin is invalid")
            sequence = payload.get("seq")
            if type(sequence) is not int or sequence < 1:
                raise ValueError("Invalid event sequence")
            if sequence in self._deliveries:
                old = self._deliveries[sequence]
                if old.payload != payload:
                    raise ValueError("A sequence cannot describe another event")
                return old
            if sequence != self._next:
                raise ValueError("Events must arrive in order")
            delivery = _Delivery(payload)
            self._deliveries[sequence] = delivery
            self._next += 1
            # Retain a bounded retry window; old sequences remain rejected.
            for previous in tuple(self._deliveries):
                if previous < sequence - 64 and self._deliveries[previous].ready.is_set():
                    del self._deliveries[previous]
            self.inbox.put(("bridge", self, delivery))
            return delivery


class _BridgeHTTP(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, inbox):
        self.session = None
        self.inbox = inbox
        super().__init__(("127.0.0.1", 0), _BridgeHandler)


class _BridgeHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def _respond(self, status, value=None):
        data = _json(value or {}).encode("utf-8")
        self.send_response(status)
        session = self.server.session
        if session is not None and self.headers.get("Origin") == session.origin:
            self.send_header("Access-Control-Allow-Origin", session.origin)
            self.send_header("Vary", "Origin")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self):
        session = self.server.session
        if (self.path != "/event" or session is None
                or self.headers.get("Origin") != session.origin):
            self._respond(403)
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", session.origin)
        self.send_header("Access-Control-Allow-Methods", "POST")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Vary", "Origin")
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if self.path != "/event" or not 0 < length <= 262144:
                raise ValueError("Invalid editor request")
            payload = _parse_json(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict) or self.server.session is None:
                raise PermissionError("No editor session")
            delivery = self.server.session.admit(self.headers.get("Origin"), payload)
        except PermissionError:
            self._respond(403)
            return
        except (ValueError, UnicodeError, TypeError):
            self._respond(400)
            return
        if not delivery.ready.wait(20):
            self._respond(503)
            return
        self._respond(200, delivery.result)


@dataclass
class _Control:
    handle: Any
    kind: str
    generation: int
    read: Callable | None = None
    build: Callable | None = None
    action: Callable | None = None
    field: str | None = None
    entity_id: str | None = None
    device_id: str | None = None
    name: str | None = None
    status: Any = None
    slot: int | None = None
    intended_disabled: bool = False

    @property
    def uuid(self):
        return self.handle._impl.uuid if self.name is None else self.name


@dataclass
class _Gesture:
    identity: str
    control: _Control
    snapshot: Snapshot
    device_id: str | None
    candidate: Candidate | None = None


class _LiveLifecycle:
    """One local lifecycle thread serializes replacement and shutdown."""

    def __init__(self, inbox):
        self.inbox = inbox
        self.condition = threading.Condition()
        self.target = None
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="tactile-live-lifecycle", daemon=True)
        self.thread.start()

    def request(self, generation, config):
        with self.condition:
            if self.closed:
                raise RuntimeError("Receiving lifecycle is closed")
            self.target = (generation, config)
            self.condition.notify()

    def _run(self):
        from .live import connect

        connection, applied = None, None
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.closed or self.target != applied)
                    target, closed = self.target, self.closed
                if connection is not None:
                    connection.disconnect()
                    connection = None
                if closed:
                    return
                generation, config = target
                applied = target
                if config is None:
                    self.inbox.put(("live_connection", generation, None, None))
                    continue
                try:
                    connection = connect(config)
                    with self.condition:
                        superseded = self.closed or self.target != target
                    if superseded:
                        continue
                    self.inbox.put(("live_connection", generation, connection, None))
                except Exception as exception:
                    self.inbox.put(("live_connection", generation, None, exception))
        finally:
            if connection is not None:
                connection.disconnect()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify()
        self.thread.join(timeout=8)
        if self.thread.is_alive():
            raise RuntimeError("Local receiving cleanup did not complete")


@dataclass(frozen=True)
class _LiveSide:
    device_id: str
    side: str
    taxel_ids: tuple
    positions: tuple
    colors: tuple
    joints: tuple
    lines: tuple
    line_colors: tuple
    force_lines: tuple
    force_colors: tuple
    meshes: tuple
    forces: FrozenMap
    joint_rows: tuple
    model_current: bool = False
    fingerprint: str | None = None
    taxel_valid: tuple = ()
    taxel_dependencies: FrozenMap = field(default_factory=FrozenMap)
    joint_names: tuple = ()
    joint_valid: tuple = ()
    line_joints: tuple = ()
    force_owners: tuple = ()


@dataclass(frozen=True)
class _LiveLayer:
    snapshot: Snapshot
    epoch: str
    seq: int
    read_monotonic_ns: int
    source: str
    synthetic: bool
    sides: tuple[_LiveSide, ...]
    issues: tuple


def _safe_viser():
    if importlib.metadata.version("viser") != VISER_VERSION:
        raise RuntimeError("This editor requires viser==1.1.1")
    spec = importlib.util.find_spec("viser")
    if spec is None or not spec.origin:
        raise RuntimeError("Viser is unavailable in the launching environment")
    package = Path(spec.origin).parent
    # ensure_client_is_built() returns immediately only for this packaged case.
    if (package.name == "viser" and package.parent.name == "src") or not (package / "client" / "build" / "index.html").is_file():
        raise RuntimeError("Viser must provide its packaged prebuilt client; automatic building is disabled")
    import viser

    return viser


class _Editor:
    def __init__(self, server, http, state, *, demo=False):
        self.server, self.http, self.state = server, http, state
        self.inbox = http.inbox
        self.clients = {}
        self.editor = None
        self.controls = {}
        self.transforms = {}
        self.generation = 0
        self.gesture = None
        self.pending_text = None
        self.pending_controls = ()
        self.preview = None
        self.model = None
        self.model_key = None
        self.selected_segment = None
        self.selected_link = None
        self.selected = ()
        self.pages = {"Segments": 0, "Layout": 0}
        self.active_tab = "Model"
        self.errors = {}
        self.messages = []
        self.check_results = {}
        self.jobs = set()
        self.job_threads = {}
        self.layout_generation = 0
        self.batch_summary = ""
        self.batch_summary_handle = None
        self.live_lifecycle = None
        self.live_connection = None
        self.live_generation = 0
        self.live_wanted = False
        self.receive_key = None
        self.live_status = ("disconnected", "", (), demo)
        self.live_frame = None
        self.live_poll_key = None
        self._last_live_poll = 0.0
        self.live_render_pending = False
        self.live_render_key = None
        self.live_render_error = None
        self.force_max = 30.0
        self.live_layer = None
        self.paused_layer = None
        self.received_cache = {}
        self.live_handles = {}
        self.record_groups = {name: True for name in METADATA_GROUPS}
        self.live_status_handle = self.live_mapping_handle = self.live_forces_handle = None
        self.display_version = 0
        self.display = None
        self.display_issues = ()
        self.hover_slot = None
        self.scene_handles = {}
        self.gizmo = None
        self.read_handles = []
        self.tabs = {}
        self.info = None
        self.demo = demo
        self.pending_operation = None
        self.stopping = False
        self.bridge_ready = False
        self.hidden_links = set()
        self._last_draw = 0.0
        self._draw_pending = False
        self.paths = {}
        self.temp = {
            "metadata_path": "", "urdf_path": "", "new_sides": "left",
            "packages": "{}", "selection": "", "through": False,
            "opacity": 0.65, "show_model": True, "point_mode": "Select taxel",
            "wrist_link": "", "wrist_origin": "[0,0,0]", "wrist_forearm": "[0,0,0.1]",
            "wrist_dorsal": "[0,0.1,0]", "batch_origin": "[0,0,0]",
            "batch_normal": "[0,0,1]", "batch_column_end": "null",
            "batch_row_end": "null", "batch_tangent": "[1,0,0]",
            "batch_column_span": "null", "batch_row_span": "null",
            "batch_rows": "1", "batch_cols": "1", "batch_scan": "row-major",
            "batch_reverse_rows": False, "batch_reverse_cols": False,
            "batch_excluded": "[]", "batch_links": "[]", "batch_seeds": "{}",
            "batch_start": "null", "batch_segment": "0", "batch_depth": "0.005",
            "batch_angle": "0", "batch_flip": False, "reorder": "[]",
            "out": "", "format": "json", "scope": "Current device",
            "visibility": "Both devices", "force_max": "30", "source_start": "0",
        }

    @property
    def device_id(self):
        return self.state.view.active_device_id

    def device(self, snapshot=None):
        return (snapshot or self.state.snapshot()).document.device(self.device_id)

    def _javascript(self, source, client=None):
        from viser import _messages

        (client or self.editor).gui._websock_interface.queue_message(_messages.RunJavascriptMessage(source=source))

    def _configure_bridge(self):
        if self.editor is None or self.http.session is None:
            return
        payload = {"revision": self.state.revision, "epoch": self.http.session.epoch,
                   "generation": self.generation,
                   "tabs": {tab._id: name for name, tab in self.tabs.items()},
                   "active_tab": self.tabs[self.active_tab]._id if self.active_tab in self.tabs else None,
                   "controls": [{"uuid": item.uuid, "kind": item.kind,
                                 "generation": item.generation, "hover": item.slot is not None,
                                 "disabled": item.handle.disabled}
                                for item in self.controls.values()],
                   "transforms": [{"uuid": item.uuid, "name": item.name,
                                   "kind": "transform", "generation": item.generation}
                                  for item in self.transforms.values()]}
        encoded = _json(payload)
        if encoded != getattr(self, "_bridge_configuration", None):
            self._bridge_configuration = encoded
            self._javascript("window.__tactileMetadataBridge.configure(" + encoded + ");")

    def _notice(self, value):
        self.messages = [*self.messages[-7:], str(value)]

    def _register(self, handle, kind, **kwargs):
        control = _Control(handle, kind, self.generation, intended_disabled=handle.disabled, **kwargs)
        handle.disabled = True
        self.controls[control.uuid] = control
        return handle

    def _button(self, label, action, *, disabled=False):
        return self._register(self.editor.gui.add_button(label, disabled=disabled), "action", action=action)

    def _text(self, label, read, build=None, *, field=None, entity=None, multiline=False, number=False):
        value = read(self.state.snapshot())
        handle = self.editor.gui.add_text(label, initial_value=str(value), multiline=multiline)
        status = self.editor.gui.add_markdown("") if field else None
        return self._register(handle, "number_text" if number else "text", read=read, build=build,
                              field=field, entity_id=entity, device_id=self.device_id, status=status)

    def _temp_text(self, label, key, *, multiline=False, number=False):
        return self._text(label, lambda _snapshot: self.temp[key],
                          lambda _snapshot, value, _source: self._set_temp(key, value),
                          multiline=multiline, number=number)

    def _set_temp(self, key, value):
        self.temp[key] = value
        if key.startswith("batch_"):
            self.layout_generation += 1
            self.preview = None
            self.batch_summary = ""
            if self.batch_summary_handle is not None:
                self.batch_summary_handle.content = "Range changed; preview again to see derived spans and spacing."
            self._button_preview_enabled()
            self._draw()

    def _choice(self, label, options, value, action):
        options = tuple(options)
        if value not in options:
            options = (value, *options)
        return self._register(self.editor.gui.add_dropdown(label, options=options, initial_value=value),
                              "choice", action=action)

    def _checkbox(self, label, value, action):
        return self._register(self.editor.gui.add_checkbox(label, initial_value=value), "checkbox", action=action)

    def _slider(self, label, value, lower, upper, build, *, field=None, entity=None, disabled=False, read=None):
        if not (_finite(value) and _finite(lower) and _finite(upper) and lower < upper and lower <= value <= upper):
            self.editor.gui.add_markdown(f"{label}: unavailable for invalid or mixed raw values")
            return None
        handle = self.editor.gui.add_slider(label, min=lower, max=upper, step=(upper - lower) / 1000,
                                             initial_value=float(value), disabled=disabled)
        return self._register(handle, "slider", build=build, read=read, field=field, entity_id=entity,
                              device_id=self.device_id)

    def _candidate_device(self, snapshot, update, label):
        return _replace_device_candidate(snapshot, update(snapshot.document.device(self.device_id)), label)

    def _cancel(self):
        self.gesture = None
        self.preview = None
        self._draw()

    def _receive(self, session, delivery):
        payload = delivery.payload
        try:
            if not session.active or session is not self.http.session:
                raise ValueError("Editor authority was revoked")
            self._event(payload)
            error = None
        except Exception as exception:
            error = str(exception)
            self._notice(error)
            self.errors[payload.get("control", "")] = error
            self.gesture = None
            self.preview = None
            try:
                self._draw()
            except Exception:
                LOGGER.exception("Unable to redraw the editor after an input error")
        delivery.result = {"seq": payload["seq"], "revision": self.state.revision,
                           "generation": self.generation, "error": error}
        delivery.ready.set()
        self._refresh()

    def _event(self, event):
        phase = event.get("phase")
        if phase == "ready":
            return
        if phase == "controls_ready":
            if event.get("generation") != self.generation:
                return
            self.bridge_ready = True
            self.pending_controls = ()
            for control in self.controls.values():
                control.handle.disabled = control.intended_disabled
            self._draw(force=True)
            return
        if phase == "controls_pending":
            if event.get("generation") == self.generation:
                self.pending_controls = tuple(
                    f"{item.get('uuid', '?')}: {item.get('reason', 'not mounted')}"
                    for item in event.get("missing", ())[:20] if isinstance(item, dict)
                )
            return
        if phase == "tab":
            if event.get("value") in self.tabs:
                self.active_tab = event["value"]
            return
        if phase != "incompatible" and not self.bridge_ready:
            raise ValueError("Editing is disabled until the browser confirms the registered controls")
        if phase in ("undo", "redo"):
            self.pending_text = None
            self.gesture = self.preview = None
            getattr(self.state, phase)()
            self._build()
            return
        control = self.controls.get(event.get("control")) or self.transforms.get(event.get("control"))
        if control is None or control.generation != event.get("generation"):
            raise ValueError("This control belongs to an obsolete page generation")
        if phase == "incompatible":
            self.pending_text = None
            self.gesture = self.preview = None
            self.bridge_ready = False
            for item in self.controls.values():
                item.handle.disabled = True
            if self.gizmo:
                self.gizmo.remove()
                self.gizmo = None
            self.transforms.clear()
            self._notice("Editing disabled: " + event.get("reason", "Browser bridge is incompatible") + "; reload after resolving the browser compatibility problem")
            return
        if phase == "cancel":
            if self.pending_text == (control.uuid, control.generation, event.get("gesture_id")):
                self.pending_text = None
            if self.gesture and self.gesture.identity == event.get("gesture_id"):
                self.gesture = self.preview = None
                self._draw()
                if control.kind == "transform":
                    self._create_gizmo(force=True)
            return
        if phase in ("focus", "dirty", "release"):
            if control.kind not in ("text", "number_text", "slider"):
                raise ValueError("Only a text field can own unconfirmed input")
            owner = (control.uuid, control.generation, event.get("gesture_id"))
            if not isinstance(owner[2], str) or not owner[2]:
                raise ValueError("Text input identity is missing")
            if phase == "focus":
                self.pending_text = owner
            elif self.pending_text != owner:
                raise ValueError("Unconfirmed text ownership changed")
            elif phase == "release":
                self.pending_text = None
            return
        if phase == "hover":
            self.hover_slot = control.slot if event.get("value") is True else None
            self._draw()
            return
        if phase == "action":
            if control.action is None or control.kind != "action":
                raise ValueError("Control is not an action")
            if self.gesture:
                self.gesture = self.preview = None
            control.action()
            return
        if phase == "commit":
            owner = (control.uuid, control.generation, event.get("gesture_id"))
            if self.pending_text == owner:
                self.pending_text = None
                if event.get("retain_focus") is True:
                    next_identity = event.get("next_gesture_id")
                    if not isinstance(next_identity, str) or not next_identity:
                        raise ValueError("Next text input identity is missing")
                    self.pending_text = (control.uuid, control.generation, next_identity)
            if control.kind in ("choice", "checkbox"):
                control.action(event.get("value"))
            elif control.build:
                candidate = control.build(self.state.snapshot(), event.get("value"), "number")
                if isinstance(candidate, Candidate):
                    self.state.dispatch(candidate)
                    self._after_document()
            self.errors.pop(control.uuid, None)
            return
        if phase == "begin":
            if control.kind not in ("slider", "transform"):
                raise ValueError("Only a slider or transform can begin a drag")
            if self.gesture:
                raise ValueError("Another gesture is active")
            if event.get("revision") != self.state.revision:
                raise ValueError("Gesture began from an obsolete document")
            identity = event.get("gesture_id")
            if not isinstance(identity, str) or not identity:
                raise ValueError("Gesture identity is missing")
            self.gesture = _Gesture(identity, control, self.state.snapshot(), self.device_id)
            self.preview = None
            return
        if phase in ("update", "end"):
            gesture = self.gesture
            if (gesture is None or gesture.control is not control
                    or gesture.identity != event.get("gesture_id")
                    or gesture.device_id != self.device_id
                    or gesture.snapshot.revision != self.state.revision):
                raise ValueError("Gesture identity, selection or revision changed")
            if event.get("value") is not None:
                candidate = control.build(gesture.snapshot, event["value"], "slider" if control.kind == "slider" else "transform")
                if not isinstance(candidate, Candidate):
                    raise ValueError("Gesture update must produce a Candidate")
                gesture.candidate = candidate
                self.preview = candidate
            if phase == "end":
                candidate = gesture.candidate
                self.gesture = self.preview = None
                if candidate is not None:
                    self.state.dispatch(candidate)
                    self._after_document()
            else:
                self._draw()
            return
        raise ValueError("Unknown editor event phase")

    def _load_model(self):
        configuration = self.state.snapshot().document.model
        key = None if configuration is None else (configuration.path, _json(configuration.packages))
        if key != self.model_key:
            self.model = None
            self.model_key = key
            if configuration is not None:
                from .model import load_urdf

                try:
                    self.model = load_urdf(configuration)
                except Exception as exception:
                    self._notice(f"Model unavailable: {exception}")
        elif self.model is not None:
            self.model = self.model.with_config(configuration)

    def _after_document(self, *, rebuild=False):
        self._load_model()
        self._sync_receiving()
        if self.device_id:
            device = self.device()
            self.selected = tuple(identifier for identifier in self.selected if identifier in {item.id for item in device.taxels})
            if self.selected_segment not in {item.id for item in device.segments}:
                self.selected_segment = None
        self.check_results = {key: value for key, value in self.check_results.items()
                              if value.revision == self.state.revision}
        if rebuild:
            self._build()
        else:
            self._draw()

    def _refresh(self):
        snapshot = self.state.snapshot()
        issues = (*self._current_issues(), *self.display_issues)
        for control in tuple(self.controls.values()):
            # Unconfirmed browser input must survive hover and background refreshes.
            active_text = self.pending_text and self.pending_text[:2] == (control.uuid, control.generation)
            if control.read and not active_text:
                try:
                    value = control.read(snapshot)
                    if control.kind == "slider":
                        valid = _finite(value)
                        valid = valid and control.handle.min <= value <= control.handle.max
                        control.handle.visible = valid
                        if valid and self.gesture is None:
                            control.handle.value = float(value)
                    else:
                        control.handle.value = str(value)
                except (KeyError, IndexError):
                    continue
            if control.status is not None:
                selected = [issue for issue in issues if self._matches_field(issue, control)]
                text = [f"**{issue.severity}:** {issue.message}" for issue in selected[:4]]
                if control.uuid in self.errors:
                    text.append("**Input error:** " + self.errors[control.uuid])
                control.status.content = "\n\n".join(text)
        self._refresh_info(issues)
        if hasattr(self, "undo_button"):
            self.undo_button.disabled = not self.bridge_ready or self.state.undo_count == 0
            self.redo_button.disabled = not self.bridge_ready or self.state.redo_count == 0
        if getattr(self, "check_report", None) is not None:
            from .checks import check_summary

            self.check_report.content = "\n\n".join("```json\n" + json.dumps(check_summary(result), indent=2, ensure_ascii=False)
                                                      + "\n```" for result in self.check_results.values())
        self._refresh_live_status()
        self._configure_bridge()

    def _refresh_info(self, issues=None):
        if self.info is not None:
            key = (id(self.state), self.state.revision)
            if issues is not None or getattr(self, "_dirty_cache", (None,))[0] != key:
                self._dirty_cache = (key, self.state.dirty())
            issues = (*self._current_issues(), *self.display_issues) if issues is None else issues
            lines = [f"Revision {self.state.revision} · {'Unsaved' if self._dirty_cache[1] else 'Saved'} · "
                     f"Undo {self.state.undo_count} / Redo {self.state.redo_count}"]
            if not self.bridge_ready:
                lines.append("**Editing disabled:** waiting for browser control identity verification, or compatibility failed.")
                lines.extend("- Waiting for control `" + reason + "`" for reason in self.pending_controls)
            if self.state.view.mode == "PREVIEW":
                lines.append("Gray markers without a valid motion result show only the q0 draft reference.")
            if self.state.view.mode == "LIVE" and not self._current_layer(self.live_layer):
                lines.append("The last live layer is unavailable history until a current frame is prepared; forces are not current.")
            if self.state.view.mode in ("LIVE", "PAUSED"):
                lines.append("Wireframe model and muted markers are the editable q0 reference; solid URDF uses JointState only.")
            for issue in issues[:20]:
                lines.append(f"- **{issue.severity}** `{issue.field or issue.code}`: {issue.message}")
            lines.extend(f"- {message}" for message in self.messages[-4:])
            lines.extend(f"- **Field error**: {message}" for message in self.errors.values())
            self.info.content = "\n".join(lines)

    @staticmethod
    def _matches_field(issue, control):
        if issue.device_id and control.device_id != issue.device_id:
            return False
        if issue.entity_id and control.entity_id and issue.entity_id != control.entity_id:
            return False
        aliases = {"skinning.indices": ("indices", "taxel_segment_indices"),
                   "skinning.weights": ("weights", "taxel_segment_weights"),
                   "taxel_bind_poses": ("bind_pose", "taxel_bind_poses"),
                   "segment_bind_poses": ("segment_bind_poses", "segment_poses")}
        return any(name in (issue.field or "") for name in aliases.get(control.field, (control.field,)))

    def _group_errors(self, fields):
        if not self.device_id:
            return
        issues = [issue for issue in self._current_issues() if issue.device_id in (None, self.device_id)]
        messages = [issue.message for issue in issues if any(name in (issue.field or "") for name in fields)]
        if messages:
            self.editor.gui.add_markdown("**Group errors:** " + "; ".join(dict.fromkeys(messages))[:1500])

    def _current_issues(self):
        snapshot = self.state.snapshot()
        cache = getattr(self, "_issue_cache", None)
        if cache is None or cache[0] != snapshot.revision:
            from .checks import declared_source_indices
            from .compiler import _numeric_rows
            from .profiles import profile_for

            issues = list(self.state.issues)
            # Full model/resource checks remain explicit background jobs.
            for device in snapshot.document.devices:
                prepared = _numeric_rows(device, allow_empty=False)
                issues.extend(prepared if isinstance(prepared, tuple) else prepared.issues)
                try:
                    issues.extend(profile_for(device).validate(device))
                except (ValueError, TypeError, OverflowError) as exception:
                    issues.append(Issue("error", "profile", str(exception), device.id, field="profile"))
                issues.extend(declared_source_indices(device)[1])
                issues.extend(Issue("error", "generation", str(taxel.generation_issue), device.id,
                                    taxel.id, "bind_pose") for taxel in device.taxels if taxel.generation_issue is not None)
            if self.model:
                issues.extend(self.model.configuration_issues(snapshot.document.bind_configuration))
            self._issue_cache = (snapshot.revision, tuple(issues))
        return self._issue_cache[1]

    def connect(self, client):
        self.clients[client.client_id] = client
        if self.editor is None:
            self.editor = client
            session = _BridgeSession(client.client_id, f"http://127.0.0.1:{self.server.get_port()}", self.inbox)
            self.http.session = session
            source = Path(__file__).with_name("ui_bridge.js").read_text(encoding="utf-8")
            configuration = {"version": VISER_VERSION, "client_id": str(client.client_id),
                             "token": session.token, "epoch": session.epoch,
                             "endpoint": f"http://127.0.0.1:{self.http.server_port}/event",
                             "revision": self.state.revision}
            self._javascript(source + "\nwindow.__tactileMetadataBridge.start(" + _json(configuration) + ");")
            self._build()
        else:
            self._readonly(client)

    def disconnect(self, client_id):
        self.clients.pop(client_id, None)
        if self.editor and self.editor.client_id == client_id:
            self._stop_receiving()
            self.http.session.revoke()
            self.http.session = None
            self.gesture = self.preview = None
            self.pending_text = None
            self.editor = None
            self.gizmo = None
            self.controls.clear()
            self.transforms.clear()
            self._draw()

    def _readonly(self, client):
        client.gui.reset()
        with client.gui.add_tab_group() as tabs:
            for name in ("Model", "Segments", "Layout", "Binding", "Realtime", "Export"):
                with tabs.add_tab(name):
                    client.gui.add_markdown("Read-only client. The editing client owns document changes.")
                    client.gui.add_text("Document revision", initial_value=str(self.state.revision), disabled=True)

    def _build(self):
        if self.editor is None:
            return
        self.generation += 1
        self.layout_generation += 1
        self.pending_text = None
        self.pending_controls = ()
        self.bridge_ready = False
        self.gesture = self.preview = None
        self.controls.clear()
        self.transforms.clear()
        self.errors.clear()
        self.hover_slot = None
        self._issue_cache = None
        self.editor.gui.reset()
        self._load_model()
        self._sync_receiving()
        gui = self.editor.gui
        self.undo_button = self._button("Undo", lambda: self._history("undo"), disabled=self.state.undo_count == 0)
        self.redo_button = self._button("Redo", lambda: self._history("redo"), disabled=self.state.redo_count == 0)
        devices = self.state.snapshot().document.devices
        options = {f"{item.side or 'unassigned'} · {item.id}": item.id for item in devices}
        selected = next((name for name, identifier in options.items() if identifier == self.device_id), None)
        if selected is not None and len(devices) > 1:
            self._choice("Editing device", options, selected, lambda value: self._select_device(options[value]))
            self._choice("Metadata visibility", ("Current device", "Both devices"), self.temp["visibility"],
                         lambda value: self._set_view("visibility", value))
        self._choice("Mode", ("EDIT", "PREVIEW", "LIVE", "PAUSED"), self.state.view.mode, self._mode)
        self._choice("3D metadata source", ("draft", "received"), self.state.view.metadata_source, self._source)
        with gui.add_tab_group() as group:
            for name, render in (("Model", self._model_page), ("Segments", self._segments_page),
                                 ("Layout", self._layout_page), ("Binding", self._binding_page),
                                 ("Realtime", self._realtime_page), ("Export", self._export_page)):
                with group.add_tab(name) as tab:
                    self.tabs[name] = tab
                    render()
        with gui.add_folder("Info"):
            self._draw(force=True)
            self.info = gui.add_markdown("")
            if self.device_id:
                for issue in (*self._current_issues(), *self.display_issues)[:20]:
                    self._button(f"Locate {issue.field or issue.code}: {issue.message[:55]}", lambda item=issue: self._locate(item))
        self._draw()
        self._refresh()

    def _history(self, direction):
        self.gesture = self.preview = None
        getattr(self.state, direction)()
        self._build()

    def _select_device(self, device_id):
        self.state.view = replace(self.state.view, active_device_id=device_id, selected_ids=())
        self.selected = ()
        self.selected_segment = None
        self.pages = {"Segments": 0, "Layout": 0}
        for key in ("batch_origin", "batch_column_end", "batch_row_end", "batch_normal", "batch_tangent"):
            self.temp[key] = "null"
        self.temp.update(batch_links="[]", batch_seeds="{}", batch_start="null", batch_segment="null")
        self._notice("Select this side's surface, tangent/normal, segment row, and output index before generating taxels")
        self._build()

    def _mode(self, mode):
        if mode == "PAUSED" and self.state.view.mode != "PAUSED":
            layer = self.live_layer if self.state.view.mode == "LIVE" else None
            self.paused_layer = (replace(layer, sides=tuple(self._display_live_side(layer, side) for side in layer.sides))
                                 if layer else None)
        elif mode != "PAUSED":
            self.paused_layer = None
        self.state.view = replace(self.state.view, mode=mode)
        self._build()

    def _source(self, source):
        self.state.view = replace(self.state.view, metadata_source=source)
        self._draw()

    def _locate(self, issue):
        if issue.device_id and issue.device_id != self.device_id:
            self._select_device(issue.device_id)
        device = self.device() if self.device_id else None
        if device and issue.entity_id in {item.id for item in device.taxels}:
            self.selected = (issue.entity_id,)
            self.pages["Layout"] = next(index for index, item in enumerate(device.taxels) if item.id == issue.entity_id) // PAGE_SIZE
            self.active_tab = "Binding" if "skinning" in (issue.field or "") else "Layout"
        elif device and issue.entity_id in {item.id for item in device.segments}:
            self.selected_segment = issue.entity_id
            self.active_tab = "Segments"
        else:
            self.active_tab = "Binding" if "skinning" in (issue.field or "") else "Segments"
        field = issue.field or ""
        if field.startswith(("application_binding", "hand_source", "joint_source", "live")):
            self.active_tab = "Realtime"
        elif field.startswith(("model", "link_adjustments", "wrist_mount")):
            self.active_tab = "Model"
        elif "taxel_segment_" in field or "skinning" in field:
            self.active_tab = "Binding"
        elif field.startswith("sdk"):
            self.active_tab = "Export"
        self._build()
        self._notice(f"Open {self.active_tab}; selected field: {issue.field or issue.code}")

    def _request(self, operation):
        if self.state.dirty():
            self.pending_operation = operation
            self._build()
        else:
            operation()

    def _resolve_pending(self, decision):
        operation = self.pending_operation
        if decision == "Save":
            self._save(all_devices=True)
        self.pending_operation = None
        if decision != "Cancel" and operation:
            operation()
        else:
            self._build()

    def _save(self, *, all_devices=False):
        from .document import save_metadata

        snapshot = self.state.snapshot()
        identifiers = tuple(item.id for item in snapshot.document.devices) if all_devices else (self.device_id,)
        written = []
        for identifier in identifiers:
            path = self.paths.get(identifier, self.temp["metadata_path"] if len(identifiers) == 1 else "")
            if not path:
                raise ValueError("Set a separate metadata save path for each device")
            save_metadata(snapshot, path, device_id=identifier)
            self.paths[identifier] = path
            written.append(identifier)
        for identifier in written:
            self.state.mark_saved(snapshot, identifier)
        self._notice(f"Saved document snapshot at revision {snapshot.revision}")

    def _replace_document(self, document, *, saved=False):
        self._stop_receiving(clear=True)
        self.state = State(document, saved=saved)
        self.check_results.clear()
        self.selected = ()
        self.selected_segment = self.selected_link = None
        self.pages = {"Segments": 0, "Layout": 0}
        self.model_key = object()
        self._build()

    def _open_metadata(self):
        from .document import load_metadata

        path = self.temp["metadata_path"]
        document = load_metadata(path)
        self.paths = {item.id: path for item in document.devices}
        self._replace_document(document, saved=True)

    def _replace_metadata(self):
        from .document import load_metadata

        document = load_metadata(self.temp["metadata_path"], side=self.device().side)
        if len(document.devices) != 1:
            raise ValueError("Select one metadata JSON for the current side")
        current_document = self.state.snapshot().document
        incoming_payload = _parse_json(Path(self.temp["metadata_path"]).read_text(encoding="utf-8"))
        if "editor" in incoming_payload:
            if (document.model != current_document.model
                    or document.bind_configuration != current_document.bind_configuration):
                raise ValueError("The editor model or binding configuration differs; use Open metadata JSON as a new document")
        incoming = document.devices[0]
        snapshot = self.state.snapshot()
        current = self.device(snapshot)
        incoming = replace(incoming, id=current.id, side=current.side)
        self.state.dispatch(Candidate(snapshot.revision, replace_device(snapshot.document, incoming), "Replace side metadata"))
        self.paths[current.id] = self.temp["metadata_path"]
        self.selected = ()
        self.selected_segment = None
        self._after_document(rebuild=True)

    def _associate_model(self):
        if self.state.snapshot().document.model is not None:
            raise ValueError("This document already has a fixed URDF")
        from .model import load_urdf

        snapshot = self.state.snapshot()
        declared = self.temp["new_sides"]
        config = ModelConfig(self.temp["urdf_path"], declared, FrozenMap(_parse_json(self.temp["packages"])))
        model = load_urdf(config)
        config = replace(config, asset_hashes=model.resource_hashes)
        devices = snapshot.document.devices
        if len(devices) == 1 and devices[0].side is None and declared != "both":
            devices = (replace(devices[0], side=declared),)
        if declared != "both" and any(device.side not in (None, declared) for device in devices):
            raise ValueError("Metadata side conflicts with the declared URDF side")
        updated = replace(snapshot.document, model=config, devices=devices)
        self.state.dispatch(Candidate(snapshot.revision, updated, "Associate URDF"))
        self._after_document(rebuild=True)

    def _new_model(self):
        from .model import load_urdf

        config = ModelConfig(self.temp["urdf_path"], self.temp["new_sides"],
                             FrozenMap(_parse_json(self.temp["packages"])))
        model = load_urdf(config)
        config = replace(config, asset_hashes=model.resource_hashes)
        sides = ("left", "right") if config.sides == "both" else (config.sides,)
        document = Document(model=config, devices=tuple(Device(side=side) for side in sides))
        self.paths.clear()
        self._replace_document(document)

    def _model_page(self):
        gui = self.editor.gui
        with gui.add_folder("Files"):
            self._temp_text("Metadata JSON path", "metadata_path")
            self._button("Open metadata JSON", lambda: self._request(self._open_metadata))
            if self.device_id:
                self._button("Replace current side from JSON", lambda: self._request(self._replace_metadata))
            if self.device_id:
                self._text("Save path for this device", lambda _snapshot: self.paths.get(self.device_id, ""),
                           lambda _snapshot, value, _source: self.paths.__setitem__(self.device_id, value))
                self._button("Save current draft", self._save)
                self._button("Save all drafts", lambda: self._save(all_devices=True))
            self._button("Close editor", lambda: self._request(lambda: setattr(self, "stopping", True)))
            if self.pending_operation:
                gui.add_markdown("**Unsaved document:** save, discard, or cancel before opening a new document.")
                for decision in ("Save", "Discard", "Cancel"):
                    self._button(decision, lambda decision=decision: self._resolve_pending(decision))
        with gui.add_folder("New URDF document"):
            self._temp_text("URDF path", "urdf_path")
            self._choice("Sides declared by this URDF", ("left", "right", "both"), self.temp["new_sides"],
                         lambda value: self.temp.__setitem__("new_sides", value))
            self._temp_text("Package directories JSON", "packages", multiline=True)
            self._button("Create URDF document", lambda: self._request(self._new_model))
            if self.state.snapshot().document.model is None and self.device_id:
                self._button("Associate URDF with numeric draft", self._associate_model)
        config = self.state.snapshot().document.model
        if self.device_id and self.device().side is None:
            self._choice("Assign unassigned metadata side", ("Choose side", "left", "right"), "Choose side", self._assign_side)
        if config is not None:
            gui.add_markdown(f"URDF: `{config.path}`\n\nFixed sides: **{config.sides}**")
            self._text("Current package directories JSON", lambda snapshot: _json(snapshot.document.model.packages),
                       self._packages_candidate, field="model.packages", multiline=True)
        if self.model is None:
            gui.add_markdown("No usable URDF. Numeric metadata edits, saving, checks, and export remain available.")
            return
        gui.add_markdown(f"{len(self.model.links)} links · {self.model.triangle_count} triangles")
        with gui.add_folder("Model joints and resources"):
            for joint in self.model.joints:
                gui.add_markdown(f"`{joint['name']}` · {joint['type']} · limits [{joint['lower']}, {joint['upper']}] · mimic {joint['mimic']}")
            for issue in self.model.resource_issues():
                gui.add_markdown("**Resource error:** " + issue.message)
        self._checkbox("Show model", self.temp["show_model"], lambda value: self._set_view("show_model", value))
        self._choice("Model opacity value", ("0.25", "0.5", "0.75", "1"), str(self.temp["opacity"]),
                     lambda value: self._set_view("opacity", float(value)))
        self.selected_link = self.selected_link if self.selected_link in self.model.links else self.model.links[0]
        self._choice("URDF link", self.model.links, self.selected_link, self._select_link)
        self._checkbox("Show selected link", self.selected_link not in self.hidden_links, self._link_visibility)
        self._text("Link local adjustment XYZ / XYZW", lambda snapshot: _json(
            snapshot.document.model.link_adjustments.get(self.selected_link)), self._link_adjustment_candidate,
            field="link_adjustments", entity=self.selected_link)
        if self.device_id:
            with gui.add_folder("Wrist reference at q0"):
                link = self.temp["wrist_link"] if self.temp["wrist_link"] in self.model.links else self.model.links[0]
                self.temp["wrist_link"] = link
                self._choice("Wrist link", self.model.links, link, lambda value: self.temp.__setitem__("wrist_link", value))
                for label, key in (("Origin O in model coordinates", "wrist_origin"),
                                   ("Forearm point F", "wrist_forearm"), ("Dorsal point D", "wrist_dorsal")):
                    self._temp_text(label, key)
                self._choice("Surface click action", ("Select taxel", "Wrist O", "Wrist F", "Wrist D",
                                                        "Batch origin", "Batch column end", "Batch row end", "Place taxel", "Place selected segment"),
                             self.temp["point_mode"], lambda value: self.temp.__setitem__("point_mode", value))
                self._button("Calibrate wrist reference", self._calibrate_wrist)
                self._text("Raw wrist mount JSON", lambda snapshot: _json(snapshot.document.device(self.device_id).wrist_mount),
                           self._wrist_candidate, field="wrist_mount", multiline=True)

    def _packages_candidate(self, snapshot, value, _source):
        packages = FrozenMap(_parse_json(value))
        return Candidate(snapshot.revision, replace(snapshot.document, model=replace(snapshot.document.model, packages=packages)),
                         "Set package directories")

    def _assign_side(self, value):
        if value == "Choose side":
            return
        document = self.state.snapshot().document
        if document.model and document.model.sides != "both" and value != document.model.sides:
            raise ValueError("The selected side conflicts with the fixed URDF declaration")
        if any(item.side == value for item in document.devices):
            raise ValueError("This side already has metadata")
        self.state.edit_device(self.device_id, lambda device: replace(device, side=value), "Assign metadata side")
        self._after_document(rebuild=True)

    def _set_view(self, key, value):
        self.temp[key] = value
        self._draw()

    def _select_link(self, value):
        self.selected_link = value
        self.active_tab = "Model"
        self._build()

    def _link_visibility(self, visible):
        if visible:
            self.hidden_links.discard(self.selected_link)
        else:
            self.hidden_links.add(self.selected_link)
        self._draw()

    def _link_adjustment_candidate(self, snapshot, value, _source):
        adjustments = thaw(snapshot.document.model.link_adjustments)
        adjustments[self.selected_link] = _parse_json(value) if isinstance(value, str) else value
        document = replace(snapshot.document, model=replace(snapshot.document.model, link_adjustments=adjustments))
        return Candidate(snapshot.revision, _restore(document, self.model), "Move URDF link")

    def _wrist_candidate(self, snapshot, value, _source):
        device = replace(snapshot.document.device(self.device_id), wrist_mount=_parse_json(value))
        return Candidate(snapshot.revision, _restore(replace_device(snapshot.document, device), self.model), "Set wrist reference")

    def _calibrate_wrist(self):
        from .profiles.hand import wrist_mount

        snapshot = self.state.snapshot()
        mount = wrist_mount(self.model, snapshot.document.bind_configuration, self.temp["wrist_link"],
                            _parse_json(self.temp["wrist_origin"]), _parse_json(self.temp["wrist_forearm"]),
                            _parse_json(self.temp["wrist_dorsal"]))
        self.state.set_wrist_mount(self.device_id, mount, self.model)
        self._after_document(rebuild=True)

    def _q_candidate(self, snapshot, name, value):
        values = thaw(snapshot.document.bind_configuration)
        values[name] = _number(value)
        document = replace(snapshot.document, bind_configuration=FrozenMap(values))
        return Candidate(snapshot.revision, _restore(document, self.model), "Set binding configuration")

    def _segments_page(self):
        gui = self.editor.gui
        if not self.device_id:
            return
        with gui.add_folder("Binding configuration q0"):
            if len(self.state.snapshot().document.devices) > 1:
                gui.add_markdown("Shared q0 changes affect both left and right metadata.")
            self._group_errors(("bind_configuration",))
            self._text("Raw q0 JSON", lambda snapshot: _json(snapshot.document.bind_configuration),
                       lambda snapshot, value, _source: Candidate(snapshot.revision, _restore(
                           replace(snapshot.document, bind_configuration=FrozenMap(_parse_json(value))), self.model), "Set q0"),
                       field="bind_configuration", multiline=True)
            if self.model:
                for joint in self.model.joints:
                    name = joint["name"]
                    if name not in self.model.actuated_joint_names:
                        continue
                    build = lambda snapshot, value, _source, name=name: self._q_candidate(snapshot, name, value)
                    self._text(name + " q0", lambda snapshot, name=name: _json(snapshot.document.bind_configuration.get(name, 0)),
                               build, field="bind_configuration." + name, number=True)
                    lower, upper = joint["lower"], joint["upper"]
                    if lower is not None and upper is not None and lower < upper:
                        self._slider(name + " q0 slider", self.state.snapshot().document.bind_configuration.get(name, 0),
                                     lower, upper, build, field="bind_configuration." + name,
                                     read=lambda snapshot, name=name: snapshot.document.bind_configuration.get(name, 0))
            self._button("Set bind pose from preview joints", self._bind_from_preview, disabled=self.model is None)
        device = self.device()
        with gui.add_folder("Segments"):
            self._group_errors(("segment", "semantic"))
            gui.add_markdown("Row · HandJoint · Link (row numbers are the skinning references)")
            start = self.pages["Segments"] * PAGE_SIZE
            for row, segment in enumerate(device.segments[start:start + PAGE_SIZE], start):
                self._button(f"{row} · {segment.semantic} · {segment.frame_ref}", lambda identifier=segment.id: self._select_segment(identifier))
            self._pagination(len(device.segments), "Segments")
            self._button("Add segment", self._add_segment)
            self._button("Delete selected segment", self._delete_segment, disabled=self.selected_segment is None)
            self._temp_text("Segment UUID output order JSON", "reorder", multiline=True)
            gui.add_markdown("Current order: `" + _json(tuple(item.id for item in device.segments)) + "`")
            self._button("Apply segment order", self._reorder_segments)
        segment = next((item for item in device.segments if item.id == self.selected_segment), None)
        if segment:
            row = next(index for index, item in enumerate(device.segments) if item.id == segment.id)
            with gui.add_folder("Selected segment"):
                from .profiles.hand import JOINT_NAMES

                self._choice("HandJoint", JOINT_NAMES, str(segment.semantic),
                             lambda value: self._semantic(segment.id, value))
                if self.model:
                    link = segment.frame_ref[5:] if segment.frame_ref.startswith("link:") else "root"
                    self._choice("Associated link", ("root", *self.model.links), link,
                                 lambda value: self._associate(segment.id, value))
                self._text("B row XYZ / XYZW", lambda snapshot: _json(self._segment_pose(snapshot, segment.id)),
                           lambda snapshot, value, _source: self._segment_json_pose(snapshot, segment.id, value),
                           field="segment_bind_poses", entity=segment.id)
                self._button("Initialize from selected link reference", lambda: self._initialize_segment(segment.id),
                             disabled=self.model is None)
                self._button("Place selected segment on model", lambda: self.temp.__setitem__("point_mode", "Place selected segment"),
                             disabled=self.model is None)
                self._pose_components("Segment", (segment.id,), "segment")
                gui.add_markdown(f"B row {row}; frame reference `{segment.frame_ref}`")
        with gui.add_folder("Raw skeleton draft repair"):
            self._text("Complete B array JSON", lambda snapshot: _json(snapshot.document.device(self.device_id).segment_bind_poses),
                       lambda snapshot, value, _source: self._candidate_device(snapshot, lambda device: replace(
                           device, segment_bind_poses=_array(value)), "Edit raw B array"),
                       field="segment_bind_poses", multiline=True)
            self._text("Complete joint names JSON", lambda snapshot: _json(tuple(
                item.semantic for item in snapshot.document.device(self.device_id).segments)), self._raw_semantics,
                       field="semantic", multiline=True)
        with gui.add_folder("Preview joints"):
            gui.add_markdown("These controls affect only the preview. Forms and manipulators remain at q0.")
            if self.model:
                for name in self.model.actuated_joint_names:
                    self._text(name + " preview", lambda _snapshot, name=name: _json(self.state.view.q_preview.get(name,
                        self.state.snapshot().document.bind_configuration.get(name, 0))),
                        lambda _snapshot, value, _source, name=name: self._preview_joint(name, value), number=True)

    def _segment_pose(self, snapshot, identifier):
        device = snapshot.document.device(self.device_id)
        row = next(index for index, item in enumerate(device.segments) if item.id == identifier)
        return device.segment_bind_poses[row] if row < len(device.segment_bind_poses) else None

    def _bind_from_preview(self):
        values = thaw(self.state.snapshot().document.bind_configuration)
        values.update(thaw(self.state.view.q_preview))
        self.state.set_bind_configuration(values, self.model)
        self._after_document(rebuild=True)

    def _preview_joint(self, name, value):
        values = thaw(self.state.view.q_preview)
        values[name] = _number(value)
        self.state.view = replace(self.state.view, q_preview=FrozenMap(values))
        self._build()

    def _select_segment(self, identifier):
        self.selected_segment = identifier
        self.selected = ()
        self.active_tab = "Segments"
        self._build()

    def _add_segment(self):
        snapshot = self.state.snapshot()
        segment = Segment()
        candidate = self._candidate_device(snapshot, lambda device: replace(device,
            segments=(*device.segments, segment), segment_bind_poses=(*device.segment_bind_poses, None)
            if len(device.segment_bind_poses) == len(device.segments) else device.segment_bind_poses), "Add segment")
        self.state.dispatch(candidate)
        self.selected_segment = segment.id
        self._after_document(rebuild=True)

    def _delete_segment(self):
        self.state.delete_segments(self.device_id, (self.selected_segment,))
        self.selected_segment = None
        self._after_document(rebuild=True)

    def _reorder_segments(self):
        self.state.reorder_segments(self.device_id, _array(self.temp["reorder"]))
        self._after_document(rebuild=True)

    def _semantic(self, identifier, value):
        self.state.edit_device(self.device_id, lambda device: replace(device, segments=tuple(
            replace(item, semantic=value) if item.id == identifier else item for item in device.segments)), "Set HandJoint")
        self._after_document(rebuild=True)

    def _associate(self, identifier, link):
        if link == "root":
            self.state.edit_device(self.device_id, lambda device: replace(device, segments=tuple(
                replace(item, frame_ref="root", mount_pose=None) if item.id == identifier else item
                for item in device.segments)), "Clear segment link")
        else:
            self.state.associate_segment(self.device_id, identifier, link, self.model)
        self._after_document(rebuild=True)

    def _raw_semantics(self, snapshot, value, _source):
        names = _array(value)
        device = snapshot.document.device(self.device_id)
        segments = tuple(replace(device.segments[index], semantic=name) if index < len(device.segments)
                         else Segment(semantic=name) for index, name in enumerate(names))
        return self._candidate_device(snapshot, lambda device: replace(device, segments=segments), "Edit joint names")

    def _segment_json_pose(self, snapshot, identifier, value):
        device = snapshot.document.device(self.device_id)
        row = next(index for index, item in enumerate(device.segments) if item.id == identifier)
        if row >= len(device.segment_bind_poses):
            raise ValueError("The B row is missing. Edit the complete B array to explicitly add it")
        return _pose_candidate(snapshot, self.device_id, identifier, _parse_json(value), "segment", self.model)

    def _initialize_segment(self, identifier):
        import numpy as np
        from .geometry import matrix_pose
        from .profiles import profile_for

        snapshot = self.state.snapshot()
        device = self.device(snapshot)
        segment = next(item for item in device.segments if item.id == identifier)
        if segment.semantic == "WRIST":
            pose = (0, 0, 0, 0, 0, 0, 1)
        else:
            if not segment.frame_ref.startswith("link:"):
                raise ValueError("Associate this segment with a model link before initializing its reference")
            configured = self.model.with_config(snapshot.document.model)
            root = profile_for(device).root_pose(device, configured, snapshot.document.bind_configuration)
            pose = matrix_pose(np.linalg.inv(root) @ configured.link_pose(segment.frame_ref[5:], snapshot.document.bind_configuration))
        self.state.dispatch(_pose_candidate(snapshot, self.device_id, identifier, pose, "segment", self.model))
        self._after_document(rebuild=True)

    def _pagination(self, count, page_key):
        pages = max(1, (count + PAGE_SIZE - 1) // PAGE_SIZE)
        page = self.pages[page_key] = min(self.pages[page_key], pages - 1)
        self.editor.gui.add_markdown(f"Page {page + 1} / {pages}; {PAGE_SIZE} rows per page")
        self._button("Previous page", lambda: self._set_page(page_key, page - 1), disabled=page == 0)
        self._button("Next page", lambda: self._set_page(page_key, page + 1), disabled=page >= pages - 1)

    def _set_page(self, page_key, page):
        self.pages[page_key] = page
        self._build()

    def _pose_components(self, label, identifiers, kind):
        def poses(snapshot):
            device = snapshot.document.device(self.device_id)
            return tuple(self._segment_pose(snapshot, identifier) if kind == "segment" else
                         next(item.bind_pose for item in device.taxels if item.id == identifier) for identifier in identifiers)

        values = poses(self.state.snapshot())
        if not all(isinstance(pose, tuple) and len(pose) == 7 for pose in values):
            self.editor.gui.add_markdown("Missing or malformed pose row; use the raw XYZ / XYZW JSON field.")
            return
        for component, name in enumerate(("X m", "Y m", "Z m", "Qx", "Qy", "Qz", "Qw")):
            value, hint = _common(pose[component] for pose in values)
            if hint:
                self.editor.gui.add_markdown(name + ": " + hint)
            def read(snapshot, component=component):
                return _common(pose[component] for pose in poses(snapshot))[0]
            def build(snapshot, value, _source, component=component):
                current = snapshot
                for identifier, pose in zip(identifiers, poses(snapshot)):
                    updated = list(pose)
                    updated[component] = _number(value)
                    candidate = _pose_candidate(current, self.device_id, identifier, updated, kind, self.model)
                    current = Snapshot(candidate.document, snapshot.revision)
                return Candidate(snapshot.revision, current.document, f"Set {label} {name}")
            self._text(label + " " + name, read, build, field="segment_bind_poses" if kind == "segment" else "taxel_bind_poses",
                       entity=identifiers[0] if len(identifiers) == 1 else None, number=True)

    def _layout_page(self):
        gui = self.editor.gui
        if not self.device_id:
            return
        device = self.device()
        with gui.add_folder("Selection"):
            gui.add_markdown(f"{len(device.taxels)} taxels · {len(self.selected)} selected")
            self._temp_text("Taxel indices or ranges", "selection")
            self._button("Select indices", self._select_indices)
            self._checkbox("Through selection", self.temp["through"], lambda value: self.temp.__setitem__("through", value))
            gui.add_markdown("Click a taxel to select; Ctrl/Cmd-click toggles. Shift-drag selects a rectangle.")
            self._button("Clear selection", lambda: self._select_taxels(()))
            page = self.pages["Layout"]
            for taxel in device.taxels[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]:
                self._button(f"Taxel {taxel.taxel_index} · {taxel.id}", lambda identifier=taxel.id: self._select_taxels((identifier,)))
            self._pagination(len(device.taxels), "Layout")
        with gui.add_folder("Range and count"):
            gui.add_markdown("Coordinates are in the hand root at q0. Surface click actions are on the Model page.")
            self.batch_summary_handle = gui.add_markdown(self.batch_summary or "A null span derives length from the picked endpoints; preview reports spans and spacing.")
            for label, key in (("Origin O", "batch_origin"), ("Column endpoint C or null", "batch_column_end"),
                               ("Row endpoint R or null", "batch_row_end"), ("Normal", "batch_normal"),
                               ("Tangent", "batch_tangent"), ("Column span m", "batch_column_span"),
                               ("Row span m", "batch_row_span"), ("Rows", "batch_rows"), ("Columns", "batch_cols")):
                self._temp_text(label, key)
            self._choice("Scan order", ("row-major", "column-major", "row-snake", "column-snake"), self.temp["batch_scan"],
                         lambda value: self._set_temp("batch_scan", value))
            for label, key in (("Reverse rows", "batch_reverse_rows"), ("Reverse columns", "batch_reverse_cols"),
                               ("Flip projection normal", "batch_flip")):
                self._checkbox(label, self.temp[key], lambda value, key=key: self._set_temp(key, value))
            for label, key in (("Excluded [row,column] cells JSON", "batch_excluded"), ("Allowed link names JSON", "batch_links"),
                               ("Surface component seeds JSON", "batch_seeds"), ("Start taxel index or null", "batch_start"),
                               ("Initial segment row", "batch_segment"), ("Projection depth m", "batch_depth"),
                               ("Tangent angle rad", "batch_angle")):
                self._temp_text(label, key)
            self._button("Preview generated taxels", self._preview_batch, disabled=self.model is None)
            self._button("Apply generated taxels", self._apply_batch, disabled=self.preview is None)
            self._button("Cancel generated taxels", self._cancel_batch)
            if self.preview:
                generated = self.preview.document.device(self.device_id)
                new = tuple(item for item in generated.taxels if item.id not in {old.id for old in device.taxels})
                gui.add_markdown("Preview: " + str(len(new)) + " taxels\n\n" + "\n".join(
                    f"- {item.taxel_index}: {'projection failed' if item.bind_pose is None else _json(item.bind_pose[:3])}"
                    for item in new[:25]))
            self._button("Reproject selected taxels", self._reproject, disabled=self.model is None or not self.selected)
        with gui.add_folder("Selected taxels"):
            self._group_errors(("taxel_bind_poses", "bind_pose", "taxel_index"))
            self._pose_components("Taxel", self.selected, "taxel") if self.selected else None
            if len(self.selected) == 1:
                identifier = self.selected[0]
                self._text("Taxel index", lambda snapshot: _json(next(item.taxel_index for item in
                    snapshot.document.device(self.device_id).taxels if item.id == identifier)),
                    lambda snapshot, value, _source: self._candidate_device(snapshot, lambda device: replace(device, taxels=tuple(
                        replace(item, taxel_index=_number(value)) if item.id == identifier else item for item in device.taxels)),
                        "Set taxel index"), field="taxel_index", entity=identifier, number=True)
                self._text("Raw taxel pose XYZ / XYZW", lambda snapshot: _json(next(item.bind_pose for item in
                    snapshot.document.device(self.device_id).taxels if item.id == identifier)),
                    lambda snapshot, value, _source: _pose_candidate(snapshot, self.device_id, identifier,
                        _parse_json(value), "taxel"), field="taxel_bind_poses", entity=identifier)
            self._button("Flip selected normals", self._flip_normals, disabled=not self.selected)
            self._temp_text("Selected tangent rotation rad", "batch_angle", number=True)
            self._button("Rotate selected tangents", self._rotate_tangents, disabled=not self.selected)
            self._button("Delete selected taxels", self._delete_taxels, disabled=not self.selected)
        with gui.add_folder("Output order"):
            self._temp_text("Taxel UUID output order JSON", "reorder", multiline=True)
            gui.add_markdown("Current order: `" + _json(tuple(item.id for item in device.taxels)) + "`")
            self._button("Preview output index changes", self._preview_order)
            self._button("Apply taxel output order", self._reorder_taxels)
        with gui.add_folder("Raw taxel draft repair"):
            self._text("Complete P array JSON", lambda snapshot: _json(tuple(item.bind_pose for item in
                snapshot.document.device(self.device_id).taxels)), self._raw_poses, field="taxel_bind_poses", multiline=True)

    def _batch_spec(self):
        from .layout import BatchSpec

        parse = lambda key: _parse_json(self.temp["batch_" + key])
        return BatchSpec(self.device_id, origin=parse("origin"), normal=parse("normal"),
                         rows=parse("rows"), cols=parse("cols"), column_end=parse("column_end"),
                         row_end=parse("row_end"), tangent=parse("tangent"), column_span=parse("column_span"),
                         row_span=parse("row_span"), scan=self.temp["batch_scan"],
                         reverse_rows=self.temp["batch_reverse_rows"], reverse_cols=self.temp["batch_reverse_cols"],
                         excluded=parse("excluded"), allowed_links=parse("links"), component_seeds=parse("seeds"),
                         start_index=parse("start"), segment_row=parse("segment"), depth=parse("depth"),
                         tangent_angle=parse("angle"), flip_normal=self.temp["batch_flip"])

    def _preview_batch(self):
        from .layout import generate_batch, grid_samples

        spec = self._batch_spec()
        snapshot, starting_state = self.state.snapshot(), self.state
        device_id, model = self.device_id, self.model.with_config(snapshot.document.model)
        selected = tuple(self.selected)
        self.layout_generation += 1
        generation = self.layout_generation
        self.preview = None
        self._draw()
        self._button_preview_enabled()
        def generate():
            _points, _column, _normal, pitch = grid_samples(spec)
            return generate_batch(spec, snapshot, model), pitch
        def complete(result):
            if (self.state is not starting_state or not self.state.is_current(snapshot)
                    or self.device_id != device_id or self.selected != selected or self.layout_generation != generation):
                self._notice(f"Discarded obsolete layout preview for revision {snapshot.revision}")
                return
            self.preview, pitch = result
            self.batch_summary = (f"Column span {pitch[0] * (spec.cols - 1)} m; row span {pitch[1] * (spec.rows - 1)} m; "
                                  f"column/row spacing {pitch} m")
            if self.batch_summary_handle is not None:
                self.batch_summary_handle.content = self.batch_summary
            self._notice(f"Generated preview at revision {snapshot.revision}; {self.batch_summary}")
            self._draw()
            self._button_preview_enabled()
        self._job("Generate layout preview", snapshot, generate, complete)

    def _button_preview_enabled(self):
        for control in self.controls.values():
            if control.kind == "action" and control.action == self._apply_batch:
                control.handle.disabled = self.preview is None

    def _apply_batch(self):
        if self.preview is None:
            raise ValueError("Generate a preview first")
        candidate = self.preview
        self.preview = None
        self.state.dispatch(candidate)
        self.temp["batch_excluded"] = "[]"
        self._after_document(rebuild=True)

    def _cancel_batch(self):
        self.layout_generation += 1
        self.preview = None
        self._draw()
        self._button_preview_enabled()

    def _reproject(self):
        from .layout import reproject_taxels

        spec = self._batch_spec()
        snapshot, starting_state = self.state.snapshot(), self.state
        device_id, selected = self.device_id, tuple(self.selected)
        model = self.model.with_config(snapshot.document.model)
        self.layout_generation += 1
        generation = self.layout_generation
        def complete(candidate):
            if (self.state is not starting_state or not self.state.is_current(snapshot) or self.device_id != device_id
                    or self.selected != selected or self.layout_generation != generation):
                self._notice(f"Discarded obsolete reprojection for revision {snapshot.revision}")
                return
            self.state.dispatch(candidate)
            self._after_document()
        self._job("Reproject taxels", snapshot, lambda: reproject_taxels(snapshot, device_id, selected, model,
                  allowed_links=spec.allowed_links, component_seeds=spec.component_seeds, depth=spec.depth,
                  flip_normal=spec.flip_normal), complete)

    def _select_indices(self):
        from .layout import select_indices

        self._select_taxels(select_indices(self.device(), self.temp["selection"]))

    def _select_taxels(self, identifiers):
        self.selected = tuple(identifiers)
        self.selected_segment = None
        self.state.view = replace(self.state.view, selected_ids=self.selected)
        self._build()

    def _rotate_poses(self, angle, axis):
        import numpy as np
        from trimesh import transformations
        from .geometry import matrix_pose, pose_matrix

        snapshot = self.state.snapshot()
        document = snapshot.document
        for identifier in self.selected:
            device = document.device(self.device_id)
            taxel = next(item for item in device.taxels if item.id == identifier)
            pose = pose_matrix(taxel.bind_pose) @ transformations.rotation_matrix(angle, np.asarray(axis))
            candidate = _pose_candidate(Snapshot(document, snapshot.revision), self.device_id, identifier,
                                        matrix_pose(pose), "taxel")
            document = candidate.document
        self.state.dispatch(Candidate(snapshot.revision, document, "Rotate selected taxel frames"))
        self._after_document(rebuild=True)

    def _flip_normals(self):
        self._rotate_poses(math.pi, (1, 0, 0))

    def _rotate_tangents(self):
        self._rotate_poses(_number(self.temp["batch_angle"]), (0, 0, 1))

    def _delete_taxels(self):
        self.state.delete_taxels(self.device_id, self.selected)
        self.selected = ()
        self._after_document(rebuild=True)

    def _preview_order(self):
        order = _array(self.temp["reorder"])
        device = self.device()
        old = {item.id: item.taxel_index for item in device.taxels}
        if set(order) != set(old) or len(order) != len(old):
            raise ValueError("The order must contain each taxel UUID exactly once")
        self._notice("Output indices: " + ", ".join(f"{old[identifier]}→{index}" for index, identifier in enumerate(order)))

    def _reorder_taxels(self):
        self._preview_order()
        self.state.reorder_taxels(self.device_id, _array(self.temp["reorder"]))
        self._after_document(rebuild=True)

    def _raw_poses(self, snapshot, value, _source):
        poses = _array(value)
        device = snapshot.document.device(self.device_id)
        taxels = tuple(replace(device.taxels[index], bind_pose=pose, surface_attachment=None,
                               generation_issue=None, provenance="manual") if index < len(device.taxels)
                       else Taxel(taxel_index=index, bind_pose=pose)
                       for index, pose in enumerate(poses))
        return self._candidate_device(snapshot, lambda device: replace(device, taxels=taxels), "Edit raw P array")

    def _skin_rows(self, snapshot, field):
        device = snapshot.document.device(self.device_id)
        rows = getattr(device.skinning, field)
        selected = set(self.selected)
        values = []
        for taxel in device.taxels:
            if taxel.id not in selected:
                continue
            index = _row_index(taxel.taxel_index, len(rows))
            if index is None:
                raise ValueError("A selected taxel has no unambiguous raw skinning row; repair the raw arrays")
            values.append((index, rows[index]))
        if len({index for index, _row in values}) != len(values):
            raise ValueError("Selected taxels share a raw row; repair taxel indices first")
        return values

    def _skin_candidate(self, snapshot, field, slot, value, source):
        from .mappings.lbs import edit_weights

        device = snapshot.document.device(self.device_id)
        rows = list(getattr(device.skinning, field))
        for index, row in self._skin_rows(snapshot, field):
            if field == "weights":
                rows[index] = edit_weights(row, slot, _number(value), source=source)
            else:
                if not isinstance(row, tuple):
                    raise ValueError("Malformed raw row; use the complete array JSON field")
                updated = list(row)
                while len(updated) <= slot:
                    updated.append(None)
                updated[slot] = _number(value)
                rows[index] = updated
        skinning = replace(device.skinning, **{field: rows})
        return _replace_device_candidate(snapshot, replace(device, skinning=skinning), "Set skinning slot")

    def _binding_page(self):
        gui = self.editor.gui
        if not self.device_id:
            return
        device = self.device()
        gui.add_markdown(f"{len(self.selected)} selected taxels. Raw slots are preserved in order.")
        try:
            indices = self._skin_rows(self.state.snapshot(), "indices")
            weights = self._skin_rows(self.state.snapshot(), "weights")
        except ValueError as exception:
            gui.add_markdown("**Raw row error:** " + str(exception))
            indices = weights = ()
        if self.selected and indices and weights:
            sums = tuple(_weight_sum(row) for _index, row in weights)
            gui.add_markdown("Weight sum: " + ("Unavailable (missing, malformed, or overflow)" if None in sums
                                              else _common(sums)[0] or "Mixed values"))
            for slot in range(4):
                with gui.add_folder(f"Slot {slot + 1}"):
                    self._group_errors(("taxel_segment", "skinning", "indices", "weights"))
                    for field, label, rows in (("indices", "Raw segment row", indices), ("weights", "Raw weight", weights)):
                        values = tuple(row[slot] if isinstance(row, tuple) and slot < len(row) else None for _index, row in rows)
                        value, hint = _common(values)
                        if hint:
                            gui.add_markdown(hint)
                        read = lambda snapshot, field=field, slot=slot: _common(tuple(
                            row[slot] if isinstance(row, tuple) and slot < len(row) else None
                            for _index, row in self._skin_rows(snapshot, field)))[0]
                        build = lambda snapshot, value, source, field=field, slot=slot: self._skin_candidate(snapshot, field, slot, value, source)
                        raw_handle = self._text(label, read, build, field="skinning." + field, number=True)
                        if field == "weights":
                            self.controls[raw_handle._impl.uuid].slot = slot
                        if field == "weights":
                            numeric = values[0] if values and all(item == values[0] for item in values) else None
                            if not all(isinstance(row, tuple) and len(row) == 4 and all(
                                _finite(item) and 0 <= item <= 1 for item in row)
                                for _index, row in weights):
                                numeric = None
                            handle = self._slider("Weight slider", numeric, 0, 1, build, field="skinning.weights",
                                                  read=lambda snapshot, slot=slot: self._slider_weight(snapshot, slot))
                            if handle:
                                self.controls[handle._impl.uuid].slot = slot
                    options = {f"{row}: {segment.semantic}": row for row, segment in enumerate(device.segments)}
                    self._choice("Choose segment", ("Keep raw slot", *options), "Keep raw slot",
                                 lambda value, slot=slot, options=options: self._choose_segment_slot(slot, options.get(value)))
        self._button("Motion preview", lambda: self._mode("PREVIEW"), disabled=self.model is None)
        self._button("Return to binding pose", lambda: self._mode("EDIT"))
        with gui.add_folder("Raw skinning draft repair"):
            gui.add_markdown("These arrays include missing, extra, and orphan rows. No row is repaired automatically.")
            for field in ("indices", "weights"):
                self._text("Complete skinning " + field + " JSON", lambda snapshot, field=field: _json(getattr(
                    snapshot.document.device(self.device_id).skinning, field)),
                    lambda snapshot, value, _source, field=field: self._raw_skinning(snapshot, field, value),
                    field="skinning." + field, multiline=True)

    def _raw_skinning(self, snapshot, field, value):
        device = snapshot.document.device(self.device_id)
        rows = _array(value)
        provenance = thaw(device.skinning.deleted_rows)
        provenance.pop(field, None)
        skinning = replace(device.skinning, **{field: rows, "deleted_rows": FrozenMap(provenance)})
        return _replace_device_candidate(snapshot, replace(device, skinning=skinning), "Edit raw skinning array")

    def _slider_weight(self, snapshot, slot):
        try:
            rows = self._skin_rows(snapshot, "weights")
            if not rows or not all(isinstance(row, tuple) and len(row) == 4 and all(_finite(value)
                    and 0 <= value <= 1 for value in row) for _index, row in rows):
                return None
            values = tuple(row[slot] for _index, row in rows)
            return values[0] if all(value == values[0] for value in values) else None
        except ValueError:
            return None

    def _choose_segment_slot(self, slot, row):
        if row is None:
            return
        candidate = self._skin_candidate(self.state.snapshot(), "indices", slot, row, "number")
        device = candidate.document.device(self.device_id)
        weights = list(device.skinning.weights)
        for index, indices in self._skin_rows(Snapshot(candidate.document, candidate.base_revision), "indices"):
            if (isinstance(indices, tuple) and len(indices) == 4 and all(_row_index(item, len(device.segments)) is not None
                    for item in indices) and index < len(weights)):
                weights[index] = (0.25, 0.25, 0.25, 0.25)
        candidate = replace(candidate, document=replace_device(candidate.document,
                            replace(device, skinning=replace(device.skinning, weights=weights))))
        self.state.dispatch(candidate)
        self._after_document(rebuild=True)

    def _realtime_page(self):
        gui = self.editor.gui
        gui.add_markdown("**SYNTHETIC input (--demo)**" if self.demo else "**Real Isaac Capture input**")
        gui.add_markdown("Read-only receiver. Times describe local getter reads, not producer timestamps or hardware synchronization.")
        self.live_status_handle = gui.add_markdown("")
        if not self.device_id:
            return
        device = self.device()
        with gui.add_folder("Sources for this side"):
            self._group_errors(("application_binding",))
            if not isinstance(device.application_binding, FrozenMap):
                gui.add_markdown("No usable source object. Initialize the default collection 'tactile' or repair the raw source JSON.")
                self._button("Initialize source settings", self._initialize_sources, disabled=device.side not in ("left", "right"))
            else:
                self._binding_text("Tactile collection ID", ("tactile_collection_id",))
                hand = device.application_binding.get("hand_source")
                if isinstance(hand, FrozenMap):
                    self._choice("HandTracker output", ("left", "right"), str(hand.get("side")),
                                 lambda value: self._commit_binding(("hand_source", "side"), value))
                    self._binding_text("HandTracker vendor ID (empty uses default)", ("hand_source", "vendor_id"), empty_none=True)
                    self._binding_text("Vendor parameters JSON", ("hand_source", "vendor_params"), json_value=True)
                    gui.add_markdown("Both sides share one HandTracker vendor configuration; configure identical vendor settings.")
                self._checkbox("Enable JointState URDF input", device.application_binding.get("joint_source") is not None,
                               self._enable_joints)
                if isinstance(device.application_binding.get("joint_source"), FrozenMap):
                    self._binding_text("JointState collection ID", ("joint_source", "collection_id"))
                    self._binding_text("JointState device ID", ("joint_source", "device_id"))
        with gui.add_folder("Named URDF joint mapping"):
            gui.add_markdown("Only valid named JointState mappings drive live URDF. HandPose supplies skeleton and LBS only.")
            rows = device.application_binding.get("joint_mapping", ()) if isinstance(device.application_binding, FrozenMap) else ()
            if isinstance(rows, tuple):
                for index, row in enumerate(rows):
                    if not isinstance(row, FrozenMap):
                        gui.add_markdown(f"Mapping row {index} is malformed; use the raw JSON repair field.")
                        continue
                    with gui.add_folder(f"Mapping row {index}"):
                        if self.model:
                            self._choice("URDF free joint", self.model.actuated_joint_names or ("",), str(row.get("urdf_joint", "")),
                                         lambda value, index=index: self._commit_binding(("joint_mapping", index, "urdf_joint"), value))
                        else:
                            self._binding_text("URDF joint name", ("joint_mapping", index, "urdf_joint"))
                        self._binding_text("Source joint name", ("joint_mapping", index, "source_joint"))
                        self._binding_text("Scale", ("joint_mapping", index, "scale"), number=True)
                        self._binding_text("Offset rad or m", ("joint_mapping", index, "offset"), number=True)
                        self._button("Delete mapping row", lambda index=index: self._delete_joint_mapping(index))
                self._button("Add named mapping", self._add_joint_mapping, disabled=not isinstance(device.application_binding, FrozenMap))
            self.live_mapping_handle = gui.add_markdown("")
        with gui.add_folder("Connection and playback"):
            self._button("Connect synthetic sources" if self.demo else "Connect real sources", self._start_receiving)
            self._button("Disconnect", self._stop_receiving)
            self._button("Play latest frame", lambda: self._mode("LIVE"))
            self._button("Pause displayed live layer", lambda: self._mode("PAUSED"))
            self._button("Return to q0 editing", lambda: self._mode("EDIT"))
            gui.add_markdown("The wireframe model is a q0 editing reference. Live solid URDF requires enabled, valid JointState input.")
            self._temp_text("Force color upper limit N", "force_max", number=True)
            self._button("Apply force color limit", self._force_limit)
            for name in METADATA_GROUPS:
                self._checkbox("Record " + name, self.record_groups[name],
                               lambda value, name=name: self.record_groups.__setitem__(name, value))
            self._button("Record received metadata into draft", self._record_metadata)
        with gui.add_folder("Explicit source taxel correspondence"):
            gui.add_markdown("A matching count does not establish correspondence. Rows bind UUID, input row, collection, side, and metadata fingerprint.")
            self._temp_text("First input row for selected taxels", "source_start", number=True)
            self._button("Bind selected taxels to current baseline", self._bind_selected_source, disabled=not self.selected)
            for taxel in device.taxels[self.pages["Layout"] * PAGE_SIZE:(self.pages["Layout"] + 1) * PAGE_SIZE]:
                descriptor = self._source_descriptor(device, taxel.id)
                gui.add_markdown(f"Taxel {taxel.taxel_index} · `{taxel.id}`\n\nDeclared source: `{_json(descriptor)}`")
                if isinstance(descriptor, FrozenMap):
                    def read_row(snapshot, identifier=taxel.id):
                        current = self._source_descriptor(snapshot.document.device(self.device_id), identifier)
                        return _json(current.get("input_taxel_index") if isinstance(current, FrozenMap) else None)
                    self._text("Input row for " + str(taxel.taxel_index), read_row,
                        lambda snapshot, value, _source, identifier=taxel.id: self._source_row_candidate(snapshot, identifier, _number(value)),
                        number=True, field="application_binding.source_taxel_map", entity=taxel.id)
                self._button("Clear source for taxel " + str(taxel.taxel_index), lambda identifier=taxel.id: self._clear_source(identifier))
            self._pagination(len(device.taxels), "Layout")
            self.live_forces_handle = gui.add_markdown("")
        with gui.add_folder("Raw receiving draft repair"):
            self._text("Application source binding JSON", lambda snapshot: _json(snapshot.document.device(self.device_id).application_binding),
                       lambda snapshot, value, _source: self._candidate_device(snapshot, lambda device: replace(
                           device, application_binding=_parse_json(value)), "Set application source binding"),
                       field="application_binding", multiline=True)

    def _binding_get(self, snapshot, path):
        value = snapshot.document.device(self.device_id).application_binding
        for key in path:
            if not isinstance(value, (FrozenMap, tuple)):
                return None
            value = value.get(key) if isinstance(value, FrozenMap) else value[key]
        return value

    def _binding_candidate(self, snapshot, path, value):
        device = snapshot.document.device(self.device_id)
        binding = thaw(device.application_binding)
        if not isinstance(binding, dict):
            raise ValueError("Repair or initialize the source settings first")
        target = binding
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        return _replace_device_candidate(snapshot, replace(device, application_binding=binding), "Set receiving source")

    def _binding_text(self, label, path, *, number=False, json_value=False, empty_none=False):
        def read(snapshot):
            value = self._binding_get(snapshot, path)
            return _json(value) if number or json_value else "" if value is None else str(value)
        def build(snapshot, value, _source):
            parsed = _number(value) if number else _parse_json(value) if json_value else None if empty_none and value == "" else value
            return self._binding_candidate(snapshot, path, parsed)
        self._text(label, read, build, number=number, multiline=json_value, field="application_binding." + str(path[0]))

    def _commit_binding(self, path, value):
        self.state.dispatch(self._binding_candidate(self.state.snapshot(), path, value))
        self._after_document(rebuild=True)

    def _initialize_sources(self):
        side = self.device().side
        if side not in ("left", "right"):
            raise ValueError("Assign the metadata side before configuring sources")
        self.state.edit_device(self.device_id, lambda device: replace(device, taxels=tuple(
            replace(taxel, input_ref=None) for taxel in device.taxels), application_binding={
            "tactile_collection_id": "tactile", "hand_source": {"side": side, "vendor_id": None, "vendor_params": {}},
            "joint_source": None, "joint_mapping": [], "source_taxel_map": [],
        }), "Initialize receiving sources")
        self._after_document(rebuild=True)

    def _enable_joints(self, enabled):
        self._commit_binding(("joint_source",), {"collection_id": "joint_state", "device_id": ""} if enabled else None)

    def _add_joint_mapping(self):
        rows = thaw(self.device().application_binding.get("joint_mapping", ()))
        if not isinstance(rows, list):
            raise ValueError("Repair the joint mapping array first")
        rows.append({"urdf_joint": self.model.actuated_joint_names[0] if self.model and self.model.actuated_joint_names else "",
                     "source_joint": "", "scale": 1, "offset": 0})
        self._commit_binding(("joint_mapping",), rows)

    def _delete_joint_mapping(self, index):
        rows = thaw(self.device().application_binding["joint_mapping"])
        del rows[index]
        self._commit_binding(("joint_mapping",), rows)

    @staticmethod
    def _source_descriptor(device, identifier):
        taxel = next(item for item in device.taxels if item.id == identifier)
        if taxel.input_ref is not None:
            return taxel.input_ref
        binding = device.application_binding
        rows = binding.get("source_taxel_map", ()) if isinstance(binding, FrozenMap) else ()
        if isinstance(rows, tuple):
            return next((row for row in rows if isinstance(row, FrozenMap) and row.get("taxel_id") == identifier), None)
        return None

    def _baseline(self):
        binding = self.device().application_binding
        collection = binding.get("tactile_collection_id") if isinstance(binding, FrozenMap) else None
        if (self.live_frame is None or self.live_connection is None or self.live_frame.epoch != self.live_connection.epoch
                or self.live_connection.status.state != "connected" or collection not in self.live_frame.metadata):
            raise ValueError("No current metadata baseline from this side's configured collection")
        return collection, self.live_frame.metadata[collection]

    def _source_row_candidate(self, snapshot, identifier, index, *, baseline=None):
        device = snapshot.document.device(self.device_id)
        descriptor = self._source_descriptor(device, identifier)
        if baseline is not None:
            collection, record = baseline
            descriptor = FrozenMap({"collection_id": collection, "side": device.side,
                                    "metadata_fingerprint": record["fingerprint"]})
        if index is not None and not isinstance(descriptor, FrozenMap):
            raise ValueError("Bind this taxel to a current baseline first")
        binding = thaw(device.application_binding)
        if index is None and (not isinstance(binding, dict) or not isinstance(binding.get("source_taxel_map", []), list)):
            return _replace_device_candidate(snapshot, replace(device, taxels=tuple(
                replace(taxel, input_ref=None) if taxel.id == identifier else taxel for taxel in device.taxels)), "Clear source reference")
        if not isinstance(binding, dict) or not isinstance(binding.get("source_taxel_map", []), list):
            raise ValueError("Repair the source table JSON first")
        rows = [row for row in binding.get("source_taxel_map", []) if not isinstance(row, dict) or row.get("taxel_id") != identifier]
        if index is not None:
            row = thaw(descriptor)
            row.update(taxel_id=identifier, input_taxel_index=index)
            rows.append(row)
        binding["source_taxel_map"] = rows
        device = replace(device, application_binding=binding, taxels=tuple(
            replace(taxel, input_ref=None) if taxel.id == identifier else taxel for taxel in device.taxels))
        return _replace_device_candidate(snapshot, device, "Set explicit source taxel row")

    def _bind_selected_source(self):
        baseline = self._baseline()
        start = _row_index(_number(self.temp["source_start"]))
        if start is None:
            raise ValueError("The first input row must be a nonnegative integer")
        snapshot = self.state.snapshot()
        current = snapshot
        for offset, identifier in enumerate(self.selected):
            candidate = self._source_row_candidate(current, identifier, start + offset, baseline=baseline)
            current = Snapshot(candidate.document, snapshot.revision)
        self.state.dispatch(Candidate(snapshot.revision, current.document, "Bind selected source rows"))
        self._after_document(rebuild=True)

    def _clear_source(self, identifier):
        self.state.dispatch(self._source_row_candidate(self.state.snapshot(), identifier, None))
        self._after_document(rebuild=True)

    def _record_metadata(self):
        from .domain import MetadataSource

        collection, baseline = self._baseline()
        groups = tuple(name for name in METADATA_GROUPS if self.record_groups[name])
        result = self.state.record_metadata(self.device_id, baseline["values"],
            source=MetadataSource(collection, self.device().side, baseline["fingerprint"]), groups=groups, backup_directory="/tmp")
        for path in result.backup_paths:
            self._notice(f"Draft backup: {path}")
        self.selected = ()
        self.selected_segment = None
        self._after_document(rebuild=True)

    def _receiver_key(self):
        fields = ("tactile_collection_id", "hand_source", "joint_source", "joint_mapping")
        return _json(tuple((device.id, device.side, FrozenMap({name: device.application_binding.get(name) for name in fields})
                            if isinstance(device.application_binding, FrozenMap) else device.application_binding)
                           for device in self.state.snapshot().document.devices))

    def _start_receiving(self):
        from .document import raw_values
        from .live import ConnectConfig

        self.live_wanted = True
        self.receive_key = self._receiver_key()
        self.live_generation += 1
        generation = self.live_generation
        self.live_connection = None
        self.live_poll_key = None
        if self.live_lifecycle is None:
            self.live_lifecycle = _LiveLifecycle(self.inbox)
        try:
            devices = self.state.snapshot().document.devices
            sides = tuple(device.side for device in devices)
            if any(side not in ("left", "right") for side in sides) or len(set(sides)) != len(sides):
                raise ValueError("Receiving requires one explicitly assigned metadata device per configured side")
            bindings = {device.side: device.application_binding for device in devices}
            demo_values = {device.application_binding["tactile_collection_id"]: raw_values(device, output_order=True)
                           for device in devices} if self.demo else {}
            config = ConnectConfig(bindings, demo=self.demo, demo_values=demo_values)
        except Exception as exception:
            self.live_lifecycle.request(generation, None)
            self.live_status = ("error", "", (Issue("error", "live_config", str(exception), field="application_binding"),), self.demo)
            raise
        self.live_status = ("connecting", "", (), self.demo)
        self.live_lifecycle.request(generation, config)
        self._notice("Connecting synthetic sources" if self.demo else "Connecting real Isaac Capture sources")

    def _sync_receiving(self):
        if self.live_wanted and self.receive_key != self._receiver_key():
            try:
                self._start_receiving()
            except (ValueError, TypeError, KeyError) as exception:
                self._notice(f"Receiving configuration is invalid: {exception}")

    def _stop_receiving(self, *, clear=False):
        self.live_wanted = False
        self.live_generation += 1
        self.live_connection = None
        self.live_status = ("disconnected", "", (), self.demo)
        self.live_poll_key = None
        if self.live_lifecycle:
            self.live_lifecycle.request(self.live_generation, None)
        if clear:
            self.live_frame = self.live_layer = self.paused_layer = None
            self.received_cache.clear()

    def close(self):
        self.stopping = True
        try:
            self._stop_receiving(clear=True)
            if self.live_lifecycle:
                self.live_lifecycle.close()
                self.live_lifecycle = None
        finally:
            # SDK timeout/reaping and export staging cleanup finish before exit.
            for thread in tuple(self.job_threads.values()):
                thread.join()

    def _force_limit(self):
        value = _number(self.temp["force_max"])
        if not _finite(value) or value <= 0:
            raise ValueError("Force color upper limit must be a positive finite number")
        self.force_max = value
        self.live_render_key = None
        self.live_render_error = None
        self._draw()

    def _refresh_live_status(self):
        state, epoch, issues, synthetic = self.live_status
        frame = self.live_frame
        if self.live_status_handle is not None:
            lines = [f"**{'SYNTHETIC' if synthetic else 'REAL'} · {state}**", f"Receiving epoch: `{epoch or 'none'}`"]
            if frame:
                lines.append(f"Last local read: {frame.read_monotonic_ns} ns · sequence {frame.seq}")
                lines.extend(f"- metadata/{name}: fingerprint `{baseline['fingerprint']}` · "
                             f"{'synthetic' if baseline.get('synthetic') else 'real'}"
                             for name, baseline in frame.metadata.items())
                for group, sources in frame.sources.items():
                    lines.extend(f"- {group}/{name}: {'valid' if source.get('valid') else 'unavailable'} · {source.get('reason', '')}"
                                 for name, source in sources.items())
            lines.extend("- " + issue.message for issue in issues)
            if self.state.view.mode == "PAUSED":
                layer = self.paused_layer
                lines.append(f"Paused complete layer: revision {layer.snapshot.revision}, sequence {layer.seq}, source {layer.source}."
                             if layer else "Paused before a complete live layer was displayed.")
            self.live_status_handle.content = "\n\n".join(lines)
        layer = self.paused_layer if self.state.view.mode == "PAUSED" else self.live_layer
        side = next((item for item in layer.sides if item.device_id == self.device_id), None) if layer else None
        if self.live_mapping_handle is not None:
            if side:
                current = side.model_current if self.state.view.mode == "PAUSED" else self._current_model_layer(layer, side)
                self.live_mapping_handle.content = (f"{'Frozen' if self.state.view.mode == 'PAUSED' else 'Available' if current else 'Unavailable/historical'} "
                    f"mapping · layer sequence {layer.seq}, local read {layer.read_monotonic_ns} ns.\n\n"
                    + "\n".join(_json(row) for row in side.joint_rows))
            else:
                self.live_mapping_handle.content = "No live JointState mapping result."
        if self.live_forces_handle is not None:
            current = self.state.view.mode != "PAUSED" and self._current_force_layer(layer, side)
            lines = ["Frozen forces from the paused layer." if self.state.view.mode == "PAUSED" else
                     "Available tactile forces (N) from the layer snapshot below." if current else
                     "No current force validity; the last geometry may remain visible."]
            geometry = {}
            if layer and side:
                lines.append(f"Layer snapshot: epoch `{layer.epoch}`, sequence {layer.seq}, local read {layer.read_monotonic_ns} ns, "
                             f"draft revision {layer.snapshot.revision}, 3D source {layer.source}.")
                hand = self.live_frame.sources.get("hands", {}).get(side.side) if self.live_frame else None
                geometry, _joints = _live_geometry_masks(side, hand, frozen=self.state.view.mode == "PAUSED")
            for identifier in self.selected:
                force = side.forces.get(identifier) if side else None
                if force is not None and (current or self.state.view.mode == "PAUSED"):
                    lines.append(f"`{identifier}`: fx={force[0]}, fy={force[1]}, fz={force[2]}, |F|={_force_magnitude(force)} N")
                    lines.append("Displayed geometry has valid Hand dependencies." if geometry.get(identifier, False) else
                                 "No current axis arrow: this geometry is unavailable or is not part of the selected 3D source.")
                else:
                    lines.append(f"`{identifier}`: unavailable")
            self.live_forces_handle.content = "\n\n".join(lines)

    def _current_layer(self, layer):
        return (layer is not None and self.live_status[0] == "connected" and self.live_connection is not None
                and layer.epoch == self.live_status[1] == self.live_connection.epoch
                and layer.snapshot.revision == self.state.revision
                and layer.snapshot.document is self.state.snapshot().document
                and layer.source == self.state.view.metadata_source
                and self.live_render_error != (id(self.state), self.state.revision, layer.epoch, layer.source))

    def _current_force_layer(self, layer, side):
        if not self._current_layer(layer) or side is None or self.live_frame is None:
            return False
        binding = layer.snapshot.document.device(side.device_id).application_binding
        collection = binding.get("tactile_collection_id") if isinstance(binding, FrozenMap) else None
        baseline = self.live_frame.metadata.get(collection)
        tactile = self.live_frame.sources.get("tactile", {}).get(collection)
        return (self.live_frame.epoch == layer.epoch and isinstance(tactile, FrozenMap) and tactile.get("valid") is True
                and baseline is not None and baseline.get("fingerprint") == side.fingerprint)

    def _current_model_layer(self, layer, side):
        from .drivers import map_joints

        if not self._current_layer(layer) or not side.model_current or self.model is None or self.live_frame is None:
            return False
        if self.live_frame.epoch != layer.epoch:
            return False
        binding = layer.snapshot.document.device(side.device_id).application_binding
        source = binding.get("joint_source") if isinstance(binding, FrozenMap) else None
        sample = self.live_frame.sources.get("joints", {}).get(source.get("collection_id")) if isinstance(source, FrozenMap) else None
        return map_joints(binding, sample, self.model, layer.snapshot.document.bind_configuration).valid

    def _display_live_side(self, layer, side):
        hand = self.live_frame.sources.get("hands", {}).get(side.side) if self.live_frame else None
        taxel_current, joint_current = _live_geometry_masks(side, hand)
        current = self._current_layer(layer)
        force_current = current and self._current_force_layer(layer, side)
        taxel_valid = tuple(current and taxel_current.get(identifier, False) for identifier in side.taxel_ids)
        joint_valid = tuple(current and joint_current.get(name, False) for name in side.joint_names)
        taxels, joints = dict(zip(side.taxel_ids, taxel_valid)), dict(zip(side.joint_names, joint_valid))
        colors = tuple(color if valid and force_current else (125, 125, 130) for color, valid in zip(side.colors, taxel_valid))
        line_colors = tuple(color if all(joints.get(name, False) for name in edge) else ((125, 125, 130),) * 2
                            for edge, color in zip(side.line_joints, side.line_colors))
        force_rows = tuple((owner, line, color) for owner, line, color in zip(side.force_owners, side.force_lines, side.force_colors)
                           if force_current and taxels.get(owner, False))
        binding = self.state.snapshot().document.device(side.device_id).application_binding
        enabled = isinstance(binding, FrozenMap) and binding.get("joint_source") is not None
        return replace(side, taxel_valid=taxel_valid, joint_valid=joint_valid, colors=colors, line_colors=line_colors,
                       force_owners=tuple(row[0] for row in force_rows), force_lines=tuple(row[1] for row in force_rows),
                       force_colors=tuple(row[2] for row in force_rows), model_current=enabled and self._current_model_layer(layer, side),
                       meshes=side.meshes if enabled else ())

    def poll_live(self):
        now = time.monotonic()
        if now - self._last_live_poll < 1 / 30:
            return
        self._last_live_poll = now
        connection = self.live_connection
        if connection is None:
            return
        status = connection.status
        frame = connection.latest()
        key = (status.state, status.epoch, status.synthetic,
               tuple((issue.code, issue.message) for issue in status.issues),
               None if frame is None else (frame.epoch, frame.seq, frame.read_monotonic_ns))
        if key != self.live_poll_key:
            self.live_poll_key = key
            self.live_status = (status.state, status.epoch, status.issues, status.synthetic)
            if frame is not None and frame.epoch == connection.epoch:
                self.live_frame = frame
            self._draw()
            self._refresh_live_status()
            self._refresh_info()
        if self.state.view.mode == "LIVE":
            self._schedule_live_layer()

    def _received_device(self, device, frame):
        from .document import import_values

        binding = device.application_binding
        collection = binding.get("tactile_collection_id") if isinstance(binding, FrozenMap) else None
        baseline = frame.metadata.get(collection)
        if baseline is None:
            return None
        key = (frame.epoch, collection, baseline["fingerprint"])
        cached = self.received_cache.get(device.side)
        if cached is not None and cached[:3] == key:
            return cached[3]
        received = import_values(baseline["values"], side=device.side).devices[0]
        rows = tuple({"taxel_id": taxel.id, "input_taxel_index": index, "collection_id": collection,
                      "side": device.side, "metadata_fingerprint": baseline["fingerprint"]}
                     for index, taxel in enumerate(received.taxels))
        values = thaw(binding)
        values["source_taxel_map"] = rows
        received = replace(received, application_binding=values)
        self.received_cache[device.side] = (*key, received)
        return received

    def _schedule_live_layer(self):
        frame = self.live_frame
        if (self.live_render_pending or self.live_status[0] != "connected" or frame is None
                or self.live_connection is None or frame.epoch != self.live_connection.epoch or len(self.jobs) >= 2):
            return
        snapshot, starting_state = self.state.snapshot(), self.state
        source, cap = self.state.view.metadata_source, self.force_max
        error_key = (id(starting_state), snapshot.revision, frame.epoch, source)
        key = (*error_key, frame.seq, cap)
        if self.live_render_key == key or self.live_render_error == error_key:
            return
        model = self.model.with_config(snapshot.document.model) if self.model else None
        status = self.live_status
        previous = self.live_layer
        self.live_render_key = key
        self.live_render_pending = True
        def complete(layer):
            self.live_render_pending = False
            if (self.state is not starting_state or not self.state.is_current(snapshot) or self.state.view.mode != "LIVE"
                    or self.state.view.metadata_source != source or self.live_connection is None
                    or self.live_connection.epoch != frame.epoch or self.live_status != status):
                self.live_render_key = None
                return
            self.live_layer = layer
            self._draw()
            self._refresh_live_status()
        self._job("Prepare live layer", snapshot,
                  lambda: self._prepare_live_layer(snapshot, frame, model, source, cap, status[3], previous), complete)

    def _prepare_live_layer(self, snapshot, frame, model, source, cap, synthetic, previous=None):
        import numpy as np
        from .drivers import current_segments, map_joints, source_correspondence, view_root
        from .geometry import matrix_pose, pose_matrix
        from .mappings.lbs import evaluate
        from .profiles.hand import JOINT_NAMES

        sides, issues = [], []
        chains = (("WRIST", "THUMB_METACARPAL", "THUMB_PROXIMAL", "THUMB_DISTAL", "THUMB_TIP"),
                  *(("WRIST", f"{finger}_METACARPAL", f"{finger}_PROXIMAL", f"{finger}_INTERMEDIATE",
                     f"{finger}_DISTAL", f"{finger}_TIP") for finger in ("INDEX", "MIDDLE", "RING", "LITTLE")),
                  ("WRIST", "PALM"))
        for side_index, device in enumerate(snapshot.document.devices):
            if device.side not in ("left", "right"):
                issues.append(Issue("error", "live_side", "Assign the metadata side before live display", device.id, field="side"))
                continue
            offset = np.asarray((0.35 * side_index, 0, 0))
            binding = device.application_binding
            collection = binding.get("tactile_collection_id") if isinstance(binding, FrozenMap) else None
            baseline = frame.metadata.get(collection)
            previous_side = next((item for item in previous.sides if item.device_id == device.id), None) if previous else None
            history = (previous_side if previous is not None and previous.epoch == frame.epoch and previous.source == source
                       and previous_side is not None and previous_side.fingerprint == (baseline["fingerprint"] if baseline else None) else None)
            hands = frame.sources.get("hands", {}).get(device.side)
            skeleton = current_segments(JOINT_NAMES, hands)
            issues.extend(replace(issue, device_id=device.id) for issue in skeleton.issues)
            joint_names, joint_positions, joint_valid = _retain_live_points(JOINT_NAMES,
                tuple(tuple(pose_matrix(pose)[:3, 3] + offset) if valid and pose is not None else None
                      for pose, valid in zip(skeleton.poses, skeleton.valid)), skeleton.valid,
                dict(zip(history.joint_names, history.joints)) if history else {})
            joint_points = dict(zip(joint_names, joint_positions))
            joint_available = dict(zip(joint_names, joint_valid))
            lines, line_colors, line_joints, force_lines, force_colors, force_owners = [], [], [], [], [], []
            for chain in chains:
                for first, second in zip(chain, chain[1:]):
                    if first in joint_points and second in joint_points:
                        lines.append((joint_points[first], joint_points[second]))
                        line_joints.append((first, second))
                        line_colors.append(((100, 180, 210) if joint_available[first] and joint_available[second]
                                            else (125, 125, 130),) * 2)
            draft_correspondence = source_correspondence(device, frame, expected_epoch=frame.epoch,
                                                       local_values=_source_values(device))
            issues.extend(draft_correspondence.issues)
            if draft_correspondence.automatic:
                issues.append(Issue("info", "source_automatic", "Temporary row correspondence is verified by complete canonical metadata equality; it is not persisted",
                                    device.id, field="application_binding.source_taxel_map"))
            view_device = self._received_device(device, frame) if source == "received" else device
            positions, colors, taxel_ids, meshes, joint_rows = (), [], (), [], ()
            taxel_valid, dependencies = (), {}
            if view_device is not None:
                correspondence = (source_correspondence(view_device, frame, expected_epoch=frame.epoch,
                                                       local_values=_source_values(view_device))
                                  if source == "received" else draft_correspondence)
                if source == "received":
                    issues.extend(replace(issue, device_id=device.id, entity_id=None) for issue in correspondence.issues)
                current = current_segments(tuple(segment.semantic for segment in view_device.segments), hands)
                issues.extend(replace(issue, device_id=device.id) for issue in current.issues)
                row_count = min(len(view_device.skinning.indices), len(view_device.skinning.weights))
                rows = tuple(_row_index(taxel.taxel_index, row_count) for taxel in view_device.taxels)
                result = evaluate(view_device.segment_bind_poses, tuple(taxel.bind_pose for taxel in view_device.taxels),
                                  tuple(view_device.skinning.indices[row] if row is not None else None for row in rows),
                                  tuple(view_device.skinning.weights[row] if row is not None else None for row in rows),
                                  current.poses, current_valid=current.valid)
                for issue in result.issues:
                    row = _row_index(int(issue.entity_id), len(view_device.taxels)) if str(issue.entity_id).isdecimal() else None
                    issues.append(replace(issue, device_id=device.id,
                        entity_id=view_device.taxels[row].id if row is not None and source == "draft" else None))
                dependencies = {taxel.id: _taxel_hand_dependencies(view_device, taxel) for taxel in view_device.taxels}
                taxel_ids, positions, taxel_valid = _retain_live_points(tuple(taxel.id for taxel in view_device.taxels),
                    tuple(tuple(result.positions[row] + offset) if valid else None for row, valid in enumerate(result.valid)),
                    result.valid, dict(zip(history.taxel_ids, history.positions)) if history else {})
                taxel_rows = {taxel.id: row for row, taxel in enumerate(view_device.taxels)}
                for identifier, position, valid in zip(taxel_ids, positions, taxel_valid):
                    point = np.asarray(position)
                    force = correspondence.forces.get(identifier) if valid else None
                    magnitude = _force_magnitude(force)
                    color = (130, 130, 135) if valid else (125, 125, 130)
                    if magnitude is not None:
                        ratio = min(1.0, magnitude / cap)
                        color = (int(255 * ratio), 40, int(255 * (1 - ratio)))
                        rigid = result.rigid_poses[taxel_rows[identifier]]
                        if rigid is not None and magnitude > 0:
                            direction = pose_matrix(rigid)[:3, :3] @ (np.asarray(force) / magnitude)
                            end = point + direction * (0.025 * ratio)
                            force_lines.append((tuple(point), tuple(end)))
                            force_colors.append((color, color))
                            force_owners.append(identifier)
                            tangent = np.cross(direction, (0, 0, 1))
                            if np.linalg.norm(tangent) < 1e-6:
                                tangent = np.cross(direction, (0, 1, 0))
                            tangent /= np.linalg.norm(tangent)
                            for sign in (-1, 1):
                                tip = end - direction * 0.004 + sign * tangent * 0.002
                                force_lines.append((tuple(end), tuple(tip)))
                                force_colors.append((color, color))
                                force_owners.append(identifier)
                    colors.append(color)
            model_current = False
            if model is not None and isinstance(binding, FrozenMap) and binding.get("joint_source") is not None:
                joint_source = binding["joint_source"]
                sample = frame.sources.get("joints", {}).get(joint_source.get("collection_id")) if isinstance(joint_source, FrozenMap) else None
                mapping = map_joints(binding, sample, model, snapshot.document.bind_configuration)
                joint_rows = mapping.rows
                issues.extend(replace(issue, device_id=device.id) for issue in mapping.issues)
                if mapping.valid:
                    try:
                        roots = view_root(device, model, mapping.q, live=True, side_offset=tuple(offset))
                        if roots.model_to_view is not None:
                            meshes = tuple((instance.geometry, matrix_pose(roots.model_to_view @ instance.pose))
                                           for instance in model.surface_snapshot(mapping.q).instances)
                            model_current = True
                    except (ValueError, TypeError, KeyError, OverflowError) as exception:
                        issues.append(Issue("warning", "live_model", str(exception), device.id, field="joint_source"))
                if not model_current and previous_side is not None:
                    meshes = previous_side.meshes
                    if meshes:
                        issues.append(Issue("warning", "live_model_history", "The last valid JointState URDF pose is retained as gray, unavailable history",
                                            device.id, field="joint_source"))
            sides.append(_LiveSide(device.id, device.side, tuple(taxel_ids), tuple(positions), tuple(colors),
                                   joint_positions, tuple(lines), tuple(line_colors), tuple(force_lines),
                                   tuple(force_colors), tuple(meshes), draft_correspondence.forces, tuple(joint_rows),
                                   model_current, baseline["fingerprint"] if baseline else None, taxel_valid, FrozenMap(dependencies),
                                   joint_names, joint_valid, tuple(line_joints), tuple(force_owners)))
        return _LiveLayer(snapshot, frame.epoch, frame.seq, frame.read_monotonic_ns, source, synthetic, tuple(sides), tuple(issues))

    def _render_live_layer(self):
        import numpy as np

        mode = self.state.view.mode
        layer = self.paused_layer if mode == "PAUSED" else self.live_layer if mode == "LIVE" else None
        used = set()
        if layer is not None:
            for stored_side in layer.sides:
                if self.temp["visibility"] == "Current device" and stored_side.device_id != self.device_id:
                    continue
                side = stored_side if mode == "PAUSED" else self._display_live_side(layer, stored_side)
                joint_colors = tuple((100, 180, 210) if valid else (125, 125, 130) for valid in side.joint_valid)
                for kind, points, colors, radius in (("taxels", side.positions, side.colors, 0.0025),
                                                     ("joints", side.joints, joint_colors, 0.003)):
                    if not points:
                        continue
                    key = (side.side, kind)
                    used.add(key)
                    handle = self.live_handles.get(key)
                    if handle is None or len(handle.batched_positions) != len(points):
                        if handle:
                            handle.remove()
                        handle = self.server.scene.add_batched_meshes_simple(f"/live/{side.side}/{kind}", vertices=self._sphere.vertices,
                            faces=self._sphere.faces, batched_positions=np.asarray(points),
                            batched_wxyzs=np.tile((1, 0, 0, 0), (len(points), 1)), batched_scales=np.full(len(points), radius),
                            batched_colors=np.asarray(colors, dtype=np.uint8), lod="off")
                        self.live_handles[key] = handle
                    else:
                        handle.batched_positions = np.asarray(points)
                        handle.batched_colors = np.asarray(colors, dtype=np.uint8)
                for kind, lines, colors in (("skeleton", side.lines, side.line_colors), ("forces", side.force_lines, side.force_colors)):
                    if not lines:
                        continue
                    key = (side.side, kind)
                    used.add(key)
                    if key in self.live_handles:
                        self.live_handles[key].remove()
                    self.live_handles[key] = self.server.scene.add_line_segments(f"/live/{side.side}/{kind}", points=np.asarray(lines),
                        colors=np.asarray(colors, dtype=np.uint8), thickness=2, thickness_units="screen")
                if side.meshes:
                    for index, (geometry, pose) in enumerate(side.meshes):
                        key = (side.side, "model", geometry.key)
                        used.add(key)
                        handle = self.live_handles.get(key)
                        if handle is None:
                            handle = self.server.scene.add_mesh_simple(f"/live/{side.side}/urdf/{index}", vertices=geometry.vertices,
                                faces=geometry.faces, color=(175, 185, 190), opacity=0.65)
                            self.live_handles[key] = handle
                        handle.position, handle.wxyz = pose[:3], (pose[6], *pose[3:6])
                        handle.visible = self.temp["show_model"] and geometry.link not in self.hidden_links
                        unavailable = not side.model_current
                        handle.color = (125, 125, 130) if unavailable else (175, 185, 190)
                        handle.opacity = min(0.35, self.temp["opacity"]) if unavailable else self.temp["opacity"]
        for key in tuple(self.live_handles):
            if key not in used:
                self.live_handles.pop(key).remove()
        return () if layer is None else layer.issues

    def _export_page(self):
        gui = self.editor.gui
        self._choice("Devices", ("Current device", "All devices"), self.temp["scope"], lambda value: self.temp.__setitem__("scope", value))
        self._temp_text("New output directory", "out")
        self._choice("Package format", ("json", "sdk"), self.temp["format"], lambda value: self.temp.__setitem__("format", value))
        self._button("Local check", lambda: self._check(False), disabled=not self.device_id)
        self._button("SDK check", lambda: self._check(True), disabled=not self.device_id)
        self._button("Export snapshot", self._export, disabled=not self.device_id)
        self.check_report = gui.add_markdown("")
        gui.add_markdown("Exports and checks use their starting revision. Continue editing while a job runs.")

    def _scope(self, snapshot):
        return tuple(item.id for item in snapshot.document.devices) if self.temp["scope"] == "All devices" else (self.device_id,)

    def _job(self, label, snapshot, function, complete):
        if self.stopping:
            raise ValueError("The editor is closing")
        if len(self.jobs) >= 2:
            raise ValueError("Two background jobs are running; wait for a result before starting another")
        identity = secrets.token_hex(8)
        self.jobs.add(identity)
        if label != "Prepare live layer":
            self._notice(f"{label} started for revision {snapshot.revision}")
        def work():
            try:
                result, error = function(), None
            except Exception as exception:
                result, error = None, exception
            self.inbox.put(("job", identity, snapshot, label, result, error, complete))
        thread = threading.Thread(target=work, name="tactile-editor-job", daemon=False)
        self.job_threads[identity] = thread
        thread.start()

    def _check(self, use_sdk):
        from .checks import check_device

        snapshot = self.state.snapshot()
        starting_state = self.state
        identifiers = self._scope(snapshot)
        model = self.model
        def complete(results):
            if self.state is starting_state and self.state.is_current(snapshot):
                self.check_results.update({result.device_id: result for result in results})
            else:
                self._notice(f"Check result for revision {snapshot.revision} is obsolete")
            from .checks import check_summary
            for result in results:
                self._notice(_json(check_summary(result)))
            self._refresh()
        self._job("SDK check" if use_sdk else "Local check", snapshot,
                  lambda: tuple(check_device(snapshot, identifier, model=model, use_sdk=use_sdk) for identifier in identifiers), complete)

    def _export(self):
        from .export import ExportOptions, export_package

        snapshot = self.state.snapshot()
        identifiers, options, model = self._scope(snapshot), ExportOptions(self.temp["out"], self.temp["format"]), self.model
        self._job("Export", snapshot, lambda: export_package(snapshot, identifiers, options, model=model),
                  lambda result: self._notice(f"Exported revision {result.revision}: {result.path}"))

    def _put_scene(self, key, create, update):
        handle = self.scene_handles.get(key)
        if handle is None:
            handle = create()
            self.scene_handles[key] = handle
        update(handle)
        return handle

    def _draw(self, *, force=False):
        now = time.monotonic()
        if not force and now - self._last_draw < 1 / 30:
            self._draw_pending = True
            return
        self._last_draw = now
        self._draw_pending = False
        import numpy as np
        import trimesh
        from .geometry import matrix_pose, pose_matrix
        from .profiles import profile_for
        from .drivers import view_root

        snapshot = Snapshot(self.preview.document, self.preview.base_revision) if self.preview else self.state.snapshot()
        document = snapshot.document
        live_mode = self.state.view.mode in ("LIVE", "PAUSED")
        used = set()
        displays = {}
        display_issues = []
        sphere = getattr(self, "_sphere", None)
        if sphere is None:
            self._sphere = sphere = trimesh.creation.icosphere(subdivisions=1)
        colors = ((92, 156, 215), (210, 137, 88), (107, 176, 128), (179, 127, 197), (208, 183, 88))
        active_slot = self.gesture.control.slot if self.gesture and self.gesture.control.slot is not None else self.hover_slot
        model_highlight_links = set()
        if self.device_id and active_slot is not None:
            active_device = document.device(self.device_id)
            for taxel in active_device.taxels:
                index = _row_index(taxel.taxel_index, len(active_device.skinning.indices))
                if taxel.id in self.selected and index is not None:
                    row = active_device.skinning.indices[index]
                    if isinstance(row, tuple) and active_slot < len(row):
                        segment_row = _row_index(row[active_slot], len(active_device.segments))
                        if segment_row is not None and active_device.segments[segment_row].frame_ref.startswith("link:"):
                            model_highlight_links.add(active_device.segments[segment_row].frame_ref[5:])
        for device_index, device in enumerate(document.devices):
            if self.temp["visibility"] == "Current device" and device.id != self.device_id:
                continue
            offset = np.asarray((0.35 * device_index, 0, 0)) if live_mode else np.zeros(3)
            q0 = document.bind_configuration
            active_slot = self.gesture.control.slot if self.gesture and self.gesture.control.slot is not None else self.hover_slot
            highlighted_segments = set()
            if device.id == self.device_id and active_slot is not None:
                for taxel in device.taxels:
                    index = _row_index(taxel.taxel_index, len(device.skinning.indices))
                    if taxel.id in self.selected and index is not None:
                        row = device.skinning.indices[index]
                        if isinstance(row, tuple) and active_slot < len(row):
                            segment_row = _row_index(row[active_slot], len(device.segments))
                            if segment_row is not None:
                                highlighted_segments.add(segment_row)
            q = dict(q0)
            if self.state.view.mode == "PREVIEW":
                q.update(self.state.view.q_preview)
            surface = draft_surface = None
            root = None
            root_failure = None
            metadata_space = draft_space = np.eye(4)
            if self.model:
                configured = self.model.with_config(document.model)
                try:
                    root = profile_for(device).root_pose(device, configured, q0)
                    if live_mode:
                        surface = draft_surface = configured.surface_snapshot(q0, root_pose=root)
                    else:
                        metadata_space = view_root(device, configured, q, live=False).metadata_to_view
                        draft_space = view_root(device, configured, q0, live=False).metadata_to_view
                        surface = configured.surface_snapshot(q)
                        draft_surface = configured.surface_snapshot(q0)
                except (ValueError, TypeError, KeyError) as exception:
                    root_failure = str(exception)
                    if root is not None and not live_mode:
                        draft_space = root
                        metadata_space = root
                    try:
                        if not live_mode:
                            surface = configured.surface_snapshot(q)
                            draft_surface = configured.surface_snapshot(q0)
                    except (ValueError, TypeError, KeyError):
                        surface = draft_surface = None
                model_device = self.device_id if self.temp["visibility"] == "Current device" else document.devices[0].id
                instances = surface.instances if surface and (live_mode or device.id == model_device) else ()
                for index, instance in enumerate(instances):
                    key = ("model", device.id, instance.geometry.key)
                    used.add(key)
                    pose = instance.pose.copy()
                    pose[:3, 3] += offset
                    row = matrix_pose(pose)
                    geometry = instance.geometry
                    handle = self._put_scene(key, lambda geometry=geometry, index=index: self.server.scene.add_mesh_simple(
                        f"/draft/{device.id}/model/{index}", vertices=geometry.vertices, faces=geometry.faces,
                        color=(175, 179, 185), opacity=self.temp["opacity"]), lambda _handle: None)
                    handle.position, handle.wxyz = row[:3], (row[6], *row[3:6])
                    handle.visible = self.temp["show_model"] and geometry.link not in self.hidden_links
                    handle.wireframe = live_mode
                    handle.opacity = min(0.2, self.temp["opacity"]) if live_mode else self.temp["opacity"] * (0.25 if model_highlight_links and geometry.link not in model_highlight_links else 1)
                    handle.color = (220, 190, 100) if geometry.link in model_highlight_links or geometry.link == self.selected_link else (175, 179, 185)
                if self.state.view.mode == "PREVIEW" and draft_surface and device is document.devices[0]:
                    for index, instance in enumerate(draft_surface.instances):
                        key = ("draft_model", device.id, instance.geometry.key)
                        used.add(key)
                        geometry = instance.geometry
                        handle = self._put_scene(key, lambda geometry=geometry, index=index: self.server.scene.add_mesh_simple(
                            f"/draft/{device.id}/q0_reference/{index}", vertices=geometry.vertices, faces=geometry.faces,
                            color=(95, 110, 125), opacity=0.25, wireframe=True), lambda _handle: None)
                        pose = instance.pose.copy()
                        pose[:3, 3] += offset
                        value = matrix_pose(pose)
                        handle.position, handle.wxyz = value[:3], (value[6], *value[3:6])
            if self.model is not None and root is None:
                display_issues.append(Issue("warning", "view_root", ("The q0 model reference is hidden; draft poses remain in the declared hand root: " if live_mode else
                                            "Metadata geometry is hidden until its model wrist reference is valid: ")
                                            + (root_failure or "reference is unavailable"), device.id, field="wrist_mount"))
                if not live_mode:
                    displays[device.id] = {"centers": {}, "raw_centers": {}, "surface": surface, "draft_surface": draft_surface,
                                          "offset": tuple(offset), "root": None, "snapshot": snapshot, "surface_space": "model"}
                    continue
            centers, raw_centers, positions, tint, orientation = {}, {}, [], [], []
            current_segments = device.segment_bind_poses
            if self.model and root is not None and self.state.view.mode == "PREVIEW":
                resolved = profile_for(device).resolve(device, configured, q, require_attachments=True)
                current_segments = resolved.segment_poses
                display_issues.extend(resolved.issues)
            moved = {}
            if self.state.view.mode == "PREVIEW":
                from .mappings.lbs import evaluate

                rows = tuple(_row_index(taxel.taxel_index, min(len(device.skinning.indices), len(device.skinning.weights)))
                             for taxel in device.taxels)
                result = evaluate(device.segment_bind_poses, tuple(taxel.bind_pose for taxel in device.taxels),
                                  tuple(device.skinning.indices[index] if index is not None else None for index in rows),
                                  tuple(device.skinning.weights[index] if index is not None else None for index in rows), current_segments)
                moved = {taxel.id: result.positions[row] for row, taxel in enumerate(device.taxels) if result.valid[row]}
                for issue in result.issues:
                    row = _row_index(int(issue.entity_id), len(device.taxels)) if str(issue.entity_id).isdecimal() else None
                    display_issues.append(replace(issue, device_id=device.id,
                                                  entity_id=device.taxels[row].id if row is not None else None))
            for taxel in device.taxels:
                try:
                    pose = pose_matrix(taxel.bind_pose)
                except (ValueError, TypeError):
                    continue
                raw_centers[taxel.id] = tuple((draft_space @ pose)[:3, 3])
                if taxel.id in moved:
                    point = metadata_space[:3, :3] @ moved[taxel.id] + metadata_space[:3, 3] + offset
                else:
                    point = (draft_space @ pose)[:3, 3] + offset
                centers[taxel.id] = tuple(point)
                positions.append(point)
                orientation.append((1, 0, 0, 0))
                index = _row_index(taxel.taxel_index, min(len(device.skinning.indices), len(device.skinning.weights)))
                color = (135, 135, 140)
                try:
                    if index is None:
                        raise IndexError("Invalid raw skinning row")
                    row, weights = device.skinning.indices[index], device.skinning.weights[index]
                    dominant = _row_index(row[max(range(4), key=lambda slot: weights[slot])], len(device.segments))
                    if dominant is not None:
                        color = colors[dominant % len(colors)]
                except (IndexError, TypeError, ValueError):
                    color = (220, 65, 65)
                if self.state.view.mode == "PREVIEW" and taxel.id not in moved:
                    color = (125, 125, 130)
                if live_mode:
                    color = (110, 110, 115)
                if taxel.id in self.selected and device.id == self.device_id and active_slot is not None:
                    try:
                        weight = device.skinning.weights[index][active_slot]
                        display = _weight_display(weight)
                        color = display[0] if display is not None else (220, 65, 65)
                    except (TypeError, IndexError, ValueError):
                        color = (220, 65, 65)
                tint.append(color)
            key = ("taxels", device.id)
            if positions:
                used.add(key)
                data = np.asarray(positions)
                handle = self.scene_handles.get(key)
                if handle is None or len(handle.batched_positions) != len(positions):
                    if handle:
                        handle.remove()
                    handle = self.server.scene.add_batched_meshes_simple(f"/draft/{device.id}/taxels", vertices=sphere.vertices,
                        faces=sphere.faces, batched_wxyzs=np.asarray(orientation), batched_positions=data,
                        batched_scales=np.full(len(data), 0.0025), batched_colors=np.asarray(tint, dtype=np.uint8), lod="off")
                    self.scene_handles[key] = handle
                else:
                    handle.batched_positions = data
                    handle.batched_colors = np.asarray(tint, dtype=np.uint8)
                handle.opacity = 0.35 if live_mode else 1.0
                selected_points = [point for identity, point in centers.items() if device.id == self.device_id and identity in self.selected]
                key = ("selection_outline", device.id)
                if selected_points:
                    used.add(key)
                    handle = self.scene_handles.get(key)
                    if handle:
                        handle.remove()
                    self.scene_handles[key] = self.server.scene.add_batched_meshes_simple(
                        f"/draft/{device.id}/selection_outline", vertices=sphere.vertices, faces=sphere.faces,
                        batched_wxyzs=np.tile((1, 0, 0, 0), (len(selected_points), 1)), batched_positions=np.asarray(selected_points),
                        batched_scales=np.full(len(selected_points), 0.003), batched_colors=(255, 220, 40), wireframe=True, lod="off")
            segment_positions = {}
            for row, segment in enumerate(device.segments):
                if row >= len(current_segments):
                    continue
                try:
                    pose = pose_matrix(current_segments[row])
                except (ValueError, TypeError):
                    continue
                pose = metadata_space @ pose
                segment_positions[row] = pose[:3, 3] + offset
                if device.id == self.device_id and (segment.id == self.selected_segment or row in highlighted_segments):
                    row_pose = matrix_pose(pose)
                    key = ("segment_axis", device.id, segment.id)
                    used.add(key)
                    frame = self._put_scene(key, lambda: self.server.scene.add_frame(f"/draft/{device.id}/segment/{segment.id}",
                        axes_length=0.02, axes_radius=0.0008), lambda _handle: None)
                    frame.position = tuple(pose[:3, 3] + offset)
                    frame.wxyz = (row_pose[6], *row_pose[3:6])
            lines, line_colors = [], []
            for taxel in device.taxels:
                if device.id != self.device_id or taxel.id not in self.selected or taxel.id not in centers:
                    continue
                try:
                    raw = draft_space @ pose_matrix(taxel.bind_pose)
                    start = raw[:3, 3] + offset
                    for axis, color in ((0, (240, 80, 80)), (1, (90, 220, 110)), (2, (80, 120, 250))):
                        lines.append((start, start + raw[:3, axis] * 0.012))
                        line_colors.append((color, color))
                    row = _row_index(taxel.taxel_index, min(len(device.skinning.indices), len(device.skinning.weights)))
                    if row is None:
                        continue
                    for slot, (segment_index, weight) in enumerate(zip(device.skinning.indices[row], device.skinning.weights[row])):
                        if active_slot is not None and slot != active_slot:
                            continue
                        segment_index = _row_index(segment_index, len(device.segments))
                        display = _weight_display(weight)
                        if segment_index in segment_positions and display is not None and weight > 0:
                            lines.append((segment_positions[segment_index], centers[taxel.id]))
                            brightness = display[1]
                            line_colors.append(((brightness, brightness, 70),) * 2)
                except (TypeError, IndexError):
                    continue
            key = ("binding_lines", device.id)
            if lines:
                used.add(key)
                points = np.asarray(lines)
                handle = self.scene_handles.get(key)
                if handle:
                    handle.remove()
                self.scene_handles[key] = self.server.scene.add_line_segments(f"/draft/{device.id}/binding_lines",
                    points=points, colors=np.asarray(line_colors, dtype=np.uint8), thickness=2, thickness_units="screen")
            displays[device.id] = {"centers": centers, "raw_centers": raw_centers, "surface": surface, "draft_surface": draft_surface,
                                    "offset": tuple(offset), "root": root, "snapshot": snapshot,
                                    "surface_space": "root" if live_mode else "model"}
        for key in tuple(self.scene_handles):
            if key not in used:
                self.scene_handles.pop(key).remove()
        self.display_version += 1
        self.display = (self.display_version, self.state.revision, displays)
        self.display_issues = (*display_issues, *self._render_live_layer())
        if self.editor:
            self._create_gizmo()
        if self.state.view.mode == "LIVE":
            self._schedule_live_layer()

    def _create_gizmo(self, *, force=False):
        import numpy as np
        from .geometry import matrix_pose, pose_matrix

        if self.editor is None:
            return
        if not self.bridge_ready:
            if self.gizmo:
                self.gizmo.remove()
                self.gizmo = None
            self.transforms.clear()
            return
        device_id = self.device_id
        kind, identity, pose = None, None, None
        if device_id:
            snapshot = self.state.snapshot()
            if len(self.selected) == 1:
                kind, identity = "taxel", self.selected[0]
                entity = next((item for item in self.device(snapshot).taxels if item.id == identity), None)
                pose = entity.bind_pose if entity else None
            elif self.selected_segment:
                kind, identity = "segment", self.selected_segment
                pose = self._segment_pose(snapshot, identity)
            elif self.selected_link and self.model:
                kind, identity = "link", self.selected_link
                try:
                    pose = matrix_pose(self.model.with_config(snapshot.document.model).link_pose(identity, snapshot.document.bind_configuration))
                except (ValueError, TypeError, KeyError, OverflowError):
                    pose = None
        key = (self.generation, device_id, kind, identity)
        if not force and self.gizmo is not None and getattr(self, "_gizmo_key", None) == key:
            if pose is None:
                self.gizmo.visible = False
                return
            if self.gizmo and self.gesture is None and pose is not None:
                try:
                    matrix = self._gizmo_space(kind, snapshot) @ pose_matrix(pose)
                    value = matrix_pose(matrix)
                    self.gizmo.position, self.gizmo.wxyz = value[:3], (value[6], *value[3:6])
                    self.gizmo.visible = True
                except (ValueError, TypeError):
                    self.gizmo.visible = False
            return
        if self.gizmo:
            self.gizmo.remove()
        self.transforms.clear()
        self.gizmo = None
        self._gizmo_key = key
        if pose is None:
            self._configure_bridge()
            return
        try:
            matrix = self._gizmo_space(kind, snapshot) @ pose_matrix(pose)
            value = matrix_pose(matrix)
        except (ValueError, TypeError):
            self._configure_bridge()
            return
        name = f"/editor/{self.editor.client_id}/gizmo/{self.generation}/{secrets.token_hex(8)}"
        handle = self.editor.scene.add_transform_controls(name, position=value[:3], wxyz=(value[6], *value[3:6]), scale=0.04)
        self.gizmo = handle
        def build(snapshot, value, _source):
            target = np.linalg.inv(self._gizmo_space(kind, snapshot)) @ pose_matrix(value)
            if kind == "link":
                configured = self.model.with_config(snapshot.document.model)
                adjustment = configured.adjustment_for_pose(identity, snapshot.document.bind_configuration, target)
                values = thaw(snapshot.document.model.link_adjustments)
                values[identity] = adjustment
                document = replace(snapshot.document, model=replace(snapshot.document.model, link_adjustments=values))
                return Candidate(snapshot.revision, _restore(document, self.model), "Move URDF link")
            return _pose_candidate(snapshot, device_id, identity, matrix_pose(target), kind, self.model)
        control = _Control(handle, "transform", self.generation, build=build, entity_id=identity, device_id=device_id, name=name)
        self.transforms[name] = control
        @handle.on_update
        async def native_update(event):
            self.inbox.put(("native_transform", name, event.phase))
        self._configure_bridge()

    def _gizmo_space(self, kind, snapshot):
        import numpy as np
        from .drivers import view_root

        space = np.eye(4)
        if not self.display or self.device_id not in self.display[2]:
            return space
        displayed = self.display[2][self.device_id]
        live_mode = self.state.view.mode in ("LIVE", "PAUSED")
        if self.model and ((kind == "link") == live_mode):
            try:
                root_space = view_root(
                    snapshot.document.device(self.device_id), self.model.with_config(snapshot.document.model),
                    snapshot.document.bind_configuration, live=False).metadata_to_view.copy()
                space = np.linalg.inv(root_space) if live_mode else root_space
            except (ValueError, KeyError, TypeError, OverflowError) as exception:
                raise ValueError("Draft manipulator requires a valid q0 wrist reference") from exception
        space[:3, 3] += displayed["offset"]
        return space

    def _register_selection(self, client):
        def capture(event, kind):
            display = self.display
            if display is None:
                return
            camera = client.camera
            copied = {"position": tuple(camera.position), "wxyz": tuple(camera.wxyz),
                      "fov": camera.fov, "aspect": camera.aspect}
            if kind == "click":
                data = (tuple(event.ray_origin), tuple(event.ray_direction), event.modifier)
            else:
                data = (tuple(event.screen_min), tuple(event.screen_max), event.modifier)
            self.inbox.put(("selection", client.client_id, kind, display, copied, data))
        for modifier in (None, "cmd/ctrl"):
            client.scene.on_click(modifier=modifier)(lambda event: capture(event, "click"))
        client.scene.on_rect_select(modifier="shift")(lambda event: capture(event, "rect"))

    def _selection(self, client_id, kind, display, camera, data):
        import numpy as np
        from .geometry import pose_matrix, transform_point, transform_direction
        from .layout import pick_taxel, rectangle_select, place_taxel

        if self.editor is None or client_id != self.editor.client_id:
            return
        if display[0] != self.display_version or display[1] != self.state.revision:
            self._notice("Selection was discarded because the displayed snapshot changed")
            return
        shown = display[2].get(self.device_id)
        if shown is None:
            return
        if kind == "rect":
            position, quaternion = camera["position"], camera["wxyz"]
            camera_pose = pose_matrix((*position, *quaternion[1:], quaternion[0]))
            camera_pose[:3, 3] -= shown["offset"]
            centers = {identity: np.asarray(point) - shown["offset"] for identity, point in shown["centers"].items()}
            selected = rectangle_select(centers, camera_pose, camera["fov"], camera["aspect"], data[0], data[1],
                                        surface=shown["surface"] if self.temp["show_model"] else None, through=self.temp["through"])
            self._select_taxels(selected)
            return
        origin, direction, modifier = data
        selected = pick_taxel(origin, direction, shown["centers"], 0.003)
        if selected and self.temp["point_mode"] == "Select taxel":
            values = set(self.selected) if modifier else set()
            values.symmetric_difference_update((selected,))
            self._select_taxels(tuple(item.id for item in self.device().taxels if item.id in values))
            return
        if self.temp["point_mode"] == "Select taxel" or shown["draft_surface"] is None:
            return
        hits = shown["draft_surface"].ray_hits(np.asarray(origin) - shown["offset"], direction, max_distance=100)
        if not hits:
            raise ValueError("The click did not hit the displayed model")
        hit = min(hits, key=lambda item: item.distance)
        mode = self.temp["point_mode"]
        if mode.startswith("Wrist "):
            point = transform_point(shown["root"], hit.position) if shown["surface_space"] == "root" else hit.position
            self.temp[{"Wrist O": "wrist_origin", "Wrist F": "wrist_forearm", "Wrist D": "wrist_dorsal"}[mode]] = _json(tuple(float(value) for value in point))
        else:
            if shown["root"] is None:
                raise ValueError("Calibrate the wrist reference before placing taxels")
            if shown["surface_space"] == "model":
                inverse = np.linalg.inv(shown["root"])
                hit = replace(hit, position=tuple(transform_point(inverse, hit.position)),
                              normal=tuple(transform_direction(inverse, hit.normal)))
            if mode == "Place selected segment":
                if self.selected_segment is None:
                    raise ValueError("Select a segment first")
                segment = next(item for item in self.device().segments if item.id == self.selected_segment)
                if segment.semantic == "WRIST":
                    raise ValueError("WRIST uses the root identity reference")
                pose = self._segment_pose(self.state.snapshot(), segment.id)
                from .geometry import pose_matrix

                pose_matrix(pose)
                self.state.dispatch(_pose_candidate(self.state.snapshot(), self.device_id, segment.id,
                                    (*hit.position, *pose[3:]), "segment", self.model))
                self._after_document(rebuild=True)
                return
            if mode == "Place taxel":
                self.state.dispatch(place_taxel(hit, self._batch_spec(), self.state.snapshot()))
                self._after_document(rebuild=True)
                return
            key = {"Batch origin": "batch_origin", "Batch column end": "batch_column_end", "Batch row end": "batch_row_end"}[mode]
            self.temp[key] = _json(hit.position)
            self.temp["batch_normal"] = _json(hit.normal)
            links = _parse_json(self.temp["batch_links"])
            if hit.link not in links:
                links.append(hit.link)
            self.temp["batch_links"] = _json(links)
            seeds = _parse_json(self.temp["batch_seeds"])
            components = seeds.setdefault(hit.geometry_key, [])
            if hit.component not in components:
                components.append(hit.component)
            self.temp["batch_seeds"] = _json(seeds)
        self._refresh()

    def process(self, item):
        kind, *args = item
        try:
            if kind == "connect":
                self.connect(args[0])
                if self.editor is args[0]:
                    self._register_selection(args[0])
            elif kind == "disconnect":
                self.disconnect(args[0])
            elif kind == "bridge":
                self._receive(*args)
            elif kind == "selection":
                self._selection(*args)
            elif kind == "job":
                identity, snapshot, label, result, error, complete = args
                self.jobs.discard(identity)
                self.job_threads.pop(identity, None)
                if error:
                    if label == "Prepare live layer":
                        self.live_render_pending = False
                        self.live_render_error = self.live_render_key[:4] if self.live_render_key else None
                    self._notice(f"{label} failed for revision {snapshot.revision}: {error}")
                else:
                    complete(result)
                if label == "Prepare live layer":
                    self._refresh_info()
                else:
                    self._refresh()
            elif kind == "live_connection":
                generation, connection, error = args
                if generation == self.live_generation:
                    self.live_connection = connection if self.live_wanted else None
                    self.live_frame = None
                    self.live_poll_key = self.live_render_key = self.live_render_error = None
                    self.received_cache.clear()
                    if error:
                        self.live_status = ("error", "", (Issue("error", "live_connect", str(error), field="live"),), self.demo)
                        self._notice(f"Receiving connection failed: {error}")
                    elif connection is not None and self.live_wanted:
                        self.poll_live()
                    self._draw()
                    self._refresh_live_status()
            elif kind == "native_transform":
                pass
        except Exception as exception:
            LOGGER.exception("Editor operation failed")
            self._notice(str(exception))
            try:
                self._refresh()
            except Exception:
                LOGGER.exception("Unable to refresh the editor after an operation error")


def _initial_document(*, urdf, sides, metadata, left_metadata, right_metadata):
    from .document import load_metadata, merge_documents

    documents = []
    if metadata:
        documents.append(load_metadata(metadata, side=sides if sides in ("left", "right") else None))
    if left_metadata:
        documents.append(load_metadata(left_metadata, side="left"))
    if right_metadata:
        documents.append(load_metadata(right_metadata, side="right"))
    document = documents[0] if len(documents) == 1 else merge_documents(tuple(documents))
    if urdf:
        if sides not in ("left", "right", "both"):
            raise ValueError("Declare left, right or both when loading a URDF")
        if document.model and Path(document.model.path).resolve() != Path(urdf).resolve():
            raise ValueError("The metadata and command line identify different URDF files")
        config = document.model or ModelConfig(str(Path(urdf).resolve()), sides)
        if config.sides != sides:
            raise ValueError("The metadata and command line declare different model sides")
        devices = document.devices
        if not devices:
            declared = ("left", "right") if sides == "both" else (sides,)
            devices = tuple(Device(side=side) for side in declared)
        document = replace(document, model=config, devices=devices)
    return document


def run_gui(*, urdf=None, sides=None, metadata=None, left_metadata=None,
            right_metadata=None, port=8080, demo=False):
    """Run in the launching interpreter; unavailable dependencies never install."""
    document = _initial_document(urdf=urdf, sides=sides, metadata=metadata,
                                 left_metadata=left_metadata, right_metadata=right_metadata)
    viser = _safe_viser()
    import numpy  # noqa: F401

    inbox = queue.Queue()
    http = _BridgeHTTP(inbox)
    threading.Thread(target=http.serve_forever, name="tactile-editor-http", daemon=True).start()
    server = None
    editor = None
    try:
        server = viser.ViserServer(host="127.0.0.1", port=port)
        server.gui.configure_theme(control_layout="fixed", control_width="medium")
        editor = _Editor(server, http, State(document, saved=bool(metadata or left_metadata or right_metadata)), demo=demo)
        if metadata and len(document.devices) == 1:
            editor.paths[document.devices[0].id] = str(metadata)
        for side, path in (("left", left_metadata), ("right", right_metadata)):
            if path:
                for device in document.devices:
                    if device.side == side:
                        editor.paths[device.id] = str(path)
        @server.on_client_connect
        async def on_connect(client):
            inbox.put(("connect", client))
        @server.on_client_disconnect
        async def on_disconnect(client):
            inbox.put(("disconnect", client.client_id))
        LOGGER.info("Tactile metadata editor: http://127.0.0.1:%s", server.get_port())
        while not editor.stopping:
            editor.poll_live()
            try:
                item = inbox.get(timeout=1 / 30)
            except queue.Empty:
                if editor._draw_pending:
                    editor._draw()
                continue
            editor.process(item)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            if editor is not None:
                editor.close()
        finally:
            if http.session:
                http.session.revoke()
            http.shutdown()
            http.server_close()
            if server:
                server.stop()

