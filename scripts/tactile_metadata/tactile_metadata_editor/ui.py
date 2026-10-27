# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import json
import logging
import math
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import trimesh
import viser
from trimesh import transformations as tf

from . import geometry as geo, sdk, storage
from .document import Document, FIELDS, GROUPS, IDENTITY, JOINTS, State, joint_number

LOG = logging.getLogger("isaaccapture.tactile_metadata.editor")


class Editor:
    def __init__(self, state, args):
        self.state, self.args, self.started = state, args, bool(state.documents)
        if not state.documents:
            state.documents["left"] = Document("left")
            state.saved["left"] = state.fingerprint("left")
        self.events, self.pool = queue.SimpleQueue(), ThreadPoolExecutor(max_workers=1)
        self.model, self.receiver, self.job = None, None, None
        self.selected, self.points, self.candidates = [], {}, []
        self.handles, self.q, self.logs, self.drive = {}, {}, [], {}
        self.frame_forces, self.frame_colors, self.frame_counts, self.issues = (
            {},
            {},
            {},
            [],
        )
        self.pick_target, self.slot, self.dragging, self.closing = (
            None,
            None,
            False,
            False,
        )
        self.sdk_result, self.sdk_revision = "not checked", None
        self.server = viser.ViserServer(host="127.0.0.1", port=args.port)
        g = self.server.gui
        g.configure_theme(control_width="medium", show_share_button=False)
        g.main_panel.dock_right()
        self.header = g.add_markdown("")
        self.side = g.add_dropdown("Edit hand", tuple(state.documents))
        self.both = g.add_checkbox("Show both hands", True)
        self.mode = g.add_dropdown("Mode", ("EDIT", "PREVIEW", "LIVE", "PAUSED"))
        self.watch(self.side, lambda _: self.switch())
        self.watch(self.mode, lambda _: None)
        with g.add_tab_group() as tabs:
            with tabs.add_tab("Model"):
                self.urdf = g.add_text("URDF", state.model.get("urdf", ""))
                self.packages = g.add_text(
                    "Local package map JSON",
                    json.dumps(state.model.get("packages", {})),
                )
                self.sides = g.add_dropdown(
                    "New model hands",
                    ("left", "right", "both"),
                    initial_value="both"
                    if len(state.documents) == 2
                    else self.side.value,
                )
                self.button("New document", lambda: self.guard(self.reset))
                self.button("Load URDF", lambda: self.guard(self.load_new))
                self.open_path = g.add_text("Metadata JSON", args.metadata or "")
                self.json_side = g.add_dropdown(
                    "Assign JSON to", ("left", "right"), initial_value=self.side.value
                )
                self.button("Open JSON", lambda: self.guard(self.open))
                self.wrist = g.add_dropdown("Wrist link", ("",))
                self.allowed = g.add_text("Allowed links, comma separated", "")
                self.button("Apply allowed links", self.set_links)
                self.pick = g.add_dropdown(
                    "Pick target",
                    (
                        "O",
                        "C",
                        "R",
                        "Wrist O",
                        "Wrist F",
                        "Wrist D",
                        "Segment",
                        "Move selected",
                        "Single tangent",
                    ),
                )
                self.pick_button = self.button(
                    "Pick on q0 model",
                    lambda: setattr(self, "pick_target", self.pick.value),
                )
                self.calibrate_button = self.button(
                    "Preview / set wrist reference", self.calibrate
                )
                self.resources = g.add_markdown("")
                self.button("Reset cameras", self.camera)
            with tabs.add_tab("Segments"):
                self.joint = g.add_dropdown("URDF free joint", ("",))
                self.angle = g.add_slider("Preview joint", -3.15, 3.15, 0.001, 0.0)
                self.angle_number = g.add_number("Preview joint value", 0.0)
                self.watch(self.joint, lambda _: self.joint_controls())
                for w in (self.angle, self.angle_number):
                    self.watch(w, self.preview_joint)
                self.button("Set shared binding pose q0", self.confirm_q0)
                self.button("Return to q0", lambda: setattr(self.mode, "value", "EDIT"))
                self.segment = g.add_dropdown("Segment row", ("",))
                self.semantic = g.add_dropdown("HandJoint", tuple(JOINTS))
                self.segment_link = g.add_dropdown("Associated link", ("",))
                self.watch(self.segment, lambda _: self.select_segment())
                self.button("Add segment", self.add_segment)
                self.button("Set semantic / link", self.associate)
                self.order = g.add_text("Segment row order", "[]")
                self.button(
                    "Reorder / delete omitted segments",
                    self.reorder_segments,
                )
                self.segment_table = g.add_markdown("")
            with tabs.add_tab("Layout"):
                self.surface = g.add_dropdown("Projection surface", ("",))
                self.rows = g.add_number("Rows", 4, min=1)
                self.cols = g.add_number("Columns", 4, min=1)
                self.centers = [
                    g.add_vector3(k + " world center", (0.0, 0.0, 0.0))
                    for k in ("O", "C", "R")
                ]
                for key, w in zip(("O", "C", "R"), self.centers):
                    self.watch(w, lambda v, key=key: self.points.update({key: list(v)}))
                self.spans = [
                    g.add_number(k + " span (m)", 0.03) for k in ("Column", "Row")
                ]
                self.button("Apply spans along picked directions", self.spans_apply)
                self.scan = g.add_dropdown(
                    "Scan", ("row-major", "column-major", "row-snake", "column-snake")
                )
                self.reverse = [
                    g.add_checkbox(k, False)
                    for k in ("Reverse rows", "Reverse columns")
                ]
                self.excluded = g.add_text("Exclude cells [row,column]", "[]")
                self.start = g.add_number("Start output index (-1 = unused)", -1)
                self.influences = g.add_text("New taxel segment rows (1–4)", "[]")
                self.grid_button = self.button("Preview grid", self.grid)
                self.button("Apply candidates", self.apply_grid)
                self.button("Cancel candidates", self.clear_layout)
                self.layout_info = g.add_markdown("")
                self.selection = g.add_text("Select output indices / ranges", "")
                self.button("Select indices", self.select_indices)
                self.rectangle = g.add_checkbox("Rectangle selection (drag)", False)
                self.through = g.add_checkbox("Select through mesh", False)
                self.watch(self.rectangle, self.rectangle_mode)
                self.button(
                    "Delete selected taxels",
                    self.delete_taxels,
                )
                self.reproject_button = self.button(
                    "Reproject selected", self.reproject
                )
                self.indices = g.add_text("New selected output indices", "[]")
                self.button("Preview / apply output indices", self.reindex)
                self.page = g.add_number("List page (25 rows)", 0, min=0)
                self.watch(self.page, lambda _: None)
                self.table = g.add_markdown("")
            with tabs.add_tab("Binding"):
                self.binding_info = g.add_markdown("")
                self.button(
                    "Assign segment rows equally",
                    lambda: self.change(
                        self.d.assign, self.selected, json.loads(self.influences.value)
                    ),
                )
                self.slots = []
                for k in range(4):
                    with g.add_folder(f"Slot {k}"):
                        segment = g.add_dropdown("Segment", ("",))
                        index = g.add_number("Set raw index", 0)
                        slider = g.add_slider("Weight", 0.0, 1.0, 0.001, 0.0)
                        number = g.add_number("Set weight", 0.0)
                        self.watch(
                            segment,
                            lambda v, k=k: (
                                self.set_slot(k, int(v.split(":")[0]))
                                if not v.startswith("Raw")
                                else None
                            ),
                        )
                        self.watch(index, lambda v, k=k: self.set_slot(k, v))
                        for w in (slider, number):
                            self.watch(w, lambda v, k=k: self.weight(k, v))
                        self.slots.append((segment, index, slider, number))
            with tabs.add_tab("Live"):
                self.collection = g.add_text("Tactile collection_id", "tactile")
                self.hand = g.add_dropdown("HandTracker output", ("left", "right"))
                self.joint_enabled = g.add_checkbox("Enable JointState", False)
                self.joint_id = g.add_text("JointState collection_id", "")
                self.button(
                    "Save this hand's source configuration", self.configure_source
                )
                self.source_joint = g.add_text("Source joint name", "")
                self.target_joint = g.add_dropdown("URDF target joint", ("",))
                self.scale = g.add_number("Scale", 1.0)
                self.offset = g.add_number("Offset", 0.0)
                self.button("Add / update mapping", self.mapping)
                self.button("Remove mapping", lambda: self.mapping(True))
                self.mapping_info = g.add_markdown("")
                self.button("Connect / disconnect", self.connect)
                self.button(
                    "Play / pause",
                    lambda: setattr(
                        self.mode,
                        "value",
                        "PAUSED" if self.mode.value == "LIVE" else "LIVE",
                    ),
                )
                self.source = g.add_dropdown(
                    "Visualization metadata", ("Received", "Draft")
                )
                self.force_max = g.add_number(
                    "Force color maximum (N)", 30.0, min=0.001
                )
                self.groups = {name: g.add_checkbox(name, True) for name in GROUPS}
                self.button("Record metadata groups", self.record)
                self.live_info = g.add_markdown("")
            with tabs.add_tab("Export"):
                self.paths = {
                    s: g.add_text(
                        s + " save JSON", state.paths.get(s, s + ".metadata.json")
                    )
                    for s in ("left", "right")
                }
                self.button("Save all drafts", self.save)
                self.all_sides = g.add_checkbox("Check / export all hands", True)
                self.destination = g.add_text("New output directory", "package")
                self.binary = g.add_checkbox("Include SDK binary", False)
                self.button("Local check", self.local_check)
                self.button("SDK check", lambda: self.background(False))
                self.button("Export snapshot", lambda: self.background(True))
                self.check_info = g.add_markdown("")
                with g.add_folder("Advanced schema repair"):
                    self.schema = g.add_text("Schema JSON", "", multiline=True)
                    self.button(
                        "Copy draft into repair field",
                        lambda: setattr(
                            self.schema,
                            "value",
                            json.dumps(
                                storage.metadata(self.d, self.state.model), indent=2
                            ),
                        ),
                    )
                    self.button("Replace draft from repair field", self.repair)
        with g.add_folder("Draft property inspector"):
            self.entity = g.add_dropdown("Edit entity", ("Taxels", "Segment"))
            self.watch(self.entity, lambda _: None)
            self.raw = g.add_markdown("")
            self.pose_fields = []
            for axis, k in enumerate("XYZ"):
                w = g.add_number("Set " + k + " (m)", 0.0)
                self.watch(w, lambda v, axis=axis: self.components([axis], [v]))
                self.pose_fields.append(w)
            self.button("Flip normal", lambda: self.rotate(np.pi, [1, 0, 0]))
            self.twist = g.add_number("Tangent twist (degrees)", 0.0)
            self.button(
                "Apply tangent twist",
                lambda: self.rotate(np.deg2rad(self.twist.value), [0, 0, 1]),
            )
            with g.add_folder("Advanced XYZW"):
                for axis, k in enumerate("XYZW", 3):
                    w = g.add_number("Set Q" + k, float(axis == 6))
                    self.watch(w, lambda v, axis=axis: self.components([axis], [v]))
                    self.pose_fields.append(w)
            self.radius = g.add_number("Marker radius (m)", 0.002, min=0.0001)
        g.add_markdown(
            "Draft manipulators always edit q0. Wire markers are draft targets during playback. Ctrl/Cmd toggles selection. Ranges use `0:15,20`."
        )
        self.info = g.add_dropdown("Info issue", ("",))
        self.button("Locate issue in inspector", self.locate)
        self.log_text = g.add_markdown("")
        self.layers = {
            s: self.server.scene.add_frame("/view/" + s, show_axes=False)
            for s in ("left", "right")
        }
        self.gizmo = self.server.scene.add_transform_controls(
            "/draft-editor", scale=0.05, visible=False
        )
        self.gizmo.on_update(self.gizmo_event)
        for modifier in (None, "cmd/ctrl"):
            self.server.scene.on_click(modifier=modifier)(self.scene_click)
        if state.model.get("urdf"):
            try:
                self.install_model(
                    geo.Model(state.model["urdf"], state.model.get("packages", {}))
                )
            except (OSError, ValueError) as exc:
                self.resources.content = str(exc)
                self.log(
                    "URDF unavailable; numerical draft editing remains available: "
                    + str(exc)
                )
        self.switch()

    @property
    def d(self):
        return self.state.documents[self.side.value]

    def watch(self, widget, callback):
        async def updated(event):
            if event.client:
                value = copy.deepcopy(event.target.value)
                self.events.put(lambda: callback(value))

        widget.on_update(updated)

    def button(self, label, callback):
        handle = self.server.gui.add_button(label)

        async def clicked(_):
            self.events.put(callback)

        handle.on_click(clicked)
        return handle

    def log(self, message):
        LOG.info("%s", message)
        self.logs = (self.logs + [str(message)])[-8:]
        self.log_text.content = "\n\n".join(self.logs)

    def confirm(self, message, action, unsaved=False, cleanup=None):
        modal = self.server.gui.add_modal("Confirm")

        def close():
            if cleanup:
                cleanup()
            modal.close()

        def finish(save=False):
            if save:
                self.save()
            action()
            close()

        with modal:
            self.server.gui.add_markdown(message)
            if unsaved:
                self.button("Save and continue", lambda: finish(True))
            self.button("Discard and continue" if unsaved else "Apply", finish)
            self.button("Cancel", close)

    def guard(self, action):
        if self.state.dirty():
            self.confirm("Unsaved drafts", action, True)
        else:
            action()

    def change(self, action, *args, **kwargs):
        result = self.state.edit(action, *args, **kwargs)
        self.refresh()
        return result

    def install_model(self, model):
        if self.model:
            for h in self.model.handles.values():
                h.remove()
        self.model = model
        self.wrist.options = self.segment_link.options = tuple(model.links)
        self.joint.options = self.target_joint.options = tuple(
            j.name for j in model.joints
        ) or ("",)
        self.q = copy.deepcopy(self.state.model.get("q0", {}))
        self.resources.content = (
            "Missing resources: "
            + str(model.missing)
            + "\n\n"
            + "\n\n".join(
                f"{j.name}: {j.parent} → {j.child}; {j.type}; limits={j.limit}; mimic={j.mimic}"
                for j in model.urdf.robot.joints
            )
        )
        self.sides.disabled = True
        self.sides.value = "both" if len(self.state.documents) == 2 else self.side.value
        self.joint_controls()

    def reset(self):
        if self.receiver:
            self.connect()
        for h in (
            *self.handles.values(),
            *(self.model.handles.values() if self.model else ()),
        ):
            h.remove()
        self.handles, self.model = {}, None
        self.state, self.started = State({"left": Document("left")}), False
        self.state.saved["left"] = self.state.fingerprint("left")
        self.side.options, self.side.value = ("left",), "left"
        self.sides.disabled, self.mode.value = False, "EDIT"
        self.frame_forces, self.frame_colors, self.frame_counts, self.drive = (
            {},
            {},
            {},
            {},
        )
        self.q, self.resources.content = {}, ""
        for w in (
            self.wrist,
            self.joint,
            self.segment_link,
            self.target_joint,
            self.surface,
        ):
            w.options = ("",)
        for side, w in self.paths.items():
            w.value = side + ".metadata.json"
        self.switch()

    def load_new(self):
        cfg = dict(
            urdf=str(Path(self.urdf.value).resolve()),
            packages={
                k: str(Path(v).resolve())
                for k, v in json.loads(self.packages.value).items()
            },
            q0={},
        )
        model = geo.Model(cfg["urdf"], cfg["packages"])
        sides = ("left", "right") if self.sides.value == "both" else (self.sides.value,)
        self.reset()
        self.state, self.started = State({s: Document(s) for s in sides}, cfg), True
        self.side.options, self.side.value = sides, sides[0]
        self.install_model(model)
        self.switch()

    def open(self):
        side, path = self.json_side.value, self.open_path.value
        d = storage.load(path, side)
        if self.model and side not in self.state.documents:
            raise ValueError("Create a new document to change the fixed model hands")
        cfg = d.editor.get("model", {})
        model = (
            geo.Model(cfg["urdf"], cfg.get("packages", {}))
            if not self.model and cfg.get("urdf")
            else None
        )

        def replace():
            if not self.started:
                self.state.documents = {}
            self.state.documents[side] = d
            if model:
                self.state.model = copy.deepcopy(cfg)

        if self.receiver:
            self.connect()
        self.mode.value = "EDIT"
        self.state.edit(replace)
        self.started = True
        self.side.options, self.side.value = tuple(self.state.documents), side
        if model:
            self.install_model(model)
        self.state.paths[side] = self.paths[side].value = str(Path(path).resolve())
        if d.editor.get("side") == side and cfg == self.state.model:
            self.state.saved[side] = self.state.fingerprint(side)
        else:
            self.state.saved.pop(side, None)
        self.switch()

    def save(self):
        storage.save(self.state, {s: self.paths[s].value for s in self.state.documents})
        self.log("Saved all drafts")

    def switch(self):
        self.selected, self.points, self.candidates = [], {}, []
        self.pick_target, self.slot = None, None
        self.segment.value = ""
        self.influences.value = "[]"
        self.allowed.value = ",".join(self.d.editor.get("allowed_links", []))
        if self.model:
            link = self.d.editor.get("root", {}).get("link")
            self.wrist.value = link if link in self.model.links else self.model.links[0]
        source = self.d.editor.get("source", {})
        self.collection.value, self.hand.value = (
            source.get("collection", "tactile"),
            source.get("hand", self.d.side),
        )
        self.joint_id.value = source.get("joint_state", "")
        self.joint_enabled.value = bool(self.joint_id.value)
        self.refresh()

    def root(self, d=None, cfg=None):
        d = d or self.d
        if self.model and "root" in d.editor:
            try:
                return self.model.root(
                    d,
                    self.model.fk(
                        cfg if cfg is not None else self.state.model.get("q0", {})
                    ),
                )
            except (KeyError, ValueError, TypeError):
                pass
        return np.eye(4)

    def surfaces(self):
        return [
            n
            for n, (link, _) in self.model.surfaces.items()
            if link in self.d.editor.get("allowed_links", [])
        ]

    def set_links(self):
        links = [v.strip() for v in self.allowed.value.split(",") if v.strip()]
        if not self.model or any(v not in self.model.links for v in links):
            raise ValueError("Choose links from the loaded model")
        self.change(self.d.editor.update, allowed_links=links)

    def camera(self):
        for client in self.server.get_clients().values():
            client.camera.position, client.camera.look_at = (0.3, -0.4, 0.3), (0, 0, 0)

    def calibrate(self):
        o, f, d = [
            np.asarray(self.points[k]) for k in ("Wrist O", "Wrist F", "Wrist D")
        ]
        z = geo.unit(f - o)
        y = geo.unit(d - o - z * ((d - o) @ z))
        reference = geo.frame(o, z, np.cross(y, z))
        for other in self.state.documents.values():
            if (
                other is not self.d
                and other.editor.get("root", {}).get("link") == self.wrist.value
            ):
                raise ValueError(
                    "Both hands require distinct wrist links in the actual model"
                )
        self.axes("/calibration-preview", [reference])
        local = (
            np.linalg.inv(
                self.model.fk(self.state.model.get("q0", {}))[self.wrist.value]
            )
            @ reference
        )
        value = dict(link=self.wrist.value, frame=local.tolist())

        def apply():
            self.change(self.d.editor.update, root=value)
            self.clear_layout()

        self.confirm(
            "Preserve all numerical bind poses and weights; recompute attachments at this wrist reference.",
            apply,
            cleanup=lambda: setattr(
                self.handles["/calibration-preview"], "visible", False
            ),
        )

    def joint_controls(self):
        if self.model and self.joint.value:
            j = self.model.urdf.joint_map[self.joint.value]
            low = j.limit.lower if j.limit and j.limit.lower is not None else -np.pi
            high = j.limit.upper if j.limit and j.limit.upper is not None else np.pi
            self.angle.min, self.angle.max = low, high
            self.angle_number.min = j.limit.lower if j.limit else None
            self.angle_number.max = j.limit.upper if j.limit else None
            self.angle_number.value = self.q.get(j.name, 0.0)
            self.angle.value = float(np.clip(self.angle_number.value, low, high))

    def preview_joint(self, value):
        if self.joint.value:
            self.q[self.joint.value] = value
            self.angle_number.value = value
            self.angle.value = float(np.clip(value, self.angle.min, self.angle.max))
            self.mode.value = "PREVIEW"

    def confirm_q0(self):
        q0 = copy.deepcopy(self.q)

        def apply():
            self.change(self.state.model.update, q0=q0)
            self.mode.value = "EDIT"
            self.clear_layout()

        self.confirm(
            "Update shared q0; bind poses and weights stay unchanged and points may leave the mesh. Affected hands: "
            + ", ".join(self.state.documents),
            apply,
        )

    def add_segment(self):
        initial = IDENTITY[:] if self.semantic.value == "WRIST" else None
        if self.semantic.value != "WRIST" and self.model and "root" in self.d.editor:
            initial = geo.pose(
                np.linalg.inv(self.root())
                @ self.model.fk(self.state.model.get("q0", {}))[self.segment_link.value]
            )

        def add():
            self.d.joints.append(self.semantic.value)
            self.d.arrays[FIELDS[0]].append(initial)
            self.d.sync()
            self.d.editor.setdefault("segment_links", {})[
                self.d.editor["segment_ids"][-1]
            ] = self.segment_link.value
            self.d.editor["approximate"] = True

        self.change(add)
        self.segment.value = str(len(self.d.arrays[FIELDS[0]]) - 1)
        self.select_segment()

    def associate(self):
        i = int(self.segment.value)

        def update():
            self.d.joints[i] = self.semantic.value
            self.d.editor.setdefault("segment_links", {})[
                self.d.editor["segment_ids"][i]
            ] = self.segment_link.value

        self.change(update)

    def select_segment(self):
        self.entity.value, self.slot = "Segment", None
        if self.segment.value:
            i = int(self.segment.value)
            if i < len(self.d.joints):
                self.semantic.value = JOINTS[joint_number(self.d.joints[i])]
            link = self.d.editor.get("segment_links", {}).get(
                self.d.editor["segment_ids"][i]
            )
            if self.model and link in self.model.links:
                self.segment_link.value = link

    def targets(self):
        if self.entity.value == "Segment" and self.segment.value:
            return FIELDS[0], [int(self.segment.value)]
        return FIELDS[1], self.selected

    def reorder_segments(self):
        ids = self.d.editor["segment_ids"]
        selected = (
            ids[int(self.segment.value)]
            if self.segment.value and int(self.segment.value) < len(ids)
            else None
        )
        self.change(self.d.segments, json.loads(self.order.value))
        ids = self.d.editor["segment_ids"]
        self.segment.value = str(ids.index(selected)) if selected in ids else ""

    def delete_taxels(self):
        self.change(self.d.delete, self.selected)
        self.selected = []

    def components(self, components, values):
        field, rows = self.targets()
        if not rows:
            raise ValueError("Select a draft segment or taxel first")
        self.change(self.d.set_components, field, rows, components, values)

    def rotate(self, angle, axis):
        field, rows = self.targets()
        if not rows:
            raise ValueError("Select a draft segment or taxel first")
        self.change(
            geo.transform,
            self.d,
            field,
            rows,
            tf.rotation_matrix(angle, axis),
            local=True,
        )

    # Inline async callbacks preserve Viser's drag phase ordering.
    async def gizmo_event(self, event):
        if event.phase == "start" and event.client:
            self.dragging = True
        elif event.phase == "end":
            if event.client:
                self.drag_end(event)
            else:
                self.dragging = False

    def drag_end(self, event):
        values = [*event.target.position, *np.asarray(event.target.wxyz)[[1, 2, 3, 0]]]

        def apply():
            try:
                field, rows = self.targets()
                if rows:
                    old = geo.matrix(self.d.arrays[field][rows[0]])
                    delta = (
                        np.linalg.inv(self.root())
                        @ geo.matrix(values)
                        @ np.linalg.inv(old)
                    )
                    self.change(geo.transform, self.d, field, rows, delta)
            finally:
                self.dragging = False

        self.events.put(apply)

    def set_slot(self, k, value):
        self.slot = k
        self.change(self.d.set_components, FIELDS[2], self.selected, [k], [value])

    def weight(self, k, value):
        self.slot = k
        rejected = self.change(self.d.weight, self.selected, k, value)
        if rejected:
            self.log("; ".join(rejected))

    def spans_apply(self):
        o = np.asarray(self.points["O"])
        for key, control, span in zip(("C", "R"), self.centers[1:], self.spans):
            if key in self.points:
                self.points[key] = (
                    o + geo.unit(np.asarray(self.points[key]) - o) * span.value
                ).tolist()
                control.value = tuple(self.points[key])

    def grid(self):
        if self.surface.value not in self.surfaces():
            raise ValueError("Choose an allowed projection surface")
        self.candidates = self.model.grid(
            self.d,
            self.model.fk(self.state.model.get("q0", {})),
            self.surface.value,
            [self.points.get(k) for k in ("O", "C", "R")],
            int(self.rows.value),
            int(self.cols.value),
            self.scan.value,
            [w.value for w in self.reverse],
            json.loads(self.excluded.value),
        )
        start = int(self.start.value)
        start = self.d.unused_start(len(self.candidates)) if start < 0 else start
        spans = [
            np.linalg.norm(np.asarray(self.points[k]) - self.points["O"])
            if k in self.points
            else 0
            for k in ("C", "R")
        ]
        spacing = [
            v / (n - 1) if n > 1 else 0
            for v, n in zip(spans, (self.cols.value, self.rows.value))
        ]
        self.layout_info.content = f"Spans C/R: {spans} m; spacing C/R: {spacing} m. {len(self.candidates)} candidates; {sum(p is None for p, _ in self.candidates)} projection failures. Output indices: {list(range(start, start + len(self.candidates)))}"

    def apply_grid(self):
        self.change(
            self.d.add,
            [p for p, _ in self.candidates],
            json.loads(self.influences.value),
            int(self.start.value),
        )
        self.clear_layout()

    def clear_layout(self):
        self.candidates, self.points = [], {}
        self.rows.value = self.cols.value = 4
        self.scan.value, self.start.value, self.excluded.value = "row-major", -1, "[]"
        for w in self.reverse:
            w.value = False
        for w in self.centers:
            w.value = (0.0, 0.0, 0.0)
        for w in self.spans:
            w.value = 0.03
        self.layout_info.content = ""

    def select_indices(self):
        indices = []
        for part in self.selection.value.split(","):
            ends = [int(v) for v in part.strip().split(":")]
            indices.extend(range(ends[0], ends[1] + 1) if len(ends) == 2 else ends)
        self.selected = [
            i
            for i, index in enumerate(self.d.editor["output_indices"])
            if index in indices
        ]
        self.entity.value = "Taxels"

    def reindex(self):
        values, order = (
            json.loads(self.indices.value),
            self.d.editor["output_indices"][:],
        )
        if len(values) != len(self.selected) or any(
            type(v) is not int or v < 0 for v in values
        ):
            raise ValueError("Supply one nonnegative integer per selected taxel")
        preview = [(order[i], value) for i, value in zip(self.selected, values)]
        for i, value in zip(self.selected, values):
            order[i] = value
        self.confirm(
            str(preview),
            lambda: self.change(self.d.editor.update, output_indices=order),
        )

    def reproject(self):
        root, fk = self.root(), self.model.fk(self.state.model.get("q0", {}))
        if self.surface.value not in self.surfaces():
            raise ValueError("Choose an allowed projection surface")

        def apply():
            for i in self.selected:
                old = root @ geo.matrix(self.d.arrays[FIELDS[1]][i])
                hit = self.model.hit(
                    old[:3, 3] + old[:3, 2], -old[:3, 2], [self.surface.value], fk
                )
                if not hit:
                    raise ValueError(
                        f"Taxel {i}: projection failed; selection unchanged"
                    )
                self.d.arrays[FIELDS[1]][i] = geo.pose(
                    np.linalg.inv(root) @ geo.frame(hit[1], hit[2], old[:3, 0])
                )
            self.d.editor["approximate"] = True

        self.change(apply)

    def configure_source(self):
        source = dict(
            collection=self.collection.value,
            hand=self.hand.value,
            joint_state=self.joint_id.value if self.joint_enabled.value else "",
        )
        if not source["collection"].strip() or (
            self.joint_enabled.value and not source["joint_state"].strip()
        ):
            raise ValueError("Enabled sources require collection_id values")
        self.change(self.d.editor.update, source=source)
        if self.receiver:
            self.log("Source configuration saved; reconnect to apply it")

    def mapping(self, remove=False):
        mappings = dict(self.d.editor.get("mapping", {}))
        if remove:
            mappings.pop(self.source_joint.value, None)
        else:
            if (
                not self.source_joint.value
                or not self.target_joint.value
                or not all(
                    math.isfinite(v) for v in (self.scale.value, self.offset.value)
                )
            ):
                raise ValueError("Choose source/target joints and finite scale/offset")
            mappings[self.source_joint.value] = [
                self.target_joint.value,
                self.scale.value,
                self.offset.value,
            ]
        self.change(self.d.editor.update, mapping=mappings)

    def connect(self):
        if self.receiver:
            if self.receiver is not True:
                self.receiver.close()
            self.receiver = None
            self.log("Disconnected")
        else:
            self.configure_source()
            self.configs = {
                s: dict(
                    d.editor.get(
                        "source", dict(collection="tactile", hand=s, joint_state="")
                    ),
                    mapping=d.editor.get("mapping", {}),
                )
                for s, d in self.state.documents.items()
            }
            self.receiver = True if self.args.demo else sdk.Receiver(self.configs)
            self.mode.value = "LIVE"

    def latest(self):
        return (
            sdk.demo(self.configs)
            if self.receiver is True
            else self.receiver.latest
            if self.receiver
            else {}
        )

    def record(self):
        value = self.latest().get(self.side.value, {}).get("metadata")
        if value is None:
            raise ValueError("No received metadata")
        received = storage.parse(value, self.side.value)
        path = storage.backup(self.d, self.state.model)
        self.log("Draft backup: " + path)
        self.change(
            self.d.record, received, [k for k, w in self.groups.items() if w.value]
        )

    def repair(self):
        d = storage.parse(json.loads(self.schema.value), self.side.value)
        self.log("Draft backup: " + storage.backup(self.d, self.state.model))
        self.change(self.state.documents.__setitem__, self.side.value, d)

    def local_check(self):
        documents = (
            self.state.documents if self.all_sides.value else {self.side.value: self.d}
        )
        self.log(str({s: d.check() for s, d in documents.items()}))

    def background(self, exporting):
        if self.job and not self.job.done():
            raise ValueError("A check/export is already running")
        documents = copy.deepcopy(
            self.state.documents if self.all_sides.value else {self.side.value: self.d}
        )
        self.job_revision = self.state.revision
        self.sdk_revision = self.job_revision
        if exporting:
            self.job = self.pool.submit(
                storage.export,
                documents,
                copy.deepcopy(self.state.model),
                self.job_revision,
                self.destination.value,
                self.binary.value,
            )
        else:
            self.job = self.pool.submit(
                lambda: {s: sdk.check(d) for s, d in documents.items()}
            )
        self.sdk_result = f"Running for revision {self.job_revision}"

    def refresh(self):
        d = self.d
        self.issues = [
            dict(v, side=s)
            for s, doc in self.state.documents.items()
            for v in doc.check()
        ]
        if self.model:
            fk = self.model.fk(self.state.model.get("q0", {}))
            for doc in self.state.documents.values():
                try:
                    if "root" in doc.editor:
                        self.model.attach(doc, fk)
                except (ValueError, KeyError, TypeError) as exc:
                    self.model.attachments.pop(doc.side, None)
                    self.issues.append(
                        dict(
                            side=doc.side,
                            severity="warning",
                            field="Model",
                            row=None,
                            message=str(exc),
                        )
                    )
            self.surface.options = tuple(self.surfaces()) or ("",)
        calibrated = bool(self.model and d.side in self.model.attachments)
        self.pick_button.disabled = self.calibrate_button.disabled = self.model is None
        self.grid_button.disabled = self.reproject_button.disabled = not calibrated
        self.angle.disabled = self.angle_number.disabled = self.model is None
        self.selected = [i for i in self.selected if 0 <= i < len(d.arrays[FIELDS[1]])]
        self.side.visible = self.both.visible = len(self.state.documents) == 2
        self.header.content = f"{Path(self.state.model.get('urdf', 'No URDF')).name}; hands: {', '.join(self.state.documents)}; {self.mode.value}\n\n{self.state.paths.get(d.side, 'Unsaved draft')}{' *' if self.state.dirty() else ''}; revision {self.state.revision}"
        self.segment.options = (
            "",
            *map(str, range(max(len(d.joints), len(d.arrays[FIELDS[0]])))),
        )
        links = d.editor.get("segment_links", {})
        self.segment_table.content = "\n\n".join(
            f"row {i}: {j}; link={links.get(sid, 'unassigned')}; attachment={'ready' if self.model and sid in self.model.attachments.get(d.side, {}) else 'missing'}"
            for i, (j, sid) in enumerate(zip(d.joints, d.editor["segment_ids"]))
        )
        first = int(self.page.value) * 25
        self.table.content = "\n\n".join(
            f"row {i}: output {d.editor['output_indices'][i]}; {p}"
            for i, p in enumerate(d.arrays[FIELDS[1]])
            if first <= i < first + 25
        )
        field, rows = self.targets()
        values = [
            d.arrays[field][i] if i < len(d.arrays[field]) else None for i in rows
        ]
        mixed = len({str(v) for v in values}) > 1
        self.raw.content = f"Draft {field}; rows {rows[:25]}; {'Multiple values' if mixed else str(values[:1])}. Set controls replace only the edited component."
        for k, w in enumerate(self.pose_fields):
            cells = [
                v[k] if isinstance(v, list) and k < len(v) else None for v in values
            ]
            if (
                cells
                and all(v == cells[0] for v in cells)
                and type(cells[0]) in (int, float)
                and abs(cells[0]) <= 3.4028234663852886e38
            ):
                w.value = float(cells[0])
        raw, sums = [], []
        for i in self.selected:
            parts = [
                d.arrays[f][i] if i < len(d.arrays[f]) else None for f in FIELDS[2:]
            ]
            raw.append((i, *parts))
            sums.append(
                sum(parts[1])
                if isinstance(parts[1], list)
                and all(type(v) in (int, float) for v in parts[1])
                else "invalid"
            )
        self.binding_info.content = f"{len(self.selected)} selected; weight sums (first 25): {sums[:25]}; {'Multiple values' if len({str(v[1:]) for v in raw}) > 1 else ''}\n\nRaw rows (index slots, weight slots, first 25): {raw[:25]}"
        for k, (dropdown, index, slider, number) in enumerate(self.slots):
            indices = [
                v[1][k] if isinstance(v[1], list) and len(v[1]) > k else None
                for v in raw
            ]
            weights = [
                v[2][k] if isinstance(v[2], list) and len(v[2]) > k else None
                for v in raw
            ]
            common = (
                indices[0]
                if indices and all(v == indices[0] for v in indices)
                else None
            )
            options = tuple(
                f"{i}: {d.joints[i] if i < len(d.joints) else 'missing joint_id'}"
                for i in range(len(d.arrays[FIELDS[0]]))
            )
            label = (
                options[common]
                if type(common) is int and 0 <= common < len(options)
                else f"Raw: {common if common is not None else 'mixed / missing'}"
            )
            dropdown.options, dropdown.value = (
                tuple(dict.fromkeys((label, *options))),
                label,
            )
            index.label = (
                "Set raw index"
                if type(common) is int
                else "Set raw index (mixed / missing)"
            )
            if type(common) is int and abs(common) <= 9007199254740991:
                index.value = common
            mixed_weight = not weights or any(v != weights[0] for v in weights)
            slider.label = number.label = (
                "Set weight (mixed / missing)" if mixed_weight else "Set weight"
            )
            if (
                weights
                and all(v == weights[0] for v in weights)
                and type(weights[0]) in (int, float)
                and abs(weights[0]) <= 3.4028234663852886e38
            ):
                number.value = float(weights[0])
                slider.value = float(np.clip(weights[0], 0, 1))
        self.info.options = tuple(
            f"{i}: {v['side']} {v['severity']} {v['field']}[{v['row']}]: {v['message']}"
            for i, v in enumerate(self.issues)
        ) or ("No local errors",)
        self.mapping_info.content = str(d.editor.get("mapping", {}))
        self.check_info.content = f"Local revision {self.state.revision}: {sum(v['severity'] == 'error' for v in self.issues)} errors. SDK/export revision {self.sdk_revision}: {self.sdk_result}{' (stale)' if self.sdk_revision is not None and self.sdk_revision != self.state.revision else ''}"

    def locate(self):
        if not self.issues:
            return
        issue = self.issues[int(self.info.value.split(":")[0])]
        if issue["side"] != self.side.value:
            self.side.value = issue["side"]
            self.switch()
        if issue["row"] is not None:
            if issue["field"] in (FIELDS[0], "joint_ids"):
                self.segment.value, self.entity.value = str(issue["row"]), "Segment"
            else:
                self.selected, self.entity.value = [issue["row"]], "Taxels"
        self.log("Located in draft inspector: " + issue["field"])

    def axes(self, name, transforms):
        positions = np.array([t[:3, 3] for t in transforms]).reshape(-1, 3)
        quaternions = np.array(
            [tf.quaternion_from_matrix(t) for t in transforms]
        ).reshape(-1, 4)
        if name not in self.handles and len(positions):
            self.handles[name] = self.server.scene.add_batched_axes(
                name, quaternions, positions, axes_length=0.015, axes_radius=0.0006
            )
        if name in self.handles:
            h = self.handles[name]
            h.visible = bool(len(positions))
            if len(positions):
                h.batched_positions, h.batched_wxyzs = positions, quaternions

    def balls(self, name, points, colors, outline=False):
        positions = np.asarray(points, dtype=float).reshape(-1, 3)
        rotations = np.tile([1.0, 0, 0, 0], (len(positions), 1))
        if name not in self.handles and len(positions):
            mesh = trimesh.creation.icosphere(subdivisions=1)
            self.handles[name] = self.server.scene.add_batched_meshes_simple(
                name, mesh.vertices, mesh.faces, rotations, positions, wireframe=outline
            )
        if name in self.handles:
            h = self.handles[name]
            h.visible = bool(len(positions))
            if len(positions):
                h.batched_positions, h.batched_wxyzs = positions, rotations
                h.batched_colors = np.asarray(colors, dtype=np.uint8)
                h.batched_scales = np.full(
                    len(positions), self.radius.value * (1.5 if outline else 1)
                )

    def draft_points(self):
        return [
            (i, t[:3, 3]) for i, t in geo.frames(self.d.arrays[FIELDS[1]], self.root())
        ]

    def rectangle_mode(self, enabled):
        if enabled:
            self.server.scene.on_rect_select()(self.box_event)
        else:
            self.server.scene.remove_rect_select_callback(self.box_event)

    async def box_event(self, event):
        self.events.put(lambda: self.select_scene(event, True))

    async def scene_click(self, event):
        self.events.put(lambda: self.click(event))

    def select_scene(self, event, box=False):
        camera, chosen = event.client.camera, []
        rotation = tf.quaternion_matrix(camera.wxyz)[:3, :3]
        for i, point in self.draft_points():
            origin = np.asarray(camera.position if box else event.ray_origin)
            direction = (
                geo.unit(point - origin) if box else np.asarray(event.ray_direction)
            )
            if box:
                p = rotation.T @ (point - origin)
                if p[2] >= 0:
                    continue
                uv = [
                    0.5 - p[0] / p[2] / (2 * np.tan(camera.fov / 2) * camera.aspect),
                    0.5 + p[1] / p[2] / (2 * np.tan(camera.fov / 2)),
                ]
                inside = all(
                    a <= b <= c
                    for a, b, c in zip(event.screen_min, uv, event.screen_max)
                )
            else:
                along = (point - origin) @ direction
                inside = (
                    along > 0
                    and np.linalg.norm(point - origin - along * direction)
                    < self.radius.value * 2
                )
            if (
                inside
                and self.model
                and getattr(self, "display_model", False)
                and not self.through.value
            ):
                surfaces = list(self.model.surfaces)
                hit = self.model.hit(origin, direction, surfaces, self.display_fk)
                inside = (
                    hit is None
                    or hit[0] >= np.linalg.norm(point - origin) - self.radius.value * 2
                )
            if inside:
                chosen.append((np.linalg.norm(point - origin), i))
        indices = [i for _, i in (chosen if box else sorted(chosen)[:1])]
        self.selected = (
            sorted(set(self.selected) ^ set(indices)) if event.modifier else indices
        )
        self.entity.value = "Taxels"

    def click(self, event):
        if self.dragging:
            return
        if not self.pick_target:
            self.select_scene(event)
            return
        target, self.pick_target = self.pick_target, None
        if self.mode.value != "EDIT" or not self.model:
            raise ValueError("Surface picking requires EDIT and a q0 model")
        fk = self.model.fk(self.state.model.get("q0", {}))
        surfaces = self.surfaces()
        if target in ("O", "C", "R", "Move selected", "Single tangent"):
            if self.surface.value not in surfaces:
                raise ValueError("Choose an allowed projection surface")
            surfaces = [self.surface.value]
        hit = self.model.hit(
            np.asarray(event.ray_origin),
            np.asarray(event.ray_direction),
            surfaces,
            fk,
        )
        if not hit:
            raise ValueError("No hit on an allowed visible surface")
        self.points[target] = hit[1].tolist()
        if target in ("O", "C", "R"):
            self.centers[("O", "C", "R").index(target)].value = tuple(hit[1])
        elif target == "Segment":
            self.entity.value = "Segment"
            self.components(
                range(7),
                geo.pose(
                    np.linalg.inv(self.root())
                    @ geo.frame(hit[1], hit[2], fk[self.segment_link.value][:3, 0])
                ),
            )
        elif target == "Move selected":
            old = geo.matrix(self.d.arrays[FIELDS[1]][self.selected[0]])
            new = np.linalg.inv(self.root()) @ geo.frame(
                hit[1], hit[2], (self.root() @ old)[:3, 0]
            )
            self.change(
                geo.transform,
                self.d,
                FIELDS[1],
                self.selected,
                new @ np.linalg.inv(old),
            )
        elif target == "Single tangent":
            self.points["C"] = hit[1].tolist()
            self.rows.value = self.cols.value = 1
            self.grid()

    def render_side(self, side, d, root, current, forces, live):
        positions, colors, frames, outlines, edges = [], [], [], [], []
        deformed = geo.deform(d, current) if current is not None else None
        errors = {
            v["row"]
            for v in d.check()
            if v["severity"] == "error" and v["field"] in FIELDS[1:]
        }
        for i, t in geo.frames(d.arrays[FIELDS[1]], root):
            rigid, color = t, (140, 140, 140)
            if deformed is not None:
                point, orientation = deformed[i]
                if point is None:
                    continue
                t[:3, 3] = (root @ np.r_[point, 1])[:3]
                rigid = root @ orientation if orientation is not None else None
            if live and forces is not None and len(forces) == len(d.arrays[FIELDS[1]]):
                force = np.asarray(forces[i], dtype=float)
                if force.shape == (3,) and np.isfinite(force).all():
                    fraction = np.clip(
                        np.linalg.norm(force) / self.force_max.value, 0, 1
                    )
                    color = (255 * fraction, 40, 255 * (1 - fraction))
            elif not live:
                try:
                    indices, weights = d.arrays[FIELDS[2]][i], d.arrays[FIELDS[3]][i]
                    if i not in errors:
                        color = np.random.default_rng(
                            indices[int(np.argmax(weights))]
                        ).integers(70, 240, 3)
                        if self.slot is not None and i in self.selected and d is self.d:
                            color = (
                                255 * weights[self.slot],
                                80,
                                255 * (1 - weights[self.slot]),
                            )
                except (IndexError, TypeError, ValueError):
                    pass
            positions.append(t[:3, 3])
            colors.append(color)
            selected = d is self.d and i in self.selected
            if selected or i in errors:
                outlines.append(t[:3, 3])
                edges.append((255, 0, 0) if i in errors else (255, 220, 0))
            if selected and rigid is not None:
                frames.append(rigid)
        prefix = "/view/" + side
        self.balls(prefix + "/taxels", positions, colors)
        self.balls(prefix + "/outlines", outlines, edges, True)
        self.axes(prefix + "/taxel-axes", frames)
        self.axes(prefix + "/root", [root])
        segments = (
            [root @ t for t in current if t is not None]
            if current is not None
            else [t for _, t in geo.frames(d.arrays[FIELDS[0]], root)]
        )
        self.axes(prefix + "/segments", segments)
        self.balls(
            prefix + "/segment-points",
            [t[:3, 3] for t in segments],
            [(90, 200, 250)] * len(segments),
        )
        self.frame_colors[side] = np.asarray(colors, dtype=np.uint8)

    def draw(self):
        latest, mode = self.latest(), self.mode.value
        playing, status = mode in ("LIVE", "PAUSED"), []
        cfg = self.q if mode == "PREVIEW" else self.state.model.get("q0", {})
        highlight, lines, line_colors = set(), [], []
        for side, draft in self.state.documents.items():
            value = latest.get(side, {})
            unmapped = set(value.get("joints", {})) - set(
                draft.editor.get("mapping", {})
            )
            status.append(
                f"{side} raw JointState: {value.get('joints', {})}; unmapped: {sorted(unmapped)}"
            )
            fresh = (
                bool(self.receiver)
                and (self.receiver is True or self.receiver.active)
                and time.monotonic() - value.get("read_time", 0) < 1
            )
            if (
                self.model
                and self.receiver
                and self.configs.get(side, {}).get("joint_state")
            ):
                for name, (target, scale, offset) in draft.editor.get(
                    "mapping", {}
                ).items():
                    raw, valid = value.get("joints", {}).get(name, (None, False))
                    converted = (
                        raw * scale + offset
                        if valid and raw is not None
                        else float("nan")
                    )
                    joint = self.model.urdf.joint_map.get(target)
                    limit = joint.limit if joint else None
                    valid = (
                        fresh and joint in self.model.joints and np.isfinite(converted)
                    )
                    if limit:
                        valid &= (limit.lower is None or converted >= limit.lower) and (
                            limit.upper is None or converted <= limit.upper
                        )
                    status.append(
                        f"{side}/{name}: raw={raw}; converted={converted}; limits={limit}; valid={valid}"
                    )
                    if valid and mode == "LIVE":
                        self.drive[target] = converted
            self.layers[side].visible = side == self.side.value or self.both.value
            if mode != "PAUSED":
                try:
                    root, d, current, skeleton = self.root(draft, cfg), draft, None, []
                    if mode == "LIVE":
                        if self.source.value == "Received":
                            d = storage.parse(value["metadata"], side)
                        current, joints, _ = geo.hand_current(d, value["hand"])
                        root = joints[1]
                        skeleton = [t for t in joints if t is not None]
                    elif mode == "PREVIEW" and self.model and "root" in draft.editor:
                        current = self.model.current(draft, self.model.fk(cfg))
                    self.render_side(
                        side,
                        d,
                        root,
                        current,
                        value.get("forces") if fresh else None,
                        playing,
                    )
                    self.axes("/view/" + side + "/live-skeleton", skeleton)
                    self.frame_forces[side] = value.get("forces")
                    self.frame_counts[side] = len(d.arrays[FIELDS[1]])
                except (ValueError, TypeError, KeyError, IndexError) as exc:
                    status.append(f"{side}: invalid visualization: {exc}")
                    fresh = False
            taxels = self.handles.get("/view/" + side + "/taxels")
            if playing and taxels and taxels.visible:
                taxels.batched_colors = (
                    self.frame_colors.get(side, (140, 140, 140))
                    if fresh
                    else (140, 140, 140)
                )
            forces = (
                self.frame_forces.get(side) if mode == "PAUSED" else value.get("forces")
            )
            force_array = (
                np.asarray(forces, dtype=float)
                if forces is not None
                else np.empty((0, 3))
            )
            valid_forces = (
                forces is not None
                and force_array.ndim == 2
                and force_array.shape[1:] == (3,)
                and np.isfinite(force_array).all()
                and len(force_array)
                == self.frame_counts.get(side, len(draft.arrays[FIELDS[1]]))
            )
            status.append(
                f"{side}: {value.get('config')}; local read={value.get('read_time')}; current={fresh}; force shape={force_array.shape}; valid/count-matched={valid_forces}; draft taxels={len(draft.arrays[FIELDS[1]])}"
            )
            hand_flags = (value.get("hand") or {}).get("is_valid", [])
            status.append(
                f"{side} HandPose: {sum(bool(v) for v in hand_flags)} valid joints; WRIST flag={len(hand_flags) == 26 and bool(hand_flags[1])}"
            )
            if side == self.side.value and fresh and valid_forces:
                status.extend(
                    f"{'Paused display' if mode == 'PAUSED' else 'Latest'} row {i}: fx,fy,fz={forces[i]}; |f|={np.linalg.norm(forces[i]):.4g} N"
                    for i in self.selected[:25]
                    if i < len(force_array)
                )
        root, d = self.root(), self.d
        points = self.draft_points() if mode != "EDIT" else []
        self.balls(
            "/draft-targets",
            [p for _, p in points],
            [
                (255, 220, 0) if i in self.selected else (140, 140, 140)
                for i, _ in points
            ],
            True,
        )
        field, rows = self.targets()
        self.axes(
            "/draft-axes",
            [t for i, t in geo.frames(d.arrays[field], root) if i in rows],
        )
        for i in self.selected:
            try:
                for j, w in zip(d.arrays[FIELDS[2]][i], d.arrays[FIELDS[3]][i]):
                    if (
                        type(j) is int
                        and 0 <= j < len(d.joints)
                        and type(w) in (int, float)
                        and w <= 1
                        and w > 0
                    ):
                        lines.append(
                            [
                                (root @ geo.matrix(d.arrays[FIELDS[0]][j]))[:3, 3],
                                (root @ geo.matrix(d.arrays[FIELDS[1]][i]))[:3, 3],
                            ]
                        )
                        line_colors.append([(int(255 * min(w, 1)), 80, 30)] * 2)
                        if (
                            self.slot is not None
                            and j == d.arrays[FIELDS[2]][i][self.slot]
                        ):
                            highlight.add(
                                d.editor.get("segment_links", {}).get(
                                    d.editor["segment_ids"][j]
                                )
                            )
            except (IndexError, TypeError, ValueError):
                pass
        if (
            self.entity.value == "Segment"
            and self.segment.value
            and int(self.segment.value) < len(d.editor["segment_ids"])
        ):
            highlight.add(
                d.editor.get("segment_links", {}).get(
                    d.editor["segment_ids"][int(self.segment.value)]
                )
            )
        if "/draft-binding" not in self.handles and lines:
            self.handles["/draft-binding"] = self.server.scene.add_line_segments(
                "/draft-binding",
                np.asarray(lines),
                np.asarray(line_colors, dtype=np.uint8),
                thickness=2,
                thickness_units="screen",
            )
        if "/draft-binding" in self.handles:
            h = self.handles["/draft-binding"]
            h.visible = bool(lines)
            if lines:
                h.points, h.colors = (
                    np.asarray(lines),
                    np.asarray(line_colors, dtype=np.uint8),
                )
        if self.model and mode != "PAUSED":
            enabled = any(
                bool(self.receiver) and self.configs.get(side, {}).get("joint_state")
                for side in self.state.documents
            )
            self.display_model = not playing or enabled
            self.display_fk = self.model.draw(
                self.server.scene,
                self.drive if playing else cfg,
                self.display_model,
                highlight - {None},
            )
        connection = (
            self.receiver.status
            if self.receiver and self.receiver is not True
            else "Demo"
            if self.receiver
            else "Disconnected"
        )
        self.live_info.content = connection + "\n\n" + "\n\n".join(status)
        candidates = [
            (root @ np.r_[geo.matrix(p)[:3, 3] if p else fallback, 1])[:3]
            for p, fallback in self.candidates
        ]
        self.balls(
            "/candidates",
            candidates,
            [(50, 220, 90) if p else (255, 0, 0) for p, _ in self.candidates],
        )
        if not self.dragging:
            try:
                t = root @ geo.matrix(d.arrays[field][rows[0]])
                self.gizmo.position, self.gizmo.wxyz = (
                    t[:3, 3],
                    tf.quaternion_from_matrix(t),
                )
                self.gizmo.visible = True
            except (ValueError, TypeError, IndexError):
                self.gizmo.visible = False

    def run(self):
        self.camera()
        last_error = None
        try:
            while not self.closing:
                try:
                    while not self.events.empty():
                        self.events.get()()
                        self.refresh()
                    if self.job and self.job.done():
                        job, self.job = self.job, None
                        self.sdk_revision = self.job_revision
                        try:
                            self.sdk_result = str(job.result())
                        except Exception as exc:
                            self.sdk_result = f"Failed: {exc}"
                        self.refresh()
                    with self.server.atomic():
                        self.draw()
                    last_error = None
                    time.sleep(1 / 30)
                except KeyboardInterrupt:
                    self.guard(lambda: setattr(self, "closing", True))
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    if message != last_error:
                        self.log(message)
                    last_error = message
                    time.sleep(0.1)
        finally:
            if self.receiver and self.receiver is not True:
                self.receiver.close()
            self.pool.shutdown(wait=True)
            self.server.stop()
