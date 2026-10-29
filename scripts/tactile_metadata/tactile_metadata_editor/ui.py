# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import json
import logging
import math
import queue
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from trimesh import transformations as tf
from viser import _messages

from . import geometry as geo, sdk, storage
from .document import Document, IDENTITY_POSE, JOINTS, Segment, State, Taxel

LOG = logging.getLogger("isaaccapture.tactile_metadata.editor")

SEGMENT_CUBE_SIDE_M = 0.004
TAXEL_SIDE_M = 0.002
URDF_OPACITY = 0.2
SEGMENT_FLAT_SHADING = True
VISUALIZER_CAST_SHADOW = False
VISUALIZER_RECEIVE_SHADOW = False
SEGMENT_COLOR = (200, 120, 0)
TAXEL_COLOR = (200, 120, 0)
SELECTED_COLOR = (0, 255, 0)
SIDE_HIGHLIGHT_COLOR = (0, 180, 255)
INACTIVE_COLOR = (200, 200, 200)
ERROR_COLOR = (255, 0, 0)
FORCE_MIN_COLOR = (0, 0, 255)
FORCE_MAX_COLOR = (255, 0, 0)
FORCE_MAX_N = 30.0
SELECTION_GIZMO_SCALE_M = 0.005
SELECTION_GIZMO_LINE_WIDTH = 2.5
ACTIVE_DATASET_OPACITY = 1
BINDING_OVERLAY_OPACITY = 0.4
SAVED_BINDING_OPACITY = 0.2


class Editor:
    CAMERA_POSITION = (0.3, 0.3, -0.4)
    CAMERA_LOOK_AT = (0.0, 0.08, 0.0)

    def __init__(self, state: State, args):
        self.state = state
        for document in state.documents.values():
            document.alignment = geo.VISUALIZATION_FROM_HAND.tolist()
        self.preview = {
            side: self._preview_from(document)
            for side, document in state.documents.items()
        }
        self.live_document: Document | None = None
        self.live_metadata_key = ""
        self.receiver: sdk.Receiver | None = None
        self.model: geo.Model | None = None
        self.urdf_alignment = geo.VISUALIZATION_FROM_URDF.copy()
        self.urdf_scale = 1.0
        self.events: queue.SimpleQueue = queue.SimpleQueue()
        self.scene_handles: dict[str, object] = {}
        self.segment_row_folder = None
        self.taxel_row_folder = None
        self.edit_joint_folder = None
        self.preview_joint_folder = None
        self.preview_segment_folder = None
        self.selection_folder = None
        self.segment_widgets, self.taxel_widgets = [], []
        self.edit_joint_widgets, self.preview_joint_widgets = {}, {}
        self.preview_segment_widgets = []
        self.selected_kind: str | None = None
        self.selected_rows: list[int] = []
        self.binding_phase: str | None = None
        self.binding_taxels: list[int] = []
        self.binding_segments: list[int] = []
        self.pick_mode: str | None = None
        self.preview_has_errors = False
        self.selection_rotation = np.zeros(3)
        self.selection_scale = 1.0
        self.selection_gizmo_dragging = False
        self.selection_gizmo_last_position: np.ndarray | None = None
        self.redraw = True
        self.server = viser.ViserServer(host="0.0.0.0", port=args.port)
        self.server.scene.set_up_direction("+y")
        for script in ("editor_interactions.js", "viewport_axes.js"):
            source = Path(__file__).with_name(script).read_text()
            self.server._websock_server.queue_message(
                _messages.RunJavascriptMessage(source=source)
            )
        self.server.initial_camera.position = self.CAMERA_POSITION
        self.server.initial_camera.look_at = self.CAMERA_LOOK_AT
        self.server.gui.configure_theme(control_width="medium", show_share_button=False)
        self.server.gui.main_panel.dock_right()
        self._build_gui()
        self._register_scene_events()
        self.selection_gizmo = self.server.scene.add_transform_controls(
            "/selection/translation",
            scale=SELECTION_GIZMO_SCALE_M,
            line_width=SELECTION_GIZMO_LINE_WIDTH,
            disable_sliders=True,
            disable_rotations=True,
            depth_test=False,
            visible=False,
        )
        self.selection_gizmo.on_update(self._selection_gizmo_event)
        self._sync_side()
        if args.urdf:
            self.urdf_path.value = str(Path(args.urdf).expanduser().resolve())
            self._load_urdf()

    @property
    def document(self) -> Document:
        return self.state.document

    @property
    def preview_document(self) -> Document:
        return self.preview[self.state.side]

    @staticmethod
    def _preview_from(document: Document) -> Document:
        preview = document.clone()
        preview.taxels = []
        return preview

    def _build_gui(self) -> None:
        gui = self.server.gui
        with gui.add_folder("General"):
            self.side_text = gui.add_markdown("")
            self._action_button("Toggle Side", self._toggle_side)
            self.metadata_path = gui.add_text("Metadata Path", "")
            self._action_button("New", self._new)
            self._action_button("Save", self._save)
            self._action_button("Load", self._load)
            self._action_button("Reset Camera", self._reset_camera)
            with gui.add_folder("Mode"):
                self.mode_edit = gui.add_checkbox("Edit", True)
                self.mode_preview = gui.add_checkbox("Preview", False)
            self.show_saved = gui.add_checkbox("Show Saved Binding Poses", False)
            self._watch(self.mode_edit, lambda checked: self._set_mode("Edit", checked))
            self._watch(
                self.mode_preview, lambda checked: self._set_mode("Preview", checked)
            )
            self._watch(self.show_saved, lambda _: self._request_draw())

        self.edit_tabs = gui.add_tab_group()
        self.urdf_tab = self.edit_tabs.add_tab("URDF")
        with self.urdf_tab:
            self.urdf_path = gui.add_text("URDF Path", "")
            self._action_button("Load URDF", self._load_urdf)
            with gui.add_folder("URDF Alignment"):
                self.urdf_position = self._numbers("Position", (0.0, 0.0, 0.0), 0.001)
                self.urdf_rotation = self._sliders("Rotation", (0.0, 0.0, 0.0))
                self.urdf_scale_widget = gui.add_number(
                    "Scale", 1.0, min=1e-6, step=0.01
                )
                for widget in (*self.urdf_position, *self.urdf_rotation):
                    self._watch(widget, lambda _: self._update_urdf_alignment())
                self._watch(
                    self.urdf_scale_widget, lambda _: self._update_urdf_alignment()
                )
            self.edit_joint_folder = gui.add_folder(
                "URDF Joints", expand_by_default=False
            )

        self.segments_tab = self.edit_tabs.add_tab("Segments")
        with self.segments_tab:
            with gui.add_folder("Segments Alignment"):
                self.segment_position = self._numbers(
                    "Position", (0.0, 0.0, 0.0), 0.001
                )
                self.segment_rotation = self._sliders("Rotation", (0.0, 0.0, 0.0))
                for widget in (*self.segment_position, *self.segment_rotation):
                    self._watch(widget, lambda _: self._update_segment_alignment())
            with gui.add_folder("Add Segment"):
                self.new_segment_joint = gui.add_dropdown("Name", tuple(JOINTS))
                self.new_segment_position = self._numbers(
                    "Position", (0.0, 0.0, 0.0), 0.001
                )
                self._action_button("Add with Input", self._add_segment_input)
                self.segment_click_button = self._action_button(
                    "Add with URDF Click", self._pick_segment
                )
            self.segment_row_folder = gui.add_folder(
                "All Segments", expand_by_default=False
            )

        self.taxels_tab = self.edit_tabs.add_tab("Taxels")
        with self.taxels_tab:
            with gui.add_folder("Add Taxels"):
                self.grid_columns = gui.add_number("Columns", 4, min=1, step=1)
                self.grid_rows = gui.add_number("Rows", 4, min=1, step=1)
                self.grid_width = gui.add_number(
                    "Area Width (m)", 0.03, min=0.0, step=0.001
                )
                self.grid_height = gui.add_number(
                    "Area Height (m)", 0.03, min=0.0, step=0.001
                )
                self.grid_center = self._numbers(
                    "Center Position", (0.0, 0.0, 0.0), 0.001
                )
                self.grid_rotation = self._sliders("Rotation", (0.0, 0.0, 0.0))
                self.taxel_capture_button = self._action_button(
                    "Capture URDF Surface Pose", self._capture_taxel_pose
                )
                self._action_button("Add Taxels", self._add_taxels_input)
            with gui.add_folder("Bind Taxels to Segments"):
                self.binding_start = gui.add_button("Start Binding to Segments")
                gui.add_html(
                    '<div style="text-align:center;font-size:1.4em;line-height:1">↓</div>'
                )
                self.binding_next = gui.add_button("Select Segments", disabled=True)
                gui.add_html(
                    '<div style="text-align:center;font-size:1.4em;line-height:1">↓</div>'
                )
                self.binding_complete = gui.add_button("Complete", disabled=True)
                self.binding_info = gui.add_markdown("")
                self._button(self.binding_start, self._start_binding)
                self._button(self.binding_next, self._select_binding_segments)
                self._button(self.binding_complete, self._complete_binding)
            self.taxel_row_folder = gui.add_folder(
                "All Taxels", expand_by_default=False
            )

        self.preview_panel = gui.add_folder("Preview", visible=False)
        with self.preview_panel:
            with gui.add_folder("Binding Metadata Source"):
                self.binding_source_editor = gui.add_checkbox("Editor", True)
                self.binding_source_live = gui.add_checkbox(
                    "Live", False, disabled=True
                )
            self.show_binding = gui.add_checkbox("Show Binding Poses", False)
            self.collection = gui.add_text("Tactile Collection ID", "tactile")
            self.apply_segments_button = self._action_button(
                "Apply Segments", self._apply_preview_segments
            )
            self.apply_taxels_button = self._action_button(
                "Apply Taxels", self._apply_preview_taxels
            )
            self.apply_urdf_button = self._action_button(
                "Apply URDF", self._apply_preview_urdf
            )
            self.apply_all_button = self._action_button(
                "Apply ALL", self._apply_preview_all
            )
            gui.add_markdown(
                "Apply ALL fully overrides the editor's segments, taxels, bindings, and URDF joint values."
            )
            self._watch(
                self.binding_source_editor,
                lambda checked: self._set_binding_source("Editor", checked),
            )
            self._watch(
                self.binding_source_live,
                lambda checked: self._set_binding_source("Live", checked),
            )
            self._watch(self.show_binding, lambda _: self._request_draw())
            with gui.add_folder("Live"):
                self.live_status = gui.add_markdown("Current Statue: Disconnected")
                self.joint_enabled = gui.add_checkbox(
                    "Use JointState to Drive URDF", False
                )
                self.joint_collection = gui.add_text(
                    "JointState Collection ID", "", disabled=True
                )
                self.joint_mapping = gui.add_text(
                    "Joint Mappings JSON", "{}", multiline=True, disabled=True
                )
                self._watch(self.joint_enabled, self._joint_enabled)
                self.connect_button = self._action_button("Connect Live", self._connect)
                gui.add_markdown("Live input overrides values in the Static section.")
            self.static_folder = gui.add_folder("Static")
            with self.static_folder:
                self._action_button(
                    "Reset Values",
                    self._restore_preview,
                )
                gui.add_markdown("The following values affect Preview only.")
                self.preview_segment_folder = gui.add_folder(
                    "Segments", expand_by_default=False
                )
                self.preview_joint_folder = gui.add_folder(
                    "URDF Joints", expand_by_default=False
                )

        self.info = gui.add_text("__Tactile Info", "")
        self.preview_error = self.info

        self.selection_panel = gui.add_panel()
        self.selection_tab = self.selection_panel.add_tab("Selected Object")
        with self.selection_tab:
            self.selection_folder = gui.add_folder("Selection")
        self.selection_panel.float(x=10, y=140, width=320, height=520)

    def _numbers(self, prefix: str, values, step: float):
        return tuple(
            self.server.gui.add_number(f"{prefix} {axis}", float(value), step=step)
            for axis, value in zip("XYZ", values)
        )

    def _sliders(self, prefix: str, values):
        return tuple(
            self.server.gui.add_slider(
                f"{prefix} {axis} (deg)", -180.0, 180.0, 1.0, float(value)
            )
            for axis, value in zip("XYZ", values)
        )

    def _watch(self, widget, callback) -> None:
        async def updated(event):
            if event.client is not None:
                value = copy.deepcopy(event.target.value)
                self.events.put(lambda: self._run_action(callback, value))

        widget.on_update(updated)

    def _button(self, widget, callback) -> None:
        async def clicked(_):
            self.events.put(lambda: self._run_action(callback))

        widget.on_click(clicked)

    def _action_button(self, label: str, callback, *, order=None):
        button = self.server.gui.add_button(label, order=order)

        async def clicked(_):
            self.events.put(lambda: self._run_action(callback))

        button.on_click(clicked)
        return button

    def _run_action(self, callback, value=None) -> None:
        self.info.value = ""
        try:
            callback() if value is None else callback(value)
        except Exception as exc:
            LOG.exception("Editor action failed")
            self.info.value = f'<span style="color:#d32f2f">Error: {exc}</span>'
        else:
            if self.mode_value == "Preview" and self.preview_has_errors:
                self._update_preview_error()
        self.redraw = True

    def _request_draw(self) -> None:
        self.redraw = True

    @property
    def mode_value(self) -> str:
        return "Edit" if self.mode_edit.value else "Preview"

    def _set_mode(self, mode: str, checked: bool) -> None:
        selected = self.mode_edit if mode == "Edit" else self.mode_preview
        other = self.mode_preview if mode == "Edit" else self.mode_edit
        if not checked:
            if not other.value:
                selected.value = True
            return
        other.value = False
        self.edit_tabs.visible = mode == "Edit"
        self.preview_panel.visible = mode == "Preview"
        self.pick_mode = None
        self.selected_kind, self.selected_rows = None, []
        self._cancel_binding()
        self._rebuild_selection()
        if mode == "Preview":
            self._update_preview_error()

    def _toggle_side(self) -> None:
        self._disconnect()
        self.state.side = "right" if self.state.side == "left" else "left"
        self._sync_side()

    def _sync_side(self) -> None:
        self.live_document, self.live_metadata_key = None, ""
        self.pick_mode = None
        self.segment_click_button.label = "Add with URDF Click"
        self.taxel_capture_button.label = "Capture URDF Surface Pose"
        self.binding_source_editor.value = True
        self.binding_source_live.value = False
        self.binding_source_live.disabled = True
        self.side_text.content = f"**Side: {self.state.side.title()}**"
        self.metadata_path.value = self.state.paths.get(self.state.side, "")
        position, rotation = geo.components(
            np.asarray(self.document.alignment, dtype=float)
        )
        self._set_values(self.segment_position, position)
        self._set_values(self.segment_rotation, rotation)
        urdf_position, urdf_rotation = geo.components(self.urdf_alignment)
        self._set_values(self.urdf_position, urdf_position)
        self._set_values(self.urdf_rotation, urdf_rotation)
        self.urdf_scale_widget.value = self.urdf_scale
        self.selected_kind, self.selected_rows = None, []
        self._cancel_binding()
        self._rebuild_edit_lists()
        self._rebuild_joint_controls()
        self._rebuild_preview_segments()
        self._rebuild_selection()
        self.redraw = True

    @staticmethod
    def _set_values(widgets, values) -> None:
        for widget, value in zip(widgets, values):
            widget.value = float(value)

    def _new(self) -> None:
        document = Document(self.state.side)
        document.alignment = geo.VISUALIZATION_FROM_HAND.tolist()
        document.segments = geo.default_segments(self.state.side)
        self.state.documents[self.state.side] = document
        self.state.saved[self.state.side] = None
        self.state.paths.pop(self.state.side, None)
        self.preview[self.state.side] = self._preview_from(document)
        self._sync_side()
        self.info.value = f"New {self.state.side} metadata created."

    def _load(self) -> None:
        path = self.metadata_path.value.strip()
        if not path:
            raise ValueError("Enter Metadata Path")
        document = storage.load(path, self.state.side)
        self.state.replace(document, path)
        self.preview[self.state.side] = self._preview_from(document)
        self._sync_side()

    def _save(self) -> None:
        path = self.metadata_path.value.strip()
        if not path:
            raise ValueError("Enter Metadata Path")
        issues = self.document.validate()
        if not issues:
            self._write_save(path)
            return
        modal = self.server.gui.add_modal("Save Metadata")
        with modal:
            self.server.gui.add_markdown(
                f"Metadata contain errors, are you sure to save? ({len(issues)} error(s))"
            )
            save_button = self.server.gui.add_button("Continue")
            cancel_button = self.server.gui.add_button("Cancel")

        async def save(_):
            self.events.put(
                lambda: self._run_action(
                    lambda: (self._write_save(path), modal.close())
                )
            )

        async def cancel(_):
            self.events.put(modal.close)

        save_button.on_click(save)
        cancel_button.on_click(cancel)

    def _write_save(self, path: str) -> None:
        storage.save(self.state, path)
        self.metadata_path.value = self.state.paths[self.state.side]
        self.redraw = True

    def _reset_camera(self) -> None:
        for client in self.server.get_clients().values():
            client.camera.position = self.CAMERA_POSITION
            client.camera.look_at = self.CAMERA_LOOK_AT

    def _load_urdf(self) -> None:
        path = self.urdf_path.value.strip()
        if not path:
            raise ValueError("Enter URDF Path")
        if self.model is not None:
            self.model.remove()
        self.model = geo.Model(path)
        self._rebuild_joint_controls()

    def _update_urdf_alignment(self) -> None:
        self.urdf_alignment = geo.transform(
            [widget.value for widget in self.urdf_position],
            [widget.value for widget in self.urdf_rotation],
        )
        self.urdf_scale = float(self.urdf_scale_widget.value)
        if not math.isfinite(self.urdf_scale) or self.urdf_scale <= 0:
            raise ValueError("URDF Scale must be positive")
        self.redraw = True

    def _update_segment_alignment(self) -> None:
        alignment = geo.transform(
            [widget.value for widget in self.segment_position],
            [widget.value for widget in self.segment_rotation],
        )
        delta = geo.realign(self.document, alignment)
        saved = self.state.saved[self.state.side]
        if saved is not None:
            geo.apply_delta(saved.segments, list(range(len(saved.segments))), delta)
            geo.apply_delta(saved.taxels, list(range(len(saved.taxels))), delta)
            saved.alignment = alignment.tolist()
        preview = self.preview_document
        geo.apply_delta(preview.segments, list(range(len(preview.segments))), delta)
        preview.alignment = alignment.tolist()
        if self.live_document is not None:
            geo.apply_delta(
                self.live_document.segments,
                list(range(len(self.live_document.segments))),
                delta,
            )
            geo.apply_delta(
                self.live_document.taxels,
                list(range(len(self.live_document.taxels))),
                delta,
            )
            self.live_document.alignment = alignment.tolist()
        self._rebuild_edit_lists()
        self._rebuild_preview_segments()

    def _rebuild_joint_controls(self) -> None:
        if self.edit_joint_folder is not None:
            self.edit_joint_folder.remove()
        if self.preview_joint_folder is not None:
            self.preview_joint_folder.remove()
        self.edit_joint_widgets, self.preview_joint_widgets = {}, {}
        with self.urdf_tab:
            self.edit_joint_folder = self.server.gui.add_folder(
                "URDF Joints", expand_by_default=False
            )
            with self.edit_joint_folder:
                self._add_joint_sliders(self.document, self.edit_joint_widgets)
        with self.static_folder:
            self.preview_joint_folder = self.server.gui.add_folder(
                "URDF Joints", expand_by_default=False
            )
            with self.preview_joint_folder:
                self._add_joint_sliders(
                    self.preview_document, self.preview_joint_widgets
                )

    def _add_joint_sliders(self, document: Document, output: dict) -> None:
        if self.model is None:
            self.server.gui.add_markdown("Load a URDF to edit joints.")
            return
        for index, joint in enumerate(self.model.joints):
            lower = (
                joint.limit.lower
                if joint.limit and joint.limit.lower is not None
                else -math.pi
            )
            upper = (
                joint.limit.upper
                if joint.limit and joint.limit.upper is not None
                else math.pi
            )
            value = min(upper, max(lower, document.q.get(joint.name, 0.0)))
            with self.server.gui.add_folder(
                f"{index}: {joint.name}", expand_by_default=False
            ):
                widget = self.server.gui.add_slider(
                    "Value",
                    float(lower),
                    float(upper),
                    0.001,
                    float(value),
                )
            self._watch(
                widget,
                lambda current, name=joint.name, document=document: (
                    document.q.__setitem__(name, float(current))
                ),
            )
            output[joint.name] = widget

    def _add_segment_input(self) -> None:
        pose = IDENTITY_POSE.copy()
        pose[:3] = [float(widget.value) for widget in self.new_segment_position]
        row = self.document.add_segment(self.new_segment_joint.value, pose)
        self.selected_kind, self.selected_rows = "segments", [row]
        self._rebuild_edit_lists()
        self._rebuild_selection()

    def _pick_segment(self) -> None:
        self.pick_mode = None if self.pick_mode == "segment" else "segment"
        self.segment_click_button.label = (
            "● Add with URDF Click" if self.pick_mode else "Add with URDF Click"
        )

    def _grid_values(self):
        center = [float(widget.value) for widget in self.grid_center]
        return (
            center,
            float(self.grid_width.value),
            float(self.grid_height.value),
            int(self.grid_rows.value),
            int(self.grid_columns.value),
            geo.transform(center, [widget.value for widget in self.grid_rotation]),
        )

    def _add_taxels_input(self) -> None:
        center, width, height, rows, columns, frame = self._grid_values()
        selected = self.document.add_taxels(
            geo.grid_poses(center, width, height, rows, columns, frame)
        )
        self.selected_kind, self.selected_rows = "taxels", selected
        self._rebuild_edit_lists()
        self._rebuild_selection()

    def _capture_taxel_pose(self) -> None:
        self.pick_mode = None if self.pick_mode == "taxel_pose" else "taxel_pose"
        self.taxel_capture_button.label = (
            "● Capture URDF Surface Pose"
            if self.pick_mode
            else "Capture URDF Surface Pose"
        )

    def _rebuild_edit_lists(self) -> None:
        self._rebuild_segment_list()
        self._rebuild_taxel_list()
        self._rebuild_selection()
        self._update_preview_error()
        self.redraw = True

    def _rebuild_segment_list(self) -> None:
        if self.segment_row_folder is not None:
            self.segment_row_folder.remove()
        self.segment_widgets = []
        issues = self.document.validate()
        with self.segments_tab:
            self.segment_row_folder = self.server.gui.add_folder(
                "All Segments", expand_by_default=False
            )
            with self.segment_row_folder:
                for row, segment in enumerate(self.document.segments):
                    row_issues = [
                        issue
                        for issue in issues
                        if issue["kind"] == "segments" and issue["row"] == row
                    ]
                    with self.server.gui.add_folder(
                        f"{row}: {segment.joint}", expand_by_default=False
                    ):
                        joint = self.server.gui.add_dropdown(
                            "⚠ Name"
                            if any(issue["field"] == "joint" for issue in row_issues)
                            else "Name",
                            tuple(JOINTS),
                            initial_value=segment.joint,
                        )
                        position, rotation = geo.components(geo.matrix(segment.pose))
                        positions = self._numbers("Position", position, 0.001)
                        rotations = self._sliders("Rotation", rotation)
                        self._watch(
                            joint,
                            lambda value, row=row: self._set_segment_joint(row, value),
                        )
                        for axis, widget in enumerate(positions):
                            self._watch(
                                widget,
                                lambda value, row=row, axis=axis: (
                                    self._set_pose_component(
                                        "segments", row, axis, value
                                    )
                                ),
                            )
                        for axis, widget in enumerate(rotations):
                            self._watch(
                                widget,
                                lambda value, row=row, axis=axis: (
                                    self._set_pose_rotation(
                                        "segments", row, axis, value
                                    )
                                ),
                            )
                        messages = [issue["message"] for issue in row_issues]
                        if messages:
                            self.server.gui.add_html(
                                '<div style="color:#d32f2f;border:1px solid #d32f2f;padding:4px">'
                                + "<br>".join(messages)
                                + "</div>"
                            )
                        self.segment_widgets.append((joint, positions, rotations))

    def _rebuild_taxel_list(self) -> None:
        if self.taxel_row_folder is not None:
            self.taxel_row_folder.remove()
        self.taxel_widgets = []
        issues = self.document.validate()
        options = ("None", *[segment.joint for segment in self.document.segments])
        with self.taxels_tab:
            self.taxel_row_folder = self.server.gui.add_folder(
                "All Taxels", expand_by_default=False
            )
            with self.taxel_row_folder:
                for row, taxel in enumerate(self.document.taxels):
                    row_issues = [
                        issue
                        for issue in issues
                        if issue["kind"] == "taxels" and issue["row"] == row
                    ]
                    with self.server.gui.add_folder(
                        f"Taxel {row}", expand_by_default=False
                    ):
                        position, rotation = geo.components(geo.matrix(taxel.pose))
                        positions = self._numbers("Position", position, 0.001)
                        rotations = self._sliders("Rotation", rotation)
                        for axis, widget in enumerate(positions):
                            self._watch(
                                widget,
                                lambda value, row=row, axis=axis: (
                                    self._set_pose_component("taxels", row, axis, value)
                                ),
                            )
                        for axis, widget in enumerate(rotations):
                            self._watch(
                                widget,
                                lambda value, row=row, axis=axis: (
                                    self._set_pose_rotation("taxels", row, axis, value)
                                ),
                            )
                        slots = []
                        for slot in range(4):
                            with self.server.gui.add_folder(
                                f"Binding {slot}", expand_by_default=False
                            ):
                                selected = (
                                    taxel.segments[slot]
                                    if taxel.segments[slot] in options
                                    else "None"
                                )
                                segment = self.server.gui.add_dropdown(
                                    "⚠ Segment"
                                    if any(
                                        issue["field"] in (f"segment{slot}", "binding")
                                        for issue in row_issues
                                    )
                                    else "Segment",
                                    options,
                                    initial_value=selected,
                                )
                                weight = self._weight_widget(
                                    taxel.weights[slot],
                                    selected == "None",
                                    any(
                                        issue["field"] in (f"weight{slot}", "weights")
                                        for issue in row_issues
                                    ),
                                )
                                self._watch(
                                    segment,
                                    lambda value, row=row, slot=slot: (
                                        self._set_taxel_segment(row, slot, value)
                                    ),
                                )
                                self._watch(
                                    weight,
                                    lambda value, row=row, slot=slot: (
                                        self._set_taxel_weight(row, slot, value)
                                    ),
                                )
                                slots.append((segment, weight))
                        messages = [issue["message"] for issue in row_issues]
                        if messages:
                            self.server.gui.add_html(
                                '<div style="color:#d32f2f;border:1px solid #d32f2f;padding:4px">'
                                + "<br>".join(dict.fromkeys(messages))
                                + "</div>"
                            )
                        self.taxel_widgets.append((positions, rotations, slots))

    def _weight_widget(self, value, disabled: bool, invalid: bool):
        label = "⚠ Weight" if invalid else "Weight"
        if type(value) in (int, float) and math.isfinite(value):
            limits = {"min": 0.0, "max": 1.0} if 0.0 <= value <= 1.0 else {}
            return self.server.gui.add_number(
                label, float(value), step=0.01, disabled=disabled, **limits
            )
        return self.server.gui.add_text(label, str(value), disabled=disabled)

    def _set_segment_joint(self, row: int, value: str) -> None:
        self.document.segments[row].joint = value
        self._rebuild_edit_lists()

    def _values(self, kind: str):
        return self.document.segments if kind == "segments" else self.document.taxels

    def _set_pose_component(self, kind: str, row: int, axis: int, value) -> None:
        self._values(kind)[row].pose[axis] = float(value)
        self.redraw = True

    def _set_pose_rotation(self, kind: str, row: int, axis: int, value) -> None:
        current = geo.matrix(self._values(kind)[row].pose)
        position, rotation = geo.components(current)
        rotation = list(rotation)
        rotation[axis] = float(value)
        self._values(kind)[row].pose = geo.pose(geo.transform(position, rotation))
        self.redraw = True

    def _set_taxel_segment(self, row: int, slot: int, value: str) -> None:
        taxel = self.document.taxels[row]
        taxel.segments[slot] = None if value == "None" else value
        if taxel.segments[slot] is None:
            taxel.weights[slot] = 0.0
        self._rebuild_taxel_list()

    def _set_taxel_weight(self, row: int, slot: int, value) -> None:
        try:
            self.document.set_weight(row, slot, float(value))
        finally:
            self._rebuild_taxel_list()

    def _start_binding(self) -> None:
        self.pick_mode = None
        self.segment_click_button.label = "Add with URDF Click"
        self.taxel_capture_button.label = "Capture URDF Surface Pose"
        self.binding_phase = "taxels"
        self.binding_taxels, self.binding_segments = [], []
        self.selected_kind, self.selected_rows = None, []
        self._set_binding_button(None)
        self.binding_info.content = (
            "Select taxel(s) to bind in 3D scene. Hold Shift to multi-select. "
            'Click "Select Segments" after selecting taxels. Click another area to cancel.'
        )

    def _select_binding_segments(self) -> None:
        if not self.binding_taxels:
            raise ValueError("Select at least one taxel")
        self.binding_phase = "segments"
        self._set_binding_button(None)
        self.binding_info.content = (
            'Select 1-4 segment(s). Hold Shift to multi-select. Click "Complete" '
            "after selecting segments. Click another area to cancel."
        )

    def _complete_binding(self) -> None:
        if not 1 <= len(self.binding_segments) <= 4:
            self._cancel_binding()
            self.info.value = (
                '<span style="color:#d32f2f">Too many segments selected. '
                'Select 1-4 segments and click "Start Binding to Segments" '
                "to start over.</span>"
            )
            return
        self.document.assign(self.binding_taxels, self.binding_segments)
        self._cancel_binding()
        self._rebuild_taxel_list()

    def _set_binding_button(self, active: str | None) -> None:
        self.binding_start.disabled = active != "start"
        self.binding_next.disabled = active != "next"
        self.binding_complete.disabled = active != "complete"

    def _cancel_binding(self) -> None:
        self.binding_phase = None
        self.binding_taxels, self.binding_segments = [], []
        if hasattr(self, "binding_next"):
            self._set_binding_button("start")
            self.binding_info.content = ""

    def _joint_enabled(self, enabled: bool) -> None:
        self.joint_collection.disabled = not enabled
        self.joint_mapping.disabled = not enabled

    def _connect(self) -> None:
        if self.receiver is not None:
            self._disconnect()
            return
        config = {
            "side": self.state.side,
            "collection": self.collection.value,
            "joint_state": self.joint_collection.value
            if self.joint_enabled.value
            else "",
        }
        self.receiver = sdk.Receiver(config)
        self.connect_button.label = "Disconnect Live"
        self.live_status.content = "Current Status: Connecting"

    def _disconnect(self) -> None:
        if self.receiver is not None:
            self.receiver.close()
            self.receiver = None
        self.connect_button.label = "Connect Live"
        self.live_status.content = "Current Status: Disconnected"
        self._update_preview_error()
        self.redraw = True

    @property
    def binding_source_value(self) -> str:
        return "Live" if self.binding_source_live.value else "Editor"

    def _set_binding_source(self, source: str, checked: bool) -> None:
        selected = (
            self.binding_source_editor
            if source == "Editor"
            else self.binding_source_live
        )
        other = (
            self.binding_source_live
            if source == "Editor"
            else self.binding_source_editor
        )
        if not checked:
            if not other.value:
                selected.value = True
            return
        other.value = False
        self._update_preview_error()
        self.redraw = True

    def _binding_document(self) -> Document:
        if self.binding_source_value == "Live":
            if self.live_document is None:
                raise ValueError("Live metadata is unavailable")
            return self.live_document
        return self.document

    def _preview_taxels(self) -> tuple[list[Taxel | None], list[str]]:
        return geo.deform(self._binding_document(), self.preview_document.segments)

    def _rendered_preview_taxels(
        self,
    ) -> tuple[list[tuple[int, Taxel]], list[str], int]:
        try:
            rows, errors = self._preview_taxels()
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            return [], [f"Binding error: {exc}"], 0
        return (
            [(row, taxel) for row, taxel in enumerate(rows) if taxel is not None],
            errors,
            len(rows),
        )

    def _update_preview_error(self) -> None:
        _, errors, _ = self._rendered_preview_taxels()
        self.preview_has_errors = bool(errors)
        self.preview_error.value = (
            '<span style="color:#d32f2f">' + "<br>".join(errors) + "</span>"
            if errors
            else ""
        )
        self.apply_taxels_button.disabled = False
        self.apply_all_button.disabled = False
        self.apply_segments_button.disabled = False
        self.apply_urdf_button.disabled = False

    def _apply_preview_segments(self) -> None:
        self.document.segments = copy.deepcopy(self.preview_document.segments)
        self._rebuild_edit_lists()

    def _apply_preview_taxels(self) -> None:
        rows, _, _ = self._rendered_preview_taxels()
        self.document.taxels = copy.deepcopy([taxel for _, taxel in rows])
        self._rebuild_edit_lists()

    def _apply_preview_urdf(self) -> None:
        self.document.q = copy.deepcopy(self.preview_document.q)
        self._rebuild_joint_controls()

    def _apply_preview_all(self) -> None:
        replacement = Document(self.state.side)
        replacement.alignment = copy.deepcopy(self.document.alignment)
        replacement.segments = copy.deepcopy(self.preview_document.segments)
        rows, _, _ = self._rendered_preview_taxels()
        replacement.taxels = copy.deepcopy([taxel for _, taxel in rows])
        replacement.q = copy.deepcopy(self.preview_document.q)
        self.state.documents[self.state.side] = replacement
        self._rebuild_edit_lists()
        self._rebuild_joint_controls()

    def _restore_preview(self) -> None:
        self.preview[self.state.side] = self._preview_from(self.document)
        self._rebuild_joint_controls()
        self._rebuild_preview_segments()
        self._update_preview_error()

    def _rebuild_preview_segments(self) -> None:
        if self.preview_segment_folder is not None:
            self.preview_segment_folder.remove()
        self.preview_segment_widgets = []
        with self.static_folder:
            self.preview_segment_folder = self.server.gui.add_folder(
                "Segments", expand_by_default=False
            )
            with self.preview_segment_folder:
                for row, segment in enumerate(self.preview_document.segments):
                    with self.server.gui.add_folder(
                        f"{row}: {segment.joint}", expand_by_default=False
                    ):
                        self.server.gui.add_markdown(f"HandJoint: `{segment.joint}`")
                        position, rotation = geo.components(geo.matrix(segment.pose))
                        positions = self._numbers("Position", position, 0.001)
                        rotations = self._sliders("Rotation", rotation)
                        for axis, widget in enumerate(positions):
                            self._watch(
                                widget,
                                lambda value, row=row, axis=axis: (
                                    self._set_preview_pose(row, axis, value, False)
                                ),
                            )
                        for axis, widget in enumerate(rotations):
                            self._watch(
                                widget,
                                lambda value, row=row, axis=axis: (
                                    self._set_preview_pose(row, axis, value, True)
                                ),
                            )
                        self.preview_segment_widgets.append((positions, rotations))

    def _set_preview_pose(
        self, row: int, axis: int, value: float, rotation_field: bool
    ) -> None:
        segment = self.preview_document.segments[row]
        if not rotation_field:
            segment.pose[axis] = float(value)
        else:
            position, rotation = geo.components(geo.matrix(segment.pose))
            rotation = list(rotation)
            rotation[axis] = float(value)
            segment.pose = geo.pose(geo.transform(position, rotation))
        self._update_preview_error()
        self.redraw = True

    def _update_live(self) -> None:
        if self.receiver is None:
            return
        self.live_status.content = "Current Status: " + self.receiver.status
        if not self.receiver.active:
            return
        frame = self.receiver.latest
        raw_metadata = frame.get("metadata")
        if raw_metadata is not None:
            key = json.dumps(raw_metadata, sort_keys=True)
            if key != self.live_metadata_key:
                self.live_document = storage.parse(
                    raw_metadata, self.state.side, self.document.alignment
                )
                self.live_metadata_key = key
                self.binding_source_live.disabled = False
        try:
            hand = frame.get("hand")
            if hand is not None:
                updated = geo.hand_segments(hand)
                joints = [segment.joint for segment in updated]
                signature = [
                    segment.joint for segment in self.preview_document.segments
                ]
                self.preview_document.segments = updated
                if signature != joints:
                    self._rebuild_preview_segments()
            if self.joint_enabled.value:
                mapping = json.loads(self.joint_mapping.value or "{}")
                for source, rule in mapping.items():
                    raw, valid = frame.get("joints", {}).get(source, (None, False))
                    if valid and isinstance(rule, list) and len(rule) == 3:
                        target, scale, offset = rule
                        self.preview_document.q[target] = raw * scale + offset
            self._sync_preview_values()
            self._update_preview_error()
        except (
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            json.JSONDecodeError,
        ) as exc:
            self.preview_error.value = (
                f'<span style="color:#d32f2f">Live error: {exc}</span>'
            )
            self.preview_has_errors = True
        self.redraw = True

    def _sync_preview_values(self) -> None:
        for row, widgets in enumerate(self.preview_segment_widgets):
            if row >= len(self.preview_document.segments):
                break
            position, rotation = geo.components(
                geo.matrix(self.preview_document.segments[row].pose)
            )
            self._set_values(widgets[0], position)
            self._set_values(widgets[1], rotation)
        for name, widget in self.preview_joint_widgets.items():
            widget.value = self.preview_document.q.get(name, 0.0)

    async def _selection_gizmo_event(self, event) -> None:
        if event.client is None:
            return
        phase = event.phase
        position = np.asarray(event.target.position, dtype=float)
        self.events.put(lambda: self._apply_selection_gizmo_event(phase, position))

    def _apply_selection_gizmo_event(self, phase: str, position: np.ndarray) -> None:
        values = self._selection_values()
        rows = [row for row in self.selected_rows if 0 <= row < len(values)]
        if not rows:
            self.selection_gizmo_dragging = False
            self.selection_gizmo_last_position = None
            return
        if phase == "start":
            self.selection_gizmo_dragging = True
            self.selection_gizmo_last_position = position
            return
        if (
            not self.selection_gizmo_dragging
            or self.selection_gizmo_last_position is None
        ):
            return
        delta = position - self.selection_gizmo_last_position
        if np.any(delta):
            transform = np.eye(4)
            transform[:3, 3] = delta
            geo.apply_delta(values, rows, transform)
            center = np.asarray([values[row].pose[:3] for row in rows]).mean(axis=0)
            self._set_values(self.selection_position_widgets, center)
            self._sync_selected_list_values()
            self.selection_gizmo_last_position = position
        if phase == "end":
            self.selection_gizmo_dragging = False
            self.selection_gizmo_last_position = None
        self.redraw = True

    def _register_scene_events(self) -> None:
        self.server.scene.on_click()(self._scene_click)
        self.server.scene.on_click(modifier="shift")(self._scene_click)
        self.server.scene.on_rect_select(modifier="shift")(self._scene_box)

    async def _scene_click(self, event) -> None:
        self.events.put(
            lambda: self._run_action(lambda: self._handle_scene_event(event, False))
        )

    async def _scene_box(self, event) -> None:
        self.events.put(
            lambda: self._run_action(lambda: self._handle_scene_event(event, True))
        )

    def _display_values(self):
        if self.mode_value == "Preview":
            return self.preview_document.segments, []
        return self.document.segments, self.document.taxels

    def _handle_scene_event(self, event, box: bool) -> None:
        if self.pick_mode is not None and not box:
            self._handle_pick(event)
            return
        segments, taxels = self._display_values()
        candidates = {
            "segments": list(enumerate(segments)),
            "taxels": list(enumerate(taxels)),
        }
        if self.binding_phase is not None:
            candidates = {
                key: values
                for key, values in candidates.items()
                if key == self.binding_phase
            }
        chosen = {
            kind: self._box_rows(event, values)
            if box
            else self._ray_rows(event, kind, values)
            for kind, values in candidates.items()
        }
        chosen = {kind: rows for kind, rows in chosen.items() if rows}
        if not chosen:
            if self.binding_phase is not None:
                self._cancel_binding()
            else:
                self.selected_kind, self.selected_rows = None, []
                self._rebuild_selection()
            self.redraw = True
            return
        kind = (
            self.selected_kind
            if self.selected_kind in chosen
            else "taxels"
            if "taxels" in chosen
            else "segments"
        )
        rows = chosen[kind]
        if self.binding_phase == "taxels":
            self.binding_taxels = (
                sorted(set(self.binding_taxels) | set(rows))
                if event.modifier == "shift"
                else rows
            )
            self._set_binding_button("next" if self.binding_taxels else None)
        elif self.binding_phase == "segments":
            self.binding_segments = (
                sorted(set(self.binding_segments) | set(rows))
                if event.modifier == "shift"
                else rows
            )
            self._set_binding_button("complete" if self.binding_segments else None)
        elif event.modifier == "shift":
            if self.selected_kind not in (None, kind):
                return
            self.selected_kind = kind
            self.selected_rows = sorted(set(self.selected_rows) | set(rows))
            self._rebuild_selection()
        elif self.selected_kind != kind or self.selected_rows != rows:
            self.selected_kind, self.selected_rows = kind, rows
            self._rebuild_selection()
        self.redraw = True

    @staticmethod
    def _ray_rows(event, kind: str, values) -> list[int]:
        origin = np.asarray(event.ray_origin)
        direction = np.asarray(event.ray_direction)
        hits = []
        for row, value in values:
            inverse = np.linalg.inv(geo.matrix(value.pose))
            local_origin = (inverse @ np.r_[origin, 1.0])[:3]
            local_direction = inverse[:3, :3] @ direction
            if kind == "segments":
                distance = Editor._ray_box_distance(
                    local_origin, local_direction, SEGMENT_CUBE_SIDE_M / 2
                )
                if distance is not None:
                    hits.append((distance, row))
                continue
            if abs(local_direction[2]) < 1e-10:
                continue
            distance = -local_origin[2] / local_direction[2]
            local_hit = local_origin + distance * local_direction
            if distance > 0 and np.max(np.abs(local_hit[:2])) <= TAXEL_SIDE_M / 2:
                hits.append((distance, row))
        return [min(hits)[1]] if hits else []

    @staticmethod
    def _ray_box_distance(origin, direction, half_extent: float) -> float | None:
        near, far = -math.inf, math.inf
        for axis in range(3):
            if abs(direction[axis]) < 1e-10:
                if abs(origin[axis]) > half_extent:
                    return None
                continue
            first = (-half_extent - origin[axis]) / direction[axis]
            second = (half_extent - origin[axis]) / direction[axis]
            near = max(near, min(first, second))
            far = min(far, max(first, second))
            if near > far:
                return None
        if far <= 0:
            return None
        return near if near > 0 else far

    @staticmethod
    def _box_rows(event, values) -> list[int]:
        camera = event.client.camera
        rotation = tf.quaternion_matrix(camera.wxyz)[:3, :3]
        origin = np.asarray(camera.position)
        rows = []
        for row, value in values:
            point = np.asarray(value.pose[:3])
            local = rotation.T @ (point - origin)
            if local[2] <= 0:
                continue
            uv = (
                0.5
                + local[0] / local[2] / (2 * np.tan(camera.fov / 2) * camera.aspect),
                0.5 + local[1] / local[2] / (2 * np.tan(camera.fov / 2)),
            )
            if all(
                lower <= value <= upper
                for lower, value, upper in zip(event.screen_min, uv, event.screen_max)
            ):
                rows.append(row)
        return rows

    def _handle_pick(self, event) -> None:
        mode, self.pick_mode = self.pick_mode, None
        self.segment_click_button.label = "Add with URDF Click"
        self.taxel_capture_button.label = "Capture URDF Surface Pose"
        if self.model is None:
            raise ValueError("Load a URDF before using URDF Click")
        hit = self.model.hit(
            np.asarray(event.ray_origin),
            np.asarray(event.ray_direction),
            self.document.q,
            self.urdf_alignment,
            self.urdf_scale,
        )
        if hit is None:
            return
        if mode == "segment":
            value = IDENTITY_POSE.copy()
            value[:3] = hit[1].tolist()
            row = self.document.add_segment(self.new_segment_joint.value, value)
            self.selected_kind, self.selected_rows = "segments", [row]
            self._rebuild_edit_lists()
        else:
            frame = geo.surface_frame(hit[1], hit[2])
            position, rotation = geo.components(frame)
            self._set_values(self.grid_center, position)
            self._set_values(self.grid_rotation, rotation)

    def _selection_values(self):
        if self.selected_kind is None:
            return []
        if self.mode_value == "Preview":
            return self.preview_document.segments
        return self._values(self.selected_kind)

    def _rebuild_selection(self) -> None:
        if self.selection_folder is not None:
            self.selection_folder.remove()
        values = self._selection_values()
        rows = [row for row in self.selected_rows if 0 <= row < len(values)]
        self.selected_rows = rows
        with self.selection_tab:
            self.selection_folder = self.server.gui.add_folder("Selection")
        with self.selection_folder:
            self.server.gui.add_markdown(
                f"Type: **{self.selected_kind if rows else 'None'}**  \n"
                f"Count: **{len(rows)}**"
            )
            center = (
                np.asarray([values[row].pose[:3] for row in rows]).mean(axis=0)
                if rows
                else np.zeros(3)
            )
            if len(rows) == 1:
                _, rotation = geo.components(geo.matrix(values[rows[0]].pose))
            else:
                rotation = (0.0, 0.0, 0.0)
            self.selection_position_widgets = self._numbers("Position", center, 0.001)
            self.selection_rotation_widgets = self._sliders("Rotation", rotation)
            self.selection_rotation = np.asarray(rotation, dtype=float)
            for widget in (
                *self.selection_position_widgets,
                *self.selection_rotation_widgets,
            ):
                widget.disabled = not rows
            for axis, widget in enumerate(self.selection_position_widgets):
                self._watch(
                    widget,
                    lambda value, axis=axis: self._set_selection_position(axis, value),
                )
            for axis, widget in enumerate(self.selection_rotation_widgets):
                self._watch(
                    widget,
                    lambda value, axis=axis: self._set_selection_rotation(axis, value),
                )
            self.selection_scale = 1.0
            self.selection_scale_widget = self.server.gui.add_number(
                "Scale", 1.0, min=1e-6, step=0.01, disabled=len(rows) < 2
            )
            self._watch(self.selection_scale_widget, self._set_selection_scale)
            delete = self._action_button("Delete", self._delete_selection)
            delete.disabled = not rows

    def _set_selection_position(self, axis: int, value: float) -> None:
        values = self._selection_values()
        center = np.asarray([values[row].pose[:3] for row in self.selected_rows]).mean(
            axis=0
        )
        delta = np.eye(4)
        delta[axis, 3] = float(value) - center[axis]
        geo.apply_delta(values, self.selected_rows, delta)
        self._sync_selected_list_values()

    def _set_selection_rotation(self, axis: int, value: float) -> None:
        values = self._selection_values()
        if len(self.selected_rows) == 1:
            row = self.selected_rows[0]
            position, rotation = geo.components(geo.matrix(values[row].pose))
            rotation = list(rotation)
            rotation[axis] = float(value)
            values[row].pose = geo.pose(geo.transform(position, rotation))
        else:
            points = np.asarray([values[row].pose[:3] for row in self.selected_rows])
            center = points.mean(axis=0)
            angle = np.deg2rad(float(value) - self.selection_rotation[axis])
            rotation = tf.rotation_matrix(angle, np.eye(3)[axis], center)
            geo.apply_delta(values, self.selected_rows, rotation)
            self.selection_rotation[axis] = float(value)
        self._sync_selected_list_values()

    def _set_selection_scale(self, value: float) -> None:
        values = self._selection_values()
        factor = float(value) / self.selection_scale
        geo.scale_positions(values, self.selected_rows, factor)
        self.selection_scale = float(value)
        self._sync_selected_list_values()

    def _sync_selected_list_values(self) -> None:
        if self.mode_value == "Edit":
            widgets = (
                self.segment_widgets
                if self.selected_kind == "segments"
                else self.taxel_widgets
            )
            values = self._selection_values()
            for row in self.selected_rows:
                if row >= len(widgets):
                    continue
                position, rotation = geo.components(geo.matrix(values[row].pose))
                position_widgets, rotation_widgets = (
                    (widgets[row][1], widgets[row][2])
                    if self.selected_kind == "segments"
                    else (widgets[row][0], widgets[row][1])
                )
                self._set_values(position_widgets, position)
                self._set_values(rotation_widgets, rotation)
        else:
            self._sync_preview_values()
        self.redraw = True

    def _delete_selection(self) -> None:
        if self.mode_value == "Preview":
            for row in sorted(self.selected_rows, reverse=True):
                self.preview_document.segments.pop(row)
            self._rebuild_preview_segments()
            self._update_preview_error()
        else:
            self.document.delete(self.selected_kind, self.selected_rows)
            self._rebuild_edit_lists()
        self.selected_kind, self.selected_rows = None, []
        self._rebuild_selection()

    def _draw_instances(
        self,
        name: str,
        kind: str,
        values,
        colors,
        opacity: float,
    ) -> None:
        matrices = [geo.matrix(value.pose) for value in values]
        if not matrices:
            if name in self.scene_handles:
                self.scene_handles[name].visible = False
            return
        positions = np.asarray([value[:3, 3] for value in matrices])
        rotations = np.asarray([tf.quaternion_from_matrix(value) for value in matrices])
        scale = SEGMENT_CUBE_SIDE_M if kind == "segments" else TAXEL_SIDE_M
        if name not in self.scene_handles:
            mesh = (
                trimesh.creation.box(extents=(1.0, 1.0, 1.0))
                if kind == "segments"
                else trimesh.Trimesh(
                    vertices=[
                        [-0.5, -0.5, 0.0],
                        [0.5, -0.5, 0.0],
                        [0.5, 0.5, 0.0],
                        [-0.5, 0.5, 0.0],
                    ],
                    faces=[[0, 1, 2], [0, 2, 3]],
                    process=False,
                )
            )
            self.scene_handles[name] = self.server.scene.add_batched_meshes_simple(
                name,
                mesh.vertices,
                mesh.faces,
                rotations,
                positions,
                batched_scales=np.full(len(values), scale),
                batched_colors=np.asarray(colors, dtype=np.uint8),
                opacity=opacity,
                flat_shading=SEGMENT_FLAT_SHADING if kind == "segments" else False,
                side="double",
                cast_shadow=VISUALIZER_CAST_SHADOW,
                receive_shadow=VISUALIZER_RECEIVE_SHADOW,
            )
        handle = self.scene_handles[name]
        handle.visible = True
        handle.batched_positions = positions
        handle.batched_wxyzs = rotations
        handle.batched_scales = np.full(len(values), scale)
        handle.batched_colors = np.asarray(colors, dtype=np.uint8)
        handle.opacity = opacity

    def _hide_prefix(self, prefix: str) -> None:
        for name, handle in self.scene_handles.items():
            if name.startswith(prefix):
                handle.visible = False

    @staticmethod
    def _side_highlight_rows(
        segments: list[Segment],
        taxels: list[Taxel],
        selected_kind: str | None,
        selected_rows: list[int],
    ) -> tuple[set[int], set[int]]:
        segment_rows: set[int] = set()
        taxel_rows: set[int] = set()
        if selected_kind == "segments":
            selected_joints = {
                segments[row].joint for row in selected_rows if 0 <= row < len(segments)
            }
            taxel_rows = {
                row
                for row, taxel in enumerate(taxels)
                if any(
                    joint in selected_joints
                    and type(weight) in (int, float)
                    and weight > 0
                    for joint, weight in zip(taxel.segments, taxel.weights)
                )
            }
        elif selected_kind == "taxels":
            bound_joints = {
                joint
                for row in selected_rows
                if 0 <= row < len(taxels)
                for joint, weight in zip(taxels[row].segments, taxels[row].weights)
                if joint is not None and type(weight) in (int, float) and weight > 0
            }
            segment_rows = {
                row
                for row, segment in enumerate(segments)
                if segment.joint in bound_joints
            }
        return segment_rows, taxel_rows

    def _draw_dataset(
        self,
        prefix: str,
        segments: list[Segment],
        taxels: list[Taxel],
        opacity: float,
        active: bool,
        forces=None,
    ) -> None:
        side_segment_rows, side_taxel_rows = (
            self._side_highlight_rows(
                segments, taxels, self.selected_kind, self.selected_rows
            )
            if active
            else (set(), set())
        )
        segment_colors = []
        for row in range(len(segments)):
            color = SEGMENT_COLOR if active else INACTIVE_COLOR
            if row in side_segment_rows:
                color = SIDE_HIGHLIGHT_COLOR
            if (
                self.selected_kind == "segments"
                and row in self.selected_rows
                and active
            ):
                color = SELECTED_COLOR
            if self.binding_phase == "segments" and row in self.binding_segments:
                color = SELECTED_COLOR
            if self.binding_phase == "taxels" and active:
                color = INACTIVE_COLOR
            if self.pick_mode is not None and active:
                color = INACTIVE_COLOR
            segment_colors.append(color)
        taxel_colors = [TAXEL_COLOR] * len(taxels)
        array = np.asarray(forces, dtype=float) if forces is not None else None
        if (
            array is not None
            and array.shape == (len(taxels), 3)
            and np.isfinite(array).all()
        ):
            fraction = np.clip(np.linalg.norm(array, axis=1) / FORCE_MAX_N, 0.0, 1.0)
            taxel_colors = [
                tuple(
                    int(low + value * (high - low))
                    for low, high in zip(FORCE_MIN_COLOR, FORCE_MAX_COLOR)
                )
                for value in fraction
            ]
        for row in range(len(taxels)):
            if row in side_taxel_rows:
                taxel_colors[row] = SIDE_HIGHLIGHT_COLOR
            if self.selected_kind == "taxels" and row in self.selected_rows and active:
                taxel_colors[row] = SELECTED_COLOR
            if self.binding_phase == "taxels" and row in self.binding_taxels:
                taxel_colors[row] = SELECTED_COLOR
            if self.binding_phase == "segments" and active:
                taxel_colors[row] = INACTIVE_COLOR
            if self.pick_mode is not None and active:
                taxel_colors[row] = INACTIVE_COLOR
        if prefix == "/edit":
            error_rows = {
                issue["row"]
                for issue in self.document.validate()
                if issue["kind"] == "taxels" and issue["row"] is not None
            }
            for row in error_rows:
                if 0 <= row < len(taxel_colors):
                    taxel_colors[row] = ERROR_COLOR
        self._draw_instances(
            prefix + "/segments", "segments", segments, segment_colors, opacity
        )
        self._draw_instances(
            prefix + "/taxels", "taxels", taxels, taxel_colors, opacity
        )

    def _draw_selection_gizmo(self) -> None:
        values = self._selection_values()
        rows = [row for row in self.selected_rows if 0 <= row < len(values)]
        if not rows:
            self.selection_gizmo.visible = False
            self.selection_gizmo_dragging = False
            self.selection_gizmo_last_position = None
            return
        self.selection_gizmo.visible = True
        if self.selection_gizmo_dragging:
            return
        self.selection_gizmo.position = np.asarray(
            [values[row].pose[:3] for row in rows]
        ).mean(axis=0)
        self.selection_gizmo.wxyz = (
            tf.quaternion_from_matrix(geo.matrix(values[rows[0]].pose))
            if len(rows) == 1
            else np.array([1.0, 0.0, 0.0, 0.0])
        )

    def _draw(self) -> None:
        active_opacity = ACTIVE_DATASET_OPACITY
        if self.model is not None:
            q = (
                self.document.q
                if self.mode_value == "Edit"
                else self.preview_document.q
            )
            self.model.draw(
                self.server.scene,
                q,
                self.urdf_alignment,
                self.urdf_scale,
                highlighted=self.pick_mode is not None,
                opacity=URDF_OPACITY * active_opacity,
            )
        if self.mode_value == "Edit":
            self._hide_prefix("/preview")
            self._draw_dataset(
                "/edit",
                self.document.segments,
                self.document.taxels,
                active_opacity,
                True,
            )
        else:
            try:
                valid_rows, _, source_count = self._rendered_preview_taxels()
            except (ValueError, TypeError, KeyError, IndexError):
                valid_rows = []
                source_count = 0
            preview_taxels = [taxel for _, taxel in valid_rows]
            raw_forces = (
                self.receiver.latest.get("forces")
                if self.receiver
                and self.receiver.active
                and time.monotonic()
                - self.receiver.latest.get("read_time", float("-inf"))
                < 0.5
                else None
            )
            forces = (
                [raw_forces[row] for row, _ in valid_rows]
                if raw_forces is not None and len(raw_forces) == source_count
                else None
            )
            self._draw_dataset(
                "/preview",
                self.preview_document.segments,
                preview_taxels,
                active_opacity,
                True,
                forces,
            )
            if self.show_binding.value:
                self._draw_dataset(
                    "/edit",
                    self.document.segments,
                    self.document.taxels,
                    BINDING_OVERLAY_OPACITY,
                    False,
                )
            else:
                self._hide_prefix("/edit")
        saved = self.state.saved[self.state.side]
        if self.show_saved.value and saved is not None:
            self._draw_dataset(
                "/saved",
                saved.segments,
                saved.taxels,
                SAVED_BINDING_OPACITY,
                False,
            )
        else:
            self._hide_prefix("/saved")
        self._draw_selection_gizmo()

    def run(self) -> None:
        try:
            while True:
                try:
                    while True:
                        self.events.get_nowait()()
                except queue.Empty:
                    pass
                self._update_live()
                if self.redraw or self.receiver is not None:
                    self._draw()
                    self.redraw = False
                time.sleep(1 / 30)
        except KeyboardInterrupt:
            pass
        finally:
            self._disconnect()
            self.server.stop()
