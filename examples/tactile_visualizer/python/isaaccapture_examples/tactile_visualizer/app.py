# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Viser application for editing and visualizing tactile hand data."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import viser
from viser.extras import ViserUrdf

from .model import (
    OPENXR_JOINT_NAMES,
    ROOT_PARENT,
    WRIST,
    SessionController,
    ShapeDocument,
    append_grid_taxels,
    apply_display_side,
    atomic_write_within,
    default_openxr_shape,
    force_colors,
    frame_wxyz,
    matrix_to_quaternion_xyzw,
    mirror_pose,
    normalize,
    project_points,
    quaternion_xyzw_to_matrix,
    resolve_current_joints,
    set_taxel_display_frame,
    set_taxel_display_position,
    skin_taxels,
)
from .sources import SnapshotSource, SourceSnapshot
from .urdf_config import UrdfConfig, map_raw_joints

logger = logging.getLogger("isaaccapture.examples.tactile_visualizer.app")


def _modifier(event: Any) -> str:
    return str(getattr(event, "modifier", "")).lower()


def _parse_numbers(text: str, count: int, kind=float) -> np.ndarray:
    values = [kind(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != count:
        raise ValueError(f"expected {count} comma-separated values")
    value = np.asarray(values)
    if value.dtype.kind == "f":
        if not np.isfinite(value).all():
            raise ValueError("values must be finite")
        if np.any(np.abs(value) > 999.99):
            raise ValueError("values must be in [-999.99, 999.99]")
    return value


def _validate_gui_vector(raw: Any) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("values must be finite")
    if np.any(np.abs(value) > 999.99):
        raise ValueError("values must be in [-999.99, 999.99]")
    return value


class TactileVisualizer:
    """Own the active shape, live snapshot, Viser scene, and GUI callbacks."""

    def __init__(
        self,
        *,
        server: viser.ViserServer,
        urdf_path: Path,
        urdf_config_path: Path | None,
        urdf_config: UrdfConfig,
        document: ShapeDocument,
        document_name: str,
        source: SnapshotSource | None,
        side: str,
        liveness_timeout_s: float,
        taxel_size_m: float,
        write_root: Path | None,
    ) -> None:
        self.server = server
        self.urdf_path = urdf_path
        self.urdf_config_path = urdf_config_path
        self.urdf_config = urdf_config
        self.document = document
        self.document_name = document_name
        self.source = source
        self.side = side
        self.controller = SessionController(source is not None, liveness_timeout_s)
        self.taxel_size_m = taxel_size_m
        self.write_root = write_root.resolve() if write_root is not None else None

        self.editor_client_id: int | None = None
        self.sequence = 0
        self.latest_snapshot: SourceSnapshot | None = None
        self.display_snapshot: SourceSnapshot | None = None
        self.latest_shapes: dict[str, ShapeDocument] = {}
        self.latest_forces: dict[str, np.ndarray] = {}
        self.display_forces: dict[str, np.ndarray] = {}
        self._loaded_remote_sides: set[str] = set()
        self.user_shape_override = False
        self.selected_kind: str | None = None
        self.selected_indices: set[int] = set()
        self.transform_mode = "Translate"
        self.pointer_mode = "Navigate"
        self._programmatic = False
        self._last_error = ""
        self._gizmo = None
        self._selection_axes = None
        self._gizmo_context: dict[str, Any] | None = None
        self._tree_root = None
        self._tree_joint_inputs: list[dict[str, Any]] = []
        self._tree_taxel_inputs: list[dict[str, Any]] = []
        self._taxel_handles: list[Any] = []
        self._joint_handles: list[Any] = []
        self._bone_handles: dict[int, Any] = {}
        self._current_positions = document.joint_bind_positions.astype(np.float64)
        self._current_orientations = document.joint_bind_orientations.astype(np.float64)
        self._display_taxel_positions = document.taxel_positions.copy()
        self._display_taxel_normals = document.taxel_normals.copy()
        self._display_taxel_tangents = document.taxel_tangents.copy()

        self._configure_scene()
        self._build_panels()
        self._build_schema_tree()
        self._build_scene_handles()
        self._install_callbacks()
        self._set_pointer_mode()
        self._refresh_editability()

    def _configure_scene(self) -> None:
        self.server.scene.set_up_direction("+y")
        self.server.scene.add_grid(
            "/grid",
            width=0.5,
            height=0.5,
            plane="xy",
            cell_size=0.01,
            section_size=0.05,
            position=(0.0, 0.07, -0.012),
        )
        self.server.initial_camera.position = (0.16, 0.11, 0.28)
        self.server.initial_camera.look_at = (0.0, 0.07, 0.0)
        self.server.initial_camera.up_direction = (0.0, 1.0, 0.0)
        self.urdf = None
        self._urdf_root = None
        self.urdf_joint_names: tuple[str, ...] = ()
        self._load_urdf()

    def _load_urdf(self) -> None:
        if self.urdf is not None:
            self.urdf.remove()
        if self._urdf_root is not None:
            self._urdf_root.remove()
        q = normalize(
            np.asarray(self.urdf_config.rotation_xyzw, dtype=np.float64),
            label="URDF rotation_xyzw",
        )
        scale = (-1.0, 1.0, 1.0) if self.side == "left" else 1.0
        self._urdf_root = self.server.scene.add_frame(
            "/glove",
            show_axes=False,
            position=np.asarray(self.urdf_config.offset_cm, dtype=np.float64) / 100.0,
            wxyz=q[[3, 0, 1, 2]],
            scale=scale,
        )
        self.urdf = None
        self.urdf_joint_names = ()
        try:
            self.urdf = ViserUrdf(
                self.server,
                self.urdf_path,
                scale=self.urdf_config.scale,
                root_node_name="/glove/model",
                mesh_color_override=(0.3, 0.55, 0.95, 0.35),
            )
            self.urdf_joint_names = self.urdf.get_actuated_joint_names()
        except Exception as error:
            logger.exception("Failed to load URDF %s", self.urdf_path)
            self._last_error = f"URDF unavailable: {error}"

    def _build_panels(self) -> None:
        selection = self.server.gui.add_panel()
        selection.dock_left()
        selection.set_width(350)
        with selection.add_tab("Selection Inspector"):
            self.selection_summary = self.server.gui.add_markdown("No selection")
            self.selection_placeholder = self.server.gui.add_markdown("")
            with self.server.gui.add_folder("Selected object", expand_by_default=True):
                bounds = (-999.99, -999.99, -999.99)
                upper = (999.99, 999.99, 999.99)
                self.position_input = self.server.gui.add_vector3(
                    "Position (cm)", (0.0, 0.0, 0.0), min=bounds, max=upper, step=0.01
                )
                self.normal_input = self.server.gui.add_vector3(
                    "Normal", (0.0, 0.0, 1.0), min=bounds, max=upper, step=0.01
                )
                self.tangent_input = self.server.gui.add_vector3(
                    "Tangent", (1.0, 0.0, 0.0), min=bounds, max=upper, step=0.01
                )
                self.orientation_text = self.server.gui.add_text(
                    "Bind quaternion XYZW", ""
                )
                self.parent_text = self.server.gui.add_text("Parent index", "")
                self.indices_text = self.server.gui.add_text("Joint indices", "")
                self.weights_text = self.server.gui.add_text("Joint weights", "")
                self.force_text = self.server.gui.add_text(
                    "Force magnitude (N)", "", disabled=True
                )
                self.current_position_text = self.server.gui.add_text(
                    "Current position (cm)", "", disabled=True
                )
                self.current_orientation_text = self.server.gui.add_text(
                    "Current quaternion XYZW", "", disabled=True
                )

        controls = self.server.gui.add_panel()
        controls.dock_right()
        controls.set_width(390)
        with controls.add_tab("Controls / Schema"):
            self.paths_markdown = self.server.gui.add_markdown("")
            self._update_paths_text()
            with self.server.gui.add_folder("File operations", expand_by_default=True):
                self.upload = self.server.gui.add_upload_button(
                    "Load TactileShape", mime_type="application/octet-stream"
                )
                self.filename = self.server.gui.add_text(
                    "Download / server filename", "tactile_shape.bin"
                )
                self.create_new = self.server.gui.add_button("Create New (do not load)")
                self.export = self.server.gui.add_button("Export active shape")
                note = (
                    f"Atomic server writes enabled below `{self.write_root}`."
                    if self.write_root is not None
                    else "Server writes disabled; files are browser downloads."
                )
                self.file_status = self.server.gui.add_markdown(note)
            with self.server.gui.add_folder("Add taxels", expand_by_default=True):
                self.add_taxel_count = self.server.gui.add_number(
                    "Amount (0-125)", 0, min=0, max=125, step=1
                )
                self.add_taxels_button = self.server.gui.add_button("Add Taxels")
                self.server.gui.add_markdown(
                    "Deterministic order: x outer, y middle, z inner; each axis "
                    "uses `[-4, -2, 0, 2, 4] cm`."
                )
            self.inspector_status = self.server.gui.add_markdown("")
            self.inspector_container = self.server.gui.add_folder(
                "Full schema", expand_by_default=True
            )

        toolbar = self.server.gui.add_panel()
        toolbar.float(x=0.27, y=0.01, width=0.46, height=0.16)
        with toolbar.add_tab("Tactile toolbar"):
            self.state_markdown = self.server.gui.add_markdown("")
            self.reset_view_button = self.server.gui.add_button("Reset view")
            self.pause_button = self.server.gui.add_button("Pause / Resume")
            self.pointer_mode_control = self.server.gui.add_dropdown(
                "Pointer mode", ("Navigate", "Box select"), initial_value="Navigate"
            )
            self.mode_control = self.server.gui.add_dropdown(
                "Taxel gizmo",
                ("Translate", "Rotate"),
                initial_value=self.transform_mode,
            )
            self.side_control = self.server.gui.add_dropdown(
                "Hand", ("left", "right"), initial_value=self.side
            )

    def _update_paths_text(self) -> None:
        config = (
            str(self.urdf_config_path)
            if self.urdf_config_path is not None
            else "built-in defaults"
        )
        self.paths_markdown.content = (
            f"**URDF:** `{self.urdf_path}`  \n"
            f"**URDF config:** `{config}`  \n"
            f"**Document:** `{self.document_name}`"
        )

    def _build_schema_tree(self) -> None:
        if self._tree_root is not None:
            self._tree_root.remove()
        self._tree_joint_inputs.clear()
        self._tree_taxel_inputs.clear()
        with self.inspector_container:
            self._tree_root = self.server.gui.add_folder(
                "TactileShape", expand_by_default=False
            )
            with self._tree_root:
                with self.server.gui.add_folder(
                    f"joints [{self.document.num_joints}]", expand_by_default=False
                ):
                    for index in range(self.document.num_joints):
                        name = (
                            OPENXR_JOINT_NAMES[index]
                            if index < len(OPENXR_JOINT_NAMES)
                            else str(index)
                        )
                        with self.server.gui.add_folder(
                            f"{index}: {name}", expand_by_default=False
                        ):
                            fields = self._add_joint_tree_fields()
                            self._tree_joint_inputs.append(fields)
                            self._bind_tree_fields("joint", index, fields)
                with self.server.gui.add_folder(
                    f"taxels [{self.document.num_taxels}]", expand_by_default=False
                ):
                    for index in range(self.document.num_taxels):
                        with self.server.gui.add_folder(
                            f"taxel[{index}]", expand_by_default=False
                        ):
                            fields = self._add_taxel_tree_fields()
                            self._tree_taxel_inputs.append(fields)
                            self._bind_tree_fields("taxel", index, fields)
            self._data_tree = self.server.gui.add_folder(
                "TactileData", expand_by_default=False
            )
            with self._data_tree:
                self._data_tree_text = self.server.gui.add_markdown(
                    "`taxel_force` (N): no sample"
                )
        self._sync_tree_values()

    def _add_joint_tree_fields(self) -> dict[str, Any]:
        bounds = (-999.99, -999.99, -999.99)
        upper = (999.99, 999.99, 999.99)
        return {
            "position": self.server.gui.add_vector3(
                "bind_pose.position (cm)",
                (0.0, 0.0, 0.0),
                min=bounds,
                max=upper,
                step=0.01,
            ),
            "orientation": self.server.gui.add_text("bind_pose.orientation XYZW", ""),
            "parent": self.server.gui.add_text("parent", ""),
        }

    def _add_taxel_tree_fields(self) -> dict[str, Any]:
        bounds = (-999.99, -999.99, -999.99)
        upper = (999.99, 999.99, 999.99)
        return {
            "position": self.server.gui.add_vector3(
                "position (cm)",
                (0.0, 0.0, 0.0),
                min=bounds,
                max=upper,
                step=0.01,
            ),
            "normal": self.server.gui.add_vector3(
                "normal", (0.0, 0.0, 1.0), min=bounds, max=upper, step=0.01
            ),
            "tangent": self.server.gui.add_vector3(
                "tangent", (1.0, 0.0, 0.0), min=bounds, max=upper, step=0.01
            ),
            "indices": self.server.gui.add_text("joint_indices", ""),
            "weights": self.server.gui.add_text("joint_weights", ""),
        }

    def _bind_tree_fields(self, kind: str, index: int, fields: dict[str, Any]) -> None:
        for field, handle in fields.items():

            @handle.on_update
            def _(event, kind=kind, index=index, field=field) -> None:
                self._edit_field_event(event, kind, index, field)

    @staticmethod
    def _format_vector(values: np.ndarray) -> str:
        return ", ".join(f"{float(value):.2f}" for value in values)

    def _sync_tree_values(self) -> None:
        for index, fields in enumerate(self._tree_joint_inputs):
            self._set_value(
                fields["position"],
                tuple((self.document.joint_bind_positions[index] * 100.0).tolist()),
            )
            self._set_value(
                fields["orientation"],
                self._format_vector(self.document.joint_bind_orientations[index]),
            )
            parent = int(self.document.joint_parents[index])
            self._set_value(
                fields["parent"],
                "ROOT_PARENT" if parent == int(ROOT_PARENT) else str(parent),
            )
        for index, fields in enumerate(self._tree_taxel_inputs):
            self._set_value(
                fields["position"],
                tuple((self.document.taxel_positions[index] * 100.0).tolist()),
            )
            self._set_value(
                fields["normal"], tuple(self.document.taxel_normals[index].tolist())
            )
            self._set_value(
                fields["tangent"],
                tuple(self.document.taxel_tangents[index].tolist()),
            )
            self._set_value(
                fields["indices"],
                ", ".join(map(str, self.document.taxel_joint_indices[index])),
            )
            self._set_value(
                fields["weights"],
                self._format_vector(self.document.taxel_joint_weights[index]),
            )

    def _build_scene_handles(self) -> None:
        for handle in (
            self._taxel_handles
            + self._joint_handles
            + list(self._bone_handles.values())
        ):
            handle.remove()
        self._taxel_handles.clear()
        self._joint_handles.clear()
        self._bone_handles.clear()
        for joint in range(self.document.num_joints):
            handle = self.server.scene.add_icosphere(
                f"/hand/joints/{joint:02d}",
                radius=0.0025,
                color=(255, 180, 60),
                subdivisions=2,
                opacity=0.55,
            )

            @handle.on_click
            def _(event, joint=joint) -> None:
                if "shift" not in _modifier(event):
                    return
                if self._claim_editor(event.client_id, event.client):
                    self._select("joint", {joint})

            self._joint_handles.append(handle)
        self._rebuild_bones()
        for taxel in range(self.document.num_taxels):
            handle = self.server.scene.add_box(
                f"/tactile/taxels/{taxel:05d}",
                dimensions=(self.taxel_size_m, self.taxel_size_m, 0.0005),
                color=(0, 0, 255),
                flat_shading=True,
                cast_shadow=False,
            )

            @handle.on_click
            def _(event, taxel=taxel) -> None:
                if self._claim_editor(event.client_id, event.client):
                    self._select("taxel", {taxel})

            self._taxel_handles.append(handle)
        self._update_rendered_values()

    @staticmethod
    def _segment_wxyz(delta: np.ndarray) -> np.ndarray:
        z_axis = normalize(delta, label="bone segment")
        seed = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(seed, z_axis))) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        y_axis = normalize(np.cross(z_axis, seed))
        x_axis = np.cross(y_axis, z_axis)
        xyzw = matrix_to_quaternion_xyzw(np.column_stack((x_axis, y_axis, z_axis)))
        return xyzw[[3, 0, 1, 2]]

    def _rebuild_bones(self) -> None:
        for handle in self._bone_handles.values():
            handle.remove()
        self._bone_handles.clear()
        positions, _ = self._display_joint_pose()
        for child, parent in enumerate(self.document.joint_parents):
            if int(parent) == int(ROOT_PARENT):
                continue
            delta = positions[child] - positions[int(parent)]
            length = float(np.linalg.norm(delta))
            if length < 1e-8:
                continue
            self._bone_handles[child] = self.server.scene.add_cylinder(
                f"/hand/bones/{child:02d}",
                radius=0.0006,
                height=length,
                color=(255, 170, 50),
                opacity=0.35,
                radial_segments=8,
                cast_shadow=False,
                position=(positions[child] + positions[int(parent)]) / 2.0,
                wxyz=self._segment_wxyz(delta),
            )

    def _install_callbacks(self) -> None:
        @self.server.on_client_connect
        def _(client) -> None:
            if self.editor_client_id is None:
                self.editor_client_id = client.client_id
                client.add_notification(
                    title="Editing enabled",
                    body="This browser owns the single editing lease.",
                )
            else:
                client.add_notification(
                    title="Read-only client",
                    body="Another browser owns the editing lease.",
                    color="yellow",
                )

        @self.server.on_client_disconnect
        def _(client) -> None:
            if client.client_id == self.editor_client_id:
                self.editor_client_id = None

        @self.reset_view_button.on_click
        def _(event) -> None:
            camera = event.client.camera
            camera.position = (0.16, 0.11, 0.28)
            camera.look_at = (0.0, 0.07, 0.0)
            camera.up_direction = (0.0, 1.0, 0.0)

        @self.pause_button.on_click
        def _(event) -> None:
            if not self._claim_editor(event.client_id, event.client):
                return
            now = time.monotonic()
            if self.controller.paused:
                self.controller.resume(now)
                if self.latest_snapshot is not None:
                    self.display_snapshot = self.latest_snapshot
                self.display_forces = {
                    side: value.copy() for side, value in self.latest_forces.items()
                }
                self._update_rendered_values()
            else:
                self.controller.pause()
            self._refresh_editability()

        @self.pointer_mode_control.on_update
        def _(event) -> None:
            if self._programmatic:
                return
            if not self._claim_editor(event.client_id, event.client):
                self._set_value(self.pointer_mode_control, self.pointer_mode)
                return
            self.pointer_mode = str(event.target.value)
            self._set_pointer_mode()

        @self.mode_control.on_update
        def _(event) -> None:
            if self._programmatic:
                return
            if not self._claim_editor(event.client_id, event.client):
                self._set_value(self.mode_control, self.transform_mode)
                return
            self.transform_mode = str(event.target.value)
            self._update_selection_gizmo()

        @self.side_control.on_update
        def _(event) -> None:
            if self._programmatic:
                return
            if not self._claim_editor(event.client_id, event.client):
                self._set_value(self.side_control, self.side)
                return
            self.side = str(event.target.value)
            self._load_urdf()
            if self.source is not None:
                ready = self.latest_snapshot is not None and self._source_ready(
                    self.latest_snapshot
                )
                self.controller.requirements_changed(
                    time.monotonic(), ready=bool(ready)
                )
            if not self.user_shape_override and self.side in self.latest_shapes:
                self._replace_document(self.latest_shapes[self.side].copy())
                self._loaded_remote_sides.add(self.side)
            self._select(None, set())
            self._update_rendered_values()

        @self.upload.on_upload
        def _(event) -> None:
            if not self._claim_editor(event.client_id, event.client):
                return
            if not self.controller.editing_enabled:
                self._report_error("Pause live playback before loading.", event.client)
                return
            try:
                document = ShapeDocument.from_bytes(event.target.value.content)
                self.user_shape_override = True
                self.document_name = event.target.value.name
                self._replace_document(document)
                self._update_paths_text()
                self._set_status(f"Loaded `{event.target.value.name}`.")
            except Exception as error:
                self._report_error(f"Load rejected: {error}", event.client)

        @self.create_new.on_click
        def _(event) -> None:
            if not self._claim_editor(event.client_id, event.client):
                return
            if not self.controller.editing_enabled:
                self._report_error("Pause live playback before creating.", event.client)
                return
            try:
                self._deliver_file(
                    default_openxr_shape().to_bytes(),
                    self.filename.value,
                    event.client,
                    "Created a 26-joint, zero-taxel shape without loading it.",
                )
            except Exception as error:
                self._report_error(f"Create failed: {error}", event.client)

        @self.export.on_click
        def _(event) -> None:
            if not self._claim_editor(event.client_id, event.client):
                return
            try:
                self._deliver_file(
                    self.document.to_bytes(),
                    self.filename.value,
                    event.client,
                    "Exported the active shape.",
                )
            except Exception as error:
                self._report_error(f"Export failed: {error}", event.client)

        @self.add_taxels_button.on_click
        def _(event) -> None:
            if not self._claim_editor(event.client_id, event.client):
                return
            if not self.controller.editing_enabled:
                self._report_error(
                    "Pause live playback before adding taxels.", event.client
                )
                return
            try:
                amount = int(self.add_taxel_count.value)
                append_grid_taxels(self.document, amount)
                self._mark_local_edit()
                self._build_schema_tree()
                self._build_scene_handles()
                self._refresh_editability()
                self._set_status(f"Added {amount} taxel(s).")
            except Exception as error:
                self._report_error(f"Add Taxels rejected: {error}", event.client)

        for handle, field in (
            (self.position_input, "position"),
            (self.normal_input, "normal"),
            (self.tangent_input, "tangent"),
            (self.orientation_text, "orientation"),
            (self.parent_text, "parent"),
            (self.indices_text, "indices"),
            (self.weights_text, "weights"),
        ):

            @handle.on_update
            def _(event, field=field) -> None:
                if self.selected_kind is None or len(self.selected_indices) != 1:
                    return
                self._edit_field_event(
                    event,
                    self.selected_kind,
                    next(iter(self.selected_indices)),
                    field,
                )

    def _set_pointer_mode(self) -> None:
        self.server.scene.remove_click_callback()
        self.server.scene.remove_rect_select_callback()
        if self.pointer_mode == "Box select":

            @self.server.scene.on_rect_select()
            def _(event) -> None:
                if not self._claim_editor(event.client_id, event.client):
                    return
                kind = "joint" if "shift" in _modifier(event) else "taxel"
                points = (
                    self._display_joint_pose()[0]
                    if kind == "joint"
                    else self._display_taxel_positions
                )
                screen, in_front = project_points(
                    points,
                    event.client.camera.position,
                    event.client.camera.wxyz,
                    event.client.camera.fov,
                    event.client.camera.aspect,
                )
                lower = np.minimum(event.screen_min, event.screen_max)
                upper = np.maximum(event.screen_min, event.screen_max)
                inside = (
                    in_front
                    & np.all(screen >= lower, axis=1)
                    & np.all(screen <= upper, axis=1)
                )
                self._select(kind, set(np.flatnonzero(inside).tolist()))

        else:

            @self.server.scene.on_click()
            def _(event) -> None:
                if not self._claim_editor(event.client_id, event.client):
                    return
                points = np.concatenate(
                    (self._display_taxel_positions, self._display_joint_pose()[0]),
                    axis=0,
                )
                if len(points):
                    screen, visible = project_points(
                        points,
                        event.client.camera.position,
                        event.client.camera.wxyz,
                        event.client.camera.fov,
                        event.client.camera.aspect,
                    )
                    distance = np.linalg.norm(screen - event.screen_pos, axis=1)
                    if np.any(visible & (distance < 0.025)):
                        return
                self._select(None, set())

    def _claim_editor(self, client_id: int | None, client: Any | None) -> bool:
        if client_id is None:
            return True
        if self.editor_client_id is None:
            self.editor_client_id = client_id
        if client_id == self.editor_client_id:
            return True
        if client is not None:
            client.add_notification(
                title="Read-only client",
                body="Another browser owns the editing lease.",
                color="yellow",
            )
        return False

    def _set_value(self, handle: Any, value: Any) -> None:
        self._programmatic = True
        try:
            handle.value = value
        finally:
            self._programmatic = False

    def _set_status(self, message: str) -> None:
        self._last_error = ""
        self.file_status.content = message

    def _report_error(self, message: str, client: Any | None = None) -> None:
        logger.error("%s", message)
        self._last_error = message
        self.file_status.content = f"**Error:** {message}"
        self._update_state_text()
        if client is not None:
            client.add_notification(
                title="Tactile visualizer", body=message, color="red"
            )

    def _deliver_file(
        self, payload: bytes, filename: str, client: Any | None, message: str
    ) -> None:
        clean_name = Path(filename).name
        if not clean_name:
            raise ValueError("filename is empty")
        results = []
        if client is not None:
            client.send_file_download(clean_name, payload)
            results.append(f"browser download `{clean_name}`")
        if self.write_root is not None:
            target = atomic_write_within(self.write_root, filename, payload)
            results.append(f"atomic server write `{target}`")
        if not results:
            raise RuntimeError("no browser client and no server write root")
        self._set_status(f"{message} Delivered by " + " and ".join(results) + ".")

    def _replace_document(self, document: ShapeDocument) -> None:
        document.validate()
        self.document = document
        self.selected_kind = None
        self.selected_indices.clear()
        self._remove_selection_visuals()
        self._build_schema_tree()
        self._build_scene_handles()
        self._refresh_editability()

    def _edit_field_event(self, event: Any, kind: str, index: int, field: str) -> None:
        if self._programmatic:
            return
        if not self._claim_editor(event.client_id, event.client):
            self._sync_tree_values()
            self._refresh_property_values()
            return
        if not self.controller.editing_enabled:
            self._report_error("Pause live playback before editing.", event.client)
            self._sync_tree_values()
            self._refresh_property_values()
            return
        backup = self.document.copy()
        try:
            self._apply_field(kind, index, field, event.target.value)
            self.document.validate()
            if kind == "joint" and field == "parent":
                self._rebuild_bones()
            self._mark_local_edit()
            self._sync_tree_values()
            self._update_rendered_values()
        except Exception as error:
            self.document = backup
            self._report_error(f"Edit rejected: {error}", event.client)
            self._sync_tree_values()
            self._update_rendered_values()

    def _apply_field(self, kind: str, index: int, field: str, raw: Any) -> None:
        if kind == "joint":
            if field == "position":
                value = _validate_gui_vector(raw) / 100.0
                self.document.joint_bind_positions[index] = value
            elif field == "orientation":
                self.document.joint_bind_orientations[index] = normalize(
                    _parse_numbers(str(raw), 4), label="bind quaternion"
                )
            elif field == "parent":
                text = str(raw).strip()
                parent = (
                    int(ROOT_PARENT) if text.upper() == "ROOT_PARENT" else int(text)
                )
                if parent == index:
                    raise ValueError("a joint cannot be its own parent")
                if (
                    parent != int(ROOT_PARENT)
                    and not 0 <= parent < self.document.num_joints
                ):
                    raise ValueError("parent is outside the joint array")
                self.document.joint_parents[index] = np.uint32(parent)
            else:
                raise ValueError(f"{field} is not a joint field")
            return
        if kind != "taxel":
            raise ValueError("no editable object selected")
        if field == "position":
            value = _validate_gui_vector(raw) / 100.0
            self.document.taxel_positions[index] = value
        elif field in ("normal", "tangent"):
            normal = (
                _validate_gui_vector(raw)
                if field == "normal"
                else self.document.taxel_normals[index]
            )
            tangent = (
                _validate_gui_vector(raw)
                if field == "tangent"
                else self.document.taxel_tangents[index]
            )
            from .model import orthonormalize_frame

            normal, tangent = orthonormalize_frame(normal, tangent)
            self.document.taxel_normals[index] = normal
            self.document.taxel_tangents[index] = tangent
        elif field == "indices":
            value = _parse_numbers(str(raw), 4, int)
            if np.any(value < 0) or np.any(value >= self.document.num_joints):
                raise ValueError("joint index is outside the active shape")
            self.document.taxel_joint_indices[index] = value.astype(np.uint32)
        elif field == "weights":
            value = _parse_numbers(str(raw), 4)
            if np.any(value < 0) or not np.isclose(value.sum(), 1.0, atol=1e-4):
                raise ValueError("weights must be nonnegative and sum to one")
            self.document.taxel_joint_weights[index] = value.astype(np.float32)
        else:
            raise ValueError(f"{field} is not a taxel field")

    def _select(self, kind: str | None, indices: set[int]) -> None:
        self.selected_kind = kind if indices else None
        self.selected_indices = indices
        self._refresh_property_values()
        self._update_selection_gizmo()
        self._update_rendered_values(update_gizmo=False)

    def _refresh_property_values(self) -> None:
        kind = self.selected_kind
        count = len(self.selected_indices)
        taxel_single = kind == "taxel" and count == 1
        joint_single = kind == "joint" and count == 1
        all_fields = (
            self.position_input,
            self.normal_input,
            self.tangent_input,
            self.orientation_text,
            self.parent_text,
            self.indices_text,
            self.weights_text,
            self.force_text,
            self.current_position_text,
            self.current_orientation_text,
        )
        for handle in all_fields:
            handle.visible = False
        if not (taxel_single or joint_single):
            self.selection_summary.content = (
                "No selection" if count == 0 else f"**{count} {kind}(s)**"
            )
            self.selection_placeholder.content = "" if count == 0 else "`-`"
            return
        self.selection_placeholder.content = ""
        index = next(iter(self.selected_indices))
        self.position_input.visible = True
        if taxel_single:
            self.selection_summary.content = "**1 taxel(s)**"
            for handle in (
                self.normal_input,
                self.tangent_input,
                self.indices_text,
                self.weights_text,
                self.force_text,
            ):
                handle.visible = True
            self._set_value(
                self.position_input,
                tuple((self.document.taxel_positions[index] * 100.0).tolist()),
            )
            self._set_value(
                self.normal_input, tuple(self.document.taxel_normals[index].tolist())
            )
            self._set_value(
                self.tangent_input,
                tuple(self.document.taxel_tangents[index].tolist()),
            )
            self._set_value(
                self.indices_text,
                ", ".join(map(str, self.document.taxel_joint_indices[index])),
            )
            self._set_value(
                self.weights_text,
                self._format_vector(self.document.taxel_joint_weights[index]),
            )
            forces = self.display_forces.get(self.side)
            force = (
                "unavailable"
                if forces is None or index >= len(forces)
                else f"{np.linalg.norm(forces[index]):.2f} N"
            )
            self._set_value(self.force_text, force)
            return
        name = (
            OPENXR_JOINT_NAMES[index] if index < len(OPENXR_JOINT_NAMES) else str(index)
        )
        self.selection_summary.content = f"**1 joint(s): {name}**"
        for handle in (
            self.orientation_text,
            self.parent_text,
            self.current_position_text,
            self.current_orientation_text,
        ):
            handle.visible = True
        self._set_value(
            self.position_input,
            tuple((self.document.joint_bind_positions[index] * 100.0).tolist()),
        )
        self._set_value(
            self.orientation_text,
            self._format_vector(self.document.joint_bind_orientations[index]),
        )
        parent = int(self.document.joint_parents[index])
        self._set_value(
            self.parent_text,
            "ROOT_PARENT" if parent == int(ROOT_PARENT) else str(parent),
        )
        positions, orientations = self._display_joint_pose()
        self._set_value(
            self.current_position_text,
            self._format_vector(positions[index] * 100.0),
        )
        self._set_value(
            self.current_orientation_text,
            self._format_vector(orientations[index]),
        )

    def _remove_selection_visuals(self) -> None:
        if self._gizmo is not None:
            self._gizmo.remove()
            self._gizmo = None
        if self._selection_axes is not None:
            self._selection_axes.remove()
            self._selection_axes = None
        self._gizmo_context = None

    def _update_selection_gizmo(self) -> None:
        self._remove_selection_visuals()
        if not self.selected_indices:
            return
        indices = sorted(self.selected_indices)
        if self.selected_kind == "joint":
            if len(indices) == 1:
                positions, orientations = self._display_joint_pose()
                index = indices[0]
                self._selection_axes = self.server.scene.add_frame(
                    "/selection/joint_axes",
                    show_axes=True,
                    axes_length=0.015,
                    axes_radius=0.0005,
                    position=positions[index],
                    wxyz=orientations[index][[3, 0, 1, 2]],
                )
            return
        if self.selected_kind != "taxel":
            return
        center = np.mean(self._display_taxel_positions[indices], axis=0)
        single = len(indices) == 1
        orientation = (
            frame_wxyz(
                self._display_taxel_normals[indices[0]],
                self._display_taxel_tangents[indices[0]],
            )
            if single
            else np.array([1.0, 0.0, 0.0, 0.0])
        )
        if not self.controller.editing_enabled:
            if single:
                self._selection_axes = self.server.scene.add_frame(
                    "/selection/taxel_axes",
                    show_axes=True,
                    axes_length=0.015,
                    axes_radius=0.0005,
                    position=center,
                    wxyz=orientation,
                )
            return
        translating = self.transform_mode == "Translate"
        self._gizmo = self.server.scene.add_transform_controls(
            "/selection/taxel_gizmo",
            scale=0.035,
            position=center,
            wxyz=orientation,
            disable_axes=not translating,
            disable_sliders=not translating,
            disable_rotations=translating,
        )
        self._gizmo_context = {
            "indices": indices,
            "origin_position": center.copy(),
            "origin_rotation": quaternion_xyzw_to_matrix(orientation[[1, 2, 3, 0]]),
            "positions": self._display_taxel_positions[indices].copy(),
            "normals": self._display_taxel_normals[indices].copy(),
            "tangents": self._display_taxel_tangents[indices].copy(),
            "current_positions": self._current_positions.copy(),
            "current_orientations": self._current_orientations.copy(),
        }

        @self._gizmo.on_update
        def _(event) -> None:
            if not self._claim_editor(event.client_id, event.client):
                return
            if not self.controller.editing_enabled or self._gizmo_context is None:
                return
            context = self._gizmo_context
            backup = self.document.copy()
            try:
                if self.transform_mode == "Translate":
                    delta = (
                        np.asarray(event.target.position) - context["origin_position"]
                    )
                    for slot, index in enumerate(context["indices"]):
                        set_taxel_display_position(
                            self.document,
                            context["current_positions"],
                            context["current_orientations"],
                            index,
                            context["positions"][slot] + delta,
                            self.side,
                        )
                else:
                    current_q = np.asarray(event.target.wxyz)[[1, 2, 3, 0]]
                    rotation = (
                        quaternion_xyzw_to_matrix(current_q)
                        @ context["origin_rotation"].T
                    )
                    for slot, index in enumerate(context["indices"]):
                        set_taxel_display_frame(
                            self.document,
                            context["current_positions"],
                            context["current_orientations"],
                            index,
                            rotation @ context["normals"][slot],
                            rotation @ context["tangents"][slot],
                            self.side,
                        )
                self.document.validate()
                self._mark_local_edit()
            except Exception as error:
                self.document = backup
                self._report_error(f"Gizmo edit rejected: {error}", event.client)
            self._sync_tree_values()
            self._update_rendered_values(update_gizmo=False)

    def _mark_local_edit(self) -> None:
        self.user_shape_override = True
        self.file_status.content = "Active shape has unexported local edits."

    def _refresh_editability(self) -> None:
        enabled = self.controller.editing_enabled
        for handle in (
            self.position_input,
            self.normal_input,
            self.tangent_input,
            self.orientation_text,
            self.parent_text,
            self.indices_text,
            self.weights_text,
            self.upload,
            self.create_new,
            self.add_taxel_count,
            self.add_taxels_button,
        ):
            handle.disabled = not enabled
        self.force_text.disabled = True
        self.current_position_text.disabled = True
        self.current_orientation_text.disabled = True
        for fields in self._tree_joint_inputs + self._tree_taxel_inputs:
            for handle in fields.values():
                handle.disabled = not enabled
        self._update_selection_gizmo()
        self._update_state_text()

    def _update_state_text(self) -> None:
        source = "offline editor" if self.source is None else "latest-only source"
        override = "local override" if self.user_shape_override else "producer metadata"
        error = f"  \n**Error:** {self._last_error}" if self._last_error else ""
        self.state_markdown.content = (
            f"**{self.controller.state.value}** · {source} · {self.side} hand · "
            f"{override}{error}"
        )
        self.inspector_status.content = (
            f"{self.document.num_joints} joints · {self.document.num_taxels} taxels"
        )

    def _active_hand(self):
        if self.display_snapshot is None:
            return None
        return self.display_snapshot.hands.get(self.side)

    def _display_joint_pose(self) -> tuple[np.ndarray, np.ndarray]:
        positions = self._current_positions
        orientations = self._current_orientations
        if self.side == "left":
            positions, orientations = mirror_pose(positions, orientations)
        return np.asarray(positions, dtype=np.float32), np.asarray(
            orientations, dtype=np.float32
        )

    def _update_rendered_values(self, *, update_gizmo: bool = True) -> None:
        hand = self._active_hand()
        self._current_positions, self._current_orientations = resolve_current_joints(
            self.document,
            None if hand is None else hand.positions,
            None if hand is None else hand.orientations,
            None if hand is None else hand.valid,
        )
        positions, normals, tangents = skin_taxels(
            self.document, self._current_positions, self._current_orientations
        )
        positions, normals, tangents = apply_display_side(
            positions, normals, tangents, self.side
        )
        assert normals is not None and tangents is not None
        self._display_taxel_positions = positions
        self._display_taxel_normals = normals
        self._display_taxel_tangents = tangents
        joint_positions, _ = self._display_joint_pose()
        for index, handle in enumerate(self._joint_handles):
            handle.position = joint_positions[index]
            handle.color = (
                (255, 40, 220)
                if self.selected_kind == "joint" and index in self.selected_indices
                else (255, 180, 60)
            )
        for child, handle in self._bone_handles.items():
            parent = int(self.document.joint_parents[child])
            delta = joint_positions[child] - joint_positions[parent]
            if float(np.linalg.norm(delta)) >= 1e-8:
                handle.position = (
                    joint_positions[child] + joint_positions[parent]
                ) / 2.0
                handle.wxyz = self._segment_wxyz(delta)
        colors = force_colors(
            self.display_forces.get(self.side), self.document.num_taxels
        )
        for index, handle in enumerate(self._taxel_handles):
            handle.position = positions[index]
            handle.wxyz = frame_wxyz(normals[index], tangents[index])
            handle.color = (
                (255, 30, 220)
                if self.selected_kind == "taxel" and index in self.selected_indices
                else tuple(int(value) for value in colors[index])
            )
        self._update_data_tree()
        raw = (
            None
            if self.display_snapshot is None
            else self.display_snapshot.raw_joints.get(self.side)
        )
        if self.urdf is not None:
            try:
                self.urdf.update_cfg(
                    map_raw_joints(
                        self.urdf_joint_names, raw, self.urdf_config.joint_map
                    )
                )
            except Exception as error:
                logger.exception("URDF animation update failed")
                self._last_error = f"URDF update failed: {error}"
        if update_gizmo and self.selected_indices:
            self._update_selection_gizmo()
        self._refresh_property_values()

    def _update_data_tree(self) -> None:
        forces = self.display_forces.get(self.side)
        if forces is None:
            self._data_tree_text.content = "`taxel_force` (N): no sample"
            return
        if len(forces) != self.document.num_taxels:
            self._data_tree_text.content = (
                f"**Rejected layout:** `taxel_force` has {len(forces)} entries; "
                f"TactileShape has {self.document.num_taxels} taxels."
            )
            return
        lines = [f"`taxel_force` (N) [{len(forces)}]"]
        for index, force in enumerate(forces):
            lines.append(
                f"- `[{index}]`: [{self._format_vector(force)}] · "
                f"|F| `{np.linalg.norm(force):.2f} N`"
            )
        self._data_tree_text.content = "\n".join(lines)

    def _source_ready(self, snapshot: SourceSnapshot) -> bool:
        shape = self.latest_shapes.get(self.side)
        forces = self.latest_forces.get(self.side)
        hand = snapshot.hands.get(self.side)
        return (
            hand is not None
            and len(hand.valid) > WRIST
            and bool(hand.valid[WRIST])
            and self.side in snapshot.raw_joints
            and shape is not None
            and forces is not None
            and len(forces) == shape.num_taxels
        )

    @staticmethod
    def _validate_snapshot(snapshot: SourceSnapshot) -> None:
        for side, hand in snapshot.hands.items():
            positions = np.asarray(hand.positions)
            orientations = np.asarray(hand.orientations)
            valid = np.asarray(hand.valid)
            if (
                positions.ndim != 2
                or positions.shape[1:] != (3,)
                or orientations.shape != (len(positions), 4)
                or valid.shape != (len(positions),)
                or not np.isfinite(positions).all()
                or not np.isfinite(orientations).all()
            ):
                raise ValueError(f"{side} hand pose has an invalid layout or value")
            norms = np.linalg.norm(orientations, axis=1)
            if np.any(np.asarray(valid, dtype=bool) & (norms < 1e-8)):
                raise ValueError(f"{side} hand pose has a zero valid quaternion")
        for side, values in snapshot.raw_joints.items():
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError(f"{side} raw-joint sample contains a non-finite value")
        for shape in snapshot.shapes.values():
            shape.validate()
        for side, forces in snapshot.forces.items():
            value = np.asarray(forces)
            if (
                value.ndim != 2
                or value.shape[1:] != (3,)
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"{side} tactile force must be a finite (N, 3) array")

    def _consume_source(self, now: float) -> None:
        if self.source is None:
            return
        sequence, snapshot, error = self.source.mailbox.latest_after(self.sequence)
        self.sequence = sequence
        if error is not None:
            self._last_error = f"source message rejected: {error}"
        if snapshot is None:
            return
        try:
            self._validate_snapshot(snapshot)
        except Exception as invalid:
            self._last_error = f"malformed live message rejected: {invalid}"
            return
        if self._last_error.startswith(("source message rejected:", "malformed live")):
            self._last_error = ""
        self.latest_snapshot = snapshot
        self.latest_shapes.update(snapshot.shapes)
        self.latest_forces.update(snapshot.forces)
        previous = self.controller.state
        self.controller.message_received(
            snapshot.received_at, ready=self._source_ready(snapshot)
        )
        if not self.controller.paused:
            self.display_snapshot = snapshot
            self.display_forces = {
                side: value.copy() for side, value in self.latest_forces.items()
            }
            if (
                not self.user_shape_override
                and self.side in snapshot.shapes
                and self.side not in self._loaded_remote_sides
            ):
                self._replace_document(snapshot.shapes[self.side].copy())
                self._loaded_remote_sides.add(self.side)
            else:
                self._update_rendered_values()
        if previous != self.controller.state:
            self._refresh_editability()

    def run(self, update_hz: float = 30.0) -> None:
        if self.source is not None:
            self.source.start()
        period = 1.0 / update_hz
        try:
            while True:
                started = time.monotonic()
                try:
                    self._consume_source(started)
                    previous = self.controller.state
                    self.controller.tick(started)
                    if previous != self.controller.state:
                        self._refresh_editability()
                    else:
                        self._update_state_text()
                except Exception as error:
                    logger.exception("Visualizer update rejected")
                    self._last_error = f"Update rejected: {error}"
                    self._update_state_text()
                time.sleep(max(0.0, period - (time.monotonic() - started)))
        except KeyboardInterrupt:
            pass
        finally:
            if self.source is not None:
                self.source.stop()
