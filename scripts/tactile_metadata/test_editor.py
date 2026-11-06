# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import queue
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from tactile_metadata_editor import geometry as geo, storage
from tactile_metadata_editor.document import JOINTS, Segment, State, Taxel
from tactile_metadata_editor.ui import ERROR_COLOR, TAXEL_COLOR, Editor


class EditorTests(unittest.TestCase):
    def test_apply_delta_transforms_duplicate_rows_once(self):
        segments = [Segment("WRIST"), Segment("INDEX_TIP")]
        delta = np.eye(4)
        delta[0, 3] = 0.01

        geo.apply_delta(segments, [0, 0], delta)

        np.testing.assert_allclose(segments[0].pose[:3], [0.01, 0.0, 0.0])
        np.testing.assert_allclose(segments[1].pose[:3], [0.0, 0.0, 0.0])

    def make_editor(self):
        editor = Editor.__new__(Editor)
        editor.state = State()
        editor.document.segments = [Segment("WRIST")]
        editor.document.taxels = [
            Taxel(segments=["WRIST", None, None, None], weights=[1.0, 0.0, 0.0, 0.0])
        ]
        editor.preview_document = editor._preview_from(editor.document)
        editor.binding_source_live = SimpleNamespace(value=False)
        editor.info = SimpleNamespace(value="")
        editor.binding_phase, editor.pick_mode = None, None
        editor.events = queue.SimpleQueue()
        editor.server = SimpleNamespace(
            gui=SimpleNamespace(_gui_input_handle_from_uuid={})
        )
        editor.server.gui.add_button = lambda *args, **kwargs: self.make_input(editor)
        editor.edit_joint_widgets = {}
        editor._rebuild_edit_lists = Mock()
        editor._sync_joint_widget_values = Mock()
        return editor

    def make_input(self, editor, value=None):
        widget = SimpleNamespace(
            value=value, disabled=False, _impl=SimpleNamespace(update_cb=[])
        )
        widget.on_update = widget._impl.update_cb.append
        widget.on_click = widget._impl.update_cb.append
        inputs = editor.server.gui._gui_input_handle_from_uuid
        inputs[str(len(inputs))] = widget
        return widget

    def make_procedure_editor(self):
        editor = self.make_editor()
        editor.mode_edit = SimpleNamespace(value=True)
        editor.selected_kind, editor.selected_rows = "taxels", [0]
        editor.binding_info = SimpleNamespace(content="")
        editor.binding_start = self.make_input(editor)
        editor.binding_next = self.make_input(editor)
        editor.binding_complete = self.make_input(editor)
        editor._button(editor.binding_start, editor._start_binding)
        editor._button(editor.binding_next, editor._select_binding_segments)
        editor._button(editor.binding_complete, editor._complete_binding)
        for mode, name in (
            ("segment", "segment_click_button"),
            ("taxel_pose", "taxel_capture_button"),
            ("segment_pose", "segment_capture_button"),
        ):
            button = editor._action_button(
                mode, lambda mode=mode: editor._toggle_pick_mode(mode)
            )
            setattr(editor, name, button)
        editor._sync_selection = Mock()
        editor._sync_taxel_binding_widgets = Mock()
        editor._document_error_html = Mock(return_value="")
        return editor

    @staticmethod
    def dispatch(widget, client=None):
        async def emit():
            for callback in widget._impl.update_cb:
                await callback(SimpleNamespace(client=client, target=widget))

        asyncio.run(emit())

    @staticmethod
    def drain_events(editor):
        while not editor.events.empty():
            editor.events.get_nowait()()

    def make_live_editor(self, frame):
        editor = self.make_editor()
        editor.receiver = SimpleNamespace(status="Live", active=True, latest=frame)
        editor.live_status = SimpleNamespace(content="")
        editor.live_document, editor.live_metadata_key = None, ""
        editor.binding_source_live.disabled = True
        editor.joint_enabled = SimpleNamespace(value=False)
        editor.joint_mapping = SimpleNamespace(value="{}")
        editor.preview_error = editor.info
        editor._sync_preview_values = Mock()
        editor._rebuild_preview_segments = Mock()
        editor._update_preview_error = Mock()
        editor.mode_edit = SimpleNamespace(value=False)
        editor.selected_kind, editor.selected_rows = None, []
        editor._sync_selection = Mock()
        return editor

    def test_malformed_live_input_reports_an_error_without_raising(self):
        for frame, mapping, message in (
            ({"metadata": {}}, "{}", "Expected geometry_mapping"),
            ({}, "[]", "Joint Mappings JSON must be an object"),
            ({}, "null", "Joint Mappings JSON must be an object"),
            ({}, "42", "Joint Mappings JSON must be an object"),
        ):
            with self.subTest(frame=frame, mapping=mapping):
                editor = self.make_live_editor(frame)
                editor.joint_enabled.value = True
                editor.joint_mapping.value = mapping

                editor._update_live()

                self.assertIn("Live error: " + message, editor.info.value)
                self.assertTrue(editor.redraw)

    def test_invalid_live_hand_retains_applicable_pose_and_recovers_status(self):
        valid_hand = {
            "positions": [[0.0, 0.0, 0.0] for _ in JOINTS],
            "orientations": [[0.0, 0.0, 0.0, 1.0] for _ in JOINTS],
            "is_valid": [True for _ in JOINTS],
        }
        invalid_wrist = {
            **valid_hand,
            "is_valid": [joint != "WRIST" for joint in JOINTS],
        }
        for hand in (None, invalid_wrist):
            with self.subTest(hand=hand):
                editor = self.make_live_editor({"hand": hand})
                retained = editor.preview_document.segments
                retained[0].pose[0] = 0.02

                editor._update_live()

                self.assertIs(editor.preview_document.segments, retained)
                self.assertEqual(
                    editor.live_status.content, "Current Status: Data Invalid"
                )
                editor._rebuild_preview_segments.assert_not_called()
                editor._apply_preview_segments()
                self.assertEqual(editor.document.segments, retained)

                editor.receiver.latest = {"hand": valid_hand}
                editor._update_live()

                self.assertEqual(editor.live_status.content, "Current Status: Live")
                self.assertEqual(
                    [segment.joint for segment in editor.preview_document.segments],
                    list(JOINTS),
                )

    def test_live_joint_changes_clear_only_preview_selection(self):
        original = ["PALM", "WRIST", "THUMB_METACARPAL"]
        for editing, joints in (
            (False, original[1:]),
            (False, original),
            (True, original[1:]),
        ):
            with self.subTest(editing=editing, joints=joints):
                editor = self.make_live_editor({"hand": {}})
                editor.mode_edit.value = editing
                editor.preview_document.segments = [
                    Segment(joint) for joint in original
                ]
                editor.selected_kind, editor.selected_rows = "segments", [1]
                with patch.object(
                    geo,
                    "hand_segments",
                    return_value=[Segment(joint) for joint in joints],
                ):
                    editor._update_live()

                cleared = not editing and joints != original
                self.assertEqual(editor.selected_kind, None if cleared else "segments")
                self.assertEqual(editor.selected_rows, [] if cleared else [1])
                self.assertEqual(editor._sync_selection.call_count, int(cleared))

    def test_reset_preview_refreshes_selection_without_retargeting_it(self):
        for editing, joints in (
            (False, ["PALM", "WRIST"]),
            (False, ["WRIST"]),
            (True, ["PALM", "WRIST"]),
        ):
            with self.subTest(editing=editing, joints=joints):
                editor = self.make_live_editor({})
                editor.mode_edit.value = editing
                editor.preview_document.segments = [Segment(joint) for joint in joints]
                editor.document.segments[0].pose[0] = 0.02
                editor.selected_kind, editor.selected_rows = "segments", [0]

                editor._restore_preview()

                cleared = not editing and joints != ["WRIST"]
                self.assertEqual(editor.selected_kind, None if cleared else "segments")
                self.assertEqual(editor.selected_rows, [] if cleared else [0])
                self.assertEqual(editor._sync_selection.call_count, int(not editing))
                self.assertEqual(
                    editor.preview_document.segments, editor.document.segments
                )

    def test_apply_all_preserves_rendered_taxels(self):
        editor = self.make_editor()
        editor.preview_document.segments[0].pose[0] = 0.01
        editor.preview_document.q = {"joint": 0.5}
        rows, errors, _ = editor._rendered_preview_taxels()
        self.assertFalse(errors)
        expected = rows[0][1].pose
        self.assertAlmostEqual(expected[0], 0.01)

        editor._apply_preview_all()

        np.testing.assert_allclose(editor.document.taxels[0].pose, expected)
        self.assertEqual(editor.document.segments, editor.preview_document.segments)
        self.assertEqual(editor.document.q, editor.preview_document.q)

    def test_joint_slider_initialization_matches_the_rendered_configuration(self):
        for lower, upper, initial, expected in (
            (1.0, 2.0, None, 1.0),
            (-2.0, -1.0, None, -1.0),
            (1.0, 2.0, 4.0, 2.0),
        ):
            with self.subTest(lower=lower, upper=upper, initial=initial):
                editor = self.make_editor()
                editor.model = SimpleNamespace(
                    joints=[
                        SimpleNamespace(
                            name="joint",
                            limit=SimpleNamespace(lower=lower, upper=upper),
                        )
                    ]
                )
                editor.server = SimpleNamespace(gui=Mock())
                editor._watch = Mock()
                if initial is not None:
                    editor.document.q["joint"] = initial

                editor._add_joint_sliders(editor.document, {})

                self.assertEqual(editor.document.q["joint"], expected)
                self.assertEqual(
                    editor.server.gui.add_slider.call_args.args[-1], expected
                )

    def test_load_preserves_current_alignment_and_visible_poses(self):
        editor = self.make_editor()
        editor.document.alignment = geo.transform(
            (0.12, 0.03, -0.04), (10.0, 20.0, 30.0)
        ).tolist()
        expected = editor.document.clone()
        editor._sync_document = Mock()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            storage.save(editor.state, path)
            editor.metadata_path = SimpleNamespace(value=str(path))

            editor._load()

        for document in (editor.document, editor.state.saved, editor.preview_document):
            np.testing.assert_allclose(document.alignment, expected.alignment)
            np.testing.assert_allclose(
                document.segments[0].pose, expected.segments[0].pose, atol=1e-12
            )
        for document in (editor.document, editor.state.saved):
            np.testing.assert_allclose(
                document.taxels[0].pose, expected.taxels[0].pose, atol=1e-12
            )
        editor._sync_document.assert_called_once()

    def test_error_html_escapes_external_text(self):
        editor = self.make_editor()
        payload = '<img src=x onerror="alert(1)">'
        editor.document.taxels[0].segments[0] = payload
        validation = editor._document_error_html()

        editor.preview_error = editor.info
        editor._rendered_preview_taxels = Mock(return_value=([], [payload], 0))
        editor._update_preview_error()
        preview = editor.info.value

        widget = SimpleNamespace(content="", visible=False)
        editor._set_error_widget(widget, [payload, payload])
        self.assertTrue(widget.visible)
        self.assertEqual(widget.content.count("&lt;img"), 1)

        with self.assertLogs("isaaccapture.tactile_metadata.editor", level="ERROR"):
            editor._run_action(Mock(side_effect=ValueError(payload)))
        for rendered in (validation, preview, widget.content, editor.info.value):
            self.assertIn("&lt;img src=x onerror=&quot;alert(1)&quot;&gt;", rendered)
            self.assertNotIn(payload, rendered)

    def test_list_pose_edits_refresh_selection_before_further_inspector_edits(self):
        for editing, kind in (
            (True, "segments"),
            (True, "taxels"),
            (False, "segments"),
        ):
            for rows in ([0], [0, 1]):
                with self.subTest(editing=editing, kind=kind, rows=rows):
                    editor = self.make_editor()
                    editor.mode_edit = SimpleNamespace(value=editing)
                    values = (
                        editor._values(kind)
                        if editing
                        else editor.preview_document.segments
                    )
                    values.append(
                        Segment("INDEX_TIP") if kind == "segments" else Taxel()
                    )
                    editor.selected_kind, editor.selected_rows = kind, rows.copy()
                    editor.selection_folder = object()
                    for name in (
                        "selection_summary",
                        "selection_names",
                        "selection_segment_editor",
                        "selection_taxel_editor",
                        "selection_segment_joint",
                        "selection_scale_widget",
                        "selection_delete",
                    ):
                        setattr(editor, name, self.make_input(editor))
                    editor.selection_position_widgets = [
                        self.make_input(editor, 0.0) for _ in range(3)
                    ]
                    editor.selection_rotation_widgets = [
                        self.make_input(editor, 0.0) for _ in range(3)
                    ]
                    editor.selection_taxel_binding_widgets = [
                        (self.make_input(editor), self.make_input(editor))
                        for _ in range(4)
                    ]
                    editor._update_preview_error = Mock()
                    editor._sync_selected_list_values = Mock()
                    editor._sync_selection()

                    if editing:
                        editor._set_pose_component(kind, 0, 0, 0.02)
                        editor._set_pose_rotation(kind, 0, 2, 30.0)
                    else:
                        editor._set_preview_pose(0, 0, 0.02, False)
                        editor._set_preview_pose(0, 2, 30.0, True)

                    position = editor.selection_position_widgets[0].value
                    rotation = editor.selection_rotation_widgets[2].value
                    self.assertAlmostEqual(position, 0.02 / len(rows))
                    self.assertAlmostEqual(rotation, 30.0 if len(rows) == 1 else 0.0)
                    editor._set_selection_position(0, position + 0.001)
                    editor._set_selection_rotation(2, rotation + 1.0)
                    center = np.mean([values[row].pose[:3] for row in rows], axis=0)
                    self.assertAlmostEqual(center[0], position + 0.001)
                    self.assertAlmostEqual(
                        geo.components(geo.matrix(values[0].pose))[1][2], 31.0
                    )

    def test_binding_overlay_follows_selected_source_and_its_validation(self):
        editor = self.make_editor()
        editor.live_document = editor.document.clone()
        editor.live_document.segments[0].pose[0] = 0.02
        editor.live_document.taxels[0].pose[0] = 0.03
        editor.document.taxels[0].segments[0] = "MISSING"
        editor.model, editor.receiver = None, None
        editor.mode_edit = SimpleNamespace(value=False)
        editor.show_binding = SimpleNamespace(value=True)
        editor.show_saved = SimpleNamespace(value=False)
        editor.selected_kind, editor.selected_rows = None, []
        editor.binding_phase, editor.binding_segments = None, []
        editor.pick_mode = None
        editor._rendered_preview_taxels = Mock(return_value=([], [], 0))
        editor._draw_instances = Mock()
        editor._hide_prefix = Mock()
        editor._draw_selection_gizmo = Mock()

        for live in (False, True):
            with self.subTest(live=live):
                editor.binding_source_live.value = live
                editor._draw_instances.reset_mock()

                editor._draw()

                calls = {
                    call.args[0]: call.args
                    for call in editor._draw_instances.call_args_list
                }
                source = editor.live_document if live else editor.document
                self.assertIs(calls["/edit/segments"][2], source.segments)
                self.assertIs(calls["/edit/taxels"][2], source.taxels)
                self.assertEqual(
                    calls["/edit/taxels"][3], [TAXEL_COLOR if live else ERROR_COLOR]
                )

    def test_start_binding_reuses_only_selected_taxels(self):
        for kind, rows in (
            ("taxels", [0]),
            ("taxels", [0, 1]),
            ("segments", [0]),
            (None, []),
        ):
            with self.subTest(kind=kind, rows=rows):
                editor = self.make_procedure_editor()
                editor.document.taxels.append(Taxel())
                editor.selected_kind, editor.selected_rows = kind, rows.copy()

                editor._start_binding()

                expected = rows if kind == "taxels" else []
                self.assertEqual(editor.binding_taxels, expected)
                self.assertEqual(editor.binding_segments, [])
                self.assertEqual(editor.selected_rows, [])
                self.assertIsNone(editor.selected_kind)
                editor._sync_selection.assert_called_once()
                self.assertEqual(editor.binding_phase, "taxels")
                self.assertTrue(editor.binding_start.disabled)
                self.assertEqual(editor.binding_next.disabled, not expected)
                self.assertTrue(editor.binding_complete.disabled)

    def test_procedures_listen_to_passive_inputs_and_remove_the_listener(self):
        for mode in ("binding", "segment", "taxel_pose", "segment_pose"):
            with self.subTest(mode=mode):
                editor = self.make_procedure_editor()
                widget = self.make_input(editor)
                if mode == "binding":
                    editor._start_binding()
                else:
                    editor._toggle_pick_mode(mode)
                self.assertIsNone(editor.selected_kind)
                self.assertEqual(editor.selected_rows, [])
                editor._sync_selection.assert_called_once()

                self.dispatch(widget)
                widget.disabled = True
                self.dispatch(widget, client=object())
                self.assertTrue(editor.events.empty())

                widget.disabled = False
                self.dispatch(widget, client=object())
                with self.assertLogs(
                    "isaaccapture.tactile_metadata.editor", level="WARNING"
                ) as logged:
                    self.drain_events(editor)
                self.assertEqual(len(logged.records), 1)
                self.assertIsNone(editor.binding_phase)
                self.assertIsNone(editor.pick_mode)
                for handle in editor.server.gui._gui_input_handle_from_uuid.values():
                    self.assertNotIn(
                        editor._procedure_interaction, handle._impl.update_cb
                    )
                self.dispatch(widget, client=object())
                self.assertTrue(editor.events.empty())

    def test_interruption_runs_before_button_and_value_business_callbacks(self):
        for button in (False, True):
            with self.subTest(button=button):
                editor = self.make_procedure_editor()
                observed = []

                def observe(*_, editor=editor, observed=observed):
                    observed.append(editor.binding_phase)

                if button:
                    widget = editor._action_button("Other action", observe)
                else:
                    widget = self.make_input(editor, 2.0)

                    async def flush_interruption(_, editor=editor):
                        self.drain_events(editor)

                    widget.on_update(flush_interruption)
                    editor._watch(widget, observe)
                editor._start_binding()

                with self.assertLogs(
                    "isaaccapture.tactile_metadata.editor", level="WARNING"
                ):
                    self.dispatch(widget, client=object())
                    self.drain_events(editor)

                self.assertEqual(observed, [None])

    def test_binding_buttons_and_scene_picks_preserve_the_flow_until_completion(self):
        editor = self.make_procedure_editor()
        editor._start_binding()
        self.dispatch(editor.binding_next, client=object())
        self.drain_events(editor)
        self.assertEqual(editor.binding_phase, "segments")

        editor._handle_scene_event = Mock()
        asyncio.run(editor._scene_click(SimpleNamespace()))
        self.drain_events(editor)
        editor._handle_scene_event.assert_called_once()
        self.assertEqual(editor.binding_phase, "segments")

        editor.binding_segments = [0]
        editor._set_binding_button("complete")
        self.dispatch(editor.binding_complete, client=object())
        self.drain_events(editor)
        self.assertIsNone(editor.binding_phase)
        self.assertEqual(editor.selected_kind, "taxels")
        self.assertEqual(editor.selected_rows, [0])
        for widget in editor.server.gui._gui_input_handle_from_uuid.values():
            self.assertNotIn(editor._procedure_interaction, widget._impl.update_cb)

    def test_capture_rectangles_do_not_select_objects_or_enable_the_gizmo(self):
        for mode in ("segment", "taxel_pose", "segment_pose"):
            with self.subTest(mode=mode):
                editor = self.make_procedure_editor()
                editor.selection_gizmo = SimpleNamespace(visible=True)
                editor._box_rows = Mock(return_value=[0])
                editor._handle_pick = Mock()
                editor._toggle_pick_mode(mode)
                editor._sync_selection.reset_mock()
                event = SimpleNamespace(modifier="shift")

                editor._handle_scene_event(event, True)
                editor._draw_selection_gizmo()

                self.assertIsNone(editor.selected_kind)
                self.assertEqual(editor.selected_rows, [])
                self.assertEqual(editor.pick_mode, mode)
                self.assertFalse(editor.selection_gizmo.visible)
                editor._sync_selection.assert_not_called()
                editor._box_rows.assert_not_called()
                editor._handle_pick.assert_not_called()

                editor._handle_scene_event(event, False)
                editor._handle_pick.assert_called_once_with(event)

    def test_capture_buttons_toggle_or_replace_the_active_procedure(self):
        editor = self.make_procedure_editor()
        editor._toggle_pick_mode("taxel_pose")
        for button, expected in (
            (editor.segment_capture_button, "segment_pose"),
            (editor.segment_capture_button, None),
        ):
            self.dispatch(button, client=object())
            with self.assertLogs(
                "isaaccapture.tactile_metadata.editor", level="WARNING"
            ) as logged:
                self.drain_events(editor)
            self.assertEqual(len(logged.records), 1)
            self.assertEqual(editor.pick_mode, expected)

    def test_deleted_segment_remains_invalid_after_round_trip(self):
        document = self.make_editor().document
        document.segments.append(Segment("INDEX_TIP"))
        document.taxels[0].segments[0] = "INDEX_TIP"
        document.delete("segments", [1])

        encoded = storage.metadata(document)
        expected = [{"values": [0xFFFFFFFF, 0, 0, 0]}]
        self.assertEqual(encoded["geometry_mapping"]["taxel_segment_indices"], expected)

        loaded = storage.parse(encoded)
        self.assertTrue(
            any(issue["field"] == "segment0" for issue in loaded.validate())
        )
        self.assertEqual(
            storage.metadata(loaded)["geometry_mapping"]["taxel_segment_indices"],
            expected,
        )

    def test_zero_weight_segment_indices_survive_round_trip(self):
        joints = ["WRIST", "INDEX_TIP", "MIDDLE_TIP", "THUMB_TIP"]
        for bindings, indices in (
            (["WRIST", None, None, None], [0, 0, 0, 0]),
            (joints, [0, 1, 2, 3]),
        ):
            with self.subTest(bindings=bindings):
                document = self.make_editor().document
                document.segments = [Segment(joint) for joint in joints]
                document.taxels[0].segments = bindings.copy()
                encoded = storage.metadata(document)

                loaded = storage.parse(encoded)

                self.assertEqual(loaded.taxels[0].segments, ["WRIST", None, None, None])
                self.assertEqual(loaded.taxels[0].weights, [1.0, 0.0, 0.0, 0.0])
                self.assertFalse(loaded.validate())
                self.assertEqual(
                    storage.metadata(loaded)["geometry_mapping"][
                        "taxel_segment_indices"
                    ],
                    [{"values": indices}],
                )

    def test_segment_edits_and_preview_apply_preserve_serialized_slots(self):
        editor = self.make_editor()
        joints = ["WRIST", "INDEX_TIP", "MIDDLE_TIP", "THUMB_TIP"]
        editor.document.segments = [Segment(joint) for joint in joints]
        editor.document.taxels[0].segments = joints.copy()
        editor.state.document = storage.parse(storage.metadata(editor.document))
        editor.preview_document = editor._preview_from(editor.document)
        editor._sync_taxel_binding_widgets = Mock()
        editor._sync_selection = Mock()

        editor._set_taxel_segment(0, 1, "MIDDLE_TIP")
        self.assertEqual(editor.document.taxels[0].stored_segments[1], "MIDDLE_TIP")
        self.assertEqual(
            storage.metadata(editor.document)["geometry_mapping"][
                "taxel_segment_indices"
            ],
            [{"values": [0, 2, 2, 3]}],
        )

        editor._set_taxel_segment(0, 1, "None")
        source = editor.document.taxels[0].stored_segments
        editor._apply_preview_taxels()
        self.assertIsNot(editor.document.taxels[0].stored_segments, source)
        self.assertEqual(
            storage.metadata(editor.document)["geometry_mapping"][
                "taxel_segment_indices"
            ],
            [{"values": [0, 0, 2, 3]}],
        )

    def test_deleting_an_unused_loaded_segment_keeps_padding_valid(self):
        document = self.make_editor().document
        document.segments.append(Segment("INDEX_TIP"))
        document.taxels[0].segments[1] = "INDEX_TIP"
        loaded = storage.parse(storage.metadata(document))

        loaded.delete("segments", [1])

        self.assertFalse(loaded.validate())
        self.assertEqual(
            storage.metadata(loaded)["geometry_mapping"]["taxel_segment_indices"],
            [{"values": [0, 0, 0, 0]}],
        )

    def test_hand_side_change_preserves_document_and_snapshots(self):
        editor = self.make_editor()
        editor.state.replace(editor.document, "metadata.json")
        document, saved = editor.document, editor.state.saved
        preview = editor.preview_document
        receiver = SimpleNamespace(close=Mock())
        editor.receiver = receiver
        editor.connect_button = SimpleNamespace(label="Disconnect Live")
        editor.live_status = SimpleNamespace(content="Live")
        editor._update_preview_error = Mock()
        editor.live_document, editor.live_metadata_key = document.clone(), "left"
        editor.binding_source_editor = SimpleNamespace(value=False)
        editor.binding_source_live = SimpleNamespace(value=True, disabled=False)

        editor._hand_side_changed("right")

        receiver.close.assert_called_once()
        editor._update_preview_error.assert_called_once()
        self.assertIsNone(editor.receiver)
        self.assertIs(editor.document, document)
        self.assertIs(editor.state.saved, saved)
        self.assertIs(editor.preview_document, preview)
        self.assertEqual(editor.state.path, "metadata.json")
        self.assertIsNone(editor.live_document)
        self.assertEqual(editor.live_metadata_key, "")
        self.assertTrue(editor.binding_source_editor.value)
        self.assertFalse(editor.binding_source_live.value)
        self.assertTrue(editor.binding_source_live.disabled)

    def test_connect_discards_metadata_from_the_previous_stream(self):
        editor = self.make_live_editor({})
        editor.receiver = None
        editor.live_document, editor.live_metadata_key = (
            editor.document.clone(),
            "previous",
        )
        editor.binding_source_editor = SimpleNamespace(value=False)
        editor.binding_source_live = SimpleNamespace(value=True, disabled=False)
        editor.hand_side = SimpleNamespace(value="left")
        editor.collection = SimpleNamespace(value="next_collection")
        editor.connect_button = SimpleNamespace(label="Connect Live")
        preview = editor.preview_document

        with patch("tactile_metadata_editor.ui.sdk.Receiver") as receiver:
            editor._connect()

        receiver.assert_called_once_with(
            {"side": "left", "collection": "next_collection", "joint_state": ""}
        )
        self.assertIs(editor.receiver, receiver.return_value)
        self.assertIsNone(editor.live_document)
        self.assertEqual(editor.live_metadata_key, "")
        self.assertTrue(editor.binding_source_editor.value)
        self.assertFalse(editor.binding_source_live.value)
        self.assertTrue(editor.binding_source_live.disabled)
        self.assertIs(editor.preview_document, preview)


if __name__ == "__main__":
    unittest.main()
