# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
        editor.edit_joint_widgets = {}
        editor._rebuild_edit_lists = Mock()
        editor._sync_joint_widget_values = Mock()
        return editor

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
                editor = self.make_editor()
                editor.document.taxels.append(Taxel())
                editor.selected_kind, editor.selected_rows = kind, rows.copy()
                editor._set_pick_mode = Mock()
                editor.binding_info = SimpleNamespace(content="")
                editor.binding_start = SimpleNamespace(disabled=False)
                editor.binding_next = SimpleNamespace(disabled=True)
                editor.binding_complete = SimpleNamespace(disabled=True)

                editor._start_binding()

                expected = rows if kind == "taxels" else []
                self.assertEqual(editor.binding_taxels, expected)
                self.assertEqual(editor.binding_segments, [])
                self.assertEqual(editor.selected_rows, expected)
                self.assertEqual(editor.selected_kind, "taxels" if expected else None)
                self.assertEqual(editor.binding_phase, "taxels")
                self.assertTrue(editor.binding_start.disabled)
                self.assertEqual(editor.binding_next.disabled, not expected)
                self.assertTrue(editor.binding_complete.disabled)

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


if __name__ == "__main__":
    unittest.main()
