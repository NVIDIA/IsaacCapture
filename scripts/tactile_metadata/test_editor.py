# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from tactile_metadata_editor.document import Segment, State, Taxel
from tactile_metadata_editor.ui import Editor


class EditorTests(unittest.TestCase):
    def make_editor(self):
        editor = Editor.__new__(Editor)
        editor.state = State()
        editor.document.segments = [Segment("WRIST")]
        editor.document.taxels = [
            Taxel(segments=["WRIST", None, None, None], weights=[1.0, 0.0, 0.0, 0.0])
        ]
        editor.preview = {"left": editor._preview_from(editor.document)}
        editor.binding_source_live = SimpleNamespace(value=False)
        editor.info = SimpleNamespace(value="")
        editor.edit_joint_widgets = {}
        editor._rebuild_edit_lists = Mock()
        editor._sync_joint_widget_values = Mock()
        return editor

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


if __name__ == "__main__":
    unittest.main()
