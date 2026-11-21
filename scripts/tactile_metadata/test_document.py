# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from tactile_metadata_editor.document import Document, Segment, State, Taxel


class BindingTests(unittest.TestCase):
    def test_segment_edits_update_the_preserved_copy(self):
        taxel = Taxel(
            segments=["WRIST", None, None, None],
            stored_segments=["WRIST", "INDEX_TIP", "WRIST", "WRIST"],
        )

        taxel.set_segment(1, "MIDDLE_TIP")
        self.assertEqual(taxel.segments[1], "MIDDLE_TIP")
        self.assertEqual(taxel.stored_segments[1], "MIDDLE_TIP")

        taxel.set_segment(1, None)
        self.assertIsNone(taxel.segments[1])
        self.assertIsNone(taxel.stored_segments[1])

    def test_assign_replaces_the_preserved_bindings(self):
        document = Document()
        document.segments = [Segment("WRIST"), Segment("INDEX_TIP")]
        document.taxels = [Taxel(stored_segments=["WRIST"] * 4)]

        document.assign([0], [1])

        expected = ["INDEX_TIP", None, None, None]
        self.assertEqual(document.taxels[0].segments, expected)
        self.assertEqual(document.taxels[0].stored_segments, expected)


class StateTests(unittest.TestCase):
    def test_replace_keeps_one_document_and_an_independent_saved_snapshot(self):
        state = State()
        first = Document()
        first.segments = [Segment("WRIST")]
        first.q = {"joint": 0.5}
        state.replace(first, "first.json")

        first.segments[0].pose[0] = 0.01
        first.alignment[0][3] = 0.01
        self.assertEqual(state.saved.segments[0].pose[0], 0.0)
        self.assertEqual(state.saved.alignment[0][3], 0.0)
        self.assertEqual(state.saved.q, {})
        self.assertEqual(first.q, {"joint": 0.5})

        second = Document()
        self.assertEqual(second.alignment[0][3], 0.0)
        state.replace(second, "second.json")
        self.assertIs(state.document, second)
        self.assertEqual(state.path, "second.json")
        self.assertIsNot(state.saved, second)
        self.assertEqual(state.saved.segments, [])


if __name__ == "__main__":
    unittest.main()
