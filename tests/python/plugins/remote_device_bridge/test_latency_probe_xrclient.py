# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""latency_probe_xrclient.fingertips_curl on get_hand_fingertips() arrays shaped like the isaac_teleop
backend's (identity = slot not filled)."""

from __future__ import annotations

import math
import unittest

import numpy as np

import latency_probe_xrclient as probe

TIP = np.array([0.03, -0.02, 0.17])


def hand(wrist_at_origin: bool) -> np.ndarray:
    joints = np.tile(np.eye(4), (25, 1, 1))
    for slot in range(25):
        joints[slot, :3, 3] = [0.0, 0.0, 0.01 * (slot + 1)]
    joints[probe.INDEX_TIP_SLOT, :3, 3] = TIP
    if wrist_at_origin:
        joints[probe.WRIST_SLOT] = np.eye(4)
    return joints


class FingertipsCurlTest(unittest.TestCase):
    def test_wrist_at_origin_still_gives_curl(self):
        # Manus without a headset wrist pose: the wrist slot is exactly eye(4).
        curl, n_set = probe.fingertips_curl(hand(wrist_at_origin=True))
        self.assertEqual(n_set, 24)
        self.assertAlmostEqual(curl, float(np.linalg.norm(TIP)))

    def test_curl_is_relative_to_wrist(self):
        joints = hand(wrist_at_origin=False)
        joints[probe.WRIST_SLOT, :3, 3] = [0.5, 0.0, 0.0]
        curl, n_set = probe.fingertips_curl(joints)
        self.assertEqual(n_set, 25)
        self.assertAlmostEqual(
            curl, float(np.linalg.norm(TIP - np.array([0.5, 0.0, 0.0])))
        )

    def test_unfilled_index_tip_has_no_curl(self):
        joints = hand(wrist_at_origin=False)
        joints[probe.INDEX_TIP_SLOT] = np.eye(4)
        curl, n_set = probe.fingertips_curl(joints)
        self.assertTrue(math.isnan(curl))
        self.assertEqual(n_set, 24)

    def test_no_hand(self):
        curl, n_set = probe.fingertips_curl(None)
        self.assertTrue(math.isnan(curl))
        self.assertEqual(n_set, 0)

    def test_all_identity(self):
        curl, n_set = probe.fingertips_curl(np.tile(np.eye(4), (25, 1, 1)))
        self.assertTrue(math.isnan(curl))
        self.assertEqual(n_set, 0)

    def test_non_finite_tip(self):
        joints = hand(wrist_at_origin=True)
        joints[probe.INDEX_TIP_SLOT, 0, 3] = np.nan
        self.assertTrue(math.isnan(probe.fingertips_curl(joints)[0]))


if __name__ == "__main__":
    unittest.main()
