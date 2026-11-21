# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock, patch

from tactile_metadata_editor.sdk import Receiver


class ReceiverTests(unittest.TestCase):
    def test_receiver_is_inactive_until_the_first_frame(self):
        context = Mock()
        context.Pipe.return_value = Mock(), Mock()
        with (
            patch(
                "tactile_metadata_editor.sdk.multiprocessing.get_context",
                return_value=context,
            ),
            patch("tactile_metadata_editor.sdk.threading.Thread"),
        ):
            receiver = Receiver({})

        self.assertFalse(receiver.active)
        self.assertEqual(receiver.status, "Connecting")

        def receive(_):
            if not receiver.latest:
                return b'{"frame": {"hand": null}}'
            self.assertTrue(receiver.active)
            self.assertEqual(receiver.status, "Live")
            raise EOFError

        receiver.connection.recv_bytes.side_effect = receive
        receiver._receive()
        self.assertEqual(receiver.latest, {"hand": None})
        self.assertFalse(receiver.active)

    def test_pipe_close_preserves_child_errors(self):
        for messages, expected in (
            (
                [b'{"status": "ImportError: SDK unavailable"}'],
                "ImportError: SDK unavailable",
            ),
            ([b'{"frame": {}}'], "Disconnected"),
            ([], "Disconnected"),
        ):
            with self.subTest(messages=messages):
                receiver = Receiver.__new__(Receiver)
                receiver.status, receiver.active = "Connecting", True
                receiver.connection = Mock()
                receiver.connection.recv_bytes.side_effect = [*messages, EOFError()]

                receiver._receive()

                self.assertEqual(receiver.status, expected)
                self.assertFalse(receiver.active)


if __name__ == "__main__":
    unittest.main()
