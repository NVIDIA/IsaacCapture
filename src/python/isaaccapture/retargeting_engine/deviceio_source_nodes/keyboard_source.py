# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Keyboard Source Node - DeviceIO to Retargeting Engine converter.

Converts raw KeyboardOutput flatbuffer data (a set of held evdev key codes, see
linux/input-event-codes.h) to a single standard output: a 256-entry bitmap covering
every standard key. Carries no semantic mapping -- which keys mean what (an axis, a
toggle, a mode switch) is entirely up to the consuming retargeter, which indexes the
bitmap via :class:`EvdevKeyCode`.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, TYPE_CHECKING
from .interface import IDeviceIOSource
from ..interface.retargeter_core_types import RetargeterIO, RetargeterIOType
from ..interface.tensor_group import TensorGroup
from ..tensor_types import NDArrayType, DLDataType
from ..interface.tensor_group_type import OptionalType, TensorGroupType
from .deviceio_tensor_types import DeviceIOKeyboardOutputTracked

if TYPE_CHECKING:
    from isaaccapture.deviceio import ITracker
    from isaaccapture.schema import KeyboardOutput

# Default collection_id matching the keyboard plugin and KeyboardTracker.
DEFAULT_KEYBOARD_COLLECTION_ID = "keyboard"


class EvdevKeyCode(IntEnum):
    """Evdev key codes (linux/input-event-codes.h) for every standard PC keyboard key.

    Values double as indices into the "keyboard_all_keys" bitmap. Covers the full
    standard keyboard (letters, digits, function keys, navigation/editing cluster,
    numpad, punctuation, and modifiers) so any retargeter can bind any of them without
    extending this table; multimedia/vendor-specific keys above this range are not
    covered.
    """

    KEY_ESC = 1
    KEY_1 = 2
    KEY_2 = 3
    KEY_3 = 4
    KEY_4 = 5
    KEY_5 = 6
    KEY_6 = 7
    KEY_7 = 8
    KEY_8 = 9
    KEY_9 = 10
    KEY_0 = 11
    KEY_MINUS = 12
    KEY_EQUAL = 13
    KEY_BACKSPACE = 14
    KEY_TAB = 15
    KEY_Q = 16
    KEY_W = 17
    KEY_E = 18
    KEY_R = 19
    KEY_T = 20
    KEY_Y = 21
    KEY_U = 22
    KEY_I = 23
    KEY_O = 24
    KEY_P = 25
    KEY_LEFTBRACE = 26
    KEY_RIGHTBRACE = 27
    KEY_ENTER = 28
    KEY_LEFTCTRL = 29
    KEY_A = 30
    KEY_S = 31
    KEY_D = 32
    KEY_F = 33
    KEY_G = 34
    KEY_H = 35
    KEY_J = 36
    KEY_K = 37
    KEY_L = 38
    KEY_SEMICOLON = 39
    KEY_APOSTROPHE = 40
    KEY_GRAVE = 41
    KEY_LEFTSHIFT = 42
    KEY_BACKSLASH = 43
    KEY_Z = 44
    KEY_X = 45
    KEY_C = 46
    KEY_V = 47
    KEY_B = 48
    KEY_N = 49
    KEY_M = 50
    KEY_COMMA = 51
    KEY_DOT = 52
    KEY_SLASH = 53
    KEY_RIGHTSHIFT = 54
    KEY_KPASTERISK = 55
    KEY_LEFTALT = 56
    KEY_SPACE = 57
    KEY_CAPSLOCK = 58
    KEY_F1 = 59
    KEY_F2 = 60
    KEY_F3 = 61
    KEY_F4 = 62
    KEY_F5 = 63
    KEY_F6 = 64
    KEY_F7 = 65
    KEY_F8 = 66
    KEY_F9 = 67
    KEY_F10 = 68
    KEY_NUMLOCK = 69
    KEY_SCROLLLOCK = 70
    KEY_KP7 = 71
    KEY_KP8 = 72
    KEY_KP9 = 73
    KEY_KPMINUS = 74
    KEY_KP4 = 75
    KEY_KP5 = 76
    KEY_KP6 = 77
    KEY_KPPLUS = 78
    KEY_KP1 = 79
    KEY_KP2 = 80
    KEY_KP3 = 81
    KEY_KP0 = 82
    KEY_KPDOT = 83
    KEY_102ND = 86
    KEY_F11 = 87
    KEY_F12 = 88
    KEY_KPENTER = 96
    KEY_RIGHTCTRL = 97
    KEY_KPSLASH = 98
    KEY_SYSRQ = 99
    KEY_RIGHTALT = 100
    KEY_HOME = 102
    KEY_UP = 103
    KEY_PAGEUP = 104
    KEY_LEFT = 105
    KEY_RIGHT = 106
    KEY_END = 107
    KEY_DOWN = 108
    KEY_PAGEDOWN = 109
    KEY_INSERT = 110
    KEY_DELETE = 111
    KEY_KPEQUAL = 117
    KEY_PAUSE = 119
    KEY_LEFTMETA = 125
    KEY_RIGHTMETA = 126
    KEY_COMPOSE = 127


# Every standard PC keyboard key (letters, digits, function keys, navigation,
# modifiers, numpad, punctuation) fits under evdev code 255; anything above that
# is an exotic/vendor key not tracked here.
ALL_KEYS_BITMAP_SIZE = 256


def KeyboardAllKeysType() -> TensorGroupType:
    """Type for the "keyboard_all_keys" output: a 256-entry uint8 bitmap indexed by evdev key code."""
    return TensorGroupType(
        "keyboard_all_keys",
        [
            NDArrayType(
                "bitmap",
                shape=(ALL_KEYS_BITMAP_SIZE,),
                dtype=DLDataType.UINT,
                dtype_bits=8,
            )
        ],
    )


class KeyboardSource(IDeviceIOSource):
    """
    Stateless converter: DeviceIO KeyboardOutput → keyboard_all_keys bitmap tensor.

    Inputs:
        - "deviceio_keyboard": Raw KeyboardOutput flatbuffer from KeyboardTracker

    Outputs (Optional — absent when the keyboard plugin has not yet streamed):
        - "keyboard_all_keys": OptionalTensorGroup, a 256-entry uint8 bitmap
          indexed by evdev key code (1 = held, 0 = released; see
          :class:`EvdevKeyCode`) covering every standard key.

    Usage:
        # In TeleopSession, the keyboard tracker is discovered from the pipeline;
        # data is polled via poll_tracker. Or manually:
        tracked = keyboard_tracker.get_keyboard_data(session)
        result = keyboard_source_node({
            "deviceio_keyboard": TensorGroup(DeviceIOKeyboardOutputTracked(), [tracked])
        })
    """

    def __init__(
        self, name: str, collection_id: str = DEFAULT_KEYBOARD_COLLECTION_ID
    ) -> None:
        """Initialize stateless keyboard source node.

        Creates a KeyboardTracker instance for TeleopSession to discover and use.

        Args:
            name: Unique name for this source node
            collection_id: Tensor collection ID for keyboard data (must match the keyboard plugin).
        """
        import isaaccapture.deviceio as deviceio

        self._keyboard_tracker = deviceio.KeyboardTracker(collection_id)
        self._collection_id = collection_id
        super().__init__(name)

    def get_tracker(self) -> "ITracker":
        """Get the KeyboardTracker instance.

        Returns:
            The KeyboardTracker instance for TeleopSession to initialize
        """
        return self._keyboard_tracker

    def poll_tracker(self, deviceio_session: Any) -> RetargeterIO:
        """Poll the keyboard tracker and return input data.

        Args:
            deviceio_session: The active DeviceIO session.

        Returns:
            Dict with "deviceio_keyboard" TensorGroup containing KeyboardOutput | None.
        """
        keys = self._keyboard_tracker.get_keyboard_data(deviceio_session)
        tg = TensorGroup(DeviceIOKeyboardOutputTracked())
        tg[0] = keys
        return {"deviceio_keyboard": tg}

    def input_spec(self) -> RetargeterIOType:
        """Declare DeviceIO keyboard input."""
        return {
            "deviceio_keyboard": DeviceIOKeyboardOutputTracked(),
        }

    def output_spec(self) -> RetargeterIOType:
        """Declare standard keyboard input output (Optional — may be absent)."""
        return {
            "keyboard_all_keys": OptionalType(KeyboardAllKeysType()),
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        """
        Convert DeviceIO KeyboardOutput to the standard keyboard_all_keys bitmap.

        Calls ``set_none()`` on the output when the keyboard plugin has not yet streamed.

        Args:
            inputs: Dict with "deviceio_keyboard" containing KeyboardOutput | None
            outputs: Dict with "keyboard_all_keys" OptionalTensorGroup
            context: Shared ComputeContext for the current step (carries GraphTime).
        """
        import numpy as np

        keys: KeyboardOutput | None = inputs["deviceio_keyboard"][0]

        all_keys_out = outputs["keyboard_all_keys"]
        if keys is None:
            all_keys_out.set_none()
            return

        bitmap = np.zeros(ALL_KEYS_BITMAP_SIZE, dtype=np.uint8)
        for code in keys.pressed_keys:
            if code < ALL_KEYS_BITMAP_SIZE:
                bitmap[code] = 1
        all_keys_out[0] = bitmap
