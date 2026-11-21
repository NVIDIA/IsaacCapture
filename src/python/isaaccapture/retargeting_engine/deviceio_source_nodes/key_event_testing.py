# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scriptable ``KeyEventSource`` for testing keyboard pipelines and provider integrations.

``FakeKeyEventSource`` behaves like a well-behaved host surface: it follows every rule of the
``KeyEventSource`` contract, so tests can drive a ``KeyboardSource`` through focus changes,
UI keyboard capture and press-only taps without a window. Host projects can compare their own
provider's behavior against it.
"""

from __future__ import annotations

from .keyboard_source import (
    FocusLostCallback,
    KeyCallback,
    KeyEventHandle,
    KeyListeners,
    _CloseOnce,
)


class FakeKeyEventSource:
    """A focused input surface driven by test code.

    Starts focused, with its UI not capturing the keyboard. Key reports are dropped while the
    surface is unfocused or its UI owns the keyboard, as a real host would drop them.
    """

    def __init__(self) -> None:
        self._listeners = KeyListeners()
        self.focused = True
        self.ui_capturing = False
        self.capture_count = 0
        self.capture_history: list[bool] = []

    # KeyEventSource ----------------------------------------------------------------
    def add_key_listener(
        self, on_key: KeyCallback, on_focus_lost: FocusLostCallback
    ) -> KeyEventHandle:
        return self._listeners.add(on_key, on_focus_lost)

    def capture_keyboard(self) -> KeyEventHandle:
        self.capture_count += 1
        self.capture_history.append(True)

        def release() -> None:
            self.capture_count -= 1
            self.capture_history.append(False)

        return _CloseOnce(release)

    # Test controls -------------------------------------------------------------------
    @property
    def captured(self) -> bool:
        """Whether any subscriber holds a keyboard capture."""
        return self.capture_count > 0

    @property
    def listener_count(self) -> int:
        return len(self._listeners)

    def press(self, code: str | int) -> None:
        if self._accepting:
            self._listeners.key(code, True)

    def release(self, code: str | int) -> None:
        if self._accepting:
            self._listeners.key(code, False)

    def tap(self, code: str | int) -> None:
        """A press-only surface's keystroke: press immediately followed by release."""
        if self._accepting:
            self._listeners.tap(code)

    def blur(self) -> None:
        """The surface loses focus: every held key must be released."""
        self.focused = False
        self._listeners.focus_lost()

    def focus(self) -> None:
        self.focused = True

    def begin_text_input(self) -> None:
        """The host UI takes the keyboard (a text field gains focus): same as focus loss."""
        self.ui_capturing = True
        self._listeners.focus_lost()

    def end_text_input(self) -> None:
        self.ui_capturing = False

    @property
    def _accepting(self) -> bool:
        return self.focused and not self.ui_capturing
