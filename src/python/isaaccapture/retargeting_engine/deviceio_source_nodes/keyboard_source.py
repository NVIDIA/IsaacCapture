# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Keyboard Source Node - in-process keyboard device for the retargeting engine.

Keys reach Isaac Teleop from whatever surface the user has focused (a native window, a
browser tab, ...) through :class:`KeyboardProvider` objects on the source's
``KeyboardTracker``. The tracker merges every provider, records to MCAP like any DeviceIO
device, and this node converts each frame to two bitmaps indexed by evdev key code
(``isaaccapture.deviceio_trackers.EvdevKeyCode``), one entry per Linux key code
(``KEYBOARD_KEY_CODE_COUNT``). Carries no semantic mapping: which keys mean what is up to the
consuming retargeter.

Hosts either drive a provider directly (``create_provider``) or hand the source any object
implementing :class:`KeyEventSource` (``attach``); :class:`KeyListeners` does the listener
bookkeeping for such an object.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Protocol, TYPE_CHECKING
from .interface import IDeviceIOSource
from ..interface.retargeter_core_types import RetargeterIO, RetargeterIOType
from ..interface.tensor_group import TensorGroup
from ..tensor_types import NDArrayType, DLDataType
from ..interface.tensor_group_type import OptionalType, TensorGroupType
from .deviceio_tensor_types import DeviceIOKeyboardOutputTracked
from isaaccapture.deviceio_trackers import KEYBOARD_KEY_CODE_COUNT

if TYPE_CHECKING:
    from isaaccapture.deviceio_trackers import ITracker, KeyboardProvider
    from isaaccapture.schema import KeyboardOutput


class KeyEventHandle(Protocol):
    """A subscription or keyboard capture returned by a :class:`KeyEventSource`.

    ``close()`` ends it; closing again does nothing.
    """

    def close(self) -> None:
        """End the subscription or capture."""


class KeyEventSource(Protocol):
    """A focused input surface that can feed a :class:`KeyboardSource`.

    Implemented by the host (a window, a browser viewer, ...) without importing Isaac Teleop
    types. Requirements:

    - Call ``on_key(code, pressed)`` for press and release only while the surface has focus;
      ``code`` is a W3C ``KeyboardEvent.code`` string ("KeyW", "ArrowUp", ...) or an evdev int.
      Autorepeat may be forwarded; repeated presses of a held key are ignored.
    - Call ``on_focus_lost()`` on blur, close or disconnect so no key can stay stuck.
    - When the host can tell that its own UI owns the keyboard, that UI wins: don't forward keys
      typed into its text fields, and call ``on_focus_lost()`` the moment its UI takes the
      keyboard (a text field gains focus), exactly as on blur -- otherwise a key held at that
      moment never sees its release. A host that cannot tell reports those keys too, and should
      document it.
    - ``capture_keyboard`` yields only host shortcuts that conflict with teleop (camera keys,
      hotkeys); it never blocks the host's text input. A viewer that moves its camera with
      WASD/QE or the arrow keys needs it, since those are the teleop motion keys; a surface
      with no bindings of its own returns a handle whose ``close()`` does nothing. Each call returns its own handle, and the
      shortcuts stay yielded while any handle is open. A call that raises must leave the host
      unchanged.
    - A surface that cannot report releases (press-only hotkeys, a plain terminal) reports each
      keystroke as ``on_key(code, True)`` immediately followed by ``on_key(code, False)``: a
      tap that reaches ``keyboard_pressed`` but is never held.
    - Callbacks may arrive on any thread, but one subscription's callbacks must run one at a
      time, in the order the events happened: a key event from before a focus loss completes
      before ``on_focus_lost()`` runs, never after it. A press delivered late stays held, and
      the consumer cannot tell it apart from a real one.
    """

    def add_key_listener(
        self,
        on_key: Callable[[str | int, bool], None],
        on_focus_lost: Callable[[], None],
    ) -> KeyEventHandle:
        """Subscribe; close the returned handle to unsubscribe."""

    def capture_keyboard(self) -> KeyEventHandle:
        """Disable the host's own conflicting key bindings (e.g. camera keys) until the returned handle closes."""


KeyCallback = Callable[[str | int, bool], None]
FocusLostCallback = Callable[[], None]


class _CloseOnce:
    """A :class:`KeyEventHandle` that runs ``on_close`` the first time it is closed."""

    def __init__(self, on_close: Callable[[], None]) -> None:
        self._on_close: Callable[[], None] | None = on_close

    def close(self) -> None:
        on_close, self._on_close = self._on_close, None
        if on_close is not None:
            on_close()


class KeyListeners:
    """Listener bookkeeping for a :class:`KeyEventSource` implementation.

    The surface's ``add_key_listener`` returns :meth:`add`, and the surface calls :meth:`key` and
    :meth:`focus_lost` as it observes events. Listeners may close their handle from a callback.
    """

    def __init__(self) -> None:
        self._listeners: list[tuple[KeyCallback, FocusLostCallback]] = []

    def add(
        self, on_key: KeyCallback, on_focus_lost: FocusLostCallback
    ) -> KeyEventHandle:
        """Subscribe; closing the returned handle unsubscribes."""
        entry = (on_key, on_focus_lost)
        self._listeners.append(entry)
        return _CloseOnce(lambda: self._listeners.remove(entry))

    def key(self, code: str | int, pressed: bool) -> None:
        """Report a press or release to every listener."""
        for on_key, _ in list(self._listeners):
            on_key(code, pressed)

    def tap(self, code: str | int) -> None:
        """Report a keystroke from a surface without release events: a press, then its release."""
        self.key(code, True)
        self.key(code, False)

    def focus_lost(self) -> None:
        """Tell every listener the surface lost the keyboard."""
        for _, on_focus_lost in list(self._listeners):
            on_focus_lost()

    def __len__(self) -> int:
        return len(self._listeners)


# One entry per Linux evdev key code; providers ignore codes outside this range.
KEY_BITMAP_SIZE = KEYBOARD_KEY_CODE_COUNT


def _key_bitmap_type(name: str) -> TensorGroupType:
    return TensorGroupType(
        name,
        [
            NDArrayType(
                "bitmap",
                shape=(KEY_BITMAP_SIZE,),
                dtype=DLDataType.UINT,
                dtype_bits=8,
            )
        ],
    )


def KeyboardHeldType() -> TensorGroupType:
    """Type for "keyboard_held": keys held at the end of the frame, indexed by evdev code."""
    return _key_bitmap_type("keyboard_held")


def KeyboardPressedType() -> TensorGroupType:
    """Type for "keyboard_pressed": keys with at least one press event during the frame.

    Catches taps shorter than a frame and gives toggles an edge without keeping state. A per-frame
    edge: several presses of one key within a frame set the bit once.
    """
    return _key_bitmap_type("keyboard_pressed")


class KeyboardAttachment(_CloseOnce):
    """A surface attached to a :class:`KeyboardSource`, from :meth:`KeyboardSource.attach`.

    ``close()`` detaches it; closing again does nothing. It is also a context manager:
    ``with keyboard.attach(window): ...``.
    """

    def __enter__(self) -> KeyboardAttachment:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class KeyboardSource(IDeviceIOSource):
    """
    In-process keyboard device: KeyboardTracker providers -> key bitmaps.

    Inputs:
        - "deviceio_keyboard": KeyboardOutput from the source's KeyboardTracker

    Outputs (Optional -- absent while no provider is attached):
        - "keyboard_held": uint8 bitmap, 1 = held at the end of the frame
        - "keyboard_pressed": uint8 bitmap, 1 = pressed at least once this frame

    Usage:
        keyboard = KeyboardSource("keyboard")
        with keyboard.attach(window):               # any KeyEventSource
            ...
        # or, driving a provider directly (same rules as a KeyEventSource):
        provider = keyboard.create_provider()
        provider.key_down("KeyW"); provider.key_up("KeyW")
    """

    def __init__(self, name: str) -> None:
        """Initialize the keyboard source and its in-process KeyboardTracker.

        Args:
            name: Unique name for this source node (also its MCAP channel base name).
        """
        from isaaccapture.deviceio_trackers import KeyboardTracker

        self._keyboard_tracker = KeyboardTracker()
        super().__init__(name)

    def get_tracker(self) -> "ITracker":
        """Get the KeyboardTracker instance for TeleopSession to register."""
        return self._keyboard_tracker

    def create_provider(self) -> "KeyboardProvider":
        """Create a provider for one input surface. Close it (or use ``with``) to detach."""
        return self._keyboard_tracker.create_provider()

    def attach(
        self, surface: KeyEventSource, *, capture: bool = True
    ) -> KeyboardAttachment:
        """Feed this source from ``surface`` until the returned attachment is closed.

        Creates a provider, subscribes it to the surface's key and focus-loss callbacks and,
        when ``capture`` is set, asks the surface to yield its own key bindings. The attachment
        detaches on ``close()`` or on leaving a ``with`` block. Detaching
        unsubscribes, releases the provider's keys and restores the surface's bindings. Every
        cleanup step runs even if an earlier one raises, and the provider is always closed;
        failures propagate with earlier ones chained as ``__context__``.
        """
        provider = self.create_provider()

        def on_key(code: str | int, pressed: bool) -> None:
            if pressed:
                provider.key_down(code)
            else:
                provider.key_up(code)

        # ExitStack runs every callback (last registered first) even when one raises, so the
        # provider -- registered first -- is always closed.
        with contextlib.ExitStack() as rollback:
            rollback.callback(provider.close)
            subscription = surface.add_key_listener(on_key, provider.release_all)
            rollback.callback(subscription.close)
            keyboard_capture = surface.capture_keyboard() if capture else None
            rollback.pop_all()

        def detach() -> None:
            with contextlib.ExitStack() as cleanup:
                cleanup.callback(provider.close)
                if keyboard_capture is not None:
                    cleanup.callback(keyboard_capture.close)
                cleanup.callback(subscription.close)

        return KeyboardAttachment(detach)

    def poll_tracker(self, deviceio_session: Any) -> RetargeterIO:
        """Poll the keyboard tracker and return input data.

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
        """Declare the held and pressed bitmaps (Optional -- absent without a provider)."""
        return {
            "keyboard_held": OptionalType(KeyboardHeldType()),
            "keyboard_pressed": OptionalType(KeyboardPressedType()),
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        """Convert KeyboardOutput to the held and pressed bitmaps.

        Calls ``set_none()`` on both outputs when no provider is attached.
        """
        import numpy as np

        from isaaccapture.schema import KeyAction

        keys: KeyboardOutput | None = inputs["deviceio_keyboard"][0]

        held_out = outputs["keyboard_held"]
        pressed_out = outputs["keyboard_pressed"]
        if keys is None:
            held_out.set_none()
            pressed_out.set_none()
            return

        held = np.zeros(KEY_BITMAP_SIZE, dtype=np.uint8)
        for code in keys.held_keys:
            if code < KEY_BITMAP_SIZE:
                held[code] = 1

        pressed = np.zeros(KEY_BITMAP_SIZE, dtype=np.uint8)
        for event in keys.events:
            if event.action == KeyAction.PRESS and event.code < KEY_BITMAP_SIZE:
                pressed[event.code] = 1

        held_out[0] = held
        pressed_out[0] = pressed
