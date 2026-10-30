"""A ``TeleopSession``-shaped tracking source that plays back a recorded
episode, for :class:`~capture_server.acquisition.isaac_capture.IsaacCaptureAdapter`.

The adapter composes its session graph through an ``sdk`` module exposing
``HeadSource``/``ControllersSource``/``HandsSource``/``OutputCombiner``/
``PluginConfig``/``TeleopSessionConfig``/``CloudXRLauncher`` and reads
status through the native status dataclasses. Those SDK-shaped
reproductions live in :mod:`capture_server.mock.mock_isaac_capture` and
are re-exported here unchanged, so this module can be passed as the
adapter's ``sdk`` exactly like the mock module is; only
:class:`TeleopSession` is replay's own. None of the mock's motion model,
scenarios or faults are involved.

``step()`` is synchronous and unpaced like the real API: the adapter's
own loop decides when to call it, and each call returns the recorded rows
current at the session's shared :class:`~capture_server.replay.playback.PlaybackClock`
position.
"""

from __future__ import annotations

from typing import Any

from capture_server.acquisition.clock import Clock, MonotonicClock
from capture_server.acquisition.tensor_indices import ControllerInputIndex, HandInputIndex, HeadInputIndex
from capture_server.mock.mock_isaac_capture import (
    MANUS_PLUGIN_NAME,
    CloudXRLauncher,
    ControllersSource,
    DeviceState,
    DeviceStatus,
    HandsSource,
    HeadSource,
    NoopContext,
    OptionalTensorGroup,
    OutputCombiner,
    PluginConfig,
    ProviderState,
    ProviderStatus,
    ProviderType,
    StatusReason,
    StatusSnapshot,
    TeleopSessionConfig,
)

from . import tracking_map
from .episode import Episode
from .playback import PlaybackClock

__all__ = [
    "CloudXRLauncher",
    "ControllersSource",
    "HandsSource",
    "HeadSource",
    "NoopContext",
    "OutputCombiner",
    "PluginConfig",
    "TeleopSession",
    "TeleopSessionConfig",
]

_REPLAY_PROVIDER_ID = "provider-replay"
_MANUS_PROVIDER_ID = "provider-manus"
# The recorder's face-button names per hand: X/Y are on the left controller, A/B on the right, matching
# the primary/secondary convention the adapter's mock counterpart uses.
_BUTTONS = {"left": ("x_button", "y_button"), "right": ("a_button", "b_button")}


class TeleopSession:
    """Serves recorded tracking rows for the groups the configured graph selects.

    A selected group whose source is not in the episode (or whose first sample lies ahead of the current
    playback position) is returned absent (``is_none``), exactly as the real SDK reports a device with no
    sample -- never as a zeroed, "valid" pose.
    """

    def __init__(
        self,
        config: TeleopSessionConfig,
        *,
        episode: Episode,
        playback: PlaybackClock,
        clock: Clock | None = None,
    ) -> None:
        self.config = config
        self.frame_count = 0
        self._episode = episode
        self._playback = playback
        self._clock = clock or MonotonicClock()
        self._entered = False

    def __enter__(self) -> TeleopSession:
        """Nothing to open: the episode is already loaded; entry only arms ``step()``."""
        self._entered = True
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Disarm ``step()``; there is no native resource to release."""
        self._entered = False
        return None

    def step(
        self,
        *,
        external_inputs: dict[str, Any] | None = None,
        graph_time: Any | None = None,
        execution_events: Any | None = None,
    ) -> dict[str, Any]:
        """Return the configured output groups for the rows current at the shared playback position."""
        if not self._entered:
            raise RuntimeError("TeleopSession.step() called before __enter__")
        self.frame_count += 1
        position = self._playback.position()
        controllers = self._episode.controllers
        hands = self._episode.hands
        # Each table is looked up on the shared timeline independently: the streams have different rates and
        # start offsets, so "the current row" is per table, never a common index.
        controller_row = controllers.latest_row(position) if controllers is not None else -1
        hand_row = hands.latest_row(position) if hands is not None else -1

        result: dict[str, OptionalTensorGroup] = {}
        for leaf_key, (source_node, selector) in self.config.pipeline.outputs.items():
            if isinstance(source_node, HeadSource):
                result[leaf_key] = self._head(controller_row)
            elif isinstance(source_node, ControllersSource):
                side = "left" if selector == ControllersSource.LEFT else "right"
                result[leaf_key] = self._controller(side, controller_row)
            elif isinstance(source_node, HandsSource):
                side = "left" if selector == HandsSource.LEFT else "right"
                result[leaf_key] = self._hand(side, controller_row, hand_row)
            else:
                raise ValueError(f"unsupported source node in pipeline.outputs: {source_node!r}")
        return result

    def _head(self, row: int) -> OptionalTensorGroup:
        """The head group for controller-table *row*, absent before the table's first sample."""
        table = self._episode.controllers
        if table is None or row < 0:
            return OptionalTensorGroup(len(HeadInputIndex))
        return tracking_map.head_group(table.arrays["head_pose"][row])

    def _controller(self, side: str, row: int) -> OptionalTensorGroup:
        """One controller's group for controller-table *row*, absent before the table's first sample."""
        table = self._episode.controllers
        if table is None or row < 0:
            return OptionalTensorGroup(len(ControllerInputIndex))
        primary_name, secondary_name = _BUTTONS[side]
        return tracking_map.controller_group(
            table.arrays[f"{side}_pose"][row],
            trigger=float(table.arrays[f"{side}_trigger"][row]),
            primary=bool(table.arrays[primary_name][row]),
            secondary=bool(table.arrays[secondary_name][row]),
        )

    def _hand(self, side: str, controller_row: int, hand_row: int) -> OptionalTensorGroup:
        """One hand's group for hand-table *hand_row*, anchored at the same side's controller pose.

        The recorded skeleton is wrist-relative (see ``tracking_map``), so the controller worn/held on that
        side is the only recorded pose that can place it; with no controller row yet, the hand is present but
        invalid rather than placed at the origin.
        """
        table = self._episode.hands
        if table is None or hand_row < 0:
            return OptionalTensorGroup(len(HandInputIndex))
        controllers = self._episode.controllers
        anchor = None
        if controllers is not None and controller_row >= 0:
            anchor = controllers.arrays[f"{side}_pose"][controller_row]
        group = tracking_map.hand_group(table.arrays[f"{side}_hand_pose"][hand_row], anchor, side)
        return group if group is not None else OptionalTensorGroup(len(HandInputIndex))

    def get_status(self) -> StatusSnapshot:
        """Native-shaped inventory describing the recording: one replay provider standing in for the OpenXR
        runtime, with a device per recorded stream the graph selected. A selected stream the episode lacks is
        listed UNAVAILABLE with the reason, so the client sees why no samples arrive."""
        episode = self._episode
        providers = [
            ProviderStatus(
                id=_REPLAY_PROVIDER_ID, type=ProviderType.OPENXR_RUNTIME, name="Recorded episode replay",
                status=ProviderState.AVAILABLE, reason=StatusReason.NONE, error="",
            ),
        ]  # fmt: skip
        devices: list[DeviceStatus] = []

        def device(id_: str, path: str, type_: str, description: str, provider: str, recorded: bool) -> None:
            """Append one device, CONNECTED when its stream was recorded, UNAVAILABLE otherwise."""
            devices.append(
                DeviceStatus(
                    id=id_, path=path, type=type_, description=description, provider_id=provider,
                    status=DeviceState.CONNECTED if recorded else DeviceState.UNAVAILABLE,
                    reason=StatusReason.NONE if recorded else StatusReason.NO_CURRENT_DATA,
                    error="" if recorded else "not in the recorded episode",
                )
            )  # fmt: skip

        selected_sources = {type(node) for node, _ in self.config.pipeline.outputs.values()}
        selected_controllers = {
            selector
            for node, selector in self.config.pipeline.outputs.values()
            if isinstance(node, ControllersSource)
        }
        selected_hands = {
            selector for node, selector in self.config.pipeline.outputs.values() if isinstance(node, HandsSource)
        }
        has_controllers = episode.controllers is not None
        if HeadSource in selected_sources:
            device(
                "device-headset",
                "/head/headset",
                "headset",
                "Recorded headset",
                _REPLAY_PROVIDER_ID,
                has_controllers,
            )
        for side in ("left", "right"):
            if side in selected_controllers:
                device(
                    f"device-controller-{side}", f"/controller/{side}", "controller",
                    f"Recorded {side} controller", _REPLAY_PROVIDER_ID, has_controllers,
                )  # fmt: skip
        # Hands come from the recorded gloves. When the session requested the Manus plugin they are listed
        # under a provider named after the Manus manifest (the adapter maps that name to the plugin type);
        # otherwise under
        # the replay provider, like runtime hand tracking would be. A hand's inventory entry says whether the
        # episode recorded it, not whether the current row is tracked -- that is per-sample validity.
        if selected_hands:
            manus_requested = any(
                plugin.plugin_name == MANUS_PLUGIN_NAME and plugin.enabled for plugin in self.config.plugins
            )
            hand_provider = _MANUS_PROVIDER_ID if manus_requested else _REPLAY_PROVIDER_ID
            if manus_requested:
                providers.append(
                    ProviderStatus(
                        id=_MANUS_PROVIDER_ID, type=ProviderType.PLUGIN, name=MANUS_PLUGIN_NAME,
                        status=ProviderState.AVAILABLE, reason=StatusReason.NONE, error="",
                    )
                )  # fmt: skip
            for side in ("left", "right"):
                if side in selected_hands:
                    device(
                        f"device-hand-{side}", f"/hand/{side}", "hand", f"Recorded {side} glove",
                        hand_provider, episode.hands is not None,
                    )  # fmt: skip
        return StatusSnapshot(
            updated_at_ns=int(self._clock.monotonic() * 1e9), providers=tuple(providers), devices=tuple(devices)
        )
