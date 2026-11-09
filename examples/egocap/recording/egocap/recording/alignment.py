# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Time alignment of a recorded episode, producing the aligned-dataset arrays downstream consumers read.

The recorded sources are resampled onto one camera's frame times (the index source). The rules are fixed by the
aligned dataset format and are implemented exactly as stated, because consumers read the result by array name and
expect these values:

* the valid range runs from the latest first time to the earliest last time over all sources; an index frame is
  kept when it is in that range and on the downsampling stride (``valid_indices``, ``valid_timestamps``);
* a camera other than the index source contributes the position of its frame nearest each kept time
  (``aligned_frame_indices``), found by one forward walk, so the earlier frame wins a tie and a frame may serve
  two kept times;
* the index source's own fields are only filtered by ``valid_indices``; every other source's fields are
  interpolated at the kept times by their declared rule, written as ``<field>_interp``: ``pose`` (4x4
  transforms: linear position, slerp rotation), ``linear`` (element-wise, also used for a 4x4 field with extra
  leading dimensions) or ``nearest`` (the value at the nearest sample); ``raw`` fields are not resampled.

There is no file access here. The times passed in are the ones to align on: a source's recorded capture time, or
its receive time minus a configured latency, chosen by the caller and never adjusted here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import interpolate
from scipy.spatial import transform as st

__all__ = [
    "AlignedEpisode",
    "AlignedSource",
    "AlignmentError",
    "FRAME_FIELD_RULES",
    "FieldRule",
    "InsufficientSamplesError",
    "align_episode",
    "find_nearest_timestamp_indices",
    "interpolate_field",
    "check_continuity",
    "qualified_sources",
    "source_rules",
]

#: How one field is resampled onto the kept times.
FieldRule = Literal["pose", "linear", "nearest", "raw"]


class AlignmentError(ValueError):
    """The episode cannot be aligned: bad input, or no frame of the index source in the common range."""


class InsufficientSamplesError(AlignmentError):
    """Fewer than two samples with strictly increasing times remain, so nothing can be interpolated."""


def qualified_sources(timing: Mapping[str, object] | None, sources: Sequence[str]) -> dict[str, bool]:
    """Which sources' recorded times are qualified event times, from an episode's ``timing`` attribute.

    ``timing["event_time"][source]["qualified"]`` is the recorded statement, per source, of whether its saved time
    is a qualified event time (a sensor's capture event, or a device measurement time). Only a literal True counts;
    a source the attribute does not list, an entry without it, or no attribute at all is False. A clock mapping
    says which clock a time is on, not what it measures, so it is never taken as qualification. A caller that
    aligns on ``receive_time - latency`` instead states that choice itself; this says nothing about it.
    """
    event_time = (timing or {}).get("event_time")
    entries = event_time if isinstance(event_time, Mapping) else {}
    # Fail closed: anything but an explicit True is not qualified.
    return {
        name: isinstance(entries.get(name), Mapping) and entries[name].get("qualified") is True for name in sources
    }


def _complete(entry: Mapping[str, object], rows: int | None) -> bool:
    """Whether a source's evidence has its initial identity and a well-formed list of break rows."""
    identity, breaks = entry.get("identity"), entry.get("breaks")
    if not isinstance(identity, (str, int)) or isinstance(identity, bool) or not isinstance(breaks, list):
        return False
    # A break is a row index: a non-negative integer, and a row the source actually has.
    return all(
        isinstance(b, int) and not isinstance(b, bool) and b >= 0 and (rows is None or b < rows) for b in breaks
    )


#: Continuity scopes that vouch for a source being one unbroken stretch. A clock mapping's epoch covers the
#: publisher's times; ``host_clock`` is a source stamped on the recording host's own clock, covering its host
#: poll and connection but not the device behind it. A transport session alone is not accepted.
ACCEPTED_CONTINUITY_SCOPES = ("clock_mapping", "host_clock")


def check_continuity(
    continuity: Mapping[str, Mapping[str, object]] | None,
    sources: Sequence[str],
    rows: Mapping[str, int] | None = None,
) -> list[str]:
    """Why the recorded continuity evidence does not allow aligning ``sources``; empty means it does.

    ``continuity`` is the per-source evidence stored with the episode (None when the episode recorded none). A
    source is refused when there is no evidence for it, the evidence is incomplete, its scope is not one that
    vouches for the source, or it records a break: interpolating across a restart would produce values the
    sensor never reported. Nothing here defaults to clean: the initial identity and an explicit list of break rows
    must both be present and well formed, and with ``rows`` (the written row count per source) every break must be
    a row of the source.
    """
    if continuity is None:
        return ["no continuity evidence was recorded for this episode"]
    problems: list[str] = []
    for name in sources:
        entry = continuity.get(name)
        if not isinstance(entry, Mapping):
            problems.append(f"{name}: no continuity evidence")
        elif entry.get("known") is not True:
            problems.append(f"{name}: continuity is unknown")
        elif entry.get("scope") not in ACCEPTED_CONTINUITY_SCOPES:
            problems.append(
                f"{name}: its continuity evidence ({entry.get('scope')!r}) does not cover the source itself"
            )
        elif not _complete(entry, None if rows is None else rows.get(name)):
            problems.append(f"{name}: continuity evidence is incomplete or malformed")
        elif entry["breaks"]:
            problems.append(f"{name}: has a restart inside the episode (rows {list(entry['breaks'])})")
    return problems


def find_nearest_timestamp_indices(source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Position in ``source_times`` of the sample nearest each target, by one forward walk.

    Both arrays are sorted. The walk only moves forward, and advances only when the next sample is strictly closer,
    so a tie keeps the earlier sample and one sample can be nearest to several targets.
    """
    m, n = len(source_times), len(target_times)
    indices = np.empty(n, dtype=int)
    j = 0
    for i in range(n):
        # Move on only for a strictly closer sample; equal distance stays on the earlier one.
        while j < m - 1 and abs(source_times[j + 1] - target_times[i]) < abs(source_times[j] - target_times[i]):
            j += 1
        indices[i] = j
    return indices


def _clean(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop each sample whose time is not later than the sample before it, as recorded."""
    drop = [i for i in range(1, len(times)) if times[i] <= times[i - 1]]
    if drop:
        times, values = np.delete(times, drop), np.delete(values, drop, axis=0)
    # An interpolator needs two distinct times to span anything.
    if len(times) < 2:
        raise InsufficientSamplesError(
            f"interpolation needs at least 2 samples with strictly increasing times; {len(times)} remain"
        )
    return times, values


def _linear(times: np.ndarray, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Element-wise linear interpolation along the first axis; targets outside the sample range are an error."""
    times, values = _clean(times, values)
    return interpolate.interp1d(times, values, kind="linear", axis=0)(targets)


def _pose(times: np.ndarray, values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Rigid transforms ``(N, 4, 4)``: linear position and slerp rotation, with a bottom row of [0, 0, 0, 1]."""
    times, values = _clean(times, values)
    positions = interpolate.interp1d(times, values[:, :3, 3], kind="linear", axis=0)(targets)
    rotations = st.Slerp(times, st.Rotation.from_matrix(values[:, :3, :3]))(targets).as_matrix()
    out = np.zeros((len(targets), 4, 4))
    out[:, :3, :3], out[:, :3, 3], out[:, 3, 3] = rotations, positions, 1.0
    return out


def interpolate_field(
    rule: FieldRule, times: np.ndarray, values: np.ndarray, targets: np.ndarray
) -> np.ndarray | None:
    """One field resampled at ``targets``; None for a ``raw`` field, which is not resampled.

    A ``pose`` field that is not a plain ``(N, 4, 4)`` array is interpolated linearly instead, element by element.
    Targets outside the range of ``times`` are an error (they cannot occur for the kept times of an alignment).
    """
    times, values = np.asarray(times, dtype=np.float64), np.asarray(values)
    if rule == "raw":
        return None
    if rule == "nearest":
        return values[find_nearest_timestamp_indices(times, targets)]
    # A pose rule only applies to a plain stack of transforms; any other shape falls back to element-wise.
    if rule == "pose" and values.ndim == 3 and values.shape[1:] == (4, 4):
        return _pose(times, values, targets)
    if rule in ("pose", "linear"):
        return _linear(times, values, targets)
    raise AlignmentError(f"unknown field rule {rule!r}")


@dataclass(frozen=True)
class AlignedSource:
    """What the alignment adds for one source."""

    #: For a camera: positions of the frames matched to the kept times (the index source's own kept positions).
    aligned_frame_indices: np.ndarray | None = None
    #: ``<field>`` -> values at the kept times; written as ``<field>_interp``.
    interpolated: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class AlignedEpisode:
    """The result of aligning one episode."""

    index_source: str
    #: Boolean mask over the index source's frames: in the valid range and on the downsampling stride.
    valid_indices: np.ndarray
    #: The index source's times at the kept frames.
    valid_timestamps: np.ndarray
    #: The valid range that was used, as (first, last).
    valid_range: tuple[float, float]
    sources: dict[str, AlignedSource]


#: How each field of a frame type is resampled when an episode is time aligned. A session's config.yaml names the
#: frame type of each source. Head and controller poses are rigid transforms (slerp); the glove's joint transforms
#: are interpolated element-wise, so their shape never selects slerp; triggers and sensor values are continuous;
#: a count is the nearest sample's; buttons and flags are not resampled, so they are absent here (raw).
FRAME_FIELD_RULES: dict[str, dict[str, FieldRule]] = {
    "RGBCameraFrame": {"pose": "pose"},
    "NumericFrame": {},
    "PicoTrackingFrame": {
        "head_pose": "pose",
        "left_pose": "pose",
        "right_pose": "pose",
        "left_trigger": "linear",
        "right_trigger": "linear",
    },
    "ManusFrame": {
        **{f"{side}_hand_pose": "linear" for side in ("left", "right")},
        **{
            f"{side}_sensor_{kind}": "linear"
            for side in ("left", "right")
            for kind in ("position", "orientation", "rotation")
        },
        **{f"{side}_sensor_count": "nearest" for side in ("left", "right")},
    },
}


def source_rules(config: Mapping[str, object]) -> dict[str, dict[str, FieldRule]]:
    """Each declared source's field rules, from the source declarations in a session's ``config.yaml``.

    A source declares its ``frame_type``; one that declares none takes the session's default for its kind. A frame
    type with no known rules is an error, since silently resampling nothing would write an aligned episode that
    looks complete and is not.
    """
    recording = config.get("recording")
    defaults = {
        "video_sources": (recording or {}).get("video_frame_type", "RGBCameraFrame")
        if isinstance(recording, Mapping)
        else "RGBCameraFrame",
        "numeric_sources": (recording or {}).get("numeric_frame_type", "NumericFrame")
        if isinstance(recording, Mapping)
        else "NumericFrame",
    }
    rules: dict[str, dict[str, FieldRule]] = {}
    for section, default in defaults.items():
        entries = config.get(section)
        # An absent or malformed section declares no sources.
        for entry in entries if isinstance(entries, (list, tuple)) else []:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("name"), str):
                raise AlignmentError(f"{section}: every source needs a name")
            frame_type = entry.get("frame_type", default)
            if frame_type not in FRAME_FIELD_RULES:
                raise AlignmentError(f"{entry['name']}: no alignment rules for frame type {frame_type!r}")
            rules[entry["name"]] = dict(FRAME_FIELD_RULES[frame_type])
    return rules


def _check_inputs(index_source: str, times: Mapping[str, np.ndarray], downsample: int) -> None:
    """Refuse input that cannot be aligned: a missing index source, a bad stride, or unusable times."""
    if index_source not in times:
        raise AlignmentError(f"the index source {index_source!r} has no times")
    if downsample < 1:
        raise AlignmentError("downsample must be at least 1")
    for name, source_times in times.items():
        arr = np.asarray(source_times)
        # The forward walk and the range logic assume sorted, finite times.
        if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
            raise AlignmentError(f"{name}: needs a non-empty list of finite times")
        if np.any(np.diff(arr) < 0):
            raise AlignmentError(f"{name}: times go backwards")


def align_episode(
    index_source: str,
    times: Mapping[str, np.ndarray],
    cameras: Sequence[str],
    fields: Mapping[str, Mapping[str, np.ndarray]],
    rules: Mapping[str, Mapping[str, FieldRule]],
    *,
    downsample: int = 1,
    epochs: Mapping[str, np.ndarray] | None = None,
) -> AlignedEpisode:
    """Align every source in ``times`` onto ``index_source``.

    ``times`` maps each source to the times to align on (sorted, finite); ``cameras`` names the sources that are
    cameras; ``fields[source][name]`` are a source's arrays and ``rules[source][name]`` how each is resampled (a
    field with no rule is not resampled). Raises ``AlignmentError`` if the index source has no frame in the valid
    range, so an alignment is never an empty success. ``epochs`` optionally labels each sample of a source with the
    unbroken stretch it belongs to (the label changes at a restart, a reconnect or an invalid interval); a source
    with more than one label is refused, since interpolating across the break would produce values the sensor
    never reported.
    """
    _check_inputs(index_source, times, downsample)
    for name, labels in (epochs or {}).items():
        # The aligned numbers follow the rules above unchanged; a break is refused, not handled differently.
        if name not in times or len(labels) != len(times[name]):
            raise AlignmentError(f"{name}: epoch labels must be one per sample of a source being aligned")
        if len(np.unique(labels)) > 1:
            raise AlignmentError(f"{name}: has a restart or invalid interval inside the episode")
    index_times = np.asarray(times[index_source], dtype=np.float64)

    # The valid range: every source has data on both sides of a kept time.
    lo = max(float(np.min(t)) for t in times.values())
    hi = min(float(np.max(t)) for t in times.values())
    in_range = (index_times >= lo) & (index_times <= hi)
    # Downsampling keeps every n-th index frame, counted from the first frame (not from the first kept one).
    stride = np.zeros_like(in_range, dtype=bool)
    stride[::downsample] = True
    valid = in_range & stride
    kept = index_times[valid]
    if kept.size == 0:
        raise AlignmentError(f"no frame of {index_source!r} is in the common range [{lo}, {hi}]")

    sources: dict[str, AlignedSource] = {}
    for name, source_times in times.items():
        source_times = np.asarray(source_times, dtype=np.float64)
        interpolated: dict[str, np.ndarray] = {}
        for field_name, rule in rules.get(name, {}).items():
            # A rule for a field that was not recorded, or a raw field, adds nothing.
            if field_name not in fields.get(name, {}) or rule == "raw":
                continue
            values = np.asarray(fields[name][field_name])
            if name == index_source:
                # The index source already sits on the kept times: only filter it.
                interpolated[field_name] = values[valid]
            else:
                interpolated[field_name] = interpolate_field(rule, source_times, values, kept)
        frames = None
        if name in cameras:
            frames = (
                np.flatnonzero(valid)
                if name == index_source
                else find_nearest_timestamp_indices(source_times, kept)
            )
        sources[name] = AlignedSource(frames, interpolated)
    return AlignedEpisode(index_source, valid, kept, (lo, hi), sources)
