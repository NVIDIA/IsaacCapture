<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Timestamp alignment: what is implemented, what the sources really stamp, what is open

This tracks the repository-root `TimestampAlignment.md` plan against the code. Nothing here establishes
synchronization: no required accuracy budget has been set, and no real-source alignment has been measured.

## Implemented

- **PC clock-map discontinuity check** (`backend/egocap_web/livekit_recording.py`, `UnixClockMap`): keeps the
  bracket uncertainty of its realtime/monotonic pair and flags an offset change between two measurements larger than
  the greater of 5 ms and four times that uncertainty. The recording feed then fails the take. This is a heuristic,
  not proof of a step or an accuracy bound; frames mapped between a step and its detection (up to one refresh, about a
  second) are already recorded, and smaller offset changes are not caught.
- LiveKit frames keep the publisher's timestamp, frame id and raw capture counter separately; only a stream declaring a
  sensor-capture basis is recorded; the native sender declares processing time and carries the counter unscaled.

## What each source actually stamps (source audit, not hardware-verified)

- **Controllers** (`src/core/live_trackers/cpp/live_controller_tracker_impl.cpp`): the update tick time
  (`last_update_time_`, a host monotonic reading) is converted to an `XrTime`, poses are located at that time with
  `xrLocateSpace`, and the `DeviceDataTimestamp` carries that update time as both availability and sample time with
  the `XrTime` as the raw value. It is the pose evaluation time of a runtime-predicted pose, not a physical
  measurement time.
- **Manus gloves** (`src/plugins/manus/core/manus_hand_tracking_plugin.cpp`): the SDK stream's `publishTime` is read
  into the skeleton info but is not used for the pushed sample. The sample is pushed with the host monotonic time at
  push for both sample and availability, and the hand poses are located at the current `XrTime`. Skeleton freshness
  uses host `steady_clock` at the callback (receipt). So glove sample time today is host processing time; the SDK
  time exists and could be mapped if the SDK documents its clock and a host correlation.
- **Camera (LiveKit)**: processing time on the Orin (conversion), not capture; see the plan for the SIPL/Argus
  counters, whose units and host correlation are not verified.

None of these is a qualified measurement time, so none can support a synchronized-recording claim yet.

## Open

- **Encoder association.** The Orin encoder does not return the submitted timestamp, so units are paired by
  submission order, which shifts after an encoder-internal drop. Header search found
  `V4L2_CID_MPEG_VIDEOENC_COPY_TIMESTAMP` (undocumented, unused elsewhere); whether it works on the driver is being
  probed on the Orin. Until a verified mechanism exists, the honest options are to stop or flag the stream on loss.
- **Persisting mapping epochs and quality with a take.** There is no per-take metadata slot: `config.yaml` is
  frozen per session, and `EpisodeResult.issues` carries failures only. Proposed minimum, needing an owner's
  decision: write the clock-map evidence (map epoch, offset, bracket uncertainty, step flag, timestamp basis per
  source) once at stop into the episode's existing record, rather than a new file format.
- **Orin-to-PC clock bridge**, sensor-time-to-host mapping, glove SDK time correlation, controller measurement
  semantics, and the accuracy budget.
