<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Timestamp alignment: what is implemented, what the sources really stamp, what is open

This tracks the repository-root `TimestampAlignment.md` plan against the code. Nothing here establishes
synchronization: no required accuracy budget has been set, and no real-source alignment has been measured.

## Implemented

- **One clock path.** A four-timestamp exchange between the PC and the publisher (and the browser) maps the
  publisher's monotonic clock onto the PC's; recording uses only that mapping, ends a take on a missing, stale,
  jumped or over-budget mapping (`EGOCAP_CLOCK_UNCERTAINTY_BUDGET_MS`, default 5 ms, a policy choice), and saves the
  mapping evidence in the episode's `timing` attributes.
- **Native sender.** Answers the exchange and, with `--timestamp-basis=capture_monotonic_us`, converts the SIPL
  counter or Argus sensor timestamp to `CLOCK_MONOTONIC` through a low-rate anchor, failing the stream on any
  implausible conversion. Written without an Orin: units unverified, not built against the SDKs.
- **Encoder probe.** `SENSING_ENCODER_TIMESTAMP_PROBE=1` logs returned unit stamps beside the last submitted stamp,
  so the first Orin run shows whether the driver returns them. The `COPY_TIMESTAMP` control is not set: its header
  is not present off the Orin.

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
- **Tracking device times.** Tracking samples carry host poll time (`tracking_time: host_poll`). Using the Manus
  SDK `publishTime` or a controller measurement time needs the vendor's documented clock and a schema field for the
  raw value; a `.fbs` change makes existing recordings unreadable (see `src/core/schema/AGENTS.md`), so it is not
  done on a guess.
- **Orin verification** of the counter conversion, the encoder probe and the Argus build, and measured
  sensor-to-tracking alignment.
