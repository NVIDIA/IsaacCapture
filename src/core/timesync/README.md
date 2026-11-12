<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# timesync

Converts a camera host's wall-clock capture time to the PC's monotonic clock and decides whether
the result may be recorded. The two hosts keep their wall clocks aligned with `chronyd`; this
library reads chrony's own report of how well it is doing and refuses a frame it cannot vouch for.

Target `timesync::timesync` (consumers include `<timesync/...>`). POSIX only, so the Windows source
build skips it. It builds without CUDA, LiveKit or Python, and on its own:

```bash
cmake -S src/core/timesync -B build -DBUILD_TESTING=ON && cmake --build build && ctest --test-dir build
```

## Pieces

| Header | Role |
| --- | --- |
| `chrony_tracking.hpp` | Strict parser for one `chronyc -c tracking` line; `is_synchronised`. |
| `local_clock.hpp` | Bracketed realtime/monotonic sample and a tracker that numbers the intervals between realtime steps. |
| `capture_qualifier.hpp` | `qualify_capture`: PC-monotonic capture time, summed uncertainty, or a typed refusal. |
| `chrony_monitor.hpp`, `command_runner.hpp` | Background `chronyc` poll: no shell, a timeout, an output cap, never in the frame path. |
| `tools/timesync_tool.py` | Sets up the chrony the library reads (below). Python standard library only. |

## Usage

A consumer keeps a `ChronyMonitor` for the PC and one for the camera host, a `LocalClockTracker`
fed by `sample_local_clock`, and calls `qualify_capture` per frame. The caller pins the clock
generation when a take starts and ends the take if a frame reports another. A `ChronyMonitor` can
query a chronyd over its command port (`host` and `port` in `ChronyMonitorConfig`), which is how a
dedicated unprivileged instance next to the system chronyd is read.

## Arrangement

The camera host keeps its own clock and serves it (it may follow UTC or serve its own clock). A second,
unprivileged chronyd on the PC measures it and never adjusts the PC's system clock, so nothing has to reach the
PC (a PC firewall does not matter) and the PC needs no root beyond installing chrony. `qualify_capture` charges
the link between the two.

## Running and deploying it

`tools/timesync_tool.py` (Python standard library only):

```bash
# On the PC: install chrony if it is missing (needs sudo).
python3 src/core/timesync/tools/timesync_tool.py ensure-local

# Install and start chrony on the camera host, let this PC measure it and read its state
# (needs passwordless sudo and the network there; bounded by --timeout, default 90 s).
python3 src/core/timesync/tools/timesync_tool.py configure-remote user@camera-host --host-lan-ip PC_LAN_IP

# On the PC, in the foreground: measure the camera host's clock.
python3 src/core/timesync/tools/timesync_tool.py follow --server CAMERA_HOST_IP

# Read-only: has this PC settled on the camera host? Exits non-zero with the reasons when not.
python3 src/core/timesync/tools/timesync_tool.py status --remote CAMERA_HOST_IP --wait 120
```

The PC instance copies the system `chronyd` binary into its work directory, because the distribution's AppArmor
profile is tied to `/usr/sbin/chronyd` and would stop it reading a configuration outside `/etc/chrony`. Deploy and
launch tools call this tool; they do not carry their own chrony setup.

## What the bound assumes

The inter-host term is both hosts' `System time`, the PC's root dispersion and root delay / 2 minus the camera
host's, a slack for the camera host's dispersion growing since the PC copied it, and each report's skew times its
age.

Why the subtraction is used: chrony's root delay and root dispersion are cumulative along the chain to the
reference. When the PC's source is the camera host, the PC's values are the camera host's plus the link between
them, so the difference isolates the link and the camera host's own upstream error (large when it follows an
internet source) cancels. That is why the bound is relative and not the sum of both hosts' absolute bounds.

What is and is not checked: the PC's selected reference id must be the camera host's address, so a different source
is refused as `ReferenceMismatch`. The assumption that the root fields are cumulative through the camera host is
chrony's documented behaviour, not something this library measures. The two reports are read at different instants,
so the difference can be slightly off; it is clamped at zero, and the copy slack and the skew-times-age terms cover
the drift between reads. The bound is derived from chrony's own reports. It has not been validated against an
independent reference such as a hardware sync pulse, so a qualified frame means "within chrony's claimed error",
not "measured within". Refusal policy: any report that is stale, has no recent source update, is unsynchronised, or
names another source is refused and the frame carries no time; nothing is estimated.

Step detection compares successive wall-minus-monotonic offsets. On Linux the two clocks share
frequency adjustment, so the difference changes only on a step. A change beyond both bracket widths
plus a margin (100 us by default) counts as one; a smaller step goes undetected, so that margin is
added to every conversion's uncertainty. Detection also takes one sample interval.

A report is refused when chronyd was last asked too long ago, and also when its last source update
is older than `max_source_update_age_ns`, so frequent polling cannot hide a quiet reference.

The parser expects the 14-field `chronyc -c tracking` layout of chrony 4.5 and fails closed on any
other field count.
