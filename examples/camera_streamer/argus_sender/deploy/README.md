<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Optional Orin/Thor service deployment

This deployment targets a clean Orin or Thor with a working Argus camera driver.
The installed board support package owns camera initialization, including any
serializer/deserializer setup needed at boot. Confirm that the cameras can
capture through Argus before deploying. The script installs the camera-streamer
services and configuration; it does not modify driver or vendor camera settings.

All deployment templates and helpers are local to this folder. The script copies
the sender, remaps and tests into a staging directory on the board, builds the
sender, runs its unit tests, and validates the config before replacing the current
installation. It installs:

- `camera-argus-sender.service`: single-process Argus capture, bounded restart
  attempts, clock pinning before start, and exclusive camera ownership.
- `camera-time-sync.service`: bounded chrony synchronization before stamping frames.
- `camera-vi-error-trace.service`: a dedicated VI error trace instance, active when
  the BSP exposes the matching tracepoint.
- A capped persistent journald drop-in, a chrony client drop-in, and the clock helper.

## Requirements

Run deployment from a Linux host with SSH, rsync, and the usual shell tools.
Configure key-based SSH access and the sudo permissions needed for service setup.
The script does not provision credentials or change sudo policy. Bare IP addresses
use the `nvidia` account unless `--user` is provided; SSH aliases use their configured
account. A target is always required.

Install the Jetson Multimedia API, CMake, a C++ compiler and chrony on the board.
The chrony configuration must include `/etc/chrony/conf.d/`; the standard Ubuntu
configuration normally provides this. The script checks that include and writes
only its own `camera-agx.conf` drop-in. Stop any manually launched sender before
using service deployment.

The host must serve a common time source over the camera LAN. The
`camera-chrony-host.conf.in` template contains `@ROBOT_LAN_SUBNET@`; render it for
your LAN and include it in the host chrony configuration. Deployment configures
the board to use the host IP passed below.

## Deploy, inspect, or restore

From this folder, with the bundled remaps present:

```bash
export SENDER_SSH=user@camera-board
export HOST_LAN_IP=host-camera-lan-address  # Replace with the actual host LAN IP.
./deploy_argus_sender.sh "$SENDER_SSH" --host-lan-ip "$HOST_LAN_IP"

# Read-only service, clock, binary, configuration and synchronization audit:
./deploy_argus_sender.sh "$SENDER_SSH" --host-lan-ip "$HOST_LAN_IP" --check

# Restore the snapshot taken immediately before the latest deployment:
./deploy_argus_sender.sh "$SENDER_SSH" --rollback
# --restore is an alias for --rollback.
```

The target is `~/camera_streamer/argus_sender`, including its `remaps/` directory.
Its tests are included in `~/camera_streamer/argus_sender/tests`. The deployment
leaves the other components and example-level tests in place. Launch the receiver
and consumers on the host using the main example README.

Deployment always builds in a fresh staging directory. To rebuild manually in the
installed source tree, remove its `build/` directory before running `build.sh`;
the transferred CMake cache contains the staging paths.

## Rollback behavior

Before activation, the script snapshots the previous sender tree, including its
tests, its owned service/configuration files, and the enabled/running states of its services,
`nvargus-daemon`, and chrony. The snapshot is under
`~/camera_streamer/.deployment-backup`. Each deployment replaces this restore point
only after the staged build and snapshot have succeeded.

If activation fails, the script automatically restores that snapshot. An explicit
`--rollback` does the same:

- For a first installation, remove the newly installed sender directory, service
  files and drop-ins, and restore the recorded platform service state.
- For an update, restore the previous binary, remaps, configuration and service
  state, including whether the previous sender was running or stopped.

Rollback does not restart unrelated camera services or modify camera drivers.
Existing receiver/consumer directories and journal logs are retained. Installed
prerequisites and runtime clock pins are not removed or reset. A repeated restore
of the same snapshot does nothing. If SSH is interrupted during activation, retry
`--rollback` after reconnecting.

## Diagnostics

The sender service conflicts with `nvargus-daemon`, because single-process Argus
must own the cameras exclusively. Deployment saves the daemon's prior state before
disabling it; rollback restores that state. Do not run another Argus owner
concurrently. Capture retries are bounded by `StartLimitIntervalSec=200`,
`StartLimitBurst=6`, and `RestartSec=15`.

```bash
ssh "$SENDER_SSH" 'journalctl -u camera-argus-sender -b --no-pager -n 100'
ssh "$SENDER_SSH" 'systemctl show -p ActiveState,NRestarts camera-argus-sender'
ssh "$SENDER_SSH" 'journalctl -u camera-vi-error-trace -f -o cat'
```

The clock helper pins runtime camera clocks and reports the current power mode.
Review the board's supported power/thermal configuration before changing it.
Wall-clock latency requires both hosts to be synchronized. The time-sync wait is
bounded, so inspect chrony state if the source was unavailable at boot.
