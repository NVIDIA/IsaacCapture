<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
SPDX-License-Identifier: Apache-2.0
-->

# AgiLink OmniGlove plugin

This plugin converts the OmniGlove SDK's wrist-relative 21-landmark
`OMNIGLOVE_STREAM_POST_DATA` stream to OpenXR hand joints and injects it into
the Isaac Capture runtime. It supports one left and one right USB glove.

## Install

Install the host USB rule once, outside a container:

```bash
./src/plugins/omniglove/install_udev_rules.sh
```

Then build the plugin. The installer uses a sibling
`../agilink_omniglove_sdk` checkout when present. Otherwise it clones the
current SDK from GitHub into `.sdk/`, or fast-forwards that managed checkout on
later runs.

```bash
./src/plugins/omniglove/install.sh
```

Set `OMNIGLOVE_SDK_ROOT` or pass `--sdk-root` to use another SDK checkout. The
installer verifies that checkout with its own `SHA256SUMS`; it does not reset
the SDK to a historical tag. The SDK package configuration selects its shared
library, so a changed ABI version number alone is not a build blocker. Missing
or changed required C APIs fail the compile/link compatibility probe. At
runtime, a header/library ABI mismatch is rejected before device discovery.

## Run

Start CloudXR, load its environment, connect a calibrated glove, and run:

```bash
python -m isaaccapture.cloudxr.service run
source ~/.cloudxr/run/cloudxr.env
./install/plugins/omniglove/omniglove_plugin
```

The glove provides finger poses relative to its wrist but no world-space wrist
pose. By default the plugin obtains wrist pose from OpenXR optical hand
tracking. Set `OMNIGLOVE_WRIST_SOURCE` to `controller` or `auto` only after
measuring controller-aim-to-wrist transforms and exporting them as
`px,py,pz,qx,qy,qz,qw`:

```bash
export OMNIGLOVE_WRIST_SOURCE=controller
export OMNIGLOVE_AIM_TO_WRIST_LEFT="px,py,pz,qx,qy,qz,qw"
export OMNIGLOVE_AIM_TO_WRIST_RIGHT="px,py,pz,qx,qy,qz,qw"
```

Replace these format placeholders with measured offsets. Controller and Auto
mode reject missing transforms. For the tested mount's actual configuration
and calibration evidence, see [VALIDATION.md](VALIDATION.md).

## Data contract

- SDK positions are metres and quaternions are `(x, y, z, w)`.
- Finger orientations are converted from SDK distal +Z to OpenXR distal
  -Z by a local Y half-turn, preserving dorsal +Y. The wrist retains its
  proximal +Z; the right-hand wrist basis is converted separately.
- The 21 supplied wrist/finger landmarks map to their OpenXR equivalents.
- OpenXR palm and the index, middle, ring, and little metacarpals remain
  invalid because the SDK stream does not supply them.
- Unmapped, unconverged, invalid-bone-model, non-finite, zero-quaternion, and
  stale frames are not injected.

## Validation status

The API probe and hand-conversion contract test run during installation.
The controller-wrist hardware take `omniglove_2026-11-06_110404` passed hand CTS
(39 pass, zero fail). It was recorded before the plugin was committed, so it
is development evidence; formal submission needs a take from the pushed final
commit. OpenXR hot-unplug/recovery remains a separate hardware acceptance item.
See [VALIDATION.md](VALIDATION.md) for the evidence and remaining steps.

## Formal recording and packaging

After committing and pushing the plugin, record from a clean checkout:

```bash
./src/plugins/omniglove/record_cts.sh --wrist-config /path/to/mount.env
```

The wrapper rejects uncommitted/unpushed code and changes to the official CTS.
It records the installed binary's SHA-256 and the wrist configuration in
provenance notes. `--allow-dirty` enables development takes with their original
dirty provenance. `--note` adds hardware/firmware details; use
`--no-launch-cloudxr-runtime` when CloudXR is already running.

Review the take with the official viewer and package it with the same builder:

```bash
src/python/isaaccapture/cts/hand/checker/.venv/bin/python -m hand_cts.panel /path/to/take-hand.mcap
src/python/isaaccapture/cts/hand/checker/.venv/bin/python src/plugins/omniglove/package_cts.py /path/to/take-hand.mcap
```

Packaging reruns all official checks, preserves the original MCAP, labels,
provenance and log, checks their hashes, and prints the archive hash. Existing
archives are not overwritten. Upload the archive through the
[CTS form](https://forms.gle/hLqAhNkae4bhAUX79) with the integration PR URL;
keep recordings and archives outside Git.
