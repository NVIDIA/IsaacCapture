<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Agibot Co., Ltd.
SPDX-License-Identifier: Apache-2.0
-->

# OmniGlove integration validation

Updated: 2026-11-06 (Asia/Shanghai). Hardware qualification and formal upstream
submission have separate evidence. A pending item below is not a pass.

## Recorded hardware baseline

| Item | Evidence |
| --- | --- |
| Isaac Capture baseline | `71f22fd93c0bdff871e731d66b1d10589b2e3aeb` |
| Plugin | `omniglove_plugin`, manifest version `1.0.0` |
| SDK release / commit | `1.0.0` / `a2f899cd45c51ae01c9bbca9b893e4afba41c0d7` |
| SDK header / runtime ABI | Both `6.0.0` |
| Platform tested | Linux x86_64; aarch64 supported by the build, not hardware-tested here |
| USB devices | Left `cafe:4001`, right `cafe:4002` |
| SDK stream | POST_DATA, DIRECT, nominal 125 Hz |
| Wrist source qualified | `controller` |
| Headset / controller model | PICO 4 Ultra / PICO 4 controllers, operator-confirmed; original take has no model fields |

## Hardware CTS evidence

- Take: `omniglove_2026-11-06_110404`, recorded at 11:04:04 UTC+08:00.
- Verdict: **PASS**; 18,678 hand records, 39 passing checks, zero failures.
- `optional_joints.geometry_consistent`: insufficient data, expected because
  the five optional joints are invalid. The overall verdict is pass.
- All 11 action windows were backed by frames. Left/right each had 9,339
  records, including 9,337 active records.
- Captured commit: the baseline above, with `repo_dirty=true`. The plugin was
  staged but not committed; this is development evidence. Do not edit its
  provenance to claim a later commit.
- Raw MCAP, label sidecar, provenance and recorder log stay together outside
  Git. The submission archive uses the official CTS bundle builder and adds
  `report.json`, `report.txt`, per-input hashes and checker provenance.
- Manual replay review and final-commit recording remain pending. The report
  establishes automated checks; it does not establish the operator's replay review.

## Tested wrist installation offsets

The independent successful take used these exact fixed transforms. Values
are metres and quaternion `qx,qy,qz,qw` in controller aim coordinates.
They apply to the tested mount, not every OmniGlove installation.

```bash
export OMNIGLOVE_WRIST_SOURCE=controller
export OMNIGLOVE_AIM_TO_WRIST_LEFT="0.044351775,-0.022869715,0.071703646,-0.122489204,-0.165170245,0.646336332,0.734822789"
export OMNIGLOVE_AIM_TO_WRIST_RIGHT="-0.044435690,-0.021819216,0.082681299,-0.130124469,0.161913518,-0.661683797,0.720434721"
```

The final left offset was estimated from the physical CTS action constraints
in `omniglove_2026-11-06_104642`: table level and fingertip contact/roll, using
held-out time blocks and excluding the final flat-hand window from fitting.
The original take's offline reprojection passed 36 checks with zero failures;
the new independent take above then passed 39 checks. Only a fixed wrist
installation transform was adjusted. Original recordings, labels, finger
geometry and checker thresholds were not changed. This was geometric mount
calibration; it was not a simultaneous optical-wrist measurement.

For a new mount, obtain a trusted wrist reference and compute
`T_aim_to_wrist = inverse(T_world_aim) * T_world_wrist` over multiple poses.
Verify the result using independent positions, wrist rotations and physical
contact, then record a fresh CTS take.

## Other existing evidence

The 2026-11-01 handover records:

- Left and right USB identification and individually valid 21-joint conversion.
- Five-minute dual-hand SDK streaming: 37,501 frames per hand, about 125 Hz;
  zero sequence gaps, source errors, malformed frames and callback drops.
- SDK hot-unplug/reconnect, tested with the right glove representing the same
  hardware/firmware pair: automatic recovery and connection generation 1→3.

These historical results establish the USB/SDK layer. They do not establish
OpenXR injector deactivation/recreation or long-term headset tracking.

## OpenXR hot-unplug/recovery evidence

Take `omniglove_2026-11-06_150705` was recorded at pushed commit
`97c62090a97c5376cf8264ff8b90acd8f72a237d`, with `repo_dirty=false`, the same
controller offsets, PICO 4 Ultra and PICO 4 controllers. The complete MCAP
contains 6,747 hand records per side over 81.94 seconds.

Both hands became inactive downstream and recovered in the same process run:

| Inactive hand | Downstream inactive interval | Other hand during that interval | Recovery generation |
| --- | --- | --- | --- |
| Left | 22.53 s | Right: 1,875/1,875 records with 21 valid joints; max record gap 14.55 ms | 1→3 |
| Right, main interval | 25.50 s | Left: 2,096/2,096 records with 21 valid joints; max record gap 15.38 ms | 3→5 |

The other hand's wrist-relative fingertip displacement reached 100.79 mm
during the left interval and 125.54 mm during the right interval, establishing
changing finger data. Both recovered hands also continued updating.
The operator waited before reconnecting; inactive interval lengths include
that wait and do not measure latency after reinsertion.

An additional right-hand inactive interval lasted 0.18 s before the main
interval, with generation 1→3 on recovery. Its physical cause is unknown;
the left hand remained active throughout. This event is retained in the evidence.

Functional hot-unplug/recovery is verified. Exact 200 ms cutoff and reinsertion
latency were not measured because SDK sample and physical unplug/reinsert
timestamps are absent. Downstream inactivity is encoded by an absent
`HandPoseRecord.data` payload, not a separate `is_active` field.
The take has no action labels and is separate from formal CTS evidence.

## OpenXR hot-unplug/recovery procedure

Keep the final wrist configuration and both gloves connected, connect the
headset, and start a session consuming the two standard hand collections.
Capture the plugin log and consumer recording as a separate hardware test.
Do not use unplug segments as CTS action labels.

1. Confirm both hands are active and their fingers are changing.
2. Unplug the left glove and wait at least two seconds. Record the physical
   unplug time and the last SDK sample time.
3. Check the left injector deactivates after 200 ms without a fresh sample
   (allow one worker scheduling interval), that the consumer reports the left
   hand inactive, and that the right hand continues changing.
4. Reconnect the left glove without restarting any process. Check that the
   left hand becomes active, its connection generation increases, and fingers
   update from new samples.
5. Repeat for the right glove and check the left remains active.
6. Save timestamps, the consumer's active flags and the plugin log. A log
   transition alone does not establish what the downstream consumer saw.

The plugin now logs `left/right hand inactive: SDK frame stale or invalid`,
`hand inactive: wrist pose unavailable`, and `hand active` with connection
generation/source sequence. Logs occur on injector transitions, not every
worker iteration.

## Software verification on 2026-11-06

- Installer: selected local SDK checksum verification, build and installation passed.
- CTest: `omniglove_api_compat_probe` and `omniglove_hand_conversion_contract_test` passed.
- All plugin C++ sources/headers passed clang-format 14.
- Strict Sphinx `make current-docs` passed with no warnings.
- Full pre-commit passed with `SKIP=check-copyright-year`, matching CI.
- Controller/Auto with missing mount offsets failed before opening USB/XR.
- Recorder smoke checks verified clean/upstream guards, unchanged CTS,
  exported offsets, binary hash and preservation of operator notes.
- Existing take packaging passed archive integrity and per-input hash checks.

## Final submission checklist

- [x] Confirm headset and controller models: PICO 4 Ultra / PICO 4 controllers.
- [x] Functional OpenXR hot-unplug/recovery evidence for both hands; timing
      measurement limits and the extra right-hand transition recorded above.
- [x] Software build, conversion/ABI tests, C++ format, strict docs and full
      pre-commit verified against the final source.
- [x] Commit the plugin with the operator's Git identity and DCO sign-off.
- [x] Push to AgibotTech/IsaacCapture and open integration PR #1445.
- [ ] Record from that pushed commit with a clean checkout and unchanged CTS:

```bash
./src/plugins/omniglove/record_cts.sh --wrist-config /path/to/mount.env \
    --no-launch-cloudxr-runtime --note "headset=PICO 4 Ultra; controller=PICO 4 controllers"
```

- [ ] Review the new take in `hand_cts.panel`; verify handedness, axes,
      continuity and the physical wrist/finger alignment.
- [ ] Package with the viewer or `package_cts.py`; quote take, verdict, archive
      size and SHA-256 prefix in the PR.
- [ ] Upload via https://forms.gle/hLqAhNkae4bhAUX79, associating the archive
      with the PR. Keep the archive outside Git.
- [ ] Resolve upstream CI and reviewer feedback.

Optical and Auto wrist modes remain unqualified by the controller take.
Repeat hardware tests after changing the SDK, firmware, skeleton conversion
or physical mount.
