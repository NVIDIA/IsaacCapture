<!--
SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: Apache-2.0
-->

# Manus glove calibration tools

A Manus glove reports the hand relative to the glove itself, not to the room. On this rig the glove is
located through the controller mounted on it, which the headset tracks. Two fixed transforms connect the
two, and these tools measure them:

```text
room --(controller pose, tracked)--> controller --(1)--> glove main coil --(2)--> wrist and fingers
```

| | Tool | Measures | Redo it when |
| --- | --- | --- | --- |
| 1 | `calibrate_aim_to_coil.py` | where the glove's main coil sits on the controller | the mount on the controller changes |
| 2 | `calibrate_coil_to_wrist.py` | where the wrist sits relative to the main coil | the glove is recalibrated in Manus, or worn differently |

Do each one for the left and the right hand. Each tool records a short take in the terminal, solves it,
prints PASS or FAIL with what to change, and saves a passed result.

## Before you start

The tools record from a CaptureServer and need nothing else of the stack: no page, cameras or LiveKit.

1. Build IsaacCapture with the Manus plugin and start the CloudXR service, as
   [`capture_server/README.md`](../../capture_server/README.md) describes under "Setup (real)".
2. Start CaptureServer from the `egocap` directory:

   ```bash
   ISAAC_ROOT=/path/to/IsaacCapture uv run capture-server
   ```

   Nothing else may be using it: stop the full stack first, since its backend holds the same server.

## Running a calibration

From the `egocap` directory:

```bash
uv run python tools/calibration/calibrate_aim_to_coil.py --side left --finger middle
uv run python tools/calibration/calibrate_coil_to_wrist.py --side left
```

The tool then:

1. Lists what to switch on and the steps of the take. `--help` shows the same.
2. Waits for the devices it needs and names any that is missing.
3. Starts recording when you press Enter, and stops when you press Enter again.
4. Solves the take and prints each check with PASS or FAIL. A failed check says what to change.

Only the hand being calibrated is needed; the other glove and controller can stay off.

### 1. Aim to coil

The glove is not worn. One fingertip sensor is held still while the controller, with the glove's main coil
on it, is turned around it; the only mount position that keeps the sensor in one place is the answer.

1. Mount the controller on the glove. Turn on the glove, the controller and the headset, and keep the
   controller where the headset can see it.
2. Clamp one fingertip sensor so it cannot move, and not against metal. Pass its finger as `--finger`.
3. Start the tool and press Enter. Slowly turn the controller in many directions for about two minutes,
   without pulling on the clamped sensor. Press Enter to stop.

### 2. Coil to wrist

The glove is worn; the controller and the headset are not needed. The fingers are moved through a set of
motions, and the tool finds the one placement in which the fingertip sensors agree with the hand skeleton
that Manus reports.

1. Put the glove on, push each fingertip sensor onto its fingertip, and keep the hand empty and off metal.
2. Start the tool and press Enter. Follow the motions it lists: single fingers, fists, spreading, thumb
   circles and counting, at 3-4 seconds per motion, about 100 seconds in all. Press Enter to stop.

### Options

| Option | Meaning |
| --- | --- |
| `--side left\|right` | the hand to calibrate |
| `--finger NAME` | aim to coil only: the finger whose sensor is clamped (`thumb`, `index`, `middle`, `ring`, `little`) |
| `--capture-server URL` | CaptureServer to record from; default `ws://127.0.0.1:8082` |
| `--takes-dir DIR` | where the recorded take is kept; default `calibration_takes/` in the working directory |
| `--results FILE` | where a passed result is saved; default `wrist_calibration_offsets.json` in the station's calibration directory |
| `EPISODE` (first argument) | solve a recording again instead of recording a new take |

`EPISODE` is an episode directory, or a session directory to use its last episode. A recording made with the
full stack also works; it has glove data only while both gloves are on, and controller data only while the
headset sees both controllers.

## Results

A passed calibration replaces its entry in `wrist_calibration_offsets.json` in the station's calibration
directory (`EGOCAP_CALIBRATION_DIR`, by default `~/egocap_data/calibration`). Anything else in that file is
kept. A recording copies the file into every session, so each session carries the calibration it was
recorded with.

```json
{"left": {"T_controller_manus": [...], "T_coil_wrist": [...]}, "right": {...}}
```

Both are 4x4, in meters. `T_controller_manus` maps the main-coil frame into the controller aim frame;
`T_coil_wrist` maps the recorded `hand_pose` wrist frame into the main-coil frame. Chained with a recorded
controller pose they place the glove's hand in the room:
`T_room_wrist = controller pose x T_controller_manus x T_coil_wrist`.

A failed take saves nothing, so the earlier result stays. To try a calibration without touching the
station's file, pass `--results` with another path.

## If it does not work

| You see | Do this |
| --- | --- |
| `Cannot reach CaptureServer ...` | Check that CaptureServer is running and that the full stack is stopped. Pass `--capture-server` if it listens on another port. On a machine with proxy variables set, unset them for this command. |
| `Waiting for the ... glove` | Turn the glove on and check that it is connected. Manus takes a few seconds to find a glove after the server starts. |
| `Waiting for the ... controller` | Turn the controller on and keep it where the headset can see it; the headset must be tracking. |
| `Only N s of ...` | The take is too short, or the device dropped out for most of it. Record again. |
| `FAIL` | Follow the line under each failed check and record again. |

## How a take is judged

Both tools (`calib_common.py`) first drop what cannot be used: lost controller tracking, or frames without
fingertip data. They then split the take into 5-second pieces and throw out a piece that fits badly when it is
left out of the fit, trim the remaining outliers, and solve from several random starts to make sure there is
only one answer.

The checks, fixed in each script's `ACCEPT`:

| Check | Aim to coil | Coil to wrist |
| --- | --- | --- |
| fit error | < 5 mm | < 2.5 mm |
| first half vs second half | < 5 deg | < 1.5 deg |
| part of the take thrown out | < 30 % | < 50 % |
| enough motion | controller turned > 90 deg | every sensor turned > 60 deg, every fingertip moved > 20 mm |
| other | glove lag between 0 and 200 ms, controller tracking lost < 30 % | sensor within 30 mm of its fingertip |
| other answers that fit as well | none within 1 deg | none within 1 deg |
