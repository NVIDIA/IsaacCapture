<!--
SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: Apache-2.0
-->

# Manus glove calibration in CaptureServer

CaptureServer can run the Manus SDK's glove calibration on its host, on the same WebSocket connection as
capture, and write the resulting MCAL files into the server's MCAL directory. The SDK work happens in the separate `manus_glove_calibrator` process (see
`src/plugins/manus/calibration/README.md`); CaptureServer supervises it and stores its output.

This is vendor glove calibration only. It needs no camera, OpenXR session or capture session.

## Setup

```sh
capture-server --manus-calibration-dir /var/lib/egocap/mcal \
               [--manus-calibrator /path/to/manus_glove_calibrator]
```

- `--manus-calibration-dir` enables the feature. New files are written there as plain `.mcal` files. It is the
  same folder the server offers its MCAL files from, so a file written here is found wherever that folder is
  read. It is created on first use.
- The worker is the explicit `--manus-calibrator` path (an unusable path is a startup error), else
  `$ISAAC_ROOT/install/plugins/manus/manus_glove_calibrator`, where the Manus component installs it. When no
  worker is found, ordinary capture is unaffected and only a calibration request is rejected with
  `INVALID_CONFIG` saying so.
- Stop every other Manus consumer first: the dongles are not shared.

## Ownership and admission

CaptureServer admits one session at a time, from acceptance through cleanup. A calibration and a capture
session are mutually exclusive: a request for one while the other is active (starting, running, or stopping)
is rejected `BUSY` and never interrupts it. The same single owner connection and single outbound writer are
used. Capture `Stop` addresses capture sessions and `StopCalibration` addresses calibration sessions; neither
stops the other kind.

The station stays unavailable (`BUSY` for both) while a worker could not be reaped, because it may still hold
the gloves. Restart the server once the process has exited. A terminal error is not evidence the hardware
was released.

If the owner connection closes or the server shuts down, the calibration is cancelled with a bounded cleanup.
There is no reconnect grace period.

## Python API

```python
client = CaptureClient("ws://host:8082")
await client.start_calibration(
    sides=(HandSide.LEFT, HandSide.RIGHT),
    on_started=...,
    on_status=...,
    on_stopped=...,
)
await client.confirm_calibration_step(step_token=status.step.token)  # on an operator action only
await client.stop_calibration()  # idempotent
```

- `start_calibration` returns when the server accepts, not when the calibration finishes. A rejected start
  raises `RequestRejectedError` and calls no callback.
- Every accepted attempt calls `on_stopped` exactly once. `CalibrationStopped.reason` is the server's
  (`COMPLETED`, `CANCELLED`, `FAILED`, `UNCERTAIN`), or `None` if the connection or protocol failed first: the
  outcome is then unknown, no files are offered, and the gloves may still be in use.
- `confirm_calibration_step` returns after the worker accepted starting the step. That is not completion of
  the step or the calibration. Repeating an accepted token returns at once and starts nothing; a stale or
  unoffered token raises `RequestRejectedError` and leaves the attempt unchanged. A pending confirm is
  released with `CalibrationEndedError` if the calibration ends first.
- Callbacks run one at a time on the client's loop. A callback that raises cancels the calibration and its
  exception is named in the terminal `error_message`.
- A client has one active attempt of either kind.

## Steps

`CalibrationStatus.step` carries what the SDK provides, never hardcoded: `title`, `instruction`, `index` and
`count` (steps for that glove), and `mode`:

- `HOLD`: the operator is in the pose, then confirms. `duration_seconds` is 0.
- `MOTION`: confirming starts a timed sampling window of `duration_seconds`.

`continuous` is true for an SDK step whose time is negative, a continuous step with an estimated duration.
`estimated_seconds` is that estimate, guidance for the operator only. The vendor documents no completion
signal for such steps, so neither CaptureServer nor the worker claims one: confirming starts the SDK step
and the next prompt follows. It is not an instantaneous capture and not a guaranteed timed completion.

## Files

A calibration that completes returns one `.mcal` basename per requested side. The worker writes the files
straight into the managed directory as `<attempt>_<side>.mcal` and never overwrites an earlier file; a failed,
cancelled or uncertain attempt removes whatever it wrote. The clients never choose paths or upload files.

Calibration only writes into the directory. It does not list, read or select files; those belong to whatever
reads the folder.

## Wire

The existing numeric envelope is extended; existing capture values and the protocol version are unchanged.

| Value | Message | Direction |
| --- | --- | --- |
| 13 | `StartCalibration` (`sides`: array of `HandSide`, `LEFT=1`, `RIGHT=2`) | client to server |
| 14 | `ConfirmCalibrationStep` (`step_token`) | client to server |
| 15 | `StopCalibration` | client to server |
| 16 | `CalibrationStarted` (`sides`) | server to client |
| 17 | `CalibrationStatus` (`phase`, `gloves`, `step`, `step_elapsed_seconds`) | server to client |
| 18 | `CalibrationStepAccepted` (`step_token`) | server to client |
| 19 | `CalibrationStopped` (`reason`, `result`, `error_message`) | server to client |

All carry `version` and `session_id` like capture messages. `RequestRejected.request_type` may be 13, 14 or
15, and a rejected `ConfirmCalibrationStep` echoes `step_token`. `CalibrationPhase`: `CONNECTING=1`,
`AWAITING_STEP=2`, `CAPTURING_STEP=3`, `SAVING=4`, `STOPPING=5`. `CalibrationStopReason`: `COMPLETED=0`,
`CANCELLED=1`, `FAILED=2`, `UNCERTAIN=3` (the worker's outcome could not be established, for example it
stopped answering and was terminated). `result` is present exactly for `COMPLETED`.

Message names in the design for this feature were symbolic (`startCalibration`, and so on); the numeric
values above are their wire form here, so the existing numeric discriminators are kept.

## Status

The CaptureServer side is exercised only against a scripted stand-in for the worker. The worker itself has
not completed a calibration with real gloves: with the real SDK and no working Manus license on the test
machine, the SDK call that starts a step did not return within the supervisor's bound, and the cause is not
established. An end-to-end calibration and MCAL export are not yet demonstrated.
