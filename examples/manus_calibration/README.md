<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Manus glove calibration

Calibrate Manus gloves from a browser and save the MCAL files on the machine the gloves are
plugged into. The page guides the operator through the steps the Manus SDK provides; the SDK work
happens in the `manus_glove_calibrator` executable (see
[`src/plugins/manus_calibration/README.md`](../../src/plugins/manus_calibration/README.md)).
No capture session, OpenXR runtime, or CaptureServer is involved.

This is vendor glove calibration only. Camera calibration, spatial alignment, and flex-sensor
offsets are separate operations.

## Prerequisites

- The Manus SDK, with the Manus plugin built against it (`MANUS_SDK_ROOT`, see
  `src/plugins/manus/install_manus.sh`), and `manus_glove_calibrator` built and installed to
  `install/plugins/manus/`: `cmake --build build --target manus_glove_calibrator` then
  `cmake --install build --component manus` from the repository root.
- A Manus license for the SDK's `Integrated` feature, as Manus requires. Glove discovery and the step
  list worked without one in testing, but what the SDK does for a step start, finish and export without a
  license has not been established (see Status).
- Gloves and dongle attached to this host, with `src/plugins/manus/70-manus-hid.rules` installed.
- **Every other Manus consumer stopped**: the tracking plugin, capture sessions, other SDK
  clients. The dongles are not shared.

## Run

The page needs only Python 3.11 or newer and the standard library. The worker supervisor and file storage
come from the `isaaccapture-manus-calibration` library in `src/plugins/manus_calibration/python`; install it
(`pip install ../../src/plugins/manus_calibration/python`) or put it on `PYTHONPATH` as below:

```bash
cd examples/manus_calibration
PYTHONPATH=python:../../src/plugins/manus_calibration/python python3 -m isaaccapture_examples.manus_calibration \
    --worker ../../install/plugins/manus/manus_glove_calibrator \
    --out-dir ~/manus-calibration
```

Open `http://127.0.0.1:8080`. The page has no authentication and can drive hardware, so it binds to
loopback and refuses requests whose Host or Origin is not its own; to use it from another machine,
forward the port (`ssh -L 8080:127.0.0.1:8080 host`).

| Option | Meaning |
| --- | --- |
| `--worker PATH` | The calibration worker executable. |
| `--out-dir DIR` | Where MCAL files are saved, on this host. A launch setting, never browser input. |
| `--sides {left,right,both}` | Gloves preselected in the page. |
| `--host`, `--port` | Where to serve. |
| `--discovery-timeout SECONDS` | How long the worker waits for the requested gloves. |
| `--continuous-pacing operator\|window` | How a continuous step is paced (default `operator`). `operator`: move, then confirm. `window`: confirm, then move for the SDK's estimate while the worker holds the step. Experimental until compared on real gloves. |

## What the operator sees

Select the gloves and start. For each SDK step the page shows the SDK's title and instruction.
Every prompt needs its own confirmation, including the first prompt of the second glove, and the
confirmation starts that step in the SDK.

- A step with a positive SDK time starts a timed sampling window when confirmed, with a progress
  bar showing elapsed time (not a quality score).
- A step with no time is confirmed once the operator is already in the pose.
- A step with a negative SDK time is a continuous step whose duration the SDK only estimates. The worker
  has no completion signal for it, so a negative time does not prove the SDK finished sampling. With
  `--continuous-pacing operator` the page asks you to get into the pose, then confirm, and the next
  prompt follows at once. With `window` it asks you to confirm and move immediately; the worker holds
  the step for the estimate and the bar shows elapsed time against it, not completion.

One browser owns an attempt, and it must have the page's event stream open to start one. Other
browsers see its progress but cannot confirm or cancel. If the owning tab disconnects, the attempt is
cancelled after a 3 second grace period, so a page reload does not cancel it; a reload in time keeps
ownership.

## Unresponsive worker

The supervisor bounds the answers only the worker can give: startup through the first prompt, the
acknowledgement of a confirmed step, and the time from the last step to the result. When the worker
stays silent past a bound the attempt ends with an uncertain outcome, nothing is saved, and the
worker is terminated. This contains an unresponsive SDK call; it does not show that a license is
missing. A silent worker restarts the bound on any event it sends, so these are inactivity bounds, not
absolute limits. The time the operator takes to confirm a step is never limited. If a worker cannot be
stopped at all, new attempts are refused until the program restarts, because it may still hold the gloves.

## Status

Exercised in tests: the worker, the supervisor, storage and the page behave as described against a scripted
stand-in for the worker, and the real SDK discovers both gloves and returns the step lists.

Not verified: a calibration run from the first step to saved files with real gloves. In testing, the SDK
call that starts a step did not return within 30 seconds, and the cause was not established. So an
exported file, its calibration quality, and loading it through the tracking plugin remain unverified.

## Saved files

`--out-dir` is a plain folder. The worker writes each file straight into it as `<attempt>_<side>.mcal`
(a half-written file carries a `.part` suffix and is never left behind), and a name that exists is never
overwritten. A cancelled, failed, or uncertain attempt removes whatever it wrote, so only a completed
calibration leaves files. When a calibration completes the page lists the files with a download link each,
so the operator can save them to the machine the browser runs on. Loading a file for tracking is done with
`manus_hand_plugin --left-calibration=FILE --right-calibration=FILE`.

## Tests

```bash
cd tests/python/examples/manus_calibration
uv run --extra dev pytest
```

CTest runs the same suite as `manus_calibration_*` (label `manus_calibration`).

The tests drive the adapter, controller and HTTP server against a scripted stand-in for the worker, so
they establish the supervision, storage, ownership and web-safety logic, not that the SDK produces a
valid MCAL file.
