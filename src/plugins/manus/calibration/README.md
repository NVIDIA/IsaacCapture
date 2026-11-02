<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Manus glove calibration worker

`manus_glove_calibrator` drives the Manus SDK's glove calibration routine and exports the resulting
opaque MCAL files. It ships in the Manus plugin package as its own executable. It is not an Isaac
Capture tracking plugin: it registers no provider and publishes no device data. Tracking later
loads the exported files through `manus_hand_plugin --left-calibration=` / `--right-calibration=`.

The worker owns the SDK connection for one calibration attempt, then exits. Stop every other Manus
consumer first (the tracking plugin, other SDK clients): the dongles are not shared.

## Process contract

```
manus_glove_calibrator --ipc-fd=N [--discovery-timeout=SECONDS]
```

- `N` is an inherited, already-connected `AF_UNIX`/`SOCK_STREAM` socket (a `socketpair` end). The
  worker reads commands from it and writes events to it. stdout/stderr carry only SDK and worker
  logging and are never parsed.
- One attempt per process. After the terminal `result` event the worker exits: `0` for
  `completed`/`cancelled`, `1` for `failed`.
- `SIGTERM`, EOF on the IPC socket, and loss of the parent each request cancellation. The worker
  stops the SDK calibration, shuts the SDK down, sends `result` (`cancelled`) if it still can, and
  exits. The supervisor escalates to `SIGKILL` after its own bounded grace period.
- `result` is sent only after SDK shutdown has been attempted. `released` is false when the SDK calibration
  could not be stopped or the SDK could not be shut down, whatever the outcome: the process then exits but
  the gloves may still be in use, and a supervisor must keep them unavailable. A supervisor that never receives `result`
  must treat the outcome as uncertain, never as success.

## Wire format

Newline-delimited UTF-8 JSON objects, at most 65536 bytes each, over the IPC socket. Every message
is an object with `"v": 1` and a string `"type"`. Unknown keys and wrong types in a command are
rejected with a `rejected` event (`reason: "invalid"`); the worker never crashes on bad input. Times
are seconds as JSON numbers. Side values are the strings `"left"` and `"right"`.

### Commands (supervisor to worker)

| `type` | Fields | Meaning |
| --- | --- | --- |
| `start` | `attempt` (string, 1-64 chars of `[A-Za-z0-9_-]`), `sides` (non-empty, unique, subset of `left`,`right`; processed in the order left then right), `staging_dir` (absolute path of an existing directory the worker may write into) | Begin the attempt. Accepted once per process. |
| `confirm` | `attempt`, `token` | Start the SDK step offered by the `prompt` with this token. |
| `cancel` | `attempt` | Cancel the attempt. Idempotent. |

### Events (worker to supervisor)

| `type` | Fields | When |
| --- | --- | --- |
| `ready` | `worker_version` (string), `sdk_version` (string or null) | Once, after the IPC socket is open and before any command is read. |
| `status` | `attempt`, `phase` (`discovering`, `awaiting_step`, `capturing_step`, `saving`, `stopping`), `gloves` (array of `{side, glove_id}` discovered so far) | On every phase change. |
| `prompt` | `attempt`, `token`, `side`, `glove_id`, `index` (0-based), `count` (steps for this glove), `title`, `instruction`, `mode` (`hold`/`motion`), `duration_s`, `continuous` (bool), `estimated_s` | When a step awaits confirmation. `token` is unique per prompt within the attempt. `mode` is `motion` when the SDK reports a positive step time. Otherwise it is `hold` with `duration_s` 0; a negative SDK time (a continuous step) also sets `continuous` and puts the SDK's estimate of how long the movement lasts in `estimated_s`, which is guidance for the operator and not a completion signal. |
| `step_accepted` | `attempt`, `token` | The SDK accepted starting the step. Re-sent unchanged for a repeated `confirm` of the same token, without starting another step. |
| `step_progress` | `attempt`, `token`, `elapsed_s` | About every 0.25 s while a `motion` step samples; `elapsed_s` never exceeds `duration_s`. Elapsed sampling time, not a quality measure. |
| `rejected` | `attempt` (string or null), `command`, `reason` (`invalid`, `unknown_attempt`, `stale_token`, `unexpected`, `busy`), `message`, `token` (when relevant) | A command was refused. No state changes. |
| `exported` | `attempt`, `side`, `glove_id`, `file` (basename inside `staging_dir`), `bytes`, `sha256` (lowercase hex) | One MCAL file written and flushed. |
| `result` | `attempt` (or null), `outcome` (`completed`/`cancelled`/`failed`), `files` (array of `{side, file, bytes, sha256}`, present only for `completed`), `error` (`{code, message, side?, step?}`, present only for `failed`), `released` (boolean, always present) | Terminal. Sent exactly once, after SDK shutdown. |

Error `code` values: `sdk_init_failed`, `no_gloves`, `glove_missing`, `glove_replaced`,
`sdk_error`, `step_failed`, `export_failed`, `protocol_error`, `internal`. The message names the
glove side and step index where known.

## Behavior rules

- Discover the requested sides for up to `--discovery-timeout` (default 15 s); pin each glove ID for
  the attempt. A pinned glove that disappears or changes ID fails the attempt
  (`glove_missing`/`glove_replaced`).
- Read step count, titles, instructions and times from the SDK; never hardcode them. Every step of a
  glove is checked before the first prompt, and a non-finite time fails the attempt with `step_failed`.
- A continuous step (negative SDK time) is paced by the operator: the operator performs the pose or
  movement, confirms, the worker starts the SDK step and offers the next prompt. The SDK documents no completion signal for such steps, so the worker never
  claims one; whether a routine made of them yields a usable calibration is established only by the
  exported files loading in tracking.
- Every prompt needs its own `confirm`, including the first prompt of the second glove. A `confirm`
  that is not for the currently offered token is `rejected` (`stale_token`) and changes nothing. A
  step is not skippable.
- `hold` steps: the operator holds the pose, then confirms; the SDK samples once started.
  `motion` steps: confirming starts the SDK sampling window; the worker emits `step_progress` and
  offers the next prompt when the window ends.
- Check both the SDK return code and the operation-result boolean on every SDK call.
- Export per glove, one glove at a time, and read the size and the bytes back to back
  (`CoreSdk_GetGloveCalibrationSize(glove_id)` then `CoreSdk_GetGloveCalibration`); never overlap
  left and right exports. Reject an empty export. Write `<file>.part`, flush and fsync, then hard-link
  to the final basename `<attempt>_<side>.mcal` inside `staging_dir`.
- Cancellation works during discovery, while awaiting confirmation, while sampling and while
  saving. After cancellation no file is reported as exported by a `completed` result.
- After `result` the worker ignores further commands without replying; it is already exiting.
- A worker blocked inside an SDK call cannot enforce its own limits. The supervisor therefore bounds every
  answer only the worker can give (startup through the first prompt, a confirmed step's acceptance, the time
  from the last step to the result) and terminates a silent worker; the outcome is then uncertain and
  nothing is published. The time the operator takes to confirm a prompt is never bounded.
- The worker never publishes anything. Making a set of exported files visible to a consumer is the
  supervisor's job (see the Python adapter).

## Build

The target is part of the `manus` component and is built only when the Manus SDK is found, like the
rest of the plugin (`MANUS_SDK_ROOT`, see `install_manus.sh`). The step/state logic and the wire
format live in a static library with no SDK dependency so they are unit-tested without hardware; the
executable adds the real SDK binding.
