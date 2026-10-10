<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Recording a take

One recording of a prescribed ten-step motion script, plus the label sidecar that
[the checker](checker/README.md) reads. A panel in your browser tells the performer
what to do; they press a button once they are in each pose.

**Linux only**: recording goes through Isaac Teleop, whose wheel is `linux_x86_64`. The
checker has no such constraint, so a take recorded here can be verified on macOS too.

Nothing under [`capture/`](capture) is meant to be read or edited — see
[Do not edit the script](#do-not-edit-the-script).

## Before you start

- **The project built**, so that an `isaaccapture` wheel exists under `install/wheels/`
  or `build/wheels/`.
- **`uv`, `curl` and `unzip`** on the same Linux host.
- **`git-lfs`**, so the clone holds the spoken cues as audio rather than LFS pointers.
- **`aplay`** — the spoken cues go through it. It comes from `alsa-utils`
  (`apt install alsa-utils` on Debian or Ubuntu). Install it before your first take: a
  missing `aplay` fails part way into the recording rather than at startup.
- **A headset**, plus CloudXR installed (by default at `~/.cloudxr`) — see
  [Get the headset streaming](#get-the-headset-streaming). It has to be connected even
  when your own device produces the joints, because the OpenXR session comes from
  CloudXR. **Body tracking** on the headset is needed only where the headset is also
  the body source.
- **A performer who can stand up and follow spoken instructions.** Two people is easier
  than one — see the note about the space bar below.

## Set up, once per clone

Two scripts, in this order. The checker's comes first because it generates what the
panel reads the recording back with.

```bash
src/python/isaaccapture/cts/full_body/checker/setup_env.sh
src/python/isaaccapture/cts/full_body/capture/setup_env.sh
```

Both are idempotent and write nothing outside their own directory. Re-run them after
pulling.

## Get the headset streaming

**Every take needs this, including one whose joints come from your own device.** CloudXR
is the OpenXR runtime the capture itself runs on, and it has no system to hand out until
a headset has loaded the web client and pressed CONNECT. Skipping it does not raise:
**Start recording** blocks inside session creation, logging `OpenXR HMD form factor is
unavailable; waiting for a system...` once a second, and Python does not take Ctrl+C
there. To record without a headset, see
[Recording from your own device](#recording-from-your-own-device).

Every `isaaccapture` command below runs under the interpreter `capture/setup_env.sh`
built. Nothing else on the host has the package, and Ubuntu has no bare `python` at all.

```bash
PY=src/python/isaaccapture/cts/full_body/capture/.venv/bin/python
```

**Accept the CloudXR EULA, once per machine.** Review the licence, then

```bash
$PY -m isaaccapture.cloudxr.service run --accept-eula
```

It is remembered in `~/.cloudxr/run/eula_accepted`, so that is once per install
directory, not once per take.

**Start the service in a terminal of its own and leave it open.**

```bash
$PY -m isaaccapture.cloudxr.service run
```

`run` is the foreground form and prints the runtime's log, which is where a headset that
will not connect says why; Ctrl+C stops it. `start` is the same service detached, with
`stop`, `status` and `logs` beside it. The runtime is a host singleton on port 48322, so
run one form or the other, never both. `record.sh` starts its own when nothing is up,
but a runtime takes tens of seconds to come up and one of these outlives any number of
takes.

**Connect the headset over Wi-Fi.** Nothing reaches the panel until this is done, and it
sits at *waiting for a frame with valid joints* meanwhile. Put the headset on the same
subnet as this host, then

```bash
$PY -m isaaccapture.cloudxr.webclient --print-only
```

That prints the streaming target `https://<this-host>:48322/` and the client URL with
that address already filled in. **Open the streaming target in the headset's browser
first and accept the self-signed certificate** — until you have, the client page loads
normally and CONNECT fails with nothing in either log. Then open the client URL and
press CONNECT.

Without `--print-only` the same command types that URL into the headset over USB `adb`,
which saves the typing and needs USB debugging authorised. The cable carries `adb`,
never the video. Re-run either form whenever the headset browser gets closed; it does
not disturb a running take.

`record.sh` passes anything it does not understand through to the panel, so the runtime
flags work from there too — `--cloudxr-install-dir` (default `~/.cloudxr`),
`--cloudxr-device-profile` (default `Quest3`), and the rest of
`capture_panel.py --help`.

Two references worth having open the first time:

- [`docs/source/references/cloudxr.rst`](../../../../../docs/source/references/cloudxr.rst) —
  device profiles, foreground vs detached service, out-of-band and USB-only setups.
- [`docs/source/device/body_tracking.rst`](../../../../../docs/source/device/body_tracking.rst)
  — PICO only, and skippable if your joints come from elsewhere: which Motion Tracker
  configurations are supported (5, 3 or 2 trackers), how to calibrate them, and what
  body tracking needs from the headset. **Read it before your first PICO take**, since
  without body tracking the recording holds no usable joints and only the container
  checks can run.

## Recording from your own device

The joints come from the headset's own `body.pico-xr` backend unless you say otherwise.
Another full-body device arrives as a **vendor**, a backend id the session resolves when
it is constructed, fed by a **plugin** process the session launches. Both are
`capture_panel.py` flags, and `record.sh` passes them through:

```bash
src/python/isaaccapture/cts/full_body/capture/record.sh mysuit \
    --plugin my_body_plugin \
    --vendor body.my-vendor \
    --vendor-param collection_id=my_body
```

- `--plugin NAME` names a directory under `plugins/` or `install/plugins/`, so the
  plugin must be installed (`cmake --install build`) first. It is launched as required:
  one that fails to load stops the take rather than recording a file with no body in it.
- `--vendor ID` is rejected at session construction if the registry does not know it,
  not at the first frame.
- `--vendor-param KEY=VALUE` is repeatable and free-form. `collection_id` must match
  what the plugin publishes under; a mismatch records an empty take and looks exactly
  like a device that never connected.
- **A headset still has to be connected.** Not for the joints — for the OpenXR session,
  which CloudXR will not create until one is. A suit has no trigger either, so the
  performer opens every window with the space bar alone.

Everything in [Record](#record) below then applies unchanged, and so does the checker:
it reads the recorded `full_body` channel and never asks what produced it.

Dropping the headset means handing the process a different OpenXR runtime:
`--no-launch-cloudxr-runtime` leaves `XR_RUNTIME_JSON` alone, so a system runtime such
as Monado serves the session instead. It is documented under *Use a system OpenXR
runtime* in [`cloudxr.rst`](../../../../../docs/source/references/cloudxr.rst); this
capture flow has not been validated with one.

**No full-body plugin ships in this tree.** `install/plugins/` holds controller, pedal and
leader-arm plugins, and the registry has the one vendor. Write the backend and its
plugin first —
[`docs/source/device/trackers.rst`](../../../../../docs/source/device/trackers.rst) under *Vendor
Selection*.

## Record

```bash
src/python/isaaccapture/cts/full_body/capture/record.sh pico4u
```

The argument names the device and only decides where the files land; `pico4u` is the
default.

Open the panel at <http://localhost:8081>. It binds every interface, so the headset's
own browser can reach it at `http://<your-host>:8081` as well.

The cues sit in a column down the right edge and the skeleton gets the room to the left
of it. Drag the column's inner edge to change the split. If the text is too small to
read from where the performer stands, zoom the browser in with `Ctrl+=` — it scales the
text and the column together, and the browser remembers the setting for that address.

The panel idles. Nothing is open against the device and no file exists yet. Press
**Start recording** when the performer is standing ready — that is when the recording is
created. The script itself begins at the first frame with valid joints, which is normally
the next moment but is not the same one.

If ten seconds go by with no valid joint, a red box appears at the bottom left of the
skeleton area listing what to check. It stops nothing: the take is still running, the
file is still being written, and the script still starts at the first valid frame
whenever it arrives. The box clears itself when it does.

Then, for each of the ten steps:

1. The cue is spoken and shown on the panel — *"T pose. Hold."*
2. The performer gets into the pose. Take as long as you need; nothing is running out.
3. **Press the controller trigger, or the space bar.**
4. A beep. Hold the pose while the countdown on the panel runs down.
5. A tick. That pose is recorded, and the next cue follows.

Two things to know before your first take:

- **The space bar is a key in the _browser_, not in the terminal.** It works in whichever
  browser has the panel open and focused — the headset's or the one on your desk. With
  two people, the one at the screen does the pressing.
- **Pressing at any other moment does nothing, deliberately, and the panel will not
  react.** It is not broken and the press is not queued. Press once, when the performer
  is in the pose.

If the performer already knows the pose, they can press during the spoken cue to cut it
short and go straight to waiting.

After the tenth pose the recording closes, *"Done. You can stop now."* plays, and the
panel reports what the labels came out as. The first rows are about the file — record
count, timestamps, whether all ten windows resolved. The rest are named after the steps
and re-derive that pose from the recording, so a `BAD` means the window does not hold
the pose it claims. How many are `BAD` says whose fault it is: one against otherwise
good ones is that step, pressed at the wrong moment or performed wrongly; most of them
at once, with degenerate numbers like `pelvis drops 0 cm`, is the joint stream, and
recording again reproduces it exactly.

The process keeps serving the report; Ctrl+C when you have read it.

## What you get

One directory per take, under `~/isaaccapture-captures/`, named for the device and the
moment you pressed the button:

```text
pico4u_2026-03-04_101530/pico4u_2026-03-04_101530-body.mcap          the recording
                        /pico4u_2026-03-04_101530-body.labels.json   the motion-step windows
                        /pico4u_2026-03-04_101530-body.json          what produced it
                        /pico4u_2026-03-04_101530-body.log           the panel's output
```

Nothing is ever overwritten. Run `record.sh` again for another take and both are kept.

The four files travel together, which is why they share a directory and repeat its name:
sending a take is sending the directory, and the recording still says what it is once it
is out of there. The labels **cannot** be regenerated afterwards.

Do not create a `latest` shortcut. The checker looks for
the labels at `<recording>.labels.json` **as you spelled the recording**, so a symlink
finds none beside itself and can turn a `retake` into a `pass` with the posture checks
silently unanswered. Give the full path.

Then run the checks — [`checker/README.md`](checker/README.md) covers reading the
verdict and packaging a take to send.

## If something goes wrong

| What you see | What to do |
|---|---|
| `Command 'python' not found` | Use the interpreter the setup built, `src/python/isaaccapture/cts/full_body/capture/.venv/bin/python`. Ubuntu has `python3` and no `python`, and neither has `isaaccapture`. |
| `missing …/.venv; run …/setup_env.sh` | Run the two setup scripts above. |
| `no isaaccapture wheel in …/wheels; build the repo first` | Build the project, then re-run `capture/setup_env.sh`. |
| `run …/checker/setup_env.sh first` | You ran the two setup scripts in the wrong order. |
| **the session did not open**, on the panel | CloudXR could not start, or a `--plugin` failed to load. The message names the reason. Nothing was recorded; fix it and run `record.sh` again. |
| **Start recording** does nothing and the terminal repeats `waiting for a system` | No headset is connected to the runtime, so there is no OpenXR system yet. Connect it as above and the take goes on by itself. Ctrl+C is not delivered during that wait — close the terminal if you need out. |
| The client page loads but CONNECT does nothing | The headset has not accepted the runtime's self-signed certificate. Open `https://<this-host>:48322/` in the headset browser, click through the warning, then go back to the client. Failing that: same subnet, and the host firewall allows 48322. |
| The service prints `running` but a second one will not start | The runtime is a host singleton on 48322. One is already up — `service status` names it, `service stop` ends a detached one, Ctrl+C a foreground one. |
| **no joint has been valid yet**, in the cue column | On PICO this is usually the browser, not the hardware or a licence: use the headset's own browser, which grants WebXR body tracking on a consumer 4 Ultra. |
| **No body data after 10 s**, in red under the skeleton | The same thing as the row above, once it has lasted ten seconds, with the causes spelled out. Two red things, one problem. Nothing has been lost: no joint has been valid since you pressed **Start recording**, so the script has not begun. Fix what the box names and the script starts on its own — same take, same file, nothing to redo. |
| No sound, or a `FileNotFoundError` naming `aplay` | `aplay` is missing. Install `alsa-utils` and record again. |
| `… is a Git LFS pointer, not audio; run git lfs pull` | The clone was made without git-lfs, so the cue WAVs are placeholders. Install git-lfs and run `git lfs pull` in the repository. |
| The panel is not at 8081 | Something else held that port, so it moved to the next free one. Use the `http://localhost:…` line the run prints, which is always the real one. |
| The headset's browser cannot load the panel | The port is not reachable from the headset. Check the host firewall; the panel itself listens on every interface. |
| One `BAD` row in the label report, the rest `ok` | That pose was pressed at the wrong moment or performed wrongly. Record another take. |
| Most rows `BAD`, with degenerate numbers (`pelvis drops 0 cm`, both hands at one height) | The joint stream is frozen: every window holds the same still pose, and one row can still read `ok` by accident. Re-performing the script changes nothing; the device or plugin is not delivering changing joints. The checker confirms it with `continuity.max_joint_velocity` at `peak 0.0 m/s`. |

## Do not edit the script

The ten steps, their spoken cues and their hold durations are part of the CTS
specification, and the checker looks its measurements up by step name. Changing any of
them under `capture/` silently changes what the verdict means.

If a take stops with

```text
the wording of cue 'squat_x2' changed, so squat_x2.wav no longer says it
```

then the capture side is out of sync with its own audio. Report it rather than working
around it.
