<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Recording a hand take

> **Preview.** The hand CTS tolerances are provisional: they have not yet been
> calibrated against glove recordings, so treat a result close to a limit as indicative.

The hand CTS checks that a hand-tracking integration — a glove plugin such as Manus or
Wuji, or a headset's own optical hands — delivers `core.HandPoseRecord` data a
retargeter can use: 26 OpenXR joints per hand (a glove may omit PALM and the four
non-thumb metacarpals, giving 21), in metres, in the OpenXR joint frame, with the
fingers where their names say and placed where the hands really are.

One take is one recording of a prescribed eleven-step hand script, plus the label
sidecar that [the checker](checker/README.md) reads. A panel in your browser speaks
each step to the performer and counts them in; nothing is pressed during the take. The
whole script takes about two minutes.

This page goes from a fresh clone to a recorded take, then to a verdict and a visual
replay of it.

Recording needs Linux. The checker also runs on macOS, so a take recorded on Linux can
be checked and viewed there.

Nothing under [`capture/`](capture) is meant to be edited — see
[Do not edit the script](#do-not-edit-the-script).

## Before you start

- **The project built**, so that an `isaaccapture` wheel exists under `install/wheels/`
  or `build/wheels/`.
- **`uv`, `curl` and `unzip`** on the same host.
- **`git-lfs`**, so the clone holds the spoken cues as audio rather than LFS pointers.
- **`aplay`** — the spoken cues and tones go through it. It comes from `alsa-utils`
  (`apt install alsa-utils` on Debian or Ubuntu). Install it before your first take: a
  missing `aplay` stops the take part way through rather than at startup.
- **A headset**, plus CloudXR installed (by default at `~/.cloudxr`) — see
  [Get the headset streaming](#get-the-headset-streaming). The headset has to be
  connected even when a glove produces the joints, because the OpenXR session comes
  from CloudXR.
- **For a glove, its plugin built and installed** into `install/plugins/`, for example
  `cmake --install build --component manus` or `--component wuji_glove`. The
  device pages cover building each: [`manus.rst`](../../../../../docs/source/device/manus.rst)
  and [`wuji_glove.rst`](../../../../../docs/source/device/wuji_glove.rst).
- **A table and a chair.** The performer sits at the table; several steps rest the
  hands flat on it. Two people is easier than one — one performs while the other watches
  the panel.

## Set up, once per clone

Two scripts, in this order. The checker's comes first because it generates the bindings
the capture panel reads the recording back with.

```bash
src/python/isaaccapture/cts/hand/checker/setup_env.sh
src/python/isaaccapture/cts/hand/capture/setup_env.sh
```

To view takes visually afterwards (see [View the take](#view-the-take)), add the viewer
to the checker's environment as well:

```bash
src/python/isaaccapture/cts/hand/checker/setup_env.sh --panel
```

All of these are idempotent and write nothing outside their own directory. Re-run them
after pulling.

## Get the headset streaming

**Every take needs this, including one whose joints come from a glove.** CloudXR is the
OpenXR runtime the capture runs on, and it has no system to hand out until a headset
has loaded the web client and pressed CONNECT. Skipping it does not raise:
**Start recording** blocks inside session creation, logging `OpenXR HMD form factor is
unavailable; waiting for a system...` once a second, and Python does not take Ctrl+C
there.

The full walkthrough, with the reasons, is in the full-body guide's
[Get the headset streaming](../full_body/README.md#get-the-headset-streaming). The
commands are the same here, under the hand capture's interpreter:

```bash
PY=src/python/isaaccapture/cts/hand/capture/.venv/bin/python
```

1. **Accept the CloudXR EULA, once per machine.** Review the licence, then
   `$PY -m isaaccapture.cloudxr.service run --accept-eula`. (Passing `--accept-eula` to
   `record.sh` does the same.)
2. **Start the service in a terminal of its own and leave it open:**
   `$PY -m isaaccapture.cloudxr.service run`. It prints the runtime's log, which is where a
   headset that will not connect says why. The runtime is a host singleton on port
   48322, so run one copy only. `record.sh` starts its own when nothing is up, but a
   runtime takes tens of seconds to come up and a separate one outlives any number of
   takes.
3. **Connect the headset over Wi-Fi**, on the same subnet as this host:
   `$PY -m isaaccapture.cloudxr.webclient --print-only`. It prints the streaming target
   `https://<this-host>:48322/` and the client URL. **Open the streaming target in the
   headset's browser first and accept the self-signed certificate**, then open the
   client URL and press CONNECT. Without `--print-only` the command types the URL into
   the headset over USB `adb` instead.

`record.sh` passes anything it does not understand through to the capture panel, so the
runtime flags work from there too: `--cloudxr-install-dir` (default `~/.cloudxr`),
`--cloudxr-device-profile` (default `Quest3`), and the rest of
`capture/capture_panel.py --help`.

## Record

The first argument names the device. It only decides where the files land, so use
something that identifies the hardware: `quest3`, `pico4u`, `manus`, `wuji`.

**The headset's own optical hands** — no plugin:

```bash
src/python/isaaccapture/cts/hand/capture/record.sh quest3
```

**Manus gloves:**

```bash
src/python/isaaccapture/cts/hand/capture/record.sh manus --plugin manus_hand_plugin
```

**Wuji gloves.** The glove stream carries no absolute wrist pose, so the plugin places
it from the headset's hand tracking or from a controller. `WUJI_GLOVE_WRIST_SOURCE` picks
which: `auto` (the default), `hand_tracking` or `controller`.

```bash
WUJI_GLOVE_WRIST_SOURCE=hand_tracking \
  src/python/isaaccapture/cts/hand/capture/record.sh wuji --plugin wuji_glove_plugin
```

The variable is read by the plugin and also written into the take's provenance file.
`--wrist-source TEXT` only changes what is written there; it does not change what the
plugin does.

`--plugin NAME` takes the plugin's name as its `plugin.yaml` spells it and looks it up
under `plugins/` and `install/plugins/`. It is launched as required: a plugin that
fails to load stops the take rather than recording a file with no hands in it. Other
options kept with the take: `--plugin-version NAME=VERSION` (otherwise read from
`plugin.yaml`) and `--note TEXT`.

Open the panel at <http://localhost:8081>. It listens on every interface, so the
headset's own browser can reach it at `http://<your-host>:8081` too. If the port is
taken the panel moves to the next free one; use the `http://localhost:…` line the run
prints.

The panel idles with the CloudXR runtime state, the hand source and where the take will
be written. Nothing is open against the device and no file exists yet. Press
**Start recording**. That creates the recording; the script itself begins once both
hands have had a valid joint.

If ten seconds go by without that, a red box titled **No hand data after 10 s** lists
what to check. It stops nothing: the take is running, and the script starts on its own
when both hands arrive.

### The eleven steps

Both hands perform every step together, except the two rolls. For each step the panel
shows the step name, the spoken text and a large number, and the same things happen
every time:

1. The cue is spoken. The panel says **listen**.
2. Three short tones, one second apart, count in. The panel says **get into position**
   and counts 3, 2, 1. The performer moves into the pose during the cue and the count.
3. A beep opens the step. The panel says **hold** and counts the seconds down. Hold the
   pose still.
4. A tick closes it, and the next cue follows.

Nothing is pressed at any point: no trigger, no key. A timer paces the script.

| # | Step | The performer | Held for |
|---|---|---|---|
| 1 | `flat_on_table_open` | Both hands flat on the table, palms down, fingers together | 4 s |
| 2 | `fist` | Two tight fists | 4 s |
| 3 | `pinch_index` | Thumb to index fingertip, both hands | 3 s |
| 4 | `pinch_middle` | Thumb to middle fingertip | 3 s |
| 5 | `pinch_ring` | Thumb to ring fingertip | 3 s |
| 6 | `pinch_little` | Thumb to little fingertip | 3 s |
| 7 | `palms_together` | Palms pressed together, fingers up | 4 s |
| 8 | `tips_together` | Both hands raised, palms facing you, middle fingertips touching | 3 s |
| 9 | `right_tip_roll` | Fingertips touching. At the beep, turn the right palm out, then back | 6 s |
| 10 | `left_tip_roll` | Fingertips touching. At the beep, turn the left palm out, then back | 6 s |
| 11 | `flat_on_table_close` | Both hands flat on the table again | 4 s |

Three things to know before the first take:

- **Be in the pose by the beep and keep still.** The checker ignores the first 40% of
  each held step, so a slightly slow arrival is fine; a pose that is not held is
  reported as a missed step and the take is asked to be repeated, not blamed on the
  device. The two roll steps are measured from beep to tick, so start the turn at the
  beep.
- **The table, the pressed palms and the touching fingertips are what the data is held
  to.** Press and touch for real: fingers flat on the table, palms against each other,
  middle fingertips meeting. Do not hover.
- **The performer cannot pause or skip.** If a step goes wrong, let the script finish
  and record again.

After the eleventh step *"Done. You can stop now."* plays and the recording closes. The
panel then writes the label sidecar and audits it against the finished file, listing
rows marked `ok` or `BAD`:

- **`hand channels`** — both hands produced records. `BAD` means one hand never arrived.
- **`segmentation.label_alignment`** — every step window is backed by frames from the
  recording.
- **`segmentation.labelled_step_actually_performed`** — each window shows the motion
  its name asks for. Its detail names the steps that were not performed.

A `BAD` on one step with the rest `ok` means that step was missed: record another take.
`BAD` on all of them, or on `hand channels`, points at the data stream rather than the
performer. The same lines are printed in the terminal and saved to the take's `.log`.

The process keeps serving the report; Ctrl+C when you have read it.

The grid in the 3D view starts on the stage floor and moves onto the table once the
first flat step has found it.

## What you get

One directory per take, under `~/isaaccapture-captures/`, named for the device and the
moment you started the script:

```text
manus_2026-03-04_101530/manus_2026-03-04_101530-hand.mcap          the recording
                       /manus_2026-03-04_101530-hand.labels.json   one window per step
                       /manus_2026-03-04_101530-hand.json          what produced it
                       /manus_2026-03-04_101530-hand.log           the panel's output
```

The `.json` records the device, hand source, plugin versions, wrist source, the pacing
and the repository commit. Nothing is ever overwritten; run `record.sh` again for
another take and both are kept.

The four files travel together, which is why they share a directory and repeat its name.
The label windows are stamped while the take runs and **cannot be regenerated** from the
recording afterwards.

Do not rename the files, and do not point the checker at a symlink or a copy. The
checker looks for the labels at `<recording>.labels.json` **as you spelled the
recording**, so a link finds none beside itself and the window checks go unanswered.
Always give the full path.

## Check the take

```bash
src/python/isaaccapture/cts/hand/checker/.venv/bin/python -m hand_cts.cli \
    ~/isaaccapture-captures/<take>/<take>-hand.mcap
```

This needs only the checker's environment, not the capture one. It prints one verdict and
one line per check, and exits with a status that says which verdict it was:

| Verdict | Exit status | Meaning |
|---|---|---|
| `pass` | 0 | Nothing in the recording argues against the integration. |
| `fail` | 1 | A failure attributable to the device or its plugin. |
| `retake` | 2 | The capture is unusable because of how it was performed — a step was not held. The device is not implicated. Record again. |
| `insufficient_data` | 3 | The recording cannot answer the question: too short, no hand channel, or a prerequisite check could not run. |

Each line in the report starts with a mark:

| Mark | What it means |
|---|---|
| `pass` | The check was answered and the answer is acceptable. |
| `FAIL` | The check was answered and the answer is not. Whether that makes the verdict `fail` or `retake` depends on whom the check blames. |
| `meas` | A number was measured and nobody judged it. Read the value; it is not an approval. |
| `n/a ` | Not answered: not enough data, or a check it depends on failed, so the measurement would describe the wrong frames. It never counts as a pass. |
| `note` | An advisory check failed. It is reported and does not move the verdict. |

Most checks report the left and right hands in one line, `left: …; right: …`.

**A take with no labels beside it gets a note and a verdict from the other checks only.**
If the report says `no motion-step labels beside the recording`, the sidecar was not
found: check the path and the file name before trusting a `pass`.

[`checker/README.md`](checker/README.md) lists every check and how to read the rest of
the report.

## View the take

```bash
src/python/isaaccapture/cts/hand/checker/.venv/bin/python -m hand_cts.panel \
    ~/isaaccapture-captures/<take>/<take>-hand.mcap
```

This needs the `--panel` setup above. It prints the same report, then serves a viewer
at <http://127.0.0.1:8080>. Add `--host 0.0.0.0` to reach it from another machine, and
`--port` to move it. It shows:

- **Both hands on one playhead**, each digit in its own colour (the legend is in the
  sidebar). A joint whose `is_valid` went false is drawn **red where it was last seen**,
  because its recorded position is arbitrary. Tick *name held joints* to label them.
- **The labelled step under the playhead**, or `unlabelled` between steps.
- **Frame rate over the last 8 seconds and valid joints over the whole take**, per hand,
  and the count of valid joints (of 26) at the playhead.
- **A grid at the table height** the flat steps measured, or on the stage floor when no
  flat step was held.
- **The results**, with the verdict banner on top: the checks that decide the take first,
  then four groups — schema, envelope and signal quality; hand shape; placement and
  frames; poses over the labelled steps. Each group lists its worst result first and
  opens by default when it is not all `pass`.

`play` runs the take at its own rate; `speed` (0.25x to 2x) and the `time s` slider move
the playhead. The result list always shows the **final** result of every check: several
need the whole recording, so a result taken mid-playback would mean nothing.

## Send a take

Send the **whole take directory**: the recording, its labels, the provenance file and the
log. There is no packaging step; compress the directory however you like
(`tar czf <take>.tar.gz <take>`). Without the labels the window checks cannot be
answered.

## If something goes wrong

| What you see | What to do |
|---|---|
| `usage: …/record.sh DEVICE [capture_panel.py options]` | The device name is required and goes first. |
| `missing …/.venv; run …/setup_env.sh` | Run the two setup scripts above. |
| `no isaaccapture wheel in …/{install,build}/wheels; build the repo first` | Build the project, then re-run `capture/setup_env.sh`. |
| `run …/checker/setup_env.sh first; the panel decodes the take it writes` | The two setup scripts were run in the wrong order. |
| `--plugin given but no plugins/ or install/plugins/ in the repo` | Build and install the glove plugin first. |
| **the session did not open**, on the panel | CloudXR could not start, or a `--plugin` failed to load. The message names the reason. Nothing was recorded; fix it and run `record.sh` again. |
| **Start recording** does nothing and the terminal repeats `waiting for a system` | No headset is connected to the runtime. Connect it as above and the take goes on by itself. Ctrl+C is not delivered during the wait; close the terminal if you need out. |
| The client page loads but CONNECT does nothing | The headset has not accepted the runtime's self-signed certificate. Open `https://<this-host>:48322/` in the headset browser, click through the warning, then go back to the client. Also check: same subnet, and the host firewall allows 48322. |
| A second CloudXR service will not start | The runtime is a host singleton on 48322. One is already up; `service status` names it, `service stop` ends a detached one, Ctrl+C a foreground one. |
| **No hand data after 10 s**, in red | With a plugin: it is not running or exited; the glove is off, unpaired, uncalibrated, or its vendor service is down; or the plugin needs a wrist from the headset or a controller and neither is tracked. Without one: the headset is not streaming, hand tracking is off in the headset, or the hands are out of view. The take is still running and starts on its own once both hands are valid. |
| One hand shows `inactive`, or `n / 26 valid` in red, in the sidebar | That hand's channel is missing or has invalid joints right now. The sidebar shows it live; the script waits for both hands to have had a valid joint, not for all 26. |
| No sound, or a `FileNotFoundError` naming `aplay` | `aplay` is missing. Install `alsa-utils` and record again. |
| `… is a Git LFS pointer, not audio; run git lfs pull` | The clone was made without git-lfs, so the cue WAVs are placeholders. Install git-lfs and run `git lfs pull` in the repository. |
| The headset's browser cannot load the panel | The port is not reachable from the headset. Check the host firewall; the panel listens on every interface. |
| One `BAD` row in the label audit, the rest `ok` | That pose was not held. Record another take. |
| `no sidecar beside the recording` from `make_labels.py` | There is no `.labels.json` next to that recording, or it was renamed. |
| The report says `no motion-step labels beside the recording` | The checker could not find `<recording>.labels.json` at the path you gave. Check the spelling and that you did not go through a symlink. |
| `hand_cts.panel` exits with `The panel's renderer is an optional extra` | Run `checker/setup_env.sh --panel`. |

To run the label audit again on an existing take:

```bash
src/python/isaaccapture/cts/hand/checker/.venv/bin/python \
    src/python/isaaccapture/cts/hand/capture/make_labels.py \
    ~/isaaccapture-captures/<take>/<take>-hand.mcap
```

## Do not edit the script

The eleven steps, their spoken cues and their hold durations are part of the CTS
specification, and the checker looks its measurements up by step name. Changing any of
them under `capture/` silently changes what the verdict means.

The cue audio is checked against the script's wording. If the panel exits at start-up
with

```text
the wording of cue 'fist' changed, so fist.wav no longer says it
```

then the script and its audio disagree: restore `capture/session.py` rather than working
around it.

## Directories

| Directory | What it is |
|---|---|
| [`capture/`](capture/README.md) | records one scripted take and its label sidecar |
| [`checker/`](checker/README.md) | reads a take and returns a verdict; needs `mcap` and `flatbuffers` only |
| [`oracle/`](oracle/README.md) | generates synthetic recordings with known defects, for testing the checker itself |
