<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Xsens Technologies B.V. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Xsens Full Body — User Guide

Stream a person wearing an **Xsens MVN** suit into **Isaac Teleop**, so any Isaac Teleop
application can read their full-body pose live.

This guide is the "how do I run it" version. The developer-facing notes — wire format, design
rationale, tests — live in [`README.md`](README.md) next to this file.

---

## 1. What the plugin does

```
  Xsens suit              MVN Studio                  xsens_full_body_plugin        your app
 ┌──────────┐   radio   ┌───────────────┐   UDP     ┌──────────────────────┐      ┌──────────┐
 │  sensors │ ────────> │ solves the    │ ────────> │ receives the stream  │ ───> │ reads    │
 │          │           │ skeleton      │  :9764    │ publishes it to      │      │ the pose │
 └──────────┘           └───────────────┘           │ Isaac Teleop         │      └──────────┘
                                                    └──────────────────────┘
```

MVN Studio converts its own skeleton into the standard 24-joint
Isaac Teleop body layout and sends it over the network. The plugin is the piece in the
middle — it listens on a UDP port and republishes what arrives as an Isaac Teleop **tensor
collection** named `xsens_full_body`.

Applications read it through the **`body.xsens`** vendor, exactly like they would read a headset
or any other body-tracking device. Nothing downstream has to know Xsens is involved.

**Linux only.** On other platforms the build skips the plugin.
---

## 2. What you need

| | |
|---|---|
| An Xsens MVN suit and **MVN Studio**, live or replaying a recording | the source of the motion |
| **Isaac Teleop**, built or installed | provides the plugin and the reader |
| A running **CloudXR runtime** | the transport the collection lives in |
| Linux (x86_64 or aarch64) | |

You do **not** need an XR headset.

---

## 3. Get the plugin binary

If you built Isaac Teleop from source (see the [source-build prerequisites](https://nvidia.github.io/IsaacCapture/main/development/build_from_source/index.html)),
the binary is already there:

```bash
cmake -B build
cmake --build build --target xsens_full_body_plugin --parallel
# -> build/src/plugins/xsens_full_body/xsens_full_body_plugin
```

`cmake --install build` places it at `<prefix>/plugins/xsens_full_body/xsens_full_body_plugin`,
next to its `plugin.yaml`.

---

## 4. Run it — five steps

### Step 1 — Start the CloudXR runtime

Once per machine. Everything else needs its environment.

```bash
python -m isaaccapture.cloudxr.service start
```

### Step 2 — Load the CloudXR environment

In **every** terminal that runs the plugin or an application reading from it:

```bash
source ~/.cloudxr/run/cloudxr.env
```

Skipping this is the single most common setup mistake.

### Step 3 — Start the plugin

```bash
./xsens_full_body_plugin
```

All defaults match MVN Studio's "Isaac Teleop" preset, so the bare command is usually right.
Add `--address=127.0.0.1` if you want to accept the suit stream only from this machine.

You should see:

```
Xsens Full Body Pusher (collection: xsens_full_body, tensor: full_body_pose, udp: 0.0.0.0:9764, max_flatbuffer_size: 4096)
listening on 0.0.0.0:9764 -> push_buffer
In MVN Studio: Options -> Network Streamer -> preset "Isaac Teleop", tick the row.
```

> **Start the plugin before your application.** The plugin is what *creates* the collection, so an
> application started first will report "no collection found" and never recover on its own.

### Step 4 — Point MVN Studio at it

In MVN Studio:

1. **Options → Network Streamer**
2. Choose the preset **"Isaac Teleop"** (destination `127.0.0.1:9764`, UDP — change the host if
   the programs are run on seperate machines).
3. **Tick** the destination row to enable it.
4. Move in the suit, or press **Play** on a recording.

### Step 5 — Check that frames are flowing

The plugin prints a status line every 250 delivered frames:

```
[XsensFullBody] delivered=250 seq=251 size=820 fnv1a64=0x... malformed=0 unverified=0 ...
```

`delivered` climbing means the pipeline is alive. If it never appears, jump to
[Troubleshooting](#7-troubleshooting).

---

## 5. Reading the data in your application

Select the `body.xsens` vendor on a full-body source. The session wires up everything else —
the required OpenXR extensions included.

```python
from isaaccapture import deviceio
from isaaccapture.retargeting_engine.deviceio_source_nodes import FullBodySource

full_body = FullBodySource(
    name="full_body",
    vendor=deviceio.TrackerVendor(
        "body.xsens",
        {
            "collection_id": "xsens_full_body",   # must match the plugin's --collection-id
            "max_flatbuffer_size": "4096",        # must match the plugin's --max-flatbuffer-size
        },
    ),
)
```

Both parameters are optional and default to the values shown. They are also the only two the
vendor accepts — anything else is rejected at startup.

> Without an explicit vendor, a full-body tracker defaults to `body.pico-xr` (headset body
> tracking) and will never see the Xsens stream.

### What the data looks like

- **24 joints**, the standard `XR_BD_body_tracking` layout: pelvis, hips, spine, knees, ankles,
  feet, neck, collars, head, shoulders, elbows, wrists, hands.
- Each joint carries a position, an orientation, and its own **validity flag**.
- Units are **meters**, the up axis is **Y**, quaternions are **xyzw**, and poses are **absolute
  and global** (not relative to a parent).

### Why you see 22 of 24 joints

MVN does not track fingers. `LEFT_HAND` (22) and `RIGHT_HAND` (23) therefore carry a copy of the
wrist pose flagged **invalid**, and the "all joints tracked" flag is always false as a result.

**22/24 is the correct, healthy state — not a fault.** Always consult the per-joint validity
flags rather than the all-joints flag.

---

## 6. Command-line reference

| Option | Default | What it does |
|---|---|---|
| `--collection-id=ID` | `xsens_full_body` | The name your application looks the stream up by. Both sides must use the same string. |
| `--address=ADDR` | `0.0.0.0` | Which network interface accepts the suit stream. `0.0.0.0` is any; `127.0.0.1` confines it to this machine. Must be a literal IPv4 address — hostnames are rejected. Launched through the plugin manager the default is `127.0.0.1` instead, because `plugin.yaml` passes it — a suit on another machine needs that line changed. |
| `--port=N` | `9764` | UDP port to listen on. Must match MVN Studio's destination port. |
| `--max-flatbuffer-size=N` | `4096` | Maximum frame size in bytes. Must match the value your application passes. |
| `--help` | | Print the same summary. |

Stop the plugin with **Ctrl-C**; it prints a final status line on the way out.

Every option is a flag. A bare word on the command line is rejected rather than interpreted, so a
mistyped flag cannot quietly become a value.

---

## 7. Troubleshooting

### My application sees no data, and nothing reports an error

This is the one failure that is completely silent, and it has three possible causes. Check them in
this order:

1. **Mismatched `collection_id`.** The plugin and the application rendezvous on that exact string.
   Different strings mean they never meet — forever, with no error on either side.
2. **The application started first.** The plugin creates the collection; restart the application
   after the plugin is listening.
3. **Mismatched address.** If you bound the plugin to one interface and MVN Studio sends to a
   different one, the packets are simply never delivered. Try `--address=0.0.0.0`.

> A `max_flatbuffer_size` mismatch behaves the *opposite* way: it throws loudly on both sides and
> names the right value. So when you see nothing at all, suspect the collection id first.

### `delivered` never increases

MVN Studio is not sending, or is not reaching the machine. Confirm the Network Streamer row is
**ticked**, that MVN is playing or the suit is live, and that the host/port matches. If MVN runs on
another machine, check the firewall on UDP 9764 (or selected port).

### The plugin exits immediately at startup

Read the message — startup failures are explicit:

- `is not a literal IPv4 address` — `--address` needs something like `127.0.0.1`, not a hostname.
- `bind to UDP ... failed` — another process already holds the port, often a second copy of the
  plugin.
- Errors mentioning the session or runtime — CloudXR is not running, or
  `source ~/.cloudxr/run/cloudxr.env` was not done in this shell.

### The counters show gaps or resets

See the status-line guide below — most of these are informational, not faults.

---

## 8. Reading the status line

```
delivered=250 seq=251 size=820 fnv1a64=0x2b... malformed=0 unverified=0 stale=0 truncated=0
gapEvents=0 seqSkipped=0 resets=0 resyncs=0 rewinds=0 nonWholeMs=0 socketRecoveries=0
socketRecoveryFailures=0 pushFailures=0 sessionRecoveries=0
```

| Counter | Meaning |
|---|---|
| `delivered` | Frames published. The number that should keep climbing. |
| `seq`, `size`, `fnv1a64` | Identity of the last frame — sequence number, byte size (820 for a live MVN pose), content hash. |
| `malformed`, `unverified`, `stale`, `truncated` | Datagrams rejected: bad framing, failed structure check, out of order, or too large. Should stay at 0. |
| `gapEvents` / `seqSkipped` | How many times the frame sequence broke, and how many frame numbers went missing in total. Both are reported because one long dropout and many single losses are very different faults with the same total. |
| `resets` | MVN started a new session. Normal when you restart a recording. |
| `resyncs` | Of those sessions, the ones recognised only after half a second of stale frames, because the packet announcing the restart was lost. One or two are unremarkable; a number that keeps climbing means the link is dropping datagrams. |
| `rewinds`, `nonWholeMs` | Timestamp oddities — scrubbing a recording, or a clock that isn't MVN's solver. **Purely informational: no frame is ever dropped because of a timestamp.** |
| `socketRecoveries`, `sessionRecoveries` | Times the plugin recovered from a broken network socket or a restarted CloudXR runtime. |
| `pushFailures`, `socketRecoveryFailures` | Failures encountered along the way to those recoveries. |

A gap is not automatically packet loss. Numbers also go missing when MVN itself declines to build a
frame, and during any outage — the plugin is not draining the socket while it retries.

---

## 9. What it handles by itself

You do not have to babysit the plugin through either of these:

| What breaks | What happens |
|---|---|
| The network socket fails | Re-binds the same port, up to 5 times over ~2.3 s |
| The CloudXR runtime restarts | Rebuilds the session and the collection, up to 8 times over ~23.5 s |

If it still can't recover, it prints its counters and exits with status 1 — so the log always says
what led up to the failure. Ctrl-C works during a retry, so you never have to wait out a backoff.

Repetitive log messages are throttled (first occurrence, then every hundredth, tagged `(xN)`), so a
lossy link cannot flood your terminal.

---

## 10. Where to go next

- [`README.md`](README.md) — wire format, frame-handling rules, tests, and the reasoning behind the
  design.
- `plugin.yaml` — the manifest used when the plugin is launched by the Isaac Teleop plugin manager
  rather than by hand.
- Isaac Teleop docs: **Devices → Body Tracking** for the 24-joint layout shared by every body
  tracking source.
