<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OGLO + MetaQuest Data-Collection Demo

End-to-end demo on a Linux host:

- **MetaQuest** connects over the same WiFi via **CloudXR** → streams hand + head pose.
- **OGLO gloves** (2) connect over **BLE** → 80 taxels + 6-axis IMU per hand at 100 Hz.
- One **time-synced MCAP** records Quest hand/head + both gloves.
- A live **tactile heatmap** is composited into the operator's Quest view (TeleViz).

```
 MetaQuest ──CloudXR/WiFi──► laptop OpenXR runtime ──► HandTracker/HeadTracker ─┐
 OGLO L/R  ──BLE──► oglo_tactile plugins ──SchemaPusher──► OgloTactileTracker ──┤
                                                                                ├─► MCAP (synced)
   oglo_heatmap ──► QuadLayer ──► TeleViz compositor ──CloudXR──► Quest screen ─┘
```

`oglo_teleop_record.py` orchestrates everything (launches the plugins, runs the
session, records, and draws the overlay). `oglo_heatmap.py` is the renderer.

---

## A. One-time setup

1. **System deps**

   ```bash
   sudo apt install libdbus-1-dev          # BlueZ for the BLE plugin
   sudo apt install pkg-config glslang-tools libvulkan-dev   # BUILD_VIZ=ON below
   uv pip install -e ./examples/oglo_tactile        # heatmap renderer + deps
   uv pip install -e './examples/oglo_tactile[gpu]'  # adds the CUDA overlay upload
   ```
   The `gpu` extra is only needed for the in-headset overlay; recording works
   without it. Match your CUDA major version if you edit the pin.

2. **Build** (plugin + viz + python wheel)

   ```bash
   cd IsaacTeleop
   cmake -B build -DBUILD_PLUGIN_OGLO=ON -DBUILD_VIZ=ON
   cmake --build build --parallel
   cmake --install build           # builds the isaaccapture python wheel/package
   ctest --test-dir build -R oglo_packet_parser --output-on-failure   # parser sanity
   ```
   The BLE backend (BlueZ via libdbus, permissive) connects out of the box.

3. **Gloves**: confirm both run schema-5 firmware, are charged, and
   advertise `OGLO LEFT` / `OGLO RIGHT` (check with `bluetoothctl scan on`).

4. **CloudXR**: install per the IsaacTeleop Quick Start (`pip install 'isaaccapture[cloudxr]'`).

---

## B. Demo scenario (run in order)

**Start the CloudXR runtime** — a background service, so this shell stays free
```bash
cd ~/Documents/IsaacTeleop
source scripts/setup_cloudxr_env.sh
python -m isaaccapture.cloudxr.service start --accept-eula   # flag needed on first run only
# Note the printed web-client URL, e.g.  https://<laptop-ip>:48322/
```

**MetaQuest — connect** (same WiFi)
- Open the headset browser → the CloudXR web client at
  `https://nvidia.github.io/IsaacCapture/client` → enter the laptop IP → accept the
  self-signed certificate → **CONNECT**. (Or run the server with `--host-client`
  and open `https://<laptop-ip>:48322/client/` instead.)
- You should now be in the CloudXR view; hand/head tracking is live.

**Gloves + recording + overlay**
```bash
cd ~/Documents/IsaacTeleop
source scripts/setup_cloudxr_env.sh
source ~/.cloudxr/run/cloudxr.env          # path printed by step above
python -m isaaccapture_examples.oglo_tactile.oglo_teleop_record
```
The script: connects both gloves over BLE, starts recording to
`oglo_teleop_<timestamp>.mcap`, and shows the **two-hand tactile heatmap** in the
headset. **Keep hands relaxed for ~1 s** at start so the baseline tares.

Press a glove → the matching finger/taxel lights up (YlOrRd) on the Quest screen.

**Stop**: `Ctrl+C` the recording script (flushes + closes the MCAP), then
`python -m isaaccapture.cloudxr.service stop`.

---

## C. Verify the recording

```bash
mcap info oglo_teleop_*.mcap
# channels: hands/{left,right}_hand, head/head, oglo_{left,right}/oglo(+_tracked)
```
All streams share `sample_time_local_common_clock`, so Quest pose and OGLO
tactile align in time.

---

## Useful flags (`oglo_teleop_record.py`)

| Flag | Purpose |
|------|---------|
| `--plugin-bin PATH` | Path to `oglo_tactile_plugin` (auto-detected under `build/`). |
| `--mcap PATH` | Output file (default timestamped). |
| `--duration S` | Auto-stop after S seconds (0 = until Ctrl+C). |
| `--raw` | Raw-ADC heatmap (no baseline subtraction). |
| `--no-overlay` | Record only (skip headset heatmap). |
| `--overlay-fullscreen` | Fullscreen overlay (facing-safe fallback). |
| `--panel-dist / --panel-right / --panel-drop / --panel-w` | Head-locked HUD placement/size (meters); default is a small bottom-right panel. |
| `--no-headlock` (`--panel-y / --panel-z`) | Use a fixed stage-space panel instead of head-locked. |
| `--plugin-stagger S` | Delay between launching the two gloves (avoids BLE scan contention; default 8 s). |
| `--no-plugins` | Don't auto-launch plugins (you start them yourself). |

---

## Troubleshooting

- **Glove not found**: `bluetoothctl scan on` should list `OGLO LEFT/RIGHT`. Power-cycle the glove; ensure it isn't already connected elsewhere.
- **Quest view but no heatmap**: `cupy` missing/mismatched CUDA → see warning in Terminal 2; install the `cupy` build matching your CUDA. Panel off-view → adjust `--panel-right/--panel-drop/--panel-dist`, or use `--overlay-fullscreen`.
- **One hand not connecting**: usually BLE scan contention — increase `--plugin-stagger` (e.g. 12). Confirm the glove advertises via `bluetoothctl scan on`.
- **No tactile in MCAP**: confirm the plugins connected (Terminal 2 logs `Connected: side=… schema_ver:5`) and that the tracker `collection_id` (`oglo/left`,`oglo/right`) matches the plugin `--collection-prefix=oglo`.
- **Quest won't connect**: same WiFi/subnet, firewall allows the CloudXR ports, accept the self-signed cert in the headset browser.
- **Stutter**: the overlay is auxiliary — drop it with `--no-overlay` to confirm recording is unaffected; lower overlay rate if needed.
