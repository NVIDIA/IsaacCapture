<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SENSING camera setup guide

This guide takes a Jetson AGX Orin and SENSING GMSL cameras from a pile of parts to cameras that stream. It covers the
physical setup, where most problems come from (which cable goes into which connector, and how the carrier board is
powered), and the operating system setup the cameras need. It stops once the cameras are confirmed streaming.

For the plugin's source layout see [`README.md`](README.md). For the published reference documentation see the
[SENSING user guide](https://nvidia.github.io/IsaacCapture/main/device/camera/sensing.html).

## Two kinds of camera

Jetson reads cameras through one of two software stacks, and which one you need depends on how the camera is wired to
the board, not on its brand.

- **SIPL cameras (GMSL)** connect through a deserializer on a carrier board, such as the SENSING SG8A, over a coax
  cable. NVIDIA's Sensor Input Processing Library (SIPL) configures the deserializer and cameras from a platform file.
  The SENSING SHW5G on the SG8A is this kind. **This guide covers SIPL cameras.**
- **Argus cameras (CSI)** are exposed by the Jetson camera driver as devices, typically MIPI CSI sensors described in
  the device tree, and are reached through `libargus` and `nvargus-daemon`. You pick one by index. Nothing in this guide
  applies to them.

| | SIPL | Argus |
| --- | --- | --- |
| Typical cameras | GMSL cameras on a carrier such as the SG8A | CSI cameras exposed as V4L2 or Argus devices |
| Configuration | Platform JSON, a named config, per-link masks | Device tree and sensor drivers |
| Camera selection | Sensor number, which follows the link index | Camera index |
| Rule of thumb | A deserializer and coax cable sit between camera and board | The camera appears as `/dev/video*` |

The same camera model can be reached either way, depending on the board. On the SG8A rig Argus reports no camera
devices (no `/dev/video*`, and "virtual enumerator not found in proc device-tree"), so SIPL is the only option there.
Do not run both stacks against the same cameras.

## Step 1: Gather the parts

| Part | What it is |
| --- | --- |
| Jetson AGX Orin Developer Kit | Runs L4T R39.2.1 (JetPack 7.2.1) and has its own power supply, which this guide does not cover. |
| SENSING SG8A board | The GMSL carrier. On this rig it is installed inside the devkit's enclosure at the front edge. Its two quad receptacles and a white 2-pin power connector face out, beside the devkit's three front-panel buttons and LED. |
| Carrier power supply | A 12 V adapter, plus a black cable with a female DC barrel jack on one end and a white 2-pin connector (red and black wires) on the other. |
| Cameras | SHW5G modules. Each has one coax jack on the rear (the vendor lists an Amphenol FAKRA Z connector). |
| Camera cables | Two 2 m black coax cables (labelled "2米", Chinese for 2 m), one per camera, each with a teal FAKRA connector and grey strain relief at the camera end. |
| Breakout harness | One teal quad plug (four coax sockets in a 2x2 block) with two coax cables leading to two single teal FAKRA jacks. |

This guide assumes the SG8A board is already installed in the devkit. Installing or removing it is covered by
SENSING's hardware manual, not here.

### What the parts look like

![The parts laid out: two cameras on a bracket with their labelled 2 m cables, loose teal connectors, jumper wires with a barrel plug, and the devkit at the lower right](images/setup-parts-overview.jpg)

*The parts laid out. The two cameras sit on a bracket, each with a labelled 2 m cable ending in a teal connector. Loose
teal connectors and a cable with a barrel plug lie nearby, and the devkit is at the lower right.*

![The breakout harness: two single teal FAKRA jacks and one teal quad plug with four coax sockets](images/setup-harness-connectors.jpg)

*The breakout harness. The two single teal FAKRA jacks each take one camera cable. The teal quad plug, with its four
coax sockets in a 2x2 block, goes into the carrier.*

![An SHW5G camera in a mounting bracket, seen from the lens end](images/setup-camera-front.jpg)

*An SHW5G camera in a mounting bracket, seen from the lens end.*

![The rear of a camera: one round coax jack recessed in the bracket](images/setup-camera-rear.jpg)

*The rear of the camera: one round coax jack, recessed in the bracket on this rig. The camera cable's teal connector
mates here.*

### The connectors, end to end

Every connector in the camera chain is teal, the FAKRA code for Z. Each connection mates one specific pair:

| Connection | One side | Other side |
| --- | --- | --- |
| Camera to camera cable | The camera's rear coax jack | The camera cable's teal FAKRA connector with grey strain relief |
| Camera cable to harness | The camera cable's other end | One of the harness's two single teal FAKRA jacks (each a square teal housing with a centre contact; one has a purple tab) |
| Harness to carrier | The harness's teal quad plug (2x2 sockets) | A black latched quad receptacle on the SG8A (four round coax pins in a 2x2 block) |
| Carrier power | The carrier's white 2-pin connector | The black power cable, whose barrel jack takes the 12 V adapter's plug |

The two receptacles are **CN1** (left, carrying COAX0 to COAX3) and **CN2** (right, carrying COAX4 to COAX7), as
labelled in the vendor documentation. The camera chain uses CN1 only. Leave CN2 empty.

Notes on the parts:

- **SHW5G and SHF5G are two models of one camera family.** The vendor lists them together as its "5MP Global"
  monocular GMSL2 cameras. Both have the same 2560x1984 sensor, frame rate (up to 60 fps in global-shutter mode, 50 fps
  in rolling-shutter mode), GMSL2 interface, FAKRA Z connector and power requirements. They differ in lens: the SHW5G
  is the wide-angle model (130° horizontal field of view) and the SHF5G is ultra-wide (200°). This guide and its
  configuration use the SHW5G.
- **The carrier needs its own supply.** Power it from a 12 V adapter, and check the current rating with SENSING before
  choosing one.
- **The cameras need no separate supply.** The vendor wiki lists them as single-coax cameras powered over the same
  cable at 9 to 16 V, drawing less than 200 mA at 12 V each. The carrier provides that power.
- **Nothing on the Orin plugs directly into a camera's round coax jack.** The only connection to the carrier is the
  quad plug on the harness. Spare FAKRA cables lying around the rig are not part of this setup.
- **Leave the jumper wires as they are.** Coloured jumper wires may be plugged into the devkit's 40-pin header. Do not
  add or remove any without checking SENSING's manual.

## Step 2: Connect the cameras and power

**Power everything down first.** The cameras are powered through the coax, so connecting them to a live carrier can
damage the cameras or the carrier's serializers. Shut the Orin down and unplug both the devkit's supply and the
carrier's 12 V supply.

The finished chain looks like this:

```text
12 V adapter ── DC barrel jack ── black cable ── white 2-pin connector ── SG8A board (front edge of the devkit)

SG8A CN1 receptacle (left) ◄── teal quad plug ── harness ─┬─ single teal jack ── 2 m coax cable ── camera
                                                          └─ single teal jack ── 2 m coax cable ── camera
```

1. **Camera cable to camera.** Push each 2 m cable's teal FAKRA connector onto the rear jack of its camera until it
   clicks. Check that it is fully seated.
2. **Camera cable to harness.** Connect the other end of each 2 m cable to one of the harness's two single teal
   jacks, until it clicks. For a single camera, connect one cable and leave the other jack open.
3. **Harness to carrier.** Push the harness's teal quad plug into the **left** receptacle (CN1) on the front edge of the
   devkit. The plug is latched and keyed, so it fits only one way and clicks when seated. If it will not go in, stop;
   do not force it.
4. **Carrier power.** Connect the white 2-pin connector to the carrier, and the black cable's barrel jack to the 12 V
   adapter's plug.
5. **Power on.** Plug in the carrier's 12 V supply first, then power on the devkit. This is the order used on this
   rig. Green LEDs light on the board when it is powered.

![The front edge of the devkit: three buttons and an LED, and the SG8A board's two empty quad receptacles, with the white 2-pin power connector at the left edge](images/setup-carrier-front-panel.jpg)

*The carrier end before connecting. The devkit's front edge has three buttons and an LED. The SG8A board's two quad
receptacles sit beside them, each a latched black housing with four coax pins in a 2x2 block. CN1 is on the left and
CN2 on the right. The white 2-pin power connector, with red and black wires, is at the left edge.*

![The harness's teal quad plug seated in the left receptacle, with two coax cables leaving it; the right receptacle is empty](images/setup-harness-in-cn1.jpg)

*The connected harness. The teal quad plug is seated in the left receptacle (CN1), with two coax cables leaving it. The
right receptacle (CN2) stays empty.*

![A side view of the unit showing the SG8A board with its two quad receptacles, its white 2-pin power connector, and jumper wires on a 40-pin header](images/setup-carrier-inside-devkit.jpg)

*A side view of the unit with its side open. The SG8A board sits at the front, with its quad receptacles at the bottom
and the white 2-pin power connector beside them. The coloured wires on the right are plugged into a 40-pin header.*

With this harness in CN1 the two cameras come up as links 2 and 3 (COAX2 and COAX3, which the vendor documents as
CAM2 and CAM3). The cameras need to be on those contacts; the harness is what puts them there.

| Cameras connected | Link masks | Sensors |
| --- | --- | --- |
| Two | `0x0000 0x1100` | `sensor=0` and `sensor=1` |
| One, on COAX3 | `0x0000 0x1000` | `sensor=1` |

Sensor numbers follow the link index, not a count from zero. To find out which of the harness's two jacks is sensor 0
and which is sensor 1, connect one camera at a time and run the Step 4 check; the sensor that streams belongs to that
jack. Either order works, and swapping the cameras only swaps the sensor numbers.

## Step 3: Set up the operating system

The Orin must be on L4T R39.2.1 (JetPack 7.2.1); check `/etc/nv_tegra_release`. Run the host setup once, choose the
Jetson-IO overlay **Jetson Sensing SG8A-AGON-G2Y-A1 SIPL GMSL2x8** when it asks, and reboot:

```bash
src/plugins/sensing/scripts/setup.sh
```

The script installs the vendor drivers, camera tuning files and device-tree overlay for the carrier. Run it with
`--help` to see its options. After the reboot the carrier is enabled and the cameras can be opened.

## Step 4: Check that the cameras stream

Use NVIDIA's own tool, which needs nothing beyond the OS setup and no `sudo`:

```bash
nvsipl_camera -t src/plugins/sensing/configs/shw5g.json -c SHW5G_2 -m "0x0000 0x1100" \
  --enable-camera-hal -s -Z -r 5
```

Use your own link masks from Step 2. A working camera reports about 60 fps and `Frame drops: 0`. Both cameras of a
two-camera rig stream together at that rate.

If initialization fails with `opcode = 0x9 / Parse failed after 5 successful operations / status: 10`, an enabled link
has no camera behind it. Power down, reseat the quad plug and the cable ends, and try again. On this rig the fix was a
restart, reseating the board and trying a different socket. If you are unsure which link a camera is on, run the
command once per mask and keep the one that reports frame rates.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `opcode = 0x9 / ... status: 10` at startup | An enabled link has no camera. Power down, reseat the quad plug and cable ends, try again (Step 4). |
| No camera on the expected sensor | The quad plug is in CN2 instead of CN1, or a cable end is not fully seated. |
| A connector will not seat | Connectors are keyed. Do not force it; check the orientation and that the connector type matches. |
| Carrier appears dead | Check the 12 V supply, the barrel jack and the white 2-pin connector. No green LEDs means no power. |
| Sensor numbers do not match what you expect | Sensor numbers follow the link index, not a count from zero. |
| Cameras are not found after a hardware change | Power down fully, unplug both supplies, reseat, and power up again. |
| `/dev/video*` is empty | Expected on this carrier. The cameras are reached through SIPL, not Argus. |

## References

- SHW5G / SHF5G monocular camera, vendor wiki:
  <https://wiki.sensing-world.com/docs/6_1_Camera/Monocular_Camera/SHW5G_SHF5G>
