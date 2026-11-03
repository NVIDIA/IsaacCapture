---
name: rebot-orin-setup
description: >-
  Use when initializing a reBot B601-RS (RobStride 6-DOF, seven motors) arm
  from an NVIDIA Jetson AGX Orin using the Orin's built-in CAN (mttcan) via an external
  SN65HVD230 transceiver — no USB-CAN/PCAN adapter. Covers the uv/motorbridge
  env, bringing up can0 @ 1 Mbit, verifying the bus, reboot persistence,
  motor-ID assignment, zero calibration, and the CAN bring-up troubleshooting
  tree (loopback-passes-but-no-motors, pinmux, motor power).
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# reBot B601-RS on Jetson AGX Orin — built-in CAN setup

Jetson/ARM adaptation of the upstream
[reBot AGENTS.md guide](https://github.com/Welt-liu/reBot-B601-Agent-Guide/blob/main/en/AGENTS.md)
and the vendor's
[B601-RS quick start](https://wiki.seeedstudio.com/rebot_b601_rs_getting_started)
(assembly, power supply, safety). Both assume a **USB-CAN (PCAN) adapter**; this skill uses the
Orin's **on-chip CAN (`mttcan`)** instead. motorbridge only ever talks to a SocketCAN channel
named `can0`, so the *only* real change is **how `can0` is created** — everything
downstream (scan, id-set, gateway, Studio) is identical.

**Tested against the B601-RS (RobStride) only, not the B601-DM (Damiao).** The DM build needs
`--vendor damiao` in place of `--vendor robstride`, and motorbridge reaches Damiao hardware over
its `dm-serial` / `dm-device` transports rather than SocketCAN — so the CAN bring-up below does not
carry over. See the
[B601-DM quick start](https://wiki.seeedstudio.com/rebot_b601_dm_getting_started).

## When to use

- Bringing up a new B601-RS arm on an AGX Orin (JetPack 6 / L4T R36.x).
- Wiring is `Orin J30 (40-pin) → SN65HVD230 → arm CAN`, no USB-CAN adapter.
- Debugging "motors don't show up on CAN" on a Jetson.

## Environment assumptions

- AGX Orin devkit, JetPack 6 (verified on L4T R36.5.2, `5.15.199-tegra`, aarch64).
- `uv` on `PATH`; this directory's `pyproject.toml` pins `motorbridge` ≥ 0.5.
- `sudo` needs a TTY — an agent's non-interactive shell cannot enter a password.
  For any `sudo` step, have the user run it with the `! ` prefix in their terminal.

## Assets (in this directory)

- `assets/set_can_pinmux.py` — route the header pins to the CAN controller via
  `/dev/mem` (idempotent). Required on every boot.
- `assets/can0-rebot.service` — systemd oneshot to bring up `can0` @ 1 Mbit on boot.
- `start-gateway.sh` — launch the motorbridge WS gateway for Studio.

---

## Hardware wiring (confirm before software)

**Pin 1 is at the top right of J30** — the numbering runs right to left.

`Orin J30 → SN65HVD230` (Waveshare labels are host-side → **straight-through, not crossed**):

| Signal | `can0` pin | `can1` pin | → Waveshare |
|---|---|---|---|
| `CANx_DOUT` (TX) | 31 | 33 | `TX` |
| `CANx_DIN` (RX)  | 29 | 37 | `RX` |
| **3.3 V** (not 5 V) | 17 | 1 | `3V3` |
| GND | 30 | 34 | `GND` |

3.3 V is on pins 1 and 17 only; grounds are on 6, 9, 14, 20, 25, 30, 34, 39. Some
board revisions silkscreen `CTX`/`CRX` rather than `TX`/`RX`.

`SN65HVD230 → arm`: `CANH→CAN_H`, `CANL→CAN_L`. Termination target **~60 Ω**
across CANH–CANL (bus off). A second arm needs only a second transceiver on the
`can1` pins; every command below then uses `can1` in place of `can0`.

**Motor power:** the arm needs its own ~24–48 V motor supply. RobStride motors run
their CAN off this rail — **no motor power ⇒ no CAN replies at all.**

---

## Procedure

### Step 1 — Environment

```bash
cd integrations/rebot
uv sync
uv run motorbridge-cli --help
```

### Step 2 — Bring up built-in can0 (replaces the guide's PCAN-driver step)

Run via `! ` (needs sudo). The pinmux comes first — the header pins are not routed
to the CAN controller out of the box, and the registers are **cleared on every
reboot**:

```bash
sudo ./assets/set_can_pinmux.py     # want: pinmux set: SUCCESS
sudo modprobe can can_raw mttcan
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set up can0
ip -details link show can0     # want: state UP, can state ERROR-ACTIVE, bitrate 1000000
```

### Step 3 — Verify

```bash
sudo apt install can-utils   # JetPack ships without it; candump/cansend themselves need no root

# Controller self-test — INTERNAL loopback; proves driver only, NOT pins/transceiver/wiring:
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000 loopback on && sudo ip link set can0 up
( timeout 2 candump -n 1 can0 & ); sleep 0.4; cansend can0 123#DEADBEEF   # expect echo
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000 loopback off restart-ms 100 && sudo ip link set can0 up

# Real scan (motors MUST be powered):
uv run motorbridge-cli scan --vendor robstride --channel can0 --start-id 1 --end-id 7
# want: scan done: 7 motor(s) found
```

Keep `--end-id 7` here. A dead ID costs 80 ms + a 120 ms param-read fallback, and
the range is swept once per host-id candidate (five by default), so `--end-id 127`
takes ~2 minutes of silence and looks hung.

### Step 4 — Persist across reboots

Neither the pinmux nor the `ip link` bring-up survives a reboot. The unit redoes
both, running the script by absolute path, so install the script first:

```bash
# Not optional: the unit runs this script by absolute path.
sudo install -D -m 755 assets/set_can_pinmux.py /opt/rebot/set_can_pinmux.py
sudo cp assets/can0-rebot.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now can0-rebot.service
systemctl --no-pager status can0-rebot.service       # want: active (exited)
```

### Step 5 — Motor IDs (1–7)

Pre-assembled arm (Studio shows 7/7 online) already has IDs 1–7 → skip to Step 6.
Otherwise, one motor on the bus at a time:

```bash
uv run motorbridge-cli scan   --vendor robstride --channel can0 --start-id 1 --end-id 127
uv run motorbridge-cli id-set --vendor robstride --channel can0 --motor-id <current> --new-id <target>
```

The wide range is deliberate here — an unassigned motor's ID is unknown, so it has
to be swept. Expect ~2 minutes with no output; it is not hung. Do not narrow it
just because the verification scan in Step 3 uses `--end-id 7`.

`id-set` may report `store_parameters failed` (timeout) — the ID still writes; re-scan to confirm.
Only use motorbridge-cli subcommands from the guide; others may spin the motors.

### Step 6 — Zero calibration

```bash
./start-gateway.sh                                                    # 127.0.0.1:9002, no token
MOTORBRIDGE_WS_TOKEN=<token> BIND=0.0.0.0:9002 ./start-gateway.sh     # reachable off-box
```

Gateway **exclusively owns the bus** (stop it before any `motorbridge-cli`). Open
Motorbridge Studio (<https://motorbridge.github.io/motorbridge-studio/>), confirm 7/7
online, pose the arm at home, set current position as zero for all 7 joints, verify ~0.

Studio is a browser app, so the loopback default reaches it only from a browser on the
Orin. Bind wider and the gateway demands a token; a browser cannot set WebSocket request
headers, so pass it in the URL: `ws://<orin-ip>:9002/?motorbridge_ws_token=<token>`.

---

## Troubleshooting tree

**Golden rule: on a single-node CAN bus, "nothing replies" almost always means no
other *powered* node is ACKing. Check motor power FIRST, before any software/pinmux work.**

| Symptom | Cause | Fix |
|---|---|---|
| scan 0 motors; `cansend`→`No buffer space available`; `TX packets=0` | Nothing ACKing | Motor power off (most common), CANH/CANL swap, no termination, unpowered transceiver |
| Loopback passes but real bus silent | Loopback is internal-only | Check motor power, transceiver 3V3, wiring, pinmux register |
| pinctrl shows CAN0 pins `MUX UNCLAIMED` | Red herring (pinctrl consumer state, not HW reg) | Read regs directly: `0x0c303010` should be `0xc400`, `0x0c303018` should be `0xc458`. `sudo ./assets/set_can_pinmux.py` sets them and prints the read-back. |
| Bus worked, dead after a reboot | Pinmux cleared on boot | Re-run `sudo ./assets/set_can_pinmux.py`, or install `can0-rebot.service` (Step 4) |
| `scan` sits silent for minutes | Not hung — a wide `--end-id` waits 200 ms on every dead ID, times five host-id candidates | Use `--end-id 7` unless discovering unknown IDs |
| `can0-rebot.service` fails, `status=2`, `can't open file '/opt/rebot/set_can_pinmux.py'` | Unit installed without the pinmux script (common when updating an existing unit) | `sudo install -D -m 755 assets/set_can_pinmux.py /opt/rebot/set_can_pinmux.py && sudo systemctl restart can0-rebot.service` |
| `RTNETLINK ... Device or resource busy` | Interface already up | `sudo ip link set canX down` first |
| Stuck TX after ENOBUFS | TX FIFO wedged | `sudo ip link set can0 down && up` |
| `busybox: command not found` | Not on JetPack | Use `set_can_pinmux.py` (Python `/dev/mem`) instead of `busybox devmem` |
| `candump`/`cansend: command not found` | JetPack ships without can-utils | `sudo apt install can-utils`. The tools open `AF_CAN`/`SOCK_RAW`, which is not `CAP_NET_RAW`-gated — do not reach for `sudo` |

**Multimeter quick-checks:** CANH–CANL (bus off) ~60 Ω good / ~120 Ω one terminator /
open = broken / ~0 short-or-swap · arm power connector ~24–48 V (0 V ⇒ e-stop/fuse/loose
plug) · transceiver `3V3`→`GND` ~3.3 V.

**CAN0 pinmux registers (AGX Orin AON block):**
`can0_dout` = `0x0c303010` → `0xc400`; `can0_din` = `0x0c303018` → `0xc458`.
