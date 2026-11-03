#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Set AGX Orin CAN0 pinmux (DOUT/DIN) live via /dev/mem.

Equivalent to NVIDIA's documented:
  busybox devmem 0x0c303010 w 0xc400   # CAN0_DOUT (TX)
  busybox devmem 0x0c303018 w 0xc458   # CAN0_DIN  (RX)

Cleared on every reboot, so it must run before each bring-up; can0-rebot.service
does this. Run with sudo. Idempotent, and doubles as a verifier — the read-back
must equal the wanted value.
"""

import mmap
import os
import struct

BASE = 0x0C303000  # AGX Orin AON pinmux block (page-aligned)
PAGE = 0x1000
REGS = {
    0x10: 0xC400,  # CAN0_DOUT_PAA0 -> function can0 (output)
    0x18: 0xC458,  # CAN0_DIN_PAA1  -> function can0 (input enabled)
}

fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
mem = mmap.mmap(
    fd, PAGE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=BASE
)
ok = True
for off, val in REGS.items():
    before = struct.unpack("<I", mem[off : off + 4])[0]
    mem[off : off + 4] = struct.pack("<I", val)
    after = struct.unpack("<I", mem[off : off + 4])[0]
    status = "OK" if after == val else "FAILED (write blocked?)"
    if after != val:
        ok = False
    print(
        f"0x{BASE + off:08x}: 0x{before:08x} -> 0x{after:08x} (wanted 0x{val:04x})  {status}"
    )
mem.close()
os.close(fd)
print("pinmux set: " + ("SUCCESS" if ok else "FAILED"))
# can0-rebot.service runs this as a bare ExecStartPre and must not go on to bring
# up a bus whose pins were never routed.
if not ok:
    raise SystemExit(1)
