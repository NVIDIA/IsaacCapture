<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Chromium key code table

`dom_code_data.inc` is Chromium's table of physical keys: for each key, its W3C UI Events
`KeyboardEvent.code` string and its USB HID, Linux evdev, XKB, Windows and macOS codes. The
keyboard tracker includes it to translate `KeyboardEvent.code` strings to evdev codes and back,
and the Python bindings build `EvdevKeyCode` from it, so no key list is maintained by hand.

- Source: <https://chromium.googlesource.com/chromium/src/+/863ae6f44a6d93d0aaa3674b0554f4a9298d5fd1/ui/events/keycodes/dom/dom_code_data.inc>
- License: BSD-3-Clause, see [`LICENSE`](LICENSE). The build ships it in the wheel as
  `isaaccapture/deviceio_trackers/CHROMIUM_LICENSE`, next to the module that compiles the table in.
- Vendored verbatim. To update, copy the file from a newer Chromium revision and update the
  revision above.
