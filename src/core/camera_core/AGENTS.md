<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `camera_core`

**CRITICAL (non-optional):** Before editing this package, complete the mandatory **`AGENTS.md` preflight** in [`../../../AGENTS.md`](../../../AGENTS.md) (read every applicable `AGENTS.md` on your paths, not just this file).

Read [`README.md`](README.md) for the terms (stream, group, window, order, lease) before changing behaviour.

- **Standard library only.** No slot memory, sockets, logging or platform headers here; that keeps the module building and tested on every platform. Memory and connections live in `camera_ipc`.
- **One type or field per concept.** A codec is not a pixel format, each plane has its own layout, counters are per group. Do not give a field a second meaning, and do not carry a type over from a prototype without modelling it for this interface.
- **Nothing ahead of its user.** A memory kind, pixel format or flag is added in the change that brings the pool or backend that needs it.
- **One lock per group.** A camera must never wait for another camera's readers, so nothing takes two groups' locks, and the registry lock is held only to look a group up.
- **`EventSink::send_event` runs on the capture path with its group's lock held.** It is `noexcept` and must not block or call back into the publisher. Callbacks to a backend (`SlotFreeCallback`) run outside the lock.
- **Reject bad input where it enters, with one check.** `problem_with` is the description check for both sides of a connection: `add_group` uses it for what a backend declares and the reader for what it was sent. Add a rule there, not in one of its callers. Sizes from another process are compared with subtraction and division, never summed.
- **Nothing is held for a newest-order reader that is not asking.** A lease taken when a frame set is published is old by the time a busy reader comes for it, and blocks the newer ones. Only `request_newest` and a release that follows it lease for such a reader.
- **Reserve, tell, activate.** `subscribe` sends nothing; the reader is told what it subscribed to, then `activate` starts delivery. Do not send from `subscribe`.
- **Publish, lease and release are three facts.** Ready, held and done (see the README) are not folded into one flag, and a reader going away is not treated as proof that its work on a slot has finished for any memory other than host memory.
