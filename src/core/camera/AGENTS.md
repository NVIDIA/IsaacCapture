<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `camera`

**CRITICAL (non-optional):** Before editing this package, complete the mandatory **`AGENTS.md` preflight** in [`../../../AGENTS.md`](../../../AGENTS.md) (read every applicable `AGENTS.md` on your paths, not just this file).

Read [`README.md`](README.md) first: it has the terms (stream, group, window, order, lease) and explains why frames cross between processes this way.

## The library

- **Linux only.** The library and its tests are added under `CMAKE_SYSTEM_NAME STREQUAL "Linux"`.

## Slots, windows and leases

- **The slot and lease code uses the standard library only.** `types.hpp`, `publisher.hpp` and their sources have no slot memory, sockets, logging or platform headers in them. Memory, connections and platform calls stay in the files of the next section.
- **One type or field per concept.** Each plane has its own layout, counters are per group. Do not give a field a second meaning, and do not carry a type over from other code without modelling it for this interface.
- **Nothing ahead of its user.** A memory kind, pixel format or flag is added in the change that brings the pool or backend that needs it.
- **One lock per group.** A camera must never wait for another camera's readers, so nothing takes two groups' locks, and the registry lock is held only to look a group up.
- **`EventSink::send_event` runs on the capture path with its group's lock held.** It is `noexcept` and must not block or call back into the publisher. Callbacks to a backend (`SlotFreeCallback`) run outside the lock.
- **Reject bad input where it enters, with one check.** `problem_with` is the description check for both sides of a connection: `add_group` uses it for what a backend declares and the reader for what it was sent. Add a rule there, not in one of its callers. Sizes from another process are compared with subtraction and division, never summed.
- **Nothing is held for a newest-order reader that is not asking.** A lease taken when a frame set is published is old by the time a busy reader comes for it, and blocks the newer ones. Only `request_newest` and a release that follows it lease for such a reader.
- **Reserve, tell, activate.** `subscribe` sends nothing; the reader is told what it subscribed to, then `activate` starts delivery. Do not send from `subscribe`.
- **Publish, lease and release are three facts.** Ready, held and done (see the README) are not folded into one flag, and a reader going away is not treated as proof that its work on a slot has finished for any memory other than host memory.

## Shared memory and the socket

- **Frame memory crosses as a file descriptor on the library's Unix socket.** Do not add a messaging or shared-memory library for it; a new kind of memory is a new pool type behind the same connection.
- **Nothing on the capture path waits for a reader.** Sends do not block; a full queue and a closed peer are statuses, not exceptions.
- **A lock file owns the socket path.** Remove a socket file only while holding it, and never remove anything that is not a socket.
- **A listener takes its path when it is made and accepts only after `listen()`.** Keep the two steps apart: a camera plugin holds its endpoint while it opens its cameras, and until it can answer a reader is refused at once instead of queued.
- **An endpoint belongs to one camera plugin, and listing them only reads the directory.** `default_endpoint_paths()` does not connect, probe, cache or turn file names back into instances. A socket file left by a killed plugin is listed; whoever connects finds out.
- **Readers cannot write.** A pool is sealed before its descriptor leaves the process, and a reader maps it read-only. Do not hand a reader a writable mapping.
- **Derive what both sides can compute.** A pool's layout follows from the slot count and slot size in the stream's description; do not send it separately.

## Tests

- **Tests need no camera and no GPU.** They carry the `unit` tag; a case that starts a second process carries `integration` instead.
