<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `camera_endpoint`

**CRITICAL (non-optional):** Before editing this package, complete the mandatory **`AGENTS.md` preflight** in [`../../../AGENTS.md`](../../../AGENTS.md) (read every applicable `AGENTS.md` on your paths, not just this file).

Read [`README.md`](README.md) for what runs on which thread before changing it.

- **Nothing sits between publish and the socket.** A frame event is encoded and sent inside `send_event`, on the publishing thread. Do not add a queue or a thread hop there, and do not log per frame.
- **`send_event` holds its camera group's lock.** It must not take another lock, block or call the publisher. That lock is also what serialises each connection's event encoder.
- **Unsubscribe before a connection is destroyed.** `Publisher::unsubscribe` returning is the guarantee that no `send_event` is running or will start for that sink.
- **A reader is untrusted.** Whatever it sends can at most get it dropped; it must not stop the thread or affect other readers.
- **Reserve, answer, activate.** The handshake response is the first message a reader receives: `subscribe`, send the response with its descriptors, then `activate`. Every way out of that sequence ends in `drop()`, which is the one place a subscription is given up.
- **A test does not assume which thread runs first.** A reader's request and the camera's publish come from two threads and are served in either order: have the reader ask for what comes after a frame set it names, not for "anything", when the test expects a particular one. Run the suite on one CPU (`taskset -c 0`) before pushing; CI's slower machines find these, a workstation does not.
- **A dropped connection returns host-memory leases only because nothing reads them any more.** Do not extend that to a kind of memory whose readers can leave work running on a slot.
