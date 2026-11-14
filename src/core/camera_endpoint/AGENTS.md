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
