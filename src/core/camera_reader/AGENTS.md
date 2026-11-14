<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `camera_reader`

**CRITICAL (non-optional):** Before editing this package, complete the mandatory **`AGENTS.md` preflight** in [`../../../AGENTS.md`](../../../AGENTS.md) (read every applicable `AGENTS.md` on your paths, not just this file).

- **The public header shows `camera_core` types only.** The socket, the pools and the protocol's messages stay in the `.cpp`; an application that links the reader does not see them.
- **No copies and no conversions here.** A view is the plugin's buffer. Pixel conversion, if a consumer wants it, is a separate step on top.
- **A frame set outlives its reader safely.** Both share one state object; keep it that way rather than making `FrameSet` point back at the reader.
- **Releasing must never throw or block.** It runs in destructors, possibly on another thread; a plugin that is gone takes its frames back by itself.
- **The plugin is trusted for content, not for bounds.** Check a frame event against the group before handing out views, and check the handshake answer in `open` before mapping anything: the protocol version, that it is the group that was asked for, `problem_with` for the description, one pool per stream.
- **A newest-order reader asks; it does not drain.** `next` sends the request and takes the answer. Do not go back to reading whatever is queued and keeping the last: what is queued was held when it was published.
- **In a binding, whatever exposes image memory holds the frame set.** An array's base is the lease, not just the mapping, so an array never shows another frame; a scope that must not keep anything checks that count when it ends. Do not hand out a view that outlives its lease, and do not build a scope on something that can be released under a view (a `memoryview` released while `numpy.frombuffer` still points at it dangles).
- **A full window is an error, not a wait.** `next` by a reader that holds its window's worth throws `CameraWindowExhausted`; it does not time out as if no frame had come.
