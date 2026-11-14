<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `camera_reader`

**CRITICAL (non-optional):** Before editing this package, complete the mandatory **`AGENTS.md` preflight** in [`../../../AGENTS.md`](../../../AGENTS.md) (read every applicable `AGENTS.md` on your paths, not just this file).

- **The public header shows `camera_core` types only.** The socket, the pools and the wire messages stay in the `.cpp`; an application that links the reader does not see them.
- **No copies and no conversions here.** A view is the plugin's buffer. Pixel conversion, if a consumer wants it, is a separate step on top.
- **A frame set outlives its reader safely.** Both share one state object; keep it that way rather than making `FrameSet` point back at the reader.
- **Releasing must never throw or block.** It runs in destructors, possibly on another thread; a plugin that is gone takes its frames back by itself.
- **The plugin is trusted for content, not for bounds.** Check a frame event against the group before handing out views.
