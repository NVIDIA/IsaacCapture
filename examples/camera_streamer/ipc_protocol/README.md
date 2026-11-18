<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# CUDA IPC slot write generations

`camera_receiver` decodes frames into a ring of GPU slots and shares them with
same-host readers over CUDA IPC. It rewrites a slot `ring_depth` pictures after
announcing it and never waits for a reader, and the replacement notice only
arrives after the new write, so a reader cannot learn from the socket that its
copy overlapped a rewrite. This module adds one shared write-generation marker
per slot so the reader can find out after its copy has finished.

| Role | Process | What it does |
| --- | --- | --- |
| Slot writer | `camera_receiver` | Advances a slot's marker, writes the slot, announces that generation. |
| Slot reader | `cuda_ipc_consumer`, `camera_livekit_bridge`, any application | Copies a notified slot into memory it owns, then checks the marker before using the copy. |

## Protocol

- The writer owns one POSIX shared-memory region (`slot_markers.hpp`, mode `0600`,
  unpredictable name, fresh epoch per run) holding one 64-bit atomic per
  stream and slot. `_slot_markers` is its Python binding; the header is the single
  implementation Python and C++ share.
- The marker counts write attempts. The writer increments it, for every stream
  the attempt writes, before launching the first GPU operation that may touch the
  slot, and keeps the result. A failed attempt publishes nothing and the marker
  is never restored, so earlier notices for that slot stay invalid.
- A frame notice's `slot_generation` is the marker value of its completed write.
  Generations skip values when attempts fail, so a gap is not a lost picture; the
  frame sequence counts those.
- `hello.features` lists `slot_write_generation`, and `subscribed.slot_markers`
  carries the region's `name`, `size`, `layout_version`, `epoch` and
  `stream_indexes`. A reader that needs intact pictures refuses a publisher
  without the feature and maps the region only if its size, layout version and
  epoch match.
- Reader: copy the notified slot into memory it owns, wait until every GPU read of
  the slot has completed, load the marker, and keep the copy only if it equals the
  notified generation. Loading it before the copy only saves work. A reader that
  skips the check, or an older reader that ignores `slot_markers`, can still
  receive a picture mixing two frames; it is outside this guarantee.
  The egocap capture server's GMSL feed reader still decides with a ring-distance
  heuristic and does not use the markers yet.

## Ordering contract

The writer's `begin_write` is a sequentially consistent `fetch_add` followed by a
sequentially consistent fence on a lock-free 64-bit atomic in shared memory, and
it runs on the CPU before the first CUDA call that can write the slot. The reader's
`load` is a sequentially consistent load after its stream has finished. The
writer announces a generation only after every worker has waited for its own GPU
stream; on failure it drains the device before the next attempt.

If a write overlapped any part of a read, the write's launch preceded the end of
the read, and the increment preceded the launch, so the reader's load sees the
increment or a later one and the generation differs. A write that begins after
the reader's load cannot change a copy that already finished. The 64-bit counter
cannot repeat within a run, a new run has a new region and epoch, and unsupported
platforms refuse to initialise rather than fall back to non-atomic access. This
holds for a plain copy and for a kernel that reads the slot, provided each has
finished before the load.

CUDA's memory model treats an unsynchronized read and write of the same bytes as a
data race and does not specify what the reader then observes. This design relies
on the race affecting only the contents of the reader's private copy, which the
marker check discards; CUDA does not document a stronger guarantee for it, so the
check is validated by tests rather than proven from the memory model. A reader
needing a documented guarantee needs the writer to exclude overlapping access
instead of detecting it.

## Build

```bash
./build.sh                      # needs a C++17 compiler, CMake and pybind11
IPC_PROTOCOL_PYTHON=/path/to/python ./build.sh
```

`camera_receiver/setup.sh` and `cuda_ipc_consumer/setup.sh` run it with their own
environment. The module file name carries the interpreter's ABI tag, so each
interpreter builds in its own tree.
