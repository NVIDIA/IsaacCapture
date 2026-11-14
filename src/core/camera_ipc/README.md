<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Camera interface: sharing frames between processes

The process that owns the cameras and the processes that read them are on one machine. This
module is what crosses between them. Linux only.

[`camera_core`](../camera_core) decides which slot holds which frame; this module provides the
memory behind the slots.

## Host pool

`HostPool` is one stream's slots in host memory: a single anonymous file in memory
(`memfd_create`), with the slots laid out one page-aligned stride apart. The owner maps it
writable, then seals it. A reader is sent the descriptor and maps it with `HostPoolView`, read
only. Both sides derive the layout from the slot count and slot size, which the stream's
description carries.

```cpp
HostPool pool(stream.slot_count, stream.slot_bytes, "camera.left");
std::span<std::byte> pixels = pool.slot(index);          // the owner writes here

HostPoolView view(std::move(received_descriptor), stream.slot_count, stream.slot_bytes);
std::span<const std::byte> image = view.slot(index);     // a reader sees the same bytes
```

Why an anonymous sealed file and not named shared memory (`shm_open`):

- **Nothing to clean up.** There is no name to collide with another instance and no file left in
  `/dev/shm` after a crash. The memory is freed when the last process that holds it goes away.
- **Readers cannot damage it.** The seals stop any process that receives the descriptor from
  mapping it writable, writing to it, or resizing it. One misbehaving reader cannot change what
  the others see, and cannot make the owner fault by shrinking the file under it.
- **Access follows the connection.** Only a process that was handed the descriptor can map the
  pool.

## Tests

`tests/cpp/core/camera_ipc`, run with `ctest -L camera_ipc`. They are tagged `unit`, so
`ctest -L unit` runs them too.
