<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Camera interface: core

The camera interface is the one place Isaac Capture applications get camera frames from. One
process owns the cameras and keeps their frames in fixed pools of shared buffers; viewers,
encoders and recorders in other processes read those buffers in place.

This module is the part with no memory and no sockets in it: it decides which pool slot holds
which frame set and who may still read it. It uses the standard library only. The shared memory
behind the slots and the connection to other processes live in separate modules.

## Terms

- **Stream**: one sensor's images, or one encoded video stream. It has a pool of **slots**; a
  slot holds one image, or one access unit when the stream is encoded.
- **Group**: one or more streams captured together, such as a stereo pair or a triggered rig.
  A group publishes one **frame set** at a time: one image per stream, under one sequence number.
- **Window**: how many frame sets are kept for a reader at a time, unread and held together.
- **Order**: `NEWEST` hands the reader the latest frame set each time it has room and skips the
  rest (a viewer, a live encoder). `IN_ORDER` hands over every frame set in sequence (a recorder).
  The default is `NEWEST` for a window of one and `IN_ORDER` otherwise.
- **Lease**: a reader's hold on a frame set. Its slots are not reused until the reader releases
  it or goes away.
- **Generation**: changes when a camera restarts, so a reader can tell a restart from a skip.

## How frames move

A backend gets a slot in one of two ways, then publishes:

```cpp
// The backend writes the image itself.
auto slot = publisher.acquire_slot(group, stream);  // nothing: every slot is in use, drop the frame
// ... write the pixels into that slot ...
publisher.publish(group, images, timeline);

// The camera SDK owns the buffers and picked one.
publisher.claim_slot(group, stream, slot_the_sdk_filled);
publisher.publish(group, images, timeline);
```

For the second way, `set_slot_free_callback` tells the backend the moment a slot is held by
nobody, so the buffer can go back to the SDK.

A reader subscribes with a window and an order, receives `FrameEvent`s through an `EventSink`,
and returns each one with `release`. Events carry slot indices, the frame set's timeline and the
number of frame sets the reader skipped. A slow reader skips; it never slows capture or another
reader, and no call blocks on a reader.

## Pool budget

A slot is in use while the ring of recent frame sets or any lease refers to it, so a subscription
is accepted only when the pool can cover it:

```text
slots >= ring history + sum of reader windows + backend reserve
```

Ring history is one frame set, or the largest window among in-order readers. The backend reserve
is the number of slots the backend needs for itself: one to write ahead, more for an SDK that
keeps several captures in flight. A subscription that does not fit is refused with a message that
gives these numbers.

## Tests

`tests/cpp/core/camera_core`, run with `ctest -L camera_core`. They are tagged `unit`, so
`ctest -L unit` runs them too.
