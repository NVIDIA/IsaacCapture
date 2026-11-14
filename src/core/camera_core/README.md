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
behind the slots and the connection to other processes live in [`camera_ipc`](../camera_ipc).

## Terms

- **Stream**: one sensor's images, or one encoded video stream. It has a pool of **slots**; a
  slot holds one image, or one access unit when the stream is encoded.
- **Group**: one or more streams captured together, such as a stereo pair or a triggered rig.
  A group publishes one **frame set** at a time: one image per stream, under one sequence number.
- **Window**: how many frame sets are kept for a reader at a time, unread and held together.
- **Order**: `NEWEST` hands the reader the newest frame set there is each time it asks, and skips
  the rest (a viewer, a live encoder). `IN_ORDER` sends every frame set in sequence (a recorder).
  The default is `NEWEST` for a window of one and `IN_ORDER` otherwise.
- **Lease**: a reader's hold on a frame set. Its slots are not reused until the reader releases
  it or goes away.
- **Session**: one run of a group, from the plugin starting to publish it until that plugin
  ends, named by a random `session_id` in the group's description.
- **Sequence**: a frame set's place among the group's captures in the session. It is the same
  for every reader, so a reader that skips sees a gap, and with the session id it names a
  capture for good.
- **Generation**: counts restarts of the camera within the session, so a reader can tell a
  restart from a skip. A restart does not reset the sequence.

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
nobody, so the buffer can go back to the SDK. A group uses one way or the other: once it has
that callback, its slots are handed out with `claim_slot` only.

A reader subscribes with a window and an order, receives `FrameEvent`s through an `EventSink`,
and returns each one with `release`. Events carry slot indices, the frame set's timeline and the
number of frame sets the reader skipped. A slow reader skips; it never slows capture or another
reader, and no call blocks on a reader. Each group has its own lock, so one camera never waits
while another camera's events are being sent.

A subscription starts in two steps: `subscribe` reserves it, which counts it in the pool's
budget, and `activate` starts delivery. In between, whoever serves the reader tells it what it
subscribed to, so that answer always comes before the first frame set.

## When a frame set is held for a reader

An in-order reader is sent each frame set as it is published, while its window has room. It
starts, the moment its delivery is activated, with what the ring kept: nothing published
between its reservation and its activation waits for the next publish.

A newest-order reader is sent one only when it asks, with `request_newest`: the newest frame set
there is at that moment, or the next one published if it already has the newest. Nothing is
held for a reader that is not asking. Holding a frame set ahead of time, the moment it is
published, would hand a reader that was busy for a second a frame that is a second old, and
keep every newer one from it.

The request names the last frame set the reader took. That lets a reader give back a frame
set it did not use (it had stopped waiting when the frame set arrived) and ask again: it gets
the same one back if nothing newer exists, and the newer one otherwise.

## A camera SDK that owns its buffers

Some capture stacks are given their buffers up front and pick the one they fill, and stop
capturing when they run out. They map onto the second way like this:

- every buffer the SDK was given is a slot, and the group is added with a **backend reserve**
  equal to the number of buffers the SDK needs in hand to keep capturing;
- when the SDK hands a filled buffer over, the backend calls `claim_slot` for it, and
  `publish` once the images of the group are there. An image whose partner does not arrive
  in time is given up with `abandon_slot` and goes straight back to the SDK;
- the slot-free callback returns a buffer to the SDK the moment the ring and every reader
  have let go of it.

Because the budget below counts the reserve, readers can never take so many buffers that the
SDK is left short. A consumer that works on the buffers in place, such as an encoder, is a
reader like any other: it holds a buffer until it has finished consuming it.

## What publishing, holding and releasing mean

Three different things are true of a slot at different times, and they are kept apart:

- **Ready**: `publish` says the image is complete. A backend publishes a slot only when
  everything that writes it has finished.
- **Held**: a lease says a reader may read the slot. It is not reused or rewritten while any
  lease or the ring refers to it, across a camera restart as well.
- **Done**: `release` says everything the reader started on the slot has finished. For host
  memory that is the moment it stops reading.

When a reader goes away its leases are taken back at once. That is right for host memory: a
reader that is gone reads nothing, and one that closed its connection while still reading
gets changing pixels, never memory that is gone. It is not evidence that work a reader handed
to a GPU or a hardware encoder has finished, so a pool kind where such work can outlive the
connection has to bring its own rule for these slots before it is added.

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
