<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Camera interface: the protocol between the plugin and a reader

What the camera plugin and a reader say to each other over their socket
([`camera_ipc`](../camera_ipc)). Each message is one socket message:

```text
reader                                  plugin
  |  handshake request                     |
  |  (version, group, window, order)       |
  |--------------------------------------->|
  |  handshake response (the group) with   |
  |  one pool descriptor per stream        |
  |<---------------------------------------|
  |  acquire (after which frame set)       |   newest order only: once per frame set
  |--------------------------------------->|
  |  frame set (lease, slot per stream,    |
  |  times)                                |
  |<---------------------------------------|
  |  release (lease)                       |
  |--------------------------------------->|
```

- **One connection, one purpose.** A connection carries one subscription, so the reader says
  what it came for in its first message and the plugin's answer settles it. A reader that
  wants two groups opens two connections. A request that names no group asks only what the
  plugin publishes, and the response lists every group.
- **The answer comes first.** The handshake response is the first message a plugin sends. The
  subscription is reserved, the response is sent, and only then does delivery start.
- **A refusal ends the connection.** The response then carries the reason: a protocol version
  the plugin no longer speaks, a group that does not exist, or a window that does not fit
  the pool.
- **A newest-order reader asks for each frame set.** `acquire` means: hold the newest frame
  set for me, now if there is one after the one I name, otherwise as soon as there is. The
  frame set is chosen when the request is served, so a reader that was busy gets what the
  camera saw last. An in-order reader is sent every frame set without asking.
- **Closing the connection is the unsubscribe.** Every frame set the reader still held goes
  back to the pool.

## What this protocol is not

It is spoken between two processes on one machine. A lease id means something on one
connection; a slot index and the file descriptors that travel with the handshake response
name memory of that machine. None of that is a network packet's identity, and nothing here
is a recording format. A transport that sends frames to another machine carries what
identifies a capture there (the group's `session_id` and a frame set's `sequence`, its
times, its flags) in its own messages, and a recorder maps the times to its own timestamps.

The descriptors attached to a handshake response are one per stream, in stream order. That
is the rule for host memory, the one kind there is. A kind of memory that needs something
else describes its attachments when it is added.

## Schema

The messages are FlatBuffers tables in
[`schema/fbs/camera_protocol.fbs`](../schema/fbs/camera_protocol.fbs), with the other
schemas of the repository. They are never recorded, but they are a contract in the same
way: a reader and a plugin can come from different releases. The golden check of
[`schema`](../schema/README.md) therefore guards them like any other schema, and the same
rules apply: fields are appended with fresh ids and left optional, nothing is renumbered.
The numbers of the messages in their union and the bits of the frame flags are written out
in the schema and pinned by the codec and its tests. The one struct, a plane's offset and
stride, is complete as it is and is never extended.

A matching golden says the bytes still line up. It does not say two releases still
understand each other, so the rules below are tested with messages built the way a newer
and an older peer would build them.

## A reader and a plugin from different releases

- **Version.** Each side has a newest and an oldest protocol version it speaks. The reader
  sends its newest; the plugin answers with the older of that and its own newest, and
  refuses a reader that is older than its oldest. The reader checks the answer against its
  own range. A version is added when one side comes to depend on an operation the other
  must carry out; fields and messages that only inform need none.
- **A message type this build does not know** is ignored when its sender marked it
  `skippable`, which is for messages that only inform. Not marked, it is an operation the
  sender counts on: the receiver ends the connection, and the sender notices, instead of
  both carrying on with different ideas of what was agreed.
- **A group this build cannot read**, because it uses a codec, a pixel format or a kind of
  memory added later, is still listed, with the reason. The groups next to it are
  unaffected, and a reader that asks for it is told why it cannot have it.
- **A request for something added later**, such as a delivery order, is refused with the
  reason.

## Codec

`MessageEncoder` and `decode_message` turn [`camera_core`](../camera_core) types into bytes
and back, so nothing outside this module sees FlatBuffers. The encoder reuses its buffer; a
frame set is encoded without allocating. It refuses a message that names more streams,
images or planes than a group can have, and writes nothing in its place.

The decoder treats its input as coming from another process: it verifies the buffer and
bounds every count. That establishes that the bytes are a message, not that what they
describe makes sense. Whether a group's planes fit its slots is checked by `problem_with`
in [`camera_core`](../camera_core), by the plugin for what a backend declares and by the
reader for what it was sent.

## Tests

`tests/cpp/core/camera_protocol`, run with `ctest -L camera_protocol`. They are tagged `unit`.
