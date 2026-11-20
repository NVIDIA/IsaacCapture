<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes — `camera`

**CRITICAL (non-optional):** Before editing this package, complete the mandatory **`AGENTS.md` preflight** in [`../../../AGENTS.md`](../../../AGENTS.md) (read every applicable `AGENTS.md` on your paths, not just this file).

Read [`README.md`](README.md) first: it has the terms (stream, group, window, order, lease) and explains why frames cross between processes this way.

## The library

- **Linux only.** The library and its tests are added under `CMAKE_SYSTEM_NAME STREQUAL "Linux"`.

## Reading cameras

- **`camera_reader.hpp` shows the types of `types.hpp` only.** The socket, the pools and the protocol's messages stay in `camera_reader.cpp`; an application that includes the reader does not see them.
- **The reader makes no copies and no conversions.** A view is the plugin's buffer. Pixel conversion, if a consumer wants it, is a separate step on top.
- **A frame set outlives its reader safely.** Both share one state object; keep it that way rather than making `FrameSet` point back at the reader.
- **Releasing must never throw or block.** It runs in destructors, possibly on another thread; a plugin that is gone takes its frames back by itself.
- **The plugin is trusted for content, not for bounds.** Check a frame event against the group before handing out views, and check the handshake answer in `open` before mapping anything: the protocol version, that it is the group that was asked for, `problem_with` for the description, one pool per stream.
- **A newest-order reader asks; it does not drain.** `next` sends the request and takes the answer. Do not go back to reading whatever is queued and keeping the last: what is queued was held when it was published.
- **In a binding, whatever exposes image memory holds the frame set.** An array's base is the lease, not just the mapping, so an array never shows another frame; a scope that must not keep anything checks that count when it ends. Do not hand out a view that outlives its lease, and do not build a scope on something that can be released under a view (a `memoryview` released while `numpy.frombuffer` still points at it dangles).
- **A full window is an error, not a wait.** `next` by a reader that holds its window's worth throws `CameraWindowExhausted`; it does not time out as if no frame had come.
- **An empty location looks; a named one does not.** With no endpoint and no instance the reader asks every endpoint `default_endpoint_paths()` lists. With either, it talks to that one plugin alone: the full handshake timeout, and an error if the plugin is not there. Do not fall back from one to the other.
- **Looking stays plain.** The endpoints are asked one after another, each with a short fixed timeout. A refused connection is passed over silently. An endpoint that fails to say what it publishes (no answer in time, a refusal, a closed connection, any other failure) is passed over too, named with its reason in `open`'s error and left out by `list`; do not narrow that to the timeout. A group on two endpoints is an error, never a choice. No cache, no index file, no retry, no thread, and no option for any of it.

## Publishing cameras

These rules hold for the publishing side of this library and for every camera plugin built on it (`src/plugins/camera_*`).

- **One camera plugin per vendor or SDK.** A plugin links exactly one camera SDK and hands exactly one `Backend` to `run_camera_plugin`. Do not add a second SDK to an existing plugin, a list of backends, or run-time loading of backends. A plugin that needs an SDK gets its own `BUILD_PLUGIN_*` option, off by default, like the repository's other vendor plugins. The steps are in [`README.md`](README.md), "Writing a camera plugin".
- **No SDK types outside a plugin's `core/` directory.** This library, a plugin's `app/` and its `plugin.yaml` see `CameraSource` and this library's own types only, and readers never link a camera SDK.
- **A plugin's `main` calls `run_camera_plugin` and does nothing else.** The arguments, the configuration file, the endpoint and the exit codes are written once, here, so they are the same for every camera plugin.
- **A plugin serves its own cameras.** The configuration file is shared by the camera plugins of a user, and its form is checked whole: an entry that is not an id with a `backend:` and `name: value` settings stops every plugin that reads the file, whichever backend it names. Of a well-formed file a plugin takes the cameras whose `backend:` is its backend's name and leaves the others alone: it never opens or lists them, except to say what the file names when none is its own.
- **A backend refuses what it does not understand.** Read options through `OptionReader` and call `finish()`, so a misspelt option is an error.
- **A camera drops a frame rather than wait.** When `acquire` has no free slot, abandon what was taken for that frame set and carry on.
- **A backend is opened once, with all its cameras.** What an SDK has open belongs to the backend and is shared by the sources it returns; a source is one camera and publishes one group. Do not open an SDK per camera when the SDK serves several from one object.
- **The backend interface shows what the `Publisher` can do, and no more.** `FrameOutput` passes through the publisher's two ways to get a slot, the slot-free callback and the restart, and `CameraSource::backend_reserve()` passes through the reserve. A further need (memory an SDK allocates, a way to publish before the device has finished writing) is added with the pool and the backend that have it, not ahead of them.
- **A camera plugin does not encode.** It publishes pixels, and streaming and recording are readers that encode what they read. Do not add a codec, an encoded stream or an encoder control to the backend interface or the protocol.
- **An option says `memory=gpu`, an application asks `on_gpu()`.** Where frames are is host or GPU for both; which kind of GPU memory carries them is the platform's, chosen by the plugin. Only code that makes or maps a pool looks at `MemoryKind`, and a view gives a GPU address per plane, never one for the slot.
- **`publish()` means complete.** A source publishes a slot only when everything that writes it has finished; readers look at it from that moment. For CUDA memory that is when the device has finished, not when the write was queued: readers are other processes, and nothing orders their reads behind the camera's CUDA streams. A copy from host memory returns before the device has the bytes, so wait for it (`copy_to_cuda` does).
- **The usage is terminal output; everything else is logged.** `run_camera_plugin` prints the usage, and the refusal of an argument together with it, as one message on one channel. Every other line goes through `isaaccapture::Logger` under `isaaccapture.camera.<Name>`.

## Slots, windows and leases

- **The slot and lease code uses the standard library only.** `types.hpp`, `publisher.hpp` and their sources have no slot memory, sockets, logging or platform headers in them. Memory, connections and platform calls stay in the files of the next section.
- **One type or field per concept.** Each plane has its own layout, counters are per group. Do not give a field a second meaning, and do not carry a type over from other code without modelling it for this interface.
- **Nothing ahead of its user.** A memory kind, pixel format or flag is added in the change that brings the pool or backend that needs it.
- **One lock per group.** A camera must never wait for another camera's readers, so nothing takes two groups' locks, and the registry lock is held only to look a group up.
- **`EventSink::send_event` runs on the capture path with its group's lock held.** It is `noexcept` and must not block or call back into the publisher. Callbacks to a backend (`SlotFreeCallback`) run outside the lock.
- **Reject bad input where it enters, with one check.** `problem_with` is the description check for both sides of a connection: `add_group` uses it for what a backend declares and the reader for what it was sent. Add a rule there, not in one of its callers. Sizes from another process are compared with subtraction and division, never summed.
- **Nothing is held for a newest-order reader that is not asking.** A lease taken when a frame set is published is old by the time a busy reader comes for it, and blocks the newer ones. Only `request_newest` and a release that follows it lease for such a reader.
- **Reserve, tell, activate.** `subscribe` sends nothing; the reader is told what it subscribed to, then `activate` starts delivery. Do not send from `subscribe`.
- **Publish, lease and release are three facts.** Ready, held and done (see the README) are not folded into one flag. A reader going away is not proof that its work on a slot has finished; taking its slots back at once is right for host and CUDA pools because a reader's mapping is read-only and keeps the memory alive, so late work can only read newer pixels. A pool kind where late work could write, or outlive the memory, needs its own rule.

## Shared memory and the socket

- **Frame memory crosses as a file descriptor on the library's Unix socket.** Do not add a messaging or shared-memory library for it; a new kind of memory is a new pool type behind the same connection.
- **Nothing on the capture path waits for a reader.** Sends do not block; a full queue and a closed peer are statuses, not exceptions.
- **A lock file owns the socket path.** Remove a socket file only while holding it, and never remove anything that is not a socket.
- **A listener takes its path when it is made and accepts only after `listen()`.** Keep the two steps apart: a camera plugin holds its endpoint while it opens its cameras, and until it can answer a reader is refused at once instead of queued.
- **An endpoint belongs to one camera plugin, and listing them only reads the directory.** `default_endpoint_paths()` does not connect, probe, cache or turn file names back into instances. A socket file left by a killed plugin is listed; whoever connects finds out.
- **Readers cannot write.** A host pool is sealed before its descriptor leaves the process and a reader maps it read-only; a CUDA pool's view is mapped read-only on the device. Do not hand a reader a writable mapping.
- **CUDA is reached through `cuda_pool.cpp` only, and never linked.** The driver's library is loaded when CUDA memory is first used, so the library, the plugins and the readers run where no NVIDIA driver is installed, and a build without the CUDA Toolkit compiles the same interface with every call saying why it cannot be used. Do not add `CUDA::` targets to the library's link line or include CUDA headers elsewhere in it. Memory is used in the device's primary context, which is what CuPy, PyTorch and the CUDA runtime use; do not create a context.
- **A CUDA device is named by its UUID between processes.** An index means something in one process only; `ImageView::cuda_device_index` is the reader's own index for the device.
- **Derive what both sides can compute.** A pool's layout follows from the slot count and slot size in the stream's description; do not send it separately.

## The protocol

The schema is [`../schema/fbs/camera_protocol.fbs`](../schema/fbs/camera_protocol.fbs). Read [`../schema/AGENTS.md`](../schema/AGENTS.md) and "The protocol" in the [`README.md`](README.md) before changing it: a reader and a plugin from different releases must keep understanding each other, and the golden check enforces it.

- **Few messages, and no fewer than correctness needs.** A connection has one purpose, stated in the handshake; do not add a message for what a handshake field can say. A newest-order reader asks for each frame set (`CameraAcquire`): do not fold that back into sending frame sets as they are published.
- **This is local IPC.** Lease ids, slot indices and attached descriptors name resources of one machine. Do not reuse the envelope as a network or recording format, and do not wrap it in the recorder's `Record` tables or its `DeviceDataTimestamp`.
- **A new message says whether it may be ignored.** One that only informs is sent with `skippable`; an operation the peer must carry out is not, and needs a protocol version so that it is only sent to a peer that knows it.
- **The numbers are the protocol.** Union members, flag bits and enum values are written out in the schema and pinned by `static_assert` and a test; a member keeps its number.
- **FlatBuffers stays inside the codec.** Its users see the library's own types; only `codec.cpp` and tests include the generated header.
- **Decoded bytes are untrusted.** Verify, then bound every count against the fixed arrays it fills, and check for a body behind a body type. The decoder establishes structure only; what a description means is checked by `problem_with` where it is used.
- **The encoder refuses, it does not truncate.** A count that does not fit is an error to the caller, never a shorter message.
- **A group this build cannot read is an entry, not an error.** A kind added by a newer plugin marks its group `unsupported` and leaves the other groups of the same response readable.
- **A new enum value needs both sides.** The enum in `types.hpp` and the schema enum carry the same numbers (pinned by `static_assert`), and the decoder must accept the value before a plugin sends it.
- **The frame path does not allocate.** Keep `encode(const FrameEvent&)` free of containers that do.
- **Evolution is tested with hand-built messages.** A passing golden comparison shows layout, not that two releases understand each other: a rule about newer or older peers gets a test that builds the message that peer would send.

## The endpoint

Read "What runs where" in the [`README.md`](README.md) before changing it.

- **Nothing sits between publish and the socket.** A frame event is encoded and sent inside `send_event`, on the publishing thread. Do not add a queue or a thread hop there, and do not log per frame.
- **The endpoint's `send_event` takes no other lock.** The group's lock it runs under is also what serialises each connection's event encoder.
- **Unsubscribe before a connection is destroyed.** `Publisher::unsubscribe` returning is the guarantee that no `send_event` is running or will start for that sink.
- **A reader is untrusted.** Whatever it sends can at most get it dropped; it must not stop the thread or affect other readers.
- **Reserve, answer, activate.** The handshake response is the first message a reader receives: `subscribe`, send the response with its descriptors, then `activate`. Every way out of that sequence ends in `drop()`, which is the one place a subscription is given up.
- **The constructor takes the path; only `start()` listens.** Readers find a plugin by connecting to every endpoint of their user, so an endpoint that accepts before it can answer costs each of them a wait. Do not listen in the constructor, and call `listen()` before the service thread begins: a listener that does not listen yet polls as hung up.

## Tests

- **Tests need no camera and no GPU.** They carry the `unit` tag; a case that starts a second process carries `integration` instead. A case that needs a CUDA device carries `gpu` and begins with `if (!cuda_unavailable().empty()) SKIP(cuda_unavailable());`, so it is skipped, not failed, where there is none. CUDA does not start under AddressSanitizer unless `ASAN_OPTIONS=protect_shadow_gap=0` is set; without it those cases skip.
- **A test of CUDA memory reads from another process, repeatedly.** Within one process the device runs a read after the write that was queued before it, so a test with the camera and the reader in one process passes with a camera that publishes too early. The cases that start the plugin as a process find that; run them with `ctest -L gpu --repeat until-fail:20 -j 4`.
- **A test does not assume which thread runs first.** A reader's request and the camera's publish come from two threads and are served in either order: have the reader ask for what comes after a frame set it names, not for "anything", when the test expects a particular one. Run the suite on one CPU (`taskset -c 0`) before pushing; CI's slower machines find these, a workstation does not.
- **A test waits for what the plugin's thread does; it does not assume it.** `next` returning says nothing about whether the plugin has taken the reader's request, so wait for the effect (`Served::wait_until_sent`) before counting what was sent.
- **A test's helper thread is stopped and joined however the test ends.** A failed `REQUIRE` throws, and a `std::thread` that is still joinable then aborts the process instead of reporting the failure. Use `std::jthread`, or a guard whose destructor joins.
- **Tests that look through the user's plugins, or let a plugin choose its endpoint or its configuration file, get a user's directories of their own.** Point `XDG_RUNTIME_DIR`, and for a plugin's configuration file `XDG_CONFIG_HOME`, at a short temporary directory and put them back, as `EndpointDirectory` (`tests/cpp/core/camera/endpoint_directory.hpp`) and `UserDirectories` (`tests/cpp/core/camera/test_camera_plugin.cpp`) do, and the `endpoint_directory` fixture of `tests/python/core/camera` for a Python test and the plugins it starts; a test that uses the real ones finds whatever camera plugin and `cameras.yaml` the developer has.
- **A test that runs `run_camera_plugin` in the test process aims the stop signal at the plugin's thread**, and starts that thread with the signal blocked (`PluginRun` in `tests/cpp/core/camera/test_camera_plugin.cpp`). A signal sent to the process would end the test instead.
- **A Python test starts plugins through `start_plugin`.** The fixture stops every process it started and waits for it, also when the test fails, before the directory with the socket files is removed. A plugin started through the plugin manager is stopped in a `finally`.
