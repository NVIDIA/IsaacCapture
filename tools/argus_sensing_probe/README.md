# Argus sensing-camera probe

Standalone smoke test, unrelated to anything else in this repo: does `libargus` see the
SENSING camera on this board at all, and can it pull one frame? This is the fastest way
to answer "will Argus work with our SENSING camera" without building the full
`camera_streamer` pipeline.

Run this **on the Orin itself, with the SENSING camera connected** (Argus only works on
Jetson hardware with the camera physically attached -- it cannot run on a PC or with
nothing plugged in).

```bash
cmake -B build
cmake --build build
./build/argus_sensing_probe        # device 0 by default
./build/argus_sensing_probe 1      # or a specific device index
```

## Reading the output

- `PROBE: Argus sees N camera device(s)` + a mode list per device -- Argus can at least
  enumerate the hardware. If this prints 0 devices, Argus cannot see the camera at all
  (most likely cause: no board descriptor/camera-mode config registered for Argus on
  this specific carrier -- SIPL having a working config does not mean Argus does, they
  are independent driver stacks for the same GMSL hardware).
- `PROBE: PASS -- captured one WxH frame from device N` -- Argus can both see the camera
  and actually pull frames. This is the real "will this work" answer.
- `PROBE: FAIL at <step>` -- stops exactly where it broke (enumeration, session, stream,
  or the capture/acquire itself) so you know which part failed.

## Not built or run yet

This has not been compiled or executed -- there's no Jetson Multimedia API / Argus
toolchain on the machine this was written on. Build it on the Orin first.
