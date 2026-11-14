<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Troubleshoot CloudXR

Use these commands only when a live device check needs CloudXR. Treat the runtime as shared: inspect
it first, and do not start or stop it or accept the EULA without explicit user authorization.
Use `docs/source/references/cloudxr.rst` and the current service CLI help as the authority.

## Check

```bash
python -m isaacteleop.cloudxr.service status
python -m isaacteleop.cloudxr.service logs -n 50
```

`status` reports the runtime that is actually serving and exits non-zero when none is available.
A successful status proves runtime IPC only, not headset connection, plugin health, or device data.
Use `logs -f` to follow a detached service's log.

## Start

Start a detached service that survives the current shell:

```bash
python -m isaacteleop.cloudxr.service start
```

On the first run, `start` cannot prompt for the EULA. After the user reviews and explicitly accepts
it, run:

```bash
python -m isaacteleop.cloudxr.service start --accept-eula
```

Use foreground mode when the terminal or container should own the service:

```bash
python -m isaacteleop.cloudxr.service run
```

`Ctrl+C` stops foreground mode. `start` refuses to replace a runtime that is already serving.

## Test Without a Physical Headset

CloudXR has no clientless mode for a live `TeleopSession`. Here, testing without a physical headset
means connecting a desktop browser as the XR client. Open
`https://nvidia.github.io/IsaacTeleop/client`; the client loads IWER (Immersive Web Emulator
Runtime), which emulates a Meta Quest 3. Enter the CloudXR host IP, accept the link to its
self-signed certificate, then click **Connect**.

To serve the client locally, start the service with:

```bash
python -m isaacteleop.cloudxr.service start --host-client
```

Then open `https://<host-ip>:48322/client/` and click **Connect**.

This is headset *emulation*, suitable for a quick synthetic pipeline check. It does not verify real
headset tracking or a physical input device. `NV_DEVICE_PROFILE=Quest3` only selects the compatible
runtime profile; it does not create a headset. Likewise, `XR_MND_headless` means that an OpenXR
application needs no graphics binding, not that it needs no OpenXR system. If no client is connected,
`XR_ERROR_FORM_FACTOR_UNAVAILABLE` remains expected.

A plugin that neither consumes XR data nor injects into OpenXR does not need CloudXR or IWER for its
live check.

## Connect the Live Check

`device_live.py.template` defaults to an already configured OpenXR runtime. For a separately started
CloudXR service, either let the generated script attach:

```bash
python examples/<device>/python/<device>_live.py test --launch-cloudxr-runtime
```

or configure the current shell and keep the no-launch mode:

```bash
source ~/.cloudxr/run/cloudxr.env
python examples/<device>/python/<device>_live.py test --no-launch-cloudxr-runtime
```

When no service exists, `--launch-cloudxr-runtime` starts a detached service that remains running
after the script exits.

## Close

Check ownership before stopping anything. Stop a detached service with:

```bash
python -m isaacteleop.cloudxr.service stop
```

`stop` exits cleanly when no detached service is running. Stop `service run` with `Ctrl+C`. A
generated script using `launch_context(args, run_embedded=True)` and invoked with
`--launch-cloudxr-runtime` owns its runtime and stops it when its context exits. For a non-default
install directory, pass the same `--cloudxr-install-dir` to `status` and `stop`.

Never use `pkill` or delete `~/.cloudxr/run` markers to simulate shutdown. If `stop` fails, inspect
`service logs` and report the unresolved runtime instead of killing unrelated processes.
