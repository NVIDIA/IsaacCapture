<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

<a id="dedicated-cloudxr-runtime"></a>

# CloudXR Service

A host runs **one** CloudXR runtime, and the CloudXR service owns it, together
with the WSS proxy the headset connects through. Applications attach to
whichever runtime is serving rather than starting their own, so the runtime —
and the headset connection — outlives the application using it.

The teleop examples start a service themselves if none is running (see
[3. Configure CloudXR (optional)](../teleop/quick_start.md#run-cloudxr-server) in the quick start), and print a notice when they do,
because that service keeps running after the example exits. Examples that use
`CloudXRLauncher` host CloudXR.js at `https://<host>:48322/client/` by
default when they start that service; pass `--no-host-client` to use the
GitHub Pages client instead. Start the service yourself when you want to:

- keep the headset connected while you restart a teleop application repeatedly
  during development,
- point OpenXR applications that do not embed `CloudXRLauncher` at CloudXR,
- use the out-of-band automation flags (`--setup-oob`, `--usb-local`)
  that only the service exposes.

> ##### Sections
> 
> * [Start the service](#start-the-service)
> * [Manage the service](#manage-the-service)
> * [Optional launch modes](#optional-launch-modes)
> * [USB-local startup checks](#usb-local-startup-checks)
> * [Re-open the client on the headset](#re-open-the-client-on-the-headset)
> * [Load CloudXR environment variables](#load-cloudxr-environment-variables)
> * [Run teleop examples against the service](#run-teleop-examples-against-the-service)
> * [Use a system OpenXR runtime (no CloudXR attach)](#use-a-system-openxr-runtime-no-cloudxr-attach)
> * [Run the service in a container or CI](#run-the-service-in-a-container-or-ci)
> * [Connect the OOB headset after application startup](#connect-the-oob-headset-after-application-startup)
> * [Configuration](#configuration)
> * [Troubleshooting](#troubleshooting)

## Start the service

With the `isaaccapture` package installed (including the `cloudxr` extra,
see [2. Install the isaaccapture pip package](../teleop/quick_start.md#install-isaaccapture-pip-package)), start the service. The first run
downloads the CloudXR Web Client SDK:

```bash
python -m isaaccapture.cloudxr.service start --accept-eula
```

`start` detaches the service from your terminal: it keeps running when the
shell that started it exits, so you can close the window, and every command in
the next section finds it again.

A detached service has no terminal to prompt on, so `start` never asks about
the EULA — it requires `--accept-eula` until acceptance has been recorded,
and tells you so. Acceptance is remembered in
`~/.cloudxr/run/eula_accepted`, so the flag is only needed once per install
directory. Review the licence before accepting; the error message links it.

Use `run` in place of `start` to keep the service in the foreground, where
its output goes to the terminal and `Ctrl+C` stops it — that is what a
container entrypoint wants. `run` does prompt when it has a terminal.

You should see output similar to:

![CloudXR service startup output](_static/cloudxr-run-output.png)

**Figure:** CloudXR service startup output

#### NOTE
`python -m isaaccapture.cloudxr` still works and is equivalent to
`python -m isaaccapture.cloudxr.service run`. It warns on startup and is
removed in Isaac Capture 1.7.

Take note of the `source ~/.cloudxr/run/cloudxr.env` path in the output. You
will need it in [Load CloudXR environment variables](#load-cloudxr-environment-variables).

## Manage the service

Every command below is a subcommand of
`python -m isaaccapture.cloudxr.service`.

| Command   | What it does                                                                                                                                                                                       |
|-----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `start`   | Start a detached service. Refuses if a runtime is already serving,<br/>rather than dropping the live session.                                                                                      |
| `run`     | Run in the foreground until `Ctrl+C`.                                                                                                                                                              |
| `stop`    | Stop the detached service and tear the runtime down with it. Exits<br/>cleanly when there is nothing to stop.                                                                                      |
| `status`  | Report the running session — device profile, log files, client URL,<br/>and whether it is detached or in the foreground. Exits non-zero when<br/>no runtime is serving, so scripts can gate on it. |
| `logs`    | Show the detached service’s log (`-n` lines, `-f` to follow).                                                                                                                                      |

`status` reports the session that is actually running, recovered from the
service’s own command line — not the defaults of the command you just typed.

## Optional launch modes

`run` and `start` accept the same optional flags, which can be combined to
control how the headset connects and how the web client is delivered.

| Command                                   | What it does                                                                                                                                                               |
|-------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `service start`                           | Plain: headset navigates to GitHub Pages URL over WiFi.                                                                                                                    |
| `service start --host-client`             | Serves the web client at `https://<ip>:48322/client/` via the WSS<br/>proxy. No separate port, no USB or TURN relay required. Useful when<br/>GitHub Pages is unreachable. |
| `service start --setup-oob`               | OOB hub + CDP automation: opens the browser on the headset and<br/>auto-clicks CONNECT over USB adb. Client URL is GitHub Pages.                                           |
| `service start --setup-oob --host-client` | OOB hub + CDP with client at `/client/` on the WSS proxy<br/>(air-gapped / proxy use).                                                                                     |
| `service start --setup-oob --usb-local`   | All traffic over USB: adb-reverse + coturn TURN relay; serves<br/>`/client/` on the WSS port via loopback. Requires `coturn`<br/>and a WiFi-associated headset.            |

`--usb-local` requires `--setup-oob`. See
[Out-of-Band Teleop Control](oob_teleop_control.md) for full OOB documentation. The OOB hub
prints a banner as it comes up, so `run` is often the more useful command
while setting those modes up for the first time.

## USB-local startup checks

On a clean `--usb-local` start, the service records the selected headset’s
negotiated USB speed, hub and controller path, and evidence of faster USB
root hubs on the same controller. After its existing adb reverse rules are
ready, it measures three seconds of host-to-headset transfer using the
already served `/client/bundle.js` over HTTPS. This finishes before the
browser is opened. The selected headset must first report
`mWakefulness=Awake`; if that cannot be confirmed, no transfer or browser
launch occurs and the JSON report records the incomplete check. If the headset
lacks a usable curl or timer, the service uses a three-second ADB-only transfer
and labels the narrower coverage in the result.

One JSON report is saved under `<CloudXR install dir>/logs/usb-health/` for
each USB-local start. It begins before ADB selection, records partial results
when device preparation or a check fails, and refreshes link topology after
a preserved browser reconnect without repeating the active transfer. It
uses a hashed device serial, and keeps at most 20 reports or 14 days of
history. A link below 5000 Mb/s or an incomplete transfer is logged as a
warning and startup continues. `hostToHeadsetMbps` is the completed bytes
divided by the full three-second wall-clock interval. Each curl request opens
a new connection, so this includes request and TLS setup. The optional
`hostToHeadsetResponseBodyMbps` instead divides those same completed bytes
by the sum of curl’s `time_total - time_starttransfer` intervals. It excludes
setup and gaps between requests; it is a response-body estimate, not sustained
goodput or USB line rate. Neither metric measures headset-to-host traffic or
CloudXR stream throughput. A faster root hub is a topology clue, not proof
that an accessible physical port is free.

## Re-open the client on the headset

If the headset browser is closed or navigated away, re-open the client from
another terminal without restarting the service:

```bash
python -m isaaccapture.cloudxr.webclient
```

This opens the versioned client over USB `adb` with this host’s `serverIP`
and `port` pre-filled. Pass a URL to override the target — one already
containing `oobEnable=` is opened verbatim, which is how to restore an OOB or
USB-local session from the URL the service printed. `--print-only` resolves
the URL without touching `adb`. Run with `--help` for the full argument
handling.

It only *opens* the page; accepting the certificate and clicking CONNECT remain
`--setup-oob`’s CDP automation.

<a id="load-cloudxr-environment-variables"></a>

## Load CloudXR environment variables

On every start the service writes the runtime’s resolved environment to
`~/.cloudxr/run/cloudxr.env` (the exact path is printed in the startup
output). Sourcing it points the OpenXR loader at CloudXR — it sets
`XR_RUNTIME_JSON` along with the `NV_CXR_*` variables — so any OpenXR
application started from that terminal connects to the running runtime.

Open a new terminal and source the setup script:

```bash
source ~/.cloudxr/run/cloudxr.env
```

#### IMPORTANT
Make sure to run the rest of the commands in the same terminal. If you have
to open a new terminal, source the CloudXR environment variables again.

Applications that embed `CloudXRLauncher` — the teleop examples — do this
for themselves and need no sourcing.

## Run teleop examples against the service

Nothing to pass. The examples under `examples/teleop` attach to
whatever runtime is serving, so with a service running they use it. Install
them once from the repository root, as in [6. Run a teleop example](../teleop/quick_start.md#run-teleop-example):

```bash
uv pip install -e ./examples/teleop
python -m isaaccapture.cloudxr.service start --accept-eula
python -m isaaccapture_examples.teleop.gripper_retargeting_example_simple
```

The example leaves the service running when it exits, so the headset stays
connected and the next run reattaches to the same session.

## Use a system OpenXR runtime (no CloudXR attach)

Examples that call `add_launcher_arguments()`
accept `--no-launch-cloudxr-runtime`. That returns a `NoopContext`:
the process does not start or attach to CloudXR and leaves `XR_RUNTIME_JSON` and
related environment variables unchanged. Use this when another runtime is already
configured (for example Monado) or when a host singleton must not be duplicated
(see `examples/robot_viz/README.md` and `--no-launch-cloudxr-runtime` there).

`OpenXRSession` refuses to start unless `NV_CXR_RUNTIME_DIR` and `XR_RUNTIME_JSON`
are both set; `ISAAC_TELEOP_DISABLE_CXR_ENV_CHECKS` skips that check for a runtime
other than CloudXR:

```bash
export XR_RUNTIME_JSON=/path/to/your/openxr.json
export ISAAC_TELEOP_DISABLE_CXR_ENV_CHECKS=1
python -m isaaccapture_examples.teleop.gripper_retargeting_example_simple --no-launch-cloudxr-runtime
```

## Run the service in a container or CI

Use `run`, never `start`. `start` detaches and returns, so as a
container’s main process it would exit immediately and take the runtime with
it; `run` stays in the foreground where signals reach it and the runtime’s
lifetime is the container’s. This is what `deps/cloudxr/runtime/entrypoint.sh`
does:

```bash
exec python -m isaaccapture.cloudxr.service run
```

Nothing can answer an EULA prompt without a terminal, so accept it up front —
either pass `--accept-eula` or pre-write the marker, as that image does:

```bash
mkdir -p ~/.cloudxr/run && printf 'accepted\n' > ~/.cloudxr/run/eula_accepted
```

When the application *is* the entrypoint, it can own the runtime in-process
instead of running a second one beside it — construct
`CloudXRLauncher(run_embedded=True)`, as the ROS 2 example node does. It
refuses to start where a runtime is already serving the install directory:
the options that configure the WSS proxy (`host_client`, `setup_oob`,
`usb_local`) only apply to a proxy the process starts itself. Attaching can
report a `host_client` mismatch, but it cannot change the running proxy.
Stop that service, or drop `run_embedded` to attach.

To let other containers attach, share the run directory as a volume and point
them at it with `XR_RUNTIME_JSON` and `NV_CXR_RUNTIME_DIR`;
`deps/cloudxr/docker-compose.test.yaml` mounts it read-only into the test
container that way.

Setting `CI` skips the interactive pause described under [Configuration](),
alongside the check for a non-interactive stdin, so automated runs never wait.

<a id="cloudxr-deferred-oob-connect"></a>

## Connect the OOB headset after application startup

Set `defer_oob_connect=True` on `CloudXRService` or `CloudXRLauncher`
to delay OOB automation and WebSocket signaling. Call `connect_headset()`
after the OpenXR session and renderer are ready.

Requires `setup_oob=True` and a fixed `NV_DEVICE_PROFILE` (not `auto*`).
For `CloudXRLauncher`, also set `run_embedded=True`. The default `False`
keeps immediate connection.

## Configuration

The service accepts the same configuration options as the launcher embedded in
an application:

- `--cloudxr-env-config <PATH>` — a `KEY=value` env file of CloudXR
  runtime overrides, e.g. `NV_DEVICE_PROFILE=auto-native`. See
  [3. Configure CloudXR (optional)](../teleop/quick_start.md#run-cloudxr-server) in the quick start for the list of supported
  environment variables.
- `--cloudxr-install-dir <PATH>` — CloudXR install directory
  (default: `~/.cloudxr`).

These configure a runtime as it starts, so they belong to the service that owns
it. An application that attaches to a running runtime cannot apply them: it
compares the requested settings against the runtime’s resolved environment,
reports the ones that would have changed, and uses the running configuration.

`--host-client` / `--no-host-client` follow the same rule. They apply when
the launcher starts a service. When attaching, the launcher recovers the
detached service flags from its command line and reports a mismatch when the
running hosting differs. A foreground service leaves no recoverable flags, so
that case stays quiet. Restart the service to change the hosting.

A launcher requesting `--setup-oob` or `--usb-local` checks the mode of an
already running detached service before attaching. A mismatch fails with a
stop-and-restart instruction; the launcher cannot enable OOB or USB-local mode
on a service that is already running. A foreground service has no recoverable
launch flags, so a requested OOB/USB mode also fails until the service is
restarted in a verifiable mode. Plain attaches without a requested OOB/USB
mode continue to use the running service.

```text
./custom.env is ignored: the CloudXR runtime already serving this host was started with its own configuration.
  NV_DEVICE_PROFILE: auto-native requested, Quest3 in effect
  Restart the service to apply it:
    python -m isaaccapture.cloudxr.service stop
    python -m isaaccapture.cloudxr.service start --cloudxr-env-config ./custom.env
  Continuing with the running configuration in 5s — press any key to abort.
```

Only settings that actually differ are listed, so passing the same
`--cloudxr-env-config` on every run stays quiet. Restart the service as the
message says to apply a change.

The five-second hold keeps the notice from scrolling away under a chatty
application; press any key to stop there instead. It is skipped when nothing
could be watching — a non-interactive stdin, or `CI` set — so container
entrypoints and automated runs continue immediately rather than waiting.

On Jetson Orin the service already selects the experimental runtime package
and main-thread join. The following are Isaac Capture launcher overrides only
(not CloudXR native settings). Export them in the process environment, or add
them to `--cloudxr-env-config` if you need them:

#### Launcher overrides

| Variable                         | Default              | Description                                                                                                                                                                                                                                                                  |
|----------------------------------|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ISAAC_TELEOP_CLOUDXR_EXP`       | unset (auto on Orin) | Select the experimental package (`isaaccapture.cloudxr_exp`).<br/>`ISAAC_TELEOP_CLOUDXR_EXP=0` forces the stable runtime on Orin;<br/>`ISAAC_TELEOP_CLOUDXR_EXP=1` forces the experimental package elsewhere<br/>(launcher fails if it is missing).                          |
| `ISAAC_TELEOP_CLOUDXR_JOIN_MAIN` | unset (auto on Orin) | Join the CloudXR service on the main thread (avoids a<br/>`Couldn't create autoTSSkey mapping` abort on some platforms).<br/>`ISAAC_TELEOP_CLOUDXR_JOIN_MAIN=0` forces worker-thread join on Orin;<br/>`ISAAC_TELEOP_CLOUDXR_JOIN_MAIN=1` forces main-thread join elsewhere. |

To inspect the active settings after startup:

```bash
cat ~/.cloudxr/run/cloudxr.env
```

## Troubleshooting

<a id="cloudxr-web-client-refresh"></a>

### PICO 4 Ultra keeps failing when retrying a connection

After a failed connection, the PICO browser can require a page refresh before
another attempt works. Clicking **Connect** again without refreshing can keep
failing.

In the web client’s **Troubleshooting** settings, **Auto Refresh on Session End**
defaults to **Refresh on any session end (including errors)** for the **Pico 4
Ultra** device profile. This reloads the page after a clean disconnect or a
connection failure, once any automatic reconnect attempts have been exhausted.
Other device profiles default to **Never refresh**. You can also choose
**Refresh on clean session end** to reload only after a clean disconnect.
Changing this setting switches the device profile to **Custom** and saves your
choice; selecting a named device profile again restores that profile’s defaults.
Selecting **Custom** preserves the current settings.

The last error message, including its hexadecimal error code when supplied by
CloudXR, remains visible after automatic or manual refreshes in the same browser
tab. It clears when a stream connects successfully. This requires browser session
storage to be available. A restored error does not itself trigger another refresh;
click **Connect** to retry from the refreshed page.

<a id="cloudxr-form-factor-unavailable"></a>

### `xrGetSystem` fails with `XR_ERROR_FORM_FACTOR_UNAVAILABLE` (-35)

-35 is returned *after* `xrCreateInstance` succeeded, so the runtime was
found and loaded — a missing runtime gives -51 (`XR_ERROR_RUNTIME_UNAVAILABLE`)
instead. It means the runtime is up but no headset session is attached to it
yet. Check, in order:

1. A client is connected. The headset must have loaded the web client and
   clicked CONNECT; until then there is no system to return.
2. The device profile matches the device. `python -m isaaccapture.cloudxr.service
   status` prints the profile the running service was started with.
3. The application is talking to the runtime you think it is. Applications that
   do not embed `CloudXRLauncher` need `source ~/.cloudxr/run/cloudxr.env`
   first, so that `XR_RUNTIME_JSON` and `NV_CXR_RUNTIME_DIR` point at it.

`OpenXRSession` does not raise it: `wait_for_system` defaults to true, so it
logs `OpenXR HMD form factor is unavailable; waiting for a system...` once and
retries every second until one appears. That is what lets you start an app before
putting the headset on — and an app that appears to hang at start-up is usually
sitting in this wait, so work the list above. Nothing times it out, and in Python
`Ctrl-C` is not delivered until a system turns up — the binding holds the GIL
through the wait. C++ callers can pass
`wait_for_system=false` for the error instead; the Python binding does not
expose it. `VizSessionConfig` is unaffected and stays fail-fast at
`xr_system_wait_seconds = 0`.
