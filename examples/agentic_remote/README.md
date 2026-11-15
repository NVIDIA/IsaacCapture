<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agentic remote execution: the robot app and the robot connection

This example builds the design in [`docs/design/claude_agentic_remote_execution.md`][design].
In that design an AI agent plans arm motion on a GPU machine (the **workstation**), and one small
program on the Jetson Thor that sits on the robot (the **Thor**) is the only one that talks to the
motors. That program is the **robot app**. The two machines exchange **link messages**: one unit
sent on a connection, such as a move or the robot's state. The design is built in phases P0 to
P4. This folder holds P0, the robot app and the link messages on a simulated arm, and the first
part of P1: the connection between the two machines and the workstation program that holds its
end. Nothing here has moved a real arm.

## Terms

- **Setpoint**: one target value per joint. **Move**: a list of timed setpoints. **Hold**: the
  arm stays where it is, motors on. **Rest**: the folded pose in which it lies on its own stops.
- **Control owner**: who may move the arm right now: the agent, a human, recovery (the robot app
  itself, for a return to rest) or nobody.
- **Session id**, **owner number**, **stop number**: a random value picked at each start, and two
  counters that go up at every change of control owner and at every STOP. Every command quotes
  them, and a command that quotes an old value is refused.
- **Checker**: the code that accepts or refuses each setpoint just before it is sent.
- **Permit**: a short-lived permission from the workstation to send the next part of a move.
  Without a valid permit the arm does not advance.
- **STOP**: ends the motion under way and makes nobody the control owner; the arm holds.
  **Fault-stop**: the same, and nothing more is accepted until a person resets it.
- **Assistance wait**: how long the arm holds for a person after the agent asked for help
  (120 s, never below 90 s). At its end the robot app may make one attempt to return to rest.
- **Carrier**: the object that carries link messages over one connection. It has three calls:
  send reliable (every link message arrives, in the order sent), send newest-wins (a newer link
  message of the same type replaces one that was not received yet) and receive. **Loopback**: a
  carrier whose two ends are in one process, with no network between them.
- **Port**: a set of calls the robot app makes without knowing what answers them: the arm port
  (motors, joint reads, jaws), the checker and the contact watch that reads the joints during a
  move (`interfaces.py`). The simulated arm and APAI each stand behind the same ports.
- **Jaws**: the gripper's two fingers. **Jaws proven empty**: the jaws were referenced (closed
  once to find their zero), their readings agree and lie at the last jaw target, that target is
  full open or the jaws are closed, and no close has ended short of its target since the jaws
  were last measured full open.
- **Operator console**, **live view**: two workstation programs of later phases. At the console
  the operator types `stop`, `rest`, `release` and the other commands; the live view shows the
  camera picture in a browser.
- **Robot connection**: the one connection between the robot app and the workstation that
  carries the link messages above. The robot app dials it; a **dial** is one attempt to open it.
- **Robot link service**: the workstation program that holds the workstation end of the robot
  connection. **Local API**: what it offers other programs on the workstation, over a Unix
  socket. Such a program is a **local client**, and it says in which **role** it attaches:
  agent, console, viewer or tool.
- **Peer timeout**: a link that receives nothing for 3 s is dead. Each end sends a heartbeat
  every 0.5 s when it has nothing else to send.

## What this folder contains

The code is the Python package `isaaccapture_examples.agentic_remote`. The file names below are
inside it; [Layout](#layout) shows where.

- The link messages (`protocol/messages.py`) and the three calls everything above a connection
  uses: send reliable, send newest-wins, receive (`protocol/carrier.py`), on a loopback and, with
  Isaac Capture built, on a real link (`protocol/link_carrier.py`).
- The robot connection: the robot app starts without one, loses it and takes a new one, and a
  dialer keeps dialing the workstation (`robot_app/connection.py`).
- The robot link service and its local API (`link_service/`).
- The robot app: commands admitted by session id, owner number and stop number; moves paced at
  one setpoint per 20 ms and gated by permits; commands for the jaws; STOP and fault-stop; the
  assistance wait; the return to rest; the guarded torque release; the state it publishes.
- A simulated arm and a simple joint-space checker (`sim.py`); they need numpy and nothing else.
- The same ports on APAI, the agent stack of the design (`apai_backend.py`): its checker and arm
  drivers, unmodified, tested on APAI's software arm and on a stand-in for the motor library.
- The Thor terminal, where an operator types at the robot (`terminal.py`): `stop`, `rest` (asks
  for a typed `yes` that the path is clear), `release` (motors off, only at rest), `quit` (return
  to rest, motors off there, exit; it asks the same `yes` when the arm has to move) and `help`.
  Its `install_signal_handlers()` makes Ctrl+C a STOP; the self-tests do not call it, the
  program started with `--connect` does.

## What it does not contain

- No agent, no operator console and no live view yet. A test client plays their parts on the
  local API.
- No camera and no perception. Permits come from the test client, not from an obstacle check.
- No headset and no human driving: a request to make a human the control owner is refused.
- No arm motion on hardware. Nothing in this folder has run on a robot, and every value in
  `configs/robot_app.yaml` is a demo value for the simulated arm. `--arm apai --driver real`
  asks its three confirmations and stops before anything is connected (see
  [The real driver](#the-real-driver)). With the shipped file `--arm apai` stops at the human
  limit set: its 1.5 rad/s is above the speed cap of APAI's profiles (see
  [What a move does](#what-a-move-does)).
- The robot connection is not encrypted and the peer proves nothing but its address: the
  service takes the robot connection only from the configured addresses. Both machines are
  meant to be on one trusted network.

## The robot connection

Two programs, one on each machine:

| Program | Machine | Started with |
|---|---|---|
| Robot link service | Workstation | `python -m isaaccapture_examples.agentic_remote.link_service --config configs/link_service.yaml` |
| Robot app | Thor | `python -m isaaccapture_examples.agentic_remote.robot_app --config configs/robot_app.yaml --arm sim --connect HOST:PORT` |

Both need Isaac Capture built and on the Python path (`isaaccapture.remote_link`); without it
each says so and exits with code 2. Either can be started first.

**The robot link service** listens on `listen.address` and `listen.port` of its file (the
shipped file: 127.0.0.1 and port 47610, for both programs on one machine) and takes the robot
connection only from an address in `allowed_peers`. One robot connection at a time: while one
is alive a second is closed at once. It prints where it listens and where its local API is,
and runs until Ctrl+C or SIGTERM. Closing it does not stop the robot; the Thor holds on its
own. A second service on the same socket prints `a robot link service already runs on ...`
and exits with code 1; the socket file of a service that died is replaced.

**The robot app** with `--connect` starts on its arm, serves the Thor terminal on its standard
input and dials the service. While nothing answers it dials again every `connection.retry_s`
(1 s, never below 0.2 s) and says once why: `robot app: no robot connection: ...`. The arm
port is connected once, when the app starts, whether or not the workstation answers. Ctrl+C is
a STOP and never ends the program; `quit` ends it (return to rest, motors off there); SIGTERM
closes it with the motors left on. Its exit code is 0 after a `quit`, a SHUTDOWN or SIGTERM,
1 when the app could not start, and 2 when nothing ran.

### The local API and its roles

A local client opens the Unix socket (`socket_path` of the service's file; by default
`$XDG_RUNTIME_DIR/agentic_remote/robot_link.sock`, else `/tmp/agentic_remote-<uid>/robot_link.sock`,
in a folder of mode 0700), says LOCAL_HELLO with its role and from then on sends and receives
the same link messages as on the robot connection. In Python `connect_local(path, role, name)`
(`link_service/local.py`) gives a carrier, so the code above it is the same on both sides.

| Role | Who | May send | At a time |
|---|---|---|---|
| `agent` | The agent program's arm adapter | MOVE, GRIPPER, PERMIT, CHECKER_STATE, STOP, and OWNER_REQUEST "agent", "help_needed", "nobody" | One |
| `console` | The operator console | STOP, RESET, RELEASE, OPEN, SHUTDOWN, and OWNER_REQUEST "rest", "nobody" | Several |
| `viewer` | The live view program | VIEW_STATUS, STOP | Several |
| `tool` | Anything that only reads | STOP | Several |

- **Anything else is refused by the service** and never reaches the Thor: with a RESULT
  `refused` under the command's id, or with a LOCAL_NOTICE when there is no id. Both name the
  role and the link message.
- **Every role may send STOP**, and it goes out at once. From then until a ROBOT_STATE shows
  a higher stop number the service drops PERMITs and refuses a MOVE or GRIPPER that quotes the
  old stop number.
- **To every client**: WELCOME, ROBOT_STATE (the newest only), EVENT, and LOCAL_STATUS: whether
  the robot connection is up, why the last one ended, the session id, the Thor's clock against
  the workstation's, and how many clients are attached in each role. **RESULT** goes to the
  client that sent the command and to every console.
- **When the agent client's socket closes while the agent is control owner**, the service
  sends OWNER_REQUEST "help_needed" with the reason `agent program gone`: the arm holds, the
  agent's ownership ends and the assistance wait starts. It is the only link message the
  service makes itself.
- **A client that does not read its socket** is dropped; it never holds up the robot
  connection or another client.

### When the robot connection is lost

The robot app notices at once when the workstation closed the connection or its process ended,
and after the peer timeout (3 s) when the workstation or the network went silent. The service
notices the same way.

- **The Thor holds, by itself.** A running agent move, a timed return to rest and a return to
  rest or a release that was asked for over the connection end in a hold with the motors on.
  If the agent owned the arm, its ownership ends: control owner nobody, owner number up by one.
  A pending assistance wait is cancelled. `rest`, `release` and `quit` still work at the Thor
  terminal.
- **The service answers for the Thor.** Every command it forwarded that has no final RESULT
  gets `failed: robot connection lost`; a command sent meanwhile is `refused` with the same
  reason; a STOP that cannot be delivered is answered with a LOCAL_NOTICE. Local clients stay
  attached and LOCAL_STATUS tells them.
- **The robot app dials again**, not sooner than `connection.retry_s` after the loss. On the
  new connection it sends WELCOME with the same session id and a ROBOT_STATE.
- **Nothing is replayed and nothing restarts.** What was queued for the old connection is gone.
  A new connection starts no move, no wait and no return to rest: the arm holds until a new
  task or an operator's command. A command that quotes the owner number from before the loss
  is refused; a command id the Thor has already answered is answered again from its store and
  not run again.
- **A restarted service changes nothing on the Thor**: the robot app dials it like any other.
  Local clients have to attach again.
- **A restarted robot app has a new session id.** Every command that quotes the old one is
  refused. The motors keep holding while the app is down only as far as the drivers do; see
  section 10 of the design.

### The real driver

Connecting APAI's driver for a real arm cuts motor torque for a moment (see
[the torque rule](#the-rule-torque-is-never-cut-away-from-rest)). So
`--arm apai --driver real --apai-profile NAME --calibration NAME` first asks three questions
at the Thor terminal, each answered with a typed `yes`: is the arm powered; is it at rest and
supported; is the calibration in use the one named with `--calibration`. Any other answer, or
the end of the input, ends the program with `not confirmed: nothing was connected`. After
three confirmations this phase prints `hardware runs start in phase P2` and stops as well:
exit code 2 in both cases, and no connection of any kind was opened. `--driver mock` (the
default) is APAI's software arm and asks nothing; a profile that drives hardware is refused
without `--driver real`.

## What the robot app does with what arrives from the link

A sender can be wrong or flood the link, so the robot app trusts nothing that arrives:

- **A STOP is served first.** The app takes everything that waits off the link before it handles
  anything, and serves a STOP the moment it sees one. The commands the STOP passed are handled
  afterwards, in their order, and refused because they quote the old stop number. A STOP with a
  payload that cannot be read is still a STOP.
- **The backlog is bounded**: at most 65536 link messages and 8 MiB of payload wait to be
  handled. A command that finds no room is refused at once with the reason `busy: ...`.
- **Sizes.** A payload is parsed only up to its size limit: 16 kB, and for a MOVE what its
  longest move needs (about 590 kB with the demo settings: 3000 setpoints of six joints). A MOVE
  with more setpoints than the Thor allows is refused before its numbers are converted. Link
  messages the robot app has no use for (WELCOME, RESULT, EVENT, ROBOT_STATE) are dropped unread.
- **Every command gets one final RESULT**, also one that cannot be decoded: it is refused with
  what the decoder found, under its command id. A link message that is no command (PERMIT,
  VIEW_STATUS, an unknown type) is dropped and logged.
- **Text is cut.** A command id or move id is at most 128 printable characters; a command with
  another id is refused under the id cut to that length. A reason is cut to 256 characters, a
  sender, request, action or mode to 64, and characters that cannot be printed become spaces
  before the text reaches a RESULT, an EVENT, ROBOT_STATE, the log or the Thor terminal.
- **A refused permit is named.** A MOVE refused for want of a permit says why its last permit
  was refused, for example `permit evidence is in the future` when the two clocks disagree.
- **Repeats are counted, not written.** A log line that a link can make repeat, and a refusal
  line at the Thor terminal, is written once per second with the number of its repeats.
- At most four jaw commands wait behind the one that runs; a further one is refused as `busy`.
  The results of the last 256 commands are kept, and a command that has not ended is never
  among those forgotten.

## What the operator's commands do

`rest`, `release`, a shutdown, the guarded jaw opening, a reset and every ownership request come
from the console as link messages or are typed at the Thor terminal.

- **They quote the owner number too.** OWNER_REQUEST, RESET, RELEASE, OPEN and SHUTDOWN carry the
  owner number and the stop number their sender last saw, like the agent's commands. One built
  before the control owner changed is refused: a `yes` typed for one state of the arm does not
  move it in another.
- **A shutdown that has to move the arm needs the path confirmed**, like a typed `rest`: SHUTDOWN
  carries `confirmed`, and `quit` at the Thor terminal asks for a typed `yes`. Unconfirmed it is
  refused with `confirm the path is clear` and the app keeps running. At rest, or with the motors
  already off, nothing is asked. The pose is measured again when the shutdown runs: if the arm
  is then found away from rest, an unconfirmed shutdown moves nothing.
- **Refused means nothing changed.** A return to rest is `refused` while its conditions are
  checked. Once recovery has taken the arm, any early end is `failed` or `cancelled`: the arm
  holds, and an agent that owned the arm no longer does.
- **A typed `rest` at the Thor terminal goes on when the agent's ownership ends** while it is
  checked (a help request, or the robot connection lost). A STOP, a fault-stop or a new owner
  still refuses it.
- **A release or a shutdown that came over the link does not outlive the link.** If the robot
  connection is lost before the motors are switched off, they stay on. The same commands typed
  at the Thor terminal go on.
- **The timed return is tried when the assistance wait runs out, or not at all.** If a return to
  rest, a release or a shutdown is running or queued at that moment, the EVENT
  `no timed return to rest: busy: ...` goes to both terminals and the arm is not moved later.
- **A hold at a contact ends when the arm is next moved**: by any later move that sent a
  setpoint and did not itself end at a contact. Until then `rest` is refused.
- **At rest at start has its own tolerance**, `rest.start_rest_tol_rad`: an arm that lies folded
  with its motors off is not as close to the rest pose as a powered return leaves it. Within
  that tolerance the jaws are referenced. The release keeps `rest.rest_tol_rad` (0.05 rad): an
  arm that starts outside it is reported as holding, and a typed `rest` brings it inside. With
  APAI the start tolerance is the arm profile's `rest_check_tol_rad`
  (`apai_backend.config_for_profile()`).

## What the jaws do

- **A close that ends short of its target is left at the light holding stiffness**, whether the
  GRIPPER command was `plain` or `close_until_contact`: the jaws stalled on something, or ran
  out of time. A plain close that stalls still ends `failed`; only the stiffness is the same.
- **A STOP ends an opening too.** An opening under way ends within one pass of the driver's
  guarded opening (50 ms on the reBot gripper) and the jaws are held lightly where they are,
  not at the opening that was asked for. The RESULT is `cancelled` and carries the opening
  measured after the stop. The same holds for a fault-stop, a help request, the agent giving
  the arm back and a lost robot connection.
- **A missed jaw answer is not a fault.** If the driver cannot read the jaws before an opening
  and has sent them nothing, the command is tried up to three times and then `refused`. A jaw
  fault after a command went out (blocked while opening, feedback lost) is still a fault-stop.
- **The jaws have three tolerances**, as fractions of their travel: `gripper.target_tol` for
  "jaws proven empty" (a reading against the last jaw target), `gripper.reach_tol` for a
  command that reached its target, and `gripper.full_open_tol` for the measured full open that
  clears "may be holding". The design gives the first as 0.1 rad: with APAI
  `config_for_profile()` sets it to 0.1 rad of the profile's jaw travel, or leaves a smaller
  value.
- **A command that follows a jaw command's RESULT does not find the jaws busy**: a `rest`, a
  `release` or a shutdown sent the moment that RESULT arrives is checked, not refused with
  `a jaw command is running`.

## What a move does

- **The first setpoint is checked from the measured pose**, not from the start pose the MOVE
  claims. A MOVE whose first setpoint is farther from where the arm is than one step at its
  owner's speed is refused, however close its claimed start is. The return to rest is planned
  again from the pose the executor itself reads, so two reads of a still arm that differ a
  little do not refuse it.
- **What held before the check is asked again before the send.** If the check took time, the
  STOP, the permit and the large delay are tested once more; a setpoint whose permit ran out
  meanwhile is not sent. A send that itself took longer than the large delay ends the move
  `late`. Not covered: the send call itself, between its start and the motors.
- **The cost of the check and the send does not stretch a move.** A late wake-up moves the
  later setpoints back, and so does a check and send that took longer than the small delay;
  what they usually cost does not. Two setpoints are never closer than one period less the
  small delay (15 ms with the demo settings), and none is skipped or sent sooner to catch up.
- **A move has the bus to itself.** The idle joint reads stop before a move reads its start
  pose and resume when it has ended; one under way is waited for. In a hold one joint is read
  every `state.idle_read_period_s` whatever the wait for a link message took, and a joint
  whose read failed is read once more at once: one or two reads without an answer are no
  fault-stop, a joint that stays silent for 0.35 s is.
- **A grasp region is bounded by the Thor's settings**, `grasp.max_radius_m` and
  `grasp.max_depth_m` (how far below the table plane it may reach). Its centre must lie in the
  checker's workspace box and its ceiling, if it has one, above its floor. A MOVE with a
  region beyond that is refused before it is accepted, never cut down, and is not kept as the
  last grasp region.
- **The hold after a contact uses a reading the contact watch trusts.** With APAI the watch
  runs APAI's own hold: one fresh read of every joint, sent as the target only if it agrees
  with the newest samples of the move. If it does not, or cannot be read, nothing is sent,
  the last setpoint stands and the RESULT says so (`contact; the hold at the measured pose
  failed: ...`). It is not a fault-stop.
- **A limit set faster than the APAI profile's speed cap is refused** when the checkers are
  built (`safety.max_joint_vel`: 0.8 rad/s for `rebot_rs`, 1.2 for `mock`). Nothing is clamped.
- **What the reBot motor library does not report.** A command that reaches no motor raises
  nothing there. A lost setpoint shows as a contact once the arm no longer follows, or as
  lost joint feedback; a lost light jaw hold or a lost release shows nowhere. So a release on
  that driver is reported as `torque release sent at rest: the driver does not confirm that
  the motors are off`, and ROBOT_STATE's `torque_on` is then what was commanded.

## Layout

The example follows [`examples/README.md`](../README.md): `pyproject.toml` at its root, the
package under `python/isaaccapture_examples/`, and relative imports inside the package.
`isaaccapture_examples/` is a namespace that every example shares, so it has no `__init__.py`.

```text
examples/agentic_remote/
  pyproject.toml              the distribution, isaaccapture-examples-agentic-remote
  configs/robot_app.yaml      settings: limits per control owner, timing, rest pose, dialing
  configs/link_service.yaml   settings of the robot link service
  python/isaaccapture_examples/agentic_remote/
    protocol/
      messages.py carrier.py             link messages and the carrier; on both machines
      link_carrier.py local_messages.py  the carrier on a real link; the local API's own messages
    robot_app/
      interfaces.py clock.py config.py   ports, clocks, settings
      authority.py permits.py            control owner and the three numbers; permits
      executor.py gripper.py             moves; jaw commands
      stop.py waits.py rest.py           STOP and fault-stop; assistance wait; rest and release
      state.py app.py connection.py      the published robot state; all of it wired; the dialer
      inbox.py text.py                   the backlog off the link; text cut and lines counted
      terminal.py __main__.py            the Thor terminal; the command line and the self-tests
      sim.py apai_backend.py             simulated arm and checker; APAI behind the same ports
    link_service/
      service.py local.py                the service; the local API's socket and its client end
      listen.py config.py __main__.py    where the robot app dials in; settings; the command line
    console_log.py            the console log level of both programs
    testing/client.py         a workstation-side client for the tests and the self-tests
tests/python/examples/agentic_remote/    the tests
```

## Run the self-tests

There are two, both on the simulated arm, which runs on the machine's clock. Each prints one
line per step.

`--selftest` drives the robot app over a loopback link with a scripted client, and at a
scripted Thor terminal: take ownership, a move with permits, a move without a permit (refused),
a STOP during a move, a new task after it with a move that quotes the stop number from before
the STOP (refused for that number), a help request that starts the assistance wait, a typed
`stop` that ends the wait, a typed `rest`, a release and a typed `quit`. About 5 s. It needs
no Isaac Capture.

`--selftest-link` runs, in one process, a robot link service on 127.0.0.1 and a free port with
its socket in a temporary folder, the robot app dialing it over a real link, and two scripted
local clients, an agent and a console: take ownership, a move with permits, a move without a
permit (refused), a STOP from the console during a move, a new task, the robot connection cut
during a move (the arm holds, the move's RESULT is `failed: robot connection lost`), the dial
that follows (the same session, nothing moved, the agent's old command refused), `rest` and
`release` from the console, and the robot app closed. About 6 s. Without Isaac Capture on the
Python path it prints one line that says so and exits with code 2.

Both need Python 3.11 to 3.13; installing the example brings numpy and pyyaml. From the
repository root, with a virtual environment active (for the second, one in which
`import isaaccapture.remote_link` works):

```bash
uv pip install -e ./examples/agentic_remote
python -m isaaccapture_examples.agentic_remote.robot_app \
    --config examples/agentic_remote/configs/robot_app.yaml --arm sim --selftest
python -m isaaccapture_examples.agentic_remote.robot_app \
    --config examples/agentic_remote/configs/robot_app.yaml --arm sim --selftest-link
```

Exit code 0 means every step passed, 1 that a step went wrong (its line says which and why), and
2 that nothing ran: unusable arguments, Isaac Capture missing where it is needed, `--arm sim`
with none of `--connect`, `--selftest` and `--selftest-link`, or the real driver.
`--log-level debug` shows the app's log.

## Run the two programs on one machine

In one terminal the robot link service, in a second the robot app on the simulated arm. The
shipped service file listens on 127.0.0.1, port 47610, and puts the local API at its default
path:

```bash
python -m isaaccapture_examples.agentic_remote.link_service \
    --config examples/agentic_remote/configs/link_service.yaml
python -m isaaccapture_examples.agentic_remote.robot_app \
    --config examples/agentic_remote/configs/robot_app.yaml --arm sim --connect 127.0.0.1:47610
```

The service prints `robot link service: waits for the robot app on 127.0.0.1:47610 ...`, the
robot app `robot app: session ...; dialing the robot link service at 127.0.0.1:47610` and then
`[connection] ...` at its Thor terminal. Type `help` there. No program that sends tasks exists
yet: a local client is three lines of Python (`connect_local()` and `testing/client.py`), as in
`tests/python/examples/agentic_remote/test_link_end_to_end.py`. For two machines, put the
workstation's own address in `listen.address`, the Thor's in `allowed_peers`, and give the
robot app the workstation's address.

With APAI's software arm in place of the simulated one (`AGENTIC_REMOTE_APAI_SRC` set, and a
robot app file whose limit sets fit the profile's speed cap), and the real driver's questions:

```bash
python -m isaaccapture_examples.agentic_remote.robot_app --config my_robot_app.yaml \
    --arm apai --driver mock --connect 127.0.0.1:47610
python -m isaaccapture_examples.agentic_remote.robot_app --config my_robot_app.yaml \
    --arm apai --driver real --apai-profile rebot_rs --calibration NAME
```

Which of these commands were run, and how, is in [What was run](#what-was-run).

## Run the tests

The tests do not need the example installed: their `conftest.py` puts
`examples/agentic_remote/python` on the import path. The test folder has its own `pyproject.toml`;
`uv run --extra dev pytest -q` there creates `.venv` on first use, which is how the repository's
CTest suite runs the tests. The commands below use that environment. From the repository root:

```bash
cd tests/python/examples/agentic_remote
.venv/bin/python -m pytest -q                    # the whole folder
.venv/bin/python -m pytest -q test_terminal.py   # one file; every file runs alone
```

No test needs a robot, a GPU or a second machine, and nearly all run on a fake clock;
`test_main.py` runs the self-test on the machine's clock and takes some seconds.

The tests marked `link` need Isaac Capture's remote link and are skipped without it. They open
TCP connections to 127.0.0.1 only, on ports the system picks, and Unix sockets in pytest's
temporary folders; `test_link_programs.py` also starts the two programs as processes of their
own and ends them with signals. With an Isaac Capture build on the Python path:

```bash
PYTHONPATH=/path/to/build/python_package/Release .venv/bin/python -m pytest -q
PYTHONPATH=/path/to/build/python_package/Release .venv/bin/python -m pytest -q \
    test_link_end_to_end.py test_link_programs.py
```

The tests that use APAI are skipped unless two things are there: an APAI checkout, with
`AGENTIC_REMOTE_APAI_SRC` pointing at its `src` folder, and the `apai` extra of the test
folder's `pyproject.toml`, which holds what APAI imports. The summary line of pytest counts
them as skipped, and `-rs` shows which of the two is missing. In the same folder, the first
command installs the extra into `.venv` and runs the tests, the second runs them again:

```bash
AGENTIC_REMOTE_APAI_SRC=/path/to/apai/src uv run --extra dev --extra apai pytest -q
AGENTIC_REMOTE_APAI_SRC=/path/to/apai/src .venv/bin/python -m pytest -q
```

The same variable lets `--arm apai` find APAI.

## The rule: torque is never cut away from rest

The arm has no brakes: with the motors off it falls, unless it lies at rest. So the robot app
switches the motors off through one call, the arm port's `release_torque()`, and makes that call
in one place, the guarded release in `robot_app/rest.py`. With APAI the call is
`ApaiArm.release_torque()`, which disconnects APAI's driver: the second and last place in the
package that cuts torque. The guarded release runs only on an operator's request (`release`, or
the shutdown that `quit` asks for), and only when two full joint reads 0.2 s apart are within
0.05 rad of rest and of each other, the jaws are proven empty, no move is running and no
fault-stop is latched.

One thing outside the rule: connecting APAI's driver for the real arm cuts torque (its
`connect()` clears motor faults with a command that does so). The robot app connects once, when
it starts, and never again: a lost or a new robot connection does not touch the arm port. The
arm has to lie at rest and be supported at that moment, which is what the three typed
confirmations of [The real driver](#the-real-driver) ask. No code here connects that driver to
hardware yet.

Everything else leaves the motors on: STOP, a fault-stop, a refused or interrupted return to
rest, the end of the assistance wait, a lost or a new robot connection, a restarted robot link
service, Ctrl+C, SIGTERM, a closed terminal and closing the robot app. Tests keep it that way:
the simulated arm records the pose at every torque
release and a test fixture fails any test in which one happened away from rest, and
`test_wiring.py` and `test_no_limp.py` scan the source for any other place that could cut
torque. This is a rule about the code; sections 9 and 10 of the design list what software
cannot prevent.

## What was run

On the workstation, 2026-11-14, with an Isaac Capture build on the Python path:

| Command | Run as | Result |
|---|---|---|
| `robot_app --arm sim --selftest` | As written | Exit code 0, 11 steps |
| `robot_app --arm sim --selftest-link` | As written | Exit code 0, 12 steps; exit code 2 and one line without Isaac Capture |
| `link_service --config configs/link_service.yaml` | Not with the shipped file. Run as a process by the tests with a file that names port 0 and a socket in a temporary folder; with the shipped file only without Isaac Capture | Exit code 0 on SIGINT and on SIGTERM, 1 for a second service on the socket; exit code 2 and one line without Isaac Capture |
| `robot_app --arm sim --connect 127.0.0.1:47610` | Not on port 47610. Run as a process by the tests against the port the service picked; with port 47610 only without Isaac Capture | Exit code 0 after `quit` and after SIGTERM; Ctrl+C is a STOP; exit code 2 and one line without Isaac Capture |
| `robot_app --arm apai --driver mock --connect ...` | By a test, in the test process, against the port the service picked, with the human limit set at 0.8 rad/s | Exit code 0 after a SHUTDOWN at rest |
| `robot_app --arm apai --driver real --apai-profile rebot_rs --calibration bench-7` | As written, with the answers piped in | Exit code 2: `hardware runs start in phase P2` after three `yes`, `not confirmed` otherwise; nothing was connected |

Not run: either program with the shipped service file's port 47610 and default socket path,
anything between two machines, and anything on a real arm.

## Next phases

| Phase | What is built |
|---|---|
| P1 | The rest of it: perception on the workstation |
| P2 | Demo 1: the agent completes a task; live view and operator console |
| P3 | Demo 2: a person takes over with the headset when the agent asks for help |
| P4 | Every fault under each control owner, measured delays, launch and recovery instructions |

[design]: ../../docs/design/claude_agentic_remote_execution.md
