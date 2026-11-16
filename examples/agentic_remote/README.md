<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agentic remote execution: the robot app, the robot connection and the agent side

This example builds the design in [`docs/design/claude_agentic_remote_execution.md`][design].
In that design an AI agent plans arm motion on a GPU machine (the **workstation**), and one small
program on the Jetson Thor that sits on the robot (the **Thor**) is the only one that talks to the
motors. That program is the **robot app**. The two machines exchange **link messages**: one unit
sent on a connection, such as a move or the robot's state. The design is built in phases P0 to
P4. This folder holds P0, the robot app and the link messages on a simulated arm, and P1: the
connection between the two machines, the workstation program that holds its end, and the agent
side on the workstation, which runs the agent with its obstacle check and drives the arm through
that connection. Nothing here has moved a real arm.

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
- **APAI**: the agent stack the design builds on. It is not part of this repository;
  `AGENTIC_REMOTE_APAI_SRC` names the `src` folder of a checkout. It has a planner, a checker,
  arm drivers and perception. **Task**: one line the operator types for the agent, such as
  `pick up and place the red object`. **Skill**: one step APAI runs for a task, such as
  `pick_and_place` or `open_gripper`.
- **Agent program**: the workstation program that runs APAI's agent and sends the arm's
  commands to the robot link service in the role `agent`. **Arm adapter**: what it gives APAI
  in place of a motor driver; each move APAI asks for becomes one MOVE and its permits.
- **Camera frame**: one image of the camera with its depth. **Stereo pair**: the left and the
  right image of one exposure; depth is computed from the two.
- **Obstacle grid**: APAI's 3D map of what the camera sees. **Workstation checker**: APAI's
  checker in the agent program; it checks each setpoint against the obstacle grid. **Check
  ahead**: the workstation checker run over the next setpoints of a move before the arm is
  sent them.
- **Evidence time**: when the camera frame behind the obstacle grid was exposed, on the Thor's
  wall clock. Every permit quotes it, and the Thor refuses a permit whose evidence is too old.
- **Self mask**: the arm's own pixels are removed from a depth image before it goes into the
  obstacle grid, with the arm pose at the time the camera frame was exposed.
- **Camera sender**, **camera receiver**: the two programs of the camera streamer, an example
  of a separate pull request: the sender runs on the Thor, the receiver on the workstation.
  `AGENTIC_REMOTE_CAMERA_PR` names that example's folder. **Camera frame bridge**: a child
  process of the agent program that hands it the receiver's decoded stereo pairs.
- **Experiment folder**: a folder outside this repository with the reBot rig's calibration, its
  depth code and its behaviour hooks for APAI. `AGENTIC_REMOTE_EXPERIMENT_DIR` names it. It is
  read, never written.
- **Skill stop**: APAI's own stop of the arm after a motion failed. It latches the workstation
  checker; only a restart of the agent program clears it.
- **Run folder**: the folder one run of the agent program writes its files into.

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
- The agent side (`agent/`): the agent program, its arm adapter, the check ahead, the camera
  input and the console command. See
  [The agent side on the workstation](#the-agent-side-on-the-workstation).

## What it does not contain

- No live view, and of the operator console only the console command, which sends one
  command and ends.
- APAI, the experiment folder and the camera streamer are not in this repository. The agent
  program imports the first two from where their environment variables point;
  `run_camera_receiver.sh` starts the third's receiver.
- No camera frame from a Thor has reached the camera input. It is tested on recorded stereo
  pairs and on stereo pairs the tests make up.
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
- **A command id belongs to its sender until its final RESULT.** A command from another client
  under the same id is `refused` with `command id in use` and not forwarded.
- **Every role may send STOP**, and it goes out at once. From then until a ROBOT_STATE shows
  a higher stop number the service drops PERMITs and refuses a MOVE or GRIPPER that quotes the
  old stop number. A PERMIT, MOVE or GRIPPER the service was already sending when the STOP
  came can reach the Thor after it; the Thor's own stop-number check covers that one.
- **A client over `local.max_clients` is closed** with a LOCAL_NOTICE. A STOP it sends first,
  alone or right after its LOCAL_HELLO and within 0.2 s, still goes out.
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

## The agent side on the workstation

The agent side is what runs beside the robot link service: APAI's agent, its camera input, its
obstacle check and the operator's console command. It reaches the arm only through the local
API. Nothing under `agent/` imports Isaac Capture or a motor driver, and APAI is not edited:
the agent program replaces names of APAI's modules while it runs and puts them back when it
ends, also when one of its start steps fails.

### What runs where

| Program | Machine | Started with |
|---|---|---|
| Robot app | Thor | see [The robot connection](#the-robot-connection) |
| Camera sender | Thor | the camera streamer's `argus_sender` with `configs/camera_sender_rebot.yaml` |
| Robot link service | Workstation | see [The robot connection](#the-robot-connection) |
| Camera receiver | Workstation | `./run_camera_receiver.sh --host THOR_ADDRESS` |
| Agent program | Workstation | `./run_agent.sh --config configs/agent.yaml --interactive` |
| Console command | Workstation | `python -m isaaccapture_examples.agentic_remote.agent.console_cmd COMMAND` |

With `--camera link` the agent program starts the camera frame bridge itself, under the
interpreter `camera.bridge_python` names. The camera receiver, the bridge and the agent
program use one GPU: index 0 in PCI bus order. Both start scripts and the agent program set
`CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=0` themselves.

### The agent program

```text
python -m isaaccapture_examples.agentic_remote.agent --config PATH
    [--task TEXT] [--interactive] [--max-steps N]
    [--camera link|mock|recorded:<dir>] [--check] [--run-dir PATH] [--log-level L]
```

- `--task TEXT` runs one task and ends; an empty text is refused. `--interactive` reads one
  task after another at the prompt `task>`; an empty line or the end of the input ends the
  program. With neither, APAI runs its own default task, which only looks: `look at the table
  and report what objects you see`.
- `--max-steps N` (30, at least 1) is the most steps APAI's planner may take for one task.
- `--camera link` (the default) takes stereo pairs from the camera receiver. `--camera mock`
  is APAI's mock camera, a made-up scene with a red cube. `--camera recorded:<dir>` reads
  `left_raw` and `right_raw` images from a folder and hands them out again and again.
- **The capture times of `--camera mock` and `--camera recorded:<dir>` are invented**: each
  camera frame is dated at the moment it is handed out, so a permit would rest on evidence
  that nobody captured. With `camera.allow_invented_times: false` (the shipped value) the
  program does not start on such a camera: exit code 2 and one line that names the setting.
  With `true` it prints a `WARNING` line at its start, and every such camera frame carries
  `invented_times: True` in its capture record. Set it to `true` only for a run against a
  simulated or mock arm. Nothing ties the setting to the robot app yet: a robot app that
  declares a simulated driver, and an agent program that asks for that declaration, is a P2
  item.
- `--check` loads and checks `agent.yaml` and APAI's configuration, attaches to nothing,
  prints the resolved values and the camera calibration id as JSON, and writes nothing into
  `state_dir`.
- `--run-dir PATH` names this run's folder; `--log-level` is `debug`, `info`, `warning` or
  `error`.

The program refuses to start (exit code 2) while any environment variable whose name begins
with `APAI_` is set and not empty, except the ones it takes: the seven store variables
(`APAI_BELIEFS_PATH`, `APAI_ARM_HOLD_DIR`, `APAI_REST_LOG`, `APAI_EPISODIC_PATH`,
`APAI_GRASP_MEMORY_PATH`, `APAI_ENVELOPE_PATH`, `APAI_PROGRAMS_PATH`), `APAI_REQUIRE_CUDA`,
`APAI_ASPIRE_LIBRARY` (where APAI reads its learned notes from) and `APAI_GRASPGENX_PORT`
(the port of a grasp server on this machine). Any other one could change what APAI builds;
the line names it.

| Exit code | Meaning |
|---|---|
| 0 | `--task`: the task succeeded and the arm was given back. `--interactive`: every task of the session succeeded and the arm was given back. `--check`: the files can be used. One Ctrl+C: the task was cancelled and a STOP sent |
| 1 | No robot link service, the robot app not ready in time, another camera calibration than the robot app's, a `rig` list that is not of the arm's joint count, APAI's runtime not built (for example no usable obstacle grid within `task.grid_wait_s`), a task that ended without success (`--interactive`: any task of the session), an owner request at a task's end that was not answered, the agent still control owner when the program ended, a skill stop (the program said `restart the agent program`), or SIGTERM: a STOP was sent first |
| 2 | The arguments, `agent.yaml` or the environment cannot be used, or APAI cannot be imported. Nothing was attached |
| 130 | A second Ctrl+C. A STOP was sent again first |

A task the operator cancelled with Ctrl+C does not count as a task without success. Two of
the causes of exit code 1 are read from the code and were not run: a robot app that is not
ready within `link.connect_timeout_s`, and an obstacle grid that is not usable in time.

### A task from its start to its end

1. **Thinking and looking take no arm.** APAI's read-only tools (observe, list objects, recall
   and the like) run without ownership and without a question to the operator.
2. **The first skill that moves asks the operator.** The agent program reads ROBOT_STATE and
   refuses at once when the control owner is not nobody, a fault-stop is latched or the motors
   are off. Otherwise it prints one question that names the task, the first skill and the
   joint speed, and waits for a typed `yes` at the terminal the task was typed at. Any other
   answer refuses the skill and sends nothing. The question is never answered from a pipe, and
   never inside the environment of an AI coding agent. What was typed at the terminal before
   the question was written is dropped, so a `yes` typed ahead answers nothing. One `yes`
   covers the later skills of the same task.
3. **Then it asks the robot app for the arm**: OWNER_REQUEST "agent", quoting the owner number
   and stop number of the ROBOT_STATE the operator saw. After the RESULT it waits for a
   ROBOT_STATE that shows the agent as control owner, and from then on every MOVE, PERMIT and
   GRIPPER of the task quotes that state's numbers. Without a RESULT within `task.end_wait_s`
   the skill is refused. The program keeps watching for an ownership that arrives later:
   before the next task takes the arm, at the task prompt and when the program ends. When
   ROBOT_STATE shows the agent as control owner and no task of the program owns the arm, it
   sends OWNER_REQUEST "nobody" with the reason `the ownership arrived late`. Before the next
   task takes the arm it waits up to `task.owner_wait_s` for that grant, and at its end up to
   3 s.
4. **A task that succeeded gives the arm back**: OWNER_REQUEST "nobody". The arm holds where
   it is. The return to rest is the operator's: `rest` at the Thor terminal or the console
   command `rest`. APAI's own `move_to_rest` skill is refused.
5. **A task that ended any other way while the agent owned the arm asks for help**:
   OWNER_REQUEST "help_needed" with the reason, which starts the assistance wait on the Thor.
   The reasons are `stuck: ...` (APAI gave up, or `task.failed_steps_limit` skills failed in a
   row), `blocked: ...`, `step budget ran out`, `agent error: ...` and `task ended without
   success: ...`. With `task.timed_return: false` the Thor makes no return to rest when the
   wait runs out.
6. **The local socket is closed after that request's RESULT**, once ROBOT_STATE no longer
   shows the agent as control owner. When the request gets no RESULT within
   `task.end_wait_s`, the program says `was not answered`, sends nothing else and ends with
   exit code 1. It then waits up to 3 s for a ROBOT_STATE without the agent as owner. If the
   agent is still the owner after that, the socket closes anyway and the program says so; the
   service then asks for help in the agent's name (`agent program gone`).
7. **Ctrl+C is a STOP.** The first one cancels the task: a running move is ended through its
   permit, then a STOP goes out with the sender `agent_terminal`. Nobody owns the arm, no help
   request is sent and no assistance wait starts. A second Ctrl+C sends STOP again and ends
   the program.
8. **A STOP from the console ends the task as well.** The running move ends `cancelled` and
   APAI's skill ends as halted. Nobody owns the arm then, so the task's help request, which
   quotes the stop number from before the STOP, is refused. The next task can take the arm;
   the program needs no restart.
9. **After a skill stop the program takes no new task.** It says `a skill stop latched the
   workstation checker: restart the agent program` and ends with exit code 1. The task itself
   ended as in step 5.

The agent program sends OWNER_REQUEST and STOP; its arm adapter sends MOVE, PERMIT, GRIPPER
and CHECKER_STATE. It sends no RELEASE, SHUTDOWN, RESET or OPEN and no request for a return
to rest, and it cuts no torque.

### A move, its permits and the camera evidence

- **One MOVE per move of APAI.** The arm adapter reads the start pose from joint readings taken
  after the call began, samples the move at the Thor's setpoint period, and has the check ahead
  vet every setpoint on an obstacle grid at most `move.start_grid_age_s` old. A refusal here
  sends nothing.
- **`ROBOT_STATE.move_stage`** tells how far the Thor has got with a move: `""` while none
  runs or its start checks have not passed, `"armed"` once the Thor has read the start pose and
  found it within the MOVE's tolerance (nothing was sent yet), and `"sending"` from the first
  permit that covers the first setpoint. The arm adapter waits for `"armed"` before the
  workstation checker sees the first setpoint, so a move the Thor refuses earlier counts as
  not started. The Thor waits `permits.first_wait_s` (0.5 s) from `"armed"` for the first
  permit.
- **Permits follow the arm.** Every `permits.renew_s` the check ahead clears the setpoints up
  to `permits.reach_setpoints` beyond the last one ROBOT_STATE shows as sent, and a PERMIT
  that reaches that far and lasts `permits.life_s` goes out (the first PERMIT of a move
  reaches setpoints 0 to 24). No PERMIT is sent while the newest ROBOT_STATE is older than
  `link.state_renew_age_s` (0.1 s); a newer ROBOT_STATE lets the permits go on. Each PERMIT quotes the evidence
  time of the obstacle grid the setpoints were checked on. When a new grid arrives, setpoints
  cleared before and not yet sent are checked again.
- **The permits end at once** when the check ahead refuses a setpoint, the workstation checker
  is latched or halted, the ownership no longer stands, the link is lost or the newest
  ROBOT_STATE is older than `link.state_max_age_s`. The arm adapter then sends one PERMIT that
  reaches setpoint -1 with the evidence time of the one before and the numbers the move
  began with. The Thor takes it also when its clock no longer accepts that evidence time,
  and ends the move
  `refused` with `permit reaches setpoint -1, not N` and the arm holds at setpoint N-1. If
  that PERMIT cannot leave, the arm goes at most as far as the last permit reached.
- **`camera_calibration_id` in WELCOME** is the name of the camera calibration the robot app's
  file gives (`camera_calibration_id`, empty by default). The agent program compares it with
  its own: `link.calibration_id`, or, when that is empty, `stereo-<12 hex>+mount-<12 hex>`
  made from the SHA-256 of `camera.calibration_dir/manifest.json` and of
  `camera.hand_eye_json`. With another name the program ends with exit code 1 before any
  task; a robot app that names none is taken only with `link.allow_unnamed_calibration: true`.
- **A camera frame is dated at its exposure.** The stereo camera refuses a stereo pair when
  the newest ROBOT_STATE is older than 1 s, its exposure lies more than 0.05 s in the future
  of the Thor's wall clock, more than 0.05 s plus the link's clock margin (0.02 s) after its
  arrival on the workstation, or more than 1 s before that arrival, or the Thor's wall clock
  stepped within the last 3 s. The evidence time a permit quotes is never later than the
  pair's arrival. The self mask uses the joint samples of ROBOT_STATE on both sides of the
  exposure; a camera frame without them puts nothing into the obstacle grid.

### `configs/agent.yaml`

Every number in the shipped file is a first value; none was measured on a robot. A relative
path is taken from the file's folder. The program refuses the file (exit code 2) when it
holds a key the program does not know, a key given twice, a block that is not a mapping, a
number that is not above 0 (`camera.max_skew_ms` may be 0) or above its upper bound,
`permits.renew_s` not below `permits.life_s`, `task.max_joint_speed` above
`thor.agent_max_joint_speed`, or two equal stream names. The shipped file names each upper
bound in a comment; for example a wait for a RESULT is at most 60 s and a joint speed at most
3 rad/s.

| Block | Keys |
|---|---|
| top | `state_dir` (see below), `log_level` |
| `link` | `socket_path` of the robot link service (empty: its default); `connect_timeout_s`; `state_max_age_s`, the oldest ROBOT_STATE a pose is read from; `state_renew_age_s`, the oldest ROBOT_STATE a PERMIT is sent on; `expected_profile_id`, which WELCOME must name (empty: any); `calibration_id` and `allow_unnamed_calibration` (above) |
| `thor` | Copies of three robot app settings WELCOME does not carry: `first_wait_s`, `settle_timeout_s`, `agent_max_joint_speed`. Keep them equal to the robot app's file |
| `move` | `start_tol_rad`: a MOVE quotes the smaller of this and half a period's travel at the Thor's agent speed limit; `fresh_wait_s`; `start_grid_age_s` and `start_grid_wait_s`; `expires_in_s`, how long a MOVE stays valid on the Thor; `cancel_wait_s` |
| `permits` | `reach_setpoints` (25), `renew_s` (0.05), `life_s` (1.0) |
| `jaws` | `fresh_s`: an older jaw reading counts as unknown; `result_timeout_s` |
| `task` | `timed_return` (false); `failed_steps_limit` (3); `max_joint_speed`, APAI's speed cap for the task, at most `thor.agent_max_joint_speed`; `end_wait_s`, `owner_wait_s` and `grid_wait_s`, the longest waits for an owner request's RESULT, for the ROBOT_STATE that shows the agent as owner, and for a usable obstacle grid at the start |
| `camera` | `bridge_python`, the camera consumer's interpreter; `camera_pr` (empty: `AGENTIC_REMOTE_CAMERA_PR`); `receiver_socket`; `streams`; `calibration_dir` and `hand_eye_json`, the workstation's copies of the stereo calibration folder and the camera mounting file in use on the Thor; `max_skew_ms` between the two eyes; `depth_backend` (`ffs` on the GPU, `opencv_cpu` without one); `fps`; `allow_invented_times` (false), see `--camera` above |
| `experiment` | `stack`: `full` uses the experiment folder's behaviour hooks, its obstacle grid bridge on `cuda:0` and its cloud model; `minimal` uses none of them, builds on APAI's own profiles `arm_profile` and `llm` (both `mock` in the shipped file) and pins APAI's mock detector for every camera, so no model server is asked |
| `rig` | Four values of the rig's arm profile on the Thor, copied: `home_q`, `park_q`, `body_mask_dilate_px`, `motion_bounds`. `home_q`, `park_q` and `motion_bounds.joint_center_rad` must have one value per joint of the arm; the program compares them with each other when it loads the file and with the joint count in WELCOME after it attached (exit code 1). `motion_bounds` takes `joint_center_rad`, `max_joint_deviation_rad`, `tcp_center_m` and `max_tcp_distance_m`, all four and no other key; the two `max_` values must be above 0 |

### The start scripts

`run_agent.sh` sets the environment and then runs
`python -m isaaccapture_examples.agentic_remote.agent` with its arguments.

| Variable | Needed | What it names |
|---|---|---|
| `AGENTIC_REMOTE_AGENT_PYTHON` | always | A Python with APAI's packages and PyYAML; for the full stack also torch with CUDA and Warp |
| `AGENTIC_REMOTE_APAI_SRC` | always | The `src` folder of the APAI checkout |
| `AGENTIC_REMOTE_EXPERIMENT_DIR` | `experiment.stack: full` | The experiment folder |
| `AGENTIC_REMOTE_CAMERA_PR` | `--camera link` | The camera streamer example folder |

Without one of the first two it prints one line that names the variable and exits with
code 2. It also sets `APAI_REQUIRE_CUDA=1` unless that is already set (a dry run on the mock
scene sets it to 0), `PYTHONDONTWRITEBYTECODE=1`, so that nothing is written into APAI's
checkout or the experiment folder, and unsets `LD_LIBRARY_PATH`.

`run_camera_receiver.sh --host THOR_ADDRESS` runs the camera streamer's
`camera_receiver/run.sh` with `configs/camera_receiver_rebot.yaml`. It needs
`AGENTIC_REMOTE_CAMERA_PR` and exits with code 2 and one line without it. A first argument
that does not begin with `-` is another configuration file; the other arguments go to the
receiver.

### The console command

```text
python -m isaaccapture_examples.agentic_remote.agent.console_cmd
    [--socket PATH] [--timeout S] state|stop|reset|rest|open|release|shutdown
```

It attaches to the robot link service in the role `console`, sends one command, prints the
answer and ends.

| Command | Typed `yes` first | What it does |
|---|---|---|
| `state` | no | Prints the service's status and the newest ROBOT_STATE; sends nothing |
| `stop` | no | STOP. It is answered by a ROBOT_STATE with a higher stop number |
| `reset` | no | RESET: clears a fault-stop |
| `rest` | yes: is the path clear | OWNER_REQUEST "rest": the robot app returns the arm to rest |
| `open` | yes | OPEN: the guarded jaw opening; what the jaws hold is let go |
| `release` | yes | RELEASE: motors off |
| `shutdown` | yes: is the path clear | SHUTDOWN: return to rest if the arm is away from it, motors off there, the robot app ends |

- **The `yes` is typed at a terminal**, before anything is attached. Without a terminal on
  standard input and output (a pipe, a script), and inside the environment of an AI coding
  agent, the command asks nothing, says `not confirmed` and sends nothing. What was typed at
  the terminal before the question was written is dropped.
- **Release and shutdown are checked on the Thor, not here.** The console command only sends
  the link message. The robot app releases torque only with the arm measured at rest and the
  jaws proven empty (see [the torque rule](#the-rule-torque-is-never-cut-away-from-rest)); a
  `release` with the arm away from rest is `refused` and the motors stay on.
- **Each command quotes the newest ROBOT_STATE's numbers**, so one built for another state of
  the arm is refused.

Exit code 0: the command's final RESULT was `completed` (`state`: a ROBOT_STATE was there;
`stop`: the higher stop number was seen). 1: refused, failed, not answered in time, not
confirmed, or no robot link service. 2: a command that does not exist.

### `state_dir` and the run folders

`state_dir` (shipped: `../.state`, which is `examples/agentic_remote/.state`) holds what the
agent program writes. The program puts a `.gitignore` with `*` into it.

- `state_dir/apai/` holds APAI's stores (beliefs, experience, grasp memory and the like). A
  store whose APAI environment variable names a place stays there. A variable that is set
  and empty counts as not set: the store goes into `state_dir/apai/`.
- `state_dir/runs/<UTC time>-<process id>/` is the run folder, unless `--run-dir` names
  another. It holds `agent-settings.json` (the resolved settings), `arm_profile.yaml` and, for
  a stereo camera, `camera_profile.yaml` (the profiles APAI was given), `task-reports.jsonl`
  (one line per task: APAI's task report with every skill's result), APAI's own trace files,
  and `experiment-files.json` (the SHA-256 of every file of the experiment folder the run
  imported) when `AGENTIC_REMOTE_EXPERIMENT_DIR` is set.

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
  configs/agent.yaml          settings of the agent program
  configs/camera_sender_rebot.yaml configs/camera_receiver_rebot.yaml
                              the camera streamer's two programs on the reBot rig
  run_agent.sh run_camera_receiver.sh    start scripts for the workstation
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
    agent/
      link_client.py                     the agent program's end of the local API
      setpoints.py arm_adapter.py        a move sampled into setpoints; APAI's arm on the link
      checkahead.py                      the check ahead and the evidence times
      self_mask.py                       the arm pose at a camera frame's exposure
      camera/frame_source.py camera/frame_bridge.py camera/bridge_wire.py
                                         stereo pairs: from the bridge, or recorded; the bridge
      camera/stereo_camera.py            APAI's camera on stereo pairs
      config.py task_gate.py             agent.yaml and APAI's configuration; a task's start and end
      launcher.py __main__.py            the agent program; its command line
      console_cmd.py                     the console command
    console_log.py            the console log level of the programs
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
`[connection] ...` at its Thor terminal. Type `help` there. Tasks come from the agent program
(see [The agent side on the workstation](#the-agent-side-on-the-workstation)). A local client
of your own is three lines of Python (`connect_local()` and `testing/client.py`), as in
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

The tests carry four markers. Each skips its tests, with the reason, when what it needs is
missing:

| Marker | Needs |
|---|---|
| `link` | Isaac Capture's remote link on the Python path |
| `apai` | `AGENTIC_REMOTE_APAI_SRC` and the `apai` extra |
| `experiment` | `AGENTIC_REMOTE_EXPERIMENT_DIR`; a test whose experiment module needs a package the environment lacks is skipped too |
| `gpu` | torch with CUDA. The test environment has no torch, so these are always skipped there |

The agent side's tests are `test_agent_link.py`, `test_setpoints.py`, `test_checkahead.py`,
`test_arm_adapter.py`, `test_frame_bridge.py`, `test_frame_source.py`, `test_stereo_camera.py`,
`test_self_mask.py`, `test_task_gate.py`, `test_launcher.py`, `test_console_cmd.py`,
`test_apai_contract.py` (the names of APAI the agent side relies on) and
`test_agent_end_to_end.py`. Where they drive the robot app with the real permit loop, the robot
app runs on the machine's clock and a move takes its real length.

`test_agent_end_to_end.py` runs the whole agent program in the test process, on APAI's mock
scene with `experiment.stack: minimal` and a stand-in for the obstacle grid bridge, against the
robot app on APAI's software arm: on a loopback carrier and, marked `link`, behind the robot
link service on a real link of 127.0.0.1. It types two tasks, `open the gripper` and
`pick up and place the red object`, and checks on the robot app that every joint target the
driver took was covered by a permit. The file takes about 70 s with everything present; the
pick and place alone takes about 22 s in each of the two runs. With everything present:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/path/to/build/python_package/Release \
AGENTIC_REMOTE_APAI_SRC=/path/to/apai/src \
AGENTIC_REMOTE_EXPERIMENT_DIR=/path/to/experiment \
    .venv/bin/python -m pytest -q test_agent_end_to_end.py
```

`PYTHONDONTWRITEBYTECODE=1` keeps Python from writing `.pyc` files into APAI's checkout and the
experiment folder. `-rP` prints the numbers the two long tests measured.

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

The agent side has no way to cut torque. The same scan reads every file under `agent/` and
finds no such call. APAI calls `disconnect()` and `stop()` on its arm when it shuts down and
after a failed motion; on the arm adapter the first sends nothing and the second at most a
STOP, which is a hold. RELEASE and SHUTDOWN can be sent only by the console command, after a
typed `yes`, and the robot app still applies its own checks to them. `test_no_limp.py` ends
an agent move in each way the agent side knows (a setpoint refused by the check ahead, a
latched workstation checker, Ctrl+C, a STOP at the Thor terminal, a lost link, the agent
program gone) and each time finds the arm holding away from rest with its motors on.

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

The agent side, on the workstation, 2026-11-15, in the test folder's environment (Python
3.12, no torch, no GPU use), with APAI's checkout and the experiment folder named:

| What | Run as | Result |
|---|---|---|
| `agent --help`, `console_cmd --help` | As processes | Exit code 0 |
| `agent` without `--config`, with a file that does not exist, with `APAI_OCCUPANCY=1`, without APAI | As processes | Exit code 2 and one line each |
| `run_agent.sh` without `AGENTIC_REMOTE_AGENT_PYTHON`, without `AGENTIC_REMOTE_APAI_SRC`; `run_camera_receiver.sh` without `AGENTIC_REMOTE_CAMERA_PR` | As processes | Exit code 2 and one line each |
| `run_agent.sh --config configs/agent.yaml --check --camera mock` | As written, with the test folder's Python as `AGENTIC_REMOTE_AGENT_PYTHON` and `APAI_REQUIRE_CUDA=0` | Exit code 2 and one line: the shipped file has `camera.allow_invented_times: false`; no `.state` folder was made |
| `--check --camera mock` with a file that sets `camera.allow_invented_times: true` and `experiment.stack: minimal` | By a test, in the test process | Exit code 0; the JSON was printed; no state folder was made |
| `run_agent.sh --config configs/agent.yaml --check` | The same way | Exit code 2: the shipped file names no `camera.calibration_dir` and `camera.hand_eye_json`. Fill them in, and `camera.bridge_python`, before `--camera link` |
| `console_cmd move`; `console_cmd release` with standard input from `/dev/null` | As processes | Exit code 2; exit code 1 with `release not confirmed (...)`, nothing sent |
| The agent program with `--interactive` and two tasks, `--camera mock`, `experiment.stack: minimal` | By a test, in the test process, with the typed lines and the `yes` scripted: on a loopback carrier and behind the robot link service on 127.0.0.1 | Exit code 1: the first task succeeded, the second ended with a help request. 905 joint targets in 8 moves, each under a permit that covered it; 470 to 591 permits per run; the camera frame behind each permit 0.03 to 0.44 s old when its setpoint went out; 22.5 s per run |
| The same with `--task` and an obstacle put into the obstacle grid during the first move | By a test, on a loopback carrier | Exit code 1. The workstation checker refused setpoint 57 of 95; the arm held after 32 to 36 setpoints, at least 0.4 s of motion before the refused one; the Thor answered the cancelling permit within 1 to 21 ms; help request, no STOP, motors on |
| Two Ctrl+C during a move | By a test: SIGINT raised in the test process | Exit code 130; two STOPs from `agent_terminal`; the arm held |
| A STOP from a console client during a move, then a second task | By a test, behind the service, with `--interactive` | The move ended `cancelled` after 40 setpoints; the task's help request was refused for its old stop number; the second task ran without a restart; exit code 1 |
| SIGTERM during a move | By a test: SIGTERM raised in the test process | Exit code 1; a STOP from `agent_terminal`; nobody owned the arm; motors on |
| An OWNER_REQUEST "agent" the robot app answers 0.6 s late, with `task.end_wait_s: 0.3` | By tests, with `--task` on a loopback carrier and behind the service, and with `--interactive` and a second task | The task was refused; the program gave the late ownership back itself (`the ownership arrived late`) before it closed the socket; with `--interactive` the second task then took the arm and succeeded; exit code 1 |
| An OWNER_REQUEST "nobody" at a task's end that gets no RESULT | By a test, on a loopback carrier | Exit code 1 after the 3 s wait; the socket closed with the agent as control owner |
| A `yes` typed before the question, for the agent program's question and for `console_cmd release` | By tests, each in a child process on a pseudo-terminal | The `yes` typed ahead was dropped; what was typed after the question counted. With `CLAUDECODE` set, `console_cmd release` asked nothing and said `not confirmed` |
| `--camera recorded:<dir>` with a recorded stereo pair of the experiment folder, its calibration and `depth_backend: opencv_cpu` | By a test, with `experiment.stack: minimal` and so with APAI's mock detector | Exit code 0; the obstacle grid went live on the camera frames made from that pair |
| The console command's `stop`, `reset` and `release` against the robot app | By tests, in the test process, behind the service | `release` at rest completed; away from rest it was refused |

Never run:

- The agent program as a process with a task: the typed `yes` needs a terminal, and the tests
  script it.
- `experiment.stack: full`: the experiment's hooks, its obstacle grid bridge on the GPU and its
  cloud model. No Python environment on the workstation has torch with CUDA and Warp yet. Only
  the merge of its configuration files is tested.
- `--camera link` through the agent program, and the camera receiver's start script with a
  receiver. The camera frame bridge is tested against a stand-in for the receiver.
- `depth_backend: ffs`, and anything else on a GPU from these tests.
- Anything between two machines, with a camera on a Thor, or on a real arm.

## Next phases

| Phase | What is built |
|---|---|
| P1 | What is left of it: the full stack on the GPU, the camera connection from the Thor, the first run on the arm |
| P2 | Demo 1: the agent completes a task; live view and operator console |
| P3 | Demo 2: a person takes over with the headset when the agent asks for help |
| P4 | Every fault under each control owner, measured delays, launch and recovery instructions |

[design]: ../../docs/design/claude_agentic_remote_execution.md
