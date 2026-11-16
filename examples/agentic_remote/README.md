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
P4. This folder holds four of them. The fourth builds the **human path**, the way from a
person's hand to the arm's joints. It turns the hand's motion into joint targets on the
Thor, as the design places it, in a program of its own beside the robot app (see the
paragraph after the list).

- P0: the robot app and the link messages on a simulated arm.
- P1: the connection between the two machines, the workstation program that holds its end, and
  the agent side on the workstation, which runs the agent with its obstacle check and drives
  the arm through that connection.
- P2, without its hardware run: the programs of **demo 1**, in which the agent completes a
  task while an operator watches a live camera view and can stop the arm from a console. These
  are the operator console, the live view and the start of the robot app on the real arm,
  with the files and the checklist for the first run on the **rig** (the real arm, its camera
  and the Thor).
- P3, without its hardware run: the programs of **demo 2**, in which a person takes the arm
  over with the right controller of a headset when the agent asks for help, moves it, brings
  it to rest and gives it back. These are the **human program**, which turns the hand's
  motion into joint targets, and the robot app's side of a person driving.

**Where the human path runs.** On the rig the human program runs on the Thor, attached at
the robot app's own socket (`--human-socket`), so that the person's stop of the arm
reaches the robot app without the connection between the two machines that the agent
uses. The headset is connected to the workstation, where the
**Teleop Session Server**, a program of Isaac Capture, reads its controllers. The human
program is that server's client: its **teleop connection** to the server is a
**remote session** of Isaac Capture, which crosses between the two machines. So the
button behind that stop still arrives from the workstation, over the teleop connection:
while it is down the human program reads no button. On one
machine and in the tests the human program attaches to the robot link service, the
workstation program that holds the workstation's end of the connection between the two
machines. The design (its sections 4, 5, 12 and 14) places this work on the Thor too, on
a thread inside the robot app; here it is a process of its own beside the robot app.
[Where the human path departs from the design](#where-the-human-path-departs-from-the-design)
says what that difference costs and what stands as the design has it.

Nothing here has moved a real arm, and no headset was connected to any of it. To try it on
one machine, read [Run demo 1 on one machine](#run-demo-1-on-one-machine) and the two
recipes after it. The operator's steps for the real arm are in
[First run on the rig](#first-run-on-the-rig).

## Terms

- **Setpoint**: one target value per joint. **Move**: a list of timed setpoints. **Hold**: the
  arm stays where it is, motors on. **Rest**: the folded pose in which it lies on its own stops.
- **Control owner**: who may move the arm right now: the agent, a human, recovery (the robot app
  itself, for a return to rest) or nobody.
- **Session id**, **owner number**, **stop number**: a random value picked at each start, and two
  counters that go up at every change of control owner and at every STOP. Every command quotes
  them, and a command that quotes an old value is refused.
- **Checker**: the code that accepts or refuses each setpoint just before it is sent.
  **Limit set**: the limits one control owner's checker applies, a joint box and a joint
  speed among them. The robot app's file holds three: `limits.agent`, `limits.human` and
  `limits.recovery`.
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
- **Operator console**, **live view**: two workstation programs. At the console the operator
  reads the robot's state and types `stop`, `rest`, `release` and the other commands; the live
  view shows the camera picture and the robot's status in a browser. **Console command**: a
  third program that sends one of the console's commands and ends.
- **Thor terminal**: the terminal on the Thor in which the robot app runs. The operator at the
  robot types there.
- **Demo 1**: the agent completes a task on the arm while an operator watches the live view and
  can stop the arm or send it to rest from the console.
- **Demo 2**: when the agent fails, a person is prompted, takes the arm over with the right
  controller of a headset, moves it, ends at rest and gives the arm back, and the agent's
  next task runs. The terms of the human path (takeover, hand-back, clutch, sample age and
  the others) are defined in [Demo 2: a person takes over](#demo-2-a-person-takes-over).
- **Rig**: the real arm (a reBot B601 with RobStride motors), its camera and the Thor.
  **CAN**: the arm's motor bus; on the rig it is the interface `can4`. **Rig owner**: the
  person who is responsible for the rig and decides what is switched on for it and what
  runs on the arm. "The owner" with no "control" before it means the rig owner.
- **Driver**: the code that talks to the motors. The robot app declares in WELCOME which one
  it runs: `sim` (the simulated arm), `apai-mock` (APAI's software arm) or `real`.
- **Typed confirmation**: a phrase the operator types before a command is sent. There are
  three: `yes`, `path is clear` and `jaws are empty`; see
  [the wording table](#what-the-operator-types-to-confirm).
- **Hold marker**: a file APAI writes when one of its runs leaves the arm powered and holding
  away from rest. APAI keeps them in `~/.apai/arm_holds`.
- **View age**: how old the picture is that the operator looks at in the live view, in
  seconds, measured on the Thor's clock.
- **Robot connection**: the one connection between the robot app and the workstation that
  carries the link messages above. The robot app dials it; a **dial** is one attempt to open it.
- **Robot link service**: the workstation program that holds the workstation end of the robot
  connection. **Local API**: what it offers other programs on the workstation, over a Unix
  socket. Such a program is a **local client**, and it says in which **role** it attaches:
  agent, console, viewer, tool or human. The robot app serves the same local API on a
  socket of its own for one client, the human program on the Thor.
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
  checker; the next task clears that latch after its typed `yes` and the Thor's grant.
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
- The Thor terminal (`terminal.py`): `stop`, `rest` (asks for a typed `path is clear`),
  `release` (motors off, only at rest), `quit` (return to rest, motors off there, exit; it
  asks for `path is clear` when the arm has to move), `reference` (see
  [What the jaws do](#what-the-jaws-do)) and `help`. Its `install_signal_handlers()` makes
  Ctrl+C a STOP; the self-tests do not call it, the program started with `--connect` does.
- The agent side (`agent/`): the agent program, its arm adapter, the check ahead, the camera
  input and the console command. See
  [The agent side on the workstation](#the-agent-side-on-the-workstation).
- The operator console (`console/`) and the live view (`view/`). See
  [The operator console](#the-operator-console) and [The live view](#the-live-view).
- The start of the robot app on the real arm (`--arm apai --driver real`) and the files for
  the rig. See [The real driver](#the-real-driver) and [The rig files](#the-rig-files).
- A person driving the arm, on the Thor (`robot_app/human.py` and `robot_app/app.py`): the
  takeover and the hand-back, joint targets from a person's hand checked for their age, the
  live view's age, their step and their speed, and a hold at every end; the jaws under the
  controller's trigger (`robot_app/gripper.py`); and the robot app's own socket for the
  human program (`robot_app/human_socket.py`). See
  [Demo 2: a person takes over](#demo-2-a-person-takes-over).
- The human program (`human/`, started with `run_human.sh`): the headset's right controller
  through Isaac Capture's retargeters and APAI's inverse kinematics to joint targets, and a
  synthetic controller that stands in for the headset in the tests and on one machine. It
  runs on the Thor on the rig and beside the robot link service on one machine. See
  [The human program](#the-human-program).
- Files for one machine: `configs/robot_app_mock.yaml` and `configs/agent_mock.yaml` for a
  task of the agent on APAI's software arm, `configs/robot_app_demo2.yaml` for demo 2 on
  the simulated arm. See
  [Run demo 1 by hand with the mock agent](#run-demo-1-by-hand-with-the-mock-agent) and
  [Run demo 2 on one machine](#run-demo-2-on-one-machine).

## What it does not contain

- APAI, the experiment folder and the camera streamer are not in this repository. The agent
  program imports the first two from where their environment variables point;
  `run_camera_receiver.sh` starts the third's receiver.
- No camera frame from a Thor has reached the camera input or the live view. Both are tested
  on recorded stereo pairs and on stereo pairs the tests make up.
- No headset was connected. Every takeover and every motion of a person in the tests comes
  from a synthetic controller or from link messages a test makes; see
  [What is not tested without a headset](#what-is-not-tested-without-a-headset). Human
  driving is switched off in `configs/robot_app.yaml` and in the rig's
  `configs/robot_app_rebot.yaml` (see
  [The settings that ship off](#the-settings-that-ship-off)).
- No arm motion on hardware. Nothing in this folder has run on a robot. `--driver real` was
  run only by tests, on APAI's driver class over a stand-in for the motor library. Every value
  in `configs/robot_app.yaml` is a demo value for the simulated arm, and every value in the
  rig files is a first value. With the shipped `configs/robot_app.yaml`, `--arm apai` stops
  at the human limit set: its 1.5 rad/s is above the speed cap of APAI's profiles (see
  [What a move does](#what-a-move-does)).
- No model client: the program the full stack asks its cloud model through is not installed
  (see [The workstation environment](#the-workstation-environment)), so no task has run on
  the full stack.
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
connection only from an address in `allowed_peers`. Each entry of that list is an IP address;
a host name is refused when the file is loaded. A link-local IPv6 address is written with the
name of the workstation's interface, as in `fe80::1%eth0`: that is how such a peer is reported,
and an entry without the name, or with the interface's number, is refused when the file is
loaded. One robot connection at a time: while one
is alive a second is closed at once. It prints where it listens and where its local API is,
and runs until Ctrl+C or SIGTERM. Closing it does not stop the robot; the Thor holds on its
own. A second service on the same socket prints `a robot link service already runs on ...`
and exits with code 1; the socket file of a service that died is replaced.

**The robot app** with `--connect` starts on its arm, serves the Thor terminal on its standard
input and dials the service. While nothing answers it dials again every `connection.retry_s`
(1 s, never below 0.2 s) and says once why: `robot app: no robot connection: ...`. The arm
port is connected once, when the app starts, whether or not the workstation answers. Ctrl+C is
a STOP and never ends the program; `quit` ends it (return to rest, motors off there); SIGTERM
closes it with the motors left on. A terminal that closes ends neither the start nor the
program: the robot app runs on without its lines and ends as before. Its exit code is 0 after
a `quit`, a SHUTDOWN or SIGTERM, 1 when the app could not start, and 2 when nothing ran.

**With `--human-socket PATH`** the robot app also serves a Unix socket of its own, for a
human program that runs on the robot app's machine (see
[The human program on the Thor](#the-human-program-on-the-thor)). The argument is taken
with `--connect` only and needs a path: otherwise the program ends with exit code 2 and
`--human-socket can be used with --connect only` or `--human-socket needs the path of the
socket`. A socket that cannot be made ends the program with exit code 2 before the arm is
connected: `robot app: --human-socket PATH cannot be served: ...`. Once it serves the
socket, the program prints `robot app: the human program's socket is PATH` after its
session line. Without the argument there is no such socket.

### The local API and its roles

A local client opens the Unix socket (`socket_path` of the service's file; by default
`$XDG_RUNTIME_DIR/agentic_remote/robot_link.sock`, else `/tmp/agentic_remote-<uid>/robot_link.sock`,
in a folder of mode 0700), says LOCAL_HELLO with its role and from then on sends and receives
the same link messages as on the robot connection. In Python `connect_local(path, role, name)`
(`link_service/local.py`) gives a carrier, so the code above it is the same on both sides.

| Role | Who | May send | At a time |
|---|---|---|---|
| `agent` | The agent program's arm adapter | MOVE, GRIPPER, PERMIT, CHECKER_STATE, STOP, and OWNER_REQUEST "agent", "help_needed", "nobody" | One |
| `console` | The operator console | STOP, RESET, RELEASE, OPEN, SHUTDOWN, and OWNER_REQUEST "rest", "nobody"; with `operator.typed_takeover: true` in the service's file also OWNER_REQUEST "human", "handback" | Several |
| `viewer` | The live view program | VIEW_STATUS, STOP | Several |
| `tool` | Anything that only reads | STOP | Several |
| `human` | The human program | HUMAN_SETPOINT, STOP, and OWNER_REQUEST "human", "handback" | One |

- **Anything else is refused by the service** and never reaches the Thor: with a RESULT
  `refused` under the command's id, or with a LOCAL_NOTICE when there is no id. Both name the
  role and the link message, as in `the console role may not send OWNER_REQUEST "human"`.
- **One agent and one human program at a time.** A second one is refused at its LOCAL_HELLO:
  `an agent client is attached already; one at a time`, `a human program is attached
  already; one at a time`. A STOP that is the record right behind a refused LOCAL_HELLO,
  within 0.2 s, still goes out.
- **`operator.typed_takeover`** (false in both shipped service files) lets the console role
  ask for a takeover and for the hand-back, which are otherwise asked with the controller's
  buttons (see [Demo 2: a person takes over](#demo-2-a-person-takes-over)). A value that is
  neither `true` nor `false` is refused when the file is loaded. The service reads its file
  when it starts, and with the setting on it says so after its first line: `robot link
  service: operator.typed_takeover is on: the operator console may ask for a takeover and
  for the hand-back`.
- **A command id belongs to its sender until its final RESULT.** A command from another client
  under the same id is `refused` with `command id in use` and not forwarded.
- **Every role may send STOP**, and it goes out at once. From then until a ROBOT_STATE shows
  a higher stop number the service drops PERMITs and HUMAN_SETPOINTs and refuses a MOVE or
  GRIPPER that quotes the old stop number. A PERMIT, MOVE or GRIPPER the service was already
  sending when the STOP came can reach the Thor after it; the Thor's own stop-number check
  covers that one. LOCAL_STATUS says while this lasts, and the console's status line shows
  it as `a STOP is not confirmed yet`.
- **A client over `local.max_clients` is closed** with a LOCAL_NOTICE. A STOP it sends first,
  alone or right after its LOCAL_HELLO and within 0.2 s, still goes out. Such a STOP is a
  **raw STOP**: written to the socket without an attachment, so nothing confirms it. When
  there is no robot connection to carry it, a second LOCAL_NOTICE says that it was not
  delivered.
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
- **The robot app's own socket speaks the same local API**, for the role `human` alone. A
  robot app started with `--human-socket PATH` serves it, and the human program on the
  Thor attaches there (see [The human program on the Thor](#the-human-program-on-the-thor)).
  A client of another role is refused there with `this socket takes the human program
  only: the role human`, and a second human program as above.

### What WELCOME, ROBOT_STATE and VIEW_STATUS carry for the operator

- **`driver` in WELCOME** is the driver the robot app runs: `sim`, `apai-mock` or `real`. An
  empty value means that the robot app declared none; the agent program and the console
  treat that as `real`.
- **`motion` in ROBOT_STATE** is the motion state, one of `idle`, `moving`, `hold`,
  `rest_hold` (at rest, motors on), `rest_unpowered` (at rest, motors off) and
  `release_failed`. The last one follows a torque release whose driver call failed: the
  motors may be on or off, `torque_on` stays true, and nothing moves the arm until a
  release has worked (see
  [the torque rule](#the-rule-torque-is-never-cut-away-from-rest)).
- **`expires_mono_s` in MOVE, PERMIT and GRIPPER** is a time of the Thor's monotonic
  clock, which the sender reads off its newest ROBOT_STATE. A MOVE is not begun at or
  after it, a PERMIT covers no setpoint from it on, and nothing of a GRIPPER starts at or
  after it. A GRIPPER without the field never expires. The agent program sets all three:
  `move.expires_in_s`, `permits.life_s` and `jaws.result_timeout_s` of its file after
  the moment it sends.
- **`view_age_s` in ROBOT_STATE** is the view age: the age the newest VIEW_STATUS reported
  plus the time since it was sent. It is empty when no VIEW_STATUS has arrived on this
  robot connection, when the newest one said that the view is not live, and when it does
  not say when it was sent.
- **When a link message was sent** is `sent_mono_s` in VIEW_STATUS and in HUMAN_SETPOINT:
  the robot app's monotonic clock at that moment, as the sender reads it off its newest
  ROBOT_STATE (that state's `clock_mono_s` plus the time since it arrived at the sender).
  That reading is never later than the true moment. The robot app adds the time from
  then until it acts to the age the sender reported, so the link message's way to the
  robot app and its wait there are part of the age, and an age is never counted as less
  than it is.
- **`health.permit` in ROBOT_STATE** names the permit the robot app holds: `move_id`,
  `through_index` (the last setpoint it covers), `evidence_age_s` (the age of the camera
  frame behind it) and `expires_in_s`. It is empty without a permit. The permit of a move that
  has ended stays there until the next one; `expires_in_s` is negative once it has expired,
  and the console then adds `expired`. **`health.permit_refusal`** says why the running
  move's newest permit was refused, or is empty.
- **VIEW_STATUS** is the one link message the live view sends: `live`, `age_s` and, while
  the view is live, `sent_mono_s`. The robot app stores it and reports it back as
  `view_age_s`. With `human.require_live_view` it also gates a person's motion: without
  a view age, or with one above `human.view_age_limit_s`, a takeover is refused and a
  human drive ends in a hold. The robot link service takes it only from the role
  `viewer`.
- **`teleop_stage` and `controller_age_s` in ROBOT_STATE** come from the newest
  HUMAN_SETPOINT, the link message of the human program: its stage (how far it is with the
  headset's controller data; `ready` is the one at which they count) and the age of those
  data, which is the age the human program reported plus the time since the link message
  was sent. The stage is empty once no HUMAN_SETPOINT has arrived for
  `human.stage_keep_s` (4 s).
- **`health.human` in ROBOT_STATE** holds the robot app's facts about a person driving:
  `enabled` (the setting `human.enabled`), `takeover_open`, `driving` (a human drive runs),
  `needs_release` (the grip has to be released before the next squeeze moves the arm),
  `takeover` (empty, or why a takeover would be refused now), `refusal` (why the newest
  engaged HUMAN_SETPOINT moves nothing), `ended` (why the last human drive ended), `note`
  (the human program's note for the operator), `source` (`teleop` or `synthetic`), `seq`
  and `round_trip_s`: the Thor's clock when the newest HUMAN_SETPOINT arrived minus the
  clock of the ROBOT_STATE it was computed on. Nothing is enforced on the round trip.
  Two more are there only at times. `local_program`, from a robot app started with
  `--human-socket`: whether a human program is attached at that socket. `jaws`, while a
  takeover is open and `human.trigger_jaws` is true: `grip_level` (the jaw opening in
  force at the takeover, as a share of the jaw travel), `matched` (the controller's
  trigger has reached the grip level, so the jaws follow it), `opened` (the jaws were
  measured fully open since the takeover), `asked` (the jaw opening the newest
  HUMAN_SETPOINT asked for) and `refusal` (why the trigger moves nothing, or why the jaws
  are held wider than it asks).
  **`health.contact_hold`** is true while the arm holds at a contact, and
  **`health.clamped`** counts the steps of a human drive that the speed clamp cut.

### When the robot connection is lost

The robot app notices at once when the workstation closed the connection or its process ended,
and after the peer timeout (3 s) when the workstation or the network went silent. The service
notices the same way.

- **The Thor holds, by itself.** A running agent move, a timed return to rest and a return to
  rest or a release that was asked for over the connection end in a hold with the motors on.
  If the agent owned the arm, its ownership ends: control owner nobody, owner number up by one.
  A human drive ends in a hold as well, and a person who owned the arm keeps it; what the
  Thor had stored of the human program and of the live view is forgotten.
  A pending assistance wait is cancelled. `rest`, `release` and `quit` still work at the Thor
  terminal.
- **A human program at the robot app's own socket stays attached.** It still gets
  ROBOT_STATE, and its STOP is still taken: the person's stop button, right B of the
  headset's controller, needs no robot connection. A squeeze of the controller's grip
  moves nothing then where the robot app's file requires the live view, because
  VIEW_STATUS comes over the robot connection (`human drive ended: the live view is not
  fresh (...)`); a takeover asked for meanwhile is refused for the same reason. The
  buttons themselves reach the Thor over the human program's own connection to the
  workstation: where one cable carries both connections and is pulled, the human
  program reads no button, and the arm holds on the missing controller data.
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
[the torque rule](#the-rule-torque-is-never-cut-away-from-rest)). So the robot app connects
it once, when it starts, and only after the checks below. It is started on the Thor with

```bash
./run_robot_app.sh --config configs/robot_app_rebot.yaml --arm apai --driver real \
    --apai-profile PROFILE --calibration NAME --connect WORKSTATION:47610
```

For demo 2 the command also carries `--human-socket ~/.agentic_remote/human.sock` (see
[The human program on the Thor](#the-human-program-on-the-thor)).

None of this was run on the arm. The tests run it with APAI's driver class over a stand-in
for the motor library (`test_real_driver.py`).

The checks, in order. Until the connect of step 9 nothing is sent to the motors and the
motor bus is not opened; check 7 builds the driver object and does not connect it. A failed
check ends the program with exit code 2 and one line.

1. `--apai-profile`, `--calibration` and `--connect` are all given, and the profile is not
   APAI's software arm.
2. The standard input and the standard output are terminals: the questions are shown to a
   person and the confirmations are typed by one.
3. The environment is not that of an AI agent (the variables APAI's own confirmation looks
   at, such as `CLAUDECODE`).
4. `--calibration` repeats `camera_calibration_id` of the robot app's file, and that name is
   not empty.
5. APAI has no hold marker for the arm's motor bus. A typed `yes` never releases a hold: do
   APAI's operator recovery first.
6. No operator confirmation for APAI's connect is on file: APAI's hold folder
   (`~/.apai/arm_holds`) holds no `*.confirm.json` file, whatever arm it names, however old
   it is and whether or not it can be read. With one, APAI's driver would connect an arm it
   does not measure at rest. The refusal names the file. A hold folder that cannot be
   listed is refused too, with a line of its own (`APAI's hold folder cannot be listed
   (...)`): a confirmation may lie in it, and APAI's revoke cannot list it either, so the
   folder's permissions have to be put right. The robot app never writes or uses such a
   confirmation, and during its connect of step 9 its process cannot claim one.
7. The file fits the profile: no limit set is faster than the profile's speed cap, and the
   file's rest pose is the profile's park pose. A profile that drives hardware must name
   its park pose (`park_q`): APAI's driver would measure rest against all zeros without
   one. Isaac Capture's remote link can be imported.
   The driver object is built here and not connected. The program then prints the motor bus,
   the calibration, the speed cap and the tolerance for "at rest at start".
8. Three questions, each answered with a typed `yes`: is the arm powered; is it at rest and
   supported; is the calibration in use the one named. What was typed before a question was
   written is dropped. Any other answer, Ctrl+C, the end of the input or a last line without
   a line end ends the program with `not confirmed: nothing was connected`. Checks 3 to 6
   are made once more after the third answer; a Ctrl+C or a SIGTERM that comes while they
   are made also ends the program with that line. That
   is a moment no person can aim at: to abort without connecting, press Ctrl+C at a
   question. A terminal that is gone when the line `robot app: confirmed three times;
   connecting the arm` is to be written ends the program the same way. From that line on, a
   Ctrl+C or a SIGTERM is one that comes while the robot app starts (below): the arm is
   connected first.
9. The connect. APAI's driver reads every joint twice before anything that cuts torque and
   refuses an arm it does not measure at rest. Then the program prints `the robot app did not
   start: ...`, says that nothing was cut, and ends with exit code 1. An operator
   confirmation that appears after check 6 was made once more is not used: APAI's claim of
   a confirmation is switched off in this process for the connect, so a raised arm, or one
   with a hold marker written meanwhile, is refused all the same and the file stays where
   it is. A connect that fails
   later, after APAI's driver enabled the motors, also ends with exit code 1; the program
   then prints that the motors may be enabled and holding and that it ends without releasing
   them. The arm lies at rest then: switch the 48 V off there.

Ctrl+C and SIGTERM do not end the connect half-way, and a terminal that closes does not
either. A Ctrl+C pressed while the robot app
starts is a STOP once it has started. A SIGTERM that comes then closes the robot app after
the start, with the motors left on. After a terminal that closed the robot app runs on
without its lines.

After the connect the program prints `robot app: the real driver is connected at measured
rest: motors on, motion hold, jaws referenced` (or `motion rest_hold`) and dials the service.
It prints the motion state, the motors and the jaws, not the joint values; APAI writes the
joint reads of its connect to `~/.apai/logs/rest_readings.jsonl`. An arm without power can
lie farther from the rest pose than a release allows (0.05 rad); the rest readings recorded
on this arm lie up to 0.08 rad away. The motion state after the start is then `hold`: type
`rest`, then `path is clear`, before the first `release`.

`--driver mock` (the default) is APAI's software arm and asks nothing; a profile that drives
hardware is refused without `--driver real`. `--arm apai` without `--connect` connects
nothing: it builds the checkers and the driver object, prints what would run, among it the
profile's speed cap, and ends with exit code 2.

`run_robot_app.sh` sets the environment on the Thor and then runs
`python -m isaaccapture_examples.agentic_remote.robot_app` with its arguments. It needs
`AGENTIC_REMOTE_THOR_PYTHON` (the Python of APAI's environment on the Thor),
`AGENTIC_REMOTE_APAI_SRC` and `AGENTIC_REMOTE_ISAAC_CAPTURE_PY` (the folder of the
Isaac Capture Python package built on the Thor for that Python); without one it prints one
line and exits with code 2. It unsets `APAI_ROBOT`, `APAI_BASE` and `APAI_ARM_HOLD_DIR`, so
the hold markers are read from APAI's default folder.

## The agent side on the workstation

The agent side is what runs beside the robot link service: APAI's agent, its camera input and
its obstacle check. The operator's programs run beside it: the console, the console command
and the live view. All of them reach the arm only through the local API. The human
program, through which a person drives the arm in demo 2, runs on the Thor on the rig; in
the one-machine recipes it runs beside them and reaches the arm the same way. Nothing under
`agent/` imports Isaac Capture or a motor driver, and APAI is not edited: the agent program replaces names of APAI's modules while it runs and puts them back when it
ends, also when one of its start steps fails.

### What runs where

| Program | Machine | Started with |
|---|---|---|
| Robot app | Thor | see [The robot connection](#the-robot-connection) |
| Camera sender | Thor | the camera streamer's `argus_sender` with `configs/camera_sender_rebot.yaml` |
| Robot link service | Workstation | see [The robot connection](#the-robot-connection) |
| Camera receiver | Workstation | `./run_camera_receiver.sh --host THOR_ADDRESS` |
| Agent program | Workstation | `./run_agent.sh --config configs/agent.yaml --interactive` |
| Operator console | Workstation | `./run_console.sh` |
| Live view | Workstation | `./run_view.sh --config configs/view.yaml` |
| Human program | Thor on the rig; the one machine of the one-machine recipes | On the Thor `bash run_human.sh --config configs/human_rebot.yaml`, beside a robot app started with `--human-socket`; on one machine `bash run_human.sh --config configs/human.yaml` (see [The human program](#the-human-program)) |
| Teleop Session Server, with the headset's streaming service | Workstation, on the rig only | Isaac Capture's own programs, for demo 2 (step F1 of [the checklist](#first-run-on-the-rig)) |
| Console command | Workstation | `python -m isaaccapture_examples.agentic_remote.agent.console_cmd COMMAND` |
| The console, the agent program, the robot link service, the live view and the camera receiver of demo 1 on the rig, together | Workstation | `bash run_workstation.sh --env FILE --thor THOR_ADDRESS`: one window of tmux, a program that splits a terminal into panes, with a pane for each of the five (see [The start scripts](#the-start-scripts)) |

With `--camera link` the agent program starts the camera frame bridge itself, under the
interpreter `camera.bridge_python` names. The live view starts a second one for itself. The
camera receiver, the bridges and the agent program use one GPU: index 0 in PCI bus order.
`run_agent.sh`, `run_view.sh`, `run_human.sh`, `run_camera_receiver.sh` and the agent
program set `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=0` themselves; the
human program needs no GPU.

### The agent program

```text
python -m isaaccapture_examples.agentic_remote.agent --config PATH
    [--task TEXT] [--interactive] [--max-steps N]
    [--camera link|mock|recorded:<dir>] [--check] [--run-dir PATH] [--log-level L]
    [--batch-milestones] [--record-debug] [--debug-max-gib GIB] [--no-timing]
    [--motion-approval task] [--operate] [--depth-backend ffs]
```

- `--task TEXT` runs one task and ends; an empty text is refused. `--interactive` reads one
  task after another at the prompt `task>`; an empty line or the end of the input ends the
  program. With neither, APAI runs its own default task, which only looks: `look at the table
  and report what objects you see`.
- `--max-steps N` (30, at least 1) is the most steps APAI's planner may take for one task.
- `--camera link` (the default) takes stereo pairs from the camera receiver. `--camera mock`
  is APAI's mock camera, a made-up scene with a red cube. `--camera recorded:<dir>` reads
  `left_raw` and `right_raw` images from a folder and hands them out again and again.
- **The exposure times of `--camera mock` and `--camera recorded:<dir>` are invented**: each
  camera frame is dated at the moment it is handed out, so a permit would rest on evidence
  that nobody captured. With `camera.allow_invented_times: false` (the shipped value) the
  program does not start on such a camera: exit code 2 and one line that names the setting.
  With `true` it prints a `WARNING` line at its start, and every such camera frame carries
  `invented_times: True` in its capture record. With `true` the program runs only against a
  robot app whose WELCOME declares a simulated driver (`sim` or `apai-mock`): against `real`,
  or a robot app that declares no driver, it ends with exit code 1 once the robot connection
  is up, before any task. The check is made again before every task's question, on the
  WELCOME of the session the task would quote. A robot app that was restarted with a driver
  that is not simulated gets no question and no owner request: the program says why, takes
  no new task and ends with exit code 1. A task whose session has no WELCOME within
  `task.owner_wait_s` is refused the same way, whatever the camera and the stack. The line
  `robot connection up: ... driver ...` is printed again whenever a WELCOME says something
  else. The minimal stack (`experiment.stack: minimal`) is refused the same way, whatever
  the camera: its check ahead computes APAI's mock arm model.
- `--check` loads and checks `agent.yaml` and APAI's configuration, attaches to nothing,
  prints the resolved values and the camera calibration id as JSON, and writes nothing into
  `state_dir`. In that JSON `obstacle_grid.stand_in` is `true` when the file sets
  `experiment.grid: clear`. The **clear grid** is a stand-in for the obstacle grid that
  reports nothing near: it answers in place of an obstacle grid bridge, and no move is
  checked against obstacles (see `experiment` in the table of `configs/agent.yaml`
  below). `--check` prints no `WARNING` for it; the run does.
- `--run-dir PATH` names this run's folder; `--log-level` is `debug`, `info`, `warning` or
  `error`.
- The seven arguments of the last two lines are those of the experiment folder's own
  launcher; see
  [The options of the experiment's launcher](#the-options-of-the-experiments-launcher).

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
| 1 | No robot link service, the robot app not ready in time, another camera calibration or arm profile than the robot app's (at the start, or before a later task after the robot app was restarted), a `rig` list that is not of the arm's joint count, APAI's runtime not built (for example no usable obstacle grid within `task.grid_wait_s`), a task that ended without success (`--interactive`: any task of the session), an owner request at a task's end that was not answered, the agent still control owner when the program ended, invented exposure times or the minimal stack against a robot app that declares no simulated driver (at the start, or before a later task after the robot app was restarted), a task whose session has no WELCOME within `task.owner_wait_s`, a skill stop that was not cleared (the program said `restart the agent program`), or SIGTERM: a STOP was sent first |
| 2 | The arguments, `agent.yaml` or the environment cannot be used, APAI cannot be imported, or a recording of the experiment cannot start. Nothing was attached |
| 130 | A second Ctrl+C. A STOP was sent again first |

A task the operator cancelled with Ctrl+C does not count as a task without success. Two of
the causes of exit code 1 are read from the code and were not run: a robot app that is not
ready within `link.connect_timeout_s`, and an obstacle grid that is not usable in time.

### The options of the experiment's launcher

The **experiment's launcher** is `thor_runtime.sh` in the experiment folder: one program
on the Thor that runs the agent and talks to the motors itself. The agent program takes
that launcher's arguments, so the launcher's command line can be given to `run_agent.sh`
as it stands:

```bash
# The experiment's launcher, on the Thor:
#   ./thor_runtime.sh --operate --interactive --record-debug --depth-backend ffs \
#       --motion-approval task --batch-milestones
# The agent program, on the workstation:
./run_agent.sh --config configs/agent_rebot.yaml --operate --interactive --record-debug \
    --depth-backend ffs --motion-approval task --batch-milestones
```

| Argument | What it does |
|---|---|
| `--batch-milestones` | The experiment's milestone batching: one model request for the **milestones** (the steps the agent splits a task into) that are checked on one camera frame |
| `--record-debug` | The experiment's **debug recording** in `<run folder>/debug`: what went into the obstacle grid (camera frames, depth), masks and map revisions. It needs the timing: with `--no-timing` or with `experiment.timing: false` the program ends with exit code 2 and `--record-debug needs the timing; ...` |
| `--debug-max-gib GIB` | The size on disk at which the debug recording stops (16). It must be a finite number above 0. Without `--record-debug` it does nothing |
| `--no-timing` | This run goes without the experiment's timing |
| `--motion-approval task` | One typed `yes` before the first skill that moves in each task (the launcher calls such a skill a motion tool). The program does that with or without the argument. `--motion-approval tool`, a `yes` before every such skill, is not built: it ends the program with exit code 2 and `--motion-approval tool is not built: ...` |
| `--operate` | Accepted and ignored: every run of the program may move the arm, after the typed `yes` |
| `--depth-backend ffs` | Accepted and ignored; `ffs` is the only value the argument takes. With the full stack and `--camera link` the depth backend is the one the experiment's `configs/sensing_thor.yaml` names (`ffs`), whatever `camera.depth_backend` says |

- **Two recordings run in every run of the full stack**, as in the experiment's
  launcher: the **timing** (`timing.jsonl` in the run folder; `timing-report.txt` and
  `timeline.html` when the program ends) and the **model transcript** (`model-trace/` in
  the run folder: every request to the model and its answer). Their file keys are
  `experiment.timing` and `experiment.model_trace`, both `true` in `configs/agent.yaml`
  and `configs/agent_rebot.yaml`; `false` switches one off.
- **With `experiment.stack: minimal` none of this runs**, and `--batch-milestones`,
  `--record-debug`, `--debug-max-gib` and `--no-timing` are refused (exit code 2, `...
  needs experiment.stack full (it is minimal): ...`): the minimal stack loads none of
  the experiment's modules.
- **What the workstation cannot record.** The motors are read and commanded by the robot
  app on the Thor, and the camera captures there. So the motor reads and sends and the
  camera's capture are not timed: a move is timed from its MOVE to its final RESULT, and
  a stereo pair from the wait for it to its depth. The debug recording holds no motor
  feedback and no motor commands (`robot.feedback`, `robot.call`, `robot.return`,
  `mask.state_source`). The program says so in its first lines, which begin `timing:`,
  `model transcript:`, `debug recording:` and `milestone batching:`.
- **The recordings start before the program attaches to anything.** One that cannot
  start ends the program with exit code 2 and `the experiment's recordings cannot start
  (...); nothing was attached`. A `debug` folder that is already in the run folder is
  such a case, so one `--run-dir` takes `--record-debug` once.
- **A STOP comes before any report.** The recordings stand around the whole robot
  connection: on Ctrl+C the STOP goes out and the local socket is closed before a
  recording writes its report. A recording that cannot end leaves the run's exit code
  as it was and says `a recording of this run did not end as it should: ...`.
- **Disk.** The debug recording stops at `--debug-max-gib`, and the experiment's
  recorder stops by itself with less than 2 GiB free.
- **`--check` with the same arguments** prints what a run would take under
  `run_options`. A run that takes any of it writes that into `run-options.json` in its
  run folder.
- **What differs from the launcher's `--motion-approval task`.** The launcher's `yes`
  also covers a return to rest inside the task; here APAI's `move_to_rest` skill is
  refused and the return to rest is the operator's. And here the `yes` is followed by
  the robot app's grant of the arm.
- **Not taken**: the launcher's `--speed-profile`, `--settings` and `--mounting` are
  unknown arguments (exit code 2). The joint speed is `task.max_joint_speed` of the
  agent's file, the experiment's `configs/thor_runtime.yaml` and `configs/thor_cloud.yaml`
  are always read from the experiment folder, and the camera's mounting is
  `camera.calibration_dir` and `camera.hand_eye_json`.

None of these options ran on the real full stack, with its GPU, its obstacle grid bridge
and its model client. The experiment's own timing, model transcript, debug recording and
milestone batching ran in one test on APAI's mock scene, with no GPU, no model request
and no camera (`test_agent_options.py`).

### A task from its start to its end

1. **Thinking and looking take no arm.** APAI's read-only tools (observe, list objects, recall
   and the like) run without ownership and without a question to the operator.
2. **The first skill that moves asks the operator.** When an earlier task of the program
   has ended its ownership, the program first waits up to `task.owner_wait_s` (2 s) for the
   ROBOT_STATE that shows that ownership as ended: a task typed right after the last one
   would otherwise be refused on a ROBOT_STATE that still shows the agent as control owner.
   Then it reads ROBOT_STATE
   and refuses, without a question, when a person owns the arm or a takeover is open
   (`takeover not finished: give the arm back with right A at rest`; see
   [Demo 2: a person takes over](#demo-2-a-person-takes-over)), the control owner is not
   nobody, a fault-stop is latched or the motors are off. Otherwise it prints one question
   that names the task, the first skill and the
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
   it is. The return to rest is the operator's: `rest` at the Thor terminal or at the
   operator console. APAI's own `move_to_rest` skill is refused.
5. **A task that ended any other way while the agent owned the arm asks for help**:
   OWNER_REQUEST "help_needed" with the reason, which starts the assistance wait on the Thor.
   The reasons are `stuck: ...` (APAI gave up, or `task.failed_steps_limit` skills failed in a
   row), `blocked: ...`, `step budget ran out`, `agent error: ...` and `task ended without
   success: ...`. With `task.timed_return: false` the Thor makes no return to rest when the
   wait runs out.
6. **The local socket is closed after that request's RESULT**, once ROBOT_STATE no longer
   shows the agent as control owner. When the request gets no RESULT within
   `task.end_wait_s`, the program says `was not answered`, sends nothing else and ends with
   exit code 1. It then waits up to 3 s for a ROBOT_STATE without the agent as control
   owner. If the agent is still the control owner after that, the socket closes anyway and
   the program says so; the service then asks for help in the agent's name
   (`agent program gone`). With
   `--interactive` a task typed while the agent still owns the arm that way is refused before
   its question, and the refusal names the remedy: the console command `stop`.
7. **Ctrl+C is a STOP.** The first one cancels the task: a running move is ended through its
   permit, then a STOP goes out with the sender `agent_terminal`. Nobody owns the arm, no help
   request is sent and no assistance wait starts. A second Ctrl+C sends STOP again and ends
   the program.
8. **A STOP from the console ends the task as well.** The running move ends `cancelled` and
   APAI's skill ends as halted. Nobody owns the arm then, so the task's help request, which
   quotes the stop number from before the STOP, is refused. The next task can take the arm;
   the program needs no restart.
9. **After a skill stop the next task clears the workstation checker's latch.** The task
   itself ended as in step 5, and the program says `the task ended in a skill stop: the next
   task clears it after its typed yes`. The next task's question begins `Note: the last task
   ended in a skill stop; it is cleared when you confirm`. After the typed `yes` and the
   Thor's grant, with a ROBOT_STATE at most 1 s old that shows the agent as owner and stop
   level `none`, and with no STOP at the robot link service that the robot app has not shown
   yet, the gate resets the latch and the task goes on. A STOP typed at the Thor terminal that
   no ROBOT_STATE shows yet cannot be seen from the workstation; the Thor refuses the task's
   commands then. Nothing on the Thor is cleared by it: under a
   fault-stop the task is refused before the question, and the latch stays. The program says
   `restart the agent program`, takes no new task and ends with exit code 1 in three cases:
   the latch did not clear (the arm is given back first), the checker cannot say whether it
   was stopped again since the question (the arm is given back first), or
   `task.clear_skill_stop` is `false`. With `false` a latch seen at any point of a task's
   start refuses the task and clears nothing; one seen before the question refuses the task
   without a question. A stop of the workstation checker that comes after the question is not
   erased: the latch stays, or is set again when the stop came while the gate cleared it; the
   arm is given back and the next task's question names it. That also holds for a stop right
   after the gate's last look at the checker: the gate looks once more when the task is
   opened, and while the checker is latched the arm adapter sends no jaw command and no move.
   A give-back that gets no completed RESULT is tried once more: before the next task takes
   the arm, at the task prompt and when the program ends.

The agent program sends OWNER_REQUEST and STOP; its arm adapter sends MOVE, PERMIT, GRIPPER
and CHECKER_STATE. It sends no RELEASE, SHUTDOWN, RESET or OPEN and no request for a return
to rest, and it cuts no torque.

### A move, its permits and the camera evidence

- **One MOVE per move of APAI.** The arm adapter reads the start pose from joint readings taken
  after the call began, samples the move at the Thor's setpoint period, and has the check ahead
  vet every setpoint on an obstacle grid at most `move.start_grid_age_s` old. A refusal here
  sends nothing.
- **A MOVE expires before its first setpoint, not after it.** It carries
  `expires_mono_s`: `move.expires_in_s` (2 s) after the arm adapter sent it, on the Thor's
  monotonic clock. At or after that time the Thor refuses the MOVE with `the move expired
  X.XXX s ago` and sends the arm nothing. It asks when it takes the MOVE up, while it
  waits for the first permit, and once more after the check of the first setpoint, right
  before that setpoint goes out: a check that took long does not start a move that
  expired meanwhile. A STOP at that moment wins: the move ends `cancelled`. Once the first
  setpoint is out the expiry is not asked again; from there the permits bound the move.
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
  The comparison is made again before each task's question, with the rest of what the
  program compared in the WELCOME at its start (protocol version, joint count, setpoint
  period, `link.expected_profile_id`): a robot app that was restarted with another
  calibration or profile gets no owner request, and the program takes no new task and ends
  with exit code 1.
- **A camera frame is dated at its exposure.** The stereo camera refuses a stereo pair when
  the newest ROBOT_STATE is older than 1 s, its exposure lies more than 0.05 s in the future
  of the Thor's wall clock, more than 0.05 s plus the link's clock margin (0.02 s) after its
  arrival on the workstation, or more than 1 s before that arrival, or the Thor's wall clock
  stepped within the last 3 s. The evidence time a permit quotes is never later than the
  pair's arrival. The self mask uses the joint samples of ROBOT_STATE on both sides of the
  exposure; a camera frame without them puts nothing into the obstacle grid.

### `configs/agent.yaml`

Every number in the shipped file is a first value; none was measured on a robot. A relative
path is taken from the file's folder. A path (`state_dir`, `link.socket_path`, five keys of
`camera`: `bridge_python`, `camera_pr`, `receiver_socket`, `calibration_dir` and
`hand_eye_json`, and `experiment.codex_bin`) may name `${AGENTIC_REMOTE_EXPERIMENT_DIR}`,
`${AGENTIC_REMOTE_CAMERA_PR}` or `${AGENTIC_REMOTE_ENV_DIR}`; another `${NAME}`, one that is
not set, one whose value is not an absolute path, and any other `$` in a path (`$NAME`
without braces, a `${NAME` that is not closed) are refused, and the message names the key. So
is a path that begins with `~name` where `name` is no user of the machine. The program
refuses the file (exit code 2) when it
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
| `jaws` | `fresh_s`: an older jaw reading counts as unknown; `result_timeout_s` (12): how long the arm adapter waits for the final RESULT of a jaw command, and how long that GRIPPER stays valid on the Thor. Without a RESULT in that time the adapter gives up: it sends a STOP with the sender `agent`, the agent is told `no RESULT for the jaw command within 12 s; a STOP was sent`, and the rest of the task sends nothing (see [What the jaws do](#what-the-jaws-do)) |
| `task` | `timed_return` (false); `failed_steps_limit` (3); `max_joint_speed`, APAI's speed cap for the task, at most `thor.agent_max_joint_speed`; `end_wait_s`, `owner_wait_s` and `grid_wait_s`, the longest waits for an owner request's RESULT, for the ROBOT_STATE that shows the agent as owner, and for a usable obstacle grid at the start; `clear_skill_stop` (true), see step 9 of "A task from its start to its end" |
| `camera` | `bridge_python`, the camera consumer's interpreter; `camera_pr` (empty: `AGENTIC_REMOTE_CAMERA_PR`); `receiver_socket`; `streams`; `calibration_dir` and `hand_eye_json`, the workstation's copies of the stereo calibration folder and the camera mounting file in use on the Thor; `max_skew_ms` between the two eyes; `depth_backend` (`ffs` on the GPU, `opencv_cpu` without one); `fps`; `allow_invented_times` (false), see `--camera` above |
| `experiment` | `stack`: `full` uses the experiment folder's behaviour hooks, its obstacle grid bridge on `cuda:0` and its cloud model; `minimal` uses none of them, builds on APAI's own profiles `arm_profile` and `llm` (both `mock` in the shipped file) and pins APAI's mock detector for every camera, so no model server is asked; `grid`: `bridge` (the shipped value) asks an obstacle grid bridge, `clear` puts the clear grid in its place, which checks nothing against obstacles and is taken only with the minimal stack and against a simulated driver (see [Run demo 1 by hand with the mock agent](#run-demo-1-by-hand-with-the-mock-agent)); `codex_bin`, the model client's program for the full stack (empty: the one the experiment's own file names, inside the experiment folder), refused with exit code 2 when it names no program that can be run; `timing` (true) and `model_trace` (true), the experiment's two recordings of a run, read by the full stack only (see [The options of the experiment's launcher](#the-options-of-the-experiments-launcher)) |
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
checkout or the experiment folder, and unsets `LD_LIBRARY_PATH`. It sets
`GIT_OPTIONAL_LOCKS=0`, so that a `git status` the experiment runs on its tools does not
rewrite their index. With `AGENTIC_REMOTE_ENV_DIR` set it points `TRITON_CACHE_DIR`,
`TORCHINDUCTOR_CACHE_DIR` and `YOLO_CONFIG_DIR` into `<folder>/caches`, each one unless it
is set and not empty; the script exits with code 2 and one line when that folder is not an
absolute path or holds a `.` or `..` component. Without `AGENTIC_REMOTE_ENV_DIR` the script
sets none of the three, and the agent program puts each one that is not set into
`<state_dir>/caches` for its run. With or without it, the script exits with code 2 and one
line when one of the three, as the program will get it, is not an absolute path, holds a `.`
or `..` component, or names a place inside `AGENTIC_REMOTE_EXPERIMENT_DIR`; symbolic links
are followed for that comparison. The agent program makes the same comparison itself (exit
code 2), also when it is started without the script and for `--check`, and refuses a
`state_dir` whose caches would lie inside the experiment folder.

`run_camera_receiver.sh --host THOR_ADDRESS` runs the camera streamer's
`camera_receiver/run.sh` with `configs/camera_receiver_rebot.yaml`. It needs
`AGENTIC_REMOTE_CAMERA_PR` and exits with code 2 and one line without it. A first argument
that does not begin with `-` is another configuration file; the other arguments go to the
receiver.

`run_workstation.sh` starts the workstation's programs of demo 1 on the rig together, in
one tmux window with five panes that are visible at once, each with its name on its
border:

```text
bash run_workstation.sh [--env FILE] [--thor ADDRESS] [--session NAME] [--print]
```

| Pane | Its command |
|---|---|
| `console` | `./run_console.sh` |
| `agent` | `./run_agent.sh --config configs/agent_rebot.yaml --interactive` |
| `service` | `python -m isaaccapture_examples.agentic_remote.link_service --config configs/link_service_rebot.yaml`, under the Python of `AGENTIC_REMOTE_AGENT_PYTHON` |
| `view` | `./run_view.sh --config configs/view.yaml` |
| `camera` | `./run_camera_receiver.sh --host ADDRESS` |

- **The service, the console and the view start at once.** The camera receiver starts
  at once when `--thor ADDRESS` gives the Thor's address; without it the command is
  typed into its pane with `THOR_ADDRESS` in the address's place and waits there.
- **The agent's command waits for Enter.** It is typed into the pane `agent` and not
  run: press Enter there once the console shows the robot connection up.
- **`--env FILE`** names a shell file of `KEY=value` lines that sets the six variables
  below. They may be set in the environment instead, and `--env` left out. **`--session
  NAME`** names the tmux session (`agentic_remote`). **`--print`** prints the five
  commands and starts nothing.
- **The six variables**: `AGENTIC_REMOTE_AGENT_PYTHON` (the Python of the workstation
  environment; the script also gives it to the console and the view as
  `AGENTIC_REMOTE_CONSOLE_PYTHON` and `AGENTIC_REMOTE_VIEW_PYTHON`),
  `AGENTIC_REMOTE_ENV_DIR`, `AGENTIC_REMOTE_ISAAC_CAPTURE_PY` (the folder of the
  Isaac Capture Python package built for that Python), `AGENTIC_REMOTE_APAI_SRC`,
  `AGENTIC_REMOTE_EXPERIMENT_DIR` and `AGENTIC_REMOTE_CAMERA_PR`. Each pane's command
  carries them itself, so a tmux server that runs already changes nothing.
- **Exit code 2 and nothing started**: an unknown argument, a variable that is not set,
  a Python that cannot be run, a folder that is none, no `tmux` on `PATH`, or a tmux
  session of that name that exists already.
- A pane keeps its last lines when its program ends, and a click selects a pane. The
  script does not show the window itself: it prints `attach with: tmux attach -t
  agentic_remote`. The file has no execute bit: start it with `bash`.
- **Its start of tmux ran in no test and was not run by hand.** What a test runs is its
  syntax check, `--print`, and its refusals of an unknown argument, of a variable that is
  not set and of a Python that cannot be run.

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

## The operator's programs

The operator has three programs on the workstation: the operator console, the console
command and the live view. Each is a local client of the robot link service. At the robot
there is the Thor terminal.

### What the operator types to confirm

One table for the Thor terminal, the operator console and the console command. Case and
spacing do not count; anything else confirms nothing. `stop` typed where a question waits is
a STOP and confirms nothing.

| Command | Where | Typed confirmation |
|---|---|---|
| `stop`, `reset`, `state`, `help` | everywhere they exist | none |
| `rest` | Thor terminal, console, console command | `path is clear` |
| The shutdown: `quit`, `shutdown` | `quit` at the Thor terminal; `shutdown` at the console and the console command | `path is clear` when the arm has to move to rest. At rest the console and the console command ask for `yes`, and the Thor terminal asks nothing |
| `open` | console, console command | `yes` |
| `release` | console, console command | `yes`. At the Thor terminal `release` asks nothing |
| `takeover`, `handback` | console, console command | `yes`. The robot link service forwards them only with `operator.typed_takeover: true` in its file |
| `reference` | Thor terminal only | `jaws are empty` |
| A task's first skill that moves | agent program | `yes` |
| The real driver's three questions | Thor terminal, at the start | `yes` each |

A confirmation is typed at a terminal. The operator console, the console command and the
agent program ask nothing and send nothing without a terminal on standard input and output
(a pipe, a script), and inside the environment of an AI agent. The robot app with `--driver
real` needs a terminal on standard input and output and refuses the environment of an AI
agent. The Thor terminal of a robot app with `--arm sim` or `--driver mock` checks neither:
it takes its commands and their confirmations from a pipe too.

What was typed before a question was written is dropped, for every question in the table,
when the input is a terminal. Lines that the Thor terminal reads from a pipe are all taken,
in their order.

At the Thor terminal's `rest`, `quit` and `reference`:

- The question is written first. One that is not written within 1 s (the terminal's output
  is stopped with Ctrl+S, or the terminal is gone) confirms nothing; what is typed meanwhile
  is read as commands.
- The terminal says how many lines it dropped. A `stop` among them is not lost: it is a
  STOP, and the question is withdrawn. A `stop` that was begun and not yet entered when the
  question was written goes with the rest; Ctrl+C is the STOP that is never dropped.
- A last line without a line end confirms nothing, from a terminal and from a pipe, as at
  the real driver's questions. `stop` without a line end is still a STOP.

### The operator console

```text
./run_console.sh [--socket PATH] [--name NAME] [--log-level L]
```

`run_console.sh` runs `python -m isaaccapture_examples.agentic_remote.console` with its
arguments. The console needs Python 3.10 or newer and nothing else: no APAI, no torch and no
Isaac Capture. `AGENTIC_REMOTE_CONSOLE_PYTHON` names the Python (default `python3`). It
attaches to the robot link service in the role `console`. It needs a terminal and refuses to
start without one or inside the environment of an AI agent (exit code 2). Its exit code is 0
after `quit` or the end of the input.

What it prints, one line each:

- `WELCOME: ...` with the robot, the profile, the session, the driver and the calibration.
- A **status line** when something changes, at most once a second, and at least every 10 s:
  the robot connection, the attached clients, the driver, the control owner with the owner
  number, the stop level with the stop number, the motion with the move's progress, the
  jaws, the motors (`torque on` or `torque off`; `torque unknown` while the motion is
  `release_failed`, after a torque release that failed), the assistance wait, the held
  permit, the view age, the parts about a
  person at the controller (below) and the age of the ROBOT_STATE it shows. An owner that
  lasts less than a second can be missing from the status lines; the EVENT lines name every
  change.
- Every EVENT of the robot app as `[kind] text`, with the terminal bell: once for a bell of
  the assistance wait and for the end of a human drive, twice for a takeover that began or
  ended, three times for a refused takeover or hand-back. The wearer of a headset cannot
  read the console, so the count tells the three apart. The bells of one line are written
  together in front of it; whether a terminal rings them apart was not tried.
- A **jaws line** after each status line while ROBOT_STATE carries `health.human.jaws`
  and the robot connection is up: `jaws: follow the trigger`, with `; <reason>` behind
  it while the jaws are held wider than the trigger asks; `jaws: trigger asks X, grip
  level Y: bring the trigger to the grip level`, X and Y as shares of the jaw travel;
  or `jaws: the trigger moves nothing: <reason>`. The rule behind it is under
  [A takeover, step by step](#a-takeover-step-by-step).
- A help request as a block: the reason, the jaws, the time left, what can be typed and
  the takeover line (below).
- A refusal another client got, marked `(another client)`; notices of the robot link
  service; and `ROBOT CONNECTION LOST: ...` with the bell.

| Typed | What it does |
|---|---|
| `stop` | STOP: the arm holds where it is, motors on; it ends a running task |
| `reset` | RESET: clears a fault-stop; starts nothing |
| `rest` | OWNER_REQUEST "rest": the robot app returns the arm to rest |
| `open` | OPEN: the guarded jaw opening; what the jaws hold is let go |
| `release` | RELEASE: motors off; the robot app does it only at measured rest |
| `shutdown` | SHUTDOWN: return to rest if the arm is away from it, motors off there, the robot app ends |
| `takeover` | OWNER_REQUEST "human": a person becomes the control owner, as with right B held 1 s. The typed fallback for the button; the robot app takes it only while nobody owns the arm, so type `stop` first |
| `handback` | OWNER_REQUEST "handback": ends a takeover with the arm at rest, as with right A held 1 s. The typed fallback for the button |
| `state` | Prints the service's status, the newest ROBOT_STATE, a status line and the takeover line; sends nothing |
| `attach` | One more attempt to attach to the robot link service |
| `help` | The list of commands |
| `quit`, or the end of the input | Leaves the console. Nothing is sent and the robot app runs on; the console's last line says so: `the operator console ends; nothing was sent for that: the robot app runs on`. While the newest LOCAL_STATUS shows no robot connection, lost or never made, the last line is `the operator console ends; nothing was sent for that`: the console knows nothing of the robot app then |
| Ctrl+C | Takes back an open question, then sends a STOP. It never ends the console. After `quit` or the end of the input a Ctrl+C is still a STOP, written raw, while the console detaches and ends its threads; that takes up to about 2 s when an attempt to attach waits for a service that does not answer |

- **A command that asks**: the console prints the newest ROBOT_STATE, then the question. The
  answer is the next typed line; what was typed ahead is dropped. A `stop` among the whole
  lines typed ahead is not lost: it is a STOP, and the question is taken back. The command
  quotes the numbers of the ROBOT_STATE that was printed, so an answer typed for another
  state of the arm is refused by the robot app.
- **A command does not block the prompt.** While a return to rest runs, `stop` is taken.
  While the robot connection is up and no ROBOT_STATE has arrived yet, `reset`, `rest`,
  `open`, `release` and `shutdown` answer at once `no ROBOT_STATE has arrived` and send
  nothing; type the command again. The attempts to attach, the first one and the one a typed
  `attach` asks for, run beside the prompt: a `stop` typed meanwhile is read at once.
- **`stop` while attached** sends one STOP. It is answered by `stop: taken` once a
  ROBOT_STATE of the same session shows a higher stop number. A ROBOT_STATE of a new session
  confirms nothing: the console prints `STOP NOT CONFIRMED` at once, with the bell. Without
  a higher stop number within 5 s of the first STOP that waits, it prints `STOP NOT
  CONFIRMED` with the bell and names the Thor terminal: type `stop` or press Ctrl+C there.
  A `stop` typed again meanwhile sends another STOP and does not move that time. A STOP
  sent before any ROBOT_STATE arrived is measured against the first one that arrives. A
  STOP waits at most 0.2 s for a robot link service that does not read the console's socket,
  or for another command's send that is stuck there; it is then written raw, on a connection
  of its own. A STOP that goes out as the console ends is said so: `stop: sent; not
  confirmed: the console ends, look at the Thor terminal`.
- **`stop` while not attached** writes a raw STOP and prints `STOP written; not confirmed:
  look at the Thor terminal`, or `STOP NOT written` with the reason.
- **Turned away at its start** (no service, or too many clients), the console stays up with
  `stop`, `attach`, `help` and `quit`, and tries to attach every 2 s, 30 times; after that
  only on a typed `attach`.
- **Release and shutdown are checked on the Thor, not here.** The console only sends the link
  message. The robot app releases torque only with the arm measured at rest and the jaws
  proven empty (see [the torque rule](#the-rule-torque-is-never-cut-away-from-rest)); a
  `release` with the arm away from rest is `refused` and the motors stay on.
- **`takeover` and `handback` are the typed fallback** for right B and right A of the
  controller (see [Demo 2: a person takes over](#demo-2-a-person-takes-over)). Each asks for
  a typed `yes` and sends its OWNER_REQUEST with the reason `typed at the operator
  console`. With `operator.typed_takeover: false` in the robot link service's file, the
  shipped value, the service refuses both and the console prints `takeover: RESULT refused:
  the console role may not send OWNER_REQUEST "human"; use right B / right A on the
  controller`. The robot app applies the same checks to a typed request as to one from the
  buttons.
- **A refusal line that repeats is printed once per second**, and its repeats are counted
  (`N more like: ...`). A refused takeover or hand-back, which rings three times, is
  counted apart from the other refusals, so a flood of those does not silence it. The same
  refused request twice within one second rings once.

**The status line's parts about a person at the controller** follow the view age, each only
while it applies:

| Part | Shown while |
|---|---|
| `human program at the robot app` | ROBOT_STATE says that a human program is attached at the robot app's own socket (`health.human.local_program`), as on the Thor. The part `clients` of the status line counts only what is attached to the robot link service, so it shows no `human` then |
| `no human program` | `human.enabled` is true in the robot app's file or a takeover is open, and no human program is attached at either place: none to the robot link service and none at the robot app's own socket |
| `headset <stage>` | Otherwise, while ROBOT_STATE carries a stage of the human program: `ready`, `no_controller`, `stale`, `no_server`, `connecting_openxr` or `waiting_for_headset` |
| `controller X s` | The age of the controller data is known |
| `human driving` | A human drive runs. The motion part then reads `moving N/0`: N setpoints sent, and no total |
| `takeover open` | A takeover was accepted and no hand-back has ended it |
| `needs release` | A person owns the arm and the grip has to be released before a squeeze moves it |
| `contact hold` | The arm holds at a contact |
| `note: <text>` | The human program gave a note: what the person does next, or why the arm does not follow. It is shown only while ROBOT_STATE carries a stage, so the note of a human program that left can stand for `human.stage_keep_s` (4 s) more |

**The takeover line** says whether a person can take the arm now, or how an open takeover
ends. It is the last line of the help block and of what `state` prints; a new status line
is printed when the robot app's reason in it changes. Its texts, the first that applies:

| While no takeover is open | When |
|---|---|
| `takeover: not possible: the robot connection is lost` | The robot connection is down |
| `takeover: not possible: human driving is switched off in this robot app's file (human.enabled)` | The robot app says so. It stands before the missing human program: starting one would change nothing |
| `takeover: not possible: no human program is attached: start run_human.sh` | The robot link service counts no client in the role `human`, and ROBOT_STATE reports none at the robot app's own socket |
| `takeover: not possible: no ROBOT_STATE` | None has arrived |
| `takeover: not possible: this robot app does not report it` | ROBOT_STATE has no `health.human.takeover` |
| `takeover: not possible: <reason>` | The robot app's own word, `health.human.takeover`. The first of: a fault-stop is latched; the motors are off; `no fresh controller data: ...`; `a synthetic controller may drive only a simulated arm; ...`; `the live view is not fresh: ...` |
| `takeover: possible now: hold right B for 1 s with the grip released` | None of the above |

| While a takeover is open | When |
|---|---|
| `takeover open: the robot connection is lost` | The robot connection is down; the takeover is the one the last ROBOT_STATE showed |
| `takeover open: a fault-stop is latched: type reset first` | A fault-stop refuses the hand-back and a new takeover alike |
| `takeover open: no controller data. To end it: type rest, then path is clear; then start the human program and hold right A, or see the README, "A takeover that cannot be ended with right A"` | The stage is not `ready`, or no human program is attached. That heading is [below](#a-takeover-that-cannot-be-ended-with-right-a) |
| `takeover open: at rest: the jaws are not proven empty: type stop, then open, then yes, then hold right A for 1 s` | The arm holds at rest and ROBOT_STATE shows the jaws as `loaded`. The hand-back needs them proven empty |
| `takeover open: at rest: hold right A for 1 s to give the arm back` | The arm holds at rest and ROBOT_STATE does not show the jaws as `loaded`. With `unknown` jaws the hand-back reads them anew; if it refuses, its RESULT says why |
| `takeover open: a person owns the arm` | A person owns it, away from rest |
| `takeover open: nobody owns the arm: a new takeover is not possible: <reason>` | A STOP made nobody the control owner and the robot app would refuse right B, for example for a live view that is not fresh |
| `takeover open: nobody owns the arm: hold right B for 1 s to take it again` | A STOP made nobody the control owner |
| `takeover open: recovery owns the arm` | A return to rest runs |

With the shipped `configs/robot_app.yaml` (`human.enabled: false`) the help block of demo 1
ends with `takeover: not possible: human driving is switched off in this robot app's file
(human.enabled)`.

### The console command

```text
python -m isaaccapture_examples.agentic_remote.agent.console_cmd
    [--socket PATH] [--timeout S]
    state|stop|reset|rest|open|release|shutdown|takeover|handback
```

It sends one command of the operator console and ends. The order is: attach in the role
`console`, read the newest ROBOT_STATE, print it, ask for the typed confirmation, send, print
the answer. The commands, their confirmations and their checks on the Thor are those of the
console; `takeover` and `handback` are refused by the robot link service in the same way
where its file does not allow them. Its `state` does not print the takeover line.

`stop` is sent at once after the attach: it waits for no ROBOT_STATE, and the higher stop
number is measured against the first ROBOT_STATE the command receives. `stop` that cannot
attach within 0.5 s (the other commands wait 5 s) writes a raw STOP, says whether it was
written, and ends with exit code 1: nothing confirmed it.

Exit code 0: the command's final RESULT was `completed` (`state`: a ROBOT_STATE was there;
`stop`: the higher stop number was seen in the same session). 1: refused, failed, not
answered in time, not confirmed (for `stop` also: the robot app showed a new session), or no
robot link service. 2: a command that does not exist.

### The live view

```text
./run_view.sh --config configs/view.yaml [--source link|recorded:<dir>|none] [--port N]
    [--log-level L]
```

The live view serves one page for a browser on the same machine: `http://127.0.0.1:47620/`
(`http.port` of its file; another address than 127.0.0.1 is refused). The page shows the
camera picture, made small, and under it the robot connection, the driver, the control
owner, the stop, the move, the held permit, the jaws and the last events. **The view only
looks**: VIEW_STATUS is the only link message it sends. Nothing was looked at in a browser:
the tests ask the page's addresses over HTTP, so the page's script ran in no test.

- **Start the robot link service first.** The view attaches in the role `viewer`. One view at
  a time: a second one ends.
- **`--source link`** (the default) takes the pictures from the camera receiver through a
  camera frame bridge of its own: a second reader of the receiver, no second decoder. That
  is one more process on GPU 0; the view itself needs no GPU. The bridge's log is
  `<state_dir>/view/frame_bridge.log`. The source is opened up to 12 times, 5 s apart; after
  that the page shows the status only. This source was never run.
- **`--source recorded:<dir>`** shows stereo pairs from a folder and is never live.
  **`--source none`** shows the status only.
- **The view is live** only while five things hold: the page that was loaded last
  acknowledged, with its tab visible and less than `status.ack_timeout_s` ago, a picture of
  the camera connection; that page fetched that picture less than `status.ack_timeout_s`
  ago; a ROBOT_STATE younger than `status.state_max_age_s` gives the Thor's clock; that
  picture arrived 3 s or more after the view's first ROBOT_STATE and after the newest step
  of the Thor's wall clock (more than 0.02 s against its monotonic clock between two
  ROBOT_STATEs); and that picture's exposure lies at most 0.05 s in the future of the
  Thor's clock. A picture that arrived earlier can be dated on the wall clock from before
  the step and is never live, however long the page acknowledges it: after its start and
  after a step the view is not live for at least 3 s. A takeover is refused meanwhile, and
  a human drive ends in a hold, where the robot app requires the live view. The view age is
  the larger of two ages: since the picture's exposure, on the Thor's clock, and since its
  arrival on the workstation. The view sends VIEW_STATUS every `status.period_s`, also
  while it is not live.
- **The view token** is a secret the view gives to one page: the one whose
  acknowledgements count.
  1. Each load of the page (`GET /`) makes a new token, makes it the only valid one and
     forgets every acknowledgement and every picture served before. So one page at a time
     counts, and a reload makes the view not live until the new page has drawn a picture.
  2. The page sends the token as the header `X-View-Token` with every `GET /frame.jpg` and
     inside the body of every `POST /shown`: `{"seq": N, "visible": true, "token": "..."}`.
  3. `GET /frame.jpg` with the valid token is answered with the newest picture, and the
     view remembers that this picture went to this page, and when. Without the valid token
     the answer is 403 and no picture.
  4. `POST /shown` counts only with the valid token, for a picture that went to this page
     less than `status.ack_timeout_s` ago and that is not older than the one acknowledged
     last. A `"visible": true` for a picture that went to this page before the page last
     sent `"visible": false` does not count either: a tab that shows again is live with
     the next picture it fetches. What does not count is answered with 403 and changes
     nothing.
  5. A page whose token was replaced gets 403 for its next `GET /frame.jpg` and then shows
     `THIS PAGE WAS REPLACED: reload it`, asks for no more pictures and acknowledges nothing
     more. A 403 for `POST /shown` alone changes nothing on the page.
  6. The token is never logged and is not in `/status.json`, which needs no token. Of a
     request the view's log holds only the method, the route and the status code.
- **What the token closes, and what it does not.** It closes a stray `POST /shown` from a
  script or a test, a forgotten second tab and an acknowledgement of a picture that was
  never fetched. It fails closed: a reload, a second tab, a stray `GET /` and a restart of
  the view each make the view not live, which only ever holds the arm. It does not stop a
  program of the same user that loads the page and plays the browser, or that attaches to
  the local API as a viewer and sends VIEW_STATUS itself: the socket's file permissions and
  127.0.0.1 are the boundary, as for every other local client. And a visible tab says that
  the tab is in front, not that a person looks at it.
- **What the page writes over the picture**: `STALE`, `NO PICTURE`, `RECORDED, NOT LIVE`,
  `THIS PAGE WAS REPLACED: reload it`, and `VIEW PROGRAM NOT ANSWERING` when the page gets
  no answer from the program. In a band over the picture it writes `HUMAN DRIVING` while
  the newest ROBOT_STATE says that a human drive runs. That comes from the newest
  ROBOT_STATE, also an old one; the row `Age of the ROBOT_STATE` says how old it is.
- **The picture is not rotated.**
- **The view age gates a person's motion** wherever the robot app's file has
  `human.require_live_view: true`: without a live view, or with a view age above
  `human.view_age_limit_s`, a takeover is refused and a human drive ends in a hold (see
  [Demo 2: a person takes over](#demo-2-a-person-takes-over)). Nothing else acts on it.
  With the shipped rates (5 pictures per second, a VIEW_STATUS every 0.2 s) the age the
  Thor sees swings by about 0.4 s before any camera delay, so the shipped limit of 0.5 s
  is a first value: step F0 of the checklist chooses it from measured ages.

`run_view.sh` needs `AGENTIC_REMOTE_VIEW_PYTHON`, a Python with numpy, OpenCV and PyYAML
(the `view` extra of this folder's `pyproject.toml`); without it
`AGENTIC_REMOTE_AGENT_PYTHON` is taken. `--source link` also needs
`AGENTIC_REMOTE_CAMERA_PR`.

In `configs/view.yaml` five keys are paths under the rules of `configs/agent.yaml`:
`socket_path`, `state_dir` and, in `camera`, `bridge_python`, `camera_pr` and
`receiver_socket`. The same text in `socket_path` or `camera.receiver_socket` names the same
file for the view and for the agent program.

| Exit code | Meaning |
|---|---|
| 0 | Ctrl+C or SIGTERM |
| 1 | No robot link service, a second view, the port is taken, or the service ended |
| 2 | The arguments or the file cannot be used |

## Demo 2: a person takes over

When the agent fails, a person is prompted at the console, takes the arm over with the right
controller of a headset, moves it, brings it to rest and gives it back, and the agent's next
task runs. No program is restarted on the way. Nothing of this was run with a headset or on
the arm; [What was run](#what-was-run) says what the tests ran.

### Terms of the human path

- **Headset**: a Quest headset with its two controllers. **Right B, right A, grip,
  trigger**: controls of the right controller. In the controller data B is
  `secondary_click`, A is `primary_click`, the grip is `squeeze_value` and the trigger is
  `trigger_value`. **Stage space**: the coordinate system the headset gives the controller
  poses in. A **recentre** of the headset view sets it anew, which moves every controller
  pose at once.
- **CloudXR**: the streaming runtime through which a headset is connected to Isaac Capture
  on the workstation. The **CloudXR service** is the program that runs it
  (`python -m isaaccapture.cloudxr.service`). The **web client** is CloudXR's page that the
  headset loads to connect to it. **Headless** is a switch in the web client's panel:
  switched on, the web client draws nothing and keeps tracking.
- **Teleop Session Server**: Isaac Capture's workstation program that reads the headset and
  the controllers and sends them to one client
  (`python -m isaaccapture.teleop_session_server`). **SAMPLES**: its link message with the
  newest headset and controller data. **READY**: its link message that says a headset has
  been tracked for about a second. **Tick**: one pass of its loop; **tick age**: now minus
  the tick time of the newest SAMPLES.
- **Remote session**: a `TeleopSession` of Isaac Capture in `SessionMode.REMOTE`: a client
  of the Teleop Session Server that needs no headset software of its own. **Teleop
  connection**: the connection between the two. On the rig it crosses between the two
  machines, from the human program on the Thor to the server on the workstation; on one
  machine it stays on 127.0.0.1. Isaac Capture's texts and logs call the
  program that runs a remote session the "robot app". Here that program is the human
  program; in this README "robot app" always means the Thor program. So the server's log
  line `Serving robot app 'agentic_remote_human'` names the human program.
- **Retargeter**: an Isaac Capture node that turns device data into a command.
  **Retargeting graph**: such nodes connected.
- **Inverse kinematics (IK)**: from a pose of the gripper tip to joint values. **Forward
  kinematics (FK)**: the other way. Both are APAI's.
- **Human program**: the program in the role `human`. It reads the controller, runs the
  retargeting graph and the IK, and sends link messages. One program serves both
  placements, and the socket it attaches to decides which: on the rig it runs on the Thor
  and attaches at the robot app's own socket; on one machine and in the tests it attaches
  to the robot link service. A **pass** is one turn of its loop, 50 per second.
  **Controller feed**: where it gets controller data. The **teleop feed** is a remote
  session. The **script feed** is a **synthetic controller**: timed controller data from
  a script, with no headset.
- **Takeover**: the change of control owner to a person. **Hand-back**: the end of a
  takeover, at rest. **Takeover open**: a takeover was accepted and no hand-back has ended
  it yet.
- **Clutch**: the arm follows the hand only while the grip is squeezed. **Engage**: the
  moment the squeeze begins and every check allows motion. **Engagement**: the stretch from
  an engage to its end, as the human program sees it. **Re-base**: at each engage the
  hand's pose at that moment is matched to the arm's measured pose, so the arm never jumps
  to where the hand is.
- **Human setpoint**: the joint target the human path sends to the Thor, in a
  HUMAN_SETPOINT link message (type 99, newest-wins). **Human target**: what the Thor's
  side takes from it: joint targets with their sample age and the owner and stop numbers
  they quote. It says nothing about where it was computed. **Human drive**: one stretch of
  human motion on the Thor, the Thor's side of one engagement.
- **Sample age**: how old the controller data behind a human setpoint are when the Thor
  uses them: the age the human program reported as it sent the HUMAN_SETPOINT (for the
  teleop feed the tick age, plus the time the pass took from asking its feed to the
  send) plus the time since it was sent.
- **Fresh controller data**: the newest HUMAN_SETPOINT has the stage `ready` and a sample
  age of at most `human.sample_age_limit_s` (0.25 s).
- **Ahead**: how far a target lies ahead of what follows it. Two distances have a limit:
  the IK solution ahead of the last joint target the Thor sent (`ik.max_ahead_rad`), and
  the hand's target ahead of the speed-limited target (`ik.max_ahead_m`). **Jump**: a large
  difference between two IK solutions that follow each other.
- **Needs-release flag**: the Thor's note that the next human drive needs a released grip
  first.
- **Jaw opening**: how far the jaws are open, as a share of their travel: 1.0 is fully
  open, 0.0 closed. A HUMAN_SETPOINT carries the jaw opening the trigger asks for
  (`jaw_opening`), or none. **Grip level**: the jaw opening in force when a person takes
  the arm over; the trigger moves the jaws only once the opening it asks for has reached
  it.
- **Table clearance**: the height over the table plane that the gripper tip keeps outside
  a grasp: `rest.table_clearance_m` of the robot app's file, or `safety.table_clearance`
  of APAI's arm profile where that is higher. **Grasp region**: the column around an
  object in which a move of the agent may go below the table clearance, for a grasp; the
  robot app keeps the last one. **Low gripper tip**: a gripper tip below the table
  clearance or inside the agent's last grasp region.
- **Human checker**: the robot app's checker of the human limit set (`limits.human`).
  The **agent's checker** and the **recovery checker** are its checkers of
  `limits.agent` and `limits.recovery`: the first checks the agent's moves, the second a
  return to rest. The agent's checker is not the workstation checker, which runs in the
  agent program.

### The human path, end to end

```text
 WORKSTATION                                      THOR
 Quest headset, right controller
      |  CloudXR
      v
 Teleop Session Server                            human program  (role "human")
 (Isaac Capture, unedited; port 47600)              controller feed:
      |                                               teleop feed = a remote session
      +--- teleop connection: SAMPLES, READY ---->    script feed = a synthetic controller
                                                    retargeting graph: right controller
                                                      -> rotated into the robot's base axes
                                                      -> clutch (re-base)
                                                      -> rate limiter (hand speed limit)
                                                      = the target of the gripper tip
                                                    APAI's IK (no restarts, an iteration
                                                      limit), jump check, ahead checks
                                                    buttons: right B = STOP, then takeover;
                                                      right A = hand-back
                                                         |  HUMAN_SETPOINT, STOP,
                                                         |  OWNER_REQUEST "human", "handback"
                                                         v
                                                    the robot app's own socket (local API)
                                                         |  ^
                                                         v  |  WELCOME, ROBOT_STATE, RESULT
 robot link service  <---- robot connection ---->  robot app
      ^                VIEW_STATUS, STOP and the     control owner human, limits.human
      |                operator's commands -->       human target -> human drive:
 operator console      <-- ROBOT_STATE, EVENT,         sample age, view age, step refusal,
 live view                 RESULT, WELCOME             speed clamp, human checker,
                                                       contact watch, a hold at every end
                                                     jaw opening -> the jaws, once it has
                                                       reached the grip level
```

That is the rig. On one machine, in the recipes and in the tests, the human program
attaches to the robot link service in place of the robot app's own socket: what it sends
crosses the robot connection, the robot app handles it as what arrives at its own
socket, and the teleop connection stays on 127.0.0.1.

One pass of the human program:

1. The controller feed gives the newest data of the right controller and their sample age.
2. The retargeting graph gives the target of the gripper tip (after the clutch and the rate
   limiter), whether the clutch is engaged, and the buttons.
3. Right B that goes down sends a STOP. B held for `buttons.hold_s` (1 s) with the grip
   released asks for the takeover; A held the same way asks for the hand-back.
4. While the clutch is not engaged, the clutch's home is set to where the arm is: the FK of
   the measured pose of the newest ROBOT_STATE, moved 0.025 rad inside the joint limits
   (the margin APAI's IK keeps), and the rate limiter is reset. So every engage starts from
   where the arm is.
5. While the clutch is engaged, the IK turns the target of the gripper tip into joint
   values, seeded with the last joint values sent. A failed solve, a jump or a target too
   far ahead ends the engagement in the human program.
6. One HUMAN_SETPOINT goes out: with the joint target while engaged, without one otherwise.
   It goes out on every pass while the person owns the arm and from a press of right B that
   counts (below) until its request is answered; otherwise `rates.idle_hz` (20) times per
   second, as a status. Each is under 512 bytes.
7. On every fresh pass while the person owns the arm, the grip squeezed or not, that
   HUMAN_SETPOINT also carries the jaw opening the trigger asks for: 1.0 minus the
   trigger's value, with a dead zone at each end. A trigger at or below 0.05 asks for
   fully open jaws, one at or above 0.95 for closed jaws. On any other pass it carries
   no jaw opening.

On the Thor, once per `executor.period_s` (0.02 s) while a human drive runs: the newest
human target must quote the current owner and stop numbers, be engaged, be fresh controller
data and hold one finite value per joint; the live view must be fresh where the file
requires it; the target must lie within `limits.human.max_step_rad` of the last setpoint
sent, else the drive ends (a far target is refused, not clipped). Then the Thor moves one
step toward the target, each joint at most 0.98 of `limits.human.max_joint_speed` (the
speed is clamped, and `health.clamped` counts it). While the measured gripper tip is low,
a target or a step that would lower it ends the drive there (the rule of the lift under
[A takeover, step by step](#a-takeover-step-by-step)). Otherwise the human checker checks
the step and it goes to the motors. A setpoint that goes out late is not made up for: the
step keeps its size.

**No obstacle check for a person**, as the design says: the human program does not read the
obstacle grid and sends no PERMIT. What stands in for a permit is a fresh engaged human
target and a fresh live view, both checked by the Thor before every setpoint. The person's
eyes on the arm and on the live view are the obstacle check, which is why the live view is
required. What the human checker enforces without a camera: finite values, the human joint
box and the speed limit of `limits.human`; with APAI's checker also the workspace box and
the table of APAI's profile and its margin from the model's joint limits. The agent's
last grasp region binds a person in one way only: inside it the gripper tip is not
lowered (the rule of the lift). The agent's **contact spots** do not bind a person. A
contact spot is the place where a move of the agent or a return to rest ended at a
contact; the agent's checker and the recovery checker keep it and refuse a move that
enters it from outside (with APAI: within 5 cm of the gripper tip's place at the
contact).

The way to the robot app is part of a sample age and of a view age. Each is what the
sender measured as it sent, plus the time since then on the Thor's monotonic clock: the
sender reads that clock off its newest ROBOT_STATE and says when it sent (`sent_mono_s`).
The reading is never later than the true moment, so an age is never counted as less
than it is. A sample age is counted as more by the ROBOT_STATE's way to the human
program. A view age is not: the view program's own age is short by that same way. For a sample
age the way is the robot app's own socket on the rig and the robot connection on one
machine; for a view age it is the robot connection. A HUMAN_SETPOINT that does not say
when it was sent is counted from the ROBOT_STATE it was computed on; a VIEW_STATUS that
does not say it gives no view age (`the live view is not fresh: the VIEW_STATUS does not
say when it was sent; ...`). The round trip (`health.human.round_trip_s`) is a
ROBOT_STATE's way to the human program, one pass of that program and the way back. It
is shown and nothing is enforced on it. How long it is on the rig is not known.

### Where the human path departs from the design

On the rig the build stands where the design does (its sections 4, 5, 12 and 14). The
work that turns the hand's motion into joint targets runs on the Thor. The Thor dials the
workstation's Teleop Session Server and receives SAMPLES over a remote session. The robot
connection stays separate from the teleop connection, and Isaac Capture's teleop code is
used unedited.

**What differs is the process.** The design runs a teleop thread inside the robot app.
Here the remote session, the retargeting graph and the IK run in the human program, a
process of its own beside the robot app, which hands the robot app its joint targets in
HUMAN_SETPOINT link messages at the robot app's own socket. The robot app holds the
motors on its own threads. Its socket is read and written on threads that only hand link
messages over, as the robot connection's reader does, and no IK and no retargeting run
inside the robot app's process. The same program, attached to the robot link service,
serves the one-machine recipes and the tests.

| | A process beside the robot app (this folder) | A thread inside the robot app (the design) |
|---|---|---|
| Strongest case | The process that holds the motors gets no `TeleopSession`, no retargeting graph and no IK loop beside its 50 Hz motor loop: when that code fails or stalls, it is the human program that ends or stalls, and the robot app holds the arm on the missing HUMAN_SETPOINTs. One program serves the rig, the one-machine recipes and the tests. The Thor's own checks (sample age, live view, step limit, speed clamp, human checker, contact watch) are the same in both | It is what the design says. The robot app sees the buttons itself, so right B's STOP and the 1 s hold are not another program's word. One program fewer to start on the Thor |
| Cost | (1) The robot app does not see the buttons: the 1 s hold of right B and the clutch are the human program's word; the robot app checks the released grip, the ages and the numbers. (2) The sample age starts from an age the human program reports, with the moment it sent as it reads the robot app's clock off a ROBOT_STATE; the robot app adds the time since then on its own clock. (3) A link message from the socket, right B's STOP among them, waits for the next pass of the robot app's dispatcher: at most one step period, 10 ms. (4) One more program to start on the Thor, after the robot app; without it there is no right B and no takeover, and it ends when the robot app ends. (5) The hand speed limit is a setting of the human program, not of the Thor's human limit set. (6) One more stage at 50 per second and one local hop between the controller data and the motors; the delay was not measured | `TeleopSession`, the retargeting graph and the IK inside the process that holds the motors. The design's bench item "the teleop thread beside the 50 Hz motor loop" would have to pass before any human motion |

**Moving into the robot app later** costs one host thread there (an estimated 150 to 250
lines) that runs the same controller feed, graph, solver and logic and calls
`RobotApp.offer_human_target`, plus a local entry for the takeover and for the hand-back of
about 15 lines each. No link message changes; HUMAN_SETPOINT can stay for the synthetic
controller. The Thor's side is built for that: it takes a human target and no link message,
and the graph, the solver, the logic and the feeds (`human/retarget.py`, `ik.py`,
`logic.py`, `feed.py`) import nothing of the workstation's programs.

Smaller departures: the design clamps a far target, here it ends the engagement or the
drive; the design's wait for READY has no time limit, here the attempts are counted
(`teleop.attempts`); the design has typed fallbacks for `rest`, `open` and `reset` only,
here `takeover` and `handback` exist, switched off; the design refuses a return to rest
from a contact hold, here it is taken while a takeover is open; a takeover is also
refused while the human checker refuses the measured pose for anything but a low
gripper tip, which the design does not ask; and the design leaves the trigger out of its
first build (its section 20), where this folder builds its rule for the trigger and the
jaws (step 5 of its section 12).

### A takeover, step by step

| # | What happens | Link messages | Typed or pressed | What the operator sees |
|---|---|---|---|---|
| 1 | The agent fails while it owns the arm | OWNER_REQUEST "help_needed" from the agent program, or from the service with `agent program gone`; EVENT `help_needed` | - | Console: the help block with the reason, the jaws, the time left and the takeover line, `takeover: possible now ...` or `takeover: not possible: <why>`. The view's page shows the EVENT |
| 2 | The assistance wait counts down | ROBOT_STATE `assistance_wait_s`; EVENTs with bells | - | Console: `wait N s`, bells, `ARM WILL MOVE TO REST`; where the timed return is off, as on the rig, `ARM WILL KEEP HOLDING` |
| 3 | The person gets ready | HUMAN_SETPOINT, as a status | Puts the headset on, picks up the right controller, stands behind the arm, recentres the headset view with the grip released | Status line: `headset ready`, `controller 0.0x s` and the human program's note, `note: hold right B for 1 s with the grip released to take over` |
| 4 | Right B is pressed | STOP with the reason `right B` and the sender `controller` | **Right B** | Console: `[stop] STOP from controller: right B; the arm holds, stop number N; the assistance wait is cancelled`. Nobody owns the arm |
| 5 | B stays down for `buttons.hold_s` (1.0 s) with the grip released, every pass fresh | OWNER_REQUEST "human" with the reason `right B held 1.0 s`, quoting the ROBOT_STATE that shows the new stop number | **Hold right B 1 s**. Nothing is typed | Accepted: EVENT `[owner] TAKEOVER: a person owns the arm, owner number N; release the grip, then squeeze it to move` and two bells; with a low gripper tip the line goes on `; the gripper tip is low: only a lift is taken until it is clear`. No jaw command is sent: the robot app notes the grip level. Refused: `[refused] owner request 'human' refused: <reason>` and three bells; the note then reads `takeover refused: <reason>` |
| 5b | The typed fallback, only with `operator.typed_takeover: true` in the service's file | OWNER_REQUEST "human" with the reason `typed at the operator console` | `takeover`, then `yes`, at the console. The control owner must be nobody: type `stop` first if the agent owns the arm | The same |
| 6 | The grip is squeezed. The Thor starts a human drive | HUMAN_SETPOINT, engaged, with the joint target | **Squeeze the grip** | Status line: `moving N/0` and `human driving`; the page writes `HUMAN DRIVING`. The arm follows the hand, starting from the measured pose. Where a joint lay on its stop, as at rest, the arm first lifts that joint 0.025 rad off the stop |
| 7 | The grip is released | HUMAN_SETPOINT, not engaged, grip released | **Release the grip** | The arm holds. The grip does not act on the jaws: they follow the trigger with the grip squeezed or not. Status line: `hold`, and the note `squeeze the grip to move` |
| 7b | The trigger asks for a jaw opening | HUMAN_SETPOINT with `jaw_opening`, engaged or not | **Trigger** | Nothing while the opening it asks for has not reached the grip level: a held object stays held. From then on the jaws follow the trigger (the rule on the jaws below). ROBOT_STATE's `health.human.jaws` says which of the two holds, and why the trigger moves nothing |
| 8 | Any loss while driving ([below](#what-ends-a-human-drive-and-what-follows)) | - | - | EVENT `[human] human drive ended: <why>; the arm holds; release the grip and squeeze again` and one bell |
| 9 | The person is done and asks for the return to rest | OWNER_REQUEST "rest", confirmed | Grip released. `rest`, then `path is clear`, at the console or at the Thor terminal | While the arm returns, recovery is the control owner and the note reads `the arm returns to rest; right B is a STOP`. Then `rest: RESULT completed`; motion `rest_hold`; the person is the control owner again; the note reads `at rest: hold right A for 1 s to give the arm back`, or, while ROBOT_STATE shows the jaws as `loaded`, `at rest: jaws not proven empty: at the console type stop, open, yes; then hold right A for 1 s` |
| 10 | The hand-back | OWNER_REQUEST "handback" with the reason `right A held 1.0 s` | **Hold right A 1 s** with the grip released. Fallback: `handback`, then `yes`, at the console, with the same service setting | Accepted: EVENT `[owner] takeover ended: the arm is given back; nobody owns it` and two bells. Refused: `[refused] owner request 'handback' refused: <reason>` and three bells |
| 11 | The agent may get a new task | OWNER_REQUEST "agent" | The task and `yes` at the agent terminal | The task runs. No program was restarted |

Rules that hold through all steps:

- **Right B is a STOP whenever the buttons are known**: on every pass on which the feed
  delivered controller data with a known sample age, also while the controller's pose is
  not valid. It ends an agent task, a return to rest and a human drive alike. One STOP goes
  out per stretch that right B is down. A stretch ends when B is seen up, and when two
  passes with known buttons are more than 1 s apart, because B may have been released and
  pressed again unseen.
- **A takeover needs more than the STOP: a press that counts.** The button was seen up on a
  fresh pass since the last pass that was not fresh, and it is then held for
  `buttons.hold_s` on fresh passes with the grip released. One pass without fresh data, or
  with a pose that is not valid, ends the hold. A hold does not count across a stall
  either: when more than 1 s passed since the last pass that knew the buttons, the holds
  of right B and of right A start anew, and the button has to be seen up before a hold
  counts. One pass with the grip squeezed spoils the press: held on, it asks for
  nothing. So right B that is held as the headset wakes up, as
  controller data return or after more than 1 s without controller data stops the arm once
  and takes nothing over: release it and hold it again for 1 s. The note `release right B,
  then hold it for 1 s with the grip released to take over` says so, also for a spoiled
  press. A button that stays down after its request went out asks for nothing more until
  it was seen up. Right A follows the same rule; its note is `release right A, then hold
  it for 1 s with the grip released to give the arm back`. A press shorter than one pass
  (20 ms) can be missed.
- **A takeover needs no help request.** Right B held 1 s takes the arm from control owner
  nobody at any time, also at rest. The request goes out only once a ROBOT_STATE shows the
  STOP of that press.
- **What the robot app checks before it accepts a takeover**, in this order, each with its
  own refusal text: `human.enabled`; as for every command, the session id, a latched
  fault-stop, the stop number, the control owner (nobody) and the owner number the request
  quotes; the motors are on; no work runs or waits and no jaw command runs or waits; fresh
  controller data (`no fresh controller data: ...`); the source of those data, which on a
  driver that is not simulated must be `teleop`, the headset (`a synthetic controller may
  drive only a simulated arm; ...`); a fresh live view where the file requires it (`the
  live view is not fresh: ...; open the live view page in a visible browser tab`); a full
  joint read; the
  human checker accepts the measured pose (`the arm is outside the human limits: <reason>;
  type rest`), where a gripper tip that is only low is no reason to refuse (the rule of
  the lift below). With the rig file that refusal also comes in the outer 0.02 rad of the
  agent's working box, where the typed `rest` is taken (see
  [The rig files](#the-rig-files)). A STOP or a change of control owner that came
  meanwhile wins. The view is
  not live for 3 s after its start and after a step of the Thor's wall clock (see
  [The live view](#the-live-view)): a takeover is refused meanwhile.
- **A refused takeover after a STOP that is not confirmed.** A STOP sent between a new
  robot connection and its first ROBOT_STATE keeps HUMAN_SETPOINTs at the robot link
  service until a later STOP is counted. If a takeover is refused with `no fresh controller
  data` while the console's status line says `a STOP is not confirmed yet`, press right B
  again.
- **Takeover open.** From an accepted takeover until an accepted hand-back the robot app
  refuses a new task of the agent with `takeover not finished: give the arm back with
  right A at rest`, also when a STOP has made nobody the control owner in between. The agent
  program refuses such a task before its question, with the same text. A fault-stop and
  its reset leave the takeover open. A new robot app session has no takeover open.
- **The hand-back** is taken from control owner human or nobody while a takeover is open.
  It needs: no fault-stop; no work and no jaw command running or waiting; the jaws proven
  empty; two full joint reads `rest.read_gap_s` apart within `rest.rest_tol_rad` of rest
  and of each other. It moves nothing and cuts nothing. Its refusals: `no takeover is
  open`; `control owner is <owner>`; the reason why the arm is not proven at rest, then
  `; type rest first`; the reason why the jaws are not proven empty, then `; type stop,
  then open, then yes, then hold right A again`. A STOP or a change of control owner while
  the joints were read refuses it too, and the takeover stays open. A hand-back, taken or
  refused, sends the motors nothing.
- **A hand-back with jaws that are not referenced.** The refusal reads `the jaws are not
  proven empty (unknown): the jaws are not referenced; type stop, then open, then yes,
  then hold right A again`, and that remedy does not work there: `open` is refused too
  (`the jaws are not referenced`). The jaws have to be referenced. Every shipped robot
  app file does that at the robot app's start, with the arm measured at rest
  (`gripper.reference_at_start: true`); the typed `reference` is switched off (see
  [What the jaws do](#what-the-jaws-do)).
- **A new squeeze.** The arm follows the hand only after a squeeze that began while
  everything allowed motion. A grip that is already squeezed when the takeover is granted,
  when a return to rest gives the arm back to the person, when controller data come back,
  when the robot connection comes back, when other work on the Thor that refused the
  squeeze has ended (a hand-back, a jaw reference, a return to rest that was refused), or
  after any drive that did not end by a released grip, moves nothing until it was
  released and squeezed again. Isaac Capture's clutch does
  not give this by itself, so the rule is built twice: in the human program and on the
  Thor (the needs-release flag). On the Thor only a HUMAN_SETPOINT that shows the grip
  released, with fresh controller data and the current numbers, clears the flag.
- **While the grip is squeezed** a typed `rest`, `release` or shutdown is refused with
  `busy: human driving is running`. Release the grip first. A STOP always works.
- **Jaws: the trigger drives them once it has reached the grip level.** At the change of
  control owner no jaw command is sent and no jaw is read: a held object stays held. The
  robot app notes the grip level: fully open when the jaws are proven empty and open,
  else the opening of the last jaw target, else the remembered jaw reading. With
  `human.trigger_jaws: true` in the robot app's file, the shipped value, the jaws then
  take the jaw opening that a HUMAN_SETPOINT carries. The human program sends 1.0 minus
  the trigger's value there on every fresh pass while the person owns the arm; a
  trigger at or below 0.05 asks for fully open, one at or above 0.95 for closed. The
  robot app's rules:
  - **The trigger is ignored until it has reached the grip level**: the opening it asks
    for is within `human.jaw_match_tol` (0.05 of the jaw travel) of the grip level, or
    two HUMAN_SETPOINTs that follow each other ask for openings on either side of it.
    A trigger that is already past the grip level at the first HUMAN_SETPOINT after the
    takeover moves nothing until it comes back to it. So a takeover neither drops a
    held object nor squeezes it.
  - **From then on the jaws follow the trigger**, the grip squeezed or not, one jaw
    command at a time and always toward the newest opening. A new jaw command goes out
    only when the opening differs from the one the jaws settled for by more than
    `gripper.reach_tol` (0.03), or asks for fully open.
  - **No tighter than the agent** until the jaws were measured fully open once since
    the takeover: an opening below the grip level is held at the grip level meanwhile
    (`no closing beyond the grip level until the jaws were measured fully open`).
  - **A close ends as the agent's does**: when the jaws stall on an object the driver
    leaves them at the light holding stiffness. **An opening** goes through the
    driver's guarded opening, one step of `gripper.open_step` (0.05) per jaw command.
  - **A close needs the live view fresh** where the file requires it; an opening does
    not (`no closing: the live view is not fresh: ...`).
  - **What ends the jaw motion**: a STOP or a fault-stop; controller data that are away
    or older than `human.sample_age_limit_s`; a HUMAN_SETPOINT that does not quote the
    numbers of a person who owns the arm; a jaw opening that is no number from 0 to 1
    (`refused: jaw_opening must be between 0 and 1, got ...`); a human drive that
    anything but a released grip ended; a typed `rest`, a hand-back, a `release` or a
    shutdown that waits or runs; a lost robot connection; the end of the takeover. Each
    ends it as a STOP ends a jaw command: the jaws are held lightly where they are,
    never opened, and the trigger has to reach the grip level again, which is then the
    opening in force at that moment. A HUMAN_SETPOINT without a jaw opening sends the
    jaws nothing, lets a jaw command under way run to its end, and the trigger has to
    reach the grip level again too.
  - **Where the grip level is not known** (the jaws are not referenced, or they have no
    target and no reading) the trigger moves nothing until the control owner changes
    (`the grip level is not known: ...`).
  - **`health.human.jaws` in ROBOT_STATE** shows all of this while a takeover is open:
    `grip_level`, `matched`, `opened`, `asked` and `refusal`, the reason the trigger
    moves nothing or the jaws are held wider than it asks, such as `the trigger has not
    reached the grip level`.
  - **With `human.trigger_jaws: false`** the trigger does nothing and no jaw command is
    sent under a person.

  The console's way to let go stays: `stop`, then `open` and `yes` (it needs control
  owner nobody), then right B again or `rest`. Wait for the RESULT of the opening
  before right B: right B's STOP cuts an opening short. The hand-back still needs the
  jaws proven empty. Jaws that the person closed on an object, or opened only part of
  the way, are not: release the trigger all the way, so that the jaws are measured
  fully open. A takeover and a hand-back are refused with `busy: a jaw command is
  running or waiting` while a jaw command runs or waits.
- **A contact while a takeover is open.** The arm holds at the measured pose. A new squeeze
  moves it away. Without a headset: clear the path by hand, then type `rest` and `path is
  clear`. While a takeover is open the return to rest is taken from a contact hold; it
  keeps its contact watch and ends at the next contact. Without an open takeover that
  `rest` is refused (`the arm is holding at a contact`).
- **`path is clear` during a takeover clears the recovery checker's contact spots.** After
  a move of the agent ended at a contact, the recovery checker refuses a return to rest
  that enters that spot from outside. Only the agent can clear its spots, and the agent
  gets no task while a takeover is open. So a return to rest that was confirmed with
  `path is clear` while a takeover is open first clears the contact spots of the recovery
  checker: a typed `rest`, at the console or at the Thor terminal, and a shutdown
  confirmed that way. The words are the person's word for those spots too. The agent's
  checker keeps its spots, the return keeps its contact watch, and an unconfirmed `rest`
  clears nothing. Without an open takeover nothing is cleared. No setting switches this:
  it can be reached only where `human.enabled` is true.
- **A shutdown is taken during a takeover.** `shutdown` at the console and `quit` at the
  Thor terminal are taken while a person owns the arm with the grip released, and while a
  takeover is open and nobody owns the arm: the arm returns to rest, torque is released
  there, at measured rest with the jaws proven empty, and the robot app ends. Away from
  rest the shutdown needs `path is clear`. At rest it needs none: the console asks for
  `yes` and the Thor terminal asks nothing. The next session of the robot app has no
  takeover open. No setting switches this either; a takeover exists only where
  `human.enabled` is true.
- **A low gripper tip: a person lifts the arm out.** With APAI's checkers, an arm whose
  gripper tip is low, below the table clearance or inside the agent's last grasp
  region, where an agent's grasp can leave it, can be taken over. The human checker
  then checks the measured pose without its table clearance for the gripper tip, and
  the takeover's line ends `; the gripper tip is low: only a lift is taken until it is
  clear`.
  - **The lift.** While the measured gripper tip is low, a squeeze is followed only
    while neither the target nor the step toward it puts the tip more than 1 mm lower
    than it is measured. Otherwise the drive ends in a hold and nothing is sent:
    `human drive ended: lift the gripper: <why the tip is low>, and the target would
    lower it to z=... m; the arm holds; release the grip and squeeze again`, or `...,
    and the step toward the target would lower it to z=... m; ...`. `<why the tip is
    low>` is `the gripper tip is at z=... m, below the table clearance ... m` or `the
    gripper tip is inside the last grasp region`. The tip is held against where it is
    measured, not where the drive began: a hand that pauses and sinks by more than 1 mm
    ends the drive.
  - **Below APAI's own table clearance every step must rise.** That is the escape rule
    of APAI's human checker: under its table clearance it refuses every step that does
    not rise, a slide at one height included (`TCP z=... below table clearance ...
    outside the grasp exemption zone`). The drive ends in a hold; lift straight up
    first.
  - **Above the clearance, inside the last grasp region,** a slide at one height is
    followed, and it is how a region without a ceiling is left: such a region is a
    column with no top, in which the tip is never lowered at any height.
  - **The typed `rest` stays refused there** (`the gripper tip is at z=... m, below the
    table clearance ... m`, or `the gripper tip is inside the last grasp region`): its
    straight joint path could drag the gripper over the table. A new task of the agent
    stays refused there too. The way out: take over, squeeze and lift until the tip is
    above the clearance and outside the region, release the grip, type `rest`, then
    `path is clear`.
  - **A takeover there is still refused for any other complaint of the human checker**
    (`the arm is outside the human limits: ...; type rest`): a tip under the workspace
    box, a link at the table, the human joint box.
  - **The simulated arm's checkers know no table**: none of this applies with
    `--arm sim`.

#### A takeover that cannot be ended with right A

The controller is dead, or right A does not arrive. Bring the arm to rest: `rest`, then
`path is clear`. Then either:

- restart the robot link service with `operator.typed_takeover: true` in its file (the arm
  holds at rest with its motors on meanwhile, the robot app dials the new service and the
  takeover stays open), attach the console again (`attach`) and type `handback`, then
  `yes`; or
- type `shutdown`, then `yes`, at rest and start the robot app again: a new session has no
  takeover open.

Never restart the robot app with the arm raised.

**That typed `rest` can be refused** by the return to rest's own checks. A refusal changes
nothing: the arm holds with its motors on and the takeover stays open. The four refusals:

| Refusal | Why | What the person does |
|---|---|---|
| `the jaws are not proven empty (...)` | The jaws may hold something, or their readings do not prove them empty | Type `stop`, then `open`, then `yes`; wait for the opening's RESULT; then `rest`. It needs no controller. With jaws that are not referenced (`the jaws are not proven empty (unknown): the jaws are not referenced`) `open` is refused too (`the jaws are not referenced`): see the rule on a hand-back with such jaws above |
| `the gripper tip is inside the last grasp region`, or `the gripper tip is at z=... m, below the table clearance ... m` | The return starts only with the gripper tip clear of the agent's last grasp region and above `rest.table_clearance_m` | Take over, squeeze and lift the gripper clear, release, then `rest`. Below the table clearance and inside the region the takeover is taken and only a lift is followed (the rule above) |
| `joint index N at X rad is outside the tested recovery region ...` | The start pose is outside `rest.region_*`. A person cannot leave the arm outside `rest.region_*`: with `human.enabled` the robot app refuses a file whose human joint box is not inside it. In the rig file the box lies 0.02 rad inside the region, except at the lower bound of the second and the third joint, so a joint driven to the edge of the box that reads a few thousandths of a rad outside the box is still inside the region, and this refusal does not come. In a file whose box equals its region such a reading on a real arm lies outside the region | With the rig file: nothing, the typed `rest` is not refused for this. With a box that equals the region: squeeze and move that joint back, release, then `rest`. That needs the person as control owner. After a STOP or a fault-stop at such a pose nobody owns the arm and a new takeover is refused too (`the arm is outside the human limits: ...; type rest`): no command moves the arm then (see [Known gaps](#known-gaps)) |
| `the path to rest was refused: setpoint N: <APAI's reason>` | The return moves every joint straight to its rest value, and the recovery checker checks that whole path first. From some poses it passes under the table clearance, leaves the workspace box, brings a link to the table or tips the camera on the arm's wrist further than APAI's profile allows (its **wrist camera rule**). Such a pose has a compound turn of the wrist or the base | Squeeze and move the arm back the way it came, release, then type `rest` again |

Three of the four need a working controller, and a start pose outside the region also
needs the person to be the control owner still. With the rig file a person's drive ends
at such a start pose only if a joint reads more than 0.02 rad outside the human joint
box: the box lies 0.02 rad inside the region, except where the second and the third
joint lie on their stops. After a STOP at an edge of that box a new takeover is refused
and the typed `rest` is not refused for its start pose. Without a working controller, no
typed command moves an arm whose return is refused for its tip, its start pose or its
path.
`rest` is refused with that text, and a shutdown that has to move the arm with
`the return to rest ended refused: ...` in front of it. A hand-back and a `release` are
refused with `the arm is not at rest: ...`, and a task of the agent with
`takeover not finished: ...`. The arm holds with its motors on. On a driver that is not
simulated only the headset path counts as a controller.

### What ends a human drive, and what follows

Every end is a hold with the motors on.

| Cause | Detected by | The arm | Control owner after | What the operator sees |
|---|---|---|---|---|
| The grip is released | HUMAN_SETPOINT: not engaged, grip released | Hold | human | Status line `hold`; no EVENT |
| Right B | STOP | Hold | nobody, takeover open | `[stop] STOP from controller: right B; ...` |
| `stop` at the console, `stop` or Ctrl+C at the Thor terminal | STOP | Hold | nobody, takeover open | `[stop] ...` |
| Headset data lost: the Teleop Session Server gone, the teleop connection lost (on the rig it crosses the wired link between the machines), SAMPLES stale, the controller not active, the human program gone | The human program sends a HUMAN_SETPOINT that is not engaged, with its stage; or the Thor's own limit on the sample age | Hold, within `human.sample_age_limit_s` | human | `human drive ended: controller data lost` (the human program's word), `human drive ended: controller data stale (...)` or `human drive ended: controller data were away` (the Thor's, below); status line `headset <stage>` or `no human program` |
| Controller data were away between two HUMAN_SETPOINTs: when one arrives, the one before it is no fresh controller data | The robot app, as the HUMAN_SETPOINT arrives | Hold | human | `human drive ended: controller data were away` |
| An engaged HUMAN_SETPOINT that the robot app refuses while a human drive waits for its start or runs, for example for old owner or stop numbers or for its source | The robot app, as the HUMAN_SETPOINT arrives | Hold | as it was | `human drive ended: an engaged human target was refused: <the refusal>` |
| A HUMAN_SETPOINT that quotes other owner or stop numbers than the drive was admitted with; an engaged one is refused as in the row above | The Thor, before every setpoint | Hold | as it was | `human drive ended: the human target quotes owner number N and stop number M, the drive was admitted at ...` |
| The headset's stream to the Teleop Session Server stalls while the server keeps its tick rate | Nobody at once: the tick age stays fresh. By Isaac Capture's notes the controller is reported not active about 1 s later; not run | The hand's target stands still, so the arm stops within the two ahead limits. A grip release and right B are not seen until the stream is back; right B that went down meanwhile is a STOP then | human | Status line `headset no_controller` after about 1 s. This is why the abort of part F of the checklist is the Thor terminal |
| The live view is not live or too old | The Thor, before every setpoint | Hold | human | `human drive ended: the live view is not fresh (...)` |
| The robot connection is lost | The robot app | Hold | human | Console: `ROBOT CONNECTION LOST`. Thor terminal: `[connection] robot connection lost: the arm holds; the person keeps the arm: squeeze again when the connection is back; rest can still be typed at the Thor terminal` |
| A target more than `limits.human.max_step_rad` from the last setpoint sent | The Thor | Hold | human | `human drive ended: the target is X rad from the arm on joint index i, the limit is Y rad` |
| While the measured gripper tip is low: a target, or the step toward it, that would lower the tip by more than 1 mm | The Thor, before the human checker | Hold; that step is not sent | human | `human drive ended: lift the gripper: ...` |
| The human checker refuses a setpoint (joint box, workspace, table; below APAI's table clearance also a step that does not rise) | The Thor | Hold | human | `human drive ended: <the checker's reason>` |
| No IK solution `ik.max_failures` passes in a row, an IK jump, a target too far ahead, a hand too fast for the rate limiter | The human program, which then sends a HUMAN_SETPOINT that is not engaged | Hold | human | `human drive ended: <note>`, with the note `no IK solution (...)`, `IK jump of X rad on joint index i, the limit is 0.05`, `the hand is X cm ahead of the arm's target`, `the solution is X rad ahead of the arm` or `the hand moved too fast for the rate limiter` |
| The newest ROBOT_STATE is older than `state.max_age_s` at the human program | The human program | Hold | human | `human drive ended: the robot state is X s old` |
| A setpoint is more than `executor.large_delay_s` late | The Thor | Hold | human | `human drive ended: setpoint N is X s late, over the limit of Y s` |
| Contact | The contact watch | Hold at the measured pose; not latched. When the pose for that hold cannot be read, in two tries, nothing is sent and the last setpoint stands | human | `human drive ended: contact ...`, with `; the hold at the measured pose failed: ...; the last setpoint stands` when it did; status line `contact hold`. A new squeeze moves the arm away |
| Motor fault, lost joint feedback, also during the hold after a contact | The Thor | Fault-stop: a hold, latched | nobody, takeover open | `FAULT-STOP: ...`; `reset` at the console |
| The agent program crashes | - | Nothing changes | human | The console's status line no longer counts an agent client |
| The agent program is ended with Ctrl+C or SIGTERM | STOP: the agent program sends one as it ends, with the sender `agent_terminal` | Hold | nobody, takeover open | `[stop] STOP from agent_terminal: ...` |

**The EVENT of an ended drive** is `[human] human drive ended: <reason>; the arm holds;
release the grip and squeeze again`, with one bell. Its data hold the drive's `status`,
the `reason` and `sent`, the number of setpoints the drive sent; `health.human.ended`
holds the reason too.

- A drive that a released grip ended has no EVENT.
- A drive that a STOP, a change of control owner or a lost robot connection ended has
  none of this kind: each of those has an EVENT of its own.
- A drive that still waited for its start is told the same way, with `sent` 0, when the
  robot app drops it because controller data were away or because an engaged
  HUMAN_SETPOINT was refused.
- No single text says "the controller was lost while driving": by what arrives first on
  the Thor the reason is `controller data lost`, `controller data stale (...)` or
  `controller data were away`.
- After every end but a released grip, the grip has to be released before a squeeze moves
  the arm again.

A typed `rest` during a takeover can be refused by the return to rest's own checks: for
the jaws, for the gripper tip, for the start pose or for the path to rest. The four
refusals and what the person does about each are under
[A takeover that cannot be ended with right A](#a-takeover-that-cannot-be-ended-with-right-a).

### Torque is never cut for a person

[The torque rule](#the-rule-torque-is-never-cut-away-from-rest) holds for the human path
without a new case: no takeover, no hand-back, no end of a human drive and no loss of the
headset, the human program, the live view or the robot connection cuts torque. While a
human drive runs the robot app refuses a `release` with `busy: human driving is running`,
and the guarded release itself sees the drive as a running move and refuses it too. With
the grip released, a `release` and a shutdown are the operator's commands as at any other
time, also while a person owns the arm: torque is cut only at measured rest with the jaws
proven empty. The hand-back reads the jaws and the joints and sends nothing. The human
program and the files under `human/` have no call that could cut torque: the human program
sends HUMAN_SETPOINT, STOP and its two OWNER_REQUESTs and nothing else, and neither the
robot link service nor the robot app's own socket lets its role send anything else. The
trigger moves the jaws through jaw commands only, and every end of a jaw motion is a
light hold.

### The settings that ship off

Each of these settings ships in its safe position, with one exception: `human.trigger_jaws`
ships on, and acts only where `human.enabled` is true. The table says what the shipped
value means, where it is switched and what a change waits for.

| Setting and shipped value | What the shipped value means | What it waits for |
|---|---|---|
| `human.enabled: false` in `configs/robot_app.yaml` and `configs/robot_app_rebot.yaml` | A takeover is refused and no human target moves the arm. `true` only in the files of the one-machine recipes and of part F-A (`robot_app_mock.yaml`, `robot_app_demo2.yaml`, `robot_app_human_check.yaml`) | It is switched on in the rig file in the session, before step 15, once part F-A of the checklist has passed. Code that never saw a headset is not reachable on the arm before that |
| `human.require_live_view: true` | A takeover and a human drive need a fresh live view. `false` with `human.enabled: true` is taken only with a simulated driver: with another the robot app does not start, and `--driver real` says so before its three questions | Nothing: `false` is for demo 2 on one machine without a camera |
| `human.view_age_limit_s: 0.5` | The oldest live view a person may drive on. A first value | Step F0: the highest measured view age plus 0.2 s, at most 1.0 s |
| `human.trigger_jaws: true` and `human.jaw_match_tol: 0.05` in every robot app file | Under a person the trigger drives the jaws once the opening it asks for has come within `human.jaw_match_tol`, a share of the jaw travel between 0 and 1, of the grip level. With `human.trigger_jaws: false` the trigger does nothing and no jaw command is sent under a person. Neither acts where `human.enabled` is false | The person's trigger has not moved the real jaws. Before step 15 the owner decides whether the first headset session on the arm runs with it: `human.trigger_jaws: false` in `configs/robot_app_rebot.yaml` is the session without it |
| `operator.typed_takeover: false` in both service files | `takeover` and `handback` typed at the console are refused by the robot link service | Step F6: if right A or right B does not arrive, or a controller that dies during a takeover. Switching it on needs a restart of the robot link service |
| `limits.human.max_joint_speed: 0.1` rad/s in the rig file; `retarget.max_linear_velocity: 0.01` m/s and `retarget.max_angular_velocity: 0.03` rad/s in `configs/human_rebot.yaml` | The gripper tip moves about 1 cm per second under a person | A bench test on this arm that supports a higher speed |
| `ik.max_ahead_m: 0.05`, `ik.max_ahead_rad: 0.08` in both human files | A hand's target more than 5 cm ahead of the speed-limited target ends the engagement; each end costs a release and a new squeeze | Nothing. A larger value lets the arm go on until it has caught up with the hand |
| `limits.human.max_step_rad: 0.1` | A human target farther than that from the last setpoint sent is refused, not clipped | Nothing. A larger value takes a farther target; no value clips one |
| `teleop.attempts: 720` in `configs/human.yaml`, `8640` in `configs/human_rebot.yaml` | The human program gives up on the Teleop Session Server after that many failed attempts in a row, 5 s apart: about an hour and a half, or a session day | Nothing. A larger count keeps the program trying for longer |
| `experiment.grid: bridge` in `configs/agent.yaml` and `configs/agent_rebot.yaml` | The obstacle grid comes from a grid bridge. `clear` only in `configs/agent_mock.yaml`, with the minimal stack and a simulated driver | Nothing: the clear grid is for the one-machine recipe |
| `jaws.allow_reference: false` in every robot app file | The typed `reference` is refused (see [What the jaws do](#what-the-jaws-do)) | A decision on that jaw rule, which the design does not have |
| `camera.allow_invented_times: false` in `configs/agent.yaml` and `configs/agent_rebot.yaml` | The agent program does not start on a mock or a recorded camera. `true` only in `configs/agent_mock.yaml` | Nothing: `true` is taken only against a simulated driver |
| `rest.auto_return: false` and `task.timed_return: false` in the rig files | The arm keeps holding when the assistance wait runs out | A later session: the timed return is not part of the first run on the rig |

Six behaviours have no setting of their own:

1. A takeover is refused while the human checker refuses the measured pose for anything
   but a low gripper tip.
2. The rig file's human joint box lies 0.02 rad inside `rest.region_*`, except at the
   lower bound of the second and the third joint (`limits.human.joint_min` and
   `joint_max` in `configs/robot_app_rebot.yaml`; see [The rig files](#the-rig-files)).
   With `human.enabled` a box wider than the region is refused; no rule asks for the
   margin.
3. While a takeover is open a typed `rest` with `path is clear` is taken from a contact
   hold. The alternative is the design's rule: after a contact under a person with the
   headset path lost, the only way out is the 48 V switch with the arm held by hand.
4. While a takeover is open a return to rest confirmed with `path is clear` clears the
   recovery checker's contact spots. The alternative is to leave the refusal: the return
   then starts only after a squeeze has brought the gripper tip back to within 5 cm of
   the agent's contact spot, which needs a working controller.
5. A shutdown is taken while a person owns the arm with the grip released and while a
   takeover is open; it returns the arm to rest and releases torque there.
6. With a low gripper tip a takeover is taken, and while the measured tip is low only a
   lift is followed. The alternative is to refuse that takeover: an arm that an agent
   left low in a grasp then takes no command that moves it.

The last four can be reached only where `human.enabled` is true, so the switch for human
driving is the switch for them too.

### What was cut from the human path

Each of these is in the design or in the plan of this phase, and is not built.

| What | What the build does instead |
|---|---|
| A step size and an effort of their own for the jaw commands of the trigger; a close that a released trigger cuts short | The trigger's jaw commands use `gripper.open_step`, `gripper.reach_tol` and the guarded opening's effort, 0.5, for a close as for an opening. A close under way runs to its end, or to its stall, before the jaws open again; a STOP cuts it at once |
| A silent agent after a hold on an expired permit counting as failed | The arm holds until someone acts; there is no prompt when an agent goes silent |
| The question "completed or abandoned?" after a takeover, stored in the task record | The task record does not say how a takeover ended |
| The console printing the takeover's steps one at a time, and whether a return to rest would be accepted now | One takeover line, the human program's note in the status line, the bells |
| A remedy table on the console for a refused `rest` | The refusal's own text; the four refusals are named under [A takeover that cannot be ended with right A](#a-takeover-that-cannot-be-ended-with-right-a) |
| A script for the synthetic controller from a file | The built-in scripts `lift`, `grasp` and `idle`; tests hand rows to the constructor |
| The takeover's facts on the live view page; buttons in the browser | `HUMAN DRIVING` over the picture; the console shows the rest |
| A limit on the round trip | It is shown as `health.human.round_trip_s`; a limit needs a bench value first. The sample age itself holds the HUMAN_SETPOINT's way to the robot app |
| The teleop feed's attempt to reach the Teleop Session Server on a thread of its own | The attempt blocks the human program's loop for up to `teleop.connect_timeout_s` (3 s); nothing can be driven then anyway, and `human.stage_keep_s` (4 s) covers it |
| A timed return to rest after lost headset data; tasks typed at the Thor terminal; a human-only mode; gloves; a camera picture in the headset; a priority path for a STOP on the link | Not built |

## The human program

```text
bash run_human.sh --config configs/human.yaml
    [--controller teleop|synthetic:<name>] [--report-s S] [--log-level L]
```

`run_human.sh` sets the environment and runs
`python -m isaaccapture_examples.agentic_remote.human` with its arguments. The program
attaches, in the role `human`, to the Unix socket that `socket_path` of its file names:
on the rig the robot app's own socket on the Thor (see
[The human program on the Thor](#the-human-program-on-the-thor)), on one machine the
robot link service's, which an empty `socket_path` stands for. It reads the right
controller from its controller feed, runs the retargeting graph and the IK once per pass,
and sends STOP, its two OWNER_REQUESTs and HUMAN_SETPOINT. **Start the program that
serves the socket first**: the robot link service, or on the Thor the robot app with
`--human-socket`. One human program at a time: a second one ends.

- **`--controller teleop`** (the default) is the teleop feed: a remote session at
  `teleop.endpoint`. **`--controller synthetic:<name>`** is the script feed with a built-in
  script, `lift`, `grasp` or `idle`.
- **A synthetic controller runs only against a simulated driver.** The program reads
  WELCOME first: with a driver other than `sim` or `apai-mock`, or with none declared,
  `--controller synthetic:<name>` ends it with exit code 1. The robot app checks the same
  and refuses such a takeover.
- **`--report-s S`** (1.0; 0 switches it off) is how often the report line is printed.
  **`--log-level`** is `debug`, `info`, `warning` or `error`; without it the file's
  `log_level` counts.
- **When the program ends while the person owns the arm it sends one STOP first**, with
  the reason `the human program ends (<why>)` and the sender `human_program`. It does so
  on four ends: Ctrl+C or SIGTERM; the attempts to reach the Teleop Session Server ran
  out; a new session of the robot app that this program must not drive; an unexpected
  error in a pass. The STOP goes out only while the newest ROBOT_STATE of the session the
  program took shows the human as control owner and the local connection is alive. Each
  of these ends is a hold with the motors on: nobody owns the arm and the takeover stays
  open. The Thor would hold on the missing HUMAN_SETPOINTs anyway. An end with exit
  code 2 sends no STOP. At each of these ends the STOP goes out before the program
  writes the lines of that end, to its log and to standard error: a terminal that takes
  no line does not hold the STOP back. When the end is a new session that the program
  must not drive, right B read down on that last pass is still sent as a STOP from
  `controller`, ahead of the program's own, and nothing else goes out for that session.
- **A pass whose logic raises an error does not end the program.** The error is logged,
  a repeating one at most once a second, and no HUMAN_SETPOINT of that pass is sent.
  When the error comes in the part of the pass that follows the buttons and their
  requests (the checks that allow motion, the clutch's home, the IK), a STOP or a
  request that was due on that pass still goes out, and nothing follows the hand until a
  new squeeze. An error before the buttons are read, as in the read of the retargeting
  graph's outputs, sends nothing on that pass and leaves the logic as it was: a squeeze
  that was followed is followed on, and right B, not read on that pass, is a STOP on the
  next pass that reads it. An error of the controller feed or of a send ends the program
  with exit code 1 and the line `human program: failed: ...`.
- **A `teleop.endpoint` that Isaac Capture cannot read**, such as a port that is no
  number, ends the program with exit code 2 at its first attempt.
- **The program prints its lines on a thread of its own**: a terminal that takes no
  output holds up neither a pass nor right B. While it takes none, up to 256 lines wait;
  newer ones are dropped.
- **Its log lines can stall it.** Isaac Capture's logging attaches two handlers to the
  logger `isaaccapture`: a console handler that writes to standard error (with
  `ISAACCAPTURE_LOG_SOCKET` set, a forwarding handler in its place) and a rotating file
  handler. Each writes on the thread that logs, with no queue and no writer thread, and
  the example attaches no handler of its own. So with a terminal that takes no log lines
  (paused with Ctrl+S, or standard error into a full pipe) the program's loop stops at
  its next log line: no pass runs, and right B sends nothing until the terminal runs
  again. A test puts a log handler that blocks in the console handler's place
  (`test_a_log_line_that_blocks_on_an_earlier_pass_still_holds_up_right_b_afterwards`).
  The line of one pass stood on the thread that runs the passes: no link message left
  the program in the next 0.5 s, in which 25 passes were due, and right B, held down in
  that time, sent nothing and printed no `B down`. Once the handler took lines again the
  passes went on at once, and right B, still down, was a STOP on the first of them. The
  STOP of the pass whose own line blocks is not held back: the logic's log lines of a
  pass are written after that pass's link messages have gone out. The arm holds
  meanwhile: a human drive ends within `human.sample_age_limit_s` of the last
  HUMAN_SETPOINT. The console's `stop` and Ctrl+C and the Thor terminal's stop still stop
  an agent's move.
- **Without its local connection the program has ended** (exit code 1, `the robot link
  service is gone (...)`). Right B then stops nothing until the program is started again:
  unlike the console, which writes a raw STOP when it is not attached, the human program
  writes none.

| Variable | Needed | What it names |
|---|---|---|
| `AGENTIC_REMOTE_HUMAN_PYTHON` | always; without it `AGENTIC_REMOTE_AGENT_PYTHON` is taken | A Python with numpy, PyYAML and pinocchio, which APAI's arm model needs. No GPU is used |
| `AGENTIC_REMOTE_APAI_SRC` | always | The `src` folder of the APAI checkout |
| `AGENTIC_REMOTE_ISAAC_CAPTURE_PY` | always | The folder of the Isaac Capture Python package built for that Python |

Without one of the three, with a Python that cannot be run or with a folder that is none,
the script prints one line that names the variable and exits with code 2. It puts the
example's `python` folder, the Isaac Capture package and APAI's `src` on `PYTHONPATH`, sets
`PYTHONDONTWRITEBYTECODE=1`, `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=0`,
and unsets `APAI_ROBOT` and `APAI_BASE`. The recipes start it with `bash`, which works
with and without the file's execute bit.

| Exit code | Meaning |
|---|---|
| 0 | Ctrl+C or SIGTERM, also before a WELCOME has arrived |
| 1 | No robot link service (`cannot attach to the robot link service: ...`), a second human program, the robot link service gone (`the robot link service is gone (...)`), an arm model whose joint count is not the one WELCOME names, a synthetic controller against a driver that is not simulated, the same two for a robot app that was restarted while the program ran, a failure the program says as `human program: failed: ...`, or the attempts to reach the Teleop Session Server ran out (`no remote session at <endpoint>: N attempts in a row failed, ...`) |
| 2 | The arguments or the file cannot be used (`human program: the settings cannot be used: ...`), an unknown script name, an `ik.apai_arm` that cannot be loaded, a `teleop.endpoint` that Isaac Capture cannot read (met at the first attempt, after the first printed line; nothing was sent to the Thor), APAI cannot be imported (one line that names `AGENTIC_REMOTE_APAI_SRC`), or Isaac Capture cannot be imported (one line that names `AGENTIC_REMOTE_ISAAC_CAPTURE_PY`) |

A Ctrl+C in the first fraction of a second of the process, before the program waits for
signals, can end it with Python's traceback; nothing was sent then.

### The human program on the Thor

On the rig the human program runs on the Thor, beside the robot app, so that right B's
STOP passes neither the robot link service nor the robot connection. None of this was
run on a Thor, between two machines or with a Teleop Session Server: the tests run it on
one machine with the synthetic controller (`test_human_on_thor.py`).

- **What right B still needs.** The buttons reach the Thor in SAMPLES, over the teleop
  connection. With that connection down, as with the wired link between the machines
  down, the human program knows no button and right B sends nothing; the arm holds,
  because the controller data are away. The stop that needs no network is the one at
  the Thor terminal.
- **The robot app serves the socket.** Started with `--human-socket
  ~/.agentic_remote/human.sock`, the robot app serves a Unix socket that speaks the local
  API for one client of the role `human`. A second human program is refused there, and
  so is every other role. `socket_path` of `configs/human_rebot.yaml` names the same
  path; a path that begins with `~` is taken from the home folder. The socket's folder
  must be the user's own with mode 0700, and the robot app makes it when it is not
  there. The socket's file has mode 0600 and goes when the robot app ends.
- **The teleop connection crosses between the machines.** `teleop.endpoint` of
  `configs/human_rebot.yaml` ships with the placeholder `192.0.2.1:47600`, an address
  at which nothing answers. Write the workstation's address on the wired link in its
  place, and start the Teleop Session Server so that it listens on that address
  (`--listen tcp://ADDRESS:47600`). Nothing checks that the placeholder was replaced:
  left in, it shows only as the stage `no_server`.
- **What the human program sends at the socket is checked as on the robot connection**:
  HUMAN_SETPOINT, STOP and its two OWNER_REQUESTs pass every check they pass there, and
  nothing else is taken from it. It receives WELCOME, ROBOT_STATE and the RESULTs of its
  own requests. Such a RESULT also goes over the robot connection, where the robot link
  service hands it to the consoles.
- **With the robot connection down** the robot app still takes the human program's STOP
  and still sends it ROBOT_STATE. A human drive under way ends in a hold (`robot
  connection lost`). A new squeeze moves nothing where the robot app's file requires
  the live view, because VIEW_STATUS comes over the robot connection, and a takeover is
  refused then with `the live view is not fresh: no VIEW_STATUS has arrived; ...`.
  With `human.require_live_view: false`, which the robot app takes only with a
  simulated driver, a person can take over and drive with the robot connection down.
- **The console** shows `human program at the robot app` in its status line. `clients`
  there lists only what is attached to the robot link service.
- **Start the robot app first, then the human program.** When the robot app ends, its
  socket closes and the human program ends with exit code 1; start it again after the
  robot app. The human program's texts name the robot link service where the socket is
  the robot app's own too: `cannot attach to the robot link service: ...` and `the
  robot link service is gone (...)`. A robot app whose `--human-socket` names a socket
  that a running robot app serves ends with `--human-socket PATH cannot be served: a
  robot link service already runs on PATH`.
- **One human program is followed.** If one runs at each end, the HUMAN_SETPOINTs of
  the second are dropped while the first has sent one within `human.stage_keep_s`
  (4 s), and the console prints once `[refused] a second human program sends
  HUMAN_SETPOINT link messages at the robot app's socket: they are dropped while the
  one over the robot connection sends; one human program at a time`, or the same with
  the two places exchanged. A STOP and an OWNER_REQUEST of the second are not dropped:
  they keep every check.
- **A link message from the socket waits for the next pass of the robot app's
  dispatcher**, at most 10 ms, where one on the robot connection wakes the dispatcher
  at once. That wait is part of the sample age of a HUMAN_SETPOINT.
- **What the Thor needs** in the Python that `AGENTIC_REMOTE_HUMAN_PYTHON` names: numpy,
  PyYAML, pinocchio, APAI's `src` folder and Isaac Capture's Python package with its
  retargeters and its remote session. `run_human.sh` does not read
  `AGENTIC_REMOTE_THOR_PYTHON`: on the Thor set `AGENTIC_REMOTE_HUMAN_PYTHON` to that
  Python.

The commands for the rig. None of them was run, on hardware or anywhere else. What a
test ran as processes, on one machine, is the robot app with `--arm sim`,
`configs/robot_app_demo2.yaml` and `--human-socket`, and `bash run_human.sh` with the
synthetic controller on a copy of `configs/human.yaml` that names that socket
(`test_the_human_program_on_the_thor_runs_as_processes`):

```bash
# Workstation: the Teleop Session Server on the wired link's address.
python -m isaaccapture.teleop_session_server --listen tcp://WORKSTATION:47600

# Thor, terminal 1: the robot app with its socket for the human program.
./run_robot_app.sh --config configs/robot_app_rebot.yaml --arm apai --driver real \
    --apai-profile PROFILE --calibration NAME --connect WORKSTATION:47610 \
    --human-socket ~/.agentic_remote/human.sock

# Thor, terminal 2: the human program, once teleop.endpoint of configs/human_rebot.yaml
# holds the workstation's address.
AGENTIC_REMOTE_HUMAN_PYTHON="$AGENTIC_REMOTE_THOR_PYTHON" \
    bash run_human.sh --config configs/human_rebot.yaml
```

The robot link service, the console, the live view and the agent program run on the
workstation as for demo 1. The Thor terminal shows `robot app: the human program's
socket is PATH` after its session line, and right B's `[stop] STOP from controller:
right B; the arm holds, stop number N` also while it shows `robot app: no robot
connection: ...`.

### `configs/human.yaml`

The program refuses the file (exit code 2) when it holds a key the program does not know,
a number outside its bound (a number without a named bound must be above 0, so
`rates.loop_hz` and `rates.idle_hz` too; for `rates.loop_hz` values above 100 are
refused), or a `retarget.controller_to_base` that is no rotation. A number too large to
use, and a file that is not UTF-8 text, are refused the same way. Each refusal is one line
on standard error; for `rates.loop_hz: 200` it reads `human program: the settings cannot
be used: rates.loop_hz must be at most 100, got 200: the rate limiter counts a pass as at
least 10 ms, so above 100 both hand speed limits would rise above their settings`. A key
the file leaves out keeps the value below. Every number is a first value; none was
measured on a robot or with a headset.

| Block | Keys (shipped values) |
|---|---|
| top | `socket_path`, the Unix socket the program attaches to (empty: the robot link service's default; the rig's file names the robot app's own socket); `log_level` |
| `teleop` | `endpoint` of the Teleop Session Server (`127.0.0.1:47600`); `tick_rate_hz` (90.0), the tick rate asked of the server; `max_tick_age_s` (0.15): older SAMPLES are stale; `connect_timeout_s` (3.0), the longest one attempt blocks; `retry_s` (5.0, at least 2), the pause between two attempts; `attempts` (720), the failed attempts in a row after which the program ends |
| `retarget` | `controller_pose` (`grip`; or `aim`), the controller pose that is followed; `position_scale` (1.0), the target's motion per motion of the hand; `squeeze_threshold` (0.5, below 1): the grip counts as squeezed above it; `max_linear_velocity` (0.10 m/s) and `max_angular_velocity` (0.4 rad/s), the hand speed limits of the rate limiter; `reject_linear_velocity` (2.0 m/s) and `reject_angular_velocity` (10.0 rad/s; `null`: none): a hand faster than that ends the engagement; `controller_to_base`, the rotation, by rows, from the stage space, the coordinate system the controller poses are given in, into the robot's base axes (x forward, y left, z up); `stage_yaw_deg` (0.0) |
| `ik` | `apai_arm` (`mock`), the APAI arm profile whose model the IK solves on; `max_iter` (50); `tol` (1e-4); `max_jump_rad` (0.05); `max_ahead_rad` (0.08); `max_ahead_m` (0.05); `max_failures` (5), the failed solves in a row that end an engagement: a single failed pass sends the last joint values again |
| `buttons` | `hold_s` (1.0), how long right B or right A is held |
| `rates` | `loop_hz` (50.0, at most 100), the passes per second; `idle_hz` (20.0), the HUMAN_SETPOINTs per second while nobody is driving |
| `state` | `max_age_s` (0.2): on an older ROBOT_STATE nothing engages and an engagement ends |

- **`controller_to_base`** rests on the person standing behind the arm after a recentre of
  the headset view: the stage space then has x to the person's right, y up and z toward
  the person, and the hand's forward is the robot's forward. **`stage_yaw_deg`** turns the
  hand's motion about the vertical axis, for a recentre that left the stage at an angle. A
  positive value turns it counter-clockwise seen from above: with 90, a hand that moves
  away from the person moves the target to the robot's left, and a hand that moves to the
  person's left moves it toward the robot's back; up stays up. For reading the arm's
  answer: on APAI's model a positive turn of the first joint swings the gripper tip to the
  robot's right.
- **A stalled pass is not made up.** The rate limiter gives one pass at most 20 ms of
  motion, so while the program's loop stalls the target moves slower than the hand speed
  limits. With `rates.loop_hz` below 50 those limits shrink in proportion: 25 per second
  halves them. A pass that comes sooner than 10 ms after the one before counts as 10 ms.
  Above 100 passes per second both hand speed limits would grow in proportion, so values
  above 100 are refused. The comment beside `rates.loop_hz` in both human files says the
  same. At 100 itself about half of the passes come a little sooner than 10 ms after the
  one before and count as 10 ms, so both hand speed limits rise by half the share of
  that unevenness: 0.5 % with passes 9.9 and 10.1 ms apart in turn and 5 % with 9 and
  11 ms, by a test on the retargeting graph; 0.06 % to 0.6 % on the program's own passes,
  measured once on the workstation apart from the tests. With a period of 15 ms or more,
  up to 66 passes per second, an early pass and the late one beside it never count for
  more than their time.
- **The hand speeds and the joint speeds**, computed on APAI's arm model and not measured
  on an arm: 1 m/s at the gripper tip needs up to 10.1 rad/s at a joint away from rest and
  12.9 rad/s at the pose a drive from rest starts from; 1 rad/s of tool rotation needs up
  to 3.0 rad/s. So the shipped 0.10 m/s and 0.4 rad/s are at most 0.044 rad per pass away
  from rest, under `ik.max_jump_rad` 0.05. Right at the pose a drive from rest starts from
  the two together reach 0.0502 rad: a hand that moves and turns at both limits there can
  end the engagement with `IK jump`. A plain lift from rest at 0.10 m/s is 0.009 rad per
  pass.
- **From rest, lift the gripper first.** The rest pose lies on joint stops. From there
  forward, back and down have no IK solution, and sideways fails after a few millimetres:
  the engagement ends with `no IK solution` and the grip has to be squeezed again. A lift
  solves on every pass. A lift at the limiter's full speed ends about 39 cm above rest
  with `IK jump`: the end of the arm's reach.
- **`configs/human_rebot.yaml`** is the file for the rig, where the program runs on the
  Thor. It differs in exactly six keys: `socket_path: "~/.agentic_remote/human.sock"`,
  the robot app's own socket; `teleop.endpoint: "192.0.2.1:47600"`, a placeholder for
  the workstation's address on the wired link; `ik.apai_arm: rebot_rs`;
  `retarget.max_linear_velocity: 0.01`; `retarget.max_angular_velocity: 0.03`; and
  `teleop.attempts: 8640`. The two speeds are low so that the joint targets stay near
  the 0.1 rad/s the Thor moves a person's target with: by the numbers above 0.01 m/s
  asks a joint for 0.10 to 0.13 rad/s. `configs/human.yaml` is the file of the
  one-machine recipes.

### What it prints

One line each on its standard output. The steps F2 to F7 of the checklist read these lines.

```text
human program: <teleop | synthetic controller "NAME">; driver <d>; arm model <apai_arm>, <n> joints
human program: Teleop Session Server <stage>[: <text>]
human program: right controller seen | right controller lost
human program: <A | B | grip | trigger> <down | up>
human program: stage <s>; tick age <lowest>/<middle>/<highest> ms;
    state age <lowest>/<middle>/<highest> ms; round trip <r> ms; owner <o>;
    engaged <yes|no>; hand x <..> y <..> z <..> m;
    tip target x <..> y <..> z <..> m, roll <..> pitch <..> yaw <..> deg;
    IK <middle>/<highest> ms; ended: no solution <n>, jump <n>, solution ahead <n>,
    hand ahead <n>, limiter <n>
    [; jaws follow the trigger <yes|no>, grip level <g>, trigger asks <a>[: <why>]]
```

- The first once, after WELCOME; a robot app that declares no driver shows as `not
  declared (treated as real)`. The second at each change of the teleop feed's stage; the
  script feed prints none. The third at each change. The fourth at each edge: the grip at
  `retarget.squeeze_threshold`, the trigger at 0.5. A controller that is not active
  reads every button as up, so `A up`, `B up` and `grip up` are also printed when the
  controller stops being active with one of them down: `grip up` is not always a
  released grip.
- The last is the **report line**, one line every `--report-s` seconds. The tick ages, the
  state ages (how long ago each pass's ROBOT_STATE arrived) and the IK times are those of
  the passes since the line before. `round trip` is
  `health.human.round_trip_s` of the newest ROBOT_STATE. `hand` is the right controller's
  position in the stage space; `tip target` is the target of the gripper tip in the
  robot's base axes, and roll, pitch and yaw are its turns about x, y and z of those
  axes. `ended:` counts the engagements the human program ended, by cause. The
  part `; jaws follow the trigger ...` is there while ROBOT_STATE carries
  `health.human.jaws`, so while a takeover is open: whether the trigger has reached the
  grip level, the grip level and the jaw opening the trigger asks for, each as a share
  of the jaw travel with two decimals, and behind a colon the robot app's reason when
  it gives one. The
  line ends with `; refused by the Thor: <reason>` while `health.human.refusal` says why
  the Thor moves nothing on an engaged HUMAN_SETPOINT. A value that is not known prints
  as `-`.
- What the program cannot do is one line on its standard error that begins `human
  program: `.

### The stages and the notes

Every HUMAN_SETPOINT carries the feed's stage, and ROBOT_STATE and the console's status
line repeat it:

| Stage | Meaning |
|---|---|
| `no_server` | No connection to the Teleop Session Server: nothing answered, the connection was lost, or the server said why it takes no client |
| `connecting_openxr`, `waiting_for_headset` | The server's own words for what it waits for |
| `ready` | The remote session was entered, the SAMPLES are fresh and the right controller is active with a valid pose. The server's own `running` is never forwarded |
| `stale` | The session was entered and the SAMPLES are older than `teleop.max_tick_age_s` |
| `no_controller` | The SAMPLES are fresh and the right controller is not active or its pose is not valid |

It also carries a **note** for the operator, which the console's status line shows as
`note: ...`: what the person does next, or why the arm does not follow. The wearer of the
headset cannot read it; the console's bells and a second person are for that.

| Note | When |
|---|---|
| `hold right B for 1 s with the grip released to take over` | No person owns the arm and the stage is `ready` |
| `release right B, then hold it for 1 s with the grip released to take over` | Right B is down with no press that can ask: it went down while controller data were missing or the controller's pose was not valid, or the grip was squeezed on a pass of the press. Holding on asks for nothing |
| `takeover refused: <reason>`, `handback refused: <reason>` | The program's request did not end `completed`; the word after the request is the RESULT's status. The note goes once ROBOT_STATE shows another control owner, STOP or motion than at that RESULT, and at the next press that counts |
| `release the grip, then squeeze it to move` | The person owns the arm and the grip is squeezed without an engagement, or the Thor's needs-release flag is set |
| `squeeze the grip to move` | The person owns the arm and the grip is released |
| `release the grip and squeeze again: <why>` | The engagement ended other than by a released grip. `<why>` is the human program's reason; or, when the robot app asks for a released grip, the `busy: ...` work that refused the squeeze, else why the robot app's last drive ended |
| `at rest: hold right A for 1 s to give the arm back` | A takeover is open and the arm holds at rest; while the person owns the arm, only once the arm was moved in this takeover |
| `release right A, then hold it for 1 s with the grip released to give the arm back` | In place of the note above while right A is down with no press that can ask (as for right B, or one such pass ended its hold). Holding on asks for nothing. While ROBOT_STATE shows the jaws as `loaded` the jaws' note comes first |
| `at rest: jaws not proven empty: at the console type stop, open, yes; then hold right A for 1 s` | In place of the two notes above while ROBOT_STATE shows the jaws as `loaded`: the robot app refuses that hand-back. Jaws shown as `unknown` keep the notes above |
| `pull the trigger to the grip level to move the jaws`, `ease the trigger back to the grip level to move the jaws` | The person owns the arm, the arm follows the hand, and ROBOT_STATE says that the trigger has not reached the grip level (`health.human.jaws`). `ease` while the trigger asks for a narrower opening than the grip level, `pull` otherwise. During an engagement there is no other note; with the grip released the notes above stand and the trigger gets none |
| `the arm returns to rest; right B is a STOP` | A takeover is open, recovery is the control owner and the stage is `ready`: a return to rest runs, for a typed `rest` or for a shutdown. It stands in place of every note above, also of a refusal's. Right B then ends the return in a hold with the motors on: nobody owns the arm and the takeover stays open. While the stage is not `ready` right B may not be read, and the feed's own text stands |
| The feed's own text | The stage is not `ready`: `the SAMPLES of the Teleop Session Server are too old`, `the right controller is not active or its pose is not valid`, `the synthetic controller is absent`, or the start of Isaac Capture's error text of a failed attempt |

A note is printable ASCII of at most 100 characters: a longer one is sent as its first 40
characters, `...` and its last 57, and any other character becomes `?`. Behind `release
the grip and squeeze again: ` a reason over 64 characters is sent as its first 40
characters, `...` and its last 21.

After a refused request, with the button still held, the plain note (`hold right B ...`,
`at rest: hold right A ...`) comes back once the refusal's note has gone, although that
press asks for nothing more: release the button and hold it again.

### The teleop feed

- **It is a remote session** with the client name `agentic_remote_human`, on the same
  retargeting graph the script feed runs. It is not wrapped in a CloudXR launcher: CloudXR
  and the Teleop Session Server are started by hand (steps F1 to F3 of the checklist).
- **On the rig the session crosses between the machines**: the human program on the Thor
  dials `teleop.endpoint`, the workstation's address on the wired link. On one machine
  the endpoint is `127.0.0.1:47600`.
- **Entering blocks one pass for at most `teleop.connect_timeout_s`.** The program sends
  nothing meanwhile; `human.stage_keep_s` (4 s) on the Thor is longer than that. A session
  that cannot be entered, or that is lost, is tried again `teleop.retry_s` later with a new
  session.
- **The stage of a failed attempt is read from Isaac Capture's error text**: a timeout that
  says the server did not answer gives `no_server`; one that names `connecting_openxr` or
  `waiting_for_headset` gives that stage; everything else gives `no_server`.
- **`teleop.attempts` counts failed attempts in a row.** An entered session sets the count
  back to 0. When it is reached the program ends with exit code 1, and the console says
  `no human program is attached: start run_human.sh`. A headset asleep on the table is not
  READY, so each attempt fails: 720 attempts are about an hour and a half.
- **The feed's sample age is the session's tick age.** With SAMPLES older than
  `teleop.max_tick_age_s` there is none and the stage is `stale`. As it sends a
  HUMAN_SETPOINT the program adds the time its pass took since it asked the feed.
- **Right B is read while the controller's pose is not valid.** On such a pass the feed
  gives the controller data with the tick age and the stage `no_controller`, so right B is
  still a STOP.

### The synthetic controller

The script feed is how the tests and the one-machine recipe feed controller data without a
headset. For each pass it builds one controller snapshot of Isaac Capture's schema for the
right controller and runs the retargeting graph on it. The graph's leaf is the same one a
remote session fills from its controller tracker. So from the controller snapshot upward
the script feed runs the code the headset path runs: the conversion, the rotation, the
clutch, the rate limiter, then the IK and everything after it.

A script is a list of rows. The built-in script `lift`:

```text
{"t": 0.0, "pos": [0.0, 1.2, -0.3], "quat": [0, 0, 0, 1], "squeeze": 0.0, "trigger": 0.0, "a": false, "b": false}
{"t": 3.0, "b": true}
{"wait_for": "owner_human"}
{"t": 3.5, "b": false}
{"t": 5.0, "squeeze": 1.0}
{"t": 9.0, "pos": [0.0, 1.25, -0.3]}
{"t": 9.5, "squeeze": 0.0}
{"wait_for": "rest_hold"}
{"t": 11.0, "a": true}
{"wait_for": "owner_nobody"}
{"t": 11.5, "a": false}
```

- `t` is seconds on the script's own clock, which starts with the first pass. A row keeps
  every value of the row before it that it does not name.
- `pos` (metres, the stage space) moves in a straight line from the row before a row that
  names `pos` to that row, over the time between those two rows. `quat` is `[x, y, z, w]`.
  `squeeze` and `trigger` run from 0 to 1; `a` and `b` are the buttons.
- `"absent": true` makes the controller not active: the stage is `no_controller` and there
  is no sample age. `"pose_valid": false` keeps it active with a pose that is not valid.
  `"age_s"` sets the sample age, which is 0 otherwise.
- `{"wait_for": WORD}` stops the script's clock until the newest ROBOT_STATE shows it:
  `owner_human`, `owner_nobody` or `rest_hold`.
- A row with another key, and a `t` before the row before it, are refused.

In `lift` the hand stays at y 1.2 until t = 5.0 and rises to y 1.25 at t = 9.0: 5 cm at
0.0125 m/s while the grip is squeezed. It presses right B, lifts, and waits for a typed
`rest` before it presses right A. When the robot app refuses the takeover, the script
keeps right B down and waits: nothing ends that wait, and the refusal shows as the note in
the console's status line. `idle` holds a controller that is there with nothing pressed.

`grasp` moves the jaws and not the arm. It presses right B as `lift` does, pulls the
trigger all the way at t = 5.0 and releases it at t = 9.0, with the grip released
throughout, and presses right A at t = 15.0, once the arm holds at rest, which it does
all along. On open and empty jaws the pulled trigger closes them on nothing and the
released one opens them again.

A script from a file is not built; the tests hand rows to the feed.

### What is not tested without a headset

The tests run the teleop feed on a real remote session against a **server in miniature**
inside the test's process, on 127.0.0.1 and a port the system picks, built from the host
side of Isaac Capture's remote session as Isaac Capture's own tests build it. Its SAMPLES
are empty and still carry their tick time. That covers the entry, READY, the stage texts,
the staleness and the re-entry. Everything from the controller snapshot upward is covered
by the script feed.

Not covered by any test, because Isaac Capture has no way to send controller SAMPLES
without a headset:

- Controller data inside SAMPLES, the Teleop Session Server program itself, CloudXR and a
  headset. Their first run is step F3 of the checklist.
- The remote session between two machines, from the human program on the Thor to the
  Teleop Session Server on the workstation. Its first run is step F2.
- The trigger of a real controller: what it reads at rest and fully pulled, against the
  two dead zones of 0.05. Step F12b.
- Whether right A and right B arrive over a remote session. Only the grip and the trigger
  have been seen on hardware. Step F6 finds out; the typed fallback is the way around.
- How the wearer sees the arm and the live view with the headset on. Step F5.
- A recentre of the headset view during a squeeze, which moves every controller pose at
  once. The rate limiter's reject speeds and the Thor's step limit are meant to turn that
  into a hold, not a jump. Step F11.
- The direction of the hand's motion against the arm's (`controller_to_base`,
  `stage_yaw_deg`). Step F7.

## The rig files

Four files hold the settings for the rig. Every number in them is a first value: none was
measured on the arm, and a comment `first value` marks what the hardware session has to
measure.

| File | Machine | For |
|---|---|---|
| `configs/robot_app_rebot.yaml` | Thor | The robot app with `--arm apai --driver real` |
| `configs/agent_rebot.yaml` | Workstation | The agent program with the full stack and `--camera link` |
| `configs/link_service_rebot.yaml` | Workstation | The robot link service: it listens on every address of the workstation |
| `configs/human_rebot.yaml` | Thor | The human program with the headset, attached at the robot app's own socket (see [The human program on the Thor](#the-human-program-on-the-thor)) |

- **Human driving is switched off** (`human.enabled: false`), and so is the typed takeover
  (`operator.typed_takeover: false`). The first is switched on in the session, after
  part F-A of the checklist; see [The settings that ship off](#the-settings-that-ship-off).
- **The trigger's setting ships on** (`human.trigger_jaws: true`, `human.jaw_match_tol:
  0.05`), as in every robot app file. It acts only once `human.enabled` is true. For a
  first headset session on the arm without the trigger, set `human.trigger_jaws: false`
  in `configs/robot_app_rebot.yaml` before step 15.
- **The human program's file is for the Thor.** `socket_path` of
  `configs/human_rebot.yaml` is `~/.agentic_remote/human.sock`, the path the robot app
  is given as `--human-socket`. `teleop.endpoint` is the placeholder `192.0.2.1:47600`:
  write the workstation's address on the wired link in before the first run (see
  [The human program on the Thor](#the-human-program-on-the-thor)).
- **The human joint box lies 0.02 rad inside `rest.region_*`.** The region, the start
  poses from which a return to rest is taken, is also the agent's working box: 120
  degrees around 0 on the first and the sixth joint, up to 120 degrees on the second and
  the third. `limits.human.joint_min` is `[-2.0744, -0.05, -0.05, -1.67, -1.55, -2.0744]`
  and `limits.human.joint_max` is `[2.0744, 2.0744, 2.0744, 1.77, 1.55, 2.0744]`; the
  region runs from `[-2.0944, -0.05, -0.05, -1.69, -1.57, -2.0944]` to
  `[2.0944, 2.0944, 2.0944, 1.79, 1.57, 2.0944]`. The lower bound of the second and the
  third joint is the region's own, with no margin: those two joints lie on their stops
  at rest and cannot be driven below them, and a margin there could refuse a takeover at
  rest.
- **What the margin is for.** A joint that a person drove to the edge of the box can read
  a few thousandths of a rad outside the box. A STOP at such a pose leaves no squeeze:
  nobody owns the arm then, and a new takeover is refused there too (`the arm is outside
  the human limits: ...; type rest`). With the margin that reading still lies inside the
  region, so the typed `rest` is taken and returns the arm. A person cannot leave the arm
  outside `rest.region_*`; the return to rest can still refuse its path or the tip.
- **What the margin costs.** In the outer 0.02 rad of the agent's working box a takeover
  is refused, also where an agent's move ended there. The typed `rest` is the way out
  (see [Known gaps](#known-gaps)). The console's takeover line does not say so
  beforehand: `health.human.takeover` in ROBOT_STATE is made without a joint read, so
  the line can read `takeover: possible now ...` there, and right B's hold is then
  refused, with three bells.
- **`test_rig_configs.py` checks both on the rig file**, with human driving switched on
  in memory, on the simulated arm and on APAI's checkers and driver over a stand-in for
  the motor library, at each of the ten bounds that have the margin. With a joint
  0.005 rad outside its bound at a STOP: the takeover and a task of the agent are
  refused, nothing is sent, and the typed `rest` returns the arm. With a joint 0.01 rad
  outside its bound and nobody as control owner: the takeover is refused and the typed
  `rest` returns the arm. At the two bounds without the margin a reading 0.005 rad below
  the bound gets neither: the typed `rest` is refused for the region.
- **No rule of the loader asks for the margin.** With `human.enabled` the robot app
  refuses a file whose human joint box is wider than the region, and takes one whose box
  equals it. The three files that switch human driving on for a simulated arm have no
  margin: in `robot_app_demo2.yaml` and `robot_app_human_check.yaml` the box equals the
  region, in `robot_app_mock.yaml` on every joint but the first. A simulated arm reads
  exactly what it was sent. On a real arm such a file keeps the gap the margin closes
  (see [Known gaps](#known-gaps)).
- **Every joint speed is 0.1 rad/s**: the agent's, the human's and recovery's. The rig's APAI
  profile caps the joint speed, and a faster limit set is refused.
- **The timed return is off** (`rest.auto_return: false`, `task.timed_return: false`): the
  arm holds when the assistance wait runs out.
- **The agent's and recovery's joint boxes are the limits of APAI's arm model**, not a
  measurement of this arm, with one change: the two joints that lie on their stops at rest
  may read down to -0.05 rad.
- **`camera_calibration_id`** in the robot app's file names the calibration in use on the rig.
  The agent program builds the same name from its copies of the calibration and prints it
  with `--check`.
- **`agent_rebot.yaml` needs three variables**: `AGENTIC_REMOTE_EXPERIMENT_DIR`,
  `AGENTIC_REMOTE_CAMERA_PR` and `AGENTIC_REMOTE_ENV_DIR`. Its paths name them, and the
  program refuses a path whose variable is not set or whose value is not an absolute path.
- **`link_service_rebot.yaml` is refused as shipped** (exit code 2): `allowed_peers` is
  empty. Write the Thor's address into it. An entry that is not an IP address is refused
  too.

The camera's two files, `configs/camera_sender_rebot.yaml` and
`configs/camera_receiver_rebot.yaml`, belong to the camera streamer's programs.

## The workstation environment

The full stack (`experiment.stack: full`) needs a Python on the workstation with APAI's
packages, torch with CUDA and Warp. `setup_workstation_env.sh` builds it once, with the
versions in `requirements-workstation.txt`:

```bash
AGENTIC_REMOTE_ENV_DIR=/absolute/folder ./setup_workstation_env.sh --dry-run
AGENTIC_REMOTE_ENV_DIR=/absolute/folder ./setup_workstation_env.sh
```

The folder lies outside every git repository and not under `/tmp`. The script makes `venv/`
(Python 3.12), `caches/` and `freeze.txt` in it, and downloads several GB. `--dry-run`
prints the commands and runs none. It refuses (exit code 2) a path that is not absolute or
holds a `.` or `..` component, a folder inside a repository (symbolic links are followed,
and a line end at the end of a folder's name counts as part of the name), a symbolic link
in the folder where the script makes something (`venv`, `caches`, `codex`, `freeze.txt`,
`exclude-gui-opencv.txt`), an existing `venv/` and a missing `uv`. `run_agent.sh` then takes
`AGENTIC_REMOTE_AGENT_PYTHON=<folder>/venv/bin/python` and `AGENTIC_REMOTE_ENV_DIR=<folder>`.

The environment was built on the workstation on 2026-11-15 with this script, without the
model client: its `freeze.txt` lists torch 2.10.0 for CUDA 13.0 and Warp 1.18.0.
`test_workstation_env.py` runs the script with stand-ins for `uv` and `npm`, so it installs
nothing. `test_gpu_stack.py` is the check to run inside the environment: torch and Warp on
GPU 0, the obstacle grid bridge, the stereo depth, the segmenter and the full stack's start
on recorded stereo pairs. Its tests are marked `gpu` and are skipped everywhere else;
[What was run](#what-was-run) reports their run in the environment.

**The model client** (the Codex command-line program the full stack asks its cloud model
through) is not installed. `--with-model-client` installs it with npm into `<folder>/codex`,
and `experiment.codex_bin` names it. Installing it and using a login for it wait for the
owner's decision; the script never reads or writes that login.

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
  before the control owner changed is refused: a confirmation typed for one state of the arm
  does not move it in another.
- **A shutdown that has to move the arm needs the path confirmed**, like a typed `rest`: SHUTDOWN
  carries `confirmed`, and `quit` at the Thor terminal asks for a typed `path is clear`.
  Unconfirmed it is refused with `confirm the path is clear` and the app keeps running. At
  rest, or with the motors already off, the Thor terminal asks nothing. The pose is measured again when the shutdown runs: if the arm
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
- **After a torque release that failed, only a release that works changes anything.** The
  motion state is `release_failed`: the motors may be on or off (see
  [the torque rule](#the-rule-torque-is-never-cut-away-from-rest)). `rest`, `open`, a new
  task, a takeover and every command that would move the arm or the jaws are refused with
  `a torque release failed and the motors may be on or off; type release to try it again,
  or restart the robot app`. `reset` clears the fault-stop and nothing else. `release`
  is then checked and tried again. `quit` and `shutdown` start no return to rest and ask
  for no `path is clear`: they try the release again, and the robot app closes only if it
  worked. The console's status line and the console command's `state` show `torque
  unknown`.
- **The timed return is tried when the assistance wait runs out, or not at all.** If a return to
  rest, a release or a shutdown is running or queued at that moment, the EVENT
  `no timed return to rest: busy: ...` goes to both terminals and the arm is not moved later.
- **A hold at a contact ends when the arm is next moved**: by any later move, or human
  drive, that sent a setpoint and did not itself end at a contact. Until then `rest` is
  refused, except while a takeover is open: no agent move can end the contact hold then,
  and a `rest` confirmed with `path is clear` is taken (see
  [Demo 2: a person takes over](#demo-2-a-person-takes-over)).
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
- **A jaw command the agent gave up on does not run later.** The arm adapter waits
  `jaws.result_timeout_s` (12 s) for the final RESULT of a jaw command. Without one it
  sends a STOP with the sender `agent`, and that STOP ends the command on the Thor: one
  that waits behind another does not start, one that arrives afterwards quotes the old
  stop number and is refused, and one that runs ends as at any STOP, with the jaws held
  lightly where they are, never opened. Nobody owns the arm and the motors stay on. Both
  terminals show `[stop] STOP from agent: the agent gave up on a jaw command: no RESULT
  within 12 s; the arm holds, stop number N`. The task's help request is then refused,
  because the ownership it quotes has ended: no help block and no assistance wait
  follow.
- **A GRIPPER can carry an expiry**, `expires_mono_s`, a time of the Thor's monotonic
  clock, as a MOVE does. The robot app refuses a GRIPPER that arrives at or after that
  time, and one whose time passes while it waits behind another jaw command, with `the
  jaw command expired X.XXX s ago`; the jaws are sent nothing. A GRIPPER without the
  field never expires. The agent program sets it to the end of its own wait for the
  RESULT. The console's guarded opening, the typed reference and the jaw steps of a
  person's trigger carry no expiry: they are no GRIPPER link messages.
- **Under a person the jaws follow the trigger**, from the grip level on, where
  `human.trigger_jaws` is true (the rule on the jaws under
  [A takeover, step by step](#a-takeover-step-by-step)). Each step is one jaw command
  that quotes the owner number and the stop number of its HUMAN_SETPOINT. It is refused
  under a fault-stop, when the control owner is not a person, when the numbers are not
  the current ones and while any other jaw command runs or waits; no link message can
  ask for such a step. A step toward a wider opening goes one `gripper.open_step`
  through the guarded opening; a step toward a narrower one is a close to that opening,
  which ends in the light hold when the jaws stall. Nothing is sent when the jaws are
  within `gripper.reach_tol` of the opening, or hold something with a jaw target no
  wider than it: the light hold stands. A STOP ends such a step as it ends every jaw
  command.
- **`reference` at the Thor terminal makes the jaw position known again.** Jaws that are not
  referenced are not proven empty, so an arm that holds away from rest with such jaws can
  neither return to rest nor be released. The command closes the jaws slowly to their stop,
  after a typed `jaws are empty`. It is taken only while nobody owns the arm, the motors are
  on, no fault-stop is latched, nothing else runs and the jaws are not referenced; a STOP
  ends it `cancelled`. While it waits or runs, a new task (OWNER_REQUEST "agent"), an
  OPEN, a takeover, a hand-back and a `rest` typed at the Thor terminal are refused with
  `busy: a jaw command is running or waiting`, so nobody owns the arm and neither an
  opening nor a return to rest waits behind the reference. A MOVE or a GRIPPER command is therefore
  refused by the owner check (`control owner is nobody, not agent`), or by the stop-number
  check if it was built before a STOP; both look for a running reference behind the owner
  check, as a second barrier. A lost
  robot connection does not cancel the reference: it was typed at the robot. If a STOP
  could not end a jaw command within 0.25 s, the STOP's line at both terminals ends `a
  cancelled jaw command has not returned: until it does the jaws take no other command, and
  rest and release are refused`. The reference proves nothing about
  what the jaws hold: an `open` follows before a `rest`. This is a jaw rule the design does not have, because it moves the jaws on a
  person's word. It waits for the owner's decision and is switched off: the robot app
  setting `jaws.allow_reference` is `false` in every shipped file, the rig file included,
  and the command is then refused without a question.

## What a move does

- **The first setpoint is checked from the measured pose**, not from the start pose the MOVE
  claims. A MOVE whose first setpoint is farther from where the arm is than one step at its
  owner's speed is refused, however close its claimed start is. The return to rest is planned
  again from the pose the executor itself reads, so two reads of a still arm that differ a
  little do not refuse it.
- **What held before the check is asked again before the send.** If the check took time, the
  STOP, the permit and the large delay are tested once more; a setpoint whose permit ran out
  meanwhile is not sent. For the first setpoint the MOVE's expiry is tested once more as
  well: a MOVE that expired during that check is `refused` with `the move expired X.XXX s
  ago` and nothing is sent, also when the same check made the setpoint late. A send that
  itself took longer than the large delay ends the move
  `late`. Not covered: the send call itself, between its start and the motors.
- **A move is `completed` only on one read of every joint.** After the last setpoint the
  move settles: the contact watch goes on reading, a few joints per period. A reading
  that is not valid takes back what an earlier one showed for its joint. Once every
  joint's newest reading is valid and within `executor.settle_tol_rad` (0.03 rad) of
  the last setpoint, one read of every joint decides: only if it has every joint valid
  and within the tolerance is the move `completed`, and the pose of that read is the
  end pose the RESULT carries. A full read that fails or disagrees sends the move back
  to settling. When `executor.settle_timeout_s` (2 s) runs out first, the move ends
  `failed` with the outcome `not_settled` and `joint index J is not within 0.03 rad of
  the last setpoint after 2 s`: nothing is sent after the last setpoint, which stands
  with the motors on.
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

## Known gaps

What is known not to work as one would wish, and is left as it is. None of these makes a
program cut torque: where they touch the arm at all, it holds.

On the human path:

- **A low gripper tip is lifted out by a person only.** With APAI's checkers, an arm that
  an agent leaves low in a grasp, its gripper tip below the table clearance or inside
  the agent's last grasp region, takes no typed command that moves it: the typed `rest`
  and a new task of the agent are refused there. A person takes it over and lifts it
  out, which needs a working controller. With the headset path dead what is left is
  the 48 V switch with the arm held by hand.
- **The lift is strict.** While the gripper tip is low, a hand that sinks by more than
  1 mm ends the drive (`lift the gripper: ...`), and below APAI's own table clearance
  its human checker ends it on every step that does not rise. A hand's tremor in the
  first centimetres of a lift can end drives; each costs a release and a new squeeze.
  No hand has tried it.
- **A grasp region without a ceiling binds a person at every height.** The agent's last
  grasp region stays with the robot app until the agent clears it or sends another, and
  the agent gets no task while a takeover is open. Inside a region with no top the
  gripper tip is never lowered: the person leaves it sideways.
- **The rule of the lift knows the gripper tip only.** An object in the jaws that
  reaches lower than the tip is not known to it: with the tip above the clearance and
  outside the region nothing keeps that object off the table.
- **A close under the trigger is not cut by a released trigger.** It runs to its end,
  or to its stall, before the jaws open again. A STOP cuts it at once. An opening goes
  one `gripper.open_step` per jaw command, so from closed to fully open it takes twenty
  jaw commands with the shipped 0.05.
- **On the real arm a close that a STOP or stale controller data cut goes on softly.**
  It is left in the light hold at the close's own target, as a close of the agent is,
  and at the next takeover that target is the grip level. The simulated arm does not
  move in the light hold, so no test shows it.
- **The grip level is not measured at the takeover.** It comes from the last jaw
  target, or from the remembered jaw reading: a takeover reads no jaw.
- **A trigger that is past the grip level at the takeover moves nothing** until it
  comes back to within `human.jaw_match_tol` of it or crosses it.
- **A jaw opening that is no number from 0 to 1 stops the jaws only.** The joint target
  of that HUMAN_SETPOINT is judged by its own fields.
- **With the robot connection down the robot app takes a takeover from its own
  socket.** Where its file requires the live view, as on the real arm, that takeover
  and every human drive are refused, because VIEW_STATUS comes over the robot
  connection. With `human.require_live_view: false`, on a simulated driver, a person
  can take over and drive with the robot connection down.
- **The human program on the Thor does not outlive the robot app.** Its socket closes
  with the robot app; the program ends with exit code 1 and is started again after the
  robot app.
- **Nothing checks that the placeholder in `teleop.endpoint` of
  `configs/human_rebot.yaml` was replaced.** Left in, it shows as the stage `no_server`.
- **A link message from the robot app's own socket waits up to 10 ms** for the next
  pass of the robot app's dispatcher: right B's STOP from the Thor's human program is
  served that much later.
- **The human program was not run between two machines or against the
  Teleop Session Server program.** The tests of the Thor's placement run on one machine
  with the synthetic controller.
- **A contact that cannot be cleared by hand, with the headset path dead.** With a takeover
  open and the arm raised, the typed `rest` is taken from the contact hold and ends at the
  same contact again. What is left is the 48 V switch with the arm held by hand.
- **A path to rest that is refused, with the headset path dead.** A person cannot leave the
  arm outside `rest.region_*`, but from a pose with a compound turn of the wrist or the
  base the return to rest can refuse its path (`the path to rest was refused: setpoint N:
  ...`). The way out is a squeeze that moves the arm back the way it came. With no working
  controller no typed command moves the arm then: it holds with its motors on. See
  [A takeover that cannot be ended with right A](#a-takeover-that-cannot-be-ended-with-right-a).
- **In the outer 0.02 rad of the agent's working box a takeover is refused on the rig.**
  The rig's human joint box lies 0.02 rad inside `rest.region_*`, which is the agent's
  working box (see [The rig files](#the-rig-files)). Where a joint stands in that outer
  band, after an agent's move that ended there or after a STOP or a fault-stop at the
  edge of the human joint box, a takeover is refused (`the arm is outside the human
  limits: ...; type rest`). The typed `rest` is the way out: the pose is inside the
  region. If its path is refused there (`the path to rest was refused: ...`), no command
  moves the arm: the squeeze that would move it back needs a takeover. What is left
  then is the 48 V switch with the arm held by hand. The lower bound of the second and
  of the third joint has no margin: a reading below -0.05 rad there at a STOP gets
  neither a takeover nor a typed `rest`. Those two joints lie on their stops at rest, and
  a joint target does not drive a joint below its stop.
- **A robot file whose human joint box equals its region keeps a wider gap on a real
  arm.** The three files that switch human driving on for a simulated arm are such files
  (see [The rig files](#the-rig-files)). On a real arm a joint driven to the edge of the
  box can read a few thousandths of a rad outside the region. After a STOP or a
  fault-stop at such a pose nobody owns the arm: a new takeover is refused, the typed
  `rest` and a shutdown are refused for the start pose, and the agent gets no task while
  the takeover is open. What is left is the 48 V switch with the arm held by hand. The
  rig file's margin closes this; no rule of the loader asks for it.
- **A hand-back with jaws that are not referenced names a remedy that does not work.** Its
  refusal ends `type stop, then open, then yes, then hold right A again`, and `open` is
  refused too. The jaws have to be referenced, which the shipped files do at the robot
  app's start.
- **With the human program gone, right B stops nothing.** When its local connection dies
  the human program ends; it writes no raw STOP. Right B is a STOP again once the program
  was started again.
- **A human program whose terminal takes no log lines stalls.** With the terminal paused
  (Ctrl+S) the program's loop stops at its next log line, and right B sends nothing until
  the terminal runs again. The arm holds meanwhile; the console's `stop` and Ctrl+C and
  the Thor terminal's stop still work.
- **Ending the agent program with Ctrl+C or SIGTERM is a STOP**, also while a person
  drives: the human drive ends in a hold, nobody owns the arm and the takeover stays open.
  Only a crash of the agent program changes nothing for the person.
- **APAI's contact monitor trips only beyond its lag band.** It counts a tracking error
  only once the arm trails its command by more than the command moved in the monitor's
  lag time: about 0.03 rad at the rig's 0.1 rad/s. Step F16 of the checklist writes down
  what it reports.
- **A `"visible": true` held up past a second fetch of its picture number still counts.**
  The view cannot tell which fetch of a picture number an acknowledgement is about, so
  such a late one makes the view live until the hidden tab's word about its second fetch,
  or until `status.ack_timeout_s` has passed.
- **A STOP between a new robot connection and its first ROBOT_STATE** keeps PERMITs and
  HUMAN_SETPOINTs at the robot link service until a later STOP is counted: the first
  ROBOT_STATE of a connection never lifts that hold, because it may be older than the
  STOP. The console's status line says `a STOP is not confirmed yet`; press right B or
  type `stop` again.
- **Right B released and pressed again within less than 1 s without controller data** is
  one stretch and sends no second STOP. The stop at the Thor terminal needs no headset.
- **A STOP behind a refused LOCAL_HELLO** goes out only when it is the record right behind
  it. A STOP that a refused client sends later is not read.
- **The hand-ahead limit compares positions only.** A hand that was turned and is then held
  still leaves the tool turning at `retarget.max_angular_velocity` until it has caught up
  or the grip is released.
- **A drive starts from the measured pose.** On an arm that sags under load each new
  squeeze with a still hand lowers the standing target by the sag, so repeated squeezes
  can let the arm creep down. Part F of the checklist watches for it.
- **A sample age is counted as more than it is, by the ROBOT_STATE's way to the human
  program.** The human program says when it sent on the robot app's clock as its newest
  ROBOT_STATE gives it, which is never later than the true moment. Where the human
  program is attached to the robot link service and the robot connection is slow, a
  sample age can pass `human.sample_age_limit_s` and hold the arm while the controller
  data are younger than the limit: a takeover is refused with `no fresh controller
  data: X s old, the limit is 0.25 s`, and a drive ends with `controller data stale
  (...)`. At the robot app's own socket that way is local. A view age is not counted
  as more: the view program's own age is short by that same way. The round trip is
  shown and nothing is enforced on it.
- **Two stretches are in no sample age.** The time from the controller's own sampling to
  the tick of the Teleop Session Server is measured by nobody. Where the teleop
  connection crosses the two machines, the tick age rests on the clock-offset estimate
  of Isaac Capture's remote session, which is an estimate and no upper bound.
- **The robot app and the programs that send to it must be of one build.** A live view
  program without `sent_mono_s` in VIEW_STATUS gets every takeover refused (`the live
  view is not fresh: the VIEW_STATUS does not say when it was sent; ...`). A human
  program without it in HUMAN_SETPOINT is counted from the ROBOT_STATE it quotes.
- **The first HUMAN_SETPOINT after the teleop feed entered a remote session is stale.**
  Its sample age holds the seconds the entry took. The next pass is fresh, and nothing
  can engage on that first one.
- **The Teleop Session Server serves one client.** While the human program holds it, a
  teleoperation session of Isaac Capture on the same server cannot run.
- **The bells of one console line are written together.** Whether one, two and three bells
  ring apart at a real terminal was not tried.
- **The page's script ran in no test** (see [The live view](#the-live-view)).

On the torque release, the agent's jaw commands and the agent program's options:

- **After a torque release that failed, three texts still speak of motors that are on
  or of an arm that holds.** The agent program's line before a task's question reads
  `torque on`, and the task is refused by the robot app. The robot app's line at a
  SIGTERM reads `robot app: SIGTERM; closing with the motors left on`; its log line
  says `torque on: not known`. The fault-stop's line ends `the arm holds until a
  person types reset`. The motion `release_failed` and the console's `torque unknown`
  are the ones to read.
- **On the real arm a failed release may leave no second try.** Read from APAI's
  driver, not run: a disconnect that raises drops the driver's motor session, so the
  joint reads fail afterwards, lost joint feedback latches a fault-stop again after
  each `reset`, and a second `release` is refused. What is left then is SIGTERM and a
  new start of the robot app, with the arm at rest and a hand on it.
- **A `release` after one that worked ends with words that do not fit.** Its line is
  `release refused: the motors are already off; the motors stay on`. The motors are
  off.
- **After the agent gave up on a jaw command no help block follows.** Its STOP makes
  nobody the control owner, so the task's help request is refused: no assistance wait
  and no timed return to rest. The arm holds with its motors on and the jaws at the
  light stiffness; the operator sees the STOP's line at both terminals.
- **A move whose full reads all fail names joint index 0.** When every read of all
  joints fails until `executor.settle_timeout_s` has run out, no joint is known and
  the `not_settled` reason names the first one.
- **`--motion-approval tool` is not built**, and the motors' feedback is not in the
  debug recording (see
  [The options of the experiment's launcher](#the-options-of-the-experiments-launcher)).
  None of those options ran on the real full stack.
- **`run_workstation.sh` has not started tmux anywhere** (see
  [The start scripts](#the-start-scripts)).

From the earlier phases, still open:

- **A `stop` that was begun before a question of the Thor terminal was written and entered
  after it is lost.** Ctrl+C is the STOP that is never dropped.
- **A Thor terminal whose output is stopped with Ctrl+S** holds up the lines the robot app
  program writes itself, such as `robot app: no robot connection: ...`, until the output
  goes on.
- **The line `the start failed after the connect began: the motors may be enabled and
  holding ...`** of a failed start with `--driver real` is written to the terminal only,
  not to the robot app's log.
- **A Ctrl+C in the last instant of the operator console's close can be lost**: after the
  console has ended its stop thread and before the program restores the signal's handler.
- **The console command's `stop` has no raw fallback once it is attached.** It writes a raw
  STOP only when it cannot attach; when the robot link service goes away under its send it
  says so and writes none.
- **Small steps of the Thor's wall clock are not caught.** The live view and the agent's
  camera count a step only above 0.02 s between two ROBOT_STATEs: ten steps of 15 ms each
  pass unseen.

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
  configs/view.yaml           settings of the live view
  configs/human.yaml          settings of the human program
  configs/robot_app_rebot.yaml configs/agent_rebot.yaml configs/link_service_rebot.yaml
  configs/human_rebot.yaml    the same four programs on the rig
  configs/robot_app_mock.yaml configs/agent_mock.yaml
                              demo 1 by hand on one machine, on APAI's software arm
  configs/robot_app_demo2.yaml
                              demo 2 on one machine, on the simulated arm
  configs/robot_app_human_check.yaml
                              the headset checks before power (part F-A of the checklist)
  configs/camera_sender_rebot.yaml configs/camera_receiver_rebot.yaml
                              the camera streamer's two programs on the reBot rig
  run_agent.sh run_camera_receiver.sh run_console.sh run_view.sh
                              start scripts for the workstation
  run_workstation.sh          the workstation's programs of demo 1 on the rig, together
                              in one tmux window
  run_human.sh                start script of the human program: for the Thor on the rig,
                              for the one machine of the recipes
  run_robot_app.sh            start script for the Thor
  setup_workstation_env.sh requirements-workstation.txt
                              the workstation's Python environment for the full stack
  python/isaaccapture_examples/agentic_remote/
    protocol/
      messages.py carrier.py             link messages and the carrier; on both machines
      link_carrier.py local_messages.py  the carrier on a real link; the local API's own messages
      confirm.py                         the typed confirmations; an AI agent's environment
    robot_app/
      interfaces.py clock.py config.py   ports, clocks, settings
      authority.py permits.py            control owner and the three numbers; permits
      executor.py gripper.py             moves; jaw commands
      stop.py waits.py rest.py           STOP and fault-stop; assistance wait; rest and release
      state.py app.py connection.py      the published robot state; all of it wired; the dialer
      human.py                           a person's joint targets and the human drive
      human_socket.py                    the robot app's own socket for the human program
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
      clear_grid.py                      the clear grid of the one-machine recipe
      console_cmd.py                     the console command
    human/
      types.py                           the human program's settings and the types its parts share
      retarget.py ik.py                  the retargeting graph; the inverse kinematics
      logic.py                           buttons, the new-squeeze rule, the ahead checks
      feed.py __main__.py                the controller feeds; the human program
    console/
      console.py status.py __main__.py   the operator console; its lines; its command line
    view/
      model.py server.py                 what the page shows and the view age; the HTTP server
      page.html __main__.py              the page; the view program and its command line
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
with none of `--connect`, `--selftest` and `--selftest-link`, or a check or a typed
confirmation of the real driver that failed.
`--log-level debug` shows the app's log.

## Run demo 1 on one machine

This runs the programs of demo 1 on one machine, with the simulated arm: no robot, no
camera, no GPU and no APAI. It needs a Python 3.11 to 3.13 environment with the example
installed (`uv pip install -e ./examples/agentic_remote`), an Isaac Capture build on the
Python path for the first two programs, and OpenCV for the live view (the `view` extra).
The other way installs nothing: the two variables `PY` and `PYTHONPATH` of
[the next recipe](#run-demo-1-by-hand-with-the-mock-agent) name a Python that has the
packages and put the example's `python` folder and the Isaac Capture build on its path;
`python` in the commands below is then `"$PY"`.
Four terminals, each in `examples/agentic_remote`.

**1. The robot link service.** The shipped file listens on 127.0.0.1, port 47610, and puts
the local API at its default path.

```bash
python -m isaaccapture_examples.agentic_remote.link_service --config configs/link_service.yaml
```

It prints `robot link service: waits for the robot app on 127.0.0.1:47610 ...`.

**2. The robot app**, on the simulated arm. This terminal is the Thor terminal.

```bash
python -m isaaccapture_examples.agentic_remote.robot_app \
    --config configs/robot_app.yaml --arm sim --connect 127.0.0.1:47610
```

It prints `robot app: session ...; dialing the robot link service at 127.0.0.1:47610` and
then `[connection] ...`. Type `help`.

**3. The operator console.**

```bash
./run_console.sh
```

It prints a `WELCOME` line that ends in `driver sim, camera calibration none` and a status
line with `owner nobody #0`. Type `state`: its last line is `takeover: not possible: human
driving is switched off in this robot app's file (human.enabled)`. Type `stop`: the console
answers `stop: taken`
and the Thor terminal shows `[stop] STOP from console: ...`. Type `rest`, then
`path is clear`: the answer is `rest: RESULT completed`.

**4. The live view**, without a camera.

```bash
AGENTIC_REMOTE_VIEW_PYTHON="$(command -v python)" ./run_view.sh \
    --config configs/view.yaml --source none
```

It prints the page's address. Open `http://127.0.0.1:47620/` in a browser on the same
machine: the page shows `NO PICTURE` and the robot's status, and the console's status line
shows `view none`.

**To end**: type `shutdown` at the console and answer `yes` (the arm is at rest), or type
`quit` at the Thor terminal; the robot app ends with exit code 0. Then `quit` at the console
and Ctrl+C in the terminals of the view and the service.

The tests run these four programs as processes, the operator console on a pseudo-terminal
(`test_the_programs_of_demo_1_start_as_processes_and_end_cleanly`). What differs from the
steps above: a port the system picks, a socket in a temporary folder, files of the test for
the service and the view, and the console and the view started with `python -m` and not
through their start scripts. The test types `rest` and `path is clear` at the console; its
`state` and `stop` go through the console command. With the shipped port 47610 and the
default socket path they were not run, and the page was not opened in a browser. For two
machines, put the workstation's own address in `listen.address`, the Thor's in
`allowed_peers`, and give the robot app the workstation's address.

**With a task of the agent**, the agent program needs more than the four programs above:
the robot app on APAI's software arm with a file whose limit sets fit the profile's speed
cap, an agent file for APAI's mock scene, and an obstacle grid. The next section is that
recipe.

The tests run the agent program inside the test process on APAI's mock scene, with a
stand-in for the obstacle grid bridge and the typed lines scripted, together with the robot
link service, the console and the view (`test_demo1_programs.py`; see
[What was run](#what-was-run)).

Which of these commands were run, and how, is in [What was run](#what-was-run).

## Run demo 1 by hand with the mock agent

This runs a task of the agent on one machine: the robot app on APAI's software arm, the
agent program on APAI's mock scene with its mock camera. No robot, no camera, no GPU and
no model. Two files hold what the shipped files lack for it:

- `configs/robot_app_mock.yaml`, for `--arm apai --driver mock`: joint boxes from APAI's
  arm model and the agent's and the human's speed at the mock profile's cap of 1.2 rad/s
  (the shipped `robot_app.yaml` has a human speed above that cap). It also has
  `human.enabled: true` and `human.require_live_view: false`, so that the same robot app
  serves demo 2, and `executor.max_setpoints: 3300`.
- `configs/agent_mock.yaml`: `experiment.stack: minimal`, `camera.allow_invented_times:
  true` and `experiment.grid: clear`. The **clear grid** stands in for the obstacle grid
  bridge, which the minimal stack does not start: it answers APAI's obstacle map with
  "nothing near". The map is still fed by the camera's frames, the check ahead still
  checks every move on it and every setpoint still goes out under a PERMIT. Only the
  grid's answer is empty: **no move is checked against obstacles**. The program takes
  `clear` only with the minimal stack and only against a robot app that declares a
  simulated driver, and says so in a `WARNING` at its start.

Nothing needs to be installed. In every terminal:

```bash
W=<the checkout of Isaac Capture>
export PY=$W/tests/python/examples/agentic_remote/.venv/bin/python
export BUILD=$W/build/python_package/Release
export PYTHONPATH=$W/examples/agentic_remote/python:$BUILD
export AGENTIC_REMOTE_APAI_SRC=<the checkout of APAI>/src
export AGENTIC_REMOTE_ISAAC_CAPTURE_PY=$BUILD
export PYTHONDONTWRITEBYTECODE=1
cd $W/examples/agentic_remote
```

`PY` is the Python of the test folder's environment with its `dev` and `apai` extras (see
[Run the tests](#run-the-tests)): a Python 3.12 with numpy, PyYAML, OpenCV and what APAI
imports. `BUILD` is the Isaac Capture build for that Python. Five terminals:

| # | Terminal | Command | Must show |
|---|---|---|---|
| 1 | service | `"$PY" -m isaaccapture_examples.agentic_remote.link_service --config configs/link_service.yaml` | `robot link service: waits for the robot app on 127.0.0.1:47610 (taken from 127.0.0.1); local API at <socket path>` |
| 2 | Thor terminal | `"$PY" -m isaaccapture_examples.agentic_remote.robot_app --config configs/robot_app_mock.yaml --arm apai --driver mock --connect 127.0.0.1:47610` | `robot app: session <id>; dialing the robot link service at 127.0.0.1:47610`, then `[connection] robot connection made: the arm holds as it was; a new connection starts no motion` |
| 3 | console | `AGENTIC_REMOTE_CONSOLE_PYTHON="$PY" ./run_console.sh` | `WELCOME: robot apai-mock-arm, profile apai-rs-model, session <id>, driver apai-mock, camera calibration apai-mock-calibration`; a status line with `driver apai-mock`, `owner nobody #0`, `rest_hold`, `jaws empty`, `torque on`, `view none` and `no human program` |
| 4 | view | `AGENTIC_REMOTE_VIEW_PYTHON="$PY" ./run_view.sh --config configs/view.yaml --source none`; open `http://127.0.0.1:47620/` | `live view: open http://127.0.0.1:47620/ in a browser on this machine; camera pictures: none; the page shows the status only`. The page shows `NO PICTURE` |
| 5 | agent | `APAI_REQUIRE_CUDA=0 AGENTIC_REMOTE_AGENT_PYTHON="$PY" ./run_agent.sh --config configs/agent_mock.yaml --camera mock --interactive` | `[agent] run folder ...`; two `[agent] WARNING:` lines, first that of the invented exposure times (`the camera frames of --camera mock carry invented capture times ...`), then `experiment.grid is clear: ...`; `[agent] robot connection up: robot apai-mock-arm, profile apai-rs-model, driver apai-mock, ...`; `[agent] obstacle grid: live (... s old); refreshes 1`; `[agent] ready. Ctrl+C at this terminal is STOP; ...`; the prompt `task> ` |
| 6 | agent | Type `open the gripper`. The question ends in `Type yes to proceed:`; type `yes` | Agent: `[agent] the agent owns the arm (owner number 1)`, `[agent] task completed: the arm holds where it is and nobody owns it; ...` and a task report with `success: True`. Console: `[owner] new task: the agent owns the arm, owner number 1`, `[owner] the agent gave the arm back: task completed; the arm holds and nobody owns it` and a status line with `owner nobody #2` |
| 7 | agent, console | Type `pick up and place the red object`, then `yes` | Console: `[owner] new task: the agent owns the arm, owner number 3`; eight moves, with status lines that show `owner agent #3`, the move's progress as in `moving 47/104 sending` and the held permit as in `permit 68, evidence 0.34 s`; `jaws loaded` during the carry; then the help request (below) |
| 8 | console | `stop`. Then `rest`, then `path is clear` | `[stop] STOP from console: stop typed at the operator console; the arm holds, stop number 1; the assistance wait is cancelled` and `stop: taken. ...`. Then `[rest] return to rest: the arm moves for 25.0 s (largest joint travel 1.20 rad)`, `[rest] return to rest: the arm is at rest, motors on` and `rest: RESULT completed {"outcome": "at_rest"}` |
| 9 | all | Ctrl+D at the agent's `task> `. `shutdown`, then `yes`, at the console; `quit`. Ctrl+C in terminals 1 and 4 | Agent: `[agent] 1 task(s) of this session ended without success`, exit code 1. Console: `[release] torque released at rest: the motors are off`, `[shutdown] shutdown: the motors are off at rest; the robot app closes`, `shutdown: RESULT completed {"outcome": "at_rest"}` and `ROBOT CONNECTION LOST: ...`. Exit code 0 for the robot app, the console, the service (`robot link service: SIGINT; closing. The robot holds on its own.`) and the view (`live view: SIGINT; closing. The robot is not touched.`) |

**The pick and place of step 7 ends in a help request.** The arm runs the mock scene's
eight moves, each under a permit. The mock scene confirms neither the grasp nor the place,
so the task ends without success and the console prints `[help_needed] HELP NEEDED: help
needed: task ended without success: ...; the arm holds and nobody owns it` and the help
block: `jaws: empty`, `time left: 120 s; then the arm keeps holding`, the words `stop` and
`rest`, and as its last line `takeover: not possible: no human program is attached: start
run_human.sh`. A status line follows with `owner nobody #4`, `hold` and `wait 119 s`. The
arm holds at about [0, 1.2, 1.2, 0, 0, 0] rad with its motors on. This is where demo 2
begins.

The reason on the console's `HELP NEEDED` lines is cut at 256 characters and ends inside
the run folder's name. The agent terminal's task report gives it in full: `[verification]
could not confirm: <run folder>/arm_profile:pick_and_place#0 unverified: red object matches
the drop point the skill itself reported, but only in the belief channel it also wrote --
no independent confirmation`. From the second run on the same state folder it ends
`refuted: red object is still within 0.0 cm of where it started`, because APAI then recalls
the objects of the run before.

By hand the agent program writes into `examples/agentic_remote/.state`: the run folders,
APAI's stores and caches, and a `.gitignore` that holds `*`.

Every "must show" above is what the programs printed in a test that ran these commands as
processes, with the differences [What was run](#what-was-run) lists. A person has not run
them.

## Run demo 2 on one machine

A synthetic controller takes the simulated arm over, lifts it and gives it back. No GPU, no
model, no camera, no headset and no arm. Every "must show" below is what the programs
printed in a test that ran these commands as processes
(`test_the_recipe_of_demo_2_runs_as_processes_with_the_shipped_files` in
`test_demo2_programs.py`), with these differences: the service, the view and the human
program ran on copies of their files that name a port the system picked, a socket and a
state folder of the test and, for the Teleop Session Server, a port nobody listens on; the
console and the view were started with `python -m`, not by their scripts; a local client
of the test in the role `tool` read ROBOT_STATE, so the console counted `1 tool` too; the
view's page was asked over HTTP, not opened in a browser. A person has not run them.

The environment lines of [the recipe before](#run-demo-1-by-hand-with-the-mock-agent) in
every terminal. The robot app runs on `configs/robot_app_demo2.yaml`: `robot_app.yaml` with
`human.enabled: true`, `human.require_live_view: false` and a human joint box of -2.5 to
2.5 rad, which lies inside `rest.region_*`. The robot app takes that file only with a
simulated driver. The human program runs on `configs/human.yaml`.

| # | Terminal | Command | Must show |
|---|---|---|---|
| 1 | service | `"$PY" -m isaaccapture_examples.agentic_remote.link_service --config configs/link_service.yaml` | `robot link service: waits for the robot app on 127.0.0.1:47610 (taken from 127.0.0.1); local API at <socket path>` |
| 2 | Thor terminal | `"$PY" -m isaaccapture_examples.agentic_remote.robot_app --config configs/robot_app_demo2.yaml --arm sim --connect 127.0.0.1:47610` | `robot app: session <id>; dialing the robot link service at 127.0.0.1:47610`, then `[connection] robot connection made: the arm holds as it was; a new connection starts no motion` |
| 3 | console | `AGENTIC_REMOTE_CONSOLE_PYTHON="$PY" ./run_console.sh`; type `state` | `WELCOME: robot sim-arm, profile sim-6dof, session <id>, driver sim, camera calibration none`; a status line with `driver sim`, `owner nobody #0`, `stop none #0`, `rest_hold`, `jaws empty`, `torque on`, `view none` and `no human program`. `state` prints the service's status, `ROBOT_STATE: control owner nobody (owner number 0), stop number 0, stop level none, motion rest_hold, jaws empty, torque on, assistance wait none`, a status line and, last, `takeover: not possible: no human program is attached: start run_human.sh` |
| 4 | view | `AGENTIC_REMOTE_VIEW_PYTHON="$PY" ./run_view.sh --config configs/view.yaml --source none`; open `http://127.0.0.1:47620/` | `live view: open http://127.0.0.1:47620/ in a browser on this machine; camera pictures: none; the page shows the status only`. The page shows `NO PICTURE`; the console's status line counts `1 viewer` and shows `view none`. This robot app does not require the view |
| 5 | human program | `AGENTIC_REMOTE_HUMAN_PYTHON="$PY" bash run_human.sh --config configs/human.yaml --controller synthetic:lift` | `human program: synthetic controller "lift"; driver sim; arm model mock, 6 joints`, then `human program: right controller seen` and a report line each second that begins `human program: stage ready` and holds `owner nobody; engaged no; hand x 0.000 y 1.200 z -0.300 m`. Console: a status line that counts `1 human` and shows `headset ready`, `controller 0.03 s` (the age differs from line to line) and `note: hold right B for 1 s with the grip released to take over` |
| 6 | console, watch | - | Human program: `human program: B down`, then `B up`, `grip down` and `grip up`. Console: `[stop] STOP from controller: right B; the arm holds, stop number 1` and a status line with `owner nobody #1` and `stop stop #1`; two bells and `[owner] TAKEOVER: a person owns the arm, owner number 2; release the grip, then squeeze it to move`; a status line with `owner human #2`, `stop none #1`, `takeover open` and `needs release`; one with `note: squeeze the grip to move`; one with `moving N/0`, `human driving` and `takeover open` (N is the number of setpoints the drive had sent when the line was printed; 0 in the test's run); then one with `hold`, `takeover open` and `note: squeeze the grip to move`. The gripper tip is 5 cm above where it is at rest |
| 7 | console | `rest`, then `path is clear` | `rest: RESULT accepted`; `[rest] return to rest: the arm moves for 5.0 s (largest joint travel 0.24 rad)`; a status line with `owner recovery #3`, `takeover open`, the return's progress, as in `moving 27/250 sending`, and `note: the arm returns to rest; right B is a STOP`; `[rest] return to rest: the arm is at rest, motors on`; `rest: RESULT completed {"outcome": "at_rest"}`; a status line with `owner human #4`, `rest_hold` and `note: at rest: hold right A for 1 s to give the arm back`. Then, by the script's right A (`human program: A down`, then `A up`): two bells and `[owner] takeover ended: the arm is given back; nobody owns it`, and a status line with `owner nobody #5` and `rest_hold` |
| 8 | console | `shutdown`, then `yes`; `quit`; Ctrl+C in terminals 1, 4 and 5 | Console: `shutdown: RESULT accepted`, `[release] torque released at rest: the motors are off`, `[shutdown] shutdown: the motors are off at rest; the robot app closes`, `shutdown: RESULT completed {"outcome": "at_rest"}`, a bell and `ROBOT CONNECTION LOST: receive failed: connection closed; the robot holds on its own`; after `quit`, `the operator console ends; nothing was sent for that`. Service: `robot link service: SIGINT; closing. The robot holds on its own.` View: `live view: SIGINT; closing. The robot is not touched.` The human program prints nothing at its end. Exit code 0 for all five programs |

The owner number goes up at every change of control owner and at the STOP: nobody #0, the
STOP's nobody #1, human #2, recovery #3 for the return to rest, human #4 and, after the
hand-back, nobody #5.

The script `lift` presses right B 3 s after the human program's first pass, holds it until
the person owns the arm, squeezes the grip, raises the hand 5 cm in 4 s and releases. It
then waits for the arm to hold at rest, which is the typed `rest` of step 7, and holds
right A. From the folded rest pose a lift is the motion the IK can always solve: from
rest, lift the gripper first.

**Wait for the status line before the next command.** The console and the agent program
quote the newest ROBOT_STATE they have, and a command typed on one that is a step old is
refused and changes nothing: type it again. After `rest: RESULT completed` wait for the
status line with `owner human` and `rest_hold` before typing `handback` (the typed
fallback for right A), and after `takeover ended` for the one with `owner nobody` before
the next task. In the tests a `handback` typed sooner got `handback: RESULT refused: owner
number 3 does not match the current 4`, and a task typed at the agent terminal within
about 50 ms of the hand-back got `takeover not finished: give the arm back with right A
at rest; no command sent`.

**With the agent failing first.** Terminal 2 runs the robot app of the recipe before
(`--config configs/robot_app_mock.yaml --arm apai --driver mock`), and a sixth terminal runs
the agent program as there (step 5 of that recipe); the human program is started later.
The console's `WELCOME` then says `driver apai-mock`. The task `pick up and place the red
object`, typed as the first task with `yes`, moves the arm under owner number 1 and ends
in the help request. The console shows the `HELP NEEDED` lines and the help block with
`jaws: empty`, `time left: 120 s; then the arm keeps holding` and, as its last line,
`takeover: not possible: no human program is attached: start run_human.sh`; then
`[countdown] assistance wait: 120 s left, then the arm keeps holding (...)` and a status
line with `owner nobody #2`, `hold`, `wait 119 s` and `no human program`. Then start
terminal 5, which prints `human program: synthetic controller "lift"; driver apai-mock;
arm model mock, 6 joints`, and go on with steps 6 and 7. What differs from the table:

- The STOP's line ends `; the assistance wait is cancelled`, and its status line shows
  `owner nobody #3` and `hold`.
- The takeover has owner number 4. The gripper tip rises 5 cm from where the agent left
  the arm, at about [0, 1.2, 1.2, 0, 0, 0] rad.
- The return to rest is longer: `[rest] return to rest: the arm moves for 30.9 s (largest
  joint travel 1.48 rad)`, with `owner recovery #5` and progress as in `moving 30/1543
  sending`. Then `owner human #6` and, after the hand-back, `owner nobody #7`.
- No `open` is needed: the jaws are `empty` at the help request and at the hand-back.

After the hand-back the task `open the gripper`, with `yes`, runs. The console prints
`[owner] new task: the agent owns the arm, owner number 8` and `[owner] the agent gave the
arm back: task completed; the arm holds and nobody owns it`, and the agent's task report
says `success: True`. No program was restarted: the console printed one `WELCOME`. Ctrl+D
at `task> ` ends the agent program with `[agent] 1 task(s) of this session ended without
success` and exit code 1; step 8 ends the other programs with exit code 0.

This variant ran as processes in the same test file
(`test_demo_2_with_the_agent_failing_first_runs_as_processes`), with the same differences
and with the agent program on a copy of `configs/agent_mock.yaml` that names the test's
socket and state folder; a person has not run it.

**With the human program at the robot app's own socket**, as it runs on the Thor. Start
the robot app of terminal 2 with `--human-socket` and a path added, and the human program
of terminal 5 on a copy of `configs/human.yaml` whose `socket_path` is that path. The
Thor terminal prints `robot app: the human program's socket is` and the path after its
session line. With the human program attached, the console's status line shows `human
program at the robot app` in place of `no human program`, and its `clients` part counts
no human. Right B, the takeover, the lift, the typed `rest` and right A go as in steps 6
and 7. End the human program with Ctrl+C before the shutdown of step 8: it ends with
exit code 0, and the console shows `no human program` again. A test ran this as
processes, without the view, on copies of the two files
(`test_the_human_program_on_the_thor_runs_as_processes` in `test_human_on_thor.py`). No
person has run it.

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

No test needs a robot or a second machine, and nearly all run on a fake clock;
`test_main.py` runs the self-test on the machine's clock and takes some seconds. Only the
tests marked `gpu` need a GPU. The `dev` extra holds OpenCV as well: without it
`test_view_program.py` and `test_demo1_programs.py` are skipped as a whole.

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
`test_apai_contract.py` (the names of APAI the agent side relies on),
`test_agent_options.py`, `test_gripper_timeout.py` and
`test_agent_end_to_end.py`. Where they drive the robot app with the real permit loop, the robot
app runs on the machine's clock and a move takes its real length.

Four files hold what the robot app and the agent program do at four edges:

- `test_freshness.py`: every stretch of time counts toward the age of controller data
  and of the live view: the time at the human program, the way to the robot app, on
  the robot connection and at the robot app's own socket, and the wait at the robot
  app. Its one test that runs the human program is marked `link` and `apai`.
- `test_release_failed.py`: a torque release whose driver call raises, on the simulated
  arm and, marked `apai`, on APAI's mock driver: the motion `release_failed`, what is
  refused after it, the second release, the shutdown.
- `test_gripper_timeout.py`: a jaw command the arm adapter gave up on, with the real
  robot app on the simulated arm (marked `apai`), and the expiry of a GRIPPER on the
  robot app.
- `test_agent_options.py`: the arguments of the experiment's launcher in the agent
  program: what a run takes, what is refused, the order of the recordings around the
  robot connection. Its last test, marked `apai` and `experiment`, runs the
  experiment's own modules on APAI's mock scene.

The executor's rules for a MOVE's expiry at its first setpoint and for the read a
`completed` move ends on are tested in `test_executor.py`.

The human path's tests are `test_takeover.py`, `test_human_drive.py`, `test_human_jaws.py`,
`test_lift_out.py`, `test_human_on_thor.py`, `test_human_logic.py`, `test_human_ik.py`,
`test_human_retarget.py`, `test_human_feed.py`, `test_human_program.py` and
`test_demo2_programs.py`.

- `test_human_jaws.py`: the jaws under a person's trigger on the robot app, with the
  human targets the tests make: the grip level, the trigger that has and has not
  reached it, the limit on closing, the light hold, what ends the jaw motion, the two
  settings. One test, marked `apai`, runs on APAI's software arm.
- `test_lift_out.py`: a takeover with a low gripper tip and the rule of the lift, on
  APAI's checkers and its software arm. Without APAI all its tests but one are skipped.
- `test_human_on_thor.py`: the human program attached at the robot app's own socket
  (`--human-socket`), as on the Thor: WELCOME, ROBOT_STATE and the RESULTs there, what
  the socket refuses, right B's STOP with the robot connection down, a client that
  does not read, one human program at a time. Its last two tests, marked `link` and
  `apai`, start the programs as processes and take about 30 s.

Those that load Isaac Capture's retargeters
(`test_human_retarget.py`, `test_human_feed.py`, `test_human_program.py` and
`test_demo2_programs.py`) are skipped as a whole without the build on the Python path, by
no marker. `test_human_ik.py` is marked `apai`, and `test_human_program.py` `link` and
`apai`. `test_demo2_programs.py` runs the programs of demo 2 together, as
`test_demo1_programs.py` does for demo 1; two of its tests run the recipe of
[Run demo 2 on one machine](#run-demo-2-on-one-machine) and its variant as processes,
and one runs the trigger and the jaws under a person. It has 18 tests and takes about
4 min with everything present. The human path on APAI's
software arm and on the rig's robot file is tested in `test_apai_app.py` and
`test_rig_configs.py`.

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

`test_demo1_programs.py` runs the programs of demo 1 together, with the same variables. The
robot link service listens on 127.0.0.1 and a port the system picks, and the robot app dials
it over a real link. Attached at once are the agent program (in the test process, on APAI's
mock scene), the operator console (the class the console program runs, its lines typed by
the test) and the live view (the view program's `main()` on a thread, with stand-in camera
pictures or with `--source none`). Its tests:

| Test | Shows |
|---|---|
| `test_demo_1_on_the_mock_arm_with_console_and_view` | Two tasks run; every joint target was under a permit; the console printed each owner change; the view age reached ROBOT_STATE and the console; the page showed the move |
| `test_a_console_stop_during_the_agents_move_ends_the_task_and_the_next_task_runs` | `stop` typed at the console during a move ends it; the next task runs without a restart |
| `test_a_console_rest_after_the_task_returns_the_arm` | `rest` with `path is clear` returns the arm; `yes` sends nothing |
| `test_an_obstacle_costs_no_restart` | An obstacle ends the first task with a help request; the same program runs the second task |
| `test_the_agent_program_gone_shows_the_help_request_and_the_countdown_on_the_console` | The assistance wait on the console; a typed `stop` ends it; nothing moves |
| `test_the_console_and_the_view_survive_a_restarted_robot_app` | A new session id is shown; a command with the old numbers is refused |
| `test_the_programs_of_demo_1_start_as_processes_and_end_cleanly` | The service, the robot app (`--arm sim`, and `--arm apai --driver mock`), the view, the operator console (on a pseudo-terminal) and the console command as processes of their own |

The file takes about 65 s. The view's page is asked over HTTP, a few times per second at
most; no test starts a browser.

## The rule: torque is never cut away from rest

The arm has no brakes: with the motors off it falls, unless it lies at rest. So the robot app
switches the motors off through one call, the arm port's `release_torque()`, and makes that call
in one place, the guarded release in `robot_app/rest.py`. With APAI the call is
`ApaiArm.release_torque()`, which disconnects APAI's driver: the second and last place in the
package that cuts torque. The guarded release runs only on an operator's request (`release`, or
the shutdown that `quit` asks for), and only when two full joint reads 0.2 s apart are within
0.05 rad of rest and of each other, the jaws are proven empty, no move is running and no
fault-stop is latched.

**A release whose driver call fails.** If the driver's disconnect raises an error, the
robot app does not know whether the motors are off: the driver may have switched some or
all of them off before it failed. The motion state becomes `release_failed`, ROBOT_STATE
keeps `torque_on` true, and a fault-stop latches with the reason `torque release failed,
the motors may be on or off: the driver's disconnect failed: <the driver's error>`. The
console shows `torque unknown`. Treat the arm as powered. From then on nothing moves it:
a task, a MOVE, a jaw command, `open`, a takeover and a return to rest are refused with
`a torque release failed and the motors may be on or off; type release to try it again,
or restart the robot app`, and with APAI the arm port sends the driver no target and no
jaw command. Type `reset`, then `release`: the release is tried again under the same
conditions (two joint reads at rest, the jaws proven empty). If it works the state is
`rest_unpowered` and the lines are those of any release; if it is refused the line ends
`; the motors may be on or off`. `quit` and `shutdown` start no return to rest in this
state and ask for no `path is clear`; they try the release again and close the robot app
only if it worked. Otherwise the app keeps running: `shutdown failed: a torque release
failed and the motors may be on or off; release refused: <why>; the robot app keeps
running`, or a new fault-stop when the driver's call fails again. SIGTERM still closes
the program and switches nothing; its log line is then `robot app closed; torque on: not
known, a torque release failed: the motors may be on or off`. An arm port that knows it
switched its motors off before the call raised, as the simulated arm does, is shown as
`rest_unpowered`. Checked on the simulated arm and on APAI's mock driver
(`test_release_failed.py`), not on the arm.

One thing outside the rule: connecting APAI's driver for the real arm cuts torque (its
`connect()` clears motor faults with a command that does so). The robot app connects once, when
it starts, and never again: a lost or a new robot connection does not touch the arm port. The
arm has to lie at rest and be supported at that moment. A typed confirmation is not enough
for that: before anything that cuts torque, APAI's driver reads every joint twice and
refuses an arm it does not measure at rest, and the robot app refuses to start while a hold
marker or an operator confirmation for APAI's connect is on file (see
[The real driver](#the-real-driver)). During the connect the robot app's process cannot
claim such a confirmation at all, whenever it was written. This connect was run only against a stand-in for the
motor library, never on the arm.

Everything else leaves the motors on: STOP, a fault-stop, a refused or interrupted return to
rest, the end of the assistance wait, a takeover, a hand-back, every end of a human drive,
a lost or a new robot connection, a restarted robot link
service, Ctrl+C, SIGTERM, a closed terminal and closing the robot app. Tests keep it that way:
the simulated arm records the pose at every torque
release and a test fixture fails any test in which one happened away from rest, and
`test_wiring.py` and `test_no_limp.py` scan the source for any other place that could cut
torque. This is a rule about the code; sections 9 and 10 of the design list what software
cannot prevent.

The agent side has no way to cut torque. The same scan reads every file under `agent/` and
finds no such call. APAI calls `disconnect()` and `stop()` on its arm when it shuts down and
after a failed motion; on the arm adapter the first sends nothing and the second at most a
STOP, which is a hold. `test_no_limp.py` ends an agent move in each way the agent side knows
(a setpoint refused by the check ahead, a latched workstation checker, Ctrl+C, a STOP at the
Thor terminal, a lost link, the agent program gone) and each time finds the arm holding away
from rest with its motors on.

The operator's programs cannot cut torque either. RELEASE and SHUTDOWN are sent from one
function, `send_command()` of the console command, which the operator console calls too,
after the typed confirmation; the robot app still applies its own checks to them. The scan
reads every file under `console/` and `view/`: none cuts torque and none names RELEASE or
SHUTDOWN. The live view sends VIEW_STATUS only.

The human path cannot cut torque either. The human program sends HUMAN_SETPOINT, STOP and
its two OWNER_REQUESTs, and neither the robot link service nor the robot app's own socket
lets its role send anything else. On the
Thor a human drive sends setpoints from one place, inside the claim a move runs in, and
every end of it leaves the last setpoint standing: a hold with the motors on. A takeover
and a hand-back change the control owner and send nothing to the motors. The scan reads
`robot_app/human.py` and every file under `human/` and finds no torque cut in them and no
RELEASE or SHUTDOWN. `test_no_limp.py` ends a human drive of the raised arm in each way it
can end and finds the motors on each time
(`test_no_ending_of_a_human_drive_cuts_torque_away_from_rest`); a release asked for while
a drive runs is refused and cuts nothing; and its sweep of every command in every state
has the states of a person who owns the arm among them.

`test_no_limp.py` also tries every command in every state away from rest. Among the states
is a robot app that started with the arm raised and the jaws not referenced; among the
commands are the lines typed at the Thor terminal, with `rest` and `quit` confirmed by
`path is clear`, the typed `reference`, and a live view that reports and goes away. Torque
is cut only by a shutdown that first brought the arm to measured rest.

## First run on the rig

This is the operator's checklist for the first session on the real arm. **None of it was
run.** It follows the programs as they are built; where a line a program prints differs
from what is written here, stop and find out why.

### Rules for the whole session

- **Two people.** The operator sits at the Thor terminal. A second person supports the arm
  where a step says so and stands at the 48 V switch of the motors.
- **Part F, demo 2, needs its own people**, because the person who wears the headset
  cannot sit at the Thor terminal and cannot read:
  - **Wearer**: wears the headset, holds the right controller, stands behind the arm's
    base and outside its reach, types nothing and never touches the arm.
  - **Stop person**: sits at the Thor terminal for all of F-B, with `stop` typed and a
    finger on Enter during every squeeze; types `rest` and `path is clear` there; says
    aloud each line that matters to the wearer ("yours", "refused: why", "drive ended:
    why", "at rest").
  - **Workstation person**: watches the live view page and the console, types at the agent
    terminal and does the workstation's actions. The stop person may do this only if both
    keyboards are within one arm's length.
  - **Supporter**: stands at the 48 V switch and supports the arm from the side, outside
    the path of the gripper, in every step of F-B until steps 21 to 26 and F16 have passed.

  F-B needs the wearer and the stop person at the least, and the supporter whenever steps
  21 to 26 did not pass before it (the variant F-B0). With fewer people F-B is not run.
- **No step cuts torque away from measured rest.** Torque is cut in four steps only: 15 and
  18 (the connect of the real driver), 17 (`release`) and 28 (the shutdown). In each the arm
  lies at rest, and the program measures that before it cuts. No step of part F cuts torque.
- **To abort, at any step**: type `stop` at the Thor terminal or press Ctrl+C there. It
  needs no network. The arm holds with its motors on. Then decide in peace. Ctrl+C is the
  one that is never dropped: a `stop` that was begun before a question of the Thor terminal
  was written and entered after it is lost (see "What the operator types to confirm").
  Two moments of
  step 15 differ. At one of the three questions Ctrl+C ends the program and nothing was
  connected. While the robot app starts (the connect and the jaw reference, with the arm at
  rest) Ctrl+C does not end the start half-way: it is a STOP as soon as the start is over.
  A terminal that closes does not end the start or the robot app either.
- **To abort in part F-B, in every step**: the stop person presses Enter on the typed
  `stop` (or Ctrl+C) at the Thor terminal and says "stop"; the wearer releases the grip.
  That needs no network and no headset. Right B is the wearer's own stop and is under test
  itself: it is not the abort. Before any work in the headset's own menus, and before a
  recentre of the headset view while a person owns the arm: right B first.
- **The 48 V switch drops the arm.** Use it only while the arm lies at rest or a person
  holds it. It is not the way to abort.
- **Never restart the robot app with the arm raised**, and never end it with a kill signal
  while the arm is raised. While the robot app is down nothing can stop the arm or return
  it, and its next start is refused as long as the arm is not at rest. Bring the arm to rest
  first (`rest`, then `path is clear`).
- **Never use APAI's own `holding the arm` confirmation** unless a person holds the arm. The
  robot app refuses to start while any `*.confirm.json` file lies in `~/.apai/arm_holds`,
  and its process cannot claim one that appears later: its connect of a raised arm is
  refused with such a file on hand as without one.
- Write every value the table asks for into the session record, with the commit id of the
  code on the Thor.

**Abort the session, not only the step, when**: the arm moves when no command asked for it;
the arm does not hold after a `stop`; the motors switch off away from rest; a step's "must
show" does not appear and the reason is not understood; the Thor's clock is set by hand or
steps; a hold marker appears. In part F also when: the arm moves with the grip released;
it moves after any hold without a new squeeze; it does not hold after the Thor terminal's
stop.

### Before the session day

- The tests pass on the workstation (see [Run the tests](#run-the-tests)).
- The workstation environment is built
  (see [The workstation environment](#the-workstation-environment)).
- The code is on the Thor. That is the operator's step: copy the worktree over the
  operator's own ssh session, or make a `git bundle` and fetch from it on the Thor. Write
  the commit id into the session record.
- Isaac Capture is built on the Thor for the Python of APAI's environment there
  (`-DISAAC_TELEOP_PYTHON_VERSION=3.12`). Nobody has made that build; its time is not known.
- The camera sender is built on the Thor.
- On the workstation, the variables of `run_agent.sh` and `run_view.sh` are set. On the
  Thor, those of `run_robot_app.sh`.

For part F, demo 2, in addition:

- The two one-machine recipes ran by hand
  ([Run demo 1 by hand with the mock agent](#run-demo-1-by-hand-with-the-mock-agent),
  [Run demo 2 on one machine](#run-demo-2-on-one-machine)), the second with the page open
  in a browser: the picture area shows `NO PICTURE`, `HUMAN DRIVING` shows while the script
  squeezes, and a second tab of the page turns the first into `THIS PAGE WAS REPLACED:
  reload it`. This is the first run of the page's script.
- The bells ring apart at the real console terminal. With a robot link service whose file
  has `operator.typed_takeover: true` and the robot app on the simulated arm with
  `configs/robot_app.yaml` (human driving off): type `takeover`, then `yes`. The line
  `[refused] owner request 'human' refused: ...` must ring three separate times. If the
  three merge into one, the wearer cannot tell a refusal from a takeover by ear: the stop
  person says each of them aloud.
- The owner has decided the settings of
  [The settings that ship off](#the-settings-that-ship-off): `human.enabled`,
  `operator.typed_takeover`, the human speeds and `human.view_age_limit_s`.
- `SRVPY` is a Python that can run CloudXR's service and the Teleop Session Server: the
  environment of the owner's tested headset recipe, with the Isaac Capture build in front
  on `PYTHONPATH`. The workstation environment and the test folder's `.venv` cannot: they
  lack `websockets` and `cryptography`, which the CloudXR service needs
  (`requirements-cloudxr.txt` of the build).
- The human program runs on the Thor. There `AGENTIC_REMOTE_HUMAN_PYTHON` names the
  Python that `AGENTIC_REMOTE_THOR_PYTHON` names, which also needs PyYAML and pinocchio,
  and the other two variables of `run_human.sh` are those of `run_robot_app.sh`. The
  Isaac Capture build on the Thor holds the retargeters and the remote session. Nobody
  has started the human program on a Thor.
- `teleop.endpoint` of `EX/configs/human_rebot.yaml` on the Thor holds `WS:47600` in
  place of its placeholder.
- `cat /sys/bus/usb/devices/usb*/power/control` prints `on` on every line before the
  headset's USB cable is plugged in. The owner's test log calls this required on this
  workstation; on 2026-11-15 every line read `auto`. Setting it needs sudo.
- The workstation's firewall lets the Thor reach port 47610, the robot connection's, before
  step 10, and port 47600, the Teleop Session Server's, before step F2: the human
  program on the Thor dials that server across the wired link.
- The headset is the Quest on its USB-C cable, as in the owner's tested recipe.

In the tables, `EX` is `examples/agentic_remote`, `PROFILE` is the rig's APAI arm profile
(the path of `configs/thor_operation` in the experiment folder on the Thor, without
`.yaml`) and `WS` is the workstation's address on the wired link. In part F, `BUILD` is
the Isaac Capture build folder on the workstation. In part F the Thor has two terminals:
the Thor terminal, in which the robot app runs, and a second one for the human program.
The commands of part F for the Thor follow the programs as they are built; none of them
was run on hardware.

Steps 7, 8, 11, 12 and 20 start the workstation's programs one by one. `bash
EX/run_workstation.sh --env FILE --thor THOR` starts the same five together, each in a
pane of one tmux window, with the agent's command of step 20 waiting for Enter in its
pane (see [The start scripts](#the-start-scripts)).

### A. Before power: no arm motion

| # | Where | Do | Must show | Write down | Abort if |
|---|---|---|---|---|---|
| 1 | Workstation | `nvidia-smi --query-gpu=index,pci.bus_id --format=csv` | Index 0 is the card the camera receiver uses (`00000000:C1:00.0` on this workstation) | The output | Another card is index 0 |
| 2 | Both | Connect the two machines by wire. Write the Thor's address into `allowed_peers` of `EX/configs/link_service_rebot.yaml` | Each machine reaches the other | Both addresses, the link speed | No wired link |
| 3 | Thor | `timedatectl` | - | Whether the clock is synchronized, and by what | The clock is being set by hand |
| 4 | Thor, by eye | 48 V off. The arm lies folded on its stops. The jaws are empty. Nothing is in the arm's reach | - | A photo | The arm is not at rest |
| 5 | Thor | Bring the motor bus up with the experiment's script (it needs sudo and sends no motor command) | The bus is `can4` at 1 Mbit/s | - | The script stops |
| 6 | Thor | `ls -la ~/.apai/arm_holds` | No hold marker for `can4` (`can4.json`) and no `*.confirm.json` file at all | - | A marker exists: do APAI's operator recovery first. A `*.confirm.json` file exists: revoke it with APAI's `scripts/robots/arm_hold.py revoke`. The folder cannot be listed (`Permission denied`): let its owner read and enter it; a revoke cannot list it either |
| 7 | Workstation | `PYTHONPATH=EX/python:BUILD/python_package/Release python -m isaaccapture_examples.agentic_remote.link_service --config EX/configs/link_service_rebot.yaml`. `BUILD` is the Isaac Capture build folder on the workstation; `python` is a Python of the version that build was made for, with numpy and PyYAML: Isaac Capture's package imports numpy, and the service reads its file with PyYAML. The Python of the workstation environment is one | `robot link service: waits for the robot app on ...:47610` | - | Exit code 1 or 2. Exit code 2 with `cannot be imported ...: No module named 'numpy'`: that Python lacks numpy |
| 8 | Workstation | `EX/run_console.sh` | A status line with `robot connection lost (not connected)` | - | It cannot attach |
| 9 | Thor | Describe only: `EX/run_robot_app.sh --config EX/configs/robot_app_rebot.yaml --arm apai --apai-profile PROFILE` | `robot app with --arm apai would run:`, the driver `built, not connected`, exit code 2 | The profile's speed cap, the two rest tolerances, the jaw tolerance | A refusal (speed cap, rest pose, joint count) |
| 10 | Thor, then console | The link without the arm: `EX/run_robot_app.sh --config EX/configs/robot_app.yaml --arm sim --connect WS:47610`. Type `stop` at the console. Leave this robot app running until step 14 | The console's `WELCOME` line says `driver sim`; `stop: taken` | The time from Enter to `taken`, five times | No connection. This is the first run of port 47610 between two machines |
| 11 | Thor, workstation | Start the camera sender with `EX/configs/camera_sender_rebot.yaml`; `EX/run_camera_receiver.sh --host THOR` | The receiver reports about 60 camera frames per second | The rate; the refused camera frames | The sender refuses the sensor mode, or another program holds the camera |
| 12 | Workstation | `EX/run_view.sh --config EX/configs/view.yaml`; open `http://127.0.0.1:47620/` | The arm at rest in the picture; no `STALE`; the console's status line shows `view` under 1 s | The view age over one minute: lowest, middle, highest; the GPU memory with receiver and view | No picture |
| 13 | View | Read the view age | Between 0 and 1 s, steady | - | Negative, above 1 s or drifting: no task today. Find the cause on the Thor (the sender's clock, a clock step); there is no correction setting. For demo 2 go on with F-A before step 14 |
| 14 | Thor, workstation | `quit` at the Thor terminal (the simulated arm is at rest: it ends); after F-A no robot app runs and there is nothing to quit. Then `EX/run_agent.sh --config EX/configs/agent_rebot.yaml --check` | Exit code 0; the calibration id it prints equals `camera_calibration_id` of `robot_app_rebot.yaml` | The id | Another id: find out which calibration the rig is in |

### F-A. Before power: headset and human path on the simulated arm

Part F is demo 2 on the rig. F-A runs before power, after step 13 and before step 14: the
headset, the human program and the robot app on the simulated arm on the Thor. No arm
motion. F-B, on the powered arm, follows step 27. In both the human program runs on the
Thor, attached at the robot app's own socket, and reaches the Teleop Session Server on
the workstation over the wired link.

| # | Where | Do | Must show | Write down | Abort if |
|---|---|---|---|---|---|
| F0 | Workstation, Thor | Switch the screen lock and blanking off for the session; keep the live view's window un-minimised with its tab in front. From step 12's numbers choose `human.view_age_limit_s`: the highest age plus 0.2 s, at most 1.0 s. With 5 pictures per second and a VIEW_STATUS every 0.2 s the age swings by about 0.4 s before any camera delay, so expect to raise the shipped 0.5. If the highest age is above 0.3 s, first set `picture.fps: 10` and `status.period_s: 0.1` in `view.yaml`, restart the view and read the age again. Write the limit into `robot_app_human_check.yaml` and `robot_app_rebot.yaml` on the Thor. Then `quit` step 10's robot app (it is at rest) and start it again with its socket for the human program: `EX/run_robot_app.sh --config EX/configs/robot_app_human_check.yaml --arm sim --connect WS:47610 --human-socket ~/.agentic_remote/human.sock` | The Thor terminal prints `robot app: the human program's socket is ...` after its session line. The console's `WELCOME` says `driver sim`; `view` stays under the chosen limit | The ages; the limit chosen | The highest age is above 0.8 s or drifts: no human motion today |
| F1 | Workstation | Terminal T1: `TELEOP_CLIENT_WARMUP_END_TIMEOUT_MS=0 PYTHONPATH=BUILD/python_package/Release SRVPY -m isaaccapture.cloudxr.service run --accept-eula --setup-oob --usb-local`. Without that variable the web client on the headset drops the stream after about 30 s. Then terminal T2, on the address the Thor reaches: `PYTHONPATH=BUILD/python_package/Release SRVPY -m isaaccapture.teleop_session_server --listen tcp://WS:47600` | T2 prints `Teleop Session Server listening on ...`, then logs `No headset is connected to CloudXR; waiting for one` | Both start lines | Exit code 1: CloudXR is not available |
| F2 | Thor, second terminal | Start the human program before the headset is connected, with the variables of `run_robot_app.sh` set: `AGENTIC_REMOTE_HUMAN_PYTHON="$AGENTIC_REMOTE_THOR_PYTHON" bash EX/run_human.sh --config EX/configs/human_rebot.yaml` | `human program: teleop; driver sim; arm model rebot_rs, 6 joints`; then `human program: Teleop Session Server connecting_openxr: ...` or `waiting_for_headset: ...`. Console: `human program at the robot app`, `headset <that stage>` and `takeover: not possible: no fresh controller data: stage ...` | The lines | The human program ends. `cannot attach to the robot link service: ...`: the robot app of F0 was started without `--human-socket`, or with another path than `socket_path` of `human_rebot.yaml`. `no_server` while T2 runs: `teleop.endpoint` still holds its placeholder, T2 does not listen on `WS`, or the workstation's firewall keeps the Thor from port 47600 |
| F3 | Headset | Headset on the USB cable. In the web client's panel switch Headless on, then press Connect | T2 logs `Opened the OpenXR session`; its line `Serving robot app 'agentic_remote_human' ...` names the human program. Without a restart of the human program: `human program: Teleop Session Server ready`, `human program: right controller seen`; console `headset ready` | The time from Connect to `ready` | No `ready` within 2 minutes. `NotSupportedError` in a loop on the headset: switch Headless on, press Connect by hand, restart T2 |
| F4 | Thor, second terminal | Read the human program's report line for one minute | `stage ready`; tick age, state age and round trip steady | Each of the three: lowest, middle, highest. The tick age is that of SAMPLES that crossed the wired link. The round trip is the way through the robot app's own socket both ways plus one pass of the human program. Also `controller X s` of the console's status line: the sample age with the way to the robot app in it | The tick age comes near `teleop.max_tick_age_s` (0.15 s), or the round trip is above 0.1 s or drifts: find out why before any human motion |
| F5 | Headset, by eye | The seeing check. Three settings. (1) Headless on: the web client draws nothing; can the wearer see the arm, and the live view, below the visor? (2) Headless off: does the room show, and do controller data keep arriving (the report line)? (3) The headset off the face with its proximity sensor overridden (`adb shell am broadcast -a com.oculus.vrpowermanager.prox_close`), resting on the forehead or on a stand that sees the right hand: the wearer watches the arm directly | - | The three answers, in words | The wearer must see the arm itself in one of the three settings. The live view rule is met by the Thor's check plus the workstation person watching the page. No setting works: no human motion on the arm today |
| F6 | Controller, Thor, console | Press right A, right B, the grip and the trigger, each alone. Tap B ten times as briefly as possible. Then hold the controller out of the headset's view, behind the back, and press right B | The human program prints each edge in the Thor's second terminal (`human program: B down`, `human program: B up`). Each B press is a `[stop] STOP from controller: right B; ...` at the Thor terminal and on the console | Which of the four arrive. How many of the ten taps became a STOP. Whether the STOP arrives with the controller out of view | B or the grip never arrives: no demo 2 with buttons. If only A or B is missing, the owner may switch the typed fallback on: `operator.typed_takeover: true` in `link_service_rebot.yaml`, then restart the robot link service. B does not arrive out of view: the wearer keeps the controller in view, and the abort stays the Thor terminal |
| F7 | Wearer, workstation | The direction check, on the simulated arm. Stand behind the arm, recentre. Hold B 1 s. Release the grip, then squeeze. Move the hand **up** about 10 cm at about 1 cm per second, then forward, then left. Then turn the hand about its forward axis by about 30 degrees and back | Console: `[owner] TAKEOVER: a person owns the arm, ...`, two bells that are heard apart. The report line's tip target: z rises for up, x for forward, y for left; roll changes with the hand | The three directions as seen. The `ended:` counts in two minutes of motion. The IK time | `no IK solution` while the tip is still at the rest height is expected: from rest, lift the gripper first. A wrong heading: set `retarget.stage_yaw_deg` in `human_rebot.yaml` and repeat; a positive value turns the hand's motion counter-clockwise seen from above. A swapped or mirrored axis: correct `retarget.controller_to_base`. Many jumps or failures once the arm is raised: no human motion on the arm today. A lift that ends with `IK jump` about 39 cm above rest is the end of the arm's reach on the model |
| F8 | View | While squeezing on the simulated arm, each once: hide the browser tab; freeze the browser (stop its process, then let it go on); close it; stop the camera sender. After each: bring the view back and wait until the console's `view` is under the limit; if it stays `none` for more than a minute, restart the receiver and the view (steps 11 and 12) and reload the page. After a restart of the view, and after any step of the Thor's clock, the view is not live for 3 s | `[human] human drive ended: the live view is not fresh (...)`, one bell; the simulated arm's motion is `hold` each time | The time from the action to the EVENT | The drive goes on |
| F9 | Headset | While squeezing, each once: switch the right controller off; take the headset off and put the controller down; switch the headset's Wi-Fi off; stop the Teleop Session Server (then start it again, switch Headless on again, wait for `ready`); pull the network cable between the two machines, which carries the teleop connection and the robot connection (then put it back and wait for `headset ready` and for `view` under the limit) | `[human] human drive ended: ...`; `hold` each time. For the cable the Thor terminal shows `[connection] robot connection lost: the arm holds; ...`, and right B is not read while the cable is out: the buttons come over it | The time from the action to the hold. For the Wi-Fi: how far the simulated arm went and the time to `no_controller`. For the cable: the time until the human program says `ready` again | The drive goes on |
| F10 | Console, controller, Thor | The STOP under load. While squeezing on the simulated arm (50 HUMAN_SETPOINTs per second at the robot app's own socket): type `stop` at the console, five times; press right B, five times; take over again between them. Then right B without the robot connection: end the robot link service with Ctrl+C, press right B, start the service again | `stop: taken`; `[stop] STOP from controller: right B; ...`. With the service ended the Thor terminal shows `robot app: no robot connection: ...` and, at the press, `[stop] STOP from controller: right B; the arm holds, stop number N` | Each time from Enter or press to the higher stop number; compare with step 10's five values | Any above 0.25 s: set `rates.loop_hz: 25` in `human_rebot.yaml` and repeat (that halves the hand speed limits); still above: no human motion on the arm today. No STOP at the Thor terminal for right B with the service ended: no human motion on the arm today |
| F11 | Controller | The recentre. While squeezing on the simulated arm: recentre the headset view | `human drive ended` or a note of the human program; the report line's tip target does not jump; the simulated arm's target moves by no more than `limits.human.max_step_rad` | What ended it | The simulated arm jumps: no human motion on the arm today |
| F12 | All | The time from a spoken "go", with the headset asleep on the table, to `TAKEOVER: a person owns the arm` on the simulated arm. A right B that is already down as the headset wakes up is a STOP and takes nothing over: the note then reads `release right B, then hold it for 1 s with the grip released to take over` | - | The time. Compare with `waits.assistance_s` (120 s) | - |
| F12b | Controller, console, Thor | The trigger, on the simulated arm, while the person owns it and its jaws are open and empty. With the grip released: pull the trigger slowly all the way, hold it, release it slowly all the way | The console's jaws line reads `jaws: follow the trigger`. The human program's report line shows `jaws follow the trigger yes, grip level 1.00`, and its `trigger asks` goes from 1.00 to 0.00 and back | What `trigger asks` reads with the trigger at rest and with it fully pulled | At rest it stays below 1.00, or fully pulled above 0.00: the dead zones of the trigger, 0.05 at each end, do not cover this controller, and a trigger at rest could leave the jaws short of fully open, where the hand-back is refused. Set `human.trigger_jaws: false` in `robot_app_rebot.yaml` for this session |

Then type `rest` and `path is clear` (the simulated arm was driven away from rest), hold
right A 1 s, `quit` at the Thor terminal, and go on with step 14. The human program ends
with that robot app, whose socket it was attached at: exit code 1 and `human program:
the robot link service is gone (...)`. It is started again in F13, after the robot app
of step 15.

F-A does not meet APAI's human checker: its robot app runs on the simulated arm. So it
meets neither the table nor the rule of the lift, which the simulated arm's checkers do
not have; the tests of that rule run on APAI's `mock` profile (`test_lift_out.py`). A
test checks APAI's human checker, built from the rig's robot file and APAI's `rebot_rs`
profile, for a takeover at measured rest and a lift of 3 cm, on a stand-in for the motor
library (`test_a_takeover_and_a_lift_run_from_each_rig_reading_with_the_rig_file`). With
the rig's own profile it is first met on the arm, in F16. Not checked anywhere: sideways
motion, a turned hand, and the wrist camera rule under larger motions. F-A does not meet
the rig's human joint box either: `robot_app_human_check.yaml` has a box of -2.5 to
2.5 rad that equals its region, and the simulated arm reads exactly what it was sent. The
rig file's box, 0.02 rad inside `rest.region_*`, is first met on the arm in F-B.

Before step 15 the owner decides whether `human.enabled` in `robot_app_rebot.yaml` is set
to `true` for this session, confirms `human.view_age_limit_s` there, and decides whether
`human.trigger_jaws` stays `true`. If F5, F6, F7 or F10 failed, `human.enabled` stays
`false` and part F-B is not run. For F-B the robot app of step 15 is started with
`--human-socket ~/.agentic_remote/human.sock`: without that socket the human program
has nothing to attach to, and adding it later needs `rest`, a shutdown and steps 15 and
16 again.

### B. Power and connect: the connect cuts torque, so at rest only

| # | Where | Do | Must show | Write down | Abort if |
|---|---|---|---|---|---|
| 15 | Thor | 48 V on. The arm still lies at rest; the second person has a hand on it. `EX/run_robot_app.sh --config EX/configs/robot_app_rebot.yaml --arm apai --driver real --apai-profile PROFILE --calibration ID --connect WS:47610`, for demo 2 with `--human-socket ~/.agentic_remote/human.sock` added; answer the three questions with `yes` | `robot app: the real driver is connected at measured rest: motors on, motion hold` (or `rest_hold`)`, jaws referenced`; then `robot app: session ...`; the console's `WELCOME` says `driver real` | The joint reads of the connect, from `~/.apai/logs/rest_readings.jsonl`; the jaw reading | A line that begins `robot app: the robot app did not start: `, then a line that begins `robot app: APAI's driver refused the connect before anything that cuts torque`: the motors were never switched on and the arm lies on its stops as in step 4. Read the reason, correct the pose by hand, start again. A line that begins `robot app: the robot app did not start: `, then a line that begins `robot app: the start failed after the connect began: the motors may be enabled and holding`: the program has ended without releasing them. The arm lies at rest: with the second person's hand on it, switch the 48 V off, read the reason, start again |
| 16 | Thor | If the motion is `hold`: type `rest`, then `path is clear`. The arm moves a few hundredths of a rad | `rest: completed`; motion `rest_hold` | The planned and the real time | The arm moves away from rest: `stop` |
| 17 | Console, Thor | The release at rest: `release`, then `yes` | `torque release sent at rest: the driver does not confirm that the motors are off`; the motors are off | Whether the motors are off, by hand feel | `refused`: the motors stay on; read why. `failed` with `fault: torque release failed, the motors may be on or off: ...` and `torque unknown` on the console: keep a hand on the arm and treat it as powered; `reset`, then `release` again (see [the torque rule](#the-rule-torque-is-never-cut-away-from-rest)) |
| 18 | Thor | After a release the robot app has to be started again (`the motors are off; restart the robot app`): `quit`, then steps 15 and 16 again | As steps 15 and 16 | - | As step 15 |
| 19 | Console, Thor | With the arm at rest: `stop` at the console; Ctrl+C at the Thor terminal. A `reset` is not needed: a STOP is no fault-stop | The stop number goes up each time; the arm does not move | The delays | The arm moves |

### C. Camera and agent: no motion

| # | Where | Do | Must show | Write down | Abort if |
|---|---|---|---|---|---|
| 20 | Workstation | `EX/run_agent.sh --config EX/configs/agent_rebot.yaml --interactive`; the task `look at the table and report what objects you see` | `driver real` in the agent program's line about the robot connection; the obstacle grid goes live; a report; the agent takes no arm; the jaws do not move when the task ends | The time to a live grid; the depth time per stereo pair; the GPU memory in all; the model's answer time | A calibration mismatch; every camera frame refused; no grid within `task.grid_wait_s` |

Step 20 needs the model client, which is not installed (see
[The workstation environment](#the-workstation-environment)). Without it steps 20 to 27
are not run: after step 19 the session goes on with the variant F-B0 of part F, or ends
with steps 28 and 29.

### D. First motion: the second person supports the arm; 0.1 rad/s; no timed return

| # | Where | Do | Must show | Write down | Abort if |
|---|---|---|---|---|---|
| 21 | Agent terminal, console | The task `open the gripper`, then `yes`. Then a task that lifts the gripper a few centimetres; type `stop` at the console while the move is under way | The jaws open; the move starts; after `stop` the arm holds, nobody owns it and the task ends | The time from Enter to the higher stop number; how far the arm went after it | More than 0.25 s |
| 22 | Agent terminal, Thor | The same move twice more: end it with Ctrl+C at the agent terminal, then with `stop` at the Thor terminal | The arm holds each time | The delays | The arm goes on |
| 23 | Console, run folder | A full small move. Read the permit in the console's status line and the move's records in the run folder | Every setpoint under a permit; no `permit evidence is in the future` | The oldest evidence; the longest gap between permits; the arm's stopping distance at 0.1 rad/s | Permits refused for their time |
| 24 | By hand | Put a foam block into the path during a move | The arm holds before the block; a help request; the console shows the assistance wait and rings its bells; type `stop` to end the wait. The next task starts without a restart of the agent program | Where it stopped; the distance to the block | The arm touches the block |
| 25 | Workstation, cable | During a move, each once: end the agent program with a signal; pull the network cable; stop the camera sender; stop the robot link service | The arm holds each time; `agent program gone` for the first; `ROBOT CONNECTION LOST` on the console within about 3 s for the cable; after the cable is back nothing moves | The delays | The arm goes on, or its motors switch off |
| 26 | Console | From a raised pose, the path clear, the arm supported: `rest`, then `path is clear`. Once more, with `stop` during the return | The arm reaches rest with its motors on; the second return ends in a hold | The time; the start pose | A contact |

### E. Demo 1

| # | Where | Do | Must show | Write down | Abort if |
|---|---|---|---|---|---|
| 27 | Agent terminal | The demo task, `pick up and place the red object` or the experiment's task text, three times in a row; the arm is not supported once steps 21 to 26 passed. `rest` from the console between the runs | Three successes; the arm is given back each time | Each run's time and task report | Two failures in a row |

### F-B. Demo 2 on the arm: 0.1 rad/s; no timed return; the stop person at the Thor terminal

F-B is run only when F-A passed and `human.enabled` is `true` in `robot_app_rebot.yaml`
(the robot app of step 15 was started with it, and with `--human-socket
~/.agentic_remote/human.sock`). With the model client installed it runs
after step 27. Without it steps 20 to 27 are not run, and F-B runs after step 19 in the
variant F-B0 below.

| # | Where | Do | Must show | Write down | Abort if |
|---|---|---|---|---|---|
| F13 | Thor, view, console | Start the human program in the Thor's second terminal as in F2: the one of F-A ended with its robot app. Check the view age again with the arm powered | `human program: teleop; driver real; arm model rebot_rs, 6 joints`. The console shows `human program at the robot app` and `headset ready`, and its `view` stays under the chosen limit for one minute | The age: lowest, middle, highest | It does not: no human motion today. Changing the limit now needs `rest`, a shutdown and steps 15 and 16 again |
| F14 | Agent terminal, by hand | A task that lifts the gripper; a foam block into the path (step 24), well above the table | The arm holds before the block; `HELP NEEDED`; the wait counts and says `ARM WILL KEEP HOLDING`; the console says `takeover: possible now ...` | The time to the help block | `takeover: not possible`: read why. An arm that the agent left with a low gripper tip, below the table clearance or inside its last grasp region, takes the takeover of F15, whose line then ends `; the gripper tip is low: only a lift is taken until it is clear`, and no `rest` before the wearer has lifted it clear: straight up, as in F16, because a sinking hand and, below the table clearance, any step that does not rise end the drive. An arm that the agent left with a joint in the outer 0.02 rad of its working box, outside the human joint box, takes no takeover either, and the takeover line does not say so: it reads `takeover: possible now ...`, and the hold of F15 is refused with `the arm is outside the human limits: ...; type rest` and three bells. There the stop person types `rest`, then `path is clear` (see [Known gaps](#known-gaps)) |
| F15 | Controller | Hold right B 1 s with the grip released. Then take the foam block and everything else out of the arm's reach | `[stop] STOP from controller: right B; ...`; the wait is cancelled; `[owner] TAKEOVER: a person owns the arm, ...`, two bells. The arm does not move. The jaws do not move: a takeover sends no jaw command, and the wearer keeps the trigger released | The time from the press to the STOP's higher stop number | The arm or the jaws move |
| F16 | Controller | The first human motion. Release the grip, then squeeze; move the hand 3 cm straight up over at least 5 s; release. The stop person presses Enter at the first motion that is not upward. The hand's target moves at 1 cm per second; a hand more than 5 cm ahead ends the engagement: move at most 3 cm at a time and wait for the arm. Keep the hand's orientation as it was at the squeeze: a turned hand turns the tool at up to `retarget.max_angular_velocity` (0.03 rad/s) until the tool has caught up or the grip is released, and the hand-ahead limit compares positions only, so it does not end that. Then squeeze and release three times with the hand held still | The arm follows upward and holds on the release. With the still hand the arm does not move | The lag by eye; any `human drive ended`, and for one that names a contact its text, with the tracking error it gives (APAI's contact monitor trips only beyond its lag band, about 0.03 rad at 0.1 rad/s); whether the arm sinks from one squeeze to the next (a drive starts from the measured pose, so an arm that sags under load can creep down by the sag per squeeze) | Any motion that is not upward; no hold on the release; a creep that does not stop |
| F17 | Controller, agent terminal | Squeeze and move; press right B during the motion. Then try a task at the agent terminal | The arm holds; nobody owns it; `takeover open`. The task is refused with `takeover not finished: give the arm back with right A at rest` | The time from the press to the hold; how far the arm went | More than 0.25 s |
| F18 | Controller, console | Hold B 1 s again. Then the trigger, with nothing between the jaws and the stop person's `stop` typed: pull it slowly all the way, hold it, release it slowly all the way | A takeover again. With `human.trigger_jaws: true`: the console's jaws line reads `jaws: follow the trigger`; the jaws close with the trigger and open again fully, in steps. With `false`: the jaws do not move | Any refusal of the takeover, with its text. With `true`: what the console's jaws line and the report line's `trigger asks` read at rest and fully pulled; the time the jaws take to open | With `true`: the jaws move while the console's jaws line does not read `jaws: follow the trigger`, or go on after the stop person's `stop`. With `false`: the jaws move. A takeover refused with `[refused] owner request 'human' refused: the arm is outside the human limits: ...; type rest` and three bells is no abort: the STOP of F17 came with a joint at the edge of the human joint box, where it reads just outside the box. The stop person types `rest`, then `path is clear`: the box lies 0.02 rad inside `rest.region_*`, so the pose is one the return starts from. Take over again from rest |
| F19 | Workstation, Thor | While squeezing, each once: hide the browser tab; stop the Teleop Session Server (then start it again, switch Headless on again in the headset, wait for `ready`); end the human program on the Thor with a kill signal (then start it again there) | The arm holds each time; the person stays the control owner; after a release and a new squeeze the arm follows again | The delays | The arm goes on, or its motors switch off |
| F20 | Cable | While squeezing: pull the network cable at the workstation's end; nobody reaches toward the Thor. The cable carries the robot connection and the teleop connection: while it is out right B is not read, and the stop is the Thor terminal's. Put it back and wait until the console shows `headset ready` and its `view` is under the limit | The arm holds; `ROBOT CONNECTION LOST` on the console; after the cable is back nothing moves; a release and a new squeeze work | The delays, among them the time until the human program says `ready` again | The arm moves by itself |
| F21 | Console, controller | Release the grip. `rest`, then `path is clear`. Then hold right A 1 s | The arm returns to rest with its motors on; `[owner] takeover ended: the arm is given back; nobody owns it`, two bells; `owner nobody` | The return's time | `rest` refused with `the path to rest was refused: ...`: the wearer squeezes, moves the arm back the way it came and releases; then `rest` again. `owner request 'handback' refused`: read why; the arm stays at rest with its motors on. Refused for jaws that are not proven empty: release the trigger all the way and wait until the jaws are fully open, then right A again |
| F22 | Agent terminal, all | Demo 2, three times: a task; the block; the takeover; the block is taken away; the person moves the arm; `rest`; right A; the next task | Three full cycles with no program restarted | Each cycle's times: help request to takeover, takeover to hand-back | Two failed cycles in a row |

**F-B0, without the agent** (steps 20 to 27 were not run). The arm is at rest after step
19, nobody owns it, `human.enabled` is true, and the supporter supports the arm: these are
the arm's first motions under this program. F13; then F15 (a takeover needs no help
request; there is no wait and no block); F16 from rest: at the squeeze the arm first lifts
the second and the third joint 0.025 rad off their stops, then follows the hand straight
up, and from rest the gripper is lifted first; F17 without the task; F18 to F21; then
three cycles of takeover, lift, `rest`, right A in place of F22. The help block before a
takeover (F14) and the refused task (F17) are then shown on one machine only.

### G. The end

| # | Where | Do | Must show | Write down | Abort if |
|---|---|---|---|---|---|
| 28 | Console or Thor | `rest`, then `path is clear`. Then `shutdown` at the console (at rest it asks for `yes`) or `quit` at the Thor terminal | The robot app measures rest, releases torque there and ends with exit code 0 | - | `refused`: the arm holds with its motors on; read why |
| 29 | All | 48 V off, with the arm at rest. Stop the camera sender, the receiver, the view, the console, the human program, the Teleop Session Server, CloudXR's service and the robot link service | - | What was left open | - |

Not in this session: reading the motors' CAN timeout parameter; ending the robot app during
a raised hold; the jaw rule with different objects; the timed return; the typed `reference`;
any speed above 0.1 rad/s; the trigger on a held object (F18 moves the empty jaws, and
only with `human.trigger_jaws: true`); the lift of a low gripper tip, unless an agent's
grasp leaves one (F14); the typed takeover unless F6 forces it; a person who drives from
another room.

## What was run

On the workstation, 2026-11-14, with an Isaac Capture build on the Python path:

| Command | Run as | Result |
|---|---|---|
| `robot_app --arm sim --selftest` | As written | Exit code 0, 11 steps |
| `robot_app --arm sim --selftest-link` | As written | Exit code 0, 12 steps; exit code 2 and one line without Isaac Capture |
| `link_service --config configs/link_service.yaml` | Not with the shipped file. Run as a process by the tests with a file that names port 0 and a socket in a temporary folder; with the shipped file only without Isaac Capture | Exit code 0 on SIGINT and on SIGTERM, 1 for a second service on the socket; exit code 2 and one line without Isaac Capture |
| `robot_app --arm sim --connect 127.0.0.1:47610` | Not on port 47610. Run as a process by the tests against the port the service picked; with port 47610 only without Isaac Capture | Exit code 0 after `quit` and after SIGTERM; Ctrl+C is a STOP; exit code 2 and one line without Isaac Capture |
| `robot_app --arm apai --driver mock --connect ...` | By tests, in the test process and as a process, against the port the service picked, with the human limit set at 0.8 rad/s | Exit code 0 after a SHUTDOWN at rest and after a typed `quit` |
| `robot_app --arm apai --driver real ... --connect ...` | Only by tests, in the test process, with APAI's driver class over a stand-in for the motor library and the answers scripted (`test_real_driver.py`). Never on the arm, and never as a process at a terminal | With three `yes` and the stand-in arm at a recorded rest reading: WELCOME says `driver real`. With the stand-in arm raised: exit code 1 and nothing was cut. A wrong `--calibration`, a hold marker, an operator confirmation on file, an agent's environment, no terminal: exit code 2 before anything was built |

The two self-tests were run again on 2026-11-15: exit code 0 both, 11 steps in 5.0 s and 12
steps in 6.3 s.

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
| `console_cmd move`; `console_cmd release`, `state` and `stop` without a robot link service, with standard input from `/dev/null` | As processes, with `--socket` naming a path nothing listens on | Exit code 2; exit code 1 each: `cannot attach to the robot link service: ...`, and for `stop` also `STOP NOT written: ...` |
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

The operator's programs, the rig files and the programs of demo 1 together, on the
workstation, 2026-11-15, in the same environment:

| What | Run as | Result |
|---|---|---|
| `run_console.sh --help`; `run_console.sh` with standard input from `/dev/null` | As processes, with the test folder's Python as `AGENTIC_REMOTE_CONSOLE_PYTHON` | Exit code 0; exit code 2 and `it needs a terminal on standard input and output; nothing sent` |
| The operator console at a terminal | By tests, in a child process on a pseudo-terminal, without a service | Ctrl+C was a STOP and did not end it; `quit` ended it with exit code 0; inside an agent's environment it asked nothing |
| The operator console against the robot app | By tests, in the test process: the console's class with its lines typed by the test | The confirmations of the wording table; `stop` taken while a `rest` ran; the help block; `ROBOT CONNECTION LOST` |
| `view --help`; `run_view.sh` without its Python; the view without a robot link service | As processes | Exit code 0; exit code 2 and one line; exit code 1 and `cannot attach to the robot link service` |
| The live view with stand-in camera pictures, with a recorded pair, with `--source none` | By tests: the program's `main()` on a thread and as a process; the page asked over HTTP | VIEW_STATUS reached the robot app and `view_age_s` showed in ROBOT_STATE; a recorded source was never live; exit code 0 on SIGTERM and SIGINT. No browser, no camera receiver, no camera frame bridge on a GPU |
| `run_robot_app.sh` without `AGENTIC_REMOTE_THOR_PYTHON` | As a process | Exit code 2 and one line |
| `link_service --config configs/link_service_rebot.yaml` | As written | Exit code 2: `allowed_peers must be a non-empty list` |
| `robot_app --config configs/robot_app.yaml --arm apai` | As written, with `AGENTIC_REMOTE_APAI_SRC` set | Exit code 2: the human limit set's 1.5 rad/s is above the mock profile's speed cap of 1.2 rad/s |
| `robot_app --arm apai --apai-profile rebot_rs` without `--connect` | By a test, in the test process, with the shipped file's human limit set at 0.8 rad/s (`test_main.py`) | Exit code 2; `the profile's speed cap: 0.8 rad/s` is printed; nothing was connected. With `configs/robot_app_rebot.yaml` and the rig's own profile this was not run: a test only loads that file and fits it to APAI's `rebot_rs` profile |
| The service, the robot app (`--arm sim`, and `--arm apai --driver mock`), the live view with `--source none`, the operator console on a pseudo-terminal and the console command as processes | By a test, on a port the system picked | The console named the driver; `console_cmd state` and `stop` ended with exit code 0; `rest` with `path is clear`, typed at the console, completed; `quit` ended the robot app and the console with exit code 0, SIGTERM the view and the service |
| Demo 1: the agent program with two tasks, a console and a live view attached at once | By a test, in the test process, behind the service on 127.0.0.1, on APAI's software arm, with stand-in camera pictures for the view | 905 joint targets in 8 moves, each under a permit; the console printed both grants and the help request; the view age on the console was 0.1 to 1.6 s while a tab acknowledged a picture every 1.5 s (two runs); the page showed the move. 22.4 to 22.7 s per run |
| `stop` typed at the console during the agent's move, then a second task | By a test, the same way, the view with `--source none` | The move ended `cancelled` at the setpoint it was typed at (40 of 95); `stop: taken` came 5 to 11 ms after the line was typed, on one machine; the second task ran without a restart |
| `rest` typed at the console after a task | By a test, the same way | `yes` sent nothing; `path is clear` returned the arm to rest in 2.6 s (105 joint targets); motors on |
| An obstacle during the first task, then a second task | By a test, the same way | The first move was refused after 33 to 35 setpoints; the console showed the help request and then the next grant, no STOP; the second task moved the arm; one run of the program, 24.0 s |
| The agent client's socket closed while the agent owned the arm | By a test, on the simulated arm | The console showed `HELP NEEDED: agent program gone` and the wait counting down; `stop` ended it; nothing moved |
| The robot app ended and a new one started | By a test, on the simulated arm | The console said `ROBOT CONNECTION LOST` and showed the new session; the view's page showed it too; a command with the old session's numbers was refused |

The workstation environment, on the workstation, 2026-11-15, in the folder
`AGENTIC_REMOTE_ENV_DIR` names, without the model client:

| What | Run as | Result |
|---|---|---|
| The whole test folder | With the environment's Python, on GPU 0, at commit 4126b9a13 | 4143 passed, 3 skipped |
| The six tests of `test_gpu_stack.py` | In that run and in a run of the file alone, on GPU 0. No test asks a model: the full stack's two model clients are stand-ins | 6 passed in each. FFS loaded and warmed up in 8.3 s and then took about 50 ms per stereo pair, on one recorded pair. The obstacle grid bridge answered its first query in 0.02 s. The segmenter loaded and segmented its first box in 1.53 s, the next in 0.031 s. The full stack started on recorded stereo pairs and its obstacle grid went live; no task ran |

The human path and the two one-machine recipes, on the workstation, 2026-11-16, in the
test folder's environment, with an Isaac Capture build and APAI's checkout named. No
headset, no Teleop Session Server and no arm:

| What | Run as | Result |
|---|---|---|
| The takeover, the hand-back and human drives on the robot app | By tests, in the test process, on the simulated arm and a fake clock, with HUMAN_SETPOINTs and human targets the tests make (`test_takeover.py`, `test_human_drive.py`); a few on APAI's checker | A takeover from nobody at rest needed no help request and cancelled the assistance wait; each refusal of the takeover and of the hand-back gave its text; a squeeze held through the grant, through a return to rest and through a gap in the controller data moved nothing until the grip was released; a fast target was clamped at 0.98 of the speed limit and a far one refused; a drive ended with its EVENT, also one that had sent nothing, when controller data were away between two HUMAN_SETPOINTs, when an engaged one was refused and when one quoted other numbers; an agent's task was refused while the takeover was open; every end of a drive left the motors on and no jaw command was sent |
| The human path on APAI's software arm | By tests, in the test process, on a fake clock, with APAI's checkers and its software arm (`test_apai_app.py`); one of them on the shipped `configs/robot_app_mock.yaml` | From a pose whose path to rest APAI refuses, the typed `rest` was refused with `the path to rest was refused: setpoint ...`; a hand-back, a shutdown, a `release` and a task of the agent were refused too, nothing was sent and the motors stayed on; after a squeeze had moved the arm to a plain pose the typed `rest` completed. After a move of the agent had ended at a contact, a typed `rest` with `path is clear` during the takeover cleared the recovery checker's contact spot, not the agent's, and returned the arm; an unconfirmed `rest` and a shutdown that needed no confirmation cleared none. A contact under a person held at the measured pose and left no contact spot, and the typed `rest` and the hand-back ran after it. A hand-back, taken or refused, sent the driver no joint target and no jaw command |
| A takeover and a lift with the rig's robot file | By a test, in the test process, on a fake clock (`test_rig_configs.py`): `configs/robot_app_rebot.yaml` with human driving switched on in memory, APAI's `rebot_rs` profile capped at 0.1 rad/s, APAI's driver class over a stand-in for the motor library, from each of the three rest readings recorded on the rig. The joint targets came from APAI's IK for a hand that rose 3 cm at 0.01 m/s | The takeover sent the motors nothing; APAI's human checker approved every setpoint of the lift; no step was above 0.98 of 0.1 rad/s; no IK solution lay 0.08 rad or more ahead of the last setpoint; the gripper tip rose 3 cm; the drive ended by the released grip, with no EVENT; the typed `rest` and the hand-back completed, the hand-back with nothing sent to the arm's motors or the jaw motor; the motors stayed on |
| The rig file's human joint box | By tests, in the test process, on a fake clock (`test_rig_configs.py`): `configs/robot_app_rebot.yaml` with human driving switched on in memory, once on the simulated arm and once on APAI's `rebot_rs` checkers and driver class over a stand-in for the motor library. Each pose was placed, not driven | The box lay 0.02 rad inside `rest.region_*` on ten bounds and on the region's own -0.05 at the lower bound of the second and the third joint. At each of the ten: with a joint 0.005 rad outside its bound at a STOP, a takeover and an agent's task were refused, nothing was sent, and the typed `rest` with `path is clear` returned the arm to rest, where a takeover was taken; with a joint 0.01 rad outside its bound and nobody as control owner, the takeover was refused and the typed `rest` returned the arm. At the two bounds without the margin, with a joint 0.005 rad below its bound at a STOP, the takeover and the typed `rest` were refused, nothing was sent and the motors stayed on |
| The retargeting graph | By tests, on Isaac Capture's retargeters with controller snapshots the tests make (`test_human_retarget.py`) | The target followed the hand in the robot's base axes; a positive `stage_yaw_deg` turned it counter-clockwise seen from above; a pass 40 ms, 100 ms or 1 s late got the motion of one pass; the buttons were read while the controller's pose was not valid |
| The IK | By tests, on APAI's arm model (`test_human_ik.py`) | From the rest pose a lift solved on every pass; forward, back and down had no solution; a jump was refused; no random restart was used. On the model 1 m/s at the gripper tip needed up to 10.1 rad/s at a joint away from rest and 12.9 rad/s at the pose a drive from rest starts from |
| The human program's logic | By tests, on stand-ins for the graph and the solver (`test_human_logic.py`) | Right B was a STOP on every pass with known buttons and one STOP per stretch; B held 1 s with the grip released asked for the takeover once, after the ROBOT_STATE that showed its STOP; a button down before a data loss counted only when pressed again; a jump, five failed solves and a target too far ahead each ended the engagement; the note named a button that is down with no press that can ask, and loaded jaws at rest; a hold of right B or of right A did not count across a stall of more than 1 s and ran on through one of 1 s or less; right B's STOP went out on a pass whose request failed, and an error before the buttons were read sent nothing and left the logic as it was; during a return to rest with a takeover open and the stage `ready` the note read `the arm returns to rest; right B is a STOP`, ahead of a refusal's note, and no link message differed but in its note; every HUMAN_SETPOINT stayed under 512 bytes |
| The human program and its two controller feeds | By tests (`test_human_feed.py`, `test_human_program.py`): the script feed with the real retargeting graph, solver and logic, pass by pass against a stand-in for the Thor; the teleop feed on a stand-in session and on a real remote session against a server in miniature in the test process, on 127.0.0.1 and a port the system picked; the program's `main()` on a thread behind a robot link service whose robot side the test scripts, and once against the robot app on the simulated arm over a real link; the program as a process for SIGINT and SIGTERM and for a standard output nobody reads; `bash run_human.sh` as a process for its variables | Right B gave one STOP from `controller` and then the takeover; a squeeze and a lift of 2 cm raised the simulated arm's gripper tip; the program's end while the person owned the arm sent one STOP from `human_program`, on a signal, when the attempts ran out and after a failure; SIGINT and SIGTERM ended it with exit code 0; the causes of exit code 1 and 2 in the table gave their code and their line (of a restarted robot app only the synthetic controller's case was run); a standard output that took nothing held up neither the passes nor right B; right B's STOP went out ahead of the log line of its pass, and a log line that blocked on an earlier pass held up right B until it returned; with the log and standard error both blocked, the program's own STOP went out ahead of the lines of its end (an error, the attempts ran out, a new session it must not drive), and right B read on the last pass for such a session was a STOP from `controller`; right B was a STOP on a pass whose request could not be built; a `rates.loop_hz` above 100 was refused, and so were a number too large to use and a file that is not UTF-8, each with exit code 2 and one line; at 100 passes per second uneven passes raised the hand speed limit by 0.5 % and by 5 %; a pass that failed in the logic did not end the program; the feed gave the stages `no_server`, `waiting_for_headset`, `no_controller` and `stale`, and entered a new session after the server had closed |
| The role `human` and the typed fallback's switch in the robot link service | By tests (`test_link_service.py`, `test_local_api.py`) | The role sent its three link messages and nothing else; a second human program was refused; a HUMAN_SETPOINT went to the Thor newest-wins and stayed at the service while a STOP was not confirmed; the console's `takeover` and `handback` were forwarded only with `operator.typed_takeover` on; a human program that left sent nothing to the Thor |
| The takeover line, the status line's human parts, the bells, `takeover` and `handback` at the console | By tests: the console's class with its lines typed by the test (`test_console.py`, `test_console_cmd.py`) | Each text of the takeover line, among them the one for loaded jaws at rest; the note shown only while a stage is reported; the console's last line with and without a robot connection; one, two and three bell characters; a refused takeover rang through a flood of other refusals; the service's refusal was shown with `use right B / right A on the controller` |
| The task gate with a takeover open | By tests (`test_task_gate.py`) | No task was started while a takeover was open; the robot app's refusal was printed with its remedy |
| The view token | By tests, over HTTP (`test_view_model.py`, `test_view_server.py`, `test_view_program.py`) | An acknowledgement without the page's token, of a picture the page never fetched and of a replaced page got 403 and changed nothing; a reload made the view not live until the new page acknowledged; a tab hidden too long after its fetch was not live; a `visible` handled after a `hidden` for the same fetch got 403; the token was in neither `/status.json` nor the log. The page's script was not run |
| The recipe of demo 1 with the mock agent | By a test, as processes (`test_demo1_recipe.py`): the robot app on the shipped `configs/robot_app_mock.yaml`; the agent program started with `bash run_agent.sh` at a pseudo-terminal that is its controlling terminal, the task lines and each `yes` typed there; the console on a pseudo-terminal. What differs from the recipe: the service, the view and the agent program ran on copies of their files that name a port the system picks, a socket and a state folder in a temporary folder; the console and the view were started with `python -m`, not through their start scripts; a fourth local client in the role `tool` read ROBOT_STATE; APAI's stores lay in the temporary folder and `APAI_GRASPGENX_PORT` named a port nobody listens on; the page was read with the tests' HTTP client | Every "must show" of the recipe. `open the gripper` succeeded; the pick and place ran 8 moves under permits and ended with the help request, the arm holding at about [0, 1.2, 1.2, 0, 0, 0] rad with its motors on; `stop`, `rest` with `path is clear` and `shutdown` with `yes` worked; exit code 1 for the agent program, 0 for the others. The process test takes 53 s |
| The recipe of demo 2 on one machine | By a test, as processes (`test_the_recipe_of_demo_2_runs_as_processes_with_the_shipped_files` in `test_demo2_programs.py`): the robot app on the shipped `configs/robot_app_demo2.yaml` with `--arm sim`; the human program started with `bash run_human.sh --controller synthetic:lift`; the console on a pseudo-terminal. What differs from the recipe: the service, the view and the human program ran on copies of their files that name a port the system picked, a socket and a state folder in a temporary folder and, as the Teleop Session Server's, a port nobody listens on; the console and the view were started with `python -m`, not through their start scripts; a fourth local client in the role `tool` read ROBOT_STATE; the page was read with the tests' HTTP client | Every "must show" of steps 1 to 8, those of steps 6 and 7 in the recipe's order. The script's right B was a STOP and took the arm over; the gripper tip rose 5 cm off rest, within 3 mm; the typed `rest` returned the arm in 5.0 s and the script's right A gave it back; the owner numbers ran from 0 to 5 up to the hand-back; `shutdown` with `yes` released torque at rest; exit code 0 for all five programs and no traceback from the human program. The test takes 23 s |
| The recipe's variant with the agent failing first | By a test, as processes (`test_demo_2_with_the_agent_failing_first_runs_as_processes`): the robot app on the shipped `configs/robot_app_mock.yaml` with `--arm apai --driver mock`; the agent program started with `bash run_agent.sh` at a pseudo-terminal that is its controlling terminal, on a copy of `configs/agent_mock.yaml`; the human program started once the agent had asked for help. The other differences are those of the row above | The pick and place ended in the help request; the help block ended with `takeover: not possible: no human program is attached: start run_human.sh`; the script's right B cancelled the assistance wait and took the arm over with owner number 4; the gripper tip rose 5 cm from where the agent had left it; the typed `rest` returned the arm in 30.9 s (largest joint travel 1.48 rad); after the hand-back `open the gripper` succeeded under owner number 8; the console printed one `WELCOME`; no `open` was typed; exit code 1 for the agent program, 0 for the others. The test takes 72 s |
| A person driving, with every program of demo 2 attached, on the simulated arm | By tests, ten of `test_demo2_programs.py`, in the test process and on the machine's clock: the robot app behind the robot link service on 127.0.0.1; the human program's `main()` on a thread with script rows the test gives; the console's class with its lines typed by the test; the live view's `main()` on a thread with stand-in camera pictures (15 per second, a VIEW_STATUS every 0.1 s) and an HTTP client that fetches a picture and acknowledges it with the page's token every 0.25 s; `human.view_age_limit_s: 2.0` in the robot app | `stop` typed at the console during a squeeze held the arm within one setpoint period, with nobody as control owner and the takeover open; the hand, still squeezed and rising, moved nothing more; an agent client's task was refused with `takeover not finished: give the arm back with right A at rest` and was taken after `rest` and right A. Right B pressed during a squeeze and held for 2.2 s was one STOP and asked for nothing. A controller that was absent for 1.5 s and a browser tab that stopped acknowledging each ended the drive in a hold with the person still the control owner; with the cause gone and the grip still squeezed nothing moved, and after a release and a new squeeze the arm followed again under the same takeover. With the human program's socket closed the last joint target went out within `human.sample_age_limit_s` plus one period of the last human target, the program ended with exit code 1 and the console showed `no human program` beside `owner human` and `takeover open`. With the agent client's socket closed during a squeeze the drive went on: no help request and no assistance wait. With the robot connection cut during a squeeze the arm held and the person kept it; `rest` typed at the Thor terminal returned it meanwhile; after the reconnect the squeezed grip moved nothing until it was released and squeezed again. With the tab away a second time, right B was a STOP and its hold was refused with `the live view is not fresh: ...; open the live view page in a visible browser tab` and three bells; so was a takeover with `--source none`, where nothing moved and no takeover was open. Right B during the assistance wait cancelled it: no return to rest started, with `rest.auto_return` true. With `operator.typed_takeover` on and no button pressed, `stop`, `takeover`, `yes`, a squeeze and a lift of 2 cm, `rest`, `path is clear`, `handback`, `yes` ended with `takeover ended` and `owner nobody`; with the setting off the same words printed the service's refusal and changed nothing |
| Demo 2 with the agent program, and a fast hand, on APAI's software arm | By tests, two of `test_demo2_programs.py`, the same way, the arm raised off its joint stops: one with the agent program in the test process on APAI's mock scene and an obstacle put into the obstacle grid during its first move; one with a hand that is 1 cm higher on the next pass and a human speed limit of 0.3 rad/s | The first task ended in a help request; the help block showed `jaws: empty` and ended `takeover: possible now: hold right B for 1 s with the grip released`; right B's STOP cancelled the wait and its hold took the arm over with two bells; the arm followed a lift of 3 cm, each joint target within 0.98 of the human speed limit times the period of the one before; `rest` with `path is clear`, then right A, gave the arm back; the second task, `open the gripper`, succeeded. The control owners in order: nobody, agent, nobody, human, recovery, human, nobody, agent, nobody. One run of every program on one session of the robot app, the motors on throughout, no jaw command from the takeover to the hand-back, every joint target of the agent under a permit. For the fast hand `health.clamped` rose by the number of joint targets that lay at the clamp (7 of 176 in one run), no drive ended, APAI's human checker approved every target and the gripper tip ended 1 cm higher |
| The two robot app files of demo 2 | By a test of `test_demo2_programs.py` that needs no APAI | `configs/robot_app_demo2.yaml` and `configs/robot_app_human_check.yaml` differ from `configs/robot_app.yaml` in exactly the keys their first comment lists; both load; each is refused with a human joint box wider than `rest.region_*`; the first is taken without a live view only with a simulated driver. Two more tests of the file check its own waits and its reading of the help block. The file has 18 tests and takes about 4 min with everything present |
| The human program at the robot app's own socket, as on the Thor | By tests (`test_human_on_thor.py`), on one machine. In the test process, on the simulated arm and a fake clock: a robot app with its socket in a temporary folder and a local client of the test in the role `human`; once the robot app's program on a thread with a dialer of the test. Two tests as processes: the robot link service, the robot app with `--arm sim`, `configs/robot_app_demo2.yaml` and `--human-socket`, the console on a pseudo-terminal and `bash run_human.sh --controller synthetic:lift` on a copy of `configs/human.yaml` that names the socket; and the robot app on `configs/robot_app_human_check.yaml` with a robot connection that nothing answers | The client got WELCOME, ROBOT_STATE and the RESULTs of its own requests, and `health.human.local_program` said whether it was attached; a takeover and a hand-back asked for at the socket were answered as on the robot connection; a second human program and every other role were refused; a client that did not read was cut off and held up no pass of the robot app; of two human programs one was followed; the socket's file had mode 0600 and went with the robot app. With the robot connection lost during a squeeze the drive ended in a hold and the person kept the arm; a new squeeze moved nothing, for want of a live view; right B's STOP was taken. As processes: right B, the takeover, a lift of 5 cm, the typed `rest` and right A ran with the console showing `human program at the robot app`; with no robot connection at all right B's STOP reached the robot app, and the takeover it asked for was refused with `the live view is not fresh: no VIEW_STATUS has arrived`. The two process tests take about 30 s |
| The jaws under a person's trigger | By tests (`test_human_jaws.py`), in the test process, on the simulated arm and a fake clock, with the human targets the tests make; one of them on APAI's software arm | A held object stayed held through a takeover with the trigger released; a trigger short of the grip level moved nothing; once it had reached the grip level, or crossed it, the jaws followed it, the grip squeezed or not; nothing closed beyond the grip level before a measured full open; after one, a close on an object ended in the light hold; a STOP, stale controller data, a lost robot connection and a drive that the grip did not end each ended the jaw motion in a light hold; without a fresh live view the jaws opened and did not close; with `human.trigger_jaws: false` the trigger did nothing; a jaw opening outside 0 to 1, or no number, was refused; without a known grip level the trigger moved nothing; the hand-back was refused until the trigger had opened the jaws fully |
| The trigger in the human program and on the console, and a person opening the jaws an agent left closed | By tests: the logic on stand-ins (`test_human_logic.py`); the script `grasp` on the real retargeting graph, pass by pass against a stand-in for the Thor (`test_human_feed.py`, `test_human_program.py`); the console's jaws line (`test_console.py`); and one test of `test_demo2_programs.py`, in the test process and on the machine's clock, on APAI's software arm with the robot link service, the console, the live view and the human program attached | The jaw opening was 1.0 minus the trigger's value, the grip squeezed or not, fully open at or below 0.05 and closed at or above 0.95; no jaw opening went out on a pass without a usable trigger, or while ROBOT_STATE did not show the person as the control owner; the trigger changed nothing else of a HUMAN_SETPOINT; the note said which way the trigger goes while the arm followed the hand, behind every note about the arm; `grasp` asked for closed and then for open jaws and moved no joint; the console printed the jaws line below its status line. With the programs together: an agent client closed the jaws on an object half the jaw travel wide and asked for help; through the takeover and a lift of 1 cm with the trigger released the jaws got no command, and the note read `pull the trigger to the grip level to move the jaws`; the trigger, pulled all the way, reached the grip level 0.00 and moved nothing; released, it opened the jaws in ten steps of 0.05 until they were measured fully open and empty; after the typed `rest` right A gave the arm back. The test takes about 14 s |
| A takeover with a low gripper tip, and the lift | By tests (`test_lift_out.py`), in the test process, on a fake clock, on APAI's checkers and its software arm with APAI's `mock` profile | A takeover was granted with the gripper tip under the table clearance, after a grasp and with no grasp region, and its line ended `; the gripper tip is low: only a lift is taken until it is clear`; with another complaint of the human checker (beside the workspace, under the table plane, outside the joint box) it was refused; the typed `rest` stayed refused there; a person's lift, step by step, brought the tip clear, and `rest` was then taken; a target and a step that would have lowered the tip were refused with `lift the gripper: ...` and the arm held; the tip was held against where it was measured, not where the drive began; a slide under the table clearance was refused by APAI's human checker, and one above it left the region; the simulated arm's checkers knew no table |
| Every stretch of time in a sample age and in a view age | By tests (`test_freshness.py`), in the test process, on the simulated arm and a fake clock, with the test's client in the role `human` once on the robot connection and once at the robot app's own socket; one test runs the human program's `main()` | Controller data that were 0.14 s old when sent and 0.12 s on the way started no drive: no joint target went out, the person kept the arm, and `health.human.refusal` read `no fresh controller data: 0.26... s old, the limit is 0.25 s`. A takeover was refused just outside the limit and taken just inside it, wherever the time lay: at the human program, on the way, or counted from the ROBOT_STATE for a sender without `sent_mono_s`. The wait at the robot app counted too. A drive ended in a hold when its targets, or the view's reports, took too long on the way. A HUMAN_SETPOINT that names no time had no known age; a time from the future was not trusted. A VIEW_STATUS delayed on its way was as old as it was, and one that does not say when it was sent was not fresh; the view age at the Thor was the picture's age whatever the link took. The human program stamped each HUMAN_SETPOINT with the age and the time of the moment it sent it |
| A torque release whose driver call fails | By tests (`test_release_failed.py`), in the test process, on a fake clock, on the simulated arm and on APAI's mock driver | The RESULT was `failed` with `fault: torque release failed, the motors may be on or off: ...`; the motion was `release_failed`, `torque_on` true, and a fault-stop latched; no line said that the motors are off. After `reset` a task, `open` and `rest` were refused with `a torque release failed and the motors may be on or off; ...` and nothing was sent to the arm or the jaws. A shutdown started no return to rest, tried the release again and left the robot app running while it failed; once the release worked, the same shutdown closed the app. A second release at measured rest ended `rest_unpowered`. An arm port that knew its motors were off was shown as `rest_unpowered`. With APAI a release that failed after the driver had disconnected was not shown as off, and the arm port counted a release only once the driver's call had returned |
| A jaw command the agent gave up on, and a GRIPPER's expiry | By tests (`test_gripper_timeout.py`): the real robot app on the simulated arm with the agent's arm adapter, its wait shortened to 0.3 s, on a robot connection that keeps link messages back; the arm adapter against a scripted robot side; the robot app on a fake clock with the test's client | A GRIPPER that arrived after the adapter had given up was `refused` and the jaws got nothing; a close that ran when the adapter gave up was cancelled and left in the light hold, never opened; the adapter sent one STOP from `agent`, also inside a skill, and its next jaw command sent nothing; a jaw command answered in time sent no STOP and the agent kept the arm. A GRIPPER past its expiry was refused at once with `the jaw command expired ... s ago`; one that expired while it waited never started, and the commands before and after it ran; one inside its expiry ran as one without; a GRIPPER without the field never expired; the agent program's GRIPPER carried the end of its own wait |
| A MOVE's expiry at its first setpoint, and the read a move completes on | By tests (`test_executor.py`), on a fake clock with joint reads the tests script | A MOVE that expired during the check of its first setpoint was refused with `the move expired 0.030 s ago` and sent nothing, with and without a permit held; one whose check also made the setpoint late was refused as expired, not ended `late`; a STOP during that check won; a slow check that ended before the expiry started the move, and an expiry that passed in a later check did not end a move under way. A joint that read near and read not valid later did not count; a full read with a joint not valid, a failed one and one with a joint 0.05 rad off did not complete the move, which completed on the next full read; full reads that stayed bad ended it `not_settled`, the arm holding at the last setpoint with its motors on; a move that completed made one full read after its last setpoint and took as long as before |
| The arguments of the experiment's launcher in the agent program | By tests (`test_agent_options.py`), in the test process; one of them with the experiment's own timing, model transcript, debug recording and milestone batching on APAI's mock scene, with no GPU, no model request and no camera | The full stack timed and transcribed by default, and the minimal stack did neither; the launcher's command line with `--operate` and `--depth-backend ffs` gave the run that the line without them gives; each argument a run cannot take ended the program with exit code 2 and one line, `--motion-approval tool` on both stacks among them; without an option nothing of the experiment was loaded and no file was left; each option installed its hook once, in the launcher's order; a STOP and the close of the link came before any recording ended; a recording that could not begin ended the program before it attached, and one that could not end left the run's exit code as it was. On the mock scene every timing hook of the experiment installed |
| `bash run_workstation.sh --print`, and the script with an unknown argument, without a variable and with a Python that cannot be run | By a test, as processes (`test_link_programs.py`) | Exit code 0 and the five commands of the five panes, nothing started; exit code 2 and one line each |

Measured once in that environment, apart from the tests: one pass of the retargeting graph
took 0.21 ms, an IK solve that converged 0.11 ms and one that failed after 50 iterations
0.82 ms.

Never run:

- The agent program with a task at a real terminal: a test runs it as a process on a
  pseudo-terminal and types the task and the `yes`; the other tests script them. No task was
  run by hand on one machine.
- The operator console and the live view by a person: no line was typed at a real terminal
  and no page was opened in a browser. So the page's script, which carries the view token
  and writes `HUMAN DRIVING` and `THIS PAGE WAS REPLACED: reload it`, ran nowhere, and no
  bell was heard.
- The recipe of demo 1 with the mock agent by hand: with the ports 47610 and 47620, the
  default socket path, `run_console.sh`, `run_view.sh` and a browser.
- The recipe of demo 2 on one machine and its variant with the agent failing first by
  hand: with the ports 47610 and 47620, the default socket path, `run_console.sh`,
  `run_view.sh` and a browser. Tests ran both as processes.
- A headset, CloudXR and the Teleop Session Server with any of this, and so controller
  data from a headset, right A and right B over a remote session, and a recentre.
- The human program on a Thor, and its remote session between two machines: the tests
  attach it at the robot app's own socket on one machine, with the synthetic controller.
- The trigger of a real controller on real jaws, and the lift of a low gripper tip by a
  hand.
- The typed `takeover` and `handback` at a real terminal.
- Human driving on the rig, and anything of part F of the checklist. With the rig's robot
  file and APAI's `rebot_rs` profile a test runs a takeover, a lift of 3 cm, the return
  to rest and the hand-back on a stand-in for the motor library; the rig's own profile
  was not loaded, and sideways motion, a turned hand and the wrist camera rule under
  larger motions are checked nowhere.
- `experiment.stack: full` with a task, and its cloud model. The model client is not
  installed. `test_gpu_stack.py` starts the full stack on recorded stereo pairs with
  stand-ins for both model clients and runs no task; the other tests cover only the merge
  of its configuration files.
- `--camera link` through the agent program, `--source link` of the live view, and the
  camera receiver's start script with a receiver. The camera frame bridge is tested against
  a stand-in for the receiver.
- `depth_backend: ffs` on stereo pairs from the camera receiver: it ran on a recorded stereo
  pair only.
- `--driver real` on the arm, `run_robot_app.sh` on a Thor, and an Isaac Capture build on a
  Thor.
- The agent program's options of the experiment's launcher on the real full stack: the
  timing, the model transcript, the debug recording and the milestone batching with a
  GPU, the obstacle grid bridge and a model client.
- `run_workstation.sh` with tmux: a test runs it with `--print` only.
- A torque release that fails, a jaw command without its RESULT and a MOVE that expires
  on the real arm: the tests make each on the simulated arm, on APAI's mock driver or
  on scripted joint reads.
- The rig files with their programs: only loaded and checked by tests.
- Anything between two machines, with a camera on a Thor, or on a real arm.

## Next phases

| Phase | What is built |
|---|---|
| P1 | What is left of it: a task on the full stack, the camera connection from the Thor |
| P2 | What is left of it: the first run on the rig, by the checklist above, and demo 1 on the arm |
| P3 | What is left of it: the first run with a headset and demo 2 on the arm, by part F of the checklist, with the human program on the Thor; the two one-machine recipes by hand; what was cut from the human path |
| P4 | Acceptance of both demos: without hardware by the tests and the two recipes, on the rig by the checklist with every value written down. Then what the design names for P4: every fault under each control owner, measured delays, launch and recovery instructions |

[design]: ../../docs/design/claude_agentic_remote_execution.md
