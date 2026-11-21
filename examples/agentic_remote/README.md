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

Where this README says that something was not run on the arm or with a headset, it speaks
of the tests and of the time each part was written. A first session on the rig, on
2026-11-16, started the programs on both machines with this README as its only guide.
**One thing ran on the real arm**, as the rig owner reported on 2026-11-16: with the agent
program on the workstation and the robot app on the Thor, a pick, a lift, a hold and a
place of an object, at 0.1 rad/s. That was before the changes made later that day, which
have not run on the arm: the `rest` that closes the jaws and switches the motors off, `on`,
the jaw commands as routines of APAI's driver, the agent's speed of 0.2 rad/s, the
start-pose retry and the wait for a late permit. A person's limits on the rig, the rig
owner's values of 2026-11-18 (1.6 rad/s a joint, a floor 0.5 cm above the table plane and
a tip box of their own), have not run on the arm either. Every other statement that
something was not run on the arm stands as it was.

**Where to start reading.** To try the example on one machine, read
[Run demo 1 on one machine](#run-demo-1-on-one-machine) and the two recipes after it. To
get both machines of the rig running, in order, read
[Set up and start the rig](#set-up-and-start-the-rig): it ends with a table of lines the
programs printed in that first session, what each means and what to do. The operator's
steps for a session on the real arm are in [First run on the rig](#first-run-on-the-rig).

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
  were last measured full open. Jaws that were told to close and read at or below the closed
  line (`gripper.closed_frac`, 0.04 of the jaw travel on the rig) count as closed on nothing,
  however far short of the closed stop the close ended. The cost: something thinner than
  that line between the jaws counts as empty too.
- **Operator console**: the workstation program at which the operator reads the robot's
  state and types `stop`, `rest`, `release` and the other commands.
  **Console command**: a
  second program that sends one of the console's commands and ends.
- **Live view**: the page of the experiment's viewer (`run_viewer.sh`) with its live
  picture showing, in a visible browser tab. The viewer tells the robot app how old the
  picture on that page is, and a person's takeover needs it on the rig. See
  [The live view](#the-live-view).
- **Thor terminal**: the terminal on the Thor in which the robot app runs. The operator at the
  robot types there.
- **Demo 1**: the agent completes a task on the arm while an operator watches and
  can stop the arm or send it to rest from the console. The operator watches the arm
  itself or the viewer's live picture.
- **Demo 2**: when the agent fails, a person takes the arm over with the right
  controller of a headset, moves it, ends at rest and gives the arm back, and the agent's
  next task runs. The person is prompted by a help request when the agent got stuck, was
  blocked, ran out of steps or raised an error; after a task the model itself ended
  without success the agent has given the arm back, and the person takes it from nobody.
  The terms of the human path (takeover, hand-back, clutch, sample age and
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
- **The experiment's viewer**: `thor_viewer.py` of the experiment folder, a web page for a
  browser on what a run recorded: the obstacle map, the measured joints, the poses APAI's
  checker refused and the timing. `run_viewer.sh` starts it on the workstation, with the
  camera's **live picture** on the same page: the camera's picture as it arrives, shown and
  not stored. With the live picture showing, that page is the live view.

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
- The Thor terminal (`terminal.py`): `stop`, `rest` (asks for a typed `path is clear`; the
  arm returns to rest, the jaws close and the motors go off there), `release` (jaws
  closed and motors off, only at measured rest), `on` (the motors on again after a rest
  or a release; asks for `yes`), `reset` (clears a fault-stop), `open` and `close` (the
  jaws), `quit` (return to rest, motors off there, exit; it asks for `path is clear`
  when the arm has to move), `reference` (see [What the jaws do](#what-the-jaws-do)) and
  `help`; see [What the operator's commands do](#what-the-operators-commands-do). Its
  `install_signal_handlers()` makes
  Ctrl+C a STOP; the self-tests do not call it, the program started with `--connect` does.
- The agent side (`agent/`): the agent program, its arm adapter, the check ahead, the camera
  input and the console command. See
  [The agent side on the workstation](#the-agent-side-on-the-workstation).
- The operator console (`console/`). See [The operator console](#the-operator-console).
- The live view: what `viewer_live/` adds to the experiment's viewer, which is its live
  picture, its live arm and its word to the robot app on how old that picture is on the
  page. See [The live view](#the-live-view).
- The start of the experiment's viewer on the run folders of the agent program
  (`run_viewer.sh`); the viewer's code stays in the experiment folder. See
  [The experiment's viewer](#the-experiments-viewer).
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
- No camera frame from a Thor has reached the camera input or the viewer's live picture
  in a test. Both are tested on recorded stereo pairs and on stereo pairs the tests make up.
- No headset was connected. Every takeover and every motion of a person in the tests comes
  from a synthetic controller or from link messages a test makes; see
  [What is not tested without a headset](#what-is-not-tested-without-a-headset). Human
  driving is switched off in `configs/robot_app.yaml` and in the rig's
  `configs/robot_app_rebot.yaml` (see
  [The settings that ship off](#the-settings-that-ship-off)).
- No arm motion on hardware by a test, and one by a person: the rig owner's pick, lift,
  hold and place of 2026-11-16, at 0.1 rad/s, named at the top of this README. The tests
  run `--driver real` only on APAI's driver class over a stand-in for the motor library. Every value
  in `configs/robot_app.yaml` is a demo value for the simulated arm, and every value in the
  rig files is a first value. With the shipped `configs/robot_app.yaml`, `--arm apai` stops
  at the human limit set: its 1.5 rad/s is above the speed cap of APAI's profiles (see
  [What a move does](#what-a-move-does)).
- No model client: the program the full stack asks its cloud model through is installed
  apart from the workstation environment's packages, and its login is the user's own (see
  [The model client on the workstation](#4-the-model-client-on-the-workstation)). No test
  asks a model. What ran with the agent on the rig is named at the top of this README.
- The robot connection is not encrypted and the peer proves nothing but its address: the
  service takes the robot connection only from the configured addresses. Both machines are
  meant to be on one trusted network. The teleop connection of demo 2 has less: the
  Teleop Session Server is not encrypted either, checks no address and serves one client,
  whichever connects first, from any address that reaches its port 47600. Open that port
  only toward the Thor.

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
a STOP and never ends the program; `quit` ends it (return to rest, motors off there). SIGTERM
closes it and never moves the arm: with the arm at measured rest and the jaws proven empty
it first closes the jaws and switches the motors off there, anywhere else it leaves the
motors on (see [What the operator's commands do](#what-the-operators-commands-do)). A
terminal that closes ends neither the start nor the
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
session line. Without the argument there is no such socket. A PATH that begins with `~`
is taken from the home folder by the robot app itself, in every form of the argument:
`--human-socket ~/.agentic_remote/human.sock`, `--human-socket=~/.agentic_remote/human.sock`
and the same path in quotes name one socket, the one that `socket_path:
"~/.agentic_remote/human.sock"` of the human program's file names. A relative PATH is
taken from the working folder of each program: write an absolute path or one that begins
with `~`.

### The local API and its roles

A local client opens the Unix socket (`socket_path` of the service's file; by default
`$XDG_RUNTIME_DIR/agentic_remote/robot_link.sock`, else `/tmp/agentic_remote-<uid>/robot_link.sock`,
in a folder of mode 0700), says LOCAL_HELLO with its role and from then on sends and receives
the same link messages as on the robot connection. In Python `connect_local(path, role, name)`
(`link_service/local.py`) gives a carrier, so the code above it is the same on both sides.

| Role | Who | May send | At a time |
|---|---|---|---|
| `agent` | The agent program's arm adapter | MOVE, GRIPPER, PERMIT, CHECKER_STATE, STOP, and OWNER_REQUEST "agent", "help_needed", "nobody", "motors_off", "motors_on", "park" | One |
| `console` | The operator console | STOP, RESET, RELEASE, OPEN, SHUTDOWN, and OWNER_REQUEST "rest", "nobody", "motors_on"; with `operator.typed_takeover: true` in the service's file also OWNER_REQUEST "human", "handback" | Several |
| `viewer` | The experiment's viewer with its live picture (`run_viewer.sh`) | VIEW_STATUS, STOP | Several |
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
- **`agent_max_joint_speed` in WELCOME** is the joint speed limit of the robot app's
  agent limit set (`limits.agent.max_joint_speed`), in rad/s: a setpoint of the agent
  that asks for more is refused. The agent program never plans faster than it.
- **`human_max_joint_speed` in WELCOME** is the joint speed limit of the human limit set
  (`limits.human.max_joint_speed`), in rad/s: the robot app follows a person's joint
  target at 0.98 of it, and the human program moves that joint target at 0.9 of that.
- **`human_tip_box` in WELCOME** is the box the robot app's human checker holds a
  person's gripper tip in: its lowest and its highest corner, in metres, in the arm's
  base frame. With `--arm apai` it is the box the human limit set names
  (`limits.human.workspace_min` and `workspace_max`), or the workspace box of APAI's arm
  profile where the file names none; the simulated arm's checkers have none, and the
  field is empty. The robot app only names it; the checker enforces it. The human
  program keeps the gripper tip 1 cm inside it (see
  [How fast the arm follows the hand](#how-fast-the-arm-follows-the-hand)).
- **`human_tip_floor` in WELCOME** is the human checker's table clearance as a height,
  in metres in the arm's base frame: the table plane plus
  `limits.human.table_clearance_m`, or, where the file does not name that key, plus the
  higher of `rest.table_clearance_m` and APAI's `safety.table_clearance`. Under it the
  human checker takes only a step that rises. The simulated arm's checkers know no
  table, and the field is empty. The human program stops the gripper tip 0.5 cm above
  it.
- **`human_joint_box` in WELCOME** is the joint box the robot app's human checker holds
  a person's joint target in: its lowest and its highest corner, one value per joint,
  in rad (`limits.human.joint_min` and `joint_max`). The human program stops its joint
  target 0.01 rad inside it. These four fields are optional and only name what the
  human checker enforces as before: a robot app without them is still read, a program
  that does not know one leaves it out, and the protocol version did not change for
  them.
- **A ROBOT_STATE behind every setpoint of a person's drive.** While a human drive
  sends setpoints the robot app sends a ROBOT_STATE after each, 50 a second, and the
  timed one (`state.publish_period_s`) comes again a period after the last of those.
  The human program reads from them how far the Thor's joint target has come. The
  agent's moves and a holding arm get ROBOT_STATE as before.
- **`motion` in ROBOT_STATE** is the motion state, one of `idle`, `moving`, `hold`,
  `rest_hold` (at rest, motors on), `rest_unpowered` (at rest, motors off) and
  `release_failed`. The last one follows a torque release whose driver call failed: the
  motors may be on or off, `torque_on` stays true, and nothing moves the arm until a
  release has worked or the robot app was started again (see
  [the torque rule](#the-rule-torque-is-never-cut-away-from-rest)). In `rest_unpowered`
  the robot app runs on: `on` switches the motors on again without a new start.
- **`torque_on` and `health.motors` in ROBOT_STATE.** `torque_on` says that the robot
  app switched the motors on and has not released them; it is no reading of the motors.
  `health.motors`, from an arm port that can say it (`--arm apai`), holds `state`
  (`holding`, `not holding` or `unknown`) and `reason`: what the joint readings and the
  motors' fault reports say of the hold. The console's status line puts the two
  together (see [The operator console](#the-operator-console)).
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
- **VIEW_STATUS** is the one link message the experiment's viewer sends: `live`, `age_s`
  and, while the view is live, `sent_mono_s`. It is live only while the viewer's page
  shows its live picture in a visible browser tab (see [The live view](#the-live-view)).
  The robot app stores it and reports it back as
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
  `enabled` (the setting `human.enabled`), `live_view_required` (the setting
  `human.require_live_view`), `takeover_open`, `driving` (a human drive runs),
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
  are not where it asks: held wider, or stalled short of an opening).
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
  moves nothing then where the robot app's file requires the live view, as the rig's
  files do, because VIEW_STATUS comes over the robot connection (`human drive ended: the
  live view is not fresh (...)`); a takeover asked for meanwhile is refused for the same
  reason. Where the file requires no live view, a new squeeze moves the arm
  again and a takeover is taken, without a robot connection. The
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
[The human program on the Thor](#the-human-program-on-the-thor)). What each argument
does, and where `PROFILE` and `NAME` come from, is in
[Start the robot app on the Thor](#9-start-the-robot-app-on-the-thor).

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
the start, as a SIGTERM at any other time does: the motors go off only if the arm is then
measured at rest with the jaws proven empty, and stay on otherwise. After a terminal that
closed the robot app runs on without its lines.

After the connect the program prints `robot app: the real driver is connected at measured
rest: torque on, hold unknown, motion hold, jaws referenced` (or `motion rest_hold`) and
dials the service. It prints the motors' state, the motion state and the jaws, not the
joint values. `hold unknown` is the word right after a connect: no joint target was sent
yet (the six texts are under [The operator console](#the-operator-console)). APAI writes the
joint reads of its connect to `~/.apai/logs/rest_readings.jsonl`. An arm without power can
lie farther from the rest pose than a release allows (0.05 rad); the rest readings recorded
on this arm lie up to 0.08 rad away. The motion state after the start is then `hold`: a
task is taken from there, and a `release` needs a `rest` first.

**How long a joint read takes.** After each connect the robot app measures its first 30
idle joint reads, which it makes anyway, and writes one line to its log: `one joint read
takes X ms on the motor bus (median of 30 idle reads; worst Y ms): the bus allows about N
joint reads a second; state.idle_read_period_s asks for 100`. When one read takes longer
than `state.idle_read_period_s`, a warning follows: the idle reads then follow each other
without a pause, and a link message waits for the read under way.

`--driver mock` (the default) is APAI's software arm and asks nothing; a profile that drives
hardware is refused without `--driver real`. `--arm apai` without `--connect` connects
nothing: it builds the checkers and the driver object, prints what would run, among it the
profile's speed cap, and ends with exit code 2.

`run_robot_app.sh` sets the environment on the Thor and then runs
`python -m isaaccapture_examples.agentic_remote.robot_app` with its arguments. It needs
`AGENTIC_REMOTE_THOR_PYTHON` (the Python of APAI's environment on the Thor),
`AGENTIC_REMOTE_APAI_SRC` and `AGENTIC_REMOTE_ISAAC_CAPTURE_PY` (the folder of the
Isaac Capture Python package built on the Thor for that Python); without one it prints one
line and exits with code 2. With `--env FILE` as its first argument it reads them from
a file of `KEY=value` lines, and the robot app does not get that argument:
[step 3](#3-the-variables-of-each-machine-in-one-file) of "Set up and start the rig"
shows the file. It unsets `APAI_ROBOT`, `APAI_BASE` and `APAI_ARM_HOLD_DIR`, so
the hold markers are read from APAI's default folder.

## The agent side on the workstation

The agent side is what runs beside the robot link service: APAI's agent, its camera input and
its obstacle check. The operator's programs run beside it: the console, the console command
and the experiment's viewer, whose page is the live view. All of them reach the arm only
through the local API. The human
program, through which a person drives the arm in demo 2, runs on the Thor on the rig; in
the one-machine recipes it runs beside them and reaches the arm the same way. Nothing under
`agent/` imports Isaac Capture or a motor driver, and APAI is not edited: the agent program replaces names of APAI's modules while it runs and puts them back when it
ends, also when one of its start steps fails.

### What runs where

| Program | Machine | Started with |
|---|---|---|
| Robot app | Thor | see [The robot connection](#the-robot-connection) |
| Camera sender | Thor | the camera streamer's `argus_sender` with `configs/camera_sender_rebot.yaml`, after the rig's camera bring-up (see [The camera, from the Thor to the workstation](#8-the-camera-from-the-thor-to-the-workstation)) |
| Robot link service | Workstation | see [The robot connection](#the-robot-connection) |
| Camera receiver | Workstation | `./run_camera_receiver.sh --host THOR_ADDRESS`: the address is the Thor's, which the receiver dials, not the workstation's own |
| Agent program | Workstation | `./run_agent.sh --config configs/agent.yaml --interactive` |
| Operator console | Workstation | `./run_console.sh` |
| The experiment's viewer | Workstation | `./run_viewer.sh`; its page is the live view (see [The experiment's viewer](#the-experiments-viewer) and [The live view](#the-live-view)) |
| Human program | Thor on the rig; the one machine of the one-machine recipes | On the Thor `bash run_human.sh --config configs/human_rebot.yaml`, beside a robot app started with `--human-socket`; on one machine `bash run_human.sh --config configs/human.yaml` (see [The human program](#the-human-program)) |
| Teleop Session Server, with the headset's streaming service | Workstation, on the rig only | Isaac Capture's own programs, for demo 2 (step F1 of [the checklist](#first-run-on-the-rig)) |
| Console command | Workstation | `python -m isaaccapture_examples.agentic_remote.agent.console_cmd COMMAND` |
| The console, the agent program, the robot link service, the camera receiver and the experiment's viewer of demo 1 on the rig, together | Workstation | `bash run_workstation.sh --env FILE --thor THOR_ADDRESS`: one window of tmux, a program that splits a terminal into panes, with a pane for each of the five (see [The start scripts](#the-start-scripts)) |

With `--camera link` the agent program starts the camera frame bridge itself, under the
interpreter `camera.bridge_python` names. The experiment's viewer starts a second one for
its live picture. The
camera receiver, the bridges and the agent program use one GPU: index 0 in PCI bus order.
`run_agent.sh`, `run_human.sh`, `run_viewer.sh`, `run_camera_receiver.sh`
and the agent program set `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=0`
themselves; the human program and the experiment's viewer need no GPU. The viewer's live
picture has a camera frame bridge of its own, on the same GPU.

### The agent program

```text
python -m isaaccapture_examples.agentic_remote.agent --config PATH
    [--task TEXT] [--interactive] [--max-steps N]
    [--camera link|mock|recorded:<dir>] [--check] [--run-dir PATH] [--log-level L]
    [--debug-map-hz HZ] [--debug-at-limit drop-oldest|stop]
    [--debug-idle-map-s SECONDS]
    [--batch-milestones] [--record-debug] [--debug-max-gib GIB] [--no-timing]
    [--motion-approval task] [--operate] [--depth-backend NAME]
    [--speed-profile conservative|reference] [--mounting PATH]
```

- `--task TEXT` runs one task and ends; an empty text is refused. `--interactive` reads one
  task after another at the prompt `task>`; an empty or blank line is read again, and
  only the end of the input (Ctrl+D) ends the program. With neither, the full stack runs
  the default task of the experiment's launcher, which only looks: `Observe the scene and
  describe what you see. Do not move the robot.`; the minimal stack runs APAI's own
  (`look at the table and report what objects you see`). With the full stack the task of
  `--task`, and that default task, get the launcher's note appended: ` Camera depth is
  stereo-derived; mounting fit accepted, independent metric accuracy not yet verified.`
- **When the program ends after a task used the arm, it asks once about the arm**: see
  step 10 of [A task from its start to its end](#a-task-from-its-start-to-its-end).
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
- `--debug-map-hz HZ` (default 2) is how many pictures of the obstacle map the debug
  recording of `--record-debug` stores a second; `--debug-map-hz 0` stores every one. It
  decides only what is written: the obstacle map, the checks and the arm are the same at
  every rate. It is the agent program's own argument, for the full stack, and without
  `--record-debug` it does nothing. What is stored at which rate, and how long the
  recording then lasts, is in the bullet on the disk under
  [The options of the experiment's launcher](#the-options-of-the-experiments-launcher).
- `--debug-at-limit drop-oldest|stop` (default `drop-oldest`) is what the debug recording
  does when it reaches `--debug-max-gib`. With `drop-oldest` its oldest map pictures are
  dropped to make room, so the recording goes on with the newest stretch and the
  experiment's viewer keeps following. With `stop` the recording ends there, as the
  experiment's own does, and the viewer no longer updates. Like `--debug-map-hz` it is the
  agent program's own argument, for the full stack; the same bullet on the disk tells what
  is dropped and what never is.
- `--debug-idle-map-s SECONDS` (default 10) is how far apart the pictures of the obstacle
  map are that the debug recording stores **while the program waits for a task and
  nothing happens at the arm**: the program is at the prompt `task>`, before the first
  task or between two tasks, nobody owns the arm and it does not move. The map refreshes
  all the time, also while nobody gives a task; without this rule minutes of an unchanged
  scene filled the recording. `--debug-idle-map-s 0` stores none then. A person who takes
  the arm over during that wait, as in Demo 2, is recorded at `--debug-map-hz` from the
  first ROBOT_STATE that shows an owner until 5 s after the hand-back, and so is any
  motion of the arm. The
  moment a task line is read the recording goes on at `--debug-map-hz`, and the first map
  refresh behind the line is stored at once. A question in the middle of a task (the typed
  `yes`, the question about the motors) is no such wait. What the arm did or was refused
  is stored as always. It decides only what is written; it is the agent program's own
  argument, for the full stack, and it is refused together with `--debug-map-hz 0`, which
  stores every map picture, also during the wait.
- The nine arguments of the last three lines are those of the experiment folder's own
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
| 1 | No robot link service, the robot app not ready in time, another camera calibration or arm profile than the robot app's (at the start, or before a later task after the robot app was restarted), a per-joint list of the arm profile or of a `rig` block that is not of the arm's joint count, APAI's runtime not built (for example, with the full stack, no camera frame within 20 s; with the minimal stack, no usable obstacle grid within `task.grid_wait_s`), a task that ended without success (`--interactive`: any task of the session), an owner request at a task's end that was not answered, the agent still control owner when the program ended, invented exposure times or the minimal stack against a robot app that declares no simulated driver (at the start, or before a later task after the robot app was restarted), a task whose session has no WELCOME within `task.owner_wait_s`, a skill stop that was not cleared (the program said `restart the agent program`), or SIGTERM: a STOP was sent first |
| 2 | The arguments, `agent.yaml` or the environment cannot be used, APAI cannot be imported, or a recording of the experiment cannot start. Nothing was attached |
| 130 | A second Ctrl+C. A STOP was sent again first |

A task the operator cancelled with Ctrl+C does not count as a task without success. Two of
the causes of exit code 1 are read from the code and were not run: a robot app that is not
ready within `link.connect_timeout_s`, and an obstacle grid that is not usable in time.

**What the full stack does at its start**, as the experiment's launcher: it saves one camera
frame into the run folder (`startup-rgbd.npz`, `startup-rgb.jpg`, `startup-depth-m.npy`,
with `runtime-status.json`), which fails when the camera gives none within 20 s; it says
the motion bounds once; and when the obstacle grid is not usable after `task.grid_wait_s`
it goes on with the line `the obstacle grid is not usable yet: the program goes on, and no
move starts without a fresh obstacle grid`. The minimal stack ends there instead. The
motors are not touched at the start: the robot app on the Thor has switched them on.

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
| `--record-debug` | The experiment's **debug recording** in `<run folder>/debug`: what went into the obstacle grid (camera frames, depth), masks, map revisions and the joint feedback the program read. [The experiment's viewer](#the-experiments-viewer) shows a run only with it. It needs the timing: with `--no-timing` or with `experiment.timing: false` the program ends with exit code 2 and `--record-debug needs the timing; ...` |
| `--debug-max-gib GIB` | The most the debug recording takes on disk (16). When it gets there, the agent program drops its oldest map pictures and the recording goes on; with the agent program's `--debug-at-limit stop` it stops there, as in the launcher. It must be a finite number above 0. Without `--record-debug` it does nothing |
| `--no-timing` | This run goes without the experiment's timing |
| `--motion-approval task` | One typed `yes` before the first skill that moves in each task (the launcher calls such a skill a motion tool). The program does that with or without the argument. `--motion-approval tool`, a `yes` before every such skill, is not built: it ends the program with exit code 2 and `--motion-approval tool is not built: ...` |
| `--operate` | Accepted: every run of the program may move the arm, after the typed `yes`, with or without it. As in the launcher it needs the operator's terminal: without one, or inside the environment of an AI agent, the program ends at its start with exit code 2 and `--operate requires the operator's interactive terminal, outside an agent environment` |
| `--depth-backend NAME` | The stereo matcher of this run, in place of the one the experiment's `configs/sensing_thor.yaml` names (`ffs`). With the full stack it must be one of the experiment's: another name ends the program with exit code 2 and `the stereo matcher '...' is not one of the experiment's (...)` |
| `--speed-profile NAME` | Full stack only: the joint speed APAI plans the agent's moves at, by the name of a **speed profile** of the experiment's launcher, `conservative` or `reference`. `reference` (the default) is `overrides.safety.max_joint_vel` of the experiment's `configs/rebot_workstation.yaml`, at most the cap of APAI's `rebot_rs` profile (0.8 rad/s); `conservative` is 0.1 rad/s. The agent plans at the lowest of three speeds: that one, `thor.agent_max_joint_speed` of the agent's file, and the Thor's own agent speed limit, which the robot app names in WELCOME. Once WELCOME is there the program says the speed, and where each of the three came from, in one line that begins `joint speed: the agent plans at` (see [The rig files](#the-rig-files)) |
| `--mounting PATH` | The **mounting record** whose hand-eye file this run uses: a file the experiment writes when a fit of the camera's place on the arm was accepted. Without the argument the file is `camera.hand_eye_json`, and with that key empty the experiment's newest record. A record counts only when its `accepted_fit` is true and its hand-eye file has the SHA-256 the record names |

- **Two recordings run in every run of the full stack**, as in the experiment's
  launcher: the **timing** (`timing.jsonl` in the run folder; `timing-report.txt` and
  `timeline.html` when the program ends) and the **model transcript** (`model-trace/` in
  the run folder: every request to the model and its answer). Their file keys are
  `experiment.timing` and `experiment.model_trace`, both `true` in `configs/agent.yaml`
  and `configs/agent_rebot.yaml`; `false` switches one off.
- **With `experiment.stack: minimal` none of this runs**, and `--batch-milestones`,
  `--record-debug`, `--debug-max-gib` and `--no-timing` are refused (exit code 2, `...
  needs experiment.stack full (it is minimal): ...`): the minimal stack loads none of
  the experiment's modules. The program's own `--debug-map-hz` and `--debug-at-limit` are
  refused the same way.
- **What the workstation cannot record.** The motors are read and commanded by the robot
  app on the Thor, and the camera captures there. So the motor reads and sends and the
  camera's capture are not timed: a move is timed from its MOVE to its final RESULT, and
  a stereo pair from the wait for it to its depth. The debug recording holds no motor
  commands and no single motor reads (`robot.call`, `robot.return`,
  `mask.state_source`). Its joint feedback (`robot.feedback`) is not the motors' own:
  it is the measured joints of the ROBOT_STATEs as they arrive, at most 10 a second, with
  the source `ROBOT_STATE over the robot connection`. The time of such a pose is the ROBOT_STATE's
  receipt on the workstation's monotonic clock less the age of the joint readings; it
  is no time of the motors, and the event says so in its field `clock`. A pose is
  recorded from ROBOT_STATE itself, not when APAI asks the arm for one; see
  [What a run records and measures](#what-a-run-records-and-measures). The program says
  this in its first lines, which begin `timing:`, `model transcript:`, `debug
  recording:` and `milestone batching:`.
- **The recordings start before the program attaches to anything.** The debug
  recording, or one of the program's own spans, that cannot start ends the program with
  exit code 2 and `the experiment's recordings cannot start (...); nothing was
  attached`. A `debug` folder that is already in the run folder is such a case, so one
  `--run-dir` takes `--record-debug` once. Two things do not end the program, by the
  experiment's own code: a hook of the timing whose target is missing is listed in
  `timing-status.json` of the run folder and the run goes on without it, and a model
  transcript that cannot be installed is one warning and the run goes on without it.
- **A STOP comes before any report.** The recordings stand around the whole robot
  connection: on Ctrl+C the STOP goes out and the local socket is closed before a
  recording writes its report. A recording that cannot end leaves the run's exit code
  as it was and says `a recording of this run did not end as it should: ...`.
- **Disk: at its size limit the debug recording drops its oldest map pictures and goes
  on.** The limit is `--debug-max-gib`, 16 GiB by default. Alone, the experiment's
  recorder stops for good when it gets there, and its viewer, which reads only the
  recording, stands still. The agent program keeps the recording under the limit
  instead (`--debug-at-limit drop-oldest`, the default): each time the next event would
  not fit, the pictures of the oldest stored map picture are removed from
  `<run folder>/debug/arrays`: the camera frame, the depth and the masks. So the
  recording holds the newest stretch and the viewer's `Follow live` goes on for as long
  as the session does. The agent program says it at its terminal when the limit is first
  reached, and once more with the next task prompt: `debug recording reached
  --debug-max-gib 16 after N min: from now on the oldest map pictures are dropped; the
  viewer keeps following, and times before HH:MM:SS (S s on its time line) show no
  picture. That time moves on with the run`. HH:MM:SS is this machine's clock time, and
  S the same moment in seconds on the viewer's time line.
  **Never dropped**: a row of `events.jsonl`, which is only appended to; an array from
  outside the watcher, so the route segments, the refusals,
  `verification.measurement`, the object masks a skill makes, `robot.feedback`,
  `robot.state` and the session's start; a map refresh that a refusal or a route
  segment names, or that was stored late for one or for a bridge call of its own; the
  first map of the session; and the four newest map pictures.
  **A dropped time in the viewer**, when the operator moves the time line back or
  presses `Play recording`: the obstacle map with its points and the arm as they were,
  and no camera picture. The viewer answers the request for that picture with `400`
  and `No such file or directory`, for that request alone. A refusal still shows its
  map and its picture. The map's points of a dropped picture stay until they take a
  quarter of the limit, about four hours of them at 2 map pictures a second; then the
  oldest go too. A time before that has no state: the page shows the error text of
  that request under the refusals and stands at that time until the time line is moved
  or `Follow live` is pressed. The stretch that has its pictures is about 30 min when
  the limit is first reached and about 22 min after four hours, for the points and the
  rows that stay; the rows take about 1 % of the default limit an hour.
  **With `--debug-at-limit stop`** the recording ends at the limit, as the experiment's
  does: from then on the viewer shows the run as it was at that moment, and the line
  is `debug recording full after N min (--debug-max-gib 16): the viewer no longer
  updates; the arm and the task are not affected. End the session and start the agent
  again for a new recording, or raise --debug-max-gib`. The same line comes with
  `drop-oldest` when nothing is left that may be dropped. Whatever the option, the
  experiment's recorder stops by itself with less than 2 GiB free on the disk: that
  line begins `debug recording stopped after N min` and names the reason.
  For a longer stretch of pictures give the agent program a larger limit, for example
  `--debug-max-gib 64` (about two hours), on its command line; in the pane `agent` of
  `run_workstation.sh` add it to the command before you press Enter. Each run with
  `--record-debug` takes up to its limit on disk, about 17 GB with the default, in
  `<state_dir>/runs/<run>/debug`, and keeps it when the session has ended. Nothing
  deletes old runs: delete the run folders you no longer need by hand, while no agent
  program writes into them.
- **What the debug recording stores of the obstacle map, and how much 16 GiB holds.** A
  **map picture** is one camera frame of APAI's watcher (the thread that keeps the
  obstacle map fresh): the map's refresh with that frame, `map.frame` to
  `map.refresh_end` with the bridge's calls and the arm's mask, and the detector's
  `object.mask` of the same frame. On the workstation the watcher takes about 7 a second
  and one map picture takes about 4.7 MB on disk (the session of 2026-11-17, camera
  1280 x 992: 17 GB in nine and a half minutes). The viewer's page asks for the state
  every 500 ms, so it can show 2 a second. The recording stores `--debug-map-hz` map
  pictures a second, 2 by default, each whole: 16 GiB then holds about 30 min, and
  `--debug-max-gib 64` about two hours. `--debug-map-hz 0` stores every one, as the
  experiment's launcher does: about 9 min. The 30 min and the two hours are computed,
  not measured: of the events of that session's recording the rate of 2 would have
  stored 5.3 GB of the 17 GB. The rig's second session of 2026-11-17 ran with the rate
  and stayed under the limit: its run `20261118T022210Z-3147525` stored 1528 map
  pictures in 13 minutes, in a recording of 8.1 GB. The
  line `debug recording:` at the start names that time and what happens when the limit
  is reached. The rate decides only
  what is written: the map takes every camera frame at every rate. The map the viewer
  shows is then up to about half a second older than the map the arm is checked
  against. **Stored whatever the rate**: every event from outside the watcher, so
  the route segments (`route.segment.begin`, `route.segment.end`), the
  refusals (`safety.refusal`), `verification.measurement`, the object masks a skill
  makes for a grasp, `robot.feedback`, `robot.state`, `robot.move_path` (the planned
  joint path of each MOVE) and the session's start; and the
  map refresh whose revision a refusal or a route segment names. When the rate had
  left that refresh out, it is stored just before the event that names it: each of its
  events then carries `recorded_late` with the time it happened (`event_mono_ns`), its
  thread and the event it was stored for, and the row's own time is the time it was
  stored. The three newest refreshes that were left out are held in memory for that.
  A refresh in which the map asks the grid bridge for something it does not ask on
  every refresh is stored the same way: `retire_target`, with which the experiment
  takes a grasped object out of the map, is such a call.
  **While the map stands still, and at the end.** When a pass of the watcher ends
  without a new map revision (the camera gave no new frame, or the arm's pose for the
  mask was not there) and no map was stored for a period of the rate, half a second by
  default, the newest map the rate had left out is stored; so is the newest one when
  the recording ends. `recorded_late` names `map.stall` or `map_rate.end` for them. So
  the map in the viewer is not older than about a second while the map stands still,
  and the last map of a session is in its recording.
  **While the program waits for a task.** The program is **idle** while it waits at
  the prompt `task>` for the operator's line, before the first task or between two
  tasks, and nothing happens at the arm: the newest ROBOT_STATE shows nobody as the
  control owner, and the arm does not move. The map refreshes all that time, and
  nothing in the scene has to change. So while the program is idle the recording
  stores one map picture per `--debug-idle-map-s` seconds, 10 by default, counted
  from the last map it stored: the viewer still shows a current map.
  `--debug-idle-map-s 0` stores none.
  **The arm moves** when ROBOT_STATE's `motion` is another word than those of an arm
  that stands (`idle`, `hold`, `rest_hold`, `rest_unpowered`, `release_failed`), when
  a measured joint is more than 0.005 rad from the pose the arm last stood at, or
  when the jaw opening is more than 0.02 of its range from where it stood. The noise
  of a standing arm stays far below that: in the run named below the joint readings
  spread over 0.00065 rad in 346 s at rest with the motors on, and the jaw opening
  over 0.00007. With the motors off the robot app reads no joint: ROBOT_STATE
  repeats the last reading. In a hold the readings went up to 0.0040 rad from where
  the hold began in six of that run's seven holds, and 0.028 rad in the first 0.6 s
  of the seventh: a hold that still settles counts as motion, for 5 s each time.
  **As soon as a ROBOT_STATE shows an owner (the agent, a person, recovery) or
  motion, the rate of `--debug-map-hz` is back**: the next map refresh is stored at
  once, whatever was stored just before, and the rate stays until the arm has been
  without an owner and without motion for 5 s while the program still waits. So **a
  takeover in [Demo 2](#demo-2-a-person-takes-over) is recorded at the normal
  rate**. There the agent's task ends with a help request, the program is back at
  `task>`, and a person takes the arm over, drives it and hands it back, all during
  that wait. From the first ROBOT_STATE with the person as owner to 5 s after the
  hand-back the recording stores 2 map pictures a second, also while the grip is
  released and while recovery returns the arm to rest. A STOP inside a takeover
  leaves nobody as the owner: once the arm has stood for 5 s the wait is idle, until
  the person takes the arm again. A return to rest on the
  timer, a hold in which the arm still settles, jaws moved by a typed `open` and an
  arm pushed out of a hold by hand are recorded the same way, and so are the
  5 s after the agent gave the arm back at the end of a task. An arm pushed by hand
  with the motors off is not seen, because no joint is read then: the wait stays
  idle. The 5 s count from
  the last ROBOT_STATE that showed an owner or motion, also from one before the
  wait: a wait that begins with an arm that has been quiet that long is idle from
  its start. **Without a
  ROBOT_STATE newer than 2 s**, with the robot connection down, the program does not
  know what happens at the arm and takes the safe side for the record: the rate of
  `--debug-map-hz`. What ROBOT_STATE does not show is not seen: a person who moves
  an object by hand while nobody owns the arm and it stands is recorded once per
  idle period.
  When the task line is read the rate of `--debug-map-hz` is back, and the first map
  refresh behind the line is stored at once, whatever was stored just before: a task
  begins with a fresh map in the recording. A question the program asks in
  the middle of a task, the typed `yes` or the question about the motors, is not idle,
  and neither is the question at the session's end. Everything that is stored whatever
  the rate is stored while the program is idle too, and a map that stands still is
  then stored once per idle period, not once per period of the rate. In the rig's
  second session of 2026-11-17 (run `20261118T022210Z-3147525`, 13 minutes, recorded
  at 2 map pictures a second) the program waited for a task for 211 s, 66 s before
  its first task and 145 s after its last. Nobody owned the arm then and it stood,
  and 423 of the 1528 map pictures the rate stored were stored then: 2.2 GB of the
  recording's 8.1 GB. With this rule 30 would have been stored, 10 of them in the
  5 s after the agent gave the arm back: about 2.0 GB less. A Demo 2 session of two
  minutes of the agent's task, a minute at the prompt, a takeover of three minutes
  and a minute at the prompt again would store about 630 map pictures, 3.3 GB at
  that run's 5.2 MB a picture, 360 of them during the takeover. That is computed
  from that recording's events and
  the waits in its `timing.jsonl`, not measured: no session on the rig has run with
  the rule yet.
  **What the recording says of itself.** Two events `recording.map_rate`: the rate at
  the start, with `idle_map_s`, a note on the rule for the wait and its four numbers
  (`arm_quiet_s`, `arm_state_age_s`, `arm_travel_rad`, `arm_jaw_travel`), and at the
  end how many map pictures came, were stored, were left out,
  were stored after all, and were named but no longer held. `recording-status.json`
  carries the same numbers as they stand, in a block `map_rate` (`map_hz`, `ticks`,
  `stored`, `skipped`, `rescued`, `missing`; and for the wait `idle_map_s`,
  `wait_seconds`, how long the program has waited for a task so far, `idle_seconds`,
  how long of that it was idle, `idle_skipped`, how many of the `skipped` map
  pictures were left out then, `idle_interrupted`, how often an owner, motion or a
  missing ROBOT_STATE ended a quiet stretch of the arm during a wait,
  `idle_interrupted_seconds`, how long the waits then went on at the rate, and
  `idle_interrupted_by`, those times by their cause: `owner`, `motion`,
  `no_state`), and what was dropped at the
  size limit in
  a block `dropped_at_limit`: `refreshes` and `bytes`; `first_time_kept`, the seconds
  on the viewer's time line from which every map has its picture; `points_dropped` and
  `first_time_points_kept`, the same for the map's points; and `named_after_drop`, the
  refusals and segments that named a map whose pictures were gone already. Both blocks
  are also there when the session ended at the limit. A map picture that the rate
  leaves out or the limit drops is not counted among the recorder's dropped events:
  those stay the events the recorder lost.
  **What `complete` means.** `complete` in that file is `false` as soon as one map
  picture was left out by the rate or dropped at the limit. The experiment's replay
  tools (`thor_replay_target_map.py`, `thor_replay_mask_margin.py`,
  `thor_replay_target_footprint.py`, `thor_replay_pickup_success.py`) rebuild the map
  from every recorded depth integration and start only on a recording that is
  `complete`: they refuse such a run, as they refuse one that lost events. Record a
  session that is meant for them with `--debug-map-hz 0 --debug-at-limit stop` and a
  limit the session stays under: the recording is then the experiment's own and
  `complete` means what it means there. The experiment's `thor_recording.py --point`
  reads the bridge's answers, which stay, and then the camera frame of one refresh,
  which may be gone. Every stored map revision holds the bridge's own answer, the map
  as it was.
- **`--check` with the same arguments** prints what a run would take under
  `run_options`. A run that takes any of it writes that into `run-options.json` in its
  run folder. With `--record-debug` both also name the map pictures a second, what
  the recording does at its limit and the seconds between two map pictures while the
  program is idle, as `debug_map_hz`, `debug_at_limit` and `debug_idle_map_s`; the
  last is `null` with `--debug-map-hz 0`, which has no rule for the wait.
- **What differs from the launcher's `--motion-approval task`.** As in the launcher,
  the `yes` also covers a return to rest inside the task (APAI's `move_to_rest`
  skill). Here the `yes` is followed by the robot app's grant of the arm, and the
  motors are switched on by the robot app on the Thor when it starts, not at the first
  skill that moves: the launcher's question `Connect the powered ReBot on can4? ...`
  is asked here only after a rest switched the motors off (step 4 of
  [A task from its start to its end](#a-task-from-its-start-to-its-end)).
- **Not taken**: the launcher's `--settings` is an unknown argument (exit code 2). The
  experiment's `configs/thor_runtime.yaml` and `configs/thor_cloud.yaml` are always
  read from the experiment folder.
- **What the full stack reads from the experiment folder in place of the agent's
  file**: the arm profile (the `arm_profile` that the experiment's
  `configs/thor_runtime.yaml` names, without the motor bus), the joint speed (the row
  `--speed-profile`), the camera's numbers and the stereo matcher
  (`configs/sensing_thor.yaml`), the two waits for an obstacle grid before a move
  (a third of `occupancy.max_age_s`, and `thor_motion.route_preflight_s`), and the
  hand-eye file when `camera.hand_eye_json` is empty (the row `--mounting`). The
  program's first lines name the arm and the hand-eye file with the first 12 digits of
  its SHA-256; the joint speed is said once the robot app's WELCOME is there.

None of these options ran on the real full stack, with its GPU, its obstacle grid bridge
and its model client. The experiment's own timing, model transcript, debug recording and
milestone batching ran in one test on APAI's mock scene, with no GPU, no model request
and no camera (`test_agent_options.py`). The experiment's viewer read such recordings in
the tests of `test_viewer.py`.

### A task from its start to its end

1. **Thinking and looking take no arm.** APAI's read-only tools (observe, list objects, recall
   and the like) run without ownership and without a question to the operator. The tools
   the model is offered are those of the experiment's launcher: its read-only tools and
   every skill of APAI that moves, the rest tool `move_to_rest` among them. With the full
   stack the model is also told: `Tools on this robot: the tool list you were given is
   complete. preview_grasp is not offered: grasp with grasp_object without a preview.`
2. **The first skill that moves asks the operator.** When an earlier task of the program
   has ended its ownership, the program first waits up to `task.owner_wait_s` (2 s) for the
   ROBOT_STATE that shows that ownership as ended: a task typed right after the last one
   would otherwise be refused on a ROBOT_STATE that still shows the agent as control owner.
   Then it reads ROBOT_STATE
   and refuses, without a question, when a person owns the arm or a takeover is open
   (`takeover not finished: give the arm back with right A at rest`; see
   [Demo 2: a person takes over](#demo-2-a-person-takes-over)), the control owner is not
   nobody or a fault-stop is latched. Such a refusal ends the task at once. Otherwise it
   prints one question that names the task, the first skill and the
   joint speed, and waits for a typed `yes` at the terminal the task was typed at. Any other
   answer refuses the skill and sends nothing. The question is never answered from a pipe, and
   never inside the environment of an AI coding agent. What was typed at the terminal before
   the question was written is dropped, so a `yes` typed ahead answers nothing. One `yes`
   covers the later skills of the same task, its retries and a return to rest inside it.
   A skill that was not let through gets the experiment launcher's answer, `<skill> not
   admitted in this session; no command sent: <why>`, and after a declined `yes` the task
   goes on: the model can report, and no help request follows. The same answer goes to a
   tool that is neither one of the read-only tools nor a skill that moves.
   **Motors that are off at rest refuse no task.** After the `yes` the program asks the
   launcher's question about the motors, `The robot app reports the motors off at rest.
   Connect the powered ReBot on can4? Stay beside the e-stop. On exit, return to rest
   through the safety checks, then release torque. Type yes to proceed: `, and on `yes`
   sends OWNER_REQUEST "motors_on": the robot app switches the motors on again as `on` at
   its terminal does, and the program says `the motors are on again`. Any other answer
   leaves them off (`Operator declined arm connection`).
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
4. **A task that ended gives the arm back**: OWNER_REQUEST "nobody", after a task that
   succeeded and after one the model itself ended without success (`task ended without
   success: ...`; as in the experiment's launcher the report is printed and the next task
   is asked for). The arm holds where it is, and the program says `...: the arm holds
   where it is and nobody owns it; type rest at the Thor terminal or run the console
   command rest`.
   **The model can send the arm to rest itself**: APAI's `move_to_rest` skill, the
   **rest tool**, is offered, and a task such as `rest` or `send the arm home` reaches
   it. It refuses with APAI's own words while the jaws hold something. Otherwise it
   moves the arm up to the ready pose with a planned move, folds it onto the rest pose in
   four measured steps, closes the jaws with the driver's close to the stop, and sends
   OWNER_REQUEST "motors_off". The fold ends on the stops of the second and third joint,
   inside the 0.02 rad the checkers otherwise keep free at a joint limit: the check
   ahead and the Thor's agent checker take that margin away for a setpoint that is
   inside it only at stops the rest pose lies on, and goes no further than the rest
   pose there. Both machines need the same rule. The robot app then switches the motors off, only with the
   arm measured at rest, and sends the jaws nothing more. If the robot app keeps the motors
   on, the skill reports `switching the motors off failed: ...` and the arm holds at rest
   with the motors on. After the rest tool the program says `...: the arm lies at rest
   with the motors off and nobody owns it`, and the next skill that moves asks the
   question about the motors of step 2.
5. **A task that got stuck, was blocked, ran out of steps or raised an error while the
   agent owned the arm asks for help**: OWNER_REQUEST "help_needed" with the reason, which
   starts the assistance wait on the Thor. The reasons are `stuck: ...` (APAI gave up),
   `blocked: ...`, `step budget ran out`, `agent error: ...` and `task ended without
   success: no report`. No count of failed skills ends a task: APAI ends it at
   `--max-steps`. With `task.timed_return: false` the Thor makes no return to rest when
   the wait runs out.
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
   the program needs no restart. The workstation checker is halted for the STOP, as by a
   stop of APAI's own: a grasp stopped below its hover keeps the column it approached
   through, and a next task "go to the initial pose", given within five minutes and before
   any other motion task, first lifts the gripper straight up that column and then moves
   home. The halt ends at the next task's first arm motion.
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
10. **When the program ends, it asks once whether the arm returns to rest.** The
   experiment's launcher returns the arm to rest and switches the motors off by itself
   when it ends. Here a STOP must start no motion, so the operator is asked: `End of the
   session: return the arm to rest and switch the motors off? The arm moves by itself;
   keep the return path clear. Type "path is clear" to start; anything else leaves the
   arm holding with the motors on: `. It is asked only when a task of this run owned the
   arm, the motors are on, nobody owns the arm, no takeover is open and no fault-stop is
   latched, also after Ctrl+C and after Ctrl+D at the task prompt. On `path is clear` the
   program sends OWNER_REQUEST "park", which the robot app runs as a typed `rest` (see
   [What the operator's commands do](#what-the-operators-commands-do)), and waits up to
   240 s for its RESULT. It then says `the arm lies at rest: the jaws are closed, the
   motors are off`, or `the return to rest ended <status>: <why>; the arm holds with the
   motors on: type rest at the Thor terminal or run the console command rest`. Any other
   answer gives the second half of that line. A Ctrl+C while the arm returns is a STOP:
   the arm holds with the motors on. With the motors already off the program says `the
   arm lies at rest with the motors off` and asks nothing.

The agent program sends OWNER_REQUEST and STOP; its arm adapter sends MOVE, PERMIT, GRIPPER,
CHECKER_STATE and one OWNER_REQUEST. It sends no RELEASE, SHUTDOWN, RESET or OPEN. Three of
its OWNER_REQUESTs concern the motors: "motors_off" from the arm adapter at the end of the
rest tool, "motors_on" after the operator's `yes`, and "park" after the operator's `path
is clear`. The robot app applies its own checks to each, and the agent side itself cuts no
torque.

### A move, its permits and the camera evidence

- **One MOVE per move of APAI.** The arm adapter reads the start pose from joint readings taken
  after the call began, samples the move at the Thor's setpoint period, and has the check ahead
  vet the setpoints the first permit reaches (`permits.reach_setpoints`, 25) on an obstacle
  grid at most `move.start_grid_age_s` old (with the full stack: a third of the
  experiment's `occupancy.max_age_s`). A refusal here sends nothing. The later setpoints
  are checked as the permits follow the arm, so a move that is blocked far ahead starts
  and stops before the obstacle, as APAI's own stream does.
- **A move along a joint path is one MOVE too.** APAI's skills that hand the arm a list
  of joint waypoints (`point_at`, a back-out) are sampled as APAI's driver samples them
  and sent as one MOVE; an empty path is refused with APAI's text.
- **A move whose start pose the arm has left is sampled again.** When the Thor refuses a
  MOVE because the arm is no longer where the move began (`start pose: ...`), and nothing
  of it was sent, the arm adapter reads the joints anew and samples the move from that
  reading: at most 8 samplings within 5 s, the numbers APAI's own stream has for this
  (`PREFLIGHT_MAX_CHECKS`, `PREFLIGHT_REBIND_BUDGET_S`). The run's log
  says `the move is sampled again from a new reading: ...`, and the move's line in
  `moves.jsonl` counts them as `start_retries`.
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
- **A setpoint whose permit is late waits for it in a hold.** The arm holds the setpoint
  before it, the contact watch goes on reading, and the later setpoints move back by the
  wait. The wait is at most the large delay: `executor.large_delay_s`, and with `--arm
  apai` the pause at which APAI's own stream ends a move, 0.29 s with the rig's profile
  (the contact monitor's `max_feedback_gap_s`, 0.35 s, less one round of joint reads). A
  permit that comes in time lets the move go on, and the Thor's log says `move ... waited
  N s in a hold for a permit that covers setpoint N`; one that does not ends the move
  `refused` with the permit's reason. A cancel, a STOP and a permit that quotes other
  numbers end the move at once. Before the first setpoint the wait stays
  `permits.first_wait_s`.
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
- **A camera frame is dated at its exposure, and the robot link refuses none.** While the
  robot link can date a stereo pair, the camera frame's time is its exposure on the
  workstation's clock, never later than the pair's arrival, and its evidence time is the
  exposure on the Thor's wall clock. While the newest ROBOT_STATE is older than 1 s, for
  3 s after a step of the Thor's wall clock, when an exposure lies more than 0.05 s in the
  future of the Thor's clock or after the pair's arrival, or when it does not grow, the
  camera frame is still handed to APAI, dated at its arrival, and carries no evidence
  time: `capture.time_problem` says why, and the log says `camera frames carry no
  evidence time: ...; no PERMIT and no self mask can rest on them` once and `camera
  frames carry an evidence time again` when it is over. The model and the detector go on
  seeing. The self mask refuses such a camera frame, so the obstacle grid does not take
  it and no permit can rest on it. The camera itself refuses only what the experiment's
  camera refuses: an arrival stamp that is no past time, a time that does not grow, a
  pair older than `max_frame_age_s` (3 s) once its depth is computed, and no pair within
  `frame_timeout_s` (10 s). The self mask uses the joint samples of ROBOT_STATE on both
  sides of the exposure; a camera frame without them puts nothing into the obstacle
  grid.
- **Against the experiment's capture on the Thor** the pictures travel as a 60 fps HEVC
  stream over TCP in place of single lossless shots: the pixels are not identical, the
  geometry is, and a picture arrives whole or late, never damaged.

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
`permits.renew_s` not below `permits.life_s`, with the minimal stack
`task.max_joint_speed` above `thor.agent_max_joint_speed`, or two equal stream names.
The shipped file names each upper
bound in a comment; for example a wait for a RESULT is at most 60 s and a joint speed at most
3 rad/s.

| Block | Keys |
|---|---|
| top | `state_dir` (see below), `log_level` |
| `link` | `socket_path` of the robot link service (empty: its default); `connect_timeout_s`; `state_max_age_s`, the oldest ROBOT_STATE a pose is read from; `state_renew_age_s`, the oldest ROBOT_STATE a PERMIT is sent on; `expected_profile_id`, which WELCOME must name (empty: any); `calibration_id` and `allow_unnamed_calibration` (above) |
| `thor` | Copies of three robot app settings: `first_wait_s` and `settle_timeout_s`, which WELCOME does not carry, and `agent_max_joint_speed`, which the robot app also names in WELCOME. Keep them equal to the robot app's file. The agent plans its moves at the lowest of the speed of the experiment launcher's speed profile (the minimal stack: `task.max_joint_speed`), `agent_max_joint_speed` and the limit WELCOME names; for a robot app that names none, the key stands for it |
| `move` | `start_tol_rad`: a MOVE quotes the smaller of this and half a period's travel at the Thor's agent speed limit; `fresh_wait_s`, the longest wait for joint readings taken after a call began; `start_grid_age_s` and `start_grid_wait_s`, read by the minimal stack only: the full stack takes a third of the experiment's `occupancy.max_age_s` and its `thor_motion.route_preflight_s`; `expires_in_s`, how long a MOVE stays valid on the Thor; `cancel_wait_s` |
| `permits` | `reach_setpoints` (25), `renew_s` (0.05), `life_s` (1.0) |
| `jaws` | `fresh_s`: an older jaw reading counts as unknown; `result_timeout_s` (12): how long the arm adapter waits for the final RESULT of a jaw command, and how long that GRIPPER stays valid on the Thor. Without a RESULT in that time the adapter gives up: it sends a STOP with the sender `agent`, the agent is told `no RESULT for the jaw command within 12 s; a STOP was sent`, and the rest of the task sends nothing (see [What the jaws do](#what-the-jaws-do)) |
| `task` | `timed_return` (false); `failed_steps_limit` (3), which the loader still takes and no program reads: no count of failed skills ends a task; `max_joint_speed`, the minimal stack's speed cap for a task, at most `thor.agent_max_joint_speed` (the full stack's speed is the row `thor`); `end_wait_s`, `owner_wait_s` and `grid_wait_s`, the longest waits for an owner request's RESULT, for the ROBOT_STATE that shows the agent as owner, and for a usable obstacle grid at the start; `clear_skill_stop` (true), see step 9 of "A task from its start to its end" |
| `camera` | `bridge_python`, the camera consumer's interpreter; `camera_pr` (empty: `AGENTIC_REMOTE_CAMERA_PR`); `receiver_socket`; `streams`; `calibration_dir` and `hand_eye_json`, the workstation's copies of the stereo calibration folder and the camera mounting file (the **hand-eye file**) in use on the Thor; with the full stack an empty `hand_eye_json` takes the hand-eye file of the experiment's newest accepted mounting record (see `--mounting`); `max_skew_ms` between the two eyes; `depth_backend` (`ffs` on the GPU, `opencv_cpu` without one) and `fps`, which the full stack with `--camera link` takes from the experiment's `configs/sensing_thor.yaml` instead; `allow_invented_times` (false), see `--camera` above |
| `experiment` | `stack`: `full` uses the experiment folder's behaviour hooks, its obstacle grid bridge on `cuda:0` and its cloud model; `minimal` uses none of them, builds on APAI's own profiles `arm_profile` and `llm` (both `mock` in the shipped file) and pins APAI's mock detector for every camera, so no model server is asked; `grid`: `bridge` (the shipped value) asks an obstacle grid bridge, `clear` puts the clear grid in its place, which checks nothing against obstacles and is taken only with the minimal stack and against a simulated driver (see [Run demo 1 by hand with the mock agent](#run-demo-1-by-hand-with-the-mock-agent)); `codex_bin`, the model client's program for the full stack, which the agent and its detector both start (empty: the one the experiment's own file names, inside the experiment folder, which is the Thor's and does not run on the workstation: see [The model client on the workstation](#4-the-model-client-on-the-workstation)), refused with exit code 2 when it names no program that can be run; `timing` (true) and `model_trace` (true), the experiment's two recordings of a run, read by the full stack only (see [The options of the experiment's launcher](#the-options-of-the-experiments-launcher)) |
| `rig` | Optional, and in neither shipped file. With the full stack the arm profile is the one the experiment's launcher loads (home pose, park pose, the arm mask's margin, the motion bounds); a `rig` block puts a value of its own in the place of the experiment's: `home_q`, `park_q`, `body_mask_dilate_px`, `motion_bounds`. `home_q`, `park_q` and `motion_bounds.joint_center_rad` must have one value per joint of the arm; the program compares them with each other when it loads the file and, with those of the arm profile, with the joint count in WELCOME after it attached (exit code 1). The full stack does not start without motion bounds. `motion_bounds` takes `joint_center_rad`, `max_joint_deviation_rad`, `tcp_center_m` and `max_tcp_distance_m`, all four and no other key; the two `max_` values must be above 0 |

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
code 2. It sets no APAI variable: `APAI_REQUIRE_CUDA`, APAI's strict CUDA mode, stays as the
operator's shell has it, as in the experiment's launcher. With `APAI_REQUIRE_CUDA=1` APAI's
grasp code fails, and the agent program says so in a `WARNING` line at its start: unset
it. The script sets `PYTHONDONTWRITEBYTECODE=1`, so that nothing is written into APAI's
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
`camera_receiver/run.sh` with `configs/camera_receiver_rebot.yaml`. The receiver dials
the sender, so `--host` is the Thor's address: the file itself names `127.0.0.1`, and a
receiver started without `--host` dials the workstation and finds no sender. It needs
`AGENTIC_REMOTE_CAMERA_PR` and exits with code 2 and one line without it. A first argument
that does not begin with `-` is another configuration file; the other arguments go to the
receiver.

`run_viewer.sh` sets the environment and then runs the experiment's `thor_viewer.py` on
the agent program's run folders; see [The experiment's viewer](#the-experiments-viewer).
It needs `AGENTIC_REMOTE_AGENT_PYTHON`, `AGENTIC_REMOTE_EXPERIMENT_DIR` and
`AGENTIC_REMOTE_APAI_SRC`, and exits with code 2 and one line without one of them. With
`AGENTIC_REMOTE_CAMERA_PR` set as well, and without `--no-live`, it runs the viewer through
the example's launcher `viewer_live`, which adds the camera's live picture to the page
and tells the robot app how old the picture on that page is: the page is then the live
view (see [The live view](#the-live-view)).

`run_workstation.sh` starts the workstation's programs of demo 1 on the rig together, in
one tmux window. **tmux** is a program that splits one terminal into **panes**, each a
terminal of its own. The window has five panes, all visible together, and each has its
name on its upper border:

```text
bash run_workstation.sh [--env FILE] [--thor ADDRESS] [--session NAME]
    [--socket-wait SECONDS] [--print]
```

| Pane | Its command |
|---|---|
| `console` | `./run_console.sh` |
| `agent` | `./run_agent.sh --config configs/agent_rebot.yaml --operate --interactive --record-debug --depth-backend ffs --motion-approval task --batch-milestones` |
| `service` | `python -m isaaccapture_examples.agentic_remote.link_service --config configs/link_service_rebot.yaml`, under the Python of `AGENTIC_REMOTE_AGENT_PYTHON` |
| `camera` | `./run_camera_receiver.sh --host ADDRESS` |
| `viewer` | `./run_viewer.sh` |

- **Which pane is which.** `console` is the operator console, where `stop`, `rest` and
  the other commands are typed. `agent` is the agent program, where tasks and their
  `yes` are typed. `service` is the robot link service, `camera` the camera receiver
  and `viewer` the experiment's viewer. The script arranges the five as tiles in rows
  of two: `console` and `agent` in the first row, `service` and `camera` in the second,
  `viewer` alone in the third. It leaves the pane `console` selected.
- **The pane `viewer` is the live view.** The experiment's viewer is the one web page of
  the workstation. With its live picture showing in a visible browser tab, that page is
  the live view, which a person's takeover needs on the rig
  (`human.require_live_view: true` in the rig's robot app file; see
  [The live view](#the-live-view)). The script says so in its last line: `live view:
  the viewer's page with its live picture showing, in a visible browser tab; a person's
  takeover needs it`.
- **The service and the viewer start at once.** The viewer attaches to the service by
  itself, also when the service starts later or is started again. The camera receiver
  starts at once when
  `--thor ADDRESS` gives the Thor's address; without it the command is typed into its
  pane with `THOR_ADDRESS` in the address's place and waits there.
- **The console starts once the service listens.** The console attaches once to the
  service's local socket and ends when nothing listens there, so the script sends its
  command only when the service takes a connection on that socket. It waits for that
  at most 30 s; `--socket-wait SECONDS` names another wait, in whole seconds. When the
  wait runs out, the command is typed into its pane and waits for Enter, and the
  script says `the robot link service took no connection on <socket> within 30 s: look
  at the pane service`: press Enter in the pane `console` once the service runs.
- **The agent's command is typed and not entered.** It stands in the pane `agent` and
  waits: click that pane and press Enter once the console shows the robot connection
  up. It carries the arguments the experiment's own launcher is started with on the rig
  (see [The options of the experiment's launcher](#the-options-of-the-experiments-launcher)),
  `--record-debug` among them, so the pane `viewer` lists the run. That pane prints what
  a browser needs to open the page (see
  [The experiment's viewer](#the-experiments-viewer)).
- **A program that ended can be started again in its pane.** Each pane is a shell into
  which the script typed the command. When the program ends, the pane is back at the
  shell's prompt with its last lines still there: click the pane, press the Up key and
  Enter. The console needs that after every restart of the service: it ends by itself
  when the service is gone. The viewer does not: it attaches again by itself.
- **The mouse belongs to tmux.** The script switches tmux's mouse support on for its
  session. A click selects a pane, and what is typed goes to the selected pane. The
  wheel scrolls back through a pane's lines; `q` leaves that scrolling. A drag along a
  border changes the panes' sizes.
- **To copy text out of a pane**, there are two ways:
  1. Drag over the text with the left button. tmux marks it inside that pane only and,
     when the button is let go, keeps it in tmux's own buffer. `Ctrl+B`, then `]`,
     pastes it into the selected pane. Whether it also reaches the desktop's clipboard
     depends on the terminal program.
  2. For the desktop's clipboard, hold Shift while dragging: the terminal program then
     selects by itself, as it does without tmux, and its own copy key works
     (`Ctrl+Shift+C` in GNOME Terminal). The terminal program knows nothing of panes,
     so a selection of several lines runs across the panes beside it. Make the pane
     fill the window first with `Ctrl+B`, then `z`; the same keys bring the other panes
     back.

  These are tmux's own defaults for a session with the mouse on. A tmux configuration
  file of the user can change them, and some terminal programs take another key than
  Shift.
- **To leave**, there are two ways:
  1. `Ctrl+B`, then `d`, detaches: the window goes and every program runs on. `tmux
     attach -t agentic_remote` shows it again.
  2. To end the programs, end each in its pane first: Ctrl+D at the agent's `task> `,
     `quit` at the console, Ctrl+C in the panes `camera` and `viewer`, and last Ctrl+C
     in the pane `service`. Then `tmux kill-session -t agentic_remote` closes
     the window. None of this stops or moves the arm: the robot app on the Thor holds
     on its own. Do not close the window while the agent owns the arm: end the task
     first (Ctrl+C in the pane `agent` is a STOP).
- **A second start is refused** while a session of that name exists: `a tmux session
  'agentic_remote' exists: tmux attach -t agentic_remote, or end it first`.
- **`--env FILE`** names a shell file of `KEY=value` lines that sets the six variables
  below. They may be set in the environment instead, and `--env` left out. **`--session
  NAME`** names the tmux session (`agentic_remote`). **`--socket-wait SECONDS`** is the
  wait for the service's socket, above.
  **`--print`** prints the five commands and starts nothing.
- **The six variables**: `AGENTIC_REMOTE_AGENT_PYTHON` (the Python of the workstation
  environment; the script also gives it to the console as
  `AGENTIC_REMOTE_CONSOLE_PYTHON`),
  `AGENTIC_REMOTE_ENV_DIR`, `AGENTIC_REMOTE_ISAAC_CAPTURE_PY` (the folder of the
  Isaac Capture Python package built for that Python), `AGENTIC_REMOTE_APAI_SRC`,
  `AGENTIC_REMOTE_EXPERIMENT_DIR` and `AGENTIC_REMOTE_CAMERA_PR`. Each pane's command
  carries them itself, so a tmux server that runs already changes nothing.
- **One more variable is optional**: `AGENTIC_REMOTE_VIEWER_HOST` names the address the
  experiment's viewer listens on, for example the workstation's address on the internal
  network, or `0.0.0.0` for every interface. With it the command of the pane `viewer`
  is `./run_viewer.sh --host <that address> --no-auth`. Anyone who can reach that
  address can then see the camera pictures, the maps and the timings of the runs,
  without a token. Without it the viewer listens on `127.0.0.1` and asks for its token.
  The script's line `viewer: ...` names the page's address in both cases. A
  value that is not an IPv4 address or a host name of letters, digits, dots and hyphens
  is refused with exit code 2 and one line.
- **Exit code 2 and nothing started**: an unknown argument, a variable that is not set,
  a Python that cannot be run, a folder that is none, no `tmux` on `PATH`, or a tmux
  session of that name that exists already.
- **The script does not show the window itself**: it prints `attach with: tmux attach -t
  agentic_remote`, and that command shows it. Started with `bash`, the script runs with
  and without its file's execute bit; its panes start the other scripts as
  `./run_agent.sh` and the like, which need theirs.
- **No test starts tmux.** What a test runs is the script's syntax check, `--print`, and
  its refusals of an unknown argument, of a variable that is not set and of a Python
  that cannot be run. The keys above are tmux's and were not tried from a test.

### `state_dir` and the run folders

`state_dir` (shipped: `../.state`, which is `examples/agentic_remote/.state`) holds what the
agent program writes. The program puts a `.gitignore` with `*` into it.

- `state_dir/apai/` holds APAI's stores (beliefs, experience, grasp memory and the like). A
  store whose APAI environment variable names a place stays there. A variable that is set
  and empty counts as not set: the store goes into `state_dir/apai/`.
- `state_dir/runs/<UTC time>-<process id>/` is the run folder, unless `--run-dir` names
  another. It holds `agent-settings.json` (the resolved settings), `arm_profile.yaml` and, for
  a stereo camera, `camera_profile.yaml` (the profiles APAI was given),
  `resolved-config.yaml` (APAI's whole configuration of the run, under the name the
  experiment's launcher gives it; written at the start, and
  [the experiment's viewer](#the-experiments-viewer) lists a run only with it),
  `task-reports.jsonl`
  (one line per task: APAI's task report with every skill's result), APAI's own trace files,
  and `experiment-files.json` (the SHA-256 of every file of the experiment folder the run
  imported) when `AGENTIC_REMOTE_EXPERIMENT_DIR` is set. When `resolved-config.yaml` cannot
  be written the run goes on and the program's log says so in a warning. It also holds
  the seven files of
  [What a run records and measures](#what-a-run-records-and-measures): `moves.jsonl`,
  `move-paths.jsonl` (the planned joint path of every MOVE), `jaw-commands.jsonl`,
  `robot-state.jsonl`, `camera-deliveries.jsonl`, `link-latency.jsonl` and
  `latency-summary.txt`.
- `state_dir/viewer.token` is the access token of the experiment's viewer, when
  `run_viewer.sh` was started beside the shipped `state_dir`.

### What a run records and measures

The motors and the camera are on the Thor, and the agent program is on the workstation.
What the program can measure of that split it measures in **every run**: with the full
and the minimal stack, with or without `--record-debug`. Seven files go into the run
folder: six of measurements, and `move-paths.jsonl` with the joint path the agent planned
for each move. The program names them in a line that begins `measurements:` at its start,
and prints the last one, the latency summary, when it ends.

**Nothing of it runs on a thread that drives the arm.** The link's receive thread, the
skill thread (the thread that sends the PERMITs and a skill's STOP) and the camera's thread
only put what they have into a queue of 4096 places. One thread, `agent-measures`, takes
the queue, counts and writes. What comes while the queue is full is dropped and counted
(`dropped`, and the summary's last line). A file that cannot be written is named once in
a warning of the program's log and left out; the run goes on. The measuring ends when the
task loop has ended and the agent has given the arm back, just before the program closes
its local socket; it waits at most 3 s for the thread there. The summary is printed
behind the close.

Three clocks are in the files. **The workstation's monotonic clock** has every name that
ends in `_mono_s` or `_after_s`, every wait, `frame_t` and `sent_to_result_s`; `utc` is
that clock as a date. **The Thor's monotonic clock** has `clock_mono_s`, `target_t`, the
jaws' `t`, every single joint read and everything under `thor` in a line. **The Thor's
wall clock** has `clock_wall_s`. A number that is taken between two machines says so
below. All durations are in seconds in the six `.jsonl` files and in milliseconds in the
summary.

| File | One line per | What it holds |
|---|---|---|
| `robot-state.jsonl` | ROBOT_STATE while the arm moves or one of the program's jaw commands runs; else one a second | What the robot app reports of the motors: the measured joints `q` (the newest reading of each joint) and `sample_span_s`, how far apart those readings ended; `reading_age_s`; `target_q` and `target_t`, the last joint target the Thor sent and when; the jaws (`opening` from 0 closed to 1 open, its time `t`, `valid`, `last_target`, `load`); the move (`move_id`, `move_index`, the count of setpoints sent, `move_total`, `move_stage`); `missed_deadlines`, the Thor's running count of setpoints it sent late; `permit`, the permit the Thor holds; `reads`, the single joint reads since the line before (count, total and longest read time); `arrived_mono_s`, `seq`, the Thor's clock pair, `transit_above_shortest_s`, the control owner and the stop number |
| `moves.jsonl` | call of the arm's move, also one the checker refused before a MOVE was sent | The stretches of a move, see below |
| `move-paths.jsonl` | MOVE the agent sent, or could not send | The joint path the agent planned for it: every setpoint, in order, and how the move ended, see below |
| `jaw-commands.jsonl` | jaw command | What a re-run needs of it: `mode`, the `opening` asked for, `effort`, `scout_opening` (the scout stage of a two-stage close, only then), `timeout_s` (how long the command was good for), `sent_mono_s` and `sent_utc` (when it left), and `skill` and `trace_step` as in `move-paths.jsonl`. Then what came of it: `sent_to_result_s`, from the GRIPPER link message to its final RESULT; `status`, `outcome`, `result_opening`; `waited_s`, with the wait for two later ROBOT_STATEs; under `thor` the ROBOT_STATEs that came meanwhile: `trace`, per ROBOT_STATE the Thor's clock, the measured opening and the last target sent (600 at most), and `target_gap_max`, the largest distance between the two |
| `camera-deliveries.jsonl` | camera frame the agent's camera returned to APAI | `frame_t`, the camera frame's time; `returned_mono_s`, when the camera handed it to APAI; `age_at_return_s`, the second less the first; `capture`, the camera frame's capture facts as the camera wrote them |
| `link-latency.jsonl` | second in which a ROBOT_STATE arrived, and a last line at the end | The robot connection: `states` and `seq_missing` (ROBOT_STATEs that arrived, and those that did not, by their numbers), `arrival_gap_max_s`, `transit_above_shortest_s` (median and longest), `clock_offset_s`, `clock_spread_s`, `clock_samples`, `clock_steps`; `joint_reads` and `read_gaps`; `reading_age_s`, `missed_deadlines`, `view_age_s` of the newest ROBOT_STATE; `permit_frame_age_max_s`; and what ended in that second: `first_result_after_s` of a move, `jaw_result_s` of a jaw command, `stop_shown_s` of a STOP |
| `latency-summary.txt` | kind of duration | count, median, 95th percentile and longest of each kind over the run, and four lines on the clocks, the setpoint pace and the counts |

**A move, in `moves.jsonl`.** A move is one call of the arm by a skill: one MOVE link
message and the PERMITs behind it. Its line holds, in the order they happen:

- `fresh_wait_s` and `fresh_reads`: the wait for joint readings the Thor took after the
  call began, and how often APAI asked for them.
- `grid_wait_s`: the wait for an obstacle grid from a camera frame at most
  `move.start_grid_age_s` old.
- `whole_check_s`: the check, on that grid, of the setpoints the first permit reaches,
  before anything is sent. `start_retries`, only when the arm had left the start pose:
  how often the move was sampled again from a new reading.
- From just before the MOVE is handed to the link, each on the workstation's clock:
  `first_result_after_s` (the robot app's first RESULT, its admission), `armed_after_s`
  (a ROBOT_STATE shows the move as armed), `first_permit_after_s` (the first PERMIT has
  left), `first_sent_after_s` (a ROBOT_STATE reports a setpoint as sent),
  `last_progress_after_s` (the count of setpoints reported as sent rose for the last
  time), `all_sent_after_s` (a ROBOT_STATE reports every setpoint as sent) and
  `result_after_s` (the final RESULT).
- From these: `sending_s`, from the first setpoint reported to the last rise of the
  count; `after_last_progress_s`, from there to the final RESULT; `settle_s`, from every
  setpoint reported to the final RESULT.
- `passes`, `pass_total_s` and `max_pass_s`: the passes of the check ahead (one pass
  clears the next setpoints for a PERMIT), counted in bulk; `permits`, `through`,
  `approved`, `sent_seen`.
- How it ended: `ended` (`completed`, `not_settled`, or the name of the error APAI got),
  and the robot app's `status`, `outcome` and `sent`.
- `path_file`, in a line with a `move_id`: `move-paths.jsonl`, where the line with the same
  `move_id` holds the move's setpoints.
- Under `thor`, from the ROBOT_STATEs that named the move, on the Thor's clock: `states`,
  how many; `setpoints_per_s`, the pace of the setpoint stream, which is the rise of
  `move_index` over the Thor's time between the first ROBOT_STATE with a setpoint sent and
  the one in which the count rose last; `missed_deadlines`, the rise of the Thor's count
  during the move; `joint_reads`, the single joint reads of that time; `permit_lead_min`,
  the fewest setpoints the held permit still covered beyond those sent; `frame_age_first_s`
  and `frame_age_max_s`, the age of the camera frame behind the held permit as the Thor
  counts it.

**The planned path of a move, in `move-paths.jsonl`.** The agent plans the joint setpoints
of a move on the workstation and sends them to the Thor in one MOVE; the robot app then
sends them to the motors one by one, `period_s` apart, as far as the permits reach. The
file holds one line per MOVE the agent sent or could not send, written when the move has
ended; a call of the arm that was sampled again from a new reading (`start_retries`)
leaves a line for each MOVE. A line holds:

- `move_id`, the id in `moves.jsonl` and in ROBOT_STATE; `utc` and `sent_mono_s`, when
  the MOVE left, on the workstation's clocks; `sent`, and `not_sent` with the reason when
  the MOVE could not be handed to the link (the line then has no `end`).
- What the MOVE was sent under and carried: `session_id`, `owner_number`, `stop_number`;
  `period_s`; `start_q`, the pose it was planned from, and `start_tol_rad`, how far from
  it the robot app may find the arm; `grasp_region` (`center_xy`, `radius_m`, `z_min`,
  `z_max`) or `null`; `evidence_wall_s`, the time of the camera frame behind the check
  before it was sent; `expires_in_s`.
- `skill` and `trace_step`: the tool that asked for the move, and the `step` of that
  tool's row in APAI's `trace.jsonl`. The step is the one the trace had ready for its
  next row when the tool was called. Both are `null` for a move no tool asked for.
- `end`, how it ended: the robot app's `status` (`completed`, `refused`, `cancelled`,
  `failed`), `sent`, **how many of the setpoints it sent to the motors**, `reason`, its
  text, `outcome` and `end_q`, the pose it measured at the end; `sent_seen`, the count
  the last ROBOT_STATE before the end showed; `error`, the error APAI got for the move on
  the workstation, `null` for a move that returned; and `utc` and `mono_s` of the end.
  `end` is `null` when the program ended before the move did.
- `count`, and last in the line `setpoints`: every joint setpoint in order, each value
  as it was sent, so the text reads back as the same number.

The setpoints `0` to `sent - 1` are what the robot app commanded; the rest of a path that
was stopped on its way was planned and never sent to the motors. **Go by `end.sent`, not
by `end.status`**, for what reached the motors. A move that was stopped on its way, by a
refusal of the check ahead or by Ctrl+C, ends `refused` with `sent` above 0: its first
`sent` setpoints were commanded. `sent` is `null` when the RESULT carried no count: for a
MOVE the robot app refused at its admission, none was sent; for a move without a RESULT
(`status` is `null` too: the robot connection was lost), at least `sent_seen` were. A
line takes about 130
bytes a setpoint: 17 kB for the 125 setpoints of a usual move of the session of
2026-11-17 and 125 kB for its longest, of 958; the 14 moves of that session would have
taken 0.5 MB. **The move is not delayed by it**: the skill thread hands the MOVE's own
arrays to the queue when the MOVE has left and how it ended when it has ended. That costs
some tens of millionths of a second per move: for all the calls a move adds, a median of
24 and at most 49 were measured in a test with paths of 3000 setpoints. The thread
`agent-measures` makes the text
from them, 16 setpoints at a time with a pause between, so that it never keeps the
interpreter for more than about half a thousandth of a second from the thread that sends
the PERMITs. A full queue or a file that cannot be written costs the line, not the move.
After one failed write `move-paths.jsonl` is shut for the rest of the run, as the other
measurement files are: the moves that follow leave no line, a line half written may be
its last, and the summary names the file behind `not written:`.

**The single joint reads** are the Thor's own: ROBOT_STATE repeats every read of its last
`state.recent_window_s` (0.5 s in the robot app's files) with the start and the end of the
read, and comes every `state.publish_period_s` (0.05 s), so each read arrives about ten
times. A read is counted once, by its joint and its end, and its read time is the end less
the start, both of the Thor's clock; no clock of the workstation is in it.

**Between the two machines.** A **clock sample** is a ROBOT_STATE's arrival on the
workstation's clock less the Thor's clock inside it: the offset between the two clocks
plus that ROBOT_STATE's transit over the robot connection. `clock_offset_s` is the
smallest of the newest 64 samples, so the shortest transit is in it;
`clock_spread_s` is the largest less the smallest; `transit_above_shortest_s` is a
ROBOT_STATE's own sample less the smallest. `stop_shown_s` is the time from a STOP this
program sent to the first ROBOT_STATE with a higher stop number.

**With `--record-debug`** the same thread gives the experiment's debug recorder three kinds
of event. `robot.feedback` is the measured joints of a ROBOT_STATE, at most 10 a second of
the Thor's clock, with `seq`, `sample_span_s`, the source `ROBOT_STATE over the robot
connection` and its `clock` text; the time of the pose is the ROBOT_STATE's arrival less
`reading_age_s`. So [the experiment's viewer](#the-experiments-viewer) draws the arm from
the ROBOT_STATEs themselves, also while APAI does not read the arm, for as long as the
agent program is attached; no pose is recorded when APAI reads the arm. `robot.state` is
one event per ROBOT_STATE with each single joint read once (`reads`: joint, value, start,
end), `target_q`, `target_t`, the jaws and the move. The viewer counts `robot.state` and
draws nothing from it. `robot.move_path` is one event per MOVE, recorded when the MOVE
has left: the fields of its line in `move-paths.jsonl` without the `end`, and
`setpoints` as one array of float64 with a row per setpoint. Its field `what` says what
it is: the **planned** path the agent sent in one MOVE, not single targets timed at the
motors. The experiment's recorder sits beside the motor driver and records each target as
a `robot.call` of `send_joint_target` with its own time; here that driver runs on the
Thor, and no such event is made up. How far a path was sent shows in `move_index` of the
`robot.state` events and in the `end` of its line in `move-paths.jsonl`. The rate never
leaves a `robot.move_path` out and the size limit never drops one. The experiment's
viewer and replay tools read none of it: `thor_replay_mask_margin.py` checks the
`send_joint_target` targets of `robot.call` events, which such a recording does not hold.

**With the experiment's timing** a move's span `motor.stream_to` (a span is one timed
stretch in `timing.jsonl`) holds `motor.fresh_joint_readings`, `safety.wait_fresh_grid`,
`safety.check_whole_move`, `motor.move_to_armed` with `motor.move_to_first_result` inside
it, and `motor.permits`, which counts the passes of the check ahead in bulk as
`safety.check_ahead_pass` (count, total, longest). Three spans without length mark the
points of the permit loop: `motor.mark:first_permit`, `motor.mark:first_setpoint` and
`motor.mark:all_sent`. `task.take_control` is the taking of the arm before a task's first
skill that moves, with the typed `yes` inside it, and `task.give_back_control` the giving
back, or the help request, after the task.

**What a run's record is good for, and what it is not.** Four things can be taken from a
run folder afterwards. The model's decisions can be replayed from the model transcript
(`model-trace/`, with the full stack): every request, the pictures in it and every
answer. Perception and the checks can be run again on the stored pictures of the debug
recording (`debug/`, with `--record-debug`): the camera frames, the depth, the masks and
the obstacle map's answers, for the map pictures that were stored. What the agent planned
and how far it was sent is in `move-paths.jsonl` and `jaw-commands.jsonl`: every joint
setpoint of every move in order, the pose it started from, how many of them the robot app
sent, and every jaw command with its mode, opening and effort. What the arm did is in
`robot-state.jsonl`: the measured joints and the last target sent, 20 times a second
while the arm moves. So a run can be played back, and a move can be sent again from the
same start pose; a policy can be trained on what was planned against what was measured.
**It is not a closed-loop re-run.** The record holds one history: what the camera saw
and what the arm did after the commands that were sent. A different command, or the
same command on a scene that has changed, has no answer in it; that needs a simulator
or the rig. Three more limits. The setpoints are the planned ones: when each reached
the motors is known only to 0.05 s, from the count in ROBOT_STATE. ROBOT_STATE carries
the joints' positions only: no joint speed and no torque is recorded. And the debug
recording holds 2 map pictures a second,
not every camera frame, unless the run was started with `--debug-map-hz 0`.

**What the workstation cannot see.**

- How long the Thor takes to send one setpoint to the motors and to check it: the robot
  app does not report it. The pace comes from the counts in ROBOT_STATE, 20 a second, so
  it is good to about one setpoint over the move; a move too short for two ROBOT_STATEs
  with progress has none. A late setpoint shows only in `missed_deadlines`.
- The camera's capture and encoding apart from the network. `frame_t` of a camera frame
  of the camera connection is the start of its exposure, moved to the workstation's clock
  by the link's estimate of the Thor's clock; `capture.receipt_monotonic_s` less `frame_t`
  is capture, encoding and the camera connection together, and it reads short by the
  shortest transit of a ROBOT_STATE and a margin of 0.02 s. The receiver's and the
  bridge's steps on the workstation are in `capture.check`. The camera frames of APAI's
  mock camera carry invented times.
- The offset between the two clocks apart from the shortest transit, and so the transit
  of one ROBOT_STATE in full.
- Reads and jaw readings between two ROBOT_STATEs when ROBOT_STATEs were lost for
  longer than the Thor's window of recent reads: `read_gaps` counts those places.
- The settling when the robot app ended the move before a ROBOT_STATE showed every
  setpoint as sent: `settle_s` is then `null`, and `after_last_progress_s` holds the last
  setpoints and the settling together. The points taken from ROBOT_STATE are late by up to
  one ROBOT_STATE, 0.05 s.
- Everything after the agent program has ended, and a STOP another program sent.

**What was run.** By tests (`test_agent_measures.py`, `test_viewer.py`,
`test_agent_options.py`), on 2026-11-16, on one machine: a scripted stream of ROBOT_STATEs
with known reads, clocks and progress, in which every number of the files is checked; the
real robot app on the simulated arm, for a move, a jaw command, a full queue and files
that cannot be written; the whole program on APAI's mock scene, once with the experiment's
own timing for the spans of a pick and place; and the experiment's own recorder and viewer
on a scripted stream. And on the rig: since its sessions of 2026-11-17 the six
measurement files exist for a real run. The run folder `20261118T022210Z-3147525` of the
second session, made with the rig's agent file, the full stack, the camera connection and
the robot connection between the two machines, holds them for its 14 moves and 4 jaw
commands, with the latency summary as the program printed it. The numbers of that run
are in its folder; this README quotes one only where it names the run.

**The planned paths and the rule for the wait: what was run.** Tests only, on 2026-11-17,
on one machine (`test_agent_measures.py`, `test_arm_adapter.py`,
`test_agent_end_to_end.py`, `test_debug_recording.py`, `test_agent_options.py`). The real
robot app on the simulated arm and on APAI's mock arm, once behind the real robot link
service on 127.0.0.1: the setpoints of each line were, value for value and in order, the
joint targets the arm's driver took, for moves that completed and for one stopped on its
way; a scripted robot side for a move that was refused, cancelled, not settled, sampled
again and not sent; a folder that could not be written. The rule for the wait: APAI's own
watcher and obstacle map on a stand-in camera and a clock the test moves, and the agent
program on APAI's mock scene with the experiment's own recorder, a recording limit of
megabytes and a stand-in prompt that waited 1.5 s. Who owns the arm and whether it moves
came from scripted ROBOT_STATEs with joint readings that spread as far as those of the
rig's standing arm: a person who owned the arm for 30 s of a wait, a standing arm at
rest and in a hold, a return to rest by recovery, a hold that settled, jaws that opened,
an arm pushed out of a hold, and 10 s without a ROBOT_STATE; once through the link
and the measuring thread. In three runs of the program, without the paths and the rule,
with the rule thinning every wait and with the rule as it is, the arm's driver took the
same targets and jaw commands. **Not run on the rig**: no session has written
`move-paths.jsonl` or recorded with `--debug-idle-map-s` yet, and no takeover was
recorded with it; the sizes, the 2.0 GB and the 3.3 GB above are computed from the
session of 2026-11-17.

## The operator's programs

The operator has three programs on the workstation: the operator console, the console
command and the experiment's viewer, whose page is the live view. Each is a local client
of the robot link service. At the robot
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
| `open` | Thor terminal, console, console command | `yes` |
| `close` | Thor terminal, console, console command | `yes`, asked only while the jaws are not proven empty; with jaws proven empty nothing is asked |
| `release` | console, console command | `yes`. At the Thor terminal `release` asks nothing |
| `on` | Thor terminal, console, console command | `yes`. At the Thor terminal it is asked only while the motors are off at rest |
| A `rest`, `release`, `quit`, `shutdown` or `handback` that the robot app refused only because the jaws are not proven empty | Thor terminal (`rest`, `release`, `quit`), console, console command | `jaws are empty`, typed as the next line; the command is then asked again with its own question |
| `takeover`, `handback` | console, console command | `yes`. The robot link service forwards them only with `operator.typed_takeover: true` in its file |
| `reference` | Thor terminal only | `jaws are empty` |
| A task's first skill that moves | agent program | `yes` |
| The motors on again, before a skill that moves after a rest | agent program | `yes` |
| The return to rest when the agent program ends | agent program | `path is clear` |
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

At the Thor terminal's `rest`, `quit`, `on`, `open`, `close` and `reference`:

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
  jaws, the motors' state (below; `torque unknown` while the motion is
  `release_failed`, after a torque release that failed), the assistance wait, the held
  permit, the missed deadlines (below), the view age, the parts about a
  person at the controller (below) and the age of the ROBOT_STATE it shows. An owner that
  lasts less than a second can be missing from the status lines; the EVENT lines name every
  change.
- **The view age** in the status line is `view N s`, or `view none` without one. It comes
  from the experiment's viewer: how old the live picture on its page is (see
  [The live view](#the-live-view)). `view none` on the rig means that no takeover is
  possible: the viewer's page does not show its live picture in a visible browser tab,
  or the viewer does not run. Where
  the robot app's file asks for no live view (`human.require_live_view: false`, as in
  the one-machine recipes) and none reports, the line says `live view not required` in
  its place. A missing live view is then no reason against a takeover.
- **The motors' state** in the status line is one of six texts. `torque on` in
  ROBOT_STATE says only that the robot app switched the motors on and has not released
  them: the reBot motor bus has no read of a motor's enable state. What the joint
  readings say of the hold is added where the arm port can say it (`health.motors`):

  | Text | Meaning |
  |---|---|
  | `torque on (holding)` | Every joint reads at the target it was told to hold |
  | `torque NOT HOLDING` | A joint has left the target it was told to hold, or a motor reports a fault: it is pushed, blocked, or its motor is off. A fault-stop says which joint. Keep a hand on the arm |
  | `torque on, hold unknown` | No joint target was sent since the connect, a joint was not read at the target in force yet, or a joint was not read for more than 1 s |
  | `torque on` | The robot app reports nothing about the hold, as on the simulated arm |
  | `torque off` | The robot app released the motors at measured rest; `on` switches them on again |
  | `torque unknown` | A torque release failed: the motors may be on or off |

  The viewer's page shows the same words in the line of readings under its 3D view, and
  the robot app's line after the real driver's connect names the state.
- **`missed deadlines N`** in the status line, behind the permit, only while N is above
  zero. A **missed deadline** is a setpoint of a move that the robot app reaches, or has
  sent, more than `executor.small_delay_s` (0.005 s) late; a move that ends `late` counts
  once more. The robot app counts them from its start (`health.missed_deadlines` in
  ROBOT_STATE); the setpoints of a human drive are not counted. The first one makes a new
  status line due. A count that grows does not: the next status line that is printed
  carries the count of then, and `state` prints it at any time.
- Every EVENT of the robot app as `[kind] text`, with the terminal bell: once for a bell of
  the assistance wait and for the end of a human drive, twice for a takeover that began or
  ended, three times for a refused takeover or hand-back. The wearer of a headset cannot
  read the console, so the count tells the three apart. The bells of one line are written
  together in front of it; whether a terminal rings them apart was not tried.
- A **jaws line** after each status line while ROBOT_STATE carries `health.human.jaws`
  and the robot connection is up: `jaws: follow the trigger`, with `; <reason>` behind
  it while the jaws are not where the trigger asks; `jaws: trigger asks X, grip
  level Y: bring the trigger to the grip level`, X and Y as shares of the jaw travel;
  or `jaws: the trigger moves nothing: <reason>`. The rule behind it is under
  [A takeover, step by step](#a-takeover-step-by-step).
- A **joints line**, only for a typed `state`, below the ROBOT_STATE:
  `joints (rad): 0.012 -0.050 -0.050 1.000 -1.571 0.250; last target: 0.010 -0.050
  -0.050 1.000 -1.571 0.250; reading age 0.02 s; missed deadlines 0`. The first list
  holds each joint's newest valid reading in rad, in the order of the joints, with three
  decimals; `invalid` stands for a joint that has no valid reading. The second list is
  the last joint target the robot app sent to the motors, `none` before the first one.
  The reading age is that of the oldest of these readings when the robot app built the
  ROBOT_STATE (`health.reading_age_s`), `unknown` while a joint has no valid reading.
  The count of missed deadlines is `not reported` for a robot app that sends none.
- A help request as a block: the reason, the jaws, the time left, what can be typed and
  the takeover line (below).
- A refusal another client got, marked `(another client)`; notices of the robot link
  service; and `ROBOT CONNECTION LOST: ...` with the bell.

| Typed | What it does |
|---|---|
| `stop` | STOP: the arm holds where it is, motors on; it ends a running task |
| `reset` | RESET: clears a fault-stop; starts nothing |
| `rest` | OWNER_REQUEST "rest": the robot app returns the arm to rest, measures rest, closes the jaws and switches the motors off there. While a takeover is open the motors stay on at rest, for the hand-back |
| `open` | OPEN: the driver's own opening, in one go; what the jaws hold is let go. Nobody may own the arm: type `stop` first |
| `close` | OPEN with the action `close`: the close of APAI's `close_gripper` skill, in two stages. It asks for `yes` only while ROBOT_STATE does not show the jaws as `empty`. Nobody may own the arm |
| `release` | RELEASE: at measured rest the jaws close and the motors go off; away from rest nothing happens |
| `on` | OWNER_REQUEST "motors_on": the motors on again after a `rest` or a `release`, without a new start of the robot app. The robot app connects the arm as at its start, and only an arm that lies at rest |
| `shutdown` | SHUTDOWN: return to rest if the arm is away from it, motors off there, the robot app ends |
| `takeover` | OWNER_REQUEST "human": a person becomes the control owner, as with right B held 1 s. The typed fallback for the button; the robot app takes it only while nobody owns the arm, so type `stop` first |
| `handback` | OWNER_REQUEST "handback": ends a takeover with the arm at rest, as with right A held 1 s. The typed fallback for the button |
| `state` | Prints the service's status, the newest ROBOT_STATE, the joints line, a status line and the takeover line; sends nothing |
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
  `open`, `close`, `release`, `on` and `shutdown` answer at once `no ROBOT_STATE has
  arrived` and send nothing; type the command again. The attempts to attach, the first
  one and the one a typed
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
- **Rest, release and shutdown are checked on the Thor, not here.** The console only sends
  the link message. The robot app releases torque only with the arm measured at rest and
  the jaws proven empty (see [the torque rule](#the-rule-torque-is-never-cut-away-from-rest)); a
  `release` with the arm away from rest is `refused` and the motors stay on.
- **The question about the jaws.** When the robot app refused a `rest`, a `release`, a
  `shutdown` or a `handback` only because the jaws are not proven empty, the console
  asks: `rest was refused because the jaws are not proven empty. Look at the jaws. If
  they hold something: stop, open, yes. If they are empty: type "jaws are empty" and
  rest is asked again on your word: `. The question stands for the next typed line only.
  On `jaws are empty` the command is asked again with its own question and sent with
  your word, which the robot app takes in place of its measurement until the jaws are
  next sent a command or the arm gets a new control owner. The robot app's line then
  reads `the operator typed that the jaws are empty: taken in place of the measurement
  for a return to rest`. The console command asks the same way; any other answer there
  ends it with `not confirmed that the jaws are empty; nothing sent`.
- **`on` and what the motors' state reads.** After a `rest` or a `release` the status
  line shows `rest_unpowered` and `torque off`, and a new task, a move, a jaw command and
  a takeover are refused with `the motors are off; switch them on again: type on at the
  Thor terminal or run the console command on`. `on` asks `Motors on again: the robot
  app connects the arm as at its start, and only an arm that lies at rest. Stay beside
  the e-stop. Type "yes" to switch the motors on: `; its RESULT is `completed` once the
  motors are on, and the robot app's line reads `[rest] the motors are on again at rest:
  motion rest_hold`. With the motors on it is refused with `the motors are on`.
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
| `takeover: not possible: <reason>` | The robot app's own word, `health.human.takeover`. The first of: a fault-stop is latched; the motors are off; `no fresh controller data: ...`; `a synthetic controller may drive only a simulated arm; ...`; `the live view is not fresh: ...; show the viewer's live picture in a visible browser tab` |
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
    state|stop|reset|rest|open|close|release|on|shutdown|takeover|handback
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

**A command that got no final RESULT in time is not withdrawn.** The console command waits
for the RESULT as long as the command may take (`rest` 180 s, `shutdown` 240 s, `on`,
`open` and `close` 60 s; `--timeout S` names another wait). When the wait runs out it
says `<command>: no final RESULT within N s (...). What was sent is not withdrawn: the
robot app may still do it. Run the command stop before anyone goes near the arm`: the
operator's commands carry no expiry, and only a STOP or a change of control owner keeps
the robot app from doing one that reaches it later.

Exit code 0: the command's final RESULT was `completed` (`state`: a ROBOT_STATE was there;
`stop`: the higher stop number was seen in the same session). 1: refused, failed, not
answered in time, not confirmed (for `stop` also: the robot app showed a new session), or no
robot link service. 2: a command that does not exist.

### The live view

The live view is the page of the experiment's viewer with its live picture showing, in a
visible browser tab. `run_viewer.sh` starts the viewer, and `run_workstation.sh` starts
it in its pane `viewer` (see [The experiment's viewer](#the-experiments-viewer)). The
viewer attaches to the robot link service in the role `viewer` and tells the robot app,
with a VIEW_STATUS every 0.2 s, how old the picture is that the page has on its screen.
**It only looks**: VIEW_STATUS is the only link message it sends.

- **What a takeover needs.** Where the robot app's file says
  `human.require_live_view: true`, as the rig's files `configs/robot_app_rebot.yaml` and
  `configs/robot_app_human_check.yaml` do, a person takes the arm over and drives it
  only while the view is live and at most `human.view_age_limit_s` old (0.5 s). For
  that, four things hold: the viewer runs and is attached to the robot link service; the
  camera's pictures arrive; the viewer's page is open in a browser on the workstation,
  in a tab that is visible; and the page shows the live picture, with at least three
  quarters of it inside the browser window. The live picture is at the top of the
  page's right column, under its button. The page then reads `Map picture` on that
  button, nothing beside it, and the line under the button ends with `The robot is
  told: live view, N s old.` The console's status line shows `view N s`.
- **What takes the live view away.** `Map picture` was pressed. A recording plays, the
  timeline was moved, a refusal or an older run was picked. The tab is hidden, or the
  browser is closed or frozen. The page is scrolled, or the window made small, so that
  less than three quarters of the live picture are inside the browser window. The page
  was loaded again in another tab on the workstation. The camera gives no picture. The
  viewer ended, was restarted, or was started with `--no-live`. The robot link service
  or the robot connection is down. Beside its button the page says why in a few words:
  `Map picture showing`, `tab hidden`, `scroll the live picture into view`. For the
  rest the words beside the button end with `see below`, and the note under the button
  says it in the warning colour: see **What the page's note says** below.
- **Only a page opened on the workstation is the live view.** The viewer gives a view
  token only to a page whose request comes from its own machine: the client's address
  is a loopback address or the address the request arrived on, both read off the
  connection and not off a header. A page opened from another machine is a watch-only
  page. It shows the live picture (`GET /api/live.jpg?after=N&watch=1`, which counts
  for nothing), with `watch only: see below` beside its button and `watch only: the
  live view is the page opened on the workstation` under it. Its acknowledgements are
  refused, and it never replaces the operator's page: watchers come and go while the
  operator's page counts. A second page opened on the workstation still replaces the
  first. The first then goes on showing the live picture as a watch-only page, and only
  a load of it makes it the live view again: a reload is a load.
- **What the page's note says** when its picture is not live or does not count. The
  note is the line under the button; it is then in the warning colour, and the few
  words beside the button point to it.
  - `the viewer cannot be reached; trying again every second; this picture is N s old`
    (beside the button: `viewer not reached: see below`). The viewer gives no answer: it
    ended, or the browser does not reach it. The page keeps its last picture, dimmed,
    says how old it is and asks again every second. Nothing is done at the page: start
    the viewer. A viewer that answers with an error is named instead: `the viewer
    answered 401; trying again every second`.
  - `the viewer was restarted: this page loads itself again` (`viewer restarted: see
    below`). The viewer that answers is not the one that served the page. The page
    loads itself again, gets a view token of the viewer that runs now and is the live
    view as after a reload by hand: nothing is done. It does so once for each start of
    the viewer and never while the viewer cannot be reached. A load that a page makes
    by itself takes the live view from no page: the page makes it only while the new
    viewer has given no page a view token. If another page was loaded on the new
    viewer first, this page says `watch only: ...` instead, as below. A hidden tab
    loads itself when it shows: hidden, it would take the view token before the page
    a person looks at.
  - `the viewer was restarted: this page loads itself again once Live picture is
    pressed`. The same, on a page that shows the map picture, a replay or an older
    run: a load would end that and make the view live with nobody at the page. The
    page waits and loads itself when `Live picture` or `Follow live` is pressed.
  - `watch only: the viewer was restarted; reload this page to make it the live view`.
    The page has loaded itself again for this viewer and is still not of it. Reload it.
  - `watch only: another page on the workstation is the live view; reload this page to
    make it the live view` (`watch only: see below`). A newer load of the page on the
    workstation has the view token. This page shows the live picture to watch and tells
    the viewer nothing. Use the other page, or reload this one.
  - `watch only: no page is the live view now; reload this page to make it the live
    view`. The same, and the page that counts asked for no picture in the last 2 s: it
    was closed, is hidden or shows the map picture, or it was no browser (a `curl` of
    the page on the workstation is a load too). Reload this page. A watch-only page
    never becomes the live view by itself, and a restart of the viewer does not make
    it load itself.
  - `watch only: the live view is the page opened on the workstation`. The page was
    opened from another machine. Open it on the workstation.
  - `no new picture from the viewer; this picture is N s old` (`no new picture: see
    below`). The viewer answers, and no picture has come to this page for 2 s.
  - With `no live picture` on the button the note is the viewer's reason: the camera.
- **What the operator sees when a takeover is refused.** The console rings three times
  and prints `[refused] owner request 'human' refused: the live view is not fresh:
  <why>; show the viewer's live picture in a visible browser tab`. Before anyone asks,
  `state` at the console prints `takeover: not possible:` with the same words, and the
  status line shows `view none`. The human program's note says `takeover refused: ...`.
  `<why>` is one of three. `no VIEW_STATUS has arrived`: the viewer is not attached;
  look at the pane `viewer`. `the view is not live`: the page does not show its live
  picture in a visible tab, or the camera gives none. `N s old, the limit is 0.5 s`:
  the picture on the page is old; the page stopped drawing, or the pictures arrive
  late.
- **What happens to a drive that runs.** It ends in a hold, with the motors on, and the
  person keeps the arm: `human drive ended: the live view is not fresh (the viewer
  reports no live picture)`, or `(N s old, the limit is 0.5 s)`. Once the live picture
  shows again, the grip is released and squeezed again, and the arm follows.
- **What to do.** Open the viewer's page and press `Live picture`, or `Follow live`.
  Keep that tab in front. Read the note under the button: a page whose note ends with
  `reload this page to make it the live view` is reloaded. After a restart of the
  viewer nothing is done: the page loads itself again, shows the live picture about a
  second after the viewer listens, and the view is live once the 3 s after the viewer's
  first ROBOT_STATE have passed.
- **How the page proves what it shows.** The page asks for each camera picture by its
  number (`GET /api/live.jpg?after=N`, answered once a picture newer than number N is
  there), draws it and then tells the viewer that it has: `POST /api/live/shown` with
  `{"seq": N, "visible": true, "token": "..."}`. When the live picture leaves the
  screen, the page says so at once with `"visible": false`. The view is live only on
  such a word, under the rules of **The view is live** and **The view token** below
  (`viewer_live/view_model.py`). Only the page opened last on the
  workstation counts. Both paths also follow the viewer's own access rule: its token, or none with
  `--no-auth`. A `POST` that names another site as its origin is refused.
- **The rate.** The page draws up to `--live-fps` pictures a second, 30 by default. A
  stand-in for a browser tab that ran the page's own script drew and acknowledged 30 a
  second, and the acknowledged picture was about 0.05 s old. In headless Chrome, on a
  recorded run of the rig with the live arm drawn beside it, the acknowledged picture was
  0.04 to 0.10 s old when the viewer sent its VIEW_STATUS (see **What the live arm
  costs** under [The experiment's viewer](#the-experiments-viewer)).
- **Without a robot link service** the viewer serves everything and says one line:
  `viewer: live view: not reported to the robot: cannot attach to the robot link
  service: ...; trying again every 5 to 60 s. The page is served as before`. It attaches
  by itself once the service runs, and says `viewer: live view: attached to the robot
  link service at <socket>; the robot is told how fresh the page's live picture is`.
  Its tries are 5, 10, 20 and 40 s apart and then 60 s, as the camera's are: after a
  service that was down for long, up to a minute passes before the view is live again.
  It looks for the service where the console does; `--link-socket PATH` names another
  socket.
- **One viewer at a time reports.** While another viewer is attached to the service, a
  viewer that starts later does not report: `another live view is attached`.
- **What this does not prove.** A visible tab says that the tab is in front, not that a
  person looks at it. A program of the same user can play the browser. And a browser on
  another machine whose connection a program on the workstation passes on (an SSH port
  forward, a proxy) counts as one on the workstation: its requests arrive from the
  workstation's own address.
- **The age limit is a first value.** The viewer sends a VIEW_STATUS every 0.2 s, so the
  age the Thor sees swings by about 0.2 s before any camera delay. Step F0 of the
  checklist chooses `human.view_age_limit_s` from measured ages.

- **The view is live** only while five things hold: the page that was loaded last
  acknowledged, with its live picture showing and less than 1 s ago, a picture of the
  camera connection; that page fetched that picture less than 1 s ago; a ROBOT_STATE
  younger than 1 s gives the Thor's clock; that picture arrived 3 s or more after the
  viewer's first ROBOT_STATE and after the newest step of the Thor's wall clock (more
  than 0.02 s against its monotonic clock between two ROBOT_STATEs); and that picture's
  exposure lies at most 0.05 s in the future of the Thor's clock. A picture that
  arrived earlier can be dated on the wall clock from before the step and is never
  live, however long the page acknowledges it: after the viewer's first ROBOT_STATE and
  after a step the view is not live for at least 3 s. A takeover is refused meanwhile,
  and a human drive ends in a hold, where the robot app requires the live view. The
  view age is the larger of two ages: since the picture's exposure, on the Thor's
  clock, and since its arrival on the workstation. The viewer sends VIEW_STATUS every
  0.2 s, also while the view is not live. These times are fixed values of
  `viewer_live/settings.py`: no argument and no file sets them.
- **The view token** is a secret the viewer gives to one page: the one whose
  acknowledgements count.
  1. Each load of the page from the workstation (`GET /`) makes a new token, makes it
     the only valid one and forgets every acknowledgement and every picture served
     before. So one page at a time counts, and a reload makes the view not live until
     the new page has drawn a picture.
  2. The page sends the token as the header `X-View-Token` with every
     `GET /api/live.jpg` and inside the body of every `POST /api/live/shown`. The token
     stands in no address.
  3. `GET /api/live.jpg` with the valid token is answered with a picture newer than the
     one the page names, and the viewer remembers that this picture went to this page,
     and when. Without the valid token the answer is 403 and no picture.
  4. `POST /api/live/shown` counts only with the valid token, for a picture that went to
     this page less than 1 s ago and that is not older than the one acknowledged last.
     A `"visible": true` for a picture that went to this page before the page last sent
     `"visible": false` does not count either: a tab that shows again is live with the
     next picture it fetches. What does not count is answered with 403 and changes
     nothing.
  5. A page whose token was replaced gets 403 for its next `GET /api/live.jpg`. From
     then on it acknowledges nothing more and asks for pictures to watch (`&watch=1`),
     which count for nothing. No request of it makes its token valid again: only a
     load of the page gives it a new one.
  6. `/api/live.json` holds no token and needs none. A page that sends its token with
     it, as `X-View-Token`, is told in `counting_page` whether it is the page that
     counts (`is_this_page`) and whether the page that counts asked for a picture in
     the last 2 s (`asks`). `loaded` there says whether any page has a view token: a
     page was loaded on the workstation since the viewer's start.
  7. A token is valid in one **viewer instance**: one start of the viewer, to its end.
     The viewer makes a name of 16 hex digits at its start. The name is in every page
     it serves, in `instance` of `/api/live.json`, and in the header
     `X-Viewer-Instance` of the 403 for a token that is not the valid one. A page that
     hears another name than its own was open across a restart of the viewer: it
     loads itself again, which is a `GET /` and gives it a token of the viewer that
     runs now. It does so once for each instance, and keeps the name it did so for in
     the tab's history entry, not in the browser's storage. It does so only while
     `loaded` of that viewer is false, in a visible tab, and not if it was replaced
     before: no page takes another page's token by a load of its own.
- **What the token closes, and what it does not.** It closes a stray
  `POST /api/live/shown` from a script or a test, a forgotten second tab and an
  acknowledgement of a picture that was never fetched. It fails closed: a reload, a
  second tab, a stray `GET /` from the workstation and a restart of the viewer each make
  the view not live, which only ever holds the arm. After a restart the page loads
  itself again; after the other three the page that was replaced says so and is
  reloaded by hand. It does not stop a program of the
  same user that loads the page and plays the browser, or that attaches to the local
  API as a viewer and sends VIEW_STATUS itself: the socket's file permissions and the
  viewer's own access rule are the boundary, as for every other local client.
- **The live picture is the left eye as the camera gives it**, at most `--live-width`
  wide. It is not rotated.
- **The view age gates a person's motion** wherever the robot app's file has
  `human.require_live_view: true`: without a live view, or with a view age above
  `human.view_age_limit_s`, a takeover is refused and a human drive ends in a hold (see
  [Demo 2: a person takes over](#demo-2-a-person-takes-over)). Nothing else acts on it.

### The experiment's viewer

```text
./run_viewer.sh [--runs FOLDER] [--host ADDRESS] [--port NUMBER] [--token-file FILE]
    [--no-auth] [--no-live] [--live-fps N] [--live-width PIXELS]
    [--receiver-socket PATH] [--link-socket PATH]
```

The **experiment's viewer** is `thor_viewer.py` of the experiment folder: a web page
for a browser on what a run recorded. `run_viewer.sh` starts it on the run folders of
the agent program; none of its code is copied or changed. With the **live picture**
the script starts it through a small launcher of this example (`viewer_live`), which
runs the viewer as it is, in its own process, and adds the camera's picture as it
arrives. The page with its live picture showing is the **live view**: the launcher
tells the robot app how old the picture on the page is (see
[The live view](#the-live-view)). The launcher also makes the page's arm the **live
arm**: the arm of each ROBOT_STATE as it arrives. With `--no-live`, or without
`AGENTIC_REMOTE_CAMERA_PR`, the script starts `thor_viewer.py` itself, alone: there is
then no live picture, no live arm and no live view, and a takeover on the rig is
refused.

- **What the page shows**: the obstacle map as the occupied cells of the map revision
  at that moment, in three dimensions; the measured joints as a stick figure of the
  arm, with the gripper's collision volumes; for each refusal of APAI's checker the
  pose that was refused, the point that was too near (the clearance point) and the
  obstacle cells near it; the camera frame the map took, with the pixels of the arm
  that were left out; and under it the timing graph of the run. **Follow live** follows
  the newest run while it is being recorded. The slider and **Play recording** replay
  a run, and the run selector picks an older one. The experiment's `THOR_VIEWER.md`
  explains the page.
- **The live picture.** The page's own picture is the camera frame a refresh of the
  obstacle map stored, and the page asks for it twice a second at most: it changes
  when the map does. The live picture is the camera's picture as it arrives on the
  workstation: the left picture of the stereo pair, as the camera receiver has it,
  made at most 960 px wide and packed as JPEG (quality 80), up to 30 a second. It is
  **shown and not stored**: the launcher keeps the newest picture in memory and
  writes no file. What a run keeps of the camera is what `--record-debug` records, as
  before. The live picture is the camera's picture as it is: not rectified, without
  the green mask of the arm and without the markers of a refusal, so it does not lie
  pixel for pixel on the stored one.
- **The two buttons.** While **Follow live** is active on the newest run and there is
  a live picture, the page shows it in the place of the stored picture. Moving the
  slider, **Play recording**, picking a refusal or an older run show the stored
  picture with its mask and its markers, exactly as the experiment's page does;
  **Follow live** brings the live picture back. The button over the picture names
  what a press gives: **Map picture** while the live picture shows (the stored
  picture, while the page goes on following the run), **Live picture** while the
  stored one shows (back to following live, with the live picture). The page is the
  live view only while the live picture shows.
- **The live arm.** The page alone draws the arm from the debug recording: it asks
  for its state twice a second, so the arm moves twice a second at best, and not at
  all without a recording. With the launcher, while **Follow live** is active on the
  newest run, the arm in the 3D view is the arm of each ROBOT_STATE as it arrives
  from the robot link service: 20 a second on the rig (`state.publish_period_s`
  0.05). That holds also when the newest run's recording has ended, and when no run
  was recorded at all: the agent program started without `--record-debug`. Nothing
  of it is stored. At its start the launcher prints `viewer: live arm: on the page
  and at /api/live/robot.json: each ROBOT_STATE of that robot link service as it
  arrives; shown, not stored`.
  - **What moves live, and at which rate**: the camera picture, up to 30 a second,
    and the arm with the gripper's collision volumes, 20 a second. Under the 3D
    view a second line gives the measured joints in degrees, the jaw opening, the
    control owner, the motion and the torque of that same ROBOT_STATE. A jaw reading
    older than 1 s reads `jaws: last read N s ago (M % open)`: the robot app takes no
    jaw reading during a move or with the motors off.
  - **The arm is called live only while its readings are fresh.** A ROBOT_STATE
    carries each joint's newest reading however old it is. With the motors off at
    rest (`rest_unpowered`), and when the joint feedback is lost, the robot app reads
    no joint and goes on publishing 20 states a second. The arm is live while the age
    of its readings (`health.reading_age_s`, and the readings' own times) plus the
    time since the ROBOT_STATE arrived is at most 1 s. Past that the page does not
    say `live`: it draws the pose the joints were last read at, and the line says in
    orange `arm: last read N s ago (torque off: the joints are not read)`, or
    `(joint feedback lost)`, or the fault the ROBOT_STATE names. N goes on counting.
    A ROBOT_STATE that does not say how old its readings are gives no arm. In the
    rig's recorded run of 2026-11-17 the readings were 0.06 s old at the median and
    0.21 s at most (1,319 ROBOT_STATEs: at rest, in holds and in moves), so 1 s is
    about five times the worst; the jaw reading was up to 20 s old during moves.
  - **What stays from the recording, at the page's own rate of twice a second**:
    the obstacle map, the refusals with their poses and points, the stored picture,
    the page's own text under the scene and the timing graph. The debug recording
    stores 2 pictures of the map a second.
  - **The line under the 3D view says which arm is drawn.** `arm: live, N a second;
    map: M s old` while live states are drawn: N states a second on this screen,
    beside a map that is M s old. `arm: last read N s ago (<why>); map: M s old`
    while the readings are old. M is the map's age in the page's own state plus
    the time since the recording's last event. While a recording runs that is the
    age of the newest stored map and up to half a second more, the page's own
    period; it grows for as long as a recording has stopped. `map: no
    recording` stands there without a recorded run, and `map: none recorded` before
    a run's first map. `arm: recorded` while the page draws its recorded state:
    after the slider was moved, while a recording plays, with a refusal or an older
    run picked. Then the page is exactly the experiment's own and no live state is
    drawn; **Follow live** brings the live arm back. `arm: no live state: <why>`
    when no ROBOT_STATE arrives: the viewer is not attached to the robot link
    service, the robot is not connected to it, or the newest ROBOT_STATE is older
    than 1 s. The page then draws what it draws alone: its recorded arm, or none.
    The same line stands there when a joint has no valid reading, and when no run
    folder names the arm's model.
  - **The arm is drawn by the experiment's own code.** The launcher computes the
    arm's points and collision volumes with the viewer's own routine for a
    recorded pose (`RunIndex.geometry`), on the joints of the ROBOT_STATE, and the
    page's own `draw()` draws them in the place of the recorded arm. The arm's
    model is the one of the newest run folder that holds `resolved-config.yaml`,
    recorded or not; the agent program writes that file at its start. When that run
    is not the run on the page, the line ends `; arm model of run <name>`. A
    ROBOT_STATE with another number of joints than that model is not drawn: `arm: no
    live state: the ROBOT_STATE has N joints and the arm model of run <name> has M`.
    In a runs folder without such a run folder no arm is drawn and the line says so;
    `/api/live/robot.json` still holds the readings.
  - **The page asks for each state** with `GET /api/live/robot.json?after=N`, which
    the launcher answers once its newest ROBOT_STATE is not number N. A page that
    draws slowly gets the newest state and misses the ones between; none is queued.
    A hidden tab and a replay ask for none. Each answer goes out in one write: a
    state was with a client that asks again at once 0.2 to 0.6 ms after the viewer
    had it, 0.4 ms at the median, on the workstation's loopback.
  - **Drawing the arm takes at most about half of the page's time.** The page's own
    `draw()` draws every cell of the obstacle map with each state. The page times
    each draw and asks for the next state only when as long has passed again; the
    viewer then hands it the newest. On a large map the line names what is skipped:
    `arm: live, 14 of 20 a second (a draw takes 25 ms); map: M s old`. The rest of
    the page's time is the live picture's, whose acknowledgements the robot app
    holds against `human.view_age_limit_s`.
- **`no live picture`** on that button means that the launcher has no camera picture
  younger than 2 s: `AGENTIC_REMOTE_CAMERA_PR` names no camera consumer's Python, the
  camera receiver is not running, no camera frames arrive, or the launcher's camera
  frame bridge ended. The text under the button says which. The launcher prints the
  same, as `viewer: no live picture: <why>`, with the pause before its next try
  (`trying again in 5 s`), once for each reason and each pause. The pause grows with
  every failed try: 5 s, then 10, 20 and 40 s, and 60 s from then on; once pictures
  arrived it is 5 s again. The launcher prints
  `viewer: live picture: the camera's pictures arrive` when they come. Until then the
  page is the experiment's own, and the recorded runs are served as before.
- **The live picture's arguments**: `--no-live` starts the viewer alone. `--live-fps N`
  is the most pictures a second, 1 to 60, 30 by default. `--live-width PIXELS` is the
  picture's greatest width, 64 to 4096, 960 by default; a narrower camera frame is
  not made wider. `--receiver-socket PATH` is the camera receiver's socket, by
  default `/tmp/camera_receiver_avs1_cuda_ipc.sock`, the `ipc.socket_path` of
  `configs/camera_receiver_rebot.yaml`. `--link-socket PATH` is the robot link
  service's local socket, for the live view; by default the one the console attaches
  to. A number outside its range ends the launcher
  with exit code 2 and one line. With the camera frame bridge as it is, at most every
  second camera frame reaches the page, 30 a second of the camera's 60, whatever
  `--live-fps` asks for: the bridge proves each copy by the camera frame after it.
- **The live picture's addresses**: `http://127.0.0.1:8766/api/live.mjpg` is the
  stream, one JPEG after the other in one answer (`multipart/x-mixed-replace`).
  `/api/live.json` says whether there is a live picture and why not (`available`,
  `reason`), how many pictures a second were asked for and made (`fps_asked`,
  `fps_measured`), the picture's width (`width`) and the age of the newest one
  (`age_s`), and in `view` what the robot app is told of the live view (`attached`,
  `why`, `live`, `age_s`). It names the viewer instance that answers (`instance`: one
  start of the viewer) and says in `counting_page` whether a page has a view token,
  whether the page that asks is that one, and whether that one asks for pictures
  (`loaded`, `is_this_page`, `asks`). The page itself does not use the stream: it asks for
  `/api/live.jpg` and answers with `/api/live/shown`, as
  [The live view](#the-live-view) tells. `/api/live/robot.json` is the newest
  ROBOT_STATE for the live arm: its number in the launcher's count (`n`), whether
  the arm is live and why not (`available`, `why`), its age on the
  workstation (`age_s`), how old its joint readings are now and the limit of live
  (`pose_age_s`, `stale_s`), the age of old readings and why they are old
  (`last_read`), the run of the arm's model (`model_run`), the states of the last
  second (`per_second`), the measured
  joints with the last joint target (`joints`, `target_q`), the jaw opening
  (`gripper`), `owner`, `stop_level`, `motion`, `torque_on` and `torque`, and in
  `robot` the arm as the page draws it. It needs no view token and a watch-only page
  of another machine reads it too: it tells the robot nothing. All five follow the
  viewer's own access rule: with a token they need the token, as the page's cookie
  or as an `Authorization: Bearer` header; with `--no-auth` they are open.
- **What the live picture costs.** One picture of the rig's camera (2560 x 1984)
  takes about 16 ms of one processor core to make 960 px wide. The page's pictures at
  30 a second take about 0.6 of a core in the viewer and 0.15 in its bridge, measured
  on 2026-11-17 on the workstation with a stand-in for the camera receiver. That is
  spent only while a browser shows the live picture: with no page open, the tab
  hidden, a replay or **Map picture**, the camera is asked for one picture a second.
  Every browser gets the same pictures, each made once; one that reads slowly gets
  the newest and misses the ones between.
- **What the live arm costs** grows with the map, as each state is a draw of every
  map cell. Measured on 2026-11-17 on the workstation in headless Chrome with
  software rendering, over 8 s each, with a stand-in camera (60 pictures a second,
  1280 px wide) and a stand-in robot app (20 ROBOT_STATEs a second):
  - On the rig's recorded run `20261118T030701Z-3363330`, whose map has 3,837
    cells: 20 states a second drawn; one `draw()` took 4.4 ms and `draw()` 0.11 of
    the page's time; the browser's tab took 0.37 of a core, and 0.24 with no arm
    drawn. The acknowledged picture was 0.04 to 0.10 s old when the viewer sent its
    VIEW_STATUS, 0.07 s at the median; with no arm drawn 0.06 to 0.12 s.
  - On a made-up map of 5,000 cells: 20 states a second; 5.8 ms a draw, 0.14 of the
    page's time; tab 0.43 of a core, 0.23 without the arm; picture 0.06 to 0.07 s
    old.
  - On a made-up map of 20,000 cells the page skipped states: 14.7 of the 20 a
    second; 24.6 ms a draw, 0.46 of the page's time; tab 0.88 of a core, 0.30
    without the arm; picture 0.06 to 0.10 s old, 0.06 s without the arm.

  Every VIEW_STATUS of these runs said live. The viewer's process, with the stand-in
  camera and robot app in it, took 0.26 to 0.29 of a core.
- **It only looks.** The viewer reads the files of the runs. For the live picture
  the launcher reads the camera receiver as well, through a camera frame bridge of
  its own: a further reader, as the agent program's bridge is one. For the live view
  it attaches to the robot link service in the role `viewer`, where it reads
  ROBOT_STATE, for the Thor's clock and for the live arm, and sends VIEW_STATUS: no
  command, and no other link message. It attaches to no motor and writes nothing into a
  run folder. The bridge's log and socket are in a folder of the launcher's own under
  the temporary folder, removed at the end. It can be started and ended while the
  agent runs. The camera receiver sends to its readers one after the other and waits
  up to 2 s on one that does not read; the agent's bridge gets no camera picture
  meanwhile. A camera frame bridge therefore loads CuPy and makes its CUDA context
  before it subscribes to the camera receiver, and reads from then on. Between the
  two it only opens the receiver's slots: 2 ms for the rig's 24 slots on the
  workstation's GPU 0, measured on 2026-11-17 without the camera receiver. In
  `test_frame_bridge.py`, on a stand-in for the camera receiver, a bridge kept its 20
  pairs a second while a second one, whose CuPy takes 3 s to load, started beside it.
  This is not yet tried beside the agent on the rig's camera receiver. A bridge that
  cannot start is tried again after the growing pause above; `--no-live` starts none.
  Ctrl+C ends the viewer, with its bridge, and nothing else; its last word to the
  robot app is that the view is not live. Nothing it does can move the arm. Without
  the live view a person's takeover is refused and a human drive ends in a hold.
- **Start it** in this folder, with the three variables set: `./run_viewer.sh`. It
  needs `AGENTIC_REMOTE_AGENT_PYTHON` (a Python with APAI's packages, numpy, OpenCV
  and PyYAML), `AGENTIC_REMOTE_EXPERIMENT_DIR` and `AGENTIC_REMOTE_APAI_SRC`, as
  `run_agent.sh` does; the live picture is on when `AGENTIC_REMOTE_CAMERA_PR` is set
  as well, and the script's lines say whether it is. The viewer prints its address,
  `http://127.0.0.1:8766/`: this machine only, on the viewer's own port. The viewer
  needs no GPU; the live picture's bridge copies the camera frames off GPU 0, where
  the camera receiver keeps them.
- **The page opens with a token.** The viewer answers a browser only after it was
  opened once with its access token: `http://127.0.0.1:8766/?token=<token>`. The
  script's second line gives `?token=<token>` to put behind the address. At the very
  first start the viewer still has to make the token; the line then names its file,
  and `echo "http://127.0.0.1:8766/?token=$(cat .state/viewer.token)"` prints the
  link. The file is `.state/viewer.token` beside the script, readable by its user
  only. `--no-auth` serves without a token: every user and program of the machine can
  then read the camera frames, the maps and the timings.
- **The picture of the obstacle map needs `--record-debug`.** The viewer lists a run
  folder that holds `debug/events.jsonl` and `resolved-config.yaml`: a run of the agent
  program that was started with `--record-debug`. A run without it is not listed and
  cannot be asked for by its name, and the viewer shows nothing of it, also not its
  timing; that run's timing graph is the file `timeline.html` in its run folder. A
  run with `--record-debug` that was recorded before the agent program wrote
  `resolved-config.yaml` is not listed either.
- **`--runs FOLDER`** names the folder that holds the run folders. The default is
  `.state/runs` beside the script, where the agent program puts its runs with the
  shipped `state_dir`; the script makes it when it is not there, with the `.gitignore`
  the agent program gives the state folder. For another `state_dir`, or for runs that
  were given a `--run-dir`, name their folder. A `--runs` that names no folder is
  refused.
- **The other arguments are the viewer's own**: `--host`, `--port`, `--token-file` and
  `--no-auth`. They go to the viewer as they are, behind the script's own `--host
  127.0.0.1`, `--runs` and `--token-file`, so the caller's win. With another `--host`
  the page is served to that network, over HTTP without encryption.
- **The script names the runs and the token** because the viewer's own places for both
  lie inside the experiment folder, which this example only reads.
- **APAI's own window and its own server stay off.** The agent program starts APAI
  with `--no-view` and `--no-serve`; the page for a run is this viewer.
- **Exit code 2 and one line, nothing started**: one of the three variables cannot be
  used, the experiment folder holds no `thor_viewer.py`, an argument that takes a
  value has none, `--runs` names no folder, or the state folder cannot be made.

**What differs from the viewer on the experiment's own recordings on the Thor**:

| On the Thor, with the experiment's launcher | Here, with the agent program on the workstation |
|---|---|
| The measured joints are the motor driver's reads: every full read, and single joint reads put together at up to 10 per second. The figure follows the motors all the time, also through the shutdown | The measured joints are the poses the agent program took from ROBOT_STATE: one per ROBOT_STATE, at most 10 a second (source `ROBOT_STATE over the robot connection`), and with the stereo camera one per camera frame the map took, at its exposure (source `ROBOT_STATE samples at the capture time`). So the figure follows the arm steadily while the agent program is attached |
| The age shown beside the pose is taken on the motors' machine | A pose's time is the ROBOT_STATE's receipt on the workstation less the age of its joint readings. The way over the robot connection is not in it: the pose is older than the page says by that much |
| The recording holds every motor command and every single motor read (`robot.call`, `robot.return`) and where each self mask's pose came from (`mask.state_source`) | None of the three is recorded. The page draws nothing from them but the poses of the row above; a tool that reads a recording for motor commands finds none |
| The figure also moves while a person holds or moves the arm and while the arm returns to rest: one program does all of it | While a person drives the arm, or while the robot app returns it to rest, the figure follows for as long as the agent program is attached. After the agent program has ended nothing more is recorded; the robot app holds the arm on |
| A refusal is one of the one checker there is | A refusal is one of APAI's checker in the agent program: of planning and of the check ahead. What the robot app's own checker on the Thor refuses is not in the recording; it shows on the console and in the task's result |
| The camera frames are those of the camera on the same machine | The camera frames are those APAI took on the workstation: the stereo pairs of the camera connection with the depth computed there |
| A run folder is named by its start time | A run folder is named `<UTC time>-<process id>`. "Newest" is the name that sorts last, so a folder named by hand with `--run-dir` sorts by that name |
| The timing has the motor reads and sends and the camera's capture | The timing has neither (see [The options of the experiment's launcher](#the-options-of-the-experiments-launcher)) |

The same in both: the map revisions and their camera frames, the refusals of APAI's
checker with their clearance points, the collision volumes, the timing graph of what
the agent program did, a `bookmark.json` a person puts into a run folder, following a
run and replaying one.

**What was run and what was not.** The viewer was run by tests (`test_viewer.py`), on
127.0.0.1 and a port the system picked, on runs the agent program recorded on APAI's
mock scene with the experiment's own recorder and a stand-in for the obstacle grid
bridge: it listed the run from its start, its answers grew while the program ran, the
measured joints went where the arm went, a refusal of the check ahead showed with its
pose and its point, and a run without `--record-debug` was left out. One test starts
it with `run_viewer.sh` as a process and asks it with its token. With the full stack's
configuration the viewer was only asked to build the arm and its collision volumes.
The live picture was run by tests too (`test_viewer_live.py`). From a stand-in camera
of 60 pictures a second the stream gave between 20 and 30 different pictures a
second, each newer than the one before; a browser that read one picture in 0.3 s
missed most and did not slow another; every path of the viewer answered under the
launcher with the same bytes as the viewer alone, the page with the block before its
`</body>` and nothing else; without the token the four live paths answered 401, as
the viewer's own. The live view was run by those tests as well. A stand-in for a
browser tab, and node with the block's own script over HTTP, fetched, drew and
acknowledged the numbered pictures: 30 a second from node. On the rig's robot app file
with human driving switched on in memory, behind a robot link service that ran as a
process, a takeover was taken and a drive ran while the tab acknowledged, on the
simulated arm. The takeover was refused and a running drive ended in a hold, with the
motors on, after each of five things: the page said that the live picture shows no
longer, the tab said nothing more, the camera stopped, the viewer ended, and the page
was loaded again elsewhere. On a clock that a test sets the viewer sent, situation by
situation, the VIEW_STATUS its rule gives: not live in twelve situations and live in
six, with the picture's age. A viewer started without a robot link
service served everything and attached once one ran. The whole launcher was started with `run_viewer.sh` on a fake
camera receiver, with the real camera frame bridge program on files in the place of
the GPU's slots and camera frames of the rig's size: the pictures came 960 px wide,
no file changed in the runs folder or beside the token, and after Ctrl+C, SIGTERM and
SIGHUP the bridge's process was gone. The block's script was run by node, through each
switch of the two buttons: on a stand-in for the page, and behind the page's own
script on stand-ins for the page's elements, where the page's own handlers for the
slider, **Play recording**, a refusal, the run selector and **Follow live** switched
the picture.
The live arm was run by those tests too, on a stand-in for the robot app that gave
the viewer 20 ROBOT_STATEs a second with a moving first joint. A client that asked
again at once got every state in its order, and one that asked every 0.3 s got the
newest each time, at most one state behind. The path answered 401 without the
viewer's token and answered a client of another machine with it. Behind a robot link
service that ran as a process, every state the robot app published on the rig's file
reached the path. The arm's points were those the viewer's own `RunIndex.geometry`
gives for the same joints, on a recorded run and on a run folder without a recording.
The viewer's own paths answered with the same bytes as the viewer alone, nothing was
written, and VIEW_STATUS stayed the only link message sent. The page's own script
with the block ran under node: its `draw()` was given each live state as it came, 19
to 20 different ones a second over HTTP, and none after the slider was moved, while a
recording played, with an older run and with a moment picked in the timing graph;
without a recorded run the arm still moved; without a state the page said why and
drew what it draws alone. Headless Chrome on the viewer's real page did the same: 20
different live states a second while following, on a finished recording and without
any, the recorded arm alone through the slider, a replay and an older run, and `arm:
no live state: the robot link service is gone (...)` once the stand-in service ended.
States whose readings stood still, 20 a second and 45 s old, with the motors off and
with the joint feedback lost, were never called live: not by the path, not under node
and not in Chrome, where the line read `arm: last read 45 s ago (torque off: the
joints are not read); map: ...` and the pose of the last reading was drawn. The robot
app's own ROBOT_STATE, built 45 s after its last read, was not live either; with
fresh readings it was.
Not run: the page in a browser with a window and a person in front of it (headless
Chrome ran it in tests), so no person saw the arm move; the live arm beside a
recording that is still being written and on the rig's robot connection (the rig's
recorded run and its arm model were used for the cost figures above, with a stand-in
robot app); no browser's acknowledgement reached a robot app on the rig;
the live picture on the real camera stream, with CuPy and the GPU;
the live view on the rig's camera, robot connection and real arm;
the viewer on a recording of the real full stack, with its GPU, its obstacle grid
bridge and the stereo camera; port 8766; and the pane `viewer` of
`run_workstation.sh`.

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
  the joint target the human program sends ahead of the measured pose before the Thor's
  drive has sent a setpoint, and after that ahead of where the Thor's own last joint
  target can be by now (`ik.max_ahead_rad`), and the hand ahead of the gripper tip of
  that joint target (`ik.max_ahead_m`). **Jump**: an IK solution far from the joint
  target the solve started from.
- **Catching up**: the hand may be ahead of the arm. The arm goes after it at the speed
  its joints are allowed, and stops where the hand stopped, or at once when the grip is
  released ([How fast the arm follows the hand](#how-fast-the-arm-follows-the-hand)).
- **Needs-release flag**: the Thor's note that the next human drive needs a released grip
  first.
- **Jaw opening**: how far the jaws are open, as a share of their travel: 1.0 is fully
  open, 0.0 closed. A HUMAN_SETPOINT carries the jaw opening the trigger asks for
  (`jaw_opening`), or none. **Grip level**: the jaw opening in force when a person takes
  the arm over; the trigger moves the jaws only once the opening it asks for has reached
  it.
- **Human checker**: the robot app's checker of the human limit set (`limits.human`).
  The **agent's checker** and the **recovery checker** are its checkers of
  `limits.agent` and `limits.recovery`: the first checks the agent's moves, the second a
  return to rest. The agent's checker is not the workstation checker, which runs in the
  agent program.
- **Table clearance**: the height over the table plane under which a checker takes only
  a step that rises. The agent's checker has `safety.table_clearance` of APAI's arm
  profile, and a return to rest starts only above `rest.table_clearance_m` of the robot
  app's file: 2 cm each on the rig. The human checker has its own,
  `limits.human.table_clearance_m`: 0.0 on the rig, the table plane; in a file that does
  not name the key, the higher of the other two. **Grasp region**: the column around an
  object in which a move of the agent may go below its table clearance, for a grasp; the
  robot app keeps the last one. **Low gripper tip**: a gripper tip under a checker's
  table clearance; for a return to rest also one inside the agent's last grasp region.

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
                                                      limit), jump check, joint speed
                                                      limit, ahead checks
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
   second, as a status. Each is under 600 bytes as it is sent.
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
speed is clamped, and `health.clamped` counts it). The human checker checks the step and
it goes to the motors: every pass of a drive sends a setpoint. The table is that
checker's: it refuses a step that ends under its table clearance and does not rise (the
floor under [A takeover, step by step](#a-takeover-step-by-step)). A setpoint that goes
out late is not made up for: the step keeps its size.

**No obstacle check for a person**, as the design says: the human program does not read the
obstacle grid and sends no PERMIT. What stands in for a permit is a fresh engaged human
target, checked by the Thor before every setpoint, and the person's eyes on the arm. The
person's eyes are the obstacle check. On the rig the file says
`human.require_live_view: true`: the Thor then also checks a fresh live view before
every setpoint, which is the viewer's page with its live picture showing (see
[The live view](#the-live-view)). Where a file says `false`, no program checks what
the person sees. What the human checker enforces without a camera: finite values, the human joint
box and the speed limit of `limits.human`; with APAI's checker also the tip box and the
table clearance of the human limit set (`limits.human.workspace_min`, `workspace_max` and
`table_clearance_m`; those of APAI's profile where the file names none) and the
profile's margin from the model's joint limits. The agent's last grasp region does not
bind a person's drive: inside it the gripper tip is lowered as anywhere else. It binds
the typed `rest`, which does not start inside it. The agent's **contact spots** do not
bind a person. A
contact spot is the place where a move of the agent or a return to rest ended at a
contact; the agent's checker and the recovery checker keep it and refuse a move that
enters it from outside (with APAI: within 5 cm of the gripper tip's place at the
contact).

The way to the robot app is part of a sample age and of a view age. Each is what the
sender measured as it sent, plus the time since then on the Thor's monotonic clock: the
sender reads that clock off its newest ROBOT_STATE and says when it sent (`sent_mono_s`).
The reading is never later than the true moment, so an age is never counted as less
than it is. A sample age is counted as more by the ROBOT_STATE's way to the human
program. A view age is not: the viewer's own age is short by that same way. For a sample
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
from a contact hold, here it is taken while a takeover is open; and a takeover is also
refused while the human checker refuses the measured pose for anything but a low
gripper tip, which the design does not ask. The rule for the trigger and the jaws is
built as the design states it (step 5 of its section 12).

### A takeover, step by step

| # | What happens | Link messages | Typed or pressed | What the operator sees |
|---|---|---|---|---|
| 1 | The agent fails while it owns the arm | OWNER_REQUEST "help_needed" from the agent program, or from the service with `agent program gone`; EVENT `help_needed` | - | Console: the help block with the reason, the jaws, the time left and the takeover line, `takeover: possible now ...` or `takeover: not possible: <why>`. The viewer's page draws no EVENT: while it draws the live arm, the line of readings under its 3D view then reads `owner nobody` |
| 2 | The assistance wait counts down | ROBOT_STATE `assistance_wait_s`; EVENTs with bells | - | Console: `wait N s`, bells, `ARM WILL MOVE TO REST`; where the timed return is off, as on the rig, `ARM WILL KEEP HOLDING` |
| 3 | The person gets ready | HUMAN_SETPOINT, as a status | Puts the headset on, picks up the right controller, stands behind the arm, recentres the headset view with the grip released | Status line: `headset ready`, `controller 0.0x s` and the human program's note, `note: hold right B for 1 s with the grip released to take over` |
| 4 | Right B is pressed | STOP with the reason `right B` and the sender `controller` | **Right B** | Console: `[stop] STOP from controller: right B; the arm holds, stop number N; the assistance wait is cancelled`. Nobody owns the arm |
| 5 | B stays down for `buttons.hold_s` (1.0 s) with the grip released, every pass fresh | OWNER_REQUEST "human" with the reason `right B held 1.0 s`, quoting the ROBOT_STATE that shows the new stop number | **Hold right B 1 s**. Nothing is typed | Accepted: EVENT `[owner] TAKEOVER: a person owns the arm, owner number N; release the grip, then squeeze it to move` and two bells; with the gripper tip under the human checker's table clearance the line goes on `; the gripper tip is low: only a lift is taken until it is clear`. No jaw command is sent: the robot app notes the grip level. Refused: `[refused] owner request 'human' refused: <reason>` and three bells; the note then reads `takeover refused: <reason>` |
| 5b | The typed fallback, only with `operator.typed_takeover: true` in the service's file | OWNER_REQUEST "human" with the reason `typed at the operator console` | `takeover`, then `yes`, at the console. The control owner must be nobody: type `stop` first if the agent owns the arm | The same |
| 6 | The grip is squeezed. The Thor starts a human drive | HUMAN_SETPOINT, engaged, with the joint target | **Squeeze the grip** | Status line: `moving N/0` and `human driving`. The arm follows the hand, starting from the measured pose. Where a joint lay on its stop, as at rest, the arm first lifts that joint 0.025 rad off the stop |
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
  live view is not fresh: ...; show the viewer's live picture in a visible browser tab`); a full
  joint read; after it the controller data, their source and the live view once more,
  because both ages count when the person becomes the control owner (such a refusal ends
  `; the joint read took ... s`, as in `no fresh controller data: 0.268 s old, the limit
  is 0.25 s; the joint read took 0.048 s`; the robot app takes up no link message while
  it reads the joints, so the data must be younger than the limit less the read, about
  0.05 s, when the takeover is asked); the
  human checker accepts the measured pose (`the arm is outside the human limits: <reason>;
  type rest`), where a gripper tip that is only under that checker's table clearance is
  no reason to refuse (the floor below). With the rig file that refusal also comes in
  the outer 0.02 rad of the agent's working box, where the typed `rest` is taken (see
  [The rig files](#the-rig-files)). A STOP or a change of control owner that came
  meanwhile wins. The view is
  not live for 3 s after its start and after a step of the Thor's wall clock (see
  [The live view](#the-live-view)): a takeover is refused meanwhile.
- **A refused takeover after a STOP that is not confirmed.** A STOP sent between a new
  robot connection and its first ROBOT_STATE keeps HUMAN_SETPOINTs at the robot link
  service until a later STOP is counted. If a takeover is refused with `no fresh controller
  data` while the console's status line says `a STOP is not confirmed yet`, press right B
  again. This holds at the robot link service only. At the robot app's own socket
  (`--human-socket`) a STOP sent before the robot app's first ROBOT_STATE, that is while
  the arm is still being connected, is served on the robot app's first pass, and that
  first ROBOT_STATE ends the hold: no second STOP is needed.
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
    command at a time and always toward the newest opening. A close under way is cut
    when the trigger asks for an opening wider than that close's by more than
    `gripper.reach_tol` (0.03): the close ends, the jaws get the light hold, and the
    next jaw command sends them the newest opening. An opening step under way, 0.05 of
    the travel, runs to its end. A new jaw command goes out only when the opening
    differs from the one the jaws settled for by more than `gripper.reach_tol`, or asks
    for fully open; a trigger released all the way always ends with the jaws at 1.0.
    Jaws that an opening of the agent left short of the grip level (it timed out, or
    was blocked) open to what the trigger asks as soon as the trigger has matched.
  - **No tighter than the agent** until the jaws were measured fully open once since
    the takeover: an opening below the grip level is held at the grip level meanwhile
    (`no closing beyond the grip level until the jaws were measured fully open`).
  - **A close under the trigger ends in the light hold** when the jaws stall on an
    object: the robot app sends it. A close of the agent keeps its stiffness instead
    (see [What the jaws do](#what-the-jaws-do)). **An opening** goes one step of
    `gripper.open_step` (0.05) per jaw command, each through the driver's guarded ramp.
  - **A close needs the live view fresh** where the file requires it; an opening does
    not (`no closing: the live view is not fresh: ...`). Without it the jaws stay where
    the jaw command under way leaves them: an opening stops at the end of its step when
    the trigger asks for less, no close follows it, and a close that was begun with a
    live view runs to its end or to its stall and is not taken up again. The gripper
    worker reads the jaws anew before each step and sends no close without that leave.
  - **An opening that something blocks**: the step that stalls or times out short of
    its target is left at the light holding stiffness, the jaws stay with the trigger,
    and `health.human.jaws.refusal` reads `the jaws stalled while opening: check what
    is against them; they are at 0.70` (the console: `jaws: follow the trigger; the
    jaws stalled ...`). The step is not tried again until the trigger asks for an
    opening that differs by more than `gripper.reach_tol`.
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
    moves nothing or the jaws are not where it asks (held wider, or stalled short of an
    opening), such as `the trigger has not reached the grip level`.
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
- **The table under a person is a floor.** With APAI's checkers the human checker has a
  table clearance of its own, `limits.human.table_clearance_m` of the robot app's file:
  `0.0` in the rig file, the table plane of the arm model. In a file that does not name
  the key it is the higher of `rest.table_clearance_m` and APAI's
  `safety.table_clearance`. The robot app names it in WELCOME as a height
  (`human_tip_floor`).
  - **The tip stops 0.5 cm above it and moves freely above that.** The human program
    keeps the gripper tip of every pose it solves for 0.5 cm above the named height
    (`TIP_FLOOR_MARGIN_M` of `human/types.py`): 0.5 cm above the table plane on the
    rig. The tip comes down to that stop, slides along it in every direction while the
    hand pushes down, and comes away from it with the hand; the note says
    `at the workspace limit: table`. Nothing else binds a person near the table. A
    squeeze with the gripper close to the table starts like any other squeeze, a level
    hand moves the tip sideways, and inside the agent's last grasp region the tip is
    lowered as anywhere else.
  - **A tip under the stop at a squeeze rises to it.** An arm that sags under a joint
    target at the stop is measured under it at the next squeeze. The human program
    then takes the tip up to the 0.5 cm, no faster than the hand speed limit, whatever
    the hand does, and follows the hand sideways meanwhile.
  - **Under the human checker's clearance only a step that rises is taken.** That is
    APAI's rule: its checker refuses a step that ends under its table clearance and
    does not rise, a slide at one height included (`TCP z=... below table clearance
    ... outside the grasp exemption zone`); a person's drive opens no grasp exemption
    on that checker. The human program sends no joint target whose tip is under its
    stop, so its drives do not meet this rule. A HUMAN_SETPOINT from elsewhere that
    asks for a tip under the clearance is followed down to the clearance and then ends
    the drive in a hold: `human drive ended: TCP z=... below table clearance ...`.
  - **A takeover with the measured tip under that clearance is taken.** On the rig
    that is a tip the arm model puts under the table plane. The human checker then
    checks the measured pose without its table clearance for the gripper tip, and the
    takeover's line ends `; the gripper tip is low: only a lift is taken until it is
    clear`. From there the human program solves straight up until the tip is at the
    named height, and then as in the item above.
  - **The arm's joints stay 1 cm above the table plane**, for every owner: a step
    that brings a joint between the base and the tip lower is refused (`link/joint N
    at z=... would hit the table`) and ends the drive.
  - **The typed `rest` keeps its own table rule.** A return to rest starts only with
    the gripper tip above `rest.table_clearance_m`, 2 cm on the rig, and outside the
    agent's last grasp region (`the gripper tip is at z=... m, below the table
    clearance ... m`, or `the gripper tip is inside the last grasp region`): its
    straight joint path could drag the gripper over the table. So before `rest`:
    squeeze and lift until the tip is above the 2 cm and outside the region, release
    the grip, type `rest`, then `path is clear`.
  - **The agent and a return to rest keep the table clearance and the tip box of
    APAI's profile.** `limits.human.table_clearance_m`, `workspace_min` and
    `workspace_max` are keys of the human limit set alone: the loader refuses them
    under `limits.agent` and `limits.recovery`.
  - **A takeover is still refused for any other complaint of the human checker**
    (`the arm is outside the human limits: ...; type rest`): a tip outside the human
    checker's tip box, a link at the table, the human joint box.
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
| `the jaws are not proven empty (...)` | The jaws may hold something, or their readings do not prove them empty | Look at the jaws. If they are empty: type `jaws are empty` when the console or the Thor terminal asks, then answer the command's own question again. If they hold something: `stop`, then `open`, then `yes`; wait for the opening's RESULT; then `rest`. It needs no controller. With jaws that are not referenced (`the jaws are not proven empty (unknown): the jaws are not referenced`) and that may hold something, `open` is refused too (`the jaws are not referenced`): see the rule on a hand-back with such jaws above. With `--arm apai`, jaws without their reference that hold nothing are referenced by the first jaw command |
| `the gripper tip is inside the last grasp region`, or `the gripper tip is at z=... m, below the table clearance ... m` | The return starts only with the gripper tip clear of the agent's last grasp region and above `rest.table_clearance_m`, 2 cm on the rig. A person's drive is bound by neither: it goes down to 0.5 cm above the table plane (the floor above) | Take over, squeeze and lift the gripper above the 2 cm and out of the region, release, then `rest` |
| `joint index N at X rad is outside the tested recovery region ...` | The start pose is outside `rest.region_*`. A person cannot leave the arm outside `rest.region_*`: with `human.enabled` the robot app refuses a file whose human joint box is not inside it. In the rig file the box lies 0.02 rad inside the region, except at the lower bound of the second and the third joint, so a joint driven to the edge of the box that reads a few thousandths of a rad outside the box is still inside the region, and this refusal does not come. In a file whose box equals its region such a reading on a real arm lies outside the region | With the rig file: nothing, the typed `rest` is not refused for this. With a box that equals the region: squeeze and move that joint back, release, then `rest`. That needs the person as control owner. After a STOP or a fault-stop at such a pose nobody owns the arm and a new takeover is refused too (`the arm is outside the human limits: ...; type rest`): no command moves the arm then (see [Known gaps](#known-gaps)) |
| `the path to rest was refused: setpoint N: <APAI's reason>` | The return moves every joint straight to its rest value, and the recovery checker checks that whole path first. From some poses it passes under the recovery checker's table clearance, leaves the workspace box of APAI's profile, which that checker keeps, brings a link to the table or tips the camera on the arm's wrist further than APAI's profile allows (its **wrist camera rule**). Such a pose has a compound turn of the wrist or the base | Squeeze and move the arm back the way it came, release, then type `rest` again |

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

### How fast the arm follows the hand

The numbers are those of the rig's two files, computed on APAI's arm model and not
measured on the arm.

- **The joints and the hand speed limits set the pace, whichever binds.** The Thor
  follows a person's joint target at 0.98 of `limits.human.max_joint_speed`, 1.6 rad/s
  by the rig owner's decision of 2026-11-18: 1.57 rad/s, which is 1.8 degrees with each
  setpoint. The robot app names that limit in WELCOME (`human_max_joint_speed`). The
  human program moves the joint target it sends at 0.9 of the Thor's speed: on the joint
  that has the farthest to go 1.41 rad/s, 81 degrees a second, and every other joint by
  the same share of its way. The Thor's own last joint target is then never left
  behind, whatever the hand does. A robot app whose WELCOME names no such limit is
  followed at 0.09 rad/s. On the rig's arm profile the robot app takes a human limit
  set of up to 2.0 rad/s (`HUMAN_SPEED_CAP` of `robot_app/apai_backend.py`).
- **The gripper tip** then moves at 27 to 36 cm a second from the ready pose, the arm
  raised over the table, by its direction; at 16 to 32 cm a second where the arm is
  folded near its base; and up to the 50 cm a second of the bound where the arm is
  stretched out and the tip moves sideways. `retarget.max_linear_velocity` (0.5 m/s)
  and `retarget.max_angular_velocity` (2.0 rad/s) of `configs/human_rebot.yaml` bound
  the tip's target: 1 cm and 0.04 rad a pass.
- **A hand that is ahead is caught up with.** The arm goes after the hand at that speed
  and stops where the hand stopped, with the hand's turn. So after the hand stops, the
  arm travels on by as far as the hand was ahead: 10 cm ahead is about a third of a
  second from the ready pose. The note on the console and the human program's report
  line say `hand N cm ahead: the arm is catching up` from 5 cm on.
- **While that note shows, the arm goes straight to where the hand is now**, not along
  the way the hand took. A hand that went up, over something and down again faster
  than the arm follows takes the gripper through what the hand went round. Go round
  things no faster than the arm follows, with no such note showing, or release the
  grip first.
- **To stop the arm, release the grip.** No joint target follows the pass that read the
  released grip, 0.02 s: the arm stands at the last setpoint the Thor sent, at most one
  setpoint further than it was, 1.8 degrees a joint, which in the ready pose is 1.6 cm
  at the gripper tip for one joint and up to 4.6 cm with every joint at its speed
  limit. That counts from the pass. From the hand, a release takes about 0.05 s and up
  to three setpoints, 5.4 degrees a joint: the age of the controller data, a pass and
  one setpoint period of the Thor. If the headset's stream stalls at that moment, the
  release is not seen until the data count as stale, `teleop.max_tick_age_s`: up to
  0.2 s, in which the arm goes on toward the hand's last pose, about 10 cm at the tip
  at the hand speed limit. Right B does the same and is a STOP. Stopping the hand does
  not stop the arm at once: it stops where the hand is.
- **A squeeze starts at the arm.** The first joint target of an engagement is the arm's
  measured pose, and it stands until a ROBOT_STATE shows the Thor's drive at it: about
  0.15 s in which the arm does not move yet. A drive that does not start within 1 s ends
  the engagement with `the arm did not follow the squeeze within 1 s`. That holds with
  the gripper close to the table as anywhere else.
- **A second squeeze right after a release waits for the arm to be at rest.** A
  ROBOT_STATE made while the arm still moved gives a pose the arm has left; an
  engagement that started there would pull the arm back, up to the 5.7 degrees a joint
  of `limits.human.max_step_rad`, or be refused. A new squeeze counts once two things
  have been shown. First, a ROBOT_STATE with no human drive that the robot app made
  after it took the pass that ended the engagement before (`health.human.seq`). Second,
  for every joint, two readings in a row that both began after the Thor's last setpoint
  (`target_t`) and agree within 0.002 rad: 5 steps of the motors' 16-bit position word,
  not a measurement of this arm's reading noise. The wait is bounded: 0.5 s after the
  engagement ended the squeeze is taken on the newest ROBOT_STATE, and the Thor's own
  step limit is the backstop. A squeeze that comes sooner is kept and the note says
  `wait: the arm is still stopping`. In real time, the human program's `main()`
  against the rig's robot app on the simulated arm, which comes after its target
  with a time constant of 0.05 s (`test_human_realtime.py`): squeezed again 40, 60,
  100 and 200 ms after a release at full joint speed, 3 runs each, the second drive's
  first setpoint went out 0.36 to 0.41 s after the release, the arm went back at most
  0.0004 rad, and nothing was refused or ended. Most of that wait is the Thor's: after
  a drive it begins its idle joint reads 0.1 s after the last setpoint
  (`IDLE_AFTER_TARGET_S` of `robot_app/state.py`) and reads one joint every 0.01 s,
  so the second reading of the sixth joint begins 0.21 s after that setpoint. If the
  robot app goes on reporting the drive for more than 1 s, the note says
  `wait: the robot app still reports the last drive; if this stays, type stop`. If no
  ROBOT_STATE comes, the states are old and the note is `no fresh robot state`. Not
  run on the arm.
- **The tip stops at the walls of the human checker's tip box.** APAI's human checker
  holds a person's gripper tip in the box the human limit set names. On the rig
  (`limits.human.workspace_min: [0.10, -0.50, -0.01]` and `workspace_max: [0.55, 0.50,
  0.70]` of `configs/robot_app_rebot.yaml`) that is 10 to 55 cm in front of the base,
  50 cm to each side, and up to 70 cm above the table plane. A file that names no box
  leaves the workspace box of the arm profile, which the agent and a return to rest
  keep on the rig: 10 to 50 cm forward, 30 cm to each side, up to 55 cm. The robot app
  names the box in WELCOME (`human_tip_box`), and the human program keeps the tip of
  the pose it solves for 1 cm inside it: the tip stops at a wall and slides along it
  while the hand goes on, and comes away from it with the hand. From the ready pose
  the front stop is 9 cm away and the top stop 25 cm. The note and the report line say
  `at the workspace limit: front` (`back`, `left`, `right`, `top`, `bottom`; two walls
  as `front, top`). The point the human program reckons with is the checker's own. The
  room of the pose solved for is opened once, at the squeeze, as far as the tip is
  outside the 1 cm line then, and from then on only narrowed: what a solve leaves
  outward of its pose is not kept from pass to pass. A tip that is outside the box at
  a squeeze may come toward it and go no further out, on each axis by itself: the
  checker's own rule. A hand is believed up to `ik.max_ahead_m`, 50 cm, from the
  gripper tip, in all, beyond a wall as inside the box (`the hand is N cm from the
  gripper tip, over the 50 cm believed`). A robot app that names no box, as the one on
  the simulated arm, bounds nothing.
- **The tip stops 0.5 cm above the table.** The robot app names in WELCOME the human
  checker's table clearance as a height (`human_tip_floor`): the table plane plus
  `limits.human.table_clearance_m`, 0.0 on the rig, in the frame of the arm model as
  the box is. The human program takes the stop 0.5 cm above that height for the lowest
  wall: the tip stops there and slides along it, and the note says
  `at the workspace limit: table`. By test on the rig's human file, pass by pass
  against a stand-in for the Thor (`test_human_program.py`): from a stretched pose with
  the tip 2.7 cm above the table, with the hand pushed 10 cm down in 0.2 s and then
  15 cm to the left, the tip came down to 4.97 mm and slid more than 10 cm along the
  stop; no joint target had its tip under it. A tip under the stop at a squeeze rises
  to it (the floor under [A takeover, step by step](#a-takeover-step-by-step)). A
  robot app that names no such height leaves the bottom of its box: there a reach
  under the human checker's table clearance ends the drive with `TCP z=... below table
  clearance ...`.
- **The joints stop at the human joint box, and out of reach the arm follows as far
  as it can.** The robot app names in WELCOME the joint box its human checker holds a
  person's joint target in (`human_joint_box`). The joint target stops 0.01 rad inside
  it, the note says `at a joint limit: joint N` (N counts from 1, as in the checker's
  refusal), and the engagement goes on: the arm goes on along the box where a step
  along it brings the tip nearer the hand, and stands where none does. A pose of the
  hand that has no IK solution is out of reach. It does not end the engagement: the
  arm follows as far as it can, the tip's position first and the wrist's angle
  second. Each pass tries, in this order: the position asked for with the wrist's
  angle of the last joint target; half and a quarter of the way to that position; and
  the solver's nearest pose, one step along the edge of the reach that brings the tip
  nearer the hand's position and turns the wrist toward the hand's angle only where
  that costs the tip nothing. Such a step keeps the hand speed limits, the tip box,
  the stop above the table and the joint box, and the arm stands where nothing brings
  the tip nearer. From the fifth pass in a row without a solution (`ik.max_failures`)
  the note says `out of reach`; it goes on the first pass that solves. The hand is
  then followed again, with no new squeeze, and the wrist turns back to the hand's
  angle. By test on the rig's human file against a stand-in for the Thor
  (`test_human_program.py`): under a stand-in that names no tip box, a hand that went
  45 cm forward from a stretched pose and 20 cm to the left took the tip 31.3 cm
  forward and more than 10 cm to the left with the note `out of reach`, and after the
  hand came back the tip and the wrist were at the hand's pose within 2 mm and
  2 mrad; with the rig's boxes, a hand that went 40 cm forward past the
  bound of the second joint and 15 cm to the left took the tip 13.5 cm forward and
  13.9 cm to the left along that bound, with the note `at a joint limit: joint 2`. In
  both nothing ended the engagement and the stand-in refused nothing. A robot app that
  names no joint box leaves the checker's refusal (`joint N target ... is outside the
  human joint limits ...`), which ends the drive.
- **A slow or pausing Thor costs speed.** The joint target goes no further than two of
  the Thor's steps, 0.063 rad, ahead of where the Thor's own joint target can be: its
  last one in the newest ROBOT_STATE, moved on by what its loop can have done since.
  The pace of that loop is read off two ROBOT_STATEs, and a state made more than a
  period and a half after the Thor's last setpoint shows a loop that stands. A loop
  that runs slower than its 20 ms is followed at its pace; the note says
  `the robot app is behind: the arm follows slower`. A joint target that is held back
  for 1 s with no new joint target of the Thor ends the engagement:
  `the arm does not follow: no new setpoint of the robot app in 1 s`.
- **A stall of the whole robot app costs speed too.** While a person drives, the robot
  app sends a ROBOT_STATE behind every setpoint, on the drive's thread: 50 a second in
  place of 20, each 7.2 kB with the simulated arm, so 360 kB/s in place of 130 on the
  robot connection and on the Thor's own socket. When the next one is overdue, one and
  a half times the time from one to the next in this drive, 0.03 s, the human program
  gives the Thor no more for the time since: the joint target stands two of the
  Thor's steps, 0.063 rad, ahead of the Thor's reported one, 0.037 rad inside the
  Thor's step limit. In real time, on the rig's robot app with the simulated arm
  (`test_human_realtime.py`): pauses of 60 and 90 ms at full joint speed, of the drive
  thread alone and of the whole robot app, 6 runs each, were ridden out in 24 of 24;
  the farthest a joint target lay from the Thor's last setpoint was 0.0632 rad, and no
  setpoint was a larger step than the Thor's clamp. Pauses of 200 and 500 ms ended the
  drive on the Thor (`executor.large_delay_s`, 0.1 s: `setpoint N went out X s late`).
  With nothing paused the arm was on its way for 0.78 s over 30 cm to the side, and
  for 0.80 to 0.88 s with a pause. The drive never waits for the ROBOT_STATE behind a
  setpoint: while another thread has the robot app's lock, as during a STOP, none is
  sent. Not run on the arm.
- **The lead** is the human program's check that the arm follows
  (`ik.max_ahead_rad`, 0.09 rad on the rig). Before the Thor's drive has sent a
  setpoint it is measured from the arm: a squeeze whose first target lies further from
  the arm ends there. After that it is measured from the Thor's last joint target as
  the newest ROBOT_STATE gives it, less what the Thor's speed clamp can have moved that
  target since the state arrived. The joint target is held back inside it, so it ends
  an engagement only where the Thor's joint target moves away: a hold at a contact
  puts it back to the measured pose. A joint target that has run on through a pause no
  state has shown is refused by the Thor itself once it lies
  `limits.human.max_step_rad`, 0.1 rad, from the Thor's last one.
- **What still ends an engagement**, each with a hold and
  `release the grip and squeeze again: <why>`, is listed under
  [What still asks for a new squeeze](#what-still-asks-for-a-new-squeeze).
- **Against a blocked joint the motors are asked for about 11 degrees more** before the
  arm holds. The Thor's setpoints go on whether the arm follows or not; only APAI's
  contact watch reads that it does not, and ends the drive in a hold at the measured
  pose: after 7 setpoints, 0.14 s, with the joint target 0.186 rad ahead of a joint that
  stood, on the stand-in motors at the rig's 1.6 rad/s (`test_rig_configs.py`). The
  watch's stall window is 0.12 s (`stall_window_s` of the rig's APAI profile), in which
  a command at a person's full speed moves 0.19 rad. The watch's thresholds are those
  of that profile, whose own speed cap is 0.8 rad/s, and no move at 1.6 rad/s was run
  with them on the arm: there it may also trip where nothing blocks the arm (see
  [Known gaps](#known-gaps)). Not measured on the arm.

### What still asks for a new squeeze

With the rig's two files and both programs as they are. Each of these ends the
engagement in a hold, and the note says `release the grip and squeeze again: <why>`.
From tests and from reading the checkers; nothing was run on the arm.

| What happened | What is said | How likely in normal driving |
|---|---|---|
| The hand is more than 50 cm from the gripper tip (`ik.max_ahead_m`), or turned more than 115 degrees ahead of it | `the hand is N cm from the gripper tip, over the 50 cm believed`, `the hand is turned N deg ahead of the arm (115 are followed)` | Unlikely: a tracking jump, or a hand carried far past a wall, a joint limit or the end of the reach |
| The hand is faster than 2 m/s or 10 rad/s (`retarget.reject_linear_velocity`, `retarget.reject_angular_velocity`) | `the hand moved too fast for the rate limiter` | Unlikely: a jerk or a tracking glitch |
| An IK jump that halving the tip's step three times does not remove | `IK jump of X rad on joint index i, the limit is 0.1` (`ik.max_jump_rad`) | Rare: near a folded or a stretched-out arm. A pass at the hand speed limit asks a joint for 0.04 to 0.09 rad between the ready pose and the arm folded near its base, as computed on the arm model. A jump among the poses tried for a hand out of reach ends nothing |
| The gripper is turned over until its wrist camera would face down | `wrist camera would face down: camera-up axis z=... < -0.10` (APAI's human checker, read in its harness, not provoked in a test) | Possible: a wrist rolled or pitched more than about a quarter turn from camera-up |
| A joint of the arm between the base and the tip comes within 1 cm of the table plane | `link/joint N at z=... would hit the table` (APAI's human checker) | Possible with the wrist bent down or the gripper laid flat close to the table: the tip may go down to 0.5 cm, a joint only to 1 cm |
| The arm touches something, or APAI's contact watch takes fast motion for a contact | `contact ...`; a new squeeze moves the arm away | Not known at 1.6 rad/s: the watch's thresholds are those of the rig's APAI profile, which no move at this speed was run with on the arm (see [Known gaps](#known-gaps)) |
| The Thor's drive pauses for more than 0.1 s | `setpoint N went out X s late, over the limit of 0.1 s` (the Thor's rule) | Not known: how the Thor's loop paces on the motor bus is not measured |
| The Thor sends no setpoint for a second while it reports its drive | `the arm does not follow: no new setpoint of the robot app in 1 s` | Rare: the Thor is stuck |
| The Thor's joint target moves away from the human program's, as with the hold at a contact | `the solution is X rad ahead of the arm` | With a contact |
| Controller data lost or stale, the live view not fresh, the robot state old, the robot connection lost | As in the table below | Whenever one of them is lost |

What does not: a wall of the tip box, the stop above the table, the human joint box,
the end of the arm's reach, a second squeeze that comes before the arm is at rest, and
a stall of the robot app of less than 0.1 s. For each the arm stops at the wall or the
table and slides along it, follows as far as it can, waits or slows; the note says
why, and the arm follows the hand again without a new squeeze.

**Both programs are to be deployed together.** The human program reads the tip box,
the table clearance and the joint box for a person off the robot app's WELCOME, and
the Thor's pace off the ROBOT_STATE behind every setpoint. Under a robot app that
names none of the three it prints `the robot app names no joint limits for a person
...` and `... it names no table clearance ...` at its start, and the checker's own
refusals end a drive at the table clearance and at the human joint box; under one
that sends no ROBOT_STATE behind a setpoint a stall of 60 ms or more is ridden out
only by chance. `ik.max_jump_rad` and `ik.max_ahead_rad` of `configs/human_rebot.yaml`
are fitted to `limits.human.max_joint_speed` and `limits.human.max_step_rad` of
`configs/robot_app_rebot.yaml`: change the two files together.

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
| The headset's stream to the Teleop Session Server stalls while the server keeps its tick rate | Nobody at once: the tick age stays fresh. By Isaac Capture's notes the controller is reported not active about 1 s later; not run | The hand stands still where the stream left it, and the arm goes on to that pose and stops there: as far as the hand was ahead, up to `ik.max_ahead_m`. A grip release and right B are not seen until the stream is back; right B that went down meanwhile is a STOP then | human | Status line `headset no_controller` after about 1 s. This is why the abort of part F of the checklist is the Thor terminal |
| The live view is not live or too old, where the file says `human.require_live_view: true`, as the rig's files do: the viewer's page does not show its live picture in a visible browser tab | The Thor, before every setpoint | Hold | human | `human drive ended: the live view is not fresh (the viewer reports no live picture)`, or `(N s old, the limit is 0.5 s)` |
| The robot connection is lost | The robot app | Hold | human | Console: `ROBOT CONNECTION LOST`. Thor terminal: `[connection] robot connection lost: the arm holds; the person keeps the arm: squeeze again when the connection is back; rest can still be typed at the Thor terminal` |
| A target more than `limits.human.max_step_rad` from the last setpoint sent | The Thor | Hold | human | `human drive ended: the target is X rad from the arm on joint index i, the limit is Y rad` |
| The human checker refuses a setpoint: the wrist camera would face down, a link at the table, a contact spot. The human checker's tip box, its table clearance and the human joint box do not end a drive of the human program under a robot app that names them in WELCOME: the tip stops 1 cm inside the walls and 0.5 cm above the table clearance, and a joint 0.01 rad inside the box. A HUMAN_SETPOINT from elsewhere that leaves them is refused | The Thor | Hold | human | `human drive ended: <the checker's reason>`, such as `wrist camera would face down: ...` or `link/joint N at z=X would hit the table`; from elsewhere, or under a robot app that names less, also `joint N target X rad is outside the human joint limits ...`, `TCP z=X below table clearance ...` or `TCP [...] outside workspace ...` |
| An IK jump that halving the tip's step three times does not remove, a hand further from the gripper tip than is believed or turned too far ahead of it, a joint target the arm does not follow, a hand too fast for the rate limiter. A pose with no IK solution ends nothing: the arm follows as far as it can | The human program, which then sends a HUMAN_SETPOINT that is not engaged | Hold | human | `human drive ended: <note>`, with the note `IK jump of X rad on joint index i, the limit is 0.1`, `the hand is X cm from the gripper tip, over the 50 cm believed`, `the hand is turned X deg ahead of the arm (115 are followed)`, `the solution is X rad ahead of the arm`, `the arm did not follow the squeeze within 1 s`, `the arm does not follow: no new setpoint of the robot app in 1 s` or `the hand moved too fast for the rate limiter` |
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

Each of these settings ships in its safe position, with three exceptions. `human.trigger_jaws`
ships on, and acts only where `human.enabled` is true. `retarget.max_linear_velocity`
and `retarget.max_angular_velocity` in `configs/human_rebot.yaml`, the bounds of the
gripper tip's target, are the second. A person's table clearance and tip box in
`configs/robot_app_rebot.yaml` are the third. The rows of the last two say what they
are: the rig owner's values of 2026-11-18, as `limits.human.max_joint_speed: 1.6` is
([The rig files](#the-rig-files)).
`human.require_live_view` ships `true`, its safe position, in `configs/robot_app.yaml`
and in the rig's files, and `false` only in the files of the one-machine recipes; it
also acts only where `human.enabled` is true. The table says what the shipped
value means, where it is switched and what a change waits for.

| Setting and shipped value | What the shipped value means | What it waits for |
|---|---|---|
| `human.enabled: false` in `configs/robot_app.yaml` and `configs/robot_app_rebot.yaml` | A takeover is refused and no human target moves the arm. `true` only in the files of the one-machine recipes and of part F-A (`robot_app_mock.yaml`, `robot_app_demo2.yaml`, `robot_app_human_check.yaml`) | It is switched on in the rig file in the session, before step 15, once part F-A of the checklist has passed. Code that never saw a headset is not reachable on the arm before that |
| `human.require_live_view: true` in `configs/robot_app.yaml` and in the rig's files `configs/robot_app_rebot.yaml` and `configs/robot_app_human_check.yaml` | With `true` a takeover and a human drive need a fresh live view: the viewer's page with its live picture showing, in a visible browser tab (see [The live view](#the-live-view)). With `false` they need none, on any driver, the real one included. A robot app with `human.enabled: true` and `false` here says so at its start, at the Thor terminal and in its log: `robot app: human driving is not tied to a live view (human.require_live_view false)`. `false` only in the files of the one-machine recipes, `configs/robot_app_demo2.yaml` and `configs/robot_app_mock.yaml`, which run without a camera | Nothing: it stays `true` on the rig |
| `limits.human.table_clearance_m: 0.0`, `limits.human.workspace_min: [0.10, -0.50, -0.01]` and `limits.human.workspace_max: [0.55, 0.50, 0.70]` in `configs/robot_app_rebot.yaml`; no other robot app file names them | Not a safe position: the rig owner's values of 2026-11-18 for a person who sees the arm. A person's gripper tip goes down to 0.5 cm above the table plane and moves in a box of 10 to 55 cm forward, 50 cm to each side and up to 70 cm. A file without the keys holds a person to the higher of `rest.table_clearance_m` and the table clearance of APAI's arm profile, and to that profile's workspace box: on the rig 2 cm, and 10 to 50 cm forward, 30 cm to each side, up to 55 cm, as for the agent | The first session on the arm: whether the arm model's table plane is the table, and how far the arm droops under its target ([Known gaps](#known-gaps)). For a session with the agent's table and box, take the three keys out of the file |
| `human.view_age_limit_s: 0.5` | The oldest live view a person may drive on. A first value. Read only with `human.require_live_view: true` | Step F0: the highest measured view age plus 0.2 s, at most 1.0 s |
| `human.trigger_jaws: true` and `human.jaw_match_tol: 0.05` in every robot app file | Under a person the trigger drives the jaws once the opening it asks for has come within `human.jaw_match_tol`, a share of the jaw travel between 0 and 1, of the grip level. With `human.trigger_jaws: false` the trigger does nothing and no jaw command is sent under a person. Neither acts where `human.enabled` is false | The person's trigger has not moved the real jaws. Before step 15 the owner decides whether the first headset session on the arm runs with it: `human.trigger_jaws: false` in `configs/robot_app_rebot.yaml` is the session without it |
| `operator.typed_takeover: false` in both service files | `takeover` and `handback` typed at the console are refused by the robot link service | Step F6: if right A or right B does not arrive, or a controller that dies during a takeover. Switching it on needs a restart of the robot link service |
| `retarget.max_linear_velocity: 0.5` m/s and `retarget.max_angular_velocity: 2.0` rad/s in `configs/human_rebot.yaml` | Not a safe position: the rig owner's values of 2026-11-18. They bound the gripper tip's target at 1 cm and 0.04 rad a pass. With `limits.human.max_joint_speed: 1.6` in `configs/robot_app_rebot.yaml`, the owner's value of the same day, the joints bind first in most poses: from the ready pose the tip moves at 27 to 36 cm a second ([How fast the arm follows the hand](#how-fast-the-arm-follows-the-hand)) | For a slower session lower `limits.human.max_joint_speed` in `configs/robot_app_rebot.yaml`: the human program follows the limit the robot app names in WELCOME. Lower the two hand speed limits as well for a slower tip where the arm is stretched out |
| `ik.max_ahead_m: 0.5` in both human files; `ik.max_ahead_rad: 0.08` in `configs/human.yaml` and `0.09` in `configs/human_rebot.yaml` | A hand more than `ik.max_ahead_m` ahead of the arm's gripper tip, or that far beyond a wall of the human checker's tip box, ends the engagement; up to it the arm catches up with the hand. A joint target more than `ik.max_ahead_rad` ahead of where the arm can be ends it too; each end costs a release and a new squeeze. That lead is measured from the measured pose until the person's drive has sent a setpoint, so an arm that holds short of an earlier target can still be driven, and after that from the Thor's last joint target, less what the Thor's speed clamp can have moved it since the ROBOT_STATE. In the rig file the value lies 0.01 rad below `limits.human.max_step_rad` of `configs/robot_app_rebot.yaml`, so a squeeze whose first joint target the Thor would refuse ends in the human program | `ik.max_ahead_m` on the rig: nothing; lower it for a shorter way after the hand stops. `ik.max_ahead_rad` stays below `limits.human.max_step_rad` and above 0.063 rad: a joint target that follows the hand is held back to two of the Thor's steps, 0.063 rad at 1.6 rad/s, ahead of the Thor's own, and a squeeze at rest begins 0.033 rad from the arm |
| `ik.max_jump_rad: 0.05` in `configs/human.yaml` and `0.10` in `configs/human_rebot.yaml` | An IK solution further than that, on any joint, from the last joint target is a jump: the tip's step is halved up to three times, and a jump that stays ends the engagement. On the rig one pass at the hand speed limit, 1 cm of the tip, asks a joint for 0.04 to 0.09 rad between the ready pose and the arm folded near its base, as computed on the arm model, so 0.10 takes such a step whole | Not below 0.056 rad on the rig: twice what the joint target goes in a pass at its speed limit, 0.028 rad, so that a halved step still moves the fastest joint at its limit |
| `limits.human.max_step_rad: 0.10` in `configs/robot_app_rebot.yaml`, and the same value in every other robot app file | A human target farther than that, on any joint, from the arm at a squeeze, or from the last setpoint sent after it, is refused, not clipped, and the arm holds. It bounds the jump at a squeeze: 5.7 degrees on every joint, which is 14.6 cm at the gripper tip in the ready pose and up to 18 cm with the arm reaching out, taken in 0.08 s. It does not bound the travel: a target that stays inside it is followed at up to 0.98 of `limits.human.max_joint_speed` | A first value. Not below `ik.max_ahead_rad` of the human program's file, which on the rig needs more than the 0.063 rad of two of the Thor's steps at 1.6 rad/s |
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
6. With the gripper tip under the human checker's table clearance a takeover is taken,
   and under that clearance only a step that rises is followed. The alternative is to
   refuse that takeover: an arm whose tip the model puts under that clearance then
   takes no command that moves it.

The last four can be reached only where `human.enabled` is true, so the switch for human
driving is the switch for them too.

### What was cut from the human path

Each of these is in the design or in the plan of this phase, and is not built.

| What | What the build does instead |
|---|---|
| A step size and an effort of their own for the jaw commands of the trigger | The trigger's jaw commands use `gripper.open_step`, `gripper.reach_tol` and one effort, 0.5, for a close as for an opening |
| A silent agent after a hold on an expired permit counting as failed | The arm holds until someone acts; there is no prompt when an agent goes silent |
| The question "completed or abandoned?" after a takeover, stored in the task record | The task record does not say how a takeover ended |
| The console printing the takeover's steps one at a time, and whether a return to rest would be accepted now | One takeover line, the human program's note in the status line, the bells |
| A remedy table on the console for a refused `rest` | The refusal's own text; the four refusals are named under [A takeover that cannot be ended with right A](#a-takeover-that-cannot-be-ended-with-right-a) |
| A script for the synthetic controller from a file | The built-in scripts `lift`, `grasp` and `idle`; tests hand rows to the constructor |
| The takeover's facts on the viewer's page; buttons in the browser | The console shows them. The viewer's page shows the owner and the motion in its line of readings while its arm is live |
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
  pass, and the teleop feed's line `entered a remote session at <endpoint>`, are written
  after that pass's link messages have gone out. Isaac Capture's own INFO lines of an
  entry (`Connecting to the Teleop Session Server at ...`, `... is ready`) are written
  inside the entry: with a terminal that takes no log line, right B that is down as a
  remote session is entered is a STOP only once the terminal runs again. The arm holds
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
| 1 | Nothing to attach to (`cannot attach to the robot link service or the robot app: ...`), a second human program, the robot link service gone (`the robot link service is gone (...)`), an arm model whose joint count is not the one WELCOME names, a synthetic controller against a driver that is not simulated, the same two for a robot app that was restarted while the program ran, a failure the program says as `human program: failed: ...` (an error of Isaac Capture at an attempt to enter a remote session after the first is one: `human program: failed: RuntimeError("the attempt to enter a remote session at <endpoint> failed: ...")`), or the attempts to reach the Teleop Session Server ran out (`no remote session at <endpoint>: N attempts in a row failed, ...`) |
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
  path. Both programs take a path that begins with `~` from the home folder themselves,
  whether or not a shell expanded it. The socket's folder
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
  The rig's files require the live view, so on the rig nobody takes over or drives
  with the robot connection down. With `human.require_live_view: false` a person can.
  The console then shows nothing of it: the stop is the Thor terminal's, and right B.
- **The console** shows `human program at the robot app` in its status line. `clients`
  there lists only what is attached to the robot link service.
- **Start the robot app first, then the human program.** When the robot app ends, its
  socket closes and the human program ends with exit code 1; start it again after the
  robot app. The human program cannot tell which of the two it is attached to, so it
  names both: `cannot attach to the robot link service or the robot app: ...`. One
  text still names the robot link service where the socket is the robot app's own: `the
  robot link service is gone (...)`. A robot app whose `--human-socket` names a socket
  that a running robot app serves ends with `--human-socket PATH cannot be served: a
  robot link service already runs on PATH`.
- **One human program is followed.** If one runs at each end, the HUMAN_SETPOINTs of
  the second are dropped while the first has sent one within `human.stage_keep_s`
  (4 s), and the console prints `[refused] a second human program sends
  HUMAN_SETPOINT link messages at the robot app's socket: they are dropped while the
  one over the robot connection sends; one human program at a time`, or the same with
  the two places exchanged, once for as long as the second goes on sending. It prints
  the line again when the second sends after it was silent for `human.stage_keep_s`, or
  after the followed one changed. A STOP and an OWNER_REQUEST of the second are not
  dropped: they keep every check.
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

The robot link service, the console, the experiment's viewer and the agent program run
on the workstation as for demo 1. The Thor terminal shows `robot app: the human program's
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
| `retarget` | `controller_pose` (`grip`; or `aim`), the controller pose that is followed; `position_scale` (1.0), the target's motion per motion of the hand; `squeeze_threshold` (0.5, below 1): the grip counts as squeezed above it; `max_linear_velocity` (0.10 m/s) and `max_angular_velocity` (0.4 rad/s), the hand speed limits: the bounds of the rate limiter on the tip's target, and of the step toward that target that each pass solves for from the arm's last joint target; `reject_linear_velocity` (2.0 m/s) and `reject_angular_velocity` (10.0 rad/s; `null`: none): a hand faster than that ends the engagement; `controller_to_base`, the rotation, by rows, from the stage space, the coordinate system the controller poses are given in, into the robot's base axes (x forward, y left, z up); `stage_yaw_deg` (0.0) |
| `ik` | `apai_arm` (`mock`), the APAI arm profile whose model the IK solves on; `max_iter` (50); `tol` (1e-4); `max_jump_rad` (0.05): a solution further than that from the last joint target is a jump, the tip's step is then halved up to three times, and a jump that stays ends the engagement; `max_ahead_rad` (0.08), the farthest the joint target may lie ahead of where the arm can be; `max_ahead_m` (0.5), the farthest the hand may be from the arm's gripper tip, in all: up to it the arm catches up; `max_failures` (5), the passes in a row without a solution from which the note says `out of reach`: a pose with no solution does not end the engagement, and the arm follows as far as it can, the tip's position first and the wrist's angle second |
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
  to 3.0 rad/s. No joint of the joint target the program sends moves faster than 0.9 of
  the robot app's speed clamp, which is 0.98 of the human joint speed limit its WELCOME
  names (`human_max_joint_speed`): where a hand speed limit asks a joint for more, the
  joint target takes that share of its way to the solution on every joint, and the tip
  comes behind. A step of the tip that asks a joint for more than `ik.max_jump_rad` in
  one pass is halved, up to three times, before it counts as an `IK jump`.
- **From rest, lift the gripper first.** The rest pose lies on joint stops. From there
  forward, back and down have no IK solution, and sideways fails after a few millimetres:
  the arm follows as far as it can, the note says `out of reach`, and it follows the
  hand once the hand goes up. A lift solves on every pass. About 39 cm above rest the
  arm's reach ends, and the note says `out of reach` there the same way.
- **`configs/human_rebot.yaml`** is the file for the rig, where the program runs on the
  Thor. It differs in exactly eight keys: `socket_path: "~/.agentic_remote/human.sock"`,
  the robot app's own socket; `teleop.endpoint: "192.0.2.1:47600"`, a placeholder for
  the workstation's address on the wired link; `ik.apai_arm: rebot_rs`;
  `retarget.max_linear_velocity: 0.5`; `retarget.max_angular_velocity: 2.0`;
  `teleop.attempts: 8640`; `ik.max_jump_rad: 0.1`; and `ik.max_ahead_rad: 0.09`, which
  lies below `limits.human.max_step_rad: 0.10` of the rig's robot file. Both files have
  `ik.max_ahead_m: 0.5`.
  The Thor clamps a person's joint motion at 0.98 of `limits.human.max_joint_speed`,
  1.6 rad/s in the rig file by the rig owner's decision of 2026-11-18, and the human
  program asks for 0.9 of that; the two hand speed limits, the owner's values of the
  same day, bind where the joints could move the tip faster
  ([How fast the arm follows the hand](#how-fast-the-arm-follows-the-hand)).
  `configs/human.yaml` is the file of the one-machine recipes.

### What it prints

One line each on its standard output. The steps F2 to F7 of the checklist read these lines.

```text
human program: <teleop | synthetic controller "NAME">; driver <d>; arm model <apai_arm>, <n> joints
human program: the arm follows the hand at up to <j> rad/s a joint (<the robot app's limit>),
    its tip at up to <v> m/s and <w> rad/s
human program: the gripper tip stops at the robot app's workspace limits:
    x <..> to <..>, y <..> to <..>, z <..> to <..> m from the arm's base;
    <its lowest is 0.5 cm above the table clearance | it names no table clearance: ...>
human program: the joints stop at the robot app's joint limits for a person,
    and the arm follows as far as it can while the hand is out of reach
human program: Teleop Session Server <stage>[: <text>]
human program: right controller seen | right controller lost
human program: <A | B | grip | trigger> <down | up>
human program: stage <s>; tick age <lowest>/<middle>/<highest> ms;
    state age <lowest>/<middle>/<highest> ms; round trip <r> ms; owner <o>;
    engaged <yes|no>; hand x <..> y <..> z <..> m;
    tip target x <..> y <..> z <..> m, roll <..> pitch <..> yaw <..> deg;
    IK <middle>/<highest> ms; ended: no solution <n>, jump <n>, solution ahead <n>,
    hand ahead <n>, limiter <n>
    [; at the workspace limit: <wall>[, <wall>]]
    [; out of reach | ; at a joint limit: joint <N>]
    [; the robot app is behind: the arm follows slower]
    [; hand <N> cm ahead: the arm is catching up]
    [; jaws follow the trigger <yes|no>, grip level <g>, trigger asks <a>[: <why>]]
```

- The first once, after WELCOME; a robot app that declares no driver shows as `not
  declared (treated as real)`. The line after it says how fast the arm follows the hand
  under that robot app, and is said again for a new session of the robot app. With the
  rig's files it reads `human program: the arm follows the hand at up to 1.41 rad/s a
  joint (the robot app's limit for a person is 1.6 rad/s), its tip at up to 0.5 m/s and
  2 rad/s`; under a robot app whose WELCOME names no limit, `... at up to 0.09 rad/s a
  joint (the robot app names no speed limit for a person), ...`. The third line, said
  with it, is where the gripper tip is held: the corners of the box the robot app names
  in WELCOME, 1 cm inside its walls, with the lowest 0.5 cm above the height it names
  for its table clearance. With the rig's robot app it reads `human program: the gripper
  tip stops at the robot app's workspace limits: x 0.110 to 0.540, y -0.490 to 0.490, z
  0.005 to 0.690 m from the arm's base; its lowest is 0.5 cm above the table
  clearance`. Under a robot app that names the box and no such height it ends `z 0.000
  to 0.690 m from the arm's base; it names no table clearance: a tip lowered under it
  ends the drive`, and under one that names neither it reads `human program: the robot
  app names no workspace box for the gripper tip: a target outside its limits ends the
  drive`. The fourth line, said with them, is `human program: the joints stop at the
  robot app's joint limits for a person, and the arm follows as far as it can while the
  hand is out of reach`, or, under a robot app that names no
  joint box, `human program: the robot app names no joint limits for a person: a joint
  target outside them ends the drive`. The next at each change of the teleop feed's stage; the
  script feed prints none. The next at each change. The next at each edge: the grip at
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
  axes. `ended:` counts the engagements the human program ended, by cause. The part
  `; hand N cm ahead: the arm is catching up` is there while the arm follows a hand
  that is 5 cm or more ahead of the gripper tip of the last joint target; it names no
  turn of the hand that the tool has still to make. Ahead of it,
  `; at the workspace limit: <wall>` names each wall of the human checker's tip box that
  the tip stands at with the hand beyond it (`table` for the lowest, where it is the
  stop above the table), `; out of reach` or `; at a joint limit: joint <N>` says
  what holds the arm short of the hand, and
  `; the robot app is behind: the arm follows slower` is there while the joint target
  is held back for a Thor whose loop pauses or runs slow. The
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
| `release right B, then hold it for 1 s with the grip released to take over` | Right B is down with no press that can ask: it went down while controller data were missing or the controller's pose was not valid, or the grip was squeezed on a pass of the press, or the STOP of the press went out while the program had no ROBOT_STATE of the robot app's current session (the robot app was started again while right B stayed down, or its first ROBOT_STATE had not arrived). Holding on asks for nothing |
| `takeover refused: <reason>`, `handback refused: <reason>` | The program's request did not end `completed`; the word after the request is the RESULT's status. The note goes once ROBOT_STATE shows another control owner, STOP or motion than at that RESULT, and at the next press that counts |
| `release the grip, then squeeze it to move` | The person owns the arm and the grip is squeezed without an engagement, or the Thor's needs-release flag is set |
| `squeeze the grip to move` | The person owns the arm and the grip is released |
| `hand N cm ahead: the arm is catching up` | The arm follows the hand, and the hand is 5 cm or more ahead of the gripper tip of the last joint target. N is 5, then 10, 20, 30 and 40, and changes only once the hand is 2 cm outside that step, so the console prints a line for a few of them and not for every pass. The arm goes on to where the hand is, by the straight way and not along the hand's: go round things no faster than the arm follows. Release the grip to stop it |
| `at the workspace limit: front` (`back`, `left`, `right`, `top`, `bottom`; two walls as `front, top`) | The arm follows the hand, the gripper tip stands 1 cm inside that wall of the human checker's tip box and the hand is 1 cm or more beyond it: the tip does not follow there. It slides along the wall with the hand and comes away from it once the hand is back inside. The walls are named by the robot's base axes, as the person behind the arm sees them. With `hand N cm ahead: ...` behind it while the hand is also ahead along the wall |
| `at the workspace limit: table` | The same for the lowest wall under a robot app that names its table clearance for a person: the tip stands 0.5 cm above it, 0.5 cm above the table plane on the rig, with the hand below, and slides along the table with the hand. A tip that was under that stop at the squeeze rises to it |
| `out of reach` | The arm follows the hand and the hand's pose has had no IK solution for five passes in a row (`ik.max_failures`). The arm follows as far as it can, the tip's position first and the wrist's angle second: the tip goes on toward the hand along the edge of the reach while the wrist may be left turned off the hand's, and the arm stands where nothing brings the tip nearer. The note goes on the first pass that solves: bring the hand back, and the arm follows it again, the wrist too, with no new squeeze. Behind a wall's note, ahead of the others |
| `at a joint limit: joint N` | The arm follows the hand and joint N, counted from 1, stands 0.01 rad inside the joint box the robot app names for a person, with the hand asking for more. The arm goes on along that bound where that brings the tip nearer the hand, and stands where it does not; it follows the hand again once a step toward the hand takes that joint back inside |
| `the robot app is behind: the arm follows slower` | The arm follows the hand, and the joint target was held back on five passes in a row, or on more than half of the last ones, for a Thor whose loop pauses or runs slow. It goes once the joint target was not held back for up to 0.4 s. With `hand N cm ahead: ...` behind it; when a wall is named too, the wall and this are said and how far the hand is ahead is not |
| `wait: the arm is still stopping` | The person owns the arm and the grip was squeezed again before the ROBOT_STATEs showed the drive of the engagement before as ended and the arm at rest: two readings of every joint that agree, both begun after the Thor's last setpoint. About a third of a second after a release at speed on the simulated arm, at most half a second. The squeeze counts from then on; keep it squeezed |
| `wait: the robot app still reports the last drive; if this stays, type stop` | The person owns the arm and the robot app has gone on reporting the drive of an ended engagement for more than a second, with the grip released or squeezed: no squeeze counts until a ROBOT_STATE shows that drive as ended. A drive ends within one setpoint period of the released grip, so this means the Thor's drive is stuck |
| `release the grip and squeeze again: <why>` | The engagement ended other than by a released grip. `<why>` is the human program's reason; or, when the robot app asks for a released grip, the `busy: ...` work that refused the squeeze, else why the robot app's last drive ended |
| `at rest: hold right A for 1 s to give the arm back` | A takeover is open and the arm holds at rest; while the person owns the arm, only once the arm was moved in this takeover |
| `release right A, then hold it for 1 s with the grip released to give the arm back` | In place of the note above while right A is down with no press that can ask (as for right B, or one such pass ended its hold). Holding on asks for nothing. While ROBOT_STATE shows the jaws as `loaded` the jaws' note comes first |
| `at rest: jaws not proven empty: at the console type stop, open, yes; then hold right A for 1 s` | In place of the two notes above while ROBOT_STATE shows the jaws as `loaded`: the robot app refuses that hand-back. Jaws shown as `unknown` keep the notes above |
| `pull the trigger to the grip level to move the jaws`, `ease the trigger back to the grip level to move the jaws` | The person owns the arm, the arm follows the hand, and ROBOT_STATE says that the trigger has not reached the grip level (`health.human.jaws`). `ease` while the trigger asks for a narrower opening than the grip level, `pull` otherwise. During an engagement there is no other note but those about the arm (`at the workspace limit: ...`, `out of reach` or `at a joint limit: ...`, `the robot app is behind: ...`, `hand N cm ahead: ...`, in that order, as many as fit in a note), which come first; with the grip released the notes above stand and the trigger gets none |
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
- **The rig requires the live view** (`human.require_live_view: true` in
  `configs/robot_app_rebot.yaml`, and in `configs/robot_app_human_check.yaml` of part
  F-A). The live view is the viewer's page with its live picture showing, in a visible
  browser tab; `run_workstation.sh` starts the viewer in its pane `viewer`. A takeover
  and a human drive need a fresh VIEW_STATUS of that viewer, and every other check of a
  takeover and of each human target stays. Without it the takeover is refused: `the
  live view is not fresh: ...; show the viewer's live picture in a visible browser
  tab` (see [The live view](#the-live-view)). A second person stays at the Thor
  terminal's stop.
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
- **The speeds: the agent and the return to rest run at 0.2 rad/s, and a person's drive
  is capped at 1.6 rad/s.** The 0.2 is the rig owner's decision of 2026-11-16, the 1.6
  that of 2026-11-18. Four keys hold the 0.2: on the Thor
  `limits.agent.max_joint_speed`, `limits.recovery.max_joint_speed` and
  `rest.recovery_speed` of `configs/robot_app_rebot.yaml`, and on the workstation
  `thor.agent_max_joint_speed` of `configs/agent_rebot.yaml`.
  `limits.human.max_joint_speed` of the robot app's file holds the 1.6.
- **A person's limits are the rig owner's values of 2026-11-18**, for a person who sees
  the arm; the agent's limits, the typed `rest` and the hand-back keep theirs. In
  `limits.human` of `configs/robot_app_rebot.yaml`: `max_joint_speed: 1.6`;
  `max_step_rad: 0.10`, the farthest a human target may lie from the arm at a squeeze
  and from the last setpoint after it; `table_clearance_m: 0.0`, the human checker's
  own table clearance, the table plane; and `workspace_min: [0.10, -0.50, -0.01]` with
  `workspace_max: [0.55, 0.50, 0.70]`, the human checker's own tip box. In
  `configs/human_rebot.yaml`: `retarget.max_linear_velocity: 0.5` and
  `retarget.max_angular_velocity: 2.0`, the hand speed limits, with `ik.max_jump_rad:
  0.1` and `ik.max_ahead_rad: 0.09` fitted to them and to the step limit. The human
  program stops the gripper tip 0.5 cm above the table plane, and where the hand asks
  for a pose out of reach the arm follows as far as it can
  ([A takeover, step by step](#a-takeover-step-by-step),
  [How fast the arm follows the hand](#how-fast-the-arm-follows-the-hand)).
- **The three keys of a person's table and tip box are optional and the human limit
  set's alone.** A file without `limits.human.table_clearance_m` holds a person to the
  higher of `rest.table_clearance_m` and the profile's table clearance; one without
  `workspace_min` and `workspace_max`, which are given together, to the profile's
  workspace box. Under `limits.agent` and `limits.recovery` the loader refuses all
  three (`limits.agent.table_clearance_m must not be set: only the human limit set has
  a table clearance and a tip box of its own`). `rest.table_clearance_m` stays 0.02:
  a typed `rest` starts only with the gripper tip above 2 cm.
- **Change the speeds on both machines together.** The agent plans at the lowest of
  three speeds: the one of the experiment launcher's speed profile, its own
  `thor.agent_max_joint_speed`, and the Thor's own agent speed limit, which the robot
  app names in WELCOME. So a speed raised on one machine only changes nothing, and a
  setpoint faster than the Thor's limit would be refused there. The agreed speed also
  sets the start tolerance a MOVE quotes. For another speed, change the four keys of
  the 0.2 to one value, on both machines, before the programs are started.
- **Where the agent's speed comes from.** The experiment's launcher runs at the speed of
  its `--speed-profile reference`: `overrides.safety.max_joint_vel` of the experiment's
  `configs/rebot_workstation.yaml`. Once the robot app's WELCOME is there the agent
  program says what it plans at, as in `joint speed: the agent plans at 0.2 rad/s, the
  lowest of 0.4 (--speed-profile reference) and 0.2 (thor.agent_max_joint_speed); the
  Thor's own agent speed limit is 0.2 (limits.agent.max_joint_speed, from the robot app's
  WELCOME)`. On the Thor the
  robot app reads the same file from the Thor's copy of the experiment folder and takes
  its speed as the cap of the agent's and of the recovery limit set: a limit set above
  that cap is refused at the start (`limits.agent.max_joint_speed X rad/s is above the
  APAI profile's speed cap of Y rad/s (safety.max_joint_vel)`).
- **The cap a human limit set is checked against is the robot app's own, 2.0 rad/s**
  (`HUMAN_SPEED_CAP` of `robot_app/apai_backend.py`), not the launcher's speed and not
  the cap of APAI's `rebot_rs` profile. That holds on the rig's profile, the one the
  experiment's launcher runs; a limit set above it is refused at the start
  (`limits.human.max_joint_speed X rad/s is above the robot app's speed cap for a person
  of 2 rad/s (apai_backend.HUMAN_SPEED_CAP)`). On a profile that no launcher runs, as
  APAI's stock `rebot_rs` (0.8 rad/s) and `mock` (1.2 rad/s), a person is bound by the
  profile's own cap, so the rig file is refused there: `limits.human.max_joint_speed 1.6
  rad/s is above the APAI profile's speed cap of 0.8 rad/s (safety.max_joint_vel)`. A
  person's motion is clamped at 0.98 of `limits.human.max_joint_speed`.
- **None of these speeds has run on the arm.** What ran there ran at 0.1 rad/s (see the
  top of this README).
- **Robot state is read as fast as the experiment's program reads it**:
  `state.idle_read_period_s: 0.01`, one joint every 0.01 s while the arm holds, the
  period of the experiment's idle joint sampler. The robot app measures after its
  connect how long one joint read takes on the motor bus and writes it to its log once
  (see [The real driver](#the-real-driver)).
- **The timed return is off** (`rest.auto_return: false`, `task.timed_return: false`): the
  arm holds when the assistance wait runs out.
- **The agent's and recovery's joint boxes are the limits of APAI's arm model**, not a
  measurement of this arm, with one change: the two joints that lie on their stops at rest
  may read down to -0.05 rad.
- **`camera_calibration_id`** in the robot app's file names the calibration in use on the rig.
  The agent program builds the same name from its copies of the calibration and prints it
  with `--check`. The robot app's `--calibration` repeats it. After a new calibration all
  three follow (see [The camera calibration and its name](#6-the-camera-calibration-and-its-name)).
- **`agent_rebot.yaml` needs three variables**: `AGENTIC_REMOTE_EXPERIMENT_DIR`,
  `AGENTIC_REMOTE_CAMERA_PR` and `AGENTIC_REMOTE_ENV_DIR`. Its paths name them, and the
  program refuses a path whose variable is not set or whose value is not an absolute path.
- **`link_service_rebot.yaml` is refused as shipped** (exit code 2, `allowed_peers must
  be a non-empty list`): `allowed_peers` is empty. Write the Thor's address into it, in
  the checkout the service is started from. The file listens on every address of the
  workstation, so that list alone decides who may be the robot. The edit belongs to the
  rig and is not committed (see
  [The edits each machine needs](#5-the-edits-each-machine-needs-which-are-not-committed)).
  An entry that is not an IP address is refused too.

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
through) is no part of that build. `--with-model-client` installs it with npm into
`<folder>/codex`, and `experiment.codex_bin` of `configs/agent_rebot.yaml` names it there.
The workstation needs the client for its own processor: the one inside the experiment
folder is the Thor's and does not run here. For an environment that is built already,
and for the login, see
[The model client on the workstation](#4-the-model-client-on-the-workstation). The
script never reads or writes that login.

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

`rest`, `release`, `on`, a shutdown, the typed `open` and `close` of the jaws, a reset and
every ownership request come from the console as link messages or are typed at the Thor
terminal. The Thor terminal's banner is `Thor terminal: type stop, rest, release, on,
reset, open, close, quit or help`. It prints every operator event of the robot app as
`[kind] text` and the end of each typed command as `<command>: <status>[: <reason>]`.

- **`rest` ends as the rest of the experiment's program does: at rest, jaws closed,
  motors off.** After the typed `path is clear` the arm returns to rest. With `--arm
  apai` it drives the route of APAI's rest tool: to the ready pose of the arm profile
  (`home_q`; in the rig's profile it lies within 0.06 rad of the rest pose) in one
  straight joint move, the leg the lines call `up to the ready pose`, unless the arm is
  within 0.08 rad of it; then the fold onto the rest pose in four measured steps. A
  ready pose outside `rest.region_*` is not driven to: the arm then folds from where it
  is. With `--arm sim` the return is one straight move. Then rest is measured, the jaws
  are closed with the driver's close, and the motors are switched off. The Thor terminal
  prints, with `--arm apai --driver real` (the times and the travel are examples):
  `[rest] return to rest: up to the ready pose: the arm moves for 9.4 s (largest joint
  travel 0.45 rad)`; four lines `[rest] return to rest: fold step N of 4: the arm moves
  for 2.5 s (largest joint travel 0.01 rad)`; `[rest] return to rest: the arm is at
  rest, motors on`; `[jaws] at rest: closing the jaws with the driver's close`; `[jaws]
  at rest: the jaws are closed`, or `[jaws] at rest: the jaws are left as they were:
  <why>`; `[release] torque released at rest: the motors are off`, or `[release] torque
  release sent at rest: the driver does not confirm that the motors are off`; and `rest:
  completed`. With `--arm sim` the move's line names no leg. A leg that does not arrive
  ends the return in a hold with the motors on: `return to rest failed: up to the ready
  pose did not track: the arm is 0.200 rad from its goal (tolerance 0.08 rad); the arm
  holds`. If the motors stay on at rest, the lines are `[refused] rest: at rest with the
  motors on; release refused: <why>` and `rest: failed: at rest with the motors on;
  release refused: <why>`. A jaw close that fails is told and the motors still go off:
  at rest the jaws hold nothing up. **While a takeover is open the motors stay on at
  rest**, for the hand-back: `[rest] return to rest: a takeover is open: the motors stay
  on at rest for the hand-back`. The Thor has no obstacle map: the straight move to the
  ready pose runs on the person's `path is clear`, where the agent's own rest tool
  plans that move with its map.
- **`release` moves nothing.** At measured rest with the jaws proven empty it closes the
  jaws and switches the motors off, with the `[jaws]` and `[release]` lines above and
  `release: completed`. Away from rest nothing is sent: `[refused] release refused:
  <why>; the motors stay on` and `release: refused: <why>`.
- **`on` switches the motors on again**, after a `rest` or a `release`, without a new
  start of the robot app and with the same session id. The Thor terminal asks `Connect
  the powered arm again? Stay beside the e-stop. The arm must lie at rest. Type "yes" to
  switch the motors on.` The driver then connects as at the start: it refuses an arm it
  does not measure at rest, and the jaws are referenced again. The lines are `[rest] the
  motors are on again at rest: motion rest_hold` and `on: completed`. The Thor terminal
  asks only while the motors are off at rest: with the motors on, and after any answer
  but `yes`, the line is `on: refused: confirm that the motors may be switched on`. The
  console asks first in every state, and its `on` with the motors on is refused with
  `the motors are on`. A connect that fails gives `[refused] the motors were not
  switched on again: <the driver's error>`, and the motors stay off. While
  the motors are off, a move, a jaw command, a new task and a takeover are refused with
  `the motors are off; switch them on again: type on at the Thor terminal or run the
  console command on`. The agent program asks for the same after its own question (step
  2 of [A task from its start to its end](#a-task-from-its-start-to-its-end)).
- **`reset` clears a fault-stop**: the arm holds and nothing is started. It is a word at
  the Thor terminal too: `reset: completed`, or `reset: refused: <why>`.
- **`open` and `close` move the jaws on the operator's word**, at the console and at the
  Thor terminal, and only while nobody owns the arm: type `stop` first. `open` asks for
  `yes` and is the driver's own opening, in one go. `close` is the close of APAI's
  `close_gripper` skill; it asks for `yes` only while the jaws are not proven empty.
  See [What the jaws do](#what-the-jaws-do).
- **`quit` is a `rest` that then ends the robot app**: it asks for `path is clear` only
  when the arm has to move, and prints `quit: completed; the robot app closes`, or the
  refusal and `; the robot app keeps running`.
- **The question about the jaws.** A `rest`, a `release` or a `quit` that the robot app
  refused only because the jaws are not proven empty is followed by `rest was refused
  because the jaws are not proven empty. Look at the jaws. If they hold something: stop,
  open, yes. If they are empty: type "jaws are empty" and rest is asked again on your
  word.` The question stands for the next typed line only. On `jaws are empty` the
  command is asked again, with its own question, and the robot app takes the operator's
  word in place of its measurement: `[jaws] the operator typed that the jaws are empty:
  taken in place of the measurement for a return to rest`. The word stands until the
  jaws are next sent a command or the arm gets a new control owner. The console asks
  the same after `rest`, `release`, `shutdown` and `handback`.
- **SIGTERM never moves the arm.** With the arm at measured rest and the jaws proven
  empty the robot app closes the jaws, switches the motors off and prints `robot app:
  SIGTERM; closing; the arm lies at measured rest: the motors are off`. With the motors
  already off: `robot app: SIGTERM; closing; the motors are off at rest`. Anywhere else
  nothing is sent: `robot app: SIGTERM; closing with the motors left on` and `robot app:
  the motors stay on because <why>`. After a torque release that failed: `robot app:
  SIGTERM; closing after a torque release that failed: the motors may be on or off`.
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
- **After a torque release that failed, only a new start of the robot app, or a release
  that works, changes anything.** The
  motion state is `release_failed`: the motors may be on or off (see
  [the torque rule](#the-rule-torque-is-never-cut-away-from-rest)). `rest`, `open`, `on`,
  a new task, a takeover and every command that would move the arm or the jaws are refused
  with `a torque release failed and the motors may be on or off; end the robot app with
  SIGTERM and start it again with the arm at rest`. `reset` clears the fault-stop and
  nothing else. On the simulated arm and APAI's mock driver `release` is then checked
  and tried again; with the arm's real driver it is refused. `quit` and `shutdown` start
  no return to rest and ask
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

- **Every jaw command is one routine of APAI's driver**, and when it has ended is the
  driver's verdict. A GRIPPER names the routine in its `mode`. `plain` is the driver's
  jaw command: an opening goes out as one guarded ramp and blocks until the driver takes
  it as done (on the rig a full open ends at 0.963 of the travel); a close or a hold
  returns once it is sent, with the outcome `sent`. `close_until_contact` with a
  `scout_opening` is the driver's two-stage close: the scout stage, then the seat stage,
  with the driver's own waits; the jaws are left pushing at the seat stage with the
  close's stiffness, and the RESULT says how many stages the driver skipped. `open` is
  the driver's own opening, and `close_torque` its close to the closed stop, which ends
  in the light hold; the robot app closes the jaws that way before it switches the
  motors off at rest. The typed `open` is that opening; the typed `close` is the two-stage
  close of APAI's `close_gripper` skill. The simulated arm has the two basic modes only
  and is a model of its own: there a close that stalls is still left at the light
  holding stiffness.
- **What the agent grips stays gripped.** No light hold follows a close: a STOP, a lost
  robot connection and the end of a task send jaws that no command is moving nothing.
  A STOP ends a running jaw command in the light hold: an opening within one pass of the
  driver's ramp (50 ms on the reBot gripper), a close before its next stage. The RESULT
  is then `cancelled` and carries the opening measured after the stop. Jaws that a
  person's trigger moved last get the light hold at a STOP, as before.
- **A fault of the jaws that leaves the arm usable is no fault-stop**: jaws blocked
  while they open, jaws that give no reading before an opening (nothing is sent), and a
  reference that finds something between the jaws. The RESULT is `failed` with the
  driver's own text, the outcome is `gripper_fault`, and the agent's skill fails with
  that text, as it does at the arm. Lost jaw feedback during an opening is still a
  fault-stop.
- **Jaws without their reference are referenced by the first jaw command**: the driver
  closes them slowly to their stop, then runs the command. Not while the robot app's
  note says they may hold something: then the command is refused with `the jaws are not
  referenced`.
- **The robot app reads the jaws while the arm holds**: once per ROBOT_STATE
  (`state.publish_period_s`, 0.05 s), and once more right after every move. It does not
  read them while a move sends setpoints, while a jaw command runs or with the motors
  off. The agent decides from that reading whether it holds something, closed on
  nothing or let go. Such a read waits for the jaw motor no longer than an idle joint
  read does (0.05 s with the rig's profile). One without an answer shows the jaws as
  unknown in that ROBOT_STATE and does not count against the jaw reference; after two
  in a row the next read comes five periods later, so that the joints keep the bus.
- **The numbers of "jaws proven empty" are APAI's** with `--arm apai`, and replace the
  `gripper` block's: closed below the profile's `grasp.air_grasp_frac`, at the target
  within the driver's reach (0.1 rad of the jaw travel), fully open within what the
  driver's guarded opening may end short (0.4 rad of the jaw travel; 0.065 of the rig's
  6.2 rad). `gripper.reach_tol` and `gripper.open_step` are read only for the jaws
  under a person's trigger. The driver's reach does not count for jaws that were told to
  close and read at or below the closed line: the driver's close before a torque release
  ends 0.12 rad short of the closed stop on the rig, and such jaws are closed on nothing.
  Something thinner than the closed line (0.25 rad of the rig's travel) between the jaws
  then counts as empty too.
- **A rest, a release, a shutdown or a hand-back refused only because the jaws are not
  proven empty asks you**: look at the jaws and type `jaws are empty`; the command is
  asked again, and the robot app takes your word until the jaws are next sent a command
  or the arm gets a new control owner (see
  [What the operator's commands do](#what-the-operators-commands-do)). If they hold
  something: `stop`, `open`, `yes`.
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
  RESULT. The typed `open` and `close`, the typed reference and the jaw steps of a
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
  little do not refuse it. With `--arm apai` the start pose is read up to three times
  when a motor misses an answer, as APAI's driver does (`MAX_READ_FAILURES`). The agent
  side samples a move anew when the arm has left the start pose (see
  [A move, its permits and the camera evidence](#a-move-its-permits-and-the-camera-evidence)).
- **What held before the check is asked again before the send.** If the check took time, the
  STOP, the permit and the large delay are tested once more; a setpoint whose permit ran out
  meanwhile is not sent: it waits for a new permit in a hold, at most the large delay.
  For the first setpoint the MOVE's expiry is tested once more as
  well: a MOVE that expired during that check is `refused` with `the move expired X.XXX s
  ago` and nothing is sent, also when the same check made the setpoint late. A send that
  itself took longer than the large delay ends the move
  `late`. Not covered: the send call itself, between its start and the motors.
- **A move is `completed` only on one read of every joint.** After the last setpoint the
  move settles: the contact watch goes on reading, a few joints per period. A reading
  that is not valid takes back what an earlier one showed for its joint. Once every
  joint's newest reading is valid and within `executor.settle_tol_rad` of
  the last setpoint (0.03 rad in the file for the simulated arm; with `--arm apai` the
  settle tolerance and settle time of APAI's driver replace the file's two values: 0.05
  rad and 2 s for the reBot arm, so a move that ends 0.04 rad off completes there as it
  does in the experiment's program), one read of every joint decides: only if it has every joint valid
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
  For the rig's profile, the one the experiment's launcher runs, the cap of the agent's
  and of the recovery limit set is the launcher's joint speed, read on the Thor from the
  experiment's `configs/rebot_workstation.yaml`; the cap of the human limit set is the
  robot app's own for a person, 2.0 rad/s (see [The rig files](#the-rig-files)).
- **The agent's checker on the Thor has the experiment's motion bounds** with the rig's
  profile: the bounds the experiment's launcher puts on its own checker (a distance of
  the gripper tip from a centre, and a joint deviation). The recovery checker does not:
  the return to rest takes another path than the experiment's.
- **A joint that leaves its hold is a fault-stop** with `--arm apai`: two readings in a
  row more than 0.10 rad from the target the arm was told to hold give `the motors do not
  hold the arm: jointN reads X rad, Y rad from the target it was told to hold (limit 0.1
  rad): it is pushed, blocked, or its motor is off`, and a motor whose fault report turns
  to a fault after the connect gives `motor NAME reports the fault 0x...: a motor with a
  fault ignores its commands, so it may not hold`. Keep a hand on the arm; `reset` only
  once the arm is supported. A joint that does not move when its motor goes off (on its
  stop, at rest) is not found this way. Neither ran on the arm.
- **What the reBot motor library does not report.** A command that reaches no motor raises
  nothing there. A lost setpoint shows as a contact once the arm no longer follows, or as
  lost joint feedback; a lost light jaw hold or a lost release shows nowhere. So a release on
  that driver is reported as `torque release sent at rest: the driver does not confirm that
  the motors are off`, and ROBOT_STATE's `torque_on` is then what was commanded.

## Known gaps

What is known not to work as one would wish, and is left as it is. None of these makes a
program cut torque: where they touch the arm at all, it holds.

On the human path:

- **A person's floor is 0.5 cm above the table plane of the arm model, not above the
  table.** The human program reckons the stop on the arm model, for the joint target
  it sends. Where the model's table plane lies under the real table, a joint's zero is
  off, or the arm droops under its joint target, the fingers touch the table before
  the tip of the model reaches the stop: 0.5 cm is all the room there is. APAI's
  contact watch and the person's eyes are what then stops the arm. Not run on the arm.
- **The floor knows the gripper tip only.** The fingers beside the tip and an object
  held in the jaws that reaches lower than the tip are not known to it: nothing keeps
  them off the table.
- **The contact watch was not run at a person's speed.** It runs with the thresholds
  of the rig's APAI profile, whose own speed cap is 0.8 rad/s; no move at 1.6 rad/s
  was run with them on the arm. The tests run them over stand-in motors that follow
  their command exactly: there a joint that stood ended a drive at 1.6 rad/s after 7
  setpoints, with the joint target 0.186 rad, about 11 degrees, ahead of it. On the
  arm, which trails its command, the watch may trip where nothing blocks it, or hold
  later at a real contact. The first session on the arm has to show both before
  the gripper is brought close to the table.
- **Under 2 cm and inside the agent's last grasp region the typed `rest` is refused,
  and a person can leave the gripper there.** A person's drive goes down to 0.5 cm and
  into that region; a return to rest starts only above `rest.table_clearance_m` and
  outside the region. A grip released there leaves an arm that takes no typed `rest`:
  the person squeezes again and lifts it out, which needs a working controller. With
  the headset path dead what is left is the 48 V switch with the arm held by hand.
- **A return to rest keeps the workspace box of APAI's profile**, which is narrower
  than a person's tip box on the rig. A return that starts with the gripper tip
  outside the profile's box is checked by the recovery checker's rule that such a tip
  may only come back toward the box. In a check on the arm model with the rig file's
  checkers, which is no test of the suite: of 600 random poses of the human joint box
  with the tip above 2 cm and outside the profile's box, the path to rest was refused
  from 4; of 1143 with the tip inside it, from 35. A refused return is the row `the
  path to rest was refused: ...` under
  [A takeover that cannot be ended with right A](#a-takeover-that-cannot-be-ended-with-right-a).
- **A close under the trigger is cut between two looks at the jaws.** On APAI's driver
  that is its next jaw poll, up to 0.15 s after the trigger asked for more; on the
  simulated arm one jaw step of 0.01 s. After a cut the jaws count as `may be holding`
  until they were measured fully open, as after a STOP in the middle of a close, so the
  hand-back then needs a full open also for jaws the person closed all the way. An
  opening goes one `gripper.open_step` per jaw command, so from closed to fully open it
  takes twenty jaw commands with the shipped 0.05; an opening step is not cut.
- **On the real arm a close that a STOP or stale controller data cut goes on softly.**
  It is left in the light hold at the close's own target, and at the next takeover that
  target is the grip level. A close that the trigger itself cut does not: its next jaw
  command aims the jaws anew. The simulated arm does not move in the light hold, so no
  test shows it.
- **A blocked opening under the trigger is left pressing open softly**, in the light
  hold at the step's target, at most `gripper.open_step` past where the jaws are. The
  agent's opening and the typed `open` are the driver's whole opening: its guard faults
  a blocked opening and holds where it stopped.
- **With a grip level of fully open, an opening that is still blocked leaves the
  trigger without effect** until the jaws were measured fully open: free the jaws, then
  right B twice, or `stop`, `open`, `yes` at the console.
- **The jaws under a person's trigger have not moved a real gripper.** Their steps of
  `gripper.open_step` at their own effort have no counterpart in the experiment's
  program; on the rig such a step may be too small to move the jaws. Human driving is
  switched off in the rig file.
- **The grip level is not measured at the takeover.** It comes from the last jaw
  target, or from the remembered jaw reading: a takeover reads no jaw.
- **A trigger that is past the grip level at the takeover moves nothing** until it
  comes back to within `human.jaw_match_tol` of it or crosses it.
- **A jaw opening that is no number from 0 to 1 stops the jaws only.** The joint target
  of that HUMAN_SETPOINT is judged by its own fields.
- **With the robot connection down the robot app takes a takeover from its own
  socket.** Where its file requires the live view, that takeover
  and every human drive are refused, because VIEW_STATUS comes over the robot
  connection. That is so with the rig's files. With `human.require_live_view: false`
  a person can take over and drive the real arm with the robot connection down. No
  console sees it then. The stops that remain are the Thor terminal's `stop` and right B.
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
  lag time: about 0.03 rad at 0.1 rad/s, and in proportion more at a higher speed. At a
  person's full joint speed, 1.57 rad/s, the motors are asked for about 0.19 rad, 11
  degrees, more before the arm holds against a joint that is blocked. Step F16 of the
  checklist writes down what it reports.
- **A `"visible": true` held up past a second fetch of its picture number still counts.**
  The view cannot tell which fetch of a picture number an acknowledgement is about, so
  such a late one makes the view live until the hidden tab's word about its second fetch,
  or until `status.ack_timeout_s` has passed.
- **A STOP between a new robot connection and its first ROBOT_STATE** keeps PERMITs and
  HUMAN_SETPOINTs at the robot link service until a later STOP is counted: the first
  ROBOT_STATE of a connection never lifts that hold, because it may be older than the
  STOP. The console's status line says `a STOP is not confirmed yet`; press right B or
  type `stop` again. This holds at the robot link service only: at the robot app's own
  socket (`--human-socket`) the robot app's first ROBOT_STATE ends the hold.
- **Right B released and pressed again within less than 1 s without controller data** is
  one stretch and sends no second STOP. The stop at the Thor terminal needs no headset.
- **A STOP behind a refused LOCAL_HELLO** goes out only when it is the record right behind
  it. A STOP that a refused client sends later is not read.
- **The note `hand N cm ahead: ...` and `ik.max_ahead_m` compare positions only.** A hand
  that was turned and is then held still leaves the tool turning, at the speed the joints
  are allowed, until it has caught up or the grip is released, and no note says so. A
  hand turned more than 115 degrees ahead of the arm ends the engagement.
- **The arm goes on after the hand stops**, by as far as the hand was ahead, up to
  `ik.max_ahead_m`: 50 cm on the rig. A hand that the headset loses track of, or a
  headset stream that stalls, leaves the hand's last pose standing, and the arm goes
  to it. Only a released grip, right B or a STOP at a terminal stops the arm at once.
- **The arm takes the straight way to where the hand is now**, not the hand's way, while
  the hand is ahead of it. The gripper can pass through what the hand went round; at
  most 50 cm of the hand's way are cut so.
- **A pause of the Thor of more than 0.1 s still ends the drive** on the Thor, with
  `setpoint N went out X s late, over the limit of 0.1 s`
  (`executor.large_delay_s`). Shorter ones were ridden out in real time on the
  simulated arm
  ([How fast the arm follows the hand](#how-fast-the-arm-follows-the-hand)). Not run on
  the arm: the pace of the Thor's loop on the real motor bus is not known.
- **Out of reach the wrist is left behind.** Where the hand's pose has no solution the
  arm follows by the tip's position first, so the gripper's angle can differ from the
  hand's while the note says `out of reach`. Once the hand is back in reach the wrist
  turns back to the hand's angle, at up to `retarget.max_angular_velocity`, and an
  object in the jaws turns with it. The solver's nearest pose is one step from the
  last joint target: the arm stands where no such step brings the tip nearer, also
  where another posture of the arm could reach further.
- **The human program does not know APAI's rule that the wrist camera stays on top of
  the gripper, or its check of the arm's joints at the table.** Each still ends a
  drive ([What still asks for a new squeeze](#what-still-asks-for-a-new-squeeze)). The
  check of the joints is met sooner with the floor at 0.5 cm: the tip may go down to
  0.5 cm, a joint between the base and the tip only to 1 cm.
- **How still a holding joint reads is not measured.** A second squeeze waits until two
  readings of every joint agree within 0.002 rad. On an arm whose readings differ by
  more at rest, every second squeeze within half a second of a release waits that half
  second.
- **A person cannot start outside the human checker's tip box.** A takeover at a pose
  whose gripper tip is outside it is refused by the human checker; the rule that a tip
  outside may only come toward the box is for an arm that got there under the person.
- **A drive starts from the measured pose, and nothing guards an arm that droops.** On
  an arm that comes to rest under its joint target each new squeeze starts again where
  the arm is measured: a squeeze with a still hand lowers the standing target by what
  the arm droops, so repeated squeezes let the tip sink by that much each, down to the
  floor. There the human program takes the tip back up to its stop, 0.5 cm above the
  table plane, and the arm droops under that again. By how much the rig's arm droops
  is not measured. Part F of the checklist watches for it.
- **A sample age is counted as more than it is, by the ROBOT_STATE's way to the human
  program.** The human program says when it sent on the robot app's clock as its newest
  ROBOT_STATE gives it, which is never later than the true moment. Where the human
  program is attached to the robot link service and the robot connection is slow, a
  sample age can pass `human.sample_age_limit_s` and hold the arm while the controller
  data are younger than the limit: a takeover is refused with `no fresh controller
  data: X s old, the limit is 0.25 s`, and a drive ends with `controller data stale
  (...)`. At the robot app's own socket that way is local. A view age is not counted
  as more: the viewer's own age is short by that same way. The round trip is
  shown and nothing is enforced on it.
- **Two stretches are in no sample age.** The time from the controller's own sampling to
  the tick of the Teleop Session Server is measured by nobody. Where the teleop
  connection crosses the two machines, the tick age rests on the clock-offset estimate
  of Isaac Capture's remote session, which is an estimate and no upper bound.
- **The robot app and the programs that send to it must be of one build**, and nothing
  checks it: every build says protocol version 1, and a program drops a field it does
  not know. A live view
  program without `sent_mono_s` in VIEW_STATUS gets every takeover refused (`the live
  view is not fresh: the VIEW_STATUS does not say when it was sent; ...`). A human
  program without it in HUMAN_SETPOINT is counted from the ROBOT_STATE it quotes. The
  other way round, a robot app older than the workstation's programs ignores a GRIPPER's
  expiry and counts a sample age and a view age without the way to the Thor, without a
  word. Compare the two machines' commit ids before every session (see
  [The same commit on both machines](#1-the-same-commit-on-both-machines)).
- **The first HUMAN_SETPOINT after the teleop feed entered a remote session is stale.**
  Its sample age holds the seconds the entry took. The next pass is fresh, and nothing
  can engage on that first one.
- **The Teleop Session Server serves one client.** While the human program holds it, a
  teleoperation session of Isaac Capture on the same server cannot run.
- **The bells of one console line are written together.** Whether one, two and three bells
  ring apart at a real terminal was not tried.
- **No person looked at the viewer's page in a browser window.** Its script ran under
  node, on stand-ins for the page and over HTTP against the viewer, and in headless
  Chrome on the launcher's real page (see [The live view](#the-live-view)).
- **The live view says that a tab shows the picture, not that a person looks at it.**
  Only a browser on the workstation gets a view token, also where the viewer listens
  on a network address without its access token: a page loaded from another machine is
  a watch-only page and never becomes the page that counts. A browser on another
  machine whose connection a program on the workstation passes on (an SSH port
  forward, a proxy) counts as one on the workstation (see
  [The live view](#the-live-view)).
- **After a robot link service that was down for long, the viewer attaches up to a
  minute late.** Its tries are 5, 10, 20, 40 and then 60 s apart. A takeover is refused
  meanwhile with `no VIEW_STATUS has arrived`.

On the torque release, the agent's jaw commands and the agent program's options:

- **After a torque release that failed, some texts still speak of motors that are on
  or of an arm that holds.** The agent program's line before a task's question reads
  `torque on`, and the task is refused by the robot app. A STOP's line still ends `the
  arm holds, stop number N`, and a reset's line reads `fault-stop cleared by reset; the
  arm holds, nothing was started`. The motion `release_failed`, the console's `torque
  unknown`, the fault-stop's line and the robot app's line at a SIGTERM are the ones
  to read.
- **With the arm's real driver a failed release leaves no second try.** Run on APAI's
  driver class over a stand-in motor library (`test_apai_backend.py`), not on the arm:
  a disconnect that raises drops the driver's connection to the motors, so the joint
  and jaw reads fail afterwards, a `release` right after a `reset` is refused, lost
  joint feedback latches a fault-stop again, and `quit` fails. The arm port refuses a
  second release there by itself. What is left is SIGTERM from a second terminal on the
  Thor and a new start of the robot app, with the arm at rest and a hand on it.
- **Whether the motors hold is read from the joints, not from the motors.** The reBot
  motor bus has no read of a motor's enable state, so `torque_on` in ROBOT_STATE says
  only that the robot app has not released the motors. `torque on (holding)` and `torque
  NOT HOLDING` come from joint readings against the target in force: a joint that does
  not move when its motor goes off (on its stop, at rest; the jaws) is not found. After
  `joint feedback lost ... in hold` the Thor's log adds that the hold cannot be read
  until the joints answer again. None of this ran on the arm.
- **The rest that switches the motors off, `on`, and SIGTERM at rest have not run on
  the arm.** They ran on the simulated arm, on APAI's mock driver and on APAI's reBot
  driver class over a stand-in motor library. The agent's rest tool sends its fold as
  moves of the agent onto the joint stops; a test runs that across the robot link onto
  APAI's reBot driver class over a stand-in motor layer (`test_rest_end_to_end.py`). On
  the arm it was not seen. If a setpoint of the fold is refused there, the tool ends
  with the arm holding and the motors on: type `rest` at the Thor terminal.
- **`on` connects the driver a second time in one process.** On the real arm that is
  APAI's connect, which cuts torque for a moment, at measured rest; nothing has run it
  twice in one process on the arm.
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
- **`--motion-approval tool` is not built**, and the joint feedback in the debug
  recording is that of ROBOT_STATE as the agent program read it, not the motors' own
  reads; no motor command is in it (see
  [The options of the experiment's launcher](#the-options-of-the-experiments-launcher)).
  None of those options ran on the real full stack.
- **The experiment's viewer shows a run only when it was recorded with
  `--record-debug`.** A run without it has no page there, also not for its timing, and
  a run recorded before the agent program wrote `resolved-config.yaml` is not listed.
  Between the moments at which APAI reads the arm the viewer's figure of the arm
  stands still, also while a person drives. Its page was opened in no browser from
  this example, and it has read no recording of the real full stack (see
  [The experiment's viewer](#the-experiments-viewer)).
- **No test starts tmux through `run_workstation.sh`** (see
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
  run_agent.sh run_camera_receiver.sh run_console.sh
                              start scripts for the workstation
  run_viewer.sh               start script of the experiment's viewer, for the
                              workstation: its page is the live view
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
      debug_recording.py                 what the debug recording stores of the obstacle map, and drops at its size
      clear_grid.py                      the clear grid of the one-machine recipe
      console_cmd.py                     the console command
    human/
      types.py                           the human program's settings and the types its parts share
      retarget.py ik.py                  the retargeting graph; the inverse kinematics
      logic.py                           buttons, the new-squeeze rule, the ahead checks
      feed.py __main__.py                the controller feeds; the human program
    console/
      console.py status.py __main__.py   the operator console; its lines; its command line
    viewer_live/
      live.py page_block.html            the live picture of the experiment's viewer: the
                                         pictures; the block that goes into the viewer's page
      picture.py settings.py             the picture's encoder; the viewer's fixed values
      view_model.py view_link.py         the live view: when it is live and how old;
                                         VIEW_STATUS from the viewer
      robot.py                           the live arm: the newest ROBOT_STATE for the page
      __main__.py                        the launcher that runs the experiment's viewer with it
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
`stop` that ends the wait, a typed `rest` that ends with the motors off at rest, a typed `on`
that switches them on again without a new start, a release and a typed `quit`. Twelve
steps, about 5 s. It needs no Isaac Capture.

`--selftest-link` runs, in one process, a robot link service on 127.0.0.1 and a free port with
its socket in a temporary folder, the robot app dialing it over a real link, and two scripted
local clients, an agent and a console: take ownership, a move with permits, a move without a
permit (refused), a STOP from the console during a move, a new task, the robot connection cut
during a move (the arm holds, the move's RESULT is `failed: robot connection lost`), the dial
that follows (the same session, nothing moved, the agent's old command refused), `rest`
from the console, which ends with the motors off at rest, and the robot app closed. Eleven
steps, about 6 s. Without Isaac Capture on the
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
installed (`uv pip install -e ./examples/agentic_remote`) and an Isaac Capture build on
the Python path for the first two programs. No live view runs in this recipe: the live
view is the page of the experiment's viewer, which the rig runs (see
[The live view](#the-live-view)).
The other way installs nothing: the two variables `PY` and `PYTHONPATH` of
[the next recipe](#run-demo-1-by-hand-with-the-mock-agent) name a Python that has the
packages and put the example's `python` folder and the Isaac Capture build on its path;
`python` in the commands below is then `"$PY"`.
Three terminals, each in `examples/agentic_remote`.

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
line with `owner nobody #0` and `view none`: no viewer reports a live view here. Type
`state`: its last line is `takeover: not possible: human
driving is switched off in this robot app's file (human.enabled)`. Type `stop`: the console
answers `stop: taken`
and the Thor terminal shows `[stop] STOP from console: ...`. Type `rest`, then
`path is clear`: the answer is `rest: RESULT completed`, the Thor terminal shows the
`[jaws]` and `[release]` lines, and the status line shows `rest_unpowered` and `torque
off`. Type `on`, then `yes`: the motors are on again (`rest_hold`, `torque on`).

**To end**: type `shutdown` at the console and answer `yes` (the arm is at rest), or type
`quit` at the Thor terminal; the robot app ends with exit code 0. Then `quit` at the console
and Ctrl+C in the service's terminal.

The tests run these three programs as processes, the operator console on a pseudo-terminal
(`test_the_programs_of_demo_1_start_as_processes_and_end_cleanly`). What differs from the
steps above: a port the system picks, a socket in a temporary folder, a file of the test for
the service, and the console started with `python -m` and not through its start script.
The test types `rest` and `path is clear` at the console; its
`state` and `stop` go through the console command. With the shipped port 47610 and the
default socket path they were not run. For two
machines, put the workstation's own address in `listen.address`, the Thor's in
`allowed_peers`, and give the robot app the workstation's address.

**With a task of the agent**, the agent program needs more than the three programs above:
the robot app on APAI's software arm with a file whose limit sets fit the profile's speed
cap, an agent file for APAI's mock scene, and an obstacle grid. The next section is that
recipe.

The tests run the agent program inside the test process on APAI's mock scene, with a
stand-in for the obstacle grid bridge and the typed lines scripted, together with the robot
link service, the console and a live view made of the viewer's own parts
(`test_demo1_programs.py`; see [What was run](#what-was-run)).

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
imports. `BUILD` is the Isaac Capture build for that Python. Four terminals:

| # | Terminal | Command | Must show |
|---|---|---|---|
| 1 | service | `"$PY" -m isaaccapture_examples.agentic_remote.link_service --config configs/link_service.yaml` | `robot link service: waits for the robot app on 127.0.0.1:47610 (taken from 127.0.0.1); local API at <socket path>` |
| 2 | Thor terminal | `"$PY" -m isaaccapture_examples.agentic_remote.robot_app --config configs/robot_app_mock.yaml --arm apai --driver mock --connect 127.0.0.1:47610` | `robot app: human driving is not tied to a live view (human.require_live_view false)`, then `robot app: session <id>; dialing the robot link service at 127.0.0.1:47610`, then `[connection] robot connection made: the arm holds as it was; a new connection starts no motion` |
| 3 | console | `AGENTIC_REMOTE_CONSOLE_PYTHON="$PY" ./run_console.sh` | `WELCOME: robot apai-mock-arm, profile apai-rs-model, session <id>, driver apai-mock, camera calibration apai-mock-calibration`; a status line with `driver apai-mock`, `owner nobody #0`, `rest_hold`, `jaws empty`, `torque on, hold unknown` (no joint target was sent yet; later `torque on (holding)`), `live view not required` and `no human program` |
| 4 | agent | `AGENTIC_REMOTE_AGENT_PYTHON="$PY" ./run_agent.sh --config configs/agent_mock.yaml --camera mock --interactive` | `[agent] run folder ...`; two `[agent] WARNING:` lines, first that of the invented exposure times (`the camera frames of --camera mock carry invented capture times ...`), then `experiment.grid is clear: ...`; `[agent] robot connection up: robot apai-mock-arm, profile apai-rs-model, driver apai-mock, ...`; `[agent] obstacle grid: live (... s old); refreshes 1`; `[agent] ready. Ctrl+C at this terminal is STOP: the arm holds with the motors on. When the program ends you are asked whether the arm returns to rest; there the robot app switches the motors off`; the prompt `task> ` |
| 5 | agent | Type `open the gripper`. The question ends in `Type yes to proceed:`; type `yes` | Agent: `[agent] the agent owns the arm (owner number 1)`, `[agent] task completed: the arm holds where it is and nobody owns it; ...` and a task report with `success: True`. Console: `[owner] new task: the agent owns the arm, owner number 1`, `[owner] the agent gave the arm back: task completed; the arm holds and nobody owns it` and a status line with `owner nobody #2` |
| 6 | agent, console | Type `pick up and place the red object`, then `yes` | Console: `[owner] new task: the agent owns the arm, owner number 3`; the task's moves, with status lines that show `owner agent #3`, the move's progress as in `moving 47/104 sending` and the held permit as in `permit 68, evidence 0.34 s`; after about two minutes the help request (below) |
| 7 | console | `stop`. Then `rest`, then `path is clear` | `[stop] STOP from console: stop typed at the operator console; the arm holds, stop number 1; the assistance wait is cancelled` and `stop: taken. ...`. Then the `[rest] return to rest: ...` lines of the return, `[rest] return to rest: the arm is at rest, motors on`, `[jaws] at rest: closing the jaws with the driver's close`, `[jaws] at rest: the jaws are closed`, `[release] torque released at rest: the motors are off` and `rest: RESULT completed {"motors": "off", "outcome": "at_rest"}`; a status line with `rest_unpowered` and `torque off` |
| 8 | all | Ctrl+D at the agent's `task> `. `shutdown`, then `yes`, at the console; `quit`. Ctrl+C in terminal 1 | Agent: `[agent] the arm lies at rest with the motors off`, then `[agent] 1 task(s) of this session ended without success`, exit code 1. Console: `[shutdown] shutdown: the motors are off at rest; the robot app closes`, `shutdown: RESULT completed` and `ROBOT CONNECTION LOST: ...`. Exit code 0 for the robot app, the console and the service (`robot link service: SIGINT; closing. The robot holds on its own.`) |

**The pick and place of step 6 ends in a help request.** The robot app's mock arm has
nothing between its jaws, and the agent decides what it holds from the jaw reading the
robot app measures. So every grasp closes on nothing: APAI opens the jaws and tries
again, each move under a permit, and after its eighth try the task is stuck. That takes
about two minutes. The agent terminal prints `[agent] HELP NEEDED: stuck: I could not
hold 'red object' after 8 grasp attempt(s): ... The arm holds and nobody owns it;
assistance wait 120 s` and a task report with `success: False`. The console prints
`[help_needed] HELP NEEDED: help needed: stuck: I could not hold 'red object' after 8
grasp attempt(s): ...; the arm holds and nobody owns it` and the help block: the jaws,
`time left: 120 s; then the arm keeps holding`, the words `stop` and `rest`, and as its
last line `takeover: not possible: no human program is attached: start run_human.sh`. A
status line follows with `owner nobody #4`, `hold` and the wait counting down, as in
`wait 119 s`. The arm holds at about [0, 1.2, 1.2, 0, 0, 0] rad with its motors on. This
is where demo 2 begins.

The agent terminal's task report gives the reason in full, as `summary: stuck at
pick_and_place(object=red object): ...`, with a line that begins `[verification] could
not confirm:`.

By hand the agent program writes into `examples/agentic_remote/.state`: the run folders,
APAI's stores and caches, and a `.gitignore` that holds `*`.

Every "must show" above is what the programs printed in a test that ran these commands as
processes, with the differences [What was run](#what-was-run) lists. A person has not run
them. Of steps 7 and 8 the test waits for `stop: taken`, `rest: RESULT completed`, the
motion `rest_unpowered` with the motors off, the agent's `the arm lies at rest with the
motors off` and the exit codes; the `[jaws]`, `[release]` and `[shutdown]` lines between
them are those the robot app prints for every such `rest` and shutdown.

## Run demo 2 on one machine

A synthetic controller takes the simulated arm over, lifts it and gives it back. No GPU, no
model, no camera, no headset and no arm. Every "must show" below is what the programs
printed in a test that ran these commands as processes
(`test_the_recipe_of_demo_2_runs_as_processes_with_the_shipped_files` in
`test_demo2_programs.py`), with these differences: the service and the human program ran
on copies of their files that name a port the system picked, a socket of the test and, for
the Teleop Session Server, a port nobody listens on; the console was started with
`python -m`, not by its script; a local client of the test in the role `tool` read
ROBOT_STATE, so the console counted `1 tool` too. No live view runs in this recipe: this
robot app's file does not require one. A person has not run them.

The environment lines of [the recipe before](#run-demo-1-by-hand-with-the-mock-agent) in
every terminal. The robot app runs on `configs/robot_app_demo2.yaml`: `robot_app.yaml` with
`human.enabled: true`, `human.require_live_view: false` and a human joint box of -2.5 to
2.5 rad, which lies inside `rest.region_*`. The robot app takes that file only with a
simulated driver. The human program runs on `configs/human.yaml`.

| # | Terminal | Command | Must show |
|---|---|---|---|
| 1 | service | `"$PY" -m isaaccapture_examples.agentic_remote.link_service --config configs/link_service.yaml` | `robot link service: waits for the robot app on 127.0.0.1:47610 (taken from 127.0.0.1); local API at <socket path>` |
| 2 | Thor terminal | `"$PY" -m isaaccapture_examples.agentic_remote.robot_app --config configs/robot_app_demo2.yaml --arm sim --connect 127.0.0.1:47610` | `robot app: human driving is not tied to a live view (human.require_live_view false)`, then `robot app: session <id>; dialing the robot link service at 127.0.0.1:47610`, then `[connection] robot connection made: the arm holds as it was; a new connection starts no motion` |
| 3 | console | `AGENTIC_REMOTE_CONSOLE_PYTHON="$PY" ./run_console.sh`; type `state` | `WELCOME: robot sim-arm, profile sim-6dof, session <id>, driver sim, camera calibration none`; a status line with `driver sim`, `owner nobody #0`, `stop none #0`, `rest_hold`, `jaws empty`, `torque on`, `live view not required` and `no human program`. `state` prints the service's status, `ROBOT_STATE: control owner nobody (owner number 0), stop number 0, stop level none, motion rest_hold, jaws empty, torque on, assistance wait none`, a status line and, last, `takeover: not possible: no human program is attached: start run_human.sh` |
| 4 | human program | `AGENTIC_REMOTE_HUMAN_PYTHON="$PY" bash run_human.sh --config configs/human.yaml --controller synthetic:lift` | `human program: synthetic controller "lift"; driver sim; arm model mock, 6 joints`, then `human program: right controller seen` and a report line each second that begins `human program: stage ready` and holds `owner nobody; engaged no; hand x 0.000 y 1.200 z -0.300 m`. Console: a status line that counts `1 human` and shows `headset ready`, `controller 0.03 s` (the age differs from line to line) and `note: hold right B for 1 s with the grip released to take over` |
| 5 | console, watch | - | Human program: `human program: B down`, then `B up`, `grip down` and `grip up`. Console: `[stop] STOP from controller: right B; the arm holds, stop number 1` and a status line with `owner nobody #1` and `stop stop #1`; two bells and `[owner] TAKEOVER: a person owns the arm, owner number 2; release the grip, then squeeze it to move`; a status line with `owner human #2`, `stop none #1`, `takeover open` and `needs release`; one with `note: squeeze the grip to move`; one with `moving N/0`, `human driving` and `takeover open` (N is the number of setpoints the drive had sent when the line was printed; 0 in the test's run); then one with `hold`, `takeover open` and `note: squeeze the grip to move`. The gripper tip is 5 cm above where it is at rest |
| 6 | console | `rest`, then `path is clear` | `rest: RESULT accepted`; `[rest] return to rest: the arm moves for 5.0 s (largest joint travel 0.24 rad)`; a status line with `owner recovery #3`, `takeover open`, the return's progress, as in `moving 24/249 sending`, and `note: the arm returns to rest; right B is a STOP`; `[rest] return to rest: the arm is at rest, motors on`; `[rest] return to rest: a takeover is open: the motors stay on at rest for the hand-back`; `rest: RESULT completed {"motors": "on", "outcome": "at_rest"}`; a status line with `owner human #4`, `rest_hold` and `note: at rest: hold right A for 1 s to give the arm back`. Then, by the script's right A (`human program: A down`, then `A up`): two bells and `[owner] takeover ended: the arm is given back; nobody owns it`, and a status line with `owner nobody #5` and `rest_hold` |
| 7 | console | `shutdown`, then `yes`; `quit`; Ctrl+C in terminals 4 and 1, in that order: with the service ended first, the human program ends by itself with exit code 1 and `the robot link service is gone (...)` | Console: `shutdown: RESULT accepted`, `[rest] return to rest: the arm is at rest, motors on`, `[jaws] at rest: closing the jaws with the driver's close` (a status line shows `jaws loaded` while the simulated jaws close), `[jaws] at rest: the jaws are closed`, `[release] torque released at rest: the motors are off`, `[shutdown] shutdown: the motors are off at rest; the robot app closes`, `shutdown: RESULT completed {"outcome": "at_rest"}`, a bell and `ROBOT CONNECTION LOST: receive failed: connection closed; the robot holds on its own`; after `quit`, `the operator console ends; nothing was sent for that`. Service: `robot link service: SIGINT; closing. The robot holds on its own.` The human program prints nothing at its end. Exit code 0 for all four programs |

The owner number goes up at every change of control owner and at the STOP: nobody #0, the
STOP's nobody #1, human #2, recovery #3 for the return to rest, human #4 and, after the
hand-back, nobody #5.

The script `lift` presses right B 3 s after the human program's first pass, holds it until
the person owns the arm, squeezes the grip, raises the hand 5 cm in 4 s and releases. It
then waits for the arm to hold at rest, which is the typed `rest` of step 6, and holds
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
(`--config configs/robot_app_mock.yaml --arm apai --driver mock`), and a fifth terminal runs
the agent program as there (step 4 of that recipe); the human program is started later.
The console's `WELCOME` then says `driver apai-mock`. The task `pick up and place the red
object`, typed as the first task with `yes`, moves the arm under owner number 1 and
ends, after about two minutes, in the help request of
[the recipe before](#run-demo-1-by-hand-with-the-mock-agent): the mock arm has nothing
between its jaws, so the task is stuck (`HELP NEEDED: stuck: I could not hold 'red
object' after 8 grasp attempt(s): ...`). The console shows the `HELP NEEDED` lines and
the help block with `time left: 120 s; then the arm keeps holding` and, as its last line,
`takeover: not possible: no human program is attached: start run_human.sh`; then
`[countdown] assistance wait: 120 s left, then the arm keeps holding (...)` and a status
line with `owner nobody #2`, `hold`, the wait counting down and `no human program`. Then
start terminal 4, which prints `human program: synthetic controller "lift"; driver
apai-mock; arm model mock, 6 joints`, and go on with steps 5 and 6. What differs from the
table:

- The STOP's line ends `; the assistance wait is cancelled`, and its status line shows
  `owner nobody #3` and `hold`.
- The takeover has owner number 4. The gripper tip rises 5 cm from where the agent left
  the arm, at about [0, 1.2, 1.2, 0, 0, 0] rad.
- The return to rest is longer: `[rest] return to rest: the arm moves for 30.9 s (largest
  joint travel 1.48 rad)`, with `owner recovery #5` and progress as in `moving 30/1543
  sending`. Then `owner human #6` and, after the hand-back, `owner nobody #7`.
- No `open` is needed: the hand-back is taken with the jaws as the agent left them.

After the hand-back the task `open the gripper`, with `yes`, runs. The console prints
`[owner] new task: the agent owns the arm, owner number 8` and `[owner] the agent gave the
arm back: task completed; the arm holds and nobody owns it`, and the agent's task report
says `success: True`. No program was restarted: the console printed one `WELCOME`. Ctrl+D
at `task> ` ends the agent program with `[agent] 1 task(s) of this session ended without
success` and exit code 1. On its way out it asks its question about the return to rest
(`End of the session: ...`), where its terminal still takes a line; an empty line leaves
the arm holding with the motors on, and step 7 ends the other programs with exit code 0.

This variant ran as processes in the same test file
(`test_demo_2_with_the_agent_failing_first_runs_as_processes`), with the same differences
and with the agent program on a copy of `configs/agent_mock.yaml` that names the test's
socket and state folder; a person has not run it.

**With the human program at the robot app's own socket**, as it runs on the Thor. Start
the robot app of terminal 2 with `--human-socket` and a path added, and the human program
of terminal 4 on a copy of `configs/human.yaml` whose `socket_path` is that path. The
Thor terminal prints `robot app: the human program's socket is` and the path after its
session line. With the human program attached, the console's status line shows `human
program at the robot app` in place of `no human program`, and its `clients` part counts
no human. Right B, the takeover, the lift, the typed `rest` and right A go as in steps 5
and 6. End the human program with Ctrl+C before the shutdown of step 7: it ends with
exit code 0, and the console shows `no human program` again. A test ran this as
processes, on copies of the two files
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
`test_viewer_live.py`, `test_demo1_programs.py` and the files that take their live view
from `live_view_bench.py` are skipped as a whole.

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
  refused after it, the fault-stop's line, the second release, the shutdown, and the
  program's line at a SIGTERM. `test_apai_backend.py` has the same on APAI's reBot
  driver class over a stand-in motor library: no second release, and a STOP while the
  driver's disconnect runs.
- `test_gripper_timeout.py`: a jaw command the arm adapter gave up on, with the real
  robot app on the simulated arm (marked `apai`), and the expiry of a GRIPPER on the
  robot app.
- `test_agent_options.py`: the arguments of the experiment's launcher in the agent
  program: what a run takes, what is refused, the order of the recordings around the
  robot connection. Its last test, marked `apai` and `experiment`, runs the
  experiment's own modules on APAI's mock scene.

`test_agent_measures.py` holds the six files of
[What a run records and measures](#what-a-run-records-and-measures): every number of them
on a scripted stream of ROBOT_STATEs, and the real robot app on the simulated arm.

Six files compare the two-machine programs with the experiment's one-machine program,
each test against that program's own routine or file. They are marked `apai` or
`experiment` and are skipped without the two folders:

- `test_jaws_conform.py`: APAI's skill code on its reBot driver directly, and through
  the arm adapter, the link and the robot app onto the same driver, both on a stand-in
  motor layer: the driver calls, the jaw motor commands, the readings and the errors of a
  pick, a place, a close on nothing, an opening, a blocked opening and the typed `open`
  and `close`.
- `test_motion_conform.py`: the first setpoint from rest, the start tolerance and the
  start-pose retry, the settle tolerance, the speed cap, the motion bounds, a joint
  path, a late permit, the start reads and the hold watch, against APAI's stream and
  the experiment's launcher.
- `test_tools_conform.py`: the agent's tool list and what a tool that is not admitted
  gets, the question about the motors, and APAI's rest tool through the arm adapter.
- `test_lifecycle_conform.py`: the route of a typed `rest` against the joint targets
  APAI's rest tool commands, the order rest proven, jaws closed, torque cut, the motors
  on again without a new start, the typed `reset`, and SIGTERM at rest and away from it.
- `test_launch_conform.py`: the agent program's start against the experiment's
  launcher: no APAI variable from `run_agent.sh`, the command `run_workstation.sh`
  types, the configuration key by key, the joint speed of each speed profile, the
  stereo matcher, the hand-eye file of the mounting record, the first camera frame.
- `test_perception_conform.py`: the same stereo pair through the experiment's camera
  and through the agent's, the age rules, the mask pose, and the hand-eye fit the rig's
  agent file names.

`test_rest_end_to_end.py`, marked `apai`, runs APAI's rest tool from the agent side
across the robot link onto the robot app with APAI's reBot driver over a stand-in motor
layer: the fold, the jaws' close, the motors off at measured rest, and the motors on
again.

`test_viewer.py` holds the experiment's viewer on runs the agent program recorded on
APAI's mock scene, asked over HTTP on a port the system picks, and `run_viewer.sh`. Its
tests of the viewer are marked `apai` and `experiment`; those of the script's arguments
and refusals need neither. The file takes about 30 s with everything present.

`test_debug_recording.py` holds what the debug recording stores of the obstacle map
(`--debug-map-hz`), what it does at its size limit (`--debug-at-limit`) and the lines the
agent program says of it. Its tests of the rate and of the limit are marked `apai` and
`experiment`: they run APAI's own watcher and obstacle map on a stand-in camera and a
stand-in grid bridge, with the experiment's recorder, viewer and report and limits of a
few megabytes, and three of them run the agent program on APAI's mock scene. The file
takes about 90 s with everything present and writes about 400 MB into the temporary
folder, which it leaves to pytest.

`test_viewer_live.py` holds the live picture of the experiment's viewer and the live
view: the stream's rate and what a slow browser gets, on a stand-in camera; the numbered
pictures and their acknowledgements; the block for the page, whose script node runs on
a stand-in for the page and over HTTP against the viewer, where node is installed; the
VIEW_STATUS the viewer sends, situation by situation on a clock the test sets; the
picture's encoder and the viewer's fixed values; the viewer's
answers under the launcher against those of the viewer alone; the whole launcher,
started by `run_viewer.sh` on a fake camera receiver; and a person's takeover on the
rig's robot app file, behind a robot link service that runs as a process. It also
holds the live arm: `/api/live/robot.json` on a stand-in for the robot app that gives
20 ROBOT_STATEs a second, with fresh readings and with readings that stand still
(motors off, joint feedback lost), which are never called live; how long an answer is
on its way; the arm's geometry against the viewer's own, and the page's
own script with the block, which node runs with the harness's clock and over HTTP and
headless Chrome runs on the launcher's real page, where Chrome is installed. Its tests
on the experiment's viewer are marked `apai` and `experiment`, those behind a service
`link`; the others need OpenCV. The file takes about 235 s with everything present.

`test_viewer_live_model.py` holds the live view's rule by itself, on times the tests
pick: when the viewer's model (`viewer_live/view_model.py`) counts the view as live, the
view token, and the age it reports. `test_viewer_live_link.py` runs the viewer's live
view on a robot link service of the test: VIEW_STATUS reaches the robot app on the
simulated arm and shows in ROBOT_STATE (marked `link`); on real clocks no picture from
before a step of the Thor's wall clock is reported live; and the viewer's files can send
VIEW_STATUS only and import no GPU library. `live_view_bench.py` is the viewer's live view for
the tests that run one beside other programs: its live picture on a stand-in camera,
its request handler and its link to the robot link service of the test, with no
experiment folder.

The executor's rules for a MOVE's expiry at its first setpoint and for the read a
`completed` move ends on are tested in `test_executor.py`.

The human path's tests are `test_takeover.py`, `test_human_drive.py`, `test_human_jaws.py`,
`test_lift_out.py`, `test_human_on_thor.py`, `test_human_logic.py`, `test_human_ik.py`,
`test_human_retarget.py`, `test_human_feed.py`, `test_human_program.py`,
`test_human_realtime.py` and `test_demo2_programs.py`.

- `test_human_jaws.py`: the jaws under a person's trigger on the robot app, with the
  human targets the tests make: the grip level, the trigger that has and has not
  reached it, the limit on closing, the light hold, what ends the jaw motion, the two
  settings. One test, marked `apai`, runs on APAI's software arm.
- `test_lift_out.py`: a person's drive close to the table with a table clearance of
  their own, 0.0 as on the rig, and a takeover with the gripper tip under it, on APAI's
  checkers and its software arm. Without APAI all its tests but one are skipped.
- `test_human_on_thor.py`: the human program attached at the robot app's own socket
  (`--human-socket`), as on the Thor: WELCOME, ROBOT_STATE and the RESULTs there, what
  the socket refuses, right B's STOP with the robot connection down, a client that
  does not read, one human program at a time. Its last two tests, marked `link` and
  `apai`, start the programs as processes and take about 30 s.
- `test_human_realtime.py`: the human program's `main()` against the rig's robot app
  on the simulated arm, both on the machine's clock: a second squeeze 40 to 200 ms
  after a release at full joint speed, pauses of the robot app of 60 to 500 ms, and
  the 30 cm move with nothing paused. It has 39 tests and takes about 4 min; what it
  measures depends on what else the machine does, so run it by itself when a time
  matters.

Those that load Isaac Capture's retargeters
(`test_human_retarget.py`, `test_human_feed.py`, `test_human_program.py`,
`test_human_realtime.py` and
`test_demo2_programs.py`) are skipped as a whole without the build on the Python path, by
no marker. `test_human_ik.py` is marked `apai`, and `test_human_program.py` and
`test_human_realtime.py` `link` and `apai`. `test_demo2_programs.py` runs the programs of demo 2 together, as
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
the test) and a live view made of the viewer's own parts (`live_view_bench.py`: its live
picture, its request handler and its link, on threads, with stand-in camera pictures or
with no camera). Its tests:

| Test | Shows |
|---|---|
| `test_demo_1_on_the_mock_arm_with_console_and_view` | Two tasks run; every joint target was under a permit; the console printed each owner change; the view age reached ROBOT_STATE and the console; the viewer had the arm's state while it moved |
| `test_a_console_stop_during_the_agents_move_ends_the_task_and_the_next_task_runs` | `stop` typed at the console during a move ends it; the next task runs without a restart |
| `test_a_console_rest_after_the_task_returns_the_arm` | `rest` with `path is clear` returns the arm; `yes` sends nothing |
| `test_an_obstacle_costs_no_restart` | An obstacle ends the first task with a help request; the same program runs the second task |
| `test_the_agent_program_gone_shows_the_help_request_and_the_countdown_on_the_console` | The assistance wait on the console; a typed `stop` ends it; nothing moves |
| `test_the_console_and_the_view_survive_a_restarted_robot_app` | A new session id is shown, and the viewer has the new robot app's states; a command with the old numbers is refused |
| `test_the_programs_of_demo_1_start_as_processes_and_end_cleanly` | The service, the robot app (`--arm sim`, and `--arm apai --driver mock`), the operator console (on a pseudo-terminal) and the console command as processes of their own |

The file takes about 65 s. The view's page is asked over HTTP, a few times per second at
most; no test starts a browser.

## The rule: torque is never cut away from rest

The arm has no brakes: with the motors off it falls, unless it lies at rest. So the robot app
switches the motors off through one call, the arm port's `release_torque()`, and makes that call
in one place, the guarded release in `robot_app/rest.py`. With APAI the call is
`ApaiArm.release_torque()`, which disconnects APAI's driver: the second and last place in the
package that cuts torque. The guarded release runs at five ends, each of which a person
asked for or allowed: a `release`; the end of a typed `rest` (also the one the agent
program asks for when it ends, after the operator's `path is clear`); the shutdown that
`quit` asks for; the end of the agent's rest tool, which asks for the motors off after it
folded the arm in a task the operator allowed; and SIGTERM. At every one of them it runs
only when two full joint reads 0.2 s apart are within 0.05 rad of rest and of each other,
the jaws are proven empty (by measurement, or on the operator's typed `jaws are empty`; the
rest tool has judged them itself before it folded the arm), no move is running and no
fault-stop is latched. Away from measured rest each of the five leaves the motors on and
says why. Before the cut the robot app closes the jaws with the driver's close; a close
that fails is told and the release follows.

**A release whose driver call fails.** If the driver's disconnect raises an error, the
robot app does not know whether the motors are off: the driver may have switched some or
all of them off before it failed. The motion state becomes `release_failed`, ROBOT_STATE
keeps `torque_on` true, and a fault-stop latches with the reason `torque release failed,
the motors may be on or off: the driver's disconnect failed: <the driver's error>`. The
console shows `torque unknown`. Treat the arm as powered. The fault-stop's line names the
way on in place of a reset: `FAULT-STOP: torque release failed, the motors may be on or
off: ...; end the robot app with SIGTERM and start it again with the arm at rest, stop
number N`. Every later fault-stop in this state ends `...; a torque release failed and the
motors may be on or off; end the robot app with SIGTERM and start it again with the arm at
rest, stop number N`. From then on nothing moves the arm: a task, a MOVE, a jaw command,
`open`, `on`, a takeover and a return to rest are refused with `a torque release failed and
the motors may be on or off; end the robot app with SIGTERM and start it again with the arm
at rest`, and with APAI the arm port sends the driver no target and no jaw command. The
way on that works with every driver is that new start: send the robot app SIGTERM from a
second terminal on the Thor (Ctrl+C is a STOP, and `quit` only tries the release again),
then start it again with the arm at rest and a hand on it. At the SIGTERM the program says
`robot app: SIGTERM; closing after a torque release that failed: the motors may be on or
off` and switches nothing; its log line is `robot app closed; torque on: not known, a
torque release failed: the motors may be on or off`. With the arm's real driver there is
no second release: a disconnect that raised drops the driver's connection to the motors,
so the joints and the jaws read no more. After a `reset` a `release` is refused there
with `the jaws are not proven empty (unknown): a jaw read failed`; lost joint feedback
then latches the fault-stop again and `release` is refused with `fault-stop latched:
...`. The arm port refuses a second release on this driver by itself (`an earlier
disconnect failed: the motors may be on or off, and this driver takes no second
release`): the driver's next disconnect would return without reaching a motor. A refused
release ends `release refused: <why>; the motors may be on or off; end the robot app with
SIGTERM and start it again with the arm at rest`. `quit` and `shutdown` start no return to
rest in this state and ask for no `path is clear`; they try the release again and close
the robot app only if it worked. Otherwise the app keeps running: `shutdown failed: a
torque release failed and the motors may be on or off; release refused: <why>; the robot
app keeps running; end the robot app with SIGTERM and start it again with the arm at
rest`, or a new fault-stop when the driver's call fails again. On the simulated arm and
on APAI's mock driver a second release can work: `reset`, then `release`, tries it again
under the same conditions (two joint reads at rest); the jaws were judged before the
release that failed and are sent nothing more. If it works the state is `rest_unpowered`
and the lines are those of any release. An arm port that knows it switched its motors
off before the call raised, as the simulated arm does, is shown as `rest_unpowered`.
While the driver's disconnect runs (over half a second on the arm, read from the motor
library, not measured) the arm port sends the driver nothing: the light jaw hold of a
STOP or of a lost robot connection is left out, and `torque_on` stays true until the call
has returned. Checked on the simulated arm and APAI's mock driver
(`test_release_failed.py`) and on APAI's reBot driver class over a stand-in motor library
(`test_apai_backend.py`), not on the arm.

One thing outside the rule: connecting APAI's driver for the real arm cuts torque (its
`connect()` clears motor faults with a command that does so). The robot app connects when
it starts, and again only for `on`, after a release at rest switched the motors off: a
lost or a new robot connection does not touch the arm port, and an arm port whose motors
are on is never connected again. The
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
service, Ctrl+C, a SIGTERM away from measured rest, a closed terminal and closing the
robot app. Tests keep it that way:
the simulated arm records the pose at every torque
release and a test fixture fails any test in which one happened away from rest, and
`test_wiring.py` and `test_no_limp.py` scan the source for any other place that could cut
torque. This is a rule about the code; sections 9 and 10 of the design list what software
cannot prevent.

The agent side has no way to cut torque itself. The same scan reads every file under
`agent/` and finds no such call. APAI calls `disconnect()` and `stop()` on its arm when it
shuts down, after a failed motion and at the end of its rest tool. On the arm adapter
`stop()` sends at most a STOP, which is a hold. `disconnect()` sends nothing outside a
task; inside a task with the motors on, where the rest tool calls it, it sends
OWNER_REQUEST "motors_off", and the robot app switches the motors off only with the arm
measured at rest (`test_lifecycle_conform.py`). `test_no_limp.py` ends an agent move in
each way the agent side knows
(a setpoint refused by the check ahead, a latched workstation checker, Ctrl+C, a STOP at the
Thor terminal, a lost link, the agent program gone) and each time finds the arm holding away
from rest with its motors on.

The operator's programs cannot cut torque either. RELEASE and SHUTDOWN are sent from one
function, `send_command()` of the console command, which the operator console calls too,
after the typed confirmation; the robot app still applies its own checks to them. The scan
reads every file under `console/` and `viewer_live/`: none cuts torque and none names
RELEASE or SHUTDOWN. The experiment's viewer sends VIEW_STATUS only
(`test_viewer_live.py`).

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
is cut only by a shutdown that first brought the arm to measured rest. The robot app of
this sweep is built without the way to switch the motors on again, and such a robot app
ends a typed `rest` at rest with the motors on; the `rest` that switches them off, the
motors on again and SIGTERM are tested in `test_lifecycle_conform.py`.
Among the states is also a robot app whose torque release raised in the driver (motion
state `release_failed`): there no command sends the arm a joint target, a jaw command or
a hold. Among the commands is the controller's trigger, pulled to its end and let go again
in steps: where a person owns the arm the jaws follow it. The seeded random sequences run
on a fake clock that moves on only as fast as the robot app's dispatcher follows, so each
runs the same way every time (`test_a_seeded_sequence_runs_the_same_way_every_time`); four
of them let a torque release raise and two of those also move the trigger.

## Set up and start the rig

This section takes a person who has never run this example from two machines to both
machines running, in order. [First run on the rig](#first-run-on-the-rig) is the checklist
of the session itself and refers back to the steps here.

The first session on the rig, on 2026-11-16, was run with this README as its only guide.
Each place where that session lost time is a step below, and the lines the programs
printed then are in the
[troubleshooting table](#troubleshooting-what-a-program-printed-and-what-to-do) at the
end. The commands of steps 1 to 3 were written after that session and were not run again
as they stand here.

Four placeholders stand for what differs from rig to rig:

- `<workstation address>` and `<Thor address>`: the two machines' addresses on the network
  that connects them.
- `<experiment folder>`: the experiment folder on the machine the command runs on. Each
  machine has its own copy.
- `<camera streamer folder>`: the camera streamer's example folder on that machine, the
  one `AGENTIC_REMOTE_CAMERA_PR` names on the workstation.
- `<checkout>`: the checkout of Isaac Capture on that machine. The example is
  `<checkout>/examples/agentic_remote`, and the commands below run in that folder unless
  they say otherwise.

### 1. The same commit on both machines

**Both machines must run one commit of this repository.** Nothing checks it. Every build
says protocol version 1, and a program drops a field of a link message that it does not
know. So a mix shows only as wrong behaviour:

- A robot app older than the workstation's programs ignores the expiry of a GRIPPER, and
  counts a sample age and a view age without the way to the Thor.
- A robot app newer than the viewer's launcher refuses every takeover (`the live view is not
  fresh: the VIEW_STATUS does not say when it was sent; ...`).

The Thor has no access to the repository's server, so the code is carried as a
**git bundle**: one file that holds commits and that git can clone and pull from.

```bash
# Workstation, in the checkout the programs are started from.
git status --short            # only the edits of step 5 may show
git branch --show-current     # the branch: <branch> below
git rev-parse HEAD            # the commit id: write it down
git bundle create /tmp/isaac_capture.bundle HEAD <branch>

# Carry /tmp/isaac_capture.bundle to the Thor, for example with scp in your own ssh
# session. Then on the Thor, the first time:
GIT_LFS_SKIP_SMUDGE=1 git clone -b <branch> isaac_capture.bundle IsaacCapture
git -C IsaacCapture rev-parse HEAD     # must print the workstation's commit id

# On the Thor, later: a newer bundle into the same clone.
GIT_LFS_SKIP_SMUDGE=1 git -C IsaacCapture pull --ff-only /absolute/path/isaac_capture.bundle <branch>
```

- **A bundle holds commits only.** What is changed and not committed on the workstation
  does not reach the Thor: commit first. The edits of step 5 are the exception: they are
  made again on each machine.
- **`GIT_LFS_SKIP_SMUDGE=1` is needed.** This repository keeps its pictures and meshes in
  Git LFS, a store beside the repository (`.gitattributes` lists the file types). A bundle
  holds none of those files and the Thor cannot fetch them, so without the variable the
  clone fails at its checkout with an error that names the `lfs` smudge filter. With it
  those files stay small pointer files. No file under `examples/agentic_remote` is kept
  in Git LFS.
- **Write both commit ids into the session record.** After every change of the code,
  compare `git rev-parse HEAD` on both machines again before a program is started.
- APAI, the experiment folder and the camera streamer are not part of the bundle. Each
  machine has its own copy of each.

### 2. Build Isaac Capture on the Thor

The robot app and the human program need Isaac Capture's Python package, built on the
Thor for the Python they run under. In the checkout on the Thor:

```bash
cmake -B build -DISAAC_TELEOP_PYTHON_VERSION=3.12 -DBUILD_VIZ=OFF -DBUILD_TESTING=OFF \
    -DENABLE_CLANG_FORMAT_CHECK=OFF
cmake --build build --parallel
```

| Setting | What it does |
|---|---|
| `-DISAAC_TELEOP_PYTHON_VERSION=3.12` | Builds the Python package for Python 3.12, the version of APAI's environment on the Thor. Without it the build is for 3.11 |
| `-DBUILD_VIZ=OFF` | Leaves out Televiz, Isaac Capture's own viewer, which no program of this example uses. Where the build would include it, configure stops when the program `glslangValidator` is not installed |
| `-DBUILD_TESTING=OFF` | Leaves out Isaac Capture's unit tests, which the Thor does not run |
| `-DENABLE_CLANG_FORMAT_CHECK=OFF` | Switches off the check of the code's formatting, which is on by default on Linux. With it on, configure stops when `clang-format` is not installed |

The built package is the folder `<checkout>/build/python_package/Release`. That is the
value of `AGENTIC_REMOTE_ISAAC_CAPTURE_PY` on the Thor. The workstation has its own
build, made for the Python of the workstation environment
(see [The workstation environment](#the-workstation-environment)).

### 3. The variables of each machine, in one file

Every start script reads its settings from environment variables, and stops with exit
code 2 and one line when one that it needs is missing. Keep them in one file per machine,
outside the checkout, so that they are not typed in every terminal.

**On the workstation** `run_workstation.sh --env FILE` reads the file itself. For a
program started by hand, read the file into the terminal first:

```bash
# <environment folder>/workstation.env: KEY=value lines, absolute paths.
AGENTIC_REMOTE_AGENT_PYTHON=<environment folder>/venv/bin/python
AGENTIC_REMOTE_ENV_DIR=<environment folder>
AGENTIC_REMOTE_ISAAC_CAPTURE_PY=<checkout>/build/python_package/Release
AGENTIC_REMOTE_APAI_SRC=<APAI checkout>/src
AGENTIC_REMOTE_EXPERIMENT_DIR=<experiment folder>
AGENTIC_REMOTE_CAMERA_PR=<camera streamer folder>
# Optional: the address the experiment's viewer listens on, without a token.
# AGENTIC_REMOTE_VIEWER_HOST=0.0.0.0
```

`AGENTIC_REMOTE_VIEWER_HOST` is optional, and only `run_workstation.sh` reads it. It
names the address the experiment's viewer listens on: the workstation's address on the
internal network, or `0.0.0.0` for every interface. With it anyone who can reach that
address can see the camera pictures, the maps and the timings of the runs, without a
token. Without it the viewer listens on `127.0.0.1` only and asks for its token.

```bash
set -a; source <environment folder>/workstation.env; set +a
./run_agent.sh --config configs/agent_rebot.yaml --check
```

`set -a` makes the shell hand every variable of the file on to the programs it starts.
`<environment folder>` is the folder `setup_workstation_env.sh` built. What each of the
six variables names is under [The start scripts](#the-start-scripts). Write absolute
paths: the panes of `run_workstation.sh` start in the example's folder, not in the
folder the script was called from.

**On the Thor** `run_robot_app.sh --env FILE` and `run_human.sh --env FILE` read the file
themselves, as `run_workstation.sh` does. `--env FILE` must be the first argument, and
the program the script starts does not get it. FILE is a shell file of `KEY=value` lines:

```bash
# ~/agentic_remote_thor.env, on the Thor: KEY=value lines, absolute paths.
AGENTIC_REMOTE_THOR_PYTHON=<experiment folder>/third_party/apai/.venv/bin/python
AGENTIC_REMOTE_APAI_SRC=<experiment folder>/third_party/apai/src
AGENTIC_REMOTE_ISAAC_CAPTURE_PY=<checkout>/build/python_package/Release
AGENTIC_REMOTE_HUMAN_PYTHON=<experiment folder>/third_party/apai/.venv/bin/python
```

```bash
./run_robot_app.sh --env ~/agentic_remote_thor.env --config configs/robot_app_rebot.yaml ...
./run_human.sh --env ~/agentic_remote_thor.env --config configs/human_rebot.yaml
```

Without `--env` each script takes its variables from the environment, as before. An
`--env` that names no file ends the script with exit code 2 and one line, for example
`run_robot_app.sh: --env needs a file of KEY=value lines (it is '...')`, and nothing is
started. The commands of this README that start the two scripts are written without
`--env`: put it in front of their arguments, or read the file into the terminal first
(`set -a; source ~/agentic_remote_thor.env; set +a`).

| Variable | What it names |
|---|---|
| `AGENTIC_REMOTE_THOR_PYTHON` | The Python of APAI's environment on the Thor. The experiment's own setup makes that environment in `third_party/apai/.venv` of the experiment folder |
| `AGENTIC_REMOTE_APAI_SRC` | The `src` folder of the APAI checkout on the Thor |
| `AGENTIC_REMOTE_ISAAC_CAPTURE_PY` | The folder of the Isaac Capture Python package built in step 2 |
| `AGENTIC_REMOTE_HUMAN_PYTHON` | The Python the human program runs under, for demo 2. `run_human.sh` does not read `AGENTIC_REMOTE_THOR_PYTHON`, so the file names the same Python once more |

### 4. The model client on the workstation

With the full stack (`experiment.stack: full`, as in `configs/agent_rebot.yaml`) the agent
asks a cloud model through the **model client**, the Codex command-line program. Two
parts of the agent program start it: the agent itself and its **detector**, the part
that finds objects in a camera frame. Both take the program that `experiment.codex_bin`
of the agent's file names.

- **The client must be the one for the workstation's processor.** `configs/agent_rebot.yaml`
  names `${AGENTIC_REMOTE_ENV_DIR}/codex/node_modules/.bin/codex`: the client in the
  workstation environment. `./setup_workstation_env.sh --with-model-client` installs it
  there, at the version the script pins, when it builds the environment. The script
  refuses a folder whose `venv/` exists, so for an environment that is built already run
  its step 4 by hand, with npm:

  ```bash
  npm install --prefix "$AGENTIC_REMOTE_ENV_DIR/codex" --no-audit --no-fund @openai/codex@0.167.0
  ```

- **The experiment folder's own client does not run on the workstation.** With an empty
  `experiment.codex_bin`, as in `configs/agent.yaml`, the program takes the client the
  experiment's own file names, inside the experiment folder. The workstation's experiment
  folder is a copy of the Thor's, and its client was installed for the Thor's processor:
  on the workstation it ends with `Missing optional dependency @openai/codex-linux-x64`.
- **A path that names no program is refused** before anything is attached: exit code 2
  and `experiment.codex_bin names no program that can be run: ...`.
- **The login is the user's own.** The client asks the model with the login of the user
  who runs it. Log in once with the client's own command; no script of this example reads
  or writes that login.

### 5. The edits each machine needs, which are not committed

Some values belong to one rig and not to the repository: an address, or the name of
today's calibration. Each is written into a shipped file on the machine that reads it.
`git status` then shows the file as modified. Do not commit it.

**Make each edit in the checkout the programs are started from.** A second checkout on
the same machine has its own files, and an edit made there changes nothing for a
program started here.

| Machine | File | Key | What to write |
|---|---|---|---|
| Workstation | `configs/link_service_rebot.yaml` | `allowed_peers` | `[<Thor address>]`: the address the robot app dials from. The shipped list is empty, and the robot link service refuses an empty list: exit code 2 and `robot link service: the configuration cannot be used: allowed_peers must be a non-empty list, ...`. The file listens on every address of the workstation (`listen.address: 0.0.0.0`), so this list is all that decides who may be the robot |
| Workstation | `configs/agent_rebot.yaml` | `camera.hand_eye_json` | Only after a new calibration of the camera's mounting: step 6 |
| Thor | `configs/robot_app_rebot.yaml` | `camera_calibration_id` | Only after a new calibration: step 6 |
| Thor | `configs/human_rebot.yaml` | `teleop.endpoint` | For demo 2 only: `<workstation address>:47600` in place of the placeholder (see [The human program on the Thor](#the-human-program-on-the-thor)) |

A pull of a newer bundle keeps these edits, unless the new commit changes the same file.
Git then stops and names the file: note your value, take the file back
(`git checkout -- <file>`), pull, and make the edit again.

### 6. The camera calibration and its name

The agent program turns what the camera sees into places in the arm's coordinate system.
For that it reads two things on the workstation: the stereo calibration folder
(`camera.calibration_dir` of `configs/agent_rebot.yaml`) and the **hand-eye file**, the
record of how the camera is mounted relative to the arm (`camera.hand_eye_json` of the
same file). Both keys name the workstation's copies, inside the workstation's experiment
folder. The calibration itself is made on the Thor, with the experiment's own programs.

The two machines agree on the calibration by its **name**: `stereo-<12 hex>+mount-<12 hex>`,
made from the SHA-256 of the stereo calibration's `manifest.json` and of the hand-eye
file. After every new calibration on the Thor:

1. **Copy the new hand-eye file to the workstation**, to the same place inside the
   workstation's experiment folder that it has inside the Thor's. Write that path into
   `camera.hand_eye_json` of `configs/agent_rebot.yaml`; the comment above the key says
   which fit the shipped file names. The other way is the experiment launcher's own: with
   `camera.hand_eye_json` empty the agent program takes the hand-eye file of the
   experiment's newest **mounting record**, the file the experiment writes when a fit of
   the camera's place was accepted. It looks in `.state/thor/mounting` of the
   workstation's experiment folder, at `latest.json` and at `<fit>/summary-thor.json` of
   a Thor's export unpacked there, and takes the record that was checked last. A record
   counts only when its `accepted_fit` is true and its hand-eye file, as it lies on the
   workstation, has the SHA-256 the record names: otherwise the program ends with exit
   code 2 and `the mounting record ... is not accepted or its hand-eye file ... changed`.
   `--mounting PATH` names one record for one run. The program's first lines say which
   hand-eye file it took and from where.
2. **Ask the workstation for the name**:

   ```bash
   ./run_agent.sh --config configs/agent_rebot.yaml --check
   ```

   It attaches to nothing and prints JSON. `"calibration_id"` in it is the name.
3. **Give the Thor the same name, twice**: as `camera_calibration_id` in
   `configs/robot_app_rebot.yaml` on the Thor, and as `--calibration NAME` on the robot
   app's command line. The robot app asks for both so that the name is typed by a person
   who knows which calibration the rig is in.

What a wrong name gives:

- **On the Thor**, a `--calibration` that is not the file's `camera_calibration_id` ends
  the robot app with exit code 2, before anything is connected: `--calibration names
  '...' but camera_calibration_id of the file is '...'; the two must be the same name,
  and not empty`.
- **On the workstation**, a robot app that names another calibration ends the agent
  program with exit code 1, before any task: `the robot app's camera calibration is
  '...', this program's '...'`.

### 7. Start the workstation's programs

**The robot link service comes first.** Every other program of the workstation attaches
to its socket.

- **The operator console** started before the service prints `cannot attach to the robot
  link service: ... not attached: stop (written raw, not confirmed), attach, help and
  quit work`. It tries again every 2 s, 30 times. After that minute it prints `no further
  attempt by itself; type attach to try again` and stays not attached until `attach` is
  typed.
- **The agent program** started before the service ends with exit code 1
  and `cannot attach to the robot link service ...`. Start it again.
- **The experiment's viewer** started before the service serves its page all the same,
  says `viewer: live view: not reported to the robot: cannot attach to the robot link
  service: ...` once, and attaches by itself once the service runs.

`bash run_workstation.sh --env FILE --thor <Thor address>` starts the five programs of
the table below in one tmux window, the service first. The viewer's page is the live
view, which a person's takeover needs: open it in a browser and keep its tab visible,
with the live picture showing.
[The start scripts](#the-start-scripts) says which pane is which, how to copy text out
of a pane and how to leave. Started by hand, one terminal each, the order is:

| # | Program | Command | It works when it prints |
|---|---|---|---|
| 1 | Robot link service | `PYTHONPATH=python:$AGENTIC_REMOTE_ISAAC_CAPTURE_PY "$AGENTIC_REMOTE_AGENT_PYTHON" -m isaaccapture_examples.agentic_remote.link_service --config configs/link_service_rebot.yaml` | `robot link service: waits for the robot app on 0.0.0.0:47610 (taken from <Thor address>); local API at <socket path>` |
| 2 | Operator console | `./run_console.sh` | `attached to the robot link service: ...`, then a status line with `robot connection lost (not connected)` until the robot app of step 9 has dialed |
| 3 | Camera receiver | `./run_camera_receiver.sh --host <Thor address>` | See step 8 |
| 4 | The experiment's viewer, whose page is the live view | `./run_viewer.sh` | Its address and how to open the page with its token (see [The experiment's viewer](#the-experiments-viewer)); `viewer: live view: attached to the robot link service at <socket>; ...`. Open the page and keep its tab visible, with the live picture showing: a person's takeover needs it |
| 5 | Agent program | `./run_agent.sh --config configs/agent_rebot.yaml --interactive` | Only once the console shows the robot connection up: `[agent] robot connection up: ... driver real ...`, `[agent] obstacle grid: live ...` and the prompt `task> ` |

**The address of the viewer's page** is in the line `viewer: ...` that
`run_workstation.sh` prints. Without `AGENTIC_REMOTE_VIEWER_HOST` in the workstation's file, open
`http://127.0.0.1:8766/?token=<the token>` in a browser on the workstation, with the
text of `.state/viewer.token` as the token. With the variable, open
`http://<that address>:8766/` in a browser on any machine that can reach that address:
the page asks for no token. With `0.0.0.0`, open it under the workstation's address.

### 8. The camera, from the Thor to the workstation

The camera sender runs on the Thor and listens. The camera receiver runs on the
workstation and dials the sender.

**On the Thor, before the sender starts:**

```bash
sudo bash <experiment folder>/thor_camera_bringup.sh
sudo systemctl stop sensing-camera-argus.service
<camera streamer folder>/argus_sender/run.sh \
    --config <checkout>/examples/agentic_remote/configs/camera_sender_rebot.yaml
```

1. **`thor_camera_bringup.sh`** is the rig's camera bring-up script, in the experiment
   folder. It needs sudo, takes no arguments and is run once after every boot of the
   Thor. It loads the camera's kernel modules, sets the 60 Hz trigger of the two eyes and
   starts `sensing-camera-argus.service`, the rig's own camera service. Its last line is
   `Camera drivers, 60 Hz trigger and Argus are ready for a live capture test.`
2. **`sensing-camera-argus.service` must be stopped again** before the sender starts. The
   sender opens the cameras inside its own process. While that service holds them the
   sender finds none and ends with `No camera devices are available`.
3. **The sender's `run.sh`** stops the system's own camera service,
   `nvargus-daemon.service`, by itself; for that it needs sudo without a password.

**On the workstation:**

```bash
./run_camera_receiver.sh --host <Thor address>
```

**`--host` is the Thor's address, not the workstation's own.** The receiver's file names
`127.0.0.1` as the sender, so without `--host`, or with the workstation's address, the
receiver dials the workstation itself and prints `waiting for AVS1 sender
127.0.0.1:5000 (...)` again and again.

What each side prints when it works, as the camera streamer's code has it:

| Side | Line |
|---|---|
| Sender, on the Thor | `Listening for TCP receiver on 0.0.0.0:5000`, then `Receiver connected from <workstation address>:<port>` |
| Receiver, on the workstation | `camera_receiver running: sender=<Thor address>:5000, socket=/tmp/camera_receiver_avs1_cuda_ipc.sock`, then `connected to AVS1 sender <Thor address>:5000`, then one line of statistics every 10 s |

Then the agent program prints `[agent] obstacle grid: live ...`. The viewer's page
shows the camera's live picture, and the viewer prints `viewer: live picture: the
camera's pictures arrive`.

### 9. Start the robot app on the Thor

In a terminal of the Thor, in the example's folder:

```bash
./run_robot_app.sh --env ~/agentic_remote_thor.env \
    --config configs/robot_app_rebot.yaml --arm apai --driver real \
    --apai-profile <experiment folder>/configs/thor_operation --calibration NAME \
    --connect <workstation address>:47610
```

| Argument | What it does |
|---|---|
| `--env ~/agentic_remote_thor.env` | The file of step 3 with the script's three variables; it must come first. Left out when the variables are set in the terminal |
| `--config configs/robot_app_rebot.yaml` | The robot app's file for the rig |
| `--arm apai` | APAI's checkers and APAI's motor driver, in place of the simulated arm |
| `--driver real` | The driver of the real motors. The robot app connects it only after its checks and three typed `yes` (see [The real driver](#the-real-driver)): the connect cuts motor torque for a moment, so the arm must lie at rest |
| `--apai-profile` | The rig's APAI arm profile: the path of `configs/thor_operation.yaml` in the Thor's experiment folder, without `.yaml` |
| `--calibration NAME` | The calibration's name of step 6 |
| `--connect` | Where the robot link service listens: the workstation's address and port 47610 |

**A robot app on the simulated arm takes no task of the rig's agent, on purpose.** The
checklist starts the robot app once with `--arm sim` (its step 10) to prove the link
between the machines without the arm. Such a robot app declares the driver `sim` and the
profile `sim-6dof`, and names no camera calibration. The agent program on
`configs/agent_rebot.yaml` expects the rig's profile and the rig's calibration. It ends
with exit code 1 before any task, with `the robot app's profile is 'sim-6dof', expected
'rebot-b601-rs-thor'`: what the agent plans for the real arm must not be sent to an arm
that is not there. For a task, start the robot app with the command above.

**The speeds of the robot app's file must fit the Thor's copy of the experiment folder.**
With this profile the robot app reads the experiment launcher's joint speed from
`configs/rebot_workstation.yaml` of the Thor's experiment folder and takes it as the cap
of the agent's and the recovery limit set. The rig file asks for 0.2 rad/s there; with a
smaller speed in that file the program ends before anything is connected, with
`limits.agent.max_joint_speed 0.2 rad/s is above the APAI profile's speed cap of ...
rad/s (safety.max_joint_vel)`. `thor.agent_max_joint_speed` of the workstation's
`configs/agent_rebot.yaml` must be the same 0.2 (see [The rig files](#the-rig-files)).

**After a `rest` the robot app runs on with the motors off.** `rest` at the console or
at the Thor terminal, and the agent program's question at its end, return the arm to
rest, close the jaws and switch the motors off. Type `on`, then `yes`, at the Thor
terminal or at the console to switch them on again; the agent program asks for the same
before its next skill that moves. The robot app is started again only after `quit`, a
`shutdown` or a SIGTERM.

The Thor's programs of demo 2 (the human program, `--human-socket`) are under
[The human program on the Thor](#the-human-program-on-the-thor).

### 10. After a run: the viewer and the run's files

Each run of the agent program writes a run folder, `.state/runs/<UTC time>-<process id>/`
in the example's folder with the shipped files.

- **Seven files are written in every run**: six of measurements, `moves.jsonl`,
  `jaw-commands.jsonl`, `robot-state.jsonl`, `camera-deliveries.jsonl`,
  `link-latency.jsonl` and `latency-summary.txt`, and `move-paths.jsonl`, with every
  joint setpoint the agent planned for each move and how many of them the robot app
  sent to the motors.
  [What a run records and measures](#what-a-run-records-and-measures) says what each
  holds, and what a run's record can and cannot be used for afterwards.
- **`./run_viewer.sh` starts the experiment's viewer**, the web page of the rig's earlier
  one-machine program, on those run folders. It lists a run only when the run folder
  holds `debug/events.jsonl` and `resolved-config.yaml`. The first is there only when the
  agent program was started with `--record-debug`. The second is written by the agent
  program of this commit at its start; a run recorded by an older checkout has none and
  is not listed. See [The experiment's viewer](#the-experiments-viewer).
- **Which address to open.** Without `AGENTIC_REMOTE_VIEWER_HOST` the page is
  `http://127.0.0.1:8766/?token=<the token>`, in a browser on the workstation; the token
  is the text of `.state/viewer.token`. With the variable in the workstation's file,
  `run_workstation.sh` starts the viewer on that address and without a token: the page
  is `http://<that address>:8766/`, in a browser on any machine that can reach it. With
  `0.0.0.0`, use the workstation's address.
- **The page's picture is live.** While **Follow live** is active the picture on the
  page is the camera's picture as it arrives, up to 30 a second; it is shown and not
  stored. The button over it switches: **Map picture** shows the stored picture of the
  obstacle map, with the arm's mask and the markers of a refusal, and **Live picture**
  goes back. A replay always shows the stored picture. `no live picture` on the button
  means that the viewer gets no camera pictures: the camera receiver does not run or
  no camera frames arrive; the text under it and the pane `viewer` say why. That text
  is in the warning colour whenever the picture is not live or does not count. While
  the viewer does not run the page keeps its last picture, dimmed, and says how old it
  is; after a restart of the viewer the page loads itself again. A page that says
  `watch only: ...; reload this page to make it the live view` is reloaded by hand.
  `./run_viewer.sh --no-live` starts the viewer without the live picture. See
  [The experiment's viewer](#the-experiments-viewer). The page is the live view only
  while the live picture shows: after a look at a replay or at the map picture, press
  **Live picture** again before a person takes the arm over (see
  [The live view](#the-live-view)).
- **The page's arm is live as well.** While **Follow live** is active the arm in the
  3D view is the arm of each ROBOT_STATE as it arrives, 20 a second, also with a
  recording that has stopped and with no recording at all. The line under the 3D
  view says `arm: live, N a second; map: M s old`: the obstacle map beside the arm
  is the recording's last and can be older than the arm. `arm: recorded` stands
  there in a replay, and `arm: no live state: <why>` when the viewer gets no
  ROBOT_STATE. With the motors off the robot app reads no joint: the line then says
  `arm: last read N s ago (torque off: the joints are not read)`, and the arm drawn
  is where the joints were last read, not where they are now.
  `./run_viewer.sh --no-live` starts the viewer without the live arm
  too.
- **The debug recording stops at 16 GiB only with `--debug-at-limit stop`.** The pane
  `agent` of `run_workstation.sh` starts the agent program with `--record-debug`. The
  recording's folder is kept under `--debug-max-gib`, 16 GiB by default. It stores 2
  pictures of the obstacle map a second, which is what the viewer's page can show, and
  everything the arm did or was refused; with that the limit is reached after about
  30 min, and the agent's line `debug recording:` says so at the start. From then on
  the oldest map pictures are dropped to make room, so the viewer's page keeps
  following for as long as the session lasts. The agent terminal says it, once at that
  moment and once more with the next task prompt, in a line that begins `debug
  recording reached --debug-max-gib 16 after N min: from now on the oldest map
  pictures are dropped; the viewer keeps following`. The line also names the time
  before which the page shows no camera picture; that time is about 30 min back and
  moves on with the run. Moved back before it, the page shows the obstacle map and the
  arm without the camera picture. What the arm did or was refused is never dropped,
  and neither is the map that a refusal or a route segment names.
  In the sessions of 2026-11-16 and 2026-11-17 the recording stored every map picture,
  about 7 a second, stopped at its limit after about ten minutes, and the page stood
  still from then on. `--debug-at-limit stop` on the agent's command asks for that
  stop; the line is then `debug recording full after N min (--debug-max-gib 16): the
  viewer no longer updates; the arm and the task are not affected`, and a new
  recording needs the session ended (Ctrl+D at the task prompt) and the agent started
  again. For a longer stretch of pictures add a larger limit to the agent's command
  before you press Enter in its pane, for example `--debug-max-gib 64` (about two
  hours), and see that the disk has the room: whatever the option, the recorder stops
  with less than 2 GiB free. `--debug-map-hz HZ` sets the map pictures a second;
  `--debug-map-hz 0` stores every one, and 16 GiB then holds about 9 min. While the
  agent waits for a task at its prompt and nobody owns or moves the arm, the recording
  stores one map picture every 10 s and not 2 a second, so a session that stands idle
  does not fill it; a person who takes the arm over in that wait is recorded at 2 a
  second. `--debug-idle-map-s SECONDS` sets that time, and 0 stores none then. What is
  stored whatever the rate, and what is never dropped, is told under
  [The options of the experiment's launcher](#the-options-of-the-experiments-launcher).
- **Old runs are deleted by hand.** Each run with `--record-debug` takes up to its
  limit, about 17 GB with the default, in `.state/runs/<run>/debug`; a session of half
  an hour or more takes all of it, and keeps it. No program removes a run folder.
  Delete the ones you no longer need, while no agent program and no viewer reads them.

### 11. The clocks of the two machines, once per machine

The programs read each machine's system clock. **chrony** is the system service that keeps
that clock right: it starts at boot and runs in the background. It is set up once per
machine, not once per session, and no program of this example starts or calls it. The
workstation serves its time to the Thor, and the Thor follows the workstation, as the
camera streamer's own deployment files arrange it
(`<camera streamer folder>/argus_sender/deploy/camera-chrony-host.conf.in` and
`camera-chrony-agx.conf.in`).

Do this while nothing records and the arm is at rest: the Thor's clock may jump once.
The commands were not run on the rig as they stand here.

On the workstation, with one `allow` line for each Thor that follows it:

```bash
printf 'allow <Thor address>\nlocal stratum 10\n' | sudo tee /etc/chrony/conf.d/robot-time-server.conf
sudo systemctl restart chrony
```

On the Thor:

```bash
sudo apt install -y chrony
printf 'server <workstation address> iburst minpoll 2 maxpoll 4 prefer\nmakestep 1.0 3\n' | sudo tee /etc/chrony/conf.d/robot-time-client.conf
sudo systemctl restart chrony
sleep 10; chronyc sources -v; chronyc tracking
```

- **Good**: in `chronyc sources -v` the line of `<workstation address>` begins with `^*`,
  and `chronyc tracking` shows a `System time` well below a millisecond.
- **Blocked**: that line begins with `^?`. The time requests (UDP port 123) do not get
  through between the two machines, or a firewall on the workstation stops them
  (`sudo ufw status` shows whether one is on).
- A Thor that is moved to another workstation gets that workstation's address in its
  `server` line, and that workstation gets the `allow` line.
- `timedatectl` on either machine shows `System clock synchronized: yes` once chrony has
  a source.

### Troubleshooting: what a program printed, and what to do

Each line below was met in the first session on the rig, is what the code prints for a
mistake that session made, or is one of the lines the programs print since the changes of
2026-11-16 (the motors off after a `rest`, the question about the jaws, the speeds).

| The line | What it means | What to do |
|---|---|---|
| git, at a clone on the Thor: an error that names `smudge filter lfs` | Git LFS tried to fetch pictures and meshes that the bundle does not hold | Clone again with `GIT_LFS_SKIP_SMUDGE=1` in front (step 1) |
| cmake, on the Thor: a list of missing tools with `glslangValidator (BUILD_VIZ=ON -- compiles Televiz shaders to SPIR-V)` | The build would include Isaac Capture's viewer, which needs a tool the Thor does not have | Configure with `-DBUILD_VIZ=OFF` (step 2) |
| cmake, on the Thor: `clang-format not found but ENABLE_CLANG_FORMAT_CHECK is ON` | The format check is on by default on Linux and needs `clang-format` | Configure with `-DENABLE_CLANG_FORMAT_CHECK=OFF` (step 2) |
| `run_robot_app.sh: AGENTIC_REMOTE_THOR_PYTHON must name the Python of APAI's environment on the Thor (it is '')` | The variables are not set in this terminal, and the script was started without `--env` | Start it with `--env ~/agentic_remote_thor.env` as its first argument (step 3) |
| `robot link service: the configuration cannot be used: allowed_peers must be a non-empty list, ...` | The service's file names no address it may take the robot connection from | Write `[<Thor address>]` into `allowed_peers` of `configs/link_service_rebot.yaml`, in the checkout the service is started from (step 5) |
| Service: `refused a robot connection from <address>: not an allowed address`. Thor: `robot app: no robot connection: ...; dialing again every 1 s` | The robot app dials from an address that is not in `allowed_peers` | Write the address the service names into the list and start the service again. The robot app goes on dialing by itself |
| Thor: `robot app: no robot connection: ...; dialing again every 1 s`, and the service prints nothing | Nothing answers at that address and port: the service does not run, `--connect` names another address, or the workstation's firewall keeps the Thor from port 47610 | Start the service (step 7), compare the address, open the port. The robot app goes on dialing by itself |
| Console: `cannot attach to the robot link service: ...`, later `no further attempt by itself; type attach to try again` | The console was started before the robot link service and has given up after a minute | Start the service, then type `attach` at the console (step 7) |
| Console: `[refused] owner request 'human' refused: the live view is not fresh: ...; show the viewer's live picture in a visible browser tab`, or `view none` in the status line | The viewer's page does not show its live picture in a visible browser tab; or the viewer is not attached to the robot link service (`no VIEW_STATUS has arrived`) | Open the viewer's page, press `Live picture`, keep the tab in front, and read the note under the button: in the warning colour it says why the picture does not count. Reload a page whose note ends with `reload this page to make it the live view`. See [The live view](#the-live-view) |
| Viewer's page: the live picture stands still, dimmed, and the note under the button says `the viewer cannot be reached; trying again every second; this picture is N s old` | The viewer does not run, or the browser does not reach it | Start the viewer (step 7). The page asks again every second and loads itself again once the new viewer answers: the live picture shows about a second later, and the view is live 3 s after the viewer's first ROBOT_STATE. Nothing is done at the page |
| Viewer's page: `watch only: another page on the workstation is the live view; reload this page to make it the live view`, or `watch only: no page is the live view now; ...` | The page was loaded again on the workstation, in another tab or by a program (a `curl` of the page is a load): that load counts, and this page only shows the picture | Reload the page a person looks at. It is then the live view, and every other page on the workstation says `watch only: ...` |
| Viewer: `viewer: live view: not reported to the robot: cannot attach to the robot link service: ...` | The robot link service does not run, or runs on another socket | Start the service; the viewer attaches by itself within a minute and says `viewer: live view: attached ...`. `--link-socket PATH` names another socket |
| Viewer: `viewer: no live picture: ...` | `AGENTIC_REMOTE_CAMERA_PR` is not set or names no camera consumer's Python, the camera receiver does not run, or no camera frames arrive | Set the variable (step 3); start the receiver (step 8). The viewer tries again by itself |
| Camera sender: `No camera devices are available` | The cameras are not set up since the last boot, or another camera service holds them | Run the bring-up script, stop `sensing-camera-argus.service`, start the sender again (step 8) |
| Camera receiver: `waiting for AVS1 sender 127.0.0.1:5000 (...)` | The receiver dials the workstation itself | Start it with `--host <Thor address>` (step 8) |
| Camera receiver: `waiting for AVS1 sender <Thor address>:5000 (...)` | The sender does not run, or does not listen yet | Start the sender on the Thor (step 8) |
| Agent program or its detector: `Missing optional dependency @openai/codex-linux-x64` | The model client that was started is the experiment folder's own, installed for the Thor's processor | Install the client in the workstation environment and name it in `experiment.codex_bin` (step 4) |
| Agent program: `the robot app's profile is 'sim-6dof', expected 'rebot-b601-rs-thor'` | The robot app on the Thor runs the simulated arm | Start the robot app on the real arm (step 9) |
| Agent program: `the robot app's camera calibration is '...', this program's '...'` | The two machines name different calibrations | Step 6 |
| Thor: `--calibration names '...' but camera_calibration_id of the file is '...'; ...` | The name on the command line is not the one in the robot app's file | Step 6 |
| The experiment's viewer lists no run | The run was started without `--record-debug`, or by a checkout that does not write `resolved-config.yaml` | Add `--record-debug` to the agent's command; run both machines on this commit (step 10) |
| The agent terminal said `debug recording reached --debug-max-gib 16 after N min: from now on the oldest map pictures are dropped; ...` | The debug recording reached its size limit, 16 GiB by default, after about 30 min with the default 2 map pictures a second. Nothing stopped: its oldest map pictures are dropped and the viewer keeps following | Nothing. For more than the last half hour of camera pictures in the viewer, start the next run with a larger `--debug-max-gib`; delete old run folders by hand (step 10) |
| The experiment's viewer shows no camera picture at an earlier time of a run, or the text `No such file or directory` under the refusals | That time lies before the stretch the debug recording still holds the pictures of, or the map's points; `dropped_at_limit` in the run's `debug/recording-status.json` names where each begins, in seconds on the page's time line | Move the time line forward, or press `Follow live` |
| The experiment's viewer shows a run and stops updating; the agent terminal said `debug recording full after N min (--debug-max-gib 16): the viewer no longer updates; ...` | The run was started with `--debug-at-limit stop` and the debug recording stopped at its size limit, or nothing was left in it that may be dropped; the run goes on | End the session and start the agent again for a new recording, without `--debug-at-limit stop` or with a larger `--debug-max-gib`; delete old run folders by hand (step 10) |
| The agent terminal said `debug recording stopped after N min (...)` | The experiment's recorder stopped for the reason in the brackets: with `less than 2 GiB free` the disk is full; the run goes on | Delete old run folders by hand (step 10), then end the session and start the agent again |
| Console or agent program: `the motors are off; switch them on again: type on at the Thor terminal or run the console command on` | A `rest` or a `release` switched the motors off at rest | `on`, then `yes`, at the Thor terminal or the console |
| `... was refused because the jaws are not proven empty. Look at the jaws. ...` | A `rest`, a `release`, a shutdown or a hand-back was refused only for the jaws | Look at the jaws. Empty: type `jaws are empty`, then answer the command's own question. Holding something: `stop`, `open`, `yes` |
| Thor, at the start: `limits.agent.max_joint_speed ... rad/s is above the APAI profile's speed cap of ... rad/s (safety.max_joint_vel)` | The Thor's copy of the experiment's `configs/rebot_workstation.yaml` holds a lower joint speed than the robot app's file | Bring the two together, and `thor.agent_max_joint_speed` on the workstation with them (step 9) |
| Agent program, once the robot connection is up: `joint speed: the agent plans at ... rad/s, the lowest of ... (--speed-profile reference) and ... (thor.agent_max_joint_speed); ...` | The speed the agent plans its moves at: the lowest of the experiment launcher's speed, the agent file's key and the Thor's own limit. With the rig files it is the rig owner's 0.2 rad/s | Nothing, when that speed is wanted. For another, change the keys on both machines ([The rig files](#the-rig-files)) |
| Agent program, at its start: `WARNING: APAI_REQUIRE_CUDA is 1 in the environment: ...` | APAI's strict CUDA mode is set in the shell; APAI's grasp code fails with it | `unset APAI_REQUIRE_CUDA`, then start the agent program again |

## First run on the rig

This is the operator's checklist for the first session on the real arm. **When it was
written, none of it had been run.** The first session on the rig, on 2026-11-16, was run
with it; what that session showed to be wrong or missing is corrected here and in
[Set up and start the rig](#set-up-and-start-the-rig). No record of that session says
which steps passed, so no step below is marked as run. The checklist follows the programs
as they are built; where a line a program prints differs from what is written here, stop
and find out why.

**The speeds.** The checklist was written for a first session at 0.1 rad/s, and the
headings of its parts D and F-B still name that speed. The rig files hold the rig
owner's speeds: 0.2 rad/s for the agent and the return to rest, and a cap of 1.6 rad/s
for a person, with a hand speed limit of 0.5 m/s (see
[The rig files](#the-rig-files)). To run the checklist at 0.1 rad/s, set the four keys
of the 0.2 and `limits.human.max_joint_speed` to 0.1, on both machines, before step 15:
the human program follows the limit the robot app names. A person's first motions on
the arm at 1.6 rad/s are the first moves of the contact watch at that speed (see
[Known gaps](#known-gaps)): keep the gripper well above the table until F16 has shown
that the watch neither trips by itself nor lets a blocked joint go on.

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
  - **Workstation person**: watches the viewer's page with its live picture and the
    console, keeps that page's tab in front, types at the agent terminal and does the
    workstation's actions. The stop person may do this only if both keyboards are
    within one arm's length.
  - **Supporter**: stands at the 48 V switch and supports the arm from the side, outside
    the path of the gripper, in every step of F-B until steps 21 to 26 and F16 have passed.

  F-B needs the wearer and the stop person at the least, and the supporter whenever steps
  21 to 26 did not pass before it (the variant F-B0). With fewer people F-B is not run.
- **A person's takeover needs the live view.** The rig's robot files
  (`robot_app_rebot.yaml` and `robot_app_human_check.yaml`) ship with
  `human.require_live_view: true`. The live view is the viewer's page with its live
  picture showing, in a visible browser tab (see [The live view](#the-live-view)). The
  viewer runs in the pane `viewer` of `run_workstation.sh`; the page is opened in step
  12 and stays open, in front, for the whole session. Without it the robot app refuses
  the takeover and ends a human drive in a hold.
- **No step cuts torque away from measured rest.** Torque is cut by the connect of the real
  driver (step 15, and step 18's `on`), by a `release` (step 17), at the end of every typed
  `rest` (steps 16 and 28, and the `rest` of any other step outside an open takeover) and
  by the shutdown (step 28). In each the arm lies at rest, and the program measures that
  before it cuts. After any `rest` the motors are off: `on`, then `yes`, switches them on
  again before the next step that moves. A `rest` while a takeover is open leaves the
  motors on, so no step of part F cuts torque before the hand-back.
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
- The code is on the Thor, at the workstation's commit. That is the operator's step:
  carry it as a git bundle and clone it with `GIT_LFS_SKIP_SMUDGE=1`
  ([step 1](#1-the-same-commit-on-both-machines) of "Set up and start the rig"). Compare
  `git rev-parse HEAD` on both machines and write both commit ids into the session
  record. Nothing checks that the two are the same.
- Isaac Capture is built on the Thor for the Python of APAI's environment there, with
  the four settings of [step 2](#2-build-isaac-capture-on-the-thor): without them
  configure stops for `glslangValidator` and for `clang-format`.
- The camera sender is built on the Thor, and the rig's camera bring-up script is known
  ([step 8](#8-the-camera-from-the-thor-to-the-workstation)).
- The variables of each machine are in one file
  ([step 3](#3-the-variables-of-each-machine-in-one-file)): on the workstation those of
  `run_workstation.sh`, on the Thor those of `run_robot_app.sh`.
- The edits each machine needs are made, in the checkout the programs are started from
  ([step 5](#5-the-edits-each-machine-needs-which-are-not-committed)): the Thor's address
  in `allowed_peers` of `EX/configs/link_service_rebot.yaml`. The viewer finds the
  camera consumer's interpreter under `AGENTIC_REMOTE_CAMERA_PR`: without that
  variable, or with another layout of the camera streamer's folder, the viewer's page
  of step 12 has no live picture, and no takeover is taken.
- The model client for the workstation's processor is installed and named in
  `experiment.codex_bin` ([step 4](#4-the-model-client-on-the-workstation)); step 20
  needs it.

For part F, demo 2, in addition:

- The two one-machine recipes ran by hand
  ([Run demo 1 by hand with the mock agent](#run-demo-1-by-hand-with-the-mock-agent),
  [Run demo 2 on one machine](#run-demo-2-on-one-machine)).
- The bells ring apart at the real console terminal. With a robot link service whose file
  has `operator.typed_takeover: true` and the robot app on the simulated arm with
  `configs/robot_app.yaml` (human driving off): type `takeover`, then `yes`. The line
  `[refused] owner request 'human' refused: ...` must ring three separate times. If the
  three merge into one, the wearer cannot tell a refusal from a takeover by ear: the stop
  person says each of them aloud.
- The owner has decided the settings of
  [The settings that ship off](#the-settings-that-ship-off): `human.enabled`,
  `operator.typed_takeover`, the human speeds and `human.view_age_limit_s`. The rig
  files hold `human.require_live_view: true`: the live view is the viewer's page with
  its live picture.
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
  program on the Thor dials that server across the wired link. Open port 47600 only
  toward the Thor: the server checks no address and serves whichever client connects
  first.
- The headset is the Quest on its USB-C cable, as in the owner's tested recipe.

In the tables, `EX` is `examples/agentic_remote`, `PROFILE` is the rig's APAI arm profile
(the path of `configs/thor_operation` in the experiment folder on the Thor, without
`.yaml`) and `WS` is the workstation's address on the wired link. In part F, `BUILD` is
the Isaac Capture build folder on the workstation. In part F the Thor has two terminals:
the Thor terminal, in which the robot app runs, and a second one for the human program.
The commands of part F for the Thor follow the programs as they are built; none of them
was run on hardware.

Steps 7, 8, 11 and 20 start the workstation's programs one by one, the robot link
service of step 7 first: every other program of the workstation attaches to it. `bash
EX/run_workstation.sh --env FILE --thor THOR` starts the same four and the experiment's
viewer together, each in a pane of one tmux window, with the agent's command of step 20
typed into its pane and waiting for Enter (see [The start scripts](#the-start-scripts)).
Step 12 opens the viewer's page, which is the live view; started by hand, the viewer is
`EX/run_viewer.sh`. Steps 12 and 13 read the view age, which a person's takeover in
part F needs.

### A. Before power: no arm motion

| # | Where | Do | Must show | Write down | Abort if |
|---|---|---|---|---|---|
| 1 | Workstation | `nvidia-smi --query-gpu=index,pci.bus_id --format=csv` | Index 0 is the card the camera receiver uses (`00000000:C1:00.0` on this workstation) | The output | Another card is index 0 |
| 2 | Both | Connect the two machines by wire. Write the Thor's address into `allowed_peers` of `EX/configs/link_service_rebot.yaml`, in the checkout the service is started from; the edit is not committed | Each machine reaches the other | Both addresses, the link speed | No wired link |
| 3 | Thor | `timedatectl` | - | Whether the clock is synchronized, and by what | The clock is being set by hand |
| 4 | Thor, by eye | 48 V off. The arm lies folded on its stops. The jaws are empty. Nothing is in the arm's reach | - | A photo | The arm is not at rest |
| 5 | Thor | Bring the motor bus up with the experiment's script (it needs sudo and sends no motor command) | The bus is `can4` at 1 Mbit/s | - | The script stops |
| 6 | Thor | `ls -la ~/.apai/arm_holds` | No hold marker for `can4` (`can4.json`) and no `*.confirm.json` file at all | - | A marker exists: do APAI's operator recovery first. A `*.confirm.json` file exists: revoke it with APAI's `scripts/robots/arm_hold.py revoke`. The folder cannot be listed (`Permission denied`): let its owner read and enter it; a revoke cannot list it either |
| 7 | Workstation | `PYTHONPATH=EX/python:BUILD/python_package/Release python -m isaaccapture_examples.agentic_remote.link_service --config EX/configs/link_service_rebot.yaml`. `BUILD` is the Isaac Capture build folder on the workstation; `python` is a Python of the version that build was made for, with numpy and PyYAML: Isaac Capture's package imports numpy, and the service reads its file with PyYAML. The Python of the workstation environment is one | `robot link service: waits for the robot app on ...:47610` | - | Exit code 1 or 2. Exit code 2 with `cannot be imported ...: No module named 'numpy'`: that Python lacks numpy. Exit code 2 with `allowed_peers must be a non-empty list`: step 2 was not done in this checkout |
| 8 | Workstation | `EX/run_console.sh`, after the service of step 7 | `attached to the robot link service: ...` and a status line with `robot connection lost (not connected)` | - | It cannot attach. A console that was started before the service tries for one minute and then waits for a typed `attach` |
| 9 | Thor | Describe only: `EX/run_robot_app.sh --config EX/configs/robot_app_rebot.yaml --arm apai --apai-profile PROFILE` | `robot app with --arm apai would run:`, the driver `built, not connected`, exit code 2 | The profile's speed cap, the two rest tolerances, the jaw tolerance | A refusal (speed cap, rest pose, joint count) |
| 10 | Thor, then console | The link without the arm: `EX/run_robot_app.sh --config EX/configs/robot_app.yaml --arm sim --connect WS:47610`. Type `stop` at the console. Leave this robot app running until step 14 | The console's `WELCOME` line says `driver sim`; `stop: taken` | The time from Enter to `taken`, five times | No connection. This is the first run of port 47610 between two machines. This robot app proves the link only: the agent program on `agent_rebot.yaml` takes no task against it and ends with `the robot app's profile is 'sim-6dof', expected 'rebot-b601-rs-thor'` |
| 11 | Thor, workstation | On the Thor, as in [step 8](#8-the-camera-from-the-thor-to-the-workstation) of "Set up and start the rig": the rig's camera bring-up script with sudo, then stop `sensing-camera-argus.service`, then the camera sender with `EX/configs/camera_sender_rebot.yaml`. On the workstation `EX/run_camera_receiver.sh --host THOR`, with the Thor's address | The sender prints `Receiver connected from ...`; the receiver prints `connected to AVS1 sender ...` and reports about 60 camera frames per second | The rate; the refused camera frames | The sender ends with `No camera devices are available`: the bring-up was not run since the boot, or `sensing-camera-argus.service` still runs. The receiver prints `waiting for AVS1 sender 127.0.0.1:5000`: `--host` is missing or names the workstation. The sender refuses the sensor mode |
| 12 | Workstation | The live view. The viewer runs in the pane `viewer` (by hand: `EX/run_viewer.sh`). Open its page in a browser on the workstation, at the address the pane names, and keep its tab in front and the live picture inside the window. The page follows live: its button reads `Map picture` | The arm at rest in the live picture. Nothing stands beside the button, and the line under it ends with `The robot is told: live view, N s old.` The pane `viewer` has printed `viewer: live picture: the camera's pictures arrive` and `viewer: live view: attached to the robot link service at ...`. The console's status line shows `view` under 1 s | The view age over one minute: lowest, middle, highest; the pictures a second the page names; the GPU memory with the receiver and the viewer | `no live picture` on the button: the text under it says why (the camera receiver, `AGENTIC_REMOTE_CAMERA_PR`). `scroll the live picture into view` beside the button: less than three quarters of the picture are inside the window. `watch only: see below`: the page was opened from another machine (open it on the workstation), or a newer load of the page on the workstation counts (the note under the button says `reload this page to make it the live view`: reload it). `viewer not reached: see below`: the viewer does not run; start it, and the page loads itself again. `The robot is not told of this live view: ...`: the robot link service does not run, or another viewer is attached. The console shows `view none` while the page shows the live picture |
| 13 | Viewer's page, console | Read the view age | Between 0 and 1 s, steady | - | Negative, above 1 s or drifting: no task today. Find the cause on the Thor (the sender's clock, a clock step); there is no correction setting. For demo 2 go on with F-A before step 14 |
| 14 | Thor, workstation | `quit` at the Thor terminal (the simulated arm is at rest: it ends); after F-A no robot app runs and there is nothing to quit. Then `EX/run_agent.sh --config EX/configs/agent_rebot.yaml --check` | Exit code 0; the calibration id it prints (`"calibration_id"` in its JSON) equals `camera_calibration_id` of `robot_app_rebot.yaml` on the Thor. It is the name step 15 gives as `--calibration` | The id | Another id: find out which calibration the rig is in. After a new calibration on the Thor the workstation's copy of the hand-eye file and the two names have to follow ([step 6](#6-the-camera-calibration-and-its-name) of "Set up and start the rig") |

### F-A. Before power: headset and human path on the simulated arm

Part F is demo 2 on the rig. F-A runs before power, after step 13 and before step 14: the
headset, the human program and the robot app on the simulated arm on the Thor. No arm
motion. F-B, on the powered arm, follows step 27. In both the human program runs on the
Thor, attached at the robot app's own socket, and reaches the Teleop Session Server on
the workstation over the wired link.

F-A runs with the live view, as F-B does: `robot_app_human_check.yaml` ships with
`human.require_live_view: true`. The viewer's page of step 12 stays open, in front, with
its live picture showing; F8 takes it away on purpose.

| # | Where | Do | Must show | Write down | Abort if |
|---|---|---|---|---|---|
| F0 | Workstation, Thor | Switch the screen lock and blanking off for the session; keep the viewer's window un-minimised with its tab in front and the live picture showing. From step 12's numbers choose `human.view_age_limit_s`: the highest age plus 0.2 s, at most 1.0 s. With a VIEW_STATUS every 0.2 s the age swings by about 0.2 s before any camera delay. Write the limit into `robot_app_human_check.yaml` and `robot_app_rebot.yaml` on the Thor. Then `quit` step 10's robot app (it is at rest) and start it again with its socket for the human program: `EX/run_robot_app.sh --config EX/configs/robot_app_human_check.yaml --arm sim --connect WS:47610 --human-socket ~/.agentic_remote/human.sock` | The Thor terminal prints `robot app: the human program's socket is ...` after its session line, and no line that begins `robot app: human driving is not tied to a live view`. The console's `WELCOME` says `driver sim`; `view` stays under the chosen limit | The ages; the limit chosen | The Thor terminal prints `robot app: human driving is not tied to a live view (human.require_live_view false)`: the file on the Thor does not require the live view; put `human.require_live_view: true` back. The highest age is above 0.8 s or drifts: no human motion today |
| F1 | Workstation | Terminal T1: `TELEOP_CLIENT_WARMUP_END_TIMEOUT_MS=0 PYTHONPATH=BUILD/python_package/Release SRVPY -m isaaccapture.cloudxr.service run --accept-eula --setup-oob --usb-local`. Without that variable the web client on the headset drops the stream after about 30 s. Then terminal T2, on the address the Thor reaches: `PYTHONPATH=BUILD/python_package/Release SRVPY -m isaaccapture.teleop_session_server --listen tcp://WS:47600` | T2 prints `Teleop Session Server listening on ...`, then logs `No headset is connected to CloudXR; waiting for one` | Both start lines | Exit code 1: CloudXR is not available |
| F2 | Thor, second terminal | Start the human program before the headset is connected, with the variables of `run_robot_app.sh` set: `AGENTIC_REMOTE_HUMAN_PYTHON="$AGENTIC_REMOTE_THOR_PYTHON" bash EX/run_human.sh --config EX/configs/human_rebot.yaml` | `human program: teleop; driver sim; arm model rebot_rs, 6 joints`; then `human program: Teleop Session Server connecting_openxr: ...` or `waiting_for_headset: ...`. Console: `human program at the robot app`, `headset <that stage>` and `takeover: not possible: no fresh controller data: stage ...` | The lines | The human program ends. `cannot attach to the robot link service or the robot app: ...`: the robot app of F0 was started without `--human-socket`, or with another path than `socket_path` of `human_rebot.yaml`. `no_server` while T2 runs: `teleop.endpoint` still holds its placeholder, T2 does not listen on `WS`, or the workstation's firewall keeps the Thor from port 47600 |
| F3 | Headset | Headset on the USB cable. In the web client's panel switch Headless on, then press Connect | T2 logs `Opened the OpenXR session`; its line `Serving robot app 'agentic_remote_human' ...` names the human program. Without a restart of the human program: `human program: Teleop Session Server ready`, `human program: right controller seen`; console `headset ready` | The time from Connect to `ready` | No `ready` within 2 minutes. `NotSupportedError` in a loop on the headset: switch Headless on, press Connect by hand, restart T2 |
| F4 | Thor, second terminal | Read the human program's report line for one minute | `stage ready`; tick age, state age and round trip steady | Each of the three: lowest, middle, highest. The tick age is that of SAMPLES that crossed the wired link. The round trip is the way through the robot app's own socket both ways plus one pass of the human program. Also `controller X s` of the console's status line: the sample age with the way to the robot app in it | The tick age comes near `teleop.max_tick_age_s` (0.15 s), or the round trip is above 0.1 s or drifts: find out why before any human motion |
| F5 | Headset, by eye | The seeing check. Three settings. (1) Headless on: the web client draws nothing; can the wearer see the arm, and the viewer's live picture, below the visor? (2) Headless off: does the room show, and do controller data keep arriving (the report line)? (3) The headset off the face with its proximity sensor overridden (`adb shell am broadcast -a com.oculus.vrpowermanager.prox_close`), resting on the forehead or on a stand that sees the right hand: the wearer watches the arm directly | - | The three answers, in words | The wearer must see the arm itself in one of the three settings. The live view rule is met by the Thor's check plus the workstation person watching the viewer's page. No setting works: no human motion on the arm today |
| F6 | Controller, Thor, console | Press right A, right B, the grip and the trigger, each alone. Tap B ten times as briefly as possible. Then hold the controller out of the headset's view, behind the back, and press right B | The human program prints each edge in the Thor's second terminal (`human program: B down`, `human program: B up`). Each B press is a `[stop] STOP from controller: right B; ...` at the Thor terminal and on the console | Which of the four arrive. How many of the ten taps became a STOP. Whether the STOP arrives with the controller out of view | B or the grip never arrives: no demo 2 with buttons. If only A or B is missing, the owner may switch the typed fallback on: `operator.typed_takeover: true` in `link_service_rebot.yaml`, then restart the robot link service. B does not arrive out of view: the wearer keeps the controller in view, and the abort stays the Thor terminal |
| F7 | Wearer, workstation | The direction check, on the simulated arm. Stand behind the arm, recentre. Hold B 1 s. Release the grip, then squeeze. Move the hand **up** about 10 cm at about 1 cm per second, then forward, then left. Then turn the hand about its forward axis by about 30 degrees and back | Console: `[owner] TAKEOVER: a person owns the arm, ...`, two bells that are heard apart. The report line's tip target: z rises for up, x for forward, y for left; roll changes with the hand | The three directions as seen. The `ended:` counts in two minutes of motion. The IK time | The note `out of reach`, with the arm hardly moving, while the tip is still at the rest height is expected: from rest, lift the gripper first. A wrong heading: set `retarget.stage_yaw_deg` in `human_rebot.yaml` and repeat; a positive value turns the hand's motion counter-clockwise seen from above. A swapped or mirrored axis: correct `retarget.controller_to_base`. Many jumps or failures once the arm is raised: no human motion on the arm today. A lift that slows and ends with `out of reach` about 39 cm above rest is the end of the arm's reach on the model. `human drive ended: TCP [...] outside workspace ...` is the tip box of APAI's human checker, which the simulated arm of F-A does not have: with the rig file 10 to 55 cm in front of the base, 50 cm to each side, 70 cm up |
| F8 | Viewer's page | While squeezing on the simulated arm, each once: press `Map picture` on the viewer's page; hide the browser tab; freeze the browser (stop its process, then let it go on); close it; load the page in a second tab; stop the camera sender; end the viewer with Ctrl+C in its pane. After each: bring the live view back (press `Live picture`, bring the tab to the front, reload the page, start the sender, start the viewer again: its page loads itself again once, and a page that says `watch only` is reloaded by hand) and wait until the console's `view` is under the limit; if it stays `none` for more than a minute, restart the receiver and the viewer (steps 11 and 12) and reload the page. After a start of the viewer, and after any step of the Thor's clock, the view is not live for 3 s | `[human] human drive ended: the live view is not fresh (...)`, one bell; the simulated arm's motion is `hold` each time. Right B held then is refused, with three bells: `[refused] owner request 'human' refused: the live view is not fresh: ...; show the viewer's live picture in a visible browser tab` | The time from the action to the EVENT | The drive goes on, or a takeover is taken while the live picture does not show |
| F9 | Headset | While squeezing, each once: switch the right controller off; take the headset off and put the controller down; switch the headset's Wi-Fi off; stop the Teleop Session Server (then start it again, switch Headless on again, wait for `ready`); pull the network cable between the two machines, which carries the teleop connection and the robot connection (then put it back and wait for `headset ready` and for `view` under the limit) | `[human] human drive ended: ...`; `hold` each time. For the cable the Thor terminal shows `[connection] robot connection lost: the arm holds; ...`, and right B is not read while the cable is out: the buttons come over it | The time from the action to the hold. For the Wi-Fi: how far the simulated arm went and the time to `no_controller`. For the cable: the time until the human program says `ready` again | The drive goes on |
| F10 | Console, controller, Thor | The STOP under load. While squeezing on the simulated arm (50 HUMAN_SETPOINTs per second at the robot app's own socket): type `stop` at the console, five times; press right B, five times; take over again between them. Then right B without the robot connection: end the robot link service with Ctrl+C, press right B, start the service again. Type `attach` at the console if it no longer tries by itself. The viewer goes on and attaches to the new service by itself, within a minute (`viewer: live view: attached to the robot link service at ...` in its pane): wait until the console's `view` is under the limit before F11 | `stop: taken`; `[stop] STOP from controller: right B; ...`. With the service ended the Thor terminal shows `robot app: no robot connection: ...` and, at the press, `[stop] STOP from controller: right B; the arm holds, stop number N` | Each time from Enter or press to the higher stop number; compare with step 10's five values | Any above 0.25 s: set `rates.loop_hz: 25` in `human_rebot.yaml` and repeat (that halves the hand speed limits); still above: no human motion on the arm today. No STOP at the Thor terminal for right B with the service ended: no human motion on the arm today |
| F11 | Controller | The recentre. While squeezing on the simulated arm: recentre the headset view | `human drive ended` or a note of the human program; the report line's tip target does not jump; the simulated arm's target moves by no more than `limits.human.max_step_rad` | What ended it | The simulated arm jumps: no human motion on the arm today |
| F12 | All | The time from a spoken "go", with the headset asleep on the table, to `TAKEOVER: a person owns the arm` on the simulated arm. A right B that is already down as the headset wakes up is a STOP and takes nothing over: the note then reads `release right B, then hold it for 1 s with the grip released to take over` | - | The time. Compare with `waits.assistance_s` (120 s) | - |
| F12b | Controller, console, Thor | The trigger, on the simulated arm, while the person owns it and its jaws are open and empty. With the grip released: pull the trigger slowly all the way, hold it, release it slowly all the way | The console's jaws line reads `jaws: follow the trigger`. The human program's report line shows `jaws follow the trigger yes, grip level 1.00`, and its `trigger asks` goes from 1.00 to 0.00 and back | What `trigger asks` reads with the trigger at rest and with it fully pulled | At rest it stays below 1.00, or fully pulled above 0.00: the dead zones of the trigger, 0.05 at each end, do not cover this controller, and a trigger at rest could leave the jaws short of fully open, where the hand-back is refused. Set `human.trigger_jaws: false` in `robot_app_rebot.yaml` for this session |

Then type `rest` and `path is clear` (the simulated arm was driven away from rest), hold
right A 1 s, `quit` at the Thor terminal, and go on with step 14. The human program ends
with that robot app, whose socket it was attached at: exit code 1 and `human program:
the robot link service is gone (...)`. It is started again in F13, after the robot app
of step 15.

F-A does not meet APAI's human checker: its robot app runs on the simulated arm. So it
meets neither the table nor a tip box, which the simulated arm's checkers do not have;
the tests of a person's drive close to the table run on APAI's `mock` profile
(`test_lift_out.py`). A test builds APAI's checkers from the rig's robot file on the
rig's profile as the launcher has it, and checks that a person alone gets the 1.6 rad/s,
the table clearance of 0.0 and the wider tip box
(`test_on_the_rig_file_a_person_is_checked_with_the_owners_values_and_nobody_else` in
`test_apai_backend.py`). On the arm that checker is first met in F16. Not checked
anywhere: the wrist camera rule under larger motions. F-A does not meet
the rig's human joint box either: `robot_app_human_check.yaml` has a box of -2.5 to
2.5 rad that equals its region, and the simulated arm reads exactly what it was sent. The
rig file's box, 0.02 rad inside `rest.region_*`, is first met on the arm in F-B.

Before step 15 the owner decides whether `human.enabled` in `robot_app_rebot.yaml` is set
to `true` for this session, confirms `human.view_age_limit_s` there, the limit of F0, and
decides whether `human.trigger_jaws` stays `true`. `human.require_live_view` stays
`true`. If F5, F6, F7 or F10 failed, `human.enabled` stays
`false` and part F-B is not run. For F-B the robot app of step 15 is started with
`--human-socket ~/.agentic_remote/human.sock`: without that socket the human program
has nothing to attach to, and adding it later needs `rest`, a shutdown and steps 15 and
16 again.

### B. Power and connect: the connect cuts torque, so at rest only

| # | Where | Do | Must show | Write down | Abort if |
|---|---|---|---|---|---|
| 15 | Thor | 48 V on. The arm still lies at rest; the second person has a hand on it. `EX/run_robot_app.sh --config EX/configs/robot_app_rebot.yaml --arm apai --driver real --apai-profile PROFILE --calibration ID --connect WS:47610`, for demo 2 with `--human-socket ~/.agentic_remote/human.sock` added; answer the three questions with `yes` | `robot app: the real driver is connected at measured rest: torque on, hold unknown, motion hold` (or `rest_hold`)`, jaws referenced`; then `robot app: session ...`, with no line `robot app: human driving is not tied to a live view ...` between those two: the rig's file requires the live view; the console's `WELCOME` says `driver real`; in the robot app's log `one joint read takes ... ms on the motor bus ...` | The joint reads of the connect, from `~/.apai/logs/rest_readings.jsonl`; the jaw reading; the time one joint read takes | A line that begins `robot app: the robot app did not start: `, then a line that begins `robot app: APAI's driver refused the connect before anything that cuts torque`: the motors were never switched on and the arm lies on its stops as in step 4. Read the reason, correct the pose by hand, start again. A line that begins `robot app: the robot app did not start: `, then a line that begins `robot app: the start failed after the connect began: the motors may be enabled and holding`: the program has ended without releasing them. The arm lies at rest: with the second person's hand on it, switch the 48 V off, read the reason, start again |
| 16 | Thor | If the motion is `hold`: type `rest`, then `path is clear`. The arm moves a few hundredths of a rad, the jaws close and the motors go off there: a `rest` now ends as step 17's release does | The `[rest]` lines of the fold, `[jaws] at rest: the jaws are closed`, `[release] torque release sent at rest: the driver does not confirm that the motors are off`, `rest: completed`; motion `rest_unpowered` | The planned and the real time; whether the motors are off, by hand feel | The arm moves away from rest: `stop`. `rest: failed: at rest with the motors on; release refused: ...`: the arm holds at rest; read why |
| 17 | Console, Thor | The release at rest, if step 16 was not needed: `release`, then `yes` | `[jaws] at rest: the jaws are closed`; `torque release sent at rest: the driver does not confirm that the motors are off`; the motors are off | Whether the motors are off, by hand feel | `refused`: the motors stay on; read why. `failed` with `fault: torque release failed, the motors may be on or off: ...` and `torque unknown` on the console: keep a hand on the arm and treat it as powered. With the arm's real driver no second release is possible and `quit` fails too: end the robot app with SIGTERM from a second terminal on the Thor (Ctrl+C is a STOP) and start it again with the arm at rest (see [the torque rule](#the-rule-torque-is-never-cut-away-from-rest)) |
| 18 | Thor | The motors on again, without a new start of the robot app: `on`, then `yes`, with the second person's hand on the arm. The driver connects as in step 15, which cuts torque for a moment at rest | `[rest] the motors are on again at rest: motion rest_hold` (or `motion hold`, when the arm lies more than 0.05 rad from the rest pose: a task is taken from there too) and `on: completed`; the session id is the same | Whether the arm stayed on its stops | `[refused] the motors were not switched on again: ...`: the motors are off and the arm lies at rest; read why. If `on` cannot be made to work: `quit`, then steps 15 and 16 again |
| 19 | Console, Thor | With the arm at rest: `stop` at the console; Ctrl+C at the Thor terminal. A `reset` is not needed: a STOP is no fault-stop | The stop number goes up each time; the arm does not move | The delays | The arm moves |

### C. Camera and agent: no motion

| # | Where | Do | Must show | Write down | Abort if |
|---|---|---|---|---|---|
| 20 | Workstation | `EX/run_agent.sh --config EX/configs/agent_rebot.yaml --interactive`; the task `look at the table and report what objects you see` | `driver real` in the agent program's line about the robot connection; the obstacle grid goes live; a report; the agent takes no arm; the jaws do not move when the task ends | The time to a live grid; the depth time per stereo pair; the GPU memory in all; the model's answer time | A calibration mismatch; every camera frame refused; no grid within `task.grid_wait_s` |

Step 20 needs the model client for the workstation's processor, installed and named in
`experiment.codex_bin`, and the user's login for it (see
[The model client on the workstation](#4-the-model-client-on-the-workstation)). Without
it steps 20 to 27 are not run: after step 19 the session goes on with the variant F-B0 of part F, or ends
with steps 28 and 29.

### D. First motion: the second person supports the arm; 0.1 rad/s; no timed return

| # | Where | Do | Must show | Write down | Abort if |
|---|---|---|---|---|---|
| 21 | Agent terminal, console | The task `open the gripper`, then `yes`. Then a task that lifts the gripper a few centimetres; type `stop` at the console while the move is under way | The jaws open; the move starts; after `stop` the arm holds, nobody owns it and the task ends | The time from Enter to the higher stop number; how far the arm went after it | More than 0.25 s |
| 22 | Agent terminal, Thor | The same move twice more: end it with Ctrl+C at the agent terminal, then with `stop` at the Thor terminal | The arm holds each time | The delays | The arm goes on |
| 23 | Console, run folder | A full small move. Read the permit in the console's status line and the move's records in the run folder | Every setpoint under a permit; no `permit evidence is in the future` | The oldest evidence; the longest gap between permits; the arm's stopping distance at the session's joint speed (0.2 rad/s with the rig files) | Permits refused for their time |
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

F-B runs with the live view: the viewer's page stays open, in front, with its live
picture showing, and the workstation person watches it. The stop person stays at the
Thor terminal for all of F-B, with `stop` typed and a finger on Enter.

| # | Where | Do | Must show | Write down | Abort if |
|---|---|---|---|---|---|
| F13 | Thor, viewer's page, console | Start the human program in the Thor's second terminal as in F2: the one of F-A ended with its robot app. The wearer takes the place and the headset setting of F5. The stop person sits at the Thor terminal with `stop` typed. The viewer's page is open, in front, with its live picture showing. Check the view age again with the arm powered | `human program: teleop; driver real; arm model rebot_rs, 6 joints`. The console shows `human program at the robot app` and `headset ready`, and its `view` stays under the chosen limit for one minute | The age: lowest, middle, highest | It does not: no human motion today. Changing the limit now needs `rest`, a shutdown and steps 15 and 16 again |
| F14 | Agent terminal, by hand | A task that lifts the gripper; a foam block into the path (step 24), well above the table | The arm holds before the block; `HELP NEEDED`; the wait counts and says `ARM WILL KEEP HOLDING`; the console says `takeover: possible now ...` | The time to the help block | `takeover: not possible`: read why. An arm that the agent left with the gripper tip under 2 cm or inside its last grasp region takes the takeover of F15 and no `rest` before the wearer has lifted it above the 2 cm and out of the region. Where the arm model puts the tip under the table plane the takeover's line ends `; the gripper tip is low: only a lift is taken until it is clear`, and the arm first rises straight up. An arm that the agent left with a joint in the outer 0.02 rad of its working box, outside the human joint box, takes no takeover either, and the takeover line does not say so: it reads `takeover: possible now ...`, and the hold of F15 is refused with `the arm is outside the human limits: ...; type rest` and three bells. There the stop person types `rest`, then `path is clear` (see [Known gaps](#known-gaps)) |
| F15 | Controller | Hold right B 1 s with the grip released. Then take the foam block and everything else out of the arm's reach | `[stop] STOP from controller: right B; ...`; the wait is cancelled; `[owner] TAKEOVER: a person owns the arm, ...`, two bells. The arm does not move. The jaws do not move: a takeover sends no jaw command, and the wearer keeps the trigger released | The time from the press to the STOP's higher stop number | The arm or the jaws move |
| F16 | Controller | The first human motion. Release the grip, then squeeze; move the hand 3 cm straight up over at least 5 s; release. The stop person presses Enter at the first motion that is not upward. The arm follows the hand as fast as the joints are allowed, 0.9 of the Thor's clamp of `limits.human.max_joint_speed`, and goes on to where the hand stopped, by the straight way and not along the hand's ([How fast the arm follows the hand](#how-fast-the-arm-follows-the-hand)): move the hand slowly, the arm then keeps its pace and the hand's way, and release the grip to stop the arm at once. Keep the hand's orientation as it was at the squeeze: a turned hand turns the tool until the tool has caught up or the grip is released, and the note `hand N cm ahead: ...` compares positions only. The gripper tip stops 1 cm inside the walls of the human checker's tip box, 25 cm above the ready pose and 9 cm in front of it, with the note `at the workspace limit: <wall>`: bring the hand back inside and the arm follows again, with no new squeeze. It stops 0.5 cm above the table plane the same way (`at the workspace limit: table`): in F16 keep the gripper well above the table. Where a joint or the reach ends the note says `at a joint limit: joint N` or `out of reach` and the arm follows as far as it can, with the wrist left turned off the hand's: bring the hand back there too. Then squeeze and release three times with the hand held still; a squeeze within half a second of a release at speed shows `wait: the arm is still stopping` and counts once the arm reads as at rest | The arm follows upward and holds on the release. With the still hand the arm does not move | The lag by eye; any `human drive ended`, and for one that names a contact its text, with the tracking error it gives (APAI's contact monitor trips only beyond its lag band, about 0.03 rad at 0.1 rad/s; it was not run at a person's 1.6 rad/s, where a setpoint is 1.8 degrees a joint: write down every drive it ends with nothing in the arm's way); any `the robot app is behind: the arm follows slower` and any `the target is X rad from the arm ...`, with the count of setpoints of that drive in the Thor's log (`human drive: ..., sent N setpoints`): under 45 a second the Thor's loop runs slow; how long `wait: the arm is still stopping` shows after a release at speed (two readings of a holding joint are taken to agree within 0.002 rad; if it shows for the full half second every time, they do not on this arm); whether the arm sinks from one squeeze to the next, and by how much (a drive starts from the measured pose, so an arm that sags under load creeps down by the sag per squeeze, and nothing stops that above the 0.5 cm over the table plane) | Any motion that is not upward; no hold on the release; a creep that does not stop |
| F17 | Controller, agent terminal | Squeeze and move; press right B during the motion. Then try a task at the agent terminal | The arm holds; nobody owns it; `takeover open`. The task is refused with `takeover not finished: give the arm back with right A at rest` | The time from the press to the hold; how far the arm went | More than 0.25 s |
| F18 | Controller, console | Hold B 1 s again. Then the trigger, with nothing between the jaws and the stop person's `stop` typed: pull it slowly all the way, hold it, release it slowly all the way | A takeover again. With `human.trigger_jaws: true`: the console's jaws line reads `jaws: follow the trigger`; the jaws close with the trigger and open again fully, in steps. With `false`: the jaws do not move | Any refusal of the takeover, with its text. With `true`: what the console's jaws line and the report line's `trigger asks` read at rest and fully pulled; the time the jaws take to open | With `true`: the jaws move while the console's jaws line does not read `jaws: follow the trigger`, or go on after the stop person's `stop`. With `false`: the jaws move. A takeover refused with `[refused] owner request 'human' refused: the arm is outside the human limits: ...; type rest` and three bells is no abort: the STOP of F17 came with a joint at the edge of the human joint box, where it reads just outside the box. The stop person types `rest`, then `path is clear`: the box lies 0.02 rad inside `rest.region_*`, so the pose is one the return starts from. Take over again from rest |
| F19 | Workstation, Thor | While squeezing, each once: hide the browser tab of the viewer's page (then bring it back to the front); stop the Teleop Session Server (then start it again, switch Headless on again in the headset, wait for `ready`); end the human program on the Thor with a kill signal (then start it again there) | The arm holds each time; the person stays the control owner; after a release and a new squeeze the arm follows again | The delays | The arm goes on, or its motors switch off |
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
| 28 | Console or Thor | `rest`, then `path is clear`. Then `shutdown` at the console (at rest it asks for `yes`) or `quit` at the Thor terminal | The `rest` measures rest, closes the jaws and releases torque there; the robot app then ends with exit code 0 | - | `refused`: the arm holds with its motors on; read why. Refused only for jaws that are not proven empty: look at them, then type `jaws are empty` when asked, or `stop`, `open`, `yes` if they hold something |
| 29 | All | 48 V off, with the arm at rest. Stop the camera sender, the receiver, the viewer, the console, the human program, the Teleop Session Server, CloudXR's service and the robot link service | - | What was left open | - |

Not in this session: reading the motors' CAN timeout parameter; ending the robot app during
a raised hold; the jaw rule with different objects; the timed return; the typed `reference`;
any speed above those of the rig files; the trigger on a held object (F18 moves the empty jaws, and
only with `human.trigger_jaws: true`); a person's drive close to the table, down to the
stop 0.5 cm above the table plane, unless the owner asks for it after F16; the typed
takeover unless F6 forces it; a person who drives from another room.

## What was run

Each table below is the record of the day it names: its rows say what the programs did
then. Three things changed on 2026-11-16, after the earlier tables were written: a typed
`rest` now ends with the jaws closed and the motors off, the two self-tests have 12 and
11 steps, and a task the model ends without success gives the arm back with no help
request. The last table records what of this ran in the tests it names. A person's
limits on the rig are the rig owner's values of 2026-11-18 (see
[The rig files](#the-rig-files)): the rows on a person's drive that name that day were
run with them. The one run on
the real arm that this README knows of is the rig owner's of
2026-11-16, named at the top; it is no row here, because no record of it is part of this
repository.

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
| `run_robot_app.sh` without `AGENTIC_REMOTE_THOR_PYTHON` | As a process | Exit code 2 and one line |
| `link_service --config configs/link_service_rebot.yaml` | As written | Exit code 2: `allowed_peers must be a non-empty list` |
| `robot_app --config configs/robot_app.yaml --arm apai` | As written, with `AGENTIC_REMOTE_APAI_SRC` set | Exit code 2: the human limit set's 1.5 rad/s is above the mock profile's speed cap of 1.2 rad/s |
| `robot_app --arm apai --apai-profile rebot_rs` without `--connect` | By a test, in the test process, with the shipped file's human limit set at 0.8 rad/s (`test_main.py`) | Exit code 2; `the profile's speed cap: 0.8 rad/s` is printed; nothing was connected. With `configs/robot_app_rebot.yaml` and the rig's own copy of the experiment's files this was not run: tests only load that file and build APAI's checkers from it on the rig's arm profile laid out as the launcher runs it. On APAI's stock `rebot_rs` profile the rig file is refused for a person's 1.6 rad/s |
| The service, the robot app (`--arm sim`, and `--arm apai --driver mock`), the operator console on a pseudo-terminal and the console command as processes | By a test, on a port the system picked | The console named the driver and showed `view none`; `console_cmd state` and `stop` ended with exit code 0; `rest` with `path is clear`, typed at the console, completed; `quit` ended the robot app and the console with exit code 0, SIGTERM the service |
| Demo 1: the agent program with two tasks, a console and a live view of the viewer's parts attached at once | By a test, in the test process, behind the service on 127.0.0.1, on APAI's software arm, with stand-in camera pictures for the view and acknowledgements that count for 4 s in that test | 905 joint targets in 8 moves, each under a permit; the console printed both grants and `the agent gave the arm back: task ended without success`, and no help request; the view age on the console was 0.1 to 1.4 s while a tab acknowledged a picture every 1.5 s; the viewer had the arm's state while it moved. 22.5 s |
| `stop` typed at the console during the agent's move, then a second task | By a test, the same way, the view without a camera | The move ended `cancelled` at the setpoint it was typed at (40 of 95); `stop: taken` came 5 to 11 ms after the line was typed, on one machine; the second task ran without a restart |
| `rest` typed at the console after a task | By a test, the same way | `yes` sent nothing; `path is clear` returned the arm to rest in 2.6 s (105 joint targets); motors on |
| An obstacle during the first task, then a second task | By a test, the same way | The first move was refused after 33 to 35 setpoints; the console showed the help request and then the next grant, no STOP; the second task moved the arm; one run of the program, 24.0 s |
| The agent client's socket closed while the agent owned the arm | By a test, on the simulated arm | The console showed `HELP NEEDED: agent program gone` and the wait counting down; `stop` ended it; nothing moved |
| The robot app ended and a new one started | By a test, on the simulated arm | The console said `ROBOT CONNECTION LOST` and showed the new session; the viewer stayed attached and had the new robot app's states; a command with the old session's numbers was refused |

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
| A takeover and a lift with the rig's two files | By a test, on 2026-11-18, in the test process, on a fake clock (`test_rig_configs.py`): `configs/robot_app_rebot.yaml` with human driving switched on in memory, the rig's arm profile laid out as the experiment's launcher runs it on the Thor (APAI's `rebot_rs` profile with the launcher's joint speed for its cap), APAI's driver class over a stand-in for the motor library, from each of the three rest readings recorded on the rig. The joint targets were the human program's own: its retargeting graph, solver and logic on `configs/human_rebot.yaml`, for a hand that rose 20 cm at 40 cm a second from the squeeze on | The takeover sent the motors nothing; APAI's human checker approved every setpoint of the lift; no step was above 0.98 of 1.6 rad/s; the first steps took the arm off its two joint stops at that clamp while the human program's first joint target stood; after them the fastest joint ran at 0.9 of the clamp, 1.41 rad/s, and the gripper tip rose to where the hand stopped; the drive ended by the released grip, with no EVENT; the typed `rest` and the hand-back completed, the hand-back with nothing sent to the arm's motors or the jaw motor; the motors stayed on |
| How fast the arm follows the hand, with the rig's two files | By tests, on 2026-11-18, at the rig's 1.6 rad/s (`test_human_program.py`): the human program's graph, solver and logic on `configs/human_rebot.yaml`, pass by pass against a stand-in for the Thor with the speed clamp and the step limit of `configs/robot_app_rebot.yaml`, a start read of 0.06 s and ROBOT_STATEs 1, 4 and 9 passes old | A hand at 5, 20 and 50 cm a second in six directions, and one that turned at 0.3, 1 and 3 rad/s, was followed to where it stopped: no engagement ended, the stand-in refused nothing, its clamp cut no step, and no joint of the joint target was faster than 0.9 of the clamp. A hand at 50 cm a second from three poses was followed with the fastest joint at that limit and the tip at up to 20 to 45 cm a second, by pose and direction. A hand moved 30 cm in 0.4 s was reached with the arm on its way for 0.92 s forward and back and 1.18 s to the side, with the note `hand N cm ahead: the arm is catching up` meanwhile. From each rig rest reading a squeeze with the hand already rising lifted the arm off its stops; a hand 50 cm ahead ended the engagement and the arm held; after a released grip and after right B the arm got no further setpoint. On a stand-in whose own joint target stood on one joint, the joint target was held back two of the Thor's steps ahead and the engagement ended a second later with `the arm does not follow: ...`; with ROBOT_STATEs 4 and 9 passes old it ended as the Thor's refusal at its step limit. Not run on the arm |
| A second squeeze after a fast move, a Thor that pauses, the joint box and the end of the reach | By tests, on 2026-11-18, at the rig's 1.6 rad/s. In real time (`test_human_realtime.py`): the human program's `main()` against the rig's robot app on the simulated arm, through a robot link service. Pass by pass (`test_human_logic.py`, `test_human_program.py`): on stand-ins, and against a stand-in Thor that names the rig's tip box, table clearance and joint box | In real time: a grip released at full joint speed and squeezed again 40, 60, 100 and 200 ms later, 3 runs each: both squeezes drove, nothing was refused or ended, the arm went back at most 0.0004 rad, and the second drive's first setpoint went out 0.36 to 0.41 s after the release, with the note `wait: the arm is still stopping` meanwhile. Pauses of 60 and 90 ms at full joint speed, of the Thor's drive thread alone and of the whole robot app, 6 runs each: 24 of 24 ridden out, the farthest a joint target lay from the Thor's last setpoint 0.0632 rad, no setpoint a larger step than the clamp; pauses of 200 and 500 ms ended the drive by the Thor's late-setpoint rule. 30 cm to the side with nothing paused: the arm was on its way for 0.78 s. Pass by pass: a hand that went 40 cm forward past the bound of the second joint and 15 cm to the left took the tip 13.5 cm forward and 13.9 cm to the left along that bound, with the notes `at the workspace limit: front; at a joint limit: joint 2` and `at a joint limit: joint 2`, no joint target outside the joint box less its 0.01 rad; under a stand-in that names no tip box, a hand that went 45 cm forward, 20 cm to the left and back to 10 cm forward took the tip 31.3 cm forward and more than 10 cm to the left with the note `out of reach`, and was at the hand's pose again within 2 mm and 2 mrad, with no new squeeze. Nothing ended an engagement and the stand-in refused nothing. Not run on the arm or with a headset |
| The rig file's human joint box | By tests, in the test process, on a fake clock (`test_rig_configs.py`): `configs/robot_app_rebot.yaml` with human driving switched on in memory, once on the simulated arm and once on APAI's checkers of the rig's arm profile, laid out as the launcher runs it, and APAI's driver class over a stand-in for the motor library. Each pose was placed, not driven | The box lay 0.02 rad inside `rest.region_*` on ten bounds and on the region's own -0.05 at the lower bound of the second and the third joint. At each of the ten: with a joint 0.005 rad outside its bound at a STOP, a takeover and an agent's task were refused, nothing was sent, and the typed `rest` with `path is clear` returned the arm to rest, where a takeover was taken; with a joint 0.01 rad outside its bound and nobody as control owner, the takeover was refused and the typed `rest` returned the arm. At the two bounds without the margin, with a joint 0.005 rad below its bound at a STOP, the takeover and the typed `rest` were refused, nothing was sent and the motors stayed on |
| The retargeting graph | By tests, on Isaac Capture's retargeters with controller snapshots the tests make (`test_human_retarget.py`) | The target followed the hand in the robot's base axes; a positive `stage_yaw_deg` turned it counter-clockwise seen from above; a pass 40 ms, 100 ms or 1 s late got the motion of one pass; the buttons were read while the controller's pose was not valid |
| The IK | By tests, on APAI's arm model (`test_human_ik.py`) | From the rest pose a lift solved on every pass; forward, back and down had no solution; a jump was refused; no random restart was used. On the model 1 m/s at the gripper tip needed up to 10.1 rad/s at a joint away from rest and 12.9 rad/s at the pose a drive from rest starts from. For a hand beyond the reach in four directions the solver's nearest pose took the tip 9.6 to 13.2 cm further than the last pose that solved, with the wrist 0.20 to 0.28 rad off the hand's, and then stood; a hand that came back into reach was solved for again; a nearest pose kept each joint inside the bounds it was given, moved no joint further than `ik.max_jump_rad` and took the tip no further from the position asked for. With the rig's human file a pass at the hand speed limit was no jump, and the tip moved at 27 to 36 cm a second from the ready pose and from a stretched pose and at 16 to 32 cm a second from a pose folded near the base |
| The human program's logic | By tests, on stand-ins for the graph and the solver (`test_human_logic.py`) | Right B was a STOP on every pass with known buttons and one STOP per stretch; B held 1 s with the grip released asked for the takeover once, after the ROBOT_STATE that showed its STOP; a button down before a data loss counted only when pressed again; a jump, five failed solves and a target too far ahead each ended the engagement; the note named a button that is down with no press that can ask, and loaded jaws at rest; a hold of right B or of right A did not count across a stall of more than 1 s and ran on through one of 1 s or less; right B's STOP went out on a pass whose request failed, and an error before the buttons were read sent nothing and left the logic as it was; during a return to rest with a takeover open and the stage `ready` the note read `the arm returns to rest; right B is a STOP`, ahead of a refusal's note, and no link message differed but in its note; every HUMAN_SETPOINT stayed under 600 bytes as it is sent |
| The human program and its two controller feeds | By tests (`test_human_feed.py`, `test_human_program.py`): the script feed with the real retargeting graph, solver and logic, pass by pass against a stand-in for the Thor; the teleop feed on a stand-in session and on a real remote session against a server in miniature in the test process, on 127.0.0.1 and a port the system picked; the program's `main()` on a thread behind a robot link service whose robot side the test scripts, and once against the robot app on the simulated arm over a real link; the program as a process for SIGINT and SIGTERM and for a standard output nobody reads; `bash run_human.sh` as a process for its variables | Right B gave one STOP from `controller` and then the takeover; a squeeze and a lift of 2 cm raised the simulated arm's gripper tip; the program's end while the person owned the arm sent one STOP from `human_program`, on a signal, when the attempts ran out and after a failure; SIGINT and SIGTERM ended it with exit code 0; the causes of exit code 1 and 2 in the table gave their code and their line (of a restarted robot app only the synthetic controller's case was run); a standard output that took nothing held up neither the passes nor right B; right B's STOP went out ahead of the log line of its pass, and a log line that blocked on an earlier pass held up right B until it returned; with the log and standard error both blocked, the program's own STOP went out ahead of the lines of its end (an error, the attempts ran out, a new session it must not drive), and right B read on the last pass for such a session was a STOP from `controller`; right B was a STOP on a pass whose request could not be built; a `rates.loop_hz` above 100 was refused, and so were a number too large to use and a file that is not UTF-8, each with exit code 2 and one line; at 100 passes per second uneven passes raised the hand speed limit by 0.5 % and by 5 %; a pass that failed in the logic did not end the program; the feed gave the stages `no_server`, `waiting_for_headset`, `no_controller` and `stale`, and entered a new session after the server had closed |
| The role `human` and the typed fallback's switch in the robot link service | By tests (`test_link_service.py`, `test_local_api.py`) | The role sent its three link messages and nothing else; a second human program was refused; a HUMAN_SETPOINT went to the Thor newest-wins and stayed at the service while a STOP was not confirmed; the console's `takeover` and `handback` were forwarded only with `operator.typed_takeover` on; a human program that left sent nothing to the Thor |
| The takeover line, the status line's human parts, the bells, `takeover` and `handback` at the console | By tests: the console's class with its lines typed by the test (`test_console.py`, `test_console_cmd.py`) | Each text of the takeover line, among them the one for loaded jaws at rest; the note shown only while a stage is reported; the console's last line with and without a robot connection; one, two and three bell characters; a refused takeover rang through a flood of other refusals; the service's refusal was shown with `use right B / right A on the controller` |
| The task gate with a takeover open | By tests (`test_task_gate.py`) | No task was started while a takeover was open; the robot app's refusal was printed with its remedy |
| The view token | By tests: the viewer's model on times the tests pick (`test_viewer_live_model.py`), and over HTTP against the viewer's request handler (`test_viewer_live.py`) | An acknowledgement without the page's token, of a picture the page never fetched and of a replaced page got 403 and changed nothing; a reload made the view not live until the new page acknowledged; a tab hidden too long after its fetch was not live; a `visible` handled after a `hidden` for the same fetch got 403 |
| A page open across a restart of the viewer, and a page that a newer load replaced | By tests (`test_viewer_live.py`), on 2026-11-17: the page's script under node, on a stand-in for the page; the viewer's request handler over HTTP; headless Chrome on the launcher's real page, with the launcher on a thread, on a stand-in camera and a stand-in for the robot link service, ended with every connection the browser had open and started again on the same port | While the viewer was gone for 5 s the page kept its last picture, dimmed, with `the viewer cannot be reached; trying again every second; this picture is N s old`, N growing to 5, and did not load itself. It loaded itself once for each of two restarts: the live picture showed 0.2 to 1.2 s after the new viewer listened, and VIEW_STATUS was live 3.2 s after, when the new viewer's 3 s had passed. A page that a newer load replaced went on drawing 30 pictures a second with `watch only: another page on the workstation is the live view; ...`, sent no acknowledgement, and the robot was told what the newer page showed; once that page asked no more, the note read `watch only: no page is the live view now; ...` within 3 s and no VIEW_STATUS was live. A viewer whose every page named another instance than its own: its watch-only page was loaded twice in all, and the second load found the instance in the tab's history entry and stayed; its page on the workstation, which had that viewer's view token, did not load itself, said `watch only: no page is the live view now; ...`, showed the pictures to watch and never counted. Under node: one load for each viewer instance, and none for a hidden tab, for a replaced page, while a page of the new viewer had its view token or while the viewer gave no answer; a page that came out of its one load and was still not of the viewer said `watch only: the viewer was restarted; ...`. Not run: with a person at a browser window, with the viewer as a process of its own, on the rig |
| The recipe of demo 1 with the mock agent | By a test, as processes (`test_demo1_recipe.py`): the robot app on the shipped `configs/robot_app_mock.yaml`; the agent program started with `bash run_agent.sh` at a pseudo-terminal that is its controlling terminal, the task lines and each `yes` typed there; the console on a pseudo-terminal. What differs from the recipe: the service and the agent program ran on copies of their files that name a port the system picks, a socket and a state folder in a temporary folder; the console was started with `python -m`, not through its start script; a local client in the role `tool` read ROBOT_STATE; APAI's stores lay in the temporary folder and `APAI_GRASPGENX_PORT` named a port nobody listens on | Every "must show" of the recipe. `open the gripper` succeeded; the pick and place ran 8 moves under permits and ended with the help request, the arm holding at about [0, 1.2, 1.2, 0, 0, 0] rad with its motors on; `stop`, `rest` with `path is clear` and `shutdown` with `yes` worked; exit code 1 for the agent program, 0 for the others. The process test takes 53 s |
| The recipe of demo 2 on one machine | By a test, as processes (`test_the_recipe_of_demo_2_runs_as_processes_with_the_shipped_files` in `test_demo2_programs.py`): the robot app on the shipped `configs/robot_app_demo2.yaml` with `--arm sim`; the human program started with `bash run_human.sh --controller synthetic:lift`; the console on a pseudo-terminal. What differs from the recipe: the service and the human program ran on copies of their files that name a port the system picked, a socket in a temporary folder and, as the Teleop Session Server's, a port nobody listens on; the console was started with `python -m`, not through its start script; a local client in the role `tool` read ROBOT_STATE | Every "must show" of steps 1 to 7, those of steps 5 and 6 in the recipe's order. The script's right B was a STOP and took the arm over; the gripper tip rose 5 cm off rest, within 3 mm; the typed `rest` returned the arm in 5.0 s and the script's right A gave it back; the owner numbers ran from 0 to 5 up to the hand-back; `shutdown` with `yes` released torque at rest; exit code 0 for all four programs and no traceback from the human program. The test takes 23 s |
| The recipe's variant with the agent failing first | By a test, as processes (`test_demo_2_with_the_agent_failing_first_runs_as_processes`): the robot app on the shipped `configs/robot_app_mock.yaml` with `--arm apai --driver mock`; the agent program started with `bash run_agent.sh` at a pseudo-terminal that is its controlling terminal, on a copy of `configs/agent_mock.yaml`; the human program started once the agent had asked for help. The other differences are those of the row above | The pick and place ended in the help request; the help block ended with `takeover: not possible: no human program is attached: start run_human.sh`; the script's right B cancelled the assistance wait and took the arm over with owner number 4; the gripper tip rose 5 cm from where the agent had left it; the typed `rest` returned the arm in 30.9 s (largest joint travel 1.48 rad); after the hand-back `open the gripper` succeeded under owner number 8; the console printed one `WELCOME`; no `open` was typed; exit code 1 for the agent program, 0 for the others. The test takes 72 s |
| A person driving, with every program of demo 2 attached, on the simulated arm | By tests, ten of `test_demo2_programs.py`, in the test process and on the machine's clock: the robot app behind the robot link service on 127.0.0.1; the human program's `main()` on a thread with script rows the test gives; the console's class with its lines typed by the test; the viewer's live view (`live_view_bench.py`) on threads with stand-in camera pictures (15 per second, a VIEW_STATUS every 0.1 s) and an HTTP client that fetches a picture and acknowledges it with the page's token every 0.25 s; `human.view_age_limit_s: 2.0` in the robot app | `stop` typed at the console during a squeeze held the arm within one setpoint period, with nobody as control owner and the takeover open; the hand, still squeezed and rising, moved nothing more; an agent client's task was refused with `takeover not finished: give the arm back with right A at rest` and was taken after `rest` and right A. Right B pressed during a squeeze and held for 2.2 s was one STOP and asked for nothing. A controller that was absent for 1.5 s and a browser tab that stopped acknowledging each ended the drive in a hold with the person still the control owner; with the cause gone and the grip still squeezed nothing moved, and after a release and a new squeeze the arm followed again under the same takeover. With the human program's socket closed the last joint target went out within `human.sample_age_limit_s` plus one period of the last human target, the program ended with exit code 1 and the console showed `no human program` beside `owner human` and `takeover open`. With the agent client's socket closed during a squeeze the drive went on: no help request and no assistance wait. With the robot connection cut during a squeeze the arm held and the person kept it; `rest` typed at the Thor terminal returned it meanwhile; after the reconnect the squeezed grip moved nothing until it was released and squeezed again. With the tab away a second time, right B was a STOP and its hold was refused with `the live view is not fresh: ...; show the viewer's live picture in a visible browser tab` and three bells; so was a takeover with `--source none`, where nothing moved and no takeover was open. Right B during the assistance wait cancelled it: no return to rest started, with `rest.auto_return` true. With `operator.typed_takeover` on and no button pressed, `stop`, `takeover`, `yes`, a squeeze and a lift of 2 cm, `rest`, `path is clear`, `handback`, `yes` ended with `takeover ended` and `owner nobody`; with the setting off the same words printed the service's refusal and changed nothing |
| Demo 2 with the agent program, and a fast hand, on APAI's software arm | By tests, two of `test_demo2_programs.py`, the same way, the arm raised off its joint stops: one with the agent program in the test process on APAI's mock scene and an obstacle put into the obstacle grid during its first move; one with a hand that is 1 cm higher on the next pass and a human speed limit of 0.3 rad/s | The first task ended in a help request; the help block showed `jaws: empty` and ended `takeover: possible now: hold right B for 1 s with the grip released`; right B's STOP cancelled the wait and its hold took the arm over with two bells; the arm followed a lift of 3 cm, each joint target within 0.98 of the human speed limit times the period of the one before; `rest` with `path is clear`, then right A, gave the arm back; the second task, `open the gripper`, succeeded. The control owners in order: nobody, agent, nobody, human, recovery, human, nobody, agent, nobody. One run of every program on one session of the robot app, the motors on throughout, no jaw command from the takeover to the hand-back, every joint target of the agent under a permit. For the fast hand the human program's joint targets moved at 0.9 of the robot app's speed clamp while the hand was ahead and never faster; `health.clamped` rose only by the steps at the clamp, which come where two passes of the human program fall between two setpoints. No drive ended, APAI's human checker approved every target and the gripper tip ended 1 cm higher |
| The two robot app files of demo 2 | By a test of `test_demo2_programs.py` that needs no APAI | `configs/robot_app_demo2.yaml` and `configs/robot_app_human_check.yaml` differ from `configs/robot_app.yaml` in exactly the keys their first comment lists; both load; each is refused with a human joint box wider than `rest.region_*`; the recipe's file asks for no live view, and the robot app's line about that is due for it; the file of the headset checks requires the live view, as the rig's file does. Two more tests of the file check its own waits and its reading of the help block. The file has 18 tests and takes about 4 min with everything present |
| The human program at the robot app's own socket, as on the Thor | By tests (`test_human_on_thor.py`), on one machine. In the test process, on the simulated arm and a fake clock: a robot app with its socket in a temporary folder and a local client of the test in the role `human`; once the robot app's program on a thread with a dialer of the test. Three tests as processes: the robot link service, the robot app with `--arm sim`, `configs/robot_app_demo2.yaml` and `--human-socket`, the console on a pseudo-terminal and `bash run_human.sh --controller synthetic:lift` on a copy of `configs/human.yaml` that names the socket; and twice the robot app with a robot connection that nothing answers, on `configs/robot_app_human_check.yaml` as shipped and on a copy that requires no live view | The client got WELCOME, ROBOT_STATE and the RESULTs of its own requests, and `health.human.local_program` said whether it was attached; a takeover and a hand-back asked for at the socket were answered as on the robot connection; a second human program and every other role were refused; a client that did not read was cut off, and no send of the robot app took a quarter of a second meanwhile; of two human programs one was followed, and a second one that kept sending was told of once; a STOP at the socket was served ahead of requests that had arrived before it; a STOP behind a refused LOCAL_HELLO (the console's raw STOP, a second human program's) reached the robot app; a STOP sent while the arm was being connected held no HUMAN_SETPOINT back afterwards; a `~` in `--human-socket` named the path of `configs/human_rebot.yaml`; the socket's file had mode 0600 and went with the robot app. With the robot connection lost during a squeeze the drive ended in a hold and the person kept the arm; a new squeeze moved nothing, for want of a live view, in a robot app whose file requires it; right B's STOP was taken. As processes: right B, the takeover, a lift of 5 cm, the typed `rest` and right A ran with the console showing `human program at the robot app`; with no robot connection at all right B's STOP reached the robot app. On `configs/robot_app_human_check.yaml` as shipped, which requires the live view, the takeover it asked for was refused with `the live view is not fresh: no VIEW_STATUS has arrived; show the viewer's live picture in a visible browser tab`. On a copy of that file with `human.require_live_view: false` the takeover was taken, the squeeze lifted the arm, and a `rest` typed at the Thor terminal and right A ended the takeover. The three process tests take about 50 s |
| The jaws under a person's trigger | By tests (`test_human_jaws.py`), in the test process, on the simulated arm and a fake clock, with the human targets the tests make; one of them on APAI's software arm | A held object stayed held through a takeover with the trigger released; a trigger short of the grip level moved nothing; once it had reached the grip level, or crossed it, the jaws followed it, the grip squeezed or not; nothing closed beyond the grip level before a measured full open; after one, a close on an object ended in the light hold; a STOP, stale controller data, a lost robot connection and a drive that the grip did not end each ended the jaw motion in a light hold; without a fresh live view the jaws opened and did not close; with `human.trigger_jaws: false` the trigger did nothing; a jaw opening outside 0 to 1, or no number, was refused; without a known grip level the trigger moved nothing; the hand-back was refused until the trigger had opened the jaws fully |
| The trigger in the human program and on the console, and a person opening the jaws an agent left closed | By tests: the logic on stand-ins (`test_human_logic.py`); the script `grasp` on the real retargeting graph, pass by pass against a stand-in for the Thor (`test_human_feed.py`, `test_human_program.py`); the console's jaws line (`test_console.py`); and one test of `test_demo2_programs.py`, in the test process and on the machine's clock, on APAI's software arm with the robot link service, the console, a live view of the viewer's parts and the human program attached | The jaw opening was 1.0 minus the trigger's value, the grip squeezed or not, fully open at or below 0.05 and closed at or above 0.95; no jaw opening went out on a pass without a usable trigger, or while ROBOT_STATE did not show the person as the control owner; the trigger changed nothing else of a HUMAN_SETPOINT; the note said which way the trigger goes while the arm followed the hand, behind every note about the arm; `grasp` asked for closed and then for open jaws and moved no joint; the console printed the jaws line below its status line. With the programs together: an agent client closed the jaws on an object half the jaw travel wide and asked for help; through the takeover and a lift of 1 cm with the trigger released the jaws got no command, and the note read `pull the trigger to the grip level to move the jaws`; the trigger, pulled all the way, reached the grip level 0.00 and moved nothing; released, it opened the jaws in ten steps of 0.05 until they were measured fully open and empty; after the typed `rest` right A gave the arm back. The test takes about 14 s |
| A person's drive close to the table, and a takeover with the gripper tip under a person's table clearance | By tests, on 2026-11-18. `test_lift_out.py`: in the test process, on a fake clock, on APAI's checkers and its software arm with APAI's `mock` profile and `limits.human.table_clearance_m: 0.0`, as on the rig. `test_human_program.py` and `test_human_logic.py`: the human program's graph, solver and logic on the rig's human file, pass by pass against a stand-in for the Thor that names the rig's tip box, table clearance and joint box | On APAI's checkers: a takeover with the tip 1 cm over the table was told as any other, and squeezes took the tip sideways there, straight down to 2 mm over the table plane, sideways there and up again; each drive began with a setpoint and APAI's human checker took every one. A target under the table plane was followed down to the plane and then refused by APAI's checker (`TCP z=... below table clearance ...`); the arm held over the table, and sideways and up were taken from there. At the mock profile's cap of 1.2 rad/s no setpoint ended under the table plane. Inside the agent's last grasp region a person lowered the tip as anywhere. On an arm made to settle 3 mm under every target, every pass of a squeeze with a still hand sent its setpoint, and a rising hand was followed from its first pass. A takeover with the tip under the table plane was granted, its line ending `; the gripper tip is low: only a lift is taken until it is clear`; a slide there was refused and a lift was followed. With another complaint of the human checker the takeover was refused. The typed `rest` stayed refused under 2 cm and inside the last grasp region. The agent's and the recovery checker kept their 2 cm and the profile's box, and WELCOME named the person's floor and box. The simulated arm's checkers knew no table. On the rig's human file: a hand pushed 10 cm down in 0.2 s and 15 cm to the left from a pose with the tip 2.7 cm over the table brought the tip down to 4.97 mm and slid it more than 10 cm along the stop, with the note `at the workspace limit: table` and no joint target under the stop; a squeeze with the tip 2 mm over the table and a level hand took the tip 10 cm to the left and up to the stop from the first pass the Thor followed. Not run on the arm or with a headset |
| Every stretch of time in a sample age and in a view age | By tests (`test_freshness.py`), in the test process, on the simulated arm and a fake clock, with the test's client in the role `human` once on the robot connection and once at the robot app's own socket; one test runs the human program's `main()` | Controller data that were 0.14 s old when sent and 0.12 s on the way started no drive: no joint target went out, the person kept the arm, and `health.human.refusal` read `no fresh controller data: 0.26... s old, the limit is 0.25 s`. A takeover was refused just outside the limit and taken just inside it, wherever the time lay: at the human program, on the way, or counted from the ROBOT_STATE for a sender without `sent_mono_s`. The wait at the robot app counted too. A drive ended in a hold when its targets, or the view's reports, took too long on the way. A HUMAN_SETPOINT that names no time had no known age; a time from the future was not trusted. A VIEW_STATUS delayed on its way was as old as it was, and one that does not say when it was sent was not fresh; the view age at the Thor was the picture's age whatever the link took. The human program stamped each HUMAN_SETPOINT with the age and the time of the moment it sent it; a takeover whose joint read let the controller data or the live view run out was refused with `...; the joint read took ... s`; the time of the controller feed's own step, for the script feed and for the teleop feed on a stand-in session, was in the sample age the human program sent |
| A torque release whose driver call fails | By tests (`test_release_failed.py`), in the test process, on a fake clock, on the simulated arm and on APAI's mock driver | The RESULT was `failed` with `fault: torque release failed, the motors may be on or off: ...`; the motion was `release_failed`, `torque_on` true, and a fault-stop latched; no line said that the motors are off. After `reset` a task, `open` and `rest` were refused with `a torque release failed and the motors may be on or off; ...` and nothing was sent to the arm or the jaws. A shutdown started no return to rest, tried the release again and left the robot app running while it failed; once the release worked, the same shutdown closed the app. A second release at measured rest ended `rest_unpowered`. An arm port that knew its motors were off was shown as `rest_unpowered`. With APAI a release that failed after the driver had disconnected was not shown as off, and the arm port counted a release only once the driver's call had returned |
| A jaw command the agent gave up on, and a GRIPPER's expiry | By tests (`test_gripper_timeout.py`): the real robot app on the simulated arm with the agent's arm adapter, its wait shortened to 0.3 s, on a robot connection that keeps link messages back; the arm adapter against a scripted robot side; the robot app on a fake clock with the test's client | A GRIPPER that arrived after the adapter had given up was `refused` and the jaws got nothing; a close that ran when the adapter gave up was cancelled and left in the light hold, never opened; the adapter sent one STOP from `agent`, also inside a skill, and its next jaw command sent nothing; a jaw command answered in time sent no STOP and the agent kept the arm. A GRIPPER past its expiry was refused at once with `the jaw command expired ... s ago`; one that expired while it waited never started, and the commands before and after it ran; one inside its expiry ran as one without; a GRIPPER without the field never expired; the agent program's GRIPPER carried the end of its own wait |
| A MOVE's expiry at its first setpoint, and the read a move completes on | By tests (`test_executor.py`), on a fake clock with joint reads the tests script | A MOVE that expired during the check of its first setpoint was refused with `the move expired 0.030 s ago` and sent nothing, with and without a permit held; one whose check also made the setpoint late was refused as expired, not ended `late`; a STOP during that check won; a slow check that ended before the expiry started the move, and an expiry that passed in a later check did not end a move under way. A joint that read near and read not valid later did not count; a full read with a joint not valid, a failed one and one with a joint 0.05 rad off did not complete the move, which completed on the next full read; full reads that stayed bad ended it `not_settled`, the arm holding at the last setpoint with its motors on; a move that completed made one full read after its last setpoint and took as long as before |
| The arguments of the experiment's launcher in the agent program | By tests (`test_agent_options.py`), in the test process; one of them with the experiment's own timing, model transcript, debug recording and milestone batching on APAI's mock scene, with no GPU, no model request and no camera | The full stack timed and transcribed by default, and the minimal stack did neither; the launcher's command line with `--operate` and `--depth-backend ffs` gave the run that the line without them gives; each argument a run cannot take ended the program with exit code 2 and one line, `--motion-approval tool` on both stacks among them; without an option nothing of the experiment was loaded and no file was left; each option installed its hook once, in the launcher's order; a STOP and the close of the link came before any recording ended; a recording that could not begin ended the program before it attached, and one that could not end left the run's exit code as it was. On the mock scene every timing hook of the experiment installed. A run wrote `resolved-config.yaml` before it attached, and one that could not write it ran on with a warning; with the debug recording each pose the program read from ROBOT_STATE was recorded as `robot.feedback` |
| `bash run_workstation.sh --print`, and the script with an unknown argument, without a variable and with a Python that cannot be run | By a test, as processes (`test_link_programs.py`) | Exit code 0 and the five commands of the five panes, nothing started; exit code 2 and one line each |
| `bash run_workstation.sh` with a stand-in for tmux | By tests, as processes (`test_launcher.py`); no tmux ran | Five panes. The viewer's command was sent at once; the console's only once a service listened on the local socket. `--live-view` was refused as an unknown argument |
| A person on the real driver, and the live view | By tests. The program with `--driver real` on APAI's driver class over a stand-in for the motor library, on a copy of the rig file with `human.enabled: true`, in real time (`test_real_driver.py`). The same driver class on a fake clock (`test_rig_configs.py`). The simulated arm with the driver declared `real` (`test_takeover.py`). The viewer's launcher on a stand-in camera, with the robot app on the simulated arm (`test_viewer_live.py`) | On the rig file as shipped, without a VIEW_STATUS, the takeover was refused with `the live view is not fresh: no VIEW_STATUS has arrived; show the viewer's live picture in a visible browser tab`. With a live VIEW_STATUS that a test sent, a takeover was taken and a lift ran on the driver class. With the viewer's own VIEW_STATUS, from a stand-in for a browser tab that drew and acknowledged its live pictures, a takeover was taken and a squeeze moved the simulated arm; the drive ended in a hold when the live picture left the page. On a copy with `human.require_live_view: false` the program printed `robot app: human driving is not tied to a live view (human.require_live_view false)` and logged it, a takeover was taken and a squeeze moved the stand-in motors, the released grip ended the drive in a hold, and right B was a STOP. Stale controller data, a synthetic controller, a grip that was not released first and a target beyond the step limit were refused either way. Never run on the arm |
| The experiment's viewer on runs the agent program recorded | By tests (`test_viewer.py`), on 2026-11-16: the viewer's own HTTP server in the test process, on 127.0.0.1 and a port the system picked, on runs recorded on APAI's mock scene with the experiment's own timing, model transcript and debug recording and a stand-in for the obstacle grid bridge; once as a process, started with `bash run_viewer.sh --runs FOLDER --port 0 --token-file FILE` under the tests' Python | The run was listed before its first task; while the program waited for task lines the events, the length and the map revisions grew, and the measured joints went from the start pose to where the pick and place left the arm; the timing gave the first task's spans while the program still ran; after its end the recording was closed and an earlier moment showed again. A refusal of the check ahead showed with the refused pose, its point at the gripper tip and the map revision it was checked against. The page, the camera frame as a JPEG, the timing page and its data answered with code 200. A run without `--record-debug` and a run folder without `resolved-config.yaml` were not listed, and asking for them by name gave code 400. As a process: code 401 without the token, the run's data with it, exit code 0 on Ctrl+C. The full stack's `resolved-config.yaml` gave the viewer the arm and the volumes `housing`, `fingers` and `jaws` |
| `run_viewer.sh` with stand-ins for the Python and the experiment folder | By tests (`test_viewer.py`), as processes, on a copy of the script in a temporary folder | The viewer got `--host 127.0.0.1`, `--runs` and `--token-file` in the copy's own `.state` and the caller's arguments behind them; the state folder was made with its `.gitignore`; eleven things that cannot be used each gave exit code 2 and one line, with nothing started |
| The debug recording's map pictures a second and the line of a full recording | By tests (`test_debug_recording.py`), on 2026-11-17, in the test process: APAI's own watcher and obstacle map on a stand-in camera of 64 x 48 pictures, 20 a second, and a stand-in grid bridge, with the experiment's own recorder; the experiment's viewer over HTTP on a port the system picked, and its report; two runs of the agent program on APAI's mock scene. Not run on the rig: the 4.7 MB and the 7 a second are read from the rig's recording of 2026-11-17, and the 30 min are computed from them | Of 20 refreshes a second about 2 were stored, each with its frame, its input, its bridge calls, its commit and its end, while the map took every one; the viewer's state went on while the run recorded, and the viewer and the report read the result. With `--debug-map-hz 0 --debug-at-limit stop` the recording had the events of the experiment's recorder alone, with the same ids. A refusal and a route segment were stored with the map refresh they named, also when another thread named it while its tick still ran; a refresh with a `retire_target` call was stored. After half a second without a new map revision the newest map that had been left out was stored, and so was the last one when the recording ended. The line `debug recording full after ...` was said once when the recorder stopped and once more at the next task prompt, and both tasks behind it ran |
| The debug recording at its size limit | By tests (`test_debug_recording.py`), on 2026-11-17, in the test process, with limits of 2 to 16 MB: the same watcher, map, recorder and viewer on a stand-in camera of 320 x 240 pictures that do not compress; the experiment's recorder alone on made-up events; two runs of the agent program on APAI's mock scene. Not run on the rig, and with no limit of gigabytes: the 30 min, the four hours of map points and the 1 % an hour of rows are computed from the rig's recording of 2026-11-17 | With 74 MB of map pictures into a limit of 12 MB the recorder never stopped and its folder stayed under the limit; `events.jsonl` was only appended to; the viewer showed the newest map, its points and its picture at every look, also over HTTP while 20 map pictures a second were recorded. The pictures of the first map, of the maps two refusals and two route segments named, of a refresh with a `retire_target` call, a grasp's mask and a measurement were all there at the end; a time in the dropped stretch gave the map and its points with code 400 for the picture, and the live state after it. A burst of 300 map pictures did not stop the recorder. A full disk (`No space left on device`) stopped it and was said; pictures that could not be removed let it stop at its size as it does alone. `recording-status.json` carried both blocks and `complete: false`, and the checks of the four replay tools refused the run; two of the tools were run themselves, the other two need `msgpack`, which the test interpreter lacks. With `--debug-at-limit stop` the recorder stopped at its size as before. The agent program sent the arm the same 905 setpoints, jaw commands and moves while its oldest map pictures were dropped as with the experiment's recorder alone |
| The live picture of the experiment's viewer | By tests (`test_viewer_live.py`), on 2026-11-17: the launcher's request handler in the test process, on a stand-in camera of 60 pictures a second; the whole launcher as a process, started with `bash run_viewer.sh`, on a fake camera receiver with the real camera frame bridge program on files in the place of the GPU's slots and camera frames of 2560 x 1984; the page's block by node, on a stand-in for the page, behind the page's own script and over HTTP against the viewer | At `--live-fps 30` the stream gave 20 to 30 different pictures a second, each newer than the one before, and 4 to 6 at 5; with no browser on the stream or asking for pictures the camera was asked once a second. A browser that read one picture in 0.3 s missed most and slowed no other. Every path of the viewer answered with the same bytes as the viewer alone, the page with the block before its `</body>` and nothing else; the four live paths answered 401 without the token. As a process: pictures 960 px wide, no file changed in the runs folder or beside the token, exit code 0 and no bridge process left after Ctrl+C, SIGTERM and SIGHUP; without a camera one line `viewer: no live picture: ...` and the runs served; with `--no-live` the experiment's own page and code 400 on the live paths |
| The live view: the viewer's page with its live picture, and the viewer's VIEW_STATUS | By tests (`test_viewer_live.py`), on 2026-11-17. The page's script under node, over HTTP against the viewer's request handler on a stand-in camera. The viewer's link on a stand-in for the local API and a clock that the test set. The viewer's launcher on a thread, on a stand-in camera, attached to a robot link service that ran as a process, with the robot app in the test process on the simulated arm and on the rig's robot app file with `human.enabled` switched on in memory; a stand-in for the human program sent 50 HUMAN_SETPOINTs a second, and a stand-in for a browser tab fetched and acknowledged the pictures. The launcher as a process, started with `bash run_viewer.sh`, without a robot link service and then with one | The script drew and acknowledged 30 pictures a second, and the acknowledged picture was about 0.05 s old; after `Map picture` the view was not live at once. The viewer sent the VIEW_STATUS its rule gives in 18 situations, among them a page that only fetched, a hidden picture, a newer load of the page, a picture with invented times, and no word for 1.1 s. On the rig file the takeover was refused without a page that showed the live picture, taken with one at a view age of about 0.12 s, and a squeeze moved the arm; the drive ended in a hold and a new takeover was refused after each of: the page's word that the live picture shows no longer, a tab that said nothing more, a stopped camera, the viewer's end, and a load of the page elsewhere. Torque stayed on throughout. Without a service the launcher served everything and said one line; it attached by itself once the service ran. No browser, no camera and no arm were used |
| The live arm of the experiment's viewer | By tests (`test_viewer_live.py`), on 2026-11-17. The launcher's request handler in the test process, on a stand-in for the robot app that gave the viewer's link 20 ROBOT_STATEs a second with a moving first joint; once behind a robot link service that ran as a process, with the robot app on the rig's file and the simulated arm. The page's own script with the block under node, on stand-ins for the page's elements: with the harness's clock, and in real time over HTTP. Headless Chrome on the launcher's real page, on a run recorded on APAI's mock scene and on a run folder without a recording | Over HTTP a client that asked again at once got every state in its order, and one that asked every 0.3 s the newest, at most one behind. 401 without the viewer's token; a client of another machine was answered. The arm's points were those of the viewer's own `RunIndex.geometry` for the same joints. The viewer's own paths gave the same bytes as the viewer alone; no file changed; only VIEW_STATUS was sent. Under node the page's own `draw()` was given 19 to 20 different live states a second while following and none through the slider, a replay, an older run and a moment of the timing graph. In Chrome: 20 different live states a second and 24 draws a second while following a finished recording, with the line `arm: live, 20 a second; map: 2.1 s old` and the map's age growing to 5.8 s in the next seconds; 20 a second without any recording, `map: no recording`; the recorded arm alone in a replay; `arm: no live state: the robot link service is gone (the service ended)` within 2.5 s of the stand-in service's end. With readings that stood still, 20 states a second and 45 s old (motors off; joint feedback lost): no answer of the path, no line under node and no line in Chrome said live; Chrome showed `arm: last read N s ago (torque off: the joints are not read)` with N from 46 on and drew the last read pose. An answer was with a client 0.2 to 0.6 ms after the viewer had the state. Not run: with a person in front of a browser window, beside a recording that is still written, on the rig's robot connection |

Measured once in that environment, apart from the tests: one pass of the retargeting graph
took 0.21 ms, an IK solve that converged 0.11 ms and one that failed after 50 iterations
0.82 ms.

The changes of 2026-11-16 to rest, release and the motors, on the workstation, that day,
in the test folder's environment, with an Isaac Capture build named. No arm:

| What | Run as | Result |
|---|---|---|
| `robot_app --arm sim --selftest-link` | By a test, in the test process (`test_link_programs.py`) | Exit code 0, 11 steps; the console's `rest` ended with the motors off, released once, at measured rest |
| SIGTERM to the robot app's process, with the simulated arm at rest and raised | By tests, as processes behind a robot link service on a port the system picked (`test_link_programs.py`) | At rest: `[jaws] at rest: the jaws are closed`, `[release] torque released at rest: the motors are off`, then `robot app: SIGTERM; closing; the arm lies at measured rest: the motors are off`; exit code 0. After a move of the agent role away from rest: `robot app: SIGTERM; closing with the motors left on` and the line that says why; no line of the jaws or of a release; exit code 0 |
| `rest`, `on`, `release` and `quit` typed at the robot app's process on the simulated arm | By a test, as a process with its lines typed into a pipe and no robot link service (`test_link_programs.py`) | The lines of [What the operator's commands do](#what-the-operators-commands-do), in their order; `on` with the motors on asked nothing and was refused; `quit` with the motors off asked nothing; exit code 0 |
| `run_robot_app.sh` and `run_human.sh` with `--env FILE` | By a test, as processes (`test_link_programs.py`) | Exit code 2 and one line without the variables, with a file that names no Python and with a file that is not there; with a file that names a Python, `run_robot_app.sh --env FILE --help` printed the robot app's help |
| The route of a typed `rest`, the motors on again, the agent's rest tool and the jaw routines of APAI's driver | By the tests of the six conformance files and `test_rest_end_to_end.py` (see [Run the tests](#run-the-tests)), on APAI's reBot driver class over a stand-in motor layer, on APAI's mock driver and on the simulated arm | Not run by the writer of this table: their authors' runs are not recorded here |

Never run:

- The agent program with a task at a real terminal: a test runs it as a process on a
  pseudo-terminal and types the task and the `yes`; the other tests script them. No task was
  run by hand on one machine.
- The operator console and the live view by a person: no line was typed at a real terminal
  and no page was opened in a browser window. So the script of the viewer's page, which
  carries the view token, ran under node and in headless Chrome only, and no bell was
  heard.
- The recipe of demo 1 with the mock agent by hand: with the port 47610, the default
  socket path and `run_console.sh`.
- The recipe of demo 2 on one machine and its variant with the agent failing first by
  hand: with the port 47610, the default socket path and `run_console.sh`. Tests ran
  both as processes.
- A headset, CloudXR and the Teleop Session Server with any of this, and so controller
  data from a headset, right A and right B over a remote session, and a recentre.
- The human program on a Thor, and its remote session between two machines: the tests
  attach it at the robot app's own socket on one machine, with the synthetic controller.
- The trigger of a real controller on real jaws, and a person's drive close to the
  table by a hand: the stop 0.5 cm above the table plane, and the arm following a hand
  that is out of reach.
- The typed `takeover` and `handback` at a real terminal.
- Human driving on the rig, and anything of part F of the checklist. With the rig's robot
  file and the rig's arm profile laid out as the launcher runs it, a test runs a
  takeover, a lift of 20 cm, the return to rest and the hand-back on a stand-in for the
  motor library; the rig's own copy of the experiment's files was not loaded, and the
  wrist camera rule under larger motions is checked nowhere.
- `experiment.stack: full` with a task, and its cloud model, by a test: no test asks a
  model. `test_gpu_stack.py` starts the full stack on recorded stereo pairs with
  stand-ins for both model clients and runs no task; the other tests cover only the merge
  of its configuration files.
- `--camera link` through the agent program, and the
  camera receiver's start script with a receiver. The camera frame bridge is tested against
  a stand-in for the receiver.
- `depth_backend: ffs` on stereo pairs from the camera receiver: it ran on a recorded stereo
  pair only.
- `--driver real` on the arm, `run_robot_app.sh` on a Thor, and an Isaac Capture build on a
  Thor.
- The agent program's options of the experiment's launcher on the real full stack: the
  timing, the model transcript, the debug recording and the milestone batching with a
  GPU, the obstacle grid bridge and a model client.
- `run_workstation.sh` with tmux: a test runs it with `--print` only, and others with a
  stand-in for tmux. The window of five panes was looked at in no tmux.
- The live view in a browser: no browser drew the viewer's live picture, and no
  browser's acknowledgement reached a robot app. A person's takeover with the live view
  on the rig's camera, robot connection and arm.
- The experiment's viewer in a browser, on port 8766, on a recording of the real full
  stack, and in the pane `viewer`: the tests ask it over HTTP on a port the system
  picks, on recordings of APAI's mock scene. Its page's script ran in no browser.
- The viewer's live picture on the real camera stream and in a browser: the tests give
  it a stand-in camera, or a fake camera receiver with files in the place of the GPU's
  slots, and read the stream over HTTP. The page's script and the block's ran in node on
  stand-ins for the page's elements, never in a browser. How many pictures a second the
  real camera frame bridge gives, and what it costs on the GPU, was not measured.
- A torque release that fails, a jaw command without its RESULT and a MOVE that expires
  on the real arm: the tests make each on the simulated arm, on APAI's mock driver or
  on scripted joint reads.
- The rig files with their programs: only loaded and checked by tests.
- The `rest` that switches the motors off, `on`, SIGTERM at rest, the typed `open` and
  `close`, the question about the jaws, the speeds of 0.2 and 1.6 rad/s, the start-pose
  retry and the wait for a late permit, on the real arm.
- Anything between two machines, with a camera on a Thor, or on a real arm, by a test.
  The rig owner reported one run on the real arm with the two machines (see the top of
  this README).

## Next phases

| Phase | What is built |
|---|---|
| P1 | What is left of it: a task on the full stack, the camera connection from the Thor |
| P2 | What is left of it: the first run on the rig, by the checklist above, and demo 1 on the arm |
| P3 | What is left of it: the first run with a headset and demo 2 on the arm, by part F of the checklist, with the human program on the Thor; the two one-machine recipes by hand; what was cut from the human path |
| P4 | Acceptance of both demos: without hardware by the tests and the two recipes, on the rig by the checklist with every value written down. Then what the design names for P4: every fault under each control owner, measured delays, launch and recovery instructions |

[design]: ../../docs/design/claude_agentic_remote_execution.md
