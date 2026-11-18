<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agentic remote execution with two links: Claude's proposal

Status: proposal by Claude (an AI assistant), revised 2026-11-13 after the project owner's decisions of that day.
Phases P0 to P2 of section 17 are built, as example code and one library wrapper (section 5), and pass the parts of
their gates that need no hardware; phase P3 is built, and tests have run its gate on one machine with a synthetic
controller, a script that stands in for the headset's controller (section 17). The example's README
(`examples/agentic_remote/README.md`) reports what has run on the rig, the real arm on the Thor with its camera
(section 2). Sessions of 2026-11-16 and 2026-11-17 ran the programs on both machines. By the report of the rig owner,
the person who is responsible for the rig, the agent ran a pick, a lift, a hold and a place of an object on the real
arm on 2026-11-16, at 0.1 rad/s, with the agent program on the workstation and the robot app on the Thor; the
repository holds no record of that run. A run folder of 2026-11-17 was made on the rig with the agent program's full
stack (section 4), the camera connection and the robot connection between the two machines, and holds the
measurements of 14 moves and 4 jaw commands. The README marks no step of its checklist for the rig as run: no record
of the session of 2026-11-16 says which steps passed. No headset has been connected to any of it, no person has
driven the arm, and the human program, the program that turns a person's hand motion into joint targets (section
21), has not run on a Thor. Every duration is an estimate, and so is every line count that is not marked "as built".
Section 21 states where the build departs from this file in the human path (how a person drives the arm with the
headset's controller, section 2), and sections 16 and 21 list what waits for the project owner's decision.

Another proposal, written by another agent, `agentic_remote_execution.md`, exists as an uncommitted file in the main
checkout. This document calls it "the first proposal" and was compared with it as saved on 2026-11-13 at 16:22. The aim,
set by the project owner, is that the two differ in three things only: the link count (the first proposal carries robot
traffic on one connection through an extended Teleop Session Server; this one adds a second connection and leaves the
teleop code unedited), the gaps this file fills where the first proposal is silent or states only a goal, and the code
that turns hand motion into joint targets. Section 15 lists what follows from the link count. Section 16 lists the other
differences and what still needs the project owner's yes.

## 1. Goal

Today the agent stack (APAI: task loop, perception, planning and motor control) runs on the Jetson Thor that sits on the
robot, with its language model reached as a cloud service. The goal is to move the heavy parts to the workstation and
keep the Thor small:

- Demo 1: the agent on the workstation receives camera frames and robot state, and commands the arm.
- Demo 2: when the agent fails, a person is prompted, takes over with the Quest headset and its right controller (gloves
  only if the chosen demo profile needs them, section 12), and finishes the task. In the first build the person moves
  the arm, moves the jaws with the controller's trigger, brings the arm to rest and gives it back; section 20 names
  what the first build leaves out of this.

The overriding instruction is to get something running quickly and refine afterwards.

## 2. Terms

The rows "workstation", "robot app" and "teleop thread" follow this design's placement of the human path, which the
row "human path" below defines. Section 21 states where the build places it; that placement waits for the project
owner's decision.

| Term | Meaning |
|---|---|
| workstation | The GPU machine (the "server machine" of `remote_teleop_session.md`). It runs CloudXR, the Teleop Session Server, the robot link service, the agent program, the operator console and the viewer. |
| Thor | The Jetson Thor on the robot (the "robot machine" there). It owns the CAN bus of the arm: six joints and a gripper, no brakes. |
| rig | The real arm on the Thor with its camera, as the current experiment uses it. |
| robot app | The one program on the Thor that talks to the motors. New. It is also the robot app of the teleop link. |
| agent program | APAI on the workstation: task loop, models, perception, planning. |
| skill | One of APAI's ready-made motion routines, such as grasp or place. |
| LeRobot | The program that drives this arm in today's human-only remote teleop. |
| launcher | `thor_runtime.py`, the program that starts APAI on the Thor today and asks the operator's confirmations. |
| power, calibration and hold confirmations | What the robot app requires at the Thor terminal before it connects to the motors of the real arm, as today's launcher does: checks that ask nothing (the camera calibration named at the start is the one in the robot app's configuration; no hold marker, the file APAI writes when a run leaves the arm powered and holding, is on file) and three questions, each answered with a typed `yes` (section 13). |
| robot link service | A workstation program that holds the robot connection to the Thor and offers a local API. New. |
| arm adapter | A class in the agent program with the methods of APAI's arm class; it sends robot link messages through the robot link service instead of motor commands. |
| operator console | A small workstation terminal program for the operator: `stop`, `reset`, `rest`, `release`, `on`, `open`, `close`, `shutdown`, `state` (prints the newest robot state and sends nothing) and the takeover prompts. Ctrl+C there is a STOP and never ends the console; `quit` leaves the console and sends nothing. New. |
| console command | A second operator program on the workstation: it sends one command of the operator console, asks the same typed confirmation, prints the answer and ends (section 13). New. |
| viewer | The workstation program that serves the one browser page of this design and tells the Thor how fresh the live view is (VIEW_STATUS, section 6). It runs the current experiment's viewer (`thor_viewer.py`, the page that shows what a run of the agent recorded) unchanged, inside a wrapper of this example (`viewer_live`) that adds the live picture and the live arm to that page (section 13). One viewer reports at a time. |
| live picture | The left camera frame as it arrives on the workstation, made small and packed as JPEG. The viewer's page shows it and nothing stores it. |
| live arm | The arm as the viewer's page draws it from each ROBOT_STATE (section 6) as it arrives. |
| live view | The viewer's page, opened in a browser on the workstation, with its live picture showing in a visible tab. Section 13 gives the whole rule. The viewer's page is the only live view. |
| view token | A secret the viewer gives to one page at a time: the page that was loaded last from the workstation. Only that page's reports count for the live view (section 13). It is not the viewer's access token, which is the secret a browser needs to open the page at all where the viewer asks for one. |
| watch-only page | A viewer's page that has no valid view token. It shows the live picture and the live arm, and its reports count for nothing. |
| viewer instance | One start of the viewer, to its end. |
| teleop thread | The robot app's background thread: it dials the Teleop Session Server, waits for READY, then reads SAMPLES and runs human driving. |
| human path | Everything between the headset's controller and the arm's motors while a person drives: reading the controller, turning hand motion into joint targets, and checking and following those targets. |
| setpoint | One set of joint targets, one value per joint. APAI sends one every 20 ms. |
| move | A list of timed setpoints from a start pose to a goal pose. |
| checker | APAI's `SafetyHarness`: the code that accepts or refuses each setpoint just before it is sent. |
| limit set | The joint, speed and workspace limits the checker applies. The Thor configuration holds one per control owner (section 8). |
| arm profile | APAI's configuration of one arm: the arm's kinematic model, the speed cap, the workspace box, the table, the tolerance of its check that the arm lies at rest (the rest check tolerance), the jaw travel and the contact monitor's settings. |
| agent's envelope | The region the agent must stay in: on the Thor a box in joint space (section 4). |
| perception watchdog | A timer in APAI's checker: the checker refuses to start a motion when nothing has reported to it for 5 s. Each checker instance has one. |
| halt count | A counter in APAI's checker that goes up each time the checker is told to halt the motion under way. |
| obstacle grid | A 3-D array giving, for each 1 cm cell, the distance to the nearest obstacle. It covers only the gripper's workspace box (0.4 x 0.6 x 0.56 m); space outside it counts as free, today and in this design. |
| run folder | The folder the agent program writes for one run: the task record (section 13), measurements, each move's planned joint path and, where it is switched on, the debug recording. |
| debug recording | What the agent program stores of a run for the viewer's page when it is started with `--record-debug`: map pictures, the refusals of the checker in the agent program, the measured joints and each move's planned joint path (section 13). A map picture is one refresh of the obstacle grid's occupied cells together with the camera frame it was made from. |
| grasp region | A small region around the object being grasped in which the table and obstacle rules are relaxed. APAI's skills set it. |
| contact monitor | APAI's code that detects from the motors' feedback that the arm has touched something. |
| contact spot | A small zone around a place where the arm touched something. APAI's checker keeps the agent out of it until it is cleared. |
| hold | The arm stays where it is with motors on. |
| limp | Motors off. Away from rest the arm falls, because it has no brakes. |
| rest, return to rest | The folded pose in which the arm lies on its own stops; the move that brings it there. |
| recovery region | The start poses from which a return to rest is allowed: a box in joint space in the Thor configuration. It is meant to hold only poses from which the return to rest has been run on the bench without touching anything. The first value for the rig is the agent's envelope, and no return to rest has been run on the bench from it (section 19). |
| jaws | The two fingers of the gripper. |
| right A, right B, grip, trigger | Controls on the right Quest controller: two face buttons, the squeeze button under the middle finger and the index-finger trigger. In this design right B is STOP and takeover, right A gives the arm back at rest, the grip is the clutch (the arm follows the hand only while it is squeezed) and the trigger drives the jaws once it has reached the grip level (section 12 step 5). |
| jaw opening, grip level | A jaw opening is how far the jaws are open, as a share of their travel: 1.0 is fully open, 0.0 closed. The grip level is the jaw opening in force when a person takes the arm over. |
| closing stiffness, light holding stiffness | The two position gains APAI's driver uses on the jaw motor: a high one while the jaws close and a low one once they have stalled on an object, so that it is held with a light, steady force. |
| guarded opening | APAI's driver routine for opening the jaws; it protects the mechanical open stops. |
| referenced | After power-on the jaw motor knows its position only within one turn, and the jaws travel more than one turn. The driver finds the reference at start; without it a jaw reading counts as unknown. APAI's driver drops the reference after three failed jaw reads in a row, so it can be lost during a run (section 21). |
| control owner | Who may move the arm right now: agent, human, recovery (the robot app itself, for a return to rest) or nobody. |
| takeover, hand-back, takeover open | A takeover makes a person the control owner. The hand-back ends it with the arm at rest (section 12 step 7). From an accepted takeover to an accepted hand-back the takeover is open, also while a STOP has made nobody the control owner; the agent gets no new task meanwhile. |
| owner number | A counter on the Thor that goes up at every change of control owner and at every STOP and fault-stop. Commands quote it. |
| session id | A random value the robot app picks each time it starts ("executor session ID" in the first proposal; not the `session_id` of the teleop link). Commands quote it, so nothing survives a restart. |
| stop number | A counter on the Thor that goes up at every STOP and every fault-stop. Commands and requests quote it. |
| stop level | None, STOP or fault-stop (section 9). A fault-stop is latched: it stays until a person types `reset`. |
| assistance wait | How long the arm holds for a person to take control after the agent has failed. |
| permit | A short-lived permission from the workstation to send the next part of a move, tied to the obstacle grid the workstation checked that part against. |
| clock pair | One reading of the Thor's monotonic clock and one of its wall clock, taken together. Camera frames carry wall-clock times, joint readings monotonic times. |
| link, link message | An Isaac Capture connection (`core::Link`) and one unit sent on it, as in `remote_teleop_session.md`. On the teleop link, SAMPLES carries the headset and controller samples, and READY says that a headset has been tracked for about a second. |
| Headless | A checkbox in the headset's CloudXR client. When it is on, the client sends tracking and draws nothing. |
| test arm | A software arm with no hardware. There are two: the simulated arm, which the robot app brings itself with a simple joint-space checker and which needs no APAI, and APAI's mock arm behind APAI's unmodified checker (section 17). |
| declared driver | What the robot app says in WELCOME (section 6) that it drives: the simulated arm, APAI's mock arm or the real arm. A workstation program treats a robot app that declares none as one on the real arm. "Driver" alone is APAI's code that commands and reads the motors of the arm and of the jaws. |
| skill stop | APAI's own stop of the arm after one of its motions has failed, for example when the workstation's check of the part ahead refuses a setpoint during a move. It ends the task with a help request (sections 9 and 13). |
| shipped files | The example's configuration files under `examples/agentic_remote/configs/`: files for software arms on one machine and files for the rig. Every limit, speed and time in them is a first value, except the joint speeds in the rig's files, which their comments mark as the owner's values (section 8); none is measured. |

## 3. Decisions made by the project owner

| Decision | Where it shows in this design |
|---|---|
| Never go limp: no torque cut away from rest | Every failure that software can act on ends in hold or rest. Sections 9 and 10 name what software cannot prevent (the 48 V switch, a motor's own fault, the robot app's death, a power failure). Section 11 gives the only way torque is released |
| Two links for robot traffic, not one. The only differences from the first proposal: two links, the gaps this file fills, and the hand-motion code (2026-11-13) | Sections 5, 15 and 16 |
| Robot data and commands travel through Isaac Capture | Met at library level: the new connection uses `core::Link`. It does not pass through the Teleop Session Server |
| Implement automatic return to rest | Sections 10 and 11. It runs in one case: after the assistance wait. After a lost headset the owner later chose hold; after a lost robot connection it is hold as well, as in the first proposal |
| One program on the Thor for now | Section 4 |
| If the Thor program dies, hold is acceptable; the operator recovers by hand | Section 10, row "Robot app dies" |
| A missing headset must not stop the agent | The headset is needed only for a takeover |
| Once help is needed, wait 90 s or more for the headset "and only then exit" | Assistance wait, section 10: 120 s at first, never below 90 s. When it runs out the robot app does not exit: it makes one attempt at a return to rest and keeps running |
| Headset or controller lost while a person drives: hold, as in the first proposal; no timer and no automatic return to rest in the first demos. This replaces the earlier "hold, wait, then go to rest" (2026-11-13) | Sections 10 and 20 |
| Obstacle check during a move: the first proposal's permits now, as the easiest start. The obstacle grid on the Thor, a backup check on the Thor and moving checks between the machines are future work (2026-11-13) | Sections 7 and 20 |
| Hand motion to joint targets: this file's version, Isaac Capture's controller nodes with APAI's inverse kinematics (2026-11-13) | Section 12 |
| How a takeover ends: as the first proposal. The person who takes over finishes the task; the arm goes back to the agent only at verified rest (right A); no hand-back in the middle of a task. The owner said this can be changed later (2026-11-13) | Section 12, steps 6 and 7 |
| Live view: as the first proposal. No human motion without a live view in the workstation browser (2026-11-13) | Sections 12 and 13 |
| Typed fallbacks `rest`, `open`, `reset` (agreed with the takeover steps) | Section 13 |
| Holding an object indefinitely is good; never drop it, never crush it | Jaw rules in sections 9, 11 and 12 |
| The typed "yes" before motion is asked where the task was sent | Section 13 |
| Tasks from the Thor terminal as well as the workstation: undecided, opinion asked | Opinion in section 13: not in the first build |
| Human-only mode may keep today's behaviour | Section 12, last paragraph |
| Operator is in the room now; later possibly in another city | Operator actions live on the workstation |
| The camera connection (pull request #1410) is temporary | Section 5; it stays separate from the robot connection |
| Time estimates are Claude's working time, not a human's | Section 18 |
| GPU 0 only on the workstation | Section 17 |
| Implementation starts now (2026-11-13) | Section 17 |

## 4. What runs where

The diagram and the row "Headset and controller input" state this design's placement of the human path. Section 21
states where the build places it; that placement waits for the project owner's decision.

```text
            WORKSTATION                                           THOR (on the robot)
  agent program (APAI) --+
  operator console ------+-- local API --> robot link     robot connection
  viewer ----------------+                 service  <--------------------------->  robot app
                                                                                   (motor driver, checker,
  headset -> CloudXR -> Teleop Session     teleop connection (SAMPLES)             hold, STOP, control owner,
                        Server  ---------------------------------------------->    assistance wait, return to
                                                                                   rest, teleop thread)
  camera receiver  <---------------------- camera connection --------------------  camera sender
```

| Job | Workstation | Thor |
|---|---|---|
| Thinking: task loop, model calls, object finding, grasp choice, planning moves | Yes | No |
| Depth from stereo, segmentation, building the obstacle grid | Yes | No |
| Check of a move against the obstacle grid | Yes: the whole move before it is sent, then the part ahead of the arm | No; the Thor checks that a valid permit covers each setpoint |
| Camera-free checks of each setpoint just before it is sent | No | Yes |
| Motor commands, joint readings, gripper control, hold, return to rest | No | Yes |
| Control owner, STOP, assistance wait | No | Yes |
| Headset and controller input | Read by the Teleop Session Server | Turned into joint targets |
| Tasks, consent, operator console, live view | Yes | A typed `stop` and Ctrl+C, each a STOP; the power, calibration and hold confirmations (at each start of the robot app); `rest`, `release`, `on`, which switches the motors on again after they were switched off at rest (section 11), `reset`, `open`, `close` and `quit`, which is the shutdown command at the Thor terminal, under the same checks as at the console |

The robot app is one process. It runs APAI's motor driver and APAI's checker, unmodified, created without an obstacle
grid (its obstacle check is off and the permit check stands in for it); none of APAI's checks is copied or rewritten.
The robot app runs APAI's contact monitor in its enforcing mode whatever the arm profile says, so the arm never moves
under the robot app without a contact monitor that can stop it. The robot app refuses to start with a limit set whose
joint speed is above the arm profile's speed cap.

Besides the permit check, two pieces around the checker are written new: the agent's envelope and the code that builds
a checker from a limit set. On the Thor the envelope is a box in joint space, for the rig 120 degrees around zero on
each joint, cut to the joint limits. A setpoint that starts or ends outside it is refused, so the envelope cannot be
re-entered. Today the envelope is a wrapper in the uncommitted Thor adapter files, and that wrapper has a second bound:
the distance of the gripper tip from the middle of the working area (0.25 m on the rig). The workstation's checker
applies that second bound when the agent program runs with its models; the Thor does not. The same bound on the Thor is
the alternative and is not built: until it is, the Thor stops a setpoint outside that bound only at the joint envelope,
the workspace box and the table.

The Thor stays small in work, not in code loaded: importing the checker loads much of APAI's Python (OpenCV included),
but no model and no GPU code. The models stay as today (cloud language model, SAM 2.1 Tiny segmentation, FFS stereo
depth). The full stack is the agent program with these models. The minimal stack is the agent program with APAI's mock
model and mock scene; the tests and the runs on one machine use it. If APAI is replaced later, only the agent program
changes with it. The robot app keeps APAI's motor driver and checker until they are replaced separately, and the robot
link messages name nothing from APAI except the value `apai-mock` of the declared driver.

## 5. Connections

This section states this design's placement of the human path, in which the teleop connection reaches the Thor. Section
21 states where the build places it; that placement waits for the project owner's decision.

Two connections carry robot traffic between the machines: the teleop connection and the robot connection. The camera
connection is separate, as in the first proposal. The first proposal counts the camera link too and so calls its own
design "two logical links"; this file does not count the camera connection.

| Connection | Carries | State |
|---|---|---|
| Teleop | SAMPLES from headset and controller, workstation to Thor | Existing code, not edited |
| Robot | Moves, permits, gripper commands and operator requests to the Thor; robot state, results and events back | New; same `core::Link` class |
| Camera | Compressed stereo video and the time of each camera frame, Thor to workstation | Isaac Capture pull request #1410; temporary, the owner expects it to be replaced. The camera receiver on the workstation dials the camera sender on the Thor. The agent program and the viewer each read the receiver's decoded stereo pairs through a child process of their own, the camera frame bridge. In the tests no camera frame from a Thor reaches either: both are tested on recorded stereo pairs and on stereo pairs the tests make up. On the rig the agent program has run on the camera connection: the example's README names a run folder of 2026-11-17 that was made with the full stack, the camera connection and the robot connection between the two machines. The README reports no run of the viewer's live picture on the rig's camera |

Why a second connection instead of one through the Teleop Session Server:

- Nothing that works today is edited. Carrying robot traffic on the teleop connection changes the teleop client, the
  host, the server and `TeleopSession`, and the automated remote tests and hardware goals 1, 2, 3 and 5 of the hardware
  test record have to be repeated.
- A CloudXR fault or a bad glove sample ends the Teleop Session Server's connection today. With two connections that
  ends human input only; the agent, the operator console and the assistance wait go on.

The price: the robot app watches two connections, each with its own "alive" state; robot data does not pass through the
Teleop Session Server; and starting the agent before any headset exists relies on the teleop thread, not yet run.

**Robot link service.** The workstation end of the robot connection is one always-running program. It offers a local
API on a socket that only programs of the same user on the workstation can open: a client receives robot state (with
the joint readings of the last half second), results and events, and sends commands and operator requests. The agent
program's arm adapter, the operator console, the console command and the viewer are clients of that API. A
client names its role when it attaches, and the service forwards from each role only what that role may send: from the
agent's role MOVE, GRIPPER, PERMIT, CHECKER_STATE and the OWNER_REQUESTs for a new task, for help, for giving the arm
back, for the motors off and on and for the return to rest when the agent program ends (section 6); from the
console's role RESET, RELEASE, OPEN, SHUTDOWN and the OWNER_REQUESTs for a typed `rest`, for control owner nobody
and for the motors on; from the viewer's role VIEW_STATUS; from every role STOP. Anything else is refused by the
service and does not reach the Thor. One agent client is attached at a time. WELCOME, ROBOT_STATE and EVENT go to every
client; a RESULT goes to the client that sent the command and to every console. STOP, reset and the takeover prompts
therefore keep working when the agent program has crashed. When the agent's API client closes while the agent is
control owner, the service sends an OWNER_REQUEST "help needed" with the reason "agent program gone"; this is how the
Thor learns of a crash. In the first proposal the same API lives inside the Teleop Session Server; the clients are the
same code in both designs.

**Teleop connection, unedited.** The teleop thread dials it when the robot app starts and waits, without a time limit,
until the server reports READY. The agent runs meanwhile. The wait uses the lower-level `RemoteSession` binding in a
retry loop, because `TeleopSession` entry cannot be cancelled and ends at once when the server is busy or shutting down.
After READY the same thread enters a `TeleopSession` in `SessionMode.REMOTE` and steps it at 50 Hz until the connection
is lost; then it waits again. Right B is read there while the agent works, and human driving runs there after a
takeover. Nothing restarts the Teleop Session Server today, so it runs under a restart script (not in the first build,
section 20). The console learns that a takeover is not possible: at once when the server process is gone, after about
1 s when the controller is lost, and after about 3 s when the teleop connection goes silent (measured with a firewall
rule on the teleop port; a pulled cable also takes the robot connection). A hung server is not noticed before READY.
After a CloudXR restart, Headless must be switched on again in the headset before a takeover is possible. If the bench
test of this thread fails, the fix is an edit to the same teleop files that the first proposal edits.

**New Isaac Capture library code.** Only a Python wrapper for `core::Link`, which Isaac Capture otherwise opens only
from C++: the module `isaaccapture.remote_link`, with a link that dials, a link made from an accepted socket and a
listener (as built: about 1,000 lines of source and 2,200 of tests, added beside existing code; four existing build
files and one existing C++ test file gain lines). The wrapper copies a few small helpers of the teleop binding in place
of sharing them, so that no teleop file is edited; the price is two copies. All other new code is example code. Robot
link message type numbers start at 64 here, and the same choice is proposed for the first proposal, which names no
numbers; the teleop link uses 1, 2, 11 to 14 and 32 to 35. Numbers 64 to 95 are reliable and 96 to 127 are newest-wins
(section 6 defines both), so the type number alone says how a robot link message is delivered. The local API's own
messages use 192 to 199 and never travel on the robot connection.

**The robot connection.** The robot app is given the workstation's address when it is started and dials the robot link
service. It dials again one second after a failed dial or a lost robot connection (`connection.retry_s`), for as long
as it runs. The service listens on a configured address and port (47610 in the shipped files), accepts the robot
connection only from the configured Thor addresses and holds one robot connection at a time. Each end sends a heartbeat
every 0.5 s and counts the connection as lost after 3 s without a link message. The robot connection is not encrypted
and the peer proves nothing but its address, so both machines are on one trusted network.

## 6. Robot link messages

Where the rows OWNER_REQUEST and ROBOT_STATE speak of a takeover and of human input, they state this design's placement
of the human path; section 21 states what the build sends.

One module defines every robot link message and its encoding, in robot terms, and both machines import it. A robot link
message is UTF-8 JSON of at most 4 MiB. A receiver refuses a link message that lacks a field that has no default value,
holds a value of the wrong kind or holds a number that is not finite, and ignores a field it does not know, so that one
end can gain a field before the other. Everything above the connection uses three calls only: send reliable, send
newest-wins, receive. "Reliable" means queued and delivered in order; "newest wins" means an unsent older link message
of the same type is replaced, and the receiving end keeps only the newest unread one of that type as well.

Every command (MOVE, GRIPPER, CHECKER_STATE, OWNER_REQUEST, RESET, RELEASE, OPEN, SHUTDOWN) carries a command id, the
session id, an owner number and a stop number. An operator's command quotes the numbers of the ROBOT_STATE shown with
its question. The agent's commands and permits quote the numbers of the grant that made the agent control owner and
never newer ones, so that nothing of an ownership that has ended is accepted. The table lists what else each carries.

| Link message | Direction | Delivery | Contents |
|---|---|---|---|
| WELCOME | Thor to workstation | Reliable | Sent first on every robot connection. Protocol version; session id; robot and profile identity; joint names; the joint limits of the agent limit set; the rest pose; the setpoint period and the largest number of setpoints a MOVE may hold; limit-set versions; the name of the camera calibration in use; the declared driver; the joint speed limit of the agent limit set (`agent_max_joint_speed`); and four optional fields that name what the Thor's human checker holds a person to (`human_max_joint_speed`, `human_tip_box`, `human_tip_floor`, `human_joint_box`; the paragraphs below the table). Joint signs, units and coordinate-system versions are not in the first build (section 20) |
| ROBOT_STATE | Thor to workstation | Newest wins | Per joint the value, read start, read end and a valid flag (measured only, never a cached value and never the commanded target), and every joint reading of the last half second; the last target sent with its send time; the jaws: the opening with its own time and valid flag, whether the gripper is referenced, the last jaw target, the "may be holding" flag (section 11) and whether the jaws are proven empty, loaded or unknown; control owner, owner number, stop number, stop level with the reason of a fault-stop, and limit-set version in force; the motion state (`idle`, `moving`, `hold`, `rest_hold` for a hold at rest, `rest_unpowered` for rest with the motors off, and `release_failed` after a torque release whose driver call failed, when the motors may be on or off) and whether torque is on, which stays true in `release_failed`; the latest command outcome; the running move's id, the index of the setpoint sent, the number of its setpoints and whether the move waits for its first permit or is sending; the permit the Thor holds (its move, the last setpoint it covers, the age of its camera frame, the time to its expiry) and why the newest permit is refused; the time left of the assistance wait; human input (teleop connection up, the stage the Teleop Session Server last reported, READY seen, age of the newest SAMPLES) and the age of the live view; for a takeover: whether human driving is switched on, whether a takeover is open, whether the arm is following the hand, whether the next human motion needs a released grip first, why the last human motion ended and why a takeover would be refused; health (the age of the oldest joint reading, read failures per joint, whether the arm holds at a contact, motor and gripper faults, motor temperatures where the driver can read them, missed deadlines, clamping); and a clock pair |
| MOVE | Agent to Thor | Reliable | The move id (its command id), the expected start pose with a tolerance, the setpoints (one for each setpoint period; the MOVE names the period), the grasp region if any, the Thor wall-clock time of the camera frame behind the grid the move was checked against, an expiry on the Thor's monotonic clock |
| GRIPPER | Agent to Thor | Reliable | Target opening, maximum effort, plain or close-until-contact |
| PERMIT | Agent to Thor | Newest wins | Move id, the session id, owner number and stop number of the agent's grant, the last setpoint it covers, the Thor wall-clock time of the camera frame behind the grid it was checked against, and an expiry on the Thor's monotonic clock |
| CHECKER_STATE | Agent to Thor | Reliable | Tells the agent checker to forget its contact spots, or to end the grasp region of a grasp that was finished or given up, when APAI's skills do so on the workstation |
| STOP | Any client to Thor | Reliable | Reason and sender |
| OWNER_REQUEST | Agent, console or robot link service to Thor | Reliable | Requested control owner (agent for a new confirmed task, recovery for a typed `rest`, nobody) or "help needed" with the reason. For "agent" it also says whether the timed return to rest (the one at the end of the assistance wait, section 10) applies to this task (the link message's default is yes; the agent program's shipped files send no, section 11). For a typed `rest` it says whether the operator typed the confirmation that the path is clear (section 11); without it the request is refused. For a typed `rest` and a typed hand-back it also says whether the operator typed that the jaws are empty (section 11). The robot link service forwards from the agent's role the requests for the agent, for nobody and for help, from the console's role those for a typed `rest` and for nobody, and sends "help needed" itself only with the reason "agent program gone" (section 5). A takeover is not requested here; it comes from right B on the teleop connection (section 12). The build has two requests for it, "human" for a takeover and "handback" for its end at rest: the service forwards them from the human program's role, and from the console's role only where its setting `operator.typed_takeover` is on (section 21). Three more requests come from the agent's role (section 11). "motors_off" is sent where APAI's rest tool ends, and the robot app switches the motors off only with the arm measured at rest. "motors_on" is sent once the operator has typed the confirmation, also by the operator console and the console command, and the robot app switches the motors on again as a typed `on` does. "park" is sent when the agent program ends, once its operator has typed `path is clear` there; the robot app runs it as a typed `rest` with that confirmation, which ends with the jaws closed and the motors off. The agent program asks for it only when a task of its run owned the arm, the motors are on, nobody owns the arm, no takeover is open and no fault-stop is latched |
| RESET, RELEASE, OPEN, SHUTDOWN | Console to Thor | Reliable | OPEN and SHUTDOWN say whether the operator typed the confirmation they need (the jaw opening, section 13; a clear path, section 11). RELEASE and SHUTDOWN say whether the operator typed that the jaws are empty (section 11). RESET carries nothing more |
| VIEW_STATUS | Viewer to Thor | Newest wins | Whether the view is live, which is whether a browser on the workstation is showing the live view (section 13 says when that holds); the age of the live picture shown; and when the link message was sent, on the Thor's monotonic clock as the viewer reads it off its newest ROBOT_STATE |
| RESULT | Thor to workstation | Reliable | For each command id: accepted or refused with the reason, then completed, cancelled or failed with the reason and the measured end pose. Every command gets one final RESULT; one that cannot be decoded is refused under its command id |
| EVENT | Thor to workstation | Reliable | Text for the operator: help needed, countdowns, refusals with reasons, the start and the end of a takeover, the end of a human motion that a released grip did not cause |

ROBOT_STATE gives times per joint, because a joint read takes about 8 ms and the 20 ms motor loop makes one or two reads
per pass, so each joint is read again only every 60 or 120 ms. The robot app sends a ROBOT_STATE every 0.05 s, and at
once when the control owner, the owner number, the stop number, the stop level, the motion state, the assistance wait
or the latest command outcome changes, or when the running move begins to wait for its first permit or begins to send.
In a person's drive, which is one stretch of human motion on the Thor (section 21), the robot app also sends a
ROBOT_STATE behind every setpoint, one per setpoint period, and the timed one comes again 0.05 s after the last of
those (`state.publish_period_s`). The human program, the program that turns a person's hand motion into joint targets
(section 21), reckons from them how far the robot app's own joint target has come, and holds its joint target back
against that (the hold-back, section 21). That ROBOT_STATE is sent on the drive's thread once the setpoint has gone
out, and none is sent while another thread holds the robot app's lock, as during a STOP: the drive never waits for it.
During an agent's move and while the arm holds, none is added.
A ROBOT_STATE carries a sequence number and no "loop alive" flag: a workstation program that has had no ROBOT_STATE
for its own age limit treats the robot state as unknown.

In WELCOME angles are in rad and in the order of the joint names. The agent program compares the calibration name with
its own, and the profile identity with the one its configuration names (`link.expected_profile_id`; an empty value, as
in `configs/agent.yaml`, accepts any profile), and takes no task when they differ. The Thor's wait for a first permit
and its settle time (section 7) are not in WELCOME: the agent program's configuration repeats them, and the two files
are kept equal by hand. The agent's speed limit is in both places. The agent program plans at the lowest of three
speeds: the one it is started with (the option `--speed-profile` with the full stack, the key `task.max_joint_speed`
with the minimal stack), its key `thor.agent_max_joint_speed`, and the limit that WELCOME names
(`agent_max_joint_speed`); where WELCOME names none, the key stands for it.

The four human fields of WELCOME name limits that the Thor's human checker, the checker instance of the human limit set
(section 8), enforces on every setpoint of a person's drive, named or not: naming them changes nothing on the Thor. The
human program (section 21) reads them so that the arm stops at such a limit and goes on following the hand, where a
joint target beyond the limit is refused by the human checker and ends the person's motion in a hold. Each field is
optional: a robot app that leaves one out is still read, and the protocol version is the same with and without them.

| Field | What it names | What the human program does with it | When it is absent |
|---|---|---|---|
| `human_max_joint_speed` | The joint speed limit of the human limit set, in rad/s (`limits.human.max_joint_speed`). The Thor follows a person's joint target at 0.98 of it, its speed clamp (`SPEED_MARGIN` in `robot_app/human.py`) | Moves the joint target it sends at 0.9 of that clamp (`JOINT_SPEED_SHARE` in `human/logic.py`), every joint by the same share of its way, so that the Thor's own joint target is not left behind | The human program takes 0.1 rad/s for the limit (`UNNAMED_JOINT_SPEED`) |
| `human_tip_box` | The box the human checker holds a person's gripper tip in: its lowest and its highest corner, in metres, in the arm's base coordinate system. With APAI's checker it is the workspace box of the arm profile. A tip outside it may only come back | Keeps the gripper tip of the pose it solves for 0.01 m inside the box (`TIP_BOX_MARGIN_M`), on each axis by itself: the tip stops at a wall and slides along it | No wall is kept: a target outside the box is refused by the human checker, and the person's motion ends in a hold. The simulated arm's checker has no such box, and its robot app names none |
| `human_tip_floor` | The height, in metres in the arm's base coordinate system, under which the Thor counts a person's gripper tip as low: the table plane plus the higher of `rest.table_clearance_m` and the table clearance of APAI's arm profile. A tip under it may only rise | Takes it for the lowest wall: the tip stops 0.01 m above it and slides along it. Under that height it moves its joint target only where every step of the Thor lifts the tip (section 21) | The bottom of the tip box stands in its place, and a tip lowered under the table clearance ends the person's motion in a hold. The simulated arm's checker knows no table, and its robot app names none |
| `human_joint_box` | The joint box the human checker holds a person's joint target in: its lowest and its highest corner, one value per joint, in rad (`limits.human.joint_min`, `limits.human.joint_max`). A joint outside it may only come back | Stops its joint target 0.01 rad inside the box (`JOINT_BOX_MARGIN_RAD`), and the arm goes on following the hand | A joint target outside the box is refused by the human checker, and the person's motion ends in a hold |

Rules:

- A command that quotes an old session id, owner number or stop number is refused. A repeated command id is answered
  again and not executed again. After a reconnect nothing is replayed.
- The Thor trusts nothing that arrives. It parses a robot link message only up to a size limit for its type, keeps a
  bounded number of link messages waiting and refuses a command beyond that, and cuts every text from a link message to
  a fixed length before it shows or logs it.
- A MOVE is refused if the measured start pose differs from the expected one by more than the tolerance the MOVE asks
  for, if that tolerance is wider than a Thor setting allows (0.05 rad), if its expiry has passed, if its period is not
  the Thor's setpoint period, if it holds more setpoints than a Thor setting allows (3,000 in the shipped files for the
  simulated arm and the rig), if another move is running, or if its grasp region is beyond the bounds the Thor's
  settings give it (its radius, and how far below the table it reaches). Limits are never in a command; the Thor picks a
  limit set by control owner.
- A PERMIT never reaches past the end of its move, and a link heartbeat is never a permit. The newest PERMIT of a move
  decides how far the move may go (section 7).
- A STOP is forwarded by the robot link service at once and served by the Thor before every other robot link message
  that has arrived. From the STOP until a ROBOT_STATE shows a higher stop number, the service drops permits and refuses
  a MOVE or GRIPPER that quotes the old stop number; one that the service has passed just before the STOP can still
  reach the Thor, where the stop-number check refuses it. A STOP with a payload that cannot be read is still a STOP.
  `core::Link` has no priority path, so a STOP waits behind the bytes of a MOVE that is already on its way: about
  590 kB for the longest move, which is about 5 ms on a 1 Gbit/s cable and about 0.25 s at 20 Mbit/s. The alternative,
  a priority path, is an edit to `core::Link`, which this design does not make (section 20); the demos run on a wired
  connection, the STOP at the Thor terminal needs no network, and section 19 measures the delay. A STOP sent between a
  new robot connection and its first ROBOT_STATE keeps the service's hold on permits until a later STOP is confirmed;
  the console shows that the STOP is not confirmed, and the operator sends a second one.
- The robot link service answers in the Thor's place with a RESULT of its own when it does not forward a command (the
  role may not send it, a STOP is not confirmed, or there is no robot connection), and when the robot connection is lost
  before a forwarded command has its final RESULT.
- The live view is fresh while the newest VIEW_STATUS says that the view is live and the age it reports, plus the time
  since it was sent, is at most the view age limit (a Thor setting, `human.view_age_limit_s`; its first value is
  0.5 s, and the value for the rig is chosen from the ages measured at the headset check of section 19). The Thor
  counts the time since it was sent on its own monotonic clock, from the send time the VIEW_STATUS names; a
  VIEW_STATUS that names no send time gives no view age, and the live view is then not fresh. A lost robot connection
  forgets the VIEW_STATUS. A link heartbeat is never a VIEW_STATUS.
- Existing Isaac Capture schemas are not changed.

## 7. The check before each setpoint

Today APAI checks every setpoint just before sending it, in under a millisecond: finite values, joint limits, speed,
workspace box, table, contact spots, and obstacle distance. The last reads the obstacle grid, which perception builds
from camera frames. After the split the grid is on the workstation, so the check is divided as in the first proposal:

- Before a move is sent, the workstation checks the whole move, obstacle distance included.
- While the arm moves, the workstation keeps checking the part just ahead of the arm against its newest grid (the check
  ahead) and sends a PERMIT for it.
- The Thor first runs the start checks of a MOVE that need no permit: the numbers it quotes, motors on, its expiry, and
  a fresh full joint read within the start tolerance. It then shows the move as armed in ROBOT_STATE, and the
  workstation sends the first PERMIT when it sees that. The Thor sends the first setpoint once a permit covers it; a
  move with no covering permit 0.5 s after it is armed (`permits.first_wait_s`) is refused with nothing sent. The
  alternative, a first PERMIT sent right after the MOVE, saves about one ROBOT_STATE period, but a start refusal then
  reaches APAI after its skill has counted the move as started, and the task ends with a help request.
- The Thor checks every setpoint just before sending it: the camera-free checks, and that an unexpired permit covers it
  and rests on a camera frame younger than the grid age limit (3 s today). It makes no network round trip per setpoint.
  Each permit also feeds the agent checker's perception watchdog, without which that checker refuses all motion.
- No permit, an expired permit, or a permit that rests on a camera frame older than the grid age limit ends the move in
  a hold. A link heartbeat is never a permit, so a hung agent program is caught as well.
- A PERMIT quotes the session id, the owner number and the stop number like a command; the Thor refuses one that quotes
  old values or arrives while the agent is not the control owner. The newest valid permit replaces the one held, so a
  permit that reaches less far shortens how far the move may go, and a permit that covers no setpoint ends the move in
  a hold at the next setpoint. That is how the agent ends its own move; apart from STOP no other link message ends a
  move. A permit that rests on an older camera frame than the one held is ignored, except that it may still shorten the
  reach.
- How far ahead a permit reaches and how long it lasts are chosen from measured arm speed and stopping distance; they
  are not copied from a camera or connection timeout. The shipped files hold first values, none measured: a permit
  reaches 25 setpoints (0.5 s of motion) beyond the last setpoint the Thor reported as sent, lasts 1.0 s on the Thor
  and is renewed every 0.05 s (workstation settings `permits.reach_setpoints`, `permits.life_s`, `permits.renew_s`).
  A session with the operator measures them (section 19). No such measurement is recorded: the rig's agent file
  holds the same three values, and the example's README names no measurement of the arm's stopping distance and
  calls the pace of the Thor's loop on the arm's CAN bus not known. What the README does report from the rig is
  that the run folder of the second session of 2026-11-17 holds the files of measurements that the agent program
  writes in every run, for that run's 14 moves and 4 jaw commands. For a move such a file holds, among other
  numbers, the fewest setpoints the held permit still covered beyond those sent, the age of the camera frame
  behind the held permit and how many setpoints the Thor sent late. The README quotes none of these three from
  that run; among the numbers it does quote from it is the spread of a standing arm's joint readings, 0.00065 rad
  in 346 s at rest with the motors on. The robot app writes to its log how long one joint read takes on that bus,
  and the README quotes no value from the rig. The files for the rig hold 0.2 rad/s for the agent
  and for the return to rest and 0.8 rad/s for a person (section 8 names the keys). Measuring before any motion is the
  alternative; it costs one more session before the first move.
  Acceptance test: an obstacle placed in the path during a move stops the arm as it does today.
- The Thor converts each camera-frame time from its wall clock to its monotonic clock on arrival and rejects a permit
  whose camera-frame time is more than 0.05 s ahead of its clock; a time less far ahead counts as the time of arrival.
- After a delay the Thor does not send setpoints faster to catch up, and it skips none. Every delay up to the
  large-delay limit moves the remaining deadlines back with it; a delay above the small-delay limit is also counted as
  a missed deadline in ROBOT_STATE. No setpoint is sent once its permit has expired. A delay above the large-delay
  limit ends the move in a hold. The first values are 5 ms and 100 ms (`executor.small_delay_s`,
  `executor.large_delay_s`); neither is measured. Both are to be chosen from measurements on the Thor in a session
  with the operator; no such measurement is recorded, and the rig's file holds the same two values. A move that
  ended early is never resumed; the agent may send a new MOVE, checked from the measured pose like any other.
- After the last setpoint the Thor reads the joints until each is within 0.03 rad of that setpoint
  (`executor.settle_tol_rad`). A move whose joints are not there after 2.0 s (`executor.settle_timeout_s`) ends as
  failed, in a hold. A contact while the arm settles ends the move like a contact during it.

Sending the grid itself to the Thor, so that the Thor runs today's full check locally, is future work (section 20). It
is also the fallback: if the permit timing measured in section 19 is too slow for the agent's speed, the speed is
lowered, and if that is not enough the grid on the Thor is built earlier than planned. The check is never weakened.

One setting stands against that sentence: `permits.required_for_agent` in the Thor configuration can switch the permit
check off. It is on in every shipped file. With it off an agent move needs no permit: the Thor checks the age of the
camera frame its MOVE quotes once, when the MOVE arrives, and nothing checks the path against the obstacle grid while
the arm moves. Only tests set it off. The robot app accepts it off on any arm, the real arm included, so only the value
in the file keeps the permit check on for the rig (section 21).

## 8. Control owners and limits

Where the row "Human" speaks of the hand speed limit and of controller samples, it states this design's placement of
the human path; section 21 states where the build places them.

Three versioned limit sets live in the Thor's configuration: agent, human and recovery. The control owner selects the
set. There are three checker instances, one per owner. A limit set in the Thor configuration holds a version, a joint
box and a joint speed limit; the agent's also holds its envelope and the human's its step limit. Every other limit of
APAI's checker, such as the workspace box and the table, comes from the APAI arm profile and is the same for all three
checker instances.

The joint speed limits in the rig's file, `configs/robot_app_rebot.yaml`, are 0.2 rad/s for the agent
(`limits.agent.max_joint_speed`), 0.2 rad/s for recovery (`limits.recovery.max_joint_speed`; a return to rest runs at
`rest.recovery_speed`, also 0.2 rad/s) and 0.8 rad/s for a person (`limits.human.max_joint_speed`). The agent
program's file for the rig, `configs/agent_rebot.yaml`, repeats the agent's 0.2 rad/s in `thor.agent_max_joint_speed`
(section 6). The comments in both files mark these values as the owner's.

| Limit set | What it applies |
|---|---|
| All three | Finite values, joint limits, a hardware speed bound, table, workspace and configured keep-out boxes, and APAI's contact monitor. The hardware speed bound is the speed cap of the APAI arm profile: the robot app refuses to start with a limit set whose speed is above it and clamps nothing to it. The contact monitor runs and stops motion under every control owner (section 4). A joint that lies outside its joint box may only move back toward it |
| Agent | Its envelope, the contact spots and grasp state it has collected, its speed limit, and the permit check of section 7 in place of the obstacle check. Its watchdog is fed by each PERMIT. An invalid agent move is refused, never clamped |
| Human | No obstacle check; its watchdog is fed by fresh controller samples. It has a hand speed limit by measured time (first values 0.10 m/s and 0.4 rad/s; computed on the arm's kinematic model, 1 m/s at the gripper tip asks a joint for 10 to 13 rad/s, so a higher hand speed limit only runs into the joint speed clamp), a joint speed clamp (first value 1.5 rad/s for the simulated arm; a value above the speed cap of APAI's arm profile is refused, and the rig's file has 0.8 rad/s; the clamp acts at 0.98 of the value, because APAI's checker refuses a step above its limit by any amount) and a per-command step clamp. Targets are clamped, then checked, and clamping is reported in ROBOT_STATE. The build refuses a step that is too large in place of clamping it; that choice waits for the project owner (section 21). The agent's stored state does not apply, because it would block a person exactly where help is needed |
| Recovery | No agent envelope, because that envelope cannot be re-entered from outside, and no obstacle check in the first build, because the grid stays on the workstation. Its perception watchdog has no age limit, because nothing can feed it; with the default 5 s limit an unmodified checker refuses every move, and today's exit move patches the watchdog out instead. It keeps no joint margin, because the rest pose lies on the joint stops, inside the margin that the agent and human checkers keep from the joint limits. It keeps the contact spots the agent checker has recorded so far: leaving one is allowed, entering one is refused. A contact during an agent move or during a return to rest is recorded as a contact spot in the agent checker and in the recovery checker; a contact while a person drives records none. When the agent's skills clear the agent's contact spots (CHECKER_STATE), the recovery checker's are cleared with them. The agent's grasp state does not apply. Its joint box contains the human joint box: the robot app refuses a Thor configuration in which it does not. A start pose inside that box can still be refused for its path or for the position of the gripper tip (section 11) |

Only one control owner at a time. Changing owner always passes through a hold. From an accepted takeover until the
hand-back (section 12), also while a STOP has made nobody the control owner, the agent's OWNER_REQUEST for a new task is
refused with `takeover not finished: give the arm back with right A at rest`.

## 9. STOP and fault-stop

Where this section speaks of right B, it states this design's placement of the human path; section 21 states the path
a STOP by right B takes in the build.

STOP needs no ownership. It can come from the operator console (a typed `stop`, or Ctrl+C there), from the Thor
terminal (a typed `stop`, or Ctrl+C: the STOP key of this file, which needs no network), from Ctrl+C at the agent
program's terminal, and from right B on the controller. The agent's own skill stop is not a STOP: the workstation
withdraws the permit of the running move (section 7), the move ends in a hold, and the task ends with a help request
(section 10). The robot app accepts a STOP whose sender is the agent and leaves the assistance wait unchanged for it.
No program of the first build sends one on purpose; the robot link service names the agent as the sender of a STOP that
it cannot read from the agent program's attachment. Right B works only while a headset is worn, the teleop connection
has reached READY and fresh SAMPLES carry the controller's buttons. The controller's pose need not be valid for a STOP
(section 12).

| | STOP | Fault-stop |
|---|---|---|
| Raised by | A person. A STOP whose sender is the agent is accepted as well | An error the motor or gripper driver raises on a command or a read; joint feedback lost for more than 0.35 s; a jaw reference that fails at the start; an internal error of the robot app |
| Arm | Stops at the next pass of the 20 ms motor loop and holds, motors on | Same |
| Jaws | No jaw command that quotes an older stop number is accepted. A close under way is ended at once by re-sending its target at the light holding stiffness. An opening under way is ended as well, and the jaws are held lightly where they are. Jaw commands that wait are dropped. Never opened, never left at the closing stiffness. If the driver has not returned from the cancelled command within 0.25 s, the text of that STOP says so; until it returns the jaws take no command and `rest` and `release` are refused | Same |
| Task | Cancelled, never replayed; control owner becomes nobody | Same |
| Assistance wait | Cancelled by every STOP that is not the agent's own: operator console, Thor terminal, right B, Ctrl+C at the agent program's terminal, a STOP whose sender is not known and a STOP whose content the Thor cannot read. The arm holds until someone acts. Unchanged by a STOP whose sender is the agent, which includes a STOP that the robot link service cannot read from the agent program's attachment | Cancelled |
| What may happen next | New task (after a takeover only once the arm was given back, section 12), takeover, typed `rest` or `open` | Nothing until a person types `reset` |

`reset` clears the latch and nothing else: it starts no motion and no wait. It is refused while the driver still
reports a fault or its fault state cannot be read, and afterwards nobody owns the arm. Neither level cuts torque. The
0.35 s is APAI's contact-monitor limit, which today applies only during a move and ends it without a latch. Using it as
a fault-stop, and in hold, is new; the value is checked on the bench.

APAI's drivers have no call that reads a motor's fault state, and the rig's motor library raises nothing when a command
reaches no motor. A motor's own fault therefore shows only as a driver error, as lost joint feedback or as a contact; a
lost light jaw hold or a lost torque release shows only in later readings, if at all.

A contact is not a fault-stop: the motion stops and the arm holds the freshly measured pose, so it does not keep
pressing. The hold is sent only if one fresh read of every joint agrees with the newest readings of the move; if it
does not, or cannot be read, nothing is sent, the last setpoint stands and the RESULT, or under a person the EVENT, says
so. A hold that fails in any other way, for example on an error the driver raises when the hold is sent, is a fault-stop
under every control owner. Under the agent the step fails, and no count of failed steps ends the task (section 10);
under a person a fresh grip squeeze resumes; during a return to rest the move ends and is not retried.

The jaw rule of a STOP also applies whenever the agent's ownership ends without a STOP: on a help request, when the
agent gives the arm back, and when the robot connection is lost.

Built from: STOP calls `SafetyHarness.halt()` on all three checker instances and raises the stop number. APAI's halt is
cleared by the next move, so it is the robot app's own stop-number check that keeps old commands out. Fault-stop calls
`SafetyHarness.estop()` and is cleared by `reset_estop()`. The driver's own `stop()` is not used, because it makes later
jaw commands vanish without an error, and `AgentOrchestrator.resume()` is not used, because it also drives the arm home.
The jaw rule is new code. The worst-case delay from a typed `stop` to the arm holding is measured during an agent move.

Ctrl+C at the Thor terminal is STOP: the arm holds and the robot app keeps running. A closed terminal is ignored. The
app exits on the operator's shutdown command, typed as `shutdown` at the operator console or as `quit` at the Thor
terminal; `quit` at the console only leaves the console (section 13). The shutdown is a return to rest followed by the
permitted torque release; if the return or the release is refused or fails, the arm holds and the app keeps running. A
shutdown or a `release` sent from the console does not outlive the robot connection: if the connection is lost before
the motors are off, they stay on. A shutdown that has to move the arm asks the same path-clear confirmation as a typed
`rest` (section 11). The app also ends on a termination signal, which never moves the arm. With the arm at measured
rest and the jaws proven empty the app closes the jaws and switches the motors off before it closes (section 11).
Anywhere else it closes with the motors left on, which is the row "Robot app dies" of section 10. The 48 V switch is
the only hardware stop, and it drops the arm.

## 10. Faults and the assistance wait

The rows on the headset, the Teleop Session Server and the teleop thread state this design's placement of the human
path; section 21 states how the same faults show in the build.

Every fault first ends in hold. Only after an agent failure does a wait then decide how long the arm holds before it
goes to rest by itself: the assistance wait. It starts when the agent has failed while it owns the arm and the robot
connection is alive, and it waits for a person to take control. The first value is 120 s, never below 90 s. When it runs
out, the robot app makes one attempt at a return to rest (the timed return), if the switches of section 11 allow it. The
robot app counts the wait on its monotonic clock and shows it as a countdown on the operator console and the Thor
terminal.

- "The agent has failed" means that a help request reached the Thor while the agent owned the arm. The agent program
  sends OWNER_REQUEST "help needed" at the end of a task that APAI gave up as stuck, whose last tool result was a
  blocked motion, whose step budget ran out, that an error in the agent program ended, or that ended with no task
  report from APAI. The step budget is the most steps APAI's planner may take for one task (the agent program's
  option `--max-steps`, 30 by default). The robot link service sends the help request when the agent's client
  closes while the agent is the control owner (section 5): after a crash, and after an end of the agent program
  whose request at the task's end got no answer. It sends it as well when a ROBOT_STATE shows the agent as
  control owner after the agent's client has left, once for each owner number.
  Three ends of a task send no help request and start no wait. A task that the model itself ends without
  success gives the arm back as one that succeeds does (section 13). A task cancelled with Ctrl+C at the agent
  program's terminal sends a STOP (section 9). After a STOP from the operator console the task's help request is
  refused, because it quotes the stop number from before that STOP. No count of failed steps ends a task: APAI ends
  it when the step budget runs out. One failed step is retried inside the task, with one exception in the first
  build: a refusal by the workstation's check ahead during a move ends the task with a help request (a skill stop,
  section 13). The
  alternative, a retry of that move inside the task as after any other failed step, is not built; the exception costs
  the operator a new task and its typed `yes` after each obstacle. Model thinking and waiting for consent never start
  the wait.
- The wait is cancelled by: a takeover the Thor has accepted, a new task confirmed with the typed "yes", a STOP from a
  person, a fault-stop, or loss of the robot connection. A headset that connects or is tracked does not cancel it.
- Before the arm moves by itself the operator console and the Thor terminal ring the terminal bell at 30 s, at 10 s and
  on each of the last 5 s, and print `ARM WILL MOVE TO REST`.
- Where the timed return does not apply, because the Thor's switch is off or the task says so (section 11), the wait
  still runs, so that the countdown and the bells still call for a person. The bell lines then read
  `ARM WILL KEEP HOLDING`, and when the wait runs out the arm keeps holding. A countdown line is printed every 10 s.
- The wait makes one attempt. A refused or stopped return leaves the arm holding with the reason shown; nothing
  automatic follows; the operator must type `rest`, take over or send a new task. The attempt is made at the moment the
  wait runs out or not at all: if a return to rest, a release or a shutdown runs or waits at that moment, both terminals
  show `no timed return to rest: busy: ...` and the arm is not moved later.

| What happens | Response |
|---|---|
| Agent fails as defined above | Hold; the agent's ownership ends (control owner nobody, owner number goes up); help requested; assistance wait |
| No valid permit during an agent move | Hold; the agent may send a new MOVE. After a refusal by the check ahead the first build ends the task instead (see above) |
| Headset absent while the agent works | Nothing; the console shows that a takeover is not possible yet |
| Teleop Session Server or CloudXR down | Human input unavailable; the agent, the console and the assistance wait go on |
| Headset, controller signal or the live view in the workstation browser lost while a person drives | Hold. The Thor ends the motion when the controller data are older than 0.25 s (`human.sample_age_limit_s`) or the live view is older than the view age limit. The person keeps control; buttons held down before the loss count for a takeover or a hand-back only when pressed again (right B held down is still a STOP), and the clutch is re-based (the hand position at the next grip squeeze becomes the new starting point); re-engaging needs fresh samples, a fresh view and a new grip squeeze. The Thor enforces the new squeeze itself: after such a hold, and after every change of control owner, it follows the hand only once it has seen the grip released on fresh controller data. No timed return to rest in the first demos. To end the takeover without a headset the operator types `rest`; if the jaws hold something: `stop`, then `open`, then `rest`. The hand-back of section 12 step 7 still needs the controller |
| The headset's stream to the Teleop Session Server stalls while the server keeps sending SAMPLES | Not noticed at once, because the SAMPLES stay fresh. The hand's target stands still, so the arm stops; a released grip and right B are not seen until the stream is back. By `remote_teleop_session.md` the controller is reported inactive about 1 s later; not run. The stop for this case is the STOP key at the Thor terminal |
| The teleop thread throws an error | Treated as controller signal lost; never a return to rest or a motor disconnect |
| Agent program crashes while a person drives | Nothing; the person keeps control. Ending the agent program with Ctrl+C is not a crash: it sends a STOP (section 9), the arm holds, nobody owns it, the takeover stays open and the person takes over again with right B |
| Robot connection or whole workstation lost | Hold. If the agent owned the arm, its ownership ends (control owner nobody, owner number goes up). If a person owned the arm, they keep it: a running human motion ends in a hold, and after the reconnect the arm follows the hand again only with a fresh live view and a new grip squeeze. A pending wait is cancelled, a timed return under way ends in a hold, and a reconnect starts neither again. A `rest`, a `release` or a shutdown sent from the console ends with the motors on. `rest`, `release` and `quit` can still be typed at the Thor terminal. The robot app dials the robot link service again (section 5); a new connection keeps the session id, sends WELCOME and starts no motion |
| Contact | Hold at the measured pose; not latched (section 9) |
| Motor fault, gripper fault, lost joint feedback | Fault-stop. A motor that reports its own fault may stop holding and software cannot prevent it; the fault is cleared only at rest or with the arm supported by hand. Under a person the takeover stays open through the fault-stop and its `reset`; the person then takes the arm again with right B |
| Robot app dies | No software is left to act. The motors have so far kept holding their last target after such exits; for how long has never been measured. Connecting cuts torque, so the robot app connects only an arm that APAI's driver measures at rest: a new start with the arm raised is refused before anything is cut, and the robot app never connects on an operator's confirmation. To recover, the operator holds the arm by hand, switches the 48 V off, lowers the arm to rest and starts the robot app again. The alternative, connecting a raised arm on a typed confirmation that a person holds it, saves the power cycle and drops the arm if the confirmation is wrong. The robot app leaves no hold marker when it ends with the motors on (section 20). The operator console shows `ROBOT CONNECTION LOST`: at once when the process dies, after about 3 s when the Thor hangs. Text and the terminal bell only; no loud alarm |
| CAN bus, power or Thor failure | No software response |

## 11. Return to rest and torque release

A typed `rest`, the rest that ends a takeover and the return after the assistance wait are the same checked move.

1. Conditions, all required, checked before the control owner changes: the jaws are proven empty; no fault-stop; the arm
   is not holding at a contact (section 9; such a hold ends when the agent or a person next moves the arm, or when the
   agent's skills clear the agent's contact spots with CHECKER_STATE; the build makes one exception while a takeover
   is open, section 21); two full joint reads 0.2 s apart are valid and agree within 0.02 rad (taken with the reader
   that fails instead of returning a cached pose); the start pose is inside the recovery region; and the gripper tip
   is above the table clearance (first value 0.02 m) and outside the last grasp region. If any fails, the return is
   refused, the operator console and the Thor terminal say why, and nothing changes. The return is also refused while
   a move, a person's motion, a jaw command, a release or another return runs or waits; a `rest` typed while the grip
   is squeezed is therefore refused.
2. Control owner becomes recovery. Motion and jaw commands from agent and human are refused. A takeover is still
   possible: the press of right B is a STOP first (section 12), which ends a running return in a hold with control
   owner nobody, and the request for control that follows is accepted from there.
3. The robot app records the stop number, the owner number and the recovery checker's halt count. The first setpoint and
   every later one are sent only if all three are unchanged, so a STOP at any moment wins.
4. The return runs as one or more legs. A leg is one straight joint-space move at the recovery speed (first value
   0.1 rad/s; the rig's file has 0.2 rad/s, `rest.recovery_speed`; the planned time of each leg is printed). Which
   legs a return has depends on the arm (`legs` in `robot_app/rest.py`):
   - On the simulated arm, and where the route of APAI's rest tool cannot be read, the return is one leg, from the
     measured pose to the saved rest pose.
   - With APAI the return follows the route of APAI's rest tool, APAI's own return-to-rest skill
     (`apai.skills.rest_pose`). The robot app reads from APAI the route's tracking tolerance, which is how far the
     measured pose may lie from a leg's goal when that leg has ended, the shortest times of its legs and its fold
     steps. The first leg goes up to the ready pose, the raised pose of the arm profile (`home_q`), and is left out
     when the arm is within the tracking tolerance of that pose. The fold then brings the arm onto the rest pose in
     the steps that APAI's rest tool gives, each a leg of its own.
   - With APAI and no ready pose the arm folds from where it is. The robot app takes no ready pose on APAI's mock
     arm, and a ready pose outside the recovery region is not driven to.

   Each leg follows a minimum-jerk time law whose peak joint speed is 0.9 of the recovery speed, and lasts at least
   its shortest time: 2 s for a return of one leg (`rest.min_duration_s`), and the times APAI's rest tool names for
   the leg up and for a fold step. Above that shortest time a leg takes, at 0.1 rad/s, about 21 s per radian of its
   largest joint travel, and half of that at the rig's 0.2 rad/s. One straight leg across the rig's recovery
   region, which is the agent's 120-degree envelope, takes about 44 s at 0.1 rad/s and about 22 s at 0.2 rad/s,
   and one across a region that reaches a joint limit of 3.14 rad about 65 s and about 33 s. These
   durations hold for a return of one straight leg. On a route they hold for the leg up to the ready pose, by that
   leg's own largest joint travel, and the legs of the fold add their times to it. No return to rest on the rig
   is recorded: the example's README names none among what ran on the real arm. Of a test that runs the programs
   as processes on APAI's mock arm, the README reports that a typed `rest` at 0.1 rad/s returned the arm in
   30.9 s, with a largest joint travel of 1.48 rad; the test itself checks neither number. The Thor plans no
   route around an obstacle: it has no obstacle grid (section 8), and
   the leg up to the ready pose is a straight joint-space move. For a typed `rest` it runs on the operator's `path
   is clear`.

   The recovery checker checks the path of every leg before the control owner changes, so a path that it refuses is
   refused before anything moves (section 21 names what a person can meet there). Each leg is planned once more
   from the pose the Thor reads as the leg starts, and the recovery checker checks every setpoint while the arm
   moves, with the contact monitor on. After each leg but the last the measured pose must lie within the route's
   tracking tolerance of that leg's goal; otherwise the return ends as failed and the arm holds. After the leg up
   to the ready pose the fold is planned and checked again from the measured pose, and a fold that is refused then
   ends the return as failed, with the arm holding where that leg left it. The return ends at the first contact.
5. The return ends at rest with motors on. Rest is proven by two full joint reads 0.2 s apart within 0.05 rad of the
   rest pose; a return that does not prove it ends as failed and the arm holds. What follows depends on what asked
   for the return. A typed `rest`, at either terminal, goes on to the torque release below: the robot app closes the
   jaws and switches the motors off there, and the motion state is `rest_unpowered`. If that release is refused,
   the `rest` ends as failed with the arm at rest and the motors on. While a takeover is open a typed `rest` leaves
   the motors on at rest, for the hand-back. The timed return leaves the motors on as well. After the `rest` that
   ends a takeover the person becomes the control owner again, so that the arm stays closed to the agent until right
   A (section 12 step 7). A
   `rest` typed after a STOP during a takeover, while nobody owns the arm, ends with control owner nobody; the takeover
   stays open, so the arm stays closed to the agent all the same. After a timed return, or a `rest` typed while nobody
   or the agent owned the arm, the control owner becomes nobody: a typed `rest` ends the agent's ownership. While an
   agent move runs a typed `rest` is refused; the operator types `stop` first. A return that ends early, in a hold away
   from rest, gives the same owner as one that reaches rest, except that a STOP always makes it nobody (section 9).

Without the robot connection no timed return runs (section 10). A typed `rest`, at either terminal, first asks the
operator to type `path is clear`, and so does a shutdown typed while the arm is away from rest; a typed `yes` confirms
neither (section 13). Whether a shutdown has to move the arm is decided on the pose measured when it runs: an
unconfirmed shutdown that finds the arm away from rest moves nothing. This prompt is new: today's supervised exit move
only prints "Keep the return path clear" and starts. The timed return has no such prompt; it relies on the tested
recovery region, the warning bell and the contact monitor. No return to rest is checked against the obstacle grid in
the first build (section 8).

**Jaws proven empty.** All of these must hold: the gripper was referenced when the robot app started and the reference
is still valid; the "may be holding" flag is clear; no jaw command runs or waits; three jaw readings 50 ms apart are
valid and agree within 2 % of jaw travel; the reading is within 0.1 rad of the last jaw target the Thor sent, or within
3 % of jaw travel where that is less; and that target is full open, or the reading is within 4 % of travel of closed.
The gripper is referenced at the start only when the arm is measured at rest. The flag is set when any close ends short
of its target or a jaw read fails during a close, and is cleared by a measured full open. Anything else counts as
loaded or unknown, and the arm holds. The jaws are never moved to decide; the typed `reference` of section 21, which
is switched off, would be the one exception. The build lets the operator's word stand in for the proof. When a
`rest`, a `release`, a shutdown or a typed hand-back (`handback` at the operator console, section 13) is refused
only because the jaws are not proven empty, the operator looks at the jaws and types `jaws are empty`. The robot
app then counts the jaws as empty, whatever they read and also where they are not referenced or cannot be read,
and clears the "may be holding" flag, which is the one way beside a measured full open that the flag is cleared.
The word ends with the next jaw command that the robot app does not refuse, with the next task granted to the
agent and with the next takeover. Known limit: an object thinner than about 4 % of jaw travel reads as
empty; that is about 3.6 mm if the configured 90 mm travel is right, which the arm profile itself questions. Not
measured.

**Torque release.** Torque is released in one place, the guarded release in `robot_app/rest.py`, and never by a
wait. The guarded release runs at five ends: a typed `release`; the end of a typed `rest` outside a takeover (step
5); the shutdown command; the end of APAI's rest tool in a task of the agent, where the agent program asks for the
motors off (OWNER_REQUEST "motors_off"); and a termination signal (section 9). At each of them it runs only when two
full joint reads 0.2 s apart are within 0.05 rad of rest and of each other, the jaws are proven empty, no move is
running and no fault-stop is latched. Away from measured rest each of the five leaves the motors on and says why.
Before the release the robot app closes the jaws with the driver's close; a close that fails is reported and the
release follows, because at rest the jaws hold nothing up. After APAI's rest tool the robot app neither judges the
jaws again nor sends them anything: that tool found no load before it folded the arm and has closed the jaws
itself. A failed or interrupted return, a stale reading, a disconnect or a second Ctrl+C never leads to a release.
The console asks for a typed `yes` before a `release`. The rig's motor library does not report a failed disable, so
on the real arm the release is reported as `torque release sent at rest: the driver does not confirm that the
motors are off`, and the torque value in ROBOT_STATE is what was commanded. A release whose driver call raises an
error leaves the motors on or off, unknown to the robot app: the motion state becomes `release_failed`, a
fault-stop latches, and nothing moves the arm or the jaws until a release has worked or the robot app was started
again.

After a release the motion state is `rest_unpowered` and the robot app runs on with the same session id. It starts
no motion: a move, a jaw command, a new task and a takeover are refused with `the motors are off; switch them on
again: type on at the Thor terminal or run the console command on`. A typed `on`, at the Thor terminal, at the
operator console or as a console command, switches the motors on again without a new start of the robot app, after
a typed `yes`. The agent program asks for the same, after a question of its own, before the first skill that moves
the arm once the motors are off. The robot app then connects the arm as at its start: the driver refuses an arm
that it does not measure at rest, and the jaws are referenced again. `on` is refused while the motors are on, while
a fault-stop is latched and after a release that failed. A new start of the robot app also switches the motors on:
it asks its confirmations at the Thor terminal, connects with the arm measured at rest and picks a new session id;
no earlier task is replayed. The cost of `on` is a second place that connects to the motors. An unpowered arm lies
less close to the rest pose than a powered return leaves it, so the start has its own, wider tolerance
(`rest.start_rest_tol_rad`; with APAI the arm profile's rest check tolerance, 0.12 rad on the rig). An arm that
starts outside the 0.05 rad of a release is shown as holding, and a typed `rest` brings it inside before the first
`release`.

**Switch.** The timed return has a switch in the Thor configuration, `rest.auto_return`. It is on in the shipped files
for the simulated arm and for APAI's mock arm, and off in the rig's file until the bench tests of section 19 have been
run on that arm. Each confirmed task also says whether the timed return applies to it; the value travels with the
agent's OWNER_REQUEST, and the agent program's setting for it, `task.timed_return`, is off in its shipped files. The
timed return runs only where both are on. With either off the assistance wait still runs and the arm keeps holding when
it ends (section 10). The owner's decision to implement the automatic return is met in code; switching it on for the
rig waits for those bench tests and for the owner (section 21). Before its first unattended use on an arm, the bench
tests of section 19 are run on that arm with a person supporting it.

Built from: the robot app's own straight joint-space planner, which plans each leg, and its move executor, the same
executor that runs the agent's moves, so that one loop applies the stop number, the owner number, the halt count and
the contact monitor before every setpoint; the recovery checker checks the whole path first, as `thor_shutdown.py`
does. With APAI the legs follow the route of APAI's rest tool (`move_to_rest` in `apai.skills.rest_pose`), from
which the robot app reads the tracking tolerance, the shortest times and the fold steps. Not used:
`SafeArm.move_joints` (it is the smaller reuse and costs a second send loop beside that one), `park_on_shutdown` (it
works only while the app is exiting) and a call of `move_to_rest` itself on the Thor (it reads flags that live on
the workstation). New: the planner, the recovery owner, the recovery checker instance, the jaw rule, the path-clear
prompt, the wait, the release command and `on`.

## 12. Human takeover

This section states this design's placement of the human path, in which the robot app reads the controller. Section 21
states where the build places it; that placement waits for the project owner's decision. Section 20 names the steps
below that the first build leaves out.

A takeover is accepted when right B was held for 1 s with the grip released and every sample in that second was fresh,
nobody owns the arm, the motors are on, no motion and no jaw command is running or waiting, a full joint read taken now
is valid, the live view in the workstation browser is fresh, and no fault is latched. The condition on the live view
holds where the Thor setting `human.require_live_view` is on, as it is in the rig's file; with it off a takeover
needs no live view (section 13, and row 9 of the table in section 21). Fresh means that the controller data are at
most 0.25 s old and the live view at most 0.5 s (first values, Thor settings). A STOP or a change of
control owner that arrives while the joints are read wins, and the takeover is refused. A takeover needs no help
request: it is accepted from control owner nobody at any time, also at rest. The build adds two conditions that wait
for the project owner (section 21): human driving is switched on in the Thor configuration, and the human checker
accepts the measured pose. A tracked headset is not proof that someone wears it.

A press of right B is always a STOP first; if B then stays down for 1 s it also requests control. Right B is a STOP
whenever fresh SAMPLES carry the controller's buttons, also while the controller's pose is not valid, and one STOP is
sent for each stretch that B stays down. The request for control needs more: B is seen up on fresh SAMPLES first and is
then held for 1 s on fresh SAMPLES with the grip released. B that is already down when the headset wakes up or when
controller data come back stops the arm once and requests nothing; the person releases it and holds it again. Right A
follows the same rule. The alternative, a STOP only for a press that begins on fresh SAMPLES, leaves one case in which
a pressed right B stops nothing.

1. The robot app starts the assistance wait. The console shows why the agent stopped, whether the jaws may be loaded,
   whether a return to rest would be accepted now, and the countdown, and prints the steps below one at a time. The
   first build prints one takeover line in place of the steps and does not say whether a return to rest would be
   accepted (section 20).
2. The operator puts on the headset, picks up the right controller, stands behind the arm and recentres the headset view
   with the grip released.
3. The operator holds right B for 1 s. The robot app makes the human the control owner and holds. The operator console
   prints `TAKEOVER: a person owns the arm, owner number N; release the grip, then squeeze it to move` and rings the
   terminal bell (section 13).
4. While the grip is squeezed the arm follows the hand, starting from the measured pose; releasing it holds the arm and
   leaves the jaws alone. Only a new squeeze moves the arm: a grip that is already squeezed when the takeover is
   accepted, when a return to rest gives the arm back to the person, when controller data or the robot connection come
   back, or after any human motion that a released grip did not end moves nothing until it is released and squeezed
   again. Where a joint lies on its stop, as at rest, the first squeeze lifts that joint 0.025 rad off the stop, because
   the inverse kinematics keeps that margin from the joint limits; from rest the hand lifts the gripper first, because
   forward, back and down have no solution there. The alternative, a start from the measured pose itself, moves nothing
   until the hand does, and APAI's checker then refuses every step that leaves a joint on its stop. While the grip is
   squeezed a typed `rest`, `release` or shutdown is refused; a STOP always works.
5. Jaws: at the owner change no gripper command is sent, so a held object stays held, and the robot app notes the
   grip level: fully open when the jaws are proven empty and open, else the opening of the last jaw target, else the
   remembered jaw reading. The trigger asks for a jaw opening, 1.0 less the trigger's value: fully open with the
   trigger released, closed with it pulled all the way. The trigger is ignored until the opening it asks for has
   reached the grip level, which is within 0.05 of the jaw travel of it (`human.jaw_match_tol`) or across it. From
   then on the jaws follow the trigger, with the grip squeezed or not. Until the jaws have been measured fully open
   once since the takeover, the trigger cannot command more closing than the grip level. A close by the person ends
   in the light hold: when the jaws stall on an object, the stiffness drops to the light holding value. Opening
   uses the driver's guarded opening in small steps (`gripper.open_step`, 0.05 of the jaw travel). A close needs a
   fresh live view where `human.require_live_view` is on; an opening needs none. Whatever ends a person's jaw
   motion, a STOP and stale controller data among them, leaves the jaws held lightly where they are, never opened,
   and the trigger then has to reach the grip level again. Where the grip level is not known, because the jaws are
   not referenced or have neither a target nor a reading, the trigger moves nothing. This is built, behind the Thor
   setting `human.trigger_jaws`, which is on in every shipped file of the robot app and acts only where human
   driving is switched on; with it off the trigger does nothing and no jaw command is sent while a person owns the
   arm. The rule still awaits the project owner's yes (section 16), and no real controller's trigger has moved real
   jaws.
6. The person finishes the task or abandons it, puts down or releases any object, and asks for the return to rest
   (`rest` at the operator console). The console then asks "completed or abandoned?" and stores the answer in the task
   record (not in the first build, section 20). Until then the person stays the control owner; releasing the grip only
   holds.
7. At rest, holding right A for 1 s with the grip released gives the arm back. It is
   accepted while a takeover is open, from control owner human or, after a STOP, nobody. It needs measured rest,
   stillness, jaws proven empty, no fault-stop and no motion or jaw command running. With torque released the build
   refuses a takeover and a person's motion (`the motors are off; ...`, section 11) and has no refusal of its own
   for the hand-back: a `release` typed at the Thor terminal at rest while a takeover is open switches the motors
   off and leaves the takeover open, and the hand-back then asks for the same reads of the joints and the jaws as
   with the motors on. No test and no run that the example's README names gives a hand-back with the motors off,
   so whether it is accepted there is not recorded. A refusal says what to do: `type
   rest first`, or, when the jaws are not proven empty, `type stop, then open, then yes, then hold right A again`. The
   robot app closes the interrupted task for good as "completed by human" or "abandoned" (not in the first build,
   section 20) and sets the control owner to nobody. Nothing moves; the agent waits for a new task. A takeover also
   ends when the robot app is started again: a new session has no takeover open.

After a takeover a new agent task is accepted only after the hand-back of step 7. From the accepted takeover to the
accepted hand-back the takeover is open: the Thor and the agent program both refuse a new task, also when a STOP has
made nobody the control owner in between, and a fault-stop and its `reset` leave the takeover open. In all cases a new
task also needs: no person owns the arm, no fault is latched and the measured pose is inside the agent limit set.
Otherwise the agent program's terminal refuses the task and names the reason: "takeover not finished: give the arm back
with right A at rest", "fault-stop: run the console command reset", or "arm outside the agent's limits: run the console
command rest".

Human driving uses Isaac Capture's controller source, clutch and end-effector rate limiter (the nodes today's teleop
uses), then APAI's inverse kinematics with no random restarts, an iteration limit and a joint-jump check, then the human
checker and APAI's driver, at 50 Hz. One solve of the inverse kinematics runs at most 50 iterations, and a solution
more than 0.05 rad from the one before it counts as a jump and holds the arm. The controller's poses are rotated into
the robot's base axes for a person who stands behind the arm; a setting turns them when the recentred headset view is
at an angle. LeRobot's package is not imported into the robot app. Its limiter and clutch values are not copied,
because none of the sources of section 22 holds them: the first values are computed on the arm's kinematic model, and
the first session with the headset replaces them with measured ones. Copying tested values stays the better start once
a tested configuration is found; until then the first values are slow on purpose. If APAI's solver fails a trial on
the test arm with the real headset, the fallback is LeRobot's solver. It cannot be loaded beside APAI's kinematics as
they are built today (section 16), so that fallback means one merged environment that neither stack has been tested
in. It is not in the estimate. The first profile is the Quest with the right controller, as in the first proposal;
gloves are started only if the chosen demo profile needs them. For tests and for demo 2 on one machine the controller
can be a script, a synthetic controller, in place of the headset. A synthetic controller takes over and drives only a
test arm; the Thor refuses it on the real arm.

Human-only mode (a person drives from the start, no agent) is today's LeRobot program, unchanged. It is not run at the
same time as the robot app, and it does not meet the never-go-limp rule today.

## 13. Operator: tasks, consent, commands and view

| Terminal | What the operator types there | What it shows |
|---|---|---|
| Agent program (workstation) | Tasks, typed one after another at its prompt or given as one start argument; the typed `yes` at a task's first skill that moves the arm; a second typed `yes` there when the motors are off, before they are switched on again (section 11); the typed `path is clear` when the program ends and asks whether the arm returns to rest (section 6); Ctrl+C, which is a STOP | Task progress; the robot's state before the question; each refusal with what to do about it |
| Operator console (workstation) | `stop`, `reset`, `rest`, `release`, `on`, `open`, `close`, `shutdown`; `state`, `attach` and `help`, which send nothing; `quit`, which leaves the console and sends nothing; Ctrl+C, which is a STOP; `takeover` and `handback`, which the shipped files switch off (section 21) | A status line (robot connection, control owner, stop level, motion, jaws, torque, assistance wait, permit, view age), printed when it changes and at least every 10 s; every EVENT; a block of lines for each help request; a takeover line; refusals with reasons, also those another client got; the loss of the robot connection, with the terminal bell |
| Thor terminal | `stop` and Ctrl+C, each a STOP that needs no network; `rest`; `release`; `on`, which switches the motors on again after they were switched off at rest (section 11); `reset`; `open` and `close`, which move the jaws while nobody owns the arm; `quit`, which is the shutdown; `help`; `reference`, which the shipped files switch off (section 21). The power, calibration and hold confirmations at each start of the robot app on the real arm | Every EVENT of the robot app as a line: owner changes, STOPs, the countdown with the terminal bell, refusals with reasons. It prints no status line |

- The "yes" is asked at the terminal that sent the task. In the first build only the agent program's terminal sends
  tasks, so it is asked there. It is asked once per task, at the task's first skill that moves the arm; skills that
  only look or think take no arm and ask nothing. Before the question the agent program prints the robot's state from
  ROBOT_STATE and refuses the skill, without a question, when a person owns the arm or a takeover is open, when the
  control owner is not nobody or when a fault-stop is latched. The question names the task,
  the first motion skill and the joint speed limit; one typed `yes` covers the later motion skills and retries of the
  same task. Motors that are off at rest refuse no task: after that `yes` the agent program asks a question of its
  own about the motors and, on a second typed `yes`, sends OWNER_REQUEST "motors_on" (section 11); any other
  answer leaves the motors off and refuses the skill.
  The agent program then sends OWNER_REQUEST "agent", which quotes the session id, owner number and stop
  number of the ROBOT_STATE shown before the question, so a `yes` typed for one state of the arm takes nothing in
  another. The prompt follows today's launcher and runs in the agent program. Whether an earlier run left the arm
  holding is read in two places, neither of them a file the agent program opens: the agent program reads ROBOT_STATE,
  and the robot app checks APAI's hold marker when it starts on the real arm. The power confirmation is asked by the
  robot app at its start. Opinion on the open question of tasks from the Thor terminal: not in the first build; it
  costs about 200 to 250 lines of source and 150 of tests and points away from an operator in another city. The task
  record is APAI's task report, one line per task in the agent program's run folder. It has no field for where a task
  came from; tasks from a second terminal need that field first.
- A task that succeeds gives the arm back: the agent program sends OWNER_REQUEST with the control owner nobody, and the
  arm holds where it is with motors on. The return to rest is the operator's `rest`, or APAI's rest tool in a task
  that calls it: the agent program sends that tool's moves as moves of the agent and then asks the robot app for the
  motors off, which the robot app switches off only with the arm measured at rest (section 11). A task that is
  cancelled with Ctrl+C at the agent program's terminal sends neither: its STOP ends the agent's ownership (section
  9). A task that the model itself ends without success gives the arm back as well, with the control owner nobody
  and no help request: the agent program prints the model's report and asks for the next task. A task that got
  stuck, was blocked, ran out of steps, was ended by an error or ended with no report while the agent owned the arm
  sends "help needed" (section 10).
- A skill stop ends the task with a help request and latches the checker in the agent program. The next task's
  question names the skill stop, and that task's typed `yes` clears the latch once the Thor has granted the arm and a
  ROBOT_STATE shows the agent as control owner with stop level none. Nothing on the Thor is cleared by it. The
  alternative, a restart of the agent program after every skill stop, reloads the models and adds no safety step,
  because the same typed `yes` starts the next task after the restart. The setting `task.clear_skill_stop` (first value
  true) selects between the two.
- What the operator types to confirm. A return to rest is confirmed with the typed phrase `path is clear`: after `rest`
  at either terminal, after the shutdown when the arm has to move, and at the agent program's terminal when that
  program ends and asks whether the arm returns to rest (the request "park", section 6). Apart from one more
  phrase, `yes` confirms everything else that asks: `open`,
  `release` and a shutdown with the arm at rest at the operator console, `on` at either terminal, and a task's first
  motion and the motors on again at the agent program's terminal. That phrase is `jaws are empty`, which is
  asked in two places. The first is after a `rest`, a
  `release`, a shutdown or a typed hand-back that the robot app refused only because the jaws are not proven empty:
  the Thor terminal (after `rest`, `release` and `quit`), the operator console and the console command (after
  `rest`, `release`, `shutdown` and `handback`) then tell the operator to look at the jaws. `jaws are empty`, typed
  as the next line, has the command asked again with its own question and sent with the operator's word, which the
  robot app takes in place of its measurement (section 11). The second is the typed `reference` of section 21,
  which is switched off. At the Thor terminal `release`, and `quit` with the arm at rest, ask nothing. Case and
  spacing do not count; any other answer confirms nothing and nothing is sent. `stop` typed where a
  question waits is a STOP and confirms nothing. Whatever is typed before a question is written is dropped. A
  confirmation is typed by a person at a terminal: the operator console, the agent program and the robot app on the
  real arm ask nothing from a pipe or inside the environment of an AI coding agent. The alternative for a return to
  rest, a plain `yes`, lets one habitual word start a motion of the arm.
- At the operator console `quit` leaves the console and sends nothing; the robot app runs on. The robot app's shutdown
  is `shutdown` at the console and `quit` at the Thor terminal. The alternative, `quit` as a second word for the
  shutdown, lets a person who only wants to close a window shut the robot down; the choice made costs one word with two
  meanings at two terminals. Ctrl+C at the console takes back an open question and then sends a STOP, as at the Thor
  terminal and at the agent program's terminal, and never ends the console. The alternative, a Ctrl+C that only takes
  back the question, leaves one of three terminals where a person in a hurry gets no STOP; the choice made costs a
  running task and a running assistance wait to a person who meant only to end the program.
- `stop` at the operator console sends one STOP. The console prints `stop: taken` once a ROBOT_STATE of the same
  session shows a higher stop number. Without one within 5 s it prints `STOP NOT CONFIRMED`, rings the terminal bell
  and names the STOP key at the Thor terminal. A console that is not attached to the robot link service writes a raw
  STOP: a STOP sent without an attachment, which the service forwards and nothing confirms. The console then prints
  `STOP written; not confirmed: look at the Thor terminal`.
- At its start on the real arm the robot app makes its checks first and asks nothing for them: the calibration named
  at the start is the one in its configuration; no hold marker of APAI is on file for the arm's CAN bus; no operator's
  confirmation for APAI's connect is on file, because with one APAI's driver would connect an arm that it does not
  measure at rest. A typed `yes` never releases a hold. It then asks three questions, each answered with a typed `yes`:
  `Is the arm powered?`, `Connecting cuts motor torque. Is the arm at rest and supported?` and `Is the calibration in
  use <name>?`. APAI's driver then reads every joint twice and refuses an arm that is not at rest before anything that
  cuts torque. A typed `yes` is a person's word; the reading is a measurement.
- The console command sends one command of the operator console (`state`, `stop`, `reset`, `rest`, `open`, `close`,
  `release`, `on` or `shutdown`, and `takeover` or `handback` where the setting of section 21 allows them). One
  module sends RESET, OPEN, RELEASE, SHUTDOWN and the typed `rest` for both programs, so one place on the workstation
  can ask for a torque release.
- The person who wears the headset cannot read the console, so the console rings the terminal bell by kind of EVENT:
  once for a bell of the assistance wait and when the arm stops following the hand for a reason other than a released
  grip or a STOP; twice for a takeover that began or ended; three times for a refused takeover or hand-back. Whether a
  terminal rings three bells apart is not tested.
- `open` opens the jaws through the driver's own opening after one typed `yes`. It is typed at the operator console
  or at the Thor terminal. The robot app accepts it only with control owner nobody, no fault-stop, the motors on and
  no other motion or jaw command running or waiting. It is the way to let go of an object without a headset. During
  a takeover the person lets go with the trigger (section 12 step 5); without the trigger the way is `stop`, then
  `open`, then `yes`, then right B again or `rest`.
- View: the owner asked whether the browser view should be served from the workstation or the Thor. Opinion: the
  workstation, because the video and the perception data are already there and the Thor stays small; the first proposal
  does the same. Both demos show live camera video and robot state in a workstation browser, from a bounded in-memory
  source that does not depend on recording. There is one browser page, the viewer's; with its live picture showing
  in a browser on the workstation it is the live view, and no other program serves one. The viewer runs the current
  experiment's viewer (`thor_viewer.py`) unchanged on the run folders of the agent program, and a wrapper of this
  example (`viewer_live`) adds two live parts to its page. The live picture is the left camera
  frame as the camera receiver has it, at most 960 pixels wide and packed as JPEG, up to 30 a second (`--live-width`
  and `--live-fps`; the defaults are `DEFAULT_WIDTH_PX` and `DEFAULT_FPS` in `viewer_live/settings.py`). The live arm
  is the arm drawn from each ROBOT_STATE as it arrives, with a line under it that gives the measured joints, the jaw
  opening, the control owner, the motion state and the torque of that ROBOT_STATE. Neither is stored. The rest of
  the page comes from the debug recording of a run: the occupied cells of the obstacle grid, the refusals of the
  checker in the agent program with the pose refused, the camera frame stored with each map picture and the timing of
  the run. The page can replay a run, and a button over the picture switches between the live picture and the stored
  one. The viewer reads the camera receiver as a second reader beside the agent program, so the view works while the
  agent program is down. One viewer reports at a time, because two would send the Thor two VIEW_STATUS in turn: a
  viewer that finds another one attached to the robot link service sends none. The viewer only looks: no button of
  its page commands the robot, and VIEW_STATUS is the only link message it sends. The alternative, a camera window
  opened by the program that reads the camera receiver, is smaller and needs no web server; it cannot say whether a
  browser shows the picture, which the takeover rule needs.
- Who can open the page. It is open to every machine on the internal network, without a token: the workstation's
  start script `run_workstation.sh` starts the viewer on the address its variable `AGENTIC_REMOTE_VIEWER_HOST` names,
  the workstation's address on the internal network or every interface, and without the viewer's access token. Every
  machine that reaches that address then sees the camera pictures, the obstacle grid and the timings of the runs, over
  HTTP without encryption. Without the variable the viewer listens on 127.0.0.1 only and asks each browser once for
  its access token, a secret in a file of the example's state folder. Opening the page from another machine does not
  make that page the live view (the next item). A live view in a browser on another machine is the alternative for an
  operator in another city; it needs a way to tell the operator's browser from any other on that network, which the
  first build does not have.
- A takeover needs a fresh live view in the workstation browser, and loss of that view while a person drives holds the
  arm, wherever the Thor setting `human.require_live_view` is on, as it is in the rig's file. The viewer reports the
  view's freshness to the Thor through the robot link service (VIEW_STATUS, section 6). Which page's reports count:
  - Only a page opened in a browser on the workstation can be the live view. The viewer takes a browser as one on the
    workstation when its address is a loopback address or the address the request arrived on; it reads both off the
    connection, and no header counts. Only such a page is given a view token. A page opened from another machine is a
    watch-only page: it shows the live picture and the live arm, and the viewer refuses its reports.
  - Each load of the page from the workstation makes a fresh view token the only valid one. So one page at a time
    counts: the page loaded last. A reload, or a second tab on the workstation, makes the view not live until the
    page loaded last has drawn a picture. The page loaded before it goes on as a watch-only page, says so, and
    becomes the live view again only when it is loaded again.
  - Replay never counts as live. A page that shows the stored picture, a replay or an older run reports that its
    live picture is not showing.
  - The view counts as live only while all of these hold. The page with the valid view token has drawn a live
    picture of the camera connection and reported it less than 1 s ago, with its browser tab visible and at least
    three quarters of the live picture inside the browser window, and it fetched that picture less than 1 s ago. A
    ROBOT_STATE younger than 1 s gives the Thor's wall clock. The picture arrived 3 s or more after the viewer's
    first ROBOT_STATE and 3 s or more after the newest step of the Thor's wall clock against its monotonic clock (as
    when the clock is set). The picture's exposure lies at most 0.05 s in the future of that clock.
  - The 3 s are a quiet time. After the viewer's first ROBOT_STATE, so after every start of the viewer, and after a
    step of the Thor's wall clock, the view is not live for at least 3 s, because a picture that arrived earlier can
    be dated on another wall clock than the one its age is measured on.
  - The age in VIEW_STATUS is the larger of two ages of that picture: since its exposure, on the Thor's wall clock,
    and since its arrival on the workstation. The viewer sends a VIEW_STATUS every 0.2 s, also while the view is not
    live. These times are fixed values in `viewer_live/settings.py` and `viewer_live/view_model.py`; no option and no
    file sets them.
  - When the viewer does not answer, the page keeps its last picture, dimmed, says how old it is and asks again
    every second. The view is not live meanwhile.
  - When the viewer was restarted, the viewer instance that answers is not the one that served the page, and the
    page's view token is of no use. The page then loads itself again, once for each viewer instance, and is the live
    view as after a load by hand, once the quiet time has passed. It does so only in a visible tab and only while
    the viewer that runs has given no page a view token, so that no page takes the live view from another by a load
    of its own. A page that shows the stored picture or a replay waits until its button for the live picture is
    pressed. A watch-only page does not load itself: it becomes the live view only when a person loads it again on
    the workstation.

  Each of these cases only holds the arm. A code that the operator reads from the page and types at the console is
  the alternative to the view token; it proves that a person read the page and costs a typed step before the headset
  goes on. Known limits: a visible tab proves that the tab is in front, not that a person looks at it; a program of
  the same user that plays the browser, or that attaches to the robot link service in the viewer's role, can claim a
  live view; and a browser on another machine whose connection a program on the workstation passes on counts as one
  on the workstation. All values are first values. Tests ran the page in a browser without a window; no person has
  looked at it in a browser window, and no report of a browser has reached a robot app on the rig.
- The debug recording. Started with `--record-debug`, the agent program stores in its run folder what the viewer's
  page shows of a run. Two rules of the agent program decide what a run keeps (`agent/launcher.py`,
  `agent/move_paths.py`, `agent/debug_recording.py`).
  - Each move's planned joint path. The agent program plans the setpoints of a move on the workstation and sends
    them in one MOVE. It writes one line for each MOVE into the file `move-paths.jsonl` of the run folder, in every
    run, with and without `--record-debug`: every setpoint in order, the numbers the MOVE quotes, and how many of the
    setpoints the Thor sent to the motors. With `--record-debug` the debug recording also holds one event for each
    MOVE with the same path (`robot.move_path`). The path is the planned one, not single targets timed at the motors:
    the motor driver runs on the Thor, and the workstation records no motor command.
  - Map pictures. The debug recording stores map pictures at the normal rate, 2 a second by default
    (`--debug-map-hz`), during a task and during a takeover. It thins them only while the agent program waits for a
    task at its prompt and ROBOT_STATE has shown no control owner and no motion of the arm for 5 s (`ARM_QUIET_S`);
    it then stores one map picture every 10 s by default (`--debug-idle-map-s`). A ROBOT_STATE that shows a control
    owner (the agent, a person or recovery) or motion brings the normal rate back at once, and with no ROBOT_STATE
    newer than 2 s (`ARM_STATE_AGE_S`) the normal rate is kept. A takeover happens while the agent program waits for
    a task, so this rule records it at the normal rate: for as long as a ROBOT_STATE shows the person or recovery as
    control owner, also with the grip released, and for 5 s after. After a STOP inside a takeover nobody is the
    control owner, and the map pictures are thinned once the arm has stood for 5 s, until the person takes the arm
    again. The rate decides only what is written: the obstacle grid, the checks and the arm see none of it.
- According to the source code of the headset's CloudXR client, a headset with Headless on shows a flat dark-blue field.
  How the operator sees the browser view while wearing the headset is settled by the headset check of section 19.

## 14. Start-up order

Step 3 states this design's placement of the human path, in which the Thor dials the Teleop Session Server. Section 21
states where the build places it and the start-up that follows; that placement waits for the project owner's decision.

1. The workstation starts its programs; none waits for a headset. The robot link service comes first, because the
   operator console, the viewer and the agent program are its clients: the agent program ends with an error without
   it, the console tries to attach for a minute and after that on a typed `attach`, and the viewer serves its page
   without it and attaches by itself once the service runs. The camera receiver, the viewer and the operator console
   follow; the viewer's page is opened in a browser on the workstation and its tab is kept visible. The agent program
   is started after step 3. For demo 2 the operator also starts CloudXR's service and the Teleop Session Server, by
   hand.
2. The Thor starts the robot app and the camera sender. The power, calibration and hold confirmations stay at the Thor
   terminal (section 13). The robot app connects to the motors at its start, with the arm measured at rest, and
   never reconnects by itself. After a torque release a typed `on` connects them again, with the arm measured at
   rest (section 11). The robot link service and the robot app start in either order: the robot app dials again
   until the service answers (section 5).
3. The Thor dials the robot link service and, in the background, the Teleop Session Server. The robot app sends WELCOME,
   and state and video start. The operator then starts the agent program. It compares WELCOME with its own settings
   (protocol version, joint count, setpoint period, the arm profile where its configuration names one, and the name of
   the camera calibration) and ends before any task when they do not fit. WELCOME also names the declared driver, and
   the agent program refuses to drive the real arm with settings meant for runs on one machine (camera frames with
   invented times, the minimal stack, the obstacle grid stand-in of section 21). The agent is ready when robot and
   perception are ready, whether or not a headset was seen: the robot app has answered within
   `link.connect_timeout_s` (first value 10 s) and the obstacle grid is usable within `task.grid_wait_s` (first value
   20 s); otherwise the agent program ends.
4. The operator types a task at the agent program's terminal, and its "yes" when the agent program asks for it at the
   task's first skill that moves the arm.

## 15. What follows from the link count

The rows that name the teleop thread, or SAMPLES read on the Thor, state this design's placement of the human path;
section 21 states what follows from the build's placement.

| Topic | First proposal (one link) | This proposal (two links) |
|---|---|---|
| Isaac Capture code | Teleop protocol, client, host, server and `TeleopSession` extended, with negotiation for old clients | A Python wrapper for `core::Link`; teleop code untouched |
| Workstation end | The API lives in the Teleop Session Server | The API lives in the robot link service |
| Connections the robot app watches | One | Two, each with its own "alive" state; robot data does not pass through the Teleop Session Server |
| Agent before any headset | By a protocol change | By the teleop thread, not yet run |
| CloudXR, OpenXR or tracker fault | If fatal, the robot connection is lost | Human input only is lost |
| Teleop Session Server dies during the assistance wait | The timed return is cancelled and the arm holds | The wait goes on; the arm returns to rest if the conditions of section 11 hold, although no takeover is possible. With the timed return switched off, as in the rig's files until its bench tests are run (section 11), the arm keeps holding when the wait runs out |
| Right B as STOP | The server can catch even a very short press | Read on the Thor from SAMPLES; a press shorter than one pass of the teleop thread (20 ms) can be missed, because each pass sees only the newest SAMPLES. The bench test of A and B measures this |
| What is tested again | Today's remote path: its automated tests and hardware goals 1, 2, 3 and 5 | New bench tests of the teleop thread and of two connections up or down |
| A later move to one connection | Not needed | The same edit as the first proposal, later (section 20); nothing above the three calls changes |

## 16. Open points

Taken from the first proposal on the project owner's decisions of 2026-11-13: permits for the obstacle check (section
7); hold without a timer when the headset or controller is lost while a person drives (section 10); a takeover that ends
at rest, with the arm given back only when right A is held at verified rest (section 12, steps 6 and 7; the owner said
this can be changed later); and no human motion without a live view in the workstation browser (sections 12 and 13).

Where the first proposal's text (16:22) still differs for reasons other than the link count. The first two rows are the
hand-motion code and its input devices. Every other row is a gap this file fills. The rows on typed `open`, the robot
app's death and the 90 s floor record what the project owner has already said. In the rows STOP, Return to rest and
Checker on the Thor the first proposal names a way to reuse APAI's code, and reading that code showed that another way
is needed or is smaller; the behaviour asked for is the same. In the last two rows the first proposal leaves a choice
open, states only a goal or is silent.

| Topic | First proposal | This file |
|---|---|---|
| Human driving | Keeps "existing human retargeting/IK initially", which today is LeRobot's solver (placo) | Isaac Capture's controller nodes with APAI's inverse kinematics (section 12); the project owner chose this version. This file's reason: placo loads only with an older build of the Pinocchio kinematics library's dependencies than the one APAI's Thor environment is locked to, and one environment holds one of them. APAI would have to run its kinematics, gravity torque and checks on the older build, which nobody has run; LeRobot's numpy and OpenCV upper bounds are also below APAI's installed versions |
| Gloves | Demo 2 attaches the current Quest and glove input path; the controller is its first interaction profile | The same first profile, the right controller. Gloves are started only if the chosen demo profile needs them (section 12) |
| Typed `open` | None | Section 13; the project owner agreed to it earlier |
| Return to rest after the robot app dies | Recorded as an unmet requirement | Hold; the project owner accepted that |
| Floor of the assistance wait | 120 s at first, configurable; no floor | 120 s at first, never below 90 s; the project owner asked for 90 s or more (section 3) |
| STOP | Reuses `SafeArm.stop()`, which calls the driver's `stop()` | Built on `halt()` and `estop()`; section 9 says why |
| Return to rest | Reuses `move_to_rest()` | Neither a call of `move_to_rest()` nor `SafeArm.move_joints`: the robot app plans the straight joint-space legs of the return itself, with APAI along the route of APAI's rest tool, and sends them through its own setpoint loop on the recovery checker, the loop that also sends the agent's moves; section 11 says why |
| Checker on the Thor | Extracts the camera-free checks from APAI's checker | No extraction: APAI's whole checker, created without a grid. That switches off only its grid checks, and the permit check of section 7 stands in for them. One instance per control owner (sections 4 and 8) |
| Torque release | A normal return to rest or shutdown "may" release torque; keeping it on at rest is also allowed | Decides the open choice: a typed `rest` outside a takeover ends at measured rest with the jaws closed and the motors off, as a typed `release` and the shutdown command do; the timed return and a `rest` typed while a takeover is open end with the motors on; section 11 names every end at which torque is released |
| Rules | States goals or is silent | Proposes rules. They are this file's own, not owner decisions, and stand unless the owner objects: how "jaws empty" is measured, the legs of the recovery move and its speed, and the path-clear prompt (section 11); what STOP does to a closing gripper, the STOP key at the Thor terminal, Ctrl+C as STOP, the shutdown command and contact as a hold that is not latched (section 9); `rest`, `release` and shutdown at the Thor terminal (section 13); contact spots that do not bind a person (section 8); the warning bell and a new task that cancels the wait (section 10); building the shared part first on one machine (section 17); the typed phrases `yes`, `path is clear` and `jaws are empty` (sections 11, 13 and 21); Ctrl+C as STOP at the operator console and at the agent program's terminal as well as at the Thor terminal (section 9); `quit` at the operator console, which leaves the console and sends nothing (section 13); and the declared driver in WELCOME, which decides whether the agent program accepts settings meant for runs on one machine (sections 6 and 14). The same holds for detail the first proposal leaves to the build, such as the link message table (section 6), the recovery checker's settings (section 8), the first form of the live view (section 13) and the first values of limits, speeds and times. The trigger limit at takeover (section 12 step 5) is also this file's own rule; it is listed below because it awaits a yes |

Awaiting the project owner's yes:

| Point | What is asked |
|---|---|
| Trigger limit at takeover (section 12 step 5; either proposal) | When a person takes over while the arm holds something, the trigger becomes the gripper control. Proposed: the trigger does nothing until it has been pulled as far as the agent had closed the jaws, and it cannot squeeze harder than the agent did until the jaws have been fully opened once. Taking over can then neither drop nor crush the object. The rule is built as section 12 step 5 states it, behind the Thor setting `human.trigger_jaws`, which is on in every shipped file of the robot app. With the setting off the trigger does nothing under a person. The owner's yes decides whether it stays on for the first session with the headset on the arm |
| The robot link service (section 5; this proposal only) | One more workstation program. It is built (as built: about 2,250 lines of source) and it does more than hold the robot connection: it decides which local client may send which link message, it holds permits back after a STOP until the Thor shows the higher stop number, and it asks for help when the agent program is gone. It stays a separate program unless the owner objects |
| Permits or the grid on the Thor | The owner chose permits as the easier start, and permits are built: the Thor's permit check and the workstation's check ahead hold, as built, about 2,300 lines with their tests, against this file's estimate of 700 to 1,100. Sending the grid to the Thor (section 20) comes on top of the permits, about 350 to 600 lines by the estimate behind this file. Permits stay unless the owner switches |
| A silent agent after a hold | If a permit runs out during a move, the arm holds. When the workstation's check ahead has ended the permits, the agent program ends its task and asks for help, which starts the assistance wait, and the robot link service asks for help when the agent program's connection closes (section 5). What stays without a rule is an agent program that stops sending and keeps its connection open: nothing calls that "failed", so no assistance wait starts and the arm holds until someone types a command. Should some seconds without any agent command after such a hold count as failed? Model thinking before a move would still never count. The rule is not built: the arm holds until someone acts (section 21) |

Section 21 lists everything else that waits for the project owner: what the lead of the build, the AI agent that
directs it, chose where the owner has not decided, and what ships switched off until the owner decides. Its first item
is the placement of the human path.

## 17. Build plan

| Phase | What is built | Gate |
|---|---|---|
| P0 | Robot app, control owners, STOP, hold, return to rest, torque rule and link message module on one machine, with a test arm over a loopback version of the three calls. Audit of every call that can disable a motor | Refused or expired commands cannot move the arm; no disable call is reachable away from rest |
| P1 | The Python wrapper for `core::Link`; the robot connection over a real link with the robot link service and its local API (the phase where the two proposals differ most); the agent program on the workstation: its arm adapter, the check ahead that sends permits, the camera input and the start and end of a task | A test arm completes commands over a real link between programs on one machine; nothing replays after a reconnect. The camera frame bridge copies each stereo pair and discards a copy whose slot may have been overwritten meanwhile; stereo pairs and the calibration id are kept; both are tested against a stand-in for the camera receiver. The comparison of depth and one grasp pose from compressed camera frames with uncompressed ones needs the camera on the Thor and is a bench test of section 19 |
| P2 | Demo 1 with the live view and the operator console; the start on the real arm at measured rest, the files for the rig and the workstation's Python environment for the full stack | On the rig: a repeatable task without a headset, three successes in a row at 0.1 rad/s; working `stop` and `rest`. Without hardware, on APAI's mock arm: the moves of a pick and place run under permits in every run with the console and the viewer attached, the task `open the gripper` ends with success, and `stop` and `rest` typed at the console work. The pick and place itself ends with a help request there, because APAI's mock scene confirms neither the grasp nor the place |
| P3 | Demo 2, built on one machine with a synthetic controller and a simulated arm; the headset checks of section 19 come before the first human motion on the rig | Takeover and the assistance wait work without restarting the robot app; a STOP during the wait prevents the timed return. Tests cover both on one machine, with the programs of demo 2 attached to the robot link service over a real link and a synthetic controller in place of the headset's (`test_demo2_programs.py`). Two of them run demo 2 as separate processes by the recipe of the example's README: on the simulated arm, and on APAI's mock arm with a task of the agent that fails first. Neither recipe has been run by hand. On the rig the timed return ships off (section 21), so there the STOP shows that the wait is cancelled |
| P4 | Acceptance of both demos: without hardware by the tests and the runs on one machine that the example's README describes, on the rig by the checklist of that README with every value written down. Then the fault table under each control owner, measured delays, launch and recovery instructions | Results written down with what remains open |

P0 needs no GPU, no Isaac Capture build and no robot. The tests run in a Python 3.12 environment of their own; those
that use APAI need APAI's source on the Python path. This was tried on the workstation on 2026-11-13: 16 APAI tests of
the checker, the kinematics and a mock-arm grasp and place pass there, and 9 of the 10 Thor shutdown tests; the rest of
APAI's tests were not run. There are two test arms behind the same calls. The simulated arm is written for this example
and needs numpy only: it has failed joint reads, a read time per joint, a joint blocked at a stop, motor faults, jaw
travel over time, an object between the jaws and a simple joint-space checker of its own. Most tests run on it without
APAI, and so do the robot app's two self-tests: scripted runs of the robot app on that arm, one on a loopback link and
one through a robot link service over a real link on one machine. APAI's mock arm runs under APAI's checker,
unmodified, in the tests that have APAI's source. On the workstation, builds, tests and perception use GPU 0 only.
GPU 0 is index 0 in PCI bus order: the workstation has two cards of the same kind, so the start script of every
workstation program that can use a GPU, and the agent program itself, set `CUDA_DEVICE_ORDER=PCI_BUS_ID` and
`CUDA_VISIBLE_DEVICES=0` themselves.

Work happens in a separate checkout on a branch from the remote teleop session branch; nothing is pushed. Example code
goes under `examples/agentic_remote/` and its tests under `tests/python/examples/agentic_remote/`. The Python wrapper
for `core::Link` is library code under `src/core/remote_link/python/` and `src/python/isaaccapture/remote_link/`. APAI
is imported from its own checkout through an environment variable, and tests skip without it. The Thor adapter files of
the current experiment exist only as uncommitted files in another checkout; what the robot app needs from them is
written new here. The agent program's full stack imports the experiment's folder, read only, through a second
environment variable: its behaviour hooks, its obstacle grid bridge, its depth code and its cloud model client. A
change in that folder can change the agent's behaviour; copying the parts used into this example gives a stable base
and costs 1,500 to 3,400 lines to keep in step. The minimal stack, which the tests use, needs none of it. The camera
connection's programs come from the checkout of pull request #1410 through a third variable, and the full stack runs in
a Python environment of its own on the workstation, built once by a script of the example.

## 18. Time estimate

Hours are Claude's active working time (coding, tests, builds, debugging), not a human engineer's time and not elapsed
time. Sessions on the arm with the operator are counted separately. The rows and scope are the first proposal's, so the
two can be compared. The first row depends on the link count; the fifth differs by half an hour (the teleop thread) and
the sixth has the same size with different tests. The two totals are the same within the error of the estimate. The
half hour for the teleop thread follows this design's placement of the human path; section 21 gives the size of what
the build has in its place.

| Work | First proposal | This proposal |
|---|---|---|
| Connection layer and local API | 4 to 7 h | 3 to 5 h |
| Thor execution, real feedback, hold, checks, torque rule | 6 to 10 h | 6 to 10 h |
| Camera, perception on the workstation, obstacle check | 5 to 9 h | 5 to 9 h |
| Live view and operator console | 4 to 7 h | 4 to 7 h |
| Human path, STOP, assistance wait, takeover | 4 to 7 h | 4 to 7.5 h |
| Integration tests, instructions, fixes found on hardware | 7 to 15 h | 7 to 15 h |
| Total | 30 to 55 h | 29 to 53.5 h |

A first task running end to end on the rig without a headset needs a session with the operator; the work before it,
estimated at about 14 to 24 hours, is built (phases P0 to P2). On one machine, on APAI's mock arm, the task
`open the gripper` runs end to end with success, and a pick and place runs its moves and ends with a help request
(section 17). On the rig the example's README names one task on the real arm: the rig owner's report of 2026-11-16
of a pick, a lift, a hold and a place of an object at 0.1 rad/s, with the agent program on the workstation and the
robot app on the Thor. The repository holds no record of that run. Of the run of 2026-11-17 that the status
paragraph at the top of this file names, the README states the count of its moves and jaw commands and not how its
tasks ended. The gate of P2 on the rig, three successes in a row, is not recorded as passed. The line estimates of
this file are low by a factor of two to three, as the
two that have an as-built count show (the `core::Link` wrapper in section 5, the permits in section 16). The hours of
the table are not checked against the build: the active hours are not recorded per row. Two to four sessions with the
operator are needed in both designs. The link count changes the risk more than the hours: with two links nothing that
works today is edited.

## 19. Bench tests that need the operator

The items that name the teleop thread, or controller samples that reach the Thor, state this design's placement of the
human path; section 21 states the tests that take their place in the build.

People. Demo 1 on the rig needs two: the operator at the Thor terminal, and a second person who supports the arm where
a test says so and stands at the 48 V switch. Demo 2 on the rig needs the person who wears the headset (the wearer),
who types nothing and cannot read a terminal; a person at the Thor terminal; and a person at the workstation who
watches the live view and the operator console (the person at the Thor terminal does this only where both keyboards
are within reach). A further person supports the arm while the stop and return tests of demo 1 have not passed. The
abort in every test is `stop` or Ctrl+C at the Thor terminal, which needs no network and no headset. Right B is under
test itself and is not the abort. The 48 V switch drops the arm and is not the abort. The robot app is never ended or
restarted with the arm raised.

Before a session on the rig the operator brings the code to the Thor (nothing is pushed), builds Isaac Capture and
the camera sender there, and writes the Thor's address into the robot link service's file for the rig; the example's
README lists the steps. The sessions of 2026-11-16 and 2026-11-17 ran the programs on both machines. The README's
commands for bringing the code to the Thor, for the build there and for the variables of each machine were written
after the first of those sessions and have not been run again as they stand there.

The arm has moved under the robot app: the run of 2026-11-16 that the status paragraph at the top of this file
names, at 0.1 rad/s. No record of that session says which steps of the README's checklist passed, and the README's
tables of what was run hold no row for a test of this section on the rig. The lists below state what this design
asks for.

Before the arm moves under the robot app:

- Read the motors' CAN timeout parameter (0x7028) on all seven motors.
- Kill the current app during a raised hold with the arm supported by hand, empty and loaded, and watch for 10 minutes.
  Log motor temperature during a long loaded hold with a tool outside the robot app: APAI's drivers give the robot app
  no motor temperature and no motor fault.
- Jaw rule with a cube, a wallet, a thin card and empty jaws.
- Return to rest from poses across the demo workspace with the arm supported, and once with a foam block in the path.
- Delay from typed `stop` to the arm holding, during an agent move.
- The 0.35 s joint feedback limit while the arm holds, at the read rate of a hold.
- Permit timing: how long the workstation's check ahead takes and how late a permit arrives; then an obstacle placed in
  the path during a move.
- Depth and one grasp pose computed from compressed camera frames, compared with uncompressed ones of the same still
  scene (the P1 gate of section 17). The checklist of the first session, in the example's README, has no step for it.

That checklist runs in another order than the list above. It moves the arm, at 0.1 rad/s, with the timed return off and
a second person supporting it, before five of these tests: the CAN timeout parameter, the app killed during a raised
hold, the jaw rule with objects, the returns to rest across the workspace, and the 0.35 s limit in a hold. In that
session it measures the delay of a typed `stop` (from the operator console, the agent program's terminal and the Thor
terminal; above 0.25 s the session ends), the permit timing and the arm's stopping distance, an obstacle placed in the
path, the hold when the agent program, the network cable, the camera sender or the robot link service goes away during
a move, and one return to rest from a raised pose. The five tests come before the speed is raised, the timed return is
switched on or the arm is left without a person beside it. Running the list above first costs one more session before
any motion; the checklist's order costs first moves on motors whose behaviour after a program death is not measured,
on first values that nobody has measured. Which order stands waits for the project owner (section 21).

The checklist and the tests of this section name 0.1 rad/s. The rig's files hold 0.2 rad/s for the agent and the
return to rest and 0.8 rad/s for a person (section 8). A session at 0.1 rad/s sets those keys to 0.1 on both machines
before the programs are started.

Before any human motion on the rig. Each test runs first with the robot app on a simulated arm on the Thor, before
power, and then on the rig at 0.1 rad/s:

- Headset check: with Headless on, can the wearer see the arm and the browser view below the visor; with Headless off,
  does the room show and do controller samples keep reaching the Thor. The check passes when the wearer sees the arm
  itself in one of three settings: Headless on and looking below the visor, Headless off, or the headset off the face
  and resting where it still tracks the right hand. Whether the person who drives must also see the live view
  themselves waits for the project owner (section 21).
- Right A and B over the teleop connection (only grip and trigger have been seen on hardware), including short presses,
  and right B with the controller outside the view of the headset's cameras: the STOP must arrive.
- The teleop thread beside the 50 Hz motor loop; server started before any headset connected; re-entry after a
  connection loss.
- APAI's inverse kinematics under a moving hand, on the simulated arm with the real headset.
- The direction of the hand against the arm: the hand moves up, forward and left, and the target of the gripper tip
  must move the same way.
- A recentre of the headset view while the grip is squeezed: the arm must hold, not jump.
- The limiter and clutch values and the hand speed limits under a moving and a turning hand; they are computed on the
  arm's kinematic model, not copied from a tested configuration.
- The view age limit: the ages of the live view measured on the rig before power, and the limit set from the highest
  age seen.
- Live view: freeze the browser, close it and stop the camera sender while a person drives; the arm must hold each time.
- Time from the help request to an accepted takeover, starting with the headset asleep; the assistance wait's first
  value is checked against it.
- On the rig: the first human motion is 3 cm straight up over at least 5 s; then right B during motion, with the time
  to the hold (above 0.25 s the session ends); then the network cable pulled while the grip is squeezed; then the
  return to rest and the hand-back; then demo 2 three times with no program restarted.
- Repeated squeeze and release with a still hand on the loaded arm: each squeeze starts from the measured pose, so an
  arm that sags can creep down.

If the headset check, the buttons, the direction or the STOP delay fails on the simulated arm, human driving stays
switched off for the rig.

## 20. Not in the first build

**Remaining work after the demos.**

| Work | Note |
|---|---|
| One connection through the Teleop Session Server | If wanted later, the service's code moves into the server and the robot link messages move onto the teleop connection. That is the first proposal's Isaac Capture edit (estimate 1,150 to 1,550 lines of source, 1,400 to 2,000 of tests, hardware goals 1, 2, 3 and 5 repeated). Nothing above the three calls changes |
| A second Thor program that survives the first one's death | Every existing motor connect cuts torque, so this needs a new way to attach to a held arm |
| The obstacle grid sent to the Thor, so that the Thor runs today's full check locally | That check reads no camera images, only the grid and the list of occupied cells (together about 0.6 MB for today's workspace box). With it: a backup check on the Thor for when the workstation fails, and moving checks between the machines as workloads move |
| A timed return to rest after the headset or controller is lost while a person drives, or after the robot connection is lost | Both cases hold in the first demos (section 10) |
| Human-only mode under the never-go-limp rule | It does not meet the rule today (section 12) |
| The question "completed or abandoned?" after a takeover (section 12 steps 6 and 7) | Not asked, and the robot app closes no task as "completed by human" or "abandoned": the task record does not say how a takeover ended. The hand-back sets the control owner to nobody and ends the takeover. Storing the answer needs a link message or a file, because the agent program may be down at that moment |
| The takeover steps printed one at a time at the operator console, and whether a return to rest would be accepted (section 12 step 1) | At a help request the console prints the reason, the state of the jaws, the time left with what the arm does when it runs out, the commands that can be typed, and one line that says whether a takeover is possible or why not. No link message says whether a return to rest would be accepted; a refused `rest` gives its reason. The person learns the next step from that one line and from the bells |
| A STOP that passes link messages already handed to the robot connection (section 6) | `core::Link` sends reliable link messages in order and has no priority, and the first build does not edit it. A STOP from the workstation waits behind a MOVE that is being sent: about 5 ms for the longest MOVE on a 1 Gbit/s cable and about 0.25 s at 20 Mbit/s. Until a priority path exists the demos run on a wired connection, and the STOP key at the Thor terminal needs no network |
| APAI's own window and its own page server inside the agent program (section 13) | The agent program starts APAI without them (`--no-view`, `--no-serve`). The page for a run is the viewer's. It lists a run, with its obstacle grid, its refusals and its replay, only when the agent program recorded that run with `--record-debug`; without such a run it shows the live picture and the live arm |
| A restart script for the Teleop Session Server (section 5) | The operator starts CloudXR's service and the Teleop Session Server by hand and starts them again by hand after a fault; after such a restart Headless is switched on again in the headset. A fault of either costs human input until a person acts |
| A hold marker when the robot app ends with the arm raised (section 10) | The robot app writes none. Its next start measures the arm and refuses a raised one, which is the guard; without a marker nothing on the Thor's disk says that an arm is left held by motors that no program drives |
| Joint signs, units and coordinate-system versions in WELCOME (section 6) | WELCOME carries none of them. Angles are in rad in the order of the joint names, the camera calibration is one name, and two Thor settings that the agent program needs, the wait for a first permit and the settle time, are copied by hand into its configuration: two files that can drift apart. The agent's speed limit is copied as well and is also named in WELCOME, and the agent program plans no faster than either (section 6) |

**Optional later.** Tasks from the Thor terminal (the owner has not decided; opinion in section 13); camera picture in
the headset (the owner asked for a feasibility check; not done yet); browser buttons; recording for datasets; fast
policies on the Thor; control with gloves alone.

## 21. The human path in the first build, and what waits for the project owner

On the rig the build's human path runs on the machine where sections 4, 5, 12 and 14 place it: the controller's
data are read and hand motion becomes joint targets on the Thor, and the teleop connection crosses between the
machines. The build departs from those sections in the process that does this work. It is a program of its own
beside the robot app, the human program, where this design has a thread inside the robot app. In the runs on one
machine the same program attaches to the robot link service. Sections 4, 5, 12 and 14, and the places in sections
2, 6, 8, 9, 10, 15, 18 and 19 that follow from them, state the design until the project owner decides. The lead of
the build, the AI agent that directs it, chose the placement below for the first build; it is not a decision of the
project owner. The table "What waits for the project owner" below lists the other choices that wait for the project
owner (rows 2 to 25).

Terms of this section. The human program is the program that the operator starts with `run_human.sh`: it reads the
right controller through a remote session with the Teleop Session Server (a `TeleopSession` in
`SessionMode.REMOTE`) and sends robot link messages to the robot app. The robot app's own socket is a Unix socket on
the Thor that the robot app serves when it is started with `--human-socket`; it speaks the local API of section 5
to one client, the human program. On the rig the human program runs on the Thor and attaches at that socket. The
one-machine recipes are the runs of the example's README in which every program runs on one machine; in them the
human program runs on that machine and attaches to the robot link service. HUMAN_SETPOINT is the robot link message
that carries one joint target from the human program to the robot app. The clutch is engaged while the grip is
squeezed and the human program lets the arm follow the hand. The sample age is how old the controller data behind a
joint target are. The stage of the human program says how far it is with its controller data: `no_server`,
`connecting_openxr`, `waiting_for_headset`, `ready`, `stale` or `no_controller`. The headset path is the way from
the headset through CloudXR and the Teleop Session Server to the human program. The speed-limited target is the
target of the gripper tip after the end-effector rate limiter. The round trip is the time from a ROBOT_STATE leaving
the robot app to the arrival there of the HUMAN_SETPOINT that is computed on it. A tick is one pass of the Teleop
Session Server's loop, and the tick age of SAMPLES is the time since the tick that made them. A pass is one turn of
the human program's loop. An engagement is one stretch in which the clutch is engaged, from a squeeze of the grip to
its end. A human drive, or a person's drive, is the Thor's side of one engagement: one stretch of human motion. A
human target is what the Thor takes from a HUMAN_SETPOINT: the joint target with its sample age and the numbers it
quotes. The robot app's joint target is the last setpoint its drive sent. A step of the robot app is the farthest
its speed clamp lets a joint move in one setpoint period. The gripper tip is low when it is below the table
clearance (section 11) or inside the agent's last grasp region.

**The placement.** This design puts the client of the Teleop Session Server on the Thor, inside the robot app: the
teleop thread of the robot app dials the server, receives SAMPLES and turns them into joint targets there. The first
proposal does the same, and it names a move of this code to the workstation as optional later work that follows
measured compute and latency needs.

On the rig the build puts that client on the Thor as well, in a process of its own. The human program dials the
Teleop Session Server on the workstation from the Thor, at the address its file for the rig names
(`teleop.endpoint` in `configs/human_rebot.yaml`, which ships with a placeholder for the workstation's address), so
the teleop connection crosses between the machines as in this design. It reads the right controller from SAMPLES,
runs Isaac Capture's controller source, clutch and end-effector rate limiter and APAI's inverse kinematics, and
hands each joint target to the robot app in a HUMAN_SETPOINT link message at the robot app's own socket
(`socket_path` in that file, and `--human-socket` of the robot app). What the human program sends on the rig, right
B's STOP included, therefore passes neither the robot link service nor the robot connection. The robot app takes
the joint target from that link message, checks its owner number and stop number, the sample age, the live view
where `human.require_live_view` is on, the size of the step and the speed, runs the human checker and the contact
monitor, and holds whenever a person's motion ends. The robot app has no teleop thread, and no retargeting and no
inverse kinematics run in its process: its socket is served on threads that only hand link messages over
(`robot_app/human_socket.py`). The teleop code stays unedited, and the robot connection stays separate from the
Teleop Session Server.

In the one-machine recipes and in the tests the same program attaches to the robot link service
(`configs/human.yaml`: an empty `socket_path`, which stands for the service's socket, and the Teleop Session Server
at `127.0.0.1:47600`). What it sends then crosses the robot connection, and the robot app handles it by the same
handlers as what arrives at its own socket.

```text
            WORKSTATION                                           THOR (on the robot)
  agent program (APAI) --+
  operator console ------+-- local API --> robot link     robot connection
  viewer ----------------+                 service  <--------------------------->  robot app
                                                                                   (motor driver, checker,
                                                                                   hold, STOP, control owner,
                                                                                   assistance wait, return to
                                                                                   rest, human motion)
                                                                                     ^
                                                                                     |  the robot app's own
                                                                                     |  socket (local API)
  headset -> CloudXR -> Teleop Session     teleop connection (SAMPLES)               |
                        Server  ---------------------------------------------->  human program
                                                                                   (controller source, clutch,
                                                                                   rate limiter, inverse
                                                                                   kinematics, buttons)
  camera receiver  <---------------------- camera connection --------------------  camera sender
```

The diagram shows the rig. In the one-machine recipes the human program attaches to the local API of the robot link
service, beside the operator console and the viewer, and every connection stays on that machine.

The case for a process of its own: the process that holds the motors gets no `TeleopSession`, no retargeting and no
inverse kinematics loop beside its 50 Hz motor loop, so when that code fails or stalls it is the human program that
ends or stalls, and the robot app holds the arm once the sample age has run out; one program serves the rig, the
one-machine recipes and the tests; and the Thor's own checks (sample age, live view, step limit, speed clamp, human
checker, contact monitor) are the same in both. What it costs:

1. A second program runs on the Thor beside the robot app, where the project owner decided on one program on the
   Thor for now (section 3). It is started after the robot app, which serves its socket, and it ends when the robot
   app ends. Without it there is no right B and no takeover.
2. The robot app sees neither the buttons nor the grip. The 1 s hold of right B, the released grip during it and the
   clutch are the human program's word. The robot app checks the ages and the numbers a request quotes, and it
   follows the hand only after a HUMAN_SETPOINT that quotes the current owner number has reported the grip released.
3. The sample age starts from an age that the human program reports. The human program also says when it sent the
   HUMAN_SETPOINT, on the Thor's monotonic clock as it reads that clock off its newest ROBOT_STATE (`sent_mono_s`),
   and the robot app adds the time since then on its own clock. The round trip is shown and nothing is enforced on
   it.
4. A link message from the robot app's own socket, right B's STOP among them, waits for the next pass of the robot
   app's dispatcher: at most one of its step periods, 10 ms.
5. The hand speed limit is a setting of the human program, not of the Thor's human limit set.
6. One more stage at 50 passes per second and one passage between two programs on the Thor, from the human program
   to the robot app. The delay they add is not measured.

The case for this design's thread: both proposals say so; the robot app measures the age of the controller data
itself; the robot app sees the buttons itself, so right B's STOP and the 1 s hold are not another program's word;
and there is one program fewer to start on the Thor. What it costs: a second thread, a second connection,
`TeleopSession`, the retargeting and the inverse kinematics inside the process that holds the motors, and the bench
test of the teleop thread beside the 50 Hz motor loop (section 19) has to pass before any human motion.

Moving into the robot app later: the robot app's side takes a joint target with its sample age and the numbers it
quotes (`RobotApp.offer_human_target`), and assumes nothing about where the target is computed. The move is one
thread in the robot app that runs the same code as the human program and hands its joint targets to the robot app
directly, plus a local entry for the takeover and one for the hand-back: an estimated 150 to 250 lines for the
thread and about 15 lines for each entry. No link message changes, and HUMAN_SETPOINT can stay for the synthetic
controller.

**What follows from that placement.**

| Topic | This design | The build |
|---|---|---|
| Connections between the machines (sections 3, 5 and 15) | The teleop connection and the robot connection | On the rig the same two: the teleop connection, between the human program on the Thor and the Teleop Session Server on the workstation, and the robot connection. A person's joint targets cross neither as robot link messages: they are computed on the Thor and handed over at the robot app's own socket. In the one-machine recipes both connections stay on that machine and HUMAN_SETPOINT travels on the robot connection. Both are separate `core::Link` connections |
| Connections the robot app watches (sections 5 and 15) | Two, each with its own "alive" state | One connection between the machines, the robot connection, and on the rig its own socket for the human program. The human program watches the teleop connection, and the robot app learns that the headset path is lost from the stage and the sample age in HUMAN_SETPOINT. ROBOT_STATE says whether a human program is attached at the robot app's own socket (`health.human.local_program`) |
| Clients of the robot link service (section 5) | The agent program, the operator console, the console command and the viewer | One more role, the human program (`human`), one at a time. The service forwards from it HUMAN_SETPOINT, the OWNER_REQUESTs for a takeover and for a hand-back, and STOP, and nothing else. The one-machine recipes and the tests use that role. On the rig the human program is no client of the workstation's service: the robot app's own socket speaks the same local API, takes the same link messages from the role `human` alone, and refuses every other role and a second human program |
| A link message for a person's joint targets (section 6) | None: the joint targets are computed on the Thor | HUMAN_SETPOINT, human program to robot app, newest wins. It carries the session id and the owner number and stop number the human program last saw; a sequence number; the source of the controller data (the Teleop Session Server or a script); the human program's stage; the sample age as the human program sends it; whether the clutch is engaged; whether the grip is released; the joint target, only while engaged; the jaw opening the trigger asks for, or none where the trigger is not usable (`jaw_opening`; the robot app refuses a value outside 0 to 1); the clock value of the ROBOT_STATE it is computed on; when it was sent, on the Thor's monotonic clock as the human program reads that clock off its newest ROBOT_STATE (`sent_mono_s`); and a note for the operator of at most 100 characters. No RESULT answers it. The human program sends one 50 times per second while a person owns the arm and 20 times per second otherwise, as a status without a joint target. The service that serves the human program's socket, the robot link service or the one inside the robot app, drops it from a STOP until a ROBOT_STATE shows the higher stop number. The Thor follows a joint target only when it quotes the current numbers, is engaged, comes with the stage `ready` and a sample age of at most 0.25 s (`human.sample_age_limit_s`) and, on the real arm, has the Teleop Session Server as its source |
| How a takeover and the hand-back are requested (sections 6 and 12) | By no link message: the teleop thread reads right B and right A on the Thor | By OWNER_REQUEST. The human program sends the request "human" when right B has been held for 1 s with the grip released, and "handback" when right A has been held for 1 s with the grip released while a takeover is open. The check is split: the human program decides on the buttons and the fresh SAMPLES and sends the request once a ROBOT_STATE shows the STOP of that press; the Thor checks the numbers the request quotes, the sample age, the live view, the joint read and the fault latch |
| Human input in ROBOT_STATE (section 6) | Teleop connection up, the stage the Teleop Session Server last reported, READY seen, the age of the newest SAMPLES | The human program's stage (`no_server`, `connecting_openxr`, `waiting_for_headset`, `ready`, `stale` or `no_controller`; empty once the program has been silent for 4 s, `human.stage_keep_s`) and the sample age of the newest HUMAN_SETPOINT; "connection up" and "READY seen" are part of the stage. Also the source of the controller data, the human program's note, the round trip and, from a robot app that serves its own socket, whether a human program is attached there. While a takeover is open and `human.trigger_jaws` is on, also the state of the jaws under the trigger (`health.human.jaws`): the grip level, whether the trigger has reached it, whether the jaws were measured fully open since the takeover, the jaw opening the newest HUMAN_SETPOINT asked for, and why the trigger moves nothing |
| The age of the controller data (section 8) | Measured by the Thor on its own clock, network included | The age the human program reports as it sends the HUMAN_SETPOINT plus the time since it was sent, which the robot app counts on its own monotonic clock from the send time the link message names (`sent_mono_s`). The way to the robot app and the wait there are so part of the age: on the rig that way is the robot app's own socket, in the one-machine recipes the robot connection. A HUMAN_SETPOINT that names no send time is counted from the ROBOT_STATE it is computed on, and one that names neither has no known sample age. The human checker's watchdog is fed with the time so computed. The round trip (the Thor's clock at the arrival of a HUMAN_SETPOINT minus the clock value of the ROBOT_STATE it is computed on) is shown and not enforced; a limit on it needs a bench value first |
| The hand speed limit (section 8) | Part of the Thor's human limit set | A setting of the human program (`retarget.max_linear_velocity`, `retarget.max_angular_velocity`). The Thor's own bounds on a person's motion are the joint speed clamp, the step limit, the human checker and the contact monitor, applied to every HUMAN_SETPOINT whoever computes it. The hand speed limit bounds the target of the gripper tip only; the pace of the arm is set by the joint speed limit of the human limit set, which the human program reads from WELCOME (section 6). The human program also ends an engagement by itself, with a HUMAN_SETPOINT that says not engaged; the paragraph "A person driving the arm" below this table lists when |
| Right B as STOP (sections 9 and 15) | Read on the Thor from SAMPLES | Read by the human program from SAMPLES. On the rig its STOP goes to the robot app at the robot app's own socket: it passes neither the robot link service nor the robot connection, the robot app takes it with the robot connection down as well, and it waits at most one step period of the robot app's dispatcher (10 ms). Right B therefore stops the arm only while the human program runs and the teleop connection brings it SAMPLES with the buttons, besides the conditions of section 9. With the teleop connection down, as when the cable between the machines is pulled, the human program reads no button; a person's drive then ends in a hold for want of fresh controller data. In the one-machine recipes the STOP travels through the robot link service and the robot connection. A press shorter than one pass of the human program (20 ms) can be missed, as in this design. When the human program ends while a person owns the arm, it sends a STOP first: the arm holds and nobody owns it. The human program ends when the program that serves its socket goes away, which on the rig is the robot app, and right B then stops nothing until the human program is started again. At the robot link service a STOP sent between a new robot connection and its first ROBOT_STATE keeps HUMAN_SETPOINT at the service until a later STOP is confirmed, and the person presses right B again; at the robot app's own socket the robot app's first ROBOT_STATE ends that hold. `stop` or Ctrl+C at the Thor terminal needs none of these programs |
| Dialing the Teleop Session Server (section 5) | The teleop thread dials when the robot app starts, with the lower-level `RemoteSession` binding in a retry loop, without a time limit | The human program dials: on the rig from the Thor to the workstation's address, in the one-machine recipes to `127.0.0.1` (`teleop.endpoint`). Each attempt enters a `TeleopSession` in `SessionMode.REMOTE` with a time limit of 3 s and is repeated 5 s after a failure or a lost teleop connection; the `RemoteSession` binding is not used. An attempt holds up the human program's loop for up to 3 s, in which it sends nothing; an attempt on a thread of its own is not built. After READY the program steps the session 50 times per second, asks the server for a tick rate of 90 Hz and treats SAMPLES with a tick age above 0.15 s as stale. The attempts are counted (row 12 of the table below) |
| How the console learns that a takeover is not possible (section 5) | At once when the server process is gone, after about 1 s when the controller is lost, after about 3 s when the teleop connection goes silent | From three places: ROBOT_STATE, in which the Thor names why a takeover would be refused and whether a human program is attached at the robot app's own socket; the robot link service, which says whether a human program is attached to it; and the loss of the robot connection. The console prints `takeover: possible now: hold right B for 1 s with the grip released`, or `takeover: not possible: ` and the reason, for example `no human program is attached: start run_human.sh` where none is attached at either place. Its status line shows `human program at the robot app` for one at the robot app's own socket, `no human program` where there is none, and the human program's stage, the sample age and the human program's note, which tells the person the next step. The delays that this design gives are not measured for the build |
| Faults of the headset path (sections 10 and 15) | The teleop thread treats its own error as controller signal lost | An error in one pass of the human program sends nothing for that pass, and right B stays a STOP. If the errors go on, the Thor ends the human motion in a hold once the sample age is above 0.25 s, and the person keeps control. When the Teleop Session Server, CloudXR or the human program is down, the agent, the console and the assistance wait go on. No end of the human program leads to a return to rest or cuts torque. A lost robot connection ends a person's drive in a hold, and the person keeps the arm. On the rig the human program stays attached at the robot app's own socket meanwhile and the robot app takes right B's STOP; a new squeeze moves nothing where `human.require_live_view` is on, as in the rig's file, because VIEW_STATUS travels on the robot connection. In the one-machine recipes the joint targets travel on the robot connection, so a lost robot connection ends human input as well |
| Agent before any headset (sections 5 and 15) | By the teleop thread, which waits in the background | The robot app waits for no headset. The agent runs with no human program and no Teleop Session Server started |
| Start-up (section 14) | The Thor dials the robot link service and the Teleop Session Server | The robot app dials the robot link service only. For demo 2 on the rig the operator starts CloudXR's service and the Teleop Session Server on the workstation, the server listening on the address the Thor reaches. On the Thor the operator starts the robot app with `--human-socket` and then the human program with `configs/human_rebot.yaml`: the robot app first, because it serves the socket. The human program can be started before the server and before any headset is connected. In the one-machine recipes the human program is started after the robot link service, which serves its socket there |
| Bench tests (sections 15 and 19) | The teleop thread beside the 50 Hz motor loop; controller samples that keep reaching the Thor | The first of these does not apply in the build, because the robot app has no teleop thread. In its place, on a simulated arm on the Thor before power: the human program started before any headset is connected, and READY reached without restarting it; the round trip through the robot app's own socket, which must stay under 0.1 s, and the tick age of SAMPLES that crossed between the machines; a typed `stop` and right B while 50 HUMAN_SETPOINT link messages per second arrive at the robot app's own socket, each within 0.25 s; right B with the robot link service ended, which must stop the arm all the same; and the Teleop Session Server stopped and started again, and the cable between the machines pulled, while the grip is squeezed. With Headless off the check reads whether controller samples keep reaching the human program |
| Size (sections 16 and 18) | The teleop thread, half an hour in section 18; one new workstation program, the robot link service | One more program, the human program, on the Thor on the rig: as built about 4,250 lines of source (`human/`). The robot app's side of a person driving holds, as built, about 700 lines in `robot_app/human.py` and about 140 in `robot_app/human_socket.py`, beside what `robot_app/app.py` and `robot_app/gripper.py` hold of the takeover and of the jaws under the trigger |
| A later move to one connection (section 20) | The robot link messages move onto the teleop connection | The Teleop Session Server serves one client, and in the build that client is the human program, on the rig a process on the Thor beside the robot app. The robot link messages can move onto the teleop connection only once the robot app is that client, so the move into the robot app comes first |

**A person driving the arm.** The robot app's checks of a human target are the authority. Before every setpoint it
checks the numbers the target quotes, the sample age, the live view where `human.require_live_view` is on, and the
step limit; it clamps the joint speed; and it runs the human checker and the contact monitor. A target it refuses ends
the drive in a hold. What follows is what the human program does in front of those checks (`human/logic.py`), so that
the arm stops at a limit and then follows the hand again without a release of the grip. It reads the limits from
WELCOME (section 6). Under a robot app whose WELCOME names none of them it keeps none, and the Thor's refusals end the
drive.

- The pace. The human program moves the joint target it sends at 0.9 of the Thor's speed clamp (`JOINT_SPEED_SHARE`)
  on the joint that has the farthest to go, and every other joint by the same share of its way. A hand that is ahead
  of the arm is caught up with: the arm goes after it at that pace and stops where the hand stopped.
- The workspace box and the table. The human program keeps the gripper tip of the pose it solves for 0.01 m
  (`TIP_BOX_MARGIN_M`) inside the walls of the workspace box and 0.01 m above the height under which the Thor counts
  the tip as low, which is the table plane plus the table clearance (`human_tip_floor`, section 6). The tip stops at
  a wall and slides along it while the hand goes on, and comes away from it with the hand. Each axis is held by
  itself. A tip that is outside the box at the squeeze may come toward the box and go no further out, and a tip under
  that height may only rise: the human checker's own rule.
- The human joint box. The joint target stops 0.01 rad (`JOINT_BOX_MARGIN_RAD`) inside the human joint box, and the
  engagement goes on.
- Out of reach. While the inverse kinematics has no solution for the hand's pose, the last joint target stands, so
  the arm stands, and the engagement goes on. The arm follows again, with no new squeeze, as soon as a step toward
  the hand solves inside the joint box.
- The hold-back. The joint target stays within two of the robot app's steps (`HOLD_BACK_STEPS`) of where the robot
  app's joint target can be: its last one in the newest ROBOT_STATE, moved on by what the robot app's loop can have
  done since, at the pace its last two ROBOT_STATEs show. The ROBOT_STATE behind every setpoint (section 6) is what
  this reckoning reads. Nothing is added once the next ROBOT_STATE is overdue, because the robot app may stand. A
  robot app that runs slow or pauses then costs speed and not the engagement. A joint target that is held back for
  1 s with no new joint target of the robot app ends the engagement (`NO_FOLLOW_S`).
- A squeeze after a release. A squeeze counts only once ROBOT_STATE has shown two things. The first is that the
  drive of the engagement before it has ended: a ROBOT_STATE that shows no human drive and was made after the robot
  app took the HUMAN_SETPOINT that ended that engagement. This wait has no time limit. The second is that the arm is
  at rest: for every joint, two readings in a row that both began after the robot app's last setpoint and agree
  within 0.002 rad (`REST_TOL_RAD`). A ROBOT_STATE made while the arm still moved gives a pose the arm has left. The
  wait for the arm at rest is at most 0.5 s from the end of that engagement (`REST_WAIT_S`); after it the squeeze
  counts on the newest ROBOT_STATE, and the Thor's step limit is the backstop. A squeeze that comes sooner is kept
  and counts from then.
- A low gripper. Where the gripper tip is low a takeover is accepted, and the Thor follows only a lift. It keeps the
  highest tip height it measured in a human drive since the takeover, or since the tip was last not low, and ends
  the drive on a target or a step that would put the tip more than 1 mm under that height (`TIP_TOL_M`). Where the
  arm has settled more than 1 mm under it, the Thor's drive waits and sends nothing until the target is back within
  1 mm of it. The human program lifts such an arm only on the hand's own rising way. Once ROBOT_STATE has shown for
  0.15 s (`LIFT_AFTER_S`) a drive that has this engagement's joint target and has sent no setpoint, the joint target
  follows the hand and stays within `ik.max_ahead_rad` less 0.005 rad (`LIFT_SLACK_RAD`) of the measured arm. Under
  the height that WELCOME names in `human_tip_floor` the joint target moves only where every step of the robot app
  lifts the tip, and while the drive waits only where the robot app's first step also keeps the tip inside the hand
  speed limit. A hand that rises lifts the gripper out with that squeeze. While the drive waits, a hand that stands
  still, goes sideways or sinks moves nothing, and the engagement ends 1 s after the joint target's tip last rose
  (`FOLLOW_WAIT_S`). An arm that has settled further under its highest point than the robot app reaches with steps
  that all rise is not lifted by a squeeze: the person presses right B, takes the arm over again, which starts the
  height at the measured tip, and then squeezes and lifts.
- What ends an engagement. A wall of the workspace box, the table clearance, the human joint box and the end of the
  arm's reach end none: the arm stops there and follows again. Each of the following ends the engagement in a hold,
  and the grip is released and squeezed again. In the human program: a hand more than `ik.max_ahead_m` from the
  gripper tip (0.5 m in both shipped files) or turned more than 2.0 rad ahead of it (`MAX_AHEAD_TURN_RAD`); a hand
  faster than `retarget.reject_linear_velocity` or `retarget.reject_angular_velocity` (2.0 m/s and 10 rad/s); a jump
  of the inverse kinematics above `ik.max_jump_rad` (0.05 rad) that halving the tip's step three times
  (`STEP_HALVINGS`) does not remove; a squeeze that no ROBOT_STATE shows the robot app's drive following within 1 s
  (`FOLLOW_WAIT_S`), which is also the end for a low gripper whose hand does not rise; a joint target that is held
  back for 1 s (`NO_FOLLOW_S`); a joint solution more than `ik.max_ahead_rad` ahead of where the arm can be, as when
  the Thor's hold at a contact puts the robot app's joint target back to the measured pose; controller data that are
  lost or stale; and a newest ROBOT_STATE older than `state.max_age_s` (0.2 s). The Thor's own refusals end the
  drive as well: controller data lost or stale (`human.sample_age_limit_s`), a live view that is not fresh, a lost
  robot connection, a target that lowers the tip inside the agent's last grasp region, the human checker's refusals
  other than the three limits above (the wrist camera would face down, a link at the table), a contact, a setpoint
  more than `executor.large_delay_s` late (0.1 s) and a target beyond the step limit (`limits.human.max_step_rad`).
- Releasing the grip, and right B. Each stops the arm at once: no joint target follows the pass that reads the
  released grip, and the arm holds at the last setpoint the Thor sent. Right B is a STOP. Stopping the hand does not
  stop the arm at once: the arm stops where the hand is.

**What waits for the project owner.** Each row gives the position the build ships in, what that costs, and the
strongest alternative with its cost. Nothing here needs an answer for the build to run: each choice with a setting can
be changed in a shipped file. Row 1 is the first to decide. Rows 23 and 25 concern a session on the rig without a
headset, and rows 2, 3, 5, 6, 10 and 11 are needed before the first session with the headset. The open points of
section 16 that the build leaves as they are (the robot link service as a separate program, and permits) are not
repeated here.

| # | Choice | Ships as | What that costs | Alternative and its cost |
|---|---|---|---|---|
| 1 | The process in which the controller is read and hand motion becomes joint targets | In the human program, a process of its own: on the rig on the Thor beside the robot app, in the one-machine recipes on that machine (the placement above). No setting switches between the two: the human program runs on the machine it is started on, and `socket_path` in its file names the socket it attaches to | The six costs listed above | This design's thread inside the robot app, with the costs listed above |
| 2 | Human driving on the rig | The Thor setting `human.enabled` is off in the rig's file (`configs/robot_app_rebot.yaml`) and in the file for the simulated arm (`configs/robot_app.yaml`). It is on in the three other shipped files of the robot app: the file for APAI's mock arm (`configs/robot_app_mock.yaml`), the file for demo 2 on the simulated arm on one machine (`configs/robot_app_demo2.yaml`) and the file for the headset checks on a simulated arm (`configs/robot_app_human_check.yaml`). While it is off a takeover is refused with `human driving is switched off in this robot app's file (human.enabled)`, and no joint target from a person moves the arm. The owner switches it on for the session, after the headset checks of section 19 have passed on a simulated arm | No demo 2 on the rig until then | Ship it on: code that has never seen a headset is then reachable on the real arm in the first run of demo 1 |
| 3 | The trigger and the jaws under a person (section 12 step 5) | Built, and switched on: the Thor setting `human.trigger_jaws` is on in every shipped file of the robot app, with `human.jaw_match_tol` at 0.05. It acts only where `human.enabled` is on. The trigger limit still awaits the owner's yes (section 16) | A jaw rule that no real controller's trigger has run on real jaws can be reached in the first session with the headset on the arm | Ship it off (`human.trigger_jaws: false`): a person can then neither grasp nor let go with the trigger, and letting go is `stop`, `open`, `yes` at the operator console |
| 4 | A silent agent after a hold (section 16) | Not built: the arm holds until someone acts | No prompt when an agent program goes silent with its connection open | Ten seconds of silence count as failed and start the assistance wait: the arm can then move to rest by itself after an agent that is only slow |
| 5 | Typed takeover and hand-back | The operator console has the commands `takeover` and `handback`, each sent after a typed `yes`, as a fallback for right B and right A. The robot link service forwards them only with its setting `operator.typed_takeover` on, which is off in both shipped files and is read when the service starts. The Thor makes the same checks as for the buttons | If right A or right B does not arrive, or the controller fails during a takeover, the takeover ends only after the setting is switched on and the robot link service is started again, or by a shutdown | Ship it on: a takeover then needs no 1 s button hold, and a hand-back can be typed, where the owner decided that the arm is given back with right A (section 3). A middle way allows the typed `handback` always, which moves nothing, and keeps only the typed `takeover` behind the setting |
| 6 | The speed of a person's motion on the rig | The rig's file has 0.8 rad/s for the joint speed limit of the human limit set (`limits.human.max_joint_speed` in `configs/robot_app_rebot.yaml`), which is the speed cap of the rig's arm profile; a faster human limit set is refused. The Thor clamps a person's motion at 0.98 of it, and the human program moves its joint target at 0.9 of that clamp. The hand speed limit in the human program's file for the rig, `configs/human_rebot.yaml`, is 0.25 m/s and 1.0 rad/s (`retarget.max_linear_velocity`, `retarget.max_angular_velocity`); it bounds only the target of the gripper tip. The first values of section 8 are 1.5 rad/s, 0.10 m/s and 0.4 rad/s. The file marks the 0.8 rad/s as the owner's value | The arm follows a hand at up to 0.8 rad/s on each joint, a speed at which no person's motion has run on the arm | A lower `limits.human.max_joint_speed` for the first session with the headset: the human program follows the limit that WELCOME names, so this one key on the Thor slows the arm. It costs a slower arm under a person |
| 7 | A hand that runs ahead of the arm | The arm catches up: it goes after the hand at the pace of the joints and stops where the hand stopped (the paragraph "A person driving the arm" above). For being ahead, the human program ends the engagement only when the hand is more than 0.5 m from the gripper tip (`ik.max_ahead_m` in both of its shipped files) or turned more than 2.0 rad ahead of it (`MAX_AHEAD_TURN_RAD` in `human/logic.py`), or when the joint solution is more than `ik.max_ahead_rad` ahead of where the arm can be (0.045 rad in the rig's file, 0.08 rad in the file for one machine). The hold-back keeps the joint target inside that, so the last of these ends an engagement only where the robot app's joint target moves away, as in a hold at a contact. Section 8 limits the hand speed and clamps instead | After the hand has stopped the arm travels on by as far as the hand was ahead, and it goes straight to where the hand is, not along the way the hand took | A small limit on how far the hand may be ahead, which ends the engagement there: each end costs a release and a new squeeze |
| 8 | A joint target far from the arm | Refused: a joint target more than `limits.human.max_step_rad` (0.05 rad in the rig's file, 0.1 rad in the files for one machine), on any joint, from the arm at a squeeze or from the last setpoint sent after it ends the human motion in a hold, and the person squeezes again. The joint speed is still clamped. The human program's hold-back keeps its joint target inside this limit while ROBOT_STATEs arrive. Section 8 clamps the step | A release and a new squeeze each time it happens | Clamp, as section 8 says: the arm can creep toward a pose nobody asked for after the inverse kinematics has changed its solution |
| 9 | The live view rule has a switch | The Thor setting `human.require_live_view` is on in the rig's file, in the file for the simulated arm (`configs/robot_app.yaml`) and in the file for the headset checks on a simulated arm (`configs/robot_app_human_check.yaml`). It is off in the two files for demo 2 on one machine, which runs without a camera (`configs/robot_app_demo2.yaml` and `configs/robot_app_mock.yaml`, the file for APAI's mock arm). The robot app accepts it off with any declared driver, the real arm included. A robot app that starts with human driving switched on and this setting off prints `human driving is not tied to a live view (human.require_live_view false)` at the Thor terminal. With it off a takeover and a person's drive need no VIEW_STATUS | On one machine a person drives a test arm without a live view. Only the value in the rig's file keeps the owner's rule of section 3 on the rig | Accept it off only on a test arm: not built. Or require the live view always: no demo 2 on one machine without a camera |
| 10 | The oldest live view a person may drive on | `human.view_age_limit_s` is 0.5 s, a first value. The viewer's page draws up to 30 live pictures a second and the viewer sends a VIEW_STATUS every 0.2 s, so the age the Thor computes swings by about 0.2 s before any camera delay; where it passes the limit a takeover is refused, or a person's motion ends in a hold. Not measured on the rig | Refused takeovers and holds until the limit is chosen | A higher first value such as 1.0 s: fewer refused takeovers, and a person may drive on an older picture. The owner chooses the limit at the headset check of section 19: the highest age measured plus 0.2 s, at most 1.0 s |
| 11 | Who watches the live view | The checklist of the first session lets the wearer drive while seeing the arm itself; a second person at the workstation watches the live view, and the Thor's check of the view's freshness stands for the owner's rule of section 3 | The person who drives may not see the live view | The wearer must see the live view themselves: no human motion until the headset check finds a setting in which they can |
| 12 | The wait for the Teleop Session Server has an end | The human program ends after `teleop.attempts` failed attempts in a row to reach the server and a tracked headset: 720 in the file for one machine (about an hour and a half) and 8640 in the rig's file (a session day), 5 s apart; a READY sets the count back to zero. Section 5 waits without a time limit | After the program has ended there is no right B and no takeover until the operator starts it again; the console says `no human program is attached: start run_human.sh` | No limit: a program that dials every few seconds for as long as it runs |
| 13 | A takeover where the human checker refuses the measured pose | Refused with `the arm is outside the human limits: `, the reason and `; type rest`, for every complaint of the human checker but a low gripper tip. Section 12 has no such condition. Where the agent stops with the gripper tip low, below the table clearance (0.02 m in the rig's file, `rest.table_clearance_m`) or inside its last grasp region, the takeover is accepted and the Thor follows only a lift (the paragraph "A person driving the arm" above). A typed `rest` and an agent task stay refused there until the tip is clear: the person lifts the gripper out, releases the grip and types `rest` | No takeover where the arm stands outside the human joint box or the workspace box, or with a link at the table. A low gripper is lifted out by a person only: with the headset path lost, what is left is the 48 V switch with the arm held by hand | Refuse a takeover with a low gripper tip as well: an arm that the agent left low in a grasp then takes no command that moves it. Or accept every takeover: a person then owns an arm that the human checker lets nobody move |
| 14 | The human joint box on the rig | With human driving on, the robot app refuses a Thor configuration whose human joint box reaches outside the recovery region, and the rig's human joint box lies 0.02 rad inside that region, which is also the agent's envelope (`limits.human.joint_min`, `limits.human.joint_max`), on every bound but the lower bound of the second and the third joint, which lie on their stops at rest. The human program stops its joint target a further 0.01 rad inside the box. A joint driven to that edge can read a few milliradians further out and still lies inside the region, where a typed `rest` is taken | A person cannot turn the first and the sixth joint, or raise the second and the third, beyond 120 degrees less that margin. In the outer 0.02 rad of the agent's envelope a takeover is refused, and the operator types `rest` there | The full joint box of the arm's kinematic model for the person: a person can then leave the arm where a typed `rest` is refused, and with the headset path lost only the 48 V switch is left |
| 15 | A typed `rest` from a contact hold while a takeover is open | Accepted when it is typed with `path is clear`. That confirmation also clears the contact spots the recovery checker holds, so that the agent's last contact spot does not refuse the path: only the agent can clear them, and the agent is refused while a takeover is open. The agent checker keeps its spots, and the return keeps its contact monitor and ends at the next contact. It has no setting of its own; it can be reached only where `human.enabled` is on. A shutdown that is confirmed with `path is clear` while a takeover is open takes the same exception (row 17). Section 11 refuses a return to rest while the arm holds at a contact | A return to rest can start at a contact, and pass a recorded contact spot, on a person's typed word | Section 11's rule: after a contact under a person with the headset path lost, the only way out is the 48 V switch with the arm held by hand |
| 16 | A typed `rest` that is refused for its path | The legs of a return are straight joint-space moves (section 11): with APAI the leg up to the ready pose and then the fold of APAI's rest tool, on the simulated arm one move to the rest pose. From a pose a person has reached, such a path can pass under the table clearance, leave the workspace box or bring a link to the table. The recovery checker checks every leg before the control owner changes, and then refuses the return with `the path to rest was refused: setpoint N: ` and the reason, before anything moves. The person squeezes the grip, brings the gripper back the way it came and types `rest` again. No setting | With no working controller no typed command moves the arm then: it holds with its motors on | The Thor plans a route around what refuses the path. The route of APAI's rest tool goes around nothing on the Thor, which has no obstacle grid; section 11 plans no other, and none is built |
| 17 | The shutdown command during a takeover | A confirmed shutdown is accepted while a person owns the arm with the grip released, and while a takeover is open with nobody owning the arm: the arm returns to rest, torque is released there and the robot app ends. The next start of the robot app has no takeover open | A takeover can end without right A, where the owner decided on right A at verified rest (section 3) | Refuse the shutdown until the hand-back: the robot app then cannot be shut down when the controller fails during a takeover |
| 18 | The question "completed or abandoned?" (section 12 step 6) | Not built (section 20) | The task record does not say how a takeover ended | Build it in the console and the task record: it needs a link message or a file for the answer |
| 19 | No obstacle check for a person (section 8, kept) | No setting | The person's eyes are the only obstacle check | A warning on the console when the hand's target is inside the obstacle grid: the human program would need the grid, which lives in the agent program |
| 20 | The permit check has a switch (section 7) | `permits.required_for_agent` is on in every shipped file. The robot app accepts it off on any arm, the real arm included | Only the value in the rig's file keeps the permit check on for the rig | Accept it off only on a test arm: not built |
| 21 | The timed return to rest (sections 10 and 11) | Built, tested on the simulated arm and on APAI's mock arm, and switched off for the rig: `rest.auto_return` is off in the rig's file, and `task.timed_return` is off in the agent program's shipped files. The owner's decision to implement the automatic return (section 3) is met in code | With the shipped files the arm keeps holding when the assistance wait runs out | Switch both on for the rig: the arm then moves by itself before the bench tests of section 19 have been run on it |
| 22 | The typed `reference` at the Thor terminal | APAI's driver drops the jaw reference after three failed jaw reads in a row. Without the reference the jaws cannot be proven empty by measurement, so a return to rest and a release are refused until the operator types `jaws are empty` after the refusal (section 11). With the Thor setting `jaws.allow_reference` on, a person at the Thor terminal types `reference` and then `jaws are empty`, and the jaws close slowly to their stop to find the reference again; the command is accepted only with control owner nobody, no fault-stop, the motors on and nothing else running, and a STOP ends it. The setting is off in every shipped file, because the command moves the jaws on a person's word, which section 11 excludes | After such a read fault with the arm raised, only the operator's typed `jaws are empty` meets the jaw condition of a return to rest; without that word it ends at the 48 V switch with the arm held by hand | Switch it on: a jaw rule that this design does not have |
| 23 | The model client on the workstation | The full stack reaches its cloud language model through a command-line program, the model client, and that program needs a login. The script that builds the workstation's Python environment (`setup_workstation_env.sh`) installs it only with its option `--with-model-client`, and no script of the example reads or writes the login. The agent program's setting `experiment.codex_bin` names the program; the rig's agent file (`configs/agent_rebot.yaml`) names the one that option installs. Installing it and using a login wait for the owner | Without the model client no task runs on the full stack: demo 1 on the rig needs it, the runs on one machine use the minimal stack, and demo 2 on the rig then runs without the agent failing first. The example's README names one run on the rig with the full stack, of 2026-11-17 (the status paragraph at the top of this file) | Install it and use the login that is on the workstation: the model calls then run on that login |
| 24 | The obstacle grid stand-in for one machine | The agent program's setting `experiment.grid` is `bridge` (the obstacle grid that the experiment's code builds from camera frames) in its two main files and `clear` only in its file for APAI's mock arm. With `clear` the obstacle grid is a stand-in with nothing near: every setpoint still goes out under a permit, and no move is checked against obstacles. The agent program accepts it only with the minimal stack and against a robot app whose declared driver is a test arm | The demo by hand on one machine checks nothing against obstacles, and says so in a warning | No such setting: no pick and place by hand on one machine without a GPU |
| 25 | The order of the bench tests (section 19) | The checklist of the first session moves the arm, slowly and supported, before five tests that section 19 puts before any motion | First moves on motors whose behaviour after a program death is not measured, on first values that nobody has measured | Section 19's order: one more session before any motion |

**What is not tested without a headset.** The tests of the human path cover the build's placement only, on one
machine: the human program attached to the robot link service, and attached at the robot app's own socket
(`test_human_on_thor.py`). Every takeover and every motion of a person in them comes from a synthetic controller
inside the human program or from HUMAN_SETPOINT link messages that a test makes. No controller sample has passed
through a Teleop Session Server into this code, because Isaac Capture has no way to send controller SAMPLES without
a headset; the human program's entry into the remote session, READY, staleness and re-entry are tested against a
stand-in for the server that carries no controller data. The first real controller data reach this code in the
session with the headset. Also never run: right A and right B over a remote session; a recentre of the headset
view; the typed `takeover` and `handback` at a real terminal; the viewer's page in a browser window with a person
in front of it; demo 2 on one machine by hand; the human program on a Thor, and its remote session between two
machines; the trigger of a real controller on real jaws; and human driving on the rig. Not measured: the round trip
through the robot app's own socket on a Thor, and the delay the human program adds.

## 22. Evidence

Based on reading code at Isaac Capture commit `4a2b98420` (on the remote teleop session branch), the APAI checkout
pinned at `3c8b773`, the uncommitted Thor adapter files of the experiment checkout, the LeRobot checkout used for
today's remote teleop, the local checkout of Isaac Capture pull request #1410, and the hardware test record. The
example is built through P3. Its tests run on the workstation alone, and the example's README lists what the tests
ran, what ran on the rig and what has never been run.

Run by tests: the automated tests on a simulated arm and on APAI's mock arm, with the robot connection as a real
link between programs on one machine; the two self-tests of the robot app; the perception stack on recorded stereo
pairs on GPU 0, with stand-ins for the model clients.

Run on the rig, by the README's account: the programs on both machines, in sessions of 2026-11-16 and 2026-11-17;
a pick, a lift, a hold and a place of an object by the agent on the real arm at 0.1 rad/s, with the agent program
on the workstation and the robot app on the Thor, which the rig owner reported on 2026-11-16 and of which the
repository holds no record; and a run of 2026-11-17 with the full stack, the camera connection and the robot
connection between the two machines, whose run folder holds the measurements of 14 moves and 4 jaw commands. The
README marks no step of its checklist for the rig as run: no record of the session of 2026-11-16 says which steps
passed.

Never run by a test: any motion of the real arm, any model call and any task on the full stack; anything between
two machines; a camera frame from a Thor; the robot app's start for the real arm other than against a stand-in for
the motor library; an Isaac Capture build on a Thor; the viewer's page in a browser window with a person in front
of it; a line typed by a person at a real terminal. The README reports no run, by a test or on the rig, of: a
headset, CloudXR or the Teleop Session Server with any of it; the human program on a Thor; a person's takeover or
drive on the arm; the live view on the rig's camera and robot connection. The README names what else has not run
on the real arm, the typed `rest` that ends with the motors off and the typed `on` among it. Also never run: the
repository's pre-commit hooks and the CTest entries of the example and of the `core::Link` wrapper; the tests under
Python 3.11 and 3.13. Every limit, speed and time in the shipped files is a first value, except the joint speeds
in the rig's files, which are the owner's values (section 8). Statements about the headset view, the motors after
a program death, permit timing and the timing of every other new path are unverified until the tests of section
19 are run.
