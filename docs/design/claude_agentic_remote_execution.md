<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Agentic remote execution with two links: Claude's proposal

Status: proposal by Claude (an AI assistant), revised 2026-11-13 after the project owner's decisions of that day.
Implementation starts now; nothing is finished, no hardware test was run. Every line count and duration is an estimate.

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
  only if the chosen demo profile needs them, section 12), and finishes the task.

The overriding instruction is to get something running quickly and refine afterwards.

## 2. Terms

| Term | Meaning |
|---|---|
| workstation | The GPU machine (the "server machine" of `remote_teleop_session.md`). It runs CloudXR, the Teleop Session Server, the robot link service, the agent program, the operator console and the live view program. |
| Thor | The Jetson Thor on the robot (the "robot machine" there). It owns the CAN bus of the arm: six joints and a gripper, no brakes. |
| robot app | The one program on the Thor that talks to the motors. New. It is also the robot app of the teleop link. |
| agent program | APAI on the workstation: task loop, models, perception, planning. |
| skill | One of APAI's ready-made motion routines, such as grasp or place. |
| LeRobot | The program that drives this arm in today's human-only remote teleop. |
| launcher | `thor_runtime.py`, the program that starts APAI on the Thor today and asks the operator's confirmations. |
| power, calibration and hold confirmations | What today's launcher requires at the Thor terminal before it connects to the motors: a typed yes to connect the powered arm, a valid camera mounting calibration, and no unresolved hold left by an earlier run. |
| robot link service | A small workstation program that holds the robot connection to the Thor and offers a local API. New. |
| arm adapter | A class in the agent program with the methods of APAI's arm class; it sends robot link messages through the robot link service instead of motor commands. |
| operator console | A small workstation terminal program for the operator: STOP, reset, rest, release, open, shutdown, takeover prompts. New. |
| live view program | A small workstation program that serves live camera pictures and a status line to a browser and tells the Thor how fresh that view is. New. |
| teleop thread | The robot app's background thread: it dials the Teleop Session Server, waits for READY, then reads SAMPLES and runs human driving. |
| setpoint | One set of joint targets, one value per joint. APAI sends one every 20 ms. |
| move | A list of timed setpoints from a start pose to a goal pose. |
| checker | APAI's `SafetyHarness`: the code that accepts or refuses each setpoint just before it is sent. |
| limit set | The joint, speed and workspace limits the checker applies. The Thor configuration holds one per control owner (section 8). |
| perception watchdog | A timer in APAI's checker: the checker refuses to start a motion when nothing has reported to it for 5 s. Each checker instance has one. |
| halt count | A counter in APAI's checker that goes up each time the checker is told to halt the motion under way. |
| obstacle grid | A 3-D array giving, for each 1 cm cell, the distance to the nearest obstacle. It covers only the gripper's workspace box (0.4 x 0.6 x 0.56 m); space outside it counts as free, today and in this design. |
| grasp region | A small region around the object being grasped in which the table and obstacle rules are relaxed. APAI's skills set it. |
| contact monitor | APAI's code that detects from the motors' feedback that the arm has touched something. |
| contact spot | A small zone around a place where the arm touched something. APAI's checker keeps the agent out of it until it is cleared. |
| hold | The arm stays where it is with motors on. |
| limp | Motors off. Away from rest the arm falls, because it has no brakes. |
| rest, return to rest | The folded pose in which the arm lies on its own stops; the move that brings it there. |
| recovery region | The start poses from which the straight move to rest has been run on the bench without touching anything. |
| jaws | The two fingers of the gripper. |
| right A, right B, grip, trigger | Controls on the right Quest controller: two face buttons, the squeeze button under the middle finger and the index-finger trigger. In this design right B is STOP and takeover, right A gives the arm back at rest, the grip is the clutch (the arm follows the hand only while it is squeezed) and the trigger drives the jaws. |
| closing stiffness, light holding stiffness | The two position gains APAI's driver uses on the jaw motor: a high one while the jaws close and a low one once they have stalled on an object, so that it is held with a light, steady force. |
| guarded opening | APAI's driver routine for opening the jaws; it protects the mechanical open stops. |
| referenced | After power-on the jaw motor knows its position only within one turn, and the jaws travel more than one turn. The driver finds the reference at start; without it a jaw reading counts as unknown. |
| control owner | Who may move the arm right now: agent, human, recovery (the robot app itself, for a return to rest) or nobody. |
| owner number | A counter on the Thor that goes up at every change of control owner. Commands quote it. |
| session id | A random value the robot app picks each time it starts ("executor session ID" in the first proposal; not the `session_id` of the teleop link). Commands quote it, so nothing survives a restart. |
| stop number | A counter on the Thor that goes up at every STOP. Commands and requests quote it. |
| stop level | None, STOP or fault-stop (section 9). A fault-stop is latched: it stays until a person types `reset`. |
| assistance wait | How long the arm holds for a person to take control after the agent has failed. |
| permit | A short-lived permission from the workstation to send the next part of a move, tied to the obstacle grid the workstation checked that part against. |
| clock pair | One reading of the Thor's monotonic clock and one of its wall clock, taken together. Camera frames carry wall-clock times, joint readings monotonic times. |
| link, link message | An Isaac Capture connection (`core::Link`) and one unit sent on it, as in `remote_teleop_session.md`. On the teleop link, SAMPLES carries the headset and controller samples, and READY says that a headset has been tracked for about a second. |
| Headless | A checkbox in the headset's CloudXR client. When it is on, the client sends tracking and draws nothing. |
| test arm | APAI's mock arm: software, no hardware (section 17). |

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

```text
            WORKSTATION                                           THOR (on the robot)
  agent program (APAI) --+
  operator console ------+-- local API --> robot link     robot connection
  live view program -----+                 service  <--------------------------->  robot app
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
| Tasks, consent, operator console, live view | Yes | The STOP key; the power, calibration and hold confirmations (at start and after a release); `rest`, `release` and shutdown under the same checks as at the console |

The robot app is one process. It runs APAI's motor driver and APAI's checker, unmodified, created without an obstacle
grid (its obstacle check is off and the permit check stands in for it); none of APAI's checks is copied or rewritten.
Besides the permit check, two pieces around it are written new: the agent's envelope (the narrow region around its
working area that the agent must stay in), which today is a wrapper in the uncommitted Thor adapter files, and the code
that builds a checker from a limit set. The Thor stays small in work, not in code loaded: importing the checker loads
much of APAI's Python (OpenCV included), but no model and no GPU code. The models stay as today (cloud language model,
SAM 2.1 Tiny segmentation, FFS stereo depth). If APAI is replaced later, only the agent program changes with it. The
robot app keeps APAI's motor driver and checker until they are replaced separately, and the robot link messages name
nothing from APAI.

## 5. Connections

Two connections carry robot traffic between the machines: the teleop connection and the robot connection. The camera
connection is separate, as in the first proposal. The first proposal counts the camera link too and so calls its own
design "two logical links"; this file does not count the camera connection.

| Connection | Carries | State |
|---|---|---|
| Teleop | SAMPLES from headset and controller, workstation to Thor | Existing code, not edited |
| Robot | Moves, permits, gripper commands and operator requests to the Thor; robot state, results and events back | New; same `core::Link` class |
| Camera | Compressed stereo video and the time of each camera frame, Thor to workstation | Isaac Capture pull request #1410; temporary, the owner expects it to be replaced; has run only as a standalone streamer so far |

Why a second connection instead of one through the Teleop Session Server:

- Nothing that works today is edited. Carrying robot traffic on the teleop connection changes the teleop client, the
  host, the server and `TeleopSession`, and the automated remote tests and hardware goals 1, 2, 3 and 5 of the hardware
  test record have to be repeated.
- A CloudXR fault or a bad glove sample ends the Teleop Session Server's connection today. With two connections that
  ends human input only; the agent, the operator console and the assistance wait go on.

The price: the robot app watches two connections, each with its own "alive" state; robot data does not pass through the
Teleop Session Server; and starting the agent before any headset exists relies on the teleop thread, not yet run.

**Robot link service.** The workstation end of the robot connection is one small, always-running program. It offers a
local API: read robot state and history, send commands, send operator requests, subscribe to events. The agent program's
arm adapter, the operator console and the live view program are clients of that API. STOP, reset and the takeover
prompts therefore keep working when the agent program has crashed. When the agent's API client closes while the agent is
control owner, the service sends an OWNER_REQUEST "help needed" with the reason "agent program gone"; this is how the
Thor learns of a crash. In the first proposal the same API lives inside the Teleop Session Server; the clients are the
same code in both designs.

**Teleop connection, unedited.** The teleop thread dials it when the robot app starts and waits, without a time limit,
until the server reports READY. The agent runs meanwhile. The wait uses the lower-level `RemoteSession` binding in a
retry loop, because `TeleopSession` entry cannot be cancelled and ends at once when the server is busy or shutting down.
After READY the same thread enters a `TeleopSession` in `SessionMode.REMOTE` and steps it at 50 Hz until the connection
is lost; then it waits again. Right B is read there while the agent works, and human driving runs there after a
takeover. Nothing restarts the Teleop Session Server today, so it runs under a restart script. The console learns that a
takeover is not possible: at once when the server process is gone, after about 1 s when the controller is lost, and
after about 3 s when the teleop connection goes silent (measured with a firewall rule on the teleop port; a pulled cable
also takes the robot connection). A hung server is not noticed before READY. After a CloudXR restart, Headless must be
switched on again in the headset before a takeover is possible. If the bench test of this thread fails, the fix is an
edit to the same teleop files that the first proposal edits.

**New Isaac Capture library code.** Only a Python wrapper for `core::Link`, which today can be opened only from C++
(estimate: 450 to 700 lines of source, 600 to 800 of tests, added beside existing code). All other new code is example
code. Robot link message type numbers start at 64 here, and the same choice is proposed for the first proposal, which
names no numbers; the teleop link uses 1, 2, 11 to 14 and 32 to 35. The robot app dials the robot link service at the
configured workstation address, and the service accepts the robot connection only from the configured Thor address.

## 6. Robot link messages

One module defines every robot link message and its encoding, in robot terms, and both machines import it. Everything
above the connection uses three calls only: send reliable, send newest-wins, receive. "Reliable" means queued and
delivered in order; "newest wins" means an unsent older link message of the same type is replaced.

Every command (MOVE, GRIPPER, CHECKER_STATE, OWNER_REQUEST, RESET, RELEASE, OPEN, SHUTDOWN) carries a command id, the
session id, and the owner number and stop number its sender last saw. The table lists what else each carries.

| Link message | Direction | Delivery | Contents |
|---|---|---|---|
| WELCOME | Thor to workstation | Reliable | Session id; robot and profile identity; joint names, signs, units and limits; limit-set versions; camera calibration and coordinate-system versions |
| ROBOT_STATE | Thor to workstation | Newest wins | Per joint the value, read start, read end and a valid flag (measured only, never a cached value and never the commanded target); the last target sent with its send time; the gripper value with its own time and valid flag; control owner, owner number, stop number, stop level and limit-set version in force; at rest, whether torque is on or released; the latest command outcome; how far ahead the accepted setpoints reach; the assistance wait and its remaining time; human input (teleop connection up, the stage the Teleop Session Server last reported, READY seen, age of the newest SAMPLES) and the age of the live view; health (reading ages, read failures, contact, motor and gripper faults, motor temperatures where the driver can read them, loop alive, missed deadlines, clamping); and a clock pair |
| MOVE | Agent to Thor | Reliable | The move id (its command id), the expected start pose with a tolerance, the timed setpoints, the grasp region if any, the Thor wall-clock time of the camera frame behind the grid the move was checked against, an expiry on the Thor clock |
| GRIPPER | Agent to Thor | Reliable | Target opening, maximum effort, plain or close-until-contact |
| PERMIT | Agent to Thor | Newest wins | Move id, the last setpoint it covers, the Thor wall-clock time of the camera frame behind the grid it was checked against, and an expiry on the Thor clock |
| CHECKER_STATE | Agent to Thor | Reliable | Tells the agent checker to forget its contact spots, or to end the grasp region of a grasp that was finished or given up, when APAI's skills do so on the workstation |
| STOP | Any client to Thor | Reliable | Reason and sender |
| OWNER_REQUEST | Agent, console or robot link service to Thor | Reliable | Requested control owner (agent for a new confirmed task, recovery for a typed `rest`, nobody) or "help needed" with the reason. For "agent" it also says whether the timed return to rest (the one at the end of the assistance wait, section 10) applies to this task (default yes). A takeover is not requested here; it comes from right B on the teleop connection (section 12) |
| RESET, RELEASE, OPEN, SHUTDOWN | Console to Thor | Reliable | OPEN and SHUTDOWN say whether the operator typed the confirmation they need (the jaw opening, section 13; a clear path, section 11). RESET and RELEASE carry nothing more |
| VIEW_STATUS | Live view program to Thor | Newest wins | Whether a browser is showing the live view, and the age of the newest camera frame shown |
| RESULT | Thor to workstation | Reliable | For each command id: accepted or refused with the reason, then completed or cancelled with the measured end pose |
| EVENT | Thor to workstation | Reliable | Text for the operator: help needed, countdowns, refusals with reasons |

ROBOT_STATE gives times per joint, because a joint read takes about 8 ms and the 20 ms motor loop makes one or two reads
per pass, so each joint is read again only every 60 or 120 ms.

Rules:

- A command that quotes an old session id, owner number or stop number is refused. A repeated command id is answered
  again and not executed again. After a reconnect nothing is replayed.
- A MOVE is refused if the measured start pose differs from the expected one, if its expiry has passed, or if it is
  longer than a Thor setting allows. Limits are never in a link message; the Thor picks a limit set by control owner.
- A PERMIT never reaches past the end of its move, and a link heartbeat is never a permit.
- The robot link service sends a STOP ahead of unsent MOVE and GRIPPER link messages and then stops sending permits.
- The live view is fresh while the newest VIEW_STATUS says that a browser is showing it and the age it reports, plus the
  time since it arrived, is under the view age limit (a Thor setting; its first value is chosen at the headset check of
  section 19). A link heartbeat is never a VIEW_STATUS.
- Existing Isaac Capture schemas are not changed.

## 7. The check before each setpoint

Today APAI checks every setpoint just before sending it, in under a millisecond: finite values, joint limits, speed,
workspace box, table, contact spots, and obstacle distance. The last reads the obstacle grid, which perception builds
from camera frames. After the split the grid is on the workstation, so the check is divided as in the first proposal:

- Before a move is sent, the workstation checks the whole move, obstacle distance included.
- While the arm moves, the workstation keeps checking the part just ahead of the arm against its newest grid and sends a
  PERMIT for it. The first PERMIT of a move is sent right after its MOVE; the Thor starts the move when it holds both.
- The Thor checks every setpoint just before sending it: the camera-free checks, and that an unexpired permit covers it
  and rests on a camera frame younger than the grid age limit (3 s today). It makes no network round trip per setpoint.
  Each permit also feeds the agent checker's perception watchdog, without which that checker refuses all motion.
- No permit, an expired permit, or a permit that rests on a camera frame older than the grid age limit ends the move in
  a hold. A link heartbeat is never a permit, so a hung agent program is caught as well.
- How far ahead a permit reaches and how late it may arrive are measured before the first motion and chosen from arm
  speed and stopping distance; they are not copied from a camera or connection timeout. Acceptance test: an obstacle
  placed in the path during a move stops the arm as it does today.
- The Thor converts each camera-frame time from its wall clock to its monotonic clock on arrival and rejects a permit
  whose camera-frame time is in the future.
- After a delay the Thor does not send setpoints faster to catch up, and it skips none. A delay below the small-delay
  limit moves the remaining deadlines back, never past the permit's expiry; a delay above the large-delay limit ends the
  move in a hold. Both limits are chosen from measurements on the Thor before the first motion. A move that ended early
  is never resumed; the agent may send a new MOVE, checked from the measured pose like any other.

Sending the grid itself to the Thor, so that the Thor runs today's full check locally, is future work (section 20). It
is also the fallback: if the permit timing measured in section 19 is too slow for the agent's speed, the speed is
lowered, and if that is not enough the grid on the Thor is built earlier than planned. The check is never weakened.

## 8. Control owners and limits

Three versioned limit sets live in the Thor's configuration: agent, human and recovery. The control owner selects the
set. There are three checker instances, one per owner.

| Limit set | What it applies |
|---|---|
| All three | Finite values, joint limits, a hardware speed bound, table, workspace and configured keep-out boxes |
| Agent | Its envelope, the contact spots and grasp state it has collected, its speed limit, and the permit check of section 7 in place of the obstacle check. Its watchdog is fed by each PERMIT. An invalid agent move is refused, never clamped |
| Human | No obstacle check; its watchdog is fed by fresh controller samples. It has a hand speed limit by measured time (first values 0.5 m/s and 2.5 rad/s), a joint speed clamp (first value 1.5 rad/s) and a per-command step clamp. Targets are clamped, then checked, and clamping is reported in ROBOT_STATE. The agent's stored state does not apply, because it would block a person exactly where help is needed |
| Recovery | No agent envelope, because that envelope cannot be re-entered from outside, and no obstacle check in the first build, because the grid stays on the workstation. Its perception watchdog has no age limit, because nothing can feed it; with the default 5 s limit an unmodified checker refuses every move, and today's exit move patches the watchdog out instead. It keeps the contact spots the agent checker has recorded so far: leaving one is allowed, entering one is refused. The agent's grasp state does not apply. It accepts as a start pose every pose the human set accepts; a unit test checks this |

Only one control owner at a time. Changing owner always passes through a hold. While a person owns the arm, an
OWNER_REQUEST from the agent is refused.

## 9. STOP and fault-stop

STOP needs no ownership. It can come from the operator console (`stop`), a key at the Thor terminal that needs no
network, right B on the controller, and the agent. Right B works only while a headset is worn, the controller is active
and the teleop connection has reached READY.

| | STOP | Fault-stop |
|---|---|---|
| Raised by | A person or the agent | Motor fault, gripper fault, joint feedback lost for more than 0.35 s, contradictory state |
| Arm | Stops at the next pass of the 20 ms motor loop and holds, motors on | Same |
| Jaws | No jaw command that quotes an older stop number is accepted. A close under way is ended at once by re-sending its target at the light holding stiffness. Never opened, never left at the closing stiffness | Same |
| Task | Cancelled, never replayed; control owner becomes nobody | Same |
| Assistance wait | Cancelled when the STOP comes from a person (operator console, Thor terminal key, right B): the arm holds until someone acts. Unchanged when it comes from the agent | Cancelled |
| What may happen next | New task (after a takeover only once the arm was given back, section 12), takeover, typed `rest` or `open` | Nothing until a person types `reset` |

`reset` clears the latch and nothing else: it starts no motion and no wait. Neither level cuts torque. The 0.35 s is
APAI's contact-monitor limit, which today applies only during a move and ends it without a latch. Using it as a
fault-stop, and in hold, is new; the value is checked on the bench.

A contact is not a fault-stop: the motion stops and the arm holds the freshly measured pose, so it does not keep
pressing. Under the agent the step fails and counts toward the retry limit; under a person a fresh grip squeeze resumes;
during a return to rest the move ends and is not retried.

Built from: STOP calls `SafetyHarness.halt()` on all three checker instances and raises the stop number. APAI's halt is
cleared by the next move, so it is the robot app's own stop-number check that keeps old commands out. Fault-stop calls
`SafetyHarness.estop()` and is cleared by `reset_estop()`. The driver's own `stop()` is not used, because it makes later
jaw commands vanish without an error, and `AgentOrchestrator.resume()` is not used, because it also drives the arm home.
The jaw rule is new code. The worst-case delay from a typed `stop` to the arm holding is measured during an agent move.

Ctrl+C at the Thor terminal is STOP: the arm holds and the robot app keeps running. A closed terminal is ignored. The
app exits only on the operator's shutdown command, typed at the operator console or the Thor terminal, which is a return
to rest followed by the permitted torque release; if that return is refused or fails, the arm holds and the app keeps
running. A shutdown that has to move the arm asks the same path-clear confirmation as a typed `rest` (section 11). The
48 V switch is the only hardware stop, and it drops the arm.

## 10. Faults and the assistance wait

Every fault first ends in hold. Only after an agent failure does a wait then decide how long the arm holds before it
goes to rest by itself: the assistance wait. It starts when the agent has failed while it owns the arm and the robot
connection is alive, and it waits for a person to take control. The first value is 120 s, never below 90 s. When it runs
out, the robot app makes one attempt at a return to rest (the timed return), if the switches of section 11 allow it. The
robot app counts the wait on its monotonic clock and shows it as a countdown on the operator console and the Thor
terminal.

- "The agent has failed" means: a step returned "stuck" or "blocked", a model error could not be recovered, a set number
  of steps failed in a row (first value 3), the step budget ran out, or the agent program crashed (reported as in
  section 5). One failed step is retried inside the task. Model thinking and waiting for consent never start the wait.
- The wait is cancelled by: a takeover the Thor has accepted, a new task confirmed with the typed "yes", a STOP from a
  person, a fault-stop, or loss of the robot connection. A headset that connects or is tracked does not cancel it.
- Before the arm moves by itself the operator console and the Thor terminal ring the terminal bell at 30 s, at 10 s and
  on each of the last 5 s, and print `ARM WILL MOVE TO REST`.
- The wait makes one attempt. A refused or stopped return leaves the arm holding with the reason shown; nothing
  automatic follows; the operator must type `rest`, take over or send a new task.

| What happens | Response |
|---|---|
| Agent fails as defined above | Hold; the agent's ownership ends (control owner nobody, owner number goes up); help requested; assistance wait |
| No valid permit during an agent move | Hold; the agent may send a new MOVE |
| Headset absent while the agent works | Nothing; the console shows that a takeover is not possible yet |
| Teleop Session Server or CloudXR down | Human input unavailable; the agent, the console and the assistance wait go on |
| Headset, controller signal or the live view in the workstation browser lost while a person drives | Hold. The person keeps control; buttons held down before the loss count only when pressed again, and the clutch is re-based (the hand position at the next grip squeeze becomes the new starting point); re-engaging needs fresh samples, a fresh view and a new grip squeeze. No timed return to rest in the first demos. To end the takeover without a headset the operator types `rest`; if the jaws hold something: `stop`, then `open`, then `rest`. The hand-back of section 12 step 7 still needs the controller |
| The teleop thread throws an error | Treated as controller signal lost; never a return to rest or a motor disconnect |
| Agent program crashes while a person drives | Nothing; the person keeps control |
| Robot connection or whole workstation lost | Hold. If the agent owned the arm, its ownership ends (control owner nobody, owner number goes up). A pending wait is cancelled, a timed return under way ends in a hold, and a reconnect starts neither again. `rest` can still be typed at the Thor terminal |
| Contact | Hold at the measured pose; not latched (section 9) |
| Motor fault, gripper fault, lost joint feedback | Fault-stop. A motor that reports its own fault may stop holding and software cannot prevent it; the fault is cleared only at rest or with the arm supported by hand |
| Robot app dies | No software is left to act. The motors have so far kept holding their last target after such exits; for how long has never been measured. The operator supports the arm by hand before restarting the robot app, because connecting cuts torque. The operator console shows "robot connection lost": at once when the process dies, after about 3 s when the Thor hangs. Text and the terminal bell only; no loud alarm |
| CAN bus, power or Thor failure | No software response |

## 11. Return to rest and torque release

A typed `rest`, the rest that ends a takeover and the return after the assistance wait are the same checked move.

1. Conditions, all required, checked before the control owner changes: the jaws are proven empty; no fault-stop; the arm
   is not holding at a contact (section 9; such a hold ends when the agent or a person next moves the arm); two full
   joint reads 0.2 s apart are valid and agree (taken with the reader that fails instead of returning a cached pose);
   the start pose is inside the recovery region; and the gripper tip is above the table clearance and outside the last
   grasp region. If any fails, the return is refused, the operator console and the Thor terminal say why, and nothing
   changes.
2. Control owner becomes recovery. Motion and jaw commands from agent and human are refused. A takeover request is still
   accepted: a running return first ends in a hold, then the human becomes owner.
3. The robot app records the stop number, the owner number and the recovery checker's halt count. The first setpoint and
   every later one are sent only if all three are unchanged, so a STOP at any moment wins.
4. The move is one straight joint-space move from the measured pose to the saved rest pose at the recovery speed (first
   value 0.1 rad/s; the planned time is printed). By APAI's timing formula that is about 21 s per radian of the largest
   joint travel: about 44 s from the edge of the agent's 120-degree envelope and up to about 65 s from a joint limit.
   The one recorded return at this speed took about 37 s. The Thor plans no route. The recovery checker checks the path
   before the first setpoint and every setpoint while moving, with the contact monitor on. The move ends at the first
   contact.
5. It ends at rest with motors on. After the `rest` that ends a takeover the person becomes the control owner again, so
   that the arm stays closed to the agent until right A (section 12 step 7). After a timed return, or a `rest` typed
   while nobody owned the arm, the control owner becomes nobody. A return that ends early, in a hold away from rest,
   gives the same owner as one that reaches rest, except that a STOP always makes it nobody (section 9).

Without the robot connection no timed return runs (section 10). A typed `rest`, at either terminal, first asks for a
typed confirmation that the path is clear, and so does a shutdown typed while the arm is away from rest. This prompt is
new: today's supervised exit move only prints "Keep the return path clear" and starts. The timed return has no such
prompt; it relies on the tested recovery region, the warning bell and the contact monitor. No return to rest is checked
against the obstacle grid in the first build (section 8).

**Jaws proven empty.** All of these must hold: the gripper was referenced when the robot app started and the reference
is still valid; the "may be holding" flag is clear; three jaw readings 50 ms apart are valid and agree; the reading is
within 0.1 rad of the last jaw target the Thor sent; and that target is full open, or the reading is within 4 % of
travel of closed. The flag is set when any close ends short of its target or a jaw read fails during a close, and is
cleared only by a measured full open. Anything else counts as loaded or unknown, and the arm holds. The jaws are never
moved to decide. Known limit: an object thinner than about 4 % of jaw travel reads as empty; that is about 3.6 mm if the
configured 90 mm travel is right, which the arm profile itself questions. Not measured.

**Torque release.** Torque is released only on an operator request (typed `release`, or the shutdown command), never by
a wait, and only when two full joint reads 0.2 s apart are within 0.05 rad of rest and of each other, the jaws are
proven empty and no move is running. A failed or interrupted return, a stale reading, a disconnect or a second Ctrl+C
never leads to a release. Motors are switched on again only through the power confirmation at the Thor terminal, with
fresh readings; no earlier task is replayed.

**Switch.** The timed return has a switch in the Thor configuration, on in the demo configuration. Each confirmed task
also says whether the timed return applies to it (default yes); the value travels with the agent's OWNER_REQUEST. Before
its first unattended use on an arm, the bench tests of section 19 are run on that arm with a person supporting it.

Built from: `SafeArm.move_joints` to the rest pose on the recovery checker, with the whole path checked first as
`thor_shutdown.py` does. Not used: `park_on_shutdown` (it works only while the app is exiting) and `move_to_rest` (it
reads flags that live on the workstation and by default ends in torque-off). New: the recovery owner, the recovery
checker instance, the jaw rule, the path-clear prompt, the wait and the release command.

## 12. Human takeover

A takeover is accepted when right B was held for 1 s with the grip released and every sample in that second was fresh, a
full joint read taken now is valid, the live view in the workstation browser is fresh, and no fault is latched. A
tracked headset is not proof that someone wears it. A press of right B is always a STOP first; if B then stays down for
1 s it also requests control.

1. The robot app starts the assistance wait. The console shows why the agent stopped, whether the jaws may be loaded,
   whether a return to rest would be accepted now, and the countdown, and prints the steps below one at a time.
2. The operator puts on the headset, picks up the right controller, stands behind the arm and recentres the headset view
   with the grip released.
3. The operator holds right B for 1 s. The robot app makes the human the control owner and holds.
4. While the grip is squeezed the arm follows the hand, starting from the measured pose; releasing it holds the arm and
   leaves the jaws alone.
5. Jaws: at the owner change no gripper command is sent, so a held object stays held. The trigger is ignored until it
   reaches the agent's last grip level, and until the jaws have been fully opened once it cannot command more closing
   than that. A close by the person ends like the agent's: when the jaws stall, the stiffness drops to the light holding
   value. Opening uses the driver's guarded opening in small steps.
6. The person finishes the task or abandons it, puts down or releases any object, and asks for the return to rest
   (`rest` at the operator console). The console then asks "completed or abandoned?" and stores the answer in the task
   record. Until then the person stays the control owner; releasing the grip only holds.
7. At rest, holding right A for 1 s with the grip released gives the arm back, also with torque released. It needs
   measured rest, stillness, jaws proven empty and no fault-stop. The robot app closes the interrupted task for good as
   "completed by human" or "abandoned" and sets the control owner to nobody. Nothing moves; the agent waits for a new
   task.

After a takeover a new agent task is accepted only after the hand-back of step 7. In all cases it also needs: no person
owns the arm, no fault is latched and the measured pose is inside the agent limit set. Otherwise the agent program's
terminal refuses the task and names the reason: "takeover not finished: give the arm back with right A at rest",
"fault-stop: type reset at the operator console", or "arm outside the agent's limits: type rest at the operator
console".

Human driving uses Isaac Capture's controller source, clutch and end-effector rate limiter (the nodes today's teleop
uses), then APAI's inverse kinematics with no random restarts, an iteration limit and a joint-jump check, then the human
checker and APAI's driver, at 50 Hz. LeRobot's package is not imported into the robot app; its tested limiter and clutch
values are copied. If APAI's solver fails a trial on the test arm with the real headset, the fallback is LeRobot's
solver. It cannot be loaded beside APAI's kinematics as they are built today (section 16), so that fallback means one
merged environment that neither stack has been tested in. It is not in the estimate. The first profile is the Quest with
the right controller, as in the first proposal; gloves are started only if the chosen demo profile needs them.

Human-only mode (a person drives from the start, no agent) is today's LeRobot program, unchanged. It is not run at the
same time as the robot app, and it does not meet the never-go-limp rule today.

## 13. Operator: tasks, consent, commands and view

| Terminal | What the operator types there | What it shows |
|---|---|---|
| Agent program (workstation) | Tasks and the typed "yes" before motion | Task progress |
| Operator console (workstation) | `stop`, `reset`, `rest`, `release`, `open`, shutdown | State, assistance wait, takeover steps, refusals with reasons |
| Thor terminal | The STOP key; the power, calibration and hold confirmations (at start and after a release); `rest`, `release` and shutdown under the same checks as at the console | The same state lines and countdowns |

- The "yes" is asked at the terminal that sent the task. In the first build only the agent program's terminal sends
  tasks, so it is asked there. The prompt code comes from today's launcher and moves to the workstation. There it must
  learn from ROBOT_STATE, not from a file on the Thor's disk, whether an earlier run left the arm holding, and the power
  confirmation moves into the robot app (estimate 30 to 50 lines of source, 40 of tests). Opinion on the open question
  of tasks from the Thor terminal: not in the first build; it costs about 200 to 250 lines of source and 150 of tests
  and points away from an operator in another city. The task record stores where a task came from, so this can be added.
- `open` opens the jaws through the driver's guarded opening after a second typed confirmation. It is accepted only in
  hold with control owner nobody and no fault-stop. It is the way to let go of an object without a headset.
- View: the owner asked whether the browser view should be served from the workstation or the Thor. Opinion: the
  workstation, because the video and the perception data are already there and the Thor stays small; the first proposal
  does the same. Both demos show live camera video and robot state in a workstation browser, from a bounded in-memory
  source that does not depend on recording. First form: the live view program serves pictures from the camera receiver
  and a status line with camera frame and state ages; APAI's existing viewer (`thor_viewer.py`), run on the workstation,
  keeps showing the obstacle grid and replay.
- A takeover needs a fresh live view in the workstation browser, and loss of that view while a person drives holds the
  arm. The live view program reports its freshness to the Thor through the robot link service (VIEW_STATUS, section 6).
  Replay never counts as live.
- According to the source code of the headset's CloudXR client, a headset with Headless on shows a flat dark-blue field.
  How the operator sees the browser view while wearing the headset is settled by the headset check of section 19.

## 14. Start-up order

1. The workstation starts its programs; none waits for a headset.
2. The Thor starts the robot app and the camera sender. The power, calibration and hold confirmations stay at the Thor
   terminal. The robot app connects to the motors once, with the arm at rest, and never reconnects by itself.
3. The Thor dials the robot link service and, in the background, the Teleop Session Server. The robot app sends WELCOME,
   and state and video start. The agent is ready when robot and perception are ready, whether or not a headset was seen.
4. The operator types a task and its "yes" on the workstation.

## 15. What follows from the link count

| Topic | First proposal (one link) | This proposal (two links) |
|---|---|---|
| Isaac Capture code | Teleop protocol, client, host, server and `TeleopSession` extended, with negotiation for old clients | A Python wrapper for `core::Link`; teleop code untouched |
| Workstation end | The API lives in the Teleop Session Server | The API lives in the robot link service |
| Connections the robot app watches | One | Two, each with its own "alive" state; robot data does not pass through the Teleop Session Server |
| Agent before any headset | By a protocol change | By the teleop thread, not yet run |
| CloudXR, OpenXR or tracker fault | If fatal, the robot connection is lost | Human input only is lost |
| Teleop Session Server dies during the assistance wait | The timed return is cancelled and the arm holds | The wait goes on; the arm returns to rest if the conditions of section 11 hold, although no takeover is possible |
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
| Return to rest | Reuses `move_to_rest()` | Built on `SafeArm.move_joints`; section 11 says why |
| Checker on the Thor | Extracts the camera-free checks from APAI's checker | No extraction: APAI's whole checker, created without a grid. That switches off only its grid checks, and the permit check of section 7 stands in for them. One instance per control owner (sections 4 and 8) |
| Torque release | A normal return to rest or shutdown "may" release torque; keeping it on at rest is also allowed | Decides the open choice: a return to rest ends with motors on, and torque is released only on a typed `release` or the shutdown command (section 11) |
| Rules | States goals or is silent | Proposes rules. They are this file's own, not owner decisions, and stand unless the owner objects: how "jaws empty" is measured, the straight recovery move and its speed, and the path-clear prompt (section 11); what STOP does to a closing gripper, the STOP key at the Thor terminal, Ctrl+C as STOP, the shutdown command and contact as a hold that is not latched (section 9); `rest`, `release` and shutdown at the Thor terminal (section 13); contact spots that do not bind a person (section 8); the warning bell and a new task that cancels the wait (section 10); and building the shared part first on one machine (section 17). The same holds for detail the first proposal leaves to the build, such as the link message table (section 6), the recovery checker's settings (section 8), the first form of the live view (section 13) and the first values of limits, speeds and times. The trigger limit at takeover (section 12 step 5) is also this file's own rule; it is listed below because it awaits a yes |

Awaiting the project owner's yes:

| Point | What is asked |
|---|---|
| Trigger limit at takeover (section 12 step 5; either proposal) | When a person takes over while the arm holds something, the trigger becomes the gripper control. Proposed: the trigger does nothing until it has been pulled as far as the agent had closed the jaws, and it cannot squeeze harder than the agent did until the jaws have been fully opened once. Taking over can then neither drop nor crush the object |
| The robot link service (section 5; this proposal only) | One more small workstation program |
| Permits or the grid on the Thor | The owner chose permits as the easier start. By the estimates behind this file permits are the larger build (about 700 to 1,100 lines with tests, against about 350 to 600 for sending the grid to the Thor). Permits stay unless the owner switches |
| A silent agent after a hold | If a permit runs out during a move, the arm holds and the agent may send a new MOVE. If the agent then sends nothing more, no rule calls that "failed", so no assistance wait starts and the arm holds until someone types a command. Should some seconds without any agent command after such a hold count as failed? Model thinking before a move would still never count |

## 17. Build plan

| Phase | What is built | Gate |
|---|---|---|
| P0 | Robot app, control owners, STOP, hold, return to rest, torque rule and link message module on one machine, with a test arm over a loopback version of the three calls. Audit of every call that can disable a motor | Refused or expired commands cannot move the arm; no disable call is reachable away from rest |
| P1 | The connection layer (the phase where the two proposals differ most) and perception on the workstation | A test arm completes commands over the real connection; nothing replays after a reconnect. Each camera consumer copies a camera frame and checks that its slot was not overwritten meanwhile; stereo pairs and the calibration id are kept; depth and one grasp pose computed from compressed camera frames are compared with uncompressed ones |
| P2 | Demo 1 with the live view and the operator console | A repeatable task without a headset; working `stop` and `rest` |
| P3 | Demo 2, after the headset checks | Takeover and the assistance wait work without restarting the robot app; a STOP during the wait prevents the timed return |
| P4 | The fault table under each control owner, measured delays, launch and recovery instructions | Results written down with what remains open |

P0 needs no GPU, no Isaac Capture build and no robot. It runs in a new Python 3.12 environment with APAI's source on the
Python path; this was tried on the workstation on 2026-11-13: 16 APAI tests of the checker, the kinematics and a
mock-arm grasp and place pass there, and 9 of the 10 Thor shutdown tests; the rest of APAI's tests were not run. The
test arm is APAI's mock arm with failed and late joint reads, per-joint read times, jaw travel over time and paced
sending added. On the workstation, builds, tests and perception use GPU 0 only.

Work happens in a separate checkout on a branch from `rahulsi/remote-teleop-session`; nothing is pushed. Example code
goes under `examples/agentic_remote/` and its tests under `tests/python/examples/agentic_remote/`. APAI is imported from
its own checkout through an environment variable, and tests skip without it. The Thor adapter files of the current
experiment exist only as uncommitted files in another checkout; what the robot app needs from them is written new here.

## 18. Time estimate

Hours are Claude's active working time (coding, tests, builds, debugging), not a human engineer's time and not elapsed
time. Sessions on the arm with the operator are counted separately. The rows and scope are the first proposal's, so the
two can be compared. The first row depends on the link count; the fifth differs by half an hour (the teleop thread) and
the sixth has the same size with different tests. The two totals are the same within the error of the estimate.

| Work | First proposal | This proposal |
|---|---|---|
| Connection layer and local API | 4 to 7 h | 3 to 5 h |
| Thor execution, real feedback, hold, checks, torque rule | 6 to 10 h | 6 to 10 h |
| Camera, perception on the workstation, obstacle check | 5 to 9 h | 5 to 9 h |
| Live view and operator console | 4 to 7 h | 4 to 7 h |
| Human path, STOP, assistance wait, takeover | 4 to 7 h | 4 to 7.5 h |
| Integration tests, instructions, fixes found on hardware | 7 to 15 h | 7 to 15 h |
| Total | 30 to 55 h | 29 to 53.5 h |

A first task running end to end without a headset is expected after about 14 to 24 hours. Two to four sessions with the
operator are needed in both designs. The link count changes the risk more than the hours: with two links nothing that
works today is edited.

## 19. Bench tests that need the operator

Before the arm moves under the robot app:

- Read the motors' CAN timeout parameter (0x7028) on all seven motors.
- Kill the current app during a raised hold with the arm supported by hand, empty and loaded, and watch for 10 minutes.
  Log motor temperature during a long loaded hold.
- Jaw rule with a cube, a wallet, a thin card and empty jaws.
- Return to rest from poses across the demo workspace with the arm supported, and once with a foam block in the path.
- Delay from typed `stop` to the arm holding, during an agent move.
- The 0.35 s joint feedback limit while the arm holds, at the read rate of a hold.
- Permit timing: how long the workstation's check ahead takes and how late a permit arrives; then an obstacle placed in
  the path during a move.

Before demo 2:

- Headset check: with Headless on, can the wearer see the arm and the browser view below the visor; with Headless off,
  does the room show and do controller samples keep reaching the Thor.
- Right A and B over the teleop connection (only grip and trigger have been seen on hardware), including short presses.
- The teleop thread beside the 50 Hz motor loop; server started before any headset connected; re-entry after a
  connection loss.
- APAI's inverse kinematics under a moving hand, on the test arm with the real headset.
- Live view: freeze the browser, close it and stop the camera sender while a person drives; the arm must hold each time.
- Time from the help request to an accepted takeover, starting with the headset asleep; the assistance wait's first
  value is checked against it.

## 20. Not in the first build

**Remaining work after the demos.**

| Work | Note |
|---|---|
| One connection through the Teleop Session Server | If wanted later, the service's code moves into the server and the robot link messages move onto the teleop connection. That is the first proposal's Isaac Capture edit (estimate 1,150 to 1,550 lines of source, 1,400 to 2,000 of tests, hardware goals 1, 2, 3 and 5 repeated). Nothing above the three calls changes |
| A second Thor program that survives the first one's death | Every existing motor connect cuts torque, so this needs a new way to attach to a held arm |
| The obstacle grid sent to the Thor, so that the Thor runs today's full check locally | That check reads no camera images, only the grid and the list of occupied cells (together about 0.6 MB for today's workspace box). With it: a backup check on the Thor for when the workstation fails, and moving checks between the machines as workloads move |
| A timed return to rest after the headset or controller is lost while a person drives, or after the robot connection is lost | Both cases hold in the first demos (section 10) |
| Human-only mode under the never-go-limp rule | It does not meet the rule today (section 12) |

**Optional later.** Tasks from the Thor terminal (the owner has not decided; opinion in section 13); camera picture in
the headset (the owner asked for a feasibility check; not done yet); browser buttons; recording for datasets; fast
policies on the Thor; control with gloves alone.

## 21. Evidence

Based on reading code at Isaac Capture commit `4a2b98420` (branch `rahulsi/remote-teleop-session`), the APAI checkout
pinned at `3c8b773`, the uncommitted Thor adapter files of the experiment checkout, the LeRobot checkout used for
today's remote teleop, the local checkout of Isaac Capture pull request #1410, and the hardware test record. No robot
motion, model call or hardware test was performed. Statements about the headset view, the motors after a program death,
permit timing and the timing of every other new path are unverified until the tests of section 19 are run.
