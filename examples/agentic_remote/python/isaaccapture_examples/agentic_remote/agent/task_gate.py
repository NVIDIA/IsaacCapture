# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a task takes the arm, keeps it and gives it back.

TaskGate wraps APAI's AgentOrchestrator.run_task and SkillRuntime.execute.
Thinking and looking never take the arm. At the first motion skill of a task
the gate reads ROBOT_STATE, asks the typed "yes" at the terminal the task was
typed at, and asks the robot app for control ownership. When the task ends
while the agent owns the arm, the gate gives the arm back (success) or sends
a help request (anything else but a cancel), waits for the RESULT, and only
then closes the task on the arm adapter.

A skill stop of APAI latches the workstation checker. The next task clears
that latch at its first motion skill, after the typed "yes" and the Thor's
grant; nothing on the Thor is cleared by it. Where the latch does not clear,
or with task.clear_skill_stop false, the program must be restarted.

The gate sends OWNER_REQUEST and nothing else.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Collection
from contextvars import ContextVar
from functools import wraps
from typing import TYPE_CHECKING, Any

from ..robot_app.text import MAX_REASON_CHARS, printable
from .link_client import CommandTimeout, LinkLost, Pinned, StaleState

if TYPE_CHECKING:
    from ..protocol.messages import RobotState
    from .link_client import AgentLink

_log = logging.getLogger("isaaccapture.examples.agentic_remote.task_gate")

# The experiment's read-only tools (READ_ONLY_TOOLS in its thor_runtime.py,
# which cannot be imported without its motor-side packages): they never take
# the arm, even where APAI lists one as a motion skill.
READ_ONLY_TOOLS = frozenset(
    {
        "get_observation",
        "describe_scene",
        "list_objects",
        "count_objects",
        "annotated_view",
        "probe_point",
        "locate_pixel",
        "analyze_scene",
        "localize_object",
        "snapshot_scene",
        "recall_memory",
        "recall_step",
        "list_arms",
        "task_done",
        "halt_motion",
        "emergency_stop",
    }
)
# The return to rest is the robot app's, never a skill of the agent.
REST_SKILL = "move_to_rest"
REST_REFUSAL = (
    "the return to rest belongs to the robot app: type rest at the Thor "
    "terminal or run the console command rest"
)
RESTART = (
    "a skill stop latched the workstation checker: restart the agent program; "
    "it takes no new task"
)
# What the typed yes of the next task covers, in its question.
SKILL_STOP_NOTE = "the last task ended in a skill stop; it is cleared when you confirm"
SKILL_STOP_TOLD = (
    "the task ended in a skill stop: the next task clears it after its typed yes"
)
SKILL_STOP_KEPT = "the skill stop is not cleared; type the task again"
NO_TASK = "motion needs a task typed at the agent terminal"
CONSENT = "consent"
OWNERSHIP = "ownership"
_COUNT_WORDS = {1: "one step", 2: "two steps", 3: "three steps"}

# What to do about a refusal of the robot app, by a piece of its reason.
_REMEDIES = (
    ("session id does not match", "the robot app was restarted: restart the agent"),
    ("the motors are off", "restart the robot app at the Thor terminal"),
    ("stop number", "the robot's state changed meanwhile: type the task again"),
    ("owner number", "the robot's state changed meanwhile: type the task again"),
    ("control owner is human", "takeover not finished: give the arm back with right A"),
    ("control owner is", "wait until nobody owns the arm, then type the task again"),
    ("fault-stop", "fault-stop: run the console command reset"),
    ("busy", "wait for the robot app to finish, then type the task again"),
    ("the measured pose could not be read", "look at the Thor terminal"),
    ("type rest first", "arm outside the agent's limits: run the console command rest"),
    ("a STOP or an owner change", "type the task again"),
    ("robot connection lost", "wait for the robot connection, then type the task"),
)


class _Task:
    """What the gate knows of the task that runs now."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.owner = False  # the agent became control owner in this task
        self.refusal: tuple[str, str] | None = None  # (reason, kind) of the start
        self.failed = 0  # results with ok false in a row, while owner
        self.last: Any = None  # the last tool result


class TaskGate:
    """One agent program's gate between APAI's tasks and the robot app.

    ask(prompt) returns what the operator typed at this terminal and may raise
    when nobody can be asked. say(text) prints one line. bind_arm() hands in
    the arm adapter and the workstation checker once APAI has built them.
    """

    def __init__(
        self,
        link: AgentLink,
        arm: Any = None,
        settings: Any = None,
        *,
        ask: Callable[[str], str],
        say: Callable[[str], None],
        motion_skills: Collection[str],
        read_only_tools: Collection[str] = READ_ONLY_TOOLS,
        harness: Any = None,
    ) -> None:
        self._link = link
        self._adapter = arm
        self._harness = harness
        task = getattr(settings, "task", settings)
        self._timed_return = bool(getattr(task, "timed_return", False))
        self._failed_limit = int(getattr(task, "failed_steps_limit", 3))
        self._speed = float(getattr(task, "max_joint_speed", 0.1))
        self._end_wait_s = float(getattr(task, "end_wait_s", 5.0))
        self._owner_wait_s = float(getattr(task, "owner_wait_s", 2.0))
        self._clear = bool(getattr(task, "clear_skill_stop", True))
        self._ask, self._say = ask, say
        self._motion = frozenset(motion_skills) - frozenset(read_only_tools)
        self._current: ContextVar[_Task | None] = ContextVar(
            "agentic_remote_task", default=None
        )
        self._restart = False  # only a restart lets this program move the arm
        self._owning = False  # a task of this gate owns the arm
        # The command id of an OWNER_REQUEST "agent" whose RESULT did not come
        # in time ("" when the id is not known); None when there is none.
        self._unanswered: str | None = None
        # The owner number of the last ownership a task of this gate ended.
        self._ended_number: int | None = None
        self._tried: Pinned | None = None  # what the last give-back quoted
        # The operator was told to restart the agent program.
        self.restart_told = False
        # An OWNER_REQUEST that ended a task got no RESULT: the agent may still own.
        self.end_unanswered = False
        # Tasks that ended without success; a task the operator cancelled is none.
        self.failed_tasks = 0
        # (request, status, reason) of every OWNER_REQUEST answered so far.
        self.requests: list[tuple[str, str, str]] = []

    def bind_arm(self, adapter: Any, harness: Any) -> None:
        """Take the arm adapter and the workstation checker APAI built."""
        self._adapter, self._harness = adapter, harness

    @property
    def skill_stop(self) -> bool:
        """The workstation checker is latched now."""
        return getattr(self._harness, "estopped", False) is True

    @property
    def restart_needed(self) -> bool:
        """A latch of the workstation checker that no task of this run clears.

        With task.clear_skill_stop false that is every latch; else one that
        stayed after the gate's reset. It stays for the run.
        """
        if not self._clear and self.skill_stop:
            self._restart = True
        return self._restart

    def tell_restart(self) -> bool:
        """Say that the program must be restarted, if it must; whether it must."""
        if not self.restart_needed:
            return False
        self.restart_told = True
        self._say(RESTART)
        return True

    # The two wrappers

    def bind_run_task(self, original: Callable[..., Any]) -> Callable[..., Any]:
        """AgentOrchestrator.run_task with the task's start and end around it."""

        @wraps(original)
        def run_task(task: str, *args: Any, **kwargs: Any) -> Any:
            if self.tell_restart():
                self.failed_tasks += 1
                return _refused_report(str(task), RESTART)
            state = _Task(str(task))
            token = self._current.set(state)
            report, error, interrupted = None, None, False
            try:
                report = original(task, *args, **kwargs)
                return report
            except Exception as raised:
                error = raised
                raise
            except BaseException:
                interrupted = True
                raise
            finally:
                try:
                    self._end(state, report, error, interrupted)
                finally:
                    cancelled = interrupted or getattr(report, "cancelled", False)
                    if not cancelled and not getattr(report, "success", False):
                        self.failed_tasks += 1
                    self._current.reset(token)

        return run_task

    def execute(
        self, original: Callable[[str, dict], dict], name: str, args: dict
    ) -> dict:
        """SkillRuntime.execute: the first motion skill of a task takes the arm."""
        if name == REST_SKILL:
            self._say(f"{name} refused: {REST_REFUSAL}")
            return _refusal(REST_REFUSAL, OWNERSHIP)
        task = self._current.get()
        if name in self._motion:
            if task is None:
                return _refusal(NO_TASK, CONSENT)
            if task.refusal is None and not task.owner:
                task.refusal = self._start(task, name, args)
            if task.refusal is not None:
                return _refusal(*task.refusal)
        result = self._run(original, name, args)
        if task is not None:
            result = self._count(task, result)
            task.last = result
        return result

    def _run(self, original: Callable[[str, dict], dict], name: str, args: dict) -> Any:
        scope = getattr(self._adapter, "skill_scope", None)
        if scope is None:
            return original(name, args)
        with scope():
            return original(name, args)

    def _count(self, task: _Task, result: Any) -> Any:
        """The three-failures rule, from the moment the agent owns the arm."""
        if not task.owner or not isinstance(result, dict):
            return result
        if result.get("ok") is True:
            task.failed = 0
        elif result.get("ok") is False:
            task.failed += 1
            stuck = result.get("outcome") == "stuck" or result.get("motion_blocked")
            if task.failed >= self._failed_limit and not stuck:
                count = _COUNT_WORDS.get(self._failed_limit, f"{task.failed} steps")
                last = result.get("error") or "no reason given"
                result = {
                    **result,
                    "ok": False,
                    "outcome": "stuck",
                    "ask": f"{count} failed in a row: {last}",
                }
        return result

    # The start of a task

    def _start(self, task: _Task, name: str, args: dict) -> tuple[str, str] | None:
        """Steps 1 to 3; why the start is refused, or None once the agent owns."""
        link = self._link
        if self.restart_needed:
            self.restart_told = True
            return self._refused(RESTART, OWNERSHIP)
        if self._adapter is None:
            return self._refused("no arm is built yet", OWNERSHIP)
        self.return_late_ownership(self._owner_wait_s)
        try:
            state = link.state(1.0)
        except (LinkLost, StaleState) as error:
            return self._refused(f"robot connection lost: {error}", OWNERSHIP)
        self._say(_status_line(state))
        reason = _not_startable(state)
        if reason is not None:
            return self._refused(reason, OWNERSHIP)
        # The typed yes covers the latch the question names, and no later stop.
        told, mark = self.skill_stop, _stop_count(self._harness)
        prompt = (
            (f"Note: {SKILL_STOP_NOTE}. " if told else "")
            + f"Authorize motion for this task: {json.dumps(task.text)}. "
            f"First tool {name}: {_short(args)}. This covers the later motion "
            "tools and retries of this task, without another prompt. Joint "
            f"speed at most {self._speed:g} rad/s. The checks of the robot app "
            "and of the workstation stay active. Keep the workspace clear; "
            "Ctrl+C at this terminal is STOP. Type yes to proceed: "
        )
        try:
            answer = self._ask(prompt)
        except Exception as error:
            return self._refused(f"no typed yes: {error}", CONSENT)
        if str(answer).strip().lower() != "yes":
            return self._refused("the operator did not type yes", CONSENT)
        quote = Pinned(state.session_id, state.owner_number, state.stop_number)
        try:
            result = link.request_owner(
                "agent",
                reason="new task",
                timed_return=self._timed_return,
                quote=quote,
                timeout_s=self._end_wait_s,
            )
        except CommandTimeout as error:
            self._unanswered = error.command_id or ""
            self.return_late_ownership()
            return self._refused(f"the robot did not answer: {error}", OWNERSHIP)
        except (LinkLost, StaleState) as error:
            return self._refused(f"robot connection lost: {error}", OWNERSHIP)
        self.requests.append(("agent", result.status, result.reason))
        if result.status != "completed":
            reason = f"task refused by the robot: {result.reason}"
            remedy = _remedy(result.reason)
            return self._refused(
                f"{reason} ({remedy})" if remedy else reason, OWNERSHIP
            )
        number = (result.detail or {}).get("owner_number")
        shown = self._await_owner(number)
        if shown is None:
            self._unanswered = ""  # granted, and not shown yet
            self.return_late_ownership()
            return self._refused(
                "no ROBOT_STATE showed the agent as control owner", OWNERSHIP
            )
        pinned = Pinned(shown.session_id, shown.owner_number, shown.stop_number)
        if self.skill_stop:
            reason = self._clear_skill_stop(pinned, told, mark)
            if reason is not None:
                self._give_back(pinned, reason)
                return self._refused(reason, OWNERSHIP)
        link.pin(pinned)
        self._adapter.open_task(pinned)
        task.owner = self._owning = True
        self._say(f"the agent owns the arm (owner number {pinned.owner_number})")
        return None

    def _clear_skill_stop(self, pinned: Pinned, told: bool, mark: Any) -> str | None:
        """Clear the workstation checker's latch; why it stays, or None.

        Called once the operator typed yes and the Thor granted the arm. The
        latch stays when the question did not name it, when the checker was
        stopped again since the question, and unless the newest ROBOT_STATE
        shows the agent as owner with the granted numbers and stop level none.
        Only the workstation's latch is reset: nothing is sent to the Thor.
        """
        harness = self._harness
        if not told or _stop_count(harness) != mark:
            return f"the workstation checker was stopped again: {SKILL_STOP_KEPT}"
        try:
            state = self._link.state()
        except (LinkLost, StaleState) as error:
            return f"robot connection lost: {error}"
        shown = (state.session_id, state.owner_number, state.stop_number)
        if (
            state.owner != "agent"
            or state.stop_level != "none"
            or Pinned(*shown) != pinned
        ):
            return (
                f"control owner {state.owner}, stop level {state.stop_level} "
                f"after the grant: {SKILL_STOP_KEPT}"
            )
        try:
            harness.reset_estop()
        except Exception:
            _log.exception("the workstation checker's latch was not reset")
        if getattr(harness, "estopped", True) is not False:
            self._restart = self.restart_told = True
            return RESTART
        self._say("the skill stop of the last task is cleared")
        return None

    def _give_back(self, quote: Pinned, reason: str) -> None:
        """Return an ownership no task was opened on; waits for the RESULT."""
        self._ended_number = quote.owner_number
        try:
            result = self._link.request_owner(
                "nobody",
                reason=printable(reason, MAX_REASON_CHARS),
                quote=quote,
                timeout_s=self._end_wait_s,
            )
        except (LinkLost, StaleState, CommandTimeout) as error:
            self.end_unanswered = True
            self._say(
                f"the arm could not be given back: {error}; the robot app holds "
                "the arm, and the agent may still own it"
            )
            return
        self.requests.append(("nobody", result.status, result.reason))
        if result.status != "completed":
            self._say(f"the arm could not be given back: {result.reason}")
            return
        self._say("the arm was given back: nobody owns it")
        self._await_released(quote.owner_number)

    def _await_owner(self, number: Any) -> RobotState | None:
        """The ROBOT_STATE that shows the agent as owner with this number."""
        link = self._link
        end = time.monotonic() + self._owner_wait_s
        while True:
            try:
                state = link.state()
            except (LinkLost, StaleState):
                return None
            if state.owner == "agent" and number in (None, state.owner_number):
                return state
            left = end - time.monotonic()
            if left <= 0:
                return None
            link.wait_change(min(left, 0.05))

    def return_late_ownership(self, wait_s: float = 0.0) -> None:
        """Give the arm back when the agent owns it and no task of this gate does.

        Called before a task takes the arm, at the task prompt and at the end
        of the program. While an OWNER_REQUEST "agent" of this gate has no
        RESULT, it waits up to wait_s for the grant; that wait is spent once.
        """
        if self._owning:
            return
        link = self._link
        end = time.monotonic() + max(0.0, wait_s)
        try:
            while True:
                state = link.state()
                if state.owner == "agent" and state.owner_number != self._ended_number:
                    break
                if self._unanswered is None or self._late_refusal():
                    return
                left = end - time.monotonic()
                if left <= 0:
                    if wait_s > 0:
                        self._forget_unanswered()
                    return
                link.wait_change(min(left, 0.05))
        except (LinkLost, StaleState):
            return
        quote = Pinned(state.session_id, state.owner_number, state.stop_number)
        if quote == self._tried:
            return
        self._tried = quote
        try:
            result = link.request_owner(
                "nobody",
                reason="the ownership arrived late",
                quote=quote,
                timeout_s=self._end_wait_s,
            )
        except (LinkLost, StaleState, CommandTimeout) as error:
            self._say(f"the arm could not be given back: {error}")
            return
        self.requests.append(("nobody", result.status, result.reason))
        if result.status != "completed":
            self._say(f"the arm could not be given back: {result.reason}")
            return
        self._forget_unanswered()
        self._ended_number = quote.owner_number
        self._say("the ownership arrived late: the arm was given back")
        self._await_released(quote.owner_number)

    def _late_refusal(self) -> bool:
        """Whether the RESULT that came late says the request was not granted."""
        result = self._link.final_result(self._unanswered) if self._unanswered else None
        if result is None or result.status == "completed":
            return False
        self._forget_unanswered()
        return True

    def _forget_unanswered(self) -> None:
        if self._unanswered:
            self._link.forget(self._unanswered)
        self._unanswered = None

    def _await_released(self, number: int) -> None:
        """Wait for a ROBOT_STATE in which that ownership has ended."""
        link = self._link
        end = time.monotonic() + self._owner_wait_s
        while True:
            try:
                state = link.state()
            except (LinkLost, StaleState):
                return
            if state.owner != "agent" or state.owner_number != number:
                return
            left = end - time.monotonic()
            if left <= 0:
                return
            link.wait_change(min(left, 0.05))

    def _refused(self, reason: str, kind: str) -> tuple[str, str]:
        reason = printable(reason, MAX_REASON_CHARS)
        self._say(f"motion refused: {reason}; no command sent")
        return reason, kind

    # The end of a task

    def _end(
        self, task: _Task, report: Any, error: Exception | None, interrupted: bool
    ) -> None:
        """Give the arm back or ask for help, then close the task on the arm."""
        if not task.owner:
            return
        pinned = self._link.pinned
        if pinned is not None:
            self._ended_number = pinned.owner_number
        try:
            request, reason = _ending(task, report, error, interrupted)
            if request is None:
                self._say(
                    "the task was cancelled: the arm holds; a STOP follows from "
                    "this terminal"
                )
            else:
                self._send_end(request, reason)
        finally:
            try:
                self._adapter.close_task()
            except Exception:
                _log.exception("closing the task on the arm adapter failed")
            self._owning = False
            if not self.tell_restart() and self.skill_stop:
                self._say(SKILL_STOP_TOLD)

    def _send_end(self, request: str, reason: str) -> None:
        """One OWNER_REQUEST with the pinned numbers; waits for its RESULT."""
        link = self._link
        reason = printable(reason, MAX_REASON_CHARS)  # one line, as the Thor cuts it
        quote = link.pinned
        if quote is None:
            self._say(f"{request} not sent: no ownership is pinned")
            return
        try:
            result = link.request_owner(
                request,
                reason=reason,
                timed_return=self._timed_return,
                quote=quote,
                timeout_s=self._end_wait_s,
            )
        except (LinkLost, StaleState, CommandTimeout) as error:
            self.end_unanswered = True
            self._say(
                f"{request} ({reason}) was not answered: {error}; the robot "
                "app holds the arm, and the agent may still own it"
            )
            return
        self.requests.append((request, result.status, result.reason))
        if result.status != "completed":
            self._say(
                f"{request} ({reason}) refused by the robot: {result.reason}; "
                "nothing else is sent"
            )
        elif request == "help_needed":
            wait = (result.detail or {}).get("assistance_s")
            self._say(
                f"HELP NEEDED: {reason}. The arm holds and nobody owns it"
                + (f"; assistance wait {wait:g} s" if _is_number(wait) else "")
            )
        else:
            self._say(
                "task completed: the arm holds where it is and nobody owns it; "
                "type rest at the Thor terminal or run the console command rest"
            )


def _ending(
    task: _Task, report: Any, error: Exception | None, interrupted: bool
) -> tuple[str | None, str]:
    """(the owner request that ends this task, its reason); None sends nothing."""
    if interrupted or getattr(report, "cancelled", False):
        return None, ""
    if error is not None:
        return "help_needed", f"agent error: {type(error).__name__}: {error}"
    if report is None:
        return "help_needed", "task ended without success: no report"
    if getattr(report, "success", False):
        return "nobody", "task completed"
    stuck = getattr(report, "stuck", None)
    if stuck:
        ask = stuck.get("ask") if isinstance(stuck, dict) else stuck
        return "help_needed", f"stuck: {ask}"
    last = task.last
    if isinstance(last, dict) and last.get("motion_blocked"):
        why = last.get("motion_block_reason") or last.get("error") or "motion blocked"
        return "help_needed", f"blocked: {why}"
    summary = str(getattr(report, "summary", ""))
    if summary.startswith("step budget exhausted"):
        return "help_needed", "step budget ran out"
    return "help_needed", f"task ended without success: {summary}"


def _not_startable(state: RobotState) -> str | None:
    """Why no task can take the arm in this state, without asking anybody."""
    if state.owner == "human":
        return "takeover not finished: give the arm back with right A at rest"
    if state.owner != "nobody":
        return f"control owner is {state.owner}"
    if state.stop_level == "fault":
        why = f" ({state.fault_reason})" if state.fault_reason else ""
        return f"fault-stop{why}: run the console command reset"
    if not state.torque_on:
        return "the motors are off; restart the robot app"
    return None


def _stop_count(harness: Any) -> Any:
    """APAI's count of the checker's stops and halts; a reset never lowers it."""
    return getattr(harness, "_observation_cancel_generation", None)


def _status_line(state: RobotState) -> str:
    wait = state.assistance_wait_s
    return (
        f"robot: control owner {state.owner}, stop level {state.stop_level}, "
        f"motion {state.motion}, jaws {state.gripper.load}, torque "
        f"{'on' if state.torque_on else 'off'}, assistance wait "
        f"{f'{wait:.0f} s left' if _is_number(wait) else 'none'}"
    )


def _remedy(reason: str) -> str:
    for piece, remedy in _REMEDIES:
        if piece in reason:
            return remedy
    return ""


def _refusal(reason: str, kind: str) -> dict[str, Any]:
    """What APAI gets for a motion skill the gate did not let through."""
    return {
        "ok": False,
        "error": f"{reason}; no command sent",
        "refused": kind,
        "motion_blocked": True,
        "motion_block_reason": reason,
    }


def _refused_report(task: str, reason: str) -> Any:
    """A TaskReport for a task that was not run."""
    from apai.agent.orchestrator import TaskReport

    return TaskReport(task, False, reason, 0)


def _short(args: Any) -> str:
    try:
        text = json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):
        text = repr(args)
    return printable(text, MAX_REASON_CHARS)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
