# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What an operator types to confirm, shared by the Thor and the workstation.

It imports only the standard library, so the robot app, the operator console
and the agent program can all use it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

# Confirms what moves nothing by itself, and a shutdown with the arm at rest.
YES = "yes"
# Confirms a return to rest, and a shutdown that has to move the arm.
PATH_IS_CLEAR = "path is clear"
# Confirms a jaw reference, at the Thor terminal only.
JAWS_ARE_EMPTY = "jaws are empty"

# Set, each of these marks the environment of an AI agent: no confirmation is
# asked there. Equal to AGENT_ENV and AGENT_ENV_PREFIXES of APAI's arm_hold.py.
AGENT_ENV: tuple[str, ...] = (
    "AI_AGENT",
    "HERMES_AGENT",
    "HERMES_SESSION_ID",
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CODEX_SANDBOX",
    "CODEX_SANDBOX_NETWORK_DISABLED",
    "CODEX_THREAD_ID",
    "APAI_MCP_TOKEN",
    "APAI_OPENCLAW_PROFILE",
)
AGENT_ENV_PREFIXES: tuple[str, ...] = ("OPENCLAW_",)


def confirms(answer: str | None, phrase: str) -> bool:
    """Whether the typed answer is the phrase; case and spacing do not count.

    No answer (None) confirms nothing.
    """
    if answer is None:
        return False
    return " ".join(answer.lower().split()) == phrase


def agent_environment(environ: Mapping[str, str] | None = None) -> str | None:
    """The variable that marks the environment as an AI agent's, or None.

    The marks are APAI's (apai.control.arm_hold); this runs without APAI.
    """
    env = os.environ if environ is None else environ
    for name in AGENT_ENV:
        if str(env.get(name, "")).strip():
            return name
    for name, value in env.items():
        if name.startswith(AGENT_ENV_PREFIXES) and str(value).strip():
            return name
    return None
