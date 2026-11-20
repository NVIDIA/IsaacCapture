# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Teleop Session Server: streams this machine's tracker data to a remote robot app.

Run it with ``python -m isaaccapture.teleop_session_server``; a robot app reaches it through
``TeleopSession`` in ``SessionMode.REMOTE``. Linux only.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaccapture.robot_state import RobotState


def hosted_robot_state() -> RobotState | None:
    """The robot state the Teleop Session Server running in this process receives.

    None when this process runs no server. The reader stays valid across robot apps.
    """
    # The server's loop imports the module that holds its store.
    session = sys.modules.get(f"{__name__}._session")
    store = None if session is None else session._hosted_robot_state
    if store is None:
        return None
    from isaaccapture.robot_state import RobotState

    return RobotState.from_store(store)
