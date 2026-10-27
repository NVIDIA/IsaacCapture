# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Teleop Session Server: streams this machine's tracker data to a remote robot app.

Run it with ``python -m isaaccapture.teleop_session_server``; a robot app reaches it through
``TeleopSession`` in ``SessionMode.REMOTE``. Linux only.
"""
