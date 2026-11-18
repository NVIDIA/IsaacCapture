# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The live view's fixed values and the bounds of the launcher's arguments.

Every number is a first value; none was measured on a robot. Times are in s.
The launcher's arguments --live-fps, --live-width, --receiver-socket and
--link-socket set what a start may change; the rest is fixed here.
"""

from __future__ import annotations

# The camera. The viewer starts a camera frame bridge of its own: a second
# reader of the camera receiver, no second decoder.
# ipc.socket_path of camera_receiver_rebot.yaml.
RECEIVER_SOCKET = "/tmp/camera_receiver_avs1_cuda_ipc.sock"
# The two eyes' names in the camera receiver; the left one is shown.
STREAMS = ("head.left", "head.right")
# The most the two eyes' exposures may lie apart, in ms.
MAX_SKEW_MS = 5.0

# The live picture: the left eye, made small and packed as JPEG.
# The most stereo pairs asked for per second (--live-fps), and its bounds.
DEFAULT_FPS = 30.0
FPS_RANGE = (1.0, 60.0)
# The greatest width (--live-width), and its bounds; a narrower camera frame
# is not made wider.
DEFAULT_WIDTH_PX = 960
WIDTH_RANGE_PX = (64, 4096)
JPEG_QUALITY = 80  # 1 to 100

# VIEW_STATUS.
# One this often, also while the view is not live.
STATUS_PERIOD_S = 0.2
# The view is live only while the page acknowledged a picture, with its tab
# visible, less than this long ago.
ACK_TIMEOUT_S = 1.0
# The view is live only on a ROBOT_STATE younger than this: it gives the
# Thor's wall clock, which the picture's age is measured on.
STATE_MAX_AGE_S = 1.0

# The attachment to the robot link service.
ATTACH_TIMEOUT_S = 5.0
# Why a viewer that finds another one attached stays silent: two would send
# the Thor two different VIEW_STATUS in turn.
SECOND_VIEW = "another live view is attached"
