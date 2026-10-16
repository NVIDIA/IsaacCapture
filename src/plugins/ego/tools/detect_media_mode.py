# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from mcap.reader import make_reader

topics = set()
with open(sys.argv[1], "rb") as stream:
    for _, channel, _ in make_reader(stream).iter_messages(log_time_order=False):
        topics.add(channel.topic)
print(
    "embedded"
    if any(topic.startswith("ego_media/") for topic in topics)
    else "metadata-only"
)
