# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The accumulator model, and the envelope and payload checks every schema runs.

These classes carry no ``gate``: each schema package subclasses them to say which of its
groups a result is reported under.
"""
