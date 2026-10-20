# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks of a SkeletonBinding against the bytes of the asset it names."""

import base64
import hashlib
import xml.etree.ElementTree as ET

from ._schema import SKELETON_PROFILE_URDF_LINK, SkeletonBinding

_NI_SHA256 = "ni:///sha-256;"
_URDF_LINK_V1 = (SKELETON_PROFILE_URDF_LINK, 1)


def asset_digest(data: bytes) -> str:
    """RFC 6920 name of ``data``: ``ni:///sha-256;`` and its unpadded base64url SHA-256."""
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return _NI_SHA256 + digest.decode("ascii")


def check_skeleton_asset(binding: SkeletonBinding, data: bytes) -> str:
    """Check that ``data`` is the asset ``binding`` names, and that its segments are in it.

    Segment names are checked for ``urdf.link`` v1, where each must be a ``<link>`` of the
    URDF. Returns "" when valid, otherwise what is wrong.
    """
    asset = binding.asset
    if asset is None:
        return "binding has no asset"
    if asset_digest(data) != asset.digest:
        return "asset bytes do not match asset.digest"

    profile = binding.profile
    if profile is None or (profile.id, profile.version) != _URDF_LINK_V1:
        return ""
    try:
        robot = ET.fromstring(data)
    except ET.ParseError as error:
        return f"asset is not XML: {error}"
    if robot.tag != "robot":
        return f"asset root is <{robot.tag}>, not <robot>"
    links = {link.get("name") for link in robot.findall("link")}
    for name in binding.segment_names:
        if name not in links:
            return f"'{name}' is not a link of the URDF"
    return ""
