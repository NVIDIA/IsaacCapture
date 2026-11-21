# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Open the existing WebXR client with synthetic controls enabled."""

import argparse
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import webbrowser


def client_url(value):
    parts = urlsplit(value)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username
        or parts.password
    ):
        raise ValueError("Use an HTTP(S) client URL without embedded credentials")
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["syntheticXR"] = "1"
    query.setdefault("immersiveMode", "vr")
    return urlunsplit(parts._replace(query=urlencode(query)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--client-url",
        required=True,
        help="URL printed by your CloudXR host; use the client built from this branch",
    )
    parser.add_argument(
        "--print-url", action="store_true", help="Print without opening a browser"
    )
    args = parser.parse_args()
    try:
        url = client_url(args.client_url)
    except ValueError as error:
        parser.error(str(error))
    print(url)
    if not args.print_url and not webbrowser.open(url):
        parser.exit(1, "Could not open a browser; open the URL above manually.\n")


if __name__ == "__main__":
    main()
