# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the publication lookup's actual shell code with fake API responses."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from repo_paths import repo_root

BASE = "https://packages.example.test/artifactory"
REPO = "python-wheels"
WHEEL = "isaaccapture-1!1.8.dev25+g11ca4eb-cp313-cp313-manylinux_2_35_x86_64.whl"
STORAGE = f"{BASE}/api/storage/{REPO}/isaaccapture/1.8/{WHEEL}"
DOWNLOAD = f"{BASE}/{REPO}/isaaccapture/1.8/{WHEEL}"


class ArtifactoryWheelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("jq"):
            raise RuntimeError("The publication workflow tests require jq")
        workflow = (repo_root() / ".github/workflows/build-ubuntu.yml").read_text()
        upload = workflow.split("    - name: Upload wheel(s) to Artifactory\n", 1)[1]
        lookup = upload[upload.index('        wheel_base="') :]
        lookup = lookup.split("\n  kitmaker:", 1)[0]
        cls.lookup = textwrap.dedent(lookup)

    def resolve(self, results, *, metadata=None, raw=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "wheels").mkdir()
            (root / "wheels" / WHEEL).touch()
            output = root / "output"
            requests = root / "requests"
            environment = {
                **os.environ,
                "ARTIFACTORY_URL": BASE,
                "ARTIFACTORY_REPO": REPO,
                "ARTIFACTORY_USERNAME": "test-user",
                "ARTIFACTORY_API_KEY": "test-key",
                "SEARCH_RESPONSE": results if raw else json.dumps({"results": results}),
                "METADATA_RESPONSE": json.dumps(
                    {"downloadUri": DOWNLOAD} if metadata is None else metadata
                ),
                "CURL_LOG": str(requests),
                "GITHUB_OUTPUT": str(output),
            }
            fake = textwrap.dedent("""\
                set -euo pipefail
                wheels=(wheels/*.whl)
                curl() {
                  local url="${!#}"
                  printf '%s\\n' "$url" >> "$CURL_LOG"
                  if [[ "$url" == "${ARTIFACTORY_URL}/api/search/artifact" ]]; then
                    printf '%s\\n' "$SEARCH_RESPONSE"
                  else
                    printf '%s\\n' "$METADATA_RESPONSE"
                  fi
                }
                """)
            result = subprocess.run(
                ["bash", "-c", fake + self.lookup],
                cwd=root,
                env=environment,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
            return (
                result,
                output.read_text() if output.exists() else "",
                requests.read_text().splitlines(),
            )

    def test_wheel_and_metadata_sidecar_in_either_order(self):
        for uris in (
            (STORAGE, STORAGE + ".metadata"),
            (STORAGE + ".metadata", STORAGE),
        ):
            with self.subTest(uris=uris):
                result, output, requests = self.resolve([{"uri": uri} for uri in uris])
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(
                    output, f"wheel_paths<<EOF\nisaaccapture/1.8/{WHEEL}\nEOF\n"
                )
                self.assertEqual(requests, [f"{BASE}/api/search/artifact", STORAGE])

    def test_zero_or_multiple_exact_wheels_fail_before_metadata_request(self):
        cases = (
            [],
            [{"uri": STORAGE + ".metadata"}],
            [{"uri": STORAGE}, {"uri": STORAGE.replace("/1.8/", "/another/")}],
            [{"uri": STORAGE}, {"uri": STORAGE}],
        )
        for results in cases:
            with self.subTest(results=results):
                result, output, requests = self.resolve(results)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "Expected exactly 1 Artifactory search result", result.stdout
                )
                self.assertEqual(output, "")
                self.assertEqual(requests, [f"{BASE}/api/search/artifact"])

    def test_wrong_wheel_repository_or_origin_is_not_selected(self):
        for uri in (
            STORAGE.replace(WHEEL, "copy-" + WHEEL),
            STORAGE.replace(REPO, REPO + "-other"),
            STORAGE.replace("packages.example.test", "elsewhere.example.test"),
            STORAGE + "?name=" + WHEEL,
            STORAGE + "?redirect=/" + WHEEL,
            STORAGE + "#/" + WHEEL,
        ):
            with self.subTest(uri=uri):
                result, output, requests = self.resolve([{"uri": uri}])
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output, "")
                self.assertEqual(requests, [f"{BASE}/api/search/artifact"])

    def test_malformed_search_responses_fail_closed(self):
        for response in (
            "not JSON",
            "{}",
            json.dumps({"results": {"wheel": {"uri": STORAGE}}}),
            json.dumps({"results": [{"uri": None}]}),
        ):
            with self.subTest(response=response):
                result, output, requests = self.resolve(response, raw=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output, "")
                self.assertEqual(requests, [f"{BASE}/api/search/artifact"])

    def test_download_uri_must_still_point_into_the_configured_repository(self):
        for metadata in (
            {},
            {"downloadUri": "https://elsewhere.example.test/" + WHEEL},
        ):
            with self.subTest(metadata=metadata):
                result, output, requests = self.resolve(
                    [{"uri": STORAGE}], metadata=metadata
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(output, "")
                self.assertEqual(requests, [f"{BASE}/api/search/artifact", STORAGE])


if __name__ == "__main__":
    unittest.main()
