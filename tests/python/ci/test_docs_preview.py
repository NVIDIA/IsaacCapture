# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from repo_paths import repo_root

spec = importlib.util.spec_from_file_location(
    "docs_preview", repo_root() / ".github/scripts/docs_preview.py"
)
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


class PreviewSelectionTests(unittest.TestCase):
    def setUp(self):
        self.repo = {
            "id": preview.CANONICAL_REPO_ID,
            "full_name": "NVIDIA/IsaacCapture",
        }
        self.run = {
            "id": 100,
            "workflow_id": 10,
            "event": "pull_request",
            "status": "completed",
            "conclusion": "success",
            "repository": self.repo,
            "head_sha": "a" * 40,
            "head_branch": "docs-change",
            "head_repository": {"id": 200},
            "pull_requests": [],
        }
        self.pr = {
            "number": 42,
            "state": "open",
            "base": {"repo": self.repo},
            "head": {"sha": "a" * 40, "ref": "docs-change", "repo": {"id": 200}},
            "changed_files": 1,
        }
        self.files = [[{"filename": "docs/source/index.rst"}]]
        self.artifacts = [{"id": 300, "name": "docs-html", "expired": False}]
        self.candidates = [self.pr]
        self.permission = "write"
        self.calls = []
        self.event = {"repository": self.repo, "workflow_run": {"id": 100}}

    def request(self, endpoint, *, paginate=False):
        self.calls.append(endpoint)
        if endpoint.endswith("/permission"):
            return {"permission": self.permission}
        if "/actions/workflows/docs-build-and-deploy.yaml/runs?" in endpoint:
            return {"workflow_runs": [self.run]}
        if endpoint.endswith("/actions/workflows/docs-build-and-deploy.yaml"):
            return {"id": 10}
        if endpoint.endswith("/actions/runs/100"):
            return self.run
        if "/commits/" in endpoint:
            self.assertTrue(paginate)
            return [self.candidates]
        if endpoint.endswith("/pulls/42"):
            return self.pr
        if "/files?" in endpoint:
            self.assertTrue(paginate)
            return self.files
        if "/artifacts?" in endpoint:
            self.assertTrue(paginate)
            return [{"artifacts": self.artifacts}]
        self.fail(f"Unexpected API call: {endpoint}")

    def select(self):
        return preview.select_preview("workflow_run", self.event, self.request)

    def test_fork_without_event_pr_association_and_same_repo(self):
        for repo_id in (200, self.repo["id"]):
            with self.subTest(repo_id=repo_id):
                self.run["head_repository"]["id"] = repo_id
                self.pr["head"]["repo"]["id"] = repo_id
                self.assertEqual(self.select()["docs_id"], 300)

    def test_docs_deletion_and_rename_out_are_eligible(self):
        for file in (
            {"filename": "docs/old.rst", "status": "removed"},
            {"filename": "README.md", "previous_filename": "docs/old.rst"},
        ):
            with self.subTest(file=file):
                self.files = [[file]]
                self.assertIsNotNone(self.select())

    def test_non_docs_changes_do_not_publish(self):
        for filename in (
            "README.md",
            ".github/workflows/docs-build-and-deploy.yaml",
            "deps/cloudxr/webxr_client/a.js",
        ):
            with self.subTest(filename=filename):
                self.files = [[{"filename": filename}]]
                self.assertIsNone(self.select())

    def test_paginated_files_and_incomplete_diff(self):
        self.files = [[{"filename": "README.md"}], [{"filename": "docs/index.rst"}]]
        self.pr["changed_files"] = 2
        self.assertIsNotNone(self.select())
        self.pr["changed_files"] = 3001
        self.assertIsNone(self.select())

    def test_failed_incomplete_or_wrong_workflow_run(self):
        for key, value in (
            ("conclusion", "failure"),
            ("status", "in_progress"),
            ("event", "push"),
            ("workflow_id", 99),
        ):
            with self.subTest(key=key):
                original = self.run[key]
                self.run[key] = value
                self.assertIsNone(self.select())
                self.run[key] = original

    def test_stale_closed_or_wrong_source_pr(self):
        for target, key, value in (
            (self.pr, "state", "closed"),
            (self.pr["head"], "sha", "b" * 40),
            (self.pr["head"], "ref", "another-branch"),
            (self.pr["head"], "repo", {"id": 999}),
            (self.pr["head"], "repo", None),
        ):
            with self.subTest(key=key):
                saved = target[key]
                target[key] = value
                self.assertIsNone(self.select())
                target[key] = saved

    def test_head_changed_after_commit_association(self):
        self.candidates = [copy.deepcopy(self.pr)]
        self.pr["head"]["sha"] = "b" * 40
        self.assertIsNone(self.select())

    def test_ambiguous_or_missing_pr_association(self):
        self.candidates = []
        self.assertIsNone(self.select())
        self.candidates = [self.pr, {**self.pr, "number": 43}]
        self.assertIsNone(self.select())

    def test_missing_expired_or_duplicate_docs_artifact(self):
        artifact = self.artifacts[0]
        for artifacts in ([], [{**artifact, "expired": True}], [artifact, artifact]):
            with self.subTest(artifacts=artifacts):
                self.artifacts = artifacts
                self.assertIsNone(self.select())

    def test_optional_webapp_is_selected_from_matching_run(self):
        self.assertEqual(self.select()["webapp_id"], "")
        self.artifacts.append({"id": 301, "name": "webapp", "expired": False})
        self.assertEqual(self.select()["webapp_id"], 301)
        self.assertIn(
            "repos/NVIDIA/IsaacCapture/actions/runs/100/artifacts?per_page=100",
            self.calls,
        )

    def test_noncanonical_repository_is_rejected_without_api_calls(self):
        self.event["repository"] = {"id": 999, "full_name": "fork/repo"}
        self.assertIsNone(self.select())
        self.assertEqual(self.calls, [])

    def test_manual_retry_requires_write_access_and_can_retry_non_docs(self):
        event = {
            "repository": self.repo,
            "issue": {"number": 42, "pull_request": {"url": "pr"}},
            "comment": {"body": "/preview-docs", "user": {"login": "maintainer"}},
        }
        self.files = [[{"filename": "README.md"}]]
        self.assertIsNotNone(
            preview.select_preview("issue_comment", event, self.request)
        )
        self.permission = "read"
        with self.assertRaises(PermissionError):
            preview.select_preview("issue_comment", event, self.request)


if __name__ == "__main__":
    unittest.main()
