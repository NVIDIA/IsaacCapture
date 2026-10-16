# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select a docs artifact using GitHub metadata, never artifact-supplied identity."""

import json
import os
import subprocess
from pathlib import Path

CANONICAL_REPO_ID = 1092046207


def api(endpoint, *, paginate=False):
    command = ["gh", "api", endpoint]
    if paginate:
        command += ["--paginate", "--slurp"]
    return json.loads(subprocess.check_output(command, text=True))


def matches(pr, run):
    return (
        pr["state"] == "open"
        and pr["base"]["repo"]["id"] == CANONICAL_REPO_ID
        and pr["head"]["sha"] == run["head_sha"]
        and pr["head"]["ref"] == run["head_branch"]
        and (pr["head"]["repo"] or {}).get("id")
        == (run["head_repository"] or {}).get("id")
    )


def select_preview(event_name, event, request=api):
    repo = event["repository"]
    if repo["id"] != CANONICAL_REPO_ID:
        return None
    prefix = f"repos/{repo['full_name']}"
    automatic = event_name == "workflow_run"
    if automatic:
        run = request(f"{prefix}/actions/runs/{event['workflow_run']['id']}")
    elif event_name == "issue_comment":
        comment = event["comment"]
        if not event["issue"].get("pull_request") or not (
            comment["body"] == "/preview-docs"
            or comment["body"].startswith("/preview-docs ")
        ):
            return None
        permission = request(
            f"{prefix}/collaborators/{comment['user']['login']}/permission"
        )["permission"]
        if permission not in {"admin", "maintain", "write"}:
            raise PermissionError("/preview-docs requires write access")
        pr = request(f"{prefix}/pulls/{event['issue']['number']}")
        if pr["state"] != "open":
            return None
        runs = request(
            f"{prefix}/actions/workflows/docs-build-and-deploy.yaml/runs?event=pull_request"
            f"&status=success&head_sha={pr['head']['sha']}&per_page=100"
        )["workflow_runs"]
        run = next((run for run in runs if matches(pr, run)), None)
        if run is None:
            raise ValueError(
                "No successful docs build for the current PR head; retry after it finishes"
            )
    else:
        return None

    if (
        run["event"] != "pull_request"
        or run["conclusion"] != "success"
        or run["status"] != "completed"
        or run["repository"]["id"] != CANONICAL_REPO_ID
    ):
        return None
    workflow = request(f"{prefix}/actions/workflows/docs-build-and-deploy.yaml")
    if run["workflow_id"] != workflow["id"]:
        return None

    if automatic:
        # Fork runs can have an empty pull_requests array; use GitHub's commit association.
        candidates = [
            pr
            for page in request(
                f"{prefix}/commits/{run['head_sha']}/pulls?per_page=100", paginate=True
            )
            for pr in page
            if matches(pr, run)
        ]
        if len(candidates) != 1:
            return None
        pr = request(f"{prefix}/pulls/{candidates[0]['number']}")
        if not matches(pr, run):
            return None
        files = [
            file
            for page in request(
                f"{prefix}/pulls/{pr['number']}/files?per_page=100", paginate=True
            )
            for file in page
        ]
        # The files API is capped at 3,000 entries; don't infer eligibility from a partial diff.
        if len(files) != pr["changed_files"] or not any(
            file["filename"].startswith("docs/")
            or file.get("previous_filename", "").startswith("docs/")
            for file in files
        ):
            return None

    artifacts = [
        artifact
        for page in request(
            f"{prefix}/actions/runs/{run['id']}/artifacts?per_page=100", paginate=True
        )
        for artifact in page["artifacts"]
        if not artifact["expired"]
    ]
    docs = [artifact for artifact in artifacts if artifact["name"] == "docs-html"]
    webapp = [artifact for artifact in artifacts if artifact["name"] == "webapp"]
    if len(docs) != 1 or len(webapp) > 1:
        return None
    return {
        "number": pr["number"],
        "head_sha": run["head_sha"],
        "head_repo_id": run["head_repository"]["id"],
        "run_id": run["id"],
        "docs_id": docs[0]["id"],
        "webapp_id": webapp[0]["id"] if webapp else "",
    }


if __name__ == "__main__":
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    selected = select_preview(os.environ["GITHUB_EVENT_NAME"], event)
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        print(f"publish={str(selected is not None).lower()}", file=output)
        for key, value in (selected or {}).items():
            print(f"{key}={value}", file=output)
    if selected is None:
        print("No eligible preview for this event.")
