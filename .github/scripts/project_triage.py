# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refresh PR age/idle fields and send bounded, public-data-only reminders."""

import argparse
import json
import logging
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOG = logging.getLogger("isaaccapture.automation.project_triage")
MARKER = "<!-- isaaccapture-triage-reminder:v1 -->"
FIELDS = {
    "Idle": "NUMBER",
    "Age": "NUMBER",
    "Ready days": "NUMBER",
    "Ready since": "DATE",
    "Triage state": "TEXT",
}
PR_QUERY = """query($owner:String!,$repo:String!,$number:Int!) {
  repository(owner:$owner,name:$repo) { pullRequest(number:$number) {
    number url state isDraft createdAt closedAt headRefOid
    mergeable mergeStateStatus reviewDecision
    commits(last:1) { nodes { commit { statusCheckRollup { state } } } }
  } }
}"""


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def iso(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def days(now, start):
    return max(0, (now - start).days)


def human(user):
    return (
        bool(user)
        and user.get("type") != "Bot"
        and not user.get("login", "").endswith("[bot]")
    )


def latest_activity(pr, comments, reviews, inline, commits):
    dates = [timestamp(pr["createdAt"])]
    for entry in comments + inline:
        if human(entry.get("user")) and MARKER not in (entry.get("body") or ""):
            dates.append(timestamp(entry["created_at"]))
    for review in reviews:
        if human(review.get("user")) and review.get("submitted_at"):
            dates.append(timestamp(review["submitted_at"]))
    for commit in commits:
        if human(commit.get("author")) or human(commit.get("committer")):
            dates.append(timestamp(commit["commit"]["committer"]["date"]))
    return max(dates)


def ready(pr):
    commits = pr.get("commits", {}).get("nodes", [])
    checks = (commits[-1]["commit"].get("statusCheckRollup") or {}) if commits else {}
    return (
        pr["state"] == "OPEN"
        and not pr["isDraft"]
        and pr["mergeable"] == "MERGEABLE"
        and pr["mergeStateStatus"] == "CLEAN"
        and pr["reviewDecision"] == "APPROVED"
        and checks.get("state") == "SUCCESS"
    )


def ready_state(pr, previous, now):
    # Daily observation cannot reconstruct earlier Ready periods or unseen transitions.
    if not ready(pr):
        return {}
    if (
        previous.get("head") == pr["headRefOid"]
        and previous.get("since")
        and previous.get("observed")
        and timedelta(0) <= now - timestamp(previous["observed"]) <= timedelta(hours=48)
        and timestamp(previous["since"]) <= now
    ):
        since = previous["since"]
    else:
        since = iso(now)
    return {"head": pr["headRefOid"], "since": since, "observed": iso(now)}


def reasons(pr, state, now):
    if pr["state"] != "OPEN" or pr["isDraft"]:
        return []
    result = []
    if state.get("since") and now - timestamp(state["since"]) >= timedelta(days=7):
        result.append(
            "This PR has been observed ready to merge for at least 7 consecutive days."
        )
    if now - timestamp(pr["createdAt"]) > timedelta(days=14):
        result.append("This PR has been open for more than 14 consecutive days.")
    return result


def due(comments, now):
    sent = [
        timestamp(c["created_at"]) for c in comments if MARKER in (c.get("body") or "")
    ]
    return not sent or now - max(sent) >= timedelta(days=7)


def recipients(detail, reviews, is_ready):
    people = {u["login"] for u in detail.get("assignees", []) if human(u)}
    if not people:
        people.update(
            u["login"] for u in detail.get("requested_reviewers", []) if human(u)
        )
    if not people and is_ready:
        people.update(
            r["user"]["login"]
            for r in reviews
            if r["state"] == "APPROVED" and human(r.get("user"))
        )
    if not people and human(detail.get("user")):
        people.add(detail["user"]["login"])
    return sorted(people)


class GitHub:
    def __init__(self, token):
        self.env = dict(os.environ, GH_TOKEN=token)

    def call(self, endpoint, payload=None, method=None):
        args = ["gh", "api", endpoint]
        if method:
            args += ["--method", method]
        if payload is not None:
            args += ["--input", "-"]
        proc = subprocess.run(
            args,
            input=json.dumps(payload) if payload is not None else None,
            text=True,
            capture_output=True,
            env=self.env,
            check=False,
            timeout=90,
        )
        if proc.returncode:
            raise RuntimeError(
                f"GitHub request failed: {endpoint}: {proc.stderr.strip()}"
            )
        data = json.loads(proc.stdout) if proc.stdout.strip() else None
        if isinstance(data, dict) and data.get("errors"):
            raise RuntimeError(f"GraphQL request failed: {data['errors']}")
        return data

    def graphql(self, query, **variables):
        return self.call("graphql", {"query": query, "variables": variables})["data"]

    def pages(self, endpoint):
        result = []
        page = 1
        while True:
            rows = self.call(f"{endpoint}?per_page=100&page={page}")
            result.extend(rows)
            if len(rows) < 100:
                return result
            page += 1


def project_data(api, org, number):
    meta = api.graphql(
        """query($org:String!,$number:Int!) {
      organization(login:$org) { projectV2(number:$number) {
        id viewerCanUpdate fields(first:100) { nodes {
          ... on ProjectV2Field { id name dataType }
        } }
      } }
    }""",
        org=org,
        number=number,
    )["organization"]["projectV2"]
    if not meta or not meta["viewerCanUpdate"]:
        raise RuntimeError("Project is unavailable or not writable")
    items = []
    cursor = None
    while True:
        page = api.graphql(
            """query($id:ID!,$cursor:String) {
          node(id:$id) { ... on ProjectV2 { items(first:100,after:$cursor) {
            pageInfo { hasNextPage endCursor } nodes {
              id isArchived content { ... on PullRequest { number state repository { nameWithOwner } } }
              fieldValues(first:100) { nodes {
                ... on ProjectV2ItemFieldNumberValue { number field { ... on ProjectV2Field { name } } }
                ... on ProjectV2ItemFieldDateValue { date field { ... on ProjectV2Field { name } } }
                ... on ProjectV2ItemFieldTextValue { text field { ... on ProjectV2Field { name } } }
              } }
            }
          } } }
        }""",
            id=meta["id"],
            cursor=cursor,
        )["node"]["items"]
        items.extend(page["nodes"])
        if not page["pageInfo"]["hasNextPage"]:
            return meta, items
        cursor = page["pageInfo"]["endCursor"]


def ensure_fields(api, project, apply):
    fields = {f["name"]: f for f in project["fields"]["nodes"] if f}
    for name, kind in FIELDS.items():
        if name in fields:
            if fields[name]["dataType"] != kind:
                raise RuntimeError(f"Unexpected type for {name}")
        elif apply:
            field = api.graphql(
                # Creation accepts CustomFieldType; FieldType also includes built-ins.
                """mutation($project:ID!,$name:String!,$type:ProjectV2CustomFieldType!) {
              createProjectV2Field(input:{projectId:$project,name:$name,dataType:$type}) {
                projectV2Field { ... on ProjectV2Field { id name dataType } }
              }
            }""",
                project=project["id"],
                name=name,
                type=kind,
            )["createProjectV2Field"]["projectV2Field"]
            fields[name] = field
    return fields


def update_field(api, project_id, item_id, field, value):
    if value is None:
        api.graphql(
            """mutation($project:ID!,$item:ID!,$field:ID!) {
          clearProjectV2ItemFieldValue(input:{projectId:$project,itemId:$item,fieldId:$field}) {
            projectV2Item { id }
          }
        }""",
            project=project_id,
            item=item_id,
            field=field["id"],
        )
    else:
        key = {"NUMBER": "number", "DATE": "date", "TEXT": "text"}[field["dataType"]]
        api.graphql(
            """mutation($project:ID!,$item:ID!,$field:ID!,$value:ProjectV2FieldValue!) {
          updateProjectV2ItemFieldValue(input:{projectId:$project,itemId:$item,fieldId:$field,value:$value}) {
            projectV2Item { id }
          }
        }""",
            project=project_id,
            item=item_id,
            field=field["id"],
            value={key: value},
        )


def show_age(api, project, fields):
    views = api.graphql(
        """query($id:ID!) { node(id:$id) { ... on ProjectV2 {
      views(first:100) { nodes { id layout configuration { visibleFields(first:100) { nodes {
        ... on ProjectV2Field { id } ... on ProjectV2SingleSelectField { id }
        ... on ProjectV2IterationField { id }
      } } } } }
    } } }""",
        id=project["id"],
    )["node"]["views"]["nodes"]
    for view in views:
        # view.fields uses project-wide order, not the user's configured column order.
        visible = [f["id"] for f in view["configuration"]["visibleFields"]["nodes"]]
        idle, age = fields["Idle"]["id"], fields["Age"]["id"]
        if view["layout"] != "TABLE_LAYOUT" or idle not in visible or age in visible:
            continue
        visible.insert(visible.index(idle) + 1, age)
        api.graphql(
            """mutation($view:ID!,$fields:[ID!]) {
          updateProjectV2View(input:{viewId:$view,configuration:{visibleFieldIds:$fields}}) {
            projectV2View { id }
          }
        }""",
            view=view["id"],
            fields=visible,
        )


def snapshot(api, owner, repo, number):
    return api.graphql(PR_QUERY, owner=owner, repo=repo, number=number)["repository"][
        "pullRequest"
    ]


def process(project_api, repo_api, meta, fields, item, repo, now, apply, notify):
    owner, name = repo.split("/")
    number = item["content"]["number"]
    pr = snapshot(repo_api, owner, name, number)
    if pr["state"] != "OPEN":
        return None
    root = f"repos/{repo}"
    comments = repo_api.pages(f"{root}/issues/{number}/comments")
    reviews = repo_api.pages(f"{root}/pulls/{number}/reviews")
    inline = repo_api.pages(f"{root}/pulls/{number}/comments")
    commits = repo_api.pages(f"{root}/pulls/{number}/commits")
    existing = {}
    for value in item["fieldValues"]["nodes"]:
        if value.get("field", {}).get("name") in FIELDS:
            existing[value["field"]["name"]] = next(
                value[k] for k in ("number", "date", "text") if k in value
            )
    previous = json.loads(existing.get("Triage state") or "{}")
    state = ready_state(pr, previous, now)
    idle = days(now, latest_activity(pr, comments, reviews, inline, commits))
    updates = {
        "Idle": idle,
        "Age": days(now, timestamp(pr["createdAt"])),
        "Ready days": days(now, timestamp(state["since"])) if state else None,
        "Ready since": state["since"][:10] if state else None,
        "Triage state": json.dumps(state, sort_keys=True) if state else None,
    }
    why = reasons(pr, state, now)
    reminder = bool(why and due(comments, now))
    if apply:
        for field, value in updates.items():
            if existing.get(field) != value:
                update_field(project_api, meta["id"], item["id"], fields[field], value)
    if notify and reminder:
        # Recheck immediately before sending; Project snapshots can outlive a merge or draft toggle.
        fresh = snapshot(repo_api, owner, name, number)
        fresh_state = (
            state if fresh["headRefOid"] == pr["headRefOid"] and ready(fresh) else {}
        )
        why = reasons(fresh, fresh_state, now)
        fresh_comments = repo_api.pages(f"{root}/issues/{number}/comments")
        if why and due(fresh_comments, now):
            detail = repo_api.call(f"{root}/pulls/{number}")
            if detail["state"] == "open" and not detail["draft"]:
                mentions = " ".join(
                    "@" + login for login in recipients(detail, reviews, ready(fresh))
                )
                body = (
                    f"{MARKER}\nAutomated PR follow-up {mentions}\n\n"
                    + "\n".join("- " + reason for reason in why)
                    + "\n\nPlease merge if ready, or share the blocker and next step. "
                    "This reminder repeats at most once every 7 days while eligible."
                )
                repo_api.call(f"{root}/issues/{number}/comments", {"body": body})
    return {
        "number": number,
        "age": updates["Age"],
        "idle": idle,
        "draft": pr["isDraft"],
        "ready_days": updates["Ready days"],
        "reminder_due": reminder,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--org", default="NVIDIA")
    parser.add_argument("--project", type=int, default=322)
    parser.add_argument("--repo", default="NVIDIA/IsaacCapture")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--notify", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.notify and not args.apply:
        parser.error("--notify requires --apply")
    project_api = GitHub(os.environ["PROJECT_TOKEN"])
    repo_api = GitHub(os.environ["REPO_TOKEN"])
    now = datetime.now(timezone.utc)
    meta, items = project_data(project_api, args.org, args.project)
    fields = ensure_fields(project_api, meta, args.apply)
    report, failures = [], []
    for item in items:
        content = item.get("content") or {}
        if (
            item["isArchived"]
            or content.get("state") != "OPEN"
            or content.get("repository", {}).get("nameWithOwner") != args.repo
        ):
            continue
        try:
            row = process(
                project_api,
                repo_api,
                meta,
                fields,
                item,
                args.repo,
                now,
                args.apply,
                args.notify,
            )
            if row:
                report.append(row)
        except (RuntimeError, ValueError, KeyError) as error:
            LOG.error("PR #%s: %s", content["number"], error)
            failures.append(content["number"])
    if args.apply:
        show_age(project_api, meta, fields)
    args.report.write_text(
        json.dumps(
            {
                "checked_at": iso(now),
                "applied": args.apply,
                "notifications_enabled": args.notify,
                "items": report,
                "failures": failures,
            },
            indent=2,
        )
        + "\n"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
