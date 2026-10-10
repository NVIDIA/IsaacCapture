<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PR triage metrics and reminders

`project-triage.yml` runs daily at 16:43 UTC (09:43 PDT / 08:43 PST), after
merging to the default branch. GitHub schedules may be delayed. Manual runs
default to a read-only preview; enable `apply` to update fields and `notify`
to send comments. The workflow never merges or closes PRs.

## Fields

- **Idle**: full 24-hour periods since the latest human comment, submitted
  review, inline review comment, or human-linked commit's committer timestamp.
  Falls back to PR creation. Bot activity, CI and these reminder comments do
  not reset it. Editing an old comment does not reset it. Historical Git
  timestamps cannot establish when a rebased commit was pushed.
- **Age**: full 24-hour periods since PR creation, independent of Idle.
- **Ready days / Ready since**: time since first observing the current head
  as non-draft, mergeable, `CLEAN`, review `APPROVED` and check rollup `SUCCESS`.
  No historical readiness is inferred on first run. Failed, pending, unknown,
  draft or new-head states reset this clock. An observation gap exceeding 48
  hours also resets it. Changes between daily observations may not be detected.
- **Triage state**: internal JSON storing the Ready head and observation times.

The script adds missing fields and places Age immediately after Idle in table
views that already show Idle. Archived items, other repositories, issues and
closed/merged PRs are skipped. Draft PRs still receive Idle and Age updates.

## Reminders

Only open, non-draft PRs receive comments:

1. Ready for at least 7 consecutive days and still unmerged.
2. Open for more than 14 consecutive days, regardless of recent activity.

These are calendar-day rules, not business-day or holiday calculations. Both
reasons share one comment and one seven-day cooldown. The comment marker makes
retries after an ambiguous send discoverable; writes are never blindly retried.
The PR is read again immediately before posting to avoid nudging a newly closed
or draft PR. No API can make that check and comment one atomic operation.

Mention PR assignees first, otherwise requested reviewers; for merge-ready PRs,
fall back to approving reviewers, then the author. All targets and comment text
come from public PR data, never private Project notes. The rules apply equally
to internal and external contributions; there is no automatic close policy.

## Authentication and verification

`PROJECT_WRITE_TOKEN` needs the `project` scope and appropriate organization
authorization. It is used only for Project GraphQL requests. The workflow's
`GITHUB_TOKEN` reads repository data and posts comments with `pull-requests: write`.
No Project credential is exposed to PR-triggered test jobs.

Run policy tests with:

```sh
python3 -m unittest discover -s tests/python/automation -v
```

For a local preview, provide `PROJECT_TOKEN` and `REPO_TOKEN` securely in the
environment, then run:

```sh
python3 .github/scripts/project_triage.py --report /tmp/triage-report.json
```

Add `--apply` for Project writes; `--notify` additionally posts due reminders.
API failures produce a failed run and never manufacture zero-valued metrics.
