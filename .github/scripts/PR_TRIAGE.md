<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PR triage metrics and reminders

The [Refresh PR triage workflow](../workflows/project-triage.yml) runs daily at
16:43 UTC from the default branch. It updates open PRs in the configured Project
and repository. Archived items, issues, closed PRs and merged PRs are skipped.
Draft PRs receive metric updates but no reminders.

## Metrics

- **Idle**: full 24-hour periods since the latest human comment, submitted
  review, inline review comment, or human-linked commit's committer timestamp.
  Falls back to PR creation. Bot activity, CI, reminder comments and edits to
  existing comments do not reset it. Commit timestamps do not identify push times.
- **Age**: full 24-hour periods since PR creation.
- **Ready days / Ready since**: time since first observing the current head
  as non-draft, mergeable, `CLEAN`, review `APPROVED` and check rollup `SUCCESS`.
  A new head, an unmet condition, or an observation gap exceeding 48 hours resets
  the clock. Daily observations cannot establish readiness before the first run
  or detect every transition between runs.
- **Triage state**: automation-managed readiness observations. Invalid values
  restart the readiness clock.

Missing fields are created automatically. Age is added after Idle in table
views that show Idle, unless Age is already visible; existing column order is
preserved.

## Reminders

Only open, non-draft PRs receive reminders after either threshold:

- Observed ready to merge for at least 7 consecutive days.
- Created more than 14 days ago, regardless of recent activity.

Thresholds use calendar days. Both reasons share one comment and a seven-day
cooldown. Eligibility is checked again before posting. The workflow never merges
or closes PRs.

Comments mention assignees, otherwise requested reviewers. Merge-ready PRs
fall back to approving reviewers, then the author. Other PRs fall back to the
author. Comments use public PR data only.

## Operation

In **Actions → Refresh PR triage → Run workflow**, select the default branch.
Manual runs preview by default; enable `apply` to update fields and additionally
enable `notify` to send reminders. Scheduled runs enable both. GitHub schedules
may be delayed. Failed API requests fail the run; completed updates remain.

The repository secret `PROJECT_WRITE_TOKEN` requires the `project` scope,
organization authorization and write access to the Project. It is used only for
Project requests. The workflow's `GITHUB_TOKEN` reads PR data and posts comments
with `pull-requests: write`. PR-triggered jobs run tests without Project credentials.

Run tests from the repository root:

```sh
python3 -m unittest discover -s tests/python/automation -v
```

For a local preview, provide `PROJECT_TOKEN` and `REPO_TOKEN` in the environment:

```sh
python3 .github/scripts/project_triage.py --report /tmp/triage-report.json
```

Add `--apply` to update fields; `--notify` also requires `--apply` and posts reminders.
The preview report contains per-PR metrics; workflow summaries show counts only.
