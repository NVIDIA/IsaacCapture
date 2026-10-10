# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy and side-effect boundary tests without GitHub credentials."""

import importlib.util
import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = next(
    p
    for p in Path(__file__).resolve().parents
    if (p / ".github/scripts/project_triage.py").is_file()
)
SPEC = importlib.util.spec_from_file_location(
    "project_triage", ROOT / ".github/scripts/project_triage.py"
)
triage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(triage)
NOW = datetime(2026, 10, 10, 16, 43, tzinfo=timezone.utc)
PERSON = {"type": "User", "login": "author"}
BOT = {"type": "Bot", "login": "github-actions[bot]"}


def pr(**kwargs):
    result = dict(
        number=42,
        url="https://github.com/example/repo/pull/42",
        state="OPEN",
        isDraft=False,
        createdAt=triage.iso(NOW - timedelta(days=20)),
        headRefOid="abc",
        mergeable="MERGEABLE",
        mergeStateStatus="CLEAN",
        reviewDecision="APPROVED",
        commits={"nodes": [{"commit": {"statusCheckRollup": {"state": "SUCCESS"}}}]},
    )
    result.update(kwargs)
    return result


def comment(when, user=PERSON, body="An update"):
    return {"created_at": triage.iso(when), "user": user, "body": body}


class PolicyTests(unittest.TestCase):
    def test_age_threshold_is_strict_and_not_idle(self):
        self.assertEqual(
            triage.reasons(pr(createdAt=triage.iso(NOW - timedelta(days=14))), {}, NOW),
            [],
        )
        self.assertEqual(
            len(
                triage.reasons(
                    pr(createdAt=triage.iso(NOW - timedelta(days=14, seconds=1))),
                    {},
                    NOW,
                )
            ),
            1,
        )

    def test_ready_threshold_and_combined_reasons(self):
        state = {"since": triage.iso(NOW - timedelta(days=7))}
        self.assertEqual(len(triage.reasons(pr(), state, NOW)), 2)
        state["since"] = triage.iso(NOW - timedelta(days=7) + timedelta(seconds=1))
        self.assertEqual(len(triage.reasons(pr(), state, NOW)), 1)

    def test_closed_merged_and_draft_never_nudged(self):
        for data in [pr(state="CLOSED"), pr(state="MERGED"), pr(isDraft=True)]:
            with self.subTest(data=data):
                self.assertEqual(
                    triage.reasons(data, {"since": "2026-01-01T00:00:00Z"}, NOW), []
                )
                self.assertEqual(triage.ready_state(data, {}, NOW), {})

    def test_all_readiness_conditions_required(self):
        self.assertTrue(triage.ready(pr()))
        for update in [
            dict(mergeable="UNKNOWN"),
            dict(mergeStateStatus="BLOCKED"),
            dict(mergeStateStatus="UNSTABLE"),
            dict(reviewDecision="REVIEW_REQUIRED"),
            dict(reviewDecision="CHANGES_REQUESTED"),
            dict(commits={"nodes": []}),
            dict(
                commits={
                    "nodes": [{"commit": {"statusCheckRollup": {"state": "PENDING"}}}]
                }
            ),
        ]:
            with self.subTest(update=update):
                self.assertFalse(triage.ready(pr(**update)))

    def test_ready_clock_starts_at_observation_and_resets(self):
        state = triage.ready_state(pr(), {}, NOW)
        self.assertEqual(state["since"], triage.iso(NOW))
        next_day = NOW + timedelta(days=1)
        self.assertEqual(
            triage.ready_state(pr(), state, next_day)["since"], state["since"]
        )
        self.assertEqual(
            triage.ready_state(pr(headRefOid="new"), state, next_day)["since"],
            triage.iso(next_day),
        )
        self.assertEqual(
            triage.ready_state(pr(), state, NOW + timedelta(days=3))["since"],
            triage.iso(NOW + timedelta(days=3)),
        )

    def test_bot_and_reminder_do_not_reset_idle(self):
        old = NOW - timedelta(days=10)
        comments = [
            comment(old),
            comment(NOW, BOT),
            comment(NOW, PERSON, triage.MARKER),
        ]
        self.assertEqual(triage.latest_activity(pr(), comments, [], [], []), old)

    def test_reviews_inline_and_human_commits_count(self):
        reviews = [
            {"user": PERSON, "submitted_at": triage.iso(NOW - timedelta(days=4))}
        ]
        inline = [comment(NOW - timedelta(days=3))]
        commits = [
            {
                "author": PERSON,
                "commit": {"committer": {"date": triage.iso(NOW - timedelta(days=2))}},
            }
        ]
        self.assertEqual(
            triage.latest_activity(pr(), [], reviews, inline, commits),
            NOW - timedelta(days=2),
        )

    def test_weekly_cooldown_including_uncertain_previous_send(self):
        self.assertTrue(triage.due([], NOW))
        self.assertFalse(
            triage.due([comment(NOW - timedelta(days=6), BOT, triage.MARKER)], NOW)
        )
        self.assertTrue(
            triage.due([comment(NOW - timedelta(days=7), BOT, triage.MARKER)], NOW)
        )

    def test_responsibility_fallback(self):
        self.assertEqual(
            triage.recipients({"assignees": [PERSON]}, [], True), ["author"]
        )
        self.assertEqual(triage.recipients({"user": PERSON}, [], False), ["author"])
        self.assertEqual(
            triage.recipients(
                {"user": PERSON},
                [{"user": {"type": "User", "login": "reviewer"}, "state": "APPROVED"}],
                True,
            ),
            ["reviewer"],
        )

    def test_api_paginates_beyond_first_hundred(self):
        api = triage.GitHub("not-a-real-token")
        with patch.object(
            api, "call", side_effect=[[{}] * 100, [{"last": True}]]
        ) as call:
            self.assertEqual(len(api.pages("example")), 101)
            self.assertIn("page=2", call.call_args.args[0])

    def test_dry_run_has_no_mutations(self):
        item = {"id": "item", "content": {"number": 42}, "fieldValues": {"nodes": []}}
        api = triage.GitHub("not-a-real-token")
        with (
            patch.object(triage, "snapshot", return_value=pr()),
            patch.object(api, "pages", return_value=[]),
            patch.object(api, "call") as call,
        ):
            result = triage.process(
                api, api, {"id": "project"}, {}, item, "example/repo", NOW, False, False
            )
            self.assertTrue(result["reminder_due"])
            call.assert_not_called()

    def test_draft_transition_before_send_suppresses_comment(self):
        item = {"id": "item", "content": {"number": 42}, "fieldValues": {"nodes": []}}
        api = triage.GitHub("not-a-real-token")
        fields = {key: {} for key in triage.FIELDS}
        with (
            patch.object(triage, "snapshot", side_effect=[pr(), pr(isDraft=True)]),
            patch.object(api, "pages", return_value=[]),
            patch.object(triage, "update_field"),
            patch.object(api, "call") as call,
        ):
            triage.process(
                api,
                api,
                {"id": "project"},
                fields,
                item,
                "example/repo",
                NOW,
                True,
                True,
            )
            call.assert_not_called()

    def test_existing_reminder_prevents_send_after_ambiguous_run(self):
        item = {"id": "item", "content": {"number": 42}, "fieldValues": {"nodes": []}}
        api = triage.GitHub("not-a-real-token")
        fields = {key: {} for key in triage.FIELDS}
        with (
            patch.object(triage, "snapshot", return_value=pr()),
            patch.object(
                api,
                "pages",
                side_effect=[[], [], [], [], [comment(NOW, BOT, triage.MARKER)]],
            ),
            patch.object(triage, "update_field"),
            patch.object(api, "call") as call,
        ):
            triage.process(
                api,
                api,
                {"id": "project"},
                fields,
                item,
                "example/repo",
                NOW,
                True,
                True,
            )
            call.assert_not_called()

    def test_merge_before_last_rest_check_suppresses_send(self):
        item = {"id": "item", "content": {"number": 42}, "fieldValues": {"nodes": []}}
        api = triage.GitHub("not-a-real-token")
        fields = {key: {} for key in triage.FIELDS}
        with (
            patch.object(triage, "snapshot", return_value=pr()),
            patch.object(api, "pages", return_value=[]),
            patch.object(triage, "update_field"),
            patch.object(
                api, "call", return_value={"state": "closed", "draft": False}
            ) as call,
        ):
            triage.process(
                api,
                api,
                {"id": "project"},
                fields,
                item,
                "example/repo",
                NOW,
                True,
                True,
            )
            call.assert_called_once_with("repos/example/repo/pulls/42")

    def test_live_reminder_combines_rules_and_mentions(self):
        previous = {
            "head": "abc",
            "since": triage.iso(NOW - timedelta(days=8)),
            "observed": triage.iso(NOW - timedelta(days=1)),
        }
        item = {
            "id": "item",
            "content": {"number": 42},
            "fieldValues": {
                "nodes": [
                    {"field": {"name": "Triage state"}, "text": json.dumps(previous)}
                ]
            },
        }
        api = triage.GitHub("not-a-real-token")
        fields = {key: {} for key in triage.FIELDS}
        detail = {"state": "open", "draft": False, "assignees": [PERSON]}
        with (
            patch.object(triage, "snapshot", return_value=pr()),
            patch.object(api, "pages", return_value=[]),
            patch.object(triage, "update_field"),
            patch.object(api, "call", side_effect=[detail, {}]) as call,
        ):
            triage.process(
                api,
                api,
                {"id": "project"},
                fields,
                item,
                "example/repo",
                NOW,
                True,
                True,
            )
            body = call.call_args.args[1]["body"]
            self.assertIn("@author", body)
            self.assertIn("7 consecutive", body)
            self.assertIn("14 consecutive", body)
            self.assertIn(triage.MARKER, body)


if __name__ == "__main__":
    unittest.main()
