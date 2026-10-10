# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orders results worst first and groups them."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ..report import CheckResult, Mark, Report

# Marks ordered worst first.
WORST_FIRST = (Mark.FAIL, Mark.UNANSWERED, Mark.MEAS, Mark.NOTE, Mark.PASS)


@dataclass(frozen=True, slots=True)
class Group:
    code: str
    title: str
    results: tuple[CheckResult, ...]

    @property
    def mark(self) -> Mark:
        """The worst mark in the group."""
        return min((r.mark for r in self.results), key=WORST_FIRST.index)

    @property
    def tally(self) -> str:
        counts = Counter(r.mark for r in self.results)
        return ", ".join(
            f"{counts[mark]} {mark}" for mark in WORST_FIRST if counts[mark]
        )


def worst_first(results: tuple[CheckResult, ...]) -> tuple[CheckResult, ...]:
    """Sorts results worst mark first, keeping report order within a mark."""
    return tuple(sorted(results, key=lambda r: WORST_FIRST.index(r.mark)))


def grouped(report: Report, titles: Sequence[tuple[str, str]]) -> tuple[Group, ...]:
    """Groups with checks in them, in the declared order. An empty group is not shown."""
    by_code: dict[str, list[CheckResult]] = {}
    for result in report.results:
        by_code.setdefault(result.group, []).append(result)
    return tuple(
        Group(code, title, worst_first(tuple(by_code[code])))
        for code, title in titles
        if code in by_code
    )


def picked(report: Report, names: Sequence[str]) -> tuple[CheckResult, ...]:
    by_name = {result.name: result for result in report.results}
    return tuple(by_name[name] for name in names if name in by_name)
