#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-commit hook: the agent indexes and skills must match the tree.

1. Relative links in llms.txt and skills/README.md name files that exist.
2. Docs-site URLs in llms.txt, docs/llms.txt and skills/ name the Markdown copy of
   a page docs/source/ builds on main, or one of main's llms.txt indexes.
3. Every skills/<name>/SKILL.md has Agent Skills frontmatter: a valid name that
   matches its directory, and a description.
4. Every skill is linked from both skills/README.md and llms.txt.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

DOCS_SITE = "https://nvidia.github.io/IsaacCapture/"
DOCS_SOURCE = "docs/source"
SKILLS = "skills"
SKILLS_INDEX = "skills/README.md"
REPO_INDEX = "llms.txt"
SITE_INDEX = "docs/llms.txt"

# Inline Markdown link targets, which in these files never hold whitespace or ")".
LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
# Every docs-site URL, linked or bare, so prose cannot carry a dead one either.
DOCS_URL_RE = re.compile(re.escape(DOCS_SITE) + r"[^\s)\]>\"'`]*")
# https://agentskills.io/specification
NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
NAME_MAX = 64
DESCRIPTION_MAX = 1024


def _repo_root() -> Path:
    """Return the work tree root, so the hook can run from any subdirectory."""
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(proc.stdout.strip())


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _local_links(root: Path, rel: str) -> tuple[list[str], set[Path]]:
    """Return problems with rel's relative links, and the files they resolve to."""
    text = _read(root / rel)
    if text is None:
        return [], set()
    problems: list[str] = []
    targets: set[Path] = set()
    for match in LINK_RE.finditer(text):
        target = match.group(1)
        if "://" in target or target.startswith(("#", "mailto:")):
            continue
        where = f"{rel}:{_line(text, match.start())}"
        resolved = ((root / rel).parent / target.split("#", 1)[0]).resolve()
        if not resolved.is_relative_to(root.resolve()):
            problems.append(f"{where}: `{target}` resolves outside the repository")
        elif not resolved.exists():
            problems.append(f"{where}: `{target}` does not exist")
        else:
            targets.add(resolved)
    return problems, targets


def _docs_url_problem(root: Path, url: str) -> str | None:
    """Return why url names nothing the main docs publish, or None if it does."""
    rest = url[len(DOCS_SITE) :].split("#", 1)[0]
    if rest == "llms.txt":
        return None
    page = rest.removeprefix("main/")
    if page == rest:
        return "is not under main/, the only version that publishes Markdown"
    source = root / DOCS_SOURCE
    if page == "llms.txt":
        return None
    if page.endswith("/llms.txt"):
        section = page.removesuffix("/llms.txt")
        if (source / section).is_dir():
            return None
        return f"names an index for {DOCS_SOURCE}/{section}/, which does not exist"
    if page.endswith(".md"):
        if (source / page).with_suffix(".rst").is_file():
            return None
        return f"names no page: {DOCS_SOURCE}/{page.removesuffix('.md')}.rst does not exist"
    return "is not a page's Markdown copy; link main/<page>.md or an llms.txt"


def _docs_links(root: Path, path: Path) -> list[str]:
    """Return one problem per docs-site URL in path that names nothing."""
    text = _read(path)
    if text is None:
        return []
    rel = path.relative_to(root)
    problems: list[str] = []
    for match in DOCS_URL_RE.finditer(text):
        url = match.group(0).rstrip(".,;:")
        reason = _docs_url_problem(root, url)
        if reason:
            problems.append(f"{rel}:{_line(text, match.start())}: {url} {reason}")
    return problems


def _frontmatter(text: str) -> dict[str, str] | None:
    """Return the top-level scalars of text's leading YAML frontmatter.

    Covers what skills use: plain, quoted and block (``>``, ``|``) scalars. A nested
    mapping reads as an empty string; None means there is no frontmatter block.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return None
    body = lines[1:end]
    fields: dict[str, str] = {}
    index = 0
    while index < len(body):
        line = body[index]
        index += 1
        if not line.strip() or line[:1].isspace() or line.startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if not separator:
            continue
        continuation: list[str] = []
        while index < len(body) and (
            not body[index].strip() or body[index][:1].isspace()
        ):
            continuation.append(body[index].strip())
            index += 1
        parts = [part for part in continuation if part]
        value = value.strip()
        if value[:1] in (">", "|"):
            value = (" " if value[0] == ">" else "\n").join(parts)
        elif len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif value:
            value = " ".join([value, *parts])
        fields[key.strip()] = value
    return fields


def _skill_problems(root: Path, skill: Path) -> list[str]:
    """Return one problem per way skill's SKILL.md breaks the Agent Skills format."""
    rel = skill.relative_to(root).as_posix()
    text = _read(skill / "SKILL.md")
    if text is None:
        return [f"{rel}/: has no SKILL.md"]
    where = f"{rel}/SKILL.md:1"
    fields = _frontmatter(text)
    if fields is None:
        return [f"{where}: has no YAML frontmatter between '---' lines"]
    problems: list[str] = []
    name = fields.get("name", "")
    if not name:
        problems.append(f"{where}: frontmatter has no name")
    else:
        if name != skill.name:
            problems.append(
                f"{where}: name {name!r} does not match its directory {skill.name!r}"
            )
        if not NAME_RE.fullmatch(name):
            problems.append(
                f"{where}: name {name!r} must be lowercase letters and digits "
                "joined by single hyphens"
            )
        if len(name) > NAME_MAX:
            problems.append(
                f"{where}: name is {len(name)} characters; the limit is {NAME_MAX}"
            )
    description = fields.get("description", "")
    if not description:
        problems.append(f"{where}: frontmatter has no description")
    elif len(description) > DESCRIPTION_MAX:
        problems.append(
            f"{where}: description is {len(description)} characters; "
            f"the limit is {DESCRIPTION_MAX}"
        )
    return problems


def check(root: Path) -> list[str]:
    """Return one message per problem in the agent indexes and skills under root."""
    problems: list[str] = []
    linked: dict[str, set[Path]] = {}
    for index in (REPO_INDEX, SKILLS_INDEX):
        found, linked[index] = _local_links(root, index)
        problems += found

    skills_dir = root / SKILLS
    for path in (
        root / REPO_INDEX,
        root / SITE_INDEX,
        *sorted(skills_dir.rglob("*.md")),
    ):
        problems += _docs_links(root, path)

    skills = (
        sorted(
            p for p in skills_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
        if skills_dir.is_dir()
        else []
    )
    for skill in skills:
        problems += _skill_problems(root, skill)
        wanted = (skill / "SKILL.md").resolve()
        for index, targets in linked.items():
            if wanted not in targets:
                problems.append(
                    f"{skill.relative_to(root).as_posix()}/: not linked from {index}"
                )
    return problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Check llms.txt and skills/.")
    parser.parse_args(argv)
    problems = check(_repo_root())
    if problems:
        print("The agent indexes or skills do not match the tree.", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
