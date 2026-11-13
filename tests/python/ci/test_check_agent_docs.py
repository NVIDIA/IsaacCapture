# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from repo_paths import repo_root

spec = importlib.util.spec_from_file_location(
    "check_agent_docs", repo_root() / "scripts/check_agent_docs.py"
)
agent_docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_docs)

SITE = "https://nvidia.github.io/IsaacCapture/main"


class CheckAgentDocsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.write("docs/source/teleop/quick_start.rst", "Quick Start\n")
        self.write("docs/source/device/index.rst", "Devices\n")
        self.write("docs/llms.txt", f"- [Docs]({SITE}/llms.txt)\n")
        self.write("skills/README.md", "# Skills\n\n")
        self.write(
            "llms.txt", f"# Demo\n\n- [Quick Start]({SITE}/teleop/quick_start.md)\n"
        )
        self.add_skill(
            "demo-skill", "name: demo-skill\ndescription: Use when testing.\n"
        )

    def write(self, rel, text):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def append(self, rel, text):
        path = self.root / rel
        path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")

    def add_skill(self, directory, frontmatter, *, readme=True, llms=True):
        self.write(
            f"skills/{directory}/SKILL.md", f"---\n{frontmatter}---\n\n# Skill\n"
        )
        if readme:
            self.append("skills/README.md", f"- [{directory}]({directory}/SKILL.md)\n")
        if llms:
            self.append("llms.txt", f"- [{directory}](skills/{directory}/SKILL.md)\n")

    def assert_one_problem(self, *fragments):
        problems = agent_docs.check(self.root)
        self.assertEqual(len(problems), 1, problems)
        for fragment in fragments:
            self.assertIn(fragment, problems[0])

    def test_consistent_tree_passes(self):
        self.assertEqual(agent_docs.check(self.root), [])

    def test_repository_passes(self):
        self.assertEqual(agent_docs.check(repo_root()), [])

    def test_relative_link_must_exist(self):
        self.append("llms.txt", "- [Gone](skills/gone/SKILL.md)\n")
        self.assert_one_problem("llms.txt:", "`skills/gone/SKILL.md` does not exist")

    def test_relative_link_must_stay_in_the_repository(self):
        self.append("skills/README.md", "- [Out](../../outside.md)\n")
        self.assert_one_problem("skills/README.md:", "resolves outside the repository")

    def test_anchors_and_external_urls_are_not_paths(self):
        self.append("llms.txt", "- [Top](#demo)\n- [GitHub](https://github.com/x/y)\n")
        self.assertEqual(agent_docs.check(self.root), [])

    def test_docs_link_must_be_a_markdown_copy(self):
        self.append("llms.txt", f"- [HTML]({SITE}/teleop/quick_start.html)\n")
        self.assert_one_problem("quick_start.html", "is not a page's Markdown copy")

    def test_docs_link_must_name_a_built_page(self):
        self.append("llms.txt", f"- [Gone]({SITE}/teleop/gone.md)\n")
        self.assert_one_problem("docs/source/teleop/gone.rst does not exist")

    def test_docs_link_must_point_at_main(self):
        url = (
            "https://nvidia.github.io/IsaacCapture/release/1.7.x/teleop/quick_start.md"
        )
        self.append("llms.txt", f"- [Old]({url})\n")
        self.assert_one_problem("is not under main/")

    def test_section_index_needs_the_section(self):
        self.append("llms.txt", f"- [Devices]({SITE}/device/llms.txt)\n")
        self.assertEqual(agent_docs.check(self.root), [])
        self.append("llms.txt", f"- [Nope]({SITE}/nope/llms.txt)\n")
        self.assert_one_problem("docs/source/nope/, which does not exist")

    def test_bare_docs_urls_in_skills_are_checked(self):
        self.append(
            "skills/demo-skill/SKILL.md", f"\nSee {SITE}/teleop/quick_start.md.\n"
        )
        self.assertEqual(agent_docs.check(self.root), [])
        self.append("skills/demo-skill/SKILL.md", f"Or {SITE}/teleop/missing.md\n")
        self.assert_one_problem("skills/demo-skill/SKILL.md:", "missing.rst")

    def test_site_index_is_checked(self):
        self.append("docs/llms.txt", f"- [Gone]({SITE}/gone.md)\n")
        self.assert_one_problem("docs/llms.txt:", "gone.rst does not exist")

    def test_name_must_match_the_directory(self):
        self.add_skill("other-skill", "name: another-skill\ndescription: Testing.\n")
        self.assert_one_problem(
            "other-skill/SKILL.md:1", "does not match its directory"
        )

    def test_name_format(self):
        self.add_skill("Bad--Name", "name: Bad--Name\ndescription: Testing.\n")
        self.assert_one_problem("Bad--Name", "lowercase letters and digits")

    def test_name_length(self):
        name = "a" * 65
        self.add_skill(name, f"name: {name}\ndescription: Testing.\n")
        self.assert_one_problem("name is 65 characters; the limit is 64")

    def test_description_is_required(self):
        self.add_skill("no-description", "name: no-description\n")
        self.assert_one_problem("frontmatter has no description")

    def test_description_length(self):
        description = "x" * 1025
        self.add_skill(
            "long-description", f"name: long-description\ndescription: {description}\n"
        )
        self.assert_one_problem("description is 1025 characters; the limit is 1024")

    def test_block_and_quoted_scalars(self):
        self.add_skill(
            "folded-skill",
            "name: 'folded-skill'\ndescription: >-\n  Use when the\n  value folds.\n"
            "metadata:\n  internal: true\n",
        )
        self.assertEqual(agent_docs.check(self.root), [])
        fields = agent_docs._frontmatter(
            (self.root / "skills/folded-skill/SKILL.md").read_text(encoding="utf-8")
        )
        self.assertEqual(fields["description"], "Use when the value folds.")
        self.assertEqual(fields["metadata"], "")

    def test_frontmatter_is_required(self):
        self.write("skills/plain-skill/SKILL.md", "# No frontmatter\n")
        self.append("skills/README.md", "- [plain-skill](plain-skill/SKILL.md)\n")
        self.append("llms.txt", "- [plain-skill](skills/plain-skill/SKILL.md)\n")
        self.assert_one_problem("plain-skill/SKILL.md:1", "no YAML frontmatter")

    def test_skill_directory_needs_skill_md(self):
        (self.root / "skills/empty-skill").mkdir()
        problems = agent_docs.check(self.root)
        self.assertIn("skills/empty-skill/: has no SKILL.md", problems)

    def test_skill_must_be_in_both_indexes(self):
        self.add_skill(
            "unlisted-skill", "name: unlisted-skill\ndescription: x\n", llms=False
        )
        self.assert_one_problem("skills/unlisted-skill/: not linked from llms.txt")
        self.add_skill(
            "hidden-skill", "name: hidden-skill\ndescription: x\n", readme=False
        )
        problems = agent_docs.check(self.root)
        self.assertIn(
            "skills/hidden-skill/: not linked from skills/README.md", problems
        )


if __name__ == "__main__":
    unittest.main()
