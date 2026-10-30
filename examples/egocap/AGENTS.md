# Ego Capture UI scope, layout, and code-quality standard

## Repository boundary

All Ego Capture UI project files and changes belong under this `egocap/` subtree. Shared repositories and repository-wide configuration outside this subtree are read-only references unless the user explicitly authorizes the specific external change. A plan, dependency requirement, or agent architectural preference is not that authorization.

## One project, one `pyproject.toml`

`egocap/` is one consolidated Python project with exactly one `pyproject.toml`, one `uv.lock`, and one `.venv`, all at `egocap/` root.

Do not create a second `pyproject.toml`/`uv.lock` anywhere under `egocap/`. The subpackages (`recording/`, `capture_server/`, and any future ones) are modules of this one project, each keeping its own importable package directory, sharing one dependency resolution so no two subpackages can ever disagree on a shared dependency's version. Tests live together under one top-level `tests/` directory (`tests/recording/`, `tests/capture_server/`, ...), not nested inside each subpackage, so one `testpaths` entry covers all of them.

## Run the tests yourself

Running `egocap/tests/` is the contributor's own responsibility for every change to this example.
Nothing else is guaranteed to catch a test failure here.

Run every suite yourself, in full (including `tests/backend` and the frontend's `npm run test`/`npm run lint` from `frontend/` when you touch `backend/` or `frontend/`):
- Before opening or updating an MR.
- After any change that touches code under `recording/` (including `recording/recording_service/`), `capture_server/`, `backend/` or `frontend/` (not just docs/config) — don't wait until the end of a work session to find out something broke three commits ago.

No external checkout is needed to run any test here; the project builds and runs entirely standalone:

```bash
uv run pytest tests/recording -q --cov=recording/egocap --cov-branch --cov-fail-under=100
uv run pytest tests/capture_server -q --cov=capture_server/capture_server --cov-branch
uv run pytest tests/recording_service -q --cov=recording/recording_service --cov-branch
```

`recording/recording_service/` (the recording service) and `egocap/recording/` (the module library) must never import `capture_server` or `egocap.capture`. `tests/recording_service/test_dependency_boundary.py` enforces this by parsing the source. The real-data service tests need `EGOCAP_REAL_EPISODE` set to a recorded episode directory (without it they are SKIPPED, which is not a pass); the full acceptance run is `recording/tools/replay_episode.py`.

A passing local run is the only verification this code gets before a human reviews it. Report an actual run's output, not an assumption that it would pass.

## Code quality — strict, because maintainers work from this documentation alone

Documentation must explain the purpose and constraints of the code so a maintainer unfamiliar
with its history can still work on it.

- **Every named function, method, class, and public property** needs a docstring describing its purpose. No exceptions in production code. Tests are exempt — a test's name and assertions should read clearly on their own; prefer a clearer name over an explanatory docstring in test files.
- **Every logical block** inside a function (validation, branch, loop, I/O, mutation, a concurrency/ownership boundary) needs a short inline comment. Do not rely on identifier names alone to carry that meaning.
- **Comments explain why, not what.** Don't restate the code (`# increment the counter`); explain the constraint, invariant, or past-failure reason a future reader can't derive by just reading the next line (e.g. why a guard exists, why a value is pinned, why an ordering matters).
- **Readable and concise**: small focused functions, clear names, shallow nesting. A small, well-defined public API that a caller can understand from its signature alone, dependencies passed in rather than reached for globally.
- Format and lint before every commit (`ruff check`/`ruff format` as configured in this project's `pyproject.toml`) — there is no pre-commit hook enforcing this here, so it is on the agent.
- This is public, open-source code: never use internal-only code names, project names, or terms anywhere under this tree — docs, code, comments, filenames, identifiers, or commit messages touching it. If unsure whether a term is internal-only or a legitimate public product name, ask before using it, rather than assuming either way.

## Before reporting a change ready

Inspect changed paths against the intended MR base and audit path/install/dependency references. If an external configuration or CI edit is actually required, identify its exact path and purpose and obtain explicit user authorization before editing it. Do not silently broaden scope or duplicate files outside the allowed tree.

Run all three test suites per "Run the tests yourself" above — check each package's own coverage target, not one combined number (see `pyproject.toml`'s `[tool.coverage.run]` comment for why).
