# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The workstation's APAI Python environment: its script, its list, the launcher.

setup_workstation_env.sh is run with stand-ins for uv, npm and mkdir as the
whole PATH, so no test can reach the real uv or npm: nothing is installed and
nothing is downloaded.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
from isaaccapture_examples.agentic_remote.agent import config as agent_config

EXAMPLE = Path(agent_config.__file__).resolve().parents[4]
SETUP = EXAMPLE / "setup_workstation_env.sh"
REQUIREMENTS = EXAMPLE / "requirements-workstation.txt"
RUN_AGENT = EXAMPLE / "run_agent.sh"
BASH = shutil.which("bash") or "/bin/bash"
# The cache variables run_agent.sh sets, and their folders under caches/.
CACHES = {
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "YOLO_CONFIG_DIR": "ultralytics",
}
# The Thor's list of installed packages, in the experiment folder.
THOR_LIST = ".state/thor/ffs/python-packages.txt"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
MODEL_CLIENT = "@openai/codex@0.167.0"


# run_agent.sh


def start_agent(tmp_path: Path, **more: str) -> subprocess.CompletedProcess[str]:
    """Run run_agent.sh on a stand-in interpreter that prints its environment."""
    fake = tmp_path / "python"
    names = ("AGENTIC_REMOTE_ENV_DIR", *CACHES)
    fake.write_text(
        f"#!{BASH}\n" + "".join(f'echo "{name}=${{{name}-unset}}"\n' for name in names)
    )
    fake.chmod(0o755)
    env = {
        "PATH": os.environ["PATH"],
        "AGENTIC_REMOTE_AGENT_PYTHON": str(fake),
        "AGENTIC_REMOTE_APAI_SRC": str(tmp_path),
        **more,
    }
    return subprocess.run(
        [str(RUN_AGENT), "--check"],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def run_agent(tmp_path: Path, **more: str) -> dict[str, str]:
    """As start_agent(), for a run that starts; what it saw of the environment."""
    done = start_agent(tmp_path, **more)
    assert done.returncode == 0, done.stderr
    return dict(line.split("=", 1) for line in done.stdout.splitlines())


def agent_refused(done: subprocess.CompletedProcess[str], piece: str) -> None:
    """Exit code 2, one line that says why, and the interpreter was not started."""
    assert done.returncode == 2, done.stdout + done.stderr
    (line,) = done.stderr.splitlines()
    assert line.startswith("run_agent.sh: ") and piece in line
    assert done.stdout == ""


def test_the_launcher_puts_the_caches_into_the_environment_folder(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "env"
    seen = run_agent(tmp_path, AGENTIC_REMOTE_ENV_DIR=str(folder))
    assert seen["AGENTIC_REMOTE_ENV_DIR"] == str(folder)  # exported
    for name, leaf in CACHES.items():
        assert seen[name] == f"{folder}/caches/{leaf}"
    assert not folder.exists()  # the launcher makes no folder


def test_the_launcher_keeps_a_cache_variable_that_is_set(tmp_path: Path) -> None:
    folder = tmp_path / "env"
    seen = run_agent(
        tmp_path,
        AGENTIC_REMOTE_ENV_DIR=str(folder),
        TRITON_CACHE_DIR="/elsewhere/triton",
        YOLO_CONFIG_DIR="",  # empty counts as not set
    )
    assert seen["TRITON_CACHE_DIR"] == "/elsewhere/triton"
    assert seen["TORCHINDUCTOR_CACHE_DIR"] == f"{folder}/caches/torchinductor"
    assert seen["YOLO_CONFIG_DIR"] == f"{folder}/caches/ultralytics"


@pytest.mark.parametrize("given", [{}, {"AGENTIC_REMOTE_ENV_DIR": ""}])
def test_without_the_environment_folder_the_launcher_sets_no_cache(
    tmp_path: Path, given: dict[str, str]
) -> None:
    # The agent program then keeps the caches under its state folder.
    seen = run_agent(tmp_path, **given)
    for name in CACHES:
        assert seen[name] == "unset"
    # A cache variable of the caller's passes through untouched.
    seen = run_agent(tmp_path, **given, TORCHINDUCTOR_CACHE_DIR="/mine")
    assert seen["TORCHINDUCTOR_CACHE_DIR"] == "/mine"
    assert seen["TRITON_CACHE_DIR"] == "unset"
    # Unless it names a place inside the experiment folder.
    experiment = tmp_path / "experiment"
    given["AGENTIC_REMOTE_EXPERIMENT_DIR"] = str(experiment)
    for name in CACHES:
        value = f"{experiment}/.state/thor/ffs/{CACHES[name]}"
        done = start_agent(tmp_path, **given, **{name: value})
        agent_refused(done, f"{name} names a place inside the experiment folder")
    seen = run_agent(tmp_path, **given, TORCHINDUCTOR_CACHE_DIR="/mine")
    assert seen["TORCHINDUCTOR_CACHE_DIR"] == "/mine"


@pytest.mark.parametrize("folder", ["env", "./env", "../env"])
def test_the_launcher_refuses_an_environment_folder_that_is_not_absolute(
    tmp_path: Path, folder: str
) -> None:
    done = start_agent(tmp_path, AGENTIC_REMOTE_ENV_DIR=folder)
    agent_refused(
        done, f"AGENTIC_REMOTE_ENV_DIR must be an absolute path (it is '{folder}')"
    )


@pytest.mark.parametrize("name", list(CACHES))
def test_the_launcher_refuses_a_cache_variable_into_the_experiment_folder(
    tmp_path: Path, name: str
) -> None:
    experiment = tmp_path / "experiment"
    given = {
        "AGENTIC_REMOTE_ENV_DIR": str(tmp_path / "env"),
        "AGENTIC_REMOTE_EXPERIMENT_DIR": f"{experiment}/",  # the slash is dropped
    }
    for value in (f"{experiment}/.state/thor/ffs/cache", str(experiment)):
        done = start_agent(tmp_path, **given, **{name: value})
        agent_refused(done, f"{name} names a place inside the experiment folder")
        assert f"({value})" in done.stderr
    # A folder whose name only begins like the experiment folder's is kept.
    beside = f"{experiment}-caches/x"
    assert run_agent(tmp_path, **given, **{name: beside})[name] == beside


def tree(folder: Path) -> list[str]:
    """Every file and folder under a folder, links not followed."""
    return sorted(str(path.relative_to(folder)) for path in folder.rglob("*"))


@pytest.mark.parametrize("with_env_dir", [True, False])
def test_the_launcher_refuses_a_cache_into_the_experiment_folder_however_spelled(
    tmp_path: Path, with_env_dir: bool
) -> None:
    experiment = tmp_path / "checkout" / "experiment"
    (experiment / ".state" / "cache").mkdir(parents=True)
    (tmp_path / "beside").mkdir()
    (tmp_path / "link-to-experiment").symlink_to(experiment)
    (tmp_path / "fast-disk").symlink_to(experiment / ".state")
    given = {"AGENTIC_REMOTE_EXPERIMENT_DIR": str(experiment)}
    if with_env_dir:
        given["AGENTIC_REMOTE_ENV_DIR"] = str(tmp_path / "env")
    inside = "names a place inside the experiment folder"
    dots = "must not hold a '.' or '..' component"
    cases = {
        f"{tmp_path}/fast-disk/cache": inside,  # through a symbolic link
        f"{tmp_path}/link-to-experiment/x": inside,
        f"{tmp_path}/checkout//experiment/.state": inside,
        f"{experiment}/not/there/yet": inside,
        f"{experiment}/": inside,
        f"{tmp_path}/beside/../checkout/experiment/.state": dots,
        f"{tmp_path}/checkout/./experiment": dots,
        # Outside, and not called inside for the way it is written.
        f"{experiment}/../beside": dots,
        ".state/cache": "must be an absolute path",
    }
    before = tree(tmp_path / "checkout")
    for name in CACHES:
        for value, piece in cases.items():
            done = start_agent(tmp_path, **given, **{name: value})
            agent_refused(done, f"{name} {piece}")
            assert value in done.stderr
    # The experiment folder named through a link, or with two slashes, is the same.
    for named in (
        f"{tmp_path}/link-to-experiment",
        f"{tmp_path}/checkout//experiment/",
    ):
        given["AGENTIC_REMOTE_EXPERIMENT_DIR"] = named
        done = start_agent(tmp_path, **given, YOLO_CONFIG_DIR=f"{experiment}/.state/x")
        agent_refused(done, f"YOLO_CONFIG_DIR {inside}")
    # A folder beside it, and one whose name begins like it, are kept.
    for value in (f"{tmp_path}/beside/x", f"{experiment}-caches"):
        seen = run_agent(tmp_path, **given, TRITON_CACHE_DIR=value)
        assert seen["TRITON_CACHE_DIR"] == value
    assert tree(tmp_path / "checkout") == before  # the launcher made nothing


def test_the_launcher_refuses_an_environment_folder_inside_the_experiment_folder(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (tmp_path / "envs").symlink_to(experiment)
    given = {"AGENTIC_REMOTE_EXPERIMENT_DIR": str(experiment)}
    # The caches the launcher itself would name are compared too.
    for folder in (f"{experiment}/.state/env", f"{tmp_path}/envs/env"):
        done = start_agent(tmp_path, **given, AGENTIC_REMOTE_ENV_DIR=folder)
        agent_refused(done, "TRITON_CACHE_DIR names a place inside the experiment")
        assert f"({folder}/caches/triton)" in done.stderr
    # A cache the caller names elsewhere does not make that folder usable.
    done = start_agent(
        tmp_path,
        **given,
        AGENTIC_REMOTE_ENV_DIR=f"{experiment}/env",
        TRITON_CACHE_DIR="/elsewhere/triton",
    )
    agent_refused(done, "TORCHINDUCTOR_CACHE_DIR names a place inside the experiment")
    # Beside the experiment folder it runs.
    seen = run_agent(tmp_path, **given, AGENTIC_REMOTE_ENV_DIR=f"{tmp_path}/env")
    assert seen["YOLO_CONFIG_DIR"] == f"{tmp_path}/env/caches/ultralytics"
    assert list(experiment.iterdir()) == []


@pytest.mark.parametrize("folder", ["env/../elsewhere", "./env", "env/.", "env/.."])
def test_the_launcher_refuses_an_environment_folder_with_a_dot_component(
    tmp_path: Path, folder: str
) -> None:
    named = f"{tmp_path}/{folder}"
    done = start_agent(tmp_path, AGENTIC_REMOTE_ENV_DIR=named)
    agent_refused(
        done,
        f"AGENTIC_REMOTE_ENV_DIR must not hold a '.' or '..' component (it is "
        f"'{named}')",
    )
    # A name that only holds dots is a name.
    seen = run_agent(tmp_path, AGENTIC_REMOTE_ENV_DIR=f"{tmp_path}/..env")
    assert seen["TRITON_CACHE_DIR"] == f"{tmp_path}/..env/caches/triton"


def test_the_launcher_names_the_environment_folder_in_its_text() -> None:
    script = RUN_AGENT.read_text()
    assert "export AGENTIC_REMOTE_ENV_DIR" in script
    for name, leaf in CACHES.items():
        assert f'export {name}="${{{name}:-${{caches}}/{leaf}}}"' in script
    assert "[ " not in script.replace("[[ ", "")  # [[ ]] only


# setup_workstation_env.sh


class Tools:
    """Stand-ins for uv, npm and mkdir in one folder, which is the whole PATH.

    Every call of uv and npm is one line of .log; nothing else is done, except
    that "uv pip freeze" prints one package. A call whose arguments hold
    fail_on ends with exit code 7.
    """

    def __init__(self, folder: Path, names: tuple[str, ...] = ("uv", "npm")) -> None:
        self.folder = folder / "tools"
        self.folder.mkdir(parents=True)
        self.log = folder / "calls.log"
        for name in names:
            tool = self.folder / name
            tool.write_text(
                f"#!{BASH}\n"
                f'echo "{name} $*" >> "${{TOOLS_LOG}}"\n'
                'if [[ -n "${TOOLS_FAIL_ON:-}" && "$*" == *"${TOOLS_FAIL_ON}"* ]]; then\n'
                "  exit 7\n"
                "fi\n"
                'if [[ "$*" == "pip freeze"* ]]; then echo "torch==2.10.0+cu130"; fi\n'
            )
            tool.chmod(0o755)
        for name in ("mkdir", "dirname"):
            real = shutil.which(name)
            assert real is not None
            (self.folder / name).symlink_to(real)

    def calls(self) -> list[str]:
        return self.log.read_text().splitlines() if self.log.exists() else []

    def run(
        self,
        *arguments: str,
        env_dir: Path | str | None,
        fail_on: str = "",
        by_name: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """by_name runs it from its own folder as `bash setup_workstation_env.sh`."""
        env = {"PATH": str(self.folder), "TOOLS_LOG": str(self.log)}
        if env_dir is not None:
            env["AGENTIC_REMOTE_ENV_DIR"] = str(env_dir)
        if fail_on:
            env["TOOLS_FAIL_ON"] = fail_on
        return subprocess.run(
            [BASH, SETUP.name if by_name else str(SETUP), *arguments],
            check=False,
            cwd=EXAMPLE if by_name else None,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )


@pytest.fixture
def tools(tmp_path: Path) -> Tools:
    return Tools(tmp_path)


def expected_calls(folder: Path, model_client: bool = False) -> list[str]:
    """The lines of uv and npm of one whole run, in order."""
    python = f"{folder}/venv/bin/python"
    calls = [
        f"uv venv {folder}/venv --python 3.12",
        f"uv pip install --python {python} --index-url {TORCH_INDEX} "
        "torch==2.10.0 torchvision==0.25.0",
        f"uv pip install --python {python} --excludes "
        f"{folder}/exclude-gui-opencv.txt -r {REQUIREMENTS}",
    ]
    if model_client:
        calls.append(
            f"npm install --prefix {folder}/codex --no-audit --no-fund {MODEL_CLIENT}"
        )
    return [*calls, f"uv pip freeze --python {python}"]


def test_the_script_is_a_bash_script_that_can_be_run() -> None:
    text = SETUP.read_text()
    assert text.startswith("#!/usr/bin/env bash\n")
    assert SETUP.stat().st_mode & stat.S_IXUSR
    assert "SPDX-License-Identifier: Apache-2.0" in text
    assert "\nset -euo pipefail\n" in text
    assert "[ " not in text.replace("[[ ", "")  # [[ ]] only
    # No path of a user's home, and nothing of the model client's login.
    assert "/home/" not in text and "~" not in text and ".codex" not in text
    assert "needs the owner's yes" in text
    done = subprocess.run([BASH, "-n", str(SETUP)], check=False, capture_output=True)
    assert done.returncode == 0, done.stderr


def test_a_dry_run_prints_every_step_and_runs_nothing(
    tmp_path: Path, tools: Tools
) -> None:
    folder = tmp_path / "env"
    done = tools.run("--dry-run", env_dir=folder)
    assert done.returncode == 0, done.stderr
    assert tools.calls() == [] and not folder.exists()
    out = done.stdout
    steps = re.findall(r"^step (\w+):", out, flags=re.MULTILINE)
    assert steps == ["0", "1", "2", "3a", "3b", "4", "5"]
    # Each command the whole run makes is printed, in order.
    at = 0
    for call in expected_calls(folder):
        at = out.index(call, at)
    assert f"mkdir -p {folder}/caches/triton" in out
    assert f"> {folder}/exclude-gui-opencv.txt" in out
    assert f"> {folder}/freeze.txt" in out
    # The model client is left out, and the script says what it waits for.
    assert "npm" not in out
    assert "step 4: the model client: left out (it needs the owner's yes" in out
    assert out.rstrip().endswith("dry run: no command was run")


def test_the_model_client_is_a_step_only_with_its_option(
    tmp_path: Path, tools: Tools
) -> None:
    folder = tmp_path / "env"
    done = tools.run("--with-model-client", "--dry-run", env_dir=folder)
    assert done.returncode == 0, done.stderr
    assert tools.calls() == []
    at = 0
    for call in expected_calls(folder, model_client=True):
        at = done.stdout.index(call, at)
    assert "step 4: the model client (downloads; needs the owner's yes)" in done.stdout


def test_a_whole_run_makes_the_steps_in_order(tmp_path: Path, tools: Tools) -> None:
    folder = tmp_path / "env"
    done = tools.run(env_dir=f"{folder}/")  # a trailing slash is dropped
    assert done.returncode == 0, done.stderr
    assert tools.calls() == expected_calls(folder)
    for leaf in CACHES.values():
        assert (folder / "caches" / leaf).is_dir()
    assert (folder / "exclude-gui-opencv.txt").read_text() == "opencv-python\n"
    assert (folder / "freeze.txt").read_text() == "torch==2.10.0+cu130\n"
    assert not (folder / "codex").exists()
    assert f"AGENTIC_REMOTE_AGENT_PYTHON={folder}/venv/bin/python" in done.stdout


def test_a_whole_run_with_the_model_client_calls_npm_before_the_freeze(
    tmp_path: Path, tools: Tools
) -> None:
    folder = tmp_path / "env"
    done = tools.run("--with-model-client", env_dir=folder)
    assert done.returncode == 0, done.stderr
    assert tools.calls() == expected_calls(folder, model_client=True)


@pytest.mark.parametrize(
    ("fail_on", "step", "made"),
    [
        ("venv", "step 1 (the Python 3.12 environment)", 1),
        ("--index-url", "step 2 (torch for CUDA", 2),
        ("--excludes", "step 3b (the packages of requirements-workstation.txt", 3),
        ("pip freeze", "step 5 (the list of what was installed)", 4),
    ],
)
def test_the_script_stops_at_the_first_step_that_fails_and_names_it(
    tmp_path: Path, tools: Tools, fail_on: str, step: str, made: int
) -> None:
    folder = tmp_path / "env"
    done = tools.run(env_dir=folder, fail_on=fail_on)
    assert done.returncode == 1
    assert tools.calls() == expected_calls(folder)[:made]
    (line,) = done.stderr.splitlines()
    assert line.startswith(f"setup_workstation_env.sh: {step}")
    assert "failed with exit code 7; the later steps were not run" in line
    assert "done:" not in done.stdout


def refused(done: subprocess.CompletedProcess[str], piece: str) -> None:
    """Exit code 2 and one line that says why; nothing was printed as a step."""
    assert done.returncode == 2, done.stdout + done.stderr
    (line,) = done.stderr.splitlines()
    assert line.startswith("setup_workstation_env.sh: ") and piece in line
    assert done.stdout == ""


@pytest.mark.parametrize("more", [(), ("--dry-run",)])
def test_the_script_refuses_what_it_cannot_build_on(
    tmp_path: Path, tools: Tools, more: tuple[str, ...]
) -> None:
    refused(tools.run(*more, env_dir=None), "AGENTIC_REMOTE_ENV_DIR must name")
    refused(tools.run(*more, env_dir=""), "AGENTIC_REMOTE_ENV_DIR must name")
    refused(tools.run(*more, env_dir="env"), "must be an absolute path")
    refused(
        tools.run(*more, "--force", env_dir=tmp_path / "env"),
        "unknown argument '--force'",
    )
    # Inside a git repository: a folder that holds .git, at any depth above.
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    for inside in (repository, repository / "a" / "b" / "env"):
        refused(
            tools.run(*more, env_dir=inside),
            f"lies inside the git repository {repository}",
        )
    # A work tree's .git is a file.
    worktree = tmp_path / "worktree"
    worktree.mkdir()
    (worktree / ".git").write_text("gitdir: elsewhere\n")
    refused(tools.run(*more, env_dir=worktree / "env"), "inside the git repository")
    # This repository.
    refused(tools.run(*more, env_dir=EXAMPLE / "env"), "inside the git repository")
    assert not (EXAMPLE / "env").exists()
    # An environment that is already there.
    built = tmp_path / "built"
    (built / "venv").mkdir(parents=True)
    refused(tools.run(*more, env_dir=built), f"{built}/venv exists")
    assert tools.calls() == []
    assert not (tmp_path / "env").exists() and not (repository / "a").exists()


def test_the_script_runs_when_it_is_named_without_a_folder(
    tmp_path: Path, tools: Tools
) -> None:
    folder = tmp_path / "env"
    done = tools.run("--help", env_dir=None, by_name=True)
    assert done.returncode == 0 and done.stderr == ""
    assert "--with-model-client" in done.stdout
    done = tools.run("--dry-run", env_dir=folder, by_name=True)
    assert done.returncode == 0, done.stderr
    assert f"-r {REQUIREMENTS}" in done.stdout  # found beside the script
    assert tools.calls() == [] and not folder.exists()
    done = tools.run(env_dir=folder, by_name=True)
    assert done.returncode == 0, done.stderr
    assert tools.calls() == expected_calls(folder)


@pytest.mark.parametrize("more", [(), ("--dry-run",)])
def test_the_script_refuses_a_folder_named_through_a_dot_component(
    tmp_path: Path, tools: Tools, more: tuple[str, ...]
) -> None:
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    (tmp_path / "outside").mkdir()
    # Out of a folder that does not exist yet, into a repository.
    for named in (
        f"{tmp_path}/outside/new/../../repository/env",
        f"{tmp_path}/outside/..",
        f"{tmp_path}/outside/../env/",
        f"{tmp_path}/./env",
        f"{tmp_path}/env/.",
    ):
        refused(tools.run(*more, env_dir=named), "'.' or '..' component")
    assert tools.calls() == []
    assert sorted(path.name for path in repository.iterdir()) == [".git"]
    assert not (tmp_path / "env").exists()
    assert not (tmp_path / "outside" / "new").exists()
    # A name that only holds dots is a name.
    for name in ("..env", "env..", ".env", "a.b"):
        done = tools.run("--dry-run", env_dir=tmp_path / name)
        assert done.returncode == 0, done.stderr


def a_repository(folder: Path) -> Path:
    """A folder that holds .git and one file of its owner's."""
    (folder / ".git").mkdir(parents=True)
    (folder / "notes.md").write_text("the owner's notes\n")
    return folder


@pytest.mark.parametrize("more", [(), ("--dry-run",)])
def test_the_script_refuses_a_folder_that_is_a_link_into_a_repository(
    tmp_path: Path, tools: Tools, more: tuple[str, ...]
) -> None:
    repository = a_repository(tmp_path / "repository")
    (repository / "deep").mkdir()
    before = tree(repository)
    # The folder itself, and a folder above one that does not exist yet.
    (tmp_path / "env").symlink_to(repository / "deep")
    (tmp_path / "disk").symlink_to(repository)
    for named in (tmp_path / "env", tmp_path / "disk" / "new" / "env"):
        done = tools.run(*more, env_dir=named)
        refused(done, f"{named} lies inside the git repository {repository}")
    assert tools.calls() == [] and tree(repository) == before


@pytest.mark.parametrize(
    "made", ["caches", "venv", "codex", "freeze.txt", "exclude-gui-opencv.txt"]
)
@pytest.mark.parametrize("more", [(), ("--dry-run",), ("--with-model-client",)])
def test_the_script_refuses_a_link_where_it_makes_something(
    tmp_path: Path, tools: Tools, more: tuple[str, ...], made: str
) -> None:
    repository = a_repository(tmp_path / "repository")
    (repository / "store").mkdir()
    before = tree(repository)
    folder = tmp_path / "env"
    folder.mkdir()
    # A link to a folder, to a file, and to nothing that is there yet.
    for target in (repository / "store", repository / "notes.md", repository / "new"):
        link = folder / made
        link.symlink_to(target)
        done = tools.run(*more, env_dir=folder)
        refused(done, f"{link} is a symbolic link; this script makes it in the folder")
        link.unlink()
    assert tools.calls() == [] and tree(repository) == before
    assert (repository / "notes.md").read_text() == "the owner's notes\n"
    assert list(folder.iterdir()) == []


@pytest.mark.parametrize("more", [(), ("--dry-run",)])
def test_a_newline_at_the_end_of_a_folders_name_is_part_of_the_name(
    tmp_path: Path, tools: Tools, more: tuple[str, ...]
) -> None:
    # A repository whose name ends with a newline: inside it, and the root itself.
    repository = a_repository(tmp_path / "repo\n")
    before = tree(repository)
    for named in (repository / "env", repository):
        done = tools.run(*more, env_dir=named)
        assert done.returncode == 2, done.stdout + done.stderr
        assert done.stdout == "" and "lies inside the git repository" in done.stderr
    assert tools.calls() == [] and tree(repository) == before
    # A folder of that name beside a repository of the name without it is none.
    a_repository(tmp_path / "plain")
    (tmp_path / "plain\n").mkdir()
    done = tools.run("--dry-run", env_dir=tmp_path / "plain\n" / "env")
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("more", [(), ("--dry-run",)])
def test_the_script_refuses_without_uv_and_without_npm_for_the_model_client(
    tmp_path: Path, more: tuple[str, ...]
) -> None:
    folder = tmp_path / "env"
    no_uv = Tools(tmp_path / "a", names=("npm",))
    refused(no_uv.run(*more, env_dir=folder), "uv is not on PATH")
    refused(
        no_uv.run(*more, "--with-model-client", env_dir=folder), "uv is not on PATH"
    )
    no_npm = Tools(tmp_path / "b", names=("uv",))
    refused(
        no_npm.run(*more, "--with-model-client", env_dir=folder),
        "npm is not on PATH",
    )
    assert no_uv.calls() == [] and no_npm.calls() == [] and not folder.exists()
    # Without the option npm is not needed.
    assert no_npm.run("--dry-run", env_dir=folder).returncode == 0


def test_help_prints_the_options_and_runs_nothing(tmp_path: Path, tools: Tools) -> None:
    done = tools.run("--help", env_dir=None)
    assert done.returncode == 0 and done.stderr == ""
    for piece in ("--dry-run", "--with-model-client", "needs the owner's yes"):
        assert piece in done.stdout
    assert "set -euo pipefail" not in done.stdout and tools.calls() == []


# requirements-workstation.txt


def pins(path: Path) -> dict[str, str]:
    """name -> version of every line `name==version`; other lines are an error."""
    found: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-e ")):
            continue
        name, separator, version = line.partition("==")
        assert separator and name and version and " " not in line, line
        assert name not in found, line
        found[name] = version
    return found


def test_the_list_pins_every_package_and_leaves_out_what_is_not_installed() -> None:
    text = REQUIREMENTS.read_text()
    assert text.startswith("# SPDX-FileCopyrightText:")
    assert "# SPDX-License-Identifier: Apache-2.0" in text
    assert "python-packages.txt" in text and "freeze.txt" in text
    wanted = pins(REQUIREMENTS)
    assert set(wanted) == {
        "numpy",
        "opencv-python-headless",
        "pyyaml",
        "jsonschema",
        "pin",
        "warp-lang",
        "scipy",
        "pyzmq",
        "msgpack",
        "msgpack-numpy",
        "ultralytics",
        "timm",
        "omegaconf",
        "einops",
        "scikit-image",
        "imageio",
        "pytest",
    }
    # torch has its own index; the rest is not installed at all.
    for name in ("torch", "torchvision", "opencv-python", "openai", "motorbridge"):
        assert name not in wanted
    assert "cupy" not in text and "apai" not in wanted


@pytest.mark.experiment
def test_each_pin_is_the_version_the_thor_ran(experiment_dir: Path) -> None:
    thor = pins(experiment_dir / THOR_LIST)
    wanted = pins(REQUIREMENTS)
    assert {name: thor.get(name) for name in wanted} == wanted
    # The torch of step 2 is the Thor's too, from the same index.
    setup = SETUP.read_text()
    assert thor["torch"] == "2.10.0+cu130" and "torch==2.10.0 " in setup
    assert thor["torchvision"] == "0.25.0+cu130" and "torchvision==0.25.0" in setup
    assert f'torch_index="{TORCH_INDEX}"' in setup
    assert thor["ultralytics"] == wanted["ultralytics"]
    assert "opencv-python" not in thor  # only the headless build
