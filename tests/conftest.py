"""Shared fixtures. Synthetic only — no real data of any kind."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from fixtures import builders

REPO_ROOT = Path(__file__).resolve().parents[1]


def sentinel_every_env_root(repo_root, monkeypatch, purpose: str) -> None:
    """setenv a sentinel for EVERY ${CLEFT_*} name the shipped configs
    reference -- one sentinel PER NAME, since collapsing two roots onto
    one fake path makes the same-artifact sweep compare hashes that
    never shared a path.

    [2026-08-24] Sweeps used to set CLEFT_SCUT_ROOT by name, so the
    first config referencing a NEW namespaced root (p15_survey's
    ${CLEFT_MEBEAUTY_ROOT}) broke every all-configs sweep at once.
    Scanning the configs makes them follow the namespace instead of a
    hand-maintained list. Lives here so test_smoke_run and test_ladder
    share one implementation.
    """
    import re

    names = set()
    for path in sorted((repo_root / "configs").glob("*.yaml")):
        names.update(
            re.findall(r"\$\{(CLEFT_[A-Z0-9_]+)\}", path.read_text(encoding="utf-8"))
        )
    assert names, "no namespaced roots found; the sentinel sweep checks nothing"
    for name in sorted(names):
        monkeypatch.setenv(name, f"/nonexistent/sentinel/for/{purpose}/{name}")


@pytest.fixture(autouse=True)
def _namespaced_roots_resolve(monkeypatch):
    """[2026-09-05] Every ``${CLEFT_*}`` a shipped config names gets a
    sentinel, for EVERY test rather than only the all-configs sweeps.

    Before the cohort paths were parameterised, a test loading one
    shipped config saw literal paths and needed nothing. Now it sees
    ``${CLEFT_PHOTOS}`` and ``load_config`` refuses -- correctly, since
    the whole point is that the value comes from the cluster.

    **Autouse is safe here because the names are read OFF THE CONFIGS.**
    The fixture cannot invent a variable, so a config naming an undefined
    one is not masked; that case is asserted directly by
    ``test_every_cohort_input_is_declared_by_role``. A test that wants a
    specific value still sets it -- monkeypatch applies in order, and the
    later setenv wins.
    """
    sentinel_every_env_root(REPO_ROOT, monkeypatch, "suite")


@pytest.fixture
def repo_root() -> Path:
    """The real cleft-aesthetics repo root. Read-only in tests."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def _repo_template(tmp_path_factory) -> Path:
    """Built once per session, then copied.

    ``make_git_repo`` costs six subprocess spawns (init, three configs, add,
    commit) and process creation is expensive on Windows. Roughly forty tests
    want a fresh repo, and copying ten small files is far cheaper than paying
    that six times over per test.

    Tests still get their own copy, because several mutate the repo -- dirtying
    it, committing to it, registering worktrees in ``.git/worktrees``. Sharing
    one live repo across tests would make them order-dependent.
    """
    return builders.make_git_repo(tmp_path_factory.mktemp("repo_template") / "repo")


def _clone_template(template: Path, destination: Path) -> Path:
    # A freshly initialised repo records no absolute paths in .git/config, so a
    # plain directory copy is a valid repo. Worktrees DO record absolute paths,
    # but those are only ever registered after the copy, in the test's own copy.
    #
    # [2026-08-24, a CI race] Git lock files are excluded: a transient lock can
    # vanish between copytree's listing and its copy (ENOENT on the runner --
    # .git/objects/maintenance.lock, from background maintenance the template
    # now disables at creation), and a lock that DOES copy is a stale lock in
    # the clone, telling git an operation is in progress that never was.
    shutil.copytree(
        template, destination, ignore=shutil.ignore_patterns("*.lock")
    )
    return destination


@pytest.fixture
def clean_repo(tmp_path: Path, _repo_template: Path) -> Path:
    """A synthetic git repo with one commit and a clean tree."""
    return _clone_template(_repo_template, tmp_path / "clean_repo")


@pytest.fixture
def dirty_repo(tmp_path: Path, _repo_template: Path) -> Path:
    """A synthetic git repo with an uncommitted file."""
    repo = _clone_template(_repo_template, tmp_path / "dirty_repo")
    return builders.dirty(repo, "untracked")


@pytest.fixture
def out_root(tmp_path: Path) -> Path:
    return tmp_path / "runs"


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """A fake immutable input artifact directory."""
    return builders.make_data_dir(tmp_path / "artifact")


@pytest.fixture
def write_config(tmp_path: Path):
    """Factory writing a valid Phase 0 config into tmp_path."""

    def _write(name: str = "cfg.yaml", **overrides) -> Path:
        return builders.write_config(tmp_path / name, **overrides)

    return _write
