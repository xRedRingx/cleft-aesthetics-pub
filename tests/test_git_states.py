"""The three git states, and the difference between "dirty" and "no answer".

Regression suite for a real cluster failure: the NFS checkout is owned by
``nobody`` and the container runs as root, so ``git status`` exited non-zero with
"detected dubious ownership". The old probe returned ``dirty=True`` for any git
failure, so a provably clean tree was recorded as dirty and demoted.

The same bug ran the other way too: a genuinely dirty tree whose git command
failed would have been reported with whatever the failure implied. Either way the
provenance record was wrong, which is the one thing this module exists to prevent.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from cleft.provenance import GuardError, RunContext, gitinfo
from cleft.provenance.gitinfo import Failure, GitState, inspect_repo

from fixtures import builders

# Real stderr from the cluster, 2026-07-27.
DUBIOUS_STDERR = (
    "fatal: detected dubious ownership in repository at "
    "'/home/user/codex/cleft-aesthetics'\n"
    "To add an exception for this directory, call:\n"
    "\n"
    "\tgit config --global --add safe.directory /home/user/codex/cleft-aesthetics"
)


@pytest.fixture
def fail_git(monkeypatch):
    """Make every git invocation fail with the given stderr."""

    def _install(stderr: str, returncode: int = 128):
        def fake_run(repo, *args):
            return returncode, "", stderr

        monkeypatch.setattr(gitinfo, "_run", fake_run)

    return _install


# --------------------------------------------------------------------------
# the probe
# --------------------------------------------------------------------------


def test_clean_repo_is_clean(clean_repo):
    info = inspect_repo(clean_repo)
    assert info.state is GitState.CLEAN
    assert info.dirty is False
    assert info.available is True
    assert info.sha and info.failure is None


def test_dirty_repo_is_dirty(dirty_repo):
    info = inspect_repo(dirty_repo)
    assert info.state is GitState.DIRTY
    assert info.dirty is True
    assert info.available is True
    assert info.sha, "a dirty tree still has a resolvable HEAD"


def test_non_repository_is_unavailable_not_dirty(tmp_path):
    plain = tmp_path / "not_a_repo"
    plain.mkdir()
    info = inspect_repo(plain)
    assert info.state is GitState.UNAVAILABLE
    assert info.dirty is False, "a missing repository is not a dirty tree"
    assert info.failure is Failure.NOT_A_REPO


def test_repository_without_commits_is_unavailable_not_dirty(tmp_path):
    """`git init` with nothing committed has no SHA, so provenance is unknown."""
    fresh = tmp_path / "unborn"
    fresh.mkdir()
    subprocess.run(["git", "-C", str(fresh), "init", "-q", "-b", "main"], check=True)
    info = inspect_repo(fresh)
    assert info.state is GitState.UNAVAILABLE
    assert info.dirty is False
    assert info.failure is Failure.NO_COMMITS


def test_dubious_ownership_is_classified(clean_repo, fail_git):
    fail_git(DUBIOUS_STDERR)
    info = inspect_repo(clean_repo)
    assert info.state is GitState.UNAVAILABLE
    assert info.failure is Failure.DUBIOUS_OWNERSHIP
    assert info.dirty is False, (
        "the exact cluster bug: a clean tree reported as dirty because git failed"
    )


def test_missing_git_binary_is_classified(clean_repo, monkeypatch):
    def explode(repo, *args):
        raise FileNotFoundError("git")

    monkeypatch.setattr(gitinfo, "_run", explode)
    info = inspect_repo(clean_repo)
    assert info.state is GitState.UNAVAILABLE
    assert info.failure is Failure.GIT_MISSING
    assert info.dirty is False


def test_unrecognised_failure_is_still_unavailable(clean_repo, fail_git):
    fail_git("fatal: something nobody has seen before")
    info = inspect_repo(clean_repo)
    assert info.state is GitState.UNAVAILABLE
    assert info.failure is Failure.GIT_ERROR
    assert info.dirty is False, "an unknown git failure must never default to dirty"


# --------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------


def test_unavailable_git_aborts_and_names_the_git_problem(
    write_config, out_root, clean_repo, fail_git
):
    """The message must blame git, not the tree. This is the regression."""
    fail_git(DUBIOUS_STDERR)
    cfg = write_config(tier="keeper")

    with pytest.raises(GuardError) as excinfo:
        RunContext(cfg, out_root, repo_root=clean_repo)

    message = str(excinfo.value)
    assert "git did not answer" in message
    assert "dubious_ownership" in message
    assert "dubious ownership" in message, "git's own words must be surfaced"
    assert "NOT a dirty tree" in message


def test_unavailable_git_message_carries_the_safe_directory_remedy(
    write_config, out_root, clean_repo, fail_git
):
    """It recurs on every fresh container, so the fix belongs in the error."""
    fail_git(DUBIOUS_STDERR)
    cfg = write_config(tier="keeper")

    with pytest.raises(GuardError) as excinfo:
        RunContext(cfg, out_root, repo_root=clean_repo)

    message = str(excinfo.value)
    assert "GIT_CONFIG_GLOBAL" in message
    assert "safe.directory" in message
    assert "entrypoint.sh" in message
    assert "chown" in message, "must say what not to do as well as what to do"


def test_unavailable_git_aborts_dev_runs_too(
    write_config, out_root, clean_repo, fail_git
):
    """Unknown provenance is worse than dirty, so it does not merely demote."""
    fail_git(DUBIOUS_STDERR)
    cfg = write_config(tier="dev")

    with pytest.raises(GuardError, match="git did not answer"):
        RunContext(cfg, out_root, repo_root=clean_repo)


def test_dirty_message_does_not_blame_git(write_config, out_root, dirty_repo):
    """The converse: a real dirty tree must not read as a git malfunction."""
    cfg = write_config(tier="keeper")

    with pytest.raises(GuardError) as excinfo:
        RunContext(cfg, out_root, repo_root=dirty_repo)

    message = str(excinfo.value)
    assert "dirty tree" in message
    assert "git did not answer" not in message
    assert "git answered" in message


def test_dirty_still_demotes_to_dev(write_config, out_root, dirty_repo):
    cfg = write_config(tier="dev")
    with RunContext(cfg, out_root, repo_root=dirty_repo) as ctx:
        assert ctx.tier == "dev"
        env = json.loads((ctx.run_dir / "env.json").read_text(encoding="utf-8"))

    assert env["git"]["state"] == "dirty"
    assert env["git"]["dirty_tree"] is True


def test_env_json_records_the_state_word(write_config, out_root, clean_repo):
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        env = json.loads((ctx.run_dir / "env.json").read_text(encoding="utf-8"))

    assert env["git"]["state"] == "clean"
    assert env["git"]["dirty_tree"] is False
    assert env["git"]["available"] is True


def test_plan_run_dir_also_refuses_an_unanswerable_git(
    write_config, out_root, clean_repo, fail_git
):
    fail_git(DUBIOUS_STDERR)
    cfg = write_config(tier="keeper")
    with pytest.raises(GuardError, match="git did not answer"):
        RunContext.plan_run_dir(cfg, out_root, repo_root=clean_repo)


def test_a_clean_tree_is_never_demoted_by_a_git_hiccup(
    write_config, out_root, clean_repo
):
    """The positive case the cluster bug destroyed: clean stays keeper."""
    cfg = write_config(tier="keeper")
    with RunContext(cfg, out_root, repo_root=clean_repo) as ctx:
        assert ctx.tier == "keeper"
        assert ctx.run_dir.parent.parent.name == "keeper"
        assert (ctx.run_dir / "code").is_dir()

    builders.git(clean_repo, "worktree", "prune")


# --------------------------------------------------------------------------
# 2026-08-24: fixture repos must be INERT (a CI copytree race)
# --------------------------------------------------------------------------


def test_fixture_repos_disable_background_maintenance_and_clones_inherit_it(
    tmp_path,
):
    """Regression for a CI fixture-setup error: `git commit` detached a
    background maintenance process in the session-scoped template, and its
    transient .git/objects/maintenance.lock vanished mid-copytree (ENOENT).
    A fixture repo must be inert -- measured here from the built repo, not
    reasoned from the builder's source -- and every copy inherits the
    settings because they live in .git/config."""
    repo = builders.make_git_repo(tmp_path / "inert")
    assert builders.git(repo, "config", "gc.auto") == "0"
    assert builders.git(repo, "config", "maintenance.auto") == "false"

    from conftest import _clone_template

    clone = _clone_template(repo, tmp_path / "inert_clone")
    assert builders.git(clone, "config", "gc.auto") == "0"
    assert builders.git(clone, "config", "maintenance.auto") == "false"


def test_cloning_a_template_leaves_git_lock_files_behind(tmp_path):
    """The copy side of the same regression: a lock present at copy time
    must not reach the clone -- a copied lock is a STALE lock, telling git
    an operation is in progress that never was -- and ignoring locks also
    immunises the copy against one vanishing mid-walk."""
    repo = builders.make_git_repo(tmp_path / "locked")
    lock = repo / ".git" / "objects" / "maintenance.lock"
    lock.write_text("transient\n", encoding="utf-8")
    (repo / ".git" / "index.lock").write_text("transient\n", encoding="utf-8")

    from conftest import _clone_template

    clone = _clone_template(repo, tmp_path / "locked_clone")
    assert not (clone / ".git" / "objects" / "maintenance.lock").exists()
    assert not (clone / ".git" / "index.lock").exists()
    # The clone is still a working repo with the fixture commit.
    assert builders.git(clone, "log", "--format=%s", "-1") == "fixture commit"
