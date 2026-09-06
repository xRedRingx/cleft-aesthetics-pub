"""Git facts, captured rather than assumed.

The probe has THREE outcomes, and conflating any two of them corrupts the
provenance record:

``CLEAN``
    The tree matches HEAD. The run may be a keeper.

``DIRTY``
    There are uncommitted or untracked changes. A known state, and a legitimate
    demotion to ``runs/dev/``.

``UNAVAILABLE``
    Git did not answer -- it is missing, the directory is not a repository, there
    are no commits yet, or the command failed. The code state is UNKNOWN, which is
    strictly worse than dirty, and the run must not proceed at either tier.

This module used to return ``dirty=True`` whenever git failed. On the cluster
that turned "detected dubious ownership" -- the NFS checkout is owned by
``nobody`` and the container runs as root -- into "the tree is dirty", on a tree
that was provably clean. The recorded provenance was wrong, and it could have
been wrong in the other direction too: a genuinely dirty tree whose ``git status``
failed would have been demoted or aborted for a reason that had nothing to do
with its actual state.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

#: Used in a run id when there is no resolvable SHA. Deliberately not a hex
#: string, so it can never be mistaken for one in a directory listing.
NO_SHA = "nogitsha"


class GitState(str, Enum):
    """Subclasses ``str`` so it serialises into env.json as a plain word."""

    CLEAN = "clean"
    DIRTY = "dirty"
    UNAVAILABLE = "unavailable"


class Failure(str, Enum):
    GIT_MISSING = "git_missing"
    NOT_A_REPO = "not_a_repo"
    NO_COMMITS = "no_commits"
    DUBIOUS_OWNERSHIP = "dubious_ownership"
    GIT_ERROR = "git_error"


@dataclass(frozen=True)
class GitInfo:
    state: GitState
    sha: str | None = None
    branch: str | None = None
    failure: Failure | None = None
    error: str | None = None
    repo: str | None = None

    @property
    def available(self) -> bool:
        return self.state is not GitState.UNAVAILABLE

    @property
    def dirty(self) -> bool:
        """True only for a tree git actually reported as dirty.

        Never true merely because git failed -- that is ``UNAVAILABLE``.
        """
        return self.state is GitState.DIRTY

    @property
    def sha8(self) -> str:
        return self.sha[:8] if self.sha else NO_SHA

    @property
    def remedy(self) -> str | None:
        """What to do about it, for the failure kinds with a known fix."""
        repo = self.repo or "<repo>"
        if self.failure is Failure.DUBIOUS_OWNERSHIP:
            return (
                "The repository is owned by a different user than the one running\n"
                "  this process. On the cluster the NFS checkout is owned by 'nobody'\n"
                "  and the container runs as root, so this recurs on EVERY fresh\n"
                "  container and has to be set inside the job rather than once by hand:\n"
                "\n"
                "      export GIT_CONFIG_GLOBAL=/tmp/gitconfig\n"
                f"      git config --global --add safe.directory {repo}\n"
                "\n"
                "  scripts/entrypoint.sh already does this; a job that bypasses the\n"
                "  entrypoint must do it itself. Do NOT chown the NFS checkout."
            )
        if self.failure is Failure.NO_COMMITS:
            return (
                "The repository has no commits, so there is no SHA to record.\n"
                "  Commit before running: a result whose code state cannot be named\n"
                "  is not reproducible at either tier."
            )
        if self.failure is Failure.NOT_A_REPO:
            return (
                f"  {repo} is not a git repository.\n"
                "  Set CLEFT_REPO_ROOT to the checkout, or run from inside it."
            )
        if self.failure is Failure.GIT_MISSING:
            return (
                "git is not installed here. The provenance module needs it at\n"
                "  runtime, not only at build time -- docker/Dockerfile installs it\n"
                "  deliberately for this reason."
            )
        return None


def _run(repo: Path, *args: str) -> tuple[int, str, str]:
    """Run git, returning (returncode, stdout, stderr). Never raises on failure.

    Deliberately not ``check=True``: a non-zero exit is information to classify,
    not an exception to swallow into a default.
    """
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _classify(stderr: str, repo: Path) -> GitInfo:
    lowered = stderr.lower()
    if "dubious ownership" in lowered:
        failure = Failure.DUBIOUS_OWNERSHIP
    elif "not a git repository" in lowered:
        failure = Failure.NOT_A_REPO
    elif "unknown revision" in lowered or "ambiguous argument 'head'" in lowered:
        failure = Failure.NO_COMMITS
    else:
        failure = Failure.GIT_ERROR
    return GitInfo(
        state=GitState.UNAVAILABLE,
        failure=failure,
        error=stderr or "git exited non-zero without a message",
        repo=str(repo),
    )


def inspect_repo(repo: str | Path) -> GitInfo:
    """Probe ``repo`` and return exactly one of the three states."""
    repo = Path(repo)
    try:
        code, out, err = _run(repo, "rev-parse", "--is-inside-work-tree")
    except FileNotFoundError:
        return GitInfo(
            state=GitState.UNAVAILABLE,
            failure=Failure.GIT_MISSING,
            error="git executable not found on PATH",
            repo=str(repo),
        )

    if code != 0:
        return _classify(err, repo)
    if out != "true":
        return GitInfo(
            state=GitState.UNAVAILABLE,
            failure=Failure.NOT_A_REPO,
            error=f"git reports this is not a work tree (rev-parse said {out!r})",
            repo=str(repo),
        )

    code, sha, err = _run(repo, "rev-parse", "HEAD")
    if code != 0:
        return _classify(err, repo)

    # ``--porcelain`` lists modified tracked files and untracked files alike.
    # Untracked code is uncommitted code, so both count as dirty.
    code, status, err = _run(repo, "status", "--porcelain")
    if code != 0:
        return _classify(err, repo)

    # A detached HEAD makes this fail harmlessly; the SHA identifies the code and
    # the branch is only a convenience, so it is not worth aborting over.
    branch_code, branch, _ = _run(repo, "rev-parse", "--abbrev-ref", "HEAD")

    return GitInfo(
        state=GitState.DIRTY if status else GitState.CLEAN,
        sha=sha,
        branch=branch if branch_code == 0 else None,
        repo=str(repo),
    )


def add_worktree(repo: str | Path, sha: str, dest: str | Path) -> None:
    """Check out ``sha`` into ``dest`` as a locked, detached worktree.

    ``dest`` must not exist. The lock stops ``git worktree prune`` from removing
    the archived copy beside a result if the run directory is ever moved.
    """
    repo, dest = Path(repo), Path(dest)
    if dest.exists():
        raise FileExistsError(f"worktree destination already exists: {dest}")
    code, _, err = _run(
        repo,
        "worktree",
        "add",
        "--detach",
        "--lock",
        "--reason",
        "archived beside a run result",
        str(dest),
        sha,
    )
    if code != 0:
        raise RuntimeError(f"git worktree add failed: {err}")
