"""Synthetic fixture builders.

Nothing here touches real data. The clinical cohort never leaves EHU
infrastructure and Phase 0 never reads it; every fixture below is generated
from a fixed seed so its hashes are stable across machines.

Fixtures are built programmatically rather than committed as files because a
committed ``.git`` directory is not portable and a committed binary fixture
cannot be audited by reading the diff.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import yaml

# --------------------------------------------------------------------------
# git
# --------------------------------------------------------------------------


#: Identity supplied through the environment rather than three `git config`
#: calls per fixture. The runner may have no global config, and on Windows each
#: subprocess spawn costs enough that halving the count is worth it across the
#: ~90 repositories this suite builds.
_GIT_ENV = {
    "GIT_AUTHOR_NAME": "Fixture",
    "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
    "GIT_COMMITTER_NAME": "Fixture",
    "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
}


def git(repo: Path, *args: str) -> str:
    """Run git in ``repo`` and return stdout. Raises on non-zero exit."""
    proc = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **_GIT_ENV},
    )
    return proc.stdout.strip()


def make_git_repo(root: Path, files: dict[str, str] | None = None) -> Path:
    """Create a tiny real git repo with exactly one commit, clean tree.

    Synthetic throughout — this is never the project repository.
    """
    root.mkdir(parents=True, exist_ok=True)
    files = files or {"README.md": "fixture repo\n", "src/thing.py": "VALUE = 1\n"}
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    git(root, "init", "-q", "-b", "main")
    # [2026-08-24, a CI race] `git commit` may DETACH a background
    # maintenance/auto-gc process that keeps writing transient lock files
    # in .git after this builder returns -- and the session-scoped
    # template is copytree'd by ~40 tests, so a lock that vanishes
    # mid-copy fails the copy (.git/objects/maintenance.lock, ENOENT on
    # the runner). A fixture repo must be INERT: the settings are written
    # into .git/config directly (zero subprocess spawns, the count this
    # module deliberately minimises) and are inherited by every copy.
    with (root / ".git" / "config").open("a", encoding="utf-8") as config:
        config.write("[gc]\n\tauto = 0\n[maintenance]\n\tauto = false\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "fixture commit")
    return root


def dirty(repo: Path, mode: str = "untracked") -> Path:
    """Make a clean fixture repo dirty.

    ``untracked`` adds an unversioned file; ``modified`` edits a tracked file.
    Both must read as dirty: uncommitted code is uncommitted code either way.
    """
    if mode == "untracked":
        (repo / "scratch_note.txt").write_text("not committed\n", encoding="utf-8")
    elif mode == "modified":
        (repo / "src" / "thing.py").write_text("VALUE = 2\n", encoding="utf-8")
    else:  # pragma: no cover - programmer error
        raise ValueError(f"unknown dirty mode: {mode}")
    return repo


# --------------------------------------------------------------------------
# data artifacts
# --------------------------------------------------------------------------


def make_data_dir(root: Path, n_files: int = 3, seed: int = 20260726) -> Path:
    """A fake immutable data artifact directory with deterministic contents."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n_files):
        payload = rng.integers(0, 256, size=32, dtype=np.uint8).tobytes()
        (root / f"item_{i:02d}.bin").write_bytes(payload)
    nested = root / "nested"
    nested.mkdir(exist_ok=True)
    (nested / "note.txt").write_text("nested content\n", encoding="utf-8")
    return root


# --------------------------------------------------------------------------
# configs
# --------------------------------------------------------------------------


def config_dict(**overrides) -> dict:
    """A minimal valid Phase 0 config. Overrides are shallow-merged.

    **[2026-09-01] The inputs a task REFERENCES are declared for it**,
    unless the caller supplies its own ``inputs``. ``schema.validate``
    now refuses a task naming an input the config does not declare -- a
    check added after six Phase 22 configs shipped unloadable -- and a
    scaffold that omits the inputs its task names would fail on that
    rather than on the rule it was written to exercise. Derived from the
    task, so these fixtures cannot drift from what they test.
    """
    cfg = {
        "schema_version": 1,
        "phase": "p0",
        "tier": "dev",
        "seed": 1337,
        "inputs": [],
        "task": {"kind": "smoke", "n_samples": 16},
    }
    cfg.update(overrides)
    if "inputs" not in overrides:
        from cleft.config.schema import _input_references

        cfg["inputs"] = [
            {"name": name, "path": f"/tmp/{name}", "rollup_sha256": "ab" * 32}
            for name in sorted(
                {value for _, value in _input_references(cfg["task"])}
            )
        ]
    return cfg


def write_config(path: Path, **overrides) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config_dict(**overrides), sort_keys=True),
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------
# metric fixtures
# --------------------------------------------------------------------------


def graded_pair(n: int = 120, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Truth/prediction pair on the raw 1-5 aesthetic scale, correlated but not equal.

    Spans all three reporting classes so the 3-class collapse is exercised.
    """
    rng = np.random.default_rng(seed)
    truth = rng.uniform(1.0, 5.0, size=n)
    pred = np.clip(truth + rng.normal(0.0, 0.8, size=n), 1.0, 5.0)
    return truth, pred


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    return path
