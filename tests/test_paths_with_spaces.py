"""Every real input path contains spaces.

    the score sheet workbook and the cropped photo directory, whose real
    names carry spaces (declared as ${CLEFT_SCORESHEET} / ${CLEFT_PHOTOS})

A path with a space is the classic thing that works on a developer's tidy
fixture tree and fails on the real data, usually inside a shell interpolation or
an unquoted subprocess argument. None of the existing fixtures contain one, so
this file exercises the whole chain — config, hashing, guards, run directory —
with spaces throughout.
"""

from __future__ import annotations

import json

import pytest

from cleft.config import load_config
from cleft.data import scoresheet as S
from cleft.provenance import RunContext, hash_dir, hash_file

from fixtures import builders, cohort as C

# [UPDATED 2026-09-05] Neutral fixture names. What these exercise is a
# path CONTAINING SPACES, never this cohort in particular -- and the
# real names are no longer written down anywhere
# (config.schema.COHORT_ENV_REFERENCES).
SPACED_DIR = "238 Cropped Anonymised Photos"
SPACED_FILE = "Frontal AP Scores - sheet 1.xlsx"


@pytest.fixture
def spaced_artifact(tmp_path):
    root = tmp_path / "bch" / SPACED_DIR
    root.mkdir(parents=True)
    (root / "1").mkdir()
    (root / "1" / "1001 frontal.jpg").write_bytes(b"a")
    (root / "1" / "1002 basal.jpg").write_bytes(b"b")
    return root


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------


def test_hash_dir_walks_a_path_with_spaces(spaced_artifact):
    result = hash_dir(spaced_artifact)
    assert result["file_count"] == 2
    assert sorted(result["files"]) == ["1/1001 frontal.jpg", "1/1002 basal.jpg"]


def test_relative_paths_with_spaces_stay_forward_slashed(spaced_artifact):
    """The same artifact must hash identically on Windows and on the cluster."""
    assert all("\\" not in rel for rel in hash_dir(spaced_artifact)["files"])


def test_hash_file_handles_spaces(tmp_path):
    path = tmp_path / SPACED_FILE
    path.write_bytes(b"workbook bytes")
    assert len(hash_file(path)) == 64


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def test_a_config_can_declare_a_path_containing_spaces(tmp_path, spaced_artifact):
    config = builders.write_config(
        tmp_path / "spaced.yaml",
        inputs=[
            {
                "name": "photos",
                "path": str(spaced_artifact),
                "rollup_sha256": hash_dir(spaced_artifact)["rollup"],
            }
        ],
    )
    loaded = load_config(config)
    assert loaded["inputs"][0]["path"] == str(spaced_artifact)
    assert " " in loaded["inputs"][0]["path"]


def test_guard_three_accepts_a_spaced_path(tmp_path, clean_repo, out_root, spaced_artifact):
    config = builders.write_config(
        tmp_path / "spaced.yaml",
        tier="keeper",
        inputs=[
            {
                "name": "photos",
                "path": str(spaced_artifact),
                "rollup_sha256": hash_dir(spaced_artifact)["rollup"],
            }
        ],
    )
    with RunContext(config, out_root, repo_root=clean_repo) as ctx:
        recorded = json.loads((ctx.run_dir / "inputs.json").read_text(encoding="utf-8"))

    assert " " in recorded["inputs"][0]["path"]


def test_guard_three_still_catches_corruption_behind_a_spaced_path(
    tmp_path, clean_repo, out_root, spaced_artifact
):
    """The space must not become a hole in the hash check."""
    from cleft.provenance import GuardError

    config = builders.write_config(
        tmp_path / "spaced.yaml",
        tier="keeper",
        inputs=[
            {
                "name": "photos",
                "path": str(spaced_artifact),
                "rollup_sha256": hash_dir(spaced_artifact)["rollup"],
            }
        ],
    )
    (spaced_artifact / "1" / "1001 frontal.jpg").write_bytes(b"tampered")

    with pytest.raises(GuardError, match="hash"):
        RunContext(config, out_root, repo_root=clean_repo)


# --------------------------------------------------------------------------
# the loader
# --------------------------------------------------------------------------


def test_the_scoresheet_loads_from_a_spaced_filename(tmp_path):
    built = C.build_cohort(tmp_path / "cohort")
    path = C.write_scoresheet(built, tmp_path / "bch" / SPACED_FILE, form="integer")
    assert " " in path.name
    assert len(S.load(path).rows) == 251


# --------------------------------------------------------------------------
# the run directory
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# platform-neutral absolute paths
# --------------------------------------------------------------------------


def test_a_posix_absolute_path_is_absolute_on_every_platform():
    """The cluster paths are POSIX-absolute and this runs on Windows too.

    ``Path("/home/...").is_absolute()`` is False on Windows, which would re-root
    a cluster input under the repo and make guard 3 hash the wrong thing.
    """
    from cleft.provenance.context import is_absolute_path

    # A POSIX path with spaces, of the shape the cohort roots resolve to.
    assert is_absolute_path("/data/cohort/238 Cropped Anonymised Photos")
    assert is_absolute_path("C:/Users/example/data")
    assert is_absolute_path("C:\\Users\\example\\data")
    assert not is_absolute_path("data/smoke/v1")
    assert not is_absolute_path("configs/smoke.yaml")


def test_a_posix_absolute_input_is_not_re_rooted_under_the_repo(
    tmp_path, clean_repo, out_root
):
    """The regression: the guard must report the declared path, not repo/declared."""
    from cleft.provenance import GuardError

    declared = "/data/cohort/238 Cropped Anonymised Photos"
    config = builders.write_config(
        tmp_path / "cluster.yaml",
        tier="keeper",
        inputs=[{"name": "photos", "path": declared, "rollup_sha256": "0" * 64}],
    )

    with pytest.raises(GuardError) as excinfo:
        RunContext(config, out_root, repo_root=clean_repo)

    message = str(excinfo.value)
    assert declared in message.replace("\\", "/")
    assert str(clean_repo) not in message, (
        "a POSIX-absolute cluster path was re-rooted under the repo"
    )


def test_declare_inputs_resolves_paths_the_same_way_the_run_does():
    """**The regression for a SECOND resolver [2026-07-30].**

    ``scripts/declare_inputs.py`` had its own resolution:
    ``Path(entry["path"]).is_absolute()``, joining onto the repo root otherwise.
    That is the platform-NATIVE predicate where the run uses the
    platform-NEUTRAL one, so on Linux a laptop path became
    ``/home/user/codex/cleft-aesthetics/C:/data/...`` and was reported as MISSING
    -- a malformed declaration presenting as absent data.

    **``is_absolute_path`` was correct and tested all along.** The defect was
    having two implementations, which is the same shape as the two mask
    implementations that would have let the domains diverge. Asserted by reading
    the script's source, because a duplicate that is reintroduced would pass any
    behavioural test written against the shared helper.
    """
    import re
    from pathlib import Path as _Path

    source = (
        _Path(__file__).resolve().parents[1] / "scripts" / "declare_inputs.py"
    ).read_text(encoding="utf-8")

    assert "resolve_declared_path" in source, (
        "declare_inputs.py must use the run's resolver, not its own"
    )
    # No platform-native absoluteness check outside a comment.
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    assert not re.search(r"\.is_absolute\(\)", code), (
        "declare_inputs.py has a platform-native is_absolute() check again; on "
        "Linux that treats C:/... as relative and concatenates it onto the repo"
    )


def test_declare_inputs_echoes_the_declaration_not_the_resolution():
    """The paste block must echo ``${CLEFT_SCUT_ROOT}``. Printing the resolved
    path would invite pasting a machine-specific path back into a config whose
    point is that it holds none -- the tool handing you the defect it was fixed to
    avoid, wearing the tool's own authority."""
    import sys
    from pathlib import Path as _Path

    scripts = _Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        import declare_inputs

        assert (
            declare_inputs._as_declared(
                {"path": "/resolved/abs", "path_declared_as": "${CLEFT_SCUT_ROOT}"}
            )
            == "${CLEFT_SCUT_ROOT}"
        )
        # A plain entry has no declaration to prefer, so it echoes its path.
        assert declare_inputs._as_declared({"path": "data/smoke/v1"}) == "data/smoke/v1"
    finally:
        sys.path.remove(str(scripts))


def test_a_drive_letter_path_is_not_re_rooted_by_the_shared_resolver():
    """The other half of the reported defect, at the resolver. A drive-lettered
    path must come back as itself on EVERY platform -- on Linux the native check
    would call it relative."""
    from pathlib import Path as _Path

    from cleft.provenance.context import resolve_declared_path

    repo = _Path("/home/user/codex/cleft-aesthetics")
    resolved = resolve_declared_path("C:/data/scut-fbp5500/SCUT-FBP5500_v2", repo)
    assert str(resolved).replace("\\", "/").startswith("C:/data/")
    assert "cleft-aesthetics" not in str(resolved), (
        "a drive-lettered path was concatenated onto the repo root"
    )


def test_an_absolute_input_still_hashes_correctly(
    tmp_path, clean_repo, out_root, spaced_artifact
):
    """tmp_path is absolute on both platforms, so this exercises the same branch."""
    config = builders.write_config(
        tmp_path / "abs.yaml",
        tier="keeper",
        inputs=[
            {
                "name": "photos",
                "path": str(spaced_artifact),
                "rollup_sha256": hash_dir(spaced_artifact)["rollup"],
            }
        ],
    )
    with RunContext(config, out_root, repo_root=clean_repo) as ctx:
        recorded = json.loads((ctx.run_dir / "inputs.json").read_text(encoding="utf-8"))
    assert recorded["inputs"][0]["path"] == str(spaced_artifact)


def test_a_run_directory_can_live_under_a_spaced_root(
    tmp_path, clean_repo, write_config
):
    """`--out` may point somewhere with a space, and the worktree must survive it."""
    out = tmp_path / "codex runs"
    config = write_config(tier="keeper")
    with RunContext(config, out, repo_root=clean_repo) as ctx:
        assert " " in str(ctx.run_dir)
        assert (ctx.run_dir / "code").is_dir(), (
            "git worktree add must handle a destination containing a space"
        )
        builders.git(ctx.run_dir / "code", "rev-parse", "HEAD")
