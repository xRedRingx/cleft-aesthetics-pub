"""The freeze, made mechanical.

``phase3-freeze`` (``eef8f56``) froze the **measurement apparatus** -- the harness,
the gates, the metrics, fold handling and provenance. Phase 4 and everything after
it add *arms*, which is additive and permitted. Changing the apparatus is not:
every result recorded before the change is invalidated by it, including the seed
band that all of Phase 7's claims are measured against.

That rule was a sentence in a brief, which is the same kind of thing as "remember
to commit before running" -- and the previous project demonstrated exactly how
well those hold. So it is a test. If a frozen module changes, the suite fails and
names it, and touching the apparatus becomes a deliberate act with a visible
consequence instead of an accident nobody notices until the numbers disagree.

**This module deliberately does its own hashing.** Reusing
``provenance.hashing`` would mean the guard depended on one of the modules it
guards -- a change there could silence the test that exists to catch it. Ten lines
of duplication buys independence, which is R7: a verifier passing means only that
the thing it tests is right, and it cannot test itself.

**If this test fails and the change was intended:** the change is an escalation,
not a fix. Everything measured before it is re-run. Update the digests below only
as part of that decision, never to make the suite green again.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

# --------------------------------------------------------------------------
# the frozen set
# --------------------------------------------------------------------------

#: Individual modules, path -> SHA-256 of their bytes at ``phase3-freeze``.
#:
#: The brief names five: the harness, the gates, the metrics, fold handling and
#: provenance. Two more are here because the results depend on them just as
#: directly:
#:
#: ``train/determinism.py``
#:     Gate 1 is a statement about what this file configures. If it changes, the
#:     byte-identical result recorded in ``GATE1_REFERENCE`` is a statement about
#:     a different configuration.
#: ``clutter.py``
#:     Decides which files enter a rollup, so it decides what every input hash in
#:     every config actually means. ``provenance/hashing.py`` is frozen and imports
#:     it; freezing the importer and not the imported would be a gap.
FROZEN_FILES = {
    "src/cleft/train/harness.py":
        "4cb39048eb3b468c06cf1df3e636c98047e3765f27d536218bcd53877241df68",
    "src/cleft/train/gates.py":
        "05123860014790912a7bef79ece06f5c08439e7f182d599a7b1988eba5ff4acc",
    "src/cleft/train/determinism.py":
        "35913f4de61f3bcaea1b3ee95b8bba32133faa3e4e525bb701b22a020279f603",
    "src/cleft/eval/metrics.py":
        "4cb1cf26bd15067b4af9ea59e6de14e0c35c1a0572f634017a34cd063321a50b",
    "src/cleft/data/folds.py":
        "a36e14658e5a65421ff35c5588fc5f4dbdd2268d153d932814edf3e24e9bc687",
    "src/cleft/clutter.py":
        "df27d892efd0f3e65c0307c83dce801f3e2ccb5c15a18684c3640c0951b3b347",
    # ---- added 2026-07-28, before Phase 5 -------------------------------
    # **The frozen set covered the measurement apparatus but not the code that
    # produced the data every arm consumes.** These three staged `staged_v1`:
    # `trapezium.py` defines the mask, `staging.py` does pad-square-then-resize,
    # `mapping.py` places every box in each image's own pixels.
    #
    # Changing them would leave the artifact's hash on disk unchanged while
    # making it **no longer reproducible from the code that claims to produce
    # it** -- a divergence guard 3 cannot see, because guard 3 hashes the data,
    # not the generator.
    #
    # Phase 5 stages SCUT through the same geometry. The rule is SCUT-specific
    # entry points that CALL these, never edits to them; this makes that
    # mechanical rather than a matter of discipline. If something genuinely
    # cannot be done without modifying one, that is an escalation to surface,
    # not to work around.
    "src/cleft/geometry/staging.py":
        "1b4ca78fa247cf229ab1d20de1bbea2beea6a7edf4012604340a81afe895e56e",
    "src/cleft/geometry/trapezium.py":
        "04eeb17ee5c9b36d358c968346449ad9406fbbe707b5f4c32531bb8d2193d6b5",
    "src/cleft/geometry/mapping.py":
        "42758f0e96a6b7f9afc882c3527c367e123bc7e5effbbbd0d6587788cc774344",
}

#: Data-producing modules with the **same property** as the three above, not yet
#: frozen. Recorded rather than left implicit, because "the frozen set is
#: complete" is exactly the assumption that made the gap above.
#:
#: * ``geometry/stage_build.py`` -- built the artifact directory itself.
#: * ``geometry/patches.py`` and ``geometry/generators.py`` -- define the boxes
#:   recorded in ``staged_v1/patches.json``.
#:
#: They are left out **deliberately**: Phase 7's patch work is expected to extend
#: the generators, and freezing them now would guarantee an escalation rather
#: than prevent an accident. The anatomy scheme is separately pinned by
#: ``generators.ACCEPTED_ANATOMY`` and its regression test. **The grid generator
#: is not pinned by anything**, which is the live gap in this list.
NOT_YET_FROZEN = (
    "src/cleft/geometry/stage_build.py",
    "src/cleft/geometry/patches.py",
    "src/cleft/geometry/generators.py",
)

#: Whole packages, because a per-file list cannot see a file being **added**.
#: ``provenance/`` is the contract every run directory is written through; a new
#: module appearing inside it is a change to the apparatus even though no
#: existing file moved.
FROZEN_TREES = {
    "src/cleft/provenance": {
        "rollup": "68b75958fa4a39439ac6c701124bc5f75c171b0c2564b79887ece966092e8bfe",
        "members": 4,
    },
    "src/cleft/eval": {
        "rollup": "cc823c62239beded0e5b03d95901d0c56c62752d9aad8727174fbf3d412f6b5b",
        "members": 2,
    },
}

FREEZE_TAG = "phase3-freeze"
FREEZE_COMMIT = "eef8f56"

CONSEQUENCE = (
    f"\n\nThis module was frozen at {FREEZE_TAG} ({FREEZE_COMMIT}). Changing it "
    "invalidates every result recorded before the change -- gate 1's "
    "byte-identical determinism runs, the ten-seed band in PLAN §4.12, and every "
    "delta measured against that band.\n\n"
    "If the change was NOT intended, revert it.\n"
    "If it was, that is a decision to escalate: everything measured before it is "
    "re-run, and the digest here is updated as part of that -- never on its own "
    "to make the suite green.\n\n"
    "If you came here because a COMMENT inside a frozen module looks wrong, do "
    "not change it: see docs/FROZEN_KNOWN_STALE.md, which carries the correction "
    "without touching the module."
)

#: Prose inside frozen modules that is known wrong and cannot be corrected in
#: place. Each entry is registered in ``docs/FROZEN_KNOWN_STALE.md`` with the
#: correction and where the right answer lives.
KNOWN_STALE = {
    "src/cleft/eval/metrics.py": "the Nadeau-Bengio correction is applied",
}

STALE_REGISTER = "docs/FROZEN_KNOWN_STALE.md"


# --------------------------------------------------------------------------
# hashing, independent of the module it guards
# --------------------------------------------------------------------------


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def members_of(root: Path) -> list[Path]:
    """Every ``.py`` under ``root``, sorted, ignoring bytecode caches.

    Only Python source: the apparatus is source, and restricting the walk means
    the digest cannot move because of a stray file the clutter rule happens not
    to name.
    """
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def roll_up(repo_root: Path, paths: list[Path]) -> str:
    """Fold per-file digests into one, keyed by relative posix path.

    Keyed by path as well as content, so a rename is a change. Forward slashes so
    the laptop and the cluster agree -- the same reason ``hashing.py`` does it,
    arrived at independently here rather than imported.
    """
    digest = hashlib.sha256()
    for path in paths:
        rel = path.relative_to(repo_root).as_posix()
        digest.update(f"{rel}\0{sha256_bytes(path.read_bytes())}\n".encode("utf-8"))
    return digest.hexdigest()


# --------------------------------------------------------------------------
# the guard
# --------------------------------------------------------------------------


@pytest.mark.parametrize("relative", sorted(FROZEN_FILES))
def test_a_frozen_module_has_not_changed(repo_root, relative):
    path = repo_root / relative
    assert path.is_file(), (
        f"{relative} is frozen but does not exist. Deleting a frozen module is a "
        "change to the apparatus." + CONSEQUENCE
    )

    actual = sha256_bytes(path.read_bytes())
    assert actual == FROZEN_FILES[relative], (
        f"{relative} has changed.\n"
        f"  expected {FROZEN_FILES[relative]}\n"
        f"  actual   {actual}" + CONSEQUENCE
    )


@pytest.mark.parametrize("relative", sorted(FROZEN_TREES))
def test_a_frozen_package_has_not_gained_or_lost_a_module(repo_root, relative):
    expected = FROZEN_TREES[relative]
    root = repo_root / relative
    assert root.is_dir(), f"{relative} is frozen but is not a directory{CONSEQUENCE}"

    found = members_of(root)
    names = [p.relative_to(repo_root).as_posix() for p in found]
    assert len(found) == expected["members"], (
        f"{relative} holds {len(found)} python modules, not {expected['members']}. "
        f"Found: {names}\nAdding a module to a frozen package is a change to the "
        "apparatus even though no existing file moved." + CONSEQUENCE
    )

    actual = roll_up(repo_root, found)
    assert actual == expected["rollup"], (
        f"{relative} has changed.\n"
        f"  expected rollup {expected['rollup']}\n"
        f"  actual   rollup {actual}\n"
        f"  members: {names}" + CONSEQUENCE
    )


# --------------------------------------------------------------------------
# teeth
# --------------------------------------------------------------------------


def test_the_guard_notices_a_one_byte_change(repo_root, tmp_path):
    """A guard nobody has seen fire is not evidence (Part 7 §3).

    One added comment character to a copy of the harness, and the digest must
    move. This is the whole mechanism, demonstrated rather than assumed.
    """
    original = (repo_root / "src/cleft/train/harness.py").read_bytes()
    assert sha256_bytes(original) == FROZEN_FILES["src/cleft/train/harness.py"]

    tampered = original + b"\n# one more byte\n"
    assert sha256_bytes(tampered) != FROZEN_FILES["src/cleft/train/harness.py"]


def test_the_rollup_notices_an_added_module(repo_root, tmp_path):
    """The per-file list cannot see an addition; the rollup must."""
    package = tmp_path / "provenance"
    package.mkdir()
    (package / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    before = roll_up(tmp_path, members_of(package))

    (package / "sneaked_in.py").write_text("y = 2\n", encoding="utf-8")
    after = roll_up(tmp_path, members_of(package))

    assert before != after


def test_the_rollup_notices_a_rename(repo_root, tmp_path):
    """Keyed by path as well as content, so moving a module is a change."""
    package = tmp_path / "pkg"
    package.mkdir()
    original = package / "context.py"
    original.write_text("x = 1\n", encoding="utf-8")
    before = roll_up(tmp_path, members_of(package))

    original.rename(package / "run_context.py")
    assert before != roll_up(tmp_path, members_of(package))


# --------------------------------------------------------------------------
# the known-stale register
# --------------------------------------------------------------------------


def test_the_register_exists_and_names_the_escalation_rule(repo_root):
    register = repo_root / STALE_REGISTER
    assert register.is_file(), (
        f"{STALE_REGISTER} is missing. It is where a correction goes when the "
        "text to correct is inside a frozen module."
    )
    text = register.read_text(encoding="utf-8")
    assert "phase3-freeze" in text
    assert "escalation" in text.lower()


@pytest.mark.parametrize("relative", sorted(KNOWN_STALE))
def test_registered_stale_text_is_still_there(repo_root, relative):
    """The register must not outlive what it describes.

    If the text is gone, either the module was corrected -- in which case the
    digest test above has already failed and this is the second half of the same
    message -- or the entry was always wrong. Either way the register needs
    editing, and a register nobody trusts is worse than none.
    """
    needle = KNOWN_STALE[relative]
    source = (repo_root / relative).read_text(encoding="utf-8")
    assert needle in source, (
        f"{STALE_REGISTER} lists {relative} as carrying stale text "
        f"({needle!r}) but it is no longer there. Remove the entry."
    )


def test_every_registered_file_is_actually_frozen(repo_root):
    """Registering a file that is not frozen would be pointless -- and worse,
    misleading, because it could simply have been fixed in place."""
    frozen = set(FROZEN_FILES)
    for tree in FROZEN_TREES:
        frozen.update(
            p.relative_to(repo_root).as_posix()
            for p in members_of(repo_root / tree)
        )
    for relative in KNOWN_STALE:
        assert relative in frozen, (
            f"{relative} is registered as known-stale but is not frozen. "
            "Correct it in place instead."
        )


def test_the_register_documents_every_entry(repo_root):
    text = (repo_root / STALE_REGISTER).read_text(encoding="utf-8")
    for relative in KNOWN_STALE:
        assert relative in text, f"{relative} is not documented in {STALE_REGISTER}"


# --------------------------------------------------------------------------
# the frozen set itself
# --------------------------------------------------------------------------


def test_the_frozen_set_covers_what_the_brief_names(repo_root):
    """Harness, gates, metrics, fold handling, provenance -- plus the two the
    results depend on just as directly. A frozen set that quietly lost an entry
    would pass every test above."""
    covered = set(FROZEN_FILES) | {
        member
        for tree in FROZEN_TREES
        for member in (
            p.relative_to(repo_root).as_posix()
            for p in members_of(repo_root / tree)
        )
    }
    for required in (
        "src/cleft/train/harness.py",
        "src/cleft/train/gates.py",
        "src/cleft/train/determinism.py",
        "src/cleft/eval/metrics.py",
        "src/cleft/data/folds.py",
        "src/cleft/provenance/context.py",
        "src/cleft/provenance/hashing.py",
        "src/cleft/provenance/gitinfo.py",
        # The data-producing path, added before Phase 5 staged SCUT through it.
        "src/cleft/geometry/staging.py",
        "src/cleft/geometry/trapezium.py",
        "src/cleft/geometry/mapping.py",
    ):
        assert required in covered, f"{required} is not covered by the freeze guard"


def test_the_unfrozen_data_producing_modules_are_named(repo_root):
    """"The frozen set is complete" is the assumption that created the gap the
    staging modules just closed. The remainder is listed rather than assumed
    away, and each named file must exist so the list cannot go stale silently."""
    for relative in NOT_YET_FROZEN:
        assert (repo_root / relative).is_file()
        assert relative not in FROZEN_FILES, (
            f"{relative} is frozen; remove it from NOT_YET_FROZEN"
        )


def test_the_cleft_geometry_figures_are_in_the_codebase(repo_root):
    """Phase 5's parity check must source the cleft figures from code, not from
    a brief. A number that lives only in a document cannot be re-derived, and
    parity measured against an untraceable figure is not a measurement."""
    from cleft.geometry import staging

    recorded = staging.CLEFT_STAGED_GEOMETRY
    assert recorded["n_patients"] == 237
    assert recorded["aspect_ratio"]["median"] == 0.7404
    assert recorded["pad_fraction"] == {"min": 0.0179, "mean": 0.2534, "max": 0.4464}
    assert recorded["trapezium"]["area_of_content"] == 0.802
    assert staging.PAD_FRACTION_MEAN == recorded["pad_fraction"]["mean"]
    assert "p2-stage-1" in recorded["source"], "the provenance travels with them"


def test_every_digest_is_a_full_sha256():
    """An abbreviated digest is not a usable reference -- the same rule the image
    digests follow, for the same reason."""
    digests = list(FROZEN_FILES.values()) + [
        tree["rollup"] for tree in FROZEN_TREES.values()
    ]
    for digest in digests:
        assert len(digest) == 64, f"{digest} is not a full SHA-256"
        assert set(digest) <= set("0123456789abcdef"), f"{digest} is not hex"


def test_what_is_frozen_is_not_what_phase_4_extends(repo_root):
    """The arms Phase 4 adds live outside the frozen set, by design.

    ``phase3.py`` wires a run together and ``torch_backbone.py`` builds models --
    both are extended by new arms and neither is frozen. If a Phase 4 change needs
    to reach into the frozen list instead, that is the signal to escalate rather
    than to edit.
    """
    extendable = {
        "src/cleft/train/phase3.py",
        "src/cleft/train/stub.py",
        "src/cleft/train/torch_backbone.py",
    }
    assert not (extendable & set(FROZEN_FILES))
    for relative in extendable:
        assert (repo_root / relative).is_file()
