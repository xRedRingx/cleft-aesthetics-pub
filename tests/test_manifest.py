"""The manifest, and the failure this phase exists to prevent.

Phase 1 brief §1.2, THE CRITICAL DESIGN POINT: assign frontal by **score-sheet
membership**. The odd/even rule is a **cross-check only**, and it must disagree in
exactly folders 143 and 238.

Building from the rule would silently swap patient 143's two views. Nothing would
error, no count would change, and every downstream number would be computed on a
basal image labelled frontal. The first test below makes that visible: it builds
folder 143 both ways and asserts the rule gets it wrong.
"""

from __future__ import annotations

import pytest

from cleft.data import manifest as M

from fixtures import cohort as C


# Module-scoped: the clean cohort is read-only in every test that uses it, and
# rebuilding 237 folders per test is wasted time against the 60-second budget.
# The broken variants below stay per-test, since each needs its own defect.
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return C.build_cohort(tmp_path_factory.mktemp("cohort"))


def broken(tmp_path, kind):
    return C.build_cohort(tmp_path / f"cohort_{kind}", broken=kind)


# --------------------------------------------------------------------------
# folder 143 -- the whole reason this phase is built the way it is
# --------------------------------------------------------------------------


def test_folder_143_the_rule_is_wrong_and_the_lookup_is_right(built):
    """The rule predicts 523. The frontal is 524. Lookup gets it right.

    Both halves matter. If the first assertion ever fails, the rule has been
    quietly changed to special-case 143 and the cross-check has stopped being an
    independent check. If the second fails, the manifest is swapping the views of
    a real patient.
    """
    images = built.folders[C.FOLDER_143]
    assert sorted(images) == [523, 524]

    by_rule = M.frontal_by_rule(C.FOLDER_143, images)
    assert by_rule == 523, "the rule must genuinely predict the wrong image here"

    by_lookup = M.frontal_by_lookup(images, set(built.scored_ids))
    assert by_lookup == 524, "the scored id IS the frontal"

    assert by_rule != by_lookup, (
        "folder 143 is the exception the cross-check exists to catch; if these "
        "agree the fixture no longer reproduces it"
    )


def test_a_rule_built_manifest_would_swap_patient_143s_views(built):
    """Spell out the consequence, so it is in the suite and not only in a doc."""
    images = built.folders[C.FOLDER_143]

    rule_frontal = M.frontal_by_rule(C.FOLDER_143, images)
    rule_basal = next(i for i in images if i != rule_frontal)
    true_frontal = M.frontal_by_lookup(images, set(built.scored_ids))
    true_basal = next(i for i in images if i != true_frontal)

    assert rule_frontal == true_basal
    assert rule_basal == true_frontal, (
        "a rule-built manifest does not merely mislabel patient 143, it exchanges "
        "the two views: the basal is scored as frontal and vice versa"
    )


def test_folder_238_is_the_other_exception(built):
    """One image, odd, past the split. Frontal, and there is no basal."""
    images = built.folders[C.FOLDER_238]
    assert images == [581]

    # Past the split the rule wants an even id, and there is none.
    assert M.frontal_by_rule(C.FOLDER_238, images) != 581
    assert M.frontal_by_lookup(images, set(built.scored_ids)) == 581


# --------------------------------------------------------------------------
# the cross-check
# --------------------------------------------------------------------------


def test_cross_check_flags_exactly_the_two_known_exceptions(built):
    result = M.build(built.folders, set(built.scored_ids))
    assert result.rule_disagreements == {C.FOLDER_143, C.FOLDER_238}


def test_a_third_disagreement_is_a_hard_failure(tmp_path):
    """§1.2: 'A third disagreement is a hard failure.'"""
    bad = broken(tmp_path, "third_rule_violation")
    with pytest.raises(M.ManifestError) as excinfo:
        M.build(bad.folders, set(bad.scored_ids))

    message = str(excinfo.value)
    assert "7" in message, "the message must name the offending folder"
    assert "143" in message and "238" in message, "and the expected exceptions"


def test_the_known_exceptions_are_not_hardcoded_into_the_rule():
    """The rule must stay a naive parity rule to be an independent check."""
    assert M.frontal_by_rule(143, [523, 524]) == 523
    assert M.frontal_by_rule(144, [525, 526]) == 525
    assert M.frontal_by_rule(200, [701, 702]) == 702


# --------------------------------------------------------------------------
# counts (§1.1)
# --------------------------------------------------------------------------


def test_structure_counts(built):
    result = M.build(built.folders, set(built.scored_ids))
    assert len(result.rows) == 237
    assert sum(1 for r in result.rows if r.frontal_id is not None) == 237
    assert sum(1 for r in result.rows if r.basal_id is not None) == 236
    assert result.n_images == 473


def test_folder_52_is_present_and_empty_not_absent(built):
    """CORRECTED 2026-07-27. The plan said absent; the directory exists.

    Same consequence -- 237 patients with photographs -- different condition.
    It is recorded as empty rather than inferred from a gap in the numbering.
    """
    assert C.EMPTY_FOLDER in built.folders
    assert built.folders[C.EMPTY_FOLDER] == []

    result = M.build(built.folders, set(built.scored_ids))
    assert C.EMPTY_FOLDER not in {r.patient_id for r in result.rows}
    assert result.empty_folders == [C.EMPTY_FOLDER]
    assert len(result.rows) == 237


def test_the_arithmetic_identity_is_asserted(built):
    """237 + 14 = 251. §1.1 says to assert this explicitly."""
    result = M.build(built.folders, set(built.scored_ids))
    assert len(result.rows) + len(result.photoless_ids) == 251


def test_patient_238_has_no_basal(built):
    result = M.build(built.folders, set(built.scored_ids))
    row = next(r for r in result.rows if r.patient_id == 238)
    assert row.frontal_id == 581
    assert row.basal_id is None


# --------------------------------------------------------------------------
# the 14 photo-less ids
# --------------------------------------------------------------------------


def test_photoless_ids_are_recorded_not_dropped(built):
    """§6: 'Do not silently drop the 14 photo-less IDs.'"""
    result = M.build(built.folders, set(built.scored_ids))
    assert len(result.photoless_ids) == 14
    assert set(result.photoless_ids) == set(C.PHOTOLESS_IDS)
    assert result.photoless_ids == sorted(result.photoless_ids)


def test_photoless_ids_are_even_and_contiguous(built):
    result = M.build(built.folders, set(built.scored_ids))
    ids = result.photoless_ids
    assert all(i % 2 == 0 for i in ids)
    assert ids == list(range(ids[0], ids[-1] + 1, 2))


# --------------------------------------------------------------------------
# the broken variants (§3)
# --------------------------------------------------------------------------


def test_a_folder_with_no_scored_image_is_rejected(tmp_path):
    bad = broken(tmp_path, "missing_frontal")
    with pytest.raises(M.ManifestError, match="no scored image|11"):
        M.build(bad.folders, set(bad.scored_ids))


def test_a_duplicate_scored_id_is_rejected(tmp_path):
    bad = broken(tmp_path, "duplicate_scored_id")
    with pytest.raises(M.ManifestError, match="duplicate"):
        M.build(bad.folders, set(bad.scored_ids), scored_sequence=bad.scored_ids)


def test_a_folder_with_three_images_is_rejected(tmp_path):
    bad = broken(tmp_path, "three_images")
    with pytest.raises(M.ManifestError, match="9|three|expected 2"):
        M.build(bad.folders, set(bad.scored_ids))


def test_every_rejection_names_the_specific_violation(tmp_path):
    """Exit criterion 3: rejected 'with a message naming the specific violation'."""
    expectations = {
        "third_rule_violation": "rule",
        "missing_frontal": "scored",
        "three_images": "image",
    }
    for kind, word in expectations.items():
        bad = broken(tmp_path, kind)
        with pytest.raises(M.ManifestError) as excinfo:
            M.build(bad.folders, set(bad.scored_ids))
        assert word in str(excinfo.value).lower(), (
            f"{kind} rejected without naming what was wrong: {excinfo.value}"
        )
