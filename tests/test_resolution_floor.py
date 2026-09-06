"""The cohort's resolution, in three tiers that may never be merged.

Banked because a figure nowhere in the repo -- "the detection floor is a
delta-PCC of about 0.12 to 0.14" -- was asserted in a measured voice.
These tests hold the tiers apart, verify the measured ones at their
sources, and pin the prohibited phrasing as a literal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cleft import ladder, phase18, results_ledger, roadb

REPO = Path(__file__).resolve().parents[1]


def _flat(text: str) -> str:
    return " ".join(text.split())


# --------------------------------------------------------------------------
# tier 1 -- every measured figure, checked at its own source
# --------------------------------------------------------------------------


def test_the_unresolvable_band_is_quoted_from_its_record():
    measured = ladder.SMALLEST_RESOLVABLE_DIFFERENCE["measured"]
    quoted = _flat(measured["unresolvable_band"])
    source = _flat(ladder.COHORT_CANNOT_RESOLVE["finding"])

    # The quote is the record's own sentence, not a paraphrase of it.
    assert "this cohort cannot resolve PCC differences of 0.04 to 0.10" in (
        source
    )
    assert "this cohort cannot resolve PCC differences of 0.04 to 0.10" in (
        quoted
    )
    # And the at-scale figures are the record's, re-read here.
    at_scale = ladder.COHORT_CANNOT_RESOLVE["at_scale"]
    assert (at_scale["tested"], at_scale["survived"], at_scale["withdrawn"]) == (
        30, 1, 29
    )
    for fragment in ("30 tested", "1 survived", "29 withdrawn"):
        assert fragment in quoted
    # The survivor's caveat travels with the count.
    assert "0.0065 baseline" in quoted
    assert "0.0065 baseline" in _flat(at_scale["the_one"])


def test_the_smallest_resolvable_delta_is_the_one_the_record_holds():
    cell = ladder.SMALLEST_RESOLVABLE_DIFFERENCE["measured"][
        "smallest_resolvable"
    ]
    verdict = ladder.PHASE_7D_CLOSING["verdicts"][
        "p7d__b32_512_vs_patch16_512"
    ]
    assert cell["delta"] == verdict["delta"] == 0.1386
    assert verdict["n_excluding_zero"] == "5/5"
    assert verdict["margin"] == 5.32
    assert verdict["verdict"] == "CLAIMABLE"
    assert cell["contrast"] == "p7d__b32_512_vs_patch16_512"
    assert "p7d-grid-density" in cell["home"]

    # It really is the SMALLEST delta to satisfy both conditions --
    # derived from the ledger, not asserted.
    passed = [
        entry for entry in results_ledger.ENTRIES
        if entry["status"] == "CLAIMABLE"
    ]
    assert passed, "there are claimable rows to check against"
    # p7d-grid-density is among them and carries this delta.
    grid = next(e for e in passed if e["id"] == "p7d-grid-density")
    assert "+0.1386" in grid["claim"]


def test_the_margin_structure_is_carried_by_reference_not_copied():
    measured = _flat(
        ladder.SMALLEST_RESOLVABLE_DIFFERENCE["measured"]["margin_structure"]
    )
    structure = roadb.CONDITION_1_MARGIN_STRUCTURE
    # The ledger carries it BY REFERENCE -- the same object, not a copy.
    assert results_ledger.MARGIN_TABLE is structure

    assert "nothing at or below 2.55x has ever passed" in _flat(
        structure["floor"]
    )
    assert "nothing at or below 2.55x has ever passed" in measured
    assert "everything at or above 5.32x has passed -- five for five" in _flat(
        structure["ceiling"]
    )
    assert "everything at or above 5.32x has passed, five for five" in measured
    middle = _flat(structure["the_middle_is_the_finding"])
    assert "3.83 passed while 4.34 and 4.69 failed" in middle
    assert "3.83 passed while 4.34 and 4.69 failed" in measured


# --------------------------------------------------------------------------
# tier 2 -- the inference, and the fence around it
# --------------------------------------------------------------------------


def test_the_reasoned_tier_is_labelled_and_explicitly_non_gating():
    reasoned = ladder.SMALLEST_RESOLVABLE_DIFFERENCE["reasoned"]
    assert reasoned["tag"].startswith("[REASONED]")
    assert "NON-GATING" in reasoned["tag"]
    assert "NOT A THRESHOLD" in reasoned["tag"]

    inference = _flat(reasoned["the_inference"])
    assert "between 0.10 and 0.1386" in inference
    # The bracket's endpoints ARE the two measured figures, so the
    # inference cannot drift from the tier it rests on.
    assert "0.04 to 0.10" in _flat(ladder.COHORT_CANNOT_RESOLVE["finding"])
    assert ladder.SMALLEST_RESOLVABLE_DIFFERENCE["measured"][
        "smallest_resolvable"
    ]["delta"] == 0.1386

    untested = _flat(reasoned["the_band_is_untested"])
    assert "NOTHING has ever been tested between 0.10 and 0.1386" in untested
    assert "UNLOCATED" in untested

    may_not = _flat(reasoned["may_not"])
    for forbidden in ("gate a phase", "be quoted as a threshold",
                      "too small to pursue", "justify not running"):
        assert forbidden in may_not
    # And the self-referential reason the fence exists at all.
    why = _flat(reasoned["why_these_prohibitions_are_written_down"])
    assert "MEASURED VOICE" in why
    assert "precisely the error that produced this record" in why


def test_the_three_tiers_are_separate_objects():
    """Collapsing them is the error; the structure makes that visible."""
    record = ladder.SMALLEST_RESOLVABLE_DIFFERENCE
    assert set(record) >= {"measured", "reasoned", "quotable"}
    assert isinstance(record["measured"], dict)
    assert isinstance(record["reasoned"], dict)
    # No measured figure leaks into the reasoned tier's numbers, and no
    # reasoned language leaks into the measured one.
    measured_text = _flat(str(record["measured"]))
    assert "0.12" not in measured_text and "0.14" not in measured_text
    assert "[REASONED]" not in measured_text
    assert "collapsing them" in _flat(record["why_this_record_exists"]).lower()


# --------------------------------------------------------------------------
# tier 3 -- the quotable line and the prohibition literal
# --------------------------------------------------------------------------


def test_the_quotable_figure_is_bound_to_one_phrasing():
    quotable = ladder.SMALLEST_RESOLVABLE_DIFFERENCE["quotable"]
    assert quotable["the_one_figure"] == 0.1386
    assert quotable["the_one_phrasing"] == (
        "the smallest resolvable difference this cohort has demonstrated"
    )
    never = _flat(quotable["never"])
    assert "'the detection floor'" in never
    assert "a threshold" in never
    assert "0.12-0.14" in never
    assert quotable["prohibition"] == "DETECTION_FLOOR_PROHIBITION"

    # The numeric coincidence the sweep turned up, and the arithmetic
    # that shows the two really are different quantities.
    from cleft import classification

    coincidence = _flat(quotable["the_0_1386_coincidence"])
    assert "SAME DIGITS, DIFFERENT QUANTITY" in coincidence
    banked = classification.CLASSIFICATION_METRICS_BANKED[
        "the_intervals_sharpen_it"
    ]
    assert "above it by 0.1386 at the lower bound" in _flat(banked)
    assert round(0.3614 - 0.2228, 4) == 0.1386   # the other 0.1386


def test_the_detection_floor_prohibition_is_a_literal_string():
    """Sibling to the CleftGNN-IEM prohibition, and tested the same way
    so no later turn reaches for the phrase."""
    prohibition = ladder.DETECTION_FLOOR_PROHIBITION
    assert prohibition.startswith(
        "THIS COHORT HAS NO MEASURED DETECTION FLOOR AND THE PHRASE IS "
        "NEVER USED"
    )
    flat = _flat(prohibition)
    assert "no threshold at 0.12-0.14 or anywhere else" in flat
    assert "never measured, never registered, and appears in no record" in flat
    assert "THE SMALLEST RESOLVABLE DIFFERENCE THIS COHORT HAS DEMONSTRATED" in (
        flat
    )
    # The distinction that keeps 0.1386 from becoming a limit.
    assert "a demonstrated instance, not a limit" in flat
    assert "what was resolved once, not what can be" in flat
    assert "UNTESTED" in flat
    assert "No phase is gated on any of this" in flat

    # It is the same SHAPE as the prohibition it is a sibling of.
    iem = phase18.DELIVERABLES_REGISTERED["cleftgnn_iem_prohibition"]
    assert iem.startswith("OUR IEM VALUES ARE NEVER PLACED BESIDE CLEFTGNN'S")
    assert prohibition.isupper() is False   # prose, not a shout
    assert len(prohibition) > 200


# --------------------------------------------------------------------------
# the provenance of the error, and the sweep that bounded it
# --------------------------------------------------------------------------


def test_the_error_provenance_states_both_halves():
    record = ladder.THE_ERROR_PROVENANCE
    assert record["the_figure"] == "detection floor ~= 0.12-0.14 delta-PCC"
    # Half one: where it came from and that it was stated as measured.
    assert "CONVERSATION" in record["origin"]
    assert "memory summaries" in record["origin"]
    assert "as MEASURED" in record["how_it_was_stated"]
    assert "interpretation" in record["what_it_governed"]

    # Half two: no verdict was affected, and WHY that is true.
    did_not = _flat(record["what_it_did_not_touch"])
    assert "NO VERDICT WAS AFFECTED" in did_not
    assert "combined_claimable_delta" in did_not
    assert "existed nowhere to be read" in did_not
    assert "stands as computed" in did_not
    assert "INTERPRETIVE, not evidential" in _flat(record["the_damage"])

    # The sweep, and its outcome.
    sweep = _flat(record["the_sweep"])
    assert "ZERO genuine instances" in sweep
    assert "memory files" in sweep
    preventive = _flat(record["the_correction_is_preventive"])
    assert "nothing had to be rewritten" in preventive
    assert "the sweep was run rather than assumed" in preventive


def test_the_wrong_figure_appears_in_no_record():
    """The sweep, re-run as a test so it cannot rot.

    The phrase may appear ONLY in the record-state document, where it is
    the correction, and in this project's own prohibition/provenance
    records, where it is the thing being forbidden.
    """
    allowed = {
        "docs/RECORD_STATE_2026-08-31.md",   # states it is NOT in the repo
        "src/cleft/ladder.py",               # the prohibition and provenance
        "tests/test_resolution_floor.py",    # this file
    }
    offenders = []
    for path in list((REPO / "src").rglob("*.py")) + \
            list((REPO / "tests").rglob("*.py")) + \
            list((REPO / "configs").rglob("*.yaml")) + \
            list((REPO / "docs").rglob("*.md")):
        rel = path.relative_to(REPO).as_posix()
        if rel in allowed:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if "detection floor" in text or "detection limit" in text:
            offenders.append(rel)
    assert offenders == [], (
        f"the prohibited phrase appears in {offenders} -- "
        "ladder.DETECTION_FLOOR_PROHIBITION"
    )


def test_cohort_cannot_resolve_points_at_what_it_does_not_license():
    assert ladder.COHORT_CANNOT_RESOLVE[
        "what_it_does_not_license_2026_08_31"
    ] == "SMALLEST_RESOLVABLE_DIFFERENCE"
    # And the pointer resolves.
    assert hasattr(ladder, "SMALLEST_RESOLVABLE_DIFFERENCE")
    assert hasattr(ladder, "DETECTION_FLOOR_PROHIBITION")
    assert hasattr(ladder, "THE_ERROR_PROVENANCE")
