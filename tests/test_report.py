"""The SHAREABLE summary block.

Exit criteria 6-10 are verified by someone pasting this block out of a cluster
log, so the one property that must hold absolutely is that it contains no patient
identifier — apart from folders 143 and 238, which criterion 7 requires it to
name and which are already published in the plan.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cleft.data import folds as F
from cleft.data import labels as L
from cleft.data import reliability as R
from cleft.data import report as REP

from test_reliability import panel

#: Patient ids far from any count, index or class label, so finding one of these
#: in the rendered block is unambiguous evidence of a leak.
LEAK_IDS = [900_001 + i for i in range(120)]


@pytest.fixture
def summary():
    matrix = panel(n=120, seed=7)
    reliability = R.summarise(matrix)
    learn = L.learnability_table(matrix, orthodontist_index=1)

    labels = np.zeros(120, dtype=int)
    labels[:20] = 2
    labels[20:55] = 1
    rng = np.random.default_rng(3)
    rng.shuffle(labels)
    fold_result = F.generate(LEAK_IDS, labels)

    return REP.build_summary(
        counts=REP.CohortCounts(
            patients=237, frontal=237, basal=236, photoless=14, images=473,
            scored_rows=251,
        ),
        rule_disagreements={143, 238},
        reliability=reliability,
        learnability=learn,
        expected_ordering=L.EXPECTED_ORDERING,
        fold_sizes=[len(fold_result.test_ids(f)) for f in range(fold_result.n_folds)],
        fold_class_counts=fold_result.class_counts,
        rater_names=tuple(f"Rater {i}" for i in range(5)),
    )


# --------------------------------------------------------------------------
# the property that matters
# --------------------------------------------------------------------------


def test_no_patient_identifier_reaches_the_block(summary):
    rendered = summary.render()
    serialised = json.dumps(summary.as_dict())
    for pid in LEAK_IDS:
        assert str(pid) not in rendered, f"patient {pid} leaked into the rendered block"
        assert str(pid) not in serialised, f"patient {pid} leaked into the payload"


def test_the_only_patient_numbers_present_are_the_published_exceptions(summary):
    """143 and 238 are required by exit criterion 7 and published in PLAN §4.1."""
    rendered = summary.render()
    assert "143" in rendered and "238" in rendered
    assert summary.rule_disagreements == REP.PUBLISHED_EXCEPTIONS


def test_the_builder_never_receives_patient_level_data():
    """Aggregate-only BY CONSTRUCTION: there is nothing patient-level in scope.

    If a future edit wants to add a patient id to this block, it will have to
    change the signature first, which is a visible act rather than an oversight.
    """
    import inspect

    parameters = set(inspect.signature(REP.build_summary).parameters)
    forbidden = {"manifest", "rows", "assignments", "grades", "matrix", "patient_ids"}
    assert not (parameters & forbidden), (
        f"build_summary accepts patient-level input: {parameters & forbidden}"
    )


# --------------------------------------------------------------------------
# content
# --------------------------------------------------------------------------


def test_every_exit_criterion_has_a_line(summary):
    rendered = summary.render()
    for marker in (
        "exit criterion 6",
        "exit criterion 7",
        "exit criterion 8",
        "exit criterion 9",
        "exit criterion 10",
    ):
        assert marker in rendered, f"the block does not cover {marker}"


def test_the_block_is_labelled_shareable(summary):
    assert "SHAREABLE" in summary.render()


def test_the_counts_line_shows_the_arithmetic(summary):
    """237 + 14 = 251, visible rather than asserted elsewhere."""
    rendered = summary.render()
    assert "251" in rendered and "(237 + 14)" in rendered


def test_the_ceiling_is_labelled_as_the_square_root(summary):
    """So a reader cannot mistake it for the reliability, which is the old error."""
    rendered = summary.render()
    assert "sqrt" in rendered.lower()
    assert summary.pcc_ceiling_237 > summary.reliability_237


def test_the_population_note_is_present(summary):
    """A reader comparing against §1.7's 251-row numbers must be warned."""
    rendered = summary.render()
    assert "251" in rendered
    assert "Different population" in rendered or "different population" in rendered


def test_rule_mismatch_is_flagged_not_hidden():
    summary = REP.build_summary(
        counts=REP.CohortCounts(237, 237, 236, 14, 473, 251),
        rule_disagreements={7, 143, 238},
        reliability=R.summarise(panel(n=60, seed=8)),
        learnability=L.learnability_table(panel(n=60, seed=8), orthodontist_index=1),
        expected_ordering=L.EXPECTED_ORDERING,
        fold_sizes=[12] * 5,
        fold_class_counts={f: {0: 8, 1: 3, 2: 1} for f in range(5)},
    )
    assert summary.rule_matches_expected is False
    assert "MISMATCH" in summary.render()


def test_ordering_mismatch_is_flagged(summary):
    """The synthetic panel will not reproduce the real ordering, and says so."""
    assert isinstance(summary.ordering_matches_expected, bool)
    marker = "OK" if summary.ordering_matches_expected else "MISMATCH"
    assert marker in summary.render()


def test_fold_class_counts_appear_per_fold(summary):
    rendered = summary.render()
    assert rendered.count("fold ") >= 5


def test_the_payload_round_trips_as_json(summary):
    restored = json.loads(json.dumps(summary.as_dict()))
    assert restored["counts"]["patients"] == 237
    assert restored["rule_disagreements"] == [143, 238]
