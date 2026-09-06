"""Cross-validation folds.

Brief §2.5: StratifiedGroupKFold, 5 folds, groups = patient, stratified on the
3-class label, generated **once** and written as a versioned artifact.

Three invariants, asserted here and re-asserted by ``folds.verify`` at load time,
because R7 says a verifier passing only proves the thing it tests — and the fold
file is consumed by every arm in the ladder, so a defect in it would contaminate
every result at once rather than one of them.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cleft.data import folds as F

N = 237


def cohort(n: int = N, seed: int = 0, minority: int = 30):
    """Patient ids and 3-class labels with a deliberately skewed distribution.

    The real cohort's {4,5} class is small — PLAN §4.6 records g1=8 and g5=4 in
    5-class terms — so a balanced fixture would not exercise the case that
    actually bites.
    """
    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=int)
    labels[:minority] = 2
    labels[minority : minority + 60] = 1
    rng.shuffle(labels)
    return list(range(1, n + 1)), labels


@pytest.fixture
def built():
    ids, labels = cohort()
    return F.generate(ids, labels), ids, labels


# --------------------------------------------------------------------------
# the three invariants
# --------------------------------------------------------------------------


def test_every_patient_appears_in_exactly_one_test_fold(built):
    result, ids, _ = built
    seen = [pid for fold in range(result.n_folds) for pid in result.test_ids(fold)]
    assert sorted(seen) == sorted(ids)
    assert len(seen) == len(set(seen)), "a patient appears in two test folds"


def test_test_folds_are_disjoint_and_cover_the_cohort(built):
    result, ids, _ = built
    sets = [set(result.test_ids(f)) for f in range(result.n_folds)]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert not sets[i] & sets[j]
    assert set().union(*sets) == set(ids)


def test_train_and_test_never_share_a_patient(built):
    result, _, _ = built
    for fold in range(result.n_folds):
        assert not set(result.train_ids(fold)) & set(result.test_ids(fold))


def test_every_fold_contains_all_three_classes(built):
    result, _, _ = built
    for fold in range(result.n_folds):
        present = set(result.class_counts[fold])
        assert present == {0, 1, 2}, f"fold {fold} is missing a class: {present}"
        assert all(count > 0 for count in result.class_counts[fold].values())


def test_a_patient_with_several_rows_never_spans_folds():
    """Today one row per patient, so grouping is a no-op. Phase 2 adds the basal
    view and it stops being one. The invariant must already hold.
    """
    ids, labels = cohort(n=120, seed=1, minority=20)
    doubled_ids = [pid for pid in ids for _ in range(2)]
    doubled_labels = np.repeat(labels, 2)

    result = F.generate(doubled_ids, doubled_labels, groups=doubled_ids)
    for pid in ids:
        assigned = {result.assignments[pid]}
        assert len(assigned) == 1


def test_fold_sizes_are_roughly_balanced(built):
    result, ids, _ = built
    sizes = [len(result.test_ids(f)) for f in range(result.n_folds)]
    assert sum(sizes) == len(ids)
    assert max(sizes) - min(sizes) <= 2


# --------------------------------------------------------------------------
# determinism — the artifact is generated once and reused forever
# --------------------------------------------------------------------------


def test_same_inputs_and_seed_give_identical_folds():
    ids, labels = cohort()
    assert F.generate(ids, labels).assignments == F.generate(ids, labels).assignments


def test_the_seed_does_not_affect_the_partition():
    """Recorded for provenance, but it is NOT fold control.

    The splitter runs with shuffle=False, which is what stratifies well, and the
    partition is then a deterministic function of (groups, labels, n_folds).
    Permuting the input by seed was measured and produced byte-identical
    membership for five seeds. Asserting this stops a reader assuming the seed
    varies the split, and stops anyone "fixing" it by turning shuffle back on.
    """
    ids, labels = cohort()
    a = F.generate(ids, labels, seed=1337)
    b = F.generate(ids, labels, seed=2024)
    assert a.assignments == b.assignments
    assert a.seed == 1337 and b.seed == 2024


# --------------------------------------------------------------------------
# stratification actually took effect
# --------------------------------------------------------------------------


def test_per_class_spread_is_within_tolerance(built):
    """The check that was missing. 'All classes present' passed a bad split.

    On the real cohort (88/119/30) shuffle=True gave class-1 counts ranging
    22..26 — every fold contained every class, and the split was still badly
    unstratified.
    """
    result, _, _ = built
    for cls in (0, 1, 2):
        per_fold = [result.class_counts[f][cls] for f in range(result.n_folds)]
        spread = max(per_fold) - min(per_fold)
        assert spread <= F.MAX_CLASS_SPREAD, (
            f"class {cls} counts {per_fold} have spread {spread}"
        )


def test_the_real_distribution_stratifies_essentially_perfectly():
    """88/119/30 over 5 folds should give 18/24/6 with spread of at most one."""
    labels = np.array([0] * 88 + [1] * 119 + [2] * 30)
    np.random.default_rng(0).shuffle(labels)
    result = F.generate(list(range(1, 238)), labels)

    for cls, ideal in ((0, 17.6), (1, 23.8), (2, 6.0)):
        per_fold = [result.class_counts[f][cls] for f in range(5)]
        assert max(per_fold) - min(per_fold) <= 1, f"class {cls}: {per_fold}"
        assert all(abs(n - ideal) <= 1 for n in per_fold), f"class {cls}: {per_fold}"


def test_verify_rejects_an_unstratified_assignment(built):
    """Skew one class without emptying any fold, so the spread guard is isolated.

    Moving a class out of a fold entirely would trip the older "all classes
    present" check instead, and prove nothing about the new one.
    """
    result, ids, labels = built
    label_of = dict(zip(ids, labels))

    donor = [
        pid for pid in result.test_ids(4) if label_of[pid] == 2
    ][:4]
    assert donor, "fixture no longer has class-2 patients in fold 4"
    for pid in donor:
        result.assignments[pid] = 0

    result.class_counts = {
        f: {
            c: sum(
                1 for pid in ids
                if result.assignments[pid] == f and label_of[pid] == c
            )
            for c in (0, 1, 2)
        }
        for f in range(result.n_folds)
    }

    # Every fold still holds every class; only the balance is wrong.
    for fold in range(result.n_folds):
        assert all(n > 0 for n in result.class_counts[fold].values())

    with pytest.raises(F.FoldError, match="not stratified"):
        F.verify(result, ids, labels)


def test_input_order_does_not_change_the_assignment():
    """A manifest sorted differently must not produce different folds."""
    ids, labels = cohort()
    order = np.argsort([hash(i) for i in ids])
    shuffled_ids = [ids[i] for i in order]
    shuffled_labels = labels[order]

    a = F.generate(ids, labels)
    b = F.generate(shuffled_ids, shuffled_labels)
    assert a.assignments == b.assignments


def test_the_seed_is_recorded(built):
    result, _, _ = built
    assert result.seed == F.DEFAULT_SEED
    assert result.n_folds == F.N_FOLDS == 5


# --------------------------------------------------------------------------
# rejection
# --------------------------------------------------------------------------


def test_duplicate_patient_ids_without_groups_are_rejected():
    with pytest.raises(F.FoldError, match="duplicate"):
        F.generate([1, 2, 2, 3] * 10, np.zeros(40, dtype=int))


def test_length_mismatch_is_rejected():
    with pytest.raises(F.FoldError, match="length"):
        F.generate([1, 2, 3], np.zeros(4, dtype=int))


def test_a_class_too_small_to_appear_in_every_fold_is_rejected():
    """Three members cannot populate five folds; say so rather than emit them."""
    ids = list(range(1, 101))
    labels = np.zeros(100, dtype=int)
    labels[:3] = 2
    labels[3:40] = 1
    with pytest.raises(F.FoldError, match="class 2|too few|3"):
        F.generate(ids, labels)


def test_inconsistent_labels_within_a_group_are_rejected():
    """One patient cannot be in two classes at once."""
    ids = [1, 1] + list(range(2, 60))
    labels = np.zeros(60, dtype=int)
    labels[0], labels[1] = 0, 2
    labels[2:20] = 1
    labels[20:30] = 2
    with pytest.raises(F.FoldError, match="inconsistent|group"):
        F.generate(ids, labels, groups=ids)


def test_labels_outside_the_three_classes_are_rejected():
    ids = list(range(1, 61))
    labels = np.zeros(60, dtype=int)
    labels[:10] = 7
    with pytest.raises(F.FoldError, match="class"):
        F.generate(ids, labels)


# --------------------------------------------------------------------------
# the artifact
# --------------------------------------------------------------------------


def test_round_trip_preserves_the_assignment(built, tmp_path):
    result, _, _ = built
    path = tmp_path / "folds_v1.json"
    F.save(result, path)
    loaded = F.load(path)

    assert loaded.assignments == result.assignments
    assert loaded.seed == result.seed
    assert loaded.n_folds == result.n_folds


def test_the_artifact_records_how_it_was_made(built, tmp_path):
    result, _, _ = built
    path = tmp_path / "folds_v1.json"
    F.save(result, path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    for key in ("n_folds", "seed", "n_patients", "class_counts", "assignments"):
        assert key in payload, f"artifact does not record {key}"


def test_the_summary_is_shareable_and_the_assignment_is_not():
    """Fold membership is patient-keyed; the class counts are aggregates.

    Patient ids are offset well clear of the fold and class indices, because
    ids 1-237 overlap them numerically and a substring search would then match
    "class 1" and call it a leak.
    """
    _, labels = cohort()
    ids = [10_000 + i for i in range(N)]
    result = F.generate(ids, labels)

    summary = result.summary()
    assert "assignments" not in summary

    text = json.dumps(summary)
    for pid in ids[:25]:
        assert str(pid) not in text, f"patient {pid} leaked into the SHAREABLE summary"

    # And the full artifact does carry them, which is why it is CLUSTER-ONLY.
    assert str(ids[0]) in json.dumps(result.as_dict())


def test_verify_accepts_a_good_artifact(built):
    result, ids, labels = built
    F.verify(result, ids, labels)


def test_verify_rejects_a_tampered_assignment(built):
    """R7: the generator being right is not evidence the file on disk is."""
    result, ids, labels = built
    victim = ids[0]
    result.assignments[victim] = (result.assignments[victim] + 1) % result.n_folds
    with pytest.raises(F.FoldError):
        F.verify(result, ids, labels)


def test_verify_rejects_a_missing_patient(built):
    result, ids, labels = built
    del result.assignments[ids[0]]
    with pytest.raises(F.FoldError, match="missing|exactly one"):
        F.verify(result, ids, labels)


def test_verify_rejects_a_fold_index_out_of_range(built):
    result, ids, labels = built
    result.assignments[ids[0]] = 99
    with pytest.raises(F.FoldError, match="range|99"):
        F.verify(result, ids, labels)
