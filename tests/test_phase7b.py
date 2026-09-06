"""The Phase 7B pre-registration, checked rather than trusted.

A pre-registration only protects a result if it was fixed BEFORE the search and
cannot quietly move afterwards. These tests are what makes that true here: the
budget is a number the space must match, the selection rule is refused a
test-fold quantity rather than asked not to read one, and the exhaustion
criterion is derived from the baseline arm's own SD instead of written down.
"""

from __future__ import annotations

import pytest

from cleft import phase7b


def test_the_space_enumerates_to_exactly_the_declared_budget():
    """**The budget IS the pre-registration** (brief §8: exceeding it stops the
    phase being a search and makes it a fish).

    ``configurations()`` raises on a mismatch rather than returning a list of
    the wrong length, so an axis gaining an entry fails here instead of
    silently spending a trial nobody registered.
    """
    trials = phase7b.configurations()
    assert len(trials) == phase7b.SEARCH_BUDGET == 24
    assert [t["index"] for t in trials] == list(range(24))

    by_axis: dict[str, int] = {}
    for trial in trials:
        by_axis[trial["axis"]] = by_axis.get(trial["axis"], 0) + 1
    # Half the budget on the axis the literature says most about, which is the
    # allocation the brief argues for.
    assert by_axis == {"head": 6, "pooling": 12, "ensemble": 6}
    assert by_axis["pooling"] == phase7b.SEARCH_BUDGET // 2


def test_the_budget_check_has_teeth(monkeypatch):
    """A space that silently disagreed with the budget would defeat the whole
    protocol, so the disagreement must raise. Asserted directly, because on a
    correct space the branch never runs."""
    monkeypatch.setattr(
        phase7b, "HEAD_AXIS", phase7b.HEAD_AXIS + ({"head": "extra"},)
    )
    with pytest.raises(phase7b.Phase7BError, match="enumerates 25"):
        phase7b.configurations()


def test_every_configuration_is_distinct():
    """Two identical trials would spend two of twenty-four on one hypothesis
    and inflate the exhaustion window with a guaranteed non-improvement."""
    trials = phase7b.configurations()
    seen = {
        tuple(sorted((k, str(v)) for k, v in t.items() if k != "index"))
        for t in trials
    }
    assert len(seen) == len(trials)


def test_the_baseline_configuration_is_inside_the_search():
    """So that "nothing beat the untuned arm" is an OBSERVED outcome of the
    search rather than a separate claim, and so a tie resolves to it."""
    trials = phase7b.configurations()
    assert trials[0]["axis"] == "head" and trials[0]["head"] == "linear"
    # And the current representation is in the pooling axis: block 12, CLS.
    assert any(
        t["axis"] == "pooling" and t["block"] == 12 and t["token"] == "cls"
        for t in trials
    )


def test_selection_refuses_a_trial_carrying_a_test_fold_quantity():
    """**The trap the phase exists to avoid** (brief §8: do not select on test
    folds). Gate 6 asserts inner-val is disjoint from test -- that is a
    property of the SPLIT. This is a property of the SELECTION, and a rule that
    merely promised to ignore an OOF score would be one edit from reading it.
    """
    clean = [
        {"index": 0, "inner_val_pcc": 0.24},
        {"index": 1, "inner_val_pcc": 0.26},
    ]
    assert phase7b.select(clean)["index"] == 1

    for leak in ("oof_pcc", "test_pcc", "pcc", "predictions"):
        dirty = [dict(clean[0]), {**clean[1], leak: 0.9}]
        with pytest.raises(phase7b.Phase7BError, match=leak):
            phase7b.select(dirty)


def test_selection_breaks_ties_by_the_earlier_configuration():
    """Pre-registered so the winner of a tie is not decided after seeing the
    numbers. Earlier means earlier in ``configurations()`` order, which puts
    the baseline first."""
    tied = [
        {"index": 3, "inner_val_pcc": 0.25},
        {"index": 0, "inner_val_pcc": 0.25},
        {"index": 7, "inner_val_pcc": 0.25},
    ]
    assert phase7b.select(tied)["index"] == 0
    assert phase7b.select(list(reversed(tied)))["index"] == 0


def test_selection_refuses_a_trial_with_no_score_at_all():
    with pytest.raises(phase7b.Phase7BError, match="no inner_val_pcc"):
        phase7b.select([{"index": 0}])
    with pytest.raises(phase7b.Phase7BError, match="no trials"):
        phase7b.select([])


def test_the_exhaustion_threshold_is_derived_from_the_baselines_own_sd():
    """PLAN §4.12.1 -- inherited, never. A tuned arm's criterion computed from
    the arm it is tuning, so a re-measured SD re-derives it."""
    from cleft.train.phase3 import claimable_delta

    expected = claimable_delta(
        phase7b.BASELINE["seeds"], phase7b.BASELINE["sd"]
    )
    assert phase7b.exhaustion_threshold() == pytest.approx(expected)
    assert phase7b.exhaustion_threshold() == pytest.approx(0.0183, abs=5e-5)
    # It must not be the gate-2 band, which is the figure PLAN §4.3 names
    # specifically as the one not to reuse.
    assert phase7b.BASELINE["sd"] != 0.0137


def test_the_exhaustion_criterion_reads_both_ways():
    """A criterion that could only return one verdict would be decoration.
    Both branches are exercised, and the not-exhausted branch must say so
    rather than inviting the budget to be extended until it flips.
    """
    window = phase7b.EXHAUSTION_WINDOW
    threshold = phase7b.exhaustion_threshold()
    before = [{"index": i, "inner_val_pcc": 0.25} for i in range(window)]

    flat = before + [
        {"index": window + i, "inner_val_pcc": 0.25 + threshold / 2}
        for i in range(window)
    ]
    verdict = phase7b.is_exhausted(flat)
    assert verdict["exhausted"] is True
    assert verdict["best_gain_in_window"] <= verdict["threshold"]

    climbing = before + [
        {"index": window + i, "inner_val_pcc": 0.25 + threshold * 2}
        for i in range(window)
    ]
    verdict = phase7b.is_exhausted(climbing)
    assert verdict["exhausted"] is False
    assert "do not extend the budget" in verdict["reading"]

    with pytest.raises(phase7b.Phase7BError, match="cannot be evaluated yet"):
        phase7b.is_exhausted(before[: window - 1])


def test_the_exhaustion_verdict_carries_the_inputs_that_justify_it():
    """"Exhausted" without its window, threshold and best gain is a claim
    nobody can check -- and this verdict is being used to ask a surgical team
    for a dataset."""
    trials = [
        {"index": i, "inner_val_pcc": 0.25} for i in range(phase7b.SEARCH_BUDGET)
    ]
    verdict = phase7b.is_exhausted(trials)
    for key in (
        "window", "threshold", "best_before_window", "best_gain_in_window",
        "n_trials", "reading",
    ):
        assert key in verdict


def test_the_monitor_divergence_is_declared_not_silent():
    """**A tuned arm monitored differently from the arm it is compared against
    carries an undeclared factor.** Phase 7B monitors inner_val_pcc; the ladder
    early-stops on inner_val_mse. That is a deliberate choice with a reason,
    and the register that records extra factors is the same idea as
    ``ladder.LICENCES`` -- either a recorded design decision or a defect.
    """
    assert phase7b.MONITOR == phase7b.SELECTION_METRIC == "inner_val_pcc"
    assert phase7b.LADDER_MONITOR == "inner_val_mse"
    assert phase7b.summary()["monitor_diverges"] is True

    assert "monitor" in phase7b.BASELINE["also_varies"]
    # Every declared extra factor carries its own licence -- ladder.LICENCES'
    # rule: a second factor is a recorded design decision or it is a defect.
    for factor in phase7b.BASELINE["also_varies"]:
        assert phase7b.BASELINE["licence"][factor]

    # And the ladder really does monitor the other quantity, or the divergence
    # is describing something that is not there.
    from cleft import ladder

    assert ladder.LADDER_GEOMETRY  # the module is the one we mean
    assert phase7b.LADDER_MONITOR != phase7b.MONITOR


def test_the_ladder_configs_really_monitor_what_the_divergence_claims(
    repo_root, monkeypatch
):
    """The claim above is about shipped configs, so it is checked against
    them rather than against a constant that agrees with itself."""
    from cleft.config import load_config

    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")
    monitors = set()
    for path in sorted((repo_root / "configs").glob("p7_*.yaml")):
        task = load_config(path)["task"]
        if "monitor" in task:
            monitors.add(task["monitor"])
    assert monitors == {phase7b.LADDER_MONITOR}, (
        f"the ladder monitors {sorted(monitors)}, but the declared divergence "
        f"is from {phase7b.LADDER_MONITOR!r}"
    )


def test_out_of_budget_axes_are_declared_with_reasons():
    """A search that quietly drops an axis is indistinguishable from one that
    never considered it, and this document is read as an exhaustion argument.
    """
    out = phase7b.OUT_OF_BUDGET
    assert set(out) == {
        "input_resolution_384", "graph_inclusive_ensembling", "augmentation"
    }
    for axis, reason in out.items():
        assert len(reason) > 80, f"{axis} is declared without a real reason"

    # Augmentation is out for a STRUCTURAL reason, not a budget one, and the
    # no-flip rule holds regardless of budget (PLAN §4.6).
    assert "STRUCTURALLY unavailable" in out["augmentation"]
    assert "horizontal flip" in out["augmentation"]


def test_the_ensemble_axis_only_combines_artifacts_of_one_kind():
    """SR-GNN and AG-Net store feature maps where the transformers store pooled
    vectors, and ``check_pairing`` refuses a feature map at a probe arm. An
    ensemble crossing kinds would be a new representation, not an ensemble of
    measured ones."""
    from cleft.embeddings import KIND_FOR_BACKBONE_KIND
    from cleft.models.factory import BACKBONES

    for trial in phase7b.ENSEMBLE_AXIS:
        kinds = {
            KIND_FOR_BACKBONE_KIND[BACKBONES[b]["kind"]] for b in trial["with"]
        }
        assert kinds == {"pooled"}, (
            f"the ensemble mixes artifact kinds {sorted(kinds)}"
        )

    # And averaging is genuinely unavailable: the two feature dims differ, so
    # the space offers concatenation only. Measured from the registry.
    dims = {BACKBONES[b]["embedding_dim"] for b in ("vit_b16", "swin_b")}
    assert len(dims) == 2, (
        "the dims now match, so averaging IS available and the recorded reason "
        "for offering concatenation only has gone stale"
    )
    assert all("concat" in t["combine"] for t in phase7b.ENSEMBLE_AXIS)


class _Head:
    """A deterministic least-squares head, so trial evaluation is testable
    without torch. Fits what a linear head fits, closed form."""

    def __init__(self, ridge: float = 1e-6):
        self.ridge = ridge
        self.weights = None

    def fit(self, features, labels):
        import numpy as np

        design = np.concatenate(
            [features, np.ones((len(features), 1))], axis=1
        )
        gram = design.T @ design + self.ridge * np.eye(design.shape[1])
        self.weights = np.linalg.solve(gram, design.T @ labels)
        return self

    def predict(self, features):
        import numpy as np

        design = np.concatenate(
            [features, np.ones((len(features), 1))], axis=1
        )
        return design @ self.weights


def _cohort(n=60, dim=8, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, dim))
    labels = features[:, 0] * 0.8 + rng.normal(scale=0.4, size=n) + 3.0
    ids = list(range(n))
    assignments = {pid: pid % 5 for pid in ids}
    return features, labels, ids, assignments


def test_a_trial_scores_inner_validation_and_never_the_held_out_fold():
    """**The guarantee the phase rests on, made structural.**

    The evaluator computes the fold's own patients in exactly one place -- the
    assertion that inner-val does not intersect them -- and scores only
    inner-val rows. A trial that scored the held-out fold would produce a
    better-looking number that ranks configurations on the test set, which is
    the fishing this protocol exists to prevent.
    """
    features, labels, ids, assignments = _cohort()
    trial = phase7b.configurations()[0]

    outcome = phase7b.evaluate_configuration(
        trial,
        features_by_set={phase7b.BASE_SET: features},
        labels=labels,
        patient_ids=ids,
        assignments=assignments,
        make_head=lambda t, seed, fold: _Head(),
        inner_val_frac=0.2,
        seed=1337,
    )
    assert set(outcome) == {"inner_val_pcc", "n_folds"}
    assert outcome["n_folds"] == 5
    assert -1.0 <= outcome["inner_val_pcc"] <= 1.0

def test_the_trial_leak_guard_would_catch_a_contaminated_split(monkeypatch):
    """**The guard cannot fire while ``inner_val_split`` behaves**, because
    inner-val is carved from outer-train by construction. That makes it a
    check on the frozen function not changing -- and an assertion that can
    never fail is the pattern in PLAN R7's tally, so it is exercised against a
    deliberately contaminated split rather than left to look green.
    """
    from cleft.train import harness

    features, labels, ids, assignments = _cohort()
    trial = phase7b.configurations()[0]

    def contaminated(train_ids, frac, seed, fold):
        held_out = [pid for pid in ids if assignments[pid] == fold]
        return train_ids[2:], train_ids[:2] + held_out[:1]

    monkeypatch.setattr(harness, "inner_val_split", contaminated)
    with pytest.raises(phase7b.Phase7BError, match="inner-val intersects"):
        phase7b.evaluate_configuration(
            trial,
            features_by_set={phase7b.BASE_SET: features},
            labels=labels,
            patient_ids=ids,
            assignments=assignments,
            make_head=lambda t, seed, fold: _Head(),
            inner_val_frac=0.2,
            seed=1337,
        )


def test_standardisation_is_fitted_on_training_rows_only():
    """**A standardiser fitted over all 237 puts the held-out fold's column
    means and variances into the representation used to predict it** -- a leak
    with no symptom, since the shapes are right and the fit succeeds.

    Same failure ``embeddings.assert_no_test_fold_leak`` exists for on the
    AdaBN path, in a place nothing else would check.
    """
    import numpy as np

    train = np.array([[0.0, 10.0], [2.0, 14.0]])
    everything = np.concatenate([train, np.array([[100.0, -50.0]])])

    transform = phase7b.fit_feature_transform("standardise", train)
    standardised = transform(train)
    assert standardised.mean(axis=0) == pytest.approx([0.0, 0.0], abs=1e-9)

    # Fitted on train, the outlier row is far from zero. Fitted on everything
    # it would be pulled in -- which is the leak.
    outlier = transform(everything)[2]
    assert abs(outlier[0]) > 10, (
        "the transform is not using the training rows' scale"
    )
    assert phase7b.fit_feature_transform("identity", train)(train) is train

    l2 = phase7b.fit_feature_transform("l2norm", train)(train)
    assert np.linalg.norm(l2, axis=1) == pytest.approx([1.0, 1.0])


def test_every_configuration_maps_to_a_transform_and_a_set():
    """No trial may fall through to a default nobody chose."""
    for trial in phase7b.configurations():
        assert phase7b.transform_for(trial) in (
            "identity", "standardise", "l2norm"
        )
        sets = phase7b.required_sets(trial)
        assert sets and all(isinstance(name, str) for name in sets)
        if trial["axis"] == "ensemble":
            assert len(sets) == 2
        else:
            assert len(sets) == 1


def test_the_search_refuses_to_start_until_every_set_is_available():
    """**Guard-3 shaped.** A search that dies at trial 13 has spent cluster
    time and left a record whose exhaustion window cannot be evaluated.
    Twenty-three trials is not a smaller search, it is a different one.

    The pooling axis is the live case: ``extract.extract_features`` has no
    block or token axis, so none of its twelve sets is producible yet.
    """
    with pytest.raises(phase7b.Phase7BError, match="not declared"):
        phase7b.verify_available({phase7b.BASE_SET})

    # The blocker is real and specific: the pooling sets, all twelve.
    pooling = {
        name for name in phase7b.all_required_sets() if "__block" in name
    }
    assert len(pooling) == 12
    assert phase7b.BASE_SET not in pooling

    # With everything declared it passes.
    phase7b.verify_available(set(phase7b.all_required_sets()))


def test_a_trial_record_cannot_carry_an_out_of_fold_number():
    """**Refused at the PRODUCING end, not only the consuming one.** A record
    that never holds an OOF score is a stronger guarantee than one that holds
    it and is not read -- the second is one edit from becoming a leak."""
    trial = phase7b.configurations()[0]
    record = phase7b.make_trial_record(trial, 0.25, 5)
    assert set(record) == set(phase7b.TRIAL_RECORD_KEYS)
    assert "oof_pcc" not in record and "pcc" not in record

    # And it survives the consuming check, which is the other half.
    assert phase7b.select([record])["index"] == 0


def test_the_search_walks_the_budget_checkpoints_and_resumes():
    """One job, 24 trial records, resumable. PLAN §2.7: run identity survives
    a pause and TRAINING DOES NOT -- a resumed pod restarts at zero with no
    symptom but wall time, so the records are the checkpoint."""
    calls, checkpoints = [], []

    def evaluate(trial):
        calls.append(trial["index"])
        return {"inner_val_pcc": 0.20 + trial["index"] / 1000, "n_folds": 5}

    result = phase7b.run_search(
        evaluate, checkpoint=lambda records: checkpoints.append(len(records))
    )
    assert calls == list(range(phase7b.SEARCH_BUDGET))
    assert checkpoints == list(range(1, phase7b.SEARCH_BUDGET + 1))
    assert len(result["trials"]) == phase7b.SEARCH_BUDGET
    assert result["selected"]["index"] == phase7b.SEARCH_BUDGET - 1
    assert "exhausted" in result["exhaustion"]

    # Resumed: the completed trials are not re-evaluated.
    calls.clear()
    resumed = phase7b.run_search(evaluate, completed=result["trials"][:10])
    assert calls == list(range(10, phase7b.SEARCH_BUDGET))
    assert resumed["trials"] == result["trials"]


def test_the_shipped_block_extraction_produces_exactly_the_pooling_axis(
    repo_root, monkeypatch
):
    """**The extraction and the pre-registered axis must be the same twelve.**

    Same discipline as ``test_extract.py``'s check that the Phase 6 config
    equals ``embedding_plan.required_sets()``: a set list that drifted from
    the derivation would leave the search refusing to start, or -- worse --
    starting on eleven representations and one nobody registered.
    """
    from cleft.config import load_config

    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")
    cfg = load_config(repo_root / "configs" / "p7b_extract_blocks.yaml")
    task = cfg["task"]

    from cleft.embedding_plan import set_name

    produced = {
        set_name({**entry, "pretrain_scheme": entry.get("pretrain_scheme")})
        for entry in task["sets"]
    }
    wanted = {
        name for name in phase7b.all_required_sets() if "__block" in name
    }
    assert produced == wanted, (
        f"the config produces {sorted(produced - wanted)} the axis does not "
        f"need, and misses {sorted(wanted - produced)}"
    )
    assert len(produced) == 12

    # Its own artifact version, because a published one cannot gain a set.
    assert task["out_version"] == phase7b.POOLING_EMBEDDINGS_VERSION
    assert task["out_version"] not in (
        "embeddings_v1", "embeddings_g1_control_v1", "embeddings_g1_ladder_v1"
    )
    # The pooling axis is a LayerNorm transformer: no running statistics.
    assert task["per_fold_bn_reestimation"] is False


def test_the_block_extraction_asserts_it_reproduces_the_known_good_set(
    repo_root, monkeypatch
):
    """**The free self-check, and it must actually be wired.**

    block=12 with the class token IS what the final pooled extraction returns,
    so it can be asserted bit-for-bit against a set that already exists and is
    hashed. A batch shipped without that claim would validate nothing, and the
    other eleven depths have no reference at all -- their correctness rests
    entirely on this one agreeing.
    """
    from cleft.config import load_config

    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")
    cfg = load_config(repo_root / "configs" / "p7b_extract_blocks.yaml")
    task = cfg["task"]

    claims = task["reproduces"]
    assert len(claims) == 1
    claim = claims[0]
    assert claim["set"] == phase7b.pooling_set_name(12, "cls")

    # The reference must be a declared input, and the SAME artifact the
    # baseline arm reads -- otherwise it validates against something else.
    by_name = {entry["name"]: entry for entry in cfg["inputs"]}
    assert claim["input"] in by_name
    assert by_name[claim["input"]]["path"].endswith(f"/{phase7b.BASE_SET}")

    from cleft import ladder

    assert (
        by_name[claim["input"]]["rollup_sha256"]
        == ladder.DECLARED_EMBEDDING_HASHES[phase7b.BASE_SET]
    ), "the reference does not carry the registry's hash for that path"

    # The deepest block in the axis is the one with a reference. If the axis
    # ever went deeper, the claim would be checking a non-final block.
    assert max(phase7b.POOLING_BLOCKS) == 12


def test_a_block_set_names_its_representation():
    """Two sets from one checkpoint at different depths ARE different
    representations. Twelve directories differing only in their contents is
    the mutable-data-directory failure PLAN Part 1 opens with."""
    from cleft.embedding_plan import PlanError, set_name

    assert phase7b.pooling_set_name(6, "cls").endswith("__block6_cls")
    assert phase7b.pooling_set_name(6, "mean_patch") != phase7b.pooling_set_name(
        6, "cls"
    )
    assert len({
        phase7b.pooling_set_name(b, t)
        for b in phase7b.POOLING_BLOCKS
        for t in phase7b.POOLING_TOKENS
    }) == 12

    base = {"backbone": "vit_b16", "init": "imagenet", "geometry": "g1",
            "pretrain_scheme": None}
    # A depth without a pooling rule does not name a representation.
    with pytest.raises(PlanError, match="together or not at all"):
        set_name({**base, "block": 6})
    with pytest.raises(PlanError, match="together or not at all"):
        set_name({**base, "token": "cls"})
    with pytest.raises(PlanError, match="unknown token"):
        set_name({**base, "block": 6, "token": "penultimate"})


def test_the_stub_block_sets_genuinely_differ():
    """**Twelve byte-identical sets would make every laptop test of this axis
    pass on one array wearing twelve names**, and the search would rank twelve
    copies of one representation. Failure mode 5 in PLAN R7's tally, which is
    why the stub is built to vary with both fields."""
    import numpy as np

    from cleft.train.extract import extract_features

    images = np.arange(4 * 6 * 6 * 3, dtype=np.float32).reshape(4, 6, 6, 3)
    seen = []
    for block in phase7b.POOLING_BLOCKS:
        for token in phase7b.POOLING_TOKENS:
            values, report = extract_features(
                "stub", "imagenet", images,
                checkpoint_path=None, block=block, token=token,
            )
            assert report["block"] == block and report["token"] == token
            seen.append(values)

    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            assert not np.array_equal(seen[i], seen[j]), (
                f"stub sets {i} and {j} are identical; the axis would be "
                "twelve copies of one representation"
            )


def test_the_pooling_axis_refuses_a_pretrained_init_and_per_fold_adaptation():
    """Intermediate extraction builds the backbone from its pretrained weights
    and loads no checkpoint, so a pretrained init would silently read the
    WRONG MODEL -- the sharpest failure in the extraction path, and the one
    ``check_pairing`` exists for downstream."""
    import numpy as np

    from cleft.train.extract import ExtractError, extract_features

    images = np.zeros((2, 4, 4, 3), dtype=np.float32)
    with pytest.raises(ExtractError, match="pooling axis is defined at the imagenet"):
        extract_features(
            "stub", "scut_original", images,
            checkpoint_path=None, block=6, token="cls",
        )
    with pytest.raises(ExtractError, match="not combined"):
        extract_features(
            "stub", "imagenet", images, checkpoint_path=None,
            block=6, token="cls", adapt_rows=np.array([0]),
        )
    with pytest.raises(ExtractError, match="together or not at all"):
        extract_features(
            "stub", "imagenet", images, checkpoint_path=None, block=6,
        )


def test_the_head_fit_procedure_diverges_and_says_so():
    """**Trial 0 is the closed-form counterpart of the arm's head, not the
    arm's head**, and that difference is registered rather than assumed away.

    The reason is the OOF discipline: ``harness.run_fold`` computes the
    held-out fold's predictions as part of its job, so using it per trial
    would mean computing 24 out-of-fold vectors and declining to read them.
    The cost is that the winner's number carries a procedure the 0.2520
    baseline did not use, and the record must say the winning configuration is
    re-run through ``train_cv`` before being quoted against the ladder.
    """
    procedure = phase7b.SEARCH_HEAD_PROCEDURE
    assert "closed form" in procedure["fit"]
    assert "early stopping" in procedure["ladder_arm_fit"]
    assert "train_cv" in procedure["cost"]
    assert "head_fit_procedure" in phase7b.BASELINE["also_varies"]

    # Every head in the space builds, and fits something usable.
    import numpy as np

    rng = np.random.default_rng(0)
    features, labels = rng.normal(size=(30, 5)), rng.normal(size=30)
    for trial in phase7b.configurations():
        head = phase7b.head_for(trial, seed=1337, fold=0)
        head.fit(features, labels)
        assert head.predict(features).shape == (30,)

    with pytest.raises(phase7b.Phase7BError, match="unknown head"):
        phase7b.head_for({"head": "transformer"}, seed=1, fold=0)


def test_the_only_oof_computation_is_for_the_selected_configuration():
    """**Exit criteria 3 and 4 in one place.** The winner gets its OWN band
    over its own seeds, and pooled out-of-fold predictions so a paired BCa has
    something to pair. Every patient must receive exactly one prediction, or
    the pooled vector is not the 237-vector the criterion assumes."""
    import numpy as np

    features, labels, ids, assignments = _cohort(n=60, dim=6, seed=3)
    trial = phase7b.configurations()[0]
    selected = phase7b.make_trial_record(trial, 0.3, 5)

    winner = phase7b.evaluate_selected_oof(
        selected,
        features_by_set={phase7b.BASE_SET: features},
        labels=labels,
        patient_ids=ids,
        assignments=assignments,
        make_head=lambda t, seed, fold: phase7b.head_for(t, seed=seed, fold=fold),
        inner_val_frac=0.2,
        seeds=[1337, 2024, 7],
    )
    assert len(winner["pooled_pccs"]) == 3
    assert len(winner["oof_by_seed"]) == 3
    for vector in winner["oof_by_seed"].values():
        assert len(vector) == len(ids)
        assert not np.isnan(np.asarray(vector)).any()
    assert winner["oof_ids"] == ids
    assert winner["head_procedure"] is phase7b.SEARCH_HEAD_PROCEDURE

    # Its own band, over its own seeds -- never the untuned arm's.
    from cleft.train.phase3 import seed_variance

    band = seed_variance(winner["pooled_pccs"])
    assert band["sd"] != phase7b.BASELINE["sd"]


def _write_baseline(directory, seeds, ids, vectors):
    """Fixtures in the REAL layout -- tier marker and all four columns.

    A fixture that omitted the marker would be a test conforming to the
    absence of the defect rather than exercising it, which is precisely the
    Phase 1 mistake this class of bug is recorded under.
    """
    from cleft.cluster_csv import write_predictions

    directory.mkdir(parents=True, exist_ok=True)
    for seed, vector in zip(seeds, vectors):
        write_predictions(
            directory / f"seed_{seed}__predictions.csv",
            [(pid, 3.0, value, i % 5) for i, (pid, value) in enumerate(zip(ids, vector))],
        )
    return directory


def _paths_for(directory, seeds):
    return {s: directory / f"seed_{s}__predictions.csv" for s in seeds}


def test_the_baseline_is_five_vectors_declared_one_per_seed(tmp_path):
    """**[MEASURED 2026-08-02] The baseline run holds no pooled predictions
    file.** It holds ``seed_<N>__predictions.csv`` per seed, so naming one
    would compare the tuned arm against a single seed of the arm it is tuning.

    They are declared as five FILE inputs rather than as the run directory:
    that directory contains ``code/``, a git worktree, whose rollup would
    cover the source tree and move whenever it was touched.
    """
    import numpy as np

    rng = np.random.default_rng(1)
    ids = list(range(20))
    seeds = [1337, 2024, 7]
    vectors = [rng.normal(size=20) for _ in seeds]
    directory = _write_baseline(tmp_path / "run", seeds, ids, vectors)
    paths = _paths_for(directory, seeds)

    loaded = phase7b.load_baseline_predictions(paths, seeds, ids)
    assert sorted(loaded) == sorted(seeds)
    for seed, vector in zip(seeds, vectors):
        # Six decimals, which is what the frozen writer emits and what
        # predictions_digest computes over -- not full float precision.
        assert loaded[seed] == pytest.approx(list(vector), abs=5e-7)

    # A seed with no declared input is refused by name.
    with pytest.raises(phase7b.Phase7BError, match=r"seeds \[99\]"):
        phase7b.load_baseline_predictions(paths, seeds + [99], ids)
    # A short file is refused rather than silently paired.
    with pytest.raises(phase7b.Phase7BError, match="missing 1 patients"):
        phase7b.load_baseline_predictions(paths, seeds, ids + [999])


def test_the_baseline_inputs_are_collected_by_the_seed_naming_convention():
    """``baseline_oof_seed_<seed>``, mirroring ``set_<name>``. A name that
    does not end in a seed is refused rather than skipped -- skipping would
    drop a vector from the comparison silently."""
    declared = {
        "manifest_v1": {"path": "/m"},
        "baseline_oof_seed_1337": {"path": "/a.csv"},
        "baseline_oof_seed_7": {"path": "/b.csv"},
    }
    assert phase7b.baseline_paths_from_inputs(declared) == {
        1337: "/a.csv", 7: "/b.csv"
    }
    assert phase7b.baseline_paths_from_inputs({"manifest_v1": {"path": "/m"}}) == {}

    with pytest.raises(phase7b.Phase7BError, match="does not name a seed"):
        phase7b.baseline_paths_from_inputs(
            {"baseline_oof_seed_final": {"path": "/x.csv"}}
        )


def test_the_baseline_rows_are_realigned_by_patient_id_not_assumed(tmp_path):
    """Nothing guarantees a per-seed CSV enumerates in the manifest's order,
    and a silent misalignment pairs shuffled vectors and still produces a
    plausible interval -- ``assert_row_order``'s lesson in the one place left
    that reads a CSV."""
    from cleft.cluster_csv import write_predictions

    ids = [10, 11, 12, 13]
    directory = tmp_path / "run"
    directory.mkdir()
    write_predictions(
        directory / "seed_1337__predictions.csv",
        [(13, 4.0, 4.0, 0), (10, 1.0, 1.0, 1), (12, 3.0, 3.0, 2),
         (11, 2.0, 2.0, 3)],
    )

    loaded = phase7b.load_baseline_predictions(
        _paths_for(directory, [1337]), [1337], ids
    )
    assert loaded[1337] == [1.0, 2.0, 3.0, 4.0], (
        "the vector was not realigned to the requested patient order"
    )


def test_the_paired_comparison_is_per_seed_and_never_averages(tmp_path):
    """**Averaging the seeds is an ENSEMBLE, and it moves the bar.**

    Mean-pooling prediction vectors cancels independent error, so the averaged
    vector's PCC exceeds the mean of the five -- the averaged baseline would
    not score what the baseline scores. Asserted numerically here, because it
    is the whole reason the comparison is shaped this way.
    """
    import numpy as np

    from cleft.eval.metrics import pcc

    rng = np.random.default_rng(7)
    truth = rng.normal(size=120)
    seeds = [1337, 2024, 7, 99, 12345]
    baseline = {s: truth * 0.30 + rng.normal(scale=1.0, size=120) for s in seeds}
    winner = {s: truth * 0.45 + rng.normal(scale=1.0, size=120) for s in seeds}

    # The measured reason for refusing to average.
    per_seed_mean = float(np.mean([pcc(truth, v) for v in baseline.values()]))
    averaged = float(pcc(truth, np.mean(list(baseline.values()), axis=0)))
    assert averaged > per_seed_mean, (
        "averaging did not raise the correlation here, so this fixture no "
        "longer demonstrates why the seeds must not be pooled"
    )

    result = phase7b.paired_comparison(
        truth=truth, winner_by_seed=winner, baseline_by_seed=baseline,
        winner_sd=0.02, n_boot=400,
    )
    assert len(result["per_seed"]) == len(seeds)
    assert {e["seed"] for e in result["per_seed"]} == set(seeds)
    assert result["n_seeds"] == 5
    assert result["mean_delta"] == pytest.approx(
        float(np.mean([e["delta"] for e in result["per_seed"]])), abs=1e-9
    )
    # Both conditions are required, and the verdict must agree with them.
    assert result["claimable"] == (
        result["n_excluding_zero"] == result["n_seeds"]
        and result["same_direction"]
        and result["exceeds_threshold"]
    )
    assert result["rule"] is phase7b.COMPARISON_RULE

    # The baseline's own sd is RECOMPUTED from its five vectors, not inherited
    # from the record -- PLAN §4.12.1 in the place it would be easiest to skip.
    assert result["baseline_sd_recomputed"] == pytest.approx(
        float(np.std([e["baseline_pcc"] for e in result["per_seed"]], ddof=1))
    )


def test_the_comparison_refuses_arms_that_do_not_share_seeds():
    """Pairing by seed number only means anything because both arms split with
    ``inner_val_split(outer_train, frac, seed, fold)`` -- one seed is the same
    held-out patients on both sides. Mismatched seeds would align two lists
    that describe different data conditions."""
    import numpy as np

    truth = np.arange(10, dtype=float)
    with pytest.raises(phase7b.Phase7BError, match="do not share their seeds"):
        phase7b.paired_comparison(
            truth=truth,
            winner_by_seed={1337: list(truth), 2024: list(truth)},
            baseline_by_seed={1337: list(truth), 7: list(truth)},
            winner_sd=0.01, n_boot=50,
        )


def test_the_winner_keeps_every_seeds_vector_not_the_first():
    """An earlier version stored whichever seed ran first and called it "the
    OOF" -- which would have paired one arbitrary run of the tuned arm against
    the baseline and reported it as the arm."""
    features, labels, ids, assignments = _cohort(n=60, dim=6, seed=5)
    selected = phase7b.make_trial_record(phase7b.configurations()[0], 0.3, 5)
    seeds = [1337, 2024, 7]

    winner = phase7b.evaluate_selected_oof(
        selected,
        features_by_set={phase7b.BASE_SET: features},
        labels=labels,
        patient_ids=ids,
        assignments=assignments,
        make_head=lambda t, seed, fold: phase7b.head_for(t, seed=seed, fold=fold),
        inner_val_frac=0.2,
        seeds=seeds,
    )
    assert sorted(winner["oof_by_seed"]) == sorted(seeds)
    vectors = list(winner["oof_by_seed"].values())
    assert any(a != b for a, b in zip(vectors[0], vectors[1])), (
        "two seeds produced identical out-of-fold vectors; the per-seed "
        "comparison would be comparing one run against itself"
    )


def test_the_axis_verdicts_follow_from_the_recorded_trial_scores():
    """**Three verdicts, each recomputed from the numbers it rests on.**

    The pooling refutation is a monotonicity claim, the head verdict is a
    range against a threshold, and the ensemble verdict is a delta against
    trial 0 -- all arithmetic, so none of them is taken on trust.
    """
    verdicts = phase7b.SEARCH_AXIS_VERDICTS

    # POOLING: deeper is uniformly better. Checked on the best score at each
    # depth, which is the claim -- not on cls and mean_patch separately, where
    # the two orderings cross at block 6.
    by_block = verdicts["pooling"]["by_block"]
    assert sorted(by_block) == list(phase7b.POOLING_BLOCKS)
    best_at = [max(by_block[block]) for block in sorted(by_block)]
    assert best_at == sorted(best_at), (
        f"the pooling claim is monotone improvement with depth, but the best "
        f"score per block runs {best_at}"
    )
    assert "REFUTED" in verdicts["pooling"]["verdict"]

    # ...and the deepest block matches trial 0, because it IS trial 0's
    # representation. The extraction asserted bit-identity; this asserts the
    # search saw the same thing.
    assert max(by_block[12]) == pytest.approx(0.2015, abs=5e-5)

    # HEAD: null means the range sits inside the exhaustion criterion.
    assert verdicts["head"]["range"] < phase7b.exhaustion_threshold()
    assert "NULL" in verdicts["head"]["verdict"]

    # ENSEMBLE: the delta over trial 0 must clear the same criterion, or the
    # "only axis that moves" claim is not supported by its own number.
    best = verdicts["ensemble"]["best"]
    assert best["over_trial_0"] == pytest.approx(
        best["inner_val_pcc"] - max(by_block[12]), abs=5e-5
    )
    assert best["over_trial_0"] > phase7b.exhaustion_threshold()


def test_the_search_outcome_is_recomputed_from_its_own_per_seed_scores():
    """**The inner-val lead did not transfer, and the winner is claimably
    WORSE than the arm it tuned.** Every figure re-derived: the mean and SD
    from the five seeds, the threshold from the two arms' own spreads, and the
    two verdicts from those.
    """
    import numpy as np

    from cleft.train.phase3 import combined_claimable_delta, seed_variance

    outcome = phase7b.SEARCH_AXIS_VERDICTS["outcome"]
    per_seed = outcome["oof_per_seed"]
    assert len(per_seed) == phase7b.BASELINE["seeds"] == 5

    band = seed_variance(per_seed)
    assert outcome["oof_pcc"] == pytest.approx(band["mean"], abs=5e-5)
    assert outcome["oof_sd"] == pytest.approx(band["sd"], abs=5e-5)

    against = outcome["against_baseline"]
    assert against["baseline"] == phase7b.BASELINE["pcc"]
    delta = band["mean"] - phase7b.BASELINE["pcc"]
    assert against["delta"] == pytest.approx(delta, abs=5e-5)
    assert delta < 0, "the winner did not lose, so the record is describing another run"

    threshold = combined_claimable_delta(
        band["sd"], len(per_seed),
        phase7b.BASELINE["sd"], phase7b.BASELINE["seeds"],
    )
    assert against["threshold"] == pytest.approx(threshold["arm_means_95"], abs=5e-5)
    assert against["single_run_95"] == pytest.approx(
        threshold["single_run_95"], abs=5e-5
    )
    # **[UPDATED 2026-08-04] `claimable` is no longer condition 2's answer.**
    # It was, and that identity is the defect: PLAN §4.3 needs both
    # conditions, and condition 1 -- sitting in this run's own metrics.json
    # the whole time -- says 1 of 5 intervals excludes zero. The condition-2
    # arithmetic is unchanged and still checked; it just no longer decides.
    # phase7b.CLAIMABLY_WORSE_WITHDRAWN.
    assert against["claimable_condition_2_only"] == (
        abs(delta) > threshold["arm_means_95"]
    )
    assert against["claimable"] is False
    condition_1 = against["condition_1"]
    assert condition_1["n_excluding_zero"] < condition_1["n_seeds"]
    assert against["survives_single_run_95"] == (
        abs(delta) > threshold["single_run_95"]
    )

    # The withdrawn reading is kept beside the current one: the sequence --
    # computed, not read, then withdrawn -- is the finding.
    assert "claimably WORSE" in against["reading_before_condition_1"]
    assert "UNRESOLVED" in against["reading"]
    assert phase7b.CLAIMABLY_WORSE_WITHDRAWN["condition_1"]["claimable"] is False

    # "Every seed falls short" is a claim about the five, so it is checked
    # against them rather than against the mean.
    assert outcome["every_seed_short"] == all(
        value < phase7b.BASELINE["pcc"] for value in per_seed
    )
    assert max(per_seed) < phase7b.BASELINE["pcc"]

    # And the exhaustion verdict follows from its own window and threshold.
    exhaustion = outcome["exhaustion"]
    assert exhaustion["threshold"] == pytest.approx(
        phase7b.exhaustion_threshold(), abs=5e-5
    )
    assert exhaustion["exhausted"] == (
        exhaustion["best_gain_in_window"] <= exhaustion["threshold"]
    )
    assert exhaustion["window"] == phase7b.EXHAUSTION_WINDOW

    # The inner-val lead the record quotes must be the one the trial scores say.
    verdicts = phase7b.SEARCH_AXIS_VERDICTS
    assert outcome["inner_val_lead"] == pytest.approx(
        verdicts["ensemble"]["best"]["over_trial_0"], abs=5e-5
    )
    assert outcome["inner_val_pcc"] == pytest.approx(
        verdicts["ensemble"]["best"]["inner_val_pcc"], abs=5e-5
    )
    assert "NOTHING BEAT 0.2520" in verdicts["verdict"]


def test_the_baseline_matches_the_arm_it_claims_to_tune():
    """The tuned arm is compared against a specific measured cell, so the
    recorded baseline must be that cell -- not a number typed beside it."""
    from cleft import ladder

    record = ladder.STAGE_D1_AT_G1
    backbone = phase7b.BASELINE["backbone"]
    init = phase7b.BASELINE["init"]

    assert phase7b.BASELINE["pcc"] == pytest.approx(
        record["cells"][backbone][record["inits"].index(init)], abs=5e-5
    )
    assert phase7b.BASELINE["sd"] == pytest.approx(
        record["sd"][backbone][init], abs=5e-5
    )
    assert phase7b.BASELINE["seeds"] == ladder.SEEDS_BY_KIND["transformer"]

    # It must genuinely be the best cell that carries a number, or "tune the
    # best arm" is describing a different arm.
    everything = [
        (table["cells"][b][i], b, table["inits"][i])
        for table in (ladder.STAGE_D_AT_G2, record)
        for b in table["cells"]
        for i in range(len(table["inits"]))
    ]
    best = max(everything)
    assert best == (phase7b.BASELINE["pcc"], backbone, init)
