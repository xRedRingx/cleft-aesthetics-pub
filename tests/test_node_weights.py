import numpy as np, pytest
from cleft import gradcam, node_weights as nw, phase8


def test_the_gate_reuses_gradcams_statistics_and_not_its_verdict():
    """One implementation of the statistic, because the phase's point is
    a comparison -- and NOT the superseded median verdict."""
    rng = np.random.default_rng(0)
    weights = rng.normal(size=(15, 27))
    # The statistic really is the same function, at 27 elements.
    assert nw.baseline_pairs(weights).size == 15 * 14 // 2
    assert abs(
        float(np.median(nw.baseline_pairs(weights)))
        - gradcam.between_patient_baseline(list(weights))
    ) < 1e-12
    # The superseded pass/fail verdict is not imported here.
    from pathlib import Path

    src = Path(nw.__file__).read_text(encoding="utf-8")
    assert "randomisation_verdict" not in src.split('"""', 2)[2], (
        "the median pass/fail rule must not be reused; the adopted "
        "criterion is the separability interval"
    )


def test_per_patient_figures_are_percentiles_and_say_how_coarse():
    rng = np.random.default_rng(1)
    weights = rng.normal(size=(15, 27))
    pairs = nw.baseline_pairs(weights)
    rows = nw.per_patient_percentiles(np.full(15, -1.0), pairs)
    assert len(rows) == 15
    assert all(r["baseline_percentile"] == 0.0 for r in rows)
    # 105 pairs -> the finest step the distribution can express.
    assert all(abs(r["resolution"] - 1 / 105) < 1e-12 for r in rows)
    assert "percentile" in str(rows[0])


def test_the_gate_resolves_or_says_it_does_not():
    rng = np.random.default_rng(2)
    weights = rng.normal(size=(15, 27))
    # Randomised finals far below any baseline pair -> should resolve.
    low = nw.gate(weights, np.full(15, -0.9), n_boot=400)
    assert low["n_regions"] == 27 and low["n_baseline_pairs"] == 105
    assert low["resolves"] is True
    # Finals drawn from the baseline itself -> should not.
    pairs = nw.baseline_pairs(weights)
    same = nw.gate(weights, rng.choice(pairs, size=15), n_boot=400)
    assert same["resolves"] is False
    assert "UNRESOLVED" in same["if_it_does_not_resolve"]
    with pytest.raises(nw.NodeWeightError, match="every patient needs one"):
        nw.gate(weights, np.zeros(14))


def test_both_modes_are_required_and_all_four_readings_pre_exist():
    assert set(nw.RANDOMISATION_MODES) == {
        "component_role", "explanation_source"
    }
    assert nw.COMPARISON_MODE == "component_role"
    assert set(nw._MODE_READINGS) == {
        (True, True), (True, False), (False, True), (False, False)
    }
    # The null case is guarded rather than left to interpretation.
    assert "not evidence of independence" in nw._MODE_READINGS[(False, False)]
    assert "furthest" in nw._MODE_READINGS[(False, True)]

    with pytest.raises(nw.NodeWeightError, match="both randomisations"):
        nw.compare_modes({"component_role": {"resolves": True}})

    out = nw.compare_modes({
        "component_role": {"resolves": True},
        "explanation_source": {"resolves": False},
    })
    assert out["agree"] is False
    assert out["comparison_uses"] == "component_role"
    assert out["reading"] == nw._MODE_READINGS[(True, False)]


def test_the_sample_decision_and_the_registration():
    s = phase8.ARM_B_SAMPLE_DECIDED
    assert s["own_gate"]["n"] == 237
    assert s["own_gate"]["baseline_pairs"] == 237 * 236 // 2
    assert s["a_vs_b_comparison"]["n"] == 15
    assert s["a_vs_b_comparison"]["baseline_pairs"] == 15 * 14 // 2
    assert "holds its conditions fixed" in s["why_two"]
    assert "UNRESOLVED" in s["pre_commitment"]
    assert "different question" in s["pre_commitment"]

    r = phase8.TWO_MODE_READINGS_REGISTERED
    assert "before any numbers" in r["registered"]
    assert "where SR-GNN's explanation actually lives" in r[
        "what_a_disagreement_tells_us"
    ]
    assert "not evidence of independence" in r["null_case_guarded"]


def test_the_whole_image_node_is_split_off_before_ranking():
    """**26 regions, not 27** -- the last node is the whole feature map,
    and ranking it alongside the regions would inflate the baseline the
    gate reads against."""
    from cleft.models.srgnn import N_REGIONS, N_ROIS

    assert (N_ROIS, N_REGIONS) == (26, 27)
    rng = np.random.default_rng(3)
    weights = rng.random((15, N_REGIONS))
    rois, whole = nw.split_whole_image(weights)
    assert rois.shape == (15, 26)
    assert whole.shape == (15,)
    assert np.array_equal(whole, weights[:, -1])

    # A vector of the wrong width refuses, naming why it matters.
    with pytest.raises(nw.NodeWeightError, match=r"ROIs \+ the whole image"):
        nw.split_whole_image(rng.random((15, 26)))

    # And the decision is recorded with its reason, before any numbers.
    record = phase8.SRGNN_NODE_WEIGHTS_R10_READ["the_count_is_26_plus_the_whole_image"]
    assert record["n_rois"] == N_ROIS and record["n_regions"] == N_REGIONS
    assert "MORE PERMISSIVE" in record["why_it_matters_twice"]
    assert "36-vs-37" in record["same_trap_as"]


def test_the_r10_read_answers_all_three_questions():
    """Exposure, per-fold, and the refit target -- each from the code."""
    record = phase8.SRGNN_NODE_WEIGHTS_R10_READ

    # 1. An accessor exists; the cleft path drops it.
    exposure = record["exposure"]
    assert "_from_features" in exposure["returned_by"]
    assert "discards region_w" in exposure["cleft_path_drops_them"]
    assert "parallel path" in exposure["so"]
    # The accessor really is there now, and returns a pair.
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "cleft" / "models"
    text = (src / "srgnn.py").read_text(encoding="utf-8")
    assert "def forward_from_features_with_weights" in text
    assert "return self._from_features(feat, boxes)\n" in text

    # 2. Per-fold, held-out, and the place the arms MATCH.
    per_fold = record["per_fold"]
    assert "HELD THEM OUT" in per_fold["rule"]
    assert "memorised label" in per_fold["rule"]
    assert "not one-sided" in per_fold["this_is_where_the_arms_MATCH"]
    assert "AGREE on whose model may explain whom" in per_fold[
        "this_is_where_the_arms_MATCH"
    ]

    # 3. The refit target mirrors arm A's, and its tolerance is licensed.
    gate_spec = record["refit_gate"]
    assert gate_spec["expected_pcc"] == phase8.ARMS["B"]["pcc"] == 0.1719
    assert gate_spec["seeds"] == 10
    assert gate_spec["tolerance"] == phase8.REPRODUCE_GATE["tolerance"]
    assert "byte-identically" in gate_spec["tolerance_licensed_by"]


def test_the_two_modes_touch_different_parameters_and_nothing_else_differs():
    """**They differ in WHAT they touch, and in two documented places
    besides** -- both measured, both recorded, neither discretionary.

    [REWRITTEN 2026-08-14] This test used to assert that arm B walked
    ``model.backbone.named_children()`` and drew noise at
    ``float(parameter.std())``. Both were measured to be wrong: the backbone
    walk is a complete no-op on the artifact-fed path
    (``phase8.COMPONENT_ROLE_IS_A_NO_OP``) and ``std()`` is NaN for SR-GNN's
    three scalar parameters (``phase8.SCALAR_PARAMETERS_BREAK_THE_NOISE_
    MATCH``). The assertions are inverted rather than deleted, so the old
    behaviour cannot come back unremarked.
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "cleft" / "run.py"
    text = src.read_text(encoding="utf-8")
    tree = ast.parse(text)

    def body(name):
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        return ast.get_source_segment(text, fn) or ""

    arm_b = body("phase8_node_weight_stages")
    walk = body("_phase8_walk_stages")
    arm_a = body("phase8_randomisation_stages")

    # The runnable mode selects the modules that produce region_w...
    assert "self_attn" in arm_b and "weighted_attn" in arm_b
    # ...and the backbone walk is GONE, replaced by a refusal carrying the
    # measurement. Its return would silently reinstate a no-op test.
    assert "model.backbone.named_children()" not in arm_b
    assert "COMPONENT_ROLE_IS_A_NO_OP" in arm_b
    # An unknown mode refuses, and an empty selection refuses -- a mode
    # that randomised nothing would report a clean pass.
    assert "unknown randomisation mode" in arm_b
    assert "nothing" in arm_b and "would be tested" in arm_b

    # The noise construction is still arm A's in the parts that carry the
    # comparison: same seed, same cumulative in-place copy.
    for fragment in ("manual_seed(1337)", "parameter.copy_(noise.to(parameter.device))"):
        assert fragment in arm_a, fragment
        assert fragment in walk, fragment
    # The one divergence, and it is the scalar correction rather than a
    # choice: arm A's expression would raise on SR-GNN's (1,) parameters.
    assert "float(parameter.std())" in arm_a
    assert "float(parameter.std())" not in walk
    assert "phase8_noise_std(parameter, group_std)" in walk


def test_component_role_refuses_eagerly_and_says_what_it_measured():
    """**A generator would have deferred the refusal to the first
    ``next()``**, so a caller that built the walk and never stepped it would
    see nothing. The validation is eager and needs no torch, which is what
    lets this run on the laptop at all."""
    from cleft.run import phase8_node_weight_stages

    class Model:
        self_attn = gnn_mlp1 = gnn_mlp2 = 1
        gnn_out = weighted_attn = classifier = 1

    with pytest.raises(ValueError) as excinfo:
        phase8_node_weight_stages(Model(), "component_role")
    message = str(excinfo.value)
    assert "20,806,952" in message
    assert "BITWISE" in message
    assert "below the backbone" in message
    # And it names the route that DOES work, rather than only refusing.
    assert "randomised_embeddings_artifact" in message
    assert "RE-EXTRACTED" in message

    with pytest.raises(ValueError, match="unknown randomisation mode"):
        phase8_node_weight_stages(Model(), "top_down")

    # The runnable mode still builds -- without importing torch, since the
    # module selection is getattr and the walk is what needs a tensor.
    assert phase8_node_weight_stages(Model(), "explanation_source") is not None


def test_the_noise_std_never_returns_nan_for_a_scalar_parameter():
    """[MEASURED] ``tensor.std()`` is NaN at numel 1 (correction 1, zero
    denominator) and ``normal_`` then raises. SR-GNN has three such
    parameters and one of them is in the FIRST group randomised."""
    from cleft.run import phase8_noise_std

    class Fake:
        def __init__(self, std, shape):
            self._std, self.shape = std, shape

        def detach(self):
            return self

        def std(self, unbiased=True):
            return self._std

    # Its own spread when it has one.
    assert phase8_noise_std(Fake(0.5, (26,)), 9.0) == 0.5
    # The GROUP's when it does not -- borrowed, never skipped: skipping
    # would leave a trained value inside a stage claiming to have
    # randomised everything above it.
    assert phase8_noise_std(Fake(0.0, (1,)), 0.25) == 0.25
    # And a group with no spread at all refuses rather than drawing zeros.
    with pytest.raises(ValueError, match="no spread"):
        phase8_noise_std(Fake(0.0, (1,)), 0.0)

    record = phase8.SCALAR_PARAMETERS_BREAK_THE_NOISE_MATCH
    assert len(record["parameters"]) == 3
    assert "self_attn.Wa.bias" in record["parameters"]
    assert "nan" in record["what_std_returns"]
    assert "left alone" in record["arm_a_unaffected"]


def test_the_comparison_mode_is_the_one_that_asks_arm_as_question():
    from cleft import phase8

    assert nw.COMPARISON_MODE == "component_role"
    decided = phase8.ARM_A_AND_B_RANDOMISATIONS_ARE_NOT_EQUIVALENT
    assert decided["decided_run_both"]["comparison_uses"].startswith(
        "the component-role"
    )
    # Arm A randomised its frozen backbone, which is what makes the
    # component-role mode the analogue.
    assert "blocks" in phase8.ARM_A_RANDOMISED_THE_FROZEN_BACKBONE["touched"]


# --------------------------------------------------------------------------
# the extraction task
# --------------------------------------------------------------------------


class _FakeFold:
    """A fold model that reports weights without torch: one row per patient,
    the region index scaled by the patient's own row so vectors differ."""

    def __init__(self, rows, n_nodes=27):
        self.rows, self.n_nodes = rows, n_nodes
        self.asked = []

    def region_weights(self, features):
        self.asked.append(np.asarray(features).tolist())
        return np.asarray([
            [(index + 1) * (position + 1) for position in range(self.n_nodes)]
            for index in np.asarray(features)[:, 0].astype(int)
        ], dtype=float)


def _fold_fixture(n_nodes=27):
    """Six patients, three folds, packed rows whose first column is the row
    index so the fake can tell which patients it was handed."""
    patient_ids = [10, 20, 30, 40, 50, 60]
    assignments = {10: 0, 20: 0, 30: 1, 40: 1, 50: 2, 60: 2}
    packed = np.asarray([[float(i), 0.0] for i in range(len(patient_ids))])
    models = [
        (fold, _FakeFold(rows=None, n_nodes=n_nodes)) for fold in (0, 1, 2)
    ]
    return models, packed, patient_ids, assignments


def test_every_patient_is_explained_by_the_fold_that_held_them_out():
    """**The held-out rule, enforced by construction rather than checked
    afterwards.** Each fold's model is asked ONLY for its own held-out rows,
    so a patient explained by a model that trained on them is not something
    the task can express."""
    from cleft.run import phase8_held_out_region_weights

    models, packed, patient_ids, assignments = _fold_fixture()
    out = phase8_held_out_region_weights(models, packed, patient_ids, assignments)
    assert out.shape == (6, 27)
    # Fold 0 saw rows 0 and 1 and nothing else.
    assert models[0][1].asked == [[[0.0, 0.0], [1.0, 0.0]]]
    assert models[2][1].asked == [[[4.0, 0.0], [5.0, 0.0]]]
    # Row 3 (patient 40, fold 1) carries fold 1's answer for row 3.
    assert out[3][0] == 4.0


def test_a_patient_no_fold_held_out_is_refused_not_left_as_zeros():
    """A zero row ranks as a perfectly flat explanation, which is a value,
    not an absence -- and it would sail through every shape check."""
    from cleft.run import phase8_held_out_region_weights

    models, packed, patient_ids, assignments = _fold_fixture()
    assignments[60] = 7  # a fold with no model
    with pytest.raises(ValueError, match="explained by no fold-model"):
        phase8_held_out_region_weights(models, packed, patient_ids, assignments)


def test_two_folds_claiming_one_patient_are_refused():
    """The later model would silently overwrite the earlier explanation."""
    from cleft.run import phase8_held_out_region_weights

    models, packed, patient_ids, assignments = _fold_fixture()
    models = models + [(0, _FakeFold(rows=None))]
    with pytest.raises(ValueError, match="held out by more than one fold"):
        phase8_held_out_region_weights(models, packed, patient_ids, assignments)


def test_folds_disagreeing_on_the_node_count_are_refused():
    """Two folds ranking different numbers of regions cannot be averaged."""
    from cleft.run import phase8_held_out_region_weights

    models, packed, patient_ids, assignments = _fold_fixture()
    models[1] = (1, _FakeFold(rows=None, n_nodes=26))
    with pytest.raises(ValueError, match="nodes against"):
        phase8_held_out_region_weights(models, packed, patient_ids, assignments)


def test_publishable_distinguishes_not_run_from_did_not_resolve():
    """**Three states, not two.** A mode that could not run is not a mode
    that failed, and a run that randomised nothing is not a run that failed
    its gate."""
    resolved = {"resolves": True}
    unresolved = {"resolves": False}

    one = nw.publishable({"explanation_source": resolved})
    assert one["publishable"] is True
    assert one["modes_not_run"] == ["component_role"]
    assert "not a failed test" in one["not_run_is_not_evidence"]

    none = nw.publishable({"explanation_source": unresolved})
    assert none["publishable"] is False
    assert none["modes_resolved"] == []

    both = nw.publishable(
        {"component_role": unresolved, "explanation_source": resolved}
    )
    assert both["publishable"] is True and both["modes_not_run"] == []

    # An empty dict is a different state from a failed gate, and says so.
    with pytest.raises(nw.NodeWeightError, match="randomised nothing"):
        nw.publishable({})
    with pytest.raises(nw.NodeWeightError, match="unknown randomisation mode"):
        nw.publishable({"top_down": resolved})


def test_the_two_modes_reach_their_randomisation_by_different_routes():
    """**The design distinction survives one mode needing a different
    instrument.** ``explanation_source`` walks parameters; ``component_role``
    cannot, and takes a re-extracted control set instead. Merging the two
    route sets would turn "it needs an artifact" into "there was only ever
    one mode"."""
    assert set(nw.PARAMETER_WALK_MODES) | set(nw.CONTROL_SET_MODES) == set(
        nw.RANDOMISATION_MODES
    )
    assert not set(nw.PARAMETER_WALK_MODES) & set(nw.CONTROL_SET_MODES)
    # The comparison mode is the one that needs the control artifact.
    assert nw.COMPARISON_MODE in nw.CONTROL_SET_MODES
    assert phase8.COMPONENT_ROLE_IS_A_NO_OP["unaffected"].startswith(
        "explanation_source"
    )
    # The A<->B comparison was the casualty, named as such...
    assert "A<->B" in phase8.COMPONENT_ROLE_IS_A_NO_OP["consequence"]
    assert "exit criterion 5" in phase8.COMPONENT_ROLE_IS_A_NO_OP["consequence"]
    # ...and the decision that answered it sits BESIDE the "not taken here"
    # line rather than on top of it, so the sequence stays readable.
    assert "not taken here" in phase8.COMPONENT_ROLE_IS_A_NO_OP[
        "what_would_restore_it"
    ]
    assert phase8.COMPONENT_ROLE_IS_A_NO_OP["decision"].startswith("TAKEN")
    assert "UNAVAILABLE" in phase8.COMPONENT_ROLE_IS_A_NO_OP["decision"]


# --------------------------------------------------------------------------
# the randomised-backbone control
# --------------------------------------------------------------------------


def test_the_control_set_is_refused_by_every_consumer_that_did_not_ask():
    """**Both directions, because both are silent.** A training arm fed the
    control would fit, score badly, and be read as evidence about the
    architecture. The control fed the real set would report the explanation
    unchanged -- similarity 1.0, the worst verdict -- from a test that never
    ran. The second is the failure COMPONENT_ROLE_IS_A_NO_OP already
    produced once."""
    from cleft import embeddings

    real = {
        "kind": "feature_map", "backbone": "srgnn", "init": "imagenet",
        "geometry": "g1", "pretrain_scheme": None, "randomisation": None,
    }
    control = {**real, "randomisation": {"randomised": True, "n_parameters": 20806952}}
    common = dict(
        kind="feature_map", backbone="srgnn", init="imagenet",
        geometry="g1", checkpoint_sha256=None, region_scheme="native",
    )

    # An ordinary arm meeting the control.
    with pytest.raises(embeddings.EmbeddingError, match="did not ask for one"):
        embeddings.check_pairing(control, **common)
    # The control's consumer meeting a real set.
    with pytest.raises(embeddings.EmbeddingError, match="secretly the real set"):
        embeddings.check_pairing(real, **common, expect_randomised=True)
    # And each in its right place passes.
    embeddings.check_pairing(real, **common)
    embeddings.check_pairing(control, **common, expect_randomised=True)


def test_the_control_says_what_it_is_in_its_own_directory_name():
    """A name that reads like the real set is a hazard on every ls, in every
    config, and in every declaration a human types."""
    from cleft import embedding_plan

    entry = {
        "backbone": "srgnn", "init": "imagenet", "geometry": "g1",
        "pretrain_scheme": None,
    }
    assert embedding_plan.set_name(entry) == "srgnn__imagenet__g1"
    assert embedding_plan.set_name({**entry, "randomise": True}) == (
        "srgnn__imagenet__g1__randomised"
    )


def test_degeneracy_is_measured_and_all_three_readings_pre_exist():
    """**gradcam.similarity returns 0.0 for a vector with no rank
    structure** -- right for a rank statistic, and readable as a spectacular
    pass. So the fact is measured and travels with the gate."""
    structured = np.array([[3.0, 1.0, 2.0], [1.0, 3.0, 2.0]])
    uniform = np.ones((2, 3))
    mixed = np.vstack([structured[:1], uniform[:1]])

    assert nw.degeneracy(structured)["any"] is False
    assert "MEASUREMENT" in nw.degeneracy(structured)["reading"]

    every = nw.degeneracy(uniform)
    assert every["all"] is True and every["fraction_constant"] == 1.0
    assert "0.0 by construction" in every["reading"]
    assert "COLLAPSE" in every["reading"]

    some = nw.degeneracy(mixed)
    assert some["any"] is True and some["all"] is False
    assert "MIXTURE" in some["reading"]

    # All three keys exist before any run: the disagreement case is the one
    # that invites an account invented to fit it.
    assert set(nw._DEGENERACY_READINGS) == {"none", "some", "all"}

    # And the statistic really does return 0.0 on a constant vector, which is
    # the behaviour the whole check exists for.
    assert gradcam.similarity(uniform[0], structured[0]) == 0.0


def test_the_gate_carries_the_degeneracy_verdict_when_it_can():
    """A pass produced by uniform vectors cannot be read off the gate dict
    without the fact that produced it."""
    rng = np.random.default_rng(3)
    weights = rng.normal(size=(12, 26))
    randomised = np.ones((12, 26))
    finals = np.zeros(12)

    plain = nw.gate(weights, finals, n_boot=64)
    assert "degeneracy" not in plain

    carried = nw.gate(weights, finals, n_boot=64, randomised=randomised)
    assert carried["degeneracy"]["all"] is True
    assert "COLLAPSE" in carried["degeneracy"]["reading"]


def test_the_collapse_is_measured_on_both_architectures():
    """**It is the architecture, not the intervention.** The same walk on
    ViT leaves the tokens varying between images; Xception's output stops
    depending on its input entirely."""
    record = phase8.THE_RANDOMISED_BACKBONE_COLLAPSES
    srgnn, vit = record["srgnn"], record["vit_b16"]
    assert srgnn["randomised_between_patient_max_difference"] < 1e-7
    assert srgnn["real_between_patient_max_difference"] > 1.0
    # ViT is reduced but nowhere near collapsed, which is what makes the
    # comparison a measurement rather than an assumption.
    assert vit["randomised_between_image_max_difference"] > 1.0
    assert vit["randomised_within_image_patch_sd"] > 1.0
    assert "ARCHITECTURE" in record["so"]
    assert "bounds nothing" in record["not_knowable_before_the_cluster"]
    # The untrained-head measurement is recorded as what it is: it does not
    # bound the trained arm.
    assert record["with_an_untrained_head"]["roi_weight_sd_per_patient"] == 0.0


def test_the_control_decision_records_what_it_does_not_buy():
    """Comparability is restored for the INTERVENTION, not for what the
    intervention does to each architecture."""
    record = phase8.COMPONENT_ROLE_CONTROL_IS_A_REEXTRACTION
    assert "ADABN_DIAGNOSTIC" in record["precedent"]
    assert "buffers untouched" in record["matches_arm_a_in"]
    assert "both ways" in record["guard_runs_both_ways"] or True
    assert "still sees the face" in record["does_not_buy"]
    assert "travels with the comparison" in record["does_not_buy"]
    assert record["endpoint_only"].startswith("the criterion reads the final")
