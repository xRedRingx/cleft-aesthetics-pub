"""Phase 9: the framing decision, the ledger's append-only teeth, the
prototype registrations, and the medoid machinery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pathlib

from cleft import phase9, results_ledger, roadb

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# the framing decision
# --------------------------------------------------------------------------


def test_the_framing_decision_is_recorded_with_its_three_consequences():
    """2026-08-16: Road A is the main project, Road B the annex. Dropped
    convergence, unchanged evidence, unchanged paths."""
    framing = phase9.ROAD_B_IS_THE_ANNEX
    assert framing["decided"].startswith("2026-08-16")
    assert "annex" in framing["framing"]
    consequences = framing["consequences"]
    assert consequences["convergence_deliverable"].startswith("DROPPED")
    assert "unchanged" in consequences["evidential_status"]
    assert "roadb_ prefixes stay" in consequences["naming"]
    assert "never two ledgers" in framing["ledger_field"]


def test_roadb_docstring_carries_the_dated_reframing_note():
    """The 'control arm' inversion is corrected in prose, dated, with the
    original paragraph left standing as history."""
    doc = roadb.__doc__
    assert "It is the control arm." in doc  # history preserved
    assert "REFRAMED 2026-08-16" in doc
    assert "ANNEX" in doc
    assert "never a\nparallel or co-equal road" in doc or (
        "never a parallel or co-equal road" in doc.replace("\n", " ")
    )
    assert "no convergence at Phase 9" in doc.replace("\n", " ")


# --------------------------------------------------------------------------
# the ledger
# --------------------------------------------------------------------------

#: The BORN population's cumulative checksum, pinned 2026-08-16. This
#: constant is the append-only rule's teeth: appending entries leaves it
#: valid (the prefix is unchanged); rewriting or deleting ANY existing
#: entry breaks it. When entries are appended, add a NEW pin below for
#: the new length -- never touch this one.
#:
#: **[RE-DERIVED 2026-09-05]** the pins for n >= 19 below were
#: re-derived once, for the anonymisation recorded at
#: ``results_ledger.CHAIN_RE_DERIVED``, which also carries the twelve
#: superseded values. **This pin was NOT among them**: the first
#: rewritten row sits at position 19, so the born prefix hashes
#: exactly as it did on 2026-08-16 and the instruction above still
#: holds unbroken.
BORN_ENTRIES = 17
BORN_CHECKSUM = "c46fd95054d3eaebb8dd0e1c6e5377a6fb349c7b8c9347362ca55716cb8cc779"


def test_the_ledger_is_append_only_by_checksum():
    assert len(results_ledger.ENTRIES) >= BORN_ENTRIES
    assert results_ledger.cumulative_checksum(BORN_ENTRIES) == BORN_CHECKSUM


def test_the_ledger_validates_and_carries_the_registered_structure():
    results_ledger.validate()
    assert results_ledger.LEDGER_BORN == "2026-08-16"
    assert results_ledger.STATUSES == (
        "CLAIMABLE", "WITHDRAWN", "UNRESOLVED-WITHDRAWN", "VOID",
        "DESCRIPTIVE",
    )
    assert results_ledger.FRAMINGS == ("main", "additional")
    # One margin table, by reference -- no second copy to drift.
    assert results_ledger.MARGIN_TABLE is roadb.CONDITION_1_MARGIN_STRUCTURE
    for entry in results_ledger.ENTRIES:
        for field in ("id", "claim", "status", "framing", "phase", "record",
                      "condition_1", "condition_2", "run_dirs", "date",
                      "caveats", "corrects"):
            assert field in entry, (entry["id"], field)


def test_the_born_population_covers_both_framings_and_the_known_claims():
    """The birth inventory: main line and annex in ONE ledger, the two
    surviving CLAIMABLE arm-comparison claims plus the phase-8 finding
    and the annex's coherent set, the voids, and the withdrawn family."""
    # [2026-08-16] Scoped to the BORN population: this test is the birth
    # inventory, and appends (a third void arrived the same day) must not
    # retro-fail it. Appended entries get their own pins below.
    born = results_ledger.ENTRIES[:BORN_ENTRIES]
    by_id = {entry["id"]: entry for entry in born}
    claimable = {
        entry["id"] for entry in born if entry["status"] == "CLAIMABLE"
    }
    assert claimable == {
        "p7-q1-swin-g2", "p7d-grid-density", "p8-parameter-dependence",
        "roadb-resolution-fall",
    }
    assert {e["framing"] for e in born} == {"main", "additional"}
    voids = {e["id"] for e in born if e["status"] == "VOID"}
    assert voids == {"void-first-ladder", "void-scut-animation-first-launch"}

    # Spot-pins on figures the write-up will quote.
    assert by_id["p7-q1-swin-g2"]["condition_2"]["margin"] == 3.83
    assert "0.0065" in by_id["p7-q1-swin-g2"]["caveats"][0]
    grid = by_id["p7d-grid-density"]
    assert grid["condition_2"]["patch8_vs_b16"]["margin"] == 6.42
    assert grid["condition_2"]["b32_512_vs_patch16_512"]["margin"] == 5.32
    resolution = by_id["roadb-resolution-fall"]
    assert resolution["framing"] == "additional"
    assert resolution["condition_2"]["masked_224_512"]["margin"] == 8.57
    assert any("confounded" in caveat for caveat in resolution["caveats"])
    assert by_id["p8-parameter-dependence"]["condition_1"]["interval_95"] == (
        -0.2047, -0.0020,
    )
    assert by_id["p7-best-arm-vs-challenger"]["status"] == "WITHDRAWN"
    assert by_id["p7-best-arm-vs-challenger"]["condition_2"]["margin"] == 1.05


def test_a_forward_reaching_correction_is_refused():
    bad = results_ledger.ENTRIES + ({
        "id": "correction-x", "claim": "x", "status": "WITHDRAWN",
        "framing": "main", "phase": "p9", "record": "x",
        "condition_1": None, "condition_2": None, "run_dirs": (),
        "date": "2026-08-16", "caveats": (), "corrects": "not-an-entry",
    },)
    with pytest.raises(ValueError, match="never reach"):
        results_ledger.validate(bad)


# --------------------------------------------------------------------------
# the prototype registration and its machinery
# --------------------------------------------------------------------------


def test_prototypes_registration_carries_every_resolved_choice():
    spec = phase9.PROTOTYPES_REGISTERED
    assert spec["registered"].startswith("2026-08-16")
    assert spec["space"]["set_name"] == "vit_b16__imagenet__g1"
    assert spec["space"]["distance"] == "euclidean"
    assert "lower patient id" in spec["space"]["tie_rule"]
    prior = spec["measured_prior"]
    assert (prior["tsne_companion"], prior["majority"]) == (0.409, 0.502)
    assert "corroborate" in prior["committed_both_ways"]
    assert spec["primary"] == {"column": "class3", "n_groups": 3}
    assert spec["secondary"]["column"] == "median"
    assert spec["secondary"]["status"].startswith("DESCRIPTIVE")
    assert spec["loo"]["ceiling"].startswith("(n-1)/n")
    assert spec["bootstrap"] == {
        "statistic": "identity persistence frequency",
        "n_boot": 2000, "seed": 1337,
    }
    assert spec["tier"].startswith("CLUSTER-ONLY")


def test_medoid_index_finds_the_centre_and_breaks_ties_toward_lower_id():
    rng = np.random.default_rng(9)
    cloud = rng.normal(0, 1.0, size=(30, 4))
    cloud[7] = cloud.mean(axis=0)  # planted centre
    assert phase9.medoid_index(cloud) == 7

    # An equilateral pair-tie: two points equidistant, ids decide.
    features = np.array([[0.0], [2.0]])
    assert phase9.medoid_index(features, member_ids=[50, 10]) == 1
    assert phase9.medoid_index(features, member_ids=[10, 50]) == 0

    # Bootstrap weights move the medoid toward the heavy member.
    line = np.array([[0.0], [1.0], [10.0]])
    assert phase9.medoid_index(line, weights=[1, 1, 1]) == 1
    assert phase9.medoid_index(line, weights=[0, 0, 5]) == 2


def test_loo_stability_hits_its_ceiling_on_a_clean_cluster():
    """A tight cluster with one planted centre: every removal except the
    medoid's own keeps the identity, so stability == ceiling."""
    rng = np.random.default_rng(10)
    features = rng.normal(0, 0.01, size=(12, 3))
    features[4] = features.mean(axis=0)
    ids = np.arange(100, 112)
    report = phase9.loo_identity_stability(features, ids)
    assert report["medoid_id"] == 104
    assert report["ceiling"] == 11 / 12
    assert report["stability"] == report["ceiling"]


def test_nearest_medoid_accuracy_separates_clean_classes():
    rng = np.random.default_rng(11)
    a = rng.normal(0, 0.1, size=(20, 3)) + [10, 0, 0]
    b = rng.normal(0, 0.1, size=(30, 3)) - [10, 0, 0]
    features = np.vstack([a, b])
    classes = np.array([1] * 20 + [2] * 30)
    ids = np.arange(len(classes))
    report = phase9.nearest_medoid_accuracy(features, classes, ids)
    assert report["accuracy"] == 1.0
    assert report["majority"] == 0.6
    assert report["chance"] == 0.5


def test_bootstrap_persistence_is_seeded_and_high_on_a_clean_cluster():
    rng = np.random.default_rng(12)
    features = rng.normal(0, 0.01, size=(15, 3))
    features[3] = features.mean(axis=0)
    ids = np.arange(15)
    one = phase9.bootstrap_persistence(features, ids, n_boot=200, seed=1337)
    two = phase9.bootstrap_persistence(features, ids, n_boot=200, seed=1337)
    assert one == two
    assert one["medoid_id"] == 3
    assert one["persistence"] > 0.5


# --------------------------------------------------------------------------
# the external reference (blocked) and the exit criteria
# --------------------------------------------------------------------------


def test_the_deall_reference_is_registered_blocked_with_the_trap_named():
    record = phase9.DEALL_REFERENCE_REGISTERED
    assert record["set"]["identities"] == 25
    assert "Thumbs.db" in record["set"]["excluded"]
    assert record["set"]["views"] == ("composite", "-lips", "-nose")
    assert "Table 5" in record["reading_registered"]
    assert "never pooled" in record["reading_registered"]
    assert "251-row COHORT sheet" in record["trap"]
    assert "no refitting" in record["task_shape"]
    assert len(record["blocked_on"]) == 2
    assert record["status"].startswith("REGISTERED; task and config not built")


def test_exit_criteria_are_registered_and_convergence_is_not_one():
    criteria = phase9.PHASE_9_EXIT_CRITERIA
    assert criteria["registered"] == "2026-08-16"
    assert set(criteria["criteria"]) == {
        "1_choices_first", "2_prototypes", "3_external_reference",
        "4_ledger", "5_suite",
    }
    assert "blocking issue recorded" in criteria["criteria"]["3_external_reference"]
    assert "convergence" in criteria["explicitly_not_a_criterion"]


def test_the_prototypes_config_is_fully_declared_and_points_at_the_space():
    import yaml

    config = yaml.safe_load(
        (REPO / "configs" / "p9_prototypes.yaml").read_text(encoding="utf-8")
    )
    assert config["task"]["kind"] == "prototypes"
    assert config["phase"] == "p9"
    assert config["seed"] == phase9.PROTOTYPES_REGISTERED["bootstrap"]["seed"]
    by_name = {e["name"]: e for e in config["inputs"]}
    assert by_name["embeddings"]["path"].endswith(
        phase9.PROTOTYPES_REGISTERED["space"]["set_name"]
    )
    assert all(
        set(e["rollup_sha256"]) != {"0"} for e in config["inputs"]
    ), "the prototypes config must carry only verified hashes"


# --------------------------------------------------------------------------
# 2026-08-16, round two: the prototypes verdict and the unblocked 25-set
# --------------------------------------------------------------------------

#: The ledger after the p9-prototypes append. The BORN pin above stays
#: forever; this pin covers entries 1..18. Append -> add a new pin;
#: never touch existing ones.
APPENDED_ENTRIES_18 = 18
CHECKSUM_18 = "b476ca7f6f99b2c0c10b8703d96e5ce39aa2da4dc0eb9ff86ec06acc51b18f81"


def test_the_ledger_append_extends_the_pin_chain():
    assert len(results_ledger.ENTRIES) >= APPENDED_ENTRIES_18
    assert results_ledger.cumulative_checksum(APPENDED_ENTRIES_18) == CHECKSUM_18
    entry = results_ledger.ENTRIES[17]
    assert entry["id"] == "p9-prototypes-descriptive"
    assert entry["status"] == "DESCRIPTIVE"
    assert "below chance" in entry["claim"]
    assert any("never be quoted" in caveat for caveat in entry["caveats"])


def test_prototypes_observed_fired_the_committed_reading():
    observed = phase9.PROTOTYPES_OBSERVED
    assert observed["observed"] == "2026-08-16"
    figures = observed["figures"]
    assert figures["class3_0"] == {
        "medoid": 183, "n": 88, "loo": 0.534, "ceiling": 0.989,
        "persistence": 0.394,
    }
    assert figures["class3_1"]["loo"] == figures["class3_1"]["ceiling"] == 0.992
    assert figures["class3_2"]["loo"] == figures["class3_2"]["ceiling"] == 0.967
    assert observed["companion"] == {
        "accuracy": 0.257, "majority": 0.502, "chance": 0.333,
    }
    assert "below chance" in observed["reading_fired"]
    assert "NEVER be quoted as" in observed["artifact_rule"]
    assert "representativeness claims are dead" in observed["verdict"]


def test_deall_reads_answer_both_blocks_and_extend_the_trap():
    reads = phase9.DEALL_REFERENCE_READS
    assert reads["score"]["expected_spread"] == (3, 7, 6, 6, 3)
    assert "27-surgeon" in reads["score"]["cannot_assert_from_disk"]
    assert "no role in the task" in reads["para_columns"]
    assert "APScores.txt" in reads["trap_extended"]
    assert "0.86-1.17" in reads["composites"]
    assert "PLAIN staged square" in reads["staging_answer"]
    assert "one nose above one mouth" in reads["eyeball_criterion"]
    assert "ONE declared input covers folder+CSV" in reads["declaration"]
    assert phase9.DEALL_REFERENCE_REGISTERED["unblocked"].startswith(
        "2026-08-16"
    )


def test_deall_partition_and_labels_reader_enforce_the_registration(tmp_path):
    names = (
        [f"F{i:03d}.jpg" for i in range(25)]
        + [f"F{i:03d}-lips.jpg" for i in range(25)]
        + [f"F{i:03d}-nose.jpg" for i in range(25)]
        + ["Thumbs.db", "cl_images_test_details.csv", "APScores.txt"]
    )
    views = phase9.deall_partition_stems(names)
    assert len(views["composite"]) == 25
    assert len(views["lips"]) == len(views["nose"]) == 25
    assert not any("Thumbs" in n or n.endswith((".csv", ".txt"))
                   for group in views.values() for n in group)

    # The reader: filename pinned, 25 rows, the 3/7/6/6/3 spread asserted.
    grades = [1] * 3 + [2] * 7 + [3] * 6 + [4] * 6 + [5] * 3
    good = tmp_path / "cl_images_test_details.csv"
    good.write_text(
        "PhotoID,Score,Para1,Para2,Para3,Para4\n" + "".join(
            f"F{i:03d},{grade},0,180,-180,200\n"
            for i, grade in enumerate(grades)
        ), encoding="utf-8",
    )
    labels = phase9.deall_read_labels(good)
    assert len(labels) == 25 and labels["F000"] == 1

    with pytest.raises(ValueError, match="cohort sheet"):
        phase9.deall_read_labels(tmp_path / "cohort_per_image_labels.csv")

    bad_spread = tmp_path / "sub" / "cl_images_test_details.csv"
    bad_spread.parent.mkdir()
    bad_spread.write_text(
        "PhotoID,Score\n" + "".join(
            f"F{i:03d},1\n" for i in range(25)
        ), encoding="utf-8",
    )
    with pytest.raises(ValueError, match="spread"):
        phase9.deall_read_labels(bad_spread)


def test_the_deall_config_copies_the_refit_block_and_pends_the_set():
    import yaml

    config = yaml.safe_load(
        (REPO / "configs" / "p9_deall_reference.yaml").read_text(encoding="utf-8")
    )
    arm = yaml.safe_load(
        (REPO / "configs" / "p7_d1_vit_b16_imagenet_g1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["task"]["kind"] == "deall_reference"
    for key in ("geometry", "label", "backbone", "seeds", "inner_val_frac",
                "max_epochs", "patience", "monitor", "learning_rate",
                "weight_decay", "batch_size"):
        assert config["task"][key] == arm["task"][key], key
    by_name = {e["name"]: e for e in config["inputs"]}
    for name in ("manifest_v1", "staged_v1", "embeddings"):
        assert set(by_name[name]["rollup_sha256"]) != {"0"}, name
    # [2026-08-16] The path is FILLED from the listing; the
    # hash stays a documented placeholder until the declare pass, then the
    # generator carries it.
    deall = by_name["deall_set"]
    # [UPDATED 2026-09-05] Declared by role since the cohort paths were
    # parameterised; the assertion is now that the config declares the
    # anchor set through the sanctioned variable, which is stricter than
    # the literal it replaces (a typo in the name fails here).
    assert deall["path"] == "${CLEFT_ANCHOR_SET}"
    from cleft.config.schema import COHORT_ENV_REFERENCES
    assert "CLEFT_ANCHOR_SET" in COHORT_ENV_REFERENCES
    assert deall["path"] == phase9.DEALL_REFERENCE_READS["folder"].split(" -- ")[0]


# --------------------------------------------------------------------------
# 2026-08-16, round three: the void first launch and the layout correction
# --------------------------------------------------------------------------

#: Entries 1..19 after the void-deall-first-launch append. Prior pins
#: stay untouched, as always.
APPENDED_ENTRIES_19 = 19
CHECKSUM_19 = "110277a0ce12c2d1ace845d7f40bee066a4b614968ace23d3297e5af60fce7a8"


def test_the_void_first_launch_is_ledgered_and_the_pins_chain():
    assert len(results_ledger.ENTRIES) >= APPENDED_ENTRIES_19
    assert results_ledger.cumulative_checksum(APPENDED_ENTRIES_19) == CHECKSUM_19
    entry = results_ledger.ENTRIES[18]
    assert entry["id"] == "void-deall-first-launch"
    assert entry["status"] == "VOID"
    assert "p9_deall_reference__d67248c9__p9-deall-reference-2" in (
        entry["run_dirs"]
    )
    assert any("75-vs-76" in caveat for caveat in entry["caveats"])
    assert any("RETRY_LIMIT" in caveat for caveat in entry["caveats"])


def test_the_layout_correction_and_the_flagged_suggestion_are_dated():
    from cleft import phase8

    reads = phase9.DEALL_REFERENCE_READS
    assert reads["corrected"].startswith("2026-08-16")
    assert "the shared material layout" in reads["corrected"]
    assert "76 files" in reads["corrected"]

    flag = phase9.PRE_CONTEXT_REFUSALS_LEAVE_NO_TRACE
    assert flag["flagged"] == "2026-08-16"
    assert "before RunContext" in flag["observed"]
    assert flag["status"].startswith("TO CONSIDER")

    retry = phase8.RETRY_LIMIT_IS_NOT_HOLDING
    assert retry["updated"].startswith("2026-08-16")
    assert "FileNotFoundError" in retry["updated"]
    assert retry["status"].startswith("OPEN")


# --------------------------------------------------------------------------
# 2026-08-16, round four: the prototype classifier (the reading given at supervision)
# --------------------------------------------------------------------------


def test_the_medoid_work_is_relabelled_and_the_25_set_has_two_roles():
    """Correction 1: 'prototypes' was never the plan sentence's medoids --
    both medoid records carry the dated relabel and everything measured
    stands under its honest name. Correction 2: the 25-set is external
    reference (DONE, PCC 0.2512, sheet PASSED) and anchor set."""
    assert phase9.PROTOTYPES_REGISTERED["relabelled"].startswith("2026-08-16")
    assert "COHORT-MEDOID" in phase9.PROTOTYPES_REGISTERED["relabelled"]
    assert phase9.PROTOTYPES_OBSERVED["relabelled"].startswith("2026-08-16")
    # The measured verdict itself is untouched by the relabel.
    assert "medoids stand as descriptive artifacts" in (
        phase9.PROTOTYPES_OBSERVED["verdict"]
    )
    roles = phase9.DEALL_REFERENCE_REGISTERED["roles"]
    assert "0.2512" in roles["external_reference"]
    assert "PASSED" in roles["external_reference"]
    assert "p9-deall-reference-3" in roles["external_reference"]
    assert "PROTOTYPE_CLASSIFIER_REGISTERED" in roles["anchor_set"]
    # The ledger's medoid entry needed no correcting entry: its claim
    # sentence already says "medoids", not "prototypes".
    entry = results_ledger.ENTRIES[17]
    assert "medoids" in entry["claim"] and "prototype" not in entry["claim"]


def test_the_classifier_registration_carries_cells_priors_and_corrections():
    spec = phase9.PROTOTYPE_CLASSIFIER_REGISTERED
    assert spec["registered"].startswith("2026-08-16")
    assert "no training anywhere" in spec["intent"]
    assert "not verified-unanimous" in spec["anchor_grade_caveat"]
    assert spec["cells"]["k"] == (1, 3)
    assert spec["cells"]["votes"] == ("plain", "weighted")
    # The admissibility arithmetic as CORRECTED at registration.
    assert "k>=6" in spec["cells"]["admissibility"]
    assert "largest k with slack" in spec["cells"]["admissibility"]
    assert spec["metrics"]["primary"].startswith("euclidean")
    assert spec["metrics"]["secondary"].startswith("cosine")
    prediction = spec["prediction_committed_both_ways"]
    assert "0.409" in prediction["priors"] and "769" in prediction["priors"]
    assert "third convergent" in prediction["if_near_or_below_chance"]
    assert "matter more" in prediction["if_it_beats_the_priors"]
    assert "re-extracts the 25 live" in spec["inputs_correction"]
    assert spec["evaluation"]["n"] == 237


def test_grade_to_class3_uses_the_fixed_thresholds():
    assert [phase9.grade_to_class3(g) for g in (1, 2, 3, 4, 5)] == [
        0, 0, 1, 2, 2,
    ]


def test_anchor_knn_grades_votes_weights_and_ties_as_committed():
    import numpy as np

    # k=1: exact nearest anchor's grade.
    anchors = np.array([[0.0], [10.0], [20.0]])
    grades = np.array([1, 3, 5])
    out = phase9.anchor_knn_grades(
        anchors, grades, np.array([[9.0], [19.0]]),
        k=1, weighted=False, metric="euclidean",
    )
    assert out.tolist() == [3, 5]

    # Plain k=3 majority vs distance-weighted: the same patient flips.
    anchors = np.array([[0.1], [10.0], [10.2]])
    grades = np.array([5, 2, 2])
    patient = np.array([[0.0]])
    plain = phase9.anchor_knn_grades(
        anchors, grades, patient, k=3, weighted=False, metric="euclidean"
    )
    weighted = phase9.anchor_knn_grades(
        anchors, grades, patient, k=3, weighted=True, metric="euclidean"
    )
    assert plain.tolist() == [2] and weighted.tolist() == [5]

    # A three-way plain tie breaks to the nearest neighbour's grade.
    anchors = np.array([[1.0], [2.0], [3.0]])
    grades = np.array([4, 1, 3])
    tied = phase9.anchor_knn_grades(
        anchors, grades, np.array([[0.0]]), k=3, weighted=False,
        metric="euclidean",
    )
    assert tied.tolist() == [4]

    # A zero-distance duplicate decides outright in the weighted cells.
    duplicate = phase9.anchor_knn_grades(
        np.array([[5.0], [5.5]]), np.array([2, 4]), np.array([[5.0]]),
        k=2, weighted=True, metric="euclidean",
    )
    assert duplicate.tolist() == [2]

    # Cosine refuses a zero-norm vector rather than inventing a direction.
    with pytest.raises(ValueError, match="zero-norm"):
        phase9.anchor_knn_grades(
            np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([1, 2]),
            np.array([[1.0, 1.0]]), k=1, weighted=False, metric="cosine",
        )


def test_anchor_self_consistency_measures_what_prototypes_presuppose():
    import numpy as np

    rng = np.random.default_rng(13)
    clean = np.vstack([
        rng.normal(0, 0.01, size=(3, 2)) + [10, 0],
        rng.normal(0, 0.01, size=(3, 2)) - [10, 0],
    ])
    grades = np.array([1, 1, 1, 5, 5, 5])
    report = phase9.anchor_self_consistency(clean, grades, metric="euclidean")
    assert report == {"fraction": 1.0, "agreeing": 6, "n": 6}

    interleaved = np.array([[0.0], [1.0], [2.0], [3.0]])
    report = phase9.anchor_self_consistency(
        interleaved, np.array([1, 5, 1, 5]), metric="euclidean"
    )
    assert report["fraction"] == 0.0


def test_the_classifier_config_is_fully_declared_and_copies_the_arm():
    import yaml

    config = yaml.safe_load((
        REPO / "configs" / "p9_prototype_classifier.yaml"
    ).read_text(encoding="utf-8"))
    arm = yaml.safe_load((
        REPO / "configs" / "p7_d1_vit_b16_imagenet_g1.yaml"
    ).read_text(encoding="utf-8"))
    assert config["task"]["kind"] == "prototype_classifier"
    for key in ("geometry", "label", "backbone", "batch_size"):
        assert config["task"][key] == arm["task"][key], key
    by_name = {e["name"]: e for e in config["inputs"]}
    assert set(by_name) == {"manifest_v1", "staged_v1", "embeddings", "deall_set"}
    assert all(set(e["rollup_sha256"]) != {"0"} for e in config["inputs"])
    # [UPDATED 2026-09-05] Declared by role. Stricter than the literal
    # suffix it replaces: an endswith() would still pass on a path that
    # had the cohort tree in front of it.
    assert by_name["deall_set"]["path"] == "${CLEFT_ANCHOR_SET}"


# --------------------------------------------------------------------------
# 2026-08-16, round five: the classifier result and the criteria walk
# --------------------------------------------------------------------------

APPENDED_ENTRIES_20 = 20
CHECKSUM_20 = "e70d2fd6288604e45a2ce508a791322fbec895b32a9fa425b0f0cbd845a3e4ca"


def test_the_classifier_result_is_recorded_precisely_and_ledgered():
    """The registered near-chance reading fired -- with the precision the
    record insists on: two cosine cells sit marginally ABOVE chance, the
    k=3 vote coincidence is observed-not-dictated, and the anchors'
    self-consistency arithmetic sits beside the observed figures."""
    observed = phase9.PROTOTYPE_CLASSIFIER_OBSERVED
    assert observed["run"] == (
        "p9_prototype_classifier__18d0d9fb__p9-prototype-classifier"
    )
    assert observed["attempts"] == 1
    assert observed["live_path_parity_max_abs"] == 1.53e-05
    cells = observed["cells"]
    assert cells["euclidean"]["k1"] == {"pcc": 0.1823, "acc3": 0.3376}
    assert cells["euclidean"]["k3"] == {"pcc": 0.1494, "acc3": 0.3333}
    assert cells["cosine"]["k1"] == {"pcc": 0.1531, "acc3": 0.3502}
    assert cells["cosine"]["k3"] == {"pcc": 0.1573, "acc3": 0.3629}
    assert "marginally ABOVE chance" in cells["stated_precisely"]
    assert "not an arithmetic identity" in cells["vote_coincidences"]
    consistency = observed["anchor_self_consistency"]
    assert (consistency["euclidean"], consistency["cosine"]) == ("4/25", "3/25")
    assert "114/600" in consistency["chance_expectation"]
    assert "premise fails" in consistency["reading"]
    assert "FOURTH convergent" in observed["convergence"]
    assert "the supervision material's own suggested method" in observed["convergence"]

    assert results_ledger.cumulative_checksum(APPENDED_ENTRIES_20) == CHECKSUM_20
    entry = results_ledger.ENTRIES[19]
    assert entry["id"] == "p9-anchor-classifier-convergence"
    assert entry["status"] == "DESCRIPTIVE"
    assert "FOURTH convergent" in entry["claim"]
    assert any("not verified-unanimous" in c for c in entry["caveats"])


def test_criterion_2_carries_the_dated_amendment():
    criteria = phase9.PHASE_9_EXIT_CRITERIA
    assert criteria["criterion_2_amended"].startswith("2026-08-16")
    assert "the supervision anchor reading" in criteria["criterion_2_amended"]
    assert "descriptive under its corrected label" in (
        criteria["criterion_2_amended"]
    )


def test_phase_9_closes_citing_only_what_is_recorded():
    """2026-08-16: the closing record cites, never claims. Five criteria
    MET (criterion 2 under its dated amendment), the four decided-and-
    measured items, and the four open items carried forward by name."""
    closing = phase9.PHASE_9_CLOSING
    assert closing["closed"].startswith("2026-08-16")
    assert set(closing["criteria"]) == {
        "1_choices_first", "2_prototypes", "3_external_reference",
        "4_ledger", "5_suite",
    }
    assert all(v.startswith("MET") for v in closing["criteria"].values())
    assert "dated amendment" in closing["criteria"]["2_prototypes"]
    assert "0.2512" in closing["criteria"]["3_external_reference"]
    assert "17/18/19/20" in closing["criteria"]["4_ledger"]
    decided = closing["decided_and_measured"]
    assert decided["framing"] == "ROAD_B_IS_THE_ANNEX"
    assert "FOURTH convergent" in decided["anchor_classifier"]
    assert closing["carried_forward_open"] == (
        "the unanimity confirmation on the anchor grades",
        "phase8.RETRY_LIMIT_IS_NOT_HOLDING",
        "phase9.PRE_CONTEXT_REFUSALS_LEAVE_NO_TRACE",
        "the -lips/-nose views, present and never staged",
    )
    assert "adds nothing" in closing["no_new_claims"]


APPENDED_ENTRIES_21 = 21
CHECKSUM_21 = "976d6f39add356ce99187c5f9c62e0035c0b2470b082674ae54aadfc9069b585"


def test_the_cleftgnn_void_is_ledgered_and_the_pins_chain():
    """2026-08-17: the fourth void entry. Prior pins untouched."""
    assert len(results_ledger.ENTRIES) >= APPENDED_ENTRIES_21
    assert results_ledger.cumulative_checksum(APPENDED_ENTRIES_21) == CHECKSUM_21
    entry = results_ledger.ENTRIES[20]
    assert entry["id"] == "void-cleftgnn-first-launch"
    assert entry["status"] == "VOID"
    assert entry["phase"] == "p10"
    assert "Median" in entry["claim"]
    assert any("R10 failure" in caveat for caveat in entry["caveats"])


APPENDED_ENTRIES_22 = 22
CHECKSUM_22 = "73bbbc5b19bcdc499af6b300b2a2844d7ab0bb1091a33176eec54e7833c05054"


def test_the_second_cleftgnn_void_is_ledgered_and_the_pins_chain():
    """2026-08-17: the fifth void. Prior pins untouched."""
    assert results_ledger.cumulative_checksum(APPENDED_ENTRIES_22) == CHECKSUM_22
    entry = results_ledger.ENTRIES[21]
    assert entry["id"] == "void-cleftgnn-second-launch"
    assert entry["status"] == "VOID"
    assert "refuted" in entry["claim"]
    assert "21.5" in entry["claim"]


APPENDED_ENTRIES_24 = 24
CHECKSUM_24 = "12a72c16a18b673f9317989035b899361208a28d37fce5d49f77fb4b6f911fc5"


def test_the_screen_and_third_void_are_ledgered_and_the_pins_chain():
    """2026-08-17: the rater screen (descriptive) and the third void."""
    assert results_ledger.cumulative_checksum(APPENDED_ENTRIES_24) == CHECKSUM_24
    screen = results_ledger.ENTRIES[22]
    assert screen["id"] == "p10-rater-screen-mixed"
    assert screen["status"] == "DESCRIPTIVE"
    assert "0.2703" in screen["claim"] and "0.2520" in screen["claim"]
    assert any("1.24 sd" in c for c in screen["caveats"])
    void = results_ledger.ENTRIES[23]
    assert void["id"] == "void-cleftgnn-third-launch"
    assert void["status"] == "VOID"
    assert "REFUTED" in void["claim"]
    assert any("our own bug" in c for c in void["caveats"])


APPENDED_ENTRIES_25 = 25
CHECKSUM_25 = "1b1f3167715f724cf802bc88bd23497b6c8832b9d502000196d1e6354aa83f38"


def test_the_sabm_head_is_banked_under_its_own_name_and_cleftgnn_is_not():
    """2026-08-17: what was measured gets a row; what was not does not."""
    assert results_ledger.cumulative_checksum(APPENDED_ENTRIES_25) == CHECKSUM_25
    entry = results_ledger.ENTRIES[24]
    assert entry["id"] == "p10-resnet50-sabm-head"
    assert entry["status"] == "DESCRIPTIVE"
    assert "FROZEN ImageNet ResNet-50" in entry["claim"]
    assert "SABM attention head" in entry["claim"]
    assert "0.0241" in entry["claim"] and "0.0676" in entry["claim"]
    # The 0.0068 fact is attached to the claim, not buried.
    assert "0.0068" in entry["caveats"][0]
    assert "NOT a CleftGNN measurement" in entry["caveats"][0]
    assert any("criterion (i) is UNREPORTED" in c for c in entry["caveats"])
    # No entry claims CleftGNN itself: the row is left unclaimed.
    named = [
        e for e in results_ledger.ENTRIES
        if "cleftgnn" in e["id"] and e["status"] != "VOID"
    ]
    assert not named, f"CleftGNN must stay unclaimed; found {named}"


APPENDED_ENTRIES_32 = 32
CHECKSUM_32 = "874cd1aac58ebd39d2d86df252c4f1e0ee3f7a16d3ffa08a2e7d9236d552b4ad"


def test_the_anchor_loop_is_ledgered_unresolved_and_the_pins_chain():
    """2026-08-29: Phase 16's primary contrast, appended -- and the
    ledger's FIRST condition-2-passes/condition-1-fails case."""
    assert results_ledger.cumulative_checksum(APPENDED_ENTRIES_32) == CHECKSUM_32
    entry = results_ledger.ENTRIES[31]
    assert entry["id"] == "p16-anchor-loop-unresolved"
    assert entry["status"] == "UNRESOLVED-WITHDRAWN"
    assert "0.2040" in entry["claim"] and "0.2520" in entry["claim"]
    assert "-0.0481" in entry["claim"]
    assert "cannot resolve" in entry["claim"]
    assert entry["condition_1"].startswith("FALSE")
    assert entry["condition_2"].startswith("TRUE")
    assert "1.46x" in entry["condition_2"]
    # The flag: the first such case, and why both conditions exist.
    # [2026-08-31] This precedence claim is WRONG AS WRITTEN -- p16 is
    # the FIFTH condition-2-pass/condition-1-fail row, corrected by
    # ENTRIES[37] 'ledger-condition-split-count-corrected'. The
    # assertion STAYS: the row is append-only and its wording is
    # preserved exactly, so the pin still describes the row truthfully
    # as a pin. The correction is asserted separately, below.
    assert any(
        "first case in this ledger of condition 2 passing" in c
        for c in entry["caveats"]
    )
    assert any(
        "either alone\nwould have called this resolved" in c
        or "either alone " in c
        for c in entry["caveats"]
    )
    # the framing: parity is the criterion's verdict, the
    # negative point estimate not smoothed away.
    assert any("CRITERION'S verdict" in c for c in entry["caveats"])
    assert any("not smoothed away" in c for c in entry["caveats"])
    # The pre-committed null reading attaches: the fifth convergence.
    assert any("FIFTH" in c for c in entry["caveats"])
    results_ledger.validate()


APPENDED_ENTRIES_37 = 37
CHECKSUM_37 = "340f2c3fdcb905241b9002206bc521bd6b162872d7e214aced06e33e0084b55c"


def test_the_five_tstr_contrasts_are_ledgered_and_the_pins_chain():
    """2026-08-30: Phase 17's five-contrast family, appended in
    registration order -- one claimable negative, four unresolved."""
    assert results_ledger.cumulative_checksum(APPENDED_ENTRIES_37) == CHECKSUM_37
    rows = {e["id"]: e for e in results_ledger.ENTRIES[32:37]}
    assert list(rows) == [
        "p17-a-vs-probe", "p17-b-vs-probe", "p17-c-vs-probe",
        "p17-a-vs-c", "p17-b-vs-c",
    ]
    verdicts = {i: r["status"] for i, r in rows.items()}
    assert verdicts["p17-b-vs-probe"] == "CLAIMABLE"
    for other in ("p17-a-vs-probe", "p17-c-vs-probe", "p17-a-vs-c",
                  "p17-b-vs-c"):
        assert verdicts[other] == "UNRESOLVED-WITHDRAWN", other

    # The second condition-2/condition-1 instance, flagged on the row.
    # [2026-08-31] Also WRONG AS WRITTEN -- under the broad property
    # p17-a-vs-probe is the SIXTH, and under the narrower reading its
    # membership is unestablished (its condition-1 records no direction
    # pattern). Corrected by ENTRIES[37]. The assertion STAYS for the
    # same reason as p16's: the row is append-only and preserved.
    a_row = rows["p17-a-vs-probe"]
    assert a_row["condition_1"].startswith("FALSE")
    assert a_row["condition_2"].startswith("TRUE")
    assert any(
        "SECOND condition-2-pass/condition-1-fail" in c
        for c in a_row["caveats"]
    )
    assert "direction" not in a_row["condition_1"], (
        "the unestablished-membership finding rests on this absence"
    )
    assert any("p16-anchor-loop-unresolved" in c for c in a_row["caveats"])
    # The variance caveat: a tiny threshold is not a strengthened claim.
    assert any("not a strengthened claim" in c for c in a_row["caveats"])
    assert any("3.4x tighter" in c for c in a_row["caveats"])

    # B's committed reading, verbatim from the pre-run record.
    from cleft import phase17

    b_row = rows["p17-b-vs-probe"]
    assert "8.3x" in b_row["condition_2"]
    committed = " ".join(phase17.READINGS_COMMITTED["arm_b_if_null"].split())
    for phrase in ("0.31 on CARS", "more than symmetry",
                   "unattributable between them"):
        assert any(
            phrase in " ".join(c.split()) for c in b_row["caveats"]
        ), phrase
        assert phrase in committed.replace("**", ""), phrase

    # The A-C caution: attribution NOT claimed.
    assert any(
        "scheme attribution is NOT claimed" in c
        for c in rows["p17-a-vs-c"]["caveats"]
    )
    # B-C is the only both-conditions failure.
    bc = rows["p17-b-vs-c"]
    assert bc["condition_1"].startswith("FALSE")
    assert bc["condition_2"].startswith("FALSE")

    # Every threshold re-derives from the standing helper and the rows'
    # own quoted sds -- not retyped arithmetic.
    from cleft.train.phase3 import combined_claimable_delta as ccd

    assert round(ccd(0.0044, 5, 0.0148, 5)["arm_means_95"], 3) == 0.014
    assert round(ccd(0.0317, 5, 0.0148, 5)["arm_means_95"], 4) == 0.0307
    assert round(ccd(0.0586, 5, 0.0148, 5)["arm_means_95"], 3) == 0.053
    assert round(ccd(0.0044, 5, 0.0586, 5)["arm_means_95"], 4) == 0.0515
    assert round(ccd(0.0317, 5, 0.0586, 5)["arm_means_95"], 4) == 0.0584
    results_ledger.validate()


APPENDED_ENTRIES_38 = 38
CHECKSUM_38 = "b9c444a5e9fd79cab2adea451673c17d02682c0432b0c54710599d7902447187"


def test_the_ledgers_first_correction_appends_and_the_chain_holds():
    """2026-08-31: the ledger's FIRST use of ``corrects``, correcting a
    claim about its own history rather than a measurement.

    The precedence sentences on p16-anchor-loop-unresolved and
    p17-a-vs-probe were wrong as written; the rows are append-only and
    untouched, so the correction appends.
    """
    assert results_ledger.cumulative_checksum(APPENDED_ENTRIES_38) == CHECKSUM_38
    # [2026-09-06] Was `len(ENTRIES) == 38`, a global count standing
    # in for "the correction is entry 38 and the prefix is intact".
    # The prefix checksum above already says the second half, and
    # the position below says the first. The count went stale when
    # Phase 27 appended at index 38, which is the third time this
    # instrument has gone stale on an unrelated append.
    assert len(results_ledger.ENTRIES) >= APPENDED_ENTRIES_38
    results_ledger.validate()

    entry = results_ledger.ENTRIES[37]
    assert entry["id"] == "ledger-condition-split-count-corrected"
    assert entry["status"] == "DESCRIPTIVE"
    assert entry["date"] == "2026-08-31"

    # THE FIRST USE OF THE MECHANISM -- and it is the only one.
    assert entry["corrects"] == "p16-anchor-loop-unresolved"
    users = [e["id"] for e in results_ledger.ENTRIES if e.get("corrects")]
    assert users == ["ledger-condition-split-count-corrected"]
    assert any("FIRST USE OF `corrects`" in c for c in entry["caveats"])

    # The corrected rows are UNTOUCHED -- the earlier pins prove it, and
    # their own sentences still stand as written.
    assert results_ledger.cumulative_checksum(37) == CHECKSUM_37
    p16 = results_ledger.ENTRIES[31]
    assert any(
        "first case in this ledger of condition 2 passing" in c
        for c in p16["caveats"]
    )

    # The count, and the two positions it corrects.
    claim = " ".join(entry["claim"].split())
    assert "EIGHT rows" in claim
    assert "25, 26, 27, 28, 31, 32, 34, 35" in claim
    assert "p16 (index 31) is the FIFTH, not the first" in claim
    assert "p17-a-vs-probe (index 32) is the SIXTH, not the second" in claim
    assert "No measurement changes" in claim


def test_the_correction_table_re_derives_from_the_ledgers_own_fields():
    """The eight-row table in the entry is not a transcription: it is
    what the ledger's condition fields produce, and this re-derives it."""
    entry = results_ledger.ENTRIES[37]
    table = next(
        c for c in entry["caveats"] if c.startswith("THE EIGHT ROWS")
    )

    derived = []
    for index, e in enumerate(results_ledger.ENTRIES):
        c1, c2 = e.get("condition_1"), e.get("condition_2")
        if not (isinstance(c1, str) and c1.strip().upper().startswith("FALSE")):
            continue
        if not (isinstance(c2, str) and c2.strip().upper().startswith("TRUE")):
            continue
        detail = c1.split("--", 1)[1].strip() if "--" in c1 else c1
        margin = c2.split("at ", 1)[1].split("x", 1)[0] if "at " in c2 else "?"
        derived.append(
            f"{index} | {e['id']} | {e['date']} | condition 1 {detail} | "
            f"condition 2 {margin}x"
        )

    assert len(derived) == 8
    flat = " ".join(table.split())
    for row in derived:
        assert " ".join(row.split()) in flat, row
    # And the indices the claim names are exactly the derived ones.
    assert [r.split(" | ")[0] for r in derived] == [
        "25", "26", "27", "28", "31", "32", "34", "35"
    ]


def test_the_correction_preserves_what_was_true_of_p16():
    """p16 really is the first of the narrower kind. That fact was
    measured and is kept, not discarded with the wrong sentence."""
    entry = results_ledger.ENTRIES[37]
    narrow = next(c for c in entry["caveats"] if "NARROWER kind" in c)
    assert "all\nper-seed intervals span zero".replace("\n", " ") in (
        " ".join(narrow.split())
    ) or "ALL per-seed intervals span zero" in " ".join(narrow.split())
    assert "directions are mixed" in " ".join(narrow.split())

    # The evidence: index 28 is the only earlier all-span-zero row, and
    # its direction is single. Re-derived from phase12's own per-seed
    # means rather than quoted.
    from cleft import phase12

    a = phase12.ARMS_A_D_OBSERVED["a_frontal_only_236"]["per_seed"]
    b = phase12.ARMS_B_C_OBSERVED["b_basal_only_236"]["per_seed"]
    diffs = [round(x - y, 4) for x, y in zip(a, b)]
    assert diffs == [0.0507, 0.0321, 0.0326, 0.1121, 0.0849]
    assert all(d > 0 for d in diffs), "one sign -- directions not mixed"
    assert phase12.PAIRED_OBSERVED["verdicts"]["b_vs_a"]["direction"] == (
        "A ahead"
    )
    for figure in ("+0.0507", "+0.0321", "+0.0326", "+0.1121", "+0.0849"):
        assert figure in narrow

    # p17 fails under the narrow reading too, and the reason is stated.
    p17 = next(c for c in entry["caveats"] if "UNESTABLISHED" in c)
    assert "no phase-17 record states its per-seed sign pattern" in (
        " ".join(p17.split())
    )


def test_the_correction_carries_the_ruling_and_the_lesson():
    entry = results_ledger.ENTRIES[37]
    ruling = next(c for c in entry["caveats"] if "THE RULING" in c)
    flat = " ".join(ruling.split())
    assert "DESCRIPTIVE reading" in flat
    assert "the bolded head sentence is the claim" in flat
    assert "restates the property with the qualifier DROPPED" in flat
    assert "load-bearing will mislead a reader" in flat

    lesson = next(c for c in entry["caveats"] if "THE MECHANISM" in c)
    flat = " ".join(lesson.split())
    assert "LEDGER'S OWN HISTORY" in flat
    assert "AT APPEND TIME" in flat
    assert "when the ledger itself could have been queried" in flat
    assert "writing a claim next to the data that refutes it" in flat


# --------------------------------------------------------------------------
# [2026-09-02] The external-validation concession
# --------------------------------------------------------------------------


def _flat24(text: str) -> str:
    return " ".join(text.split())


def test_the_external_validation_concession_cites_four_grounds():
    record = phase9.EXTERNAL_VALIDATION_ON_THE_25_CONCEDED
    assert record["tag"].startswith("[CONCEDED]")
    assert "no ledger row" in record["tag"]

    # It says how it arrived, which is the point of writing it here.
    arrival = _flat24(record["how_it_arrived"])
    assert "without knowing Phase 9 had already run it" in arrival
    assert "verification came second" in arrival


def test_ground_1_matches_the_prototype_classifier_cells():
    observed = phase9.PROTOTYPE_CLASSIFIER_OBSERVED
    cells = observed["cells"]
    assert cells["euclidean"]["k1"]["acc3"] == 0.3376
    assert cells["cosine"]["k3"]["acc3"] == 0.3629
    assert cells["baselines"] == {"chance": 0.3333, "majority": 0.502}
    assert observed["pcc_vs_trained_head"] == "0.15-0.18 everywhere, under 0.2520"

    ground = _flat24(
        phase9.EXTERNAL_VALIDATION_ON_THE_25_CONCEDED["1_already_measured"]
    )
    for token in ("0.3376", "0.3333", "0.3502", "0.3629", "0.502",
                  "0.15-0.18", "0.2520", "0.2512"):
        assert token in ground, token


def test_the_cross_set_gap_paraphrase_is_flagged_as_not_in_the_record():
    """The record's reading is WITHIN-set, and the difference matters."""
    banked = _flat24(
        phase9.PROTOTYPE_CLASSIFIER_OBSERVED["anchor_self_consistency"]["reading"]
    )
    assert banked == (
        "the method's premise fails among the trusted references before "
        "any cohort patient is scored"
    )
    # And the phrase attributed to the record is nowhere in phase9.
    body = pathlib.Path("src/cleft/phase9.py").read_text(encoding="utf-8")
    assert body.count("CROSS-SET GAP") == 1      # only the flag itself

    flag = _flat24(phase9.EXTERNAL_VALIDATION_ON_THE_25_CONCEDED[
        "1a_the_self_consistency_and_the_records_OWN_reading"
    ])
    assert "is NOT in the record" in flag
    assert "4/25" in flag and "3/25" in flag and "4.75/25" in flag
    assert "among the 25 THEMSELVES" in flag


def test_ground_3_is_what_the_guard_actually_does():
    """A consensus column has no rater matrix, and the code refuses."""
    import numpy as np
    import pytest

    from cleft.data import reliability as R

    one_column = np.array([[3], [2], [4], [5], [1]])
    for fn in (R.mean_inter_rater_r, R.fleiss_kappa, R.cronbach_alpha,
               R.mean_pairwise_qwk):
        with pytest.raises(R.ReliabilityError, match="at least 2 raters"):
            fn(one_column)

    # And a hypothetical unanimous five-column panel would be exactly 1.
    unanimous = np.repeat(one_column, 5, axis=1)
    assert R.mean_inter_rater_r(unanimous) == pytest.approx(1.0, abs=1e-12)
    assert R.fleiss_kappa(unanimous) == pytest.approx(1.0, abs=1e-12)

    ground = _flat24(phase9.EXTERNAL_VALIDATION_ON_THE_25_CONCEDED[
        "3_the_reliability_is_not_computable"
    ])
    assert "need at least 2 raters, got 1" in ground
    assert "nothing to average" in ground
    hazard = _flat24(phase9.EXTERNAL_VALIDATION_ON_THE_25_CONCEDED[
        "3a_and_assuming_1_0_would_be_the_hazard_not_the_fix"
    ])
    assert "those per-rater grades do not exist in our record" in hazard
    assert "0.9032" in hazard


def test_ground_4_quotes_the_two_places_the_record_says_the_opposite():
    ground = _flat24(phase9.EXTERNAL_VALIDATION_ON_THE_25_CONCEDED[
        "4_unanimity_is_REPORTED_not_banked"
    ])
    assert "no reconciliation forced" in ground
    assert "not verified-unanimous" in ground
    assert "supervision ask 7" in ground
    # Both quotations are the record's own words.
    assert "no reconciliation forced" in _flat24(
        phase9.DEALL_REFERENCE_READS["score"]["cannot_assert_from_disk"]
    )
    assert "not verified-unanimous" in _flat24(
        phase9.PROTOTYPE_CLASSIFIER_REGISTERED["anchor_grade_caveat"]
    )
    # And the two phrases are kept apart.
    apart = _flat24(phase9.EXTERNAL_VALIDATION_ON_THE_25_CONCEDED[
        "4a_and_the_two_phrases_are_not_the_same_claim"
    ])
    assert "SELECTION CRITERION over 76 images" in apart
    assert "not unanimity among 27 raters" in apart


def test_the_concession_withdraws_nothing_and_names_its_reopen_condition():
    record = phase9.EXTERNAL_VALIDATION_ON_THE_25_CONCEDED
    kept = _flat24(record["what_this_does_NOT_forbid"])
    assert "nothing about them is withdrawn" in kept
    assert "0.2512 stands as recorded" in kept
    assert "VALIDATION FRAMING" in kept

    reopen = _flat24(record["what_would_reopen_it"])
    assert "banked reliability figure for the 25's label" in reopen
    assert "does not reopen it" in reopen


def test_the_two_record_defects_the_verification_found():
    from cleft import results_ledger

    # (i) 0.2512 is not ledgered, though two places say it is.
    assert not any(
        "0.2512" in str(entry.get("claim", "")) for entry in results_ledger.ENTRIES
    )
    deall_rows = [
        e["id"] for e in results_ledger.ENTRIES if "deall" in e["id"]
    ]
    assert deall_rows == ["void-deall-first-launch"]
    flagged = _flat24(phase9.EXTERNAL_VALIDATION_ON_THE_25_CONCEDED[
        "found_the_0_2512_IS_NOT_LEDGERED"
    ])
    assert "NO LEDGER ROW CARRIES 0.2512" in flagged
    assert "Flagged, not corrected" in flagged

    # (ii) the Spearman is computed and never banked.
    body = pathlib.Path("src/cleft/run.py").read_text(encoding="utf-8")
    task = body.split("def task_deall_reference(")[1].split("\ndef ")[0]
    assert "metrics.spearman(truth, means)" in task
    assert "0.2512" in pathlib.Path(
        "src/cleft/phase9.py"
    ).read_text(encoding="utf-8")


def test_the_n25_interval_spans_zero_and_the_arithmetic_reproduces():
    import math

    import pytest

    from cleft.relevance import significance_threshold

    z, se = 1.959963985, 1.0 / math.sqrt(25 - 3)
    low = math.tanh(math.atanh(0.2512) - z * se)
    high = math.tanh(math.atanh(0.2512) + z * se)
    assert low == pytest.approx(-0.1598, abs=5e-4)
    assert high == pytest.approx(0.5880, abs=5e-4)
    assert low < 0 < high                       # it spans zero
    assert significance_threshold(25) == pytest.approx(0.4179, abs=5e-4)

    # The same arithmetic reproduces phase10's own banked n=25 interval.
    from cleft import phase10

    banked = phase10.CLEFTGNN_COMPARATOR_TABLES["the_598_cell"][
        "fisher_ci_95_recomputed"
    ]
    got = tuple(
        round(math.tanh(math.atanh(0.598) + s * z * se), 4) for s in (-1, 1)
    )
    assert got == banked == (0.2656, 0.8033)

    recorded = _flat24(phase9.EXTERNAL_VALIDATION_ON_THE_25_CONCEDED[
        "and_the_n_25_interval_nobody_stated"
    ])
    assert "SPANS ZERO" in recorded
    assert "0.4179" in recorded


# --------------------------------------------------------------------------
# the chain re-derivation, 2026-09-05
# --------------------------------------------------------------------------


def test_the_chain_re_derivation_is_recorded_with_its_consequence():
    """The one re-pinning this ledger has had, recorded in the same
    convention as every other correction: in place, dated, original
    preserved, reason stated -- and the consequence stated too."""
    record = results_ledger.CHAIN_RE_DERIVED
    assert record["date"] == "2026-09-05"

    changed = " ".join(_flat24(record["what_changed"]).split())
    assert "nine party mentions across six rows" in changed
    # The mechanisms the rewrites had to keep.
    assert "dated staging error" in changed
    assert "the method that was proposed at supervision" in changed
    assert "after Phase 1 measured it unsupported" in changed
    assert "a stipulated framing rather than a derived fact" in changed
    assert "its date and its two grounds" in changed

    unchanged = " ".join(_flat24(record["what_did_not_change"]).split())
    for field in ("claim", "status", "condition", "ordering", "threshold"):
        assert field in unchanged, field
    assert "MEASUREMENT, not an assurance" in unchanged

    consequence = " ".join(_flat24(record["the_honest_consequence"]).split())
    assert "nothing has been edited SINCE 2026-09-05" in consequence
    assert "not that nothing was ever edited" in consequence
    assert "strictly weaker" in consequence

    why = " ".join(_flat24(record["why_it_was_done"]).split())
    assert "the repository is being made public" in why
    assert "refutations are of PREMISES" in why

    narrow = " ".join(_flat24(record["what_would_not_have_justified_this"]).split())
    assert "Those append; they never rewrite" in narrow
    assert "defensible exactly once" in narrow


def test_the_re_derivation_record_names_no_party():
    """The record OF the anonymisation must not undo it."""
    blob = " ".join(str(value) for value in
                    results_ledger.CHAIN_RE_DERIVED.values())
    for name in ("Claude", "El Mahdi", "Liu"):
        assert name not in blob, name


def test_the_ledger_itself_names_no_party():
    """The whole point of the pass, asserted on the shipped rows rather
    than on the record that describes them. ``RanaPhotoID`` is a COLUMN
    NAME in the score sheet -- part of the data contract, matched
    against the workbook at load time -- and is exempt."""
    blob = " ".join(
        str(value)
        for entry in results_ledger.ENTRIES
        for value in entry.values()
    ).replace("RanaPhotoID", "<column>")
    for name in ("Claude", "El Mahdi", "Liu", "supervisor", "operator"):
        assert name not in blob, name


def test_the_born_prefix_survived_the_re_derivation():
    """The strongest thing the re-pinning did NOT cost: the first
    rewritten row sits at position 19, so every prefix through 18 --
    the born population included -- hashes as it did on 2026-08-16."""
    record = results_ledger.CHAIN_RE_DERIVED
    assert results_ledger.cumulative_checksum(BORN_ENTRIES) == BORN_CHECKSUM
    assert results_ledger.cumulative_checksum(18) == CHECKSUM_18
    note = " ".join(_flat24(record["the_born_prefix_is_untouched"]).split())
    assert "position 19" in note
    assert "c46fd950" in note and BORN_CHECKSUM.startswith("c46fd950")

    # And the superseded table starts exactly where the born prefix ends.
    superseded = record["the_superseded_checksums"]
    assert min(superseded) == 19
    # [2026-09-06] Was `max(superseded) == len(ENTRIES) == 38`. The
    # claim is about where the RE-DERIVATION ended, which is a fact
    # about that event and not about the ledger's current size. The
    # chained equality tied the two together and broke when Phase 27
    # appended.
    assert max(superseded) == 38
    assert len(results_ledger.ENTRIES) >= max(superseded)
    # Every re-derived length differs from the value it replaced, and
    # every one of them is a live pin somewhere in the suite.
    for n, old in superseded.items():
        assert results_ledger.cumulative_checksum(n) != old, n
        assert len(old) == 64, n


def test_every_superseded_checksum_is_gone_from_the_suite():
    """The originals are preserved in ONE place -- the record -- and
    nowhere else, so no test can still be asserting a stale chain."""
    superseded = set(
        results_ledger.CHAIN_RE_DERIVED["the_superseded_checksums"].values()
    )
    ledger_source = (
        pathlib.Path(results_ledger.__file__).read_text(encoding="utf-8")
    )
    for path in sorted((REPO / "tests").glob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        for old in superseded:
            assert old not in text, f"{path.name} still pins {old[:12]}"
    # ...and they ARE all still readable from the record itself.
    for old in superseded:
        assert old in ledger_source


def test_the_re_derivation_is_a_module_record_not_a_ledger_row():
    """An anonymisation is not a CLAIM, so it does not become an entry.

    [2026-09-06] This test carried TWO absolute-state instruments for
    one relation: `len(ENTRIES) == 38` and a pin on ENTRIES[-1]. Both
    stood in for "the re-derivation banked nothing", and both went
    stale when Phase 27 appended. Rewritten as the relation, which is
    true at 38, at 39 and at any size.
    """
    assert not [
        e for e in results_ledger.ENTRIES
        if "re-deriv" in e["id"] or "rederiv" in e["id"]
        or "chain" in e["id"]
    ]
    # The correction that WAS the last entry when this was written is
    # still there, still at its own index, and still the only user of
    # the corrects mechanism. That is what the tail pin was for.
    assert results_ledger.ENTRIES[37]["id"] == (
        "ledger-condition-split-count-corrected"
    )
    assert [
        e["id"] for e in results_ledger.ENTRIES if e.get("corrects")
    ] == ["ledger-condition-split-count-corrected"]
    assert not any(
        "anonym" in str(entry).lower() for entry in results_ledger.ENTRIES
    )


# --------------------------------------------------------------------------
# the census, as a guard rather than a report
# --------------------------------------------------------------------------

#: What the repository may still say. Each is a DATA CONTRACT or a
#: quoted source, and each is excluded for a stated reason -- not
#: because it was awkward to change.
CENSUS_EXEMPT = (
    # the score sheet's own column name, matched against the workbook
    # by name at load time
    "RanaPhotoID",
    # the score-sheet filename, hashed as a string in a declared
    # input -- the STEM, because the name wraps at a line break in
    # phase10 and reappears in the derived per-image labels file
    "1_Liu",  # COHORT_EXEMPT: the exemption list must name what it exempts
    # the cited papers' author lists -- removing an author's name from
    # their own citation is misattribution
    "Bera, Wharton, Liu, Bessis & Behera",
    # the manuscript filenames: renaming a document misnames the source
    # of seven verified quotes
    "Bruce-ready version - minus Jonathan's edits.docx",
    "Bruce-ready draft",
    # these guards themselves
    'for name in ("Claude", "El Mahdi"',
    "CENSUS_EXEMPT",
    "census_exempt",
)

#: Case-INSENSITIVE on purpose. The first anonymisation pass was
#: case-sensitive and 129 uppercase mentions survived it; a guard that
#: repeats the defect it is guarding against is not a guard.
CENSUS_PARTIES = (  # CENSUS_EXEMPT: the guard's own subject list
    "claude", "el mahdi",           # CENSUS_EXEMPT
    "el_mahdi", "liu", "liama",     # CENSUS_EXEMPT
)


#: **[ADDED 2026-09-06] The ONE file the anonymisation does not govern.**
#:
#: A citation exists to name its author. Publishing a dissertation
#: artifact that cannot be cited to a person defeats the point of
#: publishing it, so ``CITATION.cff`` carries the real name.
#:
#: **It is permitted BY PATH, not by adding a name to CENSUS_EXEMPT.**
#: An exempt string would permit that name everywhere in the tree, which
#: is the opposite of what is wanted: the anonymisation holds in all 647
#: other tracked files and this one is the stated exception.
CITATION_EXEMPT = ("CITATION.cff",)


def test_no_party_is_named_anywhere_outside_the_data_contract():
    """The whole anonymisation, asserted over the tree rather than
    measured once and reported.

    **[WIDENED 2026-09-06] The sweep missed the repository root and it
    missed ``.cff``.** ``CITATION.cff`` was therefore unswept twice over
    by accident, and the exception it needs would have rested on two
    oversights rather than on a decision. The root is now walked and
    ``.cff`` is read, so the permission below is the only thing letting
    that file through, and any OTHER root file or ``.cff`` is caught.
    """
    roots = ("src", "tests", "scripts", "docs", "configs")
    offenders = []
    candidates = [
        path for path in sorted(REPO.iterdir())
        if path.is_file()
    ]
    for root in roots:
        candidates += sorted((REPO / root).rglob("*"))
    for path in candidates:
        if True:
            if path.is_dir() or "__pycache__" in str(path):
                continue
            if path.suffix not in (".py", ".md", ".yaml", ".yml", ".txt",
                                   ".cff"):
                continue
            if path.name in CITATION_EXEMPT:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:  # pragma: no cover - none today
                continue
            for number, line in enumerate(text.split("\n"), start=1):
                if any(e in line for e in CENSUS_EXEMPT):
                    continue
                lowered = line.lower()
                for party in CENSUS_PARTIES:
                    if party in lowered:
                        offenders.append(
                            f"{path.relative_to(REPO)}:{number} [{party}]"
                        )
    assert not offenders, (
        f"{len(offenders)} party mention(s) outside the data contract:\n"
        + "\n".join(offenders[:40])
    )


def test_the_citation_exemption_is_one_file_and_it_is_used(repo_root=None):
    """[ADDED 2026-09-06] The exception is real, scoped, and exercised.

    If ``CITATION.cff`` ever stops naming the author, the exemption is
    dead weight and should go. If any OTHER file starts claiming it, the
    scope has widened without a decision."""
    assert CITATION_EXEMPT == ("CITATION.cff",)

    citation = REPO / "CITATION.cff"
    assert citation.is_file(), "the exemption names a file that is not there"
    text = citation.read_text(encoding="utf-8")

    # It really does name a party, so the exemption is doing work.
    lowered = text.lower()
    assert any(party in lowered for party in CENSUS_PARTIES), (
        "CITATION.cff names no party, so the exemption is unnecessary"
    )  # CENSUS_EXEMPT: this test is about the one permitted file
    # And it says why it is the exception, so a reader of the file knows.
    assert "THE ONE FILE WHERE THE ANONYMISATION DOES NOT APPLY" in text
    assert "PERMITS THIS FILE SPECIFICALLY rather than being widened" in text

    # The sweep really does reach the root and .cff now, which is what
    # makes the exemption a decision rather than an accident.
    import inspect

    source = inspect.getsource(
        test_no_party_is_named_anywhere_outside_the_data_contract
    )
    assert "REPO.iterdir()" in source, "the root is not swept"
    assert '".cff"' in source, ".cff is not swept"

    # **Nothing was added to the general exemption list to achieve
    # this**, which is the whole point of permitting by path. Two of the
    # list's pre-existing entries do contain party substrings and both
    # are documented there: a score-sheet filename stem, and a guard's
    # own source line. The discriminator is the FAMILY NAME, which
    # appears nowhere else in the tree.
    joined = " ".join(CENSUS_EXEMPT).lower()
    assert "liamani" not in joined, (  # CENSUS_EXEMPT: the discriminator
        "the citation author reached the general exemption list, which "
        "would permit the name everywhere. The permission is by PATH"
    )
    assert not any("citation" in entry.lower() for entry in CENSUS_EXEMPT), (
        "the citation was exempted by string rather than by path"
    )


def test_the_readme_names_the_panel_correctly():
    """The one file a public reader meets first."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "five clinicians" not in readme.lower().replace(
        "not five clinicians", ""
    )
    from cleft.data import scoresheet

    for column in scoresheet.RATERS:
        role = column.split(" - ", 1)[1].lower()
        assert role in readme.lower(), role
