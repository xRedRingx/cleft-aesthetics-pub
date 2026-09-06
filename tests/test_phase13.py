"""Phase 13's registration: Reading B, criteria pending agreement."""

from __future__ import annotations

from pathlib import Path

from cleft import phase13


def test_reading_b_resolves_the_framing_tension_with_the_reckoning():
    record = phase13.PHASE_13_REGISTERED
    assert record["registered"].startswith("2026-08-24")
    assert record["reading"].startswith("B")
    assert "not a training objective" in record["reading"]

    # The tension is resolved dated, with the later-registration path
    # carrying the SCUT-transfer reckoning LDL-style -- and the figures
    # are the ladder's own.
    resolved = record["framing_tension_resolved"]
    assert "ACKNOWLEDGED" in resolved and "not pursued" in resolved
    assert "SEPARATE, LATER" in resolved
    assert "0.2520" in resolved and "0.1952" in resolved and "0.0830" in resolved
    from cleft import ladder

    cells = ladder.STAGE_D1_AT_G1["cells"]["vit_b16"]
    assert cells == (0.2520, 0.1952, 0.0830)

    assert "no encoder gradient" in record["by_construction"]
    silences = record["silences_resolved"]
    assert "shakedown arm FIRST" in silences["reconstructs"]
    assert "WHAT-THE-EMBEDDING-KEEPS" in silences["purpose"]
    assert "DESCRIPTIVE throughout" in silences["success"]
    assert "no new flag" in silences["governance"]
    assert silences["ladder_entry"].startswith("NONE")
    assert "RETRY_LIMIT_IS_NOT_HOLDING" in silences["compute"]


def test_the_failure_mode_reading_is_written_and_bound_in_advance():
    record = phase13.FAILURE_MODE_READING_BOUND
    assert record["committed"].endswith("before any arm exists")
    assert "not scoring like a clinician" in record["the_registered_warning"]
    reading = record["the_reading"]
    assert "NO BETTER (or worse)" in reading
    assert "EXPECTED PRIMARY FINDING, not a failure" in reading
    assert "four convergent nulls made visible" in reading
    assert "never re-composed after the images are seen" in record["bound_to"]


def test_the_exit_criteria_are_numbered_and_pending_agreement():
    record = phase13.PHASE_13_EXIT_CRITERIA
    criteria = record["criteria"]
    assert len(criteria) == record["expected_count"] == 7
    assert [c.split(".")[0] for c in criteria] == [str(i) for i in range(1, 8)]
    # [2026-08-24] Was PENDING at registration; the maintainer agreed the same
    # day with one amendment (criterion 5's statistic = the double
    # difference), and the agreement is dated in place.
    assert record["agreed"].startswith("2026-08-24")
    assert "one amendment" in record["agreed"]

    joined = " ".join(criteria)
    assert "SCUT SHAKEDOWN FIRST" in joined
    assert "3300/2200" in joined
    assert "cleft_reconstruction_acknowledged" in joined
    assert "CLUSTER-ONLY without exception" in joined
    assert "before any cohort\nreconstruction is seen" in joined or (
        "before any cohort" in joined
    )
    assert "nasolabial strip" in joined
    assert "trivially reconstructible" in joined
    assert "DESCRIPTIVE only" in joined and "no ledger row" in joined
    assert "BY CONSTRUCTION" in joined
    assert "WRITTEN, not\ndiscarded" in joined or "WRITTEN, not" in joined


def test_the_regional_readings_commit_both_ways_before_any_image():
    record = phase13.REGIONAL_COMPARISON_READINGS
    assert record["registered"].endswith("before any reconstruction exists")
    # Normalised by the image's own error, mapped by shipped machinery.
    assert "normalised by the image's own" in record["the_measure"]
    assert "nothing new placed by hand" in record["the_measure"]
    assert "failure-mode reading fires" in record["reading_if_relevant_worse"]
    # The null direction gets an honest sentence too, not a shrug.
    better = record["reading_if_no_difference_or_better"]
    assert "sharpens the puzzle" in better
    assert "downstream of appearance retention" in better
    assert "DESCRIPTIVE" in record["descriptive_throughout"]

    # The regional vocabulary is the shipped anatomy bands.
    import inspect

    from cleft.geometry import patches

    source = inspect.getsource(patches)
    for band in ("eyes", "nose", "lips"):
        assert band in source


def test_the_architecture_is_proposed_with_reasons_not_picked():
    record = phase13.DECODER_ARCHITECTURE_PROPOSED
    assert "for the maintainer to accept or amend" in record["proposed"]
    assert "768 -> 7x7x256" in record["shape"]
    assert "checkerboard" in record["why_this_shape"]
    # The instrument argument: trained on SCUT only, cleft never seen.
    assert "never sees a cleft pixel in training" in record["trained_on"]
    assert "REGISTERED\nEXTENSION" in record["trained_on"] or (
        "REGISTERED" in record["trained_on"]
    )
    # The masked-vs-original question answered explicitly, with reasons.
    assert "MASKED artifact is the target, explicitly" in (
        record["target_masked_not_original"]
    )
    assert "distribution-matched sibling" in (
        record["target_masked_not_original"]
    )
    assert "224 G1" in record["resolution_224_g1"]
    assert "never\nsaw" in record["resolution_224_g1"] or (
        "never" in record["resolution_224_g1"]
    )
    # SSIM proposed, LPIPS named and declined; L2 trains, SSIM reports.
    assert "not optimised" in record["loss"]
    assert "registered failure mode by construction" in record["loss"]
    assert "named and declined" in record["perceptual_metric_choice"]
    # What needs the pretraining pipeline: nothing -- flagged honestly.
    assert "NOTHING needs task_pretrain" in record["what_needs_new_code"]
    assert "NEW code" in record["what_needs_new_code"]


def test_the_shakedown_gate_and_stops_follow_the_standing_precedents():
    shakedown = phase13.SCUT_SHAKEDOWN_PLAN
    assert len(shakedown["steps"]) == 4
    assert any("3,300" in s for s in shakedown["steps"])
    assert any("2,200" in s and "never\nfit" in s or "never" in s
               for s in shakedown["steps"])
    assert any("by eye or\nnot at all" in s or "by eye" in s
               for s in shakedown["steps"])
    assert "no governance question exists" in shakedown["what_it_validates"]

    gate = phase13.CLEFT_RECONSTRUCTION_GATE
    assert gate["flag"] == "cleft_reconstruction_acknowledged"
    assert "GENERATED PATIENT-DERIVED" in gate["why"]
    assert "NO\ndefault" in gate["mechanism"] or "NO" in gate["mechanism"]
    assert "basal-flag precedent" in gate["mechanism"]
    assert "CLUSTER-ONLY\nwithout exception" in gate["tier"] or (
        "CLUSTER-ONLY" in gate["tier"]
    )
    # The encoder-only question is answered, citing standing practice.
    assert "the ladder does\nit" in gate["encoding_needs_no_flag"] or (
        "the ladder does" in gate["encoding_needs_no_flag"]
    )

    stops = phase13.PHASE_13_STOPS
    assert set(stops["stops"]) == {"1", "2", "3", "rule"}
    assert "REGISTERED here, before any cohort" in stops["stops"]["1"]
    assert stops["rule" if "rule" in stops else "stops"] is not None
    assert stops["stops"]["rule"] == "stop after each for review"


def test_the_phase_summary_carries_every_record():
    summary = phase13.summary()
    assert set(summary) == {
        "registered", "failure_mode", "exit_criteria", "regional_readings",
        "architecture", "shakedown", "gate", "stops",
        "double_difference", "stop_1", "parameter_band",
        "stop_1_reviewed", "scut_normal_prior", "stop_2",
        "stop_2_reviewed", "coarse_severity_hypothesis", "stop_3",
        "everywhere_gap", "reading_applied_to_nobody",
        "reading_count_guard", "writer_died_after_save",
        "anchor_quantities_differ", "stop_3_banked",
        "probe", "closing", "closing_addendum",
        "p1_banked", "band_names", "p2_banked", "p3_triggered",
        "p3_banked", "asym_encoding_probe", "asym_probe_banked",
    }


# --------------------------------------------------------------------------
# 2026-08-24, the criteria agreed with the double-difference amendment;
# stop 1 built
# --------------------------------------------------------------------------


def test_the_criteria_are_agreed_and_the_count_correction_recorded():
    record = phase13.PHASE_13_EXIT_CRITERIA
    assert record["agreed"].startswith("2026-08-24")
    assert "double difference" in record["agreed"]
    # The held-out count is corrected in place: 2,199, CM152 excluded.
    assert "2,199, not 2,200" in record["small_correction"]
    assert "CM152" in record["small_correction"]


def test_the_double_difference_rule_answers_the_defect_before_registering():
    record = phase13.DOUBLE_DIFFERENCE_RULE
    assert record["registered"].endswith("before any reconstruction exists")
    assert "SAME\nband's error" in record["statistic"] or (
        "SAME" in record["statistic"]
    )
    assert "difficulty baseline" in record["why"]
    # The defect is named with its DIRECTION -- against the failure-mode
    # reading -- and the mitigation is in the rule, not a later patch.
    assert "deflates" in record["defect_answered"]
    assert "AGAINST the failure-mode reading" in record["defect_answered"]
    assert "CONTENT PIXELS ONLY" in record["mitigation_registered"]
    assert "250/channel" in record["mitigation_registered"]
    assert "not patched later" in record["mitigation_registered"]
    # The comparability measurement has a threshold and a reading.
    assert "more than 2x" in record["comparability_measurement"]
    assert "report, never gate" in record["comparability_measurement"]
    # The residual limitation is carried, and the band rule is explicit
    # about its one departure from criterion 3's letter.
    assert "POSITIONAL, not anatomical" in record["residual_limitation"]
    assert "declined as machinery" in record["residual_limitation"]
    assert "THIRDS" in record["band_rule"]
    assert "trapezium mapping SCUT lacks" in record["band_rule"]
    assert "departure from criterion 3's letter" in record["band_rule"]
    assert "DOUBLE DIFFERENCE" in record["readings_attach_here"]


def test_the_decoder_builds_to_its_registered_shape():
    pytest = __import__("pytest")
    torch = pytest.importorskip("torch")

    from cleft import decoder

    # [2026-08-24] the maintainer picked (a) on PARAMETER_BAND_MISESTIMATED:
    # the 7x7x96 shape at EXACTLY 3,862,339 parameters, registered by
    # exact equality -- no band, no estimate. build() constructs again.
    model = decoder.build(seed=1337)
    total = sum(p.numel() for p in model.parameters())
    assert total == decoder.REGISTERED_PARAMETERS == 3_862_339
    assert total == decoder.expected_parameters()

    # Upsample+conv by registration -- no ConvTranspose anywhere, so
    # checkerboard artifacts cannot contaminate the regional comparison.
    assert not any(
        isinstance(m, torch.nn.ConvTranspose2d) for m in model.modules()
    )

    # Forward: 768-d pooled embedding -> 224x224x3, sigmoid in [0, 1].
    with torch.no_grad():
        out = model(torch.randn(2, decoder.EMBED_DIM))
    assert tuple(out.shape) == (2, 3, 224, 224)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0

    # Seeding is deterministic: same seed, same weights; a different
    # seed differs.
    again = decoder.build(seed=1337)
    for p, q in zip(model.parameters(), again.parameters()):
        assert torch.equal(p, q)
    other = decoder.build(seed=7)
    assert any(
        not torch.equal(p, q)
        for p, q in zip(model.parameters(), other.parameters())
    )

    # A drifted construction is refused by exact equality: patch the
    # registered literal and the guard fires.
    import unittest.mock

    with unittest.mock.patch.object(decoder, "REGISTERED_PARAMETERS", 1):
        with pytest.raises(decoder.DecoderError, match="exact equality"):
            decoder.build(seed=1337)


def test_ssim_and_band_machinery_behave():
    import numpy as np

    from cleft import decoder

    rng = np.random.default_rng(0)
    image = (rng.random((64, 48, 3)) * 255).astype(np.uint8)
    # Identity reconstructs perfectly; noise scores below it.
    assert abs(decoder.ssim(image, image) - 1.0) < 1e-9
    noisy = np.clip(
        image.astype(float) + rng.normal(0, 40, image.shape), 0, 255
    )
    assert decoder.ssim(image, noisy) < 0.9

    # Content mask: white excluded on BOTH families' images.
    padded = np.full((30, 30, 3), 255, dtype=np.uint8)
    padded[10:20, 10:20] = 100
    mask = decoder.content_mask(padded)
    assert mask.sum() == 100
    # Bands are content-box thirds, in output rows.
    box = (0, 30, 100, 90)
    assert decoder.band_rows(box, "eyes") == (30, 60)
    assert decoder.band_rows(box, "nose") == (60, 90)
    assert decoder.band_rows(box, "lips") == (90, 120)
    with __import__("pytest").raises(decoder.DecoderError):
        decoder.band_rows(box, "forehead")

    # band_errors: content-only MSE, per band.
    reference = np.full((90, 30, 3), 100, dtype=np.uint8)
    reconstruction = reference.copy().astype(float)
    reconstruction[0:30] += 10.0   # eyes band off by 10
    errors = decoder.band_errors(
        reference, reconstruction, (0, 0, 30, 90)
    )
    assert abs(errors["eyes"]["mse"] - 100.0) < 1e-9
    assert errors["nose"]["mse"] == 0.0
    assert errors["lips"]["mse"] == 0.0
    assert errors["eyes"]["content_fraction"] == 1.0

    # The double difference: positive = grader-relevant worse.
    result = decoder.double_difference(
        {"eyes": 1.0, "nose": 4.0, "lips": 4.0},
        {"eyes": 1.0, "nose": 2.0, "lips": 2.0},
    )
    assert result["grade_irrelevant_mean"] == 1.0
    assert result["grader_relevant_mean"] == 2.0
    assert result["double_difference"] == 1.0
    assert "fires on" in result["positive_means"]
    with __import__("pytest").raises(decoder.DecoderError, match="unusable"):
        decoder.double_difference(
            {"eyes": 1.0, "nose": 1.0, "lips": 1.0},
            {"eyes": 0.0, "nose": 1.0, "lips": 1.0},
        )


def test_the_scut_artifact_roundtrips_and_owns_its_own_creation(tmp_path):
    import numpy as np

    from cleft import decoder

    values = np.arange(3 * 768, dtype=np.float32).reshape(3, 768)
    stems = ["AF1", "AM2", "CF3"]
    payload = decoder.save_scut_embeddings(
        tmp_path / "set_v1", values, stems, {"backbone": "vit_b16"}
    )
    assert len(payload["rollup"]) == 64
    loaded, loaded_stems, metadata = decoder.load_scut_embeddings(
        tmp_path / "set_v1"
    )
    assert np.array_equal(loaded, values)
    assert loaded_stems == stems
    assert metadata["backbone"] == "vit_b16"

    # The save layer owns existence: a second save refuses on the
    # artifact (the guard-after-mkdir lesson, applied at build time).
    with __import__("pytest").raises(
        decoder.DecoderError, match="already exists"
    ):
        decoder.save_scut_embeddings(
            tmp_path / "set_v1", values, stems, {}
        )
    # And the shape contracts hold.
    with __import__("pytest").raises(decoder.DecoderError, match="stems"):
        decoder.save_scut_embeddings(
            tmp_path / "set_v2", values, stems[:2], {}
        )
    with __import__("pytest").raises(decoder.DecoderError, match="repeated"):
        decoder.save_scut_embeddings(
            tmp_path / "set_v3", values, ["A", "A", "B"], {}
        )


def test_the_tasks_keep_the_boundaries_the_registration_promises():
    import inspect

    from cleft import run as run_module

    for kind in ("extract_scut_embeddings", "train_scut_decoder"):
        assert kind in run_module.TASKS

    extract_source = inspect.getsource(run_module.task_extract_scut_embeddings)
    # Reuse, not reimplementation; the save layer owns the path.
    assert "pretrain.load_masked_features" in extract_source
    assert "extract.extract_features" in extract_source
    assert "decoder_module.save_scut_embeddings" in extract_source
    # Call shapes, not words -- the docstring NAMES the guard-after-mkdir
    # lesson, which is the point; what must be absent is the CALL.
    assert ".mkdir(" not in extract_source
    assert "out_dir.exists()" not in extract_source

    decoder_source = inspect.getsource(run_module.task_train_scut_decoder)
    # The pretraining discipline, stated and implemented.
    assert "no early stopping" in decoder_source.lower()
    assert "epoch_" in decoder_source and "torch.save" in decoder_source
    assert "curves.csv" in decoder_source
    # No cohort pixel: the cleft staged artifact feeds statistics only,
    # and no cleft embedding path exists in this task.
    assert "NO COHORT PIXEL" in decoder_source
    assert "embeddings_g1_ladder" not in decoder_source
    # The band baseline and comparability table are written.
    assert "scut_band_baseline.json" in decoder_source
    assert "comparability" in decoder_source
    assert "ratio > 2.0" in decoder_source
    # Sheets: worst-first plus a seeded sample; metrics over ALL.
    assert "sheet_worst" in decoder_source
    assert "replace=False" in decoder_source


def test_the_stop_one_configs_are_two_pass_and_carry_the_scut_inputs():
    import importlib.util

    import yaml

    repo = Path(__file__).resolve().parents[1]
    extract = yaml.safe_load(
        (repo / "configs" / "p13_extract_scut.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert extract["task"]["expect_train"] == 3300
    assert extract["task"]["expect_test"] == 2199
    assert all(
        set(e["rollup_sha256"]) != {"0"} for e in extract["inputs"]
    )
    # Carried verbatim from the shipped pretraining config.
    pretrain = yaml.safe_load(
        (repo / "configs" / "p6_pretrain_vit_b16_masked_g1.yaml").read_text(
            encoding="utf-8"
        )
    )
    pretrain_inputs = {e["name"]: e for e in pretrain["inputs"]}
    for entry in extract["inputs"]:
        assert entry == dict(pretrain_inputs[entry["name"]]), entry["name"]

    decoder_payload = yaml.safe_load(
        (repo / "configs" / "p13_scut_decoder.yaml").read_text(
            encoding="utf-8"
        )
    )
    task = decoder_payload["task"]
    assert task["epochs"] == 30
    assert task["ssim_inner_sample"] == 64
    assert (task["sheet_worst"], task["sheet_sample"]) == (24, 96)
    # [2026-08-24, declare pass done] The extraction ran and the
    # scut-embeddings rollup (44e558c2...) was filled and is carried by
    # the generator: every input verified, nothing left for guard 3.
    for entry in decoder_payload["inputs"]:
        assert set(entry["rollup_sha256"]) != {"0"}, entry["name"]
        assert len(entry["rollup_sha256"]) == 64
    scut_set = next(
        e for e in decoder_payload["inputs"]
        if e["name"] == "scut_embeddings"
    )
    assert scut_set["rollup_sha256"].startswith("44e558c2")

    header = (repo / "configs" / "p13_scut_decoder.yaml").read_text(
        encoding="utf-8"
    )
    assert "NO COHORT PIXEL IS GENERATED HERE" in header
    assert "never optimised" in header
    assert "WORST 24" in header
    assert "**RESOLVED.**" in header
    assert "PLACEHOLDER" not in header

    spec = importlib.util.spec_from_file_location(
        "generate_phase13_configs",
        repo / "scripts" / "generate_phase13_configs.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--check"]) == 0


def test_stop_one_is_recorded_with_the_policies_that_ride_it():
    record = phase13.STOP_1_BUILT
    assert record["built"].endswith("NOT launched")
    assert "as proposed" in record["architecture_accepted"]
    assert "cannot drift apart" in record["module"]
    assert "guard-after-mkdir" in record["tasks"]["extract_scut_embeddings"]
    assert "NO cohort pixel" in record["tasks"]["train_scut_decoder"]
    assert "WORST 24" in record["sheet_policy"]
    assert "never the sheet sample" in record["sheet_policy"]
    assert "trend line" in record["ssim_cost_control"]
    assert "two-pass" in record["configs"]
    assert "the visual check" in record["next"]


# --------------------------------------------------------------------------
# 2026-08-24, the parameter band mis-estimate: diagnosed, laptop-fixed
# --------------------------------------------------------------------------


def test_expected_parameters_is_the_constructed_count_without_torch():
    """The laptop fix for the mis-estimate class: pure integer
    arithmetic over the layer dims, runnable in the pinned venv, pinned
    to the cluster-measured truth."""
    from cleft import decoder

    # [2026-08-24] The registration after the pick: the 7x7x96
    # shape, EXACT equality between the literal, the arithmetic, and
    # (under torch, in the construction test) the built model.
    assert decoder.SEED_CHANNELS == 96
    assert decoder.STAGE_CHANNELS == (96, 96, 64, 32, 16)
    assert decoder.expected_parameters() == 3_862_339
    assert decoder.expected_parameters() == decoder.REGISTERED_PARAMETERS

    # The REFUSED original shape, kept reproducible by explicit args:
    # exactly the cluster's figure, with the seed Linear carrying the
    # bulk the estimate missed.
    assert decoder.expected_parameters(
        256, (256, 128, 64, 32, 16)
    ) == 10_628_771
    seed_linear = 768 * (7 * 7 * 256) + (7 * 7 * 256)
    assert seed_linear == 9_646_336

    # The two candidate (a) shapes, counts CONSTRUCTED not estimated;
    # the 96-start is the one picked.
    assert decoder.expected_parameters(96, (96, 96, 64, 32, 16)) == 3_862_339
    assert decoder.expected_parameters(
        128, (128, 128, 64, 32, 16)
    ) == 5_215_651


def test_the_mis_estimate_is_diagnosed_with_both_claims_verified():
    from cleft import phase13

    record = phase13.PARAMETER_BAND_MISESTIMATED
    assert record["defect"].endswith("resolution is the maintainer's")
    assert "+2 on the retry record" in record["run"]
    # Claim 1: the arithmetic, with the exact figures.
    assert "9,646,336" in record["verified_1"]
    assert "10,628,771" in record["verified_1"]
    assert "MIS-ESTIMATE at proposal time" in record["verified_1"]
    # Claim 2: the venv skip made the cluster the first construction.
    assert "importorskip" in record["verified_2"]
    assert "FIRST" in record["verified_2"]
    # The guard is defended: it did what it was built for.
    assert "did what it was built for" in record["verified_2"]
    # The lesson names its class and its fix.
    assert "ESTIMATED count is not a CONSTRUCTED count" in record["lesson"]
    assert "expected_parameters()" in record["lesson"]

    # Both resolutions carry arguments; the shrink carries CONSTRUCTED
    # counts that the arithmetic reproduces.
    a = record["a_shrink"]
    assert "MORE CONSERVATIVE instrument" in a["argument"]
    assert "3,862,339" in a["shapes_constructed_not_estimated"]["seed_96"]
    assert "5,215,651" in a["shapes_constructed_not_estimated"]["seed_128"]
    # The first draft's hand-estimated counts were themselves wrong --
    # caught by this very test's ancestor; the correction is in place.
    assert "demonstrated" in a["first_draft_corrected"]
    assert "exact" in a["re_registration_rule"]
    assert "no band, no estimate" in a["re_registration_rule"]
    assert "never trained a step" in a["nothing_lost"]
    b = record["b_amend"]
    assert "10,628,771" in b["argument"]
    assert "BLUNTS the instrument reading" in b["cost"]

    # The pick is recorded dated on the record itself: (a), 7x7x96,
    # 3,862,339, exact equality -- and the machinery agrees with it.
    picked = record["picked"]
    assert picked.startswith("2026-08-24")
    assert "(a)" in picked and "7x7x96" in picked
    assert "3,862,339" in picked
    assert "EXACT EQUALITY" in picked
    from cleft import decoder

    assert decoder.REGISTERED_PARAMETERS == 3_862_339
    assert decoder.expected_parameters() == decoder.REGISTERED_PARAMETERS

    # The architecture proposal carries the dated correction in place,
    # its original wording preserved -- and the resolution beside it.
    corrected = phase13.DECODER_ARCHITECTURE_PROPOSED["corrected"]
    assert corrected.startswith("2026-08-24")
    assert "10,628,771" in corrected
    assert "~3-5M" in phase13.DECODER_ARCHITECTURE_PROPOSED["shape"] or (
        "3-5M" in phase13.DECODER_ARCHITECTURE_PROPOSED["shape"]
    )
    resolved = phase13.DECODER_ARCHITECTURE_PROPOSED["resolved"]
    assert "(a) shrink" in resolved and "3,862,339" in resolved


# --------------------------------------------------------------------------
# 2026-08-24, the shakedown reviewed PASS; stop 2 built behind the gate
# --------------------------------------------------------------------------


def test_the_shakedown_review_and_the_scut_normal_prior_are_recorded():
    reviewed = phase13.STOP_1_REVIEWED
    assert reviewed["reviewed"].endswith("PASS")
    assert "checkerboard ABSENT" in reviewed["machinery"]
    assert "trapezium geometry" in reviewed["machinery"]
    assert "HARD FACES" in reviewed["reconstructions"]
    assert "not machinery defects" in reviewed["reconstructions"]
    # The checkpoint pick is the maintainer's, dated, and names the file
    # through the one shared implementation.
    assert "EPOCH-20" in reviewed["checkpoint_pick"]
    assert "checkpoint_name(20)" in reviewed["checkpoint_pick"]
    from cleft import decoder

    assert decoder.checkpoint_name(20) == "epoch_20.pt"
    assert "BEFORE any cohort pixel" in (
        reviewed["criterion_1_precondition"]
    )

    prior = phase13.SCUT_NORMAL_PRIOR
    assert "before any cohort" in prior["registered"]
    assert "SCUT-NORMAL" in prior["observation"]
    assert "TOWARD" in prior["observation"] and (
        "AVERAGE" in prior["observation"]
    )
    # The mitigation is the double difference's band-relative design,
    # and the reading is bound to SAY SO whichever way it lands.
    assert "BAND-RELATIVE" in prior["mitigation"]
    assert "DOUBLE_DIFFERENCE_RULE" in prior["mitigation"]
    assert "MUST SAY SO" in prior["mitigation"]
    assert "cleft_band_errors.json" in prior["travels_with"]


def test_the_cohort_gate_is_first_and_the_task_keeps_the_boundaries():
    import inspect

    from cleft import decoder
    from cleft import run as run_module

    assert "reconstruct_cohort" in run_module.TASKS
    source = inspect.getsource(run_module.task_reconstruct_cohort)

    # The gate is the FIRST act: its check precedes every read. Call
    # sites, not words -- the imports sit above everything.
    gate_at = source.index('cleft_reconstruction_acknowledged") is not True')
    for read in (
        "load_manifest(manifest_dir", "cleft_embeddings_module.load(",
        "np.load(", "torch.load(",
    ):
        assert gate_at < source.index(read), read
    # The refusal names the reason and the person, not just the flag.
    assert "GENERATED PATIENT-DERIVED" in source
    assert "the maintainer" in source

    # Every reconstruction is CLUSTER-ONLY; the sheets carry the tier
    # at the call site, and the raw tensor is never persisted -- the
    # only pixels written are the sheets.
    sheet_at = source.index("cohort_reconstruction_sheet_")
    assert 'tier="CLUSTER-ONLY"' in source[sheet_at:sheet_at + 200]
    assert "np.save(" not in source
    # No double difference here: band_errors are banked, the composed
    # statistic is stop 3's. Call shape, not the phrase -- the
    # docstring and metrics NAME the statistic, which is the point.
    assert "decoder_module.band_errors(" in source
    assert "double_difference(" not in source
    # The prior travels by name, and the pick is reported beside the
    # shakedown's own selection -- never gated on it.
    assert "SCUT_NORMAL_PRIOR" in source
    assert "never gated" in source

    # One checkpoint-filename implementation, shared by the writer and
    # the reader.
    train_source = inspect.getsource(run_module.task_train_scut_decoder)
    assert "decoder_module.checkpoint_name(" in source
    assert "decoder_module.checkpoint_name(" in train_source
    assert decoder.checkpoint_name(7) == "epoch_07.pt"

    # The instrument reads its own representation and no other.
    assert '"imagenet"' in source
    assert "EMBED_DIM" in source

    record = phase13.STOP_2_BUILT
    assert record["built"].endswith("NOT launched")
    assert "refuses FIRST" in record["gate_wired"]
    assert "epoch-20" in record["instrument"]
    assert "reported, never gated" in record["instrument"]
    assert "all 237" in record["cohort"]
    assert "DELIBERATELY NOT PERSISTED" in record["outputs"]
    assert "NO double difference here" in record["boundary"]
    assert "ON THE CLUSTER" in record["review"]


def test_the_cohort_config_carries_the_gate_and_the_two_pass_input(tmp_path):
    import importlib.util

    import yaml

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "generate_phase13_configs_cohort",
        repo / "scripts" / "generate_phase13_configs.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # A fresh render -- no shipped config -- writes FALSE and the
    # PENDING input.
    missing = tmp_path / "nothing.yaml"
    assert module._carried_cohort_flag(missing) is False
    fresh = module._carried_decoder_run(missing)
    assert "/PENDING_" in fresh["path"]
    assert set(fresh["rollup_sha256"]) == {"0"}

    # the true, and a pasted real path, are carried from the
    # file -- even while the hash is still a placeholder.
    edited = tmp_path / "edited.yaml"
    edited.write_text(
        "inputs:\n"
        "- name: decoder_run\n"
        "  path: /cluster/runs/keeper/p13/p13_scut_decoder__ab12cd34__x\n"
        "  rollup_sha256: " + "0" * 64 + "\n"
        "task:\n"
        "  cleft_reconstruction_acknowledged: true\n",
        encoding="utf-8",
    )
    assert module._carried_cohort_flag(edited) is True
    carried = module._carried_decoder_run(edited)
    assert carried["path"].endswith("__x")
    # Anything short of literal true is false.
    stringy = tmp_path / "stringy.yaml"
    stringy.write_text(
        "task:\n  cleft_reconstruction_acknowledged: yes-ish\n",
        encoding="utf-8",
    )
    assert module._carried_cohort_flag(stringy) is False

    # The shipped config: the flag is a real boolean and the header
    # describes the state the body carries -- the value is the maintainer's,
    # never pinned (the phase-12 CI lesson).
    shipped = repo / "configs" / "p13_cohort_reconstruct.yaml"
    text = shipped.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    task = payload["task"]
    assert task["kind"] == "reconstruct_cohort"
    assert task["checkpoint_epoch"] == 20
    assert task["expect_patients"] == 237
    assert task["geometry"] == "g1"
    assert isinstance(task["cleft_reconstruction_acknowledged"], bool)
    if task["cleft_reconstruction_acknowledged"]:
        assert "A DATED\n# CONFIRMATION" in text
        assert "THE TASK WILL\n# REFUSE" not in text
    else:
        assert "THE TASK WILL\n# REFUSE" in text
        assert "A DATED\n# CONFIRMATION" not in text
    assert "CLUSTER-ONLY WITHOUT EXCEPTION" in text

    # Four inputs: the pending run dir plus three carried verbatim from
    # the shipped ladder config -- same paths, same verified hashes.
    inputs = {e["name"]: e for e in payload["inputs"]}
    assert set(inputs) == {
        "decoder_run", "embeddings", "manifest_v1", "staged_v1",
    }
    ladder_payload = yaml.safe_load(
        (repo / "configs" / "p7_d1_vit_b16_imagenet_g1.yaml").read_text(
            encoding="utf-8"
        )
    )
    ladder_inputs = {e["name"]: e for e in ladder_payload["inputs"]}
    for name in ("embeddings", "manifest_v1", "staged_v1"):
        assert inputs[name] == dict(ladder_inputs[name]), name
        assert set(inputs[name]["rollup_sha256"]) != {"0"}
    # The decoder_run input is consistent with the header's state --
    # pending until the paste, resolved thereafter.
    run_entry = inputs["decoder_run"]
    if "/PENDING_" in run_entry["path"]:
        assert set(run_entry["rollup_sha256"]) == {"0"}
        assert "PENDING" in text
    assert module.main(["--check"]) == 0


# --------------------------------------------------------------------------
# 2026-08-24, stop 2 reviewed (eye-impressions tagged); stop 3's scope
# --------------------------------------------------------------------------


def test_the_cohort_review_tags_every_impression_as_an_impression():
    record = phase13.STOP_2_REVIEWED
    assert record["reviewed"].endswith("PASS on fidelity")
    # The machinery verdict is a verdict; the pose-miss figure is not.
    assert "ZERO machinery artifacts" in record["fidelity_MEASURED_BY_EYE"]
    assert "AN IMPRESSION, NOT A COUNT" in record["fidelity_MEASURED_BY_EYE"]
    # The clinical-detail claims carry the tag in the KEY, not only the
    # prose -- a later turn reading keys alone still sees it.
    assert "EYE_IMPRESSION" in "".join(record)
    assert "EYE-IMPRESSION, NOT MEASUREMENT" in (
        record["clinical_detail_EYE_IMPRESSION"]
    )
    assert "SCUT-NORMAL" in record["clinical_detail_EYE_IMPRESSION"]
    worst = record["worst_12_EYE_IMPRESSION"]
    assert "EYE-IMPRESSION, NOT A COUNT" in worst
    assert "~6 of" in worst and "~4 of" in worst
    assert "never as" in worst and "measurements" in worst
    assert "must carry the tag" in record["tagging_rule"]


def test_the_severity_hypothesis_is_registered_as_untested():
    record = phase13.COARSE_SEVERITY_HYPOTHESIS
    assert "HYPOTHESIS THE NUMBERS MUST TEST" in record["registered"]
    assert "not claimable" in record["registered"]
    assert "COARSELY" in record["the_hypothesis"]
    assert "199 of 237" in record["the_hypothesis"]
    # It explains the plateau and the nulls only CONDITIONALLY.
    assert "IF IT SURVIVES MEASUREMENT" in record["what_it_would_explain"]
    assert "~0.25 plateau" in record["what_it_would_explain"]
    # A hypothesis with no failure mode is not a hypothesis.
    assert "how_it_could_fail" in record
    assert "0.2505" in record["how_it_could_fail"]


def test_stop_three_commits_every_reading_before_any_number():
    record = phase13.STOP_3_REGISTERED
    assert "BEFORE any number exists" in record["registered"]

    a = record["a_double_difference"]
    assert "CONTENT PIXELS ONLY" in a["what"]
    assert "BY NAME" in a["readings"]

    b = record["b_band_error_vs_grade"]
    assert "reading_if_rises" in b and "reading_if_flat" in b
    assert "SAMPLING" in b["reading_if_flat"]
    assert "NECESSARY, NOT SUFFICIENT" in b["registered_limit"]
    assert "report-never-gate" in b["status"]

    c = record["c_asymmetry_retention"]
    assert "band_asymmetry" in c["what"]
    assert "ORDERING" in c["statistics"]
    assert "SANITY ANCHOR" in c["statistics"]
    assert "0.158" in c["anchor"]
    # The blur caveat is registered WITH its design answer, not alone.
    assert "shrinks |left-right| EVERYWHERE" in c["blur_caveat_and_its_answer"]
    assert "RELATIVE" in c["blur_caveat_and_its_answer"]
    assert "IDENTICALLY" in c["blur_caveat_and_its_answer"]
    assert "no-persist" in c["pixels"]

    d = record["d_grade_decodability"]
    assert "BYTE-IDENTICAL" in d["what"]
    assert "0.2505" in d["reading_if_near_a"]
    assert "NO PAIRED CLAIM" in d["status"]
    assert "not contested" in d["status"]
    assert "CLUSTER-ONLY" in d["governance"]

    e = record["e_scar_trace_not_testable"]
    assert "NOT QUANTITATIVELY TESTABLE" in e["ruling"]
    assert "NEW UNVALIDATED" in e["ruling"]

    placement = record["compute_placement"]
    assert "(a) and (b)" in placement["pod_arithmetic"]
    assert "PURE ARITHMETIC" in placement["pod_arithmetic"]
    assert "(c) and (d)" in placement["gpu_jobs"]


def test_band_asymmetry_measures_what_it_claims():
    """Measured, not reasoned: the four properties the statistic rests
    on, including the units defect found before first use."""
    import numpy as np

    from cleft import decoder

    box = (10, 6, 40, 54)
    half = np.tile(np.arange(20, dtype=np.uint8)[None, :], (54, 1))
    symmetric = np.concatenate([half, half[:, ::-1]], axis=1)
    image = np.full((60, 60, 3), 255, np.uint8)
    image[6:60, 10:50] = symmetric[:, :, None]

    # A mirror-symmetric face scores exactly zero.
    assert decoder.band_asymmetry(image, box, "nose") == 0.0

    # Breaking one side scores above zero.
    broken = image.copy()
    broken[20:30, 12:20] = 0
    asymmetric = decoder.band_asymmetry(broken, box, "eyes")
    assert asymmetric > 0

    # [2026-08-24, THE DEFECT THIS TEST EXISTS FOR] mirror.as_float
    # scales uint8 but TRUSTS float as already [0, 1]. Originals are
    # uint8 and reconstructions are float 0-255, so dtype-trusting would
    # have inflated every retention ratio by ~255x -- reading as
    # "asymmetry fully retained", the exact claim under test. The same
    # content in either dtype must give the SAME number.
    assert decoder.band_asymmetry(
        broken.astype(np.float64), box, "eyes"
    ) == asymmetric

    # A pixel whose mirror partner is pad must not count as asymmetry:
    # content on one side only has no symmetric pair at all.
    one_sided = np.full((60, 60, 3), 255, np.uint8)
    one_sided[6:60, 10:30] = 40
    assert np.isnan(decoder.band_asymmetry(one_sided, box, "nose"))

    # Blur shrinks the measure -- the registered caveat, demonstrated.
    blurred = broken.astype(np.float64)
    for _ in range(8):
        blurred[1:-1, 1:-1] = (
            blurred[:-2, 1:-1] + blurred[2:, 1:-1] + blurred[1:-1, :-2]
            + blurred[1:-1, 2:] + blurred[1:-1, 1:-1]
        ) / 5
    assert decoder.band_asymmetry(blurred, box, "eyes") < asymmetric


def test_stop_three_tasks_keep_the_gate_and_the_no_persist_boundary():
    import inspect

    from cleft import run as run_module

    for kind in (
        "regional_comparison", "asymmetry_retention",
        "extract_reconstruction_embeddings",
    ):
        assert kind in run_module.TASKS

    # (a)+(b) is pure arithmetic: no pixels, no model, and therefore NO
    # gate -- asserted as the ABSENCE of the machinery, not as prose.
    regional = inspect.getsource(run_module.task_regional_comparison)
    assert "import torch" not in regional
    assert "decoder_module.build(" not in regional
    assert "cleft_reconstruction_acknowledged" not in regional
    assert "double_difference(" in regional
    # Both readings and the prior are applied from the record, not
    # re-typed here.
    assert "REGIONAL_COMPARISON_READINGS" in regional
    assert "SCUT_NORMAL_PRIOR" in regional
    assert "registered_limit" in regional

    # The regenerating helper carries the gate FIRST, before any read.
    helper = inspect.getsource(run_module._cohort_reconstructions)
    gate_at = helper.index('cleft_reconstruction_acknowledged") is not True')
    for read in ("load_manifest(", "cleft_embeddings_module.load(", "np.load("):
        assert gate_at < helper.index(read), read
    assert "REGENERATES" in helper
    assert "never written to disk" in helper
    # Nothing it produces lands on disk.
    assert "np.save(" not in helper
    assert "ctx.path(" not in helper

    # (c) uses the shared helper -- one gate implementation, not two --
    # and computes the SCUT floor with the SAME function.
    asym = inspect.getsource(run_module.task_asymmetry_retention)
    assert "_cohort_reconstructions(" in asym
    assert "band_asymmetry" in asym
    assert "scut_floor" in asym
    assert "MEASURED_BASELINE" in asym
    # (c) writes no pixels either.
    assert "render.save_sheet(" not in asym
    assert "np.save(" not in asym

    # (d) part 1 uses the same helper, writes a CLUSTER-ONLY set, and
    # does NOT run the probe itself -- that is a separate train_cv run.
    extract = inspect.getsource(
        run_module.task_extract_reconstruction_embeddings
    )
    assert "_cohort_reconstructions(" in extract
    assert "embeddings_module.save(" in extract
    assert "CLUSTER-ONLY" in extract
    assert "phase3.run(" not in extract


def test_the_stop_three_configs_are_generated_and_carry_their_sources():
    import importlib.util

    import yaml

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "generate_phase13_configs_stop3",
        repo / "scripts" / "generate_phase13_configs.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cohort = yaml.safe_load(
        (repo / "configs" / "p13_cohort_reconstruct.yaml").read_text(
            encoding="utf-8"
        )
    )
    cohort_inputs = {e["name"]: e for e in cohort["inputs"]}
    acknowledged = cohort["task"]["cleft_reconstruction_acknowledged"]

    # (a)+(b): no gate key at all -- the task cannot generate a pixel.
    regional = yaml.safe_load(
        (repo / "configs" / "p13_regional.yaml").read_text(encoding="utf-8")
    )
    assert regional["task"]["kind"] == "regional_comparison"
    assert "cleft_reconstruction_acknowledged" not in regional["task"]
    assert regional["task"]["checkpoint_epoch"] == 20
    header = (repo / "configs" / "p13_regional.yaml").read_text(
        encoding="utf-8"
    )
    assert "NO GATE" in header
    assert "Pod arithmetic, not a GPU job" in header

    # (c) and (d) part 1: the gate is INHERITED from the cohort config's
    # state -- the same decision, carried, never asked twice.
    for name in ("p13_asymmetry.yaml", "p13_extract_recon.yaml"):
        payload = yaml.safe_load(
            (repo / "configs" / name).read_text(encoding="utf-8")
        )
        assert payload["task"]["cleft_reconstruction_acknowledged"] is (
            acknowledged
        ), name
        assert payload["task"]["expect_patients"] == 237, name
        assert payload["task"]["checkpoint_epoch"] == 20, name
        # The shakedown run and the ladder inputs are carried VERBATIM
        # from the cohort config -- one declaration, not a second copy.
        inputs = {e["name"]: e for e in payload["inputs"]}
        for shared in ("decoder_run", "embeddings", "manifest_v1"):
            assert inputs[shared] == cohort_inputs[shared], (name, shared)

    # (d) part 2: the probe's task block is the shipped ladder arm's,
    # with exactly ONE key changed. That is what makes "byte-identical
    # recipe" a fact about the file rather than a claim.
    probe = yaml.safe_load(
        (repo / "configs" / "p13_probe_recon.yaml").read_text(
            encoding="utf-8"
        )
    )
    ladder = yaml.safe_load(
        (repo / "configs" / "p7_d1_vit_b16_imagenet_g1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert probe["task"]["kind"] == "train_cv"
    differing = {
        key for key in set(probe["task"]) | set(ladder["task"])
        if probe["task"].get(key) != ladder["task"].get(key)
    }
    assert differing == {"embeddings_artifact"}
    assert probe["task"]["embeddings_artifact"] == "embeddings_recon"
    assert probe["task"]["seeds"] == ladder["task"]["seeds"]
    probe_header = (repo / "configs" / "p13_probe_recon.yaml").read_text(
        encoding="utf-8"
    )
    assert "NO PAIRED CLAIM" in probe_header
    assert "CLUSTER-ONLY" in probe_header

    assert module.main(["--check"]) == 0


# --------------------------------------------------------------------------
# 2026-08-24, stop 3's first pass: one banked result, four defects fixed
# --------------------------------------------------------------------------


class _RecordingContext:
    """The smallest thing _registered_reading needs: a log sink."""

    def __init__(self):
        self.lines = []

    def log(self, message):
        self.lines.append(message)


def test_a_registered_reading_is_refused_over_the_wrong_row_count():
    """The structural guard: a reading's applies-when is part of the
    reading (phase13.READING_COUNT_GUARD)."""
    from cleft import run as run_module

    # Counts agree -> the sentence is applied and logged.
    ctx = _RecordingContext()
    applied = run_module._registered_reading(
        ctx, reading="the eye's impression was SAMPLING",
        joined=237, expected=237, what="b",
    )
    assert applied["applied"] is True
    assert applied["reading"].endswith("SAMPLING")
    assert any("SAMPLING" in line for line in ctx.lines)

    # Counts differ -> REFUSED. The sentence must not appear as an
    # applied reading, and the mismatch must be reported in its place.
    ctx = _RecordingContext()
    refused = run_module._registered_reading(
        ctx, reading="the eye's impression was SAMPLING",
        joined=0, expected=237, what="b, band error vs grade",
    )
    assert refused["applied"] is False
    assert "reading" not in refused
    assert refused["reading_withheld"].endswith("SAMPLING")
    assert refused["joined"] == 0 and refused["expected"] == 237
    joined_log = " ".join(ctx.lines)
    assert "READING REFUSED" in joined_log
    assert "237" in joined_log and "joined 0" in joined_log
    # The withheld sentence is named in the record but never presented
    # as having been applied.
    assert "registered reading (b" not in joined_log


def test_the_paired_mask_stops_pad_pixels_faking_asymmetry():
    """Defect (c-i), measured: a smooth reconstruction never reaches the
    white level, so its OWN mask swallows the trapezium pad the original
    excludes -- and the asymmetry it reports is the pad's, not the
    face's."""
    import numpy as np

    from cleft import decoder

    box = (10, 6, 40, 54)
    half = np.tile(np.arange(20, dtype=np.uint8)[None, :], (54, 1))
    original = np.full((60, 60, 3), 255, np.uint8)
    original[6:60, 10:50] = np.concatenate(
        [half, half[:, ::-1]], axis=1
    )[:, :, None]
    for row in range(6, 24):  # symmetric white trapezium corners
        inset = 24 - row
        original[row, 10:10 + inset] = 255
        original[row, 50 - inset:50] = 255

    # The reconstruction renders those corners near-white but BELOW the
    # threshold, with slightly uneven left-right corner rendering.
    reconstruction = original.astype(np.float64).copy()
    reconstruction[reconstruction >= 250] = 243.0
    reconstruction[6:24, 10:20] += 6.0

    own = decoder.band_asymmetry(reconstruction, box, "eyes")
    paired = decoder.band_asymmetry(reconstruction, box, "eyes", original)
    # The own-mask reading is inflated by pixels the reference excludes.
    assert own > paired > 0

    # And the two masks really are different sizes -- the mechanism.
    reference_mask = decoder.content_mask(original[6:24, 10:50])
    reference_mask &= reference_mask[:, ::-1]
    own_mask = decoder.content_mask(reconstruction[6:24, 10:50])
    own_mask &= own_mask[:, ::-1]
    assert int(own_mask.sum()) > int(reference_mask.sum())

    # A reference measured against itself is unchanged by the parameter.
    assert decoder.band_asymmetry(
        original, box, "eyes", original
    ) == decoder.band_asymmetry(original, box, "eyes")

    # A mask source of the wrong shape is refused, not broadcast.
    import pytest

    with pytest.raises(decoder.DecoderError, match="mask source"):
        decoder.band_asymmetry(
            reconstruction, box, "eyes", original[:20, :20]
        )


def test_the_stop_three_defects_are_recorded_with_their_roots():
    # (a) banked, with the larger fact bound to be stated first.
    everywhere = phase13.EVERYWHERE_GAP_IS_THE_LARGER_FACT
    assert "+0.0775" in everywhere["measured"]
    assert "3.2570" in everywhere["measured"] and "3.1794" in (
        everywhere["measured"]
    )
    assert "THE LARGER FACT IS THE EVERYWHERE-GAP" in (
        everywhere["must_be_stated_first"]
    )
    assert "THREE TIMES WORSE" in everywhere["must_be_stated_first"]
    assert "+2.4%" in everywhere["must_be_stated_first"]
    assert "REPORT-NEVER-GATE" in (
        everywhere["dispersion_required_before_quoting"]
    )
    assert "SCUT_NORMAL_PRIOR" in everywhere["what_the_everywhere_gap_means"]

    # (b) the retraction, by name, with the applies-when spelled out.
    retraction = phase13.READING_APPLIED_TO_NOBODY
    assert "ZERO patients" in retraction["defect"]
    assert "WITHDRAWN BY NAME" in retraction["withdrawn"]
    assert "FLAT CORRELATION OVER 237" in retraction["withdrawn"]
    assert "may not be cited" in retraction["withdrawn"]
    assert "still stands for the rerun" in retraction["withdrawn"]
    # The root names the shipped resolver and its own warning.
    assert "FRONTAL PHOTO" in retraction["root_cause"]
    assert "median_by_patient" in retraction["root_cause"]
    assert "THIRD copy" in retraction["root_cause"]
    assert "(c-ii)" in retraction["same_root_as"]

    # The structural guard, with why it is the right level of fix.
    guard = phase13.READING_COUNT_GUARD
    assert "test-pinned" in guard["registered"]
    assert "joined/row count differs" in guard["the_rule"]
    assert "WRONG SUBJECT" in guard["why_it_is_the_right_fix"]
    assert "APPLIES-WHEN is part of" in guard["why_it_is_the_right_fix"]
    assert "INTERPRETATION" in guard["refusal_is_not_failure"]
    assert "(c-ii)" in guard["fired_on"]

    # (d) the new failure family, and what is still unknown.
    writer = phase13.WRITER_DIED_AFTER_SAVE
    assert "KeyError" in writer["defect"]
    assert "+2 on the retry record" in writer["defect"]
    assert "TWO WRITERS, TWO SHAPES" in writer["root_cause"]
    assert "COMPLETENESS IS UNKNOWN" in writer["the_family_note"]
    assert "verify_recon_artifact.py" in writer["verdict_pending_measurement"]
    assert "hash_dir" in writer["fixed"]


def test_both_stop_three_tasks_resolve_grades_through_the_shipped_reader():
    import inspect

    from cleft import run as run_module

    # ONE label resolution: the shipped median_by_patient, reached
    # through one helper. Neither task may call load_median itself --
    # that was the third copy, and it joined nothing.
    helper = inspect.getsource(run_module._grades_for)
    assert "median_by_patient(" in helper
    assert "FRONTAL PHOTO" in helper

    for task in (
        run_module.task_regional_comparison,
        run_module.task_asymmetry_retention,
    ):
        source = inspect.getsource(task)
        assert "_grades_for(" in source, task.__name__
        assert "load_median(" not in source, task.__name__
        # And every registered reading goes through the count guard.
        assert "_registered_reading(" in source, task.__name__

    regional = inspect.getsource(run_module.task_regional_comparison)
    # The everywhere-gap is logged BEFORE the double difference's own
    # reading -- the larger fact first, in the output as in the record.
    assert "THE LARGER FACT FIRST" in regional
    assert regional.index("THE LARGER FACT FIRST") < regional.index(
        "_registered_reading("
    )
    # Dispersion is computed and named report-never-gate.
    assert "bootstrap_ci95" in regional
    assert "REPORT-NEVER-GATE" in regional

    asym = inspect.getsource(run_module.task_asymmetry_retention)
    # The paired mask is used for both members, both families.
    assert "band_mean(original, box, original)" in asym
    assert "band_mean(reconstructions[i], box, original)" in asym
    # Vectors are described before they are correlated, and a
    # degenerate operand is named rather than returned as a number.
    assert "def describe(" in asym
    assert "DEGENERATE OPERAND" in asym
    assert "is_constant" in asym
    # The SCUT floor is the WHOLE held-out side: the sample field is no
    # longer READ and the count is asserted instead. Call shape, not the
    # word -- the comment above the fix names the retired field, which
    # is the point of the comment (the standing word-matching lesson).
    assert 'task["scut_floor_sample"]' not in asym
    assert "[: int(" not in asym
    assert 'task["expect_scut_test"]' in asym

    extract = inspect.getsource(
        run_module.task_extract_reconstruction_embeddings
    )
    # The rollup comes from hash_dir, not from save's return value.
    assert "hash_dir(out_dir)" in extract
    assert "metadata_written" in extract


def test_the_stop_three_configs_carry_the_fixes():
    import importlib.util

    import yaml

    repo = Path(__file__).resolve().parents[1]
    regional = yaml.safe_load(
        (repo / "configs" / "p13_regional.yaml").read_text(encoding="utf-8")
    )
    # The manifest the grade join needs, carried with a real hash.
    inputs = {e["name"]: e for e in regional["inputs"]}
    assert "manifest_v1" in inputs
    assert set(inputs["manifest_v1"]["rollup_sha256"]) != {"0"}
    assert regional["task"]["manifest_artifact"] == "manifest_v1"
    assert regional["task"]["expect_patients"] == 237
    assert regional["task"]["n_boot"] == 10000

    asymmetry = yaml.safe_load(
        (repo / "configs" / "p13_asymmetry.yaml").read_text(encoding="utf-8")
    )
    # [defect (c-iii)] The cohort's 237 must never size a SCUT set.
    assert "scut_floor_sample" not in asymmetry["task"]
    assert asymmetry["task"]["expect_scut_test"] == 2199
    assert asymmetry["task"]["expect_patients"] == 237

    spec = importlib.util.spec_from_file_location(
        "generate_phase13_configs_fixes",
        repo / "scripts" / "generate_phase13_configs.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--check"]) == 0


def test_the_artifact_verifier_only_measures():
    """The partial-write verdict is the maintainer's. The script supplies
    evidence and must not act on the artifact itself."""
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "scripts" / "verify_recon_artifact.py").read_text(
        encoding="utf-8"
    )
    # Read-only: no deletion, no writing, no moving.
    for destructive in (
        "unlink(", "rmtree(", "shutil", "write_text(", "np.save(",
        "os.remove", "rename(",
    ):
        assert destructive not in source, destructive
    # It checks the things that distinguish whole from partial.
    for check in (
        "values.npy", "metadata.json", "patient_ids", "isfinite",
        "all-zero", "row order",
    ):
        assert check in source.lower() or check in source, check
    assert "VERDICT: WHOLE" in source and "VERDICT: PARTIAL" in source


# --------------------------------------------------------------------------
# 2026-08-24, stop 3 banked; the anchor comparison refused by measurement
# --------------------------------------------------------------------------


def test_the_anchor_comparison_is_refused_with_the_axes_named():
    record = phase13.ANCHOR_QUANTITIES_DIFFER
    assert "DIFFERENT QUANTITIES" in record["verdict"]
    assert "NOT a finding about the instrument" in record["verdict"]
    axes = record["the_axes"]
    assert set(axes) == {
        "statistic", "geometry", "midline", "mask_rule", "bands_and_label",
    }
    # The decisive axis: a trained fit vs a raw correlation.
    assert "RIDGE REGRESSION" in axes["statistic"]
    assert "RAW Pearson r of ONE scalar" in axes["statistic"]
    assert "MEAN label" in axes["bands_and_label"]
    assert "MEDIAN grade" in axes["bands_and_label"]
    assert "nobody compares +0.0131 to 0.158" in record["rule"]
    assert "ON ITS OWN MEASUREMENT" in record["consequence_for_c"]

    # And the axes are true of the code: Phase 4 really is a G2-guarded
    # 22-feature arm.
    from cleft.geometry import mirror

    assert mirror.REQUIRED_GEOMETRY == "g2"
    assert mirror.MEASURED_BASELINE["n_features"] == 22
    assert mirror.MEASURED_BASELINE["mean"] == 0.158
    assert "ridge" in mirror.MEASURED_BASELINE["arm"]
    assert "label mean" in mirror.MEASURED_BASELINE["arm"]

    # The registered anchor text now carries the dated correction and
    # points at this record; the original wording is preserved.
    anchor = phase13.STOP_3_REGISTERED["c_asymmetry_retention"]["anchor"]
    assert "0.158" in anchor
    assert "CORRECTED 2026-08-24" in anchor
    assert "ANCHOR_QUANTITIES_DIFFER" in anchor


def test_stop_three_is_banked_with_every_rider_attached():
    record = phase13.STOP_3_BANKED
    assert "the phase's last run" in record["banked"]

    a = record["a_double_difference"]
    assert "+0.0775" in a["figures"]
    assert "3.23x STATED FIRST" in a["figures"]
    assert "sd 0.6115" in a["figures"] and "58.6%" in a["figures"]
    assert "[-0.0023, +0.1538]" in a["figures"]
    assert "INCLUDES ZERO" in a["figures"]
    assert "UNRESOLVED AT 237 PATIENTS" in a["reading"]
    assert "descriptive cousin of the paired verdicts" in a["reading"]

    b = record["b_band_error_vs_grade"]
    assert "237/237" in b["figures"] and "READING_COUNT_GUARD" in b["figures"]
    assert "+0.0424" in b["figures"]
    assert "2330/2454/2491/2496/3376" in b["figures"]
    assert "STRICTLY\nMONOTONE" in b["figures"] or "STRICTLY" in b["figures"]
    assert "CLEANEST CONSISTENCY" in b["reading"]
    assert "CONSISTENCY-NOT-CONFIRMATION" in b["reading"]
    # The thin tails ride with the figures, always.
    assert "n=5" in b["reading"] and "n=3" in b["reading"]

    c = record["c_asymmetry_retention"]
    assert "+0.0082" in c["i_no_ordering"]
    assert "DECODER-GENERIC" in c["i_no_ordering"]
    assert "SHARPEST SINGLE NUMBER OF THE PHASE" in c["i_no_ordering"]
    assert "REFUTED AS MEASURED" in c["ii_caveat_refuted"]
    assert "1.01-1.15" in c["ii_caveat_refuted"]
    assert "ADDS generic asymmetry" in c["ii_caveat_refuted"]
    assert "1.01/1.15/1.09/0.98/0.71" in c["iii_retention_by_grade"]
    assert "floor 0.66" in c["iii_retention_by_grade"]
    assert "3\ngrade-5 patients" in c["iii_retention_by_grade"] or (
        "3 " in c["iii_retention_by_grade"]
    )
    assert "UNABLE TO STRONGLY TEST" in c["status"]
    assert "+0.0131" in c["status"]
    assert "ANCHOR_QUANTITIES_DIFFER" in c["status"]

    assert "ONE clean attempt" in record["d_artifact"]
    assert "WHOLE" in record["d_artifact"]
    assert "87d24577" in record["d_artifact"]
    assert "waits for\nthe declare pass" in record["d_artifact"] or (
        "declare pass" in record["d_artifact"]
    )
    assert "+3 on the retry record" in record["retries"]
    assert "FRACTIONAL GPU SLICE" in record["retries"]
    assert "does NOT guarantee a full card" in record["retries"]


def test_the_refuted_caveat_is_corrected_in_place_with_the_original_kept():
    text = phase13.STOP_3_REGISTERED["c_asymmetry_retention"][
        "blur_caveat_and_its_answer"
    ]
    # The original prediction is preserved verbatim...
    assert "raw \nretention sits below 1" in text or (
        "retention sits below 1" in text
    )
    # ...and the dated refutation rides beside it, with the measured
    # mechanism and what survives.
    assert "CORRECTED 2026-08-24, REFUTED AS MEASURED" in text
    assert "1.01-1.15" in text
    assert "ADDS generic\nasymmetry" in text or "ADDS generic" in text
    assert "survives its own caveat's refutation" in text


# --------------------------------------------------------------------------
# 2026-08-24, the probe banked and Phase 13 closed
# --------------------------------------------------------------------------


def test_the_probe_is_banked_with_its_triangulation_and_seed_note():
    record = phase13.PROBE_ON_RECONSTRUCTIONS
    assert "single attempt" in record["banked"]
    for seed_pcc in ("0.1871", "0.1679", "0.2062", "0.1242", "0.1610"):
        assert seed_pcc in record["figures"], seed_pcc
    assert "mean 0.1693" in record["figures"] and "sd 0.0305" in (
        record["figures"]
    )
    # The registered reading fired, and stayed descriptive.
    assert "WELL BELOW" in record["reading_applied"]
    assert "0.2505" in record["reading_applied"]
    assert "STATED CONTEXT" in record["reading_applied"]
    assert "~0.081" in record["reading_applied"]
    assert "NO PAIRED CLAIM" in record["reading_applied"]
    # The triangulation names both numbers it joins.
    assert "TWO-THIRDS" in record["triangulation"]
    assert "+0.0082" in record["triangulation"]
    assert "does NOT ride the" in record["triangulation"]
    assert "EYE-IMPRESSIONS" in record["triangulation"]
    # The noisy-arm pattern, with the thresholding rule attached.
    assert "4x A's own seed sd" in record["seed_sd_note"]
    assert "NOISY-ARM" in record["seed_sd_note"]
    assert "4.12.1" in record["seed_sd_note"]


def test_the_closing_walks_all_seven_criteria_and_adds_no_claim():
    closing = phase13.PHASE_13_CLOSING
    walk = closing["criteria_walk"]
    assert len(walk) == phase13.PHASE_13_EXIT_CRITERIA["expected_count"] == 7
    assert [key.split("_")[0] for key in walk] == [
        str(i) for i in range(1, 8)
    ]
    # Every criterion is answered MET -- and criterion 3 carries its
    # registered departure rather than claiming a clean pass.
    for key, text in walk.items():
        assert "MET" in text, key
    assert "POSITIONAL THIRDS" in walk["3_regional_split_registered_first"]
    assert "departure from criterion 3's letter" in (
        walk["3_regional_split_registered_first"]
    )
    # Criterion 4: the ledger really did not move.
    assert "31 entries" in walk["4_descriptive_only"]
    from cleft import results_ledger

    # [AMENDED 2026-08-29] This line read `len(ENTRIES) == 31` --
    # "the ledger did not move" written as a length pin, which is a
    # statement about a MOMENT expressed as a statement about forever
    # on an append-only ledger designed to grow. Phase 16's append
    # (entry 32) fired it. The durable form of what it guarded: the
    # 31 entries that existed at Phase 13's close are byte-unchanged
    # (the prefix checksum), and Phase 13 added none of them.
    assert results_ledger.cumulative_checksum(31) == (
        "c96084c2841c10061fad2f927a1324ca1fa7c442e71b78a2aea6e0787af2995d"
    )
    assert not any(
        entry["phase"] == "p13" for entry in results_ledger.ENTRIES
    )
    # Criterion 5 does NOT claim the failure-mode reading fired.
    assert "NOT claimed to have fired" in (
        walk["5_regional_comparison_with_readings"]
    )
    assert "UNRESOLVED at 237" in walk["5_regional_comparison_with_readings"]

    measured = closing["what_the_phase_measured"]
    assert set(measured) == {"a", "b", "c", "d", "e"}
    assert "INCLUDES ZERO" in measured["a"]
    assert "3.23x" in measured["a"]
    assert "consistency-not-confirmation" in measured["b"]
    assert "n=5" in measured["b"] and "n=3" in measured["b"]
    assert "+0.0082" in measured["c"] and "REFUTED" in measured["c"]
    assert "0.1693" in measured["d"] and "0.2505" in measured["d"]
    assert "untestable" in measured["e"]

    # The eye keeps its tags at the close.
    assert "EYE-IMPRESSIONS" in closing["what_the_eye_saw_and_what_it_is"]
    assert "a verdict" in closing["what_the_eye_saw_and_what_it_is"]
    # The hypothesis is not promoted by having survived.
    assert "NOT confirmed" in closing["the_hypothesis_after_measurement"]
    assert "leaves\nthe phase as it entered" in (
        closing["the_hypothesis_after_measurement"]
    ) or "as it entered" in closing["the_hypothesis_after_measurement"]

    residue = closing["defects_and_the_structural_residue"]
    assert "READING_COUNT_GUARD" in residue
    assert "WRITER_DIED_AFTER_SAVE" in residue
    assert "10,628,771" in residue and "3,862,339" in residue
    assert closing["no_new_claims"].startswith("every sentence")


def test_the_closing_carries_the_open_items_forward_by_name():
    carried = phase13.PHASE_13_CLOSING["carried_forward_by_name"]
    assert set(carried) == {
        "supervisor_question_8", "ar_confound_caveat", "fractional_gpu_note",
        "retry_limit_item", "instrument_limitation_from_c", "scar_traces",
    }
    assert "still open" in carried["supervisor_question_8"]
    # The AR caveat cites the record that owns it, with its figure.
    assert "BASAL_CONFOUND_OBSERVED" in carried["ar_confound_caveat"]
    assert "+0.1601" in carried["ar_confound_caveat"]
    from cleft import phase12

    assert "BASAL_CONFOUND_OBSERVED" in dir(phase12)
    # The new note and the retry arithmetic.
    assert "13.04 GiB" in carried["fractional_gpu_note"]
    assert "does NOT guarantee a full card" in carried["fractional_gpu_note"]
    assert "+2" in carried["retry_limit_item"] and "+3" in (
        carried["retry_limit_item"]
    )
    assert "not closed" in carried["retry_limit_item"]
    from cleft import phase8

    assert "RETRY_LIMIT_IS_NOT_HOLDING" in dir(phase8)
    # (c)'s limitation stays open AND refuses the tempting comparison.
    assert "OPEN" in carried["instrument_limitation_from_c"]
    assert "+0.0131" in carried["instrument_limitation_from_c"]
    assert "ANCHOR_QUANTITIES_DIFFER" in (
        carried["instrument_limitation_from_c"]
    )
    assert "NOT a comparison against Phase 4's 0.158" in (
        carried["instrument_limitation_from_c"]
    )
    assert "OPEN and untestable" in carried["scar_traces"]


# --------------------------------------------------------------------------
# 2026-08-24, the closing addendum: P1, P2 registered; P3 conditional
# --------------------------------------------------------------------------


def test_the_addendum_commits_both_readings_for_both_probes():
    record = phase13.CLOSING_ADDENDUM_REGISTERED
    assert "before any addendum number exists" in record["registered"]
    assert "only by exclusion" in record["why_it_belongs_to_phase_13"]

    p1 = record["p1_confound_ceiling"]
    assert "NON-ANATOMICAL GLOBAL" in p1["what"]
    assert "reading_if_substantial" in p1 and "reading_if_near_zero" in p1
    assert "MUST TRAVEL WITH THE HEADLINE NUMBER" in (
        p1["reading_if_substantial"]
    )
    assert "BOUNDED" in p1["reading_if_near_zero"]
    # It is the multivariate version of a measurement that already exists.
    assert "+0.1601" in p1["composes_with"]
    assert "MULTIVARIATE" in p1["composes_with"]
    assert "POD ARITHMETIC" in p1["compute"]
    assert "No gate" in p1["compute"]

    p2 = record["p2_band_occlusion"]
    assert "THREE times" in p2["what"]
    assert "byte-identical" in p2["what"]
    assert "COLLAPSES the PCC" in p2["reading_if_relevant_bands_collapse"]
    assert "DOES NOT\nCOME FROM THE CLINICAL REGION" in (
        p2["reading_if_relevant_bands_barely_dent"]
    ) or "DOES NOT" in p2["reading_if_relevant_bands_barely_dent"]
    # The blank value is justified, not assumed.
    assert "WHITE" in p2["white_not_black"]
    assert "Black would" in p2["white_not_black"]
    # The governance distinction is recorded rather than blurred.
    assert "INHERITANCE AND CAUTION" in p2["governance"]
    assert "MODIFIES an existing crop rather than GENERATING" in (
        p2["governance"]
    )
    assert "THREE GPU runs" in p2["compute"]

    p3 = record["p3_conditional_not_built"]
    assert "REGISTERED BUT NOT BUILT" in p3["condition"]
    assert "ONLY IF" in p3["condition"]
    assert "0.1693" in p3["what"] and "0.2505" in p3["what"]
    # The caveat is registered NOW, before any friendly number.
    assert "BAND-SMEARED" in p3["pre_registered_caveat"]
    assert "BLUNTER INSTRUMENT" in p3["pre_registered_caveat"]
    assert "registered NOW" in p3["pre_registered_caveat"]

    criteria = record["exit_addendum_criteria"]
    assert len(criteria) == 5
    assert [c.split(".")[0] for c in criteria] == [str(i) for i in range(1, 6)]
    joined = " ".join(criteria)
    assert "READING_COUNT_GUARD" in joined
    assert "no ledger row" in joined and "no paired claim" in joined
    assert "no cohort pixel is persisted" in joined
    assert "DATED\nADDENDUM" in joined or "DATED" in joined


def test_the_closing_is_amended_in_place_with_the_original_preserved():
    # [2026-08-24, later the same day] This test pinned the PENDING
    # string; the verdicts landed and the field became a dict carrying
    # the registration verbatim. The preserved-original property is what
    # this test always guarded, so it now asserts THAT, and the full
    # absorption is covered by
    # test_the_addendum_is_absorbed_with_the_original_preserved.
    closing = phase13.PHASE_13_CLOSING
    assert closing["no_new_claims"].startswith("every sentence")
    assert "MET" in closing["criteria_walk"]["1_shakedown_first"]
    registered = closing["addendum"]["registered"]
    assert registered.startswith("2026-08-24")
    assert "VERDICTS PENDING" in registered
    assert "CLOSING_ADDENDUM_REGISTERED" in registered
    assert "amended in place" in registered and "never rewritten" in registered


def test_occlude_band_blanks_exactly_one_third_and_copies():
    """Measured: the thirds partition the crop, the blank is white, and
    the caller's array is never modified."""
    import numpy as np

    from cleft import decoder

    image = np.full((60, 60, 3), 255, np.uint8)
    image[6:60, 10:50] = 120
    image[10:20, 12:24] = 40
    box = (10, 6, 40, 54)

    covered = []
    for band in decoder.BANDS:
        occluded = decoder.occlude_band(image, box, band)
        top, bottom = decoder.band_rows(box, band)
        covered.append((top, bottom))
        # The band is fully white...
        assert np.all(occluded[top:bottom] == 255), band
        # ...every other row is untouched...
        assert np.array_equal(
            np.delete(occluded, np.s_[top:bottom], axis=0),
            np.delete(image, np.s_[top:bottom], axis=0),
        ), band
        # ...and the caller's own array is unchanged (it is a COPY).
        assert image[10, 12, 0] == 40, band

    # The three bands partition the content box exactly: contiguous,
    # non-overlapping, covering it end to end.
    assert covered[0][0] == 6
    assert covered[-1][1] == 60
    for (_, end), (start, _) in zip(covered, covered[1:]):
        assert end == start


def test_the_global_statistics_are_five_and_none_can_see_anatomy():
    import numpy as np

    from cleft import decoder

    image = np.full((60, 60, 3), 255, np.uint8)
    image[6:60, 10:50] = 120
    box = (10, 6, 40, 54)
    stats = decoder.global_statistics(image, box)

    assert tuple(stats) == decoder.GLOBAL_STATISTIC_NAMES
    assert len(decoder.GLOBAL_STATISTIC_NAMES) == 5
    # The aspect ratio is the CONTENT BOX's own -- the quantity phase12
    # already measured one variable at a time.
    assert stats["aspect_ratio"] == 40 / 54
    # Brightness and contrast are over CONTENT pixels: an all-content
    # constant patch has zero spread whatever the pad does.
    assert stats["brightness_mean"] > 0
    assert stats["contrast_sd"] >= 0
    assert 0.0 <= stats["content_fraction"] <= 1.0
    assert 0.0 <= stats["corner_white_fraction"] <= 1.0
    # Every value is finite -- a nan here would silently poison the ridge.
    assert all(np.isfinite(v) for v in stats.values())


def test_the_addendum_tasks_keep_their_registered_boundaries():
    import inspect

    from cleft import run as run_module

    for kind in ("confound_ceiling", "extract_occluded_embeddings"):
        assert kind in run_module.TASKS

    # P1 is pod arithmetic: no backbone, no torch, no gate -- asserted
    # as the ABSENCE of the machinery.
    #
    # **[2026-08-31, THE PIN FIRED AS DESIGNED and the assertion moved
    # with the code it guards.]** The statistics and the ridge loop were
    # EXTRACTED to `_confound_statistics` / `_confound_ceiling_oof` so
    # Phase 20's Arm C could score the same five against the panel mean
    # through one implementation rather than a second copy
    # (phase20.ARM_C_REGISTERED). The boundary this test protects is
    # unchanged -- it is still pod arithmetic through the shipped ridge
    # and the shipped metric -- so the assertions now read the task
    # TOGETHER WITH the helpers it delegates to. Reading only the task
    # body would have quietly stopped checking anything.
    p1_task = inspect.getsource(run_module.task_confound_ceiling)
    p1 = p1_task + inspect.getsource(
        run_module._confound_statistics
    ) + inspect.getsource(run_module._confound_ceiling_oof)
    assert "import torch" not in p1
    assert "extract_features(" not in p1
    assert "cleft_reconstruction_acknowledged" not in p1
    # It uses the shipped ridge, the shipped metric and the shipped
    # label resolution -- no fourth copy of anything.
    assert "RidgeBackbone(" in p1
    assert "metrics.pcc(" in p1
    assert "_grades_for(" in p1_task
    assert "_registered_reading(" in p1_task
    # The threshold comes from the config, never from the number seen.
    assert 'task["substantial_pcc"]' in p1_task
    # And the extraction is an extraction, not a fork: exactly one of
    # each, and the task reaches both by call.
    for helper in ("_confound_statistics", "_confound_ceiling_oof"):
        assert f"{helper}(" in p1_task, helper
        assert inspect.getsource(run_module).count(f"def {helper}") == 1

    # P2 touches cohort pixels, so the gate is FIRST, before any read.
    p2 = inspect.getsource(run_module.task_extract_occluded_embeddings)
    gate_at = p2.index('cleft_reconstruction_acknowledged") is not True')
    for read in ("load_manifest(", "np.load("):
        assert gate_at < p2.index(read), read
    # The occluded crops never land; only the embedding set does.
    assert "occlude_band(" in p2
    assert "embeddings_module.save(" in p2
    assert "np.save(" not in p2
    assert "ctx.path(" not in p2
    assert "hash_dir(out_dir)" in p2
    # The honest note about WHY the gate is here.
    assert "INHERITANCE" in p2
    assert "MODIFIES" in p2 and "GENERATING" in p2


def test_the_addendum_configs_are_one_factor_and_two_pass():
    import importlib.util

    import yaml

    repo = Path(__file__).resolve().parents[1]
    cohort = yaml.safe_load(
        (repo / "configs" / "p13_cohort_reconstruct.yaml").read_text(
            encoding="utf-8"
        )
    )
    acknowledged = cohort["task"]["cleft_reconstruction_acknowledged"]

    # P1: no gate key at all, threshold declared, every input verified.
    confound = yaml.safe_load(
        (repo / "configs" / "p13_confound_ceiling.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert confound["task"]["kind"] == "confound_ceiling"
    assert "cleft_reconstruction_acknowledged" not in confound["task"]
    assert confound["task"]["substantial_pcc"] == 0.10
    assert confound["task"]["expect_patients"] == 237
    assert confound["task"]["seeds"] == [1337, 2024, 7, 99, 12345]
    for entry in confound["inputs"]:
        assert set(entry["rollup_sha256"]) != {"0"}, entry["name"]
    header = (repo / "configs" / "p13_confound_ceiling.yaml").read_text(
        encoding="utf-8"
    )
    assert "NO GATE" in header and "POD ARITHMETIC" in header

    # P2: three extraction configs differing in ONE key, and three
    # probes each one substitution away from A's shipped arm.
    ladder = yaml.safe_load(
        (repo / "configs" / "p7_d1_vit_b16_imagenet_g1.yaml").read_text(
            encoding="utf-8"
        )
    )
    tasks = {}
    for band in ("eyes", "nose", "lips"):
        payload = yaml.safe_load(
            (repo / "configs" / f"p13_occlude_{band}.yaml").read_text(
                encoding="utf-8"
            )
        )
        assert payload["task"]["occluded_band"] == band
        assert payload["task"]["out_version"] == (
            f"embeddings_occluded_{band}_v1"
        )
        assert payload["task"][
            "cleft_reconstruction_acknowledged"
        ] is acknowledged
        tasks[band] = payload["task"]

        probe = yaml.safe_load(
            (repo / "configs" / f"p13_probe_occluded_{band}.yaml").read_text(
                encoding="utf-8"
            )
        )
        differing = {
            key for key in set(probe["task"]) | set(ladder["task"])
            if probe["task"].get(key) != ladder["task"].get(key)
        }
        assert differing == {"embeddings_artifact"}, band
        assert probe["task"]["embeddings_artifact"] == "embeddings_occluded"

    # ONE FACTOR: the three extraction tasks differ in the band and the
    # artifact name it derives, and in nothing else.
    for band in ("nose", "lips"):
        differing = {
            key for key in tasks["eyes"]
            if tasks["eyes"][key] != tasks[band][key]
        }
        assert differing == {"occluded_band", "out_version"}, band

    spec = importlib.util.spec_from_file_location(
        "generate_phase13_configs_addendum",
        repo / "scripts" / "generate_phase13_configs.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--check"]) == 0


# --------------------------------------------------------------------------
# 2026-08-24, P1 banked at the boundary; the band-name correspondence
# --------------------------------------------------------------------------


def test_p1_is_banked_with_the_boundary_stated_not_smoothed():
    record = phase13.P1_CONFOUND_CEILING_BANKED
    assert "single clean pod attempt" in record["banked"]
    assert "237/237" in record["banked"]
    for pcc in ("+0.1183", "+0.0659", "+0.1216", "+0.0897", "+0.1044"):
        assert pcc in record["figures"], pcc
    assert "mean +0.1000" in record["figures"]
    assert "DECLARED threshold 0.10" in record["figures"]
    # The reading fired, limitations-grade, travelling with the headline.
    assert "FIRES" in record["reading_fired"]
    assert "LIMITATIONS-GRADE" in record["reading_fired"]
    assert "TRAVELS WITH THE HEADLINE NUMBER" in record["reading_fired"]
    # The boundary is part of the verdict, with the quotable sentence
    # fixed here so nobody later sharpens it.
    boundary = record["fires_at_the_boundary"]
    assert "EXACTLY on" in boundary
    assert "+0.066 to +0.122" in boundary
    assert "~40% of the headline arm's" in boundary
    assert "cannot see a nose" in boundary
    assert "never as a razor-edge pass" in boundary
    # Firing at the boundary is reportable ONLY because the threshold
    # predates the number.
    assert "declared before the number existed" in boundary
    # One confound sentence now, not two.
    assert "SUBSUMES" in record["composition"]
    assert "+0.1601" in record["composition"]
    assert "no ledger row" in record["status"]


def test_the_band_name_correspondence_binds_readings_to_stems():
    from cleft import decoder

    record = phase13.BAND_NAME_CORRESPONDENCE
    the_map = record["the_map"]
    assert set(the_map) == set(decoder.BANDS)
    assert the_map["eyes"].startswith("top")
    assert the_map["nose"].startswith("middle")
    assert the_map["lips"].startswith("bottom")

    # The map agrees with the MACHINERY, not just with itself: BANDS'
    # order IS the band_rows order, top to bottom.
    box = (0, 0, 224, 210)
    rows = [decoder.band_rows(box, band) for band in decoder.BANDS]
    assert rows == sorted(rows), "BANDS must run top to bottom"
    assert [band for band in decoder.BANDS] == ["eyes", "nose", "lips"]

    # 'middle+bottom' in the registered readings == GRADER_RELEVANT.
    assert decoder.GRADER_RELEVANT == ("nose", "lips")
    assert decoder.GRADE_IRRELEVANT == ("eyes",)
    assert "GRADER_RELEVANT" in record["the_binding"]
    assert "GRADE_IRRELEVANT" in record["the_binding"]

    # The incident is recorded honestly: dead filenames, no retry cost.
    assert "NONEXISTENT FILENAMES" in record["why_it_exists"]
    assert "no retry cost" in record["why_it_exists"]
    assert "nothing to void" in record["why_it_exists"]


def test_the_occlusion_probe_fills_are_carried_and_resolved():
    import yaml

    repo = Path(__file__).resolve().parents[1]
    expected = {
        "eyes": "f37e5df0", "nose": "30c4d33a", "lips": "bf161f23",
    }
    for band, prefix in expected.items():
        name = f"p13_probe_occluded_{band}.yaml"
        text = (repo / "configs" / name).read_text(encoding="utf-8")
        payload = yaml.safe_load(text)
        entry = next(
            e for e in payload["inputs"]
            if e["name"] == "embeddings_occluded"
        )
        assert entry["rollup_sha256"].startswith(prefix), band
        assert len(entry["rollup_sha256"]) == 64
        assert "**RESOLVED.**" in text, band
        assert "PLACEHOLDER" not in text, band


# --------------------------------------------------------------------------
# 2026-08-24, P2 banked (lips-dominant); P3 triggered and built
# --------------------------------------------------------------------------


def test_p2_is_banked_as_the_addendums_central_finding():
    record = phase13.P2_BAND_OCCLUSION_BANKED
    assert "single-attempt clean" in record["banked"]
    for figure in (
        "0.2143 [0.194, 0.236]", "0.1684 [0.144, 0.180]",
        "0.0885 [0.068, 0.125]", "0.2505", "0.1000",
    ):
        assert figure in record["figures"], figure
    # The reading fired, with the monotone localisation.
    assert "FIRES" in record["reading_fired"]
    assert "MONOTONE" in record["reading_fired"]
    assert "-0.036" in record["reading_fired"]
    assert "-0.082" in record["reading_fired"]
    assert "-0.162" in record["reading_fired"]
    # The central finding, with its discipline attached.
    finding = record["the_central_finding"]
    assert "LIPS-DOMINANT" in finding
    assert "repair site" in finding
    assert "INDEPENDENTLY\nTRAINED" in finding or "INDEPENDENTLY" in finding
    assert "floors and spreads" in finding
    # Eyes as the area-matched control -- what makes the lips drop real.
    assert "AREA-MATCHED CONTROL" in record["eyes_is_the_builtin_control"]
    assert "not an occlusion-area artifact" in (
        record["eyes_is_the_builtin_control"]
    )
    # The floor overlap, stated with both spreads.
    floor = record["lips_occluded_sits_at_the_floor"]
    assert "OVERLAPS" in floor
    assert "0.066-0.122" in floor
    assert "five blind\nimage statistics" in floor or "blind" in floor
    # The 0.1693 ~ 0.1684 note is an OBSERVATION and says so twice.
    observation = record["observation_not_finding"]
    assert "UNREGISTERED OBSERVATION" in observation
    assert "0.1693" in observation and "0.1684" in observation
    assert "never as a finding" in observation
    assert "no paired claim" in record["status"]


def test_p3_is_triggered_with_readings_committed_before_numbers():
    record = phase13.P3_TRIGGERED
    # The condition's own figures are cited -- the trigger is auditable.
    assert "-0.082" in record["triggered"] and "-0.162" in record["triggered"]
    assert "the instruction says run" in record["triggered"]
    assert "epoch-20" in record["what_runs"]
    assert "gate\ninherited" in record["what_runs"] or "gate" in (
        record["what_runs"]
    )
    assert "byte-identical" in record["what_runs"]
    # Both readings exist BEFORE any number.
    assert "PRESERVES the lip signal" in (
        record["reading_if_lips_dominant_like_the_original"]
    )
    assert "BAND-DIFFUSE" in record["reading_if_occlusions_barely_move_it"]
    assert "IS the finding" in record["reading_if_occlusions_barely_move_it"]
    # The caveat is attached exactly as registered, and predates P2.
    caveat = record["caveat_attached_as_registered"]
    assert "band-smeared" in caveat
    assert "BLUNTER" in caveat
    assert "before P2's\nnumbers existed" in caveat or "before P2" in caveat


def test_the_p3_task_composes_shipped_pieces_and_keeps_the_boundaries():
    import inspect

    from cleft import run as run_module

    assert "extract_occluded_recon_embeddings" in run_module.TASKS
    source = inspect.getsource(
        run_module.task_extract_occluded_recon_embeddings
    )
    # ONE composition of two shipped pieces -- no second gate copy, no
    # second occlusion copy.
    assert "_cohort_reconstructions(" in source
    assert "decoder_module.occlude_band(" in source
    assert 'cleft_reconstruction_acknowledged") is not True' not in source
    # Nothing persisted but the embedding set.
    assert "np.save(" not in source
    assert "ctx.path(" not in source
    assert "embeddings_module.save(" in source
    assert "hash_dir(out_dir)" in source
    # The caveat rides in the metrics, cited from the record.
    assert "pre_registered_caveat" in source
    assert "P3_TRIGGERED" in source


def test_the_p3_configs_mirror_p2s_one_factor_shape():
    import importlib.util

    import yaml

    repo = Path(__file__).resolve().parents[1]
    cohort = yaml.safe_load(
        (repo / "configs" / "p13_cohort_reconstruct.yaml").read_text(
            encoding="utf-8"
        )
    )
    cohort_inputs = {e["name"]: e for e in cohort["inputs"]}
    acknowledged = cohort["task"]["cleft_reconstruction_acknowledged"]
    ladder = yaml.safe_load(
        (repo / "configs" / "p7_d1_vit_b16_imagenet_g1.yaml").read_text(
            encoding="utf-8"
        )
    )

    tasks = {}
    for band in ("eyes", "nose", "lips"):
        payload = yaml.safe_load(
            (repo / "configs" / f"p13_occlude_recon_{band}.yaml").read_text(
                encoding="utf-8"
            )
        )
        task = payload["task"]
        assert task["kind"] == "extract_occluded_recon_embeddings"
        assert task["occluded_band"] == band
        assert task["out_version"] == f"embeddings_occluded_recon_{band}_v1"
        assert task["checkpoint_epoch"] == 20
        assert task["cleft_reconstruction_acknowledged"] is acknowledged
        tasks[band] = task
        # The four inputs are carried verbatim from the cohort config.
        inputs = {e["name"]: e for e in payload["inputs"]}
        for shared in (
            "decoder_run", "embeddings", "manifest_v1", "staged_v1",
        ):
            assert inputs[shared] == cohort_inputs[shared], (band, shared)

        probe = yaml.safe_load(
            (
                repo / "configs" / f"p13_probe_occluded_recon_{band}.yaml"
            ).read_text(encoding="utf-8")
        )
        differing = {
            key for key in set(probe["task"]) | set(ladder["task"])
            if probe["task"].get(key) != ladder["task"].get(key)
        }
        assert differing == {"embeddings_artifact"}, band

    # ONE FACTOR across the three extracts.
    for band in ("nose", "lips"):
        differing = {
            key for key in tasks["eyes"]
            if tasks["eyes"][key] != tasks[band][key]
        }
        assert differing == {"occluded_band", "out_version"}, band

    spec = importlib.util.spec_from_file_location(
        "generate_phase13_configs_p3",
        repo / "scripts" / "generate_phase13_configs.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--check"]) == 0


# --------------------------------------------------------------------------
# 2026-08-24, P3 banked (neither reading verbatim); the addendum absorbed
# --------------------------------------------------------------------------


def test_p3_records_the_third_outcome_and_applies_neither_reading():
    record = phase13.P3_BANKED
    for figure in (
        "0.0744 [0.051, 0.089]", "0.0375 [-0.008, 0.059]",
        "0.0975 [0.017, 0.161]", "0.1693",
    ):
        assert figure in record["figures"], figure
    assert "INTERVAL INCLUDES\nZERO" in record["figures"] or (
        "INTERVAL INCLUDES" in record["figures"]
    )
    assert "eyes\n-0.095" in record["figures"] or "-0.095" in record["figures"]
    assert "-0.132" in record["figures"] and "-0.072" in record["figures"]

    # The Phase 10 pattern: neither sentence applied, and the refusal
    # names WHY for each.
    neither = record["neither_reading_fires"]
    assert "APPLY NEITHER, AND SAY SO" in neither
    assert "SMALLEST drop" in neither
    assert "moved the probe substantially" in neither
    assert "neither may be quoted as\napplied" in neither or (
        "neither may be quoted" in neither
    )

    # Component (i): the inversion, both figures side by side.
    inversion = record["component_i_the_inversion"]
    assert "-0.162 on ORIGINALS" in inversion
    assert "-0.072 on RECONSTRUCTIONS" in inversion
    assert "DID NOT SURVIVE THE DECODE" in inversion
    assert "IN SUBSTANCE" in inversion
    # Component (ii): the entanglement, super-additive.
    entanglement = record["component_ii_the_entanglement"]
    assert "-0.30 against a 0.17 total" in entanglement
    assert "SUPER-ADDITIVE" in entanglement
    assert "BAND-DIFFUSE AND HOLISTIC" in entanglement

    # The caveat graduated from pre-registered to operative, and the
    # wide intervals are named as its signature, not noise.
    caveat = record["caveat_now_operative"]
    assert "NOW THE OPERATIVE\nLIMITATION" in caveat or "OPERATIVE" in caveat
    assert "out-of-distribution" in caveat
    assert "BLUNT BY CONSTRUCTION" in caveat
    assert "nose spanning zero" in caveat
    assert "NOTHING here is claimable" in record["status"]


def test_the_addendum_is_absorbed_with_the_original_preserved():
    # [2026-08-24] This test previously pinned the PENDING state; the
    # verdicts landed the same day and the field became a dict with the
    # registration preserved verbatim -- asserted here.
    closing = phase13.PHASE_13_CLOSING
    addendum = closing["addendum"]
    assert isinstance(addendum, dict)
    # The original registration text survives, word for word.
    assert addendum["registered"].startswith("2026-08-24, REGISTERED")
    assert "VERDICTS PENDING" in addendum["registered"]
    assert "never rewritten" in addendum["registered"]
    assert "absorbed here" in addendum["absorbed"]

    # The three verdicts, each citing its banked record.
    assert "+0.1000" in addendum["p1"] and "cannot see a nose" in (
        addendum["p1"]
    )
    assert "TRAVELS\nWITH THE HEADLINE" in addendum["p1"] or "TRAVELS" in (
        addendum["p1"]
    )
    assert "LIPS-DOMINANT" in addendum["p2"]
    assert "-0.162" in addendum["p2"] and "area-matched" in addendum["p2"]
    assert "NEITHER registered reading fired verbatim" in addendum["p3"]
    assert "INVERSION" in addendum["p3"] and "ENTANGLEMENT" in addendum["p3"]
    assert "OPERATIVE LIMITATION" in addendum["p3"]

    # The mechanism sentence: every clause carries its citation, and the
    # hypothesis is not quietly promoted by it.
    mechanism = addendum["mechanism_summary_cited_not_new"]
    for citation in (
        "P2_BAND_OCCLUSION_BANKED", "P1_CONFOUND_CEILING_BANKED",
        "monotone", "+0.0082", "PROBE_ON_RECONSTRUCTIONS",
    ):
        assert citation in mechanism, citation
    assert "NO CLAUSE\nhere is new" in mechanism or "NO CLAUSE" in mechanism
    assert "remains\nunconfirmed" in mechanism or "unconfirmed" in mechanism

    # The closing body itself was never rewritten.
    assert closing["no_new_claims"].startswith("every sentence")
    assert "MET" in closing["criteria_walk"]["1_shakedown_first"]


def test_the_addendum_criteria_walk_is_complete_and_honest():
    walk = phase13.PHASE_13_CLOSING["addendum"][
        "exit_addendum_criteria_walk"
    ]
    registered = phase13.CLOSING_ADDENDUM_REGISTERED[
        "exit_addendum_criteria"
    ]
    assert len(walk) == len(registered) == 5
    assert [key.split("_")[0] for key in walk] == [
        str(i) for i in range(1, 6)
    ]
    for key, text in walk.items():
        assert "MET" in text, key
    # Criterion 1 is honest about P3: the sentences were REFUSED on
    # direction, which is the guard's spirit, not a mechanical count.
    assert "NEITHER was applied" in walk["1_readings_precommitted"]
    assert "failed on\ndirection" in walk["1_readings_precommitted"] or (
        "direction" in walk["1_readings_precommitted"]
    )
    # Criterion 2 is verified against the ledger, not asserted.
    assert "31" in walk["2_descriptive_throughout"]
    from cleft import results_ledger

    # [AMENDED 2026-08-29] This line read `len(ENTRIES) == 31` --
    # "the ledger did not move" written as a length pin, which is a
    # statement about a MOMENT expressed as a statement about forever
    # on an append-only ledger designed to grow. Phase 16's append
    # (entry 32) fired it. The durable form of what it guarded: the
    # 31 entries that existed at Phase 13's close are byte-unchanged
    # (the prefix checksum), and Phase 13 added none of them.
    assert results_ledger.cumulative_checksum(31) == (
        "c96084c2841c10061fad2f927a1324ca1fa7c442e71b78a2aea6e0787af2995d"
    )
    assert not any(
        entry["phase"] == "p13" for entry in results_ledger.ENTRIES
    )
    assert "memory only" in walk["3_no_cohort_pixel_persisted"]
    assert "-0.082" in walk["4_p3_conditional_honoured"]
    assert "preserved verbatim" in walk["5_absorbed_original_preserved"]


# --------------------------------------------------------------------------
# 2026-08-24, the post-closing check: the asymmetry-encoding probe
# --------------------------------------------------------------------------


def test_the_encoding_probe_resolves_a_named_conflation():
    record = phase13.ASYMMETRY_ENCODING_PROBE_REGISTERED
    assert "before any number exists" in record["registered"]
    # The conflation is stated as two mechanisms one r cannot separate.
    conflation = record["the_conflation_it_resolves"]
    assert "+0.0082" in conflation
    assert "never encoded" in conflation
    assert "DECODER\nfails to render" in conflation or "fails to render" in (
        conflation
    )
    assert "shares this blindness" in conflation
    # The test bypasses the decoder and CONSUMES the banked scalars.
    direct = record["the_direct_test"]
    assert "FEATURE-SPACE" in direct
    assert "CONSUMED from the banked" in direct
    assert "never\nrecomputed" in direct or "never recomputed" in direct
    assert "POD" in direct and "no gate" in direct
    # Both readings, each a different sentence for (c).
    assert "IT WAS\nNEVER ENCODED" in record["reading_if_near_zero"] or (
        "NEVER ENCODED" in record["reading_if_near_zero"]
    )
    assert "ENCODER, not the decoder" in record["reading_if_near_zero"]
    assert "DECODER is the\nlossy stage" in record["reading_if_substantial"] or (
        "lossy stage" in record["reading_if_substantial"]
    )
    assert "MEASURED mechanism" in record["reading_if_substantial"]
    # The threshold predates the number, with its rationale and the
    # boundary-honesty precedent.
    threshold = record["threshold_declared"]
    assert "0.10" in threshold and "DECLARED IN THE CONFIG" in threshold
    assert "0.0228" in threshold
    assert "boundary-honesty" in threshold
    # The (c) limitation rides either way.
    limitation = record["limitation_rides_either_way"]
    assert "+0.0131" in limitation
    assert "not whether the scalar matters" in limitation
    assert "READING_COUNT_GUARD" in record["status"]

    # The closing carries the dated check, body preserved.
    check = phase13.PHASE_13_CLOSING["post_closing_check"]
    assert check.startswith("2026-08-24")
    assert "ASYMMETRY_ENCODING_PROBE_REGISTERED" in check
    assert "amends this\nclosing dated" in check or "dated" in check
    assert phase13.PHASE_13_CLOSING["no_new_claims"].startswith(
        "every sentence"
    )


def test_the_encoding_probe_task_is_pod_and_consumes_the_banked_scalars():
    import inspect

    from cleft import run as run_module

    assert "asymmetry_encoding_probe" in run_module.TASKS
    source = inspect.getsource(run_module.task_asymmetry_encoding_probe)
    # Pod: no torch, no extraction, no decoder build, no gate.
    assert "import torch" not in source
    assert "extract_features(" not in source
    assert "decoder_module" not in source
    assert "cleft_reconstruction_acknowledged" not in source
    # The shipped machinery, no new copies.
    assert "RidgeBackbone(" in source
    assert "metrics.pcc(" in source
    assert "_registered_reading(" in source
    # The target is CONSUMED from the banked run, and a missing scalar
    # is a refusal, not a silent drop.
    assert '"asym_orig"' in source
    assert "computes none" in source
    # The threshold comes from the config, and the boundary case is
    # named in the log (the P1 precedent).
    assert 'task["substantial_pcc"]' in source
    assert "AT THE BOUNDARY" in source


def test_the_encoding_probe_config_declares_the_threshold_and_two_pass():
    import importlib.util

    import yaml

    repo = Path(__file__).resolve().parents[1]
    text = (repo / "configs" / "p13_asym_encoding.yaml").read_text(
        encoding="utf-8"
    )
    payload = yaml.safe_load(text)
    task = payload["task"]
    assert task["kind"] == "asymmetry_encoding_probe"
    assert task["substantial_pcc"] == 0.10
    assert task["expect_patients"] == 237
    assert task["seeds"] == [1337, 2024, 7, 99, 12345]
    assert "cleft_reconstruction_acknowledged" not in task
    assert "NO GATE" in text
    inputs = {e["name"]: e for e in payload["inputs"]}
    assert set(inputs) == {"asymmetry_run", "embeddings", "manifest_v1"}
    # The run-dir input follows the standing two-pass; the carried two
    # are verified.
    run_entry = inputs["asymmetry_run"]
    if "/PENDING_" in run_entry["path"]:
        assert set(run_entry["rollup_sha256"]) == {"0"}
        assert "PLACEHOLDER" in text
    for name in ("embeddings", "manifest_v1"):
        assert set(inputs[name]["rollup_sha256"]) != {"0"}, name

    spec = importlib.util.spec_from_file_location(
        "generate_phase13_configs_asym",
        repo / "scripts" / "generate_phase13_configs.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--check"]) == 0


# --------------------------------------------------------------------------
# 2026-08-24, the encoding probe's verdict banked; consequences in place
# --------------------------------------------------------------------------


def test_the_encoding_probe_verdict_is_banked_with_its_figures():
    record = phase13.ASYMMETRY_ENCODING_PROBE_BANKED
    assert "single clean pod attempt" in record["banked"]
    assert "+0.4856" in record["figures"]
    assert "sd 0.0231" in record["figures"]
    assert "0.4463 to\n0.5013" in record["figures"] or "0.4463" in (
        record["figures"]
    )
    assert "DECISIVELY" in record["figures"]
    assert "DECODER is the\nlossy stage" in record["reading_fired"] or (
        "lossy stage" in record["reading_fired"]
    )
    assert "237/237" in record["reading_fired"]
    # The methods caution is named for the write-up, with its origin.
    caution = record["methods_caution_for_the_writeup"]
    assert "LOWER BOUND" in caution
    assert "richer witness" in caution
    assert "0.49-vs-0.008" in caution
    assert "why not feature-space" in caution
    assert "ONE POD RUN" in caution
    # The limitation is restated verbatim, and the headline account is
    # explicitly untouched.
    limitation = record["limitation_restated_verbatim"]
    assert "+0.0131" in limitation
    assert "not whether the scalar matters" in limitation
    assert "stands\nUNTOUCHED" in limitation or "UNTOUCHED" in limitation
    assert record["status"] == "DESCRIPTIVE, no ledger row"


def test_the_four_consequences_are_applied_in_place_originals_preserved():
    closing = phase13.PHASE_13_CLOSING

    # (i) (c)'s sentence: the original words are STILL THERE, and the
    # dated upgrade sits inside the same entry.
    c = closing["what_the_phase_measured"]["c"]
    assert "+0.0082" in c and "DECODER-GENERIC" in c and "REFUTED" in c
    assert "[UPGRADED 2026-08-24" in c
    assert "ENCODED AT r~0.49" in c
    assert "'the chain loses it'\n" in c or "is superseded" in c
    # The measured set is unchanged -- no new keys, the walk holds.
    assert set(closing["what_the_phase_measured"]) == {
        "a", "b", "c", "d", "e",
    }

    # (ii) the prior's mechanism is a measurement now, original fields
    # untouched.
    prior = phase13.SCUT_NORMAL_PRIOR
    assert "SCUT-NORMAL" in prior["observation"]
    mechanism = prior["measured_mechanism"]
    assert mechanism.startswith("2026-08-24")
    assert "EVEN WHERE THE" in mechanism
    assert "r~0.49" in mechanism and "0.0082" in mechanism
    assert "measured property of the decoder" in mechanism

    # (iii) the mechanism summary keeps every original clause and gains
    # exactly the cited one.
    summary_text = closing["addendum"]["mechanism_summary_cited_not_new"]
    assert "LIPS-DOMINANT" in summary_text
    assert "unconfirmed" in summary_text
    assert "[ONE CLAUSE ADDED\n2026-08-24" in summary_text or (
        "ONE CLAUSE ADDED" in summary_text
    )
    assert "RICHER WITNESS" in summary_text
    assert "LOWER BOUND" in summary_text

    # The post-closing check records its landing, registration intact.
    check = closing["post_closing_check"]
    assert "ASYMMETRY_ENCODING_PROBE_REGISTERED" in check
    assert "[LANDED 2026-08-24" in check
    assert "+0.4856" in check

    # (iv) nothing entered the ledger.
    from cleft import results_ledger

    # [AMENDED 2026-08-29] This line read `len(ENTRIES) == 31` --
    # "the ledger did not move" written as a length pin, which is a
    # statement about a MOMENT expressed as a statement about forever
    # on an append-only ledger designed to grow. Phase 16's append
    # (entry 32) fired it. The durable form of what it guarded: the
    # 31 entries that existed at Phase 13's close are byte-unchanged
    # (the prefix checksum), and Phase 13 added none of them.
    assert results_ledger.cumulative_checksum(31) == (
        "c96084c2841c10061fad2f927a1324ca1fa7c442e71b78a2aea6e0787af2995d"
    )
    assert not any(
        entry["phase"] == "p13" for entry in results_ledger.ENTRIES
    )
