"""Phase 14: opened and closed at the reckoning -- CONCEDED-COVERED."""

from __future__ import annotations

from cleft import phase14


def test_the_phase_is_conceded_covered_on_a_line_by_line_identity():
    record = phase14.PHASE_14_CONCEDED_COVERED
    assert "CONCEDED-\nCOVERED" in record["closed"] or "CONCEDED-" in (
        record["closed"]
    )
    assert "Nothing was built" in record["closed"]

    # The prior measurement is cited with its own figures, both
    # operating points, and the pinned verdict sentence.
    prior = record["the_prior_measurement"]
    assert "STAGE_G_LABEL_FORMULATION" in prior
    assert "0.2336" in prior and "0.2520" in prior
    assert "CLAIMABLY HARMFUL" in prior
    assert "no information" in prior

    # The identity is stated axis by axis and sourced to the module,
    # not to memory -- the reckoning's whole discipline.
    identity = record["the_mechanism_identity"]
    assert "train/ldl.py" in identity
    assert "LDLHeadBackbone" in identity
    assert "FIVE-OUTPUT SOFTMAX" in identity
    assert "KL LOSS" in identity
    assert "SOFT_1..SOFT_5" in identity
    assert "EXPECTATION" in identity
    assert "HEAD, LOSS, TARGET, READOUT,\nRECIPE" in identity or (
        "HEAD, LOSS, TARGET, READOUT" in identity
    )
    assert "not a judgement call" in identity

    # And the identity claim is TRUE of the module, verified here
    # rather than trusted: the closed arm's head reports itself.
    from cleft.train import ldl

    head = ldl.LDLHeadBackbone(embedding_dim=768)
    assert ldl.N_GRADES == 5
    import inspect

    source = inspect.getsource(ldl.LDLHeadBackbone)
    assert "kl_div" in source
    assert '"head": "softmax_5"' in source
    assert '"loss": "kl_divergence"' in source
    assert "expectation over grades 1-5" in source
    assert head.embedding_dim == 768


def test_the_supporting_evidence_is_cited_and_the_variants_stay_unregistered():
    record = phase14.PHASE_14_CONCEDED_COVERED
    evidence = record["supporting_evidence_as_cited"]
    assert "CLEFTGNN_COMPARATOR_TABLES" in evidence["rater_screen"]
    assert "4 OF 5 BELOW" in evidence["rater_screen"]
    assert "0.0305" in evidence["the_closed_arms_own_sd"]
    assert "0.0148" in evidence["the_closed_arms_own_sd"]
    assert "LOOSER, not tighter" in evidence["the_closed_arms_own_sd"]
    assert "lips-dominant" in evidence["coarse_features_bar"]
    assert "199 of 237" in evidence["coarse_features_bar"]

    variants = record["variants_named_not_pursued"]
    assert "EXPLICITLY UNREGISTERED" in variants
    assert "no fishing" in variants
    assert "PRETRAINING" in variants
    assert "DIFFERENT REPRESENTATION" in variants
    assert "fresh idea" in variants

    # The record says what would reopen -- closing is not forbidding.
    assert "NEW measured reason" in record["what_would_reopen"]
    assert "closing is not forbidding" in record["what_would_reopen"]
    # The sequence advances to 15 with its registered criterion.
    assert "Phase 15" in record["sequence_advances"]
    assert "comparative verdict\nagainst SCUT" in (
        record["sequence_advances"]
    ) or "SCUT" in record["sequence_advances"]
    assert record["ledger"].startswith("no entry")
    assert set(phase14.summary()) == {"conceded_covered"}


def test_the_reckoning_fields_misdescription_is_corrected_in_place():
    from cleft import phase12

    record = phase12.PHASE_SEQUENCE_RENUMBERED_2
    # The original clause survives verbatim -- the standing pattern.
    assert "through the standard head" in (
        record["ldl_prior_measurement_to_reckon_with"]
    )
    # The correction sits beside it, dated, figures standing.
    corrected = record["ldl_mechanism_description_corrected"]
    assert corrected.startswith("2026-08-24")
    assert "did NOT go through" in corrected
    assert "train/ldl.py" in corrected
    assert "THE PINNED FIGURES\nSTAND" in corrected or (
        "PINNED FIGURES" in corrected
    )
    assert "READING THE MODULE" in corrected
    assert "PHASE_14_CONCEDED_COVERED" in corrected
