"""Phase 13 -- the decoder, as a representation-measurement instrument.

Opened 2026-08-24 under ``phase12.PHASE_SEQUENCE_RENUMBERED_2``. Framing
decided : **Reading B** -- the decoder measures and visualises
what the frozen embedding keeps; it is NOT a training objective, and the
encoder and the ladder are untouched by construction.
"""

from __future__ import annotations


class Phase13Error(RuntimeError):
    """A Phase 13 contract is not usable."""


#: **[REGISTERED 2026-08-24, the framing decision] THE PHASE,
#: WHOLE -- READING B, WITH THE FRAMING TENSION RESOLVED DATED.**
PHASE_13_REGISTERED = {
    "registered": "2026-08-24, framing decided ",
    "reading": (
        "B -- the decoder is a REPRESENTATION-MEASUREMENT AND "
        "VISUALISATION INSTRUMENT, not a training objective"
    ),
    "framing_tension_resolved": (
        "the amendment's 'training objective' wording "
        "(PLAN_AMENDMENT_2026-08-13 section 1's table) is ACKNOWLEDGED "
        "and not pursued in this phase. A reconstruction-pretrained "
        "encoder entering the ladder would be a SEPARATE, LATER "
        "registration carrying the SCUT-transfer reckoning LDL-style: "
        "imagenet 0.2520 against scut_original 0.1952 and scut_masked "
        "0.0830 (ladder.STAGE_D1_AT_G1), and 'SCUT test PCC turned out "
        "not to predict cleft transfer at all'"
    ),
    "by_construction": (
        "the decoder trains FROM the frozen ViT-B/16 embeddings -- the "
        "0.2520 arm's own representation, the one every result rides on "
        "-- with no encoder gradient anywhere; the encoder and the "
        "ladder are untouched, asserted in code and test"
    ),
    "silences_resolved": {
        "reconstructs": (
            "CLEFT CROPS (the maintainer's directive, standing), with the "
            "masked SCUT artifact as the machinery-shakedown arm FIRST: "
            "the decoder architecture is validated on public shareable "
            "data before any cohort pixel is generated"
        ),
        "purpose": (
            "WHAT-THE-EMBEDDING-KEEPS, with the registered question: do "
            "the grader-relevant regions (scar, nasal base, vermilion) "
            "reconstruct WORSE than grade-irrelevant ones (pose, tone, "
            "lighting)? The four-nulls-made-visible deliverable"
        ),
        "success": (
            "NOT a reconstruction-loss number as an end. L2 + SSIM "
            "registered as DESCRIPTIVE; the phase's finding is the "
            "REGIONAL COMPARISON; no ladder entry, so no two-condition "
            "claim arises -- DESCRIPTIVE throughout"
        ),
        "governance": (
            "cleft_reconstruction_acknowledged gate on the basal-flag "
            "precedent -- the edit, refused before the first "
            "cohort reconstruction is written; reconstructions are "
            "patient-derived and CLUSTER-ONLY without exception. "
            "ENCODING cleft crops needs no new flag: standing practice, "
            "the ladder does it"
        ),
        "ladder_entry": "NONE this phase -- recorded",
        "compute": (
            "decoder training is a pod GPU run; "
            "phase8.RETRY_LIMIT_IS_NOT_HOLDING's standing caveat applies "
            "to any long run"
        ),
    },
}


#: **[COMMITTED 2026-08-24, BEFORE ANY ARM] THE FAILURE-MODE READING,
#: WRITTEN AND BOUND** -- the reading the amendment required in advance.
FAILURE_MODE_READING_BOUND = {
    "committed": "2026-08-24, before any arm exists",
    "the_registered_warning": (
        "a reconstruction loss optimises for looking like a face, and "
        "looking like a face is not scoring like a clinician"
    ),
    "the_reading": (
        "if reconstructions look increasingly face-like while the "
        "regional comparison shows grader-relevant regions reconstructing "
        "NO BETTER (or worse) than grade-irrelevant ones, that is the "
        "registered warning REALISED -- and it is the phase's EXPECTED "
        "PRIMARY FINDING, not a failure: the embedding keeps what makes "
        "a face generic and discards what makes a repair gradable, which "
        "is the four convergent nulls made visible"
    ),
    "bound_to": (
        "the cleft reconstruction arm; the reading applies by rule when "
        "the regional comparison lands, never re-composed after the "
        "images are seen"
    ),
}


#: **[REGISTERED 2026-08-24; AGREED the same day, one amendment] THE
#: EXIT CRITERIA, numbered, written before any code. [The "agreed" field
#: read PENDING at registration; the dated agreement below replaced it --
#: a DUPLICATE key briefly carried both, masked by last-key-wins, found
#: and removed 2026-08-24.]**
PHASE_13_EXIT_CRITERIA = {
    "registered": "2026-08-24",
    "criteria": (
        "1. SCUT SHAKEDOWN FIRST: the decoder trained on the masked SCUT "
        "artifact (G1, 224) from frozen imagenet ViT-B/16 embeddings, "
        "reconstructions of HELD-OUT SCUT faces (the official 3300/2200 "
        "split's test side, never fit) rendered to SHAREABLE sheets and "
        "passed by the visual check BEFORE any cohort pixel is generated",
        "2. the cleft_reconstruction_acknowledged gate refuses before "
        "the first cohort reconstruction is written -- the edit, "
        "carried by the generator, never defaulted; every reconstruction "
        "is CLUSTER-ONLY without exception",
        "3. the REGIONAL SPLIT is registered before any cohort "
        "reconstruction is seen: grader-relevant = the anatomy middle "
        "(nose) and bottom (lips) bands -- the instrument's own "
        "nasolabial strip; grade-irrelevant = the top (eyes) band; the "
        "white pad excluded from both and reported separately, since "
        "constant padding is trivially reconstructible",
        "4. reconstruction metrics (L2 and SSIM) are DESCRIPTIVE only; "
        "no ladder entry, no two-condition claim, no ledger row for a "
        "reconstruction number",
        "5. the regional comparison is the phase's finding, computed by "
        "the registered rule with both readings committed in advance "
        "(REGIONAL_COMPARISON_READINGS), and the bound failure-mode "
        "reading (FAILURE_MODE_READING_BOUND) applied where it fires",
        "6. the encoder and the ladder are untouched BY CONSTRUCTION -- "
        "the decoder consumes cached embeddings, no encoder gradient "
        "exists anywhere, asserted in code and test",
        "7. per-epoch curves and decoder checkpoints are WRITTEN, not "
        "discarded (the computed-then-discarded lesson, four instances); "
        "suite green",
    ),
    "expected_count": 7,
    # **[AGREED 2026-08-24, the maintainer, with ONE amendment to criterion
    # 5's rule.]** The regional comparison's registered statistic is the
    # DOUBLE DIFFERENCE (DOUBLE_DIFFERENCE_RULE); the bound failure-mode
    # reading and the null direction's sentence both attach to it.
    "agreed": (
        "2026-08-24 -- with one amendment: criterion 5's "
        "statistic is the double difference (DOUBLE_DIFFERENCE_RULE)"
    ),
    "small_correction": (
        "the held-out side is 2,199, not 2,200: the official split's "
        "test side excludes CM152 (p6_pretrain configs' expect_test: "
        "2199; train.pretrain's own note). The criteria's '3300/2200' "
        "names the official split; the count that runs is 3,300 / 2,199"
    ),
}


#: **[REGISTERED 2026-08-24, BEFORE ANY RECONSTRUCTION EXISTS] THE
#: DOUBLE-DIFFERENCE RULE -- the amendment, with the defect raised at
#: supervision answered and its mitigation registered.**
#:
#: **The statistic**: per-band cleft reconstruction error, normalised by
#: the SAME band's error on the SCUT held-out faces -- same decoder, also
#: never fit -- then grader-relevant (middle + bottom) compared against
#: grade-irrelevant (top). The SCUT profile is the difficulty baseline,
#: and it exists by construction. Pad excluded and reported separately.
#:
#: **THE DEFECT, NAMED BEFORE REGISTERING** (the question asked): SCUT
#: masked faces and cleft crops differ in a way that CAN break band
#: comparability, and the largest identifiable channel is **masked-white
#: content inside the bands**. A masked SCUT face carries removed hair
#: and background as baked white; a band containing much baked white has
#: an artificially LOW error (white is trivially reconstructible), which
#: DEFLATES that band's baseline and INFLATES the cleft/SCUT ratio for
#: it. The top band is the likeliest carrier (hair-line), and top is the
#: grade-IRRELEVANT side -- so the bias direction is AGAINST the
#: failure-mode reading: it would make grader-relevant regions look
#: relatively better-reconstructed than they are.
#:
#: **THE MITIGATION, REGISTERED INTO THE RULE, not patched later**: every
#: band error, BOTH families, is computed over CONTENT PIXELS ONLY --
#: pixels below the white level (>= 250/channel, the basal.py precedent)
#: are excluded exactly as the pad is. The double difference then
#: compares reconstruction of actual content against reconstruction of
#: actual content, and the baked-white channel is closed by construction.
#:
#: **THE MEASUREMENT THAT SETTLES WHAT REMAINS**, computed in stop 1's
#: run before any cohort reconstruction: a per-band COMPARABILITY TABLE
#: for both families -- content-pixel fraction and content-pixel variance
#: per band, cleft staged (237) beside SCUT held-out (2,199). Reading
#: registered with it: if any band's content-pixel variance differs
#: between families by MORE THAN 2x, a band-comparability caveat ATTACHES
#: to the double difference for that band -- a report-with-reading, never
#: a gate.
#:
#: **THE RESIDUAL LIMITATION, CARRIED NOT HIDDEN**: the bands are
#: POSITIONAL, not anatomical. On the cleft crop the thirds carry the
#: anatomy vocabulary (top = eyes, middle = nose, bottom = lips); on a
#: full masked SCUT face the same positions hold different anatomy. The
#: normalisation is therefore "same decoder, same position, faces it was
#: trained on" -- a positional difficulty baseline, not an anatomical
#: one. A landmark-matched SCUT banding would need a landmark pipeline
#: this project does not have, and is declined as machinery.
#:
#: **The band rule, stated exactly** (cross-family applicable, one
#: implementation): bands are CONTENT-BOX HORIZONTAL THIRDS -- y in
#: [0, 1/3) / [1/3, 2/3) / [2/3, 1] of each image's own content box --
#: named by the anatomy vocabulary on the cleft crop. The anatomy PATCH
#: boxes themselves exist only through the trapezium mapping, which SCUT
#: faces do not have; the thirds are the positional form both families
#: admit. Flagged to the maintainer as the one place the rule's letter departs
#: from criterion 3's "anatomy bands" wording while keeping its intent.
DOUBLE_DIFFERENCE_RULE = {
    "registered": "2026-08-24, before any reconstruction exists",
    "statistic": (
        "per-band cleft reconstruction error normalised by the SAME "
        "band's error on SCUT held-out faces (same decoder, also never "
        "fit); grader-relevant (middle+bottom) against grade-irrelevant "
        "(top); pad excluded and reported separately"
    ),
    "why": (
        "raw regional error confounds embedding retention with intrinsic "
        "band difficulty; the SCUT profile is the difficulty baseline "
        "and exists by construction"
    ),
    "defect_answered": (
        "baked masked-white inside SCUT bands deflates that band's "
        "baseline (white is trivially reconstructible) and inflates the "
        "cleft/SCUT ratio -- likeliest in the TOP band, so the bias runs "
        "AGAINST the failure-mode reading"
    ),
    "mitigation_registered": (
        "band errors, BOTH families, computed over CONTENT PIXELS ONLY "
        "(white >= 250/channel excluded like the pad, the basal.py "
        "precedent) -- the baked-white channel closed by construction, "
        "in the rule, not patched later"
    ),
    "comparability_measurement": (
        "stop 1 reports the per-band content-pixel fraction and variance "
        "for BOTH families (cleft staged 237 beside SCUT held-out "
        "2,199); READING: any band whose content variance differs by "
        "more than 2x between families gets a band-comparability caveat "
        "ATTACHED to the double difference -- report, never gate"
    ),
    "residual_limitation": (
        "the bands are POSITIONAL, not anatomical: on a full SCUT face "
        "the cleft crop's thirds hold different anatomy. The baseline is "
        "'same decoder, same position, faces it was trained on'. A "
        "landmark-matched banding is declined as machinery"
    ),
    "band_rule": (
        "content-box horizontal THIRDS, named by the anatomy vocabulary "
        "on the cleft crop (top=eyes, middle=nose, bottom=lips) -- the "
        "positional form both families admit, since the anatomy patch "
        "boxes exist only through the trapezium mapping SCUT lacks. "
        "Flagged as the one departure from criterion 3's letter, keeping "
        "its intent"
    ),
    "readings_attach_here": (
        "the bound failure-mode reading (FAILURE_MODE_READING_BOUND) and "
        "the null direction's sentence (REGIONAL_COMPARISON_READINGS) "
        "both attach to the DOUBLE DIFFERENCE"
    ),
}


#: **[REGISTERED 2026-08-24, BOTH WAYS BEFORE ANY IMAGE] THE REGIONAL
#: COMPARISON'S READINGS.**
REGIONAL_COMPARISON_READINGS = {
    "registered": "2026-08-24, before any reconstruction exists",
    "the_measure": (
        "per-region reconstruction error, normalised by the image's own "
        "whole-content error so lighting and pose difficulty divide out; "
        "regions mapped per image through the shipped anatomy patch set "
        "and content_box machinery -- nothing new placed by hand"
    ),
    "reading_if_relevant_worse": (
        "grader-relevant regions reconstruct WORSE -> the embedding "
        "keeps the generic face and discards the gradable detail: the "
        "failure-mode reading fires, the four nulls are made visible, "
        "and the write-up gains its mechanism illustration"
    ),
    "reading_if_no_difference_or_better": (
        "grader-relevant regions reconstruct AS WELL or BETTER -> the "
        "embedding RETAINS the gradable regions' appearance, and the "
        "r~0.3 plateau is not an encoding loss at that granularity -- "
        "which sharpens the puzzle rather than solving it, and says the "
        "bottleneck is downstream of appearance retention"
    ),
    "descriptive_throughout": (
        "either way the numbers are DESCRIPTIVE: region-level "
        "reconstruction error has no registered interval machinery here "
        "and enters no ledger row"
    ),
}


#: **[PROPOSED 2026-08-24, NOT PICKED] THE DECODER ARCHITECTURE.**
DECODER_ARCHITECTURE_PROPOSED = {
    "proposed": "2026-08-24, for the maintainer to accept or amend",
    # [CORRECTED 2026-08-24, by the guard at first construction] The
    # "~3-5M parameters" below was a proposal-time ESTIMATE; the shape
    # as ordinarily built constructs to 10,628,771 (the seed Linear
    # alone is 9,646,336), verified by pure arithmetic that now lives in
    # decoder.expected_parameters(). PARAMETER_BAND_MISESTIMATED carries
    # the diagnosis and the two resolutions; the wording below stays as
    # what was believed at proposal time.
    "corrected": (
        "2026-08-24: '~3-5M' was an estimate; the shape constructs to "
        "10,628,771 (PARAMETER_BAND_MISESTIMATED). The resolution -- "
        "shrink or amend -- is the maintainer's"
    ),
    "resolved": (
        "2026-08-24, same day, the maintainer: (a) shrink -- the 7x7x96 "
        "shape, 3,862,339 parameters, inside the registered band; "
        "re-registered by exact equality with "
        "decoder.expected_parameters(), no band, no estimate "
        "(PARAMETER_BAND_MISESTIMATED['picked'])"
    ),
    "shape": (
        "a bottleneck deconvolutional decoder from the 768-d pooled "
        "embedding: Linear 768 -> 7x7x256, then five upsample+conv "
        "stages 7->14->28->56->112->224 to 224x224x3, ~3-5M parameters"
    ),
    "why_this_shape": (
        "it inverts exactly the pipeline the embeddings came from (224 "
        "G1 staged -> ViT-B/16 -> 768-d pooled), so 'what the embedding "
        "keeps' is measured at the representation the ladder actually "
        "used; upsample+conv over ConvTranspose to avoid checkerboard "
        "artifacts that would contaminate the regional comparison"
    ),
    "trained_on": (
        "masked SCUT G1 224 ONLY (the shakedown arm) -- the decoder "
        "never sees a cleft pixel in training, so applying it to cleft "
        "embeddings is the strongest instrument reading: retention "
        "measured by a decoder with no opportunity to memorise the "
        "cohort. A cleft-refit decoder (5-fold OOF) is a REGISTERED "
        "EXTENSION, not built unless the SCUT-trained instrument proves "
        "domain-gapped beyond use"
    ),
    "target_masked_not_original": (
        "the MASKED artifact is the target, explicitly: the cleft crops "
        "are background-stripped white-padded clinical images, and "
        "masked SCUT is their distribution-matched sibling -- that was "
        "its purpose. Reconstructing unmasked originals would spend "
        "decoder capacity on hair and background the cleft application "
        "never sees"
    ),
    "resolution_224_g1": (
        "224 G1, matching the embeddings' own input: the ladder set was "
        "extracted from 224 G1 staged tensors, and a 512/768 target "
        "would ask the decoder to hallucinate detail the encoder never "
        "saw"
    ),
    "loss": (
        "plain L2 for training (descriptive metric anyway); SSIM "
        "reported beside it, not optimised -- optimising a perceptual "
        "metric would bake 'looks face-like' into the objective, which "
        "is the registered failure mode by construction"
    ),
    "perceptual_metric_choice": (
        "SSIM, not LPIPS: SSIM is dependency-light and deterministic; "
        "LPIPS drags in a pretrained network with its own weights, "
        "determinism story and download -- named and declined"
    ),
    "what_needs_new_code": (
        "the decoder's training loop is NEW code outside both the "
        "frozen harness (whose gates are scalar-prediction shaped) and "
        "the pretraining pipeline (which trains encoders) -- small, "
        "checkpointed per epoch, seeded. NOTHING needs task_pretrain; "
        "the SCUT embedding extraction reuses extract_features exactly "
        "as the basal set did"
    ),
}


#: **[PLANNED 2026-08-24] THE SCUT SHAKEDOWN.**
SCUT_SHAKEDOWN_PLAN = {
    "planned": "2026-08-24",
    "steps": (
        "extract frozen imagenet ViT-B/16 embeddings over the masked "
        "SCUT artifact at G1 224 (the basal set's exact machinery; "
        "SHAREABLE -- SCUT is public, phase8's governance confirmation)",
        "train the decoder on the official split's 3,300 fit side, "
        "inner-val early stopping, per-epoch curves and checkpoints "
        "written",
        "reconstruct HELD-OUT faces from the 2,200 test side -- never "
        "fit, so face-likeness is generalisation, not memorisation -- "
        "and render SHAREABLE review sheets",
        "the visual check passes the sheets BEFORE any cohort pixel is "
        "generated -- the standing rule: sheets are reviewed by eye or "
        "not at all",
    ),
    "what_it_validates": (
        "the architecture, the training loop, the metric plumbing and "
        "the sheet rendering -- on public data, so every defect of the "
        "machinery is found where no governance question exists"
    ),
}


#: **[REGISTERED 2026-08-24, BUILT AT ITS STOP] THE CLEFT RECONSTRUCTION
#: GATE** -- the basal-flag precedent, applied to generated images.
CLEFT_RECONSTRUCTION_GATE = {
    "registered": "2026-08-24",
    "flag": "cleft_reconstruction_acknowledged",
    "why": (
        "a reconstructed cleft face is a GENERATED PATIENT-DERIVED "
        "image -- a new kind of output the REC crop conditions never "
        "contemplated, and that judgement belongs to the maintainer, not code"
    ),
    "mechanism": (
        "the basal-flag precedent exactly: a required boolean with NO "
        "default, written false by the generator, carried true only "
        "from the maintainer's own edit, refused by the task BEFORE the first "
        "cohort reconstruction is written"
    ),
    "tier": (
        "every reconstruction is patient-derived and CLUSTER-ONLY "
        "without exception; sheets of cohort reconstructions are "
        "CLUSTER-ONLY and reviewed on the cluster"
    ),
    "encoding_needs_no_flag": (
        "ENCODING cleft crops is standing practice -- the ladder does "
        "it; the gate attaches at the first GENERATED cohort pixel, "
        "answered  on the question raised at the restatement"
    ),
}


#: **[PROPOSED 2026-08-24] THE STOP STRUCTURE -- three stops, review
#: after each, the phase-12 discipline.**
PHASE_13_STOPS = {
    "proposed": "2026-08-24",
    "stops": {
        "1": (
            "SCUT shakedown: embedding extraction + decoder + training "
            "+ held-out reconstruction sheets -> the visual check. The "
            "regional split is REGISTERED here, before any cohort "
            "reconstruction exists to be seen"
        ),
        "2": (
            "the gate + the cohort reconstructions (CLUSTER-ONLY "
            "sheets) -> the visual check on the cluster"
        ),
        "3": (
            "the regional comparison + descriptive metrics + the "
            "phase's closing, with the bound reading applied where it "
            "fires"
        ),
        "rule": "stop after each for review",
    },
}


#: **[BUILT 2026-08-24, STOP 1 OF 3 -- NOT LAUNCHED] THE SHAKEDOWN.**
STOP_1_BUILT = {
    "built": "2026-08-24, stop 1 of 3, NOT launched",
    "architecture_accepted": (
        "as proposed: bottleneck upsample+conv, ~3-5M parameters "
        "(asserted at construction), SCUT-only training, masked target, "
        "224 G1, L2 trains / SSIM reports, LPIPS declined, per-epoch "
        "checkpoints -- cleft.decoder.build "
        "[corrected 2026-08-24: '~3-5M' was the mis-estimate; the built "
        "shape is 7x7x96 at EXACTLY 3,862,339 parameters, the maintainer's "
        "pick on PARAMETER_BAND_MISESTIMATED]"
    ),
    "module": (
        "cleft.decoder -- the model, SSIM, the content-pixel mask, the "
        "band machinery and the double difference live ONCE, so the "
        "shakedown task and the later cohort task cannot drift apart"
    ),
    "tasks": {
        "extract_scut_embeddings": (
            "pretrain.load_masked_features (stem-aligned) + "
            "extract.extract_features; the artifact is "
            "decoder.save_scut_embeddings' own values+stems format, "
            "which owns existence and creation (the guard-after-mkdir "
            "lesson applied at build time)"
        ),
        "train_scut_decoder": (
            "fixed 30-epoch budget, best checkpoint by inner L2, no "
            "early stopping (the pretraining discipline); every epoch's "
            "checkpoint and curve row written; held-out reconstruction "
            "of ALL 2,199 with metrics over all; the SCUT band baseline "
            "(the double difference's denominator) and the comparability "
            "table written; NO cohort pixel generated -- staged_v1 feeds "
            "content statistics only"
        ),
    },
    "sheet_policy": (
        "original beside reconstruction, WORST 24 by L2 first, then a "
        "seeded sample of 96 -- ten sheets, SHAREABLE (SCUT public), 12 "
        "pairs per sheet; metrics computed over ALL held-out faces, "
        "never the sheet sample"
    ),
    "ssim_cost_control": (
        "the per-epoch inner-val SSIM is computed on a seeded fixed "
        "subsample of 64 (declared in the config); the FINAL SSIM is "
        "over all 2,199 -- the curve's SSIM is a trend line, the "
        "reported number is complete"
    ),
    "configs": (
        "p13_extract_scut.yaml (both inputs carried verbatim from the "
        "shipped pretraining config), p13_scut_decoder.yaml (embedding "
        "artifact a documented placeholder until extraction runs -- the "
        "standing two-pass flow)"
    ),
    "next": (
        "the visual check on the sheets, then stop 2: the gate and the "
        "cohort reconstructions"
    ),
}


#: **[DEFECT 2026-08-24, DIAGNOSED -- RESOLUTION IS OWED] THE
#: REGISTERED PARAMETER BAND WAS A MIS-ESTIMATE, AND THE GUARD REFUSED
#: THE SHAPE IT WAS REGISTERED BESIDE.**
#:
#: ``p13-scut-decoder`` failed at construction, both attempts (+2 on
#: ``phase8.RETRY_LIMIT_IS_NOT_HOLDING``): ``DecoderError: 10,628,771
#: parameters is outside the registered 3-5M band``.
#:
#: **BOTH OF THE OPERATOR'S CLAIMS VERIFIED, by pure arithmetic:**
#:
#: 1. **The registered shape IS ~10.6M as ordinarily built.** Counted
#:    from the layer dims, no torch: the seed Linear
#:    ``768 -> 7x7x256`` alone is **9,646,336** (the maintainer's ~9.6M);
#:    the six convs add 982,435; total **10,628,771 -- exactly the
#:    cluster's figure**. The ~3-5M in the proposal was a MIS-ESTIMATE
#:    at proposal time, not a build error. The shape and the number were
#:    registered together and disagreed from the start.
#: 2. **The construction test skips in the torch-less venv**
#:    (``importorskip("torch")``), so the cluster's construction was the
#:    FIRST real one. The guard did exactly what it was built for --
#:    refuse a registration-vs-build drift -- it just caught the
#:    registration's own number rather than a later drift.
#:
#: **THE LESSON, and it is the offline-idealisation lesson in parameter
#: arithmetic**: an ESTIMATED count is not a CONSTRUCTED count. The fix
#: that catches this class on the laptop is built:
#: ``decoder.expected_parameters()`` computes the constructed count by
#: pure integer arithmetic over the layer dims, torch-free, and the venv
#: test pins the registration to it -- a mismatch now fails before any
#: launch.
#:
#: **THE RESOLUTIONS, PROPOSED NOT PICKED:**
#:
#: **(a) SHRINK to meet the registration** -- the smaller decoder is the
#: more conservative instrument: less capacity to hallucinate detail the
#: embedding does not carry, which SHARPENS the what-the-embedding-keeps
#: reading. Two concrete shapes, counts CONSTRUCTED by the arithmetic,
#: not estimated:
#:
#:     7x7x96,  stages (96, 96, 64, 32, 16)   ->  3,862,339  (in 3-5M)
#:     7x7x128, stages (128, 128, 64, 32, 16) ->  5,215,651  (just over)
#:
#: [Corrected the same hour: this record's FIRST draft carried
#: hand-estimated candidate counts (3,779,299 / 5,068,067) that dropped
#: one conv each -- caught by the new venv test against
#: ``expected_parameters()`` before the record was ever cited. The
#: lesson demonstrated itself on its own record.]
#:
#: The 96-start sits inside the registered band and is the conservative
#: extreme. Under (a) the band is RE-REGISTERED FROM THE CONSTRUCTED
#: COUNT -- exact equality with ``expected_parameters()``, no band, no
#: estimate ever again. Nothing is lost: no result exists yet, the
#: shakedown had not trained a step.
#:
#: **(b) AMEND the registration to the measured count** -- 10,628,771,
#: with a dated correction naming the mis-estimate, on the argument that
#: the SHAPE was the registered thing and the number was its
#: description. Cost: the larger decoder has more capacity to
#: hallucinate, which BLUNTS the instrument reading (a face-like
#: reconstruction could owe more to the decoder than the embedding) --
#: the exact axis the phase measures on.
#:
#: **[PICKED 2026-08-24]: (a), the 7x7x96 shape --
#: 3,862,339 parameters, inside the registered band.** Reason as this
#: record states it: the smaller decoder is the more conservative
#: instrument for a what-the-embedding-keeps reading; (b)'s added
#: capacity blunts the instrument on the measured axis. The
#: registration is now EXACT EQUALITY with the constructed count --
#: ``decoder.REGISTERED_PARAMETERS = expected_parameters() =
#: 3,862,339``, no band, no estimate ever again. The relaunch is the maintainer's
#:.
PARAMETER_BAND_MISESTIMATED = {
    "defect": "2026-08-24, diagnosed -- resolution is the maintainer's",
    "picked": (
        "2026-08-24, the maintainer: (a), the 7x7x96 shape -- 3,862,339 "
        "parameters, inside the registered band. The smaller decoder is "
        "the more conservative instrument for a what-the-embedding-"
        "keeps reading; (b)'s added capacity blunts the instrument on "
        "the measured axis. Re-registered by EXACT EQUALITY with "
        "expected_parameters() (decoder.REGISTERED_PARAMETERS) -- no "
        "band, no estimate. The relaunch is the maintainer's"
    ),
    "run": "p13-scut-decoder, both attempts, +2 on the retry record",
    "verified_1": (
        "the registered shape IS ~10.6M as built: the seed Linear alone "
        "is 9,646,336, the convs add 982,435, total 10,628,771 -- "
        "exactly the cluster's figure, by pure arithmetic. The ~3-5M was "
        "a MIS-ESTIMATE at proposal time; the shape and its number "
        "disagreed from the start"
    ),
    "verified_2": (
        "the construction test skips in the torch-less venv "
        "(importorskip), so the cluster's construction was the FIRST "
        "real one. The guard did what it was built for; it caught the "
        "registration's own number"
    ),
    "lesson": (
        "an ESTIMATED count is not a CONSTRUCTED count -- the "
        "offline-idealisation lesson in parameter arithmetic. Fixed on "
        "the laptop: decoder.expected_parameters() computes the "
        "constructed count torch-free, and the venv test pins the "
        "registration to it"
    ),
    "a_shrink": {
        "argument": (
            "the smaller decoder is the MORE CONSERVATIVE instrument: "
            "less capacity to hallucinate detail the embedding does not "
            "carry, sharpening the what-the-embedding-keeps reading"
        ),
        "shapes_constructed_not_estimated": {
            "seed_96": "7x7x96, stages (96,96,64,32,16) -> 3,862,339",
            "seed_128": "7x7x128, stages (128,128,64,32,16) -> 5,215,651",
        },
        "first_draft_corrected": (
            "the same hour: the first draft hand-estimated these two "
            "counts (3,779,299 / 5,068,067), dropping one conv each -- "
            "caught by the venv test against expected_parameters() "
            "before the record was cited. The lesson demonstrated "
            "itself on its own record"
        ),
        "re_registration_rule": (
            "the band is re-registered FROM the constructed count: exact "
            "equality with expected_parameters(), no band, no estimate "
            "ever again"
        ),
        "nothing_lost": "no result exists; the shakedown never trained a step",
    },
    "b_amend": {
        "argument": (
            "the SHAPE was the registered thing and the number was its "
            "description -- amend to 10,628,771 with a dated correction"
        ),
        "cost": (
            "more capacity to hallucinate BLUNTS the instrument reading "
            "-- a face-like reconstruction could owe more to the decoder "
            "than the embedding, the exact axis the phase measures on"
        ),
    },
}


#: **[REVIEWED 2026-08-24 -- PASS] THE SHAKEDOWN SHEETS**, the maintainer's
#: eye with a second reading joined (SCUT is public; the sheets are SHAREABLE).
STOP_1_REVIEWED = {
    "reviewed": "2026-08-24, reviewed -- PASS",
    "machinery": (
        "CLEAN: alignment correct, no artifacts, the trapezium geometry "
        "learned, checkerboard ABSENT -- the upsample+conv choice did "
        "what it was picked for"
    ),
    "reconstructions": (
        "the EXPECTED smoothed-global-structure signature of a 768-d "
        "bottleneck; the worst-24 are HARD FACES -- extreme pose, "
        "expression, and faces far from SCUT's demographic centre "
        "(males, non-Asian) -- not machinery defects"
    ),
    "checkpoint_pick": (
        "2026-08-24, the maintainer: the cohort arm runs through the "
        "EPOCH-20 checkpoint of the shakedown run "
        "(checkpoints/epoch_20.pt, decoder.checkpoint_name(20))"
    ),
    "criterion_1_precondition": (
        "met: the eye passed the sheets BEFORE any cohort pixel "
        "exists; stop 2 may build"
    ),
}


#: **[REGISTERED 2026-08-24, BEFORE ANY COHORT PIXEL] THE SCUT-NORMAL
#: PRIOR** -- the one observation the shakedown review sends forward,
#: recorded at the instruction for the cohort reading to carry.
SCUT_NORMAL_PRIOR = {
    "registered": (
        "2026-08-24, from the shakedown review, before any cohort "
        "pixel exists"
    ),
    "observation": (
        "the decoder's prior is SCUT-NORMAL: trained only on SCUT "
        "faces, it reconstructs structurally-unusual features TOWARD "
        "THE AVERAGE -- the worst-24 already show it on faces far from "
        "SCUT's demographic centre"
    ),
    "consequence_for_the_cohort": (
        "cleft-affected regions are structurally unusual by "
        "definition, so elevated error there can reflect the prior, "
        "not only what the embedding dropped"
    ),
    "mitigation": (
        "the double difference's BAND-RELATIVE design "
        "(DOUBLE_DIFFERENCE_RULE): each band is normalised by its own "
        "SCUT baseline and grader-relevant is read AGAINST "
        "grade-irrelevant, and the reading MUST SAY SO -- whichever "
        "way stop 3 lands, the sentence names this prior"
    ),
    "travels_with": (
        "stop 3's regional comparison, by name, and the per-patient "
        "band figures stop 2 banks (cleft_band_errors.json)"
    ),
    # [MEASURED 2026-08-24, the post-closing probe] The prior's
    # mechanism, upgraded from inference to measurement.
    "measured_mechanism": (
        "2026-08-24 (ASYMMETRY_ENCODING_PROBE_BANKED): the decoder "
        "pulls toward its SCUT-normal prior EVEN WHERE THE "
        "REPRESENTATION CARRIES THE PATIENT'S OWN GEOMETRY -- the "
        "embedding predicts asym_orig at r~0.49 while the rendered "
        "reconstruction correlates at 0.0082. What the worst-24 "
        "suggested by eye is now a measured property of the decoder"
    ),
}


#: **[BUILT 2026-08-24, STOP 2 OF 3 -- NOT LAUNCHED] THE COHORT ARM.**
STOP_2_BUILT = {
    "built": "2026-08-24, stop 2 of 3, NOT launched",
    "gate_wired": (
        "cleft_reconstruction_acknowledged: required boolean, NO "
        "default; the generator writes false and CARRIES the maintainer's "
        "true (the basal three-state carry, --check clean either way); "
        "task_reconstruct_cohort refuses FIRST, before any read"
    ),
    "instrument": (
        "the shakedown decoder through the maintainer's epoch-20 checkpoint "
        "(STOP_1_REVIEWED); decoder.checkpoint_name is the ONE "
        "filename implementation the writer and the reader share; the "
        "shakedown's own inner-L2 selection is logged beside the pick "
        "-- reported, never gated"
    ),
    "cohort": (
        "all 237 manifest patients: the 0.2520 arm's frozen frontal "
        "imagenet ViT-B/16 G1 set (embeddings.load, row order asserted "
        "against cleft_v1; init/geometry/width asserted -- the "
        "instrument reads its OWN representation and no other); "
        "references are the staged_v1 crops the encoder saw, content "
        "boxes from geometry.csv by patient_id"
    ),
    "outputs": (
        "CLUSTER-ONLY sheets of ALL 237 original-beside-reconstruction "
        "pairs, worst L2 first; per-patient l2/ssim/band figures "
        "banked as numbers (cleft_band_errors.json, the double "
        "difference's numerator side). The raw reconstruction tensor "
        "is DELIBERATELY NOT PERSISTED: generated cohort pixels exist "
        "in the sheets alone -- the smallest generated-patient-image "
        "surface that serves the eye and the figures"
    ),
    "boundary": (
        "NO double difference here: the regional comparison is stop "
        "3's, composed from this run's band figures and the "
        "shakedown's scut_band_baseline.json, with the bound readings "
        "applied where they fire and SCUT_NORMAL_PRIOR travelling by "
        "name"
    ),
    "review": (
        "the sheets go to the visual check ON THE CLUSTER before stop 3"
    ),
}


#: **[REVIEWED 2026-08-24 -- PASS ON FIDELITY] THE COHORT SHEETS.**
#:
#: **Provenance is part of this record.** The fidelity verdict is a
#: verdict; the clinical-detail observations are ONE REVIEWER'S
#: IMPRESSIONS FROM LOOKING, not counts, and every figure below is
#: tagged so no later turn can promote them by citing them.
STOP_2_REVIEWED = {
    "reviewed": "2026-08-24, the visual check, all sheets -- PASS on fidelity",
    "fidelity_MEASURED_BY_EYE": (
        "tone, pose, lighting and trapezium shape track pair by pair; "
        "ZERO machinery artifacts. A small number of pose misses -- the "
        "reviewer's estimate roughly 3 across the sheets, AN IMPRESSION, "
        "NOT A "
        "COUNT"
    ),
    "clinical_detail_EYE_IMPRESSION": (
        "**EYE-IMPRESSION, NOT MEASUREMENT**: detail barely survives. "
        "Severe cases show blurred-but-visible nose/lip asymmetry; mild "
        "cases and good repairs reconstruct as SCUT-NORMAL"
    ),
    "worst_12_EYE_IMPRESSION": (
        "**EYE-IMPRESSION, NOT A COUNT**: on the worst-12 sheet, ~6 of "
        "12 showed visible asymmetry and ~4 of those a scar-like trace. "
        "One reviewer, uncounted formally, no annotation exists -- "
        "these numbers may be QUOTED AS IMPRESSIONS and never as "
        "measurements"
    ),
    "tagging_rule": (
        "every figure in this record except the machinery verdict is "
        "EYE-IMPRESSION; a later turn citing them must carry the tag"
    ),
}


#: **[HYPOTHESIS 2026-08-24 -- NOT A FINDING] COARSE SEVERITY CODING.**
#: The interpretation the eye review suggests, registered as something
#: the numbers MUST TEST before it may be believed.
COARSE_SEVERITY_HYPOTHESIS = {
    "registered": (
        "2026-08-24, as a HYPOTHESIS THE NUMBERS MUST TEST -- not a "
        "finding, not claimable, and it earns no status from having "
        "been thought of first"
    ),
    "the_hypothesis": (
        "the embedding encodes severity COARSELY -- the extremes are "
        "separable -- but NOT FINELY: the mild-moderate middle, where "
        "199 of 237 patients sit, collapses toward SCUT-normal"
    ),
    "what_it_would_explain": (
        "both the ~0.25 plateau and the four convergent neighbourhood "
        "nulls -- IF IT SURVIVES MEASUREMENT. A hypothesis that "
        "explains everything already measured is cheap; the tests in "
        "STOP_3_REGISTERED are what would make it cost something"
    ),
    "tested_by": (
        "stop 3's checks (b), (c) and (d): band error vs grade, "
        "asymmetry retention by grade, and grade decodability from the "
        "reconstructions"
    ),
    "how_it_could_fail": (
        "flat band error against grade, retention ratios that do not "
        "separate severe from mild, or a probe PCC near A's 0.2505 "
        "(which would say the grade-relevant content SURVIVES the "
        "chain, and the collapse is not where the eye placed it)"
    ),
}


#: **[REGISTERED 2026-08-24, READINGS COMMITTED BEFORE ANY NUMBER] STOP
#: 3'S SCOPE**: the registered composition (a) plus three quantitative
#: checks (b)(c)(d) that TEST the eye's impressions, and one thing named
#: as untestable (e).
STOP_3_REGISTERED = {
    "registered": (
        "2026-08-24, every reading committed BEFORE any number exists"
    ),
    "a_double_difference": {
        "what": (
            "the registered statistic, unchanged: cleft band errors "
            "(cleft_band_errors.json) against the shakedown's "
            "scut_band_baseline.json, grader-relevant (nose+lips) vs "
            "grade-irrelevant (eyes), CONTENT PIXELS ONLY, "
            "comparability caveats carried"
        ),
        "readings": (
            "both committed readings stand verbatim "
            "(REGIONAL_COMPARISON_READINGS); the SCUT_NORMAL_PRIOR "
            "sentence is applied BY NAME whichever way it lands"
        ),
    },
    "b_band_error_vs_grade": {
        "what": (
            "per-patient nose+lips band error against the median "
            "grade, cleft side: correlation and per-grade means"
        ),
        "reading_if_rises": (
            "severe cases reconstruct WORSE in grader-relevant bands "
            "-- consistent with COARSE_SEVERITY_HYPOTHESIS, which is "
            "consistency, not confirmation"
        ),
        "reading_if_flat": (
            "the eye's impression was SAMPLING -- recorded as such, "
            "and the hypothesis loses its first support"
        ),
        "registered_limit": (
            "NECESSARY, NOT SUFFICIENT: high error on severe cases "
            "cannot distinguish smoothed-away ASYMMETRY from other "
            "lost detail -- which is precisely why (c) exists. This "
            "limit is registered BEFORE the number so it cannot be "
            "forgotten if the number is friendly"
        ),
        "status": "DESCRIPTIVE, report-never-gate",
    },
    "c_asymmetry_retention": {
        "what": (
            "the DIRECT test of the eye claim: Phase 4's mirror "
            "difference (decoder.band_asymmetry -- content-box "
            "midline, symmetric content mask, nose+lips positional "
            "bands) on original and reconstruction per patient"
        ),
        "statistics": (
            "1) r(asym_orig, asym_recon) over 237 -- does the decoder "
            "preserve asymmetry ORDERING; 2) retention ratio "
            "asym_recon/asym_orig BY GRADE -- the eye's exact claim, "
            "severe retains and mild normalises; 3) r(asym_orig, "
            "grade) as the SANITY ANCHOR beside them"
        ),
        "anchor": (
            "Phase 4's mirror index sits at r~0.158 (mirror."
            "MEASURED_BASELINE) -- asymmetry itself predicts grade "
            "WEAKLY, and reporting the anchor beside the retention "
            "figures is what stops anyone over-reading them "
            "[CORRECTED 2026-08-24: this juxtaposition invited a "
            "same-quantity comparison that must not be made -- the "
            "0.158 is a trained 22-feature ridge CV fit at G2 against "
            "the MEAN label, a DIFFERENT QUANTITY from (c)'s raw "
            "single-scalar r against the MEDIAN grade on five axes "
            "(ANCHOR_QUANTITIES_DIFFER). (c)'s anchor is its own "
            "measured number]"
        ),
        "blur_caveat_and_its_answer": (
            "smoothing shrinks |left-right| EVERYWHERE, so raw "
            "retention sits below 1 for every patient and an absolute "
            "ratio means nothing. The surviving comparison is "
            "RELATIVE: severe-vs-mild against each other, and against "
            "the SCUT held-out reconstructions' own asymmetry floor as "
            "the fully-normalised reference -- the double-difference "
            "pattern again. The floor is computed from the shakedown "
            "side IDENTICALLY, same function, same mask rule "
            "[CORRECTED 2026-08-24, REFUTED AS MEASURED: the "
            "below-1-for-everyone prediction was wrong -- mild grades "
            "retain 1.01-1.15, because the decoder ADDS generic "
            "asymmetry of its own, so where the original has little "
            "the ratio exceeds 1 "
            "(STOP_3_BANKED['c_asymmetry_retention']). The RELATIVE "
            "design survives its own caveat's refutation: "
            "severe-vs-mild and the SCUT floor remain the comparison]"
        ),
        "pixels": (
            "reconstructions are regenerated IN MEMORY from the "
            "checkpoint and embeddings; nothing is written to disk -- "
            "the no-persist boundary holds, and the gate check rides "
            "with the regeneration"
        ),
        "status": "DESCRIPTIVE, report-never-gate",
    },
    "d_grade_decodability": {
        "what": (
            "the END-TO-END CEILING test: re-embed the 237 "
            "reconstructions through the SAME frozen ViT-B/16 and run "
            "the standard linear probe -- the 0.2520 arm's recipe "
            "BYTE-IDENTICAL, five seeds, cleft_v1 folds"
        ),
        "reading_if_near_a": (
            "probe PCC near A's 0.2505 -> the decode PRESERVES the "
            "grade-relevant content the embedding carries, and the "
            "collapse the eye saw is not where it looked"
        ),
        "reading_if_below": (
            "the gap MEASURES the grade signal destroyed by the "
            "bottleneck-plus-decoder chain -- quantified, which is "
            "more than the eye could give"
        ),
        "status": (
            "DESCRIPTIVE: no ladder entry, NO PAIRED CLAIM. A's 0.2505 "
            "is stated as CONTEXT, not contested -- the comparator is "
            "not a claim and no §4.3 machinery is invoked"
        ),
        "governance": (
            "reconstructed-image embeddings are PATIENT-DERIVED "
            "artifacts: CLUSTER-ONLY like everything else, and the "
            "extraction task carries the SAME "
            "cleft_reconstruction_acknowledged check because it "
            "regenerates cohort pixels in memory"
        ),
    },
    "e_scar_trace_not_testable": {
        "what": "scar-trace survival, from the worst-12 impression",
        "ruling": (
            "recorded as NOT QUANTITATIVELY TESTABLE: no annotations "
            "exist, and a scar detector would be a NEW UNVALIDATED "
            "INSTRUMENT whose errors nobody could bound. It stays an "
            "EYE IMPRESSION, tagged (STOP_2_REVIEWED), and the "
            "write-up may say the eye saw traces and that nothing "
            "measured them"
        ),
    },
    "compute_placement": {
        "pod_arithmetic": (
            "(a) and (b) are PURE ARITHMETIC on banked artifacts -- "
            "two JSON files and the score sheet, no pixels, no model"
        ),
        "gpu_jobs": (
            "(c) and (d) regenerate reconstructions; (d) also extracts "
            "and probes -- likely jobs, and (d) is two runs: the "
            "extraction, then the probe on the set it writes"
        ),
    },
}


#: **[BANKED 2026-08-24] (a) THE DOUBLE DIFFERENCE, WITH THE LARGER FACT
#: STATED FIRST.** +0.0775 is real and small; the everywhere-gap it rides
#: on is ~3x and was nearly reported as a footnote.
EVERYWHERE_GAP_IS_THE_LARGER_FACT = {
    "banked": "2026-08-24, run p13-regional (first pass, (a) only)",
    "measured": (
        "double difference +0.0775; grader-relevant mean ratio 3.2570, "
        "grade-irrelevant 3.1794; no comparability caveats attached"
    ),
    "must_be_stated_first": (
        "THE LARGER FACT IS THE EVERYWHERE-GAP: both ratios sit near "
        "3.2, so the cohort reconstructs about THREE TIMES WORSE than "
        "held-out SCUT in EVERY band. The grader-relevant deficit is "
        "+2.4% RELATIVE to the irrelevant band, riding on top of that. "
        "Quoting +0.0775 alone would foreground a ~2% effect while "
        "omitting the ~200% one -- so the everywhere-gap is stated "
        "first, always"
    ),
    "what_the_everywhere_gap_means": (
        "the decoder is a SCUT instrument being read on out-of-domain "
        "faces: cleft crops are not SCUT faces, and the ~3x is mostly "
        "that domain gap, not a cleft-specific finding. It is exactly "
        "the SCUT_NORMAL_PRIOR at cohort scale, which is why the "
        "band-relative design exists"
    ),
    "dispersion_required_before_quoting": (
        "+0.0775 is a mean over 237 patients and was reported without "
        "spread. The rerun computes the per-patient double difference "
        "(sd, range, fraction positive) and a patient bootstrap 95% CI "
        "from the banked JSONs -- DESCRIPTIVE, REPORT-NEVER-GATE, "
        "entering no PLAN 4.3 condition"
    ),
    "readings_applied": (
        "both registered sentences were applied and stand; the "
        "SCUT_NORMAL_PRIOR sentence rides with them by name"
    ),
}


#: **[DEFECT + RETRACTION 2026-08-24] (b) A REGISTERED READING WAS
#: APPLIED TO A COMPUTATION OVER NOBODY.**
READING_APPLIED_TO_NOBODY = {
    "defect": (
        "2026-08-24, run p13-regional: r +nan over ZERO patients -- the "
        "cleft_band_errors-to-score-sheet join produced no rows -- and "
        "the task then printed the registered 'the eye's impression was "
        "SAMPLING' sentence anyway"
    ),
    "withdrawn": (
        "**THE SAMPLING SENTENCE IS WITHDRAWN BY NAME.** Its "
        "applies-when was a FLAT CORRELATION OVER 237 PATIENTS, not an "
        "empty join. Nothing was measured about sampling, the eye's "
        "impression is neither supported nor undermined by that run, "
        "and the sentence may not be cited from it. The reading itself "
        "is unchanged and still stands for the rerun"
    ),
    "root_cause": (
        "scoresheet.load_median is keyed by the sheet's FRONTAL PHOTO "
        "id, not the patient id, so joining it on patient_id matched "
        "nothing. run.median_by_patient already resolves patient -> "
        "frontal photo -> Median through the manifest, and its own "
        "docstring says why it was extracted on 2026-08-17: 'a second "
        "copy of a label resolution is how two runs quietly measure "
        "against different labels.' This turn wrote a THIRD copy and "
        "the warning came true the same week"
    ),
    "fixed": (
        "both stop-3 tasks now resolve grades through the shipped "
        "median_by_patient (run._grades_for), so there is one label "
        "resolution again; p13_regional gains the manifest input it "
        "needs to do so"
    ),
    "same_root_as": (
        "(c-ii)'s empty retention-by-grade and +nan anchor -- ONE root "
        "cause with two surfaces, fixed in one place both tasks consume"
    ),
}


#: **[STRUCTURAL GUARD 2026-08-24, TEST-PINNED] A READING MAY NOT PRINT
#: OVER THE WRONG ROW COUNT** -- the general fix for the class
#: READING_APPLIED_TO_NOBODY is an instance of.
READING_COUNT_GUARD = {
    "registered": "2026-08-24, applied across stop 3, test-pinned",
    "the_rule": (
        "no registered reading prints when its computation's "
        "joined/row count differs from the DECLARED expectation. The "
        "task refuses the sentence and reports the count mismatch in "
        "its place"
    ),
    "why_it_is_the_right_fix": (
        "the (b) sentence was not wrong about its own subject -- it was "
        "applied to the WRONG SUBJECT, silently, because nothing "
        "checked the count first. A reading's APPLIES-WHEN is part of "
        "the reading, so the code that applies it must know the count"
    ),
    "refusal_is_not_failure": (
        "the run continues and the numbers are still written; what is "
        "withheld is the INTERPRETATION -- which is the thing that "
        "would have been quoted"
    ),
    "fired_on": (
        "(b) fired it; (c-ii)'s empty retention-by-grade would have "
        "fired it too -- the near-twice failure that made it structural "
        "rather than a one-line fix"
    ),
    "implementation": "run._registered_reading, one helper, three call sites",
}


#: **[DEFECT FAMILY 2026-08-24] THE WRITER DIED AFTER THE SAVE, BEFORE
#: ITS OWN BOOKKEEPING** -- a new arrival, and the artifact's
#: completeness is unknown until measured.
WRITER_DIED_AFTER_SAVE = {
    "defect": (
        "2026-08-24, run p13-extract-recon attempt 0: "
        "embeddings_recon_v1 was (partly?) written, then the task "
        "crashed at KeyError: 'rollup' composing its OWN metrics. "
        "Attempts 1-2 then correctly refused the existing directory "
        "(+2 on the retry record) -- the immutability guard working"
    ),
    "root_cause": (
        "embeddings.save returns the artifact's METADATA (kind, "
        "backbone, patient_ids, shape) and has NO 'rollup' key; "
        "decoder.save_scut_embeddings returns a hash_dir payload, which "
        "does. The task read one writer's return shape off the other. "
        "TWO WRITERS, TWO SHAPES, one caller assuming"
    ),
    "the_family_note": (
        "**a new failure family: the artifact's writer died AFTER the "
        "save but BEFORE its bookkeeping, so the artifact's "
        "COMPLETENESS IS UNKNOWN from the run record alone.** The "
        "immutability guard then makes the state sticky -- retries "
        "refuse, and nothing can proceed until someone MEASURES the "
        "directory. Neither 'it exists' nor 'the run failed' answers "
        "whether it is whole"
    ),
    "verdict_pending_measurement": (
        "scripts/verify_recon_artifact.py measures it on the cluster: "
        "shape, dtype, finiteness, and row alignment against cleft_v1's "
        "manifest order. If WHOLE it is kept with this fix recorded; if "
        "PARTIAL the maintainer deletes it and the fixed task re-extracts "
        "into the same version. Nothing downstream consumed it -- the "
        "probe never ran, correctly, having no input"
    ),
    "fixed": (
        "the rollup is computed with hash_dir(out_dir), the way every "
        "other caller computes it, and the metadata's own kind is "
        "recorded from what save returned"
    ),
}


#: **[CHECKED 2026-08-24] THE ANCHOR AND PHASE 4'S 0.158 ARE DIFFERENT
#: QUANTITIES -- THE COMPARISON IS REFUSED.** the maintainer's demanded check,
#: landed before anything else was concluded from (c).
ANCHOR_QUANTITIES_DIFFER = {
    "checked": (
        "2026-08-24, from mirror.py's own definitions, before any "
        "conclusion was drawn from the 0.0131-vs-0.158 gap"
    ),
    "verdict": (
        "DIFFERENT QUANTITIES, on five axes -- the discrepancy is NOT a "
        "finding about the instrument, and the two numbers may never be "
        "compared"
    ),
    "the_axes": {
        "statistic": (
            "Phase 4's 0.158 is the 5-fold CV PCC of a RIDGE REGRESSION "
            "over 22 features (whole + bands + anatomy regions; 23 "
            "trained parameters, five seeds 0.140-0.183); stop 3's "
            "+0.0131 is the RAW Pearson r of ONE scalar. A trained "
            "multi-feature fit and a single feature's correlation are "
            "categorically different numbers"
        ),
        "geometry": (
            "Phase 4 is G2 by a guard (require_g2: the mask is the full "
            "square, the midline exact, every pixel data); stop 3 is G1 "
            "with white trapezium pad"
        ),
        "midline": (
            "Phase 4 flips about the image midline (= the face midline "
            "at G2); stop 3 flips about the CONTENT BOX's midline "
            "inside a padded G1 crop"
        ),
        "mask_rule": (
            "Phase 4 has NO content mask -- at G2 none is needed; stop "
            "3 uses the symmetric content mask (mask & mask[:, ::-1], "
            "WHITE_LEVEL 250)"
        ),
        "bands_and_label": (
            "Phase 4's bands are normalised patch-scheme boxes plus "
            "frozen anatomy regions; stop 3's are content-box "
            "horizontal thirds, positional not anatomical, nose+lips "
            "only. And the labels differ: Phase 4 fits the MEAN label, "
            "stop 3 correlates against the MEDIAN grade"
        ),
    },
    "rule": (
        "nobody compares +0.0131 to 0.158. The 0.158 stays what it is "
        "-- Phase 4's trained-arm result -- and (c)'s anchor is its OWN "
        "measured number"
    ),
    "consequence_for_c": (
        "(c)'s status stands ON ITS OWN MEASUREMENT: the stop-3 scalar "
        "barely tracks grade in the ORIGINALS (r +0.0131), so (c) "
        "cannot strongly test the eye's claim -- a fact about this "
        "instrument on this cohort, not a contradiction of Phase 4"
    ),
}


#: **[BANKED 2026-08-24] STOP 3'S RESULTS** -- (a) and (b) clean, (c)
#: with three findings including a refutation of our own registered
#: caveat, (d) whole on re-extraction. The probe is the phase's last run.
STOP_3_BANKED = {
    "banked": "2026-08-24; the probe (d part 2) is the phase's last run",
    "a_double_difference": {
        "figures": (
            "+0.0775; everywhere-gap 3.23x STATED FIRST "
            "(EVERYWHERE_GAP_IS_THE_LARGER_FACT); dispersion sd 0.6115, "
            "58.6% of patients positive, patient bootstrap 95% CI "
            "[-0.0023, +0.1538] -- the interval INCLUDES ZERO"
        ),
        "reading": (
            "the registered sentence stands, and now carries: DIRECTION "
            "AS REGISTERED, UNRESOLVED AT 237 PATIENTS -- the "
            "descriptive cousin of the paired verdicts. DESCRIPTIVE, no "
            "ledger row, no PLAN 4.3 machinery"
        ),
    },
    "b_band_error_vs_grade": {
        "figures": (
            "237/237 joined via the shipped resolver, READING_COUNT_GUARD "
            "passed; r +0.0424; per-grade means 2330/2454/2491/2496/3376 "
            "(grades 1..5, the run's reported scale) -- STRICTLY "
            "MONOTONE, middle three nearly flat, grade 5 far above"
        ),
        "reading": (
            "the phase's CLEANEST CONSISTENCY with "
            "COARSE_SEVERITY_HYPOTHESIS: extremes separable, middle "
            "collapsed -- which is exactly why r is small. "
            "CONSISTENCY-NOT-CONFIRMATION, as the registered reading "
            "already says, and the tails are THIN: n=5 at grade 1, n=3 "
            "at grade 5"
        ),
    },
    "c_asymmetry_retention": {
        "i_no_ordering": (
            "r(asym_orig, asym_recon) = +0.0082 over 237 -- the decoder "
            "preserves NO per-patient asymmetry ordering: reconstructed "
            "asymmetry is DECODER-GENERIC, not the patient's. THE "
            "SHARPEST SINGLE NUMBER OF THE PHASE"
        ),
        "ii_caveat_refuted": (
            "the registered blur-caveat's prediction ('raw retention "
            "sits below 1 for every patient') is REFUTED AS MEASURED: "
            "mild grades retain 1.01-1.15. Mechanism as measured: the "
            "decoder ADDS generic asymmetry of its own, so where the "
            "original has little, the ratio exceeds 1. The registered "
            "text carries a dated correction "
            "(STOP_3_REGISTERED['c_asymmetry_retention'])"
        ),
        "iii_retention_by_grade": (
            "1.01/1.15/1.09/0.98/0.71 (grades 1..5) against the SCUT "
            "fully-normalised floor 0.66 -- severe pulled HARDEST "
            "toward normal, directionally the SCUT_NORMAL_PRIOR. On 3 "
            "grade-5 patients, attached as such"
        ),
        "status": (
            "(c) is UNABLE TO STRONGLY TEST the eye's claim: the "
            "instrument barely tracks grade in the originals (its own "
            "anchor r +0.0131) -- a statement about THIS instrument on "
            "THIS cohort, scoped by ANCHOR_QUANTITIES_DIFFER, not a "
            "contradiction of Phase 4"
        ),
    },
    "d_artifact": (
        "p13-extract-recon-2, ONE clean attempt with the fixed writer; "
        "verifier verdict WHOLE (scripts/verify_recon_artifact.py); "
        "run-reported rollup 87d24577... -- the config fill waits for "
        "the declare pass, the standing two-pass"
    ),
    "retries": (
        "+3 on the retry record from p13-asymmetry-2's OOM on a 13.04 "
        "GiB device -- a FRACTIONAL GPU SLICE. Worth carrying forward: "
        "the standard job template does NOT guarantee a full card, and "
        "a job sized for one may land on a slice"
    ),
}


#: **[BANKED 2026-08-24] (d) PART 2: THE PROBE ON THE RECONSTRUCTIONS**
#: -- the phase's last run, single attempt.
PROBE_ON_RECONSTRUCTIONS = {
    "banked": "2026-08-24, p13-probe-recon, single attempt -- the last run",
    "figures": (
        "per-seed PCC 0.1871 / 0.1679 / 0.2062 / 0.1242 / 0.1610 "
        "(seeds 1337, 2024, 7, 99, 12345); mean 0.1693, sd 0.0305"
    ),
    "reading_applied": (
        "the registered 'WELL BELOW' sentence FIRES: against A's 0.2505 "
        "as STATED CONTEXT, the bottleneck-plus-decoder chain destroys "
        "~0.081 of correlation -- quantified, which is more than the "
        "eye could give. DESCRIPTIVE: no ladder entry, NO PAIRED CLAIM, "
        "no ledger row; the comparator is context and was never "
        "contested (STOP_3_REGISTERED['d_grade_decodability'])"
    ),
    "triangulation": (
        "the reconstructions retain about TWO-THIRDS of A's grade "
        "signal while carrying NONE of the patients' asymmetry ordering "
        "((c)(i), r +0.0082). So the surviving signal does NOT ride the "
        "mirror-scalar's asymmetry -- it rides whatever else the "
        "decoder preserves, which is consistent with (b)'s "
        "coarse-severity means and with the eye's impressions "
        "(STOP_2_REVIEWED, whose figures remain EYE-IMPRESSIONS)"
    ),
    "seed_sd_note": (
        "sd 0.0305 is about 4x A's own seed sd -- the NOISY-ARM pattern "
        "again: a spread that wide is a fact about the arm, and any "
        "future comparison against it must be thresholded from the two "
        "arms' own spreads (PLAN 4.12.1), never from an inherited band"
    ),
}


#: **[CLOSED 2026-08-24] PHASE 13'S CLOSING.** Every sentence below
#: cites a record written before it; NO NEW CLAIM is made here.
PHASE_13_CLOSING = {
    "closed": "2026-08-24, all three stops complete, the probe last",
    "criteria_walk": {
        "1_shakedown_first": (
            "MET. The decoder trained on masked SCUT G1 224 from frozen "
            "imagenet ViT-B/16 embeddings; held-out reconstructions "
            "(3,300 / 2,199 -- the small_correction's count) went to "
            "SHAREABLE sheets and the visual check PASSED them before any "
            "cohort pixel existed (STOP_1_REVIEWED: machinery clean, "
            "checkerboard absent, worst-24 hard faces not defects)"
        ),
        "2_gate_and_tier": (
            "MET. cleft_reconstruction_acknowledged refused first, "
            "before any read; the edit (REC 13/SW/0064 and "
            "23/YH/0037) was carried by the generator and never "
            "defaulted; stop 3's regenerating tasks inherited the SAME "
            "decision through one helper. Every reconstruction stayed "
            "CLUSTER-ONLY, and the raw tensor was never persisted at "
            "all -- cohort pixels existed only in the sheets "
            "(STOP_2_BUILT)"
        ),
        "3_regional_split_registered_first": (
            "MET, with its registered departure standing: the split was "
            "registered before any cohort reconstruction was seen, but "
            "the bands are CONTENT-BOX POSITIONAL THIRDS, not the "
            "anatomy scheme's boxes -- a departure from criterion 3's "
            "letter, flagged when registered (DOUBLE_DIFFERENCE_RULE) "
            "and never quietly reconciled"
        ),
        "4_descriptive_only": (
            "MET. No ladder entry, no two-condition claim, NO LEDGER "
            "ROW was written for any reconstruction number -- the "
            "ledger stands where Phase 12 left it, at 31 entries"
        ),
        "5_regional_comparison_with_readings": (
            "MET. The double difference was computed by the registered "
            "rule with both readings committed in advance; the "
            "everywhere-gap is stated first "
            "(EVERYWHERE_GAP_IS_THE_LARGER_FACT) and the SCUT_NORMAL_"
            "PRIOR sentence rides by name. The bound failure-mode "
            "reading is NOT claimed to have fired: the direction is as "
            "registered but UNRESOLVED at 237 patients"
        ),
        "6_encoder_untouched": (
            "MET BY CONSTRUCTION. The decoder consumed cached "
            "embeddings throughout; no encoder gradient exists anywhere "
            "in the phase, asserted in code and test"
        ),
        "7_curves_checkpoints_suite": (
            "MET. Every epoch's curve row and checkpoint was written, "
            "not discarded -- and epoch 20, the pick, was "
            "loadable precisely because none had been thrown away. "
            "Suite green at the close"
        ),
    },
    "what_the_phase_measured": {
        "a": (
            "double difference +0.0775, riding on an EVERYWHERE-GAP of "
            "3.23x -- the cohort reconstructs about three times worse "
            "than held-out SCUT in every band. Dispersion sd 0.6115, "
            "58.6% positive, bootstrap 95% CI [-0.0023, +0.1538], which "
            "INCLUDES ZERO: direction as registered, unresolved at 237"
        ),
        "b": (
            "band error vs grade r +0.0424 over 237/237; per-grade "
            "means 2330/2454/2491/2496/3376 -- strictly monotone, "
            "middle three nearly flat, grade 5 far above. The phase's "
            "cleanest CONSISTENCY with COARSE_SEVERITY_HYPOTHESIS, "
            "consistency-not-confirmation, tails thin (n=5, n=3)"
        ),
        "c": (
            "r(asym_orig, asym_recon) = +0.0082 -- the decoder "
            "preserves NO per-patient asymmetry ordering; reconstructed "
            "asymmetry is DECODER-GENERIC. The registered blur caveat "
            "was REFUTED by its own measurement (mild grades retain "
            "1.01-1.15; the decoder ADDS asymmetry). Retention by grade "
            "1.01/1.15/1.09/0.98/0.71 against the SCUT floor 0.66 -- "
            "severe pulled hardest toward normal, on n=3 "
            "[UPGRADED 2026-08-24, the post-closing probe "
            "(ASYMMETRY_ENCODING_PROBE_BANKED): ENCODED AT r~0.49, "
            "RENDERED AT r=0.0082 -- the decoder discards per-patient "
            "geometry the representation carries; 'the chain loses it' "
            "is superseded]"
        ),
        "d": (
            "the probe on the reconstructions: mean PCC 0.1693 (sd "
            "0.0305) against A's 0.2505 as stated context -- the 'well "
            "below' reading fired, ~0.081 of correlation destroyed by "
            "the chain. Reconstructions keep about two-thirds of the "
            "grade signal while keeping none of the asymmetry ordering "
            "(PROBE_ON_RECONSTRUCTIONS)"
        ),
        "e": (
            "scar-trace survival was NOT quantitatively tested and is "
            "recorded as untestable here: no annotations exist and a "
            "scar detector would be a new unvalidated instrument. It "
            "remains an EYE IMPRESSION, tagged"
        ),
    },
    "what_the_eye_saw_and_what_it_is": (
        "the cohort review PASSED on fidelity with zero machinery "
        "artifacts -- a verdict. Its clinical-detail figures (~3 pose "
        "misses; ~6 of 12 with visible asymmetry; ~4 with a scar-like "
        "trace) are EYE-IMPRESSIONS, tagged in their own key names, and "
        "may be quoted only as impressions (STOP_2_REVIEWED)"
    ),
    "the_hypothesis_after_measurement": (
        "COARSE_SEVERITY_HYPOTHESIS is NOT confirmed. (b)'s monotone "
        "means with a flat middle are CONSISTENT with it; (c) could not "
        "strongly test it, and (d) shows two-thirds of the grade signal "
        "surviving a chain that keeps no asymmetry ordering. It leaves "
        "the phase as it entered: a hypothesis, better specified"
    ),
    "defects_and_the_structural_residue": (
        "four defects across three tasks were found and fixed in one "
        "pass: a THIRD copy of a label resolution that joined zero rows "
        "(READING_APPLIED_TO_NOBODY), an unpaired content mask that "
        "measured two families over different pixel sets, a cohort "
        "count crossed into a SCUT field, and a writer whose caller "
        "read the wrong return shape (WRITER_DIED_AFTER_SAVE, a new "
        "family: completeness unknown after a post-save crash). The "
        "STRUCTURAL RESIDUE is READING_COUNT_GUARD -- no registered "
        "reading prints over a row count it was not registered against. "
        "Earlier in the phase, PARAMETER_BAND_MISESTIMATED: the "
        "registered '~3-5M' was an ESTIMATE and the shape constructed "
        "to 10,628,771, refused by its own guard; resolved to the "
        "7x7x96 shape at EXACTLY 3,862,339, re-registered by equality "
        "with a torch-free arithmetic function"
    ),
    "carried_forward_by_name": {
        "supervisor_question_8": (
            "unchanged and still open (phase11's ask list, phase12's "
            "framing): nothing in Phase 13 touched it"
        ),
        "ar_confound_caveat": (
            "phase12.BASAL_CONFOUND_OBSERVED's staging-geometry caveat, "
            "r(AR, mean) = +0.1601 exceeding 0.13, still travels with "
            "arms B and C by name -- untouched here"
        ),
        "fractional_gpu_note": (
            "NEW: p13-asymmetry-2 hit OOM on a 13.04 GiB device -- the "
            "standard job template does NOT guarantee a full card, and "
            "a job sized for one may land on a slice "
            "(STOP_3_BANKED['retries'])"
        ),
        "retry_limit_item": (
            "phase8.RETRY_LIMIT_IS_NOT_HOLDING took +2 (parameter band), "
            "+2 (partial write) and +3 (the OOM) this phase -- the item "
            "is not closed and the arithmetic is now larger"
        ),
        "instrument_limitation_from_c": (
            "OPEN: the stop-3 asymmetry scalar barely tracks grade in "
            "the ORIGINALS (r +0.0131), so it could not strongly test "
            "the eye's claim. What would license a stronger test is a "
            "measure whose grade-tracking is established on this "
            "cohort -- and NOT a comparison against Phase 4's 0.158, "
            "which ANCHOR_QUANTITIES_DIFFER refuses as a different "
            "quantity on five axes"
        ),
        "scar_traces": (
            "OPEN and untestable with what exists: annotations would be "
            "needed, and none do "
            "(STOP_3_REGISTERED['e_scar_trace_not_testable'])"
        ),
    },
    "no_new_claims": (
        "every sentence in this record cites a record written before "
        "it; the phase adds no claim at its close that it did not "
        "measure during it"
    ),
    # **[ADDENDUM REGISTERED 2026-08-24, VERDICTS PENDING]** The closing
    # above stands as written. It locates the surviving grade signal only
    # by exclusion -- "whatever else the decoder preserves" -- and two
    # registered probes (CLOSING_ADDENDUM_REGISTERED) convert that into a
    # location. When they land, their verdicts are absorbed HERE as a
    # dated addendum with this original text preserved, per the standing
    # correction pattern.
    # [ABSORBED 2026-08-24, the same day: the addendum's verdicts landed
    # and the field became a dict -- the ORIGINAL registration text is
    # preserved verbatim under "registered", per the standing pattern.]
    "addendum": {
        "registered": (
            "2026-08-24, REGISTERED, VERDICTS PENDING: P1 the confound "
            "ceiling (pod) and P2 the band-occlusion probe (three GPU "
            "runs plus three probes) answer what "
            "what_the_phase_measured['d']'s triangulation leaves open "
            "-- where the surviving signal lives. P3 is registered "
            "CONDITIONAL and unbuilt. Readings for all of them are "
            "committed in CLOSING_ADDENDUM_REGISTERED before any "
            "number exists; this closing is amended in place when they "
            "return, never rewritten"
        ),
        "absorbed": (
            "2026-08-24: all three verdicts banked "
            "(P1_CONFOUND_CEILING_BANKED, P2_BAND_OCCLUSION_BANKED, "
            "P3_BANKED) and absorbed here, the original registration "
            "preserved above"
        ),
        "p1": (
            "the confound ceiling FIRES AT THE BOUNDARY: +0.1000 (sd "
            "0.0228, seeds 0.066-0.122) against the pre-declared 0.10. "
            "LIMITATIONS-GRADE: approximately 0.10 of correlation, "
            "~40% of the headline arm's, is recoverable from five "
            "statistics that cannot see a nose -- and this TRAVELS "
            "WITH THE HEADLINE NUMBER wherever it is quoted, subsuming "
            "the single-variable AR confound"
        ),
        "p2": (
            "THE ADDENDUM'S CENTRAL FINDING: the grade signal is "
            "LIPS-DOMINANT (drop -0.162), nose secondary (-0.082), "
            "eyes near-nothing (-0.036, the built-in area-matched "
            "control), monotone down the face, with lips-occluded "
            "overlapping P1's image-statistics floor"
        ),
        "p3": (
            "NEITHER registered reading fired verbatim (applied "
            "neither, said so): the INVERSION -- lips-dominance did "
            "not survive the decode, the decoder destroys precisely "
            "the band-specific lip signal (-0.162 on originals, "
            "-0.072 on reconstructions) -- and the ENTANGLEMENT -- "
            "drops summing -0.30 against a 0.17 total, so what "
            "survives is band-diffuse and holistic. The band-smearing "
            "caveat is the OPERATIVE LIMITATION on every P3 quote "
            "(P3_BANKED)"
        ),
        "mechanism_summary_cited_not_new": (
            "the mechanism chapter's sentence, every clause cited: the "
            "headline arm's signal is LIPS-DOMINANT with an "
            "image-statistics floor of ~0.10 (P2_BAND_OCCLUSION_BANKED, "
            "P1_CONFOUND_CEILING_BANKED); the embedding's face-shaped "
            "compression preserves coarse severity ((b)'s monotone "
            "means, the probe's two-thirds survival) but destroys "
            "band-specific lip detail (P3's inversion) and per-patient "
            "asymmetry ((c)'s r +0.0082); what survives a decode is "
            "DIFFUSE GLOBAL APPEARANCE (P3's entanglement, "
            "PROBE_ON_RECONSTRUCTIONS' triangulation). NO CLAUSE here "
            "is new; COARSE_SEVERITY_HYPOTHESIS itself remains "
            "unconfirmed as the closing records. [ONE CLAUSE ADDED "
            "2026-08-24, cited to ASYMMETRY_ENCODING_PROBE_BANKED: the "
            "embedding is a RICHER WITNESS than its reconstruction -- "
            "decoder-based evidence is a LOWER BOUND on what a "
            "representation carries, demonstrated at 0.49-vs-0.008 on "
            "this cohort]"
        ),
        "exit_addendum_criteria_walk": {
            "1_readings_precommitted": (
                "MET: every P1/P2/P3 reading predates its number; P1's "
                "was applied through READING_COUNT_GUARD in-run at "
                "237/237, P2's at banking over full-cohort probes, and "
                "P3's verdict is the guard's SPIRIT on a different "
                "axis -- the sentences' applies-when failed on "
                "direction, so NEITHER was applied, and the refusal is "
                "the record"
            ),
            "2_descriptive_throughout": (
                "MET: no ladder entry, no paired claim, A's 0.2505 "
                "stayed stated context, and the ledger stands at 31 -- "
                "no addendum number bought a row"
            ),
            "3_no_cohort_pixel_persisted": (
                "MET: P1 touched no pixel; P2 and P3's occluded crops "
                "and regenerated reconstructions lived in memory only, "
                "with CLUSTER-ONLY embedding sets the sole artifacts"
            ),
            "4_p3_conditional_honoured": (
                "MET: P3 was built only after its registered condition "
                "fired on P2's own figures (nose -0.082, lips -0.162), "
                "and its band-smearing caveat rode as registered -- "
                "now operative"
            ),
            "5_absorbed_original_preserved": (
                "MET: this dict is the dated absorption; the "
                "registration text is preserved verbatim above and "
                "the closing body was never rewritten"
            ),
        },
    },
    # [POST-CLOSING CHECK REGISTERED 2026-08-24, the amend-in-place
    # pattern: the closing body above is untouched.]
    "post_closing_check": (
        "2026-08-24: ASYMMETRY_ENCODING_PROBE_REGISTERED -- the maintainer's "
        "question exposed that (c)'s r +0.0082 conflates 'never "
        "encoded' with 'encoded but not rendered'. A feature-space "
        "ridge probe (embeddings -> asym_orig, pod, no gate, readings "
        "committed, threshold declared) resolves which sentence (c) "
        "gets. DESCRIPTIVE, no ledger row; the verdict amends this "
        "closing dated, in place, when it lands. [LANDED 2026-08-24: "
        "+0.4856 (sd 0.0231) against 0.10 -- the second reading fired "
        "decisively; the upgrades are applied in place "
        "(ASYMMETRY_ENCODING_PROBE_BANKED)]"
    ),
}


#: **[REGISTERED 2026-08-24, READINGS COMMITTED BEFORE ANY NUMBER] THE
#: CLOSING ADDENDUM.** Phase 13's closing says what the surviving grade
#: signal is NOT -- not the mirror-scalar's asymmetry -- and characterises
#: it only as "whatever else the decoder preserves". Two cheap probes turn
#: that characterisation into a LOCATION, and both answer this phase's
#: question rather than Phase 14's.
CLOSING_ADDENDUM_REGISTERED = {
    "registered": (
        "2026-08-24, before any addendum number exists; the closing is "
        "amended in place when the verdicts land, original preserved"
    ),
    "why_it_belongs_to_phase_13": (
        "PHASE_13_CLOSING locates the surviving signal only by "
        "exclusion. These two probes ask WHERE it lives -- the same "
        "question the phase opened with (what the embedding keeps), not "
        "Phase 14's"
    ),
    "p1_confound_ceiling": {
        "what": (
            "regress the median grade on NON-ANATOMICAL GLOBAL "
            "STATISTICS alone -- aspect ratio, brightness mean, "
            "contrast (pixel sd), content-pixel fraction, corner-white "
            "fraction -- computed from staged_v1 and the geometry rows, "
            "through the standard recipe's shape: cleft_v1's own folds, "
            "ridge, five seeds, inner-val split per seed"
        ),
        "reading_if_substantial": (
            "PCC >= ~0.10 -> A SUBSTANTIAL SLICE OF THE 0.25 RIDES "
            "IMAGE STATISTICS RATHER THAN ANATOMY. This is a "
            "limitations-grade finding and MUST TRAVEL WITH THE "
            "HEADLINE NUMBER wherever it is quoted"
        ),
        "reading_if_near_zero": (
            "PCC ~ 0 -> the signal is at least FACIAL, and the AR "
            "confound's reach is BOUNDED -- which strengthens every "
            "anatomy-based reading in the phase without proving any"
        ),
        "composes_with": (
            "phase12.BASAL_CONFOUND_OBSERVED, which measured r(AR, "
            "mean) = +0.1601 on ONE variable. P1 is the MULTIVARIATE "
            "version of that same worry, and its verdict subsumes the "
            "single-variable one rather than repeating it"
        ),
        "status": "DESCRIPTIVE, report-never-gate, no ladder entry",
        "compute": (
            "POD ARITHMETIC -- pixels and scalars, no backbone, no GPU. "
            "No gate: nothing here generates or modifies a cohort pixel"
        ),
    },
    "p2_band_occlusion": {
        "what": (
            "re-embed the cohort THREE times with one positional third "
            "blanked to white each time (top / middle / bottom, the "
            "same band_rows thirds), the blanking applied to the staged "
            "G1 crops IN MEMORY and nothing persisted but the embedding "
            "sets; then A's probe on each, five seeds, recipe "
            "byte-identical"
        ),
        "reading_if_relevant_bands_collapse": (
            "blanking nose+lips (middle+bottom) COLLAPSES the PCC -> "
            "the grade signal LIVES WHERE GRADERS LOOK, and the "
            "coarse-severity story sharpens"
        ),
        "reading_if_relevant_bands_barely_dent": (
            "blanking them barely dents it -> THE GRADE SIGNAL DOES NOT "
            "COME FROM THE CLINICAL REGION -- the sharpest possible "
            "mechanism finding of the whole project, and P1's result "
            "frames what it rides instead"
        ),
        "white_not_black": (
            "the blank value is WHITE, which is what the staging pad "
            "already is: an occluded band is then indistinguishable "
            "from pad to everything downstream, and pad is the one "
            "thing the phase already established as trivially "
            "reconstructible and excluded everywhere. Black would "
            "introduce a value the pipeline never sees"
        ),
        "governance": (
            "the occluded embeddings are PATIENT-DERIVED intermediates: "
            "CLUSTER-ONLY, crops in memory only. The gate is applied by "
            "INHERITANCE AND CAUTION, not because its trigger fires -- "
            "occlusion MODIFIES an existing crop rather than GENERATING "
            "a new image, so the gate's own words do not strictly reach "
            "it; the instruction is that it rides wherever "
            "cohort pixels are touched, and the distinction is recorded "
            "rather than blurred"
        ),
        "status": "DESCRIPTIVE, report-never-gate, no ladder entry",
        "compute": (
            "THREE GPU runs (one per band) plus THREE probes; the "
            "probes themselves are ordinary train_cv arms"
        ),
    },
    "p3_conditional_not_built": {
        "what": (
            "the same three occlusions applied to RECONSTRUCTIONS "
            "(regenerated in memory, gate inherited) with three probes "
            "-- splitting (d)'s 0.1693 by band against P2's split of "
            "A's 0.2505, to locate WHERE the decoder destroys grade "
            "signal"
        ),
        "condition": (
            "REGISTERED BUT NOT BUILT. It runs ONLY IF P2 shows a "
            "substantial middle+bottom drop; otherwise there is no "
            "per-band destruction to locate"
        ),
        "pre_registered_caveat": (
            "the decoder's errors are BAND-SMEARED -- the 3.23x "
            "everywhere-gap (EVERYWHERE_GAP_IS_THE_LARGER_FACT) is not "
            "band-specific -- so per-band attribution on "
            "RECONSTRUCTIONS is a BLUNTER INSTRUMENT than the same "
            "attribution on originals. This caveat is registered NOW so "
            "it cannot be forgotten if P3's numbers turn out friendly"
        ),
    },
    "exit_addendum_criteria": (
        "1. both readings for P1 and both for P2 are committed here, "
        "BEFORE any addendum number exists, and are applied through "
        "READING_COUNT_GUARD like every other reading in the phase",
        "2. DESCRIPTIVE throughout: no ladder entry, no ledger row, no "
        "PLAN 4.3 machinery, no paired claim -- A's 0.2505 stays stated "
        "context as it was for (d)",
        "3. no cohort pixel is persisted: P1 touches none, P2's "
        "occluded crops live in memory and only embedding sets land, "
        "CLUSTER-ONLY",
        "4. P3 stays UNBUILT unless P2's condition is met, and carries "
        "its band-smearing caveat if it ever runs",
        "5. the verdicts are absorbed into PHASE_13_CLOSING as a DATED "
        "ADDENDUM with the original closing preserved -- the standing "
        "correction pattern, never a rewrite",
    ),
}


#: **[BANKED 2026-08-24] P1'S VERDICT: THE CONFOUND CEILING FIRES -- AT
#: THE BOUNDARY, AND SAID SO.**
P1_CONFOUND_CEILING_BANKED = {
    "banked": (
        "2026-08-24, p13-confound-ceiling, single clean pod attempt, "
        "237/237 through the shipped resolver"
    ),
    "figures": (
        "per-seed OOF PCC +0.1183 / +0.0659 / +0.1216 / +0.0897 / "
        "+0.1044 (seeds 1337, 2024, 7, 99, 12345); mean +0.1000, sd "
        "0.0228, against the DECLARED threshold 0.10"
    ),
    "reading_fired": (
        "the registered reading FIRES: a substantial slice of the 0.25 "
        "rides image statistics rather than anatomy -- "
        "LIMITATIONS-GRADE, and it TRAVELS WITH THE HEADLINE NUMBER "
        "wherever that number is quoted"
    ),
    "fires_at_the_boundary": (
        "**HONEST NOTE, part of the verdict**: the mean sits EXACTLY on "
        "the declared threshold, with seeds spanning +0.066 to +0.122. "
        "The quotable sentence is therefore: 'approximately 0.10 of "
        "correlation, ~40% of the headline arm's, is recoverable from "
        "five statistics that cannot see a nose' -- STATED WITH THE "
        "SEED SPREAD, never as a razor-edge pass. The threshold was "
        "declared before the number existed, which is the only reason "
        "firing at the boundary is reportable at all"
    ),
    "composition": (
        "printed as registered: this SUBSUMES phase12."
        "BASAL_CONFOUND_OBSERVED's single-variable r(AR, mean) = "
        "+0.1601 as its multivariate version -- one confound sentence "
        "now, not two"
    ),
    "status": "DESCRIPTIVE, report-never-gate, no ladder entry, no ledger row",

    # -----------------------------------------------------------------
    # [POINTER 2026-08-31 -- NOTHING ABOVE IS CHANGED]
    #
    # This ceiling is scored against the MEDIAN grade (run._grades_for);
    # the headline 0.2520 is against the PANEL MEAN. The two are not the
    # same quantity (phase3.LEARNABILITY_237: mean 0.6022 vs median
    # 0.5708), so "~40% of the headline arm's" above compares ACROSS
    # TARGETS.
    #
    # THIS IS NOT A CORRECTION AND NOTHING HERE IS REWRITTEN. +0.1000 is
    # CORRECT FOR ITS OWN QUANTITY and its threshold was declared before
    # the number existed. What was wrong was the COMPARISON, made later
    # and elsewhere -- see phase20.CROSS_TARGET_ERROR_PROVENANCE for
    # where it came from and when it was caught.
    #
    # A like-for-like ceiling on the PANEL MEAN is registered as a NEW
    # ARM in Phase 20 (phase20.ARM_C_REGISTERED), not as a repair of
    # this row. Panel-mean figures should be anchored to Arm C's number
    # once it exists; this figure remains the anchor for median-target
    # quantities.
    # -----------------------------------------------------------------
    "target_is_the_median_grade_2026_08_31": (
        "this ceiling is scored against the MEDIAN grade "
        "(run._grades_for); the headline 0.2520 is against the PANEL "
        "MEAN, and '~40% of the headline arm's' above therefore "
        "compares ACROSS TARGETS. NOT A CORRECTION -- the figure is "
        "correct for its own quantity and nothing here is rewritten; "
        "the COMPARISON is what was wrong. A like-for-like panel-mean "
        "ceiling is a NEW registered arm: phase20.ARM_C_REGISTERED. "
        "Provenance of the bad comparison: "
        "phase20.CROSS_TARGET_ERROR_PROVENANCE"
    ),
}


#: **[BANKED 2026-08-31] P1's per-seed figures, in a MACHINE-READABLE
#: home** -- the ``phase16.ARM_MEANS`` / ``phase17.ARM_MEANS`` pattern,
#: applied here for the same reason and on the same standing rule:
#: **values are never regex-parsed out of prose.**
#:
#: It exists because Phase 20's Arm C compares its own panel-mean ceiling
#: against this median-target one, and the first draft of that comparison
#: read the mean by splitting ``P1_CONFOUND_CEILING_BANKED["figures"]`` on
#: the string ``"mean +"``. That is the prohibited move, and a sentence
#: edit -- a comma, a re-wording -- would have changed a number a run
#: reads. **Nothing above is rewritten**: the prose keeps every figure,
#: and this is where code takes them from.
#:
#: **Transcribed, not parsed**, and checked: the five seeds' mean is
#: +0.1000 and their sample sd 0.0228, which is what the prose states.
#: The check runs in the suite, so the two cannot drift apart silently.
P1_CEILING_FIGURES = {
    "banked": "2026-08-31",
    "provenance": (
        "TRANSCRIBED from P1_CONFOUND_CEILING_BANKED['figures'], never "
        "regex-parsed; the mean and sd are re-derived from the per-seed "
        "values in the suite and must agree with the prose"
    ),
    # [MEASURED 2026-08-31, recorded rather than smoothed] Re-deriving
    # the sd from the FOUR-DECIMAL per-seed values above gives
    # 0.0228567, not 0.0228: the run computed it at full precision and
    # the record rounds the inputs. The mean reproduces exactly. This is
    # the same last-digit story phase17.ARM_MEANS and
    # record_audit.RECORD_ARTIFACT_CHECK both measured, and the banked
    # sd is the RUN's -- it is not adjusted to match a re-derivation
    # from rounded inputs.
    "sd_rederives_to": 0.0228567,
    "sd_rederivation_note": (
        "the sample sd of the four-decimal per-seed values is 0.0228567; "
        "the banked 0.0228 is the RUN's, computed at full precision. The "
        "record is not adjusted to match a re-derivation from rounded "
        "inputs -- the difference is the rounding, not a disagreement"
    ),
    "target": (
        "the MEDIAN grade (run._grades_for) -- NOT the panel mean. See "
        "P1_CONFOUND_CEILING_BANKED['target_is_the_median_grade_2026_08_31']"
    ),
    "pcc_by_seed": {
        1337: 0.1183, 2024: 0.0659, 7: 0.1216, 99: 0.0897, 12345: 0.1044,
    },
    "pcc_mean": 0.1000,
    "pcc_sd": 0.0228,
    "declared_threshold": 0.10,
    "run": "p13-confound-ceiling, 2026-08-24, 237/237",
}


#: **[RECORDED 2026-08-24] THE BAND-NAME CORRESPONDENCE** -- anatomical
#: stems, positional readings, bound explicitly so neither can drift.
BAND_NAME_CORRESPONDENCE = {
    "recorded": "2026-08-24, after three launches died on filenames",
    "the_map": {
        "eyes": "top (the first band_rows third)",
        "nose": "middle (the second third)",
        "lips": "bottom (the third third)",
    },
    "why_it_exists": (
        "the configs and artifacts are stemmed ANATOMICALLY (eyes / "
        "nose / lips, decoder.BANDS' vocabulary) while the registered "
        "P2 readings are phrased POSITIONALLY ('middle+bottom'). The "
        "launch instructions assumed positional stems and the three "
        "first launches died on NONEXISTENT FILENAMES -- before any run "
        "directory existed, so no retry cost and nothing to void"
    ),
    "the_binding": (
        "'middle+bottom' in the registered readings == nose+lips == "
        "decoder.GRADER_RELEVANT; 'top' == eyes == "
        "decoder.GRADE_IRRELEVANT. The names are the anatomy VOCABULARY "
        "over POSITIONAL thirds (the registered residual limitation, "
        "decoder.BANDS) -- the same correspondence the double "
        "difference has used since stop 1, now written where the "
        "occlusion readings bind to it"
    ),
}


#: **[BANKED 2026-08-24] P2'S VERDICT -- THE ADDENDUM'S CENTRAL FINDING:
#: THE GRADE SIGNAL IS LIPS-DOMINANT.**
P2_BAND_OCCLUSION_BANKED = {
    "banked": (
        "2026-08-24, three probes, all single-attempt clean, from "
        "seed_variance.json"
    ),
    "figures": (
        "eyes-occluded mean 0.2143 [0.194, 0.236]; nose-occluded "
        "0.1684 [0.144, 0.180]; lips-occluded 0.0885 [0.068, 0.125] -- "
        "against A's 0.2505 (stated context) and P1's image-statistics "
        "floor 0.1000"
    ),
    "reading_fired": (
        "the registered 'signal lives where graders look' reading "
        "FIRES, with a SHARPER localisation than the reading asked "
        "for: the drop is MONOTONE down the face -- eyes -0.036, nose "
        "-0.082, lips -0.162"
    ),
    "the_central_finding": (
        "the grade signal is LIPS-DOMINANT (the repair site), nose "
        "secondary, eyes near-nothing, floor ~0.10 image statistics. "
        "DESCRIPTIVE throughout: the deltas are across INDEPENDENTLY "
        "TRAINED probes, not a decomposition of one model, and every "
        "quote carries the printed floors and spreads"
    ),
    "eyes_is_the_builtin_control": (
        "the eyes band is the BUILT-IN AREA-MATCHED CONTROL: same "
        "occluded area as each other band, negligible cost (-0.036) -- "
        "so the lips drop is not an occlusion-area artifact"
    ),
    "lips_occluded_sits_at_the_floor": (
        "lips-occluded (0.0885 [0.068, 0.125]) OVERLAPS P1's floor "
        "(0.1000, seeds 0.066-0.122): remove the lip band and what "
        "remains of the grade signal is approximately what five blind "
        "image statistics recover. The anatomy the probe reads above "
        "the floor is, to within these spreads, the lip band"
    ),
    "observation_not_finding": (
        "**UNREGISTERED OBSERVATION, recorded as such**: the "
        "reconstruction probe's 0.1693 (PROBE_ON_RECONSTRUCTIONS) is "
        "numerically close to the nose-occluded 0.1684 -- SUGGESTIVE "
        "that the decoder chain costs about one nose-band of signal, "
        "but no reading was registered for this comparison and it may "
        "be cited only as an observation, never as a finding"
    ),
    "status": (
        "DESCRIPTIVE: no ladder entry, no ledger row, no paired claim; "
        "A's 0.2505 remains stated context"
    ),
}


#: **[TRIGGERED 2026-08-24, READINGS COMMITTED BEFORE ANY NUMBER] P3
#: RUNS.** The registered condition fired -- P2 shows a substantial
#: middle+bottom drop -- and the instruction says run it.
P3_TRIGGERED = {
    "triggered": (
        "2026-08-24: P2's middle+bottom drop is substantial (nose "
        "-0.082, lips -0.162), the registered condition "
        "(CLOSING_ADDENDUM_REGISTERED['p3_conditional_not_built']) "
        "fires, and the instruction says run"
    ),
    "what_runs": (
        "the same three occlusions applied to RECONSTRUCTIONS -- "
        "regenerated in memory through the epoch-20 checkpoint, gate "
        "inherited, occluded by the same band_rows thirds, nothing "
        "persisted but the CLUSTER-ONLY embedding sets -- then three "
        "probes, A's recipe byte-identical, splitting (d)'s 0.1693 by "
        "band against P2's split of A's 0.2505"
    ),
    "reading_if_lips_dominant_like_the_original": (
        "the reconstruction probe's 0.169 drops lips-first, like the "
        "original's -> the decoder PRESERVES the lip signal it has and "
        "destroys signal elsewhere -- the destruction is localisable"
    ),
    "reading_if_occlusions_barely_move_it": (
        "the occlusions barely move 0.169 -> the surviving signal is "
        "BAND-DIFFUSE, consistent with decoder-generic global "
        "appearance -- and the band-smearing caveat is then not a "
        "caveat on the finding, it IS the finding"
    ),
    "caveat_attached_as_registered": (
        "the pre-registered band-smearing caveat rides: the decoder's "
        "errors are band-smeared (the 3.23x everywhere-gap), so "
        "per-band attribution on reconstructions is a BLUNTER "
        "instrument than on originals -- registered before P2's "
        "numbers existed, attached now exactly as written"
    ),
    "status": "DESCRIPTIVE, report-never-gate, no ladder entry",
}


#: **[BANKED 2026-08-24] P3'S VERDICT: NEITHER REGISTERED READING FIRES
#: VERBATIM -- the third outcome, recorded as such (the Phase 10
#: pattern: apply neither and say so).**
P3_BANKED = {
    "banked": (
        "2026-08-24, three probes, all single-attempt clean, from "
        "seed_variance.json"
    ),
    "figures": (
        "on RECONSTRUCTIONS: eyes-occluded 0.0744 [0.051, 0.089]; "
        "nose-occluded 0.0375 [-0.008, 0.059] -- INTERVAL INCLUDES "
        "ZERO; lips-occluded 0.0975 [0.017, 0.161]; against the "
        "unoccluded reconstruction probe's 0.1693. The drops: eyes "
        "-0.095, nose -0.132, lips -0.072"
    ),
    "neither_reading_fires": (
        "**APPLY NEITHER, AND SAY SO** -- the Phase 10 pattern. "
        "Reading 1 (lips-first, like the original) does not fire: lips "
        "is the SMALLEST drop, not the largest. Reading 2 (occlusions "
        "barely move it) does not fire either: every occlusion moved "
        "the probe substantially. The registered sentences' "
        "applies-when failed on DIRECTION, so neither may be quoted as "
        "applied; the third outcome below is what was measured"
    ),
    "component_i_the_inversion": (
        "lips-occlusion costs -0.162 on ORIGINALS (the largest drop, "
        "P2) and -0.072 on RECONSTRUCTIONS (the smallest). The "
        "lips-dominance DID NOT SURVIVE THE DECODE: the decoder "
        "destroys precisely the band-specific lip signal the original "
        "probe rides -- consistent IN SUBSTANCE with registered "
        "reading 2 and with (a)'s direction, without either sentence "
        "firing verbatim"
    ),
    "component_ii_the_entanglement": (
        "the three drops sum to -0.30 against a 0.17 total -- "
        "MASSIVELY SUPER-ADDITIVE. Whatever grade signal survives the "
        "decode is BAND-DIFFUSE AND HOLISTIC: no band carries it, and "
        "removing any band costs more than its share"
    ),
    "caveat_now_operative": (
        "the pre-registered band-smearing caveat is NOW THE OPERATIVE "
        "LIMITATION, stated with every quote: a white stripe on a "
        "SMOOTH reconstruction plausibly pushes the ViT further "
        "out-of-distribution than the same stripe on a sharp original, "
        "so P3's per-band attributions are BLUNT BY CONSTRUCTION -- "
        "and the wide intervals (nose spanning zero) are that "
        "bluntness's visible signature, not noise to be argued away"
    ),
    "status": (
        "NOTHING here is claimable and nothing enters the ledger; "
        "DESCRIPTIVE with floors and spreads attached to every quote"
    ),
}


#: **[REGISTERED 2026-08-24, READINGS COMMITTED BEFORE ANY NUMBER] THE
#: ASYMMETRY-ENCODING PROBE** -- a post-closing check on (c), from the maintainer's
#: question, which exposed a CONFLATION in the closed finding.
ASYMMETRY_ENCODING_PROBE_REGISTERED = {
    "registered": (
        "2026-08-24, post-closing, before any number exists; "
        "DESCRIPTIVE, no ledger row"
    ),
    "the_conflation_it_resolves": (
        "(c)'s r(asym_orig, asym_recon) = +0.0082 CANNOT DISTINGUISH "
        "two mechanisms: 'the embedding never encoded per-patient "
        "asymmetry' from 'the embedding encodes it and the DECODER "
        "fails to render it' -- a generative decoder can drop what the "
        "representation carries. Every pixel-space statistic in stop 3 "
        "shares this blindness, because every one passes through the "
        "decoder"
    ),
    "the_direct_test": (
        "a FEATURE-SPACE probe the decoder never touches: ridge from "
        "the frozen 768-d embeddings (the 0.2520 arm's cached set) to "
        "asym_orig (the banked per-patient scalars from the stop-3 "
        "asymmetry run -- CONSUMED from the banked metrics, never "
        "recomputed: a second computation is a second place for "
        "drift), out-of-fold on cleft_v1's own folds, five seeds. POD "
        "ARITHMETIC, no pixels, no gate"
    ),
    "reading_if_near_zero": (
        "the embedding GENUINELY DOES NOT CARRY the mirror-scalar: "
        "(c)'s sentence upgrades from 'the chain loses it' to 'IT WAS "
        "NEVER ENCODED', and the mechanism chapter hardens -- the "
        "face-shaped compression discards per-patient asymmetry at "
        "the ENCODER, not the decoder"
    ),
    "reading_if_substantial": (
        "the asymmetry IS in the embedding and the DECODER is the "
        "lossy stage -- a different sentence for (c), and the "
        "SCUT-normal prior gains a MEASURED mechanism: the decoder "
        "pulls toward its prior even where the representation carries "
        "the patient's own geometry"
    ),
    "threshold_declared": (
        "substantial_pcc = 0.10, DECLARED IN THE CONFIG before the "
        "number: the P1 convention, and >= 3x the five-seed sd these "
        "pod probes have shown (P1's was 0.0228) -- a mean at or "
        "above it cannot be seed noise. The P1 boundary-honesty "
        "precedent applies: if the mean lands at the threshold, the "
        "verdict says so with the spread"
    ),
    "limitation_rides_either_way": (
        "the instrument limitation carried on (c) RIDES ALONG "
        "unchanged: the mirror scalar barely tracks grade in the "
        "originals (r +0.0131), so this probes whether the embedding "
        "CARRIES the scalar, not whether the scalar matters "
        "clinically -- neither reading touches the clinical question"
    ),
    "status": (
        "DESCRIPTIVE, report-never-gate, no ladder entry, no ledger "
        "row; applied through READING_COUNT_GUARD at 237"
    ),
}


#: **[BANKED 2026-08-24] THE ASYMMETRY-ENCODING PROBE'S VERDICT: THE
#: ASYMMETRY IS IN THE EMBEDDING; THE DECODER IS THE LOSSY STAGE.**
ASYMMETRY_ENCODING_PROBE_BANKED = {
    "banked": "2026-08-24, p13-asym-encoding, single clean pod attempt",
    "figures": (
        "mean OOF PCC +0.4856 (sd 0.0231), seeds spanning 0.4463 to "
        "0.5013, against the DECLARED threshold 0.10 -- the second "
        "registered reading fires DECISIVELY, nearly 5x the threshold "
        "and 17x above zero in seed-sd units"
    ),
    "reading_fired": (
        "the asymmetry IS in the embedding and the DECODER is the "
        "lossy stage (ASYMMETRY_ENCODING_PROBE_REGISTERED"
        "['reading_if_substantial'], applied at 237/237 through the "
        "count guard)"
    ),
    "c_sentence_upgraded": (
        "(c)'s sentence upgrades, dated in place: from 'the chain "
        "loses it' to 'ENCODED AT r~0.49, RENDERED AT r=0.0082 -- the "
        "decoder discards per-patient geometry the representation "
        "carries'"
    ),
    "prior_gains_mechanism": (
        "SCUT_NORMAL_PRIOR gains its MEASURED mechanism: the decoder "
        "pulls toward its prior even where the representation carries "
        "the patient's own geometry -- no longer an inference from the "
        "worst-24, a measurement"
    ),
    "methods_caution_for_the_writeup": (
        "**BY NAME, for the write-up's methods-caution section**: "
        "decoder-based evidence is a LOWER BOUND on what a "
        "representation carries -- the embedding is a richer witness "
        "than its reconstruction, demonstrated at 0.49-vs-0.008 on "
        "this cohort. the maintainer's question ('why not feature-space "
        "probing?') exposed the conflation; the probe answered it in "
        "ONE POD RUN"
    ),
    "limitation_restated_verbatim": (
        "the instrument limitation carried on (c) RIDES ALONG "
        "unchanged: the mirror scalar barely tracks grade in the "
        "originals (r +0.0131), so this probes whether the embedding "
        "CARRIES the scalar, not whether the scalar matters "
        "clinically -- neither reading touches the clinical question. "
        "The lips-dominant-plus-floor account of the 0.25 stands "
        "UNTOUCHED by this result"
    ),
    "status": "DESCRIPTIVE, no ledger row",
}


def summary() -> dict:
    """The phase's records, importable as one object."""
    return {
        "registered": PHASE_13_REGISTERED,
        "failure_mode": FAILURE_MODE_READING_BOUND,
        "exit_criteria": PHASE_13_EXIT_CRITERIA,
        "regional_readings": REGIONAL_COMPARISON_READINGS,
        "architecture": DECODER_ARCHITECTURE_PROPOSED,
        "shakedown": SCUT_SHAKEDOWN_PLAN,
        "gate": CLEFT_RECONSTRUCTION_GATE,
        "stops": PHASE_13_STOPS,
        "double_difference": DOUBLE_DIFFERENCE_RULE,
        "stop_1": STOP_1_BUILT,
        "parameter_band": PARAMETER_BAND_MISESTIMATED,
        "stop_1_reviewed": STOP_1_REVIEWED,
        "scut_normal_prior": SCUT_NORMAL_PRIOR,
        "stop_2": STOP_2_BUILT,
        "stop_2_reviewed": STOP_2_REVIEWED,
        "coarse_severity_hypothesis": COARSE_SEVERITY_HYPOTHESIS,
        "stop_3": STOP_3_REGISTERED,
        "everywhere_gap": EVERYWHERE_GAP_IS_THE_LARGER_FACT,
        "reading_applied_to_nobody": READING_APPLIED_TO_NOBODY,
        "reading_count_guard": READING_COUNT_GUARD,
        "writer_died_after_save": WRITER_DIED_AFTER_SAVE,
        "anchor_quantities_differ": ANCHOR_QUANTITIES_DIFFER,
        "stop_3_banked": STOP_3_BANKED,
        "probe": PROBE_ON_RECONSTRUCTIONS,
        "closing": PHASE_13_CLOSING,
        "closing_addendum": CLOSING_ADDENDUM_REGISTERED,
        "p1_banked": P1_CONFOUND_CEILING_BANKED,
        "band_names": BAND_NAME_CORRESPONDENCE,
        "p2_banked": P2_BAND_OCCLUSION_BANKED,
        "p3_triggered": P3_TRIGGERED,
        "p3_banked": P3_BANKED,
        "asym_encoding_probe": ASYMMETRY_ENCODING_PROBE_REGISTERED,
        "asym_probe_banked": ASYMMETRY_ENCODING_PROBE_BANKED,
    }
