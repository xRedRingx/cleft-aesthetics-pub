"""Phase 12 -- the view ablation (frontal / basal / both).

Opened 2026-08-23 under ``phase11.PHASE_SEQUENCE_RENUMBERED`` (the
ablation took the decoder's slot; the decoder moved to Phase 13, the
write-up to 14).

**The premise is UNRESOLVED and the design is built to be informative
under either resolution** (``ladder.BASAL_RATIONALE_UNSUPPORTED``): the
frontal arm sees what the raters' scores are keyed to; the basal view
exists for 236 of 237 patients and no basal ID is scored. the supervision answer to
ask-list question 8 decides whether adding basal RESTORES evidence the
label already contains or MEASURES what a view the raters never saw adds.
Both readings are live, the arms are identical either way, and both
interpretation sentences are registered below -- before any number.
"""

from __future__ import annotations


class Phase12Error(RuntimeError):
    """A Phase 12 contract is not usable."""


#: **[REGISTERED 2026-08-23, BEFORE ANY BUILD] THE PHASE, WHOLE.**
PHASE_12_REGISTERED = {
    "registered": "2026-08-23",
    "question": (
        "with the correction in force: the frontal arm sees what the "
        "raters' scores are keyed to; the basal view exists for 236 of "
        "237 patients and no basal ID is scored. Does adding the basal "
        "view change what this cohort's models can do?"
    ),
    "premise_status": (
        "UNRESOLVED (ladder.BASAL_RATIONALE_UNSUPPORTED) -- and the "
        "design is informative under either resolution: the arms are "
        "identical either way, only the interpretation sentence differs, "
        "and both sentences are registered now "
        "(INTERPRETATION_SENTENCES_PREWRITTEN)"
    ),
    "cohort": {
        "n": 236,
        "rule": (
            "every patient with both views. Folder 238 (frontal only) is "
            "excluded from ALL arms of this phase, the frontal-only arm "
            "INCLUDED, so every contrast is within-cohort and one-factor"
        ),
        "exception_pairing": "folder 143: 524 = frontal, 523 = basal",
        "manifest": (
            "cleft_v1_views -- derived from cleft_v1 (whose pairing came "
            "from score-sheet lookup), rows and FOLDS carried verbatim, "
            "the two known exceptions asserted by test (data/views.py)"
        ),
    },
    "arms": {
        "shared": (
            "frozen ImageNet ViT-B/16 + linear head, G1, five seeds, the "
            "standard criterion"
        ),
        "a_frontal_only_236": (
            "the baseline RE-RUN on the 236 cohort. Not reused from the "
            "237 runs -- the cohort changed, so the bar must be "
            "re-measured on it"
        ),
        "b_basal_only_236": "768-d basal embedding, 769-parameter head",
        "c_two_view_concat": (
            "frontal (+) basal, 1536-d, 1537-parameter head"
        ),
        "d_capacity_control": (
            "frontal (+) frontal -- the SAME vector duplicated: 1536-d, "
            "1537-parameter head. D has C's capacity and A's information. "
            "C > A with C ~ D refutes the view explanation in favour of "
            "capacity; C > D isolates the view. Cheap, and without it "
            "C vs A is confounded"
        ),
    },
    "paired_scope": (
        "B, C, D against A through the standard paired machinery, "
        "generated through the existing generator, nothing hand-built"
    ),
    "stops": {
        "1": "manifest cleft_v1_views + its tests",
        "2": (
            "basal staging + review sheets -- the visual-check gate. The "
            "staging decision for basal (own trapezium parameters or a "
            "measured argument that the frontal ones transfer), staged at "
            "G1, review sheets of all 236 staged basal crops reviewed BY "
            "EYE on the cluster before any embedding is extracted. The "
            "registered rule stands: sheets are reviewed by eye or not at "
            "all"
        ),
        "3": "the four arms' tasks and configs through the generator",
        "4": "the paired scope",
        "rule": "stop after each for review",
    },
    "ethics_gate": (
        "config-level, built at stop 2: the REC approvals' crop "
        "conditions were written around the rated frontal view, and using "
        "the captured-but-unrated submental view is a DIFFERENT USE of "
        "the same images. The staging task refuses without an explicit "
        "basal_use_acknowledged flag that the maintainer must set -- the "
        "confirmation, not the code's assumption. The gate attaches at "
        "first PIXEL use (staging), not at ID bookkeeping: cleft_v1 has "
        "carried basal IDs since Phase 1"
    ),
}


#: **[REGISTERED 2026-08-23, BOTH SENTENCES BEFORE ANY NUMBER] The
#: interpretation of a positive result depends on the supervision answer to
#: question 8, and BOTH sentences are pre-written so neither is composed
#: after the numbers arrive.**
INTERPRETATION_SENTENCES_PREWRITTEN = {
    "registered": "2026-08-23",
    "applies_when": (
        "reading 1 fires -- C > A claimable AND C > D. Under any other "
        "outcome neither sentence is used"
    ),
    "if_raters_saw_both": (
        "adding the basal view RESTORES evidence the label already "
        "contains: the score was given over both views, and a "
        "frontal-only model was predicting a label from evidence it "
        "could not see"
    ),
    "if_raters_saw_frontal_only": (
        "adding the basal view adds signal the RATERS NEVER SAW: the "
        "model exceeds the evidence basis of its own labels' provenance, "
        "and the gain is anatomical correlation, not label restoration"
    ),
    # [2026-08-23, as registered] Preserved exactly as written. The
    # selection below did not amend either sentence.
    "neither_is_chosen_here": (
        "the supervision answer to ask-list question 8 selects the sentence; the "
        "arms and every number are identical under both"
    ),
    "selected_2026_08_31": "SENTENCE_SELECTED",
}


#: **[SELECTED 2026-08-31] ``if_raters_saw_frontal_only`` SELECTS, on
#: documentary evidence.**
#:
#: The registration said the supervision answer to question 8 would select the
#: sentence. **The manuscripts answered it first**
#: (``RATING_PROCEDURE_DOCUMENTED``): each rater independently scored one
#: cropped Frontal-Eye-View image; no basal, submental, profile or
#: lateral view was shown. So the selected sentence is:
#:
#:     "adding the basal view adds signal the RATERS NEVER SAW: the
#:      model exceeds the evidence basis of its own labels' provenance,
#:      and the gain is anatomical correlation, not label restoration"
#:
#: **BOTH SENTENCES ARE PRESERVED UNAMENDED** in
#: ``INTERPRETATION_SENTENCES_PREWRITTEN``. Selecting is not editing, and
#: a reader must be able to see what the other branch would have said.
#:
#: **NOTHING MEASURED CHANGES.** The registration promised this and it
#: holds: *"the arms and every number are identical under both"*. No arm
#: was re-run, no figure moves, and no ledger row is touched. What
#: changes is one interpretive sentence.
#:
#: **What it means for C.** The concat-over-frontal gain -- **0.2640 vs
#: 0.2505, +0.0135** -- was already inside the resolution floor
#: (``ladder.COHORT_CANNOT_RESOLVE``: 0.04-0.10 unresolvable; the
#: smallest ever resolved is 0.1386). Under the selected sentence it
#: reads simply as **an extra view helping marginally or not at all,
#: UNRESOLVED** -- not as a frontal-only model recovering evidence its
#: labels already contained. The registered reading 2 (descriptive only,
#: condition 1 fails) is unchanged and remains what fired.
SENTENCE_SELECTED = {
    "selected": "2026-08-31, on documentary evidence",
    "which": "if_raters_saw_frontal_only",
    "the_basis": (
        "RATING_PROCEDURE_DOCUMENTED -- the supervisor's own manuscripts "
        "describe the procedure directly: each rater independently "
        "scored ONE cropped Frontal-Eye-View image; no basal, submental, "
        "profile or lateral view was shown to anyone"
    ),
    "the_selected_sentence": (
        "adding the basal view adds signal the RATERS NEVER SAW: the "
        "model exceeds the evidence basis of its own labels' provenance, "
        "and the gain is anatomical correlation, not label restoration"
    ),
    "both_preserved_unamended": (
        "INTERPRETATION_SENTENCES_PREWRITTEN keeps BOTH sentences "
        "byte-unchanged. Selecting is not editing, and a reader must be "
        "able to see what the other branch would have said"
    ),
    "nothing_measured_changes": (
        "**the registration promised exactly this and it holds**: 'the "
        "arms and every number are identical under both'. No arm was "
        "re-run, no figure moves, no ledger row is touched. What changes "
        "is ONE INTERPRETIVE SENTENCE"
    ),
    "what_it_means_for_arm_c": (
        "the concat-over-frontal gain -- 0.2640 vs 0.2505, +0.0135 -- "
        "was ALREADY inside the resolution floor "
        "(ladder.COHORT_CANNOT_RESOLVE: 0.04-0.10 measured unresolvable; "
        "the smallest ever resolved is 0.1386). Under the selected "
        "sentence it reads simply as AN EXTRA VIEW HELPING MARGINALLY OR "
        "NOT AT ALL, UNRESOLVED -- not as a frontal-only model "
        "recovering evidence its labels already contained"
    ),
    "the_fired_reading_is_unchanged": (
        "PHASE_12_READINGS['2_descriptive_only'] is what fired and stays "
        "fired: C > A descriptively, condition 1 fails. The selection "
        "changes the INTERPRETATION SENTENCE that reading 1 would have "
        "carried, not which reading fired"
    ),
}


#: **[LITERATURE-primary, via NotebookLM against the supervisor's own source
#: documents, the maintainer 2026-08-31] THE RATING PROCEDURE, DESCRIBED
#: DIRECTLY BY THE MANUSCRIPTS.**
#:
#: **Each rater independently scored ONE CROPPED FRONTAL-EYE-VIEW image
#: per patient. No basal, submental, profile or lateral view was shown to
#: anyone.**
#:
#: This is what the earlier evidence could not reach. The sheet
#: measurement and the three instrument primaries could establish only
#: that **nothing distinguished frontal-only rating from both-shown**
#: (``ladder.BASAL_RATIONALE_UNSUPPORTED["unresolved"]``); the
#: manuscripts **describe the procedure itself**, and settle it.
#:
#: **Quoted verbatim, never paraphrased**, with each source document
#: named. Every quote below is asserted present in this record by
#: ``tests/test_phase12.py``.
RATING_PROCEDURE_DOCUMENTED = {
    "tag": (
        "[LITERATURE-primary, via NotebookLM against the supervisor's own "
        "source documents, the maintainer 2026-08-31]"
    ),
    "the_finding": (
        "each rater independently scored ONE CROPPED FRONTAL-EYE-VIEW "
        "image per patient. NO BASAL, SUBMENTAL, PROFILE OR LATERAL VIEW "
        "WAS SHOWN TO ANYONE"
    ),

    "bruce_ready_draft": {
        "document": "Bruce-ready version - minus Jonathan's edits.docx",
        "what_it_is": "the CleftGNN draft",
        "data_source_ethics_and_image_preprocessing": (
            "The primary image dataset comprised 181 standardised "
            "frontal two-dimensional facial photographs of five-year-old "
            "children who had undergone primary surgical repair for cUCL."
        ),
        "the_crop": (
            "The resulting cropped Frontal-Eye-View (FEV) included the "
            "medial canthi, nose and upper lip region."
        ),
        "human_assessor_panel": (
            "Each cropped image was independently scored by a "
            "multidisciplinary panel of five cleft professionals... "
            "Assessors scored postoperative facial appearance using a "
            "five-point VAS."
        ),
    },

    "latest_version_draft": {
        "document": (
            "The latest version - mid re-write post Physicists comments "
            "+ BR tracked changes.docx"
        ),
        "same_passages": (
            "the same FEV and the same scoring passages as the "
            "Bruce-ready draft"
        ),
        "figure_2_legend": (
            "Expert cleft professionals evaluate 2D cropped, Frontal eye "
            "View (FEV) photographs (depicting the post-operative state) "
            "and each assign a ground-truth Visual Analogue Score (VAS)."
        ),
    },

    "bmj_open": {
        "document": "bmjopen-15-8.pdf",
        "facial_appearance": (
            "A two-dimensional assessment of the child's face was made "
            "using frontal photographs... The images were anonymised and "
            "cropped to allow unbiased assessment of only the nose and "
            "lip area."
        ),
    },

    "bakaki_thesis": {
        "document": "Paul_Bakaki_Final_Thesis (1).pdf",
        "section_4_3_1": (
            "the facial images partially only reveal the nose and "
            "mouth/lips."
        ),
    },

    # -----------------------------------------------------------------
    # [VERIFIED AT SOURCE 2026-08-31, by this session, INDEPENDENTLY of
    # the NotebookLM route]
    #
    # Two of the four documents are reachable from this machine. Their
    # text was extracted from the .docx XML and each quote attributed to
    # them was searched for literally: ALL SEVEN FOUND, whitespace- and
    # punctuation-normalised only.
    #
    # A term scan over the same extracted text adds a NEGATIVE that no
    # quote can carry: across both manuscripts, "submental", "profile
    # view", "lateral view" and "three-view" appear ZERO times, and
    # "basal" appears exactly ONCE -- in a bibliography entry, as the
    # author surname "Basalamah, A." It is not a view mention. So the
    # manuscripts do not merely fail to mention a second view; they
    # contain no non-frontal view term at all.
    #
    # The two PDFs are NOT reachable here, so their quotes rest on an
    # off-machine NotebookLM verification alone. Recorded as such rather
    # than folded in with the verified ones.
    # -----------------------------------------------------------------
    "verified_at_source_2026_08_31": {
        "by": "this session, independently of the NotebookLM route",
        "method": (
            "text extracted from the .docx XML; each attributed quote "
            "searched for literally, whitespace- and punctuation-"
            "normalised only"
        ),
        "docx_quotes_found": "ALL SEVEN -- four from the Bruce-ready "
                             "draft, three from the latest version",
        "the_negative_no_quote_can_carry": (
            "across BOTH manuscripts 'submental', 'profile view', "
            "'lateral view' and 'three-view' appear ZERO times, and "
            "'basal' appears exactly ONCE -- in a bibliography entry, as "
            "the author surname 'Basalamah, A.', not a view. The "
            "manuscripts do not merely fail to mention a second view: "
            "they contain no non-frontal view term at all"
        ),
        "not_verified_here": (
            "bmjopen-15-8.pdf and Paul_Bakaki_Final_Thesis (1).pdf are "
            "NOT reachable from this machine. Their quotes rest on an "
            "off-machine NotebookLM verification alone, and are recorded as "
            "such rather than folded in with the verified ones"
        ),
    },
}


#: **[2026-08-31] SUPERVISOR QUESTION 8: DOCUMENTARILY ANSWERED, PENDING
#: IN-PERSON CONFIRMATION. NOT CLOSED.**
#:
#: The question, as asked (``ladder.BASAL_RATIONALE_UNSUPPORTED
#: ["for_supervisor"]``): *"When the five raters scored these 251 patients, were
#: they shown the frontal photograph only, or the frontal and submental
#: together, with one score recorded under the frontal's ID?"*
#:
#: **The manuscripts answer it: the frontal photograph only**, one
#: cropped Frontal-Eye-View per patient, scored independently by each of
#: five professionals (``RATING_PROCEDURE_DOCUMENTED``, with the quotes
#: attached).
#:
#: **It is NOT closed outright, and the reason is not politeness.** The
#: manuscripts describe the procedure their authors wrote down. the supervision material ran
#: the panel and may know it was shown material the manuscripts do not
#: describe -- a contact sheet, a second image on screen, an
#: unrecorded practice. A document is evidence about a procedure, not
#: the procedure. So the question becomes a **confirmation**: the quotes
#: travel with it, and supervision is asked to confirm or correct rather than to
#: recall unaided.
SUPERVISOR_QUESTION_8_DOCUMENTARILY_ANSWERED = {
    "status": (
        "DOCUMENTARILY ANSWERED, PENDING IN-PERSON CONFIRMATION -- NOT "
        "CLOSED"
    ),
    "answered": "2026-08-31, by RATING_PROCEDURE_DOCUMENTED",
    "the_question_as_asked": (
        "When the five raters scored these 251 patients, were they shown "
        "the frontal photograph only, or the frontal and submental "
        "together, with one score recorded under the frontal's ID?"
    ),
    "the_documentary_answer": (
        "**the frontal photograph only** -- one cropped "
        "Frontal-Eye-View per patient, scored independently by each of "
        "five cleft professionals"
    ),
    "why_it_is_not_closed_outright": (
        "**the manuscripts describe the procedure their authors WROTE "
        "DOWN.** the supervision material ran the panel and may know it was shown material "
        "the manuscripts do not describe -- a contact sheet, a second "
        "image on screen, an unrecorded practice. A DOCUMENT IS EVIDENCE "
        "ABOUT A PROCEDURE, NOT THE PROCEDURE"
    ),
    "what_to_put_to_supervision": (
        "the quotes, attached, so the ask is a CONFIRMATION rather than "
        "a request to recall unaided: 'the manuscripts say each rater "
        "scored one cropped Frontal-Eye-View and no other view -- is "
        "that what happened, or was the panel shown anything else?'"
    ),
    "what_turns_on_the_answer": (
        "**nothing measured.** If it is confirmed, the selection stands "
        "unchanged; if it is corrected, SENTENCE_SELECTED is re-selected "
        "the other way and the arms and every figure are STILL "
        "identical -- that is what the registration guaranteed by "
        "pre-writing both"
    ),
}


#: **[RECORDED 2026-08-31, DATED] WHERE THE TWO-VIEW CLAIM CAME FROM, and
#: how long it survived.** Named on the same footing as
#: ``ladder.THE_ERROR_PROVENANCE``, ``phase20.CROSS_TARGET_ERROR_PROVENANCE``
#: and ``phase20.S_DESCRIPTION_ERROR_PROVENANCE``.
#:
#: **The error.** *"Two views carry the label -- raters saw frontal and
#: basal together"* entered as **`[MEASURED]`** in the GOVERNING
#: DOCUMENT (``docs/PLAN.md`` §4.6, line 861). It cited no source and was
#: never measured: it was reasoned from a plausible account of how the
#: score sheet must have worked.
#:
#: **How long it survived, and where.** From the PLAN's writing until
#: 2026-08-23, when the sheet measurement downgraded it to UNSUPPORTED --
#: **but the PLAN line itself was never corrected**, so the governing
#: document carried the `[MEASURED]` tag for a further eight days. And
#: **the claim was repeated in discussion without being checked against the record as recently as this
#: session** while the downgrade was already on the record.
#:
#: **How it was caught.** **the challenge was it and checked the
#: primary sources.** Not a test, not a sweep -- a person disbelieving a
#: sentence and going to the documents.
#:
#: **Its own recorded lesson applies to itself.**
#: ``BASAL_RATIONALE_UNSUPPORTED["lesson"]``: *verification effort flows
#: to the items tagged REASONED, so a mis-tagged rationale is precisely
#: the claim nobody audits.* This is the **sixth instance** of that
#: class, and it demonstrated the lesson twice -- once by surviving
#: mis-tagged, and once by surviving eight more days in the governing
#: document AFTER being downgraded everywhere else.
TWO_VIEW_CLAIM_PROVENANCE = {
    "recorded": "2026-08-31",
    "the_claim": (
        "'Two views carry the label -- raters saw frontal and basal "
        "together', docs/PLAN.md 4.6 line 861"
    ),
    "how_it_entered": (
        "as **[MEASURED]** in the GOVERNING DOCUMENT. It cited NO SOURCE "
        "and was NEVER MEASURED: reasoned from a plausible account of "
        "how the score sheet must have worked"
    ),
    "how_long_it_survived": (
        "from the PLAN's writing until 2026-08-23, when the sheet "
        "measurement downgraded it to UNSUPPORTED -- **but the PLAN line "
        "was never corrected**, so the governing document carried the "
        "[MEASURED] tag for a further EIGHT DAYS, until 2026-08-31"
    ),
    "it_was_repeated": (
        "**in conversation as recently as this session**, while the "
        "downgrade was already on the record. The record was right and "
        "the repetition was not checked against it"
    ),
    "how_it_was_caught": (
        "**the challenge was it and checked the primary sources.** Not "
        "a test, not a sweep -- a person disbelieving a sentence and "
        "going to the documents"
    ),
    "its_own_lesson_applied_to_itself": (
        "BASAL_RATIONALE_UNSUPPORTED['lesson']: verification effort "
        "flows to items tagged REASONED, so a mis-tagged rationale is "
        "precisely the claim nobody audits. **The SIXTH instance of that "
        "class** -- and it demonstrated the lesson TWICE: once by "
        "surviving mis-tagged, and once by surviving eight more days in "
        "the governing document AFTER being downgraded everywhere else"
    ),
    "the_second_lesson": (
        "**a correction that does not reach the governing document is "
        "not finished.** The code record had it right from 2026-08-23; "
        "the PLAN's precedence rule (code > this document > memory) "
        "meant nothing downstream was wrong -- and a reader starting "
        "from the governing document still got [MEASURED]"
    ),
}


#: **[REGISTERED 2026-08-23, BEFORE ANY NUMBER] THE FOUR READINGS.**
PHASE_12_READINGS = {
    "registered": "2026-08-23",
    "1_view_signal": (
        "C > A claimable (BOTH conditions) and C > D -> basal carries "
        "signal beyond capacity; the interpretation sentence then depends "
        "on the supervision answer, both sentences pre-written "
        "(INTERPRETATION_SENTENCES_PREWRITTEN)"
    ),
    "2_descriptive_only": (
        "C > A descriptively but condition 1 fails -- THE REGISTERED "
        "PRIOR, on the 7D concat precedent (+0.0073, withdrawn) and the "
        "four convergent neighbourhood nulls"
    ),
    "3_no_gain": (
        "C ~ A or C < A -> the missing-view explanation stays dead, and "
        "the record already says so "
        "(ladder.BASAL_RATIONALE_UNSUPPORTED)"
    ),
    "4_b_near_a": (
        "B alone near A would be SURPRISING and gets no story until "
        "measured"
    ),
    "capacity_control_logic": (
        "C > A with C ~ D refutes the view explanation in favour of "
        "capacity; C > D isolates the view -- D is the reading that "
        "makes C vs A attributable at all"
    ),
}


#: **[BUILT 2026-08-23, STOP 1 OF 4] THE VIEWS MANIFEST.**
#:
#: ``data/views.py`` derives ``cleft_v1_views`` from ``cleft_v1``: 237
#: rows in, 236 out, folder 238 dropped, everything else -- labels, soft
#: labels, class3, and the FOLD column -- carried VERBATIM.
#:
#: **The pairing is carried, not re-derived.** ``cleft_v1``'s frontal came
#: from score-sheet lookup and its basal is the folder's other image; a
#: second implementation of that lookup is the R10 duplication, and it is
#: also how folder 143's views could quietly swap. The derivation ASSERTS
#: instead: 143 must be 524-frontal / 523-basal, 238 must be the only
#: single-view patient, every kept row must have two distinct views, and
#: the source must be exactly the 237-row artifact Phase 1 measured.
#:
#: **Folds carried verbatim, and the reason recorded**: re-stratifying on
#: 236 would reshuffle every assignment, making "arm A on 236 vs the
#: 0.2520 arm on 237" differ by a fold reshuffle AND a patient -- two
#: factors where the design wants one. Carrying them, the baseline
#: re-measurement differs from the ladder by exactly one patient's
#: removal, and all four arms share one fold structure by construction.
#:
#: **Same columns, same order as cleft_v1** -- every existing loader reads
#: the derived artifact unchanged; only ``basal_id``'s contract tightens
#: (never empty here).
STOP_1_MANIFEST = {
    "built": "2026-08-23, stop 1 of 4",
    "artifact": "data/manifests/cleft_v1_views",
    "derivation": (
        "cleft_v1 -> drop folder 238 -> 236 rows verbatim; pairing "
        "CARRIED from the artifact that measured it, never re-derived"
    ),
    "asserted": (
        "source is exactly 237 rows with unique patient ids",
        "exactly one patient lacks a basal, and it is 238",
        "folder 143 is 524-frontal / 523-basal",
        "every kept row has two distinct views",
        "derived counts are 236/236/236",
        "no fold vanishes in the drop",
    ),
    "folds": (
        "carried verbatim from cleft_v1 -- one factor, not two, between "
        "arm A on 236 and the ladder's 0.2520 on 237"
    ),
    "columns": (
        "cleft_v1's, same names and order, so every existing loader "
        "reads it unchanged; basal_id is never empty here"
    ),
    "config": "configs/p12_views_manifest.yaml, generated, hash carried",
    "next_stop": (
        "2 -- basal staging + review sheets, gated by the visual check and "
        "the basal_use_acknowledged flag"
    ),
}


#: **[DECIDED 2026-08-23, STOP 2 -- MEASURED FROM THE CODE PATH, WITH THE
#: PIXEL HALF DEFERRED TO REGISTERED MEASUREMENTS] THE BASAL STAGING
#: DECISION.**
#:
#: The registered choice was "own trapezium parameters for the submental
#: view, or a measured argument that the frontal parameters transfer".
#: **Measurement shows it is a false choice for this phase's arms: at
#: whole-image G1, NO trapezium parameter is consumed at all.**
#:
#: **The code-path measurement, checkable in place:**
#:
#: * the staged tensor is ``staging.stage(image)`` alone --
#:   pad-square-white-centre-resize. Its only input fact about the image
#:   is the ASPECT RATIO; it references no landmark, no midline, no
#:   anatomy (``stage_build.build``: ``base.image if geometry == "g1"``).
#: * the trapezium's frontal half-widths (0.301 / 0.500) are consumed in
#:   exactly two places: **G2's ``unwarp``** and the **patch
#:   generators** -- and Phase 12's four arms are whole-image G1, using
#:   neither.
#: * the whole-image embedding path reads ``staged_patient_g1.npy``
#:   directly; no mask is applied between staging and the backbone.
#:
#: So the frontal parameters do not "transfer" -- **they are not
#: exercised**. The honest form of the decision is the third option the
#: question didn't list: no anatomical parameter enters basal staging,
#: and asserting that from the code path is a measurement, not a
#: rationale (the Phase 12 lesson applied: the sixth-instance error was a
#: rationale tagged MEASURED; this is the reverse -- a claim that is
#: checkable by reading the executed path).
#:
#: **THE BOUNDARY, registered with the decision**: this covers
#: WHOLE-IMAGE G1 ONLY. Any region/patch work or any G2 unwarp on basal
#: images consumes the trapezium, whose parameters are FRONTAL
#: measurements -- submental geometry (nostril-dominant, different
#: midline cues, different white-corner placement) was never
#: characterised, and such a use REOPENS this decision and requires its
#: own characterisation first. The boundary is asserted in the staging
#: task: it writes no G2 tensor and no patch definitions.
#:
#: **WHAT ``stage()`` DOES ASSUME, measured on frontals only -- two
#: facts, deferred to the pixel run with thresholds registered before any
#: basal pixel is read** (``geometry/basal.py``):
#:
#: 1. **White-pad continuity.** Pad value 255 was chosen because the
#:    frontal crops' corners "already arrive white, baked in at source".
#:    Whether basal crops share that is UNKNOWN from the laptop. Measured
#:    per image (four 16x16 source corners, white at >= 250/channel);
#:    an image below 0.90 is FLAGGED on the sheets; **if the cohort
#:    MEDIAN is below 0.90 the rationale does not transfer and this
#:    decision REOPENS before any embedding is extracted.**
#: 2. **The aspect-ratio distribution.** 0.553-1.099 is a frontal
#:    measurement; the basal distribution is uncharacterised. ``stage()``
#:    is total over any ratio -- nothing can break -- so this is
#:    REPORTED beside the frontal range and put in front of the eye,
#:    never refused by code.
STOP_2_STAGING_DECISION = {
    "decided": "2026-08-23, stop 2",
    "registered_choice": (
        "own trapezium parameters for the submental view, or a measured "
        "argument that the frontal parameters transfer"
    ),
    "measured_answer": (
        "a false choice for this phase's arms: at whole-image G1 NO "
        "trapezium parameter is consumed at all. stage() is "
        "pad-square-white-centre-resize, whose only input fact is the "
        "aspect ratio; the frontal half-widths (0.301/0.500) are "
        "consumed only by G2's unwarp and the patch generators, and the "
        "four arms use neither. The frontal parameters do not "
        "'transfer' -- they are NOT EXERCISED"
    ),
    "why_this_is_a_measurement": (
        "checkable by reading the executed path (stage_build.build: "
        "base.image if geometry == 'g1'; load_inputs reads "
        "staged_patient_g1.npy with no mask between staging and the "
        "backbone) -- the reverse of the sixth-instance error, which was "
        "a rationale tagged MEASURED"
    ),
    "boundary": (
        "WHOLE-IMAGE G1 ONLY. Region/patch work or G2 unwarp on basal "
        "images consumes the trapezium, whose parameters are FRONTAL "
        "measurements; submental geometry (nostril-dominant, different "
        "midline cues) was never characterised, and such a use REOPENS "
        "this decision. Asserted in the task: no G2 tensor, no patch "
        "definitions are written"
    ),
    "deferred_to_pixels": {
        "white_pad_continuity": (
            "pad 255 matches corners 'already white, baked in at source' "
            "-- a FRONTAL measurement. Measured per basal image (four "
            "16x16 source corners, white >= 250/channel); < 0.90 flags "
            "the image on the sheets; cohort MEDIAN < 0.90 REOPENS the "
            "decision before any embedding (geometry/basal.py, "
            "thresholds registered before any pixel)"
        ),
        "aspect_ratio": (
            "0.553-1.099 is a frontal measurement; the basal "
            "distribution is uncharacterised. stage() is total over any "
            "ratio, so this is REPORTED beside the frontal range and put "
            "in front of the eye, never refused by code"
        ),
    },
}


#: **[REGISTERED 2026-08-23] THE ETHICS GATE, AS BUILT.**
#:
#: The REC approvals' crop conditions were written around the RATED
#: frontal view. The submental view was captured under the same
#: approvals but never rated, and using it is a DIFFERENT USE of the
#: same images -- a judgement for the named investigator, not for code.
#:
#: **The mechanism**: ``basal_use_acknowledged`` is a required boolean in
#: the staging config with NO default. The generator writes ``false`` on
#: a fresh render and CARRIES a ``true`` only from the shipped config --
#: the same three-state carry as a filled hash -- so the only way the
#: flag becomes true is the maintainer editing the config, and that edit
#: survives regeneration with ``--check`` clean. The staging task refuses
#: before touching any pixel while the flag is false, naming the maintainer and the
#: reason in the refusal.
#:
#: **The gate attaches at first PIXEL use** -- the stop-2 staging task --
#: not at ID bookkeeping: ``cleft_v1`` has carried basal IDs since
#: Phase 1, and stop 1's derivation touched no image.
BASAL_USE_GATE = {
    "registered": "2026-08-23",
    "why": (
        "the REC approvals' crop conditions were written around the "
        "RATED frontal view; using the captured-but-unrated submental "
        "view is a DIFFERENT USE of the same images -- a judgement for "
        "the named investigator, not for code"
    ),
    "mechanism": (
        "basal_use_acknowledged: required boolean, NO default. The "
        "generator writes false on a fresh render and carries a true "
        "only from the shipped config (the three-state carry), so the "
        "flag becomes true only by the edit, and the edit "
        "survives regeneration with --check clean. The task refuses "
        "before touching any pixel while it is false"
    ),
    "attaches_at": (
        "first PIXEL use -- the stop-2 staging task. cleft_v1 has "
        "carried basal IDs since Phase 1; stop 1 touched no image"
    ),
    # **[ACKNOWLEDGED 2026-08-23, the maintainer, commit 2e43cbd]** The flag
    # was set to true in the shipped config under REC 13/SW/0064 and
    # 23/YH/0037 -- that edit, carried by the generator exactly as the
    # mechanism promised. The gate worked end to end: code wrote false,
    # refused while false, and only that edit opened it.
    "acknowledged": (
        "2026-08-23, the maintainer, commit 2e43cbd, under REC 13/SW/0064 and "
        "23/YH/0037. The generator carried the edit; the task now runs"
    ),
    # The one defect the acknowledgement surfaced, recorded because it is
    # a test-design lesson rather than a gate lesson: two suite tests had
    # PINNED the flag at false -- pinning a value that was always the
    # maintainer's to change -- so that acknowledgement broke CI. The header
    # also lagged that edit by one regeneration, because the generated
    # comment depends on the flag's value. Both fixed the same day: the
    # tests now assert CONSISTENCY (the flag is a boolean and the header
    # describes the state the body carries), not the value.
    "ci_failure_on_acknowledgement": (
        "2026-08-23: two tests pinned the flag at false and broke when El "
        "the maintainer set it -- the pin was on a value that was THEIRS "
        "to change, "
        "and the invariant was consistency, not the value. Fixed with the "
        "header regeneration the same day"
    ),
}


#: **[BUILT 2026-08-23, STOP 2 OF 4 -- NOT LAUNCHED] BASAL STAGING AND
#: THE REVIEW SHEETS.**
#:
#: ``stage_basal_views``: gate first, then all 236 basal images through
#: the FROZEN ``staging.stage()`` unchanged, writing
#: ``data/staged/staged_basal_v1`` (G1 tensor + geometry.csv +
#: MANIFEST.json, same layout as ``staged_v1``'s G1 half) and the review
#: sheets.
#:
#: **What the sheets show**: every one of the 236 staged basal crops --
#: not a sample -- 24 per sheet, 6 columns, mid-grey ground (the render
#: module's own rationale: white-padded crops on a white ground would
#: hide where an image ends), each panel labelled with patient id, basal
#: id, aspect ratio, and corner-white fraction, with a ``FLAG`` suffix
#: below 0.90. **Ordered by ascending corner-white fraction, so the most
#: suspicious images land on sheet 1, in front of the eye first.**
#:
#: **The rule stands**: sheets are reviewed by eye or not at all, and NO
#: EMBEDDING IS EXTRACTED before the maintainer's pass. The run's metrics stamp
#: ``review: PENDING`` and the whiteness verdict computed by rule.
STOP_2_STAGING = {
    "built": "2026-08-23, stop 2 of 4, NOT launched",
    "task": "stage_basal_views",
    "artifact": "data/staged/staged_basal_v1",
    "layout": (
        "staged_patient_g1.npy (236, 224, 224, 3) uint8 in geometry.csv "
        "row order, geometry.csv with per-image AR / content box / pad "
        "fraction / corner-white fraction, MANIFEST.json with the rollup "
        "and this decision -- staged_v1's G1 layout, so downstream "
        "loaders read it unchanged"
    ),
    "boundary_asserted": "no G2 tensor, no patch definitions are written",
    "sheets": (
        "ALL 236 staged crops, 24 per sheet in 6 columns on mid-grey, "
        "labelled patient/basal-id/AR/corner-white with FLAG below 0.90, "
        "ordered ascending by corner-white fraction so the most "
        "suspicious land on sheet 1"
    ),
    "gate": "basal_use_acknowledged, refused before any pixel (BASAL_USE_GATE)",
    "no_embedding_before_the_eye": (
        "the registered rule stands: sheets are reviewed by eye or not "
        "at all, and no embedding is extracted before the maintainer's pass; "
        "metrics stamp review: PENDING and the whiteness verdict "
        "computed by rule (geometry/basal.whiteness_verdict)"
    ),
    "config": "configs/p12_stage_basal.yaml, generated, flag false until set",
    "next_stop": "3 -- the four arms' tasks and configs, after the eye",
}


#: **[OBSERVED 2026-08-23, ``p12_stage_basal__978b67c4__p12-stage-basal``]
#: STOP 2's PIXEL RUN: THE REGISTERED REOPEN FIRED, AND THE EYE PASSED
#: THE IMAGES.**
#:
#: **The run**: attempt 0 clean; attempts 1-3 are the broken backoff
#: hitting the never-overwrite guard, which refused correctly -- the
#: guard working, and **three more points for
#: ``phase8.RETRY_LIMIT_IS_NOT_HOLDING``**. Artifact and all ten sheets
#: written by attempt 0.
#:
#: **The figures**: 236 staged; AR **0.624-1.562** against the frontal
#: 0.553-1.099 -- the basal family runs WIDER-THAN-TALL past anything the
#: frontal cohort contains; corner-white median **0.5234**, **all 236
#: flagged** against the 0.90 threshold. By the registered rule
#: (``geometry/basal.py``), **the white-pad rationale measured on
#: frontals does not transfer, and the staging decision REOPENED before
#: any embedding** -- which is the rule doing what it was registered to
#: do, not a failure.
#:
#: **the visual check, all ten sheets: PASS as images** -- upright, square,
#: recognisable submental views, nothing garbled. The corner report:
#: partial white concentrated in the TOP corners, less toward the bottom,
#: varying per image -- some white at bottom, some top, some both. **So
#: 0.5234 is the basal crop family's OWN corner geometry measured**
#: (plausibly a baked trapezoid with its wide edge at the top for a
#: worm's-eye view), **not a defect** in the images or the staging.
STOP_2_OBSERVED = {
    "observed": "2026-08-23, p12_stage_basal__978b67c4__p12-stage-basal",
    "attempts": (
        "attempt 0 clean; attempts 1-3 were the broken backoff hitting "
        "the never-overwrite guard, refused correctly -- +3 on "
        "phase8.RETRY_LIMIT_IS_NOT_HOLDING; artifact and sheets are "
        "attempt 0's"
    ),
    "figures": {
        "n_staged": 236,
        "aspect_ratio": (0.624, 1.562),
        "frontal_ar_for_comparison": (0.553, 1.099),
        "corner_white_median": 0.5234,
        "flagged": "ALL 236 against the 0.90 threshold",
    },
    "reopen_fired": (
        "by the registered rule: median 0.5234 < 0.90, so the white-pad "
        "rationale measured on frontals does NOT transfer and the "
        "decision reopened before any embedding -- the rule doing what "
        "it was registered to do"
    ),
    "eye_review": (
        "PASS as images, all ten sheets: upright, square, recognisable "
        "submental views, nothing garbled"
    ),
    "corner_structure": (
        "partial white concentrated in the TOP corners, less toward the "
        "bottom, varying per image -- the basal family's OWN corner "
        "geometry (plausibly a baked trapezoid, wide edge at top for a "
        "worm's-eye view), measured at 0.5234, NOT a defect"
    ),
}


#: **[PROPOSED 2026-08-23, NOT PICKED] THE REOPENED PAD DECISION, with
#: the measurements the choice should be made on.**
#:
#: **What died**: "the corners already arrive white, baked in at source"
#: -- true of frontals, measured FALSE of basals (median 0.5234, all 236
#: below 0.90, white concentrated at the top and varying per image).
#:
#: **THE PAD QUANTIFICATION, made visible before the choice.** Pad
#: fraction is a pure function of the aspect ratio
#: (``basal.pad_fraction_for_ar``, verified against the frozen
#: ``stage()`` to ~0.002): ``1 - ar`` tall, ``1 - 1/ar`` wide, zero at
#: square, **maximised at the endpoints of any AR interval**. So:
#:
#:     frontal (AR 0.553-1.099)  pad 0.018-0.4464, mean 0.2534 (measured)
#:     basal   (AR 0.624-1.562)  pad endpoints 0.3760 and 0.3598
#:
#: **No basal image can carry more pad than the frontal cohort's
#: most-padded image** (both basal endpoints sit below the frontal max
#: 0.4464). What IS new is **placement**: the frontal cohort is
#: essentially all tall (pad = side columns); the basal AR>1 images --
#: which the frontal cohort does not contain past 1.099 -- take
#: horizontal white BANDS above and below, up to ~36% of the square at
#: AR 1.562. The exact basal mean is already computed in the run's
#: SHAREABLE metrics.json ``pad_fraction`` block; one paste completes the
#: table, and the bound above does not depend on it.
#:
#: **(a) PAD WHITE ANYWAY -- the PROPOSED candidate.** Three arguments,
#: two of them measured:
#:
#: 1. **Recipe identity across the one-factor design.** All four arms,
#:    and BOTH channels of arm C's concatenation, stage through one
#:    function with one pad value -- the same convention the 0.2520 arm
#:    and every ladder arm used. Under (a), C differs from A by the VIEW
#:    and nothing else, which is the phase's entire design.
#: 2. **The one-fill argument, applied to the measurement, lands on
#:    white.** The original rationale was "keep ONE fill value in the
#:    image rather than two". The basal corners are ~52% baked white with
#:    per-image structure -- so most images already carry baked white
#:    somewhere. White pad adds NO second artificial fill beside it; ANY
#:    other pad value guarantees TWO fills (its own, adjacent to the
#:    baked white the maintainer saw). The frontal rationale dies as
#:    "continuity with corners"; it survives as "fewest distinct fills",
#:    and it survives MEASUREDLY.
#: 3. **The reviewed artifact stands.** ``staged_basal_v1`` IS pad-white
#:    through the frozen ``stage()``: reusable as-is, and the visual check
#:    pass covers exactly those pixels. No re-stage, no new sheets, no
#:    second eye pass.
#:
#: The honest recording: the rationale is **consistency-not-continuity**
#: (plus fewest-fills), and it is recorded as such rather than as the
#: frontal rationale surviving.
#:
#: **(b) PER-IMAGE EDGE-STATISTIC PAD.** Costs, each concrete:
#:
#: * arm C's two channels would come from DIFFERENT padding conventions,
#:   so C-vs-A and C-vs-D carry a second factor -- the exact confound the
#:   D control was built to exclude;
#: * on any image with partial baked white -- most of them, per the eye
#:   report -- an edge-statistic pad GUARANTEES two fills, the thing the
#:   original rationale exists to avoid;
#: * staging becomes CONTENT-dependent (the pad depends on the pixels),
#:   which is new code outside the frozen module with its own determinism
#:   story;
#: * ``staged_basal_v1`` and its eye review are DISCARDED: re-stage into
#:   v2, new sheets, a second pass of the visual check.
#:
#: **(c) NAMED AND EXAMINED**:
#:
#: * **crop-to-square instead of pad** -- rejected on Phase 2's own
#:   ground: at AR 1.562 a centre crop deletes ~36% of the width, and
#:   which anatomy that removes on a submental view is uncharacterised;
#: * **mid-grey or mean-colour pad** -- introduces a fill present in NO
#:   image, two fills guaranteed everywhere;
#: * **reconstruct the basal baked mask and pad to its fill** -- its
#:   fill IS white where it exists, collapsing back to (a) with extra
#:   machinery.
#:
#: **IS THERE A CHEAP DECIDING MEASUREMENT? Honestly: no.** The decision
#: is driven by DESIGN (one-factor comparability, fewest fills, a
#: reviewed artifact) rather than by pixels. A sensitivity extraction --
#: basal embeddings under both pads, per-image displacement reported --
#: is cheap and is registrable ONLY as sensitivity for the limitations
#: section: **choosing the pad by which yields the better arm is
#: outcome-shopping and is refused by name.** Small displacement would
#: say the choice is immaterial; large would say the convention matters
#: -- and under both readings the design argument still points the same
#: way, which is exactly why the measurement cannot decide it.
#:
#: **ONE CHEAP MEASUREMENT THAT IS WORTH REGISTERING (either candidate):
#: the staging-geometry confound check.** Corner-white fraction and AR
#: are per-image numbers already in ``geometry.csv``; correlate both
#: against the grade in the stop-3 arm run and report the two r's. If
#: staging geometry correlates with the label, arm B/C gains ride a
#: confound no pad choice fixes -- worth knowing BEFORE the arms, and it
#: is a report-with-reading, not a gate.
PAD_DECISION_REOPENED = {
    "proposed": "2026-08-23, NOT picked -- the choice is the maintainer's",
    "what_died": (
        "'corners already arrive white, baked in at source' -- true of "
        "frontals, measured FALSE of basals: median 0.5234, all 236 "
        "below 0.90, white concentrated at the top, varying per image"
    ),
    "pad_quantification": {
        "closed_form": (
            "pad(ar) = 1 - ar (tall) or 1 - 1/ar (wide); zero at square; "
            "maximised at the ENDPOINTS of any AR interval "
            "(basal.pad_fraction_for_ar, verified against the frozen "
            "stage() to ~0.002)"
        ),
        "frontal": "AR 0.553-1.099 -> pad 0.018-0.4464, mean 0.2534 measured",
        "basal_endpoints": {"ar_0.624": 0.3760, "ar_1.562": 0.3598},
        "bound": (
            "NO basal image can carry more pad than the frontal cohort's "
            "most-padded image -- both basal endpoints sit below the "
            "frontal max 0.4464"
        ),
        "what_is_new_is_placement": (
            "the frontal cohort is essentially all tall (pad = side "
            "columns); basal AR>1 images take horizontal white BANDS "
            "above and below, up to ~36% of the square at AR 1.562"
        ),
        "exact_basal_mean": (
            "already computed in the run's SHAREABLE metrics.json "
            "pad_fraction block; one paste completes the table, and the "
            "bound does not depend on it"
        ),
    },
    "a_pad_white": {
        "status": "PROPOSED",
        "recipe_identity": (
            "all four arms and BOTH channels of C's concatenation stage "
            "through one function with one pad value -- the ladder's own "
            "convention -- so C differs from A by the VIEW and nothing "
            "else, which is the phase's entire design"
        ),
        "one_fill_measured": (
            "the corners are ~52% baked white with per-image structure, "
            "so white pad adds NO second artificial fill while ANY other "
            "value guarantees TWO (its own beside the baked white). The "
            "frontal rationale dies as continuity-with-corners and "
            "survives MEASUREDLY as fewest-distinct-fills"
        ),
        "artifact_stands": (
            "staged_basal_v1 IS pad-white through the frozen stage(): "
            "reusable AS-IS, and the visual check pass covers exactly those "
            "pixels -- no re-stage, no new sheets, no second pass"
        ),
        "honest_recording": (
            "consistency-not-continuity, plus fewest-fills -- recorded as "
            "such, not as the frontal rationale surviving"
        ),
    },
    "b_edge_statistic_pad": {
        "status": "costed, not proposed",
        "costs": (
            "arm C's two channels under DIFFERENT conventions -- a second "
            "factor on C-vs-A and C-vs-D, the exact confound D exists to "
            "exclude",
            "on any image with partial baked white (most, per the eye "
            "report) an edge-statistic pad GUARANTEES two fills",
            "staging becomes CONTENT-dependent: new code outside the "
            "frozen module, its own determinism story",
            "staged_basal_v1 AND its eye review are discarded -- re-stage "
            "v2, new sheets, a second pass of the visual check",
        ),
    },
    "c_examined": {
        "crop_to_square": (
            "rejected on Phase 2's own ground: at AR 1.562 a centre crop "
            "deletes ~36% of the width, and which submental anatomy that "
            "removes is uncharacterised"
        ),
        "mid_grey_or_mean_pad": (
            "a fill present in NO image; two fills guaranteed everywhere"
        ),
        "pad_to_the_baked_masks_fill": (
            "its fill IS white where it exists -- collapses to (a) with "
            "extra machinery"
        ),
    },
    "no_cheap_deciding_measurement": (
        "honestly: none. The decision is driven by DESIGN -- one-factor "
        "comparability, fewest fills, a reviewed artifact -- not by "
        "pixels. A both-pads sensitivity extraction is cheap and "
        "registrable ONLY as sensitivity for limitations: choosing the "
        "pad by which yields the better arm is OUTCOME-SHOPPING, refused "
        "by name. Under both of its readings the design argument points "
        "the same way, which is exactly why the measurement cannot "
        "decide it"
    ),
    "confound_check_registered": (
        "worth doing under EITHER candidate: corner-white fraction and "
        "AR are per-image numbers already in geometry.csv; the stop-3 "
        "arm run correlates both against the grade and reports the two "
        "r's. If staging geometry correlates with the label, B/C gains "
        "ride a confound no pad choice fixes -- a report-with-reading, "
        "not a gate"
    ),
    "reusability": {
        "under_a": "staged_basal_v1 reusable as-is, eye pass included",
        "under_b_or_c": (
            "re-stage into staged_basal_v2, new sheets, new eye pass -- "
            "the eye rule attaches to the pixels that will be embedded"
        ),
    },
}


#: **[DECIDED 2026-08-23, the maintainer] PAD WHITE -- (a), recorded exactly as
#: proposed: CONSISTENCY-NOT-CONTINUITY, PLUS FEWEST-FILLS.**
#:
#: Recipe identity across all four arms and both of C's channels; no
#: second artificial fill on images that mostly carry baked white;
#: ``staged_basal_v1`` STANDS with the visual check pass attached -- no
#: re-stage, no second review. (b) rejected as manufacturing the confound
#: D exists to exclude; (c)'s variants as examined. **The refusal to
#: invent a deciding measurement is part of the decision**: pad-by-outcome
#: is outcome-shopping, and the record says so by name
#: (``PAD_DECISION_REOPENED``).
#:
#: **THE PAD TABLE, COMPLETED with the run's measured figures**:
#:
#:     frontal   pad min 0.0179  mean 0.2534  max 0.4464
#:     basal     pad min 0.0     mean 0.1451  max 0.375
#:
#: **Basal pads LESS on average** (0.1451 against 0.2534), and the
#: endpoint bound held almost exactly: the predicted maximum from the AR
#: endpoints was 0.3760, the measured maximum is 0.375. **The residual
#: novelty under (a) is PLACEMENT alone**: horizontal white bands above
#: and below on the AR>1 images -- a configuration the frontal cohort
#: never showed the backbone -- and it is recorded as the one thing (a)
#: does not make identical, to be carried beside arm B/C results rather
#: than rediscovered.
PAD_DECISION_TAKEN = {
    "decided": "2026-08-23 -- (a), pad white",
    "rationale_as_recorded": "consistency-not-continuity, plus fewest-fills",
    "b_rejected": "manufactures the confound D exists to exclude",
    "c_rejected": "as examined in PAD_DECISION_REOPENED",
    "no_deciding_measurement": (
        "part of the decision: pad-by-outcome is OUTCOME-SHOPPING, "
        "refused by name"
    ),
    "artifact": (
        "staged_basal_v1 STANDS, the visual check pass attached -- no "
        "re-stage, no second review"
    ),
    "pad_table_completed": {
        "frontal": {"min": 0.0179, "mean": 0.2534, "max": 0.4464},
        "basal": {"min": 0.0, "mean": 0.1451, "max": 0.375},
        "reading": (
            "basal pads LESS on average, and the endpoint bound held "
            "almost exactly (predicted max 0.3760, measured 0.375)"
        ),
    },
    "residual_novelty": (
        "PLACEMENT alone: horizontal white bands above and below on AR>1 "
        "images, a configuration the frontal cohort never showed the "
        "backbone -- carried beside arm B/C results rather than "
        "rediscovered"
    ),
}


#: **[REGISTERED 2026-08-23, STOP 3, BEFORE ANY BUILD] THE FOUR ARMS AND
#: THE EXTRACTION.**
STOP_3_REGISTERED = {
    "registered": "2026-08-23",
    "frontal_embeddings_reused": {
        "artifact": (
            "data/embeddings/embeddings_g1_ladder_v1/vit_b16__imagenet__g1 "
            "-- the 0.2520 arm's own set, hash carried verified from "
            "p7_d1_vit_b16_imagenet_g1.yaml"
        ),
        "why_reusable": (
            "it covers all 237 from staged_v1 and NO PIXEL CHANGED; the "
            "236 cohort is a subset by id"
        ),
        "alignment": (
            "the strict loaders assert EXACT row order, so the 237-set is "
            "loaded whole and aligned BY ID to the views manifest, with "
            "the dropped set asserted to be exactly {238} -- alignment by "
            "key lookup, never by position"
        ),
    },
    "basal_extraction": {
        "task": "extract_basal_embeddings",
        "artifact": "data/embeddings/embeddings_basal_v1/vit_b16__imagenet__g1",
        "machinery": (
            "the existing extract.extract_features (vit_b16, imagenet, "
            "checkpoint None -- deterministic, eval, no_grad) and "
            "embeddings.save with the views manifest ids, so the row "
            "order is checked at write time. No new extraction code"
        ),
    },
    "arms": {
        "task": "view_arm -- ONE task, the channel list the only difference",
        "channels": {
            "a": ("frontal",), "b": ("basal",),
            "c": ("frontal", "basal"), "d": ("frontal", "frontal"),
        },
        "head_parameters": {"a": 769, "b": 769, "c": 1537, "d": 1537},
        "recipe": (
            "the 0.2520 arm's, carried by the generator from its config: "
            "lr 0.001, weight decay 0.01, five seeds, 0.2/40/5, "
            "inner_val_mse -- recipe identity by construction"
        ),
        "labels_and_folds": (
            "the views manifest's, carried verbatim from cleft_v1; "
            "evaluation PCC against the panel mean, as everywhere"
        ),
    },
    "confound_report": {
        "where": (
            "the extraction task -- the one place staged_basal_v1's "
            "geometry.csv and the views manifest's labels meet once"
        ),
        "what": (
            "corr(corner_white_fraction, mean grade) and "
            "corr(aspect_ratio, mean grade) over the 236"
        ),
        "reading_registered": (
            "|r| >= 0.13 (two-sided p ~ 0.05 at n = 236) -> the "
            "staging-geometry confound caveat ATTACHES to arm B and C "
            "results; below -> reported as null. REPORT, NEVER GATE -- "
            "both r values are reported whichever way they fall"
        ),
    },
    "stop_4_unchanged": "the paired scope B/C/D vs A stays stop 4",
}


#: **[DEFECT 2026-08-23, FOUND BY THE FAILED LAUNCH, FIXED]
#: GUARD-AFTER-MKDIR: THE EXTRACTION TASK'S SCAFFOLDING TRIPPED THE SAVE
#: LAYER'S ARTIFACT GUARD.**
#:
#: ``p12-extract-basal`` failed all seven attempts. Attempt 0 died at
#: ~11s with the SAVE layer's ``EmbeddingError ... already exists`` (PLAN
#: §2.6 wording); attempts 1-6 with the task-level pre-check. The
#: artifact directory was EMPTY -- zero files -- created at 00:41:45
#: against attempt 0's first log line at 00:42:02, i.e. **by attempt 0
#: itself** within clock skew. No prior run existed; nothing was ever
#: written.
#:
#: **The mechanism**: the task pre-checked existence, then
#: ``mkdir(parents=True)``, then called ``embeddings.save`` -- which owns
#: its OWN existence guard and its OWN ``mkdir``, atomically. The task
#: handed save a directory the task had just created, and save refused
#: it, correctly: an existence guard cannot distinguish an artifact that
#: exists from an empty directory its own process just made.
#:
#: **The fix is on the TASK side, and the guard is untouched.** The task
#: no longer creates or pre-checks the save path; the save layer owns the
#: whole lifecycle. The alternative -- teaching the guard to accept empty
#: directories -- is refused by name: **an empty-dir exemption would also
#: accept a genuinely interrupted prior extraction's husk**, which is
#: exactly what the guard exists to refuse.
#:
#: **The same-pattern sweep over the other p12 tasks: CLEAN.**
#: ``build_views_manifest`` and ``stage_basal_views`` both
#: guard-then-mkdir-then-write **directly** (``atomic_write_text`` /
#: ``np.save``, no guarded layer beneath) -- Phase 1's own pattern,
#: correct there, and both already ran clean on the cluster. The conflict
#: exists only where a guarded save layer sits DOWNSTREAM of task-level
#: scaffolding, which was the extraction task alone.
#:
#: **The regression test runs the task twice against a clean tree**:
#: first run succeeds and writes the artifact; the second REFUSES, and
#: the refusal is asserted to fire from the SAVE layer on the artifact
#: (``EmbeddingError``, PLAN §2.6 wording) -- never from scaffolding.
#:
#: **Retry record**: +7 on ``phase8.RETRY_LIMIT_IS_NOT_HOLDING``
#: (attempts 0-6, a new per-launch record).
#:
#: **Cleanup before relaunch, exactly one path**:
#: ``data/embeddings/embeddings_basal_v1/`` -- the parent AND its empty
#: ``vit_b16__imagenet__g1`` child, both created by the task's mkdir,
#: both empty. Nothing else: save refused before writing any file,
#: metrics.json was never written, and the failed attempts' run
#: directories are ordinary voids that stay.
EXTRACT_GUARD_AFTER_MKDIR = {
    "defect": "2026-08-23, found by the failed launch, fixed",
    "run": "p12-extract-basal, attempts 0-6, all failed",
    "evidence": (
        "attempt 0: the SAVE layer's EmbeddingError 'already exists' at "
        "~11s; attempts 1-6: the task-level pre-check; the directory "
        "EMPTY (zero files), created at 00:41:45 against attempt 0's "
        "first log line 00:42:02 -- created by attempt 0 itself"
    ),
    "mechanism": (
        "the task pre-checked, mkdir'd, then called embeddings.save, "
        "which owns its OWN guard and mkdir atomically -- the task "
        "handed save a directory the task had just created, and save "
        "refused correctly: an existence guard cannot distinguish an "
        "artifact from an empty directory its own process just made"
    ),
    "fix_side": (
        "the TASK: it no longer creates or pre-checks the save path; the "
        "save layer owns the whole lifecycle. The guard is untouched -- "
        "an empty-dir exemption would also accept a genuinely "
        "interrupted extraction's husk, which is exactly what the guard "
        "exists to refuse"
    ),
    "same_pattern_sweep": (
        "CLEAN: build_views_manifest and stage_basal_views "
        "guard-then-mkdir-then-write DIRECTLY with no guarded layer "
        "beneath (Phase 1's own pattern, correct there); the conflict "
        "exists only where a guarded save layer sits DOWNSTREAM of "
        "task scaffolding -- the extraction task alone"
    ),
    "regression_test": (
        "the task runs twice against a clean tree: first succeeds, "
        "second refuses, and the refusal is asserted to fire from the "
        "SAVE layer on the artifact -- never from scaffolding"
    ),
    "retry_item": "+7 on phase8.RETRY_LIMIT_IS_NOT_HOLDING (attempts 0-6)",
    "cleanup": (
        "exactly one path: data/embeddings/embeddings_basal_v1/ -- the "
        "parent and its empty vit_b16__imagenet__g1 child, both created "
        "by the task's mkdir, both empty. Nothing else: no file was "
        "written, no metrics.json, and the failed attempts' run "
        "directories are ordinary voids that stay"
    ),
}


#: **[OBSERVED 2026-08-23,
#: ``p12_extract_basal__ab9b1d62__p12-extract-basal-2``] THE BASAL SET IS
#: EXTRACTED, AND THE REGISTERED CONFOUND READING FIRED ON ONE OF TWO.**
#:
#: Single clean attempt after the guard-after-mkdir fix: **(236, 768)**
#: written to ``embeddings_basal_v1/vit_b16__imagenet__g1``.
#:
#: **The confound report, reading applied by the registered rule
#: (threshold 0.13, |r|):**
#:
#:     r(aspect_ratio,  mean grade)  +0.1601   >= 0.13  -> ATTACHES
#:     r(corner_white,  mean grade)  +0.0977   <  0.13  -> null
#:
#: **THE STAGING-GEOMETRY CAVEAT ATTACHES TO ARMS B AND C**: wider basal
#: images correlate weakly with worse grades. **The report cannot
#: distinguish real anatomy from crop-geometry artifact** -- a broader
#: nasal base may photograph wider AND rate worse, or the crop geometry
#: may leak -- **which is exactly why the reading attaches rather than
#: gates**. Every B and C result now travels with it.
BASAL_CONFOUND_OBSERVED = {
    "observed": "2026-08-23, p12_extract_basal__ab9b1d62__p12-extract-basal-2",
    "extraction": (
        "single clean attempt after the guard-after-mkdir fix; (236, 768) "
        "written"
    ),
    "r_aspect_ratio_vs_mean": 0.1601,
    "r_corner_white_vs_mean": 0.0977,
    "threshold": 0.13,
    "fired_on": "aspect ratio only",
    "caveat": (
        "ATTACHES to arms B and C: wider basal images correlate weakly "
        "with worse grades, and the report cannot distinguish real "
        "anatomy (a broader nasal base photographing wider and rating "
        "worse) from crop-geometry artifact -- which is exactly why the "
        "reading attaches rather than gates. Every B and C result "
        "travels with it"
    ),
}


#: **[OBSERVED 2026-08-23, both single-attempt clean] ARMS A AND D: THE
#: BAR RE-MEASURED, AND THE CAPACITY CONTROL DOING ITS JOB.**
#:
#:     A  frontal-only, 236   0.2433/0.2484/0.2633/0.2462/0.2511
#:                            mean ~0.2505
#:     D  frontal (+) frontal 0.2440/0.2502/0.2713/0.2484/0.2530
#:                            mean ~0.2534
#:
#: **A is the views-cohort bar, and it is essentially the ladder's
#: 0.2520**: dropping patient 238 moved nothing, which is what the
#: carried-folds one-factor design predicted -- the baseline
#: re-measurement differs from the ladder by one patient's removal, and
#: the number confirms the removal was inert.
#:
#: **D sits within noise of A**: doubling the head's capacity on the SAME
#: information adds ~nothing (+0.0029 against seed sds of ~0.008). The
#: control is doing exactly its job, and **stop 4's C-vs-D floor is now
#: measured**: for arm C to mean anything as a VIEW result it must clear
#: what duplication alone gives, which is ~0.2534.
ARMS_A_D_OBSERVED = {
    "observed": "2026-08-23, both single-attempt clean",
    "runs": ("p12_arm_a_frontal__0dc7c79b", "p12_arm_d_capacity__0dc7c79b"),
    "a_frontal_only_236": {
        "per_seed": (0.2433, 0.2484, 0.2633, 0.2462, 0.2511),
        "mean": 0.2505,
        "reading": (
            "the views-cohort bar, essentially the ladder's 0.2520 -- "
            "dropping patient 238 moved nothing, confirming the "
            "carried-folds one-factor design's premise: the removal was "
            "inert"
        ),
    },
    "d_capacity_control": {
        "per_seed": (0.2440, 0.2502, 0.2713, 0.2484, 0.2530),
        "mean": 0.2534,
        "reading": (
            "within noise of A: doubling head capacity on the SAME "
            "information adds ~nothing (+0.0029 against seed sds ~0.008). "
            "The control doing exactly its job"
        ),
    },
    "stop_4_floor": (
        "MEASURED: for arm C to mean anything as a VIEW result it must "
        "clear what duplication alone gives -- ~0.2534"
    ),
}


#: **[OBSERVED 2026-08-23, both single-attempt clean] ARMS B AND C -- and
#: the caveat travels on both.**
#:
#:     B  basal-only  0.1926/0.2163/0.2307/0.1341/0.1662  mean ~0.1880
#:     C  concat      0.2736/0.2818/0.3028/0.2285/0.2333  mean ~0.2640
#:
#: **B carries real frontal-independent signal** -- well below A's 0.2505,
#: but ~5x A's seed sd above zero: the view the raters' scores are not
#: keyed to predicts on its own. **C is descriptively the project's
#: highest arm mean** -- C-A +0.0135, C-D +0.0106 -- **and the per-seed
#: direction is 3/5 against BOTH A and D**: seeds 99 and 12345 run
#: negative, and they are exactly B's two worst seeds -- **the basal
#: channel's noise carried into the concatenation.** The
#: staging-geometry caveat (``BASAL_CONFOUND_OBSERVED``) travels on both
#: results.
ARMS_B_C_OBSERVED = {
    "observed": "2026-08-23, both single-attempt clean",
    "runs": ("p12_arm_b_basal__0fb33b6a", "p12_arm_c_concat__0fb33b6a"),
    "b_basal_only_236": {
        "per_seed": (0.1926, 0.2163, 0.2307, 0.1341, 0.1662),
        "mean": 0.1880,
        "reading": (
            "real frontal-independent signal: well below A, but ~5x A's "
            "seed sd above zero -- the view the raters' scores are not "
            "keyed to predicts on its own"
        ),
    },
    "c_two_view_concat": {
        "per_seed": (0.2736, 0.2818, 0.3028, 0.2285, 0.2333),
        "mean": 0.2640,
        "reading": (
            "descriptively the project's highest arm mean; C-A +0.0135, "
            "C-D +0.0106; per-seed direction 3/5 against BOTH A and D, "
            "with seeds 99 and 12345 negative -- exactly B's two worst "
            "seeds, the basal channel's noise carried into the "
            "concatenation"
        ),
    },
    "caveat": (
        "the staging-geometry caveat (BASAL_CONFOUND_OBSERVED, r(AR, "
        "mean) = +0.1601) travels on BOTH results"
    ),
}


#: What Phase 12's paired scope covers: the three registered contrasts,
#: on the shared five seeds, over the same 236 patients and the same
#: carried folds. Everything else about the comparison -- the loader, the
#: truth cross-check, the BCa, the two conditions -- is the ONE paired
#: implementation, reached through the same dispatcher as every other
#: scope.
PAIRED_CLAIM_COVERAGE = {
    "covers": (
        "three pairs on the shared five seeds: B vs A (does basal alone "
        "reach the frontal bar), C vs A (does adding basal beat the "
        "bar), C vs D (does it beat CAPACITY -- the pair that makes "
        "C vs A attributable at all)"
    ),
    "excluded": {
        "any_pair_against_the_237_ladder": (
            "different cohort (236 vs 237) and different fold populations "
            "-- the within-phase arms exist precisely so no cross-cohort "
            "pair is needed"
        ),
    },
}

#: The four arms' stems, named once so the generator and the record
#: cannot drift.
PAIRED_STEMS = {
    "a": "p12_arm_a_frontal",
    "b": "p12_arm_b_basal",
    "c": "p12_arm_c_concat",
    "d": "p12_arm_d_capacity",
}


def paired_claim_pairs(scope: str = "p12") -> list[dict]:
    """Phase 12's three contrasts, shaped like every other scope's so the
    ONE paired implementation serves them.

    **Every recorded figure is DERIVED from the observed records** --
    means and seed sds recomputed from the per-seed tuples in
    ``ARMS_A_D_OBSERVED`` and ``ARMS_B_C_OBSERVED``, thresholds through
    the frozen ``combined_claimable_delta`` -- so a corrected observation
    lands here automatically. The convention is the headline scope's:
    ``b`` minus ``a``, and each pair records which side a positive delta
    favours.
    """
    if scope != "p12":
        raise Phase12Error(f"unknown Phase 12 paired-claim scope {scope!r}")

    import numpy as np

    from . import ladder
    from .train.phase3 import combined_claimable_delta

    per_seed = {
        "a": ARMS_A_D_OBSERVED["a_frontal_only_236"]["per_seed"],
        "d": ARMS_A_D_OBSERVED["d_capacity_control"]["per_seed"],
        "b": ARMS_B_C_OBSERVED["b_basal_only_236"]["per_seed"],
        "c": ARMS_B_C_OBSERVED["c_two_view_concat"]["per_seed"],
    }
    stats = {
        key: (
            float(np.mean(values)), float(np.std(values, ddof=1))
        )
        for key, values in per_seed.items()
    }
    seeds = list(ladder.SEED_POOL[:5])

    def pair(key, question, a, b, positive_means):
        mean_a, sd_a = stats[a]
        mean_b, sd_b = stats[b]
        threshold = combined_claimable_delta(
            sd_a, len(seeds), sd_b, len(seeds)
        )["arm_means_95"]
        delta = mean_b - mean_a
        return {
            "key": key, "question": question, "varies": "view",
            "a": PAIRED_STEMS[a], "b": PAIRED_STEMS[b],
            "seeds": seeds,
            "recorded": {
                "delta_of_means": round(delta, 4),
                "threshold": round(threshold, 4),
                "margin": round(abs(delta) / threshold, 2),
                "positive_means": positive_means,
                "source": (
                    "derived from ARMS_A_D_OBSERVED / ARMS_B_C_OBSERVED "
                    "per-seed values; DESCRIPTIVE, not claimable -- the "
                    "run computes the real one"
                ),
            },
        }

    return [
        pair(
            "p12__basal_vs_frontal_bar", "does basal alone reach the bar",
            "b", "a", "positive = the frontal bar A is ahead",
        ),
        pair(
            "p12__concat_vs_frontal_bar", "does adding basal beat the bar",
            "a", "c", "positive = the concat C is ahead",
        ),
        pair(
            "p12__concat_vs_capacity", "does it beat capacity",
            "d", "c", "positive = the concat C is ahead",
        ),
    ]


#: **[REGISTERED 2026-08-23, BEFORE THE RUN] STOP 4: THE PAIRED SCOPE,
#: WITH THE PREDICTION AND WHAT LITTLE REMAINS AT RISK.**
#:
#: ``configs/p12_paired.yaml``: B vs A, C vs A, C vs D, through the
#: existing generator and the one paired implementation -- standard
#: two-pass, 10k BCa, both conditions, shared five seeds. **It fits
#: nothing.**
#:
#: **THE PREDICTION, and it is arithmetic on recorded figures, not
#: foresight.** The per-seed signs are already visible in
#: ``ARMS_B_C_OBSERVED``: C runs positive against A and D on 3/5 seeds,
#: negative on 99 and 12345. So the reading-2 prior predicts **C-vs-A
#: fails condition 1 at 3/5 with a positive mean delta** -- and the
#: derived condition-2 arithmetic is ALREADY DECIDED by the recorded
#: seed spreads:
#:
#:     C vs A   delta +0.0135  threshold 0.0289  margin 0.47x  FAILS 2
#:     C vs D   delta +0.0106  threshold 0.0296  margin 0.36x  FAILS 2
#:     B vs A   delta 0.0625 (A ahead)  threshold 0.0347  margin 1.8x
#:              passes 2,
#:              but 1.81x sits far below the 2.55x floor beneath which
#:              nothing in this project has ever passed condition 1
#:
#: C's seed sd (~0.033, inherited from B's ~0.039 through the basal
#: channel) is ~4x A's, and the thresholds are driven by the noisier
#: side -- the Phase 10 lesson again: an unstable arm cannot claimably
#: beat, or be beaten, at this cohort's resolution.
#:
#: **WHAT REMAINS AT RISK: the per-patient BCa interval widths alone**
#: (condition 1's exclusion counts). Everything else above is fixed by
#: figures already recorded. **The honest sentence either way is
#: pre-written by the readings**: reading 2 (C > A descriptively,
#: condition 1 fails) keeps the missing-view story unrevived and the
#: phase closes descriptive; reading 1 would need BOTH conditions on
#: C-vs-A AND C-vs-D, which the arithmetic above already rules out --
#: so if the intervals surprise, it is the RECORDED FIGURES that were
#: wrong, and that would be the finding.
STOP_4_REGISTERED = {
    "registered": "2026-08-23, before the run",
    "config": "configs/p12_paired.yaml",
    "pairs": "B vs A, C vs A, C vs D -- shared five seeds, 10k BCa",
    "fits_nothing": (
        "every vector was written by a keeper arm run that already "
        "happened"
    ),
    "prediction_is_arithmetic": (
        "the per-seed signs are already visible (C positive on 3/5 "
        "against both A and D, negative on 99 and 12345), so the "
        "reading-2 prior's 'C-vs-A fails condition 1 at 3/5 with a "
        "positive mean delta' is arithmetic on recorded figures, not "
        "foresight"
    ),
    "condition_2_already_decided": {
        "c_vs_a": "delta +0.0135, threshold 0.0289, margin 0.47x -- FAILS",
        "c_vs_d": "delta +0.0106, threshold 0.0296, margin 0.36x -- FAILS",
        "b_vs_a": (
            "delta 0.0625 (A ahead), threshold 0.0347, margin 1.8x -- "
            "passes 2 but sits far below the 2.55x floor beneath which "
            "nothing has ever passed condition 1"
        ),
        "why": (
            "C's seed sd (~0.033, inherited from B's ~0.039 through the "
            "basal channel) is ~4x A's, and the thresholds are driven by "
            "the noisier side -- the Phase 10 lesson again"
        ),
    },
    "what_remains_at_risk": (
        "the per-patient BCa interval widths alone (condition 1's "
        "exclusion counts); everything else is fixed by recorded figures"
    ),
    "honest_sentence_prewritten": (
        "reading 2 keeps the missing-view story unrevived and the phase "
        "closes descriptive; reading 1 would need BOTH conditions on "
        "C-vs-A AND C-vs-D, which the arithmetic already rules out -- if "
        "the intervals surprise, the RECORDED FIGURES were wrong and "
        "THAT would be the finding"
    ),
}


#: **[OBSERVED 2026-08-23, ``p12-paired``, single attempt] THE PAIRED
#: VERDICTS -- THE PREDICTION CONFIRMED ON EVERY FIGURE, AND THE
#: INTERVALS WIDER THAN THE SIGNS SUGGESTED.**
#:
#:     pair     d        excl 0  cond 1  cond 2          verdict
#:     B vs A   +0.0625  0/5     False   True  (1.8x)    WITHDRAWN
#:     C vs A   +0.0135  0/5     False   False (0.47x)   unresolved
#:     C vs D   +0.0106  0/5     False   False (0.36x)   unresolved
#:
#: Arms as declared: A 0.2505 sd 0.0077, B 0.1880 sd 0.0388, C 0.2640 sd
#: 0.0321, D 0.2534 sd 0.0105.
#:
#: **THE PREDICTION WAS CONFIRMED ON EVERY FIGURE** -- deltas, thresholds
#: and margins exactly as derived (1.8x, 0.47x, 0.36x), which was
#: arithmetic on recorded values and confirms only that the vectors are
#: the runs they claim to be. **What the run alone could answer came back
#: STRONGER than anticipated: 0/5 everywhere** against the sign
#: structure's 3/5 -- no seed's per-patient paired difference is
#: distinguishable from zero on ANY contrast. The intervals are wider
#: than the seed-level signs suggested.
#:
#: **READING 2 -- the registered prior -- FIRED VERBATIM**: C exceeds A
#: descriptively (+0.0135, the project's highest arm mean) and is not
#: claimable. Applied as registered; **the pre-written interpretation
#: sentences stay unused**, exactly as their own applies-when clause
#: requires.
#:
#: **THE PHASE'S SHARPEST FINDING: EVEN B-vs-A IS WITHDRAWN.** The cohort
#: cannot resolve a 0.0625 gap between the frontal bar and a basal-only
#: arm -- a contrast whose DIRECTION NOBODY DOUBTS -- because the noisier
#: arm sets the threshold and every per-seed interval includes zero. **The
#: SIXTH arrival at COHORT_CANNOT_RESOLVE**, and the first where the
#: direction of the difference was never in question.
PAIRED_OBSERVED = {
    "observed": "2026-08-23, p12-paired, single attempt",
    "arms_as_declared": {
        "a": {"mean": 0.2505, "sd": 0.0077},
        "b": {"mean": 0.1880, "sd": 0.0388},
        "c": {"mean": 0.2640, "sd": 0.0321},
        "d": {"mean": 0.2534, "sd": 0.0105},
    },
    "verdicts": {
        "b_vs_a": {
            "d": 0.0625, "direction": "A ahead", "excludes_zero": "0 of 5",
            "condition_1": False, "condition_2": True, "margin": 1.8,
            "verdict": "WITHDRAWN",
        },
        "c_vs_a": {
            "d": 0.0135, "direction": "C ahead", "excludes_zero": "0 of 5",
            "condition_1": False, "condition_2": False, "margin": 0.47,
            "verdict": "unresolved",
        },
        "c_vs_d": {
            "d": 0.0106, "direction": "C ahead", "excludes_zero": "0 of 5",
            "condition_1": False, "condition_2": False, "margin": 0.36,
            "verdict": "unresolved",
        },
    },
    "prediction_confirmed": (
        "on every figure -- deltas, thresholds and margins exactly as "
        "derived (1.8x, 0.47x, 0.36x). That was arithmetic on recorded "
        "values; what the run alone could answer came back STRONGER than "
        "anticipated: 0/5 everywhere against the sign structure's 3/5 -- "
        "no seed's per-patient paired difference is distinguishable from "
        "zero on any contrast. The intervals are wider than the "
        "seed-level signs suggested"
    ),
    "reading_2_fired_verbatim": (
        "C exceeds A descriptively (+0.0135, the project's highest arm "
        "mean) and is not claimable. The pre-written interpretation "
        "sentences stay UNUSED, as their applies-when clause requires"
    ),
    "sharpest_finding": (
        "even B-vs-A is WITHDRAWN: the cohort cannot resolve a 0.0625 "
        "gap between the frontal bar and a basal-only arm -- a contrast "
        "whose DIRECTION NOBODY DOUBTS -- because the noisier arm sets "
        "the threshold and every per-seed interval includes zero. The "
        "SIXTH arrival at COHORT_CANNOT_RESOLVE, and the first where the "
        "direction was never in question"
    ),
    "caveat": (
        "the staging-geometry caveat (BASAL_CONFOUND_OBSERVED, r(AR, "
        "mean) = +0.1601) travels on every contrast involving B or C -- "
        "which is all three"
    ),
}


#: **[CLOSED 2026-08-23] PHASE 12 CLOSES.**
#:
#: The phase asked, with the corrected premise in force, whether adding
#: the basal view changes what this cohort's models can do. **The answer:
#: descriptively yes, claimably no -- and the premise stays UNSUPPORTED,
#: with supervision question 8 still open.**
#:
#: **THE FOUR STOPS, WALKED:**
#:
#: 1. **Manifest** -- ``cleft_v1_views`` derived from ``cleft_v1``, folds
#:    carried verbatim, six invariants asserted, verified on the cluster
#:    (236, exceptions 143 and 238 exactly).
#: 2. **Staging** -- the third-option decision measured from the code
#:    path (no trapezium parameter exercised at whole-image G1); the
#:    registered REOPEN fired (corner-white median 0.5234, all 236
#:    flagged); the visual check passed the images; the pad decision taken
#:    as (a) consistency-not-continuity plus fewest-fills, with placement
#:    the residual novelty; the ethics gate opened only 's
#:    edit, under REC 13/SW/0064 and 23/YH/0037.
#: 3. **The arms** -- extraction through the existing machinery (after
#:    the guard-after-mkdir defect was found by the failed launch and
#:    fixed on the task side, +7 on the retry record); the confound
#:    report fired on one of two (r(AR, mean) +0.1601 ATTACHES to B and
#:    C); A 0.2505 (the bar, dropping 238 inert), D 0.2534 (capacity
#:    null), B 0.1880 (real frontal-independent signal), C 0.2640
#:    (descriptively the project's highest).
#: 4. **The paired scope** -- prediction confirmed on every figure;
#:    B-vs-A WITHDRAWN, C-vs-A and C-vs-D unresolved, 0/5 everywhere;
#:    reading 2 verbatim; the interpretation sentences unused.
#:
#: **NO NEW CLAIMS.** Three ledger rows (29-31), none claimable, each
#: carrying its figures and the confound caveat by name. Everything cited
#: here was recorded when it happened.
#:
#: **CARRIED FORWARD OPEN, BY NAME:**
#:
#: * **supervision question 8** -- were the raters shown the frontal only, or
#:   both views with one score under the frontal's ID? It selects between
#:   the two pre-written interpretation sentences, which stay banked
#:   unused; with reading 2 fired, no current result depends on it, but
#:   the write-up's WORDING of B's signal does.
#: * **the AR-confound caveat** (r(AR, mean) = +0.1601) -- travels with
#:   ANY future use of the basal view, not only this phase's.
#: * **the whole-image-G1-only boundary** on the staging decision -- any
#:   region/patch or G2 use of basal images REOPENS it.
#: * **the standing infrastructure items** --
#:   ``phase8.RETRY_LIMIT_IS_NOT_HOLDING`` (now +7 and +3 richer) and
#:   ``phase9.PRE_CONTEXT_REFUSALS_LEAVE_NO_TRACE``.
PHASE_12_CLOSING = {
    "closed": "2026-08-23",
    "question": (
        "with the corrected premise in force: does adding the basal view "
        "change what this cohort's models can do?"
    ),
    "answer": (
        "descriptively yes, claimably no -- and the premise stays "
        "UNSUPPORTED, with supervision question 8 still open"
    ),
    "stops_walked": {
        "1_manifest": (
            "MET -- cleft_v1_views derived from cleft_v1, folds carried "
            "verbatim, six invariants asserted, verified on the cluster: "
            "236, exceptions 143 and 238 exactly"
        ),
        "2_staging": (
            "MET -- the third-option decision measured from the code "
            "path; the registered REOPEN fired (median 0.5234, all 236 "
            "flagged); the eye passed the images; pad decision (a), "
            "consistency-not-continuity plus fewest-fills, placement the "
            "residual novelty; the ethics gate opened only 's "
            "edit under REC 13/SW/0064 and 23/YH/0037"
        ),
        "3_arms": (
            "MET -- extraction through existing machinery after the "
            "guard-after-mkdir defect was fixed on the task side (+7 "
            "retries); confound fired on one of two (r(AR) +0.1601 "
            "ATTACHES to B and C); A 0.2505 the bar with 238's drop "
            "inert, D 0.2534 the capacity null, B 0.1880 real "
            "frontal-independent signal, C 0.2640 descriptively the "
            "project's highest"
        ),
        "4_paired": (
            "MET -- prediction confirmed on every figure; B-vs-A "
            "WITHDRAWN, C pairs unresolved, 0/5 everywhere; reading 2 "
            "verbatim; the interpretation sentences unused"
        ),
        "suite": "MET -- green throughout",
    },
    "no_new_claims": (
        "three ledger rows (29-31), none claimable, each carrying its "
        "figures and the confound caveat by name; everything cited here "
        "was recorded when it happened"
    ),
    "carried_forward_open": (
        "supervision question 8 -- frontal only, or both views with one score "
        "under the frontal's ID? It selects between the two pre-written "
        "interpretation sentences, banked unused; with reading 2 fired "
        "no current result depends on it, but the write-up's WORDING of "
        "B's signal does",
        "the AR-confound caveat (r(AR, mean) = +0.1601) -- travels with "
        "ANY future use of the basal view, not only this phase's",
        "the whole-image-G1-only boundary on the staging decision -- any "
        "region/patch or G2 use of basal images REOPENS it "
        "(STOP_2_STAGING_DECISION)",
        "phase8.RETRY_LIMIT_IS_NOT_HOLDING -- the standing infrastructure "
        "item, +7 and +3 richer from this phase",
        "phase9.PRE_CONTEXT_REFUSALS_LEAVE_NO_TRACE -- to consider, "
        "deliberately not built",
    ),
    "next": "Phase 13, the decoder/reconstruction, per the renumbering",
}


#: **[DECIDED 2026-08-23 -- THE SECOND SEQUENCE AMENDMENT] THE
#: REMAINING PHASES.** The first (``phase11.PHASE_SEQUENCE_RENUMBERED``)
#: moved the view ablation to 12; this one sequences everything after 13.
#: The original sequence stays visible in both records; nothing already
#: written is silently renumbered.
#:
#:     was (after the first amendment)   becomes
#:     13  decoder/reconstruction        13  decoder/reconstruction
#:     14  write-up                      14  LABEL DISTRIBUTION LEARNING
#:                                       15  the SECOND BEAUTY DATASET
#:                                           (Road B Branch 2, unparked)
#:                                       16  TSTR on the best from 15
#:                                           (the Rosero arm, unparked)
#:                                           [2026-08-29: "Rosero arm" is
#:                                           amended wording -- see
#:                                           synthesis.PARKED
#:                                           ["arm_amended_2026_08_29"];
#:                                           the parked design is in the
#:                                           Rosero FAMILY, not Rosero's]
#:                                       17  write-up
#:
#: **Status changes, dated**: TSTR parked -> SCHEDULED (16); LDL open
#: idea -> SCHEDULED (14); Road B Branch 2 parked -> SCHEDULED (15).
#:
#: **Phase 13** carries the amendment's registered warning unchanged: a
#: reconstruction loss optimises for looking like a face, and looking
#: like a face is not scoring like a clinician.
#:
#: **[2026-09-02] Fleiss kappa is UNWEIGHTED** -- it scores 1-vs-2 as
#: identically wrong to 1-vs-5, which is the wrong loss for an ordinal
#: scale. The distance-aware figures on the same matrix are QWK 0.4276
#: and mean inter-rater r 0.4696. Nothing below is edited:
#: ``record_audit.THE_KAPPA_LIMITATION``.
#: **Phase 14 (LDL)** attacks the measured central problem -- label
#: noise, Fleiss kappa 0.1662 -- directly, training on the score sheet's
#: ``soft_1..soft_5`` distribution columns rather than a scalar target;
#: Phase 11's arm-plus-matched-control template is the expected shape.
#: **AND IT HAS A PRIOR MEASUREMENT TO RECKON WITH, recorded here so the
#: phase cannot open without meeting it**: the ladder already carries a
#: clean negative under the name LDL (``ladder``, measured 2026-08-02 --
#: ldl 0.2336 against the mean's 0.2520 at imagenet/G1, claimably
#: harmful at masked/G2, verdict "LDL does not help ... the distribution
#: carries no information the mean discards that this head can use").
#: Phase 14's registration must position itself against that verdict
#: EXPLICITLY -- either by distinguishing what it trains (a distribution
#: head and a distribution loss over soft_1..5) from what the ladder's
#: ldl label-triple measured through the standard head, or by stating a
#: new measured reason the closed negative does not cover it. Scheduling
#: is not reopening; opening without meeting the prior measurement would
#: be.
#:
#: **Phase 15 (second beauty dataset)** carries its registered note NOW:
#: its exit criteria MUST include a comparative verdict against SCUT,
#: because **Phase 16 consumes "the best" as a measured answer, not a
#: vibe.**
#:
#: [2026-08-29] "Rosero-design" below is amended wording: the parked
#: design is in the Rosero FAMILY, not Rosero's design
#: (synthesis.PARKED["arm_amended_2026_08_29"]). Preserved as written.
#: **Phase 16 (TSTR)** runs on the best beauty dataset from 15, or a
#: better face dataset if 15's search surfaces one -- the Rosero-design
#: arm, unparked from PLAN Part 6.
PHASE_SEQUENCE_RENUMBERED_2 = {
    "decided": "2026-08-23 -- the second sequence amendment",
    "first_amendment": "phase11.PHASE_SEQUENCE_RENUMBERED (view ablation to 12)",
    "was": {"13": "decoder/reconstruction", "14": "write-up"},
    "becomes": {
        "13": "decoder/reconstruction (unchanged; the registered warning "
              "stands: a reconstruction loss optimises for looking like a "
              "face, and looking like a face is not scoring like a "
              "clinician)",
        "14": "LABEL DISTRIBUTION LEARNING -- soft_1..soft_5 as the "
              "target; the arm-plus-matched-control template expected",
        "15": "the SECOND BEAUTY DATASET (Road B Branch 2, unparked) -- "
              "exit criteria MUST include a comparative verdict against "
              "SCUT, because Phase 16 consumes 'the best' as a measured "
              "answer, not a vibe",
        # [2026-08-29] "Rosero-design" is amended wording
        # (synthesis.PARKED["arm_amended_2026_08_29"]): Rosero FAMILY,
        # not Rosero's design. Preserved as written.
        "16": "TSTR on the best from 15 (or a better face dataset if "
              "15's search surfaces one) -- the Rosero-design arm, "
              "unparked from PLAN Part 6",
        "17": "write-up",
    },
    # [2026-08-24] A THIRD amendment inserts the METRIC-SPACE ABLATION
    # at 16, pushing TSTR to 17 and the write-up to 18
    # (phase15.PHASE_SEQUENCE_RENUMBERED_3). The mapping above is
    # preserved as what it was; "16: TSTR" here means Phase 17 now.
    "third_amendment": "phase15.PHASE_SEQUENCE_RENUMBERED_3",
    # [2026-08-24, later] A FOURTH amendment promotes the anchor loop to
    # 16, moving the metric-space ablation to 18 and the write-up to 19;
    # TSTR stays at 17 (phase15.PHASE_SEQUENCE_RENUMBERED_4). This
    # record's numbers are two amendments old and preserved as written.
    "fourth_amendment": "phase15.PHASE_SEQUENCE_RENUMBERED_4",
    # [2026-08-31] A FIFTH amendment APPENDS Phase 20 (the permutation
    # control). It renumbers nothing -- 19 remains the write-up -- which
    # is why it is EXTENDED_5 rather than RENUMBERED_5.
    "fifth_amendment": "phase20.PHASE_SEQUENCE_EXTENDED_5",
    # [2026-09-01] A SEVENTH amendment schedules 22 (ranking and
    # pairwise losses), 23 (the statistical instruments), 24 (all five
    # raters) and 25 (foundation-model features) as
    # SCHEDULED-NOT-REGISTERED. It renumbers nothing, and leaves the
    # write-up's number OPEN rather than resolving it
    # (phase21.WRITE_UP_NUMBER_OPEN).
    "seventh_amendment": "phase21.PHASE_SEQUENCE_EXTENDED_7",
    # [2026-08-31] A SIXTH amendment APPENDS Phase 21 (the ensemble
    # probe and error-consistency diagnosis). It renumbers nothing --
    # the first amendment written UNDER the write-up-runs-last rule
    # rather than establishing it.
    "sixth_amendment": "phase21.PHASE_SEQUENCE_EXTENDED_6",
    "status_changes": {
        "tstr": "parked -> SCHEDULED (Phase 16)",
        "ldl": "open idea -> SCHEDULED (Phase 14)",
        "roadb_branch_2": "parked -> SCHEDULED (Phase 15)",
    },
    # [ACKNOWLEDGED 2026-08-23, the maintainer] Accepted exactly as framed:
    # Phase 14 opens by positioning against the ladder's closed LDL
    # verdict -- distinguish the mechanism with a measured argument, or
    # concede coverage -- and scheduling is not reopening. The pinned
    # figures stand.
    "ldl_reckoning_accepted": (
        "2026-08-23 -- accepted exactly as framed; the pinned "
        "figures stand"
    ),
    "ldl_prior_measurement_to_reckon_with": (
        "the ladder carries a clean negative under the name LDL "
        "(measured 2026-08-02: ldl 0.2336 against the mean's 0.2520 at "
        "imagenet/G1, claimably harmful at masked/G2, verdict 'LDL does "
        "not help'). Phase 14's registration must position itself "
        "against that verdict EXPLICITLY -- distinguish what it trains "
        "(a distribution head and loss over soft_1..5) from what the "
        "ladder's ldl label-triple measured through the standard head, "
        "or state a new measured reason the closed negative does not "
        "cover it. Scheduling is not reopening; opening without meeting "
        "the prior measurement would be"
    ),
    # [CORRECTED 2026-08-24, at the reckoning itself] The clause above,
    # "through the standard head", is WRONG about the prior measurement
    # -- preserved verbatim above per the standing pattern, corrected
    # here.
    "ldl_mechanism_description_corrected": (
        "2026-08-24: the ladder's ldl label-triple did NOT go through "
        "the standard scalar head. Verified against train/ldl.py "
        "(LDLHeadBackbone) and the shipped config at Phase 14's "
        "reckoning: it trained a FIVE-OUTPUT SOFTMAX head under a KL "
        "loss on soft_1..soft_5, reading out the EXPECTATION for PCC "
        "-- exactly the mechanism Phase 14 proposed, which is why the "
        "phase conceded coverage "
        "(phase14.PHASE_14_CONCEDED_COVERED). THE PINNED FIGURES "
        "STAND; only the mechanism description was wrong. Provenance: "
        "the mis-description originated in the phase-sequencing "
        "discussion and was caught by READING THE MODULE rather than "
        "trusting the description"
    ),
    "nothing_silently_renumbered": (
        "the original sequence stays visible in both amendment records; "
        "phase8's 'Road B Branch 2 stays PARKED' and the TSTR-parked "
        "precedent lines stay as written, with this record as the dated "
        "pointer's source"
    ),
    "nothing_built": "Phase 13's restate-before-building message follows",
    "eighth_amendment": (
        "phase25.PHASE_SEQUENCE_EXTENDED_8, 2026-09-05. Calibration "
        "ablation as 26, anchor set as a training set as 27, fine "
        "tuning as 28 with small backbones as an arm inside it. "
        "Renumbered nothing"
    ),
}


#: **[BANKED 2026-08-23, ALL THREE PRIMARIES OBTAINED AND READ]
#: ASHER-McDADE 1991, SHAW 1992 PART 1, ASHER-McDADE 1992 PART 4 -- the
#: instrument's founding papers, read at source. Record only; nothing
#: built.** Tags on every claim below upgrade from
#: [LITERATURE] read second-hand (the NotebookLM pass) to [LITERATURE]
#: read at source. [TAG NORMALISED 2026-09-01 from
#: '[LITERATURE-secondhand]' and '[LITERATURE-primary]': the tags are
#: exactly three and the reading route is prose, not part of the tag.]
PRIMARY_SOURCES_BANKED = {
    "banked": "2026-08-23, all three primaries obtained and read",
    "sources": (
        "Asher-McDade 1991 (the instrument)",
        "Shaw 1992 Part 1 (the six-centre study's frame)",
        "Asher-McDade 1992 Part 4 (the operational panel)",
    ),
    "tag_upgrade": (
        "[LITERATURE] via NotebookLM -> [LITERATURE-primary] throughout: "
        "the derived readings held at source and now rest on the papers "
        "themselves"
    ),
    # ---- (1) the label correction, confirmed on both primaries --------
    "label_correction": {
        "was": "'Asher-McDade composite' (docs/PLAN.md:903, 913, 1512)",
        "becomes": "Asher-McDade-DERIVED SINGLE OVERALL SCORE",
        "confirmed": (
            "the 1991 instrument is FOUR COMPONENTS (nasal form; nasal "
            "deviation/symmetry; vermilion border; profile incl. upper "
            "lip), each 1-5, summable 4-20; the 1991 judges explicitly "
            "found component scoring 'more satisfactory... than to give "
            "an overall score'; Part 4's operational panel ALSO scored "
            "components separately. Our sheet's one gestalt 1-5 per "
            "rater departs from the lineage at both the pilot and the "
            "operational study -- its closest ancestor is the 7-point "
            "overall scale 1991 TRIED AND REJECTED"
        ),
        "error_class": (
            "the SEVENTH quantity-under-one-name instance -- two "
            "different quantities travelling under one label (the "
            "class reliability.py's own docstring warns about, and the "
            "class 27-regions, CASE/IEM and consensus/median all "
            "inhabit). The PLAN's wording is preserved at its cited "
            "lines; this record is the correction's source"
        ),
    },
    # ---- (2) the panel-composition departure ---------------------------
    "panel_departure": {
        "tag": "LITERATURE-primary",
        "theirs": (
            "Part 4: SIX ORTHODONTISTS, one per centre, with a "
            "pre-rating familiarization task"
        ),
        "ours": (
            "five mixed disciplines (incl. a cleft patient and a "
            "psychologist), no recorded calibration"
        ),
        "reading": (
            "a measured lineage departure that PLAUSIBLY contributes to "
            "our lower agreement (mean r 0.4696 against their "
            "single-judge 0.60) -- a write-up sentence, not a defect"
        ),
    },
    # ---- (3) the reliability anchors -----------------------------------
    "reliability_anchors": {
        "tag": "LITERATURE-primary",
        "figures": (
            "1991 single-judge ICCs 0.43-0.60 (components), 0.60 "
            "(total); Part 4 computes NO new coefficients and cites "
            "1991's"
        ),
        "reading": (
            "0.43-0.60 is the lineage's ONLY measured agreement, and it "
            "matches ours -- 'low agreement is the field's condition' "
            "now rests on the founding papers"
        ),
    },
    # ---- (4) the ceiling's provenance, both halves ---------------------
    "ceiling_provenance": {
        "the_lineages_half": (
            "Spearman-Brown is the lineage's OWN machinery: 1991 names "
            "the formula (citing Fleiss 1986) and Part 4 sizes its panel "
            "on the projections ('six examiners would produce a pooled "
            "panel reliability of 0.9')"
        ),
        "our_half": (
            "neither paper does attenuation-ceiling reasoning; the "
            "sqrt(reliability) bound (data/reliability.py: 0.8158 -> "
            "0.9032) is STANDARD PSYCHOMETRICS APPLIED ON TOP. The "
            "write-up claims exactly these two halves, separately"
        ),
        "trap_registered": (
            "their projected PANEL RELIABILITY 0.90/0.902 is NOT our "
            "CORRELATION CEILING 0.9032 -- different quantities, "
            "coincidentally adjacent; NEVER placed in proximity "
            "unqualified"
        ),
    },
    # ---- (5) the distribution match ------------------------------------
    "distribution_match": {
        "tag": "LITERATURE-primary",
        "theirs": "Part 4's six-centre total means 2.8-3.4, SDs 0.3-0.6",
        "ours": "middle-loading counts 5/89/110/30/3",
        "reading": (
            "mid-scale clustering is the INSTRUMENT'S home behaviour; "
            "our cohort's middle-loading is lineage-typical, not rater "
            "idiosyncrasy"
        ),
    },
    # ---- (6) basal absent from all three primaries ---------------------
    "basal_absent_at_source": {
        "tag": "LITERATURE-primary",
        "measured_at_source": (
            "frontal and lateral only, in ALL THREE papers, masked with "
            "card overlays to the nasolabial strip; 1991's stated "
            "reason -- judges are influenced by general attractiveness "
            "-- cited verbatim by Part 4. Anchors confirmed 1 = very "
            "good .. 5 = very poor on both scales"
        ),
        "consequence": (
            "Phase 12's corrected premise now has FULL PRIMARY SUPPORT: "
            "the instrument excludes the basal view, so arm B's 0.1880 "
            "is signal from a view ENTIRELY OUTSIDE THE INSTRUMENT. supervision "
            "question 8 -- what OUR panel was actually shown -- stays "
            "open; the primaries settle the instrument, not our "
            "panel's practice"
        ),
    },
    # ---- (7) the ancestral citation for the resolution finding ---------
    "resolution_findings_ancestor": {
        "tag": "LITERATURE-primary",
        "their_chain": (
            "Part 4's own argument: pooled reliability 0.9 is what lets "
            "'statistically significant differences be discerned'"
        ),
        "reading": (
            "our cohort-cannot-resolve argument in 1992 vocabulary -- "
            "the lineage itself reasoned from panel reliability to "
            "discernibility. Banked BESIDE the six arrivals at "
            "COHORT_CANNOT_RESOLVE as their ancestral citation, not as a "
            "seventh arrival"
        ),
    },
}


def summary() -> dict:
    """The phase's records, importable as one object."""
    return {
        "registered": PHASE_12_REGISTERED,
        "sentences": INTERPRETATION_SENTENCES_PREWRITTEN,
        "readings": PHASE_12_READINGS,
        "stop_1": STOP_1_MANIFEST,
        "staging_decision": STOP_2_STAGING_DECISION,
        "gate": BASAL_USE_GATE,
        "stop_2": STOP_2_STAGING,
        "stop_2_observed": STOP_2_OBSERVED,
        "pad_reopened": PAD_DECISION_REOPENED,
        "pad_taken": PAD_DECISION_TAKEN,
        "stop_3": STOP_3_REGISTERED,
        "guard_after_mkdir": EXTRACT_GUARD_AFTER_MKDIR,
        "confound_observed": BASAL_CONFOUND_OBSERVED,
        "arms_a_d": ARMS_A_D_OBSERVED,
        "arms_b_c": ARMS_B_C_OBSERVED,
        "paired_coverage": PAIRED_CLAIM_COVERAGE,
        "stop_4": STOP_4_REGISTERED,
        "paired_observed": PAIRED_OBSERVED,
        "closing": PHASE_12_CLOSING,
        "sequence_2": PHASE_SEQUENCE_RENUMBERED_2,
        "primaries": PRIMARY_SOURCES_BANKED,
    }
