"""Phase 25: foundation-model features -- DINOv2 through the probe.

**One arm, one factor, and a registered expectation that it will not
move.** Eleven prior measurements varied the feature source and none
beat 0.2520 claimably; the best was parity. All eleven were SUPERVISED.
This is the first self-supervised checkpoint the cohort has seen, and
the phase exists to close that axis by measurement rather than leave the
write-up carrying "we never tried it".

**REGISTERED, NOT BUILT.** No task, no config, no arm, no run.
"""

from __future__ import annotations

#: **[RECKONING 2026-09-02] PHASE 25 OPENS, AND THE RECORD ARGUES
#: AGAINST IT.**
#:
#: the ruling is RUN IT. The reckoning states both sides at full
#: strength, with the counter-evidence first, because the counter-evidence
#: is stronger than the case.
PHASE_25_RECKONING = {
    "opened": "2026-09-02 -- registration only; nothing built",
    "the_ruling": "RUN IT (2026-09-02)",
    "the_amendment_argued_against_itself": (
        "**phase21.PHASE_SEQUENCE_EXTENDED_7 sequenced this LAST on its "
        "own weak prior**: 'the project's own evidence says feature "
        "source has moved little: MEBeauty 0.2537 against ImageNet "
        "0.2520. A new backbone is the most expensive intervention with "
        "the weakest prior, so it is sequenced last.' **That tempering "
        "stands and is not softened here**"
    ),
    "and_it_was_written_before_phase_22": (
        "**the motivation predates the objective result.** Phase 22 has "
        "since measured that the TRAINING OBJECTIVE is inert too -- "
        "'bounded ranking matches the probe to within 0.003 and buys "
        "nothing measurable on PCC' (PHASE_22_CLOSING). So the prior "
        "against this phase is now stronger than when it was scheduled, "
        "and the ruling is made against that, not around it"
    ),
    "what_makes_it_worth_running_anyway": (
        "**every one of the eleven is SUPERVISED** -- ImageNet "
        "classification, or a supervised aesthetic set (SCUT, "
        "MEBeauty). **Self-supervised features have never been tried on "
        "this cohort** (verified: no dinov2 / self-supervised / "
        "foundation-model / CLIP / MAE reference exists anywhere in "
        "src/ or configs/). The 'already covered' ground that closed "
        "three of Phase 24's four designs **does not exist here**"
    ),
    "the_ground_for_running_rather_than_conceding": (
        "**closing the axis by MEASUREMENT is worth twenty minutes, and "
        "'we never tried self-supervised features' is otherwise a gap "
        "the write-up CARRIES rather than an answer.** A null that was "
        "measured is a different object from a null that was assumed, "
        "and only one of them survives a reviewer asking why"
    ),
    "the_honest_expectation": (
        "**the record predicts UNRESOLVED**, and that is registered as "
        "the expectation rather than discovered afterwards "
        "(READINGS_COMMITTED). **A phase that predicts its own null and "
        "gets it has learned something about the axis** -- it converts "
        "an assumption into a measurement, which is the whole of what "
        "is being bought"
    ),
    "no_ledger_row_is_promised": (
        "a claimable result would earn one; unresolved or below would "
        "not. **The row is not promised in advance either way**"
    ),
    "tag": "[REGISTERED] -- nothing built",
}


#: **[THE COUNTER-EVIDENCE 2026-09-02] The eleven, and the sharpest of
#: them -- with two corrections to the figures as proposed.**
#:
#: **"Moved little" is not true of all eleven and the record should not
#: say it is.** Several moved it a great deal -- downward. What is true
#: of all eleven: **none beat 0.2520 claimably, and the best was
#: parity.**
THE_ELEVEN_AND_THE_SHARPEST = {
    "recorded": "2026-09-02",
    "the_accurate_form_of_the_claim": (
        "**eleven measurements varied the feature source; NONE beat "
        "0.2520 claimably; the best two were PARITY.** Not 'all eleven "
        "moved it little' -- vit_b8 and the resolution set moved it "
        "enormously, downward. The invariant is the direction and the "
        "ceiling, not the size of the movement"
    ),
    "the_eleven": {
        "1_mebeauty_original": "0.2537 vs 0.2520 -- +0.0017, PARITY",
        "2_scut_original": "0.1952 -- -0.0568, CLAIMABLY WORSE",
        "3_mebeauty_g1_masked": "-0.0693",
        "4_mebeauty_g2_masked": "-0.0299",
        "5_scut_masked_g2": "0.2001",
        "6_vit_b32": "0.2519 -- PARITY to four decimals, different patch size",
        "7_vit_b8": "0.0432 -- collapse, ZERO interpolation",
        "8_mvitv2_b": "0.1844",
        "9_swin_b": "0.2092 -- ledger row 1, the comparison WITHDRAWN",
        "10_resolution": "512 -> 0.0894, 768 -> 0.0937; endpoint -0.1583 at -13.29 sigma",
        "11_objective_phase_22": (
            "bounded ranking matches the probe to within 0.003 -- not "
            "feature SOURCE, but the same lesson on the adjacent axis"
        ),
    },
    "the_sharpest_ONE_FACTOR_contrast": (
        "**within the ViT patch family, at the SAME dataset, geometry "
        "and resolution: 0.2519 (patch32) to 0.0432 (patch8), a span of "
        "0.2087 -- while the pretraining SOURCE at fixed encoder and "
        "geometry moves it 0.0017.** Both sides are one-factor "
        "comparisons, which is what makes this the phase's strongest "
        "counter-evidence: **the encoder is worth two orders of "
        "magnitude more than the source on this cohort**"
    ),
    "CORRECTION_the_srgnn_contrast_proposed_is_not_one_factor": (
        "**[The restate proposed srgnn_imagenet_224 at 0.1719 against "
        "0.2520 as 'the same images with the same weights', moving it "
        "0.08. Two things are wrong with it, both in the record.]** "
        "**(i) It is not one factor**: ladder.TRADE_OFF_PAIR records "
        "that the pair 'varies backbone, init and geometry at once -- "
        "not a one-factor comparison'. **(ii) 0.1719 is a TEN-SEED "
        "mean and the paired comparison is 0.1884 over the five seeds "
        "ViT shares**; the ladder's own note says 'Two numbers for one "
        "arm, and a reader WILL try to reconcile them ... quote "
        "whichever matches the comparison being made, and say which'. "
        "**The measured delta is +0.0637 at margin 1.342, 0 of 5 seeds "
        "excluding zero, ``claimable: False``** -- not the +0.0801 the "
        "recorded means implied. The 0.08 is ``delta_if_recorded_means_"
        "hold``, superseded by the measurement. The patch-family "
        "contrast above is substituted because it is clean on both "
        "counts"
    ),
    "what_survives_the_correction": (
        "**the direction and the order of magnitude.** Even at the "
        "measured, unclaimable 0.0637, the encoder axis moved ~37x what "
        "the source axis moved. The counter-evidence is not weakened in "
        "substance -- only in the figure quoted for it"
    ),
}


#: **[REGISTERED BEFORE ANY NUMBER 2026-09-02] THE ATTRIBUTION PROBLEM,
#: and the control that does not exist.**
#:
#: **DINOv2 changes BOTH the encoder and the pretraining objective**
#: relative to the probe. A positive result is unattributable between
#: them without a second arm.
THE_ATTRIBUTION_PROBLEM = {
    "registered": "2026-09-02, before any number",
    "what_changes_at_once": (
        "**two factors, not one.** Against ``vit_base_patch16_224`` "
        "(supervised in21k->in1k, patch 16, native 224), DINOv2-base is "
        "**patch 14, native 518, self-supervised (LVD-142M)**. The "
        "encoder and the objective move together"
    ),
    "what_the_phase_MAY_claim": (
        "**'these features are better/worse/indistinguishable on this "
        "cohort'** -- a statement about one checkpoint, which is what "
        "the arm measures"
    ),
    "what_the_phase_MAY_NOT_claim": (
        "**'self-supervision is what did it'**, in either direction. A "
        "gain could be the objective, the patch size, the pretraining "
        "corpus, the resolution regime, or any combination. **This "
        "bound is registered BEFORE the number so it cannot be "
        "renegotiated after one**"
    ),
    "the_same_encoder_control_IS_NOT_AVAILABLE_and_this_is_MEASURED": (
        "**a ViT-B/16 DINOv2 checkpoint does not exist, and neither "
        "does a supervised ViT-B/14.** Measured against the timm "
        "index, not assumed: ``list_models('*dinov2*', pretrained=True)`` "
        "returns **eight entries, ALL patch14** (small/base/large/giant "
        "x plain/reg4); ``list_models('*dinov2*patch16*')`` returns "
        "**[]**; and ``list_models('vit_base_patch14*', "
        "pretrained=True)`` returns **exactly the two DINOv2 entries and "
        "nothing else**. **So the objective cannot be isolated in "
        "either direction** -- there is no DINOv2 at the probe's patch "
        "size and no supervised checkpoint at DINOv2's"
    ),
    "therefore_the_unattributability_is_a_BOUND_THE_PHASE_ACCEPTS": (
        "**not a choice the phase declines to make.** No second arm is "
        "registered because none would isolate anything: a second "
        "DINOv2 variant (reg4, or small/large) would vary capacity or "
        "registers, never the objective. **The bound is accepted and "
        "stated, and no extra extraction is proposed**"
    ),
    "where_these_measurements_came_from": (
        "**Python 3.13 with timm 1.0.27 and torch 2.13, NOT the suite's "
        "interpreter.** The 3.11 venv `scripts/test.ps1` runs has "
        "neither installed, so **the DINOv2 tests SKIP in the suite** "
        "rather than pinning these facts. They are reproducible by "
        "running the 3.13 interpreter and are recorded here with that "
        "provenance -- a measurement whose home is named is not the "
        "same as one the suite enforces, and the difference is stated"
    ),
    "the_measurement_was_made_on_THIS_machine_at_a_DIFFERENT_TIMM": (
        "**timm 1.0.27 here; the cluster is pinned at 1.0.7** "
        "(ladder.PHASE_7D_BUILT, measured against the pinned install). "
        "The model index may differ between them. **Whether 1.0.7 "
        "carries DINOv2 at all is a BLOCKING READ**, on the Phase 9 "
        "precedent, and is listed in SETTINGS_TO_DECLARE rather than "
        "assumed"
    ),
}


#: **[REGISTERED 2026-09-02 -- NOT BUILT] THE ARM.**
#:
#: One factor: the backbone. Everything else held at the probe's values.
THE_ARM_REGISTERED = {
    "registered": "2026-09-02 -- REGISTERED, NOT BUILT",
    "the_arm": (
        "**DINOv2 through the pooled-embedding path at the probe's own "
        "settings** -- same geometry (g1), same resolution (224), same "
        "linear head, same five seeds (1337/2024/7/99/12345), same "
        "5-fold OOF, same ``mean`` label. **Everything held except the "
        "backbone**"
    ),
    "the_contrast": (
        "against the **0.2520 probe** (``p7_d1_vit_b16_imagenet_g1``) "
        "under the **full PLAN 4.3 criterion** -- per-seed paired BCa "
        "excluding zero AND the delta exceeding the combined seed "
        "uncertainty. Both conditions, as every claimable comparison in "
        "this project requires"
    ),

    # ---- what reuses, and what does NOT --------------------------------
    "CORRECTION_there_is_no_pooled_features_task": (
        "**[The restate says 'reuse the shipped extraction; no new task "
        "if ``pooled_features`` parameterises on backbone name alone, "
        "as reported'. NO SUCH TASK EXISTS, and it was not reported.]** "
        "The extraction task kind is **``extract_embeddings``**; "
        "``pooled`` is an EMBEDDING KIND (``embeddings.EMBEDDING_KINDS``) "
        "selected by ``KIND_FOR_BACKBONE_KIND = {'transformer': "
        "'pooled', 'graph': 'feature_map'}``. What WAS reported is that "
        "``models.factory.BACKBONES`` has seven entries and no DINOv2, "
        "and that ``snapshot_pretrained_init``'s ``backbone`` field is a "
        "**closed choices list**"
    ),
    "what_genuinely_reuses": (
        "**the task and the whole downstream path.** Any "
        "``kind == 'transformer'`` backbone flows through "
        "``extract_embeddings`` to a pooled ``(N, D)`` artifact and then "
        "through the probe's existing head. **No new task is needed**"
    ),
    "what_MUST_be_added_and_it_is_CODE_not_config": (
        "**(i)** a ``BACKBONES`` entry (timm_name, kind='transformer', "
        "norm, embedding_dim); **(ii)** the name added to "
        "``snapshot_pretrained_init``'s closed choices; **(iii)** a "
        "``timm_kwargs_for`` branch -- see the launch hazard below"
    ),

    # ---- the hazard, measured rather than predicted ---------------------
    "THE_224_SHORTCUT_WOULD_KILL_THE_RUN_AT_LAUNCH": (
        "**measured, not reasoned.** ``timm_kwargs_for`` returns ``{}`` "
        "at (224, 224) -- the historical shortcut that keeps Road A's "
        "embeddings byte-identical. Every registered backbone is "
        "224-NATIVE, so ``{}`` is correct for all seven. **DINOv2 is "
        "518-native**, and building it with no kwargs and running a "
        "224 input raises ``AssertionError: Input height (224) doesn't "
        "match model (518)``. **This is the exact failure "
        "``timm_kwargs_for`` was written to prevent** -- its docstring: "
        "'written because the 768 seed band would have died at the "
        "patch-embed assertion'. It would die at launch, not at config "
        "time"
    ),
    "the_fix_and_its_caveat": (
        "``dynamic_img_size=True`` makes 224 work and returns 768-d -- "
        "verified. **But the position grid is then interpolated DOWN "
        "from 37x37 to 16x16.** Road B's ViT case interpolated UP "
        "(14x14 to 48x48) and the record treats grid interpolation as "
        "'part of the procedure being measured, not a defect'. **This "
        "is the first DOWNWARD interpolation in the project and it is a "
        "large one**; it is a declared caveat on the arm, not a hidden "
        "step"
    ),
    "the_resolution_ladder_is_UNREACHABLE_for_this_backbone": (
        "**patch 14 against staging's multiples of 16.** 224 satisfies "
        "both. **Road B's 512 and 768 divide by neither 14**, so "
        "``timm_kwargs_for`` would raise ``FactoryError``. Reachable "
        "sizes are multiples of 112. **Recorded so nobody proposes a "
        "resolution sweep for this arm without new staged sizes**"
    ),
    "the_head_does_NOT_change": (
        "**measured: DINOv2-base is 768-dim, 86,579,712 parameters** "
        "against ViT-B/16's 768-dim, 86,743,224. **Same embedding "
        "width, so the head is the same 769 fitted parameters** and the "
        "capacity axis is not confounded with the backbone axis. "
        "Verified by building the architecture, not by assuming ViT-B "
        "means 768"
    ),
    "cost": (
        "**one extraction pass over 237 patients through a frozen "
        "backbone, plus a head refit at five seeds.** **NO banked "
        "extraction-pass cost exists in the record** to quote -- the "
        "only banked GPU figure is the annex's 3.47 GPU-hours for 2,500 "
        "fits. The cost is stated as small and MEASURED at the run, not "
        "asserted here"
    ),
}



#: **[RULED 2026-09-02] TWO ARMS, not one.**
THE_TWO_ARMS_RULED = {
    "ruled": "2026-09-02 -- option (b), two arms",
    "the_ground": (
        "**one arm would measure the axis the record says matters LESS "
        "(source, 0.0017) tangled with the one it says matters MORE "
        "(encoder, ~0.08), and return a result attributable to "
        "neither.** Two arms untangle them for one extra extraction "
        "pass. **The cost is one pass; the alternative is an "
        "uninterpretable number**"
    ),
    "the_pairs_logic_stated_before_the_numbers": (
        "**if BOTH move together, the OBJECTIVE is doing the work** -- "
        "the one thing D1 and D2 share is self-supervision. **If only "
        "D2 moves, it is the encoder, the patch size or the native "
        "scale**, and the phase says plainly that it cannot separate "
        "those three. The logic is written here so neither outcome can "
        "be given its reading afterwards"
    ),
    "why_this_is_not_two_bites": (
        "**the family is FIXED AT THREE before any number** "
        "(THE_FAMILY_OF_THREE) and every outcome is reported. Adding "
        "the control is not adding a chance to win; it is the "
        "difference between an attributable result and an "
        "uninterpretable one"
    ),

    "arm_D2": {
        "checkpoint": "vit_base_patch14_dinov2.lvd142m",
        "registry_name": "vit_b14_dinov2",
        "what_it_is": (
            "DINOv2 base, self-supervised on LVD-142M. **768-d, so the "
            "head stays 769 parameters** and capacity is not confounded "
            "with the backbone"
        ),
        "what_it_varies": (
            "**objective, patch size (14 vs 16) and native resolution "
            "(518 vs 224) TOGETHER**, against the probe"
        ),
        "it_is_NOT_a_clean_control": (
            "**and must never be described as one.** A movement here is "
            "attributable to no single factor. This sentence is a "
            "tested literal (LIMITS_AS_LITERALS)"
        ),
    },
    "arm_D1": {
        "checkpoint": "vit_base_patch16_224.dino",
        "registry_name": "vit_b16_dino",
        "what_it_is": (
            "**the SAME-ENCODER control**: patch 16, native 224, "
            "self-supervised. **The same timm architecture string as the "
            "probe's ``vit_base_patch16_224`` with a different pretrained "
            "tag**, which is what makes the encoder exactly fixed"
        ),
        "what_it_isolates": (
            "**the OBJECTIVE, on a fixed encoder**, against the probe's "
            "ImageNet-supervised ViT-B/16"
        ),
        "it_is_DINO_not_DINOv2_and_that_limit_travels": (
            "**a DIFFERENT self-supervised method** -- original DINO "
            "(self-distillation, ImageNet-1k, no labels) rather than "
            "DINOv2 (LVD-142M, curated, with registers and a different "
            "recipe). **So the control isolates THE OBJECTIVE, not "
            "DINOv2 specifically**: if D1 moves and D2 does not, that is "
            "not evidence about DINOv2. Recorded because the natural "
            "misreading is that D1 is 'DINOv2 at patch 16', and it is "
            "not"
        ),
    },
    "everything_else_held": (
        "**geometry g1, resolution setting 224, head, learning rate, "
        "epochs, patience, monitor, label and the five seeds are the "
        "probe's own, copied from its config** rather than retyped -- so "
        "the single varying factor per arm is the backbone, by "
        "construction"
    ),
}


#: **[MEASURED 2026-09-02] The checkpoints resolve -- HERE, not in the
#: pinned image.**
THE_CHECKPOINTS_RESOLVED = {
    "measured": "2026-09-02, laptop",
    "both_resolve": (
        "**both tags are present with pretrained weights.** "
        "``vit_base_patch16_224.dino``: input 224x224, num_classes 0. "
        "``vit_base_patch14_dinov2.lvd142m``: input 518x518, "
        "num_classes 0, num_features 768, 86,579,712 parameters"
    ),
    "WHERE_that_was_measured": (
        "**timm 1.0.27 / torch 2.13 under Python 3.13 on the laptop.** "
        "**NOT the pinned image**, and not the suite's interpreter "
        "either -- the 3.11 venv ``scripts/test.ps1`` runs has neither "
        "timm nor torch, so the tests asserting these facts SKIP there"
    ),
    "THE_BLOCKING_READ_IS_NOT_CLOSED": (
        "**the cluster is pinned at timm 1.0.7 "
        "(ladder.PHASE_7D_BUILT, measured against the pinned install), "
        "and nothing about 1.0.7's model index is verifiable from this "
        "machine.** Resolving a tag in 1.0.27 is not resolving it in "
        "1.0.7. **If either tag is absent from the pin, neither config "
        "can run**, and the fix is a decision about the pin -- which "
        "touches every other arm's reproducibility -- not an edit to "
        "these files. The configs say so in their own headers"
    ),
    "an_observation_that_changes_nothing_here": (
        "1.0.27 also carries ``vit_base_patch16_dinov3.lvd1689m`` -- a "
        "patch-16 DINOv3, which WOULD be a same-encoder control for the "
        "DINOv2 family rather than for self-supervision generally. "
        "**Recorded as an observation, not proposed**: it is a third "
        "method again, it is far newer than the pin, and the arms are "
        "ruled"
    ),
}


#: **[RECORDED 2026-09-02] The weights are factory-pretrained, like the
#: probe's -- and that is the reason, not the convenience.**
THE_WEIGHTS_ARE_FACTORY_PRETRAINED_LIKE_THE_PROBES = {
    "the_decision": (
        "both arms use ``init`` values that load **no declared "
        "checkpoint**: the factory's own pretrained weights for the tag, "
        "which is exactly what ``init: imagenet`` has always meant "
        "(``train.extract``: 'init imagenet loads no checkpoint -- the "
        "factory's pretrained weights ARE that init')"
    ),
    "why_not_declared_snapshots": (
        "**the probe they are compared against takes its weights the "
        "same way.** Declaring these and not the probe would make the "
        "comparison asymmetric IN PROVENANCE -- on a one-factor "
        "contrast, that is the factor moving twice. Symmetry with the "
        "comparator beats a stricter declaration that only one side "
        "gets"
    ),
    "why_they_are_NOT_called_imagenet": (
        "**because they are not.** DINOv2's corpus is LVD-142M; DINO's "
        "is ImageNet-1k **without labels**, which is a different "
        "pretraining from ImageNet classification. The set directory "
        "carries the init in its name, so borrowing 'imagenet' would put "
        "three provenances under one name -- **the R2 shape**. The two "
        "new values are ``dinov2_lvd142m`` and ``dino_in1k``"
    ),
    "one_definition_not_two": (
        "``extract.FACTORY_PRETRAINED_INITS`` is the single list, and "
        "``config.schema``'s checkpoint requirement IMPORTS it rather "
        "than repeating it -- a second copy would drift the day a fourth "
        "init is added"
    ),
}


#: **[FIXED AT THREE 2026-09-02, BEFORE ANY NUMBER] The contrast family.**
THE_FAMILY_OF_THREE = {
    "fixed": "2026-09-02, before any arm ran",
    "count": 3,
    "primaries": (
        "**D2 vs the 0.2520 probe** and **D1 vs the 0.2520 probe** -- "
        "each under the full PLAN 4.3 criterion, both conditions"
    ),
    "secondary": (
        "**D2 vs D1** -- the two self-supervised arms against each "
        "other, which is the only contrast in the family that holds the "
        "objective fixed and varies the encoder"
    ),
    "no_alpha_correction": (
        "**on the standing ground: the count is FIXED BEFORE ANY NUMBER "
        "and every outcome is reported.** No contrast is added, dropped "
        "or selected after a result exists, so there is no multiplicity "
        "to correct for -- the family is the unit and it is declared"
    ),
    "the_arms_are_NOT_independent": (
        "**D2 and D1 share the probe as comparator and share the "
        "self-supervised objective**, so the three results are "
        "correlated. That is why the readings include COMBINATION "
        "cells rather than three separate verdicts read side by side "
        "(READINGS_COMMITTED)"
    ),
}


#: **[COMMITTED 2026-09-02, BEFORE ANY NUMBER] The readings -- three
#: contrasts both ways, and the COMBINATION cells, because the arms are
#: not independent.**
#:
#: **[SUPERSEDES the one-arm readings written at the restate.** Those were
#: written when the phase had one arm; the ruling made it two, and three
#: separate verdicts read side by side would not say what the pair
#: measures. The one-arm expectation is unchanged and restated below.]**
READINGS_COMMITTED = {
    "committed": "2026-09-02, before either arm is built",

    # ---- the combination cells, which are the point of two arms -------
    "cell_BOTH_above": (
        "**the OBJECTIVE is doing the work, and self-supervised features "
        "are better here.** The one thing D1 and D2 share is "
        "self-supervision; D1 shares its encoder with the probe. Both "
        "moving is the only cell that licenses an objective claim, and "
        "it licenses it about SELF-SUPERVISION IN GENERAL no further "
        "than two checkpoints reach"
    ),
    "cell_D2_above_D1_not": (
        "**the encoder, the patch size, or the native scale -- and the "
        "phase says WHICH IT CANNOT SEPARATE.** Three candidate causes, "
        "no arm distinguishing them, and no fourth arm is added after "
        "the fact (EXIT_CRITERIA). The result is reported with its three "
        "candidates named"
    ),
    "cell_D1_above_D2_not": (
        "**a result requiring EXPLANATION rather than a story, and it is "
        "recorded as an OBSERVATION.** The smaller, older, "
        "ImageNet-scale self-supervised model beating the larger "
        "LVD-142M one would be genuinely odd; the honest response is to "
        "report it and say the phase does not know why. **No mechanism "
        "is to be invented for this cell** -- naming it in advance is "
        "what stops one being invented"
    ),
    "cell_NEITHER": (
        "**the EXPECTED outcome. The twelfth and thirteenth "
        "measurements agree with the eleven, and the feature-source axis "
        "CLOSES BY MEASUREMENT rather than by assumption.** The write-up "
        "gains the sentence it cannot currently write: not 'we did not "
        "try', but 'we tried two self-supervised checkpoints, one of "
        "them a same-encoder control, and neither moved it'"
    ),

    # ---- each contrast, both ways -------------------------------------
    "D2_vs_probe": (
        "**claimable above** -> the first feature source in twelve "
        "measurements to move it, reported with the attribution bound "
        "(THE_ATTRIBUTION_PROBLEM) and never as an objective claim. "
        "**Unresolved or below** -> point estimate with its interval, no "
        "claim"
    ),
    "D1_vs_probe": (
        "**claimable above** -> the objective moves it ON A FIXED "
        "ENCODER, which is the strongest single sentence this phase "
        "could produce. **Unresolved or below** -> the objective does "
        "not move it at fixed encoder, which is the cleanest of the "
        "three nulls"
    ),
    "D2_vs_D1_secondary": (
        "**holds the objective fixed and varies the encoder.** "
        "Claimable either way -> the encoder axis is live within "
        "self-supervised features too, corroborating the patch-family "
        "evidence (THE_ELEVEN_AND_THE_SHARPEST). Unresolved -> reported "
        "as unresolved; a secondary is not promoted to carry the phase"
    ),

    "the_expectation_registered_explicitly": (
        "**the record predicts UNRESOLVED, on all three.** Eleven prior "
        "measurements, best case parity; the probe's seed sd is 0.0148 "
        "and the criterion needs both conditions. **A phase that "
        "predicts its own null and gets it has learned something about "
        "the axis** -- it converts an assumption into a measurement, "
        "which is the whole of what is being bought"
    ),
    "no_cell_is_ranked_as_the_good_one": (
        "all four combination cells produce a write-up sentence the "
        "project does not currently have. **The phase is not hoping for "
        "any of them**"
    ),
}


#: **[BOUND 2026-09-02] What this phase does NOT test.**
WHAT_THIS_DOES_NOT_TEST = {
    "one_checkpoint_not_self_supervision": (
        "**it tests ONE self-supervised checkpoint, not "
        "self-supervision.** DINOv2-base/LVD-142M is one point in a "
        "large family; a null here does not close self-supervised "
        "features in general, and a gain does not open them"
    ),
    "not_resolution_geometry_or_head": (
        "**all three are HELD at the probe's values so the single "
        "factor is the backbone.** No resolution sweep (and the ladder "
        "is unreachable for patch 14 anyway), no geometry variation, no "
        "head change (the width is 768 either way). **A phase that "
        "varied any of them would not be measuring the backbone**"
    ),
    "not_the_attribution": (
        "the encoder/objective confound is a BOUND, not an omission "
        "(THE_ATTRIBUTION_PROBLEM): the isolating control does not "
        "exist in timm"
    ),
    "the_downward_interpolation_rides_along": (
        "**the 37x37 -> 16x16 position-grid interpolation is part of "
        "the arm, not a nuisance parameter.** A null could be the "
        "checkpoint or could be that interpolation, and the phase "
        "cannot separate them either. Stated with the other bounds "
        "rather than discovered in the discussion"
    ),
}


#: **[DRAFT 2026-09-02 -- NOT LOCKED] Exit criteria.**
EXIT_CRITERIA_DRAFT = {
    "status": "DRAFT, NOT LOCKED",
    "1_the_reckoning_states_the_counter_evidence_first": (
        "the phase records the eleven and the one-factor patch-family "
        "contrast BEFORE its own case, and does not soften the "
        "amendment's tempering"
    ),
    "2_one_factor_verified_in_code": (
        "geometry, resolution, head, seeds, folds and label identical to "
        "the probe's, **checked against the probe's own config rather "
        "than asserted** -- the Phase 22 precedent, where the "
        "one-factor claim failed once before it was verified"
    ),
    "3_the_attribution_bound_travels_with_every_quotation": (
        "no output of this phase pairs a result with a claim about "
        "self-supervision. The bound is in the record before the number "
        "and in the run's own output after it"
    ),
    "4_the_full_criterion_both_conditions": (
        "the contrast is judged under PLAN 4.3 with BOTH conditions; "
        "an unresolved result is reported as unresolved with its point "
        "estimate and interval, never as 'no difference'"
    ),
    "5_the_224_hazard_is_closed_before_launch": (
        "``timm_kwargs_for`` handles the 518-native case explicitly and "
        "a test pins it, so the run cannot die at the patch-embed "
        "assertion. **The failure mode is measured and named in "
        "advance** (THE_ARM_REGISTERED)"
    ),
    "6_the_interpolation_is_declared_not_discovered": (
        "the 37x37 -> 16x16 position-grid interpolation appears in the "
        "arm's registration and in its output"
    ),
    "7_the_embedding_width_is_verified_at_extraction": (
        "the artifact's ``feature_dim`` is checked to be 768 against the "
        "declared ``embedding_dim``, so a silently different width "
        "refuses rather than changing the head's parameter count "
        "underneath the comparison"
    ),
    "8_no_second_arm_is_added_after_numbers_exist": (
        "the same-encoder control is measured UNAVAILABLE now; if one "
        "appears later it is a new registration, **not an addition to "
        "this phase once a result is visible**"
    ),
    "9_a_ledger_row_only_if_claimable": (
        "unresolved or below closes DESCRIPTIVE with no row; the ledger "
        "stays at 38 unless condition 1 and condition 2 both hold"
    ),
}


#: **[TO DECLARE 2026-09-02] The settings this phase needs ruled.**
SETTINGS_TO_DECLARE = {
    "0_BLOCKING_does_the_pinned_timm_carry_dinov2": (
        "**a blocking read, on the Phase 9 precedent.** The cluster is "
        "pinned at **timm 1.0.7**; this laptop measured 1.0.27. "
        "**Nothing about 1.0.7's model index is verifiable from here.** "
        "If 1.0.7 does not carry DINOv2, the phase needs either a pin "
        "change (which touches every other arm's reproducibility) or a "
        "checkpoint loaded outside timm -- **a decision, not a "
        "detail**"
    ),
    "1_the_checkpoint_identifier": (
        "**proposed ``vit_base_patch14_dinov2.lvd142m``** -- base, "
        "matching ViT-B/16's capacity. **Open**: the ``reg4`` variant "
        "(``vit_base_patch14_reg4_dinov2.lvd142m``) is the other base "
        "entry and adds register tokens; small/large/giant vary "
        "capacity, which would confound the backbone axis with size"
    ),
    "2_its_source": (
        "**open, and it matters for the declare.** timm downloads "
        "weights on first use; every other backbone's init is a "
        "declared ``snapshot_pretrained_init`` artifact. **Proposed: "
        "snapshot it the same way**, so the checkpoint is hashed and "
        "declared rather than fetched at run time"
    ),
    "3_output_dimensionality": (
        "**MEASURED: 768** (num_features 768, 86,579,712 parameters, "
        "verified by building the architecture). **So the head does not "
        "change** -- 769 fitted parameters, as the probe. Recorded as "
        "measured rather than left open, since the restate asked "
        "whether it changes"
    ),
    "4_the_224_handling": (
        "**open, and it is the one code decision.** "
        "``dynamic_img_size=True`` at 224 (verified working, 768-d out, "
        "grid interpolated 37->16) versus staging at 518 (which no "
        "staged artifact provides and which would change the "
        "resolution factor). **Proposed: dynamic_img_size at 224**, "
        "because holding resolution at the probe's value is what makes "
        "the backbone the single factor"
    ),
    "5_normalization": (
        "**MEASURED: ImageNet statistics**, read from the checkpoint's "
        "own ``pretrained_cfg`` -- **not** ViT-B/16's 0.5/0.5. **The "
        "values are deliberately NOT written here**: "
        "``tests/test_normalization`` forbids the ImageNet triple "
        "outside ``models/factory``, and it FIRED on this record's first "
        "draft, which had quoted them. The guard was right. "
        "``normalization_for`` already reads each model's own cfg, "
        "which is exactly why it exists ('Two backbones in the same "
        "registry disagreeing is the Stage-1 defect's precondition'). "
        "**Nothing to rule; recorded so it is not re-derived**"
    ),
}



#: **[LITERALS 2026-09-02] The limits, as tested strings.**
#:
#: Each is asserted verbatim by the suite, on the
#: ``phase24.A_CEILING_IS_NOT_A_SCORE`` precedent: a bound that lives only
#: in prose is a bound that gets paraphrased away.
LIMITS_AS_LITERALS = (
    "THIS PHASE TESTS TWO CHECKPOINTS, NOT SELF-SUPERVISION. DINOv2-base "
    "on LVD-142M and original DINO-B/16 on ImageNet-1k are two points in "
    "a large family. A null here does not close self-supervised features "
    "in general, and a gain does not open them.",
    "ARM D2 IS NOT A CLEAN CONTROL AND MUST NEVER BE DESCRIBED AS ONE. It "
    "varies objective, patch size and native resolution together against "
    "the probe, so a movement on D2 alone is attributable to no single "
    "factor. D1 is the control; D2 is the arm.",
    "RESOLUTION, GEOMETRY AND THE HEAD ARE HELD AT THE PROBE'S VALUES AND "
    "ARE NOT REVISITED. The single varying factor per arm is the "
    "backbone. No resolution sweep is available for D2 in any case: patch "
    "14 against staging's multiples of 16 leaves only multiples of 112.",
)


#: **[LOCKED 2026-09-02] EXIT_CRITERIA.**
EXIT_CRITERIA = {
    "locked": "2026-09-02 -- nine criteria, two arms, a family of three",
    "1_the_reckoning_states_the_counter_evidence_first": (
        "the phase records the eleven and the one-factor patch-family "
        "contrast BEFORE its own case, and does not soften the "
        "amendment's tempering"
    ),
    "2_one_factor_per_arm_verified_in_code": (
        "geometry, resolution setting, head, learning rate, epochs, "
        "patience, monitor, label, seeds and folds identical to the "
        "probe's, **checked against the probe's own config rather than "
        "asserted** -- the Phase 22 precedent, where a one-factor claim "
        "failed once before it was verified"
    ),
    "3_the_attribution_bound_travels_with_every_quotation": (
        "no output pairs a D2 result with an objective claim, and no "
        "output describes D2 as a control. LIMITS_AS_LITERALS is "
        "asserted by the suite"
    ),
    "4_the_full_criterion_on_all_three": (
        "every contrast is judged under PLAN 4.3 with BOTH conditions; "
        "an unresolved result is reported as unresolved with its point "
        "estimate and interval, never as 'no difference'"
    ),
    "5_the_518_native_hazard_is_closed_before_launch": (
        "``timm_kwargs_for`` handles the 518-native case explicitly and "
        "the suite pins it, so the run cannot die at the patch-embed "
        "assertion. **The failure mode was measured and named in "
        "advance**"
    ),
    "6_the_interpolation_is_declared_not_discovered": (
        "the 37x37 -> 16x16 position-grid interpolation appears in D2's "
        "registration, in its config header and in its output. **D1 has "
        "none, and that asymmetry is part of what D1 controls for**"
    ),
    "7_the_embedding_width_is_verified_at_extraction": (
        "each artifact's ``feature_dim`` is checked to be 768 against the "
        "declared ``embedding_dim``, so a silently different width "
        "refuses rather than changing the head's parameter count "
        "underneath the comparison"
    ),
    "8_the_family_is_three_and_no_arm_is_added_after_numbers_exist": (
        "two primaries and one secondary, fixed before any number, no "
        "alpha correction because nothing is selected. **A third arm to "
        "separate D2's three candidate causes is a NEW REGISTRATION**, "
        "not an addition to this phase"
    ),
    "9_a_ledger_row_only_where_claimable": (
        "each contrast earns a row only if both conditions hold; "
        "unresolved or below closes DESCRIPTIVE. The ledger stays at 38 "
        "otherwise"
    ),
    "the_settings": "THE_SETTINGS_RULED",
    "nothing_added_after": (
        "**NOTHING IS ADDED ONCE NUMBERS EXIST.** Not a third arm, not a "
        "fourth contrast, not a second checkpoint of either family, not "
        "a resolution variant, not a geometry variant. **The three "
        "candidate causes D2 confounds are named in advance precisely so "
        "that separating them later is visibly a new phase** rather than "
        "a follow-up this one licensed"
    ),
    "status": "LOCKED",
}


#: **[RULED 2026-09-02] The settings, with per-value provenance.**
THE_SETTINGS_RULED = {
    "ruled": "2026-09-02 -- two arms, (b)",
    "d2_checkpoint": {
        "value": "vit_base_patch14_dinov2.lvd142m",
        "provenance": "RULED; RESOLVED in timm 1.0.27 (laptop)",
    },
    "d1_checkpoint": {
        "value": "vit_base_patch16_224.dino",
        "provenance": "RULED; RESOLVED in timm 1.0.27 (laptop)",
    },
    "embedding_width": {
        "value": 768,
        "provenance": (
            "MEASURED by building the architecture (num_features 768); "
            "**the head stays 769 parameters for both arms**"
        ),
    },
    "geometry": {"value": "g1", "provenance": "the probe's, copied from its config"},
    "resolution_setting": {
        "value": 224,
        "provenance": (
            "the probe's. **D2 reaches it via dynamic_img_size** "
            "(37x37 -> 16x16 grid interpolation); D1 is 224-native and "
            "needs nothing"
        ),
    },
    "seeds": {
        "value": (1337, 2024, 7, 99, 12345),
        "provenance": "the probe's five, copied",
    },
    "head_and_recipe": {
        "value": "trainable: head, lr 0.001, 40 epochs, patience 5, inner_val_mse",
        "provenance": "the probe's, copied field by field",
    },
    "init_names": {
        "value": ("dinov2_lvd142m", "dino_in1k"),
        "provenance": (
            "**CHOSEN HERE and recorded as a choice**: neither is "
            "'imagenet', because neither is ImageNet-supervised "
            "(THE_WEIGHTS_ARE_FACTORY_PRETRAINED_LIKE_THE_PROBES)"
        ),
    },
    "normalization": {
        "value": "each checkpoint's own pretrained_cfg",
        "provenance": (
            "MEASURED as ImageNet statistics for both, **and the values "
            "are deliberately not written into this record**: "
            "tests/test_normalization forbids the triple outside "
            "models/factory, and it FIRED on the restate's first draft. "
            "The guard was right"
        ),
    },
    "the_pin": {
        "value": "timm 1.0.7 on the cluster",
        "provenance": (
            "**BLOCKING and UNVERIFIED from here** "
            "(THE_CHECKPOINTS_RESOLVED)"
        ),
    },
}



#: **[DEFECT 2026-09-02, FOUND ON THE CLUSTER] The extraction configs
#: named an init outside ``embeddings.INITS``.**
#:
#: Both p25 extraction configs failed at load, before any output:
#: ``EmbeddingError: unknown init 'dinov2_lvd142m'; expected one of
#: ('imagenet', 'scut_original', 'scut_masked')``.
THE_INIT_WAS_OUTSIDE_THE_VOCABULARY = {
    "found": "2026-09-02, on the cluster, ",
    "the_mechanism": (
        "**``task_extract_embeddings`` calls "
        "``embeddings.expected_variant(init, geometry)`` for every "
        "declared set, and that function refuses any init outside a "
        "CLOSED tuple.** The schema's ``init`` choices were extended for "
        "Phase 25; ``embeddings.INITS`` was not. The config validated, "
        "the generator agreed with itself, the suite passed, and the run "
        "died at the first set"
    ),

    # ---- item 1: what the vocabulary is FOR ---------------------------
    "what_INITS_is_actually_for": (
        "**NOT 'a weight source' in general -- the Phase 7 LADDER'S "
        "LATTICE.** Its docstring says so ('The three inits of the Phase "
        "7 ladder'), ``VARIANT_FOR_INIT`` maps an init to one of OUR "
        "SCUT pretraining runs, ``expected_variant``'s docstring is "
        "'which pretraining checkpoint this (init, geometry) pair must "
        "come from', and ``check_init_geometry`` exists to stop a "
        "masked-G1 checkpoint feeding G2. **A foundation checkpoint is "
        "not a variant of this project's own pretraining**"
    ),
    "why_the_fix_is_NOT_a_fourth_member_of_INITS": (
        "**because ``INITS`` is load-bearing arithmetic.** "
        "``expected_set_count`` multiplies by ``len(INITS)`` and "
        "``embedding_plan.required_sets`` iterates it, so **a fourth "
        "member would have turned the banked 24 embedding sets into 40 "
        "and the 12 pretraining runs into a different number** -- "
        "silently, and in a figure Phase 6 4 states. **That is the "
        "``LADDER_BACKBONES`` defect exactly**, in a different registry: "
        "'registering Phase 7D's three backbones for construction "
        "silently derived twenty-one new LADDER arms that no one "
        "designed'"
    ),
    "the_fix_follows_that_precedent": (
        "**membership of a LATTICE and membership of a VOCABULARY are "
        "different facts, and the iteration now names which one it "
        "means** -- the same shape as ``BACKBONES`` versus "
        "``LADDER_BACKBONES``. ``INITS`` stays closed at three and keeps "
        "every count; ``FOUNDATION_INITS`` holds the two; ``ALL_INITS`` "
        "is what an ARTIFACT may carry and what ``expected_variant`` "
        "accepts. **The lattice is asserted still 24 / 12 / 3**"
    ),
    "and_the_answer_it_gives_is_TRUE_not_degenerate": (
        "the foundation inits map to variant ``None`` -- 'no pretraining "
        "checkpoint of ours' -- which is the same answer ``imagenet`` "
        "gives and a correct one, not a hole punched in the check. "
        "``check_init_geometry`` still refuses a crossed masked pair"
    ),
    "a_third_namespace_already_existed": (
        "**MEBeauty's.** ``extract_mebeauty_embeddings`` and "
        "``probe_mebeauty`` carry ``mebeauty_masked`` / "
        "``mebeauty_original`` and NEVER reach ``expected_variant`` -- "
        "the schema says 'MEBeauty's OWN init namespace, never the "
        "ladder's'. **Found by the new sweep on its first run**, and it "
        "is why the sweep is scoped to the tasks that actually route "
        "init rather than to every init field"
    ),

    # ---- item 2: why nothing caught it --------------------------------
    "why_nothing_caught_it": (
        "**a config field whose value is validated against the SCHEMA's "
        "copy of a vocabulary and never against the CODE's.** The schema "
        "carries ``choices`` literals; ``embeddings`` carries the real "
        "tuple; nothing compared them. The generator's ``--check`` "
        "compares the config to the generator, not to the code, and the "
        "suite had no test that crossed the boundary"
    ),
    "the_guard_existed_for_ONE_FIELD_and_not_this_one": (
        "**``test_every_schema_backbone_choice_is_constructible_by_its_"
        "factory`` is this test, one field over**, and its docstring is "
        "this defect's ancestor: '[FOUND 2026-08-14] srgnn was added to "
        "the schema's train_cv backbone choices and not to "
        "phase3.BACKBONES. The config validated and the run died at "
        "make_factory after submission.' **That guard fired on this same "
        "build** and made Phase 25 add both names to ``phase3.BACKBONES``. "
        "There was no equivalent for ``init``, so init reached the "
        "cluster"
    ),
    "the_settings_sweep_would_NOT_have_caught_it": (
        "**and the reason is worth stating exactly.** The "
        "config-to-code sweep asks whether a declared value REACHES "
        "acting code. This one does -- it reaches "
        "``expected_variant`` and is REFUSED there. **Consumed and "
        "refused is a different failure from declared and ignored**, and "
        "a sweep built for the second cannot see the first"
    ),
    "the_error_class": (
        "**the fourth arrival of one-layer-at-a-time**: invented input "
        "names (p22), ``pair_source`` logged and never consumed (p22), "
        "the null ``condition_1`` computed and discarded (p22), and now "
        "a value a validator accepts and the code refuses. **Each was "
        "caught one layer further out than the last**"
    ),

    # ---- item 3 and 4 -------------------------------------------------
    "the_test_that_would_have_caught_it": (
        "**two, in tests/test_workflow_hygiene.** (i) every schema "
        "``init`` choice on a task that CALLS ``expected_variant`` must "
        "be accepted by it -- the routing derived from run.py rather "
        "than listed, and MEBeauty's separate namespace named rather "
        "than skipped; (ii) **every shipped config's init VALUE** is put "
        "through ``expected_variant``, which is the end the cluster "
        "failed at. **Before the fix (ii) fails on both p25 extraction "
        "configs with the cluster's own message**; after it, both pass "
        "and the lattice is asserted unchanged"
    ),
    "the_arm_configs_were_checked_and_are_CLEAN": (
        "**probed rather than reasoned about**, since the pattern has "
        "arrived one layer at a time four times. Every vocabulary the "
        "arm path touches was called with the new values: "
        "``embedding_plan.set_name`` -> "
        "``vit_b14_dinov2__dinov2_lvd142m__g1`` (which is the path the "
        "arm configs declare), ``extract.backbone_kind_for`` -> "
        "``transformer``, ``phase3.make_factory`` at ``trainable: head`` "
        "-> builds, ``embeddings.check_pairing`` consults no vocabulary "
        "(it compares config against the artifact's own metadata). "
        "**``checkpoint_input_name`` raises for a None variant and is "
        "never called for one** -- the call site guards on "
        "``if variant is not None``. **No second wall on the arm side**"
    ),
    "tag": "[FIXED] -- extraction not yet rerun; the launch is the maintainer's",
}



#: **[DEFECT + LIMITATION 2026-09-02, FOUND ON THE CLUSTER] The checkpoint
#: declaration the new inits cannot make -- and the provenance limitation
#: that is the honest answer.**
#:
#: Both arms failed at feature load: ``EmbeddingError: init 'dino_in1k' is
#: pretrained and the artifact must be traced to its checkpoint, and none
#: was declared``.
WHAT_THE_TWO_INITS_CAN_HONESTLY_DECLARE = {
    "found": "2026-09-02, on the cluster, ",

    # ---- item 1: what a checkpoint declaration IS ---------------------
    "what_a_checkpoint_declaration_is": (
        "**a DECLARED INPUT naming one of OUR pretraining runs' output "
        "directories, and its VERIFIED ROLLUP HASH.** The arm's config "
        "carries ``task.checkpoint`` (an input name); ``run`` resolves it "
        "to ``declared[task['checkpoint']]['rollup']``; "
        "``check_pairing`` compares that against the "
        "``checkpoint_sha256`` the EXTRACTION recorded in the artifact's "
        "metadata. **It is a hash of bytes this project produced**, not a "
        "name and not a config string"
    ),
    "where_the_three_existing_inits_declare_theirs": (
        "``scut_original`` and ``scut_masked`` come from the twelve "
        "pretraining runs and declare those checkpoint artifacts; "
        "``scut_masked`` is geometry-bound, so it declares a DIFFERENT "
        "one per geometry. **``imagenet`` declares none and is exempt** -- "
        "and the schema refuses a checkpoint on an imagenet arm, in both "
        "directions"
    ),
    "what_the_guard_was_actually_checking": (
        "that the artifact's recorded checkpoint hash EQUALS the arm's "
        "declared one -- 'training against features from a DIFFERENT "
        "representation than the one declared would run to completion and "
        "nothing downstream would raise'"
    ),

    # ---- the mechanism ------------------------------------------------
    "the_mechanism_the_reader_and_the_writer_disagreed": (
        "**``embeddings.save`` requires a checkpoint hash ``if variant is "
        "not None``; ``check_pairing`` required one ``if init != "
        "'imagenet'``.** Those are the same test until an init exists "
        "that is neither imagenet nor ours. **The extraction wrote "
        "``checkpoint_sha256: None`` LEGITIMATELY** -- the foundation "
        "inits have no variant -- **and the reader then demanded a "
        "declaration for a checkpoint that does not exist.** The artifact "
        "and the guard disagreed about what 'pretrained' means"
    ),

    # ---- item 2: the honest answer ------------------------------------
    "what_they_can_honestly_declare_NOTHING": (
        "**nothing, in this codebase's sense of the word, and that is a "
        "PROVENANCE LIMITATION rather than a gap to fill with a "
        "plausible value.** The declaration mechanism is a hash of OUR "
        "pretraining output. DINOv2 and DINO weights were downloaded by "
        "``timm`` at extraction time from an external source and **this "
        "project never hashed them**"
    ),
    "the_candidates_and_why_each_fails": (
        "**the timm tag** (``vit_base_patch14_dinov2.lvd142m``) is a "
        "NAME, not a hash: it pins a label, not bytes, and it is already "
        "recorded in ``BACKBONES`` and in the set directory's own name. "
        "**A hash of the cached weights file** would record what happened "
        "to be in the cache after the run, not what was declared before "
        "it -- and the cache is not a declared artifact. **The "
        "``pretrained_cfg``** is metadata, not a hash. **None of the "
        "three is a checkpoint declaration**, and offering one as though "
        "it were would be the plausible-looking value this record "
        "refuses"
    ),
    "THE_LIMITATION_IS_NOT_NEW_AND_THAT_IS_THE_POINT": (
        "**``imagenet`` has exactly the same limitation and the record "
        "never stated it.** The probe's own weights are a timm download "
        "too. So the two foundation inits are not a weaker case than the "
        "arm they are compared against -- **they are the same case, and "
        "Phase 25 is where the record finally says so.** Declared here "
        "as a limitation on the probe as much as on the arms"
    ),
    "the_route_that_WOULD_close_it_and_why_it_is_not_taken": (
        "**``snapshot_pretrained_init`` exists precisely for this** -- it "
        "was built to perform 'the LAST undeclared download', producing a "
        "declared, hashed artifact of factory weights. It is a DIFFERENT "
        "mechanism from ``checkpoint_sha256`` (it feeds ``init_dir`` at "
        "extraction, not the pairing check), it needs its own run, and "
        "**snapshotting these two while the probe's imagenet weights stay "
        "undeclared would break the provenance symmetry that makes the "
        "contrast one-factor** "
        "(THE_WEIGHTS_ARE_FACTORY_PRETRAINED_LIKE_THE_PROBES). "
        "**Available, costed, and not taken -- recorded as a decision**"
    ),

    # ---- item 3: the sweep --------------------------------------------
    "why_nothing_caught_it": (
        "**the last cycle's sweep bound ``init`` to the VOCABULARY; this "
        "guard is a REQUIREMENT that membership carries.** Passing "
        "``expected_variant`` says the name is known, not that everything "
        "keyed on it is satisfiable. **Third arrival of one-layer-deeper**: "
        "schema choices, then the vocabulary, then the pairing check at "
        "feature load"
    ),
    "the_full_init_keyed_sweep": (
        "**every site keyed on init, swept so this does not arrive a "
        "fourth time. CORRECT AS WRITTEN (8):** "
        "``embeddings.expected_variant`` (ALL_INITS); "
        "``embeddings.save`` (keyed on variant -- it was always right); "
        "``schema._check_train_cv_init``'s checkpoint rule and its "
        "artifact rule; ``extract._load_model`` (FACTORY_PRETRAINED_INITS, "
        "both directions); ``embedding_plan``/``expected_set_count`` "
        "(iterate INITS, the lattice, deliberately); the concat arm's "
        "imagenet-only rule; the pooling-axis rule, which is defined AT "
        "the imagenet init and should refuse a foundation one. "
        "**WRONG (2), both fixed here:** ``check_pairing``'s checkpoint "
        "requirement (the live failure) and its ``pretrain_scheme`` "
        "branch, which read ``init == 'imagenet'`` and let the foundation "
        "case fall through to a branch that happened to pass -- **luck, "
        "not a check**. **STALE MESSAGE (1):** "
        "``checkpoint_input_name``'s 'imagenet sets have no checkpoint "
        "input', now true of foundation sets too"
    ),
    "a_reverse_direction_nothing_checked": (
        "**found while fixing.** The guard checked that a DECLARED hash "
        "matches the artifact's; nothing checked the other way -- an "
        "artifact carrying a checkpoint hash while the arm declares none "
        "would have loaded silently. Now refused: 'the set came from a "
        "different init than the arm declares'"
    ),
    "tag": "[FIXED] -- the arms are not rerun; the launch is the maintainer's",
}



#: **[WITHDRAWN 2026-09-02 -- THE FIGURES IT RESTS ON DO NOT EXIST.
#: EVERYTHING BELOW IS PRESERVED AS WRITTEN.]**
#:
#: The +0.0656 this record reasons about was computed from arm means that
#: are not in the runs. The arms measure **D1 0.2393 and D2 0.1326**
#: against the probe's 0.2520, so there is no positive delta to sit
#: inside any band (``THE_FABRICATED_FIGURES_WITHDRAWN``,
#: ``WHAT_THE_ARMS_ACTUALLY_SHOW``).
#:
#: **The reasoning is not withdrawn because it was wrong.** Its arithmetic
#: about the band, and its refusal to let a five-seed sweep substitute for
#: the criterion, would have been correct had the delta existed. It is
#: withdrawn because **it has no subject**.
#:
#: [ORIGINAL HEADER, PRESERVED] "The delta sits inside the band this
#: cohort has never resolved. D1's arm mean is the highest single-arm PCC
#: the project has produced. That does not make its contrast claimable,
#: and the reason is arithmetic rather than caution."
THE_DELTA_IS_INSIDE_THE_UNRESOLVED_BAND = {
    "WITHDRAWN_2026_09_02": (
        "**the +0.0656 does not exist.** D1 is 0.2393 and the probe is "
        "0.2520, so the delta is NEGATIVE. Nothing below is edited; the "
        "record is preserved and marked"
    ),
    "registered": "2026-09-02, before the contrast task runs",
    "the_delta": (
        "**+0.0656** against the probe -- the **largest positive delta "
        "this project has produced**, and the first arm to sit above the "
        "probe at every seed"
    ),
    "the_band_it_sits_in": (
        "**0.04-0.10 is the band the cohort has never resolved**, and "
        "**the smallest delta ever resolved here is 0.1386**. +0.0656 is "
        "less than half of that and squarely inside the unresolvable "
        "band. **A five-of-five-seeds-above result may therefore still "
        "return UNRESOLVED**"
    ),
    "why_that_is_not_a_contradiction": (
        "condition 1 asks whether every seed's paired BCa interval "
        "excludes zero; condition 2 asks whether the delta exceeds the "
        "two arms' combined seed uncertainty. **Five arm means above five "
        "arm means is neither of those questions.** An arm can win every "
        "seed and still have intervals that straddle zero, and PLAN 4.3 "
        "requires BOTH"
    ),
    "the_combination_is_ITSELF_the_finding_if_it_happens": (
        "**the largest positive delta the project has produced, and it "
        "still does not pass.** That is a statement about the COHORT'S "
        "RESOLUTION, not about DINO -- it says the instrument cannot "
        "certify a real-looking effect of this size at n=237 and five "
        "seeds. **It must be stated as such and not argued around**: no "
        "post-hoc appeal to the seed sweep, to the arm means, to how "
        "clean the separation looks, or to the direction being "
        "unambiguous. The registered reading already covers it -- point "
        "estimate with its interval, no claim"
    ),
    "and_if_it_DOES_pass": (
        "then it is the first feature source in twelve measurements to "
        "move the probe, reported with the attribution bound "
        "(THE_ATTRIBUTION_PROBLEM) and never as an objective claim about "
        "self-supervision. **Registered symmetrically, so neither outcome "
        "arrives with a reading written for it afterwards**"
    ),
    "tag": "[REGISTERED] -- before the numbers",
}


#: **[WITHDRAWN ENTIRELY 2026-09-02 -- THE PATTERN DID NOT OCCUR.
#: EVERYTHING BELOW IS PRESERVED AS WRITTEN.]**
#:
#: **D1-above-D2-not never happened.** D1 (0.2393) is BELOW the probe
#: (0.2520) and D2 (0.1326) is well below it, so the cell that fired is
#: **NEITHER** -- the registered expectation.
#:
#: **The token-density hypothesis is withdrawn ENTIRELY, not qualified**:
#: it was constructed to explain a 0.0679 gap between D1 and D2 that does
#: not exist at that size, in a direction that does not exist relative to
#: the probe. **It explains nothing.** Its Phase 7D citation ("196 tokens
#: 0.2520, 256 tokens 0.2280") remains true of Phase 7D and is not
#: withdrawn there; what is withdrawn is its use here.
#:
#: **This is the strongest reason a [REASONED] tag is not a small
#: thing.** The tag was correct -- the hypothesis was never concluded,
#: never tested, and named its own unfalsifiability. It still should not
#: have been written, because **the phenomenon it explained was never
#: measured.** A mechanism for a difference that does not exist is not a
#: cautious claim; it is a claim about nothing.
#:
#: [ORIGINAL HEADER, PRESERVED] "The combination cell that fired is the
#: one no reading called likely. READINGS_COMMITTED named four cells. The
#: arm means point at D1-above-D2-not ... the candidate mechanism is
#: tagged and NOT concluded."
THE_UNEXPECTED_CELL = {
    "WITHDRAWN_2026_09_02": (
        "**the pattern did not occur and the mechanism explains "
        "nothing.** D1 0.2393 and D2 0.1326 against the probe's 0.2520: "
        "the NEITHER cell fired, as registered. Nothing below is edited "
        "(THE_FABRICATED_FIGURES_WITHDRAWN)"
    ),
    "observed": "2026-09-02, from the two arms' run artifacts",
    "which_cell": (
        "**D1 above, D2 not** -- registered as the cell that would need "
        "explaining, and the one no reading called likely. The registered "
        "expectation was cell NEITHER, on eleven prior measurements"
    ),
    "it_is_an_OBSERVATION_beside_the_unfired_cells": (
        "**Phase 17 style**: the cells stand PRESERVED AND UNFIRED as "
        "written, this record sits beside them, and **no reading is "
        "amended after the fact**. What fired is what fired"
    ),
    "the_candidate_mechanism_REASONED_not_concluded": (
        "**[REASONED]** D1 varies **only the objective** -- the same "
        "ViT-B/16, patch 16, 224 input, **196 tokens** -- while D2 varies "
        "the objective **plus** patch 14, a 518 native scale and **256 "
        "tokens** at the probe's 224. And Phase 7D measured grid density "
        "directly, verbatim: *'49 tokens 0.2519, 196 tokens 0.2520, "
        "**256 tokens 0.2280**'*. **So D2's architecture may be giving "
        "back what its objective gains** -- 256 tokens cost 0.0240 on "
        "the supervised ladder, and D2 sits 0.0679 below D1"
    ),
    "the_arithmetic_is_SUGGESTIVE_and_NOT_a_decomposition": (
        "**0.0240 is not 0.0679 and this record does not pretend it "
        "is.** The 7D figure is a different backbone at a different "
        "input size on supervised weights; borrowing it as a correction "
        "term would be arithmetic across quantities. It is cited as a "
        "MEASURED reason the token axis is live on this cohort, nothing "
        "further"
    ),
    "the_phase_did_NOT_test_it_and_CANNOT_resolve_it": (
        "**two arms cannot separate three factors.** Objective, patch "
        "size and native scale move together in D2, and "
        "THE_ATTRIBUTION_PROBLEM registered that bound before either arm "
        "ran -- including that the isolating control does not exist in "
        "timm (no DINOv2 at patch 16, no supervised checkpoint at patch "
        "14). **This hypothesis is unfalsifiable within Phase 25 and is "
        "recorded as one, not as a conclusion**"
    ),
    "what_would_test_it": (
        "an arm holding the objective fixed and varying only the token "
        "count -- DINO-B/8 against DINO-B/16 at 224 would do it, both "
        "self-supervised, same corpus, same recipe. **NOT REGISTERED "
        "and not proposed**: naming it here is what stops a later turn "
        "presenting it as a fresh idea, per phase14's clause"
    ),
    "tag": "[OBSERVED] + [REASONED] -- no cell is amended",
}


#: **[FIXED AT THREE 2026-09-02] The contrast family, as ruled.**
#:
#: Two primaries and one secondary, declared before any arm ran
#: (``THE_FAMILY_OF_THREE``). **Derived here rather than listed in a
#: config**, so the family cannot grow through a config edit.
CONTRASTS = (
    {
        "key": "d2-vs-probe",
        "kind": "primary",
        "a": "p25_arm_d2", "b": "probe",
        "question": "does DINOv2-B/14 move the probe",
        "varies": "objective + patch size + native scale, together",
    },
    {
        "key": "d1-vs-probe",
        "kind": "primary",
        "a": "p25_arm_d1", "b": "probe",
        "question": "does the OBJECTIVE move the probe at a fixed encoder",
        "varies": "objective only -- the same ViT-B/16 at 224",
    },
    {
        "key": "d2-vs-d1",
        "kind": "secondary",
        "a": "p25_arm_d2", "b": "p25_arm_d1",
        "question": "the encoder axis WITHIN self-supervised features",
        "varies": "patch size + native scale, objective held",
    },
)


def contrast_family() -> list[dict]:
    """The three, and the count is asserted rather than trusted."""
    family = [dict(contrast) for contrast in CONTRASTS]
    if len(family) != 3:
        raise ValueError(
            f"the family is {len(family)} contrasts, not the ruled 3 "
            "(THE_FAMILY_OF_THREE: fixed before any number, no alpha "
            "correction because nothing is selected)"
        )
    if len({c["key"] for c in family}) != 3:
        raise ValueError("duplicate contrast keys")
    return family



#: **[WITHDRAWN 2026-09-02] THE ARM FIGURES WERE FABRICATED. They are not
#: in the runs, and nothing in the phase directory contains them.**
THE_FABRICATED_FIGURES_WITHDRAWN = {
    "withdrawn": "2026-09-02, on reading the artifacts",
    "what_was_withdrawn": (
        "**D1 mean 0.3176 (0.3080 / 0.3210 / 0.3223 / 0.3168 / 0.3200, "
        "sd 0.0055) and D2 mean 0.2497.** Neither mean, neither sd, and "
        "none of the five per-seed values is in any run artifact"
    ),
    "what_the_artifacts_say": (
        "``seed_variance.json``: **D1 mean 0.2393** (min 0.1721, max "
        "0.2949, 95% interval [0.1765, 0.2917]); **D2 mean 0.1326** (min "
        "0.0979, max 0.1724)"
    ),
    "the_grep_that_settles_it": (
        "**one occurrence of any quoted digit across the whole phase "
        "directory**: ``0.308062`` in ``seed_7__curves.csv``, a per-epoch "
        "INNER-VALIDATION value from one fold -- a different quantity, a "
        "different population and a different stage of training. **The "
        "five-value set does not exist anywhere.** So it was not a "
        "mis-read column or a wrong file: there was no source"
    ),
    "the_provenance": (
        "**a TRANSCRIPTION SLIP** -- the figures reached the record in "
        "the message that asked for the contrasts to be built, and "
        "**it was banked them into three records and one shipped log "
        "line without any route to check them.** The laptop cannot read "
        "the cluster, so there was nothing to check them against; that "
        "is the reason they went in unverified and it is not an excuse "
        "for it"
    ),
    "the_rule_that_should_have_stopped_it": (
        "**'never quote a hash from a chat log -- run declare_inputs.py "
        "and use what it prints'** is the standing rule, and **its "
        "principle is not about hashes.** A per-seed PCC vector is as "
        "unverifiable from a message as a rollup is. The rule was applied "
        "to hashes and not extended to figures, and **this phase's own "
        "contrast task was written to read the arms from their artifacts "
        "precisely because that is the right route** -- and then records "
        "were written from the message anyway"
    ),
    "what_it_cost": (
        "three records and one hardcoded log line, all withdrawn or "
        "corrected here; **no measurement, no ledger row, no banked "
        "figure elsewhere.** The contrasts had not run, which is the only "
        "reason the damage stops at prose"
    ),
    "what_is_NOT_affected": (
        "**the 0.0656 in phase21/phase22 is a DIFFERENT quantity** -- the "
        "minimum cross-arm shrinkage ratio (p16 identity baseline) -- and "
        "is untouched. So is phase22's 0.2497. Checked before anything "
        "was edited, because the withdrawal itself could otherwise have "
        "corrupted two unrelated records"
    ),
    "tag": "[WITHDRAWN] -- originals preserved, nothing edited in place",
}


#: **[MEASURED 2026-09-02, from the runs' own artifacts] What the arms
#: actually show.**
WHAT_THE_ARMS_ACTUALLY_SHOW = {
    "measured": "2026-09-02, read from seed_variance.json ",
    "the_three_numbers": (
        "**D1 (DINO-B/16) 0.2393. D2 (DINOv2-B/14) 0.1326. Probe "
        "0.2520.** Both self-supervised arms sit **at or below** the "
        "probe: D1 is 0.0127 under it and D2 is 0.1194 under it"
    ),
    "the_cell_that_fired_is_NEITHER": (
        "**the registered expectation, verbatim**: 'the twelfth and "
        "thirteenth measurements agree with the eleven, and the "
        "feature-source axis CLOSES BY MEASUREMENT rather than by "
        "assumption'. **The phase predicted its own null and got it**"
    ),
    "the_spread_is_WIDE_not_tight": (
        "**D1's seeds span 0.1721 to 0.2949 -- a range of 0.1228**, "
        "against the probe's seed sd of 0.0148. Any note that D1's "
        "variance was unusually tight is withdrawn: **it is wider than "
        "the probe's, not narrower**, and the withdrawn sd of 0.0055 "
        "would have been the tightest in the project"
    ),
    "no_claim_of_a_highest_arm": (
        "**D1 is not the project's highest single-arm PCC and no arm here "
        "is.** 0.2393 sits below the probe's 0.2520 and below Phase 12's "
        "concat at 0.2640. Withdrawn"
    ),
    "the_axis_closes_across_THIRTEEN": (
        "eleven prior measurements varied the feature source and none "
        "beat 0.2520 claimably; **these two are the twelfth and "
        "thirteenth, and neither does either.** The best alternative "
        "feature source in the project remains PARITY (MEBeauty +0.0017, "
        "vit_b32 -0.0001), and **self-supervised features are now "
        "measured rather than assumed**"
    ),
    "the_contrasts_have_NOT_run": (
        "**these are ARM MEANS, not verdicts.** No contrast has been "
        "computed: the family of three waits on the run directories "
        "(``owed()``). When it runs it tests THESE figures and not the "
        "withdrawn ones, and both PLAN 4.3 conditions decide each "
        "contrast -- an arm below the probe is not automatically a "
        "claimable negative"
    ),
    "tag": "[MEASURED] -- arm means only; no contrast, no ledger row",
}


#: **[FILED 2026-09-02] The transcription slip, beside its two
#: predecessors -- and it is a different failure from either.**
THE_TRANSCRIPTION_SLIP = {
    "filed": "2026-09-02",
    "the_family": (
        "**three figures that entered the record without a source.** The "
        "figure ``ladder.DETECTION_FLOOR_PROHIBITION`` forbids -- named "
        "by that record and deliberately NOT repeated here, since the "
        "prohibition is on the phrase itself -- and the **0.068** were "
        "figures RECALLED WRONGLY FROM ADJACENT QUANTITIES: each had a "
        "real source and was mis-attached to it. **This one had no source "
        "at all**"
    ),
    "why_that_is_worse_not_milder": (
        "a mis-recalled figure is findable: the adjacent quantity exists, "
        "the digits match something, and a sweep of the record turns it "
        "up. **A figure with no source is invisible to every check this "
        "project has** -- it matches nothing, contradicts nothing, and "
        "sits in prose that reads exactly like the prose around it. It "
        "was found by reading the artifacts, which is the only thing that "
        "could have found it"
    ),
    "the_shape_that_made_it_dangerous": (
        "**it was PLAUSIBLE and it was FLATTERING.** 0.3176 with a tight "
        "sd is what a genuine breakthrough looks like, and the record "
        "responded by building an explanation for it "
        "(THE_UNEXPECTED_CELL). **A figure that confirms nothing gets "
        "checked; a figure that rewards the phase gets elaborated.** That "
        "asymmetry is the finding here"
    ),
    "the_correction_to_practice": (
        "**a figure from a message is [REPORTED] until an artifact says "
        "otherwise**, exactly as a label construction is "
        "(phase9.EXTERNAL_VALIDATION_ON_THE_25_CONCEDED ground 4) and "
        "exactly as a hash is. **Records may cite it as reported; they "
        "may not reason from it.** The three withdrawn records all "
        "reasoned from it"
    ),
    "what_would_have_caught_it_earlier": (
        "**nothing in the suite, and that is not fixable from here** -- "
        "the laptop cannot read the cluster, so no test can compare a "
        "quoted figure against an artifact it cannot open. **What is "
        "fixable is not writing the record until the artifact is "
        "readable**, which is what the contrast task already does and "
        "what the prose did not"
    ),
    "tag": "[FILED] -- a practice correction, not a measurement",
}



#: **[RULED 2026-09-02] An exemption is NOISY, not silent --
#: and not time-capped.**
THE_EXEMPTION_IS_NOISY_NOW = {
    "ruled": "2026-09-02 -- option (1)",
    "what_went_wrong": (
        "``test_every_task_kind_has_at_least_one_config`` was exempted "
        "for ``p25_contrasts`` while its config was owed. **The exemption "
        "was dated, reasoned and self-removing, and it was still "
        "wrong**: the suite stayed GREEN, and the owed config surfaced "
        "three cycles later as a ``FileNotFoundError`` at launch. **A "
        "green suite is how this project reads its own state**, and an "
        "exemption that does not disturb it is a state nobody sees"
    ),
    "option_1_TAKEN_print_what_is_owed_on_every_run": (
        "**a standing, visible line on every suite run.** Implemented as "
        "a ``warnings.warn`` rather than a ``print``, because pytest "
        "captures stdout by default and surfaces warnings in its summary "
        "-- **noisy means noisy without a flag**. The registry is "
        "``tests/test_smoke_run.OWED_TASK_KINDS``, and every entry "
        "carries a PREDICATE, so an exemption expires on the condition "
        "that created it"
    ),
    "option_2_RECORDED_NOT_TAKEN_a_time_cap": (
        "fail the suite after N days. **Rejected, and the reason is the "
        "shape rather than the duration**: a cap converts a silent state "
        "into a SUDDEN FAILURE AT AN ARBITRARY DATE -- a different "
        "silence with a worse ending, and one that arrives when nobody "
        "is working on the thing it concerns. Recorded so the option is "
        "visibly weighed rather than unconsidered"
    ),
    "the_phase_25_entry_stands_as_the_worked_example": (
        "its predicate is 'the generator has no run directories', and "
        "**it now returns False** -- the names were supplied and the "
        "config is shipped. The entry stays as the example rather than "
        "being deleted, so the next owed kind has a shape to copy"
    ),
    "what_it_does_NOT_do": (
        "**it does not make an owed config acceptable.** It makes it "
        "loud. The rule that a task with no config is invisible stands, "
        "and the guard still FAILS for any kind without a documented "
        "entry -- the exemption still costs an edit, as its docstring "
        "always required"
    ),
    "tag": "[RULED] -- practice, not a measurement",
}


#: **[EMITTED 2026-09-02] The contrasts config, from the supplied names.**
THE_CONTRASTS_CONFIG_EMITTED = {
    "emitted": "2026-09-02",
    "the_run_pair": (
        "**``2d54e77a``**, identified BY MEASUREMENT rather than by "
        "recency: the ``0bf86712`` pair has **no log** (the pre-fix "
        "crashes) and the ``2d54e77a`` pair carries **five seeds each** "
        "with a ``seed_variance.json`` giving the banked means (D1 "
        "0.2393, D2 0.1326). Both names are baked into the generator and "
        "the live sha8 is recorded, so a later reader need not re-derive "
        "which pair is which"
    ),
    "why_the_names_are_BAKED_IN_now": (
        "``--run-dirs`` was the intake route while they were unknown. "
        "**A ``--check`` that needs a file at the drive root checks "
        "nothing for anyone else** -- and it had a hole: with an empty "
        "mapping the contrasts config was not built, so ``--check`` did "
        "not compare it and reported '4 configs match' **while a fifth "
        "sat shipped and unverified**. Baking closes that; bare "
        "``--check`` now covers all five"
    ),
    "a_supplied_file_may_CONFIRM_never_replace": (
        "``--run-dirs`` still works and now REFUSES a path that "
        "contradicts the baked one: two directories for one arm means "
        "one of them is not the run that produced its numbers. Changing "
        "the run is a deliberate edit to the generator"
    ),
    "what_pends": (
        "**the two rollups, and only those.** The manifest and the "
        "probe's run directory carry hashes copied byte-identically from "
        "configs that already declare them"
    ),
    "tag": "[EMITTED] -- two hashes pend on the declare",
}



#: **[DEFECT 2026-09-02, THE FOURTH OF ONE CLASS IN THIS PHASE] A shipped
#: loader called with an argument shape it does not accept.**
THE_ARM_DICT_LACKED_THE_LOADERS_KEY = {
    "found": "2026-09-02, at launch",
    "the_mechanism": (
        "``_p21_load_arm_predictions`` opens "
        "``seed_<n>__{arm['csv']}.csv``, so **every arm dict must carry "
        "``csv``, the prediction-file STEM.** Phase 21 and 22 declare it "
        "per arm in their configs; Phase 25's ``arms`` spec declared "
        "``name``, ``input``, ``seeds`` and ``n_patients`` and not that "
        "one. ``KeyError: 'csv'`` at the first file open"
    ),
    "the_correct_value_READ_not_assumed": (
        "**``predictions``**, from ``phase3.write_outputs``: "
        "``ctx.path(f\"{prefix}predictions.csv\", tier=\"CLUSTER-ONLY\")``, "
        "with every task passing ``prefix=f\"seed_{seed}__\"``. Read from "
        "the WRITER rather than inferred from filenames on disk"
    ),
    "why_it_is_a_CONFIG_field_and_not_a_task_constant": (
        "**because it genuinely varies.** 461 shipped arm entries say "
        "``predictions`` and **7 say ``identity_predictions``** (Phase "
        "16's identity baseline, in p18_metric_space and both p21 "
        "configs). A stem hardcoded in this task would make Phase 25's "
        "call differ from every other caller's for no reason, and would "
        "hide a quantity the record varies"
    ),
    "the_loader_was_NOT_loosened": (
        "**Phase 21 and 22 depend on it and it is shipped.** The fix "
        "supplies what it requires: one declared field, and the "
        "generator emits it for all three arms"
    ),
    "the_class_and_the_pattern": (
        "**fourth of one class in this phase alone, each found at LAUNCH "
        "and each one layer deeper**: an init outside "
        "``embeddings.INITS`` (at ``expected_variant``), a checkpoint "
        "requirement the new inits cannot meet (at ``check_pairing``), a "
        "task with no config (at ``load_config``), and now a shipped "
        "function called with a short dict (at the first file open). "
        "**Each fix was correct and none of them looked further in**"
    ),
    "what_ends_it": (
        "**the end-to-end fixture test, which Phase 22 built for exactly "
        "this and Phase 25 did not have.** "
        "``tests/test_p25_contrasts_end_to_end`` drives the task through "
        "the REAL ``RunContext``, ``TASKS`` entry, loader and "
        "``paired_comparison`` against three small fixture run "
        "directories. **Pre-fix it raises ``KeyError: 'csv'``; post-fix "
        "it runs to three verdicts** -- and it would have caught all "
        "three earlier defects, each of which dies on the same path"
    ),
    "the_audit_that_should_have_been_one_pass": (
        "**every call this task makes into shipped code, audited in one "
        "pass** rather than one launch per defect (the Phase 22 "
        "signature-audit precedent). It closes clean, and it now checks "
        "**required DICT KEYS as well as parameters** -- because this "
        "defect was invisible to a signature check: the call was "
        "well-formed and the dict was short"
    ),
    "tag": "[FIXED] -- the contrasts are not rerun; the launch is the maintainer's",
}



#: **[VOID 2026-09-02] The pre-fix contrast launch.**
CONTRAST_LAUNCH_VOID = {
    "run": "p25_contrasts__195416d1",
    "void": "2026-09-02",
    "what_happened": (
        "**seven attempts, every one dying on the same "
        "``KeyError: 'csv'``** in ``_p21_load_arm_predictions``. "
        "**Nothing was scored** -- no verdict, no arm statistic, no "
        "metrics.json"
    ),
    "VOID_not_superseded": (
        "**the distinction matters and the record already draws it.** A "
        "SUPERSEDED run's numbers are correct and a later run says more "
        "(THE_RERUN_IS_THE_CITABLE_RUN, Phase 23). **This one produced "
        "nothing at all**, so there is nothing to supersede. Void is the "
        "ledger's own word for it (results_ledger "
        "``void-deall-first-launch``, ``void-cleftgnn-first-launch``)"
    ),
    "no_ledger_row": (
        "the two existing VOID rows are launches of CLAIMING phases; "
        "Phase 25's contrasts are DESCRIPTIVE and register no row, so "
        "the void is recorded here rather than in the ledger"
    ),
    "tag": "[VOID] -- nothing scored",
}


#: **[MEASURED 2026-09-02] THE THREE VERDICTS, from the run's own
#: artifacts.** Run ``p25_contrasts__c41d40a6__p25-contrasts``.
THE_THREE_VERDICTS_OBSERVED = {
    "run": "p25_contrasts__c41d40a6__p25-contrasts",
    "observed": "2026-09-02",
    "arms_as_measured": (
        "**D1 (DINO-B/16) 0.2393, sd 0.0469. D2 (DINOv2-B/14) 0.1326, sd "
        "0.0319. Probe 0.2520, sd 0.0148.** Each arm's mean and sd are "
        "computed from its own per-seed PCCs in the task, not declared "
        "-- which is why ``winner_sd`` is not a config field"
    ),
    "verdicts": {
        "d2-vs-probe": {
            "kind": "primary",
            "mean_delta": -0.1194,
            "condition_1": False,
            "condition_2": True,
            "verdict": "unresolved",
        },
        "d1-vs-probe": {
            "kind": "primary",
            "mean_delta": -0.0127,
            "condition_1": False,
            "condition_2": False,
            "verdict": "unresolved",
        },
        "d2-vs-d1": {
            "kind": "secondary",
            "mean_delta": -0.1067,
            "condition_1": False,
            "condition_2": True,
            "verdict": "unresolved",
        },
    },
    "all_three_UNRESOLVED": (
        "**not one contrast is claimable, in either direction.** Under "
        "PLAN 4.3 both conditions must hold, and **condition 1 fails on "
        "all three** -- no contrast has every seed's paired BCa interval "
        "excluding zero in one direction"
    ),
    "the_deltas_reconcile_with_the_arm_means": (
        "**a free check, and it passes.** 0.1326 - 0.2520 = -0.1194; "
        "0.2393 - 0.2520 = -0.0127; 0.1326 - 0.2393 = -0.1067. The "
        "paired deltas are computed per seed over patients and the arm "
        "means over seeds, so agreement to four decimals is a "
        "consistency check rather than an identity"
    ),
    "d1_is_NOMINALLY_below_and_unresolvable_either_way": (
        "**-0.0127 with BOTH conditions false** -- the only contrast in "
        "the family that fails condition 2 as well. It is the weakest "
        "possible result: the arm is nominally behind and the cohort "
        "cannot order the two at all"
    ),
    "tag": "[MEASURED] -- DESCRIPTIVE, no ledger row",
}


#: **[FIRED 2026-09-02] The registered NEITHER cell, and it was the
#: phase's predicted outcome.**
THE_NEITHER_CELL_FIRED = {
    "fired": "2026-09-02",
    "the_committed_reading_VERBATIM": (
        "READINGS_COMMITTED['cell_NEITHER'], as written before either "
        "arm ran: **'the EXPECTED outcome. The twelfth and thirteenth "
        "measurements agree with the eleven, and the feature-source axis "
        "CLOSES BY MEASUREMENT rather than by assumption.' The write-up "
        "gains the sentence it cannot currently write: not \'we did not "
        "try\', but \'we tried two self-supervised checkpoints, one of "
        "them a same-encoder control, and neither moved it\'**"
    ),
    "it_was_PREDICTED_before_the_arms_ran": (
        "``READINGS_COMMITTED['the_expectation_registered_explicitly']``: "
        "**'the record predicts UNRESOLVED, on all three'** -- and all "
        "three are unresolved. **A phase that predicts its own null and "
        "gets it has learned something about the axis**; that sentence "
        "was registered before the numbers and is now earned rather than "
        "claimed"
    ),
    "neither_checkpoint_beats_the_probe": (
        "**D1 nominally below and unresolvable either way; D2 well "
        "below.** Neither is a claimable negative either -- the phase "
        "measures that self-supervised features are not better here, not "
        "that they are worse"
    ),
    "no_cell_was_amended": (
        "the four combination cells stand as written. **The withdrawn "
        "D1-above-D2-not observation was withdrawn last cycle**, before "
        "these numbers existed, so nothing was adjusted to fit them"
    ),
}


#: **[DERIVED 2026-09-02] The feature-source axis closes -- and the count
#: is recomputed rather than incremented.**
THE_AXIS_CLOSES_DERIVED = {
    "derived": "2026-09-02, from THE_ELEVEN_AND_THE_SHARPEST",
    "the_recount_and_what_it_found": (
        "**the list holds ELEVEN entries, of which TEN vary the feature "
        "source.** Entry 11 is Phase 22's objective result, and the "
        "record itself labels it *'not feature SOURCE, but the same "
        "lesson on the adjacent axis'*. **So the strict feature-source "
        "count is 10 + 2 = TWELVE**, while the record's own 'eleven' "
        "convention -- which counts the adjacent objective axis -- gives "
        "11 + 2 = THIRTEEN"
    ),
    "which_number_goes_where": (
        "**the committed reading says 'the twelfth and thirteenth', so "
        "THIRTEEN is what the registered sentence means** and it is "
        "banked unchanged. **TWELVE is the strict count** and is banked "
        "beside it. Deriving rather than incrementing is what surfaced "
        "the difference; incrementing would have carried it silently"
    ),
    "the_axis_measured": (
        "pretraining dataset (ImageNet / SCUT / MEBeauty), geometry "
        "(whole / masked), architecture (patch 32 / 16 / 8, MViTv2, "
        "Swin), resolution (224 / 512 / 768), and now **pretraining "
        "OBJECTIVE (supervised / self-supervised)** -- the last being "
        "the one axis nothing had touched"
    ),
    "what_closing_it_BUYS": (
        "**'we never tried self-supervised features' was a gap the "
        "write-up would otherwise carry as an untested assumption. It is "
        "now a measurement.** The difference is what survives a reviewer "
        "asking why: an assumption invites the question, a null answers "
        "it. Nothing else about the project's conclusions moves"
    ),
    "what_it_does_NOT_close": (
        "**self-supervision in general.** Two checkpoints, one family "
        "each, and the bounds registered before the arms ran stand "
        "(LIMITS_AS_LITERALS): one checkpoint is not a method, D2 is not "
        "a clean control, and resolution, geometry and head were held"
    ),
}


#: **[OBSERVED 2026-09-02] The seed instability -- which no reading
#: anticipated.**
THE_SEED_INSTABILITY_OBSERVED = {
    "observed": "2026-09-02",
    "the_arithmetic": (
        "**D1's seed sd is 0.0469 against the probe's 0.0148 -- 3.17x. "
        "D2's is 0.0319 -- 2.16x.** Both computed from the arms' own "
        "per-seed PCCs in the run, and both stated as ratios because the "
        "probe's 0.0148 is the band every threshold in this project is "
        "built from"
    ),
    "what_it_says": (
        "**the self-supervised features are not only no better here -- "
        "they are markedly LESS STABLE across seeds.** A wider seed band "
        "also widens the criterion's own threshold, so an unstable arm "
        "is harder to resolve in either direction: instability is not "
        "neutral, it costs resolution"
    ),
    "no_reading_anticipated_it": (
        "**none of the four combination cells mentions seed stability**, "
        "and neither does the attribution bound. It is recorded as an "
        "OBSERVATION beside them, not as an amendment to any"
    ),
    "it_is_NOT_a_mechanism": (
        "**nothing in this phase tests WHY**, and the record does not "
        "guess. Candidates exist -- a 768-d representation carrying less "
        "linearly-decodable grade signal would make the head's fit more "
        "seed-sensitive; DINOv2's downward grid interpolation is another "
        "-- **and naming them here is not proposing them.** The phase "
        "varied the backbone and measured the spread; it did not "
        "decompose it"
    ),
    "the_withdrawn_claim_it_replaces": (
        "**the withdrawn figures said D1's sd was 0.0055, which would "
        "have been the TIGHTEST in the project.** The measured 0.0469 is "
        "the opposite finding, 8.5x wider than the fabricated one -- "
        "recorded because the two point in opposite directions and a "
        "reader meeting both should know which is measured"
    ),
    "tag": "[OBSERVED] -- arithmetic only, no mechanism",
}


#: **[RECOUNTED 2026-09-02] The condition-2-pass / condition-1-fail
#: phenomenon, derived from the ledger's own fields.**
THE_CONDITION_SPLIT_RECOUNTED = {
    "recounted": "2026-09-02",
    "the_derivation": (
        "**EIGHT ledger rows**, derived by reading each entry's own "
        "``condition_1`` and ``condition_2`` fields -- indices 25, 26, "
        "27, 28, 31, 32, 34, 35 -- which **reproduces entry 37's list "
        "exactly**. The fields are prose beginning 'FALSE'/'TRUE', not "
        "booleans, so the derivation reads the prefix; a boolean test "
        "returns zero and would have looked like a refutation"
    ),
    "the_two_new_instances_are_NOT_rows": (
        "**``d2-vs-probe`` and ``d2-vs-d1`` both have condition 1 FALSE "
        "and condition 2 TRUE.** They are DESCRIPTIVE contrasts and "
        "**register no ledger rows**, so the ledger count stays at "
        "EIGHT and the instance count becomes TEN"
    ),
    "the_distinction_entry_38_established": (
        "**rows aggregate contrasts, so a row count and an instance "
        "count are different quantities and both are correct about "
        "different things.** Entry 37 corrected two precedence claims by "
        "deriving from the fields rather than trusting the prose; this "
        "recount does the same and keeps the two numbers apart"
    ),
    "ledger_rows": 8,
    "instances_including_descriptive": 10,
    "the_phenomenon_itself": (
        "a delta that clears the combined seed uncertainty while the "
        "per-seed intervals disagree -- **the shape this cohort produces "
        "most often**, and the reason PLAN 4.3 requires both conditions "
        "rather than either"
    ),
}


#: **[SWEPT 2026-09-02] The withdrawal propagated -- nothing survives it
#: as current.**
THE_WITHDRAWAL_PROPAGATION_SWEPT = {
    "swept": "2026-09-02",
    "what_was_searched": (
        "the seven fabricated figures (0.3176, 0.3080, 0.3210, 0.3223, "
        "0.3168, 0.3200, 0.0055), the derived 0.2497, and the "
        "**D1-above-D2-not** pattern, across ``src/``, ``tests/``, "
        "``configs/``, ``scripts/`` and ``docs/``"
    ),
    "the_finding": (
        "**four records carry a fabricated figure and NOT ONE carries it "
        "as current.** ``THE_FABRICATED_FIGURES_WITHDRAWN`` names all "
        "eight as withdrawn; ``THE_TRANSCRIPTION_SLIP`` cites 0.3176 as "
        "the example of the shape that made it dangerous; "
        "``THE_UNEXPECTED_CELL`` is WITHDRAWN-marked and holds the "
        "pattern name in its preserved body; "
        "``WHAT_THE_ARMS_ACTUALLY_SHOW`` names 0.0055 to contrast it "
        "with the measured 0.0469. **No config, no script, no task and "
        "no doc carries any of them**"
    ),
    "the_run_confirms_it_from_the_other_side": (
        "**the run's own output cites "
        "``THE_DELTA_IS_INSIDE_THE_UNRESOLVED_BAND`` as WITHDRAWN FOR "
        "ITS SUBJECT while the band argument stands** -- which is the "
        "correct handling and is the withdrawal reaching the artifact, "
        "not just the record"
    ),
    "why_a_sweep_and_not_a_grep": (
        "a grep says where the digits are; **it does not say whether the "
        "record around them asserts or withdraws them.** The sweep "
        "classifies each occurrence by its record's status, which is the "
        "question -- and it is the check a fabrication that had been "
        "half-withdrawn would fail"
    ),
}


#: **[CLOSED 2026-09-02] PHASE 25.**
#:
#: **The registered null, measured.** Two self-supervised checkpoints,
#: three contrasts, all unresolved; the feature-source axis closes by
#: measurement. And an operational history worth as much as the result.
PHASE_25_CLOSING = {
    "closed": "2026-09-02, run p25_contrasts__c41d40a6__p25-contrasts",
    "the_finding": (
        "**NEITHER SELF-SUPERVISED CHECKPOINT BEATS THE PROBE, AND THE "
        "COHORT CANNOT ORDER ANY OF THE THREE CONTRASTS.** D1 0.2393, D2 "
        "0.1326, probe 0.2520; all three contrasts UNRESOLVED with "
        "condition 1 failing on each. **The registered NEITHER cell "
        "fired, which the phase predicted before the arms ran**"
    ),
    "and_what_it_did_not_do": (
        "**it moved no figure and closed no question the cohort cannot "
        "close.** Not one contrast is claimable in either direction; the "
        "phase converts an untested assumption into a measured null and "
        "nothing more"
    ),
    "criterion_walk": {
        "1_the_reckoning_states_the_counter_evidence_first": (
            "MET. The eleven and the one-factor patch-family contrast "
            "are recorded before the phase's own case, and the "
            "amendment's tempering is quoted unsoftened"
        ),
        "2_one_factor_per_arm_verified_in_code": (
            "MET, and verified rather than asserted: the generator "
            "copies the probe's recipe field by field at generation "
            "time, and a test compares every copied field against the "
            "probe's own config"
        ),
        "2_CORRECTED_2026_09_06_the_check_is_weaker_than_the_claim": (
            "**the sentence above is preserved and it claims more than "
            "the check performs.** The test compares every COPIED field, "
            "not every field, so a field absent from the copied list is "
            "invisible to it rather than caught by it. Two recipe fields "
            "were absent, weight decay and batch size, and the generator "
            "list carries nine of the reference recipe's fourteen"
        ),
        "2_CORRECTED_2026_09_06_what_the_divergence_actually_IS": (
            "**batch size 16 against the probe's 32.** Weight decay is "
            "unaffected because the schema default and the probe's value "
            "are both 0.01. Batch size is not, because the schema default "
            "is 16 and the probe declares 32, so the two arms resolve to "
            "a value the probe did not run at"
        ),
        "2_CORRECTED_2026_09_06_and_it_is_INERT_on_this_path": (
            "**the one factor claim stands, and it stands on a "
            "measurement rather than on the enumeration.** Both arms and "
            "the probe run ``trainable: head`` over a declared embeddings "
            "artifact. On that path ``phase3.make_factory`` returns "
            "``EmbeddingHeadBackbone``, whose fields are learning rate, "
            "weight decay, max steps and seed. **Batch size is never "
            "passed to it**, and the class records that the head trains "
            "by full batch steps. Batch size reaches only "
            "``TorchBackbone``, the branch taken when trainable is "
            "anything other than head, which no Phase 25 arm takes. So no "
            "banked figure depends on it and none changes"
        ),
        "2_CORRECTED_2026_09_06_the_general_shape": (
            "**a completeness claim was verified by a fidelity check.** "
            "Iterating the copied list can only ever confirm that copying "
            "worked. The record has met this shape before, in the Phase 7C "
            "key that reached the run record and never the execution. The "
            "durable form is a comparison over the UNION of both task "
            "blocks, so absence from the list is a failure rather than an "
            "invisibility, and it is now a test"
        ),
        "3_the_attribution_bound_travels_with_every_quotation": (
            "MET, and **vacuous in outcome**: no arm moved, so no gain "
            "needed attributing. The bound stands for the record rather "
            "than having been exercised"
        ),
        "4_the_full_criterion_on_all_three": (
            "MET. All three judged under PLAN 4.3 with BOTH conditions "
            "recorded explicitly; three unresolved results reported as "
            "unresolved with their point estimates, never as 'no "
            "difference'"
        ),
        "5_the_518_native_hazard_is_closed_before_launch": (
            "MET. ``timm_kwargs_for`` routes the 518-native case before "
            "the 224 shortcut and the suite pins it; the run did not die "
            "at the patch-embed assertion"
        ),
        "6_the_interpolation_is_declared_not_discovered": (
            "MET. The 37x37 -> 16x16 grid interpolation is in D2's "
            "registration and its config header. **D1 has none, and that "
            "asymmetry is part of what D1 controls for** -- which "
            "matters here, because D1 is the arm that came closest"
        ),
        "7_the_embedding_width_is_verified_at_extraction": (
            "MET. Both artifacts are 768-d, so the head is 769 "
            "parameters for both arms and the capacity axis is not "
            "confounded with the backbone"
        ),
        "8_the_family_is_three_and_no_arm_is_added_after_numbers_exist": (
            "MET. Three contrasts, derived from "
            "``phase25.contrast_family()`` rather than declared in the "
            "config, and **nothing was added once the numbers existed** "
            "-- including the DINO-B/8 arm the withdrawn hypothesis "
            "would have motivated"
        ),
        "9_a_ledger_row_only_where_claimable": (
            "MET, and vacuous: no contrast is claimable, so no row. The "
            "ledger stays at 38"
        ),
    },

    # ---- the operational history -----------------------------------
    "FOUR_DEFECTS_EACH_FOUND_AT_LAUNCH_ONE_LAYER_DEEPER": (
        "**the phase's real cost, and it is not the compute.** (1) an "
        "init outside ``embeddings.INITS``, at ``expected_variant``; (2) "
        "a checkpoint declaration the new inits cannot make, at "
        "``check_pairing``; (3) a task registered with no config that "
        "reaches it, at ``load_config``; (4) a shipped loader called "
        "with an arm dict missing the key it opens files with, at the "
        "first file read. **Each fix was correct. None of them looked "
        "further in**, and each cost a cluster round trip"
    ),
    "what_they_have_in_common": (
        "**every one is a reused component called with something it does "
        "not accept**, and every one was invisible to the checks that "
        "existed: the schema validated, the generator agreed with "
        "itself, and the suite was green. **A laptop cannot read the "
        "cluster, so the only thing that could have caught them is "
        "running the task against fixtures** -- which Phase 22 built for "
        "exactly this reason and Phase 25 did not have until the fourth"
    ),
    "THE_FABRICATION_IS_A_DIFFERENT_FAILURE_AND_IS_NAMED_SEPARATELY": (
        "**it does not belong in that tally.** The four are argument-"
        "shape defects, found by the code refusing. The fabrication was "
        "**figures with no source at all**, banked into three records "
        "and one shipped log line, and **no check could have caught it** "
        "-- it matched nothing, contradicted nothing, and was found only "
        "by reading the artifacts. THE_FABRICATED_FIGURES_WITHDRAWN, "
        "THE_TRANSCRIPTION_SLIP"
    ),
    "the_fixture_test_what_it_COVERS": (
        "``tests/test_p25_contrasts_end_to_end`` drives "
        "``task_p25_contrasts`` through the REAL RunContext, TASKS "
        "entry, ``_p21_load_arm_predictions`` and "
        "``phase7b.paired_comparison`` against three fixture run "
        "directories: **argument shapes, missing dict keys, required "
        "columns, the family's completeness, and both conditions on "
        "REAL OUTPUT rather than in source.** It reproduces the "
        "``KeyError: 'csv'`` pre-fix and would have caught all three "
        "earlier defects, which die on the same path"
    ),
    "the_fixture_test_what_it_does_NOT_cover": (
        "**the numbers.** Its arms are synthetic noise, so it tests that "
        "the task RUNS and that its output has the right shape -- never "
        "that a verdict is right. **It cannot see a wrong artifact, a "
        "crossed hash, or a fabricated figure**, and it cannot run the "
        "extraction or arm tasks, which need a GPU and clinical data. "
        "The defects it catches are the ones that raise; the ones that "
        "produce plausible wrong numbers are still caught only by "
        "reading the artifacts"
    ),
    "no_ledger_row": (
        "**this phase measured and did not claim.** Three unresolved "
        "contrasts earn no row; the ledger is unchanged at 38 entries "
        "and validate() is clean"
    ),
    "tag": "[CLOSED] -- the registered null, measured",
}


def summary() -> dict:
    """The phase's records, importable as one object."""
    return {
        "reckoning": PHASE_25_RECKONING,
        "counter_evidence": THE_ELEVEN_AND_THE_SHARPEST,
        "attribution_problem": THE_ATTRIBUTION_PROBLEM,
        "arm": THE_ARM_REGISTERED,
        "readings": READINGS_COMMITTED,
        "does_not_test": WHAT_THIS_DOES_NOT_TEST,
        "exit_criteria_draft": EXIT_CRITERIA_DRAFT,
        "settings_to_declare": SETTINGS_TO_DECLARE,
        "two_arms_ruled": THE_TWO_ARMS_RULED,
        "checkpoints_resolved": THE_CHECKPOINTS_RESOLVED,
        "factory_pretrained": THE_WEIGHTS_ARE_FACTORY_PRETRAINED_LIKE_THE_PROBES,
        "family": THE_FAMILY_OF_THREE,
        "limits_as_literals": LIMITS_AS_LITERALS,
        "exit_criteria": EXIT_CRITERIA,
        "settings_ruled": THE_SETTINGS_RULED,
        "init_vocabulary_defect": THE_INIT_WAS_OUTSIDE_THE_VOCABULARY,
        "checkpoint_declaration": WHAT_THE_TWO_INITS_CAN_HONESTLY_DECLARE,
        "delta_inside_the_band": THE_DELTA_IS_INSIDE_THE_UNRESOLVED_BAND,
        "the_unexpected_cell": THE_UNEXPECTED_CELL,
        "fabricated_figures_withdrawn": THE_FABRICATED_FIGURES_WITHDRAWN,
        "what_the_arms_actually_show": WHAT_THE_ARMS_ACTUALLY_SHOW,
        "transcription_slip": THE_TRANSCRIPTION_SLIP,
        "exemption_is_noisy": THE_EXEMPTION_IS_NOISY_NOW,
        "contrasts_config_emitted": THE_CONTRASTS_CONFIG_EMITTED,
        "arm_dict_key_defect": THE_ARM_DICT_LACKED_THE_LOADERS_KEY,
        "contrast_launch_void": CONTRAST_LAUNCH_VOID,
        "three_verdicts": THE_THREE_VERDICTS_OBSERVED,
        "neither_cell_fired": THE_NEITHER_CELL_FIRED,
        "axis_closes": THE_AXIS_CLOSES_DERIVED,
        "seed_instability": THE_SEED_INSTABILITY_OBSERVED,
        "condition_split_recounted": THE_CONDITION_SPLIT_RECOUNTED,
        "withdrawal_swept": THE_WITHDRAWAL_PROPAGATION_SWEPT,
        "closing": PHASE_25_CLOSING,
        "contrasts": CONTRASTS,
    }


#: **[AMENDED 2026-09-05] THE EIGHTH SEQUENCE AMENDMENT. Phases 26, 27
#: and 28 are SCHEDULED, NOT REGISTERED.**
#:
#: Four parked ideas were ruled to run. Three become scheduled phases and
#: the fourth runs inside one of them rather than separately.
PHASE_SEQUENCE_EXTENDED_8 = {
    "decided": (
        "2026-09-05, the EIGHTH sequence amendment, and the third that "
        "renumbers nothing"
    ),
    "why_it_lives_in_phase25": (
        "the established pattern puts an amendment in the module current "
        "when it was decided. The first went to phase11, the second to "
        "phase12, the third and fourth to phase15, the fifth to phase20, "
        "the sixth and seventh to phase21. Phase 25 is the phase current "
        "on this date, so this is where the eighth belongs. A closed "
        "module is not a reason to move it, because the amendment records "
        "when the decision was taken rather than what it decides about"
    ),

    # ---- the full backward chain, with dates ------------------------
    "first_amendment": (
        "phase11.PHASE_SEQUENCE_RENUMBERED, 2026-08-23. View ablation to "
        "12, decoder to 13, write-up to 14"
    ),
    "second_amendment": (
        "phase12.PHASE_SEQUENCE_RENUMBERED_2, 2026-08-23. LDL to 14, "
        "second beauty dataset to 15, TSTR to 16, write-up to 17"
    ),
    "third_amendment": (
        "phase15.PHASE_SEQUENCE_RENUMBERED_3, 2026-08-24. Metric-space "
        "ablation to 16, TSTR to 17, write-up to 18"
    ),
    "fourth_amendment": (
        "phase15.PHASE_SEQUENCE_RENUMBERED_4, 2026-08-24. Anchor loop to "
        "16, TSTR unchanged at 17, metric-space ablation to 18, write-up "
        "to 19"
    ),
    "fifth_amendment": (
        "phase20.PHASE_SEQUENCE_EXTENDED_5, 2026-08-31. Permutation "
        "control appended as 20, renumbering nothing, and the first to "
        "adopt the rule that the write-up runs LAST BY RULE rather than "
        "at the highest number"
    ),
    "sixth_amendment": (
        "phase21.PHASE_SEQUENCE_EXTENDED_6, 2026-08-31. Ensemble probe "
        "and error-consistency diagnosis appended as 21, renumbering "
        "nothing, and the first amendment written UNDER the "
        "write-up-runs-last rule rather than establishing it"
    ),
    "seventh_amendment": (
        "phase21.PHASE_SEQUENCE_EXTENDED_7, 2026-09-01. Phases 22 through "
        "25 scheduled SCHEDULED-NOT-REGISTERED, renumbering nothing. The "
        "three standing clauses this amendment inherits were written "
        "there"
    ),

    "was": {
        "25": "foundation-model features, closed 2026-09-02",
        "26": "NOTHING. The chain stopped at 25",
    },
    "becomes": {
        "26": "THE CALIBRATION ABLATION",
        "27": "THE ANCHOR SET AS A TRAINING SET",
        "28": "FINE TUNING, with small backbones as an arm inside it",
    },
    "status_changes": {
        "calibration_ablation": "parked idea to PHASE 26",
        "anchor_set_as_training_set": "parked idea to PHASE 27",
        "fine_tuning": "parked idea to PHASE 28",
        "small_backbones": (
            "parked idea to AN ARM INSIDE PHASE 28, not a phase of its own"
        ),
        "write_up": "Phase 19 unchanged, and last by rule",
    },

    # ---- 26 ---------------------------------------------------------
    "phase_26_the_calibration_ablation": {
        "status": "SCHEDULED, NOT REGISTERED",
        "what": (
            "sweep weight decay and epoch budget on the probe's recipe, "
            "reporting PCC, shrinkage, RMSE, MAE, three-class accuracy "
            "and IEM together rather than one at a time"
        ),
        "motivated_by": (
            "PCC is scale invariant and cannot see squashing. A clinical "
            "tool outputs a value rather than a rank, so a prediction "
            "that ranks patients correctly and compresses them toward "
            "the label mean is a worse tool at the same correlation. IEM "
            "is where calibration shows, because it measures the value "
            "and not the order"
        ),
        "the_only_one_with_a_supervision_request_behind_it": (
            "recorded because it is the distinguishing fact. The standing "
            "request is for IEM to be as small as possible, banked at "
            "phase18.py under the fifth design decision, and this phase "
            "answers it directly. The other two scheduled phases are "
            "motivated by the record alone"
        ),
        "cost": "runs on cached embeddings",
    },

    # ---- 27 ---------------------------------------------------------
    "phase_27_the_anchor_set_as_a_training_set": {
        "status": "SCHEDULED, NOT REGISTERED",
        "what": (
            "train on the 25 anchor grades and evaluate on the 237"
        ),
        "motivated_by_the_case_FOR": (
            "the anchor labels are cleaner than anything this project "
            "trains on. The cohort target is a five-rater mean whose "
            "panel agreement is Fleiss kappa 0.1662, banked at "
            "ladder.py beside QWK 0.4276. A consensus grade carries no "
            "such disagreement inside it"
        ),
        "motivated_by_the_case_AGAINST": (
            "25 rows against a 769-parameter head is an underdetermined "
            "fit, and the record predicts it will fail. The head size is "
            "banked at phase12.py under head_parameters, which records "
            "769 for the single-embedding arms"
        ),
        "the_constraint_the_restate_MUST_address": (
            "the label incomparability is already conceded and must not "
            "be inherited silently. "
            "phase9.EXTERNAL_VALIDATION_ON_THE_25_CONCEDED ground 2 "
            "records that the two targets are a different quantity on "
            "both scale and construction. The cohort target is a panel "
            "mean of five integer grades, so it is continuous on rater "
            "steps of 0.2, while the anchor Score is an integer from 1 "
            "to 5. A correlation against one is not a correlation "
            "against the other. Training on one and evaluating on the "
            "other crosses exactly that boundary, so the restate has to "
            "say what the crossing means before any number exists"
        ),
        "and_unanimity_is_REPORTED_not_banked": (
            "the same concession records that the record says the "
            "opposite twice. The anchor grades are trusted single grades "
            "from the survey lineage and are NOT verified unanimous, and "
            "the manuscript's own claim is the 25 highest-agreement "
            "images of 76 from a separate 27-rater study, which is a "
            "SELECTION CRITERION rather than unanimity. The scheduling "
            "carries the word as REPORTED and the restate may not "
            "upgrade it without a source"
        ),
        "cost": "runs on cached embeddings",
    },

    # ---- 28 ---------------------------------------------------------
    "phase_28_fine_tuning": {
        "status": "SCHEDULED, NOT REGISTERED",
        "what": (
            "fine tuning, with small backbones as an arm inside it "
            "rather than a phase of its own"
        ),
        "motivated_by": (
            "the frozen-backbone constraint was a comparability "
            "decision and never a judgement about fine tuning, and fine "
            "tuning is the one method axis this record has never tested. "
            "phase15.py records the position directly, that if a "
            "genuinely distinct second sense was meant, fine tuning "
            "rather than a frozen head, then it HAS NOT BEEN RUN, and "
            "that this is a maintainer decision rather than the "
            "record's"
        ),
        "why_small_backbones_belong_HERE_and_not_alone": (
            "they are uninteresting frozen and interesting only as the "
            "tractable form of fine tuning at this cohort size. The "
            "feature-source axis is closed by measurement, so a smaller "
            "frozen backbone would be one more measurement on an axis "
            "that has stopped moving. Under fine tuning the question "
            "changes, because capacity and cohort size interact and a "
            "smaller model is the one that can be adapted at all here"
        ),
        "binding_the_literature_is_banked_BEFORE_the_restate": (
            "this phase may not be restated until its literature is in "
            "the record. The sources found are not in the notebook, and "
            "the two central questions, whether fine tuning helps at a "
            "cohort of this size and whether higher resolution requires "
            "adaptation, are questions the field has answered many "
            "times. Registering a phase whose answer may already be "
            "published, without reading what was published, is how a "
            "measurement becomes a rediscovery"
        ),
        "cost": (
            "not cached. This is the only one of the three that needs "
            "training rather than a head refit"
        ),
    },

    # ---- the standing clauses, inherited from the seventh -----------
    "binding_1_scope_at_the_restate": (
        "each phase's scope, exit criteria and readings are written at "
        "ITS OWN RESTATE and never here. Motivation is not scope. This "
        "amendment records why each phase is worth running and nothing "
        "about what would count as an answer"
    ),
    "binding_2_cancellation_is_recorded": (
        "a scheduled phase that does not run is recorded as CANCELLED "
        "WITH A REASON and never silently dropped. A schedule that "
        "quietly loses entries is indistinguishable from one that never "
        "held them"
    ),
    "binding_3_reorder_on_evidence": (
        "any of the three may be REORDERED OR CANCELLED on evidence from "
        "an earlier one. Phase 26's calibration result may change "
        "whether 27 is worth running at all, which is exactly why no "
        "readings are written here"
    ),

    # ---- the ordering ruled -----------------------------------------
    "the_ordering_ruled": (
        "26 first, then 27, then 28. Phase 26 runs on cached embeddings "
        "in minutes and its result is wanted for a supervision meeting "
        "on 2026-09-07. Phase 27 also runs on cached embeddings. Phase "
        "28 needs training and is not cheap"
    ),
    "what_fits_before_the_meeting": (
        "26 and 27 are cheap enough to complete before 2026-09-07. 28 is "
        "NOT, and the schedule says so rather than leaving it to be "
        "discovered on the day"
    ),

    "scheduled_not_registered": (
        "the distinction the record already draws, and it is one step "
        "earlier than either older precedent. "
        "phase15.PHASE_16_SCHEDULED is SCHEDULED-NOT-SCOPED and "
        "phase15.ANCHOR_LOOP_REGISTERED is REGISTERED-NOT-BUILT. These "
        "three are scheduled only. Nothing is built and nothing is "
        "registered by this amendment"
    ),
    "nothing_silently_renumbered": (
        "nothing moved. Phases 19 through 25 are untouched, the write-up "
        "stays at 19, and it runs last by rule rather than by number"
    ),
    "module_names_never_change": (
        "unchanged. A module file is named for the number its phase "
        "HOLDS, and a record key keeps the number it was WRITTEN with. "
        "Nothing is renamed because nothing moved"
    ),

    # ---- two figures the brief carried that the record does not -----
    "TWO_FIGURES_NOT_TAKEN_FROM_THE_BRIEF": (
        "the scheduling brief said fifteen banked measurements show the "
        "feature source barely moves this task. The record does not "
        "carry fifteen. phase25.THE_AXIS_CLOSES_DERIVED derives TWELVE "
        "as the strict feature-source count and THIRTEEN under the "
        "record's own convention, which also counts one measurement on "
        "the adjacent training-objective axis, and banks both. Fifteen "
        "is the size of Phase 22's CONTRAST family on that adjacent "
        "axis, which is a different quantity under a similar name. The "
        "count is stated here as THIRTEEN, the registered convention, "
        "and the strict TWELVE beside it. The second figure is "
        "unanimity, handled at "
        "phase_27_the_anchor_set_as_a_training_set"
    ),
    "tag": "[SCHEDULED] -- three phases, nothing built, nothing registered",
}
