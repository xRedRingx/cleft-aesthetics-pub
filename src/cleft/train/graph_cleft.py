"""Cleft fine-tuning for the graph backbones: the THIRD REGIME.

Frozen feature maps in, trained graph layers out (brief §2.3): the backbone's
contribution arrives precomputed as a ``feature_map`` embedding artifact, and
everything after the map -- the message passing and the attention, which ARE
the contribution -- trains on the 237. Freezing those would reduce the arm to
a linear probe and test nothing; fine-tuning the backbone on 237 images
measured PCC -0.013. This regime is the one whose seed band Phase 3's does
not cover, and the band is a gate before any SR-GNN or AG-Net delta.

**Everything runs through the FROZEN harness.** ``run_cv`` slices feature
rows agnostically, so the maps (with the scheme's boxes packed alongside)
ride the same 5-fold loop, the same inner-val split, the same gate-3 check
and the same early stopping as every other cleft arm -- early stopping STAYS
here, where collapse is the documented behaviour (the fixed-budget decision
was pretraining's, and its gate-4 amendment says exactly this).

----------------------------------------------------------------------------
PROVENANCE IS PAIRED, NOT ASSEMBLED
----------------------------------------------------------------------------
* The artifact's ``checkpoint_sha256`` must equal the declared checkpoint
  input's verified hash: the maps and the warm-started graph layers must come
  from the SAME pretraining run, or the arm trains layers against features
  from a different representation and nothing would raise.
* The artifact's ``pretrain_scheme`` must equal the arm's ``region_scheme``
  for pretrained inits -- the consistency rule, checked mechanically. An
  imagenet init has no pretraining scheme, so any scheme is legitimate there
  and the freedom is recorded rather than implied.

----------------------------------------------------------------------------
WARM START, AND THE ONE PIECE THAT IS RE-INITIALISED
----------------------------------------------------------------------------
For pretrained inits the graph layers start from the checkpoint's weights --
that is what "init" MEANS for this regime. The classifier alone is
re-initialised (zero weights, bias at the training-fold mean): the SCUT head
maps to SCUT ratings, and gate 3 requires an untrained head to predict the
cleft training mean -- which zero-init satisfies by construction, BN running
statistics included, because a zero-weight head is constant whatever arrives.
For ``imagenet`` init the graph layers are seed-initialised: ImageNet covers
only the backbone, and there is nothing else they could start from.

----------------------------------------------------------------------------
BOXES TRAVEL WITH THE FEATURES
----------------------------------------------------------------------------
The harness has no side channel, so each patient's row is the flattened map
with the scheme's boxes appended (``pack``/``unpack``, exact float32
round-trip). Generated schemes place Phase 2's patches through each patient's
own recorded content box -- the same machinery pretraining used. AG-Net's
native scheme is SIFT+GMM over the staged images, computed once per patient
(seeded, deterministic) and padded to a rectangle with whole-image rows --
the same padding ``pool_regions_spatial`` itself performs. SR-GNN's native
scheme is its own grid buffer (boxes carry zero width in the packed row).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import embeddings
from ..models.factory import BACKBONES
from . import checkpoint as ckpt
from . import determinism, gates
from .harness import TrainConfig, run_cv
from .phase3 import (
    Phase3Result,
    determinism_fingerprint,
    sanity_report,
    seed_variance,  # noqa: F401  (re-exported for the handler)
)
from .pretrain import (
    GENERATED_SCHEMES,
    MODEL_PREFIX,
    boxes_for_model,
    scheme_frame_boxes,
)

#: The regime's own backbones. "stub_graph" runs the whole task on the laptop
#: through the harness stub, restricted to the native scheme (no torch, no
#: box conventions).
GRAPH_CLEFT_BACKBONES = ("srgnn", "agnet", "stub_graph")

CLEFT_SCHEMES = ("native", "grid", "anatomy", "random")

#: What the arm optimises. ``graph_layers`` is THE REGIME (brief §2.3): the
#: message passing and the attention are the contribution, and freezing them
#: would reduce the arm to a linear probe and test nothing.
#:
#: ``classifier`` is that sentence run deliberately as a DIAGNOSTIC. It is not
#: a ladder arm and must never be reported as one -- it exists to make the
#: reduction the brief warns about, on purpose, once, because the reduced arm
#: is the only thing directly comparable to the frozen ViT probe's 0.2529.
#: See ``THIRD_REGIME_MEAN_IS_OPEN`` and ``FROZEN_GRAPH_DIAGNOSTIC``.
#:
#: ``classifier_adabn`` is the same probe with the ONE alternative explanation
#: its low result left standing removed: BatchNorm running statistics
#: re-estimated on each training fold before the classifier is fitted. Also a
#: diagnostic, also never a ladder arm. See ``ADABN_DIAGNOSTIC``.
TRAINABLE_POLICIES = ("graph_layers", "classifier", "classifier_adabn")

#: The two policies that reduce the arm to a probe. Grouped because every
#: place that asks "is this the diagnostic" must treat them alike -- eval
#: mode, the regime label, the not-a-ladder-arm stamp -- and a membership test
#: written as ``== "classifier"`` would have silently excluded the AdaBN arm
#: from all three. That is the exact shape of the 2026-07-28 defect (a check
#: keyed to one literal), so the set exists rather than the comparison.
PROBE_POLICIES = ("classifier", "classifier_adabn")

#: **[MEASURED 2026-08-01, run p6-srgnn-seedband-1 -- figures pasted by the
#: maintainer] The third regime's band. The gate is met; this is the number
#: every graph delta divides by.**
#:
#: Ten seeds, one job, one ``env.json``: SR-GNN at G2 on the masked init,
#: native scheme, frozen feature maps with TRAINED graph layers, mean label,
#: 5-fold CV over the Phase 1 folds.
#:
#: **What the seed varies here, and it is NOT what Phase 3's varied.** The
#: graph layers are warm-started from the paired checkpoint and are therefore
#: **byte-identical across all ten seeds** -- the one thing a reader would
#: assume a graph-regime band covers is the one thing it does not. What the
#: seed drives is the classifier re-initialisation, the batch order
#: (``GraphHeadBackbone.train_epoch`` permutes off ``seed``, unlike the Phase
#: 3 backbone), and ``harness.inner_val_split``. Say all three when quoting
#: it: PLAN §4.12.1 exists because two arms can both report "seed SD" and mean
#: different things, and this is the third such quantity in the project.
#:
#: **Where it lands.** SD 0.025053 is **1.83x** the frozen linear probe's
#: 0.0137 and BELOW Phase 4's patch arms at 0.045-0.048 -- so trainable graph
#: parameters cost less seed stability than the patch path did, which was not
#: the way the brief's §5 worry pointed. Five seeds resolve 0.031, ten resolve
#: 0.022. The one-seed figure 0.0694 is again essentially the whole observed
#: range (0.0706), exactly as at gate 2 -- that relation is a property of the
#: formula meeting ten draws, not a coincidence worth reading into.
MEASURED_GRAPH_SEED_BAND = {
    "n_seeds": 10,
    "mean": 0.15067,
    "sd": 0.025053,
    "min": 0.102447,
    "max": 0.173075,
    "range": 0.070628,
    "claimable_delta_at_n_seeds": {"1": 0.069442, "5": 0.031056, "10": 0.021960},
    "arm": (
        "srgnn, scut_masked init at g2, native scheme, frozen feature maps + "
        "TRAINED graph layers, label mean, 5-fold CV"
    ),
    "measures": "classifier_init_and_batch_order_and_inner_val_split",
    "does_not_measure": (
        "graph-layer initialisation -- warm-started from the paired checkpoint, "
        "identical across all ten seeds"
    ),
    "against_frozen_probe_sd": 0.0137,
    "against_patch_arm_sd": "0.045-0.048",
    "run": "p6-srgnn-seedband-1",
    "measured": "2026-08-01",
}

#: **[MEASURED 2026-08-01, same run] The band was met and the MEAN is the
#: problem -- and two readings produce identical evidence, so this records the
#: open question rather than a verdict.**
#:
#: Pooled OOF PCC 0.1507 against the frozen ViT-B/16 linear probe's 0.2529: a
#: **-0.102 delta**, which clears the combined threshold comfortably in either
#: reading. SR-GNN with trained graph layers, warm-started from beauty
#: pretraining, loses to an off-the-shelf ImageNet embedding with a linear
#: head.
#:
#: **The signature, and why it is not self-interpreting.** ``selected_epochs``
#: are 1-4 with one seed selecting epoch 1 in all five folds; ``shrinkage``
#: 0.14-0.18, so predictions compress to a sixth of the label spread;
#: ``qwk_3cat`` 0.0 at ``n_distinct_3class_bins: 1``. That is the SAME
#: signature as the ViT full fine-tune that scored -0.013 (PLAN §4.7) --
#: 11,681,859 trainable parameters on ~152 training samples, stopping almost
#: immediately.
#:
#: Two readings fit it exactly:
#:
#: * **architecture** -- SR-GNN's representation is genuinely weaker for this
#:   task at n=237, and the trained layers are not the cause;
#: * **configuration** -- the arm is over-parameterised the way ViT's full
#:   fine-tune was, and the features are fine.
#:
#: **They are separated by ONE measurement, not by argument**: the same arm
#: with the graph layers frozen too (``trainable="classifier"``), which is an
#: SR-GNN linear probe on the same folds and therefore directly comparable to
#: the 0.2529. ~0.25 indicts the configuration; ~0.15 indicts the
#: representation. Either answer is a real finding, and neither is a licence
#: to tune: this is one hypothesis test, and a search over learning rates
#: dressed as a diagnostic would answer a question nobody asked.
THIRD_REGIME_MEAN_IS_OPEN = {
    "measured": "2026-08-01",
    "graph_regime_mean": 0.15067,
    "frozen_vit_probe_mean": 0.2529,
    "delta": -0.10223,
    "signature": {
        "selected_epochs": "1-4, one seed selecting epoch 1 in all five folds",
        "shrinkage": "0.14-0.18",
        "qwk_3cat": 0.0,
        "n_distinct_3class_bins": 1,
        "trainable_parameters": 11_681_859,
        "train_samples_per_fold": 152,
        "matches": "the ViT full fine-tune at PCC -0.013 (PLAN §4.7)",
    },
    "readings": {
        "architecture": "SR-GNN's representation is genuinely weaker at n=237",
        "configuration": "over-parameterised as ViT's full fine-tune was",
    },
    "separated_by": (
        "the same arm at trainable='classifier' -- graph layers frozen too, an "
        "SR-GNN linear probe on the same folds, comparable to 0.2529 directly"
    ),
    # --- ANSWERED IN PART, 2026-08-01, run p6-srgnn-frozen-graph ----------
    #
    # The status below said OPEN and named what would flip it. That ran, and
    # this is the rewrite it asked for -- the placeholder closing on schedule
    # rather than going stale.
    # --- CLOSED 2026-08-01 by the completed 2x2 (NORMALISATION_2X2) --------
    #
    # The status below tracked this question through four rewrites, each one
    # the flip its predecessor asked for. This is the last: the configuration
    # reading died at the frozen probe, and the normalisation alternative has
    # now been tested in all four cells with every reachable correction worse
    # than doing nothing.
    "status": (
        "CLOSED. Configuration reading DEAD (frozen probe at 2,049 trainable "
        "scored 0.1340). Normalisation alternative REFUTED in all four cells "
        "-- A 0.1340 > C 0.1093 > B 0.0468 > D 0.0262, monotonically worse "
        "with more cleft adaptation. The ARCHITECTURE READING STANDS."
    ),
    "closed_by": "NORMALISATION_2X2",
    "configuration_reading_closed_by": {
        "measured": "2026-08-01",
        "run": "p6-srgnn-frozen-graph",
        "pcc": 0.1340,
        "trainable_parameters": 2049,
        "selected_epochs": [1, 2, 3, 2, 4],
        "shrinkage": 0.2943,
        "beats_constant": False,
        "argument": (
            "over-parameterisation cannot be the cause when the arm has no "
            "capacity to overfit with. The same probe setup that gives ViT "
            "0.2529 gives SR-GNN 0.1340 at 2,049 trainable parameters, so the "
            "0.15 is not a consequence of 11.7M parameters on 152 samples"
        ),
        # And it reframes the arm it was diagnosing, in the direction opposite
        # to the one the signature suggested.
        "reframes_the_trained_arm": (
            "0.15067 with 11,681,859 trainable against 0.1340 with 2,049: "
            "training the graph layers HELPS by ~0.017. That is inside the "
            "0.031 five-seed threshold so it is NOT claimable, but the "
            "direction is opposite to overfitting -- the signature that "
            "matched the ViT full fine-tune was not the mechanism it looked "
            "like"
        ),
    },
    # The one alternative the frozen arm could not exclude -- named in advance
    # in FROZEN_GRAPH_DIAGNOSTIC.outcome_low, and now load-bearing rather than
    # hypothetical.
    "surviving_alternative": (
        "normalisation-statistics mismatch: the probe runs on the "
        "checkpoint's SCUT BatchNorm statistics against cleft inputs. AdaBN "
        "is the documented remedy (PLAN §4.7) -- and it is the SAME mechanism "
        "that was the root cause of the void ladder's head-init defect, so "
        "this is a failure mode with a record in this project, not a "
        "speculative one"
    ),
    #: **How the surviving alternative was closed** -- and the field that
    #: used to say ``flips_when`` now says what flipped it. Both halves of the
    #: normalisation chain were varied independently: the 2 reachable at cleft
    #: time (the adabn policy) and the 40 fixed at extraction (a per-fold
    #: re-extraction, five leak-checked artifacts). Four cells, and every
    #: correction is worse than none.
    "flipped_by": (
        "configs/p6_srgnn_frozen_graph_adabn.yaml (cell B), "
        "configs/p6_extract_srgnn_adabn.yaml + "
        "configs/p6_srgnn_probe_adabn_artifact{,_full}.yaml (cells C and D). "
        "See NORMALISATION_2X2"
    ),
    #: The scoping correction that made the re-extraction necessary, kept
    #: because it is why the question could be closed at all: the alternative
    #: as first written spanned the whole chain, and the artifact boundary
    #: splits it 2/40. A cleft-only arm could never have settled it, and the
    #: first draft of this record claimed it could.
    "why_a_re_extraction_was_required": (
        "the alternative spanned the whole normalisation chain, but only 2 of "
        "42 BatchNorms are reachable at cleft time -- the other 40 ran during "
        "EXTRACTION and are baked into the artifact. Testing it needed a new "
        "artifact, not a new arm"
    ),
    # **Neither 0.1340 nor 0.15067's relation to it is quotable as a value.**
    # One seed, against a regime whose measured SD is 0.025 -- so 0.1340 could
    # sit anywhere in roughly 0.11-0.16. The conclusion survives that interval
    # (its top falls well short of 0.25), which is why the reading closes; the
    # NUMBER does not, and the same applies to whatever the AdaBN arm returns.
    "one_seed_caveat": {
        "seeds": 1,
        "band_sd_used": 0.025053,
        "plausible_interval": [0.11, 0.16],
        "conclusion_survives_it": True,
        "value_is_quotable": False,
        "why": (
            "even the top of the interval falls well short of 0.2529, so the "
            "configuration reading dies across the whole of it -- but no point "
            "value and no delta may be quoted from one seed"
        ),
    },
}

#: **[DECIDED 2026-08-01] The diagnostic that separates the two readings, and
#: what each outcome does and does not license -- written BEFORE it runs.**
#:
#: One SR-GNN arm at ``trainable="classifier"``: the backbone map frozen (as
#: always) AND the graph layers frozen, fitting only the 2,049-parameter
#: classifier. That is an SR-GNN linear probe on the same 237 patients, the
#: same folds, the same label and the same harness as the frozen ViT probe, so
#: its pooled OOF PCC is comparable to 0.2529 directly.
#:
#: **What "frozen" has to mean here, and it is not just requires_grad.**
#: SR-GNN carries ``Dropout(0.2)`` applied twice and two BatchNorms. Left in
#: train mode the pre-classifier stack would emit different activations every
#: batch and every epoch, and the arm would not be a probe at all -- it would
#: be a linear head on a stochastic representation, wearing a probe's name.
#: So this policy runs the model in EVAL mode: dropout off, BN reading its
#: running statistics rather than the batch's. The frozen stack becomes a
#: fixed function of the map, which is the property that makes the ViT
#: comparison mean anything (PLAN R2 -- a number's name is not evidence about
#: which question it answers).
#:
#: **The rate is the probe convention, and that is part of the hypothesis.**
#: 1e-3, this project's head rate and the one the 0.2529 comparator used, not
#: the band arm's 1e-4 (a full-fine-tune scale chosen for a 12M stack). So the
#: diagnostic varies parameter count AND rate together and cannot separate
#: them -- deliberately: both are "the configuration", which is the whole of
#: what is being tested against "the representation". It is NOT a rate sweep,
#: and turning it into one would answer a question nobody asked.
#:
#: **Read the outcomes asymmetrically. This is the part that must be written
#: down first.**
#:
#: * **~0.25 is conclusive for the configuration reading.** The frozen arm is
#:   the one running under WORSE-matched BatchNorm statistics (SCUT's, from
#:   the checkpoint, on cleft inputs) than the trained arm, which adapts them
#:   from cleft batches. A handicapped probe beating the trained arm can only
#:   strengthen the over-parameterisation reading.
#: * **~0.15 is CONSISTENT with the architecture reading but does not
#:   establish it**, because that same BN asymmetry is a live alternative: the
#:   representation would be computed with statistics from a different
#:   distribution, and AdaBN is the documented remedy for exactly this on
#:   these backbones (PLAN §4.7, [MEASURED + LITERATURE]). A low score makes
#:   BN re-estimation the next arm, not the conclusion.
#:
#: **One seed, and what one seed can carry.** The two hypotheses sit ~0.10
#: apart, and the closest available resolution figure -- the trained arm's
#: one-seed 0.0694, itself only a PROXY, since this arm's own band is
#: unmeasured and a 2,049-parameter probe should be tighter -- is comfortably
#: below that. So a single seed discriminates 0.25 from 0.15 and licenses
#: nothing finer: no delta against any other arm, and no quoted value for this
#: arm beyond which side of the gap it fell.
FROZEN_GRAPH_DIAGNOSTIC = {
    "decided": "2026-08-01",
    "config": "configs/p6_srgnn_frozen_graph.yaml",
    "is_a_ladder_arm": False,
    "purpose": (
        "separate 'the training configuration is wrong the way ViT's was' "
        "from 'the SR-GNN representation is genuinely weaker at n=237'"
    ),
    "trainable": "classifier",
    "trainable_parameters": 2049,
    "eval_mode_while_training": True,
    "why_eval_mode": (
        "Dropout(0.2) x2 and two BatchNorms: in train mode the pre-classifier "
        "stack is stochastic and batch-dependent, so the arm would not be a "
        "probe at all -- a linear head on a moving representation wearing a "
        "probe's name"
    ),
    #: **[MEASURED 2026-08-01, system python, torch 2.13.0 CPU] The eval-mode
    #: line is load-bearing, and requires_grad does not cover for it.** Over
    #: one epoch under this policy, exactly two tensors move --
    #: ``classifier.weight`` and ``classifier.bias``. With the model left in
    #: train mode instead, EIGHT move: those two plus ``bn1``/``bn2``
    #: ``running_mean``, ``running_var`` and ``num_batches_tracked``. Those are
    #: BUFFERS, not parameters, so no gradient flows through them and nothing
    #: about ``requires_grad_(False)`` stops train mode updating them on every
    #: batch. Freezing only the parameters would have left the "frozen"
    #: representation drifting under the head fitting on it, silently, with
    #: every shape and every metric still plausible.
    #:
    #: ``scripts/verify_backbone_builds.py`` compares parameters AND buffers
    #: for this reason, and the negative was run before the positive was
    #: trusted: a check that cannot fail is not a check.
    "eval_mode_verified": {
        "measured": "2026-08-01",
        "environment": "system python, torch 2.13.0+cu126, CPU",
        "moves_under_policy": ["classifier.bias", "classifier.weight"],
        "moves_without_eval_mode": 8,
        "extra_are": "bn1/bn2 running_mean, running_var, num_batches_tracked",
        "why_it_matters": (
            "BN running statistics are buffers; requires_grad does not protect "
            "them, so parameter-only freezing leaves the representation moving"
        ),
    },
    "learning_rate": 1e-3,
    "why_that_rate": (
        "the probe convention the 0.2529 comparator used, for a "
        "2,049-parameter head; the band arm's 1e-4 is a 12M-stack rate. Rate "
        "and parameter count vary together ON PURPOSE -- both are 'the "
        "configuration'"
    ),
    "compared_against": {"frozen_vit_probe": 0.2529, "trained_graph_arm": 0.15067},
    "outcome_high": (
        "~0.25 CONCLUSIVE for configuration: this arm runs under worse-matched "
        "BN statistics (SCUT's) than the trained arm, so beating it can only "
        "strengthen the reading"
    ),
    "outcome_low": (
        "~0.15 CONSISTENT with architecture but NOT established -- the same BN "
        "asymmetry is a live alternative, and AdaBN is the documented remedy "
        "(PLAN §4.7). A low score makes BN re-estimation the next arm"
    ),
    "seeds": 1,
    "what_one_seed_licenses": (
        "which side of a ~0.10 gap the arm fell, and nothing finer -- no delta "
        "against any arm, no quoted value. The 0.0694 one-seed figure is the "
        "TRAINED arm's and is only a proxy: this arm's own band is unmeasured"
    ),
    "not_a_search": (
        "one hypothesis test. Do not tune toward 0.2529 -- a configuration "
        "found by searching would answer 'can this arm be made to score 0.25', "
        "which is not the question and has no interesting negative"
    ),
    #: **[MEASURED 2026-08-01, run p6-srgnn-frozen-graph] It ran, and it
    #: landed on the outcome_low branch.** Recorded here beside the prediction
    #: rather than in a separate place, so the pairing is visible: the reading
    #: applied is the one written down before the number arrived.
    #:
    #: The freeze took on the cluster exactly as it did locally -- 2,049
    #: trainable, which is the check that says the policy did what it claims
    #: rather than merely being configured.
    "result": {
        "measured": "2026-08-01",
        "run": "p6-srgnn-frozen-graph",
        "pcc": 0.1340,
        "trainable_parameters": 2049,
        "selected_epochs": [1, 2, 3, 2, 4],
        "shrinkage": 0.2943,
        "beats_constant": False,
        "branch_taken": "outcome_low",
        "licensed": (
            "the configuration reading is dead -- an arm with 2,049 trainable "
            "parameters cannot be over-parameterised. The architecture reading "
            "is NOT thereby established: outcome_low said so in advance, and "
            "BN re-estimation is now the arm that tests the last alternative"
        ),
        "not_licensed": (
            "quoting 0.1340 as a value, or the +0.017 against the trained arm "
            "as a delta. One seed against SD 0.025"
        ),
    },
}

#: **[DECIDED 2026-08-01] The AdaBN probe: the last live alternative, and what
#: each outcome licenses -- written BEFORE it runs.**
#:
#: The frozen probe scored 0.1340 and killed the configuration reading, but it
#: ran on the checkpoint's **SCUT** BatchNorm running statistics against cleft
#: inputs. So a representation computed with statistics from the wrong
#: distribution is still a complete explanation of the gap, and it is not a
#: speculative one: AdaBN is the documented remedy for exactly this on these
#: backbones (PLAN §4.7, [MEASURED + LITERATURE]), and **the same mechanism
#: was the root cause of the void ladder's "head-init defect"**. This project
#: has been bitten by it once already.
#:
#: **The procedure, per fold.** Reset each BatchNorm's running statistics;
#: put the BN modules ALONE into train mode with ``momentum=None``; forward
#: the training fold under ``no_grad`` to accumulate; return to eval and fit
#: the classifier exactly as the frozen probe does. Everything else is
#: identical to ``p6_srgnn_frozen_graph``, so the pair differs in one factor.
#:
#: **Three things that would each make it a no-op reporting success**, all
#: measured on torch 2.13.0 before the code was written
#: (``MECHANISM_VERIFIED`` below):
#:
#: 1. **Not resetting first.** ``momentum=None`` is a CUMULATIVE average
#:    weighted ``1/num_batches_tracked``, and a pretrained checkpoint arrives
#:    with that counter in the thousands. Measured: from ``n=3000`` and a
#:    running mean of -99.0, four batches of target data move it to -98.86 --
#:    0.14% of the way. The arm would run, the config would say ``adabn``, and
#:    the statistics would still be SCUT's.
#: 2. **Calling ``model.train()``.** SR-GNN applies ``Dropout(0.2)`` twice, so
#:    the statistics would be accumulated over dropped-out activations that
#:    eval-mode inference never sees. Only the BN modules go into train mode.
#: 3. **Re-estimating on the wrong rows.** The harness calls ``predict``
#:    before any ``train_epoch``, on the INNER-VAL rows -- adapting there
#:    would compute the representation from the data the early stopping is
#:    judged on. Adaptation is bound to ``train_epoch``, which receives
#:    inner-train only.
#:
#: **Read the outcomes, stated first.**
#:
#: * **~0.25** -- the whole gap was normalisation-statistics mismatch. The
#:   architecture reading dies; SR-GNN's representation is fine and the
#:   frozen-probe result was an artefact of feeding it SCUT statistics. **This
#:   direction is conclusive**, and it is the one the scoping below does not
#:   weaken: if re-normalising the last two layers alone recovers the gap, the
#:   representation was never the problem.
#: * **~0.134** -- the alternative is **NARROWED, NOT REMOVED.** See the
#:   scoping immediately below. An earlier draft of this record said such a
#:   result would ESTABLISH the architecture reading. **That was wrong**, and
#:   it was wrong in the project's most familiar way: a measurement read as
#:   answering a question one size larger than the one it asks.
#:
#: ----------------------------------------------------------------------
#: **WHAT THIS ARM CANNOT TEST -- and it is most of the normalisation chain**
#: ----------------------------------------------------------------------
#: [MEASURED 2026-08-01] SR-GNN has 42 BatchNorm modules. This arm re-estimates
#: **two** of them -- ``bn1`` and ``bn2``, the last two layers before the
#: classifier. The other 40 are Xception's, and the reason they are excluded is
#: correct but its consequence was missed: they do not run *on the cleft path*
#: because the feature map arrives precomputed.
#:
#: **They ran during EXTRACTION.** ``train/extract.py`` loads the pretraining
#: checkpoint's full state dict -- running statistics included -- calls
#: ``model.eval()``, and runs ``model.backbone(batch)`` over the cleft images.
#: In eval mode those 40 BatchNorms normalise with **SCUT running statistics**,
#: and their output IS the ``feature_map`` artifact. So the statistics mismatch
#: this arm exists to test is **baked into the artifact it consumes**, forty
#: layers deep, and no cleft-time policy can reach it.
#:
#: **The confound is one-directional against the comparator.** ViT-B/16 is
#: LayerNorm, which has no running statistics at all -- it normalises per
#: sample, so nothing equivalent happened on the 0.2529 side. The asymmetry
#: therefore runs against SR-GNN in exactly the direction that would make it
#: look weaker.
#:
#: **What would test it: a re-extraction, not another cleft arm.** The backbone
#: would have to run over the cleft images with its own BN statistics
#: re-estimated on them, producing a new embedding set. That is a Phase 6
#: extraction change with its own artifact and its own hash -- deliberately not
#: built here, because it is a different piece of work and because inventing it
#: to rescue a conclusion would be fitting the instrument to the answer.
#:
#: **The same one-seed caveat binds this arm.** SD 0.025 puts a single result
#: in a ~0.05-wide interval, so neither number is quotable as a value. What
#: survives the interval is which of two ~0.12-separated hypotheses it falls
#: under, and that is all this arm is asked for.
ADABN_DIAGNOSTIC = {
    "decided": "2026-08-01",
    "config": "configs/p6_srgnn_frozen_graph_adabn.yaml",
    "is_a_ladder_arm": False,
    "trainable": "classifier_adabn",
    "differs_from_frozen_probe_in": "the BN re-estimation, and nothing else",
    "purpose": (
        "test the last alternative the frozen probe could not exclude: that "
        "the gap is normalisation-statistics mismatch rather than the "
        "representation"
    ),
    "procedure": (
        "per fold: reset running statistics; BN modules alone to train mode "
        "with momentum=None; forward the inner-train fold under no_grad; "
        "return to eval; fit the classifier as before"
    ),
    "precedent": (
        "AdaBN is the documented remedy (PLAN §4.7) and the same mechanism was "
        "the root cause of the void ladder's head-init defect -- a failure "
        "mode with a record in this project"
    ),
    "outcome_high": (
        "~0.25 -- CONCLUSIVE: the whole gap was statistics mismatch, the "
        "architecture reading dies, and SR-GNN's representation is fine. This "
        "direction is unaffected by the scoping below"
    ),
    "outcome_low": (
        "~0.134 -- the normalisation alternative is NARROWED, not removed: it "
        "survives upstream, in the 40 backbone BatchNorms frozen into the "
        "artifact. The architecture reading is NOT established"
    ),
    #: **The scope limit, and it is most of the chain.** [MEASURED] This arm
    #: re-estimates 2 of SR-GNN's 42 BatchNorms. The other 40 ran at
    #: EXTRACTION time -- extract.py loads the checkpoint's running statistics,
    #: calls eval(), and runs model.backbone(batch) over the cleft images --
    #: so their SCUT-statistics normalisation is baked into the feature_map
    #: artifact and no cleft-time policy can reach it.
    "does_not_test": {
        "modules_adapted": 2,
        "modules_total": 42,
        "unreachable": (
            "the 40 Xception BatchNorms, which normalised the cleft images "
            "with SCUT running statistics during extraction; their output IS "
            "the artifact this arm consumes"
        ),
        "evidence": "train/extract.py: load_state_dict, model.eval(), model.backbone(batch)",
        "comparator_asymmetry": (
            "ViT-B/16 is LayerNorm and has no running statistics, so nothing "
            "equivalent happened on the 0.2529 side -- the confound runs "
            "against SR-GNN in the direction that makes it look weaker"
        ),
        "what_would_test_it": (
            "a RE-EXTRACTION with the backbone's BN statistics re-estimated on "
            "cleft images, producing a new embedding set -- a Phase 6 "
            "extraction change with its own artifact and hash, not another "
            "cleft arm. Deliberately not built here"
        ),
    },
    "corrected": (
        "an earlier draft of outcome_low said ~0.134 would ESTABLISH the "
        "architecture reading. Wrong, and wrong in this project's most "
        "familiar way -- a measurement read as answering a question one size "
        "larger than the one it asks (R2). Found by adversarial review before "
        "the arm ran"
    ),
    "seeds": 1,
    "one_seed_caveat": (
        "SD 0.025 puts a single result in a ~0.05-wide interval, so neither "
        "number is quotable as a value. What survives is which of two "
        "~0.12-separated hypotheses it falls under"
    ),
    "not_a_search": (
        "one hypothesis test. If it lands low, the answer is the architecture "
        "reading -- not a hunt for a fourth thing to adjust"
    ),
    #: **[MEASURED 2026-08-01, run p6-srgnn-frozen-graph-adabn] IT FELL
    #: OUTSIDE BOTH PRE-REGISTERED READINGS, and the surprise is the
    #: evidence.**
    #:
    #: 0.0468 -- **worse than the plain frozen probe's 0.1340 by 0.087**, about
    #: 3.5 seed-SDs. Neither ``outcome_high`` (~0.25) nor ``outcome_low``
    #: (~0.134) covers it, and it is recorded here as an outcome the design did
    #: not anticipate rather than folded into the nearer of the two. Writing the
    #: readings down first is what makes that statement possible: a prediction
    #: recorded in advance can be MISSED, which is the whole value of recording
    #: it.
    #:
    #: **The mechanism is the one the scope limit named.** Re-estimating 2 of
    #: 42 BatchNorms BREAKS THE CHAIN: bn1 and bn2 now expect cleft-statistics
    #: inputs while receiving activations that passed through 40 Xception
    #: layers still normalised with SCUT statistics. Partial adaptation is
    #: worse than none, because a uniform mismatch becomes an internally
    #: inconsistent one. ``shrinkage`` moved 0.2943 -> 0.4221: the head fits
    #: harder to a disturbed representation.
    #:
    #: **This is direct evidence FOR the correction, and it upgrades it from
    #: an argument to a measurement.** If touching 2 of 42 layers moves the
    #: result by 0.087, the 40 baked into the artifact are demonstrably not
    #: idle history. So the frozen probe's 0.1340 cannot be attributed to
    #: SR-GNN's representation while forty layers of its normalisation chain
    #: are calibrated for the wrong domain -- and ViT carries no equivalent
    #: handicap, LayerNorm having no running statistics at all.
    #:
    #: **The architecture reading therefore remains UNTESTED**, and the
    #: re-extraction moved from optional to necessary: without it the write-up
    #: would report that the documented remedy was applied to 2 of 42 layers
    #: and made things worse, which is a fact about this experiment rather than
    #: about the architecture.
    "result": {
        "measured": "2026-08-01",
        "run": "p6-srgnn-frozen-graph-adabn",
        "pcc": 0.0468,
        "against_plain_probe": -0.087,
        "in_seed_sds": 3.5,
        "shrinkage": {"plain_probe": 0.2943, "adabn": 0.4221},
        "branch_taken": "NEITHER -- outside both pre-registered readings",
        "mechanism": (
            "partial adaptation breaks the chain: bn1/bn2 expect "
            "cleft-statistics inputs while receiving activations normalised by "
            "40 upstream layers still on SCUT statistics. A uniform mismatch "
            "became an internally inconsistent one, which is worse than none"
        ),
        "licensed": (
            "the 40 extraction-time BatchNorms are demonstrably load-bearing -- "
            "2 of 42 moved the result 0.087. The scope limit is now measured, "
            "not argued"
        ),
        "not_licensed": (
            "any reading about SR-GNN's representation. The architecture "
            "reading is UNTESTED: 0.1340 cannot be attributed to the "
            "representation while 40 layers are calibrated for the wrong domain"
        ),
        "consequence": (
            "the re-extraction is necessary, not optional -- see "
            "PER_FOLD_REEXTRACTION"
        ),
    },
}

#: **[MEASURED 2026-08-01, system python, torch 2.13.0+cu126, CPU] What the
#: AdaBN implementation stands on, probed before it was written.**
#:
#: Every one of these is a way the arm could report success while doing
#: nothing, so none of them is taken from memory or from documentation.
#: ``scripts/verify_backbone_builds.py`` re-checks them against the shipped
#: code path; these are the isolated measurements that decided the design.
MECHANISM_VERIFIED = {
    "measured": "2026-08-01",
    "environment": "system python, torch 2.13.0+cu126, CPU",
    # momentum=None is a cumulative average, so after a reset the running
    # statistics are exactly the mean of the batch statistics -- and with
    # equal batch sizes that is exactly the dataset statistic.
    "momentum_none_is_cumulative": True,
    "after_reset_equals_mean_of_batch_stats": True,
    # THE ONE THAT MATTERS. Without the reset, a pretrained counter makes the
    # update weight 1/n with n in the thousands.
    "without_reset_is_effectively_a_noop": {
        "start_running_mean": -99.0,
        "start_num_batches_tracked": 3000,
        "target_mean": 5.0,
        "after_four_batches": -98.86,
        "fraction_of_the_way": 0.0014,
        "verdict": "reset_running_stats() is MANDATORY, not tidiness",
    },
    # model.eval() then BN-only .train() leaves dropout off: two passes over
    # the same input are bitwise equal while the BN counter still advances.
    "bn_only_train_mode_leaves_dropout_off": True,
    # BatchNorm1d refuses a batch of 1 in train mode, so the accumulation
    # pass needs a guard rather than a stack trace mid-fold.
    "batch_of_one_raises_in_train_mode": (
        "ValueError: Expected more than 1 value per channel when training"
    ),
    #: **A stated approximation, not a defect.** The cumulative average
    #: weights every batch EQUALLY regardless of size, so a final short batch
    #: is over-weighted relative to its row count.
    #:
    #: **[MEASURED 2026-08-01, this arm's real folds]** ``inner_val_split``
    #: over 237 patients at 5 folds and ``inner_val_frac`` 0.2 gives inner-train
    #: sizes **151, 151, 152, 152, 152** -- so at batch 16 every fold runs
    #: **10 batches** with a last of 7 or 8. That short batch carries 1/10 =
    #: 0.1000 of the weight against a true share of 7/151 = 0.0464 or
    #: 8/152 = 0.0526: over-weighted by **1.9x to 2.2x**.
    #:
    #: An earlier draft of this comment said "inner-train ~121 rows ... 7 full
    #: batches and one of 9". That was wrong -- it applied the 0.2 inner-val
    #: fraction twice, to a figure that was already the inner-train count.
    #: Corrected against the project's own splitter rather than re-derived by
    #: hand a second time.
    "equal_batch_weighting_caveat": (
        "the cumulative average weights batches equally, so a short final "
        "batch is over-weighted relative to its row count; the statistics are "
        "the mean of BATCH statistics, not exactly the fold's"
    ),
    "measured_batching": {
        "inner_train_sizes": [151, 151, 152, 152, 152],
        "batch_size": 16,
        "n_batches_per_fold": 10,
        "last_batch_sizes": [7, 7, 8, 8, 8],
        "short_batch_weight": 0.1,
        "short_batch_true_share": [0.0464, 0.0526],
        "over_weighted_by": "1.9x to 2.2x",
    },
    #: **The policy is SR-GNN-only, and by architecture rather than by
    #: choice.** SR-GNN carries 42 BatchNorm modules of which exactly 2
    #: (``bn1``, ``bn2``) sit after the frozen boundary; AG-Net carries 53 and
    #: **none** of them do -- every one is inside the ResNet-50 backbone,
    #: which never runs when the feature map arrives precomputed.
    #:
    #: The policy must therefore refuse on AG-Net rather than silently adapt
    #: nothing.
    #:
    #: **[CORRECTED 2026-08-01] It does NOT follow that AG-Net needs no
    #: statistics caveat -- the opposite follows.** An earlier draft of this
    #: entry concluded that with no BatchNorm after the frozen boundary there
    #: were no statistics to be mismatched. That reads "after the boundary" as
    #: "at all", and the 53 modules are not idle: they ran during EXTRACTION,
    #: in eval mode, normalising the cleft images with SCUT running statistics
    #: on the way to the artifact. So AG-Net's whole normalisation chain is
    #: baked in and **none of it is reachable at cleft time** -- it is the
    #: worse case, not the exempt one. Same error as the AdaBN outcome
    #: overclaim, in a second place, from the same missing step.
    "post_backbone_batchnorms": {
        "srgnn": {"total": 42, "after_frozen_boundary": 2, "named": ["bn1", "bn2"]},
        "agnet": {"total": 53, "after_frozen_boundary": 0},
        "consequence": (
            "adabn is SR-GNN-only by architecture; on AG-Net the policy refuses"
        ),
        "not_a_consequence": (
            "that AG-Net results carry no statistics caveat. Its 53 modules "
            "ran at EXTRACTION with SCUT running statistics, so its entire "
            "normalisation chain is baked into the artifact and none of it is "
            "reachable at cleft time -- AG-Net is the worse case, not the "
            "exempt one"
        ),
    },
}


#: **[MEASURED 2026-08-01, all four cells] THE NORMALISATION HYPOTHESIS IS
#: REFUTED, and so is the account that replaced it. Both by their own
#: pre-registered predictions.**
#:
#: Two normalisation halves, set independently -- the 40 Xception BatchNorms
#: fixed at extraction, and the 2 post-backbone ones reachable at cleft time::
#:
#:                          | bn1/bn2 SCUT   | bn1/bn2 cleft
#:     ---------------------+----------------+---------------
#:     40 backbone BN SCUT  | A  0.1340      | B  0.0468
#:     40 backbone BN cleft | C  0.1093      | D  0.0262
#:
#: **A > C > B > D: more cleft adaptation is monotonically WORSE.** The
#: unadapted cell is the best of the four and the fully consistent cell is the
#: worst.
#:
#: **What each refutation rests on -- the predictions were written first.**
#:
#: * The **normalisation hypothesis** said D would land near 0.25. It landed
#:   at 0.0262, 0.224 below. Refuted.
#: * The **partial-adaptation account** -- which explained B's 0.0468 as an
#:   internally inconsistent chain, two layers calibrated for cleft fed by
#:   forty calibrated for SCUT -- predicted that making the chain consistent
#:   would recover. D is 0.021 BELOW B. Consistency made it worse, so
#:   inconsistency was never the mechanism. Refuted.
#:
#: Recording both as refuted is the whole return on registering them. An
#: account adjusted after each result would have survived all four cells and
#: explained nothing.
NORMALISATION_2X2 = {
    "measured": "2026-08-01",
    "cells": {
        "A": {"backbone_bn": "scut", "head_bn": "scut", "pcc": 0.1340},
        "B": {"backbone_bn": "scut", "head_bn": "cleft", "pcc": 0.0468},
        "C": {"backbone_bn": "cleft", "head_bn": "scut", "pcc": 0.1093,
              "spearman": 0.0987, "shrinkage": 0.2344,
              "selected_epochs": [4, 6, 3, 2, 4]},
        "D": {"backbone_bn": "cleft", "head_bn": "cleft", "pcc": 0.0262,
              "spearman": 0.0005, "shrinkage": 0.5285,
              "selected_epochs": [1, 4, 4, 1, 2]},
    },
    "trainable_parameters": 2049,
    "ordering": "A > C > B > D -- monotonically worse with more cleft adaptation",
    "refutes": {
        "normalisation_hypothesis": {
            "predicted": "D ~ 0.25",
            "measured": 0.0262,
            "verdict": "REFUTED by its own pre-registered prediction",
        },
        "partial_adaptation_account": {
            "predicted": "the consistent cell (D) recovers relative to B",
            "measured": "D is 0.021 BELOW B",
            "verdict": (
                "REFUTED -- consistency made it worse, so an inconsistent "
                "chain was never the mechanism"
            ),
        },
    },
    #: **An explanation worth naming and NOT worth chasing.** Re-estimating
    #: BatchNorm on 152 inner-train images at batch 16 gives **ten batches per
    #: layer** to estimate 2048 channels, with the short final batch
    #: over-weighted 1.9-2.2x by the cumulative average -- and forty such
    #: layers compose. SCUT's statistics come from 3,300 images (~104 batches)
    #: and are far better estimated even though the domain is wrong. **A
    #: well-estimated wrong normalisation beats a badly-estimated right one.**
    #:
    #: It predicts the monotonicity: every step toward cleft statistics adds
    #: estimation noise, and D adds it twice.
    #:
    #: **Not pursued, deliberately.** Testing it means re-estimating on all
    #: 237 or at a larger batch -- a different question, and running variants
    #: until one improves would be searching for a configuration rather than
    #: testing a hypothesis. Recorded with the arithmetic so the next person
    #: inherits the reasoning rather than the itch.
    "unpursued_explanation": {
        "name": "estimation quality, not domain",
        "claim": (
            "a well-estimated wrong normalisation beats a badly-estimated "
            "right one"
        ),
        "arithmetic": {
            "cleft_images": 152,
            "cleft_batch_size": 16,
            "cleft_batches_per_layer": 10,
            "channels_estimated": 2048,
            "short_batch_over_weighted": "1.9x to 2.2x",
            "layers_composing": 40,
            "scut_images": 3300,
            "scut_batches": 104,
        },
        "explains": "the monotonicity -- each step toward cleft adds noise",
        "would_be_tested_by": "re-estimating on all 237, or at a larger batch",
        "status": "UNPURSUED -- a different question; chasing it would be a search",
    },
    #: **What the architecture reading now rests on, stated exactly.** Cell A
    #: is the best of four, and every normalisation correction reachable from
    #: the artifact boundary was measured and is worse. That is what the
    #: reading stands on.
    #:
    #: It is NOT the same as "the domain mismatch does not exist". The
    #: residual possibility is named above: a BETTER-ESTIMATED cleft
    #: normalisation might help. It is recorded as unpursued, not eliminated
    #: -- this sequence has already produced one overclaim of exactly that
    #: shape, and the distinction between "no reachable correction helps" and
    #: "no correction could help" is the one that was missed then.
    "what_stands": {
        "best_srgnn_configuration": "A",
        "frozen_probe": 0.1340,
        # The band's mean at its recorded precision, not a rounded retype --
        # a second copy of a number is a second thing that can drift.
        "trained_graph_layers": 0.15067,
        "vit_frozen_probe": 0.2529,
        "fraction_of_vit": {"frozen": 0.530, "trained_graph_layers": 0.596},
        "architecture_reading": (
            "stands: SR-GNN's representation is weaker than an off-the-shelf "
            "ImageNet ViT embedding for this task at n=237, at roughly 60% of "
            "it. The statistics alternative was tested four ways and every "
            "reachable correction is worse"
        ),
        "residual": (
            "bounded and named: estimation QUALITY, not domain. No reachable "
            "correction helps; whether a better-estimated one would is "
            "unpursued rather than eliminated"
        ),
    },
}


#: **[MEASURED 2026-08-01, Stage E0] THE SCHEME AXIS IS NULL AT CLEFT TIME
#: TOO -- the fourth null, and the one measured under the condition the
#: scheme design was actually built for.**
#:
#: SR-GNN at ImageNet init, four schemes, ten seeds each, trained graph
#: layers. ImageNet has no pretraining checkpoint, so the four cells share one
#: scheme-free embedding set and differ in **placement alone**::
#:
#:     native 0.0926   random 0.0794   grid 0.0778   anatomy 0.0765
#:
#: Range **0.0161** against a ten-seed threshold of ~0.0286. The largest gap,
#: native-anatomy at 0.0161, needs 0.0286. **Nothing is claimable.**
#:
#: **This closes the axis.** Four independent attempts now: Phase 4's pooled
#: frozen patch embeddings; the superseded epoch-confounded pretraining run;
#: the thirty-run fixed-budget pretraining lattice; and this -- cleft data,
#: trained graph layers, message passing live, no checkpoint confound.
#: ``SCHEME_AXIS_AT_PRETRAINING``'s caveat said the cleft comparison remained
#: live precisely because pretraining could not answer it. It has been
#: answered, and the answer is the same.
#:
#: **Region placement does not affect performance on this task.** the supervision material's
#: thirds-based grid, the anatomy-anchored regions, the architecture's own
#: generator and a random control are indistinguishable once the confound is
#: removed. That is a finding about the design's central premise, not a
#: failure to measure.
SCHEME_AXIS_AT_CLEFT = {
    "measured": "2026-08-01",
    "arm": "srgnn, imagenet init, g2, trained graph layers, 10 seeds",
    "why_imagenet": (
        "imagenet has no pretraining checkpoint, so the four cells share one "
        "scheme-free embedding set and vary placement ALONE"
    ),
    "means": {
        "native": 0.0926, "random": 0.0794, "grid": 0.0778, "anatomy": 0.0765,
    },
    "range": 0.0161,
    "largest_gap": {"pair": "native-anatomy", "delta": 0.0161, "threshold": 0.0286},
    "claimable": False,
    "null_number": 4,
    "the_four": (
        "Phase 4 pooled frozen patch embeddings; the superseded "
        "epoch-confounded pretraining run; the thirty-run fixed-budget "
        "pretraining lattice; this cleft arm with trained graph layers and no "
        "checkpoint confound"
    ),
    "verdict": (
        "region placement does not affect performance on this task, measured "
        "under the condition the scheme design was built for"
    ),
}

#: **[MEASURED 2026-08-01, Stage E against Stage E0] SCUT TEST PCC DOES NOT
#: PREDICT CLEFT TRANSFER QUALITY.**
#:
#: The four scheme-matched SR-GNN checkpoints are **statistically equal on
#: SCUT** -- the scheme axis at pretraining ranged 0.0016-0.0154, inside the
#: 0.0137 seed band, with no scheme winning twice. Consuming them at cleft
#: time, the same four schemes spread by **0.095**::
#:
#:     Stage E  (masked-G2, scheme-matched checkpoints):  0.0554 - 0.1507
#:     Stage E0 (imagenet, one shared representation):    0.0765 - 0.0926
#:
#: Placement contributes 0.016 of that. The remaining ~0.08 is **transfer
#: quality differing between checkpoints a 2,199-face SCUT test set called
#: equivalent**.
#:
#: **This is the licence that was wrongly granted, now measured.** Stage E's
#: checkpoint variation was licensed on the pretraining null; the null is real
#: and the inference from it was not. A held-out score on the pretraining task
#: bounds performance on that task, and says nothing about which
#: representation transfers -- 2,199 beauty ratings against 237 clinical
#: images is a different question, and R2 is the rule that should have caught
#: it.
#:
#: **It generalises beyond Stage E**: every ladder arm picks a checkpoint on
#: the strength of its SCUT number. Q1 and Q2 are exactly that comparison, and
#: this says the pretraining metric cannot stand in for the cleft one.
PRETRAINING_DOES_NOT_PREDICT_TRANSFER = {
    "measured": "2026-08-01",
    "scut_scheme_range": 0.0154,
    "scut_seed_band": 0.0137,
    "scut_verdict": "statistically equal on 2,199 held-out faces",
    "cleft_range_with_those_checkpoints": 0.0953,
    "cleft_range_with_one_representation": 0.0161,
    "attributable_to_placement": 0.0161,
    "attributable_to_transfer_quality": "~0.08",
    "verdict": (
        "SCUT test PCC does not predict cleft transfer quality. A held-out "
        "score bounds performance on the pretraining task and says nothing "
        "about which representation transfers to 237 clinical images"
    ),
    "consequence": (
        "the licence Stage E rested on was an inference from a null, not a "
        "measurement. Every arm that picks a checkpoint on its SCUT number "
        "inherits the same gap -- Q1 and Q2 are that comparison"
    ),
    # [2026-08-31] The range's ENDPOINTS are now traceable to named arms:
    # STAGE_E_PER_ARM banks the two cells this record only ever carried
    # inside a range. Nothing numeric here changes -- the range was
    # already correct -- but "0.0554" is now a rater-free arm name
    # (anatomy) rather than a bare endpoint.
    "endpoints_now_traceable_2026_08_31": "STAGE_E_PER_ARM",
}


#: **[MEASURED 2026-08-31]** -- by the runs, transcribed here. [TAG
#: NORMALISED 2026-09-01 from '[REPORTED 2026-08-31]'.]
#: **STAGE E's TWO MISSING PER-ARM VALUES, BANKED.**
#:
#: ``PRETRAINING_DOES_NOT_PREDICT_TRANSFER`` has carried the Stage E spread
#: as a RANGE (0.0554 - 0.1507) since 2026-08-01, and ``STAGE_E_COMPLETE``
#: banked only the random arm. The grid and anatomy cells existed inside
#: that range without ever being written down as arms. They are here now::
#:
#:     p7_e_srgnn_scut_masked_g2_grid      0.0889  (sd 0.0224, 10 seeds)
#:     p7_e_srgnn_scut_masked_g2_anatomy   0.0554  (sd 0.0251, 10 seeds)
#:
#: **Provenance, and it is not a new measurement.** These come from Phase
#: 18 run 5's own table (``p18_metric_space__9411267e__p18-metric-space-5``),
#: which recomputed every locked arm's PCC **from the arms' own banked
#: prediction CSVs** -- the same CSVs the original runs wrote. **No arm was
#: re-run.** Extraction ; tagged ``[REPORTED]`` on the same
#: footing as the Phase 10 annex's per-rater table, because the run
#: directory is CLUSTER-ONLY and was not read on the authoring machine.
#:
#: **The range reconciles exactly, and that is the check that makes these
#: safe to bank.** The four masked-G2 scheme cells are native 0.1507
#: (Stage D), grid 0.0889, anatomy 0.0554, random 0.1354. Their spread is
#: 0.1507 - 0.0554 = **0.0953**, which is
#: ``cleft_range_with_those_checkpoints`` to four decimals; the E0 cells
#: (native 0.0926, grid 0.0778, anatomy 0.0765, random 0.0794) spread
#: **0.0161**, which is ``cleft_range_with_one_representation`` exactly.
#: **Anatomy is the 0.0554 endpoint** the docstring has quoted all along.
#: Two independently-banked ranges reproducing from these four values is
#: corroboration, not arithmetic that was going to work anyway.
#:
#: **No ledger row.** These are descriptive per-arm values sitting inside
#: an already-banked range; no contrast was run, no verdict rests on them,
#: and the range they compose was banked in 2026-08-01.
STAGE_E_PER_ARM = {
    "reported": "2026-08-31",
    "provenance": (
        "[REPORTED] -- Phase 18 run 5's table "
        "(p18_metric_space__9411267e__p18-metric-space-5), recomputed "
        "from the arms' own banked prediction CSVs; NO arm was re-run. "
        "Extraction ; the run directory is CLUSTER-ONLY and "
        "was not read on the authoring machine"
    ),
    "arms": {
        "p7_e_srgnn_scut_masked_g2_grid": {
            "pcc": 0.0889, "sd": 0.0224, "n_seeds": 10,
        },
        "p7_e_srgnn_scut_masked_g2_anatomy": {
            "pcc": 0.0554, "sd": 0.0251, "n_seeds": 10,
        },
    },
    "completes": (
        "the four masked-G2 scheme cells: native 0.1507 (Stage D), grid "
        "0.0889, anatomy 0.0554, random 0.1354 (STAGE_E_COMPLETE)"
    ),
    "range_reconciles": (
        "0.1507 - 0.0554 = 0.0953, which is "
        "PRETRAINING_DOES_NOT_PREDICT_TRANSFER's "
        "cleft_range_with_those_checkpoints to four decimals; the E0 "
        "cells spread 0.0161, its cleft_range_with_one_representation "
        "exactly. Two independently-banked ranges reproducing from these "
        "values is corroboration, not arithmetic that had to work"
    ),
    "anatomy_is_the_endpoint": (
        "anatomy 0.0554 IS the low endpoint the range has quoted since "
        "2026-08-01; the endpoint now has an arm's name"
    ),
    "no_ledger_row": (
        "descriptive per-arm values inside an already-banked range; no "
        "contrast was run and no verdict rests on them"
    ),
}


#: **[MEASURED 2026-08-02] STAGE E IS COMPLETE -- four schemes, four complete
#: ten-seed runs, and the range is unchanged.**
#:
#: ``p7-e-srgnn-random-2`` relaunched and finished at **0.1354, SD 0.0276 over
#: ten seeds** -- identical to the figure the stopped run reported. The
#: original was halted prematurely, not failed, so this confirms rather than
#: revises.
#:
#: **Nothing numeric changes, and that is the point worth recording.** The
#: 0.0554-0.1507 range in ``PRETRAINING_DOES_NOT_PREDICT_TRANSFER`` already
#: carried this arm, and 0.1354 is not an endpoint of it. What changes is the
#: PROVENANCE of the range: it rested on three complete runs and one partial,
#: and now rests on four complete ones. **A range quoted from a partial run is
#: a different quantity from the same range quoted from complete runs** -- R2,
#: in the mildest form it takes -- and the two agreeing is the evidence, not
#: an excuse for not having checked.
STAGE_E_COMPLETE = {
    "measured": "2026-08-02",
    "arm": "p7-e-srgnn-random-2",
    "mean": 0.1354,
    "sd": 0.0276,
    "n_seeds": 10,
    "reproduces": "the stopped run's figure exactly",
    "why_it_was_rerun": "the original was halted prematurely, not a failure",
    "range_unchanged": True,
    "what_changed": (
        "the 0.0554-0.1507 range now rests on four complete ten-seed runs "
        "rather than three complete and one partial. The number is the same; "
        "the quantity behind it is not"
    ),
}


class GraphCleftError(RuntimeError):
    """The graph cleft arm cannot run as configured."""


# --------------------------------------------------------------------------
# packing: maps and boxes in one harness-sliceable array
# --------------------------------------------------------------------------


def pack(maps: np.ndarray, boxes: np.ndarray | None) -> np.ndarray:
    """(N, C, H, W) maps [+ (N, R, 4) boxes] -> (N, C*H*W [+ R*4]) float32."""
    array = np.asarray(maps, dtype=np.float32)
    if array.ndim != 4:
        raise GraphCleftError(f"maps must be (N, C, H, W); got {array.shape}")
    flat = array.reshape(array.shape[0], -1)
    if boxes is None:
        return flat
    box_array = np.asarray(boxes, dtype=np.float32)
    if box_array.ndim != 3 or box_array.shape[0] != array.shape[0]:
        raise GraphCleftError(
            f"boxes must be (N, R, 4) with N={array.shape[0]}; got "
            f"{box_array.shape}"
        )
    return np.concatenate([flat, box_array.reshape(array.shape[0], -1)], axis=1)


def pack_indexed(n_rows: int, boxes: np.ndarray | None) -> np.ndarray:
    """Rows of ``[row_index, boxes...]`` for the PER-FOLD artifact.

    **Why the features are not in the row here.** Per-fold re-extraction gives
    five feature sets, one per fold, and the frozen ``run_cv`` slices a single
    array. Stacking all five into every row would make each row five times
    wider and ship 80% of it to the GPU to be discarded, on every batch of
    every epoch of every fold of every seed.

    So the row carries the patient's ROW INDEX and the backbone gathers from
    the fold's own array. The indirection is safe to check because the index
    column is the identity: ``features[rows][:, 0] == rows`` for any slice the
    harness takes, which is asserted where it is used. Boxes still travel in
    the row -- they are small and genuinely per-patient.
    """
    index = np.arange(n_rows, dtype=np.float32).reshape(n_rows, 1)
    if boxes is None:
        return index
    box_array = np.asarray(boxes, dtype=np.float32)
    if box_array.ndim != 3 or box_array.shape[0] != n_rows:
        raise GraphCleftError(
            f"boxes must be (N, R, 4) with N={n_rows}; got {box_array.shape}"
        )
    return np.concatenate(
        [index, box_array.reshape(n_rows, -1)], axis=1
    ).astype(np.float32)


def unpack_indexed(
    packed: np.ndarray, n_regions: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """The inverse of ``pack_indexed``: integer row indices, and boxes."""
    array = np.asarray(packed, dtype=np.float32)
    expected = 1 + n_regions * 4
    if array.shape[1] != expected:
        raise GraphCleftError(
            f"indexed rows are {array.shape[1]} wide; {n_regions} boxes needs "
            f"{expected}"
        )
    indices = np.rint(array[:, 0]).astype(int)
    if n_regions == 0:
        return indices, None
    return indices, array[:, 1:].reshape(-1, n_regions, 4)


def unpack(
    packed: np.ndarray, map_shape: tuple[int, int, int], n_regions: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """The exact inverse of ``pack`` for rows of known layout."""
    array = np.asarray(packed, dtype=np.float32)
    map_size = int(np.prod(map_shape))
    expected = map_size + n_regions * 4
    if array.shape[1] != expected:
        raise GraphCleftError(
            f"packed rows are {array.shape[1]} wide; layout "
            f"{map_shape}+{n_regions} boxes needs {expected}"
        )
    maps = array[:, :map_size].reshape(-1, *map_shape)
    if n_regions == 0:
        return maps, None
    return maps, array[:, map_size:].reshape(-1, n_regions, 4)


# --------------------------------------------------------------------------
# the trained half: graph layers over frozen maps
# --------------------------------------------------------------------------


@dataclass
class GraphHeadBackbone:
    """The harness Backbone protocol over packed (map, boxes) rows. Torch."""

    name: str
    map_shape: tuple[int, int, int]
    n_regions: int
    #: The pretraining checkpoint's arrays (loaded ONCE by the caller), or
    #: None for imagenet -- seed-initialised graph layers.
    checkpoint_arrays: dict | None
    #: ``graph_layers`` is the regime; ``classifier`` and ``classifier_adabn``
    #: are the diagnostics that reduce it to a probe on purpose
    #: (``FROZEN_GRAPH_DIAGNOSTIC``, ``ADABN_DIAGNOSTIC``).
    trainable: str = "graph_layers"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 16
    seed: int = 1337
    device: str = "cuda"
    parameter_report: dict = field(default_factory=dict)

    #: PER-FOLD artifact mode. ``maps_by_fold[fold]`` is that fold's full
    #: (N, C, H, W) array; ``fold`` says which one THIS instance is for; and
    #: ``test_rows`` is that fold's test row set, used to assert at runtime
    #: that the arm never trains on rows its own features were adapted to
    #: exclude. All None in the ordinary single-artifact mode.
    maps_by_fold: dict | None = None
    fold: int | None = None
    test_rows: frozenset | None = None

    _model: Any = field(default=None, repr=False)
    _optimizer: Any = field(default=None, repr=False)
    _epoch: int = field(default=0, repr=False)
    #: Set once the BN statistics have been re-estimated for this fold. The
    #: frozen stack does not change afterwards, so re-running it every epoch
    #: would recompute identical values.
    _adapted: dict | None = field(default=None, repr=False)

    def reset(self, train_labels: np.ndarray) -> None:
        import torch

        from ..models.factory import create_backbone

        torch.manual_seed(self.seed)
        self._model = create_backbone(self.name, pretrained=False, num_outputs=1)
        self._epoch = 0

        if self.checkpoint_arrays is not None:
            loaded = {
                key[len(MODEL_PREFIX):]: torch.as_tensor(np.asarray(value))
                for key, value in self.checkpoint_arrays.items()
                if key.startswith(MODEL_PREFIX)
            }
            expected = set(self._model.state_dict())
            if set(loaded) != expected:
                missing = sorted(expected - set(loaded))[:3]
                raise GraphCleftError(
                    f"checkpoint arrays do not match the {self.name!r} "
                    f"architecture (e.g. missing {missing}); warm-starting "
                    "from a different architecture is not a warm start"
                )
            self._model.load_state_dict(loaded)

        # The classifier alone is re-initialised: zero weights, bias at the
        # training-fold mean -- gate 3's premise, satisfied by construction.
        head = self._model.classifier
        with torch.no_grad():
            torch.nn.init.zeros_(head.weight)
            head.bias.fill_(float(np.mean(train_labels)))

        # Freeze the backbone; train everything after the map -- or, under the
        # diagnostic policy, freeze the graph layers too and train the
        # classifier alone. The backbone never even runs here (features are
        # precomputed), but freezing makes the parameter report say what the
        # arm IS rather than what it does.
        if self.trainable not in TRAINABLE_POLICIES:
            raise GraphCleftError(
                f"unknown trainable policy {self.trainable!r}; expected one of "
                f"{TRAINABLE_POLICIES}"
            )
        # A SET, not `== "classifier"`. The AdaBN probe freezes exactly the
        # same parameters; a literal comparison here would have trained its
        # graph layers while its config said otherwise.
        classifier_only = self.trainable in PROBE_POLICIES
        self._adapted = None

        trained_parameters = []
        frozen = trainable = 0
        for name, parameter in self._model.named_parameters():
            trains = (
                name.startswith("classifier.")
                if classifier_only
                else not name.startswith("backbone.")
            )
            parameter.requires_grad_(trains)
            if trains:
                trained_parameters.append(parameter)
                trainable += parameter.numel()
            else:
                frozen += parameter.numel()
        if not trained_parameters:
            raise GraphCleftError(
                f"policy {self.trainable!r} left nothing to train on "
                f"{self.name!r}; nothing would learn"
            )

        self.parameter_report = {
            "total_parameters": frozen + trainable,
            "trainable_parameters": trainable,
            "frozen_backbone_parameters": frozen,
            "policy": (
                "frozen_backbone_frozen_graph_layers_trained_classifier"
                if classifier_only
                else "frozen_backbone_trained_graph_layers"
            ),
            "trainable": self.trainable,
            "bn_reestimation": self.trainable == "classifier_adabn",
            **(
                {
                    "diagnostic_note": (
                        "NOT a ladder arm: the brief's own trap ('do not freeze "
                        "the graph layers') run deliberately once, because the "
                        "reduced arm is what compares to the frozen ViT probe"
                    )
                }
                if classifier_only
                else {}
            ),
        }

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        self._optimizer = torch.optim.AdamW(
            trained_parameters, lr=self.learning_rate, weight_decay=self.weight_decay
        )

    @property
    def model(self):
        """The trained model itself.

        **Exposed for Phase 8's arm B**, whose randomisation walks named
        submodules and whose snapshot/restore is a ``state_dict`` round trip
        -- both public torch API on a module this class otherwise keeps to
        itself. Same reasoning as ``FrozenExtractor.model``: the alternative
        is a second class that rebuilds the model, and then "the frozen
        boundary" would be defined in two places.
        """
        if self._model is None:
            raise GraphCleftError(
                "this backbone has not been reset yet, so there is no model. "
                "An explanation read off an unbuilt model would be an "
                "explanation of nothing"
            )
        return self._model

    def _tensors(self, features: np.ndarray):
        import torch

        device = next(self._model.parameters()).device
        if self.maps_by_fold is None:
            maps, boxes = unpack(features, self.map_shape, self.n_regions)
        else:
            indices, boxes = unpack_indexed(features, self.n_regions)
            maps = self.maps_by_fold[self.fold][indices]
        maps_t = torch.as_tensor(np.ascontiguousarray(maps), device=device)
        boxes_t = None if boxes is None else torch.as_tensor(boxes, device=device)
        return maps_t, boxes_t

    def _assert_not_training_on_test_rows(self, features: np.ndarray) -> None:
        """**The row-level half of the leak check.**

        ``embeddings.assert_no_test_fold_leak`` checks the ARTIFACT against
        the fold's test patients at load time. This checks the ROWS actually
        handed to the optimiser, every epoch, against the same fold -- which
        is also what validates that this instance's ``fold`` is the fold the
        harness is really running.

        The two are worth having separately: the first catches an artifact
        built wrong, the second catches the artifact being used for the wrong
        fold. Only the second can see a misalignment between the factory's
        fold counter and ``run_cv``'s iteration.
        """
        if self.maps_by_fold is None or self.test_rows is None:
            return
        indices, _ = unpack_indexed(features, self.n_regions)
        offending = sorted(set(int(i) for i in indices) & self.test_rows)
        if offending:
            raise GraphCleftError(
                f"fold {self.fold}: {len(offending)} training row(s) belong to "
                f"this fold's TEST set, e.g. {offending[:5]}. Either the "
                "per-fold features are being used for the wrong fold, or the "
                "fold assignment moved. Both would train on the patients this "
                "fold is scored on."
            )

    def _forward(self, maps_t, boxes_t):
        if boxes_t is None:
            return self._model.forward_from_features(maps_t)
        return self._model.forward_from_features(maps_t, boxes_t)

    def region_weights(self, features: np.ndarray) -> np.ndarray:
        """``(n, n_nodes)`` -- the model's OWN attention over its regions.

        **[ADDED 2026-08-14] Phase 8 arm B's explanation, read through this
        instance rather than beside it.** ``_tensors`` is what turns a
        harness row into (maps, boxes) on the right device, handling the
        per-fold gather and the box unpacking; an extraction that re-derived
        that would be a second definition of what the model is fed, which is
        the defect class this project keeps finding. So the accessor sits
        HERE, on the object that already owns the packing.

        The weights are not recomputed: ``forward_from_features_with_weights``
        returns the tuple ``_from_features`` already builds
        (``phase8.THE_ARTIFACT_PATH_DISCARDS_THE_EXPLANATION``). Eval mode
        and ``no_grad``, as ``predict`` does -- SR-GNN carries two Dropouts,
        and weights read in train mode would be an explanation of a
        stochastic pass.
        """
        import torch

        accessor = getattr(
            self._model, "forward_from_features_with_weights", None
        )
        if accessor is None:
            raise GraphCleftError(
                f"{self.name!r} has no forward_from_features_with_weights, so "
                "it cannot report which regions it weighted. Both graph "
                "models expose one; a backbone without it has no node-weight "
                "explanation to give"
            )
        self._model.eval()
        out: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(features), self.batch_size):
                maps_t, boxes_t = self._tensors(
                    features[start : start + self.batch_size]
                )
                pair = (
                    accessor(maps_t) if boxes_t is None
                    else accessor(maps_t, boxes_t)
                )
                if not isinstance(pair, tuple) or len(pair) != 2:
                    raise GraphCleftError(
                        f"{self.name!r}'s weight accessor returned "
                        f"{type(pair).__name__}, not the (logits, weights) "
                        "pair _from_features builds"
                    )
                weights = pair[1]
                if weights is None:
                    raise GraphCleftError(
                        f"{self.name!r} returned no region weights. An "
                        "explanation cannot be read off a model that does not "
                        "produce one"
                    )
                array = weights.detach().float().cpu().numpy()
                if array.ndim != 2:
                    raise GraphCleftError(
                        f"region weights are {array.shape}; expected "
                        "(batch, n_nodes). A per-node vector is what the gate "
                        "ranks, and a different rank would be ranked silently"
                    )
                out.append(array)
        if not out:
            raise GraphCleftError(
                "no rows were supplied, so there are no region weights. An "
                "empty explanation set would pass every downstream shape check"
            )
        return np.concatenate(out)

    def _batch_norms(self) -> list:
        """The BatchNorms this arm actually RUNS -- not every one it owns.

        **[MEASURED 2026-08-01] SR-GNN has 42 BatchNorm modules and 40 of them
        are inside the backbone**, which never executes on this path: the
        feature map arrives precomputed, so ``forward_from_features`` enters
        below the backbone entirely. Only ``bn1`` and ``bn2`` see data.

        Adapting all 42 would reset 40 modules that then accumulate nothing,
        leaving them at the freshly-initialised 0/1 rather than the
        checkpoint's pretrained statistics -- silently destroying state this
        arm does not own and never measures, and breaking the model for any
        later caller that does run ``forward()`` from images. Computationally
        inert for this arm, which is exactly why it would not have shown up in
        a result. Caught by the verification asserting every adapted module
        tracked the expected batch count; 40 of them tracked zero.

        The predicate mirrors the freezing logic above, deliberately: what is
        frozen and what is adapted must be decided by the same rule, or the
        two can drift apart.

        **This narrows what the arm can conclude, and that belongs here rather
        than only in the record.** Excluding the 40 is right -- they cannot
        accumulate anything on this path -- but they are not inert history:
        they RAN at extraction time, in eval mode, normalising the cleft
        images with the checkpoint's SCUT running statistics on the way to the
        ``feature_map`` artifact. So the statistics mismatch this policy exists
        to test is only partly reachable from here, and a low result narrows
        the alternative rather than removing it. See
        ``ADABN_DIAGNOSTIC.does_not_test``.
        """
        import torch

        return [
            module
            for name, module in self._model.named_modules()
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
            and not name.startswith("backbone.")
        ]

    def adapt_batchnorm(self, features: np.ndarray) -> dict:
        """AdaBN: re-estimate BN running statistics on this fold's TRAIN rows.

        The remedy PLAN §4.7 records for these backbones, and the mechanism
        behind the void ladder's head-init defect. The probe otherwise
        normalises cleft activations with SCUT's statistics, which is a
        complete alternative explanation for its result.

        Three details are load-bearing, each measured before this was written
        (``MECHANISM_VERIFIED``) because each would leave the arm reporting
        success while changing nothing:

        * **reset first.** ``momentum=None`` gives a cumulative average
          weighted ``1/num_batches_tracked``, and the checkpoint arrives with
          that counter in the thousands -- measured, four batches then move
          the mean 0.14% of the way. Without the reset this method is a no-op
          with a flag on it.
        * **BN modules alone into train mode.** ``model.train()`` would also
          switch on the two ``Dropout(0.2)`` layers, so the statistics would
          describe activations that eval-mode inference never sees.
        * **no gradients, no optimiser.** Nothing here trains; the classifier
          is still the only thing fitted.

        The pass is UNSHUFFLED and runs under ``no_grad``, so the statistics
        are a deterministic function of the fold and of the weights producing
        the activations. **On a warm-started arm those weights are identical
        across seeds, so the adapted representation is too** -- worth knowing
        before any seed band is measured here, since like the warm-started
        graph layers it would contribute nothing to the spread. On an imagenet
        init the graph layers are seed-initialised and that does not hold;
        ``seed_independent`` in the returned record says which case applies
        rather than asserting the convenient one.
        """
        import torch

        modules = self._batch_norms()
        if not modules:
            # **[MEASURED 2026-08-01] This fires for AG-Net, and it is a
            # finding rather than a limitation.** All 53 of AG-Net's BatchNorm
            # modules live inside the frozen ResNet-50, which never runs on
            # this path -- it has NO normalisation after the frozen boundary.
            # So the statistics-mismatch alternative that motivates this arm
            # cannot arise for AG-Net at all, and running the policy there
            # would be an expensive no-op wearing a remedy's name.
            raise GraphCleftError(
                f"{self.name!r} has no BatchNorm modules after the frozen "
                "boundary, so there are no running statistics to re-estimate. "
                "AG-Net is the measured case: all 53 of its BN modules are "
                "inside the backbone, which never runs here (the feature map "
                "arrives precomputed). The adabn policy is SR-GNN-only by "
                "architecture -- and for AG-Net the normalisation-mismatch "
                "explanation this arm exists to test does not apply."
            )

        rows = len(features)
        # BatchNorm refuses a batch of 1 in train mode ("Expected more than 1
        # value per channel"). Caught here, before a fold dies part-way with a
        # stack trace from inside the model.
        if rows < 2:
            raise GraphCleftError(
                f"BN re-estimation needs at least 2 rows; this fold offers "
                f"{rows}. Batch statistics are undefined on one sample."
            )
        batches = [
            (start, min(start + self.batch_size, rows))
            for start in range(0, rows, self.batch_size)
        ]
        # A trailing batch of exactly one would raise inside the model, so it
        # is merged into its predecessor rather than dropped -- dropping it
        # would silently exclude a patient from the statistics.
        if len(batches) > 1 and batches[-1][1] - batches[-1][0] == 1:
            batches[-2] = (batches[-2][0], batches[-1][1])
            batches.pop()

        previous_momentum = []
        for module in modules:
            previous_momentum.append(module.momentum)
            module.reset_running_stats()
            module.momentum = None  # cumulative average over this fold

        self._model.eval()  # dropout OFF ...
        for module in modules:
            module.train()  # ... and BN alone accumulating

        with torch.no_grad():
            for start, stop in batches:
                maps_t, boxes_t = self._tensors(features[start:stop])
                self._forward(maps_t, boxes_t)

        # Freeze what was just measured: back to eval everywhere, and the
        # momentum restored so nothing later mistakes this for a live setting.
        self._model.eval()
        for module, momentum in zip(modules, previous_momentum):
            module.momentum = momentum

        record = {
            "n_batchnorm_modules": len(modules),
            "adapted": "post-backbone BatchNorms only -- the backbone's 40 "
                       "never run on this path (features are precomputed)",
            "n_rows": rows,
            "n_batches": len(batches),
            "batch_sizes": [stop - start for start, stop in batches],
            "batches_tracked": [int(m.num_batches_tracked) for m in modules],
            "source": "inner-train rows only (never inner-val or test)",
            # **Conditional, and the condition is load-bearing.** The pass is
            # unshuffled and gradient-free, so the statistics depend only on
            # the fold and the weights producing the activations. Those
            # weights are seed-independent ONLY because they are warm-started
            # from the checkpoint. On an imagenet init the graph layers are
            # seed-initialised, the activations differ by seed, and so do
            # these statistics -- measured, not assumed.
            "seed_independent": self.checkpoint_arrays is not None,
            "seed_independent_because": (
                "warm-started graph layers are identical across seeds, and the "
                "pass is unshuffled and gradient-free"
                if self.checkpoint_arrays is not None
                else "NOT seed-independent: imagenet init seed-initialises the "
                     "graph layers, so the activations and these statistics "
                     "vary with the seed"
            ),
        }
        self._adapted = record
        return record

    def train_epoch(self, features: np.ndarray, labels: np.ndarray) -> float:
        import torch

        # **Under the diagnostic policy the model TRAINS IN EVAL MODE, and
        # that is the substance of the policy rather than a detail.** SR-GNN
        # applies Dropout(0.2) twice and carries two BatchNorms; in train mode
        # the pre-classifier stack emits different activations every batch and
        # updates its running statistics, so the "frozen" representation would
        # move under the head fitting on it. That is not a probe, and calling
        # it one would make the comparison against the frozen ViT probe's
        # 0.2529 a comparison between two different things (PLAN R2). Eval
        # mode makes the frozen stack a fixed function of the map, which is
        # the property the comparison rests on.
        #
        # It costs nothing for the head: the classifier is a bare Linear, and
        # train/eval do not touch it.
        #
        # **AdaBN happens HERE, and the placement is the check.** The harness
        # calls predict() twice before this method is ever reached -- on the
        # INNER-VAL rows for gate 3, then on the test rows. Adapting on the
        # first forward of any kind would therefore compute the representation
        # from the data early stopping is judged on, or from the test fold.
        # Binding it to train_epoch, which receives inner-train only, is what
        # keeps the statistics on the right side of the split.
        #
        # Once per fold: nothing upstream of the classifier trains, so a
        # second pass would recompute identical values.
        # Every epoch, not once: cheap, and it is the only check that can see
        # this instance's fold drifting from the harness's.
        self._assert_not_training_on_test_rows(features)

        if self.trainable == "classifier_adabn" and self._adapted is None:
            self.adapt_batchnorm(features)
            # ``run`` reads parameter_report off the LAST fold's instance, so
            # this describes that fold rather than all five. Said in the key
            # name: a record silently scoped to one fold, read as the run's,
            # is a small version of the quantity confusion R2 exists for.
            # (The counts differ per fold: inner-train sizes are not equal.)
            self.parameter_report["bn_reestimation_last_fold"] = self._adapted

        self._model.eval() if self.trainable in PROBE_POLICIES else self._model.train()
        self._epoch += 1
        # Deterministic given (seed, epoch), different across epochs -- the
        # same-permutation-every-epoch quirk of the Phase 3 torch backbone is
        # deliberately not inherited.
        order = np.random.default_rng(
            self.seed * 100_003 + self._epoch
        ).permutation(len(labels))

        device = next(self._model.parameters()).device
        total, seen = 0.0, 0
        for start in range(0, len(order), self.batch_size):
            index = order[start : start + self.batch_size]
            maps_t, boxes_t = self._tensors(features[index])
            target = torch.as_tensor(
                np.asarray(labels[index], dtype=np.float32), device=device
            )
            self._optimizer.zero_grad(set_to_none=True)
            prediction = self._forward(maps_t, boxes_t).squeeze(-1)
            loss = torch.nn.functional.mse_loss(prediction, target)
            loss.backward()
            self._optimizer.step()
            total += float(loss.item()) * len(index)
            seen += len(index)
        return total / max(seen, 1)

    def predict(self, features: np.ndarray) -> np.ndarray:
        import torch

        self._model.eval()
        out: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(features), self.batch_size):
                maps_t, boxes_t = self._tensors(
                    features[start : start + self.batch_size]
                )
                out.append(
                    self._forward(maps_t, boxes_t).squeeze(-1).float().cpu().numpy()
                )
        return np.concatenate(out) if out else np.empty(0)


@dataclass
class IndexedStubBackbone:
    """The stub over PER-FOLD features. Torch-free, so the laptop runs it.

    Without this the per-fold path would have no laptop coverage at all: the
    harness rows are indices, and ``StubBackbone`` would happily fit a linear
    head to the index column and produce a curve -- a run that exercises the
    gather, the fold selection and the leak check not at all, while looking
    exactly like one that did.

    It resolves indices against the fold's own array, and it runs the same
    row-level leak assertion the torch backbone does.
    """

    maps_by_fold: dict
    fold: int
    n_regions: int
    test_rows: frozenset | None = None
    learning_rate: float = 0.15
    parameter_report: dict = field(default_factory=dict)
    _inner: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        from .stub import StubBackbone

        self._inner = StubBackbone(learning_rate=self.learning_rate)

    def _resolve(self, features: np.ndarray) -> np.ndarray:
        indices, boxes = unpack_indexed(features, self.n_regions)
        maps = self.maps_by_fold[self.fold][indices]
        flat = np.asarray(maps, dtype=float).reshape(len(indices), -1)
        if boxes is None:
            return flat
        return np.concatenate([flat, boxes.reshape(len(indices), -1)], axis=1)

    def reset(self, train_labels: np.ndarray) -> None:
        self._inner.reset(train_labels)

    def train_epoch(self, features: np.ndarray, labels: np.ndarray) -> float:
        if self.test_rows is not None:
            indices, _ = unpack_indexed(features, self.n_regions)
            offending = sorted(set(int(i) for i in indices) & self.test_rows)
            if offending:
                raise GraphCleftError(
                    f"fold {self.fold}: {len(offending)} training row(s) "
                    f"belong to this fold's TEST set, e.g. {offending[:5]}"
                )
        loss = self._inner.train_epoch(self._resolve(features), labels)
        self.parameter_report = dict(self._inner.parameter_report)
        return loss

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._inner.predict(self._resolve(features))


# --------------------------------------------------------------------------
# boxes per scheme, per patient
# --------------------------------------------------------------------------


def content_boxes_from_geometry(rows: list[dict], patient_ids: list[int]) -> np.ndarray:
    """(N, 4) recorded content boxes, aligned to the manifest's patient order."""
    by_id: dict[int, list[int]] = {}
    for row in rows:
        try:
            by_id[int(row["patient_id"])] = [
                int(row["content_x"]), int(row["content_y"]),
                int(row["content_w"]), int(row["content_h"]),
            ]
        except KeyError as exc:
            raise GraphCleftError(
                f"geometry.csv is missing {exc.args[0]!r}; the staged artifact "
                "must record the content box per patient for the per-image "
                "patch mapping"
            ) from exc
    missing = [pid for pid in patient_ids if pid not in by_id]
    if missing:
        raise GraphCleftError(
            f"{len(missing)} patient(s) have no geometry row, e.g. {missing[:5]}"
        )
    return np.array([by_id[pid] for pid in patient_ids], dtype=np.int64)


def native_agnet_boxes(images: np.ndarray) -> np.ndarray:
    """SIFT+GMM regions per staged cleft image, padded to one rectangle.

    Deterministic (seeded GMM) and computed ONCE per patient. Variable region
    counts are padded with whole-image rows -- the identical semantics
    ``pool_regions_spatial`` applies within a batch, made global so the boxes
    pack rectangularly.
    """
    from ..models.agnet import generate_srs

    per_image = [generate_srs(np.asarray(image)[:, :, ::-1]) for image in images]
    widest = max(len(boxes) for boxes in per_image)
    out = np.tile(
        np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32), (len(per_image), widest, 1)
    )
    for index, boxes in enumerate(per_image):
        out[index, : len(boxes)] = boxes
    return out


def boxes_for_arm(
    *,
    backbone: str,
    region_scheme: str,
    geometry: str,
    geometry_rows: list[dict] | None,
    patient_ids: list[int],
    staged_images: np.ndarray | None,
) -> np.ndarray | None:
    """The (N, R, 4) boxes one arm trains under, in the backbone's convention.

    None means the model's own default (SR-GNN's grid buffer). The stub is
    native-only: it has no box convention, and giving it one would exercise
    nothing.
    """
    if region_scheme not in CLEFT_SCHEMES:
        raise GraphCleftError(
            f"unknown region_scheme {region_scheme!r}; expected one of "
            f"{CLEFT_SCHEMES}"
        )
    if backbone == "stub_graph":
        if region_scheme != "native":
            raise GraphCleftError(
                "the stub arm is native-only: it has no box convention, and a "
                "generated scheme would be packed and then ignored"
            )
        return None

    if region_scheme == "native":
        if backbone == "srgnn":
            return None  # the model's own grid buffer
        if staged_images is None:
            raise GraphCleftError(
                "AG-Net's native scheme is SIFT+GMM over the staged images, "
                "which were not supplied"
            )
        return native_agnet_boxes(staged_images)

    if geometry_rows is None:
        raise GraphCleftError(
            f"the {region_scheme!r} scheme places patches through each "
            "patient's recorded content box; geometry rows were not supplied"
        )
    content = content_boxes_from_geometry(geometry_rows, patient_ids)
    frame = scheme_frame_boxes(region_scheme, geometry, content)
    return boxes_for_model(backbone, frame)


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


def check_artifact_pairing(
    metadata: dict,
    *,
    backbone: str,
    init: str,
    geometry: str,
    region_scheme: str,
    checkpoint_sha256: str | None,
) -> None:
    """The artifact must BE the arm's declared representation, exactly.

    **The rules live in ``embeddings.check_pairing``, not here.** The
    transformer ladder consumes pooled sets through the identical set of
    checks, differing only in the kind it demands, so a second copy of this
    logic would be one place for the scheme rule or the hash comparison to
    drift -- the ``is_absolute_path`` shape the tally already records twice.
    The error type is translated so this module's callers and tests keep
    meeting ``GraphCleftError``.
    """
    try:
        embeddings.check_pairing(
            metadata,
            kind="feature_map",
            backbone=backbone,
            init=init,
            geometry=geometry,
            checkpoint_sha256=checkpoint_sha256,
            region_scheme=region_scheme,
        )
    except embeddings.EmbeddingError as exc:
        raise GraphCleftError(str(exc)) from exc


def load_labels_and_folds(
    manifest_dir: Path, label: str
) -> tuple[list[int], np.ndarray, dict[int, int]]:
    """patient_ids (manifest order), labels, fold assignments."""
    from ..data.manifest import load_manifest

    rows = load_manifest(Path(manifest_dir) / "manifest.csv")
    if not rows:
        raise GraphCleftError("the manifest is empty")
    if label not in rows[0]:
        raise GraphCleftError(f"the manifest has no column {label!r}")
    patient_ids = [int(row["patient_id"]) for row in rows]
    labels = np.array([float(row[label]) for row in rows])
    assignments = {int(row["patient_id"]): int(row["fold"]) for row in rows}
    return patient_ids, labels, assignments


def run(
    *,
    manifest_dir: Path,
    staged_dir: Path,
    embeddings_dir: Path | None = None,
    #: PER-FOLD artifact: ``{fold: directory}``, one set per fold, each
    #: adapted on that fold's TRAINING patients only
    #: (``extract.PER_FOLD_REEXTRACTION``). Mutually exclusive with
    #: ``embeddings_dir``.
    embeddings_dirs: dict | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_sha256: str | None = None,
    backbone: str = "srgnn",
    init: str = "scut_masked",
    geometry: str = "g2",
    region_scheme: str = "native",
    label: str = "mean",
    trainable: str = "graph_layers",
    #: Whether ``use_deterministic_algorithms`` stays on. **AG-Net must state
    #: it explicitly** -- None is refused there, because the answer is a
    #: measured property of the image rather than a default anyone should
    #: inherit. Every other backbone is bitwise under the flag and leaves it
    #: alone.
    deterministic: bool | None = None,
    seed: int = 1337,
    train_config: TrainConfig | None = None,
    backbone_config: dict | None = None,
    #: **[ADDED 2026-08-14] The only channel out for the TRAINED fold
    #: models.** Phase 8 arm B reads each patient's node weights off the
    #: fold-model that held them out, and ``Phase3Result`` carries metrics
    #: rather than models. Called once, AFTER ``run_cv`` returns and after
    #: the fold-count check -- before that the instances exist but are not
    #: trained, and an explanation read off an untrained model would be an
    #: explanation of nothing. Signature:
    #: ``(models=[(fold, instance), ...], packed=..., patient_ids=...,
    #: assignments=...)``; ``packed`` is handed over so the caller cannot
    #: pack the rows a second way.
    on_fold_models=None,
    log=print,
) -> Phase3Result:
    """One graph cleft arm: 5-fold CV through the frozen harness."""
    if backbone not in GRAPH_CLEFT_BACKBONES:
        raise GraphCleftError(
            f"unknown backbone {backbone!r}; expected one of "
            f"{GRAPH_CLEFT_BACKBONES}"
        )
    if trainable not in TRAINABLE_POLICIES:
        raise GraphCleftError(
            f"unknown trainable policy {trainable!r}; expected one of "
            f"{TRAINABLE_POLICIES}"
        )
    # **AG-Net must SAY what it is doing about determinism.** Defaulting it
    # either way is the failure: default-on and the arm dies mid-run on an op
    # nobody expected; default-off and its results silently carry a ~2.3e-05
    # tolerance nobody declared. The config states it, and metrics.json
    # records both the setting and the bound.
    if backbone == "agnet" and deterministic is None:
        raise GraphCleftError(
            "an AG-Net arm must declare `deterministic` explicitly. "
            "[MEASURED, models.agnet.TRAINING_DETERMINISM] the deterministic "
            "roi_align substitution cannot be satisfied in the pinned image "
            "(no C++ compiler for inductor), so `deterministic: true` will "
            "raise mid-run, and `deterministic: false` means this arm's "
            "same-seed reproducibility is bounded at ~2.3e-05 rather than "
            "bitwise -- which every comparison against it has to carry. "
            "Neither is a sensible default; say which."
        )

    if trainable in PROBE_POLICIES and backbone == "stub_graph":
        # The stub is a linear head on flattened features under EITHER policy,
        # so it cannot distinguish them -- a stub run reporting
        # "trainable: classifier" would be evidence about nothing while
        # looking like evidence about the diagnostic. Same reasoning as the
        # native-only rule above, and the same failure class the tally tracks:
        # a check that reports success on its own failure mode. The torch half
        # is verified by scripts/verify_backbone_builds.py.
        raise GraphCleftError(
            "the stub arm cannot run the classifier-only diagnostics: it is a "
            "linear head under every policy and has no BatchNorm to "
            "re-estimate, so it would report a policy it does not implement. "
            "Verify these on a real graph backbone "
            "(scripts/verify_backbone_builds.py)."
        )
    # **[DECIDED 2026-08-01] The determinism opt-out, and why it exists.**
    #
    # This path had no knob because determinism was assumed achievable.
    # Phase 6 MEASURED that it is not, for AG-Net, in the pinned image:
    # torchvision's deterministic ``roi_align`` substitution is gated on
    # torch.compile -> inductor -> a C++ compiler the image does not have
    # (``Failed to find C compiler``), and the flag never raises on that op
    # anyway, so gate 1's raised-on-nothing canary is blind to it
    # (``models.agnet.TRAINING_DETERMINISM``). The assumption is false, so the
    # knob is honest rather than permissive -- and it is already the settled
    # decision for AG-Net PRETRAINING, where all thirty runs ran
    # ``deterministic: false``. Excluding AG-Net from the cleft ladder for a
    # toolchain reason would lose one of the two architectures the project
    # exists to compare, on grounds that have nothing to do with the science.
    #
    # **``configure`` stays the single owner of the setup.** It is called
    # unconditionally, because it seeds python, numpy and torch and exports
    # ``CUBLAS_WORKSPACE_CONFIG`` -- and reimplementing any of that here is
    # exactly the CuBLAS defect (two copies of one setup, every backbone
    # raising in the image). Only the ONE setting that cannot be satisfied is
    # then relaxed, and the record is corrected so it does not claim a flag
    # that is off.
    relaxation = None
    determinism_record = determinism.configure(
        seed, require_torch=(backbone != "stub_graph")
    )
    if deterministic is False:
        # Only if torch is actually present. `configure` records that, and on
        # the torch-free stub arm there is no flag to turn off -- but the
        # DECLARATION is still recorded, because what the config asked for is
        # part of the run's provenance whether or not it had an effect here.
        if determinism_record["torch"]:
            import torch

            torch.use_deterministic_algorithms(False)
        from ..models.agnet import TRAINING_DETERMINISM

        relaxation = {
            "requested": False,
            "why": (
                "the deterministic roi_align substitution needs a C++ compiler "
                "the pinned image lacks; measured, not assumed"
            ),
            "measured_tolerance": TRAINING_DETERMINISM[
                "flag_off_post_restore_step_max_diff"
            ],
            "mechanism": TRAINING_DETERMINISM["mechanism"],
            "how_to_compare": (
                "two runs of this arm agree to ~2.3e-05, NOT bitwise. Compare "
                "at that tolerance; an equality check will fail and mean "
                "nothing. A delta between AG-Net arms needs its own seed band "
                "measured under this condition, with kernel noise as the floor."
            ),
            "seeding_unaffected": (
                "determinism.configure still ran: python, numpy and torch are "
                "seeded and CUBLAS_WORKSPACE_CONFIG is exported. Only "
                "use_deterministic_algorithms was turned back off."
            ),
        }
        determinism_record = {
            **determinism_record,
            "torch_deterministic_algorithms": False,
            "relaxed": relaxation,
        }
    log(f"determinism: {determinism_record}")

    if (embeddings_dir is None) == (embeddings_dirs is None):
        raise GraphCleftError(
            "declare exactly one of embeddings_dir (one set, valid for every "
            "fold) or embeddings_dirs (one set PER FOLD, each adapted on that "
            "fold's training patients). Both, or neither, leaves it ambiguous "
            "which representation the arm ran on."
        )

    patient_ids, labels, assignments = load_labels_and_folds(manifest_dir, label)
    row_of = {pid: i for i, pid in enumerate(patient_ids)}

    maps_by_fold: dict | None = None
    leak_reports: dict = {}
    if embeddings_dirs is not None:
        folds = sorted(set(assignments.values()))
        missing = [f for f in folds if f not in embeddings_dirs]
        if missing:
            raise GraphCleftError(
                f"the manifest has folds {folds} but no embedding set was "
                f"declared for {missing}. A missing fold's arm would silently "
                "reuse another fold's representation."
            )
        maps_by_fold, metadata = {}, None
        for fold in folds:
            fold_maps, fold_metadata = embeddings.load(
                embeddings_dirs[fold], manifest_ids=patient_ids
            )
            check_artifact_pairing(
                fold_metadata,
                backbone=backbone, init=init, geometry=geometry,
                region_scheme=region_scheme, checkpoint_sha256=checkpoint_sha256,
            )
            if fold_metadata.get("fold") != fold:
                raise GraphCleftError(
                    f"the set declared for fold {fold} records fold "
                    f"{fold_metadata.get('fold')!r}. Using a set built for a "
                    "different fold puts that fold's test patients inside this "
                    "one's representation."
                )
            # **THE LEAK CHECK.** Its test patients must not have contributed
            # to its own BatchNorm statistics.
            test_ids = [pid for pid in patient_ids if assignments[pid] == fold]
            leak_reports[str(fold)] = embeddings.assert_no_test_fold_leak(
                fold_metadata, test_ids
            )
            maps_by_fold[fold] = fold_maps
            metadata = fold_metadata
        maps = maps_by_fold[folds[0]]
        log(
            f"{len(patient_ids)} patients; PER-FOLD maps {maps.shape} x "
            f"{len(folds)} folds, test-fold leak checked on every one"
        )
    else:
        maps, metadata = embeddings.load(embeddings_dir, manifest_ids=patient_ids)
        check_artifact_pairing(
            metadata,
            backbone=backbone, init=init, geometry=geometry,
            region_scheme=region_scheme, checkpoint_sha256=checkpoint_sha256,
        )
        log(
            f"{len(patient_ids)} patients; maps {maps.shape} from "
            f"{metadata['init']}/{metadata['geometry']}"
            + (
                f"/{metadata['pretrain_scheme']}"
                if metadata.get("pretrain_scheme")
                else ""
            )
        )

    checkpoint_arrays = None
    if init != "imagenet" and backbone != "stub_graph":
        checkpoint_arrays = ckpt.load(checkpoint_path).arrays

    needs_boxes = region_scheme in GENERATED_SCHEMES or (
        backbone == "agnet" and region_scheme == "native"
    )
    geometry_rows = None
    staged_images = None
    if needs_boxes:
        if region_scheme in GENERATED_SCHEMES:
            from .phase3 import load_geometry_rows

            geometry_rows = load_geometry_rows(Path(staged_dir))
        else:
            staged = np.load(
                Path(staged_dir) / f"staged_patient_{geometry}.npy", mmap_mode="r"
            )
            staged_images = np.array(staged)
    boxes = boxes_for_arm(
        backbone=backbone, region_scheme=region_scheme, geometry=geometry,
        geometry_rows=geometry_rows, patient_ids=patient_ids,
        staged_images=staged_images,
    )
    n_regions = 0 if boxes is None else int(boxes.shape[1])
    map_shape = tuple(int(v) for v in maps.shape[1:])
    if maps_by_fold is None:
        packed = pack(maps, boxes)
    else:
        packed = pack_indexed(len(patient_ids), boxes)
    log(f"packed rows: {packed.shape} (map {map_shape}, {n_regions} boxes)")

    backbone_config = backbone_config or {}
    config = train_config or TrainConfig(seed=seed)

    trained: list = []
    #: ``run_cv`` calls ``make_backbone`` exactly once per fold, in
    #: ``sorted(folds)`` order (harness.run_cv's list comprehension over
    #: run_fold, each of which constructs one backbone). That is the only
    #: channel the frozen harness offers for telling an instance which fold it
    #: is serving -- and it is an ORDERING assumption, so it is not trusted:
    #: every ``train_epoch`` asserts that none of its rows belongs to this
    #: instance's test fold, which is exactly what a miscount would violate.
    fold_order = sorted(set(assignments.values()))
    fold_calls = {"n": 0}

    def make_backbone():
        # **[CHANGED 2026-08-14] The fold is derived on BOTH paths.**
        #
        # It used to be derived only for the per-fold artifact, leaving the
        # single-artifact path with no fold at all -- so an explanation
        # produced by one of these instances could not be attributed to the
        # patients it held out (phase8's arm B needs exactly that, and the
        # held-out rule is why: explaining a patient with a model that
        # trained on them explains a memorised label).
        #
        # Deriving it here removes the implicit mapping rather than routing
        # around it, and the count check below fires on both paths now.
        #
        # **What this does NOT do, stated because the difference matters.**
        # The per-row leak assertion in ``assert_no_test_rows`` reads a row
        # INDEX out of the packed features, and only ``pack_indexed`` (the
        # per-fold layout) writes one -- ``pack`` writes flattened maps with
        # no patient identity in the row. So on the single-artifact path the
        # fold is derived and the ordering is still an ordering assumption;
        # what changed is that it is now explicit, counted, and available to
        # attribute weights, instead of being absent.
        if fold_calls["n"] >= len(fold_order):
            raise GraphCleftError(
                f"the harness asked for backbone {fold_calls['n'] + 1} but "
                f"only {len(fold_order)} folds exist. Either run_cv changed "
                "how many models it builds, or the fold set moved -- and "
                "every explanation attributed by call order would be wrong."
            )
        this_fold = fold_order[fold_calls["n"]]
        fold_calls["n"] += 1
        test_rows = None
        if maps_by_fold is not None:
            test_rows = frozenset(
                row_of[pid] for pid in patient_ids if assignments[pid] == this_fold
            )

        if backbone == "stub_graph":
            from .stub import StubBackbone

            instance = (
                StubBackbone()
                if maps_by_fold is None
                else IndexedStubBackbone(
                    maps_by_fold=maps_by_fold,
                    fold=this_fold,
                    n_regions=n_regions,
                    test_rows=test_rows,
                )
            )
        else:
            instance = GraphHeadBackbone(
                name=backbone,
                map_shape=map_shape,
                n_regions=n_regions,
                checkpoint_arrays=checkpoint_arrays,
                trainable=trainable,
                learning_rate=backbone_config.get("learning_rate", 1e-4),
                weight_decay=backbone_config.get("weight_decay", 0.01),
                batch_size=backbone_config.get("batch_size", 16),
                seed=seed,
                maps_by_fold=maps_by_fold,
                fold=this_fold,
                test_rows=test_rows,
            )
        trained.append(instance)
        return instance

    cv = run_cv(
        features=packed,
        labels=labels,
        patient_ids=patient_ids,
        assignments=assignments,
        make_backbone=make_backbone,
        config=config,
        log=log,
    )

    # **[ADDED 2026-08-14] The count check, on both paths.** ``trained[i]``
    # is claimed to be ``fold_order[i]``'s model, and an explanation
    # attributed by that claim is wrong if the harness built a different
    # number of models than there are folds. Cheap, sound, and it fires
    # before anything is attributed rather than after.
    if len(trained) != len(fold_order):
        raise GraphCleftError(
            f"{len(trained)} backbones were built for {len(fold_order)} "
            "folds. trained[i] is taken to be fold_order[i]'s model, so "
            "any per-fold explanation read off this run would be "
            "attributed to the wrong patients."
        )

    # Handed over only now: the instances are trained by run_cv, and the
    # zip above is total only because the count check just passed.
    if on_fold_models is not None:
        on_fold_models(
            models=list(zip(fold_order, trained)),
            packed=packed,
            patient_ids=patient_ids,
            assignments=assignments,
            # Handed over so a caller substituting a DIFFERENT map -- arm B's
            # randomised-backbone control -- packs it through the same boxes
            # and the same layout this run trained on. Re-deriving them would
            # be a second definition of the arm's node structure.
            boxes=boxes,
            map_shape=map_shape,
        )

    head_report = getattr(trained[-1], "parameter_report", {}) if trained else {}
    gate_reports = {
        "gate5_metric_verification": gates.verify_metrics(seed=seed),
        "gate6_oof_reconstruction": gates.reconstruct_oof(
            cv.fold_record(),
            expected_patients=set(patient_ids),
            expected_assignments=assignments,
            oof_ids=cv.oof_ids,
        ),
    }
    log("gates 5 and 6 passed")

    diagnostic = trainable in PROBE_POLICIES
    adabn = trainable == "classifier_adabn"
    summary = {
        "regime": "graph_probe_diagnostic" if diagnostic else "graph_third",
        "regime_note": (
            (
                "DIAGNOSTIC, NOT A LADDER ARM: the graph layers are frozen "
                "TOO, so this is an SR-GNN linear probe, with its "
                "POST-BOUNDARY BatchNorm running statistics RE-ESTIMATED on "
                "each training fold. It tests part of the alternative the "
                "plain frozen probe left standing -- that its 0.1340 was a "
                "normalisation-statistics mismatch rather than the "
                "representation. Stated before the run: ~0.25 is CONCLUSIVE "
                "against the architecture reading; ~0.134 NARROWS the "
                "alternative WITHOUT removing it, because this arm reaches 2 "
                "of SR-GNN's 42 BatchNorms and the other 40 normalised the "
                "cleft images with SCUT statistics during EXTRACTION -- baked "
                "into the artifact, unreachable at cleft time. Closing it "
                "fully needs a re-extraction, not another cleft arm. See "
                "graph_cleft.ADABN_DIAGNOSTIC.does_not_test."
                if adabn
                else
                "DIAGNOSTIC, NOT A LADDER ARM: the graph layers are frozen "
                "TOO, so this is an SR-GNN linear probe -- the brief's own "
                "trap run deliberately once, because the reduced arm is the "
                "only thing directly comparable to the frozen ViT probe's "
                "0.2529. Read the outcomes asymmetrically: ~0.25 is "
                "conclusive for the configuration reading; ~0.15 is "
                "consistent with the architecture reading but does not "
                "establish it, because this arm uses the checkpoint's SCUT "
                "BatchNorm statistics on cleft inputs while the trained arm "
                "adapts them, and AdaBN is the documented remedy (PLAN §4.7). "
                "See graph_cleft.FROZEN_GRAPH_DIAGNOSTIC."
            )
            if diagnostic
            else (
                "frozen backbone feature maps in, TRAINED graph layers -- the "
                "third regime (brief §2.3). Phase 3's seed band does not cover "
                "it; this regime's own band is a gate before any graph delta."
            )
        ),
        "trainable": trainable,
        **(
            {
                "diagnostic": ADABN_DIAGNOSTIC if adabn else FROZEN_GRAPH_DIAGNOSTIC,
                "is_a_ladder_arm": False,
                **({"bn_mechanism": MECHANISM_VERIFIED} if adabn else {}),
            }
            if diagnostic
            else {"seed_band": MEASURED_GRAPH_SEED_BAND}
        ),
        "n_patients": len(patient_ids),
        "backbone": backbone,
        "init": init,
        "geometry": geometry,
        "region_scheme": region_scheme,
        "label": label,
        "seed": seed,
        "fingerprint": determinism_fingerprint(cv),
        "determinism": determinism_record,
        # **The setting AND the bound, in the SHAREABLE record.** A reader
        # comparing two runs of this arm needs to know that equality is the
        # wrong test before they run the comparison, not after it fails.
        "determinism_policy": {
            "declared": deterministic,
            "use_deterministic_algorithms": determinism_record[
                "torch_deterministic_algorithms"
            ],
            # **Three states, not two.** The flag being off does not by
            # itself mean a relaxation: on the torch-free stub arm
            # `configure` never reaches the torch settings at all. Reporting
            # that as "tolerance 2.3e-05" would attach AG-Net's measured
            # bound to a run that has nothing to do with it -- a number
            # correct about one thing and recorded against another (R2).
            "comparison": (
                f"tolerance {relaxation['measured_tolerance']:.2e}"
                if relaxation is not None
                else "bitwise"
                if determinism_record["torch_deterministic_algorithms"]
                else "not applicable: torch absent, so the flag was never set"
            ),
            **({"relaxed": relaxation} if relaxation else {}),
        },
        "features": {
            "features": "frozen_backbone_feature_maps",
            "artifact_kind": metadata["kind"],
            "map_shape": list(map_shape),
            "n_regions_packed": n_regions,
            "pretrain_scheme": metadata.get("pretrain_scheme"),
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
            **(
                {
                    "per_fold_bn_reextraction": True,
                    "n_fold_sets": len(maps_by_fold),
                    "bn_reestimation": metadata.get("bn_reestimation"),
                    # The leak check's verdict per fold, in the SHAREABLE
                    # record -- a check whose result is not written down is a
                    # check nobody can confirm ran.
                    "test_fold_leak_check": leak_reports,
                }
                if maps_by_fold is not None
                else {"per_fold_bn_reextraction": False}
            ),
        },
        "parameters": head_report,
        "train_config": {
            "max_epochs": config.max_epochs,
            "patience": config.patience,
            "inner_val_frac": config.inner_val_frac,
            "monitor": config.monitor,
            "seed": config.seed,
            "optimizer": "AdamW" if backbone != "stub_graph" else "sgd_stub",
            "learning_rate": backbone_config.get("learning_rate", 1e-4),
            "weight_decay": backbone_config.get("weight_decay", 0.01),
            "batch_size": backbone_config.get("batch_size", 16),
            "trainable": trainable,
            "eval_mode_while_training": diagnostic,
            "bn_reestimation": adabn,
            "loss": "mse",
            "label_scale": "raw_1_to_5",
        },
        "sanity": sanity_report(cv),
        "oof": cv.metrics(),
        "gates": gate_reports,
    }
    return Phase3Result(cv=cv, summary=summary, gate_reports=gate_reports)
