"""Arm B's node weights, and the gate that must pass before any are shown.

Phase 8's second explanation. Arm B is ``p7_d1_srgnn_imagenet_g1_native``
and its native account of itself is which regions its attention pooling
weighted -- "regions voting", which is what supervision asked for.

**26 regions, not 27.** ``srgnn.N_REGIONS`` is ``N_ROIS + 1``: the whole
feature map is appended as the last node before attention, so the weight
vector's final entry is not a region at all. The similarity is computed
over the 26 ROI weights and the whole-image weight is reported beside
them (``phase8.SRGNN_NODE_WEIGHTS_R10_READ``) -- a shared near-constant
element would inflate the between-patient baseline, and that baseline is
the bar a randomised vector must fall BELOW, so including it would make
the gate more permissive for a reason unrelated to the explanation.

----------------------------------------------------------------------------
THE STATISTICS ARE GRAD-CAM'S, DELIBERATELY
----------------------------------------------------------------------------
``gradcam.similarity``, ``between_patient_baseline``,
``between_patient_distribution`` and ``separability`` are generic over
vector length -- they ravel their inputs and need only matching shapes.
So this module REUSES them rather than defining a parallel set, and the
two arms' numbers are produced by one implementation. A second Spearman
would be the duplication R10 exists to prevent, and worse here than
usual: the whole point is comparing the arms, and a comparison across two
implementations of "the same" statistic is not one.

**What is NOT reused is the verdict.** ``gradcam.randomisation_verdict``
implements the median pass/fail rule, and
``phase8.MEDIAN_CUT_WAS_NOT_DEFENSIBLE`` established that rule was never
sound. Under ``phase8.RANDOMISATION_CRITERION_ADOPTED`` the criterion is
the phase-level separability interval and per-patient figures are
PERCENTILES. So this module computes those instead, and the old verdict
function is left where it is, serving the run that already used it.

----------------------------------------------------------------------------
TWO RANDOMISATIONS, AND WHY
----------------------------------------------------------------------------
``phase8.ARM_A_AND_B_RANDOMISATIONS_ARE_NOT_EQUIVALENT``: arm A randomised
its frozen backbone, which for it was also nearly the whole model and the
source of its map. For arm B those come apart, so both are run:

* ``component_role`` -- randomise the frozen xception. Asks arm A's
  question, and is the ONLY mode the A<->B comparison uses.
* ``explanation_source`` -- randomise the trained graph layers, which is
  where arm B's weights actually come from.

Their agreement or disagreement is itself informative, and both readings
are registered in ``phase8`` before any number exists.

**[MEASURED 2026-08-14] component_role cannot be run as a parameter
randomisation on this arm and is refused.** Randomising all 20,806,952
backbone parameters leaves ``region_w`` bitwise identical: the artifact-fed
path enters below the backbone, so the frozen representation is an INPUT
here rather than a computation (``phase8.COMPONENT_ROLE_IS_A_NO_OP``). The
mode names are kept -- the design distinction is still the right one -- and
what changed is that asking arm A's question of arm B needs the map
re-extracted from a randomised backbone, which is a new artifact rather than
a walk over this one's parameters. ``explanation_source`` is unaffected.
"""

from __future__ import annotations

import numpy as np

from . import gradcam


class NodeWeightError(ValueError):
    """A node-weight vector or gate input is not usable."""


#: The two randomisations, and the one the comparison may use.
RANDOMISATION_MODES = ("component_role", "explanation_source")
COMPARISON_MODE = "component_role"

#: **The two modes reach their randomisation by two different routes**, and
#: the split is measured rather than stylistic
#: (``phase8.COMPONENT_ROLE_IS_A_NO_OP``).
#:
#: ``explanation_source`` walks the model's own parameters at cleft time --
#: the graph layers that produce ``region_w`` are right there and trained.
#:
#: ``component_role`` cannot: the frozen representation is an INPUT on this
#: path, so randomising it means re-extracting the feature map from a
#: randomised backbone and feeding THAT to the same trained layers
#: (``extract.RANDOMISED_BACKBONE``). Same intervention as arm A's, moved to
#: where arm B's representation is computed.
PARAMETER_WALK_MODES = ("explanation_source",)
CONTROL_SET_MODES = ("component_role",)


def noise_std(parameter, group_std: float) -> float:
    """The std the randomising noise is drawn at, for one parameter tensor.

    **[MEASURED 2026-08-14, torch 2.13.0 CPU] ``parameter.std()`` returns NaN
    for a single-element tensor** -- the default correction is 1, so the
    denominator is zero -- and ``normal_`` then raises ``normal expects std
    >= 0.0, but found std -nan``. SR-GNN has three such parameters
    (``phase8.SCALAR_PARAMETERS_BREAK_THE_NOISE_MATCH``) and one of them sits
    in the first group ``explanation_source`` randomises, so the walk would
    have died on the cluster after the ten-seed refit.

    A tensor with no spread of its own (one element, or a constant) borrows
    the GROUP's spread rather than being skipped: skipping would leave a
    trained parameter in place inside a stage that claims to have randomised
    everything above it, which is a check quietly doing less than it says.

    **One definition, two call sites** -- the parameter walk
    (``run._phase8_walk_stages``) and the extraction-time randomisation
    (``train.extract.randomise_backbone``). They differ in WHAT they touch;
    the noise rule is shared so they cannot differ in anything else. Torch is
    not imported here: the arithmetic is the same for any object exposing
    ``detach``/``std``/``shape``, which is also what lets the laptop test it.
    """
    own = float(parameter.detach().std(unbiased=False))
    if own > 0.0:
        return own
    if group_std > 0.0:
        return group_std
    raise NodeWeightError(
        f"a parameter of shape {tuple(parameter.shape)} has no spread and "
        "neither does its group, so randomising it would change nothing. A "
        "stage that randomises nothing reports a test it did not run"
    )


def degeneracy(vectors) -> dict:
    """Are these weight vectors CONSTANT across regions?

    **[REGISTERED 2026-08-14, before the randomised set exists]
    ``gradcam.similarity`` returns 0.0 when either vector has no rank
    structure** -- which is right for a rank statistic and wrong to read as a
    measurement. A randomised vector that is uniform over the 26 regions
    scores 0.0 against every patient, falls below any positive baseline, and
    the gate RESOLVES. That is a pass produced by the vector having no
    explanation at all rather than by its ranks disagreeing.

    It is not a hypothetical: ``phase8.THE_RANDOMISED_BACKBONE_COLLAPSES``
    measures the randomised map as constant across patients to 5.96e-08, and
    with an untrained head the resulting weights are uniform to float32.
    Whether arm B's TRAINED attention amplifies the residual differences
    enough to keep rank structure cannot be known before it runs.

    So the fact is measured and reported beside the gate. **The reading is
    registered here rather than composed afterwards** -- a uniform vector is
    a different statement from "the ranks disagree", and the two must not be
    reported under one sentence.

    **[CORRECTED 2026-08-15, after the run.]** This docstring's registered
    reading originally continued: "a uniform randomised attention IS a
    strong form of parameter dependence -- the model has no explanation once
    the representation is destroyed". The maintainer queried it and the check
    refuted it: both attention layers are permutation-symmetric over the
    region axis, so indistinguishable descriptors give uniform weights for
    EVERY parameter value -- measured across five independent parameter
    draws, sd exactly 0.0 in each. A result every parameter setting produces
    carries no information about parameters. The collapse is UNINFORMATIVE,
    not evidence (``phase8.CHECK_1_UNIFORM_CONTROL_IS_ARITHMETIC``).
    """
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 2:
        raise NodeWeightError(
            f"weight vectors must be (n_patients, n_regions); got {array.shape}"
        )
    constant = np.std(array, axis=1) == 0.0
    n_constant = int(np.count_nonzero(constant))
    return {
        "n_patients": int(array.shape[0]),
        "n_constant": n_constant,
        "fraction_constant": float(n_constant / array.shape[0]),
        "any": bool(n_constant),
        "all": bool(n_constant == array.shape[0]),
        "min_within_patient_sd": float(np.min(np.std(array, axis=1))),
        "reading": _DEGENERACY_READINGS[
            "all" if n_constant == array.shape[0]
            else "some" if n_constant else "none"
        ],
    }


#: The three outcomes, written before the randomised set exists.
_DEGENERACY_READINGS = {
    "none": (
        "every randomised vector has rank structure, so the similarity is a "
        "MEASUREMENT and the gate reads as designed"
    ),
    "some": (
        "some randomised vectors are uniform and some are not, so the finals "
        "are a MIXTURE of measured disagreement and structural zeros. Report "
        "the two groups separately; a pooled interval over both answers "
        "neither question"
    ),
    "all": (
        "every randomised vector is uniform: the similarity is 0.0 by "
        "construction, so the gate's pass must be reported as the COLLAPSE "
        "it is, not as a measured separation. [CORRECTED 2026-08-15: an "
        "earlier wording called this 'a strong form of parameter "
        "dependence'. Too generous, and measured wrong -- both attention "
        "layers are permutation-symmetric over regions, so indistinguishable "
        "descriptors give uniform weights for EVERY parameter value; the "
        "outcome is UNINFORMATIVE about parameters. "
        "phase8.CHECK_1_UNIFORM_CONTROL_IS_ARITHMETIC]"
    ),
}


def split_whole_image(weights) -> tuple[np.ndarray, np.ndarray]:
    """``(B, 27)`` -> the 26 ROI weights, and the whole-image weight.

    ``_from_features`` appends the whole feature map as the LAST node, so
    ``region_w[..., -1]`` is that node. Splitting here rather than at
    every call site means the gate cannot accidentally rank a
    non-region alongside the regions.
    """
    from .models.srgnn import N_REGIONS, N_ROIS

    array = np.asarray(weights, dtype=np.float64)
    if array.ndim != 2:
        raise NodeWeightError(
            f"region weights must be (n_patients, n_nodes); got {array.shape}"
        )
    if array.shape[1] != N_REGIONS:
        raise NodeWeightError(
            f"expected {N_REGIONS} nodes ({N_ROIS} ROIs + the whole image); "
            f"got {array.shape[1]}. If the region set changed, the gate's "
            "vector length and its noise change with it "
            "(phase8.NODE_WEIGHT_STATISTIC_DECIDED)"
        )
    return array[:, :N_ROIS], array[:, N_ROIS]


def check_weights(weights) -> np.ndarray:
    """``(n_patients, n_regions)`` float, validated.

    A weight vector shorter than a Grad-CAM map is the whole reason
    ``phase8.NODE_WEIGHT_STATISTIC_DECIDED`` exists, so the shape is
    checked and reported rather than assumed.
    """
    array = np.asarray(weights, dtype=np.float64)
    if array.ndim != 2:
        raise NodeWeightError(
            f"node weights must be (n_patients, n_regions); got {array.shape}"
        )
    if array.shape[0] < 2:
        raise NodeWeightError(
            f"the between-patient baseline needs at least two patients, got "
            f"{array.shape[0]}"
        )
    if array.shape[1] < 2:
        raise NodeWeightError(
            "a rank correlation needs at least two regions; got "
            f"{array.shape[1]}"
        )
    return array


def baseline_pairs(weights) -> np.ndarray:
    """Every between-patient similarity, as the distribution not a point.

    ``gradcam.between_patient_baseline`` returns the median of exactly
    these; the percentiles the adopted criterion reports need the whole
    set, so it is computed once here and both are read off it.
    """
    array = check_weights(weights)
    return np.asarray([
        gradcam.similarity(array[i], array[j])
        for i in range(len(array)) for j in range(i + 1, len(array))
    ], dtype=np.float64)


def per_patient_percentiles(finals, pairs) -> list[dict]:
    """Each patient's randomised similarity, as a PERCENTILE of the
    baseline -- never as pass/fail.

    ``phase8.DEFENSIBLE_CRITERION_FOR_FUTURE_USE``: a per-patient mark is
    not estimable at fifteen patients, and the corrected level sits below
    the resolution of the distribution it would be read from. A
    percentile carries the same information without pretending to a
    threshold, and it says how coarse it is: with ``len(pairs)`` pairs the
    finest distinguishable step is ``1/len(pairs)``.
    """
    pairs = np.asarray(pairs, dtype=np.float64)
    if pairs.size == 0:
        raise NodeWeightError("no baseline pairs; the percentile is undefined")
    resolution = 1.0 / pairs.size
    out = []
    for index, value in enumerate(np.asarray(finals, dtype=np.float64)):
        below = float(np.mean(pairs <= value))
        out.append({
            "patient_index": index,
            "final": float(value),
            "baseline_percentile": below,
            "resolution": resolution,
            # Stated per row so a reader cannot take a percentile finer
            # than the distribution can express.
            "finer_than_resolution": bool(
                below not in (0.0, 1.0) and below < resolution
            ),
        })
    return out


def gate(
    weights, finals, *, n_boot: int = 2000, seed: int = 1337, randomised=None
) -> dict:
    """The adopted criterion, applied to one randomisation mode.

    Phase-level separability is THE criterion
    (``phase8.RANDOMISATION_CRITERION_ADOPTED``); the per-patient
    percentiles ride alongside and decide nothing.

    ``randomised`` is the randomised weight matrix the finals came from. It
    is optional only because a caller may have finals from elsewhere -- when
    it is given, ``degeneracy`` runs and its verdict travels WITH the gate,
    so a pass produced by uniform vectors cannot be read off this dict
    without the fact that produced it.
    """
    array = check_weights(weights)
    finals = np.asarray(finals, dtype=np.float64)
    if finals.shape[0] != array.shape[0]:
        raise NodeWeightError(
            f"{finals.shape[0]} randomised finals against {array.shape[0]} "
            "patients; every patient needs one"
        )
    pairs = baseline_pairs(array)
    separable = gradcam.separability(
        list(finals), list(array), n_boot=n_boot, seed=seed
    )
    return {
        "criterion": "phase-level separability interval",
        "n_patients": int(array.shape[0]),
        "n_regions": int(array.shape[1]),
        "n_baseline_pairs": int(pairs.size),
        "baseline_median": float(np.median(pairs)),
        "separability": separable,
        "resolves": bool(separable["excludes_zero"]),
        "per_patient": per_patient_percentiles(finals, pairs),
        **(
            {} if randomised is None
            else {"degeneracy": degeneracy(randomised)}
        ),
        "per_patient_is_not_a_gate": (
            "percentiles of the baseline, reported alongside. A "
            "per-patient pass mark is not estimable below n=41 "
            "(phase8.PER_FACE_CLAIM_COSTS_N_41)"
        ),
        "if_it_does_not_resolve": (
            "reported UNRESOLVED, not rescued by switching statistic "
            "afterwards (phase8.NODE_WEIGHT_STATISTIC_DECIDED)"
        ),
    }


def compare_modes(by_mode: dict) -> dict:
    """The two randomisations read together, per the registration.

    Both readings were fixed before any number existed
    (``phase8.TWO_MODE_READINGS_REGISTERED``), because a DISAGREEMENT is
    the more interesting outcome and is exactly the kind that invites an
    account invented to fit it.
    """
    missing = [mode for mode in RANDOMISATION_MODES if mode not in by_mode]
    if missing:
        raise NodeWeightError(
            f"both randomisations are required; missing {missing}. Running "
            "one and reading it as the arm's gate is the asymmetry the "
            "two-mode design exists to avoid"
        )
    component = bool(by_mode[COMPARISON_MODE]["resolves"])
    explanation = bool(by_mode["explanation_source"]["resolves"])
    return {
        "component_role_resolves": component,
        "explanation_source_resolves": explanation,
        "agree": component == explanation,
        "comparison_uses": COMPARISON_MODE,
        "reading": _MODE_READINGS[(component, explanation)],
    }


def publishable(by_mode: dict) -> dict:
    """May any node weights be shown, given the modes that RAN?

    The rule comes from ``_MODE_READINGS[(False, False)]`` -- "neither
    resolves: the gate is UNRESOLVED for this arm, no node weights may be
    shown" -- generalised to the case where only one mode ran. A
    ``component_role`` result needs its control artifact declared
    (``CONTROL_SET_MODES``), and a run without one has not tested it. A mode
    that did not run is not a mode that failed, and neither is it evidence:
    it contributes nothing in either direction.

    Refuses an empty dict rather than returning False. "No gate ran" and
    "the gate did not resolve" are different states, and a publication rule
    that conflates them would let a run with no randomisation at all report
    the same verdict as one that failed its own test.
    """
    if not by_mode:
        raise NodeWeightError(
            "no randomisation mode was run, so there is no gate to publish "
            "against. A run that randomised nothing is not a run that failed "
            "its gate"
        )
    unknown = [mode for mode in by_mode if mode not in RANDOMISATION_MODES]
    if unknown:
        raise NodeWeightError(
            f"unknown randomisation mode(s) {unknown}; expected "
            f"{RANDOMISATION_MODES}"
        )
    resolved = sorted(mode for mode in by_mode if by_mode[mode]["resolves"])
    not_run = [mode for mode in RANDOMISATION_MODES if mode not in by_mode]
    return {
        "publishable": bool(resolved),
        "modes_run": sorted(by_mode),
        "modes_resolved": resolved,
        "modes_not_run": not_run,
        "rule": (
            "weights may be shown if at least one randomisation the arm "
            "COULD run resolves; if none does, the gate is UNRESOLVED and "
            "nothing is shown (_MODE_READINGS[(False, False)])"
        ),
        "not_run_is_not_evidence": (
            "a mode that could not be run contributes nothing in either "
            "direction -- it is not a failed test and not a passed one"
        ),
    }


#: The four outcomes, written before the numbers. Keyed
#: ``(component_role resolves, explanation_source resolves)``.
_MODE_READINGS = {
    (True, True): (
        "both resolve: the weights depend on the frozen representation AND "
        "on the trained attention. The explanation is not an artefact of "
        "either alone, and the A<->B comparison proceeds on the "
        "component-role result"
    ),
    (True, False): (
        "the representation matters and the trained attention does not "
        "detectably: the weights track what the backbone encodes, and the "
        "graph layers are closer to a readout than to the explanation's "
        "source. Notable because arm B's weights are PRODUCED by those "
        "layers"
    ),
    (False, True): (
        "the trained attention matters and the representation does not "
        "detectably: the weights are a property of what the graph layers "
        "learned rather than of the features they read. This is the case "
        "where arm A's question and arm B's explanation come furthest "
        "apart, and the A<->B comparison is weakest"
    ),
    (False, False): (
        "neither resolves: the gate is UNRESOLVED for this arm, no node "
        "weights may be shown, and nothing is inferred from the pair -- "
        "two unresolved tests are not evidence of independence"
    ),
}
