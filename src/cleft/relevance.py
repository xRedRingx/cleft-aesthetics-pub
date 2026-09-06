"""Univariate feature relevance: which regions' asymmetry tracks the grade?

**A diagnostic, not an arm.** No fitting, no seeds, no folds, nothing tuned on
it. Each feature is correlated against the label on its own and the list is
reported ranked.

**Why univariate rather than model coefficients.** The obvious move is to read
ridge coefficients off the mirror-difference arm, and it would be wrong: the 22
region features are strongly correlated with each other -- a face asymmetric at
the alar base is usually asymmetric at the nostril sill too -- and correlated
predictors make individual coefficients unstable. They can flip sign with a
resample while the model's predictions barely move. A univariate correlation
asks a cleaner question and answers it cleanly: *does this region's asymmetry, by
itself, order patients the way the raters did?*

**Both statistics, and they answer different things.** Spearman asks whether the
ordering agrees, Pearson whether the relationship is linear. Asymmetry indices
are bounded below at zero and right-skewed, so the two can disagree, and a
feature that ranks well but correlates poorly is still a useful signal.

**The data already exists.** The mirror-difference arm computed all 22 features
for all 237 patients, so this costs one pass and no training.

Lives at the package top level rather than in ``eval`` because ``eval`` is
frozen -- adding a module there would change the package rollup and invalidate
every result measured with it. The freeze guard caught exactly that on the first
attempt, which is the guard doing its job rather than an inconvenience.

**[MEASURED 2026-07-30] It has been run.** ``MEASURED_RELEVANCE`` holds the
result, ``NOSE_TIP_DECISION`` what it did and did not decide about the nose tip,
and ``AGGREGATION_NOTE`` a methodological finding that outlives this diagnostic.
The headline, because it governs how any of it may be quoted: **nothing clears
the Bonferroni bar.** Two features clear the uncorrected one, and ``report`` emits
both bars precisely so those two are not read as findings.
"""

from __future__ import annotations

import math

import numpy as np

from .eval import metrics


class RelevanceError(RuntimeError):
    """The relevance report cannot be built."""


def significance_threshold(n: int, n_features: int = 1, alpha: float = 0.05) -> float:
    """The |r| a correlation must reach to clear ``alpha``, Bonferroni-adjusted.

    A normal approximation, ``z / sqrt(n - 3)``, which is what the Fisher
    transform gives for a two-sided test. Computed rather than memorised so a
    different ``n`` or feature count gives a re-derived bar, and so the
    uncorrected and corrected thresholds are visibly the SAME formula at
    different ``n_features`` -- the distinction that decides what this diagnostic
    can and cannot claim.

    At n=237: 0.128 uncorrected, 0.199 across 22 features.
    """
    if n < 5:
        raise RelevanceError(f"n={n} is too small for a correlation threshold")
    if n_features < 1:
        raise RelevanceError(f"n_features must be at least 1, got {n_features}")
    # Two-sided, Bonferroni across n_features.
    tail = alpha / (2.0 * n_features)
    # Inverse normal CDF via the complementary error function's inverse. Newton
    # on erfc is overkill; a bisection over a generous bracket is exact enough
    # for a reporting threshold and has no dependency.
    low, high = 0.0, 10.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if 0.5 * math.erfc(middle / math.sqrt(2.0)) > tail:
            low = middle
        else:
            high = middle
    z = (low + high) / 2.0
    return z / math.sqrt(n - 3)


#: **[MEASURED 2026-07-30] The diagnostic, run on the cluster over `staged_v1`
#: and `cleft_v1`. This is what it decided and what it did not.**
#:
#: Ranked by |Spearman|, n=237, mean label, G2. Thresholds from
#: ``significance_threshold``: **0.128** uncorrected, **0.199** Bonferroni across
#: 22 features.
#:
#: | rank | feature | |Spearman| | clears |
#: |---|---|---|---|
#: | 1 | `philtrum` | **0.151** | uncorrected only |
#: | 2 | `labial_tubercle` | **0.138** | uncorrected only |
#: | 3 | `philtral_column` | 0.122 | neither -- just under |
#: | 4 | `subnasale` | — | neither |
#: | 19 | `alar_base` | ~0 | neither |
#: | 20 | `commissure` | ~0 | neither |
#: | 22 | `nasal_tip` | **0.002** | neither |
#:
#: **Nothing clears the corrected bar.** So this is a ranking that informs a
#: design decision, exactly as declared before the run, and not a set of claims.
#:
#: **The top four are all midline, all lower face, all in the philtral and
#: upper-lip complex.** That coheres with the group's own finding that raters are
#: more reliable scoring lips alone -- an independent line of evidence pointing at
#: the same anatomy, which is worth more than either on its own.
MEASURED_RELEVANCE = {
    "measured": "2026-07-30",
    "n_patients": 237,
    "n_features": 22,
    "label": "mean",
    "geometry": "g2",
    "ranked_by": "abs(spearman)",
    "threshold_uncorrected": 0.128,
    "threshold_bonferroni_22": 0.199,
    "n_clearing_bonferroni": 0,
    "top": (
        {"rank": 1, "feature": "philtrum", "abs_spearman": 0.151},
        {"rank": 2, "feature": "labial_tubercle", "abs_spearman": 0.138},
        {"rank": 3, "feature": "philtral_column", "abs_spearman": 0.122},
        {"rank": 4, "feature": "subnasale", "abs_spearman": None},
    ),
    "notable_ranks": {
        "alar_base": 19,
        "commissure": 20,
        "nasal_tip": 22,
    },
    "nasal_tip_abs_spearman": 0.002,
    "mad_whole_abs_spearman": 0.004,
    "coherence": (
        "the top four are all MIDLINE, all LOWER FACE, all in the philtral and "
        "upper-lip complex -- which coheres with the group's own finding that "
        "raters are more reliable scoring lips alone. Two independent lines of "
        "evidence pointing at the same anatomy."
    ),
}

#: **[MEASURED 2026-07-30] The nose-tip question, decided narrowly.**
#:
#: ``nasal_tip`` ranks **22 of 22** at |Spearman| 0.002. **So the suggestion raised at supervision is
#: not supported by this measurement** -- and the scope of "this measurement" is
#: the whole of what makes the result usable, because two other readings of the
#: nose tip are untouched by it.
#:
#: What was measured: **pixel-level mirror asymmetry within a box placed by
#: anatomy fractions.** What was not:
#:
#: * **nasal tip deviation from the midline** -- a DISPLACEMENT of a structure,
#:   not a regional pixel difference. A tip displaced bodily off the midline can
#:   leave the mirror difference inside its own box almost unchanged, because the
#:   box travels with the anatomy fractions rather than with the face's midline.
#:   This is also the quantity the Phase 2 cleft sheet's tilt gradient pointed at.
#: * **the tip as an alignment anchor** -- a use, not a feature. Its value would
#:   be in registering faces before anything is measured, and a zero correlation
#:   of its regional asymmetry says nothing either way about that.
#:
#: **Both remain live and neither is tested here.** Recording the null as "the
#: nose tip does not matter" would be the R2 error in a new place: the right
#: number attached to a question it cannot answer.
NOSE_TIP_DECISION = {
    "measured": "2026-07-30",
    "question": "how the nose tip enters the pipeline (PLAN Part 5, Phase 5)",
    "result": "nasal_tip ranks 22 of 22, abs(spearman) 0.002",
    "decided": (
        "NOT SUPPORTED as a weighted region on this evidence: its regional "
        "pixel mirror asymmetry does not track the grade"
    ),
    "what_was_measured": (
        "pixel-level mirror asymmetry WITHIN A BOX placed by anatomy fractions"
    ),
    "still_live_and_untested": {
        "tip_deviation_from_midline": (
            "a DISPLACEMENT of the structure, not a regional pixel difference. A "
            "tip displaced bodily off the midline can leave the mirror "
            "difference inside its own box nearly unchanged, because the box is "
            "placed by anatomy fractions and travels with the anatomy. It is "
            "also the quantity the Phase 2 cleft sheet's tilt gradient pointed "
            "at -- worst at the nose tips, least at the lips."
        ),
        "tip_as_alignment_anchor": (
            "a USE, not a feature. Its value would be in registering faces "
            "before anything is measured, and a zero correlation of regional "
            "asymmetry says nothing either way about that."
        ),
    },
    "do_not_conclude": (
        "'the nose tip does not matter'. That would attach a correct number to "
        "a question it cannot answer -- the R2 error in a new place. Two of the "
        "three ways the tip could enter are untested."
    ),
}

#: **[MEASURED 2026-07-30] A methodological note the ranking produced, and it
#: generalises past this diagnostic.**
#:
#: ``mad_whole`` scores **0.004** while individual regions carry **three to thirty
#: times more**, and the band features are near zero. So the mirror-difference
#: arm's **PCC 0.158 comes from the ridge combining many weak regional signals**,
#: not from any single site and not from the aggregate.
#:
#: **An aggregate that averages heterogeneous signals reports none of them.** This
#: is the same lesson as the fairness quartiles -- where a cohort-level pass rate
#: hid a brightness-linked gradient -- and as ``separability``, where a whole-frame
#: statistic measured face-versus-background instead of lip-versus-skin. Three
#: instances now, in three different parts of the project.
#:
#: The practical rule: **report the per-unit breakdown beside any aggregate whose
#: units might differ from each other**, and treat an aggregate near zero as
#: uninformative about its parts rather than as evidence they are all zero.
AGGREGATION_NOTE = {
    "measured": "2026-07-30",
    "mad_whole_abs_spearman": 0.004,
    "regional_range_vs_whole": "individual regions carry 3x to 30x more",
    "bands": "near zero",
    "consequence": (
        "the mirror-difference arm's PCC 0.158 comes from the RIDGE COMBINING "
        "weak regional signals, not from any single site and not from the "
        "aggregate. An aggregate near zero is uninformative about its parts, "
        "not evidence that the parts are zero."
    ),
    "same_lesson_as": (
        "the fairness quartiles, where a cohort-level pass rate hid a "
        "brightness-linked gradient; and separability, where a whole-frame "
        "statistic measured face-versus-background rather than lip-versus-skin"
    ),
    "rule": (
        "report the per-unit breakdown beside any aggregate whose units might "
        "differ from one another"
    ),
}


def rank_features(
    features: np.ndarray, labels, names: list[str]
) -> list[dict]:
    """Each feature against the label, on its own, ranked by |Spearman|."""
    matrix = np.asarray(features, dtype=float)
    target = np.asarray(labels, dtype=float)

    if matrix.ndim != 2:
        raise RelevanceError(f"expected an (N, D) matrix, got shape {matrix.shape}")
    if matrix.shape[0] != target.size:
        raise RelevanceError(
            f"{matrix.shape[0]} feature rows against {target.size} labels"
        )
    if matrix.shape[1] != len(names):
        raise RelevanceError(
            f"{matrix.shape[1]} feature columns against {len(names)} names"
        )

    rows = []
    for column, name in enumerate(names):
        values = matrix[:, column]
        # A constant feature has no correlation to report, and asking for one
        # would divide by a zero standard deviation.
        if np.std(values) <= 1e-12:
            rows.append(
                {
                    "feature": name,
                    "spearman": None,
                    "pearson": None,
                    "constant": True,
                    "sd": 0.0,
                }
            )
            continue
        rows.append(
            {
                "feature": name,
                "spearman": round(float(metrics.spearman(target, values)), 6),
                "pearson": round(float(metrics.pcc(target, values)), 6),
                "constant": False,
                "sd": round(float(np.std(values)), 6),
            }
        )

    rows.sort(key=lambda row: abs(row["spearman"] or 0.0), reverse=True)
    for position, row in enumerate(rows, start=1):
        row["rank_by_abs_spearman"] = position
    return rows


def report(features: np.ndarray, labels, names: list[str]) -> dict:
    """The ranked table plus what it is and is not. SHAREABLE -- aggregate."""
    ranked = rank_features(features, labels, names)
    usable = [row for row in ranked if not row["constant"]]
    n = int(np.asarray(labels).size)

    # **Both bars, reported together.** No correction is APPLIED -- the ranking
    # stands as declared -- but the corrected bar is stated beside the
    # uncorrected one so a reader can see which features would survive it. On the
    # 2026-07-30 run the answer was NONE, and that is the single most important
    # fact about how strongly this may be quoted. Leaving the corrected figure
    # out would let the two features clearing 0.128 read as findings.
    uncorrected = round(significance_threshold(n), 6) if n >= 5 else None
    corrected = (
        round(significance_threshold(n, len(names)), 6)
        if n >= 5 and names
        else None
    )

    def clears(row: dict, bar: float | None) -> bool:
        return bool(
            bar is not None
            and row["spearman"] is not None
            and abs(row["spearman"]) >= bar
        )

    return {
        "n_patients": n,
        "n_features": len(names),
        "ranked": ranked,
        "n_constant": sum(1 for row in ranked if row["constant"]),
        "strongest": usable[0]["feature"] if usable else None,
        "max_abs_spearman": (
            round(abs(usable[0]["spearman"]), 6) if usable else None
        ),
        "threshold_uncorrected": uncorrected,
        "threshold_bonferroni": corrected,
        "clearing_uncorrected": [
            row["feature"] for row in usable if clears(row, uncorrected)
        ],
        "clearing_bonferroni": [
            row["feature"] for row in usable if clears(row, corrected)
        ],
        "spearman_and_pearson_disagree": [
            row["feature"]
            for row in usable
            if row["spearman"] * row["pearson"] < 0
        ],
        "note": (
            "DIAGNOSTIC, NOT AN ARM. No fitting, no seeds, no folds, and nothing "
            "is tuned on this. It says which regions' asymmetry tracks the grade "
            "on its own. "
            "Univariate rather than ridge coefficients ON PURPOSE: the region "
            "features are strongly correlated with each other, which makes "
            "individual coefficients unstable -- they can flip sign on a resample "
            "while predictions barely move. "
            "Spearman asks whether the ORDERING agrees, Pearson whether the "
            "relationship is LINEAR; asymmetry indices are bounded at zero and "
            "right-skewed, so the two can disagree and a feature ranking well "
            "while correlating poorly is still signal. Ranking is by |Spearman|. "
            "NO MULTIPLE-COMPARISON CORRECTION IS APPLIED. With 22 features some "
            "will correlate by chance; treat this as a ranking to inform a design "
            "decision, not as a set of claims. Anything acted on needs its own "
            "arm with the usual claim criterion. "
            "BOTH BARS ARE REPORTED ANYWAY, and read clearing_bonferroni first: "
            "on the 2026-07-30 run it was EMPTY, which is the single most "
            "important fact about how strongly any of this may be quoted. "
            "Reporting only the uncorrected bar would let the features above it "
            "read as findings. "
            "An aggregate feature scoring near zero says nothing about its "
            "parts -- mad_whole was 0.004 while regions carried 3x to 30x more. "
            "See AGGREGATION_NOTE."
        ),
    }
