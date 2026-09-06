"""The SHAREABLE block printed at the end of a Phase 1 build.

Exit criteria 6-10 are verified by someone reading numbers pasted out of a
cluster log. Nothing patient-keyed can appear in that paste, so this block is
built to be aggregate-only **by construction rather than by care**:
``build_summary`` never receives the manifest rows, the grade matrix or the fold
assignments. It takes counts and statistics. There is nothing patient-level in
scope for it to leak.

The one exception is deliberate: the rule cross-check names folders **143 and
238**. Exit criterion 7 requires exactly that, and those two folder numbers are
already published in the governing plan as the two known exceptions. No other
patient identifier may appear, and ``tests/test_report.py`` asserts it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: The two folders where the odd/even rule is known to disagree with the score
#: sheet. Published in PLAN §4.1, so naming them is not a disclosure.
PUBLISHED_EXCEPTIONS = (143, 238)


@dataclass(frozen=True)
class CohortCounts:
    patients: int
    frontal: int
    basal: int
    photoless: int
    images: int
    scored_rows: int


@dataclass(frozen=True)
class Phase1Summary:
    counts: CohortCounts
    rule_disagreements: tuple[int, ...]
    rule_matches_expected: bool
    fleiss_kappa: float
    mean_pairwise_qwk: float
    mean_inter_rater_r: float
    reliability_237: float
    pcc_ceiling_237: float
    learnability_237: dict[str, float]
    learnability_ordering: tuple[str, ...]
    ordering_matches_expected: bool
    fold_sizes: tuple[int, ...]
    fold_class_counts: dict[str, dict[str, int]]
    per_rater_r: dict[str, float] = field(default_factory=dict)
    learnability_population: int = 237
    learnability_caveat: str = ""

    def as_dict(self) -> dict:
        return {
            "counts": vars(self.counts),
            "rule_disagreements": list(self.rule_disagreements),
            "rule_matches_expected": self.rule_matches_expected,
            "fleiss_kappa": self.fleiss_kappa,
            "mean_pairwise_qwk": self.mean_pairwise_qwk,
            "mean_inter_rater_r": self.mean_inter_rater_r,
            "reliability_237": self.reliability_237,
            "pcc_ceiling_237": self.pcc_ceiling_237,
            # Nested rather than a flat ``learnability_237`` key. The population
            # suffix was deliberate -- these values differ by population and must
            # never be compared across one -- but as a top-level key name it was
            # undiscoverable: a consumer reading the artifact looked for
            # ``summary.learnability`` and found nothing at all. The population is
            # now explicit INSIDE the block, which keeps the warning and makes the
            # field findable.
            "learnability": {
                "population": self.learnability_population,
                "scores": dict(self.learnability_237),
                "ordering": list(self.learnability_ordering),
                "matches_expected_ordering": self.ordering_matches_expected,
                # Travels with the numbers, which was always the intent; it was
                # reaching the log and not the artifact.
                "caveat": self.learnability_caveat,
            },
            "fold_sizes": list(self.fold_sizes),
            "fold_class_counts": self.fold_class_counts,
            "per_rater_r": dict(self.per_rater_r),
        }

    def render(self) -> str:
        """A paste-able block. Aggregates only."""
        counts = self.counts
        rule = ", ".join(str(f) for f in self.rule_disagreements) or "none"
        tick = "OK" if self.rule_matches_expected else "MISMATCH"
        order_tick = "OK" if self.ordering_matches_expected else "MISMATCH"

        lines = [
            "=" * 70,
            "PHASE 1 SUMMARY  [SHAREABLE - aggregate scalars only]",
            "=" * 70,
            "",
            "-- structure (exit criterion 6) ------------------------------------",
            f"  patients             {counts.patients}",
            f"  frontal images       {counts.frontal}",
            f"  basal images         {counts.basal}",
            f"  photo-less scored    {counts.photoless}",
            f"  total images         {counts.images}",
            f"  score-sheet rows     {counts.scored_rows}"
            f"   ({counts.patients} + {counts.photoless})",
            "",
            "-- frontal rule cross-check (exit criterion 7) ----------------------",
            f"  rule disagrees in folders: {rule}   [{tick}]",
            f"  expected exactly:          {', '.join(str(f) for f in PUBLISHED_EXCEPTIONS)}",
            "",
            "-- reliability, on the 237 (exit criterion 8) -----------------------",
            # **[2026-09-02] Fleiss kappa is UNWEIGHTED** -- 1-vs-2 scores as
            # identically wrong to 1-vs-5. Distance-aware beside it: QWK 0.4276,
            # mean inter-rater r 0.4696. record_audit.THE_KAPPA_LIMITATION.
            f"  Fleiss kappa           {self.fleiss_kappa:.4f}",
            f"  mean pairwise QWK      {self.mean_pairwise_qwk:.4f}",
            f"  mean inter-rater r     {self.mean_inter_rater_r:.4f}",
            f"  reliability_237 (SB)   {self.reliability_237:.4f}",
            f"  pcc_ceiling_237        {self.pcc_ceiling_237:.4f}   (= sqrt of the above)",
            "",
            "-- learnability, on the 237 (exit criterion 9) ----------------------",
        ]
        for target, value in self.learnability_237.items():
            lines.append(f"  {target:<16} {value:.4f}")
        lines += [
            f"  ordering: {' > '.join(self.learnability_ordering)}   [{order_tick}]",
            "  NOTE: these are the 237-patient values. Brief §1.7 quotes the 251-row",
            "        values, which are about 0.012 lower. Different population, not",
            "        a different result.",
            "",
            "-- folds (exit criterion 10) ----------------------------------------",
            f"  fold sizes  {list(self.fold_sizes)}",
        ]
        for fold, counts_by_class in sorted(self.fold_class_counts.items()):
            rendered = ", ".join(f"{k}:{v}" for k, v in sorted(counts_by_class.items()))
            lines.append(f"  fold {fold}      {rendered}")

        if self.per_rater_r:
            lines += ["", "-- per-rater item-total r ------------------------------------------"]
            for rater, value in self.per_rater_r.items():
                lines.append(f"  {rater:<44} {value:.4f}")

        lines += ["", "=" * 70]
        return "\n".join(lines)


def build_summary(
    *,
    counts: CohortCounts,
    rule_disagreements: set[int],
    reliability,
    learnability,
    expected_ordering: tuple[str, ...],
    fold_sizes: list[int],
    fold_class_counts: dict[int, dict[int, int]],
    rater_names: tuple[str, ...] = (),
) -> Phase1Summary:
    """Assemble the block.

    Deliberately takes counts and statistics, never the manifest rows, the grade
    matrix or the fold assignments -- so there is no patient-level value in scope
    to leak, regardless of how carefully or carelessly this is edited later.
    """
    disagreements = tuple(sorted(rule_disagreements))
    ordering = tuple(learnability.ordering())

    per_rater = {}
    if rater_names:
        for stat in reliability.per_rater:
            if stat.index < len(rater_names):
                per_rater[rater_names[stat.index]] = stat.r

    return Phase1Summary(
        counts=counts,
        rule_disagreements=disagreements,
        rule_matches_expected=disagreements == PUBLISHED_EXCEPTIONS,
        fleiss_kappa=reliability.fleiss_kappa,
        mean_pairwise_qwk=reliability.mean_pairwise_qwk,
        mean_inter_rater_r=reliability.mean_inter_rater_r,
        reliability_237=reliability.reliability,
        pcc_ceiling_237=reliability.pcc_ceiling,
        learnability_237=dict(learnability.scores),
        learnability_ordering=ordering,
        ordering_matches_expected=ordering == tuple(expected_ordering),
        fold_sizes=tuple(fold_sizes),
        fold_class_counts={
            str(fold): {str(c): n for c, n in sorted(counts_by_class.items())}
            for fold, counts_by_class in sorted(fold_class_counts.items())
        },
        per_rater_r=per_rater,
        learnability_population=getattr(learnability, "population", 237),
        learnability_caveat=getattr(learnability, "caveat", ""),
    )
