"""Generate Phase 22's three post-run configs.

The diagnostic, the shrinkage measurement and the contrast family. All
three consume the four cohort arms' run directories, which are NEW
declared inputs -- so their hashes are emitted as all-zero placeholders
and the maintainer's ``declare_inputs.py`` fills them. Nothing else here is
new: the 63 comparison arms and the manifest are carried VERBATIM from
the shipped Phase 21 configs, so no other hash enters.

**The 63 are derived, not typed**: read from
``configs/p21_error_consistency.yaml``'s own 64 arms with ``p17_arm_a``
removed -- it is the reference the diagnostic's LOW threshold comes
from, and putting it on both sides would compare it with itself.

Usage::

    python scripts/generate_phase22_postrun_configs.py            # write
    python scripts/generate_phase22_postrun_configs.py --check    # diff
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"

#: Carried verbatim: the 63 comparison arms and the manifest.
DIAGNOSTIC_DONOR = "p21_error_consistency.yaml"
SHRINKAGE_DONOR = "p21_cross_arm_shrinkage.yaml"

#: The arm the diagnostic excludes from its comparison set.
ARM_A = "p17_arm_a"

#: **[FILLED 2026-09-01] The four arm hashes, from the run outputs's
#: ``scripts/declare_inputs.py`` output** -- identical across all three
#: configs, which the generator asserts and a test re-asserts: a
#: per-config divergence would mean one config points at a different
#: object under the same name. Supplied rather than derived, recorded
#: verbatim, never
#: derived here.
DECLARED_HASHES = {
    "r_all_bounded":
        "d67d44028f3cb5a4383e9b1a35f2b65f90f3de6b7231ed027978acab3b1ca017",
    "r_all_unbounded":
        "aabc3cbf7c8a9409b24d4739fc76ba091844093a45ff1afc0cc483d42745ccd6",
    "r_clear_bounded":
        "4108e900764e50a32f5f6091ec82f4c45612f3861083db524df2cad48c56e957",
    "r_clear_unbounded":
        "057d15cc7ab8baa116974b29475d2e6114219f835ea5f0559d88288d9044f446",
}

ARMS = (
    "r_all_bounded", "r_all_unbounded",
    "r_clear_bounded", "r_clear_unbounded",
)


def _arm_config(arm: str) -> dict:
    """The arm's own training config -- the source for its seeds and
    patient count, which were TYPED in the first version."""
    return _load(f"p22_{arm}.yaml")


def _seeds() -> list:
    """Read from the arm configs, and required to agree across all four.

    [CORRECTED 2026-09-01] These were typed as a literal. They happened
    to be right; they were not read.
    """
    seen = {tuple(_arm_config(arm)["task"]["seeds"]) for arm in ARMS}
    if len(seen) != 1:
        raise SystemExit(f"the four arms declare different seeds: {seen}")
    return list(seen.pop())


def _n_patients() -> int:
    """Read from the arm configs, likewise."""
    seen = {int(_arm_config(arm)["task"]["expect_patients"]) for arm in ARMS}
    if len(seen) != 1:
        raise SystemExit(f"the four arms declare different cohorts: {seen}")
    return seen.pop()

#: **[CORRECTED 2026-09-01] The four run directory names, SUPPLIED, not
#: constructed.**
#:
#: The first version built these from the ``<stem>__<sha8>__<job-id>``
#: contract and got the job-id wrong: every real name ends ``-2``, which
#: no contract predicts. **The job-id is whatever the maintainer set at
#: launch. It is a fact only the run knows, and a generator that builds
#: it is guessing.**
#:
#: **How this table is obtained**: the maintainer runs ``ls`` in
#: ``runs/keeper/p22/`` and supplies the names; they are recorded here
#: VERBATIM. The laptop cannot list that directory -- it is cluster-only
#: -- so there is no honest way to derive them here, and the convenient
#: way (rebuilding from the contract) is exactly what failed. If a
#: future run's directories differ, this table is updated from a fresh
#: ``ls`` and nowhere else.
#:
#: Supplied 2026-09-01 from ``ls runs/keeper/p22/*44ad1f7e*``.
RUN_DIRECTORIES = {
    "r_all_bounded": "p22_r_all_bounded__44ad1f7e__p22-r-all-bounded-2",
    "r_all_unbounded": "p22_r_all_unbounded__44ad1f7e__p22-r-all-unbounded-2",
    "r_clear_bounded": "p22_r_clear_bounded__44ad1f7e__p22-r-clear-bounded-2",
    "r_clear_unbounded": (
        "p22_r_clear_unbounded__44ad1f7e__p22-r-clear-unbounded-2"
    ),
}

RUN_ROOT = "/home/user/codex/cleft-aesthetics/runs/keeper/p22"


def _load(stem: str) -> dict:
    import yaml

    return yaml.safe_load((CONFIGS / stem).read_text(encoding="utf-8"))


def _new_arm_inputs() -> list[dict]:
    """The four arm run directories, from the SUPPLIED name table.

    **This function must never build a run directory name.** It looks
    one up and refuses if it is missing, so a name that was not supplied
    cannot reach a config by being plausible.
    """
    entries = []
    for arm in ARMS:
        directory = RUN_DIRECTORIES.get(arm)
        if not directory:
            raise SystemExit(
                f"no supplied run directory name for {arm!r}. RUN_DIRECTORIES "
                "is filled from the run outputs's `ls` of runs/keeper/p22 and is "
                "NOT derivable from the config -- the job-id fragment is a "
                "launch-time fact. Ask for the name; do not construct it."
            )
        digest = DECLARED_HASHES.get(arm)
        if not digest or len(digest) != 64 or not re.fullmatch(
            r"[0-9a-f]{64}", digest
        ):
            raise SystemExit(
                f"no declared hash for {arm!r}. DECLARED_HASHES is filled "
                "from declare_inputs.py output only."
            )
        entries.append({
            "name": f"p22_{arm}",
            "path": f"{RUN_ROOT}/{directory}",
            "rollup_sha256": digest,
        })
    return entries


def _new_arm_entries() -> list[dict]:
    """**The arm `name` is the BARE arm name, not the input name.**

    `phase22.contrast_family()` names its arms `r_all_bounded` and so
    on; the declared INPUT is `p22_r_all_bounded`. The first version
    used the prefixed name for both, so the contrast task found ZERO
    evaluable contrasts against a config declaring eight. Caught before
    launch by checking the filter rather than by reading the code.
    """
    return [
        {
            "name": arm,
            "input": f"p22_{arm}",
            "seeds": _seeds(),
            # The CSV stem is the donor's convention, read from it
            # rather than retyped -- phase3.write_outputs writes
            # `seed_<n>__predictions.csv` for every arm.
            "csv": _donor_csv_stem(),
            # A LABEL, invented here and consumed only as a grouping
            # key in the report. Nothing derives it, and nothing checks
            # it against anything -- named as invented rather than
            # dressed as read.
            "group": "p22_ranking",
            "n_patients": _n_patients(),
        }
        for arm in ARMS
    ]


def _high_kappa() -> dict:
    """The all-pairs means, from phase21.ARM_B_OBSERVED."""
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from cleft import phase21

    return {
        b: float(phase21.ARM_B_OBSERVED[b]["mean_kappa"])
        for b in ("residual_sign", "worst_quartile")
    }


def _low_kappa() -> dict:
    """Arm A's measured level, from its structured mirror."""
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from cleft import phase21

    return {
        k: float(v) for k, v in
        phase21.ARM_A_THE_SHRINKAGE_CONTROL["kappa_measured"].items()
    }


def _donor_csv_stem() -> str:
    """The `csv` token for a ranking arm, READ FROM THE WRITER.

    [CORRECTED 2026-09-01] The first attempt read it from the Phase 21
    donor, which turned out to use TWO stems -- `predictions` and
    `identity_predictions` -- so there is no single donor convention to
    copy. The honest source is the code that writes the file:
    ``phase3.write_outputs`` emits ``<prefix>predictions.csv``, and the
    ranking task passes ``prefix=f"seed_{seed}__"``. Verified against
    that source rather than assumed.
    """
    writer = (REPO / "src" / "cleft" / "train" / "phase3.py").read_text(
        encoding="utf-8"
    )
    if 'f"{prefix}predictions.csv"' not in writer:
        raise SystemExit(
            "phase3.write_outputs no longer writes <prefix>predictions.csv; "
            "the csv stem these configs declare must be re-derived from it."
        )
    return "predictions"


def diagnostic() -> tuple[str, str]:
    import yaml

    donor = _load(DIAGNOSTIC_DONOR)
    by_name = {e["name"]: e for e in donor["inputs"]}
    comparison = [a for a in donor["task"]["arms"] if a["name"] != ARM_A]
    if len(comparison) != 63:
        raise SystemExit(
            f"{len(comparison)} comparison arms after removing {ARM_A!r}; "
            "expected 63"
        )
    manifest = donor["task"]["manifest_artifact"]
    # **Several arms share one run directory** -- different prediction
    # CSVs from the same run -- so the input list must be DEDUPED by
    # name while keeping order. Caught by schema.validate's duplicate
    # check, not by reading the donor.
    inputs, seen = [], set()
    for entry in [by_name[a["input"]] for a in comparison] + [
        by_name[manifest]
    ]:
        if entry["name"] not in seen:
            seen.add(entry["name"])
            inputs.append(entry)
    inputs += _new_arm_inputs()

    task = {
        "kind": "p22_diagnostic",
        "manifest_artifact": manifest,
        "arms": comparison + _new_arm_entries(),
        "seeds": _seeds(),
        "subject_arms": list(ARMS),
        "expect_comparison_arms": 63,
        "binarisations": ["residual_sign", "worst_quartile"],
        # [CORRECTED 2026-09-01] READ from phase21, not typed. The
        # task checks these against the same sources at run time, so a
        # typed value that drifted would have been caught there -- but
        # deriving them means the config cannot be wrong in the first
        # place.
        "expect_high_kappa": _high_kappa(),
        "expect_low_kappa": _low_kappa(),
        "expect_patients": _n_patients(),
    }
    header = f"""\
# PHASE 22 -- THE FEATURE-DIFFERENCE DIAGNOSTIC
#
# GENERATED by scripts/generate_phase22_postrun_configs.py.
# Do not hand-edit: --check re-derives and diffs.
#
# phase22.FEATURE_DIFFERENCE_DIAGNOSTIC. REGISTERS NOTHING --
# both readings and both thresholds were committed before any
# arm ran. This reports which cell each arm fires.
#
# THE FOUR COHORT ARMS against THE 63 SHRINKING ARMS. The 63
# are the shipped Phase 21 set with {ARM_A} removed: it is the
# reference the LOW threshold comes from, and putting it on both
# sides would compare it with itself.
#
# THE THRESHOLDS ARE READ FROM phase21 AT RUN TIME, never from
# this file. expect_high_kappa and expect_low_kappa declare what
# the task expects to FIND -- if either source figure has
# changed, EXIT_CRITERIA criterion 6 stops the phase rather than
# reading on.
#   HIGH  0.7012 / 0.5729  phase21.ARM_B_OBSERVED
#   LOW   0.0956 / -0.0743 phase21.ARM_A_THE_SHRINKAGE_CONTROL
#
# Both binarizations, REPORTED SEPARATELY and never averaged.
#
# The kappa arithmetic is the SHIPPED one -- error_indicators and
# error_consistency_matrix, the same functions Phase 21's task
# calls. No second implementation.
#
# The four p22_r_* inputs are the ranking arms' RUN DIRECTORIES,
# produced by the p22_r_*.yaml training runs at sha 44ad1f7e.
# [FILLED 2026-09-01 from the declare_inputs.py output;
# they were all-zero placeholders until then.] The same hash
# appears for each arm in all three post-run configs -- asserted
# by the generator and by a test.
#
#   bash -lc "cd /home/user/codex/cleft-aesthetics && PYTHONPATH=src \\
#     python -m cleft.run --config configs/p22_diagnostic.yaml"
"""
    body = {
        "schema_version": 1, "phase": "p22", "tier": "keeper", "seed": 1337,
        "inputs": inputs, "task": task,
    }
    return "p22_diagnostic.yaml", header + "\n" + yaml.safe_dump(
        body, sort_keys=False
    )


def shrinkage() -> tuple[str, str]:
    import yaml

    donor = _load(SHRINKAGE_DONOR)
    task = dict(donor["task"])
    inputs = list(donor["inputs"]) + _new_arm_inputs()
    task["arms"] = list(task["arms"]) + _new_arm_entries()
    task["expect_arms"] = len(task["arms"])

    header = f"""\
# PHASE 22 -- SHRINKAGE ON THE FOUR COHORT ARMS
#
# GENERATED by scripts/generate_phase22_postrun_configs.py.
#
# The SAME cross_arm_shrinkage task Phase 21 used, unchanged,
# with the four ranking arms added to the 64 it already measured.
#
# This tests phase22.BOUNDED_HEAD_PREDICTION_REGISTERED
# DIRECTLY: a bounded ranking arm should shrink toward the label
# mean like the other 63; an unbounded one should not. The
# prediction FIRES NO CELL and GATES NOTHING -- it was registered
# to be refuted, and these are the first data that bear on it.
#
# The reading thresholds are the ones Phase 21 ruled and are
# carried verbatim from its config; this run adds arms, it does
# not re-rule anything.
#
# The four p22_r_* inputs are the ranking arms' RUN DIRECTORIES,
# produced by the p22_r_*.yaml training runs at sha 44ad1f7e.
# [FILLED 2026-09-01 from the declare_inputs.py output;
# they were all-zero placeholders until then.] The same hash
# appears for each arm in all three post-run configs -- asserted
# by the generator and by a test.
#
#   bash -lc "cd /home/user/codex/cleft-aesthetics && PYTHONPATH=src \\
#     python -m cleft.run --config configs/p22_shrinkage.yaml"
"""
    body = {
        "schema_version": 1, "phase": "p22", "tier": "keeper", "seed": 1337,
        "inputs": inputs, "task": task,
    }
    return "p22_shrinkage.yaml", header + "\n" + yaml.safe_dump(
        body, sort_keys=False
    )


def contrasts() -> tuple[str, str]:
    import yaml

    donor = _load(DIAGNOSTIC_DONOR)
    by_name = {e["name"]: e for e in donor["inputs"]}
    manifest = donor["task"]["manifest_artifact"]

    # [CORRECTED 2026-09-01] The probe's run stem is READ from
    # ladder.TRADE_OFF_PAIR, not asserted here. It was typed in the
    # first version; it was right, and it was not derived -- and if it
    # had been wrong every primary would have compared against the
    # wrong baseline.
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from cleft import ladder

    probe = ladder.TRADE_OFF_PAIR["best_arm"]["stem"]
    if probe not in by_name:
        raise SystemExit(f"{probe!r} is not declared by {DIAGNOSTIC_DONOR}")
    probe_arm = next(
        a for a in donor["task"]["arms"] if a["name"] == probe
    )

    inputs = [by_name[manifest], by_name[probe]] + _new_arm_inputs()
    arms = [dict(probe_arm, name="vit_paired_mean")] + _new_arm_entries()

    task = {
        "kind": "p22_contrasts",
        "manifest_artifact": manifest,
        "arms": arms,
        "seeds": _seeds(),
        "expect_evaluable": 8,
        # [CORRECTED 2026-09-01] Read from ladder, not typed.
        "winner_sd": float(ladder.TRADE_OFF_PAIR["result"]["vit_sd"]),
        "n_boot": 10000,
        "expect_patients": _n_patients(),
    }
    header = """\
# PHASE 22 -- THE CONTRAST FAMILY, EVALUABLE SUBSET
#
# GENERATED by scripts/generate_phase22_postrun_configs.py.
#
# phase7b.paired_comparison UNCHANGED: the per-seed paired BCa
# and the two-condition PLAN 4.3 verdict come from the shipped
# function, not a second implementation.
#
# THE PAIR LIST IS NOT A CONFIG FIELD. It is derived from
# phase22.contrast_family() -- the locked FIFTEEN -- and filtered
# to those whose BOTH members are declared here. The family
# cannot grow through a config edit.
#
# EIGHT of the fifteen are evaluable now: four primaries (each
# cohort arm vs the 0.2520 probe), two head contrasts (R-all
# bounded vs unbounded, R-clear bounded vs unbounded), and the
# noise-pair contrast at EACH bounding. The other SEVEN wait on
# R-syn and are DEFERRED, not withdrawn -- criterion 8 still owes
# them a verdict before the phase closes.
#
# The four p22_r_* inputs are the ranking arms' RUN DIRECTORIES,
# produced by the p22_r_*.yaml training runs at sha 44ad1f7e.
# [FILLED 2026-09-01 from the declare_inputs.py output;
# they were all-zero placeholders until then.] The same hash
# appears for each arm in all three post-run configs -- asserted
# by the generator and by a test.
#
#   bash -lc "cd /home/user/codex/cleft-aesthetics && PYTHONPATH=src \\
#     python -m cleft.run --config configs/p22_contrasts.yaml"
"""
    body = {
        "schema_version": 1, "phase": "p22", "tier": "keeper", "seed": 1337,
        "inputs": inputs, "task": task,
    }
    return "p22_contrasts.yaml", header + "\n" + yaml.safe_dump(
        body, sort_keys=False
    )


BUILDERS = (diagnostic, shrinkage, contrasts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    drift = []
    for builder in BUILDERS:
        name, text = builder()
        path = CONFIGS / name
        if args.check:
            current = path.read_text(encoding="utf-8") if path.is_file() else ""
            if current != text:
                drift.append(name)
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {name}")
    if args.check:
        if drift:
            print("DRIFT: " + ", ".join(drift))
            return 1
        print(f"{len(BUILDERS)} configs match the generator")
    return 0


if __name__ == "__main__":
    sys.exit(main())
