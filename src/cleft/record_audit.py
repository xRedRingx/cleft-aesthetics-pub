"""Record-versus-artifact agreement, as a standing invariant.

**The question this module makes answerable on demand:** does every arm's
PCC as *written in the records* still equal the PCC you get by recomputing
it from that arm's own prediction CSVs?

That question was answered once, by hand, on 2026-08-31
(``RECORD_ARTIFACT_CHECK``: 66 arms, zero mismatches at the 0.005
threshold, largest delta 0.0007). A one-off answer decays -- it is true of
the records as they stood that morning and says nothing about the next
edit. This module is its durable form.

**The split, and why it is not a suite test end to end**
(``STANDING_CHECK_DESIGN``): the artifact side is per-seed prediction CSVs
under ``runs/keeper/**``, which are CLUSTER-ONLY patient data and are not
on any authoring machine. A suite test therefore *cannot* read them. What
the suite can own is everything except the read:

* ``banked_arm_pcc()`` -- gather each locked arm's banked PCC **from the
  repo's own records**, never from a document, with the source of each
  value recorded beside it;
* ``compare_to_artifact()`` -- the comparison itself, a pure function;
* coverage -- every locked arm resolves to a banked value or is named as
  having none.

The artifact read is a declared job (``run.task_record_artifact_check``),
which reuses the shared readers rather than reimplementing them.
"""

from __future__ import annotations

#: The agreement threshold. **0.005 is the maintainer's, from the one-off run**,
#: and it is a reporting precision bound rather than a statistical one:
#: banked values are quoted to four decimals, so two figures agreeing to
#: better than half a unit in the third decimal agree as far as the record
#: can express. The largest delta the manual pass found was 0.0007 --
#: seven times inside it -- and that delta was itself rounding on a
#: 4-dp quote (``p17_arm_b``).
AGREEMENT_THRESHOLD = 0.005

#: Backbone / init / geometry vocabularies, closed. A name that does not
#: parse into these RAISES rather than resolving to a neighbouring cell:
#: a silent mis-parse would compare an arm against another arm's banked
#: value and report agreement, which is the failure mode this whole module
#: exists to prevent.
BACKBONES = ("vit_b16", "swin_b", "srgnn", "agnet")
INITS = ("imagenet", "scut_original", "scut_masked")


class RecordAuditError(RuntimeError):
    """A banked value could not be resolved, or a comparison is unusable."""


def _ladder_cell(backbone: str, init: str, geometry: str):
    """One ladder cell's (pcc, sd, source) from the Stage D records."""
    from . import ladder

    record, name = (
        (ladder.STAGE_D1_AT_G1, "ladder.STAGE_D1_AT_G1")
        if geometry == "g1"
        else (ladder.STAGE_D_AT_G2, "ladder.STAGE_D_AT_G2")
    )
    inits = list(record["inits"])
    if init not in inits:
        raise RecordAuditError(f"{name}: init {init!r} not in {inits}")
    if backbone not in record["cells"]:
        raise RecordAuditError(f"{name}: backbone {backbone!r} not present")
    index = inits.index(init)
    pcc = float(record["cells"][backbone][index])
    sd = record.get("sd", {}).get(backbone, {}).get(init)
    return pcc, (None if sd is None else float(sd)), (
        f"{name}['cells'][{backbone!r}][{index}]"
    )


def _parse_ladder_name(name: str):
    """(backbone, init, geometry) from a ladder arm name, or None.

    Deliberately strict: the arm name is split on a closed vocabulary and
    anything unrecognised returns None so the caller can fall through to
    the explicit table, rather than being guessed into a nearby cell.
    """
    if not name.startswith("p7_"):
        return None
    body = name[len("p7_"):]
    # strip the stage token (c0, c, d1, d, e, e0 ...)
    parts = body.split("_")
    if not parts:
        return None
    body = "_".join(parts[1:])
    # scheme-suffixed graph arms (native/grid/anatomy/random) are handled
    # by the explicit table, except "native", which IS the ladder cell.
    for suffix in ("_grid", "_anatomy", "_random"):
        if body.endswith(suffix):
            return None
    if body.endswith("_native"):
        body = body[: -len("_native")]
    geometry = None
    for candidate in ("g1", "g2"):
        if body.endswith("_" + candidate):
            geometry = candidate
            body = body[: -(len(candidate) + 1)]
            break
    if geometry is None:
        return None
    backbone = next((b for b in BACKBONES if body.startswith(b)), None)
    if backbone is None:
        return None
    init = body[len(backbone):].lstrip("_")
    if init not in INITS:
        return None
    return backbone, init, geometry


def _explicit_sources() -> dict:
    """Every non-ladder arm, mapped to the record that banks it.

    Each entry is (value, sd, source-string). Values are READ from the
    records here -- none is written out in this module, so a record edit
    moves the check rather than silently disagreeing with it.
    """
    from . import phase11, phase12, phase15, phase16, phase17, roadb
    from .train import graph_cleft

    out: dict = {}

    # ---- Stage E / E0 graph schemes ----------------------------------
    stage_e = graph_cleft.STAGE_E_PER_ARM["arms"]
    for arm, cell in stage_e.items():
        out[arm] = (
            float(cell["pcc"]), float(cell["sd"]),
            f"graph_cleft.STAGE_E_PER_ARM['arms'][{arm!r}]",
        )
    complete = graph_cleft.STAGE_E_COMPLETE
    out["p7_e_srgnn_scut_masked_g2_random"] = (
        float(complete["mean"]), float(complete["sd"]),
        "graph_cleft.STAGE_E_COMPLETE",
    )
    e0 = graph_cleft.SCHEME_AXIS_AT_CLEFT["means"]
    for scheme in ("grid", "anatomy", "random"):
        out[f"p7_e0_srgnn_imagenet_g2_{scheme}"] = (
            float(e0[scheme]), None,
            f"graph_cleft.SCHEME_AXIS_AT_CLEFT['means'][{scheme!r}]",
        )

    # ---- p11 / p12 / p15 / p16 / p17 ---------------------------------
    out["p11_mse_control"] = (
        float(phase11.LOSS_ARMS_OBSERVED["merits"]["pcc"]["control"]), None,
        "phase11.LOSS_ARMS_OBSERVED['merits']['pcc']['control']",
    )
    for arm, record, key, path in (
        ("p12_arm_a_frontal", phase12.ARMS_A_D_OBSERVED, "a_frontal_only_236",
         "phase12.ARMS_A_D_OBSERVED"),
        ("p12_arm_d_capacity", phase12.ARMS_A_D_OBSERVED, "d_capacity_control",
         "phase12.ARMS_A_D_OBSERVED"),
        ("p12_arm_b_basal", phase12.ARMS_B_C_OBSERVED, "b_basal_only_236",
         "phase12.ARMS_B_C_OBSERVED"),
        ("p12_arm_c_concat", phase12.ARMS_B_C_OBSERVED, "c_two_view_concat",
         "phase12.ARMS_B_C_OBSERVED"),
    ):
        cell = record[key]
        out[arm] = (float(cell["mean"]), None, f"{path}[{key!r}]['mean']")

    for arm, key in (
        ("p15_probe_mebeauty_g1", "mebeauty_masked@g1"),
        ("p15_probe_mebeauty_g2", "mebeauty_masked@g2"),
        ("p15_probe_mebeauty_original", "mebeauty_original"),
    ):
        cell = phase15.STOP_4B_BANKED["cells"][key]
        out[arm] = (
            float(cell["pcc"]), float(cell["sd"]),
            f"phase15.STOP_4B_BANKED['cells'][{key!r}]",
        )

    # ---- p16 and p17 --------------------------------------------------
    # [2026-08-31] These five were the gap: their PCCs existed only in
    # prose, so no checker could see them. They now have structured
    # homes -- phase16.ARM_MEANS and phase17.ARM_MEANS -- transcribed
    # from the same closings the prose lives in, with each figure
    # cross-checked against a quantity banked elsewhere. Read from the
    # constants here; the prose is untouched.
    for module, name in ((phase16, "phase16"), (phase17, "phase17")):
        for arm, cell in module.ARM_MEANS["arms"].items():
            out[arm] = (
                float(cell["pcc"]),
                None if cell["sd"] is None else float(cell["sd"]),
                f"{name}.ARM_MEANS['arms'][{arm!r}]",
            )

    # ---- Road B ------------------------------------------------------
    values = roadb.PHASE_7_TWENTY_TWO_ARMS["values"]
    for family, by_res in values.items():
        backbone, init = family.split("__")
        # The LOCK spells the masked init `masked`; the record spells it
        # `scut_masked`. Both names are emitted and only the one the lock
        # actually uses is ever consumed -- bridging the two spellings
        # here beats renaming either, which would edit a banked record or
        # a locked list to suit a checker.
        tokens = {init, "masked"} if init == "scut_masked" else {init}
        for resolution, pair in by_res.items():
            source = (
                f"roadb.PHASE_7_TWENTY_TWO_ARMS['values'][{family!r}]"
                f"[{resolution!r}]"
            )
            for token in tokens:
                out[f"roadb_{backbone}_{token}_{resolution}"] = (
                    float(pair[0]), float(pair[1]), source,
                )
    crop = roadb.REGION_CROP_ARMS_OBSERVED["values"]
    for scheme, cell in crop.items():
        out[f"roadb_rc_{scheme}"] = (
            float(cell["pcc"]), float(cell["sd"]),
            f"roadb.REGION_CROP_ARMS_OBSERVED['values'][{scheme!r}]",
        )
    return out


def _first_float(record: dict, keys) -> float:
    for key in keys:
        if key in record:
            return float(record[key])
    raise RecordAuditError(
        f"none of {list(keys)} is present; the record moved and this "
        "gather must be updated rather than guessing"
    )


def banked_arm_pcc() -> dict:
    """``{arm_name: {"pcc", "sd", "source"}}`` for every locked arm.

    Read from the records, never written out here. An arm the gather
    cannot resolve is returned with ``pcc = None`` and a reason, so a gap
    is visible rather than absent -- ``unresolved()`` is the accessor for
    that, and the suite asserts the set of unresolved arms is exactly the
    one recorded in ``BANKED_VALUE_GAPS``.
    """
    from . import phase18

    explicit = _explicit_sources()
    out: dict = {}
    for entry in phase18.locked_arm_entries():
        name = entry["name"]
        if name in explicit:
            pcc, sd, source = explicit[name]
            out[name] = {"pcc": pcc, "sd": sd, "source": source,
                         "group": entry["group"]}
            continue
        parsed = _parse_ladder_name(name)
        if parsed is not None:
            pcc, sd, source = _ladder_cell(*parsed)
            out[name] = {"pcc": pcc, "sd": sd, "source": source,
                         "group": entry["group"]}
            continue
        out[name] = {"pcc": None, "sd": None, "group": entry["group"],
                     "source": None,
                     "reason": "no banked value found in any record"}
    return out


def unresolved(banked: dict | None = None) -> list:
    """Arm names with no banked PCC anywhere in the records."""
    banked = banked_arm_pcc() if banked is None else banked
    return sorted(n for n, cell in banked.items() if cell["pcc"] is None)


def compare_to_artifact(
    banked: dict, measured: dict, threshold: float = AGREEMENT_THRESHOLD,
) -> dict:
    """Compare banked PCCs against artifact-recomputed ones.

    ``measured`` is ``{arm_name: pcc}`` recomputed from prediction CSVs.
    Returns every arm's delta, the mismatches at ``threshold``, and the
    arms present on one side only -- **a one-sided arm is reported, never
    skipped**: an artifact with no banked value and a banked value with no
    artifact are different failures and both matter.
    """
    if threshold <= 0:
        raise RecordAuditError(f"threshold must be positive, got {threshold}")
    deltas, mismatches = {}, []
    for name, cell in sorted(banked.items()):
        if cell["pcc"] is None or name not in measured:
            continue
        delta = float(measured[name]) - float(cell["pcc"])
        deltas[name] = delta
        if abs(delta) >= threshold:
            mismatches.append({
                "arm": name, "banked": float(cell["pcc"]),
                "measured": float(measured[name]), "delta": delta,
                "source": cell["source"],
            })
    banked_only = sorted(
        n for n, c in banked.items()
        if c["pcc"] is not None and n not in measured
    )
    measured_only = sorted(set(measured) - set(banked))
    largest = max(deltas.items(), key=lambda kv: abs(kv[1]), default=None)
    return {
        "n_compared": len(deltas),
        "threshold": threshold,
        "mismatches": mismatches,
        "agree": not mismatches,
        "largest_delta": (
            None if largest is None
            else {"arm": largest[0], "delta": largest[1]}
        ),
        "banked_without_artifact": banked_only,
        "artifact_without_banked": measured_only,
        "deltas": deltas,
    }


#: **[RECORDED 2026-08-31] THE ONE-OFF RESULT, and it passed.**
#:
#: the maintainer compared every banked arm PCC against Phase 18 run 5's
#: recomputation from the arms' own prediction CSVs
#: (``p18_metric_space__9411467e``-era table; the citable run is
#: ``p18_metric_space__9411267e__p18-metric-space-5``):
#:
#:     arms compared            66
#:     mismatches >= 0.005       0
#:     largest delta        0.0007  (p17_arm_b)
#:
#: The largest delta is **rounding on a 4-dp quote**, not a disagreement:
#: at four decimals 0.0007 is the last-digit gap between a value written
#: down and the same value recomputed.
#:
#: **Why 66 and not 68.** The lock holds 68 arms; the two Stage E cells
#: had no banked value to compare against until
#: ``graph_cleft.STAGE_E_PER_ARM`` banked them **the same day, from this
#: very table**. So the pass covered every arm that *had* a banked
#: figure, and the two it could not cover are now bankable precisely
#: because this comparison produced them. A re-run of the check today
#: would compare 68.
#:
#: **What this result is NOT.** It is a snapshot. It says the records
#: agreed with the artifacts on 2026-08-31 and nothing about the next
#: edit to a record -- which is the whole reason ``STANDING_CHECK_DESIGN``
#: exists.
RECORD_ARTIFACT_CHECK = {
    "recorded": "2026-08-31",
    "source": (
        "the maintainer's comparison of banked arm PCCs against Phase 18 run "
        "5's recomputation from the arms' own prediction CSVs "
        "(p18_metric_space__9411267e__p18-metric-space-5)"
    ),
    "arms_compared": 66,
    "mismatches_at_0_005": 0,
    "largest_delta": {"arm": "p17_arm_b", "delta": 0.0007},
    "largest_delta_reading": (
        "rounding on a 4-dp quote, not a disagreement: 0.0007 is the "
        "last-digit gap between a value written down to four decimals "
        "and the same value recomputed"
    ),
    "why_66_not_68": (
        "the lock holds 68; the two Stage E cells had no banked value to "
        "compare against until graph_cleft.STAGE_E_PER_ARM banked them "
        "the same day FROM THIS TABLE. The pass covered every arm that "
        "had a banked figure, and a re-run today would compare 68"
    ),
    "what_it_is_not": (
        "a SNAPSHOT. It says the records agreed with the artifacts on "
        "2026-08-31 and nothing about the next edit to a record -- which "
        "is why STANDING_CHECK_DESIGN exists"
    ),
    "verdict": "PASS -- zero mismatches at the 0.005 threshold",
    # [2026-08-31] The machinery ran. FIRST_EXECUTION is its result, and
    # it is stricter than this manual pass -- see the precision note
    # there.
    "superseded_as_the_live_check_by": "FIRST_EXECUTION",
}


#: **[OBSERVED 2026-08-31] THE CHECK RAN, AND IT AGREES.**
#:
#: Run ``record_artifact_check__e2d8280f__p7d1-classification-metrics``,
#: keeper tier, on the pod. CPU-only arithmetic over already-written
#: CSVs; the provenance warning for a keeper run outside the pinned
#: image was expected and is self-recorded in the run's ``env.json`` as
#: ``keeper_outside_pinned_image``.
#:
#:     agree                       true
#:     arms compared               63   (at the time; 68 now -- see below)
#:     threshold                   0.005
#:     mismatches                  0
#:     largest delta               5.052e-05  (p7_d_swin_b_scut_original_g2)
#:     banked without artifact     none
#:     artifact without banked     none
#:
#: The five then-uncovered arms were reported by name with their
#: reasons, exactly as the design requires -- never silently dropped.
#: **They are covered now**: ``phase16.ARM_MEANS`` and
#: ``phase17.ARM_MEANS`` closed that gap the same day
#: (``BANKED_VALUE_GAPS_CLOSED``), so a re-run compares **68**.
#:
#: **THE PRECISION DISTINCTION, and it is the reason the machinery
#: exists.** ``RECORD_ARTIFACT_CHECK``'s manual pass found a largest
#: delta of **0.0007**; this run found **5.052e-05**, fourteen times
#: smaller. Nothing about the artifacts changed between them. The manual
#: pass compared against values **parsed to four decimals out of a
#: markdown document**, so its floor was the document's rounding; this
#: run reads **full-precision banked values from the records
#: themselves**. **The machinery is stricter than the document it
#: replaces**, and a check whose floor is a document's rounding cannot
#: see a disagreement smaller than that rounding.
#:
#: **A directory-name oddity, recorded so no reader is misled**: the
#: job-id slot reads ``p7d1-classification-metrics``, which has nothing
#: to do with this run. It was inherited from a stale ``CLEFT_JOB_ID``
#: left in the shell environment. **The task and the sha in the
#: directory name are correct**; only the job-id fragment is wrong, and
#: no result is affected. See ``JOB_ID_STALENESS_HAZARD``.
FIRST_EXECUTION = {
    "observed": (
        "2026-08-31, run "
        "record_artifact_check__e2d8280f__p7d1-classification-metrics, "
        "keeper tier, on the pod"
    ),
    "compute": (
        "CPU-only arithmetic over already-written CSVs; the keeper "
        "run's provenance warning for running outside the pinned image "
        "was expected and is self-recorded in env.json as "
        "keeper_outside_pinned_image"
    ),
    "agree": True,
    "arms_compared": 63,
    "threshold": AGREEMENT_THRESHOLD,
    "mismatches": 0,
    "largest_delta": {
        "arm": "p7_d_swin_b_scut_original_g2", "delta": 5.052e-05,
    },
    "banked_without_artifact": (),
    "artifact_without_banked": (),
    "uncovered_reported_by_name": (
        "the five then-uncovered arms were reported with their reasons, "
        "as the design requires -- never silently dropped. They are "
        "COVERED NOW (BANKED_VALUE_GAPS_CLOSED), so a re-run compares 68"
    ),
    "precision_distinction": (
        "**the reason the machinery exists.** RECORD_ARTIFACT_CHECK's "
        "manual pass found a largest delta of 0.0007; this run found "
        "5.052e-05, fourteen times smaller, with nothing about the "
        "artifacts changed. The manual pass compared against values "
        "PARSED TO FOUR DECIMALS OUT OF A MARKDOWN DOCUMENT, so its "
        "floor was the document's rounding; this run reads "
        "FULL-PRECISION banked values from the records themselves. The "
        "machinery is stricter than the document it replaces, and a "
        "check whose floor is a document's rounding cannot see a "
        "disagreement smaller than that rounding"
    ),
    "directory_name_oddity": "JOB_ID_STALENESS_HAZARD",
    "verdict": "PASS -- records and artifacts agree",
}


#: **[MEASURED 2026-08-31]** -- a recommendation; NOTHING IS EDITED HERE.
#: [TAG NORMALISED 2026-09-01 from '[REPORTED 2026-08-31, a
#: recommendation -- NOTHING IS EDITED HERE]'; the qualifier is prose.]
#: A STALE ``CLEFT_JOB_ID`` SILENTLY MISLABELS A RUN DIRECTORY.**
#:
#: This run landed at
#: ``record_artifact_check__e2d8280f__p7d1-classification-metrics``. The
#: contract is ``<config-stem>__<sha8>__<job-id>`` (PLAN 2.4). The stem
#: and the sha are this run's; **the job id belongs to a different task
#: entirely**, inherited from a ``CLEFT_JOB_ID`` still set in the shell
#: from an earlier launch.
#:
#: **Nothing is wrong with the result** -- the config, the code and the
#: inputs are all this run's. The damage is to LEGIBILITY, and it is the
#: same shape as ``roadb.MISTYPED_LAUNCH_DELETED``: the name carries the
#: identity twice, humans skim the job id, and declarations match the
#: whole string. A reader scanning run directories would file this under
#: the classification-metrics work.
#:
#: **The existing guard does not catch it, and that is measured, not
#: assumed.** ``run_names.check_run_dir`` on this exact name returns
#: **zero contradictions** and passes: its rule is *a job id may DROP
#: what the stem says but may not CONTRADICT it*, enforced over closed
#: axes (resolution, backbone, geometry, init, scheme). Neither
#: ``record_artifact_check`` nor ``p7d1-classification-metrics`` carries
#: a token from any of those axes, so there is nothing to contradict.
#: The guard is working as designed; this failure is simply outside its
#: design.
#:
#: **Recommendation: WARN, do not refuse.** The reasoning is the same
#: measurement that shaped the existing guard -- job ids legitimately
#: abbreviate (``p6-pt-swin-g1-v2`` for
#: ``p6_pretrain_swin_b_masked_g1``), and the literal rule held for only
#: 22 of 83 real directories. A hard refusal on "job id unrelated to
#: config stem" would fire on those abbreviations and be switched off,
#: and *a switched-off guard is worse than none* -- the module's own
#: sentence.
#:
#: **What a warning could honestly check**, at ``RunContext`` setup
#: where the job id is resolved: whether the job id shares **any** token
#: with the config stem. This run shares none, which is the signal;
#: ``p6-pt-swin-g1-v2`` shares ``swin`` and ``g1``. Emitted on the same
#: channel as the existing keeper-outside-pinned-image warning and
#: recorded in ``env.json``, so it is visible at launch and auditable
#: afterwards without blocking anything.
#:
#: **The deeper point, and why a refusal would be wrong anyway**:
#: ``CLEFT_JOB_ID`` is first in ``JOB_ID_ENV_VARS`` precisely so the
#: maintainer can pin an id that survives pod recreation -- stability is
#: the feature. Staleness is that feature's cost, and the fix belongs at
#: the launch script (unset or set it per launch), not in a rule that
#: makes the stable-id mechanism harder to use. **A one-line
#: ``unset CLEFT_JOB_ID`` in the launch wrapper removes the cause; the
#: warning catches the day someone forgets.**
JOB_ID_STALENESS_HAZARD = {
    "reported": "2026-08-31 -- a RECOMMENDATION; nothing is edited",
    "what_happened": (
        "record_artifact_check__e2d8280f__p7d1-classification-metrics: "
        "the stem and sha are this run's, the job id belongs to a "
        "different task, inherited from a CLEFT_JOB_ID still set in the "
        "shell from an earlier launch"
    ),
    "no_result_is_affected": (
        "the config, code and inputs are all this run's. The damage is "
        "to LEGIBILITY -- the roadb.MISTYPED_LAUNCH_DELETED shape: the "
        "name carries identity twice and humans skim the job id"
    ),
    "the_existing_guard_passes_it_measured": (
        "run_names.check_run_dir on this exact name returns ZERO "
        "contradictions. Its rule enforces closed axes (resolution, "
        "backbone, geometry, init, scheme) and neither half carries an "
        "axis token, so there is nothing to contradict. The guard works "
        "as designed; this failure is outside its design"
    ),
    "recommendation": (
        "WARN, do not refuse. Job ids legitimately abbreviate and the "
        "literal rule held for only 22 of 83 real directories; a hard "
        "refusal would fire on those and be switched off, and a "
        "switched-off guard is worse than none (run_names' own "
        "sentence)"
    ),
    "what_the_warning_would_check": (
        "at RunContext setup where the job id resolves: whether the job "
        "id shares ANY token with the config stem. This run shares "
        "none; p6-pt-swin-g1-v2 shares swin and g1. Emitted on the same "
        "channel as the keeper-outside-pinned-image warning and "
        "recorded in env.json -- visible at launch, auditable after, "
        "blocking nothing"
    ),
    "the_real_fix_is_upstream": (
        "CLEFT_JOB_ID is FIRST in JOB_ID_ENV_VARS so an maintainer can "
        "pin an id that survives pod recreation -- stability is the "
        "feature and staleness is its cost. A one-line 'unset "
        "CLEFT_JOB_ID' in the launch wrapper removes the cause; the "
        "warning catches the day someone forgets"
    ),
    "not_built": (
        "reported only, per instruction. Building it touches "
        "provenance/context.py, which is FROZEN apparatus -- an "
        "escalation, not a tidy-up"
    ),
}


#: **[DESIGNED 2026-08-31] THE STANDING FORM, and why it splits.**
#:
#: **It is a DECLARED JOB, not a suite test, and the reason is what the
#: suite can see.** The artifact side is ``seed_<n>__predictions.csv``
#: under ``runs/keeper/**`` -- patient-keyed, CLUSTER-ONLY, and absent
#: from every authoring machine. A suite test cannot read them.
#:
#: **The pin option was considered and is unavailable.** Pinning run 5's
#: recomputed values into the repo would let the suite compare records
#: against the pin -- but that is a **record-versus-record** check: it
#: detects a later record edit drifting from the pin, and can never
#: detect the artifact changing or the original banking being wrong. It
#: would also need the 66 per-arm recomputed values, and **only the
#: summary reached this machine** (66 / 0 / 0.0007), not the per-arm
#: table. So the pin is not merely weaker, it is not currently
#: constructible. **If the maintainer pastes the per-arm values, a pinned
#: regression check becomes possible and should be named for what it is**
#: -- record-versus-snapshot, not record-versus-artifact.
#:
#: **What the suite owns anyway**, because it is the half that can go
#: wrong silently:
#:
#: * ``banked_arm_pcc()`` resolves every locked arm from the RECORDS;
#: * every arm resolves or is named in ``BANKED_VALUE_GAPS`` -- so a
#:   check that quietly covered three arms and passed is impossible;
#: * ``compare_to_artifact()`` has a **teeth test**: a planted mismatch
#:   must fire. A comparison that never runs against a real artifact on
#:   this machine would otherwise be a green light with no evidence
#:   behind it, which is the project's own failure-mode 5.
#:
#: **One implementation.** The job reuses ``cluster_csv.read_cluster_csv``
#: and the frozen ``eval.metrics.pcc`` -- the same reader and the same
#: metric ``task_metric_space_analysis`` uses. It does **not** re-derive
#: either, and it does not touch the Phase 18 task, whose run 5 is
#: citable and whose exit criteria are locked.
STANDING_CHECK_DESIGN = {
    "designed": "2026-08-31",
    "verdict": "A DECLARED JOB, not a suite test",
    "why_not_the_suite": (
        "the artifact side is seed_<n>__predictions.csv under "
        "runs/keeper/**, patient-keyed and CLUSTER-ONLY, absent from "
        "every authoring machine. A suite test cannot read them"
    ),
    "the_pin_option": (
        "CONSIDERED AND UNAVAILABLE. Pinning run 5's recomputed values "
        "would make it a RECORD-VERSUS-RECORD check -- it detects a "
        "later record edit drifting from the pin, never the artifact "
        "changing or the original banking being wrong. And it needs the "
        "66 per-arm values, of which only the SUMMARY reached this "
        "machine, so it is not currently constructible. If those values "
        "arrive, a pinned regression check is possible and must be named "
        "record-versus-SNAPSHOT"
    ),
    "what_the_suite_owns": (
        "banked_arm_pcc() resolving every locked arm from the records; "
        "coverage, so a check that quietly covered three arms cannot "
        "pass; and a TEETH TEST on compare_to_artifact() where a planted "
        "mismatch must fire -- without it, a comparison that never runs "
        "here is a green light with no evidence"
    ),
    "one_implementation": (
        "the job reuses cluster_csv.read_cluster_csv and the frozen "
        "eval.metrics.pcc -- the same reader and metric "
        "task_metric_space_analysis uses. It re-derives neither, and it "
        "does not touch the Phase 18 task, whose run 5 is citable and "
        "whose exit criteria are locked"
    ),
    "job": "run.task_record_artifact_check, configs/record_artifact_check.yaml",
}


#: **[CLOSED 2026-08-31] Locked arms with NO banked PCC in any record --
#: NONE. The gap existed for one day and is recorded rather than
#: quietly emptied.**
#:
#: **The gap, as it stood.** Building the gather on 2026-08-31 surfaced
#: that five of the 68 locked arms had their PCCs **only inside prose**
#: -- ``phase16.PHASE_16_CLOSING``'s criterion text ("identity 0.2151 ->
#: loop 0.2040") and
#: ``phase17.PHASE_17_CLOSING["criterion_2_three_arms_zero_real"]``
#: ("Per-arm means: A 0.2334 (sd 0.0044), B -0.0044 (sd 0.0317), C
#: -0.0185 (sd 0.0586)"), plus the ledger's claim sentences. No
#: structured key held any of them, so the standing check covered 63 of
#: 68 and said so.
#:
#: They were **not** regex-parsed out of that prose -- the project's own
#: rule is to prefer structured constants because grepping docstrings
#: breaks on line reflow, failing for a reason unrelated to the claim.
#:
#: **How it closed, the same day**: ``phase16.ARM_MEANS`` and
#: ``phase17.ARM_MEANS`` bank the five as structured constants,
#: transcribed from those closings with the prose left unedited and a
#: dated pointer beside it. Each figure is cross-checked against a
#: quantity banked elsewhere -- the ledger's paired deltas and the
#: thresholds ``combined_claimable_delta`` derives from the sds -- so a
#: transcription slip fails rather than passing quietly.
#:
#: **Coverage is now 68 of 68.** This tuple stays as an empty *named*
#: constant rather than being deleted: the suite asserts ``unresolved()``
#: equals it exactly, so an arm losing its banked value in a future edit
#: fails loudly instead of silently dropping out of the comparison.
BANKED_VALUE_GAPS: tuple = ()

#: The gap's history, kept after it closed. An empty list with no memory
#: reads as though the coverage was always complete; it was not, for one
#: day, and the reason it closed is a small piece of work someone might
#: otherwise undo.
BANKED_VALUE_GAPS_CLOSED = {
    "opened": "2026-08-31, when building banked_arm_pcc() surfaced it",
    "closed": "2026-08-31, the same day",
    "was": (
        "p16_anchor_loop, p16_identity_baseline, p17_arm_a, p17_arm_b, "
        "p17_arm_c -- five of 68 whose PCCs existed only in prose, in "
        "the phase-16 and phase-17 closings and the ledger claims"
    ),
    "closed_by": (
        "phase16.ARM_MEANS and phase17.ARM_MEANS, transcribed from "
        "those closings (NOT regex-parsed), prose unedited, dated "
        "pointers added beside it"
    ),
    "each_figure_cross_checked": (
        "PCC against the ledger's mean paired deltas (p17 A and C "
        "reproduce exactly; B differs by 0.0006 on the known "
        "paired-vs-pooled distinction) and sd against the thresholds "
        "combined_claimable_delta derives from them (0.0329, 0.0307, "
        "0.0530 exact; 0.0135 vs a banked 0.0136 on 4-dp rounding)"
    ),
    "coverage_now": "68 of 68",
    "why_the_empty_tuple_stays": (
        "the suite asserts unresolved() equals it exactly, so an arm "
        "losing its banked value in a future edit fails loudly rather "
        "than dropping silently out of the comparison"
    ),
}

#: Kept for the same reason as the tuple: the reasons the gap existed,
#: so a future reader can see what "only in prose" meant in practice.
#: Empty now, and the history is in ``BANKED_VALUE_GAPS_CLOSED``.
BANKED_VALUE_GAP_REASONS: dict = {}


#: **[RECORDED 2026-09-01] REPOSITORY POLICY, AND A VIOLATION OF IT.**
#:
#: **THE POLICY.** This repository has **one committer**. The cluster
#: copy is **pull-only** -- it is never written to from an authoring
#: machine -- and **runs are launched by the maintainer**, never as a
#: side effect of preparing them. Authoring machines edit and test;
#: committing, pushing, pulling and launching are separate acts,
#: performed deliberately.
#:
#: **The policy is not a technical guard and cannot be one.** A commit
#: is a local write to a directory the authoring machine can already
#: write to, so nothing in the repository can stop it. **The protocol is
#: the whole of the enforcement**, which is why it is written down.
#:
#: **THE VIOLATION, 2026-09-01.** A commit was made locally at
#: ``9c91064``, followed by ``git pull --ff-only`` and an attempted
#: ``python -m cleft.run`` on ``p21_cohort_pair_separation.yaml``. All
#: three are outside the policy. The instruction that prompted them
#: ended *"Then commit, pull, and run both"*, and that was taken as
#: authorisation. **It is not sufficient authorisation**: a standing
#: policy is not lifted by a sentence in a message, and the correct
#: response was to say so and stop.
#:
#: **WHAT IT COST, measured.** The commit **never reached the remote** --
#: ``git push`` was never run, and whether the commit was ahead of the
#: remote was never checked. The cluster stayed at ``78f559f``, where
#: **neither config existed**. ``git pull``'s *"Already up to date"* was
#: then reported as though it confirmed the repository was in order.
#: **It confirmed only that no NEW commits existed upstream**; it said
#: nothing about the local commit sitting unpushed. The commit was reset
#: and the same work recommitted under the maintainer's own chain as
#: ``48a0c69``.
#:
#: **WHAT STOPPED WHAT.** The run attempt was stopped by **guard 3** --
#: ``GuardError: declared input 'manifest_v1' does not exist`` -- because
#: the cohort data is not on an authoring machine. **The commit was
#: stopped by nothing.**
#:
#: **THE CHECK ADDED, since a guard is impossible.** Two commands are
#: runnable and would have caught it:
#:
#:     git rev-list --count @{u}..HEAD     -> non-zero means unpushed
#:     git status -sb                      -> "[ahead N]" in the header
#:
#: After the violation the first returned **1**; it returns **0** now,
#: and the count is reported at the end of every working session.
#: **Neither is a substitute for not committing** -- a check that catches
#: a policy crossing after the fact is a REPORT, not a GUARD -- but
#: reporting ``git pull``'s output as evidence of a clean state, without
#: either of them, is what turned one error into a false all-clear.
#:
#: **THE SECOND ERROR IS THE WORSE ONE.** Committing outside the policy
#: was a rule broken. **Reporting "pull clean" as though the state were
#: verified was a claim not checked** -- and this project's whole
#: discipline is that a claim in a measured voice is the one nobody
#: re-checks (``ladder.BASAL_RATIONALE_UNSUPPORTED["lesson"]``). It is
#: the same pattern the phase provenance entries record seven times
#: over, arriving here in the repository's own operation rather than in
#: a figure.
#: **[CONTINUED 2026-09-06] THE ANONYMISATION MISSED PRONOUNS, AND THAT
#: IS ONE PASS RATHER THAN TWO.**
#:
#: A continuation of the name sweep, not a new category.
THE_PRONOUNS_THE_NAME_SWEEP_MISSED = {
    "continued": "2026-09-06",
    "of_what": (
        "the anonymisation recorded at ``results_ledger.CHAIN_RE_DERIVED`` "
        "and applied across the tree. It removed party NAMES. It did not "
        "remove the pronouns attached to them"
    ),
    "why_it_was_missed": (
        "**the pass searched for names and never looked at pronouns.** "
        "Its census counted occurrences of the parties by name, its guard "
        "asserts the absence of those names, and both were satisfied "
        "while forty two masculine pronouns stood in the prose beside "
        "them. A guard that checks what a sweep removed cannot find what "
        "the sweep never looked for"
    ),
    "what_was_swept": (
        "**39 of 42 occurrences**, rewritten to the role or to a neutral "
        "construction. Rulings became rulings rather than someone's, "
        "acknowledgements became deliberate edits, and a cited paper's "
        "author became the work. The identifier "
        "the key that asked what turns on the answer went with them, "
        "because a "
        "key name carries a pronoun as loudly as a sentence does"
    ),
    "WHAT_WAS_LEFT_AND_WHY": (
        "**none, and that took a second pass.** The first sweep left "
        "three, all the same quotation: ``phase21`` reproduced the "
        "registration message's empty placeholder as the evidence that "
        "a ruling had been requested and not supplied, and the "
        "placeholder carried a pronoun. **[RULED 2026-09-06] The fact "
        "survives and the artifact does not.** The record now states "
        "directly that the ruling was requested, that the constraint was "
        "given and the rule was left blank, and that it has not been "
        "supplied since. Nothing is reproduced, so nothing needs "
        "excepting, and the guard asserts a clean tree with no exemption "
        "list at all"
    ),
    "the_general_rule_it_sets": (
        "**a pronoun in this record's own prose is swept, and a quoted "
        "artifact is not a reason to keep one.** The first pass kept "
        "three on the ground that rewriting a quotation destroys what it "
        "records. That was true of the quotation and false of the "
        "record: what the passage had to establish was that a ruling was "
        "requested and not supplied, and a sentence saying so "
        "establishes it without reproducing anything. **An exception "
        "that can be written out of existence was never a necessary "
        "exception**"
    ),
    "nothing_ambiguous_was_guessed": (
        "**every replacement resolved to a named role from its own "
        "context.** Four needed reading rather than pattern matching: "
        "a cited paper's author in ``phase21``'s out-of-distribution "
        "comparison became 'that work', a supervision request in "
        "``roadb`` became 'supervision', "
        "a review pass in ``run`` became 'that pass', and a reviewer's "
        "impression in ``phase13`` became 'the reviewer's'. None was a "
        "coin toss and none was left to a default"
    ),
}


#: **[METHOD 2026-09-05] A SEARCH THAT SILENTLY DOES NOT RUN IS NOT A
#: NEGATIVE RESULT.**
#:
#: Found during the pre-publication sweep, and it invalidates part of what
#: earlier sweeps concluded.
THE_GREP_THAT_DID_NOT_RUN = {
    "found": "2026-09-05",
    "what_happens": (
        "**on this machine, a ``git grep`` pattern that BEGINS WITH ``/`` is "
        "rewritten before git sees it.** Git-Bash/MSYS treats a leading "
        "slash as a POSIX path and converts it, so ``git grep '/home/'`` "
        "searches for ``C:/Program Files/Git/home/``. It does not error. It "
        "returns a small number of hits and exit code 0"
    ),
    "the_measurement": (
        "**``git grep '/home/|/Users/'`` returned 1 hit. The same pattern "
        "through ripgrep returned 3,006.** The corrupted form was not "
        "empty, which is what made it credible -- an empty result invites a "
        "second look; a plausible small one does not"
    ),
    "what_it_invalidates": (
        "**any earlier conclusion of the form 'searched the repo, found "
        "nothing' where the pattern began with a slash.** That shape covers "
        "every sweep for absolute POSIX paths, cluster roots and mount "
        "points. Those sweeps were partly blind and their negatives do not "
        "stand; sweeps whose patterns began with a word are unaffected"
    ),
    "the_rule": (
        "**use ripgrep, not ``git grep``, for any pattern beginning with "
        "``/``** -- and cross-check a negative with a second tool before "
        "banking it. The project's own R1 covers this: measure before "
        "asserting. A search is a measurement, and this one had no error bar"
    ),
    "why_it_is_recorded_here_rather_than_fixed": (
        "**there is nothing to fix in the repository.** It is a property of "
        "the shell on one authoring machine, so the only durable form is a "
        "note that survives the person who found it. The alternative is "
        "rediscovering it, and the cost of rediscovering it is a false "
        "negative in a publication sweep"
    ),
    "the_general_shape": (
        "**a tool that fails by returning a plausible answer is worse than "
        "one that fails loudly**, and this record has met the pattern "
        "before: ``git pull``'s 'Already up to date' reported as though it "
        "verified the repository state "
        "(REPOSITORY_POLICY['what_it_cost']), and a green suite in the "
        "wrong environment (``tests/test_environment.py``). Same failure "
        "mode, three instruments"
    ),
}


REPOSITORY_POLICY = {
    "recorded": "2026-09-01",
    "the_policy": (
        "**one committer; the cluster copy is PULL-ONLY; runs are "
        "launched by the maintainer.** Authoring machines edit and test. "
        "Committing, pushing, pulling and launching are separate "
        "deliberate acts, and **an instruction inside a message is not "
        "sufficient authorisation for them**"
    ),
    "the_violation_2026_09_01": (
        "a local commit at 9c91064, a ``git pull --ff-only``, and an "
        "attempted run of p21_cohort_pair_separation.yaml -- all three "
        "outside the policy, on an instruction reading 'Then commit, "
        "pull, and run both'. **A standing policy is not lifted by a "
        "sentence in a message**"
    ),
    "what_it_cost": (
        "**the commit never reached the remote.** ``git push`` was never "
        "run and the ahead-count never checked; the cluster stayed at "
        "78f559f, where neither config existed. ``git pull``'s 'Already "
        "up to date' was reported as though it verified the repository, "
        "when it confirmed only that no NEW commits existed upstream. "
        "The commit was reset and the work recommitted as 48a0c69"
    ),
    "what_stopped_what": (
        "the run was stopped by **guard 3** -- the cohort data is not on "
        "an authoring machine. **The commit was stopped by nothing**, "
        "because a commit is a local write and no guard in the "
        "repository can prevent one"
    ),
    "the_check_added": (
        "``git rev-list --count @{u}..HEAD`` (non-zero means unpushed) "
        "and ``git status -sb``. It returned 1 after the violation and "
        "returns 0 now, and **the count is reported at the end of every "
        "working session**. A check that catches a crossing after the "
        "fact is a REPORT, not a GUARD -- the policy remains the "
        "enforcement"
    ),
    "the_second_error_is_the_worse_one": (
        "**committing outside the policy was a rule broken; reporting "
        "'pull clean' as though the state were verified was a claim not "
        "checked.** The project's discipline is that a claim in a "
        "measured voice is the one nobody re-checks "
        "(ladder.BASAL_RATIONALE_UNSUPPORTED['lesson'])"
    ),
    "what_happens_when_a_message_says_commit": (
        "**the policy is cited and the work stops there.** Not 'ask "
        "whether it still applies' -- a standing policy re-opened on "
        "each request is not standing. The work is finished when the "
        "edits are made and the suite is green; the report says what is "
        "uncommitted and stops"
    ),
}


#: **[RECORD HYGIENE 2026-09-02] FLEISS KAPPA IS UNWEIGHTED, AND THE
#: RECORD NEVER SAID SO.**
#:
#: Raised by an external review. **The figure is right and its
#: interpretation is missing** -- the same shape as the two-view
#: mis-tagging: a reader starting from the record gets the misleading
#: reading, and nothing in the record corrects it.
THE_KAPPA_LIMITATION = {
    "recorded": "2026-09-02, from an external critique",
    "the_limitation": (
        "**Fleiss kappa is UNWEIGHTED. It scores a 1-versus-2 "
        "disagreement as identically wrong to a 1-versus-5** -- every "
        "non-identical pair of grades counts the same. On a five-point "
        "ORDINAL scale where 1 is excellent and 5 is very poor, that "
        "discards the thing the scale is for: **kappa is the wrong "
        "statistic for these data**, and 0.1662 is a lower figure than "
        "the panel's actual concordance because of it"
    ),
    "the_distance_aware_figures_beside_it": (
        "**both are already computed on the same 237 x 5 matrix and both "
        "are banked.** ``QWK_237 = 0.4276`` -- mean pairwise QUADRATIC "
        "weighted kappa, whose weights are ``(i - j)^2 / (n - 1)^2``, so "
        "a one-grade disagreement costs 1/16 of a four-grade one. "
        "``MEAN_R_237 = 0.4696`` -- mean inter-rater Pearson r, which is "
        "fully distance-aware by construction. **The right statistics "
        "were there all along; only the sentence saying which is which "
        "was missing**"
    ),
    "NO_RESULT_CHANGES_and_this_is_verified": (
        "**kappa gates nothing and enters no arithmetic.** "
        "``FLEISS_237`` is defined, placed in "
        "``FLEISS_BY_POPULATION`` (used only for a population check and "
        "in tests), rendered by ``data.report``, and quoted in prose. "
        "**[CORRECTED 2026-09-03: this said ``run.py`` does not "
        "reference Fleiss AT ALL. True when written; false the same "
        "day.]** ``task_rater_icc`` now recomputes it as a CROSS-CHECK "
        "-- it refuses to report an ICC unless the matrix reproduces "
        "the banked Fleiss, QWK and r, which verifies the panel rather "
        "than gating a decision. **Kappa still gates nothing.** Every "
        "derived quantity in the project -- "
        "reliability 0.8158, the ceiling 0.9032, the ROPE half-width, "
        "every seed threshold, Phase 24's curve -- comes from "
        "``MEAN_R_237``. **So this is a presentation defect, not a "
        "measurement defect, and no figure moves**"
    ),
    "where_it_nonetheless_bit": (
        "**two of the ten sites cite kappa ALONE as the motivation for a "
        "phase**: ``phase12`` ('Phase 14 (LDL) attacks the measured "
        "central problem -- label noise, Fleiss kappa 0.1662 -- "
        "directly') and ``train/ldl.py`` ('Fleiss kappa 0.1662 is the "
        "research problem'). **Only ``ladder.py:1528`` pairs it with "
        "QWK.** A motivation resting on the unweighted figure alone "
        "overstates the disagreement it is motivated by"
    ),
    "the_shape_it_shares_with_the_two_view_claim": (
        "**right figures, missing interpretation.** The two-view claim "
        "carried a ``[MEASURED]`` tag that was never earned; this "
        "carries a correct number with an unstated property that "
        "reverses how a reader weighs it. In both cases **the record is "
        "not wrong and is nonetheless misleading**, and in both the fix "
        "is a sentence rather than a recomputation"
    ),
    "the_ten_sites_carry_a_POINTER_not_an_edit": (
        "**none of the ten is rewritten.** Each gains a dated pointer to "
        "this record, so the original wording stands as written and a "
        "reader meeting kappa anywhere is one line from the "
        "qualification. Sites: ``ladder`` 1528, ``phase12`` 1815, "
        "``phase22`` 295, ``phase24`` 534, ``phase9`` 182 and 238, "
        "``data/reliability`` (the constants and ``fleiss_kappa``), "
        "``data/report`` (the rendered line), ``train/ldl`` 8"
    ),
    "what_it_does_NOT_say": (
        "**kappa is not withdrawn and is not wrong.** It is the standard "
        "index for categorical agreement and the 0.1662 is correctly "
        "computed. What is corrected is the implication a reader draws "
        "from it unaccompanied"
    ),
    "tag": "[RECORD HYGIENE] -- no result changes",
}


#: **[REGISTERED 2026-09-02, BEFORE THE NUMBER EXISTS] The ICC(2,k)
#: prediction.**
#:
#: **Registered before computing, because the prediction is the point**:
#: an ICC that matches Spearman-Brown says one thing and an ICC well
#: below it says another, and deciding which afterwards would be reading
#: the meaning off the result.
ICC_PREDICTION_REGISTERED = {
    "registered": "2026-09-02, before the computation",
    "the_two_quantities_measure_different_things": (
        "**Spearman-Brown on mean pairwise r is BLIND TO RATER BIAS.** "
        "Pearson r is invariant to an additive offset, so **a rater who "
        "is consistently harsh but perfectly ordered contributes FULLY "
        "to r and NOTHING to absolute agreement**. ICC(2,k) treats "
        "raters as a random effect and puts their between-rater variance "
        "in the denominator, so exactly that offset is penalised"
    ),
    "the_prediction": (
        "**ICC(2,k) <= 0.8158 is expected, and THE GAP MEASURES "
        "SYSTEMATIC RATER OFFSET.** Equality would mean the five raters "
        "sit on the same scale as well as in the same order"
    ),
    "nothing_in_the_record_measures_offset_today": (
        "**the per-rater item-total figures are about ORDERING** -- "
        "``item_total`` correlates each rater against the mean of the "
        "other four (Pearson and Spearman), both offset-invariant. So a "
        "systematically harsh or lenient rater is invisible to every "
        "per-rater figure the record holds. **This is the first "
        "measurement of the panel's absolute agreement**"
    ),
    "reading_a_close_agreement": (
        "**no meaningful offset; the banked reliability stands.** 0.8158 "
        "and the 0.9032 ceiling are then correct under both definitions "
        "and the write-up needs no qualification"
    ),
    "reading_a_materially_lower_ICC": (
        "**the panel carries systematic bias, the reliability of its "
        "mean is LOWER than banked, and the ceiling drops with it.** "
        "That does not withdraw 0.8158 -- it is correct for what it "
        "measures -- but it makes the ceiling a RANGE rather than a "
        "point (THE_CEILING_AS_A_RANGE)"
    ),
    "what_counts_as_materially_lower": (
        "**not pre-specified, deliberately.** No threshold exists in the "
        "record for 'how much offset matters', and inventing one now to "
        "grade the result afterwards is the shape this project refuses. "
        "**The gap is reported as a number and its size is the maintainer's to "
        "read**"
    ),
    "tag": "[REGISTERED] -- before the number",
}


#: **[REPORTED 2026-09-02, NOT CHOSEN] Which ICC form is appropriate.**
ICC_FORM_DISTINCTION = {
    "reported": "2026-09-02 -- the distinction, not the choice",
    "icc_2k_raters_as_a_RANDOM_effect": (
        "the five raters are a **SAMPLE of possible raters**, and the "
        "figure generalises to a panel of five drawn from the same "
        "population. Between-rater variance counts AGAINST the "
        "reliability, because a different five would bring different "
        "offsets"
    ),
    "icc_3k_raters_as_a_FIXED_effect": (
        "these five raters ARE the population of interest, and the "
        "figure describes THIS panel. Between-rater variance is removed "
        "rather than penalised. **[CORRECTED 2026-09-03: this said "
        "ICC(3,k) >= ICC(2,k) ALWAYS. That is FALSE.]** The two differ "
        "only in the term ``(MSC - MSE) / n`` in ICC(2,k)'s "
        "denominator, so **ICC(3,k) >= ICC(2,k) exactly when MSC >= "
        "MSE** -- when the between-rater mean square exceeds the "
        "residual. That holds whenever the raters carry any real "
        "offset, and REVERSES by sampling noise when they carry none, "
        "which is how the correction was found: a synthetic panel with "
        "no offset put ICC(2,k) 0.0007 ABOVE ICC(3,k) and the test "
        "asserting the ordering failed"
    ),
    "what_the_choice_turns_on": (
        "**whether the write-up claims a property of THIS panel or of "
        "a five-rater panel in general.** A methods paper reporting how "
        "reliably five mixed-discipline raters score cleft outcome wants "
        "(2,k); a paper describing this cohort's own labels wants (3,k)"
    ),
    "what_the_record_leans_toward_and_why_that_is_NOT_a_choice": (
        "the project's framing has been generalising throughout -- the "
        "attenuation ceiling, the Spearman-Brown projection to other k, "
        "and phase24's 'how many raters would this cohort need' all "
        "treat raters as exchangeable draws, which is (2,k)'s "
        "assumption. **That is an observation about existing framing, "
        "not a ruling**: the choice is the maintainer's and both are computed "
        "so neither has to be re-derived later"
    ),
    "the_relation_to_spearman_brown": (
        "**Spearman-Brown at k on the mean inter-rater r is closely "
        "related to ICC(3,k) under its assumptions, and NOT identical to "
        "it** -- one averages pairwise Pearson correlations, the other "
        "decomposes variance. The record does not claim the identity; "
        "both are reported and the reader is told which is which"
    ),
    "tag": "[REPORTED] -- the choice is not made here",
}


#: **[REPORTING FORM 2026-09-02 -- NOT A NEW FIGURE] The ceiling as a
#: range.**
#:
#: **The existing 0.9032 stands as computed and is NOT withdrawn.** A
#: range is a more honest presentation of the SAME quantity under two
#: reliability definitions, not a replacement for it.
THE_CEILING_AS_A_RANGE = {
    "registered": "2026-09-02, as a reporting form",
    "the_form": (
        "**[sqrt(ICC(2,k)), sqrt(0.8158)]**, or whichever ordering the "
        "numbers give -- the prediction is ICC(2,k) <= 0.8158, so the "
        "ICC endpoint is expected to be the LOWER one, but the form is "
        "written to be ordered by the values rather than by the "
        "expectation (``ICC_PREDICTION_REGISTERED``)"
    ),
    "which_derivation_each_endpoint_comes_from": (
        "**the record must say, and this is the part that makes it "
        "honest rather than merely wider.** The UPPER endpoint is "
        "``sqrt(RELIABILITY_237)`` = 0.9032, where the reliability is "
        "Spearman-Brown on the mean pairwise Pearson r -- **blind to "
        "rater offset**. The LOWER endpoint is ``sqrt(ICC(2,k))``, where "
        "the reliability is a two-way random-effects ICC -- **offset "
        "penalised**. A range whose endpoints are not attributed is two "
        "numbers, not an interval"
    ),
    "what_it_is_NOT": (
        "**not a confidence interval and not an uncertainty band.** "
        "Neither endpoint has sampling error attached; both are point "
        "estimates under different definitions of the same word. "
        "Presenting it as a CI would be a different quantity under one "
        "name -- the R2 shape"
    ),
    "the_existing_figure_is_not_withdrawn": (
        "**0.9032 is correct for what it measures** and every use of it "
        "stands: the attenuation argument, phase24's curve, and "
        "``A_CEILING_IS_NOT_A_SCORE``. What changes is that a write-up "
        "quoting a single ceiling now has to say WHICH reliability it "
        "came from"
    ),
    "it_waits_on_the_number": (
        "**the lower endpoint does not exist yet.** ``configs/"
        "p1b_rater_icc.yaml`` computes it; until it runs this is a FORM "
        "with one endpoint, and the record says so rather than "
        "presenting a range it cannot fill"
    ),
    "and_the_prohibition_still_governs": (
        "a range is still a ceiling. ``phase24.A_CEILING_IS_NOT_A_SCORE`` "
        "applies to both endpoints: **raising or lowering what a model "
        "COULD reach says nothing about what one WOULD reach**, and the "
        "project's own evidence is that the gap to the ceiling is the "
        "binding constraint -- 0.2520 against either endpoint"
    ),
    "tag": "[REPORTING FORM] -- no figure replaced, one endpoint pending",
}


#: **[RULED 2026-09-03] ICC(2,k) IS PRIMARY.**
#:
#: **The choice went against the more flattering number**, which is the
#: part worth recording.
ICC_FORM_RULED = {
    "ruled": "2026-09-03 -- ICC(2,k) primary",
    "the_ground": (
        "**the dissertation asks whether the TASK is automatable, not "
        "whether these five individuals are predictable.** ICC(3,k) "
        "treats the panel as the population of interest, which would "
        "mean the results generalise to no other panel -- **a strong "
        "claim, and a much larger concession than the 0.008 it buys**. "
        "ICC(2,k) treats the raters as a sample of possible raters, "
        "which is what a generalisable claim requires"
    ),
    "it_was_made_KNOWING_the_other_is_higher": (
        "**ICC(3,k) 0.8218 against ICC(2,k) 0.8069; ceiling 0.9065 "
        "against 0.8983.** The ruling was made with both numbers "
        "visible, so **the choice went against the more flattering "
        "one** -- recorded because a form chosen after seeing which is "
        "larger is a different act from one chosen on its assumptions, "
        "and this was the latter"
    ),
    "what_it_costs": (
        "**0.0149 of reliability and 0.0082 of ceiling** -- the latter on the "
        "reported four-decimal endpoints (0.0083 at full precision). Both are "
        "smaller than the difference between any two of the three "
        "derivations and vastly smaller than the gap to 0.2520 "
        "(THE_CEILING_WAS_NEVER_THE_BINDING_CONSTRAINT)"
    ),
    "what_it_does_NOT_settle": (
        "**ICC(3,k) is not wrong and is not withdrawn.** It is the "
        "correct figure for a claim about THIS panel, and it is reported "
        "beside the primary so a reader who wants that claim has it. "
        "What is ruled is which one the write-up leads with"
    ),
    "tag": "[RULED] -- the form, not the figures",
}


#: **[REPORTING FORM 2026-09-03, BOUND] The primary form for reliability
#: and the ceiling.**
THE_PRIMARY_REPORTING_FORM = {
    "bound": "2026-09-03",
    "reliability": (
        "**ICC(2,k) = 0.8069 is PRIMARY.** Reported beside it, each with "
        "its derivation named: **Spearman-Brown 0.8158** (on the mean "
        "pairwise Pearson r, blind to rater offset) and **ICC(3,k) "
        "0.8218** (two-way ANOVA, raters fixed, offset removed rather "
        "than penalised). ICC(2,k) is the two-way ANOVA with raters "
        "random, offset penalised"
    ),
    "the_ceiling_is_a_RANGE": (
        "**[0.8983, 0.9065], with 0.9032 inside it.** **Lower endpoint "
        "= sqrt(ICC(2,k))** -- raters random, offset penalised. **Upper "
        "endpoint = sqrt(ICC(3,k))** -- raters fixed, offset removed. "
        "**The banked 0.9032 = sqrt(Spearman-Brown 0.8158)** sits "
        "between them, which is the arithmetic saying what the prose "
        "should: Pearson r ignores offset rather than removing or "
        "penalising it, so it lands in the middle"
    ),
    "the_endpoints_verify": (
        "sqrt(0.8069) = 0.8983, sqrt(0.8218) = 0.9065, sqrt(0.8158) = "
        "0.9032 -- **all three reproduce to four decimals**, checked "
        "before banking"
    ),
    "0_9032_STANDS_and_is_not_withdrawn": (
        "**it is one of three defensible derivations of the same "
        "quantity, not an error.** Every use of it stands: the "
        "attenuation argument, phase24's curve, "
        "``phase24.A_CEILING_IS_NOT_A_SCORE``. What changes is that a "
        "write-up quoting a single ceiling now says which reliability it "
        "came from"
    ),
    "the_prohibition_governs_all_three_endpoints": (
        "a range is still a ceiling. **Raising or lowering what a model "
        "COULD reach says nothing about what one WOULD reach**, and "
        "0.2520 sits far below every endpoint"
    ),
}


#: **[MEASURED 2026-09-03] SYSTEMATIC RATER OFFSET -- the project's first
#: measurement of rater BIAS.**
THE_RATER_OFFSET_MEASURED = {
    "measured": "2026-09-03, from the ICC run",
    "the_finding": (
        "**systematic offset EXISTS and is REAL, and it is SMALL against "
        "patient variance.** The between-rater mean square is **2.35x "
        "the residual** (MS_R 1.1832 against MS_E 0.5028), so the raters "
        "do differ systematically from one another rather than only by "
        "noise. Against the between-patient term (MS_P 2.6465) it is "
        "**0.45x**, which is what makes it small where it matters"
    ),
    "it_is_the_FIRST_measurement_of_bias_in_the_project": (
        "**every prior rater figure measures agreement on ORDERING, not "
        "OFFSET.** ``item_total``'s Pearson and Spearman, "
        "``mean_inter_rater_r``, and QWK are all invariant to an "
        "additive shift -- **a consistently harsh or lenient rater is "
        "invisible to every one of them**. Nothing before this could "
        "have detected one"
    ),
    "the_sentence_the_write_up_can_now_make": (
        "**'we measured systematic rater offset and it is small'** -- "
        "which it previously could not, because it had no instrument "
        "that could see offset at all. **The absence of a measurement is "
        "not evidence of absence**, and until now that is all the record "
        "had"
    ),
    "and_the_prediction_held": (
        "``ICC_PREDICTION_REGISTERED`` predicted ICC(2,k) <= 0.8158 with "
        "the gap measuring offset. **ICC(2,k) 0.8069 is below 0.8158 by "
        "0.0089**, and the gap is in the predicted direction. The "
        "prediction was registered before the number and is not amended"
    ),

    # ---- the check that does not close --------------------------------
    "THE_MEAN_SQUARES_DO_NOT_RECONCILE_WITH_THE_ICCS": (
        "**[REPORTED, NOT BANKED AS CONSISTENT.]** From MS_P 2.6465, "
        "MS_R 1.1832 and MS_E 0.5028 the two-way formulas give "
        "**ICC(3,k) = 0.8100 and ICC(2,k) = 0.8091**, not the reported "
        "0.8218 and 0.8069. **The three mean squares and the two ICCs "
        "are each internally coherent and do not reconcile with each "
        "other**: the mean squares reproduce the 2.35x ratio exactly "
        "(1.1832 / 0.5028 = 2.3532), and the ICCs reproduce their "
        "ceilings exactly (sqrt gives 0.8983 and 0.9065). **What they "
        "disagree about is the SIZE of the offset penalty** -- the mean "
        "squares imply a 0.0009 gap between the two forms and the ICCs "
        "show 0.0149"
    ),
    "what_that_does_and_does_not_affect": (
        "**The ruling and the reporting form do not depend on it.** Both "
        "rest on the two ICCs, which are self-consistent with their "
        "ceilings, and on the DIRECTION of the offset, which every "
        "figure agrees on. **What is unsettled is the mean squares' own "
        "values** -- so the 2.35x is reported as [REPORTED] pending the "
        "run's ``metrics.json``, and the qualitative finding (offset "
        "exists, offset is small against patient variance) stands on "
        "either set"
    ),
    "why_it_is_flagged_rather_than_resolved": (
        "**the laptop cannot open the run.** Resolving it means reading "
        "``metrics.json``'s own ``ms_rows``, ``ms_cols`` and "
        "``ms_error`` -- which the task writes -- rather than deciding "
        "here which of two coherent sets is right. **A figure that fails "
        "an internal check is not banked as though it passed** "
        "(THE_FABRICATED_FIGURES_WITHDRAWN, one cycle ago)"
    ),
    "a_NAMING_hazard_worth_recording": (
        "**MS_R reads as 'rows' in the code and means 'raters' in the "
        "report.** ``icc_two_way`` returns ``ms_rows`` (between "
        "PATIENTS), ``ms_cols`` (between RATERS) and ``ms_error``; the "
        "reporting convention is MS_P, MS_R, MS_E. **So ``ms_rows`` is "
        "MS_P and ``ms_cols`` is MS_R** -- the same three letters "
        "pointing at different terms. Recorded before it causes a "
        "transposition"
    ),
    "tag": "[MEASURED] -- direction certain, mean squares [REPORTED]",
}


#: **[FRAMING 2026-09-03] The ceiling was never the binding constraint,
#: and it is now demonstrated rather than asserted.**
THE_CEILING_WAS_NEVER_THE_BINDING_CONSTRAINT = {
    "recorded": "2026-09-03",
    "the_arithmetic": (
        "**three defensible reliability definitions span 0.008** "
        "(0.8069, 0.8158, 0.8218) and their ceilings span 0.008 "
        "(0.8983, 0.9032, 0.9065). **The gap from the best arm's 0.2520 "
        "to any of those ceilings is about 0.65** -- roughly eighty "
        "times the spread between the definitions"
    ),
    "what_it_demonstrates": (
        "**the ceiling was never the binding constraint, and this is now "
        "MEASURED rather than argued.** ``phase24.A_CEILING_IS_NOT_A_"
        "SCORE`` asserted it from the gap; this shows that even the "
        "CHOICE of ceiling definition -- a choice with real "
        "methodological content -- moves the bound by an amount "
        "invisible beside the distance still to travel"
    ),
    "it_is_a_REPORTING_correction_not_a_sensitivity_check": (
        "**a figure quoted as a single number for months has three "
        "derivations, none of them wrong.** That is a statement about "
        "how the project REPORTS, not a robustness result about a "
        "conclusion -- no conclusion depended on which derivation was "
        "used, which is precisely why nobody noticed there were three. "
        "**The write-up should say so rather than pick one silently**"
    ),
    "what_it_does_NOT_license": (
        "**it does not make the ceiling irrelevant.** A bound that is "
        "far away is still a bound, and the attenuation argument -- that "
        "no model can correlate perfectly with a noisy target -- is "
        "unaffected. What is licensed is not treating the third decimal "
        "of a reliability figure as though a result turned on it"
    ),
}
