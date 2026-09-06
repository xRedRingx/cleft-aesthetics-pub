"""Write the paired-claim configs from ``cleft.ladder``.

    PYTHONPATH=src python scripts/generate_paired_claim_configs.py [--check]

Two configs, one per scope. **The pair list is NOT in either of them** -- it
is derived from ``ladder.paired_claim_pairs``, where the suite checks it, for
the reason ``generate_phase7c_configs.py`` gives: a comparison list a config
could edit is not a pre-registration. What a config carries is the scope and
the declared vectors.

**Resolved paths and hashes are carried forward by name**, so re-running this
after the cluster paste does not revert them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

from cleft import ladder, phase7c  # noqa: E402

CLUSTER_ROOT = "/home/user/codex/cleft-aesthetics"
RUNS_ROOT = f"{CLUSTER_ROOT}/runs/keeper/p7"
PLACEHOLDER = "0" * 64

SCOPES = ("headline", "ladder", "tradeoff", "p7d")


def run_root_for(stem: str) -> str:
    """A stem's runs directory follows its phase. The 7D and Road B stems
    joined with the p7d scope; every earlier scope is p7-only, so those
    configs cannot change under this routing."""
    if stem.startswith("p7d_"):
        return f"{CLUSTER_ROOT}/runs/keeper/p7d"
    if stem.startswith("roadb_"):
        return f"{CLUSTER_ROOT}/runs/keeper/roadb_p7"
    return RUNS_ROOT


def hashes_by_path() -> dict:
    """Every prediction vector any shipped config already declares, by path.

    **A path's contents are immutable, so one hash serves every config
    declaring it** -- which is exactly what ``declare_ladder_inputs.py`` says
    when it pastes by path rather than per config, and why the cross-config
    invariant exists. Reusing here satisfies that invariant BY CONSTRUCTION
    instead of by remembering to update the sibling, which is the one failure
    it has already had.

    The practical effect: a new scope over arms that are already declared
    needs no cluster round trip at all.
    """
    known: dict[str, dict] = {}
    for path in sorted((REPO / "configs").glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for entry in payload.get("inputs") or []:
            rollup = entry.get("rollup_sha256")
            # Prediction vectors only. Other declared inputs (embedding
            # directories, manifests) are not run-directory-shaped and are
            # not what any scope here consumes.
            if (
                rollup and rollup != PLACEHOLDER
                and "*" not in entry["path"]
                and entry["path"].endswith("__predictions.csv")
                and entry["path"].count("/") >= 2
            ):
                known.setdefault(entry["path"], entry)
    return known


def config_name(scope: str) -> str:
    # "p7_paired_p7d" would stutter and mis-file the phase; the 7D scope
    # names itself the way the other 7D configs do.
    return "p7d_paired" if scope == "p7d" else f"p7_paired_{scope}"


def existing_inputs(scope: str) -> dict:
    """Entries the cluster has RESOLVED, keyed by name.

    **[MEASURED 2026-08-04] Placeholders are deliberately not carried
    forward.** Carrying everything meant a generator change could not reach an
    already-written config: pinning ``p7_e_srgnn_scut_masked_g2_random`` to its
    job id produced no change at all, because the old unpinned glob was
    preserved. What must survive regeneration is what cannot be re-derived
    here -- a resolved path and a real hash. A placeholder is by definition
    re-derivable, and preserving it freezes whatever rule wrote it.

    **[EXTENDED 2026-08-15] A resolved PATH with a placeholder hash is
    carried too** -- the p7d vectors resolve in two passes (run directories
    pasted from the listing, hashes from the declare pass after), so between
    the passes the path is the thing that cannot be re-derived while the
    hash still can. A PENDING path or a glob is still never carried; the
    2026-08-04 rule keyed on the wrong half of the entry, not the wrong
    idea. Same three-state shape as generate_roadb_configs'
    existing_entries(), arrived at from the same flow.
    """
    path = REPO / "configs" / f"{config_name(scope)}.yaml"
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        entry["name"]: entry
        for entry in payload.get("inputs", [])
        if entry.get("rollup_sha256") != PLACEHOLDER
        or (
            "/PENDING_" not in entry["path"] and "*" not in entry["path"]
        )
    }


def inputs_for(scope: str) -> list[dict]:
    """One declared vector per ``(stem, seed)`` the scope needs.

    Paths are GLOBS on the SHA and the job id both, because unlike Phase 7C
    these runs were not re-run under colliding stems -- each ladder config
    stem should name exactly one keeper run. If any of them does not,
    ``declare_ladder_inputs.py`` refuses it as AMBIGUOUS and lists the
    candidates, which is the answer we want rather than a guess.
    """
    carried = existing_inputs(scope)
    known = hashes_by_path()
    # A stem another config has already resolved: reuse its run directory's
    # FULL prefix so the path matches byte-for-byte and the hash carries
    # over. [CORRECTED 2026-08-15] This used to rebuild the path as
    # RUNS_ROOT + run_dir, which was only right for p7 stems -- a roadb or
    # p7d stem resolved elsewhere would have produced a path under the wrong
    # runs directory, with a placeholder where a known hash existed.
    resolved_prefix = {}
    for path in known:
        directory, _, _ = path.rpartition("/")
        run_dir = directory.rpartition("/")[2]
        resolved_prefix.setdefault(run_dir.split("__")[0], directory)

    entries = []
    for stem, seed in ladder.paired_claim_vectors(scope):
        name = (
            f"{phase7c.OOF_INPUT_PREFIX}{stem}"
            f"{phase7c.OOF_SEED_SEPARATOR}{seed}"
        )
        if name in carried:
            entries.append(carried[name])
            continue
        # [CHANGED 2026-08-15] Unresolved paths take Road B's PENDING form,
        # not a glob. A glob like `stem__*__*` has the run-directory SHAPE,
        # so the mistyped-launch sweep (run_names.check_run_dir over every
        # declared run dir) rejects `*` as a sha -- correctly. PENDING_<stem>
        # has no `__`, stays outside the shape until the path is real, and
        # is the form the roadb paired config already resolved through:
        # paste the run directories, then declare.
        path = (
            f"{run_root_for(stem)}/PENDING_{stem}/"
            f"seed_{seed}__predictions.csv"
        )
        if stem in resolved_prefix:
            path = f"{resolved_prefix[stem]}/seed_{seed}__predictions.csv"
        entries.append({
            "name": name,
            "path": path,
            "rollup_sha256": known.get(path, {}).get(
                "rollup_sha256", PLACEHOLDER
            ),
        })
    return entries


def header(scope: str, inputs: list[dict], pairs: list[dict]) -> str:
    unresolved = sum(1 for e in inputs if e["rollup_sha256"] == PLACEHOLDER)
    pending_paths = sum(1 for e in inputs if "/PENDING_" in e["path"])
    if pending_paths:
        provenance = (
            f"# **THE {unresolved} HASHES BELOW ARE ALL-ZERO PLACEHOLDERS and\n"
            f"# {pending_paths} paths are PENDING_<stem> placeholders.** A run\n"
            "# directory carries a SHA and a job id, neither derivable on a\n"
            "# laptop -- and a glob would wear the run-directory shape the\n"
            "# mistyped-launch sweep checks, so the pending form stays\n"
            "# outside it. Resolve in two passes, the Road B flow: paste the\n"
            "# run directories from the run listing (paths), then\n"
            "# declare_inputs.py (hashes). Guard 3 refuses the run until\n"
            "# both are real -- the placeholder working, not a mistake.\n"
        )
    elif unresolved:
        provenance = (
            "# **PATHS RESOLVED, HASHES PENDING.** Every path is a real run\n"
            f"# directory (each one passed run_names.check_run_dir), and the\n"
            f"# {unresolved} remaining hashes are all-zero placeholders until\n"
            "# declare_inputs.py verifies the files. Guard 3 refuses the run\n"
            "# until they are real -- the second pass of the Road B flow.\n"
        )
    else:
        provenance = (
            f"# **RESOLVED.** All {len(inputs)} vectors declared at real paths\n"
            "# with verified hashes. Guard 3 checks every one before a row is\n"
            "# read.\n"
        )
    scope_note = (
        "# **SCOPE: tradeoff.** Phase 8's §0, tested BEFORE the phase is\n"
        "# designed around it. The brief states 'the model that performs best\n"
        "# is the least interpretable, and the models built to be\n"
        "# interpretable perform worse' -- a comparative claim of exactly the\n"
        "# size the same brief's §2 warns is unresolvable here, in the section\n"
        "# naming the phase's contribution.\n"
        "#\n"
        "# It is NOT one of the 26 audited: it varies backbone, init and\n"
        "# geometry at once. The ladder headline had that shape at a SMALLER\n"
        "# delta (0.0428) and came back 0 of 5.\n"
        "#\n"
        "# The interpretable arm is SR-GNN's BEST cell (0.1719, imagenet G1,\n"
        "# native) rather than the brief's masked-G2 arm B at 0.1507: it is\n"
        "# the arm the trade-off sentence cites, it keeps the native region\n"
        "# scheme, and both arms at G1 removes the geometry confound from the\n"
        "# brief's region comparison.\n"
        "#\n"
        "# FIVE seeds, not ten. SR-GNN ran at ten and ViT at five; pairing\n"
        "# needs the same held-out patients on both sides. SR-GNN's 0.1719 is\n"
        "# a ten-seed mean and its five-seed mean may differ -- the run\n"
        "# recomputes each arm's own band over the seeds used, which PLAN\n"
        "# §4.12.1 requires anyway. ladder.TRADE_OFF_PAIR.\n"
        "#\n"
        "# If unresolvable, the contribution becomes a DESCRIPTION -- 'the\n"
        "# best-SCORING model is the least interpretable' -- which is licensed\n"
        "# either way. A resolvable result is the upside, not the premise.\n"
        if scope == "tradeoff" else
        "# **SCOPE: headline.** The single thinnest claim in the project --\n"
        "# ladder.BEST_ARM.nearest_challenger, claimed at a 1.05x margin on\n"
        "# condition 2 alone. Two run directories, ten vectors. It is its own\n"
        "# config so this can be answered before anyone resolves globs for\n"
        "# twenty-eight other run directories.\n"
        "#\n"
        "# If condition 1 fails here, '0.2520 is claimably ahead of\n"
        "# everything else' becomes 'it is the highest-scoring arm, not\n"
        "# claimably the highest'. The PCC is untouched, and the exhaustion\n"
        "# argument does not depend on the comparative sentence.\n"
        if scope == "headline" else
        "# **SCOPE: p7d.** The six Phase 7D contrasts\n"
        "# (ladder.P7D_PAIRED_CONTRASTS): b8/b32/concat/mvitv2 against the\n"
        "# b16 control, mvitv2 against swin_b's matched imagenet-G1 cell,\n"
        "# and b32@512 against Road B's patch16@512 comparator.\n"
        "#\n"
        "# **The cross-phase pair is legitimate, and mechanically so**: the\n"
        "# Road B arm scored the SAME manifest (same patients, same fold\n"
        "# column), the loader cross-checks the truth vector across every\n"
        "# file and the task re-checks it across seed groups to 1e-9, and\n"
        "# its ten seeds are Road A's pool -- the pair runs on the five\n"
        "# shared seeds, the TRADE_OFF_PAIR precedent. Only those five of\n"
        "# its vectors are declared, so the short-band check sees five of\n"
        "# five, not five of ten.\n"
        "#\n"
        "# Every 7D mean is DESCRIPTIVE until this run: arm 4's +0.0074 in\n"
        "# particular runs AGAINST the 7B prior's direction and may not be\n"
        "# quoted as overturning it without condition 1\n"
        "# (ladder.PHASE_7D_OBSERVED).\n"
        "#\n"
        "# ATTRIBUTION (maintainer, 2026-08-15): arm 6 is the direct response\n"
        "# to the supervision shared paper (Fan et al., Multiscale Vision\n"
        "# Transformers, ICCV 2021); arms 1-5 are the project's\n"
        "# decomposition of the 'multi-scale, not 16x16' request,\n"
        "# registered before the paper was shared. ladder.PHASE_7D_BUILT.\n"
        if scope == "p7d" else
        f"# **SCOPE: ladder.** All {len(pairs)} computable claims.\n"
        "#\n"
        "# EXCLUDED, deliberately (ladder.PAIRED_CLAIM_COVERAGE): the label\n"
        "# formulation, because `label` selects a manifest COLUMN so median\n"
        "# and mean arms are scored against DIFFERENT truth vectors -- their\n"
        "# delta compares two task difficulties, not two models on one task,\n"
        "# and a paired bootstrap has no common truth to pair on. Condition 1\n"
        "# is UNDEFINED for those four, not uncomputed.\n"
        "#\n"
        "# Also excluded: arms at an unrun stage that reuse nothing. Note a\n"
        "# G0 arm that REUSES a D-stage run is included -- the stage label is\n"
        "# not the test, and filtering on it dropped a Q1 comparison once.\n"
    )
    return (
        f"# PHASE 7 — PLAN §4.3 CONDITION 1 FOR THE LADDER'S CLAIMS\n"
        "#\n"
        "# GENERATED by scripts/generate_paired_claim_configs.py from"
        " cleft.ladder.\n"
        "#\n"
        "# **THIS RUN FITS NOTHING.** Every vector it reads was written by a\n"
        "# train_cv keeper run that already happened.\n"
        "#\n"
        "# [MEASURED 2026-08-04] 43 claimable verdicts across ladder.py,\n"
        "# phase7b.py and phase7c.py; NONE carried an interval. Phase 7C then\n"
        "# computed condition 1 and it withdrew all nine of that phase's\n"
        "# verdicts. Phase 7B had computed it, written it to its own\n"
        "# metrics.json, and recorded 'claimably worse' from condition 2 while\n"
        "# condition 1 sat in the same file saying UNRESOLVED. Two of two\n"
        "# phases audited had the defect. ladder.CLAIMS_REST_ON_HALF_THE_CRITERION.\n"
        "#\n"
        + scope_note +
        "#\n"
        "# The condition-2 margin is NOT a safety indicator: 7C failed\n"
        "# condition 1 at 3.62x. Results are reported thinnest-first on the\n"
        "# margin computed in the run, not on the recorded thresholds -- those\n"
        "# are the numbers under audit.\n"
        "#\n"
        + provenance +
        "#\n"
        "# Seed bands are NOT uniform: transformer arms ran at five seeds and\n"
        "# graph arms at ten, so the vector list is derived per stem rather\n"
        "# than assumed. ladder.paired_claim_vectors.\n"
        "#\n"
        f"#   bash -lc \"cd {CLUSTER_ROOT} && PYTHONPATH=src \\\n"
        f"#     python -m cleft.run --config configs/{config_name(scope)}.yaml\"\n\n"
    )


def render(scope: str) -> str:
    inputs = inputs_for(scope)
    pairs = ladder.paired_claim_pairs(scope)
    payload = {
        "schema_version": 1,
        "phase": "p7d" if scope == "p7d" else "p7",
        "tier": "keeper",
        "seed": 1337,
        "inputs": inputs,
        "task": {"kind": "paired_claims", "scope": scope, "n_boot": 10000},
    }
    body = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    return header(scope, inputs, pairs) + body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    drifted, written = [], 0
    for scope in SCOPES:
        path = REPO / "configs" / f"{config_name(scope)}.yaml"
        text = render(scope)
        if args.check:
            if not path.is_file():
                drifted.append(f"{path.name}: missing")
            elif path.read_text(encoding="utf-8") != text:
                drifted.append(f"{path.name}: differs from the derived scope")
        else:
            path.write_text(text, encoding="utf-8")
            written += 1

    if args.check:
        for line in drifted:
            print(f"  DRIFT {line}")
        print(f"{len(SCOPES)} configs checked, {len(drifted)} drifted")
        return 1 if drifted else 0

    for scope in SCOPES:
        pairs = ladder.paired_claim_pairs(scope)
        print(
            f"  {scope:9} {len(pairs):2} pairs, "
            f"{len(ladder.paired_claim_stems(scope)):2} run dirs, "
            f"{len(ladder.paired_claim_vectors(scope)):3} vectors"
        )
    print(f"wrote {written} configs -- neither fits anything")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
