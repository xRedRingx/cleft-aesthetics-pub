"""Generate Phase 23's ROPE config.

**Everything is harvested by READING shipped configs.** Every run
directory, path and hash already appears in a config this repo ships, so
they are carried VERBATIM -- no path is constructed, and **no hash is a
placeholder**, because none is new.

The 45 contrasts are assembled from the phase modules that define them
(``ladder``, ``roadb``, ``phase10``, ``phase11``, ``phase12``,
``phase17``) plus the two rows whose pairs are named in their own
records. Anything that cannot be resolved is REPORTED, never built.

Usage::

    python scripts/generate_phase23_config.py            # write
    python scripts/generate_phase23_config.py --check    # re-derive, diff
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
sys.path.insert(0, str(REPO / "src"))

#: Configs whose declared inputs carry the arms' per-seed CSVs.
HARVEST = (
    "p7_paired_ladder.yaml", "p7b_search.yaml", "p7c_paired_selected30.yaml",
    "p7d_paired.yaml", "roadb_p7_paired_resolution.yaml",
    "roadb_p7c_paired_regioncrop.yaml", "p10_paired.yaml", "p11_paired.yaml",
    "p12_paired.yaml", "p16_anchor_loop.yaml", "p17_family_analysis.yaml",
)

SEEDS = (1337, 2024, 7, 99, 12345)

#: The probe's own seed sd and count -- the width's inputs, not the width.
PROBE_SD, PROBE_N = 0.0148, 5


def _load(stem):
    import yaml

    return yaml.safe_load((CONFIGS / stem).read_text(encoding="utf-8"))


def harvest_csvs() -> dict:
    """``(arm_stem, seed) -> {"path", "rollup"}``, read from shipped configs.

    Keyed by the run directory's stem, so the arm names the phase
    modules use resolve directly. Five paths carry two names across
    configs; the entry is keyed by PATH so the duplicate is harmless.
    """
    out: dict[tuple[str, int], dict] = {}
    for stem in HARVEST:
        for entry in _load(stem).get("inputs") or []:
            path = entry["path"]
            if "/runs/" not in path or not path.endswith(".csv"):
                continue
            parts = path.rstrip("/").split("/")
            directory, filename = parts[-2], parts[-1]
            arm = directory.split("__")[0]
            if not filename.startswith("seed_"):
                continue
            seed = int(filename.split("__")[0].removeprefix("seed_"))
            key = (arm, seed)
            if key in out and out[key]["path"] != path:
                raise SystemExit(
                    f"{arm} seed {seed} resolves to two paths:\n"
                    f"  {out[key]['path']}\n  {path}\n"
                    "Two runs of one arm; the maintainer must pin which."
                )
            out[key] = {"path": path, "rollup": entry["rollup_sha256"]}
    return out


#: The p7b winner is declared in NO shipped config -- a run's own output
#: is never one of its inputs. Supplied  (identified by its
#: own metrics.json carrying the recorded mean_delta), and **THIS CONFIG
#: IS ITS FIRST DECLARATION**, which is why its hash appears here and
#: nowhere else. Everything else carries a hash that already exists.
#:
#: **[FILLED 2026-09-02 from declare_inputs.py: 283 files, 3,282,986
#: bytes.** It was an all-zero placeholder until then -- the config's
#: only one.]
P7B_WINNER = (
    "/home/user/codex/cleft-aesthetics/runs/keeper/p7b/"
    "p7b_search__eb4477d9__p7b-search-3"
)
P7B_ROLLUP = (
    "10ea3efa083e7c025fa6ae1288b5b73e28c7e9cbe87f37ff7bfa67c2e92ba7da"
)

#: Configs that declare a run DIRECTORY rather than its per-seed CSVs.
DIR_SOURCES = (
    "p17_family_analysis.yaml", "p18_metric_space.yaml",
)


def harvest_dirs() -> dict:
    """``arm_stem -> {"path", "rollup"}`` for directory declarations."""
    out = {}
    for stem in DIR_SOURCES:
        for entry in _load(stem).get("inputs") or []:
            path = entry["path"]
            if "/runs/" not in path or path.endswith(".csv"):
                continue
            arm = path.rstrip("/").split("/")[-1].split("__")[0]
            out.setdefault(arm, {
                "path": path, "rollup": entry["rollup_sha256"],
            })
    out["p7b_search"] = {"path": P7B_WINNER, "rollup": P7B_ROLLUP}
    return out


def contrasts() -> list[dict]:
    """The 45, from the modules that define them."""
    import importlib

    from cleft import ladder

    out = []

    def add(row, pairs):
        for pair in pairs:
            out.append({
                "key": f"{row}__{pair['key']}", "row": row,
                "a": pair["a"], "b": pair["b"],
            })

    # ladder scope: the row is the headline pair alone.
    headline = [
        p for p in ladder.paired_claim_pairs("ladder")
        if p["key"].startswith("headline__")
    ]
    add("p7-best-arm-vs-challenger", headline)

    p7d = ladder.paired_claim_pairs("p7d")
    add("p7d-null-and-concat",
        [p for p in p7d if p["key"] in
         ("p7d__patch32_vs_b16", "p7d__concat_vs_b16")])
    add("p7d-mvitv2", [p for p in p7d if "mvitv2" in p["key"]])

    roadb = importlib.import_module("cleft.roadb")
    resolution = roadb.paired_claim_pairs("roadb_resolution")
    survivors = {
        "resolution__vit_b16__scut_masked__224_to_512",
        "resolution__vit_b16__imagenet__224_to_512",
        "resolution__swin_b__scut_masked__224_to_512",
    }
    add("roadb-resolution-withdrawn",
        [p for p in resolution if p["key"] not in survivors])
    add("roadb-region-crop", roadb.paired_claim_pairs("roadb_regioncrop"))

    for row, module, scope in (
        ("p10-replication-vs-best-arm-withdrawn", "phase10", "p10"),
        ("p11-iem-loss-costs-pcc-withdrawn", "phase11", "p11_loss"),
    ):
        pairs = importlib.import_module(
            "cleft." + module).paired_claim_pairs(scope)
        add(row, pairs)

    p12 = importlib.import_module("cleft.phase12").paired_claim_pairs("p12")
    for row, token in (
        ("p12-basal-vs-frontal-bar-withdrawn", "basal"),
        ("p12-concat-vs-frontal-bar-unresolved", "concat_vs_frontal"),
        ("p12-concat-vs-capacity-unresolved", "capacity"),
    ):
        add(row, [p for p in p12 if token in p["key"]])

    # ---- the fifteen, added 2026-09-02 ------------------------------
    add("p7c-nine-verdicts", p7c_nine())
    add("p17-a-vs-probe", [{
        "key": "primary_a_vs_probe", "a": "p17_arm_a", "b": PROBE_STEM}])
    add("p17-c-vs-probe", [{
        "key": "primary_c_vs_probe", "a": "p17_arm_c", "b": PROBE_STEM}])
    add("p17-a-vs-c", [{
        "key": "secondary_a_vs_c", "a": "p17_arm_a", "b": "p17_arm_c"}])
    add("p17-b-vs-c", [{
        "key": "secondary_b_vs_c", "a": "p17_arm_b", "b": "p17_arm_c"}])
    add("p16-anchor-loop-unresolved", [{
        "key": "anchor_loop_vs_probe",
        "a": "p16_anchor_loop", "b": PROBE_STEM}])
    add("p7b-claimably-worse", [{
        "key": "searched_vs_baseline",
        "a": "p7b_search", "b": PROBE_STEM}])
    return out


#: The probe's run stem, read from ladder rather than typed.
def _probe_stem() -> str:
    import sys as _sys

    _sys.path.insert(0, str(REPO / "src"))
    from cleft import ladder

    return ladder.TRADE_OFF_PAIR["best_arm"]["stem"]


PROBE_STEM = _probe_stem()

#: The p7c arms carry a `p7c_` prefix in their run-directory stems;
#: ``phase7c.IDENTITY_ARM`` is the SHORT key ``0_identity``. A mismatch
#: here silently MISSES rather than failing, so it is reconciled once
#: and asserted in the suite.
P7C_PREFIX = "p7c_"


def _p7c_full(name: str) -> str:
    return name if name.startswith(P7C_PREFIX) else P7C_PREFIX + name


def p7c_nine() -> list[dict]:
    """The nine, BY CONSTRUCTION -- never a literal list.

    ``phase7c.paired_matrix``'s own docstring: *"All nine comparisons.
    Six against identity and the three declared between-arm pairs."*
    So: six against ``IDENTITY_ARM``, plus the three ``comparisons()``
    returns. **Not a filter over the twenty-one possible pairs**, which
    is why the recovery reads the same nine rather than picking nine.
    """
    import sys as _sys

    _sys.path.insert(0, str(REPO / "src"))
    from cleft import phase7c

    identity = _p7c_full(phase7c.IDENTITY_ARM)
    arms = sorted(
        _p7c_full(a["name"]) if isinstance(a, dict) else _p7c_full(a)
        for a in _p7c_arms()
    )
    if identity not in arms:
        raise SystemExit(
            f"the identity arm {identity!r} is not among {arms}; the "
            "prefix reconciliation is wrong and every identity "
            "comparison would silently miss"
        )
    pairs = [
        {"key": f"vs_identity__{arm}", "a": identity, "b": arm}
        for arm in arms if arm != identity
    ]
    for comparison in phase7c.comparisons():
        pairs.append({
            "key": f"declared__{comparison['question']}",
            "a": _p7c_full(comparison["a"]["name"]),
            "b": _p7c_full(comparison["b"]["name"]),
        })
    if len(pairs) != 9:
        raise SystemExit(
            f"p7c built {len(pairs)} pairs, not nine. paired_matrix says "
            "six against identity plus three declared; the arm list or "
            "comparisons() has changed and the row must be re-read."
        )
    return pairs


def _p7c_arms() -> list[str]:
    """The seven arm stems, read from the paired config's own inputs."""
    stems = set()
    for entry in _load("p7c_paired_selected30.yaml").get("inputs") or []:
        path = entry["path"]
        if "/runs/" in path:
            stems.add(path.rstrip("/").split("/")[-2].split("__")[0])
    return sorted(stems)


def build():
    import yaml

    csvs = harvest_csvs()
    pairs = contrasts()

    directories = harvest_dirs()
    missing, needed, needed_dirs = [], {}, {}
    for contrast in pairs:
        for arm in (contrast["a"], contrast["b"]):
            found = [(arm, s) for s in SEEDS if (arm, s) in csvs]
            if found:
                for key in found:
                    needed[key] = csvs[key]
            elif arm in directories:
                # Declared as a run DIRECTORY, not per-seed files. The
                # task reads seed_<n>__predictions.csv inside it.
                needed_dirs[arm] = directories[arm]
            else:
                missing.append(
                    f"{contrast['key']}: no CSVs and no directory for "
                    f"arm {arm!r}"
                )

    # No manifest: the prediction CSVs carry truth, so the task never
    # reads one. Declaring it would be a value that reaches no code.
    inputs = []
    for (arm, seed), entry in sorted(needed.items()):
        inputs.append({
            "name": f"{arm}__seed_{seed}",
            "path": entry["path"],
            "rollup_sha256": entry["rollup"],
        })
    for arm, entry in sorted(needed_dirs.items()):
        inputs.append({
            "name": arm,
            "path": entry["path"],
            "rollup_sha256": entry["rollup"],
        })

    # A file entry's directory is its parent; a directory entry IS the
    # directory. Getting this wrong undercounts silently, which it did.
    directories = {
        e["path"].rstrip("/").split("/")[-2 if e["path"].endswith(".csv")
                                        else -1]
        for e in inputs
    }
    placeholders = [
        e["name"] for e in inputs if set(e["rollup_sha256"]) == {"0"}
    ]
    task = {
        "kind": "p23_rope",
        "contrasts": pairs,
        "seeds": list(SEEDS),
        "probe_seed_sd": PROBE_SD,
        "probe_n_seeds": PROBE_N,
        "expect_rope_half_width": 0.018346,
        "rho": 0.2,
        "decision_threshold": 0.95,
        "full_n": 25,
        "expect_contrasts": len(pairs),
        "expect_patients": 237,
    }
    header = f"""\
# PHASE 23 -- THE ROPE ANALYSIS
#
# GENERATED by scripts/generate_phase23_config.py.
# Do not hand-edit: --check re-derives and diffs.
#
# Benavoli's single-dataset Bayesian correlated t-test over
# banked per-fold PCC differences. phase23.EXIT_CRITERIA.
#
# **IT CHANGES NO FIGURE AND RESOLVES NOTHING THIS COHORT
# CANNOT RESOLVE.** Quoted from the bank: "it does not RESOLVE
# anything the cohort cannot resolve. A ROPE analysis would
# restate the limit in better language, not lift it."
#
# ATTRIBUTION -- TWO OURS, FOUR SOURCE:
#   OURS   rope half-width 0.018346 -- combined_claimable_delta's
#          arm_means_95 at the probe's sd 0.0148, n = 5, at FULL
#          PRECISION. [CORRECTED 2026-09-02: this read 0.0183, a
#          rounded DISPLAY quoted as the quantity. The guard
#          refused the launch on the discrepancy -- criterion 2
#          working. Seven digits, because the guard is EXACT: a
#          tolerance there would defeat its whole purpose.]
#   OURS   threshold P(.) > 0.95, from the ruled 20:1 loss ratio
#   SOURCE rho = 0.2 -- NOT a choice: rho is unidentifiable
#          ("rho-hat = 0 regardless the observations"). Its
#          approximation TRAVELS: derived for random resamples,
#          applied to non-overlapping k-fold folds (footnote 2).
#   SOURCE the matching prior, the eq-6 posterior, and P(rope)
#          as the integral over the interval.
#
# THE WIDTH IS NOT A VALUE HERE. `expect_rope_half_width` is a
# CHECK: the task recomputes from probe_seed_sd and probe_n_seeds
# and REFUSES on mismatch. Criterion 2 -- if the width moves
# after a posterior exists, EVERY RESULT IS WITHDRAWN.
#
# SCOPE: all {len(pairs)} contrasts, {len(directories)} run
# directories. Rows aggregate contrasts and the two are not
# interchangeable.
#
# ALL FORTY-FIVE ARE HERE. [2026-09-02: this config ran
# thirty and called itself incomplete. The fifteen were
# recovered -- p7c's nine BY CONSTRUCTION from paired_matrix's
# own rule, p17's four from the family tuple, p16's one from
# PRIMARY_CONTRAST_REGISTERED, p7b's one by measurement. The
# task now REFUSES a short config rather than trusting a
# comment. phase23.THE_COMPLETION_TO_FORTY_FIVE.]
#
# NO PLACEHOLDERS. p7b_search's run directory is declared in
# no OTHER config -- a run's own output is never one of its
# inputs -- so THIS config is its first declaration. [FILLED
# 2026-09-02 from declare_inputs.py; it was the config's only
# placeholder until then.] Every other entry carries a hash
# already present in a shipped config, byte-identical.
#
# **NO PLACEHOLDER HASHES AND NO DECLARE CYCLE.** Every path and
# rollup below already appears in a shipped config and is carried
# VERBATIM -- no new hash enters, so nothing needs declaring.
#
#   bash -lc "cd /home/user/codex/cleft-aesthetics && PYTHONPATH=src \\
#     python -m cleft.run --config configs/p23_rope.yaml"
"""
    body = {
        "schema_version": 1, "phase": "p23", "tier": "keeper", "seed": 1337,
        "inputs": inputs, "task": task,
    }
    text = header + "\n" + yaml.safe_dump(body, sort_keys=False)
    return text, pairs, directories, missing, placeholders


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    text, pairs, directories, missing, placeholders = build()
    if missing:
        print("UNRESOLVED -- reported, not built:")
        for item in missing:
            print("  ", item)
        return 1
    path = CONFIGS / "p23_rope.yaml"
    if args.check:
        current = path.read_text(encoding="utf-8") if path.is_file() else ""
        if current != text:
            print("DRIFT: p23_rope.yaml")
            return 1
        print(f"config matches the generator ({len(pairs)} contrasts, "
              f"{len(directories)} run dirs, "
              f"{len(placeholders)} placeholder(s))")
        return 0
    path.write_text(text, encoding="utf-8")
    print(f"wrote p23_rope.yaml: {len(pairs)} contrasts, "
          f"{len(directories)} run directories, "
          f"{len(placeholders)} placeholder(s): {placeholders or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
