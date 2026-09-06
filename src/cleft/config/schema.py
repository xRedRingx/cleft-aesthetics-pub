"""The config schema.

Part 2.3: nothing scientific is ever a command-line flag, so the config is the
provenance record. That only works if the loader is strict — a silently absorbed
typo is a result whose settings are not what the file appears to say.

Adding a field means adding it here. That friction is the point.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any

import yaml

SUPPORTED_SCHEMA_VERSIONS = (1,)


class ConfigError(ValueError):
    """The config is not valid. Always fatal, never a warning."""


# --------------------------------------------------------------------------
# declared input paths: portability, and the one indirection that is allowed
# --------------------------------------------------------------------------

#: An input path may name an environment variable, and **only** in the
#: ``inputs[].path`` field. ``${CLEFT_SCUT_ROOT}`` or
#: ``${CLEFT_SCUT_ROOT}/Images``.
#:
#: **The name must be ``CLEFT_``-prefixed, and that is a safety property rather
#: than tidiness.** Unrestricted expansion would let a committed config read any
#: variable in the environment -- ``PATH``, a token, a credential -- and paste it
#: into a run record that is then shared. The namespace keeps the set of readable
#: variables to ones this project defines.
ENV_REFERENCE = re.compile(r"^\$\{(CLEFT_[A-Z0-9_]+)\}(/.+)?$")

#: How a declared path is written. Recorded as a closed set so a new shape --
#: ``~/data``, ``%USERPROFILE%\\data``, a bare drive letter -- classifies as
#: something rather than falling through a check silently.
DECLARED_PATH_KINDS = (
    "env_reference",    # ${CLEFT_...}[/subpath]
    "posix_absolute",   # /home/user/... -- resolves on the cluster
    "repo_relative",    # data/smoke/v1
    "drive_letter",     # C:/... or C:\... -- laptop only
    "home_relative",    # ~/... -- depends on who is logged in
    "windows_env",      # %VAR% -- not expanded anywhere in this project
)

#: What a **committed** config may use. The rule is portability: a committed
#: config is one that gets reproduced on the cluster, so its declared paths must
#: be resolvable there.
#:
#: Deliberately NOT enforced by ``load_config``: a test declares
#: ``str(tmp_path)``, which is drive-lettered on Windows, and a one-off local run
#: may legitimately point at a local directory. The constraint is on what is
#: committed, and ``test_smoke_run.py`` is where it is enforced.
PORTABLE_PATH_KINDS = ("env_reference", "posix_absolute", "repo_relative")


#: **[PARAMETERISED 2026-09-05] The cohort's declared paths, by ROLE.**
#:
#: The literals these replace named the holding institution and both
#: national studies the cohort is drawn from, and two people, in every
#: committed config. With the published n that is enough to identify the
#: cohort, so the repository refers to each artifact by what it IS and the
#: value is supplied on the cluster.
#:
#: **The leaves are parameterised, not only the tree.** A
#: ``${CLEFT_COHORT_ROOT}`` covering the parent directory would have left
#: the identifying filename standing in the declaration, which achieves a
#: third of the point. Each variable therefore holds an ABSOLUTE path to
#: one artifact; there is no root variable because every identifying
#: string was a leaf, and a root nothing refers to is a fiction.
#:
#: **Guard 3 is unaffected, and this is the reason it is safe.** A
#: declared path is the LOCATION a run resolves; the ``rollup_sha256``
#: beside it digests the artifact's BYTES (for a directory, the paths of
#: the files inside it relative to that directory). The declared path has
#: never entered any hash, so moving it behind a variable changes what a
#: reader can see and nothing a run verifies.
COHORT_ENV_REFERENCES = {
    "CLEFT_SCORESHEET": "the primary score sheet workbook, integer grades",
    "CLEFT_SCORESHEET_TEXT": "the parallel workbook, text grades -- an equivalence check, not a second source of truth",
    "CLEFT_SOFT_LABELS": "the derived per-image labels CSV",
    "CLEFT_PHOTOS": "the cropped anonymised patient photographs",
    "CLEFT_ANCHOR_SET": "the 25-image external anchor set",
}

#: **[STILL OPEN 2026-09-05] Where the cohort is still named by path, and
#: why each one resisted.** Enumerated so the guard can pass without the
#: exceptions becoming invisible: the list is the exception, and it cannot
#: grow without editing this record.
COHORT_PATHS_STILL_OPEN = {
    "frozen_tree": (
        "``src/cleft/provenance/`` is a FROZEN TREE "
        "(``tests/test_frozen_apparatus.py``) and cannot be edited without "
        "breaking the freeze guard. The only marker inside it is ``/bch/`` "
        "in ``context.py``'s docstring, an ILLUSTRATIVE path "
        "(``/home/user/codex/bch/photos``) showing why a POSIX path reads "
        "as relative on Windows. **It names no study and no person** -- "
        "``photos`` is generic -- so the residual disclosure is the "
        "institution shorthand alone, in a comment. Unfreezing a module to "
        "edit a docstring costs more than it buys. **Ruled to stay**"
    ),
    "task_fields_CLOSED_2026_09_05": (
        "**CLOSED.** Four lines in three dev-tier configs carried the "
        "cohort path under ``task:``, where ``expand_input_paths`` does not "
        "reach. They now declare the artifact under ``inputs:`` with a "
        "``${CLEFT_*}`` reference and a placeholder rollup, and the task "
        "field carries the input's NAME -- the pattern 63 other configs "
        "already used (``scut_root``, ``mebeauty_root``). "
        "``run.declared_input`` resolves it and raises rather than falling "
        "back, because a silent fallback is how the hole stayed open"
    ),
    "and_the_second_defect_it_hid_CLOSED_2026_09_05": (
        "**CLOSED, and it was the worse half.** Those configs carried "
        "``inputs: []``, so guard 3 verified nothing about the file they "
        "opened -- **the workbook every label in this project derives from "
        "was read on the strength of a path alone**, by the two "
        "diagnostics whose whole job is to check it. The hashes are "
        "PENDING as all-zero placeholders until a cluster declare pass "
        "fills them; until then the runs refuse rather than proceed, which "
        "is the intended state and not a regression"
    ),
    "what_replaced_the_reasoning": (
        "``p1_scan_folders.yaml`` carried a note justifying the placement: "
        "*a diagnostic runs before the hash round trip a declared input "
        "requires, and produces no artifact any result depends on*. **The "
        "first half is answered by the placeholder convention** -- that is "
        "how every other config bootstraps. **The second half was never "
        "the point**: the hazard is not what a diagnostic WRITES, it is "
        "what it READS. The placement was deliberate and the reasoning was "
        "wrong, which is why the guard is a rule about SHAPE "
        "(``test_no_task_field_carries_a_filesystem_path``) rather than a "
        "list of field names"
    ),
}


#: **Segments that must not appear in a tracked file.** Named here rather
#: than inline in the test so a new spelling is added in ONE place. Each
#: is identifying on its own: two name national studies, one names the
#: holding institution, two are surnames carried inside a filename.
#:
#: This is the blacklist half. The POSITIVE half -- that a cohort input is
#: declared through ``COHORT_ENV_REFERENCES`` -- is what actually holds;
#: the blacklist exists because a positive rule cannot see a path that
#: nobody declared as an input at all.
COHORT_PATH_MARKERS = (
    "/bch/",
    "CSAG",
    "CCUK",
    "Bruce 1",
    "1_Liu",
)


def classify_declared_path(raw: str) -> str:
    """Which of ``DECLARED_PATH_KINDS`` this declared path is.

    Order matters: an env reference is checked first because ``${CLEFT_X}/y`` has
    no drive and no leading slash and would otherwise read as repo-relative.
    """
    if not isinstance(raw, str) or not raw:
        raise ConfigError(f"declared path must be a non-empty string, got {raw!r}")
    if ENV_REFERENCE.match(raw):
        return "env_reference"
    if raw.startswith("$") or "${" in raw:
        raise ConfigError(
            f"declared path {raw!r} looks like a variable reference but does not "
            f"match {ENV_REFERENCE.pattern}. Only ${{CLEFT_<NAME>}} is expanded, "
            "optionally followed by /subpath -- see ENV_REFERENCE for why the "
            "name is namespaced."
        )
    if re.match(r"^%[A-Za-z0-9_]+%", raw):
        return "windows_env"
    if raw.startswith("~"):
        return "home_relative"
    if PureWindowsPath(raw).drive:
        return "drive_letter"
    if raw.startswith("/"):
        return "posix_absolute"
    return "repo_relative"


def expand_input_paths(inputs: list[dict], environ=None) -> list[dict]:
    """Resolve ``${CLEFT_*}`` references in declared input paths.

    **Why the config layer and not the provenance layer.** ``RunContext`` calls
    ``load_config``, so expanding here reaches every consumer -- the run, and
    ``scripts/declare_inputs.py`` -- without touching ``provenance/``, which is a
    frozen tree.

    **Why the provenance record is still complete.** The run directory copies
    ``config.yaml`` **verbatim**, so it preserves the ``${CLEFT_SCUT_ROOT}``
    declaration; and ``inputs.json`` records the **resolved** path beside the
    rollup that verifies it. Declaration and resolution are therefore both on
    disk, next to the hash that makes the resolution checkable. That is a better
    record than baking a machine path into the config, and it is why this did not
    need ``env.json`` -- which is built inside the frozen module and deliberately
    records an allowlist rather than sweeping the environment.

    **Expansion applies to ``inputs[].path`` and nothing else.** Letting a task
    field read the environment would put a scientific setting outside the config,
    which is Chain 3 of the original failure (Part 2.3).
    """
    environ = os.environ if environ is None else environ
    expanded = []
    for entry in inputs:
        raw = entry["path"]
        match = ENV_REFERENCE.match(raw) if isinstance(raw, str) else None
        if match is None:
            expanded.append(entry)
            continue

        name, subpath = match.group(1), match.group(2) or ""
        value = environ.get(name)
        if not value:
            raise ConfigError(
                f"input {entry['name']!r} declares {raw}, but {name} is not set "
                "in the environment.\n"
                "  This config is portable ON PURPOSE: the same data lives at a "
                "different path on each machine, so the path comes from the "
                "environment and the declared rollup_sha256 verifies whatever "
                "was resolved.\n"
                f"  Set {name} to the directory on this machine. The config "
                "header records the known locations."
            )

        # An env var holding a relative path would be re-rooted under the repo by
        # resolve_declared_path -- which is exactly the concatenation defect this
        # indirection exists to remove, arriving by a different door.
        if not _is_absolute_path(value):
            raise ConfigError(
                f"input {entry['name']!r} resolved {name} to {value!r}, which is "
                "not an absolute path. A relative value would be re-rooted under "
                "the repository, producing a path nobody declared."
            )

        resolved = dict(entry)
        resolved["path"] = value.rstrip("/\\") + subpath
        resolved["path_declared_as"] = raw
        resolved["path_from_env"] = name
        expanded.append(resolved)
    return expanded


def _is_absolute_path(raw: str) -> bool:
    """Absolute on ANY platform, from the single frozen definition.

    Imported inside the function rather than at module scope because
    ``provenance.context`` imports this module -- a top-level import would be a
    cycle. Deferred, it resolves at call time when both modules are loaded.

    **Not reimplemented here**, deliberately: a second copy of this predicate is
    exactly the defect that produced
    ``/home/user/codex/cleft-aesthetics/C:/data/...`` in
    ``scripts/declare_inputs.py``, which had its own platform-native check.
    """
    from ..provenance.context import is_absolute_path

    return is_absolute_path(raw)


@dataclass(frozen=True)
class Field:
    type: type
    required: bool = True
    choices: tuple | None = None
    spec: dict[str, "Field"] | None = None
    item_spec: dict[str, "Field"] | None = None
    default: Any = None
    doc: str = ""


INPUT_SPEC: dict[str, Field] = {
    "name": Field(str, doc="how the code refers to this artifact"),
    "path": Field(str, doc="absolute, or relative to the repo root"),
    "rollup_sha256": Field(str, doc="hash_dir()['rollup'] of the artifact"),
}

#: Task schemas, keyed by ``task.kind``. Later phases add entries; each one is
#: validated strictly against its own spec, so a key from one task kind cannot
#: leak into another.
TASK_SPECS: dict[str, dict[str, Field]] = {
    "smoke": {
        "kind": Field(str, choices=("smoke",)),
        "n_samples": Field(int, doc="asserted against the input at runtime"),
    },
    # Read-only diagnostics.
    #
    # [CORRECTED 2026-09-05] These used to take a PATH under `task` rather
    # than declaring an input, and the comment here justified it: "they run
    # BEFORE the hash round trip that pinning requires ... they write no
    # artifact". **Both halves failed.** The first is answered by the
    # placeholder convention, which is how every other config bootstraps.
    # The second was never the point -- the hazard is not what a diagnostic
    # WRITES, it is what it READS, and these two read the workbook every
    # label in this project derives from. Guard 3 verified nothing about it.
    #
    # They now name a declared input, like every other task that reads one.
    "scan_folders": {
        "kind": Field(str, choices=("scan_folders",)),
        "patient_folders": Field(str, doc="name of the declared photo-tree input"),
    },
    "inspect_scoresheet": {
        "kind": Field(str, choices=("inspect_scoresheet",)),
        "scoresheet": Field(str, doc="name of the declared workbook input"),
        "compare_with": Field(
            str,
            required=False,
            doc=(
                "name of a second declared workbook input; reports whether "
                "the two forms are identical"
            ),
        ),
    },
    # Phase 2's visual check. Read-only over the images; writes a CLUSTER-ONLY
    # PNG and a SHAREABLE aggregate report.
    "contact_sheet": {
        "kind": Field(str, choices=("contact_sheet",)),
        "manifest_artifact": Field(str, doc="input name of data/manifests/<version>"),
        "patient_folders": Field(str, doc="input name of the patient folder tree"),
        "generator": Field(str, choices=("grid", "random", "anatomy")),
        "geometries": Field(list, item_spec=None, doc="subset of ['g1', 'g2']"),
        "n_patients": Field(int, doc="how many patients to render"),
        "columns": Field(int, required=False, default=4),
        "boundary_rule": Field(str, required=False, default="keep"),
        "patch_seed": Field(int, required=False, default=1337),
    },
    # Phase 4 §2: the segmentation feasibility gate. Decides whether SymNose is
    # built at all. CLUSTER-ONLY sheet, SHAREABLE verdict, no pipeline.
    "segmentation_probe": {
        "kind": Field(str, choices=("segmentation_probe",)),
        "manifest_artifact": Field(str, doc="input name of data/manifests/<version>"),
        "patient_folders": Field(str, doc="input name of the patient folder tree"),
        "n_patients": Field(int, doc="a handful -- this is a diagnostic"),
        #: G1 by default and deliberately: G2 resamples every row to full width,
        #: which smooths the local colour structure under test.
        "geometry": Field(str, required=False, default="g1", choices=("g1", "g2")),
        #: Where a lip may be looked for. "none" reproduces run 1, which searched
        #: the whole frame and found face-versus-background on every patient.
        "spatial_prior": Field(
            str,
            required=False,
            default="bottom",
            choices=("none", "top", "middle", "bottom"),
        ),
        #: Keeps the search off the white padding. False, with spatial_prior
        #: "none", re-derives run 1 exactly.
        "face_mask": Field(bool, required=False, default=True),
        #: How far the contour is relaxed outward, as a fraction of the image's
        #: own class gap. Re-tune against the UPPER lip once the fissure
        #: constraint is in: with the crossing prevented, 0.15 may be right.
        "contour_relaxation": Field(float, required=False, default=0.15),
        #: Per-image renormalisation of the discriminant inside the region. The
        #: FAIRNESS hypothesis -- see segmentation.DISCRIMINANT_NORMALISATIONS.
        #: Default "none" so this is a separate run under R6, one change at a
        #: time, rather than confounded with the fissure constraint.
        "normalisation": Field(
            str, required=False, default="none",
            choices=("none", "rank", "zscore"),
        ),
        #: How dark the fissure row must be relative to the mask's typical row.
        #: A config field, not a source constant, so the sweep is a config change
        #: -- and `metrics.json` records which value produced which detection
        #: rate. See fissure.sweep for what each candidate would have found.
        "fissure_min_depth": Field(float, required=False, default=0.08),
        #: How the lip mask is divided into upper and lower. "centroid" and
        #: "fraction" are shape-derived and apply uniformly; "fissure" is the
        #: run-4 behaviour, kept only so it can be re-derived and compared.
        "split_rule": Field(
            str, required=False, default="centroid",
            choices=("centroid", "fraction", "fissure"),
        ),
    },
    # Phase 4 §3.2: does the segmentation hold across the whole cohort, or only
    # across the 9 the gate looked at? Diagnostic only -- nothing is tuned.
    "symnose_audit": {
        "kind": Field(str, choices=("symnose_audit",)),
        "manifest_artifact": Field(str, doc="input name of data/manifests/<version>"),
        "staged_artifact": Field(str, doc="input name of data/staged/<version>"),
        #: g2 only. ``symnose.feature_matrix`` refuses g1 outright, but the audit
        #: does not go through that refusal -- it only uses the value to choose
        #: ``staged_patient_<geometry>.npy`` -- so offering g1 here was permission
        #: the science does not grant and nothing downstream would have caught.
        "geometry": Field(str, required=False, default="g2", choices=("g2",)),
        #: `label` is deliberately absent. The handler discarded the labels array
        #: it produced (``images, _, patient_ids, _ = load_inputs(...)``), so the
        #: field's only observable effect was whether the run raised on an unknown
        #: column name. A knob that cannot change a result is worse than no knob:
        #: it reads as though the audit were run against a chosen target.
        "normalisation": Field(
            str, required=False, default="none",
            choices=("none", "rank", "zscore"),
        ),
        "split_rule": Field(
            str, required=False, default="centroid",
            choices=("centroid", "fraction", "fissure"),
        ),
    },
    # Phase 4: sensitivity of the estimate to how the data is partitioned.
    # NOT fold-assignment variance -- the frozen generator is deterministic, so
    # that cannot be measured here. See train.partition.
    "partition_sensitivity": {
        "kind": Field(str, choices=("partition_sensitivity",)),
        "manifest_artifact": Field(str, doc="input name of data/manifests/<version>"),
        "staged_artifact": Field(str, doc="input name of data/staged/<version>"),
        "geometry": Field(str, choices=("g1", "g2")),
        "label": Field(str),
        "backbone": Field(str, choices=("vit_b16", "stub", "ridge")),
        #: `choices` matters here more than it looks. Without it a typo takes a
        #: path that still runs: `prepare_features` branches on `== "full"`, so
        #: `hed` gets FROZEN embeddings, while `make_factory` branches on
        #: `== "head"`, so it builds a full fine-tuning backbone. The run
        #: produces numbers from an arm nobody configured.
        "trainable": Field(
            str, required=False, default="head", choices=("head", "full")
        ),
        #: Fold counts to compare. All use the frozen generator unchanged.
        "fold_counts": Field(list, required=False, default=(5, 6, 10)),
        #: 237 folds, deterministic, no assignment choice at all.
        "include_lopo": Field(bool, required=False, default=True),
        "max_epochs": Field(int),
        "patience": Field(int),
        "inner_val_frac": Field(float),
        "monitor": Field(str, choices=("inner_val_mse", "inner_val_pcc")),
        "learning_rate": Field(float, required=False, default=1e-3),
        "weight_decay": Field(float, required=False, default=0.01),
        "batch_size": Field(int, required=False, default=32),
    },
    #: **Phase 7C exit criterion 5.** The one check no test substitutes for:
    #: a protect-list covering the wrong pixels passes every assertion in the
    #: suite and is wrong in the only way that matters. Renders the pixels the
    #: ARM will see -- same staged artifact, same per-patient content boxes,
    #: same mask builder -- so a sheet and an arm cannot disagree.
    #:
    #: The policy is fixed at the full one (arm 5) rather than configurable:
    #: the review is of the protection mask and of what full-strength
    #: augmentation does to it, and a sheet rendered at some other policy
    #: would not be evidence about either.
    "augmentation_contact_sheet": {
        "kind": Field(str, choices=("augmentation_contact_sheet",)),
        "manifest_artifact": Field(str),
        "staged_artifact": Field(str),
        "geometry": Field(str, choices=("g1", "g2")),
        "n_patients": Field(int, required=False, default=12),
    },
    #: Phase 7B: the pre-registered search, in ONE job. The budget, the space,
    #: the selection rule and the exhaustion criterion are NOT config fields --
    #: they live in ``cleft.phase7b`` where the suite checks them, because a
    #: pre-registration a config could edit is not one. What the config
    #: supplies is the data: the embedding sets (named ``set_<name>``, one per
    #: representation the space reads) and the winner's seed list.
    "phase7b_search": {
        "kind": Field(str, choices=("phase7b_search",)),
        "manifest_artifact": Field(str, doc="input name of data/manifests/<version>"),
        "staged_artifact": Field(str, doc="input name of data/staged/<version>"),
        "geometry": Field(str, choices=("g1", "g2")),
        "label": Field(str),
        "inner_val_frac": Field(float),
        #: The winner's own band (exit criterion 3). PLAN §4.12.1: a tuned arm
        #: reports its OWN sd, never the untuned arm's.
        "seeds": Field(list),
        #: The untuned arm's out-of-fold predictions are declared as INPUTS
        #: named ``baseline_oof_seed_<seed>``, one per seed, not as a task
        #: field -- the same convention the ``set_<name>`` embedding inputs
        #: use. [MEASURED 2026-08-02] the baseline is five vectors, not one,
        #: and its run directory could not be declared whole because it
        #: contains ``code/``, a git worktree whose rollup would cover the
        #: source tree and move whenever it was touched.
        #:
        #: Absent, the task completes and says plainly in metrics.json that
        #: exit criterion 4 is unmet -- silence there would read as a
        #: comparison that was made. See ``phase7b.BASELINE_INPUT_PREFIX``.
    },
    #: Phase 7C exit criterion 6: PLAN §4.3's condition 1, over vectors that
    #: already exist. **This task fits nothing** -- it reads one prediction CSV
    #: per arm per seed, declared as inputs named ``oof_<arm>_seed_<seed>``
    #: (the ``baseline_oof_seed_<seed>`` convention, widened to seven arms),
    #: and computes the per-seed paired BCa the phase promised and never ran.
    #:
    #: The arm list, the comparisons and the strengths are NOT config fields:
    #: they live in ``cleft.phase7c``. What the config supplies is which round
    #: the vectors belong to, so the task can refuse vectors that do not
    #: reproduce it -- three rounds of these arms exist on disk and two are
    #: void. See ``phase7c.PAIRED_BCA_IS_THE_MISSING_CONDITION``.
    #: Road B Phase 2: stage ONE setting into its OWN artifact.
    #:
    #: One run per setting, so a partial re-run is one job and moves only that
    #: hash (``roadb.ARTIFACT_PER_SETTING``). The setting list is NOT a config
    #: field -- ``roadb.staging_settings()`` derives it, so a config cannot
    #: add a cell the axis does not contain.
    "roadb_stage": {
        "kind": Field(str, choices=("roadb_stage",)),
        "setting": Field(str, doc="a name from roadb.staging_settings()"),
        "manifest_artifact": Field(str, doc="input name of data/manifests/<version>"),
        "patient_folders": Field(str, doc="input name of the source folders"),
        "out_version": Field(str, doc="the artifact directory name"),
    },
    #: Road B Phase 2: the per-family residual gate, over artifacts it does
    #: NOT own.
    #:
    #: A separate task because no single staging run sees all three
    #: resolutions. It declares the ten artifacts as inputs so guard 3 hashes
    #: them, and **records those rollups in its own summary** -- a re-run
    #: after the verdict would otherwise leave the verdict silently wrong.
    "roadb_residual_gate": {
        "kind": Field(str, choices=("roadb_residual_gate",)),
    },
    #: Road B Phase 2: both contact sheets, over the same ten artifacts as
    #: the gate. CLUSTER-ONLY PNGs -- the review they feed gates Phase 3.
    #:
    #: The axis-patient selection rule is NOT a config field: it is measured
    #: from the 224 non-square G1 artifact's geometry, extremes included,
    #: where a hand-picked list could quietly avoid the aspect ratios that
    #: expose framing defects. What the config supplies is the strip's cap
    #: and row width. ``max_patients`` defaults to the measured 768 frontal
    #: count (41), so nothing truncates at current counts and any truncation
    #: is REPORTED; ``columns`` is the width a setting wraps at.
    #: Branch 3's gate (brief §5 criterion 3): 22 anatomy crops per patient
    #: at NATIVE resolution beside their position on the face. No arm runs
    #: until it is reviewed.
    #: Branch 3 criterion 2: 22 crops per patient, embedded ONCE into a
    #: region_vectors set. No new torch surface -- the crops are flattened
    #: and handed to the existing extract_features.
    "roadb_region_crop_extract": {
        "kind": Field(str, choices=("roadb_region_crop_extract",)),
        "staged_artifact": Field(str, doc="the non-square G2 staged artifact"),
        "out_version": Field(str, doc="the embedding artifact version"),
        "backbone": Field(str, choices=("vit_b16", "srgnn")),
        "init": Field(str, choices=("imagenet",)),
        #: "whole" is the CONTROL: one region that is the entire content
        #: box, cut and resized to 224 exactly as each crop is, so
        #: crop-vs-whole differs in content alone. It goes through this
        #: task because non-square staging has no stacked tensor for the
        #: extraction task to read (roadb_regioncrop.whole_frame_patch).
        "layout": Field(str, choices=("anatomy", "random", "whole")),
        "batch_size": Field(int, required=False, default=32),
    },
    "roadb_region_crop_sheet": {
        "kind": Field(str, choices=("roadb_region_crop_sheet",)),
        "staged_artifact": Field(str, doc="the non-square G2 staged artifact"),
        "n_patients": Field(int, required=False, default=8),
        "columns": Field(int, required=False, default=6),
        #: "native" shows the crops AS CUT (the information present);
        #: "as_fed" shows them AS THE BACKBONE RECEIVES THEM, squared by
        #: resize_nearest. Two sheets because they answer two questions --
        #: roadb.SHEETS_MUST_SHOW_WHAT_THE_CONSUMER_RECEIVES.
        "mode": Field(str, required=False, default="native",
                      choices=("native", "as_fed")),
    },
    "roadb_sheets": {
        "kind": Field(str, choices=("roadb_sheets",)),
        "max_patients": Field(int, required=False, default=41),
        "columns": Field(int, required=False, default=6),
    },
    #: Road B Phase 6, gate 1: the swin-at-resolution build on the pinned
    #: image. No data inputs -- it builds models and compares parameters.
    #: Runs FIRST because it is the one gate that could refute the
    #: four-backbone conclusion, and it is short
    #: (roadb.PHASE_6_STRUCTURE build order).
    "roadb_swin_build_gate": {
        "kind": Field(str, choices=("roadb_swin_build_gate",)),
        "sizes": Field(list, required=False, default=(512, 768)),
        "pretrained": Field(bool, required=False, default=True),
    },
    #: Road B Phase 6, gate 3: the graph backbones' region geometry at the
    #: new sizes. No data inputs -- synthetic content, deterministic. Each
    #: backbone gated on ITS OWN contract: SR-GNN's fixed-42-map invariance,
    #: AG-Net's count law and normalised bounds (placement drift is a
    #: property, reported not gated).
    "roadb_graph_geometry_gate": {
        "kind": Field(str, choices=("roadb_graph_geometry_gate",)),
        "sizes": Field(list, required=False, default=(224, 512, 768)),
        "backbones": Field(list, required=False, default=("srgnn", "agnet")),
    },
    #: The pretrained init frozen as a versioned, hashed artifact -- closing
    #: the one input guard 3 never verified. One backbone per run, one
    #: artifact per backbone (consumed selectively). This run performs the
    #: LAST undeclared download.
    "snapshot_pretrained_init": {
        "kind": Field(str, choices=("snapshot_pretrained_init",)),
        "backbone": Field(
            str, choices=(
                "vit_b16", "swin_b", "srgnn", "agnet",
                # Phase 7D: the patch axis and the hierarchical arm.
                "vit_b32", "vit_b8", "mvitv2_b",
            )
        ),
    },
    #: Road B Phase 6, gate 2: the pretrain path builds every backbone at
    #: every size and the optimiser sees the parameters. Construction and
    #: registration ONLY -- training stability at interpolated positions is
    #: Phase 6's first pretraining run, not a gate.
    "roadb_pretrain_build_gate": {
        "kind": Field(str, choices=("roadb_pretrain_build_gate",)),
        "sizes": Field(list, required=False, default=(224, 512, 768)),
        "backbones": Field(
            list, required=False,
            default=("vit_b16", "swin_b", "srgnn", "agnet"),
        ),
        "pretrained": Field(bool, required=False, default=True),
    },
    #: Phase 8: Grad-CAM for arm A, gated by the randomisation test.
    #:
    #: **The target layer, the sample rule, the head-selection rule and the
    #: pass criterion are NOT config fields.** They live in ``cleft.phase8``
    #: where the suite checks them -- a pre-registration a config could edit is
    #: not one, and the layer in particular is the field most likely to be
    #: nudged because a map looked better (``phase8.GRAD_CAM_TARGET``).
    #:
    #: What the config supplies is the data and arm A's own hyperparameters,
    #: which must match ``p7_d1_vit_b16_imagenet_g1`` or the refit gate fails.
    "grad_cam": {
        "kind": Field(str, choices=("grad_cam",)),
        "manifest_artifact": Field(str, doc="input name of data/manifests/<version>"),
        "staged_artifact": Field(str, doc="input name of data/staged/<version>"),
        "embeddings_artifact": Field(str, doc="input name of the arm's embedding set"),
        "geometry": Field(str, choices=("g1", "g2")),
        "label": Field(str),
        "backbone": Field(str),
        "seeds": Field(list),
        "inner_val_frac": Field(float),
        "max_epochs": Field(int),
        "patience": Field(int),
        "monitor": Field(str),
        "learning_rate": Field(float),
        "weight_decay": Field(float),
        "batch_size": Field(int),
    },
    #: Phase 8b: the Grad-CAM VARIANT -- softmax weighting, negatives
    #: dropped -- a SECOND METHOD beside "grad_cam", never a modification.
    #: The two deltas, the gate form and both committed readings live in
    #: ``phase8.GRAD_CAM_SOFTMAX_REGISTERED``, not here; the config supplies
    #: the same data and arm-A hyperparameters the original's does, or the
    #: refit gate fails identically.
    "grad_cam_softmax": {
        "kind": Field(str, choices=("grad_cam_softmax",)),
        "manifest_artifact": Field(str, doc="input name of data/manifests/<version>"),
        "staged_artifact": Field(str, doc="input name of data/staged/<version>"),
        "embeddings_artifact": Field(str, doc="input name of the arm's embedding set"),
        "geometry": Field(str, choices=("g1", "g2")),
        "label": Field(str),
        "backbone": Field(str),
        "seeds": Field(list),
        "inner_val_frac": Field(float),
        "max_epochs": Field(int),
        "patience": Field(int),
        "monitor": Field(str),
        "learning_rate": Field(float),
        "weight_decay": Field(float),
        "batch_size": Field(int),
    },
    #: Phase 8c, the static sheet: a RENDERING of existing artifacts --
    #: nothing measured (``phase8c.SHEET_DESIGN``). The panel content rules
    #: (22 regions, stamps, both-numbers sentence, flatness axis) live in
    #: ``cleft.phase8c`` where the suite checks them.
    "phase8c_sheet": {
        "kind": Field(str, choices=("phase8c_sheet",)),
        "manifest_artifact": Field(str),
        "staged_artifact": Field(str),
        "methods_artifact": Field(str, doc="input name of grad_cam_methods.npz"),
        "node_weights_artifact": Field(str, doc="input name of node_weights.npz"),
        #: [2026-08-15, the clinical rebuild] The animation run's directory:
        #: the pages EMBED its APNGs, so the bytes are guard-3-verified like
        #: every other input. Ordering: animation rerun -> declare -> sheet.
        "animation_artifact": Field(
            str, doc="input name of the phase8c_animation run directory"
        ),
        "geometry": Field(str, choices=("g1", "g2")),
        "label": Field(str),
        #: The five per-seed predictions.csv inputs, and the five curves.csv
        #: inputs -- named lists so guard 3 verifies every file the panels
        #: read.
        "prediction_inputs": Field(list),
        "curve_inputs": Field(list),
    },
    #: Phase 8c, the animation: measured frames over three real axes
    #: (``phase8c.ANIMATION_REGISTERED``). Refits arm A (same gate), so it
    #: carries the same hyperparameter block as grad_cam.
    "phase8c_animation": {
        "kind": Field(str, choices=("phase8c_animation",)),
        "manifest_artifact": Field(str),
        "staged_artifact": Field(str),
        "embeddings_artifact": Field(str),
        "methods_artifact": Field(str, doc="endpoint frames come from here"),
        "geometry": Field(str, choices=("g1", "g2")),
        "label": Field(str),
        "backbone": Field(str),
        "seeds": Field(list),
        "inner_val_frac": Field(float),
        "max_epochs": Field(int),
        "patience": Field(int),
        "monitor": Field(str),
        "learning_rate": Field(float),
        "weight_decay": Field(float),
        "batch_size": Field(int),
    },
    #: Phase 8 arm B: SR-GNN's node weights -- the "regions voting"
    #: explanation -- gated by a refit and a randomisation test.
    #:
    #: **The gate values, the sample sizes, the whole-image split and the
    #: randomisation modes are NOT config fields.** They live in
    #: ``cleft.phase8`` and ``cleft.node_weights`` where the suite checks
    #: them; a config that could edit 0.1719, or choose which randomisation
    #: to run, is not a pre-registration.
    #:
    #: What the config supplies is the data and arm B's own hyperparameters,
    #: which must match ``p7_d1_srgnn_imagenet_g1_native`` or the refit gate
    #: fails after ten seeds of fitting.
    "node_weights": {
        "kind": Field(str, choices=("node_weights",)),
        "manifest_artifact": Field(str, doc="input name of data/manifests/<version>"),
        "staged_artifact": Field(str, doc="input name of data/staged/<version>"),
        "embeddings_artifact": Field(str, doc="input name of the arm's embedding set"),
        #: **The component-role CONTROL** -- the same arm's map re-extracted
        #: from a randomised backbone (``extract.RANDOMISED_BACKBONE``).
        #: Without it that mode does not run, and the A<->B comparison is
        #: reported UNAVAILABLE rather than unresolved. ``check_pairing``
        #: refuses a real set here and refuses this one everywhere else.
        "randomised_embeddings_artifact": Field(
            str, required=False, default=None,
            doc="input name of the randomised-backbone control set",
        ),
        #: Present only for a pretrained init; arm B is imagenet and has none.
        "checkpoint": Field(str, required=False, default=None),
        "geometry": Field(str, choices=("g1", "g2")),
        "label": Field(str),
        "backbone": Field(str, choices=("srgnn", "agnet")),
        "init": Field(str),
        "region_scheme": Field(str),
        #: **One choice, deliberately.** The refit gate is arm B's 0.1719,
        #: which only the third regime reproduces -- the probe diagnostics
        #: score 0.1340 and 0.0468 and would fail it after ten seeds of
        #: fitting. Explaining a probe is a coherent thing to want and needs
        #: its own gate value, so it costs an edit here rather than a config
        #: field nobody would notice.
        "trainable": Field(
            str, required=False, default="graph_layers",
            choices=("graph_layers",),
        ),
        "deterministic": Field(bool, required=False, default=None),
        "seeds": Field(list),
        "inner_val_frac": Field(float),
        "max_epochs": Field(int),
        "patience": Field(int),
        "monitor": Field(str),
        "learning_rate": Field(float),
        "weight_decay": Field(float),
        "batch_size": Field(int),
    },
    #: PLAN §4.3 condition 1 for the LADDER's claims, over vectors that
    #: already exist. Fits nothing. The pair list is not a config field --
    #: ``ladder.paired_claim_pairs(scope)`` derives it, so a config cannot
    #: quietly add or drop a comparison. ``scope`` picks how much to answer:
    #: ``headline`` is the 1.05x claim alone, two run directories.
    "paired_claims": {
        "kind": Field(str, choices=("paired_claims",)),
        #: "roadb_resolution" is Road B's scope: same task, same paired
        #: BCa, same loader -- only the PAIR ENUMERATION differs, and it
        #: lives in roadb (that road's structure is its own file's
        #: business). A second paired-comparison implementation is the
        #: duplication R10 forbids; a scope is not.
        "scope": Field(
            str,
            choices=(
                "headline", "ladder", "tradeoff",
                "roadb_resolution", "roadb_regioncrop",
                #: Phase 7D's six contrasts (ladder.P7D_PAIRED_CONTRASTS),
                #: including the cross-phase pair against Road B's
                #: patch16@512 arm on the five shared seeds.
                "p7d",
                #: Phase 10's single pair -- the CleftGNN replication
                #: against the 0.2520 arm, five shared seeds, same 237
                #: patients and same fold column. Its enumeration lives in
                #: phase10 on the same terms as Road B's
                #: (phase10.PHASE_10_PAIRED_REGISTERED); the phase's
                #: fourth exit criterion and its last outstanding run.
                "p10",
                #: Phase 11's loss contrast -- the IEM arm against its
                #: MATCHED MSE control, same cell, seeds, folds and
                #: target, differing in the loss alone. The ONLY scope
                #: that reports two metrics, because IEM alone would be
                #: marking its own homework
                #: (phase11.PAIRED_CLAIM_COVERAGE).
                "p11_loss",
                #: Phase 12's three view contrasts -- B vs A, C vs A,
                #: C vs D -- on the shared five seeds over the 236
                #: both-views cohort; enumeration in
                #: phase12.paired_claim_pairs (phase12.STOP_4_REGISTERED).
                "p12",
                #: [ADDED 2026-09-06] Phase 27's single pair -- the
                #: anchor-trained head against the 0.2520 arm on the
                #: five shared seeds, same 237 patients and same truth
                #: column. Its enumeration lives in phase27 on the same
                #: terms as Phase 10's. It is the phase's LAST owed
                #: computation: condition 1 was UNCOMPUTED at the close
                #: and the ledger row is DESCRIPTIVE until it exists
                #: (phase27.THE_CONDITION_1_REQUIREMENT).
                "p27",
            ),
        ),
        "n_boot": Field(int, required=False, default=10000),
        #: p11_loss only: IEM's target is the sheet's MEDIAN GRADE, which
        #: the prediction CSVs do not carry -- their truth column is the
        #: panel mean, and that carries the PCC half.
        "manifest_artifact": Field(str, required=False),
        "scoresheet_artifact": Field(str, required=False),
    },
    # Phase 11's pre-step: the asymmetric error on predictions already on
    # disk, both direction conventions, DESCRIPTIVE only
    # (phase11.PRE_STEP_REGISTERED). It fits nothing -- every vector it
    # reads was written by a keeper run that already happened -- and it
    # declares the manifest and the score sheet because IEM's G is the
    # sheet's MEDIAN GRADE, not the CSVs' panel-mean truth column.
    "iem_prestep": {
        "kind": Field(str, choices=("iem_prestep",)),
        "manifest_artifact": Field(str, doc="input name of data/manifests/<version>"),
        "scoresheet_artifact": Field(
            str,
            doc="input name of the primary score sheet -- its Median "
                "column is G, the grade IEM is defined against",
        ),
        "shared_seeds": Field(
            list,
            doc="the band every arm has, reported beside each arm's own "
                "band so neither choice is silent",
        ),
    },
    # Phase 12, stop 1: derive cleft_v1_views (the 236 both-views cohort)
    # from cleft_v1 -- rows and folds carried verbatim, folder 238
    # dropped, the two known exceptions asserted (phase12.STOP_1_MANIFEST).
    # The expected shape is declared here per configs-not-flags and must
    # agree with views.VIEWS_COHORT or the run refuses.
    "build_views_manifest": {
        "kind": Field(str, choices=("build_views_manifest",)),
        "manifest_artifact": Field(
            str, doc="input name of the SOURCE manifest, cleft_v1"
        ),
        "out_version": Field(
            str, doc="output artifact version; never overwrite an existing one"
        ),
        "expect_patients": Field(int),
        "expect_frontal": Field(int),
        "expect_basal": Field(int),
    },
    # Phase 12, stop 2: stage all 236 basal views at G1 and render the
    # review sheets (phase12.STOP_2_STAGING). The ethics gate is a
    # REQUIRED boolean with NO default: the generator writes false and
    # carries the true, so acknowledgement can only come from a
    # deliberate edit (phase12.BASAL_USE_GATE).
    "stage_basal_views": {
        "kind": Field(str, choices=("stage_basal_views",)),
        "manifest_artifact": Field(
            str, doc="input name of the cleft_v1_views manifest"
        ),
        "patient_folders": Field(
            str, doc="input name of the patient photo tree"
        ),
        "out_version": Field(
            str, doc="output artifact version under data/staged; never "
                     "overwrite an existing one"
        ),
        "expect_patients": Field(int),
        "basal_use_acknowledged": Field(
            bool,
            doc="the maintainer's explicit confirmation that using the "
                "captured-but-unrated submental view is within the REC "
                "approvals -- the task refuses before any pixel while "
                "false, and code never defaults it true",
        ),
    },
    # Phase 12, stop 3: extract the basal embedding set through the
    # existing extraction machinery, and compute the registered
    # staging-geometry confound report where geometry.csv and the labels
    # meet once (phase12.STOP_3_REGISTERED).
    "extract_basal_embeddings": {
        "kind": Field(str, choices=("extract_basal_embeddings",)),
        "manifest_artifact": Field(
            str, doc="input name of the cleft_v1_views manifest"
        ),
        "staged_artifact": Field(
            str, doc="input name of staged_basal_v1"
        ),
        "backbone": Field(str, choices=("vit_b16",)),
        "init": Field(str, choices=("imagenet",)),
        "geometry": Field(str, choices=("g1",)),
        "out_version": Field(
            str, doc="output artifact version under data/embeddings"
        ),
        "expect_patients": Field(int),
        "batch_size": Field(int, required=False, default=32),
    },
    # Phase 12, stop 3: one view-ablation arm. ONE kind for A, B, C and
    # D -- the `views` channel list is the only thing that differs
    # between the four configs, which is what keeps the contrasts
    # one-factor by construction (phase12.STOP_3_REGISTERED).
    "view_arm": {
        "kind": Field(str, choices=("view_arm",)),
        "arm": Field(
            str,
            choices=(
                "a_frontal_only_236", "b_basal_only_236",
                "c_two_view_concat", "d_capacity_control",
            ),
        ),
        "views": Field(
            list,
            doc="channel list, concatenated in order: frontal and/or "
                "basal; d duplicates frontal for C's capacity at A's "
                "information",
        ),
        "manifest_artifact": Field(
            str, doc="input name of the cleft_v1_views manifest"
        ),
        "frontal_embeddings": Field(
            str, required=False,
            doc="input name of the 0.2520 arm's 237-patient set, reused "
                "and aligned by id",
        ),
        "basal_embeddings": Field(
            str, required=False,
            doc="input name of the basal set extracted from staged_basal_v1",
        ),
        "label": Field(str),
        "seeds": Field(list),
        "inner_val_frac": Field(float),
        "max_epochs": Field(int),
        "patience": Field(int),
        "monitor": Field(str, choices=("inner_val_mse", "inner_val_pcc")),
        "learning_rate": Field(float),
        "weight_decay": Field(float),
        "expect_patients": Field(int),
        "expect_head_parameters": Field(int),
    },
    # Phase 13, stop 1: frozen imagenet ViT-B/16 embeddings over the
    # masked SCUT artifact (phase13.SCUT_SHAKEDOWN_PLAN). SHAREABLE
    # throughout -- SCUT is public.
    "extract_scut_embeddings": {
        "kind": Field(str, choices=("extract_scut_embeddings",)),
        "masked_scut": Field(str, doc="input name of data/scut/masked_v1"),
        "scut_root": Field(str, doc="input name of the SCUT root (split)"),
        "backbone": Field(str, choices=("vit_b16",)),
        "init": Field(str, choices=("imagenet",)),
        "geometry": Field(str, choices=("g1",)),
        "out_version": Field(str),
        "expect_train": Field(int),
        "expect_test": Field(int),
        "batch_size": Field(int, required=False, default=32),
    },
    # Phase 13, stop 1: the bottleneck decoder trained on the SCUT fit
    # side, held-out reconstructions, the SCUT band baseline and the
    # comparability table (phase13.STOP_1_BUILT). NO cohort pixel is
    # generated here; the cleft staged artifact feeds CONTENT STATISTICS
    # only. Fixed epoch budget, best checkpoint by inner L2 -- the
    # pretraining discipline, no early stopping.
    "train_scut_decoder": {
        "kind": Field(str, choices=("train_scut_decoder",)),
        "scut_embeddings": Field(str, doc="input name of the SCUT set"),
        "masked_scut": Field(str, doc="input name of data/scut/masked_v1"),
        "scut_root": Field(str, doc="input name of the SCUT root (split)"),
        "cleft_staged": Field(
            str, doc="input name of staged_v1 -- read ONLY for the "
                     "comparability table's content statistics"
        ),
        "geometry": Field(str, choices=("g1",)),
        "epochs": Field(int),
        "batch_size": Field(int),
        "learning_rate": Field(float),
        "inner_val_frac": Field(float),
        "ssim_inner_sample": Field(int),
        "sheet_worst": Field(int),
        "sheet_sample": Field(int),
        "expect_train": Field(int),
        "expect_test": Field(int),
    },
    # Phase 13, stop 2: cohort reconstructions through the shakedown's
    # picked checkpoint, behind the cleft_reconstruction_acknowledged
    # gate (phase13.CLEFT_RECONSTRUCTION_GATE) -- the basal-flag
    # precedent exactly: a REQUIRED boolean with NO default, written
    # false by the generator, carried true only from the edit.
    # Every reconstruction is CLUSTER-ONLY without exception.
    "reconstruct_cohort": {
        "kind": Field(str, choices=("reconstruct_cohort",)),
        "decoder_run": Field(
            str, doc="input name of the shakedown run directory"
        ),
        "checkpoint_epoch": Field(
            int,
            doc="the maintainer's checkpoint pick (phase13.STOP_1_REVIEWED); "
                "the file is checkpoints/<decoder.checkpoint_name>",
        ),
        "cleft_embeddings": Field(
            str, doc="input name of the 0.2520 arm's frozen frontal set "
                     "-- the decoder's own representation",
        ),
        "manifest_artifact": Field(str, doc="input name of cleft_v1"),
        "staged_artifact": Field(
            str, doc="input name of staged_v1 -- the reference crops "
                     "the encoder saw",
        ),
        "geometry": Field(str, choices=("g1",)),
        "cleft_reconstruction_acknowledged": Field(
            bool,
            doc="the maintainer's explicit confirmation that generating "
                "patient-derived reconstructions is within the REC "
                "approvals -- the task refuses before any read while "
                "false, and code never defaults it true",
        ),
        "expect_patients": Field(int),
        "batch_size": Field(int, required=False, default=32),
    },
    # Phase 13, stop 3 checks (a) and (b): the double difference and band
    # error vs grade (phase13.STOP_3_REGISTERED). PURE ARITHMETIC on
    # banked artifacts -- no pixels, no model, so NO GATE: this task
    # cannot regenerate a cohort image.
    "regional_comparison": {
        "kind": Field(str, choices=("regional_comparison",)),
        "cohort_run": Field(
            str, doc="input name of the stop-2 run directory -- its "
                     "cleft_band_errors.json is the numerator side",
        ),
        "decoder_run": Field(
            str, doc="input name of the shakedown run directory -- its "
                     "scut_band_baseline.json is the denominator",
        ),
        "checkpoint_epoch": Field(
            int, doc="asserted against the cohort run's own record: the "
                     "numerator and the declaration name ONE instrument",
        ),
        "scoresheet_artifact": Field(
            str, doc="input name of the primary score sheet -- its "
                     "Median column is the grade",
        ),
        "scoresheet_file": Field(str, required=False, default=None),
        # [2026-08-24] The grade join goes patient -> FRONTAL PHOTO ->
        # Median through the manifest (run.median_by_patient), so the
        # manifest is required: without it the join matched nothing
        # (phase13.READING_APPLIED_TO_NOBODY).
        "manifest_artifact": Field(str, doc="input name of cleft_v1"),
        "expect_patients": Field(
            int, doc="the count every registered reading is declared "
                     "against; a reading is REFUSED below it "
                     "(phase13.READING_COUNT_GUARD)",
        ),
        "n_boot": Field(int, required=False, default=10000),
    },
    # Phase 13, stop 3 check (c): asymmetry retention, the direct test of
    # the eye's claim (phase13.STOP_3_REGISTERED). REGENERATES cohort
    # reconstructions in memory, so it carries the acknowledged gate;
    # nothing is written to disk.
    "asymmetry_retention": {
        "kind": Field(str, choices=("asymmetry_retention",)),
        "decoder_run": Field(str, doc="input name of the shakedown run"),
        "checkpoint_epoch": Field(int),
        "cleft_embeddings": Field(str),
        "manifest_artifact": Field(str),
        "staged_artifact": Field(str),
        "scoresheet_artifact": Field(str),
        "scoresheet_file": Field(str, required=False, default=None),
        "masked_scut": Field(
            str, doc="input name of data/scut/masked_v1 -- the floor's "
                     "originals",
        ),
        "scut_root": Field(str, doc="input name of the SCUT root (split)"),
        "scut_embeddings": Field(
            str, doc="input name of the SCUT set -- the floor's "
                     "reconstructions come from the same checkpoint",
        ),
        # [2026-08-24, defect (c-iii)] Was `scut_floor_sample`, which the
        # generator filled with 237 -- the COHORT's count, crossed in
        # from the wrong variable -- and the task then took an
        # ALPHABETICAL prefix of that size. The floor is now the WHOLE
        # held-out side and this field ASSERTS its size instead of
        # choosing it, so a cohort number can never size a SCUT set.
        "expect_scut_test": Field(
            int, doc="the whole held-out SCUT side the floor covers; "
                     "asserted, never used to subsample",
        ),
        "geometry": Field(str, choices=("g1",)),
        "cleft_reconstruction_acknowledged": Field(
            bool,
            doc="confirmation -- required because this task "
                "regenerates cohort pixels in memory",
        ),
        "expect_patients": Field(int),
        "batch_size": Field(int, required=False, default=32),
    },
    # Phase 13, stop 3 check (d) part 1: re-embed the reconstructions
    # through the same frozen ViT-B/16 (phase13.STOP_3_REGISTERED). The
    # set is PATIENT-DERIVED and CLUSTER-ONLY; the probe is a separate
    # train_cv run on it, the 0.2520 arm's recipe byte-identical.
    "extract_reconstruction_embeddings": {
        "kind": Field(str, choices=("extract_reconstruction_embeddings",)),
        "decoder_run": Field(str, doc="input name of the shakedown run"),
        "checkpoint_epoch": Field(int),
        "cleft_embeddings": Field(str),
        "manifest_artifact": Field(str),
        "staged_artifact": Field(str),
        "backbone": Field(str, choices=("vit_b16",)),
        "init": Field(str, choices=("imagenet",)),
        "geometry": Field(str, choices=("g1",)),
        "out_version": Field(str),
        "cleft_reconstruction_acknowledged": Field(
            bool,
            doc="confirmation -- required because this task "
                "regenerates cohort pixels in memory",
        ),
        "expect_patients": Field(int),
        "batch_size": Field(int, required=False, default=32),
        "extract_batch_size": Field(int, required=False, default=32),
    },
    # Phase 13 addendum P1: the confound ceiling
    # (phase13.CLOSING_ADDENDUM_REGISTERED). POD ARITHMETIC -- pixels and
    # five scalars, closed-form ridge, no backbone and no GPU. NO GATE:
    # nothing here generates or modifies a cohort pixel.
    "confound_ceiling": {
        "kind": Field(str, choices=("confound_ceiling",)),
        "manifest_artifact": Field(
            str, doc="input name of cleft_v1 -- its fold column is the "
                     "partition and its rows carry the frontal photo id",
        ),
        "staged_artifact": Field(str, doc="input name of staged_v1"),
        #: **[2026-08-31] Required exactly when `label` is absent.** The
        #: default target is the MEDIAN grade, which run._grades_for
        #: resolves through this sheet; Arm C reads a manifest column and
        #: never opens it. Both directions enforced by
        #: _check_confound_target.
        "scoresheet_artifact": Field(str, required=False, default=None),
        "scoresheet_file": Field(str, required=False, default=None),
        #: **[2026-08-31] The target, DECLARED, with an unmoved default.**
        #:
        #: Absent is Phase 13's quantity: the median grade. A manifest
        #: column name ("mean") is Phase 20's **Arm C**, the like-for-like
        #: ceiling on the PANEL MEAN (phase20.ARM_C_REGISTERED).
        #:
        #: It is a field rather than a new task kind because the two
        #: differ in the LABEL VECTOR and nothing else -- same five
        #: statistics, same ridge, same folds -- and two ceilings are
        #: only comparable when they came from one recipe. The default
        #: does not move: Phase 13's run is reproducible from its own
        #: config, unchanged.
        "label": Field(
            str, required=False, default=None,
            doc="manifest column; absent means Phase 13's median grade",
        ),
        #: Arm C's reading threshold: how far from Phase 13's +0.1000
        #: counts as "materially different". Declared here so the cell
        #: that fires cannot be chosen after the number is seen.
        "materially_different": Field(float, required=False, default=None),
        "seeds": Field(list),
        "inner_val_frac": Field(float),
        "alpha": Field(float),
        "substantial_pcc": Field(
            float,
            doc="the registered threshold the reading turns on; "
                "declared in the config so it cannot be chosen after "
                "the number is seen",
        ),
        "expect_patients": Field(int),
    },
    # Phase 13 addendum P2: one band-occluded embedding set
    # (phase13.CLOSING_ADDENDUM_REGISTERED). ONE kind, three configs --
    # `occluded_band` is the only key that differs, which is what keeps
    # the three one-factor by construction. The gate rides by
    # inheritance: occlusion MODIFIES cohort pixels in memory.
    "extract_occluded_embeddings": {
        "kind": Field(str, choices=("extract_occluded_embeddings",)),
        "occluded_band": Field(str, choices=("eyes", "nose", "lips")),
        "manifest_artifact": Field(str),
        "staged_artifact": Field(str),
        "backbone": Field(str, choices=("vit_b16",)),
        "init": Field(str, choices=("imagenet",)),
        "geometry": Field(str, choices=("g1",)),
        "out_version": Field(str),
        "cleft_reconstruction_acknowledged": Field(
            bool,
            doc="confirmation, inherited from the cohort "
                "config: the gate rides wherever cohort pixels are "
                "touched",
        ),
        "expect_patients": Field(int),
        "extract_batch_size": Field(int, required=False, default=32),
    },
    # Phase 13 addendum P3: one band-occluded RECONSTRUCTION embedding
    # set (phase13.P3_TRIGGERED). Same one-kind-three-configs shape as
    # P2; regenerates cohort pixels in memory, so the gate is required.
    "extract_occluded_recon_embeddings": {
        "kind": Field(str, choices=("extract_occluded_recon_embeddings",)),
        "occluded_band": Field(str, choices=("eyes", "nose", "lips")),
        "decoder_run": Field(str, doc="input name of the shakedown run"),
        "checkpoint_epoch": Field(int),
        "cleft_embeddings": Field(str),
        "manifest_artifact": Field(str),
        "staged_artifact": Field(str),
        "backbone": Field(str, choices=("vit_b16",)),
        "init": Field(str, choices=("imagenet",)),
        "geometry": Field(str, choices=("g1",)),
        "out_version": Field(str),
        "cleft_reconstruction_acknowledged": Field(
            bool,
            doc="confirmation -- required because this task "
                "regenerates cohort pixels in memory",
        ),
        "expect_patients": Field(int),
        "batch_size": Field(int, required=False, default=32),
        "extract_batch_size": Field(int, required=False, default=32),
    },
    # Phase 13 post-closing check: the asymmetry-encoding probe
    # (phase13.ASYMMETRY_ENCODING_PROBE_REGISTERED). POD ARITHMETIC over
    # the cached embeddings and the BANKED stop-3 scalars -- no pixels,
    # no model, NO GATE. The threshold is declared here so it cannot be
    # chosen after the number is seen.
    "asymmetry_encoding_probe": {
        "kind": Field(str, choices=("asymmetry_encoding_probe",)),
        "asymmetry_run": Field(
            str, doc="input name of the stop-3 asymmetry run directory "
                     "-- its metrics.json carries the banked asym_orig "
                     "scalars, CONSUMED never recomputed",
        ),
        "cleft_embeddings": Field(
            str, doc="input name of the 0.2520 arm's frozen set -- the "
                     "probe reads the representation the decoder read",
        ),
        "manifest_artifact": Field(str, doc="input name of cleft_v1"),
        "seeds": Field(list),
        "inner_val_frac": Field(float),
        "alpha": Field(float),
        "substantial_pcc": Field(
            float,
            doc="the registered threshold the reading turns on; "
                "declared before the number exists",
        ),
        "expect_patients": Field(int),
    },
    # Phase 15, stop 1: the MEBeauty survey (phase15.STOP_1_BUILT).
    # PUBLIC data, SHAREABLE, NO GATE. The frontal threshold is declared
    # here so it exists before the screen runs
    # (phase15.FRONTAL_FRACTION_FIRST); the root is env-portable on the
    # ${CLEFT_SCUT_ROOT} precedent (phase15.LANDING_CORRECTED).
    "survey_mebeauty": {
        "kind": Field(str, choices=("survey_mebeauty",)),
        "mebeauty_root": Field(
            str, doc="input name of the MEBeauty clone root "
                     "(${CLEFT_MEBEAUTY_ROOT})",
        ),
        "published_images": Field(
            int, doc="the paper's count, stated as CONTEXT -- the "
                     "measured count is what gets banked",
        ),
        "frontal_max_offset": Field(
            float,
            doc="the declared pose threshold: normalised centroid "
                "offset of the landmark x-coordinates; declared before "
                "the screen runs, never after the distribution is seen",
        ),
        "sheet_accept": Field(int),
        "sheet_reject": Field(int),
    },
    # Phase 15, stop 2a: verify the dlib-68 mapping before any pixel is
    # staged (phase15.STOP_2A_BUILT). The three thresholds are declared
    # HERE, before the run: if a check fails the MAPPING reopens, never
    # the check (phase15.STOP_2_RULINGS).
    "verify_mebeauty_mapping": {
        "kind": Field(str, choices=("verify_mebeauty_mapping",)),
        "mebeauty_root": Field(str),
        "frontal_max_offset": Field(
            float, doc="stop 1's declared screen, reproduced exactly"
        ),
        "expect_usable": Field(
            int, doc="stop 1's banked usable count; the screen must "
                     "reproduce it or the stops measure different sets",
        ),
        "midline_tolerance": Field(
            float, doc="check 1: median |x| offset of landmarks 33/51 "
                       "from the mirror-pair midline, as a fraction of "
                       "face width",
        ),
        "eye_symmetry_tolerance": Field(
            float, doc="check 2: median eye-centre imbalance about the "
                       "midline, as a fraction of face width",
        ),
        "corner_margin": Field(
            float, doc="check 3: how much wider 48/54 must be than any "
                       "other symmetric mouth pair",
        ),
        "sheet_faces": Field(int),
    },
    # Phase 15, stop 2a-ii: the landmark-QUALITY screen
    # (phase15.STOP_2A_II_BUILT). The pose screen is blind to detector
    # failure; these five checks are not. Every threshold is declared
    # here, before the run -- a face that fails is REJECTED, and the
    # check is never loosened to keep it.
    "screen_mebeauty_landmarks": {
        "kind": Field(str, choices=("screen_mebeauty_landmarks",)),
        "mebeauty_root": Field(str),
        "frontal_max_offset": Field(float),
        "expect_usable": Field(
            int, doc="stop 1's banked count, reproduced or refused"
        ),
        "span_min": Field(
            float, doc="face bbox as a fraction of image side: below "
                       "this is a collapsed detection or a foreign "
                       "coordinate frame",
        ),
        "span_max": Field(float),
        "interocular_min": Field(
            float, doc="eye-centre distance over face span -- "
                       "scale-free, so it tests the constellation's "
                       "shape and not the image's size",
        ),
        "interocular_max": Field(float),
        "bounds_margin": Field(
            float, doc="how far outside the image a point may lie, as "
                       "a fraction of the image side; 0.0 is strict",
        ),
        # [2026-08-24, phase15.BOUNDS_VIOLATION_READINGS] The second
        # tolerance: how MANY points may lie outside at all. Strict (0)
        # until a value is chosen from the MEASURED distribution -- the
        # mechanism ships ahead of the number so the number's arrival
        # is a visible config diff and nothing else.
        "bounds_max_points_outside": Field(
            int, required=False, default=0,
            doc="how many of the 68 points may exceed bounds_margin; "
                "0 is strict, and any other value must be chosen from "
                "the measured violation distribution, never from an "
                "eye pass alone",
        ),
        "diagnose_images": Field(
            list, required=False, default=None,
            doc="images the report must explain BY NAME -- 119.jpg "
                "passed a screen it should never have reached, and "
                "whatever let it through may not be unique to it",
        ),
        "sheet_survivors": Field(int),
        "sheet_rejected": Field(int),
    },
    # Phase 15, stop 2b: stage the usable MEBeauty faces through the
    # SCUT composition (phase15.STOP_2B_BUILT). Both screens are
    # declared so the staged set is REPRODUCED, not re-decided; the
    # split counts are asserted so a shipped-split change cannot pass
    # unnoticed. PUBLIC data, SHAREABLE, no gate.
    "stage_mebeauty": {
        "kind": Field(str, choices=("stage_mebeauty",)),
        "mebeauty_root": Field(str),
        "frontal_max_offset": Field(float),
        "span_min": Field(float),
        "span_max": Field(float),
        "interocular_min": Field(float),
        "interocular_max": Field(float),
        "bounds_margin": Field(float),
        "bounds_max_points_outside": Field(int),
        "expect_survivors": Field(
            int, doc="the banked surviving count at THIS tolerance"
        ),
        "expect_train": Field(int),
        "expect_test": Field(int),
        "out_version": Field(str),
        # [2026-08-24] VARIANTS, not geometries: g1 and g2 are the
        # trapezium-masked pair and `original` is the whole image
        # through the frozen stage -- the third cell SCUT occupies
        # (phase15.STOP_3_AMENDED).
        "variants": Field(list),
        "size": Field(int, required=False, default=224),
        "ar_sampling": Field(str, required=False, default="observed"),
        "sheet_faces": Field(int),
    },
    # Phase 15, stop 3: pretrain on MEBeauty through the Phase 6 recipe
    # (phase15.STOP_3_AMENDED). Only the DATA differs -- every training
    # knob below carries the Phase 6 value, and the generator derives
    # them from the shipped p6 config so "byte-identical" is a fact
    # about the file rather than a claim.
    "pretrain_mebeauty": {
        "kind": Field(str, choices=("pretrain_mebeauty",)),
        "mebeauty_root": Field(str),
        "mebeauty_staged": Field(
            str, doc="input name of the staged artifact, in the SCUT "
                     "masked-artifact format",
        ),
        "source": Field(
            str, choices=("masked_g1", "masked_g2", "masked_original")
        ),
        "backbone": Field(str, choices=("vit_b16",)),
        "region_scheme": Field(str, choices=("none",)),
        "expect_train": Field(int),
        "expect_test": Field(int),
        "epochs": Field(int),
        "inner_val_frac": Field(float),
        "monitor": Field(str),
        "deterministic": Field(bool),
        "learning_rate": Field(float),
        "weight_decay": Field(float),
        "batch_size": Field(int),
        "checkpoint_every": Field(int),
    },
    # Phase 15, stop 4a: cleft-cohort embeddings through a MEBeauty
    # checkpoint (phase15.STOP_4_BUILT). `expect_source` is what the
    # run's OWN metrics.json must record -- the arm-to-run mapping is
    # asserted, never trusted. CLUSTER-ONLY output.
    "extract_mebeauty_embeddings": {
        "kind": Field(str, choices=("extract_mebeauty_embeddings",)),
        "pretrain_run": Field(
            str, doc="input name of the MEBeauty pretraining run dir"
        ),
        "expect_source": Field(
            str,
            choices=("masked_g1", "masked_g2", "masked_original"),
            doc="the source the run's own metrics.json must record",
        ),
        "init": Field(
            str, choices=("mebeauty_masked", "mebeauty_original"),
            doc="MEBeauty's OWN init namespace, never the ladder's",
        ),
        "geometry": Field(str, choices=("g1", "g2")),
        "manifest_artifact": Field(str),
        "staged_artifact": Field(str),
        "backbone": Field(str, choices=("vit_b16",)),
        "out_version": Field(str),
        "expect_patients": Field(int),
        "batch_size": Field(int, required=False, default=32),
    },
    # Phase 15, stop 4b: A's probe on a MEBeauty set. ITS OWN KIND, by
    # the ruling -- reusing train_cv would widen the ladder's
    # init vocabulary, and a shared vocabulary is what lets an
    # undeclared swap manufacture a factor effect. The recipe knobs
    # below are DERIVED from the shipped p7 arm; the comparator and its
    # spread are DECLARED before any number exists.
    # A post-Phase-15-closing addendum on the best arm's EXISTING
    # predictions (classification.CLASSIFICATION_METRICS_SECONDARY).
    # Pod arithmetic: no training, no inference. Every knob that shapes
    # a number is declared here so it is fixed BEFORE the run -- the
    # discretisation's class count, the seeds, and the majority floor
    # the task asserts it reproduces.
    # Phase 16: the 25 Deall anchors, persisted ONCE
    # (phase16.ANCHOR_ARTIFACT_PERSISTED). CLUSTER-ONLY output.
    "extract_anchor_embeddings": {
        "kind": Field(str, choices=("extract_anchor_embeddings",)),
        "deall_artifact": Field(str),
        "manifest_artifact": Field(str),
        "staged_artifact": Field(str),
        "embeddings_artifact": Field(
            str, doc="the cached cohort set the parity check compares "
                     "a live re-extraction against"),
        "geometry": Field(str, choices=("g1",)),
        "backbone": Field(str, choices=("vit_b16",)),
        "batch_size": Field(int),
        "out_version": Field(str),
        "expect_anchors": Field(int, choices=(25,)),
    },
    # Phase 16: the anchor loop (phase16.EXIT_CRITERIA, locked
    # 2026-08-29). **The forbidden variants are absent BY SURFACE**: no
    # key can select a target type (continuous is the only one), a W
    # shape (full linear is the only one), a fold policy (folds come
    # from the manifest, always), or early stopping (patience does not
    # exist here -- a config that writes it is refused as an unknown
    # key). Every scientific setting is required with no default
    # (ANCHOR_LOOP_RULINGS['weight_decay_is_a_scientific_setting'],
    # TAU_DECLARED['never_tuned']).
    "anchor_loop": {
        "kind": Field(str, choices=("anchor_loop",)),
        "arm": Field(str),
        "probe_run": Field(
            str, doc="declared input: the 0.2520 probe's run directory, "
                     "whose per-seed OOF predictions the primary "
                     "contrast pairs against"),
        "anchor_embeddings": Field(str),
        "embeddings_artifact": Field(str),
        "manifest_artifact": Field(str),
        "seeds": Field(list),
        "max_epochs": Field(int),
        "learning_rate": Field(float),
        "batch_size": Field(int),
        "inner_val_frac": Field(float),
        "monitor": Field(str, choices=("inner_val_mse",)),
        "lambda_identity": Field(
            float, doc="the identity-decay strength -- a SCIENTIFIC "
                       "SETTING, never tuned across runs; movement is "
                       "a dated amendment"),
        "tau_scale": Field(
            float, doc="multiplies the frozen anchor-cloud scale "
                       "(phase16.TAU_DECLARED); a scientific setting "
                       "under the same clause"),
        "n_boot": Field(int, choices=(10000,)),
    },
    # Phase 17 arm A (phase17.EXIT_CRITERIA): frozen embeddings of the
    # TPS synthesis set, linear head, MAGNITUDE-MAPPED labels. Zero real
    # patient images in training; all 237 pure test.
    "tstr_regression": {
        "kind": Field(str, choices=("tstr_regression",)),
        "arm": Field(str, choices=("p17_arm_a",)),
        "synth_set": Field(str),
        "manifest_artifact": Field(str),
        "embeddings_artifact": Field(str),
        "backbone": Field(str, choices=("vit_b16",)),
        "seeds": Field(list),
        "max_epochs": Field(int),
        "learning_rate": Field(float),
        "batch_size": Field(int),
        "inner_val_frac": Field(float),
        "monitor": Field(str, choices=("inner_val_mse",)),
        "extract_batch_size": Field(int, required=False, default=32),
    },
    # Phase 17 arms B/C: frozen branches over fixed-midline left/right
    # views, trained 768x768 linear projection, contrastive loss,
    # distance readout normalized-and-rounded
    # (phase17.DECLARED_SETTINGS_17). Which arm a config is comes from
    # the synthesis set it DECLARES: piecewise-affine -> B, TPS -> C.
    # **branch_trainability's vocabulary is ("frozen",) BY RULING** --
    # trainable branches are the unexercised regime (three novelties at
    # once, unattributable; phase17.PHASE_17_RULINGS
    # ['trainable_branches_unexercised']), so a config that writes
    # "trainable" is REFUSED by the choices check.
    "siamese_contrastive": {
        "kind": Field(str, choices=("siamese_contrastive",)),
        "arm": Field(str, choices=("p17_arm_b", "p17_arm_c")),
        "synth_set": Field(str),
        "manifest_artifact": Field(str),
        "staged_artifact": Field(str),
        "geometry": Field(str, choices=("g1",)),
        "backbone": Field(str, choices=("vit_b16",)),
        "branch_trainability": Field(str, choices=("frozen",)),
        "margin": Field(
            float, doc="a SCIENTIFIC SETTING (DECLARED_SETTINGS_17): "
                       "the projection space's unit; never tuned"),
        "pair_rule": Field(
            str, choices=("symmetric_if_zero_magnitude",),
            doc="the declared pair-construction rule"),
        "readout_normalization": Field(
            str, choices=("round_1_plus_4_min_d_over_margin",),
            doc="the declared distance-to-grade rule"),
        "projection_dim": Field(int, choices=(768,)),
        "seeds": Field(list),
        "max_epochs": Field(int),
        "learning_rate": Field(float),
        "batch_size": Field(int),
        "inner_val_frac": Field(float),
        "monitor": Field(str, choices=("inner_val_loss",)),
        "extract_batch_size": Field(int, required=False, default=32),
    },
    # Phase 17: the five-contrast family and nothing beyond it.
    "tstr_family_analysis": {
        "kind": Field(str, choices=("tstr_family_analysis",)),
        "arm_a_run": Field(str),
        "arm_b_run": Field(str),
        "arm_c_run": Field(str),
        "probe_run": Field(str),
        "manifest_artifact": Field(str),
        "seeds": Field(list),
        "n_boot": Field(int, choices=(10000,)),
    },
    # Phase 18: the metric-space ablation (phase18.EXIT_CRITERIA,
    # locked 2026-08-30). Pure CPU arithmetic over declared run
    # directories; no patient images, no training. The arm list is the
    # LOCKED enumeration -- the generator derives it from
    # phase18.ARM_LIST_LOCKED, and the task resolves arms ONLY through
    # this list (never a glob).
    "metric_space_analysis": {
        "kind": Field(str, choices=("metric_space_analysis",)),
        "manifest_artifact": Field(str),
        "arms": Field(list, item_spec={
            "name": Field(str),
            "input": Field(str, doc="declared-input name of the run dir"),
            "seeds": Field(list),
            "csv": Field(str, choices=("predictions", "identity_predictions")),
            "group": Field(str),
            #: The arm's DOCUMENTED row count per seed -- 237 for the
            #: cohort arms, 236 for p12's both-views cohort
            #: (phase12.STOP_1_MANIFEST). The task's guard asserts
            #: against THIS, per group, never a blanket 237. Unpinned
            #: here so fixture-scale tests can drive the task; the
            #: SHIPPED config's values are pinned by test to {236, 237}.
            "n_patients": Field(int),
        }),
        "contrasts": Field(list, item_spec={
            "name": Field(str),
            "winner": Field(str),
            "baseline": Field(str),
            "pcc_verdict": Field(
                str, choices=("unresolved", "claimable_negative"),
                doc="the LEDGER's verdict, carried -- never recomputed"),
        }),
        "n_boot": Field(int, choices=(10000,)),
    },
    # [2026-08-31] The standing record-versus-artifact check
    # (record_audit.STANDING_CHECK_DESIGN). A job rather than a suite
    # test because the prediction CSVs are CLUSTER-ONLY. Deliberately
    # thin: it declares run dirs and a threshold, and takes the arms'
    # banked values from the RECORDS at run time -- putting banked
    # figures in a config would make the check compare a document
    # against an artifact instead of the record against the artifact.
    "record_artifact_check": {
        "kind": Field(str, choices=("record_artifact_check",)),
        "manifest_artifact": Field(str),
        "arms": Field(list, item_spec={
            "name": Field(str),
            "input": Field(str, doc="declared-input name of the run dir"),
            "seeds": Field(list),
            "csv": Field(str, choices=("predictions", "identity_predictions")),
            "n_patients": Field(int),
        }),
        #: Pinned to record_audit.AGREEMENT_THRESHOLD. A configurable
        #: tolerance is a tolerance someone widens until the check
        #: passes, which is the one way this check can fail silently.
        "threshold": Field(float, choices=(0.005,)),
    },
    "classification_metrics": {
        "kind": Field(str, choices=("classification_metrics",)),
        "arm": Field(str, doc="the arm these predictions came from"),
        "arm_run": Field(str, doc="declared input: that arm's run directory"),
        "manifest_artifact": Field(str),
        "seeds": Field(list),
        "n_classes": Field(
            int, choices=(3,),
            doc="THREE. Two thresholds cut a line into three parts; a "
                "5-class family would need four thresholds and only two "
                "are registered (classification.THREE_NOT_FIVE)",
        ),
        "expect_majority": Field(
            float,
            doc="phase9's banked class3 majority floor. The task REFUSES "
                "if the truth column does not reproduce it -- a metric "
                "quoted against the wrong floor is worse than none",
        ),
    },
    "probe_mebeauty": {
        "kind": Field(str, choices=("probe_mebeauty",)),
        "mebeauty_embeddings": Field(str),
        "manifest_artifact": Field(str),
        "staged_artifact": Field(str),
        "init": Field(str, choices=("mebeauty_masked", "mebeauty_original")),
        "geometry": Field(str, choices=("g1", "g2")),
        "label": Field(str),
        "backbone": Field(str, choices=("vit_b16",)),
        #: A's vocabulary, not a wider one: the recipe is DERIVED from
        #: the shipped ladder arm, so the field that says how much of
        #: the backbone moves must be constrained exactly as train_cv
        #: constrains it. Unconstrained, a typo here would train an arm
        #: nobody configured and still pass the difference-set test,
        #: because that test compares to whatever A's file says.
        "trainable": Field(str, choices=("head", "full")),
        "patch_scheme": Field(str),
        "feature_source": Field(str),
        "seeds": Field(list),
        "max_epochs": Field(int),
        "patience": Field(int),
        "inner_val_frac": Field(float),
        "monitor": Field(str),
        "learning_rate": Field(float),
        "weight_decay": Field(float),
        "batch_size": Field(int),
        "comparator_init": Field(
            str,
            choices=("scut_masked", "scut_original", "imagenet"),
            doc="the SAME-NAMED banked cell this arm is read against",
        ),
        "comparator_geometry": Field(str, choices=("g1", "g2")),
        "comparator_sd": Field(
            float, doc="the comparator's own five-seed SD, from the "
                       "ladder -- the threshold uses BOTH arms' spreads",
        ),
        "comparator_n": Field(int),
    },
    # Phase 11's loss arm and its MATCHED MSE control
    # (phase11.PHASE_11_BUILD_REGISTERED). One kind, two configs: `loss`
    # is the only key that differs between them, which is what makes the
    # control matched rather than merely similar. The score sheet is
    # declared because the TARGET is its Median grade -- eq (16) is
    # defined against a consensus grade -- while evaluation stays PCC
    # against the panel mean.
    "iem_arm": {
        "kind": Field(str, choices=("iem_arm",)),
        "loss": Field(
            str, choices=("iem", "mse"),
            doc="iem is eq (16) verbatim under convention A; mse is the "
                "matched control, identical in every other respect",
        ),
        "manifest_artifact": Field(str, doc="input name of data/manifests/<version>"),
        "staged_artifact": Field(str, doc="input name of data/staged/<version>"),
        "embeddings_artifact": Field(
            str, doc="input name of the 0.2520 arm's frozen embeddings"
        ),
        "scoresheet_artifact": Field(
            str, doc="input name of the primary sheet -- its Median column "
                     "is the training target"
        ),
        "geometry": Field(str, choices=("g1", "g2")),
        "label": Field(str),
        "seeds": Field(list),
        "inner_val_frac": Field(float),
        "max_epochs": Field(int),
        "patience": Field(int),
        "monitor": Field(str, choices=("inner_val_mse", "inner_val_pcc")),
        "learning_rate": Field(float),
        "weight_decay": Field(float),
    },
    "phase7c_paired": {
        "kind": Field(str, choices=("phase7c_paired",)),
        "round": Field(str, choices=("selected30", "matched3")),
        "seeds": Field(list),
        "n_boot": Field(int, required=False, default=10000),
    },
    # Phase 5 SS2: the masked SCUT variants, plus the sheet that shows what they
    # look like beside the placement they came from. SHAREABLE -- SCUT is public.
    "masked_scut": {
        "kind": Field(str, choices=("masked_scut",)),
        "scut_root": Field(str, doc="input name of the SCUT-FBP5500_v2 root"),
        "geometries": Field(list, required=False, default=("g1", "g2")),
        #: Road B's resolution axis (roadb.PHASE_6_STRUCTURE): the frozen
        #: composition at a new size. Every SCUT source is 350x350, so any
        #: size above that interpolates ALL faces uniformly -- the sheet and
        #: metrics say so rather than let it pass unremarked.
        "size": Field(int, required=False, default=224),
        #: 0 builds every face. A small number is the sheet-only mode.
        "n_faces": Field(int, required=False, default=0),
        "n_sheet": Field(int, required=False, default=6),
        #: How the per-face aspect ratio is drawn from the cleft distribution.
        #:
        #: "observed"      bootstrap with replacement over the 237 raw ratios --
        #:                 the correct one. Reproduces the cohort exactly,
        #:                 including the sparse tail.
        #: "empirical"     inverse CDF over 21 percentiles. SUPERSEDED: realises
        #:                 2.75% landscape against the cohort's 0.42%, because
        #:                 interpolation invents a uniform tail across
        #:                 [p95, p100]. See ar_distribution.AR_TAIL_DEFECT.
        #: "uniform_range" uniform over [min, max]. Was called "sampled", a name
        #:                 that read as though it sampled the cohort; it gives
        #:                 mean 0.826 against the true 0.7475.
        #: "fixed_median"  every face at the median: the no-variation control.
        #:
        #: The superseded modes are kept so the runs made with them stay
        #: re-derivable. The DEFAULT here is deliberately the schema's own and is
        #: overridden by masked.DEFAULT_AR_SAMPLING at the call site; both shipped
        #: configs set it explicitly, and a test asserts they say "observed".
        "ar_sampling": Field(
            str, required=False, default="observed",
            choices=("observed", "empirical", "uniform_range", "fixed_median"),
        ),
        #: Off by default: the sheet is the cheap review pass, and writing an
        #: immutable artifact is a deliberate act. **A FULL build (n_faces: 0)
        #: with this false is refused at runtime**: Phase 5 ran that silent
        #: combination three times, and the artifact the pretraining configs
        #: declared had never existed.
        "write_artifact": Field(bool, required=False, default=False),
        "out_version": Field(str, required=False, default="masked_v1"),
        #: Faces between checkpoints when writing the artifact. A Run:AI pause
        #: reattaches the run directory but NOT the work (PLAN §2.7); without
        #: this a paused build silently restarts at face 0.
        "checkpoint_every": Field(
            int, required=False, default=250,
            doc="faces between checkpoints; the resume granularity",
        ),
    },
    # Phase 5 §5: unilateral cleft-like asymmetry by thin-plate spline
    # (PLAN §4.11 Q-b). Deformation only -- no scar, and the magnitude-to-grade
    # mapping is an assumption. Both are recorded in synthesis.LIMITATIONS and
    # carried into metrics.json before the arm runs.
    "asymmetry_synthesis": {
        "kind": Field(str, choices=("asymmetry_synthesis",)),
        "scut_root": Field(str, doc="input name of the SCUT-FBP5500_v2 root"),
        "geometries": Field(list, required=False, default=("g1", "g2")),
        #: 0 builds every face. A small number is the sheet-only mode.
        "n_faces": Field(int, required=False, default=0),
        "n_sheet": Field(int, required=False, default=6),
        "ar_sampling": Field(
            str, required=False, default="empirical",
            choices=("observed", "empirical", "uniform_range", "fixed_median"),
        ),
        #: [2026-08-29, Phase 17] The warp family. Default "tps" keeps
        #: every shipped p5 config byte-identical; "piecewise_affine" is
        #: arm B's family (scut.piecewise -- same control points, same
        #: signature), so B-vs-C isolates the transformation family.
        "warp_family": Field(
            str, required=False, default="tps",
            choices=("tps", "piecewise_affine"),
        ),
        #: Displacement in units of crop width. An ORDERED series, not a graded
        #: one: nothing here relates a displacement to an Asher-McDade grade.
        #: See synthesis.LIMITATIONS["magnitude_to_grade_is_an_assumption"].
        #:
        #: **[MEASURED 2026-07-30] The upper end is bounded by anchor spacing,
        #: not by taste.** displacement/nearest-anchor scales at 13.2x the
        #: magnitude, so the cupid's bow peak reaches ratio 1.0 at ~0.076 -- it
        #: is displaced further than the distance to its nearest anchor, and the
        #: field folds into a chevron instead of deforming. Visible on the sheet
        #: at 0.09; confirmed to be the field and not the resampling, since the
        #: shape is identical under bilinear sampling. The default series stays
        #: under ratio 0.5 (magnitude 0.038). See
        #: synthesis.MAX_DISPLACEMENT_TO_ANCHOR_RATIO.
        "magnitudes": Field(
            list, required=False, default=(0.0, 0.015, 0.025, 0.035),
            doc="0.0 is the undeformed reference and must be present",
        ),
        #: Arrays stream into memmaps as they are produced, so the cross product
        #: is DISK rather than memory: the full set at four magnitudes and both
        #: geometries is ~6.6 GiB written, one face resident. Set above the real
        #: build on purpose -- it catches a runaway cross product while the run is
        #: still cheap to re-scope, and must not be what decides how many
        #: magnitudes the arm has.
        "disk_budget_bytes": Field(
            int, required=False, default=16 * 1024**3,
            doc="refuse a run whose staged arrays would exceed this on disk",
        ),
        #: How often the build checkpoints, in faces. A Run:AI pause reattaches
        #: the run directory but NOT the work (PLAN §2.7), so without this a
        #: paused build silently restarts at face 0. Small enough that a pause
        #: loses little, large enough that checkpointing is not the cost.
        "checkpoint_every": Field(
            int, required=False, default=250,
            doc="faces between checkpoints; the resume granularity",
        ),
        #: Off by default: the sheet is the cheap review pass, and writing an
        #: immutable artifact is a deliberate act.
        "write_artifact": Field(bool, required=False, default=False),
        "out_version": Field(str, required=False, default="synth_v1"),
    },
    # Phase 6: full fine-tuning on SCUT, the official split, MSE on the 1-5
    # label. Twelve shipped configs -- 4 backbones x 3 sources -- each differing
    # from its lattice neighbours in exactly one field (backbone or source),
    # asserted by test.
    "pretrain": {
        "kind": Field(str, choices=("pretrain",)),
        "scut_root": Field(str, doc="input name of the SCUT-FBP5500_v2 root"),
        #: "stub" is the laptop path: the loop, the checkpoint wiring and the
        #: band check are all exercised without torch.
        "backbone": Field(
            str, choices=("vit_b16", "swin_b", "srgnn", "agnet", "stub")
        ),
        #: ONE field, not two (original-vs-masked plus a geometry), so every
        #: config differs from its lattice neighbours in exactly one field.
        #: Masked sources additionally require an input named "masked_scut";
        #: original configs must NOT declare it -- an input the run never reads
        #: would still enter inputs.json as if it fed the run.
        "source": Field(
            str,
            choices=("original", "masked_g1", "masked_g2", "masked_original"),
        ),
        #: [DECIDED 2026-07-31] The scheme axis, GRAPH backbones only:
        #: "native" is the architecture's own generator (SR-GNN grid-26,
        #: AG-Net SIFT+GMM); grid/anatomy/random are Phase 2's generators
        #: mapped through each face's recorded content box. Transformers and
        #: the stub declare "none"; the loop refuses a scheme that cannot act
        #: on its backbone. The same scheme must be used in pretraining and
        #: cleft fine-tuning. [MEASURED, PLAN §4.4] scheme comparisons are
        #: read at G2 -- at G1 the schemes carry different background
        #: exposure, so a G1 scheme difference is partly a background
        #: difference.
        "region_scheme": Field(
            str, choices=("none", "native", "grid", "anatomy", "random")
        ),
        #: The official split's sizes after CM152 exclusion, declared and then
        #: asserted at runtime (smoke's n_samples pattern). The shipped configs
        #: say 3300/2199 -- asserted by test; laptop fixtures declare their own.
        "expect_train": Field(int, doc="split rows on the train side, post-exclusion"),
        "expect_test": Field(int, doc="split rows on the test side, post-exclusion"),
        #: [DECIDED 2026-07-31] The FIXED budget: every run trains exactly
        #: this many epochs, keeping the best checkpoint by `monitor`. There
        #: is deliberately NO patience field -- patience terminated on noise
        #: in a flat region and made run length track luck rather than
        #: scheme; the eight pre-policy runs are superseded. See
        #: train/pretrain.py FIXED_BUDGET_POLICY for why this does not
        #: contradict Phase 3 gate 4 (whose collapse was a 152-sample cleft
        #: behaviour; SCUT's 3,300 shows none, and early stopping stays for
        #: the cleft ladder).
        "epochs": Field(int),
        "inner_val_frac": Field(float),
        #: Best-checkpoint selection. inner_val_pcc in all thirty configs:
        #: the primary metric, on a train-side split, test disjoint and
        #: touched once.
        "monitor": Field(str, choices=("inner_val_mse", "inner_val_pcc")),
        #: [DECIDED 2026-07-31, from image measurements] The twelve shipped
        #: configs say FALSE: flag-off, vit_b16/swin_b/srgnn are bitwise in
        #: the image and the flag cannot fix agnet there (the deterministic
        #: roi_align substitution needs a C++ compiler the image lacks).
        #: AG-Net is recorded as deterministic to ~2.3e-05 instead --
        #: models/agnet.py TRAINING_DETERMINISM. REQUIRED so every run states
        #: which reproducibility regime its numbers were produced under; true
        #: routes through the frozen determinism.configure.
        "deterministic": Field(bool),
        #: All REQUIRED, no schema defaults, deliberately: _validate_mapping
        #: fills absent optionals with the schema default, which is how
        #: train_cv's learning_rate once trained at a rate nobody chose while
        #: metrics.json faithfully recorded it. A scientific knob on a
        #: twelve-run lattice is written down twelve times.
        "learning_rate": Field(float),
        "weight_decay": Field(float),
        "batch_size": Field(int),
        #: Epochs between checkpoints. The durable unit is the EPOCH -- see
        #: train/pretrain.py for why that satisfies the brief's intent.
        "checkpoint_every": Field(int, required=False, default=1),
    },
    # Phase 8, the last original item: t-SNE with its k-NN companion. Only
    # the artifact names live here -- perplexities, seed, k, tie rule and the
    # baselines are REGISTERED details (phase8.TSNE, phase8.TSNE_COMPANION)
    # and a registered detail in a config field is a knob.
    "tsne": {
        "kind": Field(str, choices=("tsne",)),
        "manifest_artifact": Field(str, doc="input name of data/manifests/<version>"),
        "embeddings_artifact": Field(
            str,
            doc="input name of the registered subject set "
                "(phase8.TSNE_COMPANION names it; the task refuses another)",
        ),
    },
    # Phase 8's final addition: the SCUT pretraining animation, one variant
    # per run (phase8.SCUT_ANIMATION_BUILT). The training fields mirror
    # "pretrain" exactly -- the run IS a rerun of the declared Phase 6 cell,
    # with an epoch-end observation hook rendering frames in-run. ViT only:
    # the Grad-CAM target layer is a ViT-block registration.
    "scut_animation": {
        "kind": Field(str, choices=("scut_animation",)),
        "scut_root": Field(str, doc="input name of the SCUT-FBP5500_v2 root"),
        "shipped_artifact": Field(
            str,
            doc="input name of the shipped Phase 6 cell's pretrained.npz -- "
                "the fingerprint target and the endpoint frame's weights",
        ),
        "backbone": Field(str, choices=("vit_b16",)),
        "source": Field(str, choices=("original", "masked_g1")),
        "region_scheme": Field(str, choices=("none",)),
        "expect_train": Field(int),
        "expect_test": Field(int),
        "epochs": Field(int),
        "inner_val_frac": Field(float),
        "monitor": Field(str, choices=("inner_val_mse", "inner_val_pcc")),
        "deterministic": Field(bool),
        "learning_rate": Field(float),
        "weight_decay": Field(float),
        "batch_size": Field(int),
        "checkpoint_every": Field(int, required=False, default=1),
    },
    # Phase 9: the prototypes. Artifact names only -- the space, distance,
    # tie rule, groupings, and both validations are REGISTERED details
    # (phase9.PROTOTYPES_REGISTERED); geometry/label mirror the arm the
    # embeddings came from so the staged faces align.
    "prototypes": {
        "kind": Field(str, choices=("prototypes",)),
        "manifest_artifact": Field(str, doc="input name of data/manifests/<version>"),
        "staged_artifact": Field(str, doc="input name of data/staged/<version>"),
        "embeddings_artifact": Field(
            str,
            doc="input name of the registered space "
                "(phase9.PROTOTYPES_REGISTERED names it; the task refuses "
                "another)",
        ),
        "geometry": Field(str, choices=("g1", "g2")),
        "label": Field(str),
    },
    # Phase 9: the 25-image external reference. The refit block mirrors
    # grad_cam exactly -- same arm, same 0.2520 gate; the set's facts
    # (25 composites, the labels filename, the grade spread, the trap)
    # are REGISTERED details (phase9.DEALL_REFERENCE_REGISTERED/_READS)
    # and asserted at runtime, never config fields.
    "deall_reference": {
        "kind": Field(str, choices=("deall_reference",)),
        "manifest_artifact": Field(str, doc="input name of data/manifests/<version>"),
        "staged_artifact": Field(str, doc="input name of data/staged/<version>"),
        "embeddings_artifact": Field(str, doc="input name of arm A's embedding set"),
        "deall_artifact": Field(
            str,
            doc="input name of the 25-set folder (labels CSV inside; one "
                "hash covers both)",
        ),
        "geometry": Field(str, choices=("g1", "g2")),
        "label": Field(str),
        "backbone": Field(str),
        "seeds": Field(list),
        "inner_val_frac": Field(float),
        "max_epochs": Field(int),
        "patience": Field(int),
        "monitor": Field(str),
        "learning_rate": Field(float),
        "weight_decay": Field(float),
        "batch_size": Field(int),
    },
    # Phase 9: the prototype classifier -- the reading given at supervision of "prototypes".
    # Artifact names, the extraction backbone and its batch size only; the
    # cells, metrics, vote rules, tie rules and the committed prediction
    # are REGISTERED details (phase9.PROTOTYPE_CLASSIFIER_REGISTERED).
    "prototype_classifier": {
        "kind": Field(str, choices=("prototype_classifier",)),
        "manifest_artifact": Field(str, doc="input name of data/manifests/<version>"),
        "staged_artifact": Field(str, doc="input name of data/staged/<version>"),
        "embeddings_artifact": Field(str, doc="input name of arm A's embedding set"),
        "deall_artifact": Field(str, doc="input name of the 25-set anchor folder"),
        "geometry": Field(str, choices=("g1", "g2")),
        "label": Field(str),
        "backbone": Field(str),
        "batch_size": Field(int),
    },
    # Phase 10: the CleftGNN replication under this project's criterion.
    # The architecture, region set, SABM form, CE init and both output
    # readings are REGISTERED details (phase10.PHASE_10_REGISTERED,
    # PHASE_10_UNBLOCKED, CLEFTGNN_BUILT); the recipe knobs appear here
    # because scientific knobs are written down, never defaulted.
    "cleftgnn_cv": {
        "kind": Field(str, choices=("cleftgnn_cv",)),
        "manifest_artifact": Field(str, doc="input name of data/manifests/<version>"),
        "staged_artifact": Field(str, doc="input name of data/staged/<version>"),
        "scoresheet_artifact": Field(
            str,
            doc="input name of the primary score sheet -- the consensus "
                "column's home; the sheet wins any disagreement with the "
                "computed mode",
        ),
        "geometry": Field(str, choices=("g1", "g2")),
        "label": Field(str),
        "seeds": Field(list),
        "inner_val_frac": Field(float),
        "max_epochs": Field(int),
        "patience": Field(int),
        "monitor": Field(str, choices=("inner_val_mse", "inner_val_pcc")),
        # [2026-08-17] TWO CELLS, ONE TASK
        # (phase10.NOTEBOOK_RECIPE_CELL_REGISTERED). Both are OPTIONAL and
        # both default to the MANUSCRIPT's values, which is what leaves
        # p10_cleftgnn.yaml untouched and its results still describing it.
        # A third recipe would need a third documented artifact by the
        # group, so the choices are closed rather than free text.
        "recipe": Field(
            str, required=False, choices=("manuscript", "notebook"),
            default="manuscript",
            doc="which of the group's artifacts the architecture is "
                "faithful to: the manuscript (SGD, additive eq 9, "
                "ResNet-50, SABM) or the notebook (Adam, f_t + f_t*v, "
                "frozen ViT-B/16, their acm_w_beta head)",
        ),
        "optimizer": Field(
            str, required=False, choices=("sgd", "adam"), default="sgd",
            doc="sgd is the manuscript's, adam the notebook's cell 9",
        ),
        "learning_rate": Field(float),
        # Adam has no momentum knob, so the notebook cell declares none;
        # the model REFUSES a config that declares both rather than
        # ignoring one of them.
        "momentum": Field(float, required=False, default=0.0),
        "batch_size": Field(int),
    },
    # Phase 10: CleftGNN under THEIR protocol -- rater-specific models, an
    # 85:15 stratified split, their Top-1 metrics, a single run with no
    # intervals (phase10.FAITHFUL_ARM_REGISTERED). The cell beside
    # cleftgnn_cv; the two differ ONLY in protocol.
    "cleftgnn_faithful": {
        "kind": Field(str, choices=("cleftgnn_faithful",)),
        "manifest_artifact": Field(str, doc="input name of data/manifests/<version>"),
        "staged_artifact": Field(str, doc="input name of data/staged/<version>"),
        "scoresheet_artifact": Field(
            str, doc="input name of the primary sheet -- the five rater columns"
        ),
        "deall_artifact": Field(
            str, doc="input name of the 25-set (their Benchmark shape)"
        ),
        "geometry": Field(str, choices=("g1", "g2")),
        "label": Field(str),
        "inner_val_frac": Field(float),
        "max_epochs": Field(int),
        "patience": Field(int),
        "learning_rate": Field(float),
        "momentum": Field(float),
        "batch_size": Field(int),
    },
    # Phase 10: the rater-specific screen -- the 0.2520 arm refit once per
    # rater (phase10.RATER_SCREEN_REGISTERED, prior committed both ways).
    "rater_screen": {
        "kind": Field(str, choices=("rater_screen",)),
        "manifest_artifact": Field(str, doc="input name of data/manifests/<version>"),
        "staged_artifact": Field(str, doc="input name of data/staged/<version>"),
        "embeddings_artifact": Field(str, doc="input name of the 0.2520 arm's set"),
        "scoresheet_artifact": Field(
            str, doc="input name of the primary sheet -- the five rater columns"
        ),
        "geometry": Field(str, choices=("g1", "g2")),
        "label": Field(str),
        "seeds": Field(list),
        "inner_val_frac": Field(float),
        "max_epochs": Field(int),
        "patience": Field(int),
        "monitor": Field(str, choices=("inner_val_mse", "inner_val_pcc")),
        "learning_rate": Field(float),
        "weight_decay": Field(float),
    },
    # [2026-08-30] Phase 10 ANNEX: the compute gate -- ONE plain-random
    # 153/28 draw, ONE rater, ONE full-trainable ResNet-50 CleftGNN fit
    # under the notebook's recipe and its CIFAR-10 statistics
    # (phase10_annex.COMPUTE_GATE_DESIGNED; rulings
    # RULING_B_FULL_TRAINABLE and NORMALISATION_RULED). Every scientific
    # knob is REQUIRED with NO default and pinned to the registered
    # value by choices -- the gate exists to measure cost, not to admit
    # variants. NO draw-count key exists here ON PURPOSE: N is declared
    # only after the measured cost lands and the ruling is it.
    "p10x_gate_fullfit": {
        "kind": Field(str, choices=("p10x_gate_fullfit",)),
        "manifest_artifact": Field(str, doc="input name of data/manifests/<version>"),
        "staged_artifact": Field(str, doc="input name of data/staged/<version>"),
        "scoresheet_artifact": Field(
            str, doc="input name of the primary sheet -- the five rater columns"
        ),
        "geometry": Field(str, choices=("g1", "g2")),
        "label": Field(
            str,
            doc="manifest column for ALIGNMENT LOADING ONLY; the training "
                "label is the rater column's own integer grades",
        ),
        #: The five sheet columns, verbatim -- asserted equal to
        #: data.scoresheet.RATERS by test so the two cannot drift. The
        #: annex reports UNMATCHED raters
        #: (phase10_annex.RATER_PANEL_DISCREPANCY_EIGHTH), so the value
        #: is a sheet identity, never a claimed CleftGNN A-E letter.
        "rater": Field(
            str,
            choices=(
                "Rater 7 - Cleft patient",
                "Rater 8 - Orthodontist",
                "Rater 9 - Speech and language therapist",
                "Rater 10 - Plastic surgeon",
                "Rater 11 - Psychologist",
            ),
        ),
        #: The draw's own seed, separate from the top-level model seed.
        #: Procedure, declared here so the config and the task cannot
        #: disagree: permutation of the cohort rows under this seed;
        #: first test_size rows are the test side, the next train_size
        #: rows the train side, the remainder UNUSED and counted.
        "split_seed": Field(int),
        "train_size": Field(int, choices=(153,)),
        "test_size": Field(int, choices=(28,)),
        #: 36 is what their code ran and what produced every published
        #: number; 27 is refused BY NAME in _check_p10x_gate with the
        #: record's reason, everything else generically here.
        "region_count": Field(int),
        #: "full" REQUIRED -- the one vocabulary in the project where it
        #: is (phase10_annex.RULING_B_FULL_TRAINABLE: a frozen backbone
        #: would leave "we froze it" as a permanent alternative
        #: explanation). Scoped: the model additionally refuses "full"
        #: under any other recipe, so the departure is structural at
        #: both layers.
        "trainable": Field(str, choices=("full",)),
        #: The notebook's recipe, whole (phase10.NOTEBOOK_RECIPE_IS_ADAM,
        #: cell 9): every field pinned to the executed value; the
        #: manuscript's SGD 0.01 is UNREGISTERED and refused by choices.
        "recipe": Field(dict, spec={
            "optimizer": Field(str, choices=("adam",)),
            "learning_rate": Field(float, choices=(0.001,)),
            "batch_size": Field(int, choices=(16,)),
            "epochs": Field(int, choices=(5,)),
            "loss": Field(str, choices=("cross_entropy",)),
            #: EPOCHS = 5, no validation split, no early stopping, no
            #: checkpoint selection -- the notebook's last epoch IS its
            #: model (phase10.NOTEBOOK_BUDGET_AND_NORMALISATION).
            "model_selection": Field(str, choices=("last_epoch",)),
        }),
        #: Pinned to the notebook's CIFAR-10 constants by
        #: _check_p10x_gate, which refuses anything else --
        #: NORMALISATION_RULED: a deliberately replicated defect,
        #: replicated knowingly, never "a recipe".
        "normalisation": Field(dict, spec={
            "mean": Field(list),
            "std": Field(list),
        }),
    },
    # [2026-08-31] Phase 10 ANNEX: the DISTRIBUTION ARM -- 500 draws x 5
    # raters through the gate's own fit path
    # (phase10_annex.DISTRIBUTION_ARM_REGISTERED, N_RULED). Everything
    # the gate pins is pinned identically here, because the arm exists
    # to run the gate's fit many times over many splits and NOTHING
    # else may vary. The draw set is ENUMERATED rather than globbed and
    # is re-derived from its declared root at load time.
    "p10x_distribution": {
        "kind": Field(str, choices=("p10x_distribution",)),
        "manifest_artifact": Field(str, doc="input name of data/manifests/<version>"),
        "staged_artifact": Field(str, doc="input name of data/staged/<version>"),
        "scoresheet_artifact": Field(
            str, doc="input name of the primary sheet -- the five rater columns"
        ),
        "geometry": Field(str, choices=("g1", "g2")),
        "label": Field(
            str,
            doc="manifest column for ALIGNMENT LOADING ONLY; the training "
                "labels are the rater columns' own integer grades",
        ),
        #: ALL FIVE, by ruling (a) -- the across-rater spread is part of
        #: what is characterised, so a subset is refused in
        #: _check_p10x_distribution rather than quietly run.
        "raters": Field(list, doc="all five sheet columns, in sheet order"),
        #: The draw set: a root, a count, and the enumerated seeds the
        #: declared rule derives from that root. The list is written out
        #: so the draws are auditable without running anything, and
        #: checked against the root so it cannot drift from it.
        "root_seed": Field(int),
        "n_draws": Field(int, choices=(500,), doc="N, ruled 2026-08-31"),
        "split_seeds": Field(list, doc="run.p10x_split_seeds(root_seed, n_draws)"),
        "train_size": Field(int, choices=(153,)),
        "test_size": Field(int, choices=(28,)),
        "region_count": Field(int),
        "trainable": Field(str, choices=("full",)),
        "recipe": Field(dict, spec={
            "optimizer": Field(str, choices=("adam",)),
            "learning_rate": Field(float, choices=(0.001,)),
            "batch_size": Field(int, choices=(16,)),
            "epochs": Field(int, choices=(5,)),
            "loss": Field(str, choices=("cross_entropy",)),
            "model_selection": Field(str, choices=("last_epoch",)),
        }),
        "normalisation": Field(dict, spec={
            "mean": Field(list),
            "std": Field(list),
        }),
        #: **The shard rule, DECLARED rather than improvised.** Draws
        #: where ``draw_index % count == index``. Shipped at
        #: ``{count: 1, index: 0}`` -- one job -- because 2,500 fits at
        #: the gate's measured cost is ~3.5 GPU-hours; splitting is a
        #: launch decision the config already supports rather than a
        #: rebuild. Sharding by DRAW keeps all five raters in every
        #: shard, so a lost shard costs draws, not a whole rater.
        "shard": Field(dict, spec={
            "count": Field(int),
            "index": Field(int),
        }),
    },
    # Phase 6 §4: the required embedding sets, derived from the arm list
    # (embedding_plan) rather than cross-producted. The shipped config's set
    # list must equal the derivation, asserted by test.
    "extract_embeddings": {
        "kind": Field(str, choices=("extract_embeddings",)),
        "manifest_artifact": Field(str, doc="input name of data/manifests/<version>"),
        "staged_artifact": Field(str, doc="input name of data/staged/<version>"),
        "out_version": Field(str, doc="data/embeddings/<version>; never overwrite"),
        "batch_size": Field(int),
        #: Re-estimate the BACKBONE's BatchNorm running statistics on cleft
        #: data before extracting. Turns each declared set into FIVE -- one per
        #: fold, each adapted on that fold's TRAINING patients only.
        #:
        #: It has to be per fold: adapting once over all 237 puts every test
        #: fold's own statistics inside the representation used to predict it,
        #: which inflates every downstream arm and is invisible in the outputs.
        #: The exclusion is asserted at build and again at consumption.
        #: See train.extract.PER_FOLD_REEXTRACTION.
        "per_fold_bn_reestimation": Field(bool, required=False, default=False),
        #: One entry per set. Checkpoint inputs are declared separately, named
        #: by extract.checkpoint_input_name; imagenet sets need none. The stub
        #: backbones exist so the whole task is laptop-testable.
        "sets": Field(list, item_spec={
            "backbone": Field(str, choices=(
                "vit_b16", "swin_b", "srgnn", "agnet", "stub", "stub_graph",
                "vit_b32", "vit_b8", "mvitv2_b",
                # Phase 25: the two self-supervised checkpoints.
                "vit_b14_dinov2", "vit_b16_dino",
            )),
            #: **Phase 25 adds two inits that name their PRETRAINING rather
            #: than borrowing "imagenet".** Both are factory-pretrained --
            #: timm's own weights for that tag, no declared checkpoint --
            #: which is exactly what "imagenet" means here. Calling DINOv2's
            #: LVD-142M weights "imagenet" would put two provenances under
            #: one name in every set directory (R2).
            "init": Field(str, choices=(
                "imagenet", "scut_original", "scut_masked",
                "dinov2_lvd142m", "dino_in1k",
            )),
            "geometry": Field(str, choices=("g1", "g2")),
            "pretrain_scheme": Field(
                str, required=False, default=None,
                choices=("native", "grid", "anatomy", "random"),
            ),
            #: Phase 7B's pooling axis. Declared together or not at all -- a
            #: depth without a pooling rule does not name a representation,
            #: and extract.extract_features refuses the half-declared case.
            #: The set name carries both, so twelve sets from one checkpoint
            #: are twelve distinguishable directories rather than twelve
            #: differing only in their contents.
            "block": Field(int, required=False, default=None),
            "token": Field(
                str, required=False, default=None,
                choices=("cls", "mean_patch"),
            ),
            #: **Phase 8 arm B's component-role CONTROL**
            #: (``extract.RANDOMISED_BACKBONE``). Every backbone parameter
            #: replaced with noise before the extraction pass, so arm B's
            #: trained graph layers can be asked what they explain when the
            #: representation is destroyed. Arm A does this live; arm B's map
            #: is precomputed, so for it the same intervention only exists
            #: here (``phase8.COMPONENT_ROLE_IS_A_NO_OP``).
            #:
            #: The set is named ``__randomised`` and marked in its metadata,
            #: and ``embeddings.check_pairing`` refuses it for any consumer
            #: that did not ask for one -- in both directions.
            "randomise": Field(bool, required=False, default=False),
        }),
        #: **A produced set asserted BIT-FOR-BIT against an already-verified
        #: input.** The pooling axis at block=depth, token=cls is the same
        #: representation the final pooled extraction returns, so it can be
        #: checked against a set that already exists and is hashed. That
        #: validates the whole intermediate path against known-good data for
        #: free, at build time, on the machine holding both.
        "reproduces": Field(list, required=False, default=(), item_spec={
            "set": Field(str, doc="a set name this run produces"),
            "input": Field(str, doc="input name of the artifact it must equal"),
        }),
    },
    # Univariate feature relevance over the mirror-difference features.
    # A DIAGNOSTIC: no fitting, no seeds, nothing tuned on it.
    "feature_relevance": {
        "kind": Field(str, choices=("feature_relevance",)),
        "manifest_artifact": Field(str, doc="input name of data/manifests/<version>"),
        "staged_artifact": Field(str, doc="input name of data/staged/<version>"),
        "geometry": Field(str, required=False, default="g2", choices=("g1", "g2")),
        "label": Field(str, required=False, default="mean"),
    },
    # Phase 6/7: the graph cleft arm -- frozen feature maps, trained graph
    # layers, the THIRD REGIME. The 10-seed SR-GNN band (the gate before any
    # graph delta) runs through this kind.
    "train_graph_cv": {
        "kind": Field(str, choices=("train_graph_cv",)),
        "manifest_artifact": Field(str, doc="input name of data/manifests/<version>"),
        "staged_artifact": Field(
            str, doc="input name of data/staged/<version>; content boxes for "
            "generated schemes, staged images for AG-Net's native SIFT+GMM"
        ),
        #: [CORRECTED 2026-08-14] Not always a feature_map set: with
        #: `consume: nodes` this is a region_vectors set whose regions were
        #: cropped BEFORE the backbone, so there is no map. The doc said
        #: feature_map only, which is the mismatch that would have surfaced
        #: on the cluster after submission.
        "embeddings_artifact": Field(
            str, required=False,
            doc="input name of ONE set directory: a feature_map set, or a "
                "region_vectors set when consume is 'nodes'",
        ),
        #: PER-FOLD artifact: input names keyed by fold, e.g.
        #: {0: emb_fold0, ...}. Each set is BN-re-estimated on its own fold's
        #: training patients, so it is valid for exactly that fold and the arm
        #: asserts the fold's test patients did not contribute. Mutually
        #: exclusive with embeddings_artifact.
        #: See train.extract.PER_FOLD_REEXTRACTION.
        "embeddings_artifacts_by_fold": Field(dict, required=False, default=None),
        #: **How this arm reads its embeddings artifact**
        #: (``embeddings.as_consumed``). Absent means feature maps, which
        #: is every Road A graph arm. "nodes" is Road B Branch 3: the
        #: regions are already vectors, cropped BEFORE the backbone, so
        #: there is no map to pool them out of.
        "consume": Field(str, required=False, default=None,
                         choices=("nodes",)),
        #: Input name of the pretraining checkpoint the graph layers warm-start
        #: from. Required for scut inits (the pairing check compares its
        #: verified hash against the artifact's recorded one); absent for
        #: imagenet, whose graph layers are seed-initialised.
        "checkpoint": Field(str, required=False),
        "backbone": Field(str, choices=("srgnn", "agnet", "stub_graph")),
        "init": Field(str, choices=("imagenet", "scut_original", "scut_masked")),
        "geometry": Field(str, choices=("g1", "g2")),
        #: The scheme the graph layers train under. Must equal the artifact's
        #: pretrain_scheme for pretrained inits (the consistency rule,
        #: enforced at runtime); free for imagenet. No "none": graph only.
        "region_scheme": Field(
            str, choices=("native", "grid", "anatomy", "random")
        ),
        "label": Field(str, required=False, default="mean"),
        #: What the arm optimises. "graph_layers" IS the regime and is the
        #: default, so the brief's §7 trap ("do not freeze the graph layers --
        #: that reduces the arm to a linear probe and tests nothing") is never
        #: reached by omission.
        #:
        #: "classifier" runs that reduction DELIBERATELY, as a diagnostic and
        #: never as a ladder arm: it freezes the graph layers too and fits the
        #: 2,049-parameter classifier alone, which is the only SR-GNN arm
        #: directly comparable to the frozen ViT probe's 0.2529. It also runs
        #: the model in eval mode, without which dropout and BatchNorm would
        #: leave the "frozen" representation moving under the head. The run
        #: stamps `is_a_ladder_arm: false` into metrics.json.
        #: See train.graph_cleft.FROZEN_GRAPH_DIAGNOSTIC.
        #:
        #: "classifier_adabn" is that same probe with BatchNorm running
        #: statistics RE-ESTIMATED on each training fold before the classifier
        #: is fitted -- the last alternative explanation for the plain probe's
        #: 0.1340, and the remedy PLAN §4.7 records for these backbones. Also
        #: a diagnostic, never a ladder arm.
        #: See train.graph_cleft.ADABN_DIAGNOSTIC.
        "trainable": Field(
            str,
            required=False,
            default="graph_layers",
            choices=("graph_layers", "classifier", "classifier_adabn"),
        ),
        "seeds": Field(list, required=False, default=()),
        #: Whether ``use_deterministic_algorithms`` stays on.
        #:
        #: **AG-Net arms MUST set it; the run refuses None there.** Phase 6
        #: measured that the deterministic ``roi_align`` substitution cannot
        #: be satisfied in the pinned image (it needs torch.compile ->
        #: inductor -> a C++ compiler the image lacks), so `true` raises
        #: mid-run and `false` bounds this arm's same-seed reproducibility at
        #: ~2.3e-05 rather than bitwise -- which every comparison against it
        #: has to carry. Neither is a defensible default, so there is none.
        #:
        #: Every other graph backbone is bitwise under the flag and omits it.
        #: See models.agnet.TRAINING_DETERMINISM.
        "deterministic": Field(bool, required=False, default=None),
        "max_epochs": Field(int),
        "patience": Field(int, doc="identical across every arm, not a per-arm knob"),
        "inner_val_frac": Field(float),
        "monitor": Field(str, choices=("inner_val_mse", "inner_val_pcc")),
        "learning_rate": Field(float, required=False, default=1e-4),
        "weight_decay": Field(float, required=False, default=0.01),
        "batch_size": Field(int, required=False, default=16),
    },
    # Phase 3: the minimal training path and gates 3-6.
    "train_cv": {
        "kind": Field(str, choices=("train_cv",)),
        "manifest_artifact": Field(str),
        "staged_artifact": Field(str),
        #: G1 is the neutral default for gate purposes; see phase3.DEFAULT_GEOMETRY.
        "geometry": Field(str, choices=("g1", "g2")),
        #: A manifest column, or "ldl". 'mean' is PRIMARY per §4.2.
        #:
        #: "ldl" is the one value that is NOT a column: it selects Label
        #: Distribution Learning over soft_1..soft_5 with a 5-output softmax
        #: head and a KL loss. It is still SCORED on `mean`, so its truth
        #: vector is the mean arm's -- which is what lets Stage G compare
        #: them. See train.ldl.
        "label": Field(str, doc="manifest column, or 'ldl'; 'mean' is PRIMARY"),
        #: "ridge" is Phase 4's mirror-difference arm: no backbone at all, features
        #: computed from the geometry and a closed-form regressor over them.
        #: "swin_b" arrives with Phase 7 and is ARTIFACT-ONLY: live
        #: extraction builds ViT-B/16 and nothing else, so a Swin arm
        #: without an embeddings_artifact would train on ViT features
        #: while its metrics.json said swin_b. Refused at load time by
        #: _check_train_cv_init.
        #: "srgnn" arrives with Road B Branch 3 and is ARTIFACT-ONLY for
        #: the same reason swin_b is: it reads a region_vectors set and
        #: concatenates it, so there is no live-extraction path to fall
        #: back to. Refused without an artifact by _check_train_cv_init.
        #: Phase 7D adds "vit_b32", "vit_b8" and "mvitv2_b" -- ARTIFACT-ONLY
        #: like swin_b -- and "concat", arm 4's linear head over several
        #: pooled sets concatenated. A concat arm declares
        #: concat_embeddings_artifacts instead of embeddings_artifact, and
        #: each constituent is pairing-checked against its OWN backbone.
        "backbone": Field(
            str, choices=(
                "vit_b16", "swin_b", "srgnn", "stub", "ridge",
                "vit_b32", "vit_b8", "mvitv2_b", "concat",
                # Phase 25.
                "vit_b14_dinov2", "vit_b16_dino",
            )
        ),
        #: Arm 4's constituents, IN THE ORDER CONCATENATED. Only valid with
        #: backbone "concat"; each entry names a declared input and the
        #: backbone whose set it must be (checked against the artifact's own
        #: metadata, never trusted).
        "concat_embeddings_artifacts": Field(
            list, required=False, default=(), item_spec={
                "artifact": Field(str, doc="declared input name of one pooled set"),
                "backbone": Field(
                    str, choices=("vit_b16", "swin_b", "vit_b32", "vit_b8",
                                  "mvitv2_b"),
                ),
            },
        ),
        #: **How this arm reads its embeddings artifact**
        #: (``embeddings.as_consumed``). Absent means "pooled", which is
        #: every Road A arm. Declared rather than inferred because one
        #: region_vectors set serves both "concat" and "nodes", and the
        #: wrong pairing trains happily and reports a number.
        "consume": Field(str, required=False, default=None,
                         choices=("pooled", "concat", "nodes")),
        #: "head" freezes the backbone. "full" trained all ~86M parameters on 237
        #: images and produced worse-than-constant predictions -- see
        #: models.factory.TRAINABLE_POLICIES.
        "trainable": Field(str, required=False, default="head", choices=("head", "full")),
        #: **Phase 7C: augment the training pixels and re-extract per epoch.**
        #:
        #: Flags only. The STRENGTHS are not config fields -- they live in
        #: ``cleft.phase7c``, pre-registered and checked by the suite, on the
        #: same principle as Phase 7B: a pre-registration a config could edit
        #: is not one.
        #:
        #: Declaring this changes what ``prepare_features`` hands the harness
        #: -- raw pixels plus a per-patient strength mask instead of stored
        #: embeddings -- and routes ``make_factory`` to the augmenting
        #: backbone. ``trainable`` stays "head": the backbone is still frozen
        #: and the head is still 769 parameters, so the policy check is
        #: unchanged. What differs is only that the features are recomputed.
        #: Validated by ``_check_augmentation`` -- the keys are a closed set
        #: and rotation without geometric is refused at LOAD time, so a policy
        #: the augmenter cannot build costs nothing on the cluster.
        "augmentation": Field(dict, required=False, default=None),
        #: Phase 4 §3.3/§3.4. "whole" is the Phase 3 bar and takes no patches.
        #: The three schemes must be compared at G2, where all of them see zero
        #: background -- at G1 a scheme difference is partly a background
        #: difference (PLAN §4.4).
        "patch_scheme": Field(
            str,
            required=False,
            default="whole",
            choices=("whole", "grid", "anatomy", "random"),
        ),
        #: Which geometric index a `backbone: ridge` arm computes. §3.1's
        #: mirror-difference index, or §3.2's SymNose reimplementation over the
        #: segmented upper lip. Ignored by every other backbone.
        "feature_source": Field(
            str, required=False, default="mirror", choices=("mirror", "symnose")
        ),
        #: Phase 7: input name of ONE pooled embedding set directory. Present
        #: means the arm consumes precomputed vectors instead of running the
        #: backbone over the images -- which is the only way to reach a
        #: pretrained init, since live extraction builds ViT-B/16 at ImageNet
        #: and nothing else. See train.pooled.
        "embeddings_artifact": Field(str, required=False),
        #: Which pretraining the representation came from. Only meaningful
        #: with `embeddings_artifact`: without one the arm IS at imagenet,
        #: because that is the only init live extraction can produce, and the
        #: loader refuses a config that says otherwise rather than running an
        #: arm whose metrics.json would claim a pretraining that never
        #: happened.
        "init": Field(
            str,
            required=False,
            default="imagenet",
            choices=(
                "imagenet", "scut_original", "scut_masked",
                # Phase 25: factory-pretrained, but NOT imagenet weights.
                "dinov2_lvd142m", "dino_in1k",
            ),
        ),
        #: Input name of the pretraining checkpoint the artifact must have come
        #: from. Required for pretrained inits: its VERIFIED hash is compared
        #: against the artifact's recorded `checkpoint_sha256`, so features
        #: from one representation cannot be trained under another's name.
        "checkpoint": Field(str, required=False),
        #: How per-patch embeddings become one vector per patient. "concat" is
        #: excluded by design: 27x768 features for 237 patients is memorisation,
        #: not a probe. See geometry.patch_features.POOLINGS.
        "pooling": Field(str, required=False, default="mean", choices=("mean", "max")),
        #: Gate 2. Omit for a single-seed run; give a list to sweep in one job.
        "seeds": Field(list, required=False, default=()),
        "max_epochs": Field(int),
        "patience": Field(int, doc="identical across every arm, not a per-arm knob"),
        "inner_val_frac": Field(float),
        "monitor": Field(str, choices=("inner_val_mse", "inner_val_pcc")),
        #: 1e-3, matching every code path that documents a default for the
        #: frozen head: run.py's fallback, `phase3.make_factory`, and the
        #: `train_config` record in metrics.json.
        #:
        #: **It said 1e-4, and the schema was the one that took effect.**
        #: `_validate_mapping` fills every absent optional with its schema
        #: default, so `task["learning_rate"]` is always present afterwards and
        #: the `task.get(..., 1e-3)` fallbacks are unreachable. A config omitting
        #: the field would have trained at the full-fine-tuning rate while
        #: metrics.json faithfully recorded 0.0001 -- correct provenance for a
        #: value nobody chose. Every shipped config sets it explicitly, so the
        #: trap was never sprung.
        "learning_rate": Field(float, required=False, default=1e-3),
        "weight_decay": Field(float, required=False, default=0.01),
        "batch_size": Field(int, required=False, default=16),
        #: Ridge penalty, used only by the "ridge" backbone. Fixed and not
        #: selected against anything -- see train.ridge.DEFAULT_ALPHA.
        "alpha": Field(float, required=False, default=1.0),
        #: **[2026-08-31] Phase 20's label-permutation control.**
        #:
        #: Absent, the arm is unchanged in every respect -- this is the
        #: probe's own path and the control's value depends on its
        #: staying that way (``phase20.ARM_P_REGISTERED``).
        #:
        #: Present, it declares TWO things and no others:
        #:
        #: * ``kind``: "plain" (Arm P, the integrity gate) or
        #:   "stratified" (Arm S, the attribution).
        #: * ``k_strata``: 1 for plain, and for stratified **the locked
        #:   10** (``phase20.STRATIFICATION_RULED``). A locked scientific
        #:   setting is declared here so that MOVING it is a config
        #:   change with a diff, not a runtime choice -- and
        #:   ``_check_permutation`` refuses any other value, so the lock
        #:   is enforced rather than merely written down.
        #:
        #: The strata VARIABLE is deliberately not a field: it is the
        #: locked rule (the in-sample fitted value over all five confound
        #: statistics), and a pre-registration a config could edit is not
        #: one -- the same principle as ``phase7c``'s strengths.
        "permutation": Field(
            dict, required=False, default=None,
            doc="Phase 20: {kind: plain|stratified, k_strata: 1|10}",
        ),
    },
    # Phase 2 exit criteria 7, 8 and 10: stage every patient, generate patches at
    # both geometries, write the versioned hashed artifact Phase 3 consumes.
    "stage_and_patch": {
        "kind": Field(str, choices=("stage_and_patch",)),
        "manifest_artifact": Field(str),
        "patient_folders": Field(str),
        "out_version": Field(str, doc="artifact version; never overwrite one"),
        "generators": Field(list),
        "geometries": Field(list),
    },
    # One job, a grid of candidate anatomy placements, same faces throughout.
    "anatomy_sweep": {
        "kind": Field(str, choices=("anatomy_sweep",)),
        "manifest_artifact": Field(str),
        "patient_folders": Field(str),
        "n_patients": Field(int, doc="faces per variant; keep small, 3-4"),
        "v_offsets": Field(list, doc="shift down, normalised units"),
        "scales": Field(list, doc="region size multipliers"),
        "scales_x": Field(list, required=False, default=(1.0,), doc="lateral spread"),
        "box_scales_x": Field(
            list, required=False, default=(1.0,), doc="box width, not spread"
        ),
        "geometry": Field(str, required=False, default="g2", choices=("g1", "g2")),
    },
    "build_manifest": {
        "kind": Field(str, choices=("build_manifest",)),
        "primary_scoresheet": Field(str, doc="input name of the authoritative workbook"),
        "equivalence_scoresheet": Field(
            str, required=False, doc="second workbook, checked for agreement only"
        ),
        "soft_label_crosscheck": Field(
            str, required=False, doc="CSV whose soft_1..soft_5 must match ours"
        ),
        "patient_folders": Field(str, doc="input name of the patient folder tree"),
        "orthodontist_rater": Field(
            str, doc="named not indexed, so a column reorder cannot change it"
        ),
        "out_version": Field(str, doc="artifact version; never overwrite an existing one"),
        "n_folds": Field(int),
        "expect_patients": Field(int),
        "expect_frontal": Field(int),
        "expect_basal": Field(int),
        "expect_photoless": Field(int),
        "expect_scored_rows": Field(int),
    },
}

#: **[2026-08-31] Phase 20's arms P and S.**
#:
#: The spec is ``train_cv``'s, key for key, with two differences: the
#: kind, and ``permutation`` promoted from optional to REQUIRED. Built by
#: copying rather than by hand because the arms' whole claim is that they
#: are the probe's fit path with one vector reordered -- a hand-written
#: near-copy of this spec would let the two drift a field at a time, and
#: a control that has drifted from its arm measures nothing.
#:
#: The separate KIND (rather than a bare ``train_cv`` carrying a
#: permutation block) is what makes ``permutation`` required and what
#: puts the arm's identity in the run directory's name, so a permuted
#: run can never be mistaken for a probe run in a listing.
TASK_SPECS["p20_permutation"] = {
    **TASK_SPECS["train_cv"],
    "kind": Field(str, choices=("p20_permutation",)),
    "permutation": Field(
        dict, doc="REQUIRED here: {kind: plain|stratified, k_strata: 1|10}"
    ),
    #: **NARROWED to "head", deliberately.** The copy above would have
    #: inherited ``train_cv``'s ("head", "full"), and a fourth kind
    #: admitting full fine-tuning is exactly the silent widening
    #: ``phase10_annex.RULING_B_FULL_TRAINABLE`` called "a scoped
    #: departure, not a general unlock".
    #:
    #: The narrowing is also what the arms ARE: a control is only a
    #: control if it is the same computation as the thing it controls,
    #: and the probe is the 769-parameter frozen head. A permuted arm at
    #: ``full`` would be a control for an arm this project has never run.
    "trainable": Field(
        str, required=False, default="head", choices=("head",),
        doc="head only -- the control must be the probe's own recipe",
    ),
}

#: **[2026-08-31] Phase 20's stratification diagnostic.**
#:
#: Half Arm S (the permutation block, the label, the seeds) and half Arm
#: C (the ridge recipe), because that is exactly what it measures: Arm
#: C's model on Arm S's labels. It declares NO embeddings artifact -- it
#: is five scalars and a closed-form ridge, and it never opens them.
TASK_SPECS["p20_stratification_diagnostic"] = {
    "kind": Field(str, choices=("p20_stratification_diagnostic",)),
    "manifest_artifact": Field(str),
    "staged_artifact": Field(str),
    #: The panel mean. Carried from Arm S, whose labels these are.
    "label": Field(str),
    #: **Carried VERBATIM from Arm S's config.** If the two blocks ever
    #: differ, this run scores a permutation Arm S never trained on and
    #: its number says nothing about Arm S. A test asserts the equality;
    #: `_check_permutation` enforces the locked k on both.
    "permutation": Field(dict),
    "seeds": Field(list),
    "inner_val_frac": Field(float),
    "alpha": Field(float),
    #: **Arm C's OBSERVED figure** (phase20.ARMS_OBSERVED), carried. It
    #: is the denominator of every fraction this task reports, so it is
    #: declared rather than looked up: a run must record which ceiling
    #: its percentages were taken against.
    "ceiling_pcc": Field(
        float, doc="the intact confound signal; Arm C's measured mean"
    ),
    #: The two reading thresholds, as fractions of `ceiling_pcc`.
    #: Declared here so the cell that fires cannot be chosen after the
    #: number is seen -- the `substantial_pcc` pattern.
    "destroyed_fraction": Field(
        float, doc="at or below this fraction: the near_zero cell"
    ),
    "survived_fraction": Field(
        float, doc="at or above this fraction: the near_the_ceiling cell"
    ),
    "expect_patients": Field(int),
}

#: **[2026-09-01] Phase 21's two arms.** Both consume CACHED OOF
#: PREDICTIONS and nothing else: no backbone, no embeddings artifact, no
#: pixels. The `arms` list is LITERAL -- the generator derives it from
#: phase18.ARM_LIST_LOCKED filtered by phase21.ARM_SET_RULED, so no glob
#: exists anywhere in the arm resolution and the config cannot drift
#: from the lock.
_P21_ARM_SPEC = {
    "name": Field(str),
    "input": Field(str, doc="declared input naming this arm's run directory"),
    "seeds": Field(list, doc="the shared five; every arm, without exception"),
    "csv": Field(str, doc="predictions | identity_predictions"),
    "group": Field(str),
    "n_patients": Field(int, doc="237 for every ruled arm"),
}

TASK_SPECS["p21_ensemble_probe"] = {
    "kind": Field(str, choices=("p21_ensemble_probe",)),
    "manifest_artifact": Field(str),
    "arms": Field(list, item_spec=_P21_ARM_SPEC),
    "seeds": Field(list, doc="the declared shared basis (SEED_BASIS_RULED)"),
    #: The arm whose predictions the ensemble is contrasted against. It
    #: must itself be in the ruled set, so the contrast is within-phase
    #: and paired over the same patients.
    "probe_arm": Field(str),
    #: Both anchors, declared. `probe_pcc` is the record's 0.2520;
    #: `concat_precedent_delta` is Phase 7B's withdrawn +0.0073, quoted
    #: beside any gain so it cannot read as though combination had never
    #: been tried (phase21.EXIT_CRITERIA criterion 2).
    "probe_pcc": Field(float),
    "concat_precedent_delta": Field(float),
    "n_boot": Field(int, required=False, default=10000),
    "expect_patients": Field(int),
}

TASK_SPECS["p21_error_consistency"] = {
    "kind": Field(str, choices=("p21_error_consistency",)),
    "manifest_artifact": Field(str),
    #: Required: the derived rater sd is cross-checked against
    #: phase10_rater_grades' direct read, because the label join has
    #: produced two defects before.
    "scoresheet_artifact": Field(str),
    "scoresheet_file": Field(str, required=False, default=None),
    "arms": Field(list, item_spec=_P21_ARM_SPEC),
    "seeds": Field(list),
    "binarisations": Field(
        list, doc="residual_sign and worst_quartile, reported SEPARATELY"
    ),
    #: The ruled thresholds (phase21.CONSISTENCY_THRESHOLDS_RULED).
    #: Declared here so the cell that fires cannot be chosen after the
    #: number is seen -- and refused at load if they do not bracket.
    "low_kappa": Field(float),
    "high_kappa": Field(float),
    "n_permutations": Field(int),
    "alpha": Field(float, required=False, default=0.05),
    "n_raters": Field(int, required=False, default=5),
    "expect_patients": Field(int),
}

#: **[2026-09-01] Two RECKONING INPUTS for Phase 22 -- not Phase 22
#: machinery.** Both run on cached data: the manifest, and (for
#: shrinkage) the same per-seed prediction CSVs Phase 18 and Phase 21
#: already declare. No new artifact, no new hash, no GPU, no gate.
TASK_SPECS["cohort_pair_separation"] = {
    "kind": Field(str, choices=("cohort_pair_separation",)),
    "manifest_artifact": Field(str),
    "label": Field(str, doc="the panel mean; the column the pairs order"),
    "multiples": Field(
        list, doc="reported at 1x, 2x, 3x of BOTH scales -- a profile, "
                  "because a single threshold would be a choice",
    ),
    #: The scale and multiple the READING turns on, declared so the cell
    #: cannot be chosen after the six fractions are seen. `se_diff` is
    #: the proper scale: a pair's ordering turns on the DIFFERENCE of
    #: two means, not on one mean.
    "primary_scale": Field(str, choices=("se_single", "se_diff")),
    "primary_multiple": Field(int, choices=(1, 2, 3)),
    "expect_patients": Field(int),
}

TASK_SPECS["cross_arm_shrinkage"] = {
    "kind": Field(str, choices=("cross_arm_shrinkage",)),
    "manifest_artifact": Field(str),
    "arms": Field(list, item_spec=_P21_ARM_SPEC),
    "seeds": Field(list),
    "probe_arm": Field(str, doc="reported separately; must be in the set"),
    #: The three reading thresholds, declared. Precedence is
    #: fails-to-shrink, then varies-widely, then
    #: shrinks-substantially, so two cells cannot both fire -- a single
    #: non-shrinking arm is the more consequential fact.
    #:
    #: `fails_to_shrink_at` and `shrinks_substantially_at` test the MAX
    #: of the per-arm mean shrinkage ratios.
    #:
    #: **`varies_widely_at` tests the SAMPLE SD (ddof=1) of those per-arm
    #: means -- RULED 2026-09-01, replacing a range statistic that was
    #: built, simulated and rejected on evidence.** The range's estimand
    #: grows with arm count (mean range 0.2282 -> 0.3750 from n=8 to
    #: n=64 at fixed sigma 0.08), so at 64 arms it fired 94.65% of the
    #: time on genuinely homogeneous arms where the SD fires 0.33%. See
    #: phase21.SHRINKAGE_DISPERSION_STATISTIC_RULED.
    "fails_to_shrink_at": Field(float),
    "shrinks_substantially_at": Field(float),
    "varies_widely_at": Field(
        float, doc="SD (ddof=1) of the per-arm means; NOT their range",
    ),
    "expect_arms": Field(int),
    "expect_patients": Field(int),
}

#: **[2026-09-01] Phase 22's tie measurement -- a PRECONDITION of the
#: target-map ruling, not a Phase 22 arm.** It gates
#: ``phase22.TARGET_MAP_RULED``: hard targets assign every exact tie a
#: definite WRONG order, and the record held the below-SE_diff fraction
#: without ever separating out the ties inside it.
#:
#: Cached manifest, aggregates only, no GPU, no new artifact, no gate.
#: The tie test is **exact integer equality on the recovered rater
#: sums** (``phase22.TIE_COMPARISON_RULE_DECLARED``) -- no float
#: comparison and no tolerance, and ``phase8c.rater_multiset`` confirms
#: the rater count from the data by refusing anything that is not five.
TASK_SPECS["tie_fraction"] = {
    "kind": Field(str, choices=("tie_fraction",)),
    "manifest_artifact": Field(str),
    "label": Field(str, doc="the panel mean; cross-check only -- the tie "
                            "test runs on the integer rater sums"),
    #: The registered boundary (phase22.TIE_FRACTION_MEASUREMENT_DESIGNED).
    #: Declared here so the cell that fires cannot be chosen after the
    #: number is seen. Its reference quantity is derived (the
    #: below-SE_diff set the R-all/R-clear contrast is defined on); the
    #: half taken of it is a stated judgement, not a derivation.
    "boundary": Field(float),
    "report_smallest": Field(
        int, doc="how many distinct separations to report from the "
                 "bottom -- the tie count means little without them",
    ),
    #: Cross-checks against the separation run. If either disagrees the
    #: task STOPS: two measurements of the same cohort that differ on
    #: its pair count are not both right, and proceeding would bank a
    #: tie fraction whose denominator nobody trusts.
    "expect_pairs": Field(int),
    "expect_below_se_diff": Field(
        float, required=False, default=None,
        doc="the separation run's se_diff_x1 fraction, to 4 places; "
            "null disables the check for fixtures",
    ),
    "expect_patients": Field(int),
}

#: **[2026-09-01] Phase 22's six ranking arms.** ONE NEW KIND
#: (``phase22.AXIS_RULED``): ``train_cv``'s closed vocabulary and the
#: shipped ``siamese_contrastive`` are both untouched, and no existing
#: arm's configuration surface is reopened.
#:
#: The loss, the tie rule and the pair sampling are RULED and are not
#: config fields -- they live in ``train/ranking.py`` and cannot vary
#: between arms. What a config declares is the ARM: which pair source,
#: and which head.
TASK_SPECS["train_rank_cv"] = {
    "kind": Field(str, choices=("train_rank_cv",)),
    "arm": Field(str, choices=(
        "r_syn_bounded", "r_syn_unbounded",
        "r_all_bounded", "r_all_unbounded",
        "r_clear_bounded", "r_clear_unbounded",
    )),
    #: Setting 6 (phase22.DECLARED_SETTINGS_22): which pairs the arm
    #: trains on. `cohort_se_diff` is R-clear's restriction, DECLARED IN
    #: ADVANCE and never a post-hoc filter.
    "pair_source": Field(str, choices=(
        "synth_within_face", "cohort_all", "cohort_se_diff",
    )),
    #: Setting 3: bounded scores on the label scale, or order only.
    #: BOTH are built -- they are separate registered arms, and the
    #: contrast between them is three of the fifteen.
    "bounded": Field(bool),
    #: Setting 4, DERIVED from `bounded` and written explicitly anyway:
    #: a derived value that is never written is one nobody can check.
    #: Unbounded arms carry a DECLARED LIMITATION
    #: (phase22.MONITOR_BIND_RESOLVED).
    "monitor": Field(str, choices=("inner_val_mse", "inner_val_pcc")),
    #: Setting 7, R-clear only: recomputed at build from sd_obs and
    #: RELIABILITY_237, never pinned as a literal.
    "se_diff_threshold": Field(float, required=False, default=None),
    #: Settings 9 and 10, R-syn only and STILL OPEN. -1 is the UNRULED
    #: sentinel: the task refuses it by name rather than letting a
    #: plausible default stand in for a ruling nobody made.
    "pretrain_epochs": Field(int, required=False, default=None),
    "finetune_trainable_code": Field(int, required=False, default=None),
    "manifest_artifact": Field(str),
    "staged_artifact": Field(str),
    "embeddings_artifact": Field(str, required=False, default=None),
    "backbone": Field(str, choices=("vit_b16", "stub")),
    "geometry": Field(str),
    "label": Field(str),
    "seeds": Field(list),
    "init": Field(str, required=False, default="imagenet"),
    "learning_rate": Field(float, required=False, default=1e-3),
    "weight_decay": Field(float, required=False, default=0.01),
    "max_epochs": Field(int),
    "patience": Field(int),
    "inner_val_frac": Field(float),
    "expect_patients": Field(int),
}

#: **[2026-09-01] Phase 22's two post-run steps.** Both consume cached
#: OOF prediction CSVs from the arm run directories. No GPU, no new
#: artifact, no gate, no ledger row.
TASK_SPECS["p22_diagnostic"] = {
    "kind": Field(str, choices=("p22_diagnostic",)),
    "manifest_artifact": Field(str),
    "arms": Field(list, item_spec=_P21_ARM_SPEC),
    "seeds": Field(list),
    #: The arms being DIAGNOSED. Everything else in `arms` is the
    #: comparison set, which must be the 63 shrinking arms -- arm A is
    #: excluded because it is the reference the LOW threshold comes
    #: from (phase22.FEATURE_DIFFERENCE_DIAGNOSTIC).
    "subject_arms": Field(list),
    "expect_comparison_arms": Field(int),
    "binarisations": Field(
        list, doc="residual_sign and worst_quartile, reported SEPARATELY"
    ),
    #: **Checked, never used as the threshold.** The task READS both
    #: thresholds from their phase21 sources at run time; these declare
    #: what it expects to find, so a changed source figure STOPS the
    #: phase (EXIT_CRITERIA criterion 6) instead of being read past.
    "expect_high_kappa": Field(dict),
    "expect_low_kappa": Field(dict),
    "expect_patients": Field(int),
}

#: **Phase 25's three contrasts** (phase25.THE_FAMILY_OF_THREE).
#: The pair list is NOT a field: it comes from
#: ``phase25.contrast_family()``, so the family cannot grow through a
#: config edit. ``winner_sd`` is NOT a field either -- it is computed
#: from each winner's OWN per-seed PCCs, because a declared sd is a
#: number that can disagree with the artifact it describes.
#: **[2026-09-02] The panel's ICCs, for record_audit.ICC_PREDICTION_
#: REGISTERED.** Needs the SCORE SHEET, because ICC(2,k) decomposes
#: BETWEEN-RATER variance and therefore needs rater IDENTITY -- which the
#: manifest's soft_1..soft_5 do not carry (they give the multiset, not
#: who gave what). DESCRIPTIVE; no ledger row.
TASK_SPECS["rater_icc"] = {
    "kind": Field(str, choices=("rater_icc",)),
    "manifest_artifact": Field(str),
    "scoresheet_artifact": Field(str),
    "expect_patients": Field(int),
    "expect_raters": Field(int),
}

TASK_SPECS["p25_contrasts"] = {
    "kind": Field(str, choices=("p25_contrasts",)),
    "manifest_artifact": Field(str),
    #: **[ADDED 2026-09-02] ``csv`` was missing and the loader requires
    #: it.** ``_p21_load_arm_predictions`` opens
    #: ``seed_<n>__<csv>.csv``, and Phase 21 and 22 declare the stem per
    #: arm because **it genuinely varies**: 461 shipped arm entries say
    #: ``predictions`` and 7 say ``identity_predictions`` (Phase 16's
    #: identity baseline). A field that varies belongs in the config, so
    #: Phase 25 declares it the same way rather than hardcoding a stem
    #: that would make this caller differ from every other one.
    "arms": Field(list, item_spec={
        "name": Field(str),
        "input": Field(str, doc="input name of the run DIRECTORY"),
        "seeds": Field(list),
        "n_patients": Field(int),
        "csv": Field(
            str,
            doc="prediction-file stem: seed_<n>__<csv>.csv "
                "(phase3.write_outputs writes <prefix>predictions.csv)",
        ),
    }),
    "seeds": Field(list),
    "n_boot": Field(int),
    "expect_contrasts": Field(int),
    "expect_patients": Field(int),
}


#: **[ADDED 2026-09-06] Phase 26's six-metric union over the sixteen
#: calibration cells.**
#:
#: The CELLS are ordinary ``train_cv`` runs. Only the consolidation is a
#: new kind, because the training path emits four of the six readouts and
#: never emits IEM. The arm entries carry each cell's two factor levels
#: so the table can be read as a grid without re-deriving them from a
#: config stem.
TASK_SPECS["p26_calibration_table"] = {
    "kind": Field(str, choices=("p26_calibration_table",)),
    "manifest_artifact": Field(str),
    "arms": Field(list, item_spec={
        "name": Field(str),
        "input": Field(str, doc="input name of the run DIRECTORY"),
        "seeds": Field(list),
        "n_patients": Field(int),
        "csv": Field(str, doc="prediction-file stem: seed_<n>__<csv>.csv"),
        "weight_decay": Field(float, doc="this cell's level of factor one"),
        "patience": Field(int, doc="this cell's level of factor two"),
    }),
    "seeds": Field(list),
    "expect_cells": Field(int),
    "expect_patients": Field(int),
    #: The anchor cell IS the probe, so it must reproduce the probe. A
    #: declared expectation rather than a discovered one, because a check
    #: that reads its own target off the data checks nothing.
    "anchor_cell": Field(str),
    "anchor_pcc": Field(float),
    "anchor_sd": Field(float),
    #: Optional. The Phase 18 constant-predictor IEM, recovered from that
    #: run's metrics.json, for the cross-check that criterion 2 requires.
    #: Absent means the cross-check is REPORTED AS UNAVAILABLE rather
    #: than silently skipped.
    "phase18_floor_iem": Field(float, required=False),
}

TASK_SPECS["p22_contrasts"] = {
    "kind": Field(str, choices=("p22_contrasts",)),
    "manifest_artifact": Field(str),
    "arms": Field(list, item_spec=_P21_ARM_SPEC),
    "seeds": Field(list),
    #: How many of the locked fifteen both members are declared for.
    #: Declared so a missing arm is a REFUSAL rather than a quietly
    #: shorter family.
    "expect_evaluable": Field(int),
    "winner_sd": Field(float),
    "n_boot": Field(int),
    "expect_patients": Field(int),
}

#: **[2026-09-02] Phase 23's ROPE analysis.** Reads banked prediction
#: CSVs and produces posteriors. No GPU, no new artifact, no gate, no
#: ledger row.
#:
#: **The width is NOT a value here.** ``expect_rope_half_width`` is a
#: CHECK against the value the task recomputes from
#: ``combined_claimable_delta``; a mismatch refuses the run
#: (``phase23.EXIT_CRITERIA`` criterion 2).
TASK_SPECS["p23_rope"] = {
    "kind": Field(str, choices=("p23_rope",)),
    #: **No manifest_artifact.** [2026-09-02: the first version
    #: declared one and never read it -- the truth column is in the
    #: prediction CSVs, so the manifest is not needed. Caught by this
    #: phase's own config-to-code sweep, in its own new code, before
    #: the launch.]
    #: Each: {key, row, a, b}. The list is fixed at the lock; criterion
    #: 8 refuses a count that does not match `expect_contrasts`.
    "contrasts": Field(list),
    "seeds": Field(list),
    #: The inputs the width is RECOMPUTED from -- the probe's own seed
    #: sd and seed count, not the width itself.
    "probe_seed_sd": Field(float),
    "probe_n_seeds": Field(int),
    "expect_rope_half_width": Field(float),
    #: The source's, with its approximation recorded in phase23.
    "rho": Field(float),
    #: Ours, from the ruled 20:1 loss ratio.
    "decision_threshold": Field(float),
    #: 25 where both arms ran five seeds on 237. A contrast below it is
    #: reported with its own degrees of freedom, not disqualified.
    "full_n": Field(int),
    "expect_contrasts": Field(int),
    "expect_patients": Field(int),
}

SCHEMA: dict[str, Field] = {
    "schema_version": Field(int, choices=SUPPORTED_SCHEMA_VERSIONS),
    "phase": Field(str, doc="p0..p10, becomes a run directory level"),
    "tier": Field(str, choices=("keeper", "dev")),
    "seed": Field(int),
    "inputs": Field(list, required=False, item_spec=INPUT_SPEC, default=()),
    "task": Field(dict),
}


def _type_name(type_: type) -> str:
    return {int: "an integer", float: "a number", str: "a string", bool: "a boolean",
            list: "a list", dict: "a mapping"}.get(type_, type_.__name__)


def _check_type(value: Any, spec: Field, where: str) -> Any:
    # ``isinstance(True, int)`` is True in python, so ``seed: true`` would pass a
    # naive check. A boolean where a number belongs is always a mistake.
    if spec.type is not bool and isinstance(value, bool):
        raise ConfigError(f"{where}: expected {_type_name(spec.type)}, got a boolean")
    if spec.type is float and isinstance(value, int):
        return float(value)
    if not isinstance(value, spec.type):
        raise ConfigError(
            f"{where}: expected {_type_name(spec.type)}, got {type(value).__name__}"
        )
    return value


def _validate_mapping(raw: Any, spec: dict[str, Field], where: str) -> dict:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where or 'config'}: expected a mapping")

    prefix = f"{where}." if where else ""
    unknown = sorted(set(raw) - set(spec))
    if unknown:
        known = ", ".join(sorted(spec))
        raise ConfigError(
            f"unknown key(s) {', '.join(prefix + k for k in unknown)}. "
            f"Known keys at this level: {known}. A typo must not be absorbed."
        )

    out: dict[str, Any] = {}
    for key, field_spec in spec.items():
        location = f"{prefix}{key}"
        if key not in raw:
            if field_spec.required:
                raise ConfigError(f"missing required key: {location}")
            out[key] = list(field_spec.default) if field_spec.type is list else field_spec.default
            continue

        value = _check_type(raw[key], field_spec, location)

        if field_spec.choices is not None and value not in field_spec.choices:
            allowed = ", ".join(repr(c) for c in field_spec.choices)
            raise ConfigError(f"{location}: {value!r} is not one of {allowed}")

        if field_spec.spec is not None:
            value = _validate_mapping(value, field_spec.spec, location)
        elif field_spec.item_spec is not None:
            value = [
                _validate_mapping(item, field_spec.item_spec, f"{location}[{i}]")
                for i, item in enumerate(value)
            ]
        out[key] = value
    return out


def _validate_task(raw: dict) -> dict:
    if "kind" not in raw:
        raise ConfigError("missing required key: task.kind")
    kind = raw["kind"]
    if kind not in TASK_SPECS:
        allowed = ", ".join(sorted(TASK_SPECS))
        raise ConfigError(f"task.kind: {kind!r} is not one of {allowed}")
    task = _validate_mapping(raw, TASK_SPECS[kind], "task")
    if kind in ("train_cv", "p20_permutation"):
        _check_train_cv_init(task)
        _check_augmentation(task)
        _check_permutation(task)
    if kind == "train_graph_cv":
        _check_graph_determinism(task)
    if kind in ("p10x_gate_fullfit", "p10x_distribution"):
        _check_p10x_gate(task)
    if kind == "p10x_distribution":
        _check_p10x_distribution(task)
    if kind == "confound_ceiling":
        _check_confound_target(task)
    if kind == "p20_stratification_diagnostic":
        _check_permutation(task)
        _check_stratification_diagnostic(task)
    if kind in ("p21_ensemble_probe", "p21_error_consistency",
                "cross_arm_shrinkage"):
        _check_p21_arms(task)
    if kind == "cross_arm_shrinkage":
        _check_cross_arm_shrinkage(task)
    if kind == "p21_error_consistency":
        _check_p21_consistency(task)
    return task


def _check_cross_arm_shrinkage(task: dict) -> None:
    """[2026-09-01] The three reading thresholds must be ORDERED.

    Precedence is fails-to-shrink, then varies-widely, then
    shrinks-substantially. If ``shrinks_substantially_at`` were not
    strictly below ``fails_to_shrink_at``, an arm could satisfy both
    "shrinks substantially" and "does not shrink" at once, and the
    reading would be decided by evaluation order rather than by the
    record. Refused at LOAD, so a wrong pair costs nothing on the
    cluster.
    """
    substantially = float(task["shrinks_substantially_at"])
    fails = float(task["fails_to_shrink_at"])
    widely = float(task["varies_widely_at"])
    if not 0.0 < substantially < fails <= 1.0:
        raise ConfigError(
            f"task.shrinks_substantially_at ({substantially}) and "
            f"task.fails_to_shrink_at ({fails}) must satisfy "
            "0 < substantially < fails <= 1. Otherwise one arm could "
            "satisfy both cells and the reading would be decided by "
            "evaluation order rather than by the record"
        )
    if not 0.0 < widely < 1.0:
        raise ConfigError(
            f"task.varies_widely_at ({widely}) must lie in (0, 1): it is "
            "the SD of the per-arm shrinkage ratios, each in [0, ~1]. "
            "**It is an SD, not a range** -- the range statistic was "
            "rejected on evidence 2026-09-01 "
            "(phase21.SHRINKAGE_DISPERSION_STATISTIC_RULED), and a value "
            "calibrated for a range would be far too lax as an SD"
        )


def _check_p21_arms(task: dict) -> None:
    """[2026-09-01] The ruled arm set and seed basis, enforced at LOAD.

    * **Every arm is on the 237 cohort** (``phase21.ARM_SET_RULED``). A
      236-patient arm here would average over misaligned patient sets --
      the one thing the ruling exists to prevent -- and it would do so
      silently, producing a plausible number.
    * **Every arm declares the shared five seeds** exactly
      (``phase21.SEED_BASIS_RULED``). An arm carrying its own ten would
      weight the graph regime twice as heavily in the average, an
      unequal weighting nobody chose.
    * **No arm appears twice.** A duplicated arm is a silent reweighting
      of the mean.

    Refused at load rather than at runtime, so a drifted config costs
    nothing on the cluster.
    """
    from ..phase21 import SEED_BASIS_RULED  # noqa: F401  (documents the rule)

    seeds = [int(s) for s in task["seeds"]]
    if seeds != [1337, 2024, 7, 99, 12345]:
        raise ConfigError(
            f"task.seeds: {seeds} is not the ruled shared basis "
            "[1337, 2024, 7, 99, 12345] (phase21.SEED_BASIS_RULED)"
        )
    names = [arm["name"] for arm in task["arms"]]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ConfigError(
            f"task.arms: {duplicates} appear more than once. A duplicated "
            "arm is a silent reweighting of the ensemble mean"
        )
    for arm in task["arms"]:
        if int(arm["n_patients"]) != 237:
            raise ConfigError(
                f"task.arms: {arm['name']} declares n_patients "
                f"{arm['n_patients']}. The ruled set is the 237 COHORT ONLY "
                "(phase21.ARM_SET_RULED): averaging across cohort sizes "
                "would average over misaligned patient sets, silently"
            )
        if [int(s) for s in arm["seeds"]] != seeds:
            raise ConfigError(
                f"task.arms: {arm['name']} declares seeds {arm['seeds']}, "
                f"not the shared basis {seeds}. An arm carrying its own ten "
                "would weight the graph regime twice as heavily in the "
                "average (phase21.SEED_BASIS_RULED)"
            )
    if task.get("probe_arm") and task["probe_arm"] not in names:
        raise ConfigError(
            f"task.probe_arm {task['probe_arm']!r} is not in the arm set. "
            "The contrast must be within-phase and paired over the same "
            "patients"
        )


def _check_p21_consistency(task: dict) -> None:
    """[2026-09-01] Arm B's thresholds and binarisations, at LOAD time.

    The thresholds are ``phase21.CONSISTENCY_THRESHOLDS_RULED``'s, and
    they must bracket: inverted or equal thresholds would let two of the
    three cells fire, or none, and the reading would then be chosen by
    whoever read the output rather than declared before the run.

    The two binarisations are a closed set and are reported SEPARATELY;
    a config naming one of them would silently narrow the arm.
    """
    from ..phase21 import CONSISTENCY_THRESHOLDS_RULED

    low, high = float(task["low_kappa"]), float(task["high_kappa"])
    if not -1.0 < low < high < 1.0:
        raise ConfigError(
            f"task.low_kappa ({low}) and task.high_kappa ({high}) must "
            "satisfy -1 < low < high < 1. Otherwise two of the three "
            "committed cells fire, or none does, and the reading is chosen "
            "by whoever reads the output"
        )
    # Compared against the NUMERIC home, never against the record's
    # sentences: values are not parsed out of prose, and a re-wording
    # must not be able to move a threshold.
    if low != float(CONSISTENCY_THRESHOLDS_RULED["low_value"]):
        raise ConfigError(
            f"task.low_kappa {low} is not the RULED "
            f"{CONSISTENCY_THRESHOLDS_RULED['low_value']}. It is a locked "
            "setting; amend the record first"
        )
    if high != float(CONSISTENCY_THRESHOLDS_RULED["high_value"]):
        raise ConfigError(
            f"task.high_kappa {high} is not the RULED "
            f"{CONSISTENCY_THRESHOLDS_RULED['high_value']}. It is a locked "
            "setting; amend the record first"
        )
    if list(task["binarisations"]) != ["residual_sign", "worst_quartile"]:
        raise ConfigError(
            f"task.binarisations: {task['binarisations']} is not the "
            "registered pair ['residual_sign', 'worst_quartile']. Both are "
            "reported SEPARATELY (phase21.ARM_B_REGISTERED); naming one "
            "would silently narrow the arm"
        )


def _check_stratification_diagnostic(task: dict) -> None:
    """[2026-08-31] The diagnostic's three refusals, at LOAD time.

    * **A plain permutation has nothing to diagnose.** Arm P destroys
      everything by construction -- that is its entire purpose -- so
      running this against a ``plain`` block would measure a known
      answer and report it as a finding about Arm S.
    * **The thresholds must be ordered and inside (0, 1).** Overlapping
      or inverted thresholds would let two of the three committed cells
      fire, or none, and the reading would then be chosen by whoever
      read the output.
    * **A zero ceiling makes every fraction undefined**, and a negative
      one flips the sign of every comparison silently.
    """
    block = task.get("permutation") or {}
    if block.get("kind") != "stratified":
        raise ConfigError(
            f"task.permutation.kind: the diagnostic requires 'stratified', "
            f"got {block.get('kind')!r}. A 'plain' permutation destroys "
            "everything BY CONSTRUCTION -- that is Arm P's purpose -- so "
            "there is nothing to diagnose and the result would be a known "
            "answer reported as a finding about Arm S"
        )
    ceiling = float(task["ceiling_pcc"])
    if ceiling <= 0.0:
        raise ConfigError(
            f"task.ceiling_pcc: {ceiling!r} is not positive. It is the "
            "denominator of every fraction this task reports: zero makes "
            "them undefined and a negative flips every comparison silently"
        )
    destroyed = float(task["destroyed_fraction"])
    survived = float(task["survived_fraction"])
    if not 0.0 < destroyed < survived < 1.0:
        raise ConfigError(
            f"task.destroyed_fraction ({destroyed}) and "
            f"task.survived_fraction ({survived}) must satisfy "
            "0 < destroyed < survived < 1. Otherwise two of the three "
            "committed cells fire, or none does, and the reading is "
            "chosen by whoever reads the output rather than declared "
            "before the run"
        )


def _check_permutation(task: dict) -> None:
    """[2026-08-31] Phase 20's permutation block, refused at LOAD time.

    **The locked setting is enforced here, not merely documented.**
    ``phase20.STRATIFICATION_RULED`` fixes k at 10 and says it is "NEVER
    TUNED ACROSS RUNS, movement requires a dated amendment". A rule that
    lives only in a docstring is one a config can quietly leave behind
    -- the loader is where "never tuned" becomes true.

    Arm P is **k = 1** by construction, because a plain permutation IS
    the stratified one at a single stratum. A plain block carrying k > 1
    would name one arm and run the other.

    Refused at load rather than at runtime, so a wrong k costs nothing on
    the cluster and cannot be discovered after a job is queued.
    """
    from ..phase20 import STRATIFICATION_RULED

    block = task.get("permutation")
    if not block:
        return
    if set(block) != {"kind", "k_strata"}:
        raise ConfigError(
            f"task.permutation: keys must be exactly {{kind, k_strata}}, got "
            f"{sorted(block)}. The stratification VARIABLE is not a config "
            "field -- it is the locked rule in "
            "phase20.STRATIFICATION_RULED, and a pre-registration a config "
            "could edit is not one"
        )
    kind = block["kind"]
    if kind not in ("plain", "stratified"):
        raise ConfigError(
            f"task.permutation.kind: {kind!r} is not a permutation kind; "
            "expected 'plain' (Arm P, the integrity gate) or 'stratified' "
            "(Arm S, the attribution)"
        )
    k = block["k_strata"]
    if not isinstance(k, int) or isinstance(k, bool):
        raise ConfigError(f"task.permutation.k_strata: {k!r} is not an integer")
    if kind == "plain" and k != 1:
        raise ConfigError(
            f"task.permutation.k_strata: a 'plain' permutation is ONE "
            f"stratum, so k_strata must be 1, not {k}. A plain block "
            "carrying k > 1 would name Arm P and run Arm S"
        )
    locked = int(STRATIFICATION_RULED["k_strata"])
    if kind == "stratified" and k != locked:
        raise ConfigError(
            f"task.permutation.k_strata: {k} is not the locked {locked} "
            "(phase20.STRATIFICATION_RULED). k is a LOCKED SCIENTIFIC "
            "SETTING -- 'never tuned across runs, movement requires a "
            "dated amendment'. Amend the record first; the loader will "
            "then accept the new value"
        )


def _check_confound_target(task: dict) -> None:
    """[2026-08-31] The confound ceiling's target, and what each needs.

    ``label`` absent is **Phase 13's measurement**: the median grade
    through ``run._grades_for``, which reads the score sheet -- so the
    scoresheet input is required exactly then, and its absence is a
    load-time refusal rather than a KeyError halfway through a job.

    ``label`` present is **Phase 20's Arm C**, reading a manifest column,
    which needs no score sheet at all. Declaring one there would put an
    input in the run's provenance that the run never opens.

    ``materially_different`` is Arm C's reading threshold and is required
    with it: which of the two committed cells fires
    (``phase20.ARM_C_REGISTERED["readings"]``) cannot be a runtime
    choice, and a threshold chosen after the number is not a threshold.
    """
    if task.get("label"):
        if task.get("scoresheet_artifact"):
            raise ConfigError(
                "task.scoresheet_artifact is declared beside task.label: a "
                "labelled confound ceiling reads a MANIFEST COLUMN and "
                "never opens the score sheet, so declaring it would record "
                "an input the run does not use"
            )
        if task.get("materially_different") is None:
            raise ConfigError(
                "task.materially_different is required with task.label: it "
                "is Arm C's reading threshold "
                "(phase20.ARM_C_REGISTERED['readings']), and a threshold "
                "chosen after the number is not a threshold"
            )
    elif not task.get("scoresheet_artifact"):
        raise ConfigError(
            "task.scoresheet_artifact is required without task.label: the "
            "default target is the MEDIAN grade, which run._grades_for "
            "resolves through the score sheet"
        )


def _check_p10x_distribution(task: dict) -> None:
    """[2026-08-31] The distribution arm's three bespoke refusals.

    * **All five raters, by ruling (a)** -- the across-rater spread is
      part of what is characterised
      (``phase10_annex.RULING_A_ALL_FIVE_RATERS``), so a subset is a
      different experiment and is refused rather than run.
    * **The enumerated seeds must re-derive from the declared root.** A
      list that does not is either hand-edited or stale, and either way
      the draw set would no longer be reproducible from the root the
      config claims -- the generator-carries-filled-values pattern,
      applied to seeds.
    * **The shard must be a real partition member**: ``count >= 1`` and
      ``0 <= index < count``. An out-of-range index would run zero
      draws and report an empty distribution as though it were a
      finished one.
    """
    from ..data.scoresheet import RATERS
    from ..run import p10x_split_seeds

    if tuple(task["raters"]) != RATERS:
        raise ConfigError(
            f"task.raters must be all five sheet columns in sheet order "
            f"{list(RATERS)}; got {task['raters']}. Ruling (a) runs all "
            "five -- the across-rater spread is part of what the annex "
            "characterises (phase10_annex.RULING_A_ALL_FIVE_RATERS)"
        )

    n_draws = int(task["n_draws"])
    seeds = [int(s) for s in task["split_seeds"]]
    if len(seeds) != n_draws:
        raise ConfigError(
            f"task.split_seeds has {len(seeds)} entries but task.n_draws "
            f"is {n_draws}; a short draw set would silently be a smaller "
            "experiment than the one N was ruled for"
        )
    expected = p10x_split_seeds(int(task["root_seed"]), n_draws)
    if seeds != expected:
        first = next(
            i for i, (a, b) in enumerate(zip(seeds, expected)) if a != b
        )
        raise ConfigError(
            f"task.split_seeds does not re-derive from task.root_seed "
            f"{task['root_seed']} (first disagreement at index {first}: "
            f"{seeds[first]} vs {expected[first]}). The enumerated list "
            "must be exactly what the declared rule produces, or the "
            "draw set is not reproducible from the root the config claims"
        )

    count, index = int(task["shard"]["count"]), int(task["shard"]["index"])
    if count < 1 or not 0 <= index < count:
        raise ConfigError(
            f"task.shard {{count: {count}, index: {index}}} is not a "
            "partition member: count must be >= 1 and 0 <= index < "
            "count. An out-of-range shard would run zero draws and "
            "report an empty distribution as a finished one"
        )


def _check_p10x_gate(task: dict) -> None:
    """[2026-08-30] The annex gate's two bespoke refusals.

    * ``region_count`` other than 36 is refused; **27 is refused BY
      NAME** with the record's reason -- 27 is what the group states
      and 36 is what their code ran and what produced every published
      number (``phase10.REGION_COUNT_BELIEF_VS_EXECUTION``), so a 27
      here would run the architecture that has never been run and call
      it a replication.
    * ``normalisation`` must be the notebook's CIFAR-10 constants,
      exactly (``phase10_annex.NORMALISATION_RULED``): the ruling
      reproduces the defect KNOWINGLY, and any other statistics --
      ImageNet's included -- would reopen 'our normalisation differed'
      as an alternative explanation for any gap.
    """
    from ..models.cleftgnn import CIFAR10_NORMALISATION

    count = task["region_count"]
    if count == 27:
        raise ConfigError(
            "task.region_count: 27 is UNREGISTERED for the annex "
            "(phase10.REGION_COUNT_BELIEF_VS_EXECUTION): the group states "
            "27 in three places and their code runs 36 -- every published "
            "number came from 36, and the reported 27-region architecture "
            "has never been run. The annex replicates what ran."
        )
    if count != 36:
        raise ConfigError(
            f"task.region_count: {count!r} is not 36, the notebook's own "
            "executed enumeration (phase10.NOTEBOOK_ROI_SET_MEASURED)"
        )
    mean, std = CIFAR10_NORMALISATION
    declared = (
        tuple(float(v) for v in task["normalisation"]["mean"]),
        tuple(float(v) for v in task["normalisation"]["std"]),
    )
    if declared != (mean, std):
        raise ConfigError(
            f"task.normalisation: {declared!r} is not the notebook's "
            f"CIFAR-10 statistics {(mean, std)!r} "
            "(phase10_annex.NORMALISATION_RULED -- the defect is "
            "replicated KNOWINGLY, and nothing else is admitted)"
        )


#: The closed set of augmentation flags. Phase 7C's strengths are NOT here:
#: they are pre-registered in ``cleft.phase7c`` where the suite checks them,
#: because a pre-registration a config could edit is not one.
AUGMENTATION_FLAGS = ("photometric", "geometric", "rotation", "region_aware")


def _check_augmentation(task: dict) -> None:
    """**A policy the augmenter cannot build is refused at LOAD time.**

    Two things, both cheap here and expensive on the cluster:

    * **an unknown key** -- a config saying ``photmetric: true`` would
      otherwise validate, run, and produce an arm with no augmentation at all
      whose metrics.json said it was augmented. Silent, plausible, wrong.
    * **rotation without geometric** -- rotation IS a geometric transform, so
      the flag alone declares something with no meaning. ``augment.Policy``
      refuses it too; this refuses it before a job is queued.
    """
    augmentation = task.get("augmentation")
    if augmentation is None:
        return

    unknown = sorted(set(augmentation) - set(AUGMENTATION_FLAGS))
    if unknown:
        raise ConfigError(
            f"task.augmentation has unknown key(s) {unknown}; expected any of "
            f"{list(AUGMENTATION_FLAGS)}. An unrecognised flag would leave the "
            "arm unaugmented while its record said otherwise."
        )
    for name, value in augmentation.items():
        if not isinstance(value, bool):
            raise ConfigError(
                f"task.augmentation.{name} is {value!r}; the flags are "
                "booleans and the STRENGTHS live in cleft.phase7c, "
                "pre-registered rather than set per config"
            )
    if augmentation.get("rotation") and not augmentation.get("geometric"):
        raise ConfigError(
            "task.augmentation declares rotation without geometric. Rotation "
            "is a geometric transform and cannot be applied on its own -- see "
            "phase7c.ROTATION_IS_AN_ARM, where it varies between two arms that "
            "both carry the family."
        )
    if task.get("trainable", "head") != "head":
        raise ConfigError(
            f"task.augmentation with trainable={task.get('trainable')!r}. The "
            "augmenting path keeps the backbone frozen and trains the head; "
            "full fine-tuning on 237 images measured PCC -0.013 (PLAN §4.7) "
            "and is not what this phase varies."
        )


def _check_graph_determinism(task: dict) -> None:
    """**An AG-Net arm must say what it does about determinism.**

    ``graph_cleft.run`` refuses an undeclared AG-Net arm at runtime; this
    refuses it at LOAD time, so it costs nothing on the cluster and cannot be
    discovered after a job has been queued.

    Not a default either way, and that is the point: default-on and the arm
    dies mid-run on an op nobody expected; default-off and its results carry
    a ~2.3e-05 tolerance nobody declared. See
    ``models.agnet.TRAINING_DETERMINISM``.
    """
    if task.get("backbone") == "agnet" and task.get("deterministic") is None:
        raise ConfigError(
            "task.backbone is 'agnet' but task.deterministic is not set. "
            "[MEASURED] the deterministic roi_align substitution cannot be "
            "satisfied in the pinned image (no C++ compiler for inductor), so "
            "true raises mid-run and false bounds this arm's same-seed "
            "reproducibility at ~2.3e-05 rather than bitwise. Every "
            "comparison against this arm carries that, so it is declared "
            "rather than defaulted."
        )


def _check_train_cv_init(task: dict) -> None:
    """**A pretrained init has to come from somewhere.**

    Live extraction builds ViT-B/16 at ImageNet init and nothing else
    (``phase3.prepare_features``), so ``init: scut_masked`` without an
    ``embeddings_artifact`` would run happily on ImageNet vectors and record
    a pretraining that never touched this arm. Every field would look right.

    Refused at LOAD time rather than at runtime, so it costs nothing on the
    cluster and cannot be discovered after a job has been queued.
    """
    init = task.get("init", "imagenet")
    artifact = task.get("embeddings_artifact")
    checkpoint = task.get("checkpoint")
    backbone = task.get("backbone")

    #: Live extraction builds ViT-B/16 and nothing else
    #: (``torch_backbone.extract_embeddings`` at ``DEFAULT_BACKBONE``), so any
    #: OTHER backbone without an artifact would train on ViT features while
    #: metrics.json named it. Every shape stays right and the number is
    #: attributed to a model that never ran. The stub and ridge arms compute
    #: their own features and are exempt.
    concat_entries = task.get("concat_embeddings_artifacts")
    if backbone == "concat":
        # Arm 4's shape, closed both ways: a concat arm without its
        # constituents has nothing to train on, and constituents beside a
        # single embeddings_artifact would leave which representation ran
        # ambiguous -- graph_cleft refuses the same ambiguity for folds.
        if not concat_entries or len(concat_entries) < 2:
            raise ConfigError(
                "task.backbone is 'concat' but concat_embeddings_artifacts "
                "declares fewer than two sets. A concatenation of one set "
                "is that set wearing a different arm's name."
            )
        if artifact:
            raise ConfigError(
                "a concat arm declares concat_embeddings_artifacts, not "
                "embeddings_artifact -- both together leave ambiguous which "
                "representation the arm ran on."
            )
        if init != "imagenet":
            raise ConfigError(
                "the concat arm is imagenet-only: its constituents are the "
                "7D singles, none of which has a pretraining checkpoint to "
                "pair against."
            )
    elif concat_entries:
        raise ConfigError(
            f"concat_embeddings_artifacts is declared but task.backbone is "
            f"{backbone!r}. Only the 'concat' arm consumes several sets; on "
            "any other arm the extra declarations would be verified and "
            "recorded as feeding a run they never fed."
        )

    if backbone in (
        "swin_b", "srgnn", "vit_b32", "vit_b8", "mvitv2_b",
        "vit_b14_dinov2", "vit_b16_dino",
    ) and not artifact:
        raise ConfigError(
            f"task.backbone is {backbone!r} but no task.embeddings_artifact is "
            "declared. Live extraction builds ViT-B/16 only, so this arm would "
            f"train on ViT features while reporting {backbone!r}. Declare the "
            "pooled set for that backbone."
        )

    if init != "imagenet" and not artifact:
        raise ConfigError(
            f"task.init is {init!r} but no task.embeddings_artifact is "
            "declared. Live extraction is ImageNet-only, so this arm would "
            "train on ImageNet vectors while its metrics.json claimed "
            f"{init!r}. Declare the pooled set for that init."
        )
    # **[2026-09-02] The test is FACTORY-PRETRAINED, not "is imagenet".**
    # It was the same test until Phase 25 added two inits whose weights are
    # also the factory's own -- DINOv2 and DINO have no pretraining
    # checkpoint to declare, exactly as ImageNet has none. The list is
    # imported from train.extract so there is ONE definition of which
    # inits carry a checkpoint, not two that can drift.
    from ..train.extract import FACTORY_PRETRAINED_INITS

    if init not in FACTORY_PRETRAINED_INITS and not checkpoint:
        raise ConfigError(
            f"task.init is {init!r} but no task.checkpoint is declared. The "
            "artifact's recorded checkpoint_sha256 is compared against the "
            "declared checkpoint's VERIFIED hash; without it, features from "
            "one representation could be trained under another's name."
        )
    if init in FACTORY_PRETRAINED_INITS and checkpoint:
        raise ConfigError(
            f"task.init is {init!r} but task.checkpoint is declared. "
            "ImageNet has no pretraining checkpoint, so the declaration "
            "describes a run this one is not -- and the input would still be "
            "hash-verified and recorded in inputs.json."
        )
    if artifact and task.get("patch_scheme", "whole") != "whole":
        raise ConfigError(
            "task.embeddings_artifact is declared alongside "
            f"patch_scheme {task['patch_scheme']!r}. The pooled sets are "
            "WHOLE-IMAGE vectors; the patch path extracts its own per-patch "
            "embeddings live, so one of the two would be silently discarded."
        )


def validate(raw: Any) -> dict:
    """Validate a raw mapping into a normalised config dict."""
    if raw is None:
        raise ConfigError("config is empty")
    config = _validate_mapping(raw, SCHEMA, "")
    config["task"] = _validate_task(config["task"])
    _check_input_names(config["inputs"])
    _check_input_references(config["task"], config["inputs"])
    # After name and hash validation, so a config with a duplicate input or a
    # malformed digest fails on that rather than on an unset variable.
    config["inputs"] = expand_input_paths(config["inputs"])
    return config

#: Phase 27 (``phase27.EXIT_CRITERIA``, locked 2026-09-06): a linear head
#: fit on the 25 anchor grades and evaluated on all 237 as pure test.
#:
#: **The three fields a training task normally carries are ABSENT and
#: that is the ruling, not an omission**: ``inner_val_frac``, ``monitor``
#: and ``patience`` are not in this spec, so a config cannot reintroduce
#: checkpoint selection at a size that cannot support it
#: (``phase27.THE_INNER_VALIDATION_RULED``). ``max_epochs`` is therefore
#: a fixed BUDGET rather than a ceiling.
#:
#: Every input reference here follows the ``*_artifact`` convention, so
#: guard 3 hashes all three without extending ``INPUT_REFERENCE_KEYS``.
TASK_SPECS["p27_anchor_train"] = {
    "kind": Field(str, choices=("p27_anchor_train",)),
    "arm": Field(str, choices=("p27_anchor_train",)),
    #: TRAINING. The 25-row anchor set. phase16.load_anchor_set refuses
    #: any directory whose namespace is not anchor_deall, which is part
    #: one of the disjointness guard
    #: (phase27.THE_DISJOINTNESS_GUARD_DESIGNED).
    "anchor_artifact": Field(str),
    #: EVALUATION. The cohort manifest supplies the panel mean and the
    #: patient ids; the embeddings artifact supplies the features.
    "manifest_artifact": Field(str),
    "embeddings_artifact": Field(str),
    "geometry": Field(str, choices=("g1",)),
    "backbone": Field(str, choices=("vit_b16",)),
    "seeds": Field(list),
    #: A fixed budget. The loop runs all of it: there is no early stop.
    "max_epochs": Field(int, choices=(40,)),
    #: Where the phase reads, declared in advance and not selectable.
    #: A config naming any other pair is refused by the choices check
    #: (phase27.THE_TWO_READOUT_EPOCHS_DECLARED).
    "readout_epochs": Field(list),
    "primary_epoch": Field(int, choices=(40,)),
    "learning_rate": Field(float),
    "weight_decay": Field(float),
    #: The setting that actually governs this head: full-batch
    #: optimiser steps inside one epoch. batch_size is NOT here,
    #: because EmbeddingHeadBackbone never reads it
    #: (phase27.THE_INIT_IS_THE_PROBES).
    "max_steps": Field(int, choices=(50,)),
    #: The 0.2056 offset between the two targets' means, declared so the
    #: value metrics are reported with it attached rather than beside it
    #: (phase27.THE_LABEL_INCOMPARABILITY_ADDRESSED).
    "target_offset": Field(float),
    "expect_anchors": Field(int, choices=(25,)),
    "expect_patients": Field(int, choices=(237,)),
}



#: Task fields that name a DECLARED INPUT instead of carrying a value.
#: The convention is ``*_artifact``; these three do not follow it, and
#: were found by sweeping all 332 configs rather than recalled.
INPUT_REFERENCE_KEYS = ("artifact", "synth_set", "input")


def _input_references(node: Any, trail: str = "") -> list[tuple[str, str]]:
    """Every ``(field-path, value)`` in a task that names an input."""
    found: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for raw_key, value in node.items():
            key = str(raw_key)
            if isinstance(value, (dict, list)):
                found += _input_references(value, f"{trail}{key}.")
            elif isinstance(value, str) and (
                key.endswith("_artifact") or key in INPUT_REFERENCE_KEYS
            ):
                found.append((f"{trail}{key}", value))
    elif isinstance(node, list):
        for item in node:
            found += _input_references(item, trail)
    return found


def _check_input_references(task: dict, inputs: list[dict]) -> None:
    """**A task may not name an input the config does not declare.**

    **[ADDED 2026-09-01, after six configs shipped unloadable.]** The
    field types were validated and the references never were, so
    ``embeddings_artifact: pooled_vit_b16_imagenet_g1`` -- a name that
    exists in no config anywhere -- passed every check and died as a
    bare ``KeyError`` inside the task, after the launch.

    ``Field(str)`` is satisfied by any string. This is the check that
    the string means something. Same class as Phase 17's writer/reader
    filename break: two halves each self-consistent, disagreeing at the
    join, with nothing testing the join.
    """
    declared = {entry["name"] for entry in inputs}
    for field, value in _input_references(task):
        if value not in declared:
            raise ConfigError(
                f"task.{field} = {value!r} names no declared input. Declared: "
                f"{sorted(declared) or '(none)'}. A by-name reference to an "
                "input that does not exist fails at run time as a KeyError, "
                "so it is refused here."
            )


def _check_input_names(inputs: list[dict]) -> None:
    seen = set()
    for entry in inputs:
        if entry["name"] in seen:
            raise ConfigError(f"duplicate input name: {entry['name']!r}")
        seen.add(entry["name"])
        digest = entry["rollup_sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ConfigError(
                f"inputs[{entry['name']}].rollup_sha256 is not a 64-character "
                f"lowercase hex digest: {digest!r}"
            )


def load_config(path: str | Path) -> dict:
    """Load and validate a config file."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    try:
        return validate(raw)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from None


#: Keys ``expand_input_paths`` adds to an input entry. **Derived, not authored**:
#: they record where a resolved path came from, and a config file never contains
#: them. ``dump_config`` drops them so that dumping a loaded config yields
#: something that loads again -- without this, ``load_config(dump_config(cfg))``
#: fails on its own output for any config using ``${CLEFT_*}``, because the
#: loader is strict about unknown keys and is right to be.
DERIVED_INPUT_KEYS = ("path_declared_as", "path_from_env")


def dump_config(config: dict, path: str | Path | None = None) -> str | Path:
    """Dump the AUTHORED config canonically. Returns the text, or the path.

    Derived fields are dropped (``DERIVED_INPUT_KEYS``) and an expanded input path
    is restored to the declaration it came from, so the dump is a config file
    rather than a snapshot of one machine's resolution. Dumping the resolved path
    would quietly turn a portable config into a machine-specific one -- the same
    trap as ``declare_inputs.py``'s paste block.

    The run directory does not use this: it copies ``config.yaml`` byte for byte,
    and ``inputs.json`` carries the resolution.
    """
    dumped = dict(config)
    inputs = dumped.get("inputs")
    if isinstance(inputs, list):
        restored = []
        for entry in inputs:
            if not isinstance(entry, dict):
                restored.append(entry)
                continue
            clean = {k: v for k, v in entry.items() if k not in DERIVED_INPUT_KEYS}
            declared = entry.get("path_declared_as")
            if declared:
                clean["path"] = declared
            restored.append(clean)
        dumped["inputs"] = restored

    text = yaml.safe_dump(dumped, sort_keys=True, default_flow_style=False)
    if path is None:
        return text
    path = Path(path)
    path.write_text(text, encoding="utf-8")
    return path
