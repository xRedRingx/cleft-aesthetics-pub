"""End-to-end exercise of the whole path via ``cleft.run``.

configs/smoke.yaml does no real work but touches every part of the contract:
declared input, hash check, run directory, metrics, tiered outputs. It is the
template every real config follows, so if this drifts the templates drift.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from datetime import datetime, timezone

import numpy as np
import warnings

import pytest

from cleft.provenance import GuardError, hash_dir
from cleft.run import main

from fixtures import builders

FIXED_NOW = datetime(2026, 7, 26, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def values_dir(tmp_path):
    """A synthetic values.csv artifact, shaped like the real smoke input."""
    root = tmp_path / "smoke_input"
    root.mkdir()
    truth, pred = builders.graded_pair(n=64, seed=1337)
    with (root / "values.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "truth", "pred"])
        for i, (t, p) in enumerate(zip(truth, pred), start=1):
            writer.writerow([f"P{i:04d}", f"{t:.6f}", f"{p:.6f}"])
    return root


@pytest.fixture
def smoke_config(tmp_path, values_dir):
    return builders.write_config(
        tmp_path / "p0_smoke.yaml",
        tier="keeper",
        inputs=[
            {
                "name": "values",
                "path": str(values_dir),
                "rollup_sha256": hash_dir(values_dir)["rollup"],
            }
        ],
        task={"kind": "smoke", "n_samples": 64},
    )


@pytest.fixture
def on_fixture_repo(monkeypatch, clean_repo):
    monkeypatch.setenv("CLEFT_REPO_ROOT", str(clean_repo))
    return clean_repo


def _run(config, out_root, argv_extra=()):
    return main(["--config", str(config), "--out", str(out_root), *argv_extra])


def test_smoke_run_produces_a_compliant_run_directory(
    smoke_config, out_root, on_fixture_repo
):
    """Exit criterion 1."""
    run_dir = _run(smoke_config, out_root)

    for member in ("config.yaml", "env.json", "inputs.json", "outputs.json", "log.txt", "code"):
        assert (run_dir / member).exists(), f"missing {member}"
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "curves.csv").is_file()
    assert (run_dir / "predictions.csv").is_file()


def test_running_twice_never_overwrites_the_first(smoke_config, out_root, on_fixture_repo):
    """Exit criterion 2."""
    first = _run(smoke_config, out_root)
    first_bytes = (first / "metrics.json").read_bytes()

    second = _run(smoke_config, out_root)

    assert second != first
    assert first.exists()
    assert (first / "metrics.json").read_bytes() == first_bytes


def test_smoke_metrics_are_correct(smoke_config, out_root, on_fixture_repo, values_dir):
    from cleft.eval import metrics as M

    run_dir = _run(smoke_config, out_root)
    reported = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    with (values_dir / "values.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    truth = np.array([float(r["truth"]) for r in rows])
    pred = np.array([float(r["pred"]) for r in rows])

    assert reported["n"] == 64
    assert reported["pcc"] == pytest.approx(M.pcc(truth, pred), abs=1e-9)
    assert reported["spearman"] == pytest.approx(M.spearman(truth, pred), abs=1e-9)
    assert reported["qwk_3cat"] == pytest.approx(M.qwk_3cat(truth, pred), abs=1e-9)
    assert reported["mae"] == pytest.approx(M.mae(truth, pred), abs=1e-9)
    assert reported["rmse"] == pytest.approx(M.rmse(truth, pred), abs=1e-9)
    lo, hi = reported["pcc_ci95"]
    assert lo < reported["pcc"] < hi


def test_metrics_json_contains_no_patient_identifiers(
    smoke_config, out_root, on_fixture_repo
):
    """metrics.json is SHAREABLE, so it must be aggregate scalars only."""
    run_dir = _run(smoke_config, out_root)
    text = (run_dir / "metrics.json").read_text(encoding="utf-8")
    assert "P0001" not in text
    assert "patient_id" not in text


def test_predictions_csv_is_marked_cluster_only(smoke_config, out_root, on_fixture_repo):
    run_dir = _run(smoke_config, out_root)
    header = (run_dir / "predictions.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "CLUSTER-ONLY" in header

    index = json.loads((run_dir / "outputs.json").read_text(encoding="utf-8"))
    tiers = {e["name"]: e["tier"] for e in index["outputs"]}
    assert tiers["predictions.csv"] == "CLUSTER-ONLY"
    assert tiers["metrics.json"] == "SHAREABLE"


def test_outputs_index_records_a_hash_of_every_output(
    smoke_config, out_root, on_fixture_repo
):
    from cleft.provenance import hash_file

    run_dir = _run(smoke_config, out_root)
    index = json.loads((run_dir / "outputs.json").read_text(encoding="utf-8"))
    for entry in index["outputs"]:
        assert entry["sha256"] == hash_file(run_dir / entry["name"])


def test_dirty_tree_is_diverted_to_dev(smoke_config, out_root, monkeypatch, dirty_repo):
    """Exit criterion 3, from the run.py side."""
    monkeypatch.setenv("CLEFT_REPO_ROOT", str(dirty_repo))

    # keeper + dirty aborts outright
    with pytest.raises(GuardError, match="dirty"):
        _run(smoke_config, out_root)

    # the same config at tier dev proceeds, with the flag recorded
    raw = smoke_config.read_text(encoding="utf-8").replace("tier: keeper", "tier: dev")
    smoke_config.write_text(raw, encoding="utf-8")
    run_dir = _run(smoke_config, out_root)

    assert run_dir.parent.parent.name == "dev"
    env = json.loads((run_dir / "env.json").read_text(encoding="utf-8"))
    assert env["git"]["dirty_tree"] is True


def test_corrupted_input_refuses_to_start(
    smoke_config, out_root, on_fixture_repo, values_dir
):
    """Exit criterion 5, from the run.py side."""
    assert _run(smoke_config, out_root).exists()

    path = values_dir / "values.csv"
    payload = bytearray(path.read_bytes())
    payload[-2] ^= 0x01
    path.write_bytes(bytes(payload))

    with pytest.raises(GuardError, match="hash"):
        _run(smoke_config, out_root)


class OwedTaskKind(UserWarning):
    """A task kind with no config yet -- surfaced on EVERY suite run.

    Not an error: an owed config is a legitimate state (the run
    directories a contrast reads do not exist until the arms run). But it
    is a state that must be VISIBLE, because the alternative is a green
    suite and a FileNotFoundError at launch.
    """


def _p25_contrasts_still_owed() -> bool:
    """True while the Phase 25 generator has no run directories."""
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "gen_p25_owed", root / "scripts" / "generate_phase25_configs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return not module.RUN_DIRECTORIES


#: ``{task kind: (why it is owed, predicate that is True while it still
#: is)}``. **Every entry must carry a predicate that eventually returns
#: False**, so an exemption expires on the condition that created it
#: rather than on a date -- and warns, loudly, until it does.
#:
#: Empty is the healthy state. Phase 25's entry stands as the worked
#: example: it was live for three cycles and is now satisfied, so its
#: predicate returns False and nothing warns.
OWED_TASK_KINDS = {
    "p25_contrasts": (
        "it reads two Phase 25 arm RUN DIRECTORIES, whose names are the "
        "cluster's and are not derivable here; supply them with "
        "scripts/generate_phase25_configs.py --run-dirs",
        _p25_contrasts_still_owed,
    ),
}


def test_every_task_kind_has_at_least_one_config(repo_root):
    """**A task with no config is invisible until someone tries to launch it.**

    The shipped-config test above checks the configs that exist. It cannot see a
    task kind that has none -- the code is written, registered, schema'd and
    tested, and there is simply nothing to point ``--config`` at. That is how
    ``feature_relevance`` shipped unrunnable: everything about it was present
    except the one file that makes it reachable.

    Nothing scientific is a flag (Part 2.3), so a task is reachable ONLY through
    a config. No allowlist here on purpose: a kind that genuinely needs none is a
    deliberate decision, and it should cost an edit to this test to say so.
    """
    import yaml

    from cleft.config.schema import TASK_SPECS

    shipped: dict[str, list[str]] = {}
    for path in sorted((repo_root / "configs").glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        kind = (payload.get("task") or {}).get("kind")
        shipped.setdefault(kind, []).append(path.name)

    # **[2026-09-02] An exemption here is NOISY, never silent.**
    #
    # The Phase 25 exemption below removed itself as designed -- but while
    # it stood, the suite was GREEN and the owed config surfaced only as a
    # ``FileNotFoundError`` at launch, three cycles later. A green suite is
    # how this project reads its own state, and an exemption that does not
    # disturb it is a state nobody sees.
    #
    # **Two options were weighed and the first was taken (the maintainer's
    # ruling; recorded at ``phase25.THE_EXEMPTION_IS_NOISY_NOW``):**
    #  (1) PRINT WHAT IS OWED ON EVERY RUN -- a standing, visible line.
    #  (2) a TIME CAP, failing the suite after N days.
    # (2) converts a silent state into a sudden failure at an arbitrary
    # date, which is a different silence with a worse ending; (1) keeps
    # the state in front of whoever runs the suite, every time.
    #
    # ``warnings.warn`` rather than ``print`` because pytest captures
    # stdout by default and surfaces warnings in its summary -- noisy
    # means noisy WITHOUT a flag.
    owed_kinds = set()
    for kind, (why, still_owed) in OWED_TASK_KINDS.items():
        if not still_owed():
            continue
        owed_kinds.add(kind)
        warnings.warn(
            f"OWED: task kind {kind!r} has no shipped config -- {why}",
            OwedTaskKind,
            stacklevel=2,
        )

    unrunnable = sorted(set(TASK_SPECS) - set(shipped) - owed_kinds)
    assert not unrunnable, (
        f"task kinds with no config, so nothing can launch them: {unrunnable}. "
        "Add configs/<phase>_<name>.yaml."
    )


def test_no_shipped_config_runs_a_full_build_without_writing_its_artifact(
    repo_root, monkeypatch
):
    """A task kind that DECLARES an artifact (``out_version`` plus
    ``write_artifact`` in its schema) must not ship a config in the silent
    mode: full cohort, nothing persisted.

    **[MEASURED 2026-07-31] This is not hypothetical.** All three
    p5_masked_scut runs took exactly that mode -- parity measured over 5,499
    faces, sheet approved -- and ``data/scut/masked_v1``, which the eight
    p6_pretrain masked configs declare as an input, never existed on any
    machine. Same class as ``test_every_task_kind_has_a_config``: everything
    green, and the deliverable unreachable. The task handlers refuse the
    combination at runtime; this catches it before anything is launched, and
    sweeps every artifact-declaring kind automatically rather than naming two.
    """
    from cleft.config import load_config
    from cleft.config.schema import TASK_SPECS

    artifact_kinds = {
        kind
        for kind, spec in TASK_SPECS.items()
        if "out_version" in spec and "write_artifact" in spec
    }
    assert artifact_kinds, (
        "no artifact-declaring task kinds found; the sweep is checking nothing"
    )

    _sentinel_every_env_root(repo_root, monkeypatch, "artifact-sweep")
    swept, offenders = 0, []
    for path in sorted((repo_root / "configs").glob("*.yaml")):
        task = load_config(path)["task"]
        if task["kind"] not in artifact_kinds:
            continue
        swept += 1
        if not task["write_artifact"] and not task["n_faces"]:
            offenders.append(path.name)

    assert swept, "no shipped config uses an artifact-declaring kind"
    assert not offenders, (
        f"configs in the silent full-build mode (n_faces 0, write_artifact "
        f"false): {offenders}. A full build must write its artifact; a review "
        "pass must say how many faces it renders."
    )


def test_no_committed_config_declares_an_unportable_path(repo_root):
    """**A committed config is one that gets reproduced on the cluster, so its
    declared paths must resolve there.**

    This test did not exist, and that is why the class was not caught. A drive
    letter cannot resolve on Linux, so `C:/data/scut-fbp5500/SCUT-FBP5500_v2`
    made `p5_masked_scut.yaml` a laptop-only config — and the failure it produced
    was not "path not found" but a **silent concatenation** onto the repo root,
    `/home/user/codex/cleft-aesthetics/C:/data/...`, which reads as missing data
    rather than as a malformed declaration.

    Stated as a POSITIVE rule over a closed set of kinds rather than a blacklist,
    so a new machine-specific shape — `~/data`, `%USERPROFILE%\\data` — classifies
    as something and fails, instead of falling through a check that only knew
    about drive letters. That is the difference between this and a test keyed to
    one exact string (PLAN R7, instance 4).
    """
    import yaml

    from cleft.config.schema import (
        PORTABLE_PATH_KINDS,
        classify_declared_path,
    )

    offenders = []
    for path in sorted((repo_root / "configs").glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        for entry in payload.get("inputs") or []:
            declared = entry.get("path")
            kind = classify_declared_path(declared)
            if kind not in PORTABLE_PATH_KINDS:
                offenders.append(f"{path.name}: {entry.get('name')} -> {declared} [{kind}]")

    assert not offenders, (
        "committed configs declaring machine-specific paths:\n  "
        + "\n  ".join(offenders)
        + "\n\nA committed config must declare each input as one of "
        f"{PORTABLE_PATH_KINDS}: repo-relative, POSIX-absolute (the cluster), or "
        "${CLEFT_<NAME>} resolved from the environment with the declared "
        "rollup_sha256 verifying whatever it resolved to."
    )


def test_no_tracked_file_names_the_cohort_by_path(repo_root):
    """**[WIDENED 2026-09-05] The guard above globbed ``configs/*.yaml``,
    and the thing it was guarding against lived in four places.**

    When the cohort paths were parameterised, the literals were found in
    38 configs, 3 generators and 16 source and test files -- 3,001
    occurrences. The narrow guard passed throughout, because it only ever
    looked at ``inputs[].path`` inside ``configs/``. A guard whose scope
    is narrower than its subject reports a clean sweep it did not run.

    Asserted by ROLE, not by one literal: ``COHORT_PATH_MARKERS`` names
    the identifying segments in one place, so a path reintroduced under a
    different spelling is a one-line addition here rather than a silent
    pass. The positive half -- that a cohort input is declared through
    ``COHORT_ENV_REFERENCES`` -- is checked by the test below it.
    """
    import subprocess

    from cleft.config.schema import COHORT_PATH_MARKERS

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=repo_root,
        capture_output=True, text=True, check=True,
    ).stdout.split("\n")

    # The record ABOUT the parameterisation has to name what it replaced,
    # and the two places it could not reach are enumerated in the record
    # rather than skipped quietly -- see COHORT_PATHS_STILL_OPEN.
    from cleft.config.schema import COHORT_PATHS_STILL_OPEN

    assert "frozen_tree" in COHORT_PATHS_STILL_OPEN

    EXEMPT = {
        "src/cleft/config/schema.py",     # COHORT_PATH_MARKERS itself
        "tests/test_smoke_run.py",        # this guard
    }
    # frozen tree: editing it would break tests/test_frozen_apparatus.py.
    # [2026-09-05] The three dev configs that used to sit here are GONE
    # from this list -- they declare their inputs now, so they pass the
    # guard rather than being excused from it.
    EXEMPT_PREFIXES = ("src/cleft/provenance/",)
    EXEMPT_TASK_FIELDS = ()

    offenders = []
    for name in tracked:
        if not name or name in EXEMPT:
            continue
        if name.startswith(EXEMPT_PREFIXES) or name in EXEMPT_TASK_FIELDS:
            continue
        path = repo_root / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.split("\n"), start=1):
            if "COHORT_EXEMPT" in line:
                continue
            for marker in COHORT_PATH_MARKERS:
                if marker in line:
                    offenders.append(f"{name}:{number} [{marker}] {line.strip()[:70]}")

    assert not offenders, (
        f"{len(offenders)} tracked line(s) name the cohort by path.\n  "
        + "\n  ".join(offenders[:30])
        + "\n\nDeclare the artifact by role instead -- see "
        "cleft.config.schema.COHORT_ENV_REFERENCES."
    )


def test_the_task_field_exception_closed_and_stayed_closed(repo_root):
    """**[REWRITTEN 2026-09-05] This used to pin what was still OPEN.**

    Four task-field lines across three dev configs carried the cohort
    path where ``expand_input_paths`` could not reach, and this test
    asserted there were exactly four so a fifth could not join quietly.
    They are closed: the artifacts are declared, the task fields name
    them, and the rollups are carried from ``p1_build_manifest.yaml``.

    So the test inverts. It now pins that the three configs DECLARE what
    they read, with real hashes, and that the record says the exception
    closed rather than still standing.
    """
    import yaml

    from cleft.config.schema import COHORT_PATHS_STILL_OPEN, COHORT_PATH_MARKERS

    was_open = (
        "configs/p1_inspect_scoresheet.yaml",
        "configs/p1_inspect_scoresheet_text.yaml",
        "configs/p1_scan_folders.yaml",
    )
    for name in was_open:
        body = (repo_root / name).read_text(encoding="utf-8")
        # No marker survives anywhere in the file, comments included.
        for number, line in enumerate(body.split("\n"), start=1):
            for marker in COHORT_PATH_MARKERS:
                assert marker not in line, f"{name}:{number} {line.strip()[:60]}"

        payload = yaml.safe_load(body)
        assert payload["inputs"], f"{name} declares nothing it reads"
        for entry in payload["inputs"]:
            assert entry["path"].startswith("${CLEFT_"), entry["name"]
            assert set(entry["rollup_sha256"]) != {"0"}, (
                f"{name}: {entry['name']} is still a placeholder"
            )

    # The record reports what REMAINS, not what remained.
    assert "task_fields_CLOSED_2026_09_05" in COHORT_PATHS_STILL_OPEN
    closed = " ".join(
        COHORT_PATHS_STILL_OPEN["and_the_second_defect_it_hid_CLOSED_2026_09_05"].split()
    )
    assert "CLOSED, and it was the worse half" in closed
    assert "read on the strength of a path alone" in closed

    # The frozen-tree exception is the only one left standing.
    still_open = [
        key for key in COHORT_PATHS_STILL_OPEN if "CLOSED" not in key
    ]
    assert still_open == ["frozen_tree", "what_replaced_the_reasoning"], still_open


def test_every_cohort_input_is_declared_by_role(repo_root):
    """The positive half. Every ``${CLEFT_*}`` reference a committed
    config uses must be one this project defines, and every cohort role
    must actually be used -- a variable nothing refers to is a fiction,
    and an undefined one is a typo that fails only on the cluster."""
    import re

    import yaml

    from cleft.config.schema import COHORT_ENV_REFERENCES, ENV_REFERENCE

    used = set()
    for path in sorted((repo_root / "configs").glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        for entry in payload.get("inputs") or []:
            match = ENV_REFERENCE.match(entry.get("path") or "")
            if match:
                used.add(match.group(1))

    unknown = {
        name for name in used
        if name not in COHORT_ENV_REFERENCES
        and not name.endswith("_ROOT")  # the dataset roots predate this
    }
    assert not unknown, f"configs reference undefined CLEFT vars: {sorted(unknown)}"

    unused = set(COHORT_ENV_REFERENCES) - used
    assert not unused, (
        f"cohort roles defined but never declared: {sorted(unused)} -- "
        "a variable nothing refers to is a fiction"
    )

    # Each role says what the artifact IS, never where it lives.
    for name, description in COHORT_ENV_REFERENCES.items():
        assert not any(c in description for c in ("/", "\\")), name


def test_the_portability_rule_still_catches_the_real_thing():
    """**The detector must reject what it was built for.** A rule that passes
    everything has not been shown to test anything -- the same reason the
    `latest`-tag and version-drift detectors carry self-checks in this file."""
    from cleft.config.schema import PORTABLE_PATH_KINDS, classify_declared_path

    # The exact declaration that was committed, and its neighbours.
    for unportable in (
        "C:/data/scut-fbp5500/SCUT-FBP5500_v2",
        "C:\\data\\scut-fbp5500\\SCUT-FBP5500_v2",
        "D:/data",
        "~/data/scut",
        "%USERPROFILE%/data",
    ):
        kind = classify_declared_path(unportable)
        assert kind not in PORTABLE_PATH_KINDS, f"{unportable} classified {kind}"

    for portable in (
        "data/smoke/v1",
        "/home/user/codex/scut/SCUT-FBP5500_v2",
        "/data/cohort/238 Cropped Anonymised Photos",
        "${CLEFT_SCUT_ROOT}",
        "${CLEFT_SCUT_ROOT}/Images",
    ):
        kind = classify_declared_path(portable)
        assert kind in PORTABLE_PATH_KINDS, f"{portable} classified {kind}"


def test_a_malformed_variable_reference_is_refused_not_read_as_a_subdirectory():
    """`$CLEFT_SCUT_ROOT` without braces, or a name outside the namespace, must
    fail loudly. Classified as repo-relative it would be concatenated onto the
    repo root and reported as missing data -- the original defect's shape."""
    from cleft.config.schema import ConfigError, classify_declared_path

    for malformed in (
        "$CLEFT_SCUT_ROOT",
        "${SCUT_ROOT}",
        "${cleft_scut_root}",
        "${CLEFT_SCUT_ROOT}extra",
        "prefix/${CLEFT_SCUT_ROOT}",
    ):
        with pytest.raises(ConfigError, match="variable reference"):
            classify_declared_path(malformed)


def test_no_config_declares_a_kind_the_schema_does_not_know(repo_root):
    """The other direction: a config naming an unknown kind would fail only at
    launch, on the cluster, after the queue wait."""
    import yaml

    from cleft.config.schema import TASK_SPECS

    for path in sorted((repo_root / "configs").glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        kind = (payload.get("task") or {}).get("kind")
        assert kind in TASK_SPECS, f"{path.name} declares unknown kind {kind!r}"


def test_the_schema_and_the_task_registry_agree():
    """The same defect class one layer down.

    A kind in the schema with no handler in ``TASKS`` validates and then dies
    with a KeyError at dispatch; a handler with no schema entry cannot be
    configured at all. Both are only discoverable by launching, which on this
    project means a cluster round trip.
    """
    from cleft.config.schema import TASK_SPECS
    from cleft.run import TASKS

    assert sorted(TASK_SPECS) == sorted(TASKS), (
        f"schema-only kinds: {sorted(set(TASK_SPECS) - set(TASKS))}; "
        f"handler-only kinds: {sorted(set(TASKS) - set(TASK_SPECS))}"
    )


def test_every_shipped_config_is_valid_and_its_declared_hashes_match(
    repo_root, monkeypatch
):
    """Every config in configs/ must load and its inputs must hash as declared.

    This is what makes a tracked config worth having: a config that only exists
    in /tmp cannot be checked, cannot be reproduced, and cannot be pointed at
    when someone asks what produced a result.

    ``${CLEFT_*}`` inputs need their variable set merely to LOAD, because
    ``load_config`` refuses to leave one unresolved -- an unexpanded reference
    would fall through to ``resolve_declared_path`` as a repo-relative path and be
    concatenated onto the repo root, which is the defect the indirection removes.
    So a deliberately fake absolute value is supplied here: this test checks that
    declarations are WELL FORMED, and it already cannot check hashes for data that
    is not on this machine. Guard 3 enforces the hash where the data is.
    """
    from cleft.config import load_config

    shipped = sorted((repo_root / "configs").glob("*.yaml"))
    assert shipped, "no configs shipped"

    placeholder = "0" * 64
    _sentinel_every_env_root(repo_root, monkeypatch, "config-validation")

    for path in shipped:
        cfg = load_config(path)
        for entry in cfg["inputs"]:
            # NOT Path(...).is_absolute(): on Windows a POSIX path like
            # /home/user/codex/bch/... has no drive letter and reports False, so
            # a cluster input would be mistaken for a repo-relative one. The
            # cluster is Linux and the check has to agree with it from here.
            declared_path = PurePosixPath(entry["path"])

            if declared_path.is_absolute() or PureWindowsPath(entry["path"]).drive:
                # A cluster-data input. The clinical cohort is not on this
                # machine and must never be, so its hash cannot be checked here
                # -- only that the declaration is well formed. The hash is
                # enforced on the cluster by guard 3, which is where the data is.
                assert len(entry["rollup_sha256"]) == 64
                continue

            # A repo-shipped fixture. This one is checkable, so it is checked.
            resolved = repo_root / Path(entry["path"])
            assert resolved.is_dir(), f"{path.name}: input missing at {entry['path']}"
            assert entry["rollup_sha256"] != placeholder, (
                f"{path.name}: input {entry['name']!r} ships with a placeholder "
                "hash but its data is in the repo, so it can and must be declared"
            )
            assert hash_dir(resolved)["rollup"] == entry["rollup_sha256"], (
                f"{path.name}: input {entry['name']!r} no longer matches its declared hash"
            )


def test_a_keeper_config_is_tracked_in_the_repo(repo_root):
    """The keeper path must be exercisable without writing a config in /tmp."""
    from cleft.config import load_config

    path = repo_root / "configs" / "smoke_keeper.yaml"
    assert path.is_file(), "configs/smoke_keeper.yaml is missing"
    assert load_config(path)["tier"] == "keeper"


def test_the_shipped_smoke_config_is_valid_and_declares_its_input(repo_root):
    """The template itself must load and must hash-declare its data."""
    from cleft.config import load_config

    cfg = load_config(repo_root / "configs" / "smoke.yaml")
    assert cfg["task"]["kind"] == "smoke"
    assert cfg["tier"] == "dev", "the shipped template must not default to keeper"
    assert len(cfg["inputs"]) == 1
    entry = cfg["inputs"][0]
    assert len(entry["rollup_sha256"]) == 64

    declared = repo_root / entry["path"]
    assert declared.is_dir(), f"shipped smoke input missing: {entry['path']}"
    assert hash_dir(declared)["rollup"] == entry["rollup_sha256"], (
        "the shipped smoke input no longer matches its declared hash"
    )


# --------------------------------------------------------------------------
# declared hashes: the invariant that does not go stale
# --------------------------------------------------------------------------


PLACEHOLDER_SHA = "0" * 64


def _hash_disagreements(rows):
    """Paths declared with more than one distinct rollup, as
    ``{path: {rollup: [config, ...]}}``.

    A pure function over ``(config, input, path, rollup)`` rows, so the
    invariant it expresses can be falsified against synthetic declarations
    rather than only observed to be green on the real ones. A sweep that
    passes because nothing disagrees today says nothing about whether it
    WOULD catch a disagreement.
    """
    by_path: dict[str, dict[str, list[str]]] = {}
    for config_name, _input, path, rollup in rows:
        by_path.setdefault(path, {}).setdefault(rollup, []).append(config_name)
    return {path: seen for path, seen in by_path.items() if len(seen) > 1}


def test_the_hash_disagreement_detector_has_teeth():
    """**The case it was built for, and the case it caught.**

    On 2026-08-01 the five per-fold rollups were filled into one arm config
    and not its sibling; this is that shape, in miniature. Also asserted: it
    stays silent when both are placeholders and when both are filled, because
    a detector that fires on the unbuilt state would make every new config
    fail until its artifact existed.
    """
    path = "/data/embeddings_adabn_v1/fold0"
    both_placeholder = [
        ("cell_c.yaml", "emb", path, PLACEHOLDER_SHA),
        ("cell_d.yaml", "emb", path, PLACEHOLDER_SHA),
    ]
    both_filled = [
        ("cell_c.yaml", "emb", path, "128edb" + "0" * 58),
        ("cell_d.yaml", "emb", path, "128edb" + "0" * 58),
    ]
    one_filled = [
        ("cell_c.yaml", "emb", path, PLACEHOLDER_SHA),
        ("cell_d.yaml", "emb", path, "128edb" + "0" * 58),
    ]
    mistyped = [
        ("cell_c.yaml", "emb", path, "128edb" + "0" * 58),
        ("cell_d.yaml", "emb", path, "128edc" + "0" * 58),
    ]

    assert not _hash_disagreements(both_placeholder), (
        "an unbuilt artifact declared consistently is the correct state"
    )
    assert not _hash_disagreements(both_filled)

    caught = _hash_disagreements(one_filled)
    assert list(caught) == [path] and len(caught[path]) == 2, (
        "the real 2026-08-01 drift -- one config filled, its sibling not"
    )
    assert _hash_disagreements(mistyped), "a mistyped digest must also be caught"


from conftest import sentinel_every_env_root as _sentinel_every_env_root


def _declared_inputs(repo_root, monkeypatch):
    """(config name, input name, path, rollup) for every shipped declaration."""
    from cleft.config import load_config

    _sentinel_every_env_root(repo_root, monkeypatch, "hash-sweep")
    rows = []
    for path in sorted((repo_root / "configs").glob("*.yaml")):
        for entry in load_config(path)["inputs"]:
            rows.append(
                (path.name, entry["name"], entry["path"], entry["rollup_sha256"])
            )
    assert rows, "the sweep found no declared inputs, so it checks nothing"
    return rows


def test_no_declared_run_directory_contradicts_itself(repo_root, monkeypatch):
    """**The run-directory name carries the identity twice; the halves must
    agree.**

    [FOUND 2026-08-12] ``roadb_p7_arm_swin_b_masked_224__30cfbfe2__roadb-
    p7-arm-swin-b-masked-768`` -- a job launched under the 768 id that ran
    the 224 config. Nothing in the apparatus compared the two halves, so it
    was caught by eye, and a 224 result filed as the missing 768 point
    would have entered a family whose whole claim is its shape
    (``roadb.MISTYPED_LAUNCH_DELETED``).

    **This is the declaration-time guard.** ``harness.py`` creates run
    directories and is frozen, and the damage lands when a directory is
    DECLARED -- which is here, over every shipped config, before any hash
    is trusted.
    """
    from cleft.run_names import check_run_dir, RunNameError

    offenders, checked = [], 0
    for config_name, input_name, path, _rollup in _declared_inputs(
        repo_root, monkeypatch
    ):
        if "/runs/" not in path:
            continue
        for part in path.split("/"):
            if part.count("__") != 2:
                continue
            checked += 1
            try:
                check_run_dir(part)
            except RunNameError as exc:
                offenders.append(f"{config_name}:{input_name}: {exc}")

    assert not offenders, (
        f"{len(offenders)} declared run directories contradict themselves: "
        + "; ".join(offenders[:3])
    )
    # Not vacuous: the repo really does declare run directories, and this
    # walked them. A silent zero here would be the check covering nothing.
    assert checked > 100, f"only {checked} run directories swept"


def test_the_run_name_guard_permits_abbreviation_and_catches_the_real_one():
    """**Measured before it was written**: the obvious rule ("job id is the
    stem hyphenated") holds for 22 of 83 declared directories and fails for
    61, because job ids legitimately abbreviate. A guard that fires on 61
    legitimate names gets switched off, so the rule is subset, not equality
    -- a job id may DROP what the stem says and may not CONTRADICT it."""
    from cleft.run_names import (
        RunNameError, check_run_dir, job_id_contradictions, parse_run_dir,
    )

    # The real one, with the axis and both readings named.
    deleted = (
        "roadb_p7_arm_swin_b_masked_224__30cfbfe2__"
        "roadb-p7-arm-swin-b-masked-768"
    )
    found = job_id_contradictions(deleted)
    assert found == [{
        "axis": "resolution",
        "job_id_says": ["768"],
        "config_stem_says": ["224"],
    }]
    with pytest.raises(RunNameError, match="contradicts itself"):
        check_run_dir(deleted)

    # Abbreviation, attempt suffixes and variant markers all pass -- these
    # are real directory names this repo declares.
    for legitimate in (
        "p6_pretrain_swin_b_masked_g1__5e927402__p6-pt-swin-g1-v2",
        "p7_d1_vit_b16_imagenet_g1__3f71a6a9__p7-d1-vit-imagenet",
        "p7_c_swin_b_scut_masked_g1__4cb62c05__p7-c-swin-g1",
        "roadb_p6_pretrain_vit_b16_masked_g1_512__5b784842__"
        "roadb-p6-vit-512-unfused-2",
        "p7_e_srgnn_scut_masked_g2_random__1234abcd__p7-e-srgnn-random-2",
        "p7c_2_geometric__deadbeef__p7c-arm2-geometric-v3",
    ):
        check_run_dir(legitimate)

    # A backbone swap is caught for the same reason a resolution swap is.
    assert job_id_contradictions(
        "roadb_p7_arm_srgnn_masked_512__a54cdfae__roadb-p7-arm-agnet-masked-512"
    )[0]["axis"] == "backbone"

    # And a name that does not carry the contract RAISES rather than being
    # skipped -- a skipped name is how the check comes to cover nothing.
    for malformed in ("no-separators", "a__b", "stem__nothex1__job", "__x__y"):
        with pytest.raises(RunNameError):
            parse_run_dir(malformed)


def test_the_same_artifact_declares_the_same_hash_in_every_config(
    repo_root, monkeypatch
):
    """**The invariant that replaces a fourth placeholder flip.**

    Data artifacts are immutable and versioned (PLAN §2.6), so a PATH
    determines its contents exactly. Therefore every config declaring that
    path must declare the same rollup -- and this is true while the artifact
    is unbuilt (all placeholders), true once it exists (all real), and false
    only in between, which is precisely the drift worth catching: one config
    filled from ``declare_inputs.py`` and its sibling forgotten.

    **It never needs inverting.** The three flips so far -- the seed band's
    embeddings hash, the xception marker, the per-fold sets -- were each a
    test asserting a STATE, so each was correct exactly once. This asserts a
    RELATION between configs, which holds in both states.

    What it cannot do is verify a hash against the cluster artifact; nothing
    on the laptop can. It makes every config sharing a path a witness for the
    others, which is the strongest laptop-side check available, and guard 3
    recomputes the real thing at launch.
    """
    shared = _hash_disagreements(_declared_inputs(repo_root, monkeypatch))
    assert not shared, "\n".join(
        [f"{len(shared)} artifact path(s) declared with disagreeing hashes:"]
        + [
            f"  {path}\n"
            + "\n".join(
                f"    {'PLACEHOLDER' if h == PLACEHOLDER_SHA else h[:12]} "
                f"<- {', '.join(sorted(names))}"
                for h, names in sorted(seen.items())
            )
            for path, seen in sorted(shared.items())
        ]
        + [
            "A path is immutable, so its contents cannot differ. Either a hash "
            "was filled in one config and not its sibling, or one is mistyped."
        ]
    )

    by_path: dict[str, dict[str, list[str]]] = {}
    for config_name, _input, path, rollup in _declared_inputs(repo_root, monkeypatch):
        by_path.setdefault(path, {}).setdefault(rollup, []).append(config_name)

    # And the sweep must actually be exercising shared paths, or its silence
    # means nothing.
    reused = [path for path, seen in by_path.items() if sum(map(len, seen.values())) > 1]
    assert reused, (
        "no artifact path is declared by more than one config, so this "
        "cross-check compared nothing"
    )


def test_a_placeholder_hash_is_always_documented_in_its_config(
    repo_root, monkeypatch
):
    """A placeholder must never be silent.

    An all-zeros rollup means "this artifact does not exist yet, and guard 3
    will refuse the run" -- which is the placeholder working. But a reader
    meeting it with no explanation cannot tell that from a mistake, and the
    next person to see the run refuse may simply fill in whatever hash makes
    it start.

    **One-directional, so it never goes stale**: having placeholders implies
    documenting them; once they are filled the requirement is vacuous and the
    comment may stay or go. That is the difference between this and the three
    tests that had to be inverted.
    """
    offenders, seen_placeholders = [], 0
    for config_name, input_name, _path, rollup in _declared_inputs(
        repo_root, monkeypatch
    ):
        if rollup != PLACEHOLDER_SHA:
            continue
        seen_placeholders += 1
        text = (repo_root / "configs" / config_name).read_text(encoding="utf-8")
        if not _placeholder_is_documented(text):
            offenders.append(f"{config_name}:{input_name}")

    assert not offenders, (
        f"all-zeros hashes with no explanation in the config: {offenders}. "
        "Say what artifact is missing and what produces it, or a reader "
        "cannot distinguish a deliberate placeholder from a mistake."
    )
    # **Vacuous today, and that is the design rather than a gap.** Every
    # shipped config's hashes are currently real, so the loop body never runs.
    # The predicate is exercised directly by the teeth test below, so its
    # silence here is not the only evidence it works.
    assert seen_placeholders == 0 or offenders == []


def _placeholder_is_documented(config_text: str) -> bool:
    """Does this config explain its all-zeros hashes?"""
    return "PLACEHOLDER" in config_text.upper()


def test_the_placeholder_documentation_rule_has_teeth():
    """The rule is one-directional -- placeholders imply documentation -- so on
    a fully-filled repo it is vacuous. That makes a green sweep no evidence at
    all, which is failure mode 5 in PLAN R7's tally. The predicate is
    therefore tested against text directly."""
    assert _placeholder_is_documented(
        "# HASHES ARE PLACEHOLDERS UNTIL THE RE-EXTRACTION RUNS\ninputs: []"
    )
    assert _placeholder_is_documented("# the placeholder is working as intended")
    assert not _placeholder_is_documented(
        "# a config that declares all-zeros and says nothing about why"
    )


def _task_keys_read(source: str) -> set:
    """Every ``task["literal"]`` key a task subscripts, COMMENTS AND
    DOCSTRINGS EXCLUDED.

    Subscripts only: ``task.get(...)`` is optional by construction and
    cannot raise. Comments are stripped because a task that EXPLAINS a
    renamed key would otherwise be reported as reading it -- the
    word-matching trap, which bit the first version of this very
    helper.
    """
    import ast

    tree = ast.parse(source.strip())
    keys = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "task"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            keys.add(node.slice.value)
    return keys


def test_every_shipped_config_carries_every_key_its_task_subscripts(
    repo_root, monkeypatch
):
    """**The cluster must not be the first place a task reads its own
    config** (phase15.KEY_MISMATCH_DIAGNOSED).

    p15-stage-mebeauty-v2 failed five times on ``KeyError:
    'geometries'`` -- the config had been renamed to ``variants`` and
    one read in the task had not. Schema validation passed, because the
    schema validates what the config DECLARES, not what the task READS.
    Nothing on the laptop connected the two, so the mismatch travelled
    to the cluster.

    This closes that: for every shipped config, resolve its task and
    assert that every key the task SUBSCRIPTS is present. A subscript
    is a promise the key exists; ``.get`` is not, and is excluded.
    """
    import inspect

    from cleft.config import load_config
    from cleft.run import TASKS

    _sentinel_every_env_root(repo_root, monkeypatch, "task-key-sweep")
    missing, checked = [], 0
    for path in sorted((repo_root / "configs").glob("*.yaml")):
        task_block = load_config(path)["task"]
        kind = task_block["kind"]
        function = TASKS.get(kind)
        if function is None:
            continue
        checked += 1
        for key in _task_keys_read(inspect.getsource(function)):
            if key not in task_block:
                missing.append(f"{path.name}: task reads {key!r}, config has no such key")

    assert checked, "no shipped config resolved to a task; this checks nothing"
    assert not missing, (
        "a task subscripts a key its config does not declare -- a "
        "cluster-side KeyError waiting to happen:\n  "
        + "\n  ".join(missing)
    )


def test_no_task_field_carries_a_filesystem_path(repo_root):
    """**[ADDED 2026-09-05] The guard for the class, not the instance.**

    A task field is either a scientific setting or the NAME of a declared
    input. A field carrying a PATH is the third thing, and it is the shape
    that slips both checks at once: ``expand_input_paths`` resolves
    ``inputs[].path`` and nothing else, so a path under ``task:`` is never
    parameterised; and guard 3 hashes declared inputs, so a path under
    ``task:`` is never verified either.

    That is how ``p1_inspect_scoresheet.yaml`` came to read the workbook
    every label in this project derives from, on the strength of a path
    alone, for the life of the repository. The config even carried a note
    explaining why the path lived under ``task:`` -- the placement was
    deliberate and the reasoning was wrong, which is exactly the kind of
    thing only a mechanical check catches.

    Stated as a rule about SHAPE rather than a list of field names, so a
    new task field cannot reintroduce the class under a name nobody
    thought to add.
    """
    import yaml

    offenders = []
    for path in sorted((repo_root / "configs").glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        task = (payload or {}).get("task") or {}
        for key, value in task.items():
            if not isinstance(value, str):
                continue
            looks_like_a_path = (
                "/" in value
                or "\\" in value
                or value.startswith("${")
                or (len(value) > 1 and value[1] == ":")   # drive letter
            )
            if looks_like_a_path:
                offenders.append(f"{path.name}: task.{key} = {value[:60]}")

    assert not offenders, (
        "task fields carrying a filesystem path:\n  "
        + "\n  ".join(offenders)
        + "\n\nA task field names a DECLARED INPUT; the path belongs in "
        "inputs[].path, where expand_input_paths parameterises it and the "
        "rollup_sha256 verifies it. See run.declared_input."
    )


def test_every_task_field_that_names_an_input_names_a_declared_one(repo_root):
    """The positive half: a name that resolves to nothing fails on the
    cluster, at the far end of a queue. It is free to check here."""
    import yaml

    # Fields whose VALUE is an input name. Read off the configs rather
    # than hand-listed: any task value that exactly matches a declared
    # input name in the same config is treated as a reference.
    checked = 0
    for path in sorted((repo_root / "configs").glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        task = (payload or {}).get("task") or {}
        names = {e["name"] for e in (payload or {}).get("inputs") or []}
        for key in ("scoresheet", "compare_with", "patient_folders",
                    "scut_root", "mebeauty_root"):
            value = task.get(key)
            if not isinstance(value, str):
                continue
            checked += 1
            assert value in names, (
                f"{path.name}: task.{key} = {value!r} names no declared "
                f"input. Declared: {sorted(names)}"
            )
    assert checked >= 60, (
        f"only {checked} input-naming task fields found; the sweep that "
        "found this class counted 63, so this check has gone blind"
    )


# --------------------------------------------------------------------------
# one-factor families, checked over the UNION rather than over a list
# --------------------------------------------------------------------------

#: The families whose arms the record claims are one-factor against a
#: reference config. Each entry names the reference, the arms, and the
#: fields the family is DECLARED to vary.
ONE_FACTOR_FAMILIES = {
    "p25": (
        "p7_d1_vit_b16_imagenet_g1",
        ("p25_arm_d1", "p25_arm_d2"),
        {"backbone", "init"},
    ),
    "p7d": (
        "p7_d1_vit_b16_imagenet_g1",
        ("p7d_arm_vit_b32", "p7d_arm_vit_b8", "p7d_arm_mvitv2_b",
         "p7d_arm_vit_b32_512", "p7d_arm_concat_multiscale"),
        {"backbone", "init", "resolution", "pooling", "patch_scheme"},
    ),
}

#: Keys that name WHICH artifact or WHAT the task is, rather than HOW the
#: fit runs. Every arm necessarily differs on these.
_STRUCTURAL = {
    "kind", "manifest_artifact", "staged_artifact", "embeddings_artifact",
    "concat_embeddings_artifacts", "out_version", "expect_patients",
    "checkpoint", "checkpoint_path", "run_dirs", "arms", "csv",
    "feature_source", "consume", "permutation",
}

#: **[RECORDED 2026-09-06] The one divergence the union check finds, and
#: it is exempted BY NAME with its record rather than by widening the
#: rule.** Phase 25's arms omit batch_size and resolve to the schema
#: default 16 against the probe's 32. It is INERT, because the head is
#: fit full batch and batch_size never reaches EmbeddingHeadBackbone.
#: The values are left as they ran, so the configs keep describing the
#: runs that produced the banked figures.
_RECORDED_DIVERGENCES = {
    ("p25_arm_d1", "batch_size"),
    ("p25_arm_d2", "batch_size"),
}


def _task(repo_root, stem, monkeypatch):
    from cleft.config import load_config

    _sentinel_every_env_root(repo_root, monkeypatch, "one-factor-union")
    return load_config(repo_root / "configs" / f"{stem}.yaml")["task"]


def test_one_factor_arms_match_their_reference_over_the_union(
    repo_root, monkeypatch
):
    """**[ADDED 2026-09-06] The check the Phase 25 one-factor claim
    needed and did not have.**

    The generator compares every COPIED field against the reference, so a
    field that was never added to the copied list is invisible to it. The
    check that catches that compares over the UNION of both task blocks,
    where absence is a value like any other.

    The one divergence this finds is recorded, dated and exempted by
    name at ``_RECORDED_DIVERGENCES``, with its inertness measured at
    ``phase25.PHASE_25_CLOSING`` criterion 2. Widening the rule to hide
    it would rebuild the defect.
    """
    offenders = []
    for family, (reference, arms, declared) in ONE_FACTOR_FAMILIES.items():
        ref = _task(repo_root, reference, monkeypatch)
        for stem in arms:
            task = _task(repo_root, stem, monkeypatch)
            if task.get("kind") != ref.get("kind"):
                continue  # a different task has no shared recipe
            union = (set(ref) | set(task)) - _STRUCTURAL - declared
            for field in sorted(union):
                if (stem, field) in _RECORDED_DIVERGENCES:
                    continue
                here = ref.get(field, "<ABSENT>")
                there = task.get(field, "<ABSENT>")
                if here != there:
                    offenders.append(
                        f"{family}/{stem}: {field} = {there!r}, "
                        f"reference {reference} = {here!r}"
                    )

    assert not offenders, (
        "one-factor arms diverge from their reference on a field nobody "
        "declared as the factor:\n  " + "\n  ".join(offenders)
    )


def test_every_recorded_divergence_is_still_real_and_still_recorded(
    repo_root, monkeypatch
):
    """An exemption that has silently become unnecessary is a stale rule.
    This fails if a recorded divergence is repaired without the record
    following it."""
    from cleft import phase25

    ref = _task(repo_root, "p7_d1_vit_b16_imagenet_g1", monkeypatch)
    for stem, field in sorted(_RECORDED_DIVERGENCES):
        task = _task(repo_root, stem, monkeypatch)
        assert task.get(field) != ref.get(field), (
            f"{stem}.{field} now matches the reference. Remove it from "
            "_RECORDED_DIVERGENCES and update phase25.PHASE_25_CLOSING."
        )

    walk = phase25.PHASE_25_CLOSING["criterion_walk"]
    assert "2_CORRECTED_2026_09_06_what_the_divergence_actually_IS" in walk
    stated = " ".join(
        walk["2_CORRECTED_2026_09_06_what_the_divergence_actually_IS"].split()
    )
    assert "batch size 16 against the probe's 32" in stated
    inert = " ".join(
        walk["2_CORRECTED_2026_09_06_and_it_is_INERT_on_this_path"].split()
    )
    assert "Batch size is never passed to it" in inert
    assert "none changes" in inert


def test_an_omitted_field_is_only_safe_when_the_default_matches(repo_root):
    """**The latent form, and it is the one worth having.**

    Three generators omit fields from their copied list and are safe
    only because the schema default happens to equal the reference's
    value. That coincidence is not a design. This asserts it, so an
    omission becomes a failure the moment a default moves.
    """
    import importlib.util

    from cleft.config import load_config
    from cleft.config.schema import TASK_SPECS

    generators = {
        "generate_phase7d_configs.py": "p7_d1_vit_b16_imagenet_g1",
        "generate_phase8_configs.py": "p7_d1_vit_b16_imagenet_g1",
        "generate_phase9_configs.py": "p7_d1_vit_b16_imagenet_g1",
        "generate_phase25_configs.py": "p7_d1_vit_b16_imagenet_g1",
    }
    # The one field a generator omits whose default does NOT match, kept
    # by name with its record rather than by relaxing the rule.
    known = {("generate_phase25_configs.py", "batch_size")}

    offenders = []
    for filename, reference in generators.items():
        path = repo_root / "scripts" / filename
        spec = importlib.util.spec_from_file_location(f"gen_{filename}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        copied = set(getattr(module, "COPIED", ()))
        assert copied, filename

        ref = load_config(repo_root / "configs" / f"{reference}.yaml")["task"]
        spec_fields = TASK_SPECS[ref["kind"]]
        # A field the family DECLARES as its factor is not an omission.
        declared = {"backbone", "init", "resolution", "pooling",
                    "patch_scheme"}
        for field in sorted(set(ref) - copied - _STRUCTURAL - declared):
            if (filename, field) in known:
                continue
            if field not in spec_fields:
                continue
            default = spec_fields[field].default
            if default != ref[field]:
                offenders.append(
                    f"{filename}: {field} is not copied and its schema "
                    f"default {default!r} differs from the reference's "
                    f"{ref[field]!r}"
                )

    assert not offenders, (
        "a field omitted from a copied list resolves to something the "
        "reference did not run at:\n  " + "\n  ".join(offenders)
    )
