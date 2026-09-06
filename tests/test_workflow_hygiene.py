"""Regression test for the root cause of the previous pipeline's failure.

Evidence, from the old repository:

  * ``Dockerfile:40``  ``COPY . .``                     - the image baked the code in
  * ``.github/workflows/docker.yml``  ``paths: '**.py'`` - so every edit rebuilt gigabytes
  * pushed ``:latest`` alongside a SHA tag             - so a pending job could pick up new code

The chain that followed: slow builds, avoided commits, ``dirty_tree: true`` on
every run ever produced, and PCC 0.218 vs 0.150 for the same nominal arm with no
way to separate code drift from non-determinism.

This is the cheapest test in the suite and it guards the most expensive failure
the project has had. It reads files rather than importing anything, so it cannot
be disabled by a broken environment.
"""

from __future__ import annotations

import ast


def _run_context_attributes(repo_root) -> set:
    """Every name a task may reach through ``ctx``.

    ``dir(RunContext)`` alone is not it: that gives methods and class
    attributes, and the fields tasks use most -- ``config``, ``inputs``,
    ``repo_root``, ``run_dir`` -- are assigned on the instance in
    ``__init__``. A check built on ``dir`` alone would flag every task in the
    file, which is how a check ends up disabled instead of fixed.
    """
    from cleft.provenance.context import RunContext

    names = set(dir(RunContext))
    source = (
        repo_root / "src" / "cleft" / "provenance" / "context.py"
    ).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef) or node.name != "RunContext":
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Attribute)
                and isinstance(inner.value, ast.Name)
                and inner.value.id == "self"
                and isinstance(inner.ctx, ast.Store)
            ):
                names.add(inner.attr)
    return names


def test_every_key_phase3_reads_from_backbone_config_is_supplied_by_the_task(
    repo_root,
):
    """**[MEASURED 2026-08-03] ``backbone_config`` is a WHITELIST, and a key
    read but never supplied is silent.**

    ``phase3.prepare_features`` and ``make_factory`` both read
    ``config.get("augmentation")``; ``task_train_cv`` built the dict from five
    named keys and that was not one. So every Phase 7C arm ran unaugmented,
    all seven produced byte-identical predictions, and arm 0's gate passed
    because everything passed it.

    ``config.get(key)`` returning ``None`` is indistinguishable from "this arm
    does not use it", which is why nothing raised. The general form of the
    defect is a read with no writer, and that is what this checks -- statically,
    across the whole file, so the NEXT key added to phase3 fails here rather
    than on the cluster.

    Scope is ``task_train_cv`` deliberately: it is the task that reaches
    ``phase3.run`` for the arms this guards, and a union over every task would
    be satisfiable by any one of them supplying a key.
    """
    import re

    phase3_source = (
        repo_root / "src" / "cleft" / "train" / "phase3.py"
    ).read_text(encoding="utf-8")
    reads = set(re.findall(r'config\.get\(\s*"([a-z_]+)"', phase3_source))
    assert reads, "no config.get(...) found in phase3; the sweep checks nothing"

    run_source = (repo_root / "src" / "cleft" / "run.py").read_text(encoding="utf-8")
    tree = ast.parse(run_source)
    supplied = None
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "task_train_cv"):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.keyword)
                and inner.arg == "backbone_config"
                and isinstance(inner.value, ast.Dict)
            ):
                supplied = {
                    key.value for key in inner.value.keys
                    if isinstance(key, ast.Constant)
                }
    assert supplied, "task_train_cv no longer builds backbone_config as a literal"

    # [2026-09-01] A key whose ABSENCE is the designed state for every
    # arm this guard covers, exempted BY NAME with its reason. NOT a
    # union over tasks -- the docstring rejects that, and rightly: a
    # union is satisfiable by any one task. Each entry here is a key
    # where `None` means "this arm is not that kind of arm", not "the
    # feature is switched off".
    ABSENCE_IS_THE_DESIGNED_STATE = {
        # phase22.AXIS_RULED: the ranking objective travels in
        # backbone_config so train_cv's closed vocabulary stays
        # untouched. Absent means "not a ranking arm", which is the
        # correct state for every ladder arm -- and the branch it gates
        # then reads `bounded` by SUBSCRIPT, so a half-specified
        # ranking arm raises instead of defaulting.
        "objective",
    }
    missing = sorted(reads - supplied - ABSENCE_IS_THE_DESIGNED_STATE)
    assert not missing, (
        f"phase3 reads {missing} from backbone_config and task_train_cv never "
        f"supplies them. config.get() returns None silently, so the arm runs "
        f"with the feature switched off and reports a plausible number. "
        f"Supplied: {sorted(supplied)}"
    )


#: Task fields that name a DECLARED INPUT rather than carrying a value.
#: The convention is ``*_artifact``; these are the ones that do not
#: follow it, found by sweeping every config rather than by memory.
INPUT_REFERENCE_KEYS = ("artifact", "synth_set", "input")


def _input_references(node, declared, trail=""):
    """Every (field-path, value) in a task that names an input."""
    found = []
    if isinstance(node, dict):
        for raw_key, value in node.items():
            key = str(raw_key)
            if isinstance(value, (dict, list)):
                found += _input_references(value, declared, f"{trail}{key}.")
            elif isinstance(value, str) and (
                key.endswith("_artifact") or key in INPUT_REFERENCE_KEYS
            ):
                found.append((f"{trail}{key}", value))
    elif isinstance(node, list):
        for item in node:
            found += _input_references(item, declared, trail)
    return found


def test_every_by_name_input_reference_resolves_to_a_declared_input(repo_root):
    """**[2026-09-01, WRITTEN AFTER A BREAK IT WOULD HAVE CAUGHT.]**

    Six Phase 22 arm configs validated, ``--check`` passed, and 3,072
    tests passed while all six were unloadable: their task blocks named
    ``staged_g1`` and ``pooled_vit_b16_imagenet_g1``, neither of which
    is a declared input in any config. The failure was a bare
    ``KeyError`` at ``declared[task["embeddings_artifact"]]``, on the
    cluster, after the launch.

    **Nothing rejected it because the schema validates the TYPE of these
    fields and never the REFERENCE.** ``Field(str)`` is satisfied by any
    string, so a name that exists nowhere passes every check the config
    layer has.

    **The same class as Phase 17's writer/reader filename break**: two
    halves written in one cycle, each internally consistent, disagreeing
    at the join. The join is what nothing was testing.

    Scope is every config and every by-name reference, not the one field
    that broke -- so the NEXT invented name fails here rather than on the
    cluster.
    """
    import yaml

    configs = sorted((repo_root / "configs").glob("*.yaml"))
    assert len(configs) > 300, "the sweep should cover the whole directory"

    total, broken = 0, []
    for path in configs:
        try:
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError:  # pragma: no cover - malformed configs
            continue
        if not isinstance(config, dict) or "task" not in config:
            continue
        declared = {entry["name"] for entry in (config.get("inputs") or [])}
        for field, value in _input_references(config["task"], declared):
            total += 1
            if value not in declared:
                broken.append(
                    f"{path.name}: task.{field} = {value!r} names no declared "
                    f"input (declared: {sorted(declared)})"
                )

    assert total > 800, f"only {total} references swept; the check is thin"
    assert not broken, (
        f"{len(broken)} by-name input reference(s) resolve to nothing. Each "
        "is a run that validates, passes --check, and dies with a KeyError "
        "at load:\n  " + "\n  ".join(broken)
    )


def test_the_config_layer_now_refuses_an_unresolvable_reference(repo_root):
    """The sweep above is static; this is the load-time guard, so a
    HAND-WRITTEN config fails the same way a generated one does.

    Before 2026-09-01 this validated cleanly and died later as a
    ``KeyError`` inside the task.
    """
    import pytest
    import yaml

    from cleft.config.schema import ConfigError, validate

    raw = yaml.safe_load(
        (repo_root / "configs" / "p22_r_all_bounded.yaml").read_text(
            encoding="utf-8"
        )
    )
    # The good config still validates.
    validate(dict(raw))

    for field, invented in (
        ("embeddings_artifact", "pooled_vit_b16_imagenet_g1"),
        ("staged_artifact", "staged_g1"),
        ("manifest_artifact", "manifest_that_never_was"),
    ):
        broken = yaml.safe_load(
            (repo_root / "configs" / "p22_r_all_bounded.yaml").read_text(
                encoding="utf-8"
            )
        )
        broken["task"][field] = invented
        with pytest.raises(ConfigError, match="names no declared input"):
            validate(broken)


def test_the_ranking_arms_carry_the_donors_inputs_byte_for_byte(repo_root):
    """'No new hash enters' -- verified, not restated."""
    import yaml

    donor = yaml.safe_load(
        (repo_root / "configs" / "p7_d1_vit_b16_imagenet_g1.yaml")
        .read_text(encoding="utf-8")
    )
    for path in sorted((repo_root / "configs").glob("p22_r_*.yaml")):
        arm = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert arm["inputs"] == donor["inputs"], path.name
        # And the REFERENCES are the donor's too, which is the fix: the
        # two halves are carried from one place, so they cannot drift.
        for field in ("manifest_artifact", "staged_artifact",
                      "embeddings_artifact"):
            assert arm["task"][field] == donor["task"][field], (
                path.name, field
            )


def test_the_ranking_task_supplies_every_key_its_own_branch_reads(repo_root):
    """[2026-09-01] The same guard, for the second task that reaches
    ``phase3.run``.

    ``task_train_rank_cv`` is exempted from the sweep above only for
    ``objective``; it must still supply everything the ranking branch
    of ``make_factory`` reads, or a Phase 22 arm runs with a setting
    silently defaulted -- which is the exact failure the original guard
    was written for, in a new place.
    """
    import re

    phase3_source = (
        repo_root / "src" / "cleft" / "train" / "phase3.py"
    ).read_text(encoding="utf-8")
    branch = phase3_source.split('config.get("objective")')[1].split(
        "if backbone =="
    )[0]
    reads = set(re.findall(r'config\.get\(\s*"([a-z_]+)"', branch))
    subscripts = set(re.findall(r'config\[\s*"([a-z_]+)"\s*\]', branch))
    assert reads and subscripts, "the ranking branch reads nothing"

    run_source = (repo_root / "src" / "cleft" / "run.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(run_source)
    supplied = None
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.FunctionDef)
            and node.name == "task_train_rank_cv"
        ):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Assign)
                and isinstance(inner.value, ast.Dict)
                and any(
                    isinstance(t, ast.Name) and t.id == "backbone_config"
                    for t in inner.targets
                )
            ):
                supplied = {
                    key.value for key in inner.value.keys
                    if isinstance(key, ast.Constant)
                }
    assert supplied, "task_train_rank_cv no longer builds backbone_config"

    missing = sorted((reads | subscripts) - supplied)
    assert not missing, (
        f"the ranking branch reads {missing} and task_train_rank_cv never "
        f"supplies them. Supplied: {sorted(supplied)}"
    )
    # `objective` and `bounded` are the two that decide WHICH ARM this
    # is, so both must be present.
    assert {"objective", "bounded"} <= supplied


def test_every_schema_backbone_choice_is_constructible_by_its_factory():
    """**Two lists of the same thing, required to agree -- by CONSTRUCTION,
    not by comparison.**

    [FOUND 2026-08-14] ``srgnn`` was added to the schema's ``train_cv``
    backbone choices and not to ``phase3.BACKBONES``. The config validated
    and the run died at ``make_factory`` after submission -- a config value
    a validator accepts with no builder behind it, which is the
    config-read-with-no-writer sweep one layer over.

    Comparing the tuples catches one direction. CONSTRUCTING each choice
    catches both: a name present in the registry tuple but missing from
    ``make_factory``'s branches falls through to its raise, and this fires
    on that too.
    """
    from cleft.config.schema import TASK_SPECS
    from cleft.train import graph_cleft, phase3

    cv_choices = set(TASK_SPECS["train_cv"]["backbone"].choices)
    assert cv_choices == set(phase3.BACKBONES), (
        "the schema's train_cv backbone choices and phase3.BACKBONES "
        f"disagree: schema-only {sorted(cv_choices - set(phase3.BACKBONES))}, "
        f"registry-only {sorted(set(phase3.BACKBONES) - cv_choices)}"
    )
    for backbone in sorted(cv_choices):
        assert callable(phase3.make_factory(backbone, "head", {}, seed=0)), (
            f"{backbone} is an accepted config value that make_factory "
            "cannot build -- the run would die after submission"
        )

    graph_choices = set(TASK_SPECS["train_graph_cv"]["backbone"].choices)
    assert graph_choices == set(graph_cleft.GRAPH_CLEFT_BACKBONES), (
        "the schema's train_graph_cv backbone choices and "
        "graph_cleft.GRAPH_CLEFT_BACKBONES disagree: schema-only "
        f"{sorted(graph_choices - set(graph_cleft.GRAPH_CLEFT_BACKBONES))}, "
        "registry-only "
        f"{sorted(set(graph_cleft.GRAPH_CLEFT_BACKBONES) - graph_choices)}"
    )

    # srgnn is HEAD-ONLY here: a trained SR-GNN is the graph task's arm,
    # so asking for one through train_cv must refuse rather than build the
    # wrong model under the right name.
    with pytest.raises(phase3.Phase3Error, match="train_graph_cv"):
        phase3.make_factory("srgnn", "full", {}, seed=0)


def test_the_backbone_config_check_has_teeth():
    """It is silent on a correct file, so the predicate is exercised against a
    read with no writer -- the exact shape of the Phase 7C defect."""
    reads = {"learning_rate", "augmentation"}
    supplied = {"learning_rate", "weight_decay"}
    assert sorted(reads - supplied) == ["augmentation"]


def test_every_task_calls_only_attributes_RunContext_actually_has(repo_root):
    """**[MEASURED 2026-08-02] A task called ``ctx.write_metrics``, which does
    not exist, and found out on the cluster after 24 trials had run.**

    The name appeared exactly once in ``run.py``; every other task writes its
    summary through ``ctx.atomic``. Nothing caught it because nothing
    exercised that handler -- ``phase7b.py`` had thirty tests of its functions
    and the task wiring them had none.

    An end-to-end test now covers that one task, but a test per task only ever
    closes the tasks somebody remembered to write a test for. **This closes
    the class**: every ``ctx.<name>`` in every handler is checked against the
    real ``RunContext``, statically, so a misnamed context call fails on the
    laptop in a second rather than after the compute.

    Static because it has to be: a runtime check would need each task to be
    driven with real inputs, which is exactly the thing that was missing.
    """
    available = _run_context_attributes(repo_root)

    source = (repo_root / "src" / "cleft" / "run.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    offenders, checked, tasks = [], 0, 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("task_"):
            continue
        tasks += 1
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Attribute)
                and isinstance(inner.value, ast.Name)
                and inner.value.id == "ctx"
            ):
                checked += 1
                if inner.attr not in available:
                    offenders.append(
                        f"{node.name}:{inner.lineno} -> ctx.{inner.attr}"
                    )

    # A sweep that walked nothing is green for the wrong reason -- failure
    # mode 5 in PLAN R7's tally, so the guard is on the input.
    assert tasks > 10, f"only {tasks} task handlers found; the walk is not covering run.py"
    assert checked > 50, f"only {checked} ctx.* uses found; the walk is not reaching them"
    assert not offenders, (
        f"tasks call attributes RunContext does not have: {offenders}. "
        f"Available: {sorted(n for n in available if not n.startswith('_'))}"
    )


def test_the_context_attribute_check_has_teeth(repo_root):
    """The sweep above is silent on a correct file, so the predicate is
    exercised against a handler that calls a method the class lacks."""
    tree = ast.parse(
        "def task_bogus(ctx):\n"
        "    ctx.log('fine')\n"
        "    ctx.config['task']\n"
        "    ctx.write_metrics({'a': 1})\n"
    )
    available = _run_context_attributes(repo_root)
    found = [
        inner.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("task_")
        for inner in ast.walk(node)
        if isinstance(inner, ast.Attribute)
        and isinstance(inner.value, ast.Name)
        and inner.value.id == "ctx"
        and inner.attr not in available
    ]
    assert found == ["write_metrics"], (
        "the predicate must flag exactly the attribute RunContext lacks"
    )
    # And the instance fields tasks actually use are recognised, or the check
    # would flag every handler in the file and get disabled rather than fixed.
    for name in ("log", "atomic", "path", "config", "inputs", "repo_root", "run_dir"):
        assert name in available, f"{name} is not recognised as a ctx attribute"

import re
from functools import lru_cache
from pathlib import Path

import pytest
import yaml

from fixtures import pins

REPO = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO / "docker" / "Dockerfile"
WORKFLOWS = REPO / ".github" / "workflows"

#: Runner labels, not image references. ``ubuntu-latest`` has a hyphen, not a
#: colon, so the ``:latest`` scan does not see it — these are belt and braces.
RUNNER_LABELS = {"ubuntu-latest", "macos-latest", "windows-latest"}

TEXT_SUFFIXES = {
    ".py", ".yaml", ".yml", ".sh", ".toml", ".txt", ".md", ".cfg", ".ini", ".json",
}
SKIP_DIRS = {".git", "runs", "data", ".pytest_cache", "__pycache__", ".venv", "node_modules"}


@lru_cache(maxsize=1)
def _text_files() -> tuple[Path, ...]:
    """Cached: three tests scan the whole tree, and it does not change mid-run."""
    out = []
    for path in REPO.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(REPO).parts[:-1]):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name == "Dockerfile":
            out.append(path)
    return tuple(sorted(out))


def _dockerfile_instructions(text: str) -> list[tuple[int, str, str]]:
    """Return (lineno, instruction, arguments), joining backslash continuations."""
    out: list[tuple[int, str, str]] = []
    buffer = ""
    start = 0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not buffer:
            if not line or line.startswith("#"):
                continue
            start = lineno
        if line.endswith("\\"):
            buffer += line[:-1] + " "
            continue
        buffer += line
        head, _, rest = buffer.strip().partition(" ")
        out.append((start, head.upper(), rest.strip()))
        buffer = ""
    return out


# --------------------------------------------------------------------------
# the image must never contain application code
# --------------------------------------------------------------------------


def test_dockerfile_exists():
    assert DOCKERFILE.is_file(), f"expected {DOCKERFILE.relative_to(REPO)}"


def test_dockerfile_does_not_contain_copy_dot_dot():
    """The literal line that caused Chain 1."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert not re.match(r"^COPY\s+\.\s+\.\s*$", stripped, re.IGNORECASE), (
            f"docker/Dockerfile:{lineno} bakes the code into the image. "
            "This is the exact root cause of the previous pipeline's failure."
        )


def test_every_dockerfile_copy_source_is_requirements_txt():
    instructions = _dockerfile_instructions(DOCKERFILE.read_text(encoding="utf-8"))
    for lineno, verb, args in instructions:
        if verb != "COPY":
            continue
        tokens = [t for t in args.split() if not t.startswith("--")]
        assert len(tokens) >= 2, f"docker/Dockerfile:{lineno} malformed COPY"
        sources = tokens[:-1]
        for source in sources:
            assert Path(source).name == "requirements.txt", (
                f"docker/Dockerfile:{lineno} copies {source!r} into the image. "
                "The image is environment only: requirements.txt and nothing else. "
                "Code reaches the cluster by git pull, not by rebuild."
            )


def test_dockerfile_has_no_add_instruction():
    """ADD is COPY with remote fetch and tar expansion. Not needed, not allowed."""
    verbs = [v for _, v, _ in _dockerfile_instructions(DOCKERFILE.read_text(encoding="utf-8"))]
    assert "ADD" not in verbs


def test_dockerfile_base_image_is_pinned_by_digest():
    """A tag can move under us silently; a digest cannot."""
    froms = [
        args
        for _, verb, args in _dockerfile_instructions(DOCKERFILE.read_text(encoding="utf-8"))
        if verb == "FROM"
    ]
    assert froms, "no FROM instruction found"
    for args in froms:
        image = args.split()[0]
        assert re.search(r"@sha256:[0-9a-f]{64}$", image), (
            f"base image {image!r} is referenced by tag, not by digest"
        )


def test_dockerfile_enforces_the_python_declaration_in_build():
    """The interpreter check must be a build layer, not a CI step afterwards.

    As a build layer, an image whose python disagrees with the declaration cannot
    be built. As a CI step it could only stop the push -- and it needed
    ``load: true`` to inspect the image, which exhausted the runner's disk
    exporting a multi-GB torch image and re-importing it.
    """
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "/tmp/requirements.txt" in text, "the Dockerfile must copy requirements.txt in"
    assert "sys.version_info" in text, (
        "docker/Dockerfile must measure the interpreter during the build"
    )
    assert "python:" in text or "python_declaration_status" in text or "re.search" in text, (
        "docker/Dockerfile must read the declared version out of requirements.txt"
    )
    assert "sys.exit" in text, (
        "the interpreter check must fail the build, not just print"
    )


def test_no_workflow_loads_the_image_into_the_daemon():
    """``load: true`` exports and re-imports gigabytes and hit the disk ceiling.

    Nothing needs the image in the local daemon: the checks that used to require
    it now run inside the build.
    """
    for path in _workflow_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert not re.match(r"^load:\s*true\b", stripped), (
                f"{path.name}:{lineno} sets load: true. The interpreter check lives "
                "in the Dockerfile now, so nothing needs the image loaded, and "
                "loading a multi-GB torch image fills the runner's disk."
            )


def test_the_image_is_built_and_pushed_in_one_step():
    """Two build steps meant two chances to diverge, and doubled the disk churn."""
    data = _load_workflow(WORKFLOWS / "build-image.yml")
    build_steps = [
        step
        for step in data["jobs"]["build"]["steps"]
        if "docker/build-push-action" in str(step.get("uses", ""))
    ]
    assert len(build_steps) == 1, (
        f"expected exactly one build step, found {len(build_steps)}"
    )


def test_dockerfile_declares_no_default_training_command():
    """The old CMD ran train.py from inside the image. Entry is via the repo."""
    for _, verb, args in _dockerfile_instructions(DOCKERFILE.read_text(encoding="utf-8")):
        if verb in {"CMD", "ENTRYPOINT"}:
            assert "train" not in args.lower(), (
                f"{verb} {args!r} runs application code from the image"
            )


# --------------------------------------------------------------------------
# no workflow may rebuild the image on a code change
# --------------------------------------------------------------------------


def _load_workflow(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # YAML 1.1 parses the bare key ``on`` as the boolean True.
    triggers = data.get("on", data.get(True))
    data["_triggers"] = triggers if isinstance(triggers, dict) else {}
    if isinstance(triggers, str):
        data["_triggers"] = {triggers: None}
    elif isinstance(triggers, list):
        data["_triggers"] = {name: None for name in triggers}
    return data


def _builds_an_image(path: Path, data: dict) -> bool:
    text = path.read_text(encoding="utf-8").lower()
    return "docker/build-push-action" in text or "docker build" in text


def _workflow_files() -> list[Path]:
    return sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))


def test_at_least_one_workflow_exists():
    assert _workflow_files(), "no CI workflows found"


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_image_building_workflow_is_manual_only(path: Path):
    data = _load_workflow(path)
    if not _builds_an_image(path, data):
        return

    triggers = data["_triggers"]
    assert "push" not in triggers, (
        f"{path.name} rebuilds the image on push. Image builds are a separate "
        "manual workflow, triggered only when requirements.txt changes."
    )
    assert "pull_request" not in triggers
    assert "workflow_dispatch" in triggers, (
        f"{path.name} builds an image but has no manual trigger"
    )

    flat = yaml.safe_dump(data)
    assert "**.py" not in flat and "'**.py'" not in flat, (
        f"{path.name} triggers an image build on python changes"
    )


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_test_workflow_does_not_build_an_image(path: Path):
    data = _load_workflow(path)
    triggers = data["_triggers"]
    if "push" in triggers or "pull_request" in triggers:
        assert not _builds_an_image(path, data), (
            f"{path.name} runs on push/PR and builds an image. CI runs pytest; "
            "it does not build images."
        )


# --------------------------------------------------------------------------
# no mutable image reference anywhere
# --------------------------------------------------------------------------


#: An actual image reference: a name character immediately before the colon.
#: Prose and comments that merely mention the tag (``No ':latest'``, or a
#: markdown ``:latest``) have a quote or a backtick there instead, so they do not
#: match. Forbidding the bare string would make it impossible to write down why
#: it is forbidden.
LATEST_IMAGE_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-/]*:latest\b")


def test_no_file_references_a_latest_image_tag():
    offenders = []
    for path in _text_files():
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            if not LATEST_IMAGE_REF.search(line):
                continue
            if any(label in line for label in RUNNER_LABELS):
                continue
            if line.strip().startswith("#"):
                continue
            rel = path.relative_to(REPO).as_posix()
            # This test file names the pattern in order to forbid it.
            if rel == "tests/test_workflow_hygiene.py":
                continue
            offenders.append(f"{rel}:{lineno}: {line.strip()}")

    assert not offenders, (
        "mutable 'latest' image reference found. A pending Run:AI job can pull "
        f"new code mid-experiment: {offenders}"
    )


#: A digest is 64 lowercase hex characters. Anything shorter is a copy-paste from
#: a summary or a chat log and is not a usable image reference.
FULL_DIGEST = re.compile(r"@?sha256:([0-9a-f]{64})\b")
#: Only the image-reference form ``@sha256:``. A bare ``sha256:`` also appears in
#: ``rollup_sha256`` (a data-artifact hash, a different thing entirely, already
#: length-checked by the config schema) and in test string concatenation.
ANY_IMAGE_DIGEST = re.compile(r"@sha256:([0-9a-zA-Z]*)")

#: The publish target in build-image.yml is necessarily a tag -- you cannot push
#: to a digest, the registry computes it. Every other image reference must be a
#: digest.
PUBLISH_TAG_FILE = "build-image.yml"


def test_no_truncated_image_digest_anywhere():
    """``sha256:2135e27b…`` is not a reference, and docs/PLAN.md writes it that way.

    A truncated digest copied into a config or a workload spec fails at pull time
    with an obscure error, or worse, is silently 'fixed' by someone re-adding a
    tag.
    """
    offenders = []
    for path in _text_files():
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith("docs/") or rel == "tests/test_workflow_hygiene.py":
            # docs/ holds verbatim copies of the governing documents and briefs,
            # which legitimately abbreviate digests in prose. They are never read
            # by a machine.
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            for match in ANY_IMAGE_DIGEST.finditer(line):
                if len(match.group(1)) != 64:
                    offenders.append(f"{rel}:{lineno}: {line.strip()[:100]}")

    assert not offenders, (
        "truncated or malformed sha256 digest. A digest is exactly 64 hex "
        f"characters: {offenders}"
    )


def test_our_own_image_is_only_ever_referenced_by_digest():
    """Same rule as the base image, applied to the image we build.

    A mutable reference lets a queued Run:AI job pull a different environment
    mid-experiment, and then the arm is not comparable with the ones before it.
    """
    image_with_tag = re.compile(r"[A-Za-z0-9._-]+/cleft-aesthetics:[A-Za-z0-9._${}-]+")
    offenders = []
    for path in _text_files():
        rel = path.relative_to(REPO).as_posix()
        if rel == f".github/workflows/{PUBLISH_TAG_FILE}" or rel.startswith("docs/"):
            continue
        if rel == "tests/test_workflow_hygiene.py":
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            if line.strip().startswith("#"):
                continue
            if image_with_tag.search(line) and "@sha256:" not in line:
                offenders.append(f"{rel}:{lineno}: {line.strip()[:100]}")

    assert not offenders, (
        "the project image referenced by tag rather than by digest. Only "
        f"{PUBLISH_TAG_FILE} may name a tag, because that is the push target: "
        f"{offenders}"
    )


def test_the_readme_records_the_pinned_digest():
    """The digest existed only in a chat log until it was written down here."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    digests = FULL_DIGEST.findall(readme)
    assert digests, "README.md records no full image digest"
    assert any(
        f"cleft-aesthetics@sha256:{digest}" in readme for digest in digests
    ), "README.md must pin the project image by full digest, not only the base image"


def test_the_latest_detector_still_catches_the_real_thing():
    """The detector was loosened to allow prose; prove it did not go toothless.

    The first line is the exact tag the old repository pushed alongside its SHA
    tag, which is what let a queued job pick up different code.
    """
    must_flag = [
        "            ${{ secrets.DOCKERHUB_USERNAME }}/cleft-training:latest",
        "image: redring/cleft-training:latest",
        'FROM pytorch/pytorch:latest',
    ]
    for line in must_flag:
        assert LATEST_IMAGE_REF.search(line), line

    must_ignore = [
        "    runs-on: ubuntu-latest",
        "- any file references a `:latest` image tag",
        "# Exactly one immutable tag. No ':latest' -- a mutable tag is unsafe.",
    ]
    for line in must_ignore:
        flagged = bool(LATEST_IMAGE_REF.search(line)) and not any(
            label in line for label in RUNNER_LABELS
        )
        assert not flagged, line


# --------------------------------------------------------------------------
# the declared python version
# --------------------------------------------------------------------------


def test_requirements_declares_the_python_version_explicitly():
    """Read off a running container, not inferred from the base image tag."""
    requirements = REPO / "docker" / "requirements.txt"
    assert requirements.is_file()
    match = re.search(
        r"^#\s*python:\s*(\d+\.\d+\.\d+)\s*$",
        requirements.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match, (
        "docker/requirements.txt must declare '# python: X.Y.Z', the version "
        "measured inside the image with `python -VV`"
    )


def test_git_does_not_rewrite_line_endings():
    """A data artifact must hash identically on the laptop and on the cluster.

    Without this, git converts LF <-> CRLF on checkout, the rollup hash of a
    committed artifact differs by platform, and guard 3 aborts every cluster run
    with a mismatch that has nothing to do with the data.
    ``test_smoke_run.py::test_the_shipped_smoke_config_is_valid_and_declares_its_input``
    is what would actually fail on Linux CI; this names the cause.
    """
    attributes = REPO / ".gitattributes"
    assert attributes.is_file(), "no .gitattributes: line endings are unprotected"
    lines = [
        line.split("#")[0].strip()
        for line in attributes.read_text(encoding="utf-8").splitlines()
    ]
    assert "* -text" in lines, (
        ".gitattributes must contain '* -text' so artifact bytes survive checkout"
    )
    assert any(line.startswith("*.sh") and "eol=lf" in line for line in lines), (
        "shell scripts must be pinned to LF or they fail on the cluster"
    )


def test_python_declaration_status_is_recorded():
    """Whether the version was measured or guessed is itself a recorded fact.

    Phase 0 ships UNVERIFIED because the image has not been built yet. The value
    is carried into env.json either way, so no result can silently rest on a
    guess, and the in-container guard turns a wrong guess into an abort.
    """
    from cleft.provenance import DECLARED_PYTHON_STATUS

    assert DECLARED_PYTHON_STATUS in {"MEASURED", "UNVERIFIED"}


def test_declared_python_matches_the_provenance_module():
    """One source of truth: the module reads the declaration, never a copy of it."""
    from cleft.provenance.context import DECLARED_PYTHON

    requirements = (REPO / "docker" / "requirements.txt").read_text(encoding="utf-8")
    match = re.search(r"^#\s*python:\s*(\d+\.\d+\.\d+)\s*$", requirements, re.MULTILINE)
    assert DECLARED_PYTHON == match.group(1)


def test_numpy_is_pinned_below_2():
    """A real guard, not incidental: the old image asserted it at build time."""
    requirements = (REPO / "docker" / "requirements.txt").read_text(encoding="utf-8")
    match = re.search(r"^numpy==(\d+)\.", requirements, re.MULTILINE)
    assert match, "numpy must be pinned exactly"
    assert match.group(1) == "1", "numpy must stay below 2"


# One parser, shared with test_environment, so "the lists agree with each other"
# and "the environment agrees with the lists" cannot diverge for parsing reasons.
_normalise = pins.normalise
_pinned = pins.pinned
_test_extra = pins.test_extra


def test_ci_and_image_agree_on_shared_versions():
    """CI must not test against a different numpy from the one the cluster runs.

    A result that reproduces in CI and not in the image, or the reverse, costs
    more to diagnose than this test costs to keep.
    """
    extra = _test_extra()
    image = _pinned((REPO / "docker" / "requirements.txt").read_text(encoding="utf-8").splitlines())

    shared = sorted(set(extra) & set(image))
    assert shared, "no packages in common; one of the two lists is not being parsed"

    mismatched = {
        name: (extra[name], image[name]) for name in shared if extra[name] != image[name]
    }
    assert not mismatched, (
        "CI and the image disagree on a pinned version "
        f"(package: test-extra vs requirements.txt): {mismatched}"
    )


def test_the_drift_detector_still_catches_the_real_thing():
    """Prove the comparison is not toothless, the way the latest-tag check is.

    The numpy pair below is real: the first cluster keeper run reported numpy
    1.24.4 from the workspace against the image's pinned 1.26.4.
    """
    extra = {"numpy": "1.24.4", "pyyaml": "6.0.1"}
    image = {"numpy": "1.26.4", "pyyaml": "6.0.1"}
    shared = set(extra) & set(image)
    mismatched = {n: (extra[n], image[n]) for n in shared if extra[n] != image[n]}
    assert mismatched == {"numpy": ("1.24.4", "1.26.4")}

    # And that name normalisation does not let a rename hide a mismatch.
    assert _normalise("PyYAML") == _normalise("pyyaml") == "pyyaml"
    assert _normalise("scikit_learn") == "scikit-learn"
    assert _pinned(['    "numpy==1.26.4",', "scipy==1.13.1", "# pytest==9"]) == {
        "numpy": "1.26.4",
        "scipy": "1.13.1",
    }


def test_pyproject_is_valid_toml_and_declares_the_test_extra():
    """The raw-text parser above must not mask a file that no longer parses."""
    import tomllib

    with (REPO / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)

    extras = data["project"]["optional-dependencies"]
    assert "test" in extras, "no 'test' extra declared"
    assert _pinned(extras["test"]) == _test_extra(), (
        "the raw-text parser and tomllib disagree about the test extra"
    )


def test_every_test_dependency_is_pinned():
    extra_block = re.search(
        r"^test\s*=\s*\[(.*?)\]",
        (REPO / "pyproject.toml").read_text(encoding="utf-8"),
        re.MULTILINE | re.DOTALL,
    )
    entries = [
        line.split("#")[0].strip().strip('",\'')
        for line in extra_block.group(1).splitlines()
    ]
    entries = [entry for entry in entries if entry]
    assert entries, "the test extra is empty"
    unpinned = [entry for entry in entries if "==" not in entry]
    assert not unpinned, f"unpinned test dependency: {unpinned}"


def test_ci_does_not_install_torch():
    """torch in CI is what would push the suite past a minute."""
    extra = _test_extra()
    assert "torch" not in extra
    assert "torchvision" not in extra


def test_ci_installs_the_test_extra_rather_than_a_hand_picked_list():
    """A hand-picked list drifts, and guessing its minimum already broke a build.

    pytest loads tests/conftest.py whatever subset of files is selected, so any
    workflow running pytest needs the whole extra, not the packages someone
    thought that one test file imported.
    """
    for path in _workflow_files():
        text = path.read_text(encoding="utf-8")
        if "pytest" not in text:
            continue
        assert '".[test]"' in text or "'.[test]'" in text, (
            f"{path.name} runs pytest without installing the test extra"
        )
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("pip install") and "[test]" not in stripped:
                assert "-e" not in stripped, (
                    f"{path.name} hand-picks packages instead of the test extra: {stripped}"
                )


# --------------------------------------------------------------------------
# the runner scripts
# --------------------------------------------------------------------------


def test_runner_scripts_exist_for_both_shells():
    """Typing `pytest` must not be the easiest way to run the suite.

    The wrong environment should require deliberately bypassing a script, rather
    than being what you get by default. test_environment.py is the backstop for
    when someone bypasses it anyway.
    """
    for name in ("test.sh", "test.ps1"):
        path = REPO / "scripts" / name
        assert path.is_file(), f"scripts/{name} is missing"
        text = path.read_text(encoding="utf-8")
        assert "[test]" in text, f"scripts/{name} does not install the pinned extra"
        assert "venv" in text, f"scripts/{name} does not create a venv"


def test_runner_scripts_read_the_python_series_rather_than_hardcoding_it():
    """One source of truth: docker/requirements.txt declares the interpreter."""
    for name in ("test.sh", "test.ps1"):
        text = (REPO / "scripts" / name).read_text(encoding="utf-8")
        assert "requirements.txt" in text, (
            f"scripts/{name} must read the python series from docker/requirements.txt"
        )


def test_readme_documents_the_runner_script():
    """[UPDATED 2026-09-05] The README moved to the Windows-idiomatic
    ``.\\scripts\\test.ps1``, which is the form used everywhere else in
    this project. The separator is not the invariant -- that the runner
    is documented at all is -- so both are accepted and the assertion
    still fails if neither appears."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "scripts/test" in readme or "scripts\\test" in readme, (
        "the README must document the runner script"
    )
    # And it must document the extra that pins the numerical stack:
    # `pip install -e .` alone installs no pytest and floats scipy, so a
    # reader following the README verbatim gets a red suite.
    assert 'pip install -e ".[test]"' in readme, (
        "the README must document the [test] extra, not a bare editable "
        "install -- see tests/test_environment.py RECIPE"
    )


def test_tracked_shell_scripts_are_executable():
    """A shell script committed as 100644 fails on the cluster with 'Permission denied'.

    core.filemode is false on Windows, so the mode has to be set in the index
    explicitly; it will never be noticed by editing the file.
    """
    import subprocess

    try:
        listing = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "-s", "--", "*.sh"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("not a git checkout")

    non_executable = []
    for line in listing.splitlines():
        if not line.strip():
            continue
        mode, _, rest = line.partition(" ")
        name = rest.split("\t", 1)[-1]
        if mode != "100755":
            non_executable.append(f"{name} ({mode})")

    assert not non_executable, (
        "shell scripts committed without the executable bit: "
        f"{non_executable}\n"
        "Fix with:  git update-index --chmod=+x scripts/*.sh"
    )


def test_no_requirement_floats():
    requirements = (REPO / "docker" / "requirements.txt").read_text(encoding="utf-8")
    offenders = []
    for lineno, line in enumerate(requirements.splitlines(), start=1):
        stripped = line.split("#")[0].strip()
        if not stripped:
            continue
        if not re.search(r"(==|<|>)", stripped):
            offenders.append(f"{lineno}: {stripped}")
    assert not offenders, f"unpinned requirement: {offenders}"


def test_every_report_key_the_grad_cam_task_reads_is_one_something_produces(
    repo_root,
):
    """**[MEASURED 2026-08-08] Second stale reference to a removed field to
    reach the cluster in this phase.**

    ``PAD_FRACTION_AT_MEDIAN_AR`` was removed and the report's
    ``out_of_content_uniform`` key went with it, but a log line still read it.
    The run computed everything, wrote the sheet PNG, and died formatting the
    summary -- so the corrected numbers did not exist and the whole run had to
    be repeated. The first instance was ``backbone_config``'s missing
    ``augmentation`` key, guarded above.

    **Same shape: a read with no writer.** ``report[key]`` raises where
    ``config.get(key)`` returned None, so this one is loud rather than silent
    -- but it is loud on the cluster, after the compute, which is the same
    cost. Checked statically here so the NEXT key fails on the laptop.

    The report is about to grow again for Road B, which is why this is a sweep
    rather than a fix to the one line.
    """
    import re

    run_source = (repo_root / "src" / "cleft" / "run.py").read_text(encoding="utf-8")
    tree = ast.parse(run_source)
    task = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "task_grad_cam"
    )
    body = ast.get_source_segment(run_source, task) or ""

    # Read: report['key'] and report.get("key"). Subscripted reads are what
    # raise; .get is included so a silent None is caught too.
    read = set(re.findall(r"report\[[\"']([a-z0-9_]+)[\"']\]", body))
    read |= set(re.findall(r"report\.get\(\s*[\"']([a-z0-9_]+)[\"']", body))
    assert read, "no report keys are read; this sweep checks nothing"

    # Written: assignments in the task itself, plus everything summarise puts
    # in. summarise is the producer, so its keys are read out of its source
    # rather than by calling it -- the aggregate keys are conditional on the
    # data and calling it with a fixture would under-report them.
    written = set(re.findall(r"report\[[\"']([a-z0-9_]+)[\"']\]\s*=", body))
    sheet_source = (
        repo_root / "src" / "cleft" / "gradcam_sheet.py"
    ).read_text(encoding="utf-8")
    summarise = next(
        node for node in ast.walk(ast.parse(sheet_source))
        if isinstance(node, ast.FunctionDef) and node.name == "summarise"
    )
    produced = ast.get_source_segment(sheet_source, summarise) or ""
    written |= set(re.findall(r"summary\[[\"']([a-z0-9_]+)[\"']\]\s*=", produced))
    written |= set(re.findall(r"[\"']([a-z0-9_]+)[\"']\s*:", produced))
    # f-string keys built by .format(**...) inside summarise's literal dict.
    written |= {"caveat", "review_questions"}

    missing = sorted(read - written)
    assert not missing, (
        f"task_grad_cam reads report[{missing}] and nothing produces "
        "them. That is a read with no writer -- it raises on the cluster "
        "after the compute, which is where the last two went"
    )


# --------------------------------------------------------------------------
# input-entry keys: every key a task reads must be one the context produces
# --------------------------------------------------------------------------


def _input_entry_keys_produced(repo_root) -> set:
    """The keys ``_verify_inputs`` actually puts on a resolved entry.

    Parsed from the producer's own source rather than hardcoded, so this
    sweep cannot drift from it the way the gate drifted from it.
    """
    source = (
        repo_root / "src" / "cleft" / "provenance" / "context.py"
    ).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.FunctionDef) and node.name == "_verify_inputs"):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "append"
                and inner.args
                and isinstance(inner.args[0], ast.Dict)
            ):
                return {
                    key.value for key in inner.args[0].keys
                    if isinstance(key, ast.Constant)
                }
    raise AssertionError(
        "_verify_inputs no longer appends a dict literal; the producer "
        "moved and this sweep must follow it"
    )


def _input_entry_reads(function: ast.FunctionDef) -> list:
    """``(lineno, key)`` for every string key read from a ctx.inputs entry.

    Tracked from the binding site, not by variable NAME: tasks reuse
    ``entry`` for report rows and seed records, and a name-based sweep would
    flag those legitimate keys until someone disabled it. What is tracked:

    * loop and comprehension targets iterating ``ctx.inputs`` directly;
    * dicts built as ``{entry[k]: entry for entry in ctx.inputs}`` (and
      re-derivations of those via ``.items()``), whose VALUES are entries --
      then targets iterating their ``.items()``/``.values()``, possibly
      wrapped in ``sorted()``/``list()``, and direct ``d[...]["key"]`` reads.

    Reads are collected only inside the binding construct, because a
    comprehension's target does not leak and a later loop reusing the name
    binds something else.
    """

    def is_ctx_inputs(expr) -> bool:
        return (
            isinstance(expr, ast.Attribute) and expr.attr == "inputs"
            and isinstance(expr.value, ast.Name) and expr.value.id == "ctx"
        )

    def unwrap(expr):
        while (
            isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)
            and expr.func.id in ("sorted", "list") and expr.args
        ):
            expr = expr.args[0]
        return expr

    entry_dicts: set = set()

    def entry_dict_call(expr, methods) -> bool:
        expr = unwrap(expr)
        return (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Attribute)
            and expr.func.attr in methods
            and isinstance(expr.func.value, ast.Name)
            and expr.func.value.id in entry_dicts
        )

    def bound_entry_name(target, iterable):
        """The name this loop/generator binds to a full entry, if any."""
        source = unwrap(iterable)
        if is_ctx_inputs(source) and isinstance(target, ast.Name):
            return target.id
        if (
            entry_dict_call(iterable, ("items",))
            and isinstance(target, ast.Tuple) and len(target.elts) == 2
            and isinstance(target.elts[1], ast.Name)
        ):
            return target.elts[1].id
        if entry_dict_call(iterable, ("values",)) and isinstance(target, ast.Name):
            return target.id
        return None

    # Pass 1, to fixpoint: which assigned names are dicts OF entries.
    for _ in range(3):
        before = len(entry_dicts)
        for inner in ast.walk(function):
            if not (
                isinstance(inner, ast.Assign)
                and len(inner.targets) == 1
                and isinstance(inner.targets[0], ast.Name)
                and isinstance(inner.value, ast.DictComp)
            ):
                continue
            generator = inner.value.generators[0]
            name = bound_entry_name(generator.target, generator.iter)
            if name is not None and (
                isinstance(inner.value.value, ast.Name)
                and inner.value.value.id == name
            ):
                entry_dicts.add(inner.targets[0].id)
        if len(entry_dicts) == before:
            break

    # Pass 2: collect reads at each binding site, scoped to that construct.
    reads: list = []

    def collect(scope, name: str) -> None:
        for inner in ast.walk(scope):
            if (
                isinstance(inner, ast.Subscript)
                and isinstance(inner.value, ast.Name)
                and inner.value.id == name
                and isinstance(inner.slice, ast.Constant)
                and isinstance(inner.slice.value, str)
            ):
                reads.append((inner.lineno, inner.slice.value))
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr == "get"
                and isinstance(inner.func.value, ast.Name)
                and inner.func.value.id == name
                and inner.args
                and isinstance(inner.args[0], ast.Constant)
                and isinstance(inner.args[0].value, str)
            ):
                reads.append((inner.lineno, inner.args[0].value))

    for node in ast.walk(function):
        if isinstance(
            node, (ast.DictComp, ast.ListComp, ast.SetComp, ast.GeneratorExp)
        ):
            for generator in node.generators:
                name = bound_entry_name(generator.target, generator.iter)
                if name is not None:
                    collect(node, name)
        elif isinstance(node, ast.For):
            name = bound_entry_name(node.target, node.iter)
            if name is not None:
                collect(node, name)
        elif isinstance(node, ast.Subscript):
            # declared["values"]["path"] -- a key read one subscript deep.
            if (
                isinstance(node.value, ast.Subscript)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id in entry_dicts
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                reads.append((node.lineno, node.slice.value))
    return reads


def test_every_input_entry_key_a_task_reads_is_one_the_context_produces(
    repo_root,
):
    """**[MEASURED 2026-08-09] The residual gate read
    ``entry["rollup_sha256"]`` from ``ctx.inputs`` and died on the cluster
    with a KeyError.** That is the CONFIG's field name; ``_verify_inputs``
    resolves it into ``declared_rollup`` and ``rollup``. Guard 3 passed, the
    ten artifacts verified, and the run failed reading back the very hashes
    it had just checked.

    Same shape as the report-key sweep one object over: a read with no
    writer, loud on the cluster after the compute. The ``ctx.<name>`` sweep
    could not see it -- it checks attributes, and this is a KEY on the entry
    dicts the attribute yields. This closes the class for ctx.inputs, over
    run.py's tasks and roadb_tasks' handlers both, since Road B's remaining
    tasks will read the same structure.
    """
    produced = _input_entry_keys_produced(repo_root)
    assert "name" in produced and "path" in produced, (
        f"the producer parse looks wrong: {sorted(produced)}"
    )

    offenders, total_reads, functions = [], 0, 0
    for filename in ("run.py", "roadb_tasks.py"):
        source = (repo_root / "src" / "cleft" / filename).read_text(
            encoding="utf-8"
        )
        for node in ast.walk(ast.parse(source)):
            if not (
                isinstance(node, ast.FunctionDef)
                and node.args.args
                and node.args.args[0].arg == "ctx"
            ):
                continue
            functions += 1
            for lineno, key in _input_entry_reads(node):
                total_reads += 1
                if key not in produced:
                    offenders.append(
                        f"{filename}:{lineno} {node.name} reads "
                        f"entry[{key!r}]"
                    )

    # A sweep that walked nothing is green for the wrong reason.
    assert functions > 10, f"only {functions} ctx handlers found"
    assert total_reads > 20, f"only {total_reads} entry-key reads found"
    assert not offenders, (
        f"tasks read input-entry keys the context never produces: "
        f"{offenders}. _verify_inputs supplies {sorted(produced)}; "
        "rollup_sha256 is the CONFIG field, not the runtime one"
    )


def test_the_input_entry_key_check_has_teeth(repo_root):
    """Exercised against the exact defect idiom -- the gate's dict-of-entries
    then ``.items()`` loop -- and against a reused ``entry`` name that must
    NOT be flagged, because a sweep that cries wolf gets disabled."""
    tree = ast.parse(
        "def task_bogus(ctx):\n"
        "    declared = {entry['name']: entry for entry in ctx.inputs}\n"
        "    for name, entry in sorted(declared.items()):\n"
        "        record(entry['path'], entry['rollup_sha256'])\n"
        "    for entry in per_patient_report:\n"
        "        show(entry['patient_id'])\n"
    )
    function = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    )
    produced = _input_entry_keys_produced(repo_root)
    keys = {key for _, key in _input_entry_reads(function)}
    assert "rollup_sha256" in keys, "the defect idiom was not tracked"
    assert "path" in keys and "name" in keys
    assert "patient_id" not in keys, (
        "a reused loop-variable name was misattributed to ctx.inputs"
    )
    flagged = sorted(keys - produced)
    assert flagged == ["rollup_sha256"], flagged


def test_every_schema_init_choice_is_accepted_by_the_embedding_vocabulary():
    """**The backbone check, one field over -- and this one arrived the
    hard way.**

    [FOUND 2026-09-02, ON THE CLUSTER] ``dinov2_lvd142m`` was added to the
    schema's ``init`` choices and not to ``embeddings.INITS``. The config
    validated, the generator's ``--check`` passed, the suite passed, and
    ``task_extract_embeddings`` died at ``embeddings.expected_variant``
    before writing anything -- **a config value a validator accepts and
    the code refuses**, which is the sibling of
    ``test_every_schema_backbone_choice_is_constructible_by_its_factory``
    and of the srgnn defect that test was written for.

    The settings-consumption sweep does NOT catch this shape: the value IS
    consumed. It is consumed and REFUSED.
    """
    from cleft import embeddings
    from cleft.config.schema import TASK_SPECS

    # **Which task kinds route their init to expected_variant**, derived
    # from run.py rather than listed here, plus train_cv -- whose init
    # must name an artifact one of them produced.
    run_body = (REPO / "src" / "cleft" / "run.py").read_text(encoding="utf-8")
    routed = {"train_cv"}
    for chunk in run_body.split("\ndef task_")[1:]:
        name = chunk.split("(")[0]
        if "expected_variant(" in chunk:
            routed.add(name.removeprefix("task_"))
    assert "extract_embeddings" in routed, routed

    # **The one documented separate namespace**, named rather than
    # silently skipped: MEBeauty pretrains its own checkpoints and runs
    # its own extraction task, so its inits never reach expected_variant
    # ("MEBeauty's OWN init namespace, never the ladder's").
    SEPARATE_NAMESPACES = {
        "extract_mebeauty_embeddings": ("mebeauty_masked", "mebeauty_original"),
        "probe_mebeauty": ("mebeauty_masked", "mebeauty_original"),
    }

    offenders, unexplained = [], []
    for kind, spec in TASK_SPECS.items():
        field = spec.get("init")
        choices = getattr(field, "choices", None) if field else None
        for value in choices or ():
            if value in embeddings.ALL_INITS:
                continue
            if kind in routed:
                offenders.append((kind, value))
            elif value not in SEPARATE_NAMESPACES.get(kind, ()):
                unexplained.append((kind, value))
    assert not offenders, (
        "schema init choices the embedding vocabulary rejects, on a task "
        f"that CALLS expected_variant: {offenders}. "
        f"embeddings.ALL_INITS is {embeddings.ALL_INITS}."
    )
    assert not unexplained, (
        "init choices outside embeddings.ALL_INITS on tasks that do not "
        f"call expected_variant, and not a documented namespace: "
        f"{unexplained}. Either add them to the vocabulary or record the "
        "separate namespace, as MEBeauty's is recorded."
    )

    # Comparing the tuples catches one direction; CALLING the function
    # catches both, exactly as constructing each backbone does.
    for value in embeddings.ALL_INITS:
        for geometry in embeddings.GEOMETRIES:
            embeddings.expected_variant(value, geometry)

    # And the schema really does permit the two new ones where they run.
    extract_init = TASK_SPECS["extract_embeddings"]["sets"].item_spec["init"]
    assert set(embeddings.FOUNDATION_INITS) <= set(extract_init.choices)


def test_every_shipped_config_names_an_init_the_code_accepts():
    """The same check at the other end: a shipped config's VALUE, not just
    the schema's permitted set."""
    import yaml

    from cleft import embeddings

    seen = []
    for path in sorted((REPO / "configs").glob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            continue
        task = config.get("task") or {}
        if "mebeauty" in str(task.get("kind", "")):
            continue          # its own namespace; see the test above
        values = []
        if isinstance(task.get("init"), str):
            values.append(task["init"])
        for entry in task.get("sets") or ():
            if isinstance(entry, dict) and isinstance(entry.get("init"), str):
                values.append(entry["init"])
        for value in values:
            geometry = task.get("geometry")
            if geometry not in embeddings.GEOMETRIES:
                geometry = "g1"
            embeddings.expected_variant(value, geometry)
            seen.append((path.name, value))

    assert seen, "no config declares an init; the sweep found nothing to check"
    # The two Phase 25 configs are in it, which is what makes this a
    # regression test rather than a tautology.
    assert ("p25_extract_d2_dinov2.yaml", "dinov2_lvd142m") in seen
    assert ("p25_extract_d1_dino.yaml", "dino_in1k") in seen


def test_the_two_foundation_inits_were_outside_the_LADDER_vocabulary():
    """**The pre-fix state, pinned.** Both are legitimate members of the
    artifact vocabulary and neither is a ladder init -- which is the whole
    distinction the fix rests on. If a later edit collapses the two
    tuples, this fires."""
    from cleft import embeddings

    for value in embeddings.FOUNDATION_INITS:
        assert value not in embeddings.INITS, value
        assert value in embeddings.ALL_INITS, value
        # They consume no pretraining checkpoint of ours -- the same
        # answer imagenet gives, and a true one.
        assert embeddings.expected_variant(value, "g1") is None
        assert embeddings.expected_variant(value, "g2") is None


def test_adding_a_foundation_init_did_not_inflate_the_ladder_lattice():
    """**The reason the fix is not 'add a member to INITS'.**

    ``expected_set_count`` multiplies by ``len(INITS)`` and
    ``embedding_plan`` iterates it, so a fourth member would have turned
    the banked 24 sets into 40 -- the LADDER_BACKBONES defect in a
    different registry.
    """
    from cleft import embedding_plan, embeddings

    counts = embeddings.expected_set_count()
    assert counts["inits"] == 3
    assert counts["embedding_sets"] == 24
    assert counts["pretraining_runs"] == 12
    assert len(embeddings.INITS) == 3

    # The derived plan is unchanged too, and names no foundation init.
    plan_inits = {entry["init"] for entry in embedding_plan.required_sets()}
    assert plan_inits <= set(embeddings.INITS)
    assert not plan_inits & set(embeddings.FOUNDATION_INITS)


def test_the_factory_pretrained_list_is_derived_not_repeated():
    from cleft import embeddings
    from cleft.train.extract import FACTORY_PRETRAINED_INITS

    assert FACTORY_PRETRAINED_INITS == ("imagenet",) + embeddings.FOUNDATION_INITS
    body = (REPO / "src" / "cleft" / "train" / "extract.py").read_text(
        encoding="utf-8"
    )
    assert 'FACTORY_PRETRAINED_INITS = ("imagenet",) + _FOUNDATION_INITS' in body
    assert '"dinov2_lvd142m"' not in body      # no second literal copy


# --------------------------------------------------------------------------
# the README's own counts, 2026-09-06
# --------------------------------------------------------------------------
#
# **[ADDED 2026-09-06] Four counts on the front page were stale at once**:
# the ledger held 39 entries where the README said 38, 9 DESCRIPTIVE where
# it said 8, 3,450 test functions where it said 3,389, and the phase
# sequence had reached 27 where it said twenty-five. Two phases had closed
# without the README moving.
#
# **A project whose thesis is that a figure must be re-derived rather than
# recalled cannot carry recalled figures on its front page.** These pins
# are the same device this module already applies to a floating
# requirement or a `:latest` tag: the number is computed from the tree and
# the build fails on drift, so the next staleness is caught here rather
# than by a reader who then stops trusting the correlation.
#
# **Each pin states the quantity it means.** "tests" was a pytest
# COLLECTED count, which moves with parametrisation and cannot be
# recomputed without running the suite inside itself. The README now says
# "test functions" and the pin counts `def test_`, which is the same
# quantity under a name that says so.

README_NUMBER = re.compile(r"\*\*([\d,]+) entries\*\*")


def _readme(repo_root) -> str:
    """The README with its wrapping normalised.

    **[2026-09-06] The first version of these pins matched the raw
    text and failed on "8\\n`UNRESOLVED-WITHDRAWN`".** The README is
    wrapped prose, so a phrase a pin looks for can straddle a line
    break. Normalising is the fix; loosening the phrase would have
    been the wrong one."""
    raw = (repo_root / "README.md").read_text(encoding="utf-8")
    return " ".join(raw.split())


def test_the_readme_ledger_counts_are_derived_from_the_ledger(repo_root):
    """Five numbers in one sentence, every one recomputed."""
    from collections import Counter

    from cleft import results_ledger

    text = _readme(repo_root)
    counts = Counter(entry["status"] for entry in results_ledger.ENTRIES)

    total = len(results_ledger.ENTRIES)
    assert f"**{total} entries**" in text, (
        f"the README does not say {total} entries. The ledger holds "
        f"{total}; re-derive the sentence rather than editing one number"
    )
    for status, n in sorted(counts.items()):
        if status == "CLAIMABLE":
            assert f"**only {n} `CLAIMABLE`**" in text, status
        else:
            assert f"{n} `{status}`" in text, (
                f"the README does not say {n} `{status}`"
            )
    # Every status the ledger uses appears, so a new one cannot be added
    # to the ledger and left out of the front page.
    assert len(counts) == 5, (
        f"the ledger now uses {len(counts)} statuses; the README sentence "
        "enumerates five and needs re-deriving"
    )


def test_the_readme_phase_count_is_derived_from_the_phase_modules(repo_root):
    """The sequence has reached whatever the highest phase module says."""
    WORDS = {
        20: "Twenty", 21: "Twenty-one", 22: "Twenty-two",
        23: "Twenty-three", 24: "Twenty-four", 25: "Twenty-five",
        26: "Twenty-six", 27: "Twenty-seven", 28: "Twenty-eight",
        29: "Twenty-nine", 30: "Thirty",
    }
    numbers = sorted(
        int(re.match(r"phase(\d+)", path.stem).group(1))
        for path in (repo_root / "src" / "cleft").glob("phase*.py")
        if re.match(r"phase\d+", path.stem)
    )
    highest = max(numbers)
    assert highest in WORDS, (
        f"phase {highest} has no spelled form here; add it rather than "
        "letting the pin skip"
    )
    assert f"{WORDS[highest]} phases of work" in _readme(repo_root), (
        f"the README does not say {WORDS[highest]} phases. The highest "
        f"phase module is phase{highest}.py"
    )


def test_the_readme_test_count_is_a_floor_and_the_floor_holds(repo_root):
    """**An exact test count cannot be pinned, because the pin is a
    test.**

    [2026-09-06] The first version of this asserted the exact
    number. Adding the four pins in this block moved it from 3,450
    to 3,454, so the pin failed on its own arrival. A count that
    changes on every commit that adds a test is a pin that gets
    bumped mechanically rather than read, which is the shape of the
    thirty `len(ENTRIES) == 38` assertions swept on this same date.

    So the README states a FLOOR. It is derivable, it does not
    drift on ordinary work, and it cannot be gamed downward: the
    suite shrinking below it fails the build, which is the only
    direction that would be a finding.
    """
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "tests/*.py"], cwd=repo_root,
        capture_output=True, text=True, check=True,
    ).stdout.split()
    total = sum(
        len(re.findall(
            r"^def test_", (repo_root / name).read_text(encoding="utf-8"), re.M
        ))
        for name in tracked
    )
    stated = re.search(r"([\d,]+)\+ test functions", _readme(repo_root))
    assert stated, "the README no longer states a test-function floor"
    floor = int(stated.group(1).replace(",", ""))
    assert total >= floor, (
        f"the suite has {total:,} test functions, below the README's "
        f"floor of {floor:,}. Tests were removed, or the floor is wrong"
    )
    # And the floor must not be so far below as to say nothing.
    assert total < floor * 2, (
        f"the suite has {total:,} test functions against a floor of "
        f"{floor:,}. The floor is stale enough to be uninformative; "
        "raise it to the next round number"
    )


def test_the_readme_names_no_count_the_tree_contradicts(repo_root):
    """The sweep behind the three pins above, so a NEW count added to the
    README is caught rather than silently unpinned.

    Every bolded integer in the README is either pinned above, or is a
    banked figure from a closed phase, or is listed here with the record
    it comes from. **A bolded number with no home fails.**"""
    import subprocess

    text = _readme(repo_root)
    # [2026-09-06] Was `([\d,]+)`, which matches a bare comma and
    # reported one as an unpinned count. A digit is required first.
    bolded = set(re.findall(r"\*\*(\d[\d,]*) ", text))

    # Pinned by the three tests above.
    from cleft import results_ledger

    derived = {
        str(len(results_ledger.ENTRIES)),
        f"{len(results_ledger.ENTRIES):,}",
    }

    # Banked figures from closed phases. Each is a measurement, not a
    # count of the tree, so it moves only when its record moves -- and
    # the record it comes from is named so a reader can check it.
    BANKED = {
        "237": "the cohort size, data.manifest",
        "15": "phase22.CONTRAST_FAMILY, the training-objective contrasts",
        "45": "phase23, the ROPE contrasts",
        "4": "Stage G label alternatives, ladder",
        "5": "the rater screen's single-rater targets, scoresheet.RATERS",
        "0": "the ROPE result: none equivalent",
    }

    unpinned = sorted(bolded - derived - set(BANKED))
    assert not unpinned, (
        f"bolded count(s) {unpinned} in README.md are neither derived from "
        "the tree by the pins above nor listed as a banked figure with its "
        "record. Add the pin or add the entry -- a number on the front "
        "page with no home is the shape this module exists to catch"
    )
