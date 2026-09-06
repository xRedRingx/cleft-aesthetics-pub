"""The config schema.

Chain 3 of the previous failure: scientific settings lived in CLI flags whose
names changed, so the command that produced a result was not recoverable from
the result. The config is now the provenance record, which only works if a typo
is fatal rather than silently ignored.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

from cleft.config import (
    ConfigError,
    dump_config,
    expand_input_paths,
    load_config,
)

from fixtures import builders


# --------------------------------------------------------------------------
# ${CLEFT_*} input paths: the same data at a different path on each machine
# --------------------------------------------------------------------------


def an_input(path: str, name: str = "scut_root") -> dict:
    return {"name": name, "path": path, "rollup_sha256": "0" * 64}


def test_an_env_reference_resolves_from_the_environment():
    """SCUT is the first dataset that exists on both machines, so its config
    cannot name a single path. The declared rollup then verifies whatever the
    variable resolved to, which is why the indirection costs no provenance."""
    resolved = expand_input_paths(
        [an_input("${CLEFT_SCUT_ROOT}")],
        environ={"CLEFT_SCUT_ROOT": "/home/user/codex/scut/SCUT-FBP5500_v2"},
    )
    assert resolved[0]["path"] == "/home/user/codex/scut/SCUT-FBP5500_v2"
    # Both halves of the record: what was declared, and where it came from.
    assert resolved[0]["path_declared_as"] == "${CLEFT_SCUT_ROOT}"
    assert resolved[0]["path_from_env"] == "CLEFT_SCUT_ROOT"


def test_a_subpath_after_the_reference_is_kept():
    resolved = expand_input_paths(
        [an_input("${CLEFT_SCUT_ROOT}/Images")],
        environ={"CLEFT_SCUT_ROOT": "/data/scut"},
    )
    assert resolved[0]["path"] == "/data/scut/Images"


def test_a_trailing_separator_in_the_variable_does_not_double_up():
    for value in ("/data/scut/", "/data/scut"):
        resolved = expand_input_paths(
            [an_input("${CLEFT_SCUT_ROOT}/Images")],
            environ={"CLEFT_SCUT_ROOT": value},
        )
        assert resolved[0]["path"] == "/data/scut/Images"


def test_a_windows_value_resolves_too():
    """The laptop half. The value is absolute on Windows even though it has no
    leading slash, which is the case a platform-native check gets wrong."""
    resolved = expand_input_paths(
        [an_input("${CLEFT_SCUT_ROOT}")],
        environ={"CLEFT_SCUT_ROOT": "C:\\data\\scut-fbp5500\\SCUT-FBP5500_v2"},
    )
    assert resolved[0]["path"] == "C:\\data\\scut-fbp5500\\SCUT-FBP5500_v2"


def test_an_unset_variable_is_refused_with_both_locations_named():
    """The error has to be actionable on whichever machine hit it: an unset
    variable is a setup step, not a corrupt config."""
    with pytest.raises(ConfigError, match="CLEFT_SCUT_ROOT is not set"):
        expand_input_paths([an_input("${CLEFT_SCUT_ROOT}")], environ={})


def test_an_empty_variable_is_treated_as_unset():
    """An exported-but-empty variable would otherwise resolve to the repo root."""
    with pytest.raises(ConfigError, match="not set"):
        expand_input_paths(
            [an_input("${CLEFT_SCUT_ROOT}")], environ={"CLEFT_SCUT_ROOT": ""}
        )


def test_a_relative_variable_value_is_refused():
    """**The original defect arriving by a different door.** A relative value
    would be re-rooted under the repository by resolve_declared_path, producing
    exactly the /repo/C:/data/... shape this indirection removes."""
    with pytest.raises(ConfigError, match="not an absolute path"):
        expand_input_paths(
            [an_input("${CLEFT_SCUT_ROOT}")], environ={"CLEFT_SCUT_ROOT": "data/scut"}
        )


def test_only_cleft_prefixed_variables_can_be_read():
    """**A safety property, not tidiness.** Unrestricted expansion would let a
    committed config read PATH, a token or a credential and paste it into a run
    record that is then shared."""
    from cleft.config import classify_declared_path

    with pytest.raises(ConfigError, match="variable reference"):
        classify_declared_path("${HOME}")
    with pytest.raises(ConfigError, match="variable reference"):
        classify_declared_path("${AWS_SECRET_ACCESS_KEY}")

    # And the expander leaves a non-matching string alone rather than half-doing
    # it -- classification is what rejects it, at a single place.
    untouched = expand_input_paths([an_input("data/smoke/v1")], environ={})
    assert untouched[0]["path"] == "data/smoke/v1"
    assert "path_from_env" not in untouched[0]


def test_a_plain_path_is_passed_through_unchanged_and_ungarnished():
    """Non-env entries must come out identical, or every existing config and
    every test fixture that declares str(tmp_path) changes shape."""
    entry = an_input("/home/user/codex/cleft-aesthetics/data/staged/staged_v1")
    result = expand_input_paths([entry], environ={"CLEFT_SCUT_ROOT": "/x"})
    assert result[0] is entry


def test_expansion_does_not_reach_task_fields(write_config, tmp_path):
    """**Chain 3 of the original failure.** A scientific setting resolved from the
    environment is a setting the config does not record, which is the whole reason
    nothing is a CLI flag (Part 2.3). Expansion is confined to inputs[].path."""
    path = write_config()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["task"]["n_samples"] = 8
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    import os

    os.environ["CLEFT_TEST_LEAK"] = "99"
    try:
        loaded = load_config(path)
        assert loaded["task"]["n_samples"] == 8
    finally:
        del os.environ["CLEFT_TEST_LEAK"]


def test_load_config_expands_and_a_missing_variable_names_the_config(tmp_path):
    """End to end through the loader, because that is the path RunContext takes."""
    path = builders.write_config(
        tmp_path / "cfg.yaml",
        inputs=[an_input("${CLEFT_SCUT_ROOT_ABSENT_IN_TESTS}")],
    )
    with pytest.raises(ConfigError, match="ABSENT_IN_TESTS is not set"):
        load_config(path)


def test_an_expanded_config_survives_a_dump_and_reload(tmp_path, monkeypatch):
    """**The latent trap.** ``expand_input_paths`` adds two derived keys, and the
    loader is strict about unknown keys -- so without stripping them,
    ``load_config(dump_config(cfg))`` failed on its own output for every config
    using ``${CLEFT_*}``. The existing round-trip test passed only because its
    fixtures declare no variables."""
    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/data/scut")
    path = builders.write_config(
        tmp_path / "cfg.yaml", inputs=[an_input("${CLEFT_SCUT_ROOT}")]
    )
    first = load_config(path)
    assert first["inputs"][0]["path"] == "/data/scut"

    reloaded = load_config(dump_config(first, tmp_path / "again.yaml"))
    assert reloaded["inputs"][0]["path"] == "/data/scut"
    assert dump_config(first) == dump_config(reloaded)


def test_dumping_restores_the_declaration_not_the_resolution(tmp_path, monkeypatch):
    """A dump must stay PORTABLE. Emitting the resolved path would quietly turn a
    portable config into a machine-specific one -- the same trap as the
    declare_inputs paste block, arriving through a different tool."""
    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/data/scut")
    path = builders.write_config(
        tmp_path / "cfg.yaml", inputs=[an_input("${CLEFT_SCUT_ROOT}/Images")]
    )
    text = dump_config(load_config(path))

    assert "${CLEFT_SCUT_ROOT}/Images" in text
    assert "/data/scut" not in text
    for derived in ("path_declared_as", "path_from_env"):
        assert derived not in text


def test_valid_config_loads(write_config):
    cfg = load_config(write_config())
    assert cfg["phase"] == "p0"
    assert cfg["tier"] == "dev"
    assert cfg["seed"] == 1337


def test_unknown_key_raises(write_config, tmp_path):
    path = write_config()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["learing_rate"] = 0.01  # the typo that must not be absorbed
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="learing_rate"):
        load_config(path)


def test_unknown_nested_key_raises(write_config):
    path = write_config()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["task"]["n_smaples"] = 8
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="task.n_smaples"):
        load_config(path)


def test_unknown_key_inside_a_list_item_raises(write_config, tmp_path):
    path = write_config(
        inputs=[
            {
                "name": "a",
                "path": str(tmp_path),
                "rollup_sha256": "0" * 64,
                "rolup_sha256": "0" * 64,
            }
        ]
    )
    with pytest.raises(ConfigError, match=r"inputs\[0\].rolup_sha256"):
        load_config(path)


@pytest.mark.parametrize("missing", ["schema_version", "phase", "tier", "seed", "task"])
def test_missing_required_key_raises(write_config, missing):
    path = write_config()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    del raw[missing]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match=missing):
        load_config(path)


def test_wrong_type_raises(write_config):
    path = write_config(seed="1337")
    with pytest.raises(ConfigError, match="seed"):
        load_config(path)


def test_bool_is_not_accepted_as_an_int(write_config):
    """In python ``True == 1``; a config that says ``seed: true`` is a mistake."""
    path = write_config(seed=True)
    with pytest.raises(ConfigError, match="seed"):
        load_config(path)


def test_tier_outside_the_allowed_set_raises(write_config):
    path = write_config(tier="production")
    with pytest.raises(ConfigError, match="tier"):
        load_config(path)


def test_unsupported_schema_version_raises(write_config):
    path = write_config(schema_version=99)
    with pytest.raises(ConfigError, match="schema_version"):
        load_config(path)


def test_config_round_trips(write_config):
    path = write_config(
        inputs=[{"name": "a", "path": "data/x/v1", "rollup_sha256": "0" * 64}]
    )
    first = load_config(path)
    reloaded = load_config(dump_config(first, path.parent / "again.yaml"))
    assert first == reloaded
    assert dump_config(first) == dump_config(reloaded)


def test_dump_is_canonical(write_config, tmp_path):
    """Key order in the source file must not change the dumped form."""
    a = write_config("a.yaml")
    raw = builders.config_dict()
    b = tmp_path / "b.yaml"
    b.write_text(
        yaml.safe_dump(dict(reversed(list(raw.items()))), sort_keys=False),
        encoding="utf-8",
    )
    assert dump_config(load_config(a)) == dump_config(load_config(b))


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


# --------------------------------------------------------------------------
# no scientific setting may be overridden by a CLI flag
# --------------------------------------------------------------------------


def test_run_py_accepts_only_config_and_out():
    """Part 2.3: exactly two CLI arguments exist, forever."""
    from cleft.run import build_parser

    options = set()
    for action in build_parser()._actions:
        options.update(action.option_strings)

    assert options == {"-h", "--help", "--config", "--out"}


def test_run_main_takes_no_scientific_keyword_arguments():
    """A keyword default is a flag with extra steps."""
    from cleft.run import main

    params = set(inspect.signature(main).parameters) - {"argv"}
    assert not params, f"main() grew arguments: {sorted(params)}"


# --------------------------------------------------------------------------
# defaults and choices: the two ways a config lies quietly
# --------------------------------------------------------------------------


def test_every_trainable_field_constrains_its_choices():
    """**A typo here runs a different arm rather than failing.**

    ``prepare_features`` branches on ``trainable == "full"`` and ``make_factory``
    on ``trainable == "head"``, so a value that is neither -- ``hed`` -- takes the
    FROZEN-embedding path for features and the FULL fine-tuning path for the
    model. It runs, it produces numbers, and metrics.json records ``hed``.

    ``partition_sensitivity`` shipped without ``choices`` while ``train_cv`` had
    them, which is exactly the inconsistency that makes the hole invisible: a
    reader who has seen the loader reject the typo once assumes it always does.

    **The assertion was keyed to the literal ``("head", "full")`` and is now
    keyed to the invariant.** ``train_graph_cv`` grew a ``trainable`` field
    with its own vocabulary -- ``graph_layers`` / ``classifier``, the third
    regime and its diagnostic -- and a check spelling out one task's values
    cannot express "every trainable field is constrained". It failed loudly
    rather than silently skipping, which is the better half of the pattern the
    parameter/policy check ran into (PLAN R7, instance 4), but the fix is the
    same shape: assert the property, then pin each known vocabulary separately
    so drift in one still fails.
    """
    from cleft.config.schema import TASK_SPECS

    #: What each task's policy vocabulary IS. Pinned per kind, so adding a
    #: choice to an existing task is a deliberate edit here rather than a
    #: silent widening -- and so the two vocabularies cannot be confused for
    #: each other, which is the R2 error waiting in a shared field name.
    expected = {
        "train_cv": ("head", "full"),
        "partition_sensitivity": ("head", "full"),
        #: Phase 15's probe DERIVES its recipe from A's shipped arm, so it
        #: takes A's vocabulary. Pinned HERE, separately, for the reason the
        #: docstring gives: a second kind sharing the field name must not
        #: inherit the first one's freedom. It was shipped unconstrained and
        #: THIS TEST CAUGHT IT -- instance 5 of the same shape.
        "probe_mebeauty": ("head", "full"),
        "train_graph_cv": ("graph_layers", "classifier", "classifier_adabn"),
        #: One choice: the refit gate is arm B's 0.1719, which only the third
        #: regime reproduces. The probe diagnostics score 0.1340 and 0.0468
        #: and would fail it after ten seeds of fitting -- so explaining a
        #: probe needs its own gate value and costs an edit, not a config
        #: field. Third vocabulary under one field name; R2's shape again.
        "node_weights": ("graph_layers",),
        #: [2026-08-30, the pin fired as designed] The Phase 10 annex's
        #: compute gate: ONE choice, "full", REQUIRED -- the only kinds in
        #: the project where full fine-tuning is operationalised
        #: (phase10_annex.RULING_B_FULL_TRAINABLE, a scoped departure).
        #: The model refuses "full" under any non-annex recipe too, so
        #: the scope is enforced at both layers.
        "p10x_gate_fullfit": ("full",),
        #: [2026-08-31] The annex's distribution arm, which runs the
        #: gate's own fit 2,500 times: same vocabulary, same reason.
        "p10x_distribution": ("full",),
        #: [2026-08-31, the pin fired as designed] Phase 20's permutation
        #: arms. The spec is COPIED from train_cv, so this field arrived
        #: as ("head", "full") and was NARROWED to head alone -- the
        #: copy is what makes the control the probe's own fit path, and
        #: the narrowing is what stops the copy from quietly making a
        #: fourth kind admit full fine-tuning. A control at "full" would
        #: control an arm this project has never run.
        "p20_permutation": ("head",),
    }

    seen = set()
    for kind, spec in TASK_SPECS.items():
        field = spec.get("trainable")
        if field is None:
            continue
        seen.add(kind)
        assert field.choices, (
            f"{kind}.trainable does not constrain its choices, so a typo is "
            "absorbed and trains an arm nobody configured"
        )
        # The default must be reachable through the same gate the config is.
        # A default outside `choices` takes effect on every run that omits the
        # field while being a value the loader would reject if written down.
        #
        # **AMENDED 2026-08-24, and the amendment is MEASURED below.** The
        # sentence above names the harm exactly: it happens "on every run that
        # omits the field". A REQUIRED field cannot be omitted -- the loader
        # raises at schema.py:2150 before `default` is read -- so for those
        # kinds the default is dead code and `None` is the honest value. The
        # guard was over-broad, not the new field wrong; narrowing it here
        # keeps the harm it was built for fully covered.
        if not field.required:
            assert field.default in field.choices, (
                f"{kind}.trainable defaults to {field.default!r}, which its "
                f"own choices {field.choices} would refuse"
            )
        assert kind in expected, (
            f"{kind} grew a trainable field with vocabulary {field.choices} "
            "and no expectation recorded here; two tasks whose trainable "
            "fields mean different things must each be pinned"
        )
        assert field.choices == expected[kind], (
            f"{kind}.trainable is {field.choices}, expected {expected[kind]}"
        )

    assert seen == set(expected), (
        f"a task lost its trainable field: expected {sorted(expected)}, "
        f"found {sorted(seen)}"
    )


def test_schema_defaults_agree_with_the_handler_fallbacks():
    """**The schema default is the one that takes effect, always.**

    ``_validate_mapping`` fills every absent optional with its schema default, so
    ``task["learning_rate"]`` is present by the time a handler runs and the
    ``task.get(..., fallback)`` fallbacks are unreachable. When the two disagree,
    the run uses the schema's value while the code documents the other -- and
    metrics.json faithfully records a number nobody chose.

    ``train_cv`` declared 1e-4 against three code paths documenting 1e-3.
    """
    from cleft.config.schema import TASK_SPECS

    for kind in ("train_cv", "partition_sensitivity"):
        field = TASK_SPECS[kind]["learning_rate"]
        assert field.default == 1e-3, (
            f"{kind}.learning_rate defaults to {field.default}, but the frozen "
            "head is documented at 1e-3 in run.py and phase3.make_factory"
        )


def test_an_absent_optional_is_filled_from_the_schema_not_the_handler():
    """The mechanism the test above depends on, asserted directly rather than
    inferred -- if this ever changed, the two fixes would silently stop
    mattering and nothing would say so."""
    from cleft.config.schema import Field, _validate_mapping

    spec = {"knob": Field(str, required=False, default="from_schema")}
    assert _validate_mapping({}, spec, "task")["knob"] == "from_schema"


def test_symnose_audit_offers_only_the_geometry_its_module_accepts():
    """``symnose.feature_matrix`` refuses g1, but the audit never goes through
    that refusal -- it only picks a staged tensor -- so g1 would have run."""
    from cleft.config.schema import TASK_SPECS

    assert TASK_SPECS["symnose_audit"]["geometry"].choices == ("g2",)


def test_symnose_audit_declares_no_label():
    """It discarded the labels array, so the field could not change a result.
    A knob that does nothing is worse than no knob: it reads as though the audit
    were run against a chosen target."""
    from cleft.config.schema import TASK_SPECS

    assert "label" not in TASK_SPECS["symnose_audit"]


def test_a_required_trainable_field_cannot_be_omitted():
    """The narrowing above, measured rather than reasoned.

    ``test_every_trainable_field_constrains_its_choices`` stopped demanding a
    reachable default for REQUIRED trainable fields, on the argument that the
    loader refuses an omitted required key before the default is consulted.
    That argument is checked here against the loader itself, on a SHIPPED
    config -- so if the required/default handling ever changes, the exemption
    fails with it rather than quietly widening.
    """
    from cleft.config.schema import TASK_SPECS, _validate_mapping

    repo = Path(__file__).resolve().parents[1]
    required = sorted(
        kind for kind, spec in TASK_SPECS.items()
        if (f := spec.get("trainable")) is not None and f.required
    )
    # [2026-08-30, the pin fired as designed] p10x_gate_fullfit joined:
    # the annex gate's trainable is REQUIRED at ("full",) -- see the
    # vocabulary pin above -- and its shipped config is checked below.
    # [2026-08-31] p10x_distribution joined for the same reason.
    assert required == [
        "p10x_distribution", "p10x_gate_fullfit", "probe_mebeauty"
    ], "a new kind made trainable required; give it a shipped config here"

    for kind, config_name in (
        ("probe_mebeauty", "p15_probe_mebeauty_g1.yaml"),
        ("p10x_gate_fullfit", "p10x_gate_fullfit.yaml"),
        ("p10x_distribution", "p10x_distribution.yaml"),
    ):
        task = yaml.safe_load(
            (repo / "configs" / config_name).read_text(encoding="utf-8")
        )["task"]
        # It loads as shipped...
        loaded = _validate_mapping(task, TASK_SPECS[kind], "task")
        assert loaded["trainable"] in TASK_SPECS[kind]["trainable"].choices
        # ...and omitting it is REFUSED, so no default can take effect.
        without = {k: v for k, v in task.items() if k != "trainable"}
        with pytest.raises(
            ConfigError, match="missing required key: task.trainable"
        ):
            _validate_mapping(without, TASK_SPECS[kind], "task")
