"""Write the Phase 7D configs from ``cleft.ladder`` and arm 2's own config.

    PYTHONPATH=src python scripts/generate_phase7d_configs.py [--check]

**Generated, and the hyperparameters are COPIED from the control arm rather
than retyped.** Every 7D arm is compared against
``p7_d1_vit_b16_imagenet_g1`` at 0.2520, so its training configuration IS the
axis's fixed factor: a retyped learning rate would vary two things under a
one-factor name. Arm 2 itself gets no config here -- it already ran, and the
ledger references the existing result.

**The pre-registration is not here.** The arms, the committed readings, the
concat prior and the fifth-arm arithmetic live in
``ladder.PHASE_7D_PATCH_AXIS_REGISTERED`` (2026-08-14, before any artifact)
and the build facts in ``ladder.PHASE_7D_BUILT``, where the suite checks
them.

**Arm 5 [DECIDED 2026-08-15, maintainer]: SQUARE, ``roadb_512_square_g1_v1``**
-- the artifact the patch16@512 comparator itself extracted from, so the
comparison varies patch size and nothing else. Its verified hash is carried
from the comparator's own shipped config
(``roadb_p7_arm_vit_b16_imagenet_512.yaml``), and the vit_b32 SNAPSHOT is
shared: one init serves both the 224 and the 512 extraction, because the
weights do not depend on the input size (dynamic_img_size interpolates the
position grid at forward time, which is the thing being measured).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

from cleft import embedding_plan, ladder  # noqa: E402

CLUSTER_ROOT = "/home/user/codex/cleft-aesthetics"

#: Arm 2, the control. Hyperparameters and the shared input hashes come from
#: here, never retyped.
CONTROL_CONFIG = "p7_d1_vit_b16_imagenet_g1.yaml"

#: Copied verbatim from the control arm.
COPIED = (
    "geometry", "label", "seeds", "inner_val_frac", "max_epochs",
    "patience", "monitor", "learning_rate", "weight_decay", "batch_size",
    "trainable",
)

#: The 7D backbones this generator builds the chain for. Arm 5 (vit_b32 at
#: 512) is deliberately ABSENT: held on the staging decision.
BACKBONES_7D = ("vit_b32", "vit_b8", "mvitv2_b")

#: Extraction batch sizes -- vit_b8's is the measured memory decision
#: (ladder.PHASE_7D_BUILT.extraction_memory): 785 tokens put a 29.6 MB
#: attention matrix per image per block, 15.9x vit_b16's, so the batch drops
#: 4x to keep the tokens^2-per-batch product below the b16 extraction's.
EXTRACT_BATCH = {"vit_b32": 32, "vit_b8": 8, "mvitv2_b": 32}

PLACEHOLDER = "0" * 64

#: Arm 5's staged input -- the maintainer's decision, and the comparator's own
#: artifact. The hash is CARRIED from the shipped comparator config.
STAGED_512 = "roadb_512_square_g1_v1"
COMPARATOR_CONFIG = "roadb_p7_arm_vit_b16_imagenet_512.yaml"


def control() -> dict:
    return yaml.safe_load(
        (REPO / "configs" / CONTROL_CONFIG).read_text(encoding="utf-8")
    )


def _named_inputs(config: dict) -> dict:
    return {entry["name"]: entry for entry in config["inputs"]}


def set_dir(backbone: str) -> str:
    name = embedding_plan.set_name({
        "backbone": backbone, "init": "imagenet", "geometry": "g1",
        "pretrain_scheme": None,
    })
    return f"{CLUSTER_ROOT}/data/embeddings/embeddings_7d_{backbone}_v1/{name}"


def _carried(config_name: str, input_name: str, expected_path: str) -> str:
    """A resolved hash CARRIED from the shipped config, placeholder otherwise.

    Hashes come only from declare_inputs.py against built artifacts; this
    generator can only carry what a declare pass filled in, and only while
    the path still matches -- the generate_roadb_configs rule.
    """
    path = REPO / "configs" / config_name
    if not path.is_file():
        return PLACEHOLDER
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    for entry in payload.get("inputs") or []:
        if entry.get("name") != input_name:
            continue
        rollup = str(entry.get("rollup_sha256") or "")
        if (
            entry.get("path") == expected_path
            and len(rollup) == 64
            and set(rollup) <= set("0123456789abcdef")
            and set(rollup) != {"0"}
        ):
            return rollup
    return PLACEHOLDER


def _init_note(config_name: str, init_path: str, snapshot_config: str) -> str:
    """The pretrained_init paragraph, matching the hash's actual state.

    A header claiming an all-zeros placeholder beside a filled hash would be
    a false statement in a generated file -- the p8_node_weights precedent:
    the text follows what the declare pass did, not what the first draft
    assumed.
    """
    if _carried(config_name, "pretrained_init", init_path) != PLACEHOLDER:
        return (
            "# pretrained_init's rollup is the declare pass's own figure --\n"
            "# filled 2026-08-15 against the built snapshot artifact and\n"
            "# CARRIED by this generator, never typed. If the snapshot is\n"
            "# ever rebuilt, re-run declare_inputs.py; guard 3 refuses a\n"
            "# stale hash.\n"
        )
    return (
        "# pretrained_init's rollup is an all-zeros PLACEHOLDER until\n"
        f"# {snapshot_config} has run and declare_inputs.py\n"
        "# has hashed the artifact. Guard 3 refuses the placeholder, which\n"
        "# is the order being enforced: snapshot, declare, extract.\n"
    )


def _emb_note(config_name: str, entries: list, producer: str) -> str:
    """The embeddings paragraph for an ARM config, matching the hashes'
    actual state -- ``entries`` is ``[(input_name, path), ...]``. All filled
    means the declare pass ran; anything less keeps the placeholder text."""
    filled = all(
        _carried(config_name, name, path) != PLACEHOLDER
        for name, path in entries
    )
    plural = "rollups are" if len(entries) > 1 else "rollup is"
    if filled:
        return (
            f"# The embeddings {plural} the declare pass's own figures --\n"
            f"# filled 2026-08-15 against the sets {producer} built, and\n"
            "# CARRIED by this generator, never typed. If a set is ever\n"
            "# rebuilt, re-run declare_inputs.py; guard 3 refuses a stale\n"
            "# hash.\n"
        )
    return (
        f"# The embeddings {plural} all-zeros PLACEHOLDER(s) until\n"
        f"# {producer} has run and declare_inputs.py has hashed the\n"
        "# set(s). Guard 3 refuses a placeholder.\n"
    )


def _attribution() -> str:
    return (
        "# ATTRIBUTION (maintainer, 2026-08-15): arm 6 is the direct response\n"
        "# to the supervision shared paper (Fan et al., Multiscale Vision\n"
        "# Transformers, ICCV 2021, shared 2026-08-13); arms 1-5 are the\n"
        "# project's decomposition of the 'multi-scale, not 16x16' request\n"
        "# into patch granularity at fixed input, registered before the\n"
        "# paper was shared. ladder.PHASE_7D_BUILT.\n"
    )


def render_snapshot(backbone: str) -> str:
    spec = ladder.PHASE_7D_BUILT
    extra = ""
    if backbone == "mvitv2_b":
        sub = spec["arm_6"]["substitution"]
        extra = (
            "#\n"
            f"# THE SUBSTITUTION: the paper is {sub['paper']} and v1 does not\n"
            "# exist in the pinned timm 1.0.7 at all -- the arm runs\n"
            f"# {sub['ran']}. What v2 adds over the paper's architecture:\n"
            f"# {sub['v2_adds']}.\n"
        )
    payload = {
        "schema_version": 1,
        "phase": "p7d",
        "tier": "keeper",
        "seed": 1337,
        "task": {"kind": "snapshot_pretrained_init", "backbone": backbone},
    }
    body = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    return (
        f"# PHASE 7D — THE PRETRAINED INIT, FROZEN: {backbone}\n"
        "#\n"
        "# GENERATED by scripts/generate_phase7d_configs.py from cleft.ladder.\n"
        "#\n"
        "# Downloads ONCE, snapshots the num_classes=0 state dict into\n"
        f"# data/inits/init_{backbone}_v1/ with the upstream identity (timm\n"
        "# pretrained_cfg, INCLUDING the exact pretrained tag) in\n"
        "# MANIFEST.json, and reports the rollup the extraction config\n"
        "# declares. Same mechanism as the four existing snapshots\n"
        "# (roadb.PRETRAINED_INIT_IS_NOW_DECLARABLE).\n"
        + extra
        + "#\n"
        + _attribution()
        + "#\n"
        f"#   bash -lc \"cd {CLUSTER_ROOT} && PYTHONPATH=src \\\n"
        f"#     python -m cleft.run --config configs/p7d_init_snapshot_{backbone}.yaml\"\n\n"
    ) + body


def render_extract(backbone: str) -> str:
    inputs_by_name = _named_inputs(control())
    init_path = f"{CLUSTER_ROOT}/data/inits/init_{backbone}_v1"
    config_name = f"p7d_extract_{backbone}.yaml"
    payload = {
        "schema_version": 1,
        "phase": "p7d",
        "tier": "keeper",
        "seed": 1337,
        "inputs": [
            dict(inputs_by_name["manifest_v1"]),
            dict(inputs_by_name["staged_v1"]),
            {
                "name": "pretrained_init",
                "path": init_path,
                "rollup_sha256": _carried(config_name, "pretrained_init", init_path),
            },
        ],
        "task": {
            "kind": "extract_embeddings",
            "manifest_artifact": "manifest_v1",
            "staged_artifact": "staged_v1",
            "out_version": f"embeddings_7d_{backbone}_v1",
            "batch_size": EXTRACT_BATCH[backbone],
            "sets": [
                {"backbone": backbone, "init": "imagenet", "geometry": "g1"},
            ],
        },
    }
    body = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    memory = ladder.PHASE_7D_BUILT["extraction_memory"]
    memory_note = ""
    if backbone in ("vit_b8", "mvitv2_b"):
        memory_note = (
            "#\n# MEMORY [MEASURED]: " + memory[backbone].replace("\n", " ")
            + "\n"
        )
    return (
        f"# PHASE 7D — EXTRACTION: {backbone}, ImageNet, G1, whole image, 224\n"
        "#\n"
        "# GENERATED by scripts/generate_phase7d_configs.py from cleft.ladder\n"
        f"# and {CONTROL_CONFIG} (whose manifest/staged hashes are carried,\n"
        "# never retyped).\n"
        "#\n"
        + _init_note(config_name, init_path, f"p7d_init_snapshot_{backbone}.yaml")
        + memory_note
        + "#\n"
        + _attribution()
        + "#\n"
        f"#   bash -lc \"cd {CLUSTER_ROOT} && PYTHONPATH=src \\\n"
        f"#     python -m cleft.run --config configs/{config_name}\"\n\n"
    ) + body


def _arm_readings(backbone: str) -> str:
    reg = ladder.PHASE_7D_PATCH_AXIS_REGISTERED
    if backbone in ("vit_b32", "vit_b8"):
        grid = reg["arms"][f"{backbone}_224"]
        return (
            "# PRE-REGISTERED READING (2026-08-14, verbatim): no\n"
            "# interpolation anywhere -- the grid is the model's own\n"
            f"# ({grid['grid']}, {grid['tokens']} tokens) -- so the Road B\n"
            "# cliff mechanism is absent by construction. A clean test of\n"
            "# patch granularity.\n"
        )
    framing = ladder.PHASE_7D_BUILT["arm_6"]["framing"]
    return (
        "# COMMITTED FRAMING (2026-08-15, before any result, verbatim):\n"
        f"# {framing['why']}.\n"
        f"# Near Swin's cleft result -> {framing['reading_if_near_swin']};\n"
        f"# materially different -> {framing['reading_if_different']}.\n"
        f"# Neither reading is 'MViT should win': "
        f"{framing['neither_is_mvit_should_win']}.\n"
    )


def render_arm(backbone: str) -> str:
    arm = control()
    inputs_by_name = _named_inputs(arm)
    config_name = f"p7d_arm_{backbone}.yaml"
    embeddings_path = set_dir(backbone)
    task = {
        "kind": "train_cv",
        "manifest_artifact": "manifest_v1",
        "staged_artifact": "staged_v1",
        "embeddings_artifact": "embeddings",
        "backbone": backbone,
        "init": "imagenet",
    }
    missing = [name for name in COPIED if name not in arm["task"]]
    if missing:
        raise SystemExit(f"{CONTROL_CONFIG} no longer carries {missing}")
    task.update({name: arm["task"][name] for name in COPIED})
    payload = {
        "schema_version": 1,
        "phase": "p7d",
        "tier": "keeper",
        "seed": arm["task"]["seeds"][0],
        "inputs": [
            dict(inputs_by_name["manifest_v1"]),
            dict(inputs_by_name["staged_v1"]),
            {
                "name": "embeddings",
                "path": embeddings_path,
                "rollup_sha256": _carried(config_name, "embeddings", embeddings_path),
            },
        ],
        "task": task,
    }
    body = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    pooling = (
        "# POOLING: mean over the final stage's 49 tokens -- NO cls token --\n"
        "# and 768-dim, both MEASURED off the built model, so the head width\n"
        "# is deliberate rather than defaulted.\n"
        if backbone == "mvitv2_b"
        else ""
    )
    return (
        f"# PHASE 7D — ARM: {backbone}, ImageNet, G1, whole image, frozen\n"
        "# backbone + linear probe, five seeds, standard ladder criterion\n"
        "# (PCC primary).\n"
        "#\n"
        "# GENERATED by scripts/generate_phase7d_configs.py. Hyperparameters\n"
        f"# COPIED from {CONTROL_CONFIG} -- the control at 0.2520 -- so the\n"
        "# axis varies the backbone and nothing else.\n"
        "#\n"
        + _arm_readings(backbone)
        + pooling
        + "#\n"
        + _emb_note(
            config_name, [("embeddings", embeddings_path)],
            f"p7d_extract_{backbone}.yaml",
        )
        + "#\n"
        + _attribution()
        + "#\n"
        f"#   bash -lc \"cd {CLUSTER_ROOT} && PYTHONPATH=src \\\n"
        f"#     python -m cleft.run --config configs/{config_name}\"\n\n"
    ) + body


def _staged_512_entry() -> dict:
    """The square-512 staged input, hash carried from the comparator's own
    shipped config -- the one-factor property made mechanical: if the
    comparator's declaration ever changes, this drifts rather than diverges."""
    payload = yaml.safe_load(
        (REPO / "configs" / COMPARATOR_CONFIG).read_text(encoding="utf-8")
    )
    for entry in payload["inputs"]:
        if entry["name"] == STAGED_512:
            return dict(entry)
    raise SystemExit(
        f"{COMPARATOR_CONFIG} no longer declares {STAGED_512}; the one-factor "
        "carry has nothing to carry"
    )


def _fifth_arm_reading() -> str:
    """The committed reading, composed from the registration's own record so
    the header cannot drift from what the suite checks."""
    reg = ladder.PHASE_7D_PATCH_AXIS_REGISTERED
    readings = reg["fifth_arm_readings_committed"]
    return (
        "# COMMITTED READING (2026-08-14, verbatim from\n"
        "# ladder.PHASE_7D_PATCH_AXIS_REGISTERED). patch32@512 and\n"
        "# patch16@512 share the IDENTICAL 2.286x position-grid stretch\n"
        "# (7x7 -> 16x16 against 14x14 -> 32x32). This arm CANNOT 'beat the\n"
        "# cliff by milder interpolation' -- that rationale was arithmetic\n"
        "# error, caught and recorded before anything ran\n"
        "# (amendment_arithmetic_corrected). What it does is DISCRIMINATE:\n"
        f"# cliffs like patch16@512 -> {readings['cliffs_like_patch16_at_512']};\n"
        f"# does not cliff -> {readings['does_not_cliff']}.\n"
        f"# Either way: {readings['either_way']}.\n"
    )


def render_extract_512() -> str:
    inputs_by_name = _named_inputs(control())
    init_path = f"{CLUSTER_ROOT}/data/inits/init_vit_b32_v1"
    config_name = "p7d_extract_vit_b32_512.yaml"
    payload = {
        "schema_version": 1,
        "phase": "p7d",
        "tier": "keeper",
        "seed": 1337,
        "inputs": [
            dict(inputs_by_name["manifest_v1"]),
            _staged_512_entry(),
            {
                "name": "pretrained_init",
                "path": init_path,
                "rollup_sha256": _carried(config_name, "pretrained_init", init_path),
            },
        ],
        "task": {
            "kind": "extract_embeddings",
            "manifest_artifact": "manifest_v1",
            "staged_artifact": STAGED_512,
            "out_version": "embeddings_7d_vit_b32_512_v1",
            "batch_size": 32,
            "sets": [
                {"backbone": "vit_b32", "init": "imagenet", "geometry": "g1"},
            ],
        },
    }
    body = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    return (
        "# PHASE 7D — ARM 5 EXTRACTION: vit_b32 at 512, SQUARE G1\n"
        "#\n"
        "# GENERATED by scripts/generate_phase7d_configs.py from cleft.ladder\n"
        f"# and {COMPARATOR_CONFIG}.\n"
        "#\n"
        "# STAGING [DECIDED 2026-08-15, maintainer]: SQUARE --\n"
        f"# {STAGED_512}, the artifact the patch16@512 comparator\n"
        "# (0.0857) itself extracted from, hash CARRIED from its shipped\n"
        "# config. One factor: the comparison varies patch size and nothing\n"
        "# else.\n"
        "#\n"
        "# THE SNAPSHOT IS SHARED with the 224 extraction\n"
        "# (data/inits/init_vit_b32_v1): the weights do not depend on the\n"
        "# input size -- dynamic_img_size interpolates the position grid\n"
        "# 7x7 -> 16x16 at forward time, which is the thing being measured.\n"
        "#\n"
        + _fifth_arm_reading()
        + "#\n"
        + _init_note(config_name, init_path, "p7d_init_snapshot_vit_b32.yaml")
        + "#\n"
        + _attribution()
        + "#\n"
        f"#   bash -lc \"cd {CLUSTER_ROOT} && PYTHONPATH=src \\\n"
        f"#     python -m cleft.run --config configs/{config_name}\"\n\n"
    ) + body


def render_arm_512() -> str:
    arm = control()
    inputs_by_name = _named_inputs(arm)
    config_name = "p7d_arm_vit_b32_512.yaml"
    embeddings_path = (
        f"{CLUSTER_ROOT}/data/embeddings/embeddings_7d_vit_b32_512_v1/"
        + embedding_plan.set_name({
            "backbone": "vit_b32", "init": "imagenet", "geometry": "g1",
            "pretrain_scheme": None,
        })
    )
    task = {
        "kind": "train_cv",
        "manifest_artifact": "manifest_v1",
        "staged_artifact": STAGED_512,
        "embeddings_artifact": "embeddings",
        "backbone": "vit_b32",
        "init": "imagenet",
    }
    task.update({name: arm["task"][name] for name in COPIED})
    payload = {
        "schema_version": 1,
        "phase": "p7d",
        "tier": "keeper",
        "seed": arm["task"]["seeds"][0],
        "inputs": [
            dict(inputs_by_name["manifest_v1"]),
            _staged_512_entry(),
            {
                "name": "embeddings",
                "path": embeddings_path,
                "rollup_sha256": _carried(config_name, "embeddings", embeddings_path),
            },
        ],
        "task": task,
    }
    body = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    return (
        "# PHASE 7D — ARM 5: vit_b32 at 512, SQUARE G1, whole image, frozen\n"
        "# backbone + linear probe, five seeds, standard ladder criterion\n"
        "# (PCC primary). The mechanism-discriminating arm.\n"
        "#\n"
        "# GENERATED by scripts/generate_phase7d_configs.py. Hyperparameters\n"
        f"# COPIED from {CONTROL_CONFIG}; staging [DECIDED 2026-08-15,\n"
        f"# maintainer]: SQUARE, {STAGED_512} -- the comparator's own\n"
        "# artifact, hash carried from its shipped config, so the\n"
        "# patch16@512 comparison varies patch size and NOTHING else.\n"
        "#\n"
        + _fifth_arm_reading()
        + "#\n"
        + _emb_note(
            config_name, [("embeddings", embeddings_path)],
            "p7d_extract_vit_b32_512.yaml",
        )
        + "#\n"
        + _attribution()
        + "#\n"
        f"#   bash -lc \"cd {CLUSTER_ROOT} && PYTHONPATH=src \\\n"
        f"#     python -m cleft.run --config configs/{config_name}\"\n\n"
    ) + body


def render_concat() -> str:
    arm = control()
    inputs_by_name = _named_inputs(arm)
    config_name = "p7d_arm_concat_multiscale.yaml"
    # 8 + 16 + 32, the order the arm is named in. b16's set is the EXISTING
    # verified artifact -- the same input the control itself declared.
    b16_entry = dict(inputs_by_name["embeddings"])
    b16_entry["name"] = "emb_vit_b16"
    concat_inputs = [
        {
            "name": "emb_vit_b8",
            "path": set_dir("vit_b8"),
            "rollup_sha256": _carried(config_name, "emb_vit_b8", set_dir("vit_b8")),
        },
        b16_entry,
        {
            "name": "emb_vit_b32",
            "path": set_dir("vit_b32"),
            "rollup_sha256": _carried(config_name, "emb_vit_b32", set_dir("vit_b32")),
        },
    ]
    task = {
        "kind": "train_cv",
        "manifest_artifact": "manifest_v1",
        "staged_artifact": "staged_v1",
        "backbone": "concat",
        "init": "imagenet",
        "concat_embeddings_artifacts": [
            {"artifact": "emb_vit_b8", "backbone": "vit_b8"},
            {"artifact": "emb_vit_b16", "backbone": "vit_b16"},
            {"artifact": "emb_vit_b32", "backbone": "vit_b32"},
        ],
    }
    task.update({name: arm["task"][name] for name in COPIED})
    payload = {
        "schema_version": 1,
        "phase": "p7d",
        "tier": "keeper",
        "seed": arm["task"]["seeds"][0],
        "inputs": [
            dict(inputs_by_name["manifest_v1"]),
            dict(inputs_by_name["staged_v1"]),
            *concat_inputs,
        ],
        "task": task,
    }
    body = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    return (
        "# PHASE 7D — ARM 4: CONCATENATED 8+16+32, one factor from the\n"
        "# singles, no new extraction. Linear probe over the 2304-dim\n"
        "# concatenation, five seeds, standard ladder criterion.\n"
        "#\n"
        "# GENERATED by scripts/generate_phase7d_configs.py. Hyperparameters\n"
        f"# COPIED from {CONTROL_CONFIG}; vit_b16's set is the control's own\n"
        "# verified artifact, carried with its hash.\n"
        "#\n"
        "# PRE-REGISTERED PRIOR [MEASURED, 2026-08-14, verbatim]: Phase 7B\n"
        "# measured embedding concatenation at +0.0221 inner-validation and\n"
        "# -0.0542 out-of-fold. A measured prior AGAINST this arm on this\n"
        "# cohort, stated where the result will be read\n"
        "# (phase7b.SEARCH_AXIS_VERDICTS; the loader repeats it in the run\n"
        "# record).\n"
        "#\n"
        + _emb_note(
            config_name,
            [
                ("emb_vit_b8", set_dir("vit_b8")),
                ("emb_vit_b32", set_dir("vit_b32")),
            ],
            "the b8/b32 extractions",
        )
        + "#\n"
        + _attribution()
        + "#\n"
        f"#   bash -lc \"cd {CLUSTER_ROOT} && PYTHONPATH=src \\\n"
        f"#     python -m cleft.run --config configs/{config_name}\"\n\n"
    ) + body


def configs() -> dict:
    out = {}
    for backbone in BACKBONES_7D:
        out[f"p7d_init_snapshot_{backbone}.yaml"] = (
            lambda b=backbone: render_snapshot(b)
        )
        out[f"p7d_extract_{backbone}.yaml"] = lambda b=backbone: render_extract(b)
        out[f"p7d_arm_{backbone}.yaml"] = lambda b=backbone: render_arm(b)
    out["p7d_arm_concat_multiscale.yaml"] = render_concat
    out["p7d_extract_vit_b32_512.yaml"] = render_extract_512
    out["p7d_arm_vit_b32_512.yaml"] = render_arm_512
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    drifted = 0
    entries = configs()
    for name, builder in entries.items():
        path = REPO / "configs" / name
        text = builder()
        if args.check:
            if not path.is_file():
                print(f"  DRIFT {name}: missing")
                drifted += 1
            elif path.read_text(encoding="utf-8") != text:
                print(f"  DRIFT {name}: differs from the derived config")
                drifted += 1
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {name}")
    if args.check:
        print(f"{len(entries)} configs checked, {drifted} drifted")
        return 1 if drifted else 0
    print(
        f"{len(entries)} configs: 3 snapshots, 4 extractions, 5 arms. "
        "Arm 2 references the existing 0.2520; arm 5 extracts from "
        f"{STAGED_512} (decided square, 2026-08-15) and shares vit_b32's "
        "snapshot with the 224 extraction."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
