"""Phase 25's five configs, from one source.

    python scripts/generate_phase25_configs.py            # write
    python scripts/generate_phase25_configs.py --check    # re-derive, diff

Two extraction configs, two arm configs and the three-contrast config
(``phase25.THE_TWO_ARMS_RULED``, ``THE_FAMILY_OF_THREE``).
**Every field of the arms except the backbone and its init is COPIED FROM
THE PROBE'S OWN CONFIG at generation time** rather than retyped here, so
"one factor" is true by construction: if the probe's recipe ever moved,
``--check`` would report drift rather than the arms quietly differing on a
second axis.

**THE CROSSED-FILL GUARD.** D2's artifact hash and D1's are both 64 hex
characters over sets of nearly identical size (731,250 and 731,243 bytes),
and **a crossed fill would run each arm on the other's features with
nothing downstream noticing**: the shapes match, the pairing check compares
the config against the artifact's own metadata, and both arms would produce
entirely ordinary numbers for the wrong model. The rollups are therefore
keyed to the SET DIRECTORY NAME, not to the arm, and ``build`` refuses if a
config's declared path and its hash disagree.
"""

from __future__ import annotations

import argparse
import pathlib

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"

#: The probe whose recipe the arms copy.
PROBE = "p7_d1_vit_b16_imagenet_g1"

#: The config the shared inputs are copied from, byte for byte.
DONOR = "p9_deall_reference.yaml"

#: **[OWED] The two arm RUN directories -- EMPTY, and deliberately.**
#:
#: A run directory is named ``<config-stem>__<sha8>__<job-id>`` and
#: **neither half is knowable from here**: the sha8 is the commit the run
#: was launched at, the job id is the cluster's. **Phase 22 invented four
#: such names and all four were wrong.**
#:
#: A first attempt emitted visible ``PENDING_SHA8`` markers instead of
#: guesses, and **``test_no_declared_run_directory_contradicts_itself``
#: refused them** -- correctly: a declared run directory must LOOK like
#: one, and a marker that fails the naming rule is still a path a config
#: declares. **The guard was right and the config is not shipped until the
#: names exist**; loosening the guard to admit a placeholder would be
#: tuning a check until it passes.
#:
#: **[2026-09-02] It is now ONE COMMAND, not an edit to this file.** The
#: names arrive the way every other run-directory list has arrived -- a
#: file at the drive root, one path per line (the ``p7d_arm_dirs.txt``
#: precedent):
#:
#:     runs/keeper/p25/p25_arm_d2__<sha8>__<job-id>/
#:     runs/keeper/p25/p25_arm_d1__<sha8>__<job-id>/
#:
#:     python scripts/generate_phase25_configs.py --run-dirs <path>/p25_arm_dirs.txt
#:
#: Order does not matter and a trailing slash is fine: each line is matched
#: to its arm by the STEM the contract puts first, so a mis-ordered file
#: cannot swap the two arms. The rollups stay all-zero -- ``declare_inputs``
#: fills them, and never this script.
#: **[SUPPLIED 2026-09-02, and now BAKED IN.]** ``--run-dirs`` was the
#: intake route while the names were unknown; once known they belong in
#: the source, like every other declared path in these generators.
#:
#: **A ``--check`` that needs a file at the drive root checks nothing for
#: anyone else**, and it had a hole: with an empty mapping the contrasts
#: config was not built, so ``--check`` did not compare it and reported
#: "4 configs match" while a fifth sat shipped and unverified. Baking the
#: names closes that.
#:
#: **Identified by MEASUREMENT, not by picking the newer pair**: the
#: ``0bf86712`` pair has no log (the pre-fix crashes), and the
#: ``2d54e77a`` pair carries five seeds each with a
#: ``seed_variance.json`` giving the banked means (D1 0.2393, D2 0.1326).
#: The rollups stay all-zero -- ``declare_inputs`` fills those, never this
#: script.
RUN_DIRECTORIES: dict[str, str] = {
    "p25_arm_d2": (
        "/home/user/codex/cleft-aesthetics/runs/keeper/p25/"
        "p25_arm_d2__2d54e77a__p25-arm-d2-2"
    ),
    "p25_arm_d1": (
        "/home/user/codex/cleft-aesthetics/runs/keeper/p25/"
        "p25_arm_d1__2d54e77a__p25-arm-d1-2"
    ),
}

#: **[FILLED 2026-09-02 from declare_inputs.py]** Keyed by the RUN
#: DIRECTORY NAME, which is what each hash is actually of -- the same
#: discipline the extraction fill used, and for the same reason.
#:
#: **THE CROSSED-FILL HAZARD IS SHARPER HERE than it was for the
#: extraction sets.** The two runs hold **645 files each** and differ by
#: **136 bytes** -- 11,421,618 against 11,421,754 -- and are otherwise
#: identical in shape: same seeds, same folds, same 237 patients, same
#: CSV columns. A crossed fill would load D1's predictions under D2's
#: name, **compute three entirely ordinary contrasts, and satisfy every
#: check downstream**: the pairing check compares a config against an
#: artifact's own metadata, and both artifacts are well-formed. Nothing
#: would raise, and the verdicts would be about the wrong arms.
RUN_ROLLUPS = {
    "p25_arm_d2__2d54e77a__p25-arm-d2-2":
        "eed3b646040c9f14a5d056b3d34a3106fddf8f50e143b214edfe592996b7dec8",
    "p25_arm_d1__2d54e77a__p25-arm-d1-2":
        "9b30f39203264af154f7130ff9d1f7ba589d1d2ee26f2a06c8a38478a4f4277c",
}


def run_rollup(path: str) -> str:
    """The rollup for a run directory, looked up BY ITS NAME.

    **An arm cannot be handed another arm's hash without changing the
    path it reads**, which is the whole point of keying on the directory
    rather than on the arm. Two directories sharing a rollup is refused:
    645 files apiece differing by 136 bytes cannot hash the same, so an
    equality here means one entry was pasted twice.
    """
    name = path.rstrip("/").rsplit("/", 1)[-1]
    rollup = RUN_ROLLUPS.get(name, "0" * 64)
    if set(rollup) != {"0"}:
        twins = [other for other, value in RUN_ROLLUPS.items()
                 if other != name and value == rollup]
        if twins:
            raise DriftError(
                f"{name} carries the same rollup as {twins}: two run "
                "directories cannot hash alike, so one entry was pasted "
                "twice -- and a crossed fill computes ordinary contrasts "
                "for the wrong arms."
            )
    return rollup

#: The run the arms' banked means come from. A second pair exists
#: (``0bf86712``, the pre-fix crashes, no log), and naming the live one
#: here means a later reader does not have to re-derive which is which.
LIVE_RUN_SHA8 = "2d54e77a"

#: Where a run directory lives, so a supplied bare name still resolves.
RUNS_ROOT = "/home/user/codex/cleft-aesthetics"

#: The prediction-file stem the loader opens, ``seed_<n>__<stem>.csv``.
#: **Read from ``phase3.write_outputs``**, which writes
#: ``ctx.path(f"{prefix}predictions.csv", tier="CLUSTER-ONLY")`` with
#: every task passing ``prefix=f"seed_{seed}__"`` -- not assumed from the
#: filenames on disk.
PREDICTIONS_STEM = "predictions"


def load_run_directories(path) -> dict[str, str]:
    """``{arm: absolute path}`` from a one-path-per-line file.

    **Matched by STEM, never by line order.** The run-name contract puts
    the config stem first (``run_names``), so ``p25_arm_d2__...`` is D2's
    whatever line it sits on. A file that names an arm twice, or names
    neither, is refused rather than half-read: Phase 22 invented four run
    directory names and all four were wrong, and a half-read file is the
    same failure with a file behind it.
    """
    lines = [
        line.strip().rstrip("/")
        for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    out: dict[str, str] = {}
    for line in lines:
        stem = line.rsplit("/", 1)[-1].split("__", 1)[0]
        if stem not in ARMS:
            raise DriftError(
                f"{line!r} names {stem!r}, which is not a Phase 25 arm "
                f"({sorted(ARMS)}). A contrast config may only declare the "
                "arms the family names."
            )
        if stem in out:
            raise DriftError(
                f"{stem} appears twice in {path}: {out[stem]!r} and {line!r}. "
                "Two directories for one arm means one of them is not the "
                "run that produced its numbers."
            )
        out[stem] = line if line.startswith("/") else f"{RUNS_ROOT}/{line}"
    missing = sorted(set(ARMS) - set(out))
    if missing:
        raise DriftError(
            f"{path} names {sorted(out)} but the family needs {sorted(ARMS)}; "
            f"missing {missing}. All three contrasts run or none do."
        )
    return out

#: The probe's run directory, COPIED byte for byte from the config that
#: already declares it, so the contrast reads the same artifact Phase 22
#: read.
PROBE_RUN_DONOR = "p22_contrasts.yaml"
PROBE_RUN_INPUT = "p7_d1_vit_b16_imagenet_g1"

#: **[FILLED 2026-09-02 from declare_inputs.py]** Keyed by the SET
#: DIRECTORY, which is what the hash is actually of -- so a crossed fill is
#: a mismatch this file can see rather than a silent swap.
#:
#:   dinov2_g1_v1/vit_b14_dinov2__dinov2_lvd142m__g1  2 files, 731,250 bytes
#:   dino_g1_v1/vit_b16_dino__dino_in1k__g1           2 files, 731,243 bytes
ROLLUPS = {
    "dinov2_g1_v1/vit_b14_dinov2__dinov2_lvd142m__g1":
        "b354326677cbc78f39286e43389be1a6eebc0aef300373cb32bff294c20b0edd",
    "dino_g1_v1/vit_b16_dino__dino_in1k__g1":
        "5290095906192da10f7e2313a9f47e8f51a83a264d86634306e7fff599c1d23d",
}

HEAD = """# PHASE 25 -- {title}
#
# phase25.EXIT_CRITERIA (LOCKED 2026-09-02), phase25.THE_TWO_ARMS_RULED.
# {note}
#
# **BLOCKING, AND IT IS NOT CLOSED BY THIS FILE**: the checkpoint tag below
# resolved in timm 1.0.27 under Python 3.13 on the laptop. **The cluster is
# pinned at timm 1.0.7 and nothing about 1.0.7's model index is verifiable
# from here** (phase25.THE_CHECKPOINTS_RESOLVED). If the tag does not
# resolve in the pinned image this config CANNOT run, and the fix is a
# decision about the pin, not an edit here.
#
#   bash -lc "cd /home/user/codex/cleft-aesthetics && PYTHONPATH=src \\
#     python -m cleft.run --config configs/{stem}.yaml"

schema_version: 1
phase: p25
tier: keeper
seed: 1337
"""

EXTRACT_NOTE = {
    "p25_extract_d2_dinov2": (
        "One extraction pass, 237 patients, frozen backbone. **DINOv2 is "
        "518-native**, so timm_kwargs_for passes dynamic_img_size at the "
        "probe's 224 and the position grid interpolates DOWN 37x37 -> 16x16 "
        "-- a declared caveat on the arm, not a hidden step."
    ),
    "p25_extract_d1_dino": (
        "Same timm architecture as the probe (vit_base_patch16_224), "
        "different pretrained tag, native 224 -- so NO size kwargs and no "
        "interpolation. This is what makes it the control."
    ),
}

ARM_NOTE = {
    "p25_arm_d2": (
        "PRIMARY 1 of the three-contrast family. **NOT a clean control**: it "
        "varies objective, patch size and native resolution together "
        "(phase25.THE_TWO_ARMS_RULED)."
    ),
    "p25_arm_d1": (
        "PRIMARY 2, and the SAME-ENCODER control: patch 16, native 224, "
        "self-supervised. Isolates the OBJECTIVE on a fixed encoder. It is "
        "original DINO, not DINOv2 -- a different self-supervised method, and "
        "that limit travels with it."
    ),
}

ARMS = {
    "p25_arm_d2": ("vit_b14_dinov2", "dinov2_lvd142m", "dinov2_g1_v1"),
    "p25_arm_d1": ("vit_b16_dino", "dino_in1k", "dino_g1_v1"),
}
EXTRACTS = {
    "p25_extract_d2_dinov2": ("vit_b14_dinov2", "dinov2_lvd142m", "dinov2_g1_v1"),
    "p25_extract_d1_dino": ("vit_b16_dino", "dino_in1k", "dino_g1_v1"),
}

#: The probe fields the arms copy verbatim. The two NOT here -- backbone and
#: init -- are the factor.
COPIED = (
    "geometry", "label", "max_epochs", "patience", "inner_val_frac",
    "monitor", "seeds", "trainable", "learning_rate",
)


class DriftError(RuntimeError):
    """The generator cannot build a config it can stand behind."""


def _load(stem: str) -> dict:
    return yaml.safe_load(
        (CONFIGS / f"{stem}.yaml").read_text(encoding="utf-8")
    )


def _shared_inputs() -> list[dict]:
    donor = yaml.safe_load((CONFIGS / DONOR).read_text(encoding="utf-8"))
    by_name = {entry["name"]: entry for entry in donor["inputs"]}
    return [by_name["manifest_v1"], by_name["staged_v1"]]


def _render_inputs(entries) -> str:
    lines = ["inputs:"]
    for entry in entries:
        rollup = entry["rollup_sha256"]
        quoted = f'"{rollup}"' if set(rollup) == {"0"} else rollup
        lines += [
            f"- name: {entry['name']}",
            f"  path: {entry['path']}",
            f"  rollup_sha256: {quoted}",
        ]
    return "\n".join(lines) + "\n"


def _set_directory(out_version: str, backbone: str, init: str) -> str:
    return f"{out_version}/{backbone}__{init}__g1"


def build() -> tuple[dict[str, str], list[str]]:
    """``{stem: text}`` and the placeholders remaining."""
    shared = _shared_inputs()
    probe = _load(PROBE)["task"]
    texts, placeholders = {}, []

    for stem, (backbone, init, out_version) in EXTRACTS.items():
        head = HEAD.format(title=_title(stem), note=EXTRACT_NOTE[stem], stem=stem)
        texts[stem] = (
            head + _render_inputs(shared) + "\n"
            "task:\n"
            "  kind: extract_embeddings\n"
            "  manifest_artifact: manifest_v1\n"
            "  staged_artifact: staged_v1\n"
            f"  out_version: {out_version}\n"
            "  batch_size: 32\n"
            "\n"
            "  # ONE set. The arm varies the BACKBONE and nothing else, so the\n"
            "  # extraction that feeds it varies the backbone and nothing else.\n"
            "  sets:\n"
            f"    - backbone: {backbone}\n"
            f"      init: {init}\n"
            "      geometry: g1\n"
        )

    for stem, (backbone, init, out_version) in ARMS.items():
        directory = _set_directory(out_version, backbone, init)
        rollup = ROLLUPS.get(directory, "0" * 64)
        if set(rollup) == {"0"}:
            placeholders.append(stem)
        # **The crossed-fill guard.** The hash is looked up BY DIRECTORY, so
        # the only way to give an arm the wrong one is to change the path --
        # which changes what the arm reads, visibly.
        for other, other_hash in ROLLUPS.items():
            if other != directory and other_hash == rollup:
                raise DriftError(
                    f"{stem} would carry the rollup of {other}: the two arms "
                    "share a hash, which means one is reading the other's "
                    "features and no downstream check would notice."
                )
        entries = shared + [{
            "name": "embeddings",
            "path": f"/home/user/codex/cleft-aesthetics/data/embeddings/{directory}",
            "rollup_sha256": rollup,
        }]
        head = HEAD.format(title=_title(stem), note=ARM_NOTE[stem], stem=stem)
        body = ["task:", "  kind: train_cv",
                "  manifest_artifact: manifest_v1",
                "  staged_artifact: staged_v1",
                "  embeddings_artifact: embeddings", "",
                "  # Every field below is COPIED from the probe's own",
                f"  # config (configs/{PROBE}.yaml) rather than retyped.",
                "  #",
                "  # [CORRECTED 2026-09-06] This read \"EVERY field below is",
                "  # the probe's own ... so one factor is true by",
                "  # construction rather than by intention\". Two recipe",
                "  # fields are NOT copied and are not below: weight_decay,",
                "  # which takes a schema default of 0.01 and so matches the",
                "  # probe, and batch_size, which takes a schema default of",
                "  # 16 against the probe's 32. The values are left as they",
                "  # ran, because a config must keep describing the run that",
                "  # produced its figures. The divergence is INERT here: the",
                "  # head is fit full batch and batch_size is never passed",
                "  # to EmbeddingHeadBackbone. See phase25.PHASE_25_CLOSING",
                "  # criterion 2 and its dated correction."]
        for field in COPIED:
            if field not in probe:
                raise DriftError(
                    f"the probe's config has no {field!r}: the arms copy it, "
                    "so a missing field means the recipe moved and the "
                    "one-factor claim is no longer verifiable here."
                )
            value = probe[field]
            if isinstance(value, list):
                body.append(f"  {field}:")
                body += [f"  - {item}" for item in value]
            else:
                body.append(f"  {field}: {value}")
        body += ["", "  # The ONE factor.",
                 f"  backbone: {backbone}", f"  init: {init}"]
        texts[stem] = (
            head
            + "# **ONE PLACEHOLDER: `embeddings`.** The set does not exist until\n"
            "# the extraction config above runs, so its rollup CANNOT be known\n"
            "# here. Fill from scripts/declare_inputs.py after the extraction,\n"
            "# never from a chat log.\n"
            if set(rollup) == {"0"} else
            head
            + "# **NO PLACEHOLDER.** `embeddings` was filled 2026-09-02 from\n"
            "# declare_inputs.py after the extraction ran. The hash is keyed\n"
            "# in the generator BY SET DIRECTORY, so a crossed fill -- each\n"
            "# arm on the other's features -- is a mismatch the generator\n"
            "# refuses rather than a swap nothing downstream would notice.\n"
        )
        texts[stem] += _render_inputs(entries) + "\n" + "\n".join(body) + "\n"

    contrasts = _contrasts(shared[0])
    if contrasts is not None:
        texts["p25_contrasts"] = contrasts
        # The two arm rollups pend on declare_inputs; the paths do not.
        # **Report what is ACTUALLY unfilled**, not what once was: the
        # rollups are looked up by run directory, so this asks each one.
        placeholders.extend(
            f"p25_contrasts:{arm}"
            for arm, path in sorted(RUN_DIRECTORIES.items())
            if set(run_rollup(path)) == {"0"}
        )
    return texts, placeholders


def owed() -> list[str]:
    """What the contrasts config waits on. Reported, never guessed."""
    if RUN_DIRECTORIES:
        return []
    return [
        f"{name}: run directory path + rollup_sha256 "
        "(<config-stem>__<sha8>__<job-id>; neither is knowable here)"
        for name in ("p25_arm_d2", "p25_arm_d1")
    ]


def _contrasts(manifest: dict) -> str | None:
    """The three-contrast config. **The pair list is NOT in it** -- it comes
    from ``phase25.contrast_family()``, so the family cannot grow here."""
    donor = yaml.safe_load(
        (CONFIGS / PROBE_RUN_DONOR).read_text(encoding="utf-8")
    )
    probe = next(
        entry for entry in donor["inputs"]
        if entry["name"] == PROBE_RUN_INPUT
    )
    if not RUN_DIRECTORIES:
        return None
    entries = [manifest, probe] + [
        {"name": name, "path": path, "rollup_sha256": run_rollup(path)}
        for name, path in RUN_DIRECTORIES.items()
    ]
    head = HEAD.format(
        title="THE THREE CONTRASTS",
        note=(
            "Two primaries (each arm vs the 0.2520 probe) and one secondary "
            "(D2 vs D1), from phase25.contrast_family() -- **the pair list "
            "is not a field here**. phase7b.paired_comparison UNCHANGED, "
            "full criterion, BOTH conditions recorded."
        ),
        stem="p25_contrasts",
    )
    seeds = "\n".join(f"  - {s}" for s in (1337, 2024, 7, 99, 12345))
    arms = []
    for name, input_name in (
        ("probe", PROBE_RUN_INPUT),
        ("p25_arm_d2", "p25_arm_d2"),
        ("p25_arm_d1", "p25_arm_d1"),
    ):
        arms += [
            f"  - name: {name}",
            f"    input: {input_name}",
            "    seeds:",
            *[f"    - {s}" for s in (1337, 2024, 7, 99, 12345)],
            # The stem the loader opens: phase3.write_outputs writes
            # <prefix>predictions.csv and every task passes
            # prefix=f"seed_{seed}__". READ from the writer, not assumed.
            f"    csv: {PREDICTIONS_STEM}",
            "    n_patients: 237",
        ]
    return (
        head
        + "# **NO PLACEHOLDERS.** Both arm rollups were filled 2026-09-02 from\n"
        "# declare_inputs.py. They are keyed in the generator BY RUN\n"
        "# DIRECTORY NAME, so a crossed fill -- each contrast reading the\n"
        "# other arm -- is a mismatch the generator refuses rather than a\n"
        "# swap that would compute ordinary numbers for the wrong arms.\n"
        "# The two runs hold 645 files each and differ by 136 bytes.\n"
        "# Their PATHS\n"
        "# came from --run-dirs (the names are the cluster's and are not\n"
        "# derivable here -- run_names records that the obvious job-id rule\n"
        "# holds for 22 of 83 directories and fails for 61). Fill the two\n"
        "# rollups from declare_inputs.py, never from a chat log.\n"
        + _render_inputs(entries) + "\n"
        "task:\n"
        "  kind: p25_contrasts\n"
        "  manifest_artifact: manifest_v1\n"
        "\n"
        "  # The three contrasts are phase25.contrast_family()'s, not this\n"
        "  # file's. What the config declares is which arms are readable.\n"
        "  arms:\n"
        + "\n".join(arms) + "\n"
        "\n"
        "  seeds:\n" + seeds + "\n"
        "  n_boot: 10000\n"
        "  expect_contrasts: 3\n"
        "  expect_patients: 237\n"
    )


def _title(stem: str) -> str:
    return {
        "p25_extract_d2_dinov2": "EXTRACT D2 (DINOv2-B/14)",
        "p25_extract_d1_dino": "EXTRACT D1 (DINO-B/16, the same-encoder control)",
        "p25_arm_d2": "ARM D2 (DINOv2-B/14) vs the 0.2520 probe",
        "p25_arm_d1": "ARM D1 (DINO-B/16) vs the 0.2520 probe",
        "p25_contrasts": "THE THREE CONTRASTS",
    }[stem]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--run-dirs",
        help=(
            "file listing the two arm run directories, one per line "
            "(the p7d_arm_dirs.txt precedent). Their rollups stay "
            "all-zero: declare_inputs fills those, never this script."
        ),
    )
    args = parser.parse_args()

    if args.run_dirs:
        supplied = load_run_directories(args.run_dirs)
        for arm, path in supplied.items():
            known = RUN_DIRECTORIES.get(arm)
            if known is not None and known != path:
                raise DriftError(
                    f"{arm}: the file names {path!r} but this generator "
                    f"has {known!r} baked in. **A supplied file may "
                    "confirm the baked name, never silently replace it** "
                    "-- two directories for one arm means one of them is "
                    "not the run that produced its numbers. Edit the "
                    "generator deliberately if the run changed."
                )
            RUN_DIRECTORIES[arm] = path

    texts, placeholders = build()
    for item in owed():
        print("OWED -- reported, not guessed:", item)
    if args.check:
        drifted = [
            stem for stem, text in sorted(texts.items())
            if (CONFIGS / f"{stem}.yaml").read_text(encoding="utf-8") != text
        ]
        if drifted:
            print("DRIFT:", ", ".join(drifted))
            return 1
        print(
            f"{len(texts)} configs match the generator "
            f"({len(placeholders)} placeholder(s): {placeholders or 'none'})"
        )
        return 0
    for stem, text in sorted(texts.items()):
        (CONFIGS / f"{stem}.yaml").write_text(text, encoding="utf-8")
    print(
        f"wrote {len(texts)} configs; "
        f"{len(placeholders)} placeholder(s): {placeholders or 'none'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
