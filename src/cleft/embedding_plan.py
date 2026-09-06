"""Which embedding sets Phase 7 actually needs, derived from the arm list.

[Instruction, 2026-07-31]: derive the required sets from the arm list rather
than extracting the full cross product -- the scheme axis multiplied it well
past the 24 originally planned. Two structural facts do most of the work:

* **feature-map extraction is per-CHECKPOINT, not per-scheme-at-extraction**:
  the map is the frozen boundary and region schemes are applied at cleft-train
  time. But a graph checkpoint fine-tuned under a scheme has scheme-shaped
  WEIGHTS, and the consistency rule (same scheme in pretraining and cleft
  fine-tuning) means a scheme arm needs the map from its own scheme's
  checkpoint. So a graph set is (backbone, init, geometry, pretrain_scheme) --
  and only the arms that vary scheme multiply.
* **masked inits are geometry-bound** (``embeddings.check_init_geometry``), so
  the init x geometry grid is never the full product.

The derived total is 16 sets against a naive cross product of ~60, consuming
12 of the 30 pretraining checkpoints.

----------------------------------------------------------------------------
THE ARM LIST, AS DATA
----------------------------------------------------------------------------
Mirrors PLAN Part 6. Entries tagged [PLAN-provisional] are placements the
Part 6 table does not pin and are the maintainer's to confirm; each records its
reasoning so confirming or overriding is one decision, not an excavation:

* **Stage D (init ladder, 12 arms)**: every backbone x every init, at
  ``LADDER_GEOMETRY``.
* **Stage C (geometry, the mechanism test)**: BOTH transformers at
  ``GEOMETRY_STAGE_INIT``, both geometries -- the tiling prior predicts ViT
  separates and Swin does not, stated before the runs.
* **Stage E (scheme, 3 arms)**: SR-GNN at G2 on the masked init, schemes
  {grid, anatomy, random} against the native-scheme ladder arm. Adds the
  three scheme-matched checkpoint maps.
* **Stages B (view) and G (label)** reuse Stage D's sets: they vary the view
  and the label formulation, not the representation.
* **Stage A baselines** consume no pretrained embeddings.

**[MEASURED 2026-07-31 context for Stage E]** The scheme axis is NULL at
pretraining (``pretrain.SCHEME_AXIS_AT_PRETRAINING``) -- which is exactly why
Stage E stays live at cleft time and why ``CANONICAL_GRAPH_SCHEME`` is a free
choice rather than a fitted one.
"""

from __future__ import annotations

from .embeddings import GEOMETRIES, INITS, expected_variant
from .models.factory import BACKBONES, LADDER_BACKBONES

#: [DECIDED 2026-08-01, confirmed] The geometry the init ladder runs at. G2:
#: the scheme comparison is only clean there (PLAN §4.4), and ViT's 0.041 G1
#: penalty has a candidate mechanism -- running the ladder at G1 would
#: disadvantage one backbone for a geometry reason while INIT is the axis
#: under test. Stage C carries the geometry question separately.
LADDER_GEOMETRY = "g2"

#: [DECIDED 2026-08-01] Stage C runs on BOTH transformers, and the reason is
#: methodological: choosing ViT alone because pretraining showed the largest
#: gap would be selecting on the outcome. The tiling mechanism is a PRIOR --
#: fixed 16x16 patches meeting G1's high-contrast white corners -- and it
#: PREDICTS, stated before the cleft runs: **ViT separates, Swin does not**
#: (shifted windows are less exposed; Swin's pretraining indifference was
#: 0.8718 against 0.8719). Running both makes Stage C a mechanism test
#: rather than a pick. On the masked init: the geometry arm compares
#: matched-G1 against matched-G2 pipelines (the twelve-not-eight reasoning).
GEOMETRY_STAGE_BACKBONES = ("vit_b16", "swin_b")
GEOMETRY_STAGE_INIT = "scut_masked"

#: [DECIDED 2026-07-31; confirmed 2026-08-01] The scheme whose checkpoints
#: feed every graph arm that does NOT vary scheme. "native" -- the
#: architecture's own generator -- and the choice is free because the scheme
#: axis measured null at pretraining; picking a generated scheme here would
#: be a choice with no measured basis wearing one. **The thirds grid raised
#: at supervision is tested explicitly in Stage E**, so native elsewhere does not sideline the
#: prescription -- it keeps the prescription's test clean.
CANONICAL_GRAPH_SCHEME = "native"

#: How a Phase 7B pooling set reduces a block's token sequence to a vector.
#: ``cls`` takes the class token; ``mean_patch`` averages the patch tokens,
#: excluding every prefix token (timm's ``num_prefix_tokens``, not a
#: hard-coded 1 -- a registers-carrying ViT has more than one).
POOLING_TOKENS = ("cls", "mean_patch")

#: Stage E: the cleft-time scheme comparison, G2 only (PLAN §4.4).
STAGE_E = {
    "backbone": "srgnn",
    "init": "scut_masked",
    "geometry": "g2",
    "schemes": ("grid", "anatomy", "random"),
}


class PlanError(RuntimeError):
    """The embedding plan cannot be derived."""


def _scheme_for(backbone: str, init: str, scheme: str | None = None) -> str | None:
    """The pretrain_scheme a set carries. None where no checkpoint exists
    (imagenet) or no scheme axis exists (transformers)."""
    if BACKBONES[backbone]["kind"] != "graph" or init == "imagenet":
        return None
    return scheme or CANONICAL_GRAPH_SCHEME


def _set(backbone: str, init: str, geometry: str, scheme: str | None = None) -> dict:
    if backbone not in BACKBONES:
        raise PlanError(f"unknown backbone {backbone!r}")
    if init not in INITS or geometry not in GEOMETRIES:
        raise PlanError(f"unknown init/geometry {init!r}/{geometry!r}")
    return {
        "backbone": backbone,
        "backbone_kind": BACKBONES[backbone]["kind"],
        "init": init,
        "geometry": geometry,
        "variant": expected_variant(init, geometry),
        "pretrain_scheme": _scheme_for(backbone, init, scheme),
    }


def _key(entry: dict) -> tuple:
    return (
        entry["backbone"], entry["init"], entry["geometry"],
        entry["pretrain_scheme"],
    )


def required_sets() -> list[dict]:
    """Every embedding set the arm list consumes, deduplicated, in a stable
    order. This is the extraction's work list -- the shipped extraction
    config must enumerate exactly these, asserted by test."""
    sets: dict[tuple, dict] = {}

    # Stage D: the init ladder.
    for backbone in LADDER_BACKBONES:
        for init in INITS:
            entry = _set(backbone, init, LADDER_GEOMETRY)
            sets[_key(entry)] = entry

    # Stage C: geometry, both sides, both transformers (the mechanism test).
    for backbone in GEOMETRY_STAGE_BACKBONES:
        for geometry in GEOMETRIES:
            entry = _set(backbone, GEOMETRY_STAGE_INIT, geometry)
            sets[_key(entry)] = entry

    # Stage E: scheme-matched checkpoints for the cleft scheme arms.
    for scheme in STAGE_E["schemes"]:
        entry = _set(
            STAGE_E["backbone"], STAGE_E["init"], STAGE_E["geometry"], scheme
        )
        sets[_key(entry)] = entry

    return [sets[key] for key in sorted(sets, key=lambda k: tuple(map(str, k)))]


def set_name(entry: dict) -> str:
    """The artifact directory name for one set. Says everything the set is.

    A per-fold BN-re-extracted set carries ``adabn_fold<N>`` -- it is a
    DIFFERENT representation from the plain set of the same backbone/init/
    geometry, valid for exactly one fold, and the name has to say so or two
    incompatible artifacts sit side by side looking interchangeable
    (``extract.PER_FOLD_REEXTRACTION``).

    **A Phase 7B pooling set carries ``block<N>_<token>`` for the same
    reason.** Two sets from one checkpoint at different depths are different
    representations, and a trial's provenance has to name which one it used --
    otherwise twelve directories differ only in their contents, which is the
    mutable-data-directory failure PLAN Part 1 opens with.
    """
    scheme = entry["pretrain_scheme"]
    suffix = f"__{scheme}" if scheme else ""

    block, token = entry.get("block"), entry.get("token")
    if (block is None) != (token is None):
        raise PlanError(
            "block and token are declared together or not at all; a depth "
            "without a pooling rule does not name a representation"
        )
    if block is not None:
        if token not in POOLING_TOKENS:
            raise PlanError(
                f"unknown token {token!r}; expected one of {POOLING_TOKENS}"
            )
        suffix += f"__block{int(block)}_{token}"

    if entry.get("fold") is not None:
        suffix += f"__adabn_fold{int(entry['fold'])}"
    # **A randomised-backbone CONTROL says so in its own name.** Its metadata
    # already carries the mark and ``check_pairing`` refuses it for any
    # consumer that did not ask -- but a directory name that reads like the
    # real set is a hazard on every ``ls``, in every config, and in every
    # declaration a human types. ``extract.RANDOMISED_BACKBONE``.
    if entry.get("randomise"):
        suffix += "__randomised"
    return f"{entry['backbone']}__{entry['init']}__{entry['geometry']}{suffix}"


def required_checkpoints() -> list[dict]:
    """The distinct pretraining runs the plan consumes, with their config
    names -- every one must be producible by a shipped config, which the
    tests assert against configs/."""
    checkpoints: dict[tuple, dict] = {}
    for entry in required_sets():
        if entry["variant"] is None:
            continue
        scheme = entry["pretrain_scheme"]
        key = (entry["backbone"], entry["variant"], scheme)
        token = "" if scheme is None else f"{scheme}_"
        checkpoints[key] = {
            "backbone": entry["backbone"],
            "variant": entry["variant"],
            "pretrain_scheme": scheme,
            "config": f"p6_pretrain_{entry['backbone']}_{token}{entry['variant']}.yaml",
        }
    return [
        checkpoints[key]
        for key in sorted(checkpoints, key=lambda k: tuple(map(str, k)))
    ]


def summary() -> dict:
    """The plan's arithmetic, visible: what is needed against the naive
    product the instruction warned about."""
    sets = required_sets()
    naive_transformer = 2 * len(INITS) * len(GEOMETRIES)
    # Naive graph: every init x geometry x every scheme option per backbone
    # (native + the three generated), imagenet scheme-free.
    naive_graph = 2 * len(GEOMETRIES) * (1 + 2 * 4)
    return {
        "required_sets": len(sets),
        "required_checkpoints": len(required_checkpoints()),
        "of_pretraining_runs": 30,
        "naive_cross_product": naive_transformer + naive_graph,
        "ladder_geometry": LADDER_GEOMETRY,
        "canonical_graph_scheme": CANONICAL_GRAPH_SCHEME,
        "note": (
            "feature-map extraction is per-checkpoint; region schemes are "
            "applied at cleft-train time, so only the arms that VARY scheme "
            "(Stage E) multiply the sets -- and they multiply by "
            "scheme-matched checkpoints, per the consistency rule."
        ),
    }
