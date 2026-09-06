"""The Phase 7 arm list, as data — so the lattice can be checked, not trusted.

Every comparison the ladder makes is a pair of arms differing in ONE field.
Writing the arms out by hand and asserting that afterwards would be checking a
list against itself; deriving them from the stages that define them, and
asserting the one-factor property over the derived set, is a check with
something to say.

----------------------------------------------------------------------------
SEEDS ARE SET BY REGIME, NOT BY STAGE
----------------------------------------------------------------------------
[DECIDED 2026-08-01] Graph arms run **ten** seeds, transformer arms **five**.

The rule is the regime, not the stage: the frozen probe measured SD 0.0137
(five-seed threshold 0.017) against 0.0251 for the graph regime. A graph delta
smaller than 0.031 is unresolvable at five seeds — and for Stage E, with the
scheme axis already null three times, deltas inside 0.031 are *foreseeable
rather than contingent*, so five seeds would report "unresolved" by
construction rather than by measurement. The same argument applies to Q1/Q2 on
the graph backbones.

Ten seeds cost minutes over precomputed features. The brief's "ten only where
a delta lands between the thresholds" is preserved in spirit — this is the
case where it lands there predictably.

**Keyed on the regime so a new graph arm inherits it.** Writing "10" beside
four named arms is how the next one gets five by omission.

> **[MEASURED 2026-08-02] 0.0251 IS ONE ARM'S SD, AND THIS SECTION QUOTED IT
> AS THE REGIME'S.** It is SR-GNN's ``scut_masked`` arm — and now that all six
> Stage D graph arms have reported, the *narrowest of its own backbone's
> three*. The six span **0.0251 to 0.0567**, so what ten seeds resolve is not
> one number: 0.0220 at that narrowest arm and 0.0497 at AG-Net's imagenet
> arm. The measured Stage D thresholds bear it out — SR-GNN's Q2 came out at
> 0.0294 and AG-Net's Q1 at 0.0451, twice apart. The "five-seed 0.031,
> ten-seed 0.022" parenthesis was DELETED above rather than repaired: it
> describes one arm and was written as though it described a regime, which is
> the inherited-band error PLAN §4.12.1 forbids by name.
>
> **Nothing measured is affected.** Every Stage D threshold was derived per
> comparison from the two arms' own SDs (``STAGE_D_AT_G2``), never from this
> figure. What it affected was PLANNING — read as the band, it makes ten seeds
> look about twice as sharp as they are.
>
> **[DECIDED 2026-08-02] Ten stands, and Stage D1's counts do not rise.**
> Resolving AG-Net's own observed Q1 of 0.0309 from its own two SDs needs
> **n = 22**; a round 0.03 needs 23. Running 22 seeds to reach a threshold is
> chasing the threshold rather than answering the question, and the claim
> criterion exists precisely so that **"unresolved" is an available answer**
> (PLAN §4.3, brief §5). A delta landing inside the band is reported as
> unresolved — a finding about what this data supports, not a gap to close.

----------------------------------------------------------------------------
CELLS AND RUNS ARE DIFFERENT COUNTS
----------------------------------------------------------------------------
A stage's *cells* are the comparison it makes; its *runs* are what has to
execute. Stage C is four cells and two runs, because its masked-G2 cells ARE
Stage D arms — same backbone, init, geometry, label and regime, so running
them again would be two draws of one thing rather than a comparison.

``reuses`` records that explicitly. ``distinct_runs`` counts what the cluster
does; ``cells`` counts what the summary table reports. Confusing the two is
how an arm list ends up promising more than it ran.
"""

from __future__ import annotations

from . import embedding_plan
from .embeddings import INITS
from .models.factory import BACKBONES, LADDER_BACKBONES

#: Every ladder arm sits at G2 (PLAN §4.4; ViT's 0.041 G1 penalty has a
#: mechanism), except Stage C, whose whole question is the geometry.
LADDER_GEOMETRY = "g2"

#: The graph backbones' scheme everywhere except Stage E, whose question it is.
CANONICAL_SCHEME = "native"

#: Seeds by regime. See the module docstring: the band decides, not the stage.
SEEDS_BY_KIND = {"transformer": 5, "graph": 10}

#: The canonical seed order, shared with the Phase 6 graph seed band. A
#: five-seed arm takes the first five and a ten-seed arm all ten, so the two
#: share a spine and a five-seed arm's seeds are a subset of a ten-seed one's
#: -- which makes the pair eyeballable and means a later extension from five
#: to ten reuses the five already run rather than replacing them.
SEED_POOL = (1337, 2024, 7, 99, 12345, 42, 271828, 314159, 161803, 777)


def seeds_for(kind: str) -> list[int]:
    count = SEEDS_BY_KIND[kind]
    if count > len(SEED_POOL):
        raise LadderError(
            f"{kind} needs {count} seeds and the pool holds {len(SEED_POOL)}"
        )
    return list(SEED_POOL[:count])

#: Stage C: the mechanism test. The tiling account predicts ViT separates at
#: G1 and Swin does not — stated before the runs, so this is a test rather
#: than an observation. Running ViT alone would be selecting on the outcome.
GEOMETRY_BACKBONES = ("vit_b16", "swin_b")
GEOMETRY_INIT = "scut_masked"

#: Stage E: the scheme question, in the TRAINED-GRAPH-LAYER regime. The probe
#: cannot answer it — message passing is the thing being compared.
SCHEME_BACKBONE = "srgnn"
SCHEMES = ("native", "grid", "anatomy", "random")

#: Stage G: label formulation on one representation. The base arm is a Stage D
#: arm, so the `mean` cell reuses it.
LABEL_BACKBONE = "vit_b16"
LABEL_INIT = "scut_masked"
LABELS = ("mean", "median", "ldl")

#: **[DECIDED 2026-08-02] Stage G1 -- label formulation AT G1, on the
#: ``imagenet`` init rather than Stage G's ``scut_masked``.**
#:
#: **Why at G1 at all.** Q1 and Q2 both changed sign or magnitude between the
#: geometries (``STAGE_D1_AT_G1``), so an answer measured only at G2 carries
#: exactly the exposure the init ladder just demonstrated. The label question
#: gets asked at both operating points for the same reason the init question
#: did.
#:
#: **Why not ``scut_masked``, which would hold init constant.** That is ViT's
#: weakest cell at G1 -- 0.0830, the lowest of the twelve, with the widest SD
#: of the three candidates at 0.0247. A label triple there could claim nothing
#: smaller than **0.0306, which is 36.9% of everything the arm achieves**. An
#: instrument that can only return "unresolved" is not a control.
#:
#: The comparison, computed rather than argued (all assuming a triple shares
#: its base arm's SD):
#:
#: =========================  ======  ======  =========  ==============
#: base arm                    base      sd  threshold   % of the arm
#: =========================  ======  ======  =========  ==============
#: ``scut_masked`` g1         0.0830  0.0247     0.0306          36.9%
#: ``imagenet`` g1            0.2520  0.0148     0.0183           7.3%
#: ``scut_masked`` g2 (G)     0.2001  0.0142     0.0176           8.8%
#: =========================  ======  ======  =========  ==============
#:
#: **``imagenet`` at G1 is also the better MATCH to the existing G2 triple**
#: -- 7.3% against 8.8%, where ``scut_masked`` at G1 would be 36.9%. Holding
#: init constant would buy nominal comparability between two triples with
#: wildly different power to detect the same effect, and matching SENSITIVITY
#: is the comparability that matters when the question is whether an answer
#: holds at both operating points.
#:
#: **What it costs, and it goes in the write-up.** The two triples differ in
#: geometry AND init, so there is no clean label x geometry comparison in the
#: ladder: agreement between them is reassurance, disagreement would be
#: unattributable. The disambiguating follow-up, if it is ever needed, is a
#: ``vit_b16 imagenet g2`` label pair -- two arms, whose ``mean`` cell already
#: exists as ``p7_d_vit_b16_imagenet_g2``. Not pre-bought against an outcome
#: that has not happened.
#:
#: **Costs no extraction**: ``vit_b16__imagenet__g1`` is already in
#: ``embeddings_g1_ladder_v1`` and declared, and the ``mean`` cell IS the
#: Stage D1 arm, so this is three cells and two runs.
LABEL_CONTROL_INIT = "imagenet"
LABEL_CONTROL_GEOMETRY = "g1"

#: **[DECIDED 2026-08-02] Stage G0 -- the label triple at ``imagenet`` and
#: **G2**, bought because Stage G and Stage G1 disagreed.**
#:
#: The disagreement was predicted when Stage G1's init was chosen, and it
#: happened: label is NULL at imagenet/G1 and CLAIMABLY NEGATIVE at
#: masked/G2. Those two cells differ in geometry AND init, so the
#: disagreement is unattributable as it stands -- the exact cost recorded in
#: ``LABEL_CONTROL_INIT`` and the exact follow-up named there.
#:
#: **It completes three corners of a 2x2**::
#:
#:                imagenet              scut_masked
#:     g1         NULL (Stage G1)       not run -- see below
#:     g2         Stage G0 (this)       CLAIMABLE (Stage G)
#:
#: With init held at ``imagenet`` across the two geometries, a label effect
#: appearing at G2 attributes the disagreement to **geometry**; its absence
#: there attributes it to **init**, since the G2 row would then carry the
#: effect only at ``scut_masked``. Either way one edge of the square is
#: resolved.
#:
#: **The fourth corner is deliberately not run.** ``scut_masked`` at G1 is
#: ViT's weakest cell (0.0830, SD 0.0247), where the smallest claimable label
#: delta is 36.9% of the arm's own value -- it can only return "unresolved",
#: so it would add a cell without adding information. Three corners is what
#: this data supports.
#:
#: **The instrument is weaker, and the sensitivity is reported WITH the
#: result rather than after it.** Base 0.1347, SD 0.0193, so a label delta
#: must clear ~0.0239 -- **17.8% of the arm**, against 7.3% at Stage G1 and
#: 8.8% at Stage G. But relative sensitivity is not the question; whether it
#: can detect the effects it must attribute is. Using each label arm's own SD
#: as measured at masked/G2:
#:
#:     median: detectable >= 0.0198, effect to attribute 0.0595 -- **3.0x**
#:     ldl:    detectable >= 0.0262, effect to attribute 0.0418 -- **1.6x**
#:
#: So a null here is informative rather than merely quiet, and the numbers
#: above are what makes it so. **They go in the write-up beside the result**,
#: because a null reported without its detectable effect size is
#: indistinguishable from an underpowered arm.
#:
#: **Costs no extraction**: ``vit_b16__imagenet__g2`` is in ``embeddings_v1``
#: and declared, and the ``mean`` cell IS ``p7_d_vit_b16_imagenet_g2`` at
#: 0.1347 -- three cells, two runs.
LABEL_GEOMETRY_CONTROL_GEOMETRY = "g2"

#: **Stage E0 [DECIDED 2026-08-01]: the scheme comparison with the checkpoint
#: held constant.**
#:
#: Stage E varies scheme AND checkpoint -- the consistency rule gives each
#: scheme its own scheme-matched checkpoint. That was licensed by the scheme
#: axis measuring null AT PRETRAINING, and that licence does not hold: SCUT
#: test PCC is beauty prediction on 2,199 faces and says nothing about
#: transfer to 237 cleft images. Stage D shows the transfer gap is large --
#: SR-GNN native runs 0.0674 to 0.1507 across inits, comparable to the whole
#: scheme separation of 0.095. So Stage E's ordering could be placement or
#: transfer quality, and nothing separates them.
#:
#: **ImageNet has no checkpoint at all**, so the four schemes there share one
#: scheme-free embedding set and differ in placement alone. That isolates the
#: question at the cost of seed-initialised rather than warm-started graph
#: layers -- a different question about the same axis, and the only one
#: available without breaking the consistency rule.
#:
#: Costs no new extraction: ``srgnn__imagenet__g2`` already exists and is
#: hashed, and ``check_pairing`` permits any scheme on an imagenet init
#: because there is no pretraining structure to be consistent with.
SCHEME_CONTROL_INIT = "imagenet"

#: **Stage C0 [DECIDED 2026-08-01]: geometry with the checkpoint held
#: constant, on a PRETRAINED init.**
#:
#: Stage C's cells consume masked_g1 and masked_g2 -- different checkpoints,
#: so it measures a matched PIPELINE and cannot test the tiling prediction.
#: ``scut_original`` is NOT geometry-bound: one ``original`` checkpoint serves
#: both geometries (``embeddings.VARIANT_FOR_INIT``), so G1-vs-G2 there varies
#: geometry alone on pretrained weights.
#:
#: That is what distinguishes a genuine geometry x init interaction from
#: ``masked_g1`` simply being the weakest checkpoint of the twelve (0.7893 at
#: pretraining against masked_g2's 0.8306). The ImageNet pair already gives
#: pure geometry at ImageNet weights; this gives it at pretrained weights, and
#: the two together say whether the effect depends on the init.
#:
#: **Needs an extraction this project has not run**: ``vit_b16__scut_original__g1``
#: is not in ``embeddings_v1``, because the plan only ever derived G1 sets for
#: Stage C's masked cells.
GEOMETRY_CONTROL_BACKBONE = "vit_b16"
GEOMETRY_CONTROL_INIT = "scut_original"

#: The artifact version each arm reads. G1 sets outside Stage C's two do not
#: exist in ``embeddings_v1`` and are extracted into their own versions,
#: because data artifacts are immutable (PLAN §2.6) -- a set cannot be added
#: to a published directory.
DEFAULT_EMBEDDINGS_VERSION = "embeddings_v1"
G1_CONTROL_EMBEDDINGS_VERSION = "embeddings_g1_control_v1"

#: **[DECIDED 2026-08-02] Stage D1's nine sets need a THIRD version, and the
#: reason is mechanical rather than stylistic.**
#:
#: ``embeddings_g1_control_v1`` was built on 2026-08-01 holding exactly one
#: set (``DECLARED_EMBEDDING_PROVENANCE``), Stage C0 consumed it, and its
#: rollup is verified. ``run.task_extract_embeddings`` refuses an
#: ``out_version`` whose directory already exists -- the same immutability
#: rule that put the C0 set outside ``embeddings_v1`` in the first place. So
#: the nine cannot be dropped in beside it; the guard would abort the
#: extraction before the first set was written.
#:
#: The routing here previously sent every G1 set that was not Stage C's to the
#: control version, which was correct while C0's was the only one. Stage D1
#: made it wrong, and it would have surfaced as an aborted cluster run rather
#: than as a laptop-side failure -- ``embeddings_version_for`` derives the
#: answer now instead of restating the rule.
G1_LADDER_EMBEDDINGS_VERSION = "embeddings_g1_ladder_v1"

#: The single set ``embeddings_g1_control_v1`` holds. Named rather than
#: derived from a stage, because a stage is a comparison and a version is a
#: batch of extraction: Stage D1's cell for this arm REUSES the C0 run, so
#: routing by stage would give one run two artifact versions.
G1_CONTROL_SET = "vit_b16__scut_original__g1"


#: **[VERIFIED 2026-08-01, on the cluster] The ladder's embedding-set rollups,
#: recorded ONCE and keyed by set name.**
#:
#: Obtained by running ``scripts/declare_ladder_inputs.py`` on the cluster and
#: pasting its output here. All sixteen live under
#: ``data/embeddings/embeddings_v1/``, which the generator prefixes.
#:
#: **Why a registry rather than nineteen hand-edits.** A path is immutable, so
#: one hash serves every config declaring it -- and ``vit_b16__scut_masked__g2``
#: is declared by three. Pasting per config makes the cross-config invariant a
#: thing to remember; recording per path makes it structural, and the one
#: observed failure of that invariant was exactly a sibling forgotten.
#:
#: **This does not weaken guard 3.** ``declare_inputs.py`` refuses to write
#: hashes because a tool that HASHES AND WRITES in one step makes the guard
#: agree with reality by definition. Nothing here hashes: these values were
#: obtained by a deliberate act on the machine that holds the data, read, and
#: written down once. The generator only propagates a recorded value into
#: derived files -- which is why the configs stay generated output rather than
#: becoming edited files.
#:
#: ``srgnn__scut_masked__g2__native`` is deliberately ABSENT: it was already
#: verified for the Phase 6 seed band and is read from that config, so
#: recording it again would be a second copy of one fact. The generator
#: asserts the two sources agree wherever they overlap.
DECLARED_EMBEDDING_HASHES = {
    "agnet__imagenet__g2":
        "da58c17472fd760dea9428b03a1add46663b52ce4e3e6094a76034bf6307d6cb",
    "agnet__scut_masked__g2__native":
        "3b802f9a06a2dece6e5631ad9d56bb9146c73482a2d6070b3a23b37f8396c64b",
    "agnet__scut_original__g2__native":
        "e87b3221a2bad94b4bce3e0bd90045ded178043785e6a86e610fb94c501c6b33",
    "srgnn__imagenet__g2":
        "b2cf61544896e9003b18dcf1b34813068d11fc760307afc55785cf310fba838d",
    "srgnn__scut_masked__g2__anatomy":
        "a0862d15d69f0046c7772725245fc70124f5a27b1509497bdbd422c76dead167",
    "srgnn__scut_masked__g2__grid":
        "ffeded42a7181c15ce76418672697921cd34d9db5cf54eb13e0a74366501acf5",
    "srgnn__scut_masked__g2__random":
        "92cefd7c5f6c747eaba9ad8724d2ab0d0ce03643fda52fc6a2d6c2bd21aff4d4",
    "srgnn__scut_original__g2__native":
        "efb68dc90d7602553f2628cf1fed9d873510edf6982ece07ca244dc459e18260",
    "swin_b__imagenet__g2":
        "0345d5a34f96c17fe7590f93e45d7d5df8be25784b590e903bca7217e822e182",
    "swin_b__scut_masked__g1":
        "c81221478c2920e9c9adb7aebb34035082113e1e7307a94cd84c41f9d6e33f9d",
    "swin_b__scut_masked__g2":
        "706f35f185dae7c04ecf7197bc760d9a5442a20ce6fce9e79441273b3335a357",
    "swin_b__scut_original__g2":
        "5e02eb43591ce1522cbbe87e818aa3e89cfde0b88293f588f819c3ae7bd02f39",
    "vit_b16__imagenet__g2":
        "40e7424aea9f0b149f8960178322c47bbccd8d62ae2a0c2be11aceafe4c4a4a7",
    "vit_b16__scut_masked__g1":
        "b2ef8f7dc3220f942464c7420ddaea79efb218fc0e07238dda02a318d09e7af3",
    "vit_b16__scut_masked__g2":
        "bded2f72c9d5cef30bfa8d67c2f874151c811246aca60e7441befdc0ce04ebbe",
    "vit_b16__scut_original__g2":
        "3e55db73dcee3bd6e9ab7dda066a6d74db3386d36a5ce60fcc9e6f0b894aa0e1",
    #: **[VERIFIED 2026-08-01] Stage C0's set, alone in
    #: ``embeddings_g1_control_v1``.** It is there because a published
    #: artifact cannot gain a set (PLAN §2.6), and it exists because
    #: ``scut_original`` is the one pretrained init that is not
    #: geometry-bound -- so G1 against G2 there varies geometry alone on
    #: pretrained weights, which Stage C cannot do.
    "vit_b16__scut_original__g1":
        "498899b5a8f0d47fe9d403df0d3f668196f3b777b99c60860be8d3aa766528e0",

    #: **[VERIFIED 2026-08-02] Stage D1's nine, in
    #: ``embeddings_g1_ladder_v1``.** A third version rather than an addition
    #: to the control one, which already existed with its single set --
    #: ``run.task_extract_embeddings`` refuses an out_version whose directory
    #: exists, so the nine could not go beside it.
    #:
    #: Produced by ``configs/p7_extract_g1_ladder.yaml`` in one run:
    #: 573,500,008 bytes, artifact rollup
    #: ``fa46e50fd5c201cf7130c3e7a59ae4e0c7594af6d7d3799a2de401534c127e3d``.
    #: Row order asserted per set against an independent manifest read, and
    #: normalisation read per model -- the two-two split (ViT-B/16 at 0.5/0.5,
    #: Swin-B on ImageNet statistics) came through correctly, which is the
    #: check ``swin_b__imagenet__g1`` needed before it can decide anything
    #: about ``STAGE_D_BACKBONE_CONFLICT``.
    "agnet__imagenet__g1":
        "4c03654c00933495e9af935779a09dacdf75546dbf33a46002e6e412ee8f9b7b",
    "agnet__scut_masked__g1__native":
        "991e7fed8eced185b6a373bb5c9170a29150e92bf313759f4323c088a22027e4",
    "agnet__scut_original__g1__native":
        "2297cdfe6704634bbfcf3c2ac89506adafbc38b010fbe96ec1750b4eb5cff1b7",
    "srgnn__imagenet__g1":
        "b5a2c0b3bf788f838f33f3780fe2e411bc31e7266891a8e47b5cafd130e11da9",
    "srgnn__scut_masked__g1__native":
        "fe260b5b88665c5c18f7f066fe241ffdebadad3d30654e62d0a1579a32b6211f",
    "srgnn__scut_original__g1__native":
        "20653f0769c8a5e3cc252b7c3c39b745e548e935752bb390589929115ce93f6a",
    "swin_b__imagenet__g1":
        "52c99db13a9f59c43a9685c6b792e11ed55dec6ea18263e8fbde229f886fd40e",
    "swin_b__scut_original__g1":
        "65301bd73fea1007c3899d6d7e5c9386b5065a1c15fa90e6be485fd1aedcf796",
    "vit_b16__imagenet__g1":
        "ba4b12535f60aca4cc679246a60ecbcdc880c5ca66319937126e4f85c632d7ab",
}

#: Where the hashes above were obtained, so a reader can re-derive rather than
#: trust. A recorded hash without the environment that produced it is a number
#: nobody can check.
DECLARED_EMBEDDING_PROVENANCE = {
    "verified": "2026-08-01",
    "by": "scripts/declare_ladder_inputs.py, run on the cluster",
    #: TWO batches and two artifact versions, recorded separately because
    #: they were obtained by separate deliberate acts. Collapsing them into
    #: one count would lose which extraction produced what.
    "batches": [
        {
            "artifact": "data/embeddings/embeddings_v1",
            "n_sets": 16,
            "for": "the original 19 ladder arms",
        },
        {
            "artifact": "data/embeddings/embeddings_g1_control_v1",
            "n_sets": 1,
            "for": (
                "Stage C0 -- vit_b16__scut_original__g1, the geometry "
                "comparison with the checkpoint held constant"
            ),
        },
        {
            "artifact": "data/embeddings/embeddings_g1_ladder_v1",
            "n_sets": 9,
            "for": (
                "Stage D1 -- the init ladder at G1, so the geometry x init "
                "interaction becomes a result rather than a caveat"
            ),
            "verified": "2026-08-02",
            "rollup": (
                "fa46e50fd5c201cf7130c3e7a59ae4e0c7594af6d7d3799a2de401534c127e3d"
            ),
            "total_bytes": 573_500_008,
        },
    ],
    "n_sets": 26,
    "absent_because_already_declared": "srgnn__scut_masked__g2__native",
}


#: **[MEASURED 2026-08-01] The clean geometry measurements, each with the
#: arm's OWN five-seed SD.**
#:
#: "Clean" means both cells share one set of weights, so geometry varies
#: alone. Only two inits qualify: ``imagenet`` (no checkpoint) and
#: ``scut_original`` (one ``original`` checkpoint at both geometries).
#: ``scut_masked`` is geometry-bound and does NOT qualify.
#:
#: **The SD was wrong and is corrected here.** The ImageNet-G1 figure was
#: quoted as 0.2521 with sd 0.0137 -- a five-seed MEAN paired with the
#: TEN-seed gate-2 SD. PLAN §4.12.1 forbids that by name ("each arm measures
#: and reports its own seed SD. Inherited, never"), and §4.3 names 0.0137
#: specifically as the figure not to reuse. Recomputed from the five per-seed
#: values, the arm's own SD is **0.01483**. Every verdict survives, which is
#: not a licence for the wrong pairing -- the threshold formula wants that
#: arm's spread, and a number that happens to clear a slightly wrong bar has
#: still been compared against the wrong bar.
#:
#: **Both sides of the ImageNet pair are LIVE extraction**, same code path,
#: same five seeds -- so that measurement carries no live/artifact caveat at
#: all. Its G2 SD (0.01929) independently reproduces the artifact arm's
#: 0.0193, which corroborates the extraction equivalence at the SD as well as
#: at the mean.
CLEAN_GEOMETRY_MEASUREMENTS = {
    "measured": "2026-08-01",
    "imagenet": {
        "g1": {"mean": 0.25206, "sd": 0.01483, "n": 5, "source": "p3_train_cv (live)"},
        "g2": {"mean": 0.13474, "sd": 0.01929, "n": 5, "source": "p7_repro_g2_live"},
        "delta": 0.1173, "threshold": 0.0213, "claimable": True,
        "note": "both sides live, same path, same seeds -- no extraction caveat",
    },
    "scut_original": {
        "g1": {"mean": 0.1952, "sd": 0.0280, "n": 5, "source": "p7_c0 (artifact)"},
        "g2": {"mean": 0.1406, "sd": 0.0333, "n": 5, "source": "p7_d (artifact)"},
        "delta": 0.0546, "threshold": 0.0381, "claimable": True,
    },
    #: **[AMENDED 2026-08-02] Stage D1 takes this from two clean comparisons
    #: to EIGHT, and the headline no longer holds unconditionally.**
    #:
    #: At ``imagenet`` G1 wins in all four backbones -- ViT +0.1173, Swin
    #: +0.1011, SR-GNN +0.0793 all claimable, AG-Net +0.0392 inside its 0.0440.
    #: At ``scut_original`` it splits: ViT +0.0546 and SR-GNN +0.0524 claimably
    #: G1, **Swin -0.0902 claimably G2**, AG-Net -0.0321 inside its 0.0362.
    #:
    #: So "G1 beats G2 on cleft data" is right at ImageNet weights and
    #: **backbone-dependent at pretrained ones** -- one backbone claimably
    #: prefers G2 there. The two original measurements were both ViT, and a
    #: verdict stated over two cells of one backbone read as a property of the
    #: data. It is a property of that backbone at that init.
    "all_eight_clean_comparisons": {
        "imagenet": {
            "vit_b16": (0.1173, 0.0213, True), "swin_b": (0.1011, 0.0410, True),
            "srgnn": (0.0793, 0.0447, True), "agnet": (0.0392, 0.0440, False),
        },
        "scut_original": {
            "vit_b16": (0.0546, 0.0381, True), "swin_b": (-0.0902, 0.0423, True),
            "srgnn": (0.0524, 0.0351, True), "agnet": (-0.0321, 0.0362, False),
        },
        "format": "(G1 - G2, threshold, claimable)",
        "excluded": (
            "scut_masked is geometry-BOUND, so its G1/G2 pair consumes "
            "different checkpoints and is a matched-PIPELINE comparison"
        ),
    },
    "verdict": (
        "G1 beats G2 at ImageNet weights in all four backbones; at "
        "scut_original it is backbone-dependent, with Swin claimably the "
        "other way. Not a property of the data alone"
    ),
    "geometry_x_init_interaction": {
        "delta": 0.0627, "threshold": 0.0437, "claimable": True,
        "meaning": (
            "the geometry penalty is NOT a constant offset -- it is 0.117 at "
            "ImageNet and 0.055 at scut_original. So a G2 answer does not "
            "GENERALISE to G1, though it remains VALID at G2"
        ),
    },
    "corrected_sd_pairing": {
        "was": "five-seed mean 0.2521 with the ten-seed SD 0.0137",
        "now": "0.25206 with its own five-seed SD 0.01483",
        "verdicts_unchanged": True,
        "why_it_still_mattered": (
            "PLAN §4.12.1 -- inherited, never. Surviving a slightly wrong bar "
            "is not the same as clearing the right one"
        ),
    },
}

#: **[MEASURED 2026-08-02] STAGE D AT G2 IS COMPLETE -- twelve cells, twelve
#: SDs, and every Q1 and Q2 verdict re-derived from the two arms' own spreads.**
#:
#: Means and SDs pasted from the arms' own metrics.json; graph arms ten seeds,
#: transformers five (``SEEDS_BY_KIND``). **Every arm's SD is its own** (PLAN
#: §4.12.1), so ``pending_sd`` is empty in both questions and no threshold
#: here is inherited from another arm. Re-deriving is not ceremony: the
#: formula wants the two arms' spreads, and "claimable" is a comparison rather
#: than a property of a delta.
#:
#: **Q2 -- three claimable positives and one claimable negative. A CONFLICT,
#: confirmed on measured SDs.** ViT +0.0595, SR-GNN +0.0833, AG-Net +0.0770,
#: each past its own threshold; Swin -0.1067 past its own threshold in the
#: other direction. Masking helps three backbones and hurts the fourth, and
#: all four verdicts rest on the arms' own spreads. **Writing it up as "helps,
#: except Swin" would report an agreement the data does not contain** -- the
#: exception is not weaker evidence than the rule, it is the same strength
#: pointing the other way.
#:
#: **Q1 -- null in three of four, one claimable value, not reportable as a
#: single answer.** ViT +0.0059, SR-GNN -0.0252 and AG-Net +0.0309 all fall
#: inside their thresholds; Swin's +0.2027 clears its own and is the only
#: delta in Stage D to survive ``single_run_95``. Neither Swin arm is a broken
#: run (``STAGE_D_BACKBONE_CONFLICT``). **And the one claimable Q1 sits in the
#: backbone whose ImageNet reference arm spans zero** -- so the largest effect
#: in the table is measured from the least stable baseline in it, which is
#: what ``swin_b__imagenet__g1`` in the Stage D1 batch exists to test.
STAGE_D_AT_G2 = {
    "measured": "2026-08-02",
    "source": "pasted from each arm's metrics.json",
    "seeds": {"transformer": 5, "graph": 10},
    #: (imagenet, scut_original, scut_masked)
    "cells": {
        "vit_b16": (0.1347, 0.1406, 0.2001),
        "swin_b": (0.0065, 0.2092, 0.1025),
        "srgnn": (0.0926, 0.0674, 0.1507),
        "agnet": (0.0221, 0.0530, 0.1300),
    },
    "inits": ("imagenet", "scut_original", "scut_masked"),
    #: Each arm's own five- or ten-seed SD, and ``shrinkage`` =
    #: sd(pred)/sd(truth), which is what says whether a low PCC is a weak
    #: representation or a head that never fitted (PLAN §4.3).
    "sd": {
        "vit_b16": {"imagenet": 0.0193, "scut_original": 0.0333, "scut_masked": 0.0142},
        "swin_b": {"imagenet": 0.0414, "scut_original": 0.0439, "scut_masked": 0.0414},
        "srgnn": {"imagenet": 0.0350, "scut_original": 0.0402, "scut_masked": 0.0251},
        "agnet": {"imagenet": 0.0567, "scut_original": 0.0457, "scut_masked": 0.0301},
    },
    "shrinkage": {
        "swin_b": {"imagenet": 0.4299, "scut_original": 0.3936, "scut_masked": 0.3008},
    },
    "q1": {
        "delta": {
            "vit_b16": 0.0059, "swin_b": 0.2027,
            "srgnn": -0.0252, "agnet": 0.0309,
        },
        "verdict": "NOT REPORTABLE as a single answer -- three null, one large",
        "rederived": {
            "vit_b16": {
                "threshold": 0.0337, "claimable": False,
                "single_run_95": 0.0754, "survives_single_run_95": False,
            },
            "srgnn": {
                "threshold": 0.0330, "claimable": False,
                "single_run_95": 0.1045, "survives_single_run_95": False,
            },
            "agnet": {
                "threshold": 0.0451, "claimable": False,
                "single_run_95": 0.1427, "survives_single_run_95": False,
            },
            "swin_b": {
                "threshold": 0.0529, "claimable": True,
                # **[MEASURED 2026-08-04] THE ONE SURVIVOR of the whole
                # audit** -- 5 of 5 intervals exclude zero, margin 3.83x, one
                # of twenty-six pairs. And it must never be quoted without
                # the two facts below, which is why they live here rather
                # than an entry away. LADDER_PAIRED_AUDIT.
                "condition_1": {
                    "n_excluding_zero": 5, "n_seeds": 5, "claimable": True,
                    "margin": 3.83,
                },
                # **The delta is +0.2027 because the BASELINE is 0.0065.**
                # Swin's ImageNet cell at G2 is a near-zero correlation
                # (SWIN_G2_IMAGENET_CELL). The claim is therefore "beauty
                # pretraining beats a representation that barely correlates
                # at all", not "beauty pretraining helps".
                "rests_on": "SWIN_G2_IMAGENET_CELL, a baseline of 0.0065",
                # **And it does not reproduce at the other geometry.** The
                # same comparison at G1 -- same backbone, same inits, same
                # factor -- is +0.0114 at a 0.44x margin with 0 of 5
                # intervals excluding zero, because the same weights give
                # 0.1076 at G1. So the surviving claim is specific to a cell
                # whose value is specific to G2.
                "does_not_reproduce_at_g1": {
                    "pair": "Q1_at_g1__swin_b", "delta": 0.0114,
                    "margin": 0.44, "n_excluding_zero": 0, "n_seeds": 5,
                },
                #: **The only delta in the whole of Stage D to clear the
                #: conservative companion** -- it survives without averaging,
                #: which is the stronger statement PLAN §4.3 says to make when
                #: it is available. Here it is available exactly once, and in
                #: the arm the ladder is least sure of.
                "single_run_95": 0.1183, "survives_single_run_95": True,
            },
        },
        "pending_sd": [],
    },
    "q2": {
        "delta": {
            "vit_b16": 0.0595, "swin_b": -0.1067,
            "srgnn": 0.0833, "agnet": 0.0770,
        },
        "verdict": (
            "CONFLICT: three backbones claim masking helps, Swin claims it "
            "hurts. Not a consensus with an exception"
        ),
        #: **[RECORDED 2026-08-02] Q2 IS A CLAIM ABOUT ARM MEANS, and PLAN
        #: §4.3 asks which reading a number is, so both are carried.** All
        #: four deltas clear ``arm_means_95``; **none clears
        #: ``single_run_95``**, the spread of a difference between two SINGLE
        #: runs. So Q2 says these arms differ, not that any one seed of the
        #: masked arm beats any one seed of the original. The gap is not
        #: marginal anywhere -- the closest is SR-GNN at 0.0833 against
        #: 0.0929, and the widest AG-Net at 0.0770 against 0.1073.
        #:
        #: Worth stating because §4.3 says to quote the conservative figure
        #: "when a difference survives even without averaging, which is a
        #: stronger statement worth making when true". Here it is not true,
        #: and the same sentence obliges saying so. **In the whole of Stage D
        #: exactly one delta survives without averaging: Swin's Q1.**
        "rederived": {
            "vit_b16": {
                "threshold": 0.0317, "claimable": True,
                "single_run_95": 0.0710, "survives_single_run_95": False,
            },
            "swin_b": {
                "threshold": 0.0529, "claimable": True,
                "single_run_95": 0.1183, "survives_single_run_95": False,
            },
            "srgnn": {
                "threshold": 0.0294, "claimable": True,
                "single_run_95": 0.0929, "survives_single_run_95": False,
            },
            "agnet": {
                "threshold": 0.0339, "claimable": True,
                "single_run_95": 0.1073, "survives_single_run_95": False,
            },
        },
        "pending_sd": [],
    },
    #: **With all twelve SDs in hand, exactly two arms are not
    #: distinguishable from zero, and both are ImageNet** -- Swin 0.0065 at
    #: t = 0.35 and AG-Net 0.0221 at t = 1.23. Their Q1 values are therefore
    #: differences measured FROM baselines that have not been shown to learn
    #: anything, which is a different statement from a small effect, and it is
    #: most of why the ImageNet column's init effect is -0.0451. **The one
    #: claimable Q1 in the table sits in the backbone whose reference arm is
    #: the least stable of the four.**
    "arms_spanning_zero": ["swin_b__imagenet__g2", "agnet__imagenet__g2"],
}

#: **[MEASURED 2026-08-02] STAGE D1 -- the init ladder at G1. Twelve cells,
#: and both of Stage D's claimable findings turn out to be operating-point
#: artefacts.**
#:
#: Every verdict below was re-derived with ``phase3.combined_claimable_delta``
#: from these arms' own SDs. Two differ from the differences as first read,
#: and both differences are recorded because a delta is not a verdict:
#:
#: * **AG-Net's Q2 at G1 IS claimable** (+0.0425 against **0.0283**). It was
#:   read against ~0.045, which is AG-Net's *G2* threshold -- its G1 arms are
#:   tighter (0.0363 / 0.0278 against 0.0457 / 0.0301), so the bar moves with
#:   them. Same shape as the correction that started this: a threshold from
#:   the wrong arms (PLAN §4.12.1).
#: * **Q1 at G1 is claimably NEGATIVE in three backbones**, not merely
#:   "negative or null" -- ViT -0.0568, SR-GNN -0.0521, AG-Net -0.0404 all
#:   clear their thresholds. Only Swin is null (+0.0114 against 0.0259).
#:
#: **Q1: beauty pretraining claimably HURTS at G1.** Three of four backbones,
#: no backbone positive. At G2 the same question read null in three and hugely
#: positive in Swin, and that positive was the artefact
#: (``SWIN_G2_IMAGENET_CELL``). The central question of the project has an
#: answer, and it is not the hoped-for one.
#:
#: **Q2: the conflict does not reproduce, and does not simply reverse
#: either.** At G2 it was three claimable positives and one claimable
#: negative; at G1 it is ViT claimably negative (-0.1122, and the only Stage
#: D1 delta to survive ``single_run_95``), AG-Net claimably positive
#: (+0.0425), Swin and SR-GNN null (+0.0137, +0.0003). **Masking's answer
#: depends on the operating point**, which is the same conclusion the geometry
#: interaction reached and now has a second instance.
STAGE_D1_AT_G1 = {
    "measured": "2026-08-02",
    "source": "pasted from each arm's metrics.json",
    "inits": INITS,
    "cells": {
        "vit_b16": (0.2520, 0.1952, 0.0830),
        "swin_b": (0.1076, 0.1190, 0.1327),
        "srgnn": (0.1719, 0.1198, 0.1201),
        "agnet": (0.0613, 0.0209, 0.0634),
    },
    "sd": {
        "vit_b16": {"imagenet": 0.0148, "scut_original": 0.0280, "scut_masked": 0.0247},
        "swin_b": {"imagenet": 0.0217, "scut_original": 0.0201, "scut_masked": 0.0267},
        "srgnn": {"imagenet": 0.0630, "scut_original": 0.0399, "scut_masked": 0.0211},
        "agnet": {"imagenet": 0.0426, "scut_original": 0.0363, "scut_masked": 0.0278},
    },
    "q1": {
        "delta": {
            "vit_b16": -0.0568, "swin_b": 0.0114,
            "srgnn": -0.0521, "agnet": -0.0404,
        },
        "verdict": (
            "beauty pretraining claimably HURTS at G1 -- negative in three of "
            "four, null in the fourth, positive in none"
        ),
        "rederived": {
            "vit_b16": {
                "threshold": 0.0278, "claimable": True,
                "single_run_95": 0.0621, "survives_single_run_95": False,
            },
            "swin_b": {
                "threshold": 0.0259, "claimable": False,
                "single_run_95": 0.0580, "survives_single_run_95": False,
            },
            "srgnn": {
                "threshold": 0.0462, "claimable": True,
                "single_run_95": 0.1462, "survives_single_run_95": False,
            },
            "agnet": {
                "threshold": 0.0347, "claimable": True,
                "single_run_95": 0.1097, "survives_single_run_95": False,
            },
        },
        "pending_sd": [],
    },
    "q2": {
        "delta": {
            "vit_b16": -0.1122, "swin_b": 0.0137,
            "srgnn": 0.0003, "agnet": 0.0425,
        },
        "verdict": (
            "no reproduction of the G2 conflict: one claimable negative, one "
            "claimable positive, two null. Masking's answer is "
            "operating-point dependent"
        ),
        "rederived": {
            "vit_b16": {
                "threshold": 0.0327, "claimable": True,
                #: The only Stage D1 delta to clear the conservative
                #: companion, as Swin's Q1 was the only one at G2.
                "single_run_95": 0.0732, "survives_single_run_95": True,
            },
            "swin_b": {
                "threshold": 0.0293, "claimable": False,
                "single_run_95": 0.0655, "survives_single_run_95": False,
            },
            "srgnn": {
                "threshold": 0.0280, "claimable": False,
                "single_run_95": 0.0885, "survives_single_run_95": False,
            },
            "agnet": {
                "threshold": 0.0283, "claimable": True,
                "single_run_95": 0.0896, "survives_single_run_95": False,
            },
        },
        "pending_sd": [],
    },
    #: One arm, and it is AG-Net's ``scut_original`` at t = 1.82 -- so its
    #: claimable Q2 is measured from a baseline that has not been shown to
    #: learn anything, the same caveat its G2 Q1 carried. Every other G1 arm
    #: excludes zero, including **Swin's imagenet at t = 11.09**, which is the
    #: measurement ``SWIN_G2_IMAGENET_CELL`` turns on.
    "arms_spanning_zero": ["agnet__scut_original__g1"],
    #: **The G1 extraction is sound, checked two ways before anything above is
    #: read.** ``vit_b16 imagenet g1`` came in at 0.2520 (sd 0.0148) through
    #: the artifact path against the live 0.25206 (sd 0.01483) already
    #: recorded in ``CLEAN_GEOMETRY_MEASUREMENTS`` -- agreeing to four decimal
    #: places on the mean AND on the SD, a second instance of the live/artifact
    #: equivalence measured at G2. And ``vit_b16 scut_original g1`` reproduces
    #: Stage C0's 0.1952/0.0280 exactly, from a different extraction batch.
    "extraction_corroborated_by": {
        "vit_b16__imagenet__g1": "0.2520/0.0148 artifact vs 0.25206/0.01483 live",
        "vit_b16__scut_original__g1": "0.1952/0.0280, reproduces Stage C0 exactly",
    },
}

#: **[MEASURED 2026-08-02] The Swin G2 ImageNet cell needs LESS explaining
#: than it looked like, and that is the finding.**
#:
#: The cell that manufactured the project's largest apparent effect is
#: ``swin_b imagenet g2`` = 0.0065, spanning zero at t = 0.35. At G1 the same
#: weights give **0.1076 at t = 11.09**. So the representation is fine, the
#: head fitted normally (shrinkage 0.4299), and the extraction batch is
#: corroborated two ways. Nothing is broken.
#:
#: **The hypothesis, and it is arithmetic rather than mechanism: the cell is
#: what you get by subtracting an ORDINARY geometry penalty from a MODEST
#: starting value.**
#:
#: All four backbones lose PCC from G1 to G2 at ImageNet -- 0.1173, 0.1011,
#: 0.0793, 0.0392, mean 0.0842, sd 0.0338. **Swin's drop is +0.50 SD from that
#: mean, and ViT's is larger at +0.98.** Swin's G1 value is 0.1076, the second
#: lowest of the four. Subtract the mean drop and you predict +0.0234 against
#: an observed 0.0065 -- a residual well inside the arm's own SD of 0.0414.
#: **Every ingredient is unremarkable; only the sum lands near zero.**
#:
#: **Why it read as an artefact: Q1 is a difference measured FROM this cell,
#: and a reference near zero maximises it.** Swin's Q1 at G2 was +0.2027,
#: claimable, surviving ``single_run_95`` -- the strongest-looking result in
#: Stage D. At G1, where the reference is an ordinary 0.1076, the same
#: comparison gives +0.0114 and does not clear its threshold.
#:
#: **[GENERALISATION, and it is NOT MASKED_G1_ARTEFACT's]** That one was a
#: weak *artifact* bound by a rule to one level of a factor -- a hidden second
#: input. Here there is no weak artifact and no hidden factor: two ordinary
#: effects compose to put one cell near zero. **An arm near zero is not
#: necessarily an anomalous arm.** It may be an ordinary arm whose value
#: happens to land there, and every difference measured from it will look
#: large. The check is not "what is wrong with this cell" but "is this cell
#: predicted by the effects already measured" -- and this one is.
#:
#: **What remains genuinely open, and it is smaller:** why Swin's ImageNet
#: representation is only middling at G1 (0.1076, against ViT's 0.2520) when
#: Swin was the strongest of the four at SCUT pretraining (0.9122). That is a
#: question about transfer, and
#: ``graph_cleft.PRETRAINING_DOES_NOT_PREDICT_TRANSFER`` already says SCUT
#: test PCC does not predict cleft transfer quality. It needs no new arm.
SWIN_G2_IMAGENET_CELL = {
    "measured": "2026-08-02",
    "cell": "swin_b imagenet g2",
    "value": 0.0065,
    "spans_zero": True,
    "same_weights_at_g1": {"value": 0.1076, "t": 11.09, "spans_zero": False},
    "not_defective": [
        "shrinkage 0.4299, the highest of Swin's three -- the head fitted",
        "the G1 arm from the same weights is healthy",
        "the G1 extraction batch is corroborated two ways (STAGE_D1_AT_G1)",
    ],
    "hypothesis": (
        "an ordinary geometry penalty subtracted from a modest starting "
        "value. The four G1->G2 ImageNet drops are 0.1173 / 0.1011 / 0.0793 / "
        "0.0392; Swin's is +0.50 SD from their mean and ViT's is larger at "
        "+0.98. 0.1076 - 0.0842 predicts +0.0234 against 0.0065 observed, "
        "inside the arm's own SD of 0.0414"
    ),
    "why_it_read_as_an_effect": (
        "Q1 is a difference measured FROM this cell, and a reference near "
        "zero maximises it: +0.2027 at G2 against +0.0114 at G1"
    ),
    "generalisation": (
        "an arm near zero is not necessarily an anomalous arm -- it may be an "
        "ordinary arm whose value lands there, and every difference measured "
        "from it looks large. Ask whether the cell is PREDICTED by the "
        "effects already measured, not what is wrong with it"
    ),
    "distinct_from_masked_g1_artefact": (
        "that was a weak artifact bound by a rule to one level of a factor, "
        "a hidden second input. Here nothing is weak and nothing is hidden: "
        "two ordinary effects compose"
    ),
    "still_open": (
        "why Swin's ImageNet representation is middling at G1 (0.1076) when "
        "Swin led the four at SCUT pretraining (0.9122) -- covered by "
        "graph_cleft.PRETRAINING_DOES_NOT_PREDICT_TRANSFER, needs no new arm"
    ),
}

#: **[MEASURED 2026-08-02] THE BEST ARM IN THE PROJECT IS AN OFF-THE-SHELF
#: IMAGENET EMBEDDING WITH A LINEAR HEAD, AT G1.**
#:
#: ``p7_d1_vit_b16_imagenet_g1`` -- **0.2520, sd 0.0148, five seeds**. It is
#: the highest of the twenty-four ladder cells and beats every one of them
#: claimably. It also reproduces the Phase 3 gate-2 baseline of 0.2529, which
#: was measured on the same representation at the same geometry through the
#: LIVE path, so the headline number of the project has now been obtained
#: twice by different routes.
#:
#: **The margin over the nearest challenger is thin and must be quoted as
#: such.** Against ``swin_b scut_original g2`` at 0.2092 the delta is +0.0428
#: against a threshold of 0.0406 -- claimable at **1.05x**, and it does NOT
#: survive ``single_run_95`` (0.0908). Against everything else the margin is
#: comfortable: 2.89x over ``vit_b16 scut_masked g2`` and 2.05x over
#: ``vit_b16 scut_original g1``. "Nothing beats it" is true and its nearest
#: rival is only just excluded, and a write-up that says the first without the
#: second is overstating a 1.05x.
#:
#: **This is the outcome the brief was structured for** (§3, "If nothing beats
#: 0.2529"): a frozen ImageNet embedding with a linear head is the strongest
#: model, and neither beauty pretraining, nor domain-matched masking, nor
#: region structure, nor graph message passing improves on it at n=237. Every
#: Phase 4 baseline pointed this way and Stage D1 closes it -- Q1 is claimably
#: NEGATIVE at G1 in three backbones of four.
#: **The claim's run identity: which run directory holds an arm's vectors.**
#:
#: An arm may REUSE another's run -- ``p7_g1_vit_b16_imagenet_g1`` is the same
#: fits as ``p7_d1_vit_b16_imagenet_g1``, declared under a second name so the
#: label question has a base arm. Pairing on the arm NAME would look for a run
#: directory that was never created.
def run_stem(arm: dict) -> str:
    """The config stem whose run directory holds this arm's predictions."""
    return arm.get("reuses") or arm["name"]


#: **[DECIDED 2026-08-04] Which claims the paired task can settle, and which
#: it cannot -- stated rather than left to whatever the pair list happens to
#: contain.**
#:
#: **Excluded: the label formulation.** ``load_inputs`` takes ``label`` as a
#: manifest COLUMN, so a ``median`` arm is trained and scored against the
#: median column and a ``mean`` arm against the mean column. Their PCCs are
#: computed against DIFFERENT truth vectors, so subtracting them is a
#: comparison of two task difficulties, not of two models on one task -- and a
#: paired bootstrap over patients has no common truth to pair on. Condition 1
#: is not merely uncomputed for these four; it is undefined for them as they
#: stand. ``load_oof_vectors`` refuses them on the truth cross-check, which is
#: the guard working rather than an obstacle. R2's shape exactly: the delta is
#: a correct number answering a different question from the one its name
#: implies.
#:
#: **Excluded: arms at an unrun stage that reuse nothing.** Stage G0 was built
#: and never run. **[CORRECTED] The stage label alone is not the test**:
#: ``p7_g0_vit_b16_imagenet_g2`` reuses ``p7_d_vit_b16_imagenet_g2``, which
#: ran, so filtering on "G0 appears" silently dropped a Q1 comparison that is
#: perfectly computable. What has no vectors is an arm at an unrun stage with
#: no ``reuses``, which here is exactly the two G0 label arms -- already
#: excluded by the label reason.
#:
#: **[MEASURED 2026-08-04] ``comparisons()`` does not contain the D1-at-G1 Q1
#: and Q2 claims.** ``STAGE_D1_AT_G1`` carries eight ``rederived`` verdicts and
#: the project's own comparison list has no entry for any of them -- it covers
#: the G2 ladder plus the scheme, geometry and label questions. Same shape as
#: ``declare_ladder_inputs.py``'s discovery glob before it was widened: the
#: list was right about what it contained and silent about what it did not.
#: They are derived here from ``arms()`` instead, so the audit is not narrowed
#: to whatever the comparison list happens to reach.
#: Stages built but never run, so no run directory holds their predictions.
UNRUN_STAGES = ("G0",)

#: **[PASTED 2026-08-04] Stems with sibling runs, pinned by job id.**
#:
#: **The audit is complete and found exactly one.** The ladder scope declares
#: 240 vectors across 30 run directories; ``declare_ladder_inputs.py`` hashed
#: 230, refused 10 as AMBIGUOUS, and reported none MISSING. So 29 stems
#: resolve to a single run, and no stem resolves only to a partial.
#:
#: ``p7_e_srgnn_scut_masked_g2_random`` matches three:
#:
#:     4cb62c05__p7-e-srgnn-random     4 seeds   interrupted, PARTIAL
#:     f46226fb__p7-e-srgnn-random-2  10 seeds   0.1354 sd 0.0276
#:     63344885__p7-e-srgnn-random-3  10 seeds   0.1354 sd 0.0276
#:
#: ``-2`` is declared: the relaunch after the interruption, and the run Stage
#: E's 0.1354 was recorded from.
#:
#: **The partial is the exposure the guard caught, and it is a new shape.**
#: Every earlier ambiguity was between a VOID run and a good one, where the
#: void run's numbers are wrong and a verification against the recorded band
#: would catch it. This one is a run whose numbers are *right* and whose
#: SEED COUNT is not. Declaring it would have paired against four seeds while
#: the recorded band said ten -- condition 1 computed over four intervals
#: instead of ten, in a criterion that is a universal over the seeds present
#: (``phase7c.TEN_SEED_ROUND_NOT_RUN``), so fewer intervals is a strictly
#: EASIER bar. It would have returned entirely ordinary output.
#:
#: **And an unplanned determinism check passed on the way.** ``-2`` and ``-3``
#: are different SHAs and agree to four decimals on the mean AND the SD. The
#: void ladder's measured hazard was the same nominal arm differing by 0.068
#: PCC across SHAs (PLAN §4.12.1); this is the opposite result, obtained for
#: free, on a graph arm at ten seeds.
LADDER_JOB_IDS = {
    "p7_e_srgnn_scut_masked_g2_random": "p7-e-srgnn-random-2",
}

SIBLING_RUNS_AUDIT = {
    "measured": "2026-08-04",
    "scope": "ladder -- 240 vectors across 30 run directories",
    "hashed": 230, "ambiguous": 10, "missing": 0,
    "stems_with_siblings": sorted(LADDER_JOB_IDS),
    "conclusion": (
        "29 of 30 stems resolve to exactly one run, and none resolves only to "
        "a partial. The one exception is pinned by job id"
    ),
    "the_partial_is_a_new_shape": (
        "earlier ambiguities were VOID-versus-good, where wrong numbers give "
        "the verification something to catch. A PARTIAL has right numbers and "
        "a short seed count -- and condition 1 is a universal over the seeds "
        "PRESENT, so four intervals is an easier bar than ten"
    ),
    "unplanned_determinism_check": (
        "runs -2 and -3 are different SHAs and agree to four decimals on mean "
        "AND sd at ten seeds. The void ladder's hazard was 0.068 PCC drift "
        "across SHAs on the same nominal arm; this is the opposite result"
    ),
}

PAIRED_CLAIM_COVERAGE = {
    "decided": "2026-08-04",
    "unrun_stages": list(UNRUN_STAGES),
    "excluded_label_formulation": (
        "load_inputs takes `label` as a manifest COLUMN, so median and mean "
        "arms are scored against different truth vectors. Their delta "
        "compares two task difficulties, not two models on one task, and a "
        "paired bootstrap has no common truth. Condition 1 is UNDEFINED for "
        "these, not uncomputed"
    ),
    "excluded_stage_g0": "built and never run -- there are no vectors",
    "comparisons_does_not_cover_d1_at_g1": (
        "STAGE_D1_AT_G1 carries eight rederived verdicts and comparisons() "
        "has no entry for any of them; they are derived from arms() here"
    ),
}


def _g1_init_pairs() -> list[dict]:
    """The D1-at-G1 Q1 and Q2 pairs, derived from the arm list.

    ``comparisons()`` does not carry these (``PAIRED_CLAIM_COVERAGE``), so
    they are built the same way it builds its own: one factor varied, arms
    matched on everything else.
    """
    at_g1 = {
        (arm["backbone"], arm["init"]): arm
        for arm in arms()
        if arm.get("geometry") == "g1" and arm.get("label") == "mean"
    }
    out = []
    for backbone in ("vit_b16", "swin_b", "srgnn", "agnet"):
        for question, first, second in (
            ("Q1_at_g1", "imagenet", "scut_original"),
            ("Q2_at_g1", "scut_original", "scut_masked"),
        ):
            a, b = at_g1.get((backbone, first)), at_g1.get((backbone, second))
            if a is None or b is None:
                continue
            out.append({
                "key": f"{question}__{backbone}",
                "question": question, "varies": "init",
                "a": run_stem(a), "b": run_stem(b),
                "seeds": list(a["seed_list"]),
            })
    return out


#: **[DECIDED 2026-08-04] Phase 8's contribution, tested before it is
#: written.** The brief's §0 states *"the model that performs best is the
#: least interpretable, and the models built to be interpretable perform
#: worse"* -- which is a comparative claim of exactly the size §2 of that same
#: brief warns is unresolvable on this cohort, in the section naming the
#: phase's contribution.
#:
#: **It is not among the 26 audited**: it varies backbone, init and geometry
#: at once, so it is not a one-factor comparison. The ladder headline had the
#: same shape at a SMALLER delta (0.0428) and came back 0 of 5.
#:
#: **The interpretable arm is SR-GNN's BEST cell, not the brief's arm B.**
#: ``p7_d1_srgnn_imagenet_g1_native`` at 0.1719, rather than the masked-G2
#: cell at 0.1507: it is the arm the trade-off sentence actually cites, it
#: keeps the native region scheme so "voting" is unchanged, and putting both
#: arms at G1 removes the geometry confound from the brief's §5 region
#: comparison rather than requiring a cross-geometry region mapping to be
#: built and justified (``geometry/patches.py``: geometry and patch scheme
#: interact and cannot be treated as independent factors).
#:
#: **Five seeds, not ten, and the reason belongs with the result.** SR-GNN ran
#: at ten and ViT at five; pairing needs the same held-out patients on both
#: sides, so the comparison uses the five they share. SR-GNN's recorded 0.1719
#: is a TEN-seed mean, and its five-seed mean over the paired subset may
#: differ -- the run recomputes each arm's own band over the seeds actually
#: used, which is what PLAN §4.12.1 requires anyway.
TRADE_OFF_PAIR = {
    "decided": "2026-08-04",
    "question": "is the interpretability-performance trade-off a CLAIM or a DESCRIPTION",
    "interpretable_arm": {
        "stem": "p7_d1_srgnn_imagenet_g1_native", "recorded_pcc": 0.1719,
        "recorded_seeds": 10,
        "why_this_cell": (
            "SR-GNN's best, native scheme so region voting is unchanged, and "
            "at G1 so the region comparison needs no cross-geometry mapping"
        ),
    },
    "best_arm": {"stem": "p7_d1_vit_b16_imagenet_g1", "recorded_pcc": 0.2520,
                 "recorded_seeds": 5},
    "delta_if_recorded_means_hold": 0.0801,
    "seeds_used": 5,
    "not_in_the_audit": (
        "varies backbone, init and geometry at once -- not a one-factor "
        "comparison. The ladder headline had that shape at a SMALLER delta "
        "and came back 0 of 5"
    ),
    "if_unresolvable": (
        "the contribution shifts from a claim about the trade-off to a "
        "description of it: 'the best-SCORING model is the least "
        "interpretable', which is licensed regardless of the result"
    ),
    # ---- the outcome -----------------------------------------------------
    "measured": "2026-08-04",
    "result": {
        "srgnn_paired_mean": 0.1884, "srgnn_sd": 0.0520,
        "vit_paired_mean": 0.2520, "vit_sd": 0.0148,
        "n_seeds": 5, "delta": 0.0637, "margin": 1.342,
        "n_excluding_zero": 0, "claimable": False,
        "per_seed_delta_range": [0.0004, 0.1404],
    },
    # **Two numbers for one arm, and a reader WILL try to reconcile them.**
    # The ladder tables record SR-GNN's imagenet-G1 cell at 0.1719 over TEN
    # seeds. This comparison pairs the FIVE seeds it shares with ViT, and over
    # those five the same arm scores 0.1884. Neither is wrong; they are
    # different subsets of the same ten fits. The delta is +0.0637, not the
    # +0.0801 the recorded means implied.
    "why_two_means": (
        "0.1719 is the ten-seed mean recorded in the ladder tables; 0.1884 is "
        "the same arm over the five seeds ViT also ran, which is the subset "
        "pairing requires. Different subsets of the same ten fits -- quote "
        "whichever matches the comparison being made, and say which"
    ),
    "verdict": (
        "UNRESOLVED, 0 of 5. Phase 8's §0 takes the pre-registered "
        "descriptive form: the best-SCORING model is the least interpretable"
    ),
    "sign_consistent_magnitude_not": (
        "ViT is ahead at every seed, but by +0.0004 on seed 2024 and +0.1404 "
        "at the other end -- indistinguishable to four decimals on one seed "
        "and a third of the scale on another"
    ),
}


#: **[MEASURED 2026-08-04] The interpretable model is 3.5x less stable, on the
#: same five seeds and the same folds.**
#:
#: SR-GNN 0.0520 against ViT 0.0148 -- and this is not a property of the
#: comparison, it is a property of the architectures. Identical seeds mean
#: identical inner-val splits and identical fold assignments, so nothing about
#: the evaluation differs between them; what differs is how much the fitted
#: model moves when the seed does.
#:
#: **It is what drives the margin down to 1.342x.** The delta is +0.0637,
#: which is larger than the ladder headline's +0.0428 -- but the threshold is
#: dominated by SR-GNN's band, so a bigger gap clears a lower multiple.
#:
#: **The project already responded to this without naming it.** Graph arms run
#: at ten seeds and transformer arms at five (``SEED_POOL``, PLAN §4.12), a
#: decision taken because the graph arms' spread demanded it. The corroboration
#: is in the ladder's own companions: at G1, Q1's ``single_run_95`` is 0.1462
#: for SR-GNN and 0.0621 for ViT. So the procedure had already absorbed the
#: fact; what is new is that it is reportable.
#:
#: **It belongs beside the means in any statement about which model to
#: prefer.** "0.2520 against 0.1884" and "0.2520 ± 0.0148 against 0.1884 ±
#: 0.0520" support different decisions, and only the second is the
#: measurement. A model whose seed-to-seed spread is a third of its own score
#: is a different proposition from one whose spread is a fifteenth.
SEED_STABILITY_ASYMMETRY = {
    "measured": "2026-08-04",
    "srgnn_sd": 0.0520, "vit_sd": 0.0148, "ratio": 3.51,
    "same_seeds_same_folds": (
        "identical seeds mean identical inner-val splits and fold "
        "assignments, so this is architecture, not evaluation"
    ),
    "drives_the_margin": (
        "the delta +0.0637 is LARGER than the ladder headline's +0.0428, but "
        "the threshold is dominated by SR-GNN's band, so a bigger gap clears "
        "a lower multiple -- 1.342x against 1.05x"
    ),
    "already_absorbed_by_the_procedure": (
        "graph arms run at ten seeds and transformers at five because the "
        "spread demanded it. Corroborated by Q1's single_run_95 at G1: 0.1462 "
        "for SR-GNN against 0.0621 for ViT"
    ),
    "report_it_with_the_means": (
        "'0.2520 against 0.1884' and '0.2520 +/- 0.0148 against 0.1884 +/- "
        "0.0520' support different decisions, and only the second is the "
        "measurement"
    ),
}


def paired_claim_pairs(scope: str = "ladder") -> list[dict]:
    """Every arm pair whose condition 1 this project can compute.

    ``scope="headline"`` is the 1.05x comparison alone -- ten vectors, two run
    directories, and the answer to the thinnest claim in the project. It is a
    separate scope because that claim is worth resolving before anyone
    resolves globs for twenty-six other run directories.
    """
    challenger = BEST_ARM["nearest_challenger"]
    headline = [{
        "key": "headline__best_vs_nearest_challenger",
        "question": "best_arm", "varies": "backbone+init+geometry",
        "a": "p7_d_swin_b_scut_original_g2", "b": BEST_ARM["arm"],
        "seeds": list(SEED_POOL[:BEST_ARM["n"]]),
        "recorded": {
            "delta": challenger["delta"], "threshold": challenger["threshold"],
            "margin": challenger["margin"],
        },
    }]
    if scope == "headline":
        return headline

    # Phase 8's §0, tested before the phase is designed around it.
    # TRADE_OFF_PAIR. `b` is the best arm, as in `headline`, so a positive
    # delta means the less interpretable arm is ahead.
    trade_off = [{
        "key": "tradeoff__best_vs_most_interpretable",
        "question": "trade_off", "varies": "backbone+init+geometry",
        "a": TRADE_OFF_PAIR["interpretable_arm"]["stem"],
        "b": TRADE_OFF_PAIR["best_arm"]["stem"],
        "seeds": list(SEED_POOL[:TRADE_OFF_PAIR["seeds_used"]]),
        "recorded": {
            "delta": TRADE_OFF_PAIR["delta_if_recorded_means_hold"],
            "note": (
                "SR-GNN's 0.1719 is a ten-seed mean; this pairs the five "
                "seeds the two arms share"
            ),
        },
    }]
    if scope == "tradeoff":
        return trade_off

    # Phase 7D's six contrasts, five seeds each -- the 7D arms ran five, and
    # the two ten-seed comparators (the graph precedent) pair on the shared
    # five (P7D_PAIRED_CONTRASTS' cross-phase note). The recorded deltas are
    # DERIVED from PHASE_7D_OBSERVED so a later correction cannot land in
    # one record and miss the other.
    if scope == "p7d":
        observed = PHASE_7D_OBSERVED["arms"]
        means = {
            "p7_d1_vit_b16_imagenet_g1": observed["vit_b16"]["mean"],
            "p7_d1_swin_b_imagenet_g1": (
                STAGE_D1_AT_G1["cells"]["swin_b"][0]
            ),
            "roadb_p7_arm_vit_b16_imagenet_512": 0.0857,
            "p7d_arm_vit_b32": observed["vit_b32"]["mean"],
            "p7d_arm_vit_b8": observed["vit_b8"]["mean"],
            "p7d_arm_mvitv2_b": observed["mvitv2_b"]["mean"],
            "p7d_arm_vit_b32_512": observed["vit_b32_512"]["mean"],
            "p7d_arm_concat_multiscale": observed["concat"]["mean"],
        }
        return [
            {
                "key": key, "question": question, "varies": varies,
                "a": a, "b": b,
                "seeds": list(SEED_POOL[:5]),
                "recorded": {
                    "delta_of_means": round(means[b] - means[a], 4),
                    "source": "PHASE_7D_OBSERVED; descriptive, not claimable",
                },
            }
            for key, question, a, b, varies in P7D_PAIRED_CONTRASTS
        ]

    if scope != "ladder":
        raise LadderError(f"unknown paired-claim scope {scope!r}")

    out = list(headline)
    for check in comparisons():
        a, b = check["a"], check["b"]
        # The label question has no common truth to pair on, and an unrun arm
        # has no vectors. PAIRED_CLAIM_COVERAGE.
        #
        # **[CORRECTED] "Stage G0" is not the test.** `p7_g0_vit_b16_imagenet
        # _g2` REUSES `p7_d_vit_b16_imagenet_g2`, which ran -- so filtering on
        # the stage label dropped a Q1 comparison that is perfectly
        # computable. What has no vectors is an arm at an unrun stage that
        # reuses nothing.
        if check["varies"] == "label" or any(
            arm["stage"] in UNRUN_STAGES and arm.get("reuses") is None
            for arm in (a, b)
        ):
            continue
        out.append({
            "key": f"{check['question']}__{a['backbone']}__{run_stem(b)}",
            "question": check["question"], "varies": check["varies"],
            "a": run_stem(a), "b": run_stem(b),
            "seeds": list(a["seed_list"]),
        })
    out.extend(_g1_init_pairs())
    return out


def paired_claim_stems(
    scope: str = "ladder", pairs: list[dict] | None = None
) -> list[str]:
    """Every run directory the given scope reads, deduplicated.

    **``pairs`` is for a scope whose enumeration lives elsewhere.** Road B's
    resolution pairs are derived in ``roadb`` -- that road's structure is
    its own file's business -- but the three derivations below are generic
    over a pair list, and a second copy of them is exactly the duplication
    R10 exists to prevent. Pass the pairs; the logic stays here, once.
    """
    stems = set()
    for pair in paired_claim_pairs(scope) if pairs is None else pairs:
        stems.update((pair["a"], pair["b"]))
    return sorted(stems)


def paired_claim_vectors(
    scope: str = "ladder", pairs: list[dict] | None = None
) -> list[tuple]:
    """Every ``(stem, seed)`` the scope needs, deduplicated and sorted.

    **Seed counts are not uniform.** The transformer arms ran at five seeds
    and the graph arms at ten (``SEED_POOL``), so a stem's vector list is
    whatever the pairs it appears in require -- not a fixed five. Deriving it
    rather than assuming is what keeps a ten-seed arm from being declared with
    five files and silently compared on half its band.
    """
    wanted: dict[str, set] = {}
    for pair in paired_claim_pairs(scope) if pairs is None else pairs:
        for stem in (pair["a"], pair["b"]):
            wanted.setdefault(stem, set()).update(pair["seeds"])
    return sorted(
        (stem, seed) for stem, seeds in wanted.items() for seed in seeds
    )


def paired_claim_seed_groups(
    scope: str = "ladder", pairs: list[dict] | None = None
) -> dict:
    """``{seed tuple: [pair, ...]}`` -- pairs that share a seed list.

    The paired comparison needs both arms on the same seeds, and the loader
    takes one seed list per call, so the run walks these groups rather than
    assuming a single band across the whole ladder.
    """
    groups: dict[tuple, list] = {}
    for pair in paired_claim_pairs(scope) if pairs is None else pairs:
        groups.setdefault(tuple(pair["seeds"]), []).append(pair)
    return groups


#: **[MEASURED 2026-08-04] Every claimable verdict in Phase 7, 7B and 7C was
#: computed from PLAN §4.3's condition 2 alone. Audited: 43 records carry a
#: ``claimable`` field. NONE carries an interval.**
#:
#: §4.3 claims a delta only if BOTH hold: (1) the paired BCa CI over patients
#: excludes zero, and (2) the delta exceeds combined seed uncertainty. The
#: whole of Phase 7 has been reporting (2) as though it were the criterion.
#:
#: **This is not hypothetical any more.** Phase 7C computed condition 1 on
#: 2026-08-04 over its 35 stored vectors and it withdrew all nine of that
#: phase's verdicts -- 0 of 45 per-seed intervals excluded zero.
#: ``phase7c.PAIRED_BCA_WITHDREW_EVERY_VERDICT``.
#:
#: **The condition-2 margin does NOT predict condition 1, and 7C proves it.**
#: The tempting reading is that thin claims are exposed and comfortable ones
#: are safe. 7C's arm 2 cleared its threshold by **3.62x** and still failed
#: condition 1. Every claim in the audit sits at 5.51x or below, and all but
#: one at 3.67x or below. So there is no margin here that 7C has not already
#: falsified as a safety indicator: the per-seed interval's width is set by
#: patient-level disagreement between the two vectors, which the ratio of two
#: seed-level summaries says nothing about.
#:
#: **What is exposed, and how badly.** The thinnest is this record's own
#: ``nearest_challenger`` at **1.05x** -- the project's headline. If condition
#: 1 fails there, "0.2520 is claimably ahead of everything else" becomes "it
#: is the highest-scoring arm, not claimably the highest". The PCC itself is
#: unaffected; what goes is the comparative claim, which is the sentence the
#: write-up is structured around.
#:
#: **It is checkable without a single new fit.** Every ladder arm is a
#: ``train_cv`` keeper run and wrote ``seed_<n>__predictions.csv`` per seed --
#: that is how Phase 7B declared this very arm as its baseline.
#: ``run.task_phase7c_paired`` already does the work for a fixed arm list;
#: generalising it to arbitrary arm pairs is what closes this.
#:
#: **Not a reason to withdraw anything yet.** 7C's arms are augmented, so
#: their two vectors disagree idiosyncratically patient by patient in a way an
#: ImageNet-versus-SCUT contrast need not. Condition 1 is UNCOMPUTED here, not
#: failed. The correct present statement is that these verdicts rest on half
#: their criterion -- which, per PLAN R7's thirteenth instance, means they are
#: not weaker verdicts, they are not yet verdicts.
CLAIMS_REST_ON_HALF_THE_CRITERION = {
    "measured": "2026-08-04",
    "audited": "ladder.py, phase7b.py, phase7c.py, phase8.py",
    "records_with_a_claimable_field": 48,
    # **[UPDATED 2026-08-04] The audit began at 43 records and 0 with
    # condition 1.** Phase 7B's is now resolved -- computed, read, and its
    # "claimably worse" withdrawn (``phase7b.CLAIMABLY_WORSE_WITHDRAWN``).
    # Phase 7C's nine were withdrawn too but are recorded as a phase-level
    # outcome rather than per comparison.
    #
    # **[UPDATED 2026-08-15] 47 -> 48: PHASE_7D_CLOSING joined, and it is
    # the first record in the family BORN evidenced** -- its two claims
    # arrived with condition 1 attached (both 5/5 from the p7d-paired BCa)
    # instead of being claimed on condition 2 and resolved later. The audit
    # count moved because a claim was ADDED with its interval, which is the
    # direction this audit exists to make normal.
    "carrying_condition_1": [
        "ladder.BEST_ARM.nearest_challenger",
        "ladder.BEST_ARM.nearest_challenger.condition_1",
        "ladder.PHASE_7D_CLOSING",
        "ladder.STAGE_D_AT_G2.q1.rederived.swin_b",
        "phase7b.SEARCH_AXIS_VERDICTS.outcome.against_baseline",
    ],
    "resolved_so_far": (
        "COMPLETE 2026-08-04. 7C's nine, all withdrawn; 7B's one, withdrawn; "
        "the ladder's 26, of which ONE survives (LADDER_PAIRED_AUDIT). "
        "Twenty-nine of thirty comparisons fail condition 1 -- "
        "COHORT_CANNOT_RESOLVE"
    ),
    "claimed_true_with_a_numeric_margin": 18,
    "thinnest": {
        "record": "ladder.BEST_ARM.nearest_challenger",
        "margin": "1.05x",
        "at_stake": (
            "'0.2520 is claimably ahead of everything else' becomes 'it is "
            "the highest-scoring arm, not claimably the highest'. The PCC is "
            "unaffected; the comparative claim is not"
        ),
    },
    "margin_does_not_predict_condition_1": (
        "phase 7C's arm 2 cleared its threshold by 3.62x and still failed "
        "condition 1. Interval width is set by patient-level disagreement "
        "between the two vectors; the ratio of two seed-level summaries "
        "carries no information about it. Nothing in this audit exceeds 5.51x"
    ),
    "checkable_without_fits": (
        "every ladder arm is a train_cv keeper run with seed_<n>__predictions"
        ".csv per seed -- how phase7b declared p7_d1_vit_b16_imagenet_g1 as "
        "its baseline. run.task_phase7c_paired generalised to arbitrary arm "
        "pairs closes it"
    ),
    "status": (
        "UNCOMPUTED, not failed. 7C's arms are augmented and need not "
        "generalise. But per PLAN R7's thirteenth instance a verdict from a "
        "subset of the criterion's conditions is not a weaker verdict"
    ),
}

BEST_ARM = {
    "measured": "2026-08-02",
    "arm": "p7_d1_vit_b16_imagenet_g1",
    "pcc": 0.2520, "sd": 0.0148, "n": 5,
    "reproduces": {
        "phase3_gate2": 0.2529,
        "note": "same representation and geometry, measured through the LIVE path",
    },
    "nearest_challenger": {
        "arm": "swin_b scut_original g2", "pcc": 0.2092,
        "delta": 0.0428, "threshold": 0.0406, "claimable": True,
        "margin": "1.05x", "survives_single_run_95": False,
        "quote_it": (
            "claimable and thin. A write-up saying 'nothing beats it' without "
            "this is overstating a 1.05x"
        ),
        # **[WITHDRAWN 2026-08-04] Computed, and 0 of 5 intervals exclude
        # zero.** `claimable` above is condition 2 only and no longer decides.
        "claimable_condition_2_only": True,
        "condition_1": {
            "n_excluding_zero": 0, "n_seeds": 5, "claimable": False,
            "per_seed": [
                {"seed": 7, "delta": -0.0027, "lo": -0.138, "hi": 0.133},
                {"seed": 99, "delta": 0.0135, "lo": -0.129, "hi": 0.166},
                {"seed": 1337, "delta": 0.0238, "lo": -0.117, "hi": 0.166},
                {"seed": 2024, "delta": 0.0758, "lo": -0.065, "hi": 0.215},
                {"seed": 12345, "delta": 0.1040, "lo": -0.036, "hi": 0.255},
            ],
            # **The count understates it: the SIGN FLIPS.** Seed 7 has Swin
            # ahead; seed 12345 has ViT ahead by 0.104. Intervals ~0.28 wide
            # against a mean delta of 0.043, so the arms are indistinguishable
            # on every seed taken individually.
            "sign_flips": True,
            "mean_interval_width": 0.28,
        },
    },
    # **[WITHDRAWN 2026-08-04] The comparative claim is gone; the number is
    # not.** 0 of 5 intervals exclude zero and the sign flips between seeds.
    "verdict": (
        "ViT-B/16 ImageNet G1 at 0.2520 is the HIGHEST-SCORING arm of the "
        "forty-seven. Swin-B scut_original G2 at 0.2092 is NOT distinguishable "
        "from it on this cohort. The PCC is untouched; 'strongest' and "
        "'nothing improves on it' are not supported"
    ),
    "verdict_before_condition_1": (
        "an off-the-shelf ImageNet embedding with a linear head at G1 is the "
        "strongest model in the ladder. Nothing built since improves on it"
    ),
    "what_survives": (
        "the ordering is descriptive and still worth reporting: nothing built "
        "SCORED higher. What is withdrawn is that anything was shown to be "
        "worse"
    ),
    "brief_section": "§3 -- the outcome the write-up was structured for",
}


#: **[MEASURED 2026-08-04] THE PROJECT'S CENTRAL FINDING. This cohort cannot
#: resolve PCC differences of 0.04 to 0.10 between arms.**
#:
#: Not a limitation attached to each withdrawal -- a result in its own right,
#: and the strongest one the project has. It is measured, on four independent
#: comparisons, across three unrelated interventions:
#:
#:     Phase 7B, hyperparameter search    delta -0.0542   margin 2.37x   1 of 5
#:     Phase 7C, augmentation (arm 2)     delta -0.0891   margin 3.62x   0 of 5
#:     Phase 7C, all nine verdicts        to -0.1005      to 3.62x       0 of 45
#:     The ladder headline                delta +0.0428   margin 1.05x   0 of 5
#:
#: **Four for four.** Every claim in this project that has been tested against
#: PLAN §4.3's full criterion has failed it, at margins from 1.05x to 3.62x on
#: condition 2. Nothing has survived, and the failures are not clustered in one
#: method: a hyperparameter search, an augmentation family, and a
#: backbone-plus-init contrast are three different ways of changing a model.
#:
#: **The mechanism is interval width, and it is a property of the cohort.** A
#: per-seed paired BCa over the 237 patients is roughly 0.28 wide at the
#: headline. A mean delta of 0.043 sits inside that many times over, so the two
#: arms are indistinguishable on every seed taken individually -- and at the
#: headline the SIGN FLIPS between seeds, which no summary statistic shows.
#: **[2026-09-02] Fleiss kappa is UNWEIGHTED** -- it scores 1-vs-2 as
#: identically wrong to 1-vs-5, which is the wrong loss for an ordinal
#: scale. The distance-aware figures on the same matrix are QWK 0.4276
#: and mean inter-rater r 0.4696. Nothing below is edited:
#: ``record_audit.THE_KAPPA_LIMITATION``.
#: 237 patients with this label noise (Fleiss kappa 0.1662, QWK 0.4276) cannot
#: separate arms this close, however many seeds are averaged: seeds narrow the
#: estimate of a MEAN, and condition 1 asks about patients.
#:
#: **What this licenses, and what it does not.** Descriptive orderings are
#: still reportable -- one arm scored higher than another, and that is a fact
#: about the measurements. What is not reportable is that any arm was shown to
#: be better or worse than another. Every "claimably" sentence in the write-up
#: has to become a "scored higher" sentence.
#:
#: **And it is a stronger argument for the full-face data than any thin
#: claimed win would have been.** A 1.05x victory would have been a fragile
#: sentence inviting the reviewer's scepticism. "We measured what this cohort
#: can resolve, and it is coarser than every effect we set out to detect" is
#: an argument from evidence, and it generalises: it says what a next dataset
#: has to be big enough to do.
#: **[MEASURED 2026-08-04] The ladder audit: ONE of twenty-six pairs survives
#: condition 1.** Phase 7 closes on this.
#:
#: **Finding 1 -- the condition-2 margin does not predict condition 1, and
#: that is now measured across 26 pairs rather than inferred from one.** The
#: two HIGHEST margins both fail: 4.69x at 2 of 5, and 4.34x at 4 of 10. The
#: survivor is only the third-highest at 3.83x. Ranking claims by how
#: comfortably they clear the seed threshold orders them by something that
#: does not govern whether they hold, which is why this audit was never
#: ordered by recorded margin.
#:
#: **Finding 2 -- ``COHORT_CANNOT_RESOLVE`` is confirmed at scale: 25 of 26.**
#: Four for four became twenty-nine for thirty. Across Q1 and Q2 at two
#: geometries, both region schemes, the geometry contrasts, the search, the
#: augmentation family and the headline, this cohort resolves one comparison.
#:
#: **The survivor's caveat travels WITH it, in
#: ``STAGE_D_AT_G2["q1"]["rederived"]["swin_b"]``, not here.** Two facts, and
#: neither is optional when it is quoted:
#:
#: * Its +0.2027 is large because the BASELINE is 0.0065 -- Swin's ImageNet
#:   cell at G2 is a near-zero correlation. The claim is "beauty pretraining
#:   beats a representation that barely correlates at all".
#: * **It does not reproduce at the other geometry.** ``Q1_at_g1__swin_b`` is
#:   the same comparison, same backbone, same inits, same varied factor:
#:   +0.0114 at 0.44x, 0 of 5.
#:
#: **On "unexplained".** ``SWIN_G2_IMAGENET_CELL`` is not a mystery with no
#: account -- shrinkage 0.4299 refuted the broken-run reading, and the G1->G2
#: drop is unremarkable in size (+0.50 SD against the four ImageNet drops,
#: where ViT's is larger at +0.98). What is unexplained is narrower and still
#: fatal to quoting the claim plainly: no mechanism is identified for why
#: Swin's G2 representation correlates near zero in the first place, only that
#: the drop TO it is an ordinary size. A claim resting on that cell inherits
#: the gap.
LADDER_PAIRED_AUDIT = {
    "measured": "2026-08-04",
    "run": "p7_paired_ladder, 26 pairs, 240 vectors, 30 run directories",
    "survived": 1,
    "withdrawn": 25,
    "survivor": {
        "pair": "Q1__swin_b", "n_excluding_zero": 5, "n_seeds": 5,
        "margin": 3.83,
        "quote_it_with": (
            "its baseline is 0.0065, and the same comparison at G1 is 0.44x "
            "with 0 of 5 -- both recorded beside the claim itself"
        ),
    },
    "margin_does_not_predict_condition_1": {
        "measured_across": 26,
        "two_highest_margins_both_fail": [
            {"margin": 4.69, "excluding": "2 of 5"},
            {"margin": 4.34, "excluding": "4 of 10"},
        ],
        "survivor_rank_by_margin": 3,
        "consequence": (
            "ranking claims by how comfortably they clear the seed threshold "
            "orders them by something that does not govern whether they hold"
        ),
    },
    "confirms": "COHORT_CANNOT_RESOLVE, at scale -- 25 of 26",
}


COHORT_CANNOT_RESOLVE = {
    "measured": "2026-08-04",
    "finding": (
        "this cohort cannot resolve PCC differences of 0.04 to 0.10 between "
        "arms. Four of four comparisons tested against PLAN §4.3's full "
        "criterion have failed condition 1"
    ),
    "tested": [
        {"what": "phase 7B hyperparameter search", "delta": -0.0542,
         "margin": 2.37, "excluding": "1 of 5"},
        {"what": "phase 7C augmentation, arm 2", "delta": -0.0891,
         "margin": 3.62, "excluding": "0 of 5"},
        {"what": "phase 7C, all nine verdicts", "delta": -0.1005,
         "margin": 3.62, "excluding": "0 of 45"},
        {"what": "the ladder headline", "delta": 0.0428,
         "margin": 1.05, "excluding": "0 of 5"},
    ],
    # **[CONFIRMED AT SCALE 2026-08-04] The ladder audit ran all 26 pairs and
    # one survived.** Four of four became TWENTY-NINE OF THIRTY across every
    # comparison this project can compute. LADDER_PAIRED_AUDIT.
    "at_scale": {
        "tested": 30, "survived": 1, "withdrawn": 29,
        "the_one": (
            "Q1__swin_b, and it rests on a 0.0065 baseline and does not "
            "reproduce at the other geometry"
        ),
    },
    "three_unrelated_interventions": (
        "a hyperparameter search, an augmentation family, and a "
        "backbone-plus-init contrast -- three different ways of changing a "
        "model, one result"
    ),
    # [2026-08-31] What this record does and does NOT license about a
    # detectability threshold is stated separately, because a figure
    # that was never here got quoted as though it were:
    # SMALLEST_RESOLVABLE_DIFFERENCE.
    "what_it_does_not_license_2026_08_31": "SMALLEST_RESOLVABLE_DIFFERENCE",
    "mechanism": (
        "a per-seed paired BCa over 237 patients is ~0.28 wide at the "
        "headline. A 0.043 mean delta sits inside that many times over, and "
        "the sign flips between seeds"
    ),
    "more_seeds_do_not_help": (
        "seeds narrow the estimate of a MEAN; condition 1 asks about "
        "patients, and it is a universal over the seeds present so more of "
        "them is a harder bar. phase7c.TEN_SEED_ROUND_NOT_RUN"
    ),
    "licenses": (
        "descriptive orderings -- one arm SCORED higher, which is a fact "
        "about the measurements"
    ),
    "forbids": (
        "any claim that an arm was shown to be better or worse. Every "
        "'claimably' sentence becomes a 'scored higher' sentence"
    ),
    "for_the_write_up": (
        "a stronger argument for the full-face data than a thin claimed win "
        "would have been: it is measured rather than asserted, and it says "
        "what a next dataset has to be big enough to do"
    ),
}


#: **[BANKED 2026-08-31] WHAT THIS COHORT'S
#: RESOLUTION ACTUALLY IS -- in three tiers that may never be merged.**
#:
#: This record exists because a figure that is nowhere in the repo --
#: "the detection floor is a ΔPCC of about 0.12 to 0.14" -- was asserted
#: repeatedly in a measured voice and used to read results
#: interpretively. It was never measured, never registered, and never
#: entered any record (``THE_ERROR_PROVENANCE`` below, and the sweep
#: that established it). The three tiers below are separated
#: deliberately: **collapsing them is the error that produced this
#: record.**
#:
#: ---------------------------------------------------------------
#: **TIER 1 -- [MEASURED].** Each verified at its source, quoted.
#:
#: 1. **Deltas of 0.04 to 0.10 were unresolvable.**
#:    ``COHORT_CANNOT_RESOLVE["finding"]``, verbatim: *"this cohort
#:    cannot resolve PCC differences of 0.04 to 0.10 between arms. Four
#:    of four comparisons tested against PLAN §4.3's full criterion have
#:    failed condition 1"*. Confirmed at scale in the same record's
#:    ``at_scale``: **30 tested, 1 survived, 29 withdrawn** -- and the
#:    one survivor *"rests on a 0.0065 baseline and does not reproduce
#:    at the other geometry"*.
#:
#: 2. **The smallest delta ever to satisfy BOTH conditions is +0.1386.**
#:    Contrast ``p7d__b32_512_vs_patch16_512`` (b32@512 against
#:    patch16@512), ``PHASE_7D_CLOSING["verdicts"]``: delta 0.1386,
#:    5/5 excluding zero, margin 5.32x, CLAIMABLE. Ledgered as
#:    ``p7d-grid-density``.
#:
#: 3. **The margin structure**
#:    (``roadb.CONDITION_1_MARGIN_STRUCTURE``, carried by reference as
#:    ``results_ledger.MARGIN_TABLE``): floor -- *"nothing at or below
#:    2.55x has ever passed"*; ceiling -- *"everything at or above 5.32x
#:    has passed -- five for five"*; and the middle, which is the
#:    finding -- *"between 3.6x and 4.7x the outcome is MIXED and margin
#:    does not order it: 3.83 passed while 4.34 and 4.69 failed"*.
#:
#: ---------------------------------------------------------------
#: **TIER 2 -- [REASONED]. NON-GATING. NOT A THRESHOLD.**
#:
#: The practical floor lies **somewhere between 0.10 and 0.1386**:
#: above the largest delta shown unresolvable, at or below the smallest
#: shown resolvable. **Nothing has ever been tested in that band**, so
#: the boundary is **unlocated** -- not approximately known, not
#: bracketed to a useful precision, simply not measured anywhere inside
#: it.
#:
#: **This inference may not gate a phase. It may not be quoted as a
#: threshold. It may not be used to declare an effect too small to
#: pursue, or to justify not running a comparison.** Its only legitimate
#: use is as an explicitly-labelled inference in prose that also states
#: the band is untested.
#:
#: **And the reason those prohibitions are here rather than assumed:
#: stating this inference in a measured voice is precisely the error
#: that produced this record.** A reasoned bracket spoken as a
#: measurement is indistinguishable, downstream, from a measurement.
#:
#: ---------------------------------------------------------------
#: **TIER 3 -- THE QUOTABLE LINE, BOUND.**
#:
#: One citable figure: **0.1386, "the smallest resolvable difference
#: this cohort has demonstrated"**. Never "the detection floor". Never a
#: threshold. The prohibited phrasings are a tested literal --
#: ``DETECTION_FLOOR_PROHIBITION`` -- on the pattern of
#: ``phase18.DELIVERABLES_REGISTERED["cleftgnn_iem_prohibition"]``, so
#: no later turn reaches for the phrase.
SMALLEST_RESOLVABLE_DIFFERENCE = {
    "banked": "2026-08-31, the ruling on how it is to be stated",
    "why_this_record_exists": (
        "a figure nowhere in the repo -- 'the detection floor is a "
        "delta-PCC of about 0.12 to 0.14' -- was asserted in a measured "
        "voice and used to read results interpretively. The three tiers "
        "below are separated deliberately: collapsing them IS the error"
    ),

    "measured": {
        "unresolvable_band": (
            "COHORT_CANNOT_RESOLVE: 'this cohort cannot resolve PCC "
            "differences of 0.04 to 0.10 between arms' -- confirmed at "
            "scale, 30 tested, 1 survived, 29 withdrawn, and the one "
            "survivor rests on a 0.0065 baseline and does not reproduce "
            "at the other geometry"
        ),
        "smallest_resolvable": {
            "delta": 0.1386,
            "contrast": "p7d__b32_512_vs_patch16_512",
            "what": "b32@512 against patch16@512",
            "conditions": "5/5 excluding zero, margin 5.32x, CLAIMABLE",
            "home": "PHASE_7D_CLOSING['verdicts']; ledger p7d-grid-density",
        },
        "margin_structure": (
            "roadb.CONDITION_1_MARGIN_STRUCTURE (results_ledger."
            "MARGIN_TABLE by reference): nothing at or below 2.55x has "
            "ever passed; everything at or above 5.32x has passed, five "
            "for five; and between 3.6x and 4.7x the outcome is MIXED "
            "and margin does not order it -- 3.83 passed while 4.34 and "
            "4.69 failed"
        ),
    },

    "reasoned": {
        "tag": "[REASONED] -- NON-GATING, NOT A THRESHOLD",
        "the_inference": (
            "the practical floor lies somewhere between 0.10 and "
            "0.1386: above the largest delta shown unresolvable, at or "
            "below the smallest shown resolvable"
        ),
        "the_band_is_untested": (
            "NOTHING has ever been tested between 0.10 and 0.1386, so "
            "the boundary is UNLOCATED -- not approximately known, not "
            "usefully bracketed, simply not measured anywhere inside it"
        ),
        "may_not": (
            "gate a phase; be quoted as a threshold; be used to declare "
            "an effect too small to pursue; or justify not running a "
            "comparison"
        ),
        "only_legitimate_use": (
            "an explicitly-labelled inference in prose that ALSO states "
            "the band is untested"
        ),
        "why_these_prohibitions_are_written_down": (
            "stating this inference in a MEASURED VOICE is precisely "
            "the error that produced this record. A reasoned bracket "
            "spoken as a measurement is indistinguishable, downstream, "
            "from a measurement"
        ),
    },

    "quotable": {
        "the_one_figure": 0.1386,
        "the_one_phrasing": (
            "the smallest resolvable difference this cohort has "
            "demonstrated"
        ),
        "never": (
            "'the detection floor'; a threshold; a detectability limit; "
            "any number in the 0.12-0.14 range"
        ),
        "prohibition": "DETECTION_FLOOR_PROHIBITION",
        # [2026-08-31, found by the sweep] A NUMERIC COINCIDENCE, noted
        # so a future grep for 0.1386 is not misled: the same four
        # digits appear in classification.CLASSIFICATION_METRICS_BANKED
        # ['the_intervals_sharpen_it'] as macro F1's margin above its
        # majority floor (0.3614 - 0.2228 = 0.1386). DIFFERENT QUANTITY
        # -- a metric's distance from a floor on one arm, not a
        # resolvable difference between two arms. R2's shape, caught
        # before it could be confused rather than after.
        "the_0_1386_coincidence": (
            "classification.CLASSIFICATION_METRICS_BANKED carries 0.1386 "
            "as macro F1's margin above its 0.2228 majority floor "
            "(0.3614 - 0.2228). SAME DIGITS, DIFFERENT QUANTITY: a "
            "metric's distance from a floor on one arm, not a resolvable "
            "difference between two arms"
        ),
    },
}


#: **[2026-08-31] The prohibited phrasing, as a literal.** A tested
#: string, sibling to
#: ``phase18.DELIVERABLES_REGISTERED["cleftgnn_iem_prohibition"]``, so
#: no later turn reaches for the phrase and finds nothing stopping it.
DETECTION_FLOOR_PROHIBITION = (
    "THIS COHORT HAS NO MEASURED DETECTION FLOOR AND THE PHRASE IS "
    "NEVER USED. There is no threshold at 0.12-0.14 or anywhere else: "
    "that figure was never measured, never registered, and appears in "
    "no record. The single citable quantity is 0.1386 -- THE SMALLEST "
    "RESOLVABLE DIFFERENCE THIS COHORT HAS DEMONSTRATED -- and it is a "
    "demonstrated instance, not a limit: it says what was resolved "
    "once, not what can be. Deltas of 0.04-0.10 are measured "
    "unresolvable; the band between 0.10 and 0.1386 is UNTESTED and "
    "its boundary unlocated. No phase is gated on any of this."
)


#: **[RECORDED 2026-08-31, DATED] WHERE THE WRONG FIGURE CAME FROM, and
#: what it did and did not touch. Both halves, plainly.**
#:
#: **The error.** The figure "detection floor ≈ 0.12-0.14" originated in
#: CONVERSATION -- the record's, during the 2026-08-31 session, and carried
#: forward in that session's own memory summaries. It was **stated as
#: measured**, without a `[REASONED]` tag and without a record name
#: beside it, and it was used to read results interpretively: to judge
#: whether an observed difference was worth remarking on, and how close
#: a withdrawn contrast had come.
#:
#: **What it did NOT touch, and this half matters equally.** **No
#: verdict was affected.** The machinery never consulted it: every
#: claim ran through PLAN §4.3's two conditions -- per-seed paired BCa
#: excluding zero, and the delta exceeding the two arms' combined seed
#: uncertainty -- computed from the arms' own SDs by
#: ``phase3.combined_claimable_delta``. No threshold in any config, task
#: or test ever read a 0.12-0.14 figure, because it existed nowhere to
#: be read. Every CLAIMABLE, WITHDRAWN and UNRESOLVED row in the ledger
#: stands exactly as computed.
#:
#: **So the damage was interpretive, not evidential** -- which is the
#: less serious kind and the harder kind to notice, because nothing
#: fails when a narration drifts from the machinery it narrates.
#:
#: **The sweep.** Every record was searched (2026-08-31) for the figure
#: and for phrasings built on it -- "detection floor", "detection
#: limit", the 0.12/0.13/0.14 range, "below the floor", "inside the
#: floor", "claimable effect" -- across ``src/``, ``docs/``, ``tests/``,
#: ``configs/`` and the session's memory files. **Zero genuine
#: instances.** Every numeric hit is a different quantity (CleftGNN's
#: Study-Set mean ~0.14; the ViT resolution drop ~0.146; the AdaBN
#: probe's ~0.134 reading; the macro-F1 and majority floors; the
#: IEM gradient crossover exponents), and the only occurrences of the
#: phrase itself are in ``docs/RECORD_STATE_2026-08-31.md``, where they
#: state that it is NOT in the repo.
#:
#: **The correction is therefore preventive, not remedial**: nothing had
#: to be rewritten. That is the good outcome, and it is only knowable
#: because the sweep was run rather than assumed.
THE_ERROR_PROVENANCE = {
    "recorded": "2026-08-31",
    "the_figure": "detection floor ~= 0.12-0.14 delta-PCC",
    "origin": (
        "CONVERSATION -- the record's, during the 2026-08-31 session, and "
        "carried forward in that session's own memory summaries"
    ),
    "how_it_was_stated": (
        "as MEASURED: no [REASONED] tag, no record name beside it"
    ),
    "what_it_governed": (
        "interpretation -- whether an observed difference was worth "
        "remarking on, and how close a withdrawn contrast had come"
    ),
    "what_it_did_not_touch": (
        "**NO VERDICT WAS AFFECTED.** The machinery never consulted it: "
        "every claim ran PLAN 4.3's two conditions -- per-seed paired "
        "BCa excluding zero, and the delta exceeding the arms' combined "
        "seed uncertainty from phase3.combined_claimable_delta. No "
        "threshold in any config, task or test ever read a 0.12-0.14 "
        "figure, because it existed nowhere to be read. Every "
        "CLAIMABLE, WITHDRAWN and UNRESOLVED row stands as computed"
    ),
    "the_damage": (
        "INTERPRETIVE, not evidential -- the less serious kind and the "
        "harder to notice, because nothing fails when a narration "
        "drifts from the machinery it narrates"
    ),
    "the_sweep": (
        "every record searched 2026-08-31 for the figure and for "
        "phrasings built on it across src/, docs/, tests/, configs/ and "
        "the session's memory files. ZERO genuine instances: every "
        "numeric hit is a different quantity, and the only occurrences "
        "of the phrase are in docs/RECORD_STATE_2026-08-31.md, which "
        "states it is NOT in the repo"
    ),
    "the_correction_is_preventive": (
        "nothing had to be rewritten -- the good outcome, and only "
        "knowable because the sweep was run rather than assumed"
    ),
}


#: **[MEASURED 2026-08-02] Swin disagrees with the other three backbones on
#: BOTH questions, claimably, and no arm involved is defective.**
#:
#: Q1 +0.2027 and Q2 -0.1067, against a threshold of 0.0529 from Swin's own
#: SDs (0.0414 / 0.0439 / 0.0414, five seeds). Q1 clears ``single_run_95``
#: (0.1183) as well, so it survives without averaging.
#:
#: **The collapse hypothesis is refuted by shrinkage.** Swin's ImageNet arm
#: has shrinkage **0.4299 -- the HIGHEST of its three** -- so the head fitted
#: normally and produced a near-zero correlation. A collapsed arm shows
#: shrinkage near zero (PLAN §4.3). 0.0065 is a weak representation, not a
#: broken run, and 0.2092 at shrinkage 0.3936 is an ordinary arm that performs
#: well. **Both are valid measurements**, which is what makes this a conflict
#: between backbones rather than an artefact in one of them.
#:
#: ----------------------------------------------------------------------------
#: THE SHARED-TERM ARGUMENT STANDS; THE READING BUILT ON IT DID NOT
#: ----------------------------------------------------------------------------
#: Q1 is ``original - imagenet`` and Q2 is ``masked - original``, so
#: ``original`` is the shared term entering with opposite signs: one
#: upward-biased ``original`` cell inflates Q1 and deflates Q2 at once, and a
#: depressed ``imagenet`` cell moves Q1 only. That argument identified the
#: only cell whose bias could explain both anomalies, and it was right to
#: point at ``scut_original`` rather than at the ImageNet arm everyone looks
#: at first.
#:
#: **What it does NOT establish is that any cell is biased at all** -- and
#: this record previously read it as though it did. With shrinkage in hand,
#: nothing about ``swin_b scut_original`` is anomalous: ordinary SD, ordinary
#: shrinkage, high PCC.
#:
#: **[R2, and it is mine] The residual answers a different question from the
#: one it was asked.** Residuals against an additive backbone+init model
#: measure where ADDITIVITY BREAKS, which is a backbone x init INTERACTION.
#: The +0.0947 at ``swin_b scut_original`` is real and correctly computed. It
#: was read as "which cell is anomalous", and when every arm is well-behaved
#: the same number reads as **"the backbones respond differently to the init,
#: and Swin most of all"** -- a finding rather than a defect. Same quantity,
#: different question, and the name "outlier" is what carried the wrong one.
#: Seventh instance in PLAN R2's tally, and the first where the misreading was
#: in a diagnostic built during Phase 7 rather than inherited.
#:
#: **[REASONED] A mechanism, unverified.** Swin's shifted windows pool over
#: local regions where ViT attends globally, which is a plausible reason to
#: respond differently to a masked field of view. Note this is the SAME
#: architectural difference Stage C's tiling prior rests on -- there it
#: predicts ViT separates at G1 and Swin does not. One architectural fact, two
#: predictions, and Stage D1 runs every backbone at G1, so it bears on both.
#:
#: ----------------------------------------------------------------------------
#: HELD AS CONTESTED, AND WHAT DECIDES IT
#: ----------------------------------------------------------------------------
#: ``swin_b__imagenet__g1`` is in the Stage D1 extraction batch: the same
#: weights at the other geometry, extracted in the same run.
#:
#: * **near-zero at G1 too** -> a property of Swin's ImageNet representation
#:   on this task, the conflict is real, and Q1 reports as four backbones
#:   disagreeing.
#: * **normal at G1** -> something specific to the G2 pipeline, and the Swin
#:   G2 cells are the ones to re-examine.
#:
#: Still worth reading when D1 lands: ``normalization_source`` per Swin set
#: from the extraction summary. ``factory.normalization_for`` reads each
#: model's own ``pretrained_cfg`` BECAUSE the four backbones split two-two --
#: ViT-B/16 at 0.5/0.5, Swin-B on ImageNet statistics -- so Swin sets
#: disagreeing on their source would be a defect in one of them.
#:
#: **Q1 stays unreported until D1**, and it is a result either way: no
#: consistent benefit from facial beauty pretraining, with the backbones
#: disagreeing about it. That is an answer to an open question (PLAN Part 6,
#: brief §1), not a failure to find one.
STAGE_D_BACKBONE_CONFLICT = {
    "measured": "2026-08-02",
    #: **[RESOLVED 2026-08-02] The deciding measurement was read.**
    #: ``swin_b__imagenet__g1`` = 0.1076 at t = 11.09, so the near-zero G2
    #: cell is G2-specific and not a property of Swin's ImageNet
    #: representation -- the branch this record named in advance.
    #: ``SWIN_G2_IMAGENET_CELL`` carries the account. Both of the
    #: disagreements below are therefore artefacts of the operating point:
    #: Q1's +0.2027 falls to +0.0114 and stops being claimable, and Q2's
    #: conflict does not reproduce at G1 (``STAGE_D1_AT_G1``).
    #:
    #: **Kept rather than deleted.** The conflict framing was correct at G2
    #: and the write-up needs the G2 table; what changed is that a second
    #: operating point exists to read it against. Deleting the record would
    #: leave the G2 numbers with no account of why they do not generalise.
    "status": "RESOLVED 2026-08-02 -- G2-specific, see SWIN_G2_IMAGENET_CELL",
    "resolved_by": {
        "set": "swin_b__imagenet__g1",
        "value": 0.1076, "t": 11.09,
        "branch_taken": "normal at G1 -> something specific to the G2 pipeline",
    },
    "backbone": "swin_b",
    "disagrees_on": ["Q1", "Q2"],
    "threshold": 0.0529,
    "collapse_refuted_by": {
        "shrinkage": 0.4299,
        "reading": (
            "the HIGHEST of Swin's three, so the head fitted normally and "
            "produced a near-zero correlation -- weak representation, not a "
            "broken run"
        ),
    },
    #: The arithmetic is unchanged and still recorded; what changed is what it
    #: is taken to mean. See the R2 note above.
    "largest_additive_residual": {"cell": "swin_b scut_original", "value": 0.0947},
    "next_largest": {"cell": "swin_b imagenet", "residual": -0.0544},
    "residual_measures": (
        "where an additive backbone+init model breaks -- an INTERACTION. With "
        "every arm well-behaved that is a finding about backbones responding "
        "differently to the init, not a flagged cell"
    ),
    "shared_term_argument": (
        "original enters Q1 and Q2 with opposite signs, so it is the only "
        "single cell whose bias could explain both anomalies. It identifies "
        "WHICH cell to suspect; it does not establish that any cell is biased"
    ),
    "mechanism_reasoned": (
        "Swin's shifted windows pool locally where ViT attends globally -- the "
        "same architectural difference Stage C's tiling prior rests on"
    ),
    "decided_by": "swin_b__imagenet__g1: near-zero at G1 -> representation; "
                  "normal at G1 -> G2 pipeline",
    "blocks": "Q1",
}

#: **[REFUTED 2026-08-02] "Seed SD is monotone in warm-start" -- predicted
#: from AG-Net, tested on SR-GNN, false.**
#:
#: AG-Net's bands are a clean 0.0567 / 0.0457 / 0.0301 across imagenet,
#: scut_original and scut_masked, and the mechanism offered for that ordering
#: was warm-start: an imagenet graph arm seed-initialises its entire trainable
#: stack while a pretrained one starts from a checkpoint, so seed variance
#: should be widest at imagenet and narrowest at masked. **That is a specific
#: prediction with a mechanism, and it names the arm that tests it.**
#:
#: SR-GNN gives **0.0350 / 0.0402 / 0.0251**. Its widest arm is
#: ``scut_original``, not ``imagenet``. The prediction fails on the one
#: backbone it named, so AG-Net's ordering is a property of AG-Net rather than
#: of graph arms, and the mechanism is not doing the work it was credited
#: with.
#:
#: **What survives is weaker and is stated as such: the masked arm has the
#: narrowest band of its backbone's three.** Strictly in ViT (0.0142), SR-GNN
#: (0.0251) and AG-Net (0.0301), and tied-narrowest in Swin (0.0414, level
#: with its imagenet arm). No ordering is claimed among the other two.
#:
#: **And that pattern spans BOTH regimes, which rules the mechanism out
#: rather than merely failing to support it.** ViT and Swin train a
#: 769-parameter head over frozen embeddings (PLAN §4.7) -- there is no
#: trainable stack to warm-start at all -- so whatever makes masked arms
#: narrowest cannot be the thing the refuted prediction proposed. Left
#: recorded as an observation with no mechanism attached, which is the honest
#: state: inventing a second mechanism to fit four numbers is how the first
#: one got quoted.
#:
#: **Every band here is its own arm's** (PLAN §4.12.1). Reading AG-Net's
#: 0.0567 "against the graph band's 0.0251" was the inherited-band comparison
#: the plan forbids by name -- 0.0251 is SR-GNN's, and it turns out to be
#: SR-GNN's *masked* arm specifically, the narrowest of its three. The
#: recorded ~2.3e-05 same-seed nondeterminism (``deterministic: false``) is a
#: floor three orders of magnitude below any of this and explains none of it.
STAGE_D_SEED_BANDS = {
    "measured": "2026-08-02",
    #: The twelve bands themselves live in STAGE_D_AT_G2["sd"] -- one copy,
    #: because this record is about what they do and not what they are.
    "refuted": {
        "claim": "seed SD is monotone in warm-start (widest at imagenet)",
        "predicted_from": "agnet",
        "tested_on": "srgnn",
        "mechanism_credited": (
            "an imagenet graph arm seed-initialises its entire trainable "
            "stack; a pretrained one starts from a checkpoint"
        ),
        "outcome": (
            "srgnn's widest band is scut_original (0.0402), not imagenet "
            "(0.0350). AG-Net's ordering is AG-Net's, not graph arms'"
        ),
    },
    "survives": {
        "claim": "the masked arm has the narrowest band of its backbone's three",
        "strict_in": ["vit_b16", "srgnn", "agnet"],
        "tied_in": ["swin_b"],
        "mechanism": None,
        "why_not_warm_start": (
            "it holds in the transformers too, and they train a 769-parameter "
            "head over frozen embeddings -- no stack to warm-start"
        ),
    },
    "bands_are_never_inherited": (
        "PLAN §4.12.1. 0.0251 is SR-GNN's masked arm, the narrowest of its "
        "three, and was quoted as though it were a band for graph arms"
    ),
    "nondeterminism_floor": 2.3e-05,
}

#: **[MEASURED 2026-08-01] ONE WEAK CHECKPOINT, THREE PHANTOM FINDINGS.**
#:
#: ``vit_b16 masked_g1`` is the weakest of the twelve pretraining checkpoints
#: -- SCUT test PCC **0.7893**, the lowest number in the table, against
#: masked_g2's 0.8306. Because ``scut_masked`` is GEOMETRY-BOUND
#: (``embeddings.VARIANT_FOR_INIT``), every comparison that moves geometry on
#: a masked init silently swaps that checkpoint in, and its weakness is read
#: as whatever the comparison claims to measure.
#:
#: It generated three separate "findings", each of which dissolved under a
#: comparison that held the checkpoint constant:
#:
#: 1. **A geometry x init sign flip.** ViT masked G1 0.0830 against G2 0.2001
#:    looked like G2 beating G1 at a pretrained init, opposite to ImageNet.
#:    Stage C0 held the checkpoint constant and found G1 ahead at
#:    ``scut_original`` too. No sign flip.
#: 2. **Stage C's mechanism result.** Read as the tiling prediction confirmed.
#:    It is a matched-PIPELINE comparison and cannot test that prediction;
#:    the ImageNet pair does, and confirms it.
#: 3. **Q2's flip.** +0.0595 at G2 against -0.1122 at G1 looked like the
#:    masking answer reversing with geometry. The difference-in-differences is
#:    -0.1717, of which the two clean geometry measurements account for
#:    +0.0546 in the OPPOSITE direction -- the whole flip is the checkpoint.
#:
#: **The pattern, named so it is recognised rather than rediscovered:** a
#: single weak artifact, bound by a rule to one level of a factor, is
#: indistinguishable from an effect of that factor. Every instance was caught
#: only by a comparison that held the artifact constant, and none by review of
#: the number itself. That is why ``resolved_inputs`` and ``also_varies``
#: exist: the declared fields said "one factor" in all three cases.
#:
#: Fourth instance of the same artifact, and the same class as
#: ``graph_cleft.PRETRAINING_DOES_NOT_PREDICT_TRANSFER`` -- checkpoint
#: identity dominating nearly every other factor measured so far.
MASKED_G1_ARTEFACT = {
    "measured": "2026-08-01",
    "checkpoint": "vit_b16 masked_g1",
    "scut_test_pcc": 0.7893,
    "against": {"masked_g2": 0.8306, "rank": "lowest of the twelve"},
    "why_it_propagates": (
        "scut_masked is GEOMETRY-BOUND, so any comparison moving geometry on "
        "a masked init swaps this checkpoint in without declaring it"
    ),
    "phantom_findings": [
        "a geometry x init sign flip (refuted by Stage C0)",
        "Stage C's tiling-mechanism result (it is a pipeline comparison)",
        "Q2 reversing with geometry (the DiD is the checkpoint, not geometry)",
    ],
    "each_caught_by": "a comparison holding the checkpoint constant, never by review",
    "generalisation": (
        "a single weak artifact bound by a rule to one level of a factor is "
        "indistinguishable from an effect of that factor"
    ),
}


#: **[MEASURED 2026-08-02] LDL DOES NOT HELP. A clean negative on the one
#: idea in the project that addressed label noise directly.**
#:
#: Both label triples are in, at five seeds each, every delta re-derived from
#: the two arms' own SDs:
#:
#: ===================  =======  ======  ========  =========  ==============
#: triple                  arm     PCC     delta   threshold  verdict
#: ===================  =======  ======  ========  =========  ==============
#: G1 imagenet/g1        mean    0.2520        --         --  --
#: G1 imagenet/g1        median  0.2483   -0.0037     0.0168  not claimable
#: G1 imagenet/g1        ldl     0.2336   -0.0184     0.0297  not claimable
#: G  masked/g2          mean    0.2001        --         --  --
#: G  masked/g2          median  0.1406   -0.0595     0.0162  CLAIMABLE
#: G  masked/g2          ldl     0.1583   -0.0418     0.0235  CLAIMABLE
#: ===================  =======  ======  ========  =========  ==============
#:
#: **Every measured label delta is NEGATIVE**, at both operating points and
#: for both alternatives. Two are inside their thresholds and two clear them;
#: none is positive anywhere. **The mean is not beaten by either alternative
#: at either operating point.**
#:
#: **LDL specifically.** It was the one idea addressing label noise directly
#: rather than working around it -- the soft columns hold the full five-rater
#: distribution per patient, and it needed a different head and a different
#: loss to use them (brief §1). The answer is that **the distribution carries
#: no information the mean discards that this head can use**: null at
#: imagenet/G1 and claimably harmful at masked/G2. Recorded as what it is, a
#: clean negative on a real idea, which is a result and not a disappointment.
#:
#: **The mean is confirmed as the target by TWO INDEPENDENT ROUTES**, and
#: that is the part worth carrying:
#:
#: 1. **Phase 1 learnability**, before any ladder arm ran -- mean 0.6022
#:    against median 0.5708 (PLAN §4.2 records 0.5900/0.5561 on the earlier
#:    cut; the ordering is what is stable, not the third decimal).
#: 2. **The ladder**, now -- median claimably worse at masked/G2 and null at
#:    imagenet/G1.
#:
#: Different data, different procedure, same conclusion. A target chosen on
#: one measurement and confirmed by another is a stronger claim than either,
#: and it is the kind of agreement this project has mostly NOT had -- the
#: scheme axis needed four attempts, and SCUT test PCC turned out not to
#: predict cleft transfer at all.
#:
#: **One delta survives ``single_run_95``**: median at masked/G2, 0.0595
#: against 0.0362. That is the second in the whole ladder, after Swin's Q1 --
#: worth stating because PLAN §4.3 says to make the stronger claim when it is
#: true, and it is true here and almost nowhere else.
#:
#: **The two triples DISAGREE, and it was predicted.** Null at one operating
#: point, claimable at the other, across cells differing in geometry AND
#: init. ``LABEL_GEOMETRY_CONTROL_GEOMETRY`` is the arm bought to attribute
#: it. Until that runs, the honest statement is the one at the top -- no
#: label alternative beats the mean anywhere -- which does NOT depend on the
#: attribution.
STAGE_G_LABEL_FORMULATION = {
    "measured": "2026-08-02",
    "seeds": 5,
    "triples": {
        "imagenet__g1": {
            "stage": "G1",
            "arms": {
                "mean": {"pcc": 0.2520, "sd": 0.0148},
                "median": {"pcc": 0.2483, "sd": 0.0121},
                "ldl": {"pcc": 0.2336, "sd": 0.0305},
            },
            "rederived": {
                "median": {
                    "delta": -0.0037, "threshold": 0.0168, "claimable": False,
                    "single_run_95": 0.0375, "survives_single_run_95": False,
                },
                "ldl": {
                    "delta": -0.0184, "threshold": 0.0297, "claimable": False,
                    "single_run_95": 0.0664, "survives_single_run_95": False,
                },
            },
            "verdict": "label is NULL at this operating point",
        },
        "scut_masked__g2": {
            "stage": "G",
            "arms": {
                "mean": {"pcc": 0.2001, "sd": 0.0142},
                "median": {"pcc": 0.1406, "sd": 0.0118},
                "ldl": {"pcc": 0.1583, "sd": 0.0228},
            },
            "rederived": {
                "median": {
                    "delta": -0.0595, "threshold": 0.0162, "claimable": True,
                    "single_run_95": 0.0362, "survives_single_run_95": True,
                },
                "ldl": {
                    "delta": -0.0418, "threshold": 0.0235, "claimable": True,
                    "single_run_95": 0.0526, "survives_single_run_95": False,
                },
            },
            "verdict": "both alternatives claimably WORSE than the mean",
        },
    },
    "verdict": (
        "LDL does not help -- null at one operating point, claimably harmful "
        "at the other. No label alternative beats the mean anywhere"
    ),
    "ldl_reading": (
        "the five-rater distribution carries no information the mean discards "
        "that this head can use"
    ),
    "mean_confirmed_by": [
        "Phase 1 learnability, before any arm ran: mean 0.6022 > median 0.5708",
        "the ladder: median claimably worse at masked/g2, null at imagenet/g1",
    ],
    "disagreement": (
        "the two triples differ in geometry AND init, so which one explains "
        "the disagreement is unattributable until Stage G0 runs. The headline "
        "-- nothing beats the mean -- does not depend on that attribution"
    ),
}

#: **[DECIDED 2026-08-02] STAGE B IS NOT RUN, and it is recorded as a decision
#: rather than as a gap.**
#:
#: The view ablation -- frontal, basal, both -- needs **basal staging, which
#: does not exist**. Phase 2 built frontal only, and a submental view needs
#: its own trapezium placement and its own contact-sheet review: Phase
#: 2-shaped work, inside Phase 7, for one question, with every other stage
#: answered.
#:
#: **The same treatment the TSTR arm got** (PLAN Part 6, "Stage F is struck").
#: That arm was built and parked because both its outcomes were
#: uninterpretable; this one is not built because the cost is a phase and the
#: ladder does not depend on it. In both cases what is dropped is the CLAIM
#: that the ladder contains the arm -- PLAN's own warning, that an arm list
#: promising something the project does not deliver is the inconsistency that
#: survives into a write-up.
#:
#: **What stays untested, and must be stated as untested.** PLAN §4.6 and the
#: brief both carry it: raters saw frontal and basal together and gave one
#: score covering both, so a frontal-only model predicts a label from evidence
#: it cannot see, and **[REASONED]** that this explains the r≈0.3 plateau.
#: That reasoning is unchanged and remains **[REASONED]**. PLAN's provenance
#: rule is that a [REASONED] item must not gate a phase until checked or
#: explicitly accepted as an assumption -- this accepts it as an assumption,
#: explicitly, and the write-up says the ablation that would test it was not
#: run and why.
#:
#: **It is a limitation with a stated reason, not a defect.** The distinction
#: matters for how it reads: "we did not measure this, here is the cost" is an
#: honest boundary; "the missing view explains the plateau" would be a claim
#: this project has no measurement for. Only the first is available.
STAGE_B_NOT_RUN = {
    "decided": "2026-08-02",
    "stage": "B",
    "arms": ("frontal", "basal", "both"),
    "status": "NOT RUN -- out of scope, deliberately",
    "blocker": (
        "basal staging does not exist. Phase 2 built frontal only, and a "
        "submental view needs its own trapezium placement and its own "
        "contact-sheet review"
    ),
    "cost": "Phase 2-shaped work inside Phase 7, for one question",
    "precedent": "the TSTR arm, parked at PLAN Part 6 -- what is dropped is "
                 "the CLAIM that the ladder contains it",
    "claim_left_untested": {
        "text": (
            "the missing basal view explains the r~0.3 plateau -- raters saw "
            "both views and gave one score, so a frontal-only model predicts "
            "a label from evidence it cannot see"
        ),
        "provenance": "REASONED",
        "status": "accepted as an assumption, explicitly, and not measured",
        "plan_rule": (
            "PLAN's provenance tagging -- a [REASONED] item must not gate a "
            "phase until checked or explicitly accepted as an assumption"
        ),
        # **[CORRECTED 2026-08-23, BASAL_RATIONALE_UNSUPPORTED]** This
        # item's PREMISE -- "raters saw both views and gave one score" --
        # was carried by PLAN 4.6 as [MEASURED] and was never measured.
        # Checked against the score sheet and the instrument literature:
        # the sheet is keyed to frontal IDs and no basal ID receives a
        # score; the instrument and the collaborator's own protocol
        # describe frontal-based rating. Whether raters nonetheless SAW
        # the basal remains UNRESOLVED, so the premise is UNSUPPORTED,
        # not refuted -- and an explanation resting on an unsupported
        # premise cannot be relied on. The text above is preserved as
        # what was believed.
        "corrected": (
            "2026-08-23: the premise is UNSUPPORTED, not refuted "
            "(ladder.BASAL_RATIONALE_UNSUPPORTED). The explanation cannot "
            "be relied on until the premise is settled -- one sentence "
            "from supervision decides it"
        ),
    },
    "write_up": (
        "state that the ablation was not run and why. A limitation with a "
        "stated reason, not a gap, and not evidence either way about the "
        "plateau"
    ),
}


#: **[CORRECTION 2026-08-23] THE BASAL RATIONALE WAS NEVER MEASURED. THE
#: CLAIM IS DOWNGRADED TO UNSUPPORTED -- NOT TO REFUTED.**
#:
#: **The original claim, preserved as what was believed** (PLAN 4.6,
#: tagged **[MEASURED]**; quoted into ``STAGE_B_NOT_RUN``'s premise):
#:
#:     "Basal is core, not optional: raters saw frontal and basal
#:     together and gave one score per patient covering both; a
#:     frontal-only model predicts a label from evidence it cannot see."
#:
#: **Checked against the artifacts on 2026-08-23, two independent lines:**
#:
#: **[MEASURED]** -- the declared primary score sheet
#: (``${CLEFT_SCORESHEET}``), run today: 251 rows, IDs 241-742. Folder
#: 143's frontal (524) is in the sheet; **its basal (523) is not**.
#: Folder 238's single frontal (581) is present. Across all 251 rows
#: there are only **2 adjacent-ID pairs both present** -- consistent with
#: coincidence, not with per-view scoring. **The sheet is keyed to
#: frontal image IDs and carries no basal IDs; no basal ID receives a
#: score.** And the contradiction was in the PLAN itself all along: its
#: own section-2 inventory says "basal (unscored) | 236", five sections
#: above the [MEASURED] claim.
#:
#: **[LITERATURE]** -- via NotebookLM over the instrument papers and the
#: CleftGNN manuscript: Asher-McDade's method shows raters masked
#: **frontal and lateral (profile)** views -- not basal. The national
#: three-view standard (frontal / submental / lateral) is a PHOTOGRAPHY
#: CAPTURE protocol, not a rating protocol. The CleftGNN rating protocol
#: shows raters a single cropped Frontal-Eye-View.
#:
#: **[UNRESOLVED]** -- whether raters were nonetheless SHOWN the basal
#: view alongside the frontal while recording one score under the
#: frontal's ID. **Nothing in the sheet could distinguish that from
#: frontal-only rating**, which is exactly why the downgrade stops at
#: UNSUPPORTED: the sheet's silence is evidence the claim was never
#: measured, not evidence it is false.
#:
#: **The linked item is corrected with it**: ``STAGE_B_NOT_RUN``'s
#: [REASONED] "the missing basal view explains the r~0.3 plateau" is no
#: longer merely unverified -- **its premise is unsupported, so the
#: explanation cannot be relied on**, and its record now says so in
#: place.
#:
#: **THE ERROR'S CLASS, because it is the SIXTH of its kind**: a claim
#: reasoned from a plausible account of how an artifact must have worked,
#: tagged with a provenance it did not have, never checked against an
#: artifact that had been on disk since Phase 1 -- and settled today by
#: one command. The five predecessors, each the same move: the
#: consensus-column premise and the MODE reading (both refuted by opening
#: the sheet, phase10.CONSENSUS_LABEL_IS_THE_MEDIAN); "the notebook
#: states no batch size", written from one cell of it
#: (phase10.NOTEBOOK_RECIPE_IS_ADAM); grid_rois attributed to the
#: notebook on the amendment's word (phase10.ROI_CHECK_MEASURED,
#: corrected); and the breach direction read from the wrong field
#: (phase11.DOMAIN_BREACHES_OBSERVED).
#:
#: **THE LESSON**: *a rationale is not a measurement, and tagging it
#: MEASURED makes it unfalsifiable in practice* -- verification effort
#: flows to the items tagged REASONED, so a mis-tagged rationale is
#: precisely the claim nobody audits. This one gated a phase's design for
#: three weeks and sat one command away from its own check the entire
#: time.
#:
#: **FOR SUPERVISION, the highest-value question on the ask list**: "When the
#: five raters scored these 251 patients, were they shown the frontal
#: photograph only, or the frontal and submental together, with one score
#: recorded under the frontal's ID?" One sentence decides which
#: experiment Phase 12 is: restoring evidence the label already contains,
#: or measuring what a view the raters never saw adds.
BASAL_RATIONALE_UNSUPPORTED = {
    "corrected": "2026-08-23",
    "original_claim_preserved": (
        "Basal is core, not optional: raters saw frontal and basal "
        "together and gave one score per patient covering both; a "
        "frontal-only model predicts a label from evidence it cannot see "
        "-- PLAN 4.6, tagged [MEASURED]"
    ),
    "downgrade": (
        "MEASURED rationale -> UNSUPPORTED. NOT refuted -- the difference "
        "matters and the record must not overstate it"
    ),
    "measured": (
        "the sheet (run 2026-08-23): 251 rows, IDs 241-742; folder 143's "
        "frontal 524 present, its basal 523 ABSENT; folder 238's single "
        "frontal 581 present; only 2 adjacent-ID pairs both present in "
        "251 rows -- coincidence, not per-view scoring. The sheet is "
        "keyed to frontal IDs and no basal ID receives a score. The "
        "PLAN's own section-2 inventory already said 'basal (unscored)'"
    ),
    "literature": (
        "NotebookLM over the instrument papers and the CleftGNN "
        "manuscript: Asher-McDade masks frontal and LATERAL, not basal; "
        "the national audit's three-view standard is a CAPTURE protocol; "
        "CleftGNN's "
        "rating protocol shows a single cropped Frontal-Eye-View"
    ),
    # [UPGRADED 2026-08-23] The [LITERATURE] half above was the
    # NotebookLM-derived reading. All three primaries (Asher-McDade 1991,
    # Shaw 1992 Part 1, Asher-McDade 1992 Part 4) are now obtained and
    # read, and the reading HELD at source: frontal and lateral only in
    # all three, masked with card overlays to the nasolabial strip,
    # 1991's stated reason (judges influenced by general attractiveness)
    # cited verbatim by Part 4. phase12.PRIMARY_SOURCES_BANKED.
    "literature_upgraded_to_primary": (
        "2026-08-23: all three primaries read; the derived reading held "
        "at source -- frontal and lateral only, card overlays to the "
        "nasolabial strip (phase12.PRIMARY_SOURCES_BANKED). The "
        "UNRESOLVED item below is untouched: the primaries settle the "
        "INSTRUMENT, not what our panel was shown"
    ),
    # [2026-08-23, as written] Preserved exactly. RESOLVED 2026-08-31 by
    # the manuscripts -- see upgraded_to_contradicted_2026_08_31 below.
    "unresolved": (
        "[RESOLVED 2026-08-31 -- see "
        "upgraded_to_contradicted_2026_08_31] whether raters were "
        "nonetheless SHOWN the basal alongside the frontal while "
        "recording one score under the frontal's ID -- nothing in the "
        "sheet could distinguish that from frontal-only rating, which is "
        "why the downgrade stops at UNSUPPORTED"
    ),

    # -----------------------------------------------------------------
    # [UPGRADED 2026-08-31] UNSUPPORTED -> **CONTRADICTED**.
    #
    # Everything above is preserved with its 2026-08-23 reasoning
    # intact. Nothing in it was wrong; it was INCOMPLETE, and it said so.
    # -----------------------------------------------------------------
    "upgraded_to_contradicted_2026_08_31": (
        "**UNSUPPORTED -> CONTRADICTED.** the maintainer verified the rating "
        "procedure against the supervisor's own documents in NotebookLM: "
        "each rater independently scored ONE CROPPED FRONTAL-EYE-VIEW "
        "image per patient, and NO BASAL, SUBMENTAL, PROFILE OR LATERAL "
        "VIEW WAS SHOWN TO ANYONE "
        "(phase12.RATING_PROCEDURE_DOCUMENTED, quoted verbatim with "
        "each source document named)"
    ),
    "what_changed_and_why_it_could_not_have_before": (
        "**the earlier evidence measured the ARTIFACT; the manuscripts "
        "describe the PROCEDURE.** The sheet measurement and the three "
        "instrument primaries could establish only that NOTHING "
        "DISTINGUISHED frontal-only rating from both-shown -- which is "
        "exactly why the downgrade stopped at UNSUPPORTED and said so. "
        "The manuscripts describe what the raters were given, directly, "
        "and settle it"
    ),
    "both_bodies_of_evidence_stand_together": (
        "the 2026-08-23 corroboration is NOT superseded: 251 rows keyed "
        "to frontal IDs, folder 143's absent basal 523, only 2 "
        "coincidental adjacent-ID pairs, and all three instrument "
        "primaries using frontal and LATERAL. The new documentary "
        "evidence is what converts a body of consistent absences into a "
        "positive description of the procedure. Together: the "
        "instrument excludes basal, the sheet never scored it, and the "
        "manuscripts say the panel never saw it"
    ),
    "the_claim_is_now_false_not_merely_unbacked": (
        "'raters saw frontal and basal together' is CONTRADICTED by the "
        "documents. The distinction that mattered at UNSUPPORTED -- "
        "'NOT refuted, and the record must not overstate it' -- no "
        "longer applies in that direction: overstating the evidence "
        "AGAINST it would now be the error, and the record must not "
        "understate it either"
    ),
    "what_it_licenses_downstream": (
        "phase12.SENTENCE_SELECTED -- if_raters_saw_frontal_only selects "
        "on this evidence. NOTHING MEASURED CHANGES: the arms and every "
        "figure are identical under both readings, as the registration "
        "promised. The linked [REASONED] plateau inference is VOID, not "
        "merely unrelied-on (see linked_item_corrected)"
    ),
    "linked_item_corrected": (
        "STAGE_B_NOT_RUN's [REASONED] plateau explanation: its premise is "
        "unsupported, so the explanation cannot be relied on -- marked in "
        "place, dated. **[2026-08-31] The premise is now CONTRADICTED, "
        "so the inference is VOID, not merely unrelied-on** -- and the "
        "PLAN's own copy of it (docs/PLAN.md line 865) was never "
        "touched by the 2026-08-23 marking; corrected there now"
    ),
    "for_supervisor_2026_08_31": (
        "**DOCUMENTARILY ANSWERED, PENDING IN-PERSON CONFIRMATION.** "
        "The question below stands as asked but is now a CONFIRMATION "
        "rather than an open ask: the manuscripts answer it, and supervision may "
        "still know the panel was shown material the manuscripts do not "
        "describe. See phase12.SUPERVISOR_QUESTION_8_DOCUMENTARILY_ANSWERED"
    ),
    "error_class": {
        "instance": "the SIXTH of its kind",
        "class": (
            "a claim reasoned from a plausible account of how an artifact "
            "must have worked, tagged with a provenance it did not have, "
            "never checked against an artifact on disk since Phase 1 -- "
            "settled today by one command"
        ),
        "predecessors": (
            "the consensus-column premise "
            "(phase10.CONSENSUS_LABEL_IS_THE_MEDIAN)",
            "the MODE reading (same record)",
            "'the notebook states no batch size', from one cell "
            "(phase10.NOTEBOOK_RECIPE_IS_ADAM)",
            "grid_rois attributed to the notebook on the amendment's word "
            "(phase10.ROI_CHECK_MEASURED, corrected)",
            "the breach direction read from the wrong field "
            "(phase11.DOMAIN_BREACHES_OBSERVED)",
        ),
    },
    "lesson": (
        "A RATIONALE IS NOT A MEASUREMENT, AND TAGGING IT MEASURED MAKES "
        "IT UNFALSIFIABLE IN PRACTICE -- verification effort flows to the "
        "items tagged REASONED, so a mis-tagged rationale is precisely "
        "the claim nobody audits"
    ),
    "for_supervisor": (
        "When the five raters scored these 251 patients, were they shown "
        "the frontal photograph only, or the frontal and submental "
        "together, with one score recorded under the frontal's ID? One "
        "sentence decides which experiment Phase 12 is"
    ),
}


#: **[REGISTERED 2026-08-14, BEFORE ANY PHASE 7D ARTIFACT EXISTS] The patch
#: axis, its five arms, and the amendment's fifth-arm arithmetic corrected
#: by measurement before anything runs.**
#:
#: **This is not a repeat of Road B's resolution branch, and the record says
#: so** (amendment §2). Road B fed a BIGGER IMAGE to a 16x16-patch model,
#: interpolating the learned position grid from 14x14 to 32x32/48x48, and
#: measured the cliff: 0.2319 -> 0.0857. Phase 7D changes the PATCH SIZE at
#: a fixed 224 input, so each model runs on its own native grid and nothing
#: is interpolated -- the cliff's mechanism does not apply to arms 1-4.
#:
#: **The concat arm carries a measured prior against it, registered here so
#: it is read with the result**: Phase 7B tested embedding concatenation as
#: an axis on this cohort and a +0.0221 inner-validation lead became a
#: -0.0542 out-of-fold loss (``phase7b.SEARCH_AXIS_VERDICTS``).
#:
#: **THE FIFTH ARM, AND THE AMENDMENT'S ARITHMETIC DID NOT SURVIVE
#: MEASUREMENT.** The amendment registers ``patch32`` at 512 as "a 16x16
#: grid against a pretrained 14x14 -- a much milder interpolation" than
#: patch16 at 512's. Measured (timm, 2026-08-14): ``vit_base_patch32_224``'s
#: pretrained grid is **7x7** -- the amendment's own arms table says 49
#: tokens, contradicting its rationale in the same section. So::
#:
#:     patch32 @ 512:  7x7  -> 16x16   ratio 2.286
#:     patch16 @ 512: 14x14 -> 32x32   ratio 2.286
#:
#: **The two interpolations have IDENTICAL relative stretch.** "Much milder"
#: is false as stated. What actually differs is everything absolute: target
#: grid 16x16 against 32x32, 256 tokens against 1,024, and a 32-pixel patch
#: footprint against 16.
#:
#: **That makes the fifth arm sharper than the amendment thought, and the
#: readings are committed now because after the 224 results exist this
#: becomes post-hoc:**
#:
#: * **cliffs like patch16@512 did** -- the cliff tracks RELATIVE grid
#:   stretch, which both share at 2.286. The interpolation-mechanism account
#:   is supported in its ratio form.
#: * **does not cliff** -- relative stretch is NOT the mechanism; what
#:   matters is absolute token geometry or patch footprint. The
#:   interpolation account survives only in a revised form, stated as a
#:   revision.
#: * Either way the arm discriminates between mechanisms. **Neither outcome
#:   licenses tuning**, and the arm may not be re-read after the fact as
#:   "the configuration that beat the cliff" -- that framing died with the
#:   14x14 claim.
PHASE_7D_PATCH_AXIS_REGISTERED = {
    "registered": "2026-08-14, before any 7D snapshot, extraction or arm",
    "not_road_b": (
        "Road B varied INPUT SIZE at fixed patch 16, interpolating the "
        "position grid (0.2319 -> 0.0857). Phase 7D varies PATCH SIZE at "
        "fixed 224, so arms 1-4 run on native grids and nothing is "
        "interpolated"
    ),
    "arms": {
        "vit_b32_224": {"grid": "7x7", "tokens": 49},
        "vit_b16_224": {
            "grid": "14x14", "tokens": 196,
            "control": "EXISTING -- p7_d1_vit_b16_imagenet_g1 at 0.2520",
        },
        "vit_b8_224": {"grid": "28x28", "tokens": 784},
        "concat_8_16_32": {
            "prior_against": (
                "phase7b: +0.0221 inner-val became -0.0542 out-of-fold, "
                "measured on this cohort"
            ),
        },
        "vit_b32_512": {
            "grid": "16x16 interpolated from 7x7",
            "role": "the mechanism-discriminating arm; see the readings",
        },
    },
    "seeds_per_arm": 5,
    "amendment_arithmetic_corrected": {
        "claimed": "patch32@512 is a 16x16 grid against a pretrained 14x14",
        "measured": (
            "vit_base_patch32_224's pretrained grid is 7x7 (49 tokens -- "
            "the amendment's own table), so patch32@512 interpolates "
            "7x7 -> 16x16 at ratio 2.286, IDENTICAL to patch16@512's "
            "14x14 -> 32x32"
        ),
        "consequence": (
            "'much milder interpolation' is false as stated; the two "
            "configurations differ in ABSOLUTE grid, token count and patch "
            "footprint, not in relative stretch"
        ),
        "found": "2026-08-14, by measuring timm's grids before registering",
    },
    "fifth_arm_readings_committed": {
        "cliffs_like_patch16_at_512": (
            "the cliff tracks RELATIVE grid stretch, shared at 2.286; the "
            "interpolation mechanism is supported in its ratio form"
        ),
        "does_not_cliff": (
            "relative stretch is not the mechanism -- absolute token "
            "geometry or patch footprint is. The interpolation account "
            "survives only as a stated revision"
        ),
        "either_way": (
            "the arm discriminates mechanisms. Neither outcome licenses "
            "tuning, and the arm may not be re-read as 'the configuration "
            "that beat the cliff' -- that framing died with the 14x14 claim"
        ),
    },
    "why_registered_now": (
        "once the 224 results exist, adding the 512 arm becomes post-hoc "
        "and no care afterwards recovers it (amendment §2's own warning)"
    ),
}


#: **[BUILT 2026-08-15] Phase 7D's configs exist, and the registration
#: PREDATES the build** -- which is what the priority claim needed and what
#: the old no-artifacts test protected until now.
#:
#: **What existed before this build**: the registration
#: (``PHASE_7D_PATCH_AXIS_REGISTERED``, 2026-08-14), arm 2's result
#: (``p7_d1_vit_b16_imagenet_g1`` at 0.2520 -- referenced, no new work),
#: and nothing else: no 7D config, snapshot, extraction or embedding set.
#:
#: **Arm 6 is NEW SCOPE, not part of the registration**, and the
#: attribution is the maintainer's: arm 6 is the direct response to the paper
#: shared at supervision on 2026-08-13 (Fan et al., Multiscale Vision Transformers,
#: ICCV 2021); arms 1-5 are the project's decomposition of the "multi-scale,
#: not 16x16" request into patch granularity at fixed input, registered
#: before the paper was shared.
#:
#: **THE SUBSTITUTION, measured against the pinned timm 1.0.7 installed
#: standalone**: the paper is MViTv1 and **v1 does not exist in the pin at
#: all** -- the registry's only mvit entries are ``mvitv2_*`` (``samvit_*``
#: is SAM, unrelated). So the arm runs **MViTv2-Base** (``mvitv2_base``,
#: tag ``fb_in1k``), and the record names what v2 adds over the paper's
#: architecture: decomposed relative position embeddings and residual
#: pooling connections. Capacity: 50,703,744 parameters against ViT-B's
#: ~86M and Swin-B's ~88M -- the closest variant with in1k weights
#: (``mvitv2_large`` is 218M, the wrong class).
#:
#: **Measured structure, so the head config is deliberate rather than
#: defaulted**: ``num_features`` 768 (same width as the ViTs);
#: ``forward_features`` -> (B, 49, 768) -- the final stage's 7x7 -- and
#: **NO cls token: ``global_pool='avg'``**, mean-pool where every ViT arm is
#: CLS-token. Its normalization statistics are ImageNet's, not the ViTs'
#: 0.5/0.5 -- the values live in the model's own ``pretrained_cfg``, which
#: ``normalization_for`` reads (never a constant repeated here).
#: The standard extraction path (``forward_features`` +
#: ``forward_head(pre_logits=True)``) covers it unchanged, measured.
#:
#: **One recipe wrinkle on the patch axis, recorded rather than absorbed**:
#: the pinned default tags are ``augreg_in21k_ft_in1k`` for patch32 and
#: ``augreg2_in21k_ft_in1k`` for patch16 and patch8 -- same pretraining
#: corpus (in21k -> in1k), one recipe revision apart. The snapshot task
#: records each artifact's exact tag in MANIFEST.json.
#:
#: **ARM 5 IS HELD, NOT BUILT.** The registration does not specify square
#: or nonsquare staging for the 512 extraction, and the choice is the
#: maintainer's. The decision-relevant fact, measured from the shipped
#: configs: the comparator -- Road B's patch16@512 at 0.0857
#: (``roadb_p7_arm_vit_b16_imagenet_512``) -- extracted from
#: ``roadb_512_square_g1_v1``, SQUARE G1. A matched one-factor comparison
#: therefore points at the same artifact; a nonsquare choice would vary
#: staging and patch size together.
PHASE_7D_BUILT = {
    "built": "2026-08-15",
    "registration_predates_build": True,
    "existed_before": (
        "PHASE_7D_PATCH_AXIS_REGISTERED (2026-08-14) and arm 2's result "
        "(p7_d1_vit_b16_imagenet_g1, 0.2520); no 7D config, snapshot, "
        "extraction or set"
    ),
    "attribution": (
        "arm 6 is the direct response to the supervision shared paper "
        "(Fan et al., Multiscale Vision Transformers, ICCV 2021, shared "
        "2026-08-13); arms 1-5 are the project's decomposition of the "
        "'multi-scale, not 16x16' request into patch granularity at fixed "
        "input, registered before the paper was shared"
    ),
    "arm_6": {
        "requested": "MViT, ImageNet-pretrained, 224, G1, whole image",
        "substitution": {
            "paper": "MViTv1",
            "ran": "mvitv2_base (tag fb_in1k), timm 1.0.7",
            "why": (
                "v1 does not exist in the pinned timm; the only mvit "
                "entries are mvitv2_*"
            ),
            "v2_adds": (
                "decomposed relative position embeddings and residual "
                "pooling connections"
            ),
        },
        "capacity": {"parameters": 50_703_744, "vit_b": "~86M", "swin_b": "~88M"},
        "embedding_dim": 768,
        "pooling": "mean over the final stage's 49 tokens -- NO cls token",
        "normalization": (
            "ImageNet statistics, read from its own pretrained_cfg by "
            "normalization_for -- not the ViTs' 0.5/0.5"
        ),
        "framing": {
            "why": (
                "Swin-B already gives this cohort a measured data point on "
                "hierarchical multi-scale transformers; MViT tests whether "
                "that result is Swin-specific or a property of the "
                "hierarchical class"
            ),
            "reading_if_near_swin": "class property",
            "reading_if_different": (
                "the hierarchy alone does not determine the outcome and the "
                "mechanisms differ (pooling attention vs shifted windows)"
            ),
            "neither_is_mvit_should_win": (
                "the paper's ImageNet margins are from full training on "
                "1.28M images; the frozen-probe regime at n=237 is a "
                "different question"
            ),
        },
    },
    "recipe_wrinkle": (
        "default tags: patch32 augreg_in21k_ft_in1k, patch16/patch8 "
        "augreg2_in21k_ft_in1k -- same corpus, one recipe revision apart; "
        "each snapshot records its exact tag"
    ),
    "arm_5_held": {
        "why": (
            "the registration does not specify square or nonsquare staging "
            "for the 512 extraction; the maintainer decides before launch"
        ),
        "decision_relevant_fact": (
            "the comparator roadb_p7_arm_vit_b16_imagenet_512 (0.0857) "
            "extracted from roadb_512_square_g1_v1 -- SQUARE G1. Matched "
            "one-factor comparison points at the same artifact"
        ),
        #: **[DECIDED 2026-08-15, maintainer.]** The hold above is kept as the
        #: state it recorded; the decision sits beside it.
        "decided": {
            "date": "2026-08-15",
            "staging": "SQUARE -- roadb_512_square_g1_v1",
            "why": (
                "one-factor match against the patch16@512 comparator, which "
                "extracted from that same artifact"
            ),
            "configs": (
                "p7d_extract_vit_b32_512.yaml, p7d_arm_vit_b32_512.yaml"
            ),
            "snapshot_shared": (
                "one vit_b32 snapshot serves the 224 and 512 extractions -- "
                "the weights do not depend on the input size; "
                "dynamic_img_size interpolates the position grid 7x7 -> "
                "16x16 at forward time, which is the thing being measured"
            ),
        },
    },
    "extraction_memory": {
        "vit_b8": (
            "785 tokens; a float32 attention matrix is 29.6 MB per image "
            "per block, 15.9x vit_b16's 1.9 MB -- at batch 32 that is "
            "~0.95 GB transient per block for attention alone. Extraction "
            "batch_size is set to 8 (~16x fewer tokens^2 per batch than "
            "b16 at 32)"
        ),
        "mvitv2_b": (
            "pooling attention SHRINKS the token set stage by stage "
            "(56x56 -> 7x7), so peak memory is below ViT-B/16's; batch 32 "
            "stands"
        ),
    },
}


#: **[MEASURED 2026-08-15, seed-variance means pasted by the maintainer] The
#: five 7D arms, placed against their pre-registered readings -- each reading
#: was committed before these numbers existed, which is what lets them be
#: APPLIED here rather than composed.**
#:
#: ::
#:
#:     arm 1  vit_b32       0.2519  sd 0.0313   (49 tokens, native 7x7)
#:     arm 2  vit_b16       0.2520              (control, existing result)
#:     arm 3  vit_b8        0.0432  sd 0.0340   (784 tokens, native 28x28)
#:     arm 4  concat        0.2594  sd 0.0208
#:     arm 5  vit_b32@512   0.2280  sd 0.0217   (256 tokens, 7x7 -> 16x16)
#:     arm 6  mvitv2_b      0.1844  sd 0.0410
#:
#: **(1) ARM 5'S DISCRIMINATION FIRED, for token-count/receptive-field.**
#: 0.2280 against the comparator's 0.0857 under the IDENTICAL 2.286x
#: position-grid stretch: relative stretch is not the mechanism, exactly the
#: ``does_not_cliff`` reading as registered. **And arm 3 corroborates from
#: the other side**: vit_b8 collapses to 0.0432 with ZERO interpolation --
#: its 28x28 grid is native -- so the cliff's variable is not interpolation
#: at all.
#:
#: **The two registered candidates are ONE variable** (maintainer, recorded
#: with the result): at a fixed input size, patch size p determines the
#: token count (side/p)^2 AND each patch's receptive fraction (p/side)^2
#: deterministically -- they cannot be varied separately by any arm of this
#: design. The implicated variable is GRID DENSITY: dense grids collapse
#: (784 tokens 0.0432; Road B's 1024-token 512 arm 0.0857; its 2304-token
#: 768 arm 0.0898), coarse and moderate grids do not (49 at 0.2519, 196 at
#: 0.2520, 256 at 0.2280). A refinement of the registered reading's wording,
#: not a new registration.
#:
#: **(2) ARM 1 IS A NULL against the control**: 0.2519 vs 0.2520 -- 49
#: tokens carry what 196 carry, on this cohort, descriptively.
#:
#: **(3) ARM 4 is +0.0074 against the control -- the DIRECTION is against
#: the 7B prior** (which measured concat at -0.0542 out-of-fold), and the
#: size is far inside every threshold. UNCLAIMABLE PENDING the paired BCa;
#: not to be quoted as the prior being overturned.
#:
#: **(4) ARM 6, against the registered Swin-class reading.** The matched
#: cell is Swin-B imagenet G1 (``STAGE_D1_AT_G1``): **0.1076** (sd 0.0217).
#: MViTv2-B at **0.1844** sits +0.0768 above it -- outside both arms'
#: bands. **The reading that applies is reading_if_different**: the
#: hierarchy alone does not determine the outcome and the mechanisms differ
#: (pooling attention vs shifted windows). Descriptive pending the BCa, as
#: every number here is.
PHASE_7D_OBSERVED = {
    "measured": "2026-08-15, seed-variance means pasted by the maintainer",
    "arms": {
        "vit_b32": {"mean": 0.2519, "sd": 0.0313},
        "vit_b16": {"mean": 0.2520, "control": "existing result, no new run"},
        "vit_b8": {"mean": 0.0432, "sd": 0.0340},
        "concat": {"mean": 0.2594, "sd": 0.0208},
        "vit_b32_512": {"mean": 0.2280, "sd": 0.0217},
        "mvitv2_b": {"mean": 0.1844, "sd": 0.0410},
    },
    "arm_5_discrimination": {
        "fired": "does_not_cliff",
        "figures": (
            "0.2280 against the comparator's 0.0857, identical 2.286x "
            "stretch on both sides"
        ),
        "reading_applied": (
            "relative stretch is not the mechanism -- as registered, "
            "verbatim, before the number existed"
        ),
        "arm_3_corroborates": (
            "vit_b8 collapses to 0.0432 with ZERO interpolation; the "
            "cliff's variable is not interpolation at all"
        ),
    },
    "grid_density": {
        "recorded": "2026-08-15, with the result (maintainer)",
        "link": (
            "at fixed input size, patch size determines token count AND "
            "per-patch receptive fraction deterministically -- not two "
            "separable candidates"
        ),
        "implicated": "grid density",
        "ordering": (
            "49 tokens 0.2519, 196 tokens 0.2520, 256 tokens 0.2280 | "
            "784 tokens 0.0432, 1024 tokens 0.0857 (Road B 512), 2304 "
            "tokens 0.0898 (Road B 768)"
        ),
        "status": "a refinement of the registered wording, not a new registration",
    },
    "arm_1": {
        "verdict": "NULL against the control",
        "delta": -0.0001,
        "descriptive": "49 tokens carry what 196 carry, on this cohort",
    },
    "arm_4": {
        "delta": 0.0074,
        "against_prior": (
            "direction OPPOSES the 7B prior's -0.0542; size far inside "
            "every threshold"
        ),
        "status": (
            "UNCLAIMABLE PENDING the paired BCa -- not to be quoted as the "
            "prior overturned"
        ),
    },
    "arm_6": {
        "mvitv2_b": 0.1844,
        "swin_b_matched_cell": {
            "value": 0.1076, "sd": 0.0217,
            "which": "imagenet G1 frozen probe (STAGE_D1_AT_G1) -- the "
                     "one-factor comparator",
        },
        "difference": 0.0768,
        "reading_applied": "reading_if_different",
        "sentence": (
            "the hierarchy alone does not determine the outcome and the "
            "mechanisms differ (pooling attention vs shifted windows)"
        ),
        "descriptive_pending_bca": True,
        #: **[DOWNGRADED 2026-08-15, after the paired BCa.]** The reading
        #: above fired on the descriptive +0.0768 and the BCa withdrew it:
        #: 0 of 5 intervals exclude zero BOTH ways (margins 1.77x against
        #: b16, 1.89x against swin). The fields above are kept as the state
        #: they recorded; this is what stands now.
        "downgraded": {
            "date": "2026-08-15",
            "bca": {
                "vs_vit_b16": {"delta": -0.0676, "n_excluding_zero": "0/5",
                               "margin": 1.77},
                "vs_swin_b": {"delta": 0.0768, "n_excluding_zero": "0/5",
                              "margin": 1.89},
            },
            "stands": (
                "MViTv2 is distinguishable from NEITHER ViT-B/16 nor "
                "Swin-B on this cohort. The Swin-class question closes "
                "UNRESOLVED both ways"
            ),
            "forbidden": (
                "'the hierarchy alone does not determine the outcome' as a "
                "CLAIM -- it survives only as the unresolved descriptive "
                "note it always was"
            ),
        },
    },
    "bca_pending": (
        "six contrasts enumerated in paired_claim_pairs('p7d'); nothing "
        "above is claimable until condition 1 runs"
    ),
}


#: **[CLOSED 2026-08-15, verdicts pasted from p7d-paired's metrics.json]
#: PHASE 7D IS CLOSED: two claims, both about grid density; four
#: withdrawals; the headline unchanged.**
#:
#: ::
#:
#:     patch8_vs_b16            d -0.2089   5/5   6.42x   CLAIMABLE
#:     b32_512_vs_patch16_512   d +0.1386   5/5   5.32x   CLAIMABLE
#:     patch32_vs_b16           d -0.0001   0/5           withdrawn
#:     concat_vs_b16            d +0.0073   0/5           withdrawn
#:     mvitv2_vs_b16            d -0.0676   0/5   1.77x   unresolved-withdrawn
#:     mvitv2_vs_swin           d +0.0768   0/5   1.89x   unresolved-withdrawn
#:
#: **(1) The verdicts landed exactly where the pre-BCa record put them,
#: and the date order is git-provable.** ``PHASE_7D_OBSERVED`` -- committed
#: before the paired run existed (the run dirs sit at a LATER commit's
#: sha) -- marked the grid-density contrasts as the claim candidates, arm
#: 1 NULL, arm 4 UNCLAIMABLE PENDING, arm 6 descriptive-pending. Every
#: verdict resolved on the side that record anticipated: the two
#: grid-density contrasts claimed, the null and the concat withdrew, and
#: arm 6 stayed open and then closed unresolved.
#:
#: **(2) THE GRID-DENSITY FINDING IS NOW CLAIMED, on two paired
#: contrasts**: densifying the grid destroys the frozen-probe correlation
#: (784 against 196 tokens, -0.2089 at 6.42x, 5/5) and coarsening at 512
#: recovers it (256 against 1024 tokens, +0.1386 at 5.32x, 5/5) -- both
#: under conditions 1 AND 2. Arm 3's collapse with ZERO interpolation and
#: arm 5's survival under the IDENTICAL 2.286x stretch jointly retire the
#: interpolation mechanism for the Road B cliff. The deterministic
#: linkage travels with the claim: at fixed input size, token count and
#: per-patch receptive fraction are one variable, not two separable
#: candidates -- the variable is GRID DENSITY.
#:
#: **(3) Arm 6 closes UNRESOLVED both ways** and the earlier
#: ``reading_if_different`` is downgraded in place, dated
#: (``PHASE_7D_OBSERVED.arm_6.downgraded``): MViTv2 is distinguishable
#: from neither comparator on this cohort, and "the hierarchy alone does
#: not determine the outcome" is FORBIDDEN as a claim.
#:
#: **(4) Concat closes withdrawn, and the 7B prior is NEITHER confirmed
#: nor overturned.** The two measurements disagree in sign (+0.0073 here
#: against 7B's -0.0542 out-of-fold) and both are individually
#: unclaimable -- what stands is that this cohort cannot resolve the
#: concat question in either direction.
#:
#: **(5) The margin table moved** (``roadb.CONDITION_1_MARGIN_STRUCTURE``,
#: updated in place, dated): 5.32x passed -- the first point in the
#: previously EMPTY 4.7x-6.91x interval -- so the pass floor is now
#: <= 5.32x; 6.42x passed with it; 1.77x and 1.89x failed, consistent
#: with the sub-2.55x record.
#:
#: **(6) The headline is unchanged**: nothing in 7D beats 0.2520. The
#: closest thing to news below the claims is DESCRIPTIVE: vit_b32 matches
#: the control at -0.0001 with a QUARTER of the tokens (49 against 196).
PHASE_7D_CLOSING = {
    "closed": "2026-08-15, verdicts from p7d-paired's metrics.json",
    "verdicts": {
        "p7d__patch8_vs_b16": {
            "delta": -0.2089, "n_excluding_zero": "5/5", "margin": 6.42,
            "verdict": "CLAIMABLE",
        },
        "p7d__b32_512_vs_patch16_512": {
            "delta": 0.1386, "n_excluding_zero": "5/5", "margin": 5.32,
            "verdict": "CLAIMABLE",
        },
        "p7d__patch32_vs_b16": {
            "delta": -0.0001, "n_excluding_zero": "0/5",
            "verdict": "WITHDRAWN",
        },
        "p7d__concat_vs_b16": {
            "delta": 0.0073, "n_excluding_zero": "0/5",
            "verdict": "WITHDRAWN",
        },
        "p7d__mvitv2_vs_b16": {
            "delta": -0.0676, "n_excluding_zero": "0/5", "margin": 1.77,
            "verdict": "UNRESOLVED-WITHDRAWN",
        },
        "p7d__mvitv2_vs_swin": {
            "delta": 0.0768, "n_excluding_zero": "0/5", "margin": 1.89,
            "verdict": "UNRESOLVED-WITHDRAWN",
        },
    },
    "expectation_resolution": {
        "resolved_as_registered": True,
        "date_order": (
            "PHASE_7D_OBSERVED was committed before the paired run existed "
            "-- the p7d run dirs carry a later commit's sha -- and every "
            "verdict landed on the side that record anticipated: grid-"
            "density contrasts claimed, null and concat withdrew, arm 6 "
            "open then unresolved"
        ),
    },
    #: **This record is IN the claim audit, deliberately** -- it carries a
    #: ``claimable`` field so ``CLAIMS_REST_ON_HALF_THE_CRITERION``'s sweep
    #: finds it, and a ``condition_1`` dict so it counts as EVIDENCED: the
    #: first claims in the Phase 7 family born with the full criterion
    #: rather than resolved into it later.
    "claimable": True,
    "condition_1": {
        "p7d__patch8_vs_b16": "5/5, all excluding zero",
        "p7d__b32_512_vs_patch16_512": "5/5, all excluding zero",
        "p7d__patch32_vs_b16": "0/5",
        "p7d__concat_vs_b16": "0/5",
        "p7d__mvitv2_vs_b16": "0/5",
        "p7d__mvitv2_vs_swin": "0/5",
        "source": "p7d-paired metrics.json, n_boot 10000, pooled OOF",
    },
    "claims": {
        "finding": "GRID DENSITY, claimed on two paired contrasts",
        "densifying_destroys": (
            "784 against 196 tokens: -0.2089 at 6.42x, 5/5, conditions 1+2"
        ),
        "coarsening_recovers": (
            "256 against 1024 tokens at 512 input: +0.1386 at 5.32x, 5/5, "
            "conditions 1+2"
        ),
        "interpolation_mechanism_retired": (
            "arm 3 collapsed with ZERO interpolation and arm 5 survived "
            "under the IDENTICAL 2.286x stretch -- jointly, the Road B "
            "cliff's mechanism was never interpolation"
        ),
        "one_variable": (
            "token count and per-patch receptive fraction are "
            "deterministically linked at fixed input size; the variable is "
            "grid density, not two separable candidates"
        ),
    },
    "arm_6_closes": (
        "UNRESOLVED both ways; reading_if_different downgraded in place, "
        "dated (PHASE_7D_OBSERVED.arm_6.downgraded); 'hierarchy alone does "
        "not determine the outcome' forbidden as a claim"
    ),
    "concat_closes": (
        "WITHDRAWN. The 7B prior (-0.0542 OOF) is neither confirmed nor "
        "overturned: the two measurements disagree in SIGN and both are "
        "individually unclaimable -- the cohort cannot resolve the concat "
        "question in either direction"
    ),
    "margin_table": (
        "updated in roadb.CONDITION_1_MARGIN_STRUCTURE, dated: 6.42x and "
        "5.32x pass (the 4.7x-6.91x interval had been empty; the pass "
        "floor is now <= 5.32x); 1.77x and 1.89x fail, consistent with the "
        "sub-2.55x record"
    ),
    "headline": (
        "unchanged -- nothing in 7D beats 0.2520. Descriptive: vit_b32 "
        "matches the control at -0.0001 with a quarter of the tokens"
    ),
    "still_untouched": ("8b", "8c", "t-SNE"),
}


#: The five 7D run stems plus the three arms they pair against. The
#: cross-phase comparator is legitimate because pairing needs only per-seed
#: OOF vectors on a common truth: the Road B arm scored the SAME manifest
#: (fb5177..., same fold column), the loader cross-checks the truth vector
#: across every file and the task re-checks it across seed groups to 1e-9,
#: and its ten seeds are Road A's pool, so the five shared seeds pair
#: exactly -- the TRADE_OFF_PAIR precedent, which paired a five-seed arm
#: with a ten-seed arm on the shared five.
P7D_PAIRED_CONTRASTS = (
    ("p7d__patch8_vs_b16", "p7d_grid_density",
     "p7_d1_vit_b16_imagenet_g1", "p7d_arm_vit_b8", "patch_size"),
    ("p7d__patch32_vs_b16", "p7d_grid_density",
     "p7_d1_vit_b16_imagenet_g1", "p7d_arm_vit_b32", "patch_size"),
    ("p7d__concat_vs_b16", "p7d_multiscale_concat",
     "p7_d1_vit_b16_imagenet_g1", "p7d_arm_concat_multiscale",
     "representation"),
    ("p7d__b32_512_vs_patch16_512", "p7d_grid_density_at_512",
     "roadb_p7_arm_vit_b16_imagenet_512", "p7d_arm_vit_b32_512",
     "patch_size_at_512"),
    ("p7d__mvitv2_vs_b16", "p7d_architecture",
     "p7_d1_vit_b16_imagenet_g1", "p7d_arm_mvitv2_b", "architecture"),
    ("p7d__mvitv2_vs_swin", "p7d_hierarchical_class",
     "p7_d1_swin_b_imagenet_g1", "p7d_arm_mvitv2_b",
     "hierarchical_architecture"),
)


class LadderError(RuntimeError):
    """The arm list cannot be derived."""


def embedding_set_name(arm: dict) -> str:
    """The artifact directory name of the set this arm consumes.

    **One implementation, because there were four** -- ``_arm``, the
    generator, ``resolved_inputs`` and the lattice test each rebuilt the same
    ``backbone__init__geometry[__scheme]`` string. They agreed, which is the
    state that precedes disagreeing: the scheme clause alone carries two
    conditions (graph backbones only, and never on an imagenet init, which has
    no pretraining to have a scheme), and a copy updated in three places out
    of four points an arm at a set that exists and is not its own.
    """
    kind = arm.get("backbone_kind") or BACKBONES[arm["backbone"]]["kind"]
    return embedding_plan.set_name({
        "backbone": arm["backbone"],
        "init": arm["init"],
        "geometry": arm["geometry"],
        "pretrain_scheme": (
            arm["region_scheme"]
            if kind == "graph" and arm["init"] != "imagenet"
            else None
        ),
    })


def embeddings_version_for(set_name: str) -> str:
    """Which artifact version holds one set.

    **Derived from what each extraction produced, not from the arm's stage.**
    ``embeddings_v1`` holds exactly ``embedding_plan.required_sets()`` -- that
    equality is asserted against the shipped extraction config
    (``tests/test_extract.py``), so membership here follows the same source
    rather than restating it. The control version holds its one named set, and
    everything left over is the Stage D1 batch.

    Written this way so a set moving between batches moves by itself: the
    previous rule named Stage C's backbones and inits inline, and went wrong
    the moment a G1 set appeared that was neither Stage C's nor C0's.
    """
    if set_name in _plan_set_names():
        return DEFAULT_EMBEDDINGS_VERSION
    if set_name == G1_CONTROL_SET:
        return G1_CONTROL_EMBEDDINGS_VERSION
    return G1_LADDER_EMBEDDINGS_VERSION


def _plan_set_names() -> frozenset:
    return frozenset(
        embedding_plan.set_name(entry) for entry in embedding_plan.required_sets()
    )


def g1_ladder_sets() -> list[dict]:
    """The sets ``configs/p7_extract_g1_ladder.yaml`` must produce, derived.

    Nine, not twelve: Stage D1's twelve cells include three whose runs already
    exist -- C0's ``vit_b16 scut_original g1`` and Stage C's two masked-G1
    transformer arms -- so their sets are already extracted and declared. This
    walks the DISTINCT runs, which is where that reuse is already resolved.

    Ordered by set name so the shipped config and the derivation can be
    compared element by element rather than as sets.
    """
    entries: dict[str, dict] = {}
    for arm in distinct_runs():
        if arm["embeddings_version"] != G1_LADDER_EMBEDDINGS_VERSION:
            continue
        entries[embedding_set_name(arm)] = {
            "backbone": arm["backbone"],
            "init": arm["init"],
            "geometry": arm["geometry"],
            "pretrain_scheme": (
                arm["region_scheme"]
                if arm["backbone_kind"] == "graph" and arm["init"] != "imagenet"
                else None
            ),
        }
    return [entries[name] for name in sorted(entries)]


def _arm(
    stage: str,
    backbone: str,
    init: str,
    geometry: str = LADDER_GEOMETRY,
    scheme: str | None = None,
    label: str = "mean",
) -> dict:
    if backbone not in BACKBONES:
        raise LadderError(f"unknown backbone {backbone!r}")
    if init not in INITS:
        raise LadderError(f"unknown init {init!r}")
    kind = BACKBONES[backbone]["kind"]
    arm = {
        "stage": stage,
        "backbone": backbone,
        "backbone_kind": kind,
        "init": init,
        "geometry": geometry,
        # Transformers have no region-scheme axis; graph arms carry one.
        "region_scheme": (scheme or CANONICAL_SCHEME) if kind == "graph" else None,
        "label": label,
        "seeds": SEEDS_BY_KIND[kind],
        "seed_list": seeds_for(kind),
        # The trained-graph-layer regime for graph arms; a frozen head for
        # transformers. Stage E depends on this being the regime, not a probe.
        "trainable": "graph_layers" if kind == "graph" else "head",
        "task": "train_graph_cv" if kind == "graph" else "train_cv",
        #: **[MEASURED] AG-Net alone declares this, and declares it FALSE.**
        #: The deterministic roi_align substitution cannot be satisfied in the
        #: pinned image, so the flag would raise mid-run; false bounds the
        #: arm's same-seed reproducibility at ~2.3e-05, which every comparison
        #: against it carries. Already the settled decision for AG-Net
        #: pretraining, where all thirty runs ran deterministic: false.
        #: Every other backbone is bitwise under the flag and omits the field.
        "deterministic": False if backbone == "agnet" else None,
    }
    # Which artifact version holds this arm's set. A published artifact cannot
    # gain a set (PLAN §2.6), so the G1 sets live in the version of the
    # extraction batch that produced them -- see embeddings_version_for, which
    # derives that rather than restating it.
    arm["embeddings_version"] = embeddings_version_for(embedding_set_name(arm))
    return arm


def identity(arm: dict) -> tuple:
    """What makes two arms the same run. Seeds are excluded deliberately: an
    arm at five seeds and the same arm at ten is one arm measured twice, not
    two arms, and the reuse map has to see that."""
    return (
        arm["backbone"], arm["init"], arm["geometry"],
        arm["region_scheme"], arm["label"], arm["trainable"],
    )


def arms() -> list[dict]:
    """Every cell the ladder reports, in stage order, with reuse resolved.

    An arm whose identity already appeared carries ``reuses`` naming the
    earlier one and is not a separate run.
    """
    cells: list[dict] = []

    # Stage D -- the init ladder, the project's central question.
    for backbone in LADDER_BACKBONES:
        for init in INITS:
            cells.append(_arm("D", backbone, init))

    # Stage C -- geometry, both transformers, both geometries. The G2 cells
    # are Stage D arms.
    for backbone in GEOMETRY_BACKBONES:
        for geometry in ("g1", LADDER_GEOMETRY):
            cells.append(_arm("C", backbone, GEOMETRY_INIT, geometry=geometry))

    # Stage E -- region scheme, trained graph layers. `native` is a Stage D arm.
    for scheme in SCHEMES:
        cells.append(
            _arm("E", SCHEME_BACKBONE, GEOMETRY_INIT, scheme=scheme)
        )

    # Stage G -- label formulation. `mean` is a Stage D arm.
    for label in LABELS:
        cells.append(_arm("G", LABEL_BACKBONE, LABEL_INIT, label=label))

    # Stage E0 -- scheme with the CHECKPOINT HELD CONSTANT. ImageNet has no
    # checkpoint, so these four differ in placement alone. `native` is a
    # Stage D arm.
    for scheme in SCHEMES:
        cells.append(
            _arm("E0", SCHEME_BACKBONE, SCHEME_CONTROL_INIT, scheme=scheme)
        )

    # Stage C0 -- geometry with the CHECKPOINT HELD CONSTANT, on a pretrained
    # init. scut_original is not geometry-bound, so one checkpoint serves
    # both cells. The G2 cell is a Stage D arm.
    for geometry in ("g1", LADDER_GEOMETRY):
        cells.append(
            _arm(
                "C0", GEOMETRY_CONTROL_BACKBONE, GEOMETRY_CONTROL_INIT,
                geometry=geometry,
            )
        )

    # Stage D1 -- the init ladder AT G1. Not a repair of Stage D, whose G2
    # numbers are valid AS G2 measurements: the operating point was chosen on
    # SCUT evidence since measured not to transfer, and the one clean test
    # (Q1) shows the central question's answer moves with geometry. Both
    # geometries are reported, so the interaction becomes a result.
    for backbone in LADDER_BACKBONES:
        for init in INITS:
            cells.append(_arm("D1", backbone, init, geometry="g1"))

    # Stage G1 -- label formulation at G1. AFTER Stage D1 deliberately: the
    # `mean` cell is the D1 arm `p7_d1_vit_b16_imagenet_g1`, and the reuse map
    # names the FIRST occurrence as canonical. Ordered the other way, an
    # init-ladder arm would be recorded as reusing a label arm, which inverts
    # which stage owns the run.
    for label in LABELS:
        cells.append(
            _arm(
                "G1", LABEL_BACKBONE, LABEL_CONTROL_INIT,
                geometry=LABEL_CONTROL_GEOMETRY, label=label,
            )
        )

    # Stage G0 -- the same triple at G2, holding init at imagenet, so the
    # G/G1 disagreement becomes attributable. Its `mean` cell is the Stage D
    # arm p7_d_vit_b16_imagenet_g2, which is why this follows Stage D in the
    # list for the same reason Stage G1 follows Stage D1.
    for label in LABELS:
        cells.append(
            _arm(
                "G0", LABEL_BACKBONE, LABEL_CONTROL_INIT,
                geometry=LABEL_GEOMETRY_CONTROL_GEOMETRY, label=label,
            )
        )

    seen: dict[tuple, str] = {}
    for index, cell in enumerate(cells):
        key = identity(cell)
        cell["name"] = arm_name(cell)
        if key in seen:
            cell["reuses"] = seen[key]
        else:
            seen[key] = cell["name"]
    return cells


def arm_name(arm: dict) -> str:
    """The config stem. Says everything that distinguishes the arm."""
    parts = [f"p7_{arm['stage'].lower()}", arm["backbone"], arm["init"], arm["geometry"]]
    if arm["region_scheme"]:
        parts.append(arm["region_scheme"])
    if arm["label"] != "mean":
        parts.append(arm["label"])
    return "_".join(parts)


def distinct_runs() -> list[dict]:
    """The arms the cluster actually executes -- reused cells excluded."""
    return [arm for arm in arms() if "reuses" not in arm]


def resolved_inputs(arm: dict) -> dict:
    """What the arm actually CONSUMES, as opposed to what it declares.

    **The declared fields and the resolved inputs are not the same thing, and
    the gap is where a two-factor comparison hides.** ``init: scut_masked`` is
    one value, but ``expected_variant`` maps it to ``masked_g1`` at G1 and
    ``masked_g2`` at G2 -- different checkpoints, with different measured
    pretraining quality (0.7893 against 0.8306). So two cells agreeing on
    every declared field can still differ in the weights they start from, and
    a check over the fields alone reports one factor while two moved.
    """
    from .embeddings import expected_variant

    variant = expected_variant(arm["init"], arm["geometry"])
    scheme = arm["region_scheme"] if arm["backbone_kind"] == "graph" else None
    return {
        "embedding_set": embedding_set_name(arm),
        # None for imagenet, which has no pretraining checkpoint at all.
        "checkpoint": (
            None if variant is None
            else f"{arm['backbone']}__{variant}" + (f"__{scheme}" if scheme else "")
        ),
    }


#: What each named factor ALREADY IMPLIES about the resolved inputs.
#:
#: Changing ``init`` obviously changes the weights -- that IS the factor, and
#: nobody reading "Q1 varies init" is surprised that the checkpoint moved. But
#: **changing ``geometry`` does not imply changing the weights**, and a reader
#: of "Stage C varies geometry" would not expect it to. It happens anyway,
#: because masked inits are geometry-bound. Same for ``region_scheme`` under
#: the consistency rule.
#:
#: So an "extra" factor is a resolved difference the named factor does not
#: entail -- which is exactly the set a reader would be surprised by, and
#: exactly the set that has to be declared rather than discovered.
ENTAILED = {
    "init": {"checkpoint"},
    "geometry": set(),
    "region_scheme": set(),
    "label": set(),
}

#: Why a comparison is allowed to vary something beyond its named factor.
#: Every entry in a comparison's ``also_varies`` must have one, so an extra
#: factor is a recorded design decision rather than an oversight.
LICENCES = {
    "geometry:checkpoint": (
        "masked inits are GEOMETRY-BOUND (embeddings.check_init_geometry), so "
        "the checkpoint is forced to match the geometry and cannot be held "
        "constant. This is therefore a matched-PIPELINE comparison by design "
        "-- PLAN's twelve-runs-not-eight reasoning -- and NOT a measurement of "
        "geometry alone. The pure-geometry number is the imagenet pair, which "
        "shares one set of weights across both cells."
    ),
    "region_scheme:checkpoint": (
        "the consistency rule requires the graph layers to see the same node "
        "structure in pretraining and cleft fine-tuning, so a scheme arm takes "
        "its own scheme's checkpoint. **THIS IS NOT LICENSED, and the licence "
        "it was granted on has been MEASURED FALSE.** The claim was that "
        "SCHEME_AXIS_AT_PRETRAINING's null made the four checkpoints "
        "equivalent; the null is real and the inference from it was not. "
        "Those four checkpoints, statistically equal on 2,199 SCUT faces, "
        "spread by 0.095 at cleft time -- of which placement contributes "
        "0.016 (graph_cleft.SCHEME_AXIS_AT_CLEFT) and transfer quality the "
        "rest (graph_cleft.PRETRAINING_DOES_NOT_PREDICT_TRANSFER). Stage E's "
        "ordering is therefore NOT attributable to placement. Stage E0 is the "
        "controlled comparison and is the one to read."
    ),
}


def comparisons() -> list[dict]:
    """Every delta the ladder claims, with the ONE field it varies.

    This is what the lattice test checks: each comparison must differ in
    exactly its declared field and agree on every other. A pair differing in
    two fields is not a measurement of either.
    """
    by_key = {identity(a): a for a in arms()}

    def find(**kwargs) -> dict:
        backbone = kwargs["backbone"]
        kind = BACKBONES[backbone]["kind"]
        key = (
            backbone, kwargs["init"], kwargs.get("geometry", LADDER_GEOMETRY),
            (kwargs.get("scheme") or CANONICAL_SCHEME) if kind == "graph" else None,
            kwargs.get("label", "mean"),
            "graph_layers" if kind == "graph" else "head",
        )
        if key not in by_key:
            raise LadderError(f"no arm for {kwargs}")
        return by_key[key]

    def annotate(entry: dict) -> dict:
        """Record every resolved difference, not just the named factor."""
        ra, rb = resolved_inputs(entry["a"]), resolved_inputs(entry["b"])
        entailed = ENTAILED[entry["varies"]]
        extra = sorted(
            k for k in ra
            if ra[k] != rb[k] and k != "embedding_set" and k not in entailed
        )
        # The embedding set differs whenever ANY factor does -- it encodes
        # backbone, init, geometry and scheme -- so it is never an extra
        # factor in its own right.
        entry["also_varies"] = extra
        entry["licences"] = {
            key: LICENCES[f"{entry['varies']}:{key}"] for key in extra
        }
        return entry

    out: list[dict] = []
    for backbone in LADDER_BACKBONES:
        out.append({
            "question": "Q1", "varies": "init", "stage": "D",
            "a": find(backbone=backbone, init="imagenet"),
            "b": find(backbone=backbone, init="scut_original"),
        })
        out.append({
            "question": "Q2", "varies": "init", "stage": "D",
            "a": find(backbone=backbone, init="scut_original"),
            "b": find(backbone=backbone, init="scut_masked"),
        })
    for backbone in GEOMETRY_BACKBONES:
        out.append({
            "question": "geometry", "varies": "geometry", "stage": "C",
            "a": find(backbone=backbone, init=GEOMETRY_INIT, geometry="g1"),
            "b": find(backbone=backbone, init=GEOMETRY_INIT, geometry="g2"),
        })
    for scheme in SCHEMES:
        if scheme == CANONICAL_SCHEME:
            continue
        out.append({
            "question": "scheme", "varies": "region_scheme", "stage": "E",
            "a": find(backbone=SCHEME_BACKBONE, init=GEOMETRY_INIT,
                      scheme=CANONICAL_SCHEME),
            "b": find(backbone=SCHEME_BACKBONE, init=GEOMETRY_INIT, scheme=scheme),
        })
    for scheme in SCHEMES:
        if scheme == CANONICAL_SCHEME:
            continue
        out.append({
            "question": "scheme_controlled", "varies": "region_scheme",
            "stage": "E0",
            "a": find(backbone=SCHEME_BACKBONE, init=SCHEME_CONTROL_INIT,
                      scheme=CANONICAL_SCHEME),
            "b": find(backbone=SCHEME_BACKBONE, init=SCHEME_CONTROL_INIT,
                      scheme=scheme),
        })
    out.append({
        "question": "geometry_controlled", "varies": "geometry", "stage": "C0",
        "a": find(backbone=GEOMETRY_CONTROL_BACKBONE,
                  init=GEOMETRY_CONTROL_INIT, geometry="g1"),
        "b": find(backbone=GEOMETRY_CONTROL_BACKBONE,
                  init=GEOMETRY_CONTROL_INIT, geometry=LADDER_GEOMETRY),
    })
    for label in LABELS:
        if label == "mean":
            continue
        out.append({
            "question": "label", "varies": "label", "stage": "G",
            "a": find(backbone=LABEL_BACKBONE, init=LABEL_INIT, label="mean"),
            "b": find(backbone=LABEL_BACKBONE, init=LABEL_INIT, label=label),
        })
    # The two imagenet triples, named in parallel so the pair is visible: the
    # disagreement they exist to resolve is between GEOMETRIES at one init.
    for stage, geometry in (
        ("G1", LABEL_CONTROL_GEOMETRY),
        ("G0", LABEL_GEOMETRY_CONTROL_GEOMETRY),
    ):
        for label in LABELS:
            if label == "mean":
                continue
            out.append({
                "question": f"label_at_imagenet_{geometry}",
                "varies": "label", "stage": stage,
                "a": find(
                    backbone=LABEL_BACKBONE, init=LABEL_CONTROL_INIT,
                    geometry=geometry, label="mean",
                ),
                "b": find(
                    backbone=LABEL_BACKBONE, init=LABEL_CONTROL_INIT,
                    geometry=geometry, label=label,
                ),
            })
    return [annotate(entry) for entry in out]


def summary() -> dict:
    """The arithmetic, visible rather than asserted."""
    cells = arms()
    runs = distinct_runs()
    by_stage: dict[str, dict] = {}
    for cell in cells:
        entry = by_stage.setdefault(cell["stage"], {"cells": 0, "runs": 0})
        entry["cells"] += 1
        entry["runs"] += 0 if "reuses" in cell else 1
    return {
        "cells": len(cells),
        "distinct_runs": len(runs),
        "by_stage": by_stage,
        "seeds_by_kind": dict(SEEDS_BY_KIND),
        "total_fits": sum(arm["seeds"] for arm in runs),
        "note": (
            "cells are what the summary table reports; distinct_runs is what "
            "the cluster executes. Stage C's G2 cells and Stage E's native "
            "cell and Stage G's mean cell are Stage D arms -- running them "
            "again would be two draws of one thing, not a comparison"
        ),
    }
