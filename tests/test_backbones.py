"""Phase 6 backbones: the registry, and the parameter cross-checks.

**No torch.** Everything here tests the specification and the routing, which is
what can be checked on a laptop. The built models are asserted against these same
specifications on the cluster, at construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cleft.models import agnet, factory, srgnn


def module_source(module) -> str:
    """Read a module's own text. Several checks here are about what the file
    SAYS -- provenance tags, recorded departures -- which is exactly the kind of
    claim that rots silently if only a human ever reads it."""
    return Path(module.__file__).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------


def test_the_four_phase_six_backbones_are_registered():
    """[UPDATED 2026-09-02, THIS PIN FIRED AS DESIGNED] Phase 25's ruling
    registered ``vit_b14_dinov2`` and ``vit_b16_dino``. The four Phase 6
    backbones it was written about are unchanged and still asserted; the
    set is now nine (phase25.THE_TWO_ARMS_RULED)."""
    """[REWRITTEN 2026-08-15] The registry grew Phase 7D's three backbones,
    so 'registered' and 'in the ladder' are now different facts with
    different names -- the constructable registry is seven, the CLOSED
    ladder set is still the four, and the ladder derives from the latter
    (factory.LADDER_BACKBONES; the 21-phantom-arms incident is recorded on
    that constant)."""
    assert set(factory.BACKBONES) == {
        "vit_b16", "swin_b", "srgnn", "agnet",
        "vit_b32", "vit_b8", "mvitv2_b",
        "vit_b14_dinov2", "vit_b16_dino",
    }
    assert factory.LADDER_BACKBONES == ("vit_b16", "swin_b", "srgnn", "agnet")
    assert set(factory.LADDER_BACKBONES) < set(factory.BACKBONES)


def test_every_backbone_declares_a_kind_from_the_closed_set():
    """**The kind decides three separate things** -- what cleft fine-tuning
    trains, what an embedding artifact holds, and which seed band applies -- so a
    backbone that omitted it would default into the transformer path and be
    silently wrong on all three."""
    for name in factory.BACKBONES:
        spec = factory.backbone_spec(name)
        assert spec["kind"] in factory.BACKBONE_KINDS

    assert factory.backbone_spec("vit_b16")["kind"] == "transformer"
    assert factory.backbone_spec("swin_b")["kind"] == "transformer"
    assert factory.backbone_spec("srgnn")["kind"] == "graph"
    assert factory.backbone_spec("agnet")["kind"] == "graph"


def test_an_unknown_backbone_is_refused_by_name():
    with pytest.raises(factory.FactoryError, match="unknown backbone"):
        factory.backbone_spec("resnet50")


def test_the_graph_backbones_declare_no_timm_name():
    """They are not timm models, and a timm name would route them to
    ``timm.create_model``, which would fail on the cluster after the queue wait
    rather than here."""
    for name in ("srgnn", "agnet"):
        assert factory.backbone_spec(name)["timm_name"] is None
    for name in ("vit_b16", "swin_b"):
        assert factory.backbone_spec(name)["timm_name"]


def test_phase_threes_default_backbone_still_resolves():
    """Phase 3's call sites pass the raw timm name. Breaking them would change
    what a frozen-apparatus arm builds."""
    assert factory.DEFAULT_BACKBONE == "vit_base_patch16_224"
    assert factory.BACKBONES["vit_b16"]["timm_name"] == factory.DEFAULT_BACKBONE


# --------------------------------------------------------------------------
# SR-GNN: the specification, and the check that says it is wrong
# --------------------------------------------------------------------------


def test_the_srgnn_parameter_count_is_derived_per_component():
    """A single total that is 8M light says nothing about which layer is wrong."""
    counts = srgnn.parameter_count()
    parts = (
        "self_attention", "gnn_mlp1", "gnn_mlp2", "gnn_out",
        "weighted_attention", "batch_norms", "classifier",
    )
    assert set(counts) >= {"backbone", "head_total", "total", *parts}
    assert counts["total"] == counts["backbone"] + counts["head_total"]
    assert counts["head_total"] == sum(counts[part] for part in parts)


def test_appnp_contributes_no_parameters():
    """[LITERATURE] APPNP is a fixed personalised-PageRank propagation, so K does
    not appear in the count. If a future edit makes the total depend on
    ``APPNP_K``, the propagation has stopped being APPNP."""
    baseline = srgnn.parameter_count()["total"]
    original = srgnn.APPNP_K
    try:
        srgnn.APPNP_K = original * 5
        assert srgnn.parameter_count()["total"] == baseline
    finally:
        srgnn.APPNP_K = original



def test_the_ported_spec_lands_near_the_paper():
    """**[RESOLVED 2026-07-31] The previous version of this test asserted the
    MISMATCH**, because the architecture was then reconstructed from the citation
    and came out 8.07M short. It said that rewriting it would be the moment to
    check what drove the change. What drove it: **the architecture was replaced
    by a port of a weight-transplant-verified implementation**, not by tuning
    widths to hit 30.9M. Not one width was chosen to close the gap -- every one is
    read off the verified source.

    Compared at ``num_outputs=200`` because the paper's figure is for CUB.
    Comparing a 1-output regression head against a 200-way classifier would be
    two different models -- the quantity confusion this project keeps finding.
    """
    result = srgnn.check_against_paper(num_outputs=200)
    assert result["agrees_within_tolerance"] is True
    assert result["ratio"] == pytest.approx(1.065, abs=0.01)
    assert "weight transplant" in result["verified_by"]
    assert "257/257" in result["verified_by"]


def test_the_residual_two_million_is_recorded_rather_than_chased():
    """The port is **+2.0M over** the paper's figure and that is not explained.
    Recorded rather than closed: the verification is the transplant (257/257
    tensors, <1e-6 at every stage), and a few percent against a figure quoted as
    "~30.9M" is within what counting conventions cover. Tuning the ported widths
    to remove it would break a verified port to satisfy an approximate number."""
    result = srgnn.check_against_paper(num_outputs=200)
    assert result["difference"] > 0
    assert 1_500_000 < result["difference"] < 2_500_000


def test_the_self_attention_layer_is_where_the_mass_is():
    """**The term a reconstruction cannot guess.** ``SeqSelfAttention`` projects
    the FLATTENED region descriptor -- 2048x7x7 = 100,352 dims -- to 32 units
    twice: 6.42M parameters in one layer. Missing it is what made the
    reconstruction 8.07M light. Pinned so a later "simplification" of the region
    descriptor cannot silently change the model's size class."""
    counts = srgnn.parameter_count(num_outputs=200)
    assert srgnn.REGION_FEATURE_DIM == 2048 * 7 * 7 == 100_352
    assert counts["self_attention"] == 6_422_593
    assert counts["self_attention"] > counts["head_total"] / 2, (
        "the self-attention is over half the non-backbone parameters"
    )


def test_the_provenance_is_the_transplant_not_the_paper():
    """A [REASONED] reconstruction and a transplant-verified port are different
    provenances, and the module must say which -- the previous revision tagged
    guessed widths [LITERATURE], which is the error that produced the gap."""
    source = module_source(srgnn)
    assert "WEIGHT-TRANSPLANT-VERIFIED" in source
    assert "257 of 257 weight tensors map" in source
    # The one part whose provenance was NOT the transplant is now verified
    # separately, and the record says which check covers it.
    assert srgnn.BUILD_VERIFICATION["appnp_is_verified"] is True
    assert "torch_geometric" in srgnn.BUILD_VERIFICATION["appnp_checked_against"]


def test_the_region_count_comes_from_the_ported_construction():
    """26 grid-combinatorial boxes plus the whole image. NOT a departure -- the
    previous revision recorded one because it had substituted Phase 2's scheme;
    the port uses SR-GNN's own, and it agrees at 27."""
    assert srgnn.N_ROIS == 26
    assert srgnn.N_REGIONS == 27


def test_the_backbone_constants_are_measured_with_their_image_as_provenance():
    """**[MEASURED 2026-07-31] Confirmed in the pinned image**, exact matches to
    the arithmetic derivations -- so two independent routes agree.

    The digest travels with the numbers because they are only true of that
    environment: a different timm or torchvision would resolve the backbones
    differently, and a measured constant without the environment that produced it
    cannot be re-derived.
    """
    assert srgnn.XCEPTION_FEATURE_PARAMETERS == 20_806_952
    assert agnet.RESNET50_BODY_PARAMETERS == 23_508_032
    assert agnet.RESNET50_FULL_PARAMETERS == 25_557_032
    # The subtraction is checkable rather than asserted.
    assert (
        agnet.RESNET50_FULL_PARAMETERS - agnet.RESNET50_BODY_PARAMETERS
        == 2048 * 1000 + 1000
    )

    digest = "sha256:2135e27b5d82b28cb5e2059c606aadf5736df80e50cb4e52fc669cc8ba33d2ab"
    for module in (srgnn, agnet):
        assert digest in module.MEASURED_IN_IMAGE
        assert "MEASURED 2026-07-31" in module_source(module)


def test_the_deprecated_xception_alias_is_not_relied_on():
    """**[MEASURED] timm warns that ``xception`` is deprecated and remaps it to
    ``legacy_xception``.** It resolves correctly today, and that is the risk: a
    version bump could point the alias at a different Xception variant, the
    parameter count would move, and the port would silently stop corresponding to
    the transplant-verified architecture -- a plausible model rather than a right
    one.

    So the explicit name is used, and the count is re-asserted at build time.
    """
    assert srgnn.BACKBONE_TIMM_NAME == "legacy_xception"
    source = module_source(srgnn)
    assert "deprecated" in source
    assert "silently remaps" in source
    # The build path names the constant explicitly rather than the alias, and
    # never passes a string literal to create_model.
    assert "BACKBONE_TIMM_NAME, pretrained=pretrained" in source
    assert 'create_model("xception"' not in source
    assert "create_model('xception'" not in source


def test_a_remapped_backbone_fails_loudly_rather_than_producing_a_model():
    """The build-time assertion is what makes the alias risk survivable. Checked
    with a stub, since torch is not in the test extra."""

    class FakeParameter:
        def __init__(self, n):
            self._n = n

        def numel(self):
            return self._n

    class FakeModel:
        def __init__(self, n):
            self._n = n

        def parameters(self):
            return [FakeParameter(self._n)]

    ok = srgnn.assert_backbone_is_the_measured_one(
        FakeModel(srgnn.XCEPTION_FEATURE_PARAMETERS)
    )
    assert ok["backbone_parameters"] == srgnn.XCEPTION_FEATURE_PARAMETERS
    assert "sha256:" in ok["measured_in"]

    with pytest.raises(srgnn.SRGNNSpecError, match="different Xception variant"):
        srgnn.assert_backbone_is_the_measured_one(FakeModel(22_855_952))


def test_build_refuses_rather_than_half_building_without_torch():
    """**The contract changed 2026-07-31: build() is real code now** -- the
    verified construction moved in from the first-construction script and the
    forwards from the transplant-verified Stage 1 lineage. What this test pins
    is what remains true on a machine without torch: the refusal is a NAMED
    error saying what is missing and where the laptop suite goes instead, not
    a raw ModuleNotFoundError from half-way down the build."""
    with pytest.raises(srgnn.SRGNNSpecError, match="pinned image"):
        srgnn.build()
    with pytest.raises(srgnn.SRGNNSpecError, match="stub"):
        srgnn.build()


def test_build_overrides_are_refused_by_name():
    """No overrides are defined, and a silently absorbed one would run an arm
    nobody configured -- the config loader's rule, applied to the build
    signature. Checked before the torch imports, so it runs on the laptop."""
    with pytest.raises(srgnn.SRGNNSpecError, match="unknown build override"):
        srgnn.build(dropout=0.5)
    with pytest.raises(ValueError, match="unknown build override"):
        agnet.build(kappa=4)


def test_the_grid_regions_are_the_papers_26():
    """The default-box path of forward(x): the grid-combinatorial enumeration,
    numpy-only and therefore testable here while the model is not. 26 boxes,
    inside the 42x42 map, whole image excluded (it is appended as region 27
    inside forward), deterministic."""
    boxes = srgnn.grid_rois()
    assert len(boxes) == srgnn.N_ROIS == 26
    assert boxes.dtype.name == "float32"
    for x, y, w, h in boxes.tolist():
        assert 0 <= x < srgnn.ROI_RESOLUTION and 0 <= y < srgnn.ROI_RESOLUTION
        assert w > 0 and h > 0
        assert x + w <= srgnn.ROI_RESOLUTION and y + h <= srgnn.ROI_RESOLUTION
        assert not (
            x == 0 and y == 0
            and w == srgnn.ROI_RESOLUTION and h == srgnn.ROI_RESOLUTION
        ), "the whole image must not appear twice"
    assert (srgnn.grid_rois() == boxes).all(), "the enumeration must be deterministic"


def test_training_determinism_is_recorded_per_backbone_with_its_environment():
    """**[MEASURED 2026-07-31, in the image] AG-Net's post-restore step is
    not bitwise flag-off (2.26e-05; 2.31e-05 in the local corroboration) and
    the other three are** -- a property to record, not a defect to hide. The
    records must carry the environment that produced each figure, the pasted
    provenance, and the script that re-derives them."""
    srgnn_record = srgnn.TRAINING_DETERMINISM
    assert srgnn_record["bitwise_flag_off"] is True
    assert srgnn_record["flag_off_post_restore_step_max_diff"] == 0.0
    assert "pasted" in srgnn_record["environment"]
    assert "verify_train_determinism" in srgnn_record["authoritative_for_runs"]

    agnet_record = agnet.TRAINING_DETERMINISM
    assert agnet_record["bitwise_flag_off"] is False
    assert agnet_record["flag_off_post_restore_step_max_diff"] == pytest.approx(
        2.26e-05
    )
    assert agnet_record["local_corroboration_max_diff"] == pytest.approx(2.31e-05)
    assert "atomicAdd" in agnet_record["mechanism"]
    assert "pasted" in agnet_record["environment"], (
        "the figures arrived from the cluster by paste and the record must say so"
    )
    assert "seed band" in agnet_record["flag_off_consequence"]
    # The settled decision travels with both records.
    assert "deterministic: false" in agnet_record["decision"]
    assert "deterministic: false" in srgnn_record["decision"]


def test_the_flag_cannot_fix_agnet_and_the_record_says_why():
    """Two reasons, both measured, both easy to reason past later: the
    deterministic roi_align substitution needs a C++ compiler the image does
    not have (InvalidCxxCompiler), and the op never raises under the flag --
    so gate-1's raised-on-nothing canary cannot see it in either state."""
    entry = agnet.TRAINING_DETERMINISM["under_deterministic_flag"]
    assert "C++ compiler" in entry
    assert "InvalidCxxCompiler" in entry
    assert "never raises" in entry
    assert "canary" in entry


def test_the_sift_gmm_regions_are_deterministic_and_normalised():
    """The seeded GMM is the recorded deviation from Stage 1 -- an unseeded one
    would give a resumed run different boxes than the run it resumes. Skipped
    where cv2 is absent (the test extra); it runs in the image and on the
    system python."""
    cv2 = pytest.importorskip("cv2")
    del cv2

    rng = np.random.default_rng(20260731)
    image = np.full((224, 224, 3), 30, dtype=np.uint8)
    for _ in range(6):
        cy, cx = rng.integers(30, 194, size=2)
        y, x = np.ogrid[:224, :224]
        image[(y - cy) ** 2 + (x - cx) ** 2 <= 144] = 200

    first = agnet.generate_srs(image)
    second = agnet.generate_srs(image)
    assert (first == second).all(), "same image, same boxes -- or resume breaks"
    assert first.ndim == 2 and first.shape[1] == 4
    assert (first >= 0.0).all() and (first <= 1.0).all()
    assert agnet.GMM_RANDOM_STATE == 0


# --------------------------------------------------------------------------
# AG-Net: a DIFFERENT provenance, and the region question that closed itself
# --------------------------------------------------------------------------


def test_agnet_provenance_is_weaker_than_srgnn_and_says_so():
    """**The two ports are NOT equally underwritten, and the record must not
    let them read as though they were.**

    SR-GNN was verified by weight transplant against the authors' released
    weights: 257/257 tensors, <1e-6 at every stage. **No authors' weights are
    released for AG-Net**, so no transplant is possible. What exists is a
    paper-faithful implementation plus one third-party reproduction that a
    forensic analysis discredited -- its 98.3% is void, its box geometry buggy,
    its trained self-attention effectively off.

    Describing "two ported architectures" without this distinction would be the
    provenance error that produced the 8.07M SR-GNN gap, in a new place.
    """
    described = agnet.describe()
    assert described["provenance_is_weaker_than_srgnn"] is True
    assert "NOT a weight transplant" in described["verified_by"]
    assert "no authors' weights are released" in described["verified_by"].lower()

    source = module_source(agnet)
    assert "PROVENANCE IS **NOT** SR-GNN's" in source
    assert "discredited" in source
    # And SR-GNN's record still claims the stronger thing, so the two differ.
    assert "257 of 257 weight tensors map" in module_source(srgnn)


def test_the_region_count_is_constant_by_construction():
    """**The open §4 question, closed by the authors' own design.** SIFT+GMM
    regions are image-dependent, so the schema question was padding vs
    variable-length vs fixing the component count as a recorded departure.

    None applies: the regions are the kappa primaries plus every unordered pair,
    so ``R+1 = 1 + kappa(kappa+1)/2`` is fixed once kappa is. The artifact is
    rectangular with no departure to record.
    """
    assert agnet.region_count(8) == 37
    assert agnet.region_count(4) == 11
    assert agnet.region_count(1) == 2  # one primary + whole image
    assert agnet.describe()["region_count_is_constant"] is True

    with pytest.raises(ValueError, match="kappa must be positive"):
        agnet.region_count(0)


def test_the_parameter_total_does_not_depend_on_the_region_count():
    """Every attention and SE weight is shared across regions, so kappa changes
    how many regions there are and not how many parameters. That independence IS
    the divergence from the reproduction, and it is worth being able to see."""
    totals = {
        kappa: agnet.parameter_count(num_outputs=200)["total"]
        for kappa in (4, 8, 16)
    }
    assert len(set(totals.values())) == 1, (
        f"the parameter total varies with kappa: {totals} -- weights are no "
        "longer shared across regions, which is the reproduction's design"
    )


def test_the_reproduction_count_is_recorded_as_not_a_target():
    """57.2M counts per-region and per-BATCH-POSITION weight lists that this port
    deliberately shares. Recorded so the gap is explained rather than discovered
    by someone comparing against the only number they can find."""
    described = agnet.describe()
    assert described["reproduction_parameters"] == 57_223_560
    assert "not a target" in module_source(agnet).lower()
    assert "BATCH POSITION" in described["why_not_a_target"]
    # The port is far smaller, which is the expected direction.
    assert described["parameters"]["total"] < described["reproduction_parameters"]


def test_agnet_has_no_published_parameter_target():
    """SR-GNN can be cross-checked against the paper's ~30.9M. AG-Net cannot --
    no such figure is published -- so the check that exists for one arm does not
    exist for the other. Stated, rather than a missing check going unnoticed."""
    assert not hasattr(agnet, "PAPER_PARAMETERS")
    assert hasattr(srgnn, "PAPER_PARAMETERS")


def test_the_two_backbones_are_the_same_order_of_size():
    """Both land near 30M with a 2048-channel backbone. Not a verification -- it
    is the weakest possible check -- but a port that came out 3x either way would
    be worth stopping for."""
    srgnn_total = srgnn.parameter_count(num_outputs=200)["total"]
    agnet_total = agnet.parameter_count(num_outputs=200)["total"]
    assert 0.5 < agnet_total / srgnn_total < 2.0


def test_agnet_build_refuses_rather_than_half_building():
    """Same contract change as SR-GNN's (2026-07-31): build() is real code,
    and without torch the refusal is a named error pointing at the image and
    the stub, not a raw import traceback."""
    with pytest.raises(ValueError, match="pinned image"):
        agnet.build()
    with pytest.raises(ValueError, match="stub"):
        agnet.build()


# --------------------------------------------------------------------------
# what was actually built and measured, 2026-07-31
# --------------------------------------------------------------------------


def test_the_appnp_reimplementation_is_verified_not_reasoned():
    """**[MEASURED] The one part of the SR-GNN port whose provenance was not the
    weight transplant.** The propagation was written directly rather than adding
    ``torch_geometric``, on the grounds that at K=1 it reduces to one line --
    and "reduces to" is what a parity check is for.

    Confirmed at **0.92 ULP worst case** across region counts, widths, both
    float dtypes, and K in {1,2,5,10}: sub-single-ULP, so the two are the same
    computation and the K=1 agreement is not a coincidence of that setting.
    """
    assert srgnn.APPNP_PARITY_WORST_ULP < 8
    assert srgnn.BUILD_VERIFICATION["appnp_is_verified"] is True
    assert srgnn.BUILD_VERIFICATION["appnp_parity_worst_ulp"] < 1.0, (
        "sub-single-ULP: the two are the same computation, not merely close"
    )
    # The dependency did not enter the TEST EXTRA or this module's imports.
    # (docker/requirements.txt has pinned torch-geometric since Phase 0 -- an
    # earlier claim that it was absent from the image was corrected 2026-07-31;
    # the direct propagation stands on the parity, not on absence.)
    assert "outside the project venv" in (
        srgnn.BUILD_VERIFICATION["appnp_checked_against"]
    )
    assert "CORRECTED 2026-07-31" in module_source(srgnn)


def test_both_models_hit_their_specifications_when_built():
    """**[MEASURED 2026-07-31] The first construction of either module.**

    Until this ran, both specifications were arithmetic that nothing had tested
    against an ``nn.Module``. Both matched exactly, first time -- which is what
    the build-time assertions exist to keep true.
    """
    assert srgnn.BUILT_PARAMETERS_CUB == 32_896_562
    assert agnet.BUILT_PARAMETERS_CUB == 30_742_924
    assert srgnn.parameter_count(num_outputs=200)["total"] == srgnn.BUILT_PARAMETERS_CUB
    assert agnet.parameter_count(num_outputs=200)["total"] == agnet.BUILT_PARAMETERS_CUB


def test_the_built_counts_were_checked_outside_the_pinned_image():
    """Recorded, because it changes what the check proves. Building on a
    different timm/torchvision tests the SPECIFICATION rather than the image --
    and the backbone terms still matched the image-measured constants, which is
    what makes those numbers properties of the architecture rather than of one
    environment."""
    for module in (srgnn, agnet):
        verification = module.BUILD_VERIFICATION
        assert verification["in_pinned_image"] is False
        assert verification["matched_spec_first_construction"] is True
        assert "SPECIFICATION, not the image" in verification["what_this_proves"]
        assert verification["built_parameters_cub"] == (
            module.parameter_count(num_outputs=200)["total"]
        )


def test_swin_embedding_dim_was_confirmed_against_the_built_model():
    """1024 was a claim in the registry until it was checked. timm reports
    num_features 1024 and a forward pass returns 1024."""
    assert factory.BACKBONES["swin_b"]["embedding_dim"] == 1024
    assert factory.BACKBONES["vit_b16"]["embedding_dim"] == 768
    source = module_source(factory)
    assert "confirmed against the BUILT model" in source


def test_the_two_transformers_normalise_differently():
    """**Measured, and the reason `normalization_for` reads each model's own
    config.** Swin-B uses ImageNet statistics; ViT-B/16 uses 0.5/0.5. Two
    backbones in one registry disagreeing is the precondition for the Stage-1
    defect of applying one triple to a model not trained with it."""
    assert "ImageNet statistics where ViT-B/16's is 0.5/0.5" in module_source(factory)


# --------------------------------------------------------------------------
# normalization: four backbones, two conventions, and one that reports nothing
# --------------------------------------------------------------------------


class _Reports:
    """A model that reports preprocessing the way timm does."""

    def __init__(self, mean, std):
        self.pretrained_cfg = {"mean": mean, "std": std}


class _ReportsNothing:
    """A model that reports preprocessing the way torchvision does: not at all."""


def test_normalization_refuses_rather_than_defaulting():
    """**[MEASURED 2026-07-31] The hole, and why a default is not an option.**

    torchvision's ResNet-50 carries no ``pretrained_cfg`` and no ``default_cfg``
    -- not on the model, not on the ``children()[:8]`` body AG-Net builds. So for
    one of the four backbones there was nothing to read, and the ways out were to
    refuse or to substitute a default. **The default IS the Stage-1 defect.**

    And no default would even be majority-right: the four split two-two between
    0.5/0.5 and ImageNet statistics.
    """
    with pytest.raises(factory.FactoryError) as excinfo:
        factory.normalization_for(_ReportsNothing())
    message = str(excinfo.value)
    assert "DO NOT SUBSTITUTE A DEFAULT" in message
    assert "two-two" in message
    assert "transforms()" in message, "the message must name the way out"


def test_a_partially_populated_config_is_not_a_licence_to_fill_in_the_rest():
    class Half:
        pretrained_cfg = {"mean": (0.5, 0.5, 0.5)}

    with pytest.raises(factory.FactoryError, match="missing mean/std"):
        factory.normalization_for(Half())


def test_a_stamp_supplies_what_torchvision_cannot_report():
    """The values still come from the weights -- ``transforms()`` on the enum --
    so this is reading the model's provenance, not being told what to assume."""
    model = factory.stamp_normalization(
        _ReportsNothing(), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
        source="torchvision ResNet50_Weights.IMAGENET1K_V2.transforms()",
    )
    mean, std = factory.normalization_for(model)
    assert mean == (0.485, 0.456, 0.406)
    assert std == (0.229, 0.224, 0.225)
    assert "IMAGENET1K_V2" in factory.normalization_source(model)


def test_an_unattributed_stamp_is_refused():
    """Without a source it is a hardcoded constant with a function call in front
    of it, which is the defect wearing a wrapper."""
    with pytest.raises(factory.FactoryError, match="must name its source"):
        factory.stamp_normalization(_ReportsNothing(), (0.5,) * 3, (0.5,) * 3, source="")


def test_the_reported_source_distinguishes_the_two_mechanisms():
    assert factory.normalization_source(_Reports((0.5,) * 3, (0.5,) * 3)) == (
        "timm pretrained_cfg"
    )
    with pytest.raises(factory.FactoryError, match="no normalization source"):
        factory.normalization_source(_ReportsNothing())


def test_a_stamp_takes_precedence_and_both_paths_return_the_models_own_values():
    """Two backbones in one registry disagreeing is the Stage-1 precondition, so
    the resolver must return each model's own values rather than a shared one."""
    vit_like = _Reports((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    swin_like = _Reports((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    assert factory.normalization_for(vit_like)[0] == (0.5, 0.5, 0.5)
    assert factory.normalization_for(swin_like)[0] == (0.485, 0.456, 0.406)


def test_agnet_records_that_its_normalization_comes_from_the_weights_enum():
    """And that the build path must stamp it -- without the stamp,
    normalization_for refuses and the only other way forward is the default."""
    assert "transforms()" in agnet.NORMALIZATION_SOURCE
    assert agnet.RESNET_WEIGHTS_DEFAULT == "IMAGENET1K_V2"
    # The build path must stamp from that exact enum's transforms(), and the
    # stamp source names it -- asserted on the source text here, behaviourally
    # by scripts/verify_backbone_builds.py where torch exists.
    source = module_source(agnet)
    assert "stamp_normalization(" in source
    assert "IMAGENET1K_V2.transforms()" in source


def test_the_weights_enum_choice_is_not_a_normalization_choice():
    """V1 and V2 carry the SAME mean/std and differ only in resize (256 vs 232),
    which does not reach this pipeline -- staged crops arrive at 224. Recorded so
    the two are not conflated: they are different quantities."""
    source = module_source(agnet)
    assert "same mean/std" in source.lower()
    assert "different quantities" in source
