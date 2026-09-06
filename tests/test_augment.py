"""The Phase 7C augmenter: identity, lockstep, softness, determinism.

The checks that matter here are the ones a plausible-looking image would
otherwise pass: that identity is exactly identity (arm 0 is a gate), that the
mask and the boxes move with the pixels rather than beside them, that the
falloff is genuinely soft, and that a resumed epoch reproduces its views.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleft import phase7c
from cleft.train import augment


def _image(size=32, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3)).astype(np.float64)


def _settings():
    return phase7c.PHOTOMETRIC, phase7c.GEOMETRIC


# --------------------------------------------------------------------------
# arm 0: identity must be exactly identity
# --------------------------------------------------------------------------


def test_the_identity_policy_returns_the_input_untouched():
    """**Arm 0 is a gate on reproducing 0.2520.** A policy that round-tripped
    every pixel through float64 and back would be testing something slightly
    different from the arm it must reproduce, and the difference would look
    like noise rather than like a bug."""
    images = np.arange(2 * 8 * 8 * 3, dtype=np.uint8).reshape(2, 8, 8, 3)
    photometric, geometric = _settings()

    out = augment.augment_batch(
        images,
        policy=augment.Policy(),
        seed=1337, fold=0, epoch=0,
        photometric_settings=photometric,
        geometric_settings=geometric,
    )
    assert out is not None
    assert np.array_equal(out, images)
    assert out.dtype == images.dtype

    assert augment.Policy().is_identity
    assert not augment.Policy(photometric=True).is_identity
    assert not augment.Policy(geometric=True).is_identity


def test_integer_pixels_are_rounded_not_truncated():
    """``astype`` truncates toward zero -- on non-negative pixels that is a
    systematic downward bias of about half a level on every augmented pixel.
    A brightness shift applied to the training set and not the evaluation set
    is exactly what would read as an augmentation effect."""
    photometric, geometric = _settings()
    images = np.full((8, 16, 16, 3), 128, dtype=np.uint8)

    out = augment.augment_batch(
        images, policy=augment.Policy(photometric=True),
        seed=1337, fold=0, epoch=0,
        photometric_settings=photometric, geometric_settings=geometric,
    )
    assert out.dtype == np.uint8

    # Truncation would pull the mean below the float result; rounding does not.
    exact = np.stack([
        augment.augment_image(
            images[i].astype(np.float64), policy=augment.Policy(photometric=True),
            rng=augment.rng_for(1337, 0, 0),
            photometric_settings=photometric, geometric_settings=geometric,
        )["image"]
        for i in range(1)
    ])
    assert abs(out[0].mean() - exact[0].mean()) < 0.5


def test_every_declared_arm_builds_a_policy():
    for arm in phase7c.arms():
        policy = augment.Policy.from_arm(arm)
        assert policy.photometric == arm["photometric"]
        assert policy.region_aware == arm["region_aware"]
        assert policy.is_identity == (arm["index"] == 0)


def test_rotation_without_geometric_is_refused():
    """Rotation is a geometric transform; an arm carrying it alone declares a
    policy the augmenter cannot build."""
    with pytest.raises(augment.AugmentError, match="rotation without geometric"):
        augment.Policy(rotation=True)


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_the_same_coordinates_reproduce_the_same_views():
    """**A resumed run restarts the epoch loop**, and PLAN §2.7's measured
    hazard is that run identity survives a pause while training does not. A
    view drawn from a shared, advancing stream would depend on how many draws
    came before it; one drawn from (seed, fold, epoch) does not.
    """
    images = _image()[None]
    photometric, geometric = _settings()
    policy = augment.Policy(photometric=True, geometric=True)

    def run(seed, fold, epoch):
        return augment.augment_batch(
            images, policy=policy, seed=seed, fold=fold, epoch=epoch,
            photometric_settings=photometric, geometric_settings=geometric,
        )

    assert np.array_equal(run(1337, 0, 0), run(1337, 0, 0))
    # Different coordinates must genuinely differ, or the schedule is not
    # producing varied views and augmentation is doing nothing.
    for other in ((1337, 0, 1), (1337, 1, 0), (2024, 0, 0)):
        assert not np.array_equal(run(1337, 0, 0), run(*other)), other


def test_the_generator_is_derived_not_shared():
    a = augment.rng_for(1337, 0, 0).random(4)
    b = augment.rng_for(1337, 0, 0).random(4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, augment.rng_for(1337, 0, 1).random(4))


# --------------------------------------------------------------------------
# lockstep
# --------------------------------------------------------------------------


def test_the_forward_and_inverse_matrices_cannot_disagree():
    """They come from one construction. A mask placed by one and pixels warped
    by the other is how the mask ends up somewhere the image is not."""
    params = {"degrees": 7.0, "translate_x": 3.0, "translate_y": -2.0, "scale": 1.05}
    forward, inverse = augment.affine_matrices(params, 32)
    assert np.allclose(forward @ inverse, np.eye(3), atol=1e-12)


def test_an_identity_affine_leaves_pixels_and_boxes_where_they_were():
    params = {"degrees": 0.0, "translate_x": 0.0, "translate_y": 0.0, "scale": 1.0}
    forward, inverse = augment.affine_matrices(params, 16)
    image = _image(16)

    assert np.allclose(augment.warp_image(image, inverse), image, atol=1e-9)
    boxes = [(2.0, 3.0, 4.0, 5.0)]
    assert augment.warp_boxes(boxes, forward) == pytest.approx(boxes)


def test_a_box_lands_where_its_content_lands():
    """**Exit criterion 4, as arithmetic.** A rotated image with a stationary
    mask is a different experiment from the one intended, so the box is
    checked against where the PIXELS it covered actually went -- not against a
    separately-computed box, which would only prove two functions agree.
    """
    size = 64
    image = np.full((size, size, 3), 255.0)
    # A bright marker inside a known box.
    box = (20.0, 24.0, 8.0, 6.0)
    x, y, w, h = (int(v) for v in box)
    image[y:y + h, x:x + w] = 0.0

    params = {"degrees": 5.0, "translate_x": 2.0, "translate_y": -3.0, "scale": 1.0}
    forward, inverse = augment.affine_matrices(params, size)

    warped = augment.warp_image(image, inverse)
    moved = augment.warp_boxes([box], forward)[0]

    # Every dark pixel must fall inside the moved box's envelope.
    dark_y, dark_x = np.where(warped[..., 0] < 128.0)
    assert dark_x.size > 0, "the marker vanished; the warp lost the content"
    mx, my, mw, mh = moved
    assert dark_x.min() >= mx - 1.5 and dark_x.max() <= mx + mw + 1.5
    assert dark_y.min() >= my - 1.5 and dark_y.max() <= my + mh + 1.5


def test_a_rotated_box_becomes_an_envelope_and_only_grows():
    """A rotated rectangle is not a rectangle. The envelope protects slightly
    more than the region, which is the safe direction for a spatial prior on
    what to preserve -- and it is recorded rather than silently applied."""
    box = (10.0, 10.0, 8.0, 4.0)
    still, _ = augment.affine_matrices(
        {"degrees": 0.0, "translate_x": 0.0, "translate_y": 0.0, "scale": 1.0}, 32
    )
    turned, _ = augment.affine_matrices(
        {"degrees": 5.0, "translate_x": 0.0, "translate_y": 0.0, "scale": 1.0}, 32
    )
    _, _, w0, h0 = augment.warp_boxes([box], still)[0]
    _, _, w1, h1 = augment.warp_boxes([box], turned)[0]
    assert w1 > w0 and h1 > h0


def test_the_warp_fills_from_outside_with_white_not_black():
    """The crops are white-padded (PLAN §4.4). Black would introduce a border
    no real image has, appearing only in training -- a feature a model can key
    on, which is the same concern as the hard-mask seam."""
    image = np.zeros((32, 32, 3))
    params = {"degrees": 0.0, "translate_x": 10.0, "translate_y": 0.0, "scale": 1.0}
    _, inverse = augment.affine_matrices(params, 32)
    warped = augment.warp_image(image, inverse)
    assert warped[:, 0, 0] == pytest.approx(augment.FILL_VALUE)
    assert augment.FILL_VALUE == 255.0


# --------------------------------------------------------------------------
# the protection mask
# --------------------------------------------------------------------------


def test_the_mask_is_soft_and_reaches_both_strengths():
    """A hard seam sits at a fixed anatomical offset on the regions the grade
    is about. Softness is the point, so it is measured: the mask must take
    values strictly between the two levels."""
    mask = augment.protection_mask(
        64, 64, [(20, 20, 24, 24)],
        inside_strength=phase7c.REGION_AWARE["inside_strength"],
        sigma_fraction=phase7c.BLEND["sigma_fraction_of_crop_width"],
    )
    inside = phase7c.REGION_AWARE["inside_strength"]

    assert mask.min() >= inside - 1e-9
    assert mask.max() <= 1.0 + 1e-9
    between = (mask > inside + 0.02) & (mask < 0.98)
    assert between.sum() > 0, "the mask has no transition band; it is a hard seam"

    # The centre is protected and a far corner is not.
    assert mask[32, 32] < inside + 0.05
    assert mask[0, 0] > 0.95


def test_a_hard_mask_would_fail_that_check():
    """The softness assertion is only worth having if a hard mask fails it."""
    hard = augment.protection_mask(
        64, 64, [(20, 20, 24, 24)], inside_strength=0.25, sigma_fraction=0.0
    )
    between = (hard > 0.27) & (hard < 0.98)
    assert between.sum() == 0
    assert set(np.unique(hard)) == {0.25, 1.0}


def test_the_mask_blur_does_not_pull_the_border_toward_protection():
    """Zero padding at the edge would drag the mask down along the frame,
    protecting the periphery -- the opposite of the policy. Phase 1's row
    profile was a zero pad doing this at the ends of a signal."""
    mask = augment.protection_mask(
        48, 48, [(20, 20, 8, 8)], inside_strength=0.25, sigma_fraction=0.03
    )
    assert mask[0, 0] > 0.99 and mask[-1, -1] > 0.99


def test_an_out_of_range_strength_is_refused():
    with pytest.raises(augment.AugmentError, match="outside"):
        augment.protection_mask(8, 8, [], inside_strength=1.5, sigma_fraction=0.0)


# --------------------------------------------------------------------------
# photometric, and what the mask does to it
# --------------------------------------------------------------------------


def test_region_protection_moves_protected_pixels_less():
    """The whole claim of region-awareness, measured: inside a protected box
    the pixels must change less than outside it."""
    photometric, _ = _settings()
    image = np.full((64, 64, 3), 120.0)
    boxes = [(20, 20, 24, 24)]
    strength = augment.protection_mask(
        64, 64, boxes,
        inside_strength=phase7c.REGION_AWARE["inside_strength"],
        sigma_fraction=phase7c.BLEND["sigma_fraction_of_crop_width"],
    )

    plain = augment.apply_photometric(image, augment.rng_for(1, 0, 0), photometric)
    masked = augment.apply_photometric(
        image, augment.rng_for(1, 0, 0), photometric, strength
    )

    inside = np.abs(masked - image)[32, 32].mean()
    outside = np.abs(masked - image)[2, 2].mean()
    assert inside < outside, "protected pixels moved as much as unprotected ones"
    # Outside the boxes the masked result is the unmasked one.
    assert masked[2, 2] == pytest.approx(plain[2, 2], abs=1e-6)


def test_hue_rotation_preserves_luma():
    """Otherwise hue jitter silently doubles as brightness jitter and the
    photometric family confounds itself."""
    image = _image(16)
    rotated = augment._rotate_hue(image, 0.05)
    luma = augment._RGB_TO_YIQ[0]
    assert (image @ luma) == pytest.approx(rotated @ luma, abs=1e-9)


def test_photometric_stays_in_range():
    photometric, _ = _settings()
    out = augment.apply_photometric(
        _image(24), augment.rng_for(7, 0, 0), photometric
    )
    assert out.min() >= 0.0 and out.max() <= 255.0


# --------------------------------------------------------------------------
# the whole policy
# --------------------------------------------------------------------------


def test_a_non_square_or_wrong_shaped_crop_is_refused():
    photometric, geometric = _settings()
    with pytest.raises(augment.AugmentError, match=r"\(H, W, 3\)"):
        augment.augment_image(
            np.zeros((8, 8)), policy=augment.Policy(photometric=True),
            rng=augment.rng_for(0, 0, 0),
            photometric_settings=photometric, geometric_settings=geometric,
        )
    with pytest.raises(augment.AugmentError, match="square"):
        augment.augment_image(
            np.zeros((8, 12, 3)), policy=augment.Policy(photometric=True),
            rng=augment.rng_for(0, 0, 0),
            photometric_settings=photometric, geometric_settings=geometric,
        )


def test_region_awareness_modulates_photometric_and_geometry_stays_global():
    """**[DECIDED] A global affine cannot be spatially modulated** without
    tearing, so region-awareness applies to photometric only. Arms 5 and 6
    still differ in exactly one factor -- both carry the same global geometry
    -- but the record must not imply the rotation was masked."""
    photometric, geometric = _settings()
    strength = augment.protection_mask(
        64, 64, [(20, 20, 24, 24)],
        inside_strength=phase7c.REGION_AWARE["inside_strength"],
        sigma_fraction=phase7c.BLEND["sigma_fraction_of_crop_width"],
    )

    aware = augment.augment_image(
        _image(64), policy=augment.Policy(photometric=True, geometric=True,
                                          region_aware=True),
        rng=augment.rng_for(3, 0, 0),
        photometric_settings=photometric, geometric_settings=geometric,
        strength=strength,
    )
    assert aware["applied"]["photometric"]["region_aware"] is True
    assert aware["applied"]["photometric"]["min_strength"] < 1.0
    # Geometry ran, and it ran on the whole frame -- ONE affine, not a field.
    # There is no per-region geometry and there cannot be: a global affine
    # has no spatial modulation that does not tear the image.
    assert aware["applied"]["affine"] is not None
    assert set(aware["applied"]["affine"]) == {
        "degrees", "translate_x", "translate_y", "scale"
    }

    # An arm with geometric but NOT region_aware ignores the mask entirely.
    plain = augment.augment_image(
        _image(64), policy=augment.Policy(photometric=True, geometric=True),
        rng=augment.rng_for(3, 0, 0),
        photometric_settings=photometric, geometric_settings=geometric,
        strength=strength,
    )
    assert plain["applied"]["photometric"]["region_aware"] is False


def test_a_strength_map_that_does_not_match_the_image_is_refused():
    """The masks are per patient and travel as an image channel; a mismatch
    means the rows and the masks came apart, which would protect the wrong
    anatomy in every image and still produce a plausible number."""
    photometric, geometric = _settings()
    with pytest.raises(augment.AugmentError, match="does not match the image"):
        augment.augment_image(
            _image(32), policy=augment.Policy(photometric=True, region_aware=True),
            rng=augment.rng_for(0, 0, 0),
            photometric_settings=photometric, geometric_settings=geometric,
            strength=np.ones((16, 16)),
        )

    with pytest.raises(augment.AugmentError, match="strength maps"):
        augment.augment_batch(
            np.zeros((3, 8, 8, 3)), policy=augment.Policy(photometric=True),
            seed=1, fold=0, epoch=0,
            photometric_settings=photometric, geometric_settings=geometric,
            strength_per_image=np.ones((2, 8, 8)),
        )


def test_a_policy_without_rotation_never_rotates():
    """Arms 2, 5 and 6 carry geometric without rotation. If the draw ignored
    the flag, the rotation arm would not isolate rotation."""
    photometric, geometric = _settings()
    for _ in range(20):
        params = augment.sample_affine(
            augment.rng_for(11, 0, 0),
            augment.Policy(geometric=True), geometric, 224,
        )
        assert params["degrees"] == 0.0
    turned = augment.sample_affine(
        augment.rng_for(11, 0, 0),
        augment.Policy(geometric=True, rotation=True), geometric, 224,
    )
    assert abs(turned["degrees"]) <= geometric["rotation_degrees"]


def test_the_augmenting_backbone_augments_training_and_never_evaluation():
    """**The structural guarantee, exercised.**

    ``train_epoch`` augments; ``predict`` does not. Driven with a stub
    extractor so the claim is about the call graph rather than about ViT --
    the same images through ``predict`` twice must give identical features,
    and through ``train_epoch`` twice must not.
    """
    from cleft.train.torch_backbone import AugmentingHeadBackbone

    photometric, geometric = _settings()
    seen: dict[str, list] = {"extracted": []}

    class _Extractor:
        parameter_report = {"backbone_frozen": True}

        def __call__(self, images):
            seen["extracted"].append(np.asarray(images, dtype=np.float64).copy())
            return np.asarray(images, dtype=np.float32).mean(axis=(1, 2))

    class _Head:
        parameter_report = {"trainable_parameters": 4}

        def reset(self, labels):
            self.mean = float(np.mean(labels))

        def train_epoch(self, features, labels):
            return 0.0

        def predict(self, features):
            return np.full(len(features), self.mean)

    backbone = AugmentingHeadBackbone(
        policy=augment.Policy(photometric=True, geometric=True),
        photometric_settings=photometric,
        geometric_settings=geometric,
        seed=1337, fold_ordinal=0, extractor=_Extractor(),
    )
    backbone._head = _Head()
    backbone._head.reset(np.array([2.0, 3.0]))
    backbone._epoch = 0

    pixels = np.stack([_image(16), _image(16, seed=1)])
    strength = np.ones((2, 16, 16))
    features = np.concatenate([pixels, strength[..., None]], axis=3)

    backbone.predict(features)
    backbone.predict(features)
    assert np.array_equal(seen["extracted"][0], seen["extracted"][1]), (
        "predict augmented its input; evaluation must never be augmented"
    )
    assert np.array_equal(seen["extracted"][0], pixels), (
        "predict altered the pixels at all"
    )

    seen["extracted"].clear()
    backbone.train_epoch(features, np.array([2.0, 3.0]))
    backbone.train_epoch(features, np.array([2.0, 3.0]))
    assert not np.array_equal(seen["extracted"][0], seen["extracted"][1]), (
        "two epochs saw identical pixels; the epoch counter is not advancing"
    )
    assert not np.array_equal(seen["extracted"][0], pixels)


def test_two_epochs_extract_different_pixels_or_the_augmenter_is_not_running():
    """**The decisive check, and the one that was missing.**

    [MEASURED 2026-08-03] All seven Phase 7C arms returned byte-identical
    predictions, including arm 6 which augments the whole frame at full
    strength. Seven policies cannot produce identical pixels. The cause was a
    whitelist: ``task_train_cv`` builds ``backbone_config`` from five named
    keys and ``augmentation`` was not one, so the policy reached the run
    record and never the code.

    Everything around it was tested. ``prepare_features`` was tested with a
    config handed to it directly; the policy flags were tested against the
    shipped YAML; the augmenter was tested on arrays. **Every link was
    exercised and the chain was not** -- and a unit test that supplies the
    input the production path fails to supply cannot fail.

    This asserts the property that was actually violated: under a non-identity
    policy, two epochs must not see the same pixels.
    """
    from cleft.train.torch_backbone import AugmentingHeadBackbone

    photometric, geometric = _settings()
    extracted = []

    class _Extractor:
        def __call__(self, images):
            extracted.append(np.asarray(images, dtype=np.float64).copy())
            return np.asarray(images, dtype=np.float32).mean(axis=(1, 2))

    class _Head:
        parameter_report: dict = {}

        def reset(self, labels):
            pass

        def train_epoch(self, features, labels):
            return 0.0

        def predict(self, features):
            return np.zeros(len(features))

    features = np.concatenate(
        [np.stack([_image(16), _image(16, seed=1)]), np.ones((2, 16, 16, 1))],
        axis=3,
    )
    for policy in (
        augment.Policy(photometric=True),
        augment.Policy(geometric=True),
        augment.Policy(geometric=True, rotation=True),
        augment.Policy(photometric=True, geometric=True),
    ):
        extracted.clear()
        backbone = AugmentingHeadBackbone(
            policy=policy,
            photometric_settings=photometric,
            geometric_settings=geometric,
            seed=1337, fold_ordinal=0, extractor=_Extractor(),
        )
        backbone._head = _Head()
        backbone._epoch = 0
        backbone.train_epoch(features, np.array([2.0, 3.0]))
        backbone.train_epoch(features, np.array([2.0, 3.0]))

        assert not np.array_equal(extracted[0], extracted[1]), (
            f"{policy} extracted identical pixels in two epochs -- the "
            "augmenter is not in the call path"
        )
        assert not np.array_equal(extracted[0], features[..., :3]), (
            f"{policy} extracted the input unchanged"
        )


def test_the_backbone_refuses_features_without_the_strength_channel():
    """The mask travels as a fourth channel because the harness slices rows.
    A three-channel array means it came apart from the pixels somewhere."""
    from cleft.train.torch_backbone import AugmentingHeadBackbone

    with pytest.raises(ValueError, match=r"\(n, H, W, 4\)"):
        AugmentingHeadBackbone.split(np.zeros((2, 8, 8, 3)))
    pixels, strength = AugmentingHeadBackbone.split(np.zeros((2, 8, 8, 4)))
    assert pixels.shape == (2, 8, 8, 3) and strength.shape == (2, 8, 8)


def test_the_factory_hands_out_one_backbone_per_fold_in_order():
    """**A checked assumption about frozen code.** The zero-argument factory
    protocol cannot pass the fold, so the ordinal comes from the call index --
    which is only the fold's position because ``run_cv`` calls the factory
    once per fold, in order. Asserted rather than trusted."""
    from cleft.train.phase3 import Phase3Error, _AugmentingFactory

    factory = _AugmentingFactory(
        "vit_b16", {"batch_size": 4}, 1337,
        {"photometric": True, "geometric": False,
         "rotation": False, "region_aware": False},
    )
    assert factory.calls == 0
    factory.assert_called_once_per_fold(0)

    # The extractor is built lazily and shared, so a laptop without torch can
    # still check the counting -- and a real run builds ViT once, not per fold.
    factory._extractor = object()
    handed = [factory() for _ in range(5)]
    assert [b.fold_ordinal for b in handed] == [0, 1, 2, 3, 4]
    assert len({id(b) for b in handed}) == 5, "the factory returned one instance"
    assert len({id(b.extractor) for b in handed}) == 1, "the backbone was rebuilt"
    factory.assert_called_once_per_fold(5)

    with pytest.raises(Phase3Error, match="once per fold"):
        factory.assert_called_once_per_fold(4)


def test_prepare_features_hands_the_harness_pixels_plus_a_mask():
    """The routing: an augmented arm gets (n, H, W, 4), not stored embeddings.
    Without region-awareness the mask is all ones, so the channel is present
    and inert rather than absent -- one array shape for every augmented arm."""
    from cleft.train.phase3 import prepare_features

    images = np.zeros((3, 16, 16, 3), dtype=np.uint8)
    features, report = prepare_features(
        images, "vit_b16", "head",
        {"augmentation": {"photometric": True, "region_aware": False}},
        geometry="g1",
    )
    assert features.shape == (3, 16, 16, 4)
    assert np.all(features[..., 3] == 1.0)
    assert report["features"] == "augmented_pixels_and_strength_mask"
    assert report["region_aware"] is False


def test_the_declared_magnitudes_bound_what_is_drawn():
    """Pre-registered strengths are only pre-registered if the sampler
    respects them."""
    photometric, geometric = _settings()
    rng = augment.rng_for(5, 2, 3)
    low, high = geometric["scale"]
    for _ in range(200):
        params = augment.sample_affine(
            rng, augment.Policy(geometric=True, rotation=True), geometric, 224
        )
        assert abs(params["degrees"]) <= geometric["rotation_degrees"]
        assert low <= params["scale"] <= high
        limit = geometric["translate_fraction"] * 224
        assert abs(params["translate_x"]) <= limit
        assert abs(params["translate_y"]) <= limit
