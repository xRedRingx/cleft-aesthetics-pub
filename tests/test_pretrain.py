"""Phase 6 pretraining: the loop, the checkpoint wiring, and the config lattice.

Torch never appears: every training test runs the stub model through the SAME
loop, the same checkpoint calls and the same band check the cluster runs will
use. The synthetic SCUT root encodes each face's label in its pixel values, so
the stub genuinely learns and the test-side correlation is a real number the
band check can act on -- a loop tested only on noise would leave the
"consumable" path exercised by nothing.

The resume tests are what give the checkpoint wiring teeth on a machine with
no Run:AI to pause it: byte-identity for the journal, state-identity for the
deliverable, and a counterfactual resume that LOSES optimizer state and must
be seen to diverge (a byte-identity test that cannot fail is not a test --
PLAN R7).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from cleft.train import checkpoint as ckpt
from cleft.train import pretrain
from cleft.train.harness import HarnessError
from cleft.train.pretrain import PretrainConfig, PretrainError, StubPretrainModel

from fixtures import builders


# --------------------------------------------------------------------------
# synthetic SCUT: split files + images whose pixels encode the label
# --------------------------------------------------------------------------

#: Distinct labels, so any inner-val subset has variance (gate 3 requires it).
TRAIN_ROWS = [
    ("AF1", 1.2), ("AF2", 4.6), ("AM1", 2.1), ("AM2", 3.8),
    ("CF1", 1.7), ("CF2", 4.1), ("CM1", 2.6), ("CM2", 3.3),
]
TEST_ROWS = [("TF1", 1.5), ("TF2", 4.4), ("TM1", 2.4), ("TM2", 3.6)]


def value_of(label: float) -> int:
    """The pixel value that encodes a label. Linear, well inside 0-255."""
    return int(round((label - 1.0) / 4.0 * 200)) + 20


def write_split(root: Path, train_rows, test_rows) -> None:
    from cleft.scut.dataset import SPLIT_DIR

    directory = root / SPLIT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    for side, rows in (("train", train_rows), ("test", test_rows)):
        (directory / f"{side}.txt").write_text(
            "".join(f"{stem}.jpg {value}\n" for stem, value in rows),
            encoding="utf-8",
        )


def write_images(root: Path, rows, *, size: int = 32, noise_seed: int | None = None):
    """JPEGs under Images/. Constant-valued from the label, or seeded noise."""
    from PIL import Image

    from cleft.scut.dataset import IMAGE_DIR

    directory = root / IMAGE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(noise_seed)
    for stem, label in rows:
        if noise_seed is None:
            array = np.full((size, size, 3), value_of(label), dtype=np.uint8)
        else:
            array = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
        Image.fromarray(array).save(directory / f"{stem}.jpg")


def make_scut_root(root: Path, *, noise_seed: int | None = None) -> Path:
    write_split(root, TRAIN_ROWS, TEST_ROWS)
    write_images(root, TRAIN_ROWS + TEST_ROWS, noise_seed=noise_seed)
    return root


def make_masked_artifact(
    directory: Path, rows_in_faces_order, *, content_boxes: bool = True
) -> Path:
    """faces.json + both geometry arrays, row i encoding faces[i]'s label.

    Per-geometry records carry ``content_box`` like the real artifact does
    (the frozen ``stage()``'s output, recorded at build time) -- varied per
    face so a scheme test can see per-face box placement. ``content_boxes:
    False`` reproduces an artifact predating the recorded geometry.
    """
    directory.mkdir(parents=True, exist_ok=True)
    faces = []
    for position, (stem, _) in enumerate(rows_in_faces_order):
        entry = {"stem": stem, "aspect_ratio": 0.74}
        if content_boxes:
            width = 160 + 8 * (position % 5)
            box = [(224 - width) // 2, 0, width, 224]
            entry["g1"] = {"content_box": box}
            entry["g2"] = {"content_box": box}
        faces.append(entry)
    (directory / "faces.json").write_text(
        json.dumps(faces, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for geometry in ("g1", "g2"):
        array = np.stack(
            [
                np.full((8, 8, 3), value_of(label), dtype=np.uint8)
                for _, label in rows_in_faces_order
            ]
        )
        np.save(directory / f"masked_{geometry}.npy", array)
    return directory


def small_config(**overrides) -> PretrainConfig:
    values = dict(
        epochs=12, inner_val_frac=0.25, seed=7,
        deterministic=False,
        monitor="inner_val_pcc", batch_size=4, checkpoint_every=1,
    )
    values.update(overrides)
    return PretrainConfig(**values)


def run(tmp_path: Path, root: Path, *, subdir: str = "run", **overrides):
    run_dir = tmp_path / subdir
    run_dir.mkdir(parents=True, exist_ok=True)
    arguments = dict(
        scut_root=root,
        source="original",
        backbone="stub",
        region_scheme="none",
        masked_dir=None,
        expect_train=len(TRAIN_ROWS),
        expect_test=len(TEST_ROWS),
        run_dir=run_dir,
        curves_path=run_dir / "curves.csv",
        pretrained_path=run_dir / "pretrained.npz",
        config=small_config(),
    )
    arguments.update(overrides)
    return pretrain.run_pretraining(**arguments), run_dir


# --------------------------------------------------------------------------
# the loop, end to end
# --------------------------------------------------------------------------


def test_pretraining_learns_the_synthetic_originals(tmp_path):
    """End to end on the learnable fixture: the label is in the pixels, so the
    stub must reach a high test correlation and the band check must mark the
    checkpoint consumable -- the path the twelve real runs are meant to take."""
    root = make_scut_root(tmp_path / "scut")
    result, run_dir = run(tmp_path, root)

    assert not result.paused
    summary = result.summary
    assert summary["n_train"] == 8 and summary["n_test"] == 4
    assert summary["n_fit"] == 6 and summary["n_inner_val"] == 2
    assert summary["test"]["pcc"] > 0.9
    assert summary["band_check"]["applies"] is True
    assert summary["band_check"]["checkpoint_consumable"] is True
    assert summary["parameters"]["trainable_fraction"] == 1.0
    assert summary["resumed"] is False
    assert summary["epoch0"]["pred_mean"] == pytest.approx(
        np.mean([label for _, label in TRAIN_ROWS]), abs=1.0
    )

    # The journal has one row per epoch run.
    lines = (run_dir / "curves.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "epoch,train_loss,inner_val_mse,inner_val_pcc"
    assert len(lines) - 1 == summary["epochs_run"]

    # The deliverable exists in the verified format; resume state is gone.
    assert not (run_dir / ckpt.CHECKPOINT_NAME).exists()
    saved = ckpt.load(run_dir / "pretrained.npz")
    assert saved.step == summary["selected_epoch"]
    assert saved.extra["test_pcc"] == summary["test"]["pcc"]
    assert all(key.startswith(pretrain.MODEL_PREFIX) for key in saved.arrays)

    # Test table: sorted stems, truth matching the fixture.
    stems = [stem for stem, _, _ in result.test_table]
    assert stems == sorted(stem for stem, _ in TEST_ROWS)


def test_the_budget_is_fixed_and_nothing_stops_a_run_early(tmp_path):
    """[DECIDED 2026-07-31] Every run trains exactly its budget, whatever the
    monitor does -- the noise fixture's flat curve is precisely the shape
    patience used to terminate on. The kept checkpoint is the best-by-monitor
    epoch, and the policy record travels in the summary so a reader of any
    metrics.json meets the gate-4 distinction without excavating the module."""
    root = make_scut_root(tmp_path / "scut", noise_seed=99)
    result, run_dir = run(tmp_path, root, config=small_config(epochs=9))
    summary = result.summary

    assert summary["epochs_run"] == summary["epochs_budget"] == 9
    lines = (run_dir / "curves.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) - 1 == 9, "the journal must show the full budget"
    assert 1 <= summary["selected_epoch"] <= 9

    assert "stopped_early" not in summary
    assert "patience" not in summary["train_config"]
    policy = summary["selection_policy"]
    assert policy["policy"] == "fixed_epoch_budget_best_checkpoint"
    assert policy["early_stopping"] is False
    assert "gate 4" in policy["gate4_distinction"] or "gate4" in str(policy)
    assert "superseded" in policy["supersedes"]
    assert summary["train_config"]["monitor"] == "inner_val_pcc"


def test_a_failed_original_run_completes_and_is_marked_not_consumable(tmp_path):
    """Noise images: nothing to learn, so the correlation lands far below the
    floor -- the run still completes, still writes its checkpoint, and records
    the verdict downstream code must obey."""
    root = make_scut_root(tmp_path / "scut", noise_seed=99)
    result, run_dir = run(tmp_path, root)

    summary = result.summary
    assert summary["test"]["pcc"] < pretrain.FAILURE_FLOOR
    assert summary["band_check"]["checkpoint_consumable"] is False
    assert "PRETRAINING FAILED" in summary["band_check"]["note"]
    assert (run_dir / "pretrained.npz").exists(), (
        "a failed run's checkpoint is evidence and must still exist; "
        "consumability is the recorded verdict, not the file's absence"
    )


def test_masked_source_aligns_rows_by_stem_not_position(tmp_path):
    """The artifact's faces.json is deliberately SHUFFLED relative to sorted
    stem order. A positional read would train every face against another
    face's label and score ~0; stem alignment scores high. This is the
    row-order lesson with teeth."""
    root = make_scut_root(tmp_path / "scut")
    shuffled = [TRAIN_ROWS[i] for i in (5, 0, 7, 2, 6, 1, 4, 3)] + [
        TEST_ROWS[i] for i in (2, 0, 3, 1)
    ]
    masked = make_masked_artifact(tmp_path / "masked_v1", shuffled)

    for source in ("masked_g1", "masked_g2"):
        result, _ = run(
            tmp_path, root, subdir=f"run_{source}", source=source, masked_dir=masked
        )
        summary = result.summary
        assert summary["test"]["pcc"] > 0.9, source
        assert summary["band_check"]["applies"] is False
        assert summary["band_check"]["checkpoint_consumable"] is None
        assert summary["band_check"]["compare_against"] == "stub original"


# --------------------------------------------------------------------------
# refusals: the run must stop rather than train on the wrong thing
# --------------------------------------------------------------------------


def test_declared_split_counts_are_asserted(tmp_path):
    root = make_scut_root(tmp_path / "scut")
    with pytest.raises(PretrainError, match="expect_train=9"):
        run(tmp_path, root, expect_train=9)


def test_an_overlapping_split_is_refused(tmp_path):
    root = tmp_path / "scut"
    write_split(root, TRAIN_ROWS, TEST_ROWS + [TRAIN_ROWS[0]])
    write_images(root, TRAIN_ROWS + TEST_ROWS)
    with pytest.raises(PretrainError, match="BOTH sides"):
        run(tmp_path, root, expect_test=len(TEST_ROWS) + 1)


def test_a_missing_image_is_refused_not_skipped(tmp_path):
    root = make_scut_root(tmp_path / "scut")
    from cleft.scut.dataset import IMAGE_DIR

    (root / IMAGE_DIR / "AM1.jpg").unlink()
    with pytest.raises(PretrainError, match="AM1"):
        run(tmp_path, root)


def test_a_masked_artifact_missing_a_stem_is_refused(tmp_path):
    root = make_scut_root(tmp_path / "scut")
    masked = make_masked_artifact(
        tmp_path / "masked_v1", TRAIN_ROWS[1:] + TEST_ROWS
    )
    with pytest.raises(PretrainError, match="not in the masked artifact"):
        run(tmp_path, root, source="masked_g1", masked_dir=masked)


def test_a_duplicate_stem_in_faces_json_is_refused(tmp_path):
    root = make_scut_root(tmp_path / "scut")
    masked = make_masked_artifact(
        tmp_path / "masked_v1", TRAIN_ROWS + TEST_ROWS + [TRAIN_ROWS[0]]
    )
    with pytest.raises(PretrainError, match="twice"):
        run(tmp_path, root, source="masked_g1", masked_dir=masked)


def test_a_masked_array_shorter_than_its_index_is_refused(tmp_path):
    root = make_scut_root(tmp_path / "scut")
    masked = make_masked_artifact(tmp_path / "masked_v1", TRAIN_ROWS + TEST_ROWS)
    array = np.load(masked / "masked_g1.npy")
    np.save(masked / "masked_g1.npy", array[:-1])
    with pytest.raises(PretrainError, match="does not describe"):
        run(tmp_path, root, source="masked_g1", masked_dir=masked)


def test_a_partially_frozen_model_is_refused(tmp_path):
    """Pretraining is FULL fine-tuning [DECIDED]; a model reporting anything
    frozen is running a different arm than the config declares."""

    class PartiallyFrozen(StubPretrainModel):
        def reset(self, train_features, train_labels):
            super().reset(train_features, train_labels)
            self.parameter_report["trainable_parameters"] -= 1

    root = make_scut_root(tmp_path / "scut")
    with pytest.raises(PretrainError, match="FULL fine-tuning"):
        run(tmp_path, root, model_factory=PartiallyFrozen)


def test_epoch0_miscalibration_is_caught_by_gate_3(tmp_path):
    """The BatchNorm-shaped defect: a head far from the training mean at epoch
    0 must stop the run before any optimiser step -- the frozen gate, applied
    to the new loop."""

    class Miscalibrated(StubPretrainModel):
        def reset(self, train_features, train_labels):
            super().reset(train_features, train_labels)
            self._bias += 1.5

    root = make_scut_root(tmp_path / "scut")
    with pytest.raises(HarnessError, match="epoch-0"):
        run(tmp_path, root, model_factory=Miscalibrated)


def test_band_check_semantics():
    passed = pretrain.band_check("original", "vit_b16", 0.90)
    assert passed["applies"] and passed["checkpoint_consumable"] is True

    failed = pretrain.band_check("original", "vit_b16", 0.50)
    assert failed["checkpoint_consumable"] is False
    assert "PRETRAINING FAILED" in failed["note"]

    # A NaN correlation is a failed run, not a lucky pass.
    assert pretrain.band_check("original", "vit_b16", float("nan"))[
        "checkpoint_consumable"
    ] is False

    masked = pretrain.band_check("masked_g2", "srgnn", 0.70)
    assert masked["applies"] is False
    assert masked["checkpoint_consumable"] is None
    assert masked["compare_against"] == "srgnn original"
    assert masked["decided_at_review"] is True

    with pytest.raises(PretrainError):
        pretrain.band_check("bogus", "vit_b16", 0.9)


# --------------------------------------------------------------------------
# the region-scheme axis [DECIDED 2026-07-31]
# --------------------------------------------------------------------------


def test_the_scheme_null_and_geometry_findings_are_recorded_with_their_scope():
    """[MEASURED 2026-07-31] The scheme axis nulls at pretraining (ranges
    0.0016-0.0154 inside the 0.0137 band); geometry separates for ViT alone
    (masked-G1 0.7893 vs G2 0.8306). The records must carry their scope --
    pretraining, not cleft fine-tuning -- and the mechanism must stay tagged
    [REASONED], because promoting it without an arm is the R2 error."""
    scheme = pretrain.SCHEME_AXIS_AT_PRETRAINING
    assert scheme["range_max"] > scheme["seed_band"] > scheme["range_min"]
    assert "null at pretraining" in scheme["verdict"]
    assert "PRETRAINING only" in scheme["caveat"]
    assert "Stage E" in scheme["caveat"], (
        "the record must keep the cleft-time comparison alive, not close it"
    )
    assert "pasted" in scheme["environment"]

    geometry = pretrain.GEOMETRY_EFFECT_AT_PRETRAINING
    assert geometry["vit_b16_masked_g1"] == pytest.approx(0.7893)
    assert geometry["vit_b16_masked_g2"] == pytest.approx(0.8306)
    assert geometry["gap"] == pytest.approx(
        geometry["vit_b16_masked_g2"] - geometry["vit_b16_masked_g1"], abs=5e-4
    ), "the gap must be the difference of its own figures"
    assert "[REASONED]" in geometry["mechanism"]
    assert "unverified" in geometry["mechanism"]


def test_a_scheme_that_cannot_act_on_its_backbone_is_refused():
    """A transformer given "grid" would train identically and record a scheme
    it never used; a graph backbone given "none" would run its native regions
    under a config that does not say so. Both are the knob-that-cannot-act
    defect, refused by name."""
    pretrain.check_scheme_for_backbone("vit_b16", "none")
    pretrain.check_scheme_for_backbone("swin_b", "none")
    pretrain.check_scheme_for_backbone("stub", "none")
    for scheme in ("native", "grid", "anatomy", "random"):
        pretrain.check_scheme_for_backbone("srgnn", scheme)
        pretrain.check_scheme_for_backbone("agnet", scheme)

    with pytest.raises(PretrainError, match="no region structure"):
        pretrain.check_scheme_for_backbone("vit_b16", "grid")
    with pytest.raises(PretrainError, match="no region structure"):
        pretrain.check_scheme_for_backbone("stub", "anatomy")
    with pytest.raises(PretrainError, match="needs a region scheme"):
        pretrain.check_scheme_for_backbone("srgnn", "none")
    with pytest.raises(PretrainError, match="unknown region_scheme"):
        pretrain.check_scheme_for_backbone("srgnn", "bogus")


def test_every_generated_scheme_yields_the_27_patch_set_at_both_geometries():
    """Phase 2's generators through the registry: 27 patches each, at G1 and
    G2, from one full-frame content box."""
    content = np.array([[0, 0, 224, 224]], dtype=np.int64)
    for scheme in pretrain.GENERATED_SCHEMES:
        for geometry in ("g1", "g2"):
            boxes = pretrain.scheme_frame_boxes(scheme, geometry, content)
            assert boxes.shape == (1, 27, 4), (scheme, geometry, boxes.shape)


def test_scheme_boxes_move_with_each_faces_content_box():
    """The property assert_varies_with_aspect_ratio guards in Phase 2, here:
    two faces with different content boxes must receive different pixel boxes
    for the same scheme, and every box must stay inside its own content."""
    content = np.array([[32, 0, 160, 224], [12, 0, 200, 224]], dtype=np.int64)
    boxes = pretrain.scheme_frame_boxes("grid", "g2", content)
    assert boxes.shape == (2, 27, 4)
    assert not np.array_equal(boxes[0], boxes[1]), (
        "different content boxes produced identical pixel boxes: the scheme "
        "is placed in padded-square coordinates, the Phase 2 defect"
    )
    for face, (x0, _, cw, _) in zip(boxes, content):
        assert (face[:, 0] + 1 >= x0).all()
        assert (face[:, 0] + face[:, 2] <= x0 + cw + 1).all()


def test_boxes_convert_to_each_backbones_convention():
    frame = np.array([[[0, 0, 224, 224], [56, 56, 112, 112]]], dtype=np.float32)

    srgnn = pretrain.boxes_for_model("srgnn", frame)
    assert np.allclose(srgnn[0, 0], [0, 0, 42, 42])
    assert np.allclose(srgnn[0, 1], [10.5, 10.5, 21, 21])

    agnet = pretrain.boxes_for_model("agnet", frame)
    assert np.allclose(agnet[0, 0], [0, 0, 1, 1])
    assert np.allclose(agnet[0, 1], [0.25, 0.25, 0.75, 0.75])

    with pytest.raises(PretrainError, match="no box convention"):
        pretrain.boxes_for_model("vit_b16", frame)


class BoxRecordingStub(StubPretrainModel):
    """Records every boxes argument the loop hands over."""

    def __init__(self):
        super().__init__()
        self.train_boxes: list = []
        self.predict_boxes: list = []

    def train_batch(self, features, labels, boxes=None):
        self.train_boxes.append(boxes)
        return super().train_batch(features, labels, boxes)

    def predict(self, features, boxes=None):
        self.predict_boxes.append(boxes)
        return super().predict(features, boxes)


def test_the_loop_threads_scheme_boxes_to_the_model(tmp_path):
    """THE wiring test: a graph backbone with a generated scheme must receive
    (B, R, 4) boxes in its own convention on every train batch and every
    predict -- computed per face through the artifact's recorded content
    boxes, which the fixture varies."""
    root = make_scut_root(tmp_path / "scut")
    masked = make_masked_artifact(tmp_path / "masked_v1", TRAIN_ROWS + TEST_ROWS)

    recorder = BoxRecordingStub()
    result, _ = run(
        tmp_path, root,
        source="masked_g2", masked_dir=masked, backbone="srgnn",
        region_scheme="grid", model_factory=lambda: recorder,
    )

    assert result.summary["region_scheme"] == "grid"
    assert result.summary["n_regions"] == 27
    assert result.summary["band_check"]["compare_against"] == "srgnn grid original"

    assert recorder.train_boxes and recorder.predict_boxes
    for boxes in recorder.train_boxes + recorder.predict_boxes:
        assert boxes is not None, "a generated scheme reached the model boxless"
        assert boxes.ndim == 3 and boxes.shape[2] == 4
        # srgnn convention: 42x42 map pixels.
        assert boxes.max() <= 42.0 + 1e-6

    # Per-face placement: the fixture varies content boxes, so a full-train
    # predict must carry non-identical rows.
    widest = max(recorder.predict_boxes, key=lambda b: b.shape[0])
    assert not np.array_equal(widest[0], widest[1])


def test_an_original_run_with_a_generated_scheme_also_threads_boxes(tmp_path):
    """Original source: content boxes come off the frozen staging directly,
    and the full-square (G2) parameterisation applies [DECIDED]."""
    root = make_scut_root(tmp_path / "scut")
    recorder = BoxRecordingStub()
    result, _ = run(
        tmp_path, root, backbone="agnet", region_scheme="anatomy",
        model_factory=lambda: recorder,
    )
    assert result.summary["n_regions"] == 27
    for boxes in recorder.train_boxes:
        assert boxes is not None
        # agnet convention: normalised (x1, y1, x2, y2).
        assert boxes.max() <= 1.0 + 1e-6
        assert (boxes[..., 2] >= boxes[..., 0]).all()


def test_an_artifact_without_content_boxes_refuses_generated_schemes(tmp_path):
    """An artifact predating the recorded geometry cannot feed a generated
    scheme; the refusal names the gap rather than placing every face's boxes
    at a guessed position."""
    root = make_scut_root(tmp_path / "scut")
    masked = make_masked_artifact(
        tmp_path / "masked_v1", TRAIN_ROWS + TEST_ROWS, content_boxes=False
    )
    with pytest.raises(PretrainError, match="content_box"):
        run(
            tmp_path, root,
            source="masked_g1", masked_dir=masked, backbone="srgnn",
            region_scheme="grid", model_factory=BoxRecordingStub,
        )


# --------------------------------------------------------------------------
# resume: the Phase 6 entry gate, wired into the real loop
# --------------------------------------------------------------------------


def test_a_resumed_run_matches_an_uninterrupted_one(tmp_path):
    """THE wiring test. Killed after three epochs, resumed, and the journal is
    byte-identical while the deliverable is state-identical (npz bytes embed
    zip timestamps, so the CONTENT is compared through the verified loader).
    """
    root = make_scut_root(tmp_path / "scut")
    config = small_config(epochs=8)

    straight, straight_dir = run(tmp_path, root, subdir="straight", config=config)

    paused, resumed_dir = run(
        tmp_path, root, subdir="resumed", config=config, stop_after_epochs=3
    )
    assert paused.paused
    assert (resumed_dir / ckpt.CHECKPOINT_NAME).exists()
    resumed, _ = run(tmp_path, root, subdir="resumed", config=config)

    assert not resumed.paused
    assert resumed.summary["resumed"] is True

    assert (resumed_dir / "curves.csv").read_bytes() == (
        straight_dir / "curves.csv"
    ).read_bytes()

    a = ckpt.load(straight_dir / "pretrained.npz")
    b = ckpt.load(resumed_dir / "pretrained.npz")
    assert set(a.arrays) == set(b.arrays)
    for key in a.arrays:
        assert np.array_equal(a.arrays[key], b.arrays[key]), key
    assert a.extra["test_pcc"] == b.extra["test_pcc"]

    exclude = {"resumed"}
    assert {k: v for k, v in straight.summary.items() if k not in exclude} == {
        k: v for k, v in resumed.summary.items() if k not in exclude
    }
    assert straight.test_table == resumed.test_table
    assert not (resumed_dir / ckpt.CHECKPOINT_NAME).exists()


def test_a_resume_that_loses_optimizer_state_diverges(tmp_path):
    """The counterfactual that gives the test above teeth: restore everything
    EXCEPT the optimizer's momentum and the continued run must differ. If this
    passes with identical journals, the byte-identity test can no longer fail
    and proves nothing."""

    class ForgetfulResume(StubPretrainModel):
        def load_state_arrays(self, arrays):
            super().load_state_arrays(
                {
                    key: value
                    for key, value in arrays.items()
                    if not key.startswith(pretrain.OPTIMIZER_PREFIX)
                }
            )

    root = make_scut_root(tmp_path / "scut")
    config = small_config(epochs=8)

    _, straight_dir = run(tmp_path, root, subdir="straight", config=config)
    paused, forgetful_dir = run(
        tmp_path, root, subdir="forgetful", config=config, stop_after_epochs=3
    )
    assert paused.paused
    run(
        tmp_path, root, subdir="forgetful", config=config,
        model_factory=ForgetfulResume,
    )

    assert (forgetful_dir / "curves.csv").read_bytes() != (
        straight_dir / "curves.csv"
    ).read_bytes()


def test_resume_reproduces_rows_written_after_the_checkpoint(tmp_path):
    """checkpoint_every=2 and a pause after epoch 3: the journal holds one row
    the checkpoint does not know about. The resume must truncate it and
    re-produce it identically, not append a duplicate epoch."""
    root = make_scut_root(tmp_path / "scut")
    config = small_config(epochs=6, checkpoint_every=2)

    _, straight_dir = run(tmp_path, root, subdir="straight", config=config)
    paused, resumed_dir = run(
        tmp_path, root, subdir="resumed", config=config, stop_after_epochs=3
    )
    assert paused.paused
    assert ckpt.load(resumed_dir / ckpt.CHECKPOINT_NAME).step == 2
    lines = (resumed_dir / "curves.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) - 1 == 3, "the pause left a post-checkpoint row behind"

    run(tmp_path, root, subdir="resumed", config=config)
    assert (resumed_dir / "curves.csv").read_bytes() == (
        straight_dir / "curves.csv"
    ).read_bytes()


def test_a_journal_shorter_than_its_checkpoint_refuses_resume(tmp_path):
    """Rows the checkpoint believes are durable are missing: resuming into the
    gap would produce a journal whose early epochs never happened."""
    root = make_scut_root(tmp_path / "scut")
    config = small_config(epochs=8, checkpoint_every=2)
    paused, run_dir = run(
        tmp_path, root, subdir="short", config=config, stop_after_epochs=4
    )
    assert paused.paused

    lines = (run_dir / "curves.csv").read_text(encoding="utf-8").splitlines()
    (run_dir / "curves.csv").write_text(
        "".join(line + "\n" for line in lines[:3]), encoding="utf-8"
    )
    with pytest.raises(PretrainError, match="short of its checkpoint"):
        run(tmp_path, root, subdir="short", config=config)


def test_a_checkpoint_from_a_different_arm_refuses_resume(tmp_path):
    """A resumed directory carrying another arm's checkpoint is not a resume;
    the fingerprint must catch it before any state is restored."""
    root = make_scut_root(tmp_path / "scut")
    paused, _ = run(
        tmp_path, root, subdir="arm", config=small_config(epochs=8),
        stop_after_epochs=3,
    )
    assert paused.paused
    with pytest.raises(PretrainError, match="fingerprint"):
        run(tmp_path, root, subdir="arm", config=small_config(epochs=8, seed=8))


# --------------------------------------------------------------------------
# through main(): the task handler, the schema, and the run record
# --------------------------------------------------------------------------


def pretrain_task(**overrides) -> dict:
    task = {
        "kind": "pretrain",
        "scut_root": "scut_root",
        "backbone": "stub",
        "source": "original",
        "expect_train": len(TRAIN_ROWS),
        "expect_test": len(TEST_ROWS),
        "epochs": 6,
        "inner_val_frac": 0.25,
        "monitor": "inner_val_pcc",
        # False in tests: the stub never imports torch, and the flag is a
        # process-global torch setting no test should flip.
        "deterministic": False,
        "region_scheme": "none",
        "learning_rate": 0.0001,
        "weight_decay": 0.01,
        "batch_size": 4,
        "checkpoint_every": 1,
    }
    task.update(overrides)
    return task


def declare(name: str, path: Path) -> dict:
    from cleft.provenance import hash_dir

    return {
        "name": name,
        "path": str(path),
        "rollup_sha256": hash_dir(path)["rollup"],
    }


def test_the_pretrain_task_runs_through_main(tmp_path):
    from cleft.provenance.hashing import hash_file
    from cleft.run import main

    root = make_scut_root(tmp_path / "scut")
    config = builders.write_config(
        tmp_path / "p6.yaml",
        phase="p6",
        inputs=[declare("scut_root", root)],
        task=pretrain_task(),
    )
    run_dir = main(["--config", str(config), "--out", str(tmp_path / "runs")])

    metrics_payload = json.loads((run_dir / "metrics.json").read_text("utf-8"))
    assert metrics_payload["band_check"]["applies"] is True
    assert metrics_payload["pretrained_checkpoint"]["sha256"] == hash_file(
        run_dir / "pretrained.npz"
    )
    assert not (run_dir / ckpt.CHECKPOINT_NAME).exists()

    scores = (run_dir / "scut_test_scores.csv").read_text("utf-8").splitlines()
    assert scores[0] == "stem,truth,pred"
    assert len(scores) - 1 == len(TEST_ROWS)
    assert [line.split(",")[0] for line in scores[1:]] == sorted(
        stem for stem, _ in TEST_ROWS
    )


def test_the_masked_task_runs_through_main_and_requires_its_input(tmp_path):
    from cleft.run import main

    root = make_scut_root(tmp_path / "scut")
    masked = make_masked_artifact(tmp_path / "masked_v1", TRAIN_ROWS + TEST_ROWS)

    config = builders.write_config(
        tmp_path / "p6_masked.yaml",
        phase="p6",
        inputs=[declare("scut_root", root), declare("masked_scut", masked)],
        task=pretrain_task(source="masked_g2"),
    )
    run_dir = main(["--config", str(config), "--out", str(tmp_path / "runs")])
    metrics_payload = json.loads((run_dir / "metrics.json").read_text("utf-8"))
    assert metrics_payload["source"] == "masked_g2"
    assert metrics_payload["band_check"]["compare_against"] == "stub original"

    # Masked without the artifact: refused by name.
    config = builders.write_config(
        tmp_path / "p6_missing.yaml",
        phase="p6",
        inputs=[declare("scut_root", root)],
        task=pretrain_task(source="masked_g1"),
    )
    with pytest.raises(ValueError, match="masked_scut"):
        main(["--config", str(config), "--out", str(tmp_path / "runs2")])


def test_an_original_run_refuses_an_unread_masked_input(tmp_path):
    """An input the run never reads would still be hash-verified and recorded
    in inputs.json as if it fed the run -- a false provenance claim."""
    from cleft.run import main

    root = make_scut_root(tmp_path / "scut")
    masked = make_masked_artifact(tmp_path / "masked_v1", TRAIN_ROWS + TEST_ROWS)
    config = builders.write_config(
        tmp_path / "p6_extra.yaml",
        phase="p6",
        inputs=[declare("scut_root", root), declare("masked_scut", masked)],
        task=pretrain_task(source="original"),
    )
    with pytest.raises(ValueError, match="does not read"):
        main(["--config", str(config), "--out", str(tmp_path / "runs")])


# --------------------------------------------------------------------------
# the thirty shipped configs: two lattices differing in exactly one field
# --------------------------------------------------------------------------

TRANSFORMERS = ("vit_b16", "swin_b")
GRAPH_BACKBONES = ("srgnn", "agnet")
SOURCES = ("original", "masked_g1", "masked_g2")
SCHEMES = ("native", "grid", "anatomy", "random")


def test_the_thirty_pretraining_configs_form_one_field_lattices(
    repo_root, monkeypatch
):
    """[DECIDED 2026-07-31] Transformers: 2 backbones x 3 sources at
    region_scheme none. Graph backbones: 2 x 4 schemes x 3 sources -- the
    scheme became a pretraining axis because the graph layers must see the
    SAME node structure in pretraining and cleft fine-tuning. Within each
    block every pair of lattice neighbours differs in exactly one field,
    asserted the strong way: strip the three axis fields and all thirty task
    dicts must be IDENTICAL. The scheme axis exists exactly where it can act
    -- region_scheme is "none" iff the backbone has no region structure."""
    from cleft.config import load_config

    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel/for/lattice")

    paths = sorted((repo_root / "configs").glob("p6_pretrain_*.yaml"))
    assert len(paths) == 30, [p.name for p in paths]

    configs: dict[tuple[str, str, str], dict] = {}
    for path in paths:
        cfg = load_config(path)
        task = cfg["task"]
        key = (task["backbone"], task["region_scheme"], task["source"])
        assert key not in configs, f"duplicate arm {key}"
        configs[key] = cfg

        # The filename must say what the config does: transformers omit the
        # scheme token (they have none), graph configs carry it.
        backbone, scheme, source = key
        expected_name = (
            f"p6_pretrain_{backbone}_{source}.yaml"
            if scheme == "none"
            else f"p6_pretrain_{backbone}_{scheme}_{source}.yaml"
        )
        assert path.name == expected_name, (path.name, expected_name)

    expected_keys = {(b, "none", s) for b in TRANSFORMERS for s in SOURCES} | {
        (b, sch, s)
        for b in GRAPH_BACKBONES
        for sch in SCHEMES
        for s in SOURCES
    }
    assert set(configs) == expected_keys

    # The scheme axis exists exactly where it can act.
    for backbone, scheme, _ in configs:
        assert (scheme == "none") == (backbone in TRANSFORMERS), (backbone, scheme)

    # Strip the axes; everything remaining must be identical across all 30.
    reference_key = ("vit_b16", "none", "original")
    axes = {"backbone", "source", "region_scheme"}
    reference = {
        k: v for k, v in configs[reference_key]["task"].items() if k not in axes
    }
    for key, cfg in configs.items():
        stripped = {k: v for k, v in cfg["task"].items() if k not in axes}
        assert stripped == reference, (
            f"{key} differs from {reference_key} beyond the axis fields: "
            f"{ {k for k in stripped if stripped[k] != reference.get(k)} }"
        )
        assert cfg["phase"] == "p6"
        assert cfg["tier"] == "keeper", "the pretraining runs are citable"
        assert cfg["seed"] == configs[reference_key]["seed"]

    # The official split, declared: the runtime invariant the loop enforces.
    assert reference["expect_train"] == 3300
    assert reference["expect_test"] == 2199

    # [DECIDED 2026-07-31, from image measurements] All thirty run FLAG-OFF:
    # vit_b16/swin_b/srgnn are bitwise without the flag in the image, and the
    # flag cannot fix agnet there -- the deterministic roi_align substitution
    # needs a C++ compiler the image lacks (InvalidCxxCompiler, measured).
    # AG-Net's ~2.3e-05 bound is recorded in models/agnet.py instead. A
    # config flipped true is a deliberate change, made visible by this test.
    assert reference["deterministic"] is False

    # [DECIDED 2026-07-31] The fixed budget: exactly 30 epochs, selection by
    # inner_val_pcc, and NO patience field anywhere -- patience terminated on
    # noise in a flat region and made run length track luck rather than
    # scheme. The eight pre-policy runs are superseded; gate 4's rejection of
    # fixed budgets was a 152-sample cleft behaviour and does not transfer.
    assert reference["epochs"] == 30
    assert reference["monitor"] == "inner_val_pcc"
    assert "patience" not in reference


def test_the_scheme_axis_carries_the_g2_comparison_note(repo_root):
    """[MEASURED, PLAN §4.4] At G1 the schemes carry different background
    exposure, so a G1 scheme difference is partly a background difference --
    the scheme axis is compared at G2. the instruction: recorded in
    the configs rather than remembered. Every graph config must carry it."""
    for path in sorted((repo_root / "configs").glob("p6_pretrain_*.yaml")):
        if not any(b in path.name for b in GRAPH_BACKBONES):
            continue
        text = path.read_text(encoding="utf-8")
        # Comment prose wraps, so phrases are matched on the FLATTENED text --
        # a check keyed to one exact line-broken string is the R7 instance 4
        # defect, and the first version of this test was one.
        flat = " ".join(
            line.lstrip("#").strip() for line in text.splitlines()
        )
        assert "SCHEME COMPARISONS ARE READ AT G2" in flat, path.name
        assert "background" in flat, path.name
        assert "SAME scheme must be used in pretraining and cleft fine-tuning" in flat, (
            f"{path.name} must record that pretraining and cleft fine-tuning "
            "use the same scheme"
        )


def test_the_thirty_configs_declare_exactly_the_inputs_their_source_reads(
    repo_root, monkeypatch
):
    """Original: scut_root alone. Masked: scut_root plus masked_scut, all
    twenty pointing at the same artifact with the same declared hash -- the
    rollup recorded by the artifact build itself (p5-masked-scut-3)."""
    from cleft.config import load_config

    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel/for/lattice")

    scut_declarations = set()
    masked_declarations = set()
    n_masked = 0
    for path in sorted((repo_root / "configs").glob("p6_pretrain_*.yaml")):
        cfg = load_config(path)
        by_name = {entry["name"]: entry for entry in cfg["inputs"]}
        source = cfg["task"]["source"]
        expected = {"scut_root"} if source == "original" else {
            "scut_root", pretrain.MASKED_INPUT_NAME,
        }
        assert set(by_name) == expected, path.name

        entry = by_name["scut_root"]
        scut_declarations.add(
            (entry.get("path_declared_as", entry["path"]), entry["rollup_sha256"])
        )
        if source != "original":
            n_masked += 1
            masked = by_name[pretrain.MASKED_INPUT_NAME]
            masked_declarations.add((masked["path"], masked["rollup_sha256"]))

    assert n_masked == 20
    assert len(scut_declarations) == 1, scut_declarations
    assert len(masked_declarations) == 1, masked_declarations
    declared_path, declared_hash = next(iter(masked_declarations))
    assert declared_path.endswith("data/scut/masked_v1")
    assert declared_hash != "0" * 64, (
        "the masked artifact exists and its measured rollup must be declared"
    )


# --------------------------------------------------------------------------
# the epoch-level observation hook (the SCUT animation's capture seam)
# --------------------------------------------------------------------------


def test_on_epoch_hook_observes_init_and_every_epoch(tmp_path):
    """Fresh start: the hook sees epoch 0 (the untrained state) and every
    trained epoch, in order, with the live model each time."""
    root = make_scut_root(tmp_path / "scut")
    seen = []
    result, _ = run(
        tmp_path, root,
        on_epoch=lambda epoch, model: seen.append((epoch, model is not None)),
    )
    assert not result.paused
    assert [epoch for epoch, _ in seen] == list(range(0, 13))
    assert all(has_model for _, has_model in seen)


def test_on_epoch_hook_does_not_replay_observed_states_on_resume(tmp_path):
    """A killed-and-resumed run re-observes NOTHING: epoch 0 is fresh-start
    only and the resumed invocation continues from the checkpointed epoch --
    frames already derived from earlier states stay derived once."""
    root = make_scut_root(tmp_path / "scut")
    first = []
    paused, _ = run(
        tmp_path, root, subdir="resumed", stop_after_epochs=5,
        on_epoch=lambda epoch, model: first.append(epoch),
    )
    assert paused.paused
    assert first == [0, 1, 2, 3, 4, 5]
    second = []
    finished, _ = run(
        tmp_path, root, subdir="resumed",
        on_epoch=lambda epoch, model: second.append(epoch),
    )
    assert not finished.paused
    assert second == list(range(6, 13))
