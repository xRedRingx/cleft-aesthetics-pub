"""Phase 7C exit criterion 6: PLAN §4.3's condition 1, over stored vectors.

**This file exists because the phase reported nine verdicts from ONE of the
claim criterion's two conditions.** ``NOT_A_SEARCH["carried_over"]`` promised
the paired BCa in writing and exit criterion 6 required it; no interval
appeared anywhere in the phase. Everything here is re-analysis -- the task
fits nothing, and every vector it reads already exists.

The tests that matter most are not the arithmetic ones. They are:

* ``test_verification_refuses_vectors_that_do_not_reproduce_the_round`` --
  three rounds of these arms are on disk under the same config stems and two
  are VOID. Void vectors would load, pair, and produce plausible intervals.
* ``test_a_null_cannot_be_rescued_by_the_paired_condition`` -- the conditions
  are ANDed, so condition 1 can only ever withdraw claims. A test that let a
  null become claimable would be encoding the exact move
  ``ROUND_2_IS_ROUND_1_TRUNCATED`` was written about.
* ``test_truth_disagreement_between_arms_is_refused`` -- the arms must have
  been fitted against the same labels or the pairing is meaningless, and this
  project runs the same cell under more than one label formulation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cleft import phase7c
from cleft.cluster_csv import write_predictions
from cleft.eval.metrics import pcc
from cleft.train.phase3 import combined_claimable_delta, seed_variance

from fixtures import builders

from test_extract import declare

N_PATIENTS = 40
SEEDS = list(phase7c.SEEDS)

#: Enough replicates for a stable interval, few enough that 45 of them are
#: quick. The task's default is 10000; nothing here asserts an interval's
#: exact width, only which side of zero it falls.
N_BOOT = 200

#: Per-arm noise scale. Larger scale means a worse arm, so arm 0 is the best
#: and the rest degrade -- the shape of the real result, which matters because
#: the direction conventions are what these tests check.
ARM_NOISE = {
    "0_identity": 0.35,
    "1_photometric": 0.55,
    "2_geometric": 0.85,
    "3_geometric_rotation": 0.95,
    "4_region_photometric": 0.58,
    "5_region_full": 0.86,
    "6_whole_image_full": 0.75,
}


def _build_vectors(root: Path, *, seeds=None, noise=None) -> dict:
    """Seven arms x five seeds of prediction CSVs, in the real layout.

    Written with ``cluster_csv.write_predictions`` -- the same writer the
    training task uses -- so the tier marker, the column order and the six
    decimals are the real ones rather than a test's idea of them.
    """
    seeds = list(seeds or SEEDS)
    noise = noise or ARM_NOISE
    rng = np.random.default_rng(20260803)
    patient_ids = list(range(1, N_PATIENTS + 1))
    truth = rng.normal(size=N_PATIENTS) * 0.8 + 3.0

    written: dict[str, dict] = {}
    for arm, scale in noise.items():
        directory = root / arm
        directory.mkdir(parents=True, exist_ok=True)
        by_seed = {}
        for seed in seeds:
            vector = truth + rng.normal(scale=scale, size=N_PATIENTS)
            path = directory / f"seed_{seed}__predictions.csv"
            write_predictions(
                path,
                [
                    (pid, t, v, pid % 6)
                    for pid, t, v in zip(patient_ids, truth, vector)
                ],
            )
            by_seed[seed] = (path, vector)
        written[arm] = by_seed
    return {"truth": truth, "patient_ids": patient_ids, "arms": written}


def _inputs_for(built: dict, *, seeds=None) -> list[dict]:
    seeds = list(seeds or SEEDS)
    entries = []
    for arm, by_seed in built["arms"].items():
        for seed in seeds:
            entries.append(declare(
                f"{phase7c.OOF_INPUT_PREFIX}{arm}"
                f"{phase7c.OOF_SEED_SEPARATOR}{seed}",
                by_seed[seed][0],
            ))
    return entries


def _observed_round(built: dict, *, seeds=None) -> dict:
    """A ``STAGE_7C_RESULTS`` round matching what the fixture actually scores.

    The recorded round is patched to the fixture's own numbers rather than the
    verification being disabled, for the reason ``test_phase7b_task`` gives:
    disabling it would leave the one guard against pairing the wrong run
    untested by the only test that reaches it.
    """
    seeds = list(seeds or SEEDS)
    truth = built["truth"]
    arms = {}
    for arm, by_seed in built["arms"].items():
        pccs = [float(pcc(truth, by_seed[seed][1])) for seed in seeds]
        band = seed_variance(pccs)
        arms[arm] = {"pcc": band["mean"], "sd": band["sd"]}

    def verdict(a: str, b: str) -> dict:
        delta = arms[b]["pcc"] - arms[a]["pcc"]
        threshold = combined_claimable_delta(
            arms[a]["sd"], len(seeds), arms[b]["sd"], len(seeds)
        )["arm_means_95"]
        return {
            "delta": round(delta, 4),
            "threshold": round(threshold, 4),
            "claimable": bool(abs(delta) > threshold),
        }

    return {
        "epoch_policy": "fixture",
        "arms": arms,
        "against_identity": {
            arm: verdict(phase7c.IDENTITY_ARM, arm)
            for arm in arms if arm != phase7c.IDENTITY_ARM
        },
        "comparisons": {
            pair["question"]: verdict(
                phase7c.result_key(pair["a"]["name"]),
                phase7c.result_key(pair["b"]["name"]),
            )
            for pair in phase7c.comparisons()
        },
    }


@pytest.fixture
def built(tmp_path):
    return _build_vectors(tmp_path / "runs")


@pytest.fixture
def patched(built, monkeypatch):
    monkeypatch.setitem(
        phase7c.STAGE_7C_RESULTS["rounds"], "selected30", _observed_round(built)
    )
    return built


def _load(built, **kwargs):
    declared = {entry["name"]: entry for entry in _inputs_for(built, **kwargs)}
    return phase7c.load_oof_vectors(
        phase7c.oof_paths_from_inputs(declared), **kwargs
    )


# --------------------------------------------------------------------------
# input naming
# --------------------------------------------------------------------------


def test_input_names_parse_into_an_arm_and_a_seed(built):
    paths = phase7c.oof_paths_from_inputs(
        {entry["name"]: entry for entry in _inputs_for(built)}
    )
    assert sorted(paths) == sorted(ARM_NOISE)
    assert all(sorted(by_seed) == sorted(SEEDS) for by_seed in paths.values())


def test_an_arm_name_containing_the_separator_still_parses():
    """``rpartition`` splits on the LAST ``_seed_``, so an arm may contain it.

    Not hypothetical bookkeeping: the arm keys are free text from the arm
    list, and a left-split would silently attribute the vector to an arm that
    does not exist rather than failing.
    """
    paths = phase7c.oof_paths_from_inputs(
        {"oof_9_seed_bank_seed_1337": {"path": "/x.csv"}}
    )
    assert paths == {"9_seed_bank": {1337: "/x.csv"}}


@pytest.mark.parametrize("name", ["oof_nope", "oof__seed_12", "oof_arm_seed_x"])
def test_a_malformed_input_name_is_refused(name):
    with pytest.raises(phase7c.Phase7CError, match="does not name an arm"):
        phase7c.oof_paths_from_inputs({name: {"path": "/x.csv"}})


def test_inputs_that_are_not_vectors_are_ignored(built):
    declared = {entry["name"]: entry for entry in _inputs_for(built)}
    declared["manifest_v1"] = {"path": "/somewhere"}
    assert "manifest_v1" not in phase7c.oof_paths_from_inputs(declared)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def test_vectors_load_and_reproduce_their_own_pccs(built):
    loaded = _load(built)
    assert len(loaded["patient_ids"]) == N_PATIENTS
    assert sorted(loaded["arms"]) == sorted(ARM_NOISE)
    for arm, by_seed in built["arms"].items():
        entry = loaded["arms"][arm]
        expected = [
            float(pcc(built["truth"], by_seed[s][1])) for s in entry["seeds"]
        ]
        assert entry["pccs"] == pytest.approx(expected, abs=1e-5)


def test_the_seed_order_of_the_pccs_is_recorded_not_positional(built):
    """``pccs`` is in SORTED seed order and the configs declare 1337 first, so
    a reader pairing the two by index would mis-attribute four of five."""
    entry = _load(built)["arms"]["0_identity"]
    assert entry["seeds"] == sorted(SEEDS) != SEEDS
    assert len(entry["pccs"]) == len(entry["seeds"])


def test_rows_are_realigned_by_patient_id_not_taken_in_file_order(built, tmp_path):
    """A shuffled file must give the same vector, not a shuffled one.

    ``load_baseline_predictions`` learned this once for five files. This reads
    thirty-five, and a silent misalignment pairs shuffled vectors and still
    produces a plausible interval.
    """
    reference = _load(built)["arms"]["0_identity"]["pccs"]

    path, vector = built["arms"]["0_identity"][SEEDS[0]]
    order = list(range(N_PATIENTS))[::-1]
    write_predictions(
        path,
        [
            (i + 1, built["truth"][i], vector[i], (i + 1) % 6)
            for i in order
        ],
    )
    assert _load(built)["arms"]["0_identity"]["pccs"] == pytest.approx(
        reference, abs=1e-5
    )


def test_truth_disagreement_between_arms_is_refused(built):
    """Different labels means the pairing is meaningless.

    ``STAGE_G_LABEL_FORMULATION`` is why this is live rather than paranoid:
    this project runs the same cell under more than one label formulation, and
    two such runs would differ ONLY in this column.
    """
    path, vector = built["arms"]["2_geometric"][SEEDS[0]]
    write_predictions(
        path,
        [
            (pid, t + 0.5, v, pid % 6)
            for pid, t, v in zip(built["patient_ids"], built["truth"], vector)
        ],
    )
    with pytest.raises(phase7c.Phase7CError, match="TRUTH column"):
        _load(built)


def test_a_different_patient_set_is_refused(built):
    path, vector = built["arms"]["1_photometric"][SEEDS[0]]
    write_predictions(
        path,
        [
            (pid + 1000, t, v, pid % 6)
            for pid, t, v in zip(built["patient_ids"], built["truth"], vector)
        ],
    )
    with pytest.raises(phase7c.Phase7CError, match="different patient set"):
        _load(built)


def test_a_missing_seed_is_refused_by_name(built):
    declared = {entry["name"]: entry for entry in _inputs_for(built)}
    del declared[f"oof_4_region_photometric_seed_{SEEDS[2]}"]
    with pytest.raises(phase7c.Phase7CError, match=r"no vector for seeds"):
        phase7c.load_oof_vectors(phase7c.oof_paths_from_inputs(declared))


def test_no_declared_vectors_says_criterion_6_is_unmet():
    with pytest.raises(phase7c.Phase7CError, match="criterion 6"):
        phase7c.load_oof_vectors({})


# --------------------------------------------------------------------------
# verification against the recorded round -- the void-vector guard
# --------------------------------------------------------------------------


def test_verification_passes_when_the_vectors_are_the_recorded_ones(patched):
    report = phase7c.verify_against_record(
        _load(patched), round_label="selected30"
    )
    assert len(report["arms"]) == len(ARM_NOISE)
    assert all(entry["delta_pcc"] < 1e-4 for entry in report["arms"])


def test_verification_refuses_vectors_that_do_not_reproduce_the_round(patched):
    """**The guard against pairing a VOID round.**

    v1 (the augmenter never ran) and v2 (every arm stopped at epoch 1) left
    prediction files under the same config stems as v3. They would load, they
    would pair, and they would produce entirely plausible intervals. Here one
    arm is replaced by a differently-scored vector, which is exactly what
    declaring the wrong run directory would do.
    """
    path, _ = patched["arms"]["5_region_full"][SEEDS[0]]
    rng = np.random.default_rng(5)
    wrong = patched["truth"] + rng.normal(scale=3.0, size=N_PATIENTS)
    write_predictions(
        path,
        [
            (pid, t, v, pid % 6)
            for pid, t, v in zip(patched["patient_ids"], patched["truth"], wrong)
        ],
    )
    with pytest.raises(phase7c.Phase7CError, match="do not reproduce round"):
        phase7c.verify_against_record(_load(patched), round_label="selected30")


def test_verification_checks_the_sd_and_not_only_the_mean(patched):
    """Two rounds of this phase share 23 of 35 fits, so means are a weak
    discriminator between them and spreads are not."""
    loaded = _load(patched)
    loaded["arms"]["3_geometric_rotation"]["sd"] += 0.05
    with pytest.raises(phase7c.Phase7CError, match="do not reproduce round"):
        phase7c.verify_against_record(loaded, round_label="selected30")


def test_an_unknown_round_is_refused(patched):
    with pytest.raises(phase7c.Phase7CError, match="no round"):
        phase7c.verify_against_record(_load(patched), round_label="round47")


# --------------------------------------------------------------------------
# the matrix
# --------------------------------------------------------------------------


def test_all_nine_comparisons_are_computed(patched):
    matrix = phase7c.paired_matrix(
        _load(patched), round_label="selected30", n_boot=N_BOOT
    )
    assert len(matrix["against_identity"]) == 6
    assert len(matrix["comparisons"]) == 3
    assert matrix["n_comparisons"] == 9
    assert phase7c.IDENTITY_ARM not in matrix["against_identity"]


def test_every_comparison_names_its_direction_in_words(patched):
    """**The sign convention has already been quoted backwards once.**

    ``comparisons()`` puts the PROTECTED arm in ``b`` for both region tests,
    so both deltas are protected-minus-unprotected. Reading 1-vs-4 in the
    opposite order from 5-vs-6 turns agreement into a sign disagreement.
    """
    matrix = phase7c.paired_matrix(
        _load(patched), round_label="selected30", n_boot=N_BOOT
    )
    for group in (matrix["against_identity"], matrix["comparisons"]):
        for entry in group.values():
            assert entry["delta_is"] == f"{entry['b']} minus {entry['a']}"

    region = matrix["comparisons"]["region_awareness"]
    region_photometric = matrix["comparisons"]["region_awareness_photometric"]
    assert region["b"] == "5_region_full"
    assert region_photometric["b"] == "4_region_photometric"
    # Both protected arms are on the same side of the subtraction.
    assert region["delta_is"].startswith("5_region_full minus")
    assert region_photometric["delta_is"].startswith("4_region_photometric minus")


def test_the_delta_sign_matches_the_arms_own_pccs(patched):
    loaded = _load(patched)
    matrix = phase7c.paired_matrix(
        loaded, round_label="selected30", n_boot=N_BOOT
    )
    for group in (matrix["against_identity"], matrix["comparisons"]):
        for entry in group.values():
            recomputed = (
                loaded["arms"][entry["b"]]["mean"]
                - loaded["arms"][entry["a"]]["mean"]
            )
            assert entry["paired"]["mean_delta"] == pytest.approx(
                recomputed, abs=2e-3
            )


def test_a_null_cannot_be_rescued_by_the_paired_condition(patched):
    """**The conditions are ANDed, so condition 1 can only WITHDRAW claims.**

    A comparison that fails the seed threshold must not come out claimable no
    matter how its intervals fall. Letting one through would be promoting a
    result by swapping in a friendlier variance after seeing which way it
    went -- ``ROUND_2_IS_ROUND_1_TRUNCATED``, arrived at from a new direction.
    """
    matrix = phase7c.paired_matrix(
        _load(patched), round_label="selected30", n_boot=N_BOOT
    )
    for group in (matrix["against_identity"], matrix["comparisons"]):
        for name, entry in group.items():
            if not entry["condition_2_exceeds_threshold"]:
                assert not entry["claimable"], (
                    f"{name} failed the seed threshold and was still claimed"
                )


def test_claimable_is_exactly_the_conjunction(patched):
    matrix = phase7c.paired_matrix(
        _load(patched), round_label="selected30", n_boot=N_BOOT
    )
    for group in (matrix["against_identity"], matrix["comparisons"]):
        for entry in group.values():
            assert entry["claimable"] == (
                entry["condition_1_paired_bca"]
                and entry["condition_2_exceeds_threshold"]
            )


def test_underpowered_nulls_are_named_rather_than_called_absent(patched):
    """A null whose intervals exclude zero is a POWER failure, and the record
    has to say so -- ``REGION_TESTS_ARE_NOT_INDEPENDENT`` insisted on that
    distinction in words and this is where it gets evidence."""
    matrix = phase7c.paired_matrix(
        _load(patched), round_label="selected30", n_boot=N_BOOT
    )
    for name in matrix["underpowered_nulls"]:
        entry = (
            matrix["against_identity"].get(name)
            or matrix["comparisons"][name]
        )
        assert entry["condition_1_paired_bca"]
        assert not entry["condition_2_exceeds_threshold"]
        assert not entry["claimable"]


def test_an_arm_missing_from_the_load_is_refused(patched):
    loaded = _load(patched)
    del loaded["arms"][phase7c.IDENTITY_ARM]
    with pytest.raises(phase7c.Phase7CError, match="every comparison is against"):
        phase7c.paired_matrix(loaded, round_label="selected30", n_boot=N_BOOT)


# --------------------------------------------------------------------------
# the ten-seed round
# --------------------------------------------------------------------------


def test_ten_seed_arms_differ_from_the_five_seed_ones_only_in_seeds():
    five = {arm["name"]: arm for arm in phase7c.arms()}
    for arm in phase7c.arms_at(10, only=phase7c.TEN_SEED_ARMS):
        original = five[arm["name"]]
        assert arm["seeds"] == phase7c.SEEDS_10
        assert len(arm["seeds"]) == 10
        for field in phase7c.COMPARED_FIELDS + ("backbone", "init", "geometry",
                                                "label", "index"):
            assert arm[field] == original[field], field


def test_the_first_five_of_ten_are_the_five(patched=None):
    """The ten-seed round re-runs the five, so their predictions must come back
    bit-identical -- a determinism re-verification obtained for free."""
    assert phase7c.SEEDS_10[:5] == phase7c.SEEDS


def test_the_ten_seed_arms_are_the_arms_whose_contrasts_turn_on_n():
    """Arms 0/1 carry photometric-vs-identity; arms 1/4 carry the region
    photometric contrast. Nothing else is raised, because nothing else needs
    it -- PLAN §4.12."""
    assert set(phase7c.TEN_SEED_ARMS) == {0, 1, 4}
    pair = next(
        c for c in phase7c.comparisons()
        if c["question"] == "region_awareness_photometric"
    )
    assert {pair["a"]["index"], pair["b"]["index"]} == {1, 4}


def test_both_rounds_get_ten_seeds():
    """Raising only the round whose nulls flip would promote the sensitivity
    analysis by a second route."""
    assert phase7c.TEN_SEED_ROUND["rounds"] == ["selected30", "matched3"]


def test_arms_at_refuses_more_seeds_than_the_pool_holds():
    with pytest.raises(phase7c.Phase7CError, match="pool holds"):
        phase7c.arms_at(99)


def test_arms_at_refuses_an_index_that_does_not_exist():
    with pytest.raises(phase7c.Phase7CError, match="matched"):
        phase7c.arms_at(10, only=(0, 42))


# --------------------------------------------------------------------------
# which runs ARE the round
# --------------------------------------------------------------------------


def test_every_arm_has_a_v3_job_id():
    assert set(phase7c.V3_JOB_IDS) == {arm["index"] for arm in phase7c.arms()}
    assert all(job.endswith("-v3") for job in phase7c.V3_JOB_IDS.values())
    assert len(set(phase7c.V3_JOB_IDS.values())) == 7


def test_the_job_ids_are_not_derivable_from_the_arm_names(repo_root):
    """**Why they are written out rather than computed.**

    Six follow ``p7c-arm<N>-<descriptive-name>-v3``. Arm 0 does not -- it is
    bare ``p7c-arm0-v3``. A derivation rule would need a special case that a
    later arm could silently break, and being wrong here does not fail loudly:
    it pairs against a void run and returns plausible intervals.
    """
    def derived(arm):
        name = phase7c.result_key(arm["name"]).split("_", 1)[1]
        return f"p7c-arm{arm['index']}-{name.replace('_', '-')}-v3"

    mismatched = [
        arm["index"] for arm in phase7c.arms()
        if derived(arm) != phase7c.V3_JOB_IDS[arm["index"]]
    ]
    assert mismatched == [0], (
        "the derivation rule now matches every arm or fails on a different "
        "one; either way the written-out mapping is what ships, and this test "
        "records why it is written out"
    )


def _paired_config(repo_root) -> dict:
    import yaml

    return yaml.safe_load(
        (repo_root / "configs" / "p7c_paired_selected30.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_the_paired_config_names_one_run_per_arm_with_no_wildcard_left(repo_root):
    """**[MEASURED 2026-08-03] The first version globbed the job id too, and
    all 35 were refused as ambiguous** -- three rounds share these config stems
    and two are VOID. Pinning the job id resolved each to a single run.

    Now that they are resolved, a surviving wildcard would be an input guard 3
    cannot hash-verify, so the assertion is the opposite of what it was.
    """
    inputs = _paired_config(repo_root)["inputs"]
    assert len(inputs) == 7 * 5

    for entry in inputs:
        assert "*" not in entry["path"], entry["path"]
        assert entry["rollup_sha256"] != "0" * 64, entry["name"]
        assert len(entry["rollup_sha256"]) == 64
        stem, sha, job = entry["path"].split("/")[-2].split("__")
        assert len(sha) == 8 and all(c in "0123456789abcdef" for c in sha), sha
        assert job in set(phase7c.V3_JOB_IDS.values()), job
        assert not stem.startswith("p7c_m3"), (
            "the paired config is the selected30 round; an m3 stem here would "
            "pair the sensitivity round's fits under the primary round's label"
        )


def test_all_seven_arms_were_run_at_the_same_commit(repo_root):
    """A mixed SHA would mean the arms ran under different code.

    The one-factor property every comparison rests on is that the arms differ
    ONLY in their augmentation policy. Two arms at different commits differ in
    whatever else changed between them, and nothing downstream would show it.
    """
    shas = {
        entry["path"].split("/")[-2].split("__")[1]
        for entry in _paired_config(repo_root)["inputs"]
    }
    assert len(shas) == 1, f"arms were run at {len(shas)} different commits: {shas}"


def test_no_two_vectors_share_a_hash(repo_root):
    """Thirty-five distinct fits, so thirty-five distinct files.

    A repeat would mean two declarations point at one vector -- the shape the
    v1 round already produced once, when every arm came back byte-identical
    because the augmenter never ran.
    """
    rollups = [entry["rollup_sha256"] for entry in _paired_config(repo_root)["inputs"]]
    assert len(set(rollups)) == len(rollups) == 35


def test_each_arms_five_vectors_come_from_that_arms_run(repo_root):
    """A crossed path would pair an arm against itself or against a sibling,
    and every downstream number would still look ordinary."""
    import yaml

    payload = yaml.safe_load(
        (repo_root / "configs" / "p7c_paired_selected30.yaml").read_text(
            encoding="utf-8"
        )
    )
    by_arm: dict[str, set] = {}
    for entry in payload["inputs"]:
        arm = entry["name"][len(phase7c.OOF_INPUT_PREFIX):].rpartition(
            phase7c.OOF_SEED_SEPARATOR
        )[0]
        run_dir = entry["path"].split("/")[-2]
        by_arm.setdefault(arm, set()).add(run_dir)

    assert len(by_arm) == 7
    for arm, run_dirs in by_arm.items():
        assert len(run_dirs) == 1, f"{arm} reads from {run_dirs}"
        stem, _, job = run_dirs.pop().split("__")
        assert stem == f"p7c_{arm}", f"{arm} declared against {stem}"
        index = int(arm.split("_", 1)[0])
        assert job == phase7c.V3_JOB_IDS[index]

    # Every seed present exactly once per arm.
    seeds = [
        int(entry["name"].rpartition(phase7c.OOF_SEED_SEPARATOR)[2])
        for entry in payload["inputs"]
    ]
    assert sorted(seeds) == sorted(phase7c.SEEDS * 7)


# --------------------------------------------------------------------------
# condition 1 is a universal, and that is what closed the phase
# --------------------------------------------------------------------------


def _synthetic_pair(n=60, seed=99):
    """Truth, a decent arm, and two kinds of second arm.

    ``close`` tracks the first arm almost exactly, so the paired delta is ~0
    and the interval covers zero. ``far`` is much worse, so its interval
    excludes zero. Both are asserted below rather than assumed.
    """
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=n)
    good = truth + rng.normal(scale=0.5, size=n)
    return {
        "truth": truth,
        "good": good,
        "close": good + rng.normal(scale=0.01, size=n),
        "far": truth + rng.normal(scale=3.0, size=n),
    }


def _compare(vectors, kinds):
    """One ``paired_comparison`` over ``len(kinds)`` synthetic seeds."""
    from cleft import phase7b

    seeds = list(range(1, len(kinds) + 1))
    return phase7b.paired_comparison(
        truth=vectors["truth"],
        winner_by_seed={s: vectors[k] for s, k in zip(seeds, kinds)},
        baseline_by_seed={s: vectors["good"] for s in seeds},
        # Tiny, so condition 2 always passes and the test isolates condition 1.
        winner_sd=1e-4,
        n_boot=N_BOOT,
    )


def test_the_synthetic_fixture_has_the_interval_pattern_it_claims():
    """Without this the two tests below could pass vacuously."""
    vectors = _synthetic_pair()
    assert _compare(vectors, ["far"])["per_seed"][0]["excludes_zero"]
    assert not _compare(vectors, ["close"])["per_seed"][0]["excludes_zero"]


def test_condition_1_is_a_universal_not_a_proportion():
    """**The question the ten-seed decision turned on.**

    ``COMPARISON_RULE`` says *every* seed's interval must exclude zero.
    Four of five is not four-fifths of a claim; it is not a claim.
    """
    vectors = _synthetic_pair()
    result = _compare(vectors, ["far", "far", "far", "far", "close"])
    assert result["n_excluding_zero"] == 4
    assert result["n_seeds"] == 5
    assert result["exceeds_threshold"]
    assert not result["claimable"]


def test_adding_a_seed_can_only_remove_a_claim():
    """Anti-monotone in the seed set: a claim survives adding a seed only if
    that seed also excludes zero."""
    vectors = _synthetic_pair()
    five = _compare(vectors, ["far"] * 5)
    assert five["claimable"], "the all-excluding set should claim"
    six = _compare(vectors, ["far"] * 5 + ["close"])
    assert not six["claimable"], "adding one non-excluding seed must withdraw it"


def test_a_failing_set_cannot_be_rescued_by_adding_seeds():
    """**Why the six ten-seed configs were not run.**

    Phase 7C has 0 of 5 intervals excluding zero in every comparison. Since a
    seed's fit is count-independent and the bootstrap is seeded from the seed,
    those five reappear unchanged inside a ten-seed run -- so condition 1
    fails by construction, whatever the new seeds do.
    """
    vectors = _synthetic_pair()
    failing = ["close"] * 5
    assert not _compare(vectors, failing)["claimable"]
    for added in range(1, 6):
        result = _compare(vectors, failing + ["far"] * added)
        assert not result["claimable"], (
            f"{added} excluding seed(s) rescued a set that fails at five"
        )


def test_mixed_directions_do_not_claim_even_when_every_interval_excludes():
    """The rule is 'all in the same direction', not merely 'all exclude'."""
    vectors = _synthetic_pair()
    # A seed whose winner is BETTER than the baseline: swap the roles.
    from cleft import phase7b

    result = phase7b.paired_comparison(
        truth=vectors["truth"],
        winner_by_seed={1: vectors["far"], 2: vectors["good"]},
        baseline_by_seed={1: vectors["good"], 2: vectors["far"]},
        winner_sd=1e-4,
        n_boot=N_BOOT,
    )
    assert result["n_excluding_zero"] == 2
    assert not result["same_direction"]
    assert not result["claimable"]


# --------------------------------------------------------------------------
# the phase's recorded outcome
# --------------------------------------------------------------------------


def test_the_withdrawn_verdicts_are_recorded_as_withdrawn():
    """The old verdict is kept, not overwritten -- the SEQUENCE is the
    finding, and it is only legible if both readings are on the page."""
    results = phase7c.STAGE_7C_RESULTS
    assert "SUPERSEDED" in results["verdict"]
    assert "claimably hurts" in results["verdict_before_condition_1"]
    assert "WITHDRAWN" in results["per_family"]["geometric"]["verdict"]


def test_the_outcome_record_says_the_phase_supports_nothing():
    record = phase7c.PAIRED_BCA_WITHDREW_EVERY_VERDICT
    assert record["intervals_excluding_zero"] == "0 of 45"
    assert "cannot resolve whether it helps or hurts" in record["reportable"]
    # The prediction that turned out backwards is named, not quietly dropped.
    assert "WRONG" in (
        phase7c.OUTSTANDING_BEFORE_THE_NULLS_ARE_REPORTABLE["paired_bca"]
        ["effect_on_negatives"]
    )


def test_the_ten_seed_round_is_recorded_as_not_run(repo_root):
    assert "NOT RUN" in phase7c.TEN_SEED_ROUND_NOT_RUN["status"]
    assert "SUPERSEDED" in phase7c.TEN_SEED_ROUND["status"]
    # And the configs themselves say so, since that is what a reader opens.
    stems = [
        phase7c.config_name(arm, round_["prefix"])
        for round_ in phase7c.CONFIG_ROUNDS if "s10" in round_["prefix"]
        for arm in phase7c.arms_at(round_["n_seeds"], only=round_["only"])
    ]
    assert len(stems) == 6
    for stem in stems:
        text = (repo_root / "configs" / f"{stem}.yaml").read_text(encoding="utf-8")
        assert "DO NOT RUN" in text, stem


def test_no_five_seed_config_carries_the_do_not_run_stamp(repo_root):
    """The stamp must land on the six and nowhere else -- a config that says
    DO NOT RUN when it should be run is as costly as the reverse."""
    for stem in phase7c.config_stems():
        text = (repo_root / "configs" / f"{stem}.yaml").read_text(encoding="utf-8")
        assert ("DO NOT RUN" in text) == ("s10" in stem), stem


# --------------------------------------------------------------------------
# the task, end to end
# --------------------------------------------------------------------------


def test_the_paired_task_runs_end_to_end_and_writes_its_summary(
    tmp_path, clean_repo, patched, monkeypatch
):
    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")
    config = builders.write_config(
        tmp_path / "paired.yaml",
        phase="p7c",
        inputs=_inputs_for(patched),
        task={
            "kind": "phase7c_paired",
            "round": "selected30",
            "seeds": SEEDS,
            "n_boot": N_BOOT,
        },
    )
    from cleft.run import main

    run_dir = Path(main(["--config", str(config), "--out", str(tmp_path / "runs")]))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    assert metrics["round"] == "selected30"
    assert metrics["n_patients"] == N_PATIENTS
    assert metrics["matrix"]["n_comparisons"] == 9
    assert len(metrics["verification"]["arms"]) == 7
    # Every arm's band is its own, over its own seeds -- PLAN §4.12.1.
    assert all(arm["n_seeds"] == 5 for arm in metrics["arms"].values())
    # Five intervals per comparison, never one pooled vector.
    for group in ("against_identity", "comparisons"):
        for entry in metrics["matrix"][group].values():
            assert len(entry["paired"]["per_seed"]) == 5
    # The criterion this run closes is recorded with the numbers.
    assert "condition 2 alone" in metrics["criterion"]["gap"]


def test_the_task_refuses_a_round_the_vectors_do_not_reproduce(
    tmp_path, clean_repo, built, monkeypatch
):
    """The end-to-end path must fail on wrong vectors, not just the helper."""
    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")
    config = builders.write_config(
        tmp_path / "paired.yaml",
        phase="p7c",
        inputs=_inputs_for(built),
        task={
            "kind": "phase7c_paired",
            "round": "selected30",
            "seeds": SEEDS,
            "n_boot": N_BOOT,
        },
    )
    from cleft.run import main

    with pytest.raises(phase7c.Phase7CError, match="do not reproduce round"):
        main(["--config", str(config), "--out", str(tmp_path / "runs")])


# --------------------------------------------------------------------------
# the generalised ladder task
# --------------------------------------------------------------------------


def test_the_paired_claims_task_runs_end_to_end(tmp_path, clean_repo, monkeypatch):
    """**The headline scope, through ``run.main`` with the real context.**

    ``test_phase7b_task`` exists because a task with tested helpers and an
    untested handler died on the cluster writing its summary. This one reaches
    the handler.
    """
    from cleft import ladder

    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")
    stems = ladder.paired_claim_stems("headline")
    built = _build_vectors(
        tmp_path / "runs",
        noise={stem: 0.4 + 0.2 * index for index, stem in enumerate(stems)},
    )
    config = builders.write_config(
        tmp_path / "paired.yaml",
        phase="p7",
        inputs=_inputs_for(built),
        task={"kind": "paired_claims", "scope": "headline", "n_boot": N_BOOT},
    )
    from cleft.run import main

    run_dir = Path(main(["--config", str(config), "--out", str(tmp_path / "runs")]))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    assert metrics["scope"] == "headline"
    assert metrics["n_pairs"] == 1
    assert metrics["n_patients"] == N_PATIENTS
    key = ladder.paired_claim_pairs("headline")[0]["key"]
    entry = metrics["pairs"][key]
    assert entry["delta_is"] == f"{entry['b']} minus {entry['a']}"
    assert len(entry["paired"]["per_seed"]) == 5
    assert entry["claimable"] == (
        entry["condition_1_paired_bca"] and entry["condition_2_exceeds_threshold"]
    )
    # Thinnest-first ordering, and the coverage decisions travel with the run.
    assert metrics["by_margin_thinnest_first"] == [key]
    assert "different truth vectors" in (
        metrics["coverage"]["excluded_label_formulation"]
    )


def test_the_paired_claims_task_refuses_a_missing_vector(
    tmp_path, clean_repo, monkeypatch
):
    from cleft import ladder

    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")
    stems = ladder.paired_claim_stems("headline")
    built = _build_vectors(tmp_path / "runs", noise={s: 0.5 for s in stems})
    inputs = [e for e in _inputs_for(built) if not e["name"].endswith("_seed_99")]
    config = builders.write_config(
        tmp_path / "paired.yaml",
        phase="p7",
        inputs=inputs,
        task={"kind": "paired_claims", "scope": "headline", "n_boot": N_BOOT},
    )
    from cleft.run import main

    with pytest.raises(ValueError, match="declared vectors missing"):
        main(["--config", str(config), "--out", str(tmp_path / "runs")])
