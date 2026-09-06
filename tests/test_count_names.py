"""R11: the count-name rule, and the sweep that keeps it non-vacuous."""

from pathlib import Path

from cleft import count_names, node_weights
from cleft.models import srgnn


def test_every_count_plus_constant_is_classified(repo_root):
    """**The sweep cannot judge a count -- it can refuse an unjudged one.**

    A new ``X = Y + n`` must be declared either a collection count (and
    then R11 applies: name the two quantities separately) or parameter
    arithmetic (a Linear's weights plus its bias, same shape, no hazard).
    """
    found = count_names.count_plus_constant_definitions(
        repo_root / "src" / "cleft"
    )
    classified = (
        count_names.COLLECTION_COUNTS | count_names.PARAMETER_ARITHMETIC
    )
    unclassified = set(found) - classified
    assert not unclassified, (
        f"unclassified count-plus-constant definitions: {sorted(unclassified)}. "
        "Declare each as a collection count (R11 applies) or as parameter "
        f"arithmetic. Found: {[found[k] for k in sorted(unclassified)]}"
    )
    # Non-vacuous: the shape really does occur, and the known instance is
    # among them.
    assert len(found) >= 3
    assert "models/srgnn.py:N_REGIONS" in found
    assert found["models/srgnn.py:N_REGIONS"] == "N_REGIONS = N_ROIS + 1"
    # Every classified entry must still exist, so the registry cannot rot
    # into a list of names nothing defines.
    assert classified <= set(found)


def test_the_rule_records_two_instances_not_three():
    """[CORRECTED] srs_boxes/model_regions is the FIX for the AG-Net
    episode, not a separate occurrence -- and two non-independent
    instances say more than three coincidences would."""
    instances = count_names.INSTANCES
    assert set(instances) == {"agnet", "srgnn"}
    for name, entry in instances.items():
        assert entry["total"] == entry["pre_append"] + 1, name
        assert "whole" in entry["appended"], name
    # The live one agrees with the model it describes.
    assert instances["srgnn"]["pre_append"] == srgnn.N_ROIS == 26
    assert instances["srgnn"]["total"] == srgnn.N_REGIONS == 27
    # The AG-Net one records its resolution, which is where the "third
    # instance" reading came from.
    assert "srs_boxes and model_regions" in instances["agnet"]["resolved_as"]

    # The rule earns a prediction rather than a tally.
    assert "AG-Net" in count_names.PREDICTS
    assert "arm C" in count_names.PREDICTS
    # And it is numbered where R10 already lives.
    assert count_names.NUMBERING["rule"] == "R11"
    assert count_names.NUMBERING["r10_is_also_code_side"] is True


def test_the_gate_applies_the_rule_it_was_written_from():
    """R11 is not advice here -- ``split_whole_image`` enforces it."""
    import numpy as np

    weights = np.tile(np.arange(srgnn.N_REGIONS, dtype=float), (4, 1))
    rois, whole = node_weights.split_whole_image(weights)
    assert rois.shape[1] == srgnn.N_ROIS
    assert np.all(whole == srgnn.N_REGIONS - 1)
    # The docstring of the module that reads the count names the trap.
    text = Path(node_weights.__file__).read_text(encoding="utf-8")
    assert "26 regions, not 27" in text
